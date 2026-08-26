"""Trajectory ingestion (TASK-011, component C-01).

`POST /v1/trajectories:batch`. Bodies to object storage, metadata to Postgres, writes
idempotent on content hash.

Two behaviours are load-bearing rather than incidental:

  * **Duplicate delivery is a no-op.** At-least-once delivery is the norm for log
    shipping, and a duplicated trajectory is not a harmless extra row: it double-counts
    in every corpus statistic, inflating a failure rate by exactly as much as the
    duplication rate. Idempotency is keyed on content hash, so redelivery of identical
    bytes cannot produce a second row no matter which client sends it.

  * **A record that fails a blocking rule is quarantined, not repaired.** It goes to a
    quarantine prefix with the failing rule recorded, and never reaches the metadata
    table. Spec Section 30.1 rule 14.

Errors are RFC 9457 problem details, per spec Section 16.1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import Field

from psd.core.models import StrictModel, Trajectory
from psd.ingest.normalizers import normalize
from psd.ingest.storage import ObjectStore, body_key, content_sha256, quarantine_key

#: Spec Section 10.3. Rejected with a typed error rather than truncated.
DEFAULT_MAX_BATCH_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_BATCH_RECORDS = 1000


class IngestRecord(StrictModel):
    source_format: str
    body: dict[str, Any]


class IngestBatch(StrictModel):
    tenant_id: str = "t_default"
    records: list[IngestRecord] = Field(default_factory=list)


class RecordOutcome(StrictModel):
    content_sha256: str
    status: str  # "accepted" | "duplicate" | "quarantined"
    trajectory_id: str | None = None
    body_uri: str | None = None
    quarantine_rule: str | None = None
    detail: str | None = None


class IngestResponse(StrictModel):
    accepted: int
    duplicates: int
    quarantined: int
    results: list[RecordOutcome]


class ProblemDetail(StrictModel):
    """RFC 9457 (spec Section 16.1)."""

    type: str
    title: str
    status: int
    detail: str
    instance: str | None = None


@dataclass
class MetadataStore:
    """Trajectory metadata rows. Idempotent on `(tenant, domain, content hash)`."""

    connection: Any

    def exists(self, tenant_id: str, domain: str, content_hash: str) -> bool:
        row = self.connection.execute(
            """
            SELECT 1 FROM trajectories
            WHERE tenant_id = %s AND domain_id = %s AND content_sha256 = %s
            LIMIT 1
            """,
            (tenant_id, domain, content_hash),
        ).fetchone()
        return row is not None

    def insert(self, trajectory: Trajectory, body_uri: str, content_hash: str) -> None:
        totals = trajectory.totals
        self.connection.execute(
            """
            INSERT INTO trajectories (
                trajectory_id, tenant_id, domain_id, task_id, split,
                actor_model, actor_mode, harness_version, seed,
                reward, success, steps_used, step_cap_hit,
                output_tokens, reasoning_tokens, input_tokens, cost_usd,
                error_types, stalled, body_uri, content_sha256, redaction_version
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s
            )
            """,
            (
                trajectory.trajectory_id,
                trajectory.tenant_id,
                trajectory.domain,
                trajectory.task_id,
                trajectory.split,
                trajectory.actor.model,
                trajectory.actor.mode,
                trajectory.harness.version,
                trajectory.seed,
                trajectory.outcome.reward,
                trajectory.outcome.success,
                trajectory.outcome.steps_used,
                trajectory.outcome.step_cap_hit,
                totals.output_tokens,
                totals.output_reasoning_tokens,
                totals.input_tokens,
                totals.cost_usd,
                list(trajectory.labels.error_types),
                bool(trajectory.labels.stall_runs),
                body_uri,
                content_hash,
                trajectory.redaction.policy_version,
            ),
        )


def ingest_batch(
    batch: IngestBatch,
    store: ObjectStore,
    metadata: MetadataStore | None = None,
    *,
    now: datetime | None = None,
) -> IngestResponse:
    """Ingest one batch. Pure enough to test without an HTTP server."""
    when = now or datetime.now(UTC)
    results: list[RecordOutcome] = []

    for record in batch.records:
        raw = json.dumps(record.body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        content_hash = content_sha256(raw)

        # Idempotency first: a duplicate must not be re-normalized, re-stored, or
        # re-counted. Checking after would still double-count in the response.
        if metadata is not None and metadata.exists(
            batch.tenant_id, str(record.body.get("domain", "")), content_hash
        ):
            results.append(RecordOutcome(content_sha256=content_hash, status="duplicate"))
            continue

        outcome = normalize(record.body, record.source_format)
        if not outcome.ok:
            assert outcome.quarantine is not None
            key = quarantine_key(
                batch.tenant_id, str(record.body.get("domain", "unknown")), content_hash, when
            )
            store.put(
                key,
                json.dumps(
                    {
                        "quarantine": outcome.quarantine.model_dump(mode="json"),
                        "body": record.body,
                    },
                    sort_keys=True,
                    indent=2,
                ).encode("utf-8"),
            )
            results.append(
                RecordOutcome(
                    content_sha256=content_hash,
                    status="quarantined",
                    quarantine_rule=outcome.quarantine.rule,
                    detail=outcome.quarantine.detail,
                )
            )
            continue

        # The AUTHENTICATED tenant wins over the one in the body. The body is client
        # supplied and untrusted, so honouring it would let a caller write into another
        # tenant's namespace, which breaks the hard isolation PR-001 requires. It also
        # keeps the row's tenant consistent with the key idempotency was checked against;
        # otherwise deduplication silently stops working.
        trajectory = outcome.unwrap()
        if trajectory.tenant_id != batch.tenant_id:
            trajectory = trajectory.model_copy(update={"tenant_id": batch.tenant_id})
        key = body_key(batch.tenant_id, trajectory.domain, content_hash, when)
        uri = store.put(key, raw)
        if metadata is not None:
            metadata.insert(trajectory, uri, content_hash)
        results.append(
            RecordOutcome(
                content_sha256=content_hash,
                status="accepted",
                trajectory_id=trajectory.trajectory_id,
                body_uri=uri,
            )
        )

    return IngestResponse(
        accepted=sum(1 for r in results if r.status == "accepted"),
        duplicates=sum(1 for r in results if r.status == "duplicate"),
        quarantined=sum(1 for r in results if r.status == "quarantined"),
        results=results,
    )


def create_app(
    store: ObjectStore,
    metadata: MetadataStore | None = None,
    *,
    max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES,
    max_batch_records: int = DEFAULT_MAX_BATCH_RECORDS,
) -> FastAPI:
    app = FastAPI(title="psd ingestion", version="1.0")

    def problem(status: int, kind: str, title: str, detail: str) -> JSONResponse:
        return JSONResponse(
            status_code=status,
            content=ProblemDetail(
                type=f"https://psd.invalid/problems/{kind}",
                title=title,
                status=status,
                detail=detail,
            ).model_dump(mode="json"),
            media_type="application/problem+json",
        )

    @app.post("/v1/trajectories:batch")
    async def ingest(request: Request) -> Any:
        raw = await request.body()
        if len(raw) > max_batch_bytes:
            return problem(
                413,
                "payload-too-large",
                "Batch exceeds the size cap",
                f"{len(raw)} bytes exceeds the {max_batch_bytes} byte cap",
            )
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return problem(400, "malformed-json", "Body is not valid JSON", str(exc))

        try:
            batch = IngestBatch.model_validate(payload)
        except Exception as exc:
            return problem(422, "invalid-batch", "Batch failed validation", str(exc))

        if len(batch.records) > max_batch_records:
            return problem(
                413,
                "too-many-records",
                "Batch exceeds the record cap",
                f"{len(batch.records)} records exceeds the {max_batch_records} record cap",
            )

        return ingest_batch(batch, store, metadata).model_dump(mode="json")

    @app.get("/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
