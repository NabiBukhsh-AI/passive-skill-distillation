"""TASK-011 acceptance tests.

Criteria:
  * Duplicate delivery is a no-op.
  * Oversized payloads are rejected with a typed error.
  * Fuzzed malformed payloads never crash the endpoint.

Idempotency gets the most attention because a duplicated trajectory is not a harmless
extra row. Every corpus statistic is a rate over trajectories, so a duplicate inflates a
failure rate by exactly the duplication rate, and the resulting number still looks
entirely plausible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from psd.ingest.api import IngestBatch, MetadataStore, create_app, ingest_batch
from psd.ingest.storage import FilesystemObjectStore, body_key, content_sha256

pytestmark = pytest.mark.integration

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "trajectories"
    / "spec_section_10_3_example.json"
)


def body(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload.update(overrides)
    return payload


def batch(*bodies: dict[str, Any]) -> IngestBatch:
    return IngestBatch(
        tenant_id="t_test",
        records=[{"source_format": "harness_run", "body": b} for b in bodies],
    )


@pytest.fixture
def store(tmp_path: Path) -> FilesystemObjectStore:
    return FilesystemObjectStore(tmp_path / "objects")


@pytest.fixture
def metadata(connection: Any) -> MetadataStore:
    return MetadataStore(connection)


# ---------------------------------------------------------------------------
# Acceptance: duplicate delivery is a no-op
# ---------------------------------------------------------------------------


def test_a_clean_record_is_accepted(store: FilesystemObjectStore, metadata: MetadataStore) -> None:
    response = ingest_batch(batch(body()), store, metadata)
    assert response.accepted == 1
    assert response.duplicates == 0
    assert response.quarantined == 0


def test_redelivering_the_same_record_is_a_no_op(
    store: FilesystemObjectStore, metadata: MetadataStore, connection: Any
) -> None:
    """TASK-011 acceptance. At-least-once delivery is the norm for log shipping."""
    ingest_batch(batch(body()), store, metadata)
    second = ingest_batch(batch(body()), store, metadata)

    assert second.accepted == 0
    assert second.duplicates == 1
    rows = connection.execute("SELECT count(*) FROM trajectories").fetchone()[0]
    assert rows == 1, "redelivery produced a second metadata row"


def test_redelivery_within_one_batch_is_also_deduplicated(
    store: FilesystemObjectStore, metadata: MetadataStore, connection: Any
) -> None:
    """A client that retries inside a single batch must not double-count either."""
    ingest_batch(batch(body()), store, metadata)
    response = ingest_batch(batch(body(), body()), store, metadata)
    assert response.accepted == 0
    assert response.duplicates == 2
    assert connection.execute("SELECT count(*) FROM trajectories").fetchone()[0] == 1


def test_a_genuinely_different_record_is_not_treated_as_a_duplicate(
    store: FilesystemObjectStore, metadata: MetadataStore, connection: Any
) -> None:
    """Negative control. Deduplication that swallows distinct records is worse."""
    ingest_batch(batch(body()), store, metadata)
    other = body(trajectory_id="trj_other", task_id="retail_train_018")
    response = ingest_batch(batch(other), store, metadata)
    assert response.accepted == 1
    assert connection.execute("SELECT count(*) FROM trajectories").fetchone()[0] == 2


def test_the_body_lands_in_object_storage_at_its_content_address(
    store: FilesystemObjectStore, metadata: MetadataStore
) -> None:
    payload = body()
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    response = ingest_batch(batch(payload), store, metadata)

    digest = content_sha256(raw)
    assert response.results[0].content_sha256 == digest
    stored = store.get(body_key("t_test", "tau2_retail", digest))
    assert stored == raw


def test_object_storage_is_write_once(store: FilesystemObjectStore) -> None:
    store.put("a/b/c.json", b"first")
    store.put("a/b/c.json", b"first")  # identical content is a no-op
    with pytest.raises(ValueError, match="write-once"):
        store.put("a/b/c.json", b"different")


def test_a_key_cannot_escape_the_object_store_root(store: FilesystemObjectStore) -> None:
    """Keys derive from tenant and domain ids, which are external input."""
    with pytest.raises(ValueError, match="escapes"):
        store.put("../../etc/passwd", b"nope")


# ---------------------------------------------------------------------------
# Quarantine, not repair
# ---------------------------------------------------------------------------


def test_a_record_with_no_reward_is_quarantined_and_never_reaches_the_table(
    store: FilesystemObjectStore, metadata: MetadataStore, connection: Any
) -> None:
    """Spec Section 30.1 rule 14, end to end through the ingest path."""
    broken = body()
    del broken["outcome"]["reward"]

    response = ingest_batch(batch(broken), store, metadata)
    assert response.quarantined == 1
    assert response.accepted == 0
    assert response.results[0].quarantine_rule == "reward_present"
    assert connection.execute("SELECT count(*) FROM trajectories").fetchone()[0] == 0


def test_a_quarantined_record_is_kept_with_its_failing_rule(
    store: FilesystemObjectStore, metadata: MetadataStore, tmp_path: Path
) -> None:
    """The record is preserved for inspection, in a separate prefix.

    Separate prefix rather than a flag, so "never enters a corpus" is a property of where
    the bytes live rather than of a query someone might forget to filter.
    """
    broken = body()
    del broken["outcome"]["reward"]
    ingest_batch(batch(broken), store, metadata)

    quarantined = list((tmp_path / "objects" / "_quarantine").rglob("*.json"))
    assert len(quarantined) == 1
    saved = json.loads(quarantined[0].read_text(encoding="utf-8"))
    assert saved["quarantine"]["rule"] == "reward_present"
    assert "body" in saved


def test_a_mixed_batch_reports_each_outcome(
    store: FilesystemObjectStore, metadata: MetadataStore
) -> None:
    broken = body(trajectory_id="trj_broken")
    del broken["outcome"]["reward"]
    response = ingest_batch(batch(body(), broken), store, metadata)
    assert (response.accepted, response.quarantined) == (1, 1)


# ---------------------------------------------------------------------------
# Acceptance: oversized payloads, and fuzz
# ---------------------------------------------------------------------------


def client(store: FilesystemObjectStore, metadata: MetadataStore, **kwargs: Any) -> TestClient:
    return TestClient(create_app(store, metadata, **kwargs))


def test_oversized_batch_is_rejected_with_a_typed_error(
    store: FilesystemObjectStore, metadata: MetadataStore
) -> None:
    response = client(store, metadata, max_batch_bytes=16).post(
        "/v1/trajectories:batch",
        json={"tenant_id": "t_test", "records": [{"source_format": "harness_run", "body": body()}]},
    )
    assert response.status_code == 413
    assert response.headers["content-type"].startswith("application/problem+json")
    problem = response.json()
    assert problem["type"].endswith("payload-too-large")
    assert problem["status"] == 413
    assert "byte cap" in problem["detail"]


def test_too_many_records_is_rejected_with_a_typed_error(
    store: FilesystemObjectStore, metadata: MetadataStore
) -> None:
    response = client(store, metadata, max_batch_records=1).post(
        "/v1/trajectories:batch",
        json={
            "tenant_id": "t_test",
            "records": [
                {"source_format": "harness_run", "body": body()},
                {"source_format": "harness_run", "body": body()},
            ],
        },
    )
    assert response.status_code == 413
    assert response.json()["type"].endswith("too-many-records")


MALFORMED = [
    b"",
    b"not json at all",
    b"{",
    b"[]",
    b"null",
    b"123",
    b'"a string"',
    b'{"records": "not a list"}',
    b'{"records": [{"source_format": 1, "body": {}}]}',
    b'{"records": [{"body": {}}]}',
    b'{"records": [{"source_format": "harness_run"}]}',
    b'{"unexpected_field": true}',
    b"\xff\xfe\x00\x01",
]


@pytest.mark.parametrize("payload", MALFORMED, ids=range(len(MALFORMED)))
def test_malformed_payloads_never_crash_the_endpoint(
    store: FilesystemObjectStore, metadata: MetadataStore, payload: bytes
) -> None:
    """Fuzz. Every rejection is a typed problem detail, never a 500."""
    response = client(store, metadata).post(
        "/v1/trajectories:batch",
        content=payload,
        headers={"content-type": "application/json"},
    )
    assert response.status_code in {400, 422}, (
        f"expected a typed client error, got {response.status_code}: {response.text[:200]}"
    )
    assert response.headers["content-type"].startswith("application/problem+json")
    assert set(response.json()) >= {"type", "title", "status", "detail"}


def test_an_unknown_source_format_quarantines_rather_than_erroring(
    store: FilesystemObjectStore, metadata: MetadataStore
) -> None:
    response = client(store, metadata).post(
        "/v1/trajectories:batch",
        json={
            "tenant_id": "t_test",
            "records": [{"source_format": "some_vendor_thing", "body": body()}],
        },
    )
    assert response.status_code == 200
    assert response.json()["quarantined"] == 1


def test_health_endpoint(store: FilesystemObjectStore, metadata: MetadataStore) -> None:
    assert client(store, metadata).get("/v1/health").json() == {"status": "ok"}


def test_the_happy_path_works_over_http(
    store: FilesystemObjectStore, metadata: MetadataStore, connection: Any
) -> None:
    response = client(store, metadata).post(
        "/v1/trajectories:batch",
        json={
            "tenant_id": "t_test",
            "records": [{"source_format": "harness_run", "body": body()}],
        },
    )
    assert response.status_code == 200
    assert response.json()["accepted"] == 1
    assert connection.execute("SELECT count(*) FROM trajectories").fetchone()[0] == 1


def test_the_authenticated_tenant_overrides_the_body(
    store: FilesystemObjectStore, metadata: MetadataStore, connection: Any
) -> None:
    """PR-001 hard isolation: a client cannot write into another tenant by asking.

    The fixture body declares tenant `t_default`; the batch is authenticated as `t_test`.
    The row must belong to `t_test`.
    """
    ingest_batch(batch(body(tenant_id="t_default")), store, metadata)
    stored = connection.execute("SELECT tenant_id FROM trajectories").fetchone()[0]
    assert stored == "t_test"


def test_the_stored_row_carries_the_reward_and_lineage(
    store: FilesystemObjectStore, metadata: MetadataStore, connection: Any
) -> None:
    ingest_batch(batch(body()), store, metadata)
    row = connection.execute(
        "SELECT reward, success, actor_mode, split, redaction_version, body_uri FROM trajectories"
    ).fetchone()
    assert row[0] == 0.0
    assert row[1] is False
    assert row[2] == "no_think"
    assert row[3] == "train"
    assert row[4] == "redaction/1.1"
    assert row[5].startswith("file://")
