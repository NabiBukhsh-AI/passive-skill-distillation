"""Normalizer framework (TASK-010, component C-02).

Turns heterogeneous agent logs into the canonical `Trajectory` of spec Section 10.3, or
refuses them with a machine-readable reason (FR-002).

The design rule that matters here is spec Section 30.1 rule 14: **quarantine over
defaulting**. A record that violates a blocking rule is not a record to be repaired. Every
silent repair in an ingest path corrupts a measurement somewhere downstream, and it does
so invisibly, because the repaired record still looks plausible.

So this module never fills in a missing value. It classifies the failure against a stable
rule id and routes the record to quarantine.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import Field, ValidationError

from psd.core.models import StrictModel, Trajectory

#: Spec Section 10.3: trajectory size must be under a configured cap. Sandbox memory
#: safety (ALG-007 Step 1) is the reason the cap exists at all.
DEFAULT_MAX_TRAJECTORY_BYTES = 8 * 1024 * 1024


class QuarantineRule:
    """Stable ids for the blocking rules of spec Section 10.3.

    These strings are persisted on quarantined records and aggregated into the
    `psd_quarantine_total{reason}` metric (spec Section 25.2), so they are an interface.
    Renaming one breaks a dashboard and a trailing-week comparison.
    """

    REWARD_PRESENT = "reward_present"
    STEP_CONTIGUITY = "step_contiguity"
    ACTOR_MODE_VALID = "actor_mode_valid"
    SYSTEM_PROMPT_HASH_PRESENT = "system_prompt_hash_present"
    TOKEN_KEYS_PRESENT = "token_keys_present"
    SPLIT_MATCHES_ARTIFACT = "split_matches_artifact"
    SIZE_CAP = "size_cap"
    UNKNOWN_SOURCE_FORMAT = "unknown_source_format"
    SCHEMA_INVALID = "schema_invalid"
    MALFORMED_PAYLOAD = "malformed_payload"


class Quarantine(StrictModel):
    """Why a record was refused. Machine-readable by requirement (FR-002)."""

    rule: str
    detail: str
    field_path: str | None = None


class NormalizationOutcome(StrictModel):
    """Either a canonical trajectory, or a quarantine record. Never both, never neither."""

    trajectory: Trajectory | None = None
    quarantine: Quarantine | None = None
    #: Every field the mapper dropped, so a lossy source format is visible rather than
    #: inferred later from a gap in the data (C-02).
    warnings: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.trajectory is not None

    def unwrap(self) -> Trajectory:
        if self.trajectory is None:
            assert self.quarantine is not None
            raise ValueError(
                f"trajectory was quarantined: {self.quarantine.rule}: {self.quarantine.detail}"
            )
        return self.trajectory


#: A mapper takes a raw body and returns (canonical-shaped dict, warnings).
Mapper = Callable[[Mapping[str, Any]], tuple[dict[str, Any], list[str]]]

_MAPPERS: dict[str, Mapper] = {}


def register_mapper(source_format: str) -> Callable[[Mapper], Mapper]:
    def decorate(fn: Mapper) -> Mapper:
        if source_format in _MAPPERS:
            raise RuntimeError(f"mapper for {source_format!r} is already registered")
        _MAPPERS[source_format] = fn
        return fn

    return decorate


def registered_formats() -> list[str]:
    return sorted(_MAPPERS)


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


def _classify(error: Mapping[str, Any]) -> tuple[str, str | None]:
    """Map one Pydantic error to a stable rule id and a field path.

    Ordering matters: the reward rule is checked first because it is the one whose
    silent repair does the most damage (ALG-006 win/loss contrast).
    """
    loc = tuple(str(part) for part in error.get("loc", ()))
    path = ".".join(loc) if loc else None
    message = str(error.get("msg", ""))

    if "outcome" in loc and "reward" in loc:
        return QuarantineRule.REWARD_PRESENT, path
    if "contiguous" in message:
        return QuarantineRule.STEP_CONTIGUITY, "steps"
    if loc[:2] == ("actor", "mode"):
        return QuarantineRule.ACTOR_MODE_VALID, path
    if "system_prompt_sha256" in loc:
        return QuarantineRule.SYSTEM_PROMPT_HASH_PRESENT, path
    if "tokens" in loc:
        return QuarantineRule.TOKEN_KEYS_PRESENT, path
    return QuarantineRule.SCHEMA_INVALID, path


def classify_validation_error(exc: ValidationError) -> Quarantine:
    """Pick the most consequential error to report.

    A malformed record usually trips several rules at once. Reporting the first one
    Pydantic happens to emit makes the quarantine metric depend on field ordering, so
    the blocking rules are ranked by how much damage a silent repair would do.
    """
    severity = {
        QuarantineRule.REWARD_PRESENT: 0,
        QuarantineRule.STEP_CONTIGUITY: 1,
        QuarantineRule.TOKEN_KEYS_PRESENT: 2,
        QuarantineRule.ACTOR_MODE_VALID: 3,
        QuarantineRule.SYSTEM_PROMPT_HASH_PRESENT: 4,
        QuarantineRule.SCHEMA_INVALID: 5,
    }
    classified = [_classify(error) for error in exc.errors()]
    rule, path = min(classified, key=lambda item: severity.get(item[0], 99))
    detail = "; ".join(
        f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg')}" for e in exc.errors()[:5]
    )
    return Quarantine(rule=rule, detail=detail, field_path=path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def normalize(
    body: Mapping[str, Any],
    source_format: str,
    *,
    max_bytes: int = DEFAULT_MAX_TRAJECTORY_BYTES,
    split_task_ids: Mapping[str, str] | None = None,
) -> NormalizationOutcome:
    """Normalize one raw trajectory.

    `split_task_ids` maps task id to split name. When supplied, a trajectory whose
    declared split contradicts the split artifact is quarantined (spec Section 10.3).
    A task id absent from the mapping is left alone: it is `unassigned` traffic, which
    ASM-004 admits and ALG-001 Step 2 filters out of every corpus anyway.
    """
    mapper = _MAPPERS.get(source_format)
    if mapper is None:
        return NormalizationOutcome(
            quarantine=Quarantine(
                rule=QuarantineRule.UNKNOWN_SOURCE_FORMAT,
                detail=(
                    f"no mapper for source_format={source_format!r}; "
                    f"registered: {registered_formats()}"
                ),
            )
        )

    try:
        encoded = json.dumps(body).encode("utf-8")
    except (TypeError, ValueError) as exc:
        return NormalizationOutcome(
            quarantine=Quarantine(
                rule=QuarantineRule.MALFORMED_PAYLOAD, detail=f"body is not JSON: {exc}"
            )
        )
    if len(encoded) > max_bytes:
        return NormalizationOutcome(
            quarantine=Quarantine(
                rule=QuarantineRule.SIZE_CAP,
                detail=f"{len(encoded)} bytes exceeds the {max_bytes} byte cap",
            )
        )

    try:
        canonical, warnings = mapper(body)
    except Exception as exc:
        return NormalizationOutcome(
            quarantine=Quarantine(
                rule=QuarantineRule.MALFORMED_PAYLOAD,
                detail=f"{source_format} mapper failed: {type(exc).__name__}: {exc}",
            )
        )

    # Validate through JSON, not through the dict. The models are strict, and strict
    # Python-mode validation refuses an ISO-8601 string for a datetime field, which JSON
    # mode accepts. JSON is also the actual ingest path (JSONL on object storage), so
    # validating any other way would test a path that never runs in production.
    try:
        trajectory = Trajectory.model_validate_json(json.dumps(canonical))
    except ValidationError as exc:
        return NormalizationOutcome(quarantine=classify_validation_error(exc))

    if split_task_ids is not None:
        expected = split_task_ids.get(trajectory.task_id)
        if expected is not None and expected != trajectory.split:
            return NormalizationOutcome(
                quarantine=Quarantine(
                    rule=QuarantineRule.SPLIT_MATCHES_ARTIFACT,
                    detail=(
                        f"task {trajectory.task_id!r} is {expected!r} in the split "
                        f"artifact but the record declares {trajectory.split!r}"
                    ),
                    field_path="split",
                )
            )

    if warnings:
        trajectory = trajectory.model_copy(
            update={"normalization_warnings": [*trajectory.normalization_warnings, *warnings]}
        )
    return NormalizationOutcome(trajectory=trajectory, warnings=warnings)
