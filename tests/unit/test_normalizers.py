"""TASK-010 acceptance tests.

Criteria:
  * A record missing `outcome.reward` is quarantined with the failing rule recorded.
  * Round-trip normalization is idempotent.
  * A mutation test asserting reward is never defaulted to 0.

The playbook calls the first and third out specifically, so they get negative controls
rather than a single happy assertion: it is easy to write a test that passes because
nothing was validated at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from psd.ingest.normalizers import (
    DEFAULT_MAX_TRAJECTORY_BYTES,
    QuarantineRule,
    normalize,
    registered_formats,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
EXAMPLE = FIXTURES / "trajectories" / "spec_section_10_3_example.json"


def body() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    return payload


def normalized(payload: dict[str, Any] | None = None, **kwargs: Any) -> Any:
    return normalize(payload if payload is not None else body(), "harness_run", **kwargs)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_harness_run_format_is_registered() -> None:
    assert "harness_run" in registered_formats()


def test_a_valid_record_normalizes() -> None:
    outcome = normalized()
    assert outcome.ok
    assert outcome.quarantine is None
    assert outcome.unwrap().task_id == "retail_train_017"


def test_unknown_source_format_quarantines_rather_than_guessing() -> None:
    outcome = normalize(body(), "some_vendor_format_we_never_saw")
    assert not outcome.ok
    assert outcome.quarantine is not None
    assert outcome.quarantine.rule == QuarantineRule.UNKNOWN_SOURCE_FORMAT


# ---------------------------------------------------------------------------
# Acceptance: reward is never defaulted
# ---------------------------------------------------------------------------


def test_missing_reward_is_quarantined_with_the_rule_recorded() -> None:
    payload = body()
    del payload["outcome"]["reward"]
    outcome = normalized(payload)

    assert not outcome.ok
    assert outcome.trajectory is None, "a record with no reward must not produce a trajectory"
    assert outcome.quarantine is not None
    assert outcome.quarantine.rule == QuarantineRule.REWARD_PRESENT
    assert outcome.quarantine.field_path == "outcome.reward"
    assert "reward" in outcome.quarantine.detail


def test_null_reward_is_quarantined() -> None:
    payload = body()
    payload["outcome"]["reward"] = None
    outcome = normalized(payload)
    assert not outcome.ok
    assert outcome.quarantine is not None
    assert outcome.quarantine.rule == QuarantineRule.REWARD_PRESENT


@pytest.mark.parametrize("absent", [None, "", "null", "None", "NaN", [], {}])
def test_reward_is_never_defaulted_to_zero(absent: Any) -> None:
    """Mutation test (TASK-010, playbook Stage 2 item 1).

    Every one of these is a way a source can express "no reward here": a JSON null, an
    empty string from a CSV export, a stringified None from a careless serializer, an
    empty container. A lenient ingest path turns all of them into 0.0.

    A false zero is indistinguishable from a genuine failed episode, so it corrupts every
    win/loss contrast (ALG-006) while leaving the numbers looking entirely sane. None of
    these may ever produce a trajectory.

    Note that a real 0 and 0.0 are deliberately NOT in this list: zero is a legitimate
    reward meaning the episode failed, and the method is failure-derived. See
    `test_a_genuine_zero_reward_is_still_accepted`.
    """
    payload = body()
    payload["outcome"]["reward"] = absent
    outcome = normalized(payload)

    if outcome.ok:
        pytest.fail(
            f"reward={absent!r} was accepted and became "
            f"{outcome.unwrap().outcome.reward!r}; it must have been quarantined"
        )
    assert outcome.quarantine is not None
    assert outcome.quarantine.rule in {
        QuarantineRule.REWARD_PRESENT,
        QuarantineRule.SCHEMA_INVALID,
    }


def test_a_genuine_zero_reward_is_still_accepted() -> None:
    """Negative control.

    Zero is a legitimate reward: it means the episode failed. If the rule above rejected
    it, the corpus would silently lose every failure, and the method is failure-derived.
    """
    payload = body()
    payload["outcome"]["reward"] = 0.0
    outcome = normalized(payload)
    assert outcome.ok
    assert outcome.unwrap().outcome.reward == 0.0


# ---------------------------------------------------------------------------
# The remaining blocking rules of spec Section 10.3
# ---------------------------------------------------------------------------


def test_step_index_gap_is_quarantined() -> None:
    payload = body()
    payload["steps"][0]["index"] = 3
    outcome = normalized(payload)
    assert outcome.quarantine is not None
    assert outcome.quarantine.rule == QuarantineRule.STEP_CONTIGUITY


def test_unknown_actor_mode_is_quarantined() -> None:
    payload = body()
    payload["actor"]["mode"] = "kinda_thinking"
    outcome = normalized(payload)
    assert outcome.quarantine is not None
    assert outcome.quarantine.rule == QuarantineRule.ACTOR_MODE_VALID


def test_missing_system_prompt_hash_is_quarantined() -> None:
    payload = body()
    del payload["harness"]["system_prompt_sha256"]
    outcome = normalized(payload)
    assert outcome.quarantine is not None
    assert outcome.quarantine.rule == QuarantineRule.SYSTEM_PROMPT_HASH_PRESENT


def test_missing_token_key_is_quarantined() -> None:
    payload = body()
    del payload["steps"][0]["tokens"]["output_reasoning"]
    outcome = normalized(payload)
    assert outcome.quarantine is not None
    assert outcome.quarantine.rule == QuarantineRule.TOKEN_KEYS_PRESENT


def test_explicit_null_token_is_admitted_not_quarantined() -> None:
    """ASM-002. A null value is present-but-unreported; only an absent key is blocking."""
    payload = body()
    payload["steps"][0]["tokens"]["output_reasoning"] = None
    outcome = normalized(payload)
    assert outcome.ok
    assert outcome.unwrap().token_accounting_complete is False


def test_oversized_trajectory_is_quarantined() -> None:
    payload = body()
    payload["steps"][0]["observation"]["text"] = "x" * 4096
    outcome = normalized(payload, max_bytes=1024)
    assert outcome.quarantine is not None
    assert outcome.quarantine.rule == QuarantineRule.SIZE_CAP


def test_the_default_size_cap_is_the_documented_eight_megabytes() -> None:
    assert DEFAULT_MAX_TRAJECTORY_BYTES == 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# Split-artifact agreement
# ---------------------------------------------------------------------------


def test_split_contradicting_the_artifact_is_quarantined() -> None:
    """Spec Section 10.3: `split` MUST match the split artifact for `task_id`."""
    outcome = normalized(split_task_ids={"retail_train_017": "test"})
    assert outcome.quarantine is not None
    assert outcome.quarantine.rule == QuarantineRule.SPLIT_MATCHES_ARTIFACT
    assert "retail_train_017" in outcome.quarantine.detail


def test_split_agreeing_with_the_artifact_passes() -> None:
    assert normalized(split_task_ids={"retail_train_017": "train"}).ok


def test_task_absent_from_the_artifact_is_admitted() -> None:
    """ASM-004: live traffic arrives before any split exists for its tasks.

    ALG-001 Step 2 filters corpus membership to train ids, so such a record can never
    reach a corpus regardless.
    """
    assert normalized(split_task_ids={"some_other_task": "train"}).ok


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_normalization_is_idempotent() -> None:
    """TASK-010 acceptance: round-trip normalization is idempotent.

    Normalizing an already-canonical record must be a fixed point. If it is not, the
    content hash of a trajectory depends on how many times it passed through ingest,
    and deduplication (C-01) silently stops working.
    """
    once = normalized().unwrap()
    again = normalize(json.loads(once.model_dump_json()), "harness_run").unwrap()
    assert once == again
    assert once.model_dump_json() == again.model_dump_json()


def test_normalization_is_idempotent_across_three_passes() -> None:
    current = normalized().unwrap()
    for _ in range(3):
        current = normalize(json.loads(current.model_dump_json()), "harness_run").unwrap()
    assert current.model_dump_json() == normalized().unwrap().model_dump_json()


# ---------------------------------------------------------------------------
# Warnings: a lossy source is visible, not inferred later
# ---------------------------------------------------------------------------


def test_unknown_top_level_fields_are_dropped_and_recorded() -> None:
    payload = body()
    payload["vendor_specific_thing"] = {"anything": 1}
    outcome = normalized(payload)
    assert outcome.ok
    assert any("vendor_specific_thing" in w for w in outcome.warnings)
    assert any("vendor_specific_thing" in w for w in outcome.unwrap().normalization_warnings)


def test_a_clean_record_produces_no_warnings() -> None:
    assert normalized().warnings == []


def test_unwrap_on_a_quarantined_outcome_raises() -> None:
    payload = body()
    del payload["outcome"]["reward"]
    with pytest.raises(ValueError, match="quarantined"):
        normalized(payload).unwrap()
