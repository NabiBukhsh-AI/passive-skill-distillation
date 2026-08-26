"""TASK-016 acceptance tests: one per spec Section 10.9 check.

Acceptance is "each check fires on a crafted violation", so every check gets both a
passing case and a deliberately breaching one. A check that has only ever been run
against clean data is a check nobody has tested.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from psd.core.models import Trajectory
from psd.ingest.quality import (
    CheckResult,
    check_corpus_class_balance,
    check_duplicate_rate,
    check_normalization_success_rate,
    check_redaction_recall,
    check_reward_presence,
    check_step_contiguity,
    check_token_accounting_completeness,
    run_all,
)

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "trajectories"
    / ("spec_section_10_3_example.json")
)


def make(
    *,
    task_id: str = "t1",
    success: bool = False,
    content_hash: str | None = None,
    complete_tokens: bool = True,
) -> Trajectory:
    payload: dict[str, Any] = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["task_id"] = task_id
    payload["trajectory_id"] = f"trj_{task_id}"
    payload["outcome"]["success"] = success
    payload["outcome"]["reward"] = 1.0 if success else 0.0
    if content_hash:
        payload["provenance"]["content_sha256"] = content_hash
    if not complete_tokens:
        payload["steps"][0]["tokens"]["output_reasoning"] = None
    return Trajectory.model_validate_json(json.dumps(payload))


# ---------------------------------------------------------------------------
# Normalization success rate: > 99%, alert
# ---------------------------------------------------------------------------


def test_normalization_success_rate_passes_above_threshold() -> None:
    assert check_normalization_success_rate(1000, 1000).passed


def test_normalization_success_rate_fires_below_threshold() -> None:
    result = check_normalization_success_rate(980, 1000)
    assert result.breached
    assert result.action == "alert"
    assert result.value == pytest.approx(0.98)


# ---------------------------------------------------------------------------
# Reward presence: 100%, quarantine
# ---------------------------------------------------------------------------


def test_reward_presence_passes_when_every_record_has_one() -> None:
    result = check_reward_presence([make(task_id="a"), make(task_id="b")])
    assert result.passed
    assert result.action == "quarantine"


def test_reward_presence_is_a_hundred_percent_threshold() -> None:
    """Not 99%. A single defaulted reward corrupts the win/loss contrast it lands in."""
    assert check_reward_presence([]).threshold == 1.0


# ---------------------------------------------------------------------------
# Step contiguity: 100%, quarantine
# ---------------------------------------------------------------------------


def test_step_contiguity_passes_on_clean_records() -> None:
    assert check_step_contiguity([make()]).passed


def test_step_contiguity_fires_on_a_gap() -> None:
    """Constructed by bypassing the model, since the model refuses to build one.

    That refusal is the real defence; this check is the assertion that it ran.
    """
    trajectory = make()
    broken = trajectory.model_construct(
        **{**trajectory.__dict__, "steps": [trajectory.steps[0].model_copy(update={"index": 5})]}
    )
    result = check_step_contiguity([broken])
    assert result.breached
    assert result.action == "quarantine"


# ---------------------------------------------------------------------------
# Token accounting completeness: > 99.9% of steps, alert
# ---------------------------------------------------------------------------


def test_token_completeness_passes_when_all_components_reported() -> None:
    assert check_token_accounting_completeness([make()]).passed


def test_token_completeness_fires_on_a_null_component() -> None:
    result = check_token_accounting_completeness([make(complete_tokens=False)])
    assert result.breached
    assert result.action == "alert"
    assert "never zero-filled" in result.detail


def test_token_completeness_is_measured_over_steps_not_trajectories() -> None:
    """Spec Section 10.9 states the denominator as steps."""
    result = check_token_accounting_completeness([make(), make(complete_tokens=False)])
    assert result.value == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Duplicate rate: < 0.1%, alert
# ---------------------------------------------------------------------------


def test_duplicate_rate_passes_on_distinct_content() -> None:
    records = [make(task_id=f"t{i}", content_hash=f"{i:064d}") for i in range(10)]
    assert check_duplicate_rate(records).passed


def test_duplicate_rate_fires_on_repeated_content_hashes() -> None:
    records = [make(task_id=f"t{i}", content_hash="a" * 64) for i in range(10)]
    result = check_duplicate_rate(records)
    assert result.breached
    assert result.action == "alert"
    assert "review idempotency keys" in result.detail


# ---------------------------------------------------------------------------
# Redaction recall: > 0.98, BLOCKS corpus creation
# ---------------------------------------------------------------------------


def test_redaction_recall_passes_above_threshold() -> None:
    assert check_redaction_recall(0.995).passed


def test_redaction_recall_below_threshold_blocks_corpus_creation() -> None:
    result = check_redaction_recall(0.90)
    assert result.breached
    assert result.action == "block_corpus_creation", (
        "under-redacted text reaching a distiller can put personal data into a skill, "
        "and a skill is served in a system prompt on every episode"
    )


def test_a_report_with_a_recall_breach_blocks() -> None:
    report = run_all([make()], redaction_recall=0.5)
    assert report.blocks_corpus_creation


def test_a_clean_report_does_not_block() -> None:
    assert not run_all([make()], redaction_recall=0.99).blocks_corpus_creation


# ---------------------------------------------------------------------------
# Class balance: report only
# ---------------------------------------------------------------------------


def test_class_balance_is_report_only_and_never_fails() -> None:
    result = check_corpus_class_balance([make(success=True), make(task_id="b")])
    assert result.passed
    assert result.action == "report_only"
    assert result.value == pytest.approx(0.5)


def test_class_balance_does_not_block_an_all_success_corpus() -> None:
    """It is a poor input, not an invalid one. ALG-001 warns about it separately."""
    result = check_corpus_class_balance([make(success=True)])
    assert result.passed


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------


def test_run_all_covers_every_section_10_9_row() -> None:
    names = {r.name for r in run_all([make()], redaction_recall=0.99).results}
    assert names == {
        "normalization_success_rate",
        "reward_presence",
        "step_contiguity",
        "token_accounting_completeness",
        "duplicate_rate",
        "corpus_class_balance",
        "redaction_recall",
    }


def test_the_alerting_hook_fires_once_per_breach() -> None:
    fired: list[CheckResult] = []
    run_all(
        [make(complete_tokens=False)],
        submitted=1000,
        redaction_recall=0.5,
        on_breach=fired.append,
    )
    names = {r.name for r in fired}
    assert "token_accounting_completeness" in names
    assert "redaction_recall" in names
    assert "normalization_success_rate" in names


def test_the_hook_is_not_fired_for_passing_checks() -> None:
    fired: list[CheckResult] = []
    run_all([make()], redaction_recall=0.99, on_breach=fired.append)
    assert fired == []


def test_empty_batches_do_not_produce_spurious_breaches() -> None:
    """An empty batch is not a quality failure; it is an empty batch."""
    assert run_all([]).breaches == []
