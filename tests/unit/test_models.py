"""TASK-005 acceptance tests.

Criteria asserted here:
  1. JSON Schema is generated for each model.
  2. The spec Section 10.3 example validates.
  3. A payload with a null reward is rejected.

Plus the remaining blocking rules of spec Section 10.3, because "quarantine over
defaulting" is only real if the type boundary actually refuses the record.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from psd.core.models import (
    CorpusManifest,
    Episode,
    EvaluationRun,
    Skill,
    Split,
    Step,
    StepTokens,
    Trajectory,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
EXAMPLE = FIXTURES / "trajectories" / "spec_section_10_3_example.json"


def load_example() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    return payload


def validate(payload: dict[str, Any]) -> Trajectory:
    """Validate through JSON, which is the real ingest path."""
    return Trajectory.model_validate_json(json.dumps(payload))


# ---------------------------------------------------------------------------
# Acceptance criterion 1: JSON Schema generation
# ---------------------------------------------------------------------------

ALL_MODELS: list[type[BaseModel]] = [
    Trajectory,
    Split,
    CorpusManifest,
    Skill,
    Episode,
    EvaluationRun,
]


@pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.__name__)
def test_json_schema_is_generated(model: type[BaseModel]) -> None:
    schema = model.model_json_schema()
    assert schema["type"] == "object"
    assert schema["title"] == model.__name__
    assert "properties" in schema


@pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.__name__)
def test_extra_fields_are_forbidden(model: type[BaseModel]) -> None:
    """Strict mode with extra='forbid' (TASK-005)."""
    assert model.model_config["extra"] == "forbid"
    assert model.model_config["strict"] is True


# ---------------------------------------------------------------------------
# Acceptance criterion 2: the spec Section 10.3 example validates
# ---------------------------------------------------------------------------


def test_spec_example_validates() -> None:
    trajectory = validate(load_example())
    assert trajectory.schema_version == "trajectory/1.0"
    assert trajectory.domain == "tau2_retail"
    assert trajectory.actor.mode == "no_think"
    assert trajectory.outcome.reward == 0.0
    assert trajectory.outcome.success is False
    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].action.kind == "tool_call"
    assert trajectory.steps[0].action.name == "find_user_id_by_email"


def test_spec_example_round_trips() -> None:
    """Normalizing twice yields identical output (C-02 property test, in miniature)."""
    once = validate(load_example())
    twice = Trajectory.model_validate_json(once.model_dump_json())
    assert once == twice


def test_unknown_field_is_rejected() -> None:
    payload = load_example()
    payload["surprise_field"] = "hello"
    with pytest.raises(ValidationError, match="surprise_field"):
        validate(payload)


# ---------------------------------------------------------------------------
# Acceptance criterion 3: a null reward is rejected
# ---------------------------------------------------------------------------


def test_null_reward_is_rejected() -> None:
    """Spec Section 10.3: `outcome.reward` MUST be present and non-null.

    A defaulted reward silently corrupts every win/loss contrast (ALG-006), and the
    corruption is invisible because the resulting numbers stay plausible.
    """
    payload = load_example()
    payload["outcome"]["reward"] = None
    with pytest.raises(ValidationError) as exc:
        validate(payload)
    assert "reward" in str(exc.value)


def test_missing_reward_is_rejected() -> None:
    payload = load_example()
    del payload["outcome"]["reward"]
    with pytest.raises(ValidationError) as exc:
        validate(payload)
    assert "reward" in str(exc.value)


def test_reward_never_defaults_to_zero() -> None:
    """Mutation test: prove no default exists, rather than trusting the annotation.

    If someone later writes `reward: float = 0.0`, this test fails, which is the point.
    """
    from psd.core.models import Outcome

    assert Outcome.model_fields["reward"].is_required(), (
        "outcome.reward acquired a default. A false zero corrupts every win/loss "
        "contrast downstream and does so invisibly. Revert it."
    )


# ---------------------------------------------------------------------------
# Remaining blocking rules of spec Section 10.3
# ---------------------------------------------------------------------------


def test_step_indices_must_be_contiguous_from_zero() -> None:
    payload = load_example()
    payload["steps"][0]["index"] = 1
    with pytest.raises(ValidationError, match="contiguous"):
        validate(payload)


def test_step_index_gap_is_rejected() -> None:
    payload = load_example()
    second = json.loads(json.dumps(payload["steps"][0]))
    second["index"] = 2  # gap at 1
    payload["steps"].append(second)
    with pytest.raises(ValidationError, match="contiguous"):
        validate(payload)


def test_actor_mode_outside_the_enum_is_rejected() -> None:
    payload = load_example()
    payload["actor"]["mode"] = "sort_of_thinking"
    with pytest.raises(ValidationError):
        validate(payload)


def test_missing_system_prompt_hash_is_rejected() -> None:
    """Condition-drift detection (ALG-010 Step 2) depends entirely on this field."""
    payload = load_example()
    del payload["harness"]["system_prompt_sha256"]
    with pytest.raises(ValidationError, match="system_prompt_sha256"):
        validate(payload)


def test_ended_before_started_is_rejected() -> None:
    payload = load_example()
    payload["ended_at"] = "2026-08-01T09:00:00Z"
    with pytest.raises(ValidationError, match="ended_at"):
        validate(payload)


# ---------------------------------------------------------------------------
# ASM-002: nullable token components
# ---------------------------------------------------------------------------


def test_missing_token_key_is_rejected() -> None:
    """A step with no `output_reasoning` key at all is a blocking violation."""
    payload = load_example()
    del payload["steps"][0]["tokens"]["output_reasoning"]
    with pytest.raises(ValidationError, match="output_reasoning"):
        validate(payload)


def test_explicit_null_token_is_admitted_but_marks_the_record_incomplete() -> None:
    """Present-but-unreported. Spec Section 15.4 requires null, never zero."""
    payload = load_example()
    payload["steps"][0]["tokens"]["output_reasoning"] = None
    trajectory = validate(payload)
    assert trajectory.steps[0].tokens.output_reasoning is None
    assert trajectory.steps[0].tokens.complete is False
    assert trajectory.token_accounting_complete is False


def test_output_total_refuses_to_partially_sum() -> None:
    """A partial sum would under-count, which spec Section 15.4 forbids."""
    tokens = StepTokens(
        input_total=10,
        input_cached=5,
        output_visible=12,
        output_tool_args=9,
        output_reasoning=None,
    )
    assert tokens.output_total is None, (
        "output_total summed around a missing component. That silently under-counts and "
        "every economic number downstream inherits the error."
    )


def test_output_total_sums_the_three_components() -> None:
    """Spec Section 5.5: visible + tool-call arguments + reasoning."""
    tokens = StepTokens(
        input_total=10,
        input_cached=5,
        output_visible=12,
        output_tool_args=9,
        output_reasoning=100,
    )
    assert tokens.output_total == 121


def test_token_counts_cannot_be_negative() -> None:
    with pytest.raises(ValidationError):
        StepTokens(
            input_total=-1,
            input_cached=0,
            output_visible=0,
            output_tool_args=0,
            output_reasoning=0,
        )


def test_complete_trajectory_reports_complete_accounting() -> None:
    assert validate(load_example()).token_accounting_complete is True


# ---------------------------------------------------------------------------
# Split artifact (spec Section 10.4)
# ---------------------------------------------------------------------------


def make_split(train: list[str], test: list[str]) -> Split:
    return Split(
        domain="alfworld",
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
        sampling={"strategy": "random_once_fixed", "seed": 20260801, "notes": None},
        train_task_ids=train,
        test_task_ids=test,
        counts={"train": len(train), "test": len(test)},
        sha256="b7c9" + "0" * 60,
    )


def test_split_accepts_disjoint_task_ids() -> None:
    split = make_split(["a", "b"], ["c", "d"])
    assert split.counts.train == 2


def test_overlapping_split_is_rejected() -> None:
    """FR-006 and spec Section 5.3: the splits are disjoint."""
    with pytest.raises(ValidationError, match="overlap"):
        make_split(["a", "b"], ["b", "c"])


def test_split_counts_must_match_the_id_lists() -> None:
    with pytest.raises(ValidationError, match=r"counts\.train"):
        Split(
            domain="alfworld",
            created_at=datetime(2026, 8, 1, tzinfo=UTC),
            sampling={"strategy": "random_once_fixed", "seed": 1, "notes": None},
            train_task_ids=["a"],
            test_task_ids=["b"],
            counts={"train": 50, "test": 1},
            sha256="b7c9" + "0" * 60,
        )


# ---------------------------------------------------------------------------
# Skill artifact (spec Section 10.6)
# ---------------------------------------------------------------------------


def test_skill_body_is_stored_verbatim() -> None:
    """RR-006: appended verbatim, unmodified, uncompressed, untruncated.

    Trailing whitespace and trailing newlines survive the model unchanged.
    """
    body = "# Rules\n\n## Rule 1: do the thing   \nDo it.\n\n"
    skill = Skill(
        skill_id="skl_1",
        key={
            "domain": "tau2_retail",
            "actor_model": "gpt-5.4-mini",
            "actor_mode": "no_think",
            "harness_version": "2026.06.1",
        },
        body_markdown=body,
        stats={"lines": 6, "tokens": 20, "rules": 1, "rules_with_citations": 0},
        lineage={
            "corpus_snapshot_sha256": "4d21" + "0" * 60,
            "corpus_composition": "paired",
            "instruction_version": "P/0.1",
            "instruction_sha256": "77aa" + "0" * 60,
            "distiller": {"runtime": "claude_code", "model": "claude-sonnet-5"},
            "analyzer_lib_version": "1.2.0",
            "distill_run_id": "dst_1",
            "distill_index": 0,
            "n_distill": 1,
        },
        state="draft",
        created_at=datetime(2026, 8, 2, tzinfo=UTC),
    )
    assert skill.body_markdown == body


def test_step_requires_its_token_block() -> None:
    with pytest.raises(ValidationError, match="tokens"):
        Step.model_validate(
            {
                "index": 0,
                "observation": {"kind": "user_turn", "text": "hi", "meta": {}},
                "output": {"text": "ok"},
                "action": {"kind": "noop"},
            }
        )
