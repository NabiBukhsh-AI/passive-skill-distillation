"""Fixture corpora with hand-computed expected statistics (Stage 3).

The playbook asks for these BEFORE the analyzers, so the analyzers are written against
known-correct tables rather than the tables being read off whatever the analyzers happen
to produce. The expected values below are computed by hand in the comments, not by
running the code.

Three corpora, each reproducing a specific shape from the paper:

  * `retail_failure_corpus`   the 59% / 94% shape (spec Section 2.2)
  * `figure2_stall_trajectory` the 20-step repeated-`look` run (paper Figure 2)
  * `step_counter_trajectory`  the same stall, with a step counter in every observation,
                               which must NOT defeat detection (ALG-005 failure condition)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from psd.core.models import Trajectory
from psd.core.ports import CorpusRef, LoadedCorpus

ZERO_HASH = "0" * 64
FIXED_START = datetime(2026, 8, 1, 9, 0, 0, tzinfo=UTC)
FIXED_END = datetime(2026, 8, 1, 9, 30, 0, tzinfo=UTC)


def step(
    index: int,
    *,
    observation: str,
    action: dict[str, Any],
    result: dict[str, Any] | None = None,
    output_text: str = "",
) -> dict[str, Any]:
    return {
        "index": index,
        "observation": {"kind": "env", "text": observation, "meta": {}},
        "output": {"text": output_text, "reasoning_text": None, "reasoning_token_count": 0},
        "action": action,
        "result": result,
        "tokens": {
            "input_total": 100,
            "input_cached": 80,
            "output_visible": 10,
            "output_tool_args": 5,
            "output_reasoning": 0,
        },
        "latency_ms": 100,
        "cost_usd": 0.0001,
    }


def trajectory(
    trajectory_id: str,
    task_id: str,
    steps: list[dict[str, Any]],
    *,
    domain: str = "tau2_retail",
    mode: str = "no_think",
    reward: float = 0.0,
    success: bool = False,
    step_cap_hit: bool = False,
    error_types: list[str] | None = None,
) -> Trajectory:
    payload: dict[str, Any] = {
        "schema_version": "trajectory/1.0",
        "trajectory_id": trajectory_id,
        "tenant_id": "t_test",
        "domain": domain,
        "task_id": task_id,
        "split": "train",
        "actor": {
            "model": "gpt-5.4-mini",
            "mode": mode,
            "mode_flag": {},
            "decoding": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 2048},
        },
        "harness": {
            "name": "test_runner",
            "version": "1.0",
            "system_prompt_sha256": "a" * 64,
            "tools_sha256": "b" * 64,
            "max_steps": 40,
            "user_simulator": None,
        },
        "seed": 1,
        "started_at": FIXED_START.isoformat().replace("+00:00", "Z"),
        "ended_at": FIXED_END.isoformat().replace("+00:00", "Z"),
        "steps": steps,
        "outcome": {
            "reward": reward,
            "success": success,
            "termination": "done",
            "steps_used": len(steps),
            "step_cap_hit": step_cap_hit,
        },
        "totals": {
            "output_tokens": 15 * len(steps),
            "output_reasoning_tokens": 0,
            "input_tokens": 100 * len(steps),
            "input_cached_tokens": 80 * len(steps),
            "turns": len(steps),
            "cost_usd": 0.001,
        },
        "labels": {
            "error_types": error_types or [],
            "stall_runs": [],
            "annotator": "fixture",
        },
        "redaction": {"applied": True, "policy_version": "redaction/1.0", "counts": {}},
        "provenance": {
            "source": "fixture",
            "source_run_id": "run_fixture",
            "content_sha256": f"{abs(hash(trajectory_id)):064x}"[:64],
        },
        "normalization_warnings": [],
    }
    import json

    return Trajectory.model_validate_json(json.dumps(payload))


def as_corpus(
    trajectories: list[Trajectory],
    domain: str = "tau2_retail",
    composition: str = "no_think_only",
) -> LoadedCorpus:
    return LoadedCorpus(
        ref=CorpusRef(corpus_id="cor_fixture", merkle_root=ZERO_HASH),
        domain=domain,
        composition=composition,
        trajectories=trajectories,
        pass_rates={},
    )


# ---------------------------------------------------------------------------
# The retail failure-frequency corpus (spec Section 2.2)
# ---------------------------------------------------------------------------
#
# The paper's skill excerpt cites 13 of 22 rollouts and 17 of 18 tool errors.
#
# HAND-COMPUTED EXPECTED TABLE for `fabricated_auth_argument`:
#
#   trajectories_with     = 13
#   trajectory_rate       = 13 / 22            = 0.590909...   (the paper's 59%)
#   occurrences           = 17
#   share_of_all_errors   = 17 / 18            = 0.944444...   (the paper's 94%)
#
# Construction:
#   * 13 trajectories fabricate. Nine fabricate once, four fabricate twice, giving
#     9 + 8 = 17 fabrication events.
#   * Every fabricating step ALSO returns a tool error, so those 17 steps each produce
#     one `not_found` event too.
#   * One further trajectory has a single non-fabrication tool error (`rate_limited`),
#     which is the 1 of 18.
#   * The remaining 8 trajectories are clean.
#
# Note the total error count: 17 fabrication events + 17 co-occurring not_found + 1
# rate_limited = 35 events across all types. The 17/18 figure is the share among TOOL
# ERRORS specifically, which is what `share_of_all_errors` computes when the taxonomy is
# restricted to error-result detectors. Both framings are asserted in the tests.

REAL_EMAIL = "shopper@realmail.co"
FABRICATED_EMAIL = "user@example.com"


def _auth_step(index: int, email: str, *, supplied: bool, error: bool) -> dict[str, Any]:
    observation = (
        f"Customer says: my address is {email}" if supplied else "Customer says: I need help"
    )
    return step(
        index,
        observation=observation,
        action={
            "kind": "tool_call",
            "name": "find_user_id_by_email",
            "arguments": {"email": email},
            "arguments_raw": f'{{"email": "{email}"}}',
        },
        result=(
            {"status": "error", "error_type": "not_found", "text": "No user found."}
            if error
            else {"status": "ok", "error_type": None, "text": "user_123"}
        ),
    )


def retail_failure_corpus() -> list[Trajectory]:
    """22 trajectories reproducing the paper's retail failure shape."""
    trajectories: list[Trajectory] = []

    # 9 trajectories, one fabrication each.
    for i in range(9):
        trajectories.append(
            trajectory(
                f"trj_fab_single_{i:02d}",
                f"retail_train_{i:03d}",
                [_auth_step(0, FABRICATED_EMAIL, supplied=False, error=True)],
                reward=0.0,
                success=False,
            )
        )

    # 4 trajectories, two fabrications each.
    for i in range(4):
        trajectories.append(
            trajectory(
                f"trj_fab_double_{i:02d}",
                f"retail_train_1{i:02d}",
                [
                    _auth_step(0, FABRICATED_EMAIL, supplied=False, error=True),
                    _auth_step(1, FABRICATED_EMAIL, supplied=False, error=True),
                ],
                reward=0.0,
                success=False,
            )
        )

    # 1 trajectory with a non-fabrication tool error.
    trajectories.append(
        trajectory(
            "trj_other_error",
            "retail_train_200",
            [
                step(
                    0,
                    observation=f"Customer says: my address is {REAL_EMAIL}",
                    action={
                        "kind": "tool_call",
                        "name": "get_order_status",
                        "arguments": {"order_id": "W123"},
                    },
                    result={
                        "status": "error",
                        "error_type": "rate_limited",
                        "text": "Slow down.",
                    },
                )
            ],
            reward=0.0,
            success=False,
        )
    )

    # 8 clean, successful trajectories that use the address the customer supplied.
    for i in range(8):
        trajectories.append(
            trajectory(
                f"trj_clean_{i:02d}",
                f"retail_train_3{i:02d}",
                [_auth_step(0, REAL_EMAIL, supplied=True, error=False)],
                reward=1.0,
                success=True,
            )
        )

    return trajectories


# ---------------------------------------------------------------------------
# The Figure 2 stall (paper Figure 2, ALG-005)
# ---------------------------------------------------------------------------
#
# HAND-COMPUTED: 20 consecutive `look` actions whose observation never changes.
#   * one repeat_action run, start=0, end=19, length=20
#   * stalled = True
#
# The observation is byte-identical every step, which is exactly the pattern the paper
# illustrates.

STALL_OBSERVATION = (
    "You are in the middle of a room. Looking quickly around you, you see a cabinet 1, "
    "a countertop 1, a fridge 1, and a garbagecan 1."
)


def figure2_stall_trajectory(steps: int = 20) -> Trajectory:
    return trajectory(
        "trj_figure2_stall",
        "alfworld_train_001",
        [
            step(
                i,
                observation=STALL_OBSERVATION,
                action={"kind": "text_action", "text": "look"},
            )
            for i in range(steps)
        ],
        domain="alfworld",
        reward=0.0,
        success=False,
        step_cap_hit=True,
    )


# ---------------------------------------------------------------------------
# The step-counter negative fixture (ALG-005 failure condition)
# ---------------------------------------------------------------------------
#
# Same stall, but the harness stamps "Step N of 40" into every observation. A fingerprint
# that does not normalize volatile fields sees 20 DIFFERENT observations and reports no
# stall at all, silently, with recall zero.
#
# HAND-COMPUTED: identical to figure2 once normalized. One run, length 20, stalled=True.


def step_counter_trajectory(steps: int = 20) -> Trajectory:
    return trajectory(
        "trj_step_counter_stall",
        "alfworld_train_002",
        [
            step(
                i,
                observation=f"Step {i + 1} of 40. {STALL_OBSERVATION}",
                action={"kind": "text_action", "text": "look"},
            )
            for i in range(steps)
        ],
        domain="alfworld",
        reward=0.0,
        success=False,
        step_cap_hit=True,
    )


# ---------------------------------------------------------------------------
# A progressing trajectory: the negative control for stall detection
# ---------------------------------------------------------------------------
#
# HAND-COMPUTED: every observation differs and actions vary, so no runs, stalled=False.


def progressing_trajectory() -> Trajectory:
    commands = [
        "go to fridge 1",
        "open fridge 1",
        "take apple 1 from fridge 1",
        "go to countertop 1",
        "put apple 1 in countertop 1",
    ]
    return trajectory(
        "trj_progressing",
        "alfworld_train_003",
        [
            step(
                i,
                observation=f"You see a distinct situation number {i}.",
                action={"kind": "text_action", "text": command},
            )
            for i, command in enumerate(commands)
        ],
        domain="alfworld",
        reward=1.0,
        success=True,
    )


# ---------------------------------------------------------------------------
# A period-2 cycle: ALG-005 Step 3
# ---------------------------------------------------------------------------
#
# HAND-COMPUTED: `open fridge 1` / `close fridge 1` alternating for 8 steps with the
# observation alternating in lockstep. One cycle run, period=2, covering 8 steps.


def cycle_trajectory(periods: int = 4) -> Trajectory:
    steps: list[dict[str, Any]] = []
    for i in range(periods * 2):
        opening = i % 2 == 0
        steps.append(
            step(
                i,
                observation="The fridge 1 is closed." if opening else "The fridge 1 is open.",
                action={
                    "kind": "text_action",
                    "text": "open fridge 1" if opening else "close fridge 1",
                },
            )
        )
    return trajectory(
        "trj_cycle",
        "alfworld_train_004",
        steps,
        domain="alfworld",
        reward=0.0,
        success=False,
    )
