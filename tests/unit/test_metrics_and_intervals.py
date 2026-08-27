"""TASK-037 acceptance tests.

The stated criterion is precise:

    gap_recovery returns status gap_nonpositive for the Qwen ALFWorld numbers and
    gap_too_small for the GPT retail numbers, never a number.

So the fixtures are the paper's exact Table 1 values (spec Section 2.2), not invented
ones. Those two cells are the reason the guard exists: one has a denominator inside noise
and the other has a NEGATIVE denominator, and in both cases the unguarded ratio returns
something that reads like a result.
"""

from __future__ import annotations

import math

import pytest

from psd.core.metrics import (
    MIN_GAP,
    GapRecovery,
    gap_recovery,
    reasoning_premium,
    success_rate,
    token_reduction,
)
from psd.core.stats import (
    Interval,
    paired_bootstrap_delta,
    wilson_interval,
)

# ---------------------------------------------------------------------------
# The paper's Table 1 (spec Section 2.2). Scores are held-out success over 3 seeds.
# ---------------------------------------------------------------------------

GPT = {
    "alfworld": {"think": 0.713, "no_think": 0.567, "skill": 0.787},
    "ssb_verified": {"think": 0.613, "no_think": 0.447, "skill": 0.560},
    "tau2_telecom": {"think": 0.450, "no_think": 0.192, "skill": 0.333},
    "tau2_retail": {"think": 0.350, "no_think": 0.325, "skill": 0.408},
}

QWEN = {
    "alfworld": {"think": 0.773, "no_think": 0.827, "skill": 0.980},
    "ssb_verified": {"think": 0.560, "no_think": 0.640, "skill": 0.673},
    "tau2_telecom": {"think": 0.933, "no_think": 0.883, "skill": 0.933},
    "tau2_retail": {"think": 0.633, "no_think": 0.600, "skill": 0.558},
}


def recover(cell: dict[str, float]) -> GapRecovery:
    return gap_recovery(cell["no_think"], cell["think"], cell["skill"])


# ---------------------------------------------------------------------------
# The two acceptance cases, named explicitly
# ---------------------------------------------------------------------------


def test_qwen_alfworld_is_gap_nonpositive() -> None:
    """TASK-037 acceptance.

    Qwen's reasoning mode HURTS on ALFWorld: 0.773 think against 0.827 no-think. The
    denominator is -0.054, so the ratio inverts and a skill that improves things would
    report a negative "recovery".
    """
    result = recover(QWEN["alfworld"])
    assert result.status == "gap_nonpositive"
    assert result.value is None
    assert result.denominator == pytest.approx(-0.054)
    assert not result.reportable


def test_gpt_retail_is_gap_too_small() -> None:
    """TASK-037 acceptance.

    The think/no-think gap is 2.5 points, well inside noise at 40 tasks by 3 seeds. The
    unguarded ratio is 332%, which reads like a triumph and means nothing.
    """
    result = recover(GPT["tau2_retail"])
    assert result.status == "gap_too_small"
    assert result.value is None
    assert result.denominator == pytest.approx(0.025)
    assert not result.reportable


def test_neither_case_ever_returns_a_number() -> None:
    """Stated as "never a number", so it is asserted as such rather than implied."""
    for result in (recover(QWEN["alfworld"]), recover(GPT["tau2_retail"])):
        assert result.value is None
        assert not isinstance(result.value, float)


def test_the_unguarded_ratio_would_have_been_misleading() -> None:
    """Shows what the guard is preventing, so the guard's value is legible.

    Computing the raw quotient by hand reproduces the numbers spec Section 5.4 calls out:
    332% for GPT retail, and a negative for Qwen ALFWorld.
    """
    retail = GPT["tau2_retail"]
    raw_retail = (retail["skill"] - retail["no_think"]) / (retail["think"] - retail["no_think"])
    assert raw_retail == pytest.approx(3.32, abs=0.01)

    alfworld = QWEN["alfworld"]
    raw_qwen = (alfworld["skill"] - alfworld["no_think"]) / (
        alfworld["think"] - alfworld["no_think"]
    )
    assert raw_qwen < 0


# ---------------------------------------------------------------------------
# Where the ratio IS meaningful
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("domain", "expected"),
    [("alfworld", 1.507), ("ssb_verified", 0.681), ("tau2_telecom", 0.547)],
)
def test_gpt_cells_with_a_real_gap_are_reportable(domain: str, expected: float) -> None:
    """Reproduces the table in spec Section 5.4 exactly."""
    result = recover(GPT[domain])
    assert result.status == "ok"
    assert result.value == pytest.approx(expected, abs=0.001)
    assert result.reportable


def test_the_paper_headline_range_is_reproduced() -> None:
    """The paper claims 55% to over 100% recovery.

    Over the GPT cells where the gap is meaningful, the range is 54.7% to 150.7%, which
    is the claim. Retail's 332% is excluded by the guard, which is the point: the headline
    range is honest only once the unstable cell is removed from it.
    """
    reportable = [recover(cell) for cell in GPT.values() if recover(cell).reportable]
    values = sorted(r.value for r in reportable if r.value is not None)
    assert len(values) == 3
    assert values[0] == pytest.approx(0.547, abs=0.001)
    assert values[-1] == pytest.approx(1.507, abs=0.001)


@pytest.mark.parametrize(
    ("domain", "status"),
    [
        ("alfworld", "gap_nonpositive"),  # -0.054, reasoning hurts
        ("ssb_verified", "gap_nonpositive"),  # -0.080, reasoning hurts
        ("tau2_retail", "gap_too_small"),  # +0.033, inside noise
    ],
)
def test_three_of_four_qwen_cells_are_unreportable(domain: str, status: str) -> None:
    """Qwen's reasoning mode is unreliable on three of its four benchmarks.

    Two have a NEGATIVE gap (reasoning actively hurts) and one is inside noise. Reporting
    a recovery number for any of them would be inventing a result.
    """
    result = recover(QWEN[domain])
    assert result.status == status
    assert not result.reportable


def test_qwen_telecom_sits_exactly_on_the_threshold() -> None:
    """The one Qwen cell that IS reportable, and it lands exactly on the boundary.

    0.933 think against 0.883 no-think is a denominator of precisely 0.050, which is
    `MIN_GAP`. That is worth knowing rather than glossing: the reporting threshold is not
    a hypothetical, it falls exactly on a published cell, so moving it by a hair flips a
    real result between "100% recovery" and "not reportable".

    Recorded here so that anyone tuning `MIN_GAP` sees what they are moving.
    """
    result = recover(QWEN["tau2_telecom"])
    assert result.denominator == pytest.approx(MIN_GAP, abs=1e-9)
    assert result.status == "ok"
    assert result.value == pytest.approx(1.0, abs=0.001)

    # A threshold a hair higher makes the same published cell unreportable.
    stricter = gap_recovery(
        QWEN["tau2_telecom"]["no_think"],
        QWEN["tau2_telecom"]["think"],
        QWEN["tau2_telecom"]["skill"],
        min_gap=0.051,
    )
    assert stricter.status == "gap_too_small"


# ---------------------------------------------------------------------------
# Boundary behavior and rendering
# ---------------------------------------------------------------------------


def test_a_denominator_of_exactly_zero_is_nonpositive() -> None:
    assert gap_recovery(0.5, 0.5, 0.7).status == "gap_nonpositive"


def test_the_threshold_is_exclusive_below_and_inclusive_at() -> None:
    just_under = gap_recovery(0.5, 0.5 + MIN_GAP - 1e-9, 0.6)
    exactly_at = gap_recovery(0.5, 0.5 + MIN_GAP, 0.6)
    assert just_under.status == "gap_too_small"
    assert exactly_at.status == "ok"


def test_a_configurable_threshold_is_honoured() -> None:
    """Spec Section 5.4: "Config, not constant"."""
    assert gap_recovery(0.5, 0.52, 0.6, min_gap=0.01).status == "ok"
    assert gap_recovery(0.5, 0.52, 0.6, min_gap=0.10).status == "gap_too_small"


def test_a_negative_numerator_is_still_reportable_when_the_gap_is_real() -> None:
    """A skill that made things worse is a real, reportable finding.

    The guard is about the denominator. Suppressing negative recoveries would hide
    exactly the regression the paper reports on one of its eight cells.
    """
    result = gap_recovery(0.60, 0.75, 0.55)
    assert result.status == "ok"
    assert result.value is not None
    assert result.value < 0


def test_rendering_never_shows_a_bare_number_when_unreportable() -> None:
    """Spec Section 5.4: never as a number, never as zero."""
    nonpositive = recover(QWEN["alfworld"]).render()
    too_small = recover(GPT["tau2_retail"]).render()

    assert "not reportable" in nonpositive
    assert "did not help" in nonpositive
    assert "not reportable" in too_small
    assert "threshold" in too_small
    for rendered in (nonpositive, too_small):
        assert "0.0%" not in rendered
        assert "332" not in rendered


def test_rendering_a_reportable_value_is_a_percentage() -> None:
    assert recover(GPT["alfworld"]).render() == "150.7%"


# ---------------------------------------------------------------------------
# Token accounting (spec Section 5.5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("t_think", "t_mode", "expected"),
    [(3723, 952, 3.9), (3723, 832, 4.5), (1615, 565, 2.9), (9232, 991, 9.3)],
)
def test_token_reduction_reproduces_table_1(t_think: int, t_mode: int, expected: float) -> None:
    """Every one of these is a published Table 1 cell, to one decimal place."""
    result = token_reduction(t_think, t_mode)
    assert result is not None
    assert result == pytest.approx(expected, abs=0.05)


def test_a_missing_token_count_yields_none_never_a_substitute() -> None:
    """Spec Section 15.4. The economic argument rests on real measurements."""
    assert token_reduction(None, 952) is None
    assert token_reduction(3723, None) is None
    assert token_reduction(3723, 0) is None


def test_reasoning_premium_is_the_think_to_nothink_ratio() -> None:
    premium = reasoning_premium(3723, 952)
    assert premium is not None
    assert premium == pytest.approx(3.91, abs=0.01)


def test_success_rate_counts_rewards_above_the_threshold() -> None:
    assert success_rate([1.0, 0.0, 1.0, 0.0]) == 0.5


def test_success_rate_over_nothing_is_an_error_not_zero() -> None:
    """Zero would be indistinguishable from a genuinely failing condition."""
    with pytest.raises(ValueError, match="undefined"):
        success_rate([])


# ---------------------------------------------------------------------------
# Wilson intervals (spec Section 5.7)
# ---------------------------------------------------------------------------


def test_wilson_interval_brackets_the_point_estimate() -> None:
    interval = wilson_interval(23, 120)  # telecom no-think shape
    assert interval.low < 23 / 120 < interval.high


def test_wilson_stays_inside_zero_and_one_at_the_extremes() -> None:
    """The reason spec Section 5.7 requires Wilson over Wald.

    At p_hat = 0 a Wald interval is [0, 0], and near it Wald can extend below zero, which
    is not a possible success rate.
    """
    at_zero = wilson_interval(0, 120)
    at_one = wilson_interval(120, 120)
    assert at_zero.low == 0.0
    assert at_zero.high > 0.0, "a zero-success arm still has an upper bound"
    assert at_one.high == 1.0
    assert at_one.low < 1.0


def test_wilson_narrows_as_n_grows() -> None:
    small = wilson_interval(30, 60)
    large = wilson_interval(300, 600)
    assert (large.high - large.low) < (small.high - small.low)


def test_wilson_matches_a_hand_computed_value() -> None:
    """n=100, 50 successes, z=1.96. Standard worked example: about [0.404, 0.596]."""
    interval = wilson_interval(50, 100)
    assert interval.low == pytest.approx(0.4038, abs=0.001)
    assert interval.high == pytest.approx(0.5962, abs=0.001)


@pytest.mark.parametrize(
    ("successes", "n", "message"),
    [
        (-1, 10, "outside"),
        (11, 10, "outside"),
        (0, 0, "at least one observation"),
    ],
)
def test_wilson_refuses_impossible_input(successes: int, n: int, message: str) -> None:
    """More successes than trials, or no trials at all, is a caller bug.

    Returning a degenerate interval instead would let a miscounted arm flow into a gate
    decision looking like a real measurement.
    """
    with pytest.raises(ValueError, match=message):
        wilson_interval(successes, n)


def test_interval_excludes_zero_is_the_gate_predicate() -> None:
    """ALG-011 G1: promote only when the CI lower bound is above zero."""
    assert Interval(0.01, 0.16).excludes_zero
    assert not Interval(-0.02, 0.16).excludes_zero
    assert not Interval(0.0, 0.16).excludes_zero


# ---------------------------------------------------------------------------
# Paired bootstrap (spec Section 5.7)
# ---------------------------------------------------------------------------


def paired_fixture(
    n_tasks: int, baseline: float, treatment: float
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    base = {f"t{i}": [1.0 if i < n_tasks * baseline else 0.0] for i in range(n_tasks)}
    treat = {f"t{i}": [1.0 if i < n_tasks * treatment else 0.0] for i in range(n_tasks)}
    return base, treat


def test_paired_bootstrap_recovers_the_point_estimate() -> None:
    base, treat = paired_fixture(100, 0.30, 0.45)
    point, _ = paired_bootstrap_delta(base, treat, resamples=2000, seed=1)
    assert point == pytest.approx(0.15, abs=1e-9)


def test_paired_bootstrap_is_deterministic_given_a_seed() -> None:
    """Spec Section 18.1 forbids unseeded randomness, and this feeds a gate decision."""
    base, treat = paired_fixture(60, 0.30, 0.45)
    first = paired_bootstrap_delta(base, treat, resamples=1000, seed=7)
    second = paired_bootstrap_delta(base, treat, resamples=1000, seed=7)
    assert first == second


def test_the_seed_actually_changes_the_resampling() -> None:
    """Negative control: a "seeded" function that ignored its seed would pass the test
    above, since every call would agree with every other.

    Compared across several seeds rather than two, and over a fixture with varied
    per-task deltas. With a coarse two-valued fixture the 2.5 and 97.5 percentiles land
    on the same lattice points for most seeds, so a two-seed check is flaky for reasons
    that have nothing to do with the property being tested.
    """
    base = {f"t{i}": [1.0, 0.0, 1.0 if i % 3 else 0.0] for i in range(40)}
    treat = {f"t{i}": [1.0, 1.0 if i % 2 else 0.0, 1.0] for i in range(40)}

    intervals = {paired_bootstrap_delta(base, treat, resamples=800, seed=s)[1] for s in range(1, 7)}
    assert len(intervals) > 1, "the seed had no effect on the resampling"


def test_a_real_effect_produces_an_interval_above_zero() -> None:
    base, treat = paired_fixture(200, 0.30, 0.55)
    _, interval = paired_bootstrap_delta(base, treat, resamples=3000, seed=3)
    assert interval.excludes_zero


def test_no_effect_produces_an_interval_straddling_zero() -> None:
    """The case the gate must refuse to promote."""
    base, treat = paired_fixture(200, 0.40, 0.40)
    _, interval = paired_bootstrap_delta(base, treat, resamples=3000, seed=4)
    assert not interval.excludes_zero


def test_seeds_within_a_task_are_averaged_not_treated_as_independent() -> None:
    """Tasks are the blocking unit; seeds are within-task replicates.

    Treating each episode as independent would ignore the pairing and understate the
    interval, which is the direction that manufactures false confidence.
    """
    base = {"t0": [0.0, 0.0, 1.0], "t1": [1.0, 1.0, 1.0]}
    treat = {"t0": [1.0, 1.0, 1.0], "t1": [1.0, 1.0, 1.0]}
    point, _ = paired_bootstrap_delta(base, treat, resamples=500, seed=5)
    # t0 improves by 2/3, t1 by 0. Mean over TASKS is 1/3.
    assert point == pytest.approx(1 / 3, abs=1e-9)


def test_a_partial_overlap_is_refused() -> None:
    """A paired comparison over a partial overlap silently changes what is compared."""
    with pytest.raises(ValueError, match="only one condition"):
        paired_bootstrap_delta({"a": [1.0], "b": [0.0]}, {"a": [1.0]}, resamples=10, seed=0)


def test_no_shared_tasks_is_refused() -> None:
    with pytest.raises(ValueError, match="not paired"):
        paired_bootstrap_delta({"a": [1.0]}, {"b": [1.0]}, resamples=10, seed=0)


def test_the_interval_brackets_the_point_estimate() -> None:
    base, treat = paired_fixture(80, 0.35, 0.50)
    point, interval = paired_bootstrap_delta(base, treat, resamples=2000, seed=6)
    assert interval.low <= point <= interval.high
    assert math.isfinite(interval.low)
    assert math.isfinite(interval.high)
