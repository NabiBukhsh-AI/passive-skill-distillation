"""Reported metrics (TASK-037, spec Sections 5.4 and 5.5).

The centrepiece is `gap_recovery`, and it is guarded for a reason that the paper's own
tables demonstrate.

Gap recovery is `(S_skill - S_nothink) / (S_think - S_nothink)`. It answers "how much of
the reasoning-mode advantage did the skill recover", which is only a meaningful question
when there IS a reasoning-mode advantage. On the published numbers:

  * GPT-5.4-mini on tau2-retail: the think/no-think gap is 2.5 points, well inside noise
    for 40 tasks by 3 seeds. The ratio computes to 332%, which carries no information but
    reads like a triumph.
  * Qwen3.6-27B on ALFWorld and SSB-Verified: the denominator is NEGATIVE, because that
    model's reasoning mode HURTS. The ratio then inverts: a skill that improves things
    produces a negative "recovery".

So the function returns `None` with a status rather than a number, and downstream
reporting must render that as "not reportable" with the reason. Spec Section 5.4 is
explicit: never as a number, never as zero.

**The promotion gate must never read this.** ALG-011 Step 3 says report only, and gates on
absolute delta with a confidence interval instead. A ratio with an unstable denominator is
not a decision variable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

#: Spec Section 5.4. Below a 5-point think/no-think gap the ratio is not reportable.
#: Config, not a constant: it is a reporting threshold, and Section 5.7's minimum
#: detectable effect at N=40 by 3 seeds is 12 to 15 points, so 5 is already generous.
MIN_GAP = 0.05

GapStatus = Literal["ok", "gap_too_small", "gap_nonpositive"]


@dataclass(frozen=True)
class GapRecovery:
    """Spec Section 5.4.

    `value` is `None` unless `status == "ok"`. Reporting code must branch on `status`
    rather than coercing `value`, because `None` here means "the question does not apply",
    which is different from zero recovery.
    """

    value: float | None
    status: GapStatus
    numerator: float
    denominator: float
    ci_low: float | None = None
    ci_high: float | None = None

    @property
    def reportable(self) -> bool:
        return self.status == "ok"

    def render(self) -> str:
        """How this appears in a report. Never a bare number when not reportable."""
        if self.status == "ok" and self.value is not None:
            return f"{self.value:.1%}"
        if self.status == "gap_nonpositive":
            return (
                "not reportable (think/no-think gap is not positive: "
                f"{self.denominator:+.3f}; the reasoning mode did not help)"
            )
        return (
            "not reportable (think/no-think gap of "
            f"{self.denominator:.3f} is below the {MIN_GAP} reporting threshold)"
        )


def gap_recovery(
    s_nothink: float, s_think: float, s_skill: float, min_gap: float = MIN_GAP
) -> GapRecovery:
    """Guarded gap-recovery ratio (spec Section 5.4).

    Returns a status rather than a number whenever the denominator makes the ratio
    meaningless. Both failure modes occur in the paper's published tables.
    """
    numerator = s_skill - s_nothink
    denominator = s_think - s_nothink

    if denominator <= 0:
        return GapRecovery(None, "gap_nonpositive", numerator, denominator)
    if denominator < min_gap:
        return GapRecovery(None, "gap_too_small", numerator, denominator)
    return GapRecovery(numerator / denominator, "ok", numerator, denominator)


# ---------------------------------------------------------------------------
# Token accounting (spec Section 5.5)
# ---------------------------------------------------------------------------


def token_reduction(t_think: float | None, t_mode: float | None) -> float | None:
    """`Red(mode) = T_think / T_mode`, the Table 1 reduction column.

    `None` when either figure is unavailable, never a substituted value. Spec
    Section 15.4: a missing token count is null, and the entire economic argument rests
    on these numbers being real measurements.
    """
    if t_think is None or t_mode is None or t_mode <= 0:
        return None
    return t_think / t_mode


def reasoning_premium(t_think: float | None, t_nothink: float | None) -> float | None:
    """`rho = T_think / T_nothink` (spec Section 5.5)."""
    return token_reduction(t_think, t_nothink)


def success_rate(rewards: Sequence[float], threshold: float = 0.0) -> float:
    """`S(mode)`, the mean terminal success indicator (spec Section 5.4).

    Averaged over every episode, which is seeds by tasks. Spec Section 5.4 defines it as
    `1/(K*N) * sum` over both, so a mean over the flat episode list is the same quantity
    provided every task carries the same number of seeds. The evaluation orchestrator
    refuses to aggregate an incomplete run (ALG-010 Step 5), which is what makes that
    hold.
    """
    if not rewards:
        raise ValueError("success_rate over an empty set of episodes is undefined")
    return sum(1 for reward in rewards if reward > threshold) / len(rewards)
