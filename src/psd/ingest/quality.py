"""Data quality checks (TASK-016, spec Section 10.9).

Each check carries a threshold AND an action, because they are not the same decision. Two
of these quarantine a record, one blocks corpus creation outright, three alert, and one is
report-only. Collapsing them into "passed / failed" would lose the part that matters.

Nothing here repairs data. A check that fails routes the batch somewhere; the routing is
the alerting hook's job (TASK-067, TASK-068).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from psd.core.models import StrictModel, Trajectory

#: What a breach does. Spec Section 10.9, "Action on breach".
Action = Literal["alert", "quarantine", "block_corpus_creation", "report_only"]


class CheckResult(StrictModel):
    name: str
    value: float | None
    threshold: float | None
    comparison: str
    passed: bool
    action: Action
    detail: str

    @property
    def breached(self) -> bool:
        return not self.passed


class QualityReport(StrictModel):
    results: list[CheckResult]

    @property
    def breaches(self) -> list[CheckResult]:
        return [r for r in self.results if r.breached]

    @property
    def blocks_corpus_creation(self) -> bool:
        """Spec Section 10.9: redaction recall below threshold blocks corpus creation."""
        return any(r.action == "block_corpus_creation" and r.breached for r in self.results)

    def by_name(self, name: str) -> CheckResult:
        for result in self.results:
            if result.name == name:
                return result
        raise KeyError(name)


def _result(
    name: str,
    value: float | None,
    threshold: float | None,
    comparison: str,
    passed: bool,
    action: Action,
    detail: str,
) -> CheckResult:
    return CheckResult(
        name=name,
        value=value,
        threshold=threshold,
        comparison=comparison,
        passed=passed,
        action=action,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# The Section 10.9 table, one function per row
# ---------------------------------------------------------------------------

NORMALIZATION_SUCCESS_MIN = 0.99
REWARD_PRESENCE_MIN = 1.0
STEP_CONTIGUITY_MIN = 1.0
TOKEN_COMPLETENESS_MIN = 0.999
DUPLICATE_RATE_MAX = 0.001
REDACTION_RECALL_MIN = 0.98


def check_normalization_success_rate(accepted: int, submitted: int) -> CheckResult:
    rate = accepted / submitted if submitted else 1.0
    return _result(
        "normalization_success_rate",
        rate,
        NORMALIZATION_SUCCESS_MIN,
        ">",
        rate > NORMALIZATION_SUCCESS_MIN or submitted == 0,
        "alert",
        f"{accepted}/{submitted} records normalized; inspect the source mapper",
    )


def check_reward_presence(trajectories: Sequence[Trajectory]) -> CheckResult:
    """Must be 100%.

    The normalizer already quarantines a record with no reward, so this reads as
    redundant. It is not: it is the assertion that the quarantine path actually ran. If a
    reward-less trajectory ever reaches this check, something upstream defaulted it, and
    that is the failure mode spec Section 10.3 exists to prevent.
    """
    present = sum(1 for t in trajectories if t.outcome.reward is not None)
    rate = present / len(trajectories) if trajectories else 1.0
    return _result(
        "reward_presence",
        rate,
        REWARD_PRESENCE_MIN,
        ">=",
        rate >= REWARD_PRESENCE_MIN,
        "quarantine",
        f"{present}/{len(trajectories)} trajectories carry a reward",
    )


def check_step_contiguity(trajectories: Sequence[Trajectory]) -> CheckResult:
    contiguous = sum(
        1 for t in trajectories if [s.index for s in t.steps] == list(range(len(t.steps)))
    )
    rate = contiguous / len(trajectories) if trajectories else 1.0
    return _result(
        "step_contiguity",
        rate,
        STEP_CONTIGUITY_MIN,
        ">=",
        rate >= STEP_CONTIGUITY_MIN,
        "quarantine",
        f"{contiguous}/{len(trajectories)} trajectories have contiguous step indices",
    )


def check_token_accounting_completeness(trajectories: Sequence[Trajectory]) -> CheckResult:
    """Measured over STEPS, not trajectories, per spec Section 10.9.

    Below threshold the action is to alert and exclude the affected records from economic
    reporting, never to treat a missing component as zero (Section 15.4).
    """
    total = sum(len(t.steps) for t in trajectories)
    complete = sum(1 for t in trajectories for s in t.steps if s.tokens.complete)
    rate = complete / total if total else 1.0
    return _result(
        "token_accounting_completeness",
        rate,
        TOKEN_COMPLETENESS_MIN,
        ">",
        rate > TOKEN_COMPLETENESS_MIN or total == 0,
        "alert",
        f"{complete}/{total} steps have complete token accounting; "
        "incomplete steps are excluded from economic reporting, never zero-filled",
    )


def check_duplicate_rate(trajectories: Sequence[Trajectory]) -> CheckResult:
    hashes = [t.provenance.content_sha256 for t in trajectories]
    duplicates = len(hashes) - len(set(hashes))
    rate = duplicates / len(hashes) if hashes else 0.0
    return _result(
        "duplicate_rate",
        rate,
        DUPLICATE_RATE_MAX,
        "<",
        rate < DUPLICATE_RATE_MAX or not hashes,
        "alert",
        f"{duplicates} duplicate content hashes in {len(hashes)} records; review idempotency keys",
    )


def check_redaction_recall(recall: float) -> CheckResult:
    """The only check whose breach blocks corpus creation outright.

    Under-redacted text reaching a distiller can put personal data into a skill, and a
    skill is served in a system prompt to every episode.
    """
    return _result(
        "redaction_recall",
        recall,
        REDACTION_RECALL_MIN,
        ">",
        recall > REDACTION_RECALL_MIN,
        "block_corpus_creation",
        f"redaction recall {recall:.4f} on the labeled fixture set",
    )


def check_corpus_class_balance(trajectories: Sequence[Trajectory]) -> CheckResult:
    """Report only. Informs sampling strategy; never blocks.

    A corpus with no failures is not invalid, it is just a poor input: the method derives
    rules from failures. ALG-001 raises that as a build warning separately.
    """
    successes = sum(1 for t in trajectories if t.outcome.success)
    share = successes / len(trajectories) if trajectories else 0.0
    return _result(
        "corpus_class_balance",
        share,
        None,
        "report_only",
        True,
        "report_only",
        f"{successes}/{len(trajectories)} trajectories succeeded "
        f"({share:.1%}); informs sampling strategy",
    )


def run_all(
    trajectories: Sequence[Trajectory],
    *,
    submitted: int | None = None,
    redaction_recall: float | None = None,
    on_breach: Callable[[CheckResult], None] | None = None,
) -> QualityReport:
    """Run every Section 10.9 check and fire the alerting hook for each breach."""
    results = [
        check_normalization_success_rate(
            len(trajectories), submitted if submitted is not None else len(trajectories)
        ),
        check_reward_presence(trajectories),
        check_step_contiguity(trajectories),
        check_token_accounting_completeness(trajectories),
        check_duplicate_rate(trajectories),
        check_corpus_class_balance(trajectories),
    ]
    if redaction_recall is not None:
        results.append(check_redaction_recall(redaction_recall))

    report = QualityReport(results=results)
    if on_breach is not None:
        for breach in report.breaches:
            on_breach(breach)
    return report
