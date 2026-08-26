"""Win/loss contrast (TASK-021, ALG-006).

This is the mechanism the paper describes for contrasting no-think failures against think
successes on identical tasks. Two modes:

  * unpaired: Fisher exact on the 2x2 table of predicate by outcome.
  * paired: exact McNemar on discordant task pairs, when the corpus carries both a think
    and a no_think arm for the same task.

ALG-006's implementation note is the important one and it is easy to get wrong: with 35 to
50 tasks **most predicates will not survive multiplicity correction, and that is not a
bug**. Effect sizes are reported with their counts and both raw and adjusted p-values, and
nothing is filtered to significance. Filtering would hand the distiller an empty table and
call it rigour.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from psd.core.models import Trajectory
from psd.core.stats import benjamini_hochberg, fisher_exact_two_sided, mcnemar_exact

#: A predicate over one trajectory. Must be pure.
Predicate = Callable[[Trajectory], bool]

DEFAULT_Q = 0.10
DEFAULT_WIN_THRESHOLD = 0.0


@dataclass(frozen=True)
class ContrastRow:
    predicate: str
    p_given_win: float
    p_given_loss: float
    lift: float
    n_win: int
    n_loss: int
    p_value: float
    p_adjusted: float
    significant: bool
    test_used: str
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContrastReport:
    rows: tuple[ContrastRow, ...]
    paired: bool
    wins: int
    losses: int
    #: Set when the corpus cannot support a contrast at all, rather than raising.
    status: str = "ok"


def _is_win(trajectory: Trajectory, threshold: float) -> bool:
    return trajectory.outcome.reward > threshold


def win_loss_contrast(
    trajectories: Sequence[Trajectory],
    predicates: dict[str, Predicate],
    *,
    paired: bool = False,
    q: float = DEFAULT_Q,
    win_threshold: float = DEFAULT_WIN_THRESHOLD,
    max_evidence: int = 3,
) -> ContrastReport:
    """ALG-006."""
    if paired:
        return _paired_contrast(trajectories, predicates, q, win_threshold, max_evidence)
    return _unpaired_contrast(trajectories, predicates, q, win_threshold, max_evidence)


def _unpaired_contrast(
    trajectories: Sequence[Trajectory],
    predicates: dict[str, Predicate],
    q: float,
    win_threshold: float,
    max_evidence: int,
) -> ContrastReport:
    ordered = sorted(trajectories, key=lambda t: t.trajectory_id)
    winners = [t for t in ordered if _is_win(t, win_threshold)]
    losers = [t for t in ordered if not _is_win(t, win_threshold)]

    # ALG-006 edge case: with no wins or no losses every lift is undefined. Return a
    # flagged report rather than raising: the caller asked a reasonable question about an
    # unsuitable corpus, and "unsuitable" is the answer.
    if not winners or not losers:
        return ContrastReport(
            rows=(),
            paired=False,
            wins=len(winners),
            losses=len(losers),
            status="degenerate_no_contrast",
        )

    raw: list[tuple[str, float, float, float, int, int, float, tuple[str, ...]]] = []
    for name in sorted(predicates):
        predicate = predicates[name]
        win_hits = [t for t in winners if predicate(t)]
        loss_hits = [t for t in losers if predicate(t)]
        a, b = len(win_hits), len(winners) - len(win_hits)
        c, d = len(loss_hits), len(losers) - len(loss_hits)

        # ALG-006 edge case: zero-variance predicates are dropped and recorded.
        if a + c == 0 or b + d == 0:
            continue

        p_win = a / len(winners)
        p_loss = c / len(losers)
        evidence = tuple(t.trajectory_id for t in (win_hits + loss_hits)[:max_evidence])
        raw.append(
            (
                name,
                p_win,
                p_loss,
                p_win - p_loss,
                a,
                c,
                fisher_exact_two_sided(a, b, c, d),
                evidence,
            )
        )

    adjusted = benjamini_hochberg([entry[6] for entry in raw], q=q)
    rows = tuple(
        ContrastRow(
            predicate=entry[0],
            p_given_win=entry[1],
            p_given_loss=entry[2],
            lift=entry[3],
            n_win=entry[4],
            n_loss=entry[5],
            p_value=entry[6],
            p_adjusted=adjusted[i].adjusted,
            significant=adjusted[i].significant,
            test_used="fisher_exact",
            evidence=entry[7],
        )
        for i, entry in enumerate(raw)
    )

    # Step 5: by adjusted significance, then |lift|. Ties broken by name so the table is
    # byte-stable.
    return ContrastReport(
        rows=tuple(sorted(rows, key=lambda r: (r.p_adjusted, -abs(r.lift), r.predicate))),
        paired=False,
        wins=len(winners),
        losses=len(losers),
    )


def _paired_contrast(
    trajectories: Sequence[Trajectory],
    predicates: dict[str, Predicate],
    q: float,
    win_threshold: float,
    max_evidence: int,
) -> ContrastReport:
    """ALG-006 Step 3.

    Pairs are built by task id, matching the think arm against the no_think arm. This is
    the contrast the paper describes: the same task, solved one way and not the other.
    """
    by_task: dict[str, dict[str, Trajectory]] = {}
    for trajectory in sorted(trajectories, key=lambda t: t.trajectory_id):
        by_task.setdefault(trajectory.task_id, {})[trajectory.actor.mode] = trajectory

    pairs = [
        (task_id, arms["think"], arms["no_think"])
        for task_id, arms in sorted(by_task.items())
        if "think" in arms and "no_think" in arms
    ]
    if not pairs:
        return ContrastReport(rows=(), paired=True, wins=0, losses=0, status="degenerate_no_pairs")

    wins = sum(1 for _, think, _ in pairs if _is_win(think, win_threshold))
    losses = len(pairs) - wins

    raw: list[tuple[str, float, float, float, int, int, float, tuple[str, ...]]] = []
    for name in sorted(predicates):
        predicate = predicates[name]
        # Discordant counts: present in exactly one arm.
        b = c = 0
        evidence: list[str] = []
        for task_id, think, no_think in pairs:
            in_think, in_no_think = predicate(think), predicate(no_think)
            if in_think and not in_no_think:
                b += 1
                evidence.append(task_id)
            elif in_no_think and not in_think:
                c += 1
                evidence.append(task_id)
        if b + c == 0:
            continue

        p_think = sum(1 for _, think, _ in pairs if predicate(think)) / len(pairs)
        p_no_think = sum(1 for _, _, nt in pairs if predicate(nt)) / len(pairs)
        raw.append(
            (
                name,
                p_think,
                p_no_think,
                p_think - p_no_think,
                b,
                c,
                mcnemar_exact(b, c),
                tuple(sorted(evidence)[:max_evidence]),
            )
        )

    adjusted = benjamini_hochberg([entry[6] for entry in raw], q=q)
    rows = tuple(
        ContrastRow(
            predicate=entry[0],
            p_given_win=entry[1],
            p_given_loss=entry[2],
            lift=entry[3],
            n_win=entry[4],
            n_loss=entry[5],
            p_value=entry[6],
            p_adjusted=adjusted[i].adjusted,
            significant=adjusted[i].significant,
            test_used="mcnemar_exact",
            evidence=entry[7],
        )
        for i, entry in enumerate(raw)
    )
    return ContrastReport(
        rows=tuple(sorted(rows, key=lambda r: (r.p_adjusted, -abs(r.lift), r.predicate))),
        paired=True,
        wins=wins,
        losses=losses,
    )
