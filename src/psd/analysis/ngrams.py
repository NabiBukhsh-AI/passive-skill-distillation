"""Action n-gram mining (TASK-019, ALG-004).

Counts recurring action sequences so the distiller can see procedure rather than
anecdote.

The `max_per_trajectory` cap is the part that earns its keep, and ALG-004's implementation
note says why: without it, the 20-step repeated-`look` run in the paper's Figure 2 would
contribute 20 to the unigram count for `look` on its own, and one pathological episode
would dominate the corpus statistics. Capped, a single episode can shout at most
`max_per_trajectory` times about any one n-gram.

Deterministic: sorted output, document frequency computed over trajectory ids, no
iteration-order dependence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from psd.core.canonicalize import DEFAULT_PROFILE, DomainProfile, canonicalize_action
from psd.core.models import Trajectory

DEFAULT_N_VALUES = (1, 2, 3)
DEFAULT_MIN_COUNT = 3
#: ALG-004 edge case. One 40-step stall must not swamp the table.
DEFAULT_MAX_PER_TRAJECTORY = 5


@dataclass(frozen=True)
class NgramRow:
    ngram: tuple[str, ...]
    n: int
    count: int
    doc_freq: int
    count_in_wins: int
    count_in_losses: int
    lift: float


@dataclass(frozen=True)
class NgramReport:
    rows: tuple[NgramRow, ...]
    n_values: tuple[int, ...]
    min_count: int
    max_per_trajectory: int
    wins: int
    losses: int


def action_sequence(trajectory: Trajectory, profile: DomainProfile = DEFAULT_PROFILE) -> list[str]:
    return [canonicalize_action(step.action, profile) for step in trajectory.steps]


def action_ngrams(
    trajectories: Sequence[Trajectory],
    profile: DomainProfile = DEFAULT_PROFILE,
    n_values: Sequence[int] = DEFAULT_N_VALUES,
    min_count: int = DEFAULT_MIN_COUNT,
    max_per_trajectory: int = DEFAULT_MAX_PER_TRAJECTORY,
    split_by_outcome: bool = True,
) -> NgramReport:
    """ALG-004."""
    counts: dict[tuple[str, ...], int] = {}
    doc_ids: dict[tuple[str, ...], set[str]] = {}
    win_docs: dict[tuple[str, ...], set[str]] = {}
    loss_docs: dict[tuple[str, ...], set[str]] = {}

    wins = 0
    losses = 0

    for trajectory in sorted(trajectories, key=lambda t: t.trajectory_id):
        is_win = trajectory.outcome.reward > 0
        if is_win:
            wins += 1
        else:
            losses += 1

        sequence = action_sequence(trajectory, profile)
        per_trajectory: dict[tuple[str, ...], int] = {}

        for n in sorted(set(n_values)):
            if len(sequence) < n:
                continue  # ALG-004 edge case: episodes shorter than n are skipped.
            for start in range(len(sequence) - n + 1):
                gram = tuple(sequence[start : start + n])
                if per_trajectory.get(gram, 0) >= max_per_trajectory:
                    continue
                per_trajectory[gram] = per_trajectory.get(gram, 0) + 1
                counts[gram] = counts.get(gram, 0) + 1

        for gram in per_trajectory:
            doc_ids.setdefault(gram, set()).add(trajectory.trajectory_id)
            if split_by_outcome:
                target = win_docs if is_win else loss_docs
                target.setdefault(gram, set()).add(trajectory.trajectory_id)

    rows: list[NgramRow] = []
    for gram in sorted(counts):
        if counts[gram] < min_count:
            continue  # Step 5: prune.
        in_wins = len(win_docs.get(gram, set()))
        in_losses = len(loss_docs.get(gram, set()))
        # Step 4: lift from DOCUMENT frequency, not raw count. A gram appearing 30 times
        # in one winning episode is not evidence that it causes winning.
        p_win = in_wins / wins if wins else 0.0
        p_loss = in_losses / losses if losses else 0.0
        rows.append(
            NgramRow(
                ngram=gram,
                n=len(gram),
                count=counts[gram],
                doc_freq=len(doc_ids.get(gram, set())),
                count_in_wins=in_wins,
                count_in_losses=in_losses,
                lift=p_win - p_loss,
            )
        )

    # Step 5: sort by |lift| desc, then count desc, then ngram asc. Fully deterministic.
    rows.sort(key=lambda r: (-abs(r.lift), -r.count, r.ngram))

    return NgramReport(
        rows=tuple(rows),
        n_values=tuple(sorted(set(n_values))),
        min_count=min_count,
        max_per_trajectory=max_per_trajectory,
        wins=wins,
        losses=losses,
    )
