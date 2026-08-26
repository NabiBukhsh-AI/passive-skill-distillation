"""Stall and loop detection (TASK-020, ALG-005).

Finds no-progress runs: the failure mode the paper reports falling from 28.7% of
trajectories to 5.3% once a skill is injected.

ALG-005 names its own most dangerous failure condition, and it is worth restating because
it is silent: if the observation fingerprint does not normalize volatile fields, nothing
ever compares equal and recall drops to **zero** while the analyzer keeps reporting
cleanly. A harness that stamps `Step 14 of 40` into every observation defeats the whole
detector. So normalization is applied first, and there is a test that injects a step
counter and asserts detection survives it.

Pure and deterministic: no wall-clock, no randomness, every output totally ordered.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from psd.core.canonicalize import DEFAULT_PROFILE, DomainProfile, canonicalize_action
from psd.core.models import Trajectory

#: ALG-005 defaults.
DEFAULT_M_MIN = 3
DEFAULT_CYCLE_MAX = 4

#: Applied to every observation before fingerprinting, on top of whatever the domain
#: profile adds. These are the volatile fields common to every harness we have seen.
_UNIVERSAL_VOLATILE = (
    r"\bstep\s+\d+(\s+of\s+\d+)?\b",
    r"\bturn\s+\d+(\s+of\s+\d+)?\b",
    r"\bt\s*=\s*\d+\b",
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?Z?\b",
    r"\b\d+\s*ms\b",
    r"\[\s*\d+\s*/\s*\d+\s*\]",
)


@dataclass(frozen=True)
class StallRunResult:
    start: int
    end: int
    action: str
    length: int
    kind: str
    period: int | None = None


@dataclass(frozen=True)
class StallReport:
    trajectory_id: str
    stall_runs: tuple[StallRunResult, ...]
    stalled: bool
    step_cap_hit: bool

    @property
    def longest(self) -> int:
        return max((run.length for run in self.stall_runs), default=0)


def normalize_observation(text: str, profile: DomainProfile = DEFAULT_PROFILE) -> str:
    """Strip volatile fields so two identical situations fingerprint identically.

    ALG-005's failure condition in one function. Order matters only in that all patterns
    are applied; the result is lowercased and whitespace-collapsed so incidental
    formatting differences do not defeat comparison either.
    """
    normalized = text
    for pattern in (*_UNIVERSAL_VOLATILE, *profile.volatile_observation_patterns):
        normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE)

    # Removing "Step 3 of 40" from "Step 3 of 40. You see..." leaves an orphaned ".",
    # which is enough to make two identical situations fingerprint differently. Tidying
    # the punctuation a substitution left behind is part of the normalization, not
    # cosmetic.
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s+([.,;:!?])", r"", normalized)
    normalized = re.sub(r"^[\s.,;:!?()\[\]-]+", "", normalized)
    return normalized.strip().lower()


def observation_fingerprint(text: str, profile: DomainProfile = DEFAULT_PROFILE) -> str:
    return hashlib.sha256(normalize_observation(text, profile).encode("utf-8")).hexdigest()


def _repeat_runs(
    actions: list[str],
    fingerprints: list[str],
    m_min: int,
    whitelist: frozenset[str],
) -> list[StallRunResult]:
    """ALG-005 Step 2: maximal runs of identical action AND unchanged observation."""
    runs: list[StallRunResult] = []
    index = 0
    length = len(actions)
    while index < length:
        end = index
        while (
            end + 1 < length
            and actions[end + 1] == actions[index]
            and fingerprints[end + 1] == fingerprints[index]
        ):
            end += 1
        run_length = end - index + 1
        if run_length >= m_min and actions[index] not in whitelist:
            runs.append(
                StallRunResult(
                    start=index,
                    end=end,
                    action=actions[index],
                    length=run_length,
                    kind="repeat_action",
                )
            )
        index = end + 1
    return runs


def _cycle_runs(
    actions: list[str],
    fingerprints: list[str],
    cycle_max: int,
    whitelist: frozenset[str],
) -> list[StallRunResult]:
    """ALG-005 Step 3: periodic patterns covering at least two full periods.

    A cycle is only a stall if the observations repeat with it. Two alternating actions
    that keep changing the world are progress, not a loop.
    """
    runs: list[StallRunResult] = []
    length = len(actions)
    for period in range(2, cycle_max + 1):
        index = 0
        while index + 2 * period <= length:
            end = index
            while (
                end + period < length
                and actions[end + period] == actions[end]
                and fingerprints[end + period] == fingerprints[end]
            ):
                end += 1
            covered = end + period - index
            if covered >= 2 * period:
                pattern = actions[index : index + period]
                # A period whose actions are all identical is a repeat-action run, not a
                # cycle. Step 2 already found it, and labelling the paper's 20-step
                # repeated `look` as a period-2 cycle would describe it wrongly.
                degenerate = len(set(pattern)) == 1
                if not degenerate and not set(pattern) <= whitelist:
                    runs.append(
                        StallRunResult(
                            start=index,
                            end=index + covered - 1,
                            action="|".join(pattern),
                            length=covered,
                            kind="cycle",
                            period=period,
                        )
                    )
                index = index + covered
            else:
                index += 1
    return runs


#: Tie-break order when two runs cover the same span. A repeat-action stall is the more
#: specific description, so it outranks a cycle that merely happens to fit.
_KIND_RANK = {"repeat_action": 0, "cycle": 1}


def _merge(runs: list[StallRunResult]) -> tuple[StallRunResult, ...]:
    """ALG-005 Step 4: merge overlapping runs, preferring the longer.

    Sorted by (start, -length, kind rank, action), so the result is a function of the runs
    and not of the order the detectors happened to produce them in.
    """
    ordered = sorted(runs, key=lambda r: (r.start, -r.length, _KIND_RANK.get(r.kind, 9), r.action))
    kept: list[StallRunResult] = []
    for run in ordered:
        if any(run.start >= k.start and run.end <= k.end for k in kept):
            continue
        kept.append(run)
    return tuple(sorted(kept, key=lambda r: (r.start, r.end, r.kind, r.action)))


def detect_stalls(
    trajectory: Trajectory,
    profile: DomainProfile = DEFAULT_PROFILE,
    m_min: int = DEFAULT_M_MIN,
    cycle_max: int = DEFAULT_CYCLE_MAX,
) -> StallReport:
    """ALG-005 over one trajectory."""
    actions = [canonicalize_action(step.action, profile) for step in trajectory.steps]
    fingerprints = [
        observation_fingerprint(step.observation.text, profile) for step in trajectory.steps
    ]
    whitelist = profile.stall_whitelist

    runs = _repeat_runs(actions, fingerprints, m_min, whitelist)
    runs.extend(_cycle_runs(actions, fingerprints, cycle_max, whitelist))
    merged = _merge(runs)

    return StallReport(
        trajectory_id=trajectory.trajectory_id,
        stall_runs=merged,
        stalled=bool(merged),
        # Recorded separately, per ALG-005's edge cases. A capped episode is strong
        # evidence of a stall even when the pattern is irregular, but it is evidence, not
        # a detection, and conflating the two would inflate the reported stall rate.
        step_cap_hit=trajectory.outcome.step_cap_hit,
    )


def stall_rate(reports: list[StallReport]) -> float:
    """Fraction of trajectories containing at least one stall (spec Section 5.8).

    This is the 28.7% / 5.3% statistic. Trajectory-level, not run-level: one episode with
    four separate loops still counts once.
    """
    if not reports:
        return 0.0
    return sum(1 for report in reports if report.stalled) / len(reports)
