"""Failure-mode frequency analysis (TASK-018, ALG-003).

This is the analyzer that produced the paper's retail figures: the fabricated-argument
bug appears in **59% of rollouts** and accounts for **94% of observed tool errors**.

Those are two different numbers answering two different questions, and ALG-003's
implementation note is explicit that both must be first-class outputs. `trajectory_rate`
tells you how widespread a failure is; `share_of_all_errors` tells you how much of the
total damage it accounts for. A failure can be rare and dominant, or ubiquitous and
trivial, and a report carrying only one of them cannot tell you which.

Deterministic: detectors run in a fixed order, evidence is chosen by lowest trajectory id
then lowest step index, and every table is totally ordered before it is emitted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from psd.core.canonicalize import DEFAULT_PROFILE, DomainProfile
from psd.core.models import Step, Trajectory

DEFAULT_MAX_EVIDENCE = 5


@dataclass(frozen=True)
class ErrorEvent:
    error_type: str
    trajectory_id: str
    step_index: int


#: A detector is a deterministic predicate over one step in the context of its trajectory.
#: It returns zero or more error type names.
Detector = Callable[[Step, Trajectory, DomainProfile], Sequence[str]]


@dataclass(frozen=True)
class Taxonomy:
    """An ordered list of detectors. Order is part of the contract: ALG-003 Step 1 runs
    every detector in taxonomy order, and the co-occurrence matrix depends on it."""

    version: str
    detectors: tuple[tuple[str, Detector], ...] = ()

    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.detectors)


@dataclass(frozen=True)
class ErrorFrequencyRow:
    error_type: str
    trajectories_with: int
    trajectory_rate: float
    occurrences: int
    share_of_all_errors: float
    evidence: tuple[ErrorEvent, ...] = ()


@dataclass(frozen=True)
class ErrorFrequencyReport:
    rows: tuple[ErrorFrequencyRow, ...]
    total_error_events: int
    trajectories: int
    taxonomy_version: str
    #: ALG-003 edge case: a step may match several detectors. They are all recorded, and
    #: the co-occurrence matrix is reported rather than resolving first-match-wins, which
    #: would make the numbers depend on detector ordering.
    co_occurrence: dict[tuple[str, str], int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Domain-agnostic detectors
# ---------------------------------------------------------------------------


def detect_tool_error(step: Step, trajectory: Trajectory, profile: DomainProfile) -> list[str]:
    """Any tool call whose result came back an error."""
    if step.result is not None and step.result.status == "error":
        return [step.result.error_type or "tool_error"]
    return []


def detect_fabricated_auth_argument(
    step: Step, trajectory: Trajectory, profile: DomainProfile
) -> list[str]:
    """The paper's retail failure (spec Section 3.1).

    Fires when a tool is called with a value-sensitive argument whose value never appeared
    in any observation the agent had seen BY THAT POINT. That "by that point" is the whole
    test: an address the user supplies at turn 6 does not retroactively justify a call at
    turn 2.

    Works on redacted corpora, because the redactor gives the same value the same
    placeholder within an episode and different values different ones.
    """
    if step.action.kind != "tool_call" or not step.action.arguments:
        return []
    sensitive = profile.sensitive_args_for(step.action.name or "")
    if not sensitive:
        return []

    seen = " ".join(
        prior.observation.text for prior in trajectory.steps if prior.index <= step.index
    )
    findings: list[str] = []
    for key in sorted(sensitive):
        value = step.action.arguments.get(key)
        if isinstance(value, str) and value.strip() and value not in seen:
            findings.append("fabricated_auth_argument")
    return findings


def detect_step_cap_hit(step: Step, trajectory: Trajectory, profile: DomainProfile) -> list[str]:
    """Recorded once, on the final step, when the episode ran out of budget."""
    if trajectory.outcome.step_cap_hit and step.index == len(trajectory.steps) - 1:
        return ["step_cap_hit"]
    return []


DEFAULT_TAXONOMY = Taxonomy(
    version="taxonomy/1.0",
    detectors=(
        ("tool_error", detect_tool_error),
        ("fabricated_auth_argument", detect_fabricated_auth_argument),
        ("step_cap_hit", detect_step_cap_hit),
    ),
)


# ---------------------------------------------------------------------------
# ALG-003
# ---------------------------------------------------------------------------


def error_frequency_report(
    trajectories: Sequence[Trajectory],
    taxonomy: Taxonomy = DEFAULT_TAXONOMY,
    profile: DomainProfile = DEFAULT_PROFILE,
    max_evidence: int = DEFAULT_MAX_EVIDENCE,
) -> ErrorFrequencyReport:
    """ALG-003. Both `trajectory_rate` and `share_of_all_errors` are first-class."""
    events: list[ErrorEvent] = []

    # Step 1: every detector, in taxonomy order, over every step.
    for trajectory in sorted(trajectories, key=lambda t: t.trajectory_id):
        for step in trajectory.steps:
            for _name, detector in taxonomy.detectors:
                for error_type in detector(step, trajectory, profile):
                    events.append(ErrorEvent(error_type, trajectory.trajectory_id, step.index))

    # Step 2: aggregate.
    occurrences: dict[str, int] = {}
    trajectories_with: dict[str, set[str]] = {}
    for event in events:
        occurrences[event.error_type] = occurrences.get(event.error_type, 0) + 1
        trajectories_with.setdefault(event.error_type, set()).add(event.trajectory_id)

    co_occurrence: dict[tuple[str, str], int] = {}
    by_step: dict[tuple[str, int], set[str]] = {}
    for event in events:
        by_step.setdefault((event.trajectory_id, event.step_index), set()).add(event.error_type)
    for types in by_step.values():
        ordered = sorted(types)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                co_occurrence[(left, right)] = co_occurrence.get((left, right), 0) + 1

    total_events = len(events)
    total_trajectories = len(trajectories)

    rows: list[ErrorFrequencyRow] = []
    for error_type in sorted(occurrences):
        # Step 3: deterministic evidence selection.
        evidence = tuple(
            sorted(
                (e for e in events if e.error_type == error_type),
                key=lambda e: (e.trajectory_id, e.step_index),
            )[:max_evidence]
        )
        rows.append(
            ErrorFrequencyRow(
                error_type=error_type,
                trajectories_with=len(trajectories_with[error_type]),
                trajectory_rate=(
                    len(trajectories_with[error_type]) / total_trajectories
                    if total_trajectories
                    else 0.0
                ),
                occurrences=occurrences[error_type],
                share_of_all_errors=(
                    occurrences[error_type] / total_events if total_events else 0.0
                ),
                evidence=evidence,
            )
        )

    # Step 4: sort by share desc, then error_type asc.
    rows.sort(key=lambda r: (-r.share_of_all_errors, r.error_type))

    return ErrorFrequencyReport(
        rows=tuple(rows),
        total_error_events=total_events,
        trajectories=total_trajectories,
        taxonomy_version=taxonomy.version,
        co_occurrence=dict(sorted(co_occurrence.items())),
    )
