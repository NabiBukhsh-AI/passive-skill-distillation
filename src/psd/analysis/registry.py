"""Analyzer registry and byte-stability harness (TASK-022, component C-05).

Every analyzer is a pure function over a corpus snapshot, wrapped here into the
`Analyzer` port shape so the distillation sandbox can mount them behind one interface.

FR-011 is the requirement this module exists to enforce: identical input yields
**byte-identical** output. Not equal-as-dicts, byte-identical. Every downstream number in
the system is computed from these reports, so an analyzer that reorders itself between
runs makes two evaluations incomparable for a reason nobody would think to look for.

Three rules, applied here rather than trusted:
  * every collection is totally ordered before it is emitted,
  * no wall-clock and no unseeded randomness anywhere in the call graph,
  * serialization is sorted-key JSON, so a dict that changes insertion order cannot
    change the bytes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from psd.analysis.contrast import Predicate, win_loss_contrast
from psd.analysis.error_frequency import DEFAULT_TAXONOMY, error_frequency_report
from psd.analysis.ngrams import DEFAULT_MAX_PER_TRAJECTORY, DEFAULT_MIN_COUNT, action_ngrams
from psd.analysis.profiles import get_profile
from psd.analysis.stalls import detect_stalls, stall_rate
from psd.core.models import Trajectory
from psd.core.ports import AnalyzerReport, LoadedCorpus

ANALYZER_LIB_VERSION = "0.1.0"


def _plain(value: Any) -> Any:
    """Render dataclasses, tuples, and sets into JSON-ready values, deterministically."""
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _plain(item) for key, item in sorted(asdict(value).items())}
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        # Sets have no order. Sorting is what makes them safe to emit at all.
        return [_plain(item) for item in sorted(value, key=str)]
    return value


def serialize_report(report: AnalyzerReport) -> str:
    """Byte-stable rendering. The harness diffs these strings."""
    return json.dumps(
        _plain(report.model_dump(mode="json")), sort_keys=True, indent=2, ensure_ascii=False
    )


# ---------------------------------------------------------------------------
# Analyzers, wrapped into the port shape
# ---------------------------------------------------------------------------


class ErrorFrequencyAnalyzer:
    """ALG-003."""

    name = "error_frequency"

    def run(self, corpus: LoadedCorpus, config: Mapping[str, Any]) -> AnalyzerReport:
        profile = get_profile(corpus.domain)
        report = error_frequency_report(
            list(corpus.trajectories),
            taxonomy=config.get("taxonomy", DEFAULT_TAXONOMY),
            profile=profile,
            max_evidence=int(config.get("max_evidence", 5)),
        )
        rows = [
            {
                "error_type": row.error_type,
                "trajectories_with": row.trajectories_with,
                "trajectory_rate": row.trajectory_rate,
                "occurrences": row.occurrences,
                "share_of_all_errors": row.share_of_all_errors,
            }
            for row in report.rows
        ]
        evidence = [
            {
                "error_type": e.error_type,
                "trajectory_id": e.trajectory_id,
                "step_index": e.step_index,
            }
            for row in report.rows
            for e in row.evidence
        ]
        return AnalyzerReport(
            analyzer=self.name,
            analyzer_lib_version=ANALYZER_LIB_VERSION,
            corpus_merkle_root=corpus.ref.merkle_root,
            rows=rows,
            evidence=evidence,
        )


class NgramAnalyzer:
    """ALG-004."""

    name = "action_ngrams"

    def run(self, corpus: LoadedCorpus, config: Mapping[str, Any]) -> AnalyzerReport:
        report = action_ngrams(
            list(corpus.trajectories),
            profile=get_profile(corpus.domain),
            n_values=tuple(config.get("n_values", (1, 2, 3))),
            min_count=int(config.get("min_count", DEFAULT_MIN_COUNT)),
            max_per_trajectory=int(config.get("max_per_trajectory", DEFAULT_MAX_PER_TRAJECTORY)),
        )
        rows = [
            {
                "ngram": list(row.ngram),
                "n": row.n,
                "count": row.count,
                "doc_freq": row.doc_freq,
                "count_in_wins": row.count_in_wins,
                "count_in_losses": row.count_in_losses,
                "lift": row.lift,
            }
            for row in report.rows
        ]
        return AnalyzerReport(
            analyzer=self.name,
            analyzer_lib_version=ANALYZER_LIB_VERSION,
            corpus_merkle_root=corpus.ref.merkle_root,
            rows=rows,
        )


class StallAnalyzer:
    """ALG-005."""

    name = "stalls"

    def run(self, corpus: LoadedCorpus, config: Mapping[str, Any]) -> AnalyzerReport:
        profile = get_profile(corpus.domain)
        reports = [
            detect_stalls(
                trajectory,
                profile,
                m_min=int(config.get("m_min", 3)),
                cycle_max=int(config.get("cycle_max", 4)),
            )
            for trajectory in sorted(corpus.trajectories, key=lambda t: t.trajectory_id)
        ]
        rows = [
            {
                "trajectory_id": report.trajectory_id,
                "stalled": report.stalled,
                "step_cap_hit": report.step_cap_hit,
                "runs": [
                    {
                        "start": run.start,
                        "end": run.end,
                        "action": run.action,
                        "length": run.length,
                        "kind": run.kind,
                        "period": run.period,
                    }
                    for run in report.stall_runs
                ],
            }
            for report in reports
        ]
        return AnalyzerReport(
            analyzer=self.name,
            analyzer_lib_version=ANALYZER_LIB_VERSION,
            corpus_merkle_root=corpus.ref.merkle_root,
            rows=rows,
            # The headline statistic: 28.7% without a skill, 5.3% with one.
            status=f"stall_rate={stall_rate(reports):.6f}",
        )


class ContrastAnalyzer:
    """ALG-006."""

    name = "win_loss_contrast"

    def run(self, corpus: LoadedCorpus, config: Mapping[str, Any]) -> AnalyzerReport:
        predicates: dict[str, Predicate] = dict(config.get("predicates", {}))
        if not predicates:
            predicates = default_predicates(corpus)
        report = win_loss_contrast(
            list(corpus.trajectories),
            predicates,
            paired=bool(config.get("paired", corpus.composition == "paired")),
            q=float(config.get("q", 0.10)),
        )
        rows = [
            {
                "predicate": row.predicate,
                "p_given_win": row.p_given_win,
                "p_given_loss": row.p_given_loss,
                "lift": row.lift,
                "n_win": row.n_win,
                "n_loss": row.n_loss,
                "p_value": row.p_value,
                "p_adjusted": row.p_adjusted,
                "significant": row.significant,
                "test_used": row.test_used,
            }
            for row in report.rows
        ]
        return AnalyzerReport(
            analyzer=self.name,
            analyzer_lib_version=ANALYZER_LIB_VERSION,
            corpus_merkle_root=corpus.ref.merkle_root,
            rows=rows,
            status=report.status,
        )


def default_predicates(corpus: LoadedCorpus) -> dict[str, Predicate]:
    """Predicates derived from the other analyzers, per ALG-006's input list.

    Built from error types actually present in this corpus plus stall presence and length
    buckets, so the contrast table is about what happened rather than about a fixed list
    someone guessed at.
    """
    profile = get_profile(corpus.domain)
    error_types = sorted(
        {
            error_type
            for trajectory in corpus.trajectories
            for error_type in trajectory.labels.error_types
        }
    )

    def error_predicate(name: str) -> Predicate:
        return lambda t: name in t.labels.error_types

    predicates: dict[str, Predicate] = {
        f"error:{name}": error_predicate(name) for name in error_types
    }
    predicates["stalled"] = lambda t: detect_stalls(t, profile).stalled
    predicates["step_cap_hit"] = lambda t: t.outcome.step_cap_hit
    predicates["long_episode"] = lambda t: len(t.steps) >= 20
    return predicates


ANALYZERS: tuple[Any, ...] = (
    ErrorFrequencyAnalyzer(),
    NgramAnalyzer(),
    StallAnalyzer(),
    ContrastAnalyzer(),
)


def registered_analyzers() -> list[str]:
    return sorted(analyzer.name for analyzer in ANALYZERS)


def run_all(
    corpus: LoadedCorpus, config: Mapping[str, Mapping[str, Any]] | None = None
) -> dict[str, AnalyzerReport]:
    """Run every analyzer. Returned in name order."""
    config = config or {}
    return {
        analyzer.name: analyzer.run(corpus, config.get(analyzer.name, {}))
        for analyzer in sorted(ANALYZERS, key=lambda a: a.name)
    }


def assert_byte_stable(
    corpus: LoadedCorpus, config: Mapping[str, Mapping[str, Any]] | None = None
) -> None:
    """FR-011 harness: run every analyzer twice, diff, require zero difference.

    Raises rather than returning a bool, because the only correct response to an unstable
    analyzer is to stop. Every number downstream depends on this holding.
    """
    first = run_all(corpus, config)
    second = run_all(corpus, config)
    differences = [
        name
        for name in sorted(first)
        if serialize_report(first[name]) != serialize_report(second[name])
    ]
    if differences:
        raise AssertionError(
            "analyzers are not byte-stable across runs: "
            + ", ".join(differences)
            + ". Look for unseeded randomness, wall-clock, or set/dict iteration order."
        )


def trajectories_from(corpus: LoadedCorpus) -> list[Trajectory]:
    return sorted(corpus.trajectories, key=lambda t: t.trajectory_id)


__all__ = [
    "ANALYZERS",
    "ANALYZER_LIB_VERSION",
    "ContrastAnalyzer",
    "ErrorFrequencyAnalyzer",
    "NgramAnalyzer",
    "StallAnalyzer",
    "assert_byte_stable",
    "default_predicates",
    "registered_analyzers",
    "run_all",
    "serialize_report",
    "trajectories_from",
]
