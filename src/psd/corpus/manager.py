"""Corpus assembly with leakage prevention (TASK-014, ALG-001, component C-04).

Implements ALG-001 exactly, including the two things the specification is emphatic about:

  * **Sampling is at TASK granularity, never trajectory granularity** (Step 4). Sampling
    trajectories independently breaks the paired contrast the win/loss analysis depends
    on: you can end up with the think arm of one task and the no-think arm of another,
    and ALG-006's matched-pair McNemar test then compares things that share nothing.

  * **Contamination aborts and writes nothing** (Step 5). Never filter and continue.
    Leakage has no symptom: a contaminated corpus produces a skill that scores BETTER on
    held-out tasks, so it reads as success. The only defence is to refuse.
"""

from __future__ import annotations

import logging
import random
import shutil
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from psd.core.models import CorpusManifest, Split, Trajectory
from psd.corpus import splits as splits_module
from psd.corpus.snapshot import (
    REWARD_VISIBILITY_MODE_LEVEL,
    mark_write_once,
    materialize,
)

log = logging.getLogger("psd.corpus")

#: ALG-001 Step 4: `failure_weighted` oversamples failures by this factor.
DEFAULT_FAILURE_WEIGHT = 2.0


class CorpusBuildError(RuntimeError):
    """Any condition that makes a corpus unsafe to build."""


class ContaminationError(CorpusBuildError):
    """Test-split task ids reached the selection (ALG-001 Step 5).

    This is a scientific-integrity event, not a validation error. Spec Section 25.4
    classifies the corresponding alert as SEV2.
    """


class CorpusBuildWarning(str):
    """A recorded, non-fatal condition. Carried in the manifest, never swallowed."""


def _task_outcome(trajectories: list[Trajectory]) -> bool:
    """Whether a TASK counts as a success, for stratification and weighting.

    A task can carry several trajectories (paired composition). The task is treated as
    successful when its `no_think` trajectory succeeded, because `no_think` is the mode
    being improved and its failures are what the method is derived from. Falls back to
    any available trajectory when no `no_think` arm exists. Recorded as ASM-007.
    """
    for trajectory in trajectories:
        if trajectory.actor.mode == "no_think":
            return trajectory.outcome.success
    return any(t.outcome.success for t in trajectories)


def select_tasks(
    by_task: dict[str, list[Trajectory]],
    strategy: str,
    sample_size: int | None,
    seed: int,
    failure_weight: float = DEFAULT_FAILURE_WEIGHT,
) -> tuple[list[str], list[str]]:
    """ALG-001 Step 4. Returns (selected task ids, warnings).

    Always at task granularity. `sorted` before any sampling so the result depends on the
    seed and not on dict insertion order.
    """
    warnings: list[str] = []
    task_ids = sorted(by_task)

    if strategy == "all":
        return task_ids, warnings

    if sample_size is None:
        raise CorpusBuildError(f"strategy {strategy!r} requires a sample_size")

    if sample_size > len(task_ids):
        warnings.append(
            f"requested {sample_size} tasks but only {len(task_ids)} are eligible; "
            "proceeding with all of them"
        )
        return task_ids, warnings

    rng = random.Random(seed)

    if strategy == "random_n":
        return sorted(rng.sample(task_ids, sample_size)), warnings

    successes = sorted(t for t in task_ids if _task_outcome(by_task[t]))
    failures = sorted(t for t in task_ids if not _task_outcome(by_task[t]))

    if strategy == "stratified_by_outcome":
        share = len(successes) / len(task_ids) if task_ids else 0.0
        want_success = min(len(successes), round(sample_size * share))
        want_failure = min(len(failures), sample_size - want_success)
        # Give any shortfall back to whichever pool still has room.
        deficit = sample_size - (want_success + want_failure)
        if deficit > 0:
            want_success = min(len(successes), want_success + deficit)
        stratified = rng.sample(successes, want_success) + rng.sample(failures, want_failure)
        return sorted(stratified), warnings

    if strategy == "failure_weighted":
        weights = {t: (failure_weight if t in set(failures) else 1.0) for t in task_ids}
        picked: list[str] = []
        remaining = list(task_ids)
        for _ in range(sample_size):
            total = sum(weights[t] for t in remaining)
            cut = rng.uniform(0.0, total)
            running = 0.0
            for candidate in remaining:
                running += weights[candidate]
                if running >= cut:
                    picked.append(candidate)
                    remaining.remove(candidate)
                    break
        return sorted(picked), warnings

    raise CorpusBuildError(f"unknown sample strategy {strategy!r}")


def compute_pass_rates(by_task: dict[str, list[Trajectory]]) -> dict[str, Any]:
    """ALG-001 Step 6. Mode-level pass rates over the selected set.

    Mode level, not per task: FR-022 and GAP-04 make per-task reward visibility a config
    switch, and `mode_level` is the reproduction default because the paper says `A` reads
    trajectory files and mode-level pass rates.
    """
    per_mode: dict[str, list[bool]] = defaultdict(list)
    for trajectories in by_task.values():
        for trajectory in trajectories:
            per_mode[trajectory.actor.mode].append(trajectory.outcome.success)

    return {
        mode: {
            "pass_rate": round(sum(results) / len(results), 6) if results else None,
            "episodes": len(results),
        }
        for mode, results in sorted(per_mode.items())
    }


def build_corpus_snapshot(
    *,
    domain: str,
    actor_model: str,
    trajectories: list[Trajectory],
    split: Split,
    destination: Path,
    composition: str = "no_think_only",
    sample_strategy: str = "all",
    sample_size: int | None = None,
    seed: int = 0,
    corpus_id: str | None = None,
    analyzer_lib_version: str = "0.1.0",
    redaction_policy_version: str = "redaction/1.0",
    analyzer_lib_source: Path | None = None,
    precomputed: dict[str, Any] | None = None,
    reward_visibility: str = REWARD_VISIBILITY_MODE_LEVEL,
    created_at: datetime | None = None,
) -> CorpusManifest:
    """ALG-001 end to end.

    Materializes into a temporary directory and moves it into place only after every
    check has passed, so an aborted build leaves nothing behind at `destination`.
    """
    warnings: list[str] = []

    # Step 1: the split must be intact. An edited split is how a test task reaches a
    # training corpus.
    splits_module.verify(split)

    # Step 2: eligible trajectories.
    train_ids = set(split.train_task_ids)
    eligible = [
        t
        for t in trajectories
        if t.domain == domain and t.actor.model == actor_model and t.task_id in train_ids
    ]

    by_task: dict[str, list[Trajectory]] = defaultdict(list)
    for trajectory in eligible:
        by_task[trajectory.task_id].append(trajectory)

    # Step 3: composition.
    dropped_unpaired: list[str] = []
    if composition == "paired":
        complete: dict[str, list[Trajectory]] = {}
        for task_id, group in by_task.items():
            modes = {t.actor.mode for t in group}
            if {"think", "no_think"} <= modes:
                complete[task_id] = group
            else:
                dropped_unpaired.append(task_id)
        by_task = defaultdict(list, complete)
        dropped_unpaired.sort()
        if dropped_unpaired:
            warnings.append(
                f"dropped {len(dropped_unpaired)} task(s) lacking both arms of the pair"
            )
        if not by_task:
            raise CorpusBuildError(
                "paired composition requires at least one task with both a think and a "
                "no_think trajectory; none were found"
            )
    elif composition == "no_think_only":
        filtered = {
            task_id: [t for t in group if t.actor.mode == "no_think"]
            for task_id, group in by_task.items()
        }
        by_task = defaultdict(list, {k: v for k, v in filtered.items() if v})
    else:
        raise CorpusBuildError(f"unknown composition {composition!r}")

    if not by_task:
        raise CorpusBuildError(
            f"no eligible trajectories for domain={domain!r} model={actor_model!r} "
            f"composition={composition!r}"
        )

    # Step 4: sample at TASK granularity.
    selected_ids, sampling_warnings = select_tasks(
        dict(by_task), sample_strategy, sample_size, seed
    )
    warnings.extend(sampling_warnings)
    selected = {task_id: by_task[task_id] for task_id in selected_ids}

    # Step 5: HARD CHECK. Abort, never filter. Nothing has been written yet.
    contamination = sorted(set(selected_ids) & set(split.test_task_ids))
    if contamination:
        log.critical(
            "contamination_abort",
            extra={
                "event": "contamination_abort",
                "domain": domain,
                "split_sha256": split.sha256,
                "contaminating_task_ids": contamination[:20],
                "contaminating_count": len(contamination),
            },
        )
        raise ContaminationError(
            f"corpus selection for domain {domain!r} contains {len(contamination)} "
            f"test-split task id(s): {contamination[:5]}"
            f"{' ...' if len(contamination) > 5 else ''}. "
            "Aborting; nothing was written. This is a scientific-integrity event: "
            "investigate the filter and the split artifact."
        )

    # The method is failure-derived. A corpus with no failures produces a weak skill, and
    # that is worth saying loudly rather than discovering after a distillation run.
    if all(_task_outcome(group) for group in selected.values()):
        warnings.append(
            "every selected task succeeded; the method derives rules from failures, so "
            "this corpus will produce a weak skill"
        )

    # Step 6.
    pass_rates = compute_pass_rates(selected)

    by_mode: dict[str, list[Trajectory]] = defaultdict(list)
    for group in selected.values():
        for trajectory in group:
            by_mode[trajectory.actor.mode].append(trajectory)

    manifest = CorpusManifest(
        corpus_id=corpus_id or f"cor_{split.sha256[:12]}_{seed}",
        domain=domain,
        actor_model=actor_model,
        composition=composition,
        sample_strategy=sample_strategy,
        sample_size=sample_size,
        seed=seed,
        split_sha256=split.sha256,
        merkle_root="0" * 64,  # replaced by materialize
        counts={
            "tasks": len(selected),
            "trajectories": sum(len(v) for v in selected.values()),
            **{f"trajectories_{mode}": len(v) for mode, v in sorted(by_mode.items())},
        },
        dropped_unpaired=dropped_unpaired,
        build_warnings=warnings,
        analyzer_lib_version=analyzer_lib_version,
        redaction_policy_version=redaction_policy_version,
        reward_visibility=reward_visibility,
        created_at=created_at or datetime.now(UTC),
    )

    # Steps 7 to 10, staged so `destination` never sees a partial corpus.
    staging = Path(tempfile.mkdtemp(prefix="psd-corpus-"))
    try:
        root = materialize(
            staging,
            manifest,
            dict(by_mode),
            pass_rates,
            analyzer_lib_source=analyzer_lib_source,
            precomputed=precomputed,
            reward_visibility=reward_visibility,
        )
        final = manifest.model_copy(update={"merkle_root": root})
        if destination.exists():
            raise CorpusBuildError(
                f"{destination} already exists; corpora are immutable and write-once"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging), str(destination))
    finally:
        if staging.exists():  # pragma: no cover - only on a failed move
            shutil.rmtree(staging, ignore_errors=True)

    # Step 10.
    mark_write_once(destination)

    for warning in warnings:
        log.warning(
            "corpus_build_warning",
            extra={
                "event": "corpus_build_warning",
                "detail": warning,
                "corpus_id": final.corpus_id,
            },
        )
    return final
