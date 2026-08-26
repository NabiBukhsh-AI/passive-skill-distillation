"""TASK-013, TASK-014, TASK-015 acceptance tests.

Criteria:
  * Identical inputs produce an identical Merkle root (determinism).
  * A poisoned filter aborts with a CRITICAL audit event and WRITES NOTHING.
  * Paired composition stays aligned; sampling is at task granularity.
  * The materialized layout matches spec Section 10.5, and the precomputed flag removes
    `analysis/precomputed/` when false.

The contamination test is the one the playbook singles out. It asserts the abort AND the
absence of output, because "abort" that leaves a half-written corpus on disk is how a
contaminated snapshot gets picked up by the next run.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from psd.core.models import Split, Trajectory
from psd.corpus.manager import (
    ContaminationError,
    CorpusBuildError,
    build_corpus_snapshot,
    compute_pass_rates,
    select_tasks,
)
from psd.corpus.snapshot import merkle_root
from psd.corpus.splits import (
    build_split,
    compute_sha256,
    load_split,
    split_from_upstream,
    task_split_map,
    verify,
    write_split,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
EXAMPLE = FIXTURES / "trajectories" / "spec_section_10_3_example.json"
FIXED_TIME = datetime(2026, 8, 1, tzinfo=UTC)


def make_trajectory(
    task_id: str,
    *,
    mode: str = "no_think",
    success: bool = False,
    domain: str = "tau2_retail",
    model: str = "gpt-5.4-mini",
    split: str = "train",
) -> Trajectory:
    payload: dict[str, Any] = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["trajectory_id"] = f"trj_{task_id}_{mode}"
    payload["task_id"] = task_id
    payload["domain"] = domain
    payload["split"] = split
    payload["actor"]["model"] = model
    payload["actor"]["mode"] = mode
    payload["outcome"]["success"] = success
    payload["outcome"]["reward"] = 1.0 if success else 0.0
    return Trajectory.model_validate_json(json.dumps(payload))


def make_split(train: list[str], test: list[str], domain: str = "tau2_retail") -> Split:
    return split_from_upstream(domain, train, test, created_at=FIXED_TIME)


# ---------------------------------------------------------------------------
# TASK-013: split artifacts
# ---------------------------------------------------------------------------


def test_split_is_content_addressed() -> None:
    split = make_split(["a", "b"], ["c", "d"])
    assert split.sha256 == compute_sha256(
        split.domain, split.sampling, split.train_task_ids, split.test_task_ids
    )
    verify(split)


def test_split_hash_ignores_creation_time() -> None:
    """Two people running the same command must get the same content address."""
    first = split_from_upstream("d", ["a"], ["b"], created_at=datetime(2026, 1, 1, tzinfo=UTC))
    second = split_from_upstream("d", ["a"], ["b"], created_at=datetime(2027, 6, 6, tzinfo=UTC))
    assert first.sha256 == second.sha256


def test_split_hash_ignores_input_ordering() -> None:
    assert (
        split_from_upstream("d", ["b", "a"], ["d", "c"]).sha256
        == split_from_upstream("d", ["a", "b"], ["c", "d"]).sha256
    )


def test_edited_split_fails_verification() -> None:
    """An edited split is exactly how a test task reaches a training corpus."""
    split = make_split(["a", "b"], ["c", "d"])
    tampered = split.model_copy(update={"train_task_ids": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(tampered)


def test_overlapping_split_is_refused_by_the_model() -> None:
    with pytest.raises(ValueError, match="overlap"):
        split_from_upstream("d", ["a", "b"], ["b", "c"])


def test_build_split_is_deterministic_given_a_seed() -> None:
    pool = [f"task_{i:03d}" for i in range(100)]
    first = build_split("alfworld", pool, 50, 50, seed=20260801, created_at=FIXED_TIME)
    second = build_split("alfworld", pool, 50, 50, seed=20260801, created_at=FIXED_TIME)
    assert first.sha256 == second.sha256
    assert first.train_task_ids == second.train_task_ids


def test_build_split_produces_disjoint_halves() -> None:
    pool = [f"task_{i:03d}" for i in range(100)]
    split = build_split("alfworld", pool, 50, 50, seed=1, created_at=FIXED_TIME)
    assert set(split.train_task_ids) & set(split.test_task_ids) == set()
    assert split.counts.train == 50
    assert split.counts.test == 50


def test_build_split_refuses_an_undersized_pool() -> None:
    with pytest.raises(ValueError, match="needs"):
        build_split("alfworld", ["a", "b"], 50, 50, seed=1)


def test_split_round_trips_through_disk(tmp_path: Path) -> None:
    split = make_split(["a", "b"], ["c", "d"])
    path = write_split(split, tmp_path)
    assert load_split(path).sha256 == split.sha256


def test_task_split_map_labels_both_halves() -> None:
    mapping = task_split_map(make_split(["a"], ["b"]))
    assert mapping == {"a": "train", "b": "test"}


# ---------------------------------------------------------------------------
# TASK-014: the contamination abort
# ---------------------------------------------------------------------------


def test_contamination_aborts_and_writes_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The poisoned-filter test (playbook Stage 2 item 2, ALG-001 Step 5).

    A trajectory whose task is in the TEST half is offered to the builder, and its task
    id has been slipped into the train list of the selection. The build must abort, emit
    a CRITICAL audit event, and leave nothing at the destination.
    """
    # A split whose train half legitimately contains t1 and t2, test half contains t9.
    split = make_split(["t1", "t2", "t9"], ["t8"])
    # The poison: t8 is in the test half, but a trajectory claims it is training data.
    trajectories = [
        make_trajectory("t1"),
        make_trajectory("t2"),
        make_trajectory("t8"),
    ]
    poisoned = split.model_copy(update={"train_task_ids": ["t1", "t2", "t9", "t8"]}, deep=True)
    # Re-address it so it passes the integrity check and the ONLY thing left to catch
    # this is ALG-001 Step 5.
    poisoned = poisoned.model_copy(
        update={
            "sha256": compute_sha256(
                poisoned.domain,
                poisoned.sampling,
                poisoned.train_task_ids,
                poisoned.test_task_ids,
            )
        }
    )

    destination = tmp_path / "corpus"
    with caplog.at_level(logging.CRITICAL), pytest.raises(ContaminationError, match="t8"):
        build_corpus_snapshot(
            domain="tau2_retail",
            actor_model="gpt-5.4-mini",
            trajectories=trajectories,
            split=poisoned,
            destination=destination,
            created_at=FIXED_TIME,
        )

    assert not destination.exists(), (
        "the corpus directory exists after a contamination abort; ALG-001 Step 5 "
        "requires the run to write nothing"
    )
    assert any(r.message == "contamination_abort" for r in caplog.records), (
        "no CRITICAL contamination_abort audit event was emitted"
    )


def test_contamination_abort_leaves_no_partial_directory(tmp_path: Path) -> None:
    """Nothing at all, not even an empty directory that a later run could mistake."""
    split = make_split(["t1", "t8"], ["t8_other"])
    poisoned = split.model_copy(update={"train_task_ids": ["t1", "t8_other"]})
    poisoned = poisoned.model_copy(
        update={
            "sha256": compute_sha256(
                poisoned.domain,
                poisoned.sampling,
                poisoned.train_task_ids,
                poisoned.test_task_ids,
            )
        }
    )
    destination = tmp_path / "corpus"
    with pytest.raises(ContaminationError):
        build_corpus_snapshot(
            domain="tau2_retail",
            actor_model="gpt-5.4-mini",
            trajectories=[make_trajectory("t1"), make_trajectory("t8_other")],
            split=poisoned,
            destination=destination,
            created_at=FIXED_TIME,
        )
    assert list(tmp_path.iterdir()) == []


def test_a_clean_build_succeeds(tmp_path: Path) -> None:
    """Negative control. Without this, the abort tests could pass for any reason."""
    split = make_split(["t1", "t2"], ["t8"])
    manifest = build_corpus_snapshot(
        domain="tau2_retail",
        actor_model="gpt-5.4-mini",
        trajectories=[make_trajectory("t1"), make_trajectory("t2")],
        split=split,
        destination=tmp_path / "corpus",
        created_at=FIXED_TIME,
    )
    assert manifest.counts["tasks"] == 2
    assert (tmp_path / "corpus" / "MANIFEST.json").is_file()


def test_a_tampered_split_aborts_before_selection(tmp_path: Path) -> None:
    split = make_split(["t1"], ["t8"])
    tampered = split.model_copy(update={"train_task_ids": ["t1", "t2"]})
    with pytest.raises(ValueError, match="hash mismatch"):
        build_corpus_snapshot(
            domain="tau2_retail",
            actor_model="gpt-5.4-mini",
            trajectories=[make_trajectory("t1")],
            split=tampered,
            destination=tmp_path / "corpus",
            created_at=FIXED_TIME,
        )
    assert not (tmp_path / "corpus").exists()


# ---------------------------------------------------------------------------
# TASK-014: determinism
# ---------------------------------------------------------------------------


def test_identical_inputs_produce_an_identical_merkle_root(tmp_path: Path) -> None:
    """TASK-014 acceptance: same inputs, same root, on any machine."""
    split = make_split(["t1", "t2", "t3"], ["t9"])
    trajectories = [make_trajectory(f"t{i}") for i in (1, 2, 3)]

    first = build_corpus_snapshot(
        domain="tau2_retail",
        actor_model="gpt-5.4-mini",
        trajectories=trajectories,
        split=split,
        destination=tmp_path / "a",
        created_at=FIXED_TIME,
    )
    second = build_corpus_snapshot(
        domain="tau2_retail",
        actor_model="gpt-5.4-mini",
        trajectories=list(reversed(trajectories)),
        split=split,
        destination=tmp_path / "b",
        created_at=FIXED_TIME,
    )
    assert first.merkle_root == second.merkle_root


def test_merkle_root_is_order_independent() -> None:
    hashes = ["c" * 64, "a" * 64, "b" * 64]
    assert merkle_root(hashes) == merkle_root(sorted(hashes))
    assert merkle_root(hashes) == merkle_root(list(reversed(hashes)))


def test_merkle_root_changes_when_content_changes(tmp_path: Path) -> None:
    split = make_split(["t1", "t2"], ["t9"])
    base = build_corpus_snapshot(
        domain="tau2_retail",
        actor_model="gpt-5.4-mini",
        trajectories=[make_trajectory("t1"), make_trajectory("t2")],
        split=split,
        destination=tmp_path / "a",
        created_at=FIXED_TIME,
    )
    changed = build_corpus_snapshot(
        domain="tau2_retail",
        actor_model="gpt-5.4-mini",
        trajectories=[make_trajectory("t1"), make_trajectory("t2", success=True)],
        split=split,
        destination=tmp_path / "b",
        created_at=FIXED_TIME,
    )
    assert base.merkle_root != changed.merkle_root


def test_corpora_are_write_once(tmp_path: Path) -> None:
    split = make_split(["t1"], ["t9"])

    def build() -> None:
        build_corpus_snapshot(
            domain="tau2_retail",
            actor_model="gpt-5.4-mini",
            trajectories=[make_trajectory("t1")],
            split=split,
            destination=tmp_path / "corpus",
            created_at=FIXED_TIME,
        )

    build()
    with pytest.raises(CorpusBuildError, match="immutable"):
        build()


# ---------------------------------------------------------------------------
# TASK-014: composition and task-granularity sampling
# ---------------------------------------------------------------------------


def test_paired_composition_keeps_both_arms(tmp_path: Path) -> None:
    split = make_split(["t1", "t2"], ["t9"])
    trajectories = [
        make_trajectory("t1", mode="no_think"),
        make_trajectory("t1", mode="think"),
        make_trajectory("t2", mode="no_think"),
        make_trajectory("t2", mode="think"),
    ]
    manifest = build_corpus_snapshot(
        domain="tau2_retail",
        actor_model="gpt-5.4-mini",
        trajectories=trajectories,
        split=split,
        destination=tmp_path / "corpus",
        composition="paired",
        created_at=FIXED_TIME,
    )
    assert manifest.counts["trajectories_think"] == 2
    assert manifest.counts["trajectories_no_think"] == 2
    assert (tmp_path / "corpus" / "trajectories" / "think").is_dir()
    assert (tmp_path / "corpus" / "trajectories" / "no_think").is_dir()


def test_paired_composition_drops_and_records_unpaired_tasks(tmp_path: Path) -> None:
    split = make_split(["t1", "t2"], ["t9"])
    trajectories = [
        make_trajectory("t1", mode="no_think"),
        make_trajectory("t1", mode="think"),
        make_trajectory("t2", mode="no_think"),  # no think arm
    ]
    manifest = build_corpus_snapshot(
        domain="tau2_retail",
        actor_model="gpt-5.4-mini",
        trajectories=trajectories,
        split=split,
        destination=tmp_path / "corpus",
        composition="paired",
        created_at=FIXED_TIME,
    )
    assert manifest.dropped_unpaired == ["t2"]
    assert manifest.counts["tasks"] == 1


def test_paired_composition_with_no_complete_pairs_aborts(tmp_path: Path) -> None:
    split = make_split(["t1"], ["t9"])
    with pytest.raises(CorpusBuildError, match="paired composition requires"):
        build_corpus_snapshot(
            domain="tau2_retail",
            actor_model="gpt-5.4-mini",
            trajectories=[make_trajectory("t1", mode="no_think")],
            split=split,
            destination=tmp_path / "corpus",
            composition="paired",
            created_at=FIXED_TIME,
        )


def test_no_think_only_composition_excludes_think_arms(tmp_path: Path) -> None:
    split = make_split(["t1"], ["t9"])
    manifest = build_corpus_snapshot(
        domain="tau2_retail",
        actor_model="gpt-5.4-mini",
        trajectories=[
            make_trajectory("t1", mode="no_think"),
            make_trajectory("t1", mode="think"),
        ],
        split=split,
        destination=tmp_path / "corpus",
        composition="no_think_only",
        created_at=FIXED_TIME,
    )
    assert manifest.counts["trajectories"] == 1
    assert not (tmp_path / "corpus" / "trajectories" / "think").exists()


def test_sampling_is_at_task_granularity_so_pairs_stay_aligned() -> None:
    """ALG-001 Step 4, the implementation note.

    Sampling trajectories independently would let the think arm of one task and the
    no-think arm of another into the same corpus, and ALG-006's matched-pair test would
    then compare episodes that share no task.
    """
    by_task = {
        f"t{i}": [make_trajectory(f"t{i}", mode="no_think"), make_trajectory(f"t{i}", mode="think")]
        for i in range(10)
    }
    selected, _ = select_tasks(by_task, "random_n", 4, seed=7)
    assert len(selected) == 4
    for task_id in selected:
        modes = {t.actor.mode for t in by_task[task_id]}
        assert modes == {"think", "no_think"}, "a task lost one of its arms"


def test_random_n_is_deterministic_given_a_seed() -> None:
    by_task = {f"t{i}": [make_trajectory(f"t{i}")] for i in range(20)}
    assert (
        select_tasks(by_task, "random_n", 5, seed=3)[0]
        == (select_tasks(by_task, "random_n", 5, seed=3)[0])
    )


def test_random_n_does_not_depend_on_dict_order() -> None:
    forward = {f"t{i}": [make_trajectory(f"t{i}")] for i in range(20)}
    backward = {k: forward[k] for k in reversed(list(forward))}
    assert (
        select_tasks(forward, "random_n", 5, seed=3)[0]
        == (select_tasks(backward, "random_n", 5, seed=3)[0])
    )


def test_oversized_sample_warns_and_proceeds_with_all() -> None:
    by_task = {f"t{i}": [make_trajectory(f"t{i}")] for i in range(3)}
    selected, warnings = select_tasks(by_task, "random_n", 50, seed=1)
    assert len(selected) == 3
    assert any("only 3 are eligible" in w for w in warnings)


def test_failure_weighted_oversamples_failures() -> None:
    by_task = {f"ok{i}": [make_trajectory(f"ok{i}", success=True)] for i in range(20)}
    by_task.update({f"bad{i}": [make_trajectory(f"bad{i}", success=False)] for i in range(20)})
    selected, _ = select_tasks(by_task, "failure_weighted", 20, seed=11)
    failures = sum(1 for t in selected if t.startswith("bad"))
    assert failures > 10, f"expected failures to be oversampled, got {failures}/20"


def test_unknown_strategy_is_refused() -> None:
    with pytest.raises(CorpusBuildError, match="unknown sample strategy"):
        select_tasks({"t1": [make_trajectory("t1")]}, "vibes", 1, seed=1)


def test_an_all_success_corpus_warns_loudly(tmp_path: Path) -> None:
    """ALG-001 edge case: the method is failure-derived."""
    split = make_split(["t1", "t2"], ["t9"])
    manifest = build_corpus_snapshot(
        domain="tau2_retail",
        actor_model="gpt-5.4-mini",
        trajectories=[make_trajectory("t1", success=True), make_trajectory("t2", success=True)],
        split=split,
        destination=tmp_path / "corpus",
        created_at=FIXED_TIME,
    )
    assert any("weak skill" in w for w in manifest.build_warnings)


def test_warnings_live_inside_the_manifest(tmp_path: Path) -> None:
    """So a corpus cannot be separated from the caveats it was built with."""
    split = make_split(["t1"], ["t9"])
    build_corpus_snapshot(
        domain="tau2_retail",
        actor_model="gpt-5.4-mini",
        trajectories=[make_trajectory("t1", success=True)],
        split=split,
        destination=tmp_path / "corpus",
        created_at=FIXED_TIME,
    )
    written = json.loads((tmp_path / "corpus" / "MANIFEST.json").read_text(encoding="utf-8"))
    assert written["build_warnings"]


def test_no_eligible_trajectories_aborts(tmp_path: Path) -> None:
    split = make_split(["t1"], ["t9"])
    with pytest.raises(CorpusBuildError, match="no eligible trajectories"):
        build_corpus_snapshot(
            domain="tau2_retail",
            actor_model="a-different-model",
            trajectories=[make_trajectory("t1")],
            split=split,
            destination=tmp_path / "corpus",
            created_at=FIXED_TIME,
        )


def test_pass_rates_are_mode_level() -> None:
    by_task = {
        "t1": [make_trajectory("t1", mode="no_think", success=True)],
        "t2": [make_trajectory("t2", mode="no_think", success=False)],
        "t3": [make_trajectory("t3", mode="think", success=True)],
    }
    rates = compute_pass_rates(by_task)
    assert rates["no_think"] == {"pass_rate": 0.5, "episodes": 2}
    assert rates["think"] == {"pass_rate": 1.0, "episodes": 1}


# ---------------------------------------------------------------------------
# TASK-015: the materialized layout
# ---------------------------------------------------------------------------


def build_at(tmp_path: Path, **overrides: Any) -> Path:
    split = make_split(["t1", "t2"], ["t9"])
    destination = tmp_path / "corpus"
    build_corpus_snapshot(
        domain="tau2_retail",
        actor_model="gpt-5.4-mini",
        trajectories=[make_trajectory("t1"), make_trajectory("t2")],
        split=split,
        destination=destination,
        created_at=FIXED_TIME,
        **overrides,
    )
    return destination


def test_layout_matches_spec_section_10_5(tmp_path: Path) -> None:
    corpus = build_at(tmp_path)
    for relative in [
        "MANIFEST.json",
        "README_FOR_DISTILLER.md",
        "pass_rates.json",
        "trajectories/no_think/t1.json",
        "trajectories/no_think/t2.json",
        "analysis/lib",
    ]:
        assert (corpus / relative).exists(), f"missing {relative}"


def test_precomputed_analysis_is_absent_by_default(tmp_path: Path) -> None:
    """Repro default. The paper's agent writes and runs its own analysis code, so
    shipping precomputed reports would change the method rather than implement it."""
    assert not (build_at(tmp_path) / "analysis" / "precomputed").exists()


def test_precomputed_analysis_appears_when_supplied(tmp_path: Path) -> None:
    corpus = build_at(tmp_path, precomputed={"error_frequency": {"rows": []}})
    assert (corpus / "analysis" / "precomputed" / "error_frequency.json").is_file()


def test_the_readme_is_orientation_not_instruction(tmp_path: Path) -> None:
    """Spec Section 10.5 is explicit: `README_FOR_DISTILLER.md` is NOT instruction P."""
    text = (build_at(tmp_path) / "README_FOR_DISTILLER.md").read_text(encoding="utf-8")
    assert "not your instruction" in text
    assert "SKILL.md" not in text, "the README must not restate P's output contract"


def test_manifest_records_lineage(tmp_path: Path) -> None:
    corpus = build_at(tmp_path)
    manifest = json.loads((corpus / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["split_sha256"]
    assert manifest["redaction_policy_version"] == "redaction/1.0"
    assert manifest["analyzer_lib_version"]
    assert len(manifest["merkle_root"]) == 64
