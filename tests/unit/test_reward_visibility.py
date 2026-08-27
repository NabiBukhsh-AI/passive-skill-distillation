"""TASK-028 acceptance: ALG-007 Steps 2 and 3, the distiller's view of the corpus.

Acceptance criterion, verbatim: "with reward_visibility=mode_level, no per-task reward
field is present anywhere under the materialized /corpus", proven by a filesystem scan.

The scan is the point. Asserting on the return value of a function would only prove the
function behaves; the thing that matters is what is actually on disk at the path the
distiller is handed, because that is the whole surface it can read.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from psd.core.models import Split, Trajectory
from psd.corpus.manager import build_corpus_snapshot
from psd.corpus.snapshot import (
    REWARD_VISIBILITY_MODE_LEVEL,
    REWARD_VISIBILITY_PER_TASK,
    strip_per_task_rewards,
)
from psd.corpus.splits import split_from_upstream

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "trajectories"
    / "spec_section_10_3_example.json"
)
FIXED_TIME = datetime(2026, 8, 1, tzinfo=UTC)


def make_trajectory(task_id: str, *, success: bool = False) -> Trajectory:
    payload: dict[str, Any] = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["trajectory_id"] = f"trj_{task_id}"
    payload["task_id"] = task_id
    payload["outcome"]["success"] = success
    payload["outcome"]["reward"] = 1.0 if success else 0.0
    return Trajectory.model_validate_json(json.dumps(payload))


def make_split() -> Split:
    return split_from_upstream("tau2_retail", ["t1", "t2"], ["t9"], created_at=FIXED_TIME)


def build(destination: Path, visibility: str) -> Path:
    build_corpus_snapshot(
        domain="tau2_retail",
        actor_model="gpt-5.4-mini",
        trajectories=[make_trajectory("t1"), make_trajectory("t2", success=True)],
        split=make_split(),
        destination=destination,
        reward_visibility=visibility,
        created_at=FIXED_TIME,
    )
    return destination


def episode_files(corpus: Path) -> list[Path]:
    return sorted((corpus / "trajectories").rglob("*.json"))


# ---------------------------------------------------------------------------
# The acceptance criterion: a filesystem scan
# ---------------------------------------------------------------------------


def test_no_per_task_reward_survives_anywhere_under_the_corpus(tmp_path: Path) -> None:
    """TASK-028 acceptance, scanned rather than asserted on a return value."""
    corpus = build(tmp_path / "corpus", REWARD_VISIBILITY_MODE_LEVEL)

    files = episode_files(corpus)
    assert files, "no episode files were written, so the scan would pass vacuously"

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "reward" not in payload["outcome"], f"{path.name} still carries outcome.reward"
        assert "success" not in payload["outcome"], f"{path.name} still carries outcome.success"


def test_the_raw_text_contains_no_reward_key_either(tmp_path: Path) -> None:
    """Scans the bytes, not the parsed object.

    A nested copy of the outcome, or a reward echoed into some other field, would slip
    past a structural check on `payload["outcome"]` alone.
    """
    corpus = build(tmp_path / "corpus", REWARD_VISIBILITY_MODE_LEVEL)
    for path in episode_files(corpus):
        text = path.read_text(encoding="utf-8")
        assert '"reward"' not in text, f"{path.name} mentions a reward key"
        assert '"success"' not in text, f"{path.name} mentions a success key"


def test_pass_rates_survive_and_are_the_only_outcome_signal(tmp_path: Path) -> None:
    """ALG-007 Step 2: "leaving only pass_rates.json".

    Stripping rewards without leaving pass rates would remove the distiller's ONLY view
    of what worked, and the method is derived from contrasting outcomes.
    """
    corpus = build(tmp_path / "corpus", REWARD_VISIBILITY_MODE_LEVEL)
    rates = json.loads((corpus / "pass_rates.json").read_text(encoding="utf-8"))
    assert rates["no_think"]["pass_rate"] == 0.5
    assert rates["no_think"]["episodes"] == 2


def test_per_task_visibility_leaves_rewards_in_place(tmp_path: Path) -> None:
    """Negative control.

    Without it, a bug that emptied `outcome` unconditionally would pass every assertion
    above while destroying the per_task mode entirely.
    """
    corpus = build(tmp_path / "corpus", REWARD_VISIBILITY_PER_TASK)
    rewards = [
        json.loads(path.read_text(encoding="utf-8"))["outcome"]["reward"]
        for path in episode_files(corpus)
    ]
    assert sorted(rewards) == [0.0, 1.0]


# ---------------------------------------------------------------------------
# What is deliberately NOT stripped (ASM-008)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["termination", "steps_used", "step_cap_hit"])
def test_non_reward_outcome_fields_survive_mode_level(tmp_path: Path, field: str) -> None:
    """These describe HOW an episode ended, not whether it scored.

    Removing them would gut a failure-derived corpus to protect a distinction they do not
    carry. Recorded as ASM-008.
    """
    corpus = build(tmp_path / "corpus", REWARD_VISIBILITY_MODE_LEVEL)
    for path in episode_files(corpus):
        assert field in json.loads(path.read_text(encoding="utf-8"))["outcome"]


def test_steps_and_observations_are_untouched(tmp_path: Path) -> None:
    """The episode content is what the distiller analyses. Only outcomes are gated."""
    corpus = build(tmp_path / "corpus", REWARD_VISIBILITY_MODE_LEVEL)
    payload = json.loads(episode_files(corpus)[0].read_text(encoding="utf-8"))
    assert payload["steps"][0]["observation"]["text"]
    assert payload["steps"][0]["action"]["name"] == "find_user_id_by_email"


# ---------------------------------------------------------------------------
# The switch is recorded, and it changes the content address
# ---------------------------------------------------------------------------


def test_the_manifest_records_what_the_distiller_could_see(tmp_path: Path) -> None:
    """Two corpora built the same way but read differently are not comparable."""
    corpus = build(tmp_path / "corpus", REWARD_VISIBILITY_MODE_LEVEL)
    manifest = json.loads((corpus / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["reward_visibility"] == "mode_level"


def test_visibility_changes_the_merkle_root(tmp_path: Path) -> None:
    """A corpus the distiller sees differently is a different corpus.

    Sharing an address between the stripped and unstripped forms would make lineage lie:
    a skill's manifest would point at content that is not what the distiller read.
    """
    stripped = json.loads(
        (build(tmp_path / "a", REWARD_VISIBILITY_MODE_LEVEL) / "MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    full = json.loads(
        (build(tmp_path / "b", REWARD_VISIBILITY_PER_TASK) / "MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    assert stripped["merkle_root"] != full["merkle_root"]


def test_mode_level_is_the_reproduction_default(tmp_path: Path) -> None:
    """GAP-04: the paper says A reads trajectory files and mode-level pass rates."""
    build_corpus_snapshot(
        domain="tau2_retail",
        actor_model="gpt-5.4-mini",
        trajectories=[make_trajectory("t1")],
        split=make_split(),
        destination=tmp_path / "corpus",
        created_at=FIXED_TIME,
    )
    manifest = json.loads((tmp_path / "corpus" / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["reward_visibility"] == "mode_level"


def test_an_unknown_visibility_is_refused(tmp_path: Path) -> None:
    """Fail loudly. A typo must not silently fall through to showing everything.

    Two layers refuse it. The manifest's `Literal` fires first, at the type boundary,
    which is the earlier and better rejection; `materialize` keeps its own guard for
    callers that reach it directly. Either message names the field, so the test asserts
    on that rather than on which layer happened to win.
    """
    with pytest.raises(ValueError, match="reward_visibility"):
        build_corpus_snapshot(
            domain="tau2_retail",
            actor_model="gpt-5.4-mini",
            trajectories=[make_trajectory("t1")],
            split=make_split(),
            destination=tmp_path / "corpus",
            reward_visibility="everything_please",
            created_at=FIXED_TIME,
        )
    assert not (tmp_path / "corpus").exists(), "a rejected build still wrote a corpus"


def test_materialize_guards_the_value_independently(tmp_path: Path) -> None:
    """The lower layer refuses too, for a caller that bypasses the manifest.

    Without this, the only thing standing between a typo and a corpus with full rewards
    would be a type annotation on a model the caller may never construct.
    """
    from psd.core.models import CorpusManifest
    from psd.corpus.snapshot import materialize

    manifest = CorpusManifest(
        corpus_id="cor_test",
        domain="tau2_retail",
        actor_model="gpt-5.4-mini",
        composition="no_think_only",
        sample_strategy="all",
        seed=0,
        split_sha256="0" * 64,
        merkle_root="0" * 64,
        analyzer_lib_version="0.1.0",
        redaction_policy_version="redaction/1.0",
        created_at=FIXED_TIME,
    )
    with pytest.raises(ValueError, match="unknown reward_visibility"):
        materialize(
            tmp_path / "corpus",
            manifest,
            {"no_think": [make_trajectory("t1")]},
            {},
            reward_visibility="everything_please",
        )


# ---------------------------------------------------------------------------
# ALG-007 Step 3: precomputed analysis
# ---------------------------------------------------------------------------


def test_precomputed_analysis_is_absent_in_the_reproduction_default(tmp_path: Path) -> None:
    corpus = build(tmp_path / "corpus", REWARD_VISIBILITY_MODE_LEVEL)
    assert not (corpus / "analysis" / "precomputed").exists()


def test_the_analysis_library_is_still_present(tmp_path: Path) -> None:
    """Step 3 removes precomputed reports, not the importable library."""
    corpus = build(tmp_path / "corpus", REWARD_VISIBILITY_MODE_LEVEL)
    assert (corpus / "analysis" / "lib").is_dir()


# ---------------------------------------------------------------------------
# The stripping function on its own
# ---------------------------------------------------------------------------


def test_strip_is_idempotent() -> None:
    payload = {"outcome": {"reward": 1.0, "success": True, "termination": "done"}}
    once = strip_per_task_rewards(dict(payload))
    twice = strip_per_task_rewards(dict(once))
    assert once == twice == {"outcome": {"termination": "done"}}


def test_strip_tolerates_a_missing_outcome() -> None:
    assert strip_per_task_rewards({"steps": []}) == {"steps": []}
