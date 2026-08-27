"""Content addressing and materialization for corpus snapshots (TASK-014, TASK-015).

The directory layout in `materialize` is a contract, not an implementation detail:
instruction `P` references these paths by name, so changing one changes the method
(spec Section 10.5).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from psd.core.models import CorpusManifest, Trajectory

CORPUS_LAYOUT_VERSION = "corpus/1.0"

README_FOR_DISTILLER = """\
# What is in this directory

Orientation only. This file is not your instruction.

- `trajectories/<mode>/<task_id>.json` is one episode per file. Each records, per step:
  the observation, the model's visible output, the action or tool call with its
  arguments, and the result. Each episode also records how it terminated.
- `pass_rates.json` holds mode-level pass rates over these episodes.
- `MANIFEST.json` describes how this corpus was assembled, including the sampling
  strategy and seed.
- `analysis/lib/` is an importable analysis library.
- `analysis/precomputed/` holds standard reports, when present. It is absent by default.

Personal data has been replaced with typed placeholders such as `<EMAIL_1>`. Placeholders
are stable within one episode: the same value always gets the same placeholder, and two
different values get different ones. So `<EMAIL_1>` appearing in both a user turn and a
later tool call means the agent used the value the user gave, and a tool call carrying
`<EMAIL_2>` that appears in no user turn means the agent supplied something the user
never said. Placeholder numbering carries no meaning across episodes.
"""


def content_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def merkle_root(file_hashes: list[str]) -> str:
    """ALG-001 Step 8: sha256 over the sorted list of per-file content hashes.

    Sorted, so the root is a function of the content and not of filesystem iteration
    order. Two machines must produce the same root from the same inputs (TASK-014
    acceptance), and directory listing order is not portable.
    """
    joined = "\n".join(sorted(file_hashes)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


#: ALG-007 Step 2, GAP-04. What the distiller is allowed to see about outcomes.
#:
#:   mode_level  only `pass_rates.json`. The paper says `A` reads trajectory files and
#:               mode-level pass rates, so this is the reproduction default.
#:   per_task    per-episode rewards left in place.
REWARD_VISIBILITY_MODE_LEVEL = "mode_level"
REWARD_VISIBILITY_PER_TASK = "per_task"

#: Stripped from every materialized episode under `mode_level`.
#:
#: `reward` is the field ALG-007 Step 2 names. `success` goes with it because it is the
#: same fact as a boolean: leaving it would let the distiller reconstruct per-task
#: outcomes exactly, which is the thing the switch exists to prevent. Recorded as ASM-008.
#:
#: Deliberately NOT stripped: `termination`, `steps_used`, `step_cap_hit`. Those describe
#: how an episode ended rather than whether it scored, and the method is failure-derived,
#: so removing them would gut the corpus to protect a distinction they do not carry.
_PER_TASK_OUTCOME_FIELDS = ("reward", "success")


def strip_per_task_rewards(payload: dict[str, Any]) -> dict[str, Any]:
    """ALG-007 Step 2. Remove per-task reward fields from a materialized episode."""
    outcome = payload.get("outcome")
    if isinstance(outcome, dict):
        payload["outcome"] = {
            key: value for key, value in outcome.items() if key not in _PER_TASK_OUTCOME_FIELDS
        }
    return payload


def trajectory_bytes(
    trajectory: Trajectory, reward_visibility: str = REWARD_VISIBILITY_PER_TASK
) -> bytes:
    """Byte-stable serialization of one trajectory.

    Sorted keys and a fixed separator. The Merkle root is taken over these bytes, so any
    instability here would make a corpus non-reproducible. Stripping rewards therefore
    changes the Merkle root, which is correct: a corpus the distiller sees differently is
    a different corpus and must not share an address with one it sees in full.
    """
    payload = json.loads(trajectory.model_dump_json())
    if reward_visibility == REWARD_VISIBILITY_MODE_LEVEL:
        payload = strip_per_task_rewards(payload)
    return (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def materialize(
    destination: Path,
    manifest: CorpusManifest,
    trajectories_by_mode: dict[str, list[Trajectory]],
    pass_rates: dict[str, Any],
    *,
    analyzer_lib_source: Path | None = None,
    precomputed: dict[str, Any] | None = None,
    reward_visibility: str = REWARD_VISIBILITY_MODE_LEVEL,
) -> str:
    """Write the spec Section 10.5 layout and return the Merkle root.

    ALG-007 Step 2, `reward_visibility`: under `mode_level` (the reproduction default,
    GAP-04) per-task reward fields are stripped from every materialized episode, leaving
    only `pass_rates.json`. The stripping happens HERE rather than in the orchestrator so
    that no code path can hand the distiller a corpus the switch never touched.

    ALG-007 Step 3, `precomputed`: omitted entirely when None, which removes
    `analysis/precomputed/`. Spec Section 10.5 requires it off in the reproduction path,
    where the paper's agent writes and runs its own analysis code; providing precomputed
    reports there would change the method rather than implement it.
    """
    if reward_visibility not in {REWARD_VISIBILITY_MODE_LEVEL, REWARD_VISIBILITY_PER_TASK}:
        raise ValueError(
            f"unknown reward_visibility {reward_visibility!r}; "
            f"expected {REWARD_VISIBILITY_MODE_LEVEL!r} or {REWARD_VISIBILITY_PER_TASK!r}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    hashes: list[str] = []

    for mode in sorted(trajectories_by_mode):
        mode_dir = destination / "trajectories" / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        for trajectory in sorted(trajectories_by_mode[mode], key=lambda t: t.task_id):
            data = trajectory_bytes(trajectory, reward_visibility)
            (mode_dir / f"{trajectory.task_id}.json").write_bytes(data)
            hashes.append(content_sha256(data))

    pass_rates_bytes = (
        json.dumps(pass_rates, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    (destination / "pass_rates.json").write_bytes(pass_rates_bytes)
    hashes.append(content_sha256(pass_rates_bytes))

    (destination / "README_FOR_DISTILLER.md").write_text(README_FOR_DISTILLER, encoding="utf-8")

    analysis_dir = destination / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    lib_dir = analysis_dir / "lib"
    if analyzer_lib_source is not None and analyzer_lib_source.is_dir():
        shutil.copytree(analyzer_lib_source, lib_dir, dirs_exist_ok=True)
    else:
        lib_dir.mkdir(parents=True, exist_ok=True)

    if precomputed is not None:
        precomputed_dir = analysis_dir / "precomputed"
        precomputed_dir.mkdir(parents=True, exist_ok=True)
        for name in sorted(precomputed):
            body = (
                json.dumps(precomputed[name], sort_keys=True, indent=2, ensure_ascii=False) + "\n"
            )
            (precomputed_dir / f"{name}.json").write_text(body, encoding="utf-8")

    root = merkle_root(hashes)
    final_manifest = manifest.model_copy(update={"merkle_root": root})
    (destination / "MANIFEST.json").write_text(
        final_manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return root


def mark_write_once(destination: Path) -> None:
    """ALG-001 Step 10.

    On a real object store this is a bucket policy. On a local filesystem the closest
    honest equivalent is clearing the write bit, and it is deliberately best-effort: a
    filesystem permission is not a durability guarantee and pretending otherwise would be
    worse than not trying.
    """
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            # Best-effort: a filesystem permission is not a durability guarantee, and on
            # some platforms chmod is a no-op. Failing the build over it would be worse.
            with contextlib.suppress(OSError):
                path.chmod(0o444)
