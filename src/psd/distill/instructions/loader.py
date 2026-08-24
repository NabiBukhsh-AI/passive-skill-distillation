"""Load versioned instruction `P` artifacts from disk (TASK-002).

`P` is a file, never an f-string in code (FR-023). It is the method itself: spec Section
18.1 marks this directory a boundary, and spec Section 30.1 rule 12 requires an explicit
approval note in the commit for any change here.

This module only reads and content-addresses the files. The registry that records a
version, audits its registration, and pins it to a distillation run is TASK-027.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from psd.core.ports import Instruction

INSTRUCTIONS_DIR = Path(__file__).resolve().parent

#: `P_0_1.md` carries version `P/0.1`.
_FILENAME_RE = re.compile(r"^P_(?P<major>\d+)_(?P<minor>\d+)\.md$")


def content_sha256(text: str) -> str:
    """Hash the instruction body.

    Hashes the exact bytes on disk with no normalization. The instruction is an
    artifact, not a string to be tidied: a whitespace change is a different instruction
    and must produce a different hash.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def version_from_filename(filename: str) -> str:
    match = _FILENAME_RE.match(filename)
    if match is None:
        raise ValueError(
            f"{filename!r} is not a valid instruction filename; expected P_<major>_<minor>.md"
        )
    return f"P/{match['major']}.{match['minor']}"


def load(version: str) -> Instruction:
    """Load one instruction by version string, for example `P/0.1`."""
    for path in available_paths():
        if version_from_filename(path.name) == version:
            text = path.read_text(encoding="utf-8")
            return Instruction(version=version, sha256=content_sha256(text), text=text)
    raise FileNotFoundError(
        f"no instruction {version!r} in {INSTRUCTIONS_DIR}; "
        f"available: {sorted(available_versions())}"
    )


def available_paths() -> list[Path]:
    """Every instruction file, in a stable order."""
    return sorted(p for p in INSTRUCTIONS_DIR.glob("P_*.md") if _FILENAME_RE.match(p.name))


def available_versions() -> list[str]:
    return [version_from_filename(p.name) for p in available_paths()]
