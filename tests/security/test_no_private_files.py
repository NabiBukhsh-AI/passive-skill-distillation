"""Blocking security suite: the specification must never enter this public repository.

This repository is public and sits beside a private directory holding the engineering
specification and the build playbook. A spec pushed to a public remote cannot be
meaningfully un-pushed, so this is a one-way failure and belongs in the blocking suite
rather than in a lint.

Spec Section 21.10 and TASK-069: these tests are blocking in CI and may never be skipped
or marked xfail.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_checker() -> ModuleType:
    """Load scripts/check_no_private_files.py, which is a script rather than a package."""
    path = REPO_ROOT / "scripts" / "check_no_private_files.py"
    spec = importlib.util.spec_from_file_location("check_no_private_files", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BLOCKED_PATHS = [
    ".spec",
    ".spec/PASSIVE_SKILL_DISTILLATION_SPEC.md",
    "docs/PASSIVE_SKILL_DISTILLATION_SPEC.md",
    "PASSIVE_SKILL_DISTILLATION_SPEC.md",
    "Files/BUILD_PLAYBOOK.md",
    "docs/spec/anything.md",
    "notes/design.private.md",
]

ALLOWED_PATHS = [
    "src/psd/core/models.py",
    "docs/GAPS.md",
    "docs/BENCHMARKS.md",
    "README.md",
    "tests/fixtures/trajectories/spec_section_10_3_example.json",
    "docs/adr/0001-something.md",
    "scripts/check_gaps.py",
]


@pytest.mark.security
@pytest.mark.parametrize("path", BLOCKED_PATHS)
def test_private_paths_are_blocked(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    checker = load_checker()
    monkeypatch.setattr(checker, "staged_paths", lambda: [path])
    assert checker.main() == 1, f"{path!r} was allowed into a public commit"


@pytest.mark.security
@pytest.mark.parametrize("path", ALLOWED_PATHS)
def test_ordinary_paths_are_allowed(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    """Negative control.

    A hook that blocks everything is not a working hook, it is a broken commit workflow
    that people will disable. Ordinary source must pass.
    """
    checker = load_checker()
    monkeypatch.setattr(checker, "staged_paths", lambda: [path])
    assert checker.main() == 0, f"{path!r} was wrongly blocked"


@pytest.mark.security
def test_a_mixed_commit_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """One private file among many ordinary ones still blocks the whole commit."""
    checker = load_checker()
    monkeypatch.setattr(
        checker,
        "staged_paths",
        lambda: ["src/psd/core/models.py", "README.md", ".spec/SPEC.md"],
    )
    assert checker.main() == 1


@pytest.mark.security
def test_spec_link_is_gitignored() -> None:
    """The `.spec` link must be ignored, not merely absent from the last commit."""
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "-v", ".spec"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        ".spec is not gitignored. It links to the private specification directory and "
        "would be committed to a public repository."
    )


@pytest.mark.security
def test_spec_link_is_not_tracked() -> None:
    """Belt and braces: nothing under .spec is in the index."""
    import subprocess

    result = subprocess.run(
        ["git", "ls-files", "--", ".spec"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout.strip() == "", f"tracked files under .spec:\n{result.stdout}"
