"""TASK-007: prove the architecture rules bite.

Acceptance criterion is "a deliberately violating import fails CI". Asserting that
`lint-imports` passes on a clean tree does not demonstrate that, because a misconfigured
contract also passes on a clean tree. So each test here *writes* a violating import into
the source tree, runs the linter, and asserts it fails, then removes the file again.

The fixture is written and removed inside the test rather than committed, because a
committed violating module would break `make imports` for everyone.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / ".import-linter"


def run_import_linter() -> subprocess.CompletedProcess[str]:
    """Invoke import-linter in a subprocess and return its real exit status.

    Two traps, both of which silently exit 0 and would make every assertion below pass
    vacuously:

      * `python -m importlinter.cli` is a click group that does nothing.
      * `importlinter.cli.lint_imports` is a plain function returning an exit status, not
        a click command. Calling it bare reads no configuration and returns success.

    So we call the function with the config filename and propagate its return value.
    `test_clean_tree_passes` additionally asserts that contracts were actually analyzed,
    so a misread config surfaces as a failure rather than as a green run.
    """
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from importlinter.cli import lint_imports;"
            " sys.exit(lint_imports(sys.argv[1]))",
            CONFIG.name,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


#: Writes a module at a path under src/psd and returns where it landed.
WriteViolation = Callable[[str, str], Path]


@pytest.fixture
def violating_module() -> Iterator[WriteViolation]:
    """Write a module containing a forbidden import, then always clean it up."""
    written: list[Path] = []

    def _write(relative_path: str, body: str) -> Path:
        path = REPO_ROOT / "src" / "psd" / relative_path
        path.write_text(body, encoding="utf-8")
        written.append(path)
        return path

    yield _write

    for path in written:
        path.unlink(missing_ok=True)


def test_clean_tree_passes() -> None:
    """Baseline. If this fails, the later assertions prove nothing."""
    result = run_import_linter()
    assert result.returncode == 0, (
        "lint-imports failed on a clean tree, so the violation tests below are "
        f"meaningless.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_contracts_are_actually_evaluated() -> None:
    """Guard against a vacuous suite.

    import-linter exits 0 when it cannot read a configuration file, printing "Could not
    read any configuration". Every violation test below would then pass for the wrong
    reason. Assert the report says contracts were kept, and how many.
    """
    result = run_import_linter()
    assert "Could not read any configuration" not in result.stdout, (
        f"import-linter read no config, so nothing was checked.\n{result.stdout}"
    )
    assert "Contracts: 4 kept, 0 broken." in result.stdout, (
        f"expected all four architecture contracts to be evaluated and kept; got:\n{result.stdout}"
    )


def test_core_importing_an_adapter_fails(violating_module: WriteViolation) -> None:
    """core imports nothing from psd.* (spec Section 18.1)."""
    violating_module(
        "core/_arch_violation.py",
        "from psd.adapters import alfworld  # noqa: F401\n",
    )
    result = run_import_linter()
    assert result.returncode != 0, (
        "core imported an adapter and the linter did not object. "
        "The core-is-pure contract is not biting."
    )
    assert "core imports nothing" in result.stdout


def test_adapter_importing_eval_fails(violating_module: WriteViolation) -> None:
    """adapters may not import from eval or distill (spec Section 18.1)."""
    violating_module(
        "adapters/_arch_violation.py",
        "from psd import eval  # noqa: F401\n",
    )
    result = run_import_linter()
    assert result.returncode != 0, "an adapter imported psd.eval and the linter did not object."
    assert "adapters import neither eval nor distill" in result.stdout


def test_serving_importing_distill_fails(violating_module: WriteViolation) -> None:
    """serving is the hot path; it may not import distill or analysis."""
    violating_module(
        "serving/_arch_violation.py",
        "from psd import distill  # noqa: F401\n",
    )
    result = run_import_linter()
    assert result.returncode != 0, "serving imported psd.distill and the linter did not object."
    assert "serving imports neither distill nor analysis" in result.stdout


def test_adapters_importing_each_other_fails(violating_module: WriteViolation) -> None:
    """Adapters never import each other (spec Section 8.4)."""
    violating_module(
        "adapters/alfworld/_arch_violation.py",
        "from psd.adapters import tau2  # noqa: F401\n",
    )
    result = run_import_linter()
    assert result.returncode != 0, "one adapter imported another and the linter did not object."
    assert "adapters never import each other" in result.stdout
