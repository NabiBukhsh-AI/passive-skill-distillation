"""TASK-008 acceptance tests.

Criteria:
  * `psd --help` lists every command from spec Section 13.5.
  * Each stub exits non-zero with a clear "not implemented" message.

The second criterion matters more than it looks. A stub that exits 0 makes
`make reproduce-r0` look like it worked, and a reproduction that looks like it worked is
worse than one that plainly did not.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from psd.cli.main import app

runner = CliRunner()

#: Every command line that appears in spec Section 13.5.
SPEC_13_5_COMMANDS = [
    ("split", "create"),
    ("split", "import"),
    ("rollout",),
    ("corpus", "build"),
    ("distill",),
    ("skill", "validate"),
    ("eval",),
    ("report", "table1"),
    ("report", "figure1"),
]


def test_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


@pytest.mark.parametrize("group", ["split", "corpus", "skill", "report"])
def test_top_level_help_lists_every_group(group: str) -> None:
    result = runner.invoke(app, ["--help"])
    assert group in result.output


@pytest.mark.parametrize("command", ["rollout", "distill", "eval"])
def test_top_level_help_lists_every_bare_command(command: str) -> None:
    result = runner.invoke(app, ["--help"])
    assert command in result.output


@pytest.mark.parametrize("path", SPEC_13_5_COMMANDS, ids=lambda p: " ".join(p))
def test_every_spec_command_exists(path: tuple[str, ...]) -> None:
    """A missing command shows up here as a non-zero exit from --help."""
    result = runner.invoke(app, [*path, "--help"])
    assert result.exit_code == 0, f"`psd {' '.join(path)} --help` failed:\n{result.output}"


REQUIRED_ARGS: dict[tuple[str, ...], list[str]] = {
    ("split", "create"): ["--domain", "alfworld", "--train", "50", "--test", "50"],
    ("split", "import"): ["--domain", "tau2_retail"],
    ("rollout",): ["--domain", "alfworld", "--model", "m", "--mode", "no_think"],
    ("corpus", "build"): ["--domain", "alfworld", "--model", "m"],
    ("distill",): ["--corpus", "abc", "--instruction", "P/0.1"],
    ("skill", "validate"): ["--skill", "skl_1"],
    ("eval",): ["--domain", "alfworld", "--model", "m"],
    ("report", "table1"): ["--domains", "alfworld", "--models", "m"],
    ("report", "figure1"): ["--out", "artifacts/figure1.png"],
}


@pytest.mark.parametrize("path", SPEC_13_5_COMMANDS, ids=lambda p: " ".join(p))
def test_every_stub_exits_non_zero(path: tuple[str, ...]) -> None:
    result = runner.invoke(app, [*path, *REQUIRED_ARGS[path]])
    assert result.exit_code != 0, (
        f"`psd {' '.join(path)}` exited 0 while unimplemented. A stub that exits 0 is "
        "indistinguishable from one that worked."
    )


@pytest.mark.parametrize("path", SPEC_13_5_COMMANDS, ids=lambda p: " ".join(p))
def test_every_stub_names_its_implementing_task(path: tuple[str, ...]) -> None:
    result = runner.invoke(app, [*path, *REQUIRED_ARGS[path]])
    output = result.output + (result.stderr or "")
    assert "not implemented" in output
    assert "TASK-" in output, "the message should name the task that implements it"


def test_bare_invocation_shows_help_rather_than_a_traceback() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output
