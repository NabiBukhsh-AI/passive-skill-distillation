"""The `psd` command line interface (TASK-008).

Every command from spec Section 13.5 exists here as a typed stub. Each stub exits
non-zero with a message naming the task that implements it, because a subcommand that
prints nothing and exits 0 is indistinguishable from one that worked.

Spec Section 30.1 rule 14: prefer failing loudly to degrading silently.
"""

from __future__ import annotations

import sys
from typing import Annotated, NoReturn

import typer

app = typer.Typer(
    name="psd",
    help=(
        "Passive Skill Distillation Platform. Turns agent trajectory logs into "
        "validated, versioned system-prompt skills. No model training anywhere."
    ),
    no_args_is_help=True,
    add_completion=False,
)

split_app = typer.Typer(help="Immutable train/test task splits.", no_args_is_help=True)
corpus_app = typer.Typer(help="Corpus snapshots.", no_args_is_help=True)
skill_app = typer.Typer(help="Skill validation and registry.", no_args_is_help=True)
report_app = typer.Typer(help="Reproduction reports.", no_args_is_help=True)

app.add_typer(split_app, name="split")
app.add_typer(corpus_app, name="corpus")
app.add_typer(skill_app, name="skill")
app.add_typer(report_app, name="report")


def not_implemented(command: str, task: str, spec_ref: str) -> NoReturn:
    """Exit non-zero, naming what would implement this command."""
    typer.secho(f"`psd {command}` is not implemented yet.", fg=typer.colors.RED, err=True)
    typer.secho(f"  implemented by: {task}", err=True)
    typer.secho(f"  specified in:   {spec_ref}", err=True)
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


@split_app.command("create")
def split_create(
    domain: Annotated[str, typer.Option(help="Domain id, for example alfworld.")],
    train: Annotated[int, typer.Option(help="Number of training tasks.")],
    test: Annotated[int, typer.Option(help="Number of held-out tasks.")],
    strategy: Annotated[str, typer.Option(help="Sampling strategy.")] = "random_once_fixed",
    seed: Annotated[int, typer.Option(help="Sampling seed. Recorded in the artifact.")] = 20260801,
) -> None:
    """Create an immutable, content-addressed split artifact."""
    not_implemented("split create", "TASK-013", "spec Section 10.4, FR-006")


@split_app.command("import")
def split_import(
    domain: Annotated[str, typer.Option(help="Domain id.")],
    test_from_upstream: Annotated[
        bool, typer.Option("--test-from-upstream", help="Use the benchmark's own test split.")
    ] = False,
    train: Annotated[int, typer.Option(help="Number of training tasks to draw.")] = 50,
) -> None:
    """Import a split whose held-out half is provided by the benchmark."""
    not_implemented("split import", "TASK-013", "spec Section 10.4, RR-009")


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------


@app.command("rollout")
def rollout(
    domain: Annotated[str, typer.Option(help="Domain id.")],
    model: Annotated[str, typer.Option(help="Actor model.")],
    mode: Annotated[str, typer.Option(help="think or no_think.")],
    split: Annotated[str, typer.Option(help="train or test.")] = "train",
    seeds: Annotated[int, typer.Option(help="Number of seeds.")] = 1,
) -> None:
    """Collect trajectories. These are ordinary evaluation rollouts, not extra work.

    RR-002: no environment rollouts are collected for the purpose of distillation. This
    command produces the corpus as a by-product of ordinary evaluation on the training
    split, which is the whole point of calling the method passive.
    """
    not_implemented("rollout", "TASK-035 (per adapter)", "spec Section 13.5 Phase B")


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------


@corpus_app.command("build")
def corpus_build(
    domain: Annotated[str, typer.Option(help="Domain id.")],
    model: Annotated[str, typer.Option(help="Actor model the corpus was collected from.")],
    composition: Annotated[str, typer.Option(help="no_think_only or paired.")] = "no_think_only",
    strategy: Annotated[str, typer.Option(help="Sampling strategy.")] = "all",
    sample_size: Annotated[int | None, typer.Option(help="Task count for random_n.")] = None,
    seed: Annotated[int, typer.Option(help="Sampling seed.")] = 0,
) -> None:
    """Build an immutable corpus snapshot, aborting on test-split contamination."""
    not_implemented("corpus build", "TASK-014", "ALG-001, spec Section 10.5")


@corpus_app.command("analyze")
def corpus_analyze(
    corpus: Annotated[str, typer.Option(help="Corpus id or Merkle root.")],
) -> None:
    """Run the deterministic analyzer library over a snapshot."""
    not_implemented("corpus analyze", "TASK-022", "ALG-003 through ALG-006")


# ---------------------------------------------------------------------------
# distill
# ---------------------------------------------------------------------------


@app.command("distill")
def distill(
    corpus: Annotated[str, typer.Option(help="Corpus Merkle root.")],
    instruction: Annotated[str, typer.Option(help="Instruction version, for example P/0.1.")],
    distiller: Annotated[str, typer.Option(help="Distiller runtime.")] = "claude_code",
    n: Annotated[int, typer.Option(help="Independent distillations to run.")] = 1,
    precomputed_analysis: Annotated[
        bool,
        typer.Option(
            "--precomputed-analysis/--no-precomputed-analysis",
            help="Repro default is false: the agent writes its own analysis code.",
        ),
    ] = False,
) -> None:
    """Run the coding agent over a corpus under isolation and budget."""
    not_implemented("distill", "TASK-026", "ALG-007, spec Section 11.2")


# ---------------------------------------------------------------------------
# skill
# ---------------------------------------------------------------------------


@skill_app.command("validate")
def skill_validate(
    skill: Annotated[str, typer.Option(help="Skill id.")],
) -> None:
    """Lint, injection scan, leakage audit, and PII scan. Blocking for registration."""
    not_implemented("skill validate", "TASK-041 through TASK-044", "ALG-008")


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


@app.command("eval")
def evaluate(
    domain: Annotated[str, typer.Option(help="Domain id.")],
    model: Annotated[str, typer.Option(help="Actor model.")],
    conditions: Annotated[
        str, typer.Option(help="Comma-separated: think,no_think,no_think_plus_skill.")
    ] = "think,no_think,no_think_plus_skill",
    skill: Annotated[str | None, typer.Option(help="Skill id for the skill condition.")] = None,
    seeds: Annotated[int, typer.Option(help="Evaluation seeds. Reproduction default 3.")] = 3,
) -> None:
    """Run the three-condition evaluation on the held-out split.

    Aborts if the harness prompt, tool schemas, or decoding differ across conditions
    (ALG-010 Step 2). That assertion is what makes the comparison mean anything.
    """
    not_implemented("eval", "TASK-036", "ALG-010, RR-005, RR-011")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@report_app.command("table1")
def report_table1(
    domains: Annotated[str, typer.Option(help="Comma-separated domain ids.")],
    models: Annotated[str, typer.Option(help="Comma-separated actor models.")],
) -> None:
    """Reproduce the paper's Table 1 layout from completed evaluation runs."""
    not_implemented("report table1", "TASK-038", "spec Section 2.2, Section 22.1")


@report_app.command("figure1")
def report_figure1(
    out: Annotated[str, typer.Option(help="Output path for the plot.")],
) -> None:
    """Reproduce the accuracy-versus-tokens Pareto plot (FR-055)."""
    not_implemented("report figure1", "TASK-038", "spec Section 22.2")


def main() -> int:  # pragma: no cover - console-script shim
    app()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
