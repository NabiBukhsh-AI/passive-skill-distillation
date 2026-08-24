"""Reproduction driver, tiers R0 through R4 (spec Section 13.1).

Phase 1 requires this to **fail loudly, not silently**, until Phase 5 lands. A
reproduction target that exits 0 while doing nothing is the single most dangerous kind of
green build in this project: it looks like evidence and is not.

Each tier turns on only when the tasks that implement it are done. The tasks are named so
the failure message tells you what is actually missing.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Tier:
    name: str
    scope: str
    blocking_tasks: str
    implemented: bool


TIERS: dict[str, Tier] = {
    "r0": Tier(
        name="R0 Smoke",
        scope="ALFWorld only, 10 train / 10 test tasks, 1 seed, 1 distillation",
        blocking_tasks="TASK-039 (and its chain: TASK-031 .. TASK-038)",
        implemented=False,
    ),
    "r1": Tier(
        name="R1 Core",
        scope="ALFWorld + tau2-retail, full splits, 3 seeds, no-think-only corpus",
        blocking_tasks="TASK-040 (and its chain: TASK-026, TASK-039)",
        implemented=False,
    ),
    "r2": Tier(
        name="R2 Full",
        scope="All four benchmarks, both models, 3 seeds",
        blocking_tasks="TASK-071, TASK-072",
        implemented=False,
    ),
    "r3": Tier(
        name="R3 Ablation",
        scope="R2 plus the paired-corpus arm",
        blocking_tasks="TASK-057 (ablation harness), R2",
        implemented=False,
    ),
    "r4": Tier(
        name="R4 Baseline",
        scope="R3 plus GEPA at a 120 metric-call budget",
        blocking_tasks="FR-058 external-baseline adapter, R3",
        implemented=False,
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tier", choices=sorted(TIERS), help="reproduction tier to run")
    args = parser.parse_args(argv)

    tier = TIERS[args.tier]
    if tier.implemented:  # pragma: no cover - flipped on by the task that lands the tier
        raise NotImplementedError(f"{tier.name} is marked implemented but has no driver wired up.")

    print(
        f"\n{tier.name} is NOT YET RUNNABLE.\n"
        f"\n  scope:    {tier.scope}"
        f"\n  blocked on: {tier.blocking_tasks}\n"
        f"\nThis target fails on purpose. A reproduction command that exits 0 without"
        f"\nrunning anything would look like evidence and would not be. Do not 'fix' this"
        f"\nby making it exit 0; land the blocking tasks.\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
