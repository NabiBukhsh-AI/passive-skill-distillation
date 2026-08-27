"""Statistical primitives (TASK-021, TASK-037, spec Section 5.7).

Fisher exact, exact McNemar, and Benjamini-Hochberg arrived with ALG-006 (TASK-021).
Wilson intervals and the seeded paired bootstrap arrived with TASK-037. The guarded
gap-recovery ratio lives in `psd.core.metrics`, next to the other reported quantities.

Everything here is exact rather than asymptotic. Spec Section 5.7 is explicit that at
`n = 120` with `p_hat` near 0.19 the normal approximations are badly calibrated, and the
corpora these run over are 35 to 50 tasks, which is smaller still.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import comb, fsum


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test on the 2x2 table [[a, b], [c, d]].

    Exact, computed by summing hypergeometric probabilities no larger than the observed
    one. No scipy dependency in `core`: this is a handful of binomial coefficients, and
    keeping `core` free of the scientific stack keeps it importable anywhere.
    """
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1, row2 = a + b, c + d
    col1 = a + c

    def probability(x: int) -> float:
        value: float = comb(row1, x) * comb(row2, col1 - x) / comb(n, col1)
        return value

    low = max(0, col1 - row2)
    high = min(row1, col1)
    observed = probability(a)
    # 1e-9 relative tolerance: floating point makes exactly-equal probabilities compare
    # unequal, which would drop the mirrored tail and halve the p-value.
    total = fsum(
        probability(x) for x in range(low, high + 1) if probability(x) <= observed * (1 + 1e-9)
    )
    return min(1.0, total)


def mcnemar_exact(discordant_b: int, discordant_c: int) -> float:
    """Exact two-sided McNemar on the discordant pair counts.

    Under the null the discordant pairs split binomially at p = 0.5, so this is an exact
    binomial test on `b` out of `b + c`. Concordant pairs carry no information about a
    difference and are correctly ignored.
    """
    n = discordant_b + discordant_c
    if n == 0:
        return 1.0
    smaller = min(discordant_b, discordant_c)
    # Exact integer arithmetic, then one division. `2**n` types as Any in typeshed
    # (negative exponents return float), and a shift keeps this exact for any n that a
    # corpus could produce.
    favourable = sum(comb(n, k) for k in range(smaller + 1))
    tail = favourable / (1 << n)
    return min(1.0, 2.0 * tail)


@dataclass(frozen=True)
class BHResult:
    index: int
    p_value: float
    adjusted: float
    significant: bool


def benjamini_hochberg(p_values: list[float], q: float = 0.10) -> list[BHResult]:
    """Benjamini-Hochberg step-up at false discovery rate `q` (spec Section 5.7).

    Returns results in the INPUT order, carrying the original index, so a caller can zip
    them back onto their predicates without re-sorting and losing the correspondence.

    Spec ALG-006's implementation note applies to how these are used: with 35 to 50 tasks
    most predicates will not survive correction, and that is not a bug. Report effect
    sizes with intervals and let the distiller weigh them. Filtering to significance only
    would leave the distiller with nothing to look at.
    """
    if not p_values:
        return []
    m = len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda pair: (pair[1], pair[0]))

    adjusted_by_rank: list[float] = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        _, p = ordered[rank - 1]
        running = min(running, p * m / rank)
        adjusted_by_rank[rank - 1] = running

    results: list[BHResult | None] = [None] * m
    for rank, (original_index, p) in enumerate(ordered, start=1):
        adjusted = adjusted_by_rank[rank - 1]
        results[original_index] = BHResult(
            index=original_index,
            p_value=p,
            adjusted=adjusted,
            significant=adjusted <= q,
        )
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Interval estimation (TASK-037, spec Section 5.7)
# ---------------------------------------------------------------------------

#: Spec Section 5.7 requires Wilson rather than Wald. At n = 120 (40 tasks by 3 seeds)
#: with p_hat near 0.19 (telecom no-think), Wald intervals are badly calibrated: they can
#: extend below zero, which is not a possible success rate.
DEFAULT_Z = 1.959963984540054  # two-sided 95%


@dataclass(frozen=True)
class Interval:
    low: float
    high: float

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    @property
    def excludes_zero(self) -> bool:
        """Whether the whole interval sits strictly above zero.

        This is gate predicate G1 in ALG-011. A point estimate that looks good with an
        interval straddling zero is not evidence, and the gate must not treat it as such.
        """
        return self.low > 0.0


def wilson_interval(successes: int, n: int, z: float = DEFAULT_Z) -> Interval:
    """Wilson score interval (spec Section 5.7).

    center = (p_hat + z^2/(2n)) / (1 + z^2/n)
    half   = (z / (1 + z^2/n)) * sqrt( p_hat*(1-p_hat)/n + z^2/(4n^2) )
    """
    if n <= 0:
        raise ValueError("wilson_interval needs at least one observation")
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} is outside [0, {n}]")

    p_hat = successes / n
    denominator = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denominator
    half = (z / denominator) * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return Interval(max(0.0, center - half), min(1.0, center + half))


def paired_bootstrap_delta(
    baseline_by_task: Mapping[str, Sequence[float]],
    treatment_by_task: Mapping[str, Sequence[float]],
    *,
    resamples: int = 10_000,
    seed: int = 0,
    z_percentiles: tuple[float, float] = (2.5, 97.5),
) -> tuple[float, Interval]:
    """Paired bootstrap over TASKS (spec Section 5.7).

    Conditions run on the same task list, so tasks are the blocking unit and seeds are
    within-task replicates. Resampling episodes instead of tasks would ignore the pairing
    and understate the interval, which is the direction that manufactures false
    confidence.

    Seeded. Spec Section 18.1 forbids unseeded randomness anywhere an analyzer output
    depends on it, and a gate decision certainly qualifies.

    Returns (point estimate of the mean paired difference, confidence interval).
    """
    shared = sorted(set(baseline_by_task) & set(treatment_by_task))
    if not shared:
        raise ValueError("no tasks appear in both conditions; the comparison is not paired")

    missing = sorted((set(baseline_by_task) | set(treatment_by_task)) - set(shared))
    if missing:
        raise ValueError(
            f"{len(missing)} task(s) appear in only one condition: {missing[:5]}"
            f"{' ...' if len(missing) > 5 else ''}. A paired comparison over a partial "
            "overlap silently changes what is being compared."
        )

    def task_delta(task: str) -> float:
        treatment = treatment_by_task[task]
        baseline = baseline_by_task[task]
        return sum(treatment) / len(treatment) - sum(baseline) / len(baseline)

    deltas = [task_delta(task) for task in shared]
    point = sum(deltas) / len(deltas)

    rng = random.Random(seed)
    n = len(deltas)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        means.append(total / n)
    means.sort()

    def percentile(pct: float) -> float:
        if len(means) == 1:
            return means[0]
        position = (pct / 100.0) * (len(means) - 1)
        lower = math.floor(position)
        upper = min(lower + 1, len(means) - 1)
        weight = position - lower
        return means[lower] * (1 - weight) + means[upper] * weight

    return point, Interval(percentile(z_percentiles[0]), percentile(z_percentiles[1]))
