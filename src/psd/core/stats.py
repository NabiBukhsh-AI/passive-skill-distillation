"""Statistical primitives (TASK-021 partial, spec Section 5.7).

Fisher exact, exact McNemar, and Benjamini-Hochberg land here now because ALG-006 needs
them. Wilson intervals, the seeded paired bootstrap, and the guarded gap-recovery ratio
are TASK-037 and are deliberately not written yet.

Everything here is exact rather than asymptotic. Spec Section 5.7 is explicit that at
`n = 120` with `p_hat` near 0.19 the normal approximations are badly calibrated, and the
corpora these run over are 35 to 50 tasks, which is smaller still.
"""

from __future__ import annotations

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
