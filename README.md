# passive-skill-distillation

Turns agent trajectory logs into validated, evaluated, versioned natural-language skills
that are appended to a non-reasoning model's system prompt.

**There is no model training anywhere in this system.** No gradient step exists in the
critical path. That is the central architectural fact, and it is what makes the platform
cheap enough to run continuously. It also relocates the primary risk: the artifact being
shipped is untrusted-derived text landing in a privileged prompt position, so prompt
supply-chain security is a first-class concern rather than an afterthought.

Implements *Reason Wide, Not Deep: Amortizing the Reasoning Premium into Distilled Skills*
(arXiv:2608.07885v1).

## Status

Stages 1 to 3 are complete. **The reproduction does not run yet**, and `make reproduce-r0`
exits non-zero on purpose until it does: a reproduction command that exits 0 without
running anything looks like evidence and is not.

| Stage | Scope | State |
|---|---|---|
| 1 | Initialization, architecture enforcement, gap register | complete |
| 2 | Data pipeline: schema, normalizer, redaction, splits, corpus snapshots | complete |
| 3 | Analyzer library: canonicalization, error frequency, n-grams, stalls, contrast | complete |
| 4 | Distillation: sandbox, budgets, distiller adapters | instruction registry done; **sandbox blocked** |
| 5 | Evaluation and reproduction, R0 then R1 | not started |
| 5 | **Stop gate.** R1 results reviewed before any platform work begins. | |
| 6 to 10 | Validation, registry, serving, deployment control, scale | not started |

Stage 4 is blocked on a container runtime. The distiller sandbox (TASK-023) has to prove
that an HTTP request and a DNS lookup from inside it both fail before any distiller code
is written, and that is not something a mock can demonstrate. Memory caps and filesystem
restriction have non-container substitutes; **network egress denial does not**, and it is
the control that matters most, because egress plus untrusted input is exfiltration.

## Requirements

- **Python 3.12.** Not 3.11: tau2-bench v1.0.1 declares `requires-python >=3.12,<3.14`,
  and the R1 reproduction needs tau2-retail. Recorded as DEV-006.
- A container runtime, from Stage 4 onward. Not needed for anything currently built.
- No database server to install. The integration suite starts a real PostgreSQL 16 from
  the `pgserver` wheel, with no elevation and no container.

## Quickstart

```bash
make setup            # locked environment plus git hooks
make check            # lint + typecheck + gap register + architecture rules + tests
make test-integration # database-backed suites; starts a real PostgreSQL
make test-security    # blocking suite, never skipped
```

`make help` lists every target.

Current state of the suites:

| Suite | Tests | Covers |
|---|---|---|
| `make check` | 373 | Unit, property, contract, plus lint, mypy, gap register, import rules |
| `make test-integration` | 55 | Ingestion and registry against real PostgreSQL 16.2 |
| `make test-security` | 17 | Blocking; may never be skipped or xfailed |

## How it works

1. **Collect.** A corpus of 35 to 50 tasks worth of trajectories that a training split
   already produced. No new rollouts are collected for distillation.
2. **Distill.** A coding agent reads the corpus directory under one fixed instruction,
   writes and runs its own analysis code (failure frequencies, action n-grams, stall
   detection, win/loss contrasts), and emits 40 to 130 lines of markdown.
3. **Inject.** That markdown is appended verbatim to the non-reasoning model's system
   prompt. Nothing else changes: not the harness, not the tools, not the decoding.

The skill occupies a cacheable prompt prefix, so the recurring reasoning-token premium is
replaced by a one-time cost plus a cheap per-call prefix read.

## Two build targets in one repository

| Mode | Purpose | Rule |
|---|---|---|
| `repro` | Reconstruct the paper's experiment as faithfully as published details allow | Frozen once validated. Nothing that changes measured behavior may leak in. |
| `platform` | Run the amortization loop continuously against real traffic | Optimized for reliability, cost, and safety. |

Conflating them is the single most likely way this project fails. Shared code lives in
`src/psd/core`.

## Repository map

| Path | Contents |
|---|---|
| `src/psd/core` | Pure domain logic and the six ports. Imports nothing from `psd.*`. mypy strict. |
| `src/psd/core/models.py` | The canonical `Trajectory` and every other versioned envelope. |
| `src/psd/core/canonicalize.py` | ALG-002. Maps an action to a stable symbol so n-grams mean something. |
| `src/psd/ingest` | Normalization, redaction, quality checks, the batch endpoint. |
| `src/psd/corpus` | Splits, corpus assembly (ALG-001), snapshot materialization. |
| `src/psd/analysis` | Deterministic analyzers. Byte-stable, no wall-clock, no unseeded randomness. |
| `src/psd/distill/instructions/` | Versions of instruction `P`. These files are the method itself. |
| `migrations/` | Plain SQL, applied in filename order. The schema is the contract. |
| `docs/GAPS.md` | What the paper does not specify, and what we do about each item. |
| `docs/ASSUMPTIONS.md` | Every inference this build made, dated. |
| `docs/DEVIATIONS.md` | Every deliberate departure from the specification or paper. |
| `docs/BENCHMARKS.md` | Upstream benchmark pinning, verified against each repository. |

Two boundary files are named in the architecture but **not written yet**, so do not go
looking for them: `src/psd/core/prompt.py` (prompt assembly, RESEARCH BOUNDARY, TASK-033,
Stage 5) and `src/psd/validate/injection.py` (injection scanning, SECURITY BOUNDARY,
TASK-042, Stage 6).

The architecture rules in `.import-linter` are enforced in CI, and
`tests/contract/test_architecture.py` proves they bite by writing deliberate violations
and asserting the linter rejects them. Two ways of invoking import-linter silently exit 0
having checked nothing, so there is also a test asserting the report says four contracts
were evaluated.

## Failures with no symptom

Most of the unusual machinery here exists for one reason: this system's worst failures do
not announce themselves. Each of these produces plausible numbers and a green build.

| Failure | Why it is invisible | What refuses it |
|---|---|---|
| A missing reward defaulted to `0` | Indistinguishable from a genuinely failed episode, and it corrupts every win/loss contrast | Quarantine at the type boundary and in the database (`reward` is `NOT NULL`) |
| A test task in a training corpus | A contaminated skill scores **better** on held-out tasks, so it reads as success | Model validator, a plpgsql trigger, and a hard abort that writes nothing |
| Redaction that scrubs text into uniformity | Trivially safe, and it deletes the exact failure the skill is compiled to prevent | Typed placeholders, stable per value within an episode |
| A duplicated trajectory | Inflates a failure rate by exactly the duplication rate | Idempotency on content hash, checked before normalization |
| A re-pointed instruction version | Every lineage record still names `P/0.1` and looks correct | Content addressing, plus immutability enforced by the database |
| A missing token count read as `0` | Silently under-counts, and every economic claim rests on that number | `null` never `0`, and partial sums refused outright |

The same instinct applies to the process. `docs/GAPS.md`, `docs/ASSUMPTIONS.md`, and
`docs/DEVIATIONS.md` are checked into the repository and validated in CI, because the
difference between a reproduction and a rewrite is whether the substitutions were written
down.

## What this repository does not contain

The engineering specification is private and lives in a separate repository. It is
reachable locally through the gitignored `.spec` link and is never committed here. A
pre-commit hook blocks it from being staged.

## Reading the results honestly

Exact numeric reproduction of the paper's Table 1 is impossible by construction: the
instruction `P`, the harness system prompts, and the held-out task ids are all unpublished
(`docs/GAPS.md`, GAP-01, GAP-05, GAP-09). Reproduction is judged on qualitative agreement
against stated acceptance bands. Any report this system emits says so rather than inviting
a number-to-number comparison the published details cannot support.
