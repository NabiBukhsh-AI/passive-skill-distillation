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

Stage 1 of 10. The reproduction does not run yet, and `make reproduce-r0` fails on purpose
until it does.

| Stage | Scope | State |
|---|---|---|
| 1 | Initialization, architecture enforcement, gap register | in progress |
| 2 to 5 | Data pipeline, analyzers, distillation, reproduction | not started |
| 5 | **Stop gate.** Review R1 results before building the platform. | |
| 6 to 10 | Validation, registry, serving, deployment control, scale | not started |

## Quickstart

```bash
make setup      # locked environment plus git hooks
make check      # lint + typecheck + gap register + architecture rules + tests
```

`make help` lists every target.

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
| `src/psd/core/prompt.py` | Prompt assembly. **RESEARCH BOUNDARY**: changing it changes the science. |
| `src/psd/validate/injection.py` | Injection scanning. **SECURITY BOUNDARY**. |
| `src/psd/distill/instructions/` | Versions of instruction `P`. These files are the method itself. |
| `src/psd/analysis` | Deterministic analyzers. Byte-stable, no wall-clock, no unseeded randomness. |
| `docs/GAPS.md` | What the paper does not specify, and what we do about each item. |
| `docs/ASSUMPTIONS.md` | Every inference this build made, dated. |
| `docs/DEVIATIONS.md` | Every deliberate departure from the specification or paper. |
| `docs/BENCHMARKS.md` | Upstream benchmark pinning, verified against each repository. |

The architecture rules in `.import-linter` are enforced in CI, and
`tests/contract/test_architecture.py` proves they bite by writing deliberate violations
and asserting the linter rejects them.

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
