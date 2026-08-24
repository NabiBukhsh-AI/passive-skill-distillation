# Gap register

Transcribed from spec Section 2.4 (GAP-01 to GAP-15) per TASK-001. This is a living
register, not a snapshot. Every gap carries a status, an owner, the config key that
encodes our chosen default, and a resolution plan.

`scripts/check_gaps.py` parses the table below and fails CI on any empty field, any
unknown status, any missing GAP id, or any `blocked` row without a named blocker.

## Status vocabulary

| Status | Meaning |
|---|---|
| `resolved` | The paper supplies enough to close the question. No further work needed. |
| `pinned-default` | The paper does not supply it. We chose a default, recorded it in `ASSUMPTIONS.md`, and carry it in the named config key. |
| `blocked` | We cannot choose responsibly until an external dependency resolves. The blocker is named. |

## Owner roles

Owners are roles, not people, because this repository currently has one maintainer.
Map each role to a person before the first multi-engineer sprint.

| Role | Scope |
|---|---|
| `method-owner` | Instruction `P`, distillation, anything that changes the measured method |
| `data-owner` | Trajectory schema, corpus assembly, splits, redaction |
| `eval-owner` | Evaluation protocol, statistics, token accounting, seeds |
| `benchmarks-owner` | Upstream harness pinning, adapters, environment availability |
| `cost-owner` | Price book, ledger, amortization economics |

## Register

| ID | Missing item | Status | Owner | Config key | Resolution plan | Blocked by |
|---|---|---|---|---|---|---|
| GAP-01 | Exact text of the fixed distiller instruction P | pinned-default | method-owner | `distill.instruction_version` | Author P/0.1 as an explicit reconstruction (TASK-002) and record it in `DEVIATIONS.md` as ours, not the paper's. Treat P as a content-addressed registry artifact under change control (FR-023). Bound the reproduction gap by running experiment X-03 (three variants of P over one corpus) before any production claim. | - |
| GAP-02 | On-disk format of the corpus handed to the coding agent | pinned-default | data-owner | `corpus.layout_version` | Adopt the layout contract in spec Section 10.5 verbatim. P references that layout by path, so it is a contract and not an implementation detail. TASK-015 asserts it with a byte-for-byte layout snapshot test. | - |
| GAP-03 | Distiller tool set, iteration limit, token budget, stopping condition | pinned-default | method-owner | `distill.budget.wall_clock_seconds`, `distill.budget.max_tool_calls`, `distill.budget.max_cost_usd`, `distill.budget.max_tokens` | Pin the Section 11.2 platform defaults: 45 minutes wall clock, 120 tool calls, 10 USD hard cap. Record actual consumption per run (FR-024, FR-026) and re-baseline the defaults from measured p95 after Stage 4. | - |
| GAP-04 | Whether the distiller saw mode-level pass rates only, or per-task rewards | pinned-default | method-owner | `distill.reward_visibility` | Expose as a switch per FR-022. Reproduction default is `mode_level`, because the paper says A reads trajectory files and mode-level pass rates. TASK-028 proves that no per-task reward field survives materialization under `mode_level`, via a filesystem scan test. | - |
| GAP-05 | System prompts of the four harnesses before skill injection | blocked | benchmarks-owner | `harness.<domain>.system_prompt_sha256` | Take each upstream harness default prompt verbatim at its pinned commit, hash it into every trajectory, and use the hash as a condition-parity key so all three conditions provably share it (RR-005, ALG-010 Step 2). Absolute scores will not be comparable with the paper; orderings will be. | Upstream harness sources, whose prompt text the paper never states and never pins to a commit |
| GAP-06 | Decoding parameters for actor and user simulator | pinned-default | eval-owner | `actor.decoding.temperature`, `actor.decoding.top_p`, `actor.decoding.max_tokens` | Default to temperature 0.0, top_p 1.0, provider-default max_tokens, byte-identical across all conditions, recorded in every run manifest. Run experiment X-02 before freezing, because at temperature 0 three seeds measure only environment variance. | - |
| GAP-07 | Seed protocol: what actually varies across the three seeds | pinned-default | eval-owner | `eval.seed_semantics` | Default: the seed controls task ordering, environment sampling, and any sampled decoding. Resolve properly with experiment X-02, which runs one condition at five seeds under both temperature 0.0 and temperature 0.7. | - |
| GAP-08 | Identity and configuration of the tau-squared user simulator | pinned-default | benchmarks-owner | `harness.tau2.user_simulator.model`, `harness.tau2.user_simulator.prompt_sha256` | TASK-003 found concrete pinnable defaults in `src/tau2/config.py` at tau2-bench v1.0.1: implementation `user_simulator`, model `gpt-4.1-2025-04-14`, temperature 0.0. Pin these, hash the prompt, record both on every episode, and treat any change as a harness version bump that invalidates existing evaluations (TASK-072, FR-045). This does not tell us what the paper used; it makes our own runs internally comparable and makes drift detectable, which is what unblocks the build. | - |
| GAP-09 | Exact held-out task ids for ALFWorld and SSB-Verified | blocked | data-owner | `split.<domain>.sha256` | Create our own split artifact once with `random_once_fixed` at seed 20260801, freeze it, and publish its sha256 so our own runs stay comparable across time. Report absolute numbers as non-comparable with the paper. | The paper samples these once and fixes them but never publishes the ids |
| GAP-10 | SSB-Verified subset definition | pinned-default | benchmarks-owner | `benchmarks.spreadsheetbench.verified_subset` | TASK-003 found the subset published upstream: `data/spreadsheetbench_verified_400.tar.gz` at the pinned commit, 400 expert-annotated instances released 2025/12 with Shortcut.AI. Use that archive, record its blob hash, and treat the 400 as the population. Which 50 train and 50 test instances the paper drew from it stays unpublished and is GAP-09, not this gap. | - |
| GAP-11 | Token accounting definition | pinned-default | eval-owner | `metrics.output_token_definition` | Pin the Section 5.5 definition: output tokens equal visible plus tool-call arguments plus reasoning. Record all three components separately on every step so any other definition can be recomputed after the fact. Missing components are null, never zero (Section 15.4). | - |
| GAP-12 | Cost accounting basis for the 1.28 to 2.44 USD figure | pinned-default | cost-owner | `pricing.price_book_version` | Maintain our own effective-dated price book and attribute every model call to a run and a purpose (NFR-050). Treat the paper's range as an order-of-magnitude acceptance band only, per Section 13.6, never as a target to match. | - |
| GAP-13 | Skill position relative to other system-prompt content, and the separator | pinned-default | method-owner | `prompt.separator`, `prompt.position` | Default separator is two newlines and position is after_system_prompt, logged in every run manifest and returned on the resolve endpoint. This is a RESEARCH BOUNDARY value: changing it requires a golden-test update and a `DEVIATIONS.md` entry. | - |
| GAP-14 | Failure-mode taxonomy behind the 35.9/11.5 and 28.7/5.3 statistics | pinned-default | data-owner | `analysis.profiles.<domain>.taxonomy_version` | Define per-domain detector lists in `src/psd/analysis/profiles/`, versioned and recorded on every report. Our taxonomy reproduces the shape of those statistics (trajectory_rate and share_of_all_errors as first-class outputs of ALG-003), not their values. | - |
| GAP-15 | Whether skills were regenerated per seed or reused across seeds | resolved | eval-owner | `eval.reuse_skill_across_seeds` | The paper states skills were distilled once per model-domain pair, so one skill is reused across all three evaluation seeds. Default true. The consequence is that the paper measures zero distillation variance, which is why experiment X-01 exists. | - |

## Change log

| Date | Change | Cause |
|---|---|---|
| 2026-08-24 | Register created with GAP-01 to GAP-15 from spec Section 2.4. | TASK-001 |
| 2026-08-24 | GAP-08 blocked to pinned-default; GAP-10 blocked to pinned-default. | TASK-003 verified both against upstream at the pinned commits. See `docs/BENCHMARKS.md`. |

## Consequences worth stating plainly

Two gaps remain `blocked`, GAP-05 and GAP-09, and both mean **exact numeric reproduction
of Table 1 is impossible by construction**. That is not a defect in this build.
Reproduction is judged on qualitative agreement against the acceptance bands in spec
Section 13.6, and every report this system emits must say so rather than inviting a
number-to-number comparison that the published details cannot support.

GAP-01 is the highest-severity gap in the register. P is the method. Every result this
platform produces is conditional on a reconstruction of an instruction nobody has seen.
