# Benchmark availability and pinning report

TASK-003. Every claim below was verified against the upstream repository at the pinned
commit on 2026-08-24, not recalled. Verification method is stated per claim so a reviewer
can repeat it.

Spec Section 13.3 requires every benchmark to be pinned by commit, never by tag or version
range, because benchmark drift is the most common cause of unreproducible agentic results.
The commit SHAs below are what `configs/repro/environment.yaml` carries.

---

## Summary

| Benchmark | Status | Pin | Blocks |
|---|---|---|---|
| ALFWorld | **runnable** | `1558ba46d078279ecb4c5d33a6cdffc96714a2d2` (tag 0.4.2) | Nothing. Assets need a download step. |
| SpreadsheetBench (SSB-Verified) | **runnable, with caveats** | `49b73a94775fb489063f60ca1865e3a650079a79` (branch `main`) | Upstream `requirements.txt` is internally unsatisfiable. Do not use it verbatim. |
| tau2-bench | **runnable on 3.12** | `fc0055dc4e0a316c3f83133267fbd6faaa770992` (tag v1.0.1, peeled) | Resolved: the project moved to Python 3.12 (DEV-006). |

Two gaps improved as a direct result of this investigation. See "Gap register changes" at
the end.

---

## ALFWorld

- **Repository:** `https://github.com/alfworld/alfworld`
- **Pinned commit:** `1558ba46d078279ecb4c5d33a6cdffc96714a2d2`
- **Corresponds to:** tag `0.4.2`. Recorded for humans only; the SHA is the pin.
- **Branch head at time of writing:** `master` = `aaba6870f86c5be6a08a491f32a50b906227bc3e`, deliberately not used.

### Installability

Verified with `uv pip install --dry-run "textworld[pddl]>=1.6.1"` against a Python 3.11.9
virtual environment: **resolves cleanly**. Also resolves cleanly on Python 3.12.

`setup.py` declares no `python_requires`, so the constraint comes entirely from
`textworld[pddl]>=1.6.1`, which is the sole line in `requirements.txt`.

### Entry points

From `setup.py`, four console scripts are installed:

| Script | Use here |
|---|---|
| `alfworld-download` | **Required.** Fetches game assets. |
| `alfworld-generate` | Not used. |
| `alfworld-play-tw` | Manual TextWorld play, useful for a by-hand smoke check. |
| `alfworld-play-thor` | Not used. We are text-only, per RR-012. |

`scripts/run_eval.py` is the upstream evaluation driver. Our adapter (TASK-035) does not
call it; it drives the environment directly through `alfworld.env`, because the harness
loop, step cap, and trajectory emission are ours to control (RR-012, RR-005).

### Required assets

`scripts/alfworld-download` pulls from GitHub release artifacts and unpacks into the path
named by `ALFWORLD_DATA`:

- `json_2.1.1_json.zip` and `json_2.1.1_pddl.zip` from release `0.2.2`
- `json_2.1.2_tw-pddl.zip` from release `0.4.0`

The MaskRCNN checkpoint, pretrained checkpoints, and seq2seq data are for the vision and
training paths and are **not** needed. `ALFWORLD_DATA` must be set in the reproduction
environment and recorded in the run manifest.

### Step cap: upstream default disagrees with the paper

`configs/base_config.yaml` and `configs/eval_config.yaml` both set
`max_nb_steps_per_episode: 50`. RR-012 and spec Section 5.3 pin the paper's cap at **40**.

**Consequence for TASK-035:** the adapter must set the cap to 40 explicitly rather than
inheriting the upstream default, and must record the configured value on every trajectory.
Inheriting 50 silently would change both the success rate and the mean turn count, and
would do so invisibly. This is exactly the drift spec Section 13.3 warns about.

### Metric

Win rate, that is the terminal success indicator, per spec Section 5.4.

---

## SpreadsheetBench and the SSB-Verified subset

- **Repository:** `https://github.com/RUCKBReasoning/SpreadsheetBench`
- **Pinned commit:** `49b73a94775fb489063f60ca1865e3a650079a79`
- **Tags:** none exist upstream. `main` is the only branch, so the SHA is the only possible pin.

### GAP-10 is substantially resolved: the Verified subset is published

The repository ships `data/spreadsheetbench_verified_400.tar.gz`:

```
100644 blob b6baaf0f23ef5adc1cd22078f4eeb3102b4a7c72  14958255  data/spreadsheetbench_verified_400.tar.gz
```

The README records: "[2025/12] We are releasing SpreadsheetBench Verified, an expert
annotated subset of 400 instances. This benchmark was developed in collaboration with
Shortcut.AI (Fundamental Research Labs)."

So "the verified subset" that the paper cites without defining **does have a published
definition**: 400 expert-annotated instances, distributed both in this repository and on
Hugging Face. GAP-10 moves from `blocked` to `pinned-default`.

What remains unresolved is which 50 train and 50 test instances the paper drew from those
400, which is GAP-09 and is unpublished for this benchmark just as it is for ALFWorld.

The other two archives are the full benchmark (`spreadsheetbench_912_v0.1.tar.gz`, 912
instances) and a 200-instance sample. Neither is what the paper used.

### Installability: upstream `requirements.txt` is broken

Verified with `uv pip install --dry-run -r requirements.txt` on Python 3.11.9:

```
Because pandas==2.2.0 depends on numpy{python_full_version < '3.12'}>=1.23.2,<2
and you require pandas==2.2.0, we can conclude that you require numpy>=1.23.2,<2.
And because you require numpy==1.22.2, we can conclude that your requirements
are unsatisfiable.
```

This is an internal contradiction in the upstream pin file (`numpy==1.22.2` against
`pandas==2.2.0`), not a Python-version problem. It is unsatisfiable on any interpreter.

**Consequence for TASK-071:** do not install upstream `requirements.txt`. Resolve our own
set. RR-013 names only what actually matters, namely `openpyxl` and `pandas` for the code
execution path. `vllm` and `transformers` in that file exist for their self-hosted
inference path, which we do not use because model access goes through our single gateway.

### Entry points and evaluation

- `inference/inference_multiple.py` plus `inference/scripts/inference_multiple_react_exec.sh`
  implement the multi-round ReAct plus code-execution setting, which is the setting RR-013
  describes.
- `evaluation/evaluation.py` implements the OJ-style metric.
- `evaluation/open_spreadsheet.py` forces formula recalculation so cached cell values are
  readable by `openpyxl`. **It requires LibreOffice 7.5 or newer** on macOS and Linux, and
  is the default backend on non-Windows platforms. This is a real system dependency for the
  evaluation path and must be in the container image.
- `code_exec_docker/` is upstream's own container for executing model-written code.

### Metric

Modification accuracy, OJ-style: each instruction carries multiple test-case workbooks and
all must pass. Spec Section 5.4 calls this modification accuracy.

---

## tau2-bench (telecom and retail)

- **Repository:** `https://github.com/sierra-research/tau2-bench`
- **Pinned commit:** `fc0055dc4e0a316c3f83133267fbd6faaa770992`
- **Corresponds to:** tag `v1.0.1`, peeled. The tag object is
  `b711c1ead46f55111bf765cf44d5da8bacc2d28c`; the commit it points at is the pin.
- **Branch head at time of writing:** `main` = `a2c024725189473d2d7cea3a5cfdbcc67478e41f`, deliberately not used.

### RESOLVED: Python version conflict with spec Section 13.3

`pyproject.toml` declares:

```toml
requires-python = ">=3.12,<3.14"
```

Spec Section 13.3 pins `python: "3.11.9"`. Verified with
`uv pip install --dry-run ./tau2` against a 3.11.9 environment:

```
Python>=3.12,<3.14 and tau2==1.0.1 depends on Python>=3.12,<3.14, we can
conclude that tau2==1.0.1 cannot be used.
```

Verified against a 3.12 environment: **resolves cleanly**. ALFWorld's `textworld[pddl]`
also resolves cleanly on 3.12, so 3.12 satisfies both.

This is escalated per spec Section 30.4 ("a benchmark's upstream behavior differs from what
Section 13 describes") rather than decided unilaterally. It is not deferrable to Stage 10:
**R1, the Stage 5 core reproduction, requires tau2-retail** (TASK-040), so the conflict
bites at the stop gate, not at the scale phase.

Note also a task-ordering problem in the spec itself: TASK-040 (R1, Stage 5) needs a
working tau2 adapter, but the tau2 adapter is TASK-072 in Phase 10. TASK-040's declared
dependencies are only TASK-039 and TASK-026. Recorded here; raised with the maintainer.

### GAP-08 is substantially resolved: the default user simulator is pinnable

`src/tau2/config.py` at the pinned commit declares:

```python
DEFAULT_USER_IMPLEMENTATION = "user_simulator"
DEFAULT_LLM_USER = "gpt-4.1-2025-04-14"
DEFAULT_LLM_TEMPERATURE_USER = 0.0
DEFAULT_MAX_STEPS = 200
DEFAULT_MAX_ERRORS = 10
DEFAULT_SEED = 300
```

The paper never states that it used the upstream default, so this does not tell us what the
paper did. What it does give us is a **specific, pinnable, recordable** simulator identity
and temperature, which is what GAP-08 actually needs in order to stop being blocked: our
own runs become internally comparable, and any change becomes a detectable harness version
bump. GAP-08 moves from `blocked` to `pinned-default`.

Note that `DEFAULT_MAX_STEPS = 200` differs from the illustrative `"max_steps": 30` in the
spec Section 10.3 example trajectory. The spec example is illustrative, not normative; the
adapter records whatever it actually configures.

### Splits

Each domain ships `data/tau2/domains/<domain>/split_tasks.json` alongside `tasks.json`.
This is the upstream-provided test split that spec Section 13.5's
`psd split import --test-from-upstream` refers to. Both `retail` and `telecom` are present,
as are `airline`, `banking_knowledge`, and `mock`, which we do not use.

### Metric

Pass rate, per spec Section 5.4, computed by `src/tau2/evaluator`.

---

## Gap register changes produced by this task

| Gap | Was | Now | Evidence |
|---|---|---|---|
| GAP-08 | blocked | pinned-default | `src/tau2/config.py` at the pinned commit gives a concrete default simulator model and temperature that we can pin, record per episode, and version-bump on change. |
| GAP-10 | blocked | pinned-default | `data/spreadsheetbench_verified_400.tar.gz` at the pinned commit is a published 400-instance expert-annotated subset. The subset is defined; only the paper's sample from it is not. |

GAP-05 and GAP-09 remain `blocked`, and both are unresolvable in principle from published
material. Exact numeric reproduction of Table 1 stays impossible, as spec Section 13.6
already states.

---

## Decision taken on the interpreter version

**The project runs on Python 3.12** (decided 2026-08-24, recorded as DEV-006).

Two alternatives were considered and rejected: keeping 3.11.9 and running tau2 out of
process behind the `EnvironmentAdapter` port, which is architecturally clean but adds a
subprocess hop, a second lockfile, and a serialization format to maintain; and pinning an
older tau2 that supports 3.11, which changes harness behavior relative to current upstream
and forfeits the v1.0.1 simulator defaults that moved GAP-08 off `blocked`.

`pyproject.toml` declares `requires-python = ">=3.12,<3.14"`. The upper bound belongs to
tau2-bench, not to us: a 3.14 environment cannot install the benchmark R1 depends on, and
a constraint that is discovered at Stage 5 is worse than one declared at Stage 1.
