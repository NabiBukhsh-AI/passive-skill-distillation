# Deviations register

Every deliberate departure from the specification or from the paper. Spec Section 30.1
rule 9 requires four things per entry: what changed, why, which requirement it affects,
and who approved it.

A deviation is not a bug and not an assumption. An assumption fills a silence
(`ASSUMPTIONS.md`); a deviation contradicts something the spec or paper actually states.

---

## DEV-001 Instruction P is our reconstruction, not the paper's text

- **What:** `src/psd/distill/instructions/P_0_1.md` is authored by this build. The paper's
  instruction P is unpublished (GAP-01).
- **Why:** P is the method. Without some P there is no distillation step at all. The
  alternative to writing one is not building the system.
- **Affects:** C1 (the paper's primary contribution), FR-020, FR-023, RR-002, every
  number the platform ever produces. All results are conditional on this text.
- **Mitigation:** P is content-addressed and versioned (TASK-027), its hash is recorded on
  every distillation run, and experiment X-03 bounds instruction sensitivity by distilling
  from three variants of P over one corpus before any production claim.
- **Approved by:** maintainer, 2026-08-24. Mandated by TASK-002.

## DEV-002 `docs/SPEC.md` is deliberately absent from this repository

- **What:** Spec Section 18 lists `docs/SPEC.md` as "this document". This repository does
  not contain it.
- **Why:** The specification is private and lives in the separate private repository. The
  supplied `.gitignore` excludes `.spec`, `.spec/`, and `docs/spec/`, which is an explicit
  instruction that the spec must never enter the public repo. Copying it to `docs/SPEC.md`
  would publish it.
- **Affects:** Spec Section 18 repository layout only. No requirement depends on the file
  being present in this repo.
- **Mitigation:** `.spec` resolves to the private directory for local reads, and every
  docstring cites the spec identifier it implements (for example ALG-005) so the mapping
  survives without the text.
- **Approved by:** maintainer, 2026-08-24.

## DEV-003 `assemble_prompt` does not concatenate tool schemas into the returned prompt

- **What:** ALG-009 Step 3 reads `prefix = prefix + serialize(tool_schemas)`. We return the
  system prompt as `system_prompt + separator + skill` and serialize the tool schemas only
  into `cache_key_prefix_hash`.
- **Why:** Spec Section 5.1 defines injection as pure concatenation at the system-prompt
  boundary and states it "must not touch tool definitions". Providers and every upstream
  benchmark runner pass tools as a separate API parameter. Concatenating them into the
  system prompt would change what the model receives relative to the unmodified upstream
  harness, which is itself a condition-parity violation.
- **Affects:** ALG-009, RR-005, RR-006, TASK-033, and the golden test that pins prompt
  assembly byte-exactly. This is a **RESEARCH BOUNDARY** change.
- **Mitigation:** Section 11.6's prefix-stability intent is preserved, because the cache
  key still covers the serialized schemas with stable ordering, so a schema change still
  invalidates the prefix hash and still trips the TASK-076 cached-fraction alert.
- **Approved by:** maintainer, 2026-08-24, in response to an explicit question. See
  `ASSUMPTIONS.md` ASM-001.

## DEV-004 `.gitignore` re-includes `tests/fixtures/**`

- **What:** Added `!tests/fixtures/**` to the supplied `.gitignore`.
- **Why:** The data rules `trajectories/`, `*.jsonl`, and `*.parquet` matched
  `tests/fixtures/trajectories/` and `tests/fixtures/pii/`, which spec Section 18 requires
  to be committed. Verified with `git check-ignore -v` before and after.
- **Affects:** TASK-012, TASK-018 through TASK-022, TASK-042, TASK-043. Without the
  negation the fixture corpora with hand-computed expected tables cannot be committed, and
  "make check passes on a clean clone" is unreachable.
- **Mitigation:** The negation is scoped to `tests/fixtures/` only. `corpora/`, `data/`,
  and a top-level `trajectories/` remain ignored; re-verified after the change.
- **Approved by:** maintainer, 2026-08-24.

## DEV-005 `.spec` is a Windows directory junction, not a POSIX symlink

- **What:** The build playbook creates `.spec` with `ln -s` or
  `New-Item -ItemType SymbolicLink`. This machine refuses both without Administrator
  rights. `.spec` is a directory junction created with `mklink /J`.
- **Why:** Junctions need no elevation and are transparent to reads, so `CLAUDE.md` stays
  byte-identical and its `.spec/PASSIVE_SKILL_DISTILLATION_SPEC.md` path keeps working.
  The playbook's own fallback was to edit that path instead, which this avoids.
- **Affects:** Local developer setup only. No requirement, no measured behavior.
- **Mitigation:** `.spec` is still matched by the `.spec/` ignore rule, verified with
  `git check-ignore -v`, and `git status --porcelain` shows it is not staged.
  Contributors on macOS or Linux create an ordinary symlink; the path is identical either
  way.
- **Approved by:** maintainer, 2026-08-24.

## DEV-006 The project runs on Python 3.12, not the 3.11.9 that spec Section 13.3 pins

- **What:** `configs/repro/environment.yaml` declares `python: "3.12"` and
  `pyproject.toml` declares `requires-python = ">=3.12,<3.14"`. Spec Section 13.3 pins
  `python: "3.11.9"`.
- **Why:** tau2-bench v1.0.1, the commit TASK-003 pinned, declares
  `requires-python = ">=3.12,<3.14"` and cannot be installed on 3.11.9. Verified by dry
  run, which fails with "tau2==1.0.1 cannot be used". R1, the Stage 5 core reproduction,
  requires tau2-retail, so this was not deferrable to the scale phase. ALFWorld's
  `textworld[pddl]` resolves cleanly on 3.12, so one interpreter serves every benchmark.
- **Affects:** spec Section 13.3 only. Section 13.3 is marked
  `ENGINEERING RECOMMENDATION`, not `PAPER`, so no `RR-xxx` requirement and no measured
  behavior is touched. Nothing in the method depends on the interpreter version.
- **Alternatives rejected:** keeping 3.11.9 and running the tau2 adapter out of process
  behind the `EnvironmentAdapter` port, which is architecturally clean but adds a
  subprocess hop, a second lockfile, and a serialization format to maintain; and pinning
  an older tau2 that supports 3.11, which changes harness behavior relative to current
  upstream and forfeits the v1.0.1 user-simulator defaults that moved GAP-08 off
  `blocked`.
- **Mitigation:** The upper bound `<3.14` is tau2-bench's constraint, not ours, and is
  declared at Stage 1 rather than discovered at Stage 5. The exact patch version is
  recorded in every run manifest, per Section 13.3's determinism block.
- **Approved by:** maintainer, 2026-08-24, in response to an explicit question.
