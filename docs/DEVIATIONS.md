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
