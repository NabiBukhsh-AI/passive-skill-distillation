# Assumptions register

Every `INFERENCE` this build makes, dated. Spec Section 30.1 rule 10: if the reasoning was
"the paper probably meant X", it is written here in exactly those words. This is the
difference between a reproduction and a rewrite.

Two kinds of entry live here:

- **A-xx**: assumptions the specification itself carries forward (spec Section 5.9).
- **ASM-xxx**: assumptions this build made where the specification left two readings open.

Every entry states where it bites, so that when a number moves, the list of things that
could have moved it is finite and written down.

---

## A-xx: assumptions inherited from the specification (Section 5.9)

### A-01 Tasks are exchangeable within a benchmark split
Dated 2026-08-24. Source: spec Section 5.9.
**Where it bites:** every confidence interval. The paired bootstrap resamples tasks with
replacement, which is only valid if tasks carry no systematic ordering or difficulty
structure that the sample does not represent.

### A-02 Seeds are independent given a task
Dated 2026-08-24. Source: spec Section 5.9.
**Where it bites:** all variance estimates. Note the interaction with GAP-07: at
temperature 0.0 the actor is deterministic, so seed independence is a statement about the
environment and the user simulator, not about the model.

### A-03 Output-token counts are gateway-reported and comparable across modes
Dated 2026-08-24. Source: spec Section 5.9.
**Where it bites:** every economic claim in the system, and the Table 1 reduction column.

### A-04 The harness system prompt is byte-identical across conditions except for the appended skill
Dated 2026-08-24. Source: spec Section 5.9.
**Where it bites:** internal validity of the entire study. This is the one assumption the
build converts into an enforced check rather than trusting: ALG-010 Step 2 aborts the run
on any mismatch.

### A-05 The distiller is stochastic and its variance is non-zero
Dated 2026-08-24. Source: spec Section 5.9.
**Where it bites:** the Section 13.6 ablation design, and every single-distillation
comparison the paper draws in its Table 2.

---

## ASM-xxx: assumptions this build made

### ASM-001 Tool schemas are a separate provider field, not part of the concatenated system prompt
Dated 2026-08-24. Decided with the maintainer on 2026-08-24.

Spec Section 5.1 defines injection as `system_prompt + separator + skill`, and gives the
`inject()` function that does exactly that and nothing else. ALG-009 Step 3 then appends
`serialize(tool_schemas)` to the same string before hashing.

The paper probably meant that the tool schemas form part of the stable cacheable prefix,
which is what Section 11.6 describes, rather than that tool definitions are concatenated
into the system-prompt string. Every provider API and every upstream benchmark runner
passes tools as a separate parameter.

**What we do:** `assemble_prompt` returns the system prompt as exactly
`system_prompt + separator + skill`, and computes `cache_key_prefix_hash` over
`system_prompt + separator + skill + serialize(tool_schemas)` so that Section 11.6's
prefix-stability intent is still captured and still testable.

**Where it bites:** TASK-033, the byte-exact golden test, and therefore the measured
effect of every skill. This is a RESEARCH BOUNDARY decision. See `DEVIATIONS.md` DEV-003.

### ASM-002 A null token component is admitted and flagged; a missing key is quarantined
Dated 2026-08-24. Decided with the maintainer on 2026-08-24.

Spec Section 10.3 makes `tokens.output_*` blocking-required on every step. Spec
Section 15.4 requires the gateway to report `null` rather than `0` when a provider omits a
component, and requires the ledger to exclude that call rather than under-count it. Spec
Section 10.9 sets token-accounting completeness at greater than 99.9% of steps, which is
below 100% and therefore anticipates some incompleteness surviving ingestion.

The paper probably meant nothing about this at all; it is a production concern. The two
spec rules reconcile if "MUST be present" is read as a statement about the key rather than
about the value.

**What we do:** a step object missing the `tokens.output_*` key entirely is a blocking
violation and the trajectory is quarantined. A key present with an explicit `null` value
means present-but-unreported: the trajectory is admitted, marked
`token_accounting_complete = false`, excluded from all economic reporting, and counted
against the Section 10.9 threshold with an alert.

**Where it bites:** TASK-005 (the field is `int | None`, not `int`), TASK-010
(quarantine rules), TASK-031 (gateway null propagation), TASK-059 (ledger exclusion).

### ASM-003 A skill "line" is a newline-delimited line of the raw markdown body
Dated 2026-08-24.

Spec Section 10.7 and ALG-008 Check 1 bound a skill at 30 to 150 lines; TASK-002 requires
P to state 40 to 130. Neither defines whether blank lines, the H1 title, or fenced code
blocks count. Two defensible counters differ by roughly 20% on a real skill, which is
enough to straddle a bound.

**What we do:** count `line_count = len(body.split("\n"))` after stripping trailing
whitespace-only lines at end of file, and after nothing else. Blank lines, the H1 title,
and fenced blocks all count. The skill body is never normalized before counting, because
RR-006 requires it to be stored and served verbatim.

**Where it bites:** TASK-002 (the bound P states), TASK-041 (the bound the linter
enforces), and `Skill.stats.lines` in the artifact.

### ASM-004 A trajectory whose task id is in no split artifact is admitted with `split = "unassigned"`
Dated 2026-08-24.

Spec Section 10.3 makes `split` required and requires it to match the split artifact for
the task id. FR-001 simultaneously mandates a streaming endpoint for live agent traffic,
where trajectories necessarily arrive before any split exists for their tasks.

**What we do:** `split` accepts `train`, `test`, and `unassigned`. A trajectory claiming
`train` or `test` that contradicts the split artifact is a blocking violation and is
quarantined. A trajectory whose task id appears in no split artifact is admitted as
`unassigned`. ALG-001 Step 2 already filters corpus membership to
`split.train_task_ids`, so an `unassigned` trajectory can never reach a corpus, and
ALG-001 Step 5 remains the hard contamination check regardless.

**Where it bites:** TASK-005, TASK-010, TASK-013, TASK-014. Recorded as an open question
against GAP-02's owner rather than treated as settled.

### ASM-005 `n_distill` defaults to 1 in repro mode and 5 in platform mode
Dated 2026-08-24.

RR-015 pins the reproduction path to one distillation per model-domain pair, to match the
paper exactly. Spec Section 5.2 and FR-025 require `n_distill >= 5` for any experiment
comparing distillation configurations, and the Section 16.4 example request uses 5.

**What we do:** `configs/repro/` pins `distill.n_distill = 1`; `configs/platform/` pins
`distill.n_distill = 5`. The value is recorded in every run manifest. A repro run can
never inherit the platform default, because the repro config sets it explicitly rather
than relying on a code-level fallback.

**Where it bites:** TASK-029, and the validity of every Table 2 comparison.

### ASM-006 Bare digit runs are redacted as phone numbers only under a per-domain policy
Dated 2026-08-24.

Spec Section 9 (C-03) requires redaction of phone numbers and simultaneously names
over-redaction as the highest risk in that component, because a distiller cannot find a
pattern in text that has been scrubbed into uniformity. Those two pull in opposite
directions on one specific input: a bare run of 10 to 15 digits.

In a tau2-retail transcript, `8471629503` is an order number, and redacting it destroys
exactly the kind of identifier the distiller reasons about. In a tau2-telecom transcript,
the same string is a subscriber number and must not reach a skill. Nothing in the text
distinguishes them.

**What we do:** the default policy requires punctuation or a leading `+` before it will
claim a digit run as a phone number, so bare runs survive. A `strict_phone` policy adds a
bare-digit-run detector and is enabled per domain in the domain profile. `TELECOM_POLICY`
sets it; `DEFAULT_POLICY` does not.

The cost is explicit and tested: under `strict_phone`, order numbers ARE redacted. That
is why it is a per-domain switch rather than a global default, and why the labeled fixture
set scores each case under the policy it belongs to.

**Where it bites:** TASK-012 recall and precision on the labeled set, TASK-014 (the
manifest records the redaction policy version), and the quality of any skill distilled
from a telecom corpus.

### ASM-007 A task's outcome for sampling is its `no_think` trajectory's outcome
Dated 2026-08-24.

ALG-001 Step 4 offers `stratified_by_outcome` and `failure_weighted` sampling strategies,
both of which need each TASK to have an outcome. Under `paired` composition a task carries
two trajectories that can disagree: the think arm succeeds and the no-think arm fails,
which is precisely the interesting case.

The paper probably meant the outcome of the mode being improved, since the whole method is
derived from where the non-reasoning model fails.

**What we do:** a task counts as successful when its `no_think` trajectory succeeded. When
no `no_think` arm exists, fall back to whether any arm succeeded.

**Where it bites:** ALG-001 Step 4 under both outcome-aware strategies, and therefore the
composition of any corpus not built with `strategy=all`. The reproduction path uses
`all`, so this does not affect R1.

### ASM-008 `share_of_all_errors` is computed against the taxonomy in force
Dated 2026-08-24.

The paper reports that the retail fabricated-argument bug accounts for 94% of observed
TOOL ERRORS. ALG-003 defines `share_of_all_errors` as
`occurrences[e] / total_error_events`, where the denominator is every event the taxonomy
produced. Those coincide only when the taxonomy contains nothing but tool-error detectors.

**What we do:** compute the ratio against whatever taxonomy is in force, and record the
taxonomy version on every report, so the denominator is always stated rather than assumed.
Reproducing the paper's 94% requires restricting the taxonomy to tool errors, which the
fixture test does explicitly.

**Where it bites:** any comparison of our error tables against the paper's percentages.
A report read without its `taxonomy_version` is not interpretable.
