"""Canonical domain models (spec Section 10.3, 10.4, 10.5, 10.6).

Implements TASK-005. Every model is strict and forbids extra fields, because the
trajectory schema is this system's spine: a field that silently passes through is a field
nobody validated.

The blocking validation rules of spec Section 10.3 are enforced here, at the type
boundary, rather than in the normalizer. A record that violates one is not a record to be
repaired; it is quarantined (spec Section 30.1 rule 14: quarantine over defaulting).

Note on nullable token components. `tokens.output_*` keys are REQUIRED but their values
may be `null`. Spec Section 10.3 makes the field blocking-required; spec Section 15.4
requires the gateway to emit `null` rather than `0` when a provider does not report a
component. Those reconcile if "must be present" is read as a statement about the key. See
`docs/ASSUMPTIONS.md` ASM-002. Never widen these to default to zero: a false zero corrupts
every economic number downstream.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

#: Strict everywhere. No coercion, no extra fields, no silent widening.
STRICT = ConfigDict(extra="forbid", strict=True, frozen=False)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonNegInt = Annotated[int, Field(ge=0)]

ActorModeName = Literal["think", "no_think"]
SplitName = Literal["train", "test", "unassigned"]
CorpusComposition = Literal["no_think_only", "paired"]
SampleStrategy = Literal["all", "random_n", "stratified_by_outcome", "failure_weighted"]
SkillState = Literal[
    "draft", "validated", "evaluated", "canary", "active", "deprecated", "quarantined"
]


class StrictModel(BaseModel):
    model_config = STRICT


# ---------------------------------------------------------------------------
# Trajectory (spec Section 10.3)
# ---------------------------------------------------------------------------


class Decoding(StrictModel):
    """Decoding parameters. GAP-06: unstated by the paper, pinned by config."""

    temperature: float
    top_p: float
    max_tokens: int | None = None


class Actor(StrictModel):
    model: str
    mode: ActorModeName
    #: Provider-specific flag, for example {"reasoning_effort": "none"} or
    #: {"enable_thinking": false}. Normalized behind ActorMode (TASK-032); never branch on
    #: model family outside the gateway adapter.
    mode_flag: dict[str, Any] = Field(default_factory=dict)
    decoding: Decoding


class UserSimulator(StrictModel):
    """tau-squared only. GAP-08: pinned to the upstream default, recorded per episode."""

    model: str
    version: str


class Harness(StrictModel):
    name: str
    version: str
    #: Required. Condition-drift detection (ALG-010 Step 2) depends on it entirely.
    system_prompt_sha256: Sha256
    tools_sha256: Sha256 | None = None
    max_steps: int
    user_simulator: UserSimulator | None = None


class Observation(StrictModel):
    kind: str
    text: str
    meta: dict[str, Any] = Field(default_factory=dict)


class StepOutput(StrictModel):
    text: str
    reasoning_text: str | None = None
    reasoning_token_count: NonNegInt | None = None


class Action(StrictModel):
    """One action. Canonicalization to a stable symbol is ALG-002, not this model's job."""

    kind: Literal["tool_call", "text_action", "code_execution", "noop"]
    name: str | None = None
    arguments: dict[str, Any] | None = None
    arguments_raw: str | None = None
    text: str | None = None
    #: Filled by ALG-002. Absent on ingest, present after normalization.
    canonical: str | None = None


class ActionResult(StrictModel):
    status: str
    error_type: str | None = None
    text: str | None = None


class StepTokens(StrictModel):
    """Per-step token accounting.

    Every field is REQUIRED (no default) and nullable. Omitting a key is a blocking
    violation; an explicit null means present-but-unreported. See the module docstring and
    `docs/ASSUMPTIONS.md` ASM-002.
    """

    input_total: NonNegInt | None
    input_cached: NonNegInt | None
    output_visible: NonNegInt | None
    output_tool_args: NonNegInt | None
    output_reasoning: NonNegInt | None

    @property
    def complete(self) -> bool:
        """True when every component was reported, so this step may inform economics."""
        return all(
            value is not None
            for value in (
                self.input_total,
                self.input_cached,
                self.output_visible,
                self.output_tool_args,
                self.output_reasoning,
            )
        )

    @property
    def output_total(self) -> int | None:
        """Spec Section 5.5: visible + tool-call arguments + reasoning.

        Returns None if any component is unreported, never a partial sum. A partial sum
        would under-count, and spec Section 15.4 forbids under-counting.
        """
        parts = (self.output_visible, self.output_tool_args, self.output_reasoning)
        if any(part is None for part in parts):
            return None
        return sum(part for part in parts if part is not None)


class Step(StrictModel):
    index: NonNegInt
    observation: Observation
    output: StepOutput
    action: Action
    result: ActionResult | None = None
    tokens: StepTokens
    latency_ms: NonNegInt | None = None
    cost_usd: float | None = None


class Outcome(StrictModel):
    """Terminal outcome.

    `reward` is required and non-nullable on purpose. A defaulted reward silently corrupts
    every win/loss contrast (ALG-006), and a corrupted contrast is invisible: it produces
    plausible numbers. Spec Section 10.3 makes this blocking.
    """

    reward: float
    success: bool
    termination: str
    steps_used: NonNegInt
    step_cap_hit: bool


class Totals(StrictModel):
    output_tokens: NonNegInt | None
    output_reasoning_tokens: NonNegInt | None
    input_tokens: NonNegInt | None
    input_cached_tokens: NonNegInt | None
    turns: NonNegInt
    cost_usd: float | None = None


class StallRun(StrictModel):
    """One detected stall (ALG-005)."""

    start: NonNegInt
    end: NonNegInt
    action: str
    length: NonNegInt
    period: int | None = None


class Labels(StrictModel):
    error_types: list[str] = Field(default_factory=list)
    stall_runs: list[StallRun] = Field(default_factory=list)
    annotator: str | None = None


class Redaction(StrictModel):
    applied: bool
    policy_version: str
    #: Counts by class only. Never the redacted values (spec Section 9, C-03).
    counts: dict[str, int] = Field(default_factory=dict)


class Provenance(StrictModel):
    source: str
    source_run_id: str | None = None
    content_sha256: Sha256


class Trajectory(StrictModel):
    """One complete episode record. Spec Section 10.3, schema `trajectory/1.0`."""

    schema_version: Literal["trajectory/1.0"] = "trajectory/1.0"
    trajectory_id: str
    tenant_id: str
    domain: str
    task_id: str
    #: `unassigned` covers live traffic arriving before any split exists (ASM-004).
    #: ALG-001 Step 2 filters corpus membership to train ids, so an unassigned trajectory
    #: can never reach a corpus.
    split: SplitName
    actor: Actor
    harness: Harness
    seed: int
    started_at: datetime
    ended_at: datetime
    steps: list[Step]
    outcome: Outcome
    totals: Totals
    labels: Labels = Field(default_factory=Labels)
    redaction: Redaction
    provenance: Provenance
    normalization_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _steps_contiguous_from_zero(self) -> Trajectory:
        """Blocking rule: `steps[].index` MUST be contiguous from 0.

        Gaps break n-gram windows (ALG-004) and stall runs (ALG-005) silently, by making
        adjacent list entries non-adjacent in the episode.
        """
        expected = list(range(len(self.steps)))
        actual = [step.index for step in self.steps]
        if actual != expected:
            raise ValueError(
                f"steps[].index must be contiguous from 0; got {actual!r}, expected {expected!r}"
            )
        return self

    @model_validator(mode="after")
    def _ended_after_started(self) -> Trajectory:
        if self.ended_at < self.started_at:
            raise ValueError("ended_at precedes started_at")
        return self

    @property
    def token_accounting_complete(self) -> bool:
        """Whether this trajectory may inform economic reporting (ASM-002)."""
        return all(step.tokens.complete for step in self.steps)


# ---------------------------------------------------------------------------
# Split artifact (spec Section 10.4)
# ---------------------------------------------------------------------------


class SplitSampling(StrictModel):
    strategy: str
    seed: int
    notes: str | None = None


class SplitCounts(StrictModel):
    train: NonNegInt
    test: NonNegInt


class Split(StrictModel):
    """Immutable train/test task split. Spec Section 10.4, schema `split/1.0`."""

    schema_version: Literal["split/1.0"] = "split/1.0"
    domain: str
    created_at: datetime
    sampling: SplitSampling
    train_task_ids: list[str]
    test_task_ids: list[str]
    counts: SplitCounts
    sha256: Sha256

    @model_validator(mode="after")
    def _splits_are_disjoint(self) -> Split:
        """`T_train` and `T_test` are disjoint (spec Section 5.3, FR-006).

        This is enforced again at the database layer (TASK-013) and again at corpus build
        (ALG-001 Step 5). Three layers is deliberate: leakage into a prompt is invisible
        without a check, and this one is free.
        """
        overlap = sorted(set(self.train_task_ids) & set(self.test_task_ids))
        if overlap:
            raise ValueError(
                f"train and test splits overlap on {len(overlap)} task id(s): "
                f"{overlap[:5]!r}{' ...' if len(overlap) > 5 else ''}"
            )
        return self

    @model_validator(mode="after")
    def _counts_match(self) -> Split:
        if self.counts.train != len(self.train_task_ids):
            raise ValueError(
                f"counts.train={self.counts.train} but train_task_ids has "
                f"{len(self.train_task_ids)} entries"
            )
        if self.counts.test != len(self.test_task_ids):
            raise ValueError(
                f"counts.test={self.counts.test} but test_task_ids has "
                f"{len(self.test_task_ids)} entries"
            )
        return self


# ---------------------------------------------------------------------------
# Corpus snapshot (spec Section 10.5)
# ---------------------------------------------------------------------------


class CorpusManifest(StrictModel):
    """`MANIFEST.json` at the root of a materialized corpus snapshot."""

    schema_version: Literal["corpus/1.0"] = "corpus/1.0"
    corpus_id: str
    domain: str
    actor_model: str
    composition: CorpusComposition
    sample_strategy: SampleStrategy
    sample_size: int | None = None
    seed: int
    split_sha256: Sha256
    #: sha256 over the sorted list of per-file content hashes (ALG-001 Step 8).
    merkle_root: Sha256
    counts: dict[str, int] = Field(default_factory=dict)
    dropped_unpaired: list[str] = Field(default_factory=list)
    #: Non-fatal conditions recorded at build time (ALG-001 edge cases). Carried in the
    #: manifest rather than a side file so they are inside the content address: a corpus
    #: cannot be separated from the caveats it was built with.
    build_warnings: list[str] = Field(default_factory=list)
    analyzer_lib_version: str
    redaction_policy_version: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Skill artifact (spec Section 10.6)
# ---------------------------------------------------------------------------


class SkillKey(StrictModel):
    """Spec Section 8, decision 5: skills are per model and per domain.

    Cross-model transfer is untested by the paper, so the model is part of the identity
    rather than an attribute of it.
    """

    domain: str
    actor_model: str
    actor_mode: ActorModeName
    harness_version: str


class SkillStats(StrictModel):
    lines: NonNegInt
    tokens: NonNegInt
    rules: NonNegInt
    rules_with_citations: NonNegInt


class DistillerIdentity(StrictModel):
    runtime: str
    model: str
    version: str | None = None


class SkillLineage(StrictModel):
    corpus_snapshot_sha256: Sha256
    corpus_composition: CorpusComposition
    instruction_version: str
    instruction_sha256: Sha256
    distiller: DistillerIdentity
    analyzer_lib_version: str
    distill_run_id: str
    distill_index: NonNegInt
    n_distill: int = Field(ge=1)


class SkillValidationRef(StrictModel):
    verdict: Literal["pass", "warn", "block"]
    report_id: str


class SkillEvaluationRef(StrictModel):
    run_id: str
    score: float
    tokens: NonNegInt | None = None


class Skill(StrictModel):
    """A distilled skill. Spec Section 10.6, schema `skill/1.0`."""

    schema_version: Literal["skill/1.0"] = "skill/1.0"
    skill_id: str
    key: SkillKey
    #: Stored and served verbatim. RR-006 forbids modifying, normalizing, compressing, or
    #: truncating this string anywhere in the system.
    body_markdown: str
    stats: SkillStats
    lineage: SkillLineage
    validation: SkillValidationRef | None = None
    evaluation: SkillEvaluationRef | None = None
    state: SkillState
    created_at: datetime


# ---------------------------------------------------------------------------
# Evaluation (spec Section 16.5, FR-052)
# ---------------------------------------------------------------------------


class EpisodeTokens(StrictModel):
    input_total: NonNegInt | None
    input_cached: NonNegInt | None
    output_visible: NonNegInt | None
    output_tool_args: NonNegInt | None
    output_reasoning: NonNegInt | None


class Episode(StrictModel):
    """One evaluation or production episode record (FR-052)."""

    episode_id: str
    run_id: str | None = None
    condition: str
    domain: str
    task_id: str
    seed: int
    actor_model: str
    actor_mode: ActorModeName
    #: Which skill version served this episode, for outcome attribution (FR-065).
    skill_id: str | None = None
    success: bool
    reward: float
    turns: NonNegInt
    tokens: EpisodeTokens
    wall_clock_ms: NonNegInt | None = None
    cost_usd: float | None = None
    model_calls: NonNegInt | None = None


class EvaluationCondition(StrictModel):
    name: str
    mode: ActorModeName
    skill_id: str | None = None


class EvaluationRun(StrictModel):
    """Spec ALG-010. A run is either complete or it is not aggregated (NFR-025)."""

    run_id: str
    domain: str
    actor_model: str
    split: SplitName
    conditions: list[EvaluationCondition]
    seeds: list[int]
    status: Literal["queued", "running", "succeeded", "failed", "incomplete", "aborted"]
    #: Aggregation is refused unless this is True (ALG-010 Step 5). Partial results must
    #: never be promoted.
    complete: bool = False
    created_at: datetime
    #: Set when the run aborted, for example on the condition-parity assertion.
    abort_reason: str | None = None
