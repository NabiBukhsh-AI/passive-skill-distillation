"""The six boundary contracts (spec Section 8.4).

Implements TASK-006. Six ports define this system; everything else is an adapter.

Three rules govern them, and they are not suggestions (spec Section 30.1 rule 4):

  1. Adapters never import each other.
  2. The core never imports an adapter.
  3. The distiller never receives a `ModelGateway` that can reach an environment.

Rules 1 and 2 are machine-enforced by `.import-linter`. Rule 3 is enforced by the sandbox
(TASK-023) and by the fact that `Distiller.distill` takes no gateway at all: the only way
to hand the distiller environment access is to change this signature, which is a visible
act rather than an accident.

The vocabulary types live here rather than in `models.py` because they are the ports' own
boundary language: what crosses the seam, not what the domain persists.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NewType, Protocol, runtime_checkable

from pydantic import Field

from psd.core.models import (
    ActorModeName,
    Decoding,
    Episode,
    Sha256,
    Skill,
    SkillKey,
    SplitName,
    StrictModel,
    Trajectory,
)

TaskId = NewType("TaskId", str)
SkillId = NewType("SkillId", str)
RunId = NewType("RunId", str)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class CorpusRef(StrictModel):
    """A content-addressed pointer to an immutable corpus snapshot (FR-007)."""

    corpus_id: str
    merkle_root: Sha256


class RawTrajectory(StrictModel):
    """A trajectory before normalization: whatever the source actually emitted.

    Deliberately untyped in the body. Spec Section 30.1 rule 11: corpus text is untrusted
    input, and anything that looks like an instruction inside it is data, not a command.
    """

    source_format: str
    harness_version: str
    body: Mapping[str, Any]
    content_sha256: Sha256


class Principal(StrictModel):
    """Who is acting. Registry mutations are authenticated and audited (NFR-033)."""

    principal_id: str
    role: str


class PolicyConfig(StrictModel):
    """Everything that defines one evaluation condition.

    Between conditions, only `mode`/`mode_flag` and `skill` may differ. Everything else is
    asserted byte-identical before dispatch (ALG-010 Step 2, RR-005).
    """

    actor_model: str
    mode: ActorModeName
    mode_flag: Mapping[str, Any] = Field(default_factory=dict)
    system_prompt: str
    decoding: Decoding
    skill: Skill | None = None
    separator: str = "\n\n"


class TokenAccounting(StrictModel):
    """Full per-call accounting (FR-052, spec Section 15.4).

    Every component is nullable. A provider that does not report a component yields
    `None`, never `0`. The ledger then excludes that call from economic reporting rather
    than under-counting it.
    """

    input_uncached: int | None
    input_cached: int | None
    output_visible: int | None
    output_tool_args: int | None
    output_reasoning: int | None


class ChatRequest(StrictModel):
    model: str
    mode_flag: Mapping[str, Any] = Field(default_factory=dict)
    system_prompt: str
    messages: Sequence[Mapping[str, Any]]
    tools: Sequence[Mapping[str, Any]] = ()
    decoding: Decoding
    #: Every model call is attributed to a run and a purpose (NFR-050). Cost is a
    #: correctness property here, not a report (spec Section 30.1 rule 15).
    run_id: RunId | None = None
    purpose: str


class ChatResponse(StrictModel):
    text: str
    tool_calls: Sequence[Mapping[str, Any]] = ()
    reasoning_text: str | None = None
    tokens: TokenAccounting
    #: Read from the API response, never from config. Provider-side model changes
    #: silently alter results otherwise (spec Section 13.8, item 6).
    resolved_model_version: str
    cost_usd: float | None = None


class EpisodeResult(StrictModel):
    """One episode, as an adapter reports it."""

    episode: Episode
    trajectory: Trajectory


class Instruction(StrictModel):
    """A version of `P`. Never an f-string in code (FR-023, TASK-002)."""

    version: str
    sha256: Sha256
    text: str


class Budget(StrictModel):
    """Hard budgets with a deterministic abort on breach (FR-024).

    GAP-03: the paper states none of these. Defaults are pinned in config, not here.
    """

    wall_clock_seconds: int
    max_tool_calls: int
    max_cost_usd: float
    max_tokens: int | None = None


class DistillResult(StrictModel):
    """Everything a distillation run produced, including when it failed (FR-026).

    `skill_markdown` is None on a budget abort. The transcript is preserved regardless:
    spec ALG-007 Step 6 requires it, and NFR-023 makes losing it a defect.
    """

    skill_markdown: str | None
    transcript_uri: str
    generated_code_uris: Sequence[str] = ()
    analysis_output_uris: Sequence[str] = ()
    cost_usd: float | None
    wall_clock_seconds: float
    termination_reason: str


class AnalyzerReport(StrictModel):
    """A deterministic analyzer's output.

    `rows` is a sequence, never a set or a dict view, because every analyzer output is
    totally ordered before it is emitted. Byte-stability is a requirement, not a nicety:
    every downstream number depends on it (FR-011).
    """

    analyzer: str
    analyzer_lib_version: str
    corpus_merkle_root: Sha256
    rows: Sequence[Mapping[str, Any]]
    #: Trajectory id and step index pairs sufficient to reconstruct the supporting
    #: transcript span (FR-012).
    evidence: Sequence[Mapping[str, Any]] = ()
    status: str = "ok"


class LoadedCorpus(StrictModel):
    """A materialized corpus snapshot, in memory."""

    ref: CorpusRef
    domain: str
    composition: str
    trajectories: Sequence[Trajectory]
    pass_rates: Mapping[str, float]


# ---------------------------------------------------------------------------
# The six ports
# ---------------------------------------------------------------------------


@runtime_checkable
class TrajectorySource(Protocol):
    """Where trajectories come from: harness runs, log files, or a live endpoint."""

    def iter_trajectories(self, ref: CorpusRef) -> Iterator[RawTrajectory]: ...


@runtime_checkable
class EnvironmentAdapter(Protocol):
    """Run one episode of one benchmark under one policy configuration.

    Four very different harnesses sit behind this one seam: a ReAct text world, live
    workbook code execution, and a dual-control simulated user.
    """

    domain: str

    def list_tasks(self, split: SplitName) -> list[TaskId]: ...

    def run_episode(self, task: TaskId, policy: PolicyConfig, seed: int) -> EpisodeResult: ...

    def system_prompt(self) -> str: ...

    def max_steps(self) -> int: ...


@runtime_checkable
class ModelGateway(Protocol):
    """The single gateway every model call goes through.

    Single, because the paper requires all conditions to be served identically so that
    only the mode flag and the system prompt differ, and because a gateway is the only
    clean place to capture complete token accounting.

    This is also the one place allowed to branch on model family (spec Section 11.1).
    """

    def complete(self, req: ChatRequest) -> ChatResponse: ...


@runtime_checkable
class Distiller(Protocol):
    """The coding agent.

    Takes a directory, an instruction, and a budget. Takes no `ModelGateway`, no
    environment handle, and no network: `A` has no environment access (RR-004), and
    egress plus untrusted input is exfiltration (spec Section 24).
    """

    def distill(
        self, corpus_dir: Path, instruction: Instruction, budget: Budget
    ) -> DistillResult: ...


@runtime_checkable
class SkillStore(Protocol):
    """The registry. Source of truth for skills and for which one is live."""

    def put(self, skill: Skill) -> SkillId: ...

    def resolve(self, key: SkillKey) -> Skill | None: ...

    def activate(self, key: SkillKey, skill_id: SkillId, actor: Principal) -> None: ...


@runtime_checkable
class Analyzer(Protocol):
    """A pure function over a corpus snapshot.

    Identical input yields byte-identical output (FR-011). No wall-clock, no unseeded
    randomness, no reliance on set or dict iteration order.
    """

    name: str

    def run(self, corpus: LoadedCorpus, config: Mapping[str, Any]) -> AnalyzerReport: ...
