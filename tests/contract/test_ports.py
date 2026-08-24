"""TASK-006 acceptance: a stub implementation of each port satisfies the protocol.

Two levels of assertion, because they catch different mistakes:

  * The module-level annotated assignments (`_: SomePort = Stub()`) are checked by mypy.
    They catch a stub whose signature drifts from the protocol.
  * The runtime `isinstance` checks catch a stub that is missing a member entirely.

Together they mean a port cannot be quietly changed without something failing.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from psd.core.models import Decoding, Episode, EpisodeTokens, Skill, SkillKey, SplitName
from psd.core.ports import (
    Analyzer,
    AnalyzerReport,
    Budget,
    ChatRequest,
    ChatResponse,
    CorpusRef,
    Distiller,
    DistillResult,
    EnvironmentAdapter,
    EpisodeResult,
    Instruction,
    LoadedCorpus,
    ModelGateway,
    PolicyConfig,
    Principal,
    RawTrajectory,
    SkillId,
    SkillStore,
    TaskId,
    TokenAccounting,
    TrajectorySource,
)

ZERO_HASH = "0" * 64


class StubTrajectorySource:
    def iter_trajectories(self, ref: CorpusRef) -> Iterator[RawTrajectory]:
        yield RawTrajectory(
            source_format="harness_run",
            harness_version="test",
            body={},
            content_sha256=ZERO_HASH,
        )


class StubEnvironmentAdapter:
    domain = "alfworld"

    def list_tasks(self, split: SplitName) -> list[TaskId]:
        return [TaskId("alfworld_test_0001")]

    def run_episode(self, task: TaskId, policy: PolicyConfig, seed: int) -> EpisodeResult:
        raise NotImplementedError("stub")

    def system_prompt(self) -> str:
        return "You are a helpful household agent."

    def max_steps(self) -> int:
        return 40


class StubModelGateway:
    def complete(self, req: ChatRequest) -> ChatResponse:
        return ChatResponse(
            text="ok",
            tokens=TokenAccounting(
                input_uncached=1,
                input_cached=0,
                output_visible=1,
                output_tool_args=0,
                output_reasoning=None,
            ),
            resolved_model_version="stub-model-2026-01-01",
        )


class StubDistiller:
    def distill(self, corpus_dir: Path, instruction: Instruction, budget: Budget) -> DistillResult:
        return DistillResult(
            skill_markdown="# Rules\n",
            transcript_uri="memory://transcript",
            cost_usd=0.0,
            wall_clock_seconds=0.0,
            termination_reason="agent_completed",
        )


class StubSkillStore:
    def put(self, skill: Skill) -> SkillId:
        return SkillId(skill.skill_id)

    def resolve(self, key: SkillKey) -> Skill | None:
        return None

    def activate(self, key: SkillKey, skill_id: SkillId, actor: Principal) -> None:
        return None


class StubAnalyzer:
    name = "stub"

    def run(self, corpus: LoadedCorpus, config: Mapping[str, Any]) -> AnalyzerReport:
        return AnalyzerReport(
            analyzer=self.name,
            analyzer_lib_version="0.0.0",
            corpus_merkle_root=ZERO_HASH,
            rows=[],
        )


# Checked by mypy: a stub whose signature drifts from its protocol fails here.
_source: TrajectorySource = StubTrajectorySource()
_adapter: EnvironmentAdapter = StubEnvironmentAdapter()
_gateway: ModelGateway = StubModelGateway()
_distiller: Distiller = StubDistiller()
_store: SkillStore = StubSkillStore()
_analyzer: Analyzer = StubAnalyzer()


def test_every_port_has_a_conforming_stub() -> None:
    assert isinstance(StubTrajectorySource(), TrajectorySource)
    assert isinstance(StubEnvironmentAdapter(), EnvironmentAdapter)
    assert isinstance(StubModelGateway(), ModelGateway)
    assert isinstance(StubDistiller(), Distiller)
    assert isinstance(StubSkillStore(), SkillStore)
    assert isinstance(StubAnalyzer(), Analyzer)


def test_there_are_exactly_six_ports() -> None:
    """Spec Section 8.4: six ports define the system; everything else is an adapter.

    A seventh port is an architectural decision, not a refactor. If this fails, that
    conversation has not happened yet.
    """
    import psd.core.ports as ports

    protocols = {
        name
        for name in dir(ports)
        if isinstance(getattr(ports, name), type)
        and getattr(getattr(ports, name), "_is_protocol", False)
        # Defined here, not merely imported: `typing.Protocol` itself is in scope.
        and getattr(ports, name).__module__ == ports.__name__
    }
    assert protocols == {
        "TrajectorySource",
        "EnvironmentAdapter",
        "ModelGateway",
        "Distiller",
        "SkillStore",
        "Analyzer",
    }, f"port set changed: {sorted(protocols)}"


def test_distiller_port_cannot_receive_a_model_gateway() -> None:
    """RR-004: `A` has no environment access.

    The distiller signature takes a directory, an instruction, and a budget. Handing it a
    gateway would have to be a visible edit to this signature rather than an accident, and
    this test is what makes it visible.
    """
    import inspect

    params = inspect.signature(StubDistiller.distill).parameters
    assert set(params) == {"self", "corpus_dir", "instruction", "budget"}


def test_a_stub_missing_a_member_does_not_satisfy_its_port() -> None:
    """Negative control. Without this, the isinstance assertions prove very little."""

    class Incomplete:
        def put(self, skill: Skill) -> SkillId:
            return SkillId("x")

        # resolve and activate deliberately absent

    assert not isinstance(Incomplete(), SkillStore)


def test_token_accounting_uses_null_not_zero_for_unreported_components() -> None:
    """Spec Section 15.4. Zero would be a measurement; null is an absence."""
    accounting = TokenAccounting(
        input_uncached=10,
        input_cached=0,
        output_visible=5,
        output_tool_args=0,
        output_reasoning=None,
    )
    assert accounting.output_reasoning is None


def test_policy_config_defaults_to_the_documented_separator() -> None:
    """GAP-13: the separator is unknown; the default is two newlines and it is logged."""
    policy = PolicyConfig(
        actor_model="gpt-5.4-mini",
        mode="no_think",
        system_prompt="sys",
        decoding=Decoding(temperature=0.0, top_p=1.0, max_tokens=None),
    )
    assert policy.separator == "\n\n"
    assert policy.skill is None


def _minimum_episode() -> Episode:
    """The smallest episode record FR-052 accepts."""
    return Episode(
        episode_id="ep_1",
        condition="no_think",
        domain="alfworld",
        task_id="alfworld_test_0001",
        seed=1,
        actor_model="gpt-5.4-mini",
        actor_mode="no_think",
        success=True,
        reward=1.0,
        turns=12,
        tokens=EpisodeTokens(
            input_total=100,
            input_cached=80,
            output_visible=50,
            output_tool_args=10,
            output_reasoning=0,
        ),
    )


def test_minimum_episode_record_validates() -> None:
    assert _minimum_episode().success is True
