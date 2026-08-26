"""TASK-027 acceptance: instructions are content-addressed, audited, and immutable.

The acceptance criterion is "a distillation run records the exact instruction hash used",
so the tests that matter are the ones proving the hash is enough to recover the text byte
for byte, and that a registered version cannot be quietly re-pointed at different text.

That second one is the failure with no symptom: re-pointing `P/0.1` would invalidate
every skill distilled from the original while every lineage record still cheerfully
names `P/0.1`.
"""

from __future__ import annotations

from typing import Any

import psycopg
import pytest

from psd.core.ports import Instruction
from psd.distill.instructions.loader import content_sha256, load
from psd.distill.instructions.store import InstructionConflictError, InstructionRegistry

pytestmark = pytest.mark.integration


def registry(connection: Any) -> InstructionRegistry:
    return InstructionRegistry(connection, tenant_id="t_test")


def instruction(version: str = "P/0.1", text: str = "# Task\n\nDo the thing.\n") -> Instruction:
    return Instruction(version=version, sha256=content_sha256(text), text=text)


# ---------------------------------------------------------------------------
# Registration and recovery
# ---------------------------------------------------------------------------


def test_registering_returns_a_content_addressed_id(connection: Any) -> None:
    instruction_id = registry(connection).register(instruction(), principal="maintainer")
    assert instruction_id.startswith("ins_")


def test_the_registered_text_is_recoverable_byte_for_byte(connection: Any) -> None:
    """TASK-027 acceptance, the part that makes a run reproducible."""
    original = instruction(text="# Task\n\nTrailing spaces matter:   \n\n")
    registry(connection).register(original, principal="maintainer")

    recovered = registry(connection).load("P/0.1")
    assert recovered.text == original.text
    assert recovered.sha256 == original.sha256


def test_a_run_can_recover_the_instruction_from_its_hash_alone(connection: Any) -> None:
    """What a distillation run actually stores is a hash. It has to be sufficient."""
    original = instruction()
    registry(connection).register(original, principal="maintainer")

    found = registry(connection).get_by_hash(original.sha256)
    assert found is not None
    assert found["body"] == original.text


def test_registering_the_real_p_0_1(connection: Any) -> None:
    """The instruction the build actually uses, end to end through the registry."""
    p = load("P/0.1")
    registry(connection).register(p, principal="maintainer", notes="reconstruction, DEV-001")
    assert registry(connection).load("P/0.1").text == p.text


def test_registration_is_idempotent_on_identical_content(connection: Any) -> None:
    first = registry(connection).register(instruction(), principal="maintainer")
    second = registry(connection).register(instruction(), principal="maintainer")
    assert first == second


# ---------------------------------------------------------------------------
# Immutability: the failure with no symptom
# ---------------------------------------------------------------------------


def test_repointing_a_version_at_different_text_is_refused(connection: Any) -> None:
    registry(connection).register(instruction(text="original\n"), principal="maintainer")
    with pytest.raises(InstructionConflictError, match="immutable"):
        registry(connection).register(
            instruction(text="something else entirely\n"), principal="maintainer"
        )


def test_even_a_whitespace_change_is_a_different_instruction(connection: Any) -> None:
    """A trailing space changes the bytes the distiller receives, so it is a new version."""
    registry(connection).register(instruction(text="body\n"), principal="maintainer")
    with pytest.raises(InstructionConflictError):
        registry(connection).register(instruction(text="body \n"), principal="maintainer")


def test_the_database_refuses_to_update_a_registered_instruction(connection: Any) -> None:
    """Defence in depth: the application refuses, and so does the database."""
    registry(connection).register(instruction(), principal="maintainer")
    with pytest.raises(psycopg.errors.RaiseException, match="immutable"):
        connection.execute("UPDATE instructions SET body = 'tampered' WHERE version = 'P/0.1'")


def test_the_database_refuses_to_delete_a_registered_instruction(connection: Any) -> None:
    registry(connection).register(instruction(), principal="maintainer")
    with pytest.raises(psycopg.errors.RaiseException, match="immutable"):
        connection.execute("DELETE FROM instructions WHERE version = 'P/0.1'")


def test_a_mismatched_hash_is_refused_before_it_is_stored(connection: Any) -> None:
    """Guards against a caller that computed the hash over something else."""
    bad = Instruction(version="P/9.9", sha256="0" * 64, text="real body\n")
    with pytest.raises(ValueError, match="hashes to"):
        registry(connection).register(bad, principal="maintainer")


def test_two_versions_may_not_share_content(connection: Any) -> None:
    """Identical text under two version numbers would make lineage ambiguous."""
    registry(connection).register(instruction(version="P/0.1"), principal="maintainer")
    with pytest.raises(psycopg.errors.UniqueViolation):
        registry(connection).register(instruction(version="P/0.2"), principal="maintainer")


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def test_registration_writes_an_audit_record(connection: Any) -> None:
    instruction_id = registry(connection).register(
        instruction(), principal="maintainer", notes="first draft", request_id="req_1"
    )
    records = registry(connection).audit_records(instruction_id)
    assert len(records) == 1
    assert records[0]["principal"] == "maintainer"
    assert records[0]["action"] == "instruction.register"
    assert records[0]["after"]["sha256"] == instruction().sha256
    assert records[0]["request_id"] == "req_1"


def test_an_idempotent_re_registration_does_not_duplicate_the_audit_record(
    connection: Any,
) -> None:
    instruction_id = registry(connection).register(instruction(), principal="maintainer")
    registry(connection).register(instruction(), principal="someone_else")
    assert len(registry(connection).audit_records(instruction_id)) == 1


def test_versions_are_listed_in_order(connection: Any) -> None:
    reg = registry(connection)
    reg.register(instruction(version="P/0.2", text="two\n"), principal="m")
    reg.register(instruction(version="P/0.1", text="one\n"), principal="m")
    assert reg.versions() == ["P/0.1", "P/0.2"]


def test_loading_an_unregistered_version_fails_loudly(connection: Any) -> None:
    with pytest.raises(KeyError, match=r"P/9\.9"):
        registry(connection).load("P/9.9")
