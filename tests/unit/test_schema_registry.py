"""TASK-009 acceptance tests.

Criteria:
  * The trajectory schema is published and versioned as `trajectory/1.0`.
  * A compatibility test fails on a breaking change without a version bump.

The second criterion is the whole point, so most of this file exercises the
classifier itself against synthetic old/new pairs. A compatibility checker that is only
ever run against a tree that already passes is a checker nobody has tested.
"""

from __future__ import annotations

from typing import Any

import pytest

from psd.core.schemas.registry import (
    PUBLISHED,
    breaking_changes,
    generate,
    load_published,
    schema_filename,
    serialize,
)

TRAJECTORY = "trajectory/1.0"


# ---------------------------------------------------------------------------
# The schema is published, versioned, and byte-stable
# ---------------------------------------------------------------------------


def test_trajectory_schema_is_published() -> None:
    schema = load_published(TRAJECTORY)
    assert schema["title"] == "Trajectory"
    assert "properties" in schema


def test_published_schema_matches_the_models() -> None:
    """Golden drift check.

    If this fails, run `python scripts/regenerate_schemas.py`, then READ the diff before
    committing it. If the diff is breaking, bump the envelope version instead.
    """
    assert serialize(generate(TRAJECTORY)) == serialize(load_published(TRAJECTORY)), (
        "the published schema no longer matches the models; "
        "run `python scripts/regenerate_schemas.py` and read the diff"
    )


def test_serialization_is_byte_stable() -> None:
    """Same input, byte-identical output. Twice, like the analyzers (FR-011)."""
    assert serialize(generate(TRAJECTORY)) == serialize(generate(TRAJECTORY))


def test_schema_version_literal_matches_the_published_version() -> None:
    """The envelope's own `schema_version` default must name the published schema."""
    schema = load_published(TRAJECTORY)
    declared = schema["properties"]["schema_version"]
    assert declared.get("const") == TRAJECTORY or TRAJECTORY in declared.get("enum", [])


@pytest.mark.parametrize(
    ("version", "expected"),
    [("trajectory/1.0", "trajectory-1.0.json"), ("skill/2.11", "skill-2.11.json")],
)
def test_schema_filename(version: str, expected: str) -> None:
    assert schema_filename(version) == expected


@pytest.mark.parametrize("version", ["trajectory", "/1.0", ""])
def test_bad_schema_version_string_is_rejected(version: str) -> None:
    with pytest.raises(ValueError, match="not a <name>/<version> string"):
        schema_filename(version)


def test_every_published_version_has_a_file() -> None:
    for version in PUBLISHED:
        assert load_published(version)["type"] == "object"


# ---------------------------------------------------------------------------
# The compatibility classifier
# ---------------------------------------------------------------------------


def obj(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
    }


def test_identical_schemas_have_no_breaking_changes() -> None:
    schema = obj({"a": {"type": "string"}}, ["a"])
    assert breaking_changes(schema, schema) == []


def test_removing_a_property_is_breaking() -> None:
    old = obj({"a": {"type": "string"}, "b": {"type": "integer"}}, ["a"])
    new = obj({"a": {"type": "string"}}, ["a"])
    assert breaking_changes(old, new) == ["property removed: b"]


def test_making_an_optional_property_required_is_breaking() -> None:
    old = obj({"a": {"type": "string"}, "b": {"type": "integer"}}, ["a"])
    new = obj({"a": {"type": "string"}, "b": {"type": "integer"}}, ["a", "b"])
    assert breaking_changes(old, new) == ["optional property became required: b"]


def test_adding_a_new_required_property_is_breaking() -> None:
    old = obj({"a": {"type": "string"}}, ["a"])
    new = obj({"a": {"type": "string"}, "b": {"type": "integer"}}, ["a", "b"])
    assert breaking_changes(old, new) == ["new required property: b"]


def test_narrowing_an_enum_is_breaking() -> None:
    old = obj({"mode": {"enum": ["think", "no_think", "hybrid"]}})
    new = obj({"mode": {"enum": ["think", "no_think"]}})
    assert breaking_changes(old, new) == ["enum narrowed at mode: lost ['hybrid']"]


def test_adding_an_optional_property_is_not_breaking() -> None:
    old = obj({"a": {"type": "string"}}, ["a"])
    new = obj({"a": {"type": "string"}, "b": {"type": "integer"}}, ["a"])
    assert breaking_changes(old, new) == []


def test_widening_an_enum_is_not_breaking() -> None:
    old = obj({"mode": {"enum": ["think", "no_think"]}})
    new = obj({"mode": {"enum": ["think", "no_think", "hybrid"]}})
    assert breaking_changes(old, new) == []


def test_description_changes_are_not_breaking() -> None:
    old = obj({"a": {"type": "string", "description": "before"}}, ["a"])
    new = obj({"a": {"type": "string", "description": "after"}}, ["a"])
    assert breaking_changes(old, new) == []


def test_breaking_changes_are_found_inside_nested_objects() -> None:
    old = obj({"outcome": obj({"reward": {"type": "number"}}, ["reward"])})
    new = obj({"outcome": obj({}, [])})
    assert breaking_changes(old, new) == ["property removed: outcome.reward"]


def test_breaking_changes_are_found_through_a_ref() -> None:
    """Nested Pydantic models are emitted as `$ref` into `$defs`, so refs must resolve."""
    old = {
        "type": "object",
        "properties": {"outcome": {"$ref": "#/$defs/Outcome"}},
        "required": ["outcome"],
        "$defs": {"Outcome": obj({"reward": {"type": "number"}}, ["reward"])},
    }
    new = {
        "type": "object",
        "properties": {"outcome": {"$ref": "#/$defs/Outcome"}},
        "required": ["outcome"],
        "$defs": {"Outcome": obj({"success": {"type": "boolean"}}, ["success"])},
    }
    findings = breaking_changes(old, new)
    assert "property removed: outcome.reward" in findings


# ---------------------------------------------------------------------------
# The acceptance criterion, against the real schema
# ---------------------------------------------------------------------------


def test_current_models_are_compatible_with_the_published_version() -> None:
    """TASK-009 acceptance.

    If this fails, the models changed in a way that would reject records other systems
    are already producing. Bump the envelope to trajectory/1.1 and write a migration.
    Do not edit the published 1.0 file to make this pass.
    """
    findings = breaking_changes(load_published(TRAJECTORY), generate(TRAJECTORY))
    assert findings == [], (
        "breaking change to trajectory/1.0 without a version bump:\n  " + "\n  ".join(findings)
    )


def test_the_acceptance_check_would_catch_a_real_breaking_change() -> None:
    """Negative control for the test above.

    Simulates dropping `outcome.reward`, the exact change spec Section 10.3 calls out as
    corrupting every win/loss contrast, and asserts the checker objects.
    """
    published = load_published(TRAJECTORY)
    mutated = {
        **published,
        "$defs": {
            **published["$defs"],
            "Outcome": {
                **published["$defs"]["Outcome"],
                "properties": {
                    k: v
                    for k, v in published["$defs"]["Outcome"]["properties"].items()
                    if k != "reward"
                },
                "required": [r for r in published["$defs"]["Outcome"]["required"] if r != "reward"],
            },
        },
    }
    findings = breaking_changes(published, mutated)
    assert any("outcome.reward" in f for f in findings), (
        f"dropping outcome.reward was not reported as breaking; got {findings}"
    )
