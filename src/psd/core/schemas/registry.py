"""Versioned JSON Schema registry for the canonical envelopes (TASK-009).

Implements the trajectory-schema half of spec Section 10.8, plus the compatibility
guarantee spec Section 30.1 rule 8 requires: the trajectory schema is one of three things
that may not break without a version bump and a migration.

The published schemas live beside this module as `<name>-<major>.<minor>.json`. They are
generated from the Pydantic models by `scripts/regenerate_schemas.py` and committed, so a
reviewer can see the contract change in a diff rather than having to run code to discover
it.

`breaking_changes` is the part that matters. A golden-file comparison alone catches every
edit, which sounds strict but trains people to regenerate the golden without reading it.
Classifying the diff means an additive change is a one-line regeneration while a breaking
change stops and says so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from psd.core.models import Trajectory

SCHEMA_DIR = Path(__file__).resolve().parent

#: Envelopes with a published JSON Schema. Others (split, corpus, skill) carry a
#: `schema_version` too and can register here when a task needs them; spec Section 10.8
#: names the trajectory schema specifically because it is the system's spine.
PUBLISHED: dict[str, type[BaseModel]] = {"trajectory/1.0": Trajectory}


def schema_filename(schema_version: str) -> str:
    """`trajectory/1.0` becomes `trajectory-1.0.json`."""
    name, _, version = schema_version.partition("/")
    if not name or not version:
        raise ValueError(f"{schema_version!r} is not a <name>/<version> string")
    return f"{name}-{version}.json"


def serialize(schema: dict[str, Any]) -> str:
    """Render a schema byte-stably.

    Sorted keys and a fixed indent, so the committed file is a function of the models
    alone and not of dict insertion order. Byte-stability is the same requirement the
    analyzers carry (FR-011); a schema that reorders itself between runs produces diff
    noise that hides real contract changes.
    """
    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def generate(schema_version: str) -> dict[str, Any]:
    model = PUBLISHED[schema_version]
    return model.model_json_schema(mode="validation")


def load_published(schema_version: str) -> dict[str, Any]:
    path = SCHEMA_DIR / schema_filename(schema_version)
    if not path.is_file():
        raise FileNotFoundError(
            f"no published schema for {schema_version!r} at {path}. "
            "Run `python scripts/regenerate_schemas.py`."
        )
    parsed: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return parsed


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------


def _resolve(schema: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    """Follow a local `$ref` one level. Nested models are emitted under `$defs`."""
    ref = node.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
        return node
    target = schema.get("$defs", {}).get(ref.removeprefix("#/$defs/"))
    return target if isinstance(target, dict) else node


def _properties(schema: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    resolved = _resolve(schema, node)
    props = resolved.get("properties")
    return props if isinstance(props, dict) else {}


def _required(schema: dict[str, Any], node: dict[str, Any]) -> set[str]:
    resolved = _resolve(schema, node)
    required = resolved.get("required")
    return set(required) if isinstance(required, list) else set()


def _enum_values(node: dict[str, Any]) -> set[Any] | None:
    values = node.get("enum")
    if isinstance(values, list):
        return set(values)
    const = node.get("const")
    if const is not None:
        return {const}
    return None


def breaking_changes(old: dict[str, Any], new: dict[str, Any], path: str = "") -> list[str]:
    """Return every backwards-incompatible difference from `old` to `new`.

    Breaking, because a record that validated against `old` would stop validating:
      * a property was removed
      * a property became required
      * an enum lost a member

    Not breaking:
      * a new optional property
      * an enum gained a member
      * descriptions, titles, defaults, examples

    Deliberately conservative in one direction: it walks only the property tree reachable
    from the roots, and it treats a `$ref` target by name. That is enough for the
    envelopes here and it never reports a false "compatible", which is the direction that
    matters.
    """
    findings: list[str] = []

    old_props = _properties(old, old)
    new_props = _properties(new, new)
    old_required = _required(old, old)
    new_required = _required(new, new)

    for name in sorted(old_props):
        where = f"{path}.{name}" if path else name
        if name not in new_props:
            findings.append(f"property removed: {where}")
            continue
        if name in new_required and name not in old_required:
            findings.append(f"optional property became required: {where}")

        old_node = _resolve(old, old_props[name])
        new_node = _resolve(new, new_props[name])

        old_enum = _enum_values(old_node)
        new_enum = _enum_values(new_node)
        if old_enum is not None and new_enum is not None:
            lost = old_enum - new_enum
            if lost:
                findings.append(f"enum narrowed at {where}: lost {sorted(map(str, lost))}")

        # Recurse whenever the OLD side had properties. Guarding on both sides would
        # miss a nested object that lost every property it had, which is the most
        # breaking change of all.
        if _properties(old, old_props[name]):
            findings.extend(_nested(old, new, old_props[name], new_props[name], where))

    for name in sorted(new_required - old_required):
        if name not in old_props:
            where = f"{path}.{name}" if path else name
            findings.append(f"new required property: {where}")

    return findings


def _nested(
    old_root: dict[str, Any],
    new_root: dict[str, Any],
    old_node: dict[str, Any],
    new_node: dict[str, Any],
    path: str,
) -> list[str]:
    """Recurse into a nested object, keeping the root for `$ref` resolution."""
    old_sub = dict(_resolve(old_root, old_node))
    new_sub = dict(_resolve(new_root, new_node))
    old_sub["$defs"] = old_root.get("$defs", {})
    new_sub["$defs"] = new_root.get("$defs", {})
    return breaking_changes(old_sub, new_sub, path)
