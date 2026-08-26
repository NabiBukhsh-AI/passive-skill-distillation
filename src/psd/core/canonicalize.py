"""Action canonicalization (TASK-017, ALG-002).

Maps a raw action to a stable symbol so that n-grams (ALG-004) and loop detection
(ALG-005) are counting the same thing across episodes.

The subtle part is Step 1's value-class escape hatch. A pure type signature turns
`find_user_id_by_email(email="alice@real.com")` and
`find_user_id_by_email(email="user@example.com")` into the same symbol, because both are
`email:str`. That erases the paper's headline retail failure, which is entirely about
argument CONTENT: the agent calls the authentication tool with an address the user never
supplied, in 59% of non-reasoning rollouts.

So a domain profile may declare specific arguments value-sensitive, and those contribute a
coarse value class instead of a type. Profiles must declare them explicitly: guessing
which arguments matter would be a per-domain judgement made in the wrong place.

Pure function, no I/O, no wall-clock. `psd.core` imports nothing from `psd.*`.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from psd.core.models import Action

#: ALG-002 Step 3: cap the API-symbol multiset so one enormous cell does not produce a
#: symbol nothing else ever matches.
MAX_CODE_SYMBOLS = 8

NOOP = "noop"
CODE_UNPARSED = "code:unparsed"


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return "unknown"


#: A value classifier maps an argument value to a coarse class string.
ValueClassifier = Callable[[Any], str]


@dataclass(frozen=True)
class DomainProfile:
    """Per-domain canonicalization vocabulary (spec Section 18, `analysis/profiles/`).

    Lives in `core` because `canonicalize` needs the type and `core` may not import from
    `psd.analysis`. The concrete profiles live in `psd/analysis/profiles/`.
    """

    domain: str

    #: Tool argument keys whose VALUE class matters, keyed by tool name. `"*"` applies to
    #: every tool. ALG-002 Step 1.
    value_sensitive_args: Mapping[str, frozenset[str]] = field(default_factory=dict)

    #: Named classifiers for those arguments, keyed by argument name.
    value_classifiers: Mapping[str, ValueClassifier] = field(default_factory=dict)

    #: Admissible-command verbs, longest first at match time. ALG-002 Step 2.
    verbs: Sequence[str] = ()

    #: Words that separate slots in a command, for example "with" in
    #: "cool tomato 1 with fridge 1".
    connectives: frozenset[str] = frozenset({"from", "with", "in", "on", "to", "at"})

    #: Nouns that are receptacles rather than portable objects, so slots are typed.
    receptacles: frozenset[str] = frozenset()

    #: Actions that legitimately repeat, exempt from stall detection (ALG-005).
    stall_whitelist: frozenset[str] = frozenset()

    #: Observation fields that change every step and must be normalized away before two
    #: observations can be compared (ALG-005 Step 1).
    volatile_observation_patterns: Sequence[str] = ()

    def sensitive_args_for(self, tool_name: str) -> frozenset[str]:
        return frozenset(
            {
                *self.value_sensitive_args.get("*", frozenset()),
                *self.value_sensitive_args.get(tool_name, frozenset()),
            }
        )


DEFAULT_PROFILE = DomainProfile(domain="default")


# ---------------------------------------------------------------------------
# Step 1: tool calls
# ---------------------------------------------------------------------------


def _canonicalize_tool_call(action: Action, profile: DomainProfile) -> str:
    name = action.name or "unknown"
    arguments = action.arguments or {}
    sensitive = profile.sensitive_args_for(name)

    parts: list[str] = []
    for key in sorted(arguments):
        if key in sensitive:
            classifier = profile.value_classifiers.get(key)
            value_class = classifier(arguments[key]) if classifier else _type_name(arguments[key])
            parts.append(f"{key}:{value_class}")
        else:
            parts.append(f"{key}:{_type_name(arguments[key])}")

    # A declared-sensitive argument that is ABSENT is itself a signal: calling an auth
    # tool with no email at all is a different failure from calling it with a guessed
    # one, and both differ from calling it correctly.
    for key in sorted(sensitive - set(arguments)):
        classifier = profile.value_classifiers.get(key)
        parts.append(f"{key}:{classifier(None) if classifier else 'null'}")

    return f"tool:{name}(" + ",".join(parts) + ")"


# ---------------------------------------------------------------------------
# Step 2: text actions
# ---------------------------------------------------------------------------


def _match_verb(tokens: list[str], profile: DomainProfile) -> tuple[str, int] | None:
    """Longest-first, so "go to" wins over "go"."""
    for verb in sorted(profile.verbs, key=lambda v: -len(v.split())):
        verb_tokens = verb.split()
        if tokens[: len(verb_tokens)] == verb_tokens:
            return verb, len(verb_tokens)
    return None


def _slot_for(segment: list[str], profile: DomainProfile) -> str:
    """Type a noun phrase as a receptacle or a portable object."""
    for token in segment:
        if token in profile.receptacles:
            return "<recep>"
    return "<obj>"


def _canonicalize_text_action(action: Action, profile: DomainProfile) -> str:
    raw = (action.text or "").strip().lower()
    if not raw:
        return NOOP
    tokens = re.findall(r"[a-z0-9_]+", raw)
    if not tokens:
        return NOOP

    matched = _match_verb(tokens, profile)
    if matched is None:
        # Unknown verb: keep the head token so the symbol is still informative, but do
        # not pretend to have parsed it.
        return f"text:{tokens[0]}"

    verb, consumed = matched
    rest = tokens[consumed:]
    if not rest:
        return verb

    rendered: list[str] = []
    segment: list[str] = []
    for token in rest:
        if token in profile.connectives:
            if segment:
                rendered.append(_slot_for(segment, profile))
                segment = []
            rendered.append(token)
        else:
            segment.append(token)
    if segment:
        rendered.append(_slot_for(segment, profile))

    return " ".join([verb, *rendered])


# ---------------------------------------------------------------------------
# Step 3: code execution
# ---------------------------------------------------------------------------


def _call_symbol(node: ast.expr) -> str | None:
    """Render a call target as a dotted symbol, for example `pandas.read_excel`."""
    parts: list[str] = []
    current: ast.expr | None = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    elif parts:
        parts.append("<expr>")
    else:
        return None
    return ".".join(reversed(parts))


def _canonicalize_code(action: Action, profile: DomainProfile) -> str:
    source = action.text or ""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return CODE_UNPARSED

    symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            symbol = _call_symbol(node.func)
            if symbol:
                symbols.add(symbol)

    if not symbols:
        return "code:none"
    capped = sorted(symbols)[:MAX_CODE_SYMBOLS]
    return "code:" + "+".join(capped)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def canonicalize_action(action: Action, profile: DomainProfile = DEFAULT_PROFILE) -> str:
    """ALG-002. Map one action to a stable symbol.

    Step 4 lowercases and strips. Deterministic: every set is sorted before it is
    rendered, so the symbol never depends on iteration order.
    """
    if action.kind == "tool_call":
        canonical = _canonicalize_tool_call(action, profile)
    elif action.kind == "text_action":
        canonical = _canonicalize_text_action(action, profile)
    elif action.kind == "code_execution":
        canonical = _canonicalize_code(action, profile)
    elif action.kind == "noop":
        canonical = NOOP
    else:  # pragma: no cover - the Literal makes this unreachable
        canonical = f"unknown:{action.kind}"

    return canonical.strip().lower()
