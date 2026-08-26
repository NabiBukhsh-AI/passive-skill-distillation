"""Redaction and PII scrubbing (TASK-012, component C-03).

Ensures no personal data reaches a distiller or a skill, without destroying the signal
the distiller needs.

**Presence-versus-absence semantics are the whole design constraint.** The paper's
headline retail failure is that a non-reasoning agent calls an authentication tool with a
*fabricated* email argument. That failure is only visible if, after redaction, you can
still tell apart:

  * a message that contains an email from one that contains none, and
  * an email the user actually supplied from a different one the agent invented.

So placeholders are typed and **stable per distinct value within a trajectory**. The same
address seen twice becomes `<EMAIL_1>` twice; a different address becomes `<EMAIL_2>`.
A tool call carrying `<EMAIL_1>` used a value the user gave; one carrying `<EMAIL_2>`
that appears in no user turn is exactly the fabrication the skill needs to forbid.

Indices are per-trajectory and carry no meaning across trajectories. Comparing
`<EMAIL_1>` in one episode to `<EMAIL_1>` in another is meaningless by construction.

C-03 names over-redaction as the highest risk here, so every detector that can be
validated is validated: card numbers must pass Luhn, national ids must pass their
structural rules. A regex alone would eat order numbers and version strings.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from psd.core.models import Redaction, Trajectory

#: Bump when detector behavior changes. Recorded on every redacted trajectory and on
#: every corpus manifest, so a corpus states which policy produced it.
REDACTION_POLICY_VERSION = "redaction/1.0"


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detector:
    """One PII class.

    `priority` breaks ties when two detectors claim overlapping spans; lower wins. A card
    number contains digit runs a phone detector would happily claim, so cards outrank
    phones.
    """

    name: str
    pattern: re.Pattern[str]
    priority: int
    validator: str | None = None


def _luhn_ok(digits: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _valid_card(text: str) -> bool:
    digits = re.sub(r"[ -]", "", text)
    if not 13 <= len(digits) <= 19:
        return False
    return _luhn_ok(digits)


def _valid_us_ssn(text: str) -> bool:
    """Reject the structurally impossible ranges the SSA never issues.

    Without this, any `123-45-6789`-shaped identifier is redacted, including order
    references and part numbers, which is exactly the over-redaction C-03 warns about.
    """
    digits = re.sub(r"[ -]", "", text)
    if len(digits) != 9:
        return False
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    if area in {"000", "666"} or area.startswith("9"):
        return False
    return group != "00" and serial != "0000"


def _valid_phone(text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    return 10 <= len(digits) <= 15


_VALIDATORS = {
    "card": _valid_card,
    "us_ssn": _valid_us_ssn,
    "phone": _valid_phone,
}

DETECTORS: tuple[Detector, ...] = (
    Detector(
        name="EMAIL",
        pattern=re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        priority=0,
    ),
    Detector(
        name="CREDIT_CARD",
        pattern=re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
        priority=1,
        validator="card",
    ),
    Detector(
        name="US_SSN",
        pattern=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        priority=2,
        validator="us_ssn",
    ),
    Detector(
        name="PHONE",
        # Requires punctuation or a leading +, so bare digit runs (order ids, totals)
        # are not claimed. Over-redaction is the failure mode that destroys signal.
        pattern=re.compile(
            r"(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?|\b\d{2,4}[ .-])\d{2,4}[ .-]\d{2,6}\b"
        ),
        priority=3,
        validator="phone",
    ),
    Detector(
        name="IBAN",
        pattern=re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
        priority=4,
    ),
)

#: Only under `strict_phone`. A bare run of 10 to 15 digits is a phone number in a
#: telecom transcript and an order number in a retail one, and nothing in the text tells
#: you which. Enabling this by default would redact `Order number 8471629503` and destroy
#: exactly the kind of identifier the distiller reasons about; leaving it off in telecom
#: would leak subscriber numbers.
#:
#: So it is a per-domain policy, set in the domain profile, not a global default. See
#: `docs/ASSUMPTIONS.md` ASM-006.
STRICT_PHONE_DETECTOR = Detector(
    name="PHONE",
    pattern=re.compile(r"\b\d{10,15}\b"),
    priority=5,
    validator="phone",
)


@dataclass(frozen=True)
class RedactionPolicy:
    """Per-domain redaction behavior.

    `strict_phone` trades precision for recall on phone numbers. Turn it on for domains
    whose transcripts genuinely carry subscriber numbers (tau2-telecom), and leave it off
    where bare digit runs are business identifiers (tau2-retail, SSB-Verified).
    """

    version: str = REDACTION_POLICY_VERSION
    strict_phone: bool = False

    def detectors(self) -> tuple[Detector, ...]:
        if self.strict_phone:
            return (*DETECTORS, STRICT_PHONE_DETECTOR)
        return DETECTORS


DEFAULT_POLICY = RedactionPolicy()
TELECOM_POLICY = RedactionPolicy(strict_phone=True)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    cls: str
    priority: int
    text: str


def find_spans(text: str, policy: RedactionPolicy = DEFAULT_POLICY) -> list[Span]:
    """Every PII span in `text`, overlaps resolved, in left-to-right order."""
    found: list[Span] = []
    for detector in policy.detectors():
        for match in detector.pattern.finditer(text):
            value = match.group(0)
            validator = _VALIDATORS.get(detector.validator or "")
            if validator is not None and not validator(value):
                continue
            found.append(Span(match.start(), match.end(), detector.name, detector.priority, value))

    # Longest span wins; ties go to the higher-priority (lower number) detector. This is
    # what stops PHONE from claiming the tail of a card number.
    found.sort(key=lambda s: (s.start, -(s.end - s.start), s.priority))
    resolved: list[Span] = []
    cursor = -1
    for span in found:
        if span.start >= cursor:
            resolved.append(span)
            cursor = span.end
    return resolved


# ---------------------------------------------------------------------------
# Redaction state
# ---------------------------------------------------------------------------


@dataclass
class RedactionState:
    """Per-trajectory placeholder assignment.

    Deterministic by construction: indices are handed out in the order values are first
    encountered during a fixed traversal, so the same trajectory always redacts to the
    same bytes (FR-011 applies downstream).
    """

    policy: RedactionPolicy = DEFAULT_POLICY
    _assigned: dict[tuple[str, str], str] = field(default_factory=dict)
    _next_index: dict[str, int] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def placeholder_for(self, cls: str, value: str) -> str:
        key = (cls, value)
        if key not in self._assigned:
            index = self._next_index.get(cls, 0) + 1
            self._next_index[cls] = index
            self._assigned[key] = f"<{cls}_{index}>"
        self.counts[cls] = self.counts.get(cls, 0) + 1
        return self._assigned[key]

    @property
    def distinct_values(self) -> int:
        return len(self._assigned)


def redact_text(text: str, state: RedactionState) -> str:
    """Replace every detected span with a typed, value-stable placeholder."""
    spans = find_spans(text, state.policy)
    if not spans:
        return text
    out: list[str] = []
    cursor = 0
    for span in spans:
        out.append(text[cursor : span.start])
        out.append(state.placeholder_for(span.cls, span.text))
        cursor = span.end
    out.append(text[cursor:])
    return "".join(out)


def _redact_value(value: Any, state: RedactionState) -> Any:
    """Walk an arbitrary JSON value, redacting strings in a deterministic order."""
    if isinstance(value, str):
        return redact_text(value, state)
    if isinstance(value, dict):
        return {key: _redact_value(value[key], state) for key in sorted(value)}
    if isinstance(value, list):
        return [_redact_value(item, state) for item in value]
    return value


# ---------------------------------------------------------------------------
# Trajectory-level entry point
# ---------------------------------------------------------------------------


def _redactable_paths(trajectory: Trajectory) -> Iterator[tuple[str, ...]]:
    """Fixed traversal order.

    Order determines placeholder numbering, so it is part of the contract. Steps in index
    order, and within a step: what the environment said, then what the model said, then
    what it did, then what came back. That is the order a reader reconstructs the episode
    in, which makes `<EMAIL_1>` mean "the first address anyone mentioned".
    """
    for index in range(len(trajectory.steps)):
        yield ("steps", str(index), "observation", "text")
        yield ("steps", str(index), "output", "text")
        yield ("steps", str(index), "output", "reasoning_text")
        yield ("steps", str(index), "action", "text")
        yield ("steps", str(index), "action", "arguments")
        yield ("steps", str(index), "action", "arguments_raw")
        yield ("steps", str(index), "result", "text")


def redact_trajectory(
    trajectory: Trajectory,
    state: RedactionState | None = None,
    policy: RedactionPolicy = DEFAULT_POLICY,
) -> tuple[Trajectory, Redaction]:
    """Redact one trajectory, returning it plus a report of counts by class.

    The report never carries redacted values. Recording what was removed alongside the
    thing you removed it from would defeat the point (C-03).
    """
    state = state or RedactionState(policy=policy)
    payload = trajectory.model_dump(mode="json")

    for path in _redactable_paths(trajectory):
        node: Any = payload
        for key in path[:-1]:
            if node is None:
                break
            node = node[int(key)] if isinstance(node, list) else node.get(key)
        if node is None:
            continue
        leaf = path[-1]
        current = node.get(leaf) if isinstance(node, dict) else None
        if current is None:
            continue
        node[leaf] = _redact_value(current, state)

    payload["redaction"] = {
        "applied": True,
        "policy_version": state.policy.version,
        "counts": dict(sorted(state.counts.items())),
    }
    # Validate through JSON, not the dict: `model_dump(mode="json")` renders datetimes
    # as ISO-8601 strings, and the models are strict, so Python-mode validation refuses
    # them. Same reason the normalizer does it (see psd.ingest.normalizers.base).
    redacted = Trajectory.model_validate_json(json.dumps(payload))
    return redacted, redacted.redaction
