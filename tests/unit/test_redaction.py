"""TASK-012 acceptance tests.

Criteria:
  * Recall > 0.98 on the labeled fixture set.
  * A regression test proves the retail-style "real email present versus absent"
    distinction survives redaction.

The second is the one that matters. Redaction that scrubs text into uniformity is
trivially safe and completely useless: the paper's headline retail finding is a failure
mode about argument CONTENT, and a scrubber that erases the difference between a real
email and a fabricated one deletes the very thing the skill is compiled to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from psd.ingest.redaction import (
    DEFAULT_POLICY,
    REDACTION_POLICY_VERSION,
    TELECOM_POLICY,
    RedactionPolicy,
    RedactionState,
    find_spans,
    redact_text,
    redact_trajectory,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
LABELED = FIXTURES / "pii" / "labeled.json"
EXAMPLE = FIXTURES / "trajectories" / "spec_section_10_3_example.json"

POLICIES = {"default": DEFAULT_POLICY, "strict_phone": TELECOM_POLICY}

#: Spec Section 10.9: redaction recall on the labeled fixture set must exceed this, and
#: falling below it blocks corpus creation.
RECALL_THRESHOLD = 0.98

#: Not in the spec, but over-redaction is the failure C-03 ranks highest, so precision is
#: measured and thresholded too rather than left to chance.
PRECISION_THRESHOLD = 0.95


def labeled_cases() -> list[dict[str, Any]]:
    payload = json.loads(LABELED.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = payload["cases"]
    return cases


def score() -> tuple[int, int, int, list[str]]:
    """Return (true positives, false negatives, false positives, notes)."""
    tp = fn = fp = 0
    notes: list[str] = []
    for case in labeled_cases():
        policy = POLICIES[case.get("policy", "default")]
        found = [span.text for span in find_spans(case["text"], policy)]
        remaining = list(found)
        for expected in case["expected"]:
            if expected in remaining:
                remaining.remove(expected)
                tp += 1
            else:
                fn += 1
                notes.append(f"MISSED {case['id']}: {expected!r} (found {found})")
        for leftover in remaining:
            fp += 1
            notes.append(f"FALSE POSITIVE {case['id']}: {leftover!r}")
    return tp, fn, fp, notes


# ---------------------------------------------------------------------------
# Acceptance: recall on the labeled set
# ---------------------------------------------------------------------------


def test_recall_exceeds_the_threshold() -> None:
    tp, fn, _fp, notes = score()
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    assert recall > RECALL_THRESHOLD, (
        f"recall {recall:.4f} is below {RECALL_THRESHOLD}; spec Section 10.9 blocks "
        "corpus creation at this point.\n" + "\n".join(notes)
    )


def test_precision_exceeds_the_threshold() -> None:
    """Over-redaction is the highest risk in C-03, so it is measured, not assumed."""
    tp, _fn, fp, notes = score()
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    assert precision > PRECISION_THRESHOLD, (
        f"precision {precision:.4f} is below {PRECISION_THRESHOLD}; the scrubber is "
        "eating signal the distiller needs.\n" + "\n".join(notes)
    )


def test_the_fixture_set_is_not_trivially_easy() -> None:
    """Guards the two thresholds above from being met by an empty or all-positive set.

    A labeled set with no negatives cannot detect over-redaction, and a set with no
    positives cannot detect under-redaction. Both thresholds would read 1.0.
    """
    cases = labeled_cases()
    positives = [c for c in cases if c["expected"]]
    negatives = [c for c in cases if not c["expected"]]
    assert len(positives) >= 20, "too few positive cases to measure recall meaningfully"
    assert len(negatives) >= 20, "too few negative cases to measure precision meaningfully"


@pytest.mark.parametrize("case", labeled_cases(), ids=lambda c: c["id"])
def test_each_labeled_case(case: dict[str, Any]) -> None:
    """Per-case, so a failure names the case rather than only moving an aggregate."""
    policy = POLICIES[case.get("policy", "default")]
    found = sorted(span.text for span in find_spans(case["text"], policy))
    assert found == sorted(case["expected"])


# ---------------------------------------------------------------------------
# Acceptance: presence versus absence survives redaction
# ---------------------------------------------------------------------------


def trajectory_payload(observation: str, tool_email: str | None) -> dict[str, Any]:
    """A retail-shaped episode: a user turn, then an authentication tool call."""
    payload: dict[str, Any] = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["steps"][0]["observation"]["text"] = observation
    if tool_email is None:
        payload["steps"][0]["action"]["arguments"] = {}
        payload["steps"][0]["action"]["arguments_raw"] = "{}"
    else:
        payload["steps"][0]["action"]["arguments"] = {"email": tool_email}
        payload["steps"][0]["action"]["arguments_raw"] = json.dumps({"email": tool_email})
    return payload


def redacted(observation: str, tool_email: str | None) -> Any:
    from psd.core.models import Trajectory

    trajectory = Trajectory.model_validate_json(
        json.dumps(trajectory_payload(observation, tool_email))
    )
    result, _ = redact_trajectory(trajectory)
    return result


def test_email_present_and_absent_remain_distinguishable() -> None:
    """TASK-012 acceptance, stated verbatim in the playbook.

    A message containing a real email and one containing no email must remain
    distinguishable after redaction.
    """
    with_email = redacted("My email is shopper@example.com", None)
    without_email = redacted("I do not remember which email I used", None)

    assert "<EMAIL_1>" in with_email.steps[0].observation.text
    assert "<EMAIL" not in without_email.steps[0].observation.text
    assert with_email.steps[0].observation.text != without_email.steps[0].observation.text


def test_a_fabricated_argument_stays_distinguishable_from_a_supplied_one() -> None:
    """The retail failure mode itself (spec Section 3.1).

    The paper's finding is that the agent calls the auth tool with an email the user
    never supplied, in 59% of non-reasoning rollouts. After redaction the distiller must
    still be able to see that:

      * supplied  -> the tool argument placeholder ALSO appears in the user turn
      * fabricated -> the tool argument placeholder appears NOWHERE in the user turn

    If redaction mapped every address to one placeholder, both cases would look
    identical and the corpus could not support the rule that fixes them.
    """
    supplied = redacted("My email is shopper@example.com", tool_email="shopper@example.com")
    fabricated = redacted("My email is shopper@example.com", tool_email="guessed@example.com")

    supplied_observation = supplied.steps[0].observation.text
    supplied_argument = supplied.steps[0].action.arguments["email"]
    assert supplied_argument in supplied_observation, (
        "a supplied email must redact to the same placeholder in both places"
    )

    fabricated_observation = fabricated.steps[0].observation.text
    fabricated_argument = fabricated.steps[0].action.arguments["email"]
    assert fabricated_argument not in fabricated_observation, (
        "a fabricated email must redact to a DIFFERENT placeholder from the one the "
        "user supplied, or the corpus cannot show the failure"
    )
    assert fabricated_argument == "<EMAIL_2>"
    assert supplied_argument == "<EMAIL_1>"


def test_calling_the_tool_with_no_email_at_all_is_distinguishable() -> None:
    """The third case: the agent calls auth before the user supplied anything."""
    none_supplied = redacted("Hi, I need help with an order", tool_email=None)
    assert none_supplied.steps[0].action.arguments == {}
    assert "<EMAIL" not in none_supplied.steps[0].observation.text


def test_the_same_value_is_stable_across_steps() -> None:
    state = RedactionState()
    first = redact_text("write to sam@example.com", state)
    second = redact_text("again, sam@example.com", state)
    assert "<EMAIL_1>" in first
    assert "<EMAIL_1>" in second


def test_distinct_values_get_distinct_placeholders() -> None:
    state = RedactionState()
    out = redact_text("from a@example.com to b@example.com", state)
    assert "<EMAIL_1>" in out
    assert "<EMAIL_2>" in out


# ---------------------------------------------------------------------------
# Report hygiene and determinism
# ---------------------------------------------------------------------------


def test_the_report_records_counts_but_never_values() -> None:
    """C-03: keep detection metadata, never the redacted values."""
    from psd.core.models import Trajectory

    trajectory = Trajectory.model_validate_json(
        json.dumps(trajectory_payload("mail me at leaked@example.com", "leaked@example.com"))
    )
    result, report = redact_trajectory(trajectory)

    assert report.applied is True
    assert report.policy_version == REDACTION_POLICY_VERSION
    assert report.counts["EMAIL"] >= 2
    serialized = json.dumps(report.model_dump(mode="json"))
    assert "leaked@example.com" not in serialized
    assert "leaked@example.com" not in result.model_dump_json()


def test_redaction_is_deterministic() -> None:
    from psd.core.models import Trajectory

    payload = trajectory_payload("write to zoe@example.com", "zoe@example.com")
    once, _ = redact_trajectory(Trajectory.model_validate_json(json.dumps(payload)))
    twice, _ = redact_trajectory(Trajectory.model_validate_json(json.dumps(payload)))
    assert once.model_dump_json() == twice.model_dump_json()


def test_redaction_is_idempotent() -> None:
    """Redacting an already-redacted trajectory must change nothing.

    A placeholder must not itself look like PII to the next pass, or repeated ingest
    would ratchet `<EMAIL_1>` into `<EMAIL_1_1>` and break value stability.
    """
    from psd.core.models import Trajectory

    payload = trajectory_payload("write to zoe@example.com", "zoe@example.com")
    once, _ = redact_trajectory(Trajectory.model_validate_json(json.dumps(payload)))
    twice, _ = redact_trajectory(once)
    assert twice.steps[0].observation.text == once.steps[0].observation.text


def test_text_without_pii_is_returned_unchanged() -> None:
    state = RedactionState()
    text = "The order shipped and arrived on time."
    assert redact_text(text, state) is text


# ---------------------------------------------------------------------------
# The strict-phone policy tradeoff (ASM-006)
# ---------------------------------------------------------------------------


def test_default_policy_keeps_bare_digit_runs() -> None:
    """A bare digit run is an order id in retail. Redacting it destroys signal."""
    assert find_spans("Order number 8471629503 is confirmed", DEFAULT_POLICY) == []


def test_strict_phone_policy_redacts_bare_digit_runs() -> None:
    """The same run is a subscriber number in telecom, and must not leak."""
    spans = find_spans("Call 5551234567 now", TELECOM_POLICY)
    assert [s.text for s in spans] == ["5551234567"]


def test_the_tradeoff_is_real_and_documented() -> None:
    """Names the cost of strict_phone explicitly, so nobody enables it by reflex.

    Under strict_phone an order number IS redacted. That is the price of catching bare
    subscriber numbers, and it is why this is a per-domain policy rather than a default.
    """
    order = "Order number 8471629503 is confirmed"
    assert find_spans(order, DEFAULT_POLICY) == []
    assert [s.text for s in find_spans(order, TELECOM_POLICY)] == ["8471629503"]


def test_policy_version_is_recorded() -> None:
    assert RedactionPolicy().version == REDACTION_POLICY_VERSION
    assert TELECOM_POLICY.strict_phone is True
    assert DEFAULT_POLICY.strict_phone is False


# ---------------------------------------------------------------------------
# Validators stop over-redaction
# ---------------------------------------------------------------------------


def test_luhn_failure_is_not_a_card() -> None:
    assert find_spans("Reference 4111111111111112 is not a card") == []


def test_luhn_success_is_a_card() -> None:
    assert [s.cls for s in find_spans("PAN 4111111111111111 stored")] == ["CREDIT_CARD"]


@pytest.mark.parametrize("ssn", ["666-45-6789", "900-45-6789", "000-45-6789"])
def test_structurally_impossible_ssns_are_not_redacted(ssn: str) -> None:
    assert find_spans(f"Case {ssn} was escalated") == []


def test_a_card_is_not_split_into_phone_fragments() -> None:
    """Priority resolution: the longer, higher-priority span wins."""
    spans = find_spans("Charged 4111-1111-1111-1111 today")
    assert len(spans) == 1
    assert spans[0].cls == "CREDIT_CARD"
