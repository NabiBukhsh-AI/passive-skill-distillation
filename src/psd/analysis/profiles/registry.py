"""Per-domain canonicalization and analysis profiles (TASK-017, spec Section 18).

The `DomainProfile` type lives in `psd.core.canonicalize`, because `core` may not import
from `psd.analysis`. The concrete instances live here.

Nothing in a profile is inferred at runtime. ALG-002's implementation note is explicit
that domain profiles must DECLARE which arguments are value-sensitive, and the reason is
that guessing would put a per-domain research judgement inside generic code where nobody
would ever find it again.
"""

from __future__ import annotations

import re
from typing import Any

from psd.core.canonicalize import DEFAULT_PROFILE, DomainProfile

# ---------------------------------------------------------------------------
# Value classifiers
# ---------------------------------------------------------------------------

#: Addresses that are obviously not a real customer's. This is the shape a fabricated
#: argument takes: the model invents something plausible from the reserved example
#: domains, or from a generic local part it has seen in documentation.
_PLACEHOLDER_DOMAINS = frozenset(
    {"example.com", "example.org", "example.net", "test.com", "email.com", "domain.com"}
)
_PLACEHOLDER_LOCALS = frozenset(
    {"user", "customer", "test", "example", "email", "your_email", "youremail", "name"}
)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.IGNORECASE)
#: A value the redactor already replaced. Its presence means an address WAS supplied.
_REDACTION_PLACEHOLDER_RE = re.compile(r"^<EMAIL_\d+>$")


def classify_email(value: Any) -> str:
    """Coarse value class for an email argument (ALG-002 Step 1).

    Three classes, exactly the ones spec Section 12 names:

      * `email_absent`      no value at all. The agent called the tool before the user
                            supplied anything.
      * `placeholder_like`  a reserved example address or a generic local part. This is
                            what a fabricated argument looks like.
      * `email_present`     an address that could be real, including one the redactor has
                            already replaced with `<EMAIL_n>`.

    A redaction placeholder counts as present, not as placeholder-like. The redactor only
    produces one when a real value was there, and conflating the two would erase the
    distinction this classifier exists to preserve.
    """
    if value is None:
        return "email_absent"
    if not isinstance(value, str):
        return "email_absent"
    text = value.strip()
    if not text:
        return "email_absent"
    if _REDACTION_PLACEHOLDER_RE.match(text):
        return "email_present"
    if not _EMAIL_RE.match(text):
        return "placeholder_like"

    local, _, domain = text.lower().partition("@")
    if domain in _PLACEHOLDER_DOMAINS or local in _PLACEHOLDER_LOCALS:
        return "placeholder_like"
    return "email_present"


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

ALFWORLD_RECEPTACLES = frozenset(
    {
        "armchair",
        "bathtubbasin",
        "bed",
        "cabinet",
        "coffeemachine",
        "coffeetable",
        "countertop",
        "desk",
        "diningtable",
        "drawer",
        "dresser",
        "fridge",
        "garbagecan",
        "handtowelholder",
        "laundryhamper",
        "microwave",
        "ottoman",
        "safe",
        "shelf",
        "sidetable",
        "sinkbasin",
        "sofa",
        "stoveburner",
        "toilet",
        "toiletpaperhanger",
        "towelholder",
        "tvstand",
    }
)

ALFWORLD = DomainProfile(
    domain="alfworld",
    verbs=(
        "go to",
        "take",
        "put",
        "open",
        "close",
        "toggle",
        "heat",
        "cool",
        "clean",
        "examine",
        "look",
        "inventory",
        "use",
        "move",
        "slice",
    ),
    receptacles=ALFWORLD_RECEPTACLES,
    # `look` and `inventory` are legitimately repeatable, but the paper's Figure 2 shows a
    # 20-step identical-observation `look` run as THE stall to catch, so neither is
    # whitelisted. Whitelisting them would suppress the exact pattern ALG-005 exists for.
    stall_whitelist=frozenset(),
    volatile_observation_patterns=(
        r"\bstep\s+\d+\b",
        r"\bturn\s+\d+\b",
        r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\b",
    ),
)

TAU2_RETAIL = DomainProfile(
    domain="tau2_retail",
    value_sensitive_args={
        # The paper's headline failure. A pure type signature would make a fabricated
        # address and a supplied one the same symbol.
        "find_user_id_by_email": frozenset({"email"}),
        "*": frozenset({"email"}),
    },
    value_classifiers={"email": classify_email},
    connectives=frozenset(),
    volatile_observation_patterns=(
        r"\bturn\s+\d+\b",
        r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\b",
    ),
)

TAU2_TELECOM = DomainProfile(
    domain="tau2_telecom",
    value_sensitive_args={"*": frozenset({"email"})},
    value_classifiers={"email": classify_email},
    connectives=frozenset(),
    volatile_observation_patterns=(
        r"\bturn\s+\d+\b",
        r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\b",
    ),
)

SSB_VERIFIED = DomainProfile(
    domain="ssb_verified",
    connectives=frozenset(),
    volatile_observation_patterns=(
        r"\bstep\s+\d+\b",
        r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\b",
    ),
)

_PROFILES: dict[str, DomainProfile] = {
    profile.domain: profile
    for profile in (ALFWORLD, TAU2_RETAIL, TAU2_TELECOM, SSB_VERIFIED, DEFAULT_PROFILE)
}


def get_profile(domain: str) -> DomainProfile:
    """Return the profile for a domain, falling back to the domain-agnostic default.

    Falling back rather than raising is deliberate: FR-014 requires a domain-agnostic
    default set so a new domain can be analysed before anyone has written its profile.
    The cost is that value-sensitive arguments are not declared for it, which shows up as
    weaker canonicalization rather than as an error.
    """
    return _PROFILES.get(domain, DEFAULT_PROFILE)


def registered_domains() -> list[str]:
    return sorted(_PROFILES)
