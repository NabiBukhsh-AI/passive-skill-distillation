"""Per-domain analysis profiles."""

from psd.analysis.profiles.registry import (
    ALFWORLD,
    SSB_VERIFIED,
    TAU2_RETAIL,
    TAU2_TELECOM,
    classify_email,
    get_profile,
    registered_domains,
)

__all__ = [
    "ALFWORLD",
    "SSB_VERIFIED",
    "TAU2_RETAIL",
    "TAU2_TELECOM",
    "classify_email",
    "get_profile",
    "registered_domains",
]
