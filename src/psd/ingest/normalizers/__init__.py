"""Trajectory normalizers (C-02, TASK-010).

Importing this package registers every mapper. Dispatch is by `source_format`; an
unknown format quarantines rather than guessing at a shape.
"""

from psd.ingest.normalizers import harness_run as _harness_run  # noqa: F401  (registers)
from psd.ingest.normalizers.base import (
    DEFAULT_MAX_TRAJECTORY_BYTES,
    NormalizationOutcome,
    Quarantine,
    QuarantineRule,
    normalize,
    register_mapper,
    registered_formats,
)

__all__ = [
    "DEFAULT_MAX_TRAJECTORY_BYTES",
    "NormalizationOutcome",
    "Quarantine",
    "QuarantineRule",
    "normalize",
    "register_mapper",
    "registered_formats",
]
