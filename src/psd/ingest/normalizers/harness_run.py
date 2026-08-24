"""Mapper for `harness_run`: trajectories our own benchmark adapters emit (TASK-010).

This format is already close to canonical, because our adapters build it. The mapper
still exists rather than being skipped, for two reasons:

  * It records dropped fields in `normalization_warnings`, so a drift between an adapter
    and the schema is visible in the data rather than discovered later as a gap.
  * It is the reference other formats are written against.

It maps. It does not repair. A missing required field is left missing so that validation
refuses the record (spec Section 30.1 rule 14).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from psd.core.models import Trajectory
from psd.ingest.normalizers.base import register_mapper

#: Top-level keys the canonical envelope defines. Anything else is dropped and recorded.
_KNOWN_TOP_LEVEL = set(Trajectory.model_fields)


@register_mapper("harness_run")
def map_harness_run(body: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    canonical = {key: value for key, value in body.items() if key in _KNOWN_TOP_LEVEL}
    warnings = [
        f"dropped unknown top-level field: {key}" for key in sorted(set(body) - _KNOWN_TOP_LEVEL)
    ]
    return canonical, warnings
