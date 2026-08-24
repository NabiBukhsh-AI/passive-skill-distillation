"""Regenerate the published JSON Schemas from the Pydantic models (TASK-009).

Run this after an intentional model change, then read the diff before committing it.
If the change is breaking, bump the envelope version instead: spec Section 30.1 rule 8
requires a version bump and a migration for the trajectory schema.
"""

from __future__ import annotations

import sys

from psd.core.schemas.registry import (
    PUBLISHED,
    SCHEMA_DIR,
    generate,
    schema_filename,
    serialize,
)


def main() -> int:
    for schema_version in sorted(PUBLISHED):
        path = SCHEMA_DIR / schema_filename(schema_version)
        rendered = serialize(generate(schema_version))
        before = path.read_text(encoding="utf-8") if path.is_file() else None
        path.write_text(rendered, encoding="utf-8")
        state = "unchanged" if before == rendered else ("created" if before is None else "UPDATED")
        print(f"{schema_version:20} {state:10} {path.relative_to(SCHEMA_DIR.parents[3])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
