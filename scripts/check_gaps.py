"""Validate docs/GAPS.md (TASK-001 acceptance check).

Fails on any empty field, any unknown status, any missing or duplicated GAP id, and any
row marked ``blocked`` that does not name its blocker. Run by ``make check`` and by CI.

This script deliberately has no third-party dependencies so it can run before, and
independently of, the project environment.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

EXPECTED_IDS = [f"GAP-{n:02d}" for n in range(1, 16)]
VALID_STATUSES = {"resolved", "pinned-default", "blocked"}
COLUMNS = ["id", "item", "status", "owner", "config_key", "plan", "blocked_by"]
EMPTY = {"", "-", "tbd", "todo", "n/a", "none"}

ROW_RE = re.compile(r"^\|(?P<body>.*)\|\s*$")


def _split_row(line: str) -> list[str] | None:
    match = ROW_RE.match(line)
    if match is None:
        return None
    return [cell.strip() for cell in match.group("body").split("|")]


def parse_rows(text: str) -> list[dict[str, str]]:
    """Return the register rows, that is every table row whose first cell is a GAP id."""
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        cells = _split_row(line)
        if cells is None or len(cells) != len(COLUMNS):
            continue
        if not cells[0].startswith("GAP-"):
            continue
        rows.append(dict(zip(COLUMNS, cells, strict=True)))
    return rows


def check(path: Path) -> list[str]:
    if not path.is_file():
        return [f"{path} does not exist"]

    rows = parse_rows(path.read_text(encoding="utf-8"))
    errors: list[str] = []

    seen = [row["id"] for row in rows]
    for gap_id in EXPECTED_IDS:
        if gap_id not in seen:
            errors.append(f"{gap_id} is missing from the register")
    for gap_id in sorted(set(seen)):
        if seen.count(gap_id) > 1:
            errors.append(f"{gap_id} appears {seen.count(gap_id)} times; ids must be unique")
    for gap_id in seen:
        if gap_id not in EXPECTED_IDS:
            errors.append(f"{gap_id} is not a known gap id")

    for row in rows:
        gap_id = row["id"]
        status = row["status"]

        if status not in VALID_STATUSES:
            errors.append(f"{gap_id}: status {status!r} is not one of {sorted(VALID_STATUSES)}")

        # Every gap must name a status, an owner, a config key, and a plan (TASK-001).
        for field in ("item", "status", "owner", "config_key", "plan"):
            if row[field].strip().lower() in EMPTY:
                errors.append(f"{gap_id}: field {field!r} is empty")

        # A blocked gap must name the external dependency that blocks it.
        if status == "blocked" and row["blocked_by"].strip().lower() in EMPTY:
            errors.append(f"{gap_id}: status is 'blocked' but no blocker is named")
        if status != "blocked" and row["blocked_by"].strip().lower() not in EMPTY:
            errors.append(
                f"{gap_id}: status is {status!r} but a blocker is named; "
                "only blocked gaps carry a blocker"
            )

    return errors


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    errors = check(root / "docs" / "GAPS.md")
    if errors:
        print("docs/GAPS.md failed validation:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    print(f"docs/GAPS.md OK: {len(EXPECTED_IDS)} gaps, all fields populated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
