"""Apply SQL migrations in order (TASK-013 support).

Deliberately small. Migrations are plain `.sql` files applied in filename order and
recorded in `schema_migrations`, because the schema in spec Section 17.2 is the contract
and an ORM's opinion about it would be a second, competing source of truth.

Idempotent: a migration already recorded is skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
  filename   TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def migration_files(directory: Path = MIGRATIONS_DIR) -> list[Path]:
    """Filename order is apply order, which is why they are numbered."""
    return sorted(directory.glob("*.sql"))


def apply_migrations(dsn: str, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply every unapplied migration. Returns the filenames applied this run."""
    applied: list[str] = []
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(_BOOTSTRAP)
        already = {
            row[0]
            for row in connection.execute("SELECT filename FROM schema_migrations").fetchall()
        }
        for path in migration_files(directory):
            if path.name in already:
                continue
            connection.execute(path.read_text(encoding="utf-8"))
            connection.execute("INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,))
            applied.append(path.name)
    return applied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", required=True, help="PostgreSQL connection string")
    args = parser.parse_args(argv)

    applied = apply_migrations(args.dsn)
    if applied:
        for name in applied:
            print(f"applied {name}")
    else:
        print("no migrations to apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
