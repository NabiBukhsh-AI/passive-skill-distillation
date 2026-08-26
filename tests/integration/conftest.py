"""A real PostgreSQL for the integration suite.

TASK-013's acceptance criterion is that an overlapping split raises **at the database
layer, not only in Python**. That cannot be demonstrated against a mock or an in-memory
substitute: the thing under test is a plpgsql trigger.

`pgserver` ships PostgreSQL 16 binaries in a wheel, so the suite runs with no admin
rights, no container runtime, and no service to install. That matters here because this
machine has none of those available.

One server per session, one fresh schema per test, so tests cannot see each other's rows
while still paying the startup cost only once.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


@pytest.fixture(scope="session")
def postgres_uri(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    pgserver = pytest.importorskip(
        "pgserver", reason="pgserver is required for the database integration suite"
    )
    data_dir = tmp_path_factory.mktemp("pgdata")
    server = pgserver.get_server(str(data_dir))
    try:
        yield str(server.get_uri())
    finally:
        server.cleanup()


@pytest.fixture
def connection(postgres_uri: str) -> Iterator[Any]:
    """A connection to a freshly migrated, isolated schema."""
    import psycopg
    from migrate import apply_migrations  # scripts/migrate.py

    apply_migrations(postgres_uri)

    with psycopg.connect(postgres_uri, autocommit=True) as conn:
        # Each test gets its own rows. Truncating rather than recreating the schema keeps
        # the migration cost at once per session.
        conn.execute("TRUNCATE splits, domains, tenants RESTART IDENTITY CASCADE")
        conn.execute("INSERT INTO tenants (tenant_id, name) VALUES ('t_test', 'test tenant')")
        for domain in ("alfworld", "tau2_retail"):
            conn.execute(
                "INSERT INTO domains (domain_id, tenant_id, display_name) VALUES (%s, %s, %s)",
                (domain, "t_test", domain),
            )
        yield conn
