"""TASK-013 acceptance: the split disjointness constraint lives in the DATABASE.

The criterion is stated precisely: "creating an overlapping split raises at the database
layer, not only in Python". So these tests bypass the `Split` model entirely and insert
raw rows, because the model refuses to build an overlapping split at all. If they went
through the model, they would be testing the Python check a second time and proving
nothing about the trigger.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from psd.corpus.splits import SplitStore, split_from_upstream

pytestmark = pytest.mark.integration

FIXED_TIME = datetime(2026, 8, 1, tzinfo=UTC)
HASH_A = "a" * 64
HASH_B = "b" * 64


def store(connection: Any) -> SplitStore:
    return SplitStore(connection)


# ---------------------------------------------------------------------------
# The acceptance criterion
# ---------------------------------------------------------------------------


def test_the_database_rejects_an_overlapping_split(connection: Any) -> None:
    """TASK-013 acceptance.

    `t2` is in both halves. Python never sees this row shape, so the only thing that can
    reject it is `trg_split_disjoint`.
    """
    with pytest.raises(psycopg.errors.RaiseException) as exc:
        store(connection).insert_raw(
            split_id="spl_overlap",
            domain_id="alfworld",
            sha256=HASH_A,
            train_task_ids=["t1", "t2"],
            test_task_ids=["t2", "t3"],
        )
    assert "overlapping train/test task ids" in str(exc.value)


def test_a_disjoint_split_inserts_cleanly(connection: Any) -> None:
    """Negative control. Without it, a trigger that rejected everything would pass."""
    store(connection).insert_raw(
        split_id="spl_ok",
        domain_id="alfworld",
        sha256=HASH_A,
        train_task_ids=["t1", "t2"],
        test_task_ids=["t3", "t4"],
    )
    row = store(connection).get("spl_ok")
    assert row is not None
    assert row["train_task_ids"] == ["t1", "t2"]
    assert row["test_task_ids"] == ["t3", "t4"]


def test_the_trigger_also_fires_on_update(connection: Any) -> None:
    """`BEFORE INSERT OR UPDATE`. An overlap introduced later is still an overlap.

    This is the path an application-layer check cannot cover: a manual UPDATE during an
    incident never passes through the Python validator.
    """
    store(connection).insert_raw(
        split_id="spl_update",
        domain_id="alfworld",
        sha256=HASH_A,
        train_task_ids=["t1"],
        test_task_ids=["t2"],
    )
    with pytest.raises(psycopg.errors.RaiseException, match="overlapping"):
        connection.execute(
            "UPDATE splits SET train_task_ids = %s WHERE split_id = %s",
            (["t1", "t2"], "spl_update"),
        )


def test_splits_are_immutable_at_the_database_layer(connection: Any) -> None:
    """Content addressing is meaningless if the content can change under the address."""
    store(connection).insert_raw(
        split_id="spl_immutable",
        domain_id="alfworld",
        sha256=HASH_A,
        train_task_ids=["t1"],
        test_task_ids=["t2"],
    )
    with pytest.raises(psycopg.errors.RaiseException, match="immutable"):
        connection.execute(
            "UPDATE splits SET strategy = 'tampered' WHERE split_id = %s", ("spl_immutable",)
        )
    with pytest.raises(psycopg.errors.RaiseException, match="immutable"):
        connection.execute("DELETE FROM splits WHERE split_id = %s", ("spl_immutable",))


def test_content_address_is_unique(connection: Any) -> None:
    """Two different split ids may not claim the same content."""
    store(connection).insert_raw(
        split_id="spl_first",
        domain_id="alfworld",
        sha256=HASH_A,
        train_task_ids=["t1"],
        test_task_ids=["t2"],
    )
    with pytest.raises(psycopg.errors.UniqueViolation):
        store(connection).insert_raw(
            split_id="spl_second",
            domain_id="alfworld",
            sha256=HASH_A,
            train_task_ids=["t9"],
            test_task_ids=["t8"],
        )


def test_empty_halves_are_rejected_by_the_check_constraint(connection: Any) -> None:
    with pytest.raises(psycopg.errors.CheckViolation):
        store(connection).insert_raw(
            split_id="spl_empty",
            domain_id="alfworld",
            sha256=HASH_B,
            train_task_ids=[],
            test_task_ids=["t1"],
        )


def test_an_unknown_domain_is_rejected_by_the_foreign_key(connection: Any) -> None:
    """Lineage: a split must belong to a domain that exists."""
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        store(connection).insert_raw(
            split_id="spl_nodomain",
            domain_id="a_domain_that_does_not_exist",
            sha256=HASH_B,
            train_task_ids=["t1"],
            test_task_ids=["t2"],
        )


# ---------------------------------------------------------------------------
# The model path still works end to end
# ---------------------------------------------------------------------------


def test_a_model_built_split_round_trips_through_the_database(connection: Any) -> None:
    split = split_from_upstream("tau2_retail", ["r1", "r2"], ["r3"], created_at=FIXED_TIME)
    store(connection).insert(split, split_id="spl_model")
    row = store(connection).get("spl_model")
    assert row is not None
    assert row["sha256"] == split.sha256
    assert row["train_task_ids"] == split.train_task_ids
    assert row["strategy"] == "from_upstream"


def test_both_layers_reject_the_same_overlap(connection: Any) -> None:
    """The point of defence in depth, stated as a test.

    Python refuses to construct the object; the database refuses to store the row. Either
    alone would be a single point of failure for a bug with no symptom.
    """
    with pytest.raises(ValueError, match="overlap"):
        split_from_upstream("alfworld", ["a", "b"], ["b"], created_at=FIXED_TIME)

    with pytest.raises(psycopg.errors.RaiseException, match="overlapping"):
        store(connection).insert_raw(
            split_id="spl_both",
            domain_id="alfworld",
            sha256=HASH_B,
            train_task_ids=["a", "b"],
            test_task_ids=["b"],
        )


def test_migrations_are_idempotent(postgres_uri: str) -> None:
    """Re-running the migrator applies nothing the second time."""
    from migrate import apply_migrations

    apply_migrations(postgres_uri)
    assert apply_migrations(postgres_uri) == []
