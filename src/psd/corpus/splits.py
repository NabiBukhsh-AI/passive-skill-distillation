"""Immutable, content-addressed train/test splits (TASK-013, spec Section 10.4).

`T_train` and `T_test` are disjoint (spec Section 5.3), and FR-006 requires the system to
refuse any distillation run whose corpus contains a test-split task id.

Disjointness is enforced at three layers, on purpose:

  1. Here, in the `Split` model validator (`psd.core.models`).
  2. In the database, by the `trg_split_disjoint` trigger of spec Section 17.2
     (`migrations/0001_splits.sql`).
  3. At corpus build, by the hard abort in ALG-001 Step 5.

Three layers is not belt-and-braces paranoia. Leakage into a prompt is invisible: a
contaminated skill produces better held-out numbers and looks like a success. There is no
symptom to notice later, so the check has to be structural.

The split is content-addressed and write-once. GAP-09 records that the paper never
published its held-out task ids, so exact numeric reproduction is impossible; what this
artifact buys is that OUR runs stay comparable to each other forever.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from psd.core.models import Split, SplitCounts, SplitSampling

#: Spec Section 10.4: sampled once and fixed across all conditions.
STRATEGY_RANDOM_ONCE_FIXED = "random_once_fixed"
STRATEGY_FROM_UPSTREAM = "from_upstream"


def canonical_bytes(
    domain: str,
    sampling: SplitSampling,
    train_task_ids: list[str],
    test_task_ids: list[str],
) -> bytes:
    """The bytes a split's sha256 is taken over.

    Deliberately excludes `created_at`: two people running the same command with the same
    seed must get the same content address, and a wall-clock timestamp would make the
    hash a function of when you ran it. `sorted` on both id lists for the same reason.
    """
    payload = {
        "schema_version": "split/1.0",
        "domain": domain,
        "sampling": {
            "strategy": sampling.strategy,
            "seed": sampling.seed,
        },
        "train_task_ids": sorted(train_task_ids),
        "test_task_ids": sorted(test_task_ids),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_sha256(
    domain: str,
    sampling: SplitSampling,
    train_task_ids: list[str],
    test_task_ids: list[str],
) -> str:
    return hashlib.sha256(
        canonical_bytes(domain, sampling, train_task_ids, test_task_ids)
    ).hexdigest()


def build_split(
    domain: str,
    task_ids: list[str],
    train_size: int,
    test_size: int,
    *,
    seed: int,
    strategy: str = STRATEGY_RANDOM_ONCE_FIXED,
    notes: str | None = None,
    created_at: datetime | None = None,
) -> Split:
    """Sample a fresh split from a task pool.

    Deterministic given the same pool and seed. The pool is sorted before sampling, so a
    caller that hands us the same ids in a different order still gets the same split.
    """
    pool = sorted(set(task_ids))
    if len(pool) < train_size + test_size:
        raise ValueError(
            f"domain {domain!r} has {len(pool)} distinct tasks but the split needs "
            f"{train_size + test_size} ({train_size} train, {test_size} test)"
        )

    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    train = sorted(shuffled[:train_size])
    test = sorted(shuffled[train_size : train_size + test_size])

    sampling = SplitSampling(strategy=strategy, seed=seed, notes=notes)
    return Split(
        domain=domain,
        created_at=created_at or datetime.now(UTC),
        sampling=sampling,
        train_task_ids=train,
        test_task_ids=test,
        counts=SplitCounts(train=len(train), test=len(test)),
        sha256=compute_sha256(domain, sampling, train, test),
    )


def split_from_upstream(
    domain: str,
    train_task_ids: list[str],
    test_task_ids: list[str],
    *,
    notes: str | None = None,
    created_at: datetime | None = None,
) -> Split:
    """Wrap a split whose held-out half the benchmark provides.

    tau2-bench ships `split_tasks.json` per domain, which spec Section 13.5's
    `psd split import --test-from-upstream` refers to. Seed is 0 because nothing was
    sampled: recording a seed here would imply a choice we did not make.
    """
    sampling = SplitSampling(strategy=STRATEGY_FROM_UPSTREAM, seed=0, notes=notes)
    train = sorted(set(train_task_ids))
    test = sorted(set(test_task_ids))
    return Split(
        domain=domain,
        created_at=created_at or datetime.now(UTC),
        sampling=sampling,
        train_task_ids=train,
        test_task_ids=test,
        counts=SplitCounts(train=len(train), test=len(test)),
        sha256=compute_sha256(domain, sampling, train, test),
    )


def verify(split: Split) -> None:
    """Raise if the recorded content address does not match the content.

    ALG-001 Step 1 asserts this before a corpus is built from a split. A split whose hash
    does not match has been edited, and an edited split is exactly how a test task ends
    up in a training corpus.
    """
    expected = compute_sha256(
        split.domain, split.sampling, split.train_task_ids, split.test_task_ids
    )
    if expected != split.sha256:
        raise ValueError(
            f"split hash mismatch for domain {split.domain!r}: "
            f"recorded {split.sha256}, computed {expected}. The artifact was modified "
            "after creation; splits are immutable."
        )


def write_split(split: Split, directory: Path) -> Path:
    """Write a split artifact write-once, named by its content address."""
    verify(split)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{split.domain}-{split.sha256[:12]}.json"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if json.loads(existing)["sha256"] != split.sha256:
            raise FileExistsError(f"{path} exists with different content")
        return path
    path.write_text(split.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_split(path: Path) -> Split:
    split = Split.model_validate_json(path.read_text(encoding="utf-8"))
    verify(split)
    return split


def task_split_map(split: Split) -> dict[str, str]:
    """Map every task id to its split name, for the normalizer's agreement check."""
    mapping = dict.fromkeys(split.train_task_ids, "train")
    mapping.update(dict.fromkeys(split.test_task_ids, "test"))
    return mapping


# ---------------------------------------------------------------------------
# Database-backed store (TASK-013)
# ---------------------------------------------------------------------------


class SplitStore:
    """Persist splits to PostgreSQL, where the database enforces disjointness.

    The `Split` model already refuses an overlapping split, so this looks redundant. It
    is not. An application-layer check protects the code path you thought of; a database
    constraint protects every path, including a manual `UPDATE` during an incident and a
    future service that talks to the same table without going through this module.

    TASK-013's acceptance is specifically that an overlapping split "raises at the
    database layer, not only in Python", so the store deliberately does NOT re-validate
    before inserting: the test needs the insert to reach the trigger.
    """

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def insert(self, split: Split, split_id: str, tenant_domain: str | None = None) -> str:
        """Insert a split. Raises whatever the database raises."""
        self._connection.execute(
            """
            INSERT INTO splits
                (split_id, domain_id, sha256, train_task_ids, test_task_ids, strategy, seed)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                split_id,
                tenant_domain or split.domain,
                split.sha256,
                list(split.train_task_ids),
                list(split.test_task_ids),
                split.sampling.strategy,
                split.sampling.seed,
            ),
        )
        return split_id

    def insert_raw(
        self,
        split_id: str,
        domain_id: str,
        sha256: str,
        train_task_ids: list[str],
        test_task_ids: list[str],
        strategy: str = "random_once_fixed",
        seed: int = 0,
    ) -> str:
        """Insert without constructing a `Split` first.

        Exists so the database-level constraint can be tested directly. Building a
        `Split` with overlapping halves is impossible (the model validator refuses), so
        without this path there would be no way to demonstrate that the trigger fires,
        and TASK-013's acceptance criterion would be untestable.
        """
        self._connection.execute(
            """
            INSERT INTO splits
                (split_id, domain_id, sha256, train_task_ids, test_task_ids, strategy, seed)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (split_id, domain_id, sha256, train_task_ids, test_task_ids, strategy, seed),
        )
        return split_id

    def get(self, split_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            """
            SELECT split_id, domain_id, sha256, train_task_ids, test_task_ids, strategy, seed
            FROM splits WHERE split_id = %s
            """,
            (split_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "split_id": row[0],
            "domain_id": row[1],
            "sha256": row[2],
            "train_task_ids": row[3],
            "test_task_ids": row[4],
            "strategy": row[5],
            "seed": row[6],
        }
