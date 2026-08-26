"""Content-addressed instruction registry (TASK-027).

METHOD BOUNDARY. `src/psd/distill/instructions/*` is the distillation method itself
(spec Section 18.1, Section 30.1 rule 12).

FR-023 requires `P` to be a versioned, content-addressed registry artifact and never
inlined in code. The reason is GAP-01: the paper does not publish `P`, so ours will be
iterated, and a result whose instruction cannot be recovered byte for byte afterwards is
not reproducible.

Registration records the exact bytes, not a reference to a file that may later change,
and writes an audit record. Registered instructions are immutable at the database layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from psd.core.ports import Instruction
from psd.distill.instructions.loader import content_sha256


class InstructionConflictError(RuntimeError):
    """A version already exists with different content.

    Not an "already registered" no-op: re-registering `P/0.1` with a different body is
    exactly the mistake that silently invalidates every skill distilled from the original.
    """


@dataclass
class InstructionRegistry:
    """Postgres-backed registry for versions of `P`."""

    connection: Any
    tenant_id: str = "t_default"

    def register(
        self,
        instruction: Instruction,
        *,
        principal: str,
        notes: str | None = None,
        request_id: str | None = None,
    ) -> str:
        """Register a version. Idempotent on identical content, conflicting otherwise."""
        digest = content_sha256(instruction.text)
        if digest != instruction.sha256:
            raise ValueError(
                f"instruction {instruction.version} carries sha256 {instruction.sha256} "
                f"but its body hashes to {digest}"
            )

        existing = self.get(instruction.version)
        if existing is not None:
            if existing["sha256"] == digest:
                return str(existing["instruction_id"])
            raise InstructionConflictError(
                f"{instruction.version} is already registered with sha256 "
                f"{existing['sha256']}, which differs from {digest}. Instructions are "
                "immutable: register a new version instead. Re-pointing a version would "
                "invalidate every skill distilled from the original, and nothing "
                "downstream would show a symptom."
            )

        instruction_id = f"ins_{digest[:16]}"
        self.connection.execute(
            """
            INSERT INTO instructions
                (instruction_id, version, body, sha256, notes, created_by)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (instruction_id, instruction.version, instruction.text, digest, notes, principal),
        )
        self._audit(
            principal=principal,
            action="instruction.register",
            resource_id=instruction_id,
            after={
                "version": instruction.version,
                "sha256": digest,
                "lines": len(instruction.text.split("\n")),
                "notes": notes,
            },
            request_id=request_id,
        )
        return instruction_id

    def get(self, version: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT instruction_id, version, body, sha256, notes, created_by
            FROM instructions WHERE version = %s
            """,
            (version,),
        ).fetchone()
        if row is None:
            return None
        return {
            "instruction_id": row[0],
            "version": row[1],
            "body": row[2],
            "sha256": row[3],
            "notes": row[4],
            "created_by": row[5],
        }

    def get_by_hash(self, sha256: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT version FROM instructions WHERE sha256 = %s", (sha256,)
        ).fetchone()
        return self.get(row[0]) if row else None

    def load(self, version: str) -> Instruction:
        """Recover an instruction from the registry, byte for byte.

        This is what makes a distillation run reproducible: the run manifest records a
        hash, and this returns the exact text that hash addresses.
        """
        record = self.get(version)
        if record is None:
            raise KeyError(f"no registered instruction {version!r}")
        return Instruction(version=record["version"], sha256=record["sha256"], text=record["body"])

    def versions(self) -> list[str]:
        rows = self.connection.execute(
            "SELECT version FROM instructions ORDER BY version"
        ).fetchall()
        return [row[0] for row in rows]

    def audit_records(self, resource_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT principal, action, resource_type, resource_id, after, request_id
            FROM audit_log
            WHERE resource_type = 'instruction' AND resource_id = %s
            ORDER BY audit_id
            """,
            (resource_id,),
        ).fetchall()
        return [
            {
                "principal": row[0],
                "action": row[1],
                "resource_type": row[2],
                "resource_id": row[3],
                "after": row[4],
                "request_id": row[5],
            }
            for row in rows
        ]

    def _audit(
        self,
        *,
        principal: str,
        action: str,
        resource_id: str,
        after: dict[str, Any],
        request_id: str | None,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO audit_log
                (tenant_id, principal, action, resource_type, resource_id, after, request_id)
            VALUES (%s, %s, %s, 'instruction', %s, %s, %s)
            """,
            (self.tenant_id, principal, action, resource_id, json.dumps(after), request_id),
        )
