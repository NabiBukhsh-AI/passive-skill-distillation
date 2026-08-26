-- Instruction registry and the audit log (TASK-027, spec Section 17.2).
--
-- `P` is the method. GAP-01 records that the paper never published it, so ours is a
-- reconstruction (DEV-001) and every result the platform produces is conditional on this
-- exact text. That is why the body is stored, content-addressed, and immutable rather
-- than referenced by version string alone: a version string can be re-pointed, and a
-- result whose instruction cannot be recovered byte for byte is not reproducible.

BEGIN;

CREATE TABLE IF NOT EXISTS instructions (
  instruction_id TEXT PRIMARY KEY,
  version        TEXT NOT NULL UNIQUE,          -- 'P/1.3'
  body           TEXT NOT NULL,
  sha256         TEXT NOT NULL UNIQUE,
  notes          TEXT,
  created_by     TEXT NOT NULL,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Registered instructions are immutable. Editing the body under a registered version
-- would silently invalidate every skill distilled with it, and nothing downstream would
-- show a symptom: the lineage record would still point at a version that now says
-- something different.
CREATE OR REPLACE FUNCTION instructions_are_immutable() RETURNS trigger AS $$
BEGIN
  RAISE EXCEPTION
    'instruction % is immutable; register a new version instead', OLD.version;
END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_instructions_immutable ON instructions;
CREATE TRIGGER trg_instructions_immutable BEFORE UPDATE OR DELETE ON instructions
  FOR EACH ROW EXECUTE FUNCTION instructions_are_immutable();

-- Append-only audit log (spec Section 17.2). TASK-064 revokes UPDATE and DELETE and adds
-- the WORM export; this migration creates the table so registrations are recorded from
-- the moment the registry exists rather than retrofitted later.
CREATE TABLE IF NOT EXISTS audit_log (
  audit_id      BIGSERIAL PRIMARY KEY,
  tenant_id     TEXT NOT NULL,
  principal     TEXT NOT NULL,
  action        TEXT NOT NULL,
  resource_type TEXT NOT NULL,
  resource_id   TEXT NOT NULL,
  before        JSONB,
  after         JSONB,
  request_id    TEXT,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_audit_resource ON audit_log (resource_type, resource_id);
CREATE INDEX IF NOT EXISTS ix_audit_tenant_time ON audit_log (tenant_id, created_at DESC);

COMMIT;
