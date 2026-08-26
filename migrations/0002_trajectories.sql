-- Trajectory metadata (spec Section 17.2). Bodies live in object storage.
--
-- Note `reward DOUBLE PRECISION NOT NULL`. That NOT NULL is deliberate and is the
-- database's half of the rule in spec Section 10.3: a record with no reward is
-- quarantined, never stored with a defaulted zero. Quarantined records therefore never
-- reach this table at all; they go to a quarantine prefix in object storage with the
-- failing rule recorded, which is why there is no nullable-reward path here.

BEGIN;

CREATE TABLE IF NOT EXISTS trajectories (
  trajectory_id     TEXT NOT NULL,
  tenant_id         TEXT NOT NULL REFERENCES tenants,
  domain_id         TEXT NOT NULL REFERENCES domains,
  task_id           TEXT NOT NULL,
  split             TEXT NOT NULL CHECK (split IN ('train','test','unassigned')),
  actor_model       TEXT NOT NULL,
  actor_mode        TEXT NOT NULL CHECK (actor_mode IN ('think','no_think')),
  harness_version   TEXT NOT NULL,
  seed              INTEGER,
  reward            DOUBLE PRECISION NOT NULL,     -- NOT NULL is deliberate; see 10.3
  success           BOOLEAN NOT NULL,
  steps_used        INTEGER NOT NULL CHECK (steps_used >= 0),
  step_cap_hit      BOOLEAN NOT NULL DEFAULT false,
  output_tokens     INTEGER,
  reasoning_tokens  INTEGER,
  input_tokens      INTEGER,
  cost_usd          NUMERIC(12,6),
  error_types       TEXT[] NOT NULL DEFAULT '{}',
  stalled           BOOLEAN NOT NULL DEFAULT false,
  body_uri          TEXT NOT NULL,
  content_sha256    TEXT NOT NULL,
  redaction_version TEXT,
  quarantined       BOOLEAN NOT NULL DEFAULT false,
  quarantine_reason TEXT,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (trajectory_id, created_at),
  -- Idempotency key for ingestion (TASK-011). Duplicate delivery of identical content is
  -- a no-op rather than a second row.
  UNIQUE (tenant_id, domain_id, content_sha256, created_at)
) PARTITION BY RANGE (created_at);

-- Spec Section 17.3 partitions by month. A DEFAULT partition is created here so inserts
-- succeed before the partition-management job (TASK-074) exists; that task replaces this
-- with real monthly partitions.
CREATE TABLE IF NOT EXISTS trajectories_default PARTITION OF trajectories DEFAULT;

CREATE INDEX IF NOT EXISTS ix_traj_selection ON trajectories
  (tenant_id, domain_id, actor_model, actor_mode, split, quarantined);
CREATE INDEX IF NOT EXISTS ix_traj_task ON trajectories (domain_id, task_id);
CREATE INDEX IF NOT EXISTS ix_traj_errors ON trajectories USING GIN (error_types);
CREATE INDEX IF NOT EXISTS ix_traj_content ON trajectories (tenant_id, domain_id, content_sha256);

COMMIT;
