-- Tenancy and domains (spec Section 17.2).
--
-- Applied before 0001_splits.sql, which references both by foreign key. `ON DELETE
-- RESTRICT` throughout: lineage is the point of this schema, and a cascading delete would
-- quietly detach a skill from the corpus that produced it.

BEGIN;

CREATE TABLE IF NOT EXISTS tenants (
  tenant_id  TEXT PRIMARY KEY,
  name       TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS domains (
  domain_id    TEXT PRIMARY KEY,               -- 'alfworld', 'tau2_retail'
  tenant_id    TEXT NOT NULL REFERENCES tenants ON DELETE RESTRICT,
  display_name TEXT NOT NULL,
  profile      JSONB NOT NULL DEFAULT '{}'::jsonb,  -- canonicalization + analyzer profile
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (tenant_id, domain_id)
);

COMMIT;
