-- TASK-013: splits, with disjointness enforced by the database (spec Section 17.2).
--
-- The application already refuses an overlapping split in the `Split` model validator.
-- This trigger exists because that is not enough: an application-layer check protects
-- against the code path you thought of, and a constraint protects against every path,
-- including a manual UPDATE during an incident at 3am.
--
-- Leakage is the failure this guards. It has no symptom: a contaminated split produces
-- BETTER held-out numbers and reads as a success. There is nothing to notice later, so
-- the check has to be structural.

BEGIN;

CREATE TABLE IF NOT EXISTS splits (
  split_id       TEXT PRIMARY KEY,
  domain_id      TEXT NOT NULL REFERENCES domains ON DELETE RESTRICT,
  sha256         TEXT NOT NULL UNIQUE,           -- content address; immutable
  train_task_ids TEXT[] NOT NULL,
  test_task_ids  TEXT[] NOT NULL,
  strategy       TEXT NOT NULL,
  seed           BIGINT,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  CHECK (cardinality(train_task_ids) > 0 AND cardinality(test_task_ids) > 0)
);

CREATE OR REPLACE FUNCTION split_disjoint() RETURNS trigger AS $$
BEGIN
  IF EXISTS (SELECT 1 FROM unnest(NEW.train_task_ids) t
             WHERE t = ANY (NEW.test_task_ids)) THEN
    RAISE EXCEPTION 'split % has overlapping train/test task ids', NEW.split_id;
  END IF;
  RETURN NEW;
END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_split_disjoint ON splits;
CREATE TRIGGER trg_split_disjoint BEFORE INSERT OR UPDATE ON splits
  FOR EACH ROW EXECUTE FUNCTION split_disjoint();

-- Splits are immutable once written. Content addressing is meaningless if the content
-- can change under the address.
CREATE OR REPLACE FUNCTION splits_are_immutable() RETURNS trigger AS $$
BEGIN
  RAISE EXCEPTION 'splits are immutable; split % cannot be updated or deleted', OLD.split_id;
END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_splits_immutable ON splits;
CREATE TRIGGER trg_splits_immutable BEFORE UPDATE OR DELETE ON splits
  FOR EACH ROW EXECUTE FUNCTION splits_are_immutable();

COMMIT;
