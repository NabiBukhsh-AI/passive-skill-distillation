# Passive Skill Distillation Platform.
#
# `make check` is the gate. Spec Section 30.1 rule 7: a task is not complete until every
# acceptance criterion is asserted by a passing, non-skipped test.

UV ?= uv

# The interpreter is pinned here rather than in a .python-version file, because the
# supplied .gitignore excludes .python-version and a pin that cannot be committed is not a
# pin. 3.12 rather than spec Section 13.3's 3.11.9: tau2-bench cannot install on 3.11.
# See docs/DEVIATIONS.md DEV-006.
UV_PYTHON ?= 3.12

.PHONY: setup lint format typecheck gaps imports test test-security check \
        reproduce-r0 reproduce-r1 clean help

help:
	@echo "setup          install the locked environment and git hooks"
	@echo "lint           ruff check and format check"
	@echo "format         ruff format, in place"
	@echo "typecheck      mypy, strict on src/psd/core"
	@echo "gaps           validate docs/GAPS.md (TASK-001)"
	@echo "imports        enforce the architecture rules (TASK-007)"
	@echo "test           unit, property, and contract suites"
	@echo "test-security  blocking security suite, never skip"
	@echo "check          lint + typecheck + gaps + imports + test"
	@echo "reproduce-r0   smoke reproduction, stub distiller"
	@echo "reproduce-r1   core reproduction, real distiller"

setup:
	$(UV) sync --all-groups --python $(UV_PYTHON)
	$(UV) run pre-commit install

lint:
	$(UV) run ruff check src tests scripts
	$(UV) run ruff format --check src tests scripts

format:
	$(UV) run ruff format src tests scripts
	$(UV) run ruff check --fix src tests scripts

typecheck:
	$(UV) run mypy

gaps:
	$(UV) run python scripts/check_gaps.py

imports:
	$(UV) run lint-imports --config .import-linter

test:
	$(UV) run pytest tests/unit tests/property tests/contract

# Blocking. Spec Section 21.1 and TASK-069: this suite may never be skipped or xfailed,
# and may not be disabled by configuration.
test-security:
	$(UV) run pytest tests/security

check: lint typecheck gaps imports test

reproduce-r0:
	$(UV) run python scripts/reproduce.py r0

reproduce-r1:
	$(UV) run python scripts/reproduce.py r1

clean:
	$(UV) run python -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache','.mypy_cache','.ruff_cache','htmlcov']]"
