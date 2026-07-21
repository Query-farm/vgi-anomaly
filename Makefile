# vgi-anomaly — dev and test targets.
#
# Usage:
#   make test       # unit (pytest) + end-to-end SQL (haybarn-unittest)
#   make test-unit  # pytest only
#   make test-sql   # DuckDB sqllogictest .test files via haybarn-unittest
#
# test-sql is self-contained: it points VGI_ANOMALY_WORKER at the worker run as
# a uv stdio subprocess (exactly how DuckDB drives it after ATTACH) and runs the
# files under test/sql/. haybarn-unittest is a uv tool:
#   uv tool install haybarn-unittest   # installs ~/.local/bin/haybarn-unittest

# Worker command DuckDB uses for ATTACH (overridable). Use the project venv's
# interpreter (populated by `uv sync`) rather than `uv run`: `uv run` on the PEP
# 723 script re-resolves its inline deps into a separate, cacheable environment
# that can pin a stale vgi-python and present the old catalog schema on ATTACH,
# whereas the venv is the exact frozen-locked SDK the gates test against.
WORKER_STDIO    ?= $(CURDIR)/.venv/bin/python $(CURDIR)/anomaly_worker.py

# haybarn-unittest lives in the uv tools bin; keep it on PATH.
HAYBARN_BIN     ?= $(HOME)/.local/bin
TEST_DIR         = .
TEST_PATTERN     = test/sql/*

.PHONY: test test-unit test-sql lint

test: test-unit test-sql

test-unit:
	uv run pytest -q

test-sql:
	PATH="$(HAYBARN_BIN):$$PATH" \
		VGI_ANOMALY_WORKER="$(WORKER_STDIO)" \
		haybarn-unittest --test-dir "$(TEST_DIR)" "$(TEST_PATTERN)"

lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy vgi_anomaly/
