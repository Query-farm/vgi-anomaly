# CLAUDE.md — vgi-anomaly

Contributor/agent notes. User-facing docs live in `README.md`; this is the
"how it's built and where the sharp edges are" companion.

## What this is

A [VGI](https://query.farm) worker that does **time-series anomaly detection** —
matrix-profile motifs/discords (`stumpy`, BSD-3), change-point detection
(`ruptures`, BSD-2), and a light z-score check (`numpy`, BSD-3) — as DuckDB
scalar functions. `anomaly_worker.py` assembles every function into one `anomaly`
catalog (single `main` schema) over stdio. Sibling style/tooling to `vgi-conform`
and `vgi-calendar`.

## Layout

```
anomaly_worker.py      repo-root stdio entry point; PEP 723 inline deps; main();
                       overrides run() to numba-warm-up before serving
vgi_anomaly/
  detectors.py         pure detection logic over stumpy/ruptures/numpy; no Arrow/VGI;
                       unit-testable. Also holds warm_up() and MAX_SERIES_LEN.
  scalars.py           per-row LIST-in scalars (change_points has an arity overload)
tests/                 pytest: test_detectors (pure), test_scalars (Client RPC)
test/sql/*.test        haybarn-unittest sqllogictest — authoritative E2E
Makefile               test / test-unit / test-sql / lint
```

To add a function: implement the logic in `detectors.py` (pure, total — returns
`None` for an invalid/short series, raises `ValueError` only for a bad *constant*
parameter), wrap it as a scalar in `scalars.py`, register it in
`anomaly_worker.py`'s `SCALAR_FUNCTIONS`.

## The series-as-argument design — THE core convention (read first)

Anomaly detection is defined over a *whole series*, but a scalar sees one row at
a time. So each function takes the series as a single `DOUBLE[]` **argument** the
caller builds in SQL (`array_agg(value ORDER BY t)` or a `[...]::DOUBLE[]`
literal) and returns an index or an array. This is the LIST-in / scalar-or-LIST-out
shape — it composes without subquery table args and runs once per group.

Two hard requirements from the SDK for LIST types:

1. A `DOUBLE[]` **parameter** needs an explicit `arrow_type` in `Param(...)`:
   `Param(arrow_type=pa.list_(pa.float64()), doc=...)`. Omitting it raises
   `TypeError: ListArray requires explicit arrow_type in Param()` at class
   definition (worker fails to import → bind fails with an opaque transport
   error).
2. A `DOUBLE[]` / `BIGINT[]` **return** needs an explicit
   `Returns(arrow_type=pa.list_(pa.float64()))` / `pa.list_(pa.int64())`.

## Scalars are positional-only

`name := value` named args are rejected for scalars. `change_points`'s optional
`n_bkps` is therefore a second `ScalarFunction` subclass sharing `Meta.name`
(`change_points(values)` / `change_points(values, n_bkps)`) — same idiom as
vgi-conform's region overloads. Const args (`window`, `n_bkps`, `threshold`)
arrive via `ConstParam`; pass them as `pa.scalar(x, type=pa.int64())` /
`pa.float64()` from the test client.

## Robustness contract

- A series that is NULL / empty / too short / contains a NULL or non-finite
  sample → the detector returns `None` → SQL `NULL`, per row, never an error.
  `_clean()` in `detectors.py` is the single chokepoint for this.
- A bad **constant** parameter (window `< 3` or `>= len`, `n_bkps` out of
  `[1, len)`, `threshold <= 0`) raises `ValueError` → a clear SQL error. These
  are identical across the whole batch, so surfacing them (rather than NULLing)
  is correct. `_map_series` in `scalars.py` deliberately catches nothing.
- Series longer than `MAX_SERIES_LEN` (1e6) raise — the matrix profile is O(n²).

## numba warm-up (don't remove)

`stumpy.stump` is numba-JIT compiled: the FIRST call in a fresh process compiles
(multi-second). `AnomalyWorker.__init__()` calls `detectors.warm_up()` before
serving, running each entry point once on a tiny dummy series. It lives in
`__init__` (NOT `run()`) on purpose: the stdio path calls `run()`, but the
`--unix` / `--tcp` launcher paths the vgi extension uses to spawn a command
`LOCATION` build the RPC server and serve directly and never call `run()`. Every
transport instantiates the worker, so `__init__` is the one hook that always
moves the compile to process spawn — the first real query is fast, the E2E suite
doesn't flake under load (a teardown SIGTERM landing during a mid-query compile
records a spurious failure), and the linter's slow-example gate stays green. It
only warms JIT caches — never changes output — and is best-effort (failures
swallowed). Heavy libs are imported once at module load.

`stumpy.stump` is also a numba `parallel=True` kernel and numba's default
workqueue threading layer aborts the process if entered from two threads at once;
`detectors._profile` serializes every stumpy call behind `_STUMP_LOCK` so
concurrent DuckDB/lint cursors can't crash the worker.

## Sharp edges (learned the hard way)

1. **`haybarn-unittest` skips `require vgi` silently.** Use an explicit
   `statement ok` / `LOAD vgi;` instead. Every `.test` file: header `# name:` +
   `# group: [vgi_anomaly]`, `require-env VGI_ANOMALY_WORKER`, then
   `ATTACH 'anomaly' AS anomaly (TYPE vgi, LOCATION '${VGI_ANOMALY_WORKER}');`.
   Catalog-qualify every scalar (`anomaly.discord_index(...)`).
2. **Determinism.** Results are deterministic given the input series + params, so
   the E2E tests assert exact indices. For a clean step, PELT auto and Dynp both
   return the exact change index. Assert profile *values* (if ever needed) with
   `ROUND`/tolerance, not exact equality.
3. **Change-point penalty.** `model="rbf"` cost is ~unit-scale regardless of
   amplitude, so the auto penalty is plain `log(n)` — do NOT scale it by variance
   (that over-penalizes large steps and misses them). See `_auto_penalty`.
4. **If `make test-sql` flakes**, re-run 2–3×; only a *consistent* failure is
   real (numba/host-load timing).

## Verify

```sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra dev
uv run --no-sync pytest -q
make test-sql
uv run --no-sync ruff check . && uv run --no-sync mypy vgi_anomaly/
```
