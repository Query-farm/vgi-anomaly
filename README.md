<p align="center">
  <img src="https://query.farm/media-kit/products/vgi/vgi-logo.png" alt="Vector Gateway Interface (VGI)" width="320">
</p>

<p align="center"><em>A <a href="https://query.farm">Query.Farm</a> VGI worker for DuckDB.</em></p>

# Time-Series Anomaly Detection — Motifs, Discords & Change Points in DuckDB

> **vgi-anomaly** · a [Query.Farm](https://query.farm) VGI worker · powered by STUMPY

[![CI](https://github.com/Query-farm/vgi-anomaly/actions/workflows/ci.yml/badge.svg)](https://github.com/Query-farm/vgi-anomaly/actions/workflows/ci.yml)

A [VGI](https://query.farm) worker that brings **time-series anomaly detection**
into DuckDB/SQL. It finds the anomalous and the repeated structure in a numeric
series — **matrix-profile motifs & discords**, **change points**, and a light
**z-score** outlier check — as SQL table functions over ordered rows, with
array-based scalar calls also available, backed by
battle-tested libraries: [`stumpy`](https://pypi.org/project/stumpy/) (BSD-3,
matrix profile), [`ruptures`](https://pypi.org/project/ruptures/) (BSD-2,
change-point detection) and [`numpy`](https://numpy.org/) (BSD-3).

```sql
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'anomaly' (TYPE vgi, LOCATION 'uvx --from git+https://github.com/Query-farm/vgi-anomaly@v0.5.0 vgi-anomaly-worker');

-- Pass an ordered query directly; results are ordinary rows.
SELECT * FROM anomaly.discord_index((SELECT v FROM series ORDER BY t), window_size := 50);
SELECT * FROM anomaly.motif_index((SELECT v FROM series ORDER BY t), window_size := 50);
SELECT * FROM anomaly.matrix_profile((SELECT v FROM series ORDER BY t), window_size := 50);
SELECT * FROM anomaly.change_points((SELECT v FROM series ORDER BY t));
SELECT * FROM anomaly.change_points((SELECT v FROM series ORDER BY t), n_bkps := 2);
SELECT * FROM anomaly.zscore_anomalies((SELECT v FROM series ORDER BY t), threshold := 3.0);

-- Existing scalar calls still work, including literal arrays and GROUP BY.
SELECT anomaly.discord_index([1.0,2.0,3.0,100.0,2.0,3.0,1.0]::DOUBLE[], 3);
```

Results are **deterministic** given the input series and parameters — the same
series always yields the same indices and profile, so the worker is hermetic and
easy to test.

The table interface was added in `v0.5.0`. Array-based scalar calls remain supported.

Explore the [executed anomaly-detection notebook](notebooks/README.md) for spikes, change points, and repeated patterns.

## Quick start

The worker isn't on PyPI; [uv](https://docs.astral.sh/uv/) runs it straight from a tagged release on GitHub. `uvx` builds an isolated environment on the first ATTACH and reuses it afterwards:

```sql
INSTALL vgi FROM community;
LOAD vgi;

ATTACH 'anomaly' AS anomaly (TYPE vgi, LOCATION 'uvx --from git+https://github.com/Query-farm/vgi-anomaly@v0.5.0 vgi-anomaly-worker');

-- Find the outlier in 19 zeros followed by 100: returns index 19.
SELECT * FROM anomaly.zscore_anomalies(
    (SELECT CASE WHEN i = 19 THEN 100.0 ELSE 0.0 END
     FROM range(20) t(i) ORDER BY i)
);

-- Find a regime change: returns index 30.
SELECT * FROM anomaly.change_points(
    (SELECT CASE WHEN i < 30 THEN 0.0 ELSE 10.0 END
     FROM range(60) t(i) ORDER BY i)
);
```

Change `@v0.5.0` to move to a newer release.

### Other ways to run it

- **Docker** (the image bundles the worker and its dependencies):

  ```sql
  ATTACH 'anomaly' AS anomaly (TYPE vgi, LOCATION 'docker://ghcr.io/query-farm/vgi-anomaly:0.5.0');
  ```

- **From a local checkout** (`git clone https://github.com/Query-farm/vgi-anomaly`), point `LOCATION` at `uv run --project /path/to/vgi-anomaly vgi-anomaly-worker`.

### Build from source

To build the worker yourself, see [Development](#development).

## Ordered rows as input

The table functions use VGI's [buffering API](https://query.farm/vgi/docs/python/how-to/function-patterns/#buffering)
to collect the entire input before running a detector. Pass a query returning
**exactly one numeric column**, with `ORDER BY` inside that query. The column can
have any name; integer, floating-point, and decimal inputs are converted to double.
Use a unique tie-breaker when timestamps repeat (`ORDER BY ts, sample_id`).

```sql
SELECT *
FROM anomaly.zscore_anomalies(
    (SELECT reading FROM readings ORDER BY ts, sample_id),
    threshold := 2.0
);
```

Each call analyzes one series. Input order is preserved across batches and worker
processes. This removes `array_agg` and `UNNEST` from the calling SQL; the algorithm
still buffers the whole series and has the same computational cost and
1,000,000-sample limit. Add `ORDER BY "index"` to the outer query when result
presentation order matters.

| Table function | Named parameters | Result columns |
| --- | --- | --- |
| `matrix_profile((query), window_size := 50)` | Required `window_size` | `index BIGINT`, `distance DOUBLE`, one row per subsequence |
| `discord_index((query), window_size := 50)` | Required `window_size` | `index BIGINT`, one row |
| `motif_index((query), window_size := 50)` | Required `window_size` | `index BIGINT`, one row |
| `change_points((query))` | Optional `n_bkps` (omit for automatic detection) | `index BIGINT`, one row per change point |
| `zscore_anomalies((query))` | Optional `threshold` (default `3.0`) | `index BIGINT`, one row per outlier |

All indices are **zero-based positions in the ordered input**. An invalid nonempty
series produces one row with NULL result columns. Empty input and valid series
with no change points or outliers produce zero rows. Bad constants raise an error.

For grouped analysis, the existing scalar interface remains convenient:

```sql
SELECT sensor_id,
       anomaly.discord_index(array_agg(reading ORDER BY ts), 50) AS anomaly_at
FROM readings
GROUP BY sensor_id;
```

The scalar functions accept a whole series as `DOUBLE[]` and return an index or
array. DuckDB distinguishes `SELECT anomaly.change_points(array)` from
`SELECT * FROM anomaly.change_points((query))`, so both interfaces use the same names.

## Scalars and argument syntax

VGI / DuckDB **scalar** functions take **positional** arguments only and resolve
overloads by *arity* (DuckDB's `name := value` syntax is a table-function feature,
not a scalar one). So `change_points`'s optional breakpoint count is an extra
positional **arity overload**:

```sql
SELECT change_points(series);       -- automatic number of change points (PELT)
SELECT change_points(series, 3);    -- exactly 3 change points (dynamic programming)
```

## Scalar function catalog

| Function | Signature | Returns | Description |
| --- | --- | --- | --- |
| `matrix_profile` | `(values DOUBLE[], window INT)` | `DOUBLE[]` | STUMP matrix profile: per-subsequence z-normalized distance to its nearest neighbour. Length = `len(values) - window + 1`. |
| `discord_index` | `(values DOUBLE[], window INT)` | `BIGINT` | Start index of the top **discord** (anomaly) = subsequence with the **largest** matrix-profile value. |
| `motif_index` | `(values DOUBLE[], window INT)` | `BIGINT` | Start index of the top **motif** (most repeated) = subsequence with the **smallest** matrix-profile value. |
| `change_points` | `(values DOUBLE[])` | `BIGINT[]` | Change-point indices via ruptures **PELT** (`model="rbf"`, automatic penalty); count chosen automatically. |
| `change_points` | `(values DOUBLE[], n_bkps INT)` | `BIGINT[]` | Change-point indices via ruptures **Dynp** (`model="rbf"`) for **exactly** `n_bkps` breakpoints. |
| `zscore_anomalies` | `(values DOUBLE[], threshold DOUBLE)` | `BIGINT[]` | Indices whose value is more than `threshold` population std devs from the mean (light, dependency-free). |
| `zscore_anomalies` | `(values DOUBLE[])` | `BIGINT[]` | Same, at the default **3-sigma** cutoff (equivalent to `zscore_anomalies(values, 3.0)`). |

### Matrix profile (`stumpy`)

For every length-`window` subsequence of the series, the matrix profile records
the z-normalized Euclidean distance to its nearest *non-trivial* neighbour
elsewhere in the series. A **discord** (largest distance) is the most anomalous
subsequence; a **motif** (smallest distance) is the most repeated pattern. Both
`discord_index` and `motif_index` return the *start* index of that subsequence.

`window` must satisfy `3 <= window < len(values)`, otherwise the function raises
a clear SQL error (e.g. window ≥ length).

### Change points (`ruptures`)

`change_points(values)` runs **PELT** with `model="rbf"` (kernel change detection,
sensitive to shifts in mean *and* distribution) and an automatic penalty of
`log(n)` — a plain BIC-style complexity term (the `rbf` cost is amplitude-invariant,
so scaling the penalty by variance would over-penalize large steps and miss them) —
letting the algorithm choose how many change points there are.
`change_points(values, n_bkps)` instead runs exact dynamic programming (`Dynp`,
same model) for precisely `n_bkps` breakpoints; `n_bkps` must satisfy
`1 <= n_bkps < len(values)`. Returned indices are *interior* breakpoints (the
trailing `len(values)` sentinel ruptures appends is dropped); each is the index
of the first sample of a new segment.

### Z-score

`zscore_anomalies(values, threshold)` is the light, dependency-free complement to
the heavy methods: it returns the indices whose value lies more than `threshold`
population standard deviations from the series mean. A constant series flags
nothing; `threshold` must be positive. Omit `threshold`
(`zscore_anomalies(values)`) to use the textbook **3-sigma** default.

## NULL and robustness semantics

For scalar calls, a series that is **NULL, empty, too short** for the requested operation, or that
contains a NULL or non-finite (`NaN`/`inf`) sample yields **NULL** output for that
row — never an error. Out-of-range constant parameters (`window`, `n_bkps`,
`threshold`) raise a clear SQL error. Invalid series yield NULL; bad constants abort the query. Table calls use the
NULL-row and empty-result semantics described above.

To bound the worst case (the matrix profile is *O(n²)*), series longer than
**1,000,000** samples are rejected with an error.

### numba warm-up

`stumpy` is JIT-compiled with [numba](https://numba.pydata.org/), so the *first*
call in a fresh process pays a multi-second compilation cost. The worker
**warms up at startup** (`detectors.warm_up()` runs each entry point once on a
tiny dummy series) so the first real query is fast and the end-to-end SQL suite
stays deterministic under load. Warm-up only populates JIT caches — it never
changes any output, and failures are swallowed.

## Native dependencies & licensing

| Library | License | Role |
| --- | --- | --- |
| [`stumpy`](https://pypi.org/project/stumpy/) | BSD-3-Clause | Matrix profile (`stump`); pulls `numba` + `numpy`. |
| [`ruptures`](https://pypi.org/project/ruptures/) | BSD-2-Clause | Change-point detection (PELT, Dynp). |
| [`numpy`](https://numpy.org/) | BSD-3-Clause | Array math, z-score. |
| [`numba`](https://numba.pydata.org/) (transitive via stumpy) | BSD-2-Clause | JIT compilation of the matrix-profile kernel. |

All dependencies are permissively licensed. This project is MIT (see `LICENSE`).

## Development

```sh
uv sync --extra dev
uv run pytest -q          # unit tests (pure logic + scalar and buffered VGI calls)
make test-sql             # end-to-end DuckDB SQL tests (haybarn-unittest)
uv run ruff check . && uv run mypy vgi_anomaly/
```

The code is split into a pure-logic module (`vgi_anomaly/detectors.py`, no Arrow
or VGI — directly unit-testable) and the Arrow/VGI adapters
(`vgi_anomaly/scalars.py` and `vgi_anomaly/buffering.py`). `anomaly_worker.py` assembles them into the `anomaly`
catalog.

---

## Authorship & License

Written by [Query.Farm](https://query.farm).

Copyright 2026 Query Farm LLC - https://query.farm
