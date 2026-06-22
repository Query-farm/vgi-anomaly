# vgi-anomaly

[![CI](https://github.com/Query-farm/vgi-anomaly/actions/workflows/ci.yml/badge.svg)](https://github.com/Query-farm/vgi-anomaly/actions/workflows/ci.yml)

A [VGI](https://query.farm) worker that brings **time-series anomaly detection**
into DuckDB/SQL. It finds the anomalous and the repeated structure in a numeric
series — **matrix-profile motifs & discords**, **change points**, and a light
**z-score** outlier check — as plain SQL scalar functions, backed by
battle-tested libraries: [`stumpy`](https://pypi.org/project/stumpy/) (BSD-3,
matrix profile), [`ruptures`](https://pypi.org/project/ruptures/) (BSD-2,
change-point detection) and [`numpy`](https://numpy.org/) (BSD-3).

```sql
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'anomaly' (TYPE vgi, LOCATION 'uv run anomaly_worker.py');

-- A series travels as one DOUBLE[] argument; build it with array_agg(... ORDER BY t).
SELECT anomaly.discord_index(array_agg(v ORDER BY t), 50)  FROM series;  -- most anomalous window
SELECT anomaly.motif_index(array_agg(v ORDER BY t), 50)    FROM series;  -- most repeated window
SELECT anomaly.matrix_profile(array_agg(v ORDER BY t), 50) FROM series;  -- DOUBLE[] profile
SELECT anomaly.change_points(array_agg(v ORDER BY t))      FROM series;  -- BIGINT[] (auto count)
SELECT anomaly.change_points(array_agg(v ORDER BY t), 2)   FROM series;  -- exactly 2 change points
SELECT UNNEST(anomaly.zscore_anomalies(array_agg(v ORDER BY t), 3.0)) FROM series;  -- beyond 3 sigma

-- Literal series work too:
SELECT anomaly.discord_index([1.0,2.0,3.0,100.0,2.0,3.0,1.0]::DOUBLE[], 3);
```

Results are **deterministic** given the input series and parameters — the same
series always yields the same indices and profile, so the worker is hermetic and
easy to test.

## The series-as-argument design

A scalar function sees one row at a time, but anomaly detection is defined over a
*whole series*. So each function takes the series as a single `DOUBLE[]`
**argument** that the caller assembles in SQL — typically
`array_agg(value ORDER BY t)` over an ordered table, or a `[1.0, 2.0, ...]::DOUBLE[]`
literal. The function returns either a single index (`BIGINT`) or an array
(`DOUBLE[]` / `BIGINT[]`). This composes cleanly without subquery table
arguments and runs once per group:

```sql
SELECT sensor_id,
       anomaly.discord_index(array_agg(reading ORDER BY ts), 50) AS anomaly_at
FROM readings
GROUP BY sensor_id;
```

## Scalars and argument syntax

VGI / DuckDB **scalar** functions take **positional** arguments only and resolve
overloads by *arity* (DuckDB's `name := value` syntax is a table-function feature,
not a scalar one). So `change_points`'s optional breakpoint count is an extra
positional **arity overload**:

```sql
SELECT change_points(series);       -- automatic number of change points (PELT)
SELECT change_points(series, 3);    -- exactly 3 change points (dynamic programming)
```

## Function catalog

| Function | Signature | Returns | Description |
| --- | --- | --- | --- |
| `matrix_profile` | `(values DOUBLE[], window INT)` | `DOUBLE[]` | STUMP matrix profile: per-subsequence z-normalized distance to its nearest neighbour. Length = `len(values) - window + 1`. |
| `discord_index` | `(values DOUBLE[], window INT)` | `BIGINT` | Start index of the top **discord** (anomaly) = subsequence with the **largest** matrix-profile value. |
| `motif_index` | `(values DOUBLE[], window INT)` | `BIGINT` | Start index of the top **motif** (most repeated) = subsequence with the **smallest** matrix-profile value. |
| `change_points` | `(values DOUBLE[])` | `BIGINT[]` | Change-point indices via ruptures **PELT** (`model="rbf"`, automatic penalty); count chosen automatically. |
| `change_points` | `(values DOUBLE[], n_bkps INT)` | `BIGINT[]` | Change-point indices via ruptures **Dynp** (`model="rbf"`) for **exactly** `n_bkps` breakpoints. |
| `zscore_anomalies` | `(values DOUBLE[], threshold DOUBLE)` | `BIGINT[]` | Indices whose value is more than `threshold` population std devs from the mean (light, dependency-free). |

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
`log(n) * max(var, 1)` — a BIC-style complexity term scaled by the series
variance — letting the algorithm choose how many change points there are.
`change_points(values, n_bkps)` instead runs exact dynamic programming (`Dynp`,
same model) for precisely `n_bkps` breakpoints; `n_bkps` must satisfy
`1 <= n_bkps < len(values)`. Returned indices are *interior* breakpoints (the
trailing `len(values)` sentinel ruptures appends is dropped); each is the index
of the first sample of a new segment.

### Z-score

`zscore_anomalies(values, threshold)` is the light, dependency-free complement to
the heavy methods: it returns the indices whose value lies more than `threshold`
population standard deviations from the series mean. A constant series flags
nothing; `threshold` must be positive.

## NULL and robustness semantics

A series that is **NULL, empty, too short** for the requested operation, or that
contains a NULL or non-finite (`NaN`/`inf`) sample yields **NULL** output for that
row — never an error. Out-of-range constant parameters (`window`, `n_bkps`,
`threshold`) raise a clear SQL error. Per-row failures never abort the batch.

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
uv run pytest -q          # unit tests (pure logic + scalars via the VGI client)
make test-sql             # end-to-end DuckDB SQL tests (haybarn-unittest)
uv run ruff check . && uv run mypy vgi_anomaly/
```

The code is split into a pure-logic module (`vgi_anomaly/detectors.py`, no Arrow
or VGI — directly unit-testable) and the Arrow/VGI scalar adapters
(`vgi_anomaly/scalars.py`). `anomaly_worker.py` assembles them into the `anomaly`
catalog.
