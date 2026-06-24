"""Per-row scalar anomaly-detection functions (LIST-in / scalar-or-LIST-out).

Each function operates on a whole numeric **series** passed as a single
``DOUBLE[]`` argument -- the caller builds it in SQL with
``array_agg(value ORDER BY t)`` -- and returns either a single index or an
array. This composes cleanly without subquery table arguments::

    SELECT anomaly.matrix_profile(array_agg(v ORDER BY t), 50) FROM series;
    SELECT anomaly.discord_index(array_agg(v ORDER BY t), 50) FROM series;
    SELECT anomaly.change_points(array_agg(v ORDER BY t))      FROM series;

A note on argument syntax
-------------------------
VGI / DuckDB *scalar* functions take **positional** arguments and resolve
overloads by *arity* (the ``name := value`` named-argument syntax is a property
of table functions, not scalars). ``change_points`` therefore exposes its
optional ``n_bkps`` as a second arity overload sharing the function ``name`` --
the same idiom the sibling ``vgi-conform`` worker uses for its optional
``region`` arguments.

NULL / robustness semantics: a NULL, empty, or too-short series (or one
containing a NULL or non-finite sample) yields NULL output, per row, never an
error. An out-of-range ``window`` / ``n_bkps`` raises a clear SQL error (see
:mod:`vgi_anomaly.detectors`). Errors are caught per row -- one bad row yields
NULL, it never aborts the batch.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated

import pyarrow as pa
from vgi.arguments import ConstParam, Param, Returns
from vgi.metadata import FunctionExample
from vgi.scalar_function import ScalarFunction

from . import detectors

_VALUES_DOC = "Numeric series as a DOUBLE[] (build with array_agg(value ORDER BY t))."
_WINDOW_DOC = "Subsequence length for the matrix profile (>= 3 and < series length)."

_LIST_DOUBLE = pa.list_(pa.float64())
_LIST_BIGINT = pa.list_(pa.int64())

# Base URL for vgi.source_url (VGI128) links into the implementing source file.
_SOURCE_BASE = "https://github.com/Query-farm/vgi-anomaly/blob/main/vgi_anomaly"
_SCALARS_SOURCE_URL = f"{_SOURCE_BASE}/scalars.py"
_DETECTORS_SOURCE_URL = f"{_SOURCE_BASE}/detectors.py"

# VGI509: at least one object must ship vgi.executable_examples — a JSON list of
# {"description","sql"} objects whose SQL is catalog-qualified and self-contained
# (no external tables) so the linter can execute every one against the worker.
# expected_result is optional and intentionally omitted.
_EXECUTABLE_EXAMPLES = json.dumps(
    [
        {
            "description": "Flag z-score outliers beyond 2 sigma; the 40.0 spike is index 5.",
            "sql": ("SELECT anomaly.zscore_anomalies([10.0,10.0,11.0,9.0,10.0,40.0,10.0,9.0,11.0]::DOUBLE[], 2.0)"),
        },
        {
            "description": "Top discord (most anomalous window) of a 25-point series, window 4.",
            "sql": (
                "SELECT anomaly.discord_index("
                "[1.0,2.0,3.0,4.0,3.0,2.0,1.0,2.0,3.0,4.0,3.0,2.0,1.0,2.0,3.0,4.0,"
                "3.0,2.0,50.0,2.0,3.0,4.0,3.0,2.0,1.0]::DOUBLE[], 4)"
            ),
        },
        {
            "description": "Automatic change-point detection on a single step at index 8.",
            "sql": (
                "SELECT anomaly.change_points("
                "[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0]"
                "::DOUBLE[])"
            ),
        },
    ]
)


def _meta_tags(
    *,
    title: str,
    description_llm: str,
    description_md: str,
    keywords: str,
) -> dict[str, str]:
    """Build the strict-profile per-object tag set shared by every scalar.

    Every function carries VGI124 ``vgi.title`` (a human display name that is
    intentionally *not* the machine name, to satisfy VGI125), VGI112
    ``vgi.description_llm`` and VGI113 ``vgi.description_md`` (Markdown
    narratives for agents and humans respectively), VGI126 ``vgi.keywords``
    (comma-separated synonyms), and VGI128 ``vgi.source_url`` (the file that
    implements the detection logic).
    """
    return {
        "vgi.title": title,
        "vgi.description_llm": description_llm,
        "vgi.description_md": description_md,
        "vgi.keywords": keywords,
        "vgi.source_url": _DETECTORS_SOURCE_URL,
    }


# ---------------------------------------------------------------------------
# Mapping helper: read each row's list, run a pure series -> result function,
# rebuild the output array. The detectors already encode the robustness split:
# a NULL/empty/too-short/non-finite *series* comes back as None (-> SQL NULL),
# while a bad *constant* argument (window/n_bkps/threshold, identical for the
# whole batch) raises ValueError, which we let propagate as a clear SQL error.
# So there is nothing to catch per row here.
# ---------------------------------------------------------------------------


def _map_series[T](
    arr: pa.ListArray,
    fn: Callable[[list[float] | None], T],
    arrow_type: pa.DataType,
) -> pa.Array:
    out: list[T | None] = []
    for row in arr.to_pylist():
        out.append(fn(row))
    return pa.array(out, type=arrow_type)


# ===========================================================================
# Matrix profile
# ===========================================================================


class MatrixProfileFunction(ScalarFunction):
    """``matrix_profile(values, window)`` -- STUMP matrix profile of the series."""

    class Meta:
        """VGI registration metadata for ``matrix_profile``."""

        name = "matrix_profile"
        description = (
            "Matrix profile (STUMP): per-subsequence z-normalized distance to its nearest "
            "neighbour; length = len(values) - window + 1. NULL for a short/invalid series."
        )
        categories = ["anomaly", "matrix_profile"]
        tags = _meta_tags(
            title="Matrix Profile of Series",
            description_llm=(
                "# matrix_profile\n\n"
                "Compute the **matrix profile** of a numeric time series with STUMP "
                "(`stumpy.stump`). The result is a `DOUBLE[]` of length "
                "`len(values) - window + 1`; element `i` is the z-normalized Euclidean "
                "distance from the length-`window` subsequence starting at `i` to its "
                "nearest non-trivial neighbour.\n\n"
                "## When to use\n"
                "Use it as the foundation for anomaly and pattern analysis on a single "
                "series: large profile values mark **discords** (anomalous subsequences), "
                "small values mark **motifs** (repeated patterns). Prefer the dedicated "
                "`discord_index` / `motif_index` helpers if you only need the top index.\n\n"
                "## Inputs / outputs\n"
                "- `values DOUBLE[]` — the whole series, built in SQL with "
                "`array_agg(value ORDER BY t)` or a `[...]::DOUBLE[]` literal.\n"
                "- `window BIGINT` — subsequence length, constant per call, "
                "`3 <= window < len(values)`.\n"
                "- Returns `DOUBLE[]` of profile distances, or `NULL`.\n\n"
                "## Edge cases\n"
                "A NULL, empty, too-short, or non-finite series returns `NULL` (per row, "
                "never an error). A `window` outside `[3, len)` raises a clear SQL error. "
                "Cost is O(n^2); series longer than 1,000,000 samples are rejected."
            ),
            description_md=(
                "# Matrix Profile\n\n"
                "Returns the matrix profile of a numeric series for a given subsequence "
                "`window`, computed with `stumpy.stump`.\n\n"
                "## Usage\n"
                "```sql\n"
                "SELECT anomaly.matrix_profile([1.0,2.0,3.0,4.0,3.0,2.0,1.0,2.0,3.0,4.0]::DOUBLE[], 4);\n"
                "SELECT anomaly.matrix_profile(array_agg(v ORDER BY t), 50) FROM series;\n"
                "```\n\n"
                "## Notes\n"
                "The output length is `len(values) - window + 1`. Pair it with `argmax` "
                "for the discord or `argmin` for the motif. NULL for an invalid/short "
                "series; an out-of-range `window` is a hard error."
            ),
            keywords=(
                "matrix profile, stumpy, stump, z-normalized distance, subsequence, "
                "anomaly, discord, motif, time series, similarity"
            ),
        )
        examples = [
            FunctionExample(
                sql=(
                    "SELECT len(anomaly.matrix_profile("
                    "[1.0,2.0,3.0,4.0,3.0,2.0,1.0,2.0,3.0,4.0,3.0,2.0,1.0,2.0,3.0,4.0,"
                    "3.0,2.0,50.0,2.0,3.0,4.0,3.0,2.0,1.0]::DOUBLE[], 4))"
                ),
                description="Matrix profile length for a 25-point series, window 4 (= 22).",
            ),
        ]

    @classmethod
    def compute(
        cls,
        values: Annotated[pa.ListArray, Param(arrow_type=_LIST_DOUBLE, doc=_VALUES_DOC)],
        window: Annotated[int, ConstParam(_WINDOW_DOC)],
    ) -> Annotated[pa.ListArray, Returns(arrow_type=_LIST_DOUBLE)]:
        """Map the matrix-profile detector over each series row."""
        return _map_series(values, lambda v: detectors.matrix_profile(v, window), _LIST_DOUBLE)


class DiscordIndexFunction(ScalarFunction):
    """``discord_index(values, window)`` -- start index of the top discord."""

    class Meta:
        """VGI registration metadata for ``discord_index``."""

        name = "discord_index"
        description = (
            "Start index of the top discord (anomaly): the subsequence with the largest "
            "matrix-profile value. NULL for a short/invalid series."
        )
        categories = ["anomaly", "matrix_profile"]
        tags = _meta_tags(
            title="Top Discord Start Index",
            description_llm=(
                "# discord_index\n\n"
                "Return the **start index of the top discord** of a numeric series: the "
                "length-`window` subsequence whose matrix-profile distance is the "
                "**largest**, i.e. the most anomalous, least-similar pattern in the "
                "series.\n\n"
                "## When to use\n"
                "Use it to locate the single most unusual stretch of a time series "
                "(a spike, a glitch, an out-of-pattern run) without materializing the "
                "full matrix profile. For the full distance vector use `matrix_profile`; "
                "for the *most repeated* pattern use `motif_index`.\n\n"
                "## Inputs / outputs\n"
                "- `values DOUBLE[]` — the series (`array_agg(value ORDER BY t)`).\n"
                "- `window BIGINT` — subsequence length, `3 <= window < len(values)`.\n"
                "- Returns a `BIGINT` start index (0-based), or `NULL`.\n\n"
                "## Edge cases\n"
                "NULL / empty / too-short / non-finite series returns `NULL`; an "
                "out-of-range `window` raises a clear SQL error. Ties resolve to the "
                "first (lowest) index, matching `numpy.argmax`."
            ),
            description_md=(
                "# Discord Index\n\n"
                "Start index of the most anomalous length-`window` subsequence (largest "
                "matrix-profile value).\n\n"
                "## Usage\n"
                "```sql\n"
                "SELECT anomaly.discord_index(\n"
                "  [1.0,2.0,3.0,4.0,3.0,2.0,1.0,2.0,3.0,4.0,3.0,2.0,1.0,2.0,3.0,4.0,\n"
                "   3.0,2.0,50.0,2.0,3.0,4.0,3.0,2.0,1.0]::DOUBLE[], 4);  -- 16\n"
                "```\n\n"
                "## Notes\n"
                "Returns NULL for an invalid/short series; an out-of-range `window` is a "
                "hard error. The index points at the first sample of the discord window."
            ),
            keywords=(
                "discord, anomaly index, most anomalous, outlier subsequence, matrix "
                "profile, argmax, time series, novelty detection"
            ),
        )
        examples = [
            FunctionExample(
                sql=(
                    "SELECT anomaly.discord_index("
                    "[1.0,2.0,3.0,4.0,3.0,2.0,1.0,2.0,3.0,4.0,3.0,2.0,1.0,2.0,3.0,4.0,"
                    "3.0,2.0,50.0,2.0,3.0,4.0,3.0,2.0,1.0]::DOUBLE[], 4)"
                ),
                description="Index of the most anomalous window (a spike at 18 -> 16).",
            ),
        ]

    @classmethod
    def compute(
        cls,
        values: Annotated[pa.ListArray, Param(arrow_type=_LIST_DOUBLE, doc=_VALUES_DOC)],
        window: Annotated[int, ConstParam(_WINDOW_DOC)],
    ) -> Annotated[pa.Int64Array, Returns()]:
        """Map the discord-index detector over each series row."""
        return _map_series(values, lambda v: detectors.discord_index(v, window), pa.int64())


class MotifIndexFunction(ScalarFunction):
    """``motif_index(values, window)`` -- start index of the top motif."""

    class Meta:
        """VGI registration metadata for ``motif_index``."""

        name = "motif_index"
        description = (
            "Start index of the top motif (most repeated pattern): the subsequence with the "
            "smallest matrix-profile value. NULL for a short/invalid series."
        )
        categories = ["anomaly", "matrix_profile"]
        tags = _meta_tags(
            title="Top Motif Start Index",
            description_llm=(
                "# motif_index\n\n"
                "Return the **start index of the top motif** of a numeric series: the "
                "length-`window` subsequence whose matrix-profile distance is the "
                "**smallest**, i.e. the most repeated, most conserved pattern in the "
                "series.\n\n"
                "## When to use\n"
                "Use it to find recurring shapes — a daily load curve, a heartbeat, a "
                "repeated motion — without computing the whole matrix profile. It is the "
                "complement of `discord_index` (anomalies): motif = smallest distance, "
                "discord = largest distance.\n\n"
                "## Inputs / outputs\n"
                "- `values DOUBLE[]` — the series (`array_agg(value ORDER BY t)`).\n"
                "- `window BIGINT` — subsequence length, `3 <= window < len(values)`.\n"
                "- Returns a `BIGINT` start index (0-based) of one copy of the motif, "
                "or `NULL`.\n\n"
                "## Edge cases\n"
                "NULL / empty / too-short / non-finite series returns `NULL`; an "
                "out-of-range `window` raises a clear SQL error. Ties resolve to the "
                "first (lowest) index, matching `numpy.argmin`."
            ),
            description_md=(
                "# Motif Index\n\n"
                "Start index of the most repeated length-`window` subsequence (smallest "
                "matrix-profile value).\n\n"
                "## Usage\n"
                "```sql\n"
                "SELECT anomaly.motif_index(\n"
                "  [0.0,2.0,4.0,2.0,0.0,0.1,0.116,0.133,0.15,0.166,0.183,0.2,\n"
                "   0.0,2.0,4.0,2.0,0.0,0.3,0.4,0.5]::DOUBLE[], 5);  -- 0\n"
                "```\n\n"
                "## Notes\n"
                "Returns NULL for an invalid/short series; an out-of-range `window` is a "
                "hard error. The complement of `discord_index`."
            ),
            keywords=(
                "motif, repeated pattern, recurring subsequence, conserved pattern, "
                "matrix profile, argmin, time series, pattern discovery"
            ),
        )
        examples = [
            FunctionExample(
                sql=(
                    "SELECT anomaly.motif_index("
                    "[0.0,2.0,4.0,2.0,0.0,0.1,0.116,0.133,0.15,0.166,0.183,0.2,"
                    "0.0,2.0,4.0,2.0,0.0,0.3,0.4,0.5]::DOUBLE[], 5)"
                ),
                description="Index of the most repeated window (two triangles -> 0).",
            ),
        ]

    @classmethod
    def compute(
        cls,
        values: Annotated[pa.ListArray, Param(arrow_type=_LIST_DOUBLE, doc=_VALUES_DOC)],
        window: Annotated[int, ConstParam(_WINDOW_DOC)],
    ) -> Annotated[pa.Int64Array, Returns()]:
        """Map the motif-index detector over each series row."""
        return _map_series(values, lambda v: detectors.motif_index(v, window), pa.int64())


# ===========================================================================
# Change-point detection -- optional n_bkps as an arity overload.
# ===========================================================================


class ChangePointsFunction(ScalarFunction):
    """``change_points(values)`` -- PELT change-point indices (auto count)."""

    class Meta:
        """VGI registration metadata for ``change_points`` (auto count)."""

        name = "change_points"
        description = (
            "Change-point indices via ruptures PELT (model='rbf', automatic log(n)*variance "
            "penalty); count chosen automatically. NULL for an invalid series."
        )
        categories = ["anomaly", "change_point"]
        tags = _meta_tags(
            title="Automatic Change Point Detection",
            description_llm=(
                "# change_points (automatic count)\n\n"
                "Detect **regime changes** in a numeric series and return the interior "
                "change-point indices as a `BIGINT[]`. This overload "
                "(`change_points(values)`) chooses the **number** of breakpoints "
                "automatically with ruptures PELT (`model='rbf'`) and a BIC-style "
                "`log(n)` penalty.\n\n"
                "## When to use\n"
                "Use it when you do not know how many shifts to expect and want the "
                "algorithm to decide — level shifts, variance changes, distribution "
                "changes. When you already know the count, use the two-argument overload "
                "`change_points(values, n_bkps)` for exactly that many breakpoints.\n\n"
                "## Inputs / outputs\n"
                "- `values DOUBLE[]` — the series (`array_agg(value ORDER BY t)`).\n"
                "- Returns `BIGINT[]`, each the index of the first sample of a new "
                "segment; the trailing `len` sentinel ruptures appends is dropped. "
                "Empty list when no change is found; `NULL` for an invalid series.\n\n"
                "## Edge cases\n"
                "NULL / empty / non-finite / length-<2 series returns `NULL`. The `rbf` "
                "cost is amplitude-invariant, so the penalty is plain `log(n)` and is not "
                "scaled by variance (scaling would miss large steps)."
            ),
            description_md=(
                "# Change Points (Automatic)\n\n"
                "Returns the change-point indices of a series, choosing the count "
                "automatically (ruptures PELT, `model='rbf'`).\n\n"
                "## Usage\n"
                "```sql\n"
                "SELECT anomaly.change_points(\n"
                "  [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0]::DOUBLE[]);\n"
                "  -- [8]\n"
                "```\n\n"
                "## Notes\n"
                "Each index is the first sample of a new segment. Use "
                "`change_points(values, n_bkps)` to force a fixed number of breakpoints. "
                "NULL for an invalid series."
            ),
            keywords=(
                "change point, changepoint, regime change, breakpoint, segmentation, "
                "PELT, ruptures, level shift, structural break, time series"
            ),
        )
        examples = [
            FunctionExample(
                sql=(
                    "SELECT anomaly.change_points("
                    "[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0]"
                    "::DOUBLE[])"
                ),
                description="Auto-detect change points on a single step at index 8 -> [8].",
            ),
        ]

    @classmethod
    def compute(
        cls,
        values: Annotated[pa.ListArray, Param(arrow_type=_LIST_DOUBLE, doc=_VALUES_DOC)],
    ) -> Annotated[pa.ListArray, Returns(arrow_type=_LIST_BIGINT)]:
        """Map the auto-count change-point detector over each series row."""
        return _map_series(values, lambda v: detectors.change_points(v, None), _LIST_BIGINT)


class ChangePointsNFunction(ScalarFunction):
    """``change_points(values, n_bkps)`` -- exactly ``n_bkps`` change points."""

    class Meta:
        """VGI registration metadata for ``change_points`` (fixed count)."""

        name = "change_points"
        description = (
            "Change-point indices via ruptures dynamic programming (Dynp, model='rbf') for "
            "exactly n_bkps breakpoints. NULL for an invalid series; error if n_bkps out of range."
        )
        categories = ["anomaly", "change_point"]
        tags = _meta_tags(
            title="Fixed Count Change Point Detection",
            description_llm=(
                "# change_points (fixed count)\n\n"
                "Detect exactly `n_bkps` **regime changes** in a numeric series and "
                "return their interior indices as a `BIGINT[]`. This overload "
                "(`change_points(values, n_bkps)`) runs ruptures dynamic programming "
                "(`Dynp`, `model='rbf'`) to find the optimal segmentation into "
                "`n_bkps + 1` segments.\n\n"
                "## When to use\n"
                "Use it when you know how many shifts you want (e.g. split a series into "
                "two regimes with `n_bkps = 1`). When the count is unknown, use the "
                "one-argument overload `change_points(values)`, which picks the count "
                "automatically with PELT.\n\n"
                "## Inputs / outputs\n"
                "- `values DOUBLE[]` — the series (`array_agg(value ORDER BY t)`).\n"
                "- `n_bkps BIGINT` — exact number of breakpoints, `1 <= n_bkps < len`.\n"
                "- Returns `BIGINT[]` of `n_bkps` indices (first sample of each new "
                "segment); `NULL` for an invalid series.\n\n"
                "## Edge cases\n"
                "NULL / empty / non-finite series returns `NULL`. An `n_bkps` outside "
                "`[1, len)` raises a clear SQL error (it is constant for the whole "
                "batch, so surfacing it is correct)."
            ),
            description_md=(
                "# Change Points (Fixed Count)\n\n"
                "Returns exactly `n_bkps` change-point indices of a series (ruptures "
                "`Dynp`, `model='rbf'`).\n\n"
                "## Usage\n"
                "```sql\n"
                "SELECT anomaly.change_points(\n"
                "  [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0]::DOUBLE[], 1);\n"
                "  -- [8]\n"
                "```\n\n"
                "## Notes\n"
                "Each index is the first sample of a new segment. An `n_bkps` outside "
                "`[1, len)` is a hard error; NULL for an invalid series."
            ),
            keywords=(
                "change point, changepoint, fixed breakpoints, n_bkps, segmentation, "
                "Dynp, dynamic programming, ruptures, regime change, time series"
            ),
        )
        examples = [
            FunctionExample(
                sql=(
                    "SELECT anomaly.change_points("
                    "[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0]"
                    "::DOUBLE[], 1)"
                ),
                description="Detect exactly one change point on a single step -> [8].",
            ),
        ]

    @classmethod
    def compute(
        cls,
        values: Annotated[pa.ListArray, Param(arrow_type=_LIST_DOUBLE, doc=_VALUES_DOC)],
        n_bkps: Annotated[int, ConstParam("Exact number of breakpoints to find (>= 1, < length).")],
    ) -> Annotated[pa.ListArray, Returns(arrow_type=_LIST_BIGINT)]:
        """Map the fixed-count change-point detector over each series row."""
        return _map_series(values, lambda v: detectors.change_points(v, n_bkps), _LIST_BIGINT)


# ===========================================================================
# Z-score (light, dependency-free)
# ===========================================================================


class ZscoreAnomaliesFunction(ScalarFunction):
    """``zscore_anomalies(values, threshold)`` -- indices where |z| > threshold."""

    class Meta:
        """VGI registration metadata for ``zscore_anomalies``."""

        name = "zscore_anomalies"
        description = (
            "Indices whose value is more than `threshold` population std devs from the mean "
            "(light, dependency-free). NULL for an invalid series; error if threshold <= 0."
        )
        categories = ["anomaly", "zscore"]
        tags = {
            **_meta_tags(
                title="Z-Score Outlier Indices",
                description_llm=(
                    "# zscore_anomalies\n\n"
                    "Flag **point outliers** in a numeric series and return their indices "
                    "as a `BIGINT[]`. A sample is flagged when its absolute z-score "
                    "(distance from the series mean in population standard deviations) "
                    "exceeds `threshold`.\n\n"
                    "## When to use\n"
                    "Use it as a light, dependency-free first pass for individual outliers "
                    "(spikes, dropouts) when you do not need subsequence/shape analysis. "
                    "For anomalous *patterns* use `discord_index`/`matrix_profile`; for "
                    "*regime shifts* use `change_points`.\n\n"
                    "## Inputs / outputs\n"
                    "- `values DOUBLE[]` — the series (`array_agg(value ORDER BY t)`).\n"
                    "- `threshold DOUBLE` — z-score magnitude cutoff, e.g. `3.0`; must be "
                    "positive and finite.\n"
                    "- Returns `BIGINT[]` of flagged indices (empty when nothing exceeds "
                    "the cutoff); `NULL` for an invalid series.\n\n"
                    "## Edge cases\n"
                    "NULL / empty / non-finite series returns `NULL`. A constant series "
                    "(zero std dev) flags nothing (empty list, not NULL). A non-positive "
                    "or non-finite `threshold` raises a clear SQL error."
                ),
                description_md=(
                    "# Z-Score Anomalies\n\n"
                    "Indices whose value is more than `threshold` population standard "
                    "deviations from the series mean.\n\n"
                    "## Usage\n"
                    "```sql\n"
                    "SELECT anomaly.zscore_anomalies(\n"
                    "  [10.0,10.0,11.0,9.0,10.0,40.0,10.0,9.0,11.0]::DOUBLE[], 2.0);  -- [5]\n"
                    "```\n\n"
                    "## Notes\n"
                    "A constant series flags nothing (empty list). A non-positive "
                    "`threshold` is a hard error; NULL for an invalid series."
                ),
                keywords=(
                    "z-score, zscore, outlier, sigma, standard deviation, threshold, "
                    "spike detection, point anomaly, time series"
                ),
            ),
            # VGI509: a guaranteed-runnable, self-contained executable example.
            "vgi.executable_examples": _EXECUTABLE_EXAMPLES,
        }
        examples = [
            FunctionExample(
                sql=("SELECT anomaly.zscore_anomalies([10.0,10.0,11.0,9.0,10.0,40.0,10.0,9.0,11.0]::DOUBLE[], 2.0)"),
                description="Flag samples beyond 2 sigma (an outlier at index 5).",
            ),
        ]

    @classmethod
    def compute(
        cls,
        values: Annotated[pa.ListArray, Param(arrow_type=_LIST_DOUBLE, doc=_VALUES_DOC)],
        threshold: Annotated[float, ConstParam("Z-score magnitude threshold (positive, e.g. 3.0).")],
    ) -> Annotated[pa.ListArray, Returns(arrow_type=_LIST_BIGINT)]:
        """Map the z-score anomaly detector over each series row."""
        return _map_series(values, lambda v: detectors.zscore_anomalies(v, threshold), _LIST_BIGINT)


SCALAR_FUNCTIONS: list[type] = [
    MatrixProfileFunction,
    DiscordIndexFunction,
    MotifIndexFunction,
    ChangePointsFunction,
    ChangePointsNFunction,
    ZscoreAnomaliesFunction,
]
