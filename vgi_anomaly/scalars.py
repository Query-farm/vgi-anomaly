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
        examples = [
            FunctionExample(
                sql="SELECT anomaly.matrix_profile(array_agg(v ORDER BY t), 50) FROM series",
                description="Matrix profile of a series with window 50",
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
        examples = [
            FunctionExample(
                sql="SELECT anomaly.discord_index(array_agg(v ORDER BY t), 50) FROM series",
                description="Index of the most anomalous window",
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
        examples = [
            FunctionExample(
                sql="SELECT anomaly.motif_index(array_agg(v ORDER BY t), 50) FROM series",
                description="Index of the most repeated window",
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
        examples = [
            FunctionExample(
                sql="SELECT anomaly.change_points(array_agg(v ORDER BY t)) FROM series",
                description="Auto-detect change points",
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
        examples = [
            FunctionExample(
                sql="SELECT anomaly.change_points(array_agg(v ORDER BY t), 2) FROM series",
                description="Detect exactly two change points",
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
        examples = [
            FunctionExample(
                sql="SELECT anomaly.zscore_anomalies(array_agg(v ORDER BY t), 3.0) FROM series",
                description="Flag samples beyond 3 sigma",
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
