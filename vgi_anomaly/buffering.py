"""Whole-series detectors accepting an ordered, single-column SQL relation.

Input batches live in execution-scoped VGI storage. Combine restores their
source order and runs the detector once; finalize streams ordinary result rows.
"""

from __future__ import annotations

import json
import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Annotated, ClassVar, cast

import pyarrow as pa
from vgi import Arg
from vgi.arguments import TableInput
from vgi.invocation import BindResponse
from vgi.table_buffering_function import TableBufferingFunction, TableBufferingParams
from vgi.table_function import BindParams, OutputCollector
from vgi_rpc import ArrowSerializableDataclass

from . import detectors
from .scalars import _examples, _meta_tags

_INPUT = b"series"
_OUTPUT = b"results"
_OUTPUT_BATCH_SIZE = 2048
_INDEX_SCHEMA = pa.schema([("index", pa.int64())])
_PROFILE_FIELDS: list[tuple[str, pa.DataType]] = [("index", pa.int64()), ("distance", pa.float64())]
_PROFILE_SCHEMA = pa.schema(_PROFILE_FIELDS)


@dataclass(frozen=True, slots=True, kw_only=True)
class SeriesArgs:
    """One numeric input column, explicitly ordered by time in SQL."""

    data: Annotated[TableInput, Arg(0, doc="One numeric column, ordered by time with ORDER BY in the input query.")]


@dataclass(frozen=True, slots=True, kw_only=True)
class WindowArgs(SeriesArgs):
    """Arguments for a matrix-profile detector."""

    window_size: Annotated[
        int, Arg("window_size", ge=3, doc="Subsequence length; at least 3 and less than the series length.")
    ]


@dataclass(frozen=True, slots=True, kw_only=True)
class ChangePointArgs(SeriesArgs):
    """Arguments for automatic or fixed-count segmentation."""

    n_bkps: Annotated[int | None, Arg("n_bkps", default=None, ge=1, doc="Exact breakpoint count; omit for PELT.")] = (
        None
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ZscoreArgs(SeriesArgs):
    """Arguments for point-outlier detection."""

    threshold: Annotated[
        float, Arg("threshold", default=3.0, gt=0, doc="Positive finite population z-score cutoff.")
    ] = 3.0


@dataclass(kw_only=True)
class DrainState(ArrowSerializableDataclass):
    """Small cursor carried across finalize calls, including HTTP calls."""

    next_batch: int = 0


class SeriesBuffer[TArgs: SeriesArgs](TableBufferingFunction[TArgs, DrainState]):
    """Shared ordered sink, whole-series combine, and batched result source."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = _INDEX_SCHEMA

    class Meta:
        """Consume the ordered relation serially; all detectors need its full series."""

        sink_order_dependent = True
        source_order_dependent = True

    @classmethod
    def on_bind(cls, params: BindParams[TArgs]) -> BindResponse:
        """Validate the input shape and constants before consuming any rows."""
        schema = params.bind_call.input_schema
        if schema is None or len(schema) != 1:
            raise ValueError("expected exactly one numeric column; use (SELECT value FROM readings ORDER BY time)")
        dtype = schema.field(0).type
        if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_decimal(dtype)):
            raise ValueError(f"expected a numeric input column, got {dtype}")
        args = params.args
        if isinstance(args, WindowArgs) and (args.window_size is None or args.window_size < detectors.MIN_WINDOW):
            raise ValueError(f"window must be >= {detectors.MIN_WINDOW}")
        if isinstance(args, ChangePointArgs) and args.n_bkps is not None and args.n_bkps < 1:
            raise ValueError("n_bkps must be >= 1")
        if isinstance(args, ZscoreArgs) and (
            args.threshold is None or args.threshold <= 0 or not math.isfinite(args.threshold)
        ):
            raise ValueError("threshold must be a positive finite number")
        return BindResponse(output_schema=cls.FIXED_SCHEMA)

    @classmethod
    def process(cls, batch: pa.RecordBatch, params: TableBufferingParams[TArgs]) -> bytes:
        """Store batches in sink order and reject oversized input immediately."""
        if batch.num_rows == 0:
            return b""
        # sink_order_dependent makes callback order match the ordered input.
        # A shared counter keeps this position valid across worker processes.
        end = params.storage.counter_add(_INPUT, b"row_count", batch.num_rows)
        cls._check_size(end)
        key = end.to_bytes(8, "big")
        numeric = pa.RecordBatch.from_arrays([batch.column(0).cast(pa.float64(), safe=False)], names=["value"])
        params.storage.state_put(_INPUT, key, params.storage.serialize_record_batch(numeric))
        return params.execution_id

    @staticmethod
    def _check_size(size: int) -> None:
        if size > detectors.MAX_SERIES_LEN:
            raise ValueError(f"series too long: {size} > MAX_SERIES_LEN ({detectors.MAX_SERIES_LEN})")

    @classmethod
    def combine(cls, state_ids: list[bytes], params: TableBufferingParams[TArgs]) -> list[bytes]:
        """Read the execution's full series, regardless of sink-ID granularity."""
        # DuckDB can retain only the last state_id from each sink thread, while
        # the Python client supplies one per batch. Treat both as a single
        # execution bucket and enumerate storage, never the state_ids list.
        size = params.storage.counter_get(_INPUT, b"row_count")
        cls._check_size(size)
        if size == 0:
            return []
        values: list[float] = []
        for _key, payload in params.storage.state_scan(_INPUT):
            values.extend(cast(list[float], params.storage.deserialize_record_batch(payload).column(0).to_pylist()))
        if len(values) != size:
            raise RuntimeError("buffered series rows are missing from VGI storage")
        result = cls.detect(values, params.args)
        for i, offset in enumerate(range(0, result.num_rows, _OUTPUT_BATCH_SIZE)):
            chunk = result.slice(offset, _OUTPUT_BATCH_SIZE)
            params.storage.state_put(_OUTPUT, i.to_bytes(8, "big"), params.storage.serialize_record_batch(chunk))
        params.storage.state_delete(_INPUT)
        params.storage.counter_delete(_INPUT, b"row_count")
        return [b"results"]

    @classmethod
    @abstractmethod
    def detect(cls, values: list[float], args: TArgs) -> pa.RecordBatch:
        """Run one detector and convert its result to rows."""

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[TArgs]) -> DrainState:
        """Start at the first result batch."""
        return DrainState()

    @classmethod
    def finalize(
        cls, params: TableBufferingParams[TArgs], finalize_state_id: bytes, state: DrainState, out: OutputCollector
    ) -> None:
        """Read one result batch from shared storage, or finish the stream."""
        payload = params.storage.state_get(_OUTPUT, state.next_batch.to_bytes(8, "big"))
        if payload is None:
            out.finish()
            return
        out.emit(params.storage.deserialize_record_batch(payload))
        state.next_batch += 1


def _indices(indices: list[int] | None) -> pa.RecordBatch:
    """Preserve invalid-series NULL separately from a valid series with no hits."""
    return pa.RecordBatch.from_pydict({"index": [None] if indices is None else indices}, schema=_INDEX_SCHEMA)


def _tags(title: str, category: str, description: str, sql: str) -> dict[str, str]:
    description += (
        " Accepts an ordered single-column numeric table. Use ORDER BY in the input query; indices are zero-based "
        "positions in that order. Buffers the entire series, including across worker batches. "
        "Invalid nonempty series return one NULL row; empty input and valid series with no matches return no rows. "
        "Series are limited to 1,000,000 samples. Array-valued scalar calls remain available in SELECT expressions."
    )
    tags = _meta_tags(
        title=title,
        category=category,
        description_llm=description,
        description_md=(
            f"# {title}\n\n{description}\n\n"
            "Input values are converted to double precision. Include a unique tie-breaker when timestamps repeat. "
            "The whole series is buffered before detection, so this interface does not change the algorithm’s "
            "memory or CPU cost. Use the scalar form with GROUP BY when analyzing separate sensors."
        ),
        keywords=["time series", "buffering", "table", "anomaly"],
        example_queries=[{"description": title, "sql": sql}],
    )
    columns = [
        {
            "name": "index",
            "type": "BIGINT",
            "description": "Zero-based position in the ordered input; NULL for an invalid series.",
        }
    ]
    if "matrix_profile" in sql:
        columns.append(
            {
                "name": "distance",
                "type": "DOUBLE",
                "description": "Z-normalized distance to the nearest non-trivial neighbour.",
            }
        )
    tags["vgi.result_columns_schema"] = json.dumps(columns)
    return tags


class MatrixProfile(SeriesBuffer[WindowArgs]):
    """Return one index and distance per subsequence."""

    FIXED_SCHEMA = _PROFILE_SCHEMA

    class Meta(SeriesBuffer.Meta):
        """Matrix-profile table function metadata."""

        name = "matrix_profile"
        comment = "Matrix-profile distances as rows from an ordered numeric table."
        tags = _tags(
            "Matrix profile rows",
            "Matrix Profile",
            comment,
            'SELECT "index", distance FROM anomaly.matrix_profile('
            "(SELECT i::DOUBLE FROM range(10) t(i) ORDER BY i), window_size := 4)",
        )
        examples = _examples(json.loads(tags["vgi.example_queries"]))

    @classmethod
    def detect(cls, values: list[float], args: WindowArgs) -> pa.RecordBatch:
        """Compute the profile and expose its positions as a column."""
        profile = detectors.matrix_profile(values, args.window_size)
        return pa.RecordBatch.from_pydict(
            {
                "index": [None] if profile is None else list(range(len(profile))),
                "distance": [None] if profile is None else profile,
            },
            schema=cls.FIXED_SCHEMA,
        )


class DiscordIndex(SeriesBuffer[WindowArgs]):
    """Return the start index of the most anomalous subsequence."""

    class Meta(SeriesBuffer.Meta):
        """Discord table function metadata."""

        name = "discord_index"
        comment = "Most anomalous window's zero-based start index from an ordered numeric table."
        tags = _tags(
            "Discord index row",
            "Matrix Profile",
            comment,
            'SELECT "index" FROM anomaly.discord_index('
            "(SELECT i::DOUBLE FROM range(10) t(i) ORDER BY i), window_size := 4)",
        )
        examples = _examples(json.loads(tags["vgi.example_queries"]))

    @classmethod
    def detect(cls, values: list[float], args: WindowArgs) -> pa.RecordBatch:
        """Compute the top discord's start index."""
        index = detectors.discord_index(values, args.window_size)
        return _indices(None if index is None else [index])


class MotifIndex(SeriesBuffer[WindowArgs]):
    """Return the start index of the most repeated subsequence."""

    class Meta(SeriesBuffer.Meta):
        """Motif table function metadata."""

        name = "motif_index"
        comment = "Most repeated window's zero-based start index from an ordered numeric table."
        tags = _tags(
            "Motif index row",
            "Matrix Profile",
            comment,
            'SELECT "index" FROM anomaly.motif_index('
            "(SELECT i::DOUBLE FROM range(10) t(i) ORDER BY i), window_size := 4)",
        )
        examples = _examples(json.loads(tags["vgi.example_queries"]))

    @classmethod
    def detect(cls, values: list[float], args: WindowArgs) -> pa.RecordBatch:
        """Compute the top motif's start index."""
        index = detectors.motif_index(values, args.window_size)
        return _indices(None if index is None else [index])


class ChangePoints(SeriesBuffer[ChangePointArgs]):
    """Return one row per interior change point."""

    class Meta(SeriesBuffer.Meta):
        """Change-point table function metadata."""

        name = "change_points"
        comment = "Regime-change indices from an ordered numeric table; optional n_bkps selects an exact count."
        tags = _tags(
            "Change point rows",
            "Change Points",
            comment,
            'SELECT "index" FROM anomaly.change_points('
            "(SELECT CASE WHEN i < 8 THEN 1.0 ELSE 9.0 END FROM range(16) t(i) ORDER BY i))",
        )
        examples = _examples(json.loads(tags["vgi.example_queries"]))

    @classmethod
    def detect(cls, values: list[float], args: ChangePointArgs) -> pa.RecordBatch:
        """Run PELT or fixed-count segmentation."""
        return _indices(detectors.change_points(values, args.n_bkps))


class ZscoreAnomalies(SeriesBuffer[ZscoreArgs]):
    """Return one row per point outlier."""

    class Meta(SeriesBuffer.Meta):
        """Z-score table function metadata."""

        name = "zscore_anomalies"
        comment = "Point-outlier indices from an ordered numeric table; threshold defaults to 3 sigma."
        tags = _tags(
            "Z-score outlier rows",
            "Outliers",
            comment,
            'SELECT "index" FROM anomaly.zscore_anomalies('
            "(SELECT CASE WHEN i = 19 THEN 100.0 ELSE 0.0 END FROM range(20) t(i) ORDER BY i))",
        )
        examples = _examples(json.loads(tags["vgi.example_queries"]))

    @classmethod
    def detect(cls, values: list[float], args: ZscoreArgs) -> pa.RecordBatch:
        """Find points exceeding the population z-score threshold."""
        return _indices(detectors.zscore_anomalies(values, args.threshold))


BUFFERING_FUNCTIONS = (MatrixProfile, DiscordIndex, MotifIndex, ChangePoints, ZscoreAnomalies)
