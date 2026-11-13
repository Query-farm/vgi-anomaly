"""Buffered detectors through RPC, plus distributed ordering/storage coverage."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest
from vgi import Arguments
from vgi.client import Client, ClientError
from vgi.function_storage import BoundStorage, FunctionStorageSqlite

from vgi_anomaly import detectors
from vgi_anomaly.buffering import DrainState, ZscoreAnomalies, ZscoreArgs


@pytest.fixture(scope="module")
def client() -> Iterator[Client]:
    worker = Path(__file__).resolve().parent.parent / "anomaly_worker.py"
    with Client(f"{sys.executable} {worker}", worker_limit=1) as c:
        yield c


def _buffered(client, name, values, *, batch_size=7, **named):
    schema = pa.schema([("reading", pa.float64())])
    batches = [
        pa.record_batch([pa.array(values[i : i + batch_size], type=pa.float64())], schema=schema)
        for i in range(0, len(values), batch_size)
    ]
    result = list(
        client.table_buffering_function(
            function_name=name,
            schema_path=["main"],
            input=iter(batches),
            input_schema=schema,
            arguments=Arguments(named={key: pa.scalar(value) for key, value in named.items()}),
        )
    )
    return [row for batch in result for row in batch.to_pylist()]


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("matrix_profile", {"window_size": 4}),
        ("discord_index", {"window_size": 4}),
        ("motif_index", {"window_size": 4}),
        ("change_points", {}),
        ("change_points", {"n_bkps": 1}),
        ("zscore_anomalies", {}),
        ("zscore_anomalies", {"threshold": 2.0}),
    ],
)
def test_matches_whole_series_detector_across_batches(client, name, kwargs):
    values = [
        1.0,
        2.0,
        3.0,
        4.0,
        3.0,
        2.0,
        1.0,
        2.0,
        3.0,
        4.0,
        3.0,
        2.0,
        1.0,
        2.0,
        3.0,
        4.0,
        3.0,
        2.0,
        50.0,
        2.0,
        3.0,
        4.0,
        3.0,
        2.0,
        1.0,
    ]
    detector_args = {"threshold": 3.0, **kwargs} if name == "zscore_anomalies" else kwargs
    if "window_size" in detector_args:
        detector_args = {"window": detector_args["window_size"]}
    expected = getattr(detectors, name)(values, **detector_args)
    rows = _buffered(client, name, values, **kwargs)
    if name == "matrix_profile":
        assert [r["index"] for r in rows] == list(range(len(expected)))
        assert [r["distance"] for r in rows] == pytest.approx(expected)
    else:
        assert [r["index"] for r in rows] == (expected if isinstance(expected, list) else [expected])


@pytest.mark.parametrize("values", [[1.0], [1.0, None, 3.0], [1.0, float("nan")], [float("inf"), 1.0]])
def test_invalid_series_is_null_row(client, values):
    # Z-score accepts singleton series (no matches); change points requires two samples.
    assert _buffered(client, "change_points", values) == [{"index": None}]


def test_no_matches_has_no_rows(client):
    assert _buffered(client, "zscore_anomalies", [5.0] * 20) == []


def test_empty_input_has_no_rows(client):
    assert _buffered(client, "change_points", []) == []


@pytest.mark.parametrize("dtype", [pa.int64(), pa.float32(), pa.decimal128(20, 0)])
def test_numeric_types_convert_to_double(client, dtype):
    values = pa.array([0] * 19 + [2**60], type=pa.int64()).cast(dtype, safe=False)
    result = list(
        client.table_buffering_function(
            function_name="zscore_anomalies",
            schema_path=["main"],
            input=iter([pa.record_batch([values], names=["reading"])]),
        )
    )
    assert result[0].to_pylist() == [{"index": 19}]


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("matrix_profile", {"window_size": 2}),
        ("matrix_profile", {"window_size": 20}),
        ("change_points", {"n_bkps": 0}),
        ("change_points", {"n_bkps": 20}),
        ("zscore_anomalies", {"threshold": 0.0}),
        ("zscore_anomalies", {"threshold": float("nan")}),
        ("zscore_anomalies", {"threshold": float("inf")}),
    ],
)
def test_bad_constants_error(client, name, kwargs):
    with pytest.raises(ClientError):
        _buffered(client, name, [1.0] * 20, **kwargs)


@pytest.mark.parametrize("columns", [{"a": [1.0], "b": [2.0]}, {"a": ["bad"]}])
def test_bind_rejects_wrong_input_shape(client, columns):
    with pytest.raises(ClientError, match="numeric"):
        list(
            client.table_buffering_function(
                function_name="zscore_anomalies", schema_path=["main"], input=iter([pa.record_batch(columns)])
            )
        )


def test_multiple_output_batches(client):
    # More than 2048 results tests the wire-serialized finalize cursor.
    rows = _buffered(client, "zscore_anomalies", [-1.0, 1.0] * 2500, batch_size=700, threshold=0.5)
    assert [r["index"] for r in rows] == list(range(5000))


def test_cross_worker_storage_with_last_sink_id(tmp_path, monkeypatch):
    # Independent SQLite handles stand in for separate sink/combine/source workers.
    path = str(tmp_path / "state.sqlite")
    args = ZscoreArgs(data=None, threshold=3.0)

    def params(execution=b"query", batch_index=None):
        return SimpleNamespace(
            storage=BoundStorage(FunctionStorageSqlite(path), execution),
            execution_id=execution,
            batch_index=batch_index,
            args=args,
        )

    # The SQL host may keep only the last sink ID; all batches must survive.
    ids = [
        ZscoreAnomalies.process(pa.record_batch({"v": [0.0] * 10}), params()),
        ZscoreAnomalies.process(pa.record_batch({"v": [0.0] * 9 + [100.0]}), params()),
    ]
    ids = ids[-1:]
    monkeypatch.setattr(detectors, "MAX_SERIES_LEN", 15)
    with pytest.raises(ValueError, match="series too long"):
        ZscoreAnomalies.combine(ids, params())
    monkeypatch.setattr(detectors, "MAX_SERIES_LEN", 20)
    ZscoreAnomalies.combine(ids, params())
    assert params(b"other-query").storage.state_get(b"results", (0).to_bytes(8, "big")) is None

    output = []
    collector = SimpleNamespace(emit=output.append, finish=lambda: output.append("finished"))
    state = DrainState()
    ZscoreAnomalies.finalize(params(), b"results", state, collector)
    # Serialize the cursor just as HTTP does between ticks.
    state = DrainState.deserialize_from_bytes(state.serialize_to_bytes())
    ZscoreAnomalies.finalize(params(), b"results", state, collector)
    assert output[0].to_pylist() == [{"index": 19}]
    assert output[1] == "finished"


def test_limit_is_enforced_during_sink(tmp_path, monkeypatch):
    params = SimpleNamespace(
        storage=BoundStorage(FunctionStorageSqlite(str(tmp_path / "state.sqlite")), b"query"), execution_id=b"query"
    )
    monkeypatch.setattr(detectors, "MAX_SERIES_LEN", 3)
    ZscoreAnomalies.process(pa.record_batch({"v": [1.0, 2.0]}), params)
    assert ZscoreAnomalies.process(pa.record_batch({"v": pa.array([], type=pa.float64())}), params) == b""
    with pytest.raises(ValueError, match="series too long"):
        ZscoreAnomalies.process(pa.record_batch({"v": [3.0, 4.0]}), params)
