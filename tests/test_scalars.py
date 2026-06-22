"""End-to-end tests for the scalar anomaly functions via the VGI client.

These spawn ``anomaly_worker.py`` as a subprocess through ``vgi.client.Client``
and call each scalar exactly as DuckDB would after ``ATTACH``: the series travels
as a ``DOUBLE[]`` input column (a ``Param``), and the constant ``window`` /
``n_bkps`` / ``threshold`` arguments go in ``positional``. NULL rows in the input
column must come back as NULL.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from vgi import Arguments
from vgi.client import Client

_WORKER = str(Path(__file__).resolve().parent.parent / "anomaly_worker.py")
_LIST_DOUBLE = pa.list_(pa.float64())


@pytest.fixture(scope="module")
def client() -> Iterator[Client]:
    # worker_limit=1 so output order matches input order for deterministic
    # per-row assertions.
    with Client(f"{sys.executable} {_WORKER}", worker_limit=1) as c:
        yield c


def _scalar(
    client: Client,
    name: str,
    series_rows: list[list[float] | None],
    *,
    positional: list[pa.Scalar] | None = None,
) -> list:
    batch = pa.RecordBatch.from_pydict({"s": pa.array(series_rows, type=_LIST_DOUBLE)})
    results = list(
        client.scalar_function(
            function_name=name,
            input=iter([batch]),
            arguments=Arguments(positional=positional or []),
        )
    )
    return results[0]["result"].to_pylist()


def _i64(x: int) -> pa.Scalar:
    return pa.scalar(x, type=pa.int64())


def _sine_with_spike(n: int = 200, spike_at: int = 120, height: float = 6.0) -> tuple[list[float], int]:
    s = np.sin(np.linspace(0, 8 * math.pi, n)).tolist()
    s[spike_at] += height
    return s, spike_at


class TestMatrixProfile:
    def test_length_and_nulls(self, client: Client) -> None:
        s, _ = _sine_with_spike()
        w = 20
        out = _scalar(client, "matrix_profile", [s, None], positional=[_i64(w)])
        assert len(out[0]) == len(s) - w + 1
        assert out[1] is None

    def test_discord_on_spike(self, client: Client) -> None:
        s, spike_at = _sine_with_spike()
        w = 20
        out = _scalar(client, "discord_index", [s], positional=[_i64(w)])
        di = out[0]
        assert di <= spike_at <= di + w - 1

    def test_motif_finds_pattern(self, client: Client) -> None:
        rng = np.random.RandomState(42)
        pat = [0.0, 1.0, 3.0, 5.0, 3.0, 1.0, 0.0, -2.0]
        w = len(pat)
        noise = lambda k: list(rng.randn(k) * 0.05)  # noqa: E731
        series = noise(10) + pat + noise(12) + pat + noise(10)
        out = _scalar(client, "motif_index", [series], positional=[_i64(w)])
        assert out[0] in (10, 30) or abs(out[0] - 10) <= 1 or abs(out[0] - 30) <= 1

    def test_window_too_large_errors(self, client: Client) -> None:
        from vgi.client import ClientError

        with pytest.raises(ClientError):
            _scalar(client, "matrix_profile", [[1.0, 2.0, 3.0]], positional=[_i64(3)])


class TestChangePoints:
    def test_step_auto(self, client: Client) -> None:
        step = [0.0] * 30 + [10.0] * 30
        out = _scalar(client, "change_points", [step])
        assert out[0] == [30]

    def test_step_fixed(self, client: Client) -> None:
        step = [0.0] * 30 + [10.0] * 30
        out = _scalar(client, "change_points", [step], positional=[_i64(1)])
        assert out[0] == [30]

    def test_null_row(self, client: Client) -> None:
        out = _scalar(client, "change_points", [None])
        assert out[0] is None


class TestZscore:
    def test_flags_outlier(self, client: Client) -> None:
        z = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 8.0, 1.0, 0.98]
        out = _scalar(client, "zscore_anomalies", [z], positional=[pa.scalar(2.0, type=pa.float64())])
        assert out[0] == [6]

    def test_empty_row(self, client: Client) -> None:
        out = _scalar(client, "zscore_anomalies", [[]], positional=[pa.scalar(3.0, type=pa.float64())])
        assert out[0] is None
