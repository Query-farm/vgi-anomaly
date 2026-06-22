"""Unit tests for the pure detector logic (no Arrow / VGI).

Every assertion uses a *constructed* series whose answer is known by design: a
sine wave with one injected spike (discord), a pattern repeated twice (motif), a
step function (change point), a single injected outlier (z-score). These exercise
the algorithms directly and fast.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vgi_anomaly import detectors as d


def _sine_with_spike(n: int = 200, spike_at: int = 120, height: float = 6.0) -> tuple[list[float], int]:
    s = np.sin(np.linspace(0, 8 * math.pi, n))
    s = s.tolist()
    s[spike_at] += height
    return s, spike_at


def _repeated_pattern() -> tuple[list[float], int, int, int]:
    rng = np.random.RandomState(42)
    pat = [0.0, 1.0, 3.0, 5.0, 3.0, 1.0, 0.0, -2.0]
    w = len(pat)
    noise = lambda k: list(rng.randn(k) * 0.05)  # noqa: E731
    p1 = 10
    series = noise(p1) + pat + noise(12) + pat + noise(10)
    p2 = p1 + w + 12
    return series, w, p1, p2


class TestMatrixProfile:
    def test_length(self) -> None:
        s, _ = _sine_with_spike()
        w = 20
        prof = d.matrix_profile(s, w)
        assert prof is not None
        assert len(prof) == len(s) - w + 1
        assert all(isinstance(x, float) for x in prof)

    def test_discord_lands_on_injected_spike(self) -> None:
        s, spike_at = _sine_with_spike()
        w = 20
        di = d.discord_index(s, w)
        assert di is not None
        # The most anomalous window must contain the injected spike.
        assert di <= spike_at <= di + w - 1

    def test_motif_finds_repeated_pattern(self) -> None:
        series, w, p1, p2 = _repeated_pattern()
        mi = d.motif_index(series, w)
        assert mi is not None
        # The top motif starts at (or adjacent to) one of the two copies.
        assert mi in (p1, p2) or abs(mi - p1) <= 1 or abs(mi - p2) <= 1


class TestChangePoints:
    def test_step_auto(self) -> None:
        step = [0.0] * 30 + [10.0] * 30
        assert d.change_points(step) == [30]

    def test_step_fixed_n(self) -> None:
        step = [0.0] * 30 + [10.0] * 30
        assert d.change_points(step, 1) == [30]

    def test_two_steps(self) -> None:
        two = [0.0] * 20 + [5.0] * 20 + [0.0] * 20
        assert d.change_points(two) == [20, 40]
        assert d.change_points(two, 2) == [20, 40]

    def test_noisy_step_within_tolerance(self) -> None:
        rng = np.random.RandomState(1)
        ns = np.concatenate([rng.randn(40) * 0.5, rng.randn(40) * 0.5 + 5.0]).tolist()
        cps = d.change_points(ns, 1)
        assert cps is not None and len(cps) == 1
        assert abs(cps[0] - 40) <= 2

    def test_constant_series_no_change(self) -> None:
        assert d.change_points([1.0] * 10) == []

    def test_nbkps_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            d.change_points([0.0, 1.0, 2.0, 3.0], 0)
        with pytest.raises(ValueError):
            d.change_points([0.0, 1.0, 2.0, 3.0], 4)


class TestZscore:
    def test_flags_injected_outlier(self) -> None:
        z = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 8.0, 1.0, 0.98]
        assert d.zscore_anomalies(z, 2.0) == [6]

    def test_constant_series_flags_nothing(self) -> None:
        assert d.zscore_anomalies([5.0] * 8, 3.0) == []

    def test_nonpositive_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            d.zscore_anomalies([1.0, 2.0, 3.0], 0.0)


class TestEdges:
    def test_none_and_empty(self) -> None:
        for fn in (
            lambda v: d.matrix_profile(v, 3),
            lambda v: d.discord_index(v, 3),
            lambda v: d.motif_index(v, 3),
            d.change_points,
            lambda v: d.zscore_anomalies(v, 3.0),
        ):
            assert fn(None) is None
            assert fn([]) is None

    def test_window_too_large_raises(self) -> None:
        with pytest.raises(ValueError):
            d.matrix_profile([1.0, 2.0, 3.0], 3)
        with pytest.raises(ValueError):
            d.discord_index([1.0, 2.0, 3.0, 4.0], 5)

    def test_window_too_small_raises(self) -> None:
        with pytest.raises(ValueError):
            d.matrix_profile([1.0, 2.0, 3.0, 4.0], 2)

    def test_non_finite_series_is_null(self) -> None:
        assert d.discord_index([1.0, float("nan"), 3.0, 4.0, 5.0], 3) is None
        assert d.zscore_anomalies([1.0, float("inf"), 3.0], 2.0) is None

    def test_null_sample_is_null(self) -> None:
        assert d.discord_index([1.0, None, 3.0, 4.0, 5.0], 3) is None  # type: ignore[list-item]

    def test_too_long_series_raises(self) -> None:
        with pytest.raises(ValueError):
            d._clean([0.0] * (d.MAX_SERIES_LEN + 1))


def test_warm_up_is_safe() -> None:
    # Must not raise even though it JIT-compiles the numba kernels.
    d.warm_up()
