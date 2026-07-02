"""Pure time-series anomaly-detection logic (no Arrow, no VGI).

Every function here takes a plain Python ``list[float]`` (a numeric *series*, in
order) and returns a plain Python result -- a list of floats, a list of ints, or
a single int -- so the whole module is directly unit-testable without spawning a
worker or touching DuckDB/Arrow.

Three families of detector are exposed:

- **Matrix profile** (``matrix_profile`` / ``discord_index`` / ``motif_index``)
  via :mod:`stumpy`'s ``stump``. The matrix profile is, for every length-``w``
  subsequence, the z-normalized Euclidean distance to its nearest non-trivial
  neighbour. A *discord* (largest distance) is the most anomalous subsequence; a
  *motif* (smallest distance) is the most repeated pattern.
- **Change-point detection** (``change_points``) via :mod:`ruptures`. With no
  ``n_bkps`` we run PELT (``model="rbf"``) with an automatic penalty derived from
  the series length and variance; with an explicit ``n_bkps`` we run dynamic
  programming (``Dynp``, same model) for exactly that many breakpoints. Returned
  indices are interior change positions (the trailing len(series) sentinel that
  ruptures appends is dropped).
- **Z-score** (``zscore_anomalies``) -- a light, dependency-free fallback: the
  indices whose value is more than ``threshold`` sample standard deviations from
  the mean.

Robustness contract (shared by every function):

- A series that is ``None``, empty, or too short for the requested operation
  returns ``None`` (the SQL layer surfaces that as a SQL ``NULL``).
- ``window`` must satisfy ``3 <= window < len(series)``; otherwise the matrix
  profile functions raise :class:`ValueError` (the SQL layer surfaces a clear
  error). ``n_bkps`` must satisfy ``1 <= n_bkps < len(series)``.
- Non-finite samples (``NaN`` / ``inf``) make the series invalid -> ``None``.
- Inputs longer than :data:`MAX_SERIES_LEN` are rejected with ``ValueError`` to
  bound worst-case CPU/memory (the matrix profile is O(n^2)).
"""

from __future__ import annotations

import math
import threading

import numpy as np
import ruptures as rpt
import stumpy

# ``stumpy.stump`` is a numba ``parallel=True`` kernel. Numba's default
# "workqueue" threading layer is NOT re-entrant: calling a parallel kernel from
# more than one Python thread at once aborts the whole process ("Concurrent
# access has been detected"). A VGI worker serves DuckDB/lint queries across
# several cursors concurrently, so guard every stumpy call with a process-wide
# lock. The matrix profile itself already parallelises internally across cores,
# so serialising the (typically sub-second) calls costs almost nothing while
# making the worker crash-proof under concurrent SQL.
_STUMP_LOCK = threading.Lock()

# A length-n series costs O(n^2) for the matrix profile; cap it so a single
# pathological row cannot wedge the worker. 1e6 samples is far above any sane
# interactive use and still completes in seconds.
MAX_SERIES_LEN = 1_000_000

# Minimum subsequence window the matrix profile accepts. stumpy itself requires
# m >= 3 for a meaningful z-normalized distance.
MIN_WINDOW = 3


def _clean(values: list[float] | None) -> np.ndarray | None:
    """Validate and convert a raw series to a float64 ndarray, or ``None``.

    Returns ``None`` for a missing/empty series or one containing a NULL or a
    non-finite (NaN/inf) sample. Raises ``ValueError`` if the series is longer
    than :data:`MAX_SERIES_LEN`.
    """
    if values is None:
        return None
    n = len(values)
    if n == 0:
        return None
    if n > MAX_SERIES_LEN:
        raise ValueError(f"series too long: {n} > MAX_SERIES_LEN ({MAX_SERIES_LEN})")
    if any(v is None for v in values):
        return None
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _check_window(window: int, n: int) -> None:
    """Raise ``ValueError`` unless ``MIN_WINDOW <= window < n``."""
    if window < MIN_WINDOW:
        raise ValueError(f"window must be >= {MIN_WINDOW}, got {window}")
    if window >= n:
        raise ValueError(f"window ({window}) must be < series length ({n})")


def _profile(arr: np.ndarray, window: int) -> np.ndarray:
    """Matrix-profile distances (column 0 of stumpy.stump) as float64.

    Serialised with :data:`_STUMP_LOCK`: ``stumpy.stump`` is a numba
    ``parallel=True`` kernel and numba's default workqueue threading layer is
    not safe to enter from multiple Python threads at once, so concurrent worker
    cursors must not run two matrix-profile computations simultaneously.
    """
    with _STUMP_LOCK:
        mp = stumpy.stump(arr, m=window)
    return np.asarray(mp[:, 0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Matrix profile
# ---------------------------------------------------------------------------


def matrix_profile(values: list[float] | None, window: int) -> list[float] | None:
    """The matrix profile of ``values`` for subsequence length ``window``.

    The result has length ``len(values) - window + 1``; element ``i`` is the
    z-normalized Euclidean distance from subsequence ``values[i:i+window]`` to
    its nearest non-trivial neighbour. ``None`` for an invalid/short series;
    raises ``ValueError`` for a bad window.
    """
    arr = _clean(values)
    if arr is None:
        return None
    _check_window(window, len(arr))
    result: list[float] = _profile(arr, window).tolist()
    return result


def discord_index(values: list[float] | None, window: int) -> int | None:
    """Start index of the top discord (most anomalous subsequence).

    The discord is the subsequence with the **largest** matrix-profile value.
    Returns its start index, or ``None`` for an invalid/short series; raises
    ``ValueError`` for a bad window.
    """
    arr = _clean(values)
    if arr is None:
        return None
    _check_window(window, len(arr))
    prof = _profile(arr, window)
    return int(np.argmax(prof))


def motif_index(values: list[float] | None, window: int) -> int | None:
    """Start index of the top motif (most repeated subsequence).

    The motif is the subsequence with the **smallest** matrix-profile value.
    Returns its start index, or ``None`` for an invalid/short series; raises
    ``ValueError`` for a bad window.
    """
    arr = _clean(values)
    if arr is None:
        return None
    _check_window(window, len(arr))
    prof = _profile(arr, window)
    return int(np.argmin(prof))


# ---------------------------------------------------------------------------
# Change-point detection
# ---------------------------------------------------------------------------


def _auto_penalty(arr: np.ndarray) -> float:
    """A BIC-style ``log(n)`` PELT penalty for the ``rbf`` cost.

    The ``rbf`` (kernel) cost is normalized to roughly the unit scale regardless
    of the signal's amplitude, so the penalty should *not* be scaled by variance
    (that would over-penalize a large-amplitude step and miss it). The standard
    ``log(n)`` model-complexity term alone cleanly recovers genuine shifts in
    mean/distribution on both clean and noisy series while resisting
    over-segmentation.
    """
    return math.log(len(arr))


def change_points(values: list[float] | None, n_bkps: int | None = None) -> list[int] | None:
    """Change-point indices in ``values``.

    With ``n_bkps is None`` we run PELT (``model="rbf"``) with an automatic
    penalty (see :func:`_auto_penalty`) and return however many change points it
    finds. With an explicit ``n_bkps`` we run dynamic programming (``Dynp``, same
    model) for exactly that many breakpoints. The returned indices are the
    *interior* breakpoints (the trailing ``len(values)`` sentinel ruptures
    appends is dropped), each the index of the first sample of a new segment.

    ``None`` for an invalid/short series. Raises ``ValueError`` if ``n_bkps`` is
    out of range (``1 <= n_bkps < len(values)``).
    """
    arr = _clean(values)
    if arr is None:
        return None
    n = len(arr)
    if n < 2:
        return None
    signal = arr.reshape(-1, 1)

    if n_bkps is None:
        algo = rpt.Pelt(model="rbf", min_size=1, jump=1).fit(signal)
        bkps = algo.predict(pen=_auto_penalty(arr))
    else:
        if n_bkps < 1 or n_bkps >= n:
            raise ValueError(f"n_bkps ({n_bkps}) must satisfy 1 <= n_bkps < length ({n})")
        algo = rpt.Dynp(model="rbf", min_size=1, jump=1).fit(signal)
        bkps = algo.predict(n_bkps=n_bkps)

    # ruptures always terminates the list with n (the end of the signal); drop it.
    return [int(b) for b in bkps if b < n]


# ---------------------------------------------------------------------------
# Z-score (light, dependency-free)
# ---------------------------------------------------------------------------


def zscore_anomalies(values: list[float] | None, threshold: float) -> list[int] | None:
    """Indices whose value is more than ``threshold`` std devs from the mean.

    Uses the population standard deviation. A non-positive ``threshold`` raises
    ``ValueError``. A constant series (zero std dev) flags nothing. ``None`` for
    an invalid/empty series.
    """
    arr = _clean(values)
    if arr is None:
        return None
    if threshold <= 0 or not math.isfinite(threshold):
        raise ValueError(f"threshold must be a positive finite number, got {threshold}")
    std = float(np.std(arr))
    if std == 0.0:
        return []
    mean = float(np.mean(arr))
    z = np.abs((arr - mean) / std)
    return [int(i) for i in np.flatnonzero(z > threshold)]


def warm_up() -> None:
    """Trigger numba JIT compilation of stumpy + ruptures at worker startup.

    ``stumpy.stump`` is numba-JIT compiled, so the *first* real call pays a
    multi-second compilation cost inline. Under the end-to-end SQL suite that
    compile happens while the runner is mid-assertion on the first file -- a long
    window in which a worker-pool teardown SIGTERM (or a loaded host) can kill the
    run and record a spurious failure, making an otherwise deterministic suite
    flaky. Running each entry point once here on a tiny dummy series moves that
    one-time cost to process spawn (before any query), so the first real query is
    fast. It only warms caches -- it never changes any output. Best-effort:
    failures are swallowed.
    """
    try:
        dummy = [float(x) for x in (1, 2, 3, 4, 5, 6, 4, 2, 1, 2, 3, 4)]
        matrix_profile(dummy, 4)
        discord_index(dummy, 4)
        motif_index(dummy, 4)
        change_points(dummy)
        change_points(dummy, 1)
        zscore_anomalies(dummy, 3.0)
    except Exception:
        pass
