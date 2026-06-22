"""Time-series anomaly detection as a VGI worker.

The implementation is split so each concern stays focused:

- ``detectors`` -- pure anomaly-detection logic over ``stumpy`` (matrix profile),
  ``ruptures`` (change points) and ``numpy``; no Arrow or VGI dependency,
  directly unit-testable. Holds the warm-up that JIT-compiles numba kernels.
- ``scalars``   -- per-row VGI scalar functions operating on a whole numeric
  series passed as a ``DOUBLE[]`` argument (LIST-in; index or LIST out). The
  optional ``n_bkps`` of ``change_points`` is an arity overload.

``anomaly_worker.py`` at the repo root assembles these into the ``anomaly``
catalog and runs the worker over stdio (or HTTP), warming numba at startup.
"""

from __future__ import annotations

__version__ = "0.1.0"
