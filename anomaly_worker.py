# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.16.0",
#     "stumpy>=1.12",
#     "ruptures>=1.1.9",
#     "numpy",
#     "pyarrow",
# ]
# ///
"""Repo-root stdio entry point for the VGI anomaly worker (thin PEP 723 shim).

The worker itself -- the ``anomaly`` catalog, the :class:`AnomalyWorker` class,
and :func:`main` -- lives in the wheel-importable :mod:`vgi_anomaly.worker`
module. This file is only a launcher so that ``uv run anomaly_worker.py`` (used
by the Makefile, ``ci/run-integration.sh``, and the ``tests/test_scalars.py``
subprocess harness) keeps resolving the inline PEP 723 dependencies and serving
the worker exactly as before. The installed package (Docker image / the
``vgi-anomaly-worker`` console script) drives the same code via
``vgi_anomaly.worker:AnomalyWorker`` / ``vgi_anomaly.worker:main``.

Usage:
    uv run anomaly_worker.py            # serve over stdio (DuckDB subprocess)

    INSTALL vgi FROM community; LOAD vgi;
    ATTACH 'anomaly' (TYPE vgi, LOCATION 'uv run anomaly_worker.py');

    -- top discord (anomaly) / motif start index for window 50:
    SELECT anomaly.discord_index(array_agg(v ORDER BY t), 50) FROM series;
"""

from __future__ import annotations

from vgi_anomaly.worker import AnomalyWorker, main

__all__ = ["AnomalyWorker", "main"]


if __name__ == "__main__":
    main()
