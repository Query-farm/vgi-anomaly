# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.8.4",
#     "stumpy>=1.12",
#     "ruptures>=1.1.9",
#     "numpy",
#     "pyarrow",
# ]
# ///
"""VGI worker exposing time-series anomaly detection to DuckDB/SQL.

Assembles the detectors in ``vgi_anomaly`` into a single ``anomaly`` catalog and
runs the worker over stdio (DuckDB subprocess) or HTTP. It does matrix-profile
motif/discord discovery (``stumpy``), change-point detection (``ruptures``), and
a light z-score fallback over numeric series, all as DuckDB scalar functions.

Each function takes a whole series as a single ``DOUBLE[]`` argument (build it in
SQL with ``array_agg(value ORDER BY t)``) so it composes without subquery table
arguments.

Usage:
    uv run anomaly_worker.py            # serve over stdio (DuckDB subprocess)

    INSTALL vgi FROM community; LOAD vgi;
    ATTACH 'anomaly' (TYPE vgi, LOCATION 'uv run anomaly_worker.py');

    -- top discord (anomaly) / motif start index for window 50:
    SELECT anomaly.discord_index(array_agg(v ORDER BY t), 50) FROM series;
    SELECT anomaly.motif_index(array_agg(v ORDER BY t), 50)   FROM series;
    -- full matrix profile (length = N - window + 1):
    SELECT anomaly.matrix_profile(array_agg(v ORDER BY t), 50) FROM series;
    -- change points (auto count, or a fixed number):
    SELECT anomaly.change_points(array_agg(v ORDER BY t))     FROM series;
    SELECT anomaly.change_points(array_agg(v ORDER BY t), 2)  FROM series;
    -- light z-score outliers beyond 3 sigma:
    SELECT UNNEST(anomaly.zscore_anomalies(array_agg(v ORDER BY t), 3.0)) FROM series;

    -- literal series also work:
    SELECT anomaly.discord_index([1.0, 2.0, 3.0, 100.0, 2.0, 3.0]::DOUBLE[], 3);
"""

from __future__ import annotations

from typing import Any

from vgi import Worker
from vgi.catalog import Catalog, Schema

from vgi_anomaly import detectors
from vgi_anomaly.scalars import SCALAR_FUNCTIONS

_CATALOG_DESCRIPTION_LLM = (
    "Time-series anomaly detection over a numeric series passed as a single DOUBLE[] argument "
    "(build it in SQL with array_agg(value ORDER BY t)). Find the most anomalous subsequence "
    "(discord) and the most repeated pattern (motif) with the matrix profile, compute the full "
    "matrix profile, detect change points (regime shifts) with ruptures PELT/Dynp, and flag "
    "individual outliers beyond a z-score threshold. Use for outlier detection, motif/discord "
    "discovery, regime-change detection, and series quality checks in SQL."
)

_CATALOG_DESCRIPTION_MD = (
    "# anomaly\n\n"
    "Time-series anomaly detection for DuckDB/SQL, backed by `stumpy` (matrix profile), "
    "`ruptures` (change points) and `numpy` (z-score).\n\n"
    "Each function takes a whole numeric series as a single `DOUBLE[]` argument "
    "(`array_agg(value ORDER BY t)`).\n\n"
    "Scalars: `matrix_profile`, `discord_index`, `motif_index`, `change_points`, "
    "`zscore_anomalies`."
)

_SCHEMA_DESCRIPTION_LLM = (
    "Anomaly-detection scalar functions operating on a whole numeric series (DOUBLE[]): "
    "matrix profile, top discord (anomaly) index, top motif index, change-point detection, "
    "and z-score outlier indices."
)

_SCHEMA_DESCRIPTION_MD = (
    "Time-series anomaly-detection scalar functions: matrix profile, discords, motifs, "
    "change points, and z-score outliers."
)

_ANOMALY_CATALOG = Catalog(
    name="anomaly",
    default_schema="main",
    comment="Time-series anomaly detection: matrix profile, change points, z-score for SQL.",
    source_url="https://github.com/Query-farm/vgi-anomaly",
    tags={
        "vgi.description_llm": _CATALOG_DESCRIPTION_LLM,
        "vgi.description_md": _CATALOG_DESCRIPTION_MD,
        "vgi.author": "Query.Farm",
        "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
        "vgi.license": "MIT",
        "vgi.support_contact": "https://github.com/Query-farm/vgi-anomaly/issues",
        "vgi.support_policy_url": "https://github.com/Query-farm/vgi-anomaly/blob/main/README.md",
    },
    schemas=[
        Schema(
            name="main",
            comment="Time-series anomaly detection: matrix profile, change points, z-score for SQL",
            tags={
                "vgi.description_llm": _SCHEMA_DESCRIPTION_LLM,
                "vgi.description_md": _SCHEMA_DESCRIPTION_MD,
            },
            functions=list(SCALAR_FUNCTIONS),
        ),
    ],
)


class AnomalyWorker(Worker):
    """Worker process hosting the ``anomaly`` catalog."""

    catalog = _ANOMALY_CATALOG

    def run(self, otel_config: Any = None) -> None:
        """JIT-compile the numba kernels once, then serve.

        ``stumpy.stump`` is numba-JIT compiled, so without warming the first real
        query of every ATTACH pays a multi-second compile inline -- a window in
        which a worker-pool teardown SIGTERM (or a loaded host) can kill the run
        mid-assertion and record a spurious E2E failure. Warming at spawn moves
        that one-time cost ahead of any query, keeping the SQL suite deterministic
        without changing any output. Best-effort; never fatal.
        """
        detectors.warm_up()
        super().run(otel_config=otel_config)


def main() -> None:
    """Run the anomaly worker process (stdio or, via flags, HTTP)."""
    AnomalyWorker.main()


if __name__ == "__main__":
    main()
