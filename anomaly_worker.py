# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.8.5",
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

import json
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
    "# Time-Series Anomaly Detection in SQL\n\n"
    "![STUMPY logo](https://raw.githubusercontent.com/TDAmeritrade/stumpy/main/docs/images/stumpy_logo_small.png)\n\n"
    "Find anomalies, motifs, discords, and regime changes in time-series data "
    "directly in DuckDB SQL — no Python notebook, no data export, just `matrix_profile`, "
    "`discord_index`, `motif_index`, `change_points`, and `zscore_anomalies` over any "
    "numeric series.\n\n"
    "The `anomaly` extension brings industrial-strength time-series anomaly detection to "
    "your SQL queries. It is for data engineers, analysts, and SREs who keep their "
    "metrics, sensor readings, financial ticks, and log volumes in DuckDB and want to "
    "spot the unusual subsequence, the repeated pattern, the structural break, or the "
    "single outlier without leaving the database. Every detector runs server-side inside "
    "a VGI worker that DuckDB attaches over Apache Arrow, so the same analysis that "
    "normally lives in a data-science script becomes an ordinary, composable SQL function.\n\n"
    "Under the hood the extension stands on three best-in-class scientific Python "
    "libraries. Matrix-profile analysis — the modern foundation for motif and discord "
    "discovery — is powered by [STUMPY](https://github.com/TDAmeritrade/stumpy) "
    "([documentation](https://stumpy.readthedocs.io/)), whose numba-accelerated "
    "implementation of the [Matrix Profile](https://www.cs.ucr.edu/~eamonn/MatrixProfile.html) "
    "computes the distance from every subsequence to its nearest neighbor. Change-point "
    "(regime-shift) detection uses [ruptures](https://github.com/deepcharles/ruptures) "
    "([documentation](https://centre-borelli.github.io/ruptures-docs/)) with its PELT and "
    "Dynp search algorithms, and the lightweight point-outlier check is plain "
    "[NumPy](https://numpy.org/) ([documentation](https://numpy.org/doc/stable/)) "
    "z-scoring.\n\n"
    "Each function takes a whole numeric series as a single `DOUBLE[]` argument that you "
    "build in SQL with `array_agg(value ORDER BY t)` (or a `[...]::DOUBLE[]` literal), so "
    "the detectors compose cleanly and run once per group. Use `matrix_profile(series, "
    "window)` to compute per-subsequence nearest-neighbor distances; `discord_index(series, "
    "window)` to locate the single most anomalous subsequence; `motif_index(series, window)` "
    "to find the most-repeated pattern; `change_points(series)` or `change_points(series, "
    "n_bkps)` to detect regime shifts via ruptures; and `zscore_anomalies(series, threshold)` "
    "to flag individual points beyond a sigma threshold. Typical use cases include outlier "
    "detection on metrics and IoT sensor data, motif/discord discovery in financial or "
    "operational series, segmentation of telemetry into stable regimes, and series quality "
    "checks — all expressed as plain SQL."
)

_SCHEMA_DESCRIPTION_LLM = (
    "Anomaly-detection scalar functions operating on a whole numeric series (DOUBLE[]): "
    "matrix profile, top discord (anomaly) index, top motif index, change-point detection, "
    "and z-score outlier indices."
)

_SCHEMA_DESCRIPTION_MD = (
    "Time-series anomaly-detection scalar functions for SQL. Each takes a whole numeric "
    "series (built with `array_agg(value ORDER BY t)`) and returns an index or array: "
    "`matrix_profile` (per-subsequence distances), `discord_index` (top anomaly), "
    "`motif_index` (top repeated pattern), `change_points` (regime shifts via ruptures), "
    "and `zscore_anomalies` (point outliers beyond a sigma threshold). Use these for "
    "outlier detection, motif/discord discovery, and regime-change analysis in SQL."
)

_ANOMALY_CATALOG = Catalog(
    name="anomaly",
    default_schema="main",
    comment="Time-series anomaly detection: matrix profile, change points, z-score for SQL.",
    source_url="https://github.com/Query-farm/vgi-anomaly",
    tags={
        "vgi.title": "Time-Series Anomaly Detection",
        "vgi.keywords": json.dumps(
            [
                "anomaly detection",
                "time series",
                "matrix profile",
                "stumpy",
                "discord",
                "motif",
                "change point",
                "ruptures",
                "PELT",
                "z-score",
                "outlier",
                "regime change",
                "segmentation",
            ]
        ),
        "vgi.doc_llm": _CATALOG_DESCRIPTION_LLM,
        "vgi.doc_md": _CATALOG_DESCRIPTION_MD,
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
                "vgi.title": "Anomaly — main",
                "vgi.keywords": json.dumps(
                    [
                        "anomaly",
                        "matrix profile",
                        "discord",
                        "motif",
                        "change points",
                        "z-score",
                        "outlier",
                        "regime change",
                        "time series",
                        "segmentation",
                        "stumpy",
                        "ruptures",
                    ]
                ),
                "vgi.doc_llm": _SCHEMA_DESCRIPTION_LLM,
                "vgi.doc_md": _SCHEMA_DESCRIPTION_MD,
                # VGI123 classifying tags use BARE keys (not vgi.-namespaced).
                "domain": "time-series",
                "category": "anomaly-detection",
                "topic": "matrix-profile-and-change-points",
                # VGI506 representative example queries (catalog-qualified, runnable).
                "vgi.example_queries": (
                    "SELECT anomaly.main.matrix_profile("
                    "[1.0,2.0,3.0,4.0,3.0,2.0,1.0,2.0,3.0,4.0]::DOUBLE[], 4);\n"
                    "SELECT anomaly.main.discord_index("
                    "[1.0,2.0,3.0,4.0,3.0,2.0,1.0,2.0,3.0,4.0,3.0,2.0,1.0,2.0,3.0,4.0,"
                    "3.0,2.0,50.0,2.0,3.0,4.0,3.0,2.0,1.0]::DOUBLE[], 4);\n"
                    "SELECT anomaly.main.motif_index("
                    "[0.0,2.0,4.0,2.0,0.0,0.1,0.116,0.133,0.15,0.166,0.183,0.2,"
                    "0.0,2.0,4.0,2.0,0.0,0.3,0.4,0.5]::DOUBLE[], 5);\n"
                    "SELECT anomaly.main.change_points("
                    "[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0]"
                    "::DOUBLE[]);\n"
                    "SELECT anomaly.main.zscore_anomalies("
                    "[10.0,10.0,11.0,9.0,10.0,40.0,10.0,9.0,11.0]::DOUBLE[], 2.0);"
                ),
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
