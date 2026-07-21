# Copyright 2026 Query Farm LLC - https://query.farm

"""VGI worker exposing time-series anomaly detection to DuckDB/SQL.

Assembles the detectors in ``vgi_anomaly`` into a single ``anomaly`` catalog and
runs the worker over stdio (DuckDB subprocess) or HTTP. It does matrix-profile
motif/discord discovery (``stumpy``), change-point detection (``ruptures``), and
a light z-score fallback over numeric series, all as DuckDB scalar functions.

Each function takes a whole series as a single ``DOUBLE[]`` argument (build it in
SQL with ``array_agg(value ORDER BY t)``) so it composes without subquery table
arguments.

This module is wheel-importable: it holds the catalog, the :class:`AnomalyWorker`
class, and :func:`main`. The repo-root ``anomaly_worker.py`` is a thin PEP 723
shim that re-exports these so ``uv run anomaly_worker.py`` keeps working while the
installed package (Docker image / console script) drives the same worker via
``vgi_anomaly.worker:AnomalyWorker`` / ``vgi_anomaly.worker:main``.

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
    -- light z-score outliers beyond a threshold (or the default 3 sigma):
    SELECT UNNEST(anomaly.zscore_anomalies(array_agg(v ORDER BY t), 3.0)) FROM series;
    SELECT UNNEST(anomaly.zscore_anomalies(array_agg(v ORDER BY t)))      FROM series;

    -- literal series also work:
    SELECT anomaly.discord_index([1.0, 2.0, 3.0, 100.0, 2.0, 3.0]::DOUBLE[], 3);
"""

from __future__ import annotations

import json
import logging
import threading

from vgi import Worker
from vgi.catalog import Catalog, Schema

from vgi_anomaly import detectors
from vgi_anomaly.scalars import SCALAR_FUNCTIONS

_CATALOG_DESCRIPTION_LLM = (
    "Time-series anomaly detection over a numeric series passed as a single `DOUBLE[]` argument "
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
    "directly in DuckDB SQL — no Python notebook, no data export, no round trip to a "
    "separate data-science stack.\n\n"
    "## What it is\n\n"
    "The `anomaly` extension brings industrial-strength time-series anomaly detection to "
    "your SQL queries. It is for data engineers, analysts, and SREs who keep their "
    "metrics, sensor readings, financial ticks, and log volumes in DuckDB and want to "
    "spot the unusual subsequence, the repeated pattern, the structural break, or the "
    "single outlier without leaving the database. Every detector runs server-side inside "
    "a VGI worker that DuckDB attaches over Apache Arrow, so the same analysis that "
    "normally lives in a data-science script becomes an ordinary, composable SQL "
    "function.\n\n"
    "## Key concepts\n\n"
    "- **Matrix profile** — for every fixed-length subsequence, its z-normalized "
    "distance to its nearest neighbour. The largest distances are *discords* "
    "(anomalies); the smallest are *motifs* (repeated patterns). This is the modern "
    "foundation for shape-based anomaly and pattern discovery.\n"
    "- **Change points** — structural breaks where a series shifts from one regime to "
    "another (a level shift, a variance change), found by penalized segmentation.\n"
    "- **Point outliers** — individual samples that sit far from the series mean, a "
    "light first pass that needs no subsequence analysis.\n\n"
    "## When to use it\n\n"
    "Reach for this worker for outlier detection on metrics and IoT sensor data, "
    "motif/discord discovery in financial or operational series, segmentation of "
    "telemetry into stable regimes, and general series quality checks — all expressed as "
    "plain, composable SQL over a series you assemble with `array_agg`.\n\n"
    "## Built on\n\n"
    "Under the hood the extension stands on three best-in-class scientific Python "
    "libraries. Matrix-profile analysis is powered by "
    "[STUMPY](https://github.com/TDAmeritrade/stumpy) "
    "([documentation](https://stumpy.readthedocs.io/)), whose numba-accelerated "
    "implementation of the [Matrix Profile](https://www.cs.ucr.edu/~eamonn/MatrixProfile.html) "
    "computes the distance from every subsequence to its nearest neighbour. Change-point "
    "(regime-shift) detection uses [ruptures](https://github.com/deepcharles/ruptures) "
    "([documentation](https://centre-borelli.github.io/ruptures-docs/)) with its PELT and "
    "dynamic-programming search algorithms, and the lightweight point-outlier check is "
    "plain [NumPy](https://numpy.org/) ([documentation](https://numpy.org/doc/stable/)) "
    "z-scoring."
)

_SCHEMA_DESCRIPTION_LLM = (
    "Anomaly-detection scalar functions operating on a whole numeric series (`DOUBLE[]`): "
    "matrix profile, top discord (anomaly) index, top motif index, change-point detection, "
    "and z-score outlier indices."
)

_SCHEMA_DESCRIPTION_MD = (
    "# Anomaly Detectors\n\n"
    "Scalar functions that analyse a whole numeric time series for anomalies, repeated "
    "patterns, and regime changes. Each one takes the series as a single array argument "
    "you assemble in SQL, so a detector is just another expression you can compose, "
    "filter, and join against.\n\n"
    "## Concepts\n\n"
    "- **Matrix profile** — shape-based analysis that surfaces the most anomalous "
    "subsequence (discord) and the most repeated one (motif) for a chosen window.\n"
    "- **Change points** — penalized segmentation that locates structural breaks / "
    "regime shifts in the level or distribution of the series.\n"
    "- **Point outliers** — a light, dependency-free z-score pass for individual "
    "spikes and dropouts.\n\n"
    "## Using it\n\n"
    "Build the input with `array_agg(value ORDER BY t)` (or a bracketed array literal) "
    "so the whole series is passed in time order, then apply the detector that matches "
    "the question — one anomalous window, one recurring pattern, the regime boundaries, "
    "or the individual outlier positions."
)

# VGI152/VGI920: an analyst task suite so `vgi-lint simulate` can measure how
# well an agent, seeing only the catalog overview, actually uses this worker.
# Each prompt embeds the exact series and parameter, and every `reference_sql`
# is catalog-qualified, self-contained, and deterministic (fixed integer index
# outputs), so the reference is sound and re-runnable. `ignore_column_names`
# grades on values only (the natural answer is a single array/scalar).
_AGENT_TEST_TASKS = json.dumps(
    [
        {
            "name": "point_outlier_indices",
            "prompt": (
                "For the numeric series [10, 10, 11, 9, 10, 40, 10, 9, 11] (already in time "
                "order), use the z-score point-outlier detector with a threshold of 2 to find "
                "the samples that lie more than 2 population standard deviations from the mean. "
                "Return the detector's result directly as a single array value in one row (the "
                "list of zero-based outlier indices) — do not UNNEST it into one row per index."
            ),
            "reference_sql": (
                "SELECT anomaly.main.zscore_anomalies("
                "[10.0,10.0,11.0,9.0,10.0,40.0,10.0,9.0,11.0]::DOUBLE[], 2.0) AS outlier_indices"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "regime_change_index",
            "prompt": (
                "The series [1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9] steps up once. "
                "Automatically detect its change point(s) and return the index/indices where a "
                "new regime begins."
            ),
            "reference_sql": (
                "SELECT anomaly.main.change_points("
                "[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0]::DOUBLE[]) "
                "AS change_points"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "top_discord_start",
            "prompt": (
                "In the 25-point series "
                "[1,2,3,4,3,2,1,2,3,4,3,2,1,2,3,4,3,2,50,2,3,4,3,2,1], using a subsequence "
                "window of 4, return the zero-based start index of the single most anomalous "
                "window (the top discord)."
            ),
            "reference_sql": (
                "SELECT anomaly.main.discord_index("
                "[1.0,2.0,3.0,4.0,3.0,2.0,1.0,2.0,3.0,4.0,3.0,2.0,1.0,2.0,3.0,4.0,"
                "3.0,2.0,50.0,2.0,3.0,4.0,3.0,2.0,1.0]::DOUBLE[], 4) AS discord_start"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "top_motif_start",
            "prompt": (
                "The 20-point series "
                "[0,2,4,2,0,0.1,0.116,0.133,0.15,0.166,0.183,0.2,0,2,4,2,0,0.3,0.4,0.5] "
                "contains a repeated triangle shape. Using a subsequence window of 5, "
                "return the zero-based start index of the single most repeated pattern (the "
                "top motif)."
            ),
            "reference_sql": (
                "SELECT anomaly.main.motif_index("
                "[0.0,2.0,4.0,2.0,0.0,0.1,0.116,0.133,0.15,0.166,0.183,0.2,"
                "0.0,2.0,4.0,2.0,0.0,0.3,0.4,0.5]::DOUBLE[], 5) AS motif_start"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "matrix_profile_length",
            "prompt": (
                "For the numeric series [1, 2, 3, 4, 3, 2, 1, 2, 3, 4] (already in time "
                "order) and a subsequence window of 4, compute the full matrix profile (the "
                "z-normalized distance from every length-4 subsequence to its nearest "
                "neighbour) and return how many values it contains, as a single number."
            ),
            "reference_sql": (
                "SELECT len(anomaly.main.matrix_profile("
                "[1.0,2.0,3.0,4.0,3.0,2.0,1.0,2.0,3.0,4.0]::DOUBLE[], 4)) AS profile_length"
            ),
            "ignore_column_names": True,
        },
    ]
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
        # VGI152/VGI920: analyst task suite for `vgi-lint simulate`.
        "vgi.agent_test_tasks": _AGENT_TEST_TASKS,
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
                # VGI413: ordered category registry for this schema. Every
                # function declares a matching `vgi.category`; categories drive
                # the worker's navigation, listing sections, and SEO copy.
                "vgi.categories": json.dumps(
                    [
                        {
                            "name": "Matrix Profile",
                            "description": (
                                "Shape-based analysis over a sliding window: locate the most "
                                "anomalous subsequence (discord) and the most repeated one "
                                "(motif), or compute the full nearest-neighbour distance profile."
                            ),
                        },
                        {
                            "name": "Change Points",
                            "description": (
                                "Regime-shift and structural-break detection via ruptures "
                                "penalized segmentation, with automatic or fixed breakpoint counts."
                            ),
                        },
                        {
                            "name": "Outliers",
                            "description": (
                                "Lightweight, dependency-free per-point outlier flagging by "
                                "z-score distance from the series mean."
                            ),
                        },
                    ]
                ),
                # VGI123 classifying tags use BARE keys (not vgi.-namespaced).
                "domain": "time-series",
                "category": "anomaly-detection",
                "topic": "matrix-profile-and-change-points",
                # VGI506/VGI515 representative example queries: a JSON list of
                # {"description","sql"} objects (catalog-qualified, runnable) so
                # every example carries a human-readable description.
                "vgi.example_queries": json.dumps(
                    [
                        {
                            "description": (
                                "Full matrix profile of a 10-point series with window 4 "
                                "(7 nearest-neighbour distances)."
                            ),
                            "sql": (
                                "SELECT anomaly.main.matrix_profile("
                                "[1.0,2.0,3.0,4.0,3.0,2.0,1.0,2.0,3.0,4.0]::DOUBLE[], 4)"
                            ),
                        },
                        {
                            "description": (
                                "Start index of the top discord (the most anomalous "
                                "window-4 subsequence) — the spike at 18 gives 16."
                            ),
                            "sql": (
                                "SELECT anomaly.main.discord_index("
                                "[1.0,2.0,3.0,4.0,3.0,2.0,1.0,2.0,3.0,4.0,3.0,2.0,1.0,2.0,3.0,4.0,"
                                "3.0,2.0,50.0,2.0,3.0,4.0,3.0,2.0,1.0]::DOUBLE[], 4)"
                            ),
                        },
                        {
                            "description": (
                                "Start index of the top motif (the most repeated window-5 "
                                "subsequence) — the two triangles give 0."
                            ),
                            "sql": (
                                "SELECT anomaly.main.motif_index("
                                "[0.0,2.0,4.0,2.0,0.0,0.1,0.116,0.133,0.15,0.166,0.183,0.2,"
                                "0.0,2.0,4.0,2.0,0.0,0.3,0.4,0.5]::DOUBLE[], 5)"
                            ),
                        },
                        {
                            "description": (
                                "Automatically detect regime changes on a single step — the "
                                "level shift at index 8 gives [8]."
                            ),
                            "sql": (
                                "SELECT anomaly.main.change_points("
                                "[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0]"
                                "::DOUBLE[])"
                            ),
                        },
                        {
                            "description": (
                                "Flag z-score point outliers beyond 2 sigma — the 40.0 spike "
                                "is index 5, so the result is [5]."
                            ),
                            "sql": (
                                "SELECT anomaly.main.zscore_anomalies("
                                "[10.0,10.0,11.0,9.0,10.0,40.0,10.0,9.0,11.0]::DOUBLE[], 2.0)"
                            ),
                        },
                    ]
                ),
            },
            functions=list(SCALAR_FUNCTIONS),
        ),
    ],
)


class AnomalyWorker(Worker):
    """Worker process hosting the ``anomaly`` catalog."""

    catalog = _ANOMALY_CATALOG

    def __init__(self, *, quiet: bool = False, log_level: int = logging.INFO) -> None:
        """Construct the worker and JIT-compile the numba kernels in the background.

        ``stumpy.stump`` is numba-JIT compiled, so the first stumpy query in a
        fresh process pays a multi-second compile inline -- a window in which a
        worker-pool teardown SIGTERM (or a loaded host) can kill the run
        mid-assertion and record a spurious E2E failure.

        Warming here in ``__init__`` -- rather than in ``run()`` -- covers *every*
        transport. The stdio path calls ``run()``, but the ``--unix`` / ``--tcp``
        launcher paths that the vgi DuckDB extension uses to spawn a command
        ``LOCATION`` build the RPC server and serve directly, never calling
        ``run()``. Every transport instantiates the worker, so ``__init__`` is the
        one hook that always runs.

        The warm-up runs in a **daemon thread** so process spawn returns
        immediately: a pooled worker becomes ready in milliseconds and its first
        (non-stumpy) query is not blocked behind the ~10 s compile, while the
        stumpy kernels finish compiling in the background before any real
        matrix-profile query needs them. :func:`detectors._profile` holds
        ``_STUMP_LOCK``, so a real stumpy query that races the warm-up simply
        serializes behind it (numba's workqueue layer is not re-entrant). It only
        warms JIT caches -- never changes output -- and is best-effort (failures
        are swallowed inside :func:`detectors.warm_up`).

        Args:
            quiet: Suppress startup logging (or set ``VGI_QUIET=1``).
            log_level: Numeric level for the ``vgi`` logger hierarchy.
        """
        super().__init__(quiet=quiet, log_level=log_level)
        threading.Thread(target=detectors.warm_up, name="numba-warmup", daemon=True).start()


def main() -> None:
    """Run the anomaly worker process (stdio or, via flags, HTTP)."""
    AnomalyWorker.main()


if __name__ == "__main__":
    main()
