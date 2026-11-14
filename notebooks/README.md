# Anomaly detection in ordered SQL telemetry

[01 · A spike, a shift, or an unusual pattern?](01_spikes_shifts_and_patterns.ipynb)
compares whole-record and segmented point outliers with automatic change points
and matrix-profile motifs/discords. It ends with a one-parameter experiment:
increase the subsequence window from 24 to 48 samples.

The saved notebook contains successful execution outputs, visible Haybarn/VGI
connection and analytical SQL, folded Python presentation code, and three charts.
It follows the local [notebook standard](STANDARD.md).

## Data and meanings

All 240 readings are **synthetic**, constructed deterministically in visible SQL.
They use invented telemetry units (TU), one sample per minute from
2026-01-01 00:00 UTC inclusive to 04:00 UTC exclusive. The final timestamp is 03:59.
The formula combines a 12-minute repeating cycle, a level shift at minute 120 and
a spike at minute 180. These are construction markers, not measured incidents or
real labels. There is no live/replay mode and no external data download.

The real pinned worker computes every detector result. The notebook uses the
preferred ordered table-in/table-out interface for the whole series and the
ordered-array scalar interface for within-segment calculations. SQL exposes the
z-score magnitudes for explanation and verifies their threshold agreement with
worker flags. Python never imports the detector libraries or calculates detector
results. The notebook is a retrospective analysis, not a streaming alarm.

The profile window counts one-minute sample slots; a 24-sample window has 23
minutes between its first and last timestamps. Matrix-profile distances are
unitless and not directly comparable across different window lengths.

## Install and run from this directory

Copy or download `notebooks/` with its helper, assets and environment files. With
[uv](https://docs.astral.sh/uv/) installed, run these commands **inside notebooks/**:

```sh
uv sync --locked
uv run --locked python execute.py
uv run --locked python execute.py --parameters '{"WINDOW_SAMPLES":48}' \
  --output-dir artifacts/window-48
```

No worker checkout, root environment or local worker build is required. The notebook
environment uses Python 3.13 and published Haybarn `2.0.0rc11`. Its separate lockfile
also pins the plotting and execution dependencies.

The visible SQL uses this literal launcher:

```text
uvx --python 3.13 --with vgi-python==0.51.1 --from git+https://github.com/Query-farm/vgi-anomaly@b933ef6e41dce2ef151e7c6f109f3a8f724f0291 vgi-anomaly-worker
```

The Git revision is the published **v0.5.0** release; its release assets are Python
packages rather than platform binaries for `github-auto`. `uvx` installs and caches
the worker independently; the compatible SDK is explicitly pinned. Initial Git,
package and VGI extension downloads need internet access. Git and uv/uvx must be on
PATH; uv can install Python 3.13 if needed. Detector inputs and calculations stay local.
To upgrade, replace the literal revision and SDK pin only after rerunning both
parameter cases and verifying detector/API compatibility. Transitive numerical
worker dependencies follow the pinned release's package constraints.

The setup limits numerical threads for this small workload. Cold startup can take
tens of seconds compiling the matrix-profile kernel; the runner permits 300 seconds
per cell. This numerical compilation is automatic, with no SQL engine build.

For interactive work, select `.venv/bin/python` from this directory as the kernel;
no global kernel registration is required. Set the kernel working directory to this
notebook directory. Edit the visible parameters, then **Restart Kernel → Run All**.
Connection SQL and analytical SQL remain expanded; implementation cells are folded.

The runner creates a temporary kernel using its own interpreter and this directory
as cwd. It saves atomically after a complete successful execution. Overrides require
a separate output destination; paths that overwrite checked-in defaults are rejected.
Review copies include branding, this guide, the standard, helper and environment
files. Logs, charts and copies in `artifacts/` are ignored by Git; successful main
notebook outputs remain embedded. To rerun a copied experiment directly, select its
notebook directory as cwd and use the original environment's interpreter, or install
its copied environment files.

## Checks and limitations

Execution checks input uniqueness and finiteness; contiguous segment mapping;
worker/SQL point-flag agreement; profile length and positions; and agreement
between the worker's motif/discord indices and the full worker profile extrema.
Zero result rows mean no matches only after validating the nonempty input. NULL
detector results are treated as unavailable and stop the analysis.

The synthetic ingredients do not establish detector performance on real systems.
Automatic segmentation uses the same observations subsequently scored. A high
score is not a probability, causal explanation, or guarantee of a real incident.
The quadratic profile calculation is deliberately limited to 240 readings here.

The unchanged [official Query.Farm Strata Sun wordmark](https://query.farm/media-kit/)
and VGI logo are stored under `assets/` and linked in the notebook. They were copied
from the approved `vgi-perps` notebook assets.
