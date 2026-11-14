# A spike, a shift, or an unusual pattern?

[Open the executed notebook](01_spikes_shifts_and_patterns.ipynb).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Query-farm/vgi-anomaly/blob/main/notebooks/01_spikes_shifts_and_patterns.ipynb)

The walkthrough compares whole-record and segmented point outliers with automatic
change points and matrix-profile motifs/discords. Three SVG charts separate the
synthetic signal, the choice of reference distribution and the shape comparison.
It ends by changing the subsequence window from 24 to 48 samples.

In Colab choose **Runtime → Run all**. The bootstrap installs only the shared base
package and retains compatible plotting and Jupyter libraries. No worker checkout,
companion files, credential or data download is needed. The separate worker runs
Python 3.13. Initial package downloads require network access; detector inputs and
calculations stay local. If an earlier setup damaged the kernel dependencies, save
work, choose **Runtime → Disconnect and delete runtime**, reconnect and run all.

## Run locally

From this directory, with [uv](https://docs.astral.sh/uv/) installed:

```sh
uv sync --locked
uv run --locked vgi-notebooks --timeout 300
uv run --locked vgi-notebooks --timeout 300 --parameters '{"WINDOW_SAMPLES":48}' --output-dir artifacts/window-48
```

The visible SQL uses a clean, literal released-worker command:

```text
uvx --python 3.13 --from git+https://github.com/Query-farm/vgi-anomaly@v0.5.0 vgi-anomaly-worker
```

The release owns the numerical-library and SDK dependencies; no override stack or
local binary is used. Setup limits numerical threads for this small workload.
Cold startup compiles the matrix-profile numerical kernel and can take tens of
seconds. The runner permits 300 seconds per cell. This is automatic numerical
compilation, not a SQL engine build. When upgrading the release, rerun the default
and experiment and compare detector semantics as well as successful startup.

The local lock adds the `[runner]` extra of
[vgi-example-notebooks v0.3.8](https://github.com/Query-farm/vgi-example-notebooks),
while Colab uses only its base. That package owns the
[standard](https://github.com/Query-farm/vgi-example-notebooks/blob/v0.3.8/docs/standard.md),
runner, branding, secret isolation and SVG presentation. The successful default
contains native SVG plus content-addressed `charts/` image links for GitHub.
Experiments retain native SVG. Review PNGs, logs and copies are ignored artifacts;
there is no copied standard or custom execution/presentation helper.

To register an optional local kernel:

```sh
uv run --locked python -m ipykernel install --user --name anomaly-examples --display-name 'Python (Anomaly examples)'
```

Edit visible parameters and restart/run all. `WINDOW_SAMPLES` accepts integers
6–60; `ZSCORE_THRESHOLD` accepts finite numbers 1.5–5.0. The runner preserves the
last successful default and requires a separate destination for overrides.

## Data and meanings

All 240 readings are synthetic and constructed in visible SQL. They use invented
telemetry units (TU), one sample per minute from 2026-01-01 00:00 UTC inclusive to
04:00 exclusive. The final timestamp is 03:59. The formula combines a 12-minute
repeating cycle, a 12-TU level shift at minute 120 and an 8-TU spike at minute 180.
These are construction markers, not measured incidents or independently labeled
faults. There is no live/replay mode.

Every detector result comes from the real worker. Ordered table inputs analyze
the whole series, while ordered-array scalars score within-segment groups. Each
worker result is materialized before a separate DataFrame capture, so displaying
it cannot repeat a detector call. SQL exposes z-score magnitudes and checks their
threshold agreement with worker flags. Python presents and verifies returned data;
it never imports detector libraries or computes detector results.

The default's injected spike is 2.21 population standard deviations globally and
4.13 within its worker-defined segment. At cutoff 3 the global detector returns
no flags, while the within-segment detector flags minute 180. The automatic boundary
at minute 120 changes the reference distribution. It does not establish a cause,
independent confirmation or a calibrated false-alarm rate. These methods use the
whole record retrospectively; future samples can affect earlier results.

The matrix profile uses dimensionless z-normalized Euclidean distance to the
nearest nontrivial shape match. Normalization removes window level and scale;
nearby overlapping matches are excluded. The worker returns the first extremum
when periodic patterns tie. A 24-sample window contains 24 one-minute slots but
only 23 minutes between its first and last timestamps. Its 217 profile starts
include the returned discord [109,132], crossing the shift, and motif [0,23].
These are window positions, not estimated incident timestamps.

## One focused experiment

Change only `WINDOW_SAMPLES` from **24 to 48**, keeping the cutoff and signal fixed.
Compare the selected starts/ends, marker overlap and number of valid profile starts.
The point flags and automatic boundaries should remain unchanged because they do
not take this parameter. Distances at different window lengths are not directly
comparable severity scores: they aggregate different numbers of samples. A changed
discord shows sensitivity to duration, not discovery of a true cause.

## Coverage and checks

[coverage.json](coverage.json) maps **all five public logical SQL surfaces** to
visible invocations, useful returned results, interpretations and successful
hash-bound receipts: `zscore_anomalies`, `change_points`, `matrix_profile`,
`discord_index` and `motif_index`. Scalar/table forms and overloads count once per
schema/name. Full native function/table/view exports are retained in `coverage/`.
The actual released v0.5.0 and public source baseline
`1bf4d7eae1a7c3f3673aa4f83f64f79dd3303959` expose the same five names and signatures.
Current-interface compatibility is reviewed separately from release execution.

Twelve relationship assertions check valid unique input, contiguous segment mapping,
worker/SQL flag agreement, profile length/positions/finiteness and agreement between
the separately computed motif/discord and full-profile extrema. Valid nonempty
input with zero output rows means no matches; a NULL detector result means
unavailable analysis and stops execution. The grouped z-score call returns a useful
nonempty flag, so its coverage does not rely on the empty global result.

Source checks pass 58 pytest tests, 49 SQL assertions across four cases, Ruff,
formatting, mypy, isolated pydoclint and executed metadata validation at 100/100,
assurance L2 with zero findings. Worker code and root dependencies are unchanged.
The synthetic ingredients do not establish performance on real telemetry, which
requires its own missingness, sampling and independent validation strategy. The
quadratic profile calculation is deliberately limited to 240 readings here.

The default and a detached notebook-only 48-sample run both pass all twelve
relationship assertions. The detached run starts with an empty worker uv cache.
The discord changes from [109,132] to [111,158] inclusive; both cross the injected
shift. Valid profile starts decrease from 217 to 193. Point flags, boundary and
segment displays, plus the first two chart PNGs, are byte-identical across runs.
All six default/experiment chart previews were inspected. Three runner guards and
two invalid-parameter checks preserve the default and stop invalid inputs before
worker attachment.

A fresh Python 3.12 kernel preloads pandas 2.2.3, Matplotlib 3.10.0, IPython 8.30.0,
ipykernel 6.17.1 and jupyter-client 7.4.9. The base-only bootstrap and complete
notebook execute while retaining those versions; `pip check` passes and nbclient
is absent from that kernel. This is a preloaded-kernel compatibility test, not a
manually exercised hosted Colab UI session.
