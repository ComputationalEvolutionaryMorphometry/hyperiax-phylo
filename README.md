# Hyperiax and Phylogenetic Inference from Shape Data

This repository contains reproduction code for phylogenetic inference from
landmark shape data using `hyperiax`, described in __Hyperiax and Phylogenetic
Inference from Shape Data__.

## Setup

The project uses `uv` and Python 3.11 or newer.

```bash
uv sync
```

The default environment installs the CPU JAX backend, which is sufficient for
small checks and short runs. For NVIDIA GPU with CUDA support (recommanded), install the GPU
dependency group:

```bash
uv sync --group gpu
```

## Data

This repository does NOT include the raw butterfly and beak datasets. Data access can be accquired by contacting the paper authors.

## Rebuild HDF5 Data

```bash
uv run python -m scripts.build_data \
  --csv-path <raw landmark csv file> \
  --tree-path <raw tree newick file>
```

## Inspect Data

Inspection writes `tree.png` and `leaves.png` next to the input `data.h5` by
default.

```bash
uv run python -m scripts.inspect_data --data-path <processed hdf5 data file>
```

## Run MCMC

Default run:

```bash
uv run python -m scripts.run_mcmc
```

Run a specific config:

```bash
uv run python -m scripts.run_mcmc --config config/butterflies.yaml
uv run python -m scripts.run_mcmc --config config/beaks.yaml
```

Run outputs are written under `runs/<run_name>/`, including:

```text
artifacts.h5
summary.json
config.json
trace.png
hist.png
```

Config values can be overridden from the command line, for example:

```bash
uv run python -m scripts.run_mcmc \
  --config config/butterflies.yaml \
  --num-samples 100 \
  --num-chains 1 \
  --progress-bar false
```

## Evaluate Runs

Evaluation regenerates run-level plots and posterior summaries from
`artifacts.h5`.

```bash
uv run python -m scripts.evaluate \
  --artifacts-path runs/butterflies/artifacts.h5 \
  --config config/butterflies.yaml
```

Default evaluation outputs are written into the run directory:

```text
leaves.png
root.png
trace.png
hist.png
```

Evaluation also writes `diagnostics.json` and `root_diagnostics.png` / `.pdf`,
with a compact Rich summary in the console. To compute these outputs alone,
without loading the observed tree, configuring a GPU, or simulating leaf shapes:

```bash
uv run python -m scripts.evaluate \
  --artifacts-path runs/butterflies/artifacts.h5 \
  --config config/butterflies.yaml --diagnostics-only
```

The `diagnostics` YAML section controls `rhat_threshold` (default 1.01),
`ess_threshold` (400, over all chains), `mcse_ratio_threshold` (0.05), and
`num_pcs` (1–3, default 3). Burn-in is read from `plot.num_burnin`; diagnostics
always use every remaining draw, regardless of `plot.thin`. The latter only
controls histogram/root-curve display. Parameter means and root means/intervals
also use all retained draws. Root curves are selected evenly across all chains.

The new evaluation uses ArviZ rank-normalized split/folded R-hat, bulk ESS,
tail ESS (5%/95% quantiles), and the mean's MCSE divided by posterior SD for
both scalar parameters and every saved root coordinate. JSON includes individual
coordinates, chain IDs, retained counts, thresholds, minima/medians/95th
percentiles/maxima, and flagged/unavailable counts. Coordinates use zero-based
landmark indices in the artifact (not necessarily the original input indices if
landmarks were removed). No additional shape alignment is performed.

Root trace/rank plots show the worst R-hat/bulk-ESS/tail-ESS coordinates (deduplicated),
up to three uncomputable coordinates, and up to three nondegenerate PC scores.
PCA uses one pooled, centered basis shared by all chains; its basis, center,
explained variance, and score diagnostics are saved. Missing/invalid root groups
produce an explicit unavailable plot, replacing any stale diagnostic figure.
Missing groups, unequal chain lengths, single-chain runs, constant/non-finite
coordinates, or fewer than four retained draws are never treated as passing.
Computed-only aggregates must be read alongside unavailable counts. These are
screening checks, not proof of convergence of the joint posterior or all ancestral
nodes. CLI exit code 0 means the report was generated, not that diagnostics passed.

The standalone legacy `load_gelman_rubin_diagnostics` helper remains available
for compatibility. New evaluation output (including the legacy-named
`EvaluationResult.gelman_rubin` field) uses the modern rank-based method; do not
compare its values as if they were the old unsplit statistic.

## Tests

```bash
uv run python -m pytest -q
```

## Performance experiments

The profiling workflow measures the current implementation without changing its
model or sampler. Configuration is read from YAML and never rewritten by the
profiling scripts. Relative paths in these configurations refer to the repository
root. The two original datasets must be available in `data/` as described above.

Start with the reduced integration experiment. The current smoke YAML uses GPU
as its primary backend (physical index 0), float64 and 25 steps per edge:

```bash
uv sync --group gpu --group dev
uv run python -m scripts.run_profiling --config config/profiling_smoke.yaml
```

This runs 25 isolated jobs: 2 full, 14 scaling and 9 reference/device jobs.
Both datasets use tips and landmarks `[4, 8, 16, 32]`; the butterfly reference
comparison uses L = 8, 16 and 32. Full smoke analyses use two processes/chains
with 100 samples per chain. Results are labelled **smoke validation**, not
performance claims; the label does not imply CPU execution.

On a CPU-only machine, use a separate YAML with `primary_backend: cpu` and
install with `uv sync --group dev`. With the other current smoke settings
unchanged, this schedules 22 jobs. Reduce sizes or edge steps explicitly if
needed. Check the selected YAML using `--dry-run` before execution.
To inspect the formal experiment matrix without running it:

```bash
uv run python -m scripts.run_profiling --config config/profiling.yaml --dry-run
```

For formal measurements, install the GPU dependencies, select the physical GPU
with `gpu_index`, and optionally restrict logical CPUs with `cpu_affinity` in the
YAML. Set `experiments` to any subset of `[full, scaling, reference]` to run only
that portion of the suite.

`cpu_threads` sets BLAS/OpenMP thread limits; it does not independently constrain
all JAX thread pools. `cpu_affinity` fixes the CPUs available to all workers.
When NVML is available, the selected physical GPU index is resolved to a UUID so
that execution and memory monitoring refer to the same device.

```bash
uv sync --group gpu --group dev
uv run python -m scripts.run_profiling --config config/profiling.yaml
```

The formal configuration expands to 83 jobs: two complete four-chain analyses,
18 unique scaling settings repeated three times, and nine reference/device
settings repeated three times. The maximum setting belongs to both scaling curves
but is measured only once per repeat. Short benchmarks use one chain/process;
full analyses explicitly use four workers on the selected device. These workers
share one GPU; this is not a multi-GPU benchmark. The current implementation's
observation handling, normalization, correction weights and proposals are also
retained in the reference traversal.

The console shows live Rich progress for each job: runtime/device initialization,
data loading, first target call/JIT, equivalence checks, target warmup/measurement,
and MCMC warmup/sampling. Complete analyses show each chain separately (including
queued chains), followed by artifact saving. Compilation has an elapsed-time
indicator, not a percentage or ETA. Iteration counts are reported at most once
per second, with immediate phase/final updates; the parent refreshes at 4 Hz.
Redirected output uses plain stage-change and 10-second heartbeat lines instead
of terminal control codes. Timeout/failure does not mark partial work as complete.

Each invocation creates a directory named `YYYYMMDD_HHMMSS` (UTC) under
`runs/profiling/`. If that directory already exists, the run stops without
overwriting it. Existing run directories are not renamed. Each directory contains:

- `config.json`, `environment.json`, `subsets.json`, `jobs.json`: resolved inputs,
  versions, hardware, Git commit and working-tree status, selected tips/landmarks,
  topology depth/width and scheduled jobs. No file checksums are calculated.
- `jobs/job_*/result.json`: per-job results and raw per-evaluation timings;
  `worker.log` retains diagnostic output. Complete analyses additionally retain
  the usual samples and chain timing records in `analysis/`.
- `jobs/job_*/progress/*.json`: latest worker/per-chain progress snapshots, written
  atomically and retained for inspection after failure or interruption.
- `jobs/job_*/diagnostics.json`: post-processed ESS/R-hat for successful `full`
  jobs; also included in aggregate `results.json` and `summary.json`.
- `results.json`, `summary.json`: all records, repeat medians/ranges and paired
  speedups. Times are in seconds and memory in bytes; unavailable metrics are
  JSON `null` with an explanation.
- `scaling.png` / `.pdf` and `speedups.png` / `.pdf`: static figures. Rich tables
  display the same summary in the console, with human-readable units.
  When rebuilding with plots enabled, an existing figure without valid current
  data is replaced by an explicit unavailable notice in both formats. A fresh
  run omits inapplicable figures. `--no-plots` leaves all existing figures untouched.

Rebuild the report from existing measurements without running inference:

```bash
uv run python -m scripts.report_profiling --run-dir runs/profiling/<run-directory>
```

Reports automatically compute full-analysis diagnostics from the stored HDF5
traces, including for older runs. They use ArviZ 0.22 bulk ESS, tail ESS (5%/95%),
and rank-normalized split R-hat for `k_alpha`, `k_sigma`, `obs_var`, and the log
target. Ancestral/root coordinates are not included. See the
[ESS](https://python.arviz.org/en/v0.22.0/api/generated/arviz.ess.html) and
[R-hat](https://python.arviz.org/en/v0.22.0/api/generated/arviz.rhat.html) definitions.
The `Full analyses` timing table additionally shows `Bulk ESS` and `Tail ESS`
after `Acceptance`. Each is the minimum over the three scalar parameters for
that chain alone, using the same diagnostic burn-in. These per-chain estimates
are not joint multi-chain ESS and do not diagnose between-chain disagreement.
`chains.chain_000` (etc.) in per-job `diagnostics.json` stores `ess_bulk`,
`ess_tail`, and the per-parameter values under `variables`. The same data appears
under `diagnostics.chains` in both aggregate JSON outputs.
If any scalar parameter is invalid or missing, the
chain minimum is unavailable (`null` in JSON, `—` in the table).
Both supplied YAMLs set `full_diagnostics.burnin_fraction: 0.1`: discard
`floor(draws_per_chain * fraction)` from each chain for diagnostics only, with no
thinning. This retains 4,500 draws per formal chain or 90 per smoke chain.
Only configurations/older runs without this setting fall back to 0.5. Report
rebuilding uses the saved run configuration, not the currently edited YAML.
This is an explicit analysis choice, not automatic burn-in detection
and not the compilation warmup. Override it without resampling:

```bash
uv run python -m scripts.report_profiling --run-dir runs/profiling/<run-directory> --no-plots --burnin-fraction 0.25
```

Diagnostics never change HDF5 samples, saved input configurations, original
per-job `result.json`, or performance measurements. They are computed in report
post-processing, outside all benchmark timing/memory windows. Bulk ESS/s uses
the **whole job wall time**, including all chains, discarded burn-in, compilation
and artifact writes; overlapping per-chain times are not summed. Short traces,
R-hat above 1.01, or bulk/tail ESS below 100 per chain produce warnings (not a
claim of convergence). Non-finite, constant, missing, or insufficient traces
produce explicit unavailable/null results. Smoke-run estimates remain smoke
diagnostics, not publication-quality inference or ESS/s comparisons.

### Measurement definitions

`target_first_call_seconds` includes tracing/compilation **and** first execution;
it is not a pure compiler measurement. `target_seconds` averages synchronized
evaluations returning both the target and ancestral root state after warmup.
Inputs are already on the device. Persistent compilation caching is disabled in
fresh workers, and GPU preallocation is disabled consistently across jobs.
Target progress reporting is outside each timed evaluation. MCMC loop timings and
end-to-end times include low-frequency progress reporting overhead; no estimated
overhead is subtracted, and progress is not claimed to be cost-free.

`mcmc_seconds` measures the existing Python sampling loop after its configured
warmup, including proposals, accept/reject synchronization and root sample
collection. Initialization and final result serialization are outside that
window. The optional `driver.profile_warmup_iterations` setting enables these
records; its default is `null`, which leaves timing instrumentation disabled.
Short chains are computational measurements, not convergence diagnostics.

`process_wall_seconds` is measured by the parent from process launch through
termination, including imports, device initialization, data loading, compilation,
sampling and artifact writes. It excludes subset construction and final figures.
For full analyses, `analysis_seconds` separately measures the existing analysis
call including its HDF5/JSON writes; `analysis_summary.elapsed_seconds` preserves
the original computation-only timing convention. Per-chain timings can overlap
and must not be summed to infer wall time. First-target timings for later chains
in the same worker can reuse compiled functions.

Memory is sampled every 50 ms over the complete worker lifetime, including child
processes, compilation and reference validation. RSS is the observed sum of
process RSS; shared pages can be counted more than once. GPU memory is NVML's
per-process allocator occupancy, not live-array memory, and missing driver/NVML
support is reported explicitly. Sampling can miss short peaks. Cold calls and
reference-validation executables can therefore affect the reported memory peak.

The Python reference uses JIT-compiled local operations with an explicit
postorder/preorder node loop and per-node state. It is checked against the current
Hyperiax target/root outputs using three identical noise inputs before recording
speedups (`rtol=1e-6`, `atol=1e-8`). Cross-device comparisons also check their saved
outputs. Ratios are paired by seed. This is a comparison with a local execution
reference, not with historical BFFG software.

Failures, timeouts and incomplete jobs remain visible in JSON and the console;
they are excluded from timing statistics and cause a nonzero exit status.
Increase iteration counts if the report warns that measurement windows are less
than ten seconds. Each run has a fresh output directory; existing analyses and
input files are never reused as writable result directories.
