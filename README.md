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

## Tests

```bash
uv run python -m pytest -q
```
