"""Local filesystem artifacts for MCMC runs."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

MCMC_PARAMETER_NAMES = ("k_alpha", "k_sigma", "obs_var")


def prepare_run_dir(run_dir: str | Path) -> Path:
    """Create an empty run directory, replacing stale local artifacts."""

    run_dir = Path(run_dir)
    if run_dir.exists():
        if run_dir.is_dir():
            shutil.rmtree(run_dir)
        else:
            run_dir.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_artifacts(
    run_dir: str | Path,
    *,
    model_config,
    driver_config,
    dataset,
    context,
    results: list,
    elapsed_seconds: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Persist full-chain MCMC arrays and run metadata."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_posteriors = np.stack([result.log_posteriors for result in results])
    accepted = np.stack([result.accepted for result in results])
    samples = _stack_parameter_samples(results)
    artifact_path = run_dir / "artifacts.h5"
    _write_artifact_hdf5(artifact_path, samples=samples, log_posteriors=log_posteriors, accepted=accepted)

    summary = {
        "num_chains": len(results),
        "num_samples": int(log_posteriors.shape[1]),
        "total_iterations": int(log_posteriors.shape[1]),
        "acceptance_rates": [float(result.acceptance_rate) for result in results],
        "final_log_posteriors": [float(result.log_posteriors[-1]) for result in results],
        "initial_params": [_params_to_floats(result.initial_params) for result in results],
        "final_params": [_final_sample_to_floats(result) for result in results],
        "elapsed_seconds": float(elapsed_seconds),
        "dataset": {
            "h5_path": str(dataset.h5_path),
            "node_count": context.node_count,
            "original_n_landmarks": dataset.original_landmark_count,
            "n_landmarks": context.n_landmarks,
            "d_landmarks": context.d_landmarks,
            "state_dim": context.state_dim,
            "n_leaves": int(context.leaf_observations.shape[0]),
            "removed_landmarks": list(dataset.removed_landmarks),
            "kept_landmarks": list(dataset.kept_landmarks),
        },
        "artifact_file": artifact_path.name,
    }
    config_payload = {
        "model": asdict(model_config),
        "driver": asdict(driver_config),
        "augment": {
            "remove_lmk": list(dataset.removed_landmarks) if dataset.removed_landmarks else None,
        },
        "dataset_h5_path": str(dataset.h5_path),
        "node_count": dataset.tree.size,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "config.json", config_payload)
    return summary, samples, log_posteriors, accepted


def _write_artifact_hdf5(
    path: Path,
    *,
    samples: dict[str, np.ndarray],
    log_posteriors: np.ndarray,
    accepted: np.ndarray,
) -> None:
    with h5py.File(path, "w") as h5:
        samples_group = h5.create_group("samples")
        for parameter_name, values_by_chain in samples.items():
            parameter_group = samples_group.create_group(parameter_name)
            _write_chain_datasets(parameter_group, values_by_chain)

        trace_group = h5.create_group("trace")
        _write_chain_datasets(trace_group.create_group("log_posteriors"), log_posteriors)
        _write_chain_datasets(trace_group.create_group("accepted"), accepted)


def _write_chain_datasets(group, values_by_chain: np.ndarray) -> None:
    for chain_index, values in enumerate(values_by_chain):
        group.create_dataset(f"chain_{chain_index:03d}", data=values)


def _stack_parameter_samples(results: list) -> dict[str, np.ndarray]:
    samples = {
        name: np.stack([result.samples[name] for result in results])
        for name in MCMC_PARAMETER_NAMES
    }
    phylo_root_samples = [result.phylo_root_samples for result in results]
    if any(values is not None for values in phylo_root_samples):
        if any(values is None for values in phylo_root_samples):
            raise ValueError("All chains must include phylo_root samples when writing that artifact.")
        samples["phylo_root"] = np.stack(phylo_root_samples)
    return samples


def _params_to_floats(params) -> dict[str, float]:
    return {name: float(getattr(params, name)) for name in MCMC_PARAMETER_NAMES}


def _final_sample_to_floats(result) -> dict[str, float]:
    return {name: float(result.samples[name][-1]) for name in MCMC_PARAMETER_NAMES}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
