"""Isolated benchmark worker. Backend selection is supplied by the parent."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import sys
import time
import traceback

from src.profiling.artifacts import read_json, write_json
from src.profiling.progress import ProgressReporter


def execute(payload: dict) -> dict:
    progress = ProgressReporter(Path(payload["job_dir"]) / "progress")
    progress.stage("Importing runtime")
    # These imports must stay inside the worker, after process environment setup.
    import jax
    import jax.numpy as jnp
    import numpy as np
    from src.bffg import MCMCModelConfig, MCMCParams, build_bffg_context
    from src.driver import MCMCDriverConfig, _build_mcmc_target_and_phylo_root_state, run_mcmc, run_mcmc_chain
    from src.loader import load_augmented_butterfly_tree
    from src.profiling.reference import build_reference_target
    from scripts.run_mcmc import _model_config_from_mapping, _driver_config_from_mapping

    job, config = payload["job"], payload["config"]
    backend = job["backend"]
    progress.stage("Initializing device")
    devices = jax.devices(backend)
    if not devices:
        raise RuntimeError(f"Requested backend {backend} has no device.")
    dataset_config = config["datasets"][job["dataset"]]
    # Reuse the analysis CLI's coercion (PyYAML reads values such as 1e-5 as strings).
    model = _model_config_from_mapping(dataset_config["model"])
    driver = _driver_config_from_mapping(dataset_config["driver"])
    jax.config.update("jax_enable_x64", model.enable_x64)
    progress.stage("Loading data")
    started = time.perf_counter()
    dataset = load_augmented_butterfly_tree(job["subset"]["h5"])
    jax.block_until_ready(dataset.tree)
    load_seconds = time.perf_counter() - started
    device_info = [{"platform": d.platform, "kind": d.device_kind, "id": d.id} for d in devices]
    result = {"status": "ok", "devices": device_info, "load_seconds": load_seconds,
              "dtype": "float64" if model.enable_x64 else "float32", "num_edge_steps": model.num_edge_steps,
              "model_config": asdict(model), "warnings": []}
    if job["experiment"] == "full":
        full = config["full"]
        driver = replace(driver, num_chains=full["num_chains"], num_samples=full["num_samples"],
                         num_processes=full["num_processes"], chain_backend="process", progress_bar=False,
                         random_seed=job["seed"], profile_warmup_iterations=0)
        if config["smoke"]:
            model = replace(model, **{f"{key}_init": value for key, value in dataset_config["params"].items()})
        result.update(model_config=asdict(model), driver_config=asdict(driver))
        run_dir = Path(payload["job_dir"]) / "analysis"
        if run_dir.exists():
            raise FileExistsError(f"Refusing to replace existing analysis artifacts: {run_dir}")
        start = time.perf_counter()
        run = run_mcmc(dataset, run_dir=str(run_dir), model_config=model, driver_config=driver,
                       status_callback=progress)
        result["analysis_seconds"] = time.perf_counter() - start
        result["analysis_summary"] = run.summary
        result["artifact_path"] = str(run_dir / "artifacts.h5")
        if not np.isfinite(run.log_posteriors).all() or any(not np.isfinite(v).all() for v in run.samples.values()):
            raise FloatingPointError("Full analysis produced non-finite samples or targets.")
        return result

    progress.stage("Building context")
    start = time.perf_counter()
    context = build_bffg_context(dataset, model)
    jax.block_until_ready(context.tree)
    result["context_seconds"] = time.perf_counter() - start
    params = MCMCParams(**dataset_config["params"])
    # Host-generated inputs are identical across CPU/GPU and both implementations.
    host_noise = np.random.default_rng(job["seed"]).normal(size=context.tree["zs"].shape)
    noise = jax.device_put(host_noise.astype(np.float64 if model.enable_x64 else np.float32))
    jax.block_until_ready((params, noise))
    start = time.perf_counter()
    target = (build_reference_target(context) if job["implementation"] == "reference"
              else _build_mcmc_target_and_phylo_root_state(context))
    result["target_factory_seconds"] = time.perf_counter() - start
    progress.stage("First target / JIT")
    start = time.perf_counter()
    actual = jax.block_until_ready(target(params, noise))
    result["target_first_call_seconds"] = time.perf_counter() - start
    for value in actual:
        if not np.isfinite(np.asarray(value)).all():
            raise FloatingPointError("Target/root output contains non-finite values.")

    if job["implementation"] == "reference":
        progress.stage("Checking equivalence", 0, 3)
        production = _build_mcmc_target_and_phylo_root_state(context)
        errors = []
        for offset in range(3):
            check_noise = jax.device_put(np.random.default_rng(job["seed"] + offset).normal(size=host_noise.shape))
            expected = jax.block_until_ready(production(params, check_noise))
            observed = jax.block_until_ready(target(params, check_noise))
            for a, b in zip(observed, expected):
                np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-8)
                errors.append(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
            progress.stage("Checking equivalence", offset + 1, 3)
        result["equivalence"] = {"rtol": 1e-6, "atol": 1e-8, "max_absolute_error": max(errors), "inputs": 3}
        del production, check_noise, expected, observed

    settings = config["benchmark"]
    progress.stage("Target warmup", 0, settings["target_warmup"])
    for iteration in range(settings["target_warmup"]):
        jax.block_until_ready(target(params, noise))
        progress.stage("Target warmup", iteration + 1, settings["target_warmup"])
    durations = []
    progress.stage("Target measurement", 0, settings["target_iterations"])
    for iteration in range(settings["target_iterations"]):
        start = time.perf_counter()
        jax.block_until_ready(target(params, noise))
        durations.append(time.perf_counter() - start)
        progress.stage("Target measurement", iteration + 1, settings["target_iterations"])
    result["target_seconds"] = float(np.mean(durations))
    result["target_durations_seconds"] = durations
    result["target_iterations"] = len(durations)
    result["check_output"] = {"log_target": float(actual[0]), "root_state": np.asarray(actual[1]).tolist()}
    if sum(durations) < settings["min_measure_seconds"]:
        result["warnings"].append("Target measurement window is shorter than min_measure_seconds; increase target_iterations.")

    # Reference comparisons concern target execution. MCMC timing uses the current
    # production driver only, and is measured for every scaling setting.
    if job["experiment"] == "scaling":
        driver = replace(driver, num_chains=1, chain_backend="sequential", progress_bar=False,
                         num_samples=settings["mcmc_warmup"] + settings["mcmc_iterations"],
                         profile_warmup_iterations=settings["mcmc_warmup"], random_seed=job["seed"])
        result["driver_config"] = asdict(driver)
        chain = run_mcmc_chain(context, driver_config=driver,
                               initial_state=(params, jnp.zeros_like(noise)), collect_phylo_root_samples=True,
                               target_and_phylo_root_state=target, status_callback=progress)
        if not np.isfinite(chain.log_posteriors).all() or not np.isfinite(chain.phylo_root_samples).all():
            raise FloatingPointError("Measured chain contains non-finite output.")
        result["mcmc_timings"] = chain.timings
        result["mcmc_seconds"] = chain.timings["measured_loop_seconds"] / chain.timings["measured_iterations"]
        result["acceptance_rate"] = chain.acceptance_rate
        if chain.timings["measured_loop_seconds"] < settings["min_measure_seconds"]:
            result["warnings"].append("MCMC measurement window is shorter than min_measure_seconds; increase mcmc_iterations.")
    return result


def main() -> int:
    payload = read_json(Path(sys.argv[1]))
    try:
        result = execute(payload)
    except Exception as error:
        result = {"status": "failed", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
    progress = ProgressReporter(Path(payload["job_dir"]) / "progress")
    progress.stage("Saving result")
    write_json(Path(payload["job_dir"]) / "worker_result.json", result)
    progress.stage("Complete" if result["status"] == "ok" else "Failed")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
