"""Hand-written MCMC driver."""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
import math
import multiprocessing
import os
from pathlib import Path
import queue
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from src.tree_state import BFFG_FIELDS
from src.bffg import (
    BFFGContext,
    MCMCModelConfig,
    MCMCParams,
    build_bffg_context,
    mcmc_log_posterior,
    mcmc_log_posterior_and_node_state,
)
from src.loader import AugmentedButterflyTree, load_augmented_butterfly_tree
from src.run_artifacts import prepare_run_dir, save_run_artifacts

MCMC_PARAMETER_NAMES = ("k_alpha", "k_sigma", "obs_var")


@dataclass(frozen=True)
class MCMCDriverConfig:
    """Configuration for MH execution."""

    num_samples: int = 1000
    num_chains: int = 4
    pcn_eta: float = 0.9
    k_alpha_proposal_var: float = 0.01
    k_sigma_proposal_var: float = 0.01
    obs_var_proposal_var: float = 0.01
    random_seed: int = 0
    progress_bar: bool = True
    chain_backend: str = "sequential"
    num_processes: int = 0


@dataclass(frozen=True)
class ChainResult:
    """In-memory samples from one chain."""

    log_posteriors: np.ndarray
    samples: dict[str, np.ndarray]
    phylo_root_samples: np.ndarray | None
    accepted: np.ndarray
    acceptance_rate: float
    initial_params: MCMCParams
    final_params: MCMCParams
    final_noise: np.ndarray


@dataclass(frozen=True)
class MCMCRunResult:
    """In-memory view of a completed MCMC run."""

    run_dir: Path
    samples: dict[str, np.ndarray]
    log_posteriors: np.ndarray
    accepted: np.ndarray
    summary: dict[str, Any]
    context: BFFGContext


def sample_initial_params(
    key: jax.Array,
    *,
    driver_config: MCMCDriverConfig,
    model_config: MCMCModelConfig,
) -> MCMCParams:
    """Sample chain initial parameters from the model priors using a seed-derived key."""

    keys = jax.random.split(key, 3)
    k_alpha = _sample_inverse_gamma(
        keys[0],
        model_config.k_alpha_prior_alpha,
        model_config.k_alpha_prior_beta,
    )
    k_sigma = _sample_inverse_gamma(
        keys[1],
        model_config.k_sigma_prior_alpha,
        model_config.k_sigma_prior_beta,
    )
    obs_var = _sample_inverse_gamma(
        keys[2],
        model_config.obs_var_prior_alpha,
        model_config.obs_var_prior_beta,
    )
    return _clamp_params_to_model_bounds(
        MCMCParams(k_alpha=k_alpha, k_sigma=k_sigma, obs_var=obs_var),
        model_config,
    )


def initial_params_for_chain(
    key: jax.Array,
    *,
    chain_index: int,
    driver_config: MCMCDriverConfig,
    model_config: MCMCModelConfig,
) -> MCMCParams:
    """Build chain initial parameters from configured values or prior draws."""

    for name in ("k_alpha_init", "k_sigma_init", "obs_var_init"):
        _validate_initial_param_config(name, getattr(model_config, name), driver_config.num_chains)
    sampled = sample_initial_params(
        key,
        driver_config=driver_config,
        model_config=model_config,
    )
    return _clamp_params_to_model_bounds(
        MCMCParams(
            k_alpha=_initial_param_value(model_config.k_alpha_init, sampled.k_alpha, chain_index),
            k_sigma=_initial_param_value(model_config.k_sigma_init, sampled.k_sigma, chain_index),
            obs_var=_initial_param_value(model_config.obs_var_init, sampled.obs_var, chain_index),
        ),
        model_config,
    )


def propose_params(
    params: MCMCParams,
    key: jax.Array,
    *,
    driver_config: MCMCDriverConfig,
    model_config: MCMCModelConfig,
) -> MCMCParams:
    keys = jax.random.split(key, 3)
    k_alpha = _lognormal_rw(params.k_alpha, keys[0], driver_config.k_alpha_proposal_var)
    k_sigma = _lognormal_rw(params.k_sigma, keys[1], driver_config.k_sigma_proposal_var)
    obs_var = _lognormal_rw(params.obs_var, keys[2], driver_config.obs_var_proposal_var)
    return _clamp_params_to_model_bounds(
        MCMCParams(k_alpha=k_alpha, k_sigma=k_sigma, obs_var=obs_var),
        model_config,
    )


def update_cn(noise: jax.Array, key: jax.Array, config: MCMCDriverConfig) -> jax.Array:
    eta = jnp.asarray(config.pcn_eta, dtype=noise.dtype)
    return noise * eta + jnp.sqrt(1.0 - eta**2) * jax.random.normal(key, noise.shape, dtype=noise.dtype)


def propose_state(
    params: MCMCParams,
    noise: jax.Array,
    key: jax.Array,
    *,
    driver_config: MCMCDriverConfig,
    model_config: MCMCModelConfig,
) -> tuple[MCMCParams, jax.Array]:
    parameter_key, noise_key = jax.random.split(key, 2)
    return (
        propose_params(params, parameter_key, driver_config=driver_config, model_config=model_config),
        update_cn(noise, noise_key, driver_config),
    )


def run_mcmc_chain(
    context: BFFGContext,
    *,
    driver_config: MCMCDriverConfig | None = None,
    rng_key: jax.Array | None = None,
    initial_state: tuple[MCMCParams, jax.Array] | None = None,
    chain_index: int | None = None,
    num_chains: int | None = None,
    target: Callable[[MCMCParams, jax.Array], jax.Array] | None = None,
    target_and_phylo_root_state: Callable[[MCMCParams, jax.Array], tuple[jax.Array, jax.Array]] | None = None,
    collect_phylo_root_samples: bool = False,
    progress_callback: Callable[[int | None, int, float, float], None] | None = None,
) -> ChainResult:
    """Run one MH chain."""

    driver_config = driver_config or MCMCDriverConfig()
    _validate_driver_config(driver_config)
    rng_key = jax.random.PRNGKey(driver_config.random_seed) if rng_key is None else rng_key
    if initial_state is None:
        rng_key, init_key = jax.random.split(rng_key)
        current_params = initial_params_for_chain(
            init_key,
            chain_index=0 if chain_index is None else chain_index,
            driver_config=driver_config,
            model_config=context.config,
        )
        current_noise = jnp.zeros_like(context.tree[BFFG_FIELDS.zs])
    else:
        current_params, current_noise = initial_state
        current_params = _clamp_params_to_model_bounds(current_params, context.config)
    chain_initial_params = current_params

    if collect_phylo_root_samples:
        target_and_phylo_root_state = target_and_phylo_root_state or _build_mcmc_target_and_phylo_root_state(context)
        current_logp, current_phylo_root_state = target_and_phylo_root_state(current_params, current_noise)
    else:
        target = target or _build_mcmc_target(context)
        current_logp = target(current_params, current_noise)
        current_phylo_root_state = None

    log_posteriors: list[float] = []
    samples = _empty_parameter_buffers()
    phylo_root_samples: list[np.ndarray] = []
    accepted: list[bool] = []
    accepted_count = 0
    progress_interval = _progress_update_interval(driver_config.num_samples)
    pending_progress = 0

    iterator = tqdm(
        range(driver_config.num_samples),
        desc=_chain_progress_description(chain_index=chain_index, num_chains=num_chains),
        disable=not driver_config.progress_bar,
        leave=True,
        dynamic_ncols=True,
    )
    for sample_index in iterator:
        rng_key, subkey = jax.random.split(rng_key)
        proposed_params, proposed_noise = propose_state(
            current_params,
            current_noise,
            subkey,
            driver_config=driver_config,
            model_config=context.config,
        )
        if collect_phylo_root_samples:
            proposed_logp, proposed_phylo_root_state = target_and_phylo_root_state(proposed_params, proposed_noise)
        else:
            proposed_logp = target(proposed_params, proposed_noise)
            proposed_phylo_root_state = None
        log_alpha = proposed_logp - current_logp
        log_uniform = jnp.log(jax.random.uniform(subkey, dtype=proposed_logp.dtype))
        accept = bool(jnp.isfinite(proposed_logp) & (log_uniform < log_alpha))
        if accept:
            current_params = proposed_params
            current_noise = proposed_noise
            current_logp = proposed_logp
            current_phylo_root_state = proposed_phylo_root_state
            accepted_count += 1

        log_posteriors.append(float(current_logp))
        _append_parameter_sample(samples, current_params)
        if collect_phylo_root_samples:
            phylo_root_samples.append(_node_state_sample_array(current_phylo_root_state, context))
        accepted.append(accept)
        sample_count = sample_index + 1
        current_logp_float = float(current_logp)
        current_acceptance_rate = accepted_count / sample_count
        if driver_config.progress_bar:
            _set_progress_postfix(iterator, current_logp_float, current_acceptance_rate)
        if progress_callback is not None:
            pending_progress += 1
            if pending_progress >= progress_interval:
                progress_callback(
                    chain_index,
                    pending_progress,
                    current_logp_float,
                    current_acceptance_rate,
                )
                pending_progress = 0

    if progress_callback is not None and pending_progress:
        progress_callback(
            chain_index,
            pending_progress,
            current_logp_float,
            current_acceptance_rate,
        )

    return ChainResult(
        log_posteriors=np.asarray(log_posteriors, dtype=np.float64),
        samples=_parameter_buffer_arrays(samples),
        phylo_root_samples=(
            np.asarray(phylo_root_samples)
            if collect_phylo_root_samples
            else None
        ),
        accepted=np.asarray(accepted, dtype=np.bool_),
        acceptance_rate=accepted_count / driver_config.num_samples,
        initial_params=chain_initial_params,
        final_params=current_params,
        final_noise=np.asarray(current_noise),
    )


def run_mcmc_chains(
    context: BFFGContext,
    *,
    driver_config: MCMCDriverConfig | None = None,
    rng_key: jax.Array | None = None,
    dataset_h5_path: str | Path | None = None,
    remove_lmk: tuple[int, ...] | None = None,
    collect_phylo_root_samples: bool = False,
) -> list[ChainResult]:
    """Run multiple independent chains with the configured chain backend."""

    driver_config = driver_config or MCMCDriverConfig()
    _validate_driver_config(driver_config)
    rng_key = jax.random.PRNGKey(driver_config.random_seed) if rng_key is None else rng_key
    keys = jax.random.split(rng_key, driver_config.num_chains)
    if driver_config.chain_backend == "process":
        return _run_mcmc_chains_process(
            context,
            driver_config=driver_config,
            keys=keys,
            dataset_h5_path=dataset_h5_path,
            remove_lmk=remove_lmk,
            collect_phylo_root_samples=collect_phylo_root_samples,
        )
    return _run_mcmc_chains_sequential(
        context,
        driver_config=driver_config,
        keys=keys,
        collect_phylo_root_samples=collect_phylo_root_samples,
    )


def _run_mcmc_chains_sequential(
    context: BFFGContext,
    *,
    driver_config: MCMCDriverConfig,
    keys: jax.Array,
    collect_phylo_root_samples: bool,
) -> list[ChainResult]:
    """Run multiple chains sequentially, with one tqdm bar per chain."""

    target = _build_mcmc_target(context)
    target_and_phylo_root_state = (
        _build_mcmc_target_and_phylo_root_state(context)
        if collect_phylo_root_samples
        else None
    )
    results = []
    for chain_index in range(driver_config.num_chains):
        chain_key, init_key = jax.random.split(keys[chain_index])
        chain_initial = (
            initial_params_for_chain(
                init_key,
                chain_index=chain_index,
                driver_config=driver_config,
                model_config=context.config,
            ),
            jnp.zeros_like(context.tree[BFFG_FIELDS.zs]),
        )
        results.append(
            run_mcmc_chain(
                context,
                driver_config=driver_config,
                rng_key=chain_key,
                initial_state=chain_initial,
                chain_index=chain_index,
                num_chains=driver_config.num_chains,
                target=target,
                target_and_phylo_root_state=target_and_phylo_root_state,
                collect_phylo_root_samples=collect_phylo_root_samples,
            )
        )
    return results


def _run_mcmc_chains_process(
    context: BFFGContext,
    *,
    driver_config: MCMCDriverConfig,
    keys: jax.Array,
    dataset_h5_path: str | Path | None,
    remove_lmk: tuple[int, ...] | None,
    collect_phylo_root_samples: bool,
) -> list[ChainResult]:
    """Run independent chain chunks in spawned worker processes."""

    if dataset_h5_path is None:
        raise ValueError("dataset_h5_path is required when chain_backend='process'.")
    num_workers = _num_process_workers(driver_config)
    work_items = [
        (chain_index, _serialize_prng_key(keys[chain_index]))
        for chain_index in range(driver_config.num_chains)
    ]
    chunks = _chunked(work_items, num_workers)
    worker_config = replace(driver_config, progress_bar=False)
    spawn_context = multiprocessing.get_context("spawn")
    manager = spawn_context.Manager() if driver_config.progress_bar else None
    progress_queue = manager.Queue() if manager is not None else None
    worker_args = [
        (
            str(dataset_h5_path),
            context.config,
            worker_config,
            chunk,
            progress_queue,
            collect_phylo_root_samples,
            remove_lmk,
        )
        for chunk in chunks
    ]

    indexed_results: list[tuple[int, ChainResult]] = []
    total_progress, chain_progress = _open_process_progress_bars(driver_config)
    try:
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=spawn_context) as executor:
            pending = {executor.submit(_run_mcmc_chain_chunk_worker, args) for args in worker_args}
            while pending:
                done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                _drain_progress_queue(progress_queue, chain_progress)
                for future in done:
                    chunk_results = future.result()
                    indexed_results.extend(chunk_results)
                    if total_progress is not None:
                        total_progress.update(len(chunk_results))
                _drain_progress_queue(progress_queue, chain_progress)
    finally:
        _drain_progress_queue(progress_queue, chain_progress)
        _close_process_progress_bars(total_progress, chain_progress)
        if manager is not None:
            manager.shutdown()

    indexed_results.sort(key=lambda item: item[0])
    return [result for _, result in indexed_results]


def _run_mcmc_chain_chunk_worker(args) -> list[tuple[int, ChainResult]]:
    dataset_h5_path, model_config, driver_config, work_items, progress_queue, collect_phylo_root_samples, remove_lmk = args
    dataset = load_augmented_butterfly_tree(dataset_h5_path, remove_lmk=remove_lmk)
    context = build_bffg_context(dataset, model_config)
    target = _build_mcmc_target(context)
    target_and_phylo_root_state = (
        _build_mcmc_target_and_phylo_root_state(context)
        if collect_phylo_root_samples
        else None
    )
    progress_callback = None
    if progress_queue is not None:
        progress_callback = (
            lambda chain_index, step_delta, log_posterior, accept_rate: progress_queue.put(
                (chain_index, step_delta, log_posterior, accept_rate)
            )
        )
    results = []
    for chain_index, key_data in work_items:
        key = _deserialize_prng_key(key_data)
        chain_key, init_key = jax.random.split(key)
        chain_initial = (
            initial_params_for_chain(
                init_key,
                chain_index=chain_index,
                driver_config=driver_config,
                model_config=context.config,
            ),
            jnp.zeros_like(context.tree[BFFG_FIELDS.zs]),
        )
        result = run_mcmc_chain(
            context,
            driver_config=driver_config,
            rng_key=chain_key,
            initial_state=chain_initial,
            chain_index=chain_index,
            num_chains=driver_config.num_chains,
            target=target,
            target_and_phylo_root_state=target_and_phylo_root_state,
            collect_phylo_root_samples=collect_phylo_root_samples,
            progress_callback=progress_callback,
        )
        results.append((chain_index, result))
    return results


def run_mcmc(
    dataset: AugmentedButterflyTree,
    *,
    run_dir: str | Path,
    model_config: MCMCModelConfig | None = None,
    driver_config: MCMCDriverConfig | None = None,
) -> MCMCRunResult:
    """Run MCMC and persist local monitoring artifacts."""

    model_config = model_config or MCMCModelConfig()
    driver_config = driver_config or MCMCDriverConfig()
    _validate_driver_config(driver_config)
    run_dir = prepare_run_dir(run_dir)

    start = time.perf_counter()
    context = build_bffg_context(dataset, model_config)
    results = run_mcmc_chains(
        context,
        driver_config=driver_config,
        rng_key=jax.random.PRNGKey(driver_config.random_seed),
        dataset_h5_path=dataset.h5_path,
        remove_lmk=dataset.removed_landmarks,
        collect_phylo_root_samples=True,
    )
    summary, samples, log_posteriors, accepted = save_run_artifacts(
        run_dir,
        model_config=model_config,
        driver_config=driver_config,
        dataset=dataset,
        context=context,
        results=results,
        elapsed_seconds=time.perf_counter() - start,
    )
    return MCMCRunResult(
        run_dir=run_dir,
        samples=samples,
        log_posteriors=log_posteriors,
        accepted=accepted,
        summary=summary,
        context=context,
    )


def _lognormal_rw(value: jax.Array, key: jax.Array, proposal_var: float) -> jax.Array:
    return jnp.exp(jnp.log(value) + jnp.sqrt(jnp.asarray(proposal_var)) * jax.random.normal(key, shape=value.shape))


def _clamp_params_to_model_bounds(params: MCMCParams, model_config: MCMCModelConfig) -> MCMCParams:
    return MCMCParams(
        k_alpha=_clamp_parameter(params.k_alpha, model_config.k_alpha_min, model_config.k_alpha_max),
        k_sigma=_clamp_parameter(params.k_sigma, model_config.k_sigma_min, model_config.k_sigma_max),
        obs_var=_clamp_parameter(params.obs_var, model_config.obs_var_min, model_config.obs_var_max),
    )


def _clamp_parameter(value: jax.Array, lower: float, upper: float) -> jax.Array:
    value = jnp.asarray(value)
    return jnp.clip(
        value,
        jnp.asarray(lower, dtype=value.dtype),
        jnp.asarray(upper, dtype=value.dtype),
    )


def _sample_inverse_gamma(key: jax.Array, alpha: float, beta: float) -> jax.Array:
    gamma_draw = jax.random.gamma(key, jnp.asarray(alpha))
    return jnp.asarray(beta) / gamma_draw


def _empty_parameter_buffers() -> dict[str, list[float]]:
    return {name: [] for name in MCMC_PARAMETER_NAMES}


def _append_parameter_sample(buffers: dict[str, list[float]], params: MCMCParams) -> None:
    for name in MCMC_PARAMETER_NAMES:
        buffers[name].append(float(getattr(params, name)))


def _parameter_buffer_arrays(buffers: dict[str, list[float]]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(buffers[name], dtype=np.float64)
        for name in MCMC_PARAMETER_NAMES
    }


def _chain_progress_description(*, chain_index: int | None, num_chains: int | None) -> str:
    if chain_index is None:
        return "chain"
    if num_chains is None:
        return f"chain {chain_index + 1}"
    return f"chain {chain_index + 1}/{num_chains}"


def _build_mcmc_target(context: BFFGContext) -> Callable[[MCMCParams, jax.Array], jax.Array]:
    @jax.jit
    def target(params, noise):
        return mcmc_log_posterior(
            context.tree,
            context.leaf_observations,
            params,
            noise,
            context.config,
        )

    return target


def _build_mcmc_target_and_phylo_root_state(
    context: BFFGContext,
) -> Callable[[MCMCParams, jax.Array], tuple[jax.Array, jax.Array]]:
    phylo_root_index = _phylo_root_index(context)

    @jax.jit
    def target_and_phylo_root_state(params, noise):
        return mcmc_log_posterior_and_node_state(
            context.tree,
            context.leaf_observations,
            params,
            noise,
            context.config,
            node_index=phylo_root_index,
        )

    return target_and_phylo_root_state


def _phylo_root_index(context: BFFGContext) -> int:
    parents = np.asarray(context.tree.topology.parents)
    child_indices = np.flatnonzero(parents == context.root_index)
    child_indices = child_indices[child_indices != context.root_index]
    if child_indices.size != 1:
        raise ValueError(
            "Expected augmented super_root to have exactly one child, "
            f"found {child_indices.size}."
        )
    return int(child_indices[0])


def _node_state_sample_array(state: jax.Array | None, context: BFFGContext) -> np.ndarray:
    if state is None:
        raise ValueError("phylo_root state was not computed for this sample.")
    return np.asarray(state).reshape((context.n_landmarks, context.d_landmarks))


def _open_process_progress_bars(config: MCMCDriverConfig):
    if not config.progress_bar:
        return None, {}
    total_progress = tqdm(
        total=config.num_chains,
        desc="chains",
        position=0,
        leave=True,
        dynamic_ncols=True,
    )
    chain_progress = {
        chain_index: tqdm(
            total=config.num_samples,
            desc=_chain_progress_description(chain_index=chain_index, num_chains=config.num_chains),
            position=chain_index + 1,
            leave=True,
            dynamic_ncols=True,
        )
        for chain_index in range(config.num_chains)
    }
    return total_progress, chain_progress


def _close_process_progress_bars(total_progress, chain_progress) -> None:
    for progress in chain_progress.values():
        progress.close()
    if total_progress is not None:
        total_progress.close()


def _drain_progress_queue(progress_queue, chain_progress) -> None:
    if progress_queue is None:
        return
    while True:
        try:
            message = progress_queue.get_nowait()
        except queue.Empty:
            return
        chain_index, step_delta, log_posterior, accept_rate = _unpack_progress_message(message)
        if chain_index in chain_progress:
            chain_progress[chain_index].update(step_delta)
            _set_progress_postfix(chain_progress[chain_index], log_posterior, accept_rate)


def _progress_update_interval(num_samples: int) -> int:
    return max(1, num_samples // 100)


def _initial_param_value(configured_value, sampled_value: jax.Array, chain_index: int) -> jax.Array:
    if configured_value is None:
        return sampled_value
    if _is_initial_param_sequence(configured_value):
        configured_value = configured_value[chain_index]
    return jnp.asarray(float(configured_value), dtype=sampled_value.dtype)


def _is_initial_param_sequence(value: object) -> bool:
    return isinstance(value, (list, tuple, np.ndarray))


def _unpack_progress_message(message) -> tuple[int | None, int, float | None, float | None]:
    if len(message) == 2:
        chain_index, step_delta = message
        return chain_index, step_delta, None, None
    chain_index, step_delta, log_posterior, accept_rate = message
    return chain_index, step_delta, log_posterior, accept_rate


def _set_progress_postfix(progress, log_posterior: float | None, accept_rate: float | None) -> None:
    if log_posterior is None or accept_rate is None:
        return
    progress.set_postfix(
        {
            "log_posterior": f"{log_posterior:.3f}",
            "accept_rate": f"{accept_rate:.3f}",
        }
    )


def _num_process_workers(config: MCMCDriverConfig) -> int:
    requested = config.num_processes or (os.cpu_count() or 1)
    return max(1, min(config.num_chains, requested))


def _chunked(items: list, num_chunks: int) -> list[list]:
    chunk_size = math.ceil(len(items) / num_chunks)
    return [
        items[start : start + chunk_size]
        for start in range(0, len(items), chunk_size)
    ]


def _serialize_prng_key(key: jax.Array) -> tuple[int, ...]:
    return tuple(int(value) for value in np.asarray(jax.random.key_data(key), dtype=np.uint32).reshape(-1))


def _deserialize_prng_key(values: tuple[int, ...]) -> jax.Array:
    return jnp.asarray(values, dtype=jnp.uint32)


def _validate_driver_config(config: MCMCDriverConfig) -> None:
    if config.num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {config.num_samples}.")
    if config.num_chains < 1:
        raise ValueError(f"num_chains must be >= 1, got {config.num_chains}.")
    if config.chain_backend not in {"sequential", "process"}:
        raise ValueError(
            "chain_backend must be 'sequential' or 'process', "
            f"got {config.chain_backend!r}."
        )
    if config.num_processes < 0:
        raise ValueError(f"num_processes must be non-negative, got {config.num_processes}.")
    if not 0.0 <= config.pcn_eta <= 1.0:
        raise ValueError(f"pcn_eta must be in [0, 1], got {config.pcn_eta}.")
    for name in ("k_alpha_proposal_var", "k_sigma_proposal_var", "obs_var_proposal_var"):
        value = getattr(config, name)
        if value < 0 or not np.isfinite(value):
            raise ValueError(f"{name} must be finite and non-negative, got {value}.")


def _validate_initial_param_config(name: str, value: object, num_chains: int) -> None:
    if value is None:
        return
    if _is_initial_param_sequence(value):
        if len(value) != num_chains:
            raise ValueError(
                f"{name} list length must match num_chains; "
                f"got length {len(value)} for num_chains={num_chains}."
            )
        for item in value:
            _validate_initial_param_scalar(name, item)
        return
    _validate_initial_param_scalar(name, value)


def _validate_initial_param_scalar(name: str, value: object) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be null, a finite positive scalar, or a list of finite positive scalars.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be null, a finite positive scalar, or a list of finite positive scalars."
        ) from error
    if numeric <= 0.0 or not np.isfinite(numeric):
        raise ValueError(f"{name} must contain finite positive scalar values, got {value!r}.")
