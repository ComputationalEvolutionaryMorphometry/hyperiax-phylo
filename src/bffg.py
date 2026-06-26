"""BFFG target for full-shape MCMC."""

from __future__ import annotations

from dataclasses import dataclass

import hyperiax as hx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln

from src.tree_state import BFFG_FIELDS, bffg_schema, build_shape_state, zero_bffg_arrays
from src.kunita import (
    covariance_matrix,
    diffusion_matrix,
)
from src.loader import AugmentedButterflyTree


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MCMCParams:
    """Positive scalar MCMC parameters carried as a JAX pytree."""

    k_alpha: jax.Array
    k_sigma: jax.Array
    obs_var: jax.Array

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_alpha", jnp.asarray(self.k_alpha))
        object.__setattr__(self, "k_sigma", jnp.asarray(self.k_sigma))
        object.__setattr__(self, "obs_var", jnp.asarray(self.obs_var))

    def tree_flatten(self):
        return (self.k_alpha, self.k_sigma, self.obs_var), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)

    def as_dict(self) -> dict[str, jax.Array]:
        return {
            "k_alpha": self.k_alpha,
            "k_sigma": self.k_sigma,
            "obs_var": self.obs_var,
        }


@dataclass(frozen=True)
class MCMCModelConfig:
    """Configuration for the factorized Kunita/BFFG target."""

    num_edge_steps: int = 25
    covar_jitter: float = 0.0
    dist_jitter: float = 0.0
    k_alpha_init: float | list[float] | tuple[float, ...] | None = None
    k_alpha_prior_alpha: float = 3.0
    k_alpha_prior_beta: float = 0.1
    k_alpha_max: float = 1.0
    k_alpha_min: float = 1e-5
    k_sigma_init: float | list[float] | tuple[float, ...] | None = None
    k_sigma_prior_alpha: float = 3.0
    k_sigma_prior_beta: float = 0.1
    k_sigma_max: float = 1.0
    k_sigma_min: float = 1e-5
    obs_var_init: float | list[float] | tuple[float, ...] | None = None
    obs_var_prior_alpha: float = 2.0
    obs_var_prior_beta: float = 0.003
    obs_var_max: float = 0.1
    obs_var_min: float = 1e-5
    enable_x64: bool = True


@dataclass(frozen=True)
class BFFGContext:
    """Node-aligned BFFG state for MCMC."""

    config: MCMCModelConfig
    tree: hx.Tree
    root_value: jax.Array
    leaf_observations: jax.Array
    root_index: int
    leaf_indices: np.ndarray
    node_count: int
    n_landmarks: int
    d_landmarks: int
    state_dim: int
    noise_size: int
    edge_len_normalizer: float


def dot_factorized(mat: jax.Array, vec: jax.Array) -> jax.Array:
    """Multiply a landmark matrix across coordinate blocks."""
    n = mat.shape[0]
    return jnp.einsum("ij,jd->id", mat, vec.reshape((n, -1))).reshape(vec.shape)


def solve_factorized(mat: jax.Array, vec: jax.Array) -> jax.Array:
    """Solve a landmark matrix against coordinate-blocked vectors or matrices."""
    n = mat.shape[0]
    return jnp.linalg.solve(mat, vec.reshape((n, -1))).reshape(vec.shape)


def normal_logpdf(x: jax.Array, loc: float | jax.Array, scale: jax.Array) -> jax.Array:
    z = (x - loc) / scale
    return -0.5 * z**2 - jnp.log(scale) - 0.5 * jnp.log(2.0 * jnp.pi)


def inverse_gamma_logpdf(x: jax.Array, alpha: float, beta: float) -> jax.Array:
    value = alpha * jnp.log(beta) - gammaln(alpha) - (alpha + 1.0) * jnp.log(x) - beta / x
    return jnp.where(x > 0.0, value, -jnp.inf)


def log_gaussian_prec(x: jax.Array, mean: jax.Array, prec: jax.Array) -> jax.Array:
    _, logdet = jnp.linalg.slogdet(prec)
    log_norm = 0.5 * (logdet - jnp.log(2.0 * jnp.pi) * prec.shape[0])
    diff = x - mean
    return log_norm - 0.5 * diff @ prec @ diff


def log_gaussian_cov(x: jax.Array, mean: jax.Array, covar: jax.Array) -> jax.Array:
    _, logdet = jnp.linalg.slogdet(covar)
    diff = x - mean
    solved = jnp.linalg.solve(covar, diff)
    return -0.5 * (covar.shape[0] * jnp.log(2.0 * jnp.pi) + logdet + diff @ solved)


def forward_guided(
    start: jax.Array,
    precs: jax.Array,
    ptnls: jax.Array,
    covar_prxy: jax.Array,
    dts: jax.Array,
    noise: jax.Array,
    params: MCMCParams,
    *,
    covar_jitter: float | jax.Array = 0.0,
    dist_jitter: float | jax.Array = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Guided forward pass with child-edge constant proxy covariance."""

    n_landmarks = precs.shape[-1]
    d_landmarks = _coordinate_dim_from_state_dim(start.shape[0], n_landmarks)
    dws = jnp.sqrt(dts)[:, None] * noise
    sigma = diffusion_matrix(
        start,
        params,
        d_landmarks=d_landmarks,
        covar_jitter=covar_jitter,
        dist_jitter=dist_jitter,
    )
    covar = covariance_matrix(
        start,
        params,
        d_landmarks=d_landmarks,
        covar_jitter=covar_jitter,
        dist_jitter=dist_jitter,
    )

    def step(carry, val):
        index, time, x, log_corr = carry
        dt, dw = val
        H = precs[index]
        F = ptnls[index]
        r_prxy = F - dot_factorized(H, x)
        x_next = x + dot_factorized(covar, r_prxy) * dt + dot_factorized(sigma, dw)
        a_minus_tildea = covar - covar_prxy
        r_prxy = r_prxy.reshape((n_landmarks, -1))
        log_corr_next = log_corr + (
            -0.5 * jnp.einsum("ij,ji->", a_minus_tildea, H)
            + 0.5 * jnp.einsum("ij,jd,id->", a_minus_tildea, r_prxy, r_prxy)
        ) * dt
        return (index + 1, time + dt, x_next, log_corr_next), (x, log_corr)

    (_, _, final_x, log_corr), (xs, _) = jax.lax.scan(
        step,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=start.dtype),
            start,
            jnp.asarray(0.0, dtype=start.dtype),
        ),
        (dts, dws),
    )
    return jnp.vstack((xs, final_x)), log_corr


def build_bffg_context(dataset: AugmentedButterflyTree, config: MCMCModelConfig) -> BFFGContext:
    """Build a factorized BFFG context from current HDF5 data."""

    jax.config.update("jax_enable_x64", config.enable_x64)
    _validate_model_config(config)
    dtype = jnp.float64 if config.enable_x64 else jnp.float32

    shape_state = build_shape_state(dataset)
    node_count = shape_state.node_count
    n_landmarks = shape_state.n_landmarks
    d_landmarks = shape_state.d_landmarks
    state_dim = shape_state.state_dim
    edge_len = jnp.asarray(shape_state.edge_len, dtype=dtype)
    root_index = shape_state.root_index
    root_value = jnp.asarray(shape_state.root_value, dtype=dtype)
    leaf_indices = shape_state.leaf_indices
    leaf_observations = jnp.asarray(shape_state.leaf_observations, dtype=dtype)

    tree = _empty_mcmc_tree(
        topology=dataset.tree.topology,
        num_edge_steps=config.num_edge_steps,
        n_landmarks=n_landmarks,
        d_landmarks=d_landmarks,
        dtype=dtype,
    ).set(edge_len=edge_len)
    vals = jnp.broadcast_to(root_value, (node_count, config.num_edge_steps + 1, state_dim))
    leaf_vals = jnp.broadcast_to(
        leaf_observations[:, None, :],
        (leaf_indices.size, config.num_edge_steps + 1, state_dim),
    )
    vals = vals.at[jnp.asarray(leaf_indices)].set(leaf_vals)
    zeros = zero_bffg_arrays(
        node_count=node_count,
        num_edge_steps=config.num_edge_steps,
        n_landmarks=n_landmarks,
        state_dim=state_dim,
        dtype=dtype,
    )
    anchors = jnp.broadcast_to(root_value, (node_count, state_dim))
    zeros[BFFG_FIELDS.vals] = vals
    zeros[BFFG_FIELDS.anchor] = anchors
    tree = tree.set(**zeros)

    return BFFGContext(
        config=config,
        tree=tree,
        root_value=root_value,
        leaf_observations=leaf_observations,
        root_index=root_index,
        leaf_indices=leaf_indices,
        node_count=node_count,
        n_landmarks=n_landmarks,
        d_landmarks=d_landmarks,
        state_dim=state_dim,
        noise_size=node_count * config.num_edge_steps * state_dim,
        edge_len_normalizer=dataset.edge_len_normalizer,
    )


@hx.up(
    reads_children=(
        BFFG_FIELDS.edge_len,
        BFFG_FIELDS.ptnl_v,
        BFFG_FIELDS.prec_v,
        BFFG_FIELDS.tildea_v,
        BFFG_FIELDS.ptnls,
    ),
    writes=(
        BFFG_FIELDS.ptnl_v,
        BFFG_FIELDS.prec_v,
        BFFG_FIELDS.tildea_v,
        BFFG_FIELDS.anchor,
        BFFG_FIELDS.log_norm,
    ),
    writes_children=(BFFG_FIELDS.ptnls, BFFG_FIELDS.precs),
)
def mcmc_up_sweep(node, children, params: MCMCParams | dict) -> dict:
    del node

    messages = children.map(_pull_back_child_message)
    prec_v = messages.m_prec.sum(0)
    ptnl_v = messages.m_ptnl.sum(0)
    d_landmarks = _coordinate_dim_from_state_dim(ptnl_v.shape[-1], prec_v.shape[-1])
    anchor = jax.vmap(lambda H, F: solve_factorized(H, F))(prec_v, ptnl_v)
    tildea_v = jax.vmap(
        lambda value, H: covariance_matrix(value, params, d_landmarks=d_landmarks)
    )(anchor, prec_v)
    return {
        BFFG_FIELDS.precs: messages.precs,
        BFFG_FIELDS.ptnls: messages.ptnls,
        BFFG_FIELDS.prec_v: prec_v,
        BFFG_FIELDS.ptnl_v: ptnl_v,
        BFFG_FIELDS.tildea_v: tildea_v,
        BFFG_FIELDS.anchor: anchor,
        BFFG_FIELDS.log_norm: messages.m_log_norm.sum(0),
    }


def _pull_back_child_message(child) -> dict:
    n_landmarks = child.prec_v.shape[-1]
    d_landmarks = _coordinate_dim_from_state_dim(child.ptnl_v.shape[-1], n_landmarks)
    num_edge_steps = child.ptnls.shape[0] - 1
    edge_len = jnp.squeeze(child.edge_len)
    eye = jnp.eye(n_landmarks, dtype=child.prec_v.dtype)
    times = jnp.linspace(0.0, edge_len, num_edge_steps + 1, dtype=edge_len.dtype)

    def message_at_time(time):
        phi_inv = eye + child.prec_v @ child.tildea_v * (edge_len - time)
        prec_t = solve_factorized(phi_inv, child.prec_v).reshape(child.prec_v.shape)
        ptnl_t = solve_factorized(phi_inv, child.ptnl_v).reshape(child.ptnl_v.shape)
        return prec_t, ptnl_t

    precs, ptnls = jax.vmap(message_at_time)(times)
    v_T = solve_factorized(child.prec_v, child.ptnl_v)
    log_norm_at_parent = jax.vmap(
        lambda coordinate_values: log_gaussian_prec(
            jnp.zeros(n_landmarks, dtype=child.prec_v.dtype),
            coordinate_values,
            precs[0],
        ),
        in_axes=1,
    )(v_T.reshape((n_landmarks, d_landmarks))).sum()
    return {
        BFFG_FIELDS.precs: precs,
        BFFG_FIELDS.ptnls: ptnls,
        "m_prec": precs[0],
        "m_ptnl": ptnls[0],
        "m_log_norm": log_norm_at_parent,
    }


@hx.down(
    reads=(
        BFFG_FIELDS.edge_len,
        BFFG_FIELDS.zs,
        BFFG_FIELDS.ptnls,
        BFFG_FIELDS.precs,
        BFFG_FIELDS.tildea_v,
    ),
    reads_parent=(BFFG_FIELDS.vals,),
    writes=(BFFG_FIELDS.vals, BFFG_FIELDS.log_corr),
)
def mcmc_down_sweep(node, parent, params: MCMCParams | dict):
    num_edge_steps = node.zs.shape[0]
    edge_len = jnp.squeeze(node.edge_len)
    increments = jnp.full((num_edge_steps,), edge_len / num_edge_steps, dtype=edge_len.dtype)
    start = parent.vals[-1]
    vals, log_corr = forward_guided(start, node.precs, node.ptnls, node.tildea_v, increments, node.zs, params)
    return {BFFG_FIELDS.vals: vals, BFFG_FIELDS.log_corr: log_corr}


def init_mcmc_messages(tree: hx.Tree, leaf_values: jax.Array, params: MCMCParams | dict) -> hx.Tree:
    leaf_indices = jnp.asarray(np.flatnonzero(np.asarray(tree.topology.is_leaf)), dtype=jnp.int32)
    n_landmarks = tree[BFFG_FIELDS.prec_v].shape[-1]
    d_landmarks = _coordinate_dim_from_state_dim(leaf_values.shape[-1], n_landmarks)
    eye = jnp.eye(n_landmarks, dtype=leaf_values.dtype)
    obs_var = _param(params, "obs_var")
    prec_leaf = eye / obs_var
    prec_v = jnp.broadcast_to(prec_leaf, (leaf_values.shape[0], n_landmarks, n_landmarks))
    ptnl_v = jax.vmap(lambda value: dot_factorized(prec_leaf, value))(leaf_values)
    covariance = obs_var * eye
    log_norm = jax.vmap(
        lambda value: jax.vmap(
            lambda coordinate_values: log_gaussian_cov(
                jnp.zeros(n_landmarks, dtype=leaf_values.dtype),
                coordinate_values,
                covariance,
            ),
            in_axes=1,
        )(value.reshape((n_landmarks, d_landmarks))).sum()
    )(leaf_values)
    tildea_v = jax.vmap(lambda value: covariance_matrix(value, params, d_landmarks=d_landmarks))(leaf_values)
    leaf_vals = tree[BFFG_FIELDS.vals][leaf_indices].at[:, -1, :].set(leaf_values)
    return tree.at[leaf_indices].set(
        vals=leaf_vals,
        prec_v=prec_v,
        ptnl_v=ptnl_v,
        log_norm=log_norm,
        tildea_v=tildea_v,
        anchor=leaf_values,
    )


def run_mcmc_sweeps(
    tree: hx.Tree,
    leaf_values: jax.Array,
    *,
    params: MCMCParams | dict,
    noise: jax.Array,
) -> hx.Tree:
    initialized = init_mcmc_messages(tree, leaf_values, params).set(zs=noise)
    filtered = mcmc_up_sweep(initialized, params=params)
    return mcmc_down_sweep(filtered, params=params)


def mcmc_target_components(
    tree: hx.Tree,
    leaf_values: jax.Array,
    params: MCMCParams,
    noise: jax.Array,
    config: MCMCModelConfig,
) -> dict[str, jax.Array]:
    guided = _guided_mcmc_tree(tree, leaf_values, params, noise, config)
    return _mcmc_target_components_from_guided(guided, leaf_values, params, config)


def mcmc_log_posterior_and_node_state(
    tree: hx.Tree,
    leaf_values: jax.Array,
    params: MCMCParams,
    noise: jax.Array,
    config: MCMCModelConfig,
    *,
    node_index: int,
) -> tuple[jax.Array, jax.Array]:
    """Return the target value and terminal state for one node from the same sweep."""

    guided = _guided_mcmc_tree(tree, leaf_values, params, noise, config)
    components = _mcmc_target_components_from_guided(guided, leaf_values, params, config)
    return components["log_target"], guided[BFFG_FIELDS.vals][node_index, -1]


def _guided_mcmc_tree(
    tree: hx.Tree,
    leaf_values: jax.Array,
    params: MCMCParams,
    noise: jax.Array,
    config: MCMCModelConfig,
) -> hx.Tree:
    runtime_params = _runtime_params(params, config)
    return run_mcmc_sweeps(tree, leaf_values, params=runtime_params, noise=noise)


def _mcmc_target_components_from_guided(
    guided: hx.Tree,
    leaf_values: jax.Array,
    params: MCMCParams,
    config: MCMCModelConfig,
) -> dict[str, jax.Array]:
    root_index = int(np.flatnonzero(np.asarray(guided.topology.is_root))[0])
    leaf_indices = jnp.asarray(np.flatnonzero(np.asarray(guided.topology.is_leaf)), dtype=jnp.int32)

    root_log_likelihood = _root_log_likelihood(guided, root_index)
    residuals = guided[BFFG_FIELDS.vals][leaf_indices, -1] - leaf_values
    leaf_log_likelihood = jnp.mean(normal_logpdf(residuals, 0.0, jnp.sqrt(params.obs_var)))
    log_corrections = guided[BFFG_FIELDS.log_corr][1:]
    log_corr = jnp.mean(log_corrections, axis=0)
    log_prior = mcmc_log_prior(params, config)
    log_likelihood = root_log_likelihood + log_corr + leaf_log_likelihood
    return {
        "log_prior": log_prior,
        "root_log_likelihood": root_log_likelihood,
        "log_corr": log_corr,
        "leaf_log_likelihood": leaf_log_likelihood,
        "log_likelihood": log_likelihood,
        "log_target": log_prior + log_likelihood,
    }


def mcmc_log_prior(params: MCMCParams, config: MCMCModelConfig) -> jax.Array:
    obs_var_prior = inverse_gamma_logpdf(params.obs_var, config.obs_var_prior_alpha, config.obs_var_prior_beta)
    return (
        inverse_gamma_logpdf(params.k_alpha, config.k_alpha_prior_alpha, config.k_alpha_prior_beta)
        + inverse_gamma_logpdf(params.k_sigma, config.k_sigma_prior_alpha, config.k_sigma_prior_beta)
        + obs_var_prior
    )


def mcmc_log_posterior(
    tree: hx.Tree,
    leaf_values: jax.Array,
    params: MCMCParams,
    noise: jax.Array,
    config: MCMCModelConfig,
) -> jax.Array:
    return mcmc_target_components(
        tree,
        leaf_values,
        params,
        noise,
        config,
    )["log_target"]


def _root_log_likelihood(guided: hx.Tree, root_index: int) -> jax.Array:
    root_value = guided[BFFG_FIELDS.vals][root_index, 0]
    log_norm = guided[BFFG_FIELDS.log_norm][root_index]
    ptnl = guided[BFFG_FIELDS.ptnl_v][root_index]
    prec = guided[BFFG_FIELDS.prec_v][root_index]
    return log_norm + ptnl @ root_value - 0.5 * root_value @ dot_factorized(prec, root_value)


def _empty_mcmc_tree(
    *,
    topology: hx.Topology,
    num_edge_steps: int,
    n_landmarks: int,
    d_landmarks: int,
    dtype,
) -> hx.Tree:
    state_dim = n_landmarks * d_landmarks
    schema = bffg_schema(
        num_edge_steps=num_edge_steps,
        n_landmarks=n_landmarks,
        d_landmarks=d_landmarks,
    )
    return hx.Tree.empty(topology, schema).set(
        edge_len=jnp.zeros((topology.size,), dtype=dtype),
        **zero_bffg_arrays(
            node_count=topology.size,
            num_edge_steps=num_edge_steps,
            n_landmarks=n_landmarks,
            state_dim=state_dim,
            dtype=dtype,
        ),
    )


def _coordinate_dim_from_state_dim(state_dim: int, n_landmarks: int) -> int:
    state_dim = int(state_dim)
    n_landmarks = int(n_landmarks)
    if n_landmarks < 1:
        raise ValueError(f"n_landmarks must be positive, got {n_landmarks}.")
    if state_dim % n_landmarks != 0:
        raise ValueError(
            f"state_dim {state_dim} is not divisible by n_landmarks {n_landmarks}."
        )
    return state_dim // n_landmarks


def _validate_model_config(config: MCMCModelConfig) -> None:
    if config.num_edge_steps < 1:
        raise ValueError(f"num_edge_steps must be >= 1, got {config.num_edge_steps}.")
    if config.covar_jitter < 0:
        raise ValueError(f"covar_jitter must be non-negative, got {config.covar_jitter}.")
    if config.dist_jitter < 0:
        raise ValueError(
            f"dist_jitter must be non-negative, got {config.dist_jitter}."
        )
    if config.k_sigma_max <= 0:
        raise ValueError(f"k_sigma_max must be positive, got {config.k_sigma_max}.")
    if config.k_sigma_min <= 0:
        raise ValueError(f"k_sigma_min must be positive, got {config.k_sigma_min}.")
    if config.k_alpha_max <= 0:
        raise ValueError(f"k_alpha_max must be positive, got {config.k_alpha_max}.")
    if config.k_alpha_min <= 0:
        raise ValueError(f"k_alpha_min must be positive, got {config.k_alpha_min}.")
    if config.obs_var_max <= 0:
        raise ValueError(f"obs_var_max must be positive, got {config.obs_var_max}.")
    if config.obs_var_min <= 0:
        raise ValueError(f"obs_var_min must be positive, got {config.obs_var_min}.")
    for name in ("k_alpha_init", "k_sigma_init", "obs_var_init"):
        _validate_initial_param_config(name, getattr(config, name))


def _validate_initial_param_config(name: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, (list, tuple, np.ndarray)):
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


def _runtime_params(params: MCMCParams, config: MCMCModelConfig) -> dict[str, jax.Array | float]:
    payload = params.as_dict()
    payload["covar_jitter"] = config.covar_jitter
    payload["dist_jitter"] = config.dist_jitter
    return payload


def _param(params: MCMCParams | dict, name: str):
    if isinstance(params, dict):
        return params[name]
    return getattr(params, name)
