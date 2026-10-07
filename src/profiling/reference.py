"""Python node traversal with JIT-compiled local operations.

This intentionally reproduces the current implementation, including its target
reductions and edge updates. It is an execution reference, not a new model.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from src.bffg import (
    _pull_back_child_message, _runtime_params, covariance_matrix, dot_factorized,
    forward_guided, init_mcmc_messages, mcmc_log_prior, normal_logpdf, solve_factorized,
)
from src.tree_state import BFFG_FIELDS as f


class Child(NamedTuple):
    edge_len: object
    ptnl_v: object
    prec_v: object
    tildea_v: object
    ptnls: object


def build_reference_target(context):
    """Return the same (log_target, phylo_root_endpoint) as the production target."""
    parents = np.asarray(context.tree.topology.parents)
    children = [np.flatnonzero(parents == i).tolist() for i in range(context.node_count)]
    children[0].remove(0)
    root_child = children[0][0]
    leaves = context.leaf_indices.tolist()
    edge_lengths = np.asarray(context.tree[f.edge_len])
    steps = context.config.num_edge_steps
    template = jnp.zeros((steps + 1, context.state_dim), dtype=context.root_value.dtype)

    @jax.jit
    def initialize(params, noise):
        tree = init_mcmc_messages(context.tree, context.leaf_observations, _runtime_params(params, context.config))
        # Split in compiled code, not with thousands of eager device gathers.
        messages = tuple((tree[f.prec_v][i], tree[f.ptnl_v][i], tree[f.tildea_v][i], tree[f.log_norm][i])
                         for i in range(context.node_count))
        return messages, tuple(noise[i] for i in range(context.node_count))

    @jax.jit
    def up(child_states, params):
        messages = tuple(_pull_back_child_message(child) for child in child_states)
        H = jnp.stack([m["m_prec"] for m in messages]).sum(0)
        F = jnp.stack([m["m_ptnl"] for m in messages]).sum(0)
        anchor = solve_factorized(H, F)
        A = covariance_matrix(anchor, params, d_landmarks=context.d_landmarks)
        c = jnp.stack([m["m_log_norm"] for m in messages]).sum(0)
        return (H, F, A, c), tuple((m[f.precs], m[f.ptnls]) for m in messages)

    @jax.jit
    def down(start, H, F, A, length, noise, params):
        dts = jnp.full((steps,), length / steps, dtype=context.root_value.dtype)
        values, correction = forward_guided(start, H, F, A, dts, noise, params)
        return values[-1], correction

    @jax.jit
    def finish(root_message, endpoints, corrections, params):
        H, F, _, c = root_message
        x = context.root_value
        root = c + F @ x - 0.5 * x @ dot_factorized(H, x)
        residuals = jnp.stack([endpoints[i] for i in leaves]) - context.leaf_observations
        leaf = jnp.mean(normal_logpdf(residuals, 0.0, jnp.sqrt(params.obs_var)))
        correction = jnp.mean(jnp.stack(corrections[1:]))
        return mcmc_log_prior(params, context.config) + root + correction + leaf, endpoints[root_child]

    def target(params, noise):
        initial, noises = initialize(params, noise)
        messages = list(initial)
        edges = [None] * context.node_count
        runtime = _runtime_params(params, context.config)
        # BFS node numbering implies reversed numbering is a valid postorder.
        for index in reversed(range(context.node_count)):
            if not children[index]:
                continue
            payloads = tuple(Child(edge_lengths[child], messages[child][1], messages[child][0],
                                   messages[child][2], template) for child in children[index])
            messages[index], updates = up(payloads, runtime)
            for child, update in zip(children[index], updates):
                edges[child] = update
        endpoints = [context.root_value] * context.node_count
        corrections = [jnp.asarray(0.0, dtype=context.root_value.dtype)] * context.node_count
        for index in range(1, context.node_count):
            H, F = edges[index]
            endpoints[index], corrections[index] = down(
                endpoints[parents[index]], H, F, messages[index][2], edge_lengths[index], noises[index], runtime,
            )
        return finish(messages[0], tuple(endpoints), tuple(corrections), params)

    return target
