"""Kunita landmark covariance and diffusion policy."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

LAPLACE_K1_R2_COEFF = 45.0 / 105.0
LAPLACE_K1_R3_COEFF = 10.0 / 105.0
LAPLACE_K1_R4_COEFF = 1.0 / 105.0


def laplace_k1_kernel(pair_diff: jax.Array, params: Any, *, dist_jitter: float | jax.Array) -> jax.Array:
    """Kunita landmark kernel."""

    k_alpha = _param(params, "k_alpha")
    k_sigma = _param(params, "k_sigma")
    r = jnp.sqrt(
        jnp.asarray(dist_jitter, dtype=pair_diff.dtype)
        + jnp.sum(jnp.square(pair_diff / k_sigma), axis=-1)
    )
    polynomial = (
        1.0
        + r
        + LAPLACE_K1_R2_COEFF * r**2
        + LAPLACE_K1_R3_COEFF * r**3
        + LAPLACE_K1_R4_COEFF * r**4
    )
    return k_alpha**2 * polynomial * jnp.exp(-r)


def kernel_matrix(
    q: jax.Array,
    params: Any,
    d_landmarks: int | None = None,
    *,
    dist_jitter: float | jax.Array,
) -> jax.Array:
    """Spatial kernel on landmark pairs."""

    points = _points_from_state(q, d_landmarks)
    pair_diff = points[:, None, :] - points[None, :, :]
    return laplace_k1_kernel(pair_diff, params, dist_jitter=dist_jitter)


def covariance_matrix(
    q: jax.Array,
    params: Any,
    *,
    d_landmarks: int | None = None,
    covar_jitter: float | jax.Array = 0.0,
    dist_jitter: float | jax.Array = 0.0,
) -> jax.Array:
    """Landmark covariance a(q) = sigma(q) sigma(q).T."""

    if isinstance(params, dict):
        covar_jitter = params.get("covar_jitter", covar_jitter)
        dist_jitter = params.get("dist_jitter", dist_jitter)
    covariance = kernel_matrix(q, params, d_landmarks=d_landmarks, dist_jitter=dist_jitter)
    covariance = 0.5 * (covariance + covariance.T)
    nugget = jnp.asarray(covar_jitter, dtype=covariance.dtype) * jnp.eye(covariance.shape[0], dtype=covariance.dtype)
    return covariance + nugget


def diffusion_matrix(
    q: jax.Array,
    params: Any,
    *,
    d_landmarks: int | None = None,
    covar_jitter: float | jax.Array,
    dist_jitter: float | jax.Array,
) -> jax.Array:
    """Cholesky factor of the scaled landmark covariance."""

    covariance = covariance_matrix(
        q,
        params,
        d_landmarks=d_landmarks,
        covar_jitter=covar_jitter,
        dist_jitter=dist_jitter,
    )
    return jax.scipy.linalg.cholesky(covariance, lower=True, check_finite=False)


def _param(params: Any, name: str):
    if isinstance(params, dict):
        return params[name]
    return getattr(params, name)


def _points_from_state(q: jax.Array, d_landmarks: int | None = None) -> jax.Array:
    q = jnp.asarray(q)
    if q.ndim == 2:
        if d_landmarks is not None and q.shape[1] != d_landmarks:
            raise ValueError(
                f"Expected point coordinates with final dimension {d_landmarks}, got shape {q.shape}."
            )
        return q
    if q.ndim == 1:
        d_landmarks = 2 if d_landmarks is None else int(d_landmarks)
        if q.shape[0] % d_landmarks != 0:
            raise ValueError(
                f"Flat state dimension {q.shape[0]} is not divisible by coordinate dimension {d_landmarks}."
            )
        return q.reshape((-1, d_landmarks))
    raise ValueError(f"Expected q with 1 or 2 dimensions, got shape {q.shape}.")
