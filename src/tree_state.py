"""Node-aligned tree state helpers for shape inputs and BFFG computation."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from src.loader import AugmentedButterflyTree


@dataclass(frozen=True)
class ShapeFields:
    """Names of input shape fields stored in augmented hyperiax trees."""

    coords: str = "coords"
    edge_len: str = "edge_len"
    is_hidden: str = "is_hidden"
    is_observed: str = "is_observed"


SHAPE_FIELDS = ShapeFields()


@dataclass(frozen=True)
class ShapeState:
    """Canonical node-aligned shape view used by inference and synthesis."""

    coords: np.ndarray
    is_hidden: np.ndarray
    is_observed: np.ndarray
    edge_len: np.ndarray
    root_index: int
    leaf_indices: np.ndarray
    root_value: np.ndarray
    leaf_observations: np.ndarray

    @property
    def node_count(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_landmarks(self) -> int:
        return int(self.coords.shape[1])

    @property
    def d_landmarks(self) -> int:
        return int(self.coords.shape[2])

    @property
    def state_dim(self) -> int:
        return self.n_landmarks * self.d_landmarks


def build_shape_state(dataset: AugmentedButterflyTree) -> ShapeState:
    """Build the canonical root/leaf/edge view of an augmented tree."""

    fields = SHAPE_FIELDS
    coords = np.asarray(dataset.tree[fields.coords], dtype=np.float32)
    if coords.ndim != 3:
        raise ValueError(f"Expected coords shape (nodes, landmarks, dims), got {coords.shape}.")

    is_hidden = np.asarray(dataset.tree[fields.is_hidden]).astype(bool)
    is_observed = np.asarray(dataset.tree[fields.is_observed]).astype(bool)
    if is_hidden.shape != (dataset.tree.size,):
        raise ValueError(f"is_hidden shape {is_hidden.shape} does not match tree size {dataset.tree.size}.")
    if is_observed.shape != (dataset.tree.size,):
        raise ValueError(f"is_observed shape {is_observed.shape} does not match tree size {dataset.tree.size}.")
    if not np.array_equal(is_observed, ~is_hidden):
        raise ValueError("is_observed must be exactly the complement of is_hidden.")

    root_mask = np.asarray(dataset.tree.topology.is_root)
    if int(root_mask.sum()) != 1:
        raise ValueError("Expected exactly one hyperiax root in augmented tree.")
    root_index = int(np.flatnonzero(root_mask)[0])

    leaf_indices = np.flatnonzero(np.asarray(dataset.tree.topology.is_leaf))
    state_dim = int(coords.shape[1] * coords.shape[2])
    root_value = coords[root_index].reshape((state_dim,))
    leaf_observations = coords[leaf_indices].reshape((leaf_indices.size, state_dim))
    if not np.isfinite(root_value).all():
        raise ValueError("super_root coordinates must be finite.")
    if not np.isfinite(leaf_observations).all():
        raise ValueError("Leaf observations must be finite.")

    edge_len = _shape_edge_len(dataset)
    if edge_len.shape != (dataset.tree.size,):
        raise ValueError(f"edge length shape {edge_len.shape} does not match tree size {dataset.tree.size}.")

    return ShapeState(
        coords=coords,
        is_hidden=is_hidden,
        is_observed=is_observed,
        edge_len=edge_len,
        root_index=root_index,
        leaf_indices=leaf_indices,
        root_value=root_value,
        leaf_observations=leaf_observations,
    )


def _shape_edge_len(dataset: AugmentedButterflyTree) -> np.ndarray:
    if dataset.edge_len_norm is not None:
        return np.asarray(dataset.edge_len_norm, dtype=np.float32)
    return np.asarray(dataset.tree[SHAPE_FIELDS.edge_len], dtype=np.float32)


@dataclass(frozen=True)
class BFFGFields:
    """Names of BFFG arrays stored in the hyperiax MCMC tree."""

    edge_len: str = "edge_len"
    vals: str = "vals"
    zs: str = "zs"
    ptnls: str = "ptnls"
    precs: str = "precs"
    ptnl_v: str = "ptnl_v"
    prec_v: str = "prec_v"
    anchor: str = "anchor"
    tildea_v: str = "tildea_v"
    log_norm: str = "log_norm"
    log_corr: str = "log_corr"


BFFG_FIELDS = BFFGFields()


def bffg_schema(*, num_edge_steps: int, n_landmarks: int, d_landmarks: int) -> dict[str, tuple]:
    """Return the hyperiax schema for factorized BFFG message state."""

    state_dim = n_landmarks * d_landmarks
    fields = BFFG_FIELDS
    return {
        fields.edge_len: (),
        fields.vals: (num_edge_steps + 1, state_dim),
        fields.zs: (num_edge_steps, state_dim),
        fields.ptnls: (num_edge_steps + 1, state_dim),
        fields.precs: (num_edge_steps + 1, n_landmarks, n_landmarks),
        fields.ptnl_v: (state_dim,),
        fields.prec_v: (n_landmarks, n_landmarks),
        fields.anchor: (state_dim,),
        fields.tildea_v: (n_landmarks, n_landmarks),
        fields.log_norm: (),
        fields.log_corr: (),
    }


def zero_bffg_arrays(
    *,
    node_count: int,
    num_edge_steps: int,
    n_landmarks: int,
    state_dim: int,
    dtype,
) -> dict[str, jnp.ndarray]:
    """Return zero-filled arrays for all factorized BFFG message fields."""

    fields = BFFG_FIELDS
    return {
        fields.vals: jnp.zeros((node_count, num_edge_steps + 1, state_dim), dtype=dtype),
        fields.zs: jnp.zeros((node_count, num_edge_steps, state_dim), dtype=dtype),
        fields.ptnls: jnp.zeros((node_count, num_edge_steps + 1, state_dim), dtype=dtype),
        fields.precs: jnp.zeros((node_count, num_edge_steps + 1, n_landmarks, n_landmarks), dtype=dtype),
        fields.ptnl_v: jnp.zeros((node_count, state_dim), dtype=dtype),
        fields.prec_v: jnp.zeros((node_count, n_landmarks, n_landmarks), dtype=dtype),
        fields.anchor: jnp.zeros((node_count, state_dim), dtype=dtype),
        fields.tildea_v: jnp.zeros((node_count, n_landmarks, n_landmarks), dtype=dtype),
        fields.log_norm: jnp.zeros((node_count,), dtype=dtype),
        fields.log_corr: jnp.zeros((node_count,), dtype=dtype),
    }
