"""Build hyperiax trees and node-aligned shape state from HDF5 datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import jax.numpy as jnp
import numpy as np

from hyperiax import Topology, Tree, from_newick

from src import data_schema as schema


@dataclass(frozen=True)
class AugmentedButterflyTree:
    """Node-aligned augmented butterfly shape data.

    The ``tree`` field is in hyperiax BFS order. Coordinates are node-aligned:
    observed rows contain scaled landmarks, and hidden rows contain NaNs.
    """

    tree: Tree
    node_names: tuple[str, ...]
    node_types: tuple[str, ...]
    leaf_names: tuple[str, ...]
    newick_aug: str
    h5_path: Path
    normalization_center: np.ndarray
    normalization_scale: float
    original_landmark_count: int = 0
    removed_landmarks: tuple[int, ...] = ()
    kept_landmarks: tuple[int, ...] = ()
    edge_len_normalizer: float = 1.0
    edge_len_aug: np.ndarray | None = None
    edge_len_norm: np.ndarray | None = None


def load_augmented_butterfly_tree(
    h5_path: str | Path,
    *,
    remove_lmk: object = None,
) -> AugmentedButterflyTree:
    """Load an augmented butterfly HDF5 file as a node-aligned hyperiax tree."""
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as h5:
        _require_keys(
            h5,
            list(schema.REQUIRED_DATASET_PATHS),
        )
        newick_aug = _read_utf8_scalar(h5[schema.TREE_NEWICK_AUG][()])
        node_names = _read_utf8_array(h5[schema.NODES_NAMES][:])
        node_types = _read_utf8_array(h5[schema.NODES_NODE_TYPE][:])
        leaf_names = _read_utf8_array(h5[schema.TREE_LEAF_NAMES][:])
        coords = h5[schema.NODES_COORDS][:].astype(np.float32, copy=False)
        edge_len_ori = h5[schema.NODES_EDGE_LEN_ORI][:].astype(np.float64, copy=False)
        edge_len_aug = h5[schema.NODES_EDGE_LEN_AUG][:].astype(np.float64, copy=False)
        edge_len_norm = h5[schema.NODES_EDGE_LEN_NORM][:].astype(np.float32, copy=False)
        is_hidden = h5[schema.NODES_IS_HIDDEN][:].astype(np.bool_)
        h5_is_leaf = h5[schema.NODES_IS_LEAF][:].astype(np.bool_)
        h5_is_root = h5[schema.NODES_IS_ROOT][:].astype(np.bool_)
        normalization_center = h5[schema.PREPROC_CENTER][:]
        normalization_scale = float(h5[schema.PREPROC_SCALE][()])
        edge_len_normalizer = float(h5[schema.PREPROC_EDGE_LEN_NORMALIZER][()])

    original_landmark_count = int(coords.shape[1])
    removed_landmarks = normalize_remove_lmk(remove_lmk, landmark_count=original_landmark_count)
    kept_landmarks = tuple(index for index in range(original_landmark_count) if index not in set(removed_landmarks))
    if removed_landmarks:
        coords = coords[:, kept_landmarks, :]

    parsed_tree = from_newick(newick_aug)
    _validate_augmented_node_alignment(
        parsed_tree=parsed_tree,
        node_names=node_names,
        node_types=node_types,
        coords=coords,
        edge_len_ori=edge_len_ori,
        edge_len_aug=edge_len_aug,
        edge_len_norm=edge_len_norm,
        edge_len_normalizer=edge_len_normalizer,
        is_hidden=is_hidden,
        h5_is_leaf=h5_is_leaf,
        h5_is_root=h5_is_root,
    )

    is_observed = np.isfinite(coords).all(axis=(1, 2))
    topology = Topology.from_parents(parsed_tree.topology.parents, names=node_names)
    tree = Tree.from_data(
        topology,
        {
            "coords": jnp.asarray(coords),
            "edge_len": jnp.asarray(edge_len_norm),
            "is_hidden": jnp.asarray(is_hidden),
            "is_observed": jnp.asarray(is_observed),
        },
    )

    return AugmentedButterflyTree(
        tree=tree,
        node_names=node_names,
        node_types=node_types,
        leaf_names=leaf_names,
        newick_aug=newick_aug,
        h5_path=h5_path,
        original_landmark_count=original_landmark_count,
        removed_landmarks=removed_landmarks,
        kept_landmarks=kept_landmarks,
        normalization_center=normalization_center,
        normalization_scale=normalization_scale,
        edge_len_normalizer=edge_len_normalizer,
        edge_len_aug=edge_len_aug,
        edge_len_norm=edge_len_norm,
    )


def normalize_remove_lmk(remove_lmk: object, *, landmark_count: int) -> tuple[int, ...]:
    """Normalize a configured landmark-removal value into sorted 0-based indices."""

    if landmark_count <= 0:
        raise ValueError(f"landmark_count must be positive, got {landmark_count}.")
    if remove_lmk is None:
        return ()
    if _is_integer(remove_lmk):
        values = (int(remove_lmk),)
    elif isinstance(remove_lmk, (list, tuple, np.ndarray)):
        values = tuple(remove_lmk)
    else:
        raise ValueError("remove_lmk must be None, an integer index, or a list of integer indices.")

    if len(values) == 0:
        return ()

    normalized = []
    for value in values:
        if not _is_integer(value):
            raise ValueError(f"remove_lmk entries must be integer indices, got {value!r}.")
        index = int(value)
        if index < 0 or index >= landmark_count:
            raise ValueError(
                f"remove_lmk index {index} is out of range for {landmark_count} landmarks "
                f"(valid indices are 0..{landmark_count - 1})."
            )
        normalized.append(index)

    if len(set(normalized)) != len(normalized):
        raise ValueError("remove_lmk must not contain duplicate landmark indices.")
    if len(normalized) >= landmark_count:
        raise ValueError("remove_lmk must leave at least one landmark.")
    return tuple(sorted(normalized))


def _is_integer(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _require_keys(h5: h5py.File, keys: list[str]) -> None:
    missing = [key for key in keys if key not in h5]
    if missing:
        raise ValueError(f"HDF5 file is missing required augmented tree keys: {missing}")


def _read_utf8_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_utf8_array(values: np.ndarray) -> tuple[str, ...]:
    return tuple(_read_utf8_scalar(value) for value in values)


def _hyperiax_edge_lens(tree: Tree) -> np.ndarray:
    if "edge_len" in tree.schema.names:
        return np.asarray(tree["edge_len"], dtype=np.float64)
    return np.asarray(tree["edge_length"], dtype=np.float64)


def _root_to_node_depths(parents, edge_lens: np.ndarray) -> np.ndarray:
    parents = np.asarray(parents)
    edge_lens = np.asarray(edge_lens, dtype=np.float64)
    depths = np.zeros_like(edge_lens, dtype=np.float64)
    for index in range(1, len(edge_lens)):
        depths[index] = depths[parents[index]] + edge_lens[index]
    return depths


def _validate_augmented_node_alignment(
    *,
    parsed_tree: Tree,
    node_names: tuple[str, ...],
    node_types: tuple[str, ...],
    coords: np.ndarray,
    edge_len_ori: np.ndarray,
    edge_len_aug: np.ndarray,
    edge_len_norm: np.ndarray,
    edge_len_normalizer: float,
    is_hidden: np.ndarray,
    h5_is_leaf: np.ndarray,
    h5_is_root: np.ndarray,
) -> None:
    if parsed_tree.size != len(node_names):
        raise ValueError(
            "HDF5 node count does not match augmented Newick BFS order: "
            f"{len(node_names)} HDF5 nodes vs {parsed_tree.size} tree nodes."
        )
    if len(node_types) != parsed_tree.size:
        raise ValueError(f"{schema.NODES_NODE_TYPE} length does not match augmented Newick node count.")
    if coords.shape[:1] != (parsed_tree.size,):
        raise ValueError(
            f"{schema.NODES_COORDS} leading axis {coords.shape[:1]} does not match tree size {parsed_tree.size}."
        )
    for name, edge_len in [
        (schema.NODES_EDGE_LEN_ORI, edge_len_ori),
        (schema.NODES_EDGE_LEN_AUG, edge_len_aug),
        (schema.NODES_EDGE_LEN_NORM, edge_len_norm),
    ]:
        if edge_len.shape != (parsed_tree.size,):
            raise ValueError(f"{name} shape {edge_len.shape} does not match tree size {parsed_tree.size}.")
        if not np.isfinite(edge_len).all():
            raise ValueError(f"{name} must be finite.")
    if edge_len_normalizer <= 0 or not np.isfinite(edge_len_normalizer):
        raise ValueError(f"{schema.PREPROC_EDGE_LEN_NORMALIZER} must be a finite positive scalar.")
    parsed_edge_len_aug = _hyperiax_edge_lens(parsed_tree)
    if not np.allclose(edge_len_aug, parsed_edge_len_aug):
        raise ValueError(f"{schema.NODES_EDGE_LEN_AUG} does not match augmented Newick BFS edge lengths.")
    if not np.allclose(edge_len_norm, edge_len_aug / edge_len_normalizer):
        raise ValueError(
            f"{schema.NODES_EDGE_LEN_NORM} must equal {schema.NODES_EDGE_LEN_AUG} "
            f"divided by {schema.PREPROC_EDGE_LEN_NORMALIZER}."
        )
    normalized_depths = _root_to_node_depths(parsed_tree.topology.parents, edge_len_norm)
    if normalized_depths.max() > 1.0 + 1e-6:
        raise ValueError("Normalized root-to-node depths must not exceed 1.")
    for name, mask in [
        (schema.NODES_IS_HIDDEN, is_hidden),
        (schema.NODES_IS_LEAF, h5_is_leaf),
        (schema.NODES_IS_ROOT, h5_is_root),
    ]:
        if mask.shape != (parsed_tree.size,):
            raise ValueError(f"{name} shape {mask.shape} does not match tree size {parsed_tree.size}.")

    if not np.array_equal(h5_is_leaf, parsed_tree.topology.is_leaf):
        raise ValueError(f"{schema.NODES_IS_LEAF} does not match augmented Newick BFS order.")
    if not np.array_equal(h5_is_root, parsed_tree.topology.is_root):
        raise ValueError(f"{schema.NODES_IS_ROOT} does not match augmented Newick BFS order.")
    if node_names[0] != schema.SUPER_ROOT_NAME or node_types[0] != schema.SUPER_ROOT_NAME:
        raise ValueError("Expected node 0 to be the augmented super_root.")
    if not h5_is_root[0] or parsed_tree.topology.parents[0] != 0:
        raise ValueError("Expected node 0 to be the hyperiax root.")

    observed = np.isfinite(coords).all(axis=(1, 2))
    if not np.array_equal(observed, ~is_hidden):
        raise ValueError("Finite coordinate rows must be exactly the non-hidden nodes.")

    parsed_names = parsed_tree.topology.names or ("",) * parsed_tree.size
    for index, (parsed_name, h5_name, node_type) in enumerate(
        zip(parsed_names, node_names, node_types, strict=True)
    ):
        if parsed_name:
            if parsed_name != h5_name:
                raise ValueError(
                    "HDF5 node order does not match augmented Newick BFS order "
                    f"at node {index}: HDF5 has {h5_name!r}, Newick has {parsed_name!r}."
                )
            continue

        if node_type not in {"hidden", "phylo_root"}:
            raise ValueError(
                "HDF5 node order does not match augmented Newick BFS order "
                f"at node {index}: unnamed Newick internal node has node_type {node_type!r}."
            )
