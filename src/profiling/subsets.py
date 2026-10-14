"""Nested subsets preserving the existing coordinate and branch length scales."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import h5py
import numpy as np
from ete3 import Tree

from src import data_schema as s


def _strings(values) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def landmark_order(mean_shape: np.ndarray) -> list[int]:
    """Farthest-point order; deterministic first point and index tie breaks."""
    chosen = [int(np.argmax(np.sum((mean_shape - mean_shape.mean(0)) ** 2, axis=1)))]
    distances = np.full(len(mean_shape), np.inf)
    while len(chosen) < len(mean_shape):
        distances = np.minimum(distances, np.sum((mean_shape - mean_shape[chosen[-1]]) ** 2, axis=1))
        distances[chosen] = -np.inf
        chosen.append(int(np.argmax(distances)))
    return chosen


def _prune(node, selected: set[str], *, keep: bool = False):
    if node.is_leaf():
        return node if node.name in selected else None
    retained = []
    for child in list(node.children):
        node.remove_child(child)
        child = _prune(child, selected)
        if child is not None:
            retained.append(child)
    for child in retained:
        node.add_child(child)
    if not retained:
        return None
    if len(retained) == 1 and not keep:
        child = retained[0]
        node.remove_child(child)
        child.dist += node.dist
        return child
    return node


def describe_dataset(path: str | Path) -> dict:
    with h5py.File(path, "r") as h5:
        tree = Tree(h5[s.TREE_NEWICK_AUG][()].decode(), format=1)
        shape = h5[s.NODES_COORDS].shape
        leaves = _strings(h5[s.TREE_LEAF_NAMES][:])
    widths = []
    queue = deque([(tree, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth == len(widths):
            widths.append(0)
        widths[depth] += 1
        queue.extend((child, depth + 1) for child in node.children)
    return {"nodes": shape[0], "tips": len(leaves), "landmarks": shape[1], "dimensions": shape[2],
            "depth_edges": len(widths) - 1, "level_widths": widths, "max_level_width": max(widths)}


def build_subset(source: str | Path, output: Path, *, tips: int, landmarks: int, seed: int) -> dict:
    source = Path(source)
    with h5py.File(source, "r") as h5:
        arrays = {key: h5[key][()] for key in s.REQUIRED_DATASET_PATHS}
        attrs = dict(h5.attrs)
    coords = arrays[s.NODES_COORDS]
    leaf_mask = arrays[s.NODES_IS_LEAF].astype(bool)
    names = _strings(arrays[s.NODES_NAMES])
    leaf_names = sorted(np.asarray(names)[leaf_mask].tolist())
    if not 2 <= tips <= len(leaf_names) or not 2 <= landmarks <= coords.shape[1]:
        raise ValueError(f"Subset ({tips} tips, {landmarks} landmarks) exceeds {source}.")
    selected_names = sorted(np.random.default_rng(seed).permutation(leaf_names)[:tips].tolist())
    selected_indices = sorted(landmark_order(coords[leaf_mask].astype(np.float64).mean(0))[:landmarks])
    identity = tips == len(leaf_names) and landmarks == coords.shape[1]
    metadata = {"source": str(source), "source_size_bytes": source.stat().st_size,
                "seed": seed, "leaf_names": selected_names, "landmark_indices": selected_indices}
    if identity:
        return dict(metadata, h5=str(source), **describe_dataset(source))

    augmented = Tree(arrays[s.TREE_NEWICK_AUG].decode(), format=1)
    root = augmented.children[0]
    _prune(root, set(selected_names), keep=True)
    # Keep the original unary super-root and the original phylogenetic root.
    nodes = list(augmented.traverse("levelorder"))
    obs_by_name = {name: value[selected_indices] for name, value, leaf in zip(names, coords, leaf_mask) if leaf}
    new_coords = np.full((len(nodes), landmarks, coords.shape[2]), np.nan, dtype=np.float32)
    node_names, node_types = [], []
    for index, node in enumerate(nodes):
        if index == 0:
            node_names.append(s.SUPER_ROOT_NAME)
            node_types.append(s.SUPER_ROOT_NAME)
        elif node.is_leaf():
            node_names.append(node.name)
            node_types.append("leaf")
            new_coords[index] = obs_by_name[node.name]
        else:
            node_names.append(f"profiling_internal_{index}")
            node_types.append("phylo_root" if index == 1 else "hidden")
    new_leaf = np.array([node.is_leaf() for node in nodes])
    new_root = np.arange(len(nodes)) == 0
    new_coords[0] = new_coords[new_leaf].mean(0)
    edge_aug = np.array([0.0] + [node.dist for node in nodes[1:]], dtype=np.float64)
    edge_ori = edge_aug.copy()
    edge_ori[1] = 0.0
    values = {
        s.NODES_COORDS: new_coords, s.NODES_EDGE_LEN_AUG: edge_aug, s.NODES_EDGE_LEN_ORI: edge_ori,
        s.NODES_EDGE_LEN_NORM: edge_aug / arrays[s.PREPROC_EDGE_LEN_NORMALIZER],
        s.NODES_IS_HIDDEN: ~(new_leaf | new_root), s.NODES_IS_LEAF: new_leaf, s.NODES_IS_ROOT: new_root,
        s.NODES_NAMES: node_names, s.NODES_NODE_TYPE: node_types,
        s.TREE_NEWICK_AUG: augmented.write(format=1, format_root_node=True, dist_formatter="%.17g"),
        s.TREE_NEWICK_ORI: root.write(format=1, dist_formatter="%.17g"),
        s.TREE_LEAF_NAMES: [node.name for node in nodes if node.is_leaf()],
        s.PREPROC_CENTER: arrays[s.PREPROC_CENTER], s.PREPROC_SCALE: arrays[s.PREPROC_SCALE],
        s.PREPROC_EDGE_LEN_NORMALIZER: arrays[s.PREPROC_EDGE_LEN_NORMALIZER],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "x") as h5:
        for key, value in attrs.items():
            h5.attrs[key] = value
        h5.attrs[s.ATTR_LANDMARK_COUNT] = landmarks
        h5.attrs["profiling_source"] = str(source)
        for key, value in values.items():
            if isinstance(value, str) or isinstance(value, list):
                h5.create_dataset(key, data=value, dtype=h5py.string_dtype())
            else:
                h5.create_dataset(key, data=value)
    return dict(metadata, h5=str(output), **describe_dataset(output))
