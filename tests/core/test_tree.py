from pathlib import Path

import h5py
import numpy as np
from hyperiax import from_newick

from src import data_schema as schema
from src.loader import load_augmented_butterfly_tree
from src.tree_state import build_shape_state


REPO_ROOT = Path(__file__).resolve().parents[2]
BUTTERFLIES_H5 = REPO_ROOT / "data/butterflies/data.h5"


def test_load_augmented_butterfly_tree_builds_node_aligned_hyperiax_tree():
    dataset = load_augmented_butterfly_tree(BUTTERFLIES_H5)

    assert dataset.tree.size == 850
    assert dataset.node_names[0] == "super_root"
    assert dataset.node_names[1] == ""
    assert dataset.node_types[0] == "super_root"
    assert dataset.node_types[1] == "phylo_root"
    assert all(
        name == ""
        for name, node_type in zip(dataset.node_names, dataset.node_types)
        if node_type in {"phylo_root", "hidden"}
    )
    assert dataset.tree.topology.names == dataset.node_names
    assert dataset.tree.schema.names == ("coords", "edge_len", "is_hidden", "is_observed")

    coords = np.asarray(dataset.tree["coords"])
    is_hidden = np.asarray(dataset.tree["is_hidden"]).astype(bool)
    is_observed = np.asarray(dataset.tree["is_observed"]).astype(bool)

    assert coords.shape == (850, 118, 2)
    assert np.array_equal(is_observed, ~is_hidden)
    assert np.isnan(coords[is_hidden]).all()
    assert np.isfinite(coords[is_observed]).all()
    assert dataset.tree.topology.parents[0] == 0
    assert dataset.tree.topology.parents[1] == 0
    assert dataset.tree.topology.is_leaf.sum() == 425


def test_loader_uses_hdf5_edge_len_as_tree_authority():
    dataset = load_augmented_butterfly_tree(BUTTERFLIES_H5)

    with h5py.File(BUTTERFLIES_H5, "r") as h5:
        newick_aug = h5[schema.TREE_NEWICK_AUG][()].decode("utf-8")
        edge_len_aug = h5[schema.NODES_EDGE_LEN_AUG][:]
        edge_len_norm = h5[schema.NODES_EDGE_LEN_NORM][:]
        edge_len_normalizer = h5[schema.PREPROC_EDGE_LEN_NORMALIZER][()]

    np.testing.assert_allclose(
        np.asarray(dataset.tree["edge_len"]),
        edge_len_norm,
    )
    np.testing.assert_allclose(dataset.edge_len_aug, edge_len_aug)
    np.testing.assert_allclose(dataset.edge_len_norm, edge_len_norm)
    parsed_tree = from_newick(newick_aug)
    np.testing.assert_allclose(edge_len_aug, np.asarray(parsed_tree["edge_length"]))
    np.testing.assert_allclose(edge_len_norm, edge_len_aug / edge_len_normalizer)

    normalized_depths = _root_to_node_depths(dataset.tree.topology.parents, np.asarray(dataset.tree["edge_len"]))
    assert normalized_depths.max() <= 1.0 + 1e-6
    np.testing.assert_allclose(normalized_depths.max(), 1.0, rtol=1e-6, atol=1e-6)


def test_loader_validates_hdf5_node_order_against_augmented_tree(tmp_path):
    broken_h5 = tmp_path / "broken.h5"
    with h5py.File(BUTTERFLIES_H5, "r") as source, h5py.File(broken_h5, "w") as target:
        for key, value in source.attrs.items():
            target.attrs[key] = value
        source.copy("nodes", target)
        source.copy("tree", target)
        source.copy("preproc", target)

    with h5py.File(broken_h5, "r+") as h5:
        names = h5[schema.NODES_NAMES][:]
        leaf_indices = np.flatnonzero(h5[schema.NODES_IS_LEAF][:].astype(bool))
        names[leaf_indices[0]], names[leaf_indices[1]] = names[leaf_indices[1]], names[leaf_indices[0]]
        h5[schema.NODES_NAMES][:] = names

    try:
        load_augmented_butterfly_tree(broken_h5)
    except ValueError as error:
        assert "does not match augmented Newick BFS order" in str(error)
    else:
        raise AssertionError("Expected loader to reject misaligned node order")


def test_shape_state_centralizes_root_leaf_and_edge_arrays():
    dataset = load_augmented_butterfly_tree(BUTTERFLIES_H5)

    state = build_shape_state(dataset)

    assert state.node_count == 850
    assert state.n_landmarks == 118
    assert state.d_landmarks == 2
    assert state.state_dim == 236
    assert state.root_index == 0
    assert state.leaf_indices.shape == (425,)
    assert state.root_value.shape == (236,)
    assert state.leaf_observations.shape == (425, 236)
    assert np.isfinite(state.root_value).all()
    assert np.isfinite(state.leaf_observations).all()
    assert np.array_equal(state.is_observed, ~state.is_hidden)
    np.testing.assert_allclose(state.edge_len, dataset.edge_len_norm)


def _root_to_node_depths(parents, edge_lens):
    parents = np.asarray(parents)
    edge_lens = np.asarray(edge_lens, dtype=np.float64)
    depths = np.zeros_like(edge_lens, dtype=np.float64)
    for index in range(1, len(edge_lens)):
        depths[index] = depths[parents[index]] + edge_lens[index]
    return depths
