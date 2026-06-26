from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from ete3 import Tree
from hyperiax import from_newick

from src import data_schema as schema
from scripts import build_data
from scripts.build_data import convert_shape_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
BUTTERFLIES_CSV = REPO_ROOT / "data/butterflies/lmks.csv"
BUTTERFLIES_TREE = REPO_ROOT / "data/butterflies/tree.nwk"
BEAKS_CSV = REPO_ROOT / "data/beaks/lmks.csv"
BEAKS_TREE = REPO_ROOT / "data/beaks/tree.nwk"

SHAPE_DATASET_CASES = (
    (
        "butterflies",
        BUTTERFLIES_CSV,
        BUTTERFLIES_TREE,
        850,
        425,
        118,
        2,
        146.57659912109375,
    ),
    (
        "beaks",
        BEAKS_CSV,
        BEAKS_TREE,
        696,
        348,
        79,
        3,
        None,
    ),
)


def test_shape_hdf5_schema_contract_names_current_paths():
    assert schema.LANDMARK_CSV_FILENAME == "lmks.csv"
    assert schema.TREE_NEWICK_FILENAME == "tree.nwk"
    assert schema.HDF5_FILENAME == "data.h5"
    assert schema.TREE_PLOT_FILENAME == "tree.png"
    assert schema.LEAVES_PLOT_FILENAME == "leaves.png"
    assert schema.NORMALIZATION_METHOD == "global_center_uniform_scale_to_unit_box"
    assert schema.NODE_ORDER == "augmented_super_root_plus_newick_bfs_order"
    assert schema.REQUIRED_DATASET_PATHS == (
        schema.NODES_COORDS,
        schema.NODES_EDGE_LEN_AUG,
        schema.NODES_EDGE_LEN_NORM,
        schema.NODES_EDGE_LEN_ORI,
        schema.NODES_IS_HIDDEN,
        schema.NODES_IS_LEAF,
        schema.NODES_IS_ROOT,
        schema.NODES_NAMES,
        schema.NODES_NODE_TYPE,
        schema.PREPROC_CENTER,
        schema.PREPROC_EDGE_LEN_NORMALIZER,
        schema.PREPROC_SCALE,
        schema.TREE_LEAF_NAMES,
        schema.TREE_NEWICK_AUG,
    )
    assert "csv_tree_consistency" in schema.FORBIDDEN_ROOT_PATHS
    assert "parent_index" in schema.FORBIDDEN_TREE_PATHS


@pytest.mark.parametrize(
    (
        "dataset_name",
        "csv_path",
        "tree_path",
        "expected_node_count",
        "expected_leaf_count",
        "expected_landmark_count",
        "expected_dims",
        "expected_scale",
    ),
    SHAPE_DATASET_CASES,
)
def test_convert_shape_dataset_matrix_preserves_hdf5_invariants(
    tmp_path,
    dataset_name,
    csv_path,
    tree_path,
    expected_node_count,
    expected_leaf_count,
    expected_landmark_count,
    expected_dims,
    expected_scale,
):
    output_path = tmp_path / f"{dataset_name}.h5"

    convert_shape_dataset(csv_path, tree_path, output_path)

    with h5py.File(output_path, "r") as h5:
        names = _decode_h5_strings(h5[schema.NODES_NAMES][:])
        coords = h5[schema.NODES_COORDS][:]
        is_leaf = h5[schema.NODES_IS_LEAF][:].astype(bool)
        is_hidden = h5[schema.NODES_IS_HIDDEN][:].astype(bool)
        is_root = h5[schema.NODES_IS_ROOT][:].astype(bool)
        scale = h5[schema.PREPROC_SCALE][()]
        leaf_names = _decode_h5_strings(h5[schema.TREE_LEAF_NAMES][:])

    leaf_coords = coords[is_leaf]

    assert names[0] == schema.SUPER_ROOT_NAME
    assert len(names) == expected_node_count
    assert coords.shape == (expected_node_count, expected_landmark_count, expected_dims)
    assert len(leaf_names) == expected_leaf_count
    assert is_leaf.sum() == expected_leaf_count
    assert is_hidden.sum() == expected_node_count - expected_leaf_count - 1
    assert is_root.tolist() == [True] + [False] * (expected_node_count - 1)
    assert np.isfinite(coords[0]).all()
    assert np.isnan(coords[is_hidden]).all()
    assert np.isfinite(leaf_coords).all()
    assert leaf_coords.min() >= -1.000001
    assert leaf_coords.max() <= 1.000001
    np.testing.assert_allclose(leaf_coords.min(), -1.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(leaf_coords.max(), 1.0, rtol=1e-6, atol=1e-6)

    expected_center, computed_scale = _expected_normalization(csv_path)
    del expected_center
    if expected_scale is None:
        expected_scale = computed_scale
    np.testing.assert_allclose(scale, expected_scale, rtol=1e-6, atol=1e-6)


def test_convert_butterfly_dataset_writes_expected_hdf5_schema(tmp_path):
    csv_path = BUTTERFLIES_CSV
    tree_path = BUTTERFLIES_TREE
    output_path = tmp_path / "data.h5"

    convert_shape_dataset(csv_path, tree_path, output_path)

    with h5py.File(output_path, "r") as h5:
        for path in schema.FORBIDDEN_ROOT_PATHS:
            assert path not in h5
        assert h5[schema.NODES_NAMES].shape == (850,)
        assert h5[schema.NODES_NODE_TYPE].shape == (850,)
        assert h5[schema.NODES_IS_LEAF].shape == (850,)
        assert h5[schema.NODES_IS_HIDDEN].shape == (850,)
        assert h5[schema.NODES_IS_ROOT].shape == (850,)
        assert h5[schema.NODES_EDGE_LEN_ORI].shape == (850,)
        assert h5[schema.NODES_EDGE_LEN_AUG].shape == (850,)
        assert h5[schema.NODES_EDGE_LEN_NORM].shape == (850,)
        for path in schema.FORBIDDEN_NODES_PATHS:
            assert path not in h5[schema.NODES_GROUP]
        assert h5[schema.NODES_COORDS].shape == (850, 118, 2)
        assert h5[schema.NODES_COORDS].dtype == np.dtype("float32")
        assert h5[schema.TREE_NEWICK_ORI][()].decode("utf-8").endswith(";")
        assert h5[schema.TREE_NEWICK_AUG][()].decode("utf-8").endswith(";")
        assert h5[schema.TREE_LEAF_NAMES].shape == (425,)
        assert h5[schema.PREPROC_CENTER].shape == (2,)
        assert h5[schema.PREPROC_SCALE].shape == ()
        assert h5[schema.PREPROC_EDGE_LEN_NORMALIZER].shape == ()
        for path in schema.FORBIDDEN_TREE_PATHS:
            assert path not in h5[schema.TREE_GROUP]
        assert h5.attrs[schema.ATTR_NODE_ORDER] == schema.NODE_ORDER
        assert h5.attrs[schema.ATTR_SOURCE_CSV] == str(csv_path)
        assert h5.attrs[schema.ATTR_SOURCE_TREE] == str(tree_path)
        assert h5.attrs[schema.ATTR_NORMALIZATION_METHOD] == schema.NORMALIZATION_METHOD
        assert h5.attrs[schema.ATTR_LANDMARK_COUNT] == 118
        assert h5.attrs[schema.ATTR_COORDINATE_DIMENSIONS] == 2


def test_normalization_uses_global_uniform_scale_and_preserves_raw_leaf_coords(tmp_path):
    csv_path = BUTTERFLIES_CSV
    tree_path = BUTTERFLIES_TREE
    output_path = tmp_path / "data.h5"

    convert_shape_dataset(csv_path, tree_path, output_path)

    with h5py.File(output_path, "r") as h5:
        coords = h5[schema.NODES_COORDS][:]
        names = _decode_h5_strings(h5[schema.NODES_NAMES][:])
        is_leaf = h5[schema.NODES_IS_LEAF][:].astype(bool)
        is_hidden = h5[schema.NODES_IS_HIDDEN][:].astype(bool)
        center = h5[schema.PREPROC_CENTER][:]
        scale = h5[schema.PREPROC_SCALE][()]

    raw_by_name, _, _ = _read_raw_shape_csv(csv_path)
    raw_leaf_coords = np.stack([raw_by_name[name] for name, leaf in zip(names, is_leaf) if leaf])
    leaf_coords = coords[is_leaf]
    expected_center, expected_scale = _expected_normalization(csv_path)

    np.testing.assert_allclose(center, expected_center, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(scale, expected_scale, rtol=1e-6, atol=1e-6)
    assert np.all(leaf_coords >= -1.000001)
    assert np.all(leaf_coords <= 1.000001)
    np.testing.assert_allclose(leaf_coords, (raw_leaf_coords - center) / scale, rtol=1e-6, atol=1e-6)
    assert np.isnan(coords[is_hidden]).all()

    normalized_range = leaf_coords.max(axis=(0, 1)) - leaf_coords.min(axis=(0, 1))
    raw_leaf_range = np.ptp(raw_leaf_coords, axis=(0, 1))
    np.testing.assert_allclose(
        normalized_range[0] / normalized_range[1],
        raw_leaf_range[0] / raw_leaf_range[1],
        rtol=1e-6,
    )


def test_augmented_nodes_are_derived_from_newick_bfs_and_leaf_csv_join(tmp_path):
    csv_path = BUTTERFLIES_CSV
    tree_path = BUTTERFLIES_TREE
    output_path = tmp_path / "data.h5"

    convert_shape_dataset(csv_path, tree_path, output_path)

    with h5py.File(output_path, "r") as h5:
        names = _decode_h5_strings(h5[schema.NODES_NAMES][:])
        node_type = _decode_h5_strings(h5[schema.NODES_NODE_TYPE][:])
        is_leaf = h5[schema.NODES_IS_LEAF][:].astype(bool)
        is_hidden = h5[schema.NODES_IS_HIDDEN][:].astype(bool)
        is_root = h5[schema.NODES_IS_ROOT][:].astype(bool)
        coords = h5[schema.NODES_COORDS][:]
        leaf_names = _decode_h5_strings(h5[schema.TREE_LEAF_NAMES][:])
        edge_len_ori = h5[schema.NODES_EDGE_LEN_ORI][:]
        edge_len_aug = h5[schema.NODES_EDGE_LEN_AUG][:]
        edge_len_norm = h5[schema.NODES_EDGE_LEN_NORM][:]
        edge_len_normalizer = h5[schema.PREPROC_EDGE_LEN_NORMALIZER][()]
        newick_ori = h5[schema.TREE_NEWICK_ORI][()].decode("utf-8")
        newick_aug = h5[schema.TREE_NEWICK_AUG][()].decode("utf-8")

    original_tree = from_newick(newick_ori)
    augmented_hx_tree = from_newick(newick_aug)
    original_newick_edge_lens = np.asarray(original_tree["edge_length"], dtype=np.float64)
    augmented_newick_edge_lens = np.asarray(augmented_hx_tree["edge_length"], dtype=np.float64)
    parsed_augmented_names = tuple(augmented_hx_tree.topology.names)

    assert tuple(names) == parsed_augmented_names
    assert names[0] == schema.SUPER_ROOT_NAME
    assert names[1] == ""
    assert all(name == "" for name, leaf, root in zip(names, is_leaf, is_root) if not leaf and not root)
    assert node_type[0] == schema.SUPER_ROOT_NAME
    assert node_type[1] == "phylo_root"
    assert all(kind == "leaf" for kind, leaf in zip(node_type, is_leaf) if leaf)
    assert sorted(name for name, leaf in zip(names, is_leaf) if leaf) == sorted(leaf_names)

    assert is_root.tolist() == [True] + [False] * (len(names) - 1)
    assert not is_leaf[0]
    assert is_hidden[1]
    assert is_hidden.sum() == len(names) - len(leaf_names) - 1
    assert not is_hidden[0]
    assert not np.isnan(coords[0]).any()
    assert np.isnan(coords[is_hidden]).all()

    np.testing.assert_allclose(edge_len_ori, np.concatenate(([0.0], original_newick_edge_lens)))
    np.testing.assert_allclose(edge_len_aug, augmented_newick_edge_lens)
    np.testing.assert_allclose(edge_len_aug[0], 0.0)
    np.testing.assert_allclose(edge_len_aug[1], original_newick_edge_lens.mean())

    assert newick_ori == tree_path.read_text().strip()
    augmented_tree = Tree(newick_aug, format=1)
    assert augmented_tree.name == schema.SUPER_ROOT_NAME
    assert len(augmented_tree.children) == 1
    assert augmented_tree.children[0].name == ""
    assert sorted(leaf.name for leaf in augmented_tree.iter_leaves()) == sorted(leaf_names)
    np.testing.assert_allclose(augmented_tree.children[0].dist, original_newick_edge_lens.mean())

    expected_normalizer = _max_root_to_node_depth(
        augmented_hx_tree.topology.parents,
        edge_len_aug,
    )
    np.testing.assert_allclose(edge_len_normalizer, expected_normalizer)
    np.testing.assert_allclose(edge_len_norm, edge_len_aug / edge_len_normalizer)
    normalized_depths = _root_to_node_depths(augmented_hx_tree.topology.parents, edge_len_norm)
    assert normalized_depths.max() <= 1.0 + 1e-6
    np.testing.assert_allclose(normalized_depths.max(), 1.0, rtol=1e-6, atol=1e-6)


def test_super_root_coordinates_are_leaf_euclidean_mean(tmp_path):
    csv_path = BUTTERFLIES_CSV
    tree_path = BUTTERFLIES_TREE
    output_path = tmp_path / "data.h5"

    convert_shape_dataset(csv_path, tree_path, output_path)

    with h5py.File(output_path, "r") as h5:
        coords = h5[schema.NODES_COORDS][:]
        is_leaf = h5[schema.NODES_IS_LEAF][:].astype(bool)

    np.testing.assert_allclose(coords[0], coords[is_leaf].mean(axis=0), rtol=1e-6, atol=1e-6)


def test_converter_rejects_malformed_landmark_coordinate_columns(tmp_path):
    csv_path = tmp_path / "bad_lmks.csv"
    tree_path = tmp_path / "tree.nwk"
    output_path = tmp_path / "bad.h5"
    tree_path.write_text("(leaf_a:1,leaf_b:1);", encoding="utf-8")
    csv_path.write_text(
        "\n".join(
            [
                "name,1.X,1.Y,2.Y",
                "leaf_a,0,0,1",
                "leaf_b,0,1,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        convert_shape_dataset(csv_path, tree_path, output_path)
    except ValueError as error:
        assert "coordinate columns must be grouped as 1.X, 1.Y" in str(error)
    else:
        raise AssertionError("Expected converter to reject malformed landmark coordinate columns")


def test_converter_rejects_csv_tree_leaf_mismatches(tmp_path):
    csv_path = tmp_path / "bad_lmks.csv"
    tree_path = tmp_path / "tree.nwk"
    output_path = tmp_path / "bad.h5"
    tree_path.write_text("(leaf_a:1,leaf_b:1);", encoding="utf-8")
    csv_path.write_text(
        "\n".join(
            [
                "name,1.X,1.Y",
                "leaf_a,0,0",
                "leaf_c,1,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="CSV leaf names must match Newick leaves"):
        convert_shape_dataset(csv_path, tree_path, output_path)


def test_build_data_cli_uses_kebab_case_paths_and_default_output_name(tmp_path):
    csv_path = BUTTERFLIES_CSV
    tree_path = BUTTERFLIES_TREE
    output_dir = tmp_path / "generated"
    output_path = output_dir / "data.h5"

    assert build_data.main(
        [
            "--csv-path",
            str(csv_path),
            "--tree-path",
            str(tree_path),
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    with h5py.File(output_path, "r") as h5:
        assert h5[schema.NODES_COORDS].shape == (850, 118, 2)


def test_build_data_cli_defaults_output_dir_to_csv_parent_and_accepts_output_name(tmp_path):
    csv_path = BUTTERFLIES_CSV
    tree_path = BUTTERFLIES_TREE
    output_path = csv_path.parent / "temporary-test-output.h5"

    try:
        assert build_data.main(
            [
                "--csv-path",
                str(csv_path),
                "--tree-path",
                str(tree_path),
                "--output-name",
                "temporary-test-output",
            ]
        ) == 0

        with h5py.File(output_path, "r") as h5:
            assert h5[schema.NODES_COORDS].shape == (850, 118, 2)
    finally:
        output_path.unlink(missing_ok=True)


def test_build_data_cli_exposes_only_kebab_case_options():
    parser = build_data._build_parser()
    option_strings = set(parser._option_string_actions)

    assert {"--csv-path", "--tree-path", "--output-dir", "--output-name"} <= option_strings
    assert "--csv_path" not in option_strings
    assert "--tree_path" not in option_strings
    assert "--output_path" not in option_strings
    assert [action for action in parser._actions if not action.option_strings] == []


def _decode_h5_strings(values):
    return [value.decode("utf-8") if isinstance(value, bytes) else value for value in values]


def _read_raw_shape_csv(csv_path: Path):
    frame = pd.read_csv(csv_path)
    coordinate_columns = [column for column in frame.columns if column != "name"]
    dims = _coordinate_dimensions(coordinate_columns)
    coords = frame[coordinate_columns].to_numpy(dtype=np.float32).reshape(len(frame), -1, dims)
    names = frame["name"].astype(str).tolist()
    return dict(zip(names, coords, strict=True)), tuple(names), dims


def _coordinate_dimensions(columns):
    first_landmark_axes = [column.split(".", 1)[1] for column in columns if column.startswith("1.")]
    return len(first_landmark_axes)


def _expected_normalization(csv_path: Path):
    raw_by_name, names, _ = _read_raw_shape_csv(csv_path)
    raw_coords = np.stack([raw_by_name[name] for name in names])
    mins = raw_coords.min(axis=(0, 1)).astype(np.float64)
    maxs = raw_coords.max(axis=(0, 1)).astype(np.float64)
    center = (mins + maxs) / 2.0
    scale = float(((maxs - mins) / 2.0).max())
    return center, scale


def _root_to_node_depths(parents, edge_lens):
    parents = np.asarray(parents)
    edge_lens = np.asarray(edge_lens, dtype=np.float64)
    depths = np.zeros_like(edge_lens, dtype=np.float64)
    for index in range(1, len(edge_lens)):
        depths[index] = depths[parents[index]] + edge_lens[index]
    return depths


def _max_root_to_node_depth(parents, edge_lens):
    return float(_root_to_node_depths(parents, edge_lens).max())
