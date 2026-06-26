"""Build a node-aligned shape `data.h5` file from landmark CSV and Newick inputs.

Example:
    uv run python -m scripts.build_data --csv-path data/butterflies/lmks.csv --tree-path data/butterflies/tree.nwk
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
from ete3 import Tree
from hyperiax import from_newick

from src import data_schema as schema

NAME_COLUMN = "name"
AXES_BY_DIMENSION = {
    2: ("X", "Y"),
    3: ("X", "Y", "Z"),
}
COORDINATE_COLUMN_PATTERN = re.compile(r"([1-9][0-9]*)\.([XYZ])")


@dataclass(frozen=True)
class LandmarkObservations:
    """Validated leaf landmark observations keyed by Newick leaf name."""

    names: tuple[str, ...]
    raw_coords: np.ndarray
    leaf_names: tuple[str, ...]
    coordinate_columns: tuple[str, ...]

    @property
    def landmark_count(self) -> int:
        return int(self.raw_coords.shape[1])

    @property
    def coordinate_dimensions(self) -> int:
        return int(self.raw_coords.shape[2])

    def raw_coords_by_name(self) -> dict[str, np.ndarray]:
        return dict(zip(self.names, self.raw_coords, strict=True))


@dataclass(frozen=True)
class CoordinateNormalization:
    """Global translation and uniform scale fitted on observed leaf rows."""

    center: np.ndarray
    scale: float


@dataclass(frozen=True)
class AugmentedNodeArrays:
    """HDF5 node-aligned arrays after adding the synthetic super_root."""

    names: np.ndarray
    node_type: np.ndarray
    is_leaf: np.ndarray
    is_hidden: np.ndarray
    is_root: np.ndarray
    coords: np.ndarray


@dataclass(frozen=True)
class AugmentedTreeBundle:
    """Original and augmented Newick plus node-aligned edge lengths."""

    newick_ori: str
    newick_aug: str
    edge_len_ori: np.ndarray
    edge_len_aug: np.ndarray
    edge_len_norm: np.ndarray
    edge_len_normalizer: float


def convert_shape_dataset(
    csv_path: str | Path,
    tree_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Convert leaf landmark observations and a Newick tree into HDF5.

    Coordinates are transformed with one global translation and one uniform
    scale fitted on observed leaf rows. Internal Newick nodes are latent states
    with NaN coordinate placeholders.
    """

    csv_path = Path(csv_path)
    tree_path = Path(tree_path)
    output_path = Path(output_path)

    newick = tree_path.read_text().strip()
    leaf_names = _leaf_names_from_newick(newick)
    _reject_named_internal_nodes(newick)
    observations = _read_landmark_observations(csv_path, leaf_names=leaf_names)
    normalization = _fit_unit_box_normalization(observations.raw_coords)
    tree_bundle = _build_augmented_tree_bundle(newick)
    nodes = _build_augmented_node_arrays(
        observations,
        tree_bundle=tree_bundle,
        normalization=normalization,
    )

    _write_hdf5(
        output_path,
        csv_path=csv_path,
        tree_path=tree_path,
        observations=observations,
        nodes=nodes,
        tree_bundle=tree_bundle,
        normalization=normalization,
    )

    return output_path


def _leaf_names_from_newick(newick: str) -> tuple[str, ...]:
    leaf_names = tuple(leaf.name for leaf in Tree(newick, format=1).iter_leaves())
    if not leaf_names:
        raise ValueError("Newick tree must contain at least one leaf.")
    if any(not name for name in leaf_names):
        raise ValueError("All Newick leaves must have names.")
    if len(set(leaf_names)) != len(leaf_names):
        raise ValueError("Newick leaf names must be unique.")
    return leaf_names


def _reject_named_internal_nodes(newick: str) -> None:
    tree = Tree(newick, format=1)
    named_internal_nodes = [node.name for node in tree.traverse() if not node.is_leaf() and node.name]
    if named_internal_nodes:
        raise ValueError(
            "Hidden/internal Newick nodes must not have names; "
            f"found {named_internal_nodes[:8]}."
        )


def _read_landmark_observations(csv_path: Path, *, leaf_names: tuple[str, ...]) -> LandmarkObservations:
    frame = pd.read_csv(csv_path)
    coordinate_columns, coordinate_dimensions = _landmark_coordinate_columns(frame)
    names = _csv_leaf_names(frame)
    _validate_csv_leaf_names(names, leaf_names)
    raw_coords = frame[list(coordinate_columns)].to_numpy(dtype=np.float32).reshape(
        len(frame),
        -1,
        coordinate_dimensions,
    )
    return LandmarkObservations(
        names=names,
        raw_coords=raw_coords,
        leaf_names=leaf_names,
        coordinate_columns=coordinate_columns,
    )


def _csv_leaf_names(frame: pd.DataFrame) -> tuple[str, ...]:
    if NAME_COLUMN not in frame.columns:
        raise ValueError("Landmark CSV is missing required column: 'name'.")
    names = tuple(frame[NAME_COLUMN].astype(str))
    if not names:
        raise ValueError("Landmark CSV must contain at least one observed leaf row.")
    if any(not name for name in names):
        raise ValueError("Landmark CSV leaf names must be non-empty.")
    if any(name.startswith("xx_") for name in names):
        raise ValueError("Landmark CSV must contain observed leaves only, not xx_* hidden rows.")
    if len(set(names)) != len(names):
        raise ValueError("Landmark CSV leaf names must be unique.")
    return names


def _validate_csv_leaf_names(names: tuple[str, ...], leaf_names: tuple[str, ...]) -> None:
    csv_name_set = set(names)
    leaf_name_set = set(leaf_names)
    missing_from_csv = sorted(leaf_name_set.difference(csv_name_set))
    extra_in_csv = sorted(csv_name_set.difference(leaf_name_set))
    if missing_from_csv or extra_in_csv:
        raise ValueError(
            "CSV leaf names must match Newick leaves exactly; "
            f"missing_from_csv={missing_from_csv[:8]}, extra_in_csv={extra_in_csv[:8]}."
        )


def _landmark_coordinate_columns(frame: pd.DataFrame) -> tuple[tuple[str, ...], int]:
    if NAME_COLUMN not in frame.columns:
        raise ValueError("Landmark CSV is missing required column: 'name'.")
    coordinate_columns = tuple(str(column) for column in frame.columns if column != NAME_COLUMN)
    if not coordinate_columns:
        raise ValueError("Landmark CSV must contain coordinate columns.")

    parsed_columns = []
    for column in coordinate_columns:
        match = COORDINATE_COLUMN_PATTERN.fullmatch(column)
        if match is None:
            raise ValueError(
                "Landmark coordinate columns must be grouped as 1.X, 1.Y[, 1.Z], "
                f"2.X, 2.Y[, 2.Z], ...; got {coordinate_columns[:8]}."
            )
        parsed_columns.append((int(match.group(1)), match.group(2)))

    first_landmark_axes = tuple(axis for index, axis in parsed_columns if index == 1)
    if first_landmark_axes == AXES_BY_DIMENSION[2]:
        coordinate_dimensions = 2
    elif first_landmark_axes == AXES_BY_DIMENSION[3]:
        coordinate_dimensions = 3
    else:
        raise ValueError(
            "Landmark coordinate columns must be grouped as 1.X, 1.Y[, 1.Z], "
            f"2.X, 2.Y[, 2.Z], ...; got {coordinate_columns[:8]}."
        )

    axes = AXES_BY_DIMENSION[coordinate_dimensions]
    if len(coordinate_columns) % coordinate_dimensions != 0:
        raise ValueError(
            "Landmark coordinate columns must be grouped as 1.X, 1.Y[, 1.Z], "
            f"2.X, 2.Y[, 2.Z], ...; got {coordinate_columns[:8]}."
        )
    landmark_count = len(coordinate_columns) // coordinate_dimensions
    expected_columns = tuple(
        f"{landmark_index}.{axis}"
        for landmark_index in range(1, landmark_count + 1)
        for axis in axes
    )
    if coordinate_columns != expected_columns:
        raise ValueError(
            "Landmark coordinate columns must be grouped as 1.X, 1.Y[, 1.Z], "
            f"2.X, 2.Y[, 2.Z], ...; got {coordinate_columns[:8]}."
        )
    return coordinate_columns, coordinate_dimensions


def _fit_unit_box_normalization(raw_coords: np.ndarray) -> CoordinateNormalization:
    mins = raw_coords.min(axis=(0, 1)).astype(np.float64)
    maxs = raw_coords.max(axis=(0, 1)).astype(np.float64)
    center = (mins + maxs) / 2.0
    half_ranges = (maxs - mins) / 2.0
    scale = float(half_ranges.max())
    if scale <= 0:
        raise ValueError("Cannot normalize degenerate coordinates with zero global range.")
    return CoordinateNormalization(center=center, scale=scale)


def _build_augmented_node_arrays(
    observations: LandmarkObservations,
    *,
    tree_bundle: AugmentedTreeBundle,
    normalization: CoordinateNormalization,
) -> AugmentedNodeArrays:
    augmented_tree = from_newick(tree_bundle.newick_aug)
    parsed_names = np.array(augmented_tree.topology.names, dtype=object)
    is_root = np.asarray(augmented_tree.topology.is_root, dtype=np.bool_)
    is_leaf = np.asarray(augmented_tree.topology.is_leaf, dtype=np.bool_)
    is_hidden = ~(is_root | is_leaf)

    augmented_shape = (
        augmented_tree.size,
        observations.landmark_count,
        observations.coordinate_dimensions,
    )
    coords = np.full(augmented_shape, np.nan, dtype=np.float32)
    scaled_by_name = {
        name: ((coords.astype(np.float64) - normalization.center) / normalization.scale).astype(np.float32)
        for name, coords in observations.raw_coords_by_name().items()
    }
    for index, name in enumerate(parsed_names):
        if is_leaf[index]:
            coords[index] = scaled_by_name[str(name)]
    coords[is_root] = coords[is_leaf].mean(axis=0)

    node_type = np.empty(augmented_tree.size, dtype=object)
    node_type[is_root] = schema.SUPER_ROOT_NAME
    node_type[is_hidden] = "hidden"
    node_type[is_leaf] = "leaf"
    node_type[1] = "phylo_root"

    return AugmentedNodeArrays(
        names=parsed_names,
        node_type=node_type,
        is_leaf=is_leaf,
        is_hidden=is_hidden,
        is_root=is_root,
        coords=coords,
    )


def _build_augmented_tree_bundle(newick: str) -> AugmentedTreeBundle:
    original_hx_tree = from_newick(newick)
    edge_len_ori_source = _hyperiax_edge_lens(original_hx_tree)

    super_root_edge_len = float(edge_len_ori_source.mean())
    newick_aug = _add_super_root_newick(newick, super_root_edge_len)
    augmented_hx_tree = from_newick(newick_aug)
    edge_len_aug = _hyperiax_edge_lens(augmented_hx_tree)

    expected_augmented_size = original_hx_tree.size + 1
    if edge_len_aug.shape != (expected_augmented_size,):
        raise ValueError(
            "Augmented Newick node count does not match source tree node count: "
            f"{edge_len_aug.shape[0]} augmented nodes vs {expected_augmented_size} expected."
        )

    edge_len_normalizer = _max_root_to_node_depth(augmented_hx_tree.topology.parents, edge_len_aug)
    if edge_len_normalizer <= 0:
        raise ValueError("Cannot normalize edge lengths for a zero-depth augmented tree.")

    return AugmentedTreeBundle(
        newick_ori=newick,
        newick_aug=newick_aug,
        edge_len_ori=np.concatenate((np.array([0.0], dtype=np.float64), edge_len_ori_source)),
        edge_len_aug=edge_len_aug,
        edge_len_norm=edge_len_aug / edge_len_normalizer,
        edge_len_normalizer=edge_len_normalizer,
    )


def _write_hdf5(
    output_path: Path,
    *,
    csv_path: Path,
    tree_path: Path,
    observations: LandmarkObservations,
    nodes: AugmentedNodeArrays,
    tree_bundle: AugmentedTreeBundle,
    normalization: CoordinateNormalization,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as h5:
        h5.attrs[schema.ATTR_SOURCE_CSV] = str(csv_path)
        h5.attrs[schema.ATTR_SOURCE_TREE] = str(tree_path)
        h5.attrs[schema.ATTR_NORMALIZATION_METHOD] = schema.NORMALIZATION_METHOD
        h5.attrs[schema.ATTR_LANDMARK_COUNT] = observations.landmark_count
        h5.attrs[schema.ATTR_COORDINATE_DIMENSIONS] = observations.coordinate_dimensions
        h5.attrs[schema.ATTR_NODE_ORDER] = schema.NODE_ORDER
        h5.attrs["node_order_note"] = (
            "Node 0 is the added super_root. Nodes 1..N preserve augmented "
            "Newick BFS order. CSV leaf rows are joined to Newick leaves by name."
        )

        nodes_group = h5.create_group(schema.NODES_GROUP)
        nodes_group.create_dataset(_dataset_name(schema.NODES_NAMES), data=nodes.names, dtype=string_dtype)
        nodes_group.create_dataset(_dataset_name(schema.NODES_NODE_TYPE), data=nodes.node_type, dtype=string_dtype)
        nodes_group.create_dataset(_dataset_name(schema.NODES_IS_LEAF), data=nodes.is_leaf)
        nodes_group.create_dataset(_dataset_name(schema.NODES_IS_HIDDEN), data=nodes.is_hidden)
        nodes_group.create_dataset(_dataset_name(schema.NODES_IS_ROOT), data=nodes.is_root)
        nodes_group.create_dataset(_dataset_name(schema.NODES_EDGE_LEN_ORI), data=tree_bundle.edge_len_ori)
        nodes_group.create_dataset(_dataset_name(schema.NODES_EDGE_LEN_AUG), data=tree_bundle.edge_len_aug)
        nodes_group.create_dataset(_dataset_name(schema.NODES_EDGE_LEN_NORM), data=tree_bundle.edge_len_norm)
        nodes_group.create_dataset(_dataset_name(schema.NODES_COORDS), data=nodes.coords, compression="gzip")

        tree_group = h5.create_group(schema.TREE_GROUP)
        tree_group.create_dataset(_dataset_name(schema.TREE_NEWICK_ORI), data=tree_bundle.newick_ori, dtype=string_dtype)
        tree_group.create_dataset(_dataset_name(schema.TREE_NEWICK_AUG), data=tree_bundle.newick_aug, dtype=string_dtype)
        tree_group.create_dataset(
            _dataset_name(schema.TREE_LEAF_NAMES),
            data=np.array(observations.leaf_names, dtype=object),
            dtype=string_dtype,
        )
        tree_group.attrs["topology_note"] = (
            "This group stores the original Newick, an augmented Newick with a unary "
            "super_root, and leaf names only. Node-aligned arrays live under /nodes."
        )

        preproc = h5.create_group(schema.PREPROC_GROUP)
        preproc.create_dataset(_dataset_name(schema.PREPROC_CENTER), data=normalization.center)
        preproc.create_dataset(_dataset_name(schema.PREPROC_SCALE), data=np.array(normalization.scale, dtype=np.float64))
        preproc.create_dataset(
            _dataset_name(schema.PREPROC_EDGE_LEN_NORMALIZER),
            data=np.array(tree_bundle.edge_len_normalizer, dtype=np.float64),
        )
        preproc.attrs["method"] = schema.NORMALIZATION_METHOD
        preproc.attrs["fit_on"] = "observed_leaf_coordinates"
        preproc.attrs["description"] = (
            "coords = (raw_coords - center) / scale, where center and scale are "
            "estimated from observed leaf coordinates only. Hidden coordinates are "
            "stored as NaN placeholders."
        )


def _hyperiax_edge_lens(tree) -> np.ndarray:
    if "edge_len" in tree.schema.names:
        return np.asarray(tree["edge_len"], dtype=np.float64)
    return np.asarray(tree["edge_length"], dtype=np.float64)


def _dataset_name(path: str) -> str:
    return path.rsplit("/", 1)[1]


def _root_to_node_depths(parents, edge_lens: np.ndarray) -> np.ndarray:
    parents = np.asarray(parents)
    edge_lens = np.asarray(edge_lens, dtype=np.float64)
    depths = np.zeros_like(edge_lens, dtype=np.float64)
    for index in range(1, len(edge_lens)):
        depths[index] = depths[parents[index]] + edge_lens[index]
    return depths


def _max_root_to_node_depth(parents, edge_lens: np.ndarray) -> float:
    return float(_root_to_node_depths(parents, edge_lens).max())


def _add_super_root_newick(newick: str, edge_length: float) -> str:
    if not newick.strip().endswith(";"):
        raise ValueError("Newick string must end with ';'.")
    phylo_tree = Tree(newick, format=1)
    phylo_tree.dist = float(edge_length)

    augmented_tree = Tree()
    augmented_tree.name = schema.SUPER_ROOT_NAME
    augmented_tree.dist = 0.0
    augmented_tree.add_child(phylo_tree)
    rendered = augmented_tree.write(format=1, dist_formatter="%.17g").strip()
    if not rendered.endswith(";"):
        raise ValueError("Generated augmented Newick did not end with ';'.")
    return f"{rendered[:-1]}{schema.SUPER_ROOT_NAME};"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, required=True, help="Landmark CSV input path.")
    parser.add_argument("--tree-path", type=Path, required=True, help="Newick tree input path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the generated HDF5 file. Defaults to the CSV parent directory.",
    )
    parser.add_argument(
        "--output-name",
        default="data",
        help="Generated HDF5 file name or stem. Default: data.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir or args.csv_path.parent
    output_path = _build_output_path(output_dir, args.output_name)
    output_path = convert_shape_dataset(args.csv_path, args.tree_path, output_path)
    print(output_path)
    return 0


def _build_output_path(output_dir: Path, output_name: str) -> Path:
    name_path = Path(output_name)
    if name_path.name != output_name:
        raise ValueError(f"output-name must be a file name, got {output_name!r}.")
    if name_path.suffix:
        filename = name_path.name
    else:
        filename = f"{name_path.name}.h5"
    return output_dir / filename


if __name__ == "__main__":
    raise SystemExit(main())
