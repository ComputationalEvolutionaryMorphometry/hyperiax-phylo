"""Shared HDF5 schema names for shape analysis artifacts."""

from __future__ import annotations


NORMALIZATION_METHOD = "global_center_uniform_scale_to_unit_box"
NODE_ORDER = "augmented_super_root_plus_newick_bfs_order"
SUPER_ROOT_NAME = "super_root"

LANDMARK_CSV_FILENAME = "lmks.csv"
TREE_NEWICK_FILENAME = "tree.nwk"
HDF5_FILENAME = "data.h5"
TREE_PLOT_FILENAME = "tree.png"
LEAVES_PLOT_FILENAME = "leaves.png"
DEPTH_PLOT_FILENAME = "depth.png"
DIST_PLOT_FILENAME = "dist.png"

NODES_GROUP = "nodes"
NODES_NAMES = "nodes/names"
NODES_NODE_TYPE = "nodes/node_type"
NODES_IS_LEAF = "nodes/is_leaf"
NODES_IS_HIDDEN = "nodes/is_hidden"
NODES_IS_ROOT = "nodes/is_root"
NODES_EDGE_LEN_ORI = "nodes/edge_len_ori"
NODES_EDGE_LEN_AUG = "nodes/edge_len_aug"
NODES_EDGE_LEN_NORM = "nodes/edge_len_norm"
NODES_COORDS = "nodes/coords"

TREE_GROUP = "tree"
TREE_NEWICK_ORI = "tree/newick_ori"
TREE_NEWICK_AUG = "tree/newick_aug"
TREE_LEAF_NAMES = "tree/leaf_names"

PREPROC_GROUP = "preproc"
PREPROC_CENTER = "preproc/center"
PREPROC_SCALE = "preproc/scale"
PREPROC_EDGE_LEN_NORMALIZER = "preproc/edge_len_normalizer"

ATTR_SOURCE_CSV = "source_csv"
ATTR_SOURCE_TREE = "source_tree"
ATTR_NORMALIZATION_METHOD = "normalization_method"
ATTR_LANDMARK_COUNT = "landmark_count"
ATTR_COORDINATE_DIMENSIONS = "coordinate_dimensions"
ATTR_NODE_ORDER = "node_order"

REQUIRED_DATASET_PATHS = (
    NODES_COORDS,
    NODES_EDGE_LEN_AUG,
    NODES_EDGE_LEN_NORM,
    NODES_EDGE_LEN_ORI,
    NODES_IS_HIDDEN,
    NODES_IS_LEAF,
    NODES_IS_ROOT,
    NODES_NAMES,
    NODES_NODE_TYPE,
    PREPROC_CENTER,
    PREPROC_EDGE_LEN_NORMALIZER,
    PREPROC_SCALE,
    TREE_LEAF_NAMES,
    TREE_NEWICK_AUG,
)

FORBIDDEN_ROOT_PATHS = (
    "landmarks",
    "csv_tree_consistency",
)

FORBIDDEN_NODES_PATHS = (
    "edge_len",
    "edge_length",
)

FORBIDDEN_TREE_PATHS = (
    "newick",
    "leaf_edge_length",
    "parent_index",
    "postorder_index",
    "traversal_node_names",
)
