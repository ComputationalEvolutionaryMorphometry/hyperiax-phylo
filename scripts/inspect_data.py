"""Inspect a processed `data.h5` file and write standard dataset figures.

Example:
    uv run python -m scripts.inspect_data --data-path data/butterflies/data.h5
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import h5py
import numpy as np
from PIL import Image, ImageChops
from ete3 import AttrFace, NodeStyle, TextFace, Tree, TreeNode, TreeStyle, faces

from src import data_schema as schema

LAYOUT_HALF_CIRCLE = "half-circle"
LAYOUT_FULL_CIRCLE_OVERVIEW = "full-circle-overview"

DEFAULT_MAX_LEAF_LABELS = 24
DEFAULT_TREE_WIDTH_PX = 2400
DEFAULT_LARGE_TREE_WIDTH_PX = 5200
DEFAULT_TREE_DPI = 500
DEFAULT_LEAF_LABEL_FONT_SIZE = 8
DEFAULT_BRANCH_LINE_WIDTH = 2
DEFAULT_BRANCH_COLOR = "#1f1f1f"
DEFAULT_TRIM_PADDING_PX = 24
DEFAULT_LINE_ENHANCE_THRESHOLD = 250
DEFAULT_FULL_CIRCLE_MARGIN_PX = 500
DEFAULT_3D_LEAF_VIEW_ELEV_DEG = 18.0
DEFAULT_3D_LEAF_VIEW_AZIM_DEG = -70.0

TREE_OUTPUT_FILENAME = schema.TREE_PLOT_FILENAME
LEAVES_OUTPUT_FILENAME = schema.LEAVES_PLOT_FILENAME


@dataclass(frozen=True)
class InspectionOutputs:
    """Standard output paths produced by the dataset inspection command."""

    tree_path: Path
    leaves_path: Path

    def __iter__(self) -> Iterator[Path]:
        return iter((self.tree_path, self.leaves_path))


def inspect_dataset(data_h5: str | Path, *, output_dir: str | Path | None = None) -> InspectionOutputs:
    """Generate the standard tree and leaves inspection PNGs."""

    data_h5 = Path(data_h5)
    output_dir = data_h5.parent if output_dir is None else Path(output_dir)
    newick, leaf_coords = _load_inspection_inputs(data_h5)
    outputs = InspectionOutputs(
        tree_path=output_dir / TREE_OUTPUT_FILENAME,
        leaves_path=output_dir / LEAVES_OUTPUT_FILENAME,
    )
    plot_tree_from_newick(newick, output_path=outputs.tree_path)
    plot_leaf_shapes(leaf_coords, output_path=outputs.leaves_path)
    return outputs


def _load_inspection_inputs(data_h5: Path) -> tuple[str, np.ndarray]:
    with h5py.File(data_h5, "r") as h5:
        newick = _decode_h5_scalar(h5[schema.TREE_NEWICK_ORI][()])
        leaf_coords = _load_leaf_coords_from_h5(h5, source_path=data_h5)
    return newick, leaf_coords


def _decode_h5_scalar(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _load_leaf_coords_from_h5(h5: h5py.File, *, source_path: Path) -> np.ndarray:
    coords = h5[schema.NODES_COORDS][:].astype(np.float64, copy=False)
    is_leaf = h5[schema.NODES_IS_LEAF][:].astype(bool, copy=False)
    if coords.shape[0] != is_leaf.shape[0]:
        raise ValueError(
            f"{schema.NODES_COORDS} has {coords.shape[0]} rows but "
            f"{schema.NODES_IS_LEAF} has {is_leaf.shape[0]} rows."
        )
    leaf_coords = coords[is_leaf]
    if leaf_coords.shape[0] == 0:
        raise ValueError(f"No leaf rows found in {source_path}.")
    if not np.isfinite(leaf_coords).all():
        raise ValueError(f"Leaf coordinates in {source_path} contain NaN or infinite values.")
    return leaf_coords


def sample_leaf_names(leaf_names: Iterable[str], *, max_leaf_labels: int = DEFAULT_MAX_LEAF_LABELS) -> set[str]:
    """Return evenly spaced leaf names to label in dense tree plots."""

    leaf_names = tuple(leaf_names)
    if max_leaf_labels < 0:
        raise ValueError(f"max_leaf_labels must be >= 0, got {max_leaf_labels}.")
    if max_leaf_labels == 0 or not leaf_names:
        return set()
    if len(leaf_names) <= max_leaf_labels:
        return set(leaf_names)
    if max_leaf_labels == 1:
        return {leaf_names[0]}

    last_index = len(leaf_names) - 1
    sampled_indices = {
        int(round(sample_index * last_index / (max_leaf_labels - 1)))
        for sample_index in range(max_leaf_labels)
    }
    return {leaf_names[index] for index in sorted(sampled_indices)}


def plot_tree_from_newick(
    newick: str,
    *,
    output_path: str | Path,
    branch_color: str = DEFAULT_BRANCH_COLOR,
) -> Path:
    """Render a raw Newick tree to PNG."""

    output_path = Path(output_path)
    tree = Tree(newick.strip(), format=1)
    leaf_names = tuple(leaf.name for leaf in tree.iter_leaves())
    if not leaf_names:
        raise ValueError("No leaves found in Newick tree.")
    layout = LAYOUT_FULL_CIRCLE_OVERVIEW if len(leaf_names) > DEFAULT_MAX_LEAF_LABELS else LAYOUT_HALF_CIRCLE
    sampled_leaf_names = (
        set(leaf_names)
        if layout == LAYOUT_FULL_CIRCLE_OVERVIEW
        else sample_leaf_names(leaf_names, max_leaf_labels=DEFAULT_MAX_LEAF_LABELS)
    )
    _style_nodes(
        tree,
        sampled_leaf_names=sampled_leaf_names,
        draw_branch_right_labels=layout == LAYOUT_HALF_CIRCLE,
        leaf_label_font_size=DEFAULT_LEAF_LABEL_FONT_SIZE,
        branch_line_width=DEFAULT_BRANCH_LINE_WIDTH,
        branch_color=branch_color,
    )

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = DEFAULT_LARGE_TREE_WIDTH_PX if layout == LAYOUT_FULL_CIRCLE_OVERVIEW else DEFAULT_TREE_WIDTH_PX
    tree.render(str(output_path), tree_style=_tree_style(layout), w=width, units="px", dpi=DEFAULT_TREE_DPI)
    _trim_png_whitespace(output_path, padding=DEFAULT_TRIM_PADDING_PX)
    _enhance_png_tree_lines(
        output_path,
        line_color=_rgb_from_hex_color(branch_color),
        threshold=DEFAULT_LINE_ENHANCE_THRESHOLD,
    )
    return output_path


def _style_nodes(
    tree: Tree,
    *,
    sampled_leaf_names: set[str],
    draw_branch_right_labels: bool,
    leaf_label_font_size: int,
    branch_line_width: int,
    branch_color: str,
) -> None:
    node_style = NodeStyle()
    node_style["size"] = 0
    node_style["fgcolor"] = branch_color
    node_style["hz_line_color"] = branch_color
    node_style["vt_line_color"] = branch_color
    node_style["hz_line_width"] = branch_line_width
    node_style["vt_line_width"] = branch_line_width

    for node in tree.traverse():
        node.set_style(node_style)
        if draw_branch_right_labels and node.is_leaf() and node.name in sampled_leaf_names:
            node.add_face(
                TextFace(node.name, fsize=leaf_label_font_size),
                column=0,
                position="branch-right",
            )


def _tree_style(layout: str) -> TreeStyle:
    if layout == LAYOUT_HALF_CIRCLE:
        tree_style = TreeStyle()
        tree_style.mode = "c"
        tree_style.arc_start = -180
        tree_style.arc_span = 180
        tree_style.show_leaf_name = False
        tree_style.show_scale = False
        tree_style.show_branch_length = False
        tree_style.show_branch_support = False
        tree_style.branch_vertical_margin = 2
        tree_style.margin_top = 8
        tree_style.margin_bottom = 8
        tree_style.margin_left = 8
        tree_style.margin_right = 8
        return tree_style

    if layout == LAYOUT_FULL_CIRCLE_OVERVIEW:
        tree_style = TreeStyle()
        tree_style.mode = "c"
        tree_style.arc_start = -180
        tree_style.arc_span = 360
        tree_style.show_leaf_name = False
        tree_style.show_scale = False
        tree_style.show_branch_length = False
        tree_style.show_branch_support = False
        tree_style.force_topology = True
        tree_style.layout_fn = _aligned_leaf_name_layout()
        tree_style.draw_guiding_lines = True
        tree_style.guiding_lines_color = "#bbbbbb"
        tree_style.guiding_lines_type = 2
        tree_style.draw_aligned_faces_as_table = True
        tree_style.complete_branch_lines_when_necessary = True
        tree_style.branch_vertical_margin = 0
        tree_style.margin_top = DEFAULT_FULL_CIRCLE_MARGIN_PX
        tree_style.margin_bottom = DEFAULT_FULL_CIRCLE_MARGIN_PX
        tree_style.margin_left = DEFAULT_FULL_CIRCLE_MARGIN_PX
        tree_style.margin_right = DEFAULT_FULL_CIRCLE_MARGIN_PX
        return tree_style

    raise ValueError(f"Unknown layout {layout!r}.")


def _aligned_leaf_name_layout():
    def layout(node: TreeNode) -> None:
        if node.is_leaf():
            faces.add_face_to_node(
                AttrFace("name", fsize=DEFAULT_LEAF_LABEL_FONT_SIZE),
                node,
                0,
                position="aligned",
            )

    return layout


def _trim_png_whitespace(path: Path, *, padding: int = DEFAULT_TRIM_PADDING_PX) -> None:
    if padding < 0:
        raise ValueError(f"padding must be >= 0, got {padding}.")

    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        rgb = rgba.convert("RGB")
        background = Image.new("RGB", rgb.size, "white")
        diff = ImageChops.difference(rgb, background)
        bbox = diff.getbbox()
        if bbox is None:
            return

        left, upper, right, lower = bbox
        crop_box = (
            max(left - padding, 0),
            max(upper - padding, 0),
            min(right + padding, rgba.width),
            min(lower + padding, rgba.height),
        )
        rgba.crop(crop_box).save(path)


def _enhance_png_tree_lines(
    path: Path,
    *,
    line_color: tuple[int, int, int] = (31, 31, 31),
    threshold: int = DEFAULT_LINE_ENHANCE_THRESHOLD,
) -> None:
    if not 0 <= threshold <= 255:
        raise ValueError(f"threshold must be in [0, 255], got {threshold}.")

    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        pixels = np.asarray(rgba).copy()
        rgb = pixels[:, :, :3]
        alpha = pixels[:, :, 3]
        line_mask = (alpha > 0) & (rgb < threshold).any(axis=2)
        pixels[:, :, :3][line_mask] = np.asarray(line_color, dtype=np.uint8)
        Image.fromarray(pixels, mode="RGBA").save(path)


def _rgb_from_hex_color(color: str) -> tuple[int, int, int]:
    if len(color) == 7 and color.startswith("#"):
        return tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
    return (31, 31, 31)


def plot_leaf_shapes(leaf_coords: np.ndarray, *, output_path: str | Path) -> Path:
    """Plot observed leaf landmark coordinates."""

    output_path = Path(output_path)
    _write_leaf_shapes_plot(output_path, np.asarray(leaf_coords, dtype=np.float64))
    return output_path


def _write_leaf_shapes_plot(output_path: Path, leaf_coords: np.ndarray) -> None:
    if leaf_coords.ndim != 3 or leaf_coords.shape[2] not in (2, 3):
        raise ValueError(
            "leaf_coords must have shape (n_leaves, n_landmarks, 2 or 3) for plotting; "
            f"got {leaf_coords.shape}."
        )
    if leaf_coords.shape[0] == 0:
        raise ValueError("At least one leaf row is required.")
    if not np.isfinite(leaf_coords).all():
        raise ValueError("Leaf coordinates contain NaN or infinite values.")

    _configure_matplotlib()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if leaf_coords.shape[2] == 3:
        _write_leaf_shapes_plot_3d(output_path, leaf_coords, plt)
        return

    _write_leaf_shapes_plot_2d(output_path, leaf_coords, plt)


def _write_leaf_shapes_plot_2d(output_path: Path, leaf_coords: np.ndarray, plt) -> None:
    fig, axis = plt.subplots(figsize=(6.8, 4.3), constrained_layout=False)
    for leaf_shape in leaf_coords:
        axis.plot(
            leaf_shape[:, 0],
            leaf_shape[:, 1],
            color="#4f7fa6",
            linewidth=0.35,
            alpha=0.055,
            zorder=1,
        )
    mean_shape = leaf_coords.mean(axis=0)
    axis.plot(mean_shape[:, 0], mean_shape[:, 1], color="#123f5f", linewidth=1.2, alpha=0.92, zorder=2)
    axis.scatter(
        mean_shape[:, 0],
        mean_shape[:, 1],
        s=7.0,
        color="#123f5f",
        edgecolors="white",
        linewidths=0.25,
        zorder=3,
    )
    _style_leaf_shape_ensemble_axis(axis, leaf_coords)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _style_leaf_shape_ensemble_axis(axis, leaf_coords: np.ndarray) -> None:
    values = leaf_coords.reshape(-1, leaf_coords.shape[-1])
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    pad = spans.max() * 0.035
    axis.set_xlim(float(mins[0] - pad), float(maxs[0] + pad))
    axis.set_ylim(float(mins[1] - pad), float(maxs[1] + pad))
    axis.set_aspect("equal", adjustable="box")
    axis.invert_yaxis()
    axis.margins(0)
    axis.set_axis_off()
    axis.set_facecolor("white")


def _write_leaf_shapes_plot_3d(output_path: Path, leaf_coords: np.ndarray, plt) -> None:
    mean_shape = leaf_coords.mean(axis=0)
    z_values = leaf_coords[:, :, 2]
    z_min = float(z_values.min())
    z_max = float(z_values.max())

    fig = plt.figure(figsize=(6.8, 4.3), constrained_layout=False)
    axis = fig.add_subplot(111, projection="3d")
    axis.set_position((0.0, 0.0, 1.0, 1.0))
    axis.scatter(
        leaf_coords[:, :, 0].reshape(-1),
        leaf_coords[:, :, 1].reshape(-1),
        leaf_coords[:, :, 2].reshape(-1),
        s=3.0,
        c=z_values.reshape(-1),
        cmap="Blues",
        vmin=z_min,
        vmax=z_max,
        alpha=0.08,
        linewidths=0,
        depthshade=False,
        zorder=1,
    )
    axis.scatter(
        mean_shape[:, 0],
        mean_shape[:, 1],
        mean_shape[:, 2],
        s=20.0,
        c=mean_shape[:, 2],
        cmap="cividis",
        vmin=z_min,
        vmax=z_max,
        edgecolors="white",
        linewidths=0.35,
        alpha=0.98,
        depthshade=False,
        zorder=2,
    )
    _style_leaf_shape_scatter_axis_3d(axis, leaf_coords)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _style_shape_axis(axis, *, title: str) -> None:
    axis.set_title(title, loc="left", fontsize=10)
    axis.set_xlabel("scaled x")
    axis.set_ylabel("scaled y")
    axis.set_aspect("equal", adjustable="box")
    axis.invert_yaxis()
    axis.grid(False)


def _style_leaf_shape_scatter_axis_3d(axis, leaf_coords: np.ndarray) -> None:
    values = leaf_coords.reshape(-1, 3)
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    pad = spans.max() * 0.045
    axis.set_xlim(float(mins[0] - pad), float(maxs[0] + pad))
    axis.set_ylim(float(mins[1] - pad), float(maxs[1] + pad))
    axis.set_zlim(float(mins[2] - pad), float(maxs[2] + pad))
    axis.set_box_aspect(tuple(float(span) for span in spans))
    axis.set_proj_type("ortho")
    axis.view_init(elev=DEFAULT_3D_LEAF_VIEW_ELEV_DEG, azim=DEFAULT_3D_LEAF_VIEW_AZIM_DEG)
    axis.margins(0)
    axis.set_axis_off()
    axis.set_facecolor("white")


def _configure_matplotlib() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import seaborn as sns

    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette="colorblind",
        font_scale=1.0,
        rc={
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "font.size": 9,
            "figure.dpi": 120,
            "savefig.dpi": 240,
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True, help="Processed dataset HDF5 file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for inspection PNG outputs. Defaults to the data.h5 parent directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    outputs = inspect_dataset(args.data_path, output_dir=args.output_dir)
    for output_path in outputs:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
