from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from src import data_schema as schema
from scripts import inspect_data


def test_inspect_dataset_writes_standard_png_outputs_from_hdf5(tmp_path, monkeypatch):
    h5_path = _write_tiny_dataset_h5(tmp_path / schema.HDF5_FILENAME)
    rendered_trees = []

    def fake_render(self, output_path, *, tree_style, w, units, dpi):
        rendered_trees.append(
            {
                "leaf_names": [leaf.name for leaf in self.iter_leaves()],
                "output_path": Path(output_path),
                "tree_style": tree_style,
                "width": w,
                "units": units,
                "dpi": dpi,
            }
        )
        Image.new("RGBA", (24, 24), "white").save(output_path)

    monkeypatch.setattr(inspect_data.Tree, "render", fake_render)

    outputs = inspect_data.inspect_dataset(h5_path)

    assert outputs.tree_path == tmp_path / "tree.png"
    assert outputs.leaves_path == tmp_path / "leaves.png"
    for output_path in outputs:
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    assert rendered_trees[0]["leaf_names"] == ["B", "A", "C"]
    assert rendered_trees[0]["output_path"] == outputs.tree_path
    assert rendered_trees[0]["dpi"] == inspect_data.DEFAULT_TREE_DPI


def test_inspect_dataset_writes_outputs_to_explicit_output_dir(tmp_path, monkeypatch):
    h5_path = _write_tiny_dataset_h5(tmp_path / schema.HDF5_FILENAME)
    output_dir = tmp_path / "inspect"

    def fake_render(self, output_path, *, tree_style, w, units, dpi):
        Image.new("RGBA", (24, 24), "white").save(output_path)

    monkeypatch.setattr(inspect_data.Tree, "render", fake_render)

    outputs = inspect_data.inspect_dataset(h5_path, output_dir=output_dir)

    assert outputs.tree_path == output_dir / "tree.png"
    assert outputs.leaves_path == output_dir / "leaves.png"
    for output_path in outputs:
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_plot_leaf_shapes_3d_draws_axis_free_colored_scatter_ensemble(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    close_figure = plt.close
    captured_figures = []
    monkeypatch.setattr(plt, "close", lambda fig: captured_figures.append(fig))
    leaf_coords = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.5], [0.0, 1.0, 1.0]],
            [[0.2, 0.1, 0.3], [1.2, 0.2, 0.7], [0.1, 1.1, 1.3]],
        ],
        dtype=np.float64,
    )
    output_path = tmp_path / "leaves.png"

    inspect_data.plot_leaf_shapes(leaf_coords, output_path=output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    fig = captured_figures.pop()
    try:
        axis = fig.axes[0]
        assert axis.name == "3d"
        assert len(axis.lines) == 0
        assert len(axis.collections) == 2
        assert axis.collections[0].get_alpha() < 0.15
        assert axis.collections[1].get_sizes().min() > axis.collections[0].get_sizes().max()
        assert axis.collections[0].get_array().shape == (leaf_coords.shape[0] * leaf_coords.shape[1],)
        assert axis.collections[1].get_array().shape == (leaf_coords.shape[1],)
        assert axis.get_title() == ""
        assert axis.get_xlabel() == ""
        assert axis.get_ylabel() == ""
        assert axis.get_zlabel() == ""
        assert not axis.axison
        np.testing.assert_allclose(axis.elev, inspect_data.DEFAULT_3D_LEAF_VIEW_ELEV_DEG)
        np.testing.assert_allclose(axis.azim, inspect_data.DEFAULT_3D_LEAF_VIEW_AZIM_DEG)
    finally:
        close_figure(fig)


def test_plot_leaf_shapes_2d_draws_axis_free_shape_ensemble(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    close_figure = plt.close
    captured_figures = []
    monkeypatch.setattr(plt, "close", lambda fig: captured_figures.append(fig))
    leaf_coords = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            [[0.1, 0.0], [1.1, 0.1], [1.0, 1.2]],
            [[-0.1, 0.1], [0.9, 0.0], [0.8, 1.0]],
        ],
        dtype=np.float64,
    )

    inspect_data.plot_leaf_shapes(leaf_coords, output_path=tmp_path / "leaves.png")

    fig = captured_figures.pop()
    try:
        axis = fig.axes[0]
        assert len(axis.lines) == leaf_coords.shape[0] + 1
        leaf_lines = axis.lines[:-1]
        mean_line = axis.lines[-1]
        assert all(line.get_alpha() < 0.15 for line in leaf_lines)
        assert mean_line.get_alpha() > 0.8
        assert mean_line.get_linewidth() > leaf_lines[0].get_linewidth()
        offsets = axis.collections[0].get_offsets()
        np.testing.assert_allclose(offsets, leaf_coords.mean(axis=0))
        assert axis.get_title() == ""
        assert axis.get_xlabel() == ""
        assert axis.get_ylabel() == ""
        assert not axis.axison
        assert axis.yaxis_inverted()
        assert axis.get_aspect() == 1.0
    finally:
        close_figure(fig)


def test_inspect_data_cli_accepts_data_path_only(tmp_path, monkeypatch, capsys):
    h5_path = _write_tiny_dataset_h5(tmp_path / schema.HDF5_FILENAME)
    expected = inspect_data.InspectionOutputs(
        tree_path=tmp_path / "tree.png",
        leaves_path=tmp_path / "leaves.png",
    )
    calls = {}

    def fake_inspect_dataset(data_h5, *, output_dir=None):
        calls["data_h5"] = data_h5
        calls["output_dir"] = output_dir
        for path in expected:
            path.write_bytes(b"png")
        return expected

    monkeypatch.setattr(inspect_data, "inspect_dataset", fake_inspect_dataset)

    assert inspect_data.main(["--data-path", str(h5_path)]) == 0

    assert calls["data_h5"] == h5_path
    assert calls["output_dir"] is None
    output = capsys.readouterr().out
    assert "tree.png" in output
    assert "leaves.png" in output


def test_inspect_data_cli_accepts_output_dir(tmp_path, monkeypatch, capsys):
    h5_path = _write_tiny_dataset_h5(tmp_path / schema.HDF5_FILENAME)
    output_dir = tmp_path / "explicit"
    expected = inspect_data.InspectionOutputs(
        tree_path=output_dir / "tree.png",
        leaves_path=output_dir / "leaves.png",
    )
    calls = {}

    def fake_inspect_dataset(data_h5, *, output_dir=None):
        calls["data_h5"] = data_h5
        calls["output_dir"] = output_dir
        for path in expected:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"png")
        return expected

    monkeypatch.setattr(inspect_data, "inspect_dataset", fake_inspect_dataset)

    assert inspect_data.main(["--data-path", str(h5_path), "--output-dir", str(output_dir)]) == 0

    assert calls["data_h5"] == h5_path
    assert calls["output_dir"] == output_dir
    output = capsys.readouterr().out
    assert str(output_dir / "tree.png") in output
    assert str(output_dir / "leaves.png") in output


def test_trim_png_whitespace_crops_empty_bottom_area(tmp_path):
    image_path = tmp_path / "tree.png"
    image = Image.new("RGBA", (40, 80), "white")
    for x in range(10, 31):
        for y in range(5, 21):
            image.putpixel((x, y), (31, 31, 31, 255))
    image.save(image_path)

    inspect_data._trim_png_whitespace(image_path, padding=4)

    with Image.open(image_path) as trimmed:
        assert trimmed.size == (29, 24)


def test_enhance_png_tree_lines_darkens_antialiased_lines(tmp_path):
    image_path = tmp_path / "tree.png"
    image = Image.new("RGBA", (3, 1), "white")
    image.putpixel((0, 0), (240, 240, 240, 255))
    image.putpixel((1, 0), (255, 255, 255, 255))
    image.putpixel((2, 0), (158, 158, 158, 255))
    image.save(image_path)

    inspect_data._enhance_png_tree_lines(image_path, line_color=(31, 31, 31), threshold=250)

    with Image.open(image_path) as enhanced:
        assert enhanced.getpixel((0, 0)) == (31, 31, 31, 255)
        assert enhanced.getpixel((1, 0)) == (255, 255, 255, 255)
        assert enhanced.getpixel((2, 0)) == (31, 31, 31, 255)


def _tiny_newick() -> str:
    return "((B:0.2,A:0.2):0.3,C:1.25):2.0;"


def _write_tiny_dataset_h5(path: Path) -> Path:
    coords = np.asarray(
        [
            [[9.0, 9.0], [9.0, 9.0], [9.0, 9.0], [9.0, 9.0]],
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
            [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [4.0, 2.0]],
            [[0.0, 0.0], [0.0, 3.0], [4.0, 3.0], [4.0, 8.0]],
            [[2.0, 2.0], [2.0, 4.0], [6.0, 4.0], [6.0, 8.0]],
        ],
        dtype=np.float32,
    )
    with h5py.File(path, "w") as h5:
        nodes = h5.create_group(schema.NODES_GROUP)
        nodes.create_dataset("coords", data=coords)
        nodes.create_dataset("is_leaf", data=np.array([False, False, True, True, True]))
        tree = h5.create_group(schema.TREE_GROUP)
        tree.create_dataset("newick_ori", data=_tiny_newick())
    return path
