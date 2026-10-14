"""Regression checks for the side-by-side speedup figure."""

import pytest

from src.profiling.report import plot_summary


@pytest.mark.parametrize("kinds", [("traversal", "hardware"), ("traversal",), ("hardware",)])
@pytest.mark.parametrize("smoke", [False, True])
def test_speedups_have_separate_implementation_and_hardware_panels(tmp_path, monkeypatch, kinds, smoke):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        fig, axes = original_subplots(*args, **kwargs)
        figures.append(fig)
        return fig, axes

    monkeypatch.setattr(plt, "subplots", capture_subplots)
    speedups = [
        dict(dataset="butterflies", tips=425, landmarks=32, backend="gpu", kind=kind,
             ratio={"median": median, "min": median - 0.5, "max": median + 0.5, "count": 3})
        for kind, median in (("traversal", 2.0), ("hardware", 4.0)) if kind in kinds
    ]
    plot_summary({"groups": [], "speedups": speedups, "smoke": smoke}, tmp_path)
    assert len(figures) == 1
    fig = figures[0]
    if smoke:
        assert "Smoke comparison" in fig._suptitle.get_text()
        assert "CPU smoke" not in fig._suptitle.get_text()
    assert tuple(fig.get_size_inches()) == (10, 4)
    assert len(fig.axes) == 2
    left, right = fig.axes
    assert left.get_subplotspec().rowspan.start == right.get_subplotspec().rowspan.start == 0
    assert left.get_subplotspec().colspan.start == 0
    assert right.get_subplotspec().colspan.start == 1
    assert "reference / Hyperiax" in left.get_ylabel()
    assert "CPU / GPU" in right.get_ylabel()
    for ax, kind, median in ((left, "traversal", 2.0), (right, "hardware", 4.0)):
        if kind in kinds:
            assert [bar.get_height() for bar in ax.patches] == [median]
            assert "L=32" in ax.get_xticklabels()[0].get_text()
        else:
            assert not ax.patches
            assert "No paired" in ax.texts[0].get_text()
    for extension in ("png", "pdf"):
        assert (tmp_path / f"speedups.{extension}").stat().st_size > 0


@pytest.mark.parametrize("stem", ["scaling", "speedups"])
@pytest.mark.parametrize("formats", [("png",), ("pdf",), ("png", "pdf")])
def test_unavailable_results_replace_stale_plots_in_both_formats(tmp_path, monkeypatch, stem, formats):
    import matplotlib.pyplot as plt

    for extension in formats:
        (tmp_path / f"{stem}.{extension}").write_bytes(b"old figure")
    messages = []
    original_close = plt.close

    def capture_close(fig=None):
        if hasattr(fig, "axes"):
            messages.extend(text.get_text() for ax in fig.axes for text in ax.texts)
            assert "CPU smoke" not in fig._suptitle.get_text()
        original_close(fig)

    monkeypatch.setattr(plt, "close", capture_close)
    plot_summary({"groups": [], "speedups": [], "smoke": True}, tmp_path)
    assert any("unavailable" in message for message in messages)
    assert (tmp_path / f"{stem}.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / f"{stem}.pdf").read_bytes().startswith(b"%PDF")
    other = "speedups" if stem == "scaling" else "scaling"
    assert not (tmp_path / f"{other}.png").exists()


def test_no_plots_preserves_existing_figures(tmp_path):
    import io
    from rich.console import Console
    from src.profiling.artifacts import write_json
    from src.profiling.report import rebuild_report

    write_json(tmp_path / "config.json", {"smoke": True})
    paths = [tmp_path / f"{stem}.{ext}" for stem in ("scaling", "speedups") for ext in ("png", "pdf")]
    for path in paths:
        path.write_bytes(b"preserved figure")
    summary = rebuild_report(tmp_path, plots=False, console=Console(file=io.StringIO()))
    assert summary["jobs"] == 0
    assert all(path.read_bytes() == b"preserved figure" for path in paths)
