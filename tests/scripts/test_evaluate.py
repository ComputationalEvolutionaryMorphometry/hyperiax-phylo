from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import yaml


def test_load_parameter_means_uses_all_chains_after_burnin_and_thinning(tmp_path):
    from scripts.evaluate import load_parameter_means

    artifact_path = _write_parameter_artifact(tmp_path / "artifacts.h5")

    means = load_parameter_means(artifact_path, num_burnin=1, thin=2)

    np.testing.assert_allclose(means["k_alpha"], np.array([3.0, 7.0, 30.0, 70.0]).mean())
    np.testing.assert_allclose(means["k_sigma"], np.array([0.3, 0.7, 3.0, 7.0]).mean())
    np.testing.assert_allclose(means["obs_var"], np.array([0.03, 0.07, 0.3, 0.7]).mean())


def test_load_gelman_rubin_diagnostics_uses_parameter_traces_after_burnin_and_thinning(tmp_path):
    from scripts.evaluate import GELMAN_RUBIN_THRESHOLD, load_gelman_rubin_diagnostics

    artifact_path = _write_parameter_artifact(tmp_path / "artifacts.h5")

    diagnostics = load_gelman_rubin_diagnostics(artifact_path, num_burnin=1, thin=2)

    selected = np.array([[3.0, 7.0], [30.0, 70.0]], dtype=np.float64)
    within_chain_var = np.var(selected, axis=1, ddof=1).mean()
    between_chain_var = selected.shape[1] * np.var(selected.mean(axis=1), ddof=1)
    pooled_var = ((selected.shape[1] - 1) / selected.shape[1]) * within_chain_var
    pooled_var += between_chain_var / selected.shape[1]
    expected_r_hat = np.sqrt(pooled_var / within_chain_var)

    for parameter_name in ("k_alpha", "k_sigma", "obs_var"):
        diagnostic = diagnostics[parameter_name]
        assert diagnostic["num_chains"] == 2
        assert diagnostic["num_samples_per_chain"] == 2
        assert diagnostic["threshold"] == GELMAN_RUBIN_THRESHOLD
        assert diagnostic["passed"] is False
        np.testing.assert_allclose(diagnostic["r_hat"], expected_r_hat)


def test_gelman_rubin_diagnostics_mark_single_chain_unavailable(tmp_path):
    from scripts.evaluate import load_gelman_rubin_diagnostics

    artifact_path = _write_full_artifact_3d(tmp_path / "artifacts.h5")

    diagnostics = load_gelman_rubin_diagnostics(artifact_path, num_burnin=0, thin=1)

    for diagnostic in diagnostics.values():
        assert diagnostic["r_hat"] is None
        assert diagnostic["passed"] is None
        assert diagnostic["num_chains"] == 1
        assert diagnostic["num_samples_per_chain"] == 4
        assert diagnostic["reason"] == "Gelman-Rubin requires at least two chains."


def test_leaf_evaluation_plot_limits_to_sixteen_leaves_and_draws_truth_sample_uncertainty(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from scripts.evaluate import UNCERTAINTY_Z, _write_leaf_evaluation_plot

    close_figure = plt.close
    captured_figures = []
    monkeypatch.setattr(plt, "close", lambda fig: captured_figures.append(fig))

    leaf_names = tuple(f"leaf_{index}" for index in range(18))
    truth = np.stack(
        [
            np.array([[index, index], [index + 0.5, index + 0.5]], dtype=np.float64)
            for index in range(18)
        ]
    )
    samples = truth + 0.25
    output_path = tmp_path / "leaves.png"

    actual_path = _write_leaf_evaluation_plot(
        output_path,
        leaf_names=leaf_names,
        sample_leaf_coords=samples,
        observed_leaf_coords=truth,
        obs_var=0.25,
        parameter_means={"k_alpha": 1.23456, "k_sigma": 0.04567, "obs_var": 0.25},
    )

    assert actual_path == output_path
    assert output_path.exists()
    fig = captured_figures.pop()
    try:
        axes = fig.axes
        assert len(axes) == 16
        assert [axis.get_title() for axis in axes] == [f"leaf_{index}" for index in range(16)]
        assert "k_alpha=1.235" in fig._suptitle.get_text()
        assert "k_sigma=0.046" in fig._suptitle.get_text()
        assert "obs_var=0.250" in fig._suptitle.get_text()
        x_limits = [axis.get_xlim() for axis in axes]
        y_limits = [axis.get_ylim() for axis in axes]
        assert all(limit == x_limits[0] for limit in x_limits)
        assert all(limit == y_limits[0] for limit in y_limits)
        assert all(axis.get_shared_x_axes().joined(axes[0], axis) for axis in axes)
        assert all(axis.get_shared_y_axes().joined(axes[0], axis) for axis in axes)
        for axis in axes:
            assert axis.get_visible()
            assert len(axis.collections) >= 2
            assert len(axis.patches) == 2
            np.testing.assert_allclose(axis.patches[0].width, 2.0 * UNCERTAINTY_Z * np.sqrt(0.25))
            np.testing.assert_allclose(axis.patches[0].height, 2.0 * UNCERTAINTY_Z * np.sqrt(0.25))
            assert axis.yaxis_inverted()
            assert axis.get_aspect() == 1.0
    finally:
        close_figure(fig)


def test_leaf_evaluation_plot_draws_3d_observation_noise_blobs(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from scripts.evaluate import _write_leaf_evaluation_plot

    close_figure = plt.close
    captured_figures = []
    monkeypatch.setattr(plt, "close", lambda fig: captured_figures.append(fig))

    leaf_names = ("leaf_a", "leaf_b")
    observed = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.2], [0.5, 0.5, 0.4]],
            [[1.0, 1.0, 1.0], [1.5, 1.0, 1.2], [1.5, 1.5, 1.4]],
        ],
        dtype=np.float64,
    )
    samples = observed + 0.1
    output_path = tmp_path / "leaves-3d.png"

    actual_path = _write_leaf_evaluation_plot(
        output_path,
        leaf_names=leaf_names,
        sample_leaf_coords=samples,
        observed_leaf_coords=observed,
        obs_var=0.04,
        parameter_means={"k_alpha": 1.0, "k_sigma": 0.5, "obs_var": 0.04},
    )

    assert actual_path == output_path
    assert output_path.exists()
    fig = captured_figures.pop()
    try:
        visible_axes = [axis for axis in fig.axes if axis.get_visible()]
        assert len(visible_axes) == 2
        assert all(axis.name == "3d" for axis in visible_axes)
        assert [axis.get_title() for axis in visible_axes] == ["leaf_a", "leaf_b"]
        for axis in visible_axes:
            assert axis.get_zlabel() == "scaled z"
            assert len(axis.collections) >= 3
            assert any(collection.get_alpha() is not None and collection.get_alpha() < 0.2 for collection in axis.collections)
            assert len(axis.patches) == 0
    finally:
        close_figure(fig)


def test_evaluate_artifacts_writes_standard_run_outputs(tmp_path, monkeypatch):
    from scripts import evaluate
    from src.bffg import MCMCModelConfig

    artifact_path = _write_full_artifact(tmp_path / "artifacts.h5")
    data_h5 = tmp_path / "data.h5"
    data_h5.write_bytes(b"placeholder")
    (tmp_path / "config.json").write_text(
        (
            '{"model": {"num_edge_steps": 7, "covar_jitter": 1e-6, '
            '"dist_jitter": 1e-7}, "augment": {"remove_lmk": [0, 2]}}\n'
        ),
        encoding="utf-8",
    )
    calls = {}

    def fake_sample_conditional_leaf_shapes(data_h5_arg, *, params, model_config, remove_lmk=None):
        calls["data_h5"] = data_h5_arg
        calls["params"] = params
        calls["model_config"] = model_config
        calls["remove_lmk"] = remove_lmk
        return SimpleNamespace(
            leaf_names=("leaf_a",),
            sample_leaf_coords=np.zeros((1, 2, 2), dtype=np.float64),
            observed_leaf_coords=np.ones((1, 2, 2), dtype=np.float64),
        )

    monkeypatch.setattr(evaluate, "sample_conditional_leaf_shapes", fake_sample_conditional_leaf_shapes)
    monkeypatch.setattr(evaluate, "_write_leaf_evaluation_plot", lambda output_path, **kwargs: Path(output_path))
    monkeypatch.setattr(evaluate, "plot_root_shape", lambda *args, **kwargs: Path(kwargs["output_path"]))

    result = evaluate.evaluate_artifacts(
        artifacts_h5=artifact_path,
        data_h5=data_h5,
        num_burnin=1,
        thin=2,
    )

    assert result.leaves_path == tmp_path / "leaves.png"
    assert result.root_path == tmp_path / "root.png"
    assert result.trace_path == tmp_path / "trace.png"
    assert result.hist_path == tmp_path / "hist.png"
    assert set(result.gelman_rubin) == {"k_alpha", "k_sigma", "obs_var"}
    assert result.gelman_rubin["k_alpha"]["num_chains"] == 2
    assert calls["data_h5"] == data_h5
    assert isinstance(calls["model_config"], MCMCModelConfig)
    assert calls["model_config"].num_edge_steps == 7
    assert calls["model_config"].covar_jitter == 1e-6
    assert calls["model_config"].dist_jitter == 1e-7
    assert calls["remove_lmk"] == [0, 2]
    np.testing.assert_allclose(float(calls["params"].k_alpha), result.parameter_means["k_alpha"])


def test_evaluate_cli_reads_data_burnin_and_thin_from_config(tmp_path, monkeypatch, capsys):
    from scripts import evaluate

    artifact_path = _write_full_artifact(tmp_path / "artifacts.h5")
    data_h5 = tmp_path / "data.h5"
    data_h5.write_bytes(b"placeholder")
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "h5": str(data_h5),
                "plot": {
                    "num_burnin": 1,
                    "thin": 2,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    expected = evaluate.EvaluationResult(
        leaves_path=tmp_path / "leaves.png",
        root_path=tmp_path / "root.png",
        trace_path=tmp_path / "trace.png",
        hist_path=tmp_path / "hist.png",
        parameter_means={"k_alpha": 1.5, "k_sigma": 0.25, "obs_var": 0.04},
    )
    calls = {}

    def fake_evaluate_artifacts(*, artifacts_h5, data_h5, num_burnin, thin):
        calls["artifacts_h5"] = artifacts_h5
        calls["data_h5"] = data_h5
        calls["num_burnin"] = num_burnin
        calls["thin"] = thin
        return expected

    monkeypatch.setattr(evaluate, "evaluate_artifacts", fake_evaluate_artifacts)

    assert evaluate.main(
        [
            "--artifacts-path",
            str(artifact_path),
            "--config",
            str(config_path),
        ]
    ) == 0

    assert calls == {
        "artifacts_h5": artifact_path,
        "data_h5": data_h5,
        "num_burnin": 1,
        "thin": 2,
    }
    output = capsys.readouterr().out
    assert "leaves.png" in output
    assert "root.png" in output
    assert "trace.png" in output
    assert "hist.png" in output
    assert '"parameter_means"' in output
    assert '"gelman_rubin"' in output
    assert '"k_alpha": 1.5' in output


def test_evaluate_cli_applies_gpu_visible_before_evaluating(tmp_path, monkeypatch):
    import os

    from scripts import evaluate

    artifact_path = _write_full_artifact(tmp_path / "artifacts.h5")
    data_h5 = tmp_path / "data.h5"
    data_h5.write_bytes(b"placeholder")
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "h5": str(data_h5),
                "gpu_visible": "2",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    expected = evaluate.EvaluationResult(
        leaves_path=tmp_path / "leaves.png",
        root_path=tmp_path / "root.png",
        trace_path=tmp_path / "trace.png",
        hist_path=tmp_path / "hist.png",
        parameter_means={"k_alpha": 1.5, "k_sigma": 0.25, "obs_var": 0.04},
    )
    calls = {}

    def fake_evaluate_artifacts(**kwargs):
        calls["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        calls["xla_preallocate"] = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
        return expected

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    monkeypatch.setattr(evaluate, "evaluate_artifacts", fake_evaluate_artifacts)

    assert evaluate.main(["--artifacts-path", str(artifact_path), "--config", str(config_path)]) == 0

    assert calls == {
        "cuda_visible_devices": "2",
        "xla_preallocate": "false",
    }


def test_trace_plots_write_trace_and_hist_pngs(tmp_path):
    from scripts.evaluate import plot_artifact_traces

    artifact_path = _write_full_artifact(tmp_path / "artifacts.h5")

    trace_path, hist_path = plot_artifact_traces(
        artifact_path,
        num_burnin=2,
        thin=2,
    )

    assert trace_path == tmp_path / "trace.png"
    assert hist_path == tmp_path / "hist.png"
    assert trace_path.exists()
    assert hist_path.exists()
    assert trace_path.stat().st_size > 0
    assert hist_path.stat().st_size > 0


def test_trace_plot_uses_two_parameters_and_shared_chain_legend(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from scripts.evaluate import PARAMETER_NAMES, _write_trace_plot

    close_figure = plt.close
    captured_figures = []
    monkeypatch.setattr(plt, "close", lambda fig: captured_figures.append(fig))
    parameter_chains = {
        parameter_name: {
            "chain_000": np.array([1.0, 3.0, 5.0, 7.0]),
            "chain_001": np.array([2.0, 4.0, 6.0, 8.0]),
        }
        for parameter_name in PARAMETER_NAMES
    }

    _write_trace_plot(tmp_path / "trace.png", parameter_chains, num_burnin=2)

    fig = captured_figures.pop()
    try:
        assert fig._suptitle is None
        assert len(fig.axes) == 2
        assert [axis.get_ylabel() for axis in fig.axes] == [r"$k_\alpha$", r"$k_\sigma$"]
        assert len(fig.legends) == 1
        legend = fig.legends[0]
        assert [text.get_text() for text in legend.get_texts()] == ["chain_000", "chain_001"]
        assert legend._ncols == 2
        for axis in fig.axes:
            assert axis.get_legend() is None
            trace_lines = [line for line in axis.lines if line.get_label() == "_nolegend_" and line.get_linestyle() == "-"]
            mean_lines = [line for line in axis.lines if line.get_label() in {"chain_000", "chain_001"}]
            assert len(trace_lines) == 2
            assert len(mean_lines) == 2
            assert all(line.get_alpha() >= 0.45 for line in trace_lines)
            assert all(line.get_linewidth() >= 1.3 for line in trace_lines)
            assert all(line.get_linewidth() >= 1.4 for line in mean_lines)
            assert all("mean=" not in line.get_label() for line in mean_lines)
    finally:
        close_figure(fig)


def test_hist_plot_uses_two_parameters_shared_chain_legend_and_unlabeled_mean_markers(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from scripts.evaluate import PARAMETER_NAMES, _write_hist_plot

    close_figure = plt.close
    captured_figures = []
    monkeypatch.setattr(plt, "close", lambda fig: captured_figures.append(fig))
    parameter_chains = {
        parameter_name: {
            "chain_000": np.linspace(0.0, 1.0, 20),
            "chain_001": np.linspace(1.0, 3.0, 20),
        }
        for parameter_name in PARAMETER_NAMES
    }

    _write_hist_plot(tmp_path / "hist.png", parameter_chains, num_burnin=0, thin=1)

    fig = captured_figures.pop()
    try:
        assert fig._suptitle is None
        assert len(fig.axes) == 2
        assert [axis.get_title(loc="left") for axis in fig.axes] == [r"$k_\alpha$", r"$k_\sigma$"]
        assert len(fig.legends) == 1
        legend = fig.legends[0]
        assert [text.get_text() for text in legend.get_texts()] == ["chain_000", "chain_001"]
        assert legend._ncols == 2
        for axis in fig.axes:
            assert axis.get_legend() is None
            assert axis.get_ylabel() == "density"
            assert axis.get_xlim() == (0.0, 3.0)
            step_lines = [line for line in axis.lines if line.get_drawstyle() == "steps-post"]
            kde_lines = [line for line in axis.lines if line.get_label() in {"chain_000", "chain_001"}]
            mean_lines = [line for line in axis.lines if line.get_label() == "_mean"]
            assert len(step_lines) == 2
            assert len(kde_lines) == 2
            assert len(mean_lines) == 2
            assert all(line.get_alpha() <= 0.25 for line in step_lines)
            assert all(line.get_linewidth() >= 2.0 for line in kde_lines)
            assert [line.get_color() for line in step_lines] == [line.get_color() for line in kde_lines]
            assert [line.get_color() for line in mean_lines] == [line.get_color() for line in kde_lines]
            assert len(axis.texts) == 0
    finally:
        close_figure(fig)


def test_trace_and_hist_plots_omit_legend_for_single_chain(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from scripts.evaluate import PARAMETER_NAMES, _write_hist_plot, _write_trace_plot

    close_figure = plt.close
    captured_figures = []
    monkeypatch.setattr(plt, "close", lambda fig: captured_figures.append(fig))
    parameter_chains = {
        parameter_name: {"chain_000": np.linspace(1.0, 2.0, 20)}
        for parameter_name in PARAMETER_NAMES
    }

    _write_trace_plot(tmp_path / "trace.png", parameter_chains, num_burnin=4)
    _write_hist_plot(tmp_path / "hist.png", parameter_chains, num_burnin=4, thin=1)

    trace_fig, hist_fig = captured_figures
    try:
        assert all(axis.get_legend() is None for axis in trace_fig.axes)
        assert all(axis.get_legend() is None for axis in hist_fig.axes)
    finally:
        close_figure(trace_fig)
        close_figure(hist_fig)


def test_root_plot_writes_clean_posterior_ensemble_without_chart_furniture(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from scripts.evaluate import plot_root_shape

    close_figure = plt.close
    captured_figures = []
    monkeypatch.setattr(plt, "close", lambda fig: captured_figures.append(fig))
    artifact_path = _write_full_artifact(tmp_path / "artifacts.h5")

    output_path = plot_root_shape(artifact_path, num_burnin=1, thin=2)

    assert output_path == tmp_path / "root.png"
    assert output_path.exists()
    fig = captured_figures.pop()
    try:
        axis = fig.axes[0]
        offsets = axis.collections[0].get_offsets()
        expected_selected = np.array(
            [
                [[101.0, 101.0], [101.0, 101.0], [101.0, 101.0]],
                [[103.0, 103.0], [103.0, 103.0], [103.0, 103.0]],
            ]
        )
        np.testing.assert_allclose(offsets, expected_selected.mean(axis=0))
        assert len(axis.lines) == len(expected_selected)
        assert all(line.get_alpha() < 0.2 for line in axis.lines)
        assert all(line.get_linewidth() < 1.0 for line in axis.lines)
        assert len(axis.patches) == 0
        assert len(axis.texts) == 0
        assert axis.get_title() == ""
        assert axis.get_xlabel() == ""
        assert axis.get_ylabel() == ""
        assert not axis.axison
        assert axis.yaxis_inverted()
        assert axis.get_aspect() == 1.0
    finally:
        close_figure(fig)


def test_root_plot_writes_3d_mean_and_marginal_intervals(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from scripts.evaluate import plot_root_shape

    close_figure = plt.close
    captured_figures = []
    monkeypatch.setattr(plt, "close", lambda fig: captured_figures.append(fig))
    artifact_path = _write_full_artifact_3d(tmp_path / "artifacts-3d.h5")

    output_path = plot_root_shape(artifact_path, num_burnin=1, thin=2)

    assert output_path == tmp_path / "root.png"
    assert output_path.exists()
    fig = captured_figures.pop()
    try:
        axis = fig.axes[0]
        assert axis.name == "3d"
        assert axis.get_zlabel() == "scaled z"
        assert len(axis.collections) >= 1
        assert len(axis.lines) >= 1
        assert any("credible intervals" in text.get_text() for text in axis.texts)
    finally:
        close_figure(fig)


def _write_parameter_artifact(path: Path) -> Path:
    with h5py.File(path, "w") as h5:
        samples = h5.create_group("samples")
        for parameter_name, chain_0, chain_1 in [
            ("k_alpha", [1.0, 3.0, 5.0, 7.0], [10.0, 30.0, 50.0, 70.0]),
            ("k_sigma", [0.1, 0.3, 0.5, 0.7], [1.0, 3.0, 5.0, 7.0]),
            ("obs_var", [0.01, 0.03, 0.05, 0.07], [0.1, 0.3, 0.5, 0.7]),
        ]:
            group = samples.create_group(parameter_name)
            group.create_dataset("chain_000", data=np.asarray(chain_0, dtype=np.float64))
            group.create_dataset("chain_001", data=np.asarray(chain_1, dtype=np.float64))
    return path


def _write_full_artifact(path: Path) -> Path:
    with h5py.File(path, "w") as h5:
        samples = h5.create_group("samples")
        for parameter_name, offset in [("k_alpha", 0.1), ("k_sigma", 0.2), ("obs_var", 0.3)]:
            parameter_group = samples.create_group(parameter_name)
            parameter_group.create_dataset("chain_000", data=[offset + value * 0.01 for value in range(8)])
            parameter_group.create_dataset("chain_001", data=[offset + 0.05 + value * 0.02 for value in range(8)])

        root_group = samples.create_group("phylo_root")
        root_group.create_dataset(
            "chain_000",
            data=np.array(
                [
                    [[100.0, 100.0], [100.0, 100.0], [100.0, 100.0]],
                    [[101.0, 101.0], [101.0, 101.0], [101.0, 101.0]],
                    [[102.0, 102.0], [102.0, 102.0], [102.0, 102.0]],
                    [[103.0, 103.0], [103.0, 103.0], [103.0, 103.0]],
                ]
            ),
        )

        trace = h5.create_group("trace")
        log_posteriors = trace.create_group("log_posteriors")
        log_posteriors.create_dataset("chain_000", data=[1.0, 2.0, 2.5, 3.0, 3.2, 3.4, 3.5, 3.6])
        log_posteriors.create_dataset("chain_001", data=[0.8, 1.8, 2.1, 2.9, 3.1, 3.2, 3.3, 3.4])
        accepted = trace.create_group("accepted")
        accepted.create_dataset("chain_000", data=[True, False, True, True, False, True, False, True])
        accepted.create_dataset("chain_001", data=[False, True, True, False, True, True, False, True])
    return path


def _write_full_artifact_3d(path: Path) -> Path:
    with h5py.File(path, "w") as h5:
        samples = h5.create_group("samples")
        for parameter_name, offset in [("k_alpha", 0.1), ("k_sigma", 0.2), ("obs_var", 0.3)]:
            parameter_group = samples.create_group(parameter_name)
            parameter_group.create_dataset("chain_000", data=[offset + value * 0.01 for value in range(4)])

        root_group = samples.create_group("phylo_root")
        root_group.create_dataset(
            "chain_000",
            data=np.array(
                [
                    [[100.0, 100.0, 100.0], [100.0, 100.0, 100.0]],
                    [[101.0, 101.0, 101.0], [101.0, 101.0, 101.0]],
                    [[102.0, 102.0, 102.0], [102.0, 102.0, 102.0]],
                    [[103.0, 103.0, 103.0], [103.0, 103.0, 103.0]],
                ]
            ),
        )

        trace = h5.create_group("trace")
        log_posteriors = trace.create_group("log_posteriors")
        log_posteriors.create_dataset("chain_000", data=[1.0, 2.0, 2.5, 3.0])
        accepted = trace.create_group("accepted")
        accepted.create_dataset("chain_000", data=[True, False, True, True])
    return path
