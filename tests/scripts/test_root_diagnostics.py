import io
import json

import arviz as az
import h5py
import numpy as np
import pytest
from rich.console import Console

from src.diagnostics import (DiagnosticConfig, compute_diagnostics, diagnose_scalar,
                             diagnostic_config, evaluate_diagnostics)


def artifact(path, *, dimensions=2, chains=4, draws=300):
    rng = np.random.default_rng(14)
    root = rng.normal(size=(chains, draws, 3, dimensions))
    with h5py.File(path, "w") as h5:
        for name in ("k_alpha", "k_sigma", "obs_var", "phylo_root"):
            group = h5.create_group(f"samples/{name}")
            values = root if name == "phylo_root" else rng.normal(size=(chains, draws))
            for index in range(chains):
                group.create_dataset(f"chain_{index:03d}", data=values[index])
    return root


@pytest.mark.parametrize("dimensions", [2, 3])
def test_coordinate_metrics_match_arviz_and_common_pca(tmp_path, dimensions):
    path = tmp_path / "artifacts.h5"
    root = artifact(path, dimensions=dimensions)
    report, traces = compute_diagnostics(path, num_burnin=20)
    values = root[:, 20:, 0, 0]
    row = report["root"]["variables"]["landmark_000_x"]
    assert row["r_hat"] == pytest.approx(float(az.rhat(values, method="rank")))
    assert row["ess_bulk"] == pytest.approx(float(az.ess(values, method="bulk")))
    assert row["ess_tail"] == pytest.approx(float(az.ess(values, method="tail", prob=(.05, .95))))
    assert row["mcse_mean"] == pytest.approx(float(az.mcse(values, method="mean")))
    assert row["mcse_mean_over_sd"] == pytest.approx(row["mcse_mean"] / values.std(ddof=1))
    assert report["root"]["summary"]["variables"] == 3 * dimensions
    assert report["thin"] == 1
    assert report["root"]["retained_draws_per_chain"] == 280
    flat = root[:, 20:].reshape(4, 280, -1)
    scores = (flat - np.asarray(report["pca"]["center"])) @ np.asarray(report["pca"]["basis"]).T
    for index in range(3):
        np.testing.assert_allclose(traces[f"PC{index + 1}"], scores[:, :, index])
    root_rows = report["root"]["variables"]
    worst = max(root_rows, key=lambda key: root_rows[key]["r_hat"])
    assert worst in traces
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("case", ["missing", "single", "short", "constant", "nonfinite", "unequal"])
def test_invalid_root_never_passes_or_emits_nan_json(tmp_path, case):
    path = tmp_path / "artifacts.h5"
    artifact(path, chains=1 if case == "single" else 4, draws=3 if case == "short" else 100)
    with h5py.File(path, "a") as h5:
        if case == "missing":
            del h5["samples/phylo_root"]
        elif case == "constant":
            h5["samples/phylo_root/chain_000"][:, 0, 0] = 0
        elif case == "nonfinite":
            h5["samples/phylo_root/chain_000"][20, 0, 0] = np.nan
        elif case == "unequal":
            del h5["samples/phylo_root/chain_000"]
            h5["samples/phylo_root"].create_dataset("chain_000", data=np.ones((3, 3, 2)))
    report, _ = compute_diagnostics(path, num_burnin=0)
    assert report["root"]["summary"]["passed"] is None
    assert report["status"] != "passed"
    assert "reason" in report["root"] or any("reason" in row for row in report["root"]["variables"].values())
    json.dumps(report, allow_nan=False)


def test_burnin_removed_before_validation_and_empty_retained_is_unavailable(tmp_path):
    path = tmp_path / "artifacts.h5"
    artifact(path)
    with h5py.File(path, "a") as h5:
        h5["samples/phylo_root/chain_000"][0, 0, 0] = np.nan
    report, _ = compute_diagnostics(path, num_burnin=10)
    assert report["root"]["summary"]["computed"] == 6
    report, _ = compute_diagnostics(path, num_burnin=300)
    assert report["root"]["summary"]["computed"] == 0
    assert report["root"]["summary"]["passed"] is None


def test_shifted_chains_are_flagged():
    values = np.random.default_rng(42).normal(size=(4, 500))
    values[0] += 10
    row = diagnose_scalar(values, DiagnosticConfig())
    assert row["r_hat"] > 1.01
    assert row["passed"] is False


@pytest.mark.parametrize("config", [{"rhat_threshold": 1}, {"ess_threshold": float("nan")},
                                    {"mcse_ratio_threshold": True}, {"num_pcs": 0}, {"thin": 3}])
def test_invalid_diagnostic_settings_rejected(config):
    with pytest.raises(ValueError):
        diagnostic_config(config)


def test_diagnostics_write_json_rich_and_plots_without_changing_samples(tmp_path):
    path = tmp_path / "artifacts.h5"
    artifact(path, draws=150)
    original = path.read_bytes()
    stream = io.StringIO()
    report = evaluate_diagnostics(path, num_burnin=10, console=Console(file=stream, width=120))
    assert path.read_bytes() == original
    assert json.loads((tmp_path / "diagnostics.json").read_text()) == report
    assert (tmp_path / "root_diagnostics.png").stat().st_size > 0
    assert (tmp_path / "root_diagnostics.pdf").stat().st_size > 0
    assert "Max R-hat" in stream.getvalue()
    assert "unavailable entries are not passes" in stream.getvalue()


def test_diagnostics_only_ignores_thinning_and_does_not_simulate_or_load_data(tmp_path, monkeypatch):
    from scripts import evaluate
    path = tmp_path / "artifacts.h5"
    artifact(path, draws=100)
    config = tmp_path / "config.yaml"
    config.write_text("plot: {num_burnin: 10, thin: 999}\ndiagnostics: {ess_threshold: 250}\n")
    def forbidden(*args, **kwargs):
        pytest.fail("Diagnostics-only must not load data, configure GPU or sample.")
    monkeypatch.setattr(evaluate, "_data_path_from_config", forbidden)
    monkeypatch.setattr(evaluate, "_apply_gpu_visibility", forbidden)
    monkeypatch.setattr(evaluate, "sample_conditional_leaf_shapes", forbidden)
    assert evaluate.main(["--artifacts-path", str(path), "--config", str(config), "--diagnostics-only"]) == 0
    report = json.loads((tmp_path / "diagnostics.json").read_text())
    assert report["thin"] == 1
    assert report["root"]["retained_draws_per_chain"] == 90
    assert report["thresholds"]["ess_threshold"] == 250


def test_root_plot_uses_all_chains_unthinned_mean_and_balanced_curves(tmp_path, monkeypatch):
    from scripts import evaluate
    path = tmp_path / "artifacts.h5"
    root = artifact(path, draws=301)
    captured = {}
    def capture(output, samples, **kwargs):
        captured.update(samples=samples, **kwargs)
    monkeypatch.setattr(evaluate, "_write_root_shape_plot", capture)
    evaluate.plot_root_shape(path, num_burnin=10, thin=3)
    np.testing.assert_array_equal(captured["samples"], root[:, 10:].reshape(-1, 3, 2))
    assert len(captured["display_samples"]) == 120
    for index in range(4):
        available = root[index, 10::3]
        expected = available[np.linspace(0, len(available) - 1, 30, dtype=int)]
        np.testing.assert_array_equal(captured["display_samples"][index * 30:(index + 1) * 30], expected)


def test_missing_root_replaces_stale_plots_with_unavailable_notice(tmp_path):
    path = tmp_path / "artifacts.h5"
    artifact(path, draws=100)
    with h5py.File(path, "a") as h5:
        del h5["samples/phylo_root"]
    (tmp_path / "root_diagnostics.png").write_bytes(b"stale")
    report = evaluate_diagnostics(path, num_burnin=0, console=Console(file=io.StringIO()))
    assert report["root"]["summary"]["passed"] is None
    assert "Missing chain group" in report["root"]["reason"]
    assert (tmp_path / "root_diagnostics.png").read_bytes().startswith(b"\x89PNG")
    assert len(report["plots"]) == 2
