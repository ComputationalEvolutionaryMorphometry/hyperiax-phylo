"""ESS post-processing must not resample, alter traces, or rewrite timings."""

import io
import json

import arviz as az
import h5py
import numpy as np
import pytest
from rich.console import Console

from src.profiling.artifacts import read_json, write_json
from src.profiling.diagnostics import diagnostic_settings, diagnose_chains, diagnose_artifact
from src.profiling.report import rebuild_report


def test_bulk_tail_and_rank_rhat_match_arviz_after_burnin():
    values = np.random.default_rng(2).normal(size=(4, 1000))
    values[:, :500] += 20
    result = diagnose_chains(values, burnin_fraction=0.5, wall_seconds=20)
    kept = values[:, 500:]
    assert result["ess_bulk"] == pytest.approx(float(az.ess(kept, method="bulk")))
    assert result["ess_tail"] == pytest.approx(float(az.ess(kept, method="tail", prob=(0.05, 0.95))))
    assert result["r_hat"] == pytest.approx(float(az.rhat(kept, method="rank")))
    assert result["ess_bulk_per_second"] == pytest.approx(result["ess_bulk"] / 20)
    assert result["retained_draws_total"] == 2000
    assert result["burnin_draws_per_chain"] == 500
    assert result["status"] == "computed"
    assert result["warnings"] == []


def test_autocorrelation_reduces_ess_and_chain_offsets_warn():
    noise = np.random.default_rng(3).normal(size=(4, 1000))
    correlated = noise.copy()
    for step in range(1, 1000):
        correlated[:, step] += 0.95 * correlated[:, step - 1]
    iid = diagnose_chains(noise, burnin_fraction=0)
    ar = diagnose_chains(correlated, burnin_fraction=0)
    shifted = diagnose_chains(noise + np.arange(4)[:, None] * 10, burnin_fraction=0)
    assert ar["ess_bulk"] < iid["ess_bulk"] / 5
    assert shifted["r_hat"] > 1.01
    assert any("mixing" in warning for warning in shifted["warnings"])


def test_per_chain_ess_is_computed_independently_not_divided_joint_ess(tmp_path):
    path = tmp_path / "artifacts.h5"
    _write_artifact(path)
    rng = np.random.default_rng(91)
    with h5py.File(path, "a") as artifact:
        for offset, name in enumerate(("k_alpha", "k_sigma", "obs_var")):
            for index in range(2):
                values = rng.normal(size=200)
                if offset == 0:
                    for step in range(1, values.size):
                        values[step] += 0.9 * values[step - 1]
                artifact[f"samples/{name}/chain_{index:03d}"][:] = values + index * 100
    diagnostics = diagnose_artifact(path, burnin_fraction=0.25)
    with h5py.File(path, "r") as artifact:
        for index in range(2):
            key = f"chain_{index:03d}"
            chain = diagnostics["chains"][key]
            assert chain["status"] == "computed"
            for name, row in chain["variables"].items():
                values = np.asarray(artifact[f"samples/{name}/{key}"])[None, 50:]
                assert row["ess_bulk"] == pytest.approx(float(az.ess(values, method="bulk")))
                assert row["ess_tail"] == pytest.approx(float(az.ess(values, method="tail", prob=(0.05, 0.95))))
                assert row["burnin_draws_per_chain"] == 50
                assert row["r_hat"] is None
            for metric in ("ess_bulk", "ess_tail"):
                assert chain[metric] == min(row[metric] for row in chain["variables"].values())
            assert chain["variables"]["obs_var"]["ess_bulk"] > diagnostics["variables"]["obs_var"]["ess_bulk"]


def test_per_chain_ess_minimum_does_not_hide_invalid_parameter(tmp_path):
    path = tmp_path / "artifacts.h5"
    _write_artifact(path)
    with h5py.File(path, "a") as artifact:
        artifact["samples/k_sigma/chain_000"][:] = 1
    result = diagnose_artifact(path)
    chain = result["chains"]["chain_000"]
    assert chain["status"] == "unavailable"
    assert chain["ess_bulk"] is None and chain["ess_tail"] is None
    assert chain["variables"]["k_alpha"]["status"] == "computed"
    assert result["chains"]["chain_001"]["status"] == "computed"
    with h5py.File(path, "a") as artifact:
        del artifact["samples/obs_var"]
    result = diagnose_artifact(path)
    assert all(chain["ess_bulk"] is None for chain in result["chains"].values())
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("values", [np.ones((2, 50)), np.arange(10)[None, :],
                                    np.ones((2, 3)), np.full((2, 20), np.nan),
                                    np.zeros((2, 30, 3)), np.empty((0, 10))])
def test_degenerate_or_insufficient_chains_are_not_valid_ess(values):
    result = diagnose_chains(values, burnin_fraction=0)
    assert result["status"] == "unavailable"
    assert result["ess_bulk"] is None and result["r_hat"] is None
    assert result["warnings"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("fraction", [True, -0.1, 1, float("nan"), "0.5"])
def test_invalid_burnin_is_rejected(fraction):
    with pytest.raises(ValueError, match="burnin_fraction"):
        diagnostic_settings({}, fraction)


def test_settings_default_and_override():
    assert diagnostic_settings({}) == {"burnin_fraction": 0.5}
    assert diagnostic_settings({"full_diagnostics": {"burnin_fraction": 0.2}}) == {"burnin_fraction": 0.2}
    assert diagnostic_settings({"full_diagnostics": {"burnin_fraction": 0.2}}, 0) == {"burnin_fraction": 0.0}
    with pytest.raises(ValueError):
        diagnostic_settings({"full_diagnostics": {"thin": 2}})


def _write_artifact(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.random.default_rng(5).normal(size=(2, 200))
    with h5py.File(path, "w") as artifact:
        for name in ("samples/k_alpha", "samples/k_sigma", "samples/obs_var", "trace/log_posteriors"):
            group = artifact.create_group(name)
            for index, chain in enumerate(values):
                group.create_dataset(f"chain_{index:03d}", data=chain)
    return values


def test_report_backfills_without_touching_original_measurements(tmp_path):
    job = tmp_path / "jobs" / "job_0000"
    artifact = job / "analysis" / "artifacts.h5"
    values = _write_artifact(artifact)
    record = dict(job_id="job_0000", dataset="test", experiment="full", tips=8, landmarks=4,
                  axes=[], implementation="hyperiax", backend="cpu", seed=0, status="ok",
                  subset={"nodes": 16, "dimensions": 2}, process_wall_seconds=20., analysis_seconds=15.,
                  analysis_summary={"chain_timings": [dict(initial_target_seconds=1., measured_loop_seconds=10.,
                                                           measured_iterations=200)] * 2,
                                    "acceptance_rates": [0.2, 0.3]}, artifact_path="/old/location/artifacts.h5")
    write_json(job / "result.json", record)
    write_json(tmp_path / "config.json", {"smoke": True})
    before = {path: path.read_bytes() for path in (artifact, job / "result.json", tmp_path / "config.json")}
    stream = io.StringIO()
    summary = rebuild_report(tmp_path, plots=False, console=Console(file=stream, width=160))
    diagnostics = read_json(job / "diagnostics.json")
    assert summary["full_analyses"][0]["diagnostics"] == diagnostics
    assert diagnostics["variables"]["k_alpha"]["retained_draws_per_chain"] == 100
    assert diagnostics["variables"]["k_alpha"]["ess_bulk"] == pytest.approx(float(az.ess(values[:, 100:], method="bulk")))
    assert read_json(tmp_path / "results.json")[0]["diagnostics"] == diagnostics
    assert read_json(tmp_path / "summary.json")["full_analyses"][0]["diagnostics"] == diagnostics
    assert len(diagnostics["chains"]) == 2
    for index in range(2):
        chain = diagnostics["chains"][f"chain_{index:03d}"]
        assert chain["ess_bulk"] == pytest.approx(float(az.ess(values[index:index + 1, 100:], method="bulk")))
        assert chain["ess_tail"] == pytest.approx(float(az.ess(values[index:index + 1, 100:], method="tail")))
    full_table = stream.getvalue().split("Full analyses:", 1)[1].split("Per-chain ESS:", 1)[0]
    assert full_table.index("Acceptance") < full_table.index("Bulk ESS") < full_table.index("Tail ESS")
    assert "Bulk ESS" in stream.getvalue() and "R-hat" in stream.getvalue()
    assert "Smoke experiment" in stream.getvalue()
    assert all(path.read_bytes() == content for path, content in before.items())
    repeated = rebuild_report(tmp_path, plots=False, console=Console(file=io.StringIO()))
    assert repeated == summary
    overridden = rebuild_report(tmp_path, plots=False, burnin_fraction=0.25, console=Console(file=io.StringIO()))
    assert overridden["full_analyses"][0]["diagnostics"]["variables"]["k_alpha"]["retained_draws_per_chain"] == 150
    assert overridden["full_analyses"][0]["diagnostics"]["chains"]["chain_000"]["variables"]["k_alpha"]["retained_draws_per_chain"] == 150
    assert all(path.read_bytes() == content for path, content in before.items())


def test_missing_or_malformed_artifacts_report_unavailability(tmp_path):
    missing = diagnose_artifact(tmp_path / "missing.h5")
    assert missing["status"] == "unavailable" and missing["warnings"]
    path = tmp_path / "bad.h5"
    _write_artifact(path)
    with h5py.File(path, "a") as artifact:
        del artifact["samples/k_alpha/chain_001"]
        artifact["samples/k_alpha"].create_dataset("chain_001", data=np.arange(5))
        del artifact["samples/k_sigma"]
    result = diagnose_artifact(path)
    assert result["status"] == "partial"
    assert result["variables"]["k_alpha"]["status"] == "unavailable"
    assert result["variables"]["k_sigma"]["status"] == "unavailable"
    assert result["variables"]["obs_var"]["status"] == "computed"
    assert result["variables"]["obs_var"]["ess_bulk_per_second"] is None
