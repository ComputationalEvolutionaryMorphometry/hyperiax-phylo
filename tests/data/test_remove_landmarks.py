from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.run_mcmc as cli
from src import driver
from src.bffg import MCMCModelConfig, build_bffg_context
from src.driver import MCMCDriverConfig
from src.loader import load_augmented_butterfly_tree


REPO_ROOT = Path(__file__).resolve().parents[2]
BUTTERFLIES_H5 = REPO_ROOT / "data/butterflies/data.h5"


def test_loader_removes_configured_landmarks_from_hdf5_coords():
    full_dataset = load_augmented_butterfly_tree(BUTTERFLIES_H5)
    single_removed_dataset = load_augmented_butterfly_tree(BUTTERFLIES_H5, remove_lmk=0)
    reduced_dataset = load_augmented_butterfly_tree(BUTTERFLIES_H5, remove_lmk=[0, 2])

    full_coords = np.asarray(full_dataset.tree["coords"])
    single_removed_coords = np.asarray(single_removed_dataset.tree["coords"])
    reduced_coords = np.asarray(reduced_dataset.tree["coords"])

    assert single_removed_dataset.removed_landmarks == (0,)
    assert single_removed_coords.shape == (850, 117, 2)
    np.testing.assert_allclose(single_removed_coords[:, 0, :], full_coords[:, 1, :], equal_nan=True)

    assert reduced_dataset.original_landmark_count == 118
    assert reduced_dataset.removed_landmarks == (0, 2)
    assert reduced_dataset.kept_landmarks == tuple(index for index in range(118) if index not in {0, 2})
    assert reduced_coords.shape == (850, 116, 2)
    np.testing.assert_allclose(reduced_coords[:, 0, :], full_coords[:, 1, :], equal_nan=True)
    np.testing.assert_allclose(reduced_coords[:, 1, :], full_coords[:, 3, :], equal_nan=True)

    context = build_bffg_context(reduced_dataset, MCMCModelConfig(num_edge_steps=1))
    assert context.n_landmarks == 116
    assert context.state_dim == 232


@pytest.mark.parametrize("remove_lmk", [[118], [-1], [0, 0], [True]])
def test_loader_rejects_invalid_remove_lmk(remove_lmk):
    with pytest.raises(ValueError):
        load_augmented_butterfly_tree(BUTTERFLIES_H5, remove_lmk=remove_lmk)


def test_run_mcmc_cli_reads_augment_remove_lmk_and_prints_reduced_dimensions(
    monkeypatch, tmp_path, capsys
):
    calls = {}

    class FakeTree:
        def __getitem__(self, name):
            if name == "coords":
                return np.zeros((850, 116, 2), dtype=np.float32)
            raise KeyError(name)

    def fake_load(path, *, remove_lmk=None):
        calls["h5_path"] = path
        calls["remove_lmk"] = remove_lmk
        return SimpleNamespace(
            tree=FakeTree(),
            removed_landmarks=tuple(remove_lmk or ()),
            original_landmark_count=118,
            h5_path=path,
        )

    def fake_run_mcmc(dataset, *, run_dir, model_config, driver_config):
        calls["dataset"] = dataset
        return SimpleNamespace(run_dir=Path(run_dir), summary={"state_dim": 232})

    monkeypatch.setattr(cli, "load_augmented_butterfly_tree", fake_load)
    monkeypatch.setattr(cli, "run_mcmc", fake_run_mcmc)
    monkeypatch.setattr(cli, "plot_artifact_traces", lambda *args, **kwargs: None)

    config_path = tmp_path / "mcmc.yaml"
    config_path.write_text(
        "h5: data/butterflies/data.h5\n"
        "run_name: reduced-run\n"
        "augment:\n"
        "  remove_lmk: [0, 2]\n"
        "model: {}\n"
        "driver: {}\n",
        encoding="utf-8",
    )

    assert cli.main(["--config", str(config_path)]) == 0

    assert calls["h5_path"] == Path("data/butterflies/data.h5")
    assert calls["remove_lmk"] == (0, 2)
    output = capsys.readouterr().out
    assert "augment.remove_lmk: skipped landmark indices [0, 2]" in output
    assert "remaining_landmarks: 116" in output
    assert "state_dim: 232" in output


def test_process_worker_reloads_dataset_with_removed_landmarks(monkeypatch):
    calls = {}

    def fake_load(path, *, remove_lmk=None):
        calls["path"] = path
        calls["remove_lmk"] = remove_lmk
        return object()

    monkeypatch.setattr(driver, "load_augmented_butterfly_tree", fake_load)
    monkeypatch.setattr(driver, "build_bffg_context", lambda dataset, config: SimpleNamespace(config=config))
    monkeypatch.setattr(driver, "_build_mcmc_target", lambda context: object())

    result = driver._run_mcmc_chain_chunk_worker(
        (
            "data/butterflies/data.h5",
            MCMCModelConfig(num_edge_steps=1),
            MCMCDriverConfig(num_samples=1),
            [],
            None,
            False,
            (0, 2),
        )
    )

    assert result == []
    assert calls["path"] == "data/butterflies/data.h5"
    assert calls["remove_lmk"] == (0, 2)


@pytest.mark.parametrize(
    "config_path",
    [
        Path("config/butterflies.yaml"),
        Path("config/beaks.yaml"),
    ],
)
def test_default_configs_include_disabled_remove_lmk_augment(config_path):
    config = cli._load_run_config(config_path)

    assert config["augment"] == {"remove_lmk": None}
