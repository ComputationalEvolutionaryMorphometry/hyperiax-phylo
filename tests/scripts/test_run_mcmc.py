import os
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src import data_schema as schema
import scripts.run_mcmc as cli


def test_run_mcmc_cli_config_keys_match_mcmc_dataclass_fields():
    assert set(cli.MODEL_CONFIG_KEYS) == {field.name for field in fields(cli.MCMCModelConfig)}
    assert set(cli.DRIVER_CONFIG_KEYS) == {field.name for field in fields(cli.MCMCDriverConfig)}


def test_run_mcmc_cli_builds_driver_configs_from_yaml_and_runs_mcmc(monkeypatch, tmp_path, capsys):
    calls = {}
    dataset = SimpleNamespace()

    def fake_load(path, *, remove_lmk=None):
        calls["h5_path"] = path
        calls["remove_lmk"] = remove_lmk
        return dataset

    def fake_run_mcmc(loaded_dataset, *, run_dir, model_config, driver_config):
        calls["dataset"] = loaded_dataset
        calls["run_dir"] = run_dir
        calls["model_config"] = model_config
        calls["driver_config"] = driver_config
        return SimpleNamespace(
            run_dir=Path(run_dir),
            summary={"num_chains": driver_config.num_chains, "num_samples": driver_config.num_samples},
        )

    monkeypatch.setattr(cli, "load_augmented_butterfly_tree", fake_load)
    monkeypatch.setattr(cli, "run_mcmc", fake_run_mcmc)
    monkeypatch.setattr(cli, "plot_artifact_traces", lambda *args, **kwargs: None)

    h5_path = tmp_path / schema.HDF5_FILENAME
    config_path = tmp_path / "mcmc.yaml"
    config_path.write_text(
        "\n".join(
            [
                "h5: config-only.h5",
                "run_name: config-run",
                "model:",
                "  num_edge_steps: 2",
                "  covar_jitter: 1e-8",
                "  dist_jitter: 1e-7",
                "  k_alpha_prior_alpha: 4.0",
                "  k_alpha_prior_beta: 0.2",
                "  k_alpha_init: 0.12",
                "  k_alpha_max: 0.9",
                "  k_alpha_min: 1e-4",
                "  k_sigma_prior_alpha: 5.0",
                "  k_sigma_prior_beta: 0.3",
                "  k_sigma_init: [0.21, 0.22]",
                "  k_sigma_max: 0.7",
                "  k_sigma_min: 2e-4",
                "  obs_var_prior_alpha: 3.0",
                "  obs_var_prior_beta: 0.004",
                "  obs_var_init: null",
                "  obs_var_max: 0.2",
                "  obs_var_min: 2e-5",
                "  enable_x64: false",
                "driver:",
                "  num_samples: 3",
                "  num_chains: 2",
                "  pcn_eta: 0.8",
                "  k_alpha_proposal_var: 0.01",
                "  k_sigma_proposal_var: 0.02",
                "  obs_var_proposal_var: 0.03",
                "  random_seed: 99",
                "  chain_backend: process",
                "  num_processes: 2",
                "  progress_bar: false",
                "plot:",
                "  num_burnin: 1",
                "  thin: 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "--config",
            str(config_path),
            "--h5",
            str(h5_path),
            "--run-name",
            "named-run",
        ]
    )

    assert exit_code == 0
    assert calls["h5_path"] == h5_path
    assert calls["remove_lmk"] is None
    assert calls["dataset"] is dataset
    assert calls["run_dir"] == Path("runs/named-run")
    assert calls["model_config"].num_edge_steps == 2
    assert calls["model_config"].covar_jitter == 1e-8
    assert calls["model_config"].dist_jitter == 1e-7
    assert calls["model_config"].k_alpha_prior_alpha == 4.0
    assert calls["model_config"].k_alpha_prior_beta == 0.2
    assert calls["model_config"].k_alpha_init == 0.12
    assert calls["model_config"].k_alpha_max == 0.9
    assert calls["model_config"].k_alpha_min == 1e-4
    assert calls["model_config"].k_sigma_prior_alpha == 5.0
    assert calls["model_config"].k_sigma_prior_beta == 0.3
    assert calls["model_config"].k_sigma_init == (0.21, 0.22)
    assert calls["model_config"].k_sigma_max == 0.7
    assert calls["model_config"].k_sigma_min == 2e-4
    assert calls["model_config"].obs_var_prior_alpha == 3.0
    assert calls["model_config"].obs_var_prior_beta == 0.004
    assert calls["model_config"].obs_var_init is None
    assert calls["model_config"].obs_var_max == 0.2
    assert calls["model_config"].obs_var_min == 2e-5
    assert calls["model_config"].enable_x64 is False
    assert calls["driver_config"].num_samples == 3
    assert calls["driver_config"].num_chains == 2
    assert calls["driver_config"].pcn_eta == 0.8
    assert calls["driver_config"].k_alpha_proposal_var == 0.01
    assert calls["driver_config"].k_sigma_proposal_var == 0.02
    assert calls["driver_config"].obs_var_proposal_var == 0.03
    assert not hasattr(calls["driver_config"], "init_k_alpha")
    assert not hasattr(calls["driver_config"], "init_k_sigma")
    assert not hasattr(calls["driver_config"], "init_obs_var")
    assert not hasattr(calls["driver_config"], "infer_obs_var")
    assert calls["driver_config"].random_seed == 99
    assert calls["driver_config"].chain_backend == "process"
    assert calls["driver_config"].num_processes == 2
    assert calls["driver_config"].progress_bar is False

    output = capsys.readouterr().out
    assert "runs/named-run" in output
    assert '"num_samples": 3' in output


def test_run_mcmc_cli_uses_default_yaml_config(monkeypatch, capsys):
    calls = {}

    monkeypatch.setattr(cli, "load_augmented_butterfly_tree", lambda path, **kwargs: object())

    def fake_run_mcmc(dataset, *, run_dir, model_config, driver_config):
        calls["run_dir"] = run_dir
        calls["model_config"] = model_config
        calls["driver_config"] = driver_config
        return SimpleNamespace(run_dir=Path(run_dir), summary={})

    monkeypatch.setattr(cli, "run_mcmc", fake_run_mcmc)
    monkeypatch.setattr(cli, "plot_artifact_traces", lambda *args, **kwargs: None)

    assert cli.main([]) == 0

    assert calls["run_dir"] == Path("runs/butterflies")
    assert calls["model_config"].num_edge_steps == 25
    assert calls["model_config"].k_alpha_init is None
    assert calls["model_config"].k_sigma_init is None
    assert calls["model_config"].obs_var_init is None
    assert not hasattr(calls["model_config"], "normalize_el")
    assert calls["driver_config"].num_samples == 5000
    assert calls["driver_config"].num_chains == 4
    assert not hasattr(calls["driver_config"], "num_burnin")
    assert not hasattr(calls["driver_config"], "thin")
    assert calls["driver_config"].pcn_eta == 0.9
    assert calls["driver_config"].chain_backend == "process"
    assert calls["driver_config"].num_processes == 0
    assert not hasattr(calls["driver_config"], "infer_obs_var")
    assert "runs/butterflies" in capsys.readouterr().out


@pytest.mark.parametrize(
    "config_path",
    [
        Path("config/butterflies.yaml"),
        Path("config/beaks.yaml"),
    ],
)
def test_default_yaml_files_expose_all_mcmc_config_fields(config_path):
    config = cli._load_run_config(config_path)

    assert set(config["model"]) == set(cli.MODEL_CONFIG_KEYS)
    assert set(config["driver"]) == set(cli.DRIVER_CONFIG_KEYS)


def test_run_mcmc_cli_rejects_unknown_driver_config_keys():
    with pytest.raises(ValueError, match="Unknown driver config keys: \\['not_a_driver_key'\\]"):
        cli._driver_config_from_mapping({"not_a_driver_key": 0.3})


def test_run_mcmc_cli_parses_initial_param_config_values():
    model_config = cli._model_config_from_mapping(
        {
            "k_alpha_init": 0.12,
            "k_sigma_init": [0.21, 0.22],
            "obs_var_init": None,
        }
    )

    assert model_config.k_alpha_init == 0.12
    assert model_config.k_sigma_init == (0.21, 0.22)
    assert model_config.obs_var_init is None


def test_run_mcmc_cli_calls_trace_plots_from_yaml_plot_config(monkeypatch, tmp_path):
    calls = {}

    monkeypatch.setattr(cli, "load_augmented_butterfly_tree", lambda path, **kwargs: object())

    def fake_run_mcmc(dataset, *, run_dir, model_config, driver_config):
        return SimpleNamespace(run_dir=Path(run_dir), summary={})

    def fake_plot_artifact_traces(artifact_path, *, num_burnin, thin):
        calls["artifact_path"] = artifact_path
        calls["num_burnin"] = num_burnin
        calls["thin"] = thin

    monkeypatch.setattr(cli, "run_mcmc", fake_run_mcmc)
    monkeypatch.setattr(cli, "plot_artifact_traces", fake_plot_artifact_traces)

    config_path = tmp_path / "mcmc.yaml"
    config_path.write_text(
        "h5: data/butterflies/data.h5\n"
        "run_name: plotted-run\n"
        "model: {}\n"
        "driver: {}\n"
        "plot:\n"
        "  num_burnin: 4\n"
        "  thin: 3\n",
        encoding="utf-8",
    )

    assert cli.main(["--config", str(config_path)]) == 0

    assert calls == {
        "artifact_path": Path("runs/plotted-run/artifacts.h5"),
        "num_burnin": 4,
        "thin": 3,
    }


def test_run_mcmc_cli_uses_yaml_h5_and_run_name_when_not_overridden(monkeypatch, tmp_path):
    calls = {}

    def fake_load(path, *, remove_lmk=None):
        calls["h5_path"] = path
        calls["remove_lmk"] = remove_lmk
        return object()

    def fake_run_mcmc(dataset, *, run_dir, model_config, driver_config):
        calls["run_dir"] = run_dir
        return SimpleNamespace(run_dir=Path(run_dir), summary={})

    monkeypatch.setattr(cli, "load_augmented_butterfly_tree", fake_load)
    monkeypatch.setattr(cli, "run_mcmc", fake_run_mcmc)
    monkeypatch.setattr(cli, "plot_artifact_traces", lambda *args, **kwargs: None)

    config_path = tmp_path / "mcmc.yaml"
    config_path.write_text(
        "h5: data/butterflies/data.h5\n"
        "run_name: config-run\n"
        "model: {}\n"
        "driver: {}\n",
        encoding="utf-8",
    )

    assert cli.main(["--config", str(config_path)]) == 0

    assert calls["h5_path"] == Path("data/butterflies/data.h5")
    assert calls["remove_lmk"] is None
    assert calls["run_dir"] == Path("runs/config-run")


def test_run_mcmc_cli_applies_gpu_visibility_from_yaml(monkeypatch, tmp_path):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    monkeypatch.setattr(cli, "load_augmented_butterfly_tree", lambda path, **kwargs: object())
    monkeypatch.setattr(
        cli,
        "run_mcmc",
        lambda dataset, *, run_dir, model_config, driver_config: SimpleNamespace(
            run_dir=Path(run_dir),
            summary={},
        ),
    )
    monkeypatch.setattr(cli, "plot_artifact_traces", lambda *args, **kwargs: None)

    config_path = tmp_path / "mcmc.yaml"
    config_path.write_text(
        "h5: data/butterflies/data.h5\n"
        "run_name: gpu-run\n"
        "gpu_visible: 1\n"
        "model: {}\n"
        "driver: {}\n",
        encoding="utf-8",
    )

    assert cli.main(["--config", str(config_path)]) == 0

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_run_mcmc_cli_accepts_all_yaml_config_fields_as_options():
    actions = cli._build_parser()._option_string_actions
    expected_options = {
        "-h",
        "--help",
        "--config",
        *{f"--{name.replace('_', '-')}" for name in cli.TOP_LEVEL_CONFIG_KEYS},
        *{f"--{name.replace('_', '-')}" for name in cli.MODEL_CONFIG_KEYS},
        *{f"--{name.replace('_', '-')}" for name in cli.DRIVER_CONFIG_KEYS},
        *{f"--{name.replace('_', '-')}" for name in cli.PLOT_CONFIG_KEYS},
        *{f"--{name.replace('_', '-')}" for name in cli.AUGMENT_CONFIG_KEYS},
    }

    assert set(actions) == expected_options
    assert "--k-alpha-init" in actions
    assert "--k-sigma-init" in actions
    assert "--obs-var-init" in actions
    assert "--gpu-visible" in actions
    assert "--init-k-alpha" not in actions
    assert "--init-k-sigma" not in actions
    assert "--init-obs-var" not in actions


def test_run_mcmc_cli_overrides_yaml_fields_from_cli(monkeypatch, tmp_path):
    calls = {}

    class FakeTree:
        def __getitem__(self, name):
            if name == "coords":
                return SimpleNamespace(shape=(850, 116, 2))
            raise KeyError(name)

    def fake_load(path, *, remove_lmk=None):
        calls["h5_path"] = path
        calls["remove_lmk"] = remove_lmk
        return SimpleNamespace(tree=FakeTree(), removed_landmarks=tuple(remove_lmk or ()))

    def fake_run_mcmc(dataset, *, run_dir, model_config, driver_config):
        calls["run_dir"] = run_dir
        calls["model_config"] = model_config
        calls["driver_config"] = driver_config
        return SimpleNamespace(run_dir=Path(run_dir), summary={})

    def fake_plot_artifact_traces(artifact_path, *, num_burnin, thin):
        calls["plot"] = {"artifact_path": artifact_path, "num_burnin": num_burnin, "thin": thin}

    monkeypatch.setattr(cli, "load_augmented_butterfly_tree", fake_load)
    monkeypatch.setattr(cli, "run_mcmc", fake_run_mcmc)
    monkeypatch.setattr(cli, "plot_artifact_traces", fake_plot_artifact_traces)

    config_path = tmp_path / "mcmc.yaml"
    config_path.write_text(
        "h5: config-only.h5\n"
        "run_name: config-run\n"
        "augment:\n"
        "  remove_lmk: null\n"
        "# keep model comments when CLI overrides persist\n"
        "model:\n"
        "  num_edge_steps: 2\n"
        "  covar_jitter: 1e-8\n"
        "  dist_jitter: 1e-7\n"
        "  k_alpha_prior_alpha: 4.0\n"
        "  k_alpha_prior_beta: 0.2\n"
        "  k_alpha_init: null\n"
        "  k_alpha_max: 0.9\n"
        "  k_alpha_min: 1e-4\n"
        "  k_sigma_prior_alpha: 5.0\n"
        "  k_sigma_prior_beta: 0.3\n"
        "  k_sigma_init: null\n"
        "  k_sigma_max: 0.7\n"
        "  k_sigma_min: 2e-4\n"
        "  obs_var_prior_alpha: 3.0\n"
        "  obs_var_prior_beta: 0.004\n"
        "  obs_var_init: null\n"
        "  obs_var_max: 0.2\n"
        "  obs_var_min: 2e-5\n"
        "  enable_x64: true\n"
        "driver:\n"
        "  num_samples: 3\n"
        "  num_chains: 2\n"
        "  pcn_eta: 0.8\n"
        "  k_alpha_proposal_var: 0.01\n"
        "  k_sigma_proposal_var: 0.02\n"
        "  obs_var_proposal_var: 0.03\n"
        "  random_seed: 99\n"
        "  chain_backend: sequential\n"
        "  num_processes: 0\n"
        "  progress_bar: true\n"
        "plot:\n"
        "  num_burnin: 1\n"
        "  thin: 2\n",
        encoding="utf-8",
    )

    cli_args = [
        "--config",
        str(config_path),
        "--h5",
        str(tmp_path / "override.h5"),
        "--run-name",
        "cli-run",
        "--num-edge-steps",
        "7",
        "--covar-jitter",
        "2e-8",
        "--dist-jitter",
        "3e-7",
        "--k-alpha-prior-alpha",
        "6.0",
        "--k-alpha-prior-beta",
        "0.6",
        "--k-alpha-init",
        "[0.11, 0.12, 0.13, 0.14, 0.15]",
        "--k-alpha-max",
        "0.8",
        "--k-alpha-min",
        "2e-5",
        "--k-sigma-prior-alpha",
        "7.0",
        "--k-sigma-prior-beta",
        "0.7",
        "--k-sigma-init",
        "0.2",
        "--k-sigma-max",
        "0.6",
        "--k-sigma-min",
        "3e-5",
        "--obs-var-prior-alpha",
        "8.0",
        "--obs-var-prior-beta",
        "0.008",
        "--obs-var-init",
        "null",
        "--obs-var-max",
        "0.3",
        "--obs-var-min",
        "4e-5",
        "--enable-x64",
        "false",
        "--num-samples",
        "9",
        "--num-chains",
        "5",
        "--pcn-eta",
        "0.75",
        "--k-alpha-proposal-var",
        "0.04",
        "--k-sigma-proposal-var",
        "0.05",
        "--obs-var-proposal-var",
        "0.06",
        "--random-seed",
        "123",
        "--chain-backend",
        "process",
        "--num-processes",
        "3",
        "--progress-bar",
        "false",
        "--num-burnin",
        "4",
        "--thin",
        "6",
        "--remove-lmk",
        "[1, 3]",
    ]

    assert cli.main(cli_args) == 0

    assert calls["h5_path"] == tmp_path / "override.h5"
    assert calls["remove_lmk"] == (1, 3)
    assert calls["run_dir"] == Path("runs/cli-run")
    assert calls["model_config"].num_edge_steps == 7
    assert calls["model_config"].covar_jitter == 2e-8
    assert calls["model_config"].dist_jitter == 3e-7
    assert calls["model_config"].k_alpha_prior_alpha == 6.0
    assert calls["model_config"].k_alpha_prior_beta == 0.6
    assert calls["model_config"].k_alpha_init == (0.11, 0.12, 0.13, 0.14, 0.15)
    assert calls["model_config"].k_alpha_max == 0.8
    assert calls["model_config"].k_alpha_min == 2e-5
    assert calls["model_config"].k_sigma_prior_alpha == 7.0
    assert calls["model_config"].k_sigma_prior_beta == 0.7
    assert calls["model_config"].k_sigma_init == 0.2
    assert calls["model_config"].k_sigma_max == 0.6
    assert calls["model_config"].k_sigma_min == 3e-5
    assert calls["model_config"].obs_var_prior_alpha == 8.0
    assert calls["model_config"].obs_var_prior_beta == 0.008
    assert calls["model_config"].obs_var_init is None
    assert calls["model_config"].obs_var_max == 0.3
    assert calls["model_config"].obs_var_min == 4e-5
    assert calls["model_config"].enable_x64 is False
    assert calls["driver_config"].num_samples == 9
    assert calls["driver_config"].num_chains == 5
    assert calls["driver_config"].pcn_eta == 0.75
    assert calls["driver_config"].k_alpha_proposal_var == 0.04
    assert calls["driver_config"].k_sigma_proposal_var == 0.05
    assert calls["driver_config"].obs_var_proposal_var == 0.06
    assert calls["driver_config"].random_seed == 123
    assert calls["driver_config"].chain_backend == "process"
    assert calls["driver_config"].num_processes == 3
    assert calls["driver_config"].progress_bar is False
    assert calls["plot"] == {
        "artifact_path": Path("runs/cli-run/artifacts.h5"),
        "num_burnin": 4,
        "thin": 6,
    }

    persisted_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert persisted_config["h5"] == str(tmp_path / "override.h5")
    assert persisted_config["run_name"] == "cli-run"
    assert persisted_config["augment"]["remove_lmk"] == [1, 3]
    assert persisted_config["model"]["num_edge_steps"] == 7
    assert persisted_config["model"]["covar_jitter"] == 2e-8
    assert persisted_config["model"]["enable_x64"] is False
    assert persisted_config["model"]["k_alpha_init"] == [0.11, 0.12, 0.13, 0.14, 0.15]
    assert persisted_config["model"]["k_sigma_init"] == 0.2
    assert persisted_config["model"]["obs_var_init"] is None
    assert persisted_config["driver"]["num_samples"] == 9
    assert persisted_config["driver"]["chain_backend"] == "process"
    assert persisted_config["driver"]["progress_bar"] is False
    assert persisted_config["plot"] == {"num_burnin": 4, "thin": 6}
    assert "# keep model comments when CLI overrides persist" in config_path.read_text(encoding="utf-8")
