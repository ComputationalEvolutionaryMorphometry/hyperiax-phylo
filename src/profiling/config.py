"""Read-only YAML configuration and deterministic experiment expansion."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import yaml

from src.profiling.diagnostics import diagnostic_settings

ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return value


def _positive_int(value, label: str, *, zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f"{label} must be a {'non-negative' if zero else 'positive'} integer.")


def load_config(path: str | Path) -> dict:
    config = deepcopy(read_yaml(resolve_path(path)))
    config.setdefault("smoke", False)
    config.setdefault("seed", 0)
    config.setdefault("gpu_index", 0)
    config.setdefault("cpu_affinity", None)
    config.setdefault("cpu_threads", 8)
    config.setdefault("output_root", "runs/profiling")
    config.setdefault("experiments", ["full", "scaling", "reference"])
    config.setdefault("reference_landmarks", [32, 64, 118])
    config.setdefault("reference_dataset", "butterflies")
    config.setdefault("model_overrides", {})
    config.setdefault("primary_backend", "gpu")
    config["full_diagnostics"] = diagnostic_settings(config)
    if config["primary_backend"] not in {"cpu", "gpu"}:
        raise ValueError("primary_backend must be cpu or gpu.")
    if not config["experiments"] or set(config["experiments"]) - {"full", "scaling", "reference"}:
        raise ValueError("experiments must contain full, scaling and/or reference.")
    defaults = {
        "repeats": 3, "target_warmup": 10, "target_iterations": 200,
        "mcmc_warmup": 20, "mcmc_iterations": 200, "min_measure_seconds": 10,
        "timeout_seconds": 1800, "memory_interval_seconds": 0.05,
    }
    config["benchmark"] = defaults | config.get("benchmark", {})
    if set(config["benchmark"]) - set(defaults):
        raise ValueError("Unknown benchmark configuration keys.")
    config["full"] = {"num_chains": 4, "num_samples": 5000, "num_processes": 4,
                      "timeout_seconds": 21600} | config.get("full", {})
    if set(config["full"]) - {"num_chains", "num_samples", "num_processes", "timeout_seconds"}:
        raise ValueError("Unknown full analysis configuration keys.")
    if not isinstance(config["smoke"], bool):
        raise ValueError("smoke must be a YAML boolean.")
    for field in ("repeats", "target_iterations", "mcmc_iterations", "timeout_seconds"):
        _positive_int(config["benchmark"][field], f"benchmark.{field}")
    for field in ("target_warmup", "mcmc_warmup"):
        _positive_int(config["benchmark"][field], f"benchmark.{field}", zero=True)
    for field in ("num_chains", "num_samples", "num_processes", "timeout_seconds"):
        _positive_int(config["full"][field], f"full.{field}")
    _positive_int(config["seed"], "seed", zero=True)
    _positive_int(config["gpu_index"], "gpu_index", zero=True)
    _positive_int(config["cpu_threads"], "cpu_threads")
    for field, allow_zero in (("min_measure_seconds", True), ("memory_interval_seconds", False)):
        value = config["benchmark"][field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
            raise ValueError(f"Invalid benchmark.{field}.")
    if config["cpu_affinity"] is not None:
        if not isinstance(config["cpu_affinity"], list) or not config["cpu_affinity"]:
            raise ValueError("cpu_affinity must be null or a non-empty list of CPU indices.")
        for value in config["cpu_affinity"]:
            _positive_int(value, "cpu_affinity", zero=True)
    if not isinstance(config.get("datasets"), dict) or not config["datasets"]:
        raise ValueError("datasets must be a non-empty mapping.")
    for name, dataset in config["datasets"].items():
        for field in ("tips", "landmarks"):
            values = dataset.get(field)
            if not isinstance(values, list) or not values or len(set(values)) != len(values):
                raise ValueError(f"{name}.{field} must be a non-empty list without duplicates.")
            for value in values:
                _positive_int(value, f"{name}.{field}")
            if min(values) < 2:
                raise ValueError(f"{name}.{field} must be at least 2.")
            dataset[field] = sorted(values)
        analysis = read_yaml(resolve_path(dataset["analysis_config"]))
        dataset["h5"] = str(resolve_path(analysis["h5"]))
        dataset["model"] = analysis.get("model", {}) | config["model_overrides"]
        dataset["driver"] = analysis.get("driver", {})
        params = dataset.get("params", {})
        if set(params) != {"k_alpha", "k_sigma", "obs_var"} or any(
            not isinstance(v, (float, int)) or not 0 < v < float("inf") for v in params.values()
        ):
            raise ValueError(f"{name}.params must give three finite positive parameters.")
    if "reference" in config["experiments"]:
        dataset = config["datasets"].get(config["reference_dataset"])
        if dataset is None:
            raise ValueError("reference_dataset must name a configured dataset.")
        for count in config["reference_landmarks"]:
            _positive_int(count, "reference_landmarks")
            if not 2 <= count <= max(dataset["landmarks"]):
                raise ValueError("reference_landmarks exceeds the configured data size.")
    return config


def scaling_cases(dataset: dict) -> list[dict]:
    """Nine unique settings for two five-point curves sharing one endpoint."""
    cases: dict[tuple[int, int], dict] = {}
    for axis in ("tips", "landmarks"):
        for count in dataset[axis]:
            tips = count if axis == "tips" else max(dataset["tips"])
            landmarks = count if axis == "landmarks" else max(dataset["landmarks"])
            case = cases.setdefault((tips, landmarks), {"tips": tips, "landmarks": landmarks, "axes": []})
            case["axes"].append(axis)
    return list(cases.values())


def expand_jobs(config: dict) -> list[dict]:
    jobs = []
    for name, dataset in config["datasets"].items():
        if "full" in config["experiments"]:
            jobs.append(dict(dataset=name, experiment="full", tips=max(dataset["tips"]),
                             landmarks=max(dataset["landmarks"]), axes=[], implementation="hyperiax",
                             backend=config["primary_backend"], repeat=0, seed=config["seed"]))
        if "scaling" in config["experiments"]:
            for case in scaling_cases(dataset):
                for repeat in range(config["benchmark"]["repeats"]):
                    jobs.append(dict(case, dataset=name, experiment="scaling", implementation="hyperiax",
                                     backend=config["primary_backend"], repeat=repeat, seed=config["seed"] + repeat))
    if "reference" in config["experiments"]:
        name = config["reference_dataset"]
        dataset = config["datasets"][name]
        variants = [("hyperiax", config["primary_backend"]), ("reference", config["primary_backend"])]
        if config["primary_backend"] != "cpu":
            variants.append(("hyperiax", "cpu"))
        for landmarks in config["reference_landmarks"]:
            for implementation, backend in variants:
                for repeat in range(config["benchmark"]["repeats"]):
                    jobs.append(dict(dataset=name, experiment="reference", tips=max(dataset["tips"]),
                                     landmarks=landmarks, axes=[], implementation=implementation,
                                     backend=backend, repeat=repeat, seed=config["seed"] + repeat))
    for index, job in enumerate(jobs):
        job["job_id"] = f"job_{index:04d}"
    return jobs
