"""Run full-shape MCMC from a YAML config and write run artifacts.

Example:
    uv run python -m scripts.run_mcmc --config config/butterflies.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass, fields
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import yaml

from src.bffg import MCMCModelConfig
from src.driver import MCMCDriverConfig, run_mcmc
from src.loader import load_augmented_butterfly_tree
from scripts.evaluate import TracePlotConfig, plot_artifact_traces


DEFAULT_CONFIG_PATH = Path("config/butterflies.yaml")
DEFAULT_H5_PATH = Path("data/butterflies/data.h5")
DEFAULT_RUN_NAME = "default"
DEFAULT_GPU_VISIBLE = None
MODEL_CONFIG_KEYS = tuple(field.name for field in fields(MCMCModelConfig))
DRIVER_CONFIG_KEYS = tuple(field.name for field in fields(MCMCDriverConfig))
PLOT_CONFIG_KEYS = ("num_burnin", "thin")
AUGMENT_CONFIG_KEYS = ("remove_lmk",)
TOP_LEVEL_CONFIG_KEYS = ("h5", "run_name", "gpu_visible")
INIT_PARAM_CONFIG_KEYS = ("k_alpha_init", "k_sigma_init", "obs_var_init")
CONFIG_SECTION_KEYS = {
    "model": MODEL_CONFIG_KEYS,
    "driver": DRIVER_CONFIG_KEYS,
    "plot": PLOT_CONFIG_KEYS,
    "augment": AUGMENT_CONFIG_KEYS,
}
CLI_OVERRIDE_UNSET = object()


@dataclass(frozen=True)
class AugmentConfig:
    """Data augmentation/removal options applied immediately after HDF5 loading."""

    remove_lmk: tuple[int, ...] | None = None


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = _apply_cli_overrides(_load_run_config(args.config), args)
    override_paths = _cli_override_paths(args)
    if override_paths:
        _write_run_config(args.config, config, override_paths)
    _apply_gpu_visibility(config.get("gpu_visible", DEFAULT_GPU_VISIBLE))
    h5_path = Path(config.get("h5", DEFAULT_H5_PATH))
    run_name = str(config.get("run_name", DEFAULT_RUN_NAME))
    augment_config = _augment_config_from_mapping(config.get("augment", {}))
    dataset = load_augmented_butterfly_tree(h5_path, remove_lmk=augment_config.remove_lmk)
    _print_landmark_augment_summary(dataset)
    result = run_mcmc(
        dataset,
        run_dir=_default_run_dir(run_name),
        model_config=_model_config_from_mapping(config.get("model", {})),
        driver_config=_driver_config_from_mapping(config.get("driver", {})),
    )
    plot_config = _trace_plot_config_from_mapping(config.get("plot", {}))
    plot_artifact_traces(
        result.run_dir / "artifacts.h5",
        num_burnin=plot_config.num_burnin,
        thin=plot_config.thin,
    )
    print(f"run_dir: {result.run_dir}")
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML run config. Default: {DEFAULT_CONFIG_PATH}",
    )
    for key in TOP_LEVEL_CONFIG_KEYS:
        _add_config_override_arg(parser, section_name=None, key=key)
    for section_name, keys in CONFIG_SECTION_KEYS.items():
        for key in keys:
            _add_config_override_arg(parser, section_name=section_name, key=key)
    return parser


def _load_run_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Run config must be a YAML mapping, got {type(config).__name__}.")
    return config


def _write_run_config(config_path: Path, config: dict, override_paths: list[tuple[str | None, str]]) -> None:
    lines = config_path.read_text(encoding="utf-8").splitlines()
    for section_name, key in override_paths:
        value = config[key] if section_name is None else config[section_name][key]
        if section_name is None:
            _replace_or_append_top_level_yaml_value(lines, key, value)
        else:
            _replace_or_append_section_yaml_value(lines, section_name, key, value)
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _add_config_override_arg(parser: argparse.ArgumentParser, *, section_name: str | None, key: str) -> None:
    flag = f"--{key.replace('_', '-')}"
    parser.add_argument(
        flag,
        dest=_cli_override_dest(section_name, key),
        default=CLI_OVERRIDE_UNSET,
        metavar=key.upper(),
        type=_parse_cli_config_value,
        help=f"Override YAML config field {_config_field_path(section_name, key)!r}.",
    )


def _cli_override_dest(section_name: str | None, key: str) -> str:
    section = "top" if section_name is None else section_name
    return f"override__{section}__{key}"


def _config_field_path(section_name: str | None, key: str) -> str:
    if section_name is None:
        return key
    return f"{section_name}.{key}"


def _parse_cli_config_value(raw_value: str):
    try:
        return yaml.safe_load(raw_value)
    except yaml.YAMLError as error:
        raise argparse.ArgumentTypeError(f"Could not parse YAML value {raw_value!r}: {error}") from error


def _apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    merged = dict(config)
    for key in TOP_LEVEL_CONFIG_KEYS:
        value = getattr(args, _cli_override_dest(None, key))
        if value is not CLI_OVERRIDE_UNSET:
            merged[key] = _coerce_top_level_override_value(key, value)

    for section_name, keys in CONFIG_SECTION_KEYS.items():
        section_overrides = {}
        for key in keys:
            value = getattr(args, _cli_override_dest(section_name, key))
            if value is not CLI_OVERRIDE_UNSET:
                section_overrides[key] = _coerce_cli_override_value(section_name, key, value)
        if not section_overrides:
            continue
        section = merged.get(section_name)
        if section is None:
            section = {}
        if not isinstance(section, dict):
            raise ValueError(f"Config section {section_name!r} must be a mapping.")
        merged[section_name] = {**section, **section_overrides}
    return merged


def _cli_override_paths(args: argparse.Namespace) -> list[tuple[str | None, str]]:
    paths = []
    for key in TOP_LEVEL_CONFIG_KEYS:
        if getattr(args, _cli_override_dest(None, key)) is not CLI_OVERRIDE_UNSET:
            paths.append((None, key))
    for section_name, keys in CONFIG_SECTION_KEYS.items():
        for key in keys:
            if getattr(args, _cli_override_dest(section_name, key)) is not CLI_OVERRIDE_UNSET:
                paths.append((section_name, key))
    return paths


def _coerce_cli_override_value(section_name: str, key: str, value: object) -> object:
    if section_name == "model":
        if key in INIT_PARAM_CONFIG_KEYS:
            return _coerce_init_param_config(value, field_path=f"model.{key}")
        return _coerce_config_value(value, getattr(MCMCModelConfig, key))
    if section_name == "driver":
        return _coerce_config_value(value, getattr(MCMCDriverConfig, key))
    if section_name == "plot":
        return _coerce_config_value(value, getattr(TracePlotConfig, key))
    if section_name == "augment" and key == "remove_lmk":
        remove_lmk = _coerce_remove_lmk_config(value)
        return None if remove_lmk is None else list(remove_lmk)
    return value


def _coerce_top_level_override_value(key: str, value: object) -> object:
    if key in {"h5", "run_name"}:
        return str(value)
    if key == "gpu_visible":
        return _coerce_gpu_visible_config(value)
    return value


def _replace_or_append_top_level_yaml_value(lines: list[str], key: str, value: object) -> None:
    for index, line in enumerate(lines):
        if _is_yaml_key_line(line, key, indent=""):
            lines[index] = _yaml_assignment_line(key, value, indent="", original_line=line)
            return
    lines.append(_yaml_assignment_line(key, value, indent=""))


def _replace_or_append_section_yaml_value(lines: list[str], section_name: str, key: str, value: object) -> None:
    section_start = _find_top_level_section(lines, section_name)
    if section_start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"{section_name}:")
        lines.append(_yaml_assignment_line(key, value, indent="  "))
        return

    _ensure_block_yaml_section(lines, section_start, section_name)
    section_end = _find_section_end(lines, section_start)
    for index in range(section_start + 1, section_end):
        line = lines[index]
        if _is_yaml_key_line(line, key, indent="  "):
            indent = line[: len(line) - len(line.lstrip(" "))]
            lines[index] = _yaml_assignment_line(key, value, indent=indent, original_line=line)
            return
    lines.insert(section_end, _yaml_assignment_line(key, value, indent="  "))


def _find_top_level_section(lines: list[str], section_name: str) -> int | None:
    for index, line in enumerate(lines):
        if _is_yaml_key_line(line, section_name, indent=""):
            return index
    return None


def _ensure_block_yaml_section(lines: list[str], section_start: int, section_name: str) -> None:
    after_colon = lines[section_start].split(":", 1)[1].strip()
    if after_colon and not after_colon.startswith("#"):
        lines[section_start] = f"{section_name}:"


def _find_section_end(lines: list[str], section_start: int) -> int:
    for index in range(section_start + 1, len(lines)):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            return index
    return len(lines)


def _is_yaml_key_line(line: str, key: str, *, indent: str) -> bool:
    pattern = rf"^{re.escape(indent)}{re.escape(key)}\s*:"
    return re.match(pattern, line) is not None


def _yaml_assignment_line(key: str, value: object, *, indent: str, original_line: str | None = None) -> str:
    suffix = ""
    if original_line is not None:
        match = re.search(r"\s+#", original_line)
        if match is not None:
            suffix = original_line[match.start() :]
    return f"{indent}{key}: {_format_yaml_value(value)}{suffix}"


def _apply_gpu_visibility(value: object) -> None:
    gpu_index = _coerce_gpu_visible_config(value)
    if gpu_index is None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _coerce_gpu_visible_config(value: object) -> int | None:
    if value is None:
        return None
    if _is_non_bool_int(value):
        if value < 0:
            raise ValueError("gpu_visible must be null or a single non-negative GPU index.")
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "none", "null"}:
            return None
        if re.fullmatch(r"\d+", stripped):
            return int(stripped)
    raise ValueError("gpu_visible must be null or a single non-negative GPU index.")


def _format_yaml_value(value: object) -> str:
    dumped = yaml.safe_dump(value, default_flow_style=True, sort_keys=False).strip()
    lines = [line for line in dumped.splitlines() if line != "..."]
    return " ".join(lines)


def _model_config_from_mapping(config: object) -> MCMCModelConfig:
    values = _checked_config_section(config, section_name="model", allowed_keys=MODEL_CONFIG_KEYS)
    return MCMCModelConfig(**_typed_config_values(MCMCModelConfig, values))


def _driver_config_from_mapping(config: object) -> MCMCDriverConfig:
    values = _checked_config_section(config, section_name="driver", allowed_keys=DRIVER_CONFIG_KEYS)
    return MCMCDriverConfig(**_typed_config_values(MCMCDriverConfig, values))


def _trace_plot_config_from_mapping(config: object) -> TracePlotConfig:
    values = _checked_config_section(config, section_name="plot", allowed_keys=PLOT_CONFIG_KEYS)
    return TracePlotConfig(**_typed_config_values(TracePlotConfig, values))


def _augment_config_from_mapping(config: object) -> AugmentConfig:
    values = _checked_config_section(config, section_name="augment", allowed_keys=AUGMENT_CONFIG_KEYS)
    return AugmentConfig(remove_lmk=_coerce_remove_lmk_config(values.get("remove_lmk")))


def _checked_config_section(config: object, *, section_name: str, allowed_keys: tuple[str, ...]) -> dict:
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Config section {section_name!r} must be a mapping.")
    unknown = sorted(set(config).difference(allowed_keys))
    if unknown:
        raise ValueError(f"Unknown {section_name} config keys: {unknown}")
    return config


def _typed_config_values(config_class, values: dict) -> dict:
    defaults = {field.name: getattr(config_class, field.name) for field in fields(config_class)}
    return {
        name: _coerce_typed_config_value(config_class, name, value, defaults[name])
        for name, value in values.items()
    }


def _coerce_typed_config_value(config_class, name: str, value: object, default: object) -> object:
    if config_class is MCMCModelConfig and name in INIT_PARAM_CONFIG_KEYS:
        return _coerce_init_param_config(value, field_path=f"model.{name}")
    return _coerce_config_value(value, default)


def _coerce_remove_lmk_config(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    if _is_non_bool_int(value):
        return (int(value),)
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        indices = []
        for item in value:
            if not _is_non_bool_int(item):
                raise ValueError(f"augment.remove_lmk entries must be integer indices, got {item!r}.")
            indices.append(int(item))
        if len(set(indices)) != len(indices):
            raise ValueError("augment.remove_lmk must not contain duplicate landmark indices.")
        return tuple(sorted(indices))
    raise ValueError("augment.remove_lmk must be null, an integer index, or a list of integer indices.")


def _is_non_bool_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _coerce_init_param_config(value: object, *, field_path: str) -> float | tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(_coerce_init_param_scalar(item, field_path=field_path) for item in value)
    return _coerce_init_param_scalar(value, field_path=field_path)


def _coerce_init_param_scalar(value: object, *, field_path: str) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"{field_path} must be null, a finite positive scalar, or a list of finite positive scalars."
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_path} must be null, a finite positive scalar, or a list of finite positive scalars."
        ) from error
    if numeric <= 0.0 or not math.isfinite(numeric):
        raise ValueError(
            f"{field_path} must be null, a finite positive scalar, or a list of finite positive scalars."
        )
    return numeric


def _coerce_config_value(value, default):
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            if value.lower() in {"true", "yes", "1"}:
                return True
            if value.lower() in {"false", "no", "0"}:
                return False
        raise ValueError(f"Expected boolean config value, got {value!r}.")
    return type(default)(value)


def _print_landmark_augment_summary(dataset) -> None:
    if not getattr(dataset, "removed_landmarks", ()):
        return
    coords_shape = dataset.tree["coords"].shape
    remaining_landmarks = int(coords_shape[1])
    coordinate_dim = int(coords_shape[2])
    state_dim = remaining_landmarks * coordinate_dim
    print(
        "augment.remove_lmk: "
        f"skipped landmark indices {list(dataset.removed_landmarks)}; "
        f"remaining_landmarks: {remaining_landmarks}; "
        f"state_dim: {state_dim}"
    )


def _default_run_dir(run_name: str) -> Path:
    return Path("runs") / run_name


if __name__ == "__main__":
    raise SystemExit(main())
