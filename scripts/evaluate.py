"""Evaluate an MCMC run, diagnostics, and standard run-level plots.

Example:
    uv run python -m scripts.evaluate --artifacts-path runs/butterflies/artifacts.h5 --config config/butterflies.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import yaml

PARAMETER_NAMES = (
    "k_alpha",
    "k_sigma",
    "obs_var",
)
PLOT_PARAMETER_NAMES = (
    "k_alpha",
    "k_sigma",
    # "obs_var",
)
PARAMETER_LABELS = {
    "k_alpha": r"$k_\alpha$",
    "k_sigma": r"$k_\sigma$",
    "obs_var": r"$\sigma^2_{\mathrm{obs}}$",
}
MAX_LEAVES_TO_PLOT = 16
GRID_ROWS = 4
GRID_COLUMNS = 4
UNCERTAINTY_Z = 1.96
GELMAN_RUBIN_THRESHOLD = 1.1
LEAVES_OUTPUT_FILENAME = "leaves.png"
ROOT_OUTPUT_FILENAME = "root.png"
TRACE_OUTPUT_FILENAME = "trace.png"
HIST_OUTPUT_FILENAME = "hist.png"
DEFAULT_ROOT_CHAIN = "chain_000"
ROOT_POSTERIOR_SAMPLE_COUNT = 120
DEFAULT_CONFIG_PATH = Path("config/butterflies.yaml")
DEFAULT_GPU_VISIBLE = None


@dataclass(frozen=True)
class TracePlotConfig:
    """Post-processing controls for artifact plots."""

    num_burnin: int = 0
    thin: int = 1


@dataclass(frozen=True)
class ConditionalLeafSamples:
    """Observed and conditionally sampled leaf coordinates in HDF5 leaf order."""

    leaf_names: tuple[str, ...]
    sample_leaf_coords: np.ndarray
    observed_leaf_coords: np.ndarray


@dataclass(frozen=True)
class EvaluationResult:
    """Filesystem and parameter summary produced by the evaluate CLI."""

    leaves_path: Path
    root_path: Path
    trace_path: Path
    hist_path: Path
    parameter_means: dict[str, float]
    gelman_rubin: dict[str, dict[str, object]] = field(default_factory=dict)

    def __iter__(self):
        return iter((self.leaves_path, self.root_path, self.trace_path, self.hist_path))


def load_parameter_means(
    artifacts_h5: str | Path,
    *,
    num_burnin: int,
    thin: int,
) -> dict[str, float]:
    """Return after-burnin/thinned means for MCMC scalar parameters."""

    config = TracePlotConfig(num_burnin=num_burnin, thin=thin)
    _validate_plot_config(config)
    artifacts_h5 = Path(artifacts_h5)

    means = {}
    with h5py.File(artifacts_h5, "r") as h5:
        for parameter_name in PARAMETER_NAMES:
            group_path = f"samples/{parameter_name}"
            if group_path not in h5:
                raise ValueError(f"{group_path} not found in {artifacts_h5}.")

            selected_chains = []
            for chain_name in sorted(h5[group_path]):
                values = h5[f"{group_path}/{chain_name}"][:].astype(np.float64, copy=False)
                selected_chains.append(
                    _post_burnin_thinned(values, num_burnin=config.num_burnin, thin=config.thin)
                )
            if not selected_chains:
                raise ValueError(f"{group_path} contains no chains.")
            selected_values = np.concatenate(selected_chains)
            means[parameter_name] = float(selected_values.mean())

    return means


def load_gelman_rubin_diagnostics(
    artifacts_h5: str | Path,
    *,
    num_burnin: int,
    thin: int,
    threshold: float = GELMAN_RUBIN_THRESHOLD,
) -> dict[str, dict[str, object]]:
    """Return Gelman-Rubin R-hat diagnostics for scalar parameter traces."""

    config = TracePlotConfig(num_burnin=num_burnin, thin=thin)
    _validate_plot_config(config)
    if threshold <= 0.0 or not np.isfinite(threshold):
        raise ValueError(f"threshold must be a finite positive scalar, got {threshold}.")

    parameter_chains = _load_parameter_chains(Path(artifacts_h5))
    return {
        parameter_name: _gelman_rubin_diagnostic(
            chains,
            num_burnin=config.num_burnin,
            thin=config.thin,
            threshold=threshold,
        )
        for parameter_name, chains in parameter_chains.items()
    }


def evaluate_artifacts(
    *,
    artifacts_h5: str | Path,
    data_h5: str | Path,
    num_burnin: int,
    thin: int,
) -> EvaluationResult:
    """Run the full artifact evaluation and write standard run figures."""

    artifacts_h5 = Path(artifacts_h5)
    data_h5 = Path(data_h5)
    output_dir = artifacts_h5.parent
    parameter_means = load_parameter_means(artifacts_h5, num_burnin=num_burnin, thin=thin)
    gelman_rubin = load_gelman_rubin_diagnostics(artifacts_h5, num_burnin=num_burnin, thin=thin)
    run_config = _load_run_config_for_artifact(artifacts_h5)
    from src.bffg import MCMCParams

    model_config = _model_config_from_run_config(run_config)
    conditional = sample_conditional_leaf_shapes(
        data_h5,
        params=MCMCParams(**parameter_means),
        model_config=model_config,
        remove_lmk=_remove_lmk_from_run_config(run_config),
    )
    leaves_path = _write_leaf_evaluation_plot(
        output_dir / LEAVES_OUTPUT_FILENAME,
        leaf_names=conditional.leaf_names,
        sample_leaf_coords=conditional.sample_leaf_coords,
        observed_leaf_coords=conditional.observed_leaf_coords,
        obs_var=parameter_means["obs_var"],
        parameter_means=parameter_means,
    )
    root_path = plot_root_shape(
        artifacts_h5,
        num_burnin=num_burnin,
        thin=thin,
        output_path=output_dir / ROOT_OUTPUT_FILENAME,
    )
    trace_path, hist_path = plot_artifact_traces(
        artifacts_h5,
        num_burnin=num_burnin,
        thin=thin,
        trace_path=output_dir / TRACE_OUTPUT_FILENAME,
        hist_path=output_dir / HIST_OUTPUT_FILENAME,
    )
    return EvaluationResult(
        leaves_path=leaves_path,
        root_path=root_path,
        trace_path=trace_path,
        hist_path=hist_path,
        parameter_means=parameter_means,
        gelman_rubin=gelman_rubin,
    )


def sample_conditional_leaf_shapes(
    data_h5: str | Path,
    *,
    params: MCMCParams,
    model_config: MCMCModelConfig | None = None,
    remove_lmk: object = None,
) -> ConditionalLeafSamples:
    """Run one BFFG down conditional sample and return leaf endpoints."""

    import jax

    from src.bffg import MCMCModelConfig, build_bffg_context, run_mcmc_sweeps
    from src.loader import load_augmented_butterfly_tree
    from src.tree_state import BFFG_FIELDS

    model_config = model_config or MCMCModelConfig()
    dataset = load_augmented_butterfly_tree(data_h5, remove_lmk=remove_lmk)
    context = build_bffg_context(dataset, model_config)
    noise_template = context.tree[BFFG_FIELDS.zs]
    noise = jax.random.normal(
        jax.random.PRNGKey(0),
        noise_template.shape,
        dtype=noise_template.dtype,
    )
    guided = run_mcmc_sweeps(
        context.tree,
        context.leaf_observations,
        params=_runtime_params_for_sampling(params, model_config),
        noise=noise,
    )

    leaf_indices = np.asarray(context.leaf_indices, dtype=np.int64)
    sample_states = np.asarray(guided[BFFG_FIELDS.vals])[leaf_indices, -1]
    sample_leaf_coords = sample_states.reshape((leaf_indices.size, context.n_landmarks, context.d_landmarks))
    observed_leaf_coords = np.asarray(context.leaf_observations).reshape(
        (leaf_indices.size, context.n_landmarks, context.d_landmarks)
    )
    leaf_names = tuple(dataset.node_names[int(index)] for index in leaf_indices)

    return ConditionalLeafSamples(
        leaf_names=leaf_names,
        sample_leaf_coords=sample_leaf_coords,
        observed_leaf_coords=observed_leaf_coords,
    )


def _runtime_params_for_sampling(params: MCMCParams, config: MCMCModelConfig) -> dict[str, object]:
    payload = params.as_dict()
    payload["covar_jitter"] = config.covar_jitter
    payload["dist_jitter"] = config.dist_jitter
    return payload


def _load_run_config_for_artifact(artifacts_h5: Path) -> dict:
    config_path = artifacts_h5.parent / "config.json"
    if not config_path.exists():
        return {}

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a JSON object.")
    return payload


def _model_config_from_run_config(payload: dict) -> MCMCModelConfig:
    from src.bffg import MCMCModelConfig

    model_payload = payload.get("model", {})
    if not isinstance(model_payload, dict):
        raise ValueError("config.json field 'model' must be a mapping.")
    return MCMCModelConfig(**model_payload)


def _remove_lmk_from_run_config(payload: dict):
    augment_payload = payload.get("augment", {})
    if augment_payload in ({}, None):
        return None
    if not isinstance(augment_payload, dict):
        raise ValueError("config.json field 'augment' must be a mapping.")
    return augment_payload.get("remove_lmk")


def _write_leaf_evaluation_plot(
    output_path: str | Path,
    *,
    leaf_names: tuple[str, ...],
    sample_leaf_coords: np.ndarray,
    observed_leaf_coords: np.ndarray,
    obs_var: float,
    parameter_means: dict[str, float],
) -> Path:
    """Write a 4x4 plot comparing conditional leaf samples against observations."""

    sample_leaf_coords = np.asarray(sample_leaf_coords, dtype=np.float64)
    observed_leaf_coords = np.asarray(observed_leaf_coords, dtype=np.float64)
    _validate_leaf_plot_inputs(leaf_names, sample_leaf_coords, observed_leaf_coords, obs_var)

    _configure_matplotlib()
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if sample_leaf_coords.shape[2] == 3:
        _write_leaf_evaluation_plot_3d(
            output_path,
            plt,
            leaf_names=leaf_names,
            sample_leaf_coords=sample_leaf_coords,
            observed_leaf_coords=observed_leaf_coords,
            obs_var=obs_var,
            parameter_means=parameter_means,
        )
    else:
        _write_leaf_evaluation_plot_2d(
            output_path,
            plt,
            leaf_names=leaf_names,
            sample_leaf_coords=sample_leaf_coords,
            observed_leaf_coords=observed_leaf_coords,
            obs_var=obs_var,
            parameter_means=parameter_means,
        )
    return output_path


def _write_leaf_evaluation_plot_2d(
    output_path: Path,
    plt,
    *,
    leaf_names: tuple[str, ...],
    sample_leaf_coords: np.ndarray,
    observed_leaf_coords: np.ndarray,
    obs_var: float,
    parameter_means: dict[str, float],
) -> None:
    import seaborn as sns
    from matplotlib.patches import Ellipse

    sigma_radius = UNCERTAINTY_Z * float(np.sqrt(obs_var))
    visible_count = min(MAX_LEAVES_TO_PLOT, len(leaf_names))
    fig, axes = plt.subplots(
        GRID_ROWS,
        GRID_COLUMNS,
        figsize=(12.0, 12.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    flat_axes = tuple(axes.ravel())
    x_limits, y_limits = _shared_leaf_axis_limits(
        sample_leaf_coords[:visible_count],
        observed_leaf_coords[:visible_count],
        sigma_radius=sigma_radius,
    )

    for axis_index, axis in enumerate(flat_axes):
        if axis_index >= visible_count:
            axis.set_visible(False)
            continue

        sample = sample_leaf_coords[axis_index]
        observed = observed_leaf_coords[axis_index]
        for center in sample:
            axis.add_patch(
                Ellipse(
                    xy=(float(center[0]), float(center[1])),
                    width=2.0 * sigma_radius,
                    height=2.0 * sigma_radius,
                    angle=0.0,
                    facecolor=(31 / 255, 78 / 255, 121 / 255, 0.12),
                    edgecolor=(31 / 255, 78 / 255, 121 / 255, 0.24),
                    linewidth=0.35,
                    zorder=1,
                )
            )
        axis.plot(sample[:, 0], sample[:, 1], color="#1f4e79", linewidth=0.75, alpha=0.78, zorder=2)
        axis.scatter(sample[:, 0], sample[:, 1], s=1.0, color="#1f4e79", label="conditional sample", zorder=3)
        axis.plot(observed[:, 0], observed[:, 1], color="#d55e00", linewidth=0.65, alpha=0.62, zorder=4)
        axis.scatter(
            observed[:, 0],
            observed[:, 1],
            s=1.0,
            color="#d55e00",
            label="observation",
            alpha=0.82,
            zorder=5,
        )
        _style_shape_axis(axis, title="")
        axis.set_title(leaf_names[axis_index], fontsize=9)
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        if axis_index == 0:
            axis.legend(loc="upper right", frameon=False, fontsize=6)

    fig.suptitle(_leaf_evaluation_title(parameter_means), x=0.01, ha="left", fontsize=12)
    sns.despine(fig=fig, trim=False)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _write_leaf_evaluation_plot_3d(
    output_path: Path,
    plt,
    *,
    leaf_names: tuple[str, ...],
    sample_leaf_coords: np.ndarray,
    observed_leaf_coords: np.ndarray,
    obs_var: float,
    parameter_means: dict[str, float],
) -> None:
    sigma_radius = UNCERTAINTY_Z * float(np.sqrt(obs_var))
    visible_count = min(MAX_LEAVES_TO_PLOT, len(leaf_names))
    fig, axes = plt.subplots(
        GRID_ROWS,
        GRID_COLUMNS,
        figsize=(12.0, 12.0),
        subplot_kw={"projection": "3d"},
    )
    flat_axes = tuple(axes.ravel())
    axis_limits = _shared_leaf_axis_limits_3d(
        sample_leaf_coords[:visible_count],
        observed_leaf_coords[:visible_count],
        sigma_radius=sigma_radius,
    )
    blob_size = _noise_blob_marker_size_3d(axis_limits, sigma_radius=sigma_radius)

    for axis_index, axis in enumerate(flat_axes):
        if axis_index >= visible_count:
            axis.set_visible(False)
            continue

        sample = sample_leaf_coords[axis_index]
        observed = observed_leaf_coords[axis_index]
        axis.scatter(
            sample[:, 0],
            sample[:, 1],
            sample[:, 2],
            s=blob_size,
            color="#1f4e79",
            alpha=0.12,
            linewidths=0,
            depthshade=True,
            label="observation noise",
            zorder=1,
        )
        axis.plot(sample[:, 0], sample[:, 1], sample[:, 2], color="#1f4e79", linewidth=0.75, alpha=0.78, zorder=2)
        axis.scatter(
            sample[:, 0],
            sample[:, 1],
            sample[:, 2],
            s=3.5,
            color="#1f4e79",
            label="conditional sample",
            zorder=3,
        )
        axis.plot(
            observed[:, 0],
            observed[:, 1],
            observed[:, 2],
            color="#d55e00",
            linewidth=0.65,
            alpha=0.62,
            zorder=4,
        )
        axis.scatter(
            observed[:, 0],
            observed[:, 1],
            observed[:, 2],
            s=3.5,
            color="#d55e00",
            label="observation",
            alpha=0.82,
            zorder=5,
        )
        _style_shape_axis_3d(axis, title="")
        axis.set_title(leaf_names[axis_index], fontsize=9)
        axis.set_xlim(axis_limits[0])
        axis.set_ylim(axis_limits[1])
        axis.set_zlim(axis_limits[2])
        if axis_index == 0:
            axis.legend(loc="upper right", frameon=False, fontsize=6)

    fig.suptitle(_leaf_evaluation_title(parameter_means), x=0.01, ha="left", fontsize=12)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _leaf_evaluation_title(parameter_means: dict[str, float]) -> str:
    return (
        "BFFG conditional leaf sample vs observed leaf shape "
        f"(k_alpha={parameter_means['k_alpha']:.3f}, "
        f"k_sigma={parameter_means['k_sigma']:.3f}, "
        f"obs_var={parameter_means['obs_var']:.3f})"
    )


def _validate_leaf_plot_inputs(
    leaf_names: tuple[str, ...],
    sample_leaf_coords: np.ndarray,
    observed_leaf_coords: np.ndarray,
    obs_var: float,
) -> None:
    if sample_leaf_coords.shape != observed_leaf_coords.shape:
        raise ValueError(
            "sample_leaf_coords and observed_leaf_coords must have the same shape; "
            f"got {sample_leaf_coords.shape} and {observed_leaf_coords.shape}."
        )
    if sample_leaf_coords.ndim != 3 or sample_leaf_coords.shape[2] not in (2, 3):
        raise ValueError(
            "leaf coordinates must have shape (n_leaves, n_landmarks, 2 or 3); "
            f"got {sample_leaf_coords.shape}."
        )
    if len(leaf_names) != sample_leaf_coords.shape[0]:
        raise ValueError(f"Expected {sample_leaf_coords.shape[0]} leaf names, got {len(leaf_names)}.")
    if sample_leaf_coords.shape[0] == 0:
        raise ValueError("At least one leaf is required.")
    if obs_var <= 0.0 or not np.isfinite(obs_var):
        raise ValueError(f"obs_var must be a finite positive scalar, got {obs_var}.")
    if not np.isfinite(sample_leaf_coords).all():
        raise ValueError("sample_leaf_coords contains NaN or infinite values.")
    if not np.isfinite(observed_leaf_coords).all():
        raise ValueError("observed_leaf_coords contains NaN or infinite values.")


def _shared_leaf_axis_limits(
    sample_leaf_coords: np.ndarray,
    observed_leaf_coords: np.ndarray,
    *,
    sigma_radius: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    values = np.vstack(
        (
            sample_leaf_coords.reshape((-1, 2)),
            observed_leaf_coords.reshape((-1, 2)),
        )
    )
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    span = max(float(np.max(maxs - mins)), 1e-6)
    pad = max(0.08 * span, sigma_radius * 1.1, 1e-3)
    return (
        (float(mins[0] - pad), float(maxs[0] + pad)),
        (float(maxs[1] + pad), float(mins[1] - pad)),
    )


def _shared_leaf_axis_limits_3d(
    sample_leaf_coords: np.ndarray,
    observed_leaf_coords: np.ndarray,
    *,
    sigma_radius: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    values = np.vstack(
        (
            sample_leaf_coords.reshape((-1, 3)),
            observed_leaf_coords.reshape((-1, 3)),
        )
    )
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    span = max(float(np.max(maxs - mins)), 1e-6)
    pad = max(0.08 * span, sigma_radius * 1.1, 1e-3)
    return (
        (float(mins[0] - pad), float(maxs[0] + pad)),
        (float(mins[1] - pad), float(maxs[1] + pad)),
        (float(mins[2] - pad), float(maxs[2] + pad)),
    )


def _noise_blob_marker_size_3d(
    axis_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    *,
    sigma_radius: float,
) -> float:
    spans = [abs(upper - lower) for lower, upper in axis_limits]
    span = max(max(spans), 1e-6)
    radius_fraction = sigma_radius / span
    return float(np.clip((radius_fraction * 1800.0) ** 2, 18.0, 720.0))


def plot_artifact_traces(
    artifact_path: str | Path,
    *,
    num_burnin: int = 0,
    thin: int = 1,
    trace_path: str | Path | None = None,
    hist_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Read an artifacts HDF5 file and write parameter trace and histogram plots."""

    config = TracePlotConfig(num_burnin=num_burnin, thin=thin)
    _validate_plot_config(config)
    artifact_path = Path(artifact_path)
    parameter_chains = _load_parameter_chains(artifact_path)
    trace_path = artifact_path.parent / TRACE_OUTPUT_FILENAME if trace_path is None else Path(trace_path)
    hist_path = artifact_path.parent / HIST_OUTPUT_FILENAME if hist_path is None else Path(hist_path)

    _write_trace_plot(trace_path, parameter_chains, num_burnin=config.num_burnin)
    _write_hist_plot(hist_path, parameter_chains, num_burnin=config.num_burnin, thin=config.thin)
    return trace_path, hist_path


def _load_parameter_chains(artifact_path: Path) -> dict[str, dict[str, np.ndarray]]:
    with h5py.File(artifact_path, "r") as h5:
        parameter_chains = {}
        for parameter_name in PARAMETER_NAMES:
            group_path = f"samples/{parameter_name}"
            if group_path not in h5:
                raise ValueError(f"{group_path} not found in {artifact_path}.")
            parameter_chains[parameter_name] = {
                chain_name: h5[f"{group_path}/{chain_name}"][:].astype(np.float64, copy=False)
                for chain_name in sorted(h5[group_path])
            }
    return parameter_chains


def _gelman_rubin_diagnostic(
    chains: dict[str, np.ndarray],
    *,
    num_burnin: int,
    thin: int,
    threshold: float,
) -> dict[str, object]:
    selected_chains = []
    for chain_name in sorted(chains):
        values = _post_burnin_thinned(chains[chain_name], num_burnin=num_burnin, thin=thin)
        if not np.isfinite(values).all():
            raise ValueError(f"{chain_name} contains NaN or infinite parameter samples.")
        selected_chains.append(values)

    num_chains = len(selected_chains)
    num_samples = min((values.size for values in selected_chains), default=0)
    diagnostic = {
        "r_hat": None,
        "passed": None,
        "threshold": float(threshold),
        "num_chains": int(num_chains),
        "num_samples_per_chain": int(num_samples),
    }
    if num_chains < 2:
        diagnostic["reason"] = "Gelman-Rubin requires at least two chains."
        return diagnostic
    if num_samples < 2:
        diagnostic["reason"] = "Gelman-Rubin requires at least two retained samples per chain."
        return diagnostic
    if any(values.size != num_samples for values in selected_chains):
        sizes = [int(values.size) for values in selected_chains]
        raise ValueError(f"Gelman-Rubin requires equal length chains after burn-in/thinning, got {sizes}.")

    r_hat = _gelman_rubin_r_hat(np.stack(selected_chains))
    diagnostic["r_hat"] = float(r_hat)
    diagnostic["passed"] = bool(np.isfinite(r_hat) and r_hat <= threshold)
    return diagnostic


def _gelman_rubin_r_hat(chains: np.ndarray) -> float:
    num_samples = chains.shape[1]
    chain_means = chains.mean(axis=1)
    within_chain_var = float(np.var(chains, axis=1, ddof=1).mean())
    between_chain_var = float(num_samples * np.var(chain_means, ddof=1))
    if within_chain_var == 0.0:
        return 1.0 if between_chain_var == 0.0 else float("inf")

    pooled_var = ((num_samples - 1.0) / num_samples) * within_chain_var
    pooled_var += between_chain_var / num_samples
    return float(np.sqrt(max(pooled_var / within_chain_var, 0.0)))


def _write_trace_plot(
    output_path: Path,
    parameter_chains: dict[str, dict[str, np.ndarray]],
    *,
    num_burnin: int,
) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chain_names = _chain_names(parameter_chains)
    colors = _chain_colors(chain_names)
    fig, axes = plt.subplots(len(PLOT_PARAMETER_NAMES), 1, figsize=(7.2, 4.4), sharex=True)
    axes = np.atleast_1d(axes)

    for axis_index, (axis, parameter_name) in enumerate(zip(axes, PLOT_PARAMETER_NAMES, strict=True)):
        chains = parameter_chains[parameter_name]
        for chain_name in chain_names:
            values = chains[chain_name]
            x = np.arange(values.shape[0])
            color = colors[chain_name]
            mean_value = _post_burnin(values, num_burnin).mean()
            axis.plot(x, values, color=color, alpha=0.52, linewidth=1.35, label="_nolegend_")
            axis.axhline(
                mean_value,
                color=color,
                linewidth=1.45,
                alpha=0.96,
                label=chain_name,
            )
        _draw_burnin_marker(axis, num_burnin, label=axis_index == 0)
        _set_trace_x_limits(axis, chains)
        axis.set_ylabel(_parameter_label(parameter_name))
        axis.grid(True, axis="y", linewidth=0.55, alpha=0.28)
        axis.grid(False, axis="x")
        sns.despine(ax=axis, trim=False)

    axes[-1].set_xlabel("iteration")
    _add_shared_chain_legend(fig, axes, chain_names)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _write_hist_plot(
    output_path: Path,
    parameter_chains: dict[str, dict[str, np.ndarray]],
    *,
    num_burnin: int,
    thin: int,
) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chain_names = _chain_names(parameter_chains)
    colors = _chain_colors(chain_names)
    fig, axes = plt.subplots(len(PLOT_PARAMETER_NAMES), 1, figsize=(7.2, 4.4))
    axes = np.atleast_1d(axes)

    for axis, parameter_name in zip(axes, PLOT_PARAMETER_NAMES, strict=True):
        chains = parameter_chains[parameter_name]
        histogram_values = []
        for chain_name in chain_names:
            values = _post_burnin_thinned(chains[chain_name], num_burnin=num_burnin, thin=thin)
            histogram_values.append(values)
            color = colors[chain_name]
            _draw_density_histogram(axis, values, color=color)
            kde_x, kde_y = _kde_pdf_estimate(values)
            axis.plot(kde_x, kde_y, color=color, linewidth=2.2, label=chain_name)
            _draw_hist_mean_marker(axis, float(values.mean()), color=color)
        _set_hist_x_limits(axis, histogram_values)
        axis.set_title(_parameter_label(parameter_name), loc="left", fontsize=10, pad=4)
        axis.set_ylabel("density")
        axis.grid(True, axis="y", linewidth=0.55, alpha=0.28)
        axis.grid(False, axis="x")
        sns.despine(ax=axis, trim=False)

    axes[-1].set_xlabel("parameter value")
    _add_shared_chain_legend(fig, axes, chain_names)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _draw_density_histogram(axis, values: np.ndarray, *, color: tuple[float, float, float]) -> None:
    density, bin_edges = np.histogram(values, bins="auto", density=True)
    axis.step(
        bin_edges,
        np.append(density, density[-1]),
        where="post",
        color=color,
        alpha=0.22,
        linewidth=1.4,
        label="_nolegend_",
    )


def _draw_hist_mean_marker(axis, mean_value: float, *, color: tuple[float, float, float]) -> None:
    axis.axvline(
        mean_value,
        color=color,
        linestyle="--",
        linewidth=1.25,
        alpha=0.9,
        label="_mean",
        zorder=3,
    )


def _kde_pdf_estimate(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_min = float(values.min())
    x_max = float(values.max())
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5

    support = np.linspace(x_min, x_max, 256)
    bandwidth = _kde_bandwidth(values)
    scaled = (support[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled**2).sum(axis=1)
    density /= values.size * bandwidth * np.sqrt(2.0 * np.pi)
    return support, density


def _kde_bandwidth(values: np.ndarray) -> float:
    if values.size < 2:
        return 1.0
    std = float(np.std(values, ddof=1))
    if std <= 0:
        return 1.0
    return 1.06 * std * values.size ** (-1.0 / 5.0)


def _set_hist_x_limits(axis, values_by_chain: list[np.ndarray]) -> None:
    values = np.concatenate(values_by_chain)
    x_min = float(values.min())
    x_max = float(values.max())
    if x_min == x_max:
        axis.set_xlim(x_min - 0.5, x_max + 0.5)
        return
    axis.set_xlim(x_min, x_max)


def _draw_burnin_marker(axis, num_burnin: int, *, label: bool = True) -> None:
    if num_burnin <= 0:
        return
    axis.axvline(num_burnin, color="0.25", linestyle="--", linewidth=1.0, alpha=0.78)
    if not label:
        return
    axis.annotate(
        f"burn-in = {num_burnin}",
        xy=(num_burnin, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4, -5),
        textcoords="offset points",
        color="0.25",
        fontsize=8,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
    )


def _parameter_label(parameter_name: str) -> str:
    return PARAMETER_LABELS.get(parameter_name, parameter_name)


def _set_trace_x_limits(axis, chains: dict[str, np.ndarray]) -> None:
    max_num_samples = max(values.shape[0] for values in chains.values())
    if max_num_samples <= 1:
        axis.set_xlim(0, 1)
        return
    axis.set_xlim(0, max_num_samples - 1)


def _chain_names(parameter_chains: dict[str, dict[str, np.ndarray]]) -> tuple[str, ...]:
    return tuple(sorted(next(iter(parameter_chains.values()))))


def _chain_colors(chain_names: tuple[str, ...]) -> dict[str, tuple[float, float, float]]:
    import seaborn as sns

    palette = sns.color_palette("colorblind", n_colors=len(chain_names))
    return dict(zip(chain_names, palette, strict=True))


def _add_shared_chain_legend(fig, axes: np.ndarray, chain_names: tuple[str, ...]) -> None:
    if len(chain_names) <= 1:
        return

    handles_by_label = {}
    for line in axes[0].lines:
        label = line.get_label()
        if label in chain_names:
            handles_by_label[label] = line
    handles = [handles_by_label[chain_name] for chain_name in chain_names]
    fig.legend(
        handles,
        chain_names,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=len(chain_names),
        frameon=False,
        fontsize=8,
    )


def _post_burnin(values: np.ndarray, num_burnin: int) -> np.ndarray:
    sliced = values[num_burnin:]
    if sliced.size == 0:
        raise ValueError("num_burnin leaves no samples to plot.")
    return sliced


def _post_burnin_thinned(values: np.ndarray, *, num_burnin: int, thin: int) -> np.ndarray:
    return _post_burnin(values, num_burnin)[::thin]


def _validate_plot_config(config: TracePlotConfig) -> None:
    if config.num_burnin < 0:
        raise ValueError(f"num_burnin must be >= 0, got {config.num_burnin}.")
    if config.thin < 1:
        raise ValueError(f"thin must be >= 1, got {config.thin}.")


def plot_root_shape(
    artifact_path: str | Path,
    *,
    num_burnin: int = 0,
    thin: int = 1,
    output_path: str | Path | None = None,
) -> Path:
    """Read phylo-root samples and plot the posterior root shape summary."""

    config = TracePlotConfig(num_burnin=num_burnin, thin=thin)
    _validate_plot_config(config)
    artifact_path = Path(artifact_path)
    output_path = artifact_path.parent / ROOT_OUTPUT_FILENAME if output_path is None else Path(output_path)

    samples = _load_phylo_root_chain(artifact_path, DEFAULT_ROOT_CHAIN)
    selected_samples = _post_burnin_thinned(samples, num_burnin=config.num_burnin, thin=config.thin)
    _write_root_shape_plot(output_path, selected_samples, chain_name=DEFAULT_ROOT_CHAIN)
    return output_path


def _load_phylo_root_chain(artifact_path: Path, chain_name: str) -> np.ndarray:
    dataset_path = f"samples/phylo_root/{chain_name}"
    with h5py.File(artifact_path, "r") as h5:
        if dataset_path not in h5:
            raise ValueError(f"{dataset_path} not found in {artifact_path}.")
        samples = h5[dataset_path][:].astype(np.float64, copy=False)

    if samples.ndim != 3 or samples.shape[2] not in (2, 3):
        raise ValueError(f"Expected {dataset_path} shape (n_iters, n_landmarks, 2 or 3), got {samples.shape}.")
    return samples


def _write_root_shape_plot(output_path: Path, samples: np.ndarray, *, chain_name: str) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean, lower, upper = _root_summary(samples)
    if samples.shape[2] == 3:
        _write_root_shape_plot_3d(output_path, plt, mean=mean, lower=lower, upper=upper, chain_name=chain_name)
        return

    _write_root_shape_plot_2d(output_path, plt, samples=samples, mean=mean)


def _write_root_shape_plot_2d(
    output_path: Path,
    plt,
    *,
    samples: np.ndarray,
    mean: np.ndarray,
) -> None:
    fig, axis = plt.subplots(figsize=(6.8, 4.3), constrained_layout=False)
    for sample in _posterior_shape_samples_for_plot(samples):
        axis.plot(
            sample[:, 0],
            sample[:, 1],
            color="#4f7fa6",
            linewidth=0.55,
            alpha=0.11,
            zorder=1,
        )
    axis.scatter(
        mean[:, 0],
        mean[:, 1],
        s=8.0,
        color="#123f5f",
        edgecolors="white",
        linewidths=0.25,
        zorder=3,
    )
    _style_root_shape_hero_axis(axis, mean, samples)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _posterior_shape_samples_for_plot(samples: np.ndarray) -> np.ndarray:
    if samples.shape[0] <= ROOT_POSTERIOR_SAMPLE_COUNT:
        return samples
    indices = np.linspace(0, samples.shape[0] - 1, ROOT_POSTERIOR_SAMPLE_COUNT, dtype=np.int64)
    return samples[indices]


def _write_root_shape_plot_3d(
    output_path: Path,
    plt,
    *,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    chain_name: str,
) -> None:
    fig = plt.figure(figsize=(5.4, 5.0))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(mean[:, 0], mean[:, 1], mean[:, 2], color="#1f4e79", linewidth=0.8, alpha=0.72, zorder=2)
    axis.scatter(
        mean[:, 0],
        mean[:, 1],
        mean[:, 2],
        s=13.0,
        color="#1f4e79",
        edgecolors="white",
        linewidths=0.25,
        zorder=3,
    )
    _draw_credible_intervals_3d(axis, mean=mean, lower=lower, upper=upper)
    _style_shape_axis_3d(axis, title=f"Phylogenetic root shape ({chain_name})")
    _set_3d_axis_limits(axis, mean, lower, upper)
    _draw_credible_interval_note_3d(axis)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _root_summary(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if samples.size == 0:
        raise ValueError("No phylo_root samples remain after burn-in/thinning.")
    mean = samples.mean(axis=0)
    lower = np.quantile(samples, 0.025, axis=0)
    upper = np.quantile(samples, 0.975, axis=0)
    return mean, lower, upper


def _style_root_shape_hero_axis(axis, mean: np.ndarray, samples: np.ndarray) -> None:
    values = np.concatenate([samples.reshape(-1, samples.shape[-1]), mean], axis=0)
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


def _draw_credible_intervals_3d(axis, *, mean: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> None:
    for center, lo, hi in zip(mean, lower, upper, strict=True):
        axis.plot(
            [lo[0], hi[0]],
            [center[1], center[1]],
            [center[2], center[2]],
            color="#1f4e79",
            alpha=0.34,
            linewidth=0.55,
            zorder=1,
        )
        axis.plot(
            [center[0], center[0]],
            [lo[1], hi[1]],
            [center[2], center[2]],
            color="#1f4e79",
            alpha=0.34,
            linewidth=0.55,
            zorder=1,
        )
        axis.plot(
            [center[0], center[0]],
            [center[1], center[1]],
            [lo[2], hi[2]],
            color="#1f4e79",
            alpha=0.34,
            linewidth=0.55,
            zorder=1,
        )


def _set_3d_axis_limits(axis, *arrays: np.ndarray) -> None:
    values = np.vstack([array.reshape((-1, 3)) for array in arrays])
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    span = max(float(np.max(maxs - mins)), 1e-6)
    pad = max(0.08 * span, 1e-3)
    axis.set_xlim(float(mins[0] - pad), float(maxs[0] + pad))
    axis.set_ylim(float(mins[1] - pad), float(maxs[1] + pad))
    axis.set_zlim(float(mins[2] - pad), float(maxs[2] + pad))


def _draw_credible_interval_note_3d(axis) -> None:
    axis.text2D(
        0.99,
        0.015,
        "95% marginal credible intervals",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.28",
    )


def _style_shape_axis(axis, *, title: str) -> None:
    axis.set_title(title, loc="left", fontsize=10)
    axis.set_xlabel("scaled x")
    axis.set_ylabel("scaled y")
    axis.set_aspect("equal", adjustable="box")
    axis.invert_yaxis()
    axis.grid(False)


def _style_shape_axis_3d(axis, *, title: str) -> None:
    axis.set_title(title, loc="left", fontsize=10)
    axis.set_xlabel("scaled x")
    axis.set_ylabel("scaled y")
    axis.set_zlabel("scaled z")
    axis.grid(False)
    axis.view_init(elev=22, azim=-62)


def _configure_matplotlib() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import seaborn as sns

    sns.set_theme(
        context="paper",
        style="white",
        palette="colorblind",
        font_scale=1.0,
        rc={
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "font.family": "serif",
            "font.size": 9,
            "mathtext.fontset": "stix",
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "xtick.direction": "out",
            "ytick.direction": "out",
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-path", type=Path, required=True, help="Path to a run artifacts.h5 file.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML run config containing h5 and plot settings. Default: {DEFAULT_CONFIG_PATH}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = _load_evaluation_config(args.config)
    _apply_gpu_visibility(config.get("gpu_visible", DEFAULT_GPU_VISIBLE))
    plot_config = _trace_plot_config_from_mapping(config.get("plot", {}))
    result = evaluate_artifacts(
        artifacts_h5=args.artifacts_path,
        data_h5=_data_path_from_config(config),
        num_burnin=plot_config.num_burnin,
        thin=plot_config.thin,
    )
    for output_path in result:
        print(output_path)
    print(
        json.dumps(
            {
                "parameter_means": result.parameter_means,
                "gelman_rubin": result.gelman_rubin,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_evaluation_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Evaluation config must be a YAML mapping, got {type(config).__name__}.")
    return config


def _data_path_from_config(config: dict) -> Path:
    if "h5" not in config:
        raise ValueError("Evaluation config must define top-level 'h5'.")
    return Path(config["h5"])


def _apply_gpu_visibility(value: object) -> None:
    gpu_index = _coerce_gpu_visible_config(value)
    if gpu_index is None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _coerce_gpu_visible_config(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            raise ValueError("gpu_visible must be null or a single non-negative GPU index.")
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "none", "null"}:
            return None
        if stripped.isdecimal():
            return int(stripped)
    raise ValueError("gpu_visible must be null or a single non-negative GPU index.")


def _trace_plot_config_from_mapping(config: object) -> TracePlotConfig:
    if config is None:
        return TracePlotConfig()
    if not isinstance(config, dict):
        raise ValueError("Evaluation config section 'plot' must be a mapping.")
    unknown = sorted(set(config).difference({"num_burnin", "thin"}))
    if unknown:
        raise ValueError(f"Unknown plot config keys: {unknown}")
    values = {}
    if "num_burnin" in config:
        values["num_burnin"] = int(config["num_burnin"])
    if "thin" in config:
        values["thin"] = int(config["thin"])
    return TracePlotConfig(**values)


if __name__ == "__main__":
    raise SystemExit(main())
