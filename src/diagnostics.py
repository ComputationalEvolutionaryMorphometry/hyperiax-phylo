"""Post-hoc parameter/root diagnostics; no JAX, sampling, or data-tree loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import h5py
import numpy as np

PARAMETERS = ("k_alpha", "k_sigma", "obs_var")
METRICS = ("r_hat", "ess_bulk", "ess_tail", "mcse_mean", "mcse_mean_over_sd")


@dataclass(frozen=True)
class DiagnosticConfig:
    rhat_threshold: float = 1.01
    ess_threshold: float = 400.0
    mcse_ratio_threshold: float = 0.05
    num_pcs: int = 3

    def __post_init__(self):
        for name in ("rhat_threshold", "ess_threshold", "mcse_ratio_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"diagnostics.{name} must be finite and positive.")
        if self.rhat_threshold <= 1:
            raise ValueError("diagnostics.rhat_threshold must exceed 1.")
        if isinstance(self.num_pcs, bool) or not isinstance(self.num_pcs, int) or not 1 <= self.num_pcs <= 3:
            raise ValueError("diagnostics.num_pcs must be an integer between 1 and 3.")


def diagnostic_config(values: dict | None = None) -> DiagnosticConfig:
    values = {} if values is None else values
    if not isinstance(values, dict) or set(values) - set(DiagnosticConfig.__dataclass_fields__):
        raise ValueError("Unknown or malformed diagnostics configuration.")
    return DiagnosticConfig(**values)


def _unavailable(reason: str) -> dict:
    return dict.fromkeys(METRICS) | {"status": "unavailable", "passed": None, "reason": reason}


def diagnose_scalar(values: np.ndarray, config: DiagnosticConfig) -> dict:
    """Diagnose unthinned (chain, retained draw) arrays with ArviZ 0.22 methods."""
    import arviz as az

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        return _unavailable("Expected (chain, draw) scalar samples.")
    metadata = {"num_chains": values.shape[0], "retained_draws_per_chain": values.shape[1]}
    reason = None
    if values.shape[0] < 2 or values.shape[1] < 4:
        reason = "Need at least two chains and four retained draws per chain."
    elif not np.isfinite(values).all():
        reason = "Non-finite retained samples."
    elif np.any(np.ptp(values, axis=1) == 0):
        reason = "At least one chain is constant; not evidence of convergence."
    if reason:
        return _unavailable(reason) | metadata
    metrics = {
        "r_hat": float(az.rhat(values, method="rank")),
        "ess_bulk": float(az.ess(values, method="bulk")),
        "ess_tail": float(az.ess(values, method="tail", prob=(0.05, 0.95))),
        "mcse_mean": float(az.mcse(values, method="mean")),
    }
    metrics["mcse_mean_over_sd"] = metrics["mcse_mean"] / float(values.std(ddof=1))
    if not all(np.isfinite(value) and value >= 0 for value in metrics.values()):
        return _unavailable("Non-finite diagnostic estimator; inspect traces.") | metadata
    passed = (metrics["r_hat"] < config.rhat_threshold
              and min(metrics["ess_bulk"], metrics["ess_tail"]) >= config.ess_threshold
              and metrics["mcse_mean_over_sd"] <= config.mcse_ratio_threshold)
    return metadata | metrics | {"status": "computed", "passed": bool(passed),
                                "warnings": ["Fewer than 100 retained draws per chain."] if values.shape[1] < 100 else []}


def summarize(variables: dict, config: DiagnosticConfig) -> dict:
    rows = list(variables.values())
    valid = [row for row in rows if row["status"] == "computed"]
    result = {"variables": len(rows), "computed": len(valid), "unavailable": len(rows) - len(valid),
              "passed": all(row["passed"] for row in valid) if rows and len(valid) == len(rows) else None,
              "rhat_at_or_above_threshold": sum(row["r_hat"] >= config.rhat_threshold for row in valid),
              "bulk_ess_below_threshold": sum(row["ess_bulk"] < config.ess_threshold for row in valid),
              "tail_ess_below_threshold": sum(row["ess_tail"] < config.ess_threshold for row in valid),
              "aggregation_scope": "Computed variables only; unavailable variables never count as passed."}
    for metric in METRICS:
        values = [row[metric] for row in valid]
        result[metric] = ({"min": float(np.min(values)), "median": float(np.median(values)),
                           "p95": float(np.quantile(values, 0.95)), "max": float(np.max(values))}
                          if values else None)
    return result


def load_chains(artifact: h5py.File, location: str, num_burnin: int, *, root: bool = False):
    if location not in artifact or not isinstance(artifact[location], h5py.Group):
        raise ValueError(f"Missing chain group: {location}.")
    keys = sorted(artifact[location])
    if not keys or any(not key.startswith("chain_") for key in keys):
        raise ValueError(f"Missing or unexpected chain identifiers in {location}.")
    arrays = [np.asarray(artifact[location][key], dtype=np.float64) for key in keys]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError(f"Unequal chain shapes in {location}; chains are not truncated.")
    shape = arrays[0].shape
    if (root and (len(shape) != 3 or shape[1] < 1 or shape[2] not in (2, 3))) or (not root and len(shape) != 1):
        raise ValueError(f"Unexpected sample shape in {location}: {shape}.")
    return keys, np.stack([array[num_burnin:] for array in arrays])


def compute_diagnostics(path: Path, *, num_burnin: int, config: DiagnosticConfig | None = None):
    """Return JSON-ready diagnostics and selected root traces for rendering."""
    import arviz as az

    if isinstance(num_burnin, bool) or not isinstance(num_burnin, int) or num_burnin < 0:
        raise ValueError("num_burnin must be a non-negative integer.")
    config = config or DiagnosticConfig()
    report = {"schema_version": 1, "artifact_path": str(path), "arviz_version": az.__version__,
              "num_burnin": num_burnin, "thin": 1, "thresholds": asdict(config),
              "methods": {"r_hat": "rank-normalized split R-hat with folding (ArviZ rank)",
                          "ess_bulk": "bulk", "ess_tail": "tail (0.05, 0.95)", "mcse_mean": "mean"},
              "scope": "Scalar parameters and phylogenetic root, not every ancestral node.",
              "interpretation": "Screening diagnostics, not proof of convergence; ESS combines all chains.",
              "parameters": {"variables": {}}, "root": {"variables": {}},
              "pca": {"variables": {}, "reason": "Root samples unavailable."}, "plot_selection": []}
    selected = {}
    with h5py.File(path, "r") as artifact:
        for name in PARAMETERS:
            try:
                keys, values = load_chains(artifact, f"samples/{name}", num_burnin)
                row = diagnose_scalar(values, config) | {"chain_ids": keys}
            except (ValueError, TypeError, KeyError) as error:
                row = _unavailable(str(error))
            report["parameters"]["variables"][name] = row
        try:
            keys, root = load_chains(artifact, "samples/phylo_root", num_burnin, root=True)
        except (ValueError, TypeError, KeyError) as error:
            report["root"]["reason"] = str(error)
        else:
            chains, draws, landmarks, dimension = root.shape
            report["root"].update(chain_ids=keys, retained_draws_per_chain=draws,
                                  num_chains=chains, landmarks=landmarks, dimensions=dimension,
                                  coordinate_indexing="Zero-based landmark index in saved artifact; no additional alignment.")
            flat = root.reshape(chains, draws, landmarks * dimension)
            names = []
            for index in range(flat.shape[-1]):
                landmark, axis = divmod(index, dimension)
                name = f"landmark_{landmark:03d}_{'xyz'[axis]}"
                names.append(name)
                report["root"]["variables"][name] = diagnose_scalar(flat[:, :, index], config) | {
                    "landmark_index": landmark, "axis": "xyz"[axis]}
            variables = report["root"]["variables"]
            valid = [name for name in names if variables[name]["status"] == "computed"]
            for metric, reverse in (("r_hat", True), ("ess_bulk", False), ("ess_tail", False)):
                if valid:
                    name = sorted(valid, key=lambda n: variables[n][metric], reverse=reverse)[0]
                    if name not in selected:
                        selected[name] = flat[:, :, names.index(name)]
                        report["plot_selection"].append({"name": name, "reason": f"Worst {metric}"})
            # Include a few uncomputable traces rather than silently hiding them.
            for name in [name for name in names if variables[name]["status"] != "computed"][:3]:
                selected[name] = flat[:, :, names.index(name)]
                report["plot_selection"].append({"name": name, "reason": variables[name]["reason"]})
            if draws >= 4 and chains >= 2 and np.isfinite(flat).all():
                pooled = flat.reshape(-1, flat.shape[-1])
                mean = pooled.mean(axis=0)
                centered = pooled - mean
                _, singular, basis = np.linalg.svd(centered, full_matrices=False)
                rank = int(np.count_nonzero(singular > singular[0] * max(centered.shape) * np.finfo(float).eps))
                count = min(config.num_pcs, rank)
                if count:
                    basis = basis[:count]
                    scores = ((flat - mean) @ basis.T)
                    report["pca"] = {"variables": {}, "basis": basis.tolist(), "center": mean.tolist(),
                                     "explained_variance_ratio": (singular[:count] ** 2 / np.sum(singular ** 2)).tolist(),
                                     "method": "Shared pooled-centered SVD; no rescaling or per-chain alignment."}
                    for index in range(count):
                        name = f"PC{index + 1}"
                        report["pca"]["variables"][name] = diagnose_scalar(scores[:, :, index], config)
                        selected[name] = scores[:, :, index]
                        report["plot_selection"].append({"name": name, "reason": "Shared root PCA"})
                else:
                    report["pca"]["reason"] = "Root samples have zero numerical rank."
            else:
                report["pca"]["reason"] = "Need finite root samples, two chains and four retained draws."
    for name in ("parameters", "root", "pca"):
        report[name]["summary"] = summarize(report[name]["variables"], config)
    statuses = [report[name]["summary"]["passed"] for name in ("parameters", "root", "pca")]
    report["status"] = "passed" if all(v is True for v in statuses) else ("warning" if False in statuses else "unavailable")
    return report, selected


def write_diagnostic_plots(path: Path, traces: dict, report: dict) -> list[str]:
    """Trace/rank plots of worst coordinates and common PC scores; no subsampling."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from scipy.stats import rankdata

    if traces:
        fig, axes = plt.subplots(len(traces), 2, figsize=(11, 2.3 * len(traces)), squeeze=False, constrained_layout=True)
    else:
        fig, axis = plt.subplots(figsize=(10, 3), constrained_layout=True)
        axis.axis("off")
        axis.text(0.5, 0.5, "Root diagnostics unavailable\n" + report["root"].get("reason", "No root traces."),
                  ha="center", va="center", wrap=True, transform=axis.transAxes)
    chain_ids = report["root"].get("chain_ids", [])
    for row, (name, values) in enumerate(traces.items()):
        for chain, samples in enumerate(values):
            axes[row, 0].plot(np.arange(len(samples)) + report["num_burnin"], samples,
                              lw=0.7, alpha=0.7, label=chain_ids[chain], color=plt.get_cmap("tab10")(chain % 10))
        axes[row, 0].set(title=name, xlabel="Iteration (zero-based)", ylabel="Sample")
        if values.size and np.isfinite(values).all():
            ranks = rankdata(values.ravel()).reshape(values.shape)
            bins = np.linspace(0.5, values.size + 0.5, min(20, values.shape[1]) + 1)
            for chain, samples in enumerate(ranks):
                axes[row, 1].hist(samples, bins=bins, histtype="step", label=chain_ids[chain],
                                  color=plt.get_cmap("tab10")(chain % 10))
            axes[row, 1].axhline(values.shape[1] / (len(bins) - 1), color="gray", ls="--", lw=0.7)
        else:
            axes[row, 1].text(0.5, 0.5, "Rank plot unavailable", ha="center", transform=axes[row, 1].transAxes)
        axes[row, 1].set(title=f"{name}: pooled ranks", xlabel="Rank", ylabel="Count per chain")
    if traces:
        axes[0, 0].legend(fontsize="small")
    fig.suptitle("Root mixing diagnostics — worst coordinates and shared PCA (not proof of convergence)")
    outputs = []
    for suffix in (".png", ".pdf"):
        output = path.with_suffix(suffix)
        fig.savefig(output, dpi=160)
        outputs.append(str(output))
    plt.close(fig)
    return outputs


def print_diagnostics(report: dict, console=None) -> None:
    from rich.console import Console
    from rich.table import Table

    console = console or Console()
    table = Table(title="Parameter / root convergence diagnostics (unthinned)")
    for label in ("Scope", "Valid/total", "Max R-hat", "R-hat flagged", "Min bulk ESS", "Min tail ESS", "Max MCSE/SD"):
        table.add_column(label)
    def fmt(value):
        return "—" if value is None else f"{value:.4g}"
    for name in ("parameters", "root", "pca"):
        summary = report[name]["summary"]
        def metric(key, stat):
            return fmt(summary[key][stat]) if summary[key] else "—"
        table.add_row(name, f"{summary['computed']}/{summary['variables']}", metric("r_hat", "max"),
                      str(summary["rhat_at_or_above_threshold"]), metric("ess_bulk", "min"),
                      metric("ess_tail", "min"), metric("mcse_mean_over_sd", "max"))
        if "reason" in report[name]:
            console.print(f"{name}: {report[name]['reason']}", markup=False)
        reasons = sorted({row["reason"] for row in report[name]["variables"].values() if "reason" in row})
        for reason in reasons:
            console.print(f"{name}: {reason}", markup=False)
        if any(row.get("warnings") for row in report[name]["variables"].values()):
            console.print(f"{name}: fewer than 100 retained draws per chain; estimates may be unstable.", markup=False)
    console.print(table)
    console.print(f"Thresholds: {report['thresholds']}; burn-in/chain={report['num_burnin']}; thin=1.", markup=False)
    console.print(f"Screening status: {report['status']}; unavailable entries are not passes.", markup=False)


def evaluate_diagnostics(path: str | Path, *, num_burnin: int,
                         config: DiagnosticConfig | None = None, console=None) -> dict:
    path = Path(path)
    report, traces = compute_diagnostics(path, num_burnin=num_burnin, config=config)
    output = path.parent / "diagnostics.json"
    # Write numerical evidence before plotting so it survives a plotting failure.
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report["plots"] = write_diagnostic_plots(path.parent / "root_diagnostics", traces, report)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print_diagnostics(report, console)
    return report
