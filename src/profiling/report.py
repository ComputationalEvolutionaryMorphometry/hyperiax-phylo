"""Aggregate JSON measurements, render Rich tables and export static figures."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import statistics

from rich.console import Console
from rich.table import Table

from src.profiling.artifacts import read_json, write_json

METRICS = ("target_first_call_seconds", "target_seconds", "mcmc_seconds", "process_wall_seconds",
           "analysis_seconds", "peak_rss_bytes", "peak_gpu_process_bytes")


def stats(values: list[float]) -> dict | None:
    if not values:
        return None
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "count": len(values)}


def summarize(records: list[dict], *, smoke: bool) -> dict:
    buckets = defaultdict(list)
    for record in records:
        key = tuple(record[name] for name in ("dataset", "experiment", "tips", "landmarks", "implementation", "backend"))
        buckets[key].append(record)
    groups = []
    for key, values in sorted(buckets.items()):
        group = dict(zip(("dataset", "experiment", "tips", "landmarks", "implementation", "backend"), key))
        good = [v for v in values if v["status"] == "ok"]
        group.update(nodes=values[0]["subset"]["nodes"], dimensions=values[0]["subset"]["dimensions"],
                     axes=values[0]["axes"], successful=len(good), total=len(values),
                     status="ok" if len(good) == len(values) else ("partial" if good else "failed"))
        for metric in METRICS:
            group[metric] = stats([v[metric] for v in good if v.get(metric) is not None])
        group["warnings"] = sorted({warning for v in good for warning in v.get("warnings", [])})
        group["errors"] = [v.get("error", v["status"]) for v in values if v["status"] != "ok"]
        groups.append(group)

    reference = {tuple(v[k] for k in ("dataset", "tips", "landmarks", "seed", "implementation", "backend")): v
                 for v in records if v["experiment"] == "reference" and v["status"] == "ok"}
    ratios = defaultdict(list)
    rejected_comparisons = []
    import numpy as np
    for key, value in reference.items():
        dataset, tips, landmarks, seed, impl, backend = key
        if impl != "hyperiax":
            continue
        candidates = [("traversal", "reference", backend)]
        if backend == "gpu":
            candidates.append(("hardware", "hyperiax", "cpu"))
        for kind, other_impl, other_backend in candidates:
            other = reference.get((dataset, tips, landmarks, seed, other_impl, other_backend))
            if other is None or (kind == "traversal" and "equivalence" not in other):
                continue
            try:
                for field in ("log_target", "root_state"):
                    np.testing.assert_allclose(other["check_output"][field], value["check_output"][field],
                                               rtol=1e-6, atol=1e-8)
            except AssertionError:
                rejected_comparisons.append(dict(dataset=dataset, tips=tips, landmarks=landmarks, seed=seed,
                                                  kind=kind, error="Numerical equivalence check failed."))
                continue
            ratios[(dataset, tips, landmarks, backend, kind)].append(other["target_seconds"] / value["target_seconds"])
    speedups = [dict(dataset=key[0], tips=key[1], landmarks=key[2], backend=key[3], kind=key[4], ratio=stats(values))
                for key, values in sorted(ratios.items())]
    full_analyses = [{"dataset": r["dataset"], "backend": r["backend"],
                      "process_wall_seconds": r["process_wall_seconds"], "analysis_seconds": r["analysis_seconds"],
                      "summary": r["analysis_summary"], "diagnostics": r.get("diagnostics")}
                     for r in records if r["experiment"] == "full" and r["status"] == "ok"]
    return {"schema_version": 1, "smoke": smoke, "units": {"time": "seconds", "memory": "bytes"},
            "jobs": len(records), "successful_jobs": sum(r["status"] == "ok" for r in records),
            "failed_jobs": sum(r["status"] != "ok" for r in records), "groups": groups,
            "speedups": speedups, "rejected_comparisons": rejected_comparisons, "full_analyses": full_analyses,
            "memory_scope": "Sampled process-tree RSS and NVML process allocator occupancy over whole worker lifetime, including compilation and reference validation."}


def _format(value, multiplier=1.0) -> str:
    if value is None:
        return "—"
    median = value["median"] * multiplier
    if value["count"] == 1:
        return f"{median:.3g}"
    return f"{median:.3g} [{value['min'] * multiplier:.3g}, {value['max'] * multiplier:.3g}]"


def print_summary(summary: dict, console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title="Smoke validation" if summary["smoke"] else "Performance experiments")
    for column in ("Case", "Dataset", "Experiment", "N / L / d", "Method / device", "OK/total"):
        table.add_column(column)
    for index, g in enumerate(summary["groups"], 1):
        table.add_row(f"C{index:02d}", g["dataset"], g["experiment"], f"{g['nodes']}/{g['landmarks']}/{g['dimensions']}",
                      f"{g['implementation']}/{g['backend']}", f"{g['successful']}/{g['total']}",
                      style="red" if g["status"] != "ok" else None)
    console.print(table)
    table = Table(title="Time: median (range across independent repeats)")
    for column in ("Case", "First call (s)", "Target (ms)", "MCMC (ms)", "Process wall (s)"):
        table.add_column(column)
    for index, g in enumerate(summary["groups"], 1):
        table.add_row(f"C{index:02d}", _format(g["target_first_call_seconds"]), _format(g["target_seconds"], 1000),
                      _format(g["mcmc_seconds"], 1000), _format(g["process_wall_seconds"]))
    console.print(table)
    table = Table(title="Observed process-tree memory peaks: median (range)")
    for column in ("Case", "RSS (MiB)", "GPU allocator (MiB)"):
        table.add_column(column)
    for index, g in enumerate(summary["groups"], 1):
        table.add_row(f"C{index:02d}", _format(g["peak_rss_bytes"], 1 / 2**20),
                      _format(g["peak_gpu_process_bytes"], 1 / 2**20))
    console.print(table)
    if summary["full_analyses"]:
        table = Table(title="Full analyses: overlapping chain clocks (do not sum)",
                      padding=(0, 0) if console.width < 100 else (0, 1))
        for column in ("Dataset", "Chain", "First target (s)", "Sampling (s)", "Iterations", "Acceptance", "Bulk ESS", "Tail ESS"):
            table.add_column(column, overflow="fold", min_width=11 if column == "Dataset" else None)
        for analysis in summary["full_analyses"]:
            info = analysis["summary"]
            chains = (analysis.get("diagnostics") or {}).get("chains", {})
            for index, timing in enumerate(info.get("chain_timings", [])):
                if timing is None:
                    continue
                chain = chains.get(f"chain_{index:03d}", {})
                ess = ["—" if chain.get(metric) is None else f"{chain[metric]:.4g}"
                       for metric in ("ess_bulk", "ess_tail")]
                table.add_row(analysis["dataset"], str(index + 1), f"{timing['initial_target_seconds']:.3g}",
                              f"{timing['measured_loop_seconds']:.3g}", str(timing["measured_iterations"]),
                              f"{info['acceptance_rates'][index]:.1%}", *ess)
        console.print(table)
        console.print("Per-chain ESS: minimum over k_alpha, k_sigma, obs_var after configured burn-in. "
                      "Estimated separately per chain; not a substitute for joint ESS/R-hat.", markup=False)
        diagnostics_table = Table(title="Full diagnostics: retained draws/chain; bulk ESS/s uses total job wall time")
        for column in ("Dataset", "Variable", "Draws", "Bulk ESS", "Tail ESS", "R-hat", "ESS/s"):
            diagnostics_table.add_column(column)
        messages = []
        for analysis in summary["full_analyses"]:
            diagnostics = analysis.get("diagnostics")
            if diagnostics is None:
                continue
            messages.append(f"{analysis['dataset']}: burn-in fraction={diagnostics['burnin_fraction']:g}, thin=1, "
                            f"diagnostics={diagnostics['status']} (scalar parameters and log target only).")
            messages.extend(f"{analysis['dataset']}: {warning}" for warning in diagnostics["warnings"])
            warning_variables = defaultdict(list)
            for name, row in diagnostics["variables"].items():
                metrics = ["—" if row[key] is None else f"{row[key]:.4g}"
                           for key in ("ess_bulk", "ess_tail", "r_hat", "ess_bulk_per_second")]
                diagnostics_table.add_row(analysis["dataset"], name, str(row.get("retained_draws_per_chain", "—")), *metrics)
                for warning in row["warnings"]:
                    warning_variables[warning].append(name)
            messages.extend(f"{analysis['dataset']} ({', '.join(names)}): {warning}"
                            for warning, names in warning_variables.items())
        if diagnostics_table.row_count:
            console.print(diagnostics_table)
        for message in messages:
            console.print(message, markup=False)
    if summary["speedups"]:
        table = Table(title="Paired target speedups: median (range)")
        for column in ("Dataset", "Tips", "L", "Backend", "Comparison", "Speedup"):
            table.add_column(column)
        for row in summary["speedups"]:
            label = "reference / Hyperiax" if row["kind"] == "traversal" else "CPU / GPU"
            table.add_row(row["dataset"], str(row["tips"]), str(row["landmarks"]), row["backend"], label,
                          _format(row["ratio"]) + "×")
        console.print(table)
    for g in summary["groups"]:
        for message in g["warnings"] + g["errors"]:
            console.print(f"{g['dataset']} {g['experiment']} L={g['landmarks']}: {message}", markup=False)
    for rejection in summary["rejected_comparisons"]:
        console.print(str(rejection), style="red", markup=False)
    console.print(f"Completed: {summary['successful_jobs']}/{summary['jobs']}; failed: {summary['failed_jobs']}")


def _replace_stale_plot(run_dir: Path, stem: str, message: str, *, smoke: bool) -> None:
    """Replace old figures with a notice; do not create inapplicable new figures."""
    if not any((run_dir / f"{stem}.{extension}").exists() for extension in ("png", "pdf")):
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, transform=ax.transAxes)
    fig.suptitle("Smoke validation — not a performance claim" if smoke else "Performance experiments")
    for extension in ("png", "pdf"):
        fig.savefig(run_dir / f"{stem}.{extension}", dpi=180)
    plt.close(fig)


def plot_summary(summary: dict, run_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    rows = [g for g in summary["groups"] if g["experiment"] == "scaling" and g["successful"]]
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, axis in zip(axes, ("tips", "landmarks")):
            for dataset in sorted({g["dataset"] for g in rows}):
                selected = sorted([g for g in rows if g["dataset"] == dataset and axis in g["axes"]],
                                  key=lambda g: g["nodes"] if axis == "tips" else g["landmarks"])
                for metric, style, label in (("target_seconds", "--", "target"), ("mcmc_seconds", "-", "MCMC")):
                    points = [g for g in selected if g[metric] is not None]
                    if not points:
                        continue
                    x = [g["nodes"] if axis == "tips" else g["landmarks"] for g in points]
                    y = np.array([g[metric]["median"] for g in points]) * 1000
                    lo = np.array([g[metric]["min"] for g in points]) * 1000
                    hi = np.array([g[metric]["max"] for g in points]) * 1000
                    ax.errorbar(x, y, yerr=[y - lo, hi - y], marker="o", linestyle=style,
                                label=f"{dataset}: {label}", capsize=3)
            ax.set(xlabel="Tree nodes (including super-root)" if axis == "tips" else "Landmarks",
                   ylabel="Time per evaluation / iteration (ms)", yscale="log")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.suptitle("Smoke validation — not a performance claim" if summary["smoke"] else "Computational scaling")
        for extension in ("png", "pdf"):
            fig.savefig(run_dir / f"scaling.{extension}", dpi=180)
        plt.close(fig)
    else:
        _replace_stale_plot(run_dir, "scaling", "Scaling unavailable: no successful scaling jobs.",
                            smoke=summary["smoke"])
    if summary["speedups"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, kind, title, ratio_label in (
            (axes[0], "traversal", "Implementation speedup", "reference / Hyperiax"),
            (axes[1], "hardware", "Hardware speedup", "CPU / GPU"),
        ):
            values = [v for v in summary["speedups"] if v["kind"] == kind]
            ax.set(title=title, ylabel=f"Target speedup ({ratio_label})")
            ax.axhline(1, color="black", linestyle="--", linewidth=1)
            if not values:
                ax.set_xticks([])
                ax.text(0.5, 0.5, f"No paired {ratio_label} measurements",
                        ha="center", va="center", transform=ax.transAxes)
                continue
            x = np.arange(len(values))
            medians = np.array([v["ratio"]["median"] for v in values])
            low = np.array([v["ratio"]["min"] for v in values])
            high = np.array([v["ratio"]["max"] for v in values])
            ax.bar(x, medians, yerr=[medians - low, high - medians], capsize=3)
            multiple_workloads = len({(v["dataset"], v["tips"]) for v in values}) > 1
            labels = [
                (f"{v['dataset']} (tips={v['tips']})\n" if multiple_workloads else "")
                + f"L={v['landmarks']}"
                + (f"\n{v['backend'].upper()}" if kind == "traversal" else "")
                for v in values
            ]
            ax.set_xticks(x, labels)
        fig.suptitle("Smoke comparison — not a performance claim" if summary["smoke"] else "Matched implementation / hardware comparisons")
        for extension in ("png", "pdf"):
            fig.savefig(run_dir / f"speedups.{extension}", dpi=180)
        plt.close(fig)
    else:
        _replace_stale_plot(run_dir, "speedups", "Speedups unavailable: no valid paired target comparisons.",
                            smoke=summary["smoke"])


def rebuild_report(run_dir: Path, *, plots: bool = True, console: Console | None = None,
                   burnin_fraction: float | None = None) -> dict:
    config = read_json(run_dir / "config.json")
    records = [read_json(path) for path in sorted((run_dir / "jobs").glob("*/result.json"))]
    from src.profiling.diagnostics import attach_full_diagnostics
    attach_full_diagnostics(records, run_dir, config, burnin_fraction=burnin_fraction)
    manifest = run_dir / "jobs.json"
    if manifest.exists():
        completed = {record["job_id"] for record in records}
        records.extend(dict(job, status="not_run", error="Job was not completed.")
                       for job in read_json(manifest) if job["job_id"] not in completed)
    summary = summarize(records, smoke=config["smoke"])
    summary["run_error"] = read_json(run_dir / "error.json") if (run_dir / "error.json").exists() else None
    write_json(run_dir / "results.json", records)
    write_json(run_dir / "summary.json", summary)
    print_summary(summary, console)
    if summary["run_error"]:
        (console or Console()).print(str(summary["run_error"]), style="red", markup=False)
    if plots:
        plot_summary(summary, run_dir)
    return summary
