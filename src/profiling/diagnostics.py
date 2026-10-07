"""Post-process full-chain artifacts; never run inside benchmark timing windows."""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np

TRACE_PATHS = {"k_alpha": "samples/k_alpha", "k_sigma": "samples/k_sigma",
               "obs_var": "samples/obs_var", "log_posterior": "trace/log_posteriors"}
SCALAR_PARAMETERS = ("k_alpha", "k_sigma", "obs_var")


def diagnostic_settings(config: dict, burnin_fraction: float | None = None) -> dict:
    supplied = config.get("full_diagnostics", {})
    if not isinstance(supplied, dict) or set(supplied) - {"burnin_fraction"}:
        raise ValueError("full_diagnostics accepts only burnin_fraction.")
    fraction = supplied.get("burnin_fraction", 0.5) if burnin_fraction is None else burnin_fraction
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction) or not 0 <= fraction < 1):
        raise ValueError("full_diagnostics.burnin_fraction must be finite and in [0, 1).")
    return {"burnin_fraction": float(fraction)}


def diagnose_chains(values: np.ndarray, *, burnin_fraction: float, wall_seconds: float | None = None,
                    allow_single_chain: bool = False) -> dict:
    """Keep chain/draw axes distinct; use ArviZ rank-based multi-chain diagnostics.

    allow_single_chain enables individual-chain ESS; R-hat stays unavailable
    for one chain. It does not approximate joint ESS by dividing by chain count.

    https://python.arviz.org/en/v0.22.0/api/generated/arviz.ess.html
    https://python.arviz.org/en/v0.22.0/api/generated/arviz.rhat.html
    """
    import arviz as az

    diagnostic_settings({}, burnin_fraction)
    values = np.asarray(values, dtype=np.float64)
    row = {"status": "unavailable", "ess_bulk": None, "ess_tail": None, "r_hat": None,
           "ess_bulk_per_second": None, "warnings": []}
    if values.ndim != 2:
        row["warnings"].append("Expected a scalar trace with shape (chain, draw).")
        return row
    chains, draws = values.shape
    discarded = math.floor(draws * burnin_fraction)
    retained = values[:, discarded:]
    row.update(chains=chains, original_draws_per_chain=draws, burnin_draws_per_chain=discarded,
               retained_draws_per_chain=retained.shape[1], retained_draws_total=retained.size)
    minimum_chains = 1 if allow_single_chain else 2
    if chains < minimum_chains or retained.shape[1] < 4:
        row["warnings"].append(f"Need at least {minimum_chains} chain(s) and four retained draws per chain.")
        return row
    if not np.isfinite(retained).all():
        row["warnings"].append("Retained samples contain non-finite values.")
        return row
    if np.any(np.ptp(retained, axis=1) == 0):
        row["warnings"].append("At least one retained chain is constant; diagnostics are not reported as valid.")
        return row
    metrics = {"ess_bulk": float(az.ess(retained, method="bulk")),
               "ess_tail": float(az.ess(retained, method="tail", prob=(0.05, 0.95)))}
    if chains >= 2:
        metrics["r_hat"] = float(az.rhat(retained, method="rank"))
    if any(not math.isfinite(value) or value <= 0 for value in metrics.values()):
        row["warnings"].append("Diagnostic estimator returned a non-finite or non-positive value.")
        return row
    row.update(metrics, status="computed")
    if wall_seconds is not None and math.isfinite(wall_seconds) and wall_seconds > 0:
        row["ess_bulk_per_second"] = row["ess_bulk"] / wall_seconds
    if retained.shape[1] < 100:
        row["warnings"].append("Fewer than 100 retained draws per chain; estimates may be unstable.")
    if row["r_hat"] is not None and row["r_hat"] > 1.01:
        row["warnings"].append("Rank-normalized split R-hat exceeds 1.01; investigate chain mixing.")
    if min(row["ess_bulk"], row["ess_tail"]) < 100 * chains:
        row["warnings"].append("Bulk or tail ESS is below the diagnostic warning threshold of 100 per chain.")
    return row


def diagnose_artifact(path: Path, *, burnin_fraction: float = 0.5,
                      wall_seconds: float | None = None, smoke: bool = False) -> dict:
    import arviz as az

    diagnostic_settings({}, burnin_fraction)
    result = {"schema_version": 1, "status": "unavailable", "artifact_path": str(path),
              "arviz_version": az.__version__, "burnin_fraction": burnin_fraction, "thin": 1,
              "scope": "Three scalar parameters and log target; excludes ancestral coordinates.",
              "chain_ess_aggregation": "Minimum over k_alpha, k_sigma, obs_var, each estimated from that chain alone; unavailable if any parameter is unavailable. Not joint multi-chain ESS.",
              "methods": {"ess_bulk": "bulk", "ess_tail": "tail (0.05, 0.95)", "r_hat": "rank"},
              "ess_per_second_denominator": "process_wall_seconds, including all chains, burn-in, compilation and artifact writes",
              "process_wall_seconds": wall_seconds, "smoke": smoke, "variables": {}, "chains": {}, "warnings": []}
    if smoke:
        result["warnings"].append("Smoke experiment: diagnostic estimates are not evidence of posterior convergence.")
    if wall_seconds is None or not math.isfinite(wall_seconds) or wall_seconds <= 0:
        result["warnings"].append("Valid process wall time is missing; ESS/s is unavailable.")
    try:
        with h5py.File(path, "r") as artifact:
            for name, location in TRACE_PATHS.items():
                try:
                    group = artifact[location]
                    keys = sorted(group.keys())
                    if not keys or any(not key.startswith("chain_") for key in keys):
                        raise ValueError("Missing or unexpected chain identifiers.")
                    arrays = [np.asarray(group[key]) for key in keys]
                    if name in SCALAR_PARAMETERS:
                        for key, array in zip(keys, arrays):
                            chain = result["chains"].setdefault(key, {"variables": {}})
                            chain["variables"][name] = diagnose_chains(
                                array[None, ...], burnin_fraction=burnin_fraction, allow_single_chain=True)
                    values = np.stack(arrays)
                    row = diagnose_chains(values, burnin_fraction=burnin_fraction, wall_seconds=wall_seconds)
                    row["chain_ids"] = keys
                except (KeyError, ValueError, TypeError, OSError) as error:
                    row = {"status": "unavailable", "ess_bulk": None, "ess_tail": None,
                           "r_hat": None, "ess_bulk_per_second": None, "warnings": [str(error)]}
                result["variables"][name] = row
    except OSError as error:
        result["warnings"].append(f"Cannot read full-analysis artifact: {error}")
        return result
    computed = sum(row["status"] == "computed" for row in result["variables"].values())
    result["status"] = "computed" if computed == len(TRACE_PATHS) else ("partial" if computed else "unavailable")
    for chain in result["chains"].values():
        for name in SCALAR_PARAMETERS:
            chain["variables"].setdefault(name, {"status": "unavailable", "ess_bulk": None,
                                                 "ess_tail": None, "warnings": ["Parameter trace is missing."]})
        valid = all(row["status"] == "computed" for row in chain["variables"].values())
        chain["status"] = "computed" if valid else "unavailable"
        for metric in ("ess_bulk", "ess_tail"):
            chain[metric] = min(row[metric] for row in chain["variables"].values()) if valid else None
    return result


def attach_full_diagnostics(records: list[dict], run_dir: Path, config: dict,
                            *, burnin_fraction: float | None = None) -> None:
    """Add derived diagnostics without modifying samples, job results or timings."""
    from src.profiling.artifacts import write_json

    settings = diagnostic_settings(config, burnin_fraction)
    for record in records:
        if record["experiment"] != "full" or record["status"] != "ok":
            continue
        job_dir = run_dir / "jobs" / record["job_id"]
        # Resolve within the selected run, so copied runs do not read an old
        # absolute artifact_path pointing to a different experiment directory.
        artifact = job_dir / "analysis" / "artifacts.h5"
        diagnostics = diagnose_artifact(artifact, **settings,
                                        wall_seconds=record.get("process_wall_seconds"),
                                        smoke=config.get("smoke", False))
        record["diagnostics"] = diagnostics
        write_json(job_dir / "diagnostics.json", diagnostics)
