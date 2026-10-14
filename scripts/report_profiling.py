"""Rebuild Rich tables, summary JSON and figures from existing per-job JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.profiling.report import rebuild_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory produced by scripts.run_profiling.")
    parser.add_argument("--no-plots", action="store_true", help="Print Rich tables and save JSON without generating figures.")
    parser.add_argument("--burnin-fraction", type=float, default=None,
                        help="Override full-diagnostic burn-in fraction in [0, 1); defaults to saved YAML setting or 0.5 for older runs. Does not modify samples.")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    summary = rebuild_report(args.run_dir.resolve(), plots=not args.no_plots, burnin_fraction=args.burnin_fraction)
    return 1 if summary["failed_jobs"] or summary["rejected_comparisons"] or summary["run_error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
