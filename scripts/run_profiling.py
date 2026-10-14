"""Run YAML-configured performance experiments in isolated worker processes."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.profiling.config import expand_jobs, load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/profiling_smoke.yaml"),
                        help="YAML configuration; defaults to the small CPU smoke experiment.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and display job counts without running experiments.")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    console = Console()
    try:
        config = load_config(args.config)
        if args.dry_run:
            counts = Counter((j["dataset"], j["experiment"], j["implementation"], j["backend"]) for j in expand_jobs(config))
            table = Table(title="Experiment plan (no measurements)")
            for column in ("Dataset", "Experiment", "Implementation", "Backend", "Jobs"):
                table.add_column(column)
            for key, count in sorted(counts.items()):
                table.add_row(*key, str(count))
            console.print(table)
            return 0
        from src.profiling.runner import run_experiments
        _, summary = run_experiments(config, console)
        return 1 if summary["failed_jobs"] or summary["rejected_comparisons"] else 0
    except (Exception, KeyboardInterrupt) as error:
        console.print(f"Profiling failed: {type(error).__name__}: {error}", style="red", markup=False)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
