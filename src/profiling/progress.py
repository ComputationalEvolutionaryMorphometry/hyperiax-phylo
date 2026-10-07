"""Low-frequency worker snapshots and parent-only Rich rendering (no JAX imports)."""

from __future__ import annotations

import json
from pathlib import Path
import time

from rich.console import Console
from rich.progress import BarColumn, Progress, ProgressColumn, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

from src.profiling.artifacts import read_json, write_json


class ProgressReporter:
    """Picklable callback; each chain owns a separate atomic JSON snapshot.

    Ordinary iteration writes are throttled to once per second. Phase boundaries
    and final counts are always written; workers never render terminal output.
    """

    def __init__(self, directory: Path, interval_seconds: float = 1.0):
        self.directory = Path(directory)
        self.interval_seconds = interval_seconds
        self._last = {}

    def __call__(self, chain_index: int | None, stage: str, completed: int = 0,
                 total: int | None = None) -> None:
        key = "worker" if chain_index is None else f"chain_{chain_index}"
        now = time.monotonic()
        previous = self._last.get(key)
        if previous is not None and previous[0] == stage:
            if previous[2:] == (completed, total):
                return
            if completed != total and now - previous[1] < self.interval_seconds:
                return
        write_json(self.directory / f"{key}.json", {
            "chain_index": chain_index, "stage": stage, "completed": completed,
            "total": total, "updated_at": time.time(),
        })
        self._last[key] = (stage, now, completed, total)

    def stage(self, name: str, completed: int = 0, total: int | None = None) -> None:
        self(None, name, completed, total)


class _CountsColumn(ProgressColumn):
    def render(self, task):
        if task.total is None:
            return Text("—")
        return Text(f"{int(task.completed)}/{int(task.total)} ({task.percentage:.0f}%)")


class JobProgress:
    """Poll snapshots without mixing diagnostic logs into the progress display.

    Redirected output gets plain periodic status lines, not terminal control codes.
    Elapsed time is per job / chain (not an ETA for compilation).
    """

    def __init__(self, console: Console, directory: Path, num_chains: int = 0):
        self.console = console
        self.directory = directory
        self.progress = Progress(SpinnerColumn(), TextColumn("{task.description}"),
                                 BarColumn(bar_width=20), _CountsColumn(), TimeElapsedColumn(),
                                 console=console, refresh_per_second=4, transient=True,
                                 disable=not console.is_terminal)
        self.tasks = {"worker": self.progress.add_task("Starting worker", total=None)}
        for index in range(num_chains):
            self.tasks[f"chain_{index}"] = self.progress.add_task(
                f"Chain {index + 1}: waiting", total=None, start=False)
        self._last_poll = float("-inf")
        self._logged = {}
        self._snapshots = {}
        self._started = time.monotonic()

    def __enter__(self):
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.progress.stop()

    def poll(self, *, force: bool = False):
        now = time.monotonic()
        if not force and now - self._last_poll < 0.25:
            return
        self._last_poll = now
        for path in sorted(self.directory.glob("*.json")):
            try:
                snapshot = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            self._snapshots[path.stem] = snapshot
        for key, snapshot in self._snapshots.items():
            index = snapshot["chain_index"]
            label = "" if index is None else f"Chain {index + 1}: "
            description = label + snapshot["stage"]
            if key not in self.tasks:
                self.tasks[key] = self.progress.add_task(description, total=None)
            task_id = self.tasks[key]
            self.progress.start_task(task_id)
            # Rich's update(total=None) preserves the old total; set explicitly
            # when a bounded phase transitions back to an indeterminate phase.
            task = self.progress.tasks[task_id]
            if task.description != description:
                task.finished_time = None
            task.total = snapshot["total"]
            self.progress.update(task_id, description=description, completed=snapshot["completed"])
            previous = self._logged.get(key)
            if not self.console.is_terminal and (
                previous is None or previous[0] != description or now - previous[1] >= 10.0
                or (force and previous[2:] != (snapshot["completed"], snapshot["total"]))
            ):
                counts = "" if snapshot["total"] is None else f" {snapshot['completed']}/{snapshot['total']}"
                self.console.print(f"  {description}{counts} — elapsed {now - self._started:.1f} s", markup=False)
                self._logged[key] = (description, now, snapshot["completed"], snapshot["total"])

    def finish(self, status: str):
        self.poll(force=True)
        # Do not turn failed/interrupted partial counts into 100% completion.
        if status != "ok":
            self.console.print(f"  Worker {status}; last reported progress retained in progress/.", markup=False)
