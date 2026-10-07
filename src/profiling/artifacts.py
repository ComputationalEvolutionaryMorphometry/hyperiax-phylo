"""JSON artifacts and lightweight environment metadata; no file digests."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

import psutil


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def environment(root: Path) -> dict:
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    versions = {}
    for package in ("hyperiax", "jax", "jaxlib", "numpy", "h5py", "rich", "psutil", "nvidia-ml-py"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    cpu_model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu_model = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                          if line.startswith("model name")), cpu_model)
    return {"python": platform.python_version(), "platform": platform.platform(), "cpu_model": cpu_model,
            "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
            "ram_bytes": psutil.virtual_memory().total,
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "packages": versions, "git_commit": git("rev-parse", "HEAD"),
            "git_status": git("status", "--short"),
            "cuda_packages": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
                              if dist.metadata["Name"].lower().startswith("nvidia-")}}
