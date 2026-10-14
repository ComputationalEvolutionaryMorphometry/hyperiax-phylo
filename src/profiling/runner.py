"""Serial experiment orchestration with fresh, explicitly configured workers."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import psutil
from rich.console import Console

from src.profiling.artifacts import environment, read_json, write_json
from src.profiling.config import ROOT, expand_jobs, resolve_path
from src.profiling.progress import JobProgress
from src.profiling.report import rebuild_report
from src.profiling.subsets import build_subset


def worker_environment(config: dict, backend: str) -> dict:
    env = os.environ.copy()
    env.update(JAX_PLATFORMS="cpu" if backend == "cpu" else "cuda",
               CUDA_VISIBLE_DEVICES="" if backend == "cpu" else str(config["gpu_index"]),
               CUDA_DEVICE_ORDER="PCI_BUS_ID",
               JAX_ENABLE_COMPILATION_CACHE="false", XLA_PYTHON_CLIENT_PREALLOCATE="false",
               OMP_NUM_THREADS=str(config["cpu_threads"]), OPENBLAS_NUM_THREADS=str(config["cpu_threads"]),
               MKL_NUM_THREADS=str(config["cpu_threads"]), MPLBACKEND="Agg", QT_QPA_PLATFORM="offscreen")
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    return env


class MemorySampler:
    """Observed peaks, not live-buffer estimates. Missing NVML is explicit."""
    def __init__(self, backend: str, gpu_index: int):
        self.rss = 0
        self.gpu = None
        self.nvml = None
        self.gpu_info = None
        self.reason = "CPU job; GPU memory is not applicable."
        if backend == "gpu":
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml = pynvml
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                self.gpu_info = {"name": pynvml.nvmlDeviceGetName(self.handle),
                                 "driver": pynvml.nvmlSystemGetDriverVersion(),
                                 "uuid": pynvml.nvmlDeviceGetUUID(self.handle),
                                 "total_memory_bytes": pynvml.nvmlDeviceGetMemoryInfo(self.handle).total,
                                 "physical_index": gpu_index}
                for key in ("name", "driver", "uuid"):
                    if isinstance(self.gpu_info[key], bytes):
                        self.gpu_info[key] = self.gpu_info[key].decode()
                self.reason = None
            except Exception as error:
                self.reason = f"NVML unavailable: {error}"
                self.close()

    def sample(self, pid: int) -> None:
        try:
            process = psutil.Process(pid)
            processes = [process] + process.children(recursive=True)
        except psutil.Error:
            return
        rss, pids = 0, set()
        for process in processes:
            try:
                rss += process.memory_info().rss
                pids.add(process.pid)
            except psutil.Error:
                pass
        self.rss = max(self.rss, rss)
        if self.nvml is not None:
            try:
                entries = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
                amounts = [p.usedGpuMemory for p in entries if p.pid in pids]
                if any(not isinstance(value, int) or value < 0 or value > self.gpu_info["total_memory_bytes"] for value in amounts):
                    raise RuntimeError("NVML per-process memory is unavailable on this platform.")
                self.gpu = max(self.gpu or 0, sum(amounts))
            except Exception as error:
                self.gpu = None
                self.reason = f"NVML sampling unavailable: {error}"
                self.close()

    def close(self):
        if self.nvml is not None:
            self.nvml.nvmlShutdown()
            self.nvml = None


def _stop(process, *, grace_seconds: float = 3.0):
    """Stop the worker's private process group, not just its leader.

    Workers use start_new_session=True on POSIX, so PID is also the group ID.
    Reap the leader while allowing descendants the rest of the grace period.
    """
    if os.name == "posix":
        pgid = process.pid
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
            return
        deadline = time.monotonic() + grace_seconds
        while True:
            process.poll()  # The leader may exit while descendants survive.
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # The last member exited between the probe and kill.
                break
            time.sleep(min(0.05, remaining))
        process.wait()
        return

    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_job(config: dict, job: dict, run_dir: Path, *, console: Console | None = None) -> dict:
    job_dir = run_dir / "jobs" / job["job_id"]
    job_dir.mkdir(parents=True, exist_ok=False)
    payload_path = job_dir / "input.json"
    write_json(payload_path, {"config": config, "job": job, "job_dir": str(job_dir)})
    env = worker_environment(config, job["backend"])
    sampler = MemorySampler(job["backend"], config["gpu_index"])
    if job["backend"] == "gpu" and sampler.gpu_info is not None:
        # Use the monitored device's UUID so CUDA/NVML index ordering cannot
        # silently select one GPU while measuring another GPU's memory.
        env["CUDA_VISIBLE_DEVICES"] = sampler.gpu_info["uuid"]
    timeout = config["full" if job["experiment"] == "full" else "benchmark"]["timeout_seconds"]
    start = time.perf_counter()
    failure = None
    display = JobProgress(console or Console(), job_dir / "progress",
                          config["full"]["num_chains"] if job["experiment"] == "full" else 0)
    with display, (job_dir / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "src.profiling.worker", str(payload_path)],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=os.name == "posix")
        try:
            while process.poll() is None:
                sampler.sample(process.pid)
                display.poll()
                if time.perf_counter() - start > timeout:
                    failure = f"Timeout after {timeout} seconds."
                    _stop(process)
                    break
                time.sleep(config["benchmark"]["memory_interval_seconds"])
        except BaseException:
            _stop(process)
            display.finish("interrupted")
            raise
        finally:
            sampler.close()
        display.finish("timeout" if failure else ("failed" if process.returncode else "ok"))
    elapsed = time.perf_counter() - start
    output = job_dir / "worker_result.json"
    result = read_json(output) if output.exists() else {"status": "failed", "error": f"Worker exited {process.returncode}; see worker.log."}
    if failure:
        result.update(status="timeout", error=failure)
    elif process.returncode and result.get("status") == "ok":
        result.update(status="failed", error=f"Worker exited {process.returncode} after writing output.")
    if sampler.reason and job["backend"] == "gpu":
        result.setdefault("warnings", []).append(sampler.reason)
    result.update(job, process_wall_seconds=elapsed, peak_rss_bytes=sampler.rss,
                  peak_gpu_process_bytes=sampler.gpu, gpu_memory_unavailable_reason=sampler.reason,
                  gpu_info=sampler.gpu_info, worker_returncode=process.returncode,
                  environment_overrides={key: env[key] for key in ("JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES",
                    "CUDA_DEVICE_ORDER", "JAX_ENABLE_COMPILATION_CACHE", "XLA_PYTHON_CLIENT_PREALLOCATE",
                    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    write_json(job_dir / "result.json", result)
    return result


def run_experiments(config: dict, console: Console | None = None) -> tuple[Path, dict]:
    console = console or Console()
    root = resolve_path(config["output_root"])
    run_dir = root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", config)
    if config["cpu_affinity"] is not None:
        psutil.Process().cpu_affinity(config["cpu_affinity"])
    write_json(run_dir / "environment.json", environment(ROOT))
    jobs = expand_jobs(config)
    subsets = {}
    try:
        for job in jobs:
            key = (job["dataset"], job["tips"], job["landmarks"])
            if key not in subsets:
                name, tips, landmarks = key
                path = run_dir / "subsets" / f"{name}_tips{tips}_landmarks{landmarks}.h5"
                subsets[key] = build_subset(config["datasets"][name]["h5"], path,
                                             tips=tips, landmarks=landmarks, seed=config["seed"])
            job["subset"] = subsets[key]
        write_json(run_dir / "subsets.json", list(subsets.values()))
        random.Random(config["seed"]).shuffle(jobs)
        write_json(run_dir / "jobs.json", jobs)
        for index, job in enumerate(jobs, 1):
            console.print(f"[{index}/{len(jobs)}] {job['dataset']} {job['experiment']} "
                          f"tips={job['tips']} L={job['landmarks']} {job['implementation']}/{job['backend']}", markup=False)
            result = run_job(config, job, run_dir, console=console)
            console.print(f"  {result['status']} — {result['process_wall_seconds']:.2f} s", markup=False)
        summary = rebuild_report(run_dir, console=console)
    except BaseException as error:
        write_json(run_dir / "error.json", {"error": f"{type(error).__name__}: {error}"})
        # Completed per-job JSON remains reportable after a failed/interrupted run.
        rebuild_report(run_dir, plots=False, console=console)
        raise
    console.print(f"Results: {run_dir}", markup=False)
    return run_dir, summary
