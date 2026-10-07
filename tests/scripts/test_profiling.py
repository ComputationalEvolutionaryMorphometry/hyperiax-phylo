from copy import deepcopy
from dataclasses import replace
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ete3 import Tree
from rich.console import Console

from src import data_schema as s
from src.bffg import MCMCModelConfig, MCMCParams, build_bffg_context
from src.driver import MCMCDriverConfig, _build_mcmc_target_and_phylo_root_state, run_mcmc_chain
from src.loader import load_augmented_butterfly_tree
from src.profiling.artifacts import read_json, write_json
from src.profiling.config import ROOT, expand_jobs, load_config, scaling_cases
from src.profiling.reference import build_reference_target
from src.profiling.report import rebuild_report, summarize
from src.profiling.runner import run_job, worker_environment
from src.profiling.subsets import build_subset, landmark_order
from scripts import report_profiling, run_profiling


def test_formal_matrix_has_18_unique_scaling_settings_and_83_jobs():
    config = load_config("config/profiling.yaml")
    assert sum(len(scaling_cases(d)) for d in config["datasets"].values()) == 18
    jobs = expand_jobs(config)
    assert len(jobs) == 83  # 2 full + 18 * 3 scaling + 3 L * 3 implementations * 3 repeats
    assert sum(j["experiment"] == "scaling" for j in jobs) == 54
    assert len({j["job_id"] for j in jobs}) == 83
    for dataset in config["datasets"].values():
        assert sum(len(c["axes"]) == 2 for c in scaling_cases(dataset)) == 1


def test_run_name_uses_second_precision_and_refuses_collisions(monkeypatch, tmp_path):
    from datetime import datetime, timezone
    from src.profiling import runner

    def fixed_now(tz):
        assert tz is timezone.utc
        return datetime(2026, 10, 6, 6, 8, 44, 123456, tzinfo=tz)

    monkeypatch.setattr(runner, "datetime", SimpleNamespace(now=fixed_now))
    monkeypatch.setattr(runner, "environment", lambda root: {})
    monkeypatch.setattr(runner, "expand_jobs", lambda config: [])
    monkeypatch.setattr(runner, "rebuild_report", lambda *args, **kwargs: {})
    config = {"output_root": str(tmp_path), "cpu_affinity": None, "seed": 0}
    console = Console(file=io.StringIO())
    run_dir, _ = runner.run_experiments(config, console)
    assert run_dir.name == "20261006_060844"
    original = (run_dir / "config.json").read_bytes()
    with pytest.raises(FileExistsError):
        runner.run_experiments(config | {"seed": 1}, console)
    assert (run_dir / "config.json").read_bytes() == original


def test_yaml_is_read_only_and_cli_help_is_complete(capsys):
    path = ROOT / "config/profiling.yaml"
    before = path.read_bytes()
    assert run_profiling.main(["--config", str(path), "--dry-run"]) == 0
    assert "Experiment plan" in capsys.readouterr().out
    assert path.read_bytes() == before
    for parser in (run_profiling._build_parser(), report_profiling._build_parser()):
        assert all(action.help for action in parser._actions if action.option_strings)


def test_backend_environment_is_explicit_and_has_no_persistent_cache():
    config = load_config("config/profiling_smoke.yaml")
    cpu = worker_environment(config, "cpu")
    gpu = worker_environment(config, "gpu")
    assert cpu["JAX_PLATFORMS"] == "cpu"
    assert cpu["CUDA_VISIBLE_DEVICES"] == ""
    assert gpu["JAX_PLATFORMS"] == "cuda"
    assert gpu["CUDA_VISIBLE_DEVICES"] == "0"
    assert cpu["JAX_ENABLE_COMPILATION_CACHE"] == "false"
    assert "JAX_COMPILATION_CACHE_DIR" not in cpu


def test_analysis_yaml_scientific_notation_and_nullable_timing_are_coerced():
    from scripts.run_mcmc import _model_config_from_mapping, _driver_config_from_mapping, _coerce_cli_override_value
    config = load_config("config/profiling_smoke.yaml")
    dataset = config["datasets"]["butterflies"]
    model = _model_config_from_mapping(dataset["model"])
    driver = _driver_config_from_mapping(dataset["driver"])
    assert model.k_sigma_min == 1e-5
    assert driver.profile_warmup_iterations is None
    assert _driver_config_from_mapping({"profile_warmup_iterations": 2}).profile_warmup_iterations == 2
    assert _coerce_cli_override_value("driver", "profile_warmup_iterations", 2) == 2
    assert _coerce_cli_override_value("driver", "profile_warmup_iterations", None) is None
    with pytest.raises(ValueError, match="profile_warmup"):
        _driver_config_from_mapping({"profile_warmup_iterations": True})


@pytest.fixture(params=["butterflies", "beaks"])
def subset(request, tmp_path):
    source = ROOT / f"data/{request.param}/data.h5"
    metadata = build_subset(source, tmp_path / "subset.h5", tips=8, landmarks=4, seed=0)
    return source, metadata


def test_subset_preserves_contract_scales_and_root_to_tip_lengths(subset, tmp_path):
    source, metadata = subset
    dataset = load_augmented_butterfly_tree(metadata["h5"])
    assert dataset.tree.size == metadata["nodes"]
    assert len(dataset.leaf_names) == 8
    assert dataset.tree["coords"].shape[1] == 4
    with h5py.File(source, "r") as original, h5py.File(metadata["h5"], "r") as small:
        for key in (s.PREPROC_CENTER, s.PREPROC_SCALE, s.PREPROC_EDGE_LEN_NORMALIZER):
            np.testing.assert_array_equal(original[key][()], small[key][()])
        old_tree = Tree(original[s.TREE_NEWICK_AUG][()].decode(), format=1)
        new_tree = Tree(small[s.TREE_NEWICK_AUG][()].decode(), format=1)
        for name in metadata["leaf_names"]:
            assert old_tree.get_distance(name) == pytest.approx(new_tree.get_distance(name), rel=1e-12)
        np.testing.assert_allclose(small[s.NODES_COORDS][0], small[s.NODES_COORDS][:][small[s.NODES_IS_LEAF][:]].mean(0))
    tiny = build_subset(source, tmp_path / "tiny.h5", tips=4, landmarks=2, seed=0)
    assert set(tiny["leaf_names"]) < set(metadata["leaf_names"])
    assert set(tiny["landmark_indices"]) < set(metadata["landmark_indices"])


def test_identity_subset_reuses_original_file(tmp_path):
    source = ROOT / "data/beaks/data.h5"
    output = tmp_path / "unused.h5"
    result = build_subset(source, output, tips=348, landmarks=79, seed=0)
    assert result["h5"] == str(source)
    assert not output.exists()
    assert result["landmark_indices"] == list(range(79))


def test_farthest_point_ties_and_invalid_subset(tmp_path):
    assert landmark_order(np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])) == [0, 3, 1, 2]
    with pytest.raises(ValueError, match="exceeds"):
        build_subset(ROOT / "data/beaks/data.h5", tmp_path / "bad.h5", tips=349, landmarks=4, seed=0)


def test_reference_matches_current_target_in_2d_and_3d(subset):
    _, metadata = subset
    context = build_bffg_context(load_augmented_butterfly_tree(metadata["h5"]), MCMCModelConfig(num_edge_steps=2))
    production = _build_mcmc_target_and_phylo_root_state(context)
    reference = build_reference_target(context)
    params = MCMCParams(k_alpha=0.134, k_sigma=0.212, obs_var=0.00125)
    for seed in range(3):
        noise = jnp.asarray(np.random.default_rng(seed).normal(size=context.tree["zs"].shape))
        expected = jax.block_until_ready(production(params, noise))
        actual = jax.block_until_ready(reference(params, noise))
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)


def test_timing_is_optional_excludes_warmup_and_preserves_samples(subset):
    _, metadata = subset
    context = build_bffg_context(load_augmented_butterfly_tree(metadata["h5"]), MCMCModelConfig(num_edge_steps=2))
    params = MCMCParams(k_alpha=0.134, k_sigma=0.212, obs_var=0.00125)
    initial = (params, jnp.zeros_like(context.tree["zs"]))
    target = _build_mcmc_target_and_phylo_root_state(context)
    config = MCMCDriverConfig(num_samples=5, num_chains=1, progress_bar=False)
    plain = run_mcmc_chain(context, driver_config=config, initial_state=initial,
                           collect_phylo_root_samples=True, target_and_phylo_root_state=target)
    statuses = []
    profiled = run_mcmc_chain(context, driver_config=replace(config, profile_warmup_iterations=2),
                              initial_state=initial, collect_phylo_root_samples=True, target_and_phylo_root_state=target,
                              status_callback=lambda *args: statuses.append(args))
    assert statuses[0] == (None, "First target / JIT", 0, None)
    assert (None, "MCMC warmup", 2, 2) in statuses
    assert (None, "MCMC sampling", 0, 3) in statuses
    assert (None, "MCMC sampling", 3, 3) in statuses
    assert statuses[-1] == (None, "Chain complete", 5, 5)
    assert plain.timings is None
    assert profiled.timings["measured_iterations"] == 3
    assert profiled.timings["warmup_iterations"] == 2
    assert profiled.timings["measured_loop_seconds"] > 0
    assert profiled.timings["warmup_seconds"] > 0
    np.testing.assert_array_equal(plain.log_posteriors, profiled.log_posteriors)
    np.testing.assert_array_equal(plain.accepted, profiled.accepted)
    np.testing.assert_array_equal(plain.phylo_root_samples, profiled.phylo_root_samples)
    with pytest.raises(ValueError, match="profile_warmup"):
        run_mcmc_chain(context, driver_config=replace(config, profile_warmup_iterations=5))


@pytest.mark.parametrize("newick", [
    "((A:0.2,B:0.2):0.2,(C:0.2,D:0.2):0.2);",
    "(A:0.2,(B:0.2,(C:0.2,D:0.2):0.2):0.2);",
])
@pytest.mark.parametrize("dimension", [2, 3])
def test_reference_topologies_and_nonzero_jitter(newick, dimension, tmp_path):
    import pandas as pd
    from scripts.build_data import convert_shape_dataset
    base = np.array([[-0.4, 0.1, 0.2], [0.3, -0.2, -0.1], [0.5, 0.4, 0.3]])[:, :dimension]
    columns = [f"{i + 1}.{axis}" for i in range(3) for axis in "XYZ"[:dimension]]
    rows = [dict(zip(columns, (base + index * 0.03).ravel()), name=name) for index, name in enumerate("ABCD")]
    csv = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    tree_path = tmp_path / "tree.nwk"
    tree_path.write_text(newick)
    output = tmp_path / "input.h5"
    convert_shape_dataset(csv, tree_path, output)
    context = build_bffg_context(load_augmented_butterfly_tree(output),
                                MCMCModelConfig(num_edge_steps=2, covar_jitter=1e-8, dist_jitter=1e-7))
    params = MCMCParams(k_alpha=0.15, k_sigma=0.2, obs_var=0.01)
    production, reference = _build_mcmc_target_and_phylo_root_state(context), build_reference_target(context)
    for seed in range(3):
        noise = jnp.asarray(np.random.default_rng(seed).normal(size=context.tree["zs"].shape))
        for actual, expected in zip(reference(params, noise), production(params, noise)):
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


def _record(**overrides):
    result = dict(dataset="butterflies", experiment="reference", tips=8, landmarks=4, axes=[],
                  implementation="hyperiax", backend="cpu", seed=0, status="ok",
                  subset={"nodes": 16, "dimensions": 2}, target_seconds=0.01,
                  check_output={"log_target": 1.0, "root_state": [0., 0.]})
    return result | overrides


def test_reports_pair_seeds_and_suppress_invalid_speedups(tmp_path):
    records = [_record(), _record(implementation="reference", target_seconds=0.02,
                                  equivalence={"max_absolute_error": 0.0}),
               _record(seed=1, status="failed", error="out of memory")]
    summary = summarize(records, smoke=True)
    assert summary["failed_jobs"] == 1
    assert summary["speedups"][0]["ratio"]["median"] == 2
    broken = deepcopy(records)
    broken[1]["check_output"]["root_state"] = [1.0, 0.]
    assert not summarize(broken, smoke=True)["speedups"]
    for index, record in enumerate(records):
        write_json(tmp_path / "jobs" / str(index) / "result.json", record)
    write_json(tmp_path / "config.json", {"smoke": True})
    stream = io.StringIO()
    rebuilt = rebuild_report(tmp_path, plots=False, console=Console(file=stream, width=160))
    assert rebuilt == read_json(tmp_path / "summary.json")
    assert "Smoke validation" in stream.getvalue()
    assert "CPU smoke" not in stream.getvalue()
    assert "out of memory" in stream.getvalue()
    assert len(read_json(tmp_path / "results.json")) == 3


def test_failed_worker_is_saved_and_gpu_memory_can_be_missing(monkeypatch, tmp_path):
    class FailedProcess:
        pid = -1
        returncode = 7

        def poll(self):
            return 7

    monkeypatch.setattr("src.profiling.runner.subprocess.Popen", lambda *args, **kwargs: FailedProcess())
    config = load_config("config/profiling_smoke.yaml")
    job = expand_jobs(config)[0]
    result = run_job(config, job, tmp_path)
    assert result["status"] == "failed"
    assert result["worker_returncode"] == 7
    assert result["peak_gpu_process_bytes"] is None
    assert read_json(tmp_path / "jobs" / job["job_id"] / "result.json")["status"] == "failed"


def test_timeout_is_recorded_and_stops_worker(monkeypatch, tmp_path):
    process = SimpleNamespace(pid=999999, returncode=None, poll=lambda: None)
    stopped = []
    monkeypatch.setattr("src.profiling.runner.subprocess.Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr("src.profiling.runner._stop", lambda p: stopped.append(p))
    clock = iter([0., 301., 302.])
    monkeypatch.setattr("src.profiling.runner.time", SimpleNamespace(perf_counter=lambda: next(clock)))
    config = load_config("config/profiling_smoke.yaml")
    result = run_job(config, expand_jobs(config)[0], tmp_path)
    assert stopped == [process]
    assert result["status"] == "timeout"
    assert "Timeout" in result["error"]


@pytest.mark.parametrize("gone_at", ["term", "probe", "kill", None])
def test_stop_checks_group_even_after_leader_exits(monkeypatch, gone_at):
    import signal
    from unittest.mock import Mock
    from src.profiling import runner

    clock = [0.0]
    signals = []
    process = Mock(pid=98765)
    process.poll.return_value = 0  # Exited leader, not proof of an empty group.

    def killpg(pgid, sig):
        assert pgid == process.pid
        signals.append(sig)
        if ((gone_at == "term" and sig == signal.SIGTERM)
                or (gone_at == "probe" and sig == 0)
                or (gone_at == "kill" and sig == signal.SIGKILL)):
            raise ProcessLookupError

    monkeypatch.setattr(runner, "os", SimpleNamespace(name="posix", killpg=killpg))
    monkeypatch.setattr(runner, "time", SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds)))
    runner._stop(process, grace_seconds=0.1)
    process.wait.assert_called_once_with()
    assert signals[0] == signal.SIGTERM
    if gone_at in ("kill", None):
        assert signals[-1] == signal.SIGKILL
        assert clock[0] == pytest.approx(0.1)
        assert process.poll.called
    else:
        assert signal.SIGKILL not in signals


def test_stop_non_posix_retains_terminate_kill_fallback(monkeypatch):
    import subprocess
    from unittest.mock import Mock
    from src.profiling import runner

    process = Mock()
    process.wait.side_effect = [subprocess.TimeoutExpired("worker", 0.1), 0]
    monkeypatch.setattr(runner, "os", SimpleNamespace(name="nt"))
    runner._stop(process, grace_seconds=0.1)
    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.wait.call_count == 2


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_stop_terminates_real_child_ignoring_sigterm():
    import os
    import select
    import signal
    import subprocess
    import time
    import psutil
    from src.profiling.runner import _stop

    child_code = ("import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                  "print('ready', flush=True); time.sleep(60)")
    leader_code = (
        "import subprocess,sys,time; "
        f"child=subprocess.Popen([sys.executable,'-u','-c',{child_code!r}],stdout=subprocess.PIPE,text=True); "
        "assert child.stdout.readline().strip()=='ready'; "
        "print(child.pid,flush=True); time.sleep(60)"
    )
    process = subprocess.Popen([sys.executable, "-u", "-c", leader_code], stdout=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        assert select.select([process.stdout], [], [], 10)[0], "Test child did not become ready."
        child_pid = int(process.stdout.readline())
        _stop(process, grace_seconds=0.1)
        assert process.poll() is not None
        deadline = time.monotonic() + 3
        while psutil.pid_exists(child_pid):
            try:
                if psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE:
                    break  # Dead orphan awaiting its new parent's reap.
            except psutil.NoSuchProcess:
                break
            assert time.monotonic() < deadline, "Descendant survived process-group cleanup."
            time.sleep(0.01)
    finally:
        # Cleanup targets only the private group created by this test.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        process.stdout.close()


def test_current_yaml_workloads_and_diagnostic_burnin():
    formal = load_config("config/profiling.yaml")
    smoke = load_config("config/profiling_smoke.yaml")
    assert formal["full_diagnostics"]["burnin_fraction"] == 0.1
    assert smoke["full_diagnostics"]["burnin_fraction"] == 0.1
    assert formal["full"]["num_samples"] * 0.9 == 4500
    assert smoke["full"]["num_samples"] * 0.9 == 90
    assert smoke["primary_backend"] == "gpu"
    assert smoke["reference_landmarks"] == [8, 16, 32]
    for dataset in smoke["datasets"].values():
        assert dataset["tips"] == dataset["landmarks"] == [4, 8, 16, 32]
    jobs = expand_jobs(smoke)
    assert len(jobs) == 25
    assert [sum(j["experiment"] == family for j in jobs) for family in ("full", "scaling", "reference")] == [2, 14, 9]
    assert len(expand_jobs(smoke | {"primary_backend": "cpu"})) == 22


def test_incomplete_run_does_not_report_success(tmp_path):
    job = _record(job_id="job_0000")
    write_json(tmp_path / "jobs.json", [job])
    write_json(tmp_path / "config.json", {"smoke": True})
    summary = rebuild_report(tmp_path, plots=False, console=Console(file=io.StringIO()))
    assert summary["jobs"] == 1
    assert summary["failed_jobs"] == 1
    assert summary["groups"][0]["status"] == "failed"


def test_gpu_execution_and_monitoring_use_same_device_uuid(monkeypatch, tmp_path):
    fake_nvml = SimpleNamespace(
        nvmlInit=lambda: None, nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetName=lambda handle: b"Test GPU",
        nvmlSystemGetDriverVersion=lambda: b"test-driver",
        nvmlDeviceGetUUID=lambda handle: f"GPU-test-{handle}".encode(),
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(total=1024),
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)
    launched = {}

    def launch(*args, **kwargs):
        launched.update(kwargs)
        return SimpleNamespace(pid=999999, returncode=0, poll=lambda: 0)

    monkeypatch.setattr("src.profiling.runner.subprocess.Popen", launch)
    config = load_config("config/profiling_smoke.yaml")
    config["gpu_index"] = 2
    job = expand_jobs(config)[0] | {"backend": "gpu"}
    result = run_job(config, job, tmp_path)
    assert launched["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-test-2"
    assert result["gpu_info"]["physical_index"] == 2
    assert result["gpu_info"]["name"] == "Test GPU"


def test_progress_snapshots_throttle_flush_boundaries_and_pickle(monkeypatch, tmp_path):
    import pickle
    from src.profiling import progress as module
    clock = [0.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0], time=lambda: clock[0]))
    reporter = module.ProgressReporter(tmp_path)
    reporter.stage("Warmup", 0, 10)
    reporter.stage("Warmup", 1, 10)
    assert read_json(tmp_path / "worker.json")["completed"] == 0
    clock[0] = 1.0
    reporter.stage("Warmup", 5, 10)
    assert read_json(tmp_path / "worker.json")["completed"] == 5
    reporter.stage("Warmup", 10, 10)
    assert read_json(tmp_path / "worker.json")["completed"] == 10
    reporter.stage("Measurement", 0, 20)
    assert read_json(tmp_path / "worker.json")["stage"] == "Measurement"
    copied = pickle.loads(pickle.dumps(reporter))
    copied(1, "MCMC sampling", 3, 20)
    assert read_json(tmp_path / "chain_1.json")["chain_index"] == 1
    assert read_json(tmp_path / "worker.json")["stage"] == "Measurement"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("terminal", [False, True])
def test_progress_display_unknown_stage_counts_and_failure(tmp_path, terminal):
    from src.profiling.progress import JobProgress, ProgressReporter
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=terminal, width=120)
    reporter = ProgressReporter(tmp_path)
    with JobProgress(console, tmp_path, num_chains=2) as display:
        reporter.stage("First target / JIT")
        display.poll(force=True)
        assert display.progress.tasks[display.tasks["worker"]].total is None
        reporter(0, "MCMC sampling", 3, 10)
        display.poll(force=True)
        task = display.progress.tasks[display.tasks["chain_0"]]
        assert task.completed == 3 and task.total == 10
        reporter.stage("Measurement", 1, 1)
        display.poll(force=True)
        assert display.progress.tasks[display.tasks["worker"]].finished
        reporter.stage("Saving result")
        display.poll(force=True)
        worker = display.progress.tasks[display.tasks["worker"]]
        assert worker.total is None and not worker.finished
        display.finish("timeout")
        assert task.completed == 3
    assert "timeout" in stream.getvalue()
    if not terminal:
        assert "First target / JIT" in stream.getvalue()
        assert "3/10" in stream.getvalue()
        assert "\x1b[" not in stream.getvalue()


def test_redirected_progress_has_throttled_heartbeat(monkeypatch, tmp_path):
    from src.profiling import progress as module
    clock = [0.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0], time=lambda: clock[0]))
    stream = io.StringIO()
    reporter = module.ProgressReporter(tmp_path)
    with module.JobProgress(Console(file=stream), tmp_path) as display:
        reporter.stage("First target / JIT")
        display.poll()
        first = stream.getvalue()
        clock[0] = 5.0
        display.poll()
        assert stream.getvalue() == first
        clock[0] = 10.0
        display.poll()
        assert "elapsed 10.0 s" in stream.getvalue()
