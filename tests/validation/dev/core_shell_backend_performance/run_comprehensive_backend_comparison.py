#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

from tests.validation.dev.cupy_rsoxs_optimization.run_cupy_rsoxs_optimization_matrix import (  # noqa: E402
    BenchmarkCase as CupyBenchmarkCase,
)
from tests.validation.dev.cupy_rsoxs_optimization.run_cupy_rsoxs_optimization_matrix import (  # noqa: E402
    CORE_SHELL_SINGLE_ENERGIES,
    EANGLE_OFF,
    SIZE_SPECS,
    TIMING_SEGMENTS,
    _json_default,
    _timestamp,
)
from tests.validation.dev.cyrsoxs_timing.run_cyrsoxs_timing_matrix import (  # noqa: E402
    BenchmarkCase as CyrsoxsBenchmarkCase,
)
from tests.validation.lib.core_shell import has_visible_gpu  # noqa: E402


OUT_ROOT = REPO_ROOT / "test-reports" / "core-shell-backend-performance-dev"
SUMMARY_NAME = "comprehensive_backend_comparison_summary.json"
SPEED_TABLE_NAME = "comprehensive_backend_comparison_speed.tsv"
MEMORY_TABLE_NAME = "comprehensive_backend_comparison_memory.tsv"
REPORT_NAME = "comprehensive_backend_comparison_report.md"
GPU_OBSERVER_SCRIPT = (
    REPO_ROOT / "tests" / "validation" / "dev" / "core_shell_backend_performance" / "cupy_gpu_mem_observer.py"
)
CYRSOXS_SCRIPT = (
    REPO_ROOT / "tests" / "validation" / "dev" / "cyrsoxs_timing" / "run_cyrsoxs_timing_matrix.py"
)
CUPY_SCRIPT = (
    REPO_ROOT
    / "tests"
    / "validation"
    / "dev"
    / "cupy_rsoxs_optimization"
    / "run_cupy_rsoxs_optimization_matrix.py"
)
ROTATION_SPECS = (
    ("no_rotation", EANGLE_OFF, "no rotation"),
    ("rot_0_5_165", (0.0, 5.0, 165.0), "0:5:165"),
)
HOST_STARTUP_MODES = ("warm", "hot")
DEVICE_STARTUP_MODES = ("steady", "hot")
GPU_MEMORY_OBSERVER_STABILIZE_WINDOW = 5
GPU_MEMORY_OBSERVER_STABILIZE_TOLERANCE_MIB = 8.0
GPU_MEMORY_OBSERVER_STARTUP_TIMEOUT_S = 30.0
GPU_MEMORY_OBSERVER_SHUTDOWN_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class ComparisonCase:
    key: str
    backend: str
    residency: str
    startup_mode: str
    execution_path: str
    path_variant: str
    z_collapse_mode: str | None
    rotation_key: str
    rotation_label: str
    eangle_rotation: tuple[float, float, float]
    script_path: Path
    worker_case: Any


def _sample_process_rss_mib(pid: int) -> float | None:
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return None

    if completed.returncode != 0:
        return None

    text = completed.stdout.strip()
    if not text:
        return None
    try:
        return float(text) / 1024.0
    except ValueError:
        return None


def _poll_process_rss_until_exit(
    proc: subprocess.Popen[str],
    poll_interval_s: float,
) -> dict[str, float]:
    baseline_rss = _sample_process_rss_mib(proc.pid)
    peak_rss = baseline_rss
    sample_count = 0

    while proc.poll() is None:
        time.sleep(poll_interval_s)
        sample_count += 1
        rss_sample = _sample_process_rss_mib(proc.pid)
        if rss_sample is not None:
            peak_rss = rss_sample if peak_rss is None else max(peak_rss, rss_sample)

    final_rss = _sample_process_rss_mib(proc.pid)
    sample_count += 1
    if final_rss is not None:
        peak_rss = final_rss if peak_rss is None else max(peak_rss, final_rss)

    probe = {
        "poll_interval_s": float(poll_interval_s),
        "sample_count": float(sample_count),
    }
    if baseline_rss is not None:
        probe["baseline_process_rss_mib"] = float(baseline_rss)
        probe["peak_process_rss_mib"] = float(peak_rss if peak_rss is not None else baseline_rss)
        probe["peak_process_rss_delta_mib"] = float(
            max(0.0, (peak_rss if peak_rss is not None else baseline_rss) - baseline_rss)
        )
    return probe


def _start_gpu_memory_observer(
    *,
    output_dir: Path,
    case: ComparisonCase,
    gpu_index: int,
    poll_interval_s: float,
) -> tuple[subprocess.Popen[str], Path, Path, Path]:
    observer_ready_path = output_dir / f"{case.key}__gpu_mem_observer_ready.json"
    observer_trace_path = output_dir / f"{case.key}__gpu_mem_observer.json"
    observer_stop_path = output_dir / f"{case.key}__gpu_mem_observer.stop"
    for path in (observer_ready_path, observer_trace_path, observer_stop_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    cmd = [
        sys.executable,
        str(GPU_OBSERVER_SCRIPT),
        "--ready-path",
        str(observer_ready_path),
        "--output-path",
        str(observer_trace_path),
        "--stop-path",
        str(observer_stop_path),
        "--device-index",
        "0",
        "--sample-interval-s",
        str(poll_interval_s),
        "--stabilize-window",
        str(GPU_MEMORY_OBSERVER_STABILIZE_WINDOW),
        "--stabilize-tolerance-mib",
        str(GPU_MEMORY_OBSERVER_STABILIZE_TOLERANCE_MIB),
        "--startup-timeout-s",
        str(GPU_MEMORY_OBSERVER_STARTUP_TIMEOUT_S),
    ]
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    return proc, observer_ready_path, observer_trace_path, observer_stop_path


def _wait_for_gpu_memory_observer_ready(
    proc: subprocess.Popen[str],
    *,
    ready_path: Path,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if ready_path.exists():
            payload = json.loads(ready_path.read_text())
            if payload.get("status") == "ready":
                return payload
            raise RuntimeError(
                "GPU memory observer failed before the worker started: "
                f"{payload.get('error_type', 'UnknownError')}: {payload.get('error', 'unknown error')}"
            )
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise RuntimeError(
                "GPU memory observer exited before signaling readiness. "
                f"stdout={stdout[-1000:]!r} stderr={stderr[-1000:]!r}"
            )
        time.sleep(0.01)
    raise TimeoutError("GPU memory observer did not reach a stable baseline in time.")


def _stop_gpu_memory_observer(
    proc: subprocess.Popen[str],
    *,
    stop_path: Path,
    output_path: Path,
    timeout_s: float,
) -> dict[str, Any]:
    stop_path.write_text("stop\n", encoding="utf-8")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if output_path.exists() and proc.poll() is not None:
            payload = json.loads(output_path.read_text())
            stdout, stderr = proc.communicate()
            try:
                stop_path.unlink()
            except FileNotFoundError:
                pass
            if stdout.strip():
                payload["observer_stdout"] = stdout[-4000:]
            if stderr.strip():
                payload["observer_stderr"] = stderr[-4000:]
            return payload
        time.sleep(0.01)

    proc.terminate()
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
    if output_path.exists():
        payload = json.loads(output_path.read_text())
        try:
            stop_path.unlink()
        except FileNotFoundError:
            pass
        if stdout.strip():
            payload["observer_stdout"] = stdout[-4000:]
        if stderr.strip():
            payload["observer_stderr"] = stderr[-4000:]
        return payload
    raise TimeoutError(
        "GPU memory observer did not flush its summary before shutdown. "
        f"stdout={stdout[-1000:]!r} stderr={stderr[-1000:]!r}"
    )


def _merge_memory_probes(
    *,
    gpu_probe: dict[str, Any],
    rss_probe: dict[str, float],
    trace_path: Path,
    ready_probe: dict[str, Any],
) -> dict[str, Any]:
    probe = {
        "probe_method": str(gpu_probe.get("probe_method", "cupy_memgetinfo_observer")),
        "poll_interval_s": float(gpu_probe.get("sample_interval_s", rss_probe.get("poll_interval_s", 0.0))),
        "sample_count": float(gpu_probe.get("sample_count", 0)),
        "baseline_gpu_used_mib": float(gpu_probe["baseline_gpu_used_mib"]),
        "peak_gpu_used_mib": float(gpu_probe["peak_gpu_used_mib"]),
        "peak_gpu_delta_mib": float(gpu_probe["peak_gpu_delta_mib"]),
        "gpu_observer_trace_path": str(trace_path),
        "gpu_observer_startup_seconds": float(gpu_probe.get("observer_startup_seconds", 0.0)),
        "gpu_observer_warmup_sample_count": float(gpu_probe.get("warmup_sample_count", 0)),
        "gpu_observer_stable_window_mib": list(
            ready_probe.get("stable_window_mib", gpu_probe.get("stable_window_mib", []))
        ),
    }
    probe.update(rss_probe)
    probe["rss_sample_count"] = probe.pop("sample_count")
    probe["sample_count"] = float(gpu_probe.get("sample_count", 0))
    return probe


def _host_cyrsoxs_case(
    *,
    startup_mode: str,
    rotation_key: str,
    rotation_label: str,
    eangle_rotation,
    size_label: str,
):
    worker_warmup_runs = 1 if startup_mode == "hot" else 0
    label = f"comprehensive__host__{startup_mode}__cyrsoxs__{rotation_key}"
    return ComparisonCase(
        key=label,
        backend="cyrsoxs",
        residency="host",
        startup_mode=startup_mode,
        execution_path="cyrsoxs",
        path_variant="cyrsoxs",
        z_collapse_mode=None,
        rotation_key=rotation_key,
        rotation_label=rotation_label,
        eangle_rotation=eangle_rotation,
        script_path=CYRSOXS_SCRIPT,
        worker_case=CyrsoxsBenchmarkCase(
            label=label,
            family="core_shell",
            shape_label=size_label,
            energies_ev=CORE_SHELL_SINGLE_ENERGIES,
            eangle_rotation=eangle_rotation,
            isotropic_representation="legacy_zero_array",
            cuda_prewarm_mode="before_prepare_inputs",
            worker_warmup_runs=worker_warmup_runs,
            notes=(
                "Comprehensive cross-backend host comparison lane for the "
                f"{size_label} single-energy CoreShell benchmark under startup_mode={startup_mode}."
            ),
        ),
    )


def _cupy_case(
    *,
    residency: str,
    startup_mode: str,
    execution_path: str,
    z_collapse_mode: str | None,
    rotation_key: str,
    rotation_label: str,
    eangle_rotation,
    size_label: str,
):
    path_variant = execution_path
    if z_collapse_mode is not None:
        path_variant = f"{execution_path}_zcollapse_{z_collapse_mode}"
    label = f"comprehensive__{residency}__{startup_mode}__{path_variant}__{rotation_key}"
    field_namespace = "numpy" if residency == "host" else "cupy"
    cuda_prewarm_mode = "before_prepare_inputs" if residency == "host" else "off"
    worker_warmup_runs = 1 if startup_mode == "hot" else 0
    backend_options = {"execution_path": execution_path}
    if z_collapse_mode is not None:
        backend_options["z_collapse_mode"] = z_collapse_mode
    return ComparisonCase(
        key=label,
        backend="cupy-rsoxs",
        residency=residency,
        startup_mode=startup_mode,
        execution_path=execution_path,
        path_variant=path_variant,
        z_collapse_mode=z_collapse_mode,
        rotation_key=rotation_key,
        rotation_label=rotation_label,
        eangle_rotation=eangle_rotation,
        script_path=CUPY_SCRIPT,
        worker_case=CupyBenchmarkCase(
            label=label,
            family="core_shell",
            backend="cupy-rsoxs",
            shape_label=size_label,
            energies_ev=CORE_SHELL_SINGLE_ENERGIES,
            eangle_rotation=eangle_rotation,
            field_namespace=field_namespace,
            isotropic_representation="legacy_zero_array",
            cuda_prewarm_mode=cuda_prewarm_mode,
            resident_mode=residency,
            input_policy="strict",
            ownership_policy="borrow",
            backend_options=backend_options,
            timing_segments=TIMING_SEGMENTS,
            worker_warmup_runs=worker_warmup_runs,
            notes=(
                "Comprehensive cross-backend comparison lane for the "
                f"{size_label} single-energy "
                f"CoreShell benchmark under residency={residency}, startup_mode={startup_mode}, "
                f"execution_path={execution_path}, z_collapse_mode={z_collapse_mode!r}."
            ),
        ),
    )


def _build_cases(*, include_z_collapse: bool, size_label: str) -> list[ComparisonCase]:
    cases: list[ComparisonCase] = []
    for rotation_key, eangle_rotation, rotation_label in ROTATION_SPECS:
        for startup_mode in HOST_STARTUP_MODES:
            cases.append(
                _host_cyrsoxs_case(
                    startup_mode=startup_mode,
                    rotation_key=rotation_key,
                    rotation_label=rotation_label,
                    eangle_rotation=eangle_rotation,
                    size_label=size_label,
                )
            )
            for execution_path in ("tensor_coeff", "direct_polarization"):
                cases.append(
                    _cupy_case(
                        residency="host",
                        startup_mode=startup_mode,
                        execution_path=execution_path,
                        z_collapse_mode=None,
                        rotation_key=rotation_key,
                        rotation_label=rotation_label,
                        eangle_rotation=eangle_rotation,
                        size_label=size_label,
                    )
                )
            if include_z_collapse:
                cases.append(
                    _cupy_case(
                        residency="host",
                        startup_mode=startup_mode,
                        execution_path="tensor_coeff",
                        z_collapse_mode="mean",
                        rotation_key=rotation_key,
                        rotation_label=rotation_label,
                        eangle_rotation=eangle_rotation,
                        size_label=size_label,
                    )
                )

        for startup_mode in DEVICE_STARTUP_MODES:
            for execution_path in ("tensor_coeff", "direct_polarization"):
                cases.append(
                    _cupy_case(
                        residency="device",
                        startup_mode=startup_mode,
                        execution_path=execution_path,
                        z_collapse_mode=None,
                        rotation_key=rotation_key,
                        rotation_label=rotation_label,
                        eangle_rotation=eangle_rotation,
                        size_label=size_label,
                    )
                )
            if include_z_collapse:
                cases.append(
                    _cupy_case(
                        residency="device",
                        startup_mode=startup_mode,
                        execution_path="tensor_coeff",
                        z_collapse_mode="mean",
                        rotation_key=rotation_key,
                        rotation_label=rotation_label,
                        eangle_rotation=eangle_rotation,
                        size_label=size_label,
                    )
                )
    return cases


def _run_case_subprocess(
    *,
    case: ComparisonCase,
    output_dir: Path,
    gpu_index: int,
    monitor_memory: bool,
    poll_interval_s: float,
    skip_existing: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{case.key}.json"
    if skip_existing and result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("status") == "ok" and (
            not monitor_memory
            or existing.get("memory_probe", {}).get("probe_method") == "cupy_memgetinfo_observer"
        ):
            existing["reused_existing"] = True
            return existing

    with tempfile.TemporaryDirectory(prefix="comprehensive_backend_", dir=output_dir) as tmp_dir:
        tmp_path = Path(tmp_dir)
        worker_case_path = tmp_path / "case.json"
        worker_result_path = tmp_path / "result.json"
        worker_case_path.write_text(
            json.dumps(asdict(case.worker_case), indent=2, default=_json_default) + "\n"
        )

        cmd = [
            sys.executable,
            str(case.script_path),
            "--worker-case-path",
            str(worker_case_path),
            "--worker-result-path",
            str(worker_result_path),
        ]
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

        observer_proc = None
        observer_ready = None
        observer_trace_path = None
        observer_stop_path = None
        if monitor_memory:
            observer_proc, observer_ready_path, observer_trace_path, observer_stop_path = _start_gpu_memory_observer(
                output_dir=output_dir,
                case=case,
                gpu_index=gpu_index,
                poll_interval_s=poll_interval_s,
            )
            observer_ready = _wait_for_gpu_memory_observer_ready(
                observer_proc,
                ready_path=observer_ready_path,
                timeout_s=GPU_MEMORY_OBSERVER_STARTUP_TIMEOUT_S,
            )

        started = time.perf_counter()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        memory_probe = None
        try:
            if monitor_memory:
                rss_probe = _poll_process_rss_until_exit(
                    proc,
                    poll_interval_s=poll_interval_s,
                )
                assert observer_proc is not None
                assert observer_trace_path is not None
                assert observer_stop_path is not None
                assert observer_ready is not None
                gpu_probe = _stop_gpu_memory_observer(
                    observer_proc,
                    stop_path=observer_stop_path,
                    output_path=observer_trace_path,
                    timeout_s=GPU_MEMORY_OBSERVER_SHUTDOWN_TIMEOUT_S,
                )
                memory_probe = _merge_memory_probes(
                    gpu_probe=gpu_probe,
                    rss_probe=rss_probe,
                    trace_path=observer_trace_path,
                    ready_probe=observer_ready,
                )
        finally:
            if monitor_memory and observer_proc is not None and observer_proc.poll() is None:
                assert observer_stop_path is not None
                observer_stop_path.write_text("stop\n", encoding="utf-8")
                observer_proc.terminate()
                try:
                    observer_proc.communicate(timeout=1.0)
                except subprocess.TimeoutExpired:
                    observer_proc.kill()
                    observer_proc.communicate()
            if monitor_memory and observer_stop_path is not None:
                try:
                    observer_stop_path.unlink()
                except FileNotFoundError:
                    pass
        stdout, stderr = proc.communicate()
        elapsed = time.perf_counter() - started

        if worker_result_path.exists():
            result = json.loads(worker_result_path.read_text())
        else:
            result = {
                "label": case.key,
                "status": "subprocess_failed",
                "error_type": "SubprocessFailure",
                "error": "Worker exited before writing a result file.",
            }

        result.update(
            {
                "comparison_key": case.key,
                "comparison_backend": case.backend,
                "comparison_residency": case.residency,
                "comparison_startup_mode": case.startup_mode,
                "comparison_execution_path": case.execution_path,
                "comparison_path_variant": case.path_variant,
                "comparison_z_collapse_mode": case.z_collapse_mode,
                "comparison_rotation_key": case.rotation_key,
                "comparison_rotation_label": case.rotation_label,
                "comparison_eangle_rotation": list(case.eangle_rotation),
                "subprocess_returncode": int(proc.returncode),
                "subprocess_seconds": float(elapsed),
            }
        )
        if stdout.strip():
            result["worker_stdout"] = stdout[-4000:]
        if stderr.strip():
            result["worker_stderr"] = stderr[-4000:]
        if memory_probe is not None:
            result["memory_probe"] = memory_probe

        result_path.write_text(json.dumps(result, indent=2, default=_json_default) + "\n")
        return result


def _case_sort_key(result: dict[str, Any]) -> tuple[int, int, int, int]:
    residency_order = {"host": 0, "device": 1}
    startup_order = {"warm": 0, "hot": 1, "steady": 2}
    backend_order = {"cyrsoxs": 0, "tensor_coeff": 1, "direct_polarization": 2}
    path_variant_order = {
        "cyrsoxs": 0,
        "tensor_coeff": 1,
        "tensor_coeff_zcollapse_mean": 2,
        "direct_polarization": 3,
    }
    rotation_order = {"no_rotation": 0, "rot_0_5_165": 1}
    return (
        residency_order.get(result.get("comparison_residency", ""), 99),
        startup_order.get(result.get("comparison_startup_mode", ""), 99),
        backend_order.get(result.get("comparison_execution_path", ""), 99),
        path_variant_order.get(result.get("comparison_path_variant", ""), 99),
        rotation_order.get(result.get("comparison_rotation_key", ""), 99),
    )


def _legacy_startup_mode_for_case(result: dict[str, Any]) -> str | None:
    startup_mode = str(result.get("comparison_startup_mode", ""))
    if startup_mode == "steady":
        return "warm"
    if startup_mode in {"warm", "hot"}:
        return startup_mode
    return None


def _legacy_baseline_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    baselines: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("comparison_backend") != "cyrsoxs":
            continue
        startup_mode = str(row.get("comparison_startup_mode", ""))
        rotation_key = str(row.get("comparison_rotation_key", ""))
        baselines[(startup_mode, rotation_key)] = row
    return baselines


def _speedup_vs_baseline(baseline_seconds: Any, candidate_seconds: Any) -> float | None:
    if baseline_seconds is None or candidate_seconds is None:
        return None
    baseline = float(baseline_seconds)
    candidate = float(candidate_seconds)
    if candidate == 0.0:
        return None
    return baseline / candidate


def _enrich_results_with_legacy_speedups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = _legacy_baseline_lookup(rows)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        enriched_row = dict(row)
        baseline_startup_mode = _legacy_startup_mode_for_case(row)
        enriched_row["comparison_cyrsoxs_baseline_startup_mode"] = baseline_startup_mode
        enriched_row["comparison_cyrsoxs_baseline_key"] = None
        enriched_row["comparison_cyrsoxs_primary_seconds"] = None
        enriched_row["comparison_speedup_vs_cyrsoxs"] = None

        if row.get("comparison_backend") == "cupy-rsoxs" and baseline_startup_mode is not None:
            baseline = baselines.get((baseline_startup_mode, str(row.get("comparison_rotation_key", ""))))
            if baseline is not None:
                enriched_row["comparison_cyrsoxs_baseline_key"] = baseline.get("comparison_key")
                if baseline.get("status") == "ok":
                    enriched_row["comparison_cyrsoxs_primary_seconds"] = baseline.get(
                        "primary_seconds"
                    )
                if row.get("status") == "ok" and baseline.get("status") == "ok":
                    enriched_row["comparison_speedup_vs_cyrsoxs"] = _speedup_vs_baseline(
                        baseline.get("primary_seconds"),
                        row.get("primary_seconds"),
                    )
        enriched.append(enriched_row)
    return enriched


def _build_human_report_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    report_rows: list[dict[str, Any]] = []
    for row in _enrich_results_with_legacy_speedups(rows):
        if row.get("comparison_backend") != "cupy-rsoxs":
            continue
        report_rows.append(
            {
                "Residency": row.get("comparison_residency", ""),
                "Startup": row.get("comparison_startup_mode", ""),
                "Execution path": row.get("comparison_execution_path", ""),
                "z collapse": row.get("comparison_z_collapse_mode") or "off",
                "Rotation": row.get("comparison_rotation_label", ""),
                "Legacy baseline": row.get("comparison_cyrsoxs_baseline_startup_mode", ""),
                "Legacy cyrsoxs": row.get("comparison_cyrsoxs_primary_seconds"),
                "cupy-rsoxs": row.get("primary_seconds"),
                "Speedup vs cyrsoxs": row.get("comparison_speedup_vs_cyrsoxs"),
                "Status": row.get("status", ""),
            }
        )
    return sorted(
        report_rows,
        key=lambda row: (
            {"host": 0, "device": 1}.get(str(row.get("Residency", "")), 99),
            {"warm": 0, "hot": 1, "steady": 2}.get(str(row.get("Startup", "")), 99),
            {"tensor_coeff": 0, "direct_polarization": 1}.get(
                str(row.get("Execution path", "")), 99
            ),
            {"off": 0, "mean": 1}.get(str(row.get("z collapse", "")), 99),
            {"no rotation": 0, "0:5:165": 1}.get(str(row.get("Rotation", "")), 99),
        ),
    )


def _memory_factor_vs_baseline(baseline_value: Any, candidate_value: Any) -> float | None:
    if baseline_value in (None, "") or candidate_value in (None, ""):
        return None
    baseline = float(baseline_value)
    candidate = float(candidate_value)
    if baseline == 0.0:
        return None
    return candidate / baseline


def _build_memory_report_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched_rows = _enrich_results_with_legacy_speedups(rows)
    baselines = _legacy_baseline_lookup(rows)
    report_rows: list[dict[str, Any]] = []
    for row in enriched_rows:
        if row.get("comparison_backend") != "cupy-rsoxs":
            continue

        baseline_startup_mode = str(row.get("comparison_cyrsoxs_baseline_startup_mode", ""))
        baseline = baselines.get((baseline_startup_mode, str(row.get("comparison_rotation_key", ""))))
        candidate_probe = row.get("memory_probe", {})
        baseline_probe = baseline.get("memory_probe", {}) if baseline is not None else {}

        baseline_peak_gpu = baseline_probe.get("peak_gpu_delta_mib")
        baseline_peak_rss = baseline_probe.get("peak_process_rss_mib")
        candidate_peak_gpu = candidate_probe.get("peak_gpu_delta_mib")
        candidate_peak_rss = candidate_probe.get("peak_process_rss_mib")

        report_rows.append(
            {
                "Residency": row.get("comparison_residency", ""),
                "Startup": row.get("comparison_startup_mode", ""),
                "Execution path": row.get("comparison_execution_path", ""),
                "z collapse": row.get("comparison_z_collapse_mode") or "off",
                "Rotation": row.get("comparison_rotation_label", ""),
                "Legacy baseline": baseline_startup_mode,
                "Legacy cyrsoxs peak GPU delta": baseline_peak_gpu,
                "cupy-rsoxs peak GPU delta": candidate_peak_gpu,
                "Peak GPU factor vs cyrsoxs": _memory_factor_vs_baseline(
                    baseline_peak_gpu,
                    candidate_peak_gpu,
                ),
                "Legacy cyrsoxs peak RSS": baseline_peak_rss,
                "cupy-rsoxs peak RSS": candidate_peak_rss,
                "Peak RSS factor vs cyrsoxs": _memory_factor_vs_baseline(
                    baseline_peak_rss,
                    candidate_peak_rss,
                ),
                "Status": row.get("status", ""),
            }
        )

    return sorted(
        report_rows,
        key=lambda row: (
            {"host": 0, "device": 1}.get(str(row.get("Residency", "")), 99),
            {"warm": 0, "hot": 1, "steady": 2}.get(str(row.get("Startup", "")), 99),
            {"tensor_coeff": 0, "direct_polarization": 1}.get(
                str(row.get("Execution path", "")), 99
            ),
            {"off": 0, "mean": 1}.get(str(row.get("z collapse", "")), 99),
            {"no rotation": 0, "0:5:165": 1}.get(str(row.get("Rotation", "")), 99),
        ),
    )


def _fmt_time(value: Any) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):.3f} s"


def _fmt_memory(value: Any) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):.0f} MiB"


def _fmt_speedup(value: Any) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):.2f}x"


def _ascii_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows available._"
    headers = list(rows[0].keys())
    rendered_rows: list[list[str]] = []
    for row in rows:
        rendered = []
        for header in headers:
            value = row[header]
            if "Speedup" in header or "factor vs cyrsoxs" in header:
                rendered.append(_fmt_speedup(value))
            elif header in {
                "Legacy cyrsoxs peak GPU delta",
                "cupy-rsoxs peak GPU delta",
                "Legacy cyrsoxs peak RSS",
                "cupy-rsoxs peak RSS",
            }:
                rendered.append(_fmt_memory(value))
            elif header in {"Legacy cyrsoxs", "cupy-rsoxs"}:
                rendered.append(_fmt_time(value))
            else:
                rendered.append(str(value))
        rendered_rows.append(rendered)

    widths = [
        max(len(str(header)), *(len(row[idx]) for row in rendered_rows))
        for idx, header in enumerate(headers)
    ]

    def _border(sep: str = "+", fill: str = "-") -> str:
        return sep + sep.join(fill * (width + 2) for width in widths) + sep

    def _render_row(values: list[str]) -> str:
        cells = [f" {value.ljust(widths[idx])} " for idx, value in enumerate(values)]
        return "|" + "|".join(cells) + "|"

    lines = [_border(), _render_row(headers), _border()]
    lines.extend(_render_row(row) for row in rendered_rows)
    lines.append(_border())
    return "\n".join(lines)


def _write_table(path: Path, rows: list[dict[str, Any]], *, include_memory: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched_rows = _enrich_results_with_legacy_speedups(rows)
    if include_memory:
        fieldnames = [
            "comparison_key",
            "comparison_backend",
            "comparison_residency",
            "comparison_startup_mode",
            "comparison_execution_path",
            "comparison_path_variant",
            "comparison_z_collapse_mode",
            "comparison_rotation_label",
            "comparison_cyrsoxs_baseline_startup_mode",
            "comparison_cyrsoxs_baseline_key",
            "status",
            "probe_method",
            "baseline_gpu_used_mib",
            "peak_gpu_used_mib",
            "peak_gpu_delta_mib",
            "baseline_process_rss_mib",
            "peak_process_rss_mib",
            "peak_process_rss_delta_mib",
            "poll_interval_s",
            "sample_count",
            "rss_sample_count",
            "gpu_observer_trace_path",
        ]
    else:
        fieldnames = [
            "comparison_key",
            "comparison_backend",
            "comparison_residency",
            "comparison_startup_mode",
            "comparison_execution_path",
            "comparison_path_variant",
            "comparison_z_collapse_mode",
            "comparison_rotation_label",
            "comparison_cyrsoxs_baseline_startup_mode",
            "comparison_cyrsoxs_baseline_key",
            "status",
            "comparison_cyrsoxs_primary_seconds",
            "primary_seconds",
            "comparison_speedup_vs_cyrsoxs",
            "subprocess_seconds",
        ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in sorted(enriched_rows, key=_case_sort_key):
            memory_probe = row.get("memory_probe", {})
            if include_memory:
                record = {
                    "comparison_key": row.get("comparison_key", ""),
                    "comparison_backend": row.get("comparison_backend", ""),
                    "comparison_residency": row.get("comparison_residency", ""),
                    "comparison_startup_mode": row.get("comparison_startup_mode", ""),
                    "comparison_execution_path": row.get("comparison_execution_path", ""),
                    "comparison_path_variant": row.get("comparison_path_variant", ""),
                    "comparison_z_collapse_mode": row.get("comparison_z_collapse_mode", ""),
                    "comparison_rotation_label": row.get("comparison_rotation_label", ""),
                    "comparison_cyrsoxs_baseline_startup_mode": row.get(
                        "comparison_cyrsoxs_baseline_startup_mode", ""
                    ),
                    "comparison_cyrsoxs_baseline_key": row.get("comparison_cyrsoxs_baseline_key", ""),
                    "status": row.get("status", ""),
                    "probe_method": memory_probe.get("probe_method", ""),
                    "baseline_gpu_used_mib": memory_probe.get("baseline_gpu_used_mib", ""),
                    "peak_gpu_used_mib": memory_probe.get("peak_gpu_used_mib", ""),
                    "peak_gpu_delta_mib": memory_probe.get("peak_gpu_delta_mib", ""),
                    "baseline_process_rss_mib": memory_probe.get("baseline_process_rss_mib", ""),
                    "peak_process_rss_mib": memory_probe.get("peak_process_rss_mib", ""),
                    "peak_process_rss_delta_mib": memory_probe.get(
                        "peak_process_rss_delta_mib", ""
                    ),
                    "poll_interval_s": memory_probe.get("poll_interval_s", ""),
                    "sample_count": memory_probe.get("sample_count", ""),
                    "rss_sample_count": memory_probe.get("rss_sample_count", ""),
                    "gpu_observer_trace_path": memory_probe.get("gpu_observer_trace_path", ""),
                }
            else:
                record = {
                    "comparison_key": row.get("comparison_key", ""),
                    "comparison_backend": row.get("comparison_backend", ""),
                    "comparison_residency": row.get("comparison_residency", ""),
                    "comparison_startup_mode": row.get("comparison_startup_mode", ""),
                    "comparison_execution_path": row.get("comparison_execution_path", ""),
                    "comparison_path_variant": row.get("comparison_path_variant", ""),
                    "comparison_z_collapse_mode": row.get("comparison_z_collapse_mode", ""),
                    "comparison_rotation_label": row.get("comparison_rotation_label", ""),
                    "comparison_cyrsoxs_baseline_startup_mode": row.get(
                        "comparison_cyrsoxs_baseline_startup_mode", ""
                    ),
                    "comparison_cyrsoxs_baseline_key": row.get("comparison_cyrsoxs_baseline_key", ""),
                    "status": row.get("status", ""),
                    "comparison_cyrsoxs_primary_seconds": row.get(
                        "comparison_cyrsoxs_primary_seconds", ""
                    ),
                    "primary_seconds": row.get("primary_seconds", ""),
                    "comparison_speedup_vs_cyrsoxs": row.get("comparison_speedup_vs_cyrsoxs", ""),
                    "subprocess_seconds": row.get("subprocess_seconds", ""),
                }
            writer.writerow(record)


def _write_human_report(path: Path, summary: dict[str, Any]) -> None:
    speed_rows = _build_human_report_rows(list(summary["speed_cases"].values()))
    memory_rows = _build_memory_report_rows(list(summary["memory_cases"].values()))
    report = "\n".join(
        [
            f"# Comprehensive Backend Comparison Report ({summary['label']})",
            "",
            f"{summary['size_label'].capitalize()} single-energy CoreShell cross-backend comparison.",
            "",
            (
                "Optional z-collapse rows are included in this report."
                if summary.get("include_z_collapse")
                else "z-collapse rows are disabled for this report."
            ),
            "",
            "Speedup convention:",
            "- host `warm` compares against legacy `cyrsoxs` `warm`.",
            "- host `hot` compares against legacy `cyrsoxs` `hot`.",
            "- device `steady` compares against legacy `cyrsoxs` `warm`.",
            "- device `hot` compares against legacy `cyrsoxs` `hot`.",
            "",
            "## Speed Pass",
            "",
            _ascii_table(speed_rows),
            "",
            "## Memory Pass",
            "",
            "Speed is assessed only from the separate speed-pass series above.",
            "This memory pass reports warmed same-GPU observer GPU deltas and parent-side RSS.",
            "Memory factors are reported as `cupy-rsoxs / cyrsoxs` using the matching legacy baseline row.",
            "",
            _ascii_table(memory_rows),
            "",
        ]
    )
    path.write_text(report, encoding="utf-8")


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")


def _result_summary_line(result: dict[str, Any], *, include_speed: bool) -> str:
    if result.get("status") != "ok":
        return (
            f"{result.get('comparison_key', 'unknown')}: "
            f"{result.get('status')} ({result.get('error_type', 'unknown')})"
        )
    parts = [f"{result['comparison_key']}:"]
    if include_speed:
        parts.append(f"primary {float(result['primary_seconds']):.3f}s")
    probe = result.get("memory_probe")
    if probe:
        if "peak_gpu_used_mib" in probe:
            parts.append(f"peak GPU {float(probe['peak_gpu_used_mib']):.0f} MiB")
        if "peak_gpu_delta_mib" in probe:
            parts.append(f"peak GPU delta {float(probe['peak_gpu_delta_mib']):.0f} MiB")
        if "peak_process_rss_mib" in probe:
            parts.append(f"peak RSS {float(probe['peak_process_rss_mib']):.0f} MiB")
    return ", ".join(parts)


def run_comparison(args: argparse.Namespace) -> int:
    if not has_visible_gpu():
        raise SystemExit("No visible NVIDIA GPU found for the comprehensive backend comparison study.")
    if args.size_label not in SIZE_SPECS:
        raise SystemExit(
            f"--size-label expects one of {sorted(SIZE_SPECS)}, got {args.size_label!r}."
        )

    run_label = args.label or _timestamp()
    run_dir = OUT_ROOT / run_label
    speed_dir = run_dir / "speed_case_results"
    memory_dir = run_dir / "memory_case_results"
    summary_path = run_dir / SUMMARY_NAME
    speed_table_path = run_dir / SPEED_TABLE_NAME
    memory_table_path = run_dir / MEMORY_TABLE_NAME
    report_path = run_dir / REPORT_NAME
    cases = _build_cases(include_z_collapse=bool(args.include_z_collapse), size_label=args.size_label)

    summary = {
        "label": run_label,
        "created_utc": _timestamp(),
        "python_executable": sys.executable,
        "gpu_index": int(args.gpu_index),
        "size_label": args.size_label,
        "energies_ev": list(CORE_SHELL_SINGLE_ENERGIES),
        "include_z_collapse": bool(args.include_z_collapse),
        "rotations": [
            {"key": key, "label": label, "eangle_rotation": list(spec)}
            for key, spec, label in ROTATION_SPECS
        ],
        "host_startup_modes": list(HOST_STARTUP_MODES),
        "device_startup_modes": list(DEVICE_STARTUP_MODES),
        "memory_poll_interval_s": float(args.memory_poll_interval_s),
        "speed_cases": {},
        "memory_cases": {},
    }

    run_dir.mkdir(parents=True, exist_ok=True)

    print("Running comprehensive speed pass...", flush=True)
    for case in cases:
        result = _run_case_subprocess(
            case=case,
            output_dir=speed_dir,
            gpu_index=args.gpu_index,
            monitor_memory=False,
            poll_interval_s=args.memory_poll_interval_s,
            skip_existing=not args.no_skip_existing,
        )
        summary["speed_cases"][case.key] = result
        _write_summary(summary_path, summary)
        print(_result_summary_line(result, include_speed=True), flush=True)

    print("Running comprehensive memory pass...", flush=True)
    for case in cases:
        result = _run_case_subprocess(
            case=case,
            output_dir=memory_dir,
            gpu_index=args.gpu_index,
            monitor_memory=True,
            poll_interval_s=args.memory_poll_interval_s,
            skip_existing=not args.no_skip_existing,
        )
        summary["memory_cases"][case.key] = result
        _write_summary(summary_path, summary)
        print(_result_summary_line(result, include_speed=False), flush=True)

    _write_table(speed_table_path, list(summary["speed_cases"].values()), include_memory=False)
    _write_table(memory_table_path, list(summary["memory_cases"].values()), include_memory=True)
    _write_human_report(report_path, summary)
    _write_summary(summary_path, summary)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {speed_table_path}", flush=True)
    print(f"Wrote {memory_table_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only comprehensive cross-backend single-size CoreShell comparison. "
            "Runs separate speed and memory passes across host warm/hot and "
            "device steady/hot lanes. The memory pass uses a warmed same-GPU "
            "CuPy memGetInfo observer plus process RSS polling."
        )
    )
    parser.add_argument("--label", default=None, help="Output subdirectory label under test-reports.")
    parser.add_argument("--gpu-index", type=int, default=0, help="Global GPU index to pin for serial runs.")
    parser.add_argument(
        "--size-label",
        default="small",
        help="CoreShell size label to run, for example 'small', 'medium', or 'large'.",
    )
    parser.add_argument(
        "--memory-poll-interval-s",
        type=float,
        default=0.01,
        help=(
            "Sampling cadence for the warmed CuPy GPU-memory observer and the "
            "parent-side RSS polling during the separate memory pass."
        ),
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Rerun cases even if completed case result files already exist.",
    )
    parser.add_argument(
        "--include-z-collapse",
        action="store_true",
        help=(
            "Add opt-in cupy-rsoxs tensor_coeff z_collapse_mode='mean' rows to the "
            "comprehensive speed and memory passes."
        ),
    )
    return parser


def main() -> int:
    return run_comparison(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
