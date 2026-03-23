#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import asdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

from tests.validation.dev.cupy_rsoxs_optimization.run_cupy_rsoxs_optimization_matrix import (  # noqa: E402
    SIZE_SPECS,
    build_scaled_core_shell_morphology,
)
from tests.validation.lib.core_shell import (  # noqa: E402
    _load_optical_constants,
    awedge_comparison_slices,
    compute_awedge_metrics,
    has_visible_gpu,
    release_runtime_memory,
    scattering_to_awedge,
)


OUT_ROOT = REPO_ROOT / "test-reports" / "core-shell-backend-performance-dev"
SUMMARY_NAME = "summary.json"
TABLE_NAME = "performance_table.tsv"
FIGURE_NAME = "core_shell_backend_performance_graphical_abstract.png"


@dataclass(frozen=True)
class PathSpec:
    key: str
    label: str
    backend: str
    input_policy: str
    ownership_policy: str | None
    field_namespace: str
    color: str
    marker: str
    zorder: int


@dataclass(frozen=True)
class AngleSpec:
    key: str
    label: str
    eangle_rotation: tuple[float, float, float]
    num_angles: int


@dataclass(frozen=True)
class PerformanceCase:
    key: str
    study_phase: str
    path_key: str
    size_label: str
    angle_key: str
    energies_ev: tuple[float, ...]
    eangle_rotation: tuple[float, float, float]
    num_angles: int
    backend: str
    input_policy: str
    ownership_policy: str | None
    field_namespace: str
    collect_awedge: bool


PATH_SPECS = (
    PathSpec(
        key="cyrsoxs_numpy",
        label="numpy -> cyrsoxs",
        backend="cyrsoxs",
        input_policy="coerce",
        ownership_policy=None,
        field_namespace="numpy",
        color="#111111",
        marker="o",
        zorder=3,
    ),
    PathSpec(
        key="cupy_coerce_numpy",
        label="numpy -> cupy-rsoxs (coerce)",
        backend="cupy-rsoxs",
        input_policy="coerce",
        ownership_policy=None,
        field_namespace="numpy",
        color="#c45a00",
        marker="s",
        zorder=4,
    ),
    PathSpec(
        key="cupy_borrow_cupy",
        label="cupy -> cupy-rsoxs (borrow)",
        backend="cupy-rsoxs",
        input_policy="strict",
        ownership_policy="borrow",
        field_namespace="cupy",
        color="#007c91",
        marker="^",
        zorder=5,
    ),
)


ANGLE_SPECS = (
    AngleSpec(
        key="off",
        label="1 angle | [0, 0, 0]",
        eangle_rotation=(0.0, 0.0, 0.0),
        num_angles=1,
    ),
    AngleSpec(
        key="step30",
        label="12 angles | [0, 30, 330]",
        eangle_rotation=(0.0, 30.0, 330.0),
        num_angles=12,
    ),
    AngleSpec(
        key="step15",
        label="24 angles | [0, 15, 345]",
        eangle_rotation=(0.0, 15.0, 345.0),
        num_angles=24,
    ),
    AngleSpec(
        key="step5",
        label="72 angles | [0, 5, 355]",
        eangle_rotation=(0.0, 5.0, 355.0),
        num_angles=72,
    ),
)


PATH_BY_KEY = {spec.key: spec for spec in PATH_SPECS}
ANGLE_BY_KEY = {spec.key: spec for spec in ANGLE_SPECS}
SIZE_LABEL_ORDER = tuple(SIZE_SPECS)


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _parse_csv_labels(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _ordered_unique_size_labels(labels: list[str] | tuple[str, ...]) -> list[str]:
    wanted = set(labels)
    ordered = [label for label in SIZE_LABEL_ORDER if label in wanted]
    extras = sorted(wanted.difference(SIZE_LABEL_ORDER))
    ordered.extend(extras)
    return ordered


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


@lru_cache(maxsize=1)
def _full_core_shell_energies() -> tuple[float, ...]:
    constants = _load_optical_constants()
    return tuple(map(float, constants[1].energies))


def _case_key(*, path_key: str, size_label: str, angle_key: str) -> str:
    return f"{path_key}__{size_label}__{angle_key}"


def _build_speed_cases(size_labels: tuple[str, ...]) -> list[PerformanceCase]:
    energies = _full_core_shell_energies()
    cases: list[PerformanceCase] = []
    for size_label in size_labels:
        for angle in ANGLE_SPECS:
            for path in PATH_SPECS:
                cases.append(
                    PerformanceCase(
                        key=_case_key(path_key=path.key, size_label=size_label, angle_key=angle.key),
                        study_phase="speed",
                        path_key=path.key,
                        size_label=size_label,
                        angle_key=angle.key,
                        energies_ev=energies,
                        eangle_rotation=angle.eangle_rotation,
                        num_angles=angle.num_angles,
                        backend=path.backend,
                        input_policy=path.input_policy,
                        ownership_policy=path.ownership_policy,
                        field_namespace=path.field_namespace,
                        collect_awedge=True,
                    )
                )
    return cases


def _build_memory_cases(size_labels: tuple[str, ...]) -> list[PerformanceCase]:
    energies = _full_core_shell_energies()
    angle = ANGLE_BY_KEY["step5"]
    cases: list[PerformanceCase] = []
    for size_label in size_labels:
        for path in PATH_SPECS:
            cases.append(
                PerformanceCase(
                    key=_case_key(path_key=path.key, size_label=size_label, angle_key=angle.key),
                    study_phase="memory",
                    path_key=path.key,
                    size_label=size_label,
                    angle_key=angle.key,
                    energies_ev=energies,
                    eangle_rotation=angle.eangle_rotation,
                    num_angles=angle.num_angles,
                    backend=path.backend,
                    input_policy=path.input_policy,
                    ownership_policy=path.ownership_policy,
                    field_namespace=path.field_namespace,
                    collect_awedge=False,
                )
            )
    return cases


def _save_awedge_npz(awedge: xr.DataArray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        data=np.asarray(awedge.values, dtype=np.float64),
        energy=np.asarray(awedge.coords["energy"].values, dtype=np.float64),
        q=np.asarray(awedge.coords["q"].values, dtype=np.float64),
    )


def _load_awedge_npz(path: Path) -> xr.DataArray:
    payload = np.load(path)
    return xr.DataArray(
        np.asarray(payload["data"], dtype=np.float64),
        dims=("energy", "q"),
        coords={
            "energy": np.asarray(payload["energy"], dtype=np.float64),
            "q": np.asarray(payload["q"], dtype=np.float64),
        },
        name="A_awedge",
    ).sortby("energy")


def _relative_rmse(current: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    diff = current - reference
    rmse = float(np.sqrt(np.mean(diff * diff, dtype=np.float64)))
    ref_rms = float(np.sqrt(np.mean(reference * reference, dtype=np.float64)))
    return rmse, float(rmse / max(ref_rms, 1.0e-30))


def _compare_awedge(current: xr.DataArray, reference: xr.DataArray) -> dict[str, Any]:
    current = current.sortby("energy")
    reference = reference.sortby("energy")

    cur_energy = np.asarray(current.coords["energy"].values, dtype=np.float64)
    ref_energy = np.asarray(reference.coords["energy"].values, dtype=np.float64)
    cur_q = np.asarray(current.coords["q"].values, dtype=np.float64)
    ref_q = np.asarray(reference.coords["q"].values, dtype=np.float64)
    if not (np.allclose(cur_energy, ref_energy) and np.allclose(cur_q, ref_q)):
        raise AssertionError("A-wedge coordinates drifted between compared backends.")

    current_vals = np.asarray(current.values, dtype=np.float64)
    reference_vals = np.asarray(reference.values, dtype=np.float64)
    finite = np.isfinite(current_vals) & np.isfinite(reference_vals)
    if not np.any(finite):
        raise AssertionError("No overlapping finite A-wedge values found for backend comparison.")

    cur = current_vals[finite]
    ref = reference_vals[finite]
    rmse, rel_rmse = _relative_rmse(cur, ref)
    comparison = awedge_comparison_slices(awedge=current, reference=reference)
    slice_metrics = compute_awedge_metrics(comparison)

    def series_relative(series_key: str, ref_key: str) -> float:
        series = np.asarray(comparison[series_key].values, dtype=np.float64)
        ref_series = np.asarray(comparison[ref_key].values, dtype=np.float64)
        mask = np.isfinite(series) & np.isfinite(ref_series)
        if not np.any(mask):
            return float("nan")
        _, rel = _relative_rmse(series[mask], ref_series[mask])
        return float(rel)

    return {
        "finite_fraction": float(np.count_nonzero(finite)) / float(current_vals.size),
        "full_rmse": rmse,
        "full_relative_rmse": rel_rmse,
        "full_max_abs_diff": float(np.max(np.abs(cur - ref))),
        "a_vs_energy_rmse": float(slice_metrics["a_vs_energy"]["rmse"]),
        "a_vs_energy_max_abs_diff": float(slice_metrics["a_vs_energy"]["max_abs_diff"]),
        "a_vs_energy_relative_rmse": series_relative("a_vs_energy", "reference_a_vs_energy"),
        "a_vs_q_284p7_rmse": float(slice_metrics["a_vs_q_284p7"]["rmse"]),
        "a_vs_q_284p7_max_abs_diff": float(slice_metrics["a_vs_q_284p7"]["max_abs_diff"]),
        "a_vs_q_284p7_relative_rmse": series_relative("a_vs_q_284p7", "reference_a_vs_q_284p7"),
        "a_vs_q_285p2_rmse": float(slice_metrics["a_vs_q_285p2"]["rmse"]),
        "a_vs_q_285p2_max_abs_diff": float(slice_metrics["a_vs_q_285p2"]["max_abs_diff"]),
        "a_vs_q_285p2_relative_rmse": series_relative("a_vs_q_285p2", "reference_a_vs_q_285p2"),
    }


def _write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")


def _sample_gpu_memory_used_mib(gpu_index: int) -> float | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_index),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return None

    if completed.returncode != 0:
        return None
    line = completed.stdout.strip().splitlines()
    if not line:
        return None
    try:
        return float(line[0].strip())
    except ValueError:
        return None


def _poll_gpu_memory_until_exit(
    proc: subprocess.Popen[str],
    *,
    gpu_index: int,
    poll_interval_s: float,
) -> dict[str, float] | None:
    baseline = _sample_gpu_memory_used_mib(gpu_index)
    if baseline is None:
        return None

    peak = baseline
    samples = 1
    while proc.poll() is None:
        time.sleep(poll_interval_s)
        sample = _sample_gpu_memory_used_mib(gpu_index)
        if sample is None:
            continue
        samples += 1
        peak = max(peak, sample)

    final_sample = _sample_gpu_memory_used_mib(gpu_index)
    if final_sample is not None:
        samples += 1
        peak = max(peak, final_sample)

    return {
        "baseline_used_mib": float(baseline),
        "peak_used_mib": float(peak),
        "peak_delta_mib": float(max(0.0, peak - baseline)),
        "sample_count": float(samples),
        "poll_interval_s": float(poll_interval_s),
    }


def _worker_main(case_path: Path, result_path: Path, awedge_path: Path | None) -> int:
    started = time.perf_counter()
    case = PerformanceCase(**json.loads(case_path.read_text()))
    result: dict[str, Any] = {
        "key": case.key,
        "study_phase": case.study_phase,
        "path_key": case.path_key,
        "size_label": case.size_label,
        "angle_key": case.angle_key,
        "energies_ev": list(case.energies_ev),
        "eangle_rotation": list(case.eangle_rotation),
        "num_angles": case.num_angles,
        "backend": case.backend,
        "input_policy": case.input_policy,
        "ownership_policy": case.ownership_policy,
        "field_namespace": case.field_namespace,
        "status": "error",
    }

    morphology = None
    backend_result = None
    scattering = None
    awedge = None
    try:
        build_start = time.perf_counter()
        morphology = build_scaled_core_shell_morphology(
            size_spec=SIZE_SPECS[case.size_label],
            energies_ev=case.energies_ev,
            eangle_rotation=case.eangle_rotation,
            backend=case.backend,
            field_namespace=case.field_namespace,
            input_policy=case.input_policy,
            ownership_policy=case.ownership_policy,
            create_cy_object=True,
        )
        result["build_seconds"] = time.perf_counter() - build_start

        run_start = time.perf_counter()
        backend_result = morphology.run(stdout=False, stderr=False, return_xarray=False)
        result["run_seconds"] = time.perf_counter() - run_start
        result["backend_timings"] = morphology.backend_timings

        export_start = time.perf_counter()
        if case.backend == "cyrsoxs":
            scattering = morphology.scattering_to_xarray(return_xarray=True)
        else:
            scattering = backend_result.to_xarray()
        result["export_seconds"] = time.perf_counter() - export_start
        result["panel_shape"] = list(scattering.shape)

        if case.collect_awedge:
            awedge_start = time.perf_counter()
            awedge = scattering_to_awedge(scattering)
            result["awedge_seconds"] = time.perf_counter() - awedge_start
            if awedge_path is not None:
                _save_awedge_npz(awedge, awedge_path)
                result["awedge_path"] = str(awedge_path)

        result["workflow_seconds"] = (
            result["build_seconds"] + result["run_seconds"] + result["export_seconds"]
        )
        result["worker_elapsed_seconds"] = time.perf_counter() - started
        result["status"] = "ok"

    except BaseException as exc:  # noqa: BLE001
        result["worker_elapsed_seconds"] = time.perf_counter() - started
        result["status"] = "error"
        result["error_type"] = exc.__class__.__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        if morphology is not None:
            try:
                morphology.release_runtime()
            except Exception:
                pass
        del awedge, scattering, backend_result, morphology
        release_runtime_memory()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2, default=_json_default) + "\n")
    return 0


def _run_case_subprocess(
    *,
    case: PerformanceCase,
    result_dir: Path,
    awedge_dir: Path | None,
    gpu_index: int,
    monitor_memory: bool,
    poll_interval_s: float,
    skip_existing: bool,
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    if awedge_dir is not None:
        awedge_dir.mkdir(parents=True, exist_ok=True)

    result_path = result_dir / f"{case.key}.json"
    awedge_path = awedge_dir / f"{case.key}.npz" if awedge_dir is not None and case.collect_awedge else None

    if skip_existing and result_path.exists():
        existing = json.loads(result_path.read_text())
        awedge_ok = (
            True
            if awedge_path is None
            else Path(existing.get("awedge_path", awedge_path)).exists()
        )
        if existing.get("status") == "ok" and awedge_ok:
            existing["reused_existing"] = True
            return existing

    with tempfile.TemporaryDirectory(prefix="core_shell_backend_perf_", dir=result_dir) as tmp_dir:
        tmp_path = Path(tmp_dir)
        case_path = tmp_path / "case.json"
        worker_result_path = tmp_path / "result.json"
        case_path.write_text(json.dumps(asdict(case), indent=2, default=_json_default) + "\n")

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-case-path",
            str(case_path),
            "--worker-result-path",
            str(worker_result_path),
        ]
        if awedge_path is not None:
            cmd.extend(["--worker-awedge-path", str(awedge_path)])

        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

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
        if monitor_memory:
            memory_probe = _poll_gpu_memory_until_exit(
                proc,
                gpu_index=gpu_index,
                poll_interval_s=poll_interval_s,
            )
        stdout, stderr = proc.communicate()
        elapsed = time.perf_counter() - started

        if worker_result_path.exists():
            result = json.loads(worker_result_path.read_text())
        else:
            result = {
                "key": case.key,
                "study_phase": case.study_phase,
                "path_key": case.path_key,
                "size_label": case.size_label,
                "angle_key": case.angle_key,
                "status": "subprocess_failed",
                "error_type": "SubprocessFailure",
                "error": "Worker exited before writing a result file.",
            }

        result["subprocess_returncode"] = int(proc.returncode)
        result["subprocess_seconds"] = elapsed
        if stdout.strip():
            result["worker_stdout"] = stdout[-4000:]
        if stderr.strip():
            result["worker_stderr"] = stderr[-4000:]
        if memory_probe is not None:
            result["memory_probe"] = memory_probe

        result_path.write_text(json.dumps(result, indent=2, default=_json_default) + "\n")
        return result


def _attach_reference_metrics(summary: dict[str, Any]) -> None:
    speed_cases = summary.setdefault("speed_cases", {})

    for size_label in SIZE_LABEL_ORDER:
        for angle in ANGLE_SPECS:
            reference_key = _case_key(
                path_key="cyrsoxs_numpy",
                size_label=size_label,
                angle_key=angle.key,
            )
            reference_result = speed_cases.get(reference_key)
            if not reference_result or reference_result.get("status") != "ok":
                continue
            reference_awedge_path = Path(reference_result["awedge_path"])
            reference_awedge = _load_awedge_npz(reference_awedge_path)

            for path in PATH_SPECS:
                if path.key == "cyrsoxs_numpy":
                    continue
                case_key = _case_key(
                    path_key=path.key,
                    size_label=size_label,
                    angle_key=angle.key,
                )
                result = speed_cases.get(case_key)
                if not result or result.get("status") != "ok":
                    continue
                awedge_path = Path(result["awedge_path"])
                current_awedge = _load_awedge_npz(awedge_path)
                result["comparison_to_cyrsoxs"] = _compare_awedge(
                    current=current_awedge,
                    reference=reference_awedge,
                )
                result["speedup_vs_cyrsoxs"] = float(reference_result["workflow_seconds"]) / float(
                    result["workflow_seconds"]
                )

    awedge_dirs = sorted(
        {
            str(Path(result["awedge_path"]).parent)
            for result in speed_cases.values()
            if result.get("status") == "ok" and result.get("awedge_path")
        }
    )
    if awedge_dirs:
        summary["awedge_dirs"] = awedge_dirs


def _select_overlay_case(summary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str] | None:
    speed_cases = summary.get("speed_cases", {})
    for size_label in ("large", "medium", "small"):
        for angle_key in ("step5", "step15", "step30", "off"):
            reference_key = _case_key(
                path_key="cyrsoxs_numpy",
                size_label=size_label,
                angle_key=angle_key,
            )
            reference = speed_cases.get(reference_key)
            if not reference or reference.get("status") != "ok":
                continue

            candidates: list[tuple[float, dict[str, Any]]] = []
            for path in PATH_SPECS:
                if path.key == "cyrsoxs_numpy":
                    continue
                case_key = _case_key(
                    path_key=path.key,
                    size_label=size_label,
                    angle_key=angle_key,
                )
                result = speed_cases.get(case_key)
                if not result or result.get("status") != "ok":
                    continue
                if "comparison_to_cyrsoxs" not in result:
                    continue
                candidates.append((float(result["workflow_seconds"]), result))

            if not candidates:
                continue
            candidates.sort(key=lambda item: item[0])
            return reference, candidates[0][1], size_label
    return None


def _summary_text_lines(summary: dict[str, Any]) -> list[str]:
    speed_cases = summary.get("speed_cases", {})
    memory_cases = summary.get("memory_cases", {})
    lines = [
        "Legacy CoreShell backend performance comparison",
        "",
        "Historical harness only. Not authoritative for current optimization timing.",
        "Timing metric: build + backend run + xarray export",
        f"Full baseline: {summary['full_energy_count']} energies",
        "EAngleRotation unique-state sweep: 1, 12, 24, 72 angles",
        "",
        "72-angle speedup vs numpy -> cyrsoxs:",
    ]

    for size_label in ("small", "medium", "large"):
        reference_key = _case_key(
            path_key="cyrsoxs_numpy",
            size_label=size_label,
            angle_key="step5",
        )
        reference = speed_cases.get(reference_key)
        if not reference or reference.get("status") != "ok":
            lines.append(f"- {size_label}: cyrsoxs missing")
            continue

        ref_time = float(reference["workflow_seconds"])
        segments = [size_label]
        for path_key in ("cupy_coerce_numpy", "cupy_borrow_cupy"):
            result = speed_cases.get(
                _case_key(path_key=path_key, size_label=size_label, angle_key="step5")
            )
            if not result or result.get("status") != "ok":
                segments.append(f"{PATH_BY_KEY[path_key].label}=n/a")
                continue
            speedup = ref_time / float(result["workflow_seconds"])
            segments.append(f"{path_key.split('_')[1]} {speedup:.2f}x")
        lines.append("- " + ", ".join(segments))

    if memory_cases:
        lines.extend(["", "Separate 72-angle peak-memory pass:"])
        for size_label in ("small", "medium", "large"):
            parts = [size_label]
            for path_key in ("cyrsoxs_numpy", "cupy_coerce_numpy", "cupy_borrow_cupy"):
                result = memory_cases.get(
                    _case_key(path_key=path_key, size_label=size_label, angle_key="step5")
                )
                probe = None if result is None else result.get("memory_probe")
                if not probe:
                    parts.append(f"{path_key.split('_')[0]} n/a")
                    continue
                parts.append(
                    f"{path_key.split('_')[0]} peak {float(probe['peak_used_mib']):.0f} MiB"
                )
            lines.append("- " + "; ".join(parts))

    return lines


def _plot_graphical_abstract(summary: dict[str, Any], out_path: Path) -> None:
    overlay = _select_overlay_case(summary)
    if overlay is None:
        raise AssertionError("No successful shared cyrsoxs/CuPy case available for overlay plotting.")
    reference_result, comparison_result, overlay_size_label = overlay
    comparison_path = PATH_BY_KEY[comparison_result["path_key"]]
    overlay_angle = ANGLE_BY_KEY[comparison_result["angle_key"]]

    reference_awedge = _load_awedge_npz(Path(reference_result["awedge_path"]))
    comparison_awedge = _load_awedge_npz(Path(comparison_result["awedge_path"]))
    comparison = awedge_comparison_slices(awedge=comparison_awedge, reference=reference_awedge)

    fig = plt.figure(figsize=(16.0, 9.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.1))

    legend_handles = []
    legend_labels = []
    y_limits: list[float] = []
    speed_cases = summary.get("speed_cases", {})

    for col, size_label in enumerate(("small", "medium", "large")):
        ax = fig.add_subplot(grid[0, col])
        size_spec = SIZE_SPECS[size_label]
        x_ticks = [angle.num_angles for angle in ANGLE_SPECS]
        for path in PATH_SPECS:
            xs: list[int] = []
            ys: list[float] = []
            for angle in ANGLE_SPECS:
                result = speed_cases.get(_case_key(path_key=path.key, size_label=size_label, angle_key=angle.key))
                if not result or result.get("status") != "ok":
                    continue
                xs.append(angle.num_angles)
                ys.append(float(result["workflow_seconds"]))
            if not xs:
                continue
            (line,) = ax.plot(
                xs,
                ys,
                color=path.color,
                marker=path.marker,
                linewidth=2.2,
                markersize=6.5,
                zorder=path.zorder,
            )
            if col == 0:
                legend_handles.append(line)
                legend_labels.append(path.label)
            y_limits.extend(ys)

        ax.set_title(f"{size_label.capitalize()} {size_spec.shape}", fontsize=11)
        ax.set_xlabel("Actual angle count")
        if col == 0:
            ax.set_ylabel("Workflow seconds")
        ax.set_xticks(x_ticks)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.55)

    if y_limits:
        ymin = min(y_limits) * 0.8
        ymax = max(y_limits) * 1.25
        for col in range(3):
            fig.axes[col].set_ylim(ymin, ymax)

    ax_energy = fig.add_subplot(grid[1, 0])
    ax_energy.plot(
        comparison["reference_a_vs_energy"].coords["energy"].values,
        comparison["reference_a_vs_energy"].values,
        color=PATH_BY_KEY["cyrsoxs_numpy"].color,
        linewidth=1.8,
        marker="o",
        markersize=2.6,
        markevery=6,
        alpha=0.9,
        label=PATH_BY_KEY["cyrsoxs_numpy"].label,
    )
    ax_energy.plot(
        comparison["a_vs_energy"].coords["energy"].values,
        comparison["a_vs_energy"].values,
        color=comparison_path.color,
        linewidth=2.2,
        linestyle="--",
        marker=comparison_path.marker,
        markersize=2.8,
        markevery=6,
        alpha=0.8,
        label=comparison_path.label,
    )
    ax_energy.set_title(
        f"A(E) parity overlay | {overlay_size_label} | {overlay_angle.num_angles} angles",
        fontsize=11,
    )
    ax_energy.set_xlabel("Energy [eV]")
    ax_energy.set_ylabel("A(E)")
    ax_energy.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
    ax_energy.legend(loc="best", fontsize=8)

    ax_q = fig.add_subplot(grid[1, 1])
    ax_q.plot(
        comparison["reference_a_vs_q_284p7"].coords["q"].values,
        comparison["reference_a_vs_q_284p7"].values,
        color="#111111",
        linewidth=1.7,
        label="284.7 eV cyrsoxs",
    )
    ax_q.plot(
        comparison["a_vs_q_284p7"].coords["q"].values,
        comparison["a_vs_q_284p7"].values,
        color="#111111",
        linewidth=2.0,
        linestyle="--",
        alpha=0.8,
        label=f"284.7 eV {comparison_path.key}",
    )
    ax_q.plot(
        comparison["reference_a_vs_q_285p2"].coords["q"].values,
        comparison["reference_a_vs_q_285p2"].values,
        color="#1f77b4",
        linewidth=1.7,
        label="285.2 eV cyrsoxs",
    )
    ax_q.plot(
        comparison["a_vs_q_285p2"].coords["q"].values,
        comparison["a_vs_q_285p2"].values,
        color="#1f77b4",
        linewidth=2.0,
        linestyle="--",
        alpha=0.8,
        label=f"285.2 eV {comparison_path.key}",
    )
    ax_q.set_title("A(q) parity overlay with residual inset", fontsize=11)
    ax_q.set_xlabel(r"q [nm$^{-1}$]")
    ax_q.set_ylabel("A(q)")
    ax_q.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
    ax_q.legend(loc="best", fontsize=7)

    inset = ax_q.inset_axes([0.54, 0.10, 0.42, 0.36])
    inset.plot(
        comparison["reference_a_vs_q_284p7"].coords["q"].values,
        comparison["a_vs_q_284p7"].values - comparison["reference_a_vs_q_284p7"].values,
        color="#111111",
        linewidth=1.4,
        label="284.7 eV",
    )
    inset.plot(
        comparison["reference_a_vs_q_285p2"].coords["q"].values,
        comparison["a_vs_q_285p2"].values - comparison["reference_a_vs_q_285p2"].values,
        color="#1f77b4",
        linewidth=1.4,
        label="285.2 eV",
    )
    inset.axhline(0.0, color="#666666", linewidth=0.8)
    inset.set_title("delta A(q)", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.grid(True, linestyle=":", linewidth=0.5, alpha=0.45)

    panel = grid[1, 2].subgridspec(2, 1, height_ratios=(1.0, 0.9))
    ax_heat = fig.add_subplot(panel[0, 0])
    rows: list[str] = []
    heat = np.full((12, 2), np.nan, dtype=np.float64)
    row_index = 0
    for size_label in ("small", "medium", "large"):
        for angle in ANGLE_SPECS:
            rows.append(f"{size_label[:1].upper()} | {angle.num_angles}")
            for col, path_key in enumerate(("cupy_coerce_numpy", "cupy_borrow_cupy")):
                result = speed_cases.get(
                    _case_key(path_key=path_key, size_label=size_label, angle_key=angle.key)
                )
                if result and result.get("status") == "ok" and "comparison_to_cyrsoxs" in result:
                    heat[row_index, col] = (
                        float(result["comparison_to_cyrsoxs"]["a_vs_energy_relative_rmse"]) * 100.0
                    )
            row_index += 1

    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad(color="#f0f0f0")
    image = ax_heat.imshow(heat, aspect="auto", cmap=cmap)
    ax_heat.set_title("A(E) relative RMSE vs cyrsoxs [%]", fontsize=11)
    ax_heat.set_xticks([0, 1], labels=["coerce", "borrow"])
    ax_heat.set_yticks(range(len(rows)), labels=rows)
    ax_heat.tick_params(axis="both", labelsize=8)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat[i, j]
            if np.isnan(value):
                text = "n/a"
                color = "#333333"
            else:
                text = f"{value:.3f}"
                color = "#111111" if value < np.nanmax(heat) * 0.55 else "white"
            ax_heat.text(j, i, text, ha="center", va="center", fontsize=7, color=color)
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.02)
    colorbar.ax.tick_params(labelsize=8)

    ax_text = fig.add_subplot(panel[1, 0])
    ax_text.axis("off")
    text_lines = _summary_text_lines(summary)
    overlay_metrics = comparison_result["comparison_to_cyrsoxs"]
    text_lines.extend(
        [
            "",
            "Overlay-case drift metrics:",
            f"- full relative RMSE: {100.0 * float(overlay_metrics['full_relative_rmse']):.4f}%",
            f"- A(E) relative RMSE: {100.0 * float(overlay_metrics['a_vs_energy_relative_rmse']):.4f}%",
            f"- A(q) 284.7 relative RMSE: {100.0 * float(overlay_metrics['a_vs_q_284p7_relative_rmse']):.4f}%",
            f"- A(q) 285.2 relative RMSE: {100.0 * float(overlay_metrics['a_vs_q_285p2_relative_rmse']):.4f}%",
        ]
    )
    ax_text.text(
        0.0,
        1.0,
        "\n".join(text_lines),
        ha="left",
        va="top",
        fontsize=8.5,
        family="monospace",
    )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "CoreShell full-energy backend comparison: cyrsoxs vs cupy-rsoxs",
        fontsize=15,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        format="png",
        dpi=170,
        bbox_inches="tight",
        metadata={
            "Title": "CoreShell backend performance graphical abstract",
            "Description": (
                "Full-energy CoreShell backend comparison across cyrsoxs and cupy-rsoxs "
                "paths with EAngleRotation sweeps and A-wedge parity overlays."
            ),
            "Author": "OpenAI Codex",
            "Software": "matplotlib",
        },
    )
    plt.close(fig)


def _write_table(summary: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "study_phase",
                "path_key",
                "size_label",
                "angle_key",
                "num_angles",
                "status",
                "build_seconds",
                "run_seconds",
                "export_seconds",
                "awedge_seconds",
                "workflow_seconds",
                "speedup_vs_cyrsoxs",
                "a_vs_energy_relative_rmse_pct_vs_cyrsoxs",
                "a_vs_q_284p7_relative_rmse_pct_vs_cyrsoxs",
                "a_vs_q_285p2_relative_rmse_pct_vs_cyrsoxs",
                "full_relative_rmse_pct_vs_cyrsoxs",
                "peak_used_mib",
                "peak_delta_mib",
            ]
        )
        for section_name in ("speed_cases", "memory_cases"):
            for case_key in sorted(summary.get(section_name, {})):
                result = summary[section_name][case_key]
                comparison = result.get("comparison_to_cyrsoxs", {})
                probe = result.get("memory_probe", {})
                writer.writerow(
                    [
                        result.get("study_phase", section_name.replace("_cases", "")),
                        result.get("path_key", ""),
                        result.get("size_label", ""),
                        result.get("angle_key", ""),
                        result.get("num_angles", ""),
                        result.get("status", ""),
                        result.get("build_seconds", ""),
                        result.get("run_seconds", ""),
                        result.get("export_seconds", ""),
                        result.get("awedge_seconds", ""),
                        result.get("workflow_seconds", ""),
                        (
                            f"{float(result['speedup_vs_cyrsoxs']):.3f}x"
                            if "speedup_vs_cyrsoxs" in result
                            else ""
                        ),
                        (
                            100.0 * float(comparison["a_vs_energy_relative_rmse"])
                            if "a_vs_energy_relative_rmse" in comparison
                            else ""
                        ),
                        (
                            100.0 * float(comparison["a_vs_q_284p7_relative_rmse"])
                            if "a_vs_q_284p7_relative_rmse" in comparison
                            else ""
                        ),
                        (
                            100.0 * float(comparison["a_vs_q_285p2_relative_rmse"])
                            if "a_vs_q_285p2_relative_rmse" in comparison
                            else ""
                        ),
                        (
                            100.0 * float(comparison["full_relative_rmse"])
                            if "full_relative_rmse" in comparison
                            else ""
                        ),
                        probe.get("peak_used_mib", ""),
                        probe.get("peak_delta_mib", ""),
                    ]
                )


def _result_summary_line(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return (
            f"{result.get('key', 'unknown')}: "
            f"{result.get('status')} ({result.get('error_type', 'unknown')})"
        )
    parts = [
        f"{result['key']}: workflow {float(result['workflow_seconds']):.3f}s",
        f"build {float(result['build_seconds']):.3f}s",
        f"run {float(result['run_seconds']):.3f}s",
        f"export {float(result['export_seconds']):.3f}s",
    ]
    if "memory_probe" in result:
        probe = result["memory_probe"]
        parts.append(f"peak {float(probe['peak_used_mib']):.0f} MiB")
    return ", ".join(parts)


def _validate_mergeable_summary(
    *,
    source_label: str,
    source_summary: dict[str, Any],
    reference_summary: dict[str, Any],
) -> None:
    for key in ("full_energy_count", "angle_sweep", "path_specs"):
        if source_summary.get(key) != reference_summary.get(key):
            raise SystemExit(
                f"Cannot merge {source_label!r}: summary field {key!r} does not match the first study."
            )


def merge_studies(args: argparse.Namespace) -> int:
    merge_labels = _parse_csv_labels(args.merge_labels)
    if not merge_labels:
        raise SystemExit("merge_studies requires at least one source label.")

    output_label = args.label or "__".join(merge_labels)
    run_dir = OUT_ROOT / output_label
    merged_summary: dict[str, Any] | None = None
    merged_size_labels: list[str] = []
    reference_summary: dict[str, Any] | None = None

    for source_label in merge_labels:
        source_path = OUT_ROOT / source_label / SUMMARY_NAME
        if not source_path.exists():
            raise SystemExit(f"Source summary not found: {source_path}")
        source_summary = json.loads(source_path.read_text())
        if reference_summary is None:
            reference_summary = source_summary
            merged_summary = {
                "label": output_label,
                "created_utc": _timestamp(),
                "python_executable": sys.executable,
                "gpu_index": None,
                "full_energy_count": source_summary["full_energy_count"],
                "size_labels": [],
                "angle_sweep": source_summary["angle_sweep"],
                "path_specs": source_summary["path_specs"],
                "speed_cases": {},
                "memory_cases": {},
                "merged_from_labels": list(merge_labels),
            }
        else:
            _validate_mergeable_summary(
                source_label=source_label,
                source_summary=source_summary,
                reference_summary=reference_summary,
            )

        assert merged_summary is not None
        merged_summary["speed_cases"].update(source_summary.get("speed_cases", {}))
        merged_summary["memory_cases"].update(source_summary.get("memory_cases", {}))
        merged_size_labels.extend(source_summary.get("size_labels", []))

    assert merged_summary is not None
    merged_summary["size_labels"] = _ordered_unique_size_labels(merged_size_labels)
    _attach_reference_metrics(merged_summary)

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_table(merged_summary, run_dir / TABLE_NAME)
    _write_summary(run_dir, merged_summary)
    _plot_graphical_abstract(merged_summary, run_dir / FIGURE_NAME)
    print(f"Wrote merged graphical abstract to {run_dir / FIGURE_NAME}", flush=True)
    return 0


def run_study(args: argparse.Namespace) -> int:
    if not has_visible_gpu():
        raise SystemExit("No visible NVIDIA GPU found for the CoreShell backend performance study.")

    run_label = args.label or _timestamp()
    size_labels = _parse_csv_labels(args.size_labels)
    unknown = [label for label in size_labels if label not in SIZE_SPECS]
    if unknown:
        raise SystemExit(f"Unsupported size_labels entries: {unknown!r}")
    run_dir = OUT_ROOT / run_label
    speed_result_dir = run_dir / "speed_case_results"
    speed_awedge_dir = run_dir / "speed_awedges"
    memory_result_dir = run_dir / "memory_case_results"
    summary_path = run_dir / SUMMARY_NAME

    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {
            "label": run_label,
            "created_utc": _timestamp(),
            "python_executable": sys.executable,
            "gpu_index": int(args.gpu_index),
            "full_energy_count": len(_full_core_shell_energies()),
            "size_labels": list(size_labels),
            "angle_sweep": {
                angle.key: {
                    "label": angle.label,
                    "eangle_rotation": list(angle.eangle_rotation),
                    "num_angles": angle.num_angles,
                }
                for angle in ANGLE_SPECS
            },
            "path_specs": {path.key: asdict(path) for path in PATH_SPECS},
            "speed_cases": {},
            "memory_cases": {},
        }

    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        print("Running full-energy speed cases serially...", flush=True)
        for case in _build_speed_cases(size_labels):
            result = _run_case_subprocess(
                case=case,
                result_dir=speed_result_dir,
                awedge_dir=speed_awedge_dir,
                gpu_index=args.gpu_index,
                monitor_memory=False,
                poll_interval_s=args.memory_poll_interval_s,
                skip_existing=not args.no_skip_existing,
            )
            summary["speed_cases"][case.key] = result
            _write_summary(run_dir, summary)
            print(_result_summary_line(result), flush=True)

        _attach_reference_metrics(summary)
        _write_summary(run_dir, summary)

        if not args.no_memory_pass:
            print("Running separate 72-angle memory subset...", flush=True)
            for case in _build_memory_cases(size_labels):
                result = _run_case_subprocess(
                    case=case,
                    result_dir=memory_result_dir,
                    awedge_dir=None,
                    gpu_index=args.gpu_index,
                    monitor_memory=True,
                    poll_interval_s=args.memory_poll_interval_s,
                    skip_existing=not args.no_skip_existing,
                )
                summary["memory_cases"][case.key] = result
                _write_summary(run_dir, summary)
                print(_result_summary_line(result), flush=True)

        _write_table(summary, run_dir / TABLE_NAME)
        _write_summary(run_dir, summary)

    _plot_graphical_abstract(summary, run_dir / FIGURE_NAME)
    print(f"Wrote graphical abstract to {run_dir / FIGURE_NAME}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Legacy development-only full-energy CoreShell backend comparison runner "
            "and graphical-abstract generator."
        )
    )
    parser.add_argument("--label", default=None, help="Output subdirectory label under test-reports.")
    parser.add_argument("--gpu-index", type=int, default=0, help="Global GPU index to pin for serial runs.")
    parser.add_argument(
        "--size-labels",
        default="small,medium,large",
        help="Comma-separated subset of size labels to run, for example 'small,medium'.",
    )
    parser.add_argument(
        "--memory-poll-interval-s",
        type=float,
        default=0.2,
        help="External GPU-memory polling cadence for the separate memory pass.",
    )
    parser.add_argument(
        "--no-memory-pass",
        action="store_true",
        help="Skip the separate peak-memory subset and only run the speed pass.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Rerun cases even if completed result files already exist.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate the graphical abstract from an existing summary without rerunning cases.",
    )
    parser.add_argument(
        "--merge-labels",
        default=None,
        help=(
            "Comma-separated existing study labels to merge into one combined summary/table/figure. "
            "When provided, no new backend runs are launched."
        ),
    )
    parser.add_argument("--worker-case-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-awedge-path", default=None, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.worker_case_path:
        return _worker_main(
            case_path=Path(args.worker_case_path),
            result_path=Path(args.worker_result_path),
            awedge_path=Path(args.worker_awedge_path) if args.worker_awedge_path else None,
        )
    if args.merge_labels:
        return merge_studies(args)
    return run_study(args)


if __name__ == "__main__":
    raise SystemExit(main())
