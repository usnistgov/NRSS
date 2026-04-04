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
import traceback
from dataclasses import asdict
from dataclasses import dataclass
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

from tests.validation.lib.core_shell import (  # noqa: E402
    BASELINE_SCENARIO,
    SIM_REFERENCE_LABEL,
    SIM_REFERENCE_PATH,
    SIM_THRESHOLDS,
    awedge_comparison_slices,
    build_core_shell_morphology,
    compute_awedge_metrics,
    has_visible_gpu,
    load_sim_reference_awedge,
    metrics_within_thresholds,
    release_runtime_memory,
    scattering_to_awedge,
)


OUT_ROOT = REPO_ROOT / "test-reports" / "core-shell-backend-performance-dev"
SUMMARY_NAME = "mixed_precision_core_shell_summary.json"
TABLE_NAME = "mixed_precision_core_shell_table.tsv"
FIGURE_BASENAME = "core_shell_{execution_path}_mixed_precision_graphical_abstract.png"
AWEDGE_DIRNAME = "awedges"
RESULT_DIRNAME = "case_results"

MODE_COLORS = {
    "default": "#c45a00",
    "mixed_precision": "#007c91",
}
MODE_LINESTYLES = {
    "default": "--",
    "mixed_precision": "-.",
}
MODE_MARKERS = {
    "default": "s",
    "mixed_precision": "^",
}


@dataclass(frozen=True)
class ComparisonCase:
    key: str
    label: str
    execution_path: str
    mode_key: str
    mixed_precision_mode: str | None
    resident_mode: str
    field_namespace: str
    input_policy: str
    ownership_policy: str | None
    array_dtype: str


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


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


def _compare_to_reference(current: xr.DataArray, reference: xr.DataArray) -> dict[str, Any]:
    current = current.sortby("energy")
    reference = reference.sortby("energy")

    cur_energy = np.asarray(current.coords["energy"].values, dtype=np.float64)
    ref_energy = np.asarray(reference.coords["energy"].values, dtype=np.float64)
    cur_q = np.asarray(current.coords["q"].values, dtype=np.float64)
    ref_q = np.asarray(reference.coords["q"].values, dtype=np.float64)
    if not (np.allclose(cur_energy, ref_energy) and np.allclose(cur_q, ref_q)):
        raise AssertionError("A-wedge coordinates drifted between current run and sim golden.")

    current_vals = np.asarray(current.values, dtype=np.float64)
    reference_vals = np.asarray(reference.values, dtype=np.float64)
    finite = np.isfinite(current_vals) & np.isfinite(reference_vals)
    if not np.any(finite):
        raise AssertionError("No overlapping finite A-wedge values found for sim-golden comparison.")

    cur = current_vals[finite]
    ref = reference_vals[finite]
    rmse, rel_rmse = _relative_rmse(cur, ref)
    comparison = awedge_comparison_slices(awedge=current, reference=reference)
    slice_metrics = compute_awedge_metrics(comparison)
    thresholds_ok, threshold_failures = metrics_within_thresholds(slice_metrics, SIM_THRESHOLDS)

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
        "slice_metrics": slice_metrics,
        "thresholds_ok": bool(thresholds_ok),
        "threshold_failures": threshold_failures,
        "a_vs_energy_relative_rmse": series_relative("a_vs_energy", "reference_a_vs_energy"),
        "a_vs_q_284p7_relative_rmse": series_relative("a_vs_q_284p7", "reference_a_vs_q_284p7"),
        "a_vs_q_285p2_relative_rmse": series_relative("a_vs_q_285p2", "reference_a_vs_q_285p2"),
    }


def _case_specs() -> tuple[ComparisonCase, ...]:
    cases: list[ComparisonCase] = []
    for execution_path in ("tensor_coeff", "direct_polarization"):
        cases.append(
            ComparisonCase(
                key=f"{execution_path}__default",
                label=f"{execution_path} default",
                execution_path=execution_path,
                mode_key="default",
                mixed_precision_mode=None,
                resident_mode="device",
                field_namespace="cupy",
                input_policy="strict",
                ownership_policy="borrow",
                array_dtype="float32",
            )
        )
        cases.append(
            ComparisonCase(
                key=f"{execution_path}__mixed_precision",
                label=f"{execution_path} mixed precision",
                execution_path=execution_path,
                mode_key="mixed_precision",
                mixed_precision_mode="reduced_morphology_bit_depth",
                resident_mode="device",
                field_namespace="cupy",
                input_policy="strict",
                ownership_policy="borrow",
                array_dtype="float16",
            )
        )
    return tuple(cases)


def _worker_main(case_path: Path, result_path: Path, awedge_path: Path) -> int:
    started = time.perf_counter()
    payload = json.loads(case_path.read_text())
    case = ComparisonCase(**payload)
    result: dict[str, Any] = {
        "key": case.key,
        "label": case.label,
        "execution_path": case.execution_path,
        "mode_key": case.mode_key,
        "mixed_precision_mode": case.mixed_precision_mode,
        "resident_mode": case.resident_mode,
        "field_namespace": case.field_namespace,
        "input_policy": case.input_policy,
        "ownership_policy": case.ownership_policy,
        "array_dtype": case.array_dtype,
        "status": "error",
    }

    morph = None
    scattering = None
    awedge = None
    try:
        backend_options = {"execution_path": case.execution_path}
        if case.mixed_precision_mode is not None:
            backend_options["mixed_precision_mode"] = case.mixed_precision_mode

        build_start = time.perf_counter()
        morph = build_core_shell_morphology(
            scenario="baseline",
            create_cy_object=True,
            backend="cupy-rsoxs",
            backend_options=backend_options,
            resident_mode=case.resident_mode,
            input_policy=case.input_policy,
            ownership_policy=case.ownership_policy,
            field_namespace=case.field_namespace,
            array_dtype=np.dtype(case.array_dtype),
        )
        result["build_seconds"] = time.perf_counter() - build_start
        result["backend_dtype"] = morph.backend_dtype
        result["runtime_dtype"] = morph.runtime_dtype
        result["runtime_compute_dtype"] = morph.runtime_compute_dtype

        run_start = time.perf_counter()
        scattering = morph.run(stdout=False, stderr=False, return_xarray=True)
        result["run_seconds"] = time.perf_counter() - run_start
        result["backend_timings"] = dict(morph.backend_timings)

        awedge_start = time.perf_counter()
        awedge = scattering_to_awedge(scattering)
        result["awedge_seconds"] = time.perf_counter() - awedge_start
        _save_awedge_npz(awedge, awedge_path)
        result["awedge_path"] = str(awedge_path)
        result["panel_shape"] = list(scattering.shape)
        result["workflow_seconds"] = (
            result["build_seconds"] + result["run_seconds"] + result["awedge_seconds"]
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
        if morph is not None:
            try:
                morph.release_runtime()
            except Exception:
                pass
        del awedge, scattering, morph
        release_runtime_memory()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2, default=_json_default) + "\n")
    return 0


def _run_case_subprocess(
    *,
    case: ComparisonCase,
    result_dir: Path,
    awedge_dir: Path,
    gpu_index: int,
    skip_existing: bool,
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    awedge_dir.mkdir(parents=True, exist_ok=True)

    result_path = result_dir / f"{case.key}.json"
    awedge_path = awedge_dir / f"{case.key}.npz"

    if skip_existing and result_path.exists() and awedge_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("status") == "ok":
            existing["reused_existing"] = True
            return existing

    with tempfile.TemporaryDirectory(prefix="core_shell_mixed_precision_", dir=result_dir) as tmp_dir:
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
            "--worker-awedge-path",
            str(awedge_path),
        ]

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
        stdout, stderr = proc.communicate()
        elapsed = time.perf_counter() - started

        if worker_result_path.exists():
            result = json.loads(worker_result_path.read_text())
        else:
            result = {
                "key": case.key,
                "execution_path": case.execution_path,
                "mode_key": case.mode_key,
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

        result_path.write_text(json.dumps(result, indent=2, default=_json_default) + "\n")
        return result


def _attach_reference_metrics(summary: dict[str, Any]) -> None:
    reference_awedge = load_sim_reference_awedge()
    cases = summary.setdefault("cases", {})
    for key, result in cases.items():
        if result.get("status") != "ok":
            continue
        awedge = _load_awedge_npz(Path(result["awedge_path"]))
        result["comparison_to_sim_reference"] = _compare_to_reference(
            current=awedge,
            reference=reference_awedge,
        )

    for execution_path in ("tensor_coeff", "direct_polarization"):
        default_key = f"{execution_path}__default"
        mixed_key = f"{execution_path}__mixed_precision"
        default_result = cases.get(default_key)
        mixed_result = cases.get(mixed_key)
        if not default_result or not mixed_result:
            continue
        if default_result.get("status") != "ok" or mixed_result.get("status") != "ok":
            continue
        default_time = float(default_result["workflow_seconds"])
        mixed_time = float(mixed_result["workflow_seconds"])
        mixed_result["speedup_vs_default"] = float(default_time / mixed_time)
        mixed_result["time_delta_seconds_vs_default"] = float(mixed_time - default_time)


def _write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")


def _write_table(summary: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "execution_path",
                "mode_key",
                "mixed_precision_mode",
                "status",
                "array_dtype",
                "backend_dtype",
                "runtime_dtype",
                "runtime_compute_dtype",
                "build_seconds",
                "run_seconds",
                "awedge_seconds",
                "workflow_seconds",
                "speedup_vs_default",
                "a_vs_energy_rmse",
                "a_vs_energy_max_abs_diff",
                "a_vs_q_284p7_rmse",
                "a_vs_q_285p2_rmse",
                "full_relative_rmse_pct_vs_sim_reference",
                "thresholds_ok",
            ]
        )
        for key in sorted(summary.get("cases", {})):
            result = summary["cases"][key]
            comparison = result.get("comparison_to_sim_reference", {})
            slice_metrics = comparison.get("slice_metrics", {})
            writer.writerow(
                [
                    result.get("execution_path", ""),
                    result.get("mode_key", ""),
                    result.get("mixed_precision_mode", ""),
                    result.get("status", ""),
                    result.get("array_dtype", ""),
                    result.get("backend_dtype", ""),
                    result.get("runtime_dtype", ""),
                    result.get("runtime_compute_dtype", ""),
                    result.get("build_seconds", ""),
                    result.get("run_seconds", ""),
                    result.get("awedge_seconds", ""),
                    result.get("workflow_seconds", ""),
                    result.get("speedup_vs_default", ""),
                    slice_metrics.get("a_vs_energy", {}).get("rmse", ""),
                    slice_metrics.get("a_vs_energy", {}).get("max_abs_diff", ""),
                    slice_metrics.get("a_vs_q_284p7", {}).get("rmse", ""),
                    slice_metrics.get("a_vs_q_285p2", {}).get("rmse", ""),
                    (
                        100.0 * float(comparison["full_relative_rmse"])
                        if "full_relative_rmse" in comparison
                        else ""
                    ),
                    comparison.get("thresholds_ok", ""),
                ]
            )


def _plot_graphical_abstract(summary: dict[str, Any], execution_path: str, out_path: Path) -> None:
    cases = summary["cases"]
    default_result = cases[f"{execution_path}__default"]
    mixed_result = cases[f"{execution_path}__mixed_precision"]
    if default_result.get("status") != "ok" or mixed_result.get("status") != "ok":
        raise AssertionError(f"Cannot plot {execution_path}; required successful runs are missing.")

    reference = load_sim_reference_awedge()
    default_awedge = _load_awedge_npz(Path(default_result["awedge_path"]))
    mixed_awedge = _load_awedge_npz(Path(mixed_result["awedge_path"]))

    default_comp = awedge_comparison_slices(awedge=default_awedge, reference=reference)
    mixed_comp = awedge_comparison_slices(awedge=mixed_awedge, reference=reference)
    default_metrics = default_result["comparison_to_sim_reference"]["slice_metrics"]
    mixed_metrics = mixed_result["comparison_to_sim_reference"]["slice_metrics"]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(
        default_comp["reference_a_vs_energy"].coords["energy"].values,
        default_comp["reference_a_vs_energy"].values,
        color="#111111",
        linewidth=1.8,
        label=SIM_REFERENCE_LABEL,
    )
    for mode_key, comp in (("default", default_comp), ("mixed_precision", mixed_comp)):
        ax.plot(
            comp["a_vs_energy"].coords["energy"].values,
            comp["a_vs_energy"].values,
            color=MODE_COLORS[mode_key],
            linewidth=2.0,
            linestyle=MODE_LINESTYLES[mode_key],
            label=cases[f"{execution_path}__{mode_key}"]["label"],
        )
    ax.set_title("A(E) overlay")
    ax.set_xlabel("Energy [eV]")
    ax.set_ylabel("A(E)")
    ax.set_xticks([280, 282, 284, 286, 288, 290])
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
    ax.legend(loc="best", fontsize=8)

    for col, energy_key, energy_label in (
        (1, "a_vs_q_284p7", "284.7 eV"),
        (2, "a_vs_q_285p2", "285.2 eV"),
    ):
        ax = axes[0, col]
        ref_key = f"reference_{energy_key}"
        ax.plot(
            default_comp[ref_key].coords["q"].values,
            default_comp[ref_key].values,
            color="#111111",
            linewidth=1.8,
            label=f"{energy_label} sim golden",
        )
        for mode_key, comp in (("default", default_comp), ("mixed_precision", mixed_comp)):
            ax.plot(
                comp[energy_key].coords["q"].values,
                comp[energy_key].values,
                color=MODE_COLORS[mode_key],
                linewidth=2.0,
                linestyle=MODE_LINESTYLES[mode_key],
                label=f"{energy_label} {mode_key}",
            )
        ax.set_title(f"A(q) overlay | {energy_label}")
        ax.set_xlabel(r"q [nm$^{-1}$]")
        ax.set_ylabel("A(q)")
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
        ax.legend(loc="best", fontsize=7)

    ax = axes[1, 0]
    for mode_key, comp in (("default", default_comp), ("mixed_precision", mixed_comp)):
        residual = comp["a_vs_energy"] - comp["reference_a_vs_energy"]
        ax.plot(
            residual.coords["energy"].values,
            residual.values,
            color=MODE_COLORS[mode_key],
            linewidth=1.8,
            linestyle=MODE_LINESTYLES[mode_key],
            marker=MODE_MARKERS[mode_key],
            markersize=2.5,
            markevery=6,
            label=f"{mode_key} - golden",
        )
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.set_title("Delta A(E) vs sim golden")
    ax.set_xlabel("Energy [eV]")
    ax.set_ylabel(r"$\Delta A(E)$")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 1]
    for mode_key, comp in (("default", default_comp), ("mixed_precision", mixed_comp)):
        for energy_key, energy_label in (
            ("a_vs_q_284p7", "284.7 eV"),
            ("a_vs_q_285p2", "285.2 eV"),
        ):
            ref_key = f"reference_{energy_key}"
            residual = comp[energy_key] - comp[ref_key]
            ax.plot(
                residual.coords["q"].values,
                residual.values,
                color=MODE_COLORS[mode_key],
                linewidth=1.7,
                linestyle=MODE_LINESTYLES[mode_key],
                alpha=0.95 if energy_key.endswith("284p7") else 0.6,
                label=f"{mode_key} {energy_label}",
            )
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.set_title("Delta A(q) vs sim golden")
    ax.set_xlabel(r"q [nm$^{-1}$]")
    ax.set_ylabel(r"$\Delta A(q)$")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
    ax.legend(loc="best", fontsize=7)

    ax = axes[1, 2]
    ax.axis("off")
    mixed_speedup = mixed_result.get("speedup_vs_default")
    lines = [
        f"Execution path: {execution_path}",
        f"Scenario: {BASELINE_SCENARIO.scenario_id}",
        f"Reference: {SIM_REFERENCE_LABEL}",
        f"Golden path: {SIM_REFERENCE_PATH}",
        "",
        "Runtime:",
        (
            f"default workflow {float(default_result['workflow_seconds']):.3f}s "
            f"(build {float(default_result['build_seconds']):.3f}s, "
            f"run {float(default_result['run_seconds']):.3f}s, "
            f"awedge {float(default_result['awedge_seconds']):.3f}s)"
        ),
        (
            f"mixed workflow {float(mixed_result['workflow_seconds']):.3f}s "
            f"(build {float(mixed_result['build_seconds']):.3f}s, "
            f"run {float(mixed_result['run_seconds']):.3f}s, "
            f"awedge {float(mixed_result['awedge_seconds']):.3f}s)"
        ),
        (
            f"mixed speedup vs default: {float(mixed_speedup):.3f}x"
            if mixed_speedup is not None
            else "mixed speedup vs default: n/a"
        ),
        "",
        "Metrics vs sim golden:",
        (
            f"default A(E) rmse {default_metrics['a_vs_energy']['rmse']:.5f}, "
            f"max abs {default_metrics['a_vs_energy']['max_abs_diff']:.5f}"
        ),
        (
            f"default A(q) 284.7 rmse {default_metrics['a_vs_q_284p7']['rmse']:.5f}, "
            f"285.2 rmse {default_metrics['a_vs_q_285p2']['rmse']:.5f}"
        ),
        (
            f"mixed A(E) rmse {mixed_metrics['a_vs_energy']['rmse']:.5f}, "
            f"max abs {mixed_metrics['a_vs_energy']['max_abs_diff']:.5f}"
        ),
        (
            f"mixed A(q) 284.7 rmse {mixed_metrics['a_vs_q_284p7']['rmse']:.5f}, "
            f"285.2 rmse {mixed_metrics['a_vs_q_285p2']['rmse']:.5f}"
        ),
        "",
        (
            f"default thresholds ok: {default_result['comparison_to_sim_reference']['thresholds_ok']}"
        ),
        (
            f"mixed thresholds ok: {mixed_result['comparison_to_sim_reference']['thresholds_ok']}"
        ),
    ]
    failures = mixed_result["comparison_to_sim_reference"].get("threshold_failures", [])
    if failures:
        lines.extend(["", "Mixed threshold failures:"])
        lines.extend(f"- {item}" for item in failures)
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=8.6,
        family="monospace",
    )

    fig.suptitle(
        f"CoreShell sim-regression comparison: {execution_path} default vs mixed precision",
        fontsize=15,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        format="png",
        dpi=170,
        bbox_inches="tight",
        metadata={
            "Title": f"CoreShell mixed-precision graphical abstract: {execution_path}",
            "Description": (
                "CoreShell sim-regression graphical abstract comparing the vendored sim golden, "
                f"the default {execution_path} path, and the reduced-morphology-bit-depth path."
            ),
            "Author": "OpenAI Codex",
            "Software": "matplotlib",
        },
    )
    plt.close(fig)


def _regenerate_outputs(summary: dict[str, Any], run_dir: Path) -> None:
    _attach_reference_metrics(summary)
    _write_table(summary, run_dir / TABLE_NAME)
    _write_summary(run_dir, summary)
    for execution_path in ("tensor_coeff", "direct_polarization"):
        _plot_graphical_abstract(
            summary,
            execution_path,
            run_dir / FIGURE_BASENAME.format(execution_path=execution_path),
        )


def run_study(args: argparse.Namespace) -> int:
    if not has_visible_gpu():
        raise SystemExit("No visible NVIDIA GPU found for the CoreShell mixed-precision study.")
    if not SIM_REFERENCE_PATH.exists():
        raise SystemExit(f"CoreShell sim reference not found: {SIM_REFERENCE_PATH}")

    run_label = args.label or _timestamp()
    run_dir = OUT_ROOT / run_label
    summary_path = run_dir / SUMMARY_NAME

    if args.plot_only:
        if not summary_path.exists():
            raise SystemExit(f"Summary not found for --plot-only: {summary_path}")
        summary = json.loads(summary_path.read_text())
        _regenerate_outputs(summary, run_dir)
        print(f"Regenerated mixed-precision graphical abstracts in {run_dir}", flush=True)
        return 0

    summary = {
        "label": run_label,
        "created_utc": _timestamp(),
        "python_executable": sys.executable,
        "gpu_index": args.gpu_index,
        "scenario": BASELINE_SCENARIO.scenario_id,
        "reference_label": SIM_REFERENCE_LABEL,
        "reference_path": str(SIM_REFERENCE_PATH),
        "cases": {},
    }

    result_dir = run_dir / RESULT_DIRNAME
    awedge_dir = run_dir / AWEDGE_DIRNAME
    for case in _case_specs():
        print(f"Running {case.key}", flush=True)
        summary["cases"][case.key] = _run_case_subprocess(
            case=case,
            result_dir=result_dir,
            awedge_dir=awedge_dir,
            gpu_index=args.gpu_index,
            skip_existing=args.skip_existing,
        )

    _regenerate_outputs(summary, run_dir)
    print(f"Wrote summary to {run_dir / SUMMARY_NAME}", flush=True)
    for execution_path in ("tensor_coeff", "direct_polarization"):
        print(
            f"Wrote graphical abstract to "
            f"{run_dir / FIGURE_BASENAME.format(execution_path=execution_path)}",
            flush=True,
        )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate CoreShell sim-regression graphical abstracts comparing default and "
            "mixed-precision cupy-rsoxs paths."
        )
    )
    parser.add_argument("--label", default=None, help="Output run label under test-reports.")
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="Single GPU index to expose to subprocess workers.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse successful cached case results when present.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate summary-derived tables and figures without rerunning worker cases.",
    )
    parser.add_argument("--worker-case-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-awedge-path", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_case_path:
        if not args.worker_result_path or not args.worker_awedge_path:
            raise SystemExit("Worker mode requires result and awedge output paths.")
        return _worker_main(
            case_path=Path(args.worker_case_path),
            result_path=Path(args.worker_result_path),
            awedge_path=Path(args.worker_awedge_path),
        )
    return run_study(args)


if __name__ == "__main__":
    raise SystemExit(main())
