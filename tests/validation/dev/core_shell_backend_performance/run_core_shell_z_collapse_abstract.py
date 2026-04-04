#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    Q_COMPARE_MAX,
    Q_COMPARE_MIN,
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
SUMMARY_NAME = "z_collapse_core_shell_summary.json"
AWEDGE_NAME = "tensor_coeff_zcollapse_mean_awedge.npz"
FIGURE_NAME = "core_shell_tensor_coeff_zcollapse_mean_graphical_abstract.png"

RUN_COLOR = "#008a6a"
RUN_LINESTYLE = "-."
RUN_MARKER = "o"


@dataclass(frozen=True)
class ComparisonCase:
    key: str
    label: str
    execution_path: str
    z_collapse_mode: str
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


def _case_spec() -> ComparisonCase:
    return ComparisonCase(
        key="tensor_coeff__zcollapse_mean",
        label='tensor_coeff z-collapse="mean"',
        execution_path="tensor_coeff",
        z_collapse_mode="mean",
        resident_mode="device",
        field_namespace="cupy",
        input_policy="strict",
        ownership_policy="borrow",
        array_dtype="float32",
    )


def _worker_main(case_path: Path, result_path: Path, awedge_path: Path) -> int:
    started = time.perf_counter()
    payload = json.loads(case_path.read_text())
    case = ComparisonCase(**payload)
    result: dict[str, Any] = {
        "key": case.key,
        "label": case.label,
        "execution_path": case.execution_path,
        "z_collapse_mode": case.z_collapse_mode,
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
        backend_options = {
            "execution_path": case.execution_path,
            "z_collapse_mode": case.z_collapse_mode,
        }

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
    awedge_path = awedge_dir / AWEDGE_NAME

    if skip_existing and result_path.exists() and awedge_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("status") == "ok":
            existing["reused_existing"] = True
            return existing

    with tempfile.TemporaryDirectory(prefix="core_shell_z_collapse_", dir=result_dir) as tmp_dir:
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
                "z_collapse_mode": case.z_collapse_mode,
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
    result = summary["case"]
    if result.get("status") != "ok":
        return
    reference_awedge = load_sim_reference_awedge()
    awedge = _load_awedge_npz(Path(result["awedge_path"]))
    result["comparison_to_sim_reference"] = _compare_to_reference(
        current=awedge,
        reference=reference_awedge,
    )


def _plot_graphical_abstract(summary: dict[str, Any], out_path: Path) -> None:
    result = summary["case"]
    if result.get("status") != "ok":
        raise AssertionError("Cannot plot CoreShell z-collapse abstract without a successful run.")

    reference = load_sim_reference_awedge()
    current_awedge = _load_awedge_npz(Path(result["awedge_path"]))
    comp = awedge_comparison_slices(awedge=current_awedge, reference=reference)
    metrics = result["comparison_to_sim_reference"]["slice_metrics"]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(
        comp["reference_a_vs_energy"].coords["energy"].values,
        comp["reference_a_vs_energy"].values,
        color="#111111",
        linewidth=1.8,
        label=SIM_REFERENCE_LABEL,
    )
    ax.plot(
        comp["a_vs_energy"].coords["energy"].values,
        comp["a_vs_energy"].values,
        color=RUN_COLOR,
        linewidth=2.0,
        linestyle=RUN_LINESTYLE,
        label=result["label"],
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
            comp[ref_key].coords["q"].values,
            comp[ref_key].values,
            color="#111111",
            linewidth=1.8,
            label=f"{energy_label} sim golden",
        )
        ax.plot(
            comp[energy_key].coords["q"].values,
            comp[energy_key].values,
            color=RUN_COLOR,
            linewidth=2.0,
            linestyle=RUN_LINESTYLE,
            label=f"{energy_label} collapse",
        )
        ax.set_title(f"A(q) overlay | {energy_label}")
        ax.set_xlabel(r"q [nm$^{-1}$]")
        ax.set_ylabel("A(q)")
        ax.set_xlim(Q_COMPARE_MIN, Q_COMPARE_MAX)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
        ax.legend(loc="best", fontsize=7)

    ax = axes[1, 0]
    residual = comp["a_vs_energy"] - comp["reference_a_vs_energy"]
    ax.plot(
        residual.coords["energy"].values,
        residual.values,
        color=RUN_COLOR,
        linewidth=1.8,
        linestyle=RUN_LINESTYLE,
        marker=RUN_MARKER,
        markersize=2.5,
        markevery=6,
        label="collapse - golden",
    )
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.set_title("Delta A(E) vs sim golden")
    ax.set_xlabel("Energy [eV]")
    ax.set_ylabel(r"$\Delta A(E)$")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 1]
    for energy_key, energy_label, alpha in (
        ("a_vs_q_284p7", "284.7 eV", 0.95),
        ("a_vs_q_285p2", "285.2 eV", 0.60),
    ):
        ref_key = f"reference_{energy_key}"
        residual = comp[energy_key] - comp[ref_key]
        ax.plot(
            residual.coords["q"].values,
            residual.values,
            color=RUN_COLOR,
            linewidth=1.7,
            linestyle=RUN_LINESTYLE,
            alpha=alpha,
            label=f"collapse {energy_label}",
        )
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.set_title("Delta A(q) vs sim golden")
    ax.set_xlabel(r"q [nm$^{-1}$]")
    ax.set_ylabel(r"$\Delta A(q)$")
    ax.set_xlim(Q_COMPARE_MIN, Q_COMPARE_MAX)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
    ax.legend(loc="best", fontsize=7)

    comparison = result["comparison_to_sim_reference"]
    threshold_failures = comparison.get("threshold_failures", [])
    ax = axes[1, 2]
    ax.axis("off")
    lines = [
        f'Execution path: {result["execution_path"]}',
        f'z_collapse_mode: {result["z_collapse_mode"]}',
        f"Scenario: {BASELINE_SCENARIO.scenario_id}",
        f"Reference: {SIM_REFERENCE_LABEL}",
        f"Golden path: {SIM_REFERENCE_PATH}",
        "",
        "Runtime:",
        (
            f'workflow {float(result["workflow_seconds"]):.3f}s '
            f'(build {float(result["build_seconds"]):.3f}s, '
            f'run {float(result["run_seconds"]):.3f}s, '
            f'awedge {float(result["awedge_seconds"]):.3f}s)'
        ),
        "",
        "Metrics vs sim golden:",
        (
            f'A(E) rmse {metrics["a_vs_energy"]["rmse"]:.5f}, '
            f'max abs {metrics["a_vs_energy"]["max_abs_diff"]:.5f}'
        ),
        (
            f'A(q) 284.7 rmse {metrics["a_vs_q_284p7"]["rmse"]:.5f}, '
            f'max abs {metrics["a_vs_q_284p7"]["max_abs_diff"]:.5f}'
        ),
        (
            f'A(q) 285.2 rmse {metrics["a_vs_q_285p2"]["rmse"]:.5f}, '
            f'max abs {metrics["a_vs_q_285p2"]["max_abs_diff"]:.5f}'
        ),
        (
            f'full relative rmse {100.0 * float(comparison["full_relative_rmse"]):.3f}%'
        ),
        "",
        f'thresholds ok: {comparison["thresholds_ok"]}',
    ]
    if threshold_failures:
        lines.extend(["", "Threshold failures:"])
        lines.extend(f"- {item}" for item in threshold_failures)
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
        'CoreShell sim-regression comparison: tensor_coeff z-collapse="mean"',
        fontsize=15,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        format="png",
        dpi=170,
        bbox_inches="tight",
        metadata={
            "Title": "CoreShell z-collapse graphical abstract: tensor_coeff mean",
            "Description": (
                "CoreShell sim-regression graphical abstract comparing the vendored sim golden "
                'and the tensor_coeff z_collapse_mode="mean" run.'
            ),
            "Author": "OpenAI Codex",
            "Software": "matplotlib",
        },
    )
    plt.close(fig)


def _write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")


def _regenerate_outputs(summary: dict[str, Any], run_dir: Path) -> None:
    _attach_reference_metrics(summary)
    _write_summary(run_dir, summary)
    _plot_graphical_abstract(summary, run_dir / FIGURE_NAME)


def run_study(args: argparse.Namespace) -> int:
    if not has_visible_gpu():
        raise SystemExit("No visible NVIDIA GPU found for the CoreShell z-collapse study.")
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
        print(f"Regenerated CoreShell z-collapse graphical abstract in {run_dir}", flush=True)
        return 0

    case = _case_spec()
    summary = {
        "label": run_label,
        "created_utc": _timestamp(),
        "python_executable": sys.executable,
        "gpu_index": args.gpu_index,
        "scenario": BASELINE_SCENARIO.scenario_id,
        "reference_label": SIM_REFERENCE_LABEL,
        "reference_path": str(SIM_REFERENCE_PATH),
        "case": {},
    }

    result_dir = run_dir / "case_results"
    awedge_dir = run_dir / "awedges"
    print(f"Running {case.key}", flush=True)
    summary["case"] = _run_case_subprocess(
        case=case,
        result_dir=result_dir,
        awedge_dir=awedge_dir,
        gpu_index=args.gpu_index,
        skip_existing=args.skip_existing,
    )

    _regenerate_outputs(summary, run_dir)
    print(f"Wrote summary to {run_dir / SUMMARY_NAME}", flush=True)
    print(f"Wrote graphical abstract to {run_dir / FIGURE_NAME}", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a CoreShell sim-regression graphical abstract for the "
            'tensor_coeff z_collapse_mode="mean" path.'
        )
    )
    parser.add_argument("--label", default=None, help="Output run label under test-reports.")
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="Single GPU index to expose to the subprocess worker.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse a successful cached case result when present.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate the figure from a saved summary without rerunning the simulation.",
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
