#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

from NRSS.backends import cupy_rsoxs as cupy_rsoxs_module  # noqa: E402
from tests.validation.dev.core_shell_backend_performance.run_comprehensive_backend_comparison import (  # noqa: E402
    ComparisonCase,
    _cupy_case,
    _run_case_subprocess,
)
from tests.validation.dev.cupy_rsoxs_optimization.run_cupy_rsoxs_optimization_matrix import (  # noqa: E402
    BenchmarkCase,
    _construct_morphology_for_timing_case,
    _json_default,
    _prepare_core_shell_case_inputs,
    _synchronize_cupy_default_stream,
    _timestamp,
    _worker_main,
)
from tests.validation.lib.core_shell import has_visible_gpu  # noqa: E402
from tests.validation.lib.core_shell import release_runtime_memory  # noqa: E402


OUT_ROOT = REPO_ROOT / "test-reports" / "core-shell-backend-performance-dev"
SUMMARY_NAME = "tensor_coeff_inplace_segment_c_recheck_summary.json"
ROTATIONS = {
    "no_rotation": {
        "rotation": (0.0, 0.0, 0.0),
        "label": "no rotation",
    },
    "rot_0_5_165": {
        "rotation": (0.0, 5.0, 165.0),
        "label": "0:5:165",
    },
}
CASE_MATRIX = {
    "small__no_rotation": {
        "size_label": "small",
        "rotation_key": "no_rotation",
    },
    "medium__no_rotation": {
        "size_label": "medium",
        "rotation_key": "no_rotation",
    },
    "medium__rot_0_5_165": {
        "size_label": "medium",
        "rotation_key": "rot_0_5_165",
    },
}
VARIANTS = {
    "baseline": {
        "label": "baseline",
        "description": "Current maintained tensor_coeff Segment C.",
    },
    "item1_inplace_segment_c": {
        "label": "item1",
        "description": (
            "Cached cuFFT C2C plan, in-place FFT on nt[idx], and in-place Igor shift."
        ),
    },
}
DEFAULT_SPEED_CASES = ("small__no_rotation", "medium__no_rotation", "medium__rot_0_5_165")
DEFAULT_MEMORY_CASES = ("small__no_rotation", "medium__no_rotation", "medium__rot_0_5_165")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _series_stats(values: list[float]) -> dict[str, float]:
    ordered = [float(value) for value in values]
    if not ordered:
        raise ValueError("Expected at least one value.")
    return {
        "count": float(len(ordered)),
        "min": float(min(ordered)),
        "median": float(statistics.median(ordered)),
        "mean": float(statistics.fmean(ordered)),
        "max": float(max(ordered)),
    }


def _segment_series_stats(runs: list[dict[str, Any]], segment: str) -> dict[str, float] | None:
    values = [
        float(result["segment_seconds"][segment])
        for result in runs
        if segment in result.get("segment_seconds", {})
    ]
    if not values:
        return None
    return _series_stats(values)


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in raw.split(",") if token.strip())


def _resolve_case_keys(raw: str, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw == "default":
        return default
    if raw == "all":
        return tuple(CASE_MATRIX)
    keys = _parse_csv(raw)
    unknown = [key for key in keys if key not in CASE_MATRIX]
    if unknown:
        raise SystemExit(f"Unsupported case keys: {unknown!r}")
    return keys


def _resolve_variants(raw: str) -> tuple[str, ...]:
    variants = _parse_csv(raw)
    unknown = [variant for variant in variants if variant not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unsupported variants: {unknown!r}")
    return variants


def _tensor_coeff_fft_plan(self, morphology, cp, arr):
    from cupyx.scipy.fftpack import get_fft_plan

    cache_key = (
        "tensor_coeff_fft_plan_complex64",
        tuple(int(v) for v in arr.shape),
        cp.dtype(arr.dtype).name,
    )
    plan = morphology._backend_runtime_state.get(cache_key)
    if plan is None:
        plan = get_fft_plan(arr, axes=(0, 1, 2), value_type="C2C")
        morphology._backend_runtime_state[cache_key] = plan
    return plan


def _patch_item1_inplace_segment_c() -> None:
    runtime_cls = cupy_rsoxs_module.CupyRsoxsBackendRuntime

    def patched_compute_fft_nt_components(
        self,
        nt,
        morphology,
        cp,
        window,
        component_indices=None,
    ):
        component_indices = (
            tuple(range(nt.shape[0])) if component_indices is None else tuple(component_indices)
        )
        if not component_indices:
            return nt
        plan = self._tensor_coeff_fft_plan(morphology, cp, nt[component_indices[0]])
        cufft = cp.cuda.cufft
        for idx in component_indices:
            component = nt[idx]
            if window is not None:
                cp.multiply(component, window, out=component)
            plan.fft(component, component, cufft.CUFFT_FORWARD)
            self._replace_dc_component(component)
            self._igor_shift_inplace(component, morphology, cp)
        return nt

    runtime_cls._tensor_coeff_fft_plan = _tensor_coeff_fft_plan
    runtime_cls._compute_fft_nt_components = patched_compute_fft_nt_components


def _apply_variant_patch(variant: str) -> None:
    if variant == "baseline":
        return
    if variant == "item1_inplace_segment_c":
        _patch_item1_inplace_segment_c()
        return
    raise ValueError(f"Unsupported variant {variant!r}.")


def _variant_from_worker_case(case_path: Path, explicit_variant: str) -> str:
    if explicit_variant and explicit_variant != "baseline":
        return explicit_variant
    case_payload = json.loads(case_path.read_text())
    label = str(case_payload.get("label", ""))
    prefix = label.split("__", 1)[0]
    if prefix == "baseline":
        return "baseline"
    if prefix == "item1":
        return "item1_inplace_segment_c"
    raise ValueError(f"Could not infer worker variant from label {label!r}.")


def _worker_entry(case_path: Path, result_path: Path, variant: str) -> int:
    variant = _variant_from_worker_case(case_path, variant)
    _apply_variant_patch(variant)
    return _worker_main(case_path, result_path)


def _run_panel_for_case(case: BenchmarkCase) -> np.ndarray:
    prepared_inputs = None
    morphology = None
    backend_result = None
    try:
        prepared_inputs = _prepare_core_shell_case_inputs(case)
        if case.field_namespace == "cupy":
            _synchronize_cupy_default_stream()
        morphology = _construct_morphology_for_timing_case(case, prepared_inputs)
        backend_result = morphology.run(stdout=False, stderr=False, return_xarray=False)
        _synchronize_cupy_default_stream()
        panel = backend_result.to_backend_array()
        if hasattr(panel, "get"):
            return np.asarray(panel.get())
        return np.asarray(panel)
    finally:
        if morphology is not None:
            try:
                morphology._clear_private_backend_timing_segments()
            except Exception:
                pass
            try:
                morphology.release_runtime()
            except Exception:
                pass
        del backend_result, morphology, prepared_inputs
        release_runtime_memory()


def _parity_worker(case_path: Path, result_path: Path) -> int:
    case = BenchmarkCase(**json.loads(case_path.read_text()))
    result: dict[str, Any] = {
        "label": case.label,
        "status": "error",
    }
    try:
        baseline_panel = _run_panel_for_case(case)
        _apply_variant_patch("item1_inplace_segment_c")
        candidate_panel = _run_panel_for_case(case)
        baseline_finite = np.isfinite(baseline_panel)
        candidate_finite = np.isfinite(candidate_panel)
        finite_overlap = baseline_finite & candidate_finite
        nan_mask_match = bool(np.array_equal(baseline_finite, candidate_finite))
        if np.any(finite_overlap):
            diff = candidate_panel[finite_overlap] - baseline_panel[finite_overlap]
            abs_diff = np.abs(diff)
            max_abs = float(np.max(abs_diff))
            rmse = float(np.sqrt(np.mean(abs_diff * abs_diff)))
            p95_abs = float(np.percentile(abs_diff, 95.0))
        else:
            max_abs = 0.0
            rmse = 0.0
            p95_abs = 0.0
        result.update(
            status="ok",
            panel_shape=list(baseline_panel.shape),
            nan_mask_match=nan_mask_match,
            finite_voxel_count=int(np.count_nonzero(finite_overlap)),
            max_abs=max_abs,
            rmse=rmse,
            p95_abs=p95_abs,
        )
    except BaseException as exc:  # noqa: BLE001
        result.update(
            status="error",
            error_type=exc.__class__.__name__,
            error=str(exc),
        )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, default=_json_default) + "\n")
    return 0


def _build_variant_case(
    *,
    variant: str,
    repeat_index: int,
    case_key: str,
) -> ComparisonCase:
    case_spec = CASE_MATRIX[case_key]
    rotation = ROTATIONS[case_spec["rotation_key"]]
    base_case = _cupy_case(
        residency="host",
        startup_mode="hot",
        execution_path="tensor_coeff",
        z_collapse_mode=None,
        rotation_key=case_spec["rotation_key"],
        rotation_label=rotation["label"],
        eangle_rotation=rotation["rotation"],
        size_label=case_spec["size_label"],
    )
    variant_meta = VARIANTS[variant]
    variant_key = f"{variant_meta['label']}__{case_key}__run{repeat_index + 1:02d}"
    return replace(
        base_case,
        key=variant_key,
        script_path=Path(__file__).resolve(),
        worker_case=replace(
            base_case.worker_case,
            label=variant_key,
            notes=(
                f"Tensor-coeff Segment C recheck variant={variant}. "
                f"Case={case_key}. Repeat {repeat_index + 1}. "
                f"{variant_meta['description']}"
            ),
        ),
    )


def _run_parity_check(*, run_dir: Path, gpu_index: int) -> dict[str, Any]:
    import subprocess
    import tempfile

    parity_case = _cupy_case(
        residency="host",
        startup_mode="hot",
        execution_path="tensor_coeff",
        z_collapse_mode=None,
        rotation_key="no_rotation",
        rotation_label="no rotation",
        eangle_rotation=ROTATIONS["no_rotation"]["rotation"],
        size_label="small",
    ).worker_case
    parity_case = replace(
        parity_case,
        label="parity__small__no_rotation",
        worker_warmup_runs=0,
        notes="Parity check for tensor_coeff in-place Segment C versus maintained baseline.",
    )

    parity_dir = run_dir / "parity_case_results"
    parity_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tensor_coeff_parity_", dir=parity_dir) as tmp_dir:
        tmp_path = Path(tmp_dir)
        case_path = tmp_path / "case.json"
        result_path = tmp_path / "result.json"
        case_path.write_text(json.dumps(asdict(parity_case), indent=2, default=_json_default) + "\n")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--parity-worker-case-path",
            str(case_path),
            "--parity-worker-result-path",
            str(result_path),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            env=env,
        )
        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            result = {
                "label": parity_case.label,
                "status": "subprocess_failed",
                "error_type": "SubprocessFailure",
                "error": "Parity worker exited before writing a result file.",
            }
        result["subprocess_returncode"] = int(completed.returncode)
        if completed.stdout.strip():
            result["worker_stdout"] = completed.stdout[-4000:]
        if completed.stderr.strip():
            result["worker_stderr"] = completed.stderr[-4000:]
        return result


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "runs": runs,
        "primary_seconds_stats": _series_stats([float(result["primary_seconds"]) for result in runs]),
    }
    for segment in ("B", "C", "D", "E"):
        segment_stats = _segment_series_stats(runs, segment)
        if segment_stats is not None:
            summary.setdefault("segment_seconds_stats", {})[segment] = segment_stats
    return summary


def _summarize_memory_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runs": runs,
        "peak_gpu_delta_mib_stats": _series_stats(
            [float(result["memory_probe"]["peak_gpu_delta_mib"]) for result in runs]
        ),
        "sample_count_stats": _series_stats(
            [float(result["memory_probe"]["sample_count"]) for result in runs]
        ),
    }


def _run_recheck(args: argparse.Namespace) -> int:
    if not has_visible_gpu():
        raise SystemExit("No visible NVIDIA GPU found for the tensor-coeff Segment C recheck.")

    variants = _resolve_variants(args.variants)
    speed_case_keys = _resolve_case_keys(args.speed_cases, default=DEFAULT_SPEED_CASES)
    memory_case_keys = _resolve_case_keys(args.memory_cases, default=DEFAULT_MEMORY_CASES)

    run_label = args.label or f"tc_mem08_inplace_segment_c_{_timestamp()}"
    run_dir = OUT_ROOT / run_label
    speed_dir = run_dir / "speed_case_results"
    memory_dir = run_dir / "memory_case_results"
    summary_path = run_dir / SUMMARY_NAME

    summary: dict[str, Any] = {
        "label": run_label,
        "created_utc": _timestamp(),
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_index": int(args.gpu_index),
        "repeats": int(args.repeats),
        "memory_poll_interval_s": float(args.memory_poll_interval_s),
        "variants": list(variants),
        "speed_case_keys": list(speed_case_keys),
        "memory_case_keys": list(memory_case_keys),
        "cases": {
            case_key: {
                "size_label": CASE_MATRIX[case_key]["size_label"],
                "rotation_key": CASE_MATRIX[case_key]["rotation_key"],
                "rotation_label": ROTATIONS[CASE_MATRIX[case_key]["rotation_key"]]["label"],
                "eangle_rotation": list(
                    ROTATIONS[CASE_MATRIX[case_key]["rotation_key"]]["rotation"]
                ),
            }
            for case_key in CASE_MATRIX
        },
        "speed_cases": {},
        "memory_cases": {},
        "parity": {},
        "decision": {},
    }

    for variant in variants:
        summary["speed_cases"][variant] = {}
        for case_key in speed_case_keys:
            run_results = []
            for repeat_index in range(args.repeats):
                case = _build_variant_case(
                    variant=variant,
                    repeat_index=repeat_index,
                    case_key=case_key,
                )
                result = _run_case_subprocess(
                    case=case,
                    output_dir=speed_dir,
                    gpu_index=args.gpu_index,
                    monitor_memory=False,
                    poll_interval_s=args.memory_poll_interval_s,
                    skip_existing=False,
                )
                run_results.append(result)
                print(
                    f"[speed] variant={variant} case={case_key} run={repeat_index + 1}/{args.repeats} "
                    f"primary={float(result['primary_seconds']):.6f}s",
                    flush=True,
                )
            summary["speed_cases"][variant][case_key] = _summarize_runs(run_results)
            _write_json(summary_path, summary)

        summary["memory_cases"][variant] = {}
        for case_key in memory_case_keys:
            run_results = []
            for repeat_index in range(args.repeats):
                case = _build_variant_case(
                    variant=variant,
                    repeat_index=repeat_index,
                    case_key=case_key,
                )
                result = _run_case_subprocess(
                    case=case,
                    output_dir=memory_dir,
                    gpu_index=args.gpu_index,
                    monitor_memory=True,
                    poll_interval_s=args.memory_poll_interval_s,
                    skip_existing=False,
                )
                run_results.append(result)
                print(
                    f"[memory] variant={variant} case={case_key} run={repeat_index + 1}/{args.repeats} "
                    f"peak_delta={float(result['memory_probe']['peak_gpu_delta_mib']):.3f}MiB "
                    f"samples={int(result['memory_probe']['sample_count'])}",
                    flush=True,
                )
            summary["memory_cases"][variant][case_key] = _summarize_memory_runs(run_results)
            _write_json(summary_path, summary)

    summary["parity"] = _run_parity_check(run_dir=run_dir, gpu_index=args.gpu_index)
    _write_json(summary_path, summary)

    baseline_variant = "baseline"
    candidate_variant = "item1_inplace_segment_c"
    if baseline_variant in variants and candidate_variant in variants:
        for case_key in memory_case_keys:
            baseline_peak = summary["memory_cases"][baseline_variant][case_key][
                "peak_gpu_delta_mib_stats"
            ]["median"]
            candidate_peak = summary["memory_cases"][candidate_variant][case_key][
                "peak_gpu_delta_mib_stats"
            ]["median"]
            memory_ratio = float(candidate_peak / baseline_peak) if baseline_peak else float("inf")
            speed_ratio = None
            segment_c_ratio = None
            if case_key in speed_case_keys:
                baseline_primary = summary["speed_cases"][baseline_variant][case_key][
                    "primary_seconds_stats"
                ]["median"]
                candidate_primary = summary["speed_cases"][candidate_variant][case_key][
                    "primary_seconds_stats"
                ]["median"]
                speed_ratio = float(candidate_primary / baseline_primary)
                baseline_seg_c = summary["speed_cases"][baseline_variant][case_key][
                    "segment_seconds_stats"
                ]["C"]["median"]
                candidate_seg_c = summary["speed_cases"][candidate_variant][case_key][
                    "segment_seconds_stats"
                ]["C"]["median"]
                segment_c_ratio = float(candidate_seg_c / baseline_seg_c)

            summary["decision"][case_key] = {
                "status": (
                    "pass"
                    if (speed_ratio is None or speed_ratio < 1.05) and memory_ratio < 1.0
                    else "fail"
                ),
                "memory_peak_ratio": float(memory_ratio),
                "speed_ratio": None if speed_ratio is None else float(speed_ratio),
                "segment_c_ratio": None if segment_c_ratio is None else float(segment_c_ratio),
                "baseline_peak_gpu_delta_mib": float(baseline_peak),
                "candidate_peak_gpu_delta_mib": float(candidate_peak),
            }

    _write_json(summary_path, summary)
    print(f"Wrote {summary_path}", flush=True)
    if summary["parity"].get("status") == "ok":
        print(
            "[parity] "
            f"max_abs={summary['parity']['max_abs']:.6e} "
            f"rmse={summary['parity']['rmse']:.6e} "
            f"p95_abs={summary['parity']['p95_abs']:.6e}",
            flush=True,
        )
    for case_key, decision in summary["decision"].items():
        speed_ratio = decision["speed_ratio"]
        seg_c_ratio = decision["segment_c_ratio"]
        line = (
            f"[decision] case={case_key} status={decision['status']} "
            f"memory_ratio={decision['memory_peak_ratio']:.4f}"
        )
        if speed_ratio is not None:
            line += f" speed_ratio={speed_ratio:.4f}"
        if seg_c_ratio is not None:
            line += f" segment_c_ratio={seg_c_ratio:.4f}"
        print(line, flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only hot-lane recheck for the tensor_coeff true in-place "
            "Segment C experiment."
        )
    )
    parser.add_argument("--label", default=None, help="Output subdirectory label under test-reports.")
    parser.add_argument("--gpu-index", type=int, default=0, help="Global GPU index to pin.")
    parser.add_argument("--repeats", type=int, default=3, help="Repeated runs per variant/case.")
    parser.add_argument(
        "--memory-poll-interval-s",
        type=float,
        default=0.001,
        help="Fast CuPy observer sampling cadence for the memory pass.",
    )
    parser.add_argument(
        "--variants",
        default="baseline,item1_inplace_segment_c",
        help="Comma-separated variant list.",
    )
    parser.add_argument(
        "--speed-cases",
        default="default",
        help="Comma-separated case keys, 'default', or 'all'.",
    )
    parser.add_argument(
        "--memory-cases",
        default="default",
        help="Comma-separated case keys, 'default', or 'all'.",
    )
    parser.add_argument("--worker-case-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-variant", default="baseline", help=argparse.SUPPRESS)
    parser.add_argument("--parity-worker-case-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--parity-worker-result-path", default=None, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.worker_case_path and args.worker_result_path:
        return _worker_entry(
            case_path=Path(args.worker_case_path),
            result_path=Path(args.worker_result_path),
            variant=str(args.worker_variant),
        )
    if args.parity_worker_case_path and args.parity_worker_result_path:
        return _parity_worker(
            case_path=Path(args.parity_worker_case_path),
            result_path=Path(args.parity_worker_result_path),
        )
    return _run_recheck(args)


if __name__ == "__main__":
    raise SystemExit(main())
