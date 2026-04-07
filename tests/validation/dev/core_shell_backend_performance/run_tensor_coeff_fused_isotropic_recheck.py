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
SUMMARY_NAME = "tensor_coeff_fused_isotropic_recheck_summary.json"
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
        "description": "Current maintained tensor_coeff float32 Segment B path.",
    },
    "item1_fused_isotropic": {
        "label": "item1",
        "description": (
            "Eliminate the float32 isotropic_term temporary by using direct "
            "float32 isotropic and anisotropic accumulation kernels."
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


def _patch_item1_fused_isotropic() -> None:
    runtime_cls = cupy_rsoxs_module.CupyRsoxsBackendRuntime
    original_compute_nt_components = runtime_cls._compute_nt_components

    def _nt_accumulate_isotropic_float32_kernel(self, cp):
        kernel = cupy_rsoxs_module._CUPY_KERNEL_CACHE.get("nt_accumulate_isotropic_float32_candidate")
        if kernel is not None:
            return kernel

        kernel = cp.RawKernel(
            r"""
            extern "C" __global__
            void nt_accumulate_isotropic_float32_candidate(
                const float* vfrac,
                const float2 isotropic_diag,
                const int need0,
                const int need3,
                float2* nt0,
                float2* nt3,
                const unsigned long long total
            ) {
                const unsigned long long idx =
                    (unsigned long long)blockDim.x * (unsigned long long)blockIdx.x
                    + (unsigned long long)threadIdx.x;
                if (idx >= total) {
                    return;
                }

                const float vf = vfrac[idx];
                if (need0) {
                    nt0[idx].x += vf * isotropic_diag.x;
                    nt0[idx].y += vf * isotropic_diag.y;
                }
                if (need3) {
                    nt3[idx].x += vf * isotropic_diag.x;
                    nt3[idx].y += vf * isotropic_diag.y;
                }
            }
            """,
            "nt_accumulate_isotropic_float32_candidate",
        )
        cupy_rsoxs_module._CUPY_KERNEL_CACHE["nt_accumulate_isotropic_float32_candidate"] = kernel
        return kernel

    def _nt_accumulate_anisotropic_float32_kernel(self, cp):
        kernel = cupy_rsoxs_module._CUPY_KERNEL_CACHE.get("nt_accumulate_anisotropic_float32_candidate")
        if kernel is not None:
            return kernel

        kernel = cp.RawKernel(
            r"""
            extern "C" __global__
            void nt_accumulate_anisotropic_float32_candidate(
                const float* vfrac,
                const float* s,
                const float* theta,
                const float* psi,
                const float2 isotropic_diag,
                const float2 aligned_base,
                const float2 anisotropic_delta,
                const int need0,
                const int need1,
                const int need2,
                const int need3,
                const int need4,
                float2* nt0,
                float2* nt1,
                float2* nt2,
                float2* nt3,
                float2* nt4,
                const unsigned long long total
            ) {
                const unsigned long long idx =
                    (unsigned long long)blockDim.x * (unsigned long long)blockIdx.x
                    + (unsigned long long)threadIdx.x;
                if (idx >= total) {
                    return;
                }

                const float vf = vfrac[idx];
                if (need0 || need3) {
                    const float iso_x = vf * isotropic_diag.x;
                    const float iso_y = vf * isotropic_diag.y;
                    if (need0) {
                        nt0[idx].x += iso_x;
                        nt0[idx].y += iso_y;
                    }
                    if (need3) {
                        nt3[idx].x += iso_x;
                        nt3[idx].y += iso_y;
                    }
                }

                const float phi = vf * s[idx];
                const float theta_i = theta[idx];
                const float psi_i = psi[idx];
                const float sin_theta = sinf(theta_i);
                const float sx = cosf(psi_i) * sin_theta;
                const float sy = sinf(psi_i) * sin_theta;
                const float sz = cosf(theta_i);

                if (need0) {
                    nt0[idx].x += phi * (aligned_base.x + anisotropic_delta.x * sx * sx);
                    nt0[idx].y += phi * (aligned_base.y + anisotropic_delta.y * sx * sx);
                }
                if (need1) {
                    nt1[idx].x += phi * anisotropic_delta.x * sx * sy;
                    nt1[idx].y += phi * anisotropic_delta.y * sx * sy;
                }
                if (need2) {
                    nt2[idx].x += phi * anisotropic_delta.x * sx * sz;
                    nt2[idx].y += phi * anisotropic_delta.y * sx * sz;
                }
                if (need3) {
                    nt3[idx].x += phi * (aligned_base.x + anisotropic_delta.x * sy * sy);
                    nt3[idx].y += phi * (aligned_base.y + anisotropic_delta.y * sy * sy);
                }
                if (need4) {
                    nt4[idx].x += phi * anisotropic_delta.x * sy * sz;
                    nt4[idx].y += phi * anisotropic_delta.y * sy * sz;
                }
            }
            """,
            "nt_accumulate_anisotropic_float32_candidate",
        )
        cupy_rsoxs_module._CUPY_KERNEL_CACHE["nt_accumulate_anisotropic_float32_candidate"] = kernel
        return kernel

    def patched_compute_nt_components(self, runtime_materials, energy, cp, required_components=None):
        if cp.dtype(runtime_materials[0].Vfrac.dtype).name == "float16":
            return original_compute_nt_components(
                self,
                runtime_materials,
                energy,
                cp,
                required_components=required_components,
            )

        required = {0, 1, 2, 3, 4} if required_components is None else set(required_components)
        need0 = np.int32(0 in required)
        need1 = np.int32(1 in required)
        need2 = np.int32(2 in required)
        need3 = np.int32(3 in required)
        need4 = np.int32(4 in required)
        shape = tuple(int(v) for v in runtime_materials[0].Vfrac.shape)
        nt = cp.zeros((5, *shape), dtype=cp.complex64)
        nt0, nt1, nt2, nt3, nt4 = (nt[idx] for idx in range(5))
        threads = 256
        isotropic_kernel = self._nt_accumulate_isotropic_float32_kernel(cp)
        anisotropic_kernel = self._nt_accumulate_anisotropic_float32_kernel(cp)

        for material in runtime_materials:
            isotropic_diag, aligned_base, anisotropic_delta = self._material_optical_scalars(
                material,
                energy,
            )
            total = np.uint64(material.Vfrac.size)
            blocks = (material.Vfrac.size + threads - 1) // threads

            if material.is_full_isotropic:
                isotropic_kernel(
                    (blocks,),
                    (threads,),
                    (
                        material.Vfrac,
                        isotropic_diag,
                        need0,
                        need3,
                        nt0,
                        nt3,
                        total,
                    ),
                )
                continue

            anisotropic_kernel(
                (blocks,),
                (threads,),
                (
                    material.Vfrac,
                    material.S,
                    material.theta,
                    material.psi,
                    isotropic_diag,
                    aligned_base,
                    anisotropic_delta,
                    need0,
                    need1,
                    need2,
                    need3,
                    need4,
                    nt0,
                    nt1,
                    nt2,
                    nt3,
                    nt4,
                    total,
                ),
            )

        return nt

    runtime_cls._nt_accumulate_isotropic_float32_kernel = _nt_accumulate_isotropic_float32_kernel
    runtime_cls._nt_accumulate_anisotropic_float32_kernel = _nt_accumulate_anisotropic_float32_kernel
    runtime_cls._compute_nt_components = patched_compute_nt_components


def _apply_variant_patch(variant: str) -> None:
    if variant == "baseline":
        return
    if variant == "item1_fused_isotropic":
        _patch_item1_fused_isotropic()
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
        return "item1_fused_isotropic"
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
        _apply_variant_patch("item1_fused_isotropic")
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
                f"Tensor-coeff fused-isotropic recheck variant={variant}. "
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
        notes="Parity check for tensor_coeff fused isotropic accumulation versus maintained baseline.",
    )

    parity_dir = run_dir / "parity_case_results"
    parity_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tensor_coeff_fused_iso_parity_", dir=parity_dir) as tmp_dir:
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
        raise SystemExit("No visible NVIDIA GPU found for the tensor-coeff fused-isotropic recheck.")

    variants = _resolve_variants(args.variants)
    speed_case_keys = _resolve_case_keys(args.speed_cases, default=DEFAULT_SPEED_CASES)
    memory_case_keys = _resolve_case_keys(args.memory_cases, default=DEFAULT_MEMORY_CASES)

    run_label = args.label or f"tc_mem09_fused_isotropic_{_timestamp()}"
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
    candidate_variant = "item1_fused_isotropic"
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
            segment_b_ratio = None
            if case_key in speed_case_keys:
                baseline_primary = summary["speed_cases"][baseline_variant][case_key][
                    "primary_seconds_stats"
                ]["median"]
                candidate_primary = summary["speed_cases"][candidate_variant][case_key][
                    "primary_seconds_stats"
                ]["median"]
                speed_ratio = float(candidate_primary / baseline_primary)
                baseline_seg_b = summary["speed_cases"][baseline_variant][case_key][
                    "segment_seconds_stats"
                ]["B"]["median"]
                candidate_seg_b = summary["speed_cases"][candidate_variant][case_key][
                    "segment_seconds_stats"
                ]["B"]["median"]
                segment_b_ratio = float(candidate_seg_b / baseline_seg_b)

            summary["decision"][case_key] = {
                "status": (
                    "pass"
                    if (speed_ratio is None or speed_ratio < 1.05) and memory_ratio < 1.0
                    else "fail"
                ),
                "memory_peak_ratio": float(memory_ratio),
                "speed_ratio": None if speed_ratio is None else float(speed_ratio),
                "segment_b_ratio": None if segment_b_ratio is None else float(segment_b_ratio),
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
        seg_b_ratio = decision["segment_b_ratio"]
        line = (
            f"[decision] case={case_key} status={decision['status']} "
            f"memory_ratio={decision['memory_peak_ratio']:.4f}"
        )
        if speed_ratio is not None:
            line += f" speed_ratio={speed_ratio:.4f}"
        if seg_b_ratio is not None:
            line += f" segment_b_ratio={seg_b_ratio:.4f}"
        print(line, flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only hot-lane recheck for the tensor_coeff fused "
            "float32 isotropic accumulation experiment."
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
        default="baseline,item1_fused_isotropic",
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
