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
    _json_default,
    _timestamp,
    _worker_main,
)
from tests.validation.lib.core_shell import has_visible_gpu  # noqa: E402


OUT_ROOT = REPO_ROOT / "test-reports" / "core-shell-backend-performance-dev"
SUMMARY_NAME = "direct_polarization_memcleanup_recheck_summary.json"
ROTATION_SPECS = (
    ("no_rotation", (0.0, 0.0, 0.0), "no rotation"),
    ("rot_0_5_165", (0.0, 5.0, 165.0), "0:5:165"),
)
MEMORY_ROTATION_KEY = "rot_0_5_165"
VARIANTS = {
    "baseline": {
        "label": "baseline",
        "description": "Current maintained direct_polarization state.",
    },
    "item1_delete_fft_early": {
        "label": "item1",
        "description": "Delete fft_x/fft_y/fft_z immediately after Segment D.",
    },
    "item2_preallocate_result_storage": {
        "label": "item2",
        "description": "Preallocate result storage and avoid projections.append + cp.stack.",
    },
    "item3_reuse_polarization_buffers": {
        "label": "item3",
        "description": "Reuse dead polarization buffers as IGOR-shift outputs.",
    },
}


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


def _patch_item1_delete_fft_early() -> None:
    runtime_cls = cupy_rsoxs_module.CupyRsoxsBackendRuntime

    def patched_project_from_direct_polarization(
        self,
        morphology,
        runtime_materials,
        energy,
        cp,
        ndimage,
        window,
        angle_family_plan,
        isotropic_base_field=None,
        shape_override=None,
        recorder=None,
    ):
        recorder = cupy_rsoxs_module._NullSegmentRecorder() if recorder is None else recorder
        projection_average = None
        valid_counts = None
        use_rot_mask = bool(morphology.RotMask)
        num_angles = len(angle_family_plan.angles)
        for angle_plan, (matrix_yx, offset_yx) in zip(
            angle_family_plan.angles,
            self._rotation_transforms(morphology, cp),
        ):
            p_x, p_y, p_z = recorder.measure(
                "B",
                lambda angle_plan=angle_plan, isotropic_base_field=isotropic_base_field: self._compute_direct_polarization(
                    morphology=morphology,
                    runtime_materials=runtime_materials,
                    energy=energy,
                    angle_plan=angle_plan,
                    isotropic_base_field=isotropic_base_field,
                    cp=cp,
                ),
            )
            fft_x, fft_y, fft_z = recorder.measure(
                "C",
                lambda p_x=p_x, p_y=p_y, p_z=p_z: self._fft_polarization_fields(
                    morphology=morphology,
                    cp=cp,
                    p_x=p_x,
                    p_y=p_y,
                    p_z=p_z,
                    window=window,
                ),
            )
            del p_x, p_y, p_z
            projection = recorder.measure(
                "D",
                lambda fft_x=fft_x, fft_y=fft_y, fft_z=fft_z: self._projection_from_fft_polarization(
                    morphology=morphology,
                    energy=energy,
                    cp=cp,
                    fft_x=fft_x,
                    fft_y=fft_y,
                    fft_z=fft_z,
                    shape_override=shape_override,
                ),
            )
            del fft_x, fft_y, fft_z
            projection_average, valid_counts = recorder.measure(
                "E",
                lambda projection=projection, projection_average=projection_average, valid_counts=valid_counts, angle_plan=angle_plan: self._accumulate_rotated_projection(
                    cp=cp,
                    ndimage=ndimage,
                    projection=projection,
                    matrix_yx=matrix_yx,
                    offset_yx=offset_yx,
                    projection_average=projection_average,
                    valid_counts=valid_counts,
                    use_rot_mask=use_rot_mask,
                    skip_rotation=angle_plan.is_identity_rotation,
                ),
            )
            del projection
        return self._finalize_rotation_average(cp, projection_average, valid_counts, num_angles)

    runtime_cls._project_from_direct_polarization = patched_project_from_direct_polarization


def _patch_item2_preallocate_result_storage() -> None:
    runtime_cls = cupy_rsoxs_module.CupyRsoxsBackendRuntime

    def patched_run(
        self,
        morphology,
        *,
        stdout: bool = True,
        stderr: bool = True,
        return_xarray: bool = True,
        print_vec_info: bool = False,
        validate: bool = False,
    ):
        del stdout, stderr, print_vec_info

        if validate:
            self.validate_all(morphology, quiet=True)
        else:
            self._validate_supported_config(morphology)

        cp, ndimage = cupy_rsoxs_module.require_cupy_modules()
        self.prepare(morphology)
        recorder = self._segment_recorder(morphology, cp)
        morphology._backend_timings = {}
        self._update_kernel_reports(morphology)

        energies = tuple(float(energy) for energy in morphology.Energies)
        runtime_materials = recorder.measure("A2", lambda: self._runtime_material_views(morphology, cp))
        window = None
        result_data = None
        try:
            if self._kernel_preload_stage(morphology) == "a2":
                recorder.measure(
                    "A2",
                    lambda: self._preload_active_rawkernels(morphology, cp, stage="a2"),
                )
            window = recorder.measure(
                "C",
                lambda: self._window_tensor(
                    morphology,
                    cp,
                    shape_override=self._segment_c_shape_override(morphology),
                ),
            )

            for energy_index, energy in enumerate(energies):
                projection = self._run_single_energy(
                    morphology=morphology,
                    runtime_materials=runtime_materials,
                    energy=energy,
                    cp=cp,
                    ndimage=ndimage,
                    window=window,
                    recorder=recorder,
                )
                if result_data is None:
                    result_shape = (len(energies), *projection.shape)
                    result_data = cp.empty(result_shape, dtype=projection.dtype)
                result_data[energy_index] = projection
                del projection
        finally:
            if window is not None:
                del window
            del runtime_materials

        result = cupy_rsoxs_module.CupyScatteringResult(
            data=result_data,
            energies=energies,
            phys_size=float(morphology.PhysSize),
            num_zyx=tuple(int(v) for v in morphology.NumZYX),
        )

        morphology._backend_result = result
        morphology.scatteringPattern = result
        self._update_kernel_reports(morphology)
        segment_seconds, segment_measurements, measurement = recorder.finalize()
        if recorder.selected_segments:
            morphology._backend_timings = {
                "measurement": measurement,
                "selected_segments": list(recorder.selected_segments),
                "segment_seconds": segment_seconds,
                "segment_measurements": segment_measurements,
            }
        morphology._simulated = True
        morphology._lock_results()

        if return_xarray:
            return result.to_xarray()
        return result

    runtime_cls.run = patched_run


def _patch_item3_reuse_polarization_buffers() -> None:
    runtime_cls = cupy_rsoxs_module.CupyRsoxsBackendRuntime

    def patched_fft_polarization_fields(self, morphology, cp, p_x, p_y, p_z, window):
        if window is not None:
            cp.multiply(p_x, window, out=p_x)
            cp.multiply(p_y, window, out=p_y)
            cp.multiply(p_z, window, out=p_z)

        fft_x = cp.fft.fftn(p_x)
        self._replace_dc_component(fft_x)
        self._igor_shift(fft_x, morphology, cp, out=p_x)
        del fft_x

        fft_y = cp.fft.fftn(p_y)
        self._replace_dc_component(fft_y)
        self._igor_shift(fft_y, morphology, cp, out=p_y)
        del fft_y

        fft_z = cp.fft.fftn(p_z)
        self._replace_dc_component(fft_z)
        self._igor_shift(fft_z, morphology, cp, out=p_z)
        del fft_z

        return p_x, p_y, p_z

    runtime_cls._fft_polarization_fields = patched_fft_polarization_fields


def _apply_variant_patch(variant: str) -> None:
    if variant == "baseline":
        return
    if variant == "item1_delete_fft_early":
        _patch_item1_delete_fft_early()
        return
    if variant == "item2_preallocate_result_storage":
        _patch_item2_preallocate_result_storage()
        return
    if variant == "item3_reuse_polarization_buffers":
        _patch_item3_reuse_polarization_buffers()
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
        return "item1_delete_fft_early"
    if prefix == "item2":
        return "item2_preallocate_result_storage"
    if prefix == "item3":
        return "item3_reuse_polarization_buffers"
    raise ValueError(f"Could not infer worker variant from label {label!r}.")


def _worker_entry(case_path: Path, result_path: Path, variant: str) -> int:
    variant = _variant_from_worker_case(case_path, variant)
    _apply_variant_patch(variant)
    return _worker_main(case_path, result_path)


def _build_variant_case(
    *,
    variant: str,
    repeat_index: int,
    rotation_key: str,
    rotation_label: str,
    eangle_rotation: tuple[float, float, float],
    size_label: str,
) -> ComparisonCase:
    base_case = _cupy_case(
        residency="device",
        startup_mode="hot",
        execution_path="direct_polarization",
        z_collapse_mode=None,
        rotation_key=rotation_key,
        rotation_label=rotation_label,
        eangle_rotation=eangle_rotation,
        size_label=size_label,
    )
    variant_meta = VARIANTS[variant]
    variant_key = f"{variant_meta['label']}__{rotation_key}__run{repeat_index + 1:02d}"
    return replace(
        base_case,
        key=variant_key,
        script_path=Path(__file__).resolve(),
        worker_case=replace(
            base_case.worker_case,
            label=variant_key,
            notes=(
                f"Direct-polarization memcleanup recheck variant={variant}. "
                f"Repeat {repeat_index + 1}. "
                f"{variant_meta['description']}"
            ),
        ),
    )


def _run_recheck(args: argparse.Namespace) -> int:
    if not has_visible_gpu():
        raise SystemExit("No visible NVIDIA GPU found for the direct-path memcleanup recheck.")

    variants = tuple(args.variants.split(","))
    unknown = [variant for variant in variants if variant not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unsupported variants: {unknown!r}")

    run_label = args.label or f"dp_memcleanup_fastdelta_recheck_{_timestamp()}"
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
        "size_label": args.size_label,
        "repeats": int(args.repeats),
        "memory_poll_interval_s": float(args.memory_poll_interval_s),
        "variants": list(variants),
        "rotations": {
            key: {
                "eangle_rotation": list(rotation),
                "label": label,
            }
            for key, rotation, label in ROTATION_SPECS
        },
        "speed_cases": {},
        "memory_cases": {},
        "decision": {},
    }

    for variant in variants:
        summary["speed_cases"][variant] = {}
        for rotation_key, eangle_rotation, rotation_label in ROTATION_SPECS:
            run_results = []
            for repeat_index in range(args.repeats):
                case = _build_variant_case(
                    variant=variant,
                    repeat_index=repeat_index,
                    rotation_key=rotation_key,
                    rotation_label=rotation_label,
                    eangle_rotation=eangle_rotation,
                    size_label=args.size_label,
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
                primary = float(result["primary_seconds"])
                print(
                    f"[speed] variant={variant} rotation={rotation_key} run={repeat_index + 1}/{args.repeats} "
                    f"primary={primary:.6f}s",
                    flush=True,
                )
            summary["speed_cases"][variant][rotation_key] = {
                "rotation_label": rotation_label,
                "eangle_rotation": list(eangle_rotation),
                "runs": run_results,
                "primary_seconds_stats": _series_stats(
                    [float(result["primary_seconds"]) for result in run_results]
                ),
            }
            _write_json(summary_path, summary)

        memory_runs = []
        for repeat_index in range(args.repeats):
            case = _build_variant_case(
                variant=variant,
                repeat_index=repeat_index,
                rotation_key=MEMORY_ROTATION_KEY,
                rotation_label="0:5:165",
                eangle_rotation=(0.0, 5.0, 165.0),
                size_label=args.size_label,
            )
            result = _run_case_subprocess(
                case=case,
                output_dir=memory_dir,
                gpu_index=args.gpu_index,
                monitor_memory=True,
                poll_interval_s=args.memory_poll_interval_s,
                skip_existing=False,
            )
            memory_runs.append(result)
            peak_delta = float(result["memory_probe"]["peak_gpu_delta_mib"])
            sample_count = int(result["memory_probe"]["sample_count"])
            print(
                f"[memory] variant={variant} run={repeat_index + 1}/{args.repeats} "
                f"peak_delta={peak_delta:.3f}MiB samples={sample_count}",
                flush=True,
            )
        summary["memory_cases"][variant] = {
            "rotation_key": MEMORY_ROTATION_KEY,
            "rotation_label": "0:5:165",
            "eangle_rotation": [0.0, 5.0, 165.0],
            "runs": memory_runs,
            "peak_gpu_delta_mib_stats": _series_stats(
                [float(result["memory_probe"]["peak_gpu_delta_mib"]) for result in memory_runs]
            ),
            "sample_count_stats": _series_stats(
                [float(result["memory_probe"]["sample_count"]) for result in memory_runs]
            ),
        }
        _write_json(summary_path, summary)

    baseline_rot_primary = summary["speed_cases"]["baseline"]["rot_0_5_165"]["primary_seconds_stats"]["median"]
    baseline_no_rot_primary = summary["speed_cases"]["baseline"]["no_rotation"]["primary_seconds_stats"]["median"]
    baseline_peak_delta = summary["memory_cases"]["baseline"]["peak_gpu_delta_mib_stats"]["median"]

    for variant in variants:
        if variant == "baseline":
            continue
        rot_primary = summary["speed_cases"][variant]["rot_0_5_165"]["primary_seconds_stats"]["median"]
        no_rot_primary = summary["speed_cases"][variant]["no_rotation"]["primary_seconds_stats"]["median"]
        peak_delta = summary["memory_cases"][variant]["peak_gpu_delta_mib_stats"]["median"]
        speed_ratio = float(rot_primary / baseline_rot_primary)
        no_rot_ratio = float(no_rot_primary / baseline_no_rot_primary)
        memory_ratio = float(peak_delta / baseline_peak_delta) if baseline_peak_delta else float("inf")
        speed_pass = speed_ratio < 1.05
        memory_pass = memory_ratio < 1.05
        summary["decision"][variant] = {
            "status": "pass" if speed_pass and memory_pass else "fail",
            "speed_gate_pass": bool(speed_pass),
            "memory_gate_pass": bool(memory_pass),
            "no_rotation_guard_ratio": float(no_rot_ratio),
            "rotation_primary_ratio": float(speed_ratio),
            "memory_peak_ratio": float(memory_ratio),
            "rotation_primary_median_seconds": float(rot_primary),
            "rotation_primary_baseline_median_seconds": float(baseline_rot_primary),
            "memory_peak_delta_median_mib": float(peak_delta),
            "memory_peak_delta_baseline_median_mib": float(baseline_peak_delta),
        }

    _write_json(summary_path, summary)
    print(f"Wrote {summary_path}", flush=True)
    for variant, decision in summary["decision"].items():
        print(
            f"[decision] variant={variant} status={decision['status']} "
            f"speed_ratio={decision['rotation_primary_ratio']:.4f} "
            f"memory_ratio={decision['memory_peak_ratio']:.4f} "
            f"no_rotation_ratio={decision['no_rotation_guard_ratio']:.4f}",
            flush=True,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only recheck runner for the April 6 direct_polarization "
            "memory-lifetime cleanup rejects using the fast CuPy delta observer."
        )
    )
    parser.add_argument("--label", default=None, help="Output subdirectory label under test-reports.")
    parser.add_argument("--size-label", default="small", help="CoreShell size label to run.")
    parser.add_argument("--gpu-index", type=int, default=0, help="Global GPU index to pin.")
    parser.add_argument("--repeats", type=int, default=5, help="Repeated runs per variant.")
    parser.add_argument(
        "--memory-poll-interval-s",
        type=float,
        default=0.001,
        help="Fast CuPy observer sampling cadence for the memory pass.",
    )
    parser.add_argument(
        "--variants",
        default="baseline,item1_delete_fft_early,item2_preallocate_result_storage,item3_reuse_polarization_buffers",
        help="Comma-separated variant list.",
    )
    parser.add_argument("--worker-case-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-variant", default="baseline", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.worker_case_path and args.worker_result_path:
        return _worker_entry(
            case_path=Path(args.worker_case_path),
            result_path=Path(args.worker_result_path),
            variant=str(args.worker_variant),
        )
    return _run_recheck(args)


if __name__ == "__main__":
    raise SystemExit(main())
