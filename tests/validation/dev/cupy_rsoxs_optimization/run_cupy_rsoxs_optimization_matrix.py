#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Development studies should default to one visible GPU unless the caller has
# already pinned the runtime.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from NRSS.morphology import Material, Morphology, OpticalConstants
from tests.validation.lib.core_shell import has_visible_gpu
from tests.validation.lib.core_shell import release_runtime_memory


OUT_ROOT = REPO_ROOT / "test-reports" / "cupy-rsoxs-optimization-dev"
CORE_SHELL_DATA_DIR = REPO_ROOT / "tests" / "validation" / "data" / "core_shell"

CORE_SHELL_SINGLE_ENERGIES = (285.0,)
CORE_SHELL_TRIPLE_ENERGIES = (284.7, 285.0, 285.2)
EANGLE_OFF = (0.0, 0.0, 0.0)
EANGLE_LIMITED = (0.0, 15.0, 165.0)
EANGLE_FULL = (0.0, 1.0, 360.0)

CORE_RADIUS_VOX = 4.0
SHELL_THICKNESS_VOX = 2.94
PHI_ISO = 0.46
DECAY_ORDER = 0.42
CENTER_Z_VOX = 15.0

SUMMARY_NAME = "summary.json"
PRIMARY_TIMING_BOUNDARY = "Morphology(...) -> synchronized run(return_xarray=False)"
TIMING_SEGMENTS = ("A", "B", "C", "D", "E", "F")


@dataclass(frozen=True)
class SizeSpec:
    label: str
    shape: tuple[int, int, int]
    phys_size_nm: float
    scale: int


@dataclass(frozen=True)
class BenchmarkCase:
    label: str
    family: str
    backend: str
    shape_label: str
    energies_ev: tuple[float, ...]
    eangle_rotation: tuple[float, float, float]
    field_namespace: str
    input_policy: str
    ownership_policy: str | None
    timing_segments: tuple[str, ...] = ()
    create_cy_object: bool = True
    validation_baseline_name: str | None = None
    notes: str | None = None


SIZE_SPECS = {
    "small": SizeSpec("small", (32, 512, 512), 2.5, 1),
    "medium": SizeSpec("medium", (64, 1024, 1024), 1.25, 2),
    "large": SizeSpec("large", (96, 1536, 1536), np.float64(2.5 / 3.0), 3),
}


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _parse_csv_labels(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_timing_segments(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return TIMING_SEGMENTS
    requested = tuple(dict.fromkeys(part.strip().upper() for part in raw.split(",") if part.strip()))
    unknown = tuple(segment for segment in requested if segment not in TIMING_SEGMENTS)
    if unknown:
        raise SystemExit(
            f"Unsupported timing segments {unknown!r}. Valid values: {TIMING_SEGMENTS!r} or 'all'."
        )
    if not requested:
        raise SystemExit("timing_segments must select at least one segment.")
    return requested


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _subset_optical_constants(optical_constants: OpticalConstants, energies_ev: tuple[float, ...], *, name: str) -> OpticalConstants:
    subset = {float(energy): optical_constants.opt_constants[float(energy)] for energy in energies_ev}
    return OpticalConstants(list(map(float, energies_ev)), subset, name=name)


@lru_cache(maxsize=1)
def _load_core_shell_coord_table() -> np.ndarray:
    coords = np.genfromtxt(CORE_SHELL_DATA_DIR / "LoG_coord.csv", delimiter=",", skip_header=1)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise AssertionError("Unexpected CoreShell coordinate table shape.")
    return coords.astype(np.float32, copy=False)


@lru_cache(maxsize=None)
def _load_core_shell_optics(energies_ev: tuple[float, ...]) -> dict[int, OpticalConstants]:
    return {
        1: _subset_optical_constants(
            OpticalConstants.load_matfile(str(CORE_SHELL_DATA_DIR / "Material1.txt"), name="core"),
            energies_ev,
            name="core",
        ),
        2: _subset_optical_constants(
            OpticalConstants.load_matfile(str(CORE_SHELL_DATA_DIR / "Material2.txt"), name="shell"),
            energies_ev,
            name="shell",
        ),
        3: _subset_optical_constants(
            OpticalConstants.load_matfile(str(CORE_SHELL_DATA_DIR / "Material3.txt"), name="matrix"),
            energies_ev,
            name="matrix",
        ),
    }



def _convert_fields_namespace(fields: dict[str, np.ndarray], field_namespace: str) -> dict[str, Any]:
    if field_namespace == "numpy":
        return fields
    if field_namespace != "cupy":
        raise AssertionError(f"Unsupported field namespace: {field_namespace}")

    import cupy as cp

    return {
        key: cp.ascontiguousarray(cp.asarray(value, dtype=cp.float32))
        for key, value in fields.items()
    }


def _local_bounds(center: float, radius: float, size: int) -> tuple[int, int]:
    start = max(0, int(np.floor(center - radius)))
    stop = min(size, int(np.ceil(center + radius)) + 1)
    return start, stop


def _shell_orientation_angles(
    shell_mask: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ox = np.where(shell_mask, dx, 0.0).astype(np.float32, copy=False)
    oy = np.where(shell_mask, dy, 0.0).astype(np.float32, copy=False)
    oz = np.where(shell_mask, dz, 0.0).astype(np.float32, copy=False)
    theta = np.arctan2(np.sqrt(ox * ox + oy * oy, dtype=np.float32), oz).astype(np.float32, copy=False)
    psi = np.arctan2(oy, ox).astype(np.float32, copy=False)
    return theta, psi


def _build_scaled_core_shell_fields(size_spec: SizeSpec) -> dict[str, np.ndarray]:
    coords = _load_core_shell_coord_table()
    scale = float(size_spec.scale)
    nz, ny, nx = size_spec.shape

    core_radius_vox = np.float32(CORE_RADIUS_VOX * scale)
    shell_thickness_vox = np.float32(SHELL_THICKNESS_VOX * scale)
    shell_radius = np.float32(core_radius_vox + shell_thickness_vox)
    center_z = np.float32(CENTER_Z_VOX * scale)

    a_b = np.zeros(size_spec.shape, dtype=bool)
    b_b = np.zeros(size_spec.shape, dtype=bool)
    radial_x = np.zeros(size_spec.shape, dtype=np.float32)
    radial_y = np.zeros(size_spec.shape, dtype=np.float32)
    radial_z = np.zeros(size_spec.shape, dtype=np.float32)

    z0, z1 = _local_bounds(float(center_z), float(shell_radius), nz)
    z = np.arange(z0, z1, dtype=np.float32)[:, None, None]

    for row in coords:
        px = np.float32(float(row[0]) * scale)
        py = np.float32(float(row[1]) * scale)
        y0, y1 = _local_bounds(float(py), float(shell_radius), ny)
        x0, x1 = _local_bounds(float(px), float(shell_radius), nx)

        y = np.arange(y0, y1, dtype=np.float32)[None, :, None]
        x = np.arange(x0, x1, dtype=np.float32)[None, None, :]

        mf = (x - px) ** 2 + (y - py) ** 2 + (z - center_z) ** 2
        core_mask = mf <= np.float32(core_radius_vox * core_radius_vox)
        shell_radius_mask = mf <= np.float32(shell_radius * shell_radius)

        a_view = a_b[z0:z1, y0:y1, x0:x1]
        a_view |= core_mask

        shell_mask = np.logical_and(~a_view, shell_radius_mask)
        b_view = b_b[z0:z1, y0:y1, x0:x1]
        b_view |= shell_mask

        dx = (x - px).astype(np.float32, copy=False)
        dy = (y - py).astype(np.float32, copy=False)
        dz = (z - center_z).astype(np.float32, copy=False)

        rx_view = radial_x[z0:z1, y0:y1, x0:x1]
        ry_view = radial_y[z0:z1, y0:y1, x0:x1]
        rz_view = radial_z[z0:z1, y0:y1, x0:x1]

        rx_view += dx * shell_mask * (rx_view == 0)
        ry_view += dy * shell_mask * (ry_view == 0)
        rz_view += dz * shell_mask * (rz_view == 0)

    b_b = np.logical_and(~a_b, b_b)
    radial_x *= b_b
    radial_y *= b_b
    radial_z *= b_b
    c_b = np.logical_and(~a_b, ~b_b)

    radial_norm = np.sqrt(
        radial_x * radial_x + radial_y * radial_y + radial_z * radial_z,
        dtype=np.float32,
    )
    ratio = np.divide(
        core_radius_vox * b_b.astype(np.float32),
        radial_norm,
        out=np.zeros_like(radial_norm, dtype=np.float32),
        where=radial_norm > 0,
    )

    vf_a = a_b.astype(np.float32)
    vf_b = b_b.astype(np.float32)
    vf_c = c_b.astype(np.float32)

    shell_s = vf_b * np.float32(1.0 - PHI_ISO) * np.power(
        ratio,
        np.float32(DECAY_ORDER),
        dtype=np.float32,
    )
    shell_s = np.nan_to_num(shell_s, copy=False)

    theta_b, psi_b = _shell_orientation_angles(
        shell_mask=b_b,
        dx=radial_x,
        dy=radial_y,
        dz=radial_z,
    )

    zeros = np.zeros(size_spec.shape, dtype=np.float32)
    return {
        "mat1_vfrac": vf_a,
        "mat1_s": zeros.copy(),
        "mat1_theta": zeros.copy(),
        "mat1_psi": zeros.copy(),
        "mat2_vfrac": vf_b,
        "mat2_s": shell_s.astype(np.float32, copy=False),
        "mat2_theta": theta_b,
        "mat2_psi": psi_b,
        "mat3_vfrac": vf_c,
        "mat3_s": zeros.copy(),
        "mat3_theta": zeros.copy(),
        "mat3_psi": zeros.copy(),
    }


def _default_morphology_config(
    energies_ev: tuple[float, ...],
    eangle_rotation: tuple[float, float, float],
) -> dict[str, Any]:
    return {
        "CaseType": 0,
        "MorphologyType": 0,
        "Energies": list(map(float, energies_ev)),
        "EAngleRotation": list(map(float, eangle_rotation)),
        "AlgorithmType": 0,
        "WindowingType": 0,
        "RotMask": 0,
        "ReferenceFrame": 1,
        "EwaldsInterpolation": 1,
    }


def build_scaled_core_shell_materials(
    *,
    size_spec: SizeSpec,
    energies_ev: tuple[float, ...],
    field_namespace: str,
) -> dict[int, Material]:
    optics = _load_core_shell_optics(tuple(map(float, energies_ev)))
    fields = _convert_fields_namespace(_build_scaled_core_shell_fields(size_spec), field_namespace)

    return {
        1: Material(
            materialID=1,
            Vfrac=fields["mat1_vfrac"],
            S=fields["mat1_s"],
            theta=fields["mat1_theta"],
            psi=fields["mat1_psi"],
            energies=list(map(float, energies_ev)),
            opt_constants=optics[1].opt_constants,
            name="core",
        ),
        2: Material(
            materialID=2,
            Vfrac=fields["mat2_vfrac"],
            S=fields["mat2_s"],
            theta=fields["mat2_theta"],
            psi=fields["mat2_psi"],
            energies=list(map(float, energies_ev)),
            opt_constants=optics[2].opt_constants,
            name="shell",
        ),
        3: Material(
            materialID=3,
            Vfrac=fields["mat3_vfrac"],
            S=fields["mat3_s"],
            theta=fields["mat3_theta"],
            psi=fields["mat3_psi"],
            energies=list(map(float, energies_ev)),
            opt_constants=optics[3].opt_constants,
            name="matrix",
        ),
    }


def build_scaled_core_shell_morphology(
    *,
    size_spec: SizeSpec,
    energies_ev: tuple[float, ...],
    eangle_rotation: tuple[float, float, float],
    backend: str,
    field_namespace: str,
    input_policy: str,
    ownership_policy: str | None,
    create_cy_object: bool,
) -> Morphology:
    materials = build_scaled_core_shell_materials(
        size_spec=size_spec,
        energies_ev=energies_ev,
        field_namespace=field_namespace,
    )
    config = _default_morphology_config(energies_ev, eangle_rotation)

    morph = Morphology(
        3,
        materials=materials,
        PhysSize=float(size_spec.phys_size_nm),
        config=config,
        create_cy_object=create_cy_object,
        backend=backend,
        input_policy=input_policy,
        ownership_policy=ownership_policy,
    )
    morph.check_materials(quiet=True)
    return morph


def _case_note(case: BenchmarkCase) -> str:
    size_spec = SIZE_SPECS.get(case.shape_label)
    if case.family == "core_shell":
        return (
            f"CoreShell {case.shape_label} {size_spec.shape} PhysSize={size_spec.phys_size_nm} "
            f"EAngleRotation={list(case.eangle_rotation)} energies={list(case.energies_ev)}"
        )
    raise AssertionError(f"Unsupported benchmark family for the timing harness: {case.family}")


def _result_summary_line(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return f"{result['label']}: {result.get('status')} ({result.get('error_type', 'unknown')})"
    segment_seconds = result.get("segment_seconds", {})
    ordered = [
        f"{segment} {float(segment_seconds[segment]):.3f}s"
        for segment in TIMING_SEGMENTS
        if segment in segment_seconds
    ]
    suffix = "" if not ordered else ", " + ", ".join(ordered)
    return f"{result['label']}: primary {float(result['primary_seconds']):.3f}s{suffix}"


def _prepare_core_shell_case_inputs(case: BenchmarkCase) -> dict[str, Any]:
    if case.family != "core_shell":
        raise AssertionError(f"Unsupported timing family: {case.family}")
    size_spec = SIZE_SPECS[case.shape_label]
    return {
        "num_material": 3,
        "materials": build_scaled_core_shell_materials(
            size_spec=size_spec,
            energies_ev=case.energies_ev,
            field_namespace=case.field_namespace,
        ),
        "phys_size": float(size_spec.phys_size_nm),
        "config": _default_morphology_config(case.energies_ev, case.eangle_rotation),
    }


def _construct_morphology_for_timing_case(case: BenchmarkCase, prepared: dict[str, Any]) -> Morphology:
    return Morphology(
        prepared["num_material"],
        materials=prepared["materials"],
        PhysSize=prepared["phys_size"],
        config=prepared["config"],
        create_cy_object=case.create_cy_object,
        backend=case.backend,
        input_policy=case.input_policy,
        ownership_policy=case.ownership_policy,
    )


def _synchronize_cupy_default_stream() -> None:
    import cupy as cp

    cp.cuda.Stream.null.synchronize()


def _worker_main(case_path: Path, result_path: Path) -> int:
    case = BenchmarkCase(**json.loads(case_path.read_text()))
    result: dict[str, Any] = {
        "label": case.label,
        "family": case.family,
        "backend": case.backend,
        "shape_label": case.shape_label,
        "energies_ev": list(case.energies_ev),
        "eangle_rotation": list(case.eangle_rotation),
        "timing_segments_requested": list(case.timing_segments),
        "timing_boundary": PRIMARY_TIMING_BOUNDARY,
        "note": _case_note(case),
        "status": "error",
    }

    if case.backend != "cupy-rsoxs":
        raise AssertionError("The optimization timing harness only supports backend='cupy-rsoxs'.")

    prepared_inputs = None
    morphology = None
    backend_result = None
    try:
        prepared_inputs = _prepare_core_shell_case_inputs(case)
        if case.field_namespace == "cupy":
            _synchronize_cupy_default_stream()

        primary_start = time.perf_counter()
        morphology = _construct_morphology_for_timing_case(case, prepared_inputs)
        segment_seconds: dict[str, float] = {}
        if "A" in case.timing_segments:
            segment_seconds["A"] = time.perf_counter() - primary_start

        morphology._set_private_backend_timing_segments(case.timing_segments)
        backend_result = morphology.run(stdout=False, stderr=False, return_xarray=False)
        _synchronize_cupy_default_stream()

        backend_timings = morphology.backend_timings
        if backend_timings:
            segment_seconds.update(backend_timings.get("segment_seconds", {}))

        result["primary_seconds"] = time.perf_counter() - primary_start
        result["segment_seconds"] = {
            segment: float(segment_seconds[segment])
            for segment in TIMING_SEGMENTS
            if segment in segment_seconds
        }
        result["backend_timing_details"] = backend_timings
        result["panel_shape"] = list(backend_result.to_backend_array().shape)
        result["status"] = "ok"

    except BaseException as exc:  # noqa: BLE001 - worker must serialize failures
        result["status"] = "error"
        result["error_type"] = exc.__class__.__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
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
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2, default=_json_default) + "\n")
    return 0


def _run_case_subprocess(
    *,
    case: BenchmarkCase,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cupy_rsoxs_opt_", dir=output_dir) as tmp_dir:
        tmp_path = Path(tmp_dir)
        case_path = tmp_path / "case.json"
        result_path = tmp_path / "result.json"
        case_path.write_text(json.dumps(asdict(case), indent=2, default=_json_default) + "\n")

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-case-path",
            str(case_path),
            "--worker-result-path",
            str(result_path),
        ]

        started = time.perf_counter()
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - started

        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            result = {
                "label": case.label,
                "backend": case.backend,
                "family": case.family,
                "shape_label": case.shape_label,
                "status": "subprocess_failed",
                "error_type": "SubprocessFailure",
                "error": "Worker exited before writing a result file.",
            }

        result["subprocess_returncode"] = int(completed.returncode)
        result["subprocess_seconds"] = elapsed
        if completed.stdout.strip():
            result["worker_stdout"] = completed.stdout[-4000:]
        if completed.stderr.strip():
            result["worker_stderr"] = completed.stderr[-4000:]
        return result


def _timing_cases(
    *,
    size_labels: tuple[str, ...],
    timing_segments: tuple[str, ...],
    include_full_small_check: bool,
) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for size_label in size_labels:
        cases.append(
            BenchmarkCase(
                label=f"core_shell_{size_label}_single_no_rotation_cupy_borrow",
                family="core_shell",
                backend="cupy-rsoxs",
                shape_label=size_label,
                energies_ev=CORE_SHELL_SINGLE_ENERGIES,
                eangle_rotation=EANGLE_OFF,
                field_namespace="cupy",
                input_policy="strict",
                ownership_policy="borrow",
                timing_segments=timing_segments,
                notes="Primary no-rotation tuning lane.",
            )
        )
        cases.append(
            BenchmarkCase(
                label=f"core_shell_{size_label}_triple_limited_rotation_cupy_borrow",
                family="core_shell",
                backend="cupy-rsoxs",
                shape_label=size_label,
                energies_ev=CORE_SHELL_TRIPLE_ENERGIES,
                eangle_rotation=EANGLE_LIMITED,
                field_namespace="cupy",
                input_policy="strict",
                ownership_policy="borrow",
                timing_segments=timing_segments,
                notes="Primary limited-EAngle tuning lane.",
            )
        )

    if include_full_small_check and "small" in size_labels:
        cases.append(
            BenchmarkCase(
                label="core_shell_small_triple_full_rotation_cupy_borrow",
                family="core_shell",
                backend="cupy-rsoxs",
                shape_label="small",
                energies_ev=CORE_SHELL_TRIPLE_ENERGIES,
                eangle_rotation=EANGLE_FULL,
                field_namespace="cupy",
                input_policy="strict",
                ownership_policy="borrow",
                timing_segments=timing_segments,
                notes="Occasional expensive checkpoint for the full parity-style rotation loop.",
            )
        )
    return cases


def _write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")


def run_matrix(args: argparse.Namespace) -> int:
    if not has_visible_gpu():
        raise SystemExit("No visible NVIDIA GPU found for the cupy-rsoxs optimization study.")

    run_label = args.label or _timestamp()
    size_labels = _parse_csv_labels(args.size_labels)
    unknown = [label for label in size_labels if label not in SIZE_SPECS]
    if unknown:
        raise SystemExit(f"Unsupported size_labels entries: {unknown!r}")
    timing_segments = _parse_timing_segments(args.timing_segments)

    run_dir = OUT_ROOT / run_label
    output_dir = run_dir / "cases"
    summary: dict[str, Any] = {
        "label": run_label,
        "created_utc": _timestamp(),
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "timing_boundary": PRIMARY_TIMING_BOUNDARY,
        "timing_segments": list(timing_segments),
        "size_labels": list(size_labels),
        "segment_g_status": "Reserved for future export timing. Not recorded in this pass.",
        "timing_cases": {},
    }

    run_dir.mkdir(parents=True, exist_ok=True)

    print("Running cupy-rsoxs timing cases...", flush=True)
    for case in _timing_cases(
        size_labels=size_labels,
        timing_segments=timing_segments,
        include_full_small_check=args.include_full_small_check,
    ):
        result = _run_case_subprocess(case=case, output_dir=output_dir)
        summary["timing_cases"][case.label] = result
        print(_result_summary_line(result), flush=True)

    _write_summary(run_dir, summary)
    print(f"Wrote {run_dir / SUMMARY_NAME}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only cupy-rsoxs timing matrix for optimization work. "
            "Primary timing starts at Morphology(...) construction and ends after "
            "synchronized run(return_xarray=False) completion."
        )
    )
    parser.add_argument("--label", default=None, help="Output subdirectory label under test-reports.")
    parser.add_argument(
        "--size-labels",
        default="small,medium,large",
        help="Comma-separated subset of size labels to run, for example 'small,medium'.",
    )
    parser.add_argument(
        "--timing-segments",
        default="all",
        help="Comma-separated timing segments to record, or 'all'. Supported segments: A-F.",
    )
    parser.add_argument(
        "--include-full-small-check",
        action="store_true",
        help="Include the expensive full-rotation small CoreShell cupy case.",
    )
    parser.add_argument("--worker-case-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result-path", default=None, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.worker_case_path:
        return _worker_main(
            case_path=Path(args.worker_case_path),
            result_path=Path(args.worker_result_path),
        )
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
