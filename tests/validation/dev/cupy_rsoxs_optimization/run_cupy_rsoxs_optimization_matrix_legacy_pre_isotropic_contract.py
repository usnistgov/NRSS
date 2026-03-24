#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
from NRSS.backends import normalize_backend_options
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
TIMING_SEGMENTS = ("A1", "A2", "B", "C", "D", "E", "F")
TIMING_SEGMENT_ALIASES = {
    "A": ("A1", "A2"),
}


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
    resident_mode: str | None
    input_policy: str
    ownership_policy: str | None
    backend_options: dict[str, Any] | None = None
    timing_segments: tuple[str, ...] = ()
    create_cy_object: bool = True
    validation_baseline_name: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class ResidentVariantSpec:
    key: str
    label_suffix: str
    field_namespace: str
    resident_mode: str
    input_policy: str
    ownership_policy: str | None
    notes: str


SIZE_SPECS = {
    "small": SizeSpec("small", (32, 512, 512), 2.5, 1),
    "medium": SizeSpec("medium", (64, 1024, 1024), 1.25, 2),
    "large": SizeSpec("large", (96, 1536, 1536), np.float64(2.5 / 3.0), 3),
}


RESIDENT_VARIANTS = {
    "host": ResidentVariantSpec(
        key="host",
        label_suffix="host",
        field_namespace="numpy",
        resident_mode="host",
        input_policy="strict",
        ownership_policy="borrow",
        notes="Default public-workflow lane with host-resident authoritative morphology fields.",
    ),
    "device": ResidentVariantSpec(
        key="device",
        label_suffix="device",
        field_namespace="cupy",
        resident_mode="device",
        input_policy="strict",
        ownership_policy="borrow",
        notes="Opt-in device-resident borrowed lane for direct CuPy morphology workflows.",
    ),
}


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _parse_csv_labels(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_positive_int_csv(raw: str) -> tuple[int, ...]:
    requested = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    if not requested:
        return ()
    values: list[int] = []
    for item in requested:
        try:
            value = int(item)
        except ValueError as exc:
            raise SystemExit(f"Expected a comma-separated list of positive integers, got {item!r}.") from exc
        if value <= 0:
            raise SystemExit(f"Energy counts must be positive integers, got {value!r}.")
        values.append(value)
    return tuple(values)


def _parse_rotation_specs(raw: str) -> tuple[tuple[float, float, float], ...]:
    requested = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    if not requested:
        return ()

    specs: list[tuple[float, float, float]] = []
    for item in requested:
        parts = tuple(part.strip() for part in item.split(":"))
        if len(parts) != 3 or any(not part for part in parts):
            raise SystemExit(
                "Rotation specs must be comma-separated 'start:increment:end' triples, "
                f"got {item!r}."
            )
        try:
            start, increment, end = (float(part) for part in parts)
        except ValueError as exc:
            raise SystemExit(
                "Rotation specs must contain finite numeric values in "
                f"'start:increment:end' order, got {item!r}."
            ) from exc
        if not all(math.isfinite(value) for value in (start, increment, end)):
            raise SystemExit(f"Rotation spec values must be finite, got {item!r}.")

        spec = tuple(0.0 if value == 0.0 else float(value) for value in (start, increment, end))
        if spec[1] == 0.0 and not math.isclose(spec[0], spec[2], rel_tol=0.0, abs_tol=1e-12):
            raise SystemExit(
                "Rotation specs with zero increment must also have start == end, "
                f"got {item!r}."
            )
        if spec[1] != 0.0 and ((spec[2] - spec[0]) / spec[1]) < 0:
            raise SystemExit(
                "Rotation specs must advance from start to end using the sign of the increment, "
                f"got {item!r}."
            )
        specs.append(spec)
    return tuple(specs)


def _parse_resident_modes(raw: str) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(part.strip().lower() for part in raw.split(",") if part.strip()))
    unknown = tuple(mode for mode in requested if mode not in RESIDENT_VARIANTS)
    if unknown:
        raise SystemExit(
            f"Unsupported resident_modes {unknown!r}. Valid values: {tuple(RESIDENT_VARIANTS)!r}."
        )
    if not requested:
        raise SystemExit("resident_modes must select at least one variant.")
    return requested


def _parse_timing_segments(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return TIMING_SEGMENTS
    expanded: list[str] = []
    for part in raw.split(","):
        cleaned = part.strip().upper()
        if not cleaned:
            continue
        expanded.extend(TIMING_SEGMENT_ALIASES.get(cleaned, (cleaned,)))
    requested = tuple(dict.fromkeys(expanded))
    unknown = tuple(segment for segment in requested if segment not in TIMING_SEGMENTS)
    if unknown:
        raise SystemExit(
            "Unsupported timing segments "
            f"{unknown!r}. Valid values: {TIMING_SEGMENTS!r}, alias 'A', or 'all'."
        )
    if not requested:
        raise SystemExit("timing_segments must select at least one segment.")
    return requested


def _parse_execution_paths(raw: str) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    if not requested:
        raise SystemExit("execution_paths must select at least one execution path.")

    normalized: list[str] = []
    for item in requested:
        try:
            resolved = normalize_backend_options(
                "cupy-rsoxs",
                {"execution_path": item},
            )["execution_path"]
        except Exception as exc:
            raise SystemExit(str(exc)) from exc
        normalized.append(resolved)
    return tuple(dict.fromkeys(normalized))


def _resolve_core_shell_energy(value: float, *, available: tuple[float, ...]) -> float:
    closest = min(available, key=lambda candidate: abs(candidate - value))
    if not math.isclose(float(closest), float(value), rel_tol=0.0, abs_tol=1e-6):
        raise SystemExit(
            f"Energy {value!r} is not present in the CoreShell optics table. "
            f"Closest available energy is {closest!r}."
        )
    return float(closest)


def _parse_energy_lists(raw: str) -> tuple[tuple[float, ...], ...]:
    requested = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    if not requested:
        return ()

    available = _load_core_shell_available_energies()
    energy_lists: list[tuple[float, ...]] = []
    for item in requested:
        parts = tuple(part.strip() for part in item.split("|") if part.strip())
        if not parts:
            raise SystemExit(
                "Energy lists must be comma-separated groups of '|' separated values, "
                f"got {item!r}."
            )
        resolved: list[float] = []
        seen_within_group: set[float] = set()
        for part in parts:
            try:
                value = float(part)
            except ValueError as exc:
                raise SystemExit(
                    "Energy lists must contain numeric values separated with '|', "
                    f"got {part!r} in {item!r}."
                ) from exc
            if not math.isfinite(value):
                raise SystemExit(f"Energy list values must be finite, got {part!r} in {item!r}.")
            resolved_value = _resolve_core_shell_energy(value, available=available)
            if resolved_value in seen_within_group:
                raise SystemExit(f"Duplicate energy {resolved_value!r} in explicit energy list {item!r}.")
            seen_within_group.add(resolved_value)
            resolved.append(resolved_value)
        energy_lists.append(tuple(resolved))
    return tuple(energy_lists)


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _format_label_float(value: float) -> str:
    text = np.format_float_positional(float(value), trim="-")
    if text == "-0":
        text = "0"
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def _rotation_label_fragment(eangle_rotation: tuple[float, float, float]) -> str:
    return "rot_" + "_".join(_format_label_float(value) for value in eangle_rotation)


def _energy_list_label_fragment(energies_ev: tuple[float, ...]) -> str:
    tokens = [_format_label_float(value) for value in energies_ev]
    joined = "_".join(tokens)
    if len(joined) <= 72:
        return "elist_" + joined

    digest_input = ",".join(np.format_float_positional(float(value), trim="-") for value in energies_ev)
    digest = hashlib.sha1(digest_input.encode("ascii")).hexdigest()[:8]
    return f"elist_{len(energies_ev)}_{tokens[0]}_{tokens[-1]}_{digest}"


def _subset_optical_constants(
    optical_constants: OpticalConstants,
    energies_ev: tuple[float, ...],
    *,
    name: str,
) -> OpticalConstants:
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


@lru_cache(maxsize=1)
def _load_core_shell_available_energies() -> tuple[float, ...]:
    optical_constants = OpticalConstants.load_matfile(str(CORE_SHELL_DATA_DIR / "Material1.txt"), name="core")
    return tuple(float(energy) for energy in optical_constants.energies)


def _centered_core_shell_energies(count: int) -> tuple[float, ...]:
    if count <= 0:
        raise ValueError(f"count must be positive, got {count!r}.")
    if count == 1:
        return CORE_SHELL_SINGLE_ENERGIES
    available = _load_core_shell_available_energies()
    if count > len(available):
        raise ValueError(
            f"Requested {count} energies, but only {len(available)} are available in the CoreShell optics table."
        )

    center_energy = CORE_SHELL_SINGLE_ENERGIES[0]
    center_index = min(
        range(len(available)),
        key=lambda idx: (abs(available[idx] - center_energy), idx),
    )
    start = center_index - (count // 2)
    stop = start + count
    if start < 0:
        start = 0
        stop = count
    if stop > len(available):
        stop = len(available)
        start = stop - count
    subset = tuple(available[start:stop])
    if len(subset) != count:
        raise AssertionError(
            f"Internal energy-selection error: expected {count} energies, got {len(subset)}."
        )
    return subset



def _convert_fields_namespace(fields: dict[str, np.ndarray], field_namespace: str) -> dict[str, Any]:
    if field_namespace == "numpy":
        return {
            key: np.ascontiguousarray(np.asarray(value, dtype=np.float32))
            for key, value in fields.items()
        }
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
    resident_mode: str | None,
    input_policy: str,
    ownership_policy: str | None,
    create_cy_object: bool,
    backend_options: dict[str, Any] | None = None,
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
        backend_options=backend_options,
        resident_mode=resident_mode,
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
            f"EAngleRotation={list(case.eangle_rotation)} energies={list(case.energies_ev)} "
            f"resident_mode={case.resident_mode} field_namespace={case.field_namespace} "
            f"input_policy={case.input_policy} ownership_policy={case.ownership_policy} "
            f"backend_options={case.backend_options or {}}"
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
        "backend_options": case.backend_options,
    }


def _construct_morphology_for_timing_case(case: BenchmarkCase, prepared: dict[str, Any]) -> Morphology:
    return Morphology(
        prepared["num_material"],
        materials=prepared["materials"],
        PhysSize=prepared["phys_size"],
        config=prepared["config"],
        create_cy_object=case.create_cy_object,
        backend=case.backend,
        backend_options=prepared["backend_options"],
        resident_mode=case.resident_mode,
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
        "resident_mode": case.resident_mode,
        "field_namespace": case.field_namespace,
        "input_policy": case.input_policy,
        "ownership_policy": case.ownership_policy,
        "backend_options": dict(case.backend_options or {}),
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
        result["resolved_backend_options"] = dict(morphology.backend_options)
        segment_seconds: dict[str, float] = {}
        if "A1" in case.timing_segments:
            segment_seconds["A1"] = time.perf_counter() - primary_start

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
    resident_modes: tuple[str, ...],
    execution_paths: tuple[str, ...],
    size_labels: tuple[str, ...],
    timing_segments: tuple[str, ...],
    include_triple_no_rotation: bool,
    include_triple_limited: bool,
    include_full_small_check: bool,
    no_rotation_energy_counts: tuple[int, ...],
    rotation_specs: tuple[tuple[float, float, float], ...],
    energy_lists: tuple[tuple[float, ...], ...],
) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for mode in resident_modes:
        variant = RESIDENT_VARIANTS[mode]
        for execution_path in execution_paths:
            backend_options = {"execution_path": execution_path}

            def append_case(
                *,
                label: str,
                size_label: str,
                energies_ev: tuple[float, ...],
                eangle_rotation: tuple[float, float, float],
                notes: str,
            ) -> None:
                cases.append(
                    BenchmarkCase(
                        label=label,
                        family="core_shell",
                        backend="cupy-rsoxs",
                        shape_label=size_label,
                        energies_ev=energies_ev,
                        eangle_rotation=eangle_rotation,
                        field_namespace=variant.field_namespace,
                        resident_mode=variant.resident_mode,
                        input_policy=variant.input_policy,
                        ownership_policy=variant.ownership_policy,
                        backend_options=backend_options,
                        timing_segments=timing_segments,
                        notes=notes,
                    )
                )

            for size_label in size_labels:
                append_case(
                    label=(
                        f"core_shell_{size_label}_single_no_rotation_"
                        f"{variant.label_suffix}_{execution_path}"
                    ),
                    size_label=size_label,
                    energies_ev=CORE_SHELL_SINGLE_ENERGIES,
                    eangle_rotation=EANGLE_OFF,
                    notes=(
                        f"{variant.notes} Primary no-rotation tuning lane. "
                        f"execution_path={execution_path}."
                    ),
                )

                if include_triple_no_rotation:
                    append_case(
                        label=(
                            f"core_shell_{size_label}_triple_no_rotation_"
                            f"{variant.label_suffix}_{execution_path}"
                        ),
                        size_label=size_label,
                        energies_ev=CORE_SHELL_TRIPLE_ENERGIES,
                        eangle_rotation=EANGLE_OFF,
                        notes=(
                            f"{variant.notes} Secondary three-energy no-rotation checkpoint lane. "
                            f"execution_path={execution_path}."
                        ),
                    )

                for energy_count in no_rotation_energy_counts:
                    if energy_count == 1:
                        continue
                    append_case(
                        label=(
                            f"core_shell_{size_label}_{energy_count}energy_no_rotation_"
                            f"{variant.label_suffix}_{execution_path}"
                        ),
                        size_label=size_label,
                        energies_ev=_centered_core_shell_energies(energy_count),
                        eangle_rotation=EANGLE_OFF,
                        notes=(
                            f"{variant.notes} Development-only centered-energy no-rotation lane "
                            f"with {energy_count} energies. execution_path={execution_path}."
                        ),
                    )

                if include_triple_limited:
                    append_case(
                        label=(
                            f"core_shell_{size_label}_triple_limited_rotation_"
                            f"{variant.label_suffix}_{execution_path}"
                        ),
                        size_label=size_label,
                        energies_ev=CORE_SHELL_TRIPLE_ENERGIES,
                        eangle_rotation=EANGLE_LIMITED,
                        notes=(
                            f"{variant.notes} Secondary limited-EAngle checkpoint lane. "
                            f"execution_path={execution_path}."
                        ),
                    )

                for eangle_rotation in rotation_specs:
                    if eangle_rotation == EANGLE_OFF:
                        continue
                    rotation_label = _rotation_label_fragment(eangle_rotation)
                    append_case(
                        label=(
                            f"core_shell_{size_label}_single_{rotation_label}_"
                            f"{variant.label_suffix}_{execution_path}"
                        ),
                        size_label=size_label,
                        energies_ev=CORE_SHELL_SINGLE_ENERGIES,
                        eangle_rotation=eangle_rotation,
                        notes=(
                            f"{variant.notes} Custom single-energy rotation lane with "
                            f"EAngleRotation={list(eangle_rotation)} "
                            f"([StartAngle, IncrementAngle, EndAngle]). "
                            f"execution_path={execution_path}."
                        ),
                    )

                for energies_ev in energy_lists:
                    if energies_ev == CORE_SHELL_SINGLE_ENERGIES:
                        continue
                    energy_label = _energy_list_label_fragment(energies_ev)
                    append_case(
                        label=(
                            f"core_shell_{size_label}_{energy_label}_no_rotation_"
                            f"{variant.label_suffix}_{execution_path}"
                        ),
                        size_label=size_label,
                        energies_ev=energies_ev,
                        eangle_rotation=EANGLE_OFF,
                        notes=(
                            f"{variant.notes} Custom explicit-energy no-rotation lane with "
                            f"energies={list(energies_ev)}. execution_path={execution_path}."
                        ),
                    )

                for eangle_rotation in rotation_specs:
                    if eangle_rotation == EANGLE_OFF:
                        continue
                    rotation_label = _rotation_label_fragment(eangle_rotation)
                    for energies_ev in energy_lists:
                        if energies_ev == CORE_SHELL_SINGLE_ENERGIES:
                            continue
                        energy_label = _energy_list_label_fragment(energies_ev)
                        append_case(
                            label=(
                                f"core_shell_{size_label}_{energy_label}_{rotation_label}_"
                                f"{variant.label_suffix}_{execution_path}"
                            ),
                            size_label=size_label,
                            energies_ev=energies_ev,
                            eangle_rotation=eangle_rotation,
                            notes=(
                                f"{variant.notes} Custom explicit-energy plus custom rotation lane "
                                f"with energies={list(energies_ev)} and "
                                f"EAngleRotation={list(eangle_rotation)} "
                                f"([StartAngle, IncrementAngle, EndAngle]). "
                                f"execution_path={execution_path}."
                            ),
                        )

            if include_full_small_check and "small" in size_labels:
                append_case(
                    label=(
                        f"core_shell_small_triple_full_rotation_"
                        f"{variant.label_suffix}_{execution_path}"
                    ),
                    size_label="small",
                    energies_ev=CORE_SHELL_TRIPLE_ENERGIES,
                    eangle_rotation=EANGLE_FULL,
                    notes=(
                        f"{variant.notes} Occasional expensive checkpoint for the full parity-style "
                        f"rotation loop. execution_path={execution_path}."
                    ),
                )
    return cases


def _write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")


def run_matrix(args: argparse.Namespace) -> int:
    if not has_visible_gpu():
        raise SystemExit("No visible NVIDIA GPU found for the cupy-rsoxs optimization study.")

    run_label = args.label or _timestamp()
    resident_modes = _parse_resident_modes(args.resident_modes)
    execution_paths = _parse_execution_paths(args.execution_paths)
    size_labels = _parse_csv_labels(args.size_labels)
    unknown = [label for label in size_labels if label not in SIZE_SPECS]
    if unknown:
        raise SystemExit(f"Unsupported size_labels entries: {unknown!r}")
    timing_segments = _parse_timing_segments(args.timing_segments)
    no_rotation_energy_counts = _parse_positive_int_csv(args.no_rotation_energy_counts)
    rotation_specs = _parse_rotation_specs(args.rotation_specs)
    energy_lists = _parse_energy_lists(args.energy_lists)

    run_dir = OUT_ROOT / run_label
    output_dir = run_dir / "cases"
    summary: dict[str, Any] = {
        "label": run_label,
        "created_utc": _timestamp(),
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "timing_boundary": PRIMARY_TIMING_BOUNDARY,
        "timing_segments": list(timing_segments),
        "resident_modes": list(resident_modes),
        "execution_paths": list(execution_paths),
        "size_labels": list(size_labels),
        "include_triple_no_rotation": bool(args.include_triple_no_rotation),
        "include_triple_limited": bool(args.include_triple_limited),
        "no_rotation_energy_counts": list(no_rotation_energy_counts),
        "rotation_specs": [list(spec) for spec in rotation_specs],
        "explicit_energy_lists": [list(energies) for energies in energy_lists],
        "segment_g_status": "Reserved for future export timing. Not recorded in this pass.",
        "timing_cases": {},
    }

    run_dir.mkdir(parents=True, exist_ok=True)

    print("Running cupy-rsoxs timing cases...", flush=True)
    for case in _timing_cases(
        resident_modes=resident_modes,
        execution_paths=execution_paths,
        size_labels=size_labels,
        timing_segments=timing_segments,
        include_triple_no_rotation=args.include_triple_no_rotation,
        include_triple_limited=args.include_triple_limited,
        include_full_small_check=args.include_full_small_check,
        no_rotation_energy_counts=no_rotation_energy_counts,
        rotation_specs=rotation_specs,
        energy_lists=energy_lists,
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
            "Default run is the small host-resident single-energy no-rotation lane. "
            "Primary timing starts at Morphology(...) construction and ends after "
            "synchronized run(return_xarray=False) completion."
        )
    )
    parser.add_argument("--label", default=None, help="Output subdirectory label under test-reports.")
    parser.add_argument(
        "--resident-modes",
        default="host",
        help="Comma-separated resident-mode variants to run. Supported values: host,device.",
    )
    parser.add_argument(
        "--execution-paths",
        default="tensor_coeff",
        help=(
            "Comma-separated cupy-rsoxs execution paths to run. "
            "Supported values: tensor_coeff,direct_polarization,nt_polarization "
            "plus aliases default,tensor,direct,nt."
        ),
    )
    parser.add_argument(
        "--size-labels",
        default="small",
        help="Comma-separated subset of size labels to run, for example 'small,medium'.",
    )
    parser.add_argument(
        "--timing-segments",
        default="all",
        help=(
            "Comma-separated timing segments to record, or 'all'. "
            "Supported segments: A1,A2,B,C,D,E,F. Alias 'A' expands to A1,A2."
        ),
    )
    parser.add_argument(
        "--include-triple-no-rotation",
        action="store_true",
        help="Include the no-rotation three-energy CoreShell lane for the selected resident-mode variants.",
    )
    parser.add_argument(
        "--include-triple-limited",
        action="store_true",
        help="Include the limited-rotation three-energy CoreShell lane for the selected resident-mode variants.",
    )
    parser.add_argument(
        "--no-rotation-energy-counts",
        default="",
        help=(
            "Optional comma-separated extra no-rotation energy counts for development sweeps, "
            "for example '2,4,8'. Energies are selected as a centered contiguous subset "
            "from the CoreShell optics table."
        ),
    )
    parser.add_argument(
        "--rotation-specs",
        default="",
        help=(
            "Optional comma-separated explicit rotation specs in start:increment:end order, "
            "for example '0:15:165,0:5:165'. Each spec maps to "
            "EAngleRotation=[StartAngle, IncrementAngle, EndAngle]."
        ),
    )
    parser.add_argument(
        "--energy-lists",
        default="",
        help=(
            "Optional comma-separated explicit energy groups. Quote this argument because each "
            "group uses '|' separators, for example '284.7|285.0|285.2,284.9|285.0|285.1|285.2'."
        ),
    )
    parser.add_argument(
        "--include-full-small-check",
        action="store_true",
        help="Include the expensive full-rotation small CoreShell case for the selected resident-mode variants.",
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
