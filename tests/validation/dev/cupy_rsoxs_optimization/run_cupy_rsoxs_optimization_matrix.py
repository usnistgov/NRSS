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

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def _query_physical_gpu_devices() -> tuple[str, ...]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return ()
    if completed.returncode != 0:
        return ()
    return tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())


def _bootstrap_single_gpu_visibility(argv: list[str]) -> dict[str, Any]:
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing is not None and existing.strip():
        return {
            "mode": "inherited",
            "requested_gpu_index": None,
            "cuda_visible_devices": existing.strip(),
        }

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-index", default="auto")
    args, _unknown = parser.parse_known_args(argv[1:])
    requested = str(args.gpu_index).strip() or "auto"

    if requested.lower() == "auto":
        devices = _query_physical_gpu_devices()
        if devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = devices[0]
            return {
                "mode": "auto_first_physical_gpu",
                "requested_gpu_index": "auto",
                "cuda_visible_devices": devices[0],
                "physical_gpu_devices": list(devices),
            }
        return {
            "mode": "auto_no_gpu_resolved",
            "requested_gpu_index": "auto",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "physical_gpu_devices": [],
        }

    os.environ["CUDA_VISIBLE_DEVICES"] = requested
    return {
        "mode": "explicit_gpu_index",
        "requested_gpu_index": requested,
        "cuda_visible_devices": requested,
    }


# Keep standalone timing runs single-GPU by default, but do not mutate
# CUDA visibility when this module is merely imported by another harness.
if __name__ == "__main__":
    _EARLY_GPU_BOOTSTRAP = _bootstrap_single_gpu_visibility(sys.argv)
else:
    _EARLY_GPU_BOOTSTRAP = None

from NRSS import SFieldMode
from NRSS.morphology import Material, Morphology, OpticalConstants
from NRSS.backends import (
    assess_array_for_backend_runtime,
    coerce_array_for_backend,
    normalize_backend_options,
    resolve_backend_runtime_contract,
)
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
ISOTROPIC_MATERIAL_REPRESENTATIONS = ("legacy_zero_array", "enum_contract")
CUDA_PREWARM_MODES = ("off", "before_prepare_inputs")


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
    isotropic_representation: str
    cuda_prewarm_mode: str
    resident_mode: str | None
    input_policy: str
    ownership_policy: str | None
    backend_options: dict[str, Any] | None = None
    timing_segments: tuple[str, ...] = ()
    create_cy_object: bool = True
    worker_warmup_runs: int = 0
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


def _parse_isotropic_material_representations(raw: str) -> tuple[str, ...]:
    cleaned = raw.strip().lower()
    if cleaned == "both":
        return ISOTROPIC_MATERIAL_REPRESENTATIONS

    requested = tuple(dict.fromkeys(part.strip().lower() for part in raw.split(",") if part.strip()))
    if not requested:
        raise SystemExit(
            "isotropic_material_representation must select at least one representation."
        )
    unknown = tuple(
        representation
        for representation in requested
        if representation not in ISOTROPIC_MATERIAL_REPRESENTATIONS
    )
    if unknown:
        raise SystemExit(
            "Unsupported isotropic material representation(s) "
            f"{unknown!r}. Valid values: {ISOTROPIC_MATERIAL_REPRESENTATIONS!r} or 'both'."
        )
    return requested


def _parse_cuda_prewarm_mode(raw: str) -> str:
    cleaned = str(raw).strip().lower()
    if cleaned not in CUDA_PREWARM_MODES:
        raise SystemExit(
            f"Unsupported cuda_prewarm {raw!r}. Valid values: {CUDA_PREWARM_MODES!r}."
        )
    return cleaned


def _parse_worker_warmup_runs(raw: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"worker_warmup_runs must be an integer, got {raw!r}.") from exc
    if value < 0:
        raise SystemExit(f"worker_warmup_runs must be non-negative, got {value!r}.")
    return value


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


def _isotropic_representation_label_suffix(
    isotropic_representation: str,
    *,
    include_suffix: bool,
) -> str:
    if not include_suffix and isotropic_representation == "legacy_zero_array":
        return ""
    return f"_{isotropic_representation}"


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
    isotropic_representation: str = "legacy_zero_array",
) -> dict[int, Material]:
    optics = _load_core_shell_optics(tuple(map(float, energies_ev)))
    fields = _convert_fields_namespace(_build_scaled_core_shell_fields(size_spec), field_namespace)

    if isotropic_representation == "legacy_zero_array":
        mat1_s = fields["mat1_s"]
        mat1_theta = fields["mat1_theta"]
        mat1_psi = fields["mat1_psi"]
        mat3_s = fields["mat3_s"]
        mat3_theta = fields["mat3_theta"]
        mat3_psi = fields["mat3_psi"]
    elif isotropic_representation == "enum_contract":
        mat1_s = SFieldMode.ISOTROPIC
        mat1_theta = None
        mat1_psi = None
        mat3_s = SFieldMode.ISOTROPIC
        mat3_theta = None
        mat3_psi = None
    else:
        raise AssertionError(
            f"Unsupported isotropic representation for the timing harness: {isotropic_representation!r}"
        )

    return {
        1: Material(
            materialID=1,
            Vfrac=fields["mat1_vfrac"],
            S=mat1_s,
            theta=mat1_theta,
            psi=mat1_psi,
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
            S=mat3_s,
            theta=mat3_theta,
            psi=mat3_psi,
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
    isotropic_representation: str = "legacy_zero_array",
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
        isotropic_representation=isotropic_representation,
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
            f"isotropic_representation={case.isotropic_representation} "
            f"cuda_prewarm_mode={case.cuda_prewarm_mode} "
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


def _strip_isotropic_representation_label_suffix(label: str, isotropic_representation: str) -> str:
    suffix = _isotropic_representation_label_suffix(
        isotropic_representation,
        include_suffix=True,
    )
    if label.endswith(suffix):
        return label[: -len(suffix)]
    return label


def _percent_change_vs_baseline(current: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return None
    return 100.0 * (current - baseline) / baseline


def _build_isotropic_representation_comparisons(
    timing_cases: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for label, result in timing_cases.items():
        isotropic_representation = str(result.get("isotropic_representation", ""))
        if isotropic_representation not in ISOTROPIC_MATERIAL_REPRESENTATIONS:
            continue
        base_label = _strip_isotropic_representation_label_suffix(label, isotropic_representation)
        grouped.setdefault(base_label, {})[isotropic_representation] = result

    comparisons: dict[str, dict[str, Any]] = {}
    for base_label in sorted(grouped):
        representations = grouped[base_label]
        legacy = representations.get("legacy_zero_array")
        enum = representations.get("enum_contract")
        if legacy is None or enum is None:
            continue

        comparison: dict[str, Any] = {
            "legacy_zero_array_label": legacy["label"],
            "enum_contract_label": enum["label"],
            "resident_mode": legacy.get("resident_mode"),
            "shape_label": legacy.get("shape_label"),
            "execution_path": dict(legacy.get("backend_options") or {}).get("execution_path"),
            "energies_ev": list(legacy.get("energies_ev", [])),
            "eangle_rotation": list(legacy.get("eangle_rotation", [])),
            "cuda_prewarm_mode": legacy.get("cuda_prewarm_requested_mode", "off"),
            "cuda_prewarm_applied_mode": legacy.get("cuda_prewarm_applied_mode", "off"),
        }

        if legacy.get("status") != "ok" or enum.get("status") != "ok":
            comparison["status"] = "comparison_unavailable"
            comparison["legacy_zero_array_status"] = legacy.get("status")
            comparison["enum_contract_status"] = enum.get("status")
            comparisons[base_label] = comparison
            continue

        legacy_primary = float(legacy["primary_seconds"])
        enum_primary = float(enum["primary_seconds"])
        comparison["status"] = "ok"
        comparison["primary_seconds"] = {
            "legacy_zero_array": legacy_primary,
            "enum_contract": enum_primary,
        }
        comparison["primary_delta_seconds"] = enum_primary - legacy_primary
        comparison["primary_percent_change_vs_legacy"] = _percent_change_vs_baseline(
            enum_primary,
            legacy_primary,
        )
        comparison["primary_speedup_factor_enum_vs_legacy"] = (
            None if enum_primary == 0.0 else legacy_primary / enum_primary
        )

        segment_comparisons: dict[str, dict[str, float | None]] = {}
        legacy_segments = legacy.get("segment_seconds", {})
        enum_segments = enum.get("segment_seconds", {})
        for segment in TIMING_SEGMENTS:
            legacy_seconds = legacy_segments.get(segment)
            enum_seconds = enum_segments.get(segment)
            if legacy_seconds is None and enum_seconds is None:
                continue

            segment_payload: dict[str, float | None] = {}
            if legacy_seconds is not None:
                segment_payload["legacy_zero_array"] = float(legacy_seconds)
            if enum_seconds is not None:
                segment_payload["enum_contract"] = float(enum_seconds)
            if legacy_seconds is not None and enum_seconds is not None:
                legacy_seconds_float = float(legacy_seconds)
                enum_seconds_float = float(enum_seconds)
                segment_payload["delta_seconds"] = enum_seconds_float - legacy_seconds_float
                segment_payload["percent_change_vs_legacy"] = _percent_change_vs_baseline(
                    enum_seconds_float,
                    legacy_seconds_float,
                )
            segment_comparisons[segment] = segment_payload
        comparison["segment_seconds"] = segment_comparisons
        comparisons[base_label] = comparison

    return comparisons


def _isotropic_representation_comparison_summary_line(
    comparison_label: str,
    comparison: dict[str, Any],
) -> str:
    if comparison.get("status") != "ok":
        return (
            f"{comparison_label}: isotropic comparison unavailable "
            f"(legacy={comparison.get('legacy_zero_array_status', 'missing')}, "
            f"enum={comparison.get('enum_contract_status', 'missing')})"
        )

    primary_seconds = comparison["primary_seconds"]
    legacy_primary = float(primary_seconds["legacy_zero_array"])
    enum_primary = float(primary_seconds["enum_contract"])
    delta = float(comparison["primary_delta_seconds"])
    percent_change = comparison.get("primary_percent_change_vs_legacy")
    percent_text = "n/a" if percent_change is None else f"{float(percent_change):+.1f}%"
    delta_text = f"{delta:+.3f}s"
    return (
        f"{comparison_label}: primary legacy {legacy_primary:.3f}s, "
        f"enum {enum_primary:.3f}s, delta {delta_text} ({percent_text})"
    )


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
            isotropic_representation=case.isotropic_representation,
        ),
        "phys_size": float(size_spec.phys_size_nm),
        "config": _default_morphology_config(case.energies_ev, case.eangle_rotation),
        "backend_options": case.backend_options,
    }


def _resolve_case_cuda_prewarm(case: BenchmarkCase) -> tuple[str, str | None]:
    if case.cuda_prewarm_mode == "off":
        return "off", "CUDA prewarm disabled."
    if case.cuda_prewarm_mode == "before_prepare_inputs":
        if case.resident_mode == "device":
            return (
                "redundant_device_prepare",
                "Device-resident field preparation already touches CuPy before primary_start.",
            )
        return "before_prepare_inputs", None
    raise AssertionError(f"Unsupported cuda_prewarm mode: {case.cuda_prewarm_mode!r}")


def _prewarm_cuda_runtime_for_host_staging(
    *,
    backend_options: dict[str, Any] | None = None,
) -> float:
    runtime_contract = resolve_backend_runtime_contract("cupy-rsoxs", backend_options)
    warm_host = np.zeros((1,), dtype=np.float32)
    started = time.perf_counter()
    plan = assess_array_for_backend_runtime(
        warm_host,
        backend_name="cupy-rsoxs",
        field_name="Vfrac",
        material_id=0,
        contract=runtime_contract,
    )
    warmed = coerce_array_for_backend(warm_host, plan)
    import cupy as cp

    cp.cuda.Stream.null.synchronize()
    del warmed
    return time.perf_counter() - started


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


def _run_case_once(
    *,
    case: BenchmarkCase,
    prepared_inputs: dict[str, Any],
    collect_timing: bool,
) -> dict[str, Any]:
    morphology = None
    backend_result = None
    try:
        start_time = time.perf_counter() if collect_timing else None
        morphology = _construct_morphology_for_timing_case(case, prepared_inputs)
        run_result: dict[str, Any] = {
            "resolved_backend_options": dict(morphology.backend_options),
            "kernel_backend_report": dict(getattr(morphology, "last_kernel_backend_report", {})),
            "kernel_preload_report": dict(getattr(morphology, "last_kernel_preload_report", {})),
        }
        if collect_timing:
            segment_seconds: dict[str, float] = {}
            if "A1" in case.timing_segments and start_time is not None:
                segment_seconds["A1"] = time.perf_counter() - start_time

            morphology._set_private_backend_timing_segments(case.timing_segments)

        backend_result = morphology.run(stdout=False, stderr=False, return_xarray=False)
        _synchronize_cupy_default_stream()

        if collect_timing:
            backend_timings = morphology.backend_timings
            if backend_timings:
                segment_seconds.update(backend_timings.get("segment_seconds", {}))

            run_result.update(
                primary_seconds=time.perf_counter() - start_time,
                segment_seconds={
                    segment: float(segment_seconds[segment])
                    for segment in TIMING_SEGMENTS
                    if segment in segment_seconds
                },
                backend_timing_details=backend_timings,
            )
        run_result["panel_shape"] = list(backend_result.to_backend_array().shape)
        run_result["kernel_backend_report"] = dict(
            getattr(morphology, "last_kernel_backend_report", {})
        )
        run_result["kernel_preload_report"] = dict(
            getattr(morphology, "last_kernel_preload_report", {})
        )
        return run_result
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
        del backend_result, morphology


def _worker_main(case_path: Path, result_path: Path) -> int:
    case = BenchmarkCase(**json.loads(case_path.read_text()))
    result: dict[str, Any] = {
        "label": case.label,
        "family": case.family,
        "backend": case.backend,
        "shape_label": case.shape_label,
        "resident_mode": case.resident_mode,
        "field_namespace": case.field_namespace,
        "isotropic_representation": case.isotropic_representation,
        "cuda_prewarm_requested_mode": case.cuda_prewarm_mode,
        "input_policy": case.input_policy,
        "ownership_policy": case.ownership_policy,
        "backend_options": dict(case.backend_options or {}),
        "energies_ev": list(case.energies_ev),
        "eangle_rotation": list(case.eangle_rotation),
        "timing_segments_requested": list(case.timing_segments),
        "timing_boundary": PRIMARY_TIMING_BOUNDARY,
        "note": _case_note(case),
        "worker_warmup_runs_requested": int(case.worker_warmup_runs),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_bootstrap": _EARLY_GPU_BOOTSTRAP,
        "status": "error",
    }

    if case.backend != "cupy-rsoxs":
        raise AssertionError("The optimization timing harness only supports backend='cupy-rsoxs'.")

    prepared_inputs = None
    morphology = None
    backend_result = None
    try:
        prewarm_mode, prewarm_note = _resolve_case_cuda_prewarm(case)
        result["cuda_prewarm_applied_mode"] = prewarm_mode
        if prewarm_note is not None:
            result["cuda_prewarm_note"] = prewarm_note
        if prewarm_mode == "before_prepare_inputs":
            result["cuda_prewarm_seconds"] = _prewarm_cuda_runtime_for_host_staging(
                backend_options=case.backend_options,
            )

        prepared_inputs = _prepare_core_shell_case_inputs(case)
        if case.field_namespace == "cupy":
            _synchronize_cupy_default_stream()

        warmup_seconds_total = 0.0
        warmup_runs_completed = 0
        for _ in range(case.worker_warmup_runs):
            warmup_started = time.perf_counter()
            warmup_result = _run_case_once(
                case=case,
                prepared_inputs=prepared_inputs,
                collect_timing=False,
            )
            warmup_seconds_total += time.perf_counter() - warmup_started
            warmup_runs_completed += 1
            if "resolved_backend_options" not in result:
                result["resolved_backend_options"] = warmup_result["resolved_backend_options"]
            del warmup_result

        result["worker_warmup_runs_completed"] = warmup_runs_completed
        if warmup_runs_completed:
            result["worker_warmup_seconds_total"] = warmup_seconds_total

        timed_result = _run_case_once(
            case=case,
            prepared_inputs=prepared_inputs,
            collect_timing=True,
        )
        result.update(timed_result)
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
    isotropic_representations: tuple[str, ...],
    cuda_prewarm_mode: str,
    execution_paths: tuple[str, ...],
    size_labels: tuple[str, ...],
    timing_segments: tuple[str, ...],
    include_triple_no_rotation: bool,
    include_triple_limited: bool,
    include_full_small_check: bool,
    no_rotation_energy_counts: tuple[int, ...],
    rotation_specs: tuple[tuple[float, float, float], ...],
    energy_lists: tuple[tuple[float, ...], ...],
    worker_warmup_runs: int,
) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    include_representation_in_label = (
        len(isotropic_representations) > 1
        or isotropic_representations[0] != "legacy_zero_array"
    )
    for mode in resident_modes:
        variant = RESIDENT_VARIANTS[mode]
        for isotropic_representation in isotropic_representations:
            representation_suffix = _isotropic_representation_label_suffix(
                isotropic_representation,
                include_suffix=include_representation_in_label,
            )
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
                            isotropic_representation=isotropic_representation,
                            cuda_prewarm_mode=cuda_prewarm_mode,
                            resident_mode=variant.resident_mode,
                            input_policy=variant.input_policy,
                            ownership_policy=variant.ownership_policy,
                            backend_options=backend_options,
                            timing_segments=timing_segments,
                            worker_warmup_runs=worker_warmup_runs,
                            notes=notes,
                        )
                    )

                for size_label in size_labels:
                    append_case(
                        label=(
                            f"core_shell_{size_label}_single_no_rotation_"
                            f"{variant.label_suffix}_{execution_path}{representation_suffix}"
                        ),
                        size_label=size_label,
                        energies_ev=CORE_SHELL_SINGLE_ENERGIES,
                        eangle_rotation=EANGLE_OFF,
                        notes=(
                            f"{variant.notes} Isotropic-material representation={isotropic_representation}. "
                            f"Primary no-rotation tuning lane. execution_path={execution_path}."
                        ),
                    )

                    if include_triple_no_rotation:
                        append_case(
                            label=(
                                f"core_shell_{size_label}_triple_no_rotation_"
                                f"{variant.label_suffix}_{execution_path}{representation_suffix}"
                            ),
                            size_label=size_label,
                            energies_ev=CORE_SHELL_TRIPLE_ENERGIES,
                            eangle_rotation=EANGLE_OFF,
                            notes=(
                                f"{variant.notes} Isotropic-material representation={isotropic_representation}. "
                                f"Secondary three-energy no-rotation checkpoint lane. "
                                f"execution_path={execution_path}."
                            ),
                        )

                    for energy_count in no_rotation_energy_counts:
                        if energy_count == 1:
                            continue
                        append_case(
                            label=(
                                f"core_shell_{size_label}_{energy_count}energy_no_rotation_"
                                f"{variant.label_suffix}_{execution_path}{representation_suffix}"
                            ),
                            size_label=size_label,
                            energies_ev=_centered_core_shell_energies(energy_count),
                            eangle_rotation=EANGLE_OFF,
                            notes=(
                                f"{variant.notes} Isotropic-material representation={isotropic_representation}. "
                                f"Development-only centered-energy no-rotation lane with {energy_count} "
                                f"energies. execution_path={execution_path}."
                            ),
                        )

                    if include_triple_limited:
                        append_case(
                            label=(
                                f"core_shell_{size_label}_triple_limited_rotation_"
                                f"{variant.label_suffix}_{execution_path}{representation_suffix}"
                            ),
                            size_label=size_label,
                            energies_ev=CORE_SHELL_TRIPLE_ENERGIES,
                            eangle_rotation=EANGLE_LIMITED,
                            notes=(
                                f"{variant.notes} Isotropic-material representation={isotropic_representation}. "
                                f"Secondary limited-EAngle checkpoint lane. "
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
                                f"{variant.label_suffix}_{execution_path}{representation_suffix}"
                            ),
                            size_label=size_label,
                            energies_ev=CORE_SHELL_SINGLE_ENERGIES,
                            eangle_rotation=eangle_rotation,
                            notes=(
                                f"{variant.notes} Isotropic-material representation={isotropic_representation}. "
                                f"Custom single-energy rotation lane with EAngleRotation={list(eangle_rotation)} "
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
                                f"{variant.label_suffix}_{execution_path}{representation_suffix}"
                            ),
                            size_label=size_label,
                            energies_ev=energies_ev,
                            eangle_rotation=EANGLE_OFF,
                            notes=(
                                f"{variant.notes} Isotropic-material representation={isotropic_representation}. "
                                f"Custom explicit-energy no-rotation lane with energies={list(energies_ev)}. "
                                f"execution_path={execution_path}."
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
                                    f"{variant.label_suffix}_{execution_path}{representation_suffix}"
                                ),
                                size_label=size_label,
                                energies_ev=energies_ev,
                                eangle_rotation=eangle_rotation,
                                notes=(
                                    f"{variant.notes} Isotropic-material representation={isotropic_representation}. "
                                    f"Custom explicit-energy plus custom rotation lane with "
                                    f"energies={list(energies_ev)} and EAngleRotation={list(eangle_rotation)} "
                                    f"([StartAngle, IncrementAngle, EndAngle]). "
                                    f"execution_path={execution_path}."
                                ),
                            )

                if include_full_small_check and "small" in size_labels:
                    append_case(
                        label=(
                            f"core_shell_small_triple_full_rotation_"
                            f"{variant.label_suffix}_{execution_path}{representation_suffix}"
                        ),
                        size_label="small",
                        energies_ev=CORE_SHELL_TRIPLE_ENERGIES,
                        eangle_rotation=EANGLE_FULL,
                        notes=(
                            f"{variant.notes} Isotropic-material representation={isotropic_representation}. "
                            f"Occasional expensive checkpoint for the full parity-style rotation loop. "
                            f"execution_path={execution_path}."
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
    isotropic_representations = _parse_isotropic_material_representations(
        args.isotropic_material_representation
    )
    cuda_prewarm_mode = _parse_cuda_prewarm_mode(args.cuda_prewarm)
    execution_paths = _parse_execution_paths(args.execution_paths)
    size_labels = _parse_csv_labels(args.size_labels)
    unknown = [label for label in size_labels if label not in SIZE_SPECS]
    if unknown:
        raise SystemExit(f"Unsupported size_labels entries: {unknown!r}")
    timing_segments = _parse_timing_segments(args.timing_segments)
    no_rotation_energy_counts = _parse_positive_int_csv(args.no_rotation_energy_counts)
    rotation_specs = _parse_rotation_specs(args.rotation_specs)
    energy_lists = _parse_energy_lists(args.energy_lists)
    worker_warmup_runs = _parse_worker_warmup_runs(args.worker_warmup_runs)

    run_dir = OUT_ROOT / run_label
    output_dir = run_dir / "cases"
    summary: dict[str, Any] = {
        "label": run_label,
        "created_utc": _timestamp(),
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_bootstrap": _EARLY_GPU_BOOTSTRAP,
        "timing_boundary": PRIMARY_TIMING_BOUNDARY,
        "timing_segments": list(timing_segments),
        "worker_warmup_runs": worker_warmup_runs,
        "resident_modes": list(resident_modes),
        "isotropic_material_representations": list(isotropic_representations),
        "cuda_prewarm_mode": cuda_prewarm_mode,
        "execution_paths": list(execution_paths),
        "size_labels": list(size_labels),
        "include_triple_no_rotation": bool(args.include_triple_no_rotation),
        "include_triple_limited": bool(args.include_triple_limited),
        "no_rotation_energy_counts": list(no_rotation_energy_counts),
        "rotation_specs": [list(spec) for spec in rotation_specs],
        "explicit_energy_lists": [list(energies) for energies in energy_lists],
        "segment_g_status": "Reserved for future export timing. Not recorded in this pass.",
        "timing_cases": {},
        "isotropic_representation_comparisons": {},
    }

    run_dir.mkdir(parents=True, exist_ok=True)

    print("Running cupy-rsoxs timing cases...", flush=True)
    for case in _timing_cases(
        resident_modes=resident_modes,
        isotropic_representations=isotropic_representations,
        cuda_prewarm_mode=cuda_prewarm_mode,
        execution_paths=execution_paths,
        size_labels=size_labels,
        timing_segments=timing_segments,
        include_triple_no_rotation=args.include_triple_no_rotation,
        include_triple_limited=args.include_triple_limited,
        include_full_small_check=args.include_full_small_check,
        no_rotation_energy_counts=no_rotation_energy_counts,
        rotation_specs=rotation_specs,
        energy_lists=energy_lists,
        worker_warmup_runs=worker_warmup_runs,
    ):
        result = _run_case_subprocess(case=case, output_dir=output_dir)
        summary["timing_cases"][case.label] = result
        print(_result_summary_line(result), flush=True)

    summary["isotropic_representation_comparisons"] = _build_isotropic_representation_comparisons(
        summary["timing_cases"]
    )
    for comparison_label, comparison in summary["isotropic_representation_comparisons"].items():
        print(
            _isotropic_representation_comparison_summary_line(comparison_label, comparison),
            flush=True,
        )

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
        "--gpu-index",
        default="auto",
        help=(
            "Single-GPU selection for standalone runs. Defaults to 'auto', which picks the first "
            "physical GPU when CUDA_VISIBLE_DEVICES is otherwise unset. Imported orchestrators "
            "should pin GPUs via CUDA_VISIBLE_DEVICES instead."
        ),
    )
    parser.add_argument(
        "--resident-modes",
        default="host",
        help="Comma-separated resident-mode variants to run. Supported values: host,device.",
    )
    parser.add_argument(
        "--isotropic-material-representation",
        default="legacy_zero_array",
        help=(
            "Isotropic-material representation to use for the CoreShell core/matrix materials. "
            "Supported values: legacy_zero_array, enum_contract, or both."
        ),
    )
    parser.add_argument(
        "--cuda-prewarm",
        default="off",
        help=(
            "Optional untimed CUDA prewarm mode for the dev harness. "
            "Supported values: off or before_prepare_inputs."
        ),
    )
    parser.add_argument(
        "--execution-paths",
        default="tensor_coeff",
        help=(
            "Comma-separated cupy-rsoxs execution paths to run. "
            "Supported values: tensor_coeff,direct_polarization "
            "plus aliases default,tensor,direct."
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
        "--worker-warmup-runs",
        type=int,
        default=0,
        help=(
            "Development-only fully-hot mode. Run this many untimed identical warm-up passes "
            "inside each worker subprocess before the timed boundary. Default: 0."
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
