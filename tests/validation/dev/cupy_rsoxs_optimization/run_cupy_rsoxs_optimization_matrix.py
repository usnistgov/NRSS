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
OPTICAL_CONSTANTS_DIR = REPO_ROOT / "tests" / "validation" / "data" / "optical_constants"

CORE_SHELL_SINGLE_ENERGIES = (285.0,)
CORE_SHELL_TRIPLE_ENERGIES = (284.7, 285.0, 285.2)
SPHERE_SINGLE_ENERGIES = (285.0,)
SPHERE_TRIPLE_ENERGIES = (284.7, 285.0, 285.2)
EANGLE_OFF = (0.0, 0.0, 0.0)
EANGLE_LIMITED = (0.0, 15.0, 165.0)
EANGLE_FULL = (0.0, 1.0, 360.0)

CORE_RADIUS_VOX = 4.0
SHELL_THICKNESS_VOX = 2.94
PHI_ISO = 0.46
DECAY_ORDER = 0.42
CENTER_Z_VOX = 15.0

SPHERE_SHAPE = (128, 128, 128)
SPHERE_DIAMETER_NM = 70.0
SPHERE_PHYS_SIZE_NM = 1.0
SPHERE_LOW_SYM_THETA = np.float32(3.0 * math.pi / 10.0)
SPHERE_LOW_SYM_PSI = np.float32(math.pi / 11.0)
SPHERE_LOW_SYM_S = np.float32(0.85)

SUMMARY_NAME = "summary.json"


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


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _cupy_memory_snapshot() -> dict[str, float] | None:
    try:
        import cupy as cp
    except Exception:
        return None

    try:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        pinned = cp.get_default_pinned_memory_pool()
        return {
            "free_bytes": float(free_bytes),
            "total_bytes": float(total_bytes),
            "pool_used_bytes": float(pool.used_bytes()),
            "pool_total_bytes": float(pool.total_bytes()),
            "pinned_free_blocks": float(pinned.n_free_blocks()),
        }
    except Exception:
        return None


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


@lru_cache(maxsize=None)
def _load_peolig_reference() -> np.ndarray:
    data = np.loadtxt(OPTICAL_CONSTANTS_DIR / "PEOlig2018.txt", skiprows=1)
    order = np.argsort(data[:, 6])
    return data[order]


@lru_cache(maxsize=None)
def _load_peolig_optics(energies_ev: tuple[float, ...]) -> OpticalConstants:
    data = _load_peolig_reference()
    ref_energy = data[:, 6]
    delta_para = np.interp(energies_ev, ref_energy, data[:, 2])
    delta_perp = np.interp(energies_ev, ref_energy, data[:, 3])
    beta_para = np.interp(energies_ev, ref_energy, data[:, 0])
    beta_perp = np.interp(energies_ev, ref_energy, data[:, 1])
    opt_constants = {
        float(energy): [
            float(d_para),
            float(b_para),
            float(d_perp),
            float(b_perp),
        ]
        for energy, d_para, b_para, d_perp, b_perp in zip(
            energies_ev,
            delta_para,
            beta_para,
            delta_perp,
            beta_perp,
        )
    }
    return OpticalConstants(list(map(float, energies_ev)), opt_constants, name="PEOlig2018")


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
    optics = _load_core_shell_optics(tuple(map(float, energies_ev)))
    fields = _convert_fields_namespace(_build_scaled_core_shell_fields(size_spec), field_namespace)

    materials = {
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

    config = {
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


def _sphere_mask(shape: tuple[int, int, int], phys_size_nm: float, diameter_nm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nz, ny, nx = shape
    radius_vox = np.float32(diameter_nm / (2.0 * phys_size_nm))
    cz = np.float32((nz - 1) / 2.0)
    cy = np.float32((ny - 1) / 2.0)
    cx = np.float32((nx - 1) / 2.0)

    zz, yy, xx = np.indices(shape, dtype=np.float32)
    dz = zz - cz
    dy = yy - cy
    dx = xx - cx
    dist2 = dx * dx + dy * dy + dz * dz
    sphere = dist2 <= np.float32(radius_vox * radius_vox)
    return sphere, dx, dy, dz


def _build_sphere_fields(
    *,
    shape: tuple[int, int, int],
    phys_size_nm: float,
    diameter_nm: float,
    orientation_mode: str,
) -> dict[str, np.ndarray]:
    sphere, dx, dy, dz = _sphere_mask(shape, phys_size_nm, diameter_nm)
    sphere_f = sphere.astype(np.float32)
    vacuum_f = (1.0 - sphere_f).astype(np.float32, copy=False)
    zeros = np.zeros(shape, dtype=np.float32)

    if orientation_mode == "radial":
        theta = np.arctan2(
            np.sqrt(dx * dx + dy * dy, dtype=np.float32),
            dz,
        ).astype(np.float32, copy=False)
        psi = np.arctan2(dy, dx).astype(np.float32, copy=False)
        s = sphere_f
    elif orientation_mode == "low_symmetry":
        theta = np.where(sphere, SPHERE_LOW_SYM_THETA, np.float32(0.0)).astype(np.float32, copy=False)
        psi = np.where(sphere, SPHERE_LOW_SYM_PSI, np.float32(0.0)).astype(np.float32, copy=False)
        s = np.where(sphere, SPHERE_LOW_SYM_S, np.float32(0.0)).astype(np.float32, copy=False)
    else:
        raise AssertionError(f"Unsupported sphere orientation mode: {orientation_mode}")

    return {
        "mat1_vfrac": sphere_f,
        "mat1_s": s,
        "mat1_theta": theta,
        "mat1_psi": psi,
        "mat2_vfrac": vacuum_f,
        "mat2_s": zeros.copy(),
        "mat2_theta": zeros.copy(),
        "mat2_psi": zeros.copy(),
    }


def build_sphere_morphology(
    *,
    energies_ev: tuple[float, ...],
    eangle_rotation: tuple[float, float, float],
    orientation_mode: str,
    backend: str,
    field_namespace: str,
    input_policy: str,
    ownership_policy: str | None,
    create_cy_object: bool,
) -> Morphology:
    sphere_optics = _load_peolig_optics(tuple(map(float, energies_ev)))
    vacuum_optics = OpticalConstants(list(map(float, energies_ev)), name="vacuum")
    fields = _convert_fields_namespace(
        _build_sphere_fields(
            shape=SPHERE_SHAPE,
            phys_size_nm=SPHERE_PHYS_SIZE_NM,
            diameter_nm=SPHERE_DIAMETER_NM,
            orientation_mode=orientation_mode,
        ),
        field_namespace,
    )

    materials = {
        1: Material(
            materialID=1,
            Vfrac=fields["mat1_vfrac"],
            S=fields["mat1_s"],
            theta=fields["mat1_theta"],
            psi=fields["mat1_psi"],
            energies=list(map(float, energies_ev)),
            opt_constants=sphere_optics.opt_constants,
            name=f"sphere_{orientation_mode}",
        ),
        2: Material(
            materialID=2,
            Vfrac=fields["mat2_vfrac"],
            S=fields["mat2_s"],
            theta=fields["mat2_theta"],
            psi=fields["mat2_psi"],
            energies=list(map(float, energies_ev)),
            opt_constants=vacuum_optics.opt_constants,
            name="vacuum",
        ),
    }

    config = {
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

    morph = Morphology(
        2,
        materials=materials,
        PhysSize=float(SPHERE_PHYS_SIZE_NM),
        config=config,
        create_cy_object=create_cy_object,
        backend=backend,
        input_policy=input_policy,
        ownership_policy=ownership_policy,
    )
    morph.check_materials(quiet=True)
    return morph


def _build_morphology_for_case(case: BenchmarkCase) -> Morphology:
    if case.family == "core_shell":
        return build_scaled_core_shell_morphology(
            size_spec=SIZE_SPECS[case.shape_label],
            energies_ev=case.energies_ev,
            eangle_rotation=case.eangle_rotation,
            backend=case.backend,
            field_namespace=case.field_namespace,
            input_policy=case.input_policy,
            ownership_policy=case.ownership_policy,
            create_cy_object=case.create_cy_object,
        )

    if case.family == "sphere_radial":
        return build_sphere_morphology(
            energies_ev=case.energies_ev,
            eangle_rotation=case.eangle_rotation,
            orientation_mode="radial",
            backend=case.backend,
            field_namespace=case.field_namespace,
            input_policy=case.input_policy,
            ownership_policy=case.ownership_policy,
            create_cy_object=case.create_cy_object,
        )

    if case.family == "sphere_low_symmetry":
        return build_sphere_morphology(
            energies_ev=case.energies_ev,
            eangle_rotation=case.eangle_rotation,
            orientation_mode="low_symmetry",
            backend=case.backend,
            field_namespace=case.field_namespace,
            input_policy=case.input_policy,
            ownership_policy=case.ownership_policy,
            create_cy_object=case.create_cy_object,
        )

    raise AssertionError(f"Unsupported benchmark family: {case.family}")


def _save_scattering_npz(scattering, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        data=np.asarray(scattering.values, dtype=np.float32),
        energy=np.asarray(scattering.coords["energy"].values, dtype=np.float64),
        qy=np.asarray(scattering.coords["qy"].values, dtype=np.float64),
        qx=np.asarray(scattering.coords["qx"].values, dtype=np.float64),
    )


def _axis_asymmetry(data: np.ndarray, qy: np.ndarray, qx: np.ndarray) -> np.ndarray:
    qx_grid, qy_grid = np.meshgrid(qx, qy)
    q = np.sqrt(qx_grid * qx_grid + qy_grid * qy_grid)
    band_width = 0.05
    q_min = 0.05
    q_max = 1.0
    horizontal_mask = np.logical_and.reduce(
        [np.abs(qy_grid) <= band_width, q >= q_min, q <= q_max, np.isfinite(q)]
    )
    vertical_mask = np.logical_and.reduce(
        [np.abs(qx_grid) <= band_width, q >= q_min, q <= q_max, np.isfinite(q)]
    )
    if int(np.count_nonzero(horizontal_mask)) == 0 or int(np.count_nonzero(vertical_mask)) == 0:
        return np.zeros((data.shape[0],), dtype=np.float64)

    asymmetry = np.zeros((data.shape[0],), dtype=np.float64)
    for idx in range(data.shape[0]):
        img = np.asarray(data[idx], dtype=np.float64)
        horiz = float(np.sum(img[horizontal_mask], dtype=np.float64))
        vert = float(np.sum(img[vertical_mask], dtype=np.float64))
        denom = max(abs(horiz) + abs(vert), 1.0e-30)
        asymmetry[idx] = (horiz - vert) / denom
    return asymmetry


def _compare_to_baseline(scattering, baseline_path: Path) -> dict[str, Any]:
    baseline = np.load(baseline_path)
    current = np.asarray(scattering.values, dtype=np.float64)
    reference = np.asarray(baseline["data"], dtype=np.float64)
    if current.shape != reference.shape:
        raise AssertionError(
            f"Shape mismatch against baseline {baseline_path.name}: "
            f"{current.shape!r} != {reference.shape!r}"
        )

    qy = np.asarray(scattering.coords["qy"].values, dtype=np.float64)
    qx = np.asarray(scattering.coords["qx"].values, dtype=np.float64)
    ref_qy = np.asarray(baseline["qy"], dtype=np.float64)
    ref_qx = np.asarray(baseline["qx"], dtype=np.float64)
    if not (np.allclose(qy, ref_qy) and np.allclose(qx, ref_qx)):
        raise AssertionError(f"Detector coordinates drifted relative to baseline {baseline_path.name}.")

    finite = np.isfinite(current) & np.isfinite(reference)
    if not np.any(finite):
        raise AssertionError(f"No overlapping finite detector values found against {baseline_path.name}.")

    diff = current[finite] - reference[finite]
    ref_vals = reference[finite]
    rmse = float(np.sqrt(np.mean(diff * diff, dtype=np.float64)))
    ref_rms = float(np.sqrt(np.mean(ref_vals * ref_vals, dtype=np.float64)))
    max_abs_diff = float(np.max(np.abs(diff)))
    mean_abs_diff = float(np.mean(np.abs(diff), dtype=np.float64))

    correlation = 1.0
    if diff.size > 1:
        correlation = float(np.corrcoef(current[finite], reference[finite])[0, 1])

    total_current = np.nansum(current, axis=(1, 2), dtype=np.float64)
    total_reference = np.nansum(reference, axis=(1, 2), dtype=np.float64)
    total_rel_err = np.max(
        np.abs(total_current - total_reference) / np.maximum(np.abs(total_reference), 1.0e-30)
    )

    axis_current = _axis_asymmetry(current, qy, qx)
    axis_reference = _axis_asymmetry(reference, ref_qy, ref_qx)

    return {
        "baseline_path": str(baseline_path),
        "finite_fraction": float(np.count_nonzero(finite)) / float(current.size),
        "rmse": rmse,
        "relative_rmse": float(rmse / max(ref_rms, 1.0e-30)),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "correlation": correlation,
        "max_total_intensity_rel_err": float(total_rel_err),
        "max_axis_asymmetry_abs_diff": float(np.max(np.abs(axis_current - axis_reference))),
    }


def _case_note(case: BenchmarkCase) -> str:
    size_spec = SIZE_SPECS.get(case.shape_label)
    if case.family == "core_shell":
        return (
            f"CoreShell {case.shape_label} {size_spec.shape} PhysSize={size_spec.phys_size_nm} "
            f"EAngleRotation={list(case.eangle_rotation)} energies={list(case.energies_ev)}"
        )
    return (
        f"{case.family} shape={SPHERE_SHAPE} PhysSize={SPHERE_PHYS_SIZE_NM} "
        f"EAngleRotation={list(case.eangle_rotation)} energies={list(case.energies_ev)}"
    )


def _result_summary_line(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return f"{result['label']}: {result.get('status')} ({result.get('error_type', 'unknown')})"
    return (
        f"{result['label']}: build {result['build_seconds']:.3f}s, "
        f"run {result['run_seconds']:.3f}s, export {result['export_seconds']:.3f}s, "
        f"workflow {result['workflow_seconds']:.3f}s"
    )


def _worker_main(case_path: Path, result_path: Path, baseline_path: Path | None, scattering_path: Path | None) -> int:
    started = time.perf_counter()
    case = BenchmarkCase(**json.loads(case_path.read_text()))
    result: dict[str, Any] = {
        "label": case.label,
        "family": case.family,
        "backend": case.backend,
        "shape_label": case.shape_label,
        "energies_ev": list(case.energies_ev),
        "eangle_rotation": list(case.eangle_rotation),
        "note": _case_note(case),
        "status": "error",
    }

    morphology = None
    backend_result = None
    scattering = None
    try:
        result["memory_before_build"] = _cupy_memory_snapshot()
        build_start = time.perf_counter()
        morphology = _build_morphology_for_case(case)
        result["build_seconds"] = time.perf_counter() - build_start
        result["memory_after_build"] = _cupy_memory_snapshot()

        run_start = time.perf_counter()
        backend_result = morphology.run(stdout=False, stderr=False, return_xarray=False)
        result["run_seconds"] = time.perf_counter() - run_start
        result["memory_after_run"] = _cupy_memory_snapshot()
        result["backend_timings"] = morphology.backend_timings

        export_start = time.perf_counter()
        if case.backend == "cyrsoxs":
            scattering = morphology.scattering_to_xarray(return_xarray=True)
        else:
            scattering = backend_result.to_xarray()
        result["export_seconds"] = time.perf_counter() - export_start
        result["memory_after_export"] = _cupy_memory_snapshot()

        result["panel_shape"] = list(scattering.shape)
        result["workflow_seconds"] = time.perf_counter() - started
        result["status"] = "ok"

        if scattering_path is not None:
            _save_scattering_npz(scattering, scattering_path)
            result["scattering_path"] = str(scattering_path)

        if baseline_path is not None:
            result["validation_metrics"] = _compare_to_baseline(scattering, baseline_path)

    except BaseException as exc:  # noqa: BLE001 - worker must serialize failures
        result["workflow_seconds"] = time.perf_counter() - started
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
        del scattering, backend_result, morphology
        release_runtime_memory()
        result["memory_after_release"] = _cupy_memory_snapshot()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2, default=_json_default) + "\n")
    return 0


def _run_case_subprocess(
    *,
    case: BenchmarkCase,
    output_dir: Path,
    baseline_path: Path | None = None,
    scattering_path: Path | None = None,
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
        if baseline_path is not None:
            cmd.extend(["--worker-baseline-path", str(baseline_path)])
        if scattering_path is not None:
            cmd.extend(["--worker-scattering-path", str(scattering_path)])

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


def _timing_cases(include_full_small_check: bool) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for size_label in ("small", "medium", "large"):
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
                notes="Primary limited-EAngle tuning lane.",
            )
        )

    if include_full_small_check:
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
                notes="Occasional expensive checkpoint for the full parity-style rotation loop.",
            )
        )
    return cases


def _cyrsoxs_reference_timing_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            label=f"core_shell_{size_label}_triple_limited_rotation_cyrsoxs_reference",
            family="core_shell",
            backend="cyrsoxs",
            shape_label=size_label,
            energies_ev=CORE_SHELL_TRIPLE_ENERGIES,
            eangle_rotation=EANGLE_LIMITED,
            field_namespace="numpy",
            input_policy="coerce",
            ownership_policy=None,
            notes="One reference cyrsoxs timing per size.",
        )
        for size_label in ("small", "medium", "large")
    ]


def _validation_baseline_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            label="sphere_radial_single_no_rotation_cyrsoxs_baseline",
            family="sphere_radial",
            backend="cyrsoxs",
            shape_label="n/a",
            energies_ev=SPHERE_SINGLE_ENERGIES,
            eangle_rotation=EANGLE_OFF,
            field_namespace="numpy",
            input_policy="coerce",
            ownership_policy=None,
        ),
        BenchmarkCase(
            label="sphere_low_symmetry_triple_limited_rotation_cyrsoxs_baseline",
            family="sphere_low_symmetry",
            backend="cyrsoxs",
            shape_label="n/a",
            energies_ev=SPHERE_TRIPLE_ENERGIES,
            eangle_rotation=EANGLE_LIMITED,
            field_namespace="numpy",
            input_policy="coerce",
            ownership_policy=None,
        ),
    ]


def _validation_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            label="sphere_radial_single_no_rotation_cupy_borrow",
            family="sphere_radial",
            backend="cupy-rsoxs",
            shape_label="n/a",
            energies_ev=SPHERE_SINGLE_ENERGIES,
            eangle_rotation=EANGLE_OFF,
            field_namespace="cupy",
            input_policy="strict",
            ownership_policy="borrow",
            validation_baseline_name="sphere_radial_single_no_rotation_cyrsoxs_baseline",
        ),
        BenchmarkCase(
            label="sphere_low_symmetry_triple_limited_rotation_cupy_borrow",
            family="sphere_low_symmetry",
            backend="cupy-rsoxs",
            shape_label="n/a",
            energies_ev=SPHERE_TRIPLE_ENERGIES,
            eangle_rotation=EANGLE_LIMITED,
            field_namespace="cupy",
            input_policy="strict",
            ownership_policy="borrow",
            validation_baseline_name="sphere_low_symmetry_triple_limited_rotation_cyrsoxs_baseline",
        ),
    ]


def _write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")


def run_matrix(args: argparse.Namespace) -> int:
    if not has_visible_gpu():
        raise SystemExit("No visible NVIDIA GPU found for the cupy-rsoxs optimization study.")

    run_label = args.label or _timestamp()
    run_dir = OUT_ROOT / run_label
    baselines_dir = Path(args.baseline_dir) if args.baseline_dir else run_dir / "baselines"
    output_dir = run_dir / "cases"
    summary: dict[str, Any] = {
        "label": run_label,
        "created_utc": _timestamp(),
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "baseline_dir": str(baselines_dir),
        "timing_cases": {},
        "validation_cases": {},
        "cyrsoxs_reference_cases": {},
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    baselines_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh_validation_baselines:
        print("Refreshing cyrsoxs validation baselines...", flush=True)
        for case in _validation_baseline_cases():
            baseline_path = baselines_dir / f"{case.label}.npz"
            result = _run_case_subprocess(
                case=case,
                output_dir=output_dir,
                baseline_path=None,
                scattering_path=baseline_path,
            )
            summary["cyrsoxs_reference_cases"][case.label] = result
            print(_result_summary_line(result), flush=True)

    if args.include_cyrsoxs_timing:
        print("Running one cyrsoxs timing case per CoreShell size...", flush=True)
        for case in _cyrsoxs_reference_timing_cases():
            result = _run_case_subprocess(case=case, output_dir=output_dir)
            summary["cyrsoxs_reference_cases"][case.label] = result
            print(_result_summary_line(result), flush=True)

    print("Running cupy-rsoxs timing cases...", flush=True)
    for case in _timing_cases(include_full_small_check=args.include_full_small_check):
        result = _run_case_subprocess(case=case, output_dir=output_dir)
        summary["timing_cases"][case.label] = result
        print(_result_summary_line(result), flush=True)

    print("Running cupy-rsoxs validation cases...", flush=True)
    for case in _validation_cases():
        baseline_path = baselines_dir / f"{case.validation_baseline_name}.npz"
        if not baseline_path.exists():
            raise SystemExit(
                f"Validation baseline {baseline_path} is missing. "
                "Run with --refresh-validation-baselines first or provide --baseline-dir."
            )
        result = _run_case_subprocess(
            case=case,
            output_dir=output_dir,
            baseline_path=baseline_path,
        )
        summary["validation_cases"][case.label] = result
        print(_result_summary_line(result), flush=True)

    _write_summary(run_dir, summary)
    print(f"Wrote {run_dir / SUMMARY_NAME}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only timing and ad hoc validation matrix for the "
            "cupy-rsoxs optimization campaign."
        )
    )
    parser.add_argument("--label", default=None, help="Output subdirectory label under test-reports.")
    parser.add_argument(
        "--baseline-dir",
        default=None,
        help="Reuse an existing baseline directory instead of creating baselines under this run directory.",
    )
    parser.add_argument(
        "--refresh-validation-baselines",
        action="store_true",
        help="Regenerate the sphere validation baselines with cyrsoxs and save them as NPZ files.",
    )
    parser.add_argument(
        "--include-cyrsoxs-timing",
        action="store_true",
        help="Run one cyrsoxs CoreShell timing case per size on the limited-angle lane.",
    )
    parser.add_argument(
        "--include-full-small-check",
        action="store_true",
        help="Include the expensive full-rotation small CoreShell cupy case.",
    )
    parser.add_argument("--worker-case-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-baseline-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-scattering-path", default=None, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.worker_case_path:
        return _worker_main(
            case_path=Path(args.worker_case_path),
            result_path=Path(args.worker_result_path),
            baseline_path=Path(args.worker_baseline_path) if args.worker_baseline_path else None,
            scattering_path=Path(args.worker_scattering_path) if args.worker_scattering_path else None,
        )
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
