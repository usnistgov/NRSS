from __future__ import annotations

import gc
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import j1


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from NRSS.morphology import Material, Morphology


DEFAULT_CONFIG = {
    "CaseType": 0,
    "MorphologyType": 0,
    "Energies": [285.0],
    "EAngleRotation": [0.0, 0.0, 0.0],
    "RotMask": 1,
    "WindowingType": 0,
    "AlgorithmType": 0,
    "ReferenceFrame": 1,
    "EwaldsInterpolation": 1,
}


@dataclass(frozen=True)
class Bragg2DCase:
    case_id: str
    lattice_kind: str
    lattice_constant_nm: float
    particle_diameter_nm: float
    azimuth_deg: float
    shape: tuple[int, int, int]
    phys_size_nm: float = 1.0
    superresolution: int = 4
    energy_eV: float = 285.0


@dataclass(frozen=True)
class Bragg3DCase:
    case_id: str
    lattice_kind: str
    lattice_constant_nm: float
    particle_diameter_nm: float
    azimuth_deg: float
    tilt_x_deg: float
    tilt_y_deg: float
    shape: tuple[int, int, int]
    phys_size_nm: float = 1.0
    superresolution: int = 2
    energy_eV: float = 285.0


@lru_cache(maxsize=1)
def has_visible_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0 and "GPU " in result.stdout


def release_runtime_memory() -> None:
    gc.collect()
    try:
        import cupy as cp
    except ImportError:
        return

    for action in (
        lambda: cp.cuda.runtime.deviceSynchronize(),
        lambda: cp.get_default_memory_pool().free_all_blocks(),
        lambda: cp.get_default_pinned_memory_pool().free_all_blocks(),
    ):
        try:
            action()
        except Exception:
            pass


def _convert_fields_namespace(
    fields: dict[str, np.ndarray],
    field_namespace: str,
) -> dict[str, np.ndarray]:
    if field_namespace == "numpy":
        return fields
    if field_namespace != "cupy":
        raise AssertionError(f"Unsupported field namespace: {field_namespace}")

    cp = __import__("cupy")
    converted: dict[str, np.ndarray] = {}
    for key, value in fields.items():
        converted[key] = cp.ascontiguousarray(cp.asarray(value, dtype=cp.float32))
    return converted


def lattice_vectors_2d_nm(lattice_kind: str, lattice_constant_nm: float, azimuth_deg: float) -> np.ndarray:
    a = float(lattice_constant_nm)
    if lattice_kind == "square":
        base = np.asarray(
            [
                [a, 0.0],
                [0.0, a],
            ],
            dtype=np.float64,
        )
    elif lattice_kind == "hexagonal":
        base = np.asarray(
            [
                [a, 0.0],
                [0.5 * a, 0.5 * np.sqrt(3.0) * a],
            ],
            dtype=np.float64,
        )
    else:
        raise AssertionError(f"Unsupported 2D lattice kind: {lattice_kind}")

    phi = np.deg2rad(float(azimuth_deg))
    rot = np.asarray(
        [
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi), np.cos(phi)],
        ],
        dtype=np.float64,
    )
    return base @ rot.T


def reciprocal_vectors_nm_inv(primitive_vectors_nm: np.ndarray) -> np.ndarray:
    primitive_vectors_nm = np.asarray(primitive_vectors_nm, dtype=np.float64)
    return 2.0 * np.pi * np.linalg.inv(primitive_vectors_nm).T


def rotation_matrix_3d(azimuth_deg: float, tilt_x_deg: float, tilt_y_deg: float) -> np.ndarray:
    az = np.deg2rad(float(azimuth_deg))
    tx = np.deg2rad(float(tilt_x_deg))
    ty = np.deg2rad(float(tilt_y_deg))

    rz = np.asarray(
        [
            [np.cos(az), -np.sin(az), 0.0],
            [np.sin(az), np.cos(az), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rx = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(tx), -np.sin(tx)],
            [0.0, np.sin(tx), np.cos(tx)],
        ],
        dtype=np.float64,
    )
    ry = np.asarray(
        [
            [np.cos(ty), 0.0, np.sin(ty)],
            [0.0, 1.0, 0.0],
            [-np.sin(ty), 0.0, np.cos(ty)],
        ],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def lattice_vectors_3d_nm(
    lattice_kind: str,
    lattice_constant_nm: float,
    azimuth_deg: float,
    tilt_x_deg: float,
    tilt_y_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    a = float(lattice_constant_nm)
    if lattice_kind == "simple_cubic":
        base = np.asarray(
            [
                [a, 0.0, 0.0],
                [0.0, a, 0.0],
                [0.0, 0.0, a],
            ],
            dtype=np.float64,
        )
        basis_frac = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
    elif lattice_kind == "hcp":
        c = np.sqrt(8.0 / 3.0) * a
        base = np.asarray(
            [
                [a, 0.0, 0.0],
                [0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0],
                [0.0, 0.0, c],
            ],
            dtype=np.float64,
        )
        basis_frac = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0 / 3.0, 1.0 / 3.0, 0.5],
            ],
            dtype=np.float64,
        )
    else:
        raise AssertionError(f"Unsupported 3D lattice kind: {lattice_kind}")

    rot = rotation_matrix_3d(
        azimuth_deg=azimuth_deg,
        tilt_x_deg=tilt_x_deg,
        tilt_y_deg=tilt_y_deg,
    )
    return base @ rot.T, basis_frac


def disk_form_factor_amplitude(q: np.ndarray, diameter_nm: float) -> np.ndarray:
    radius_nm = 0.5 * float(diameter_nm)
    qr = np.asarray(q, dtype=np.float64) * radius_nm
    amp = np.ones_like(qr)
    nonzero = np.abs(qr) > 1e-12
    qr_nz = qr[nonzero]
    amp[nonzero] = 2.0 * j1(qr_nz) / qr_nz
    return amp


def sphere_form_factor_amplitude(q: np.ndarray, diameter_nm: float) -> np.ndarray:
    radius_nm = 0.5 * float(diameter_nm)
    qr = np.asarray(q, dtype=np.float64) * radius_nm
    amp = np.ones_like(qr)
    nonzero = np.abs(qr) > 1e-12
    qr_nz = qr[nonzero]
    amp[nonzero] = 3.0 * (np.sin(qr_nz) - qr_nz * np.cos(qr_nz)) / (qr_nz ** 3)
    return amp


def structure_factor_amplitude(hkl: np.ndarray, basis_frac: np.ndarray) -> complex:
    hkl = np.asarray(hkl, dtype=np.float64)
    basis_frac = np.asarray(basis_frac, dtype=np.float64)
    phase = 2.0 * np.pi * (basis_frac @ hkl)
    return complex(np.sum(np.exp(1j * phase)))


def flat_detector_k_nm_inv(energy_eV: float) -> float:
    wavelength_nm = 1239.84197 / float(energy_eV)
    return 2.0 * np.pi / wavelength_nm


def flat_detector_qz_nm_inv(qx: np.ndarray, qy: np.ndarray, energy_eV: float) -> np.ndarray:
    qx_arr, qy_arr = np.broadcast_arrays(
        np.asarray(qx, dtype=np.float64),
        np.asarray(qy, dtype=np.float64),
    )
    k = flat_detector_k_nm_inv(energy_eV)
    qperp2 = qx_arr * qx_arr + qy_arr * qy_arr
    val = k * k - qperp2
    qz = np.full(qx_arr.shape, np.nan, dtype=np.float64)
    valid = val >= 0.0
    qz[valid] = -k + np.sqrt(val[valid])
    return qz


def flat_detector_qmag_nm_inv(qx: np.ndarray, qy: np.ndarray, energy_eV: float) -> np.ndarray:
    qx_arr, qy_arr = np.broadcast_arrays(
        np.asarray(qx, dtype=np.float64),
        np.asarray(qy, dtype=np.float64),
    )
    qz = flat_detector_qz_nm_inv(qx=qx_arr, qy=qy_arr, energy_eV=energy_eV)
    qmag = np.full(qx_arr.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(qz)
    qmag[valid] = np.sqrt(qx_arr[valid] * qx_arr[valid] + qy_arr[valid] * qy_arr[valid] + qz[valid] * qz[valid])
    return qmag


def build_bragg_2d_case_morphology(
    case: Bragg2DCase,
    *,
    backend: str | None = None,
    backend_options: dict[str, Any] | None = None,
    resident_mode: str | None = None,
    input_policy: str = "coerce",
    ownership_policy: str | None = None,
    field_namespace: str = "numpy",
) -> tuple[Morphology, dict[str, object]]:
    if int(case.shape[0]) != 1:
        raise AssertionError(f"2D Bragg case expects z=1, got shape={case.shape!r}.")
    if int(case.superresolution) < 1:
        raise AssertionError("superresolution must be >= 1.")

    primitive_vectors_nm = lattice_vectors_2d_nm(
        lattice_kind=case.lattice_kind,
        lattice_constant_nm=case.lattice_constant_nm,
        azimuth_deg=case.azimuth_deg,
    )
    vfrac, centers_xy_nm = _disk_lattice_vfrac_2d(
        shape=case.shape,
        phys_size_nm=case.phys_size_nm,
        primitive_vectors_nm=primitive_vectors_nm,
        diameter_nm=case.particle_diameter_nm,
        superresolution=case.superresolution,
    )
    fields = _convert_fields_namespace(
        {
            "vfrac": vfrac,
            "vacuum": (1.0 - vfrac).astype(np.float32),
            "zeros": np.zeros(case.shape, dtype=np.float32),
        },
        field_namespace=field_namespace,
    )
    energies = [float(case.energy_eV)]

    lattice_material = Material(
        materialID=1,
        Vfrac=fields["vfrac"],
        S=fields["zeros"].copy(),
        theta=fields["zeros"].copy(),
        psi=fields["zeros"].copy(),
        energies=energies,
        opt_constants={float(case.energy_eV): [0.0, 2e-4, 0.0, 2e-4]},
        name=f"{case.lattice_kind}_disk_lattice",
    )
    vacuum_material = Material(
        materialID=2,
        Vfrac=fields["vacuum"],
        S=fields["zeros"].copy(),
        theta=fields["zeros"].copy(),
        psi=fields["zeros"].copy(),
        energies=energies,
        name="vacuum",
    )

    config = dict(DEFAULT_CONFIG)
    config["Energies"] = energies
    morph = Morphology(
        2,
        materials={1: lattice_material, 2: vacuum_material},
        PhysSize=float(case.phys_size_nm),
        config=config,
        create_cy_object=True,
        backend=backend,
        backend_options=backend_options,
        resident_mode=resident_mode,
        input_policy=input_policy,
        ownership_policy=ownership_policy,
    )

    return morph, {
        "primitive_vectors_nm": primitive_vectors_nm,
        "centers_xy_nm": centers_xy_nm,
        "disk_count": int(len(centers_xy_nm)),
    }


def build_bragg_3d_case_morphology(
    case: Bragg3DCase,
    *,
    backend: str | None = None,
    backend_options: dict[str, Any] | None = None,
    resident_mode: str | None = None,
    input_policy: str = "coerce",
    ownership_policy: str | None = None,
    field_namespace: str = "numpy",
) -> tuple[Morphology, dict[str, object]]:
    if min(int(n) for n in case.shape) < 2:
        raise AssertionError(f"3D Bragg case expects all dimensions >= 2, got shape={case.shape!r}.")
    if int(case.superresolution) < 1:
        raise AssertionError("superresolution must be >= 1.")

    primitive_vectors_nm, basis_frac = lattice_vectors_3d_nm(
        lattice_kind=case.lattice_kind,
        lattice_constant_nm=case.lattice_constant_nm,
        azimuth_deg=case.azimuth_deg,
        tilt_x_deg=case.tilt_x_deg,
        tilt_y_deg=case.tilt_y_deg,
    )
    vfrac, centers_xyz_nm = _sphere_lattice_vfrac_3d(
        shape=case.shape,
        phys_size_nm=case.phys_size_nm,
        primitive_vectors_nm=primitive_vectors_nm,
        basis_frac=basis_frac,
        diameter_nm=case.particle_diameter_nm,
        superresolution=case.superresolution,
    )
    fields = _convert_fields_namespace(
        {
            "vfrac": vfrac,
            "vacuum": (1.0 - vfrac).astype(np.float32),
            "zeros": np.zeros(case.shape, dtype=np.float32),
        },
        field_namespace=field_namespace,
    )
    energies = [float(case.energy_eV)]

    lattice_material = Material(
        materialID=1,
        Vfrac=fields["vfrac"],
        S=fields["zeros"].copy(),
        theta=fields["zeros"].copy(),
        psi=fields["zeros"].copy(),
        energies=energies,
        opt_constants={float(case.energy_eV): [0.0, 2e-4, 0.0, 2e-4]},
        name=f"{case.lattice_kind}_sphere_lattice",
    )
    vacuum_material = Material(
        materialID=2,
        Vfrac=fields["vacuum"],
        S=fields["zeros"].copy(),
        theta=fields["zeros"].copy(),
        psi=fields["zeros"].copy(),
        energies=energies,
        name="vacuum",
    )

    config = dict(DEFAULT_CONFIG)
    config["Energies"] = energies
    morph = Morphology(
        2,
        materials={1: lattice_material, 2: vacuum_material},
        PhysSize=float(case.phys_size_nm),
        config=config,
        create_cy_object=True,
        backend=backend,
        backend_options=backend_options,
        resident_mode=resident_mode,
        input_policy=input_policy,
        ownership_policy=ownership_policy,
    )

    return morph, {
        "primitive_vectors_nm": primitive_vectors_nm,
        "basis_frac": basis_frac,
        "centers_xyz_nm": centers_xyz_nm,
        "sphere_count": int(len(centers_xyz_nm)),
    }


def predict_bragg_spots_2d(
    primitive_vectors_nm: np.ndarray,
    diameter_nm: float,
    qx: np.ndarray,
    qy: np.ndarray,
    intensity_floor_ratio: float,
) -> list[dict[str, float]]:
    reciprocal_vectors = reciprocal_vectors_nm_inv(primitive_vectors_nm)
    qx = np.asarray(qx, dtype=np.float64)
    qy = np.asarray(qy, dtype=np.float64)
    q_margin_x = 2.0 * float(np.abs(qx[1] - qx[0]))
    q_margin_y = 2.0 * float(np.abs(qy[1] - qy[0]))
    qx_min = float(qx.min()) - q_margin_x
    qx_max = float(qx.max()) + q_margin_x
    qy_min = float(qy.min()) - q_margin_y
    qy_max = float(qy.max()) + q_margin_y
    q_max = float(np.hypot(np.max(np.abs(qx)), np.max(np.abs(qy))))

    min_step = float(np.min(np.linalg.norm(reciprocal_vectors, axis=1)))
    h_max = int(np.ceil(q_max / min_step)) + 2

    rows: list[dict[str, float]] = []
    strongest = 0.0
    for h0 in range(-h_max, h_max + 1):
        for h1 in range(-h_max, h_max + 1):
            if h0 == 0 and h1 == 0:
                continue
            h = np.asarray([h0, h1], dtype=np.float64)
            q_vec = h @ reciprocal_vectors
            qx0 = float(q_vec[0])
            qy0 = float(q_vec[1])
            if not (qx_min <= qx0 <= qx_max and qy_min <= qy0 <= qy_max):
                continue
            qmag = float(np.hypot(qx0, qy0))
            intensity = float(disk_form_factor_amplitude(np.asarray([qmag]), diameter_nm)[0] ** 2)
            strongest = max(strongest, intensity)
            rows.append(
                {
                    "h": int(h0),
                    "k": int(h1),
                    "qx": qx0,
                    "qy": qy0,
                    "qmag": qmag,
                    "predicted_intensity": intensity,
                }
            )

    floor = float(intensity_floor_ratio) * strongest
    filtered = [row for row in rows if row["predicted_intensity"] >= floor]
    filtered.sort(key=lambda row: (row["qmag"], row["qx"], row["qy"]))
    return filtered


def predict_bragg_spots_3d(
    primitive_vectors_nm: np.ndarray,
    basis_frac: np.ndarray,
    diameter_nm: float,
    qx: np.ndarray,
    qy: np.ndarray,
    energy_eV: float,
    intensity_floor_ratio: float,
    qz_tolerance: float,
) -> list[dict[str, float]]:
    reciprocal_vectors = reciprocal_vectors_nm_inv(primitive_vectors_nm)
    qx = np.asarray(qx, dtype=np.float64)
    qy = np.asarray(qy, dtype=np.float64)
    q_margin_x = 2.0 * float(np.abs(qx[1] - qx[0]))
    q_margin_y = 2.0 * float(np.abs(qy[1] - qy[0]))
    qx_min = float(qx.min()) - q_margin_x
    qx_max = float(qx.max()) + q_margin_x
    qy_min = float(qy.min()) - q_margin_y
    qy_max = float(qy.max()) + q_margin_y
    qmag_max = 2.0 * flat_detector_k_nm_inv(energy_eV)

    min_step = float(np.min(np.linalg.norm(reciprocal_vectors, axis=1)))
    h_max = int(np.ceil(qmag_max / min_step)) + 2

    strongest_visible = 0.0
    visible_rows: list[dict[str, float]] = []
    for h0 in range(-h_max, h_max + 1):
        for h1 in range(-h_max, h_max + 1):
            for h2 in range(-h_max, h_max + 1):
                if h0 == 0 and h1 == 0 and h2 == 0:
                    continue
                hkl = np.asarray([h0, h1, h2], dtype=np.float64)
                q_vec = hkl @ reciprocal_vectors
                qx0 = float(q_vec[0])
                qy0 = float(q_vec[1])
                qz0 = float(q_vec[2])
                if not (qx_min <= qx0 <= qx_max and qy_min <= qy0 <= qy_max):
                    continue
                if abs(qx0) <= q_margin_x and abs(qy0) <= q_margin_y:
                    continue

                qz_detector = float(flat_detector_qz_nm_inv(qx=np.asarray([qx0]), qy=np.asarray([qy0]), energy_eV=energy_eV)[0])
                if not np.isfinite(qz_detector):
                    continue

                qperp = float(np.hypot(qx0, qy0))
                qmag_lattice = float(np.sqrt(qx0 * qx0 + qy0 * qy0 + qz0 * qz0))
                qmag_detector = float(np.sqrt(qx0 * qx0 + qy0 * qy0 + qz_detector * qz_detector))
                structure_amp = structure_factor_amplitude(hkl=hkl, basis_frac=basis_frac)
                intensity = float(
                    sphere_form_factor_amplitude(np.asarray([qmag_detector]), diameter_nm)[0] ** 2
                    * abs(structure_amp) ** 2
                )
                if abs(qz0 - qz_detector) > float(qz_tolerance):
                    continue

                strongest_visible = max(strongest_visible, intensity)
                visible_rows.append(
                    {
                        "h": int(h0),
                        "k": int(h1),
                        "l": int(h2),
                        "qx": qx0,
                        "qy": qy0,
                        "qz": qz0,
                        "qz_detector": qz_detector,
                        "qmag": qperp,
                        "qmag_detector": qmag_detector,
                        "qmag_lattice": qmag_lattice,
                        "predicted_intensity": intensity,
                        "ewald_abs_dqz": abs(qz0 - qz_detector),
                    }
                )

    if strongest_visible <= 0.0:
        return []

    floor = float(intensity_floor_ratio) * strongest_visible
    filtered = [row for row in visible_rows if row["predicted_intensity"] >= floor]
    filtered.sort(key=lambda row: (row["qmag"], row["qx"], row["qy"], row["qz"]))
    return filtered


def radial_shells_from_spots(
    spots: list[dict[str, float]],
    q_merge_tolerance: float,
) -> np.ndarray:
    q_merge_tolerance = float(q_merge_tolerance)
    if q_merge_tolerance <= 0.0:
        raise AssertionError("q_merge_tolerance must be positive.")
    if not spots:
        return np.zeros(0, dtype=np.float64)

    shells: list[float] = []
    for qmag in sorted(float(spot["qmag"]) for spot in spots):
        if not shells or abs(qmag - shells[-1]) > q_merge_tolerance:
            shells.append(qmag)
    return np.asarray(shells, dtype=np.float64)


def _disk_lattice_vfrac_2d(
    shape: tuple[int, int, int],
    phys_size_nm: float,
    primitive_vectors_nm: np.ndarray,
    diameter_nm: float,
    superresolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    _, ny, nx = shape
    radius_vox = float(diameter_nm) / (2.0 * float(phys_size_nm))
    primitive_vectors_vox = np.asarray(primitive_vectors_nm, dtype=np.float64) / float(phys_size_nm)
    centers_xy_vox = _enumerate_centers_xy_vox(
        ny=ny,
        nx=nx,
        primitive_vectors_vox=primitive_vectors_vox,
        radius_vox=radius_vox,
    )
    disk_plane = np.zeros((ny, nx), dtype=np.float32)
    for center_x_vox, center_y_vox in centers_xy_vox:
        _stamp_disk_local(
            disk_plane=disk_plane,
            center_x_vox=float(center_x_vox),
            center_y_vox=float(center_y_vox),
            radius_vox=radius_vox,
            superresolution=int(superresolution),
        )

    vfrac = np.zeros(shape, dtype=np.float32)
    vfrac[0, :, :] = disk_plane
    centers_xy_nm = centers_xy_vox.astype(np.float64) * float(phys_size_nm)
    return vfrac, centers_xy_nm


def _sphere_lattice_vfrac_3d(
    shape: tuple[int, int, int],
    phys_size_nm: float,
    primitive_vectors_nm: np.ndarray,
    basis_frac: np.ndarray,
    diameter_nm: float,
    superresolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    radius_vox = float(diameter_nm) / (2.0 * float(phys_size_nm))
    primitive_vectors_vox = np.asarray(primitive_vectors_nm, dtype=np.float64) / float(phys_size_nm)
    basis_offsets_vox = np.asarray(basis_frac, dtype=np.float64) @ primitive_vectors_vox
    centers_xyz_vox = _enumerate_centers_xyz_vox(
        shape=shape,
        primitive_vectors_vox=primitive_vectors_vox,
        basis_offsets_vox=basis_offsets_vox,
        radius_vox=radius_vox,
    )

    vfrac = np.zeros(shape, dtype=np.float32)
    for center_x_vox, center_y_vox, center_z_vox in centers_xyz_vox:
        _stamp_sphere_local(
            vfrac=vfrac,
            center_x_vox=float(center_x_vox),
            center_y_vox=float(center_y_vox),
            center_z_vox=float(center_z_vox),
            radius_vox=radius_vox,
            superresolution=int(superresolution),
        )

    centers_xyz_nm = centers_xyz_vox.astype(np.float64) * float(phys_size_nm)
    return vfrac, centers_xyz_nm


def _enumerate_centers_xy_vox(
    ny: int,
    nx: int,
    primitive_vectors_vox: np.ndarray,
    radius_vox: float,
) -> np.ndarray:
    half_x = 0.5 * float(nx - 1)
    half_y = 0.5 * float(ny - 1)
    extent = np.hypot(half_x + radius_vox, half_y + radius_vox)
    min_step = float(np.min(np.linalg.norm(primitive_vectors_vox, axis=1)))
    integer_limit = int(np.ceil(extent / min_step)) + 2
    idx = np.arange(-integer_limit, integer_limit + 1, dtype=np.int32)
    h0, h1 = np.meshgrid(idx, idx, indexing="ij")
    coeffs = np.stack([h0, h1], axis=-1).reshape(-1, 2).astype(np.float64)
    centers_xy_vox = coeffs @ primitive_vectors_vox

    x_limit = half_x - radius_vox
    y_limit = half_y - radius_vox
    inside = np.logical_and.reduce(
        [
            centers_xy_vox[:, 0] >= -x_limit,
            centers_xy_vox[:, 0] <= x_limit,
            centers_xy_vox[:, 1] >= -y_limit,
            centers_xy_vox[:, 1] <= y_limit,
        ]
    )
    selected = centers_xy_vox[inside]
    order = np.lexsort((selected[:, 0], selected[:, 1]))
    return selected[order]


def _enumerate_centers_xyz_vox(
    shape: tuple[int, int, int],
    primitive_vectors_vox: np.ndarray,
    basis_offsets_vox: np.ndarray,
    radius_vox: float,
) -> np.ndarray:
    nz, ny, nx = shape
    half_x = 0.5 * float(nx - 1)
    half_y = 0.5 * float(ny - 1)
    half_z = 0.5 * float(nz - 1)
    extent = np.sqrt((half_x + radius_vox) ** 2 + (half_y + radius_vox) ** 2 + (half_z + radius_vox) ** 2)
    min_step = float(np.min(np.linalg.norm(primitive_vectors_vox, axis=1)))
    integer_limit = int(np.ceil(extent / min_step)) + 2
    idx = np.arange(-integer_limit, integer_limit + 1, dtype=np.int32)
    h0, h1, h2 = np.meshgrid(idx, idx, idx, indexing="ij")
    coeffs = np.stack([h0, h1, h2], axis=-1).reshape(-1, 3).astype(np.float64)

    x_limit = half_x - radius_vox
    y_limit = half_y - radius_vox
    z_limit = half_z - radius_vox
    selected_rows: list[np.ndarray] = []
    for offset in np.asarray(basis_offsets_vox, dtype=np.float64):
        centers_xyz_vox = coeffs @ primitive_vectors_vox + offset[None, :]
        inside = np.logical_and.reduce(
            [
                centers_xyz_vox[:, 0] >= -x_limit,
                centers_xyz_vox[:, 0] <= x_limit,
                centers_xyz_vox[:, 1] >= -y_limit,
                centers_xyz_vox[:, 1] <= y_limit,
                centers_xyz_vox[:, 2] >= -z_limit,
                centers_xyz_vox[:, 2] <= z_limit,
            ]
        )
        if np.any(inside):
            selected_rows.append(centers_xyz_vox[inside])

    if not selected_rows:
        return np.zeros((0, 3), dtype=np.float64)

    selected = np.concatenate(selected_rows, axis=0)
    order = np.lexsort((selected[:, 0], selected[:, 1], selected[:, 2]))
    return selected[order]


def _stamp_disk_local(
    disk_plane: np.ndarray,
    center_x_vox: float,
    center_y_vox: float,
    radius_vox: float,
    superresolution: int,
) -> None:
    ny, nx = disk_plane.shape
    cx = 0.5 * float(nx - 1)
    cy = 0.5 * float(ny - 1)
    center_x_idx = cx + float(center_x_vox)
    center_y_idx = cy + float(center_y_vox)

    pad = radius_vox + 1.0
    x0 = max(0, int(np.floor(center_x_idx - pad)))
    x1 = min(nx, int(np.ceil(center_x_idx + pad)) + 1)
    y0 = max(0, int(np.floor(center_y_idx - pad)))
    y1 = min(ny, int(np.ceil(center_y_idx + pad)) + 1)

    yy, xx = np.ogrid[y0:y1, x0:x1]
    if int(superresolution) == 1:
        dy = yy.astype(np.float32) - np.float32(center_y_idx)
        dx = xx.astype(np.float32) - np.float32(center_x_idx)
        dist2 = dx * dx + dy * dy
        local_vfrac = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        local_vfrac[dist2 <= np.float32(radius_vox * radius_vox)] = 1.0
    else:
        ly = y1 - y0
        lx = x1 - x0
        ss = int(superresolution)
        y_hr, x_hr = np.ogrid[0 : ly * ss, 0 : lx * ss]
        dy = (
            y0
            + y_hr.astype(np.float32) / np.float32(ss)
            + np.float32(0.5 / ss - 0.5)
            - np.float32(center_y_idx)
        )
        dx = (
            x0
            + x_hr.astype(np.float32) / np.float32(ss)
            + np.float32(0.5 / ss - 0.5)
            - np.float32(center_x_idx)
        )
        highres_mask = (dx * dx + dy * dy) <= np.float32(radius_vox * radius_vox)
        local_vfrac = highres_mask.reshape(ly, ss, lx, ss).mean(axis=(1, 3), dtype=np.float32)

    disk_plane[y0:y1, x0:x1] = np.maximum(disk_plane[y0:y1, x0:x1], local_vfrac)


def _stamp_sphere_local(
    vfrac: np.ndarray,
    center_x_vox: float,
    center_y_vox: float,
    center_z_vox: float,
    radius_vox: float,
    superresolution: int,
) -> None:
    nz, ny, nx = vfrac.shape
    cx = 0.5 * float(nx - 1)
    cy = 0.5 * float(ny - 1)
    cz = 0.5 * float(nz - 1)
    center_x_idx = cx + float(center_x_vox)
    center_y_idx = cy + float(center_y_vox)
    center_z_idx = cz + float(center_z_vox)

    pad = radius_vox + 1.0
    x0 = max(0, int(np.floor(center_x_idx - pad)))
    x1 = min(nx, int(np.ceil(center_x_idx + pad)) + 1)
    y0 = max(0, int(np.floor(center_y_idx - pad)))
    y1 = min(ny, int(np.ceil(center_y_idx + pad)) + 1)
    z0 = max(0, int(np.floor(center_z_idx - pad)))
    z1 = min(nz, int(np.ceil(center_z_idx + pad)) + 1)

    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    if int(superresolution) == 1:
        dz = zz.astype(np.float32) - np.float32(center_z_idx)
        dy = yy.astype(np.float32) - np.float32(center_y_idx)
        dx = xx.astype(np.float32) - np.float32(center_x_idx)
        dist2 = dx * dx + dy * dy + dz * dz
        local_vfrac = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=np.float32)
        local_vfrac[dist2 <= np.float32(radius_vox * radius_vox)] = 1.0
    else:
        lz = z1 - z0
        ly = y1 - y0
        lx = x1 - x0
        ss = int(superresolution)
        z_hr, y_hr, x_hr = np.ogrid[0 : lz * ss, 0 : ly * ss, 0 : lx * ss]
        dz = (
            z0
            + z_hr.astype(np.float32) / np.float32(ss)
            + np.float32(0.5 / ss - 0.5)
            - np.float32(center_z_idx)
        )
        dy = (
            y0
            + y_hr.astype(np.float32) / np.float32(ss)
            + np.float32(0.5 / ss - 0.5)
            - np.float32(center_y_idx)
        )
        dx = (
            x0
            + x_hr.astype(np.float32) / np.float32(ss)
            + np.float32(0.5 / ss - 0.5)
            - np.float32(center_x_idx)
        )
        highres_mask = (dx * dx + dy * dy + dz * dz) <= np.float32(radius_vox * radius_vox)
        local_vfrac = highres_mask.reshape(lz, ss, ly, ss, lx, ss).mean(
            axis=(1, 3, 5),
            dtype=np.float32,
        )

    vfrac[z0:z1, y0:y1, x0:x1] = np.maximum(vfrac[z0:z1, y0:y1, x0:x1], local_vfrac)
