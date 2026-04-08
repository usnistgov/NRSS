from __future__ import annotations

import gc
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import h5py
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from NRSS.morphology import Material, Morphology, OpticalConstants


DATA_DIR = REPO_ROOT / "tests" / "validation" / "data" / "mwcnt"
EXPERIMENTAL_REFERENCE_PATH = DATA_DIR / "MWCNT_reference_A.h5"
CNT_GEOMETRY_PATH = DATA_DIR / "mwcnt_seed12345_cnts.csv"
OPTICAL_CONSTANTS_PATH = DATA_DIR / "MWCNT_opts.csv"

SHAPE = (128, 1024, 1024)
UPSCALE_FACTOR = 2.0
PHYS_SIZE_NM = 1.0
GAUSSIAN_SIGMA = 3.0
HOLLOW_FRACTION = 0.325
Q_BAND_MIN_NM = 0.6
Q_BAND_MAX_NM = 0.7
Q_COMPARE_MIN_NM = 0.2
Q_COMPARE_MAX_NM = 0.95
CHI_PARALLEL = (-20.0, 20.0)
CHI_PERP = (-110.0, -70.0)
ENERGY_CHECKS_EV = (285.0, 292.0)
EANGLE_ROTATION = [0.0, 20.0, 340.0]
WINDOWING_TYPE_DEFAULT = 0
FIELD_BOUNDARY_MODE_DEFAULT = "periodic"
GEOMETRY_SEED = 12345
RSA_NUM_TRIALS = 20_000
RSA_RADIUS_LOGNORMAL_MU = 2.225
RSA_RADIUS_LOGNORMAL_SIGMA = 0.23
RSA_THETA_MU_RAD = float(np.pi / 2.0)
RSA_THETA_SIGMA_RAD = float(1.0 / (2.0 * np.pi))
RSA_LENGTH_LOWER_NM_UPSCALED = 75.0
RSA_LENGTH_UPPER_NM_UPSCALED = 300.0
RSA_BOX_XY_VOX_UPSCALED = 2048
RSA_BOX_Z_VOX_UPSCALED = 256
TABLE_I_THETA_MU_RAD = float(np.pi / 2.0)
TABLE_I_THETA_SIGMA_RAD = float(1.0 / (2.0 * np.pi))
TABLE_I_RADIUS_MEAN_NM = 4.60
TABLE_I_RADIUS_STD_NM = 1.03
TABLE_I_HOLLOW_FRACTION = 0.325

EXPERIMENTAL_REFERENCE_LABEL = "Experimental reference"
EXPERIMENTAL_REFERENCE_CITATION = (
    "Dudenas, P. J.; Flagg, L. Q.; Goetz, K.; Shapturenka, P.; Fagan, J. A.; "
    "Gann, E.; DeLongchamp, D. M. J. Chem. Phys. 2025, 163 (6), 061501. "
    "https://doi.org/10.1063/5.0267709."
)

EXPERIMENTAL_THRESHOLDS = {
    "a_vs_energy_qband": {"correlation_min": 0.949, "max_abs_diff": 0.44, "rmse": 0.15},
    "a_vs_q_285": {"correlation_min": 0.96, "max_abs_diff": 0.17, "rmse": 0.09},
    "a_vs_q_292": {"correlation_min": 0.96, "max_abs_diff": 0.42, "rmse": 0.22},
}


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


@lru_cache(maxsize=1)
def _load_optical_constants() -> OpticalConstants:
    reference_data = pd.read_csv(OPTICAL_CONSTANTS_PATH)
    reference = load_experimental_reference_observables()
    energies = np.asarray(reference["a_vs_energy_qband"].coords["energy"].values, dtype=np.float64)
    return OpticalConstants.calc_constants(energies, reference_data, name="MWCNT")


@lru_cache(maxsize=None)
def load_cnt_geometry(path: Path | str = CNT_GEOMETRY_PATH) -> np.ndarray:
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if table.ndim != 1 or table.size == 0:
        raise AssertionError("Unexpected MWCNT geometry table shape.")
    return table


@lru_cache(maxsize=None)
def _ball_offsets(radius_vox: int) -> np.ndarray:
    if radius_vox < 0:
        raise AssertionError("Radius must be non-negative.")
    if radius_vox == 0:
        return np.zeros((1, 3), dtype=np.int16)
    grid = np.arange(-radius_vox, radius_vox + 1, dtype=np.int16)
    zz, yy, xx = np.meshgrid(grid, grid, grid, indexing="ij")
    mask = (zz * zz + yy * yy + xx * xx) <= radius_vox * radius_vox
    return np.column_stack((zz[mask], yy[mask], xx[mask])).astype(np.int16, copy=False)


def _centerline_points(row: np.void) -> np.ndarray:
    length = float(row["length"]) / UPSCALE_FACTOR
    theta = float(row["theta"])
    psi = float(row["psi"])
    num_points = max(int(float(row["length"])) + 1, 2)
    r = np.linspace(-length / 2.0, length / 2.0, num_points, dtype=np.float32)
    sin_theta = np.sin(theta, dtype=np.float32)
    cos_theta = np.cos(theta, dtype=np.float32)
    cos_psi = np.cos(psi, dtype=np.float32)
    sin_psi = np.sin(psi, dtype=np.float32)
    x = (np.float32(row["x_center"] / UPSCALE_FACTOR) + r * sin_theta * cos_psi) % SHAPE[2]
    y = (np.float32(row["y_center"] / UPSCALE_FACTOR) + r * sin_theta * sin_psi) % SHAPE[1]
    z = (np.float32(row["z_center"] / UPSCALE_FACTOR) + r * cos_theta) % SHAPE[0]
    return np.column_stack((z, y, x)).astype(np.float32, copy=False)


def _stamp_points(
    volume: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray,
    value: float,
) -> None:
    if centers.size == 0:
        return
    rounded = np.rint(centers).astype(np.int32, copy=False)
    zz = (rounded[:, None, 0] + offsets[None, :, 0]) % volume.shape[0]
    yy = (rounded[:, None, 1] + offsets[None, :, 1]) % volume.shape[1]
    xx = (rounded[:, None, 2] + offsets[None, :, 2]) % volume.shape[2]
    volume[zz, yy, xx] = np.float32(value)


def _compute_oriented_fields(
    cnt_vfrac: np.ndarray,
    field_boundary_mode: Literal["periodic", "legacy"] = FIELD_BOUNDARY_MODE_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    if field_boundary_mode == "periodic":
        blurred = gaussian_filter(cnt_vfrac, sigma=GAUSSIAN_SIGMA, mode="wrap").astype(np.float32, copy=False)
        grad_z = 0.5 * (np.roll(blurred, -1, axis=0) - np.roll(blurred, 1, axis=0))
        grad_y = 0.5 * (np.roll(blurred, -1, axis=1) - np.roll(blurred, 1, axis=1))
        grad_x = 0.5 * (np.roll(blurred, -1, axis=2) - np.roll(blurred, 1, axis=2))
    elif field_boundary_mode == "legacy":
        blurred = gaussian_filter(cnt_vfrac, sigma=GAUSSIAN_SIGMA).astype(np.float32, copy=False)
        grad_z, grad_y, grad_x = np.gradient(blurred)
    else:
        raise ValueError(f"Unknown field_boundary_mode={field_boundary_mode!r}")

    orient_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z, dtype=np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        grad_x = -grad_x / orient_mag
        grad_y = -grad_y / orient_mag
        grad_z = -grad_z / orient_mag

    for grad in (grad_x, grad_y, grad_z):
        invalid = ~np.isfinite(grad)
        if np.any(invalid):
            grad[invalid] = 0.0

    np.clip(grad_z, -1.0, 1.0, out=grad_z)
    theta = np.arccos(grad_z).astype(np.float32, copy=False)
    psi = np.arctan2(grad_y, grad_x).astype(np.float32, copy=False)
    return theta, psi


def build_mwcnt_fields(
    geometry_path: Path | str = CNT_GEOMETRY_PATH,
    field_boundary_mode: Literal["periodic", "legacy"] = FIELD_BOUNDARY_MODE_DEFAULT,
) -> dict[str, np.ndarray]:
    cnt_vfrac = np.zeros(SHAPE, dtype=np.float32)
    for row in load_cnt_geometry(geometry_path):
        centerline = _centerline_points(row)
        outer_radius = max(int(np.rint(float(row["radius"]) / UPSCALE_FACTOR)), 1)
        inner_radius = int(np.rint(float(row["radius"]) * HOLLOW_FRACTION / UPSCALE_FACTOR))

        outer_points = centerline[outer_radius:-outer_radius]
        _stamp_points(cnt_vfrac, outer_points, _ball_offsets(outer_radius), value=1.0)
        if inner_radius > 0:
            _stamp_points(cnt_vfrac, centerline, _ball_offsets(inner_radius), value=0.0)

    cnt_vfrac = np.clip(cnt_vfrac, 0.0, 1.0, out=cnt_vfrac)
    theta, psi = _compute_oriented_fields(cnt_vfrac, field_boundary_mode=field_boundary_mode)

    empty = cnt_vfrac <= 0.0
    theta[empty] = 0.0
    psi[empty] = 0.0

    vacuum = (1.0 - cnt_vfrac).astype(np.float32, copy=False)
    zeros = np.zeros(SHAPE, dtype=np.float32)
    return {
        "mat1_vfrac": cnt_vfrac,
        "mat1_s": cnt_vfrac.copy(),
        "mat1_theta": theta,
        "mat1_psi": psi,
        "mat2_vfrac": vacuum,
        "mat2_s": zeros,
        "mat2_theta": zeros.copy(),
        "mat2_psi": zeros.copy(),
    }


@lru_cache(maxsize=None)
def geometry_realization_stats(path: Path | str = CNT_GEOMETRY_PATH) -> dict[str, float]:
    geometry = load_cnt_geometry(path)
    radius_nm = np.asarray(geometry["radius"], dtype=np.float64) / UPSCALE_FACTOR
    theta_rad = np.asarray(geometry["theta"], dtype=np.float64)
    length_nm = np.asarray(geometry["length"], dtype=np.float64) / UPSCALE_FACTOR
    return {
        "accepted_count": float(geometry.shape[0]),
        "radius_mean_nm": float(np.mean(radius_nm)),
        "radius_std_nm": float(np.std(radius_nm, ddof=0)),
        "theta_mean_rad": float(np.mean(theta_rad)),
        "theta_std_rad": float(np.std(theta_rad, ddof=0)),
        "length_min_nm": float(np.min(length_nm)),
        "length_max_nm": float(np.max(length_nm)),
    }


def build_mwcnt_morphology(
    create_cy_object: bool = True,
    geometry_path: Path | str = CNT_GEOMETRY_PATH,
    eangle_rotation: list[float] | tuple[float, float, float] | None = None,
    windowing_type: int = WINDOWING_TYPE_DEFAULT,
    field_boundary_mode: Literal["periodic", "legacy"] = FIELD_BOUNDARY_MODE_DEFAULT,
    *,
    backend: str | None = None,
    backend_options: dict[str, Any] | None = None,
    resident_mode: str | None = None,
    input_policy: str = "coerce",
    ownership_policy: str | None = None,
    field_namespace: str = "numpy",
) -> Morphology:
    optical_constants = _load_optical_constants()
    energies = list(map(float, optical_constants.energies))
    fields = _convert_fields_namespace(
        build_mwcnt_fields(geometry_path=geometry_path, field_boundary_mode=field_boundary_mode),
        field_namespace=field_namespace,
    )

    materials = {
        1: Material(
            materialID=1,
            Vfrac=fields["mat1_vfrac"],
            S=fields["mat1_s"],
            theta=fields["mat1_theta"],
            psi=fields["mat1_psi"],
            energies=energies,
            opt_constants=optical_constants.opt_constants,
            name="MWCNT",
        ),
        2: Material(
            materialID=2,
            Vfrac=fields["mat2_vfrac"],
            S=fields["mat2_s"],
            theta=fields["mat2_theta"],
            psi=fields["mat2_psi"],
            energies=energies,
            opt_constants=OpticalConstants(energies, name="vacuum").opt_constants,
            name="vacuum",
        ),
    }

    config = {
        "CaseType": 0,
        "MorphologyType": 0,
        "Energies": energies,
        "EAngleRotation": list(eangle_rotation) if eangle_rotation is not None else EANGLE_ROTATION,
        "AlgorithmType": 0,
        "WindowingType": int(windowing_type),
        "RotMask": 0,
        "ReferenceFrame": 1,
        "EwaldsInterpolation": 1,
    }

    morph = Morphology(
        2,
        materials=materials,
        PhysSize=PHYS_SIZE_NM,
        config=config,
        create_cy_object=create_cy_object,
        backend=backend,
        backend_options=backend_options,
        resident_mode=resident_mode,
        input_policy=input_policy,
        ownership_policy=ownership_policy,
    )
    morph.check_materials(quiet=True)
    if create_cy_object:
        morph.validate_all(quiet=True)
    return morph


def run_mwcnt_pybind(
    geometry_path: Path | str = CNT_GEOMETRY_PATH,
    eangle_rotation: list[float] | tuple[float, float, float] | None = None,
    windowing_type: int = WINDOWING_TYPE_DEFAULT,
    field_boundary_mode: Literal["periodic", "legacy"] = FIELD_BOUNDARY_MODE_DEFAULT,
    *,
    backend: str = "cyrsoxs",
    backend_options: dict[str, Any] | None = None,
    resident_mode: str | None = None,
    input_policy: str = "coerce",
    ownership_policy: str | None = None,
    field_namespace: str = "numpy",
) -> xr.DataArray:
    morph = build_mwcnt_morphology(
        create_cy_object=True,
        geometry_path=geometry_path,
        eangle_rotation=eangle_rotation,
        windowing_type=windowing_type,
        field_boundary_mode=field_boundary_mode,
        backend=backend,
        backend_options=backend_options,
        resident_mode=resident_mode,
        input_policy=input_policy,
        ownership_policy=ownership_policy,
        field_namespace=field_namespace,
    )
    try:
        scattering = morph.run(stdout=False, stderr=False, return_xarray=True)
        if scattering is None:
            raise AssertionError("MWCNT pybind run returned no scattering data.")
        return scattering.copy(deep=True)
    finally:
        del morph
        release_runtime_memory()


def scattering_to_chiq(scattering: xr.DataArray) -> xr.DataArray:
    """
    Reduce MWCNT scattering through the maintained historical WPIntegrator path.

    This helper intentionally preserves the legacy detector-plane ``q_perp``
    remesh because the vendored MWCNT observables were derived from that
    workflow. It remains maintained for historical comparability to the
    tutorial/manuscript reduction path, not as recommended practice for new
    analytical NRSS validations.
    """
    from PyHyperScattering.integrate import WPIntegrator

    integrator = WPIntegrator(use_chunked_processing=False)
    remeshed = integrator.integrateImageStack(scattering)
    if "energy" not in remeshed.dims or "chi" not in remeshed.dims:
        raise AssertionError("PyHyperScattering output missing expected energy/chi dimensions.")
    qdim = next((dim for dim in remeshed.dims if dim == "q" or dim.startswith("q")), None)
    if qdim is None:
        raise AssertionError("PyHyperScattering output missing q dimension.")
    if qdim != "q":
        remeshed = remeshed.rename({qdim: "q"})
    return remeshed.sortby("energy")


def chi_sectors_to_anisotropy(remeshed: xr.DataArray) -> xr.DataArray:
    para = remeshed.sel(chi=slice(*CHI_PARALLEL)).mean("chi")
    perp = remeshed.sel(chi=slice(*CHI_PERP)).mean("chi")
    with np.errstate(divide="ignore", invalid="ignore"):
        anisotropy = (para - perp) / (para + perp)
    return anisotropy.sortby("energy")


def reduce_to_observables(remeshed: xr.DataArray) -> dict[str, xr.DataArray]:
    anisotropy = chi_sectors_to_anisotropy(remeshed)
    return {
        "a_qe": anisotropy,
        "a_vs_energy_qband": anisotropy.sel(q=slice(Q_BAND_MIN_NM, Q_BAND_MAX_NM)).mean("q"),
        "a_vs_q_285": anisotropy.sel(energy=ENERGY_CHECKS_EV[0], method="nearest"),
        "a_vs_q_292": anisotropy.sel(energy=ENERGY_CHECKS_EV[1], method="nearest"),
    }


def reduce_scattering_to_observables(scattering: xr.DataArray) -> dict[str, xr.DataArray]:
    return reduce_to_observables(scattering_to_chiq(scattering))


def write_reference_observables(
    observables: dict[str, xr.DataArray],
    path: Path,
    source_kind: str,
    description: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        a_vs_energy = observables["a_vs_energy_qband"]
        a_vs_q_285 = observables["a_vs_q_285"]
        a_vs_q_292 = observables["a_vs_q_292"]
        f.create_dataset("energy_ev", data=np.asarray(a_vs_energy.coords["energy"].values, dtype=np.float64))
        f.create_dataset("q_nm", data=np.asarray(a_vs_q_285.coords["q"].values, dtype=np.float64))
        f.create_dataset("A_vs_energy_qband", data=np.asarray(a_vs_energy.values, dtype=np.float64))
        f.create_dataset("A_vs_q_285", data=np.asarray(a_vs_q_285.values, dtype=np.float64))
        f.create_dataset("A_vs_q_292", data=np.asarray(a_vs_q_292.values, dtype=np.float64))
        f.attrs["source_kind"] = source_kind
        f.attrs["description"] = description
        f.attrs["shape"] = str(SHAPE)
        f.attrs["phys_size_nm"] = PHYS_SIZE_NM
        f.attrs["q_band_nm"] = str((Q_BAND_MIN_NM, Q_BAND_MAX_NM))


def load_experimental_reference_observables() -> dict[str, xr.DataArray]:
    with h5py.File(EXPERIMENTAL_REFERENCE_PATH, "r") as f:
        energy = np.asarray(f["energy_ev"][()], dtype=np.float64)
        q_nm = np.asarray(f["q_nm"][()], dtype=np.float64)
        a_vs_energy = np.asarray(f["A_vs_energy_qband"][()], dtype=np.float64)
        a_vs_q_285 = np.asarray(f["A_vs_q_285"][()], dtype=np.float64)
        a_vs_q_292 = np.asarray(f["A_vs_q_292"][()], dtype=np.float64)
    return {
        "a_vs_energy_qband": xr.DataArray(a_vs_energy, dims=("energy",), coords={"energy": energy}),
        "a_vs_q_285": xr.DataArray(a_vs_q_285, dims=("q",), coords={"q": q_nm}),
        "a_vs_q_292": xr.DataArray(a_vs_q_292, dims=("q",), coords={"q": q_nm}),
    }


def align_observables_to_reference(
    observables: dict[str, xr.DataArray],
    reference: dict[str, xr.DataArray],
    q_min_nm: float = Q_COMPARE_MIN_NM,
    q_max_nm: float = Q_COMPARE_MAX_NM,
) -> dict[str, xr.DataArray]:
    q_window = reference["a_vs_q_285"].sel(q=slice(q_min_nm, q_max_nm)).coords["q"].values
    if q_window.size == 0:
        raise AssertionError("Chosen q comparison window is empty.")
    return {
        "a_vs_energy_qband": observables["a_vs_energy_qband"].interp(
            energy=reference["a_vs_energy_qband"].coords["energy"].values
        ),
        "reference_a_vs_energy_qband": reference["a_vs_energy_qband"],
        "a_vs_q_285": observables["a_vs_q_285"].interp(q=q_window),
        "reference_a_vs_q_285": reference["a_vs_q_285"].sel(q=slice(q_min_nm, q_max_nm)),
        "a_vs_q_292": observables["a_vs_q_292"].interp(q=q_window),
        "reference_a_vs_q_292": reference["a_vs_q_292"].sel(q=slice(q_min_nm, q_max_nm)),
    }


def compute_observable_metrics(comparison: dict[str, xr.DataArray]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for key in ("a_vs_energy_qband", "a_vs_q_285", "a_vs_q_292"):
        ref_key = f"reference_{key}"
        observed = np.asarray(comparison[key].values, dtype=np.float64)
        reference = np.asarray(comparison[ref_key].values, dtype=np.float64)
        delta = observed - reference
        mask = np.isfinite(observed) & np.isfinite(reference)
        correlation = np.nan
        if np.count_nonzero(mask) >= 2:
            correlation = float(np.corrcoef(observed[mask], reference[mask])[0, 1])
        metrics[key] = {
            "max_abs_diff": float(np.nanmax(np.abs(delta))),
            "rmse": float(np.sqrt(np.nanmean(delta * delta))),
            "correlation": correlation,
        }
    return metrics


def metrics_within_thresholds(
    metrics: dict[str, dict[str, float]],
    thresholds: dict[str, dict[str, float]],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for key, limits in thresholds.items():
        for metric_name, limit in limits.items():
            if metric_name.endswith("_min"):
                base_name = metric_name.removesuffix("_min")
                value = metrics[key][base_name]
                if not np.isfinite(value) or value < float(limit):
                    failures.append(f"{key} {base_name}={value:.6g} below minimum {float(limit):.6g}")
                continue
            value = metrics[key][metric_name]
            if not np.isfinite(value) or value > float(limit):
                failures.append(f"{key} {metric_name}={value:.6g} exceeds limit {float(limit):.6g}")
    return len(failures) == 0, failures


def plot_mwcnt_validation_panel(
    comparison: dict[str, xr.DataArray],
    metrics: dict[str, dict[str, float]],
    out_path: Path,
    q_min_nm: float,
    q_max_nm: float,
    title: str = "MWCNT experimental validation",
    simulation_label: str = "Pybind + WPIntegrator (historical maintained path)",
    simulation_details: list[str] | None = None,
    eangle_rotation: list[float] | tuple[float, float, float] | None = None,
    description: str | None = None,
    geometry_source_name: str | None = None,
    geometry_path: Path | str | None = None,
    windowing_type: int = WINDOWING_TYPE_DEFAULT,
    field_boundary_mode: Literal["periodic", "legacy"] = FIELD_BOUNDARY_MODE_DEFAULT,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), constrained_layout=True)
    stats = geometry_realization_stats(geometry_path if geometry_path is not None else CNT_GEOMETRY_PATH)

    ax = axes[0, 0]
    ax.plot(
        comparison["reference_a_vs_energy_qband"].coords["energy"].values,
        comparison["reference_a_vs_energy_qband"].values,
        color="black",
        linewidth=1.7,
        label=EXPERIMENTAL_REFERENCE_LABEL,
    )
    ax.plot(
        comparison["a_vs_energy_qband"].coords["energy"].values,
        comparison["a_vs_energy_qband"].values,
        color="#d62728",
        linewidth=1.8,
        linestyle="--",
        label=simulation_label,
    )
    ax.set_xlabel("Energy [eV]")
    ax.set_ylabel("A(E)")
    ax.set_title(f"A(E), q = {Q_BAND_MIN_NM:.2f}-{Q_BAND_MAX_NM:.2f} nm$^{{-1}}$")
    ax.legend(loc="best")

    ax = axes[0, 1]
    ax.plot(
        comparison["reference_a_vs_q_285"].coords["q"].values,
        comparison["reference_a_vs_q_285"].values,
        color="black",
        linewidth=1.6,
        label="285 eV reference",
    )
    ax.plot(
        comparison["reference_a_vs_q_292"].coords["q"].values,
        comparison["reference_a_vs_q_292"].values,
        color="#1f77b4",
        linewidth=1.6,
        label="292 eV reference",
    )
    ax.plot(
        comparison["a_vs_q_285"].coords["q"].values,
        comparison["a_vs_q_285"].values,
        color="black",
        linewidth=1.8,
        linestyle="--",
        label="285 eV pybind",
    )
    ax.plot(
        comparison["a_vs_q_292"].coords["q"].values,
        comparison["a_vs_q_292"].values,
        color="#1f77b4",
        linewidth=1.8,
        linestyle="--",
        label="292 eV pybind",
    )
    ax.set_xlabel(r"q [nm$^{-1}$]")
    ax.set_ylabel("A(q)")
    ax.set_xlim(q_min_nm, q_max_nm)
    ax.set_title("A(q) slices vs q")
    ax.legend(loc="best", fontsize=9)

    ax = axes[1, 0]
    residual_285 = comparison["a_vs_q_285"] - comparison["reference_a_vs_q_285"]
    residual_292 = comparison["a_vs_q_292"] - comparison["reference_a_vs_q_292"]
    ax.plot(
        residual_285.coords["q"].values,
        residual_285.values,
        color="black",
        linewidth=1.6,
        label="285 eV residual",
    )
    ax.plot(
        residual_292.coords["q"].values,
        residual_292.values,
        color="#1f77b4",
        linewidth=1.6,
        label="292 eV residual",
    )
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.set_xlabel(r"q [nm$^{-1}$]")
    ax.set_ylabel(r"$\Delta A(q)$")
    ax.set_xlim(q_min_nm, q_max_nm)
    ax.set_title("Residuals vs q")
    ax.legend(loc="best", fontsize=9)

    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        "Maintained deterministic MWCNT validation",
        "",
        f"Shape: {SHAPE}",
        f"PhysSize: {PHYS_SIZE_NM} nm",
        f"EAngleRotation: {list(eangle_rotation) if eangle_rotation is not None else EANGLE_ROTATION}",
        f"WindowingType: {windowing_type}",
        f"Field boundary mode: {field_boundary_mode}",
        "",
        "Manuscript Table I targets:",
        (
            f"theta_mu={TABLE_I_THETA_MU_RAD:.4f} rad "
            f"({np.degrees(TABLE_I_THETA_MU_RAD):.1f} deg), "
            f"theta_sigma={TABLE_I_THETA_SIGMA_RAD:.4f} rad "
            f"({np.degrees(TABLE_I_THETA_SIGMA_RAD):.1f} deg)"
        ),
        f"radius mean/std={TABLE_I_RADIUS_MEAN_NM:.2f}/{TABLE_I_RADIUS_STD_NM:.2f} nm",
        f"hollow fraction={TABLE_I_HOLLOW_FRACTION:.3f}",
        "",
        "Tutorial/RSA generator parameters:",
        (
            f"lognormal radius mu={RSA_RADIUS_LOGNORMAL_MU:.3f}, "
            f"sigma={RSA_RADIUS_LOGNORMAL_SIGMA:.3f}"
        ),
        (
            f"theta_mu={RSA_THETA_MU_RAD:.4f} rad, "
            f"theta_sigma={RSA_THETA_SIGMA_RAD:.4f} rad"
        ),
        (
            f"upscaled length range=[{RSA_LENGTH_LOWER_NM_UPSCALED:.0f}, "
            f"{RSA_LENGTH_UPPER_NM_UPSCALED:.0f}] nm"
        ),
        (
            f"upscaled box={RSA_BOX_Z_VOX_UPSCALED}x{RSA_BOX_XY_VOX_UPSCALED}x"
            f"{RSA_BOX_XY_VOX_UPSCALED} vox, seed={GEOMETRY_SEED}, "
            f"trials={RSA_NUM_TRIALS}"
        ),
        "",
        "Realized geometry in this run:",
        f"accepted CNTs={int(stats['accepted_count'])}",
        (
            f"radius mean/std after 2x downscale="
            f"{stats['radius_mean_nm']:.3f}/{stats['radius_std_nm']:.3f} nm"
        ),
        (
            f"theta mean/std={stats['theta_mean_rad']:.4f}/"
            f"{stats['theta_std_rad']:.4f} rad"
        ),
        (
            f"length range after 2x downscale="
            f"[{stats['length_min_nm']:.1f}, {stats['length_max_nm']:.1f}] nm"
        ),
        "",
        f"Field-orientation blur sigma: {GAUSSIAN_SIGMA}",
        f"Hollow fraction: {HOLLOW_FRACTION}",
        f"Geometry source: {geometry_source_name or CNT_GEOMETRY_PATH.name}",
        f"Optical constants: {OPTICAL_CONSTANTS_PATH.name}",
        f"A(E) q-band: [{Q_BAND_MIN_NM:.2f}, {Q_BAND_MAX_NM:.2f}] nm^-1",
        f"A(q) compare q-range: [{q_min_nm:.2f}, {q_max_nm:.2f}] nm^-1",
        "",
        "Metrics:",
        (
            f"A(E): r={metrics['a_vs_energy_qband']['correlation']:.3f}, "
            f"RMSE={metrics['a_vs_energy_qband']['rmse']:.4f}, "
            f"max abs={metrics['a_vs_energy_qband']['max_abs_diff']:.4f}"
        ),
        (
            f"A(q) 285 eV: r={metrics['a_vs_q_285']['correlation']:.3f}, "
            f"RMSE={metrics['a_vs_q_285']['rmse']:.4f}, "
            f"max abs={metrics['a_vs_q_285']['max_abs_diff']:.4f}"
        ),
        (
            f"A(q) 292 eV: r={metrics['a_vs_q_292']['correlation']:.3f}, "
            f"RMSE={metrics['a_vs_q_292']['rmse']:.4f}, "
            f"max abs={metrics['a_vs_q_292']['max_abs_diff']:.4f}"
        ),
        "",
        "Reference:",
        EXPERIMENTAL_REFERENCE_CITATION,
    ]
    if simulation_details:
        lines[0] = simulation_details[0]
        if len(simulation_details) > 1:
            insertion_index = 1
            for detail in simulation_details[1:]:
                lines.insert(insertion_index, detail)
                insertion_index += 1
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=9,
        wrap=True,
    )

    fig.suptitle(title, fontsize=15)
    fig.savefig(
        out_path,
        format="png",
        dpi=170,
        bbox_inches="tight",
        metadata={
            "Title": title,
            "Description": description
            or (
                "Deterministic fixed-seed MWCNT morphology compared against reduced "
                "experimental WAXS anisotropy observables. "
                f"Reference: {EXPERIMENTAL_REFERENCE_CITATION}"
            ),
            "Author": "OpenAI Codex",
            "Software": "matplotlib",
        },
    )
    plt.close(fig)
