import gc
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from NRSS import SFieldMode
from NRSS.morphology import Material, Morphology
from tests.path_matrix import ComputationPath


pytestmark = [pytest.mark.path_matrix, pytest.mark.reference_parity]


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = REPO_ROOT / "test-reports" / "analytical-sphere-dev"
WRITE_VALIDATION_PLOTS = os.environ.get("NRSS_WRITE_VALIDATION_PLOTS", "").strip() == "1"

SHAPE = (512, 512, 512)
PHYS_SIZE_NM = 1.0
ENERGY_EV = 285.0
DIAMETERS_NM = [70.0, 128.0]
SUPERRESOLUTIONS = [1, 2, 3, 4]
ASSERT_SUPERRESOLUTION = 1
FLAT_DETECTOR_OVERSAMPLE = 8

Q_POINTWISE_MIN = 0.06
Q_POINTWISE_MAX = 1.0
Q_EXTREMA_MIN = 0.05
Q_EXTREMA_MAX = 1.0
Q_PLOT_MIN = 0.0
Q_PLOT_MAX = 1.0
MIN_SIGNAL_FOR_LOG = 1e-5
GEOMETRY_THRESHOLDS_BY_DIAMETER = {
    70.0: {
        "sr1_rms_log_max": 0.070,
        "sr1_p95_log_abs_max": 0.170,
        "sr1_flat_min_mae_max": 0.00085,
        "sr1_flat_min_rmse_max": 0.00095,
        "sr1_min_ratio_min": 20.0,
        "sr1_point_ratio_min": 2.7,
    },
    128.0: {
        "sr1_rms_log_max": 0.123,
        "sr1_p95_log_abs_max": 0.260,
        "sr1_flat_min_mae_max": 0.00070,
        "sr1_flat_min_rmse_max": 0.00080,
        "sr1_min_ratio_min": 24.0,
        "sr1_point_ratio_min": 1.8,
    },
}

SUPERRES_COLORS = {
    1: "#1f77b4",
    2: "#d62728",
    3: "#2ca02c",
    4: "#ff7f0e",
}

SPHERE_OC = {
    285.0: (0.0, 2e-4, 0.0, 2e-4),
}


@lru_cache(maxsize=1)
def _has_visible_gpu() -> bool:
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


def _cyrsoxs_detector_axis(n: int, phys_size_nm: float) -> np.ndarray:
    if int(n) < 2:
        raise AssertionError(f"CyRSoXS detector axis needs at least 2 points, got n={n}.")
    start = -np.pi / float(phys_size_nm)
    step = (2.0 * np.pi / float(phys_size_nm)) / float(int(n) - 1)
    return start + np.arange(int(n), dtype=np.float64) * step


def _to_backend_namespace(array: np.ndarray, field_namespace: str):
    if field_namespace == "numpy":
        return np.ascontiguousarray(array.astype(np.float32, copy=False))
    if field_namespace != "cupy":
        raise AssertionError(f"Unsupported field namespace {field_namespace!r}.")
    import cupy as cp

    return cp.ascontiguousarray(cp.asarray(array, dtype=cp.float32))


def _path_runtime_kwargs(nrss_path: ComputationPath) -> dict[str, object]:
    return {
        "backend": nrss_path.backend,
        "backend_options": nrss_path.backend_options,
        "resident_mode": nrss_path.resident_mode,
        "input_policy": "strict" if nrss_path.category == "cupy" else "coerce",
        "ownership_policy": nrss_path.ownership_policy,
        "field_namespace": nrss_path.field_namespace,
    }


def _release_runtime_memory() -> None:
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


def _superresolved_sphere_vfrac(diameter_nm: float, superresolution: int) -> np.ndarray:
    if int(superresolution) < 1:
        raise AssertionError("superresolution must be >= 1.")
    superresolution = int(superresolution)
    nz, ny, nx = SHAPE
    radius_vox = float(diameter_nm) / (2.0 * PHYS_SIZE_NM)
    cz = (nz - 1) / 2.0
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0

    pad = radius_vox + 2.0
    z0 = max(0, int(np.floor(cz - pad)))
    z1 = min(nz, int(np.ceil(cz + pad)) + 1)
    y0 = max(0, int(np.floor(cy - pad)))
    y1 = min(ny, int(np.ceil(cy + pad)) + 1)
    x0 = max(0, int(np.floor(cx - pad)))
    x1 = min(nx, int(np.ceil(cx + pad)) + 1)

    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    if superresolution == 1:
        dz = zz.astype(np.float32) - np.float32(cz)
        dy = yy.astype(np.float32) - np.float32(cy)
        dx = xx.astype(np.float32) - np.float32(cx)
        dist2 = dx * dx + dy * dy + dz * dz
        local_vfrac = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=np.float32)
        local_vfrac[dist2 <= np.float32(radius_vox * radius_vox)] = 1.0
    else:
        lz = z1 - z0
        ly = y1 - y0
        lx = x1 - x0
        ss = superresolution
        z_hr, y_hr, x_hr = np.ogrid[0 : lz * ss, 0 : ly * ss, 0 : lx * ss]
        dz = (
            z0
            + z_hr.astype(np.float32) / np.float32(ss)
            + np.float32(0.5 / ss - 0.5)
            - np.float32(cz)
        )
        dy = (
            y0
            + y_hr.astype(np.float32) / np.float32(ss)
            + np.float32(0.5 / ss - 0.5)
            - np.float32(cy)
        )
        dx = (
            x0
            + x_hr.astype(np.float32) / np.float32(ss)
            + np.float32(0.5 / ss - 0.5)
            - np.float32(cx)
        )
        highres_mask = (dx * dx + dy * dy + dz * dz) <= np.float32(radius_vox * radius_vox)
        local_vfrac = highres_mask.reshape(lz, ss, ly, ss, lx, ss).mean(
            axis=(1, 3, 5),
            dtype=np.float32,
        )

    vfrac = np.zeros(SHAPE, dtype=np.float32)
    vfrac[z0:z1, y0:y1, x0:x1] = local_vfrac
    return vfrac


def _sphere_and_vacuum_vfrac(diameter_nm: float, superresolution: int) -> tuple[np.ndarray, np.ndarray]:
    sphere = _superresolved_sphere_vfrac(diameter_nm=diameter_nm, superresolution=superresolution)
    vacuum = (1.0 - sphere).astype(np.float32)
    return sphere, vacuum


def _run_sphere_backend(
    diameter_nm: float,
    superresolution: int,
    energies_eV: list[float],
    runtime_kwargs: dict[str, object],
) -> xr.DataArray:
    sphere_vfrac, vacuum_vfrac = _sphere_and_vacuum_vfrac(
        diameter_nm=diameter_nm,
        superresolution=superresolution,
    )
    field_namespace = str(runtime_kwargs["field_namespace"])
    sphere_vfrac = _to_backend_namespace(sphere_vfrac, field_namespace)
    vacuum_vfrac = _to_backend_namespace(vacuum_vfrac, field_namespace)

    mat1 = Material(
        materialID=1,
        Vfrac=sphere_vfrac,
        S=SFieldMode.ISOTROPIC,
        theta=None,
        psi=None,
        energies=energies_eV,
        opt_constants={float(energy_eV): list(SPHERE_OC[float(energy_eV)]) for energy_eV in energies_eV},
        name="sphere",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=vacuum_vfrac,
        S=SFieldMode.ISOTROPIC,
        theta=None,
        psi=None,
        energies=energies_eV,
        opt_constants={float(energy_eV): [0.0, 0.0, 0.0, 0.0] for energy_eV in energies_eV},
        name="vacuum",
    )
    config = {
        "CaseType": 0,
        "MorphologyType": 0,
        "Energies": energies_eV,
        "EAngleRotation": [0.0, 0.0, 0.0],
        "RotMask": 1,
        "WindowingType": 0,
        "AlgorithmType": 0,
        "ReferenceFrame": 1,
        "EwaldsInterpolation": 1,
    }
    morph = Morphology(
        2,
        materials={1: mat1, 2: mat2},
        PhysSize=PHYS_SIZE_NM,
        config=config,
        backend=str(runtime_kwargs["backend"]),
        backend_options=dict(runtime_kwargs["backend_options"]),
        resident_mode=runtime_kwargs["resident_mode"],
        input_policy=str(runtime_kwargs["input_policy"]),
        ownership_policy=runtime_kwargs["ownership_policy"],
        create_cy_object=True,
    )
    try:
        data = morph.run(stdout=False, stderr=False, return_xarray=True)
        if data is None:
            raise AssertionError("Analytical sphere backend run returned no scattering data.")
        return data.copy(deep=True)
    finally:
        morph.release_runtime()
        del morph
        del sphere_vfrac, vacuum_vfrac
        _release_runtime_memory()


def _pyhyper_iq_by_energy(scattering) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    from PyHyperScattering.integrate import WPIntegrator

    integrator = WPIntegrator(use_chunked_processing=False)
    remeshed = integrator.integrateImageStack(scattering)

    if "energy" not in remeshed.dims or "chi" not in remeshed.dims:
        raise AssertionError("PyHyperScattering output missing expected energy/chi dimensions.")
    qdim = next((d for d in remeshed.dims if d == "q" or d.startswith("q")), None)
    if qdim is None:
        raise AssertionError("PyHyperScattering output missing q dimension.")

    iq_by_energy = {}
    for energy in np.asarray(remeshed.coords["energy"].values, dtype=np.float64):
        iq = remeshed.sel(energy=float(energy)).mean("chi")
        q = np.asarray(iq.coords[qdim].values, dtype=np.float64)
        sim_iq = np.asarray(iq.values, dtype=np.float64)
        iq_by_energy[float(energy)] = (q, sim_iq)
    return iq_by_energy


def _analytic_sphere_form_factor_iq(q: np.ndarray, diameter_nm: float) -> np.ndarray:
    radius = float(diameter_nm) / 2.0
    qr = np.asarray(q, dtype=np.float64) * radius
    f = np.ones_like(qr)
    nz = np.abs(qr) > 1e-12
    qr_nz = qr[nz]
    f[nz] = 3.0 * (np.sin(qr_nz) - qr_nz * np.cos(qr_nz)) / (qr_nz ** 3)
    return f * f


def _analytic_sphere_form_factor_binned_iq(
    q_centers: np.ndarray,
    diameter_nm: float,
    oversample: int = 64,
) -> np.ndarray:
    q_centers = np.asarray(q_centers, dtype=np.float64)
    if q_centers.ndim != 1 or q_centers.size == 0:
        raise AssertionError("q_centers must be a non-empty 1D array.")
    if q_centers.size == 1:
        return _analytic_sphere_form_factor_iq(q_centers, diameter_nm)

    dq = np.diff(q_centers)
    edges = np.empty(q_centers.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (q_centers[:-1] + q_centers[1:])
    edges[0] = max(0.0, q_centers[0] - 0.5 * dq[0])
    edges[-1] = q_centers[-1] + 0.5 * dq[-1]

    binned = np.empty_like(q_centers)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if hi <= lo:
            binned[i] = _analytic_sphere_form_factor_iq(np.asarray([q_centers[i]]), diameter_nm)[0]
            continue
        q_samples = np.linspace(lo, hi, oversample, dtype=np.float64)
        iq_samples = _analytic_sphere_form_factor_iq(q_samples, diameter_nm)
        weights = q_samples
        denom = np.trapezoid(weights, q_samples)
        if denom <= 0.0:
            binned[i] = float(np.mean(iq_samples))
        else:
            binned[i] = np.trapezoid(iq_samples * weights, q_samples) / denom
    return binned


def _flat_detector_q_geometry(energy_eV: float) -> tuple[np.ndarray, np.ndarray, float]:
    wavelength_nm = 1239.84197 / float(energy_eV)
    k = 2.0 * np.pi / wavelength_nm
    qy = _cyrsoxs_detector_axis(SHAPE[1], PHYS_SIZE_NM)
    qx = _cyrsoxs_detector_axis(SHAPE[2], PHYS_SIZE_NM)
    return qy, qx, k


def _flat_detector_qmag(qy: np.ndarray, qx: np.ndarray, k: float) -> np.ndarray:
    qperp = np.hypot(qy[:, None], qx[None, :])
    val = k * k - qperp * qperp
    qz = np.full_like(qperp, np.nan, dtype=np.float64)
    qmag = np.full_like(qperp, np.nan, dtype=np.float64)
    valid = val >= 0.0
    qz[valid] = -k + np.sqrt(val[valid])
    qmag[valid] = np.sqrt(qperp[valid] * qperp[valid] + qz[valid] * qz[valid])
    return qmag


def _flat_detector_analytic_image(
    diameter_nm: float,
    energy_eV: float,
    oversample: int = FLAT_DETECTOR_OVERSAMPLE,
) -> xr.DataArray:
    oversample = int(oversample)
    qy, qx, k = _flat_detector_q_geometry(energy_eV)

    if oversample == 1:
        qmag = _flat_detector_qmag(qy, qx, k)
        image = np.full((qy.size, qx.size), np.nan, dtype=np.float64)
        valid = np.isfinite(qmag)
        valid[-1, :] = False
        valid[:, -1] = False
        image[valid] = _analytic_sphere_form_factor_iq(qmag[valid], diameter_nm)
    else:
        dqy = float(qy[1] - qy[0])
        dqx = float(qx[1] - qx[0])
        offsets_y = ((np.arange(oversample, dtype=np.float64) + 0.5) / oversample - 0.5) * dqy
        offsets_x = ((np.arange(oversample, dtype=np.float64) + 0.5) / oversample - 0.5) * dqx
        image = np.full((qy.size, qx.size), np.nan, dtype=np.float64)
        chunk = 64
        for y0 in range(0, qy.size, chunk):
            y1 = min(qy.size, y0 + chunk)
            qy_chunk = qy[y0:y1]
            qy_sub = qy_chunk[:, None] + offsets_y[None, :]
            qx_sub = qx[:, None] + offsets_x[None, :]
            qmag = _flat_detector_qmag(
                qy_sub.reshape(-1),
                qx_sub.reshape(-1),
                k,
            ).reshape(qy_chunk.size, oversample, qx.size, oversample)
            iq_sub = np.full_like(qmag, np.nan, dtype=np.float64)
            valid = np.isfinite(qmag)
            iq_sub[valid] = _analytic_sphere_form_factor_iq(qmag[valid], diameter_nm)
            valid_count = np.count_nonzero(valid, axis=(1, 3))
            summed = np.nansum(iq_sub, axis=(1, 3))
            image_chunk = np.full((qy_chunk.size, qx.size), np.nan, dtype=np.float64)
            nonzero = valid_count > 0
            image_chunk[nonzero] = summed[nonzero] / valid_count[nonzero]
            image[y0:y1, :] = image_chunk
        image[-1, :] = np.nan
        image[:, -1] = np.nan

    return xr.DataArray(
        image[np.newaxis, :, :],
        dims=["energy", "qy", "qx"],
        coords={"energy": [float(energy_eV)], "qy": qy, "qx": qx},
    )


def _normalize_to_first_q_gt_zero(
    q: np.ndarray,
    sim_iq: np.ndarray,
    ref_iq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    candidates = np.flatnonzero(
        np.logical_and.reduce(
            [
                q > 0.0,
                np.isfinite(sim_iq),
                np.isfinite(ref_iq),
                sim_iq > 0.0,
                ref_iq > 0.0,
            ]
        )
    )
    if candidates.size == 0:
        raise AssertionError("No valid q>0 normalization point found.")
    i0 = int(candidates[0])
    return sim_iq / sim_iq[i0], ref_iq / ref_iq[i0], float(q[i0])


def _pointwise_metrics(q: np.ndarray, sim_norm: np.ndarray, ref_norm: np.ndarray) -> dict[str, float]:
    mask = np.logical_and.reduce(
        [
            q >= Q_POINTWISE_MIN,
            q <= Q_POINTWISE_MAX,
            np.isfinite(sim_norm),
            np.isfinite(ref_norm),
            sim_norm > MIN_SIGNAL_FOR_LOG,
            ref_norm > MIN_SIGNAL_FOR_LOG,
        ]
    )
    if int(np.count_nonzero(mask)) < 20:
        raise AssertionError("Insufficient q points for pointwise sphere metric.")
    resid = np.log10(sim_norm[mask]) - np.log10(ref_norm[mask])
    return {
        "rms_log": float(np.sqrt(np.mean(resid * resid))),
        "p95_log_abs": float(np.percentile(np.abs(resid), 95)),
        "max_log_abs": float(np.max(np.abs(resid))),
        "n_comp": int(np.count_nonzero(mask)),
    }


def _parabolic_extremum_q(q: np.ndarray, y: np.ndarray, idx: int, mode: str) -> float:
    x = q[idx - 1 : idx + 2]
    vals = y[idx - 1 : idx + 2]
    if x.size != 3 or np.any(~np.isfinite(vals)):
        return float(q[idx])
    coeff = np.polyfit(x, vals, deg=2)
    a, b = float(coeff[0]), float(coeff[1])
    if mode == "min" and a <= 0.0:
        return float(q[idx])
    if mode == "max" and a >= 0.0:
        return float(q[idx])
    xv = -b / (2.0 * a)
    if xv < float(x[0]) or xv > float(x[-1]):
        return float(q[idx])
    return float(xv)


def _find_all_minima(q: np.ndarray, iq: np.ndarray, qmin: float, qmax: float) -> np.ndarray:
    mask = np.logical_and.reduce([q >= qmin, q <= qmax, np.isfinite(iq), iq > 0.0])
    q_cut = np.asarray(q[mask], dtype=np.float64)
    iq_cut = np.asarray(iq[mask], dtype=np.float64)
    minima = []
    for i in range(1, len(iq_cut) - 1):
        if iq_cut[i] <= iq_cut[i - 1] and iq_cut[i] <= iq_cut[i + 1]:
            minima.append(_parabolic_extremum_q(q_cut, iq_cut, i, mode="min"))
    return np.asarray(minima, dtype=np.float64)


def _minima_alignment_metrics(q: np.ndarray, sim_norm: np.ndarray, ref_norm: np.ndarray) -> dict[str, float]:
    sim_minima = _find_all_minima(q, sim_norm, Q_EXTREMA_MIN, Q_EXTREMA_MAX)
    ref_minima = _find_all_minima(q, ref_norm, Q_EXTREMA_MIN, Q_EXTREMA_MAX)
    n_match = min(sim_minima.size, ref_minima.size)
    if n_match == 0:
        raise AssertionError("No matched minima found in sphere comparison range.")
    delta = sim_minima[:n_match] - ref_minima[:n_match]
    return {
        "n_sim": int(sim_minima.size),
        "n_ref": int(ref_minima.size),
        "n_match": int(n_match),
        "mae_abs_dq": float(np.mean(np.abs(delta))),
        "rmse_abs_dq": float(np.sqrt(np.mean(delta * delta))),
        "max_abs_dq": float(np.max(np.abs(delta))),
        "mean_signed_dq": float(np.mean(delta)),
        "sim_minima": sim_minima,
        "ref_minima": ref_minima,
    }


def _write_dev_plot(
    diameter_nm: float,
    energy_eV: float,
    q: np.ndarray,
    flat_norm: np.ndarray,
    direct_norm: np.ndarray,
    sim_norm_by_superres: dict[int, np.ndarray],
    point_metrics_by_superres: dict[int, dict[str, float]],
    flat_minima_metrics_by_superres: dict[int, dict[str, float]],
    direct_minima_metrics_by_superres: dict[int, dict[str, float]],
    timing_by_superres: dict[int, dict[str, float]],
    path_id: str,
) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.0, 8.2), sharex=True)

    plot_mask = np.logical_and.reduce(
        [
            q >= Q_PLOT_MIN,
            q <= Q_PLOT_MAX,
            np.isfinite(flat_norm),
            np.isfinite(direct_norm),
            flat_norm > 0.0,
            direct_norm > 0.0,
        ]
    )
    for superresolution in SUPERRESOLUTIONS:
        sim_norm = sim_norm_by_superres[superresolution]
        plot_mask = np.logical_and(plot_mask, np.isfinite(sim_norm) & (sim_norm > 0.0))

    ax0.plot(q[plot_mask], flat_norm[plot_mask], color="black", linewidth=2.1, label="Analytical flat-detector (box averaged)")
    ax0.plot(
        q[plot_mask],
        direct_norm[plot_mask],
        color="#7f7f7f",
        linewidth=1.6,
        linestyle="--",
        alpha=0.9,
        label="Analytical direct I(q)",
    )
    for superresolution in SUPERRESOLUTIONS:
        ax0.plot(
            q[plot_mask],
            sim_norm_by_superres[superresolution][plot_mask],
            color=SUPERRES_COLORS[superresolution],
            linewidth=1.45,
            alpha=0.55,
            label=f"Simulation sr={superresolution}",
        )
    ax0.set_yscale("log")
    ax0.set_xlim(Q_PLOT_MIN, Q_PLOT_MAX)
    ax0.set_ylabel("Normalized I(q)")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best", fontsize=8.5)

    for superresolution in SUPERRESOLUTIONS:
        sim_norm = sim_norm_by_superres[superresolution]
        mask = np.logical_and.reduce(
            [
                q >= Q_POINTWISE_MIN,
                q <= Q_POINTWISE_MAX,
                np.isfinite(sim_norm),
                np.isfinite(flat_norm),
                sim_norm > MIN_SIGNAL_FOR_LOG,
                flat_norm > MIN_SIGNAL_FOR_LOG,
            ]
        )
        resid = np.log10(sim_norm[mask]) - np.log10(flat_norm[mask])
        ax1.plot(
            q[mask],
            resid,
            color=SUPERRES_COLORS[superresolution],
            linewidth=1.1,
            alpha=0.65,
            label=f"sr={superresolution}",
        )
    ax1.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_xlim(Q_PLOT_MIN, Q_PLOT_MAX)
    ax1.set_xlabel(r"q [nm$^{-1}$]")
    ax1.set_ylabel(r"$\Delta \log_{10} I$")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best", fontsize=8.5)

    note_lines = [
        (
            f"diameter={diameter_nm:.1f} nm, energy={energy_eV:.1f} eV, "
            f"shape={SHAPE[0]}x{SHAPE[1]}x{SHAPE[2]}, PhysSize={PHYS_SIZE_NM:.2f} nm"
        ),
        (
            f"PyHyper q, minima metric uses all minima in [{Q_EXTREMA_MIN:.2f}, {Q_EXTREMA_MAX:.2f}] nm^-1; "
            f"point metric uses q in [{Q_POINTWISE_MIN:.2f}, {Q_POINTWISE_MAX:.2f}] nm^-1"
        ),
        f"flat-detector detector-pixel box average oversample={FLAT_DETECTOR_OVERSAMPLE}x",
    ]
    for superresolution in SUPERRESOLUTIONS:
        point = point_metrics_by_superres[superresolution]
        flat_min = flat_minima_metrics_by_superres[superresolution]
        direct_min = direct_minima_metrics_by_superres[superresolution]
        ratio = direct_min["mae_abs_dq"] / flat_min["mae_abs_dq"]
        timing = timing_by_superres[superresolution]
        note_lines.append(
            (
                f"sr={superresolution}: rms_log={point['rms_log']:.4f}; "
                f"flat minima mae={flat_min['mae_abs_dq']:.5f} (n={flat_min['n_match']}); "
                f"direct minima mae={direct_min['mae_abs_dq']:.5f} (n={direct_min['n_match']}); "
                f"ratio={ratio:.2f}; sim={timing['sim_seconds']:.2f}s iq={timing['iq_seconds']:.2f}s"
            )
        )
    fig.suptitle("Analytical Sphere Flat-Detector Comparison")
    fig.text(
        0.01,
        0.01,
        "\n".join(note_lines),
        fontsize=8.0,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )
    fig.tight_layout(rect=[0, 0.18, 1, 0.97])
    out = PLOT_DIR / f"{path_id}__sphere_ff_flatdet_d{int(round(diameter_nm))}_e{energy_eV:.1f}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _evaluate_geometry_case(diameter_nm: float, nrss_path: ComputationPath) -> dict[str, object]:
    iq_by_superres = {}
    timing_by_superres = {}
    runtime_kwargs = _path_runtime_kwargs(nrss_path)
    for superresolution in SUPERRESOLUTIONS:
        sim_t0 = perf_counter()
        data = _run_sphere_backend(
            diameter_nm=diameter_nm,
            superresolution=superresolution,
            energies_eV=[ENERGY_EV],
            runtime_kwargs=runtime_kwargs,
        )
        sim_seconds = perf_counter() - sim_t0
        iq_t0 = perf_counter()
        iq_by_superres[superresolution] = _pyhyper_iq_by_energy(data)
        iq_seconds = perf_counter() - iq_t0
        timing_by_superres[superresolution] = {
            "sim_seconds": float(sim_seconds),
            "iq_seconds": float(iq_seconds),
        }
        del data
        _release_runtime_memory()

    q_ref, sim_iq_assert = iq_by_superres[ASSERT_SUPERRESOLUTION][float(ENERGY_EV)]

    direct_ref_iq = _analytic_sphere_form_factor_binned_iq(q_centers=q_ref, diameter_nm=diameter_nm)
    sim_norm_assert, direct_norm, _ = _normalize_to_first_q_gt_zero(q_ref, sim_iq_assert, direct_ref_iq)
    direct_point_metrics = _pointwise_metrics(q_ref, sim_norm_assert, direct_norm)
    direct_minima_metrics = _minima_alignment_metrics(q_ref, sim_norm_assert, direct_norm)

    flat_image = _flat_detector_analytic_image(
        diameter_nm=diameter_nm,
        energy_eV=ENERGY_EV,
        oversample=FLAT_DETECTOR_OVERSAMPLE,
    )
    q_flat, flat_ref_iq = _pyhyper_iq_by_energy(flat_image)[float(ENERGY_EV)]
    if not np.allclose(q_ref, q_flat, atol=1e-12, rtol=0.0):
        raise AssertionError("Flat-detector analytical q grid does not match simulation q grid.")

    _, flat_norm, _ = _normalize_to_first_q_gt_zero(q_ref, sim_iq_assert, flat_ref_iq)

    sim_norm_by_superres = {}
    point_metrics_by_superres = {}
    flat_minima_metrics_by_superres = {}
    direct_minima_metrics_by_superres = {}
    summary_lines = []

    for superresolution in SUPERRESOLUTIONS:
        q_sr, sim_iq_sr = iq_by_superres[superresolution][float(ENERGY_EV)]
        if not np.allclose(q_sr, q_ref, atol=1e-12, rtol=0.0):
            raise AssertionError(f"q-grid mismatch between sr={ASSERT_SUPERRESOLUTION} and sr={superresolution}.")

        sim_norm_sr, _, _ = _normalize_to_first_q_gt_zero(q_ref, sim_iq_sr, flat_ref_iq)
        sim_norm_by_superres[superresolution] = sim_norm_sr

        point_metrics_by_superres[superresolution] = _pointwise_metrics(q_ref, sim_norm_sr, flat_norm)
        flat_minima_metrics_by_superres[superresolution] = _minima_alignment_metrics(q_ref, sim_norm_sr, flat_norm)
        direct_minima_metrics_by_superres[superresolution] = _minima_alignment_metrics(q_ref, sim_norm_sr, direct_norm)

        flat_mae = flat_minima_metrics_by_superres[superresolution]["mae_abs_dq"]
        direct_mae = direct_minima_metrics_by_superres[superresolution]["mae_abs_dq"]
        summary_lines.append(
            f"d={diameter_nm:.0f} sr={superresolution}: rms={point_metrics_by_superres[superresolution]['rms_log']:.4f}, "
            f"flat_min_mae={flat_mae:.5f}, direct_min_mae={direct_mae:.5f}, ratio={direct_mae / flat_mae:.2f}"
        )

    return {
        "q_ref": q_ref,
        "flat_norm": flat_norm,
        "direct_norm": direct_norm,
        "sim_norm_by_superres": sim_norm_by_superres,
        "point_metrics_by_superres": point_metrics_by_superres,
        "flat_minima_metrics_by_superres": flat_minima_metrics_by_superres,
        "direct_minima_metrics_by_superres": direct_minima_metrics_by_superres,
        "direct_point_metrics": direct_point_metrics,
        "direct_minima_metrics": direct_minima_metrics,
        "timing_by_superres": timing_by_superres,
        "summary_lines": summary_lines,
    }


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.physics_validation
@pytest.mark.toolchain_validation
@pytest.mark.parametrize("diameter_nm", DIAMETERS_NM, ids=["dia70", "dia128"])
def test_analytical_spherical_form_factor_pybind(diameter_nm: float, nrss_path: ComputationPath):
    """Validate flat-detector sphere form-factor agreement and minima alignment through the pybind-to-PyHyper workflow."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for analytical sphere form-factor test.")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    result = _evaluate_geometry_case(diameter_nm, nrss_path)

    if WRITE_VALIDATION_PLOTS:
        _write_dev_plot(
            diameter_nm=diameter_nm,
            energy_eV=ENERGY_EV,
            q=result["q_ref"],
            flat_norm=result["flat_norm"],
            direct_norm=result["direct_norm"],
            sim_norm_by_superres=result["sim_norm_by_superres"],
            point_metrics_by_superres=result["point_metrics_by_superres"],
            flat_minima_metrics_by_superres=result["flat_minima_metrics_by_superres"],
            direct_minima_metrics_by_superres=result["direct_minima_metrics_by_superres"],
            timing_by_superres=result["timing_by_superres"],
            path_id=nrss_path.id,
        )

    print("direct_point_metrics", result["direct_point_metrics"])
    print("direct_minima_metrics", {k: v for k, v in result["direct_minima_metrics"].items() if not isinstance(v, np.ndarray)})
    for line in result["summary_lines"]:
        print(line)

    thresholds = GEOMETRY_THRESHOLDS_BY_DIAMETER[float(diameter_nm)]
    sr1_point = result["point_metrics_by_superres"][ASSERT_SUPERRESOLUTION]
    sr1_flat_min = result["flat_minima_metrics_by_superres"][ASSERT_SUPERRESOLUTION]
    sr1_direct_min = result["direct_minima_metrics_by_superres"][ASSERT_SUPERRESOLUTION]
    sr1_min_ratio = sr1_direct_min["mae_abs_dq"] / sr1_flat_min["mae_abs_dq"]
    sr1_point_ratio = result["direct_point_metrics"]["rms_log"] / sr1_point["rms_log"]

    assert sr1_point["rms_log"] <= thresholds["sr1_rms_log_max"]
    assert sr1_point["p95_log_abs"] <= thresholds["sr1_p95_log_abs_max"]
    assert sr1_flat_min["mae_abs_dq"] <= thresholds["sr1_flat_min_mae_max"]
    assert sr1_flat_min["rmse_abs_dq"] <= thresholds["sr1_flat_min_rmse_max"]
    assert sr1_min_ratio >= thresholds["sr1_min_ratio_min"]
    assert sr1_point_ratio >= thresholds["sr1_point_ratio_min"]

    for superresolution in SUPERRESOLUTIONS:
        flat_point = result["point_metrics_by_superres"][superresolution]
        flat_min = result["flat_minima_metrics_by_superres"][superresolution]
        direct_min = result["direct_minima_metrics_by_superres"][superresolution]
        assert flat_point["rms_log"] <= result["direct_point_metrics"]["rms_log"] + 1e-12
        assert flat_min["mae_abs_dq"] < direct_min["mae_abs_dq"]
