import gc
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import CyRSoXS as cy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from scipy.special import j1


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = REPO_ROOT / "test-reports" / "analytical-2d-disk-dev"
WRITE_VALIDATION_PLOTS = os.environ.get("NRSS_WRITE_VALIDATION_PLOTS", "").strip() == "1"

SHAPE = (1, 2048, 2048)
PHYS_SIZE_NM = 1.0
ENERGY_EV = 285.0
DIAMETERS_NM = [70.0, 128.0]
ASSERT_SUPERRESOLUTION = 1
SUPERRESOLUTIONS = [1]

Q_POINTWISE_MIN = 0.06
Q_POINTWISE_MAX = 1.0
Q_EXTREMA_MIN = 0.05
Q_EXTREMA_MAX = 1.0
Q_PLOT_MIN = 0.0
Q_PLOT_MAX = 1.0
MIN_SIGNAL_FOR_LOG = 1e-5
GEOMETRY_THRESHOLDS_BY_DIAMETER = {
    70.0: {
        "sr1_rms_log_max": 0.090,
        "sr1_p95_log_abs_max": 0.110,
        "sr1_min_mae_max": 0.00025,
        "sr1_min_rmse_max": 0.00030,
    },
    128.0: {
        "sr1_rms_log_max": 0.100,
        "sr1_p95_log_abs_max": 0.180,
        "sr1_min_mae_max": 0.00055,
        "sr1_min_rmse_max": 0.00065,
    },
}

SUPERRES_COLORS = {
    1: "#1f77b4",
}

DISK_OC = {
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


def _superresolved_disk_vfrac(diameter_nm: float, superresolution: int) -> np.ndarray:
    if int(superresolution) < 1:
        raise AssertionError("superresolution must be >= 1.")
    superresolution = int(superresolution)
    _, ny, nx = SHAPE
    radius_vox = float(diameter_nm) / (2.0 * PHYS_SIZE_NM)
    cy0 = (ny - 1) / 2.0
    cx0 = (nx - 1) / 2.0

    pad = radius_vox + 2.0
    y0 = max(0, int(np.floor(cy0 - pad)))
    y1 = min(ny, int(np.ceil(cy0 + pad)) + 1)
    x0 = max(0, int(np.floor(cx0 - pad)))
    x1 = min(nx, int(np.ceil(cx0 + pad)) + 1)

    yy, xx = np.ogrid[y0:y1, x0:x1]
    if superresolution == 1:
        dy = yy.astype(np.float32) - np.float32(cy0)
        dx = xx.astype(np.float32) - np.float32(cx0)
        dist2 = dx * dx + dy * dy
        local_vfrac = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        local_vfrac[dist2 <= np.float32(radius_vox * radius_vox)] = 1.0
    else:
        ly = y1 - y0
        lx = x1 - x0
        ss = superresolution
        y_hr, x_hr = np.ogrid[0 : ly * ss, 0 : lx * ss]
        dy = (
            y0
            + y_hr.astype(np.float32) / np.float32(ss)
            + np.float32(0.5 / ss - 0.5)
            - np.float32(cy0)
        )
        dx = (
            x0
            + x_hr.astype(np.float32) / np.float32(ss)
            + np.float32(0.5 / ss - 0.5)
            - np.float32(cx0)
        )
        highres_mask = (dx * dx + dy * dy) <= np.float32(radius_vox * radius_vox)
        local_vfrac = highres_mask.reshape(ly, ss, lx, ss).mean(axis=(1, 3), dtype=np.float32)

    vfrac = np.zeros(SHAPE, dtype=np.float32)
    vfrac[0, y0:y1, x0:x1] = local_vfrac
    return vfrac


def _disk_and_vacuum_vfrac(diameter_nm: float, superresolution: int) -> tuple[np.ndarray, np.ndarray]:
    disk = _superresolved_disk_vfrac(diameter_nm=diameter_nm, superresolution=superresolution)
    vacuum = (1.0 - disk).astype(np.float32)
    return disk, vacuum


def _scattering_to_xarray(scattering_pattern, energies_eV: list[float]) -> xr.DataArray:
    scattering_data = scattering_pattern.writeAllToNumpy(kID=0)
    qy = _cyrsoxs_detector_axis(SHAPE[1], PHYS_SIZE_NM)
    qx = _cyrsoxs_detector_axis(SHAPE[2], PHYS_SIZE_NM)
    return xr.DataArray(
        scattering_data,
        dims=["energy", "qy", "qx"],
        coords={"energy": list(map(float, energies_eV)), "qy": qy, "qx": qx},
    )


def _run_disk_pybind(
    diameter_nm: float,
    superresolution: int,
    energies_eV: list[float],
) -> xr.DataArray:
    disk_vfrac, vacuum_vfrac = _disk_and_vacuum_vfrac(
        diameter_nm=diameter_nm,
        superresolution=superresolution,
    )

    input_data = cy.InputData(NumMaterial=2)
    input_data.setEnergies(list(map(float, energies_eV)))
    input_data.setERotationAngle(StartAngle=0.0, EndAngle=0.0, IncrementAngle=0.0)
    input_data.setPhysSize(PHYS_SIZE_NM)
    input_data.setDimensions(SHAPE, cy.MorphologyOrder.ZYX)
    input_data.setCaseType(cy.CaseType.Default)
    input_data.setMorphologyType(cy.MorphologyType.EulerAngles)
    input_data.setAlgorithm(AlgorithmID=0, MaxStreams=1)
    input_data.interpolationType = cy.InterpolationType.Linear
    input_data.windowingType = cy.FFTWindowing.NoPadding
    input_data.rotMask = True
    input_data.referenceFrame = 1
    if not input_data.validate():
        raise AssertionError("CyRSoXS InputData validation failed.")

    optical_constants = cy.RefractiveIndex(input_data)
    for energy_eV in energies_eV:
        optical_constants.addData(
            OpticalConstants=[list(DISK_OC[float(energy_eV)]), [0.0, 0.0, 0.0, 0.0]],
            Energy=float(energy_eV),
        )
    if not optical_constants.validate():
        raise AssertionError("CyRSoXS optical-constants validation failed.")

    voxel_data = cy.VoxelData(InputData=input_data)
    voxel_data.addVoxelData(Vfrac=disk_vfrac, MaterialID=1)
    voxel_data.addVoxelData(Vfrac=vacuum_vfrac, MaterialID=2)
    if not voxel_data.validate():
        raise AssertionError("CyRSoXS VoxelData validation failed.")

    scattering_pattern = cy.ScatteringPattern(InputData=input_data)
    with cy.ostream_redirect(stdout=False, stderr=False):
        cy.launch(
            VoxelData=voxel_data,
            RefractiveIndexData=optical_constants,
            InputData=input_data,
            ScatteringPattern=scattering_pattern,
        )

    data = _scattering_to_xarray(scattering_pattern, energies_eV=energies_eV).copy(deep=True)

    del scattering_pattern, voxel_data, optical_constants, input_data
    del disk_vfrac, vacuum_vfrac
    _release_runtime_memory()
    return data


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


def _analytic_disk_form_factor_iq(q: np.ndarray, diameter_nm: float) -> np.ndarray:
    radius = float(diameter_nm) / 2.0
    qr = np.asarray(q, dtype=np.float64) * radius
    f = np.ones_like(qr)
    nz = np.abs(qr) > 1e-12
    qr_nz = qr[nz]
    f[nz] = 2.0 * j1(qr_nz) / qr_nz
    return f * f


def _analytic_disk_form_factor_binned_iq(
    q_centers: np.ndarray,
    diameter_nm: float,
    oversample: int = 64,
) -> np.ndarray:
    q_centers = np.asarray(q_centers, dtype=np.float64)
    if q_centers.ndim != 1 or q_centers.size == 0:
        raise AssertionError("q_centers must be a non-empty 1D array.")
    if q_centers.size == 1:
        return _analytic_disk_form_factor_iq(q_centers, diameter_nm)

    dq = np.diff(q_centers)
    edges = np.empty(q_centers.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (q_centers[:-1] + q_centers[1:])
    edges[0] = max(0.0, q_centers[0] - 0.5 * dq[0])
    edges[-1] = q_centers[-1] + 0.5 * dq[-1]

    binned = np.empty_like(q_centers)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if hi <= lo:
            binned[i] = _analytic_disk_form_factor_iq(np.asarray([q_centers[i]]), diameter_nm)[0]
            continue
        q_samples = np.linspace(lo, hi, oversample, dtype=np.float64)
        iq_samples = _analytic_disk_form_factor_iq(q_samples, diameter_nm)
        weights = q_samples
        denom = np.trapezoid(weights, q_samples)
        if denom <= 0.0:
            binned[i] = float(np.mean(iq_samples))
        else:
            binned[i] = np.trapezoid(iq_samples * weights, q_samples) / denom
    return binned


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
        raise AssertionError("Insufficient q points for pointwise 2D disk metric.")
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
        raise AssertionError("No matched minima found in 2D disk comparison range.")
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
    ref_norm: np.ndarray,
    sim_norm_by_superres: dict[int, np.ndarray],
    point_metrics_by_superres: dict[int, dict[str, float]],
    minima_metrics_by_superres: dict[int, dict[str, float]],
    timing_by_superres: dict[int, dict[str, float]],
) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.0, 8.2), sharex=True)

    plot_mask = np.logical_and.reduce(
        [
            q >= Q_PLOT_MIN,
            q <= Q_PLOT_MAX,
            np.isfinite(ref_norm),
            ref_norm > 0.0,
        ]
    )
    for superresolution in SUPERRESOLUTIONS:
        sim_norm = sim_norm_by_superres[superresolution]
        plot_mask = np.logical_and(plot_mask, np.isfinite(sim_norm) & (sim_norm > 0.0))

    ax0.plot(
        q[plot_mask],
        ref_norm[plot_mask],
        color="black",
        linewidth=2.1,
        label="Analytical q-bin averaged",
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
                np.isfinite(ref_norm),
                sim_norm > MIN_SIGNAL_FOR_LOG,
                ref_norm > MIN_SIGNAL_FOR_LOG,
            ]
        )
        resid = np.log10(sim_norm[mask]) - np.log10(ref_norm[mask])
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
            f"PyHyper q, analytical reference is direct 2D disk I(q); "
            f"point metric uses q in [{Q_POINTWISE_MIN:.2f}, {Q_POINTWISE_MAX:.2f}] nm^-1"
        ),
        f"minima metric uses all minima in [{Q_EXTREMA_MIN:.2f}, {Q_EXTREMA_MAX:.2f}] nm^-1",
    ]
    for superresolution in SUPERRESOLUTIONS:
        point = point_metrics_by_superres[superresolution]
        minima = minima_metrics_by_superres[superresolution]
        timing = timing_by_superres[superresolution]
        note_lines.append(
            (
                f"sr={superresolution}: rms_log={point['rms_log']:.4f}; "
                f"minima mae={minima['mae_abs_dq']:.5f} rmse={minima['rmse_abs_dq']:.5f} "
                f"(n={minima['n_match']}); sim={timing['sim_seconds']:.2f}s iq={timing['iq_seconds']:.2f}s"
            )
        )
    fig.suptitle("Analytical 2D Disk Form-Factor Comparison")
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
    out = PLOT_DIR / f"disk_ff_2d_d{int(round(diameter_nm))}_e{energy_eV:.1f}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _evaluate_geometry_case(diameter_nm: float) -> dict[str, object]:
    iq_by_superres = {}
    timing_by_superres = {}
    for superresolution in SUPERRESOLUTIONS:
        sim_t0 = perf_counter()
        data = _run_disk_pybind(
            diameter_nm=diameter_nm,
            superresolution=superresolution,
            energies_eV=[ENERGY_EV],
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
    ref_iq = _analytic_disk_form_factor_binned_iq(q_centers=q_ref, diameter_nm=diameter_nm)
    sim_norm_assert, ref_norm, _ = _normalize_to_first_q_gt_zero(q_ref, sim_iq_assert, ref_iq)
    assert_metrics = _pointwise_metrics(q_ref, sim_norm_assert, ref_norm)
    assert_minima_metrics = _minima_alignment_metrics(q_ref, sim_norm_assert, ref_norm)

    sim_norm_by_superres = {}
    point_metrics_by_superres = {}
    minima_metrics_by_superres = {}
    summary_lines = []

    for superresolution in SUPERRESOLUTIONS:
        q_sr, sim_iq_sr = iq_by_superres[superresolution][float(ENERGY_EV)]
        if not np.allclose(q_sr, q_ref, atol=1e-12, rtol=0.0):
            raise AssertionError(f"q-grid mismatch between sr={ASSERT_SUPERRESOLUTION} and sr={superresolution}.")
        sim_norm_sr, _, _ = _normalize_to_first_q_gt_zero(q_ref, sim_iq_sr, ref_iq)
        sim_norm_by_superres[superresolution] = sim_norm_sr
        point_metrics_by_superres[superresolution] = _pointwise_metrics(q_ref, sim_norm_sr, ref_norm)
        minima_metrics_by_superres[superresolution] = _minima_alignment_metrics(q_ref, sim_norm_sr, ref_norm)
        summary_lines.append(
            f"d={diameter_nm:.0f} sr={superresolution}: "
            f"rms={point_metrics_by_superres[superresolution]['rms_log']:.4f}, "
            f"p95={point_metrics_by_superres[superresolution]['p95_log_abs']:.4f}, "
            f"min_mae={minima_metrics_by_superres[superresolution]['mae_abs_dq']:.5f}, "
            f"min_rmse={minima_metrics_by_superres[superresolution]['rmse_abs_dq']:.5f}"
        )

    return {
        "q_ref": q_ref,
        "ref_norm": ref_norm,
        "sim_norm_by_superres": sim_norm_by_superres,
        "point_metrics_by_superres": point_metrics_by_superres,
        "minima_metrics_by_superres": minima_metrics_by_superres,
        "assert_metrics": assert_metrics,
        "assert_minima_metrics": assert_minima_metrics,
        "timing_by_superres": timing_by_superres,
        "summary_lines": summary_lines,
    }


def _assert_geometry_case_result(diameter_nm: float, result: dict[str, object]) -> None:
    thresholds = GEOMETRY_THRESHOLDS_BY_DIAMETER[float(diameter_nm)]
    sr1_point = result["point_metrics_by_superres"][ASSERT_SUPERRESOLUTION]
    sr1_minima = result["minima_metrics_by_superres"][ASSERT_SUPERRESOLUTION]

    assert sr1_point["rms_log"] <= thresholds["sr1_rms_log_max"]
    assert sr1_point["p95_log_abs"] <= thresholds["sr1_p95_log_abs_max"]
    assert sr1_minima["mae_abs_dq"] <= thresholds["sr1_min_mae_max"]
    assert sr1_minima["rmse_abs_dq"] <= thresholds["sr1_min_rmse_max"]


def _run_validated_geometry_case(diameter_nm: float) -> None:
    result = _evaluate_geometry_case(diameter_nm)

    if WRITE_VALIDATION_PLOTS:
        _write_dev_plot(
            diameter_nm=diameter_nm,
            energy_eV=ENERGY_EV,
            q=result["q_ref"],
            ref_norm=result["ref_norm"],
            sim_norm_by_superres=result["sim_norm_by_superres"],
            point_metrics_by_superres=result["point_metrics_by_superres"],
            minima_metrics_by_superres=result["minima_metrics_by_superres"],
            timing_by_superres=result["timing_by_superres"],
        )

    print("assert_point_metrics", result["assert_metrics"])
    print(
        "assert_minima_metrics",
        {k: v for k, v in result["assert_minima_metrics"].items() if not isinstance(v, np.ndarray)},
    )
    for line in result["summary_lines"]:
        print(line)

    _assert_geometry_case_result(diameter_nm, result)


def _run_geometry_case_subprocess(diameter_nm: float) -> None:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    code = (
        "from tests.validation.test_analytical_2d_disk_form_factor import "
        f"_run_validated_geometry_case; _run_validated_geometry_case({float(diameter_nm)!r})"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    assert result.returncode == 0, (
        f"Isolated 2D analytical disk run failed for diameter {diameter_nm:.1f} nm "
        f"with exit code {result.returncode}."
    )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.physics_validation
@pytest.mark.toolchain_validation
@pytest.mark.parametrize("diameter_nm", DIAMETERS_NM, ids=["dia70", "dia128"])
def test_analytical_2d_disk_form_factor_pybind(diameter_nm: float):
    """Validate direct analytical 2D disk form-factor agreement and minima alignment through the pybind-to-PyHyper workflow."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for analytical 2D disk form-factor test.")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # Under a single visible GPU, the second large 2D analytical CyRSoXS run in the
    # same Python process can hit an upstream illegal-memory-access failure. Isolating
    # each diameter preserves the sphere-style harness while keeping the test stable.
    _run_geometry_case_subprocess(diameter_nm)
