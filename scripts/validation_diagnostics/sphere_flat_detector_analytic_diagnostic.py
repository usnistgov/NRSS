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

from NRSS.morphology import Material, Morphology


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = REPO_ROOT / "test-reports" / "sphere-flat-detector-dev"

SHAPE = (256, 1024, 1024)
PHYS_SIZE_NM = 1.0
DIAMETER_NM = 150.0
ENERGY_EV = 285.0
Q_ASSERT_MIN = 0.06
Q_ASSERT_MAX = 1.0
Q_PLOT_MIN = 0.0
Q_PLOT_MAX = 1.0
MIN_SIGNAL_FOR_LOG = 1e-5
FLAT_BIN_OVERSAMPLE = 8
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


@lru_cache(maxsize=1)
def _zero_field() -> np.ndarray:
    return np.zeros(SHAPE, dtype=np.float32)


def _cyrsoxs_detector_axis(n: int, phys_size_nm: float) -> np.ndarray:
    if int(n) < 2:
        raise AssertionError(f"CyRSoXS detector axis needs at least 2 points, got n={n}.")
    start = -np.pi / float(phys_size_nm)
    step = (2.0 * np.pi / float(phys_size_nm)) / float(int(n) - 1)
    return start + np.arange(int(n), dtype=np.float64) * step


def _with_cyrsoxs_detector_coords(scattering):
    qy = _cyrsoxs_detector_axis(int(scattering.sizes["qy"]), PHYS_SIZE_NM)
    qx = _cyrsoxs_detector_axis(int(scattering.sizes["qx"]), PHYS_SIZE_NM)
    return scattering.assign_coords(qy=qy, qx=qx)


def _sphere_and_vacuum_vfrac(diameter_nm: float) -> tuple[np.ndarray, np.ndarray]:
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
    dz = zz.astype(np.float32) - np.float32(cz)
    dy = yy.astype(np.float32) - np.float32(cy)
    dx = xx.astype(np.float32) - np.float32(cx)
    dist2 = dx * dx + dy * dy + dz * dz

    sphere = np.zeros(SHAPE, dtype=np.float32)
    local_sphere = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=np.float32)
    local_sphere[dist2 <= np.float32(radius_vox * radius_vox)] = 1.0
    sphere[z0:z1, y0:y1, x0:x1] = local_sphere
    vacuum = (1.0 - sphere).astype(np.float32)
    return sphere, vacuum


def _build_morphology() -> Morphology:
    sphere_vfrac, vacuum_vfrac = _sphere_and_vacuum_vfrac(DIAMETER_NM)
    zeros = _zero_field()
    sphere_oc = {float(e): list(SPHERE_OC[float(e)]) for e in [ENERGY_EV]}
    vacuum_oc = {float(ENERGY_EV): [0.0, 0.0, 0.0, 0.0]}

    mat_sphere = Material(
        materialID=1,
        Vfrac=sphere_vfrac,
        S=zeros,
        theta=zeros,
        psi=zeros,
        energies=[ENERGY_EV],
        opt_constants=sphere_oc,
        name="sphere",
    )
    mat_vacuum = Material(
        materialID=2,
        Vfrac=vacuum_vfrac,
        S=zeros,
        theta=zeros,
        psi=zeros,
        energies=[ENERGY_EV],
        opt_constants=vacuum_oc,
        name="vacuum",
    )

    config = {
        "CaseType": 0,
        "MorphologyType": 0,
        "Energies": [ENERGY_EV],
        "EAngleRotation": [0.0, 0.0, 0.0],
        "RotMask": 1,
        "WindowingType": 0,
        "AlgorithmType": 0,
        "ReferenceFrame": 1,
        "EwaldsInterpolation": 1,
    }

    morph = Morphology(
        2,
        materials={1: mat_sphere, 2: mat_vacuum},
        PhysSize=PHYS_SIZE_NM,
        config=config,
        create_cy_object=True,
    )
    morph.check_materials(quiet=True)
    morph.validate_all(quiet=True)
    return morph


def _pyhyper_iq(scattering) -> tuple[np.ndarray, np.ndarray]:
    from PyHyperScattering.integrate import WPIntegrator

    scattering = _with_cyrsoxs_detector_coords(scattering)
    integrator = WPIntegrator(use_chunked_processing=False)
    remeshed = integrator.integrateImageStack(scattering)
    if "energy" not in remeshed.dims or "chi" not in remeshed.dims:
        raise AssertionError("PyHyperScattering output missing expected dimensions.")
    qdim = next((d for d in remeshed.dims if d == "q" or d.startswith("q")), None)
    if qdim is None:
        raise AssertionError("PyHyperScattering output missing q dimension.")

    iq = remeshed.sel(energy=float(ENERGY_EV)).mean("chi")
    q = np.asarray(iq.coords[qdim].values, dtype=np.float64)
    sim_iq = np.asarray(iq.values, dtype=np.float64)
    return q, sim_iq


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


def _flat_detector_q_geometry() -> tuple[np.ndarray, np.ndarray, float]:
    wavelength_nm = 1239.84197 / float(ENERGY_EV)
    k = 2.0 * np.pi / wavelength_nm

    ny = SHAPE[1]
    nx = SHAPE[2]
    qy = _cyrsoxs_detector_axis(ny, PHYS_SIZE_NM)
    qx = _cyrsoxs_detector_axis(nx, PHYS_SIZE_NM)
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


def _flat_detector_analytic_image_point() -> xr.DataArray:
    qy, qx, k = _flat_detector_q_geometry()
    qmag = _flat_detector_qmag(qy, qx, k)

    image = np.full((qy.size, qx.size), np.nan, dtype=np.float64)
    valid = np.isfinite(qmag)
    valid[-1, :] = False
    valid[:, -1] = False
    image[valid] = _analytic_sphere_form_factor_iq(qmag[valid], DIAMETER_NM)

    return xr.DataArray(
        image[np.newaxis, :, :],
        dims=["energy", "qy", "qx"],
        coords={"energy": [float(ENERGY_EV)], "qy": qy, "qx": qx},
    )


def _flat_detector_analytic_image_binned(oversample: int = FLAT_BIN_OVERSAMPLE) -> xr.DataArray:
    oversample = int(oversample)
    if oversample < 1:
        raise AssertionError("oversample must be >= 1.")
    if oversample == 1:
        return _flat_detector_analytic_image_point()

    qy, qx, k = _flat_detector_q_geometry()
    dqy = float(qy[1] - qy[0])
    dqx = float(qx[1] - qx[0])
    offsets_y = ((np.arange(oversample, dtype=np.float64) + 0.5) / oversample - 0.5) * dqy
    offsets_x = ((np.arange(oversample, dtype=np.float64) + 0.5) / oversample - 0.5) * dqx

    image = np.full((qy.size, qx.size), np.nan, dtype=np.float64)
    chunk = 32
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
        iq_sub[valid] = _analytic_sphere_form_factor_iq(qmag[valid], DIAMETER_NM)
        image[y0:y1, :] = np.nanmean(iq_sub, axis=(1, 3))

    image[-1, :] = np.nan
    image[:, -1] = np.nan

    return xr.DataArray(
        image[np.newaxis, :, :],
        dims=["energy", "qy", "qx"],
        coords={"energy": [float(ENERGY_EV)], "qy": qy, "qx": qx},
    )


def _normalize_to_first_q_gt_zero(
    q: np.ndarray,
    sim_iq: np.ndarray,
    ana_iq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    candidates = np.flatnonzero(
        np.logical_and.reduce(
            [
                q > 0.0,
                np.isfinite(sim_iq),
                np.isfinite(ana_iq),
                sim_iq > 0.0,
                ana_iq > 0.0,
            ]
        )
    )
    if candidates.size == 0:
        raise AssertionError("No valid q>0 normalization point found.")
    i0 = int(candidates[0])
    return sim_iq / sim_iq[i0], ana_iq / ana_iq[i0], float(q[i0])


def _metrics(q: np.ndarray, sim_norm: np.ndarray, ana_norm: np.ndarray) -> dict[str, float]:
    mask = np.logical_and.reduce(
        [
            q >= Q_ASSERT_MIN,
            q <= Q_ASSERT_MAX,
            np.isfinite(sim_norm),
            np.isfinite(ana_norm),
            sim_norm > MIN_SIGNAL_FOR_LOG,
            ana_norm > MIN_SIGNAL_FOR_LOG,
        ]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = np.log10(sim_norm[mask]) - np.log10(ana_norm[mask])
    return {
        "rms_log": float(np.sqrt(np.mean(resid * resid))),
        "p95_log_abs": float(np.percentile(np.abs(resid), 95)),
        "max_log_abs": float(np.max(np.abs(resid))),
        "n_comp": int(np.count_nonzero(mask)),
    }


def _write_plot(
    q: np.ndarray,
    sim_norm: np.ndarray,
    ana_direct_norm: np.ndarray,
    ana_flat_point_norm: np.ndarray,
    ana_flat_bin_norm: np.ndarray,
    direct_metrics: dict[str, float],
    flat_point_metrics: dict[str, float],
    flat_bin_metrics: dict[str, float],
    build_seconds: float,
    run_seconds: float,
) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.0, 8.0), sharex=True)

    plot_mask = np.logical_and.reduce(
        [
            q >= Q_PLOT_MIN,
            q <= Q_PLOT_MAX,
            np.isfinite(sim_norm),
            np.isfinite(ana_direct_norm),
            np.isfinite(ana_flat_point_norm),
            np.isfinite(ana_flat_bin_norm),
            sim_norm > 0.0,
            ana_direct_norm > 0.0,
            ana_flat_point_norm > 0.0,
            ana_flat_bin_norm > 0.0,
        ]
    )
    ax0.plot(q[plot_mask], sim_norm[plot_mask], color="#1f77b4", linewidth=1.5, label="Simulation")
    ax0.plot(q[plot_mask], ana_direct_norm[plot_mask], color="black", linewidth=2.0, label="Analytical direct I(q)")
    ax0.plot(
        q[plot_mask],
        ana_flat_point_norm[plot_mask],
        color="#d62728",
        linewidth=1.4,
        alpha=0.75,
        label="Flat-detector analytical (point sampled)",
    )
    ax0.plot(
        q[plot_mask],
        ana_flat_bin_norm[plot_mask],
        color="#2ca02c",
        linewidth=1.4,
        alpha=0.75,
        label=f"Flat-detector analytical (box averaged, {FLAT_BIN_OVERSAMPLE}x)",
    )
    ax0.set_yscale("log")
    ax0.set_xlim(Q_PLOT_MIN, Q_PLOT_MAX)
    ax0.set_ylabel("Normalized I(q)")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best", fontsize=9)

    compare_direct = np.logical_and.reduce(
        [
            q >= Q_ASSERT_MIN,
            q <= Q_ASSERT_MAX,
            np.isfinite(sim_norm),
            np.isfinite(ana_direct_norm),
            sim_norm > MIN_SIGNAL_FOR_LOG,
            ana_direct_norm > MIN_SIGNAL_FOR_LOG,
        ]
    )
    compare_flat_point = np.logical_and.reduce(
        [
            q >= Q_ASSERT_MIN,
            q <= Q_ASSERT_MAX,
            np.isfinite(sim_norm),
            np.isfinite(ana_flat_point_norm),
            sim_norm > MIN_SIGNAL_FOR_LOG,
            ana_flat_point_norm > MIN_SIGNAL_FOR_LOG,
        ]
    )
    compare_flat_bin = np.logical_and.reduce(
        [
            q >= Q_ASSERT_MIN,
            q <= Q_ASSERT_MAX,
            np.isfinite(sim_norm),
            np.isfinite(ana_flat_bin_norm),
            sim_norm > MIN_SIGNAL_FOR_LOG,
            ana_flat_bin_norm > MIN_SIGNAL_FOR_LOG,
        ]
    )
    resid_direct = np.log10(sim_norm[compare_direct]) - np.log10(ana_direct_norm[compare_direct])
    resid_flat_point = np.log10(sim_norm[compare_flat_point]) - np.log10(ana_flat_point_norm[compare_flat_point])
    resid_flat_bin = np.log10(sim_norm[compare_flat_bin]) - np.log10(ana_flat_bin_norm[compare_flat_bin])
    ax1.plot(q[compare_direct], resid_direct, color="black", linewidth=1.2, label="Residual vs direct I(q)")
    ax1.plot(
        q[compare_flat_point],
        resid_flat_point,
        color="#d62728",
        linewidth=1.2,
        label="Residual vs flat-detector analytical (point)",
    )
    ax1.plot(
        q[compare_flat_bin],
        resid_flat_bin,
        color="#2ca02c",
        linewidth=1.2,
        label="Residual vs flat-detector analytical (box averaged)",
    )
    ax1.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_xlim(Q_PLOT_MIN, Q_PLOT_MAX)
    ax1.set_xlabel(r"q [nm$^{-1}$]")
    ax1.set_ylabel(r"$\Delta \log_{10} I$")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best", fontsize=9)

    note = "\n".join(
        [
            (
                f"diameter={DIAMETER_NM:.1f} nm, energy={ENERGY_EV:.1f} eV, "
                f"shape={SHAPE[0]}x{SHAPE[1]}x{SHAPE[2]}, PhysSize={PHYS_SIZE_NM:.2f} nm, WindowingType=0"
            ),
            (
                f"Direct I(q): rms={direct_metrics['rms_log']:.4f} "
                f"p95={direct_metrics['p95_log_abs']:.4f} max={direct_metrics['max_log_abs']:.4f}"
            ),
            (
                f"Flat-detector point: rms={flat_point_metrics['rms_log']:.4f} "
                f"p95={flat_point_metrics['p95_log_abs']:.4f} max={flat_point_metrics['max_log_abs']:.4f}"
            ),
            (
                f"Flat-detector box avg ({FLAT_BIN_OVERSAMPLE}x): rms={flat_bin_metrics['rms_log']:.4f} "
                f"p95={flat_bin_metrics['p95_log_abs']:.4f} max={flat_bin_metrics['max_log_abs']:.4f}"
            ),
            f"build={build_seconds:.2f}s run={run_seconds:.2f}s",
        ]
    )
    ax0.text(
        0.02,
        0.02,
        note,
        transform=ax0.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
    )

    fig.suptitle("Sphere Flat-Detector Analytical Diagnostic")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = PLOT_DIR / "sphere_flat_detector_d150_e285_256x1024x1024.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


@pytest.mark.gpu
@pytest.mark.slow
def test_sphere_flat_detector_analytic_diagnostic():
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for sphere flat-detector diagnostic.")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    build_t0 = perf_counter()
    morph = _build_morphology()
    build_seconds = perf_counter() - build_t0

    run_t0 = perf_counter()
    data = morph.run(stdout=False, stderr=False, return_xarray=True)
    run_seconds = perf_counter() - run_t0

    q, sim_iq = _pyhyper_iq(data)

    ana_direct = _analytic_sphere_form_factor_binned_iq(q, DIAMETER_NM)
    sim_norm, ana_direct_norm, _ = _normalize_to_first_q_gt_zero(q, sim_iq, ana_direct)
    direct_metrics = _metrics(q, sim_norm, ana_direct_norm)

    flat_point_image = _flat_detector_analytic_image_point()
    q_flat_point, ana_flat_point_iq = _pyhyper_iq(flat_point_image)
    if not np.allclose(q, q_flat_point, atol=1e-12, rtol=0.0):
        raise AssertionError("Flat-detector point-sampled analytical q grid does not match simulation reduction q grid.")
    _, ana_flat_point_norm, _ = _normalize_to_first_q_gt_zero(q, sim_iq, ana_flat_point_iq)
    flat_point_metrics = _metrics(q, sim_norm, ana_flat_point_norm)

    flat_bin_image = _flat_detector_analytic_image_binned()
    q_flat_bin, ana_flat_bin_iq = _pyhyper_iq(flat_bin_image)
    if not np.allclose(q, q_flat_bin, atol=1e-12, rtol=0.0):
        raise AssertionError("Flat-detector box-averaged analytical q grid does not match simulation reduction q grid.")
    _, ana_flat_bin_norm, _ = _normalize_to_first_q_gt_zero(q, sim_iq, ana_flat_bin_iq)
    flat_bin_metrics = _metrics(q, sim_norm, ana_flat_bin_norm)

    out = _write_plot(
        q=q,
        sim_norm=sim_norm,
        ana_direct_norm=ana_direct_norm,
        ana_flat_point_norm=ana_flat_point_norm,
        ana_flat_bin_norm=ana_flat_bin_norm,
        direct_metrics=direct_metrics,
        flat_point_metrics=flat_point_metrics,
        flat_bin_metrics=flat_bin_metrics,
        build_seconds=build_seconds,
        run_seconds=run_seconds,
    )

    print("direct_metrics", direct_metrics)
    print("flat_point_metrics", flat_point_metrics)
    print("flat_bin_metrics", flat_bin_metrics)

    assert out.exists()
