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

from NRSS.morphology import Material, Morphology


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = REPO_ROOT / "test-reports" / "sphere-fftigor-dev"

SHAPE = (256, 1024, 1024)
PHYS_SIZE_NM = 1.0
DIAMETER_NM = 150.0
ENERGY_EV = 285.0
Q_ASSERT_MIN = 0.06
Q_ASSERT_MAX = 1.0
Q_PLOT_MIN = 0.0
Q_PLOT_MAX = 1.0
MIN_SIGNAL_FOR_LOG = 1e-5
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


def _fftigor_index_map(n: int) -> np.ndarray:
    n = int(n)
    mid = n // 2
    mapping = np.empty(n, dtype=np.int64)
    for x in range(n):
        if x <= mid:
            mapping[x] = mid - x
        else:
            mapping[x] = n + (mid - x)
    return mapping


def _fftigor_axis_from_unshifted_fft(n: int, phys_size_nm: float) -> np.ndarray:
    q_unshifted = 2.0 * np.pi * np.fft.fftfreq(int(n), d=float(phys_size_nm))
    return q_unshifted[_fftigor_index_map(n)]


def _custom_fftigor_radial_average(image_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if image_2d.ndim != 2:
        raise AssertionError(f"Expected 2D detector image, got shape={image_2d.shape}.")

    ny, nx = image_2d.shape
    qy = _fftigor_axis_from_unshifted_fft(ny, PHYS_SIZE_NM)
    qx = _fftigor_axis_from_unshifted_fft(nx, PHYS_SIZE_NM)
    qmap = np.hypot(qy[:, None], qx[None, :])

    dq_candidates = []
    qx_unique = np.unique(np.abs(qx))
    qy_unique = np.unique(np.abs(qy))
    if qx_unique.size > 1:
        dq_candidates.append(float(np.min(np.diff(qx_unique))))
    if qy_unique.size > 1:
        dq_candidates.append(float(np.min(np.diff(qy_unique))))
    if not dq_candidates:
        raise AssertionError("Could not determine dq for FFTIgor radial average.")
    dq = min(dq_candidates)

    q_flat = qmap.ravel()
    img_flat = np.asarray(image_2d, dtype=np.float64).ravel()
    valid = np.isfinite(q_flat) & np.isfinite(img_flat) & (img_flat >= 0.0)
    q_flat = q_flat[valid]
    img_flat = img_flat[valid]

    bin_index = np.rint(q_flat / dq).astype(np.int64)
    nbins = int(bin_index.max()) + 1
    counts = np.bincount(bin_index, minlength=nbins)
    q_sum = np.bincount(bin_index, weights=q_flat, minlength=nbins)
    i_sum = np.bincount(bin_index, weights=img_flat, minlength=nbins)

    keep = counts > 0
    q_centers = np.zeros(nbins, dtype=np.float64)
    iq = np.zeros(nbins, dtype=np.float64)
    q_centers[keep] = q_sum[keep] / counts[keep]
    iq[keep] = i_sum[keep] / counts[keep]
    return q_centers[keep], iq[keep]


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
        log_abs = np.abs(np.log10(sim_norm[mask]) - np.log10(ana_norm[mask]))
    return {
        "p95_log_abs": float(np.percentile(log_abs, 95)),
        "max_log_abs": float(np.max(log_abs)),
        "n_comp": int(np.count_nonzero(mask)),
    }


def _find_minima(q: np.ndarray, iq: np.ndarray, qmin: float = 0.03, qmax: float = 0.5) -> list[float]:
    mask = np.logical_and.reduce([q >= qmin, q <= qmax, np.isfinite(iq)])
    q_cut = q[mask]
    iq_cut = iq[mask]
    minima = []
    last_q = -1e9
    for i in range(1, len(iq_cut) - 1):
        if iq_cut[i] <= iq_cut[i - 1] and iq_cut[i] <= iq_cut[i + 1]:
            if q_cut[i] - last_q < 0.015:
                continue
            minima.append(float(q_cut[i]))
            last_q = float(q_cut[i])
            if len(minima) >= 5:
                break
    return minima


def _write_plot(
    q_pyh: np.ndarray,
    pyh_norm: np.ndarray,
    q_fftigor: np.ndarray,
    fftigor_norm: np.ndarray,
    ana_pyh_norm: np.ndarray,
    ana_fftigor_norm: np.ndarray,
    pyh_metrics: dict[str, float],
    fftigor_metrics: dict[str, float],
    build_seconds: float,
    run_seconds: float,
) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.0, 8.0), sharex=True)

    plot_mask_pyh = np.logical_and.reduce(
        [
            q_pyh >= Q_PLOT_MIN,
            q_pyh <= Q_PLOT_MAX,
            np.isfinite(pyh_norm),
            np.isfinite(ana_pyh_norm),
            pyh_norm > 0.0,
            ana_pyh_norm > 0.0,
        ]
    )
    plot_mask_fftigor = np.logical_and.reduce(
        [
            q_fftigor >= Q_PLOT_MIN,
            q_fftigor <= Q_PLOT_MAX,
            np.isfinite(fftigor_norm),
            np.isfinite(ana_fftigor_norm),
            fftigor_norm > 0.0,
            ana_fftigor_norm > 0.0,
        ]
    )

    ax0.plot(q_pyh[plot_mask_pyh], ana_pyh_norm[plot_mask_pyh], color="black", linewidth=2.0, label="Analytical (PyHyper bins)")
    ax0.plot(q_pyh[plot_mask_pyh], pyh_norm[plot_mask_pyh], color="#1f77b4", alpha=0.6, linewidth=1.5, label="PyHyper reduction")
    ax0.plot(q_fftigor[plot_mask_fftigor], fftigor_norm[plot_mask_fftigor], color="#d62728", alpha=0.6, linewidth=1.5, label="Custom FFTIgor q-map")
    ax0.set_yscale("log")
    ax0.set_ylabel("Normalized I(q)")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best")
    ax0.set_xlim(Q_PLOT_MIN, Q_PLOT_MAX)

    compare_mask_pyh = np.logical_and.reduce(
        [
            q_pyh >= Q_ASSERT_MIN,
            q_pyh <= Q_ASSERT_MAX,
            np.isfinite(pyh_norm),
            np.isfinite(ana_pyh_norm),
            pyh_norm > MIN_SIGNAL_FOR_LOG,
            ana_pyh_norm > MIN_SIGNAL_FOR_LOG,
        ]
    )
    compare_mask_fftigor = np.logical_and.reduce(
        [
            q_fftigor >= Q_ASSERT_MIN,
            q_fftigor <= Q_ASSERT_MAX,
            np.isfinite(fftigor_norm),
            np.isfinite(ana_fftigor_norm),
            fftigor_norm > MIN_SIGNAL_FOR_LOG,
            ana_fftigor_norm > MIN_SIGNAL_FOR_LOG,
        ]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        resid_pyh = np.log10(pyh_norm[compare_mask_pyh]) - np.log10(ana_pyh_norm[compare_mask_pyh])
        resid_fftigor = np.log10(fftigor_norm[compare_mask_fftigor]) - np.log10(ana_fftigor_norm[compare_mask_fftigor])
    ax1.plot(q_pyh[compare_mask_pyh], resid_pyh, color="#1f77b4", alpha=0.6, linewidth=1.25, label="PyHyper reduction")
    ax1.plot(q_fftigor[compare_mask_fftigor], resid_fftigor, color="#d62728", alpha=0.6, linewidth=1.25, label="Custom FFTIgor q-map")
    ax1.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_xlabel(r"q [nm$^{-1}$]")
    ax1.set_ylabel(r"$\Delta \log_{10} I$")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best")
    ax1.set_xlim(Q_PLOT_MIN, Q_PLOT_MAX)

    note = "\n".join(
        [
            f"diameter={DIAMETER_NM:.1f} nm, energy={ENERGY_EV:.1f} eV, shape={SHAPE[0]}x{SHAPE[1]}x{SHAPE[2]}, PhysSize={PHYS_SIZE_NM:.2f} nm",
            f"PyHyper/CyRSoXS-grid: p95={pyh_metrics['p95_log_abs']:.4f} max={pyh_metrics['max_log_abs']:.4f} minima={_find_minima(q_pyh, pyh_norm)[:4]}",
            f"FFTIgor q-map: p95={fftigor_metrics['p95_log_abs']:.4f} max={fftigor_metrics['max_log_abs']:.4f} minima={_find_minima(q_fftigor, fftigor_norm)[:4]}",
            f"build={build_seconds:.2f}s run={run_seconds:.2f}s",
        ]
    )
    ax0.text(
        0.02,
        0.02,
        note,
        transform=ax0.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    fig.suptitle("Sphere FFTIgor q-Map Diagnostic")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = PLOT_DIR / "sphere_fftigor_qmap_d150_e285.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


@pytest.mark.gpu
@pytest.mark.slow
def test_sphere_fftigor_qmap_diagnostic():
    """Development diagnostic: compare PyHyper reduction to a custom FFTIgor-derived q-map radial average."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for sphere FFTIgor diagnostic.")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    build_t0 = perf_counter()
    morph = _build_morphology()
    build_seconds = perf_counter() - build_t0

    run_t0 = perf_counter()
    data = morph.run(stdout=False, stderr=False, return_xarray=True)
    run_seconds = perf_counter() - run_t0

    q_pyh, sim_pyh = _pyhyper_iq(data)
    ana_pyh = _analytic_sphere_form_factor_binned_iq(q_pyh, DIAMETER_NM)
    pyh_norm, ana_pyh_norm, _ = _normalize_to_first_q_gt_zero(q_pyh, sim_pyh, ana_pyh)
    pyh_metrics = _metrics(q_pyh, pyh_norm, ana_pyh_norm)

    image = np.asarray(data.sel(energy=float(ENERGY_EV)).values, dtype=np.float64)
    q_fftigor, sim_fftigor = _custom_fftigor_radial_average(image)
    ana_fftigor = _analytic_sphere_form_factor_binned_iq(q_fftigor, DIAMETER_NM)
    fftigor_norm, ana_fftigor_norm, _ = _normalize_to_first_q_gt_zero(q_fftigor, sim_fftigor, ana_fftigor)
    fftigor_metrics = _metrics(q_fftigor, fftigor_norm, ana_fftigor_norm)

    out = _write_plot(
        q_pyh=q_pyh,
        pyh_norm=pyh_norm,
        q_fftigor=q_fftigor,
        fftigor_norm=fftigor_norm,
        ana_pyh_norm=ana_pyh_norm,
        ana_fftigor_norm=ana_fftigor_norm,
        pyh_metrics=pyh_metrics,
        fftigor_metrics=fftigor_metrics,
        build_seconds=build_seconds,
        run_seconds=run_seconds,
    )

    assert out.exists()
