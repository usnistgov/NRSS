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
PLOT_DIR = REPO_ROOT / "test-reports" / "sphere-diameter-fit-dev"

SHAPE = (256, 1024, 1024)
PHYS_SIZE_NM = 1.0
SIM_DIAMETER_NM = 150.0
ENERGY_EV = 285.0
Q_FIT_MIN = 0.06
Q_FIT_MAX = 1.0
Q_PLOT_MIN = 0.0
Q_PLOT_MAX = 1.0
MIN_SIGNAL_FOR_LOG = 1e-8
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
    sphere_vfrac, vacuum_vfrac = _sphere_and_vacuum_vfrac(SIM_DIAMETER_NM)
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


def _log_residual(q: np.ndarray, sim_norm: np.ndarray, ana_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.logical_and.reduce(
        [
            q >= Q_FIT_MIN,
            q <= Q_FIT_MAX,
            np.isfinite(sim_norm),
            np.isfinite(ana_norm),
            sim_norm > MIN_SIGNAL_FOR_LOG,
            ana_norm > MIN_SIGNAL_FOR_LOG,
        ]
    )
    resid = np.log10(sim_norm[mask]) - np.log10(ana_norm[mask])
    return q[mask], resid


def _log_rms_objective(q: np.ndarray, sim_norm: np.ndarray, ana_norm: np.ndarray) -> float:
    _, resid = _log_residual(q, sim_norm, ana_norm)
    return float(np.sqrt(np.mean(resid * resid)))


def _fit_effective_diameter(q: np.ndarray, sim_iq: np.ndarray) -> tuple[float, float]:
    def objective(diameter_nm: float) -> float:
        ana = _analytic_sphere_form_factor_binned_iq(q, float(diameter_nm))
        sim_norm, ana_norm, _ = _normalize_to_first_q_gt_zero(q, sim_iq, ana)
        return _log_rms_objective(q, sim_norm, ana_norm)

    coarse = np.linspace(140.0, 160.0, 81, dtype=np.float64)
    coarse_obj = np.asarray([objective(diameter) for diameter in coarse], dtype=np.float64)
    coarse_best = int(np.argmin(coarse_obj))
    coarse_d = float(coarse[coarse_best])

    lo = max(130.0, coarse_d - 0.5)
    hi = min(170.0, coarse_d + 0.5)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    invphi = 1.0 / phi

    c = hi - (hi - lo) * invphi
    d = lo + (hi - lo) * invphi
    fc = objective(c)
    fd = objective(d)

    for _ in range(28):
        if fc < fd:
            hi = d
            d = c
            fd = fc
            c = hi - (hi - lo) * invphi
            fc = objective(c)
        else:
            lo = c
            c = d
            fc = fd
            d = lo + (hi - lo) * invphi
            fd = objective(d)

    best_diameter = float((lo + hi) / 2.0)
    best_obj = objective(best_diameter)
    return best_diameter, best_obj


def _write_plot(
    q: np.ndarray,
    sim_norm: np.ndarray,
    ana_nominal_norm: np.ndarray,
    ana_fit_norm: np.ndarray,
    best_diameter_nm: float,
    nominal_obj: float,
    best_obj: float,
    build_seconds: float,
    run_seconds: float,
) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.0, 8.5), sharex=True)

    plot_mask = np.logical_and.reduce(
        [
            q >= Q_PLOT_MIN,
            q <= Q_PLOT_MAX,
            np.isfinite(sim_norm),
            np.isfinite(ana_nominal_norm),
            np.isfinite(ana_fit_norm),
            sim_norm > MIN_SIGNAL_FOR_LOG,
            ana_nominal_norm > MIN_SIGNAL_FOR_LOG,
            ana_fit_norm > MIN_SIGNAL_FOR_LOG,
        ]
    )
    ax0.plot(q[plot_mask], sim_norm[plot_mask], color="#1f77b4", linewidth=1.5, label="Simulation")
    ax0.plot(
        q[plot_mask],
        ana_nominal_norm[plot_mask],
        color="black",
        linewidth=2.0,
        label=f"Analytical d={SIM_DIAMETER_NM:.3f} nm",
    )
    ax0.plot(
        q[plot_mask],
        ana_fit_norm[plot_mask],
        color="#d62728",
        linewidth=1.5,
        alpha=0.85,
        label=f"Analytical fit d={best_diameter_nm:.3f} nm",
    )
    ax0.set_yscale("log")
    ax0.set_xlim(Q_PLOT_MIN, Q_PLOT_MAX)
    ax0.set_ylabel("Normalized I(q)")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best", fontsize=9)

    q_nominal, resid_nominal = _log_residual(q, sim_norm, ana_nominal_norm)
    q_fit, resid_fit = _log_residual(q, sim_norm, ana_fit_norm)
    ax1.plot(q_nominal, resid_nominal, color="black", linewidth=1.2, label="Residual vs nominal analytical")
    ax1.plot(q_fit, resid_fit, color="#d62728", linewidth=1.2, label="Residual vs fit analytical")
    ax1.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_xlim(Q_PLOT_MIN, Q_PLOT_MAX)
    ax1.set_xlabel(r"q [nm$^{-1}$]")
    ax1.set_ylabel(r"$\Delta \log_{10} I$")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best", fontsize=9)

    note = "\n".join(
        [
            (
                f"sim diameter={SIM_DIAMETER_NM:.3f} nm, fitted analytical diameter={best_diameter_nm:.3f} nm, "
                f"shape={SHAPE[0]}x{SHAPE[1]}x{SHAPE[2]}, PhysSize={PHYS_SIZE_NM:.2f} nm"
            ),
            f"objective = log-RMS over q={Q_FIT_MIN:.2f}..{Q_FIT_MAX:.2f} nm^-1 after normalization to first q>0 point",
            f"WindowingType=0, nominal objective={nominal_obj:.6f}, fit objective={best_obj:.6f}",
            f"improvement factor={nominal_obj / best_obj:.3f}",
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

    fig.suptitle("Sphere Whole-Curve Diameter-Fit Diagnostic")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = PLOT_DIR / "sphere_diameter_fit_d150_e285_256x1024x1024.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


@pytest.mark.gpu
@pytest.mark.slow
def test_sphere_diameter_fit_diagnostic():
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for sphere diameter-fit diagnostic.")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    build_t0 = perf_counter()
    morph = _build_morphology()
    build_seconds = perf_counter() - build_t0

    run_t0 = perf_counter()
    data = morph.run(stdout=False, stderr=False, return_xarray=True)
    run_seconds = perf_counter() - run_t0

    q, sim_iq = _pyhyper_iq(data)
    ana_nominal = _analytic_sphere_form_factor_binned_iq(q, SIM_DIAMETER_NM)
    sim_norm, ana_nominal_norm, _ = _normalize_to_first_q_gt_zero(q, sim_iq, ana_nominal)
    nominal_obj = _log_rms_objective(q, sim_norm, ana_nominal_norm)

    best_diameter_nm, best_obj = _fit_effective_diameter(q, sim_iq)
    ana_fit = _analytic_sphere_form_factor_binned_iq(q, best_diameter_nm)
    _, ana_fit_norm, _ = _normalize_to_first_q_gt_zero(q, sim_iq, ana_fit)

    out = _write_plot(
        q=q,
        sim_norm=sim_norm,
        ana_nominal_norm=ana_nominal_norm,
        ana_fit_norm=ana_fit_norm,
        best_diameter_nm=best_diameter_nm,
        nominal_obj=nominal_obj,
        best_obj=best_obj,
        build_seconds=build_seconds,
        run_seconds=run_seconds,
    )

    print("best_diameter_nm", best_diameter_nm)
    print("nominal_log_rms", nominal_obj)
    print("best_log_rms", best_obj)
    print("improvement_factor", nominal_obj / best_obj)

    assert out.exists()
