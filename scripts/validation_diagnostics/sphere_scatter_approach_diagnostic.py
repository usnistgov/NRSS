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

import CyRSoXS as cy

from NRSS.morphology import Material, Morphology


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = REPO_ROOT / "test-reports" / "sphere-scatter-approach-dev"

SHAPE = (256, 512, 512)
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
SCATTER_APPROACHES = [
    ("partial", cy.ScatterApproach.Partial, "#1f77b4"),
    ("full", cy.ScatterApproach.Full, "#d62728"),
]


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


def _build_morphology(scatter_approach) -> Morphology:
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
    morph.inputData.scatterApproach = scatter_approach
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


def _write_plot(results: dict[str, dict]) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.0, 8.0), sharex=True)

    for label, _, color in SCATTER_APPROACHES:
        res = results[label]
        q = res["q"]
        sim = res["sim_norm"]
        ana = res["ana_norm"]
        plot_mask = np.logical_and.reduce(
            [
                q >= Q_PLOT_MIN,
                q <= Q_PLOT_MAX,
                np.isfinite(sim),
                np.isfinite(ana),
                sim > 0.0,
                ana > 0.0,
            ]
        )
        ax0.plot(
            q[plot_mask],
            sim[plot_mask],
            color=color,
            alpha=0.7,
            linewidth=1.5,
            label=f"{label.title()} scatter",
        )

    reference_label = SCATTER_APPROACHES[0][0]
    q_ref = results[reference_label]["q"]
    ana_ref = results[reference_label]["ana_norm"]
    plot_mask_ref = np.logical_and.reduce(
        [
            q_ref >= Q_PLOT_MIN,
            q_ref <= Q_PLOT_MAX,
            np.isfinite(ana_ref),
            ana_ref > 0.0,
        ]
    )
    ax0.plot(
        q_ref[plot_mask_ref],
        ana_ref[plot_mask_ref],
        color="black",
        linewidth=2.0,
        label="Analytical (PyHyper bins)",
    )
    ax0.set_yscale("log")
    ax0.set_ylabel("Normalized I(q)")
    ax0.set_xlim(Q_PLOT_MIN, Q_PLOT_MAX)
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best")

    for label, _, color in SCATTER_APPROACHES:
        res = results[label]
        q = res["q"]
        sim = res["sim_norm"]
        ana = res["ana_norm"]
        compare_mask = np.logical_and.reduce(
            [
                q >= Q_ASSERT_MIN,
                q <= Q_ASSERT_MAX,
                np.isfinite(sim),
                np.isfinite(ana),
                sim > MIN_SIGNAL_FOR_LOG,
                ana > MIN_SIGNAL_FOR_LOG,
            ]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            resid = np.log10(sim[compare_mask]) - np.log10(ana[compare_mask])
        ax1.plot(
            q[compare_mask],
            resid,
            color=color,
            alpha=0.7,
            linewidth=1.25,
            label=f"{label.title()} scatter",
        )

    ax1.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_xlabel(r"q [nm$^{-1}$]")
    ax1.set_ylabel(r"$\Delta \log_{10} I$")
    ax1.set_xlim(Q_PLOT_MIN, Q_PLOT_MAX)
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best")

    note_lines = [
        (
            f"diameter={DIAMETER_NM:.1f} nm, energy={ENERGY_EV:.1f} eV, "
            f"shape={SHAPE[0]}x{SHAPE[1]}x{SHAPE[2]}, PhysSize={PHYS_SIZE_NM:.2f} nm"
        )
    ]
    for label, _, _ in SCATTER_APPROACHES:
        metrics = results[label]["metrics"]
        minima = results[label]["minima"]
        build_seconds = results[label]["build_seconds"]
        run_seconds = results[label]["run_seconds"]
        note_lines.append(
            (
                f"{label.title()}: p95={metrics['p95_log_abs']:.4f} max={metrics['max_log_abs']:.4f} "
                f"minima={minima[:4]} build={build_seconds:.2f}s run={run_seconds:.2f}s"
            )
        )
    ax0.text(
        0.02,
        0.02,
        "\n".join(note_lines),
        transform=ax0.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    fig.suptitle("Sphere ScatterApproach Diagnostic")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = PLOT_DIR / "sphere_scatter_approach_d150_e285.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


@pytest.mark.gpu
@pytest.mark.slow
def test_sphere_scatter_approach_diagnostic():
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for sphere scatter approach diagnostic.")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    results = {}
    for label, scatter_approach, _ in SCATTER_APPROACHES:
        build_t0 = perf_counter()
        morph = _build_morphology(scatter_approach=scatter_approach)
        build_seconds = perf_counter() - build_t0

        run_t0 = perf_counter()
        data = morph.run(stdout=False, stderr=False, return_xarray=True)
        run_seconds = perf_counter() - run_t0

        q, sim_iq = _pyhyper_iq(data)
        ana_iq = _analytic_sphere_form_factor_binned_iq(q, DIAMETER_NM)
        sim_norm, ana_norm, _ = _normalize_to_first_q_gt_zero(q, sim_iq, ana_iq)

        results[label] = {
            "q": q,
            "sim_norm": sim_norm,
            "ana_norm": ana_norm,
            "metrics": _metrics(q, sim_norm, ana_norm),
            "minima": _find_minima(q, sim_norm),
            "build_seconds": build_seconds,
            "run_seconds": run_seconds,
        }

    out = _write_plot(results)
    assert out.exists()
