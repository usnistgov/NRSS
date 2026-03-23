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
PLOT_DIR = REPO_ROOT / "test-reports" / "sphere-minima-scaling-dev"

SHAPE = (256, 1024, 1024)
PHYS_SIZE_NM = 1.0
DIAMETER_NM = 150.0
ENERGY_EV = 285.0
Q_PLOT_MIN = 0.0
Q_PLOT_MAX = 1.0
MINIMA_QMIN = 0.03
MINIMA_QMAX = 0.50
MINIMA_MAX_COUNT = 8
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


def _find_minima(
    q: np.ndarray,
    iq: np.ndarray,
    qmin: float,
    qmax: float,
    min_spacing: float = 0.015,
    max_count: int | None = None,
) -> list[float]:
    mask = np.logical_and.reduce([q >= qmin, q <= qmax, np.isfinite(iq)])
    q_cut = np.asarray(q[mask], dtype=np.float64)
    iq_cut = np.asarray(iq[mask], dtype=np.float64)
    minima = []
    last_q = -1e9
    for i in range(1, len(iq_cut) - 1):
        if iq_cut[i] <= iq_cut[i - 1] and iq_cut[i] <= iq_cut[i + 1]:
            if q_cut[i] - last_q < min_spacing:
                continue
            minima.append(float(q_cut[i]))
            last_q = float(q_cut[i])
            if max_count is not None and len(minima) >= int(max_count):
                break
    return minima


def _dense_analytic_minima(diameter_nm: float) -> np.ndarray:
    q_dense = np.linspace(MINIMA_QMIN, MINIMA_QMAX, 400_000, dtype=np.float64)
    iq_dense = _analytic_sphere_form_factor_iq(q_dense, diameter_nm)
    minima = _find_minima(
        q=q_dense,
        iq=iq_dense,
        qmin=MINIMA_QMIN,
        qmax=MINIMA_QMAX,
        min_spacing=0.02,
        max_count=MINIMA_MAX_COUNT,
    )
    return np.asarray(minima, dtype=np.float64)


def _candidate_scale_factors(nx: int) -> list[tuple[str, float]]:
    nx = int(nx)
    return [
        ("1", 1.0),
        ("N/(N-1)", nx / (nx - 1)),
        ("(N-1)/N", (nx - 1) / nx),
        ("N/(N+1)", nx / (nx + 1)),
        ("(N+1)/N", (nx + 1) / nx),
        ("N/(N-2)", nx / (nx - 2)),
        ("(N-2)/N", (nx - 2) / nx),
        ("N/(N+2)", nx / (nx + 2)),
        ("(N+2)/N", (nx + 2) / nx),
    ]


def _write_plot(
    q: np.ndarray,
    sim_norm: np.ndarray,
    ana_norm: np.ndarray,
    sim_minima: np.ndarray,
    ana_minima: np.ndarray,
    ratio: np.ndarray,
    mean_ratio: float,
    best_candidates: list[tuple[str, float, float]],
    q_step: float,
    delta_q_mean: float,
    delta_q_std: float,
    build_seconds: float,
    run_seconds: float,
) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.0, 8.5), sharex=False)

    plot_mask = np.logical_and.reduce(
        [
            q >= Q_PLOT_MIN,
            q <= Q_PLOT_MAX,
            np.isfinite(sim_norm),
            np.isfinite(ana_norm),
            sim_norm > MIN_SIGNAL_FOR_LOG,
            ana_norm > MIN_SIGNAL_FOR_LOG,
        ]
    )
    ax0.plot(q[plot_mask], ana_norm[plot_mask], color="black", linewidth=2.0, label="Analytical (PyHyper bins)")
    ax0.plot(q[plot_mask], sim_norm[plot_mask], color="#1f77b4", alpha=0.7, linewidth=1.5, label="Simulation")
    sim_y = np.interp(sim_minima, q, sim_norm)
    ana_y = np.interp(ana_minima, q, ana_norm)
    ax0.scatter(ana_minima, ana_y, color="black", s=26, zorder=4, label="Analytical minima")
    ax0.scatter(sim_minima, sim_y, color="#d62728", s=26, zorder=4, label="Sim minima")
    ax0.set_yscale("log")
    ax0.set_xlim(Q_PLOT_MIN, Q_PLOT_MAX)
    ax0.set_ylabel("Normalized I(q)")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best")

    index = np.arange(1, ratio.size + 1, dtype=np.int64)
    ax1.plot(index, ratio, marker="o", color="#1f77b4", linewidth=1.4, label=r"$q_\mathrm{sim}/q_\mathrm{ana}$")
    ax1.axhline(mean_ratio, color="black", linewidth=1.2, linestyle="-", label=f"mean={mean_ratio:.6f}")
    for name, value, _ in best_candidates:
        ax1.axhline(value, linewidth=1.0, linestyle=":", label=f"{name}={value:.6f}")
    ax1.set_xlabel("Minimum Index")
    ax1.set_ylabel(r"$q_\mathrm{sim} / q_\mathrm{ana}$")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best", fontsize=9)

    note_lines = [
        (
            f"diameter={DIAMETER_NM:.1f} nm, energy={ENERGY_EV:.1f} eV, "
            f"shape={SHAPE[0]}x{SHAPE[1]}x{SHAPE[2]}, PhysSize={PHYS_SIZE_NM:.2f} nm, WindowingType=0"
        ),
        f"sim minima={np.array2string(sim_minima, precision=6, separator=', ')}",
        f"ana minima={np.array2string(ana_minima, precision=6, separator=', ')}",
        f"ratio mean={mean_ratio:.6f}, ratio std={float(np.std(ratio)):.6e}",
        f"delta_q mean={delta_q_mean:.6e} nm^-1, std={delta_q_std:.6e}, q step~{q_step:.6e}",
        f"best candidates: {', '.join([f'{name} ({value:.6f}, |d|={delta:.2e})' for name, value, delta in best_candidates])}",
        f"build={build_seconds:.2f}s run={run_seconds:.2f}s",
    ]
    ax0.text(
        0.02,
        0.02,
        "\n".join(note_lines),
        transform=ax0.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
    )

    fig.suptitle("Sphere Minima Scaling Diagnostic")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = PLOT_DIR / "sphere_minima_scaling_d150_e285_256x1024x1024.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


@pytest.mark.gpu
@pytest.mark.slow
def test_sphere_minima_scaling_diagnostic():
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for sphere minima scaling diagnostic.")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    build_t0 = perf_counter()
    morph = _build_morphology()
    build_seconds = perf_counter() - build_t0

    run_t0 = perf_counter()
    data = morph.run(stdout=False, stderr=False, return_xarray=True)
    run_seconds = perf_counter() - run_t0

    q, sim_iq = _pyhyper_iq(data)
    ana_iq = _analytic_sphere_form_factor_binned_iq(q, DIAMETER_NM)
    sim_norm, ana_norm, _ = _normalize_to_first_q_gt_zero(q, sim_iq, ana_iq)

    sim_minima = np.asarray(
        _find_minima(
            q=q,
            iq=sim_norm,
            qmin=MINIMA_QMIN,
            qmax=MINIMA_QMAX,
            min_spacing=0.02,
            max_count=MINIMA_MAX_COUNT,
        ),
        dtype=np.float64,
    )
    ana_minima = _dense_analytic_minima(DIAMETER_NM)

    count = min(sim_minima.size, ana_minima.size, MINIMA_MAX_COUNT)
    if count < 4:
        raise AssertionError(f"Expected at least 4 matched minima, got {count}.")
    sim_minima = sim_minima[:count]
    ana_minima = ana_minima[:count]

    ratio = sim_minima / ana_minima
    delta_q = sim_minima - ana_minima
    mean_ratio = float(np.mean(ratio))
    q_step = float(np.median(np.diff(q)))

    candidates = _candidate_scale_factors(SHAPE[2])
    best_candidates = sorted(
        [(name, value, abs(mean_ratio - value)) for name, value in candidates],
        key=lambda item: item[2],
    )[:4]

    out = _write_plot(
        q=q,
        sim_norm=sim_norm,
        ana_norm=ana_norm,
        sim_minima=sim_minima,
        ana_minima=ana_minima,
        ratio=ratio,
        mean_ratio=mean_ratio,
        best_candidates=best_candidates,
        q_step=q_step,
        delta_q_mean=float(np.mean(delta_q)),
        delta_q_std=float(np.std(delta_q)),
        build_seconds=build_seconds,
        run_seconds=run_seconds,
    )

    print("sim_minima", sim_minima)
    print("ana_minima", ana_minima)
    print("ratio", ratio)
    print("mean_ratio", mean_ratio)
    print("ratio_std", float(np.std(ratio)))
    print("delta_q", delta_q)
    print("delta_q_mean", float(np.mean(delta_q)))
    print("delta_q_std", float(np.std(delta_q)))
    print("q_step", q_step)
    print("best_candidates", best_candidates)

    assert out.exists()
