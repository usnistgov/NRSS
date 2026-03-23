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
from scipy.special import j1

from NRSS.morphology import Material, Morphology


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = REPO_ROOT / "test-reports" / "analytical-disk-dev"

BOX_SIZE_NM = 1024.0
DISK_DIAMETER_NM = 150.0
PHYS_SIZES_NM = [1.0, 0.5, 0.25, 0.1]
Q_ASSERT_MIN = 0.06
Q_ASSERT_MAX = 1.0
Q_PLOT_MIN = 0.0
Q_PLOT_MAX = 1.0
MIN_SIGNAL_FOR_LOG = 1e-5
ENERGIES_EV = [285.0]
DISK_OC_BY_ENERGY = {
    285.0: (0.0, 2e-4, 0.0, 2e-4),  # beta-only positive
}
PHYS_COLORS = {
    1.0: "#1f77b4",
    0.5: "#d62728",
    0.25: "#2ca02c",
    0.1: "#ff7f0e",
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


def _shape_for_phys_size(phys_size_nm: float) -> tuple[int, int, int]:
    nxy = int(round(BOX_SIZE_NM / float(phys_size_nm)))
    if nxy < 16:
        raise AssertionError(f"Invalid 2D disk grid size for PhysSize={phys_size_nm}: {nxy}")
    return (1, nxy, nxy)


def _disk_and_vacuum_vfrac(
    diameter_nm: float,
    phys_size_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    shape = _shape_for_phys_size(phys_size_nm)
    _, ny, nx = shape
    radius_vox = float(diameter_nm) / (2.0 * float(phys_size_nm))
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0

    pad = radius_vox + 2.0
    y0 = max(0, int(np.floor(cy - pad)))
    y1 = min(ny, int(np.ceil(cy + pad)) + 1)
    x0 = max(0, int(np.floor(cx - pad)))
    x1 = min(nx, int(np.ceil(cx + pad)) + 1)

    yy, xx = np.ogrid[y0:y1, x0:x1]
    dy = yy.astype(np.float32) - np.float32(cy)
    dx = xx.astype(np.float32) - np.float32(cx)
    dist2 = dx * dx + dy * dy

    disk = np.zeros(shape, dtype=np.float32)
    local_disk = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    local_disk[dist2 <= np.float32(radius_vox * radius_vox)] = 1.0
    disk[0, y0:y1, x0:x1] = local_disk
    vacuum = (1.0 - disk).astype(np.float32)
    return disk, vacuum


def _build_disk_morphology(
    diameter_nm: float,
    phys_size_nm: float,
    energies_eV: list[float],
    disk_oc_by_energy: dict[float, tuple[float, float, float, float]],
) -> Morphology:
    shape = _shape_for_phys_size(phys_size_nm)
    disk_vfrac, vacuum_vfrac = _disk_and_vacuum_vfrac(
        diameter_nm=diameter_nm,
        phys_size_nm=phys_size_nm,
    )
    zeros = np.zeros(shape, dtype=np.float32)
    disk_oc = {float(e): list(disk_oc_by_energy[float(e)]) for e in energies_eV}
    vacuum_oc = {float(e): [0.0, 0.0, 0.0, 0.0] for e in energies_eV}

    mat_disk = Material(
        materialID=1,
        Vfrac=disk_vfrac,
        S=zeros,
        theta=zeros,
        psi=zeros,
        energies=energies_eV,
        opt_constants=disk_oc,
        name="disk",
    )
    mat_vacuum = Material(
        materialID=2,
        Vfrac=vacuum_vfrac,
        S=zeros,
        theta=zeros,
        psi=zeros,
        energies=energies_eV,
        opt_constants=vacuum_oc,
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
        materials={1: mat_disk, 2: mat_vacuum},
        PhysSize=float(phys_size_nm),
        config=config,
        create_cy_object=True,
    )
    morph.check_materials(quiet=True)
    morph.validate_all(quiet=True)
    return morph


def _pyhyper_iq_by_energy(scattering) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    from PyHyperScattering.integrate import WPIntegrator

    integrator = WPIntegrator(use_chunked_processing=False)
    remeshed = integrator.integrateImageStack(scattering)

    if "energy" not in remeshed.dims:
        raise AssertionError("PyHyperScattering output missing 'energy' dimension.")
    if "chi" not in remeshed.dims:
        raise AssertionError("PyHyperScattering output missing 'chi' dimension.")
    qdim = next((d for d in remeshed.dims if d == "q" or d.startswith("q")), None)
    if qdim is None:
        raise AssertionError("PyHyperScattering output missing q-dimension.")

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
        raise AssertionError("No valid q>0 normalization point found for analytical-disk comparison.")

    i0 = int(candidates[0])
    sim_norm = sim_iq / sim_iq[i0]
    ana_norm = ana_iq / ana_iq[i0]
    return sim_norm, ana_norm, float(q[i0])


def _write_disk_resolution_plot(
    results_by_phys_size: dict[float, dict[str, object]],
    diameter_nm: float,
    energy_eV: float,
) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.0, 8.0), sharex=True)

    reference_phys = PHYS_SIZES_NM[0]
    q_ref = results_by_phys_size[reference_phys]["q"]
    ana_ref = results_by_phys_size[reference_phys]["ana_binned_norm"]
    plot_mask_ref = results_by_phys_size[reference_phys]["plot_mask"]
    ax0.plot(
        q_ref[plot_mask_ref],
        ana_ref[plot_mask_ref],
        color="black",
        linewidth=2.0,
        label="Analytical q-bin averaged",
    )

    y_all = [ana_ref[plot_mask_ref]]
    resid_all = []
    for phys_size_nm in PHYS_SIZES_NM:
        result = results_by_phys_size[phys_size_nm]
        q = result["q"]
        sim_norm = result["sim_norm"]
        ana_binned_norm = result["ana_binned_norm"]
        plot_mask = result["plot_mask"]
        assert_mask = result["assert_mask"]

        ax0.plot(
            q[plot_mask],
            sim_norm[plot_mask],
            color=PHYS_COLORS[phys_size_nm],
            linewidth=1.5,
            alpha=0.6,
            label=f"Pybind PhysSize={phys_size_nm:g} nm",
        )
        y_all.append(sim_norm[plot_mask])

        with np.errstate(divide="ignore", invalid="ignore"):
            resid = np.log10(sim_norm[assert_mask]) - np.log10(ana_binned_norm[assert_mask])
        ax1.plot(
            q[assert_mask],
            resid,
            color=PHYS_COLORS[phys_size_nm],
            linewidth=1.25,
            alpha=0.6,
            label=f"PhysSize={phys_size_nm:g} nm",
        )
        resid_all.append(resid)

    ax0.set_yscale("log")
    ax0.set_ylabel("Normalized I(q)")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best")
    ax0.set_xlim(Q_PLOT_MIN, Q_PLOT_MAX)
    y_all_flat = np.concatenate([arr for arr in y_all if arr.size > 0])
    y_all_flat = y_all_flat[np.logical_and(np.isfinite(y_all_flat), y_all_flat > 0.0)]
    if y_all_flat.size > 0:
        ymin = float(np.min(y_all_flat))
        ymax = float(np.max(y_all_flat))
        ax0.set_ylim(max(ymin * 0.8, 1e-12), ymax * 1.2)

    ax1.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_xlabel(r"q [nm$^{-1}$]")
    ax1.set_ylabel(r"$\Delta \log_{10} I$")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best")
    ax1.set_xlim(Q_PLOT_MIN, Q_PLOT_MAX)
    resid_all_flat = np.concatenate([arr for arr in resid_all if arr.size > 0])
    resid_all_flat = resid_all_flat[np.isfinite(resid_all_flat)]
    if resid_all_flat.size > 0:
        rmax = float(np.max(np.abs(resid_all_flat)))
        if rmax > 0.0:
            ax1.set_ylim(-1.1 * rmax, 1.1 * rmax)

    note_lines = [
        f"diameter={diameter_nm:.1f} nm, energy={energy_eV:.1f} eV, contrast=beta_pos",
        f"box={BOX_SIZE_NM:.1f} nm x {BOX_SIZE_NM:.1f} nm, normalization=first q>0",
        f"assert_window=[{Q_ASSERT_MIN:.2f}, {Q_ASSERT_MAX:.2f}] nm^-1, plot_window=[{Q_PLOT_MIN:.2f}, {Q_PLOT_MAX:.2f}] nm^-1",
    ]
    for phys_size_nm in PHYS_SIZES_NM:
        result = results_by_phys_size[phys_size_nm]
        shape = result["shape"]
        note_lines.append(
            f"PhysSize={phys_size_nm:g} nm shape={shape[0]}x{shape[1]}x{shape[2]} "
            f"p95={result['p95_log_abs']:.4f} max={result['max_log_abs']:.4f} "
            f"build={result['build_seconds']:.2f}s run={result['run_seconds']:.2f}s"
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

    fig.suptitle("Analytical 2D Disk Form-Factor Resolution Study")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = PLOT_DIR / f"disk_ff_d{int(round(diameter_nm))}_e{energy_eV:.1f}_resolution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


@pytest.mark.gpu
@pytest.mark.slow
def test_analytical_2d_disk_form_factor_resolution_study():
    """Development diagnostic: compare a 2D disk against analytical form factor across PhysSize."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for analytical 2D disk form-factor study.")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    results_by_phys_size: dict[float, dict[str, object]] = {}
    reference_q_prefix = None
    for phys_size_nm in PHYS_SIZES_NM:
        build_t0 = perf_counter()
        morph = _build_disk_morphology(
            diameter_nm=DISK_DIAMETER_NM,
            phys_size_nm=phys_size_nm,
            energies_eV=ENERGIES_EV,
            disk_oc_by_energy=DISK_OC_BY_ENERGY,
        )
        build_seconds = perf_counter() - build_t0

        run_t0 = perf_counter()
        data = morph.run(stdout=False, stderr=False, return_xarray=True)
        run_seconds = perf_counter() - run_t0
        iq_by_energy = _pyhyper_iq_by_energy(data)
        del morph, data
        gc.collect()

        q, sim_iq = iq_by_energy[ENERGIES_EV[0]]
        ana_binned_iq = _analytic_disk_form_factor_binned_iq(q_centers=q, diameter_nm=DISK_DIAMETER_NM)
        sim_norm, ana_binned_norm, q_norm = _normalize_to_first_q_gt_zero(
            q=q,
            sim_iq=sim_iq,
            ana_iq=ana_binned_iq,
        )

        assert_mask = np.logical_and.reduce(
            [
                q >= Q_ASSERT_MIN,
                q <= Q_ASSERT_MAX,
                np.isfinite(sim_norm),
                np.isfinite(ana_binned_norm),
                sim_norm > MIN_SIGNAL_FOR_LOG,
                ana_binned_norm > MIN_SIGNAL_FOR_LOG,
            ]
        )
        plot_mask = np.logical_and.reduce(
            [
                q >= Q_PLOT_MIN,
                q <= Q_PLOT_MAX,
                np.isfinite(sim_norm),
                np.isfinite(ana_binned_norm),
                sim_norm > 0.0,
                ana_binned_norm > 0.0,
            ]
        )
        if int(np.count_nonzero(assert_mask)) < 20:
            raise AssertionError(
                f"Insufficient q points in comparison window for analytical-disk study at PhysSize={phys_size_nm}."
            )
        if int(np.count_nonzero(plot_mask)) < 20:
            raise AssertionError(
                f"Insufficient q points in plot window for analytical-disk study at PhysSize={phys_size_nm}."
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            log_abs = np.abs(np.log10(sim_norm[assert_mask]) - np.log10(ana_binned_norm[assert_mask]))

        if reference_q_prefix is None:
            reference_q_prefix = np.asarray(q[:20], dtype=np.float64)
        else:
            current_prefix = np.asarray(q[:20], dtype=np.float64)
            if not np.allclose(current_prefix, reference_q_prefix, rtol=0.0, atol=1e-12):
                raise AssertionError(
                    f"Low-q grid drifted across PhysSize sweep. PhysSize={phys_size_nm}, "
                    f"reference prefix={reference_q_prefix[:5]}, current prefix={current_prefix[:5]}"
                )

        results_by_phys_size[phys_size_nm] = {
            "shape": _shape_for_phys_size(phys_size_nm),
            "q": q,
            "sim_norm": sim_norm,
            "ana_binned_norm": ana_binned_norm,
            "q_norm": float(q_norm),
            "p95_log_abs": float(np.percentile(log_abs, 95)),
            "max_log_abs": float(np.max(log_abs)),
            "n_comp": int(np.count_nonzero(assert_mask)),
            "build_seconds": float(build_seconds),
            "run_seconds": float(run_seconds),
            "assert_mask": assert_mask,
            "plot_mask": plot_mask,
        }

    q_norms = np.asarray([results_by_phys_size[p]["q_norm"] for p in PHYS_SIZES_NM], dtype=np.float64)
    assert float(np.max(q_norms) - np.min(q_norms)) <= 1e-12

    out = _write_disk_resolution_plot(
        results_by_phys_size=results_by_phys_size,
        diameter_nm=DISK_DIAMETER_NM,
        energy_eV=ENERGIES_EV[0],
    )
    assert out.exists()
