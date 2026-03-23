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
PLOT_DIR = REPO_ROOT / "test-reports" / "sphere-projection-dev"

PHYS_SIZE_NM = 1.0
NX = 512
NY = 512
NUMZ_VALUES = [256, 384, 512]
INTERPOLATION_VALUES = [0, 1]
INTERPOLATION_LABELS = {
    0: "nearest",
    1: "linear",
}
INTERPOLATION_COLORS = {
    0: "#d62728",
    1: "#1f77b4",
}
NUMZ_COLORS = {
    256: "#1f77b4",
    384: "#2ca02c",
    512: "#ff7f0e",
}
DIAMETER_NM = 150.0
ENERGY_EV = 285.0
SPHERE_OC = {
    285.0: (0.0, 2e-4, 0.0, 2e-4),
}
Q_ASSERT_MIN = 0.06
Q_ASSERT_MAX = 1.0
Q_PLOT_MIN = 0.0
Q_PLOT_MAX = 1.0
MIN_SIGNAL_FOR_LOG = 1e-5


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


def _shape(numz: int) -> tuple[int, int, int]:
    return (int(numz), NY, NX)


def _cyrsoxs_detector_axis(n: int, phys_size_nm: float) -> np.ndarray:
    if int(n) < 2:
        raise AssertionError(f"CyRSoXS detector axis needs at least 2 points, got n={n}.")
    start = -np.pi / float(phys_size_nm)
    step = (2.0 * np.pi / float(phys_size_nm)) / float(int(n) - 1)
    return start + np.arange(int(n), dtype=np.float64) * step


def _with_cyrsoxs_detector_coords(scattering, phys_size_nm: float):
    if "qy" not in scattering.dims or "qx" not in scattering.dims:
        raise AssertionError("Scattering output missing qx/qy dimensions.")
    qy = _cyrsoxs_detector_axis(int(scattering.sizes["qy"]), phys_size_nm)
    qx = _cyrsoxs_detector_axis(int(scattering.sizes["qx"]), phys_size_nm)
    return scattering.assign_coords(qy=qy, qx=qx)


def _sphere_and_vacuum_vfrac(
    shape: tuple[int, int, int],
    diameter_nm: float,
    phys_size_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    nz, ny, nx = shape
    radius_vox = float(diameter_nm) / (2.0 * float(phys_size_nm))
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

    sphere = np.zeros(shape, dtype=np.float32)
    local_sphere = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=np.float32)
    local_sphere[dist2 <= np.float32(radius_vox * radius_vox)] = 1.0
    sphere[z0:z1, y0:y1, x0:x1] = local_sphere
    vacuum = (1.0 - sphere).astype(np.float32)
    return sphere, vacuum


def _build_sphere_morphology(
    numz: int,
    ewalds_interpolation: int,
) -> Morphology:
    shape = _shape(numz)
    sphere_vfrac, vacuum_vfrac = _sphere_and_vacuum_vfrac(
        shape=shape,
        diameter_nm=DIAMETER_NM,
        phys_size_nm=PHYS_SIZE_NM,
    )
    zeros = np.zeros(shape, dtype=np.float32)
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
        "EwaldsInterpolation": int(ewalds_interpolation),
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

    scattering = _with_cyrsoxs_detector_coords(scattering, PHYS_SIZE_NM)
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


def _run_case(numz: int, ewalds_interpolation: int) -> dict[str, object]:
    build_t0 = perf_counter()
    morph = _build_sphere_morphology(numz=numz, ewalds_interpolation=ewalds_interpolation)
    build_seconds = perf_counter() - build_t0

    run_t0 = perf_counter()
    data = morph.run(stdout=False, stderr=False, return_xarray=True)
    run_seconds = perf_counter() - run_t0

    q, sim_iq = _pyhyper_iq(data)
    ana_binned_iq = _analytic_sphere_form_factor_binned_iq(q, DIAMETER_NM)
    sim_norm, ana_binned_norm, q_norm = _normalize_to_first_q_gt_zero(q, sim_iq, ana_binned_iq)

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
    with np.errstate(divide="ignore", invalid="ignore"):
        log_abs = np.abs(np.log10(sim_norm[assert_mask]) - np.log10(ana_binned_norm[assert_mask]))

    return {
        "q": q,
        "sim_norm": sim_norm,
        "ana_binned_norm": ana_binned_norm,
        "q_norm": float(q_norm),
        "p95_log_abs": float(np.percentile(log_abs, 95)),
        "max_log_abs": float(np.max(log_abs)),
        "build_seconds": float(build_seconds),
        "run_seconds": float(run_seconds),
        "assert_mask": assert_mask,
        "plot_mask": plot_mask,
        "minima_q": _find_minima(q, sim_norm),
    }


def _write_overlay_plot(
    title: str,
    results: dict[str, dict[str, object]],
    colors: dict[str, str],
    out_name: str,
    note_lines: list[str],
) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.0, 8.0), sharex=True)

    reference_key = next(iter(results))
    q_ref = results[reference_key]["q"]
    ana_ref = results[reference_key]["ana_binned_norm"]
    plot_mask_ref = results[reference_key]["plot_mask"]
    assert_mask_ref = results[reference_key]["assert_mask"]

    ax0.plot(
        q_ref[plot_mask_ref],
        ana_ref[plot_mask_ref],
        color="black",
        linewidth=2.0,
        label="Analytical q-bin averaged",
    )

    y_all = [ana_ref[plot_mask_ref]]
    resid_all = []
    for label, result in results.items():
        q = result["q"]
        plot_mask = result["plot_mask"]
        assert_mask = result["assert_mask"]
        sim_norm = result["sim_norm"]
        ana_binned_norm = result["ana_binned_norm"]

        ax0.plot(
            q[plot_mask],
            sim_norm[plot_mask],
            color=colors[label],
            linewidth=1.5,
            alpha=0.65,
            label=label,
        )
        y_all.append(sim_norm[plot_mask])

        with np.errstate(divide="ignore", invalid="ignore"):
            resid = np.log10(sim_norm[assert_mask]) - np.log10(ana_binned_norm[assert_mask])
        ax1.plot(
            q[assert_mask],
            resid,
            color=colors[label],
            linewidth=1.25,
            alpha=0.65,
            label=label,
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
        ax0.set_ylim(max(float(np.min(y_all_flat)) * 0.8, 1e-12), float(np.max(y_all_flat)) * 1.2)

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

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = PLOT_DIR / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


@pytest.mark.gpu
@pytest.mark.slow
def test_sphere_projection_diagnostics():
    """Development diagnostic for 3D sphere sensitivity to Ewald interpolation and NumZ."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for sphere projection diagnostics.")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    interp_results = {}
    for interpolation in INTERPOLATION_VALUES:
        interp_results[INTERPOLATION_LABELS[interpolation]] = _run_case(
            numz=NUMZ_VALUES[0],
            ewalds_interpolation=interpolation,
        )

    numz_results = {}
    for numz in NUMZ_VALUES:
        numz_results[f"NumZ={numz}"] = _run_case(
            numz=numz,
            ewalds_interpolation=1,
        )

    interp_note_lines = [
        f"diameter={DIAMETER_NM:.1f} nm, energy={ENERGY_EV:.1f} eV, shape={NUMZ_VALUES[0]}x{NY}x{NX}, PhysSize={PHYS_SIZE_NM:.2f} nm",
        "q-grid=CyRSoXS detector emulation, reduction=PyHyperScattering",
    ]
    for interpolation in INTERPOLATION_VALUES:
        label = INTERPOLATION_LABELS[interpolation]
        result = interp_results[label]
        minima = ", ".join([f"{q:.4f}" for q in result["minima_q"][:4]])
        interp_note_lines.append(
            f"{label}: q_norm={result['q_norm']:.5f} p95={result['p95_log_abs']:.4f} max={result['max_log_abs']:.4f} "
            f"build={result['build_seconds']:.2f}s run={result['run_seconds']:.2f}s minima=[{minima}]"
        )

    interp_plot = _write_overlay_plot(
        title="3D Sphere Ewald Interpolation Diagnostic",
        results=interp_results,
        colors={INTERPOLATION_LABELS[k]: INTERPOLATION_COLORS[k] for k in INTERPOLATION_VALUES},
        out_name="sphere_projection_interp_d150_e285.png",
        note_lines=interp_note_lines,
    )

    numz_note_lines = [
        f"diameter={DIAMETER_NM:.1f} nm, energy={ENERGY_EV:.1f} eV, in-plane={NY}x{NX}, PhysSize={PHYS_SIZE_NM:.2f} nm",
        "EwaldsInterpolation=linear, q-grid=CyRSoXS detector emulation, reduction=PyHyperScattering",
    ]
    for numz in NUMZ_VALUES:
        label = f"NumZ={numz}"
        result = numz_results[label]
        minima = ", ".join([f"{q:.4f}" for q in result["minima_q"][:4]])
        numz_note_lines.append(
            f"{label}: q_norm={result['q_norm']:.5f} p95={result['p95_log_abs']:.4f} max={result['max_log_abs']:.4f} "
            f"build={result['build_seconds']:.2f}s run={result['run_seconds']:.2f}s minima=[{minima}]"
        )

    numz_plot = _write_overlay_plot(
        title="3D Sphere NumZ Projection Diagnostic",
        results=numz_results,
        colors={f"NumZ={n}": NUMZ_COLORS[n] for n in NUMZ_VALUES},
        out_name="sphere_projection_numz_d150_e285.png",
        note_lines=numz_note_lines,
    )

    assert interp_plot.exists()
    assert numz_plot.exists()
