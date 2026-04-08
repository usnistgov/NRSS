import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from tests.path_matrix import ComputationPath
from tests.validation.test_analytical_sphere_form_factor import (
    ENERGY_EV,
    PHYS_SIZE_NM,
    Q_EXTREMA_MAX,
    Q_EXTREMA_MIN,
    Q_PLOT_MAX,
    Q_PLOT_MIN,
    SHAPE,
    _analytic_sphere_form_factor_binned_iq,
    _flat_detector_analytic_image,
    _has_visible_gpu,
    _minima_alignment_metrics,
    _normalize_to_first_q_gt_zero,
    _path_runtime_kwargs,
    _pointwise_metrics,
    _release_runtime_memory,
    _run_sphere_backend,
)


pytestmark = [pytest.mark.path_matrix]


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = REPO_ROOT / "test-reports" / "nrss-integrator-sphere-dev"
WRITE_VALIDATION_PLOTS = os.environ.get("NRSS_WRITE_VALIDATION_PLOTS", "").strip() == "1"

DIAMETERS_NM = [70.0, 128.0]
SUPERRESOLUTION = 1


def _reduce_wp_iq(scattering):
    from PyHyperScattering.integrate import WPIntegrator

    reduced = WPIntegrator(use_chunked_processing=False, force_np_backend=True).integrateImageStack(scattering)
    iq = reduced.sel(energy=float(ENERGY_EV)).mean("chi")
    q = np.asarray(iq.coords["q"].values, dtype=np.float64)
    return reduced, q, np.asarray(iq.values, dtype=np.float64)


def _reduce_nrss_iq(scattering):
    from PyHyperScattering.integrate import NRSSIntegrator

    reduced = NRSSIntegrator(use_chunked_processing=False, force_np_backend=True).integrateImageStack(
        scattering,
        phys_size_nm=PHYS_SIZE_NM,
        shape_zyx=SHAPE,
        energy_ev=ENERGY_EV,
    )
    iq = reduced.sel(energy=float(ENERGY_EV)).mean("chi")
    if "q_abs" in iq.coords:
        q = np.asarray(iq.coords["q_abs"].values, dtype=np.float64)
    else:
        q = np.asarray(iq.coords["q"].values, dtype=np.float64)
    q_perp = np.asarray(iq.coords["q_perp"].values, dtype=np.float64)
    return reduced, q, q_perp, np.asarray(iq.values, dtype=np.float64)


def _detector_image(scattering) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(scattering.sel(energy=float(ENERGY_EV)).values, dtype=np.float64)
    qx = np.asarray(scattering.coords["qx"].values, dtype=np.float64)
    qy = np.asarray(scattering.coords["qy"].values, dtype=np.float64)
    return image, qx, qy


def _write_validation_plot(
    *,
    diameter_nm: float,
    path_id: str,
    detector_image: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    wp_q: np.ndarray,
    wp_norm: np.ndarray,
    wp_flat_norm: np.ndarray,
    wp_direct_norm: np.ndarray,
    nrss_q: np.ndarray,
    nrss_norm: np.ndarray,
    nrss_direct_norm: np.ndarray,
    wp_direct_minima: dict[str, float],
    nrss_direct_minima: dict[str, float],
    wp_direct_metrics: dict[str, float],
    nrss_direct_metrics: dict[str, float],
    wp_flat_metrics: dict[str, float],
) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.0, 10.0), constrained_layout=True)
    axes = fig.subplot_mosaic(
        [
            ["detector", "detector"],
            ["wp", "nrss"],
            ["resid", "minima"],
        ]
    )

    ax = axes["detector"]
    display = np.where(detector_image > 0.0, np.log10(detector_image), np.nan)
    im = ax.imshow(
        display,
        origin="lower",
        extent=[float(qx[0]), float(qx[-1]), float(qy[0]), float(qy[-1])],
        aspect="equal",
        cmap="magma",
    )
    fig.colorbar(im, ax=ax, label=r"$\log_{10} I(q_x, q_y)$")
    ax.set_title("Detector Image in qx/qy")
    ax.set_xlabel(r"$q_x$ [nm$^{-1}$]")
    ax.set_ylabel(r"$q_y$ [nm$^{-1}$]")

    ax = axes["wp"]
    wp_mask = np.logical_and.reduce(
        [
            wp_q >= Q_PLOT_MIN,
            wp_q <= Q_PLOT_MAX,
            np.isfinite(wp_norm),
            np.isfinite(wp_flat_norm),
            np.isfinite(wp_direct_norm),
            wp_norm > 0.0,
            wp_flat_norm > 0.0,
            wp_direct_norm > 0.0,
        ]
    )
    ax.plot(wp_q[wp_mask], wp_norm[wp_mask], color="#1f77b4", linewidth=1.8, label="WPIntegrator")
    ax.plot(
        wp_q[wp_mask],
        wp_flat_norm[wp_mask],
        color="black",
        linewidth=1.6,
        linestyle="--",
        label="Detector-space expectation",
    )
    ax.plot(
        wp_q[wp_mask],
        wp_direct_norm[wp_mask],
        color="#7f7f7f",
        linewidth=1.2,
        linestyle=":",
        label="Pure analytical I(|q|)",
    )
    ax.set_yscale("log")
    ax.set_xlim(Q_PLOT_MIN, Q_PLOT_MAX)
    ax.set_title("WPIntegrator Reduction")
    ax.set_xlabel(r"$q_\perp$ [nm$^{-1}$]")
    ax.set_ylabel("Normalized I(q)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax = axes["nrss"]
    nrss_mask = np.logical_and.reduce(
        [
            nrss_q >= Q_PLOT_MIN,
            nrss_q <= Q_PLOT_MAX,
            np.isfinite(nrss_norm),
            np.isfinite(nrss_direct_norm),
            nrss_norm > 0.0,
            nrss_direct_norm > 0.0,
        ]
    )
    ax.plot(nrss_q[nrss_mask], nrss_norm[nrss_mask], color="#d62728", linewidth=1.8, label="NRSSIntegrator")
    ax.plot(
        nrss_q[nrss_mask],
        nrss_direct_norm[nrss_mask],
        color="black",
        linewidth=1.6,
        linestyle="--",
        label="Pure analytical I(|q|)",
    )
    ax.set_yscale("log")
    ax.set_xlim(Q_PLOT_MIN, Q_PLOT_MAX)
    ax.set_title("NRSSIntegrator Reduction")
    ax.set_xlabel(r"$|q|$ [nm$^{-1}$]")
    ax.set_ylabel("Normalized I(q)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax = axes["resid"]
    wp_resid_mask = np.logical_and.reduce(
        [wp_mask, wp_norm > 0.0, wp_direct_norm > 0.0, wp_flat_norm > 0.0]
    )
    nrss_resid_mask = np.logical_and.reduce(
        [nrss_mask, nrss_norm > 0.0, nrss_direct_norm > 0.0]
    )
    ax.plot(
        wp_q[wp_resid_mask],
        np.log10(wp_norm[wp_resid_mask]) - np.log10(wp_direct_norm[wp_resid_mask]),
        color="#1f77b4",
        linewidth=1.2,
        label="WP vs analytical |q|",
    )
    ax.plot(
        wp_q[wp_resid_mask],
        np.log10(wp_norm[wp_resid_mask]) - np.log10(wp_flat_norm[wp_resid_mask]),
        color="black",
        linewidth=1.2,
        linestyle="--",
        label="WP vs detector-space",
    )
    ax.plot(
        nrss_q[nrss_resid_mask],
        np.log10(nrss_norm[nrss_resid_mask]) - np.log10(nrss_direct_norm[nrss_resid_mask]),
        color="#d62728",
        linewidth=1.2,
        label="NRSS vs analytical |q|",
    )
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlim(Q_PLOT_MIN, Q_PLOT_MAX)
    ax.set_title("Log Residuals")
    ax.set_xlabel(r"q [nm$^{-1}$]")
    ax.set_ylabel(r"$\Delta \log_{10} I$")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax = axes["minima"]
    ax.plot(wp_q[wp_mask], wp_direct_norm[wp_mask], color="#bdbdbd", linewidth=1.0, label="Analytical |q|")
    ax.plot(wp_q[wp_mask], wp_norm[wp_mask], color="#1f77b4", linewidth=1.3, alpha=0.85, label="WPIntegrator")
    ax.plot(
        nrss_q[nrss_mask],
        nrss_norm[nrss_mask],
        color="#d62728",
        linewidth=1.3,
        alpha=0.85,
        label="NRSSIntegrator",
    )
    for qmin in wp_direct_minima["sim_minima"]:
        ax.axvline(float(qmin), color="#1f77b4", linewidth=0.8, alpha=0.35)
    for qmin in nrss_direct_minima["sim_minima"]:
        ax.axvline(float(qmin), color="#d62728", linewidth=0.8, alpha=0.35)
    for qmin in nrss_direct_minima["ref_minima"]:
        ax.axvline(float(qmin), color="black", linewidth=0.9, linestyle="--", alpha=0.25)
    ax.set_xlim(Q_EXTREMA_MIN, Q_EXTREMA_MAX)
    ax.set_yscale("log")
    ax.set_title("Minima Alignment Markers")
    ax.set_xlabel(r"q [nm$^{-1}$]")
    ax.set_ylabel("Normalized I(q)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    note_lines = [
        (
            f"path={path_id}, diameter={diameter_nm:.1f} nm, energy={ENERGY_EV:.1f} eV, "
            f"shape={SHAPE[0]}x{SHAPE[1]}x{SHAPE[2]}, PhysSize={PHYS_SIZE_NM:.2f} nm"
        ),
        (
            f"WP vs detector-space: rms_log={wp_flat_metrics['rms_log']:.4f}, "
            f"p95={wp_flat_metrics['p95_log_abs']:.4f}"
        ),
        (
            f"WP vs analytical |q|: rms_log={wp_direct_metrics['rms_log']:.4f}, "
            f"min_mae={wp_direct_minima['mae_abs_dq']:.5f}"
        ),
        (
            f"NRSS vs analytical |q|: rms_log={nrss_direct_metrics['rms_log']:.4f}, "
            f"min_mae={nrss_direct_minima['mae_abs_dq']:.5f}"
        ),
    ]
    fig.suptitle("NRSSIntegrator Sphere Form-Factor Validation")
    fig.text(
        0.01,
        0.01,
        "\n".join(note_lines),
        fontsize=8.5,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )

    out = PLOT_DIR / f"{path_id}__nrss_integrator_sphere_d{int(round(diameter_nm))}_e{ENERGY_EV:.1f}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.physics_validation
@pytest.mark.toolchain_validation
@pytest.mark.path_subset("cupy_tensor_coeff", "cupy_direct_polarization")
@pytest.mark.parametrize("diameter_nm", DIAMETERS_NM, ids=["dia70", "dia128"])
def test_nrss_integrator_sphere_form_factor_visualization(
    diameter_nm: float,
    nrss_path: ComputationPath,
):
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for NRSSIntegrator sphere-form-factor validation.")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    runtime_kwargs = _path_runtime_kwargs(nrss_path)

    scattering = _run_sphere_backend(
        diameter_nm=diameter_nm,
        superresolution=SUPERRESOLUTION,
        energies_eV=[ENERGY_EV],
        runtime_kwargs=runtime_kwargs,
    )
    try:
        _, wp_q, wp_iq = _reduce_wp_iq(scattering)
        _, nrss_q, _, nrss_iq = _reduce_nrss_iq(scattering)
        detector_image, qx, qy = _detector_image(scattering)
    finally:
        del scattering
        _release_runtime_memory()

    flat_image = _flat_detector_analytic_image(
        diameter_nm=diameter_nm,
        energy_eV=ENERGY_EV,
    )
    _, wp_q_flat, wp_flat_ref = _reduce_wp_iq(flat_image)
    if not np.allclose(wp_q, wp_q_flat, atol=1e-12, rtol=0.0):
        raise AssertionError("WPIntegrator q grid does not match detector-space analytical reduction.")

    wp_direct_ref = _analytic_sphere_form_factor_binned_iq(q_centers=wp_q, diameter_nm=diameter_nm)
    nrss_direct_ref = _analytic_sphere_form_factor_binned_iq(q_centers=nrss_q, diameter_nm=diameter_nm)

    wp_norm, wp_flat_norm, _ = _normalize_to_first_q_gt_zero(wp_q, wp_iq, wp_flat_ref)
    _, wp_direct_norm, _ = _normalize_to_first_q_gt_zero(wp_q, wp_iq, wp_direct_ref)
    nrss_norm, nrss_direct_norm, _ = _normalize_to_first_q_gt_zero(nrss_q, nrss_iq, nrss_direct_ref)

    wp_flat_metrics = _pointwise_metrics(wp_q, wp_norm, wp_flat_norm)
    wp_direct_metrics = _pointwise_metrics(wp_q, wp_norm, wp_direct_norm)
    nrss_direct_metrics = _pointwise_metrics(nrss_q, nrss_norm, nrss_direct_norm)

    wp_direct_minima = _minima_alignment_metrics(wp_q, wp_norm, wp_direct_norm)
    nrss_direct_minima = _minima_alignment_metrics(nrss_q, nrss_norm, nrss_direct_norm)

    print("wp_flat_metrics", wp_flat_metrics)
    print("wp_direct_metrics", wp_direct_metrics)
    print("nrss_direct_metrics", nrss_direct_metrics)
    print(
        "wp_direct_minima",
        {k: v for k, v in wp_direct_minima.items() if not isinstance(v, np.ndarray)},
    )
    print(
        "nrss_direct_minima",
        {k: v for k, v in nrss_direct_minima.items() if not isinstance(v, np.ndarray)},
    )

    if WRITE_VALIDATION_PLOTS:
        _write_validation_plot(
            diameter_nm=diameter_nm,
            path_id=nrss_path.id,
            detector_image=detector_image,
            qx=qx,
            qy=qy,
            wp_q=wp_q,
            wp_norm=wp_norm,
            wp_flat_norm=wp_flat_norm,
            wp_direct_norm=wp_direct_norm,
            nrss_q=nrss_q,
            nrss_norm=nrss_norm,
            nrss_direct_norm=nrss_direct_norm,
            wp_direct_minima=wp_direct_minima,
            nrss_direct_minima=nrss_direct_minima,
            wp_direct_metrics=wp_direct_metrics,
            nrss_direct_metrics=nrss_direct_metrics,
            wp_flat_metrics=wp_flat_metrics,
        )

    assert wp_flat_metrics["rms_log"] < wp_direct_metrics["rms_log"]
    assert nrss_direct_metrics["rms_log"] < wp_direct_metrics["rms_log"]
    assert nrss_direct_minima["mae_abs_dq"] < wp_direct_minima["mae_abs_dq"]
