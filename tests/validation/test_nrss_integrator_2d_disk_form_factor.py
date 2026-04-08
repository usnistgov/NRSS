import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from tests.path_matrix import ComputationPath
from tests.validation.test_analytical_2d_disk_form_factor import (
    ENERGY_EV,
    GEOMETRY_THRESHOLDS_BY_DIAMETER,
    PHYS_SIZE_NM,
    Q_EXTREMA_MAX,
    Q_EXTREMA_MIN,
    Q_PLOT_MAX,
    Q_PLOT_MIN,
    SHAPE,
    _analytic_disk_form_factor_binned_iq,
    _has_visible_gpu,
    _minima_alignment_metrics,
    _normalize_to_first_q_gt_zero,
    _path_runtime_kwargs,
    _pointwise_metrics,
    _release_runtime_memory,
    _run_disk_backend,
)


pytestmark = [pytest.mark.path_matrix]


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = REPO_ROOT / "test-reports" / "nrss-integrator-disk-2d-dev"
WRITE_VALIDATION_PLOTS = os.environ.get("NRSS_WRITE_VALIDATION_PLOTS", "").strip() == "1"

DIAMETERS_NM = [70.0]
SUPERRESOLUTION = 1


def _reduce_wp_iq(scattering):
    from PyHyperScattering.integrate import WPIntegrator

    reduced = WPIntegrator(use_chunked_processing=False, force_np_backend=True).integrateImageStack(scattering)
    iq = reduced.sel(energy=float(ENERGY_EV)).mean("chi")
    q = np.asarray(iq.coords["q"].values, dtype=np.float64)
    return reduced, q, np.asarray(iq.values, dtype=np.float64)


def _reduce_nrss_iq(scattering):
    from PyHyperScattering.integrate import NRSSIntegrator

    reduced = NRSSIntegrator(use_chunked_processing=False, force_np_backend=True).integrateImageStack(scattering)
    iq = reduced.sel(energy=float(ENERGY_EV)).mean("chi")
    q = np.asarray(iq.coords["q"].values, dtype=np.float64)
    return reduced, q, np.asarray(iq.values, dtype=np.float64)


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
    q: np.ndarray,
    ref_norm: np.ndarray,
    wp_norm: np.ndarray,
    nrss_norm: np.ndarray,
    wp_metrics: dict[str, float],
    nrss_metrics: dict[str, float],
    wp_minima: dict[str, float],
    nrss_minima: dict[str, float],
) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.0, 10.0), constrained_layout=True)
    axes = fig.subplot_mosaic(
        [
            ["detector", "detector"],
            ["curves", "curves"],
            ["resid", "delta"],
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
    ax.set_title("2D Detector Image in qx/qy")
    ax.set_xlabel(r"$q_x$ [nm$^{-1}$]")
    ax.set_ylabel(r"$q_y$ [nm$^{-1}$]")

    plot_mask = np.logical_and.reduce(
        [
            q >= Q_PLOT_MIN,
            q <= Q_PLOT_MAX,
            np.isfinite(ref_norm),
            np.isfinite(wp_norm),
            np.isfinite(nrss_norm),
            ref_norm > 0.0,
            wp_norm > 0.0,
            nrss_norm > 0.0,
        ]
    )

    ax = axes["curves"]
    ax.plot(q[plot_mask], ref_norm[plot_mask], color="black", linewidth=1.8, label="Analytical 2D disk I(q)")
    ax.plot(q[plot_mask], wp_norm[plot_mask], color="#1f77b4", linewidth=1.5, label="WPIntegrator")
    ax.plot(
        q[plot_mask],
        nrss_norm[plot_mask],
        color="#d62728",
        linewidth=1.3,
        linestyle="--",
        label="NRSSIntegrator",
    )
    ax.set_yscale("log")
    ax.set_xlim(Q_PLOT_MIN, Q_PLOT_MAX)
    ax.set_title("2D Disk Form-Factor Reduction")
    ax.set_xlabel(r"$q$ [nm$^{-1}$]")
    ax.set_ylabel("Normalized I(q)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax = axes["resid"]
    resid_mask = np.logical_and.reduce([plot_mask, ref_norm > 0.0, wp_norm > 0.0, nrss_norm > 0.0])
    ax.plot(
        q[resid_mask],
        np.log10(wp_norm[resid_mask]) - np.log10(ref_norm[resid_mask]),
        color="#1f77b4",
        linewidth=1.2,
        label="WP - analytical",
    )
    ax.plot(
        q[resid_mask],
        np.log10(nrss_norm[resid_mask]) - np.log10(ref_norm[resid_mask]),
        color="#d62728",
        linewidth=1.2,
        linestyle="--",
        label="NRSS - analytical",
    )
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlim(Q_PLOT_MIN, Q_PLOT_MAX)
    ax.set_title("Log Residuals vs Analytical 2D Disk")
    ax.set_xlabel(r"$q$ [nm$^{-1}$]")
    ax.set_ylabel(r"$\Delta \log_{10} I$")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax = axes["delta"]
    ax.plot(q[resid_mask], nrss_norm[resid_mask] - wp_norm[resid_mask], color="#6a3d9a", linewidth=1.2)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    for qmin in nrss_minima["ref_minima"]:
        ax.axvline(float(qmin), color="black", linewidth=0.8, linestyle="--", alpha=0.25)
    ax.set_xlim(Q_EXTREMA_MIN, Q_EXTREMA_MAX)
    ax.set_title("NRSSIntegrator - WPIntegrator")
    ax.set_xlabel(r"$q$ [nm$^{-1}$]")
    ax.set_ylabel(r"$\Delta I$")
    ax.grid(alpha=0.25)

    note_lines = [
        (
            f"path={path_id}, diameter={diameter_nm:.1f} nm, energy={ENERGY_EV:.1f} eV, "
            f"shape={SHAPE[0]}x{SHAPE[1]}x{SHAPE[2]}, PhysSize={PHYS_SIZE_NM:.2f} nm"
        ),
        (
            f"WP: rms_log={wp_metrics['rms_log']:.4f}, min_mae={wp_minima['mae_abs_dq']:.5f}"
        ),
        (
            f"NRSS: rms_log={nrss_metrics['rms_log']:.4f}, min_mae={nrss_minima['mae_abs_dq']:.5f}"
        ),
        "Expected 2D behavior: NRSSIntegrator == WPIntegrator and both follow the analytical 2D disk form factor.",
    ]
    fig.suptitle("NRSSIntegrator 2D Disk Form-Factor Validation")
    fig.text(
        0.01,
        0.01,
        "\n".join(note_lines),
        fontsize=8.5,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )
    out = PLOT_DIR / f"{path_id}__nrss_integrator_disk_2d_d{int(round(diameter_nm))}_e{ENERGY_EV:.1f}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.physics_validation
@pytest.mark.toolchain_validation
@pytest.mark.path_subset("cupy_tensor_coeff", "cupy_direct_polarization")
@pytest.mark.parametrize("diameter_nm", DIAMETERS_NM, ids=["dia70"])
def test_nrss_integrator_2d_disk_form_factor_visualization(
    diameter_nm: float,
    nrss_path: ComputationPath,
):
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for NRSSIntegrator 2D disk validation.")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    runtime_kwargs = _path_runtime_kwargs(nrss_path)

    scattering = _run_disk_backend(
        diameter_nm=diameter_nm,
        superresolution=SUPERRESOLUTION,
        energies_eV=[ENERGY_EV],
        runtime_kwargs=runtime_kwargs,
    )
    try:
        _, wp_q, wp_iq = _reduce_wp_iq(scattering)
        reduced_nrss, nrss_q, nrss_iq = _reduce_nrss_iq(scattering)
        detector_image, qx, qy = _detector_image(scattering)
    finally:
        del scattering
        _release_runtime_memory()

    ref_iq = _analytic_disk_form_factor_binned_iq(q_centers=wp_q, diameter_nm=diameter_nm)
    wp_norm, ref_norm, _ = _normalize_to_first_q_gt_zero(wp_q, wp_iq, ref_iq)
    nrss_norm, _, _ = _normalize_to_first_q_gt_zero(nrss_q, nrss_iq, ref_iq)

    wp_metrics = _pointwise_metrics(wp_q, wp_norm, ref_norm)
    nrss_metrics = _pointwise_metrics(nrss_q, nrss_norm, ref_norm)
    wp_minima = _minima_alignment_metrics(wp_q, wp_norm, ref_norm)
    nrss_minima = _minima_alignment_metrics(nrss_q, nrss_norm, ref_norm)

    print("wp_metrics", wp_metrics)
    print("nrss_metrics", nrss_metrics)
    print("wp_minima", {k: v for k, v in wp_minima.items() if not isinstance(v, np.ndarray)})
    print("nrss_minima", {k: v for k, v in nrss_minima.items() if not isinstance(v, np.ndarray)})

    if WRITE_VALIDATION_PLOTS:
        _write_validation_plot(
            diameter_nm=diameter_nm,
            path_id=nrss_path.id,
            detector_image=detector_image,
            qx=qx,
            qy=qy,
            q=wp_q,
            ref_norm=ref_norm,
            wp_norm=wp_norm,
            nrss_norm=nrss_norm,
            wp_metrics=wp_metrics,
            nrss_metrics=nrss_metrics,
            wp_minima=wp_minima,
            nrss_minima=nrss_minima,
        )

    thresholds = GEOMETRY_THRESHOLDS_BY_DIAMETER[float(diameter_nm)]
    np.testing.assert_allclose(nrss_q, wp_q, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(nrss_norm, wp_norm, rtol=1e-12, atol=1e-12)
    assert reduced_nrss.attrs["nrss_semantic_mode"] == "2d_reciprocal_plane"
    assert reduced_nrss.attrs["radial_semantics"] == "q_perp"
    assert wp_metrics["rms_log"] <= thresholds["sr1_rms_log_max"]
    assert wp_metrics["p95_log_abs"] <= thresholds["sr1_p95_log_abs_max"]
    assert wp_minima["mae_abs_dq"] <= thresholds["sr1_min_mae_max"]
    assert wp_minima["rmse_abs_dq"] <= thresholds["sr1_min_rmse_max"]
