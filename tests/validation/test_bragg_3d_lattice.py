import gc
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.ndimage import maximum_filter

from tests.path_matrix import ComputationPath, get_computation_path
from tests.validation.lib.bragg import (
    Bragg3DCase,
    build_bragg_3d_case_morphology,
    has_visible_gpu,
    predict_bragg_spots_3d,
    radial_shells_from_spots,
    release_runtime_memory,
)


pytestmark = [pytest.mark.path_matrix, pytest.mark.reference_parity]


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = REPO_ROOT / "test-reports" / "bragg-3d-lattice-dev"
WRITE_VALIDATION_PLOTS = os.environ.get("NRSS_WRITE_VALIDATION_PLOTS", "").strip() == "1"

SHAPE = (256, 1024, 1024)
PHYS_SIZE_NM = 1.0
ENERGY_EV = 285.0
AZIMUTH_DEG = 27.0
TILT_X_DEG = 0.0
TILT_Y_DEG = 0.0
SUPERRESOLUTION = 2

PREDICTED_INTENSITY_FLOOR_RATIO = 1e-2
PIXEL_TOLERANCE = 3.0
ANNULUS_INNER_PIXELS = 4.0
ANNULUS_OUTER_PIXELS = 9.0
RADIAL_MATCH_PIXELS = 4.0
UNEXPECTED_PEAK_FLOOR_RATIO = 0.02
QZ_TOLERANCE_FACTOR = 3.5
RADIAL_EVIDENCE_BG_INNER_PIXELS = 6.0
RADIAL_EVIDENCE_BG_OUTER_PIXELS = 14.0
RADIAL_EVIDENCE_RATIO_MIN = 1.5

CASES = [
    Bragg3DCase(
        case_id="simple_cubic_a30",
        lattice_kind="simple_cubic",
        lattice_constant_nm=30.0,
        particle_diameter_nm=11.0,
        azimuth_deg=AZIMUTH_DEG,
        tilt_x_deg=TILT_X_DEG,
        tilt_y_deg=TILT_Y_DEG,
        shape=SHAPE,
        phys_size_nm=PHYS_SIZE_NM,
        superresolution=SUPERRESOLUTION,
        energy_eV=ENERGY_EV,
    ),
    Bragg3DCase(
        case_id="hcp_a45",
        lattice_kind="hcp",
        lattice_constant_nm=45.0,
        particle_diameter_nm=11.0,
        azimuth_deg=AZIMUTH_DEG,
        tilt_x_deg=TILT_X_DEG,
        tilt_y_deg=TILT_Y_DEG,
        shape=SHAPE,
        phys_size_nm=PHYS_SIZE_NM,
        superresolution=SUPERRESOLUTION,
        energy_eV=ENERGY_EV,
    ),
]
CASES_BY_ID = {case.case_id: case for case in CASES}

CASE_THRESHOLDS = {
    "simple_cubic_a30": {
        "peak_p95_abs_dq_max": 0.006,
        "peak_max_abs_dq_max": 0.006,
        "peak_ratio_min": 100.0,
        "radial_p95_abs_dq_max": 0.020,
        "radial_max_abs_dq_max": 0.021,
        "missed_shell_count_max": 0,
        "unexpected_peak_count_max": 0,
    },
    "hcp_a45": {
        "peak_p95_abs_dq_max": 0.006,
        "peak_max_abs_dq_max": 0.018,
        "peak_ratio_min": 15.0,
        "radial_p95_abs_dq_max": 0.025,
        "radial_max_abs_dq_max": 0.026,
        "missed_shell_count_max": 1,
        "unexpected_peak_count_max": 0,
    },
}


def _sanitize_scattering(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return np.clip(arr, 0.0, None)


def _to_numpy(arr) -> np.ndarray:
    if hasattr(arr, "get"):
        arr = arr.get()
    return np.asarray(arr)


def _path_runtime_kwargs(nrss_path: ComputationPath) -> dict[str, object]:
    return {
        "backend": nrss_path.backend,
        "backend_options": nrss_path.backend_options,
        "resident_mode": nrss_path.resident_mode,
        "ownership_policy": nrss_path.ownership_policy,
        "field_namespace": nrss_path.field_namespace,
    }


def _pyhyper_iq(scattering) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    from PyHyperScattering.integrate import NRSSIntegrator

    integrator = NRSSIntegrator(use_chunked_processing=False, force_np_backend=True)
    remeshed = integrator.integrateImageStack(
        scattering,
        phys_size_nm=PHYS_SIZE_NM,
        shape_zyx=SHAPE,
        energy_ev=ENERGY_EV,
    )
    if "chi" not in remeshed.dims:
        raise AssertionError("PyHyperScattering output missing chi dimension.")
    qdim = next((dim for dim in remeshed.dims if dim == "q" or dim.startswith("q")), None)
    if qdim is None:
        raise AssertionError("PyHyperScattering output missing q dimension.")
    iq = remeshed.sel(energy=float(ENERGY_EV)).mean("chi")
    return (
        np.asarray(iq.coords[qdim].values, dtype=np.float64),
        np.asarray(iq.values, dtype=np.float64),
        dict(remeshed.attrs),
    )


def _match_expected_spots(
    image: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    predicted_spots: list[dict[str, float]],
) -> dict[str, object]:
    qx = np.asarray(qx, dtype=np.float64)
    qy = np.asarray(qy, dtype=np.float64)
    image = _sanitize_scattering(image)
    qx_grid, qy_grid = np.meshgrid(qx, qy, indexing="xy")
    q_pixel = float(min(np.abs(qx[1] - qx[0]), np.abs(qy[1] - qy[0])))
    search_radius = PIXEL_TOLERANCE * q_pixel
    annulus_inner = ANNULUS_INNER_PIXELS * q_pixel
    annulus_outer = ANNULUS_OUTER_PIXELS * q_pixel

    matches = []
    for spot in predicted_spots:
        dist = np.hypot(qx_grid - float(spot["qx"]), qy_grid - float(spot["qy"]))
        local_mask = dist <= search_radius
        if not np.any(local_mask):
            raise AssertionError(f"No detector pixels cover expected spot {spot!r}.")

        masked = np.where(local_mask, image, -np.inf)
        peak_flat = int(np.argmax(masked))
        peak_idx = np.unravel_index(peak_flat, image.shape)
        found_qy = float(qy[peak_idx[0]])
        found_qx = float(qx[peak_idx[1]])
        peak_intensity = float(image[peak_idx])
        dq = float(np.hypot(found_qx - float(spot["qx"]), found_qy - float(spot["qy"])))

        annulus_mask = np.logical_and(dist >= annulus_inner, dist <= annulus_outer)
        background = float(np.median(image[annulus_mask])) if np.any(annulus_mask) else 0.0
        peak_ratio = peak_intensity / max(background, 1e-12)
        matches.append(
            {
                "h": int(spot["h"]),
                "k": int(spot["k"]),
                "l": int(spot["l"]),
                "qx_expected": float(spot["qx"]),
                "qy_expected": float(spot["qy"]),
                "qmag_expected": float(spot["qmag"]),
                "qx_found": found_qx,
                "qy_found": found_qy,
                "peak_intensity": peak_intensity,
                "peak_ratio": peak_ratio,
                "abs_dq": dq,
            }
        )

    abs_dq = np.asarray([match["abs_dq"] for match in matches], dtype=np.float64)
    peak_ratios = np.asarray([match["peak_ratio"] for match in matches], dtype=np.float64)
    return {
        "matches": matches,
        "q_pixel": q_pixel,
        "p95_abs_dq": float(np.percentile(abs_dq, 95)),
        "max_abs_dq": float(abs_dq.max()),
        "min_peak_ratio": float(peak_ratios.min()),
        "median_peak_ratio": float(np.median(peak_ratios)),
        "min_peak_intensity": float(min(match["peak_intensity"] for match in matches)),
    }


def _find_unexpected_peaks(
    image: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    predicted_spots: list[dict[str, float]],
    q_pixel: float,
    min_peak_intensity: float,
) -> list[dict[str, float]]:
    image = _sanitize_scattering(image)
    qx_grid, qy_grid = np.meshgrid(qx, qy, indexing="xy")
    qmag = np.hypot(qx_grid, qy_grid)
    predicted_shell_qmin = min(float(spot["qmag"]) for spot in predicted_spots)
    off_center = qmag >= (0.5 * predicted_shell_qmin)
    off_center_max = float(image[off_center].max()) if np.any(off_center) else 0.0
    threshold_abs = max(
        float(min_peak_intensity),
        off_center_max * UNEXPECTED_PEAK_FLOOR_RATIO,
        1e-12,
    )

    footprint = maximum_filter(image, size=5, mode="nearest")
    maxima = np.logical_and.reduce([image == footprint, image >= threshold_abs, off_center])
    candidates = np.argwhere(maxima)
    if candidates.size == 0:
        return []

    sort_order = np.argsort(image[candidates[:, 0], candidates[:, 1]])[::-1]
    candidates = candidates[sort_order]

    keep: list[tuple[int, int]] = []
    min_sep_pix2 = float((3.0 * PIXEL_TOLERANCE) ** 2)
    for row, col in candidates:
        if all((float(row - r) ** 2 + float(col - c) ** 2) > min_sep_pix2 for r, c in keep):
            keep.append((int(row), int(col)))

    q_tol = PIXEL_TOLERANCE * q_pixel
    unexpected = []
    for row, col in keep:
        qx0 = float(qx[col])
        qy0 = float(qy[row])
        dq_min = min(np.hypot(qx0 - float(spot["qx"]), qy0 - float(spot["qy"])) for spot in predicted_spots)
        if dq_min > q_tol:
            unexpected.append(
                {
                    "qx": qx0,
                    "qy": qy0,
                    "qmag": float(np.hypot(qx0, qy0)),
                    "intensity": float(image[row, col]),
                    "nearest_expected_abs_dq": float(dq_min),
                }
            )
    return unexpected


def _radial_shell_metrics(
    q: np.ndarray,
    iq: np.ndarray,
    shell_qs: np.ndarray,
    q_pixel: float,
) -> dict[str, object]:
    q = np.asarray(q, dtype=np.float64)
    iq = _sanitize_scattering(iq)
    tol = RADIAL_MATCH_PIXELS * q_pixel
    matches = []
    for shell_q in shell_qs:
        mask = np.abs(q - float(shell_q)) <= tol
        if not np.any(mask):
            continue
        idx_local = np.argmax(iq[mask])
        q_local = q[mask]
        q_found = float(q_local[idx_local])
        matches.append(
            {
                "q_expected": float(shell_q),
                "q_found": q_found,
                "abs_dq": abs(q_found - float(shell_q)),
            }
        )
    if not matches:
        raise AssertionError("No radial shell matches were found.")

    abs_dq = np.asarray([match["abs_dq"] for match in matches], dtype=np.float64)
    return {
        "matches": matches,
        "p95_abs_dq": float(np.percentile(abs_dq, 95)),
        "max_abs_dq": float(abs_dq.max()),
    }


def _classify_radial_shell_evidence(
    q: np.ndarray,
    iq: np.ndarray,
    shell_qs: np.ndarray,
    q_pixel: float,
) -> dict[str, object]:
    q = np.asarray(q, dtype=np.float64)
    iq = _sanitize_scattering(iq)
    tol = RADIAL_MATCH_PIXELS * q_pixel
    bg_inner = RADIAL_EVIDENCE_BG_INNER_PIXELS * q_pixel
    bg_outer = RADIAL_EVIDENCE_BG_OUTER_PIXELS * q_pixel

    hits: list[float] = []
    misses: list[float] = []
    details = []
    for shell_q in np.asarray(shell_qs, dtype=np.float64):
        local_mask = np.abs(q - float(shell_q)) <= tol
        if not np.any(local_mask):
            misses.append(float(shell_q))
            details.append({"q_expected": float(shell_q), "peak_ratio": 0.0, "status": "miss"})
            continue

        q_local = q[local_mask]
        iq_local = iq[local_mask]
        idx_local = int(np.argmax(iq_local))
        q_found = float(q_local[idx_local])
        peak_intensity = float(iq_local[idx_local])

        bg_mask = np.logical_and(np.abs(q - float(shell_q)) >= bg_inner, np.abs(q - float(shell_q)) <= bg_outer)
        background = float(np.median(iq[bg_mask])) if np.any(bg_mask) else 0.0
        peak_ratio = peak_intensity / max(background, 1e-12)
        is_hit = abs(q_found - float(shell_q)) <= tol and peak_ratio >= RADIAL_EVIDENCE_RATIO_MIN
        if is_hit:
            hits.append(float(shell_q))
        else:
            misses.append(float(shell_q))
        details.append(
            {
                "q_expected": float(shell_q),
                "q_found": q_found,
                "peak_intensity": peak_intensity,
                "peak_ratio": peak_ratio,
                "status": "hit" if is_hit else "miss",
            }
        )

    return {"hits": hits, "misses": misses, "details": details}


def _write_dev_plot(
    path_id: str,
    case: Bragg3DCase,
    metadata: dict[str, object],
    image: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    all_predicted_spots: list[dict[str, float]],
    predicted_spots: list[dict[str, float]],
    peak_metrics: dict[str, object],
    unexpected_peaks: list[dict[str, float]],
    radial_q: np.ndarray,
    radial_iq: np.ndarray,
    all_shell_qs: np.ndarray,
    shell_qs: np.ndarray,
    radial_metrics: dict[str, object],
    timing: dict[str, float],
) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    image = _sanitize_scattering(image)
    detector_log = np.log10(np.maximum(image, np.max(image) * 1e-8))
    radial_evidence = _classify_radial_shell_evidence(
        q=radial_q,
        iq=radial_iq,
        shell_qs=shell_qs,
        q_pixel=float(peak_metrics["q_pixel"]),
    )
    hit_shell_qs = [float(q) for q in radial_evidence["hits"]]
    missed_shell_qs = [float(q) for q in radial_evidence["misses"]]
    suppressed_shell_qs = [
        float(shell_q)
        for shell_q in np.asarray(all_shell_qs, dtype=np.float64)
        if not np.any(np.isclose(shell_qs, shell_q, rtol=0.0, atol=1e-12))
    ]

    fig = plt.figure(figsize=(13.5, 10.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.28, wspace=0.20)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ax0.imshow(metadata["xy_slice"], origin="lower", cmap="gray_r", interpolation="none")
    ax0.set_title(
        f"{case.case_id}: mid-XY slice\n"
        f"{metadata['sphere_count']} spheres, a={case.lattice_constant_nm:.0f} nm, phi={case.azimuth_deg:.1f} deg"
    )
    ax0.set_xlabel("X index")
    ax0.set_ylabel("Y index")

    ax1.imshow(metadata["xz_slice"], origin="lower", cmap="gray_r", interpolation="none", aspect="auto")
    ax1.set_title("Mid-XZ slice")
    ax1.set_xlabel("X index")
    ax1.set_ylabel("Z index")

    im = ax2.imshow(
        detector_log,
        origin="lower",
        cmap="magma",
        extent=[float(qx.min()), float(qx.max()), float(qy.min()), float(qy.max())],
        aspect="equal",
        interpolation="nearest",
    )
    ax2.scatter(
        [spot["qx"] for spot in all_predicted_spots],
        [spot["qy"] for spot in all_predicted_spots],
        s=9,
        marker="x",
        linewidths=0.5,
        color="#90caf9",
        alpha=0.35,
        label="all geometry-visible",
    )
    ax2.scatter(
        [spot["qx"] for spot in predicted_spots],
        [spot["qy"] for spot in predicted_spots],
        s=22,
        marker="x",
        linewidths=0.9,
        color="#4cc9f0",
        label="asserted subset",
    )
    ax2.scatter(
        [match["qx_found"] for match in peak_metrics["matches"]],
        [match["qy_found"] for match in peak_metrics["matches"]],
        s=46,
        facecolors="none",
        edgecolors="white",
        linewidths=0.8,
        label="matched",
    )
    if unexpected_peaks:
        ax2.scatter(
            [peak["qx"] for peak in unexpected_peaks],
            [peak["qy"] for peak in unexpected_peaks],
            s=44,
            marker="s",
            facecolors="none",
            edgecolors="#ff595e",
            linewidths=0.9,
            label="unexpected",
        )
    ax2.set_title("Detector with detector-visible reciprocal-lattice overlays")
    ax2.set_xlabel(r"$q_x$ (nm$^{-1}$)")
    ax2.set_ylabel(r"$q_y$ (nm$^{-1}$)")
    ax2.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="log10 I")

    radial_iq = np.asarray(radial_iq, dtype=np.float64)
    plot_mask = np.logical_and(np.isfinite(radial_iq), radial_iq > 0.0)
    ax3.plot(radial_q[plot_mask], radial_iq[plot_mask], color="#1d3557", linewidth=1.8, label="PyHyper azimuthal average")
    for shell_q in all_shell_qs:
        ax3.axvline(float(shell_q), color="#adb5bd", linestyle="-", linewidth=1.5, alpha=0.45, zorder=1)
    for shell_q in hit_shell_qs:
        ax3.axvline(float(shell_q), color="xkcd:bright green", linestyle="--", linewidth=1.5, alpha=0.95, zorder=4)
    for shell_q in suppressed_shell_qs:
        ax3.axvline(float(shell_q), color="xkcd:bright blue", linestyle="--", linewidth=1.5, alpha=0.95, zorder=3)
    for shell_q in missed_shell_qs:
        ax3.axvline(float(shell_q), color="xkcd:bright red", linestyle="--", linewidth=1.5, alpha=0.95, zorder=5)
    ax3.plot([], [], color="#adb5bd", linestyle="-", linewidth=1.5, label="all predicted shells")
    ax3.plot([], [], color="xkcd:bright green", linestyle="--", linewidth=1.5, label="found visible shells")
    ax3.plot([], [], color="xkcd:bright red", linestyle="--", linewidth=1.5, label="visible but not found")
    ax3.plot([], [], color="xkcd:bright blue", linestyle="--", linewidth=1.5, label="expected not visible")
    ax3.set_xlim(0.0, min(1.2, float(radial_q.max())))
    ax3.set_yscale("log")
    ax3.set_xlabel(r"$q$ (nm$^{-1}$)")
    ax3.set_ylabel("I(q)")
    ax3.set_title("Quasi-powder workup with expected shell positions")
    ax3.legend(loc="upper right", fontsize=8)

    summary = "\n".join(
        [
            f"expected spots: {len(predicted_spots)}",
            f"hit asserted shells: {len(hit_shell_qs)}; missed asserted shells: {len(missed_shell_qs)}",
            f"FF-suppressed shell lines: {len(suppressed_shell_qs)}",
            f"2D detector peak p95 |dq| = {peak_metrics['p95_abs_dq']:.5f} nm^-1",
            f"2D detector peak max |dq| = {peak_metrics['max_abs_dq']:.5f} nm^-1",
            f"min peak/background = {peak_metrics['min_peak_ratio']:.1f}",
            f"radial shell p95 |dq| = {radial_metrics['p95_abs_dq']:.5f} nm^-1",
            f"unexpected peaks = {len(unexpected_peaks)}",
            f"build={timing['build_seconds']:.2f}s run={timing['run_seconds']:.2f}s iq={timing['iq_seconds']:.2f}s",
        ]
    )
    fig.text(
        0.012,
        0.012,
        summary,
        fontsize=8.5,
        family="monospace",
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#bbbbbb"},
    )

    out = PLOT_DIR / f"{path_id}__{case.case_id}_bragg_3d_e{case.energy_eV:.1f}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _evaluate_case(case: Bragg3DCase, nrss_path: ComputationPath) -> dict[str, object]:
    t0 = perf_counter()
    morph, metadata = build_bragg_3d_case_morphology(case, **_path_runtime_kwargs(nrss_path))
    nz, ny, _ = case.shape
    metadata["xy_slice"] = _to_numpy(morph.materials[1].Vfrac[nz // 2].copy())
    metadata["xz_slice"] = _to_numpy(morph.materials[1].Vfrac[:, ny // 2, :].copy())
    morph.check_materials(quiet=True)
    morph.validate_all(quiet=True)
    t1 = perf_counter()

    scattering = morph.run(stdout=False, stderr=False, return_xarray=True, validate=False)
    t2 = perf_counter()

    image = _sanitize_scattering(scattering.sel(energy=float(case.energy_eV)).values)
    qx = np.asarray(scattering.coords["qx"].values, dtype=np.float64)
    qy = np.asarray(scattering.coords["qy"].values, dtype=np.float64)

    qz_tolerance = QZ_TOLERANCE_FACTOR * (2.0 * np.pi / (float(case.shape[0]) * float(case.phys_size_nm)))
    all_predicted_spots = predict_bragg_spots_3d(
        primitive_vectors_nm=np.asarray(metadata["primitive_vectors_nm"], dtype=np.float64),
        basis_frac=np.asarray(metadata["basis_frac"], dtype=np.float64),
        diameter_nm=case.particle_diameter_nm,
        qx=qx,
        qy=qy,
        energy_eV=case.energy_eV,
        intensity_floor_ratio=1e-12,
        qz_tolerance=qz_tolerance,
    )
    if not all_predicted_spots:
        raise AssertionError(f"No detector-visible 3D Bragg spots were predicted for case {case.case_id}.")

    all_shell_qs = radial_shells_from_spots(
        all_predicted_spots,
        q_merge_tolerance=2.5 * float(min(np.abs(qx[1] - qx[0]), np.abs(qy[1] - qy[0]))),
        q_key="q_abs_detector",
    )
    predicted_spots = predict_bragg_spots_3d(
        primitive_vectors_nm=np.asarray(metadata["primitive_vectors_nm"], dtype=np.float64),
        basis_frac=np.asarray(metadata["basis_frac"], dtype=np.float64),
        diameter_nm=case.particle_diameter_nm,
        qx=qx,
        qy=qy,
        energy_eV=case.energy_eV,
        intensity_floor_ratio=PREDICTED_INTENSITY_FLOOR_RATIO,
        qz_tolerance=qz_tolerance,
    )
    if not predicted_spots:
        raise AssertionError(f"No asserted 3D Bragg spots were found for case {case.case_id}.")

    peak_metrics = _match_expected_spots(image=image, qx=qx, qy=qy, predicted_spots=predicted_spots)
    unexpected_peaks = _find_unexpected_peaks(
        image=image,
        qx=qx,
        qy=qy,
        predicted_spots=all_predicted_spots,
        q_pixel=float(peak_metrics["q_pixel"]),
        min_peak_intensity=float(peak_metrics["min_peak_intensity"]),
    )

    t3 = perf_counter()
    radial_q, radial_iq, radial_attrs = _pyhyper_iq(scattering)
    t4 = perf_counter()
    shell_qs = radial_shells_from_spots(
        predicted_spots,
        q_merge_tolerance=2.5 * float(peak_metrics["q_pixel"]),
        q_key="q_abs_detector",
    )
    radial_metrics = _radial_shell_metrics(
        q=radial_q,
        iq=radial_iq,
        shell_qs=shell_qs,
        q_pixel=float(peak_metrics["q_pixel"]),
    )
    radial_evidence = _classify_radial_shell_evidence(
        q=radial_q,
        iq=radial_iq,
        shell_qs=shell_qs,
        q_pixel=float(peak_metrics["q_pixel"]),
    )

    timing = {
        "build_seconds": t1 - t0,
        "run_seconds": t2 - t1,
        "iq_seconds": t4 - t3,
    }

    return {
        "case": case,
        "metadata": metadata,
        "image": image,
        "qx": qx,
        "qy": qy,
        "all_predicted_spots": all_predicted_spots,
        "predicted_spots": predicted_spots,
        "all_shell_qs": all_shell_qs,
        "peak_metrics": peak_metrics,
        "unexpected_peaks": unexpected_peaks,
        "radial_q": radial_q,
        "radial_iq": radial_iq,
        "radial_attrs": radial_attrs,
        "shell_qs": shell_qs,
        "radial_metrics": radial_metrics,
        "radial_evidence": radial_evidence,
        "timing": timing,
    }


def _assert_case_result(case: Bragg3DCase, result: dict[str, object]) -> None:
    thresholds = CASE_THRESHOLDS[case.case_id]
    peak_metrics = result["peak_metrics"]
    radial_metrics = result["radial_metrics"]
    radial_evidence = result["radial_evidence"]
    radial_attrs = result["radial_attrs"]
    unexpected_peaks = result["unexpected_peaks"]

    assert radial_attrs["source_integrator"] == "NRSSIntegrator"
    assert radial_attrs["nrss_semantic_mode"] == "3d_detector_aware"
    assert radial_attrs["radial_semantics"] == "q_abs_detector_corrected"
    assert peak_metrics["p95_abs_dq"] <= thresholds["peak_p95_abs_dq_max"]
    assert peak_metrics["max_abs_dq"] <= thresholds["peak_max_abs_dq_max"]
    assert peak_metrics["min_peak_ratio"] >= thresholds["peak_ratio_min"]
    assert radial_metrics["p95_abs_dq"] <= thresholds["radial_p95_abs_dq_max"]
    assert radial_metrics["max_abs_dq"] <= thresholds["radial_max_abs_dq_max"]
    assert len(radial_evidence["misses"]) <= thresholds["missed_shell_count_max"]
    assert len(unexpected_peaks) <= thresholds["unexpected_peak_count_max"]


def _run_validated_case(case_id: str, path_id: str) -> None:
    nrss_path = get_computation_path(path_id)
    case = CASES_BY_ID[case_id]
    result = _evaluate_case(case, nrss_path)

    print(
        "peak_metrics",
        {
            "p95_abs_dq": result["peak_metrics"]["p95_abs_dq"],
            "max_abs_dq": result["peak_metrics"]["max_abs_dq"],
            "min_peak_ratio": result["peak_metrics"]["min_peak_ratio"],
            "median_peak_ratio": result["peak_metrics"]["median_peak_ratio"],
        },
    )
    print(
        "radial_metrics",
        {
            "p95_abs_dq": result["radial_metrics"]["p95_abs_dq"],
            "max_abs_dq": result["radial_metrics"]["max_abs_dq"],
        },
    )
    print(
        f"{case.case_id}: spheres={result['metadata']['sphere_count']}, "
        f"predicted_spots={len(result['predicted_spots'])}, unexpected_peaks={len(result['unexpected_peaks'])}"
    )

    if WRITE_VALIDATION_PLOTS:
        _write_dev_plot(
            path_id=nrss_path.id,
            case=case,
            metadata=result["metadata"],
            image=result["image"],
            qx=result["qx"],
            qy=result["qy"],
            all_predicted_spots=result["all_predicted_spots"],
            predicted_spots=result["predicted_spots"],
            peak_metrics=result["peak_metrics"],
            unexpected_peaks=result["unexpected_peaks"],
            radial_q=result["radial_q"],
            radial_iq=result["radial_iq"],
            all_shell_qs=result["all_shell_qs"],
            shell_qs=result["shell_qs"],
            radial_metrics=result["radial_metrics"],
            timing=result["timing"],
        )

    _assert_case_result(case, result)

    del result
    gc.collect()
    release_runtime_memory()


def _run_case_subprocess(case_id: str, nrss_path: ComputationPath) -> None:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["NRSS_PATH"] = nrss_path.id
    env["NRSS_BACKEND"] = nrss_path.backend
    code = (
        "from tests.validation.test_bragg_3d_lattice import "
        f"_run_validated_case; _run_validated_case({case_id!r}, {nrss_path.id!r})"
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
        f"Isolated 3D Bragg lattice run failed for {case_id} with exit code {result.returncode}."
    )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.physics_validation
@pytest.mark.toolchain_validation
@pytest.mark.parametrize("case_id", [case.case_id for case in CASES])
def test_bragg_3d_lattice_pybind(case_id: str, nrss_path: ComputationPath):
    """Validate 3D Bragg peak positions for deterministic simple-cubic and HCP sphere lattices."""
    if not has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for 3D Bragg lattice validation.")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    _run_case_subprocess(case_id, nrss_path)
