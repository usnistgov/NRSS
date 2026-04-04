from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from tests.path_matrix import get_computation_path
from tests.validation.test_analytical_sphere_form_factor import (
    DIAMETERS_NM,
    ENERGY_EV,
    FLAT_DETECTOR_OVERSAMPLE,
    MIN_SIGNAL_FOR_LOG,
    Q_POINTWISE_MAX,
    Q_POINTWISE_MIN,
    Q_PLOT_MAX,
    Q_PLOT_MIN,
    _analytic_sphere_form_factor_binned_iq,
    _flat_detector_analytic_image,
    _has_visible_gpu,
    _minima_alignment_metrics,
    _normalize_to_first_q_gt_zero,
    _path_runtime_kwargs,
    _pointwise_metrics,
    _pyhyper_iq_by_energy,
    _release_runtime_memory,
    _run_sphere_backend,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "test-reports" / "cupy-rsoxs-z-collapse-sphere"
EXECUTION_PATH_TO_PATH_ID = {
    "tensor_coeff": "cupy_tensor_coeff",
    "direct_polarization": "cupy_direct_polarization",
}
DEFAULT_EXECUTION_PATHS = tuple(EXECUTION_PATH_TO_PATH_ID)


def _runtime_kwargs(execution_path: str, *, collapsed: bool) -> dict[str, object]:
    runtime_kwargs = _path_runtime_kwargs(
        get_computation_path(EXECUTION_PATH_TO_PATH_ID[execution_path])
    )
    if not collapsed:
        return runtime_kwargs
    backend_options = dict(runtime_kwargs["backend_options"])
    backend_options["z_collapse_mode"] = "mean"
    runtime_kwargs["backend_options"] = backend_options
    return runtime_kwargs


def _collapsed_runtime_kwargs(execution_path: str) -> dict[str, object]:
    return _runtime_kwargs(execution_path, collapsed=True)


def _baseline_runtime_kwargs(execution_path: str) -> dict[str, object]:
    return _runtime_kwargs(execution_path, collapsed=False)


def _run_iq(diameter_nm: float, superresolution: int, runtime_kwargs: dict[str, object]):
    sim_t0 = perf_counter()
    data = _run_sphere_backend(
        diameter_nm=diameter_nm,
        superresolution=superresolution,
        energies_eV=[ENERGY_EV],
        runtime_kwargs=runtime_kwargs,
    )
    sim_seconds = perf_counter() - sim_t0
    iq_t0 = perf_counter()
    q, iq = _pyhyper_iq_by_energy(data)[float(ENERGY_EV)]
    iq_seconds = perf_counter() - iq_t0
    del data
    _release_runtime_memory()
    return q, iq, {"sim_seconds": float(sim_seconds), "iq_seconds": float(iq_seconds)}


def _plot_case(
    *,
    execution_path: str,
    diameter_nm: float,
    q: np.ndarray,
    full_norm: np.ndarray,
    full_norm_vs_flat: np.ndarray,
    collapsed_norm: np.ndarray,
    flat_norm: np.ndarray,
    flat_norm_from_full: np.ndarray,
    direct_norm: np.ndarray,
    summary: dict[str, object],
    out_path: Path,
) -> None:
    path_label = execution_path.replace("_", " ")
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.0, 8.4), sharex=True)

    plot_mask = np.logical_and.reduce(
        [
            q >= Q_PLOT_MIN,
            q <= Q_PLOT_MAX,
            np.isfinite(full_norm),
            np.isfinite(collapsed_norm),
            np.isfinite(flat_norm),
            np.isfinite(direct_norm),
            full_norm > 0.0,
            collapsed_norm > 0.0,
            flat_norm > 0.0,
            direct_norm > 0.0,
        ]
    )
    ax0.plot(
        q[plot_mask],
        full_norm[plot_mask],
        color="#1f77b4",
        linewidth=2.0,
        label=f"Full 3D sphere ({path_label})",
    )
    ax0.plot(
        q[plot_mask],
        collapsed_norm[plot_mask],
        color="#d62728",
        linewidth=1.7,
        linestyle="--",
        label=f'Collapsed 3D sphere ({path_label}, `z_collapse_mode="mean"`)',
    )
    ax0.plot(
        q[plot_mask],
        flat_norm[plot_mask],
        color="black",
        linewidth=1.5,
        alpha=0.9,
        label="Analytical flat-detector sphere",
    )
    ax0.plot(
        q[plot_mask],
        direct_norm[plot_mask],
        color="#7f7f7f",
        linewidth=1.2,
        linestyle=":",
        alpha=0.9,
        label="Analytical direct I(q)",
    )
    ax0.set_yscale("log")
    ax0.set_xlim(Q_PLOT_MIN, Q_PLOT_MAX)
    ax0.set_ylabel("Normalized I(q)")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best", fontsize=8.5)

    resid_mask = np.logical_and.reduce(
        [
            q >= Q_POINTWISE_MIN,
            q <= Q_POINTWISE_MAX,
            np.isfinite(full_norm),
            np.isfinite(collapsed_norm),
            np.isfinite(flat_norm),
            full_norm > MIN_SIGNAL_FOR_LOG,
            collapsed_norm > MIN_SIGNAL_FOR_LOG,
            flat_norm > MIN_SIGNAL_FOR_LOG,
        ]
    )
    ax1.plot(
        q[resid_mask],
        np.log10(collapsed_norm[resid_mask]) - np.log10(full_norm[resid_mask]),
        color="#d62728",
        linewidth=1.2,
        label="Collapsed - Full",
    )
    ax1.plot(
        q[resid_mask],
        np.log10(full_norm_vs_flat[resid_mask]) - np.log10(flat_norm_from_full[resid_mask]),
        color="#1f77b4",
        linewidth=1.0,
        alpha=0.8,
        label="Full - Flat analytical",
    )
    ax1.plot(
        q[resid_mask],
        np.log10(collapsed_norm[resid_mask]) - np.log10(flat_norm[resid_mask]),
        color="black",
        linewidth=1.0,
        alpha=0.75,
        linestyle="--",
        label="Collapsed - Flat analytical",
    )
    ax1.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_xlim(Q_PLOT_MIN, Q_PLOT_MAX)
    ax1.set_xlabel(r"q [nm$^{-1}$]")
    ax1.set_ylabel(r"$\Delta \log_{10} I$")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best", fontsize=8.5)

    runtime = summary["runtime_seconds"]
    runtime_ratio = summary["runtime_ratio_full_to_collapsed"]
    full_vs_collapsed = summary["collapsed_vs_full"]
    note_lines = [
        f"diameter={diameter_nm:.1f} nm, energy={ENERGY_EV:.1f} eV, oversample={summary['superresolution']}",
        (
            f"full sim={runtime['full_3d']['sim_seconds']:.2f}s, "
            f"collapsed sim={runtime['collapsed_3d']['sim_seconds']:.2f}s, "
            f"ratio={(runtime_ratio if runtime_ratio is not None else float('nan')):.2f}x"
        ),
        (
            f"collapsed vs full: rms_log={full_vs_collapsed['pointwise']['rms_log']:.4f}, "
            f"p95_log_abs={full_vs_collapsed['pointwise']['p95_log_abs']:.4f}, "
            f"minima_mae={full_vs_collapsed['minima']['mae_abs_dq']:.5f}"
        ),
        (
            f"collapsed vs flat analytical: rms_log={summary['collapsed_vs_flat']['pointwise']['rms_log']:.4f}, "
            f"minima_mae={summary['collapsed_vs_flat']['minima']['mae_abs_dq']:.5f}"
        ),
        (
            f"collapsed vs direct analytical: rms_log={summary['collapsed_vs_direct']['pointwise']['rms_log']:.4f}, "
            f"minima_mae={summary['collapsed_vs_direct']['minima']['mae_abs_dq']:.5f}"
        ),
    ]
    fig.suptitle(f"Sphere Form Factor: Full 3D vs z-Collapsed Fast Path ({path_label})")
    fig.text(
        0.01,
        0.01,
        "\n".join(note_lines),
        fontsize=8.0,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )
    fig.tight_layout(rect=[0, 0.17, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _serializable_metrics(metrics: dict[str, object]) -> dict[str, object]:
    result = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            result[key] = [float(v) for v in value.tolist()]
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def _summarize_case(
    execution_path: str,
    diameter_nm: float,
    superresolution: int,
    output_dir: Path,
) -> dict[str, object]:
    full_q, full_iq, full_runtime = _run_iq(
        diameter_nm=diameter_nm,
        superresolution=superresolution,
        runtime_kwargs=_baseline_runtime_kwargs(execution_path),
    )
    collapsed_q, collapsed_iq, collapsed_runtime = _run_iq(
        diameter_nm=diameter_nm,
        superresolution=superresolution,
        runtime_kwargs=_collapsed_runtime_kwargs(execution_path),
    )
    if not np.allclose(full_q, collapsed_q, atol=1e-12, rtol=0.0):
        raise AssertionError("Full and collapsed sphere runs produced mismatched q grids.")

    flat_image = _flat_detector_analytic_image(
        diameter_nm=diameter_nm,
        energy_eV=ENERGY_EV,
        oversample=FLAT_DETECTOR_OVERSAMPLE,
    )
    flat_q, flat_iq = _pyhyper_iq_by_energy(flat_image)[float(ENERGY_EV)]
    if not np.allclose(full_q, flat_q, atol=1e-12, rtol=0.0):
        raise AssertionError("Analytical flat-detector q grid does not match simulation q grid.")

    direct_iq = _analytic_sphere_form_factor_binned_iq(q_centers=full_q, diameter_nm=diameter_nm)

    collapsed_norm, full_norm, _ = _normalize_to_first_q_gt_zero(full_q, collapsed_iq, full_iq)
    _, flat_norm, _ = _normalize_to_first_q_gt_zero(full_q, collapsed_iq, flat_iq)
    _, direct_norm, _ = _normalize_to_first_q_gt_zero(full_q, collapsed_iq, direct_iq)
    full_norm_vs_flat, flat_norm_from_full, _ = _normalize_to_first_q_gt_zero(full_q, full_iq, flat_iq)

    summary = {
        "execution_path": execution_path,
        "diameter_nm": float(diameter_nm),
        "energy_eV": float(ENERGY_EV),
        "superresolution": int(superresolution),
        "runtime_seconds": {
            "full_3d": full_runtime,
            "collapsed_3d": collapsed_runtime,
        },
        "runtime_ratio_full_to_collapsed": (
            float(full_runtime["sim_seconds"] / collapsed_runtime["sim_seconds"])
            if collapsed_runtime["sim_seconds"] > 0.0
            else None
        ),
        "collapsed_vs_full": {
            "pointwise": _serializable_metrics(_pointwise_metrics(full_q, collapsed_norm, full_norm)),
            "minima": _serializable_metrics(_minima_alignment_metrics(full_q, collapsed_norm, full_norm)),
        },
        "collapsed_vs_flat": {
            "pointwise": _serializable_metrics(_pointwise_metrics(full_q, collapsed_norm, flat_norm)),
            "minima": _serializable_metrics(_minima_alignment_metrics(full_q, collapsed_norm, flat_norm)),
        },
        "collapsed_vs_direct": {
            "pointwise": _serializable_metrics(_pointwise_metrics(full_q, collapsed_norm, direct_norm)),
            "minima": _serializable_metrics(_minima_alignment_metrics(full_q, collapsed_norm, direct_norm)),
        },
        "full_vs_flat": {
            "pointwise": _serializable_metrics(
                _pointwise_metrics(full_q, full_norm_vs_flat, flat_norm_from_full)
            ),
            "minima": _serializable_metrics(
                _minima_alignment_metrics(full_q, full_norm_vs_flat, flat_norm_from_full)
            ),
        },
    }

    case_stem = f"sphere_d{int(round(diameter_nm))}_sr{int(superresolution)}_{execution_path}"
    png_path = output_dir / f"{case_stem}_comparison.png"
    json_path = output_dir / f"{case_stem}_summary.json"
    _plot_case(
        execution_path=execution_path,
        diameter_nm=diameter_nm,
        q=full_q,
        full_norm=full_norm,
        full_norm_vs_flat=full_norm_vs_flat,
        collapsed_norm=collapsed_norm,
        flat_norm=flat_norm,
        flat_norm_from_full=flat_norm_from_full,
        direct_norm=direct_norm,
        summary=summary,
        out_path=png_path,
    )
    runtime_ratio = summary["runtime_ratio_full_to_collapsed"]
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"path={execution_path} "
        f"diameter={diameter_nm:.1f} nm "
        f"rms_log(collapsed/full)={summary['collapsed_vs_full']['pointwise']['rms_log']:.4f} "
        f"runtime_ratio={(runtime_ratio if runtime_ratio is not None else float('nan')):.2f}x "
        f"plot={png_path}"
    )
    return {
        "summary_path": str(json_path),
        "plot_path": str(png_path),
        "metrics": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the exploratory cupy-rsoxs z-collapse fast path against the full 3D sphere form factor."
    )
    parser.add_argument(
        "--diameters-nm",
        nargs="+",
        type=float,
        default=[float(v) for v in DIAMETERS_NM],
        help="Sphere diameters in nm.",
    )
    parser.add_argument(
        "--superresolution",
        type=int,
        default=1,
        help="Sphere voxel supersampling factor used for both full and collapsed runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG/JSON artifacts.",
    )
    parser.add_argument(
        "--execution-paths",
        nargs="+",
        choices=tuple(EXECUTION_PATH_TO_PATH_ID),
        default=list(DEFAULT_EXECUTION_PATHS),
        help="Execution paths to compare in both full and z-collapsed modes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not _has_visible_gpu():
        raise SystemExit("No visible NVIDIA GPU found for the z-collapse sphere comparison.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for execution_path in args.execution_paths:
        path_results = {}
        for diameter_nm in args.diameters_nm:
            path_results[str(float(diameter_nm))] = _summarize_case(
                execution_path=execution_path,
                diameter_nm=float(diameter_nm),
                superresolution=int(args.superresolution),
                output_dir=output_dir,
            )
        results[execution_path] = path_results

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
