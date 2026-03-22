from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Validation/dev studies should default to a single visible GPU unless the user
# or CI has already pinned a specific device set.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from tests.validation.dev.mwcnt_rsa.benchmark_mwcnt_rsa import (
    build_dynamic_geometry_rows,
    write_geometry_csv,
)
from tests.validation.lib.core_shell import has_visible_gpu
from tests.validation.lib.mwcnt import (
    EXPERIMENTAL_REFERENCE_CITATION,
    Q_COMPARE_MAX_NM,
    Q_COMPARE_MIN_NM,
    align_observables_to_reference,
    compute_observable_metrics,
    load_experimental_reference_observables,
    plot_mwcnt_validation_panel,
    reduce_scattering_to_observables,
    run_mwcnt_pybind,
    write_reference_observables,
)


OUT_DIR = REPO_ROOT / "test-reports" / "mwcnt-eangle-dev"
OBS_DIR = OUT_DIR / "observables"
SUMMARY_PATH = OUT_DIR / "mwcnt_eangle_step_sweep_summary.json"
GEOMETRY_PATH = (
    REPO_ROOT
    / "test-reports"
    / "mwcnt-multiseed-dev"
    / "geometries"
    / "mwcnt_seed12345_dynamic.csv"
)

SEED = 12345
NUM_TRIALS = 20_000
RSA_VARIANT = "kdtree_dual_dynamic"
STEP_ROTATIONS = {
    "step10": [0.0, 10.0, 350.0],
    "step15": [0.0, 15.0, 345.0],
    "step20": [0.0, 20.0, 340.0],
}


def ensure_geometry_csv() -> tuple[Path, int, float]:
    if GEOMETRY_PATH.exists():
        data = np.genfromtxt(GEOMETRY_PATH, delimiter=",", names=True, dtype=np.float64)
        return GEOMETRY_PATH, int(data.shape[0]), 0.0

    GEOMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rows = build_dynamic_geometry_rows(seed=SEED, num_trials=NUM_TRIALS)
    write_geometry_csv(rows, GEOMETRY_PATH)
    elapsed = time.perf_counter() - start
    return GEOMETRY_PATH, len(rows), elapsed


def _details_lines(step_name: str, eangle_rotation: list[float], accepted_count: int) -> list[str]:
    return [
        f"MWCNT single-seed EAngleRotation sweep: {step_name}",
        "",
        f"RSA geometry builder: {RSA_VARIANT}",
        f"Seed: {SEED}",
        f"Accepted CNTs: {accepted_count}",
        f"Candidate count: {NUM_TRIALS + 1}",
        f"EAngleRotation: {eangle_rotation}",
        "Geometry semantics: exact tutorial-compatible reject-on-overlap",
        "Runtime env: nrss-dev",
    ]


def _metrics_to_builtin(metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {metric: float(value) for metric, value in metric_dict.items()}
        for key, metric_dict in metrics.items()
    }


def main() -> None:
    if not has_visible_gpu():
        raise SystemExit("No visible NVIDIA GPU found for the MWCNT EAngleRotation sweep.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OBS_DIR.mkdir(parents=True, exist_ok=True)

    geometry_path, accepted_count, geometry_build_sec = ensure_geometry_csv()
    reference = load_experimental_reference_observables()
    results: dict[str, dict[str, object]] = {}

    for step_name, eangle_rotation in STEP_ROTATIONS.items():
        print(f"Running {step_name} with EAngleRotation={eangle_rotation}...", flush=True)
        sim_start = time.perf_counter()
        scattering = run_mwcnt_pybind(geometry_path=geometry_path, eangle_rotation=eangle_rotation)
        sim_elapsed = time.perf_counter() - sim_start

        observables = reduce_scattering_to_observables(scattering)
        comparison = align_observables_to_reference(observables, reference)
        metrics = compute_observable_metrics(comparison)

        plot_path = OUT_DIR / f"mwcnt_{step_name}.png"
        plot_mwcnt_validation_panel(
            comparison=comparison,
            metrics=metrics,
            out_path=plot_path,
            q_min_nm=Q_COMPARE_MIN_NM,
            q_max_nm=Q_COMPARE_MAX_NM,
            title=f"MWCNT experimental validation: {step_name}",
            simulation_label=step_name,
            simulation_details=_details_lines(step_name, eangle_rotation, accepted_count),
            eangle_rotation=eangle_rotation,
            description=(
                f"Single-seed MWCNT EAngleRotation sweep entry {step_name} using {RSA_VARIANT} "
                f"geometry generation and seed {SEED}. Reference: {EXPERIMENTAL_REFERENCE_CITATION}"
            ),
            geometry_source_name=geometry_path.name,
            geometry_path=geometry_path,
        )
        write_reference_observables(
            observables=observables,
            path=OBS_DIR / f"mwcnt_{step_name}_observables.h5",
            source_kind="simulated_single_seed",
            description=(
                f"Single-seed EAngleRotation sweep entry {step_name}; seed={SEED}; "
                f"EAngleRotation={eangle_rotation}; rsa_variant={RSA_VARIANT}"
            ),
        )

        results[step_name] = {
            "eangle_rotation": eangle_rotation,
            "plot_path": str(plot_path),
            "metrics": _metrics_to_builtin(metrics),
            "simulation_sec": sim_elapsed,
        }
        print(
            f"Finished {step_name}: "
            f"A(E) r={metrics['a_vs_energy_qband']['correlation']:.3f}, "
            f"A(q)285 r={metrics['a_vs_q_285']['correlation']:.3f}, "
            f"A(q)292 r={metrics['a_vs_q_292']['correlation']:.3f}",
            flush=True,
        )

    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "seed": SEED,
                "num_trials": NUM_TRIALS,
                "rsa_variant": RSA_VARIANT,
                "geometry_path": str(geometry_path),
                "geometry_build_sec": geometry_build_sec,
                "accepted_count": accepted_count,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
