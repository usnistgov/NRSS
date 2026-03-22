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


OUT_DIR = REPO_ROOT / "test-reports" / "mwcnt-windowing-dev"
OBS_DIR = OUT_DIR / "observables"
SUMMARY_PATH = OUT_DIR / "mwcnt_windowing_compare_summary.json"
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
EANGLE_ROTATION = [0.0, 20.0, 340.0]
SCENARIOS = {
    "legacy_w1": {
        "field_boundary_mode": "legacy",
        "windowing_type": 1,
        "title_suffix": "legacy fields, WindowingType=1",
    },
    "periodic_w1": {
        "field_boundary_mode": "periodic",
        "windowing_type": 1,
        "title_suffix": "periodic fields, WindowingType=1",
    },
    "periodic_w0": {
        "field_boundary_mode": "periodic",
        "windowing_type": 0,
        "title_suffix": "periodic fields, WindowingType=0",
    },
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


def _details_lines(
    scenario_name: str,
    field_boundary_mode: str,
    windowing_type: int,
    accepted_count: int,
) -> list[str]:
    return [
        f"MWCNT windowing comparison: {scenario_name}",
        "",
        f"RSA geometry builder: {RSA_VARIANT}",
        f"Seed: {SEED}",
        f"Accepted CNTs: {accepted_count}",
        f"Candidate count: {NUM_TRIALS + 1}",
        f"Field boundary mode: {field_boundary_mode}",
        f"WindowingType: {windowing_type}",
        f"EAngleRotation: {EANGLE_ROTATION}",
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
        raise SystemExit("No visible NVIDIA GPU found for the MWCNT windowing comparison.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OBS_DIR.mkdir(parents=True, exist_ok=True)

    geometry_path, accepted_count, geometry_build_sec = ensure_geometry_csv()
    reference = load_experimental_reference_observables()
    results: dict[str, dict[str, object]] = {}

    for scenario_name, scenario in SCENARIOS.items():
        field_boundary_mode = str(scenario["field_boundary_mode"])
        windowing_type = int(scenario["windowing_type"])
        title_suffix = str(scenario["title_suffix"])
        print(
            f"Running {scenario_name}: "
            f"field_boundary_mode={field_boundary_mode}, windowing_type={windowing_type}...",
            flush=True,
        )
        sim_start = time.perf_counter()
        scattering = run_mwcnt_pybind(
            geometry_path=geometry_path,
            eangle_rotation=EANGLE_ROTATION,
            windowing_type=windowing_type,
            field_boundary_mode=field_boundary_mode,
        )
        sim_elapsed = time.perf_counter() - sim_start

        observables = reduce_scattering_to_observables(scattering)
        comparison = align_observables_to_reference(observables, reference)
        metrics = compute_observable_metrics(comparison)

        plot_path = OUT_DIR / f"mwcnt_{scenario_name}.png"
        plot_mwcnt_validation_panel(
            comparison=comparison,
            metrics=metrics,
            out_path=plot_path,
            q_min_nm=Q_COMPARE_MIN_NM,
            q_max_nm=Q_COMPARE_MAX_NM,
            title=f"MWCNT experimental validation: {title_suffix}",
            simulation_label=scenario_name,
            simulation_details=_details_lines(
                scenario_name=scenario_name,
                field_boundary_mode=field_boundary_mode,
                windowing_type=windowing_type,
                accepted_count=accepted_count,
            ),
            eangle_rotation=EANGLE_ROTATION,
            description=(
                f"Fixed-seed MWCNT comparison for field_boundary_mode={field_boundary_mode} "
                f"and WindowingType={windowing_type}, using {RSA_VARIANT} geometry generation, "
                f"seed={SEED}, and EAngleRotation={EANGLE_ROTATION}. "
                f"Reference: {EXPERIMENTAL_REFERENCE_CITATION}"
            ),
            geometry_source_name=geometry_path.name,
            geometry_path=geometry_path,
            windowing_type=windowing_type,
            field_boundary_mode=field_boundary_mode,
        )
        write_reference_observables(
            observables=observables,
            path=OBS_DIR / f"mwcnt_{scenario_name}_observables.h5",
            source_kind="simulated_single_seed",
            description=(
                f"Fixed-seed MWCNT comparison entry {scenario_name}; seed={SEED}; "
                f"EAngleRotation={EANGLE_ROTATION}; rsa_variant={RSA_VARIANT}; "
                f"field_boundary_mode={field_boundary_mode}; WindowingType={windowing_type}"
            ),
        )

        results[scenario_name] = {
            "field_boundary_mode": field_boundary_mode,
            "windowing_type": windowing_type,
            "plot_path": str(plot_path),
            "metrics": _metrics_to_builtin(metrics),
            "simulation_sec": sim_elapsed,
        }
        print(
            f"Finished {scenario_name}: "
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
                "eangle_rotation": EANGLE_ROTATION,
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
