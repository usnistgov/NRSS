from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr


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


OUT_DIR = REPO_ROOT / "test-reports" / "mwcnt-multiseed-dev"
GEOMETRY_DIR = OUT_DIR / "geometries"
OBS_DIR = OUT_DIR / "observables"
SUMMARY_PATH = OUT_DIR / "mwcnt_multiseed_summary.json"

SEEDS = [12345, 12346, 12347, 12348, 12349, 12350]
AVERAGE_COUNTS = (1, 3, 6)
NUM_TRIALS = 20_000
EANGLE_ROTATION = [0.0, 5.0, 355.0]
RSA_VARIANT = "kdtree_dual_dynamic"


def _geometry_path_for_seed(seed: int) -> Path:
    return GEOMETRY_DIR / f"mwcnt_seed{seed}_dynamic.csv"


def ensure_geometry_csv(seed: int) -> tuple[Path, int, float]:
    out_path = _geometry_path_for_seed(seed)
    if out_path.exists():
        data = np.genfromtxt(out_path, delimiter=",", names=True, dtype=np.float64)
        return out_path, int(data.shape[0]), 0.0

    start = time.perf_counter()
    rows = build_dynamic_geometry_rows(seed=seed, num_trials=NUM_TRIALS)
    write_geometry_csv(rows, out_path)
    elapsed = time.perf_counter() - start
    return out_path, len(rows), elapsed


def _details_lines(average_count: int, used_seeds: list[int]) -> list[str]:
    return [
        f"MWCNT {average_count}-seed cumulative average",
        "",
        f"RSA geometry builder: {RSA_VARIANT}",
        f"Seeds: {used_seeds}",
        f"Candidate count per seed: {NUM_TRIALS + 1}",
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
        raise SystemExit("No visible NVIDIA GPU found for the MWCNT multi-seed study.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
    OBS_DIR.mkdir(parents=True, exist_ok=True)

    seed_records: list[dict[str, object]] = []
    cumulative_sum: xr.DataArray | None = None
    average_records: dict[str, dict[str, object]] = {}

    for index, seed in enumerate(SEEDS, start=1):
        geometry_path, accepted_count, geometry_elapsed = ensure_geometry_csv(seed)

        sim_start = time.perf_counter()
        scattering = run_mwcnt_pybind(geometry_path=geometry_path, eangle_rotation=EANGLE_ROTATION)
        sim_elapsed = time.perf_counter() - sim_start

        if cumulative_sum is None:
            cumulative_sum = scattering.copy(deep=True)
        else:
            cumulative_sum.values += np.asarray(scattering.values)

        seed_records.append(
            {
                "seed": seed,
                "geometry_csv": str(geometry_path),
                "accepted_count": accepted_count,
                "geometry_build_sec": geometry_elapsed,
                "simulation_sec": sim_elapsed,
            }
        )

        if index not in AVERAGE_COUNTS:
            continue

        average_scattering = cumulative_sum / float(index)
        observables = reduce_scattering_to_observables(average_scattering)
        comparison = align_observables_to_reference(observables, load_experimental_reference_observables())
        metrics = compute_observable_metrics(comparison)

        used_seeds = SEEDS[:index]
        plot_path = OUT_DIR / f"mwcnt_average_{index:02d}_seeds.png"
        plot_mwcnt_validation_panel(
            comparison=comparison,
            metrics=metrics,
            out_path=plot_path,
            q_min_nm=Q_COMPARE_MIN_NM,
            q_max_nm=Q_COMPARE_MAX_NM,
            title=f"MWCNT experimental validation: {index}-seed average",
            simulation_label=f"{index}-seed average",
            simulation_details=_details_lines(index, used_seeds),
            eangle_rotation=EANGLE_ROTATION,
            description=(
                f"Cumulative {index}-seed MWCNT average using {RSA_VARIANT} geometry generation "
                f"and EAngleRotation={EANGLE_ROTATION}. Reference: {EXPERIMENTAL_REFERENCE_CITATION}"
            ),
            geometry_source_name=f"dynamic CSVs for seeds {used_seeds}",
            geometry_path=geometry_path,
        )
        write_reference_observables(
            observables=observables,
            path=OBS_DIR / f"mwcnt_average_{index:02d}_seeds_observables.h5",
            source_kind="simulated_average",
            description=(
                f"Cumulative {index}-seed average generated with {RSA_VARIANT}, "
                f"seeds={used_seeds}, EAngleRotation={EANGLE_ROTATION}"
            ),
        )
        average_records[str(index)] = {
            "seeds": used_seeds,
            "plot_path": str(plot_path),
            "metrics": _metrics_to_builtin(metrics),
        }

    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "rsa_variant": RSA_VARIANT,
                "eangle_rotation": EANGLE_ROTATION,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "num_trials": NUM_TRIALS,
                "seeds": SEEDS,
                "seed_records": seed_records,
                "averages": average_records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {SUMMARY_PATH}")
    for key in ("1", "3", "6"):
        record = average_records[key]
        metrics = record["metrics"]
        print(
            f"{key}-seed avg: "
            f"A(E) r={metrics['a_vs_energy_qband']['correlation']:.3f}, "
            f"A(q)285 r={metrics['a_vs_q_285']['correlation']:.3f}, "
            f"A(q)292 r={metrics['a_vs_q_292']['correlation']:.3f}"
        )
if __name__ == "__main__":
    main()
