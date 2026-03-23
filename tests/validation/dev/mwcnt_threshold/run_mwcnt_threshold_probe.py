from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Validation/dev studies should default to a single visible GPU unless the user
# or CI has already pinned a specific device set.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from tests.validation.dev.mwcnt_rsa.benchmark_mwcnt_rsa import (
    LENGTH_LOWER,
    LENGTH_UPPER,
    NUM_TRIALS,
    RADIUS_MU,
    RADIUS_SIGMA,
    SEED,
    THETA_MU,
    THETA_SIGMA,
    build_dynamic_geometry_rows,
    write_geometry_csv,
)
from tests.validation.lib.core_shell import has_visible_gpu
from tests.validation.lib.mwcnt import (
    CNT_GEOMETRY_PATH,
    EANGLE_ROTATION,
    EXPERIMENTAL_REFERENCE_CITATION,
    EXPERIMENTAL_THRESHOLDS,
    FIELD_BOUNDARY_MODE_DEFAULT,
    Q_COMPARE_MAX_NM,
    Q_COMPARE_MIN_NM,
    WINDOWING_TYPE_DEFAULT,
    align_observables_to_reference,
    compute_observable_metrics,
    geometry_realization_stats,
    load_experimental_reference_observables,
    metrics_within_thresholds,
    plot_mwcnt_validation_panel,
    reduce_scattering_to_observables,
    run_mwcnt_pybind,
)


OUT_DIR = REPO_ROOT / "test-reports" / "mwcnt-threshold-dev"
GEOMETRY_DIR = OUT_DIR / "geometries"
SUMMARY_PATH = OUT_DIR / "mwcnt_threshold_probe_summary.json"


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    display_name: str
    description: str
    radius_mu: float
    radius_sigma: float
    theta_mu: float
    theta_sigma: float
    length_lower: float
    length_upper: float
    use_vendored_geometry: bool = False


SCENARIOS = (
    Scenario(
        scenario_id="baseline_vendored",
        display_name="Baseline vendored geometry",
        description="Maintained fixed-seed vendored geometry used by the official MWCNT validation test.",
        radius_mu=RADIUS_MU,
        radius_sigma=RADIUS_SIGMA,
        theta_mu=THETA_MU,
        theta_sigma=THETA_SIGMA,
        length_lower=LENGTH_LOWER,
        length_upper=LENGTH_UPPER,
        use_vendored_geometry=True,
    ),
    Scenario(
        scenario_id="theta_sigma_5over8pi",
        display_name="Broader orientation distribution",
        description="Moderately broader in-plane tilt distribution using theta_sigma = 5/(8*pi) rad.",
        radius_mu=RADIUS_MU,
        radius_sigma=RADIUS_SIGMA,
        theta_mu=THETA_MU,
        theta_sigma=5.0 / (8.0 * math.pi),
        length_lower=LENGTH_LOWER,
        length_upper=LENGTH_UPPER,
    ),
    Scenario(
        scenario_id="radius_sigma_0p275",
        display_name="Broader radius distribution",
        description="Moderately broader lognormal radius distribution using radius_sigma = 0.275.",
        radius_mu=RADIUS_MU,
        radius_sigma=0.275,
        theta_mu=THETA_MU,
        theta_sigma=THETA_SIGMA,
        length_lower=LENGTH_LOWER,
        length_upper=LENGTH_UPPER,
    ),
    Scenario(
        scenario_id="radius_mu_2p275",
        display_name="Larger mean radius",
        description="Moderately larger lognormal radius center using radius_mu = 2.275.",
        radius_mu=2.275,
        radius_sigma=RADIUS_SIGMA,
        theta_mu=THETA_MU,
        theta_sigma=THETA_SIGMA,
        length_lower=LENGTH_LOWER,
        length_upper=LENGTH_UPPER,
    ),
    Scenario(
        scenario_id="theta_sigma_5over8pi_radius_sigma_0p275",
        display_name="Broader orientation and radius distributions",
        description=(
            "Combined nearby falsification using theta_sigma = 5/(8*pi) rad and "
            "radius_sigma = 0.275."
        ),
        radius_mu=RADIUS_MU,
        radius_sigma=0.275,
        theta_mu=THETA_MU,
        theta_sigma=5.0 / (8.0 * math.pi),
        length_lower=LENGTH_LOWER,
        length_upper=LENGTH_UPPER,
    ),
)


def _scenario_geometry_path(scenario: Scenario) -> Path:
    return GEOMETRY_DIR / f"{scenario.scenario_id}.csv"


def ensure_geometry_csv(scenario: Scenario) -> tuple[Path, int, float]:
    if scenario.use_vendored_geometry:
        stats = geometry_realization_stats(CNT_GEOMETRY_PATH)
        return CNT_GEOMETRY_PATH, int(stats["accepted_count"]), 0.0

    out_path = _scenario_geometry_path(scenario)
    if out_path.exists():
        stats = geometry_realization_stats(out_path)
        return out_path, int(stats["accepted_count"]), 0.0

    start = time.perf_counter()
    rows = build_dynamic_geometry_rows(
        seed=SEED,
        num_trials=NUM_TRIALS,
        radius_mu=scenario.radius_mu,
        radius_sigma=scenario.radius_sigma,
        theta_mu=scenario.theta_mu,
        theta_sigma=scenario.theta_sigma,
        length_lower=scenario.length_lower,
        length_upper=scenario.length_upper,
    )
    write_geometry_csv(rows, out_path)
    elapsed = time.perf_counter() - start
    return out_path, len(rows), elapsed


def _details_lines(scenario: Scenario, accepted_count: int) -> list[str]:
    return [
        f"MWCNT threshold probe: {scenario.display_name}",
        "",
        f"Seed: {SEED}",
        f"Candidate count: {NUM_TRIALS + 1}",
        f"Accepted CNTs: {accepted_count}",
        f"radius_mu={scenario.radius_mu:.3f}",
        f"radius_sigma={scenario.radius_sigma:.3f}",
        f"theta_mu={scenario.theta_mu:.4f} rad",
        f"theta_sigma={scenario.theta_sigma:.4f} rad",
        f"length range=[{scenario.length_lower:.0f}, {scenario.length_upper:.0f}]",
        f"EAngleRotation={EANGLE_ROTATION}",
        f"field_boundary_mode={FIELD_BOUNDARY_MODE_DEFAULT}",
        f"WindowingType={WINDOWING_TYPE_DEFAULT}",
        "Geometry semantics: tutorial-compatible reject-on-overlap, fixed seed",
        "Runtime env: nrss-dev",
    ]


def _metrics_to_builtin(metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {metric: float(value) for metric, value in metric_dict.items()}
        for key, metric_dict in metrics.items()
    }


def main() -> None:
    if not has_visible_gpu():
        raise SystemExit("No visible NVIDIA GPU found for the MWCNT threshold probe.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)

    reference = load_experimental_reference_observables()
    summary: dict[str, dict[str, object]] = {}

    for scenario in SCENARIOS:
        geometry_path, accepted_count, geometry_build_sec = ensure_geometry_csv(scenario)
        print(f"Running {scenario.scenario_id}...", flush=True)
        sim_start = time.perf_counter()
        scattering = run_mwcnt_pybind(
            geometry_path=geometry_path,
            eangle_rotation=EANGLE_ROTATION,
            windowing_type=WINDOWING_TYPE_DEFAULT,
            field_boundary_mode=FIELD_BOUNDARY_MODE_DEFAULT,
        )
        sim_elapsed = time.perf_counter() - sim_start

        observables = reduce_scattering_to_observables(scattering)
        comparison = align_observables_to_reference(observables, reference)
        metrics = compute_observable_metrics(comparison)
        passed, failures = metrics_within_thresholds(metrics, EXPERIMENTAL_THRESHOLDS)
        stats = geometry_realization_stats(geometry_path)

        plot_mwcnt_validation_panel(
            comparison=comparison,
            metrics=metrics,
            out_path=OUT_DIR / f"{scenario.scenario_id}.png",
            q_min_nm=Q_COMPARE_MIN_NM,
            q_max_nm=Q_COMPARE_MAX_NM,
            title=f"MWCNT threshold probe: {scenario.display_name}",
            simulation_label=scenario.scenario_id,
            simulation_details=_details_lines(scenario, accepted_count),
            eangle_rotation=EANGLE_ROTATION,
            description=(
                f"MWCNT threshold/falsification probe for scenario {scenario.scenario_id}. "
                f"{scenario.description} Reference: {EXPERIMENTAL_REFERENCE_CITATION}"
            ),
            geometry_source_name=geometry_path.name,
            geometry_path=geometry_path,
            windowing_type=WINDOWING_TYPE_DEFAULT,
            field_boundary_mode=FIELD_BOUNDARY_MODE_DEFAULT,
        )

        summary[scenario.scenario_id] = {
            "display_name": scenario.display_name,
            "description": scenario.description,
            "geometry_path": str(geometry_path),
            "geometry_build_sec": geometry_build_sec,
            "simulation_sec": sim_elapsed,
            "accepted_count": accepted_count,
            "passed_current_thresholds": passed,
            "failures": failures,
            "geometry_stats": {key: float(value) for key, value in stats.items()},
            "params": {
                "radius_mu": scenario.radius_mu,
                "radius_sigma": scenario.radius_sigma,
                "theta_mu": scenario.theta_mu,
                "theta_sigma": scenario.theta_sigma,
                "length_lower": scenario.length_lower,
                "length_upper": scenario.length_upper,
                "use_vendored_geometry": scenario.use_vendored_geometry,
            },
            "metrics": _metrics_to_builtin(metrics),
        }
        print(
            f"Finished {scenario.scenario_id}: "
            f"pass={passed}, "
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
                "eangle_rotation": EANGLE_ROTATION,
                "windowing_type": WINDOWING_TYPE_DEFAULT,
                "field_boundary_mode": FIELD_BOUNDARY_MODE_DEFAULT,
                "results": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
