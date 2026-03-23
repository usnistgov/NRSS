from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.validation.lib.core_shell import (
    BASELINE_SCENARIO,
    CoreShellScenario,
    EXPERIMENTAL_REFERENCE_CITATION,
    EXPERIMENTAL_REFERENCE_LABEL,
    SIM_REFERENCE_LABEL,
    SIM_REFERENCE_PATH,
    awedge_comparison_slices,
    compute_awedge_metrics,
    load_experimental_reference_awedge,
    load_sim_reference_awedge,
    plot_core_shell_validation_panel,
    run_core_shell_pybind,
    scattering_to_awedge,
    write_awedge_reference,
)


PLOT_DIR = REPO_ROOT / "test-reports" / "core-shell-dev"

FALSIFICATION_SCENARIOS = (
    CoreShellScenario(
        scenario_id="subterfuge_half_s",
        display_name="Subterfuge radial with half fitted shell S",
        shell_s_mode="scaled_legacy_decay",
        shell_s_scale=0.5,
        shell_orientation_mode="radial",
        description=(
            "Legacy radial shell Euler field with the fitted decaying shell S profile "
            "scaled to one half of the original amplitude."
        ),
    ),
    CoreShellScenario(
        scenario_id="subterfuge_tangential",
        display_name="Subterfuge tangential with half fitted shell S",
        shell_s_mode="scaled_legacy_decay",
        shell_s_scale=0.5,
        shell_orientation_mode="tangential_latitude",
        description=(
            "Latitude-style tangential shell Euler field using the same half-amplitude "
            "fitted shell S profile as the alternate-S variant."
        ),
    ),
)


def _write_metrics_tsv(rows: list[tuple[str, str, str, float, float]], out_dir: Path) -> None:
    metrics_path = out_dir / "metrics.tsv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["scenario", "reference_kind", "series", "max_abs_diff", "rmse"])
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-sim-reference",
        action="store_true",
        help="Refresh the sim-derived baseline golden alongside the experimental comparison plots.",
    )
    args = parser.parse_args(argv)

    out_dir = PLOT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    experimental_reference = load_experimental_reference_awedge()
    metrics_rows: list[tuple[str, str, str, float, float]] = []
    baseline_awedge = None

    for scenario in (BASELINE_SCENARIO, *FALSIFICATION_SCENARIOS):
        scenario_id = scenario.scenario_id
        scattering = run_core_shell_pybind(scenario=scenario)
        awedge = scattering_to_awedge(scattering)
        if scenario_id == "baseline":
            baseline_awedge = awedge.copy(deep=True)

        comparison = awedge_comparison_slices(awedge=awedge, reference=experimental_reference)
        metrics = compute_awedge_metrics(comparison)
        for series, vals in metrics.items():
            metrics_rows.append(
                (
                    scenario_id,
                    "experimental",
                    series,
                    vals["max_abs_diff"],
                    vals["rmse"],
                )
            )
        plot_core_shell_validation_panel(
            comparison=comparison,
            metrics=metrics,
            out_path=out_dir / f"core_shell_{scenario_id}_vs_experiment.png",
            scenario=scenario,
            reference_label=EXPERIMENTAL_REFERENCE_LABEL,
            reference_citation=EXPERIMENTAL_REFERENCE_CITATION,
        )

    if args.write_sim_reference:
        if baseline_awedge is None:
            raise AssertionError("Baseline A-wedge was not generated.")
        write_awedge_reference(
            awedge=baseline_awedge,
            path=SIM_REFERENCE_PATH,
            source_kind="simulated_regression",
            description=(
                "CoreShell pybind + WPIntegrator + manual A-wedge regression golden "
                "serialized from the maintained baseline scenario."
            ),
        )

    if SIM_REFERENCE_PATH.exists() and baseline_awedge is not None:
        sim_reference = load_sim_reference_awedge()
        sim_comparison = awedge_comparison_slices(awedge=baseline_awedge, reference=sim_reference)
        sim_metrics = compute_awedge_metrics(sim_comparison)
        for series, vals in sim_metrics.items():
            metrics_rows.append(
                (
                    "baseline",
                    "sim_regression",
                    series,
                    vals["max_abs_diff"],
                    vals["rmse"],
                )
            )
        plot_core_shell_validation_panel(
            comparison=sim_comparison,
            metrics=sim_metrics,
            out_path=out_dir / "core_shell_baseline_vs_sim_reference.png",
            scenario=BASELINE_SCENARIO,
            reference_label=SIM_REFERENCE_LABEL,
            reference_citation=(
                "Sim-derived regression golden generated from the maintained baseline "
                "CoreShell scenario. This artifact guards implementation stability, not "
                "experimental truth."
            ),
        )

    _write_metrics_tsv(metrics_rows, out_dir=out_dir)
    print(f"Wrote CoreShell diagnostic artifacts to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
