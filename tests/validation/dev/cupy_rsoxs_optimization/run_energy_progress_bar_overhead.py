#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from tests.validation.dev.cupy_rsoxs_optimization import run_cupy_rsoxs_optimization_matrix as base


OUT_ROOT = REPO_ROOT / "test-reports" / "cupy-rsoxs-optimization-dev"
SUMMARY_NAME = "energy_progress_bar_summary.json"
TARGET_ROTATION = (0.0, 5.0, 165.0)
TARGET_ENERGIES = base.CORE_SHELL_TRIPLE_ENERGIES


def _json_default(value: Any):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _build_target_cases() -> list[base.BenchmarkCase]:
    seed_cases = base._timing_cases(
        resident_modes=("host",),
        isotropic_representations=("enum_contract",),
        cuda_prewarm_mode="before_prepare_inputs",
        execution_paths=("tensor_coeff", "direct_polarization"),
        size_labels=("small",),
        timing_segments=(),
        include_triple_no_rotation=False,
        include_triple_limited=False,
        include_full_small_check=False,
        no_rotation_energy_counts=(),
        rotation_specs=(TARGET_ROTATION,),
        energy_lists=(TARGET_ENERGIES,),
        worker_warmup_runs=1,
    )
    targets = [
        case
        for case in seed_cases
        if case.shape_label == "small"
        and tuple(case.energies_ev) == TARGET_ENERGIES
        and tuple(case.eangle_rotation) == TARGET_ROTATION
    ]
    if len(targets) != 2:
        raise SystemExit(
            "Expected exactly two target cases for the small host-hot triple-energy 0:5:165 lane, "
            f"found {len(targets)}."
        )
    return sorted(
        targets,
        key=lambda case: str((case.backend_options or {}).get("execution_path", "")),
    )


def _progress_variant(case: base.BenchmarkCase, enabled: bool) -> base.BenchmarkCase:
    backend_options = dict(case.backend_options or {})
    backend_options["energy_progress_bar"] = enabled
    return replace(
        case,
        label=f"{case.label}_energy_progress_{'on' if enabled else 'off'}",
        backend_options=backend_options,
        run_stdout=False,
        run_stderr=True,
        force_stderr_tty=True,
        notes=(
            f"{case.notes} Dedicated stderr-sensitive progress-bar overhead lane. "
            f"run_stderr=True, force_stderr_tty=True, energy_progress_bar={enabled}."
        ),
    )


def _successful_primary_seconds(results: list[dict[str, Any]]) -> list[float]:
    return [
        float(result["primary_seconds"])
        for result in results
        if result.get("status") == "ok" and "primary_seconds" in result
    ]


def _comparison_payload(
    *,
    execution_path: str,
    disabled_results: list[dict[str, Any]],
    enabled_results: list[dict[str, Any]],
) -> dict[str, Any]:
    off_seconds = _successful_primary_seconds(disabled_results)
    on_seconds = _successful_primary_seconds(enabled_results)
    comparison: dict[str, Any] = {
        "execution_path": execution_path,
        "trial_count_requested": len(disabled_results),
        "trial_statuses": {
            "progress_off": [result.get("status") for result in disabled_results],
            "progress_on": [result.get("status") for result in enabled_results],
        },
        "successful_trial_counts": {
            "progress_off": len(off_seconds),
            "progress_on": len(on_seconds),
        },
        "trial_primary_seconds": {
            "progress_off": off_seconds,
            "progress_on": on_seconds,
        },
    }
    if not off_seconds or not on_seconds:
        comparison["status"] = "comparison_unavailable"
        return comparison

    off_mean = statistics.fmean(off_seconds)
    on_mean = statistics.fmean(on_seconds)
    delta = on_mean - off_mean
    comparison["status"] = "ok"
    comparison["primary_seconds_mean"] = {
        "progress_off": off_mean,
        "progress_on": on_mean,
    }
    comparison["primary_seconds_min"] = {
        "progress_off": min(off_seconds),
        "progress_on": min(on_seconds),
    }
    comparison["primary_seconds_max"] = {
        "progress_off": max(off_seconds),
        "progress_on": max(on_seconds),
    }
    comparison["delta_seconds_mean"] = delta
    comparison["percent_change_vs_off_mean"] = None if off_mean == 0.0 else 100.0 * delta / off_mean
    reference_result = enabled_results[0] if enabled_results else {}
    comparison["energy_progress_bar_stream_behavior"] = {
        "run_stdout": bool(reference_result.get("run_stdout")),
        "run_stderr": bool(reference_result.get("run_stderr")),
        "force_stderr_tty": bool(reference_result.get("force_stderr_tty")),
    }
    return comparison


def run_overhead_check(args: argparse.Namespace) -> int:
    gpu_bootstrap = base._bootstrap_single_gpu_visibility(sys.argv)
    if not base.has_visible_gpu():
        raise SystemExit("No visible NVIDIA GPU found for the energy progress-bar overhead study.")

    run_label = args.label or f"energy_progress_bar_{_timestamp()}"
    run_dir = OUT_ROOT / run_label
    cases_dir = run_dir / "cases"
    run_dir.mkdir(parents=True, exist_ok=True)

    comparisons: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}

    print("Running energy progress-bar overhead cases...", flush=True)
    for case in _build_target_cases():
        execution_path = str((case.backend_options or {}).get("execution_path"))
        disabled_results: list[dict[str, Any]] = []
        enabled_results: list[dict[str, Any]] = []
        for trial_index in range(1, args.trials + 1):
            disabled = replace(
                _progress_variant(case, enabled=False),
                label=f"{case.label}_energy_progress_off_trial{trial_index}",
            )
            enabled = replace(
                _progress_variant(case, enabled=True),
                label=f"{case.label}_energy_progress_on_trial{trial_index}",
            )
            disabled_result = base._run_case_subprocess(case=disabled, output_dir=cases_dir)
            enabled_result = base._run_case_subprocess(case=enabled, output_dir=cases_dir)
            results[disabled.label] = disabled_result
            results[enabled.label] = enabled_result
            disabled_results.append(disabled_result)
            enabled_results.append(enabled_result)
            print(base._result_summary_line(disabled_result), flush=True)
            print(base._result_summary_line(enabled_result), flush=True)
        comparisons[execution_path] = _comparison_payload(
            execution_path=execution_path,
            disabled_results=disabled_results,
            enabled_results=enabled_results,
        )
        comparison = comparisons[execution_path]
        if comparison.get("status") == "ok":
            means = comparison["primary_seconds_mean"]
            percent = comparison.get("percent_change_vs_off_mean")
            percent_text = "n/a" if percent is None else f"{float(percent):+.1f}%"
            print(
                f"{execution_path}: progress_off mean {float(means['progress_off']):.3f}s, "
                f"progress_on mean {float(means['progress_on']):.3f}s, "
                f"delta {float(comparison['delta_seconds_mean']):+.3f}s ({percent_text})",
                flush=True,
            )
        else:
            print(f"{execution_path}: comparison unavailable", flush=True)

    summary = {
        "label": run_label,
        "created_utc": _timestamp(),
        "gpu_bootstrap": gpu_bootstrap,
        "timing_boundary": base.PRIMARY_TIMING_BOUNDARY,
        "purpose": (
            "Dedicated stderr/stdout-sensitive companion study for the opt-in "
            "cupy-rsoxs energy_progress_bar lane. This runner reuses the maintained "
            "optimization harness internals and only overrides run stream behavior."
        ),
        "target_lane": {
            "resident_mode": "host",
            "size_label": "small",
            "startup_mode": "hot",
            "worker_warmup_runs": 1,
            "trials": int(args.trials),
            "cuda_prewarm_mode": "before_prepare_inputs",
            "isotropic_material_representation": "enum_contract",
            "energies_ev": list(TARGET_ENERGIES),
            "eangle_rotation": list(TARGET_ROTATION),
            "execution_paths": ["tensor_coeff", "direct_polarization"],
        },
        "cases": results,
        "comparisons": comparisons,
    }
    (run_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    print(f"Wrote {run_dir / SUMMARY_NAME}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dedicated stderr/stdout-sensitive companion runner for cupy-rsoxs "
            "energy_progress_bar overhead checks on the small host-hot triple-energy 0:5:165 lane."
        )
    )
    parser.add_argument("--label", default=None, help="Output subdirectory label under test-reports.")
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of paired progress_off/progress_on trials to run per execution path. Default: 5.",
    )
    parser.add_argument(
        "--gpu-index",
        default="auto",
        help=(
            "Single-GPU selection for standalone runs. Defaults to 'auto', which picks the first "
            "physical GPU when CUDA_VISIBLE_DEVICES is otherwise unset."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.trials <= 0:
        raise SystemExit(f"--trials must be positive, got {args.trials!r}.")
    return run_overhead_check(args)


if __name__ == "__main__":
    raise SystemExit(main())
