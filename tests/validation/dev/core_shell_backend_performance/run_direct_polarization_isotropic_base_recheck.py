#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

from tests.validation.dev.core_shell_backend_performance.run_comprehensive_backend_comparison import (  # noqa: E402
    ComparisonCase,
    _cupy_case,
    _run_case_subprocess,
)
from tests.validation.dev.cupy_rsoxs_optimization.run_cupy_rsoxs_optimization_matrix import (  # noqa: E402
    _json_default,
    _timestamp,
)
from tests.validation.lib.core_shell import has_visible_gpu  # noqa: E402


OUT_ROOT = REPO_ROOT / "test-reports" / "core-shell-backend-performance-dev"
SUMMARY_NAME = "direct_polarization_isotropic_base_recheck_summary.json"
TRIPLE_ENERGIES = (284.7, 285.0, 285.2)
ROTATION = (0.0, 15.0, 165.0)
CASE_MATRIX = {
    "small_host__triple_rot_0_15_165": {
        "residency": "host",
        "startup_mode": "hot",
        "size_label": "small",
    },
    "medium_host__triple_rot_0_15_165": {
        "residency": "host",
        "startup_mode": "hot",
        "size_label": "medium",
    },
}
VARIANTS = {
    "baseline": {
        "label": "baseline",
        "description": "Current maintained direct_polarization path.",
        "direct_isotropic_mode": None,
    },
    "item1_cached_base": {
        "label": "item1",
        "description": "Precompute one isotropic base field per energy and reuse it across angles.",
        "direct_isotropic_mode": "cached_base",
    },
}
DEFAULT_SPEED_CASES = tuple(CASE_MATRIX)
DEFAULT_MEMORY_CASES = tuple(CASE_MATRIX)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _series_stats(values: list[float]) -> dict[str, float]:
    ordered = [float(value) for value in values]
    if not ordered:
        raise ValueError("Expected at least one value.")
    return {
        "count": float(len(ordered)),
        "min": float(min(ordered)),
        "median": float(statistics.median(ordered)),
        "mean": float(statistics.fmean(ordered)),
        "max": float(max(ordered)),
    }


def _segment_series_stats(runs: list[dict[str, Any]], segment: str) -> dict[str, float] | None:
    values = [
        float(result["segment_seconds"][segment])
        for result in runs
        if segment in result.get("segment_seconds", {})
    ]
    if not values:
        return None
    return _series_stats(values)


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in raw.split(",") if token.strip())


def _resolve_case_keys(raw: str, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw == "default":
        return default
    if raw == "all":
        return tuple(CASE_MATRIX)
    keys = _parse_csv(raw)
    unknown = [key for key in keys if key not in CASE_MATRIX]
    if unknown:
        raise SystemExit(f"Unsupported case keys: {unknown!r}")
    return keys


def _resolve_variants(raw: str) -> tuple[str, ...]:
    variants = _parse_csv(raw)
    unknown = [variant for variant in variants if variant not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unsupported variants: {unknown!r}")
    return variants


def _build_variant_case(*, variant: str, repeat_index: int, case_key: str) -> ComparisonCase:
    case_spec = CASE_MATRIX[case_key]
    variant_spec = VARIANTS[variant]
    case = _cupy_case(
        residency=case_spec["residency"],
        startup_mode=case_spec["startup_mode"],
        energy_key="triple",
        energy_label="triple",
        energies_ev=TRIPLE_ENERGIES,
        execution_path="direct_polarization",
        z_collapse_mode=None,
        rotation_key="rot_0_15_165",
        rotation_label="0:15:165",
        eangle_rotation=ROTATION,
        size_label=case_spec["size_label"],
    )
    backend_options = dict(case.worker_case.backend_options)
    if variant_spec["direct_isotropic_mode"] is None:
        backend_options.pop("direct_isotropic_mode", None)
    else:
        backend_options["direct_isotropic_mode"] = variant_spec["direct_isotropic_mode"]
    worker_case = replace(
        case.worker_case,
        label=f"{case.worker_case.label}__{variant}__r{repeat_index + 1}",
        backend_options=backend_options,
        notes=(
            f"{case.worker_case.notes} Temporary direct isotropic experiment={variant_spec['direct_isotropic_mode']!r}. "
            f"Variant description: {variant_spec['description']}"
        ),
    )
    return replace(
        case,
        key=f"{case.key}__{variant}__r{repeat_index + 1}",
        path_variant=(
            f"{case.path_variant}__{variant_spec['label']}"
            if variant != "baseline"
            else case.path_variant
        ),
        worker_case=worker_case,
    )


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "runs": runs,
        "primary_seconds_stats": _series_stats([float(result["primary_seconds"]) for result in runs]),
    }
    for segment in ("A2", "B", "C", "D", "E"):
        segment_stats = _segment_series_stats(runs, segment)
        if segment_stats is not None:
            summary.setdefault("segment_seconds_stats", {})[segment] = segment_stats
    return summary


def _summarize_memory_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runs": runs,
        "peak_gpu_delta_mib_stats": _series_stats(
            [float(result["memory_probe"]["peak_gpu_delta_mib"]) for result in runs]
        ),
        "sample_count_stats": _series_stats(
            [float(result["memory_probe"]["sample_count"]) for result in runs]
        ),
    }


def _run_recheck(args: argparse.Namespace) -> int:
    if not has_visible_gpu():
        raise SystemExit(
            "No visible NVIDIA GPU found for the direct-polarization isotropic-base recheck."
        )

    variants = _resolve_variants(args.variants)
    speed_case_keys = _resolve_case_keys(args.speed_cases, default=DEFAULT_SPEED_CASES)
    memory_case_keys = _resolve_case_keys(args.memory_cases, default=DEFAULT_MEMORY_CASES)

    run_label = args.label or f"dp_isotropic_base_{_timestamp()}"
    run_dir = OUT_ROOT / run_label
    speed_dir = run_dir / "speed_case_results"
    memory_dir = run_dir / "memory_case_results"
    summary_path = run_dir / SUMMARY_NAME

    summary: dict[str, Any] = {
        "label": run_label,
        "created_utc": _timestamp(),
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_device": str(args.gpu_device),
        "repeats": int(args.repeats),
        "memory_poll_interval_s": float(args.memory_poll_interval_s),
        "variants": list(variants),
        "speed_case_keys": list(speed_case_keys),
        "memory_case_keys": list(memory_case_keys),
        "fixed_lane": {
            "energies_ev": list(TRIPLE_ENERGIES),
            "eangle_rotation": list(ROTATION),
            "isotropic_representation": "enum_contract",
            "execution_path": "direct_polarization",
            "startup_mode": "hot",
        },
        "cases": {
            case_key: {
                "residency": CASE_MATRIX[case_key]["residency"],
                "startup_mode": CASE_MATRIX[case_key]["startup_mode"],
                "size_label": CASE_MATRIX[case_key]["size_label"],
            }
            for case_key in CASE_MATRIX
        },
        "speed_cases": {},
        "memory_cases": {},
        "decision": {},
    }

    for variant in variants:
        summary["speed_cases"][variant] = {}
        for case_key in speed_case_keys:
            run_results = []
            for repeat_index in range(args.repeats):
                case = _build_variant_case(
                    variant=variant,
                    repeat_index=repeat_index,
                    case_key=case_key,
                )
                result = _run_case_subprocess(
                    case=case,
                    output_dir=speed_dir,
                    gpu_device=str(args.gpu_device),
                    monitor_memory=False,
                    poll_interval_s=args.memory_poll_interval_s,
                    skip_existing=False,
                )
                run_results.append(result)
                print(
                    f"[speed] variant={variant} case={case_key} run={repeat_index + 1}/{args.repeats} "
                    f"primary={float(result['primary_seconds']):.6f}s",
                    flush=True,
                )
            summary["speed_cases"][variant][case_key] = _summarize_runs(run_results)
            _write_json(summary_path, summary)

        summary["memory_cases"][variant] = {}
        for case_key in memory_case_keys:
            run_results = []
            for repeat_index in range(args.repeats):
                case = _build_variant_case(
                    variant=variant,
                    repeat_index=repeat_index,
                    case_key=case_key,
                )
                result = _run_case_subprocess(
                    case=case,
                    output_dir=memory_dir,
                    gpu_device=str(args.gpu_device),
                    monitor_memory=True,
                    poll_interval_s=args.memory_poll_interval_s,
                    skip_existing=False,
                )
                run_results.append(result)
                print(
                    f"[memory] variant={variant} case={case_key} run={repeat_index + 1}/{args.repeats} "
                    f"peak_delta={float(result['memory_probe']['peak_gpu_delta_mib']):.3f}MiB "
                    f"samples={int(result['memory_probe']['sample_count'])}",
                    flush=True,
                )
            summary["memory_cases"][variant][case_key] = _summarize_memory_runs(run_results)
            _write_json(summary_path, summary)

    baseline_variant = "baseline"
    for variant in variants:
        if variant == baseline_variant or baseline_variant not in variants:
            continue
        summary["decision"][variant] = {}
        for case_key in CASE_MATRIX:
            decision: dict[str, Any] = {}
            if case_key in speed_case_keys:
                baseline_primary = summary["speed_cases"][baseline_variant][case_key][
                    "primary_seconds_stats"
                ]["median"]
                candidate_primary = summary["speed_cases"][variant][case_key][
                    "primary_seconds_stats"
                ]["median"]
                decision["speed_ratio_vs_baseline"] = float(candidate_primary / baseline_primary)
            if case_key in memory_case_keys:
                baseline_peak = summary["memory_cases"][baseline_variant][case_key][
                    "peak_gpu_delta_mib_stats"
                ]["median"]
                candidate_peak = summary["memory_cases"][variant][case_key][
                    "peak_gpu_delta_mib_stats"
                ]["median"]
                decision["memory_peak_ratio_vs_baseline"] = (
                    float(candidate_peak / baseline_peak) if baseline_peak else float("inf")
                )
                decision["baseline_peak_gpu_delta_mib"] = float(baseline_peak)
                decision["candidate_peak_gpu_delta_mib"] = float(candidate_peak)
            if decision:
                summary["decision"][variant][case_key] = decision

    _write_json(summary_path, summary)
    print(f"Wrote {summary_path}", flush=True)
    for variant, variant_decisions in summary["decision"].items():
        for case_key, decision in variant_decisions.items():
            line = f"[decision] variant={variant} case={case_key}"
            if "speed_ratio_vs_baseline" in decision:
                line += f" speed_ratio={decision['speed_ratio_vs_baseline']:.4f}"
            if "memory_peak_ratio_vs_baseline" in decision:
                line += f" memory_ratio={decision['memory_peak_ratio_vs_baseline']:.4f}"
            print(line, flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only recheck for temporary direct_polarization isotropic-path variants "
            "on the host-hot enum-contract triple-energy 0:15:165 lane."
        )
    )
    parser.add_argument("--label", default=None, help="Output subdirectory label under test-reports.")
    parser.add_argument(
        "--gpu-device",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0") or "0",
        help="Single visible GPU id passed through to the worker subprocesses.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated worker subprocess runs per variant/case for each pass.",
    )
    parser.add_argument(
        "--memory-poll-interval-s",
        type=float,
        default=0.02,
        help="Fast CuPy observer sampling cadence for the memory pass.",
    )
    parser.add_argument(
        "--variants",
        default="baseline,item1_cached_base",
        help=(
            "Comma-separated variant list. Supported values: "
            f"{','.join(VARIANTS)}."
        ),
    )
    parser.add_argument(
        "--speed-cases",
        default="default",
        help="Comma-separated case keys, 'default', or 'all' for the speed pass.",
    )
    parser.add_argument(
        "--memory-cases",
        default="default",
        help="Comma-separated case keys, 'default', or 'all' for the memory pass.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(_run_recheck(build_parser().parse_args()))
