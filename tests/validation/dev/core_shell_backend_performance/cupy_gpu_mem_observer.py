#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sample_gpu_used_mib(cp, device_index: int) -> float:
    cp.cuda.Device(device_index).use()
    free_b, total_b = cp.cuda.runtime.memGetInfo()
    return float(total_b - free_b) / (1024.0**2)


def _stabilize_observer(cp, args: argparse.Namespace) -> tuple[float, list[float], int]:
    window: list[float] = []
    sample_count = 0
    deadline = time.monotonic() + float(args.startup_timeout_s)
    while time.monotonic() < deadline:
        used_mib = _sample_gpu_used_mib(cp, args.device_index)
        window.append(used_mib)
        sample_count += 1
        if len(window) > args.stabilize_window:
            window.pop(0)
        if len(window) == args.stabilize_window:
            spread = max(window) - min(window)
            if spread <= args.stabilize_tolerance_mib:
                return float(window[-1]), list(window), sample_count
        time.sleep(args.sample_interval_s)
    raise TimeoutError(
        "GPU memory observer did not stabilize before the startup timeout elapsed."
    )


def _run_observer(args: argparse.Namespace) -> int:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    import cupy as cp

    cp.cuda.Device(args.device_index).use()
    # Force CUDA context creation inside the observer process before baseline capture.
    warm = cp.arange(1, dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()
    del warm
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()

    ready_path = Path(args.ready_path)
    output_path = Path(args.output_path)
    stop_path = Path(args.stop_path)

    started_monotonic = time.monotonic()
    try:
        baseline_used_mib, stable_window_mib, warmup_sample_count = _stabilize_observer(cp, args)
        ready_payload = {
            "status": "ready",
            "baseline_gpu_used_mib": float(baseline_used_mib),
            "sample_interval_s": float(args.sample_interval_s),
            "stabilize_window": int(args.stabilize_window),
            "stabilize_tolerance_mib": float(args.stabilize_tolerance_mib),
            "warmup_sample_count": int(warmup_sample_count),
            "stable_window_mib": stable_window_mib,
            "observer_startup_seconds": float(time.monotonic() - started_monotonic),
        }
        _write_json(ready_path, ready_payload)

        samples: list[dict[str, float]] = []
        peak_used_mib = baseline_used_mib
        while not stop_path.exists():
            sample_time = time.monotonic()
            used_mib = _sample_gpu_used_mib(cp, args.device_index)
            peak_used_mib = max(peak_used_mib, used_mib)
            samples.append(
                {
                    "t_monotonic_s": float(sample_time),
                    "used_mib": float(used_mib),
                    "delta_mib": float(max(0.0, used_mib - baseline_used_mib)),
                }
            )
            time.sleep(args.sample_interval_s)

        output_payload = {
            "status": "ok",
            "probe_method": "cupy_memgetinfo_observer",
            "device_index": int(args.device_index),
            "sample_interval_s": float(args.sample_interval_s),
            "baseline_gpu_used_mib": float(baseline_used_mib),
            "peak_gpu_used_mib": float(peak_used_mib),
            "peak_gpu_delta_mib": float(max(0.0, peak_used_mib - baseline_used_mib)),
            "sample_count": int(len(samples)),
            "warmup_sample_count": int(warmup_sample_count),
            "stable_window_mib": stable_window_mib,
            "observer_startup_seconds": float(time.monotonic() - started_monotonic),
            "samples": samples,
        }
        _write_json(output_path, output_payload)
        return 0
    except BaseException as exc:  # noqa: BLE001 - observer must serialize failures
        _write_json(
            ready_path,
            {
                "status": "error",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            },
        )
        _write_json(
            output_path,
            {
                "status": "error",
                "probe_method": "cupy_memgetinfo_observer",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            },
        )
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only warmed GPU-memory observer for the CoreShell backend comparison."
        )
    )
    parser.add_argument("--ready-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--stop-path", required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--sample-interval-s", type=float, default=0.01)
    parser.add_argument("--stabilize-window", type=int, default=5)
    parser.add_argument("--stabilize-tolerance-mib", type=float, default=8.0)
    parser.add_argument("--startup-timeout-s", type=float, default=30.0)
    return parser


def main() -> int:
    return _run_observer(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
