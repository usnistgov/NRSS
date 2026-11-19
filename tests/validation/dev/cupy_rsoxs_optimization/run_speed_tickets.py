"""Launch matched CoreShell workers for the pre-incidence CuPy speed tickets.

The driver uses the existing BenchmarkCase worker protocol. Run with --list or
--dry-run before allocating a GPU. See optimization/cupy_rsoxs/speed_tickets.
"""
import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
WORKER = Path("tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py")
OPTICS = Path("tests/validation/data/core_shell")
ENERGIES = {
    "one": [285.0],
    "three": [284.7, 285.0, 285.2],
    "twenty": [280.0 + i / 2 for i in range(20)],
    "scan": [i / 10 for i in range(2800, 2901)],
}
# Each entry supplies the size, energy list and [start, increment, end] in degrees.
CASES = {
    "one": ("small", "one", [0, 0, 0]),
    "angles": ("small", "one", [0, 5, 165]),
    "three": ("small", "three", [0, 15, 165]),
    "energy20": ("small", "twenty", [0, 15, 165]),
    "energy101": ("small", "scan", [0, 15, 165]),
    "depth": ("medium", "one", [0, 0, 0]),
}
PATHS = ("tensor_coeff", "direct_polarization", "default")
STAGES = [
    "A1", "A2", "B", "C", "D", "E", "F", "C+D", "B+C+D", "energy", "tile", "basis_build",
    "reduction", "d2h", "synchronization",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--case", choices=CASES, default="one")
    parser.add_argument("--energies", help="Override case energies with comma-separated eV values in output order")
    parser.add_argument("--path", choices=PATHS, default="tensor_coeff")
    parser.add_argument("--baseline-root", type=Path, default=ROOT)
    compare = parser.add_mutually_exclusive_group()
    compare.add_argument("--aa", action="store_true", help="Two identical controls to measure repeat variability")
    compare.add_argument("--candidate-root", type=Path, help="Also run the worker in this checkout")
    parser.add_argument("--candidate-path", choices=PATHS, help="Select a different execution path for arm B")
    parser.add_argument("--candidate-options", type=json.loads, default={},
                        help="JSON object of registered experimental backend options, candidate workers only")
    parser.add_argument("--resident", choices=("device", "host"), default="device", help="Input residency")
    parser.add_argument("--result-residency", choices=("host", "device"), default="host")
    parser.add_argument("--layout", choices=("detector", "integrated", "i_only", "i_para_i_perp", "i_a"),
                        default="detector")
    parser.add_argument("--result-chunk-size", type=int, default=1, help="Existing output chunk size")
    parser.add_argument("--result-q-grid-mode", choices=("intersection", "union"),
                        help="Shared q grid for both arms")
    parser.add_argument("--total-chi-wedge-deg", type=float, help="Sector wedge width for both arms")
    parser.add_argument("--shape-label", help="Override the size with a label supported by both workers")
    parser.add_argument("--fixture", choices=("core_shell", "generic"), default="core_shell")
    rotation = parser.add_mutually_exclusive_group()
    rotation.add_argument("--rotation", choices=("off", "12", "34"), help="Single E direction, 12 or 34 angles")
    rotation.add_argument("--angle", type=float, help="Hold one electric-field direction at this angle in degrees")
    parser.add_argument("--windowing-type", type=int, choices=(0, 1), default=0)
    parser.add_argument("--rot-mask", type=int, choices=(0, 1), default=0)
    parser.add_argument("--material-z", type=int, help="Change the material depth at fixed XY and voxel size")
    parser.add_argument("--geometry", action="store_true", help="Measure detector demand after a stage run")
    parser.add_argument("--transform-z", type=int, help="Probe detector geometry for a distinct transform depth")
    parser.add_argument("--gpu", help="One GPU index or UUID, defaults to inherited selection or 0")
    parser.add_argument("--repeats", type=int, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stages", action="store_true", help="Time stages and count the operations that run")
    mode.add_argument("--accuracy", choices=("screen", "full"), help="Compare first/middle/last or all energies")
    mode.add_argument("--memory", action="store_true", help="Run a separate GPU-memory observation pass")
    parser.add_argument("--memory-poll-interval", type=float, default=0.01, metavar="SECONDS")
    parser.add_argument("--accuracy-rtol", type=float, default=0.0)
    parser.add_argument("--accuracy-atol", type=float, default=0.0)
    parser.add_argument("--expect-route", help="Required actual route in the diagnostic run for B (or A alone)")
    parser.add_argument("--out-root", type=Path, default=ROOT / "test-reports" / "cupy-speed-tickets")
    args = parser.parse_args()
    if args.energies is None:
        args.energies = ENERGIES[CASES[args.case][1]]
    else:
        try:
            args.energies = [float(value) for value in args.energies.split(",")]
        except ValueError:
            parser.error("energies must be comma-separated numbers in eV")
        if any(not math.isfinite(energy) or energy <= 0 for energy in args.energies):
            parser.error("energies must be finite and positive")
    if args.repeats < 1 or args.result_chunk_size < 1:
        parser.error("repeats and result chunk size must be positive")
    if args.angle is not None and not math.isfinite(args.angle):
        parser.error("the electric-field angle must be finite")
    if (args.geometry or args.expect_route) and not args.stages:
        parser.error("geometry and route checks require --stages")
    if args.transform_z is not None and not args.geometry:
        parser.error("--transform-z requires --geometry")
    if any(depth is not None and depth < 1 for depth in (args.material_z, args.transform_z)):
        parser.error("material and transform depths must be positive")
    if args.accuracy and not (args.aa or args.candidate_root or args.candidate_path or args.candidate_options):
        parser.error("accuracy comparison requires two arms")
    if any(not math.isfinite(tol) or tol < 0 for tol in (args.accuracy_rtol, args.accuracy_atol)):
        parser.error("accuracy tolerances must be finite and nonnegative")
    if not math.isfinite(args.memory_poll_interval) or args.memory_poll_interval <= 0:
        parser.error("memory sampling interval must be finite and positive")
    if not isinstance(args.candidate_options, dict) or (args.aa and (args.candidate_options or args.candidate_path)):
        parser.error("candidate options must be a JSON object and A/A uses identical paths and options")
    if set(args.candidate_options) & {"execution_path", "result_layout", "result_chunk_size",
                                     "result_residency", "z_collapse_mode", "mixed_precision_mode",
                                     "result_q_grid_mode", "total_chi_wedge_deg"}:
        parser.error("compare the same path, physics, result contract and precision in both workers")
    return args


def check_energy_tables(root, energies):
    for material in range(1, 4):
        table = root / OPTICS / f"Material{material}.txt"
        keys = {float(m) for m in re.findall(r"^Energy\s*=\s*([^;]+);", table.read_text(), re.M)}
        missing = set(energies) - keys
        if missing:
            raise ValueError(f"{table}: missing optical keys {sorted(missing)}")


def case_payload(args, label, candidate):
    size, _, angles = CASES[args.case]
    if args.rotation:
        angles = {"off": [0, 0, 0], "12": [0, 15, 165], "34": [0, 5, 165]}[args.rotation]
    if args.angle is not None:
        angles = [args.angle, 0, args.angle]
    options = dict(result_layout=args.layout, result_residency=args.result_residency,
                   result_chunk_size=args.result_chunk_size)
    if args.result_q_grid_mode is not None:
        options["result_q_grid_mode"] = args.result_q_grid_mode
    if args.total_chi_wedge_deg is not None:
        options["total_chi_wedge_deg"] = args.total_chi_wedge_deg
    path = args.candidate_path if candidate and args.candidate_path is not None else args.path
    if path != "default":
        options["execution_path"] = path
    if candidate:
        options.update(args.candidate_options)
    return dict(label=label, family=args.fixture, backend="cupy-rsoxs", shape_label=args.shape_label or size,
                energies_ev=args.energies, eangle_rotation=angles,
                field_namespace="cupy" if args.resident == "device" else "numpy",
                isotropic_representation="enum_contract", cuda_prewarm_mode="before_prepare_inputs",
                resident_mode=args.resident, input_policy="strict", ownership_policy="borrow",
                backend_options=options, timing_segments=STAGES if args.stages else [],
                worker_warmup_runs=0 if args.accuracy else 1, run_stdout=False, run_stderr=False,
                accuracy=args.accuracy, material_z=args.material_z, geometry=args.geometry,
                transform_z=args.transform_z, windowing_type=args.windowing_type, rot_mask=args.rot_mask)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checkout_info(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    backend = root / "src" / "NRSS" / "backends" / "cupy_rsoxs.py"
    info = {"root": str(root), "revision": git("rev-parse", "HEAD"),
            "backend_sha256": hashlib.sha256(backend.read_bytes()).hexdigest(),
            "working_tree": git("status", "--short", "--untracked-files=normal")}
    basis_source = backend.with_name("_cupy_energy_basis.py")
    if basis_source.is_file():
        info["energy_basis_sha256"] = hashlib.sha256(basis_source.read_bytes()).hexdigest()
    return info


@contextmanager
def observe_memory(folder, label, gpu, interval):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tests.validation.dev.core_shell_backend_performance.run_comprehensive_backend_comparison import (
        GPU_MEMORY_OBSERVER_SHUTDOWN_TIMEOUT_S,
        GPU_MEMORY_OBSERVER_STARTUP_TIMEOUT_S,
        _start_gpu_memory_observer,
        _stop_gpu_memory_observer,
        _wait_for_gpu_memory_observer_ready,
    )

    observer, ready_path, trace_path, stop_path = _start_gpu_memory_observer(
        output_dir=folder, case_key=label, gpu_device=gpu, poll_interval_s=interval,
    )
    measurement = {}
    try:
        ready = _wait_for_gpu_memory_observer_ready(
            observer, ready_path=ready_path, timeout_s=GPU_MEMORY_OBSERVER_STARTUP_TIMEOUT_S,
        )
        yield measurement
        trace = _stop_gpu_memory_observer(
            observer, stop_path=stop_path, output_path=trace_path, timeout_s=GPU_MEMORY_OBSERVER_SHUTDOWN_TIMEOUT_S,
        )
        if trace.get("status") != "ok" or not trace["sample_count"]:
            raise RuntimeError(f"GPU memory observer failed ({trace_path})")
        measurement.update(
            free_gpu_mib_at_launch=ready["free_gpu_mib"],
            peak_gpu_delta_mib=trace["peak_gpu_delta_mib"],
            sample_interval_s=trace["sample_interval_s"],
            sample_count=trace["sample_count"],
            trace=str(trace_path),
        )
    finally:
        if observer.poll() is None:
            observer.terminate()
            try:
                observer.communicate(timeout=GPU_MEMORY_OBSERVER_SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                observer.kill()
                observer.communicate()
        stop_path.unlink(missing_ok=True)


def run_worker(root, spec, folder, env, args):
    label = spec["label"]
    case_file = folder / f"{label}.case.json"
    result_file = folder / f"{label}.result.json"
    log_file = folder / f"{label}.log"
    if spec.get("accuracy"):
        spec["output_path"] = str(folder / f"{label}.npz")
    write_json(case_file, spec)
    worker_env = env.copy()
    worker_env["NRSS_SPEED_CHECKOUT"] = str(root)
    observation = observe_memory(folder, label, env["CUDA_VISIBLE_DEVICES"], args.memory_poll_interval) if args.memory else nullcontext()
    with log_file.open("w") as log, observation as measurement:
        run = subprocess.run([sys.executable, str(ROOT / WORKER), "--worker-case-path", str(case_file),
                              "--worker-result-path", str(result_file)], cwd=root, env=worker_env,
                             stdout=log, stderr=subprocess.STDOUT)
    result = json.loads(result_file.read_text()) if result_file.exists() else {}
    elapsed = result.get("primary_seconds")
    if run.returncode or result.get("status") != "ok":
        raise RuntimeError(f"{label} failed: {result.get('error', 'see worker log')} ({log_file})")
    if not args.accuracy and (not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0):
        raise RuntimeError(f"{label}: worker reported an invalid time ({result_file})")
    if result.get("nrss_module") != str((root / "src" / "NRSS" / "__init__.py").resolve()):
        raise RuntimeError(f"{label}: worker imported NRSS from another checkout ({result_file})")
    resolved = result.get("resolved_backend_options", {})
    for option in ("execution_path", "result_layout", "result_residency"):
        if option in spec["backend_options"] and resolved.get(option) != spec["backend_options"][option]:
            raise RuntimeError(f"{label}: worker resolved a different {option} ({result_file})")
    if result.get("energies_ev") != spec["energies_ev"] or result.get("eangle_rotation") != spec["eangle_rotation"]:
        raise RuntimeError(f"{label}: energy or angle selection changed ({result_file})")
    shape = result.get("panel_shape", [])
    if not shape or shape[0] != len(spec["energies_ev"]):
        raise RuntimeError(f"{label}: returned shape {shape} has the wrong energy count")
    if args.accuracy and (not result.get("arrays") or not Path(spec["output_path"]).is_file()):
        raise RuntimeError(f"{label}: missing accuracy output ({result_file})")
    if measurement is not None:
        result["memory"] = measurement
        write_json(result_file, result)
    return result


def compare_accuracy(left, right, left_path, right_path, *, rtol, atol):
    import numpy as np

    errors = []
    channels = {}
    if left != right:
        return {"status": "failed", "errors": ["Coordinate or variable names, dimensions, or shapes changed"]}

    with np.load(left_path) as baseline, np.load(right_path) as candidate:
        energies = baseline[left["coords"]["energy"]["key"]]
        for name, item in left["coords"].items():
            a, b = baseline[item["key"]], candidate[item["key"]]
            if (list(a.shape) != item["shape"] or a.shape != b.shape
                    or np.isinf(a).any() or np.isinf(b).any()
                    or not np.array_equal(a, b, equal_nan=True)):
                errors.append(f"Coordinate {name} changed or contains infinity")

        for name, item in left["values"].items():
            a, b = baseline[item["key"]], candidate[item["key"]]
            if list(a.shape) != item["shape"] or a.shape != b.shape:
                errors.append(f"{name}: result shape changed")
                continue
            axis = item["dims"].index("energy")
            metrics = []
            for energy, x, y in zip(energies, np.moveaxis(a, axis, 0), np.moveaxis(b, axis, 0)):
                label = f"{name} at {energy:g} eV"
                finite = np.isfinite(x)
                if np.isinf(x).any() or np.isinf(y).any():
                    errors.append(f"{label} contains infinity")
                    continue
                if not np.array_equal(finite, np.isfinite(y)):
                    errors.append(f"{label}: finite/NaN mask changed")
                    continue
                reference = x[finite].astype(np.float64)
                delta = y[finite].astype(np.float64) - reference
                peak = float(np.abs(reference).max(initial=0.0))
                max_abs = float(np.abs(delta).max(initial=0.0))
                norm = float(np.linalg.norm(reference))
                l2 = float(np.linalg.norm(delta))
                metrics.append({
                    "energy_ev": float(energy), "finite_pixels": int(finite.sum()),
                    "reference_peak": peak, "max_absolute_difference": max_abs,
                    "normalized_l2": l2 / norm if norm else None,
                })
                if max_abs > atol + rtol * peak or l2 > atol * math.sqrt(reference.size) + rtol * norm:
                    errors.append(f"{label}: intensity difference exceeds tolerance")
            channels[name] = metrics

    return {"status": "failed" if errors else "ok", "errors": errors, "channels": channels,
            "rtol": rtol, "atol": atol}


def main():
    args = parse_args()
    if args.list:
        for name, (size, key, angles) in CASES.items():
            print(f"{name:10s} {size:7s} energies={len(ENERGIES[key]):3d}   E angles {angles} [start, increment, end]")
        return
    gpu = args.gpu if args.gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if not gpu.strip() or "," in gpu:
        raise SystemExit("Select one available GPU with --gpu or CUDA_VISIBLE_DEVICES")
    roots = {"A": args.baseline_root.resolve()}
    if args.aa or args.candidate_root or args.candidate_path or args.candidate_options:
        roots["B"] = (args.candidate_root or args.baseline_root).resolve()
    for root in set(roots.values()):
        if not (root / "src" / "NRSS" / "backends" / "cupy_rsoxs.py").is_file():
            raise SystemExit(f"Missing CuPy backend source in {root}")
        if args.fixture == "core_shell":
            check_energy_tables(root, args.energies)
    repeats = 1 if args.accuracy else args.repeats
    order = [list(roots) if idx % 2 == 0 else list(reversed(roots)) for idx in range(repeats)]
    mode = "accuracy" if args.accuracy else "stages" if args.stages else "memory" if args.memory else "timing"
    plan = {"host": socket.getfqdn(), "gpu": gpu, "python": sys.executable, "mode": mode,
            "checkouts": {k: checkout_info(v) for k, v in roots.items()},
            "paired_order": order, "case": args.case,
            "conditions": {k: case_payload(args, k, k == "B") for k in roots}}
    if args.accuracy:
        plan["accuracy_tolerances"] = {"rtol": args.accuracy_rtol, "atol": args.accuracy_atol}
    if args.expect_route:
        plan["expected_route"] = args.expect_route
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    args.out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(tempfile.mkdtemp(prefix=f"{stamp}-{args.case}-", dir=args.out_root.resolve()))
    write_json(folder / "plan.json", plan)
    print(folder, flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONDONTWRITEBYTECODE="1")
    pairs = []
    worker_results = []
    for index, group in enumerate(order, start=1):
        row = {}
        for arm in group:
            label = f"r{index:02d}-{arm}"
            result = run_worker(roots[arm], case_payload(args, label, arm == "B"), folder, env, args)
            if args.expect_route and arm == ("B" if "B" in roots else "A"):
                execution = result.get("backend_timing_details", {}).get("execution", {})
                if execution.get("routes") != [args.expect_route]:
                    raise RuntimeError(f"{label}: expected actual route {args.expect_route}, got {execution.get('routes')}")
            row[arm] = result
            print(f"{label} {result['primary_seconds']:.5f} s" if mode == "timing" else f"{label} {mode} complete", flush=True)
            worker_results.append(f"{label}.result.json")
        pairs.append(row)
    summary = {"mode": mode, "worker_results": worker_results}
    if mode == "accuracy":
        a, b = pairs[0]["A"], pairs[0]["B"]
        summary.update(compare_accuracy(
            a["arrays"], b["arrays"], folder / "r01-A.npz", folder / "r01-B.npz",
            rtol=args.accuracy_rtol, atol=args.accuracy_atol,
        ))
    elif mode == "timing":
        times = [{arm: row[arm]["primary_seconds"] for arm in roots} for row in pairs]
        median = {arm: statistics.median(pair[arm] for pair in times) for arm in roots}
        energy_count = len(args.energies)
        summary.update(
            primary_seconds=times,
            median_seconds=median,
            median_seconds_per_energy={arm: seconds / energy_count for arm, seconds in median.items()},
            median_energies_per_second={arm: energy_count / seconds for arm, seconds in median.items()},
        )
        if len(roots) == 2:
            ratios = [row["A"] / row["B"] for row in times]
            summary.update(paired_speedups=ratios, median_paired_speedup=statistics.median(ratios),
                           candidate_faster_pairs=sum(x > 1 for x in ratios))
            if args.aa:
                summary["aa_median_absolute_fractional_deviation"] = statistics.median(abs(x - 1) for x in ratios)
    write_json(folder / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if summary.get("status") == "failed":
        raise SystemExit(f"Accuracy comparison failed ({folder / 'summary.json'})")


if __name__ == "__main__":
    main()
