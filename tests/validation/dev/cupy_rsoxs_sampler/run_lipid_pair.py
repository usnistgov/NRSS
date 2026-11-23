"""Run matched revisions on the frozen 101-energy ZP09 lipid scan."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import resource
import socket
from statistics import median
import subprocess
import sys
import tempfile
from time import perf_counter

HERE = Path(__file__).resolve()
CASES = {
    "detector-1": ((256, 1024, 1024), "detector", 1, None),
    "detector-4": ((256, 1024, 1024), "detector", 4, None),
    "ia-full": ((1024, 1024, 1024), "i_a", 101, None),
    "ia-qmax": ((1024, 1024, 1024), "i_a", 101, 0.8),
}


def write(path, value):
    def special(item):
        if hasattr(item, "tolist"):
            return item.tolist()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, default=special, indent=2, allow_nan=False) + "\n")


def options_for(name):
    target, layout, chunk, cutoff = CASES[name]
    options = dict(virtual_3d_pad_to=target, virtual_pad_material=3,
                   result_layout=layout, result_chunk_size=chunk, result_residency="host",
                   energy_progress_bar=False)
    if cutoff is not None:
        options["q_max"] = cutoff
    return options


def run_worker(job, root, output):
    sys.path[:0] = [str(root / "src"), str(root)]
    import cupy as cp
    import numpy as np
    import NRSS
    from NRSS import SFieldMode
    from NRSS.morphology import Material, Morphology
    from tests.validation.dev.cupy_rsoxs_virtual_3dpad.run_padded_scan_baseline import load_fixture
    from tests.validation.dev.cupy_rsoxs_optimization.run_speed_tickets import STAGES, checkout_info
    from tests.validation.dev.cupy_rsoxs_qmax.run_public_pair import save_sample, source_fingerprint

    source = Path(NRSS.__file__).resolve()
    if source != root / "src/NRSS/__init__.py":
        raise RuntimeError(f"Wrong checkout imported at {source}")
    fields, energies, optics, _, hashes = load_fixture(Path(job["manifest"]), 101)
    options = options_for(job["case"])
    materials = {}
    for i, name in enumerate(("head", "tail", "water"), start=1):
        orientation = {"S": SFieldMode.ISOTROPIC}
        if name == "tail":
            orientation = {attr: fields["tail_" + attr.lower()] for attr in ("S", "theta", "psi")}
        materials[i] = Material(i, name=name, Vfrac=fields[name + "_vfrac"], energies=energies,
                                opt_constants=dict(zip(energies, optics[i-1].tolist())), **orientation)
    config = dict(Morphology.config_default, Energies=energies, EAngleRotation=[0, 0, 0],
                  WindowingType=0, RotMask=0)
    device = cp.cuda.Device(0)
    device.use()
    cp.cuda.runtime.free(0)
    morph = Morphology(3, materials=materials, PhysSize=1.5, config=config, backend="cupy-rsoxs",
                       input_policy="strict", ownership_policy="borrow", backend_options=options)
    phase = job["phase"]
    try:
        if tuple(morph.NumZYX) != (52, 52, 52) or list(morph.Energies) != energies:
            raise ValueError("Frozen morphology shape or scan changed")
        for key, expected in options.items():
            observed = morph.backend_options.get(key)
            if key == "virtual_3d_pad_to":
                observed = tuple(observed)
            if observed != expected:
                raise ValueError(f"Unexpected {key}: {observed!r}, expected {expected!r}")
        if morph.backend_options.get("q_max") != options.get("q_max"):
            raise ValueError("Mismatch in resolved cutoff")

        def once():
            device.synchronize()
            start = perf_counter()
            result = morph.run(stdout=False, stderr=False, return_xarray=False)
            device.synchronize()
            elapsed = perf_counter() - start
            return elapsed, result

        times = []
        # The first call and each warm call are separate. Do not hold an old public result.
        for _ in range(job["warms"] + 1 if phase in ("time", "memory", "stages") else 1):
            elapsed, result = once()
            times.append(elapsed)
            if phase != "check":
                del result
        extra = {}
        if phase == "stages":
            morph._set_private_backend_timing_segments(STAGES)
            _, result = once()
            extra["timings"] = morph.backend_timings
            e = extra["timings"]["execution"]
            if e["routes"] != ["virtual_3d_sum"] or e["basis_builds"] != 1 or e["distinct_energies"] != 101:
                raise RuntimeError(f"Unexpected XYZ scan: {e}")
        if phase in ("check", "stages"):
            arr = result.to_backend_array()
            if not isinstance(arr, np.ndarray) or arr.shape[0] != 101:
                raise ValueError("Expected all 101 energies in a host result")
            target, layout, _, _ = CASES[job["case"]]
            if layout == "detector" and (arr.dtype != np.float32 or arr.shape != (101, *target[1:])):
                raise ValueError(f"Unexpected detector shape or dtype: {arr.shape} {arr.dtype}")
            if layout == "i_a" and (arr.dtype != np.float64 or arr.shape[:2] != (101, 2)):
                raise ValueError(f"Unexpected I/A shape or dtype: {arr.shape} {arr.dtype}")
            extra.update(return_shape=arr.shape, return_dtype=str(arr.dtype), result_bytes=arr.nbytes)
            if phase == "check":
                save_sample(result.to_xarray(), output.with_suffix(".npz"))
            del arr, result
        if phase == "memory":
            extra["peak_host_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
                1024**2 if sys.platform == "darwin" else 1024)
            pool = cp.get_default_memory_pool()
            extra.update(pool_used_end_mib=pool.used_bytes()/2**20, pool_reserved_end_mib=pool.total_bytes()/2**20)
        props = cp.cuda.runtime.getDeviceProperties(0)
        write(output, dict(case=job["case"], phase=phase, options=options, energies=energies,
                           fixture_hashes=hashes, checkout=checkout_info(root),
                           source_sha256=source_fingerprint(root), nrss_import=str(source),
                           host=socket.gethostname(), pid=os.getpid(), gpu=props["name"].decode(),
                           gpu_uuid=bytes(props["uuid"]).hex(), driver=cp.cuda.runtime.driverGetVersion(),
                           cuda=cp.cuda.runtime.runtimeGetVersion(), cupy=cp.__version__,
                           first_seconds=times[0], warm_seconds=times[1:], **extra))
    finally:
        morph.release_runtime()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=3, type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--case", choices=CASES, default="detector-1")
    parser.add_argument("--phase", choices=("plan", "time", "check", "stages", "memory"), default="plan")
    parser.add_argument("--aa", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--warms", type=int, default=5)
    parser.add_argument("--poll-ms", type=float, default=5.0)
    args = parser.parse_args()
    if args.worker:
        job, root, output = args.worker
        run_worker(json.loads(job.read_text()), root, output)
        return
    if args.parent_root is None or args.manifest is None or (not args.aa and args.candidate_root is None):
        parser.error("Supply a manifest, a parent checkout, and either --candidate-root or --aa")
    if not args.gpu.strip() or "," in args.gpu or min(args.pairs, args.warms, args.poll_ms) <= 0:
        parser.error("Use one GPU, and positive pair, warm-call and polling settings")
    roots = {"A": args.parent_root.resolve(), "B": (args.parent_root if args.aa else args.candidate_root).resolve()}
    for root in roots.values():
        if not (root / "src/NRSS/backends/_cupy_energy_basis.py").is_file():
            parser.error(f"Missing NRSS checkout at {root}")
    manifest = args.manifest.resolve()
    n = args.pairs if args.phase in ("time", "plan") else 1
    order = [("A", "B") if i % 2 == 0 else ("B", "A") for i in range(n)]
    job = dict(case=args.case, manifest=str(manifest), phase=args.phase, warms=args.warms)
    plan = dict(job=job, options=options_for(args.case), gpu=args.gpu, aa=args.aa,
                arms={k: str(v) for k, v in roots.items()}, order=order)
    if args.phase == "plan":
        print(json.dumps(plan, indent=2))
        return
    if not manifest.is_file() or args.out is None:
        parser.error("This phase requires the frozen fixture.json on disk and --out")
    args.out.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=args.case+"-"+args.phase+"-", dir=args.out.resolve()))
    print(folder, flush=True)
    write(folder/"plan.json", plan)
    write(folder/"job.json", job)
    sys.path.insert(0, str(roots["A"]))
    from tests.validation.dev.cupy_rsoxs_optimization.run_speed_tickets import observe_memory
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID",
               PYTHONDONTWRITEBYTECODE="1")
    summary = {"primary_seconds": [], "first_seconds": []}
    for i, arms in enumerate(order, start=1):
        reports, paths = {}, {}
        for arm in arms:
            label = f"r{i:02d}-{arm}"
            out = folder / (label+".json")
            observe = observe_memory(folder, label, args.gpu, args.poll_ms/1000) if args.phase == "memory" else nullcontext()
            with observe as measurement, out.with_suffix(".log").open("w") as log:
                subprocess.run([sys.executable, str(HERE), "--worker", str(folder/"job.json"),
                                str(roots[arm]), str(out)], cwd=roots[arm], env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads(out.read_text())
            if args.phase == "memory":
                peak = measurement["process_peaks_mib"].get(str(report["pid"]))
                if peak is None:
                    raise RuntimeError(f"No NVML sample for {label}; see {folder}")
                report.update(peak_gpu_mib=peak, nvml=measurement)
                write(out, report)
            reports[arm], paths[arm] = report, out
        a, b = reports["A"], reports["B"]
        for key in ("fixture_hashes", "options", "energies", "gpu_uuid", "cupy", "cuda", "driver"):
            if a[key] != b[key]:
                raise RuntimeError(f"Pair {i} differs in {key}")
        if args.phase == "time":
            summary["primary_seconds"].append({arm: median(r["warm_seconds"]) for arm, r in reports.items()})
            summary["first_seconds"].append({arm: r["first_seconds"] for arm, r in reports.items()})
        elif args.phase == "check":
            from tests.validation.dev.cupy_rsoxs_qmax.run_public_pair import load_sample
            from tests.validation.dev.cupy_rsoxs_qmax.qmax_reference import compare
            # Same cutoff in both arms. Check every bit of the actual returned products.
            summary["comparison"] = compare(*(load_sample(paths[k].with_suffix(".npz")) for k in ("A", "B")))
        elif args.phase == "stages":
            if a["timings"]["execution"] != b["timings"]["execution"]:
                raise RuntimeError(f"Execution topology differs; inspect {paths['A']} and {paths['B']}")
            summary["execution"] = a["timings"]["execution"]
        else:
            summary["memory"] = {arm: {k: r[k] for k in ("peak_gpu_mib", "peak_host_mib")}
                                 for arm, r in reports.items()}
        write(folder/"summary.json", summary)


if __name__ == "__main__":
    main()
