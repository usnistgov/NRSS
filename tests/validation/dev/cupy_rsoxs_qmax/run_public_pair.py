"""Time and measure complete and q-limited public NRSS runs in isolated workers."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import socket
import statistics
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[4]
THIS = Path(__file__).resolve()
ROUTES = ("native", "z-sum", "z-residues", "xyz")


def options_for(case, limited):
    options = {"result_layout": case["layout"], "result_residency": case["residency"],
               "result_chunk_size": case["chunk"], "energy_progress_bar": False}
    if case["layout"] != "detector":
        options["result_q_grid_mode"] = case["grid"]
    route, target = case["route"], case["target"]
    if route == "xyz":
        options["virtual_3d_pad_to"] = target
    elif route.startswith("z-"):
        options.update(virtual_z_pad_to=target[0], virtual_z_method=route[2:])
    if route != "native" and case["fill"] is not None:
        options["virtual_pad_material"] = case["fill"]
    if limited:
        options["q_max"] = case["q_max"]
    return options


def generic_morphology(case, options):
    import numpy as np
    from NRSS import SFieldMode
    from NRSS.morphology import Material, Morphology
    from tests.validation.lib.virtual_z_padding import explicit_padded_materials

    energies, angles = case["energies"], case["rotation"]
    z, y, x = [np.linspace(-1, 1, n, dtype=np.float32) for n in case["source"]]
    z, y, x = z[:, None, None], y[None, :, None], x[None, None, :]
    fraction = 0.5 + 0.22 * np.sin(3.1*x + y*z) + 0.1 * np.cos(2.3*z - y)
    def optics(energy, oriented):
        phase = (energy - 285.) * 0.23
        delta = 0.001 + 0.00018 * math.sin(phase)
        beta = 0.00021 + 0.00006 * math.cos(phase)
        return ([1.3*delta, 0.8*beta, 0.7*delta, 1.1*beta] if oriented else [delta, beta, delta, beta])
    materials = {
        1: Material(1, Vfrac=fraction, S=SFieldMode.ISOTROPIC, energies=energies,
                    opt_constants={e: optics(e, False) for e in energies}),
        2: Material(2, Vfrac=1-fraction, S=0.5+0.2*np.cos(2*z+x-y),
                    theta=1.0+0.3*np.sin(y+x*z), psi=2.8+1.4*np.sin(2*y-z+x),
                    energies=energies, opt_constants={e: optics(e, True) for e in energies}),
    }
    target = tuple(case["target"])
    if case["route"] == "native":
        materials = explicit_padded_materials(materials, target, fill_material=case["fill"])
    elif case["route"].startswith("z-"):
        materials = explicit_padded_materials(materials, (case["source"][0], *target[1:]), fill_material=case["fill"])
    config = dict(Morphology.config_default, Energies=energies, EAngleRotation=angles, CaseType=0,
                  MorphologyType=0, AlgorithmType=0, EwaldsInterpolation=1, ReferenceFrame=1,
                  WindowingType=case["window"], RotMask=case["rot_mask"])
    return Morphology(len(materials), materials=materials, PhysSize=2.5, config=config,
                      backend="cupy-rsoxs", backend_options=options,
                      resident_mode="host", input_policy="strict", ownership_policy="borrow")


def json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def source_fingerprint(root):
    digest = hashlib.sha256()
    for path in sorted((root / "src/NRSS").rglob("*")):
        if path.is_file() and path.suffix in {".py", ".cu", ".cuh", ".cpp", ".h"}:
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def save_sample(product, path):
    import numpy as np
    import xarray as xr

    dataarray = isinstance(product, xr.DataArray)
    manifest = {"attributes": product.attrs, "array": dataarray, "name": product.name if dataarray else None}
    arrays = {}
    for key, items in (("coords", product.coords), ("fields", {"value": product} if dataarray else product.data_vars)):
        manifest[key] = {}
        for name, item in items.items():
            store = f"arr{len(arrays)}"
            manifest[key][name] = [store, list(item.dims), item.attrs]
            arrays[store] = np.asarray(item)
    np.savez(path, manifest=json.dumps(manifest, default=json_value), **arrays)


def load_sample(path):
    import numpy as np
    import xarray as xr

    with np.load(path, allow_pickle=False) as db:
        info = json.loads(str(db["manifest"]))
        def unpack(key):
            return {name: (dims, db[storage], attrs) for name, (storage, dims, attrs) in info[key].items()}
        coords, fields = unpack("coords"), unpack("fields")
    ds = xr.Dataset(fields, coords=coords, attrs=info["attributes"])
    if not info["array"]:
        return ds
    out = ds["value"].rename(info["name"])
    out.attrs = info["attributes"]
    return out


def worker(case, root, output, limited, phase):
    sys.path[:0] = [str(root / "src"), str(root)]
    os.environ["NRSS_SPEED_CHECKOUT"] = str(root)
    import cupy as cp
    import NRSS
    from qmax_reference import select_product

    if root not in Path(NRSS.__file__).resolve().parents:
        raise RuntimeError(f"NRSS loaded from {NRSS.__file__}, expected {root}")
    options = options_for(case, limited)
    morph = generic_morphology(case, options)
    if list(morph.Energies) != case["energies"] or list(morph.EAngleRotation) != case["rotation"]:
        raise ValueError("Morphology changed the requested energy list or E-angle scan")
    actual_shape = tuple(morph.NumZYX)
    logical = (tuple(options["virtual_3d_pad_to"]) if "virtual_3d_pad_to" in options else
               (options.get("virtual_z_pad_to", actual_shape[0]), *actual_shape[1:]))
    if logical != tuple(case["target"]):
        raise ValueError(f"Fixture logical shape {logical} differs from requested {case['target']}")
    for key in ("q_max", "result_layout", "result_q_grid_mode", "virtual_z_pad_to", "virtual_z_method",
                "virtual_3d_pad_to", "virtual_pad_material"):
        expected = options.get(key)
        if key == "result_q_grid_mode" and expected is None:
            continue
        value = morph.backend_options.get(key)
        if key == "virtual_3d_pad_to" and expected is not None and value is not None:
            value, expected = tuple(value), tuple(expected)
        if value != expected:
            raise ValueError(f"Morphology changed option {key}: {value!r} vs {expected!r}")
    device = cp.cuda.Device()
    gpu = cp.cuda.runtime.getDeviceProperties(device.id)
    name = gpu["name"].decode() if isinstance(gpu["name"], bytes) else str(gpu["name"])
    report = {"nrss_import": str(NRSS.__file__), "gpu": name, "cupy": cp.__version__, "host": socket.gethostname(),
              "driver": cp.cuda.runtime.driverGetVersion(), "runtime": cp.cuda.runtime.runtimeGetVersion(),
              "pid": os.getpid(), "checkout": str(root), "source_shape": actual_shape,
              "logical_shape": logical, "voxel_nm": float(morph.PhysSize), "options": options,
              "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "source_sha256": source_fingerprint(root)}
    try:
        if phase == "diagnose":
            morph._set_private_backend_timing_segments(("energy",))
        times = []
        def host_peak_mib():
            # Linux ru_maxrss is KiB. On macOS the field is in bytes.
            scale = 1024**2 if sys.platform == "darwin" else 1024
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale
        report["before_run_peak_host_mib"] = host_peak_mib()
        for _ in range(case["warms"] + 1 if phase in ("time", "memory") else 1):
            device.synchronize()
            start = time.perf_counter()
            result = morph.run(stdout=False, stderr=False, return_xarray=False)
            device.synchronize()
            times.append(time.perf_counter() - start)
            values = result.to_backend_array()
            report.update(return_shape=list(values.shape), return_dtype=str(values.dtype), return_bytes=int(values.nbytes))
            if case["residency"] == "device" and not isinstance(values, cp.ndarray):
                raise RuntimeError("Expected a GPU-resident public result")
            if case["residency"] == "host" and isinstance(values, cp.ndarray):
                raise RuntimeError("Expected a host-resident public result")
            if phase == "check":
                product = result.to_xarray()
                selected = product if limited else select_product(product, case["q_max"])
                save_sample(selected, output.with_suffix(".npz"))
            del values
            del result
        report["first_run_s"] = times[0]
        if phase == "time":
            report["warm_run_s"] = times[1:]
        if phase == "memory":
            report["peak_host_mib"] = host_peak_mib()
            report["pool_used_end_mib"] = cp.get_default_memory_pool().used_bytes() / 2**20
            report["pool_reserved_end_mib"] = cp.get_default_memory_pool().total_bytes() / 2**20
        if phase == "diagnose":
            report["execution"] = morph.backend_timings["execution"]
            if report["execution"]["basis_builds"] != 1:
                raise ValueError("Expected one compact basis build in this fixture")
        output.write_text(json.dumps(report, indent=2, default=json_value))
    finally:
        morph.release_runtime()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=5, metavar=("CASE", "ROOT", "OUT", "LIMITED", "PHASE"), help=argparse.SUPPRESS)
    parser.add_argument("--parent-root", type=Path, default=ROOT)
    parser.add_argument("--candidate-root", type=Path, default=ROOT)
    parser.add_argument("--phase", choices=("plan", "time", "check", "diagnose", "memory"), default="plan")
    parser.add_argument("--aa", action="store_true", help="Run the same unbounded parent twice")
    parser.add_argument("--route", nargs="+", choices=ROUTES, default=list(ROUTES))
    parser.add_argument("--layout", choices=("i_a", "i_only", "integrated", "i_para_i_perp", "detector"), default="detector")
    parser.add_argument("--residency", choices=("host", "device"), default="host")
    parser.add_argument("--grid", choices=("intersection", "union"), default="intersection")
    parser.add_argument("--q-max", type=float, default=0.8)
    parser.add_argument("--uncapped", action="store_true", help="Omit q_max in both arms and compare entire outputs")
    parser.add_argument("--energy-count", type=int, choices=(1, 3, 20, 101), default=3)
    parser.add_argument("--energies-json", type=Path, help="Exact ordered JSON list in eV, overrides --energy-count")
    parser.add_argument("--angles", type=int, choices=(1, 12), default=12)
    parser.add_argument("--target", type=int, nargs=3, default=[36, 64, 72], metavar=("Z", "Y", "X"))
    parser.add_argument("--source", type=int, nargs=3, default=[9, 33, 35], metavar=("Z", "Y", "X"))
    parser.add_argument("--fill", default="2", help="Registered fill ID, or vacuum")
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--window", type=int, choices=(0, 1), default=0)
    parser.add_argument("--rot-mask", type=int, choices=(0, 1), default=0)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
    parser.add_argument("--warms", type=int, default=5)
    parser.add_argument("--pairs", type=int, default=2, help="Fresh matched worker pairs for timing, alternates AB/BA")
    parser.add_argument("--poll-ms", type=float, default=10., help="Memory-worker NVML sampling interval in ms")
    parser.add_argument("--out", type=Path, default=ROOT / "test-reports/qmax")
    args = parser.parse_args()
    if args.worker:
        src, root, out, limited, action = args.worker
        worker(json.loads(Path(src).read_text()), Path(root), Path(out), limited == "yes", action)
        return
    if (not math.isfinite(args.q_max) or args.q_max < 0 or min(*args.source, *args.target, args.chunk, args.warms, args.pairs) < 1
            or not math.isfinite(args.poll_ms) or args.poll_ms <= 0):
        parser.error("Use a finite nonnegative cutoff and positive sizes, chunk, repeats and pairs")
    if "," in args.gpu or (args.layout == "detector" and args.grid == "union"):
        parser.error("Choose one physical GPU and use a supported layout/grid combination")
    if any(t < s for t, s in zip(args.target, args.source)):
        parser.error("Logical target must contain the occupied shape")
    if "z-residues" in args.route and args.target[0] % args.source[0]:
        parser.error("Z-residues needs a target depth divisible by the occupied depth")
    if args.energies_json:
        energies = json.loads(args.energies_json.read_text())
        if not isinstance(energies, list) or not energies or any(isinstance(e, bool) or not isinstance(e, (float, int))
                or not math.isfinite(e) or e <= 0 for e in energies):
            parser.error("Provide a JSON list of finite positive energies")
    elif args.energy_count == 1:
        energies = [287.55]
    else:
        energies = [(28255 + 10*round(i*100/(args.energy_count-1))) / 100 for i in range(args.energy_count)]
    try:
        fill = None if args.fill == "vacuum" else int(args.fill)
        if fill is not None and fill not in (1, 2):
            raise ValueError
    except ValueError:
        parser.error("Choose vacuum, isotropic material 1, or oriented material 2")
    arms = {"A": (args.parent_root.resolve(), False),
            "B": ((args.parent_root if args.aa else args.candidate_root).resolve(), not (args.aa or args.uncapped))}
    for root, _ in arms.values():
        if not (root / "src/NRSS/backends/cupy_rsoxs.py").is_file():
            parser.error(f"Missing NRSS source in {root}")
    cases = [dict(route=r, layout=args.layout, grid=args.grid, q_max=None if args.uncapped else args.q_max, target=args.target,
                  fill=fill, energies=energies, chunk=args.chunk, rotation=([0, 15, 165] if args.angles == 12 else [0, 0, 0]),
                  warms=args.warms, window=args.window, rot_mask=args.rot_mask, source=args.source,
                  residency=args.residency) for r in args.route]
    plan = {"cases": cases, "gpu": args.gpu, "phase": args.phase, "aa": args.aa, "poll_ms": args.poll_ms,
            "arms": {arm: {"root": str(root), "limited": opt} for arm, (root, opt) in arms.items()}}
    if args.phase == "plan":
        print(json.dumps(plan, indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=f"{args.phase}-", dir=args.out.resolve()))
    (folder / "plan.json").write_text(json.dumps(plan, indent=2))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONDONTWRITEBYTECODE="1")
    print(folder, flush=True)
    summary = {}
    for case in cases:
        route = case["route"]
        case_file = folder / f"{route}.json"
        case_file.write_text(json.dumps(case))
        ratios = []
        for index in range(args.pairs if args.phase == "time" else 1):
            paths = {}
            for arm in (("A", "B") if index % 2 == 0 else ("B", "A")):
                root, limited = arms[arm]
                output = folder / f"{route}-{index:02d}-{arm}.json"
                paths[arm] = output
                observer = nullcontext()
                if args.phase == "memory":
                    sys.path.insert(0, str(ROOT))
                    from tests.validation.dev.cupy_rsoxs_optimization.run_speed_tickets import observe_memory
                    observer = observe_memory(folder, output.stem, args.gpu, args.poll_ms / 1000)
                with observer as memory, output.with_suffix(".log").open("w") as log:
                    subprocess.run([sys.executable, str(THIS), "--worker", str(case_file), str(root),
                                    str(output), "yes" if limited else "no", args.phase],
                                   env=env, cwd=root, check=True, stdout=log, stderr=subprocess.STDOUT)
                if args.phase == "memory":
                    report = json.loads(output.read_text())
                    peak = memory["process_peaks_mib"].get(str(report["pid"]))
                    if peak is None:
                        raise RuntimeError(f"No NVML sample for worker PID {report['pid']}. See {output}")
                    report.update(peak_gpu_mib=peak, nvml=memory)
                    output.write_text(json.dumps(report, indent=2))
            reports = {k: json.loads(v.read_text()) for k, v in paths.items()}
            for key in ("source_shape", "logical_shape", "voxel_nm", "return_dtype", "gpu"):
                if reports["A"][key] != reports["B"][key]:
                    raise AssertionError(f"Unmatched {key} in {route}")
            if args.phase == "check":
                from qmax_reference import compare
                summary[route] = compare(*(load_sample(paths[k].with_suffix(".npz")) for k in ("A", "B")),
                                         q_max=case["q_max"])
            elif args.phase == "time":
                med = {k: statistics.median(v["warm_run_s"]) for k, v in reports.items()}
                ratios.append(med["A"] / med["B"])
                summary[route] = {"paired_warm_speedups": ratios, "median": statistics.median(ratios),
                                  "latest_warm_medians_s": med,
                                  "returned_bytes": {k: v["return_bytes"] for k, v in reports.items()}}
            elif args.phase == "memory":
                keys = ("peak_host_mib", "peak_gpu_mib", "before_run_peak_host_mib", "return_bytes")
                summary[route] = {k: {key: v[key] for key in keys} for k, v in reports.items()}
                for key in ("peak_host_mib", "peak_gpu_mib"):
                    old, new = reports["A"][key], reports["B"][key]
                    summary[route][key + "_saved"] = {"MiB": old-new, "percent": 100*(old-new)/old if old else None}
            else:
                summary[route] = {k: v["execution"] for k, v in reports.items()}
        (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
