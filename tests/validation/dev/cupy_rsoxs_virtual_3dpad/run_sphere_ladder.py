"""Qualification driver for the ZP08 full-detector sphere sweep. Default action prints the plan."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

from sphere_reference import (CROP, D, ENERGY, K, MEDIUM, PARTICLE, RADIUS, SIZES, VACUUM,
                              assess_full_reference, axis, branch_intensity_scale, compare_panel,
                              compare_voxel_panels, continuum, discrete_anchors,
                              fingerprint, forward_scale, sphere_fraction)

ROOT = Path(__file__).resolve().parents[4]
THIS = Path(__file__).resolve()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def plan(sizes):
    return [{"M": m, "shape_zyx": [m]*3, "panel_yx": [m,m], "grid_pixels": m*m,
             "dq_nm_inv": 2*np.pi/(m*D), "one_output_fp32_MiB": 4*m*m/2**20,
             "one_literal_fp32_volume_GiB": 4*m**3/2**30} for m in sizes]


def preflight(sizes, fill_id):
    fraction = sphere_fraction()
    rows = []
    for m in sizes:
        pixels, reference = discrete_anchors(fraction,m,fill_id)
        qy,qx = (axis(m)[t] for t in pixels.T)
        selected = (qy!=0) | (qx!=0)
        ideal = continuum(qx[selected],qy[selected],fill_id)
        rows.append({"M": m, "anchor_count": len(pixels),
                     "voxel_vs_continuum_anchor_max_abs_over_I0":
                     float(np.max(abs(reference[selected]-ideal))/forward_scale(fill_id))})
    return {"kind":"CPU voxel anchors and continuum only", "sphere_sha256":fingerprint(fraction),
            "volume_voxels":float(fraction.sum(dtype=np.float64)), "fill":fill_id, "sizes":rows}


def occupied_fraction(fraction, size, arm):
    if size == CROP or arm == "xyz":
        return fraction
    depth = CROP if arm == "film" else size
    shape = (depth, size, size)
    offsets = [(m - CROP) // 2 for m in shape]
    occupied = np.zeros(shape, dtype=fraction.dtype)
    occupied[tuple(slice(o, o + CROP) for o in offsets)] = fraction
    return occupied


def make_morph(fraction, size, fill_id, arm, z_method):
    from NRSS import SFieldMode
    from NRSS.morphology import Material, Morphology

    fraction = occupied_fraction(fraction, size, arm)
    material_list = {}
    for number, volume, optics in ((1, fraction, PARTICLE),
                                  (2, 1-fraction, MEDIUM if fill_id == 2 else VACUUM)):
        material_list[number] = Material(number, Vfrac=volume, S=SFieldMode.ISOTROPIC,
                                         energies=[ENERGY], opt_constants={ENERGY:list(optics)})
    options = {"execution_path":"tensor_coeff", "energy_basis_mode":"compact_r2c",
               "energy_progress_bar":False}
    # Native control deliberately runs on the parent as well.
    if size > CROP and arm == "xyz":
        options["virtual_3d_pad_to"] = (size,size,size)
        options["virtual_pad_material"] = fill_id
    elif size > CROP and arm == "film":
        options.update(virtual_z_pad_to=size, virtual_z_method=z_method)
        if fill_id is not None:
            options["virtual_pad_material"] = fill_id
    config = dict(Morphology.config_default, Energies=[ENERGY], EAngleRotation=[0.,0.,0.],
                  RotMask=0, WindowingType=0)
    return Morphology(2, materials=material_list, PhysSize=D, config=config, backend="cupy-rsoxs",
                      backend_options=options, resident_mode="host", input_policy="strict", ownership_policy="borrow")


def worker(size, fill_id, action, repeats, folder, args):
    import cupy as cp
    import NRSS

    checkout = args.nrss_root.resolve()
    if checkout not in Path(NRSS.__file__).resolve().parents:
        raise RuntimeError(f"Worker imported a different checkout: {NRSS.__file__}")
    fraction = sphere_fraction()
    morph = make_morph(fraction,size,fill_id,args.arm,args.z_method)
    profile = getattr(args, "performance_profile", "legacy")
    if (checkout / "src/NRSS/backends/_cupy_xyz_policy.py").is_file():
        from NRSS.backends._cupy_xyz_policy import XYZ_PERFORMANCE_PROFILE_KEY

        morph._backend_runtime_state[XYZ_PERFORMANCE_PROFILE_KEY] = profile
    elif profile != "legacy":
        raise ValueError(f"{checkout} does not support the {profile} XYZ profile.")
    if args.endpoint_plan != "legacy":
        from NRSS.backends._cupy_xyz_endpoints import XYZ_ENDPOINT_PLAN_KEY

        morph._backend_runtime_state[XYZ_ENDPOINT_PLAN_KEY] = args.endpoint_plan
    if getattr(args, "prune_zero_fields", False):
        from NRSS.backends._cupy_energy_basis import XYZ_PRUNE_ZERO_FIELDS_KEY

        morph._backend_runtime_state[XYZ_PRUNE_ZERO_FIELDS_KEY] = True
    if getattr(args, "factored_xy", False):
        from NRSS.backends._cupy_virtual_xyz import XYZ_FACTORED_XY_KEY

        morph._backend_runtime_state[XYZ_FACTORED_XY_KEY] = True
    device = cp.cuda.Device()
    start_info = cp.cuda.runtime.getDeviceProperties(device.id)
    result = {"M":size, "fill_id":fill_id, "arm":args.arm, "action":action, "worker_pid":os.getpid(),
              "sphere_sha256":fingerprint(fraction), "git":subprocess.check_output(
                  ["git","rev-parse","HEAD"],cwd=checkout,text=True).strip(),
              "dirty":bool(subprocess.check_output(["git","status","--porcelain"],cwd=checkout)),
              "endpoint_plan":args.endpoint_plan,
              "performance_profile":profile,
              "gpu_name":start_info["name"].decode(), "visible_device":os.environ.get("CUDA_VISIBLE_DEVICES"),
              "gpu_memory_bytes":int(start_info["totalGlobalMem"]), "cuda_driver":cp.cuda.runtime.driverGetVersion(),
              "cupy":cp.__version__, "nrss":str(NRSS.__file__),
              "cuda_runtime":cp.cuda.runtime.runtimeGetVersion(), "python":platform.python_version()}
    try:
        if action == "accuracy":
            segments = ("energy",)
            if args.stages:
                from tests.validation.dev.cupy_rsoxs_optimization.run_speed_tickets import STAGES
                segments = tuple(STAGES)
            morph._set_private_backend_timing_segments(segments)
        durations = []
        pool_snapshots = []
        for attempt in range(repeats+1 if action in ("timing", "memory") else 1):
            device.synchronize()
            start = time.perf_counter()
            data = morph.run(stdout=False,stderr=False,return_xarray=True)
            device.synchronize()
            durations.append(time.perf_counter()-start)
            if data.dims != ("energy","qy","qx") or data.shape != (1,size,size):
                raise AssertionError("Expected a full logical detector panel")
            if action == "memory":
                pool = cp.get_default_memory_pool()
                pool_snapshots.append({"run":attempt, "used_bytes":pool.used_bytes(),
                                       "reserved_bytes":pool.total_bytes()})
            if action == "accuracy":
                np.testing.assert_array_equal(data.qx.values,axis(size))
                np.testing.assert_array_equal(data.qy.values,axis(size))
                np.testing.assert_array_equal(data.energy.values,[ENERGY])
                if data.attrs["detector_grid_convention"]!="fft_bins_v1" or data.attrs["z_dim"]!=size:
                    raise AssertionError("Unexpected public detector metadata")
                routes = list(morph.backend_timings["execution"]["routes"])
                if size==CROP or args.arm=="literal":
                    expected = routes == ["compact_r2c"]
                elif args.arm=="film":
                    expected = routes == [f"virtual_z_{args.z_method}"]
                else:
                    expected = len(routes)==1 and routes[0].startswith("virtual_3d")
                if not expected:
                    raise AssertionError(f"Unexpected executed routes {routes}")
                panel = data.values[0]
                metrics = compare_panel(panel,size,fill_id)
                pixels, discrete = discrete_anchors(fraction,size,fill_id)
                observed = panel[tuple(pixels.T)]
                optical_scale = branch_intensity_scale(fill_id)
                nominal_error = float(np.max(abs(discrete-observed))/forward_scale(fill_id))
                discrete *= optical_scale
                error = float(np.max(abs(discrete-observed))/forward_scale(fill_id))
                metrics.update(anchor_max_abs_over_I0=error, routes=routes, fp32_optical_power_scale=optical_scale,
                               nominal_voxel_anchor_max_abs_over_I0=nominal_error,
                               anchors=[{"iy":int(p[0]),"ix":int(p[1]),"sim":float(s),"voxel":float(v)}
                                        for p,s,v in zip(pixels,observed,discrete)])
                failures = []
                if error > 2e-5:
                    failures.append(f"voxel anchor error {error:.4g}")
                if args.verify_full_voxel:
                    from residue_pilot import full_detector

                    voxel = full_detector(fraction, size, fill_id)
                    voxel *= optical_scale
                    differences = compare_voxel_panels(panel, voxel, size, fill_id)
                    voxel_metrics = compare_panel(voxel, size, fill_id)
                    metrics.update(full_voxel=differences, cpu_reference=voxel_metrics)
                    failures.extend(assess_full_reference(metrics, voxel_metrics, differences, size))
                result.update(metrics=metrics, failures=failures, backend_diagnostics=morph.backend_timings)
                np.save(folder/"positive_qy.npy", panel[size//2:,size//2])
                if args.save_panel:
                    np.save(folder/"detector.npy", panel)
                write_json(folder/"accuracy.json",result)
                if failures:
                    raise AssertionError(", ".join(failures))
            del data
        if action=="timing":
            warm = durations[1:]
            median = statistics.median(warm)
            result.update(first_s=durations[0],warm_s=warm,warm_median_s=median,
                          warm_mad_s=statistics.median(abs(t-median) for t in warm))
        if action=="memory":
            result["cupy_pool_after_runs"] = pool_snapshots
        result["peak_host_rss_MiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024  # Linux
        write_json(folder/(action+".json"),result)
    finally:
        morph.release_runtime()


def run_campaign(args):
    from tests.validation.dev.cupy_rsoxs_optimization.run_speed_tickets import observe_memory

    gpu = args.gpu or os.environ.get("CUDA_VISIBLE_DEVICES","0").split(",")[0]
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpu)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-"),dir=out))
    rows = []
    print(folder, flush=True)
    for size in args.sizes:
        rundir = folder/f"M{size}"
        rundir.mkdir()
        row = {"M":size, "status":"pending", "fill":args.fill, "arm":args.arm}
        rows.append(row)
        try:
            for action in ("timing","memory","accuracy"):
                cmd = [sys.executable,str(THIS),"--worker",action,"--sizes",str(size),"--fill",args.fill,
                       "--repeats",str(args.repeats),"--out",str(rundir),"--arm",args.arm,
                       "--nrss-root",str(args.nrss_root.resolve()),"--z-method",args.z_method,
                       "--endpoint-plan",args.endpoint_plan,"--performance-profile",args.performance_profile]
                if action == "accuracy" and args.verify_full_voxel:
                    cmd.append("--verify-full-voxel")
                if action == "accuracy" and args.save_panel:
                    cmd.append("--save-panel")
                if action == "accuracy" and args.stages:
                    cmd.append("--stages")
                if args.prune_zero_fields:
                    cmd.append("--prune-zero-fields")
                if args.factored_xy:
                    cmd.append("--factored-xy")
                observer = observe_memory(rundir,"nvml",gpu,.010) if action=="memory" else nullcontext()
                with observer as memory, (rundir/(action+".log")).open("w") as log:
                    subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                saved = json.loads((rundir/(action+".json")).read_text())
                if action=="memory":
                    peak = memory["process_peaks_mib"].get(str(saved["worker_pid"]))
                    if peak is None:
                        raise RuntimeError("Missing worker PID peak in NVML trace")
                    saved.update(nvml=memory,peak_worker_MiB=peak)
                    write_json(rundir/"memory.json",saved)
                    row.update(peak_worker_MiB=peak,peak_host_rss_MiB=saved["peak_host_rss_MiB"])
                elif action=="timing":
                    row.update(first_s=saved["first_s"], warm_median_s=saved["warm_median_s"],
                               valid_pixels_per_s=None)
                else:
                    metrics=saved["metrics"]
                    row.update(valid_pixels_per_s=metrics["valid_pixels"]/row["warm_median_s"],
                               continuum_l2=metrics["bands"]["all_non_dc"]["relative_l2"],
                               voxel_anchor_error=metrics["anchor_max_abs_over_I0"])
            row["status"]="measured"
        except Exception as error:
            row.update(status="failed",detail=str(error))
            print(f"M={size} failed, see {rundir}",file=sys.stderr,flush=True)
        write_json(folder/"summary.json",rows)
    if any(row["status"]!="measured" for row in rows):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--plan",action="store_true")
    group.add_argument("--preflight",action="store_true")
    group.add_argument("--run",action="store_true")
    group.add_argument("--worker",choices=("timing","memory","accuracy"),help=argparse.SUPPRESS)
    parser.add_argument("--fill",choices=("vacuum","2"),default="vacuum")
    parser.add_argument("--sizes",nargs="+",type=int,choices=SIZES)
    parser.add_argument("--arm",choices=("xyz","literal","film"),default="xyz",
                        help="Compact XYZ, explicitly occupied cube, or laterally explicit virtual-Z control")
    parser.add_argument("--nrss-root",type=Path,default=ROOT,help="Source checkout for the NRSS worker")
    parser.add_argument("--z-method",choices=("sum","residues"),default="sum",help="Method for the film arm")
    parser.add_argument("--save-panel",action="store_true",help="Save the complete host panel in the accuracy pass")
    parser.add_argument("--stages",action="store_true",help="Record the speed-harness stage timers in the diagnostic pass")
    parser.add_argument("--endpoint-plan", choices=("legacy", "bitmap", "spans"), default="legacy")
    parser.add_argument("--performance-profile", choices=("legacy", "energy", "resolution", "auto"),
                        default="legacy")
    parser.add_argument("--prune-zero-fields", action="store_true",
                        help="Screen exact-zero optical basis pruning on the XYZ route")
    parser.add_argument("--factored-xy", action="store_true",
                        help="Screen split X/Y FFTs on required RFFT columns")
    parser.add_argument("--gpu",help="Physical GPU index or UUID")
    parser.add_argument("--verify-full-voxel", action="store_true",
                        help="Compare all pixels with the independent CPU transform in the accuracy worker")
    parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--out",type=Path,default=ROOT/"test-reports"/"virtual-3dpad"/"sphere")
    args = parser.parse_args()
    if args.run and args.arm != "xyz" and args.sizes is None:
        parser.error("Choose --sizes for controls with physically occupied lateral arrays or cubes.")
    args.sizes = args.sizes or SIZES
    if args.repeats<1 or len(set(args.sizes))!=len(args.sizes):
        parser.error("Use positive repeats and unique sizes.")
    fill_id = None if args.fill=="vacuum" else 2
    if args.worker:
        if len(args.sizes)!=1:
            parser.error("A worker takes one size.")
        sys.path.insert(0,str(args.nrss_root.resolve()/"src"))
        worker(args.sizes[0],fill_id,args.worker,args.repeats,args.out,args)
    elif args.run:
        run_campaign(args)
    else:
        output = preflight(args.sizes,fill_id) if args.preflight else plan(args.sizes)
        print(json.dumps(output,indent=2,allow_nan=False))


if __name__=="__main__":
    sys.path.insert(0,str(ROOT))
    sys.path.insert(0,str(ROOT/"src"))
    main()
