"""Time the frozen ZP09 vesicle using the public Morphology.run boundary."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import statistics
import subprocess
import sys
from time import perf_counter

import numpy as np


HARNESS_ROOT = Path(__file__).resolve().parents[4]
OCCUPIED_SHAPE = (52, 52, 52)
TARGET_SHAPE = (256, 1024, 1024)
PITCH_NM = 1.5


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fixture(manifest, count):
    fixture = json.loads(manifest.read_text())
    geometry = fixture["geometry"]
    assert geometry["pitch_nm"] == PITCH_NM
    assert geometry["EAngleRotation_start_increment_end_deg"] == [0, 0, 0]
    assert tuple(fixture["target_zyx"]) == TARGET_SHAPE
    assert fixture["water_material_id"] == 3

    fields_path = manifest.parent / fixture["occupied_fields"]
    fields_hash = sha256(fields_path)
    assert fields_hash == fixture["occupied_fields_sha256"]
    for source in geometry["optical_constants"].values():
        assert sha256(manifest.parent / source["path"]) == source["sha256"], source["path"]

    indices = fixture["energy_index_sets"][str(count)]
    assert len(indices) == count and sorted(set(indices)) == indices
    optics_path = manifest.with_name("energy-optics.npz")
    with np.load(optics_path) as arrays:
        energies = arrays["energies"]
        optics = arrays["optics"]
    assert optics.shape == (3, 101, 4)
    assert np.isfinite(optics).all()
    np.testing.assert_array_equal(energies, np.round(282.55 + np.arange(101) * 0.1, 2))
    np.testing.assert_array_equal(energies, fixture["energies_eV"])
    with np.load(fields_path) as arrays:
        fields = {key: arrays[key] for key in arrays.files if key not in {"energies", "optics"}}
        anchor = np.flatnonzero(energies == 287.55).item()
        anchor_crop = np.flatnonzero(arrays["energies"] == energies[anchor]).item()
        np.testing.assert_array_equal(optics[:, anchor], arrays["optics"][:, anchor_crop])
    assert all(values.shape == OCCUPIED_SHAPE and values.dtype == np.float32
               for values in fields.values())
    hashes = {
        "manifest": sha256(manifest),
        "fields": fields_hash,
        "scan_optics": sha256(optics_path),
        "optical_sources": geometry["optical_constants"],
    }
    return fields, energies[indices].tolist(), optics[:, indices], indices, hashes


def panel_errors(panel, expected):
    finite = np.isfinite(expected)
    assert not np.isinf(panel).any() and not np.isinf(expected).any()
    np.testing.assert_array_equal(np.isfinite(panel), finite)
    if np.array_equal(panel[finite], expected[finite]):
        return 0.0, 0.0
    observed, reference = panel[finite].astype(np.float64), expected[finite].astype(np.float64)
    delta = observed - reference
    peak = np.max(np.abs(reference))
    return float(np.max(np.abs(delta)) / peak), float(np.linalg.norm(delta) / np.linalg.norm(reference))


def check_reduced_reference(result, reference_root):
    import xarray as xr

    path = reference_root / "virtual-101-chunk1-i-a.nc"
    actual = result.to_xarray()
    with xr.open_dataset(path) as stored:
        reference = stored.sel(energy=actual.energy)
        xr.testing.assert_identical(actual.coords.to_dataset(), reference.coords.to_dataset())
        assert actual.attrs["source_integrator"] == "cupy-rsoxs"
        errors = np.asarray([panel_errors(y, x) for y, x in zip(actual.I.values, reference.I.values)])
        assert errors.max() <= 2e-5, errors.max(axis=0)
        np.testing.assert_array_equal(np.isfinite(actual.A.values), np.isfinite(reference.A.values))
        difference = np.abs(actual.A - reference.A)
        fit_band = difference.sel(q=slice(0.06, 0.80))
        peak_a = float(difference.max())
        peak_band = float(fit_band.max())
        assert peak_band <= 2e-5, peak_band
        return {"max_abs_over_panel_peak": float(errors[:, 0].max()),
                "max_relative_l2": float(errors[:, 1].max()),
                "max_abs_a_all_q": peak_a, "max_abs_a_fit_band": peak_band, "reference": str(path)}


def check_reference(result, args, indices):
    if args.layout == "i_a":
        return check_reduced_reference(result, args.manifest.parent)
    detector = result.to_xarray()
    base = args.reference
    if base is not None:
        indices = np.arange(args.count)
    elif args.directions == 12:
        assert args.count == 20 and not args.permute
        base = args.manifest.parent / "virtual-20-chunk1-12angles"
        indices = np.arange(args.count)
    else:
        base = args.manifest.parent / "virtual-101-chunk1"
    data_path = base.with_name(base.name + "-detector.npy")
    axes_path = base.with_name(base.name + "-axes.npz")
    if args.capture_reference:
        np.save(data_path, detector.values)
        np.savez(axes_path, qx=detector.qx.values, qy=detector.qy.values, energy=detector.energy.values)
    with np.load(axes_path) as axes:
        np.testing.assert_array_equal(detector.energy.values, axes["energy"][indices])
        np.testing.assert_array_equal(detector.qx.values, axes["qx"])
        np.testing.assert_array_equal(detector.qy.values, axes["qy"])
    assert detector.attrs["phys_size_nm"] == PITCH_NM
    assert detector.attrs["z_dim"] == (args.target or TARGET_SHAPE)[0]
    reference = np.load(data_path, mmap_mode="r")
    errors = np.asarray([panel_errors(panel, reference[index]) for panel, index in zip(detector.values, indices)])
    assert errors.max() <= 2e-5, errors.max(axis=0)
    return {"exact_panels": int(np.count_nonzero(errors[:, 0] == 0)),
            "max_abs_over_panel_peak": float(errors[:, 0].max()), "max_relative_l2": float(errors[:, 1].max()),
            "reference": str(data_path)}


def benchmark(args):
    sys.path[:0] = [str(args.checkout / "src"), str(HARNESS_ROOT)]
    import cupy as cp
    import NRSS
    from NRSS import SFieldMode
    from NRSS.morphology import Material, Morphology
    from tests.validation.dev.cupy_rsoxs_optimization.run_speed_tickets import STAGES, checkout_info

    source = Path(NRSS.__file__).resolve()
    assert source == args.checkout / "src" / "NRSS" / "__init__.py", source
    fields, energies, optics, indices, hashes = load_fixture(args.manifest, args.count)
    target = tuple(args.target) if getattr(args, "target", None) else TARGET_SHAPE
    if args.permute:
        order = list(range(0, args.count, 2))[::-1] + list(range(1, args.count, 2))
        energies = [energies[i] for i in order]
        indices = [indices[i] for i in order]
        optics = optics[:, order]
    cp.cuda.Device(0).use()
    cp.cuda.runtime.free(0)
    options = {
        "energy_progress_bar": False,
        "virtual_3d_pad_to": target,
        "virtual_pad_material": 3,
        "result_layout": args.layout,
        "result_residency": args.residency,
        "result_chunk_size": args.chunk,
    }
    if args.path == "compact":
        options.update(execution_path="tensor_coeff", energy_basis_mode="compact_r2c")
    materials = {}
    for material_id, name in enumerate(("head", "tail", "water"), start=1):
        orientation = {"S": SFieldMode.ISOTROPIC}
        if name == "tail":
            orientation = {attr: fields[f"tail_{attr.lower()}"] for attr in ("S", "theta", "psi")}
        materials[material_id] = Material(
            material_id, name=name, Vfrac=fields[f"{name}_vfrac"], energies=energies,
            opt_constants=dict(zip(energies, optics[material_id - 1].tolist())),
            **orientation,
        )
    config = dict(
        Morphology.config_default, Energies=energies,
        EAngleRotation=[0, 0, 0] if args.directions == 1 else [0, 15, 165],
        RotMask=0, WindowingType=0,
    )
    morph = Morphology(
        len(materials), materials=materials, PhysSize=PITCH_NM, config=config,
        backend="cupy-rsoxs", input_policy="strict", ownership_policy="borrow",
        backend_options=options,
    )
    assert tuple(morph.NumZYX) == OCCUPIED_SHAPE
    profile = getattr(args, "performance_profile", "legacy")
    if (args.checkout / "src/NRSS/backends/_cupy_xyz_policy.py").is_file():
        from NRSS.backends._cupy_xyz_policy import XYZ_PERFORMANCE_PROFILE_KEY

        morph._backend_runtime_state[XYZ_PERFORMANCE_PROFILE_KEY] = profile
    elif profile != "legacy":
        raise ValueError(f"{args.checkout} does not support the {profile} XYZ profile.")
    if args.endpoint_plan != "legacy":
        from NRSS.backends._cupy_xyz_endpoints import XYZ_ENDPOINT_PLAN_KEY

        morph._backend_runtime_state[XYZ_ENDPOINT_PLAN_KEY] = args.endpoint_plan
    if args.tile_mode != "auto" and (args.checkout / "src/NRSS/backends/_cupy_energy_tiles.py").is_file():
        from NRSS.backends.cupy_rsoxs import XYZ_ENERGY_TILE_KEY

        morph._backend_runtime_state[XYZ_ENERGY_TILE_KEY] = (
            None if args.tile_mode == "scalar" else (args.tile_width, args.tile_mode == "share")
        )
    if getattr(args, "prune_zero_fields", False):
        from NRSS.backends._cupy_energy_basis import XYZ_PRUNE_ZERO_FIELDS_KEY

        morph._backend_runtime_state[XYZ_PRUNE_ZERO_FIELDS_KEY] = True
    if getattr(args, "factored_xy", False):
        from NRSS.backends._cupy_virtual_xyz import XYZ_FACTORED_XY_KEY

        morph._backend_runtime_state[XYZ_FACTORED_XY_KEY] = True
    stream = cp.cuda.get_current_stream()

    def timed_run():
        stream.synchronize()
        start = perf_counter()
        result = morph.run(stdout=False, stderr=False, return_xarray=False)
        stream.synchronize()
        elapsed = perf_counter() - start
        data = result.to_backend_array()
        assert isinstance(data, cp.ndarray if args.residency == "device" else np.ndarray)
        assert list(result.energies) == energies
        if args.layout == "detector":
            assert data.dtype == np.float32 and data.shape == (args.count, *target[1:])
            assert result.num_zyx == target
        else:
            assert data.dtype == np.float64 and data.ndim == 3 and data.shape[:2] == (args.count, 2)
        return elapsed, result

    try:
        if args.mode == "stages":
            preheat = [timed_run()[0] for _ in range(args.repeats + 1)]
            morph._set_private_backend_timing_segments(STAGES)
            elapsed, result = timed_run()
            timings = morph.backend_timings
            execution = timings["execution"]
            assert execution["routes"] == ["virtual_3d_sum"] and execution["basis_builds"] == 1
            if args.tile_mode in {"reload", "share"}:
                details = execution["energy_projector"]
                assert "fallback" not in details, details
                assert execution["compute_tile_size"] == details["width"] <= min(args.tile_width, args.count)
            measurement = {
                "preheat_seconds": preheat,
                "instrumented_seconds": elapsed,
                "timings": timings,
                "reference": check_reference(result, args, indices),
                "result_bytes": result.to_backend_array().nbytes,
            }
        else:
            first = timed_run()[0]
            warm = [timed_run()[0] for _ in range(args.repeats)]
            measurement = {
                "first_seconds": first,
                "warm_seconds": warm,
                "warm_median_seconds": statistics.median(warm),
            }
        properties = cp.cuda.runtime.getDeviceProperties(0)
        return {
            "mode": args.mode,
            "path": args.path,
            "endpoint_plan": args.endpoint_plan,
            "performance_profile": profile,
            "tile_mode": args.tile_mode,
            "requested_tile_width": args.tile_width,
            "nrss_module": str(source),
            "checkout": checkout_info(args.checkout),
            "harness": str(Path(__file__).resolve()),
            "fixture": str(args.manifest),
            "fixture_hashes": hashes,
            "pid": os.getpid(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "host": socket.getfqdn(),
            "gpu": properties["name"].decode(),
            "gpu_selection": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_total_mib": properties["totalGlobalMem"] / 1024**2,
            "python": sys.version.split()[0],
            "cupy": cp.__version__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver": cp.cuda.runtime.driverGetVersion(),
            "occupied_zyx": OCCUPIED_SHAPE,
            "target_zyx": target,
            "pitch_nm": PITCH_NM,
            "angle_schedule_start_increment_end_deg": config["EAngleRotation"],
            "energies_eV": energies,
            "backend_options": options,
            "resolved_backend_options": {
                key: morph.backend_options[key] for key in
                ("execution_path", "result_layout", "result_residency")
            },
            "configured_energy_basis_mode": morph._energy_basis_mode,
            "cupy_retained_pool_mib": cp.get_default_memory_pool().total_bytes() / 1024**2,
            "peak_host_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            **measurement,
        }
    finally:
        morph.release_runtime()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Frozen ZP09 fixture.json")
    parser.add_argument("--checkout", type=Path, required=True, help="NRSS source checkout")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--count", choices=(1, 20, 101), type=int, required=True)
    parser.add_argument("--path", choices=("default", "compact"), default="default")
    parser.add_argument("--mode", choices=("timing", "stages"), default="timing")
    parser.add_argument("--endpoint-plan", choices=("legacy", "bitmap", "spans"), default="legacy")
    parser.add_argument("--performance-profile", choices=("legacy", "energy", "resolution", "auto"),
                        default="legacy")
    parser.add_argument("--tile-mode", choices=("auto", "scalar", "reload", "share"), default="auto")
    parser.add_argument("--tile-width", choices=(1, 2, 4, 8), type=int, default=1)
    parser.add_argument("--directions", type=int, choices=(1, 12), default=1)
    parser.add_argument("--target", type=int, nargs=3, metavar=("Z", "Y", "X"))
    parser.add_argument("--prune-zero-fields", action="store_true")
    parser.add_argument("--factored-xy", action="store_true")
    parser.add_argument("--residency", choices=("host", "device"), default="host")
    parser.add_argument("--layout", choices=("detector", "i_a"), default="detector")
    parser.add_argument("--chunk", choices=(1, 4), type=int, default=1)
    parser.add_argument("--permute", action="store_true")
    parser.add_argument("--reference", type=Path, help="Prefix of the matched detector and axes")
    parser.add_argument("--capture-reference", action="store_true", help="Save the matched legacy detector")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.layout == "i_a" and (args.count != 101 or args.directions != 1):
        parser.error("The archived I/A comparison uses 101 energies and one direction.")
    if args.target and tuple(args.target) != TARGET_SHAPE:
        if args.layout == "i_a" or (args.mode == "stages" and not args.reference):
            parser.error("Use a matched detector reference prefix for the alternate target.")
    if args.capture_reference and (args.reference is None or args.mode != "stages" or args.layout != "detector"
                                   or args.performance_profile != "legacy" or args.endpoint_plan != "legacy"
                                   or args.tile_mode not in {"auto", "scalar"}):
        parser.error("Recording requires stage mode and a legacy scalar detector with a reference prefix.")
    args.manifest = args.manifest.resolve()
    args.checkout = args.checkout.resolve()
    report = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
