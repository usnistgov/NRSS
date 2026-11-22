"""Compare equivalent per-energy geometry on full columns and exact radius keys."""

import argparse
import json
from pathlib import Path
import statistics
from time import perf_counter
from types import SimpleNamespace

import cupy as cp
import numpy as np

from NRSS.backends.cupy_rsoxs import CupyRsoxsBackendRuntime


def geometry_kernels():
    geometry = cp.ElementwiseKernel(
        "float32 radius, bool border, float32 k, float32 k_sq, float32 limit, "
        "float32 dz, int32 mid, int32 last",
        "bool valid, int32 low, int32 high, float32 fraction",
        """
        valid = border && radius <= limit;
        float offset = 0.0f;
        if (valid && last > 0) {
            const float root = __fsqrt_rn(fmaxf(k_sq - radius, 0.0f));
            const float position = __fdiv_rn(-radius, k + root);
            offset = fmaxf(__fdiv_rn(position, dz), -(float)mid);
        }
        const float bin = floorf(offset);
        low = mid + (int)bin;
        high = min(low + 1, last);
        fraction = offset - bin;
        """, "nrss_probe_radial_geometry",
    )
    gather = cp.ElementwiseKernel(
        "int32 key, raw bool rad_valid, raw int32 rad_low, raw int32 rad_high, "
        "raw float32 rad_frac, int32 mid, int32 last",
        "bool valid, int32 low, int32 high, float32 fraction",
        """
        valid = key >= 0 && rad_valid[key];
        low = valid ? rad_low[key] : mid;
        high = valid ? rad_high[key] : min(mid + 1, last);
        fraction = valid ? rad_frac[key] : 0.0f;
        """, "nrss_probe_gather_radial_geometry",
    )
    return geometry, gather


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    fixture = json.loads(args.manifest.read_text())
    energies = fixture["energies_eV"]
    shape = tuple(fixture["target_zyx"])
    runtime = CupyRsoxsBackendRuntime()
    owner = SimpleNamespace(PhysSize=fixture["geometry"]["pitch_nm"], _backend_runtime_state={})
    detector = runtime._detector_geometry(owner, cp, shape_override=shape)
    stream = cp.cuda.get_current_stream()
    geometry, gather = geometry_kernels()
    mid, last = np.int32(shape[0] // 2), np.int32(shape[0] - 1)

    def constants(e):
        k = 2 * np.pi / (1239.84197 / e)
        depth = min(k, mid * detector.dz)
        return (np.float32(k), np.float32(k * k), np.float32(depth * (2 * k - depth)),
                np.float32(detector.dz), mid, last)

    parameters = [constants(e) for e in energies]

    def reference(energy):
        p = runtime._detector_projection_geometry(owner, energy, cp, detector, shape_override=shape)
        runtime._discard_detector_projection_geometry(owner, energy, shape_override=shape)
        return p.valid, p.safe_z0, p.safe_z1, p.frac

    def radius_table():
        columns = cp.flatnonzero(reference(energies[-1])[0])
        radii = detector.radius_sq.ravel()[columns]
        keys, inverse = cp.unique(radii.view(cp.uint32), return_inverse=True)
        key_map = cp.full(detector.radius_sq.shape, -1, dtype=cp.int32)
        key_map.ravel()[columns] = inverse.astype(cp.int32)
        table = [geometry(keys.view(cp.float32), np.bool_(True), *params) for params in parameters]
        return key_map, table

    def fused_columns():
        for params in parameters:
            geometry(detector.radius_sq, detector.border_valid, *params)

    def incumbent():
        for energy in energies:
            reference(energy)

    def gather_table(table):
        key_map, rows = table
        for row in rows:
            gather(key_map, *row, mid, last)

    def elapsed(action):
        stream.synchronize()
        start = perf_counter()
        result = action()
        stream.synchronize()
        return perf_counter() - start, result

    key_map, rows = radius_table()
    for energy, params, row in zip(energies, parameters, rows):
        expected = reference(energy)
        per_column = geometry(detector.radius_sq, detector.border_valid, *params)
        by_radius = gather(key_map, *row, mid, last)
        matches = cp.stack([cp.all(left == right) for arm in (per_column, by_radius)
                            for left, right in zip(expected, arm)])
        if not bool(matches.all()):
            raise AssertionError(f"Geometry differs at {energy} eV: {matches.get().tolist()}")

    measurements = {"incumbent": [], "fused_columns": [], "table_build": [], "table_gather": []}
    for _ in range(args.repeats):
        measurements["incumbent"].append(elapsed(incumbent)[0])
        measurements["fused_columns"].append(elapsed(fused_columns)[0])
        duration, table = elapsed(radius_table)
        measurements["table_build"].append(duration)
        measurements["table_gather"].append(elapsed(lambda: gather_table(table))[0])
    result = {
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "logical_zyx": shape,
        "energies": len(energies),
        "exact_geometry": True,
        "radius_keys": rows[0][0].size,
        "radius_table_bytes": key_map.nbytes + sum(array.nbytes for row in rows for array in row),
        "seconds": measurements,
        "median_seconds": {k: statistics.median(v) for k, v in measurements.items()},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
