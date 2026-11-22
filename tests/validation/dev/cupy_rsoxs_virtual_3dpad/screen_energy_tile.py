"""Screen scalar and shared-load CUDA projection on the frozen one-direction vesicle."""

import argparse
import json
import math
from pathlib import Path
import statistics
from time import perf_counter
from unittest.mock import patch

import cupy as cp
import numpy as np

from NRSS.backends import _cupy_energy_basis
from NRSS.backends.cupy_rsoxs import _NullSegmentRecorder
from run_padded_scan_baseline import benchmark


_TILE_SOURCE = r"""
constexpr int fields = 7;

__device__ inline void accumulate(float2 &sum, float2 weight, float2 sample) {
    sum.x += weight.x * sample.x - weight.y * sample.y;
    sum.y += weight.x * sample.y + weight.y * sample.x;
}

__device__ inline float project_x(float2 *tensor, float a, float b, float c, float d, float scale) {
    #pragma unroll
    for (int row = 0; row < 3; ++row) {
        tensor[row].x *= scale;
        tensor[row].y *= scale;
    }
    const float aa = d - a * a, bb = d - b * b, cc = d - c * c;
    const float ab = -a * b, ac = -a * c, bc = -b * c;
    const float2 x0 = project_row(aa, ab, ac, tensor[0], tensor[1], tensor[2]);
    const float2 x1 = project_row(ab, bb, bc, tensor[0], tensor[1], tensor[2]);
    const float2 x2 = project_row(ac, bc, cc, tensor[0], tensor[1], tensor[2]);
    return dot(x0, x0) + dot(x1, x1) + dot(x2, x2);
}

extern "C" __global__ void tile_projection(
    const bool *valid, const int *z0, const int *z1, const float *fraction,
    const float *qx, const float *qy, const float *qz, const float *wave_numbers,
    const int *first_bin, const long long *offsets, const float2 *values, const float2 *weights,
    float *output, const long long endpoints, const long long plane, const int nx,
    const int first_energy, const int stop_energy, const float scale, const float nan_value
) {
    const long long column = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= plane) return;
    const float a = qx[column % nx], b = qy[column / nx];
    const long long base = offsets[column] - first_bin[column];
#ifdef NRSS_SHARE_LOADS
    long long previous_lo = -1, previous_hi = -1;
    float2 cached_lo[fields], cached_hi[fields];
#endif
    for (int energy = first_energy; energy < stop_energy; ++energy) {
        const long long pixel = (long long)energy * plane + column;
        if (!valid[pixel]) {
            output[pixel] = nan_value;
            continue;
        }
        const float k = wave_numbers[energy], d = k * k;
        const int lower = z0[pixel], upper = z1[pixel];
        const long long lo = base + lower, hi = base + upper;
        const float2 *optics = weights + energy * 3 * fields;
        float low, high;
#ifdef NRSS_SEPARATE_ENDPOINTS
        low = project_endpoint<3>(lo, values, optics, endpoints, fields, a, b, k + qz[lower], d, scale, false).xx;
        high = project_endpoint<3>(hi, values, optics, endpoints, fields, a, b, k + qz[upper], d, scale, false).xx;
#else
        float2 tensor_lo[3] = {}, tensor_hi[3] = {};
        #pragma unroll
        for (int field = 0; field < fields; ++field) {
            const float2 *row = values + field * endpoints;
#ifdef NRSS_SHARE_LOADS
            const float2 sample_lo =
                lo == previous_lo ? cached_lo[field] : lo == previous_hi ? cached_hi[field] : row[lo];
            const float2 sample_hi =
                hi == previous_hi ? cached_hi[field] : hi == previous_lo ? cached_lo[field] : row[hi];
            cached_lo[field] = sample_lo;
            cached_hi[field] = sample_hi;
#else
            const float2 sample_lo = row[lo], sample_hi = row[hi];
#endif
            #pragma unroll
            for (int component = 0; component < 3; ++component) {
                const float2 weight = optics[component * fields + field];
                accumulate(tensor_lo[component], weight, sample_lo);
                accumulate(tensor_hi[component], weight, sample_hi);
            }
        }
#ifdef NRSS_SHARE_LOADS
        previous_lo = lo;
        previous_hi = hi;
#endif
        low = project_x(tensor_lo, a, b, k + qz[lower], d, scale);
        high = project_x(tensor_hi, a, b, k + qz[upper], d, scale);
#endif
        const float weight = fraction[pixel];
        output[pixel] = (1.0f - weight) * low + weight * high;
    }
}
"""


def frozen_basis(checkout, manifest, count):
    snapshot = {}
    original = _cupy_energy_basis.build_energy_basis

    def capture(runtime, owner, *args, **kwargs):
        basis = original(runtime, owner, *args, **kwargs)
        snapshot.update(runtime=runtime, owner=owner, basis=basis)
        return basis

    args = argparse.Namespace(checkout=checkout, manifest=manifest, count=count, path="compact",
                              endpoint_plan="spans", mode="timing", repeats=1, layout="detector",
                              residency="host", chunk=1, directions=1, permute=False,
                              tile_mode="scalar", tile_width=1)
    with patch.object(_cupy_energy_basis, "build_energy_basis", capture):
        report = benchmark(args)
    return snapshot["runtime"], snapshot["owner"], snapshot["basis"], report["energies_eV"]


def panel_errors(actual, expected):
    valid = np.isfinite(expected)
    np.testing.assert_array_equal(np.isfinite(actual), valid)
    assert not np.isinf(actual).any()
    if np.array_equal(actual[valid], expected[valid]):
        return 0.0, 0.0
    reference = expected[valid].astype(np.float64)
    delta = actual[valid] - reference
    return float(np.max(np.abs(delta)) / np.max(reference)), float(np.linalg.norm(delta) / np.linalg.norm(reference))


def screen(args):
    runtime, owner, basis, energies = frozen_basis(args.checkout, args.manifest, args.count)
    assert basis.column_spans is not None and basis.coefficients.shape == (args.count, 3, 7)
    shape = basis.shape_override
    detector = runtime._detector_geometry(owner, cp, shape_override=shape)
    output_shape = (args.count, *shape[1:])
    valid = cp.empty(output_shape, dtype=cp.bool_)
    z0 = cp.empty(output_shape, dtype=cp.int32)
    z1 = cp.empty_like(z0)
    fraction = cp.empty(output_shape, dtype=cp.float32)
    for i, energy in enumerate(energies):
        p = runtime._detector_projection_geometry(owner, energy, cp, detector, shape_override=shape)
        valid[i], z0[i], z1[i], fraction[i] = p.valid, p.safe_z0, p.safe_z1, p.frac
        runtime._discard_detector_projection_geometry(owner, energy, shape_override=shape)
    wave = cp.asarray([2.0 * math.pi / (1239.84197 / e) for e in energies], dtype=cp.float32)
    panel = cp.empty(output_shape, dtype=cp.float32)
    source = _cupy_energy_basis._COMPACT_TENSOR_PROJECTION_SOURCE + _TILE_SOURCE
    kernels = {
        "separate_endpoints": cp.RawKernel(source, "tile_projection", options=("-DNRSS_SEPARATE_ENDPOINTS",)),
        "scalar_loads": cp.RawKernel(source, "tile_projection"),
        "shared_loads": cp.RawKernel(source, "tile_projection", options=("-DNRSS_SHARE_LOADS",)),
    }
    plane = math.prod(shape[1:])
    stream = cp.cuda.get_current_stream()
    archive = np.load(args.manifest.with_name("virtual-101-chunk1-detector.npy"), mmap_mode="r")
    manifest = json.loads(args.manifest.read_text())
    indices = manifest["energy_index_sets"][str(args.count)]
    reference = np.empty(output_shape, dtype=np.float32)
    angles = runtime._angle_family_plan(owner)
    recorder = _NullSegmentRecorder()
    archive_errors = []
    for i, energy in enumerate(energies):
        x, _, _ = _cupy_energy_basis.project_energy_fused(runtime, owner, cp, basis, i, energy, angles, recorder)
        x.get(out=reference[i])
        archive_errors.append(panel_errors(reference[i], archive[indices[i]]))
    assert np.max(archive_errors) <= 2e-5, np.max(archive_errors, axis=0)

    def launch(kernel, width):
        for low in range(0, args.count, width):
            kernel(
                ((plane + 127) // 128,), (128,),
                (valid, z0, z1, fraction, detector.qx, detector.qy, detector.qz, wave,
                 basis.column_spans.first, basis.column_spans.offsets, basis.values, basis.coefficients,
                 panel, np.int64(basis.indices.size), np.int64(plane), np.int32(shape[2]),
                 np.int32(low), np.int32(min(low + width, args.count)),
                 np.float32(runtime._one_by_four_pi), np.float32(np.nan)),
            )

    reports = {}
    for name, kernel in kernels.items():
        widths = (1,) if name == "separate_endpoints" else (1, 2, 4, 8)
        for width in widths:
            launch(kernel, width)
            errors = np.asarray([panel_errors(panel[i].get(), reference[i]) for i in range(args.count)])
            if np.max(errors) > 2e-5:
                raise AssertionError(f"{name} width {width}: {np.max(errors, axis=0)}")
            times = []
            for _ in range(args.repeats):
                stream.synchronize()
                start = perf_counter()
                launch(kernel, width)
                stream.synchronize()
                times.append(perf_counter() - start)
            reports[f"{name}_{width}"] = {
                "width": width,
                "final_tile": args.count % width or width,
                "exact_panels": int(np.count_nonzero(errors[:, 0] == 0)),
                "max_abs_over_peak": float(errors[:, 0].max()),
                "max_relative_l2": float(errors[:, 1].max()),
                "seconds": times,
                "median_seconds": statistics.median(times),
                "registers_per_thread": kernel.num_regs,
                "local_bytes_per_thread": kernel.local_size_bytes,
            }
    return {"gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "logical_zyx": shape, "energies": args.count,
            "local_scalar_vs_archive_max_over_peak_and_l2": np.max(archive_errors, axis=0).tolist(),
            "reports": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=(1, 20, 101), default=101)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.checkout = args.checkout.resolve()
    args.manifest = args.manifest.resolve()
    args.output.write_text(json.dumps(screen(args), indent=2) + "\n")
