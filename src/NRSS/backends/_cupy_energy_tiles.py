"""Private experiments for bounded, multi-energy virtual XYZ projection."""

import math

import numpy as np

from ._cupy_energy_basis import _COMPACT_TENSOR_PROJECTION_SOURCE


_MAX_FIELDS = 16
_MAX_TILE_BYTES = 96 * 1024**2
_TILE_KERNELS = {}


_TILE_SOURCE = r"""
constexpr int fields = NRSS_TILE_FIELDS;
constexpr int components = NRSS_TILE_COMPONENTS;

extern "C" __global__ void xyz_energy_tile(
    const bool *border, const float *radius, const float *qx, const float *qy, const float *qz,
    const float *parameters, const float dz,
#ifdef NRSS_TILE_SPANS
    const int *first_bin, const long long *offsets,
#else
    const long long *indices,
#endif
    const float2 *values, const float2 *weights, float *output,
    const long long endpoints, const long long plane, const int nx, const int nz,
    const int count, const int channels, const float scale, const float nan_value
#ifdef NRSS_PIXEL_RECTANGLE
    , const int origin_y, const int origin_x, const int width, const long long logical_plane
#endif
) {
    const long long column = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= plane) return;
#ifdef NRSS_PIXEL_RECTANGLE
    const int iy = origin_y + column / width, ix = origin_x + column % width;
    const long long global_column = (long long)iy * nx + ix;
    const long long plane_stride = logical_plane;
#else
    const int iy = column / nx, ix = column % nx;
    const long long global_column = column, plane_stride = plane;
#endif
    const bool in_border = border[global_column];
    const float r2 = radius[global_column];
    const float a = qx[ix], b = qy[iy];
    const int mid = nz / 2;
    const bool cross = channels == 3;
#ifdef NRSS_TILE_SPANS
    const long long base = offsets[global_column] - first_bin[global_column];
#endif
#ifdef NRSS_SHARE_LOADS
    long long previous_lo = -1, previous_hi = -1;
    float2 cached_lo[fields], cached_hi[fields];
#endif
    for (int i = 0; i < count; ++i) {
        float *const pixel = output + (long long)i * channels * plane + column;
        const float *const p = parameters + (long long)i * 3;
        if (!in_border || !(r2 <= p[2])) {
            for (int row = 0; row < channels; ++row) pixel[row * plane] = nan_value;
            continue;
        }
        const float k = p[0], d = k * k;
        const float root = __fsqrt_rn(fmaxf(p[1] - r2, 0.0f));
        const float position = __fdiv_rn(-r2, k + root);
        const float offset = fmaxf(__fdiv_rn(position, dz), -(float)mid);
        const float bin = floorf(offset);
        const int lower = (int)bin + mid, upper = min(lower + 1, nz - 1);
#ifdef NRSS_TILE_SPANS
        const long long lo = base + lower, hi = base + upper;
#else
        const long long lo = index_of(indices, endpoints, (long long)lower * plane_stride + global_column);
        const long long hi = index_of(indices, endpoints, (long long)upper * plane_stride + global_column);
#endif
        const float2 *const optics = weights + (long long)i * components * fields;
        float2 tensor_lo[components] = {}, tensor_hi[components] = {};
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
            for (int component = 0; component < components; ++component) {
                const float2 w = optics[component * fields + field];
                tensor_lo[component].x += w.x * sample_lo.x - w.y * sample_lo.y;
                tensor_lo[component].y += w.x * sample_lo.y + w.y * sample_lo.x;
                tensor_hi[component].x += w.x * sample_hi.x - w.y * sample_hi.y;
                tensor_hi[component].y += w.x * sample_hi.y + w.y * sample_hi.x;
            }
        }
#ifdef NRSS_SHARE_LOADS
        previous_lo = lo;
        previous_hi = hi;
#endif
        const Coefficients low = project_tensor<components>(tensor_lo, a, b, k + qz[lower], d, scale, cross);
        const Coefficients high = project_tensor<components>(tensor_hi, a, b, k + qz[upper], d, scale, cross);
        const float f = offset - bin, keep = 1.0f - f;
        pixel[0] = keep * low.xx + f * high.xx;
        if (channels > 1) pixel[plane] = keep * low.yy + f * high.yy;
        if (cross) pixel[2 * plane] = keep * low.xy + f * high.xy;
    }
}
"""


class EnergyTileProjector:
    def __init__(self, cp, basis, detector, angles, parameters, width, share, scale, pixel_region=None):
        self._cp = cp
        self._basis = basis
        self._detector = detector
        self._parameters = cp.asarray(parameters)
        self._scale = np.float32(scale)
        logical_shape = (detector.y_count, detector.x_count)
        self._shape = logical_shape if pixel_region is None else pixel_region.shape
        self._region_args = ()
        if pixel_region is not None:
            self._region_args = (*pixel_region.kernel_args, np.int64(math.prod(logical_shape)))
        self.angles = angles
        self.channels = sum((angles.needs_proj_x, angles.needs_proj_y, angles.needs_proj_xy))
        self.width = width
        fields = basis.values.shape[0]
        packed = basis.column_spans is not None
        if packed:
            self._addressing = (basis.column_spans.first, basis.column_spans.offsets)
        else:
            self._addressing = (basis.indices,)
        key = (fields, len(basis.components), packed, share, pixel_region is not None)
        kernel = _TILE_KERNELS.get(key)
        if kernel is None:
            options = [f"-DNRSS_TILE_FIELDS={fields}", f"-DNRSS_TILE_COMPONENTS={len(basis.components)}"]
            if packed:
                options.append("-DNRSS_TILE_SPANS")
            if share:
                options.append("-DNRSS_SHARE_LOADS")
            if pixel_region is not None:
                options.append("-DNRSS_PIXEL_RECTANGLE")
            kernel = cp.RawKernel(_COMPACT_TENSOR_PROJECTION_SOURCE + _TILE_SOURCE,
                                  "xyz_energy_tile", options=tuple(options))
            _TILE_KERNELS[key] = kernel
        self._kernel = kernel

    def project(self, start, stop, recorder):
        count = stop - start
        panels = self._cp.empty((count, self.channels, *self._shape), dtype=self._cp.float32)
        basis, detector = self._basis, self._detector
        plane = math.prod(self._shape)
        threads = 128
        with recorder.range("zpad_projection_kernel"):
            self._kernel(
                ((plane + threads - 1) // threads,), (threads,),
                (
                    detector.border_valid, detector.radius_sq, detector.qx, detector.qy, detector.qz,
                    self._parameters[start:stop], np.float32(detector.dz), *self._addressing,
                    basis.values, basis.coefficients[start:stop], panels,
                    np.int64(basis.indices.size), np.int64(plane),
                    np.int32(detector.x_count), np.int32(detector.z_count), np.int32(count),
                    np.int32(self.channels), self._scale, np.float32(np.nan),
                    *self._region_args,
                ),
            )
        if recorder.execution is not None:
            details = recorder.execution.energy_projector
            if "registers_per_thread" not in details:
                details.update(registers_per_thread=self._kernel.num_regs,
                               local_bytes_per_thread=self._kernel.local_size_bytes)
        return panels


def make_xyz_energy_projector(runtime, morphology, cp, basis, energies, request, recorder, pixel_region=None):
    width, share = request
    if width not in (1, 2, 4, 8) or not isinstance(share, bool):
        raise ValueError("Private XYZ energy tiles require a width of 1, 2, 4, or 8 and a sharing flag.")
    if recorder.execution is not None:
        recorder.execution.energy_projector = {"requested_width": width, "share_loads": share, "width": 1}

    def fallback(reason):
        if recorder.execution is not None:
            recorder.execution.energy_projector["fallback"] = reason
        return None

    if basis is None or basis.virtual_mode != "3d":
        return fallback("requires virtual XYZ tensor basis")
    if basis.values.shape[0] == 0:
        return fallback("no nonzero basis fields")
    if basis.values.shape[0] > _MAX_FIELDS:
        return fallback("too many basis fields for the specialized kernel")
    z, y, x = basis.shape_override
    if not 1 < z <= 2**24 or any(not math.isfinite(e) or e <= 0 for e in energies):
        return fallback("geometry outside the experiment")
    detector = runtime._detector_geometry(morphology, cp, shape_override=basis.shape_override)
    angles = runtime._angle_family_plan(morphology)
    channels = sum((angles.needs_proj_x, angles.needs_proj_y, angles.needs_proj_xy))
    panel_bytes = channels * y * x * np.dtype(np.float32).itemsize
    width = min(width, len(energies), _MAX_TILE_BYTES // panel_bytes)
    if width < 1:
        return fallback("projected panel exceeds the tile workspace")
    k_parameters = [runtime._detector_projection_parameters(energy, detector) for energy in energies]
    with np.errstate(over="ignore", under="ignore"):
        parameters = np.asarray([(k, k*k, limit) for k, limit in k_parameters], dtype=np.float32)
    if (not np.isfinite(parameters).all() or (parameters[:, 0] <= 0).any()
            or not 0 < np.float32(detector.dz) < np.inf):
        return fallback("geometry outside the experiment")
    if recorder.execution is not None:
        tile_bytes = panel_bytes if pixel_region is None else panel_bytes // (y * x) * pixel_region.size
        recorder.execution.energy_projector.update(
            width=width, workspace_bytes=tile_bytes * width,
            addressing="spans" if basis.column_spans is not None else "sorted",
        )
    return EnergyTileProjector(
        cp, basis, detector, angles, parameters, width, share, runtime._one_by_four_pi, pixel_region,
    )
