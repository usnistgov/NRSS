"""Evaluate the centered XYZ Fourier grid using bounded batches of XY planes."""

import numpy as np

from ._cupy_virtual_z import VirtualZEvaluator


_XYZ_KERNEL = None
_XYZ_SOURCE = r"""
__device__ inline float2 multiply(float2 a, float2 b) {
    return make_float2(
        fmaf(a.x, b.x, -a.y * b.y),
        fmaf(a.x, b.y, a.y * b.x)
    );
}

extern "C" __global__ void virtual_xyz_accumulate(
    const float2* xy, const long long* native, const bool* reflect,
    const float2* phase_z, const float2* phase_y, const float2* phase_x,
    float2* out, long long count, long long columns, int half_x,
    int mz, int my, int mx, int oz, int oy, int ox, int first_z, int plane_count,
    bool last_batch
) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const long long column = native[i] % columns;
    const int kz = native[i] / columns;
    const float2 initial = first_z == 0 ? make_float2(0.0f, 0.0f) : out[i];
    float real = initial.x;
    float imag = initial.y;
    long long cycle = (long long) first_z * kz % mz;
    for (int iz = 0; iz < plane_count; ++iz) {
        const float2 a = xy[iz * columns + column];
        const float2 p = phase_z[cycle];
        real += fmaf(a.x, p.x, -a.y * p.y);
        imag += fmaf(a.x, p.y, a.y * p.x);
        cycle += kz;
        if (cycle >= mz) cycle -= mz;
    }
    float2 value = make_float2(real, imag);
    if (last_batch) {
        const int ky = column / half_x;
        const int kx = column % half_x;
        const float2 placement = multiply(
            multiply(phase_z[(long long)oz * kz % mz], phase_y[(long long)oy * ky % my]),
            phase_x[(long long)ox * kx % mx]
        );
        value = multiply(value, placement);
        if (reflect[i]) value.y = -value.y;
    }
    out[i] = value;
}
"""


class VirtualXYZEvaluator(VirtualZEvaluator):
    """Sample a logical XYZ transform without allocating a logical 3D array."""

    def __init__(self, cp, occupied_shape, logical_shape, native, reflect, dc_positions):
        nz, ny, nx = occupied_shape
        mz, my, mx = logical_shape
        super().__init__(cp, nz, logical_shape, native, reflect, dc_positions, "sum")
        self.lateral_shape = (my, mx)
        self.offset_y = (my - ny) // 2
        self.offset_x = (mx - nx) // 2
        self.phase_y = cp.asarray(np.exp(-2j * np.pi * np.arange(my) / my), dtype=cp.complex64)
        self.phase_x = cp.asarray(np.exp(-2j * np.pi * np.arange(mx) / mx), dtype=cp.complex64)
        plane_bytes = self.plane_size * np.dtype(np.float32).itemsize
        self.tile_depth = max(1, min(nz, (64 * 1024**2) // plane_bytes))

    def _capture_occupied(self, cp, field, out, recorder):
        global _XYZ_KERNEL
        if _XYZ_KERNEL is None:
            _XYZ_KERNEL = cp.RawKernel(_XYZ_SOURCE, "virtual_xyz_accumulate")
        my, mx = self.lateral_shape
        first = 0
        threads = 128
        while first < self.occupied_z:
            stop = min(first + self.tile_depth, self.occupied_z)
            try:
                xy = recorder.measure(
                    "zpad_xy", lambda: cp.fft.rfft2(field[first:stop], s=self.lateral_shape, axes=(-2, -1)),
                )
            except cp.cuda.memory.OutOfMemoryError:
                if stop - first == 1:
                    raise
                self.tile_depth = (stop - first) // 2
                continue
            if recorder.execution is not None:
                recorder.execution.record_fft("basis_rfft2_logical_xy", xy)
            with recorder.range("zpad_axial"):
                _XYZ_KERNEL(
                    ((out.size + threads - 1) // threads,), (threads,),
                    (
                        xy, self.native, self.reflect, self.phases, self.phase_y, self.phase_x, out,
                        np.int64(out.size), np.int64(self.columns), np.int32(self.half_x),
                        np.int32(self.transform_z), np.int32(my), np.int32(mx),
                        np.int32(self.offset), np.int32(self.offset_y), np.int32(self.offset_x),
                        np.int32(first), np.int32(stop - first), np.bool_(stop == self.occupied_z),
                    ),
                )
            del xy
            first = stop
