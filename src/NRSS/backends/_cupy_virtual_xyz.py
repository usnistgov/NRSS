"""Evaluate the centered XYZ Fourier grid using bounded batches of XY planes."""

import numpy as np

from ._cupy_virtual_z import VirtualZEvaluator


_XYZ_KERNEL = None
_XYZ_FACTORED_KERNEL = None
XYZ_FACTORED_XY_KEY = "_private_xyz_factored_xy"
_XYZ_SOURCE = r"""
__device__ inline float2 multiply(float2 a, float2 b) {
    return make_float2(
        fmaf(a.x, b.x, -a.y * b.y),
        fmaf(a.x, b.y, a.y * b.x)
    );
}

extern "C" __global__ void virtual_xyz_accumulate(
    const float2* xy, const long long* native, const bool* reflect,
#ifdef NRSS_FACTORED_XY
    const int* x_slots, long long xy_columns,
#endif
    const float2* phase_z, const float2* phase_y, const float2* phase_x,
    float2* out, long long count, long long columns, int half_x,
    int mz, int my, int mx, int oz, int oy, int ox, int first_z, int plane_count,
    bool last_batch
) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const long long column = native[i] % columns;
    const int kz = native[i] / columns;
#ifdef NRSS_FACTORED_XY
    const int ky = column / half_x;
    const int kx = column % half_x;
    const long long xy_column = (long long)x_slots[kx] * my + ky;
#else
    const long long xy_columns = columns;
    const long long xy_column = column;
#endif
    const float2 initial = first_z == 0 ? make_float2(0.0f, 0.0f) : out[i];
    float real = initial.x;
    float imag = initial.y;
    long long cycle = (long long) first_z * kz % mz;
    for (int iz = 0; iz < plane_count; ++iz) {
        const float2 a = xy[iz * xy_columns + xy_column];
        const float2 p = phase_z[cycle];
        real += fmaf(a.x, p.x, -a.y * p.y);
        imag += fmaf(a.x, p.y, a.y * p.x);
        cycle += kz;
        if (cycle >= mz) cycle -= mz;
    }
    float2 value = make_float2(real, imag);
    if (last_batch) {
#ifndef NRSS_FACTORED_XY
        const int ky = column / half_x;
        const int kx = column % half_x;
#endif
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


def full_detector_kx(runtime, cp, energies, detector, logical_shape, stencil):
    """Find the incumbent X FFT support without materializing its endpoint union."""
    _, ny, nx = logical_shape
    radius_limit = max(runtime._detector_projection_parameters(energy, detector)[1] for energy in energies)
    # Use the row nearest zero that is inside the high border. For a
    # singleton Y, the only row is invalid and the DC stencil still applies.
    row = max(0, min(ny // 2, ny - 2))
    valid = detector.border_valid[row] & (detector.radius_sq[row] <= radius_limit)
    columns = cp.concatenate((cp.flatnonzero(valid), cp.asarray(stencil % nx)))
    x = runtime._igor_axis_orders(logical_shape, cp)[2][columns]
    return cp.minimum(x, (-x) % nx)


class VirtualXYZEvaluator(VirtualZEvaluator):
    """Sample a logical XYZ transform without allocating a logical 3D array."""

    def __init__(
        self, cp, occupied_shape, logical_shape, native, reflect, dc_positions, *, factored=False, transform_kx=None,
    ):
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
        self.factored_kx = None
        self.factored_fallback = None
        if factored:
            # Each endpoint already uses canonical RFFT indices, including
            # reflected partners and the DC neighbors. Build the X map once.
            try:
                needed = cp.zeros(self.half_x, dtype=cp.bool_)
                needed[native % self.half_x if transform_kx is None else transform_kx] = True
                columns = cp.flatnonzero(needed)
                if factored is True or columns.size * 4 <= self.half_x * 3:
                    self._x_slots = cp.cumsum(needed, dtype=cp.int32) - 1
                    self.factored_kx = columns
                else:
                    self.factored_fallback = "dense transverse support"
            except cp.cuda.memory.OutOfMemoryError:
                self.factored_fallback = "factored column allocation failed"

    def _factored_xy(self, cp, field, recorder):
        my, mx = self.lateral_shape
        x_fft = recorder.measure("zpad_xy_x", lambda: cp.fft.rfft(field, n=mx, axis=-1))
        if recorder.execution is not None:
            recorder.execution.record_fft("basis_rfft_x_occupied_y", x_fft)
        packed = recorder.measure(
            "zpad_xy_pack",
            lambda: cp.ascontiguousarray(cp.take(x_fft, self.factored_kx, axis=-1).transpose(0, 2, 1)),
        )
        del x_fft
        xy = recorder.measure("zpad_xy_y", lambda: cp.fft.fft(packed, n=my, axis=-1))
        if recorder.execution is not None:
            recorder.execution.record_fft("basis_fft_y_selected_x", xy)
        return xy

    def _capture_occupied(self, cp, field, out, recorder):
        if self.factored_kx is not None:
            try:
                self._capture_batches(cp, field, out, recorder)
                return
            except cp.cuda.memory.OutOfMemoryError:
                self.factored_kx = None
                self._x_slots = None
                self.factored_fallback = "factored FFT workspace allocation failed"
        self._capture_batches(cp, field, out, recorder)

    def _capture_batches(self, cp, field, out, recorder):
        global _XYZ_KERNEL, _XYZ_FACTORED_KERNEL
        my, mx = self.lateral_shape
        if self.factored_kx is None:
            if _XYZ_KERNEL is None:
                _XYZ_KERNEL = cp.RawKernel(_XYZ_SOURCE, "virtual_xyz_accumulate")
            kernel, lookup = _XYZ_KERNEL, ()
        else:
            if _XYZ_FACTORED_KERNEL is None:
                _XYZ_FACTORED_KERNEL = cp.RawKernel(
                    _XYZ_SOURCE, "virtual_xyz_accumulate", options=("-DNRSS_FACTORED_XY",),
                )
            kernel = _XYZ_FACTORED_KERNEL
            lookup = (self._x_slots, np.int64(my * self.factored_kx.size))
        first = 0
        threads = 128
        while first < self.occupied_z:
            stop = min(first + self.tile_depth, self.occupied_z)
            try:
                if self.factored_kx is None:
                    xy = recorder.measure(
                        "zpad_xy", lambda: cp.fft.rfft2(field[first:stop], s=self.lateral_shape, axes=(-2, -1)),
                    )
                    if recorder.execution is not None:
                        recorder.execution.record_fft("basis_rfft2_logical_xy", xy)
                else:
                    xy = recorder.measure("zpad_xy", lambda: self._factored_xy(cp, field[first:stop], recorder))
            except cp.cuda.memory.OutOfMemoryError:
                if stop - first == 1:
                    raise
                self.tile_depth = (stop - first) // 2
                continue
            with recorder.range("zpad_axial"):
                kernel(
                    ((out.size + threads - 1) // threads,), (threads,),
                    (
                        xy, self.native, self.reflect, *lookup, self.phases, self.phase_y, self.phase_x, out,
                        np.int64(out.size), np.int64(self.columns), np.int32(self.half_x),
                        np.int32(self.transform_z), np.int32(my), np.int32(mx),
                        np.int32(self.offset), np.int32(self.offset_y), np.int32(self.offset_x),
                        np.int32(first), np.int32(stop - first), np.bool_(stop == self.occupied_z),
                    ),
                )
            del xy
            first = stop
