"""Evaluate a centered axial extension at the requested detector endpoints."""

import math

import numpy as np


_SUM_KERNEL = None
_SUM_SOURCE = r"""
extern "C" __global__ void virtual_z_sum(
    const float2* xy, const long long* native, const bool* reflect,
    const float2* phase, float2* out, long long count, long long columns,
    int occupied_z, int transform_z, int offset
) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const long long column = native[i] % columns;
    const int mode = native[i] / columns;
    float real = 0.0f;
    float imag = 0.0f;
    long long cycle = 0;
    for (int z = 0; z < occupied_z; ++z) {
        const float2 a = xy[z * columns + column];
        const float2 p = phase[cycle];
        real += fmaf(a.x, p.x, -a.y * p.y);
        imag += fmaf(a.x, p.y, a.y * p.x);
        cycle += mode;
        if (cycle >= transform_z) cycle -= transform_z;
    }
    const float2 p = phase[((long long)offset * mode) % transform_z];
    out[i] = make_float2(
        fmaf(real, p.x, -imag * p.y),
        (reflect[i] ? -1.0f : 1.0f) * fmaf(real, p.y, imag * p.x)
    );
}
"""


def sparse_endpoint_union(runtime, morphology, cp, energies, detector, shape_override, recorder, *, pixel_region=None):
    """Collect endpoint IDs by transverse block, with storage proportional to demand."""
    plane_size = detector.y_count * detector.x_count
    work_size = plane_size if pixel_region is None else pixel_region.size
    region_options = {} if pixel_region is None else {"pixel_region": pixel_region}
    block_size = 262_144
    blocks = [None] * math.ceil(work_size / block_size)
    for energy in dict.fromkeys(energies):
        try:
            projection = recorder.measure(
                "zpad_plan_geometry",
                lambda: runtime._detector_projection_geometry(
                    morphology, energy, cp, detector, shape_override=shape_override, **region_options,
                ),
            )
            valid = projection.valid.ravel()
            lower = None if projection.safe_z0 is None else projection.safe_z0.ravel()
            upper = None if projection.safe_z1 is None else projection.safe_z1.ravel()
            with recorder.range("zpad_endpoint_merge"):
                for block, start in enumerate(range(0, work_size, block_size)):
                    stop = start + block_size
                    columns = cp.flatnonzero(valid[start:stop]) + start
                    global_columns = columns if pixel_region is None else pixel_region.global_columns(columns, detector.x_count)
                    if lower is None:
                        ids = global_columns
                    else:
                        ids = cp.concatenate((
                            lower[columns].astype(cp.int64) * plane_size + global_columns,
                            upper[columns].astype(cp.int64) * plane_size + global_columns,
                        ))
                    if blocks[block] is not None:
                        ids = cp.concatenate((blocks[block], ids))
                    blocks[block] = cp.unique(ids)
        finally:
            runtime._discard_detector_projection_geometry(morphology, energy, shape_override=shape_override)
    with recorder.range("zpad_endpoint_merge"):
        return cp.sort(cp.concatenate(blocks))


def dc_indices(shape):
    """Return Igor IDs for the zero bin and its six native axial/lateral neighbors."""
    neighbors = ((0, 0, 1), (0, 1, 0), (0, 0, -1), (0, -1, 0), (1, 0, 0), (-1, 0, 0))
    coordinates = (np.asarray(shape) // 2 - np.asarray(((0, 0, 0), *neighbors))) % shape
    return np.ravel_multi_index(coordinates.T, shape)


class VirtualZEvaluator:
    def __init__(self, cp, occupied_z, shape, native, reflect, dc_positions, method):
        transform_z, ny, nx = shape
        self.method = method
        self.occupied_z = occupied_z
        self.transform_z = transform_z
        self.plane_size = ny * nx
        self.half_x = nx // 2 + 1
        self.columns = ny * self.half_x
        self.offset = (transform_z - occupied_z) // 2
        self.native = native
        self.reflect = reflect
        self.dc_positions = dc_positions
        self.phases = cp.asarray(np.exp(-2j * np.pi * np.arange(transform_z) / transform_z), dtype=cp.complex64)
        self.residues = []
        if method == "residues":
            pad_factor = transform_z // occupied_z
            modes = native // self.columns
            residue_ids = modes % pad_factor
            for residue in cp.asnumpy(cp.unique(residue_ids)).tolist():
                positions = cp.flatnonzero(residue_ids == residue)
                gather = (native[positions] % self.columns) * occupied_z + modes[positions] // pad_factor
                self.residues.append((residue, positions, gather))

    def capture(self, cp, field, out, recorder, *, fill_box=False, window=None, repair_dc=True):
        self._capture_occupied(cp, field, out, recorder)
        if fill_box and (window is not None or not repair_dc):
            with recorder.range("zpad_fill"):
                self._add_uniform_box(cp, out, window, recorder)
        if repair_dc:
            with recorder.range("zpad_gather"):
                out[self.dc_positions[:1]] = cp.mean(out[self.dc_positions[1:]].real)

    def _capture_occupied(self, cp, field, out, recorder):
        xy = recorder.measure("zpad_xy", lambda: cp.fft.rfft2(field, axes=(-2, -1)))
        if recorder.execution is not None:
            recorder.execution.record_fft("basis_rfft2_xy", field)
        if self.method == "sum":
            global _SUM_KERNEL
            if _SUM_KERNEL is None:
                _SUM_KERNEL = cp.RawKernel(_SUM_SOURCE, "virtual_z_sum")
            threads = 128
            with recorder.range("zpad_axial"):
                _SUM_KERNEL(
                    ((out.size + threads - 1) // threads,), (threads,),
                    (xy, self.native, self.reflect, self.phases, out, np.int64(out.size), np.int64(self.columns),
                     np.int32(self.occupied_z), np.int32(self.transform_z), np.int32(self.offset)),
                )
        else:
            self._capture_residues(cp, xy, out, recorder)

    def _add_uniform_box(self, cp, out, window, recorder):
        """Add the logical box to a transformed, windowed crop contrast row."""
        if window is None:
            out[self.dc_positions[:1]] += np.float32(self.transform_z * self.plane_size)
            return

        # Unit axial weight puts the full-box spectrum entirely on q_z = 0.
        # Gather in native RFFT order and reflect the real geometry before
        # applying complex, energy-dependent optical weights.
        wy, wx = window
        y_fft = cp.fft.fft(wy.ravel())
        x_fft = cp.fft.rfft(wx.ravel())
        if recorder.execution is not None:
            recorder.execution.record_fft("zpad_fill_fft_y", wy.ravel())
            recorder.execution.record_fft("zpad_fill_rfft_x", wx.ravel())
        positions = cp.flatnonzero(self.native < self.columns)
        columns = self.native[positions]
        spectrum = y_fft[columns // self.half_x] * x_fft[columns % self.half_x]
        out[positions] += np.float32(self.transform_z) * cp.where(
            self.reflect[positions], spectrum.conj(), spectrum,
        )

    def _capture_residues(self, cp, xy, out, recorder):
        from cupyx.scipy import fft

        source = recorder.measure("zpad_layout", lambda: cp.ascontiguousarray(xy.reshape(self.occupied_z, -1).T))
        buffer = cp.empty_like(source)
        axial = cp.arange(self.occupied_z, dtype=cp.int64)
        for residue, positions, gather in self.residues:
            with recorder.range("zpad_phase"):
                cp.multiply(source, self.phases[(axial * residue) % self.transform_z], out=buffer)
            transformed = recorder.measure("zpad_axial", lambda: fft.fft(buffer, axis=-1, overwrite_x=True))
            if recorder.execution is not None:
                recorder.execution.record_fft("basis_fft_z_residue", buffer)
            with recorder.range("zpad_gather"):
                modes = self.native[positions] // self.columns
                samples = transformed.ravel()[gather] * self.phases[(self.offset * modes) % self.transform_z]
                out[positions] = cp.where(self.reflect[positions], samples.conj(), samples)
            del transformed
