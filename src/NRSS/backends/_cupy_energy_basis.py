"""Per-run Fourier bases for CuPy energy scans."""

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


_TENSOR_AXES = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2))  # xx, xy, xz, yy, yz
_COMPACT_TENSOR_PROJECTION_KERNEL = None
_WINDOWED_BASIS_KERNELS = None


_COMPACT_TENSOR_PROJECTION_SOURCE = r"""
__device__ inline float dot(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}

__device__ inline float2 project_row(
    const float m0, const float m1, const float m2,
    const float2 v0, const float2 v1, const float2 v2
) {
    return make_float2(
        m0 * v0.x + m1 * v1.x + m2 * v2.x,
        m0 * v0.y + m1 * v1.y + m2 * v2.y
    );
}

struct Coefficients {
    float xx, yy, xy;
};

__device__ inline long long index_of(
    const long long *indices, long long count, const long long endpoint
) {
    long long first = 0;
    while (count > 0) {
        const long long half = count / 2;
        const long long middle = first + half;
        if (indices[middle] < endpoint) {
            first = middle + 1;
            count -= half + 1;
        } else {
            count = half;
        }
    }
    return first;
}

template<int components>
__device__ inline Coefficients project_endpoint(
    const long long position, const float2 *values, const float2 *weights,
    const long long endpoints, const int fields,
    const float a, const float b, const float c, const float d,
    const float scale, const bool cross
) {
    float2 tensor[components] = {};
    for (int field = 0; field < fields; ++field) {
        const float2 sample = values[field * endpoints + position];
        #pragma unroll
        for (int row = 0; row < components; ++row) {
            const float2 weight = weights[row * fields + field];
            tensor[row].x += weight.x * sample.x - weight.y * sample.y;
            tensor[row].y += weight.x * sample.y + weight.y * sample.x;
        }
    }
    #pragma unroll
    for (int row = 0; row < components; ++row) {
        tensor[row].x *= scale;
        tensor[row].y *= scale;
    }

    const float aa = d - a * a, bb = d - b * b, cc = d - c * c;
    const float ab = -a * b, ac = -a * c, bc = -b * c;
    const float2 x0 = project_row(aa, ab, ac, tensor[0], tensor[1], tensor[2]);
    const float2 x1 = project_row(ab, bb, bc, tensor[0], tensor[1], tensor[2]);
    const float2 x2 = project_row(ac, bc, cc, tensor[0], tensor[1], tensor[2]);
    Coefficients result = {dot(x0, x0) + dot(x1, x1) + dot(x2, x2), 0.0f, 0.0f};
    if (components == 5) {
        const float2 y0 = project_row(aa, ab, ac, tensor[1], tensor[3], tensor[4]);
        const float2 y1 = project_row(ab, bb, bc, tensor[1], tensor[3], tensor[4]);
        const float2 y2 = project_row(ac, bc, cc, tensor[1], tensor[3], tensor[4]);
        result.yy = dot(y0, y0) + dot(y1, y1) + dot(y2, y2);
        if (cross) {
            result.xy = 2.0f * (dot(x0, y0) + dot(x1, y1) + dot(x2, y2));
        }
    }
    return result;
}

extern "C" __global__ void compact_tensor_projection(
    const bool *valid, const int *z0, const int *z1, const float *fraction,
    const float *qx, const float *qy, const float *qz, const float wave_number,
    const long long *indices, const float2 *values, const float2 *weights,
    float *output, const long long endpoints, const int fields,
    const int components, const int channels, const int nx, const int nz,
    const long long plane_size, const float scale, const float nan_value
) {
    const long long column = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= plane_size) {
        return;
    }
    float *const result = output + column;
    if (!valid[column]) {
        for (int row = 0; row < channels; ++row) {
            result[row * plane_size] = nan_value;
        }
        return;
    }
    const float a = qx[column % nx], b = qy[column / nx];
    const float d = wave_number * wave_number;
    const int lower = nz == 1 ? 0 : z0[column];
    const long long lo = index_of(indices, endpoints, lower * plane_size + column);
    const bool cross = channels == 3;
    Coefficients low = components == 3
        ? project_endpoint<3>(lo, values, weights, endpoints, fields,
                              a, b, wave_number + qz[lower], d, scale, cross)
        : project_endpoint<5>(lo, values, weights, endpoints, fields,
                              a, b, wave_number + qz[lower], d, scale, cross);
    if (nz > 1) {
        const int upper = z1[column];
        const long long hi = index_of(indices, endpoints, upper * plane_size + column);
        const Coefficients high = components == 3
            ? project_endpoint<3>(hi, values, weights, endpoints, fields,
                                  a, b, wave_number + qz[upper], d, scale, cross)
            : project_endpoint<5>(hi, values, weights, endpoints, fields,
                                  a, b, wave_number + qz[upper], d, scale, cross);
        const float weight = fraction[column], keep = 1.0f - weight;
        low.xx = keep * low.xx + weight * high.xx;
        low.yy = keep * low.yy + weight * high.yy;
        low.xy = keep * low.xy + weight * high.xy;
    }
    result[0] = low.xx;
    if (channels > 1) {
        result[plane_size] = low.yy;
    }
    if (cross) {
        result[2 * plane_size] = low.xy;
    }
}
"""


@dataclass(frozen=True)
class EnergyBasis:
    indices: Any
    values: Any
    coefficients: Any
    components: tuple[int, ...]
    shape_override: tuple[int, int, int] | None = None
    virtual_method: str | None = None
    virtual_mode: str | None = None
    virtual_residues: int = 0
    xyz_xy_tile_planes: int | None = None
    virtual_energy_masks: int = 0

    @property
    def nbytes(self):
        return self.indices.nbytes + self.values.nbytes + self.coefficients.nbytes


def _detector_columns(cp, projection):
    columns = cp.flatnonzero(projection.valid)
    if projection.safe_z0 is None:
        return columns, None, None
    return columns, projection.safe_z0.ravel()[columns], projection.safe_z1.ravel()[columns]


def _detector_endpoint_union(runtime, morphology, cp, energies, detector):
    plane_size = detector.y_count * detector.x_count
    used = cp.zeros(detector.z_count * plane_size, dtype=cp.bool_)
    for energy in dict.fromkeys(energies):
        try:
            projection = runtime._detector_projection_geometry(morphology, energy, cp, detector)
            columns, z0, z1 = _detector_columns(cp, projection)
            if z0 is None:
                used[columns] = True
            else:
                for indices in (z0, z1):
                    used[indices.astype(columns.dtype) * plane_size + columns] = True
        finally:
            runtime._discard_detector_projection_geometry(morphology, energy)
    return cp.flatnonzero(used)


def _native_indices(runtime, cp, shape, indices, real_transform):
    nz, ny, nx = shape
    z_order, y_order, x_order = runtime._igor_axis_orders(shape, cp)
    z = z_order[indices // (ny * nx)].astype(indices.dtype)
    y = y_order[(indices // nx) % ny]
    x = x_order[indices % nx]
    if not real_transform:
        return (z * ny + y) * nx + x, None

    # The omitted half is Hermitian in all three dimensions for real fields.
    reflect = x > nx // 2
    z = cp.where(reflect, (-z) % nz, z)
    y = cp.where(reflect, (-y) % ny, y)
    x = cp.where(reflect, (-x) % nx, x)
    return (z * ny + y) * (nx // 2 + 1) + x, reflect


def _transform_basis(runtime, cp, field, plan, shape, execution):
    if plan is not None:
        plan.fft(field, field, cp.cuda.cufft.CUFFT_FORWARD)
        if execution is not None:
            execution.record_fft("basis_cufft_inplace_c2c", field)
        runtime._replace_dc_component(field)
        return field

    transformed = cp.fft.rfftn(field)
    if execution is not None:
        execution.record_fft("basis_rfftn_r2c", field)
    nz, ny, _ = shape
    neighbors = [transformed[0, 0, 1], transformed[0, 1, 0],
                 transformed[0, 0, 1].conj(), transformed[0, ny - 1, 0]]
    if nz > 1:
        neighbors += [transformed[1, 0, 0], transformed[nz - 1, 0, 0]]
    transformed[0, 0, 0] = sum(neighbors).real / np.float32(len(neighbors))
    return transformed


def _windowed_basis_kernels(cp):
    global _WINDOWED_BASIS_KERNELS
    if _WINDOWED_BASIS_KERNELS is None:
        _WINDOWED_BASIS_KERNELS = (
            cp.ElementwiseKernel(
                "float32 value, float32 zy, float32 x", "float32 field",
                "field = value * (zy * x);", "nrss_window_scalar_basis",
            ),
            cp.ElementwiseKernel(
                "float32 phi, float32 ua, float32 ub, float32 zy, float32 x", "float32 field",
                "field = ((phi * ua) * ub) * (zy * x);", "nrss_window_tensor_basis",
            ),
        )
    return _WINDOWED_BASIS_KERNELS


def build_energy_basis(
    runtime, morphology, cp, materials, energies, window, mode, execution, *, shape_override=None, recorder=None,
):
    detector = runtime._detector_geometry(morphology, cp, shape_override=shape_override)
    shape = tuple(morphology.NumZYX)
    real_transform = mode != "compact_c2c"
    virtual = None
    virtual_mode = None
    logical_window = window
    if shape_override is None:
        indices = _detector_endpoint_union(runtime, morphology, cp, energies, detector)
        native, reflect = _native_indices(runtime, cp, shape, indices, real_transform)
    else:
        from ._cupy_virtual_z import VirtualZEvaluator, dc_indices, sparse_endpoint_union

        with recorder.range("zpad_plan"):
            indices = sparse_endpoint_union(runtime, morphology, cp, energies, detector, shape_override)
            stencil = cp.asarray(dc_indices(shape_override))
            indices = cp.union1d(indices, stencil)
            native, reflect = _native_indices(runtime, cp, shape_override, indices, real_transform)
            dc_positions = cp.searchsorted(indices, stencil)
            options = morphology.backend_options
            if "virtual_3d_pad_to" in options:
                from ._cupy_virtual_xyz import VirtualXYZEvaluator

                virtual_mode = "3d"
                virtual = VirtualXYZEvaluator(cp, shape, shape_override, native, reflect, dc_positions)
                if window is not None:
                    wy, wx = window
                    y, x = virtual.offset_y, virtual.offset_x
                    window = (wy[:, y:y + shape[1], :], wx[:, :, x:x + shape[2]])
            else:
                virtual_mode = "z"
                virtual = VirtualZEvaluator(
                    cp, shape[0], shape_override, native, reflect, dc_positions,
                    options.get("virtual_z_method", "sum"),
                )
    angle_family = runtime._angle_family_plan(morphology)
    direct = mode == "direct_r2c"
    if direct:
        angle = angle_family.angles[0]
        components = (0, 1, 2)
        diagonal_rows = [0, 1]
        weights = np.asarray((angle.mx, angle.my), dtype=np.float32)
        polarization_field = cp.ElementwiseKernel(
            "float32 phi, float32 ux, float32 uy, float32 component, float32 ex, float32 ey",
            "float32 field",
            "field = phi * component * (ux * ex + uy * ey)",
            "nrss_direct_polarization_basis_field",
        )
    else:
        components = angle_family.required_nt_components
        diagonal_rows = [row for row, component in enumerate(components) if component in (0, 3)]
        weights = None
    field_count = sum(1 if m.is_full_isotropic else 2 + len(components) for m in materials)
    values = cp.empty((field_count, indices.size), dtype=cp.complex64)
    coefficients = np.zeros((len(energies), len(components), field_count), dtype=np.complex64)
    buffer = cp.empty(shape, dtype=cp.float32 if real_transform else cp.complex64)
    plan = None
    if not real_transform:
        from cupyx.scipy.fftpack import get_fft_plan

        plan = get_fft_plan(buffer, axes=(0, 1, 2), value_type="C2C")
    fused_window = mode == "compact_r2c" and window is not None
    if fused_window:
        window_scalar, window_tensor = _windowed_basis_kernels(cp)

    def capture(row):
        if window is not None and not fused_window:
            cp.multiply(buffer, window, out=buffer)
        if virtual is not None:
            virtual.capture(cp, buffer, values[row], recorder)
            return
        transformed = _transform_basis(runtime, cp, buffer, plan, shape, execution)
        samples = transformed.ravel()[native]
        if reflect is not None:
            samples = cp.where(reflect, samples.conj(), samples)
        values[row] = samples

    fill_material = morphology.backend_options.get("virtual_pad_material") if virtual is not None else None

    def capture_scalar(row, field, *, fill_box=False):
        if fill_box:
            with recorder.range("zpad_fill"):
                cp.subtract(field, np.float32(1.0), out=buffer)
                if fused_window:
                    window_scalar(buffer, *window, buffer)
            virtual.capture(cp, buffer, values[row], recorder, fill_box=True, window=logical_window)
            return
        if fused_window:
            window_scalar(field, *window, buffer)
        else:
            buffer[...] = field
        capture(row)

    row = 0
    for material in materials:
        optical = np.asarray([runtime._material_optical_scalars(material, energy) for energy in energies])
        diagonal = optical[:, :2, None]
        if weights is not None:
            diagonal = diagonal * weights
        coefficients[:, diagonal_rows, row] = diagonal[:, 0]
        capture_scalar(row, material.Vfrac, fill_box=material.materialID == fill_material)
        row += 1
        if material.is_full_isotropic:
            continue

        if runtime._has_precomputed_runtime_reusables(material):
            phi = material.phi_a
            axes = material.sx, material.sy, material.sz
        else:
            phi = material.Vfrac * material.S
            axes = runtime._orientation_components(material, cp)

        coefficients[:, diagonal_rows, row] = diagonal[:, 1]
        capture_scalar(row, phi)
        row += 1
        for component_row, component in enumerate(components):
            if direct:
                polarization_field(phi, axes[0], axes[1], axes[component], angle.mx, angle.my, buffer)
            else:
                a, b = _TENSOR_AXES[component]
                if fused_window:
                    window_tensor(phi, axes[a], axes[b], *window, buffer)
                else:
                    cp.multiply(phi, axes[a], out=buffer)
                    cp.multiply(buffer, axes[b], out=buffer)
            coefficients[:, component_row, row] = optical[:, 2]
            capture(row)
            row += 1
        del phi, axes
    return EnergyBasis(
        indices, values, cp.asarray(coefficients), components, shape_override,
        virtual_method=virtual.method if virtual is not None else None,
        virtual_mode=virtual_mode,
        virtual_residues=len(virtual.residues) if virtual is not None else 0,
        xyz_xy_tile_planes=virtual.tile_depth if virtual_mode == "3d" else None,
        virtual_energy_masks=len(set(energies)) if virtual is not None else 0,
    )


def mix_endpoint_tensor(cp, basis, energy_index, endpoints):
    """Return complex tensor components in the requested Igor endpoint order."""
    positions = cp.searchsorted(basis.indices, endpoints)
    return basis.coefficients[energy_index] @ basis.values[:, positions]


def project_energy(runtime, morphology, cp, basis, energy_index, energy, angle_family):
    shape_override = basis.shape_override
    detector = runtime._detector_geometry(morphology, cp, shape_override=shape_override)
    try:
        projection = runtime._detector_projection_geometry(morphology, energy, cp, detector, shape_override=shape_override)
        columns, z0, z1 = _detector_columns(cp, projection)
        nx = detector.x_count
        panel_shape = (detector.y_count, nx)
        plane_size = math.prod(panel_shape)
        a = detector.qx[columns % nx]
        b = detector.qy[columns // nx]
        k = np.float32(2.0 * math.pi / (1239.84197 / energy))
        k_squared = np.float32(k * k)

        def project_endpoint(endpoints, c):
            tensor = mix_endpoint_tensor(cp, basis, energy_index, endpoints)
            tensor *= runtime._one_by_four_pi
            components = dict(zip(basis.components, tensor))
            x = (components[0], components[1], components[2]) if angle_family.needs_proj_x else None
            y = (components[1], components[3], components[4]) if angle_family.needs_proj_y else None
            return runtime._detector_projection_coefficients_from_fft_slices(
                a, b, c, k_squared, x if x is not None else y, y if y is not None else x,
            )

        panels = cp.full((3, *panel_shape), np.nan, dtype=cp.float32)
        if columns.size:
            if z0 is None:
                values = project_endpoint(columns, k + detector.qz[0])
            else:
                z0_endpoints = z0.astype(columns.dtype) * plane_size + columns
                z1_endpoints = z1.astype(columns.dtype) * plane_size + columns
                lower = project_endpoint(z0_endpoints, k + detector.qz[z0])
                upper = project_endpoint(z1_endpoints, k + detector.qz[z1])
                frac = projection.frac.ravel()[columns]
                keep = np.float32(1.0) - frac
                values = tuple(keep * lo + frac * hi for lo, hi in zip(lower, upper))
            for panel, values_at_columns in zip(panels, values):
                panel.ravel()[columns] = values_at_columns

        # The shared projector duplicates its first input for a single aligned
        # family. An aligned y direction reads that result through proj_y.
        if not angle_family.needs_proj_x:
            return None, panels[0], None
        if not angle_family.needs_proj_y:
            return panels[0], None, None
        return panels[0], panels[1], panels[2] if angle_family.needs_proj_xy else None
    finally:
        runtime._discard_detector_projection_geometry(morphology, energy, shape_override=shape_override)


def project_energy_fused(runtime, morphology, cp, basis, energy_index, energy, angles):
    """Mix and project a detector pixel directly from its cached Fourier samples."""
    global _COMPACT_TENSOR_PROJECTION_KERNEL
    if _COMPACT_TENSOR_PROJECTION_KERNEL is None:
        _COMPACT_TENSOR_PROJECTION_KERNEL = cp.RawKernel(
            _COMPACT_TENSOR_PROJECTION_SOURCE, "compact_tensor_projection",
        )
    shape_override = basis.shape_override
    detector = runtime._detector_geometry(morphology, cp, shape_override=shape_override)
    try:
        geometry = runtime._detector_projection_geometry(morphology, energy, cp, detector, shape_override=shape_override)
        panel_shape = (detector.y_count, detector.x_count)
        plane_size = math.prod(panel_shape)
        channels = int(angles.needs_proj_x) + int(angles.needs_proj_y) + int(angles.needs_proj_xy)
        panels = cp.empty((channels, *panel_shape), dtype=cp.float32)
        no_array = np.uintp(0)
        k = np.float32(2.0 * math.pi / (1239.84197 / energy))
        threads = 128
        _COMPACT_TENSOR_PROJECTION_KERNEL(
            ((plane_size + threads - 1) // threads,), (threads,),
            (
                geometry.valid,
                geometry.safe_z0 if geometry.safe_z0 is not None else no_array,
                geometry.safe_z1 if geometry.safe_z1 is not None else no_array,
                geometry.frac if geometry.frac is not None else no_array,
                detector.qx, detector.qy, detector.qz, k,
                basis.indices, basis.values, basis.coefficients[energy_index], panels,
                np.int64(basis.indices.size), np.int32(basis.values.shape[0]), np.int32(len(basis.components)),
                np.int32(channels), np.int32(detector.x_count), np.int32(detector.z_count), np.int64(plane_size),
                np.float32(runtime._one_by_four_pi), np.float32(np.nan),
            ),
        )
        if not angles.needs_proj_x:
            return None, panels[0], None
        if not angles.needs_proj_y:
            return panels[0], None, None
        return panels[0], panels[1], panels[2] if angles.needs_proj_xy else None
    finally:
        runtime._discard_detector_projection_geometry(morphology, energy, shape_override=shape_override)
