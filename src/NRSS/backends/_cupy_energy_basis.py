"""Per-run Fourier bases for CuPy energy scans."""

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


_TENSOR_AXES = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2))  # xx, xy, xz, yy, yz
XYZ_PRUNE_ZERO_FIELDS_KEY = "_private_xyz_prune_zero_fields"
_COMPACT_TENSOR_PROJECTION_KERNELS = {}
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
__device__ inline Coefficients project_tensor(
    float2 *tensor, const float a, const float b, const float c, const float d, const float scale, const bool cross
) {
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
    return project_tensor<components>(tensor, a, b, c, d, scale, cross);
}

extern "C" __global__ void compact_tensor_projection(
    const bool *valid, const int *z0, const int *z1, const float *fraction,
    const float *qx, const float *qy, const float *qz, const float wave_number,
#ifdef NRSS_SPAN_LOOKUP
    const int *first, const long long *offsets,
#else
    const long long *indices,
#endif
    const float2 *values, const float2 *weights,
    float *output, const long long endpoints, const int fields,
    const int components, const int channels, const int nx, const int nz,
    const long long plane_size, const float scale, const float nan_value
#ifdef NRSS_PIXEL_RECTANGLE
    , const int origin_y, const int origin_x, const int width, const long long logical_plane
#endif
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
#ifdef NRSS_PIXEL_RECTANGLE
    const int iy = origin_y + column / width, ix = origin_x + column % width;
    const long long global_column = (long long)iy * nx + ix;
    const long long plane_stride = logical_plane;
#else
    const int iy = column / nx, ix = column % nx;
    const long long global_column = column, plane_stride = plane_size;
#endif
    const float a = qx[ix], b = qy[iy];
    const float d = wave_number * wave_number;
    const int lower = nz == 1 ? 0 : z0[column];
#ifdef NRSS_SPAN_LOOKUP
    const long long lo = offsets[global_column] + lower - first[global_column];
#else
    const long long lo = index_of(indices, endpoints, lower * plane_stride + global_column);
#endif
    const bool cross = channels == 3;
    Coefficients low = components == 3
        ? project_endpoint<3>(lo, values, weights, endpoints, fields,
                              a, b, wave_number + qz[lower], d, scale, cross)
        : project_endpoint<5>(lo, values, weights, endpoints, fields,
                              a, b, wave_number + qz[lower], d, scale, cross);
    if (nz > 1) {
        const int upper = z1[column];
#ifdef NRSS_SPAN_LOOKUP
        const long long hi = lo + upper - lower;
#else
        const long long hi = index_of(indices, endpoints, upper * plane_stride + global_column);
#endif
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
    column_spans: Any = None
    endpoint_plan: dict | None = None
    zero_fields_skipped: int | None = None
    xyz_factored_kx: int | None = None
    xyz_factored_fallback: str | None = None

    @property
    def nbytes(self):
        lookup_bytes = self.column_spans.nbytes if self.column_spans is not None else 0
        return self.indices.nbytes + self.values.nbytes + self.coefficients.nbytes + lookup_bytes


def _detector_columns(cp, projection):
    columns = cp.flatnonzero(projection.valid)
    if projection.safe_z0 is None:
        return columns, None, None
    return columns, projection.safe_z0.ravel()[columns], projection.safe_z1.ravel()[columns]


def _detector_endpoint_union(runtime, morphology, cp, energies, detector, pixel_region=None):
    logical_plane = detector.y_count * detector.x_count
    plane_size = logical_plane if pixel_region is None else pixel_region.size
    region_options = {} if pixel_region is None else {"pixel_region": pixel_region}
    used = cp.zeros(detector.z_count * plane_size, dtype=cp.bool_)
    for energy in dict.fromkeys(energies):
        try:
            projection = runtime._detector_projection_geometry(morphology, energy, cp, detector, **region_options)
            columns, z0, z1 = _detector_columns(cp, projection)
            if z0 is None:
                used[columns] = True
            else:
                for indices in (z0, z1):
                    used[indices.astype(columns.dtype) * plane_size + columns] = True
        finally:
            runtime._discard_detector_projection_geometry(morphology, energy)
    indices = cp.flatnonzero(used)
    if pixel_region is not None:
        depth, columns = cp.divmod(indices, plane_size)
        indices = depth * logical_plane + pixel_region.global_columns(columns, detector.x_count)
    return indices


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


def _basis_coefficients(runtime, materials, energies, components, weights):
    diagonal_rows = [0, 1] if weights is not None else [
        row for row, component in enumerate(components) if component in (0, 3)
    ]
    counts = [1 if material.is_full_isotropic else 2 + len(components) for material in materials]
    coefficients = np.zeros((len(energies), len(components), sum(counts)), dtype=np.complex64)
    start = 0
    for material, count in zip(materials, counts):
        optical = np.asarray([runtime._material_optical_scalars(material, energy) for energy in energies])
        diagonal = optical[:, :2, None]
        if weights is not None:
            diagonal = diagonal * weights
        coefficients[:, diagonal_rows, start] = diagonal[:, 0]
        if count > 1:
            coefficients[:, diagonal_rows, start + 1] = diagonal[:, 1]
            rows = np.arange(len(components))
            coefficients[:, rows, start + 2 + rows] = optical[:, 2, None]
        start += count
    return coefficients, counts


def _nonzero_basis_rows(runtime, cp, materials, counts, coefficients):
    keep = np.any(coefficients != 0, axis=(0, 1))
    if keep.all():
        return keep

    # For finite, bounded fields a zero coefficient removes the entire
    # contribution, including the window, exterior fill, and DC stencil.
    # Preserve the incumbent behavior for malformed, unvalidated fields.
    checks = []
    start = 0
    for material, count in zip(materials, counts):
        omitted = ~keep[start:start + count]
        start += count
        if not omitted.any():
            continue
        vfrac = material.Vfrac
        checks.append(cp.all((vfrac >= 0) & (vfrac <= 1)))
        if not omitted[1:].any():
            continue
        if runtime._has_precomputed_runtime_reusables(material):
            phi = material.phi_a
            checks.append(cp.all((phi >= 0) & (phi <= 1)))
            if omitted[2:].any():
                checks.extend(cp.all(cp.abs(axis) <= 1) for axis in (material.sx, material.sy, material.sz))
        else:
            s = material.S
            checks.append(cp.all((s >= 0) & (s <= 1)))
            if omitted[2:].any():
                checks.extend(cp.all(cp.isfinite(axis)) for axis in (material.theta, material.psi))
    if not bool(cp.all(cp.stack(checks))):
        keep.fill(True)
    return keep


def build_energy_basis(
    runtime, morphology, cp, materials, energies, window, mode, execution, *,
    shape_override=None, recorder=None, xyz_plan=None, pixel_region=None,
):
    detector = runtime._detector_geometry(morphology, cp, shape_override=shape_override)
    shape = tuple(morphology.NumZYX)
    real_transform = mode != "compact_c2c"
    virtual = None
    virtual_mode = None
    logical_window = window
    column_spans = endpoint_plan = None
    region_options = {} if pixel_region is None else {"pixel_region": pixel_region}
    angle_family = runtime._angle_family_plan(morphology)
    direct = mode == "direct_r2c"
    components = (0, 1, 2) if direct else angle_family.required_nt_components
    weights = None
    if direct:
        angle = angle_family.angles[0]
        weights = np.asarray((angle.mx, angle.my), dtype=np.float32)
    coefficients, counts = _basis_coefficients(runtime, materials, energies, components, weights)
    xyz = shape_override is not None and "virtual_3d_pad_to" in morphology.backend_options
    if xyz and morphology._backend_runtime_state.get(
        XYZ_PRUNE_ZERO_FIELDS_KEY, xyz_plan.prune_zero_fields if xyz_plan is not None else False,
    ):
        with recorder.range("zpad_optical_prune"):
            keep = _nonzero_basis_rows(runtime, cp, materials, counts, coefficients)
        field_count = int(np.count_nonzero(keep))
        zero_fields_skipped = keep.size - field_count
        destinations = np.full(keep.size, -1, dtype=np.intp)
        destinations[keep] = np.arange(field_count, dtype=np.intp)
        coefficients = np.ascontiguousarray(coefficients[:, :, keep])
    else:
        field_count = coefficients.shape[2]
        zero_fields_skipped = None
        destinations = np.arange(field_count, dtype=np.intp)
    if shape_override is None:
        indices = _detector_endpoint_union(runtime, morphology, cp, energies, detector, **region_options)
        if pixel_region is not None:
            endpoint_plan = {"layout": "bitmap", "planner_arrays_bytes": pixel_region.size * detector.z_count}
        native, reflect = _native_indices(runtime, cp, shape, indices, real_transform)
    else:
        from ._cupy_virtual_z import VirtualZEvaluator, dc_indices, sparse_endpoint_union

        with recorder.range("zpad_plan"):
            options = morphology.backend_options
            stencil = dc_indices(shape_override)
            if xyz:
                from ._cupy_xyz_endpoints import XYZ_ENDPOINT_PLAN_KEY, xyz_endpoints

                indices, column_spans, endpoint_plan = xyz_endpoints(
                    runtime, morphology, cp, energies, detector, shape_override,
                    stencil, field_count, recorder,
                    requested=morphology._backend_runtime_state.get(
                        XYZ_ENDPOINT_PLAN_KEY, xyz_plan.endpoint_plan if xyz_plan is not None else "legacy",
                    ),
                    **region_options,
                )
            else:
                indices = sparse_endpoint_union(
                    runtime, morphology, cp, energies, detector, shape_override, recorder, **region_options,
                )
                indices = cp.union1d(indices, cp.asarray(stencil))
                if pixel_region is not None:
                    endpoint_plan = {"layout": "sparse"}
            native, reflect = _native_indices(runtime, cp, shape_override, indices, real_transform)
            stencil = cp.asarray(stencil)
            dc_positions = (
                cp.searchsorted(indices, stencil) if column_spans is None
                else column_spans.positions(stencil)
            )
            if xyz:
                from ._cupy_virtual_xyz import XYZ_FACTORED_XY_KEY, VirtualXYZEvaluator, full_detector_kx

                virtual_mode = "3d"
                factored = morphology._backend_runtime_state.get(
                    XYZ_FACTORED_XY_KEY, xyz_plan.factored_xy if xyz_plan is not None else False,
                )
                transform_kx = None
                if factored and pixel_region is not None:
                    transform_kx = full_detector_kx(runtime, cp, energies, detector, shape_override, stencil)
                virtual = VirtualXYZEvaluator(
                    cp, shape, shape_override, native, reflect, dc_positions,
                    factored=factored, transform_kx=transform_kx,
                )
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
    if direct:
        polarization_field = cp.ElementwiseKernel(
            "float32 phi, float32 ux, float32 uy, float32 component, float32 ex, float32 ey",
            "float32 field",
            "field = phi * component * (ux * ex + uy * ey)",
            "nrss_direct_polarization_basis_field",
        )
    values = cp.empty((field_count, indices.size), dtype=cp.complex64)
    buffer = cp.empty(shape, dtype=cp.float32 if real_transform else cp.complex64) if field_count else None
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

    start = 0
    for material, count in zip(materials, counts):
        rows = destinations[start:start + count]
        start += count
        if rows[0] >= 0:
            capture_scalar(rows[0], material.Vfrac, fill_box=material.materialID == fill_material)
        if material.is_full_isotropic:
            continue
        aligned = rows[1]
        tensor_rows = [(component, row) for component, row in zip(components, rows[2:]) if row >= 0]
        if aligned < 0 and not tensor_rows:
            continue

        if runtime._has_precomputed_runtime_reusables(material):
            phi = material.phi_a
            if tensor_rows:
                axes = material.sx, material.sy, material.sz
        else:
            phi = material.Vfrac * material.S
            if tensor_rows:
                axes = runtime._orientation_components(material, cp)

        if aligned >= 0:
            capture_scalar(aligned, phi)
        for component, row in tensor_rows:
            if direct:
                polarization_field(phi, axes[0], axes[1], axes[component], angle.mx, angle.my, buffer)
            else:
                a, b = _TENSOR_AXES[component]
                if fused_window:
                    window_tensor(phi, axes[a], axes[b], *window, buffer)
                else:
                    cp.multiply(phi, axes[a], out=buffer)
                    cp.multiply(buffer, axes[b], out=buffer)
            capture(row)
        del phi
        if tensor_rows:
            del axes
    return EnergyBasis(
        indices, values, cp.asarray(coefficients), components, shape_override,
        virtual_method=virtual.method if virtual is not None else None,
        virtual_mode=virtual_mode,
        virtual_residues=len(virtual.residues) if virtual is not None else 0,
        xyz_xy_tile_planes=virtual.tile_depth if virtual_mode == "3d" else None,
        virtual_energy_masks=len(set(energies)) if virtual is not None else 0,
        column_spans=column_spans,
        endpoint_plan=endpoint_plan,
        zero_fields_skipped=zero_fields_skipped,
        xyz_factored_kx=virtual.factored_kx.size if xyz and virtual.factored_kx is not None else None,
        xyz_factored_fallback=virtual.factored_fallback if xyz else None,
    )


def mix_endpoint_tensor(cp, basis, energy_index, endpoints):
    """Return complex tensor components in the requested Igor endpoint order."""
    positions = (
        cp.searchsorted(basis.indices, endpoints) if basis.column_spans is None
        else basis.column_spans.positions(endpoints)
    )
    return basis.coefficients[energy_index] @ basis.values[:, positions]


def projection_channels(panels, angles):
    if not angles.needs_proj_x:
        return None, panels[0], None
    if not angles.needs_proj_y:
        return panels[0], None, None
    return panels[0], panels[1], panels[2] if angles.needs_proj_xy else None


def project_energy(runtime, morphology, cp, basis, energy_index, energy, angle_family, recorder=None):
    del recorder
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
        return projection_channels(panels, angle_family)
    finally:
        runtime._discard_detector_projection_geometry(morphology, energy, shape_override=shape_override)


def project_energy_fused(runtime, morphology, cp, basis, energy_index, energy, angles, recorder, pixel_region=None):
    """Mix and project a detector pixel directly from its cached Fourier samples."""
    packed = basis.column_spans is not None
    key = (packed, pixel_region is not None)
    kernel = _COMPACT_TENSOR_PROJECTION_KERNELS.get(key)
    if kernel is None:
        options = []
        if packed:
            options.append("-DNRSS_SPAN_LOOKUP")
        if pixel_region is not None:
            options.append("-DNRSS_PIXEL_RECTANGLE")
        kernel = cp.RawKernel(
            _COMPACT_TENSOR_PROJECTION_SOURCE, "compact_tensor_projection",
            options=tuple(options),
        )
        _COMPACT_TENSOR_PROJECTION_KERNELS[key] = kernel
    shape_override = basis.shape_override
    detector = runtime._detector_geometry(morphology, cp, shape_override=shape_override)
    region_options = {} if pixel_region is None else {"pixel_region": pixel_region}
    try:
        geometry = recorder.measure(
            "zpad_projection_geometry",
            lambda: runtime._detector_projection_geometry(
                morphology, energy, cp, detector, shape_override=shape_override, **region_options,
            ),
        )
        logical_shape = (detector.y_count, detector.x_count)
        panel_shape = logical_shape if pixel_region is None else pixel_region.shape
        region_args = () if pixel_region is None else (*pixel_region.kernel_args, np.int64(math.prod(logical_shape)))
        plane_size = math.prod(panel_shape)
        channels = int(angles.needs_proj_x) + int(angles.needs_proj_y) + int(angles.needs_proj_xy)
        panels = cp.empty((channels, *panel_shape), dtype=cp.float32)
        no_array = np.uintp(0)
        k = np.float32(2.0 * math.pi / (1239.84197 / energy))
        threads = 128
        addressing = (basis.column_spans.first, basis.column_spans.offsets) if packed else (basis.indices,)
        with recorder.range("zpad_projection_kernel"):
            kernel(
                ((plane_size + threads - 1) // threads,), (threads,),
                (
                    geometry.valid,
                    geometry.safe_z0 if geometry.safe_z0 is not None else no_array,
                    geometry.safe_z1 if geometry.safe_z1 is not None else no_array,
                    geometry.frac if geometry.frac is not None else no_array,
                    detector.qx, detector.qy, detector.qz, k,
                    *addressing, basis.values, basis.coefficients[energy_index], panels,
                    np.int64(basis.indices.size), np.int32(basis.values.shape[0]), np.int32(len(basis.components)),
                    np.int32(channels), np.int32(detector.x_count), np.int32(detector.z_count), np.int64(plane_size),
                    np.float32(runtime._one_by_four_pi), np.float32(np.nan),
                    *region_args,
                ),
            )
        return projection_channels(panels, angles)
    finally:
        runtime._discard_detector_projection_geometry(morphology, energy, shape_override=shape_override)
