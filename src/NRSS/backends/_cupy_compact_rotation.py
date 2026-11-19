"""Fused tensor weighting, bilinear rotation, and angle accumulation."""

import numpy as np


_ROTATION_KERNEL = None


_ROTATION_SOURCE = r"""
__device__ inline float weighted_pixel(
    const long long index, const float *x, const float *y, const float *xy,
    const int family, const float *weights
) {
    if (family == 0) {
        return x[index];
    }
    if (family == 1) {
        return y[index];
    }
    return x[index] * weights[0] + y[index] * weights[1] + xy[index] * weights[2];
}

extern "C" __global__ void compact_rotate_accumulate(
    const float *x, const float *y, const float *xy,
    const double *transforms, const float *weights, const int *families,
    float *output, const int height, const int width, const int angles,
    const bool mask, const float nan_value
) {
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (long long)height * width) {
        return;
    }
    const int iy = index / width, ix = index % width;
    float sum = 0.0f;
    int count = 0;
    for (int angle = 0; angle < angles; ++angle) {
        const int flags = families[angle];
        const int family = flags & 3;
        const float *const w = weights + 3 * angle;
        float pixel = nan_value;
        if (flags & 4) {
            pixel = weighted_pixel(index, x, y, xy, family, w);
        } else {
            const double *const m = transforms + 6 * angle;
            const double sample_y = fma(m[1], (double)ix, m[0] * iy) + m[2];
            const double sample_x = fma(m[4], (double)ix, m[3] * iy) + m[5];
            if (sample_y >= 0.0 && sample_y <= height - 1 &&
                sample_x >= 0.0 && sample_x <= width - 1) {
                const int y0 = (int)sample_y, x0 = (int)sample_x;
                const double fy = sample_y - y0, fx = sample_x - x0;
                const int y_count = fy == 0.0 ? 1 : 2;
                const int x_count = fx == 0.0 ? 1 : 2;
                pixel = 0.0f;
                for (int dy = 0; dy < y_count; ++dy) {
                    const double wy = dy == 0 ? 1.0 - fy : fy;
                    for (int dx = 0; dx < x_count; ++dx) {
                        const double wx = dx == 0 ? 1.0 - fx : fx;
                        const long long source = (long long)(y0 + dy) * width + x0 + dx;
                        pixel += weighted_pixel(source, x, y, xy, family, w) * (float)(wy * wx);
                    }
                }
            }
        }
        if (mask) {
            if (isfinite(pixel)) {
                sum += pixel;
                ++count;
            }
        } else {
            if (isnan(pixel)) {
                output[index] = nan_value;
                return;
            }
            sum += pixel;
        }
    }
    output[index] = mask ? (count > 0 ? sum / count : 0.0f) : sum / angles;
}
"""


def _rotation_parameters(runtime, morphology, cp, angle_family, shape):
    height, width = shape
    angles = angle_family.angles
    key = ("compact_rotation_parameters", shape, tuple(angle.angle_radians for angle in angles))
    params = morphology._backend_runtime_state.get(key)
    if params is not None:
        return params
    matrices = np.empty((len(angles), 2, 3), dtype=np.float64)
    weights = np.empty((len(angles), 3), dtype=np.float32)
    families = np.empty(len(angles), dtype=np.int32)
    family_codes = {"x": 0, "y": 1, "general": 2}
    for index, angle in enumerate(angles):
        forward_xy = runtime._rotation_forward_matrix_xy(
            width=width, height=height, angle_radians=angle.angle_radians,
        )
        matrix, offset = runtime._affine_inverse_yx_from_forward_xy(forward_xy)
        matrices[index, :, :2] = matrix
        matrices[index, :, 2] = offset
        weights[index] = angle.mx * angle.mx, angle.my * angle.my, angle.mx * angle.my
        families[index] = family_codes[angle.family] | (4 if angle.is_identity_rotation else 0)
    params = cp.asarray(matrices), cp.asarray(weights), cp.asarray(families)
    morphology._backend_runtime_state[key] = params
    return params


def rotate_and_accumulate(runtime, morphology, cp, proj_x, proj_y, proj_xy, angle_family):
    global _ROTATION_KERNEL
    if _ROTATION_KERNEL is None:
        _ROTATION_KERNEL = cp.RawKernel(_ROTATION_SOURCE, "compact_rotate_accumulate")
    shape = (proj_x if proj_x is not None else proj_y).shape
    matrices, weights, families = _rotation_parameters(runtime, morphology, cp, angle_family, shape)
    projection = cp.empty(shape, dtype=cp.float32)
    threads = 128
    _ROTATION_KERNEL(
        ((projection.size + threads - 1) // threads,), (threads,),
        (
            proj_x if proj_x is not None else np.uintp(0),
            proj_y if proj_y is not None else np.uintp(0),
            proj_xy if proj_xy is not None else np.uintp(0),
            matrices, weights, families, projection, np.int32(shape[0]), np.int32(shape[1]),
            np.int32(len(angle_family.angles)), np.bool_(morphology.RotMask), np.float32(np.nan),
        ),
    )
    return projection
