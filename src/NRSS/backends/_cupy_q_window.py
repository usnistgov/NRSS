"""Select requested q samples on the original Cartesian and radial grids."""

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

_CARTESIAN_WINDOW_KERNEL = None


@dataclass(frozen=True)
class PixelRectangle:
    shape: tuple[int, int]
    origin: tuple[int, int]

    @classmethod
    def from_slices(cls, slices):
        return cls(tuple(axis.stop - axis.start for axis in slices), tuple(axis.start for axis in slices))

    @property
    def slices(self):
        return tuple(slice(start, start + size) for start, size in zip(self.origin, self.shape))

    @property
    def size(self):
        return math.prod(self.shape)

    @property
    def kernel_args(self):
        return tuple(map(np.int32, (*self.origin, self.shape[1])))

    def global_columns(self, local_columns, logical_width):
        rows = self.origin[0] + local_columns // self.shape[1]
        cols = self.origin[1] + local_columns % self.shape[1]
        return rows * logical_width + cols


def _sample_rectangle(lower, upper, shape):
    # A forward-error allowance for the FP64 affine and polar coordinates.
    # Cover floor(s) and floor(s)+1 before clipping to the logical detector.
    guard = 32 * np.finfo(np.float64).eps * max(shape)
    lo = np.clip(np.floor(lower - guard), 0, np.asarray(shape) - 1).astype(int)
    hi = np.clip(np.floor(upper + guard) + 1, 0, np.asarray(shape) - 1).astype(int)
    return PixelRectangle(tuple(int(a) for a in hi - lo + 1), tuple(int(a) for a in lo))


@dataclass(frozen=True)
class ProjectionDemand:
    logical_shape: tuple[int, int]
    source: PixelRectangle
    output: PixelRectangle

    @classmethod
    def plan(cls, runtime, morphology, logical_shape, cartesian, reduction):
        angles = runtime._angle_family_plan(morphology).angles
        if len(angles) == 1 and not angles[0].is_identity_rotation:
            # A single off-axis angle still uses the incumbent ndimage path.
            return None
        if cartesian is None:
            if not reduction.source_q_count:
                return None
            center = np.asarray(reduction.center)
            extent = (reduction.source_q_count - 1) * np.asarray(reduction.radial_step_pixels)
            output = _sample_rectangle(center - extent, center + extent, logical_shape)
        else:
            output = PixelRectangle.from_slices(cartesian.slices)

        sources = [output] if any(angle.is_identity_rotation for angle in angles) else []
        y, x = output.origin
        height, width = output.shape
        corners = np.asarray(((y, x), (y + height - 1, x),
                              (y, x + width - 1), (y + height - 1, x + width - 1)))
        for angle in angles:
            if angle.is_identity_rotation:
                continue
            forward = runtime._rotation_forward_matrix_xy(
                width=logical_shape[1], height=logical_shape[0], angle_radians=angle.angle_radians,
            )
            matrix, offset = runtime._affine_inverse_yx_from_forward_xy(forward)
            samples = corners @ matrix.T + offset
            sources.append(_sample_rectangle(samples.min(axis=0), samples.max(axis=0), logical_shape))
        lower = tuple(min(region.origin[i] for region in sources) for i in range(2))
        upper = tuple(max(region.origin[i] + region.shape[i] for region in sources) for i in range(2))
        source = PixelRectangle(tuple(end - start for start, end in zip(lower, upper)), lower)
        if source.shape == output.shape == logical_shape:
            return None
        return cls(logical_shape, source, output)


def _detector_mask(radius, energy, z_dim, cutoff):
    if z_dim == 1:
        return radius <= cutoff
    wave_number = 2.0 * np.pi / (1239.84197 / energy)
    r_squared = radius * radius
    discriminant = wave_number * wave_number - r_squared
    valid = discriminant >= 0.0
    root = np.sqrt(discriminant, where=valid, out=np.full_like(radius, np.nan))
    physical_q = np.sqrt(r_squared + (root - wave_number) ** 2)
    return valid & (physical_q <= cutoff)


def _bounding_slice(axis, energies, z_dim, cutoff):
    radius = np.abs(axis)
    keep = np.zeros(axis.shape, dtype=bool)
    for energy in energies:
        keep |= _detector_mask(radius, energy, z_dim, cutoff)
    indices = np.flatnonzero(keep)
    return slice(int(indices[0]), int(indices[-1]) + 1)


@dataclass(frozen=True)
class CartesianQWindow:
    slices: tuple[slice, slice]
    transverse: Any
    logical_shape: tuple[int, int, int]
    cutoff: float

    @classmethod
    def plan(cls, qy, qx, energies, logical_shape, cutoff, xp=np):
        unique_energies = tuple(dict.fromkeys(energies))
        if logical_shape[0] == 1:
            unique_energies = unique_energies[:1]
        rows, cols = (
            _bounding_slice(axis, unique_energies, logical_shape[0], cutoff) for axis in (qy, qx)
        )
        transverse = np.hypot(qy[rows, None], qx[None, cols])
        return cls((rows, cols), xp.asarray(transverse), logical_shape, cutoff)

    @property
    def attrs(self):
        return {
            "q_max": self.cutoff,
            "q_window_offset_yx": tuple(axis.start for axis in self.slices),
            "q_window_logical_shape_zyx": self.logical_shape,
        }

    def select(self, projection, energy, cp, *, cropped=False):
        global _CARTESIAN_WINDOW_KERNEL
        if _CARTESIAN_WINDOW_KERNEL is None:
            # The host radius and the two FP64 square roots define the
            # inclusive public boundary. Keep the parent rounding points.
            _CARTESIAN_WINDOW_KERNEL = cp.ElementwiseKernel(
                "float64 radius, T sample, float64 k, float64 cutoff, bool plane, float64 invalid",
                "T selected",
                """
                bool keep = radius <= cutoff;
                if (!plane) {
                    const double r2 = radius * radius;
                    const double disc = k * k - r2;
                    keep = false;
                    if (disc >= 0.0) {
                        const double depth = sqrt(disc) - k;
                        keep = sqrt(r2 + depth * depth) <= cutoff;
                    }
                }
                selected = keep ? sample : (T)invalid;
                """,
                "nrss_cartesian_q_window",
                options=("--fmad=false",),
            )
        wave_number = np.float64(2.0 * np.pi / (1239.84197 / energy))
        return _CARTESIAN_WINDOW_KERNEL(
            self.transverse, projection if cropped else projection[self.slices],
            wave_number, np.float64(self.cutoff), np.bool_(self.logical_shape[0] == 1), np.float64(np.nan),
        )


@dataclass(frozen=True)
class RadialQWindow:
    indices: np.ndarray
    device_indices: object
    device_mask: object | None
    source_q_count: int
    cutoff: float

    @classmethod
    def plan(cls, reduction, cutoff, cp):
        if reduction.attrs.get("radial_coordinate_mode") == "per_slice_q_abs":
            allowed = np.isfinite(reduction.q_abs) & (reduction.q_abs <= cutoff)
            indices = np.flatnonzero(allowed.any(axis=0))
            device_mask = cp.asarray(allowed[:, indices])
        else:
            indices = np.flatnonzero(np.isfinite(reduction.q) & (reduction.q <= cutoff))
            device_mask = None
        device_indices = cp.asarray(indices)
        if not indices.size:
            source_q_count = 0
        elif reduction.device_q_bounds is None:
            source_q_count = int(indices[-1]) + 1
        else:
            # The upper neighbor is evaluated even at an exact lower-bin match.
            bounds = reduction.device_q_bounds[:, device_indices, :]
            source_q_count = int(bounds.max().item()) + 1
        return cls(indices, device_indices, device_mask, source_q_count, cutoff)

    def select(self, values, energy_start, cp):
        selected = cp.take(values, self.device_indices, axis=-1)
        if self.device_mask is not None:
            mask = self.device_mask[energy_start:energy_start + values.shape[0]]
            if values.ndim == 3:
                mask = mask[:, None, :]
            selected = cp.where(mask, selected, np.nan)
        return selected

    def public_plan(self, reduction, data_shape):
        return replace(
            reduction,
            result_shape=data_shape,
            q=reduction.q[self.indices],
            q_perp=None if reduction.q_perp is None else reduction.q_perp[self.indices],
            q_abs=None if reduction.q_abs is None else reduction.q_abs[:, self.indices],
            attrs={**reduction.attrs, "q_max": self.cutoff},
        )
