from __future__ import annotations

from dataclasses import dataclass
import gc
import math
import time
from typing import Any

import numpy as np
import xarray as xr

from .registry import BackendUnavailableError
from .runtime import BackendRuntime

_CUPY_KERNEL_CACHE: dict[str, Any] = {}


def require_cupy_modules():
    errors = []
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - exercised only when unavailable
        errors.append(f"cupy: {exc.__class__.__name__}({exc})")
        cp = None

    try:
        from cupyx.scipy import ndimage
    except Exception as exc:  # pragma: no cover - exercised only when unavailable
        errors.append(f"cupyx.scipy.ndimage: {exc.__class__.__name__}({exc})")
        ndimage = None

    if cp is None or ndimage is None:
        raise BackendUnavailableError(
            "cupy-rsoxs backend is unavailable. "
            f"Import attempts failed: {'; '.join(errors)}"
        )

    return cp, ndimage


@dataclass
class CupyScatteringResult:
    data: Any
    energies: tuple[float, ...]
    phys_size: float
    num_zyx: tuple[int, int, int]

    def to_backend_array(self):
        return self.data

    def to_xarray(self) -> xr.DataArray:
        import cupy as cp

        scattering_data = cp.asnumpy(self.data)
        ny, nx = self.num_zyx[1:]
        d = self.phys_size
        qy = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=d))
        qx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=d))
        return xr.DataArray(
            scattering_data,
            dims=["energy", "qy", "qx"],
            coords={"qy": qy, "qx": qx, "energy": list(self.energies)},
        )

    def release(self):
        self.data = None


class CupyRsoxsBackendRuntime(BackendRuntime):
    name = "cupy-rsoxs"

    _one_by_four_pi = np.float32(1.0 / (4.0 * math.pi))

    def prepare(self, morphology) -> None:
        self._validate_supported_config(morphology)
        morphology._backend_runtime_state.setdefault("prepared", True)

    def run(
        self,
        morphology,
        *,
        stdout: bool = True,
        stderr: bool = True,
        return_xarray: bool = True,
        print_vec_info: bool = False,
        validate: bool = False,
    ):
        del stdout, stderr, print_vec_info

        if validate:
            self.validate_all(morphology, quiet=True)
        else:
            self._validate_supported_config(morphology)

        cp, ndimage = require_cupy_modules()
        self.prepare(morphology)
        timings: dict[str, float] = {}

        total_start = time.perf_counter()
        energies = tuple(float(energy) for energy in morphology.Energies)
        projections = []
        window = self._window_tensor(morphology, cp)

        for energy in energies:
            energy_start = time.perf_counter()
            projection = self._run_single_energy(
                morphology=morphology,
                energy=energy,
                cp=cp,
                ndimage=ndimage,
                window=window,
                timings=timings,
            )
            projections.append(projection)
            cp.cuda.Stream.null.synchronize()
            timings[f"energy_{energy:.4f}_seconds"] = time.perf_counter() - energy_start

        result_data = cp.stack(projections, axis=0)
        result = CupyScatteringResult(
            data=result_data,
            energies=energies,
            phys_size=float(morphology.PhysSize),
            num_zyx=tuple(int(v) for v in morphology.NumZYX),
        )

        morphology._backend_result = result
        morphology.scatteringPattern = result
        morphology._backend_timings = {
            **timings,
            "total_seconds": time.perf_counter() - total_start,
        }
        morphology._simulated = True
        morphology._lock_results()

        if return_xarray:
            return result.to_xarray()
        return result

    def validate_all(self, morphology, *, quiet: bool = True) -> None:
        morphology.check_materials(quiet=quiet)
        self._validate_supported_config(morphology)
        if not quiet:
            print("CuPy backend validation completed successfully.")

    def release(self, morphology) -> None:
        result = getattr(morphology, "_backend_result", None)
        if result is not None and hasattr(result, "release"):
            result.release()
        morphology._backend_result = None
        morphology.scatteringPattern = None
        morphology._backend_runtime_state.clear()
        try:
            import cupy as cp
        except Exception:
            return
        gc.collect()
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    def _validate_supported_config(self, morphology) -> None:
        if morphology.MorphologyType != 0:
            raise NotImplementedError("cupy-rsoxs currently supports Euler morphology only.")
        if morphology.CaseType != 0:
            raise NotImplementedError("cupy-rsoxs currently supports CaseType=0 only.")
        if morphology.ReferenceFrame != 1:
            raise NotImplementedError("cupy-rsoxs currently supports Lab reference frame only.")
        if morphology.RotMask not in (0, 1):
            raise NotImplementedError("cupy-rsoxs currently supports RotMask values 0 and 1 only.")
        if morphology.EwaldsInterpolation != 1:
            raise NotImplementedError("cupy-rsoxs currently supports trilinear Ewald interpolation only.")
        if morphology.PhysSize is None:
            raise ValueError("Morphology.PhysSize must be set before running cupy-rsoxs.")
        if morphology.NumZYX is None:
            raise ValueError("Morphology.NumZYX must be set before running cupy-rsoxs.")

        start_angle, increment_angle, end_angle = map(float, morphology.EAngleRotation)
        if increment_angle == 0.0 and start_angle != end_angle:
            raise ValueError(
                "EAngleRotation with zero increment must use identical start/end angles."
            )

    def _window_tensor(self, morphology, cp):
        if morphology.WindowingType == 0:
            return None
        z, y, x = morphology.NumZYX
        wz = cp.asarray(np.hanning(z), dtype=cp.float32)[:, None, None]
        wy = cp.asarray(np.hanning(y), dtype=cp.float32)[None, :, None]
        wx = cp.asarray(np.hanning(x), dtype=cp.float32)[None, None, :]
        return wz * wy * wx

    def _run_single_energy(self, morphology, energy, cp, ndimage, window, timings):
        num_angles = self._num_angles(morphology)
        angle_start = time.perf_counter()
        nt = self._compute_nt_components(morphology, energy, cp)
        angle_projections = self._project_from_fft_nt(
            morphology=morphology,
            energy=energy,
            cp=cp,
            ndimage=ndimage,
            fft_nt=self._compute_fft_nt_components(nt=nt, cp=cp, window=window),
            num_angles=num_angles,
        )
        del nt
        cp.cuda.Stream.null.synchronize()
        timings["angle_loop_seconds"] = timings.get("angle_loop_seconds", 0.0) + (
            time.perf_counter() - angle_start
        )
        return angle_projections

    def _num_angles(self, morphology) -> int:
        start_angle, increment_angle, end_angle = map(float, morphology.EAngleRotation)
        return int(round((end_angle - start_angle) / increment_angle + 1.0)) if increment_angle else 1

    def _angles_radians(self, morphology):
        start_angle, increment_angle, end_angle = map(float, morphology.EAngleRotation)
        num_angles = self._num_angles(morphology)
        if num_angles == 1:
            return (math.radians(start_angle),)
        return tuple(math.radians(start_angle + increment_angle * idx) for idx in range(num_angles))

    def _rotation_transforms(self, morphology, cp):
        cache_key = (
            "angle_rotation_transforms",
            tuple(int(v) for v in morphology.NumZYX),
            tuple(round(angle, 12) for angle in self._angles_radians(morphology)),
        )
        transforms = morphology._backend_runtime_state.get(cache_key)
        if transforms is not None:
            return transforms

        _, height, width = map(int, morphology.NumZYX)
        transforms = tuple(
            tuple(
                cp.asarray(part, dtype=cp.float64)
                for part in self._affine_inverse_yx_from_forward_xy(
                    self._rotation_forward_matrix_xy(width=width, height=height, angle_radians=angle)
                )
            )
            for angle in self._angles_radians(morphology)
        )
        morphology._backend_runtime_state[cache_key] = transforms
        return transforms

    def _rotation_forward_matrix_xy(self, *, width, height, angle_radians):
        alpha = math.cos(angle_radians)
        beta = math.sin(angle_radians)
        return np.asarray(
            [
                [
                    alpha,
                    beta,
                    ((1.0 - alpha) * width / 2.0) - (beta * height / 2.0),
                ],
                [
                    -beta,
                    alpha,
                    (beta * width / 2.0) + ((1.0 - alpha) * height / 2.0),
                ],
            ],
            dtype=np.float64,
        )

    def _affine_inverse_yx_from_forward_xy(self, forward_xy):
        linear_xy = forward_xy[:, :2]
        offset_xy = forward_xy[:, 2]
        inverse_xy = np.linalg.inv(linear_xy)
        swap_xy_yx = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        matrix_yx = swap_xy_yx @ inverse_xy @ swap_xy_yx
        offset_yx = -(swap_xy_yx @ inverse_xy @ offset_xy)
        return matrix_yx, offset_yx

    def _apply_affine_transform(self, projection, ndimage, matrix_yx, offset_yx, *, cval):
        return ndimage.affine_transform(
            projection,
            matrix_yx,
            offset=offset_yx,
            output_shape=projection.shape,
            order=1,
            mode="constant",
            cval=cval,
            prefilter=False,
        )

    def _finalize_rotation_average(self, cp, projection_average, valid_counts, num_angles):
        if valid_counts is None:
            return projection_average / np.float32(num_angles)
        return cp.where(
            valid_counts == 0,
            np.float32(0.0),
            projection_average / valid_counts.astype(cp.float32),
        )

    def _material_optics(self, material, energy):
        d_para, b_para, d_perp, b_perp = material.opt_constants[energy]
        npar = np.complex64(complex(1.0 - float(d_para), float(b_para)))
        nper = np.complex64(complex(1.0 - float(d_perp), float(b_perp)))
        return npar, nper

    def _compute_nt_components(self, morphology, energy, cp):
        start = time.perf_counter()
        shape = tuple(int(v) for v in morphology.NumZYX)
        nt = cp.zeros((5, *shape), dtype=cp.complex64)

        for material in morphology.materials.values():
            npar, nper = self._material_optics(material, energy)
            nsum_sq = np.complex64((npar + 2.0 * nper) ** 2)
            npar_sq = np.complex64(npar * npar)
            nper_sq = np.complex64(nper * nper)

            vfrac = material.Vfrac
            s = material.S
            theta = material.theta
            psi = material.psi

            phi_a = vfrac * s
            phi_ui = vfrac - phi_a
            sin_theta = cp.sin(theta, dtype=cp.float32)
            sx = cp.cos(psi, dtype=cp.float32) * sin_theta
            sy = cp.sin(psi, dtype=cp.float32) * sin_theta
            sz = cp.cos(theta, dtype=cp.float32)

            nt[0] += phi_a * (
                npar_sq * sx * sx + nper_sq * (sy * sy + sz * sz)
            ) + (phi_ui * nsum_sq) / np.float32(9.0) - vfrac
            nt[1] += phi_a * (npar_sq - nper_sq) * sx * sy
            nt[2] += phi_a * (npar_sq - nper_sq) * sx * sz
            nt[3] += phi_a * (
                npar_sq * sy * sy + nper_sq * (sx * sx + sz * sz)
            ) + (phi_ui * nsum_sq) / np.float32(9.0) - vfrac
            nt[4] += phi_a * (npar_sq - nper_sq) * sy * sz

            del phi_a, phi_ui, sin_theta, sx, sy, sz

        cp.cuda.Stream.null.synchronize()
        morphology._backend_timings["nt_seconds"] = morphology._backend_timings.get("nt_seconds", 0.0) + (
            time.perf_counter() - start
        )
        return nt

    def _project_from_nt(self, morphology, energy, cp, ndimage, nt, window, num_angles):
        projection_average = None
        valid_counts = None
        use_rot_mask = bool(morphology.RotMask)
        for angle, (matrix_yx, offset_yx) in zip(
            self._angles_radians(morphology),
            self._rotation_transforms(morphology, cp),
        ):
            p_x, p_y, p_z = self._polarization_from_nt(nt, angle, cp)
            projection = self._projection_from_polarization(
                morphology=morphology,
                energy=energy,
                cp=cp,
                p_x=p_x,
                p_y=p_y,
                p_z=p_z,
                window=window,
            )
            rotated = self._apply_affine_transform(
                projection,
                ndimage,
                matrix_yx,
                offset_yx,
                cval=np.nan,
            )
            if use_rot_mask:
                valid = cp.isfinite(rotated)
                valid_counts = (
                    valid.astype(cp.int32)
                    if valid_counts is None
                    else valid_counts + valid.astype(cp.int32)
                )
                rotated = cp.where(valid, rotated, np.float32(0.0))
                del valid
            projection_average = rotated if projection_average is None else projection_average + rotated
            del p_x, p_y, p_z, projection, rotated
        return self._finalize_rotation_average(cp, projection_average, valid_counts, num_angles)

    def _compute_fft_nt_components(self, nt, cp, window):
        for idx in range(nt.shape[0]):
            component = nt[idx]
            if window is not None:
                cp.multiply(component, window, out=component)
            fft_component = cp.fft.fftn(component)
            self._replace_dc_component(fft_component)
            self._igor_shift(fft_component, cp, out=nt[idx])
            del component, fft_component
        return nt

    def _project_from_fft_nt(self, morphology, energy, cp, ndimage, fft_nt, num_angles):
        proj_x, proj_y, proj_xy = self._projection_coefficients_from_fft_nt(
            morphology=morphology,
            energy=energy,
            cp=cp,
            fft_nt=fft_nt,
        )
        projection_average = None
        valid_counts = None
        use_rot_mask = bool(morphology.RotMask)
        for angle, (matrix_yx, offset_yx) in zip(
            self._angles_radians(morphology),
            self._rotation_transforms(morphology, cp),
        ):
            cos_angle = np.float32(math.cos(angle))
            sin_angle = np.float32(math.sin(angle))
            projection = (
                proj_x * (cos_angle * cos_angle)
                + proj_y * (sin_angle * sin_angle)
                + proj_xy * (cos_angle * sin_angle)
            )
            rotated = self._apply_affine_transform(
                projection,
                ndimage,
                matrix_yx,
                offset_yx,
                cval=np.nan,
            )
            if use_rot_mask:
                valid = cp.isfinite(rotated)
                valid_counts = (
                    valid.astype(cp.int32)
                    if valid_counts is None
                    else valid_counts + valid.astype(cp.int32)
                )
                rotated = cp.where(valid, rotated, np.float32(0.0))
                del valid
            projection_average = rotated if projection_average is None else projection_average + rotated
            del projection, rotated
        del proj_x, proj_y, proj_xy
        return self._finalize_rotation_average(cp, projection_average, valid_counts, num_angles)

    def _projection_coefficients_from_fft_nt(self, morphology, energy, cp, fft_nt):
        basis_x = (
            fft_nt[0] * self._one_by_four_pi,
            fft_nt[1] * self._one_by_four_pi,
            fft_nt[2] * self._one_by_four_pi,
        )
        basis_y = (
            fft_nt[1] * self._one_by_four_pi,
            fft_nt[3] * self._one_by_four_pi,
            fft_nt[4] * self._one_by_four_pi,
        )

        proj_x = self._projection_from_fft_polarization(
            morphology=morphology,
            energy=energy,
            cp=cp,
            fft_x=basis_x[0],
            fft_y=basis_x[1],
            fft_z=basis_x[2],
        )
        proj_y = self._projection_from_fft_polarization(
            morphology=morphology,
            energy=energy,
            cp=cp,
            fft_x=basis_y[0],
            fft_y=basis_y[1],
            fft_z=basis_y[2],
        )
        proj_xy = self._projection_from_fft_polarization(
            morphology=morphology,
            energy=energy,
            cp=cp,
            fft_x=basis_x[0] + basis_y[0],
            fft_y=basis_x[1] + basis_y[1],
            fft_z=basis_x[2] + basis_y[2],
        )
        proj_xy = proj_xy - proj_x - proj_y
        return proj_x, proj_y, proj_xy

    def _project_from_direct_polarization(self, morphology, energy, cp, ndimage, window, num_angles):
        projection_average = None
        valid_counts = None
        use_rot_mask = bool(morphology.RotMask)
        for angle, (matrix_yx, offset_yx) in zip(
            self._angles_radians(morphology),
            self._rotation_transforms(morphology, cp),
        ):
            p_x, p_y, p_z = self._compute_direct_polarization(morphology, energy, angle, cp)
            projection = self._projection_from_polarization(
                morphology=morphology,
                energy=energy,
                cp=cp,
                p_x=p_x,
                p_y=p_y,
                p_z=p_z,
                window=window,
            )
            rotated = self._apply_affine_transform(
                projection,
                ndimage,
                matrix_yx,
                offset_yx,
                cval=np.nan,
            )
            if use_rot_mask:
                valid = cp.isfinite(rotated)
                valid_counts = (
                    valid.astype(cp.int32)
                    if valid_counts is None
                    else valid_counts + valid.astype(cp.int32)
                )
                rotated = cp.where(valid, rotated, np.float32(0.0))
                del valid
            projection_average = rotated if projection_average is None else projection_average + rotated
            del p_x, p_y, p_z, projection, rotated
        return self._finalize_rotation_average(cp, projection_average, valid_counts, num_angles)

    def _polarization_from_nt(self, nt, angle, cp):
        mx = np.float32(math.cos(angle))
        my = np.float32(math.sin(angle))
        p_x = (nt[0] * mx + nt[1] * my) * self._one_by_four_pi
        p_y = (nt[1] * mx + nt[3] * my) * self._one_by_four_pi
        p_z = (nt[2] * mx + nt[4] * my) * self._one_by_four_pi
        return p_x, p_y, p_z

    def _compute_direct_polarization(self, morphology, energy, angle, cp):
        shape = tuple(int(v) for v in morphology.NumZYX)
        p_x = cp.zeros(shape, dtype=cp.complex64)
        p_y = cp.zeros(shape, dtype=cp.complex64)
        p_z = cp.zeros(shape, dtype=cp.complex64)
        mx = np.float32(math.cos(angle))
        my = np.float32(math.sin(angle))

        for material in morphology.materials.values():
            npar, nper = self._material_optics(material, energy)
            nsum_sq = np.complex64((npar + 2.0 * nper) ** 2)
            npar_sq = np.complex64(npar * npar)
            nper_sq = np.complex64(nper * nper)

            vfrac = material.Vfrac
            s = material.S
            theta = material.theta
            psi = material.psi

            phi_a = vfrac * s
            phi_ui = vfrac - phi_a
            sin_theta = cp.sin(theta, dtype=cp.float32)
            sx = cp.cos(psi, dtype=cp.float32) * sin_theta
            sy = cp.sin(psi, dtype=cp.float32) * sin_theta
            sz = cp.cos(theta, dtype=cp.float32)

            t0 = phi_a * (
                npar_sq * sx * sx + nper_sq * (sy * sy + sz * sz)
            ) + (phi_ui * nsum_sq) / np.float32(9.0) - vfrac
            t1 = phi_a * (npar_sq - nper_sq) * sx * sy
            t2 = phi_a * (npar_sq - nper_sq) * sx * sz
            t3 = phi_a * (
                npar_sq * sy * sy + nper_sq * (sx * sx + sz * sz)
            ) + (phi_ui * nsum_sq) / np.float32(9.0) - vfrac
            t4 = phi_a * (npar_sq - nper_sq) * sy * sz

            p_x += t0 * mx + t1 * my
            p_y += t1 * mx + t3 * my
            p_z += t2 * mx + t4 * my

            del phi_a, phi_ui, sin_theta, sx, sy, sz, t0, t1, t2, t3, t4

        p_x *= self._one_by_four_pi
        p_y *= self._one_by_four_pi
        p_z *= self._one_by_four_pi
        return p_x, p_y, p_z

    def _projection_from_polarization(self, morphology, energy, cp, p_x, p_y, p_z, window):
        start = time.perf_counter()
        if window is not None:
            p_x = p_x * window
            p_y = p_y * window
            p_z = p_z * window

        fft_x = cp.fft.fftn(p_x)
        fft_y = cp.fft.fftn(p_y)
        fft_z = cp.fft.fftn(p_z)
        self._replace_dc_component(fft_x)
        self._replace_dc_component(fft_y)
        self._replace_dc_component(fft_z)
        fft_x = self._igor_shift(fft_x, cp)
        fft_y = self._igor_shift(fft_y, cp)
        fft_z = self._igor_shift(fft_z, cp)

        scatter3d = self._compute_scatter3d(
            morphology=morphology,
            energy=energy,
            cp=cp,
            p_x=fft_x,
            p_y=fft_y,
            p_z=fft_z,
        )
        projection = self._project_scatter3d(morphology, energy, cp, scatter3d)

        del fft_x, fft_y, fft_z, scatter3d
        cp.cuda.Stream.null.synchronize()
        morphology._backend_timings["fft_projection_seconds"] = morphology._backend_timings.get(
            "fft_projection_seconds", 0.0
        ) + (time.perf_counter() - start)
        return projection

    def _projection_from_fft_polarization(self, morphology, energy, cp, fft_x, fft_y, fft_z):
        start = time.perf_counter()
        scatter3d = self._compute_scatter3d(
            morphology=morphology,
            energy=energy,
            cp=cp,
            p_x=fft_x,
            p_y=fft_y,
            p_z=fft_z,
        )
        projection = self._project_scatter3d(morphology, energy, cp, scatter3d)
        del scatter3d
        cp.cuda.Stream.null.synchronize()
        morphology._backend_timings["fft_projection_seconds"] = morphology._backend_timings.get(
            "fft_projection_seconds", 0.0
        ) + (time.perf_counter() - start)
        return projection

    def _replace_dc_component(self, arr):
        z, y, x = arr.shape
        neighbors = [arr[0, 0, 1], arr[0, 1, 0], arr[0, 0, x - 1], arr[0, y - 1, 0]]
        if z > 1:
            neighbors.extend([arr[1, 0, 0], arr[z - 1, 0, 0]])
        arr[0, 0, 0] = sum(neighbors) / np.float32(len(neighbors))

    def _igor_shift_kernel(self, cp):
        kernel = _CUPY_KERNEL_CACHE.get("igor_shift_complex64")
        if kernel is None:
            kernel = cp.RawKernel(
                r"""
                extern "C" __global__
                void igor_shift_complex64(
                    const float2* input,
                    float2* output,
                    const int* z_order,
                    const int* y_order,
                    const int* x_order,
                    const int zdim,
                    const int ydim,
                    const int xdim,
                    const unsigned long long total
                ) {
                    const unsigned long long idx =
                        (unsigned long long)blockDim.x * (unsigned long long)blockIdx.x
                        + (unsigned long long)threadIdx.x;
                    if (idx >= total) {
                        return;
                    }

                    const int x = (int)(idx % (unsigned long long)xdim);
                    const unsigned long long tmp = idx / (unsigned long long)xdim;
                    const int y = (int)(tmp % (unsigned long long)ydim);
                    const int z = (int)(tmp / (unsigned long long)ydim);

                    const int in_z = z_order[z];
                    const int in_y = y_order[y];
                    const int in_x = x_order[x];

                    const unsigned long long input_idx =
                        ((unsigned long long)in_z * (unsigned long long)ydim
                         + (unsigned long long)in_y) * (unsigned long long)xdim
                        + (unsigned long long)in_x;
                    output[idx] = input[input_idx];
                }
                """,
                "igor_shift_complex64",
            )
            _CUPY_KERNEL_CACHE["igor_shift_complex64"] = kernel
        return kernel

    def _igor_axis_orders(self, shape, cp):
        cache = getattr(self, "_igor_order_cache", None)
        if cache is None:
            cache = {}
            self._igor_order_cache = cache

        key = tuple(int(v) for v in shape)
        orders = cache.get(key)
        if orders is None:
            orders = tuple(self._igor_axis_order(int(length), cp) for length in key)
            cache[key] = orders
        return orders

    def _igor_axis_order(self, n, cp):
        mid = n // 2
        left = cp.arange(mid, -1, -1, dtype=cp.int32)
        if mid + 1 >= n:
            return left
        right = cp.arange(n - 1, mid, -1, dtype=cp.int32)
        return cp.concatenate((left, right))

    def _igor_shift(self, arr, cp, out=None):
        if out is None:
            out = cp.empty_like(arr)
        z_order, y_order, x_order = self._igor_axis_orders(arr.shape, cp)
        total = int(arr.size)
        threads = 256
        blocks = (total + threads - 1) // threads
        self._igor_shift_kernel(cp)(
            (blocks,),
            (threads,),
            (
                arr,
                out,
                z_order,
                y_order,
                x_order,
                np.int32(arr.shape[0]),
                np.int32(arr.shape[1]),
                np.int32(arr.shape[2]),
                np.uint64(total),
            ),
        )
        return out

    def _compute_scatter3d(self, morphology, energy, cp, p_x, p_y, p_z):
        z, y, x = map(int, morphology.NumZYX)
        qx, qy, qz = self._q_axes(morphology, cp)
        k = np.float32(2.0 * math.pi / (1239.84197 / float(energy)))
        d = np.float32(k * k)
        scatter = cp.empty((z, y, x), dtype=cp.float32)

        a = qx[None, :]
        b = qy[:, None]
        for z_index in range(z):
            c = k + qz[z_index]
            p1 = p_x[z_index]
            p2 = p_y[z_index]
            p3 = p_z[z_index]

            term1 = (-a * a + d) * p1 - a * (b * p2 + c * p3)
            term2 = -(a * b) * p1 + (-b * b + d) * p2 - b * c * p3
            term3 = -(a * c) * p1 - b * c * p2 + (-c * c + d) * p3
            scatter[z_index] = (
                term1.real * term1.real
                + term1.imag * term1.imag
                + term2.real * term2.real
                + term2.imag * term2.imag
                + term3.real * term3.real
                + term3.imag * term3.imag
            )
            del p1, p2, p3, term1, term2, term3
        return scatter

    def _project_scatter3d(self, morphology, energy, cp, scatter3d):
        z, y, x = map(int, morphology.NumZYX)
        qx, qy, qz = self._q_axes(morphology, cp)
        k = np.float32(2.0 * math.pi / (1239.84197 / float(energy)))
        x_idx = cp.arange(x, dtype=cp.int32)[None, :]
        y_idx = cp.arange(y, dtype=cp.int32)[:, None]
        qx_grid = qx[None, :]
        qy_grid = qy[:, None]
        val = k * k - qx_grid * qx_grid - qy_grid * qy_grid

        projection = cp.full((y, x), np.float32(np.nan), dtype=cp.float32)
        valid = (val >= 0) & (x_idx != (x - 1)) & (y_idx != (y - 1))
        if z == 1:
            projection = cp.where(valid, scatter3d[0], projection)
            return projection

        pos_z = -k + cp.sqrt(cp.where(valid, val, 0), dtype=cp.float32)
        dz = qz[1] - qz[0]
        z_float = (pos_z - qz[0]) / dz
        z0 = cp.floor(z_float).astype(cp.int32)
        z1 = z0 + 1
        valid &= z0 >= 0
        valid &= z1 < z

        safe_z0 = cp.clip(z0, 0, z - 1)
        safe_z1 = cp.clip(z1, 0, z - 1)
        frac = z_float - safe_z0.astype(cp.float32)

        data1 = scatter3d[safe_z0, y_idx, x_idx]
        data2 = scatter3d[safe_z1, y_idx, x_idx]
        interp = (np.float32(1.0) - frac) * data1 + frac * data2
        projection = cp.where(valid, interp, projection)
        return projection

    def _q_axes(self, morphology, cp):
        z, y, x = map(int, morphology.NumZYX)
        phys = np.float32(morphology.PhysSize)
        start = np.float32(-math.pi / float(phys))
        qx = start + cp.arange(x, dtype=cp.float32) * np.float32((2.0 * math.pi / float(phys)) / max(x - 1, 1))
        qy = start + cp.arange(y, dtype=cp.float32) * np.float32((2.0 * math.pi / float(phys)) / max(y - 1, 1))
        if z == 1:
            qz = cp.asarray([0.0], dtype=cp.float32)
        else:
            qz = start + cp.arange(z, dtype=cp.float32) * np.float32((2.0 * math.pi / float(phys)) / max(z - 1, 1))
        return qx, qy, qz
