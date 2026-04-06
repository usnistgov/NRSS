from __future__ import annotations

from dataclasses import dataclass
import gc
import math
import os
import shutil
import time
from typing import Any

import numpy as np
import xarray as xr

from .arrays import assess_array_for_backend_runtime, coerce_array_for_backend
from .registry import BackendUnavailableError
from .runtime import BackendRuntime

_CUPY_KERNEL_CACHE: dict[str, Any] = {}
_CUPY_KERNEL_BACKEND_REPORT: dict[str, str] = {}
_CUPY_PRIVATE_TIMING_SEGMENTS_KEY = "_private_backend_timing_segments"
_CUPY_TIMED_SEGMENTS = ("A2", "B", "C", "D", "E", "F")
_CUPY_SEGMENT_MEASUREMENTS = {
    "A2": "wall_clock",
    "B": "cuda_event",
    "C": "cuda_event",
    "D": "cuda_event",
    "E": "cuda_event",
    "F": "cuda_event",
}
_HALF_BITS_TO_FLOAT_DEVICE_FUNCTION = r"""
__device__ inline float nrss_half_bits_to_float(const unsigned short h) {
    const unsigned int sign = ((unsigned int)h & 0x8000u) << 16;
    unsigned int exp = ((unsigned int)h >> 10) & 0x1Fu;
    unsigned int mant = (unsigned int)h & 0x03FFu;
    unsigned int bits = 0u;

    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;
        } else {
            exp = 127u - 15u + 1u;
            while ((mant & 0x0400u) == 0u) {
                mant <<= 1;
                exp -= 1u;
            }
            mant &= 0x03FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }

    return __uint_as_float(bits);
}
"""
_RAWKERNEL_BACKEND_OPTION_NAMES = {
    "igor_shift": "igor_shift_backend",
    "direct_polarization_generic": "direct_polarization_backend",
}


@dataclass(frozen=True)
class _RecordedEventRange:
    segment: str
    start: Any
    stop: Any


@dataclass(frozen=True)
class _RuntimeMaterialView:
    materialID: int
    opt_constants: dict[float, list[float]]
    Vfrac: Any
    S: Any
    theta: Any
    psi: Any
    is_full_isotropic: bool


@dataclass(frozen=True)
class _AnglePlan:
    angle_radians: float
    mx: np.float32
    my: np.float32
    family: str
    is_identity_rotation: bool


@dataclass(frozen=True)
class _AngleFamilyPlan:
    angles: tuple[_AnglePlan, ...]
    all_axis_aligned: bool
    required_nt_components: tuple[int, ...]
    needs_proj_x: bool
    needs_proj_y: bool
    needs_proj_xy: bool


@dataclass(frozen=True)
class _DetectorGeometry:
    qx: Any
    qy: Any
    qz: Any
    x_idx: Any
    y_idx: Any
    border_valid: Any
    radius_sq: Any
    z_count: int
    y_count: int
    x_count: int
    qz0: np.float32
    dz: np.float32


@dataclass(frozen=True)
class _DetectorProjectionGeometry:
    valid: Any
    safe_z0: Any | None
    safe_z1: Any | None
    frac: Any | None


class _NullSegmentRecorder:
    selected_segments: tuple[str, ...] = ()
    segment_measurements: dict[str, str] = {}

    def measure(self, segment: str, func):
        del segment
        return func()

    def finalize(self) -> tuple[dict[str, float], dict[str, str], str | None]:
        return {}, {}, None


class _SegmentRecorder:
    def __init__(self, cp, selected_segments: tuple[str, ...]):
        self._cp = cp
        self.selected_segments = tuple(
            segment for segment in _CUPY_TIMED_SEGMENTS if segment in selected_segments
        )
        self.segment_measurements = {
            segment: _CUPY_SEGMENT_MEASUREMENTS[segment]
            for segment in self.selected_segments
        }
        self._records: list[_RecordedEventRange] = []
        self._wall_totals: dict[str, float] = {}

    def measure(self, segment: str, func):
        if segment not in self.selected_segments:
            return func()
        measurement = self.segment_measurements[segment]
        if measurement == "wall_clock":
            start = time.perf_counter()
            result = func()
            self._cp.cuda.Stream.null.synchronize()
            self._wall_totals[segment] = self._wall_totals.get(segment, 0.0) + (
                time.perf_counter() - start
            )
            return result
        start = self._cp.cuda.Event()
        stop = self._cp.cuda.Event()
        start.record()
        result = func()
        stop.record()
        self._records.append(_RecordedEventRange(segment=segment, start=start, stop=stop))
        return result

    def finalize(self) -> tuple[dict[str, float], dict[str, str], str | None]:
        if not self._records and not self._wall_totals:
            return {}, {}, None
        totals = dict(self._wall_totals)
        if self._records:
            self._cp.cuda.Stream.null.synchronize()
            for record in self._records:
                elapsed_s = float(self._cp.cuda.get_elapsed_time(record.start, record.stop)) / 1000.0
                totals[record.segment] = totals.get(record.segment, 0.0) + elapsed_s
        measurement_modes = set(self.segment_measurements.values())
        if len(measurement_modes) == 1:
            measurement = next(iter(measurement_modes))
        else:
            measurement = "mixed"
        return totals, dict(self.segment_measurements), measurement


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
        if self._kernel_preload_stage(morphology) == "a1" and morphology.NumZYX is not None:
            cp, _ = require_cupy_modules()
            self._preload_active_rawkernels(morphology, cp, stage="a1")

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
        recorder = self._segment_recorder(morphology, cp)
        morphology._backend_timings = {}
        self._update_kernel_reports(morphology)

        energies = tuple(float(energy) for energy in morphology.Energies)
        runtime_materials = recorder.measure("A2", lambda: self._runtime_material_views(morphology, cp))
        projections = []
        window = None
        try:
            if self._kernel_preload_stage(morphology) == "a2":
                recorder.measure(
                    "A2",
                    lambda: self._preload_active_rawkernels(morphology, cp, stage="a2"),
                )
            window = recorder.measure(
                "C",
                lambda: self._window_tensor(
                    morphology,
                    cp,
                    shape_override=self._segment_c_shape_override(morphology),
                ),
            )

            for energy in energies:
                projection = self._run_single_energy(
                    morphology=morphology,
                    runtime_materials=runtime_materials,
                    energy=energy,
                    cp=cp,
                    ndimage=ndimage,
                    window=window,
                    recorder=recorder,
                )
                projections.append(projection)

            result_data = recorder.measure("F", lambda: cp.stack(projections, axis=0))
        finally:
            projections.clear()
            if window is not None:
                del window
            del runtime_materials
        result = CupyScatteringResult(
            data=result_data,
            energies=energies,
            phys_size=float(morphology.PhysSize),
            num_zyx=tuple(int(v) for v in morphology.NumZYX),
        )

        morphology._backend_result = result
        morphology.scatteringPattern = result
        self._update_kernel_reports(morphology)
        segment_seconds, segment_measurements, measurement = recorder.finalize()
        if recorder.selected_segments:
            morphology._backend_timings = {
                "measurement": measurement,
                "selected_segments": list(recorder.selected_segments),
                "segment_seconds": segment_seconds,
                "segment_measurements": segment_measurements,
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

    def _segment_recorder(self, morphology, cp):
        requested_segments = self._requested_timing_segments(morphology)
        if not requested_segments:
            return _NullSegmentRecorder()
        return _SegmentRecorder(cp=cp, selected_segments=requested_segments)

    def _requested_timing_segments(self, morphology) -> tuple[str, ...]:
        requested = morphology._backend_runtime_state.get(_CUPY_PRIVATE_TIMING_SEGMENTS_KEY, ())
        if not requested:
            return ()
        unique_segments = tuple(dict.fromkeys(str(segment) for segment in requested))
        return tuple(segment for segment in unique_segments if segment in _CUPY_TIMED_SEGMENTS)

    def _kernel_preload_stage(self, morphology) -> str:
        return str(morphology.backend_options.get("kernel_preload_stage", "off"))

    def _rawkernel_backend_option(self, morphology, family: str) -> str:
        option_name = _RAWKERNEL_BACKEND_OPTION_NAMES.get(family)
        if option_name is None:
            return "nvrtc"
        return str(morphology.backend_options.get(option_name, "nvrtc"))

    def _nvcc_path(self, cp) -> str | None:
        del cp
        nvcc_path = os.environ.get("NVCC")
        if nvcc_path:
            return nvcc_path

        nvcc_path = shutil.which("nvcc")
        if nvcc_path is None and os.path.exists("/usr/local/cuda/bin/nvcc"):
            nvcc_path = "/usr/local/cuda/bin/nvcc"
        if nvcc_path is None:
            return None
        return nvcc_path

    def _configure_cupy_nvcc_path(self, cp, nvcc_path: str) -> None:
        os.environ["NVCC"] = nvcc_path
        from cupy import _environment as cupy_environment

        cupy_environment._nvcc_path = nvcc_path

    def _resolve_requested_rawkernel_backend(
        self,
        cp,
        *,
        requested_backend: str,
        prefer_auto_nvcc: bool,
    ) -> str:
        if requested_backend == "nvrtc":
            return "nvrtc"

        nvcc_path = self._nvcc_path(cp)
        if requested_backend == "nvcc":
            if nvcc_path is None:
                return "nvrtc"
            self._configure_cupy_nvcc_path(cp, nvcc_path)
            return "nvcc"

        if requested_backend == "auto" and prefer_auto_nvcc and nvcc_path is not None:
            self._configure_cupy_nvcc_path(cp, nvcc_path)
            return "nvcc"

        return "nvrtc"

    def _record_kernel_backend(self, family: str, backend: str) -> None:
        _CUPY_KERNEL_BACKEND_REPORT[family] = backend

    def _build_rawkernel_with_fallback(
        self,
        cp,
        *,
        family: str,
        cache_key_base: str,
        source: str,
        kernel_name: str,
        requested_backend: str,
        prefer_auto_nvcc: bool = False,
    ):
        backend = self._resolve_requested_rawkernel_backend(
            cp,
            requested_backend=requested_backend,
            prefer_auto_nvcc=prefer_auto_nvcc,
        )
        primary_cache_key = f"{cache_key_base}::{backend}"
        kernel = _CUPY_KERNEL_CACHE.get(primary_cache_key)
        if kernel is not None:
            self._record_kernel_backend(family, backend)
            return kernel

        backends_to_try = (backend,)
        if backend == "nvcc":
            backends_to_try = ("nvcc", "nvrtc")

        last_exc = None
        for backend_name in backends_to_try:
            cache_key = f"{cache_key_base}::{backend_name}"
            cached = _CUPY_KERNEL_CACHE.get(cache_key)
            if cached is not None:
                self._record_kernel_backend(family, backend_name)
                return cached
            try:
                kernel = cp.RawKernel(
                    source,
                    kernel_name,
                    backend=backend_name,
                )
                kernel.compile()
            except Exception as exc:  # noqa: BLE001 - fallback path is intentional
                last_exc = exc
                if backend_name != "nvcc":
                    raise
                continue
            _CUPY_KERNEL_CACHE[cache_key] = kernel
            self._record_kernel_backend(family, backend_name)
            return kernel

        assert last_exc is not None
        raise last_exc

    def _active_rawkernel_manifest(self, morphology) -> tuple[str, ...]:
        families = ["igor_shift"]
        if self._execution_path(morphology) == "direct_polarization":
            families.append("direct_polarization_generic")
            shape_override = self._segment_c_shape_override(morphology)
            z_count = self._shape_tuple(morphology, shape_override=shape_override)[0]
            families.append(
                "direct_detector_projection_single_slice"
                if int(z_count) == 1
                else "direct_detector_projection_interpolated"
            )
        return tuple(families)

    def _kernel_preload_signature(self, morphology) -> tuple[Any, ...]:
        families = self._active_rawkernel_manifest(morphology)
        family_backends = tuple(
            (
                family,
                (
                    "nvcc_preferred"
                    if family.startswith("direct_detector_projection")
                    else self._rawkernel_backend_option(morphology, family)
                ),
            )
            for family in families
        )
        return (
            self._kernel_preload_stage(morphology),
            self._execution_path(morphology),
            tuple(int(v) for v in self._shape_tuple(morphology, self._segment_c_shape_override(morphology))),
            family_backends,
        )

    def _update_kernel_reports(self, morphology, *, preload_stage: str | None = None) -> None:
        families = self._active_rawkernel_manifest(morphology)
        last_stage = (
            preload_stage
            if preload_stage is not None
            else morphology._backend_runtime_state.get("_kernel_last_preload_stage")
        )
        morphology.last_kernel_backend_report = {
            family: _CUPY_KERNEL_BACKEND_REPORT.get(
                "direct_detector_projection" if family.startswith("direct_detector_projection") else family,
                "not_loaded",
            )
            for family in families
        }
        morphology.last_kernel_preload_report = {
            "configured_stage": self._kernel_preload_stage(morphology),
            "last_preload_stage": last_stage,
            "families": list(families),
            "kernel_backends": dict(morphology.last_kernel_backend_report),
        }

    def _preload_active_rawkernels(self, morphology, cp, *, stage: str) -> None:
        signature = self._kernel_preload_signature(morphology)
        state_key = "_kernel_preload_signature"
        if morphology._backend_runtime_state.get(state_key) == signature:
            self._update_kernel_reports(morphology, preload_stage=stage)
            return

        runtime_dtype = str(morphology._runtime_compute_contract.get("runtime_dtype", "float32"))
        for family in self._active_rawkernel_manifest(morphology):
            if family == "igor_shift":
                self._igor_shift_kernel(morphology, cp)
            elif family == "direct_polarization_generic":
                self._direct_polarization_kernel(morphology, cp, runtime_dtype)
            elif family == "direct_detector_projection_single_slice":
                self._direct_detector_projection_single_slice_kernel(cp)
            elif family == "direct_detector_projection_interpolated":
                self._direct_detector_projection_interpolated_kernel(cp)

        morphology._backend_runtime_state[state_key] = signature
        morphology._backend_runtime_state["_kernel_last_preload_stage"] = stage
        self._update_kernel_reports(morphology, preload_stage=stage)

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

    def _shape_tuple(self, morphology, shape_override=None):
        if shape_override is not None:
            return tuple(int(v) for v in shape_override)
        return tuple(int(v) for v in morphology.NumZYX)

    def _z_collapse_mode(self, morphology) -> str | None:
        return morphology.z_collapse_mode

    def _segment_c_shape_override(self, morphology):
        if self._z_collapse_mode(morphology) != "mean":
            return None
        _, y, x = self._shape_tuple(morphology)
        return (1, y, x)

    def _window_tensor(self, morphology, cp, shape_override=None):
        if morphology.WindowingType == 0:
            return None
        z, y, x = self._shape_tuple(morphology, shape_override=shape_override)
        wz = cp.asarray(np.hanning(z), dtype=cp.float32)[:, None, None]
        wy = cp.asarray(np.hanning(y), dtype=cp.float32)[None, :, None]
        wx = cp.asarray(np.hanning(x), dtype=cp.float32)[None, None, :]
        return wz * wy * wx

    def _runtime_material_views(self, morphology, cp):
        runtime_contract = morphology._runtime_compute_contract
        staging_reports = []
        runtime_materials = []
        for material_id, material in morphology.materials.items():
            is_full_isotropic = morphology._material_is_explicit_isotropic(material)
            staged_fields = {"Vfrac": None, "S": None, "theta": None, "psi": None}
            field_names = ("Vfrac",) if is_full_isotropic else ("Vfrac", "S", "theta", "psi")
            for field_name in field_names:
                value = getattr(material, field_name)
                plan = assess_array_for_backend_runtime(
                    value,
                    backend_name=self.name,
                    field_name=field_name,
                    material_id=material_id,
                    contract=runtime_contract,
                )
                staging_reports.append(plan)
                staged_fields[field_name] = coerce_array_for_backend(value, plan)

            runtime_materials.append(
                _RuntimeMaterialView(
                    materialID=material_id,
                    opt_constants=material.opt_constants,
                    Vfrac=staged_fields["Vfrac"],
                    S=staged_fields["S"],
                    theta=staged_fields["theta"],
                    psi=staged_fields["psi"],
                    is_full_isotropic=is_full_isotropic,
                )
            )

        morphology.last_runtime_staging_report = staging_reports
        return tuple(runtime_materials)

    def _execution_path(self, morphology) -> str:
        return str(morphology.backend_options.get("execution_path", "tensor_coeff"))

    def _run_single_energy(self, morphology, runtime_materials, energy, cp, ndimage, window, recorder):
        execution_path = self._execution_path(morphology)
        if execution_path == "tensor_coeff":
            return self._run_single_energy_tensor_coeff(
                morphology=morphology,
                runtime_materials=runtime_materials,
                energy=energy,
                cp=cp,
                ndimage=ndimage,
                window=window,
                recorder=recorder,
            )
        if execution_path == "direct_polarization":
            return self._run_single_energy_direct_polarization(
                morphology=morphology,
                runtime_materials=runtime_materials,
                energy=energy,
                cp=cp,
                ndimage=ndimage,
                window=window,
                recorder=recorder,
            )
        raise AssertionError(f"Unsupported cupy-rsoxs execution_path {execution_path!r}.")

    def _run_single_energy_tensor_coeff(
        self,
        morphology,
        runtime_materials,
        energy,
        cp,
        ndimage,
        window,
        recorder,
    ):
        angle_family_plan = self._angle_family_plan(morphology)
        shape_override = self._segment_c_shape_override(morphology)
        nt = recorder.measure(
            "B",
            lambda: self._compute_nt_components_for_tensor_coeff(
                morphology=morphology,
                runtime_materials=runtime_materials,
                energy=energy,
                cp=cp,
                required_components=angle_family_plan.required_nt_components,
            ),
        )
        fft_nt = recorder.measure(
            "C",
            lambda: self._compute_fft_nt_components(
                nt=nt,
                morphology=morphology,
                cp=cp,
                window=window,
                component_indices=angle_family_plan.required_nt_components,
            ),
        )
        proj_x, proj_y, proj_xy = recorder.measure(
            "D",
            lambda: self._projection_coefficients_from_fft_nt(
                morphology=morphology,
                energy=energy,
                cp=cp,
                fft_nt=fft_nt,
                angle_family_plan=angle_family_plan,
                shape_override=shape_override,
            ),
        )
        del nt, fft_nt
        angle_projections = recorder.measure(
            "E",
            lambda: self._rotate_and_accumulate_projection_coefficients(
                morphology=morphology,
                cp=cp,
                ndimage=ndimage,
                proj_x=proj_x,
                proj_y=proj_y,
                proj_xy=proj_xy,
                angle_family_plan=angle_family_plan,
            ),
        )
        del proj_x, proj_y, proj_xy
        return angle_projections

    def _run_single_energy_direct_polarization(
        self,
        morphology,
        runtime_materials,
        energy,
        cp,
        ndimage,
        window,
        recorder,
    ):
        angle_family_plan = self._angle_family_plan(morphology)
        shape_override = self._segment_c_shape_override(morphology)
        isotropic_base_field = None
        if (
            self._z_collapse_mode(morphology) != "mean"
            and cp.dtype(runtime_materials[0].Vfrac.dtype).name != "float16"
        ):
            isotropic_base_field = self._compute_direct_isotropic_base_field(
                runtime_materials,
                energy,
                cp,
            )
        return self._project_from_direct_polarization(
            morphology=morphology,
            runtime_materials=runtime_materials,
            energy=energy,
            cp=cp,
            ndimage=ndimage,
            window=window,
            angle_family_plan=angle_family_plan,
            isotropic_base_field=isotropic_base_field,
            shape_override=shape_override,
            recorder=recorder,
        )

    def _num_angles(self, morphology) -> int:
        start_angle, increment_angle, end_angle = map(float, morphology.EAngleRotation)
        return int(round((end_angle - start_angle) / increment_angle + 1.0)) if increment_angle else 1

    def _angles_radians(self, morphology):
        start_angle, increment_angle, end_angle = map(float, morphology.EAngleRotation)
        num_angles = self._num_angles(morphology)
        if num_angles == 1:
            return (math.radians(start_angle),)
        return tuple(math.radians(start_angle + increment_angle * idx) for idx in range(num_angles))

    def _angle_family_plan(self, morphology):
        angles = self._angles_radians(morphology)
        cache_key = ("angle_family_plan", tuple(round(angle, 12) for angle in angles))
        plan = morphology._backend_runtime_state.get(cache_key)
        if plan is not None:
            return plan

        all_axis_aligned = True
        required_nt_components: set[int] = set()
        needs_proj_x = False
        needs_proj_y = False
        angle_plans = []
        tol = 1e-6
        full_turn = 2.0 * math.pi

        for angle in angles:
            mx = np.float32(math.cos(angle))
            my = np.float32(math.sin(angle))
            mx_f = float(mx)
            my_f = float(my)

            if abs(my_f) <= tol and abs(abs(mx_f) - 1.0) <= tol:
                family = "x"
                needs_proj_x = True
                required_nt_components.update((0, 1, 2))
            elif abs(mx_f) <= tol and abs(abs(my_f) - 1.0) <= tol:
                family = "y"
                needs_proj_y = True
                required_nt_components.update((1, 3, 4))
            else:
                family = "general"
                all_axis_aligned = False
                needs_proj_x = True
                needs_proj_y = True
                required_nt_components = {0, 1, 2, 3, 4}

            reduced = math.remainder(angle, full_turn)
            angle_plans.append(
                _AnglePlan(
                    angle_radians=angle,
                    mx=mx,
                    my=my,
                    family=family,
                    is_identity_rotation=abs(reduced) <= tol,
                )
            )

        plan = _AngleFamilyPlan(
            angles=tuple(angle_plans),
            all_axis_aligned=all_axis_aligned,
            required_nt_components=tuple(sorted(required_nt_components or {0, 1, 2, 3, 4})),
            needs_proj_x=needs_proj_x,
            needs_proj_y=needs_proj_y,
            needs_proj_xy=not all_axis_aligned,
        )
        morphology._backend_runtime_state[cache_key] = plan
        return plan

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

    def _material_optical_scalars(self, material, energy):
        npar, nper = self._material_optics(material, energy)
        nsum_sq = np.complex64((npar + 2.0 * nper) ** 2)
        npar_sq = np.complex64(npar * npar)
        nper_sq = np.complex64(nper * nper)
        isotropic_diag = np.complex64(nsum_sq / np.float32(9.0) - np.complex64(1.0))
        aligned_base = np.complex64(nper_sq - nsum_sq / np.float32(9.0))
        anisotropic_delta = np.complex64(npar_sq - nper_sq)
        return isotropic_diag, aligned_base, anisotropic_delta

    def _orientation_components(self, material, cp):
        sin_theta = cp.sin(material.theta, dtype=cp.float32)
        sx = cp.cos(material.psi, dtype=cp.float32) * sin_theta
        sy = cp.sin(material.psi, dtype=cp.float32) * sin_theta
        sz = cp.cos(material.theta, dtype=cp.float32)
        del sin_theta
        return sx, sy, sz

    def _compute_direct_isotropic_base_field(self, runtime_materials, energy, cp):
        shape = tuple(int(v) for v in runtime_materials[0].Vfrac.shape)
        isotropic_base = cp.zeros(shape, dtype=cp.complex64)
        for material in runtime_materials:
            isotropic_diag, _, _ = self._material_optical_scalars(material, energy)
            isotropic_base += material.Vfrac * isotropic_diag
        return isotropic_base

    def _compute_nt_components(self, runtime_materials, energy, cp, required_components=None):
        if cp.dtype(runtime_materials[0].Vfrac.dtype).name == "float16":
            return self._compute_nt_components_half_input(
                runtime_materials,
                energy,
                cp,
                required_components=required_components,
            )

        required = {0, 1, 2, 3, 4} if required_components is None else set(required_components)
        need0 = 0 in required
        need1 = 1 in required
        need2 = 2 in required
        need3 = 3 in required
        need4 = 4 in required
        shape = tuple(int(v) for v in runtime_materials[0].Vfrac.shape)
        nt = cp.zeros((5, *shape), dtype=cp.complex64)

        for material in runtime_materials:
            isotropic_diag, aligned_base, anisotropic_delta = self._material_optical_scalars(
                material,
                energy,
            )
            vfrac = material.Vfrac
            isotropic_term = None
            if need0 or need3:
                isotropic_term = vfrac * isotropic_diag
                if need0:
                    nt[0] += isotropic_term
                if need3:
                    nt[3] += isotropic_term
            if material.is_full_isotropic:
                if isotropic_term is not None:
                    del isotropic_term
                continue

            phi_a = vfrac * material.S
            sx, sy, sz = self._orientation_components(material, cp)

            if need0:
                nt[0] += phi_a * (aligned_base + anisotropic_delta * sx * sx)
            if need1:
                nt[1] += phi_a * anisotropic_delta * sx * sy
            if need2:
                nt[2] += phi_a * anisotropic_delta * sx * sz
            if need3:
                nt[3] += phi_a * (aligned_base + anisotropic_delta * sy * sy)
            if need4:
                nt[4] += phi_a * anisotropic_delta * sy * sz

            if isotropic_term is not None:
                del isotropic_term
            del phi_a, sx, sy, sz

        return nt

    def _compute_nt_components_for_tensor_coeff(
        self,
        morphology,
        runtime_materials,
        energy,
        cp,
        required_components=None,
    ):
        if self._z_collapse_mode(morphology) == "mean":
            return self._compute_nt_components_collapsed_mean(
                runtime_materials,
                energy,
                cp,
                required_components=required_components,
            )
        return self._compute_nt_components(
            runtime_materials,
            energy,
            cp,
            required_components=required_components,
        )

    def _compute_nt_components_collapsed_mean(
        self,
        runtime_materials,
        energy,
        cp,
        required_components=None,
    ):
        if cp.dtype(runtime_materials[0].Vfrac.dtype).name == "float16":
            raise NotImplementedError(
                "cupy-rsoxs z_collapse_mode='mean' does not yet support the half-input "
                "mixed-precision path."
            )

        required = {0, 1, 2, 3, 4} if required_components is None else set(required_components)
        need0 = 0 in required
        need1 = 1 in required
        need2 = 2 in required
        need3 = 3 in required
        need4 = 4 in required
        _, y, x = (int(v) for v in runtime_materials[0].Vfrac.shape)
        z_count = np.float32(runtime_materials[0].Vfrac.shape[0])
        nt = cp.zeros((5, 1, y, x), dtype=cp.complex64)

        for material in runtime_materials:
            isotropic_diag, aligned_base, anisotropic_delta = self._material_optical_scalars(
                material,
                energy,
            )
            vfrac = material.Vfrac
            if need0 or need3:
                vfrac_sum = cp.sum(vfrac, axis=0, dtype=cp.float32, keepdims=True)
                isotropic_collapsed = (vfrac_sum / z_count).astype(cp.complex64, copy=False)
                if need0:
                    nt[0] += isotropic_collapsed * isotropic_diag
                if need3:
                    nt[3] += isotropic_collapsed * isotropic_diag
                del vfrac_sum, isotropic_collapsed
            if material.is_full_isotropic:
                continue

            phi_a = vfrac * material.S
            sx, sy, sz = self._orientation_components(material, cp)

            if need0:
                contrib0 = phi_a * (aligned_base + anisotropic_delta * sx * sx)
                nt[0] += cp.sum(contrib0, axis=0, dtype=cp.complex64, keepdims=True) / z_count
                del contrib0
            if need1:
                contrib1 = phi_a * anisotropic_delta * sx * sy
                nt[1] += cp.sum(contrib1, axis=0, dtype=cp.complex64, keepdims=True) / z_count
                del contrib1
            if need2:
                contrib2 = phi_a * anisotropic_delta * sx * sz
                nt[2] += cp.sum(contrib2, axis=0, dtype=cp.complex64, keepdims=True) / z_count
                del contrib2
            if need3:
                contrib3 = phi_a * (aligned_base + anisotropic_delta * sy * sy)
                nt[3] += cp.sum(contrib3, axis=0, dtype=cp.complex64, keepdims=True) / z_count
                del contrib3
            if need4:
                contrib4 = phi_a * anisotropic_delta * sy * sz
                nt[4] += cp.sum(contrib4, axis=0, dtype=cp.complex64, keepdims=True) / z_count
                del contrib4

            del phi_a, sx, sy, sz

        return nt

    def _compute_nt_components_half_input(self, runtime_materials, energy, cp, required_components=None):
        required = {0, 1, 2, 3, 4} if required_components is None else set(required_components)
        need0 = np.int32(0 in required)
        need1 = np.int32(1 in required)
        need2 = np.int32(2 in required)
        need3 = np.int32(3 in required)
        need4 = np.int32(4 in required)
        shape = tuple(int(v) for v in runtime_materials[0].Vfrac.shape)
        nt = cp.zeros((5, *shape), dtype=cp.complex64)
        nt0, nt1, nt2, nt3, nt4 = (nt[idx] for idx in range(5))
        threads = 256

        for material in runtime_materials:
            isotropic_diag, aligned_base, anisotropic_delta = self._material_optical_scalars(
                material,
                energy,
            )
            total = np.uint64(material.Vfrac.size)
            blocks = (material.Vfrac.size + threads - 1) // threads

            if material.is_full_isotropic:
                self._nt_accumulate_isotropic_half_kernel(cp)(
                    (blocks,),
                    (threads,),
                    (
                        material.Vfrac,
                        isotropic_diag,
                        need0,
                        need3,
                        nt0,
                        nt3,
                        total,
                    ),
                )
                continue

            self._nt_accumulate_anisotropic_half_kernel(cp)(
                (blocks,),
                (threads,),
                (
                    material.Vfrac,
                    material.S,
                    material.theta,
                    material.psi,
                    isotropic_diag,
                    aligned_base,
                    anisotropic_delta,
                    need0,
                    need1,
                    need2,
                    need3,
                    need4,
                    nt0,
                    nt1,
                    nt2,
                    nt3,
                    nt4,
                    total,
                ),
            )

        return nt

    def _nt_accumulate_isotropic_half_kernel(self, cp):
        kernel = _CUPY_KERNEL_CACHE.get("nt_accumulate_isotropic_half_input")
        if kernel is not None:
            return kernel

        kernel = cp.RawKernel(
            rf"""
            {_HALF_BITS_TO_FLOAT_DEVICE_FUNCTION}

            extern "C" __global__
            void nt_accumulate_isotropic_half_input(
                const unsigned short* vfrac,
                const float2 isotropic_diag,
                const int need0,
                const int need3,
                float2* nt0,
                float2* nt3,
                const unsigned long long total
            ) {{
                const unsigned long long idx =
                    (unsigned long long)blockDim.x * (unsigned long long)blockIdx.x
                    + (unsigned long long)threadIdx.x;
                if (idx >= total) {{
                    return;
                }}

                const float vf = nrss_half_bits_to_float(vfrac[idx]);
                if (need0) {{
                    nt0[idx].x += vf * isotropic_diag.x;
                    nt0[idx].y += vf * isotropic_diag.y;
                }}
                if (need3) {{
                    nt3[idx].x += vf * isotropic_diag.x;
                    nt3[idx].y += vf * isotropic_diag.y;
                }}
            }}
            """,
            "nt_accumulate_isotropic_half_input",
        )
        _CUPY_KERNEL_CACHE["nt_accumulate_isotropic_half_input"] = kernel
        return kernel

    def _nt_accumulate_anisotropic_half_kernel(self, cp):
        kernel = _CUPY_KERNEL_CACHE.get("nt_accumulate_anisotropic_half_input")
        if kernel is not None:
            return kernel

        kernel = cp.RawKernel(
            rf"""
            {_HALF_BITS_TO_FLOAT_DEVICE_FUNCTION}

            extern "C" __global__
            void nt_accumulate_anisotropic_half_input(
                const unsigned short* vfrac,
                const unsigned short* s,
                const unsigned short* theta,
                const unsigned short* psi,
                const float2 isotropic_diag,
                const float2 aligned_base,
                const float2 anisotropic_delta,
                const int need0,
                const int need1,
                const int need2,
                const int need3,
                const int need4,
                float2* nt0,
                float2* nt1,
                float2* nt2,
                float2* nt3,
                float2* nt4,
                const unsigned long long total
            ) {{
                const unsigned long long idx =
                    (unsigned long long)blockDim.x * (unsigned long long)blockIdx.x
                    + (unsigned long long)threadIdx.x;
                if (idx >= total) {{
                    return;
                }}

                const float vf = nrss_half_bits_to_float(vfrac[idx]);
                if (need0 || need3) {{
                    const float iso_x = vf * isotropic_diag.x;
                    const float iso_y = vf * isotropic_diag.y;
                    if (need0) {{
                        nt0[idx].x += iso_x;
                        nt0[idx].y += iso_y;
                    }}
                    if (need3) {{
                        nt3[idx].x += iso_x;
                        nt3[idx].y += iso_y;
                    }}
                }}

                const float phi = vf * nrss_half_bits_to_float(s[idx]);
                const float theta_i = nrss_half_bits_to_float(theta[idx]);
                const float psi_i = nrss_half_bits_to_float(psi[idx]);
                const float sin_theta = sinf(theta_i);
                const float sx = cosf(psi_i) * sin_theta;
                const float sy = sinf(psi_i) * sin_theta;
                const float sz = cosf(theta_i);

                if (need0) {{
                    nt0[idx].x += phi * (aligned_base.x + anisotropic_delta.x * sx * sx);
                    nt0[idx].y += phi * (aligned_base.y + anisotropic_delta.y * sx * sx);
                }}
                if (need1) {{
                    nt1[idx].x += phi * anisotropic_delta.x * sx * sy;
                    nt1[idx].y += phi * anisotropic_delta.y * sx * sy;
                }}
                if (need2) {{
                    nt2[idx].x += phi * anisotropic_delta.x * sx * sz;
                    nt2[idx].y += phi * anisotropic_delta.y * sx * sz;
                }}
                if (need3) {{
                    nt3[idx].x += phi * (aligned_base.x + anisotropic_delta.x * sy * sy);
                    nt3[idx].y += phi * (aligned_base.y + anisotropic_delta.y * sy * sy);
                }}
                if (need4) {{
                    nt4[idx].x += phi * anisotropic_delta.x * sy * sz;
                    nt4[idx].y += phi * anisotropic_delta.y * sy * sz;
                }}
            }}
            """,
            "nt_accumulate_anisotropic_half_input",
        )
        _CUPY_KERNEL_CACHE["nt_accumulate_anisotropic_half_input"] = kernel
        return kernel

    def _compute_fft_nt_components(self, nt, morphology, cp, window, component_indices=None):
        component_indices = tuple(range(nt.shape[0])) if component_indices is None else tuple(component_indices)
        for idx in component_indices:
            component = nt[idx]
            if window is not None:
                cp.multiply(component, window, out=component)
            fft_component = cp.fft.fftn(component)
            self._replace_dc_component(fft_component)
            self._igor_shift(fft_component, morphology, cp, out=nt[idx])
            del component, fft_component
        return nt

    def _rotate_and_accumulate_projection_coefficients(
        self,
        morphology,
        cp,
        ndimage,
        proj_x,
        proj_y,
        proj_xy,
        angle_family_plan,
    ):
        projection_average = None
        valid_counts = None
        use_rot_mask = bool(morphology.RotMask)
        num_angles = len(angle_family_plan.angles)
        for angle_plan, (matrix_yx, offset_yx) in zip(
            angle_family_plan.angles,
            self._rotation_transforms(morphology, cp),
        ):
            if angle_plan.family == "x":
                projection = proj_x
            elif angle_plan.family == "y":
                projection = proj_y
            else:
                projection = (
                    proj_x * (angle_plan.mx * angle_plan.mx)
                    + proj_y * (angle_plan.my * angle_plan.my)
                    + proj_xy * (angle_plan.mx * angle_plan.my)
                )
            rotated = self._apply_affine_transform(
                projection,
                ndimage,
                matrix_yx,
                offset_yx,
                cval=np.nan,
            ) if not angle_plan.is_identity_rotation else projection
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
        return self._finalize_rotation_average(cp, projection_average, valid_counts, num_angles)

    def _projection_coefficients_from_fft_nt(
        self,
        morphology,
        energy,
        cp,
        fft_nt,
        angle_family_plan,
        shape_override=None,
    ):
        basis_x = None
        basis_y = None
        proj_x = None
        proj_y = None

        if angle_family_plan.needs_proj_x:
            basis_x = (
                fft_nt[0] * self._one_by_four_pi,
                fft_nt[1] * self._one_by_four_pi,
                fft_nt[2] * self._one_by_four_pi,
            )

        if angle_family_plan.needs_proj_y:
            basis_y = (
                fft_nt[1] * self._one_by_four_pi,
                fft_nt[3] * self._one_by_four_pi,
                fft_nt[4] * self._one_by_four_pi,
            )

        if angle_family_plan.needs_proj_xy:
            return self._projection_coefficients_from_fft_pair(
                morphology=morphology,
                energy=energy,
                cp=cp,
                basis_x=basis_x,
                basis_y=basis_y,
                shape_override=shape_override,
            )

        # Even aligned families only need detector-plane values, so avoid
        # materializing an intermediate scatter3d volume here.
        if angle_family_plan.needs_proj_x and angle_family_plan.needs_proj_y:
            proj_x, proj_y, _ = self._projection_coefficients_from_fft_pair(
                morphology=morphology,
                energy=energy,
                cp=cp,
                basis_x=basis_x,
                basis_y=basis_y,
                shape_override=shape_override,
            )
            return proj_x, proj_y, None

        if angle_family_plan.needs_proj_x:
            proj_x, _, _ = self._projection_coefficients_from_fft_pair(
                morphology=morphology,
                energy=energy,
                cp=cp,
                basis_x=basis_x,
                basis_y=basis_x,
                shape_override=shape_override,
            )

        if angle_family_plan.needs_proj_y:
            proj_y, _, _ = self._projection_coefficients_from_fft_pair(
                morphology=morphology,
                energy=energy,
                cp=cp,
                basis_x=basis_y,
                basis_y=basis_y,
                shape_override=shape_override,
            )

        return proj_x, proj_y, None

    def _projection_coefficients_from_fft_pair(
        self,
        morphology,
        energy,
        cp,
        basis_x,
        basis_y,
        shape_override=None,
    ):
        detector_geometry = self._detector_geometry(morphology, cp, shape_override=shape_override)
        projection_geometry = self._detector_projection_geometry(
            morphology=morphology,
            energy=energy,
            cp=cp,
            detector_geometry=detector_geometry,
            shape_override=shape_override,
        )
        k = np.float32(2.0 * math.pi / (1239.84197 / float(energy)))
        d = np.float32(k * k)
        a = detector_geometry.qx[None, :]
        b = detector_geometry.qy[:, None]
        out_nan = np.float32(np.nan)

        if detector_geometry.z_count == 1:
            proj_x, proj_y, proj_xy = self._detector_projection_coefficients_from_fft_slices(
                a=a,
                b=b,
                c=k + detector_geometry.qz[0],
                d=d,
                basis_x=(basis_x[0][0], basis_x[1][0], basis_x[2][0]),
                basis_y=(basis_y[0][0], basis_y[1][0], basis_y[2][0]),
            )
            return (
                cp.where(projection_geometry.valid, proj_x, out_nan),
                cp.where(projection_geometry.valid, proj_y, out_nan),
                cp.where(projection_geometry.valid, proj_xy, out_nan),
            )

        z0 = projection_geometry.safe_z0
        z1 = projection_geometry.safe_z1
        y_idx = detector_geometry.y_idx
        x_idx = detector_geometry.x_idx
        c0 = k + detector_geometry.qz[z0]
        c1 = k + detector_geometry.qz[z1]

        proj_x0, proj_y0, proj_xy0 = self._detector_projection_coefficients_from_fft_slices(
            a=a,
            b=b,
            c=c0,
            d=d,
            basis_x=(
                basis_x[0][z0, y_idx, x_idx],
                basis_x[1][z0, y_idx, x_idx],
                basis_x[2][z0, y_idx, x_idx],
            ),
            basis_y=(
                basis_y[0][z0, y_idx, x_idx],
                basis_y[1][z0, y_idx, x_idx],
                basis_y[2][z0, y_idx, x_idx],
            ),
        )
        proj_x1, proj_y1, proj_xy1 = self._detector_projection_coefficients_from_fft_slices(
            a=a,
            b=b,
            c=c1,
            d=d,
            basis_x=(
                basis_x[0][z1, y_idx, x_idx],
                basis_x[1][z1, y_idx, x_idx],
                basis_x[2][z1, y_idx, x_idx],
            ),
            basis_y=(
                basis_y[0][z1, y_idx, x_idx],
                basis_y[1][z1, y_idx, x_idx],
                basis_y[2][z1, y_idx, x_idx],
            ),
        )

        frac = projection_geometry.frac
        keep = np.float32(1.0) - frac
        proj_x = keep * proj_x0 + frac * proj_x1
        proj_y = keep * proj_y0 + frac * proj_y1
        proj_xy = keep * proj_xy0 + frac * proj_xy1
        return (
            cp.where(projection_geometry.valid, proj_x, out_nan),
            cp.where(projection_geometry.valid, proj_y, out_nan),
            cp.where(projection_geometry.valid, proj_xy, out_nan),
        )

    def _detector_projection_coefficients_from_fft_slices(
        self,
        a,
        b,
        c,
        d,
        basis_x,
        basis_y,
    ):
        x1, y1, z1 = basis_x
        x2, y2, z2 = basis_y

        term1_x = (-a * a + d) * x1 - a * (b * y1 + c * z1)
        term2_x = -(a * b) * x1 + (-b * b + d) * y1 - b * c * z1
        term3_x = -(a * c) * x1 - b * c * y1 + (-c * c + d) * z1

        term1_y = (-a * a + d) * x2 - a * (b * y2 + c * z2)
        term2_y = -(a * b) * x2 + (-b * b + d) * y2 - b * c * z2
        term3_y = -(a * c) * x2 - b * c * y2 + (-c * c + d) * z2

        proj_x = (
            term1_x.real * term1_x.real
            + term1_x.imag * term1_x.imag
            + term2_x.real * term2_x.real
            + term2_x.imag * term2_x.imag
            + term3_x.real * term3_x.real
            + term3_x.imag * term3_x.imag
        )
        proj_y = (
            term1_y.real * term1_y.real
            + term1_y.imag * term1_y.imag
            + term2_y.real * term2_y.real
            + term2_y.imag * term2_y.imag
            + term3_y.real * term3_y.real
            + term3_y.imag * term3_y.imag
        )
        proj_xy = np.float32(2.0) * (
            (term1_x.real * term1_y.real + term1_x.imag * term1_y.imag)
            + (term2_x.real * term2_y.real + term2_x.imag * term2_y.imag)
            + (term3_x.real * term3_y.real + term3_x.imag * term3_y.imag)
        )
        del term1_x, term2_x, term3_x, term1_y, term2_y, term3_y
        return proj_x, proj_y, proj_xy

    def _project_from_direct_polarization(
        self,
        morphology,
        runtime_materials,
        energy,
        cp,
        ndimage,
        window,
        angle_family_plan,
        isotropic_base_field=None,
        shape_override=None,
        recorder=None,
    ):
        recorder = _NullSegmentRecorder() if recorder is None else recorder
        projection_average = None
        valid_counts = None
        use_rot_mask = bool(morphology.RotMask)
        num_angles = len(angle_family_plan.angles)
        for angle_plan, (matrix_yx, offset_yx) in zip(
            angle_family_plan.angles,
            self._rotation_transforms(morphology, cp),
        ):
            p_x, p_y, p_z = recorder.measure(
                "B",
                lambda angle_plan=angle_plan, isotropic_base_field=isotropic_base_field: self._compute_direct_polarization(
                    morphology=morphology,
                    runtime_materials=runtime_materials,
                    energy=energy,
                    angle_plan=angle_plan,
                    isotropic_base_field=isotropic_base_field,
                    cp=cp,
                ),
            )
            fft_x, fft_y, fft_z = recorder.measure(
                "C",
                lambda p_x=p_x, p_y=p_y, p_z=p_z: self._fft_polarization_fields(
                    morphology=morphology,
                    cp=cp,
                    p_x=p_x,
                    p_y=p_y,
                    p_z=p_z,
                    window=window,
                ),
            )
            del p_x, p_y, p_z
            projection = recorder.measure(
                "D",
                lambda fft_x=fft_x, fft_y=fft_y, fft_z=fft_z: self._projection_from_fft_polarization(
                    morphology=morphology,
                    energy=energy,
                    cp=cp,
                    fft_x=fft_x,
                    fft_y=fft_y,
                    fft_z=fft_z,
                    shape_override=shape_override,
                ),
            )
            projection_average, valid_counts = recorder.measure(
                "E",
                lambda projection=projection, projection_average=projection_average, valid_counts=valid_counts, angle_plan=angle_plan: self._accumulate_rotated_projection(
                    cp=cp,
                    ndimage=ndimage,
                    projection=projection,
                    matrix_yx=matrix_yx,
                    offset_yx=offset_yx,
                    projection_average=projection_average,
                    valid_counts=valid_counts,
                    use_rot_mask=use_rot_mask,
                    skip_rotation=angle_plan.is_identity_rotation,
                ),
            )
            del fft_x, fft_y, fft_z, projection
        return self._finalize_rotation_average(cp, projection_average, valid_counts, num_angles)

    def _compute_direct_polarization(
        self,
        morphology,
        runtime_materials,
        energy,
        angle_plan,
        cp,
        isotropic_base_field=None,
    ):
        if self._z_collapse_mode(morphology) == "mean":
            return self._compute_direct_polarization_collapsed_mean(
                runtime_materials,
                energy,
                angle_plan,
                cp,
            )
        if cp.dtype(runtime_materials[0].Vfrac.dtype).name == "float16":
            return self._compute_direct_polarization_half_input(
                runtime_materials,
                energy,
                angle_plan,
                cp,
            )
        shape = tuple(int(v) for v in runtime_materials[0].Vfrac.shape)
        p_x = cp.zeros(shape, dtype=cp.complex64)
        p_y = cp.zeros(shape, dtype=cp.complex64)
        p_z = cp.zeros(shape, dtype=cp.complex64)
        mx = angle_plan.mx
        my = angle_plan.my
        kernel = self._direct_polarization_kernel(morphology, cp, runtime_materials[0].Vfrac.dtype)
        if isotropic_base_field is not None:
            cp.multiply(isotropic_base_field, np.float32(mx), out=p_x)
            cp.multiply(isotropic_base_field, np.float32(my), out=p_y)

        for material in runtime_materials:
            isotropic_diag, aligned_base, anisotropic_delta = self._material_optical_scalars(
                material,
                energy,
            )
            vfrac = material.Vfrac
            if isotropic_base_field is None:
                isotropic_term = vfrac * isotropic_diag
                p_x += isotropic_term * mx
                p_y += isotropic_term * my
            if material.is_full_isotropic:
                if isotropic_base_field is None:
                    del isotropic_term
                continue

            kernel(
                ((vfrac.size + 255) // 256,),
                (256,),
                (
                    vfrac,
                    material.S,
                    material.theta,
                    material.psi,
                    aligned_base,
                    anisotropic_delta,
                    np.float32(mx),
                    np.float32(my),
                    p_x,
                    p_y,
                    p_z,
                    np.uint64(vfrac.size),
                ),
            )
            if isotropic_base_field is None:
                del isotropic_term

        p_x *= self._one_by_four_pi
        p_y *= self._one_by_four_pi
        p_z *= self._one_by_four_pi
        return p_x, p_y, p_z

    def _compute_direct_polarization_collapsed_mean(self, runtime_materials, energy, angle_plan, cp):
        if cp.dtype(runtime_materials[0].Vfrac.dtype).name == "float16":
            raise NotImplementedError(
                "cupy-rsoxs z_collapse_mode='mean' does not yet support the half-input "
                "mixed-precision path."
            )

        _, y, x = (int(v) for v in runtime_materials[0].Vfrac.shape)
        z_count = np.float32(runtime_materials[0].Vfrac.shape[0])
        p_x = cp.zeros((1, y, x), dtype=cp.complex64)
        p_y = cp.zeros((1, y, x), dtype=cp.complex64)
        p_z = cp.zeros((1, y, x), dtype=cp.complex64)
        mx = angle_plan.mx
        my = angle_plan.my

        for material in runtime_materials:
            isotropic_diag, aligned_base, anisotropic_delta = self._material_optical_scalars(
                material,
                energy,
            )
            vfrac_sum = cp.sum(material.Vfrac, axis=0, dtype=cp.float32, keepdims=True)
            isotropic_collapsed = (vfrac_sum / z_count).astype(cp.complex64, copy=False)
            p_x += isotropic_collapsed * isotropic_diag * mx
            p_y += isotropic_collapsed * isotropic_diag * my
            del vfrac_sum, isotropic_collapsed
            if material.is_full_isotropic:
                continue

            phi_a = material.Vfrac * material.S
            sx, sy, sz = self._orientation_components(material, cp)
            field_projection = sx * mx + sy * my

            contrib_x = phi_a * (mx * aligned_base + anisotropic_delta * sx * field_projection)
            contrib_y = phi_a * (my * aligned_base + anisotropic_delta * sy * field_projection)
            contrib_z = phi_a * (anisotropic_delta * sz * field_projection)

            p_x += cp.sum(contrib_x, axis=0, dtype=cp.complex64, keepdims=True) / z_count
            p_y += cp.sum(contrib_y, axis=0, dtype=cp.complex64, keepdims=True) / z_count
            p_z += cp.sum(contrib_z, axis=0, dtype=cp.complex64, keepdims=True) / z_count

            del phi_a, sx, sy, sz, field_projection, contrib_x, contrib_y, contrib_z

        p_x *= self._one_by_four_pi
        p_y *= self._one_by_four_pi
        p_z *= self._one_by_four_pi
        return p_x, p_y, p_z

    def _compute_direct_polarization_half_input(self, runtime_materials, energy, angle_plan, cp):
        shape = tuple(int(v) for v in runtime_materials[0].Vfrac.shape)
        p_x = cp.zeros(shape, dtype=cp.complex64)
        p_y = cp.zeros(shape, dtype=cp.complex64)
        p_z = cp.zeros(shape, dtype=cp.complex64)
        mx = angle_plan.mx
        my = angle_plan.my
        threads = 256

        for material in runtime_materials:
            isotropic_diag, aligned_base, anisotropic_delta = self._material_optical_scalars(
                material,
                energy,
            )
            total = np.uint64(material.Vfrac.size)
            blocks = (material.Vfrac.size + threads - 1) // threads

            if material.is_full_isotropic:
                self._direct_isotropic_kernel_float16(cp)(
                    (blocks,),
                    (threads,),
                    (
                        material.Vfrac,
                        isotropic_diag,
                        np.float32(mx),
                        np.float32(my),
                        p_x,
                        p_y,
                        total,
                    ),
                )
                continue

            self._direct_anisotropic_kernel_float16(cp)(
                (blocks,),
                (threads,),
                (
                    material.Vfrac,
                    material.S,
                    material.theta,
                    material.psi,
                    isotropic_diag,
                    aligned_base,
                    anisotropic_delta,
                    np.float32(mx),
                    np.float32(my),
                    p_x,
                    p_y,
                    p_z,
                    total,
                ),
            )

        p_x *= self._one_by_four_pi
        p_y *= self._one_by_four_pi
        p_z *= self._one_by_four_pi
        return p_x, p_y, p_z

    def _direct_polarization_kernel(self, morphology, cp, morphology_dtype):
        dtype_name = cp.dtype(morphology_dtype).name
        if dtype_name == "float16":
            return self._direct_generic_kernel_float16(cp)
        if dtype_name == "float32":
            return self._direct_generic_kernel_float32(morphology, cp)
        raise TypeError(
            "cupy-rsoxs direct_polarization received unsupported runtime morphology "
            f"dtype {dtype_name!r}."
        )

    def _direct_generic_kernel_float32(self, morphology, cp):
        return self._build_rawkernel_with_fallback(
            cp,
            family="direct_polarization_generic",
            cache_key_base="direct_polarization_generic_complex64",
            source=r"""
            extern "C" __global__
            void direct_polarization_generic_complex64(
                const float* vfrac,
                const float* s,
                const float* theta,
                const float* psi,
                const float2 aligned_base,
                const float2 anisotropic_delta,
                const float mx,
                const float my,
                float2* p_x,
                float2* p_y,
                float2* p_z,
                const unsigned long long total
            ) {
                const unsigned long long idx =
                    (unsigned long long)blockDim.x * (unsigned long long)blockIdx.x
                    + (unsigned long long)threadIdx.x;
                if (idx >= total) {
                    return;
                }

                const float phi = vfrac[idx] * s[idx];
                const float theta_i = theta[idx];
                const float psi_i = psi[idx];
                const float sin_theta = sinf(theta_i);
                const float sx = cosf(psi_i) * sin_theta;
                const float sy = sinf(psi_i) * sin_theta;
                const float sz = cosf(theta_i);
                const float field_projection = sx * mx + sy * my;

                p_x[idx].x += phi * (mx * aligned_base.x + anisotropic_delta.x * sx * field_projection);
                p_x[idx].y += phi * (mx * aligned_base.y + anisotropic_delta.y * sx * field_projection);
                p_y[idx].x += phi * (my * aligned_base.x + anisotropic_delta.x * sy * field_projection);
                p_y[idx].y += phi * (my * aligned_base.y + anisotropic_delta.y * sy * field_projection);
                p_z[idx].x += phi * (anisotropic_delta.x * sz * field_projection);
                p_z[idx].y += phi * (anisotropic_delta.y * sz * field_projection);
            }
            """,
            kernel_name="direct_polarization_generic_complex64",
            requested_backend=self._rawkernel_backend_option(morphology, "direct_polarization_generic"),
        )

    def _direct_generic_kernel_float16(self, cp):
        kernel = _CUPY_KERNEL_CACHE.get("direct_polarization_generic_complex64_half_input")
        if kernel is not None:
            return kernel

        kernel = cp.RawKernel(
            _HALF_BITS_TO_FLOAT_DEVICE_FUNCTION
            + r"""

            extern "C" __global__
            void direct_polarization_generic_complex64_half_input(
                const unsigned short* vfrac,
                const unsigned short* s,
                const unsigned short* theta,
                const unsigned short* psi,
                const float2 aligned_base,
                const float2 anisotropic_delta,
                const float mx,
                const float my,
                float2* p_x,
                float2* p_y,
                float2* p_z,
                const unsigned long long total
            ) {
                const unsigned long long idx =
                    (unsigned long long)blockDim.x * (unsigned long long)blockIdx.x
                    + (unsigned long long)threadIdx.x;
                if (idx >= total) {
                    return;
                }

                const float phi =
                    nrss_half_bits_to_float(vfrac[idx]) * nrss_half_bits_to_float(s[idx]);
                const float theta_i = nrss_half_bits_to_float(theta[idx]);
                const float psi_i = nrss_half_bits_to_float(psi[idx]);
                const float sin_theta = sinf(theta_i);
                const float sx = cosf(psi_i) * sin_theta;
                const float sy = sinf(psi_i) * sin_theta;
                const float sz = cosf(theta_i);
                const float field_projection = sx * mx + sy * my;

                p_x[idx].x += phi * (mx * aligned_base.x + anisotropic_delta.x * sx * field_projection);
                p_x[idx].y += phi * (mx * aligned_base.y + anisotropic_delta.y * sx * field_projection);
                p_y[idx].x += phi * (my * aligned_base.x + anisotropic_delta.x * sy * field_projection);
                p_y[idx].y += phi * (my * aligned_base.y + anisotropic_delta.y * sy * field_projection);
                p_z[idx].x += phi * (anisotropic_delta.x * sz * field_projection);
                p_z[idx].y += phi * (anisotropic_delta.y * sz * field_projection);
            }
            """,
            "direct_polarization_generic_complex64_half_input",
        )
        _CUPY_KERNEL_CACHE["direct_polarization_generic_complex64_half_input"] = kernel
        return kernel

    def _direct_isotropic_kernel_float16(self, cp):
        kernel = _CUPY_KERNEL_CACHE.get("direct_polarization_isotropic_complex64_half_input")
        if kernel is not None:
            return kernel

        kernel = cp.RawKernel(
            _HALF_BITS_TO_FLOAT_DEVICE_FUNCTION
            + r"""

            extern "C" __global__
            void direct_polarization_isotropic_complex64_half_input(
                const unsigned short* vfrac,
                const float2 isotropic_diag,
                const float mx,
                const float my,
                float2* p_x,
                float2* p_y,
                const unsigned long long total
            ) {
                const unsigned long long idx =
                    (unsigned long long)blockDim.x * (unsigned long long)blockIdx.x
                    + (unsigned long long)threadIdx.x;
                if (idx >= total) {
                    return;
                }

                const float vf = nrss_half_bits_to_float(vfrac[idx]);
                p_x[idx].x += vf * isotropic_diag.x * mx;
                p_x[idx].y += vf * isotropic_diag.y * mx;
                p_y[idx].x += vf * isotropic_diag.x * my;
                p_y[idx].y += vf * isotropic_diag.y * my;
            }
            """,
            "direct_polarization_isotropic_complex64_half_input",
        )
        _CUPY_KERNEL_CACHE["direct_polarization_isotropic_complex64_half_input"] = kernel
        return kernel

    def _direct_anisotropic_kernel_float16(self, cp):
        kernel = _CUPY_KERNEL_CACHE.get("direct_polarization_anisotropic_complex64_half_input")
        if kernel is not None:
            return kernel

        kernel = cp.RawKernel(
            _HALF_BITS_TO_FLOAT_DEVICE_FUNCTION
            + r"""

            extern "C" __global__
            void direct_polarization_anisotropic_complex64_half_input(
                const unsigned short* vfrac,
                const unsigned short* s,
                const unsigned short* theta,
                const unsigned short* psi,
                const float2 isotropic_diag,
                const float2 aligned_base,
                const float2 anisotropic_delta,
                const float mx,
                const float my,
                float2* p_x,
                float2* p_y,
                float2* p_z,
                const unsigned long long total
            ) {
                const unsigned long long idx =
                    (unsigned long long)blockDim.x * (unsigned long long)blockIdx.x
                    + (unsigned long long)threadIdx.x;
                if (idx >= total) {
                    return;
                }

                const float vf = nrss_half_bits_to_float(vfrac[idx]);
                p_x[idx].x += vf * isotropic_diag.x * mx;
                p_x[idx].y += vf * isotropic_diag.y * mx;
                p_y[idx].x += vf * isotropic_diag.x * my;
                p_y[idx].y += vf * isotropic_diag.y * my;

                const float phi = vf * nrss_half_bits_to_float(s[idx]);
                const float theta_i = nrss_half_bits_to_float(theta[idx]);
                const float psi_i = nrss_half_bits_to_float(psi[idx]);
                const float sin_theta = sinf(theta_i);
                const float sx = cosf(psi_i) * sin_theta;
                const float sy = sinf(psi_i) * sin_theta;
                const float sz = cosf(theta_i);
                const float field_projection = sx * mx + sy * my;

                p_x[idx].x += phi * (mx * aligned_base.x + anisotropic_delta.x * sx * field_projection);
                p_x[idx].y += phi * (mx * aligned_base.y + anisotropic_delta.y * sx * field_projection);
                p_y[idx].x += phi * (my * aligned_base.x + anisotropic_delta.x * sy * field_projection);
                p_y[idx].y += phi * (my * aligned_base.y + anisotropic_delta.y * sy * field_projection);
                p_z[idx].x += phi * (anisotropic_delta.x * sz * field_projection);
                p_z[idx].y += phi * (anisotropic_delta.y * sz * field_projection);
            }
            """,
            "direct_polarization_anisotropic_complex64_half_input",
        )
        _CUPY_KERNEL_CACHE["direct_polarization_anisotropic_complex64_half_input"] = kernel
        return kernel

    def _projection_from_polarization(
        self,
        morphology,
        energy,
        cp,
        p_x,
        p_y,
        p_z,
        window,
        shape_override=None,
    ):
        fft_x, fft_y, fft_z = self._fft_polarization_fields(
            morphology=morphology,
            cp=cp,
            p_x=p_x,
            p_y=p_y,
            p_z=p_z,
            window=window,
        )
        projection = self._projection_from_fft_polarization(
            morphology=morphology,
            energy=energy,
            cp=cp,
            fft_x=fft_x,
            fft_y=fft_y,
            fft_z=fft_z,
            shape_override=shape_override,
        )
        del fft_x, fft_y, fft_z
        return projection

    def _fft_polarization_fields(self, morphology, cp, p_x, p_y, p_z, window):
        if window is not None:
            cp.multiply(p_x, window, out=p_x)
            cp.multiply(p_y, window, out=p_y)
            cp.multiply(p_z, window, out=p_z)

        fft_x = cp.fft.fftn(p_x)
        fft_y = cp.fft.fftn(p_y)
        fft_z = cp.fft.fftn(p_z)
        self._replace_dc_component(fft_x)
        self._replace_dc_component(fft_y)
        self._replace_dc_component(fft_z)
        fft_x = self._igor_shift(fft_x, morphology, cp)
        fft_y = self._igor_shift(fft_y, morphology, cp)
        fft_z = self._igor_shift(fft_z, morphology, cp)
        return fft_x, fft_y, fft_z

    def _accumulate_rotated_projection(
        self,
        cp,
        ndimage,
        projection,
        matrix_yx,
        offset_yx,
        projection_average,
        valid_counts,
        use_rot_mask,
        skip_rotation=False,
    ):
        rotated = (
            projection
            if skip_rotation
            else self._apply_affine_transform(
                projection,
                ndimage,
                matrix_yx,
                offset_yx,
                cval=np.nan,
            )
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
        del rotated
        return projection_average, valid_counts

    def _projection_from_fft_polarization(
        self,
        morphology,
        energy,
        cp,
        fft_x,
        fft_y,
        fft_z,
        shape_override=None,
    ):
        return self._project_fft_polarization_direct_kernel(
            morphology=morphology,
            energy=energy,
            cp=cp,
            fft_x=fft_x,
            fft_y=fft_y,
            fft_z=fft_z,
            shape_override=shape_override,
        )

    def _project_fft_polarization_direct_kernel(
        self,
        morphology,
        energy,
        cp,
        fft_x,
        fft_y,
        fft_z,
        shape_override=None,
    ):
        detector_geometry = self._detector_geometry(
            morphology=morphology,
            cp=cp,
            shape_override=shape_override,
        )
        projection_geometry = self._detector_projection_geometry(
            morphology=morphology,
            energy=energy,
            cp=cp,
            detector_geometry=detector_geometry,
            shape_override=shape_override,
        )

        projection = cp.empty(
            (detector_geometry.y_count, detector_geometry.x_count),
            dtype=cp.float32,
        )
        total = int(detector_geometry.y_count * detector_geometry.x_count)
        threads = 256
        blocks = (total + threads - 1) // threads
        k = np.float32(2.0 * math.pi / (1239.84197 / float(energy)))
        d = np.float32(k * k)
        nan_value = np.float32(np.nan)

        if detector_geometry.z_count == 1:
            self._direct_detector_projection_single_slice_kernel(cp)(
                (blocks,),
                (threads,),
                (
                    projection_geometry.valid,
                    detector_geometry.qx,
                    detector_geometry.qy,
                    detector_geometry.qz,
                    np.float32(k),
                    d,
                    fft_x,
                    fft_y,
                    fft_z,
                    projection,
                    np.int32(detector_geometry.y_count),
                    np.int32(detector_geometry.x_count),
                    nan_value,
                    np.uint64(total),
                ),
            )
            return projection

        self._direct_detector_projection_interpolated_kernel(cp)(
            (blocks,),
            (threads,),
            (
                projection_geometry.valid,
                projection_geometry.safe_z0,
                projection_geometry.safe_z1,
                projection_geometry.frac,
                detector_geometry.qx,
                detector_geometry.qy,
                detector_geometry.qz,
                np.float32(k),
                d,
                fft_x,
                fft_y,
                fft_z,
                projection,
                np.int32(detector_geometry.z_count),
                np.int32(detector_geometry.y_count),
                np.int32(detector_geometry.x_count),
                nan_value,
                np.uint64(total),
            ),
        )
        return projection

    def _replace_dc_component(self, arr):
        z, y, x = arr.shape
        neighbors = [arr[0, 0, 1], arr[0, 1, 0], arr[0, 0, x - 1], arr[0, y - 1, 0]]
        if z > 1:
            neighbors.extend([arr[1, 0, 0], arr[z - 1, 0, 0]])
        arr[0, 0, 0] = sum(neighbors) / np.float32(len(neighbors))

    def _igor_shift_kernel(self, morphology, cp):
        return self._build_rawkernel_with_fallback(
            cp,
            family="igor_shift",
            cache_key_base="igor_shift_complex64",
            source=r"""
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
            kernel_name="igor_shift_complex64",
            requested_backend=self._rawkernel_backend_option(morphology, "igor_shift"),
        )

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

    def _igor_shift(self, arr, morphology, cp, out=None):
        if out is None:
            out = cp.empty_like(arr)
        z_order, y_order, x_order = self._igor_axis_orders(arr.shape, cp)
        total = int(arr.size)
        threads = 256
        blocks = (total + threads - 1) // threads
        self._igor_shift_kernel(morphology, cp)(
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

    def _direct_projection_rawkernel_backend(self):
        # For the direct detector kernels, nvcc avoids the large cold peak we
        # saw from the nvrtc compile/load path on the maintained issue lane.
        import cupy as cp

        return self._resolve_requested_rawkernel_backend(
            cp,
            requested_backend="auto",
            prefer_auto_nvcc=True,
        )

    def _direct_detector_projection_single_slice_kernel(self, cp):
        return self._build_rawkernel_with_fallback(
            cp,
            family="direct_detector_projection",
            cache_key_base="direct_detector_projection_single_slice_float32",
            source=r"""
            __device__ inline float nrss_direct_projection_intensity(
                const float a,
                const float b,
                const float c,
                const float d,
                const float2 p1,
                const float2 p2,
                const float2 p3
            ) {
                const float term1r = (-a * a + d) * p1.x - a * (b * p2.x + c * p3.x);
                const float term1i = (-a * a + d) * p1.y - a * (b * p2.y + c * p3.y);
                const float term2r = -(a * b) * p1.x + (-b * b + d) * p2.x - b * c * p3.x;
                const float term2i = -(a * b) * p1.y + (-b * b + d) * p2.y - b * c * p3.y;
                const float term3r = -(a * c) * p1.x - b * c * p2.x + (-c * c + d) * p3.x;
                const float term3i = -(a * c) * p1.y - b * c * p2.y + (-c * c + d) * p3.y;
                return
                    term1r * term1r + term1i * term1i
                    + term2r * term2r + term2i * term2i
                    + term3r * term3r + term3i * term3i;
            }

            extern "C" __global__
            void direct_detector_projection_single_slice_float32(
                const bool* valid,
                const float* qx,
                const float* qy,
                const float* qz,
                const float k,
                const float d,
                const float2* fft_x,
                const float2* fft_y,
                const float2* fft_z,
                float* output,
                const int ydim,
                const int xdim,
                const float nan_value,
                const unsigned long long total
            ) {
                const unsigned long long idx =
                    (unsigned long long)blockDim.x * (unsigned long long)blockIdx.x
                    + (unsigned long long)threadIdx.x;
                if (idx >= total) {
                    return;
                }
                if (!valid[idx]) {
                    output[idx] = nan_value;
                    return;
                }

                const int x = (int)(idx % (unsigned long long)xdim);
                const int y = (int)(idx / (unsigned long long)xdim);
                const float a = qx[x];
                const float b = qy[y];
                const float c = k + qz[0];

                output[idx] = nrss_direct_projection_intensity(
                    a,
                    b,
                    c,
                    d,
                    fft_x[idx],
                    fft_y[idx],
                    fft_z[idx]
                );
            }
            """,
            kernel_name="direct_detector_projection_single_slice_float32",
            requested_backend="auto",
            prefer_auto_nvcc=True,
        )

    def _direct_detector_projection_interpolated_kernel(self, cp):
        return self._build_rawkernel_with_fallback(
            cp,
            family="direct_detector_projection",
            cache_key_base="direct_detector_projection_interpolated_float32",
            source=r"""
            __device__ inline float nrss_direct_projection_intensity_interp(
                const float a,
                const float b,
                const float c,
                const float d,
                const float2 p1,
                const float2 p2,
                const float2 p3
            ) {
                const float term1r = (-a * a + d) * p1.x - a * (b * p2.x + c * p3.x);
                const float term1i = (-a * a + d) * p1.y - a * (b * p2.y + c * p3.y);
                const float term2r = -(a * b) * p1.x + (-b * b + d) * p2.x - b * c * p3.x;
                const float term2i = -(a * b) * p1.y + (-b * b + d) * p2.y - b * c * p3.y;
                const float term3r = -(a * c) * p1.x - b * c * p2.x + (-c * c + d) * p3.x;
                const float term3i = -(a * c) * p1.y - b * c * p2.y + (-c * c + d) * p3.y;
                return
                    term1r * term1r + term1i * term1i
                    + term2r * term2r + term2i * term2i
                    + term3r * term3r + term3i * term3i;
            }

            extern "C" __global__
            void direct_detector_projection_interpolated_float32(
                const bool* valid,
                const int* z0,
                const int* z1,
                const float* frac,
                const float* qx,
                const float* qy,
                const float* qz,
                const float k,
                const float d,
                const float2* fft_x,
                const float2* fft_y,
                const float2* fft_z,
                float* output,
                const int zdim,
                const int ydim,
                const int xdim,
                const float nan_value,
                const unsigned long long total
            ) {
                const unsigned long long idx =
                    (unsigned long long)blockDim.x * (unsigned long long)blockIdx.x
                    + (unsigned long long)threadIdx.x;
                if (idx >= total) {
                    return;
                }
                if (!valid[idx]) {
                    output[idx] = nan_value;
                    return;
                }

                const int x = (int)(idx % (unsigned long long)xdim);
                const int y = (int)(idx / (unsigned long long)xdim);
                const int z0_i = z0[idx];
                const int z1_i = z1[idx];
                const float frac_i = frac[idx];
                const float keep_i = 1.0f - frac_i;
                const float a = qx[x];
                const float b = qy[y];
                const float c0 = k + qz[z0_i];
                const float c1 = k + qz[z1_i];

                const unsigned long long base0 =
                    ((unsigned long long)z0_i * (unsigned long long)ydim
                     + (unsigned long long)y) * (unsigned long long)xdim
                    + (unsigned long long)x;
                const unsigned long long base1 =
                    ((unsigned long long)z1_i * (unsigned long long)ydim
                     + (unsigned long long)y) * (unsigned long long)xdim
                    + (unsigned long long)x;

                const float proj0 = nrss_direct_projection_intensity_interp(
                    a,
                    b,
                    c0,
                    d,
                    fft_x[base0],
                    fft_y[base0],
                    fft_z[base0]
                );
                const float proj1 = nrss_direct_projection_intensity_interp(
                    a,
                    b,
                    c1,
                    d,
                    fft_x[base1],
                    fft_y[base1],
                    fft_z[base1]
                );
                output[idx] = keep_i * proj0 + frac_i * proj1;
            }
            """,
            kernel_name="direct_detector_projection_interpolated_float32",
            requested_backend="auto",
            prefer_auto_nvcc=True,
        )

    def _compute_scatter3d(self, morphology, energy, cp, p_x, p_y, p_z, shape_override=None):
        detector_geometry = self._detector_geometry(
            morphology,
            cp,
            shape_override=shape_override,
        )
        z = detector_geometry.z_count
        k = np.float32(2.0 * math.pi / (1239.84197 / float(energy)))
        d = np.float32(k * k)
        scatter = cp.empty(
            (
                detector_geometry.z_count,
                detector_geometry.y_count,
                detector_geometry.x_count,
            ),
            dtype=cp.float32,
        )

        a = detector_geometry.qx[None, :]
        b = detector_geometry.qy[:, None]
        for z_index in range(z):
            c = k + detector_geometry.qz[z_index]
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

    def _project_scatter3d(self, morphology, energy, cp, scatter3d, shape_override=None):
        detector_geometry = self._detector_geometry(
            morphology,
            cp,
            shape_override=shape_override,
        )
        projection_geometry = self._detector_projection_geometry(
            morphology=morphology,
            energy=energy,
            cp=cp,
            detector_geometry=detector_geometry,
            shape_override=shape_override,
        )

        projection = cp.full(
            (detector_geometry.y_count, detector_geometry.x_count),
            np.float32(np.nan),
            dtype=cp.float32,
        )
        if detector_geometry.z_count == 1:
            projection = cp.where(projection_geometry.valid, scatter3d[0], projection)
            return projection

        data1 = scatter3d[
            projection_geometry.safe_z0,
            detector_geometry.y_idx,
            detector_geometry.x_idx,
        ]
        data2 = scatter3d[
            projection_geometry.safe_z1,
            detector_geometry.y_idx,
            detector_geometry.x_idx,
        ]
        interp = (
            (np.float32(1.0) - projection_geometry.frac) * data1
            + projection_geometry.frac * data2
        )
        projection = cp.where(projection_geometry.valid, interp, projection)
        return projection

    def _q_axes(self, morphology, cp, shape_override=None):
        detector_geometry = self._detector_geometry(morphology, cp, shape_override=shape_override)
        return detector_geometry.qx, detector_geometry.qy, detector_geometry.qz

    def _detector_geometry(self, morphology, cp, shape_override=None):
        z, y, x = self._shape_tuple(morphology, shape_override=shape_override)
        phys = np.float32(morphology.PhysSize)
        cache_key = ("detector_geometry", z, y, x, float(phys))
        cached = morphology._backend_runtime_state.get(cache_key)
        if cached is not None:
            return cached

        start = np.float32(-math.pi / float(phys))
        x_step = np.float32((2.0 * math.pi / float(phys)) / max(x - 1, 1))
        y_step = np.float32((2.0 * math.pi / float(phys)) / max(y - 1, 1))
        qx = start + cp.arange(x, dtype=cp.float32) * x_step
        qy = start + cp.arange(y, dtype=cp.float32) * y_step
        if z == 1:
            qz = cp.asarray([0.0], dtype=cp.float32)
            qz0 = np.float32(0.0)
            dz = np.float32(0.0)
        else:
            z_step = np.float32((2.0 * math.pi / float(phys)) / max(z - 1, 1))
            qz = start + cp.arange(z, dtype=cp.float32) * z_step
            qz0 = start
            dz = z_step

        x_idx = cp.arange(x, dtype=cp.int32)[None, :]
        y_idx = cp.arange(y, dtype=cp.int32)[:, None]
        border_valid = (x_idx != (x - 1)) & (y_idx != (y - 1))
        radius_sq = qx[None, :] * qx[None, :] + qy[:, None] * qy[:, None]

        detector_geometry = _DetectorGeometry(
            qx=qx,
            qy=qy,
            qz=qz,
            x_idx=x_idx,
            y_idx=y_idx,
            border_valid=border_valid,
            radius_sq=radius_sq,
            z_count=z,
            y_count=y,
            x_count=x,
            qz0=qz0,
            dz=dz,
        )
        morphology._backend_runtime_state[cache_key] = detector_geometry
        return detector_geometry

    def _detector_projection_geometry(
        self,
        morphology,
        energy,
        cp,
        detector_geometry,
        shape_override=None,
    ):
        z, y, x = self._shape_tuple(morphology, shape_override=shape_override)
        cache_key = (
            "detector_projection_geometry_current",
            z,
            y,
            x,
            float(morphology.PhysSize),
            float(energy),
        )
        cached = morphology._backend_runtime_state.get(cache_key)
        if cached is not None:
            return cached

        k = np.float32(2.0 * math.pi / (1239.84197 / float(energy)))
        val = np.float32(k * k) - detector_geometry.radius_sq
        valid = (val >= 0) & detector_geometry.border_valid

        if detector_geometry.z_count == 1:
            projection_geometry = _DetectorProjectionGeometry(
                valid=valid,
                safe_z0=None,
                safe_z1=None,
                frac=None,
            )
            morphology._backend_runtime_state[cache_key] = projection_geometry
            return projection_geometry

        pos_z = -k + cp.sqrt(cp.where(valid, val, 0), dtype=cp.float32)
        z_float = (pos_z - detector_geometry.qz0) / detector_geometry.dz
        z0 = cp.floor(z_float).astype(cp.int32)
        z1 = z0 + 1
        valid &= z0 >= 0
        valid &= z1 < detector_geometry.z_count

        safe_z0 = cp.clip(z0, 0, detector_geometry.z_count - 1)
        safe_z1 = cp.clip(z1, 0, detector_geometry.z_count - 1)
        frac = z_float - safe_z0.astype(cp.float32)

        projection_geometry = _DetectorProjectionGeometry(
            valid=valid,
            safe_z0=safe_z0,
            safe_z1=safe_z1,
            frac=frac,
        )
        morphology._backend_runtime_state[cache_key] = projection_geometry
        return projection_geometry
