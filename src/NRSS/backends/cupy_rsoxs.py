from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import gc
import importlib
import math
import os
import shutil
import sys
import time
import warnings
from typing import Any

import numpy as np
import xarray as xr

from .arrays import assess_array_for_backend_runtime, coerce_array_for_backend, inspect_array
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
    "direct_polarization_precomputed": "direct_polarization_backend",
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
    phi_a: Any | None = None
    sx: Any | None = None
    sy: Any | None = None
    sz: Any | None = None


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
        namespace = inspect_array(self.data)["namespace"]
        if namespace == "numpy":
            scattering_data = np.asarray(self.data)
        elif namespace == "cupy":
            import cupy as cp

            scattering_data = cp.asnumpy(self.data)
        else:
            raise TypeError(
                "cupy-rsoxs results require numpy or cupy storage, "
                f"received namespace {namespace!r}."
            )
        ny, nx = self.num_zyx[1:]
        d = self.phys_size
        qy = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=d))
        qx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=d))
        return xr.DataArray(
            scattering_data,
            dims=["energy", "qy", "qx"],
            coords={"qy": qy, "qx": qx, "energy": list(self.energies)},
            attrs={
                "phys_size_nm": float(self.phys_size),
                "z_dim": int(self.num_zyx[0]),
            },
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
        del print_vec_info

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
        result_residency = self._result_residency(morphology)
        copy_stream = cp.cuda.Stream(non_blocking=True) if result_residency == "host" else None
        pending_copy_done = None
        pending_host_projection = None
        chunk_buffers = None
        chunk_size = None
        active_chunk_buffer_index = 0
        active_chunk_start = 0
        active_chunk_fill = 0
        runtime_materials = recorder.measure("A2", lambda: self._runtime_material_views(morphology, cp))
        window = None
        result_data = None
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

            with self._iter_completed_energies(
                morphology,
                energies,
                stdout=stdout,
                stderr=stderr,
            ) as energy_iter:
                for energy_index, energy in energy_iter:
                    projection = self._run_single_energy(
                        morphology=morphology,
                        runtime_materials=runtime_materials,
                        energy=energy,
                        cp=cp,
                        ndimage=ndimage,
                        window=window,
                        recorder=recorder,
                    )
                    if result_data is None:
                        result_shape = (len(energies), *projection.shape)
                        if result_residency == "host":
                            result_data = self._allocate_host_result_buffer(result_shape, projection.dtype)
                            chunk_size = self._result_chunk_size(
                                morphology,
                                energy_count=len(energies),
                            )
                            if chunk_size > 1:
                                chunk_buffers = (
                                    cp.empty((chunk_size, *projection.shape), dtype=projection.dtype),
                                    cp.empty((chunk_size, *projection.shape), dtype=projection.dtype),
                                )
                        else:
                            result_data = cp.empty(result_shape, dtype=projection.dtype)
                    if result_residency == "host":
                        if int(chunk_size) == 1:
                            if pending_copy_done is not None:
                                pending_copy_done.synchronize()
                                pending_copy_done = None
                                if pending_host_projection is not None:
                                    del pending_host_projection
                                    pending_host_projection = None
                            compute_done = cp.cuda.Event()
                            compute_done.record()
                            copy_stream.wait_event(compute_done)
                            with copy_stream:
                                projection.get(out=result_data[energy_index], blocking=False)
                                pending_copy_done = cp.cuda.Event()
                                pending_copy_done.record()
                            pending_host_projection = projection
                        else:
                            chunk_buffer = chunk_buffers[active_chunk_buffer_index]
                            chunk_buffer[active_chunk_fill] = projection
                            active_chunk_fill += 1
                            del projection
                            if active_chunk_fill == chunk_buffer.shape[0] or energy_index == len(energies) - 1:
                                chunk_stop = active_chunk_start + active_chunk_fill
                                if pending_copy_done is not None:
                                    pending_copy_done.synchronize()
                                    pending_copy_done = None
                                compute_done = cp.cuda.Event()
                                compute_done.record()
                                copy_stream.wait_event(compute_done)
                                with copy_stream:
                                    chunk_buffer[:active_chunk_fill].get(
                                        out=result_data[active_chunk_start:chunk_stop],
                                        blocking=False,
                                    )
                                    pending_copy_done = cp.cuda.Event()
                                    pending_copy_done.record()
                                active_chunk_start = chunk_stop
                                active_chunk_fill = 0
                                active_chunk_buffer_index = 1 - active_chunk_buffer_index
                    else:
                        result_data[energy_index] = projection
                        del projection
        finally:
            if pending_copy_done is not None:
                pending_copy_done.synchronize()
                pending_copy_done = None
            if pending_host_projection is not None:
                del pending_host_projection
            if chunk_buffers is not None:
                for chunk_buffer in chunk_buffers:
                    del chunk_buffer
            if copy_stream is not None:
                copy_stream.synchronize()
            if window is not None:
                del window
            del runtime_materials
        result = recorder.measure(
            "F",
            lambda: self._assemble_and_retain_result(
                morphology=morphology,
                result_data=result_data,
                energies=energies,
            ),
        )
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

    def _result_residency(self, morphology) -> str:
        return str(morphology.backend_options.get("result_residency", "host"))

    def _result_chunk_size(
        self,
        morphology,
        *,
        energy_count: int,
    ) -> int:
        configured = morphology.backend_options.get("result_chunk_size")
        if configured is None:
            configured = 1
        return max(1, min(int(configured), int(energy_count)))

    def _allocate_host_result_buffer(self, shape, dtype):
        from cupyx import empty_pinned

        return empty_pinned(shape, dtype=dtype)

    def _energy_progress_bar_enabled(
        self,
        morphology,
        energies: tuple[float, ...],
        *,
        stdout: bool,
        stderr: bool,
        stream: Any | None = None,
    ) -> bool:
        if not bool(morphology.backend_options.get("energy_progress_bar", False)):
            return False
        if len(energies) <= 1:
            return False
        if not stdout:
            return False
        if not stderr:
            return False
        stream = sys.stderr if stream is None else stream
        isatty = getattr(stream, "isatty", None)
        if isatty is None:
            return False
        try:
            return bool(isatty())
        except Exception:
            return False

    def _notebook_progress_bar_enabled(
        self,
        morphology,
        energies: tuple[float, ...],
        *,
        stdout: bool,
        stderr: bool,
        stream: Any | None = None,
    ) -> bool:
        if not bool(morphology.backend_options.get("energy_progress_bar", False)):
            return False
        if len(energies) <= 1:
            return False
        if not stdout:
            return False
        if not stderr:
            return False
        if self._energy_progress_bar_enabled(
            morphology,
            energies,
            stdout=stdout,
            stderr=stderr,
            stream=stream,
        ):
            return False
        if "ipykernel" not in sys.modules:
            return False
        try:
            ipython = importlib.import_module("IPython")
        except Exception:
            return False
        get_ipython = getattr(ipython, "get_ipython", None)
        if get_ipython is None:
            return False
        try:
            shell = get_ipython()
        except Exception:
            return False
        return shell is not None

    def _set_energy_progress_value(self, progress, energy: float) -> None:
        set_postfix_str = getattr(progress, "set_postfix_str", None)
        if set_postfix_str is None:
            return
        try:
            set_postfix_str(f"{float(energy):.1f} eV", refresh=False)
        except TypeError:
            set_postfix_str(f"{float(energy):.1f} eV")

    @contextmanager
    def _iter_completed_energies(
        self,
        morphology,
        energies: tuple[float, ...],
        *,
        stdout: bool,
        stderr: bool,
        stream: Any | None = None,
    ):
        progress = None
        stream = sys.stderr if stream is None else stream
        if self._energy_progress_bar_enabled(
            morphology,
            energies,
            stdout=stdout,
            stderr=stderr,
            stream=stream,
        ):
            tqdm = importlib.import_module("tqdm").tqdm
            progress = tqdm(
                total=len(energies),
                ascii=True,
                colour="#7DF9FF",
                desc="Energy",
                file=stream,
                unit="energy",
            )
        elif self._notebook_progress_bar_enabled(
            morphology,
            energies,
            stdout=stdout,
            stderr=stderr,
            stream=stream,
        ):
            tqdm = importlib.import_module("tqdm.auto").tqdm
            progress = tqdm(
                total=len(energies),
                desc="Energy",
                file=stream,
                unit="energy",
            )

        if progress is not None:
            def wrapped():
                for energy_index, energy in enumerate(energies):
                    self._set_energy_progress_value(progress, energy)
                    yield energy_index, energy
                    progress.update(1)

            iterator = wrapped()
        else:
            iterator = enumerate(energies)
        try:
            yield iterator
        finally:
            if progress is not None:
                progress.close()

    def _assemble_and_retain_result(self, morphology, result_data, energies):
        result = CupyScatteringResult(
            data=result_data,
            energies=energies,
            phys_size=float(morphology.PhysSize),
            num_zyx=tuple(int(v) for v in morphology.NumZYX),
        )
        morphology._backend_result = result
        morphology.scatteringPattern = result
        self._update_kernel_reports(morphology)
        return result

    def validate_all(self, morphology, *, quiet: bool = True) -> None:
        morphology.check_materials(quiet=quiet)
        self._validate_supported_config(morphology)
        if not quiet:
            print("CuPy backend validation completed successfully.")

    def release(self, morphology) -> None:
        self._release_morphology_owned_cupy_state(morphology)
        self._clear_process_global_cupy_state()
        self._drain_cupy_memory_pools()

    def _release_morphology_owned_cupy_state(self, morphology) -> None:
        result = getattr(morphology, "_backend_result", None)
        if result is not None and hasattr(result, "release"):
            result.release()
        morphology._backend_result = None
        morphology.scatteringPattern = None
        if hasattr(morphology, "scattering_data"):
            morphology.scattering_data = None

        # Clear any direct CuPy arrays hanging off the morphology instance.
        for attr_name, value in tuple(morphology.__dict__.items()):
            try:
                if inspect_array(value)["namespace"] == "cupy":
                    morphology.__dict__[attr_name] = None
            except Exception:
                continue

        morphology._backend_runtime_state.clear()
        morphology.last_runtime_staging_report = []
        morphology.last_kernel_backend_report = {}
        morphology.last_kernel_preload_report = {}

    def _clear_process_global_cupy_state(self) -> None:
        _CUPY_KERNEL_CACHE.clear()
        _CUPY_KERNEL_BACKEND_REPORT.clear()

        igor_order_cache = getattr(self, "_igor_order_cache", None)
        if igor_order_cache is not None:
            igor_order_cache.clear()

    def _drain_cupy_memory_pools(self) -> None:
        try:
            import cupy as cp
        except Exception:
            return

        def _best_effort(action):
            try:
                action()
            except Exception:
                pass

        _best_effort(gc.collect)
        _best_effort(cp.clear_memo)
        if hasattr(cp, "fft") and hasattr(cp.fft, "config"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                _best_effort(cp.fft.config.clear_plan_cache)

        for _ in range(2):
            _best_effort(cp.cuda.Stream.null.synchronize)
            _best_effort(cp.cuda.runtime.deviceSynchronize)
            _best_effort(cp.get_default_memory_pool().free_all_blocks)
            _best_effort(cp.get_default_pinned_memory_pool().free_all_blocks)
            _best_effort(gc.collect)

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
            if self._uses_host_reusable_precompute(morphology):
                families.append("direct_polarization_precomputed")
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
            elif family == "direct_polarization_precomputed":
                self._direct_precomputed_kernel_float32(morphology, cp)
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

    def _is_runtime_zero_field_array(self, value, cp) -> bool:
        if value is None:
            return False
        info = inspect_array(value)
        namespace = info["namespace"]
        if namespace == "numpy":
            return bool(np.count_nonzero(np.asarray(value)) == 0)
        if namespace == "cupy":
            return bool(int(cp.count_nonzero(value).item()) == 0)
        raise TypeError(
            "cupy-rsoxs runtime zero-field detection requires numpy or cupy arrays, "
            f"received namespace {namespace!r}."
        )

    def _supports_runtime_zero_field_shortcut(self, morphology) -> bool:
        return self._execution_path(morphology) in {"tensor_coeff", "direct_polarization"}

    def _uses_host_reusable_precompute(self, morphology) -> bool:
        if str(getattr(morphology, "resident_mode", "")) != "host":
            return False
        if self._execution_path(morphology) not in {"tensor_coeff", "direct_polarization"}:
            return False
        runtime_dtype = str(morphology._runtime_compute_contract.get("runtime_dtype", "float32"))
        return runtime_dtype == "float32"

    def _direct_isotropic_mode(self, morphology) -> str | None:
        if self._execution_path(morphology) != "direct_polarization":
            return None
        mode = morphology.backend_options.get("direct_isotropic_mode")
        return None if mode is None else str(mode)

    def _uses_direct_cached_isotropic_base(self, morphology) -> bool:
        return self._direct_isotropic_mode(morphology) == "cached_base"

    def _material_is_runtime_zero_field(self, morphology, material, cp) -> bool:
        if not self._supports_runtime_zero_field_shortcut(morphology):
            return False
        if morphology._material_is_explicit_isotropic(material):
            return False
        return all(
            self._is_runtime_zero_field_array(getattr(material, field_name), cp)
            for field_name in ("S", "theta", "psi")
        )

    def _stage_runtime_field(
        self,
        value,
        *,
        field_name: str,
        material_id: int,
        runtime_contract,
        staging_reports: list[Any],
    ):
        plan = assess_array_for_backend_runtime(
            value,
            backend_name=self.name,
            field_name=field_name,
            material_id=material_id,
            contract=runtime_contract,
        )
        staging_reports.append(plan)
        return coerce_array_for_backend(value, plan)

    def _build_precomputed_runtime_fields_gpu(self, *, vfrac, s, theta, psi, cp) -> dict[str, Any]:
        phi_a = cp.empty_like(vfrac)
        sx = cp.empty_like(vfrac)
        sy = cp.empty_like(vfrac)
        sz = cp.empty_like(vfrac)
        total = np.uint64(vfrac.size)
        threads = 256
        blocks = (vfrac.size + threads - 1) // threads
        self._host_reusable_precompute_float32_kernel(cp)(
            (blocks,),
            (threads,),
            (vfrac, s, theta, psi, phi_a, sx, sy, sz, total),
        )
        return {
            "Vfrac": vfrac,
            "phi_a": phi_a,
            "sx": sx,
            "sy": sy,
            "sz": sz,
        }

    def _runtime_material_views(self, morphology, cp):
        runtime_contract = morphology._runtime_compute_contract
        staging_reports = []
        runtime_materials = []
        use_host_reusables = self._uses_host_reusable_precompute(morphology)
        for material_id, material in morphology.materials.items():
            is_full_isotropic = morphology._material_is_explicit_isotropic(material) or (
                self._material_is_runtime_zero_field(morphology, material, cp)
            )
            staged_fields = {
                "Vfrac": None,
                "S": None,
                "theta": None,
                "psi": None,
                "phi_a": None,
                "sx": None,
                "sy": None,
                "sz": None,
            }
            if is_full_isotropic:
                staged_fields["Vfrac"] = self._stage_runtime_field(
                    material.Vfrac,
                    field_name="Vfrac",
                    material_id=material_id,
                    runtime_contract=runtime_contract,
                    staging_reports=staging_reports,
                )
            elif use_host_reusables:
                staged_fields["Vfrac"] = self._stage_runtime_field(
                    material.Vfrac,
                    field_name="Vfrac",
                    material_id=material_id,
                    runtime_contract=runtime_contract,
                    staging_reports=staging_reports,
                )
                raw_s = self._stage_runtime_field(
                    material.S,
                    field_name="S",
                    material_id=material_id,
                    runtime_contract=runtime_contract,
                    staging_reports=staging_reports,
                )
                raw_theta = self._stage_runtime_field(
                    material.theta,
                    field_name="theta",
                    material_id=material_id,
                    runtime_contract=runtime_contract,
                    staging_reports=staging_reports,
                )
                raw_psi = self._stage_runtime_field(
                    material.psi,
                    field_name="psi",
                    material_id=material_id,
                    runtime_contract=runtime_contract,
                    staging_reports=staging_reports,
                )
                precomputed = self._build_precomputed_runtime_fields_gpu(
                    vfrac=staged_fields["Vfrac"],
                    s=raw_s,
                    theta=raw_theta,
                    psi=raw_psi,
                    cp=cp,
                )
                staged_fields.update(precomputed)
                del raw_s, raw_theta, raw_psi
            else:
                for field_name in ("Vfrac", "S", "theta", "psi"):
                    staged_fields[field_name] = self._stage_runtime_field(
                        getattr(material, field_name),
                        field_name=field_name,
                        material_id=material_id,
                        runtime_contract=runtime_contract,
                        staging_reports=staging_reports,
                    )

            runtime_materials.append(
                _RuntimeMaterialView(
                    materialID=material_id,
                    opt_constants=material.opt_constants,
                    Vfrac=staged_fields["Vfrac"],
                    S=staged_fields["S"],
                    theta=staged_fields["theta"],
                    psi=staged_fields["psi"],
                    is_full_isotropic=is_full_isotropic,
                    phi_a=staged_fields["phi_a"],
                    sx=staged_fields["sx"],
                    sy=staged_fields["sy"],
                    sz=staged_fields["sz"],
                )
            )

        morphology.last_runtime_staging_report = staging_reports
        return tuple(runtime_materials)

    def _execution_path(self, morphology) -> str:
        return str(morphology.backend_options.get("execution_path", "direct_polarization"))

    def _run_single_energy(self, morphology, runtime_materials, energy, cp, ndimage, window, recorder):
        execution_path = self._execution_path(morphology)
        try:
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
        finally:
            self._discard_detector_projection_geometry(
                morphology,
                energy,
                shape_override=self._segment_c_shape_override(morphology),
            )

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
        return self._project_from_direct_polarization(
            morphology=morphology,
            runtime_materials=runtime_materials,
            energy=energy,
            cp=cp,
            ndimage=ndimage,
            window=window,
            angle_family_plan=angle_family_plan,
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
            cp.divide(projection_average, np.float32(num_angles), out=projection_average)
            return projection_average
        nonzero = valid_counts != 0
        denom = cp.maximum(valid_counts, 1)
        cp.divide(projection_average, denom, out=projection_average, casting="unsafe")
        cp.copyto(projection_average, np.float32(0.0), where=~nonzero)
        del nonzero, denom
        return projection_average

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

    def _has_precomputed_runtime_reusables(self, material) -> bool:
        return all(getattr(material, name, None) is not None for name in ("phi_a", "sx", "sy", "sz"))

    def _compute_nt_components(self, runtime_materials, energy, cp, required_components=None):
        dtype_name = cp.dtype(runtime_materials[0].Vfrac.dtype).name
        if dtype_name == "float16":
            return self._compute_nt_components_half_input(
                runtime_materials,
                energy,
                cp,
                required_components=required_components,
            )

        required = {0, 1, 2, 3, 4} if required_components is None else set(required_components)
        if dtype_name != "float32":
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

                if self._has_precomputed_runtime_reusables(material):
                    phi_a = material.phi_a
                    sx = material.sx
                    sy = material.sy
                    sz = material.sz
                else:
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
                if not self._has_precomputed_runtime_reusables(material):
                    del phi_a, sx, sy, sz

            return nt

        need0 = np.int32(0 in required)
        need1 = np.int32(1 in required)
        need2 = np.int32(2 in required)
        need3 = np.int32(3 in required)
        need4 = np.int32(4 in required)
        shape = tuple(int(v) for v in runtime_materials[0].Vfrac.shape)
        nt = cp.zeros((5, *shape), dtype=cp.complex64)
        nt0, nt1, nt2, nt3, nt4 = (nt[idx] for idx in range(5))
        threads = 256
        isotropic_kernel = self._nt_accumulate_isotropic_float32_kernel(cp)
        anisotropic_kernel = self._nt_accumulate_anisotropic_float32_kernel(cp)
        anisotropic_precomputed_kernel = self._nt_accumulate_anisotropic_precomputed_float32_kernel(cp)

        for material in runtime_materials:
            isotropic_diag, aligned_base, anisotropic_delta = self._material_optical_scalars(
                material,
                energy,
            )
            total = np.uint64(material.Vfrac.size)
            blocks = (material.Vfrac.size + threads - 1) // threads

            if material.is_full_isotropic:
                isotropic_kernel(
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

            if self._has_precomputed_runtime_reusables(material):
                anisotropic_precomputed_kernel(
                    (blocks,),
                    (threads,),
                    (
                        material.Vfrac,
                        material.phi_a,
                        material.sx,
                        material.sy,
                        material.sz,
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
            else:
                anisotropic_kernel(
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

            if self._has_precomputed_runtime_reusables(material):
                phi_a = material.phi_a
                sx = material.sx
                sy = material.sy
                sz = material.sz
            else:
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

            if not self._has_precomputed_runtime_reusables(material):
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

    def _nt_accumulate_isotropic_float32_kernel(self, cp):
        kernel = _CUPY_KERNEL_CACHE.get("nt_accumulate_isotropic_float32")
        if kernel is not None:
            return kernel

        kernel = cp.RawKernel(
            r"""
            extern "C" __global__
            void nt_accumulate_isotropic_float32(
                const float* vfrac,
                const float2 isotropic_diag,
                const int need0,
                const int need3,
                float2* nt0,
                float2* nt3,
                const unsigned long long total
            ) {
                const unsigned long long idx =
                    (unsigned long long)blockDim.x * (unsigned long long)blockIdx.x
                    + (unsigned long long)threadIdx.x;
                if (idx >= total) {
                    return;
                }

                const float vf = vfrac[idx];
                if (need0) {
                    nt0[idx].x += vf * isotropic_diag.x;
                    nt0[idx].y += vf * isotropic_diag.y;
                }
                if (need3) {
                    nt3[idx].x += vf * isotropic_diag.x;
                    nt3[idx].y += vf * isotropic_diag.y;
                }
            }
            """,
            "nt_accumulate_isotropic_float32",
        )
        _CUPY_KERNEL_CACHE["nt_accumulate_isotropic_float32"] = kernel
        return kernel

    def _host_reusable_precompute_float32_kernel(self, cp):
        kernel = _CUPY_KERNEL_CACHE.get("host_reusable_precompute_float32")
        if kernel is not None:
            return kernel

        kernel = cp.RawKernel(
            r"""
            extern "C" __global__
            void host_reusable_precompute_float32(
                const float* vfrac,
                const float* s,
                const float* theta,
                const float* psi,
                float* phi_a,
                float* sx,
                float* sy,
                float* sz,
                const unsigned long long total
            ) {
                const unsigned long long idx =
                    (unsigned long long)blockDim.x * (unsigned long long)blockIdx.x
                    + (unsigned long long)threadIdx.x;
                if (idx >= total) {
                    return;
                }

                const float theta_i = theta[idx];
                const float psi_i = psi[idx];
                const float sin_theta = sinf(theta_i);
                phi_a[idx] = vfrac[idx] * s[idx];
                sx[idx] = cosf(psi_i) * sin_theta;
                sy[idx] = sinf(psi_i) * sin_theta;
                sz[idx] = cosf(theta_i);
            }
            """,
            "host_reusable_precompute_float32",
        )
        _CUPY_KERNEL_CACHE["host_reusable_precompute_float32"] = kernel
        return kernel

    def _nt_accumulate_anisotropic_float32_kernel(self, cp):
        kernel = _CUPY_KERNEL_CACHE.get("nt_accumulate_anisotropic_float32")
        if kernel is not None:
            return kernel

        kernel = cp.RawKernel(
            r"""
            extern "C" __global__
            void nt_accumulate_anisotropic_float32(
                const float* vfrac,
                const float* s,
                const float* theta,
                const float* psi,
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
            ) {
                const unsigned long long idx =
                    (unsigned long long)blockDim.x * (unsigned long long)blockIdx.x
                    + (unsigned long long)threadIdx.x;
                if (idx >= total) {
                    return;
                }

                const float vf = vfrac[idx];
                if (need0 || need3) {
                    const float iso_x = vf * isotropic_diag.x;
                    const float iso_y = vf * isotropic_diag.y;
                    if (need0) {
                        nt0[idx].x += iso_x;
                        nt0[idx].y += iso_y;
                    }
                    if (need3) {
                        nt3[idx].x += iso_x;
                        nt3[idx].y += iso_y;
                    }
                }

                const float phi = vf * s[idx];
                const float theta_i = theta[idx];
                const float psi_i = psi[idx];
                const float sin_theta = sinf(theta_i);
                const float sx = cosf(psi_i) * sin_theta;
                const float sy = sinf(psi_i) * sin_theta;
                const float sz = cosf(theta_i);

                if (need0) {
                    nt0[idx].x += phi * (aligned_base.x + anisotropic_delta.x * sx * sx);
                    nt0[idx].y += phi * (aligned_base.y + anisotropic_delta.y * sx * sx);
                }
                if (need1) {
                    nt1[idx].x += phi * anisotropic_delta.x * sx * sy;
                    nt1[idx].y += phi * anisotropic_delta.y * sx * sy;
                }
                if (need2) {
                    nt2[idx].x += phi * anisotropic_delta.x * sx * sz;
                    nt2[idx].y += phi * anisotropic_delta.y * sx * sz;
                }
                if (need3) {
                    nt3[idx].x += phi * (aligned_base.x + anisotropic_delta.x * sy * sy);
                    nt3[idx].y += phi * (aligned_base.y + anisotropic_delta.y * sy * sy);
                }
                if (need4) {
                    nt4[idx].x += phi * anisotropic_delta.x * sy * sz;
                    nt4[idx].y += phi * anisotropic_delta.y * sy * sz;
                }
            }
            """,
            "nt_accumulate_anisotropic_float32",
        )
        _CUPY_KERNEL_CACHE["nt_accumulate_anisotropic_float32"] = kernel
        return kernel

    def _nt_accumulate_anisotropic_precomputed_float32_kernel(self, cp):
        kernel = _CUPY_KERNEL_CACHE.get("nt_accumulate_anisotropic_precomputed_float32")
        if kernel is not None:
            return kernel

        kernel = cp.RawKernel(
            r"""
            extern "C" __global__
            void nt_accumulate_anisotropic_precomputed_float32(
                const float* vfrac,
                const float* phi_a,
                const float* sx,
                const float* sy,
                const float* sz,
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
            ) {
                const unsigned long long idx =
                    (unsigned long long)blockDim.x * (unsigned long long)blockIdx.x
                    + (unsigned long long)threadIdx.x;
                if (idx >= total) {
                    return;
                }

                const float vf = vfrac[idx];
                if (need0 || need3) {
                    const float iso_x = vf * isotropic_diag.x;
                    const float iso_y = vf * isotropic_diag.y;
                    if (need0) {
                        nt0[idx].x += iso_x;
                        nt0[idx].y += iso_y;
                    }
                    if (need3) {
                        nt3[idx].x += iso_x;
                        nt3[idx].y += iso_y;
                    }
                }

                const float phi = phi_a[idx];
                const float sx_i = sx[idx];
                const float sy_i = sy[idx];
                const float sz_i = sz[idx];

                if (need0) {
                    nt0[idx].x += phi * (aligned_base.x + anisotropic_delta.x * sx_i * sx_i);
                    nt0[idx].y += phi * (aligned_base.y + anisotropic_delta.y * sx_i * sx_i);
                }
                if (need1) {
                    nt1[idx].x += phi * anisotropic_delta.x * sx_i * sy_i;
                    nt1[idx].y += phi * anisotropic_delta.y * sx_i * sy_i;
                }
                if (need2) {
                    nt2[idx].x += phi * anisotropic_delta.x * sx_i * sz_i;
                    nt2[idx].y += phi * anisotropic_delta.y * sx_i * sz_i;
                }
                if (need3) {
                    nt3[idx].x += phi * (aligned_base.x + anisotropic_delta.x * sy_i * sy_i);
                    nt3[idx].y += phi * (aligned_base.y + anisotropic_delta.y * sy_i * sy_i);
                }
                if (need4) {
                    nt4[idx].x += phi * anisotropic_delta.x * sy_i * sz_i;
                    nt4[idx].y += phi * anisotropic_delta.y * sy_i * sz_i;
                }
            }
            """,
            "nt_accumulate_anisotropic_precomputed_float32",
        )
        _CUPY_KERNEL_CACHE["nt_accumulate_anisotropic_precomputed_float32"] = kernel
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
        isotropic_base_field = None
        if self._uses_direct_cached_isotropic_base(morphology):
            isotropic_base_field = recorder.measure(
                "B",
                lambda: self._compute_direct_isotropic_base_field(
                    runtime_materials=runtime_materials,
                    energy=energy,
                    cp=cp,
                ),
            )
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
            del fft_x, fft_y, fft_z
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
            del projection
        return self._finalize_rotation_average(cp, projection_average, valid_counts, num_angles)

    def _compute_direct_isotropic_base_field(self, runtime_materials, energy, cp):
        if not runtime_materials:
            return None
        if cp.dtype(runtime_materials[0].Vfrac.dtype).name != "float32":
            return None
        return self._compute_direct_isotropic_base_field_float32(runtime_materials, energy, cp)

    def _compute_direct_isotropic_base_field_float32(self, runtime_materials, energy, cp):
        isotropic_materials = [material for material in runtime_materials if material.is_full_isotropic]
        if not isotropic_materials:
            return None

        shape = tuple(int(v) for v in isotropic_materials[0].Vfrac.shape)
        base = cp.zeros(shape, dtype=cp.complex64)
        threads = 256
        kernel = self._direct_isotropic_base_accumulate_float32_kernel(cp)
        for material in isotropic_materials:
            isotropic_diag, _aligned_base, _anisotropic_delta = self._material_optical_scalars(
                material,
                energy,
            )
            total = np.uint64(material.Vfrac.size)
            blocks = (material.Vfrac.size + threads - 1) // threads
            kernel(
                (blocks,),
                (threads,),
                (
                    material.Vfrac,
                    isotropic_diag,
                    base,
                    total,
                ),
            )
        return base


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
        mx = angle_plan.mx
        my = angle_plan.my
        if isotropic_base_field is not None:
            p_x = cp.empty(shape, dtype=cp.complex64)
            p_y = cp.empty(shape, dtype=cp.complex64)
            cp.multiply(isotropic_base_field, np.float32(mx), out=p_x)
            cp.multiply(isotropic_base_field, np.float32(my), out=p_y)
        else:
            p_x = cp.zeros(shape, dtype=cp.complex64)
            p_y = cp.zeros(shape, dtype=cp.complex64)
        p_z = cp.zeros(shape, dtype=cp.complex64)
        generic_kernel = self._direct_generic_kernel_float32(morphology, cp)
        precomputed_kernel = self._direct_precomputed_kernel_float32(morphology, cp)
        isotropic_kernel = self._direct_isotropic_kernel_float32(morphology, cp)

        for material in runtime_materials:
            isotropic_diag, aligned_base, anisotropic_delta = self._material_optical_scalars(
                material,
                energy,
            )
            vfrac = material.Vfrac
            if material.is_full_isotropic:
                if isotropic_base_field is not None:
                    continue
                isotropic_kernel(
                    ((vfrac.size + 255) // 256,),
                    (256,),
                    (
                        vfrac,
                        isotropic_diag,
                        np.float32(mx),
                        np.float32(my),
                        p_x,
                        p_y,
                        np.uint64(vfrac.size),
                    ),
                )
                continue

            if self._has_precomputed_runtime_reusables(material):
                precomputed_kernel(
                    ((vfrac.size + 255) // 256,),
                    (256,),
                    (
                        vfrac,
                        material.phi_a,
                        material.sx,
                        material.sy,
                        material.sz,
                        isotropic_diag,
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
            else:
                generic_kernel(
                    ((vfrac.size + 255) // 256,),
                    (256,),
                    (
                        vfrac,
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
                        np.uint64(vfrac.size),
                    ),
                )

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

            if self._has_precomputed_runtime_reusables(material):
                phi_a = material.phi_a
                sx = material.sx
                sy = material.sy
                sz = material.sz
            else:
                phi_a = material.Vfrac * material.S
                sx, sy, sz = self._orientation_components(material, cp)
            field_projection = sx * mx + sy * my

            contrib_x = phi_a * (mx * aligned_base + anisotropic_delta * sx * field_projection)
            p_x += cp.sum(contrib_x, axis=0, dtype=cp.complex64, keepdims=True) / z_count
            del contrib_x

            contrib_y = phi_a * (my * aligned_base + anisotropic_delta * sy * field_projection)
            p_y += cp.sum(contrib_y, axis=0, dtype=cp.complex64, keepdims=True) / z_count
            del contrib_y

            contrib_z = phi_a * (anisotropic_delta * sz * field_projection)
            p_z += cp.sum(contrib_z, axis=0, dtype=cp.complex64, keepdims=True) / z_count
            del contrib_z

            if not self._has_precomputed_runtime_reusables(material):
                del phi_a, sx, sy, sz
            del field_projection

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
            self._direct_isotropic_kernel_float32(morphology, cp)
            return self._direct_generic_kernel_float32(morphology, cp)
        raise TypeError(
            "cupy-rsoxs direct_polarization received unsupported runtime morphology "
            f"dtype {dtype_name!r}."
        )

    def _direct_isotropic_kernel_float32(self, morphology, cp):
        return self._build_rawkernel_with_fallback(
            cp,
            family="direct_polarization_generic",
            cache_key_base="direct_polarization_isotropic_complex64_float32",
            source=r"""
            extern "C" __global__
            void direct_polarization_isotropic_complex64_float32(
                const float* vfrac,
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

                const float vf = vfrac[idx];
                p_x[idx].x += vf * isotropic_diag.x * mx;
                p_x[idx].y += vf * isotropic_diag.y * mx;
                p_y[idx].x += vf * isotropic_diag.x * my;
                p_y[idx].y += vf * isotropic_diag.y * my;
            }
            """,
            kernel_name="direct_polarization_isotropic_complex64_float32",
            requested_backend=self._rawkernel_backend_option(morphology, "direct_polarization_generic"),
        )

    def _direct_isotropic_base_accumulate_float32_kernel(self, cp):
        kernel = _CUPY_KERNEL_CACHE.get("direct_isotropic_base_accumulate_float32")
        if kernel is not None:
            return kernel

        kernel = cp.RawKernel(
            r"""
            extern "C" __global__
            void direct_isotropic_base_accumulate_float32(
                const float* vfrac,
                const float2 isotropic_diag,
                float2* base,
                const unsigned long long total
            ) {
                const unsigned long long idx =
                    (unsigned long long)blockDim.x * (unsigned long long)blockIdx.x
                    + (unsigned long long)threadIdx.x;
                if (idx >= total) {
                    return;
                }

                const float vf = vfrac[idx];
                base[idx].x += vf * isotropic_diag.x;
                base[idx].y += vf * isotropic_diag.y;
            }
            """,
            "direct_isotropic_base_accumulate_float32",
        )
        _CUPY_KERNEL_CACHE["direct_isotropic_base_accumulate_float32"] = kernel
        return kernel

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

                const float vf = vfrac[idx];
                p_x[idx].x += vf * isotropic_diag.x * mx;
                p_x[idx].y += vf * isotropic_diag.y * mx;
                p_y[idx].x += vf * isotropic_diag.x * my;
                p_y[idx].y += vf * isotropic_diag.y * my;

                const float phi = vf * s[idx];
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

    def _direct_precomputed_kernel_float32(self, morphology, cp):
        return self._build_rawkernel_with_fallback(
            cp,
            family="direct_polarization_precomputed",
            cache_key_base="direct_polarization_precomputed_complex64",
            source=r"""
            extern "C" __global__
            void direct_polarization_precomputed_complex64(
                const float* vfrac,
                const float* phi_a,
                const float* sx,
                const float* sy,
                const float* sz,
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

                const float vf = vfrac[idx];
                p_x[idx].x += vf * isotropic_diag.x * mx;
                p_x[idx].y += vf * isotropic_diag.y * mx;
                p_y[idx].x += vf * isotropic_diag.x * my;
                p_y[idx].y += vf * isotropic_diag.y * my;

                const float phi = phi_a[idx];
                const float sx_i = sx[idx];
                const float sy_i = sy[idx];
                const float sz_i = sz[idx];
                const float field_projection = sx_i * mx + sy_i * my;

                p_x[idx].x += phi * (mx * aligned_base.x + anisotropic_delta.x * sx_i * field_projection);
                p_x[idx].y += phi * (mx * aligned_base.y + anisotropic_delta.y * sx_i * field_projection);
                p_y[idx].x += phi * (my * aligned_base.x + anisotropic_delta.x * sy_i * field_projection);
                p_y[idx].y += phi * (my * aligned_base.y + anisotropic_delta.y * sy_i * field_projection);
                p_z[idx].x += phi * (anisotropic_delta.x * sz_i * field_projection);
                p_z[idx].y += phi * (anisotropic_delta.y * sz_i * field_projection);
            }
            """,
            kernel_name="direct_polarization_precomputed_complex64",
            requested_backend=self._rawkernel_backend_option(morphology, "direct_polarization_precomputed"),
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

        plan = self._direct_polarization_fft_plan(morphology, cp, p_x)
        cufft = cp.cuda.cufft
        for arr in (p_x, p_y, p_z):
            # Match the CyRSoXS Segment C structure more closely: one persistent
            # complex buffer per polarization component, cuFFT in place, then an
            # in-place Igor-order swap instead of a second full shifted volume.
            plan.fft(arr, arr, cufft.CUFFT_FORWARD)
            self._replace_dc_component(arr)
            self._igor_shift_inplace(arr, morphology, cp)
        return p_x, p_y, p_z

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
            valid_int = valid.astype(cp.int32)
            if valid_counts is None:
                valid_counts = valid_int
            else:
                cp.add(valid_counts, valid_int, out=valid_counts)
                del valid_int
            cp.nan_to_num(rotated, copy=False, nan=0.0)
            del valid
        if projection_average is None:
            projection_average = rotated
        else:
            cp.add(projection_average, rotated, out=projection_average)
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

    def _direct_polarization_fft_plan(self, morphology, cp, arr):
        from cupyx.scipy.fftpack import get_fft_plan

        cache_key = (
            "direct_polarization_fft_plan_complex64",
            tuple(int(v) for v in arr.shape),
            cp.dtype(arr.dtype).name,
        )
        plan = morphology._backend_runtime_state.get(cache_key)
        if plan is None:
            plan = get_fft_plan(arr, axes=(0, 1, 2), value_type="C2C")
            morphology._backend_runtime_state[cache_key] = plan
        return plan

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

    def _igor_shift_inplace_kernel(self, morphology, cp):
        return self._build_rawkernel_with_fallback(
            cp,
            family="igor_shift",
            cache_key_base="igor_shift_inplace_complex64",
            source=r"""
                extern "C" __global__
                void igor_shift_inplace_complex64(
                    float2* data,
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

                    const int mid_x = xdim / 2;
                    const int mid_y = ydim / 2;
                    const int mid_z = zdim / 2;

                    const int swap_x = (x <= mid_x) ? (mid_x - x) : (xdim + (mid_x - x));
                    const int swap_y = (y <= mid_y) ? (mid_y - y) : (ydim + (mid_y - y));
                    const int swap_z = (z <= mid_z) ? (mid_z - z) : (zdim + (mid_z - z));

                    const unsigned long long swap_idx =
                        ((unsigned long long)swap_z * (unsigned long long)ydim
                         + (unsigned long long)swap_y) * (unsigned long long)xdim
                        + (unsigned long long)swap_x;

                    if (swap_idx > idx) {
                        const float2 temp = data[swap_idx];
                        data[swap_idx] = data[idx];
                        data[idx] = temp;
                    }
                }
                """,
            kernel_name="igor_shift_inplace_complex64",
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

    def _igor_shift_inplace(self, arr, morphology, cp):
        total = int(arr.size)
        threads = 256
        blocks = (total + threads - 1) // threads
        self._igor_shift_inplace_kernel(morphology, cp)(
            (blocks,),
            (threads,),
            (
                arr,
                np.int32(arr.shape[0]),
                np.int32(arr.shape[1]),
                np.int32(arr.shape[2]),
                np.uint64(total),
            ),
        )
        return arr

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

    def _discard_detector_projection_geometry(self, morphology, energy, shape_override=None):
        z, y, x = self._shape_tuple(morphology, shape_override=shape_override)
        cache_key = (
            "detector_projection_geometry_current",
            z,
            y,
            x,
            float(morphology.PhysSize),
            float(energy),
        )
        morphology._backend_runtime_state.pop(cache_key, None)

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
