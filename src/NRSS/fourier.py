"""Prepared CuPy Fourier samples of real scalar fields on integer ZYX grids."""

from dataclasses import dataclass
import math
from operator import index
from threading import get_ident
from typing import Any

import numpy as np

__all__ = ["PreparedSampler", "prepare_sampler"]


def _r2c_addresses(xp, shape, z, y, x):
    nz, ny, nx = shape
    # R2C omits negative X. Its Hermitian partner reverses all three axes.
    reflect = x > nx // 2
    z = xp.where(reflect, (-z) % nz, z)
    y = xp.where(reflect, (-y) % ny, y)
    x = xp.where(reflect, (-x) % nx, x)
    return (z * ny + y) * (nx // 2 + 1) + x, reflect


def _repair_native_dc(transformed, shape):
    nz, ny, _ = shape
    neighbors = [transformed[0, 0, 1], transformed[0, 1, 0],
                 transformed[0, 0, 1].conj(), transformed[0, ny - 1, 0]]
    if nz > 1:
        neighbors += [transformed[1, 0, 0], transformed[nz - 1, 0, 0]]
    transformed[0, 0, 0] = sum(neighbors).real / np.float32(len(neighbors))


@dataclass(frozen=True)
class _PreparedCapture:
    """Capture using prepared addresses or an existing virtual evaluator."""

    cp: Any
    native: Any
    reflect: Any
    virtual: Any = None
    repair_dc: bool = True

    def capture_into(self, field, out, *, execution=None, recorder=None, **virtual_options):
        cp = self.cp
        if self.virtual is not None:
            if self.repair_dc:
                self.virtual.capture(cp, field, out, recorder, **virtual_options)
            else:
                self.virtual.capture(cp, field, out, recorder, repair_dc=False, **virtual_options)
            return

        transformed = cp.fft.rfftn(field)
        if execution is not None:
            execution.record_fft("basis_rfftn_r2c", field)
        if self.repair_dc:
            _repair_native_dc(transformed, field.shape)
        samples = transformed.ravel()[self.native]
        samples = cp.where(self.reflect, samples.conj(), samples)
        out[...] = samples


@dataclass(frozen=True)
class PreparedSampler:
    """A Fourier sampler tied to its input shape, device and creating stream.

    Create one with :func:`prepare_sampler`. Reuse it with independent fields
    and output buffers on the creating stream.
    """

    _capture: _PreparedCapture
    _shape: tuple[int, int, int]
    _count: int
    _device: int
    _stream: Any
    _thread: int
    _recorder: Any
    _dc_requests: Any = None
    _scratch: Any = None

    def capture_into(self, field, *, out) -> None:
        """Enqueue the capture into a caller-owned complex64 vector and return None."""
        cp = self._capture.cp
        for array, name, shape, dtype in (
            (field, "field", self._shape, cp.float32),
            (out, "out", (self._count,), cp.complex64),
        ):
            if not isinstance(array, cp.ndarray):
                raise TypeError(f"{name} must be a CuPy array")
            if array.dtype != dtype:
                raise TypeError(f"{name} must have dtype {np.dtype(dtype)}")
            if array.shape != shape or not array.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous with shape {shape}")
            if array.device.id != self._device:
                raise ValueError(f"{name} must be on the sampler's device {self._device}")
        current_stream = cp.cuda.get_current_stream().ptr
        wrong_thread = current_stream == cp.cuda.Stream.ptds.ptr and get_ident() != self._thread
        if cp.cuda.runtime.getDevice() != self._device or current_stream != self._stream.ptr or wrong_thread:
            raise ValueError("capture_into must run on the sampler's creating device and stream")
        if out.size == 0:
            return
        field_start, out_start = field.data.ptr, out.data.ptr
        if field_start < out_start + out.nbytes and out_start < field_start + field.nbytes:
            raise ValueError("field and out must use separate storage")
        if self._scratch is None:
            self._capture.capture_into(field, out, recorder=self._recorder)
        else:
            self._capture.capture_into(field, self._scratch, recorder=self._recorder)
            self._scratch[self._dc_requests] = self._scratch[self._count]
            out[...] = self._scratch[:self._count]


def _shape(value, name):
    try:
        dims = tuple(value)
        if any(isinstance(n, (bool, np.bool_)) for n in dims):
            raise TypeError
        shape = tuple(index(n) for n in dims)
    except TypeError as exc:
        raise TypeError(f"{name} must contain three integer dimensions in ZYX order") from exc
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError(f"{name} must contain three positive dimensions in ZYX order")
    if max(shape) > np.iinfo(np.int32).max or math.prod(shape) > np.iinfo(np.int64).max:
        raise ValueError(f"{name} exceeds the supported FFT index range")
    return shape


def prepare_sampler(*, shape_zyx, modes_zyx, logical_shape_zyx=None, dc="raw") -> PreparedSampler:
    """Prepare a sampler on the current CuPy device and stream.

    Signed integer modes use the interval [-floor(N/2), ceil(N/2)-1]
    per axis. Order and repeated modes are retained. The real field is
    placed at (logical_shape - shape) // 2 in a zero-filled logical grid.
    The forward transform uses the negative exponential without normalization.
    With dc="raw" the zero bin is the sum of the field. With dc="nrss"
    it is replaced by the incumbent axial-neighbor mean.
    """
    shape = _shape(shape_zyx, "shape_zyx")
    logical = shape if logical_shape_zyx is None else _shape(logical_shape_zyx, "logical_shape_zyx")
    if any(target < occupied for target, occupied in zip(logical, shape)):
        raise ValueError("logical_shape_zyx must contain shape_zyx")
    if not isinstance(dc, str) or dc not in ("raw", "nrss"):
        raise ValueError("dc must be 'raw' or 'nrss'")

    import cupy as cp

    device, stream = cp.cuda.runtime.getDevice(), cp.cuda.get_current_stream()
    if isinstance(modes_zyx, cp.ndarray):
        xp, modes = cp, modes_zyx
        if modes.device.id != device:
            raise ValueError("modes_zyx must be on the current CuPy device")
    else:
        xp, modes = np, np.asarray(modes_zyx)
    if modes.ndim != 2 or modes.shape[1] != 3:
        raise ValueError("modes_zyx must have shape (M, 3) in ZYX order")
    if modes.dtype.kind not in "iu":
        raise TypeError("modes_zyx must contain integer FFT modes")
    lengths = xp.asarray(logical, dtype=xp.int64)
    half = lengths // 2
    if xp.any(modes < -half) or xp.any(modes >= lengths - half):
        raise ValueError("modes_zyx contains a mode outside the signed FFT grid")
    modes = modes.astype(xp.int64, copy=False)
    count = modes.shape[0]
    requested_dc = None
    scratch = None
    recorder = None
    virtual = None
    dc_mask = xp.all(modes == 0, axis=1) if dc == "nrss" else None
    repair_dc = dc_mask is not None and bool(dc_mask.any())
    dc_positions = None
    padded = logical != shape
    coordinates = modes % lengths
    if repair_dc:
        if padded:
            from .backends._cupy_virtual_z import _DC_MODES

            # Use the evaluator's incumbent stencil order, including duplicated
            # native neighbors at Nyquist or on a singleton axis.
            requested_dc = cp.asarray(xp.flatnonzero(dc_mask))
            coordinates = xp.concatenate((coordinates, xp.asarray(_DC_MODES) % lengths))
            dc_positions = cp.arange(count, count + len(_DC_MODES), dtype=cp.int64)
            scratch = cp.empty(count + len(_DC_MODES), dtype=cp.complex64)
        elif min(shape[1:]) < 2:
            raise ValueError("native NRSS DC repair requires Y and X dimensions of at least two")
    native, reflect = _r2c_addresses(xp, logical, *coordinates.T)
    native, reflect = cp.asarray(native), cp.asarray(reflect)
    if padded and count:
        from .backends.cupy_rsoxs import _NullSegmentRecorder

        recorder = _NullSegmentRecorder()
        if logical[1:] == shape[1:]:
            from .backends._cupy_virtual_z import VirtualZEvaluator

            virtual = VirtualZEvaluator(cp, shape[0], logical, native, reflect, dc_positions, "sum")
        else:
            from .backends._cupy_virtual_xyz import VirtualXYZEvaluator

            virtual = VirtualXYZEvaluator(cp, shape, logical, native, reflect, dc_positions)
    capture = _PreparedCapture(cp, native, reflect, virtual, repair_dc)
    return PreparedSampler(capture, shape, count, device, stream, get_ident(), recorder, requested_dc, scratch)
