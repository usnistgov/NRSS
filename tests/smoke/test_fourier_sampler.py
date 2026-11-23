"""Check the external CuPy Fourier API and its CPU import boundary."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from NRSS.fourier import prepare_sampler
from tests.validation.dev.cupy_rsoxs_sampler.check_sampling import check, direct


def test_fourier_sampler_import_is_cpu_safe():
    source = Path(__file__).resolve().parents[2] / "src"
    subprocess.run(
        [sys.executable, "-c", "import sys; from NRSS.fourier import prepare_sampler; "
         "assert 'cupy' not in sys.modules and 'nrss_sals' not in sys.modules"],
        env={**os.environ, "PYTHONPATH": str(source)},
        check=True,
    )


@pytest.mark.gpu
def test_fourier_sampler_matches_literal_and_direct_dft():
    cp = pytest.importorskip("cupy")
    check(cp, prepare_sampler)


@pytest.mark.gpu
@pytest.mark.parametrize("target", ((3, 4, 5), (6, 4, 5), (6, 7, 8)))
def test_fourier_sampler_dc_and_off_center_modes(target):
    cp = pytest.importorskip("cupy")
    field = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    array = cp.asarray(field)
    out = cp.empty(1, dtype=cp.complex64)
    dc_only = prepare_sampler(shape_zyx=field.shape, logical_shape_zyx=target, modes_zyx=cp.zeros((1, 3), np.int64))
    assert dc_only.capture_into(array, out=out) is None
    np.testing.assert_array_equal(cp.asnumpy(out), [field.sum()])

    modes = np.array([[1, -1, -2]])
    off_center = prepare_sampler(shape_zyx=field.shape, logical_shape_zyx=target, modes_zyx=modes, dc="nrss")
    off_center.capture_into(array, out=out)
    np.testing.assert_allclose(cp.asnumpy(out), direct(field, target, modes), rtol=0, atol=5e-6 * field.sum())

    stencil = [(0, 0, 1), (0, 1, 0), (0, 0, -1), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
    if target == field.shape:
        spectrum = cp.fft.rfftn(array)
        neighbors = [spectrum[0, 0, 1], spectrum[0, 1, 0], spectrum[0, 0, 1].conj(),
                     spectrum[0, -1, 0], spectrum[1, 0, 0], spectrum[-1, 0, 0]]
        repaired = sum(neighbors).real / np.float32(len(neighbors))
    else:
        raw_stencil = prepare_sampler(shape_zyx=field.shape, logical_shape_zyx=target, modes_zyx=stencil)
        neighbors = cp.empty(len(stencil), dtype=cp.complex64)
        raw_stencil.capture_into(array, out=neighbors)
        repaired = cp.mean(neighbors.real)
    repeated = prepare_sampler(shape_zyx=field.shape, logical_shape_zyx=target,
                               modes_zyx=[(0, 0, 0), *modes.tolist(), (0, 0, 0)], dc="nrss")
    result = cp.empty(3, dtype=cp.complex64)
    repeated.capture_into(array, out=result)
    expected = cp.asarray([0, 0, 0], dtype=cp.complex64)
    expected[::2] = repaired
    expected[1] = out[0]
    cp.testing.assert_array_equal(result, expected)


@pytest.mark.gpu
def test_fourier_sampler_validates_arrays_modes_and_stream():
    cp = pytest.importorskip("cupy")
    shape = (3, 4, 6)
    with pytest.raises(ValueError, match="signed FFT grid"):
        prepare_sampler(shape_zyx=shape, modes_zyx=[(0, 0, 3)])
    with pytest.raises(TypeError, match="integer"):
        prepare_sampler(shape_zyx=shape, modes_zyx=[(0., 0., 0.)])
    with pytest.raises(ValueError, match="contain"):
        prepare_sampler(shape_zyx=shape, logical_shape_zyx=(2, 4, 6), modes_zyx=[(0, 0, 0)])
    with pytest.raises(ValueError, match="shape"):
        prepare_sampler(shape_zyx=shape, modes_zyx=[0, 0, 0])

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        plan = prepare_sampler(shape_zyx=shape, modes_zyx=[(0, 0, 0), (0, 0, -3)])
        field = cp.ones(shape, dtype=cp.float32)
        out = cp.empty(2, dtype=cp.complex64)
        plan.capture_into(field, out=out)
        with pytest.raises(TypeError, match="field"):
            plan.capture_into(field.astype(cp.float64), out=out)
        with pytest.raises(TypeError, match="out"):
            plan.capture_into(field, out=cp.empty(2, dtype=cp.complex128))
        with pytest.raises(ValueError, match="out.*C-contiguous"):
            plan.capture_into(field, out=cp.empty(4, dtype=cp.complex64)[::2])
        with pytest.raises(ValueError, match="shape"):
            plan.capture_into(field, out=out[:1])
        with pytest.raises(ValueError, match="separate storage"):
            plan.capture_into(field, out=field.view(cp.complex64).ravel()[:2])
        with pytest.raises(ValueError, match="field.*C-contiguous"):
            plan.capture_into(cp.empty((3, 4, 12), dtype=cp.float32)[..., ::2], out=out)
    with pytest.raises(ValueError, match="creating device and stream"):
        plan.capture_into(field, out=out)
    stream.synchronize()
    cp.testing.assert_array_equal(out, [72, 0])
