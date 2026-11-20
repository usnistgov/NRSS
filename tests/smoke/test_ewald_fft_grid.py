"""CPU checks for correct registration and for the existing complex DC replacement."""
from types import SimpleNamespace

import numpy as np
import pytest

from NRSS.backends.cupy_rsoxs import CupyRsoxsBackendRuntime


@pytest.mark.parametrize("shape", [(128, 9, 18), (7, 10, 11), (2, 9, 10), (1, 8, 9)])
def test_detector_coordinates_and_forward_plane(shape):
    runtime = CupyRsoxsBackendRuntime()
    spacing = 1.25
    morph = SimpleNamespace(NumZYX=shape, PhysSize=spacing, _backend_runtime_state={})
    geometry = runtime._detector_geometry(morph, np)
    for count, axis in zip(shape, (geometry.qz, geometry.qy, geometry.qx)):
        expected = np.fft.fftshift(np.fft.fftfreq(count, d=spacing)) * (2*np.pi)
        np.testing.assert_allclose(axis, expected, rtol=3e-7, atol=2e-7)
        assert axis[count//2] == 0
        assert runtime._igor_axis_order(count, np)[count//2] == 0
    iy, ix = shape[1]//2, shape[2]//2
    projection = runtime._detector_projection_geometry(morph, 285., np, geometry)
    assert projection.valid[iy, ix]
    if shape[0] == 1:
        assert projection.safe_z0 is None
    else:
        # Either neighbor representation is legitimate if all weight lands on DC.
        lower = int(projection.safe_z0[iy,ix])
        upper = int(projection.safe_z1[iy,ix])
        frac = float(projection.frac[iy,ix])
        mid = shape[0]//2
        weight_on_dc_plane = (1.-frac)*int(lower==mid) + frac*int(upper==mid)
        assert weight_on_dc_plane == pytest.approx(1., abs=1e-5)


def test_projection_accepts_endpoint_hits_and_masks_unsupported_samples():
    runtime = CupyRsoxsBackendRuntime()
    morphology = SimpleNamespace(NumZYX=(4, 4, 4), PhysSize=1., _backend_runtime_state={})
    geometry = runtime._detector_geometry(morphology, np)
    # At qx = qy = -pi and k = 3*pi/2, the forward root is qz = -pi.
    energy = 0.75 * 1239.84197
    endpoint = runtime._detector_projection_geometry(morphology, energy, np, geometry)
    outside = runtime._detector_projection_geometry(morphology, 0.99 * energy, np, geometry)
    assert endpoint.valid[0, 0]
    assert endpoint.safe_z0[0, 0] == 0
    assert endpoint.frac[0, 0] == pytest.approx(0., abs=1e-6)
    assert not outside.valid[0, 0]

    morphology.NumZYX = (2, 4, 4)
    two_plane_geometry = runtime._detector_geometry(morphology, np)
    forward = runtime._detector_projection_geometry(morphology, energy, np, two_plane_geometry)
    assert forward.valid[2, 2]
    assert forward.safe_z0[2, 2] == forward.safe_z1[2, 2] == 1
    assert forward.frac[2, 2] == 0.

    morphology.NumZYX = (1, 4, 4)
    single_plane_geometry = runtime._detector_geometry(morphology, np)
    evanescent = runtime._detector_projection_geometry(morphology, 0.5 * 1239.84197, np, single_plane_geometry)
    assert not evanescent.valid[0, 0]


@pytest.mark.parametrize("shape", [(1, 7, 8), (6, 7, 8)])
def test_replacement_averages_neighbor_amplitudes_only_at_dc(shape):
    rng = np.random.default_rng(905)
    spectrum = (rng.normal(size=shape) + 1j*rng.normal(size=shape)).astype(np.complex64)
    expected = spectrum.copy()
    indices = [(0,0,1),(0,0,-1),(0,1,0),(0,-1,0)]
    if shape[0] > 1:
        indices += [(1,0,0),(-1,0,0)]
    expected[0,0,0] = np.mean([spectrum[i] for i in indices], dtype=np.complex128)
    CupyRsoxsBackendRuntime()._replace_dc_component(spectrum)
    np.testing.assert_allclose(spectrum[0,0,0], expected[0,0,0], rtol=2e-6, atol=2e-7)
    np.testing.assert_array_equal(spectrum.ravel()[1:], expected.ravel()[1:])
