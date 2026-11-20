"""Check the radial labels against the sampled detector locations."""
from types import SimpleNamespace

import numpy as np
import pytest

from NRSS.backends.cupy_rsoxs import CupyRsoxsBackendRuntime


@pytest.mark.parametrize("shape", [(32, 32), (29, 40)])
def test_cupy_radial_labels_follow_sampled_q(shape):
    runtime = CupyRsoxsBackendRuntime()
    morphology = SimpleNamespace(NumZYX=(1, *shape), PhysSize=2., backend_options={})
    qy, qx = (2. * np.pi * np.fft.fftshift(np.fft.fftfreq(n, d=2.)) for n in shape)
    detector = 10. + qy[:, None] + qx[None, :]
    plan = runtime._reduced_result_plan(
        morphology=morphology, projection_shape=shape, energies=(285.,), cp=np, result_layout="integrated",
    )
    observed = runtime._integrate_projection_chunk(detector[None, ...], energy_start=0, reduction_plan=plan, cp=np)
    assert plan.attrs["radial_grid_convention"] == "sampled_q_v1"
    interior = plan.q < min(np.max(abs(qy)), np.max(abs(qx))) / 2
    q = plan.q[interior]
    assert len(q) > 5
    for angle in (0, 45, 90, 180):
        theta = np.deg2rad(angle)
        expected = 10. + q * (np.sin(theta) + np.cos(theta))
        np.testing.assert_allclose(observed[0, angle, interior], expected, rtol=0., atol=2e-13)
