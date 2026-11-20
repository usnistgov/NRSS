"""Public tests of the established forward-pixel DC replacement."""
import numpy as np
import pytest

from tests.validation.lib.ewald_registration import detector_run, intensity


@pytest.mark.path_subset("cupy_tensor_coeff", "cupy_direct_polarization")
@pytest.mark.physics_validation
@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(1,9,10), (2,9,10), (8,9,10)])
def test_forward_pixel_contains_replaced_dc(nrss_path, shape):
    """A pure lateral cosine gives a known replacement amplitude at the direct beam."""
    nz, ny, nx = shape
    fraction = (1+np.cos(2*np.pi*np.arange(nx)/nx))/2
    field = np.broadcast_to(fraction,shape)
    energy = 285.
    panel = detector_run(field,[energy],nrss_path)
    image = panel.sel(energy=energy).values
    # Only the two x neighbors contribute, each V/4; stencil has 4 or 6 entries.
    replaced_dc = (field.size/2)/(4 if nz==1 else 6)
    ratio = image[ny//2,nx//2]/intensity(energy,replaced_dc)
    assert ratio == pytest.approx(1., rel=1e-4)

