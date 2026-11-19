"""Check automatic Fourier reuse through the public CuPy morphology API."""

import numpy as np
import pytest

from NRSS import SFieldMode
from NRSS.morphology import Material, Morphology


@pytest.mark.parametrize(
    ("options", "mode"),
    [
        ({}, "auto"),
        ({"execution_path": "tensor_coeff"}, "auto"),
        ({"execution_path": "direct_polarization"}, None),
        ({"energy_basis_mode": None}, None),
    ],
)
def test_energy_basis_defaults_respect_explicit_backend_options(options, mode):
    """The automatic policy is enabled for default and explicit tensor workflows."""
    morphology = Morphology(1, backend="cupy-rsoxs", backend_options=options, create_cy_object=False)
    assert morphology._energy_basis_mode == mode
    assert ("energy_basis_mode" in morphology.backend_options) == ("energy_basis_mode" in options)


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
@pytest.mark.parametrize(
    ("energy_count", "shape"),
    [(1, (1, 24, 28)), (20, (32, 512, 512))],
    ids=["single_energy_single_slice", "energy_scan"],
)
@pytest.mark.parametrize("allocation_failure", [False, True], ids=["basis", "memory_fallback"])
def test_energy_basis_automatic_dispatch(nrss_path, energy_count, shape, allocation_failure, monkeypatch):
    """An eligible scan builds its basis once or recovers with the dense route."""
    cp = pytest.importorskip("cupy")
    from NRSS.backends import _cupy_energy_basis

    if allocation_failure:
        def allocation_error(*args, **kwargs):
            raise cp.cuda.memory.OutOfMemoryError(1024, 0)

        monkeypatch.setattr(_cupy_energy_basis, "build_energy_basis", allocation_error)

    energies = [280 + i / 2 for i in range(energy_count)]
    phase = np.zeros(shape, dtype=np.float32)
    inclusion = tuple(slice(n // 4, max(1, n // 2)) for n in shape)
    phase[inclusion] = np.float32(1)
    materials = {
        1: Material(
            1, Vfrac=phase, S=SFieldMode.ISOTROPIC, energies=energies,
            opt_constants={e: [0.001 + (e - 280) * 1e-5, 0.0001] * 2 for e in energies},
        ),
        2: Material(
            2, Vfrac=1 - phase, S=SFieldMode.ISOTROPIC, energies=energies,
            opt_constants={e: [0.0] * 4 for e in energies},
        ),
    }
    morphology = Morphology(
        2, materials=materials, PhysSize=2.5, backend=nrss_path.backend,
        input_policy="strict", ownership_policy="borrow",
    )
    try:
        morphology._set_private_backend_timing_segments(("energy",))
        result = morphology.run(stdout=False, stderr=False, return_xarray=False)
        assert result.to_backend_array().shape == (energy_count, *shape[1:])
        assert np.any(result.to_backend_array() > 0)
        assert morphology.materials[1].Vfrac is phase
        report = morphology.backend_timings["execution"]
        if allocation_failure:
            assert report["routes"] == ["direct_polarization"]
            assert report["fft_count"] == 3 * energy_count
            assert report["basis_builds"] == 0
            assert report["fallback_reasons"] == ["energy basis allocation exceeded available GPU memory"]
        else:
            assert report["routes"] == ["compact_r2c"]
            assert report["fft_count"] == 2
            assert report["basis_builds"] == 1
            assert report["fallback_reasons"] == []
    finally:
        morphology.release_runtime()


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
def test_windowed_energy_basis_memory_fallback(nrss_path, monkeypatch):
    """Rebuild the dense Hann window if allocating the compact basis fails."""
    cp = pytest.importorskip("cupy")
    from NRSS.backends import _cupy_energy_basis

    shape = (3, 16, 18)
    phase = np.zeros(shape, dtype=np.float32)
    phase[1, 4:10, 6:13] = np.float32(1)

    def run(options):
        materials = {
            index: Material(index, Vfrac=vfrac, S=SFieldMode.ISOTROPIC,
                            energies=[285.0], opt_constants={285.0: optics})
            for index, vfrac, optics in (
                (1, phase, [0.001, 0.0001] * 2),
                (2, 1 - phase, [0.0] * 4),
            )
        }
        morphology = Morphology(
            2, materials=materials, PhysSize=2.5, backend=nrss_path.backend, backend_options=options,
        )
        morphology.WindowingType = 1
        try:
            morphology._set_private_backend_timing_segments(("energy",))
            data = morphology.run(stdout=False, stderr=False, return_xarray=False).to_backend_array().copy()
            assert morphology.backend_timings["execution"]["routes"] == ["direct_polarization"]
            return data
        finally:
            morphology.release_runtime()

    reference = run({"execution_path": "direct_polarization"})

    def allocation_error(runtime, morphology, cp, materials, energies, window, mode, execution):
        zy, x = window
        assert zy.shape == (shape[0], shape[1], 1) and x.shape == (1, 1, shape[2])
        raise cp.cuda.memory.OutOfMemoryError(1024, 0)

    monkeypatch.setattr(_cupy_energy_basis, "build_energy_basis", allocation_error)
    assert np.any(reference > 0)
    np.testing.assert_array_equal(run({}), reference)
