"""Exercise CuPy q-window selection at the public result boundary."""

from types import SimpleNamespace

import numpy as np
import pytest

from NRSS.backends.contracts import BackendOptionError, normalize_backend_options
from NRSS.morphology import Morphology
from tests.smoke.test_cupy_virtual_z import _specimen
from tests.validation.dev.cupy_rsoxs_qmax.qmax_reference import compare, select_product


@pytest.mark.parametrize("value", (False, np.bool_(True), "0.8", -0.1, np.nan, np.inf, [0.8], 10**400))
def test_q_max_rejects_invalid_values(value):
    """The backend accepts finite nonnegative real scalars."""
    with pytest.raises(BackendOptionError, match="q_max"):
        normalize_backend_options("cupy-rsoxs", {"q_max": value})


@pytest.mark.parametrize(("value", "limit"), ((None, None), (0, 0.0), (np.float32(0.5), 0.5)))
def test_q_max_normalizes_real_values(value, limit):
    assert normalize_backend_options("cupy-rsoxs", {"q_max": value})["q_max"] == limit


@pytest.mark.parametrize("ny", (1, 2, 17))
def test_factored_x_support_matches_full_detector_and_dc(ny):
    """The FFT plan includes every column selected before the cutoff."""
    from NRSS.backends._cupy_virtual_xyz import full_detector_kx
    from NRSS.backends._cupy_virtual_z import dc_indices
    from NRSS.backends.cupy_rsoxs import CupyRsoxsBackendRuntime

    shape = (7, ny, 18)
    energies = (285., 850., 200.)
    owner = SimpleNamespace(NumZYX=shape, PhysSize=2.5, _backend_runtime_state={})
    runtime = CupyRsoxsBackendRuntime()
    detector = runtime._detector_geometry(owner, np)
    stencil = dc_indices(shape)
    needed = np.zeros(shape[2], dtype=bool)
    for energy in energies:
        geometry = runtime._detector_projection_geometry(owner, energy, np, detector)
        needed |= geometry.valid.any(axis=0)
    needed[stencil % shape[2]] = True
    x = runtime._igor_axis_orders(shape, np)[2][needed]
    expected = np.unique(np.minimum(x, (-x) % shape[2]))
    actual = full_detector_kx(runtime, np, energies, detector, shape, stencil)
    np.testing.assert_array_equal(np.unique(actual), expected)


def _morphology(shape, angles, energies=(303., 285., 340.), **options):
    materials = _specimen(shape)
    for material in materials.values():
        material.energies = list(energies)
        for energy in energies:
            material.opt_constants.setdefault(energy, material.opt_constants[285.])
    config = dict(Morphology.config_default, Energies=energies, EAngleRotation=angles, WindowingType=0, RotMask=0)
    return Morphology(
        len(materials), materials=materials, PhysSize=2.5, config=config,
        backend="cupy-rsoxs", backend_options=options, create_cy_object=False,
        input_policy="strict", ownership_policy="borrow",
    )


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
@pytest.mark.usefixtures("nrss_path")
@pytest.mark.parametrize(("shape", "padding", "angles", "residency"), [
    ((1, 13, 16), {}, (0, 0, 0), "device"),
    ((3, 13, 16), {}, (35, 0, 35), "host"),
    ((3, 13, 16), {}, (0, 15, 165), "host"),
    ((3, 13, 16), {"virtual_z_pad_to": 8, "virtual_pad_material": 1}, (0, 15, 165), "host"),
    ((3, 13, 16), {"virtual_z_pad_to": 9, "virtual_z_method": "residues"}, (0, 0, 0), "host"),
    ((3, 13, 16), {"virtual_3d_pad_to": (11, 22, 25), "virtual_pad_material": 1}, (0, 15, 165), "host"),
])
def test_detector_q_window_preserves_bits_at_inclusive_boundary(shape, padding, angles, residency):
    """The cutoff uses global geometry and retains the original detector samples."""
    morph = _morphology(shape, angles, result_residency=residency, result_chunk_size=2, **padding)
    try:
        full = morph.run(stdout=False, stderr=False)
        energy = float(full.energy[0])
        y_index = np.flatnonzero(full.qy.values == 0).item() + 2
        x_index = np.flatnonzero(full.qx.values == 0).item() + 1
        radius = np.hypot(full.qy.values[y_index], full.qx.values[x_index])
        wave_number = 2 * np.pi / (1239.84197 / energy)
        depth = 0. if full.attrs["z_dim"] == 1 else np.sqrt(wave_number**2 - radius**2) - wave_number
        cutoff = np.sqrt(radius * radius + depth**2)
        for limit in (0., np.nextafter(cutoff, 0.), cutoff, np.nextafter(cutoff, np.inf), 1000., None):
            morph._backend_options["q_max"] = limit
            result = morph.run(stdout=False, stderr=False, return_xarray=False)
            expected = select_product(full, limit)
            assert result.to_backend_array().shape == expected.shape
            actual = result.to_xarray()
            compare(expected, actual, limit)
            if limit is not None:
                assert actual.attrs["q_max"] == limit
                assert actual.attrs["q_window_offset_yx"] == (
                    np.flatnonzero(full.qy.values == actual.qy.values[0]).item(),
                    np.flatnonzero(full.qx.values == actual.qx.values[0]).item(),
                )
                assert actual.attrs["q_window_logical_shape_zyx"] == (full.attrs["z_dim"], full.qy.size, full.qx.size)
    finally:
        morph.release_runtime()


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
@pytest.mark.usefixtures("nrss_path")
@pytest.mark.parametrize(("layout", "grid", "energies", "residency", "chunk"), [
    ("integrated", "intersection", (285.,), "host", 1),
    ("i_only", "union", (340., 285., 303.), "host", 2),
    ("i_para_i_perp", "intersection", (340., 285., 303.), "device", 1),
    ("i_a", "union", (340., 285., 303.), "host", 2),
])
def test_radial_q_window_preserves_full_result_bins(layout, grid, energies, residency, chunk):
    """Reduced output selects registered bins and retains auxiliary q coordinates."""
    morph = _morphology(
        (3, 13, 16), (0, 15, 165), energies, virtual_3d_pad_to=(9, 20, 23),
        result_layout=layout, result_q_grid_mode=grid, result_residency=residency, result_chunk_size=chunk,
    )
    try:
        full = morph.run(stdout=False, stderr=False)
        morph._backend_options["q_max"] = 0.7
        morph._set_private_backend_timing_segments(("energy",))
        actual = morph.run(stdout=False, stderr=False)
        compare(select_product(full, 0.7), actual, 0.7)
        assert actual.q.size < full.q.size
        work = morph.backend_timings["execution"]["q_window"]
        assert 0 < work["polar_source_bins"] < work["full_polar_bins"] == full.q.size
    finally:
        morph.release_runtime()


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
@pytest.mark.usefixtures("nrss_path")
def test_radial_q_window_preserves_per_slice_support():
    """The positional fallback keeps each energy's physical q mask."""
    morph = _morphology(
        (3, 13, 16), (0, 0, 0), (12., 285.),
        result_layout="i_a", result_chunk_size=2,
    )
    try:
        full = morph.run(stdout=False, stderr=False)
        assert full.attrs["radial_coordinate_mode"] == "per_slice_q_abs"
        morph._backend_options["q_max"] = 0.7
        limited = morph.run(stdout=False, stderr=False)
        compare(select_product(full, 0.7), limited, 0.7)
        assert limited.q.size > 1 and limited.I.isel(energy=0, q=slice(1, None)).isnull().all()
    finally:
        morph.release_runtime()


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
@pytest.mark.usefixtures("nrss_path")
def test_detector_q_window_preserves_cutoff_after_basis_allocation_failure(monkeypatch):
    """The automatic native fallback returns the same window on its own detector."""
    import cupy as cp

    from NRSS.backends import _cupy_energy_basis

    def fail_basis(*args, **kwargs):
        raise cp.cuda.memory.OutOfMemoryError(1, 2, 3)

    monkeypatch.setattr(_cupy_energy_basis, "build_energy_basis", fail_basis)
    morph = _morphology((3, 13, 16), (0, 0, 0))
    try:
        morph._set_private_backend_timing_segments(("energy",))
        full = morph.run(stdout=False, stderr=False)
        execution = morph.backend_timings["execution"]
        assert execution["routes"] == [morph.backend_options["execution_path"]]
        assert execution["fallback_reasons"] == ["energy basis allocation exceeded available GPU memory"]
        morph._backend_options["q_max"] = 0.7
        limited = morph.run(stdout=False, stderr=False)
        assert morph.backend_timings["execution"]["routes"] == execution["routes"]
        compare(select_product(full, 0.7), limited, 0.7)
    finally:
        morph.release_runtime()


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
@pytest.mark.usefixtures("nrss_path")
@pytest.mark.parametrize("layout", ("detector", "i_a"))
def test_q_window_bounds_xyz_projection_and_preserves_factored_fft(layout):
    """The limited endpoint table retains both the bits and parent FFT topology."""
    from NRSS.backends._cupy_virtual_xyz import XYZ_FACTORED_XY_KEY

    morph = _morphology(
        (3, 13, 16), (0, 15, 165), virtual_3d_pad_to=(11, 29, 38),
        virtual_pad_material=1, result_layout=layout, result_chunk_size=2,
    )
    try:
        morph._backend_runtime_state[XYZ_FACTORED_XY_KEY] = True
        morph._set_private_backend_timing_segments(("energy",))
        full = morph.run(stdout=False, stderr=False)
        before = morph.backend_timings["execution"]
        morph._backend_options["q_max"] = 0.5
        limited = morph.run(stdout=False, stderr=False)
        after = morph.backend_timings["execution"]
        compare(select_product(full, 0.5), limited, 0.5)
        assert after["ffts"] == before["ffts"]
        assert after["basis_cache"]["xy_factored_kx"] == before["basis_cache"]["xy_factored_kx"]
        assert after["basis_cache"]["union_endpoints"] < before["basis_cache"]["union_endpoints"]
        counts = after["q_window"]
        assert counts["projection_cells_per_energy"] < counts["full_detector_cells_per_energy"]
    finally:
        morph.release_runtime()
