"""Check opt-in axial padding against literal vacuum and independent FFTs."""

import math

import numpy as np
import pytest

from NRSS import SFieldMode
from NRSS.backends.contracts import BackendOptionError, normalize_backend_options
from NRSS.morphology import Material, Morphology
from tests.validation.lib.virtual_z_padding import explicit_vacuum_materials, xy_hann_reference


@pytest.mark.parametrize("options", [
    {"virtual_z_pad_to": True},
    {"virtual_z_pad_to": 7.5},
    {"virtual_z_pad_to": 0},
    {"virtual_z_method": "sum"},
    {"virtual_z_pad_to": 20, "virtual_z_method": "dense"},
])
def test_virtual_z_options_validate(options):
    with pytest.raises(BackendOptionError, match="virtual_z"):
        normalize_backend_options("cupy-rsoxs", options)


@pytest.mark.parametrize(("options", "message"), [
    ({"virtual_z_pad_to": 2}, "at least the occupied"),
    ({"virtual_z_pad_to": 7, "virtual_z_method": "residues"}, "multiple of the occupied"),
    ({"virtual_z_pad_to": 6, "execution_path": "direct_polarization"}, "compact tensor"),
    ({"virtual_z_pad_to": 6, "z_collapse_mode": "mean"}, "full-depth"),
])
def test_virtual_z_rejects_ineligible_geometry(options, message):
    morph = Morphology(
        2, materials=_specimen((3, 10, 11)), PhysSize=2.5, backend="cupy-rsoxs",
        backend_options=options, create_cy_object=False,
    )
    with pytest.raises(ValueError, match=message):
        morph._backend_runtime._virtual_z_shape(morph)


def _specimen(shape):
    rng = np.random.default_rng(271)
    energies = [285., 303., 340.]
    fraction, s, theta, psi = rng.random((4, *shape), dtype=np.float32)
    return {
        1: Material(
            1, Vfrac=fraction, S=s, theta=theta * 2, psi=psi * 6, energies=energies,
            opt_constants={e: [0.003, 0.002, -0.002, 0.001 * e / energies[0]] for e in energies},
        ),
        2: Material(
            2, Vfrac=1 - fraction, S=SFieldMode.ISOTROPIC, energies=energies,
            opt_constants={e: [0.] * 4 for e in energies},
        ),
    }


def _morphology(materials, window, **options):
    config = dict(
        Morphology.config_default, Energies=materials[1].energies, EAngleRotation=[5., 30., 65.],
        RotMask=0,
    )
    if window is None:
        config.pop("WindowingType")
    else:
        config["WindowingType"] = window
    return Morphology(
        len(materials), materials=materials, PhysSize=2.5, config=config, backend="cupy-rsoxs",
        input_policy="strict", ownership_policy="borrow", backend_options=options,
    )


def _run(morph):
    try:
        morph._set_private_backend_timing_segments(("energy",))
        data = morph.run(stdout=False, stderr=False, return_xarray=True).copy(deep=True)
        return data, morph.backend_timings["execution"]
    finally:
        morph.release_runtime()


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
@pytest.mark.usefixtures("nrss_path")
@pytest.mark.parametrize(("shape", "target", "method", "window", "layout"), [
    ((5, 17, 18), 11, "sum", 0, "detector"),
    ((5, 16, 15), 20, "residues", 1, "detector"),
    ((1, 8, 9), 4, "sum", 1, "integrated"),
    ((3, 8, 9), 6, "residues", 0, "i_a"),
])
def test_virtual_z_matches_literal_vacuum(shape, target, method, window, layout):
    materials = _specimen(shape)
    reference = _morphology(explicit_vacuum_materials(materials, target), 0, result_layout=layout)
    if window:
        xy_hann_reference(reference)
    expected, _ = _run(reference)
    morph = _morphology(materials, window, result_layout=layout, virtual_z_pad_to=target, virtual_z_method=method)
    actual, report = _run(morph)
    assert morph.NumZYX == shape
    assert actual.attrs["occupied_z"] == shape[0]
    assert actual.attrs["transform_z"] == actual.attrs["z_dim"] == target
    assert actual.attrs["z_offset"] == (target - shape[0]) // 2
    assert actual.attrs["window_xy"] == ("hann" if window else "none")
    assert actual.attrs["window_z"] == "none"
    if layout != "detector":
        assert actual.attrs["radial_grid_convention"] == "sampled_q_v1"
    assert report["routes"] == [f"virtual_z_{method}"]
    assert report["basis_builds"] == 1
    for coord in expected.coords:
        np.testing.assert_array_equal(expected.coords[coord], actual.coords[coord])
    fields = zip(expected.data_vars.values(), actual.data_vars.values()) if hasattr(expected, "data_vars") else [
        (expected, actual),
    ]
    for reference, result in fields:
        valid = np.isfinite(reference.values)
        np.testing.assert_array_equal(valid, np.isfinite(result.values))
        assert valid.any()
        a, b = reference.values[valid].astype(np.float64), result.values[valid].astype(np.float64)
        scale = max(np.linalg.norm(a), 1.e-20)
        assert np.linalg.norm(b - a) / scale < 2.e-5
        assert np.max(np.abs(b - a)) < 2.e-5 * max(np.max(np.abs(a)), 1.e-20)


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
@pytest.mark.usefixtures("nrss_path")
def test_virtual_z_noop_and_same_process_shape_switch():
    shape = (3, 10, 11)
    morph = _morphology(_specimen(shape), 0)
    try:
        native = morph.run(stdout=False, stderr=False).copy(deep=True)
        morph._backend_options["virtual_z_pad_to"] = 6
        padded = morph.run(stdout=False, stderr=False)
        assert padded.attrs["transform_z"] == 6
        assert np.nanmax(np.abs(native.values - padded.values)) > 0
        morph._backend_options["virtual_z_pad_to"] = shape[0]
        identity = morph.run(stdout=False, stderr=False)
        np.testing.assert_array_equal(native.values, identity.values)
        assert native.attrs == identity.attrs
    finally:
        morph.release_runtime()


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
@pytest.mark.usefixtures("nrss_path")
def test_virtual_z_memory_error_cannot_return_native_depth(monkeypatch):
    import cupy as cp

    from NRSS.backends import _cupy_energy_basis

    def fail(*args, **kwargs):
        raise cp.cuda.memory.OutOfMemoryError(1024, 0)

    morph = _morphology(_specimen((3, 10, 11)), 0, virtual_z_pad_to=6)
    monkeypatch.setattr(_cupy_energy_basis, "build_energy_basis", fail)
    try:
        with pytest.raises(RuntimeError, match="padding to 6 planes exceeded"):
            morph.run(stdout=False, stderr=False)
    finally:
        morph.release_runtime()


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
@pytest.mark.usefixtures("nrss_path")
@pytest.mark.parametrize(("shape", "target", "method"), [((5, 7, 8), 11, "sum"), ((3, 4, 5), 12, "residues")])
def test_virtual_z_coefficients_match_dense_numpy(shape, target, method):
    import cupy as cp

    from NRSS.backends._cupy_energy_basis import _native_indices
    from NRSS.backends._cupy_virtual_z import VirtualZEvaluator, dc_indices
    from NRSS.backends.cupy_rsoxs import CupyRsoxsBackendRuntime, _NullSegmentRecorder

    occupied, ny, nx = shape
    logical = (target, ny, nx)
    values = np.random.default_rng(17).standard_normal(shape, dtype=np.float32)
    literal = np.zeros(logical, dtype=np.float32)
    offset = (target - occupied) // 2
    literal[offset:offset + occupied] = values
    full = np.fft.fftn(literal.astype(np.float64))
    # For each Igor index, the native bin has the opposite signed coordinate.
    orders = [(length // 2 - np.arange(length)) % length for length in logical]
    expected = full[np.ix_(*orders)].ravel()
    stencil = dc_indices(logical)
    expected[stencil[0]] = expected[stencil[1:]].real.mean()
    indices = cp.arange(math.prod(logical), dtype=cp.int64)
    native, reflect = _native_indices(CupyRsoxsBackendRuntime(), cp, logical, indices, True)
    evaluator = VirtualZEvaluator(cp, occupied, logical, native, reflect, cp.asarray(stencil), method)
    output = cp.empty(indices.shape, dtype=cp.complex64)
    evaluator.capture(cp, cp.asarray(values), output, _NullSegmentRecorder())
    np.testing.assert_allclose(cp.asnumpy(output), expected, rtol=2.e-6, atol=2.e-6 * np.max(abs(expected)))


def _scalar_materials(fraction):
    energies = [285.]
    return {
        key: Material(
            key, Vfrac=values, S=SFieldMode.ISOTROPIC, energies=energies,
            opt_constants={285.: optics},
        )
        for key, values, optics in (
            (1, fraction, [.003, .002] * 2), (2, 1 - fraction, [0.] * 4),
        )
    }


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
@pytest.mark.usefixtures("nrss_path")
@pytest.mark.parametrize("seam", (False, True), ids=("edge_matched_scalar", "discontinuous_anisotropic"))
def test_film_window_selection_is_manual(seam, monkeypatch):
    from NRSS.backends import _cupy_energy_basis

    shape = (5, 16, 17)
    if seam:
        materials = _specimen(shape)
        fraction = materials[1].Vfrac
        fraction[..., :8] = .25
        fraction[..., 8:] = .75
        materials[2].Vfrac = 1 - fraction
        # The scalar case uses the fused projector. Exercise the other
        # projector with the same weighted complex anisotropic basis.
        monkeypatch.setattr(_cupy_energy_basis, "project_energy_fused", _cupy_energy_basis.project_energy)
    else:
        materials = _scalar_materials(np.full(shape, .5, dtype=np.float32))
    omitted, _ = _run(_morphology(materials, None, virtual_z_pad_to=13))
    no_window, _ = _run(_morphology(materials, 0, virtual_z_pad_to=13))
    hann, _ = _run(_morphology(materials, 1, virtual_z_pad_to=13))
    reference = _morphology(explicit_vacuum_materials(materials, 13), 0)
    xy_hann_reference(reference)
    expected, _ = _run(reference)
    np.testing.assert_array_equal(omitted.values, no_window.values)
    assert omitted.attrs["window_xy"] == no_window.attrs["window_xy"] == "none"
    assert hann.attrs["window_xy"] == "hann"
    assert hann.attrs["window_z"] == "none"
    assert not np.allclose(hann.values, no_window.values, rtol=1.e-4, atol=0, equal_nan=True)
    np.testing.assert_array_equal(np.isfinite(hann.values), np.isfinite(expected.values))
    np.testing.assert_allclose(
        hann.values, expected.values, rtol=2.e-5, atol=2.e-7 * np.nanmax(expected.values),
    )


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
@pytest.mark.usefixtures("nrss_path")
@pytest.mark.parametrize(("method", "target"), (("sum", 257), ("residues", 256)))
@pytest.mark.parametrize("window", (0, 1))
def test_film_window_face_scatterer_phases(method, target, window, monkeypatch):
    import cupy as cp

    from NRSS.backends import _cupy_virtual_z
    from tests.validation.dev.cupy_rsoxs_virtual_zpad import check_film_window as oracle

    original_union = _cupy_virtual_z.sparse_endpoint_union
    original_capture = _cupy_virtual_z.VirtualZEvaluator.capture
    nz, ny, nx = oracle.NZ, oracle.NY, oracle.NX
    y, x = (ny // 2 - oracle.KY) % ny, (nx // 2 - oracle.KX) % nx
    bins = (target // 2 - cp.arange(target)) % target
    requested = (bins * ny + y) * nx + x
    half_x = nx // 2 + 1
    captured = []

    def include_probe_bins(*args):
        return cp.union1d(original_union(*args), requested)

    def capture(evaluator, xp, field, out, recorder):
        original_capture(evaluator, xp, field, out, recorder)
        columns = ny * half_x
        selected = (evaluator.native % columns == oracle.KY * half_x + oracle.KX) & ~evaluator.reflect
        modes = cp.asnumpy(evaluator.native[selected] // columns)
        ordered = np.empty(target, dtype=np.complex64)
        assert modes.size == target
        ordered[modes] = cp.asnumpy(out[selected])
        captured.append(ordered)

    monkeypatch.setattr(_cupy_virtual_z, "sparse_endpoint_union", include_probe_bins)
    monkeypatch.setattr(_cupy_virtual_z.VirtualZEvaluator, "capture", capture)
    lateral = "hann" if window else "none"
    for planes in ((0,), (nz // 2,), (nz - 1,), (0, nz - 1)):
        fraction = np.zeros((nz, ny, nx), dtype=np.float32)
        fraction[list(planes), oracle.Y, oracle.X] = 1.
        captured.clear()
        result, _ = _run(_morphology(
            _scalar_materials(fraction), window, virtual_z_pad_to=target, virtual_z_method=method,
        ))
        offset = (target - nz) // 2
        assert result.attrs["z_offset"] == offset
        expected = oracle.expected_coefficients(target, offset, planes, lateral)
        np.testing.assert_allclose(captured[0], expected, rtol=2.e-6, atol=3.e-7)
        if len(planes) == 1:
            np.testing.assert_allclose(abs(captured[0]) ** 2, 1 / 16 if window else 1, rtol=2.e-6)
