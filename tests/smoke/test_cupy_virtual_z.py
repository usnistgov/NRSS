"""Check opt-in axial padding against physical boxes and independent FFTs."""

import math

import numpy as np
import pytest

from NRSS import SFieldMode
from NRSS.backends.contracts import BackendOptionError, normalize_backend_options
from NRSS.morphology import Material, Morphology
from tests.validation.lib.virtual_z_padding import (
    explicit_padded_materials,
    explicit_vacuum_materials,
    xy_hann_reference,
)


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


@pytest.mark.parametrize("value", (True, np.bool_(False), 1., "1", [1], -1, 0))
def test_virtual_pad_material_rejects_non_ids(value):
    with pytest.raises(BackendOptionError, match="integer material ID"):
        normalize_backend_options("cupy-rsoxs", {"virtual_z_pad_to": 8, "virtual_pad_material": value})


def test_virtual_pad_material_normalization_and_target():
    none = normalize_backend_options("cupy-rsoxs", {"virtual_pad_material": None})
    assert none["virtual_pad_material"] is None
    options = normalize_backend_options("cupy-rsoxs", {"virtual_z_pad_to": 8, "virtual_pad_material": np.int64(1)})
    assert type(options["virtual_pad_material"]) is int
    with pytest.raises(BackendOptionError, match="requires virtual_z_pad_to"):
        normalize_backend_options("cupy-rsoxs", {"virtual_pad_material": 1})


@pytest.mark.parametrize("target", (3, 6))
def test_virtual_pad_material_validates_registration_and_optics_when_bound(target):
    materials = _specimen((3, 10, 11))
    kwargs = dict(PhysSize=2.5, backend="cupy-rsoxs", create_cy_object=False)
    with pytest.raises(ValueError, match="not a registered material ID"):
        Morphology(2, materials=materials, backend_options={
            "virtual_z_pad_to": target, "virtual_pad_material": 3,
        }, **kwargs)
    del materials[1].opt_constants[285.]
    with pytest.raises(ValueError, match="four finite optical constants at 285"):
        Morphology(2, materials=materials, backend_options={
            "virtual_z_pad_to": target, "virtual_pad_material": 1,
        }, **kwargs)


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
        morph._backend_runtime._virtual_padding_shape(morph)


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


@pytest.mark.parametrize(
    "fill_material", [None, 1, 2, 3], ids=["vacuum", "oriented", "matrix", "unoccupied"],
)
def test_literal_box_fills_only_the_exterior(fill_material):
    shape = (5, 7, 8)
    target = (14, 16, 19)
    material_fields = _specimen(shape)
    particle, matrix = material_fields[1], material_fields[2]
    hole = (2, 3, 4)
    particle.Vfrac[hole] = 0
    matrix.Vfrac[hole] = 1
    matrix.opt_constants = {e: [0.01, 0.002] * 2 for e in matrix.energies}
    material_fields[3] = Material(
        3, Vfrac=np.zeros(shape, dtype=np.float32), S=SFieldMode.ISOTROPIC,
        energies=matrix.energies, opt_constants={e: [0.02, 0.005] * 2 for e in matrix.energies},
    )
    fields = ("Vfrac", "S", "theta", "psi")
    original = {
        (key, field): np.copy(value)
        for key, material in material_fields.items()
        for field in fields
        if isinstance(value := getattr(material, field), np.ndarray)
    }

    padded = explicit_padded_materials(material_fields, target, fill_material=fill_material)

    # The extra voxel on each odd difference goes on the high-index side.
    interior = (slice(4, 9), slice(4, 11), slice(5, 13))
    exterior = np.ones(target, dtype=bool)
    exterior[interior] = False
    expected_fill = 4 if fill_material is None else fill_material
    assert set(padded) == ({1, 2, 3, 4} if fill_material is None else {1, 2, 3})
    for key, material in padded.items():
        assert material.Vfrac.shape == target
        np.testing.assert_array_equal(material.Vfrac[exterior], int(key == expected_fill))
        for field in fields[1:]:
            value = getattr(material, field)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value[exterior], 0.)
        if key in material_fields:
            assert material.opt_constants == material_fields[key].opt_constants
            if material_fields[key].S is SFieldMode.ISOTROPIC:
                assert material.S is SFieldMode.ISOTROPIC
    for (key, field), before in original.items():
        np.testing.assert_array_equal(getattr(padded[key], field)[interior], before)
        np.testing.assert_array_equal(getattr(material_fields[key], field), before)
    if fill_material is None:
        assert padded[4].S is SFieldMode.ISOTROPIC
        assert not np.any(list(padded[4].opt_constants.values()))
        np.testing.assert_array_equal(padded[4].Vfrac[interior], 0.)
    np.testing.assert_allclose(sum(material.Vfrac for material in padded.values()), 1.)


def test_literal_box_identity_and_invalid_reference():
    shape = (5, 7, 8)
    materials = _specimen(shape)
    assert explicit_padded_materials(materials, shape, fill_material=1) is materials
    with pytest.raises(ValueError, match="not registered"):
        explicit_padded_materials(materials, shape, fill_material=3)
    with pytest.raises(ValueError, match="contain the occupied crop"):
        explicit_padded_materials(materials, (12, 6, 10))


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
    _assert_full_parity(expected, actual)


def _assert_full_parity(expected, actual):
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


def _absorbing_specimen(shape):
    materials = _specimen(shape)
    particle = materials[1]
    matrix = materials[2]
    energies = [285., 340.]
    particle.energies = matrix.energies = energies
    particle.opt_constants = {285.: [.09, .035, -.04, .012], 340.: [-.08, .01, .11, .046]}
    matrix.opt_constants = {285.: [.031, .022] * 2, 340.: [.018, .005] * 2}
    hole = (1, 2, 3)
    particle.Vfrac[hole] = 0
    matrix.Vfrac[hole] = 1
    materials[3] = Material(
        3, Vfrac=np.zeros(shape, dtype=np.float32), S=SFieldMode.ISOTROPIC,
        energies=energies, opt_constants={energy: [0.] * 4 for energy in energies},
    )
    return materials


def _literal_run(materials, shape, window, fill):
    morph = _morphology(explicit_padded_materials(materials, shape, fill_material=fill), 0)
    if window:
        xy_hann_reference(morph)
    return _run(morph)[0]


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
@pytest.mark.usefixtures("nrss_path")
@pytest.mark.parametrize(("method", "target"), (("sum", 7), ("residues", 12)))
@pytest.mark.parametrize("window", (0, 1))
@pytest.mark.parametrize("fill", (None, 1, 2, 3))
def test_virtual_z_fill_matches_literal_panel(method, target, window, fill):
    shape = (3, 15, 16)
    materials = _absorbing_specimen(shape)
    expected = _literal_run(materials, (target, *shape[1:]), window, fill)
    morph = _morphology(
        materials, window, virtual_z_pad_to=target, virtual_z_method=method, virtual_pad_material=fill,
    )
    actual, report = _run(morph)
    assert morph.NumZYX == shape
    assert report["routes"] == [f"virtual_z_{method}"]
    assert report["basis_builds"] == 1
    assert actual.attrs["transform_z"] == actual.attrs["z_dim"] == target
    assert actual.attrs["z_offset"] == (target - shape[0]) // 2
    assert actual.attrs["window_xy"] == ("hann" if window else "none")
    assert actual.attrs["window_z"] == "none"
    if fill is not None:
        assert actual.attrs["pad_material_id"] == fill
        assert actual.attrs["pad_fill_order"] == "S=0"
    _assert_full_parity(expected, actual)


def _literal_tensor_fft(materials, shape, energy, use_hann):
    diagonal = np.array([1., 0., 0., 1., 0.])
    tensor = np.zeros((*shape, 5), dtype=np.complex128)
    for material in materials.values():
        dp, bp, dt, bt = material.opt_constants[energy]
        npar, nper = complex(1 - dp, bp), complex(1 - dt, bt)
        isotropic = ((npar + 2 * nper) / 3) ** 2 - 1
        fraction = material.Vfrac[..., None].astype(np.float64)
        if material.S is SFieldMode.ISOTROPIC:
            tensor += fraction * isotropic * diagonal
            continue
        s = material.S[..., None].astype(np.float64)
        theta = material.theta.astype(np.float64)
        psi = material.psi.astype(np.float64)
        x, y, z = np.cos(psi) * np.sin(theta), np.sin(psi) * np.sin(theta), np.cos(theta)
        tensor += fraction * (
            ((1 - s) * isotropic + s * (nper**2 - 1)) * diagonal
            + s * (npar**2 - nper**2) * np.stack((x*x, x*y, x*z, y*y, y*z), axis=-1)
        )
    if use_hann:
        tensor *= np.hanning(shape[1])[None, :, None, None] * np.hanning(shape[2])[None, None, :, None]
    return np.fft.fftn(tensor, axes=(0, 1, 2))


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
@pytest.mark.usefixtures("nrss_path")
@pytest.mark.parametrize(("method", "target"), (("sum", 7), ("residues", 12)))
@pytest.mark.parametrize("window", (0, 1))
def test_virtual_z_filled_tensor_before_and_after_dc(method, target, window, monkeypatch):
    shape = (3, 7, 8)
    _assert_complex_tensor_matches_literal(
        _absorbing_specimen(shape), (target, *shape[1:]), window, monkeypatch,
        virtual_z_pad_to=target, virtual_z_method=method,
    )


def _assert_complex_tensor_matches_literal(materials, logical, window, monkeypatch, **options):
    import cupy as cp

    from NRSS.backends import _cupy_energy_basis, _cupy_virtual_z

    literal = explicit_padded_materials(materials, logical, fill_material=1)
    original_capture = _cupy_virtual_z.VirtualZEvaluator.capture
    original_build = _cupy_energy_basis.build_energy_basis
    raw = []
    mixed = []

    def capture(evaluator, xp, field, out, recorder, **kwargs):
        original_capture(evaluator, xp, field, out, recorder, repair_dc=False, **kwargs)
        raw.append(out.copy())
        original_capture(evaluator, xp, field, out, recorder, **kwargs)

    def build(*args, **kwargs):
        basis = original_build(*args, **kwargs)
        assert basis.components == (0, 1, 2, 3, 4)
        raw_rows = cp.stack(raw)
        mixed.extend(cp.asnumpy(basis.coefficients @ rows) for rows in (raw_rows, basis.values))
        return basis

    monkeypatch.setattr(_cupy_virtual_z, "sparse_endpoint_union", lambda *_: cp.arange(math.prod(logical)))
    monkeypatch.setattr(_cupy_virtual_z.VirtualZEvaluator, "capture", capture)
    monkeypatch.setattr(_cupy_energy_basis, "build_energy_basis", build)
    _, execution = _run(_morphology(materials, window, virtual_pad_material=1, **options))
    if "virtual_3d_pad_to" in options:
        assert execution["routes"] == ["virtual_3d_sum"]
    assert len(mixed) == 2
    orders = [(length // 2 - np.arange(length)) % length for length in logical]
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    for ei, energy in enumerate(materials[1].energies):
        expected = _literal_tensor_fft(literal, logical, energy, window)
        for repair, candidate in enumerate(mixed):
            if repair:
                expected[0, 0, 0] = np.mean([expected[index] for index in neighbors], axis=0)
            signed = expected[np.ix_(*orders)].reshape(-1, 5).T
            for actual, reference in zip(candidate[ei], signed):
                peak = np.max(np.abs(reference))
                np.testing.assert_allclose(actual, reference, rtol=2.e-6, atol=2.e-6 * peak)


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
@pytest.mark.usefixtures("nrss_path")
def test_virtual_z_fill_revalidates_same_instance_and_retains_native_identity():
    shape, target = (3, 15, 16), 12
    materials = _absorbing_specimen(shape)
    morph = _morphology(materials, 0, virtual_z_pad_to=target, virtual_pad_material=2)
    try:
        first = morph.run(stdout=False, stderr=False)
        _assert_full_parity(_literal_run(materials, (target, *shape[1:]), 0, 2), first)
        morph._backend_options["virtual_pad_material"] = 4
        with pytest.raises(ValueError, match="not a registered material ID"):
            morph.run(stdout=False, stderr=False)
        morph._backend_options["virtual_pad_material"] = 1
        morph._WindowingType = 1
        materials[1].opt_constants[340.] = [.13, .01, -.05, .019]
        morph._Energies = [340., 285.]
        literal = _morphology(explicit_padded_materials(materials, (target, *shape[1:]), fill_material=1), 0)
        literal.Energies = [340., 285.]
        xy_hann_reference(literal)
        expected, _ = _run(literal)
        for _ in range(2):
            _assert_full_parity(expected, morph.run(stdout=False, stderr=False))
        morph.release_runtime()
        _assert_full_parity(expected, morph.run(stdout=False, stderr=False))
        morph._backend_options["virtual_z_pad_to"] = shape[0]
        native = _morphology(materials, 1)
        native.Energies = [340., 285.]
        expected_native, _ = _run(native)
        identity = morph.run(stdout=False, stderr=False)
        np.testing.assert_array_equal(expected_native.values, identity.values)
        assert identity.attrs == expected_native.attrs
    finally:
        morph.release_runtime()


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
@pytest.mark.usefixtures("nrss_path")
def test_virtual_z_homogeneous_fill_and_serialized_attrs(tmp_path):
    import xarray as xr

    shape = (3, 15, 16)
    materials = _scalar_materials(np.ones(shape, dtype=np.float32))
    options = dict(virtual_z_pad_to=12, virtual_pad_material=1)
    uniform, _ = _run(_morphology(materials, 0, **options))
    hann, _ = _run(_morphology(materials, 1, **options))
    expected = _literal_run(materials, (12, *shape[1:]), 1, 1)
    assert np.nanmax(hann.values) > 0
    np.testing.assert_array_equal(uniform.values[np.isfinite(uniform.values)], 0)
    _assert_full_parity(expected, hann)

    path = tmp_path / "filled.nc"
    hann.to_netcdf(path, engine="scipy")
    decoded = xr.load_dataarray(path, engine="scipy")
    assert decoded.attrs["pad_material_id"] == 1
    assert decoded.attrs["pad_fill_order"] == "S=0"
    np.testing.assert_array_equal(decoded.attrs["transform_shape_zyx"], (12, *shape[1:]))
    np.testing.assert_array_equal(decoded.values, hann.values)


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
@pytest.mark.usefixtures("nrss_path")
def test_oriented_fill_uses_the_response_of_an_equivalent_isotropic_material():
    shape = (3, 15, 16)
    options = dict(virtual_z_pad_to=12, virtual_z_method="residues")
    materials = _absorbing_specimen(shape)
    oriented, _ = _run(_morphology(materials, 1, virtual_pad_material=1, **options))
    materials[4] = Material(
        4, Vfrac=np.zeros(shape, dtype=np.float32), S=SFieldMode.ISOTROPIC,
        energies=materials[1].energies, opt_constants=materials[1].opt_constants,
    )
    isotropic, _ = _run(_morphology(materials, 1, virtual_pad_material=4, **options))
    _assert_full_parity(oriented, isotropic)


@pytest.mark.parametrize("bad", (False, 0, 5., "5", (7, 8), (7, 8, 9, 10), (7, True, 9), (7, 8, -1)))
def test_virtual_xyz_rejects_invalid_target(bad):
    with pytest.raises(BackendOptionError, match="virtual_3d_pad_to"):
        normalize_backend_options("cupy-rsoxs", {"virtual_3d_pad_to": bad})


def test_virtual_xyz_options_and_bound_shape():
    for raw, expected in ((np.int64(12), (12, 12, 12)), ([17, 18, 19], (17, 18, 19))):
        options = normalize_backend_options("cupy-rsoxs", {"virtual_3d_pad_to": raw})
        assert options["virtual_3d_pad_to"] == expected
    for conflict in ({"virtual_z_pad_to": 5}, {"virtual_z_method": "sum"}):
        with pytest.raises(BackendOptionError, match="virtual_3d_pad_to"):
            normalize_backend_options("cupy-rsoxs", {"virtual_3d_pad_to": 12, **conflict})
    with pytest.raises(ValueError, match="contain the occupied morphology"):
        _morphology(_specimen((5, 7, 8)), 0, virtual_3d_pad_to=(12, 6, 16))


def test_virtual_xyz_maps_large_indices_without_cube_allocation():
    from NRSS.backends._cupy_energy_basis import _native_indices
    from NRSS.backends.cupy_rsoxs import CupyRsoxsBackendRuntime

    n = 8192
    signed = (-3069, 2979, 2667)
    index = np.ravel_multi_index(np.asarray(signed) + n // 2, (n,) * 3)
    assert index > 2**31
    addresses, reflect = _native_indices(
        CupyRsoxsBackendRuntime(), np, (n,) * 3, np.array([index], dtype=np.int64), True,
    )
    z, y, x = (k % n for k in signed)
    assert reflect[0]
    assert addresses[0] == (z * n + y) * (n // 2 + 1) + x


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
@pytest.mark.usefixtures("nrss_path")
@pytest.mark.parametrize(("target", "window", "fill", "layout"), (
    ((14, 16, 19), 0, None, "detector"),
    ((14, 16, 19), 1, 1, "i_a"),
    ((5, 14, 16), 1, 2, "detector"),
    ((17, 7, 8), 0, 2, "detector"),
))
def test_virtual_xyz_matches_literal_box(target, window, fill, layout):
    shape = (5, 7, 8)
    materials = _absorbing_specimen(shape)
    explicit = _morphology(
        explicit_padded_materials(materials, target, fill_material=fill), 0, result_layout=layout,
    )
    if window:
        xy_hann_reference(explicit)
    expected, _ = _run(explicit)
    actual, report = _run(_morphology(
        materials, window, virtual_3d_pad_to=target, virtual_pad_material=fill, result_layout=layout,
    ))
    assert report["routes"] == ["virtual_3d_sum"]
    assert actual.attrs["occupied_shape_zyx"] == shape
    assert actual.attrs["transform_shape_zyx"] == target
    assert actual.attrs["pad_offsets_zyx"] == tuple((m-n)//2 for n, m in zip(shape, target))
    assert actual.attrs["z_dim"] == target[0]
    _assert_full_parity(expected, actual)
    if target[1:] == shape[1:]:
        z_only, _ = _run(_morphology(materials, window, virtual_z_pad_to=target[0], virtual_pad_material=fill))
        _assert_full_parity(z_only, actual)


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
@pytest.mark.usefixtures("nrss_path")
def test_virtual_xyz_retries_with_fewer_planes(monkeypatch):
    import cupy as cp

    shape, target = (3, 7, 8), (8, 14, 16)
    materials = _absorbing_specimen(shape)
    expected = _literal_run(materials, target, 0, 2)
    original = cp.fft.rfft2
    attempts = []

    def limited(field, *args, **kwargs):
        if kwargs.get("s") == target[1:]:
            attempts.append(field.shape[0])
            if field.shape[0] > 1:
                raise cp.cuda.memory.OutOfMemoryError(1024, 0)
        return original(field, *args, **kwargs)

    monkeypatch.setattr(cp.fft, "rfft2", limited)
    actual, report = _run(_morphology(materials, 0, virtual_3d_pad_to=target, virtual_pad_material=2))
    assert attempts[0] == shape[0]
    assert set(attempts[1:]) == {1}
    assert report["routes"] == ["virtual_3d_sum"]
    _assert_full_parity(expected, actual)


def _xyz_inclusion(shift):
    """Off-axis anisotropic inclusion with different lateral frequencies."""
    shape = (5, 7, 8)
    fields = _absorbing_specimen(shape)
    z, y, x = np.indices(shape)
    support = (1 <= z) & (z <= 3) & (1 <= y) & (y <= 4) & (2 <= x) & (x <= 5)
    inclusion = np.where(support, .45 + .12 * np.sin(x * np.pi / 2) + .09 * np.cos(y * np.pi / 3), 0.)
    fields[1].Vfrac = np.roll(inclusion.astype(np.float32), shift, axis=(1, 2))
    for name in ("S", "theta", "psi"):
        value = getattr(fields[1], name)
        setattr(fields[1], name, np.roll(value, shift, axis=(1, 2)))
    fields[2].Vfrac = 1 - fields[1].Vfrac
    return fields


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
@pytest.mark.usefixtures("nrss_path")
@pytest.mark.parametrize("target", ((14, 16, 19), (5, 14, 16)))
@pytest.mark.parametrize("window", (0, 1))
@pytest.mark.parametrize("shift", ((0, 0), (1, -1)))
def test_virtual_xyz_signed_tensor_and_dc(target, window, shift, monkeypatch):
    fields = _xyz_inclusion(shift)
    with monkeypatch.context() as capture:
        _assert_complex_tensor_matches_literal(fields, target, window, capture, virtual_3d_pad_to=target)
    literal = _literal_run(fields, target, window, 1)
    actual, _ = _run(_morphology(fields, window, virtual_3d_pad_to=target, virtual_pad_material=1))
    _assert_full_parity(literal, actual)


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
@pytest.mark.usefixtures("nrss_path")
@pytest.mark.parametrize(("layout", "residency", "q_grid"), (
    ("detector", "device", "intersection"),
    ("integrated", "device", "union"),
    ("i_only", "host", "union"),
    ("i_para_i_perp", "device", "intersection"),
    ("i_a", "device", "union"),
))
def test_virtual_xyz_result_policies_and_rotations(layout, residency, q_grid, tmp_path):
    import cupy as cp
    import xarray as xr

    fields = _xyz_inclusion((1, 0))
    target = (11, 20, 23)
    options = dict(result_layout=layout, result_q_grid_mode=q_grid)
    literal = _morphology(explicit_padded_materials(fields, target, fill_material=1), 0, **options)
    virtual = _morphology(
        fields, 1, virtual_3d_pad_to=target, virtual_pad_material=1, result_residency=residency, **options,
    )
    xy_hann_reference(literal)
    try:
        for morph in (literal, virtual):
            morph.EAngleRotation = [0., 15., 165.]
            morph.Energies = [340., 285.]
            morph.RotMask = 1
        expected, _ = _run(literal)
        virtual._set_private_backend_timing_segments(("energy",))
        backend_result = virtual.run(stdout=False, stderr=False, return_xarray=False)
        assert isinstance(backend_result.to_backend_array(), cp.ndarray) == (residency == "device")
        actual = backend_result.to_xarray()
        assert virtual.backend_timings["execution"]["routes"] == ["virtual_3d_sum"]
        assert virtual.backend_timings["execution"]["basis_builds"] == 1
        assert list(actual.energy.values) == [340., 285.]
        assert actual.attrs["pad_fill_order"] == "S=0"
        _assert_full_parity(expected, actual)
        path = tmp_path / f"{layout}.nc"
        actual.to_netcdf(path, engine="scipy")
        restored = (xr.load_dataset if isinstance(actual, xr.Dataset) else xr.load_dataarray)(path, engine="scipy")
        np.testing.assert_array_equal(restored.attrs["transform_shape_zyx"], target)
        np.testing.assert_array_equal(restored.attrs["pad_offsets_zyx"], (3, 6, 7))
        assert restored.attrs["pad_material_id"] == 1
        _assert_full_parity(actual, restored)
    finally:
        virtual.release_runtime()


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
@pytest.mark.usefixtures("nrss_path")
def test_virtual_xyz_shape_and_fill_changes_rebuild_matching_geometry():
    fields = _xyz_inclusion((0, 0))
    shape = fields[1].Vfrac.shape
    run = _morphology(fields, 0, virtual_3d_pad_to=(9, 14, 17), virtual_pad_material=2)
    try:
        for target, fill in (((9, 14, 17), 2), ((13, 17, 20), 1), ((9, 14, 17), 2)):
            run._backend_options.update(virtual_3d_pad_to=target, virtual_pad_material=fill)
            actual = run.run(stdout=False, stderr=False)
            assert actual.attrs["transform_shape_zyx"] == target
            assert actual.attrs["pad_material_id"] == fill
            _assert_full_parity(_literal_run(fields, target, 0, fill), actual)
        run._backend_options["virtual_3d_pad_to"] = shape
        native, _ = _run(_morphology(fields, 0))
        identity = run.run(stdout=False, stderr=False)
        np.testing.assert_array_equal(native.values, identity.values)
        assert native.attrs == identity.attrs
    finally:
        run.release_runtime()
