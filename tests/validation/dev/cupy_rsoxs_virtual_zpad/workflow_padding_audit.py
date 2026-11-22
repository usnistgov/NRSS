"""Compare virtual and literal padding for maintained CuPy test inputs."""

from functools import wraps
import os

import numpy as np
from scipy.fft import next_fast_len

from NRSS import SFieldMode
from NRSS.morphology import Morphology
from tests.validation.dev.cupy_rsoxs_virtual_zpad.film_parity_audit import (
    compare,
    excluded,
    materials_copy,
    record,
)
from tests.validation.lib.virtual_z_padding import explicit_padded_materials, xy_hann_reference


TOLERANCE = 2e-5


def _targets(shape):
    nz, ny, nx = shape
    depth = next_fast_len(nz + 2, real=True)
    padded_y = next_fast_len(ny + 2, real=True)
    padded_x = next_fast_len(nx + 2, real=True)
    if padded_x == padded_y:
        padded_x = next_fast_len(padded_x + 1, real=True)
    return {"virtual_3d_sum": (depth, padded_y, padded_x),
            "virtual_z_sum": (depth, ny, nx)}


def _fill_id(owner):
    candidates = [
        material for material in owner.materials.values()
        if any(np.any(material.opt_constants[energy]) for energy in owner.Energies)
    ]
    oriented = [material for material in candidates if not isinstance(material.S, SFieldMode)]
    selected = oriented or candidates
    return selected[0].materialID if selected else None


def _arm(run, source, route, target, fill, layout, *, literal):
    cfg = source.config
    options = source.backend_options
    options.update(execution_path="tensor_coeff", energy_basis_mode="compact_r2c", result_layout=layout,
                   result_residency="host", result_chunk_size=1, energy_progress_bar=False)
    hann = cfg["WindowingType"] == 1
    if literal:
        if source.resident_mode == "device":
            import cupy as xp
        else:
            xp = np
        materials = explicit_padded_materials(source.materials, target, fill_material=fill, xp=xp)
        cfg["WindowingType"] = 0
    else:
        materials = materials_copy(source.materials)
        if route == "virtual_3d_sum":
            options["virtual_3d_pad_to"] = target
        else:
            options.update(virtual_z_pad_to=target[0], virtual_z_method="sum")
        if fill is not None:
            options["virtual_pad_material"] = fill

    morph = Morphology(
        len(materials), materials=materials, PhysSize=source.PhysSize, config=cfg,
        backend="cupy-rsoxs", backend_options=options, resident_mode=source.resident_mode,
        input_policy="strict", ownership_policy="borrow",
    )
    morph.Energies = list(source.Energies)
    if not literal and route == "virtual_3d_sum" and os.environ.get("NRSS_PADDING_SPEED_AUDIT_LOG"):
        from NRSS.backends._cupy_virtual_xyz import XYZ_FACTORED_XY_KEY
        from NRSS.backends._cupy_xyz_endpoints import XYZ_ENDPOINT_PLAN_KEY
        from NRSS.backends.cupy_rsoxs import XYZ_ENERGY_TILE_KEY

        # Stress the optimized routes on small and irregular morphologies in
        # addition to the automatic selections in the ordinary panel.
        morph._backend_runtime_state[XYZ_FACTORED_XY_KEY] = True
        morph._backend_runtime_state[XYZ_ENERGY_TILE_KEY] = (8, True)
        morph._backend_runtime_state[XYZ_ENDPOINT_PLAN_KEY] = (
            "spans" if len(set(morph.Energies)) > 1 else "bitmap"
        )
    if literal and hann:
        xy_hann_reference(morph)
    try:
        morph._set_private_backend_timing_segments(("energy",))
        data = run(morph, stdout=False, stderr=False, return_xarray=True)
        execution = morph.backend_timings["execution"]
        expected = "compact_r2c" if literal else route
        assert execution["routes"] == [expected] and execution["basis_builds"] == 1
        assert data.attrs["detector_grid_convention"] == "fft_bins_v1"
        assert data.attrs["z_dim"] == target[0]
        if "q" in data.coords:
            assert data.attrs["radial_grid_convention"] == "sampled_q_v1"
        np.testing.assert_array_equal(data.energy, source.Energies)
        for axis, length in zip(("qy", "qx"), target[1:]):
            if axis in data.coords:
                bins = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(length, d=source.PhysSize))
                np.testing.assert_array_equal(data[axis], bins)
        if not literal:
            assert data.attrs["transform_shape_zyx"] == target
            assert data.attrs["occupied_shape_zyx"] == source.NumZYX
            assert data.attrs["window_xy"] == ("hann" if hann else "none")
            assert data.attrs["window_z"] == "none"
            if fill is not None:
                assert data.attrs["pad_material_id"] == fill
        return data, execution
    finally:
        morph.release_runtime()


def install(log_path):
    """Audit inputs from public runs in pytest and any inherited subprocesses."""
    run = Morphology.run
    reduced_tests = set()

    @wraps(run)
    def audited(owner, *args, **kwargs):
        native = run(owner, *args, **kwargs)
        label = os.environ.get("PYTEST_CURRENT_TEST")
        if not label:
            return native
        options = owner.backend_options
        entry = dict(
            test=label, pid=os.getpid(), backend=owner.backend, shape=list(owner.NumZYX),
            energies_eV=list(owner.Energies), angles_start_increment_end=list(owner.EAngleRotation),
            pitch_nm=owner.PhysSize, windowing_type=owner.WindowingType,
            requested_path=options.get("execution_path"), layout=options.get("result_layout"),
            native_result_residency=options.get("result_residency"),
        )
        skip = excluded(owner, label)
        if options.get("virtual_3d_pad_to") is not None:
            skip = "dedicated XYZ control"
        if skip:
            record(log_path, dict(entry, status="skipped", reason=skip))
            return native

        layouts = [options["result_layout"]]
        if label.startswith("tests/validation/") and label not in reduced_tests and layouts == ["detector"]:
            layouts.append("i_para_i_perp")
            reduced_tests.add(label)
        for layout in layouts:
            for route, target in _targets(owner.NumZYX).items():
                fill = _fill_id(owner) if route == "virtual_z_sum" else None
                case = dict(entry, route=route, transform_zyx=target, fill_material=fill,
                            layout=layout, result_residency="host", tolerance=TOLERANCE)
                if route == "virtual_z_sum" and fill is None:
                    record(log_path, dict(case, status="skipped", reason="no optically nonzero material"))
                    continue
                try:
                    _compare_arms(run, owner, route, target, fill, layout, case)
                except Exception as error:
                    record(log_path, dict(case, status="failed", error=str(error)))
                    raise
                record(log_path, dict(case, status="passed"))
        return native

    Morphology.run = audited


def _compare_arms(run, owner, route, target, fill, layout, case):
    reference, literal_execution = _arm(run, owner, route, target, fill, layout, literal=True)
    virtual, virtual_execution = _arm(run, owner, route, target, fill, layout, literal=False)
    metrics = compare(reference, virtual)
    case.update(
        metrics=metrics,
        fft_calls={"literal": literal_execution["fft_count"], "virtual": virtual_execution["fft_count"]},
        basis_cache=virtual_execution["basis_cache"],
        energy_projector=virtual_execution.get("energy_projector"),
        compute_tile_size=virtual_execution["compute_tile_size"],
    )
    if any(max(m["relative_l2"], m["max_difference_over_peak"]) > TOLERANCE
           for m in metrics.values()):
        raise AssertionError(f"{route} differs from the literal box: {metrics}")
