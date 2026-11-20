"""Opt-in B/C comparisons for the morphologies exercised by maintained pytest.

Install this wrapper in the test process and its subprocesses. Each original
run still supplies its native result to the existing assertion. Eligible
morphologies additionally run through the two public padded entries.
"""

from functools import wraps
import json
import os

import numpy as np

from NRSS.morphology import Material, Morphology
from tests.validation.lib.bragg import release_runtime_memory
from tests.validation.lib.virtual_z_padding import explicit_vacuum_materials, xy_hann_reference


def record(log_path, data):
    line = (json.dumps(data, allow_nan=False) + "\n").encode()
    fd = os.open(log_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def materials_copy(materials):
    return {
        key: Material(
            key, Vfrac=material.Vfrac, S=material.S, theta=material.theta, psi=material.psi,
            energies=material.energies, opt_constants=material.opt_constants, name=material.name,
        )
        for key, material in materials.items()
    }


def padded_run(run, source, materials, target, virtual):
    options = source.backend_options
    # Keep the selected basis contract explicit when rebuilding from the
    # resolved options, which include an execution path even for auto runs.
    options.update(execution_path="tensor_coeff", energy_basis_mode="compact_r2c")
    config = source.config
    hann = config["WindowingType"] == 1
    if virtual:
        options["virtual_z_pad_to"] = target
    else:
        config["WindowingType"] = 0
    morph = Morphology(
        len(materials), materials=materials, PhysSize=source.PhysSize, config=config,
        backend="cupy-rsoxs", backend_options=options, resident_mode=source.resident_mode,
        input_policy="strict", ownership_policy="borrow",
    )
    if hann and not virtual:
        xy_hann_reference(morph)
    try:
        morph._set_private_backend_timing_segments(("energy",))
        data = run(morph, stdout=False, stderr=False, return_xarray=True).copy(deep=True)
        route = morph.backend_timings["execution"]["routes"]
        assert route == (["virtual_z_sum"] if virtual else ["compact_r2c"])
        assert data.attrs["detector_grid_convention"] == "fft_bins_v1"
        assert data.attrs["z_dim"] == target
        if virtual:
            assert data.attrs["occupied_z"] == source.NumZYX[0]
            assert data.attrs["window_xy"] == ("hann" if hann else "none")
            assert data.attrs["window_z"] == "none"
        return data, route
    finally:
        morph.release_runtime()
        release_runtime_memory()


def compare(explicit, virtual):
    assert explicit.sizes == virtual.sizes
    assert tuple(explicit.coords) == tuple(virtual.coords)
    for name in explicit.coords:
        np.testing.assert_array_equal(explicit.coords[name], virtual.coords[name])
    if hasattr(explicit, "data_vars"):
        assert tuple(explicit.data_vars) == tuple(virtual.data_vars)
        pairs = ((name, explicit[name], virtual[name]) for name in explicit.data_vars)
    else:
        pairs = (("values", explicit, virtual),)
    output = {}
    for name, reference, candidate in pairs:
        reference = reference.transpose("energy", ...)
        candidate = candidate.transpose("energy", ...)
        worst = peak_scaled = 0.
        finite_samples = 0
        for a, b in zip(reference.values, candidate.values):
            assert not np.isinf(a).any() and not np.isinf(b).any()
            valid = np.isfinite(a)
            np.testing.assert_array_equal(valid, np.isfinite(b))
            finite_samples += int(valid.sum())
            x, y = a[valid].astype(np.float64), b[valid].astype(np.float64)
            delta = y - x
            worst = max(worst, float(np.linalg.norm(delta)) / max(float(np.linalg.norm(x)), 1.e-30))
            peak_scaled = max(peak_scaled, float(np.max(abs(delta), initial=0.)) /
                              max(float(np.max(abs(x), initial=0.)), 1.e-30))
        output[name] = {"relative_l2": worst, "max_difference_over_peak": peak_scaled,
                        "finite_samples": finite_samples}
    return output


def excluded(owner, label):
    if owner.backend != "cupy-rsoxs":
        return "other backend"
    if "test_cupy_virtual_z.py" in label or "vacuum_padding_reference" in label:
        return "dedicated padding comparison"
    if "memory_fallback" in label or "fallback" in label:
        return "injected allocation failure"
    if owner.backend_options.get("virtual_z_pad_to") is not None:
        return "explicit virtual control"
    mode, reason = owner._backend_runtime._energy_basis_selection(owner)
    if mode != "compact_r2c":
        return reason or "other execution mode"
    if min(owner.NumZYX[1:]) < 2:
        return "singleton lateral axis"
    if owner.mixed_precision_mode is not None:
        return "reduced morphology precision"
    return None


def install(log_path):
    run = Morphology.run

    @wraps(run)
    def audited(owner, *args, **kwargs):
        native = run(owner, *args, **kwargs)
        label = os.environ.get("PYTEST_CURRENT_TEST", "")
        if not label:
            return native
        entry = {
            "test": label, "backend": owner.backend, "shape": list(owner.NumZYX),
            "energy_count": len(owner.Energies), "windowing_type": owner.WindowingType,
            "requested_path": owner.backend_options.get("execution_path"),
            "layout": owner.backend_options.get("result_layout"),
        }
        skip = excluded(owner, label)
        if skip is not None:
            record(log_path, dict(entry, status="skipped", reason=skip))
            return native
        target = 2 * owner.NumZYX[0]
        entry["transform_z"] = target
        if owner.resident_mode == "device":
            import cupy as xp
        else:
            xp = np
        try:
            explicit, _ = padded_run(
                run, owner, explicit_vacuum_materials(owner.materials, target, xp=xp), target, False,
            )
            virtual, route = padded_run(run, owner, materials_copy(owner.materials), target, True)
            entry.update(route=route, metrics=compare(explicit, virtual))
            if any(value["relative_l2"] > 2.e-5 or value["max_difference_over_peak"] > 2.e-5
                   for value in entry["metrics"].values()):
                raise AssertionError(f"Virtual film differs from the explicit reference: {entry['metrics']}")
        except Exception as error:
            record(log_path, dict(entry, status="failed", error=str(error)))
            raise
        record(log_path, dict(entry, status="passed"))
        return native

    Morphology.run = audited
