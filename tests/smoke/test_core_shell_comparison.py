"""Guard the coordinates and samples used in the CoreShell residuals."""
import numpy as np
import pytest
import xarray as xr

from tests.validation.lib.core_shell import (
    EXPERIMENTAL_THRESHOLDS,
    awedge_comparison_slices,
    build_core_shell_morphology,
    compute_awedge_metrics,
    metrics_within_thresholds,
)


def _reference():
    return xr.DataArray(
        np.zeros((3, 3)), dims=("energy", "q"),
        coords={"energy": [284.7, 285.2, 285.7], "q": [0.05, 0.20, 0.39]},
    )


@pytest.mark.parametrize("mismatch", ["q", "energy", "order", "count", "mask", "dimensions"])
def test_core_shell_comparison_rejects_mismatched_samples(mismatch):
    reference = _reference()
    observed = reference.copy(deep=True)
    if mismatch == "q":
        observed = observed.assign_coords(q=[0.05, 0.20, 0.38])
    elif mismatch == "energy":
        observed = observed.assign_coords(energy=[284.7, 285.2, 285.8])
    elif mismatch == "order":
        observed = observed.isel(energy=slice(None, None, -1))
    elif mismatch == "count":
        observed = observed.isel(q=slice(1, None))
    elif mismatch == "mask":
        observed.loc[{"energy": 285.2, "q": 0.39}] = np.nan
    else:
        observed = observed.transpose("q", "energy")
    with pytest.raises(AssertionError, match="coordinates|sample mask|dimensions"):
        awedge_comparison_slices(observed, reference)


def test_core_shell_comparison_reports_peak_residual_location_and_count():
    reference = _reference()
    observed = reference.copy(deep=True)
    observed.loc[{"energy": 285.2, "q": 0.39}] = 0.065
    metrics = compute_awedge_metrics(awedge_comparison_slices(observed, reference))
    assert metrics["a_vs_q_285p2"]["n_compared"] == 3
    assert metrics["a_vs_q_285p2"]["max_abs_at_q"] == 0.39
    passed, failures = metrics_within_thresholds(metrics, EXPERIMENTAL_THRESHOLDS)
    assert not passed
    assert any("a_vs_q_285p2 max_abs_diff" in message and "at q=0.39" in message
               for message in failures)


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff", "cupy_direct_polarization")
@pytest.mark.usefixtures("nrss_path")
def test_small_core_shell_direct_polarization_matches_compact_tensor():
    """Check the direct detector against compact tensor for a small CoreShell film."""
    shape = (32, 40, 48)
    energies = [284.7, 285.2]
    centers = np.array([[13.25, 16.5], [32.5, 25.25]])
    panels = {}
    for path, route in (("tensor_coeff", "compact_r2c"),
                        ("direct_polarization", "direct_polarization")):
        morph = build_core_shell_morphology(
            backend="cupy-rsoxs", backend_options={"execution_path": path},
            shape=shape, centers_xy=centers,
        )
        try:
            morph.Energies = energies
            morph.EAngleRotation = [15.0, 30.0, 75.0]  # StartAngle, IncrementAngle, EndAngle
            morph._set_private_backend_timing_segments(("energy",))
            panels[path] = morph.run(stdout=False, stderr=False)
            assert morph.backend_timings["execution"]["routes"] == [route]
        finally:
            morph.release_runtime()

    reference, direct = panels["tensor_coeff"], panels["direct_polarization"]
    assert reference.dims == direct.dims == ("energy", "qy", "qx")
    assert reference.shape == direct.shape == (len(energies), *shape[1:])
    np.testing.assert_array_equal(direct.energy, energies)
    for coord in reference.coords:
        np.testing.assert_array_equal(direct[coord], reference[coord])
    for expected, actual in zip(reference.values, direct.values, strict=True):
        assert not np.isinf(expected).any() and not np.isinf(actual).any()
        valid = np.isfinite(expected)
        np.testing.assert_array_equal(valid, np.isfinite(actual))
        peak = expected[valid].max()
        assert peak > 0.0
        np.testing.assert_allclose(actual[valid], expected[valid], rtol=0.0, atol=2e-5 * peak)
