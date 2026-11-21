"""Guard the coordinates and samples used in the CoreShell residuals."""
import numpy as np
import pytest
import xarray as xr

from tests.validation.lib.core_shell import (
    EXPERIMENTAL_THRESHOLDS,
    awedge_comparison_slices,
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
