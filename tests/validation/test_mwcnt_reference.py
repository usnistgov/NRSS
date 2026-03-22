from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pytest

from tests.validation.lib.core_shell import has_visible_gpu
from tests.validation.lib.mwcnt import (
    EXPERIMENTAL_REFERENCE_CITATION,
    EXPERIMENTAL_REFERENCE_LABEL,
    EXPERIMENTAL_THRESHOLDS,
    Q_COMPARE_MAX_NM,
    Q_COMPARE_MIN_NM,
    align_observables_to_reference,
    compute_observable_metrics,
    load_experimental_reference_observables,
    metrics_within_thresholds,
    plot_mwcnt_validation_panel,
    reduce_scattering_to_observables,
    run_mwcnt_pybind,
)


pytestmark = [pytest.mark.backend_specific, pytest.mark.reference_parity]


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = REPO_ROOT / "test-reports" / "mwcnt-dev"
WRITE_VALIDATION_PLOTS = os.environ.get("NRSS_WRITE_VALIDATION_PLOTS", "").strip() == "1"


@lru_cache(maxsize=1)
def _baseline_observables():
    scattering = run_mwcnt_pybind()
    return {
        key: value.copy(deep=True) for key, value in reduce_scattering_to_observables(scattering).items()
    }


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.physics_validation
@pytest.mark.experimental_validation
@pytest.mark.toolchain_validation
def test_mwcnt_experimental_reference_pybind():
    """
    Compare the maintained pybind+WPIntegrator MWCNT anisotropy observables to
    the vendored experimental MWCNT WAXS reduction derived from the published
    tutorial series and manuscript Table I parameterization.

    Reference:
    P. J. Dudenas, L. Q. Flagg, K. Goetz, P. Shapturenka, J. A. Fagan,
    E. Gann, and D. M. DeLongchamp, "How to RSoXS,"
    J. Chem. Phys. 163, 061501 (2025), doi:10.1063/5.0267709.
    """
    if not has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for MWCNT experimental validation.")

    observables = _baseline_observables()
    reference = load_experimental_reference_observables()
    comparison = align_observables_to_reference(observables, reference)
    metrics = compute_observable_metrics(comparison)
    passed, failures = metrics_within_thresholds(metrics, EXPERIMENTAL_THRESHOLDS)

    if WRITE_VALIDATION_PLOTS:
        plot_mwcnt_validation_panel(
            comparison=comparison,
            metrics=metrics,
            out_path=PLOT_DIR / "mwcnt_pytest_vs_experiment.png",
            q_min_nm=Q_COMPARE_MIN_NM,
            q_max_nm=Q_COMPARE_MAX_NM,
        )

    if not passed:
        raise AssertionError(
            "MWCNT experimental validation failed against "
            f"{EXPERIMENTAL_REFERENCE_LABEL}: {EXPERIMENTAL_REFERENCE_CITATION}; "
            + "; ".join(failures)
        )
