from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pytest

from tests.validation.lib.core_shell import (
    BASELINE_SCENARIO,
    EXPERIMENTAL_REFERENCE_CITATION,
    EXPERIMENTAL_REFERENCE_LABEL,
    EXPERIMENTAL_THRESHOLDS,
    SIM_REFERENCE_LABEL,
    SIM_REFERENCE_PATH,
    SIM_THRESHOLDS,
    awedge_comparison_slices,
    compute_awedge_metrics,
    has_visible_gpu,
    load_experimental_reference_awedge,
    load_sim_reference_awedge,
    metrics_within_thresholds,
    plot_core_shell_validation_panel,
    run_core_shell_backend,
    run_core_shell_pybind,
    scattering_to_awedge,
)


pytestmark = [pytest.mark.backend_specific, pytest.mark.reference_parity]


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = REPO_ROOT / "test-reports" / "core-shell-dev"
WRITE_VALIDATION_PLOTS = os.environ.get("NRSS_WRITE_VALIDATION_PLOTS", "").strip() == "1"


@lru_cache(maxsize=8)
def _baseline_awedge(backend: str, input_policy: str, ownership_policy: str | None, field_namespace: str):
    if backend == "cyrsoxs":
        scattering = run_core_shell_pybind(scenario="baseline")
    else:
        scattering, _ = run_core_shell_backend(
            scenario="baseline",
            backend=backend,
            input_policy=input_policy,
            ownership_policy=ownership_policy,
            field_namespace=field_namespace,
        )
    return scattering_to_awedge(scattering).copy(deep=True)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.physics_validation
@pytest.mark.experimental_validation
@pytest.mark.toolchain_validation
def test_core_shell_experimental_reference_pybind(nrss_backend):
    """
    Compare the maintained pybind+WPIntegrator CoreShell A-wedge to the vendored
    experimental PGN RSoXS reference.

    Reference:
    Subhrangsu Mukherjee, Jason K. Streit, Eliot Gann, Kumar Saurabh, Daniel F.
    Sunday, Adarsh Krishnamurthy, Baskar Ganapathysubramanian, Lee J. Richter,
    Richard A. Vaia, and Dean M. DeLongchamp, "Polarized X-ray scattering
    measures molecular orientation in polymer-grafted nanoparticles," Nature
    Communications 12, 4896 (2021), doi:10.1038/s41467-021-25176-4.
    """
    if not has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for CoreShell experimental validation.")

    awedge = _baseline_awedge(
        nrss_backend,
        "coerce",
        None,
        "numpy",
    )
    reference = load_experimental_reference_awedge()
    comparison = awedge_comparison_slices(awedge=awedge, reference=reference)
    metrics = compute_awedge_metrics(comparison)
    passed, failures = metrics_within_thresholds(metrics, EXPERIMENTAL_THRESHOLDS)

    if WRITE_VALIDATION_PLOTS:
        plot_core_shell_validation_panel(
            comparison=comparison,
            metrics=metrics,
            out_path=PLOT_DIR / "core_shell_baseline_pytest_vs_experiment.png",
            scenario=BASELINE_SCENARIO,
            reference_label=EXPERIMENTAL_REFERENCE_LABEL,
            reference_citation=EXPERIMENTAL_REFERENCE_CITATION,
        )

    if not passed:
        raise AssertionError("; ".join(failures))


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.physics_validation
@pytest.mark.toolchain_validation
def test_core_shell_sim_regression_pybind(nrss_backend):
    """
    Compare the maintained pybind+WPIntegrator CoreShell A-wedge to the local
    sim-derived regression golden stored alongside the experimental reference.
    """
    if not has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for CoreShell sim-regression validation.")
    if not SIM_REFERENCE_PATH.exists():
        pytest.skip(f"CoreShell sim reference not found: {SIM_REFERENCE_PATH}")

    awedge = _baseline_awedge(
        nrss_backend,
        "coerce",
        None,
        "numpy",
    )
    reference = load_sim_reference_awedge()
    comparison = awedge_comparison_slices(awedge=awedge, reference=reference)
    metrics = compute_awedge_metrics(comparison)
    passed, failures = metrics_within_thresholds(metrics, SIM_THRESHOLDS)

    if WRITE_VALIDATION_PLOTS:
        plot_core_shell_validation_panel(
            comparison=comparison,
            metrics=metrics,
            out_path=PLOT_DIR / "core_shell_baseline_pytest_vs_sim_reference.png",
            scenario=BASELINE_SCENARIO,
            reference_label=SIM_REFERENCE_LABEL,
            reference_citation=(
                "Sim-derived regression golden generated from the maintained baseline "
                "CoreShell scenario. This artifact guards implementation stability, "
                "not experimental truth."
            ),
        )

    if not passed:
        raise AssertionError("; ".join(failures))


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.backend_specific
@pytest.mark.reference_parity
def test_core_shell_sim_regression_cupy_borrow_strict(nrss_backend):
    """
    Compare the CuPy-native borrowed CoreShell workflow to the maintained local
    sim-derived regression golden.
    """
    if not has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for CoreShell CuPy borrowed validation.")
    if nrss_backend != "cupy-rsoxs":
        pytest.skip("CuPy borrowed CoreShell validation is only applicable to cupy-rsoxs.")

    awedge = _baseline_awedge(
        "cupy-rsoxs",
        "strict",
        "borrow",
        "cupy",
    )
    reference = load_sim_reference_awedge()
    comparison = awedge_comparison_slices(awedge=awedge, reference=reference)
    metrics = compute_awedge_metrics(comparison)
    passed, failures = metrics_within_thresholds(metrics, SIM_THRESHOLDS)

    if not passed:
        raise AssertionError("; ".join(failures))
