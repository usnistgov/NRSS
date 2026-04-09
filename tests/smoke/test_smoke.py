import importlib
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pytest
import NRSS.backends.registry as backend_registry

from tests.path_matrix import ComputationPath


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from NRSS import SFieldMode
from NRSS.reader import read_config
from NRSS.backends import (
    BackendInfo,
    BackendOptionError,
    assess_array_for_backend,
    available_backends,
    coerce_array_for_backend,
    format_backend_availability,
    get_backend_info,
    inspect_array,
    resolve_backend_array_contract,
    resolve_backend_name,
    resolve_backend_runtime_contract,
    UnknownBackendError,
)
from NRSS.morphology import Material, Morphology, OpticalConstants
from NRSS.writer import write_config


pytestmark = [pytest.mark.smoke]


def _import_required(module_name: str):
    mod = importlib.import_module(module_name)
    assert mod is not None
    return mod


def _import_cyrsoxs_required():
    # CyRSoXS remains an explicitly supported legacy backend and reference target.
    try:
        return importlib.import_module("CyRSoXS")
    except Exception as exc:  # pragma: no cover - exercised when import fails
        raise AssertionError(
            "CyRSoXS import failed for the supported module name. "
            f"Failure: {exc.__class__.__name__}({exc})"
        ) from exc


def _import_cupy_required():
    try:
        mod = importlib.import_module("cupy")
    except Exception as exc:  # pragma: no cover - exercised when import fails
        raise AssertionError(
            "CuPy import failed for the supported NRSS runtime environment. "
            f"Failure: {exc.__class__.__name__}({exc})"
        ) from exc
    assert mod is not None
    return mod


def _release_cupy_memory():
    try:
        cp = importlib.import_module("cupy")
    except Exception:
        return
    cp.cuda.Device().synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _to_backend_namespace(array: np.ndarray, field_namespace: str, dtype=np.float32):
    if field_namespace == "numpy":
        return np.ascontiguousarray(array.astype(dtype, copy=False))
    if field_namespace == "cupy":
        cp = _import_cupy_required()
        return cp.ascontiguousarray(cp.asarray(array, dtype=cp.dtype(dtype)))
    raise AssertionError(f"Unsupported field namespace {field_namespace!r}.")


def _clone_backend_array(array):
    info = inspect_array(array)
    if info["namespace"] == "numpy":
        return np.ascontiguousarray(np.array(array, dtype=array.dtype, copy=True))
    if info["namespace"] == "cupy":
        cp = _import_cupy_required()
        return cp.asarray(cp.asnumpy(array), dtype=array.dtype)
    raise AssertionError(f"Unsupported backend array namespace {info['namespace']!r}.")


def _path_runtime_kwargs(nrss_path: ComputationPath) -> dict[str, object]:
    return {
        "backend": nrss_path.backend,
        "backend_options": nrss_path.backend_options,
        "resident_mode": nrss_path.resident_mode,
        "ownership_policy": nrss_path.ownership_policy,
        "field_namespace": nrss_path.field_namespace,
    }


def _tiny_smoothing_kernel(arr: np.ndarray) -> np.ndarray:
    return (
        0.25 * arr
        + 0.125 * np.roll(arr, 1, axis=0)
        + 0.125 * np.roll(arr, -1, axis=0)
        + 0.125 * np.roll(arr, 1, axis=1)
        + 0.125 * np.roll(arr, -1, axis=1)
        + 0.0625 * np.roll(np.roll(arr, 1, axis=0), 1, axis=1)
        + 0.0625 * np.roll(np.roll(arr, 1, axis=0), -1, axis=1)
        + 0.0625 * np.roll(np.roll(arr, -1, axis=0), 1, axis=1)
        + 0.0625 * np.roll(np.roll(arr, -1, axis=0), -1, axis=1)
    )


def _build_two_material_isotropic_block_morphology(
    *,
    backend: str = "cupy-rsoxs",
    backend_options: dict | None = None,
    resident_mode: str | None = None,
    field_namespace: str = "numpy",
    isotropic_representation: str = "legacy_zero_array",
    ignored_orientation_arrays: bool = False,
    array_dtype=np.float32,
):
    shape = (4, 16, 16)
    energies = [285.0]
    vfrac_1 = np.zeros(shape, dtype=np.float32)
    vfrac_1[1:3, 4:12, 5:11] = 1.0
    vfrac_2 = np.float32(1.0) - vfrac_1
    zeros = np.zeros(shape, dtype=np.float32)

    if field_namespace == "cupy":
        cp = _import_cupy_required()
        cp_dtype = cp.dtype(array_dtype)
        vfrac_1 = cp.asarray(vfrac_1, dtype=cp_dtype)
        vfrac_2 = cp.asarray(vfrac_2, dtype=cp_dtype)
        zeros = cp.asarray(zeros, dtype=cp_dtype)
    elif field_namespace != "numpy":
        raise AssertionError(f"Unsupported field namespace {field_namespace!r}.")
    else:
        vfrac_1 = np.ascontiguousarray(vfrac_1.astype(array_dtype, copy=False))
        vfrac_2 = np.ascontiguousarray(vfrac_2.astype(array_dtype, copy=False))
        zeros = np.ascontiguousarray(zeros.astype(array_dtype, copy=False))

    if isotropic_representation == "legacy_zero_array":
        mat1_s = _clone_backend_array(zeros)
        mat1_theta = _clone_backend_array(zeros)
        mat1_psi = _clone_backend_array(zeros)
        mat2_s = _clone_backend_array(zeros)
        mat2_theta = _clone_backend_array(zeros)
        mat2_psi = _clone_backend_array(zeros)
    elif isotropic_representation == "enum_contract":
        mat1_s = SFieldMode.ISOTROPIC
        mat1_theta = _clone_backend_array(zeros) if ignored_orientation_arrays else None
        mat1_psi = _clone_backend_array(zeros) if ignored_orientation_arrays else None
        mat2_s = SFieldMode.ISOTROPIC
        mat2_theta = _clone_backend_array(zeros) if ignored_orientation_arrays else None
        mat2_psi = _clone_backend_array(zeros) if ignored_orientation_arrays else None
    else:
        raise AssertionError(
            f"Unsupported isotropic_representation {isotropic_representation!r}."
        )

    mat1 = Material(
        materialID=1,
        Vfrac=vfrac_1,
        S=mat1_s,
        theta=mat1_theta,
        psi=mat1_psi,
        energies=energies,
        opt_constants={285.0: [1e-4, 2e-4, 1e-4, 2e-4]},
        name="block",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=vfrac_2,
        S=mat2_s,
        theta=mat2_theta,
        psi=mat2_psi,
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="matrix",
    )
    config = {
        "CaseType": 0,
        "MorphologyType": 0,
        "Energies": energies,
        "EAngleRotation": [0.0, 0.0, 0.0],
        "RotMask": 0,
        "WindowingType": 0,
        "AlgorithmType": 0,
        "ReferenceFrame": 1,
        "EwaldsInterpolation": 1,
    }
    return Morphology(
        2,
        materials={1: mat1, 2: mat2},
        PhysSize=5.0,
        config=config,
        backend=backend,
        backend_options=backend_options,
        resident_mode=resident_mode,
        input_policy="strict",
        ownership_policy="borrow" if backend == "cupy-rsoxs" else None,
        create_cy_object=True,
    )


def _build_two_material_isotropic_single_slice_morphology(
    *,
    backend_options: dict | None = None,
) -> Morphology:
    shape = (1, 16, 16)
    energies = [285.0]
    vfrac_1 = np.zeros(shape, dtype=np.float32)
    vfrac_1[0, 4:12, 5:11] = 1.0
    vfrac_2 = np.float32(1.0) - vfrac_1
    zeros = np.zeros(shape, dtype=np.float32)

    mat1 = Material(
        materialID=1,
        Vfrac=vfrac_1,
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [1e-4, 2e-4, 1e-4, 2e-4]},
        name="block",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=vfrac_2,
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="matrix",
    )
    config = {
        "CaseType": 0,
        "MorphologyType": 0,
        "Energies": energies,
        "EAngleRotation": [0.0, 0.0, 0.0],
        "RotMask": 0,
        "WindowingType": 0,
        "AlgorithmType": 0,
        "ReferenceFrame": 1,
        "EwaldsInterpolation": 1,
    }
    return Morphology(
        2,
        materials={1: mat1, 2: mat2},
        PhysSize=5.0,
        config=config,
        backend="cupy-rsoxs",
        backend_options=backend_options,
        resident_mode="host",
        input_policy="strict",
        create_cy_object=True,
    )


@pytest.mark.backend_agnostic_contract
def test_explicit_isotropic_contract_accepts_missing_orientation_arrays():
    """Ensure enum-backed isotropic materials validate without concrete S/theta/psi arrays."""
    morph = _build_two_material_isotropic_block_morphology(
        backend="cupy-rsoxs",
        resident_mode="host",
        field_namespace="numpy",
        isotropic_representation="enum_contract",
    )

    morph.check_materials(quiet=True)
    assert morph.materials[1].S is SFieldMode.ISOTROPIC
    assert morph.materials[1].theta is None
    assert morph.materials[1].psi is None
    assert morph.materials[1]._explicit_isotropic_contract is True


@pytest.mark.backend_agnostic_contract
def test_explicit_isotropic_contract_ignores_theta_and_psi_with_warning():
    """Ensure theta/psi are ignored with warning under the explicit isotropic contract."""
    with pytest.warns(UserWarning, match="SFieldMode\\.ISOTROPIC") as caught:
        morph = _build_two_material_isotropic_block_morphology(
            backend="cupy-rsoxs",
            resident_mode="host",
            field_namespace="numpy",
            isotropic_representation="enum_contract",
            ignored_orientation_arrays=True,
        )

    messages = [str(item.message) for item in caught]
    assert len(messages) == 4
    assert any("theta is ignored" in message for message in messages)
    assert any("psi is ignored" in message for message in messages)
    assert morph.materials[1].theta is None
    assert morph.materials[1].psi is None
    assert morph.materials[2].theta is None
    assert morph.materials[2].psi is None
    morph.check_materials(quiet=True)


@pytest.mark.backend_agnostic_contract
def test_write_to_file_materializes_effective_zero_orientation_fields_for_explicit_isotropic_contract(
    tmp_path,
):
    """Ensure HDF5 export writes concrete zero S/theta/psi datasets for enum-backed isotropic materials."""
    morph = _build_two_material_isotropic_block_morphology(
        backend="cupy-rsoxs",
        resident_mode="host",
        field_namespace="numpy",
        isotropic_representation="enum_contract",
    )
    out_path = tmp_path / "explicit_isotropic_contract.h5"

    morph.write_to_file(str(out_path))

    with h5py.File(out_path, "r") as handle:
        for material_id in (1, 2):
            np.testing.assert_array_equal(
                handle[f"Euler_Angles/Mat_{material_id}_S"][()],
                np.zeros((4, 16, 16), dtype=np.float32),
            )
            np.testing.assert_array_equal(
                handle[f"Euler_Angles/Mat_{material_id}_Theta"][()],
                np.zeros((4, 16, 16), dtype=np.float32),
            )
            np.testing.assert_array_equal(
                handle[f"Euler_Angles/Mat_{material_id}_Psi"][()],
                np.zeros((4, 16, 16), dtype=np.float32),
            )


@pytest.mark.backend_agnostic_contract
def test_cupy_optimization_harness_parser_accepts_cuda_prewarm_mode():
    """Ensure the dev optimization harness accepts the CUDA prewarm option surface."""
    from tests.validation.dev.cupy_rsoxs_optimization.run_cupy_rsoxs_optimization_matrix import (
        build_parser,
    )

    args = build_parser().parse_args(["--cuda-prewarm", "before_prepare_inputs"])
    assert args.cuda_prewarm == "before_prepare_inputs"


@pytest.mark.backend_agnostic_contract
def test_cupy_optimization_harness_isotropic_comparison_retains_cuda_prewarm_metadata():
    """Ensure paired isotropic comparison summaries preserve the harness CUDA prewarm mode."""
    from tests.validation.dev.cupy_rsoxs_optimization.run_cupy_rsoxs_optimization_matrix import (
        _build_isotropic_representation_comparisons,
    )

    comparisons = _build_isotropic_representation_comparisons(
        {
            "core_shell_small_single_no_rotation_host_tensor_coeff_legacy_zero_array": {
                "label": "core_shell_small_single_no_rotation_host_tensor_coeff_legacy_zero_array",
                "status": "ok",
                "resident_mode": "host",
                "shape_label": "small",
                "backend_options": {"execution_path": "tensor_coeff"},
                "energies_ev": [285.0],
                "eangle_rotation": [0.0, 0.0, 0.0],
                "isotropic_representation": "legacy_zero_array",
                "cuda_prewarm_requested_mode": "before_prepare_inputs",
                "cuda_prewarm_applied_mode": "before_prepare_inputs",
                "primary_seconds": 2.0,
                "segment_seconds": {"A2": 1.0, "B": 0.5},
            },
            "core_shell_small_single_no_rotation_host_tensor_coeff_enum_contract": {
                "label": "core_shell_small_single_no_rotation_host_tensor_coeff_enum_contract",
                "status": "ok",
                "resident_mode": "host",
                "shape_label": "small",
                "backend_options": {"execution_path": "tensor_coeff"},
                "energies_ev": [285.0],
                "eangle_rotation": [0.0, 0.0, 0.0],
                "isotropic_representation": "enum_contract",
                "cuda_prewarm_requested_mode": "before_prepare_inputs",
                "cuda_prewarm_applied_mode": "before_prepare_inputs",
                "primary_seconds": 1.5,
                "segment_seconds": {"A2": 0.5, "B": 0.25},
            },
        }
    )

    comparison = comparisons["core_shell_small_single_no_rotation_host_tensor_coeff"]
    assert comparison["cuda_prewarm_mode"] == "before_prepare_inputs"
    assert comparison["cuda_prewarm_applied_mode"] == "before_prepare_inputs"


@pytest.mark.backend_agnostic_contract
def test_cyrsoxs_timing_harness_parser_accepts_cuda_prewarm_mode():
    """Ensure the dev cyrsoxs timing harness accepts the CUDA prewarm option surface."""
    from tests.validation.dev.cyrsoxs_timing.run_cyrsoxs_timing_matrix import build_parser

    args = build_parser().parse_args(["--cuda-prewarm", "before_prepare_inputs"])
    assert args.cuda_prewarm == "before_prepare_inputs"


@pytest.mark.backend_agnostic_contract
def test_cyrsoxs_timing_harness_default_case_uses_host_borrow_contract():
    """Ensure the default cyrsoxs timing case mirrors the host-style cupy dev contract."""
    from tests.validation.dev.cyrsoxs_timing.run_cyrsoxs_timing_matrix import _timing_cases

    cases = _timing_cases(
        isotropic_representations=("legacy_zero_array",),
        cuda_prewarm_mode="off",
        size_labels=("small",),
        include_triple_no_rotation=False,
        include_triple_limited=False,
        include_full_small_check=False,
        no_rotation_energy_counts=(),
        rotation_specs=(),
        energy_lists=(),
        worker_warmup_runs=0,
    )

    assert len(cases) == 1
    case = cases[0]
    assert case.label == "core_shell_small_single_no_rotation_host_cyrsoxs"
    assert case.backend == "cyrsoxs"
    assert case.resident_mode == "host"
    assert case.field_namespace == "numpy"
    assert case.input_policy == "strict"
    assert case.ownership_policy == "borrow"


@pytest.mark.backend_agnostic_contract
def test_cyrsoxs_timing_harness_isotropic_comparison_retains_cuda_prewarm_metadata():
    """Ensure paired cyrsoxs isotropic comparison summaries preserve the prewarm metadata."""
    from tests.validation.dev.cyrsoxs_timing.run_cyrsoxs_timing_matrix import (
        _build_isotropic_representation_comparisons,
    )

    comparisons = _build_isotropic_representation_comparisons(
        {
            "core_shell_small_single_no_rotation_host_cyrsoxs_legacy_zero_array": {
                "label": "core_shell_small_single_no_rotation_host_cyrsoxs_legacy_zero_array",
                "status": "ok",
                "resident_mode": "host",
                "shape_label": "small",
                "energies_ev": [285.0],
                "eangle_rotation": [0.0, 0.0, 0.0],
                "isotropic_representation": "legacy_zero_array",
                "cuda_prewarm_requested_mode": "before_prepare_inputs",
                "cuda_prewarm_applied_mode": "before_prepare_inputs",
                "primary_seconds": 2.0,
            },
            "core_shell_small_single_no_rotation_host_cyrsoxs_enum_contract": {
                "label": "core_shell_small_single_no_rotation_host_cyrsoxs_enum_contract",
                "status": "ok",
                "resident_mode": "host",
                "shape_label": "small",
                "energies_ev": [285.0],
                "eangle_rotation": [0.0, 0.0, 0.0],
                "isotropic_representation": "enum_contract",
                "cuda_prewarm_requested_mode": "before_prepare_inputs",
                "cuda_prewarm_applied_mode": "before_prepare_inputs",
                "primary_seconds": 1.5,
            },
        }
    )

    comparison = comparisons["core_shell_small_single_no_rotation_host_cyrsoxs"]
    assert comparison["cuda_prewarm_mode"] == "before_prepare_inputs"
    assert comparison["cuda_prewarm_applied_mode"] == "before_prepare_inputs"


@pytest.mark.backend_agnostic_contract
def test_primary_backend_speed_comparison_parser_accepts_plot_only():
    """Ensure the principal cross-backend comparison script supports plot-only regeneration."""
    from tests.validation.dev.core_shell_backend_performance.run_primary_backend_speed_comparison import (
        build_parser,
    )

    args = build_parser().parse_args(["--label", "demo", "--plot-only"])
    assert args.label == "demo"
    assert args.plot_only is True


@pytest.mark.backend_agnostic_contract
def test_primary_backend_speed_comparison_rows_use_matching_legacy_baselines():
    """Ensure host rows use matching legacy startup baselines and device rows appear only on pre-warm rows."""
    from tests.validation.dev.core_shell_backend_performance.run_primary_backend_speed_comparison import (
        _build_row_records,
    )

    def make_summary(legacy_cold, legacy_warm, host_cold, host_warm, device):
        timing_cases = {}
        for lane in ("small", "medium", "large"):
            for fragment in ("single_no_rotation", "single_rot_0_15_165"):
                timing_cases[f"core_shell_{lane}_{fragment}_host_cyrsoxs"] = {
                    "status": "ok",
                    "primary_seconds": legacy_cold if fragment == "single_no_rotation" else legacy_warm,
                }
                timing_cases[f"core_shell_{lane}_{fragment}_host_tensor_coeff"] = {
                    "status": "ok",
                    "primary_seconds": host_cold if fragment == "single_no_rotation" else host_warm,
                }
                timing_cases[f"core_shell_{lane}_{fragment}_device_tensor_coeff"] = {
                    "status": "ok",
                    "primary_seconds": device,
                }
        return {"label": "synthetic", "timing_cases": timing_cases}

    rows = _build_row_records(
        legacy_cold=make_summary(10.0, 20.0, 0.0, 0.0, 0.0),
        legacy_prewarm=make_summary(4.0, 8.0, 0.0, 0.0, 0.0),
        cupy_host_and_device_cold=make_summary(0.0, 0.0, 5.0, 10.0, 2.0),
        cupy_host_prewarm=make_summary(0.0, 0.0, 2.0, 4.0, 2.0),
    )

    cold_no_rotation = rows[0]
    cold_some_rotation = rows[1]
    warm_no_rotation = rows[2]
    warm_some_rotation = rows[3]

    assert cold_no_rotation["startup"] == "cold"
    assert cold_no_rotation["legacy_cyrsoxs_primary_seconds"] == 10.0
    assert cold_no_rotation["cupy_rsoxs_host_primary_seconds"] == 5.0
    assert cold_no_rotation["cupy_rsoxs_host_speedup_vs_legacy"] == 2.0
    assert cold_no_rotation["cupy_rsoxs_device_primary_seconds"] is None

    assert cold_some_rotation["legacy_cyrsoxs_primary_seconds"] == 20.0
    assert cold_some_rotation["cupy_rsoxs_host_primary_seconds"] == 10.0
    assert cold_some_rotation["cupy_rsoxs_host_speedup_vs_legacy"] == 2.0
    assert cold_some_rotation["cupy_rsoxs_device_primary_seconds"] is None

    assert warm_no_rotation["startup"] == "pre-warm"
    assert warm_no_rotation["legacy_cyrsoxs_primary_seconds"] == 4.0
    assert warm_no_rotation["cupy_rsoxs_host_primary_seconds"] == 2.0
    assert warm_no_rotation["cupy_rsoxs_host_speedup_vs_legacy"] == 2.0
    assert warm_no_rotation["cupy_rsoxs_device_primary_seconds"] == 2.0
    assert warm_no_rotation["cupy_rsoxs_device_speedup_vs_legacy_prewarm"] == 2.0

    assert warm_some_rotation["legacy_cyrsoxs_primary_seconds"] == 8.0
    assert warm_some_rotation["cupy_rsoxs_host_primary_seconds"] == 4.0
    assert warm_some_rotation["cupy_rsoxs_host_speedup_vs_legacy"] == 2.0
    assert warm_some_rotation["cupy_rsoxs_device_primary_seconds"] == 2.0
    assert warm_some_rotation["cupy_rsoxs_device_speedup_vs_legacy_prewarm"] == 4.0


@pytest.mark.backend_agnostic_contract
def test_comprehensive_backend_comparison_speedups_use_matching_legacy_baselines():
    """Ensure comprehensive comparison speedups use the expected cyrsoxs baseline for host and device lanes."""
    from tests.validation.dev.core_shell_backend_performance.run_comprehensive_backend_comparison import (
        _enrich_results_with_legacy_speedups,
    )

    rows = [
        {
            "comparison_key": "comprehensive__host__warm__cyrsoxs__no_rotation",
            "comparison_backend": "cyrsoxs",
            "comparison_residency": "host",
            "comparison_startup_mode": "warm",
            "comparison_execution_path": "cyrsoxs",
            "comparison_rotation_key": "no_rotation",
            "comparison_rotation_label": "no rotation",
            "status": "ok",
            "primary_seconds": 10.0,
        },
        {
            "comparison_key": "comprehensive__host__hot__cyrsoxs__no_rotation",
            "comparison_backend": "cyrsoxs",
            "comparison_residency": "host",
            "comparison_startup_mode": "hot",
            "comparison_execution_path": "cyrsoxs",
            "comparison_rotation_key": "no_rotation",
            "comparison_rotation_label": "no rotation",
            "status": "ok",
            "primary_seconds": 4.0,
        },
        {
            "comparison_key": "comprehensive__host__warm__tensor_coeff__no_rotation",
            "comparison_backend": "cupy-rsoxs",
            "comparison_residency": "host",
            "comparison_startup_mode": "warm",
            "comparison_execution_path": "tensor_coeff",
            "comparison_rotation_key": "no_rotation",
            "comparison_rotation_label": "no rotation",
            "status": "ok",
            "primary_seconds": 5.0,
        },
        {
            "comparison_key": "comprehensive__host__hot__direct_polarization__no_rotation",
            "comparison_backend": "cupy-rsoxs",
            "comparison_residency": "host",
            "comparison_startup_mode": "hot",
            "comparison_execution_path": "direct_polarization",
            "comparison_rotation_key": "no_rotation",
            "comparison_rotation_label": "no rotation",
            "status": "ok",
            "primary_seconds": 2.0,
        },
        {
            "comparison_key": "comprehensive__device__steady__tensor_coeff__no_rotation",
            "comparison_backend": "cupy-rsoxs",
            "comparison_residency": "device",
            "comparison_startup_mode": "steady",
            "comparison_execution_path": "tensor_coeff",
            "comparison_rotation_key": "no_rotation",
            "comparison_rotation_label": "no rotation",
            "status": "ok",
            "primary_seconds": 2.5,
        },
        {
            "comparison_key": "comprehensive__device__hot__direct_polarization__no_rotation",
            "comparison_backend": "cupy-rsoxs",
            "comparison_residency": "device",
            "comparison_startup_mode": "hot",
            "comparison_execution_path": "direct_polarization",
            "comparison_rotation_key": "no_rotation",
            "comparison_rotation_label": "no rotation",
            "status": "ok",
            "primary_seconds": 1.0,
        },
    ]

    enriched = {
        row["comparison_key"]: row for row in _enrich_results_with_legacy_speedups(rows)
    }

    host_warm = enriched["comprehensive__host__warm__tensor_coeff__no_rotation"]
    assert host_warm["comparison_cyrsoxs_baseline_key"] == "comprehensive__host__warm__cyrsoxs__no_rotation"
    assert host_warm["comparison_cyrsoxs_baseline_startup_mode"] == "warm"
    assert host_warm["comparison_cyrsoxs_primary_seconds"] == 10.0
    assert host_warm["comparison_speedup_vs_cyrsoxs"] == 2.0

    host_hot = enriched["comprehensive__host__hot__direct_polarization__no_rotation"]
    assert host_hot["comparison_cyrsoxs_baseline_key"] == "comprehensive__host__hot__cyrsoxs__no_rotation"
    assert host_hot["comparison_cyrsoxs_baseline_startup_mode"] == "hot"
    assert host_hot["comparison_cyrsoxs_primary_seconds"] == 4.0
    assert host_hot["comparison_speedup_vs_cyrsoxs"] == 2.0

    device_steady = enriched["comprehensive__device__steady__tensor_coeff__no_rotation"]
    assert device_steady["comparison_cyrsoxs_baseline_key"] == "comprehensive__host__warm__cyrsoxs__no_rotation"
    assert device_steady["comparison_cyrsoxs_baseline_startup_mode"] == "warm"
    assert device_steady["comparison_cyrsoxs_primary_seconds"] == 10.0
    assert device_steady["comparison_speedup_vs_cyrsoxs"] == 4.0

    device_hot = enriched["comprehensive__device__hot__direct_polarization__no_rotation"]
    assert device_hot["comparison_cyrsoxs_baseline_key"] == "comprehensive__host__hot__cyrsoxs__no_rotation"
    assert device_hot["comparison_cyrsoxs_baseline_startup_mode"] == "hot"
    assert device_hot["comparison_cyrsoxs_primary_seconds"] == 4.0
    assert device_hot["comparison_speedup_vs_cyrsoxs"] == 4.0


@pytest.mark.backend_agnostic_contract
def test_required_imports():
    """Verify core runtime dependencies import without forcing backend imports."""
    for name in (
        "NRSS",
        "pytest",
        "numpy",
        "scipy",
        "pandas",
        "h5py",
        "xarray",
        "PyHyperScattering",
    ):
        _import_required(name)


@pytest.mark.backend_agnostic_contract
def test_backend_registry_reports_known_backends():
    """Check backend discovery is import-safe and reports known backend ids."""
    infos = available_backends(include_unavailable=True)
    names = {info.name for info in infos}

    assert "cyrsoxs" in names
    assert "cupy-rsoxs" in names
    assert "cyrsoxs" in format_backend_availability()
    assert get_backend_info("cyrsoxs").name == "cyrsoxs"
    assert get_backend_info("cupy-rsoxs").name == "cupy-rsoxs"
    assert get_backend_info("cyrsoxs").default_dtype == "float32"
    assert get_backend_info("cyrsoxs").supported_dtypes == ("float32",)
    assert get_backend_info("cupy-rsoxs").default_resident_mode == "host"
    assert get_backend_info("cupy-rsoxs").supported_resident_modes == ("host", "device")
    assert "execution_path" in get_backend_info("cupy-rsoxs").supported_backend_options
    assert "mixed_precision_mode" in get_backend_info("cupy-rsoxs").supported_backend_options
    assert "z_collapse_mode" in get_backend_info("cupy-rsoxs").supported_backend_options


@pytest.mark.backend_agnostic_contract
def test_unknown_backend_fails_cleanly():
    """Ensure explicit unknown backend selection raises a clear error."""
    with pytest.raises(UnknownBackendError, match="Unknown NRSS backend"):
        Morphology(1, create_cy_object=False, backend="definitely-not-a-backend")


@pytest.mark.cyrsoxs_only
def test_cyrsoxs_import_available_for_legacy_backend():
    """Verify the legacy CyRSoXS backend remains importable when selected."""
    _import_cyrsoxs_required()


@pytest.mark.backend_agnostic_contract
def test_cupy_import_available_for_supported_runtime():
    """Verify CuPy is importable in the supported NRSS runtime environment."""
    _import_cupy_required()


@pytest.mark.backend_agnostic_contract
def test_default_backend_resolution_prefers_cupy_rsoxs_when_available(monkeypatch):
    """Ensure default backend resolution prefers cupy-rsoxs over cyrsoxs when both are available."""
    monkeypatch.delenv("NRSS_BACKEND", raising=False)
    cupy_info = get_backend_info("cupy-rsoxs")

    def fake_get_backend_info(name):
        if name == "cupy-rsoxs":
            return BackendInfo(
                name="cupy-rsoxs",
                available=True,
                implemented=True,
                import_target="NRSS.backends.cupy_rsoxs",
                reason=None,
                supports_cli=False,
                supports_reference_parity=True,
                supports_device_input=True,
                supports_backend_native_output=True,
                default_resident_mode=cupy_info.default_resident_mode,
                supported_resident_modes=cupy_info.supported_resident_modes,
                default_dtype=cupy_info.default_dtype,
                supported_dtypes=cupy_info.supported_dtypes,
                supported_backend_options=cupy_info.supported_backend_options,
                description="Pure-Python CuPy-native NRSS backend.",
            )
        if name == "cyrsoxs":
            return BackendInfo(
                name="cyrsoxs",
                available=True,
                implemented=True,
                import_target="CyRSoXS",
                reason=None,
                supports_cli=True,
                supports_reference_parity=True,
                supports_device_input=False,
                supports_backend_native_output=False,
                default_resident_mode="host",
                supported_resident_modes=("host",),
                default_dtype="float32",
                supported_dtypes=("float32",),
                supported_backend_options=("dtype",),
                description="Legacy CyRSoXS backend accessed through Python bindings.",
            )
        raise AssertionError(f"Unexpected backend {name!r}")

    monkeypatch.setattr(backend_registry, "get_backend_info", fake_get_backend_info)
    assert resolve_backend_name(None) == "cupy-rsoxs"


@pytest.mark.backend_specific
@pytest.mark.cpu
def test_cupy_legacy_inputdata_proxy_maps_config_style_assignments():
    """Ensure legacy inputData config-style mutations still work under cupy-rsoxs."""
    shape = (1, 4, 4)
    zeros = np.zeros(shape, dtype=np.float32)
    material = Material(
        materialID=1,
        Vfrac=np.ones(shape, dtype=np.float32),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=[285.0],
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
    )

    morph = Morphology(
        1,
        materials={1: material},
        PhysSize=5.0,
        backend="cupy-rsoxs",
        create_cy_object=False,
    )

    assert morph.inputData is not None
    assert not morph.inputData

    morph.inputData.windowingType = 1
    morph.inputData.referenceFrame = 1
    morph.inputData.interpolationType = 1
    morph.inputData.rotMask = 1
    morph.inputData.setEnergies([285.0, 286.0])
    morph.inputData.setERotationAngle(StartAngle=0.0, IncrementAngle=2.0, EndAngle=10.0)
    morph.inputData.setPhysSize(7.5)
    morph.inputData.setDimensions(shape, order="ZYX")

    assert morph.WindowingType == 1
    assert morph.ReferenceFrame == 1
    assert morph.EwaldsInterpolation == 1
    assert morph.RotMask == 1
    assert morph.Energies == [285.0, 286.0]
    assert morph.EAngleRotation == [0.0, 2.0, 10.0]
    assert morph.PhysSize == 7.5
    assert morph.NumZYX == shape
    assert morph.inputData.validate()


@pytest.mark.backend_agnostic_contract
@pytest.mark.cpu
def test_cupy_backend_array_contract_defaults_to_host_resident_numpy():
    """Check the default cupy-rsoxs authoritative contract is host-resident NumPy."""
    contract = resolve_backend_array_contract("cupy-rsoxs")
    runtime_contract = resolve_backend_runtime_contract("cupy-rsoxs")
    arr = np.asfortranarray(np.ones((1, 4, 4), dtype=np.float64))
    plan = assess_array_for_backend(
        arr,
        backend_name="cupy-rsoxs",
        field_name="Vfrac",
        material_id=1,
    )
    coerced = coerce_array_for_backend(arr, plan)
    info = inspect_array(coerced)

    assert contract["resident_mode"] == "host"
    assert contract["namespace"] == "numpy"
    assert contract["device"] == "cpu"
    assert contract["mixed_precision_mode"] is None
    assert contract["z_collapse_mode"] is None
    assert contract["options"]["execution_path"] == "direct_polarization"
    assert contract["options"]["mixed_precision_mode"] is None
    assert contract["options"]["z_collapse_mode"] is None
    assert runtime_contract["namespace"] == "cupy"
    assert runtime_contract["device"] == "gpu"
    assert runtime_contract["options"]["execution_path"] == "direct_polarization"
    assert runtime_contract["options"]["mixed_precision_mode"] is None
    assert runtime_contract["options"]["z_collapse_mode"] is None
    assert plan.target_namespace == "numpy"
    assert plan.target_device == "cpu"
    assert plan.transfer == "none"
    assert plan.requires_dtype_cast
    assert plan.requires_layout_copy
    assert info["namespace"] == "numpy"
    assert str(coerced.dtype) == "float32"
    assert info["c_contiguous"]
    assert coerced.shape == arr.shape


@pytest.mark.backend_agnostic_contract
@pytest.mark.gpu
def test_cupy_device_resident_array_contract_normalizes_numpy_inputs_to_cupy():
    """Check the device-resident cupy-rsoxs authoritative contract converts NumPy host inputs to CuPy."""
    cp = _import_cupy_required()
    try:
        contract = resolve_backend_array_contract(
            "cupy-rsoxs",
            resident_mode="device",
        )
        arr = np.asfortranarray(np.ones((1, 4, 4), dtype=np.float64))
        plan = assess_array_for_backend(
            arr,
            backend_name="cupy-rsoxs",
            field_name="Vfrac",
            material_id=1,
            resident_mode="device",
        )
        coerced = coerce_array_for_backend(arr, plan)

        assert contract["resident_mode"] == "device"
        assert contract["dtype"] == "float32"
        assert contract["mixed_precision_mode"] is None
        assert plan.target_namespace == "cupy"
        assert plan.target_device == "gpu"
        assert plan.transfer == "host_to_device"
        assert plan.requires_dtype_cast
        assert plan.requires_layout_copy
        assert str(coerced.dtype) == "float32"
        assert inspect_array(coerced)["namespace"] == "cupy"
        assert cp.asnumpy(coerced).dtype == np.float32
    finally:
        _release_cupy_memory()


@pytest.mark.backend_specific
@pytest.mark.cpu
def test_cupy_backend_rejects_legacy_dtype_option_with_migration_error():
    """Ensure cupy-rsoxs rejects the removed dtype surface with a migration-focused message."""
    with pytest.raises(BackendOptionError, match="does not expose a generic dtype option"):
        Morphology(
            1,
            backend="cupy-rsoxs",
            backend_options={"dtype": "float16"},
            create_cy_object=False,
        )

def _expected_cupy_backend_options(
    *,
    execution_path: str = "direct_polarization",
    mixed_precision_mode: str | None = None,
    z_collapse_mode: str | None = None,
) -> dict[str, str | None]:
    kernel_preload_stage = "off"
    igor_shift_backend = "nvrtc"
    if execution_path == "direct_polarization":
        kernel_preload_stage = "a1"
        igor_shift_backend = "nvcc"

    return {
        "execution_path": execution_path,
        "mixed_precision_mode": mixed_precision_mode,
        "z_collapse_mode": z_collapse_mode,
        "kernel_preload_stage": kernel_preload_stage,
        "igor_shift_backend": igor_shift_backend,
        "direct_polarization_backend": "nvrtc",
    }


@pytest.mark.backend_specific
@pytest.mark.cpu
def test_cupy_backend_accepts_mixed_precision_backend_option():
    """Ensure cupy-rsoxs normalizes the approved mixed_precision_mode surface."""
    morph = Morphology(
        1,
        backend="cupy-rsoxs",
        backend_options={"mixed_precision_mode": "reduced_morphology_bit_depth"},
        create_cy_object=False,
    )
    assert morph.backend_options == _expected_cupy_backend_options(
        mixed_precision_mode="reduced_morphology_bit_depth",
    )
    assert morph.backend_array_contract["dtype"] == "float16"
    assert morph.runtime_compute_contract["dtype"] == "float16"
    assert morph.runtime_compute_contract["runtime_compute_dtype"] == "float32"


@pytest.mark.backend_specific
@pytest.mark.cpu
def test_cupy_backend_rejects_unknown_mixed_precision_backend_option():
    """Ensure cupy-rsoxs rejects unsupported mixed_precision_mode values up front."""
    with pytest.raises(BackendOptionError, match="does not support mixed_precision_mode"):
        Morphology(
            1,
            backend="cupy-rsoxs",
            backend_options={"mixed_precision_mode": "definitely-not-a-mode"},
            create_cy_object=False,
        )


@pytest.mark.backend_specific
@pytest.mark.cpu
def test_cupy_backend_accepts_execution_path_backend_option():
    """Ensure cupy-rsoxs normalizes execution_path via backend_options."""
    morph = Morphology(
        1,
        backend="cupy-rsoxs",
        backend_options={"execution_path": "direct"},
        create_cy_object=False,
    )
    assert morph.backend_options == _expected_cupy_backend_options(
        execution_path="direct_polarization",
    )


@pytest.mark.backend_specific
@pytest.mark.cpu
def test_cupy_backend_rejects_unknown_execution_path_backend_option():
    """Ensure cupy-rsoxs rejects unsupported execution_path values up front."""
    with pytest.raises(BackendOptionError, match="does not support execution_path"):
        Morphology(
            1,
            backend="cupy-rsoxs",
            backend_options={"execution_path": "definitely-not-a-path"},
            create_cy_object=False,
        )

    with pytest.raises(BackendOptionError, match="does not support execution_path"):
        Morphology(
            1,
            backend="cupy-rsoxs",
            backend_options={"execution_path": "nt"},
            create_cy_object=False,
        )


@pytest.mark.backend_specific
@pytest.mark.cpu
def test_cupy_backend_accepts_z_collapse_backend_option():
    """Ensure cupy-rsoxs normalizes the expert-only z_collapse_mode surface."""
    morph = Morphology(
        1,
        backend="cupy-rsoxs",
        backend_options={"z_collapse_mode": "mean"},
        create_cy_object=False,
    )
    assert morph.backend_options == _expected_cupy_backend_options(
        z_collapse_mode="mean",
    )
    assert morph.z_collapse_mode == "mean"


@pytest.mark.backend_specific
@pytest.mark.cpu
def test_cupy_backend_accepts_z_collapse_aliases_and_orthogonal_options():
    """Ensure z_collapse_mode stays orthogonal to execution_path."""
    morph = Morphology(
        1,
        backend="cupy-rsoxs",
        backend_options={
            "execution_path": "direct",
            "z_collapse_mode": "off",
        },
        create_cy_object=False,
    )
    assert morph.backend_options == _expected_cupy_backend_options(
        execution_path="direct_polarization",
    )


@pytest.mark.backend_specific
@pytest.mark.cpu
def test_cupy_backend_rejects_unknown_z_collapse_backend_option():
    """Ensure cupy-rsoxs rejects unsupported z_collapse_mode values up front."""
    with pytest.raises(BackendOptionError, match="does not support z_collapse_mode"):
        Morphology(
            1,
            backend="cupy-rsoxs",
            backend_options={"z_collapse_mode": "definitely-not-a-mode"},
            create_cy_object=False,
        )


@pytest.mark.backend_specific
@pytest.mark.cpu
def test_cupy_backend_rejects_z_collapse_with_mixed_precision_mode():
    """Ensure z_collapse_mode is mutually exclusive with the half-input mixed-precision path for now."""
    with pytest.raises(
        BackendOptionError,
        match="does not yet support combining z_collapse_mode with mixed_precision_mode",
    ):
        Morphology(
            1,
            backend="cupy-rsoxs",
            backend_options={
                "z_collapse_mode": "mean",
                "mixed_precision_mode": "reduced_morphology_bit_depth",
            },
            create_cy_object=False,
        )


@pytest.mark.gpu
def test_cupy_tensor_coeff_z_collapse_is_exact_identity_for_native_single_slice():
    """Ensure z_collapse_mode='mean' is a no-op for native z=1 tensor-coeff runs."""
    base = None
    collapsed = None
    try:
        base = _build_two_material_isotropic_single_slice_morphology(
            backend_options={"execution_path": "tensor_coeff"},
        )
        collapsed = _build_two_material_isotropic_single_slice_morphology(
            backend_options={"execution_path": "tensor_coeff", "z_collapse_mode": "mean"},
        )

        base_result = base.run(stdout=False, stderr=False, return_xarray=True)
        collapsed_result = collapsed.run(stdout=False, stderr=False, return_xarray=True)

        np.testing.assert_array_equal(base_result.values, collapsed_result.values)
        np.testing.assert_array_equal(base_result.qx.values, collapsed_result.qx.values)
        np.testing.assert_array_equal(base_result.qy.values, collapsed_result.qy.values)
    finally:
        for morph in (base, collapsed):
            if morph is None:
                continue
            try:
                morph.release_runtime()
            except Exception:
                pass
        _release_cupy_memory()


@pytest.mark.backend_specific
@pytest.mark.cpu
def test_morphology_normalizes_material_arrays_eagerly_for_selected_backend_default_residence(nrss_backend):
    """Ensure Morphology coerces arrays to the selected authoritative contract at construction time."""
    shape = (1, 4, 4)
    vfrac_1 = np.asfortranarray(np.ones(shape, dtype=np.float64))
    vfrac_2 = np.asfortranarray(np.zeros(shape, dtype=np.float64))
    zeros = np.asfortranarray(np.zeros(shape, dtype=np.float64))

    mat1 = Material(
        materialID=1,
        Vfrac=vfrac_1,
        S=zeros.copy(order="F"),
        theta=zeros.copy(order="F"),
        psi=zeros.copy(order="F"),
        energies=[285.0],
        opt_constants={285.0: [1e-4, 2e-4, 1e-4, 2e-4]},
        name="mat1",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=vfrac_2,
        S=zeros.copy(order="F"),
        theta=zeros.copy(order="F"),
        psi=zeros.copy(order="F"),
        energies=[285.0],
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="mat2",
    )

    morph = Morphology(
        2,
        materials={1: mat1, 2: mat2},
        PhysSize=5.0,
        backend=nrss_backend,
        create_cy_object=False,
    )

    expected_namespace = "numpy"
    expected_dtype = morph.backend_dtype
    expected_backend_options = {}
    if nrss_backend == "cupy-rsoxs":
        expected_backend_options = _expected_cupy_backend_options()
    else:
        expected_backend_options = {"dtype": expected_dtype}
    assert morph.resident_mode == get_backend_info(nrss_backend).default_resident_mode
    assert morph.backend_options == expected_backend_options
    assert morph.backend_array_contract["resident_mode"] == morph.resident_mode
    assert morph.backend_array_contract["dtype"] == expected_dtype
    assert morph.runtime_compute_contract["dtype"] == expected_dtype
    for material in morph.materials.values():
        for field_name in ("Vfrac", "S", "theta", "psi"):
            arr = getattr(material, field_name)
            info = inspect_array(arr)
            assert info["namespace"] == expected_namespace
            assert str(arr.dtype) == expected_dtype
            assert info["c_contiguous"]

    assert morph.construction_backend_coercion_report
    assert any(
        plan.requires_dtype_cast or plan.requires_layout_copy or plan.transfer != "none"
        for plan in morph.construction_backend_coercion_report
        if plan.original_namespace != "missing"
    )
    assert all(
        plan.target_namespace == expected_namespace
        for plan in morph.construction_backend_coercion_report
        if plan.original_namespace != "missing"
    )
    assert all(
        plan.target_namespace == expected_namespace
        for plan in morph.input_compatibility_report
        if plan.original_namespace != "missing"
    )


@pytest.mark.gpu
def test_cupy_direct_polarization_z_collapse_is_exact_identity_for_native_single_slice():
    """Ensure z_collapse_mode='mean' is a no-op for native z=1 direct-polarization runs."""
    base = None
    collapsed = None
    try:
        base = _build_two_material_isotropic_single_slice_morphology(
            backend_options={"execution_path": "direct_polarization"},
        )
        collapsed = _build_two_material_isotropic_single_slice_morphology(
            backend_options={
                "execution_path": "direct_polarization",
                "z_collapse_mode": "mean",
            },
        )

        base_result = base.run(stdout=False, stderr=False, return_xarray=True)
        collapsed_result = collapsed.run(stdout=False, stderr=False, return_xarray=True)

        np.testing.assert_array_equal(base_result.values, collapsed_result.values)
        np.testing.assert_array_equal(base_result.qx.values, collapsed_result.qx.values)
        np.testing.assert_array_equal(base_result.qy.values, collapsed_result.qy.values)
    finally:
        for morph in (base, collapsed):
            if morph is None:
                continue
            try:
                morph.release_runtime()
            except Exception:
                pass
        _release_cupy_memory()


@pytest.mark.gpu
def test_cupy_device_resident_morphology_normalizes_material_arrays_to_cupy():
    """Ensure resident_mode='device' keeps authoritative morphology fields on the GPU."""
    cp = _import_cupy_required()
    try:
        shape = (1, 4, 4)
        vfrac_1 = np.asfortranarray(np.ones(shape, dtype=np.float64))
        vfrac_2 = np.asfortranarray(np.zeros(shape, dtype=np.float64))
        zeros = np.asfortranarray(np.zeros(shape, dtype=np.float64))

        mat1 = Material(
            materialID=1,
            Vfrac=vfrac_1,
            S=zeros.copy(order="F"),
            theta=zeros.copy(order="F"),
            psi=zeros.copy(order="F"),
            energies=[285.0],
            opt_constants={285.0: [1e-4, 2e-4, 1e-4, 2e-4]},
            name="mat1",
        )
        mat2 = Material(
            materialID=2,
            Vfrac=vfrac_2,
            S=zeros.copy(order="F"),
            theta=zeros.copy(order="F"),
            psi=zeros.copy(order="F"),
            energies=[285.0],
            opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
            name="mat2",
        )

        morph = Morphology(
            2,
            materials={1: mat1, 2: mat2},
            PhysSize=5.0,
            backend="cupy-rsoxs",
            resident_mode="device",
            create_cy_object=False,
        )

        assert morph.resident_mode == "device"
        assert morph.backend_array_contract["namespace"] == "cupy"
        assert morph.runtime_compute_contract["namespace"] == "cupy"
        for material in morph.materials.values():
            for field_name in ("Vfrac", "S", "theta", "psi"):
                arr = getattr(material, field_name)
                info = inspect_array(arr)
                assert info["namespace"] == "cupy"
                assert str(arr.dtype) == "float32"
                assert info["c_contiguous"]
    finally:
        _release_cupy_memory()


@pytest.mark.backend_specific
@pytest.mark.cpu
def test_cupy_mixed_precision_mode_overrides_coerce_and_requires_strict_host_float16_inputs():
    """Ensure mixed_precision_mode behaves as strict and rejects host inputs that would need coercion."""
    shape = (1, 4, 4)
    vfrac_1 = np.ones(shape, dtype=np.float32)
    vfrac_2 = np.zeros(shape, dtype=np.float32)
    zeros = np.zeros(shape, dtype=np.float32)

    mat1 = Material(
        materialID=1,
        Vfrac=vfrac_1,
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=[285.0],
        opt_constants={285.0: [1e-4, 2e-4, 1e-4, 2e-4]},
        name="mat1",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=vfrac_2,
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=[285.0],
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="mat2",
    )

    with pytest.raises(TypeError, match="mixed_precision_mode"):
        Morphology(
            2,
            materials={1: mat1, 2: mat2},
            PhysSize=5.0,
            backend="cupy-rsoxs",
            backend_options={"mixed_precision_mode": "reduced_morphology_bit_depth"},
            input_policy="coerce",
            create_cy_object=False,
        )


@pytest.mark.backend_specific
@pytest.mark.cpu
def test_cupy_mixed_precision_host_resident_accepts_authoritative_numpy_float16_inputs():
    """Ensure host-resident mixed_precision_mode accepts strict authoritative NumPy float16 arrays."""
    morph = _build_two_material_isotropic_block_morphology(
        backend="cupy-rsoxs",
        backend_options={"mixed_precision_mode": "reduced_morphology_bit_depth"},
        resident_mode="host",
        field_namespace="numpy",
        isotropic_representation="legacy_zero_array",
        array_dtype=np.float16,
    )

    assert morph.backend_dtype == "float16"
    assert morph.runtime_dtype == "float16"
    for material in morph.materials.values():
        for field_name in ("Vfrac", "S", "theta", "psi"):
            arr = getattr(material, field_name)
            assert str(arr.dtype) == "float16"
            assert inspect_array(arr)["namespace"] == "numpy"
    morph.check_materials(quiet=True)


@pytest.mark.backend_specific
@pytest.mark.cpu
def test_cupy_mixed_precision_closure_budget_is_voxelwise_abs_1e_3():
    """Ensure mixed_precision_mode applies the expert-mode voxelwise closure budget."""
    shape = (1, 2, 2)
    ones = np.ones(shape, dtype=np.float16)
    zeros = np.zeros(shape, dtype=np.float16)

    def _build(delta: float) -> Morphology:
        vfrac_1 = ones.copy()
        vfrac_2 = zeros.copy()
        vfrac_2[0, 0, 0] = np.float16(delta)
        mat1 = Material(
            materialID=1,
            Vfrac=vfrac_1,
            S=zeros.copy(),
            theta=zeros.copy(),
            psi=zeros.copy(),
            energies=[285.0],
            opt_constants={285.0: [1e-4, 2e-4, 1e-4, 2e-4]},
            name="mat1",
        )
        mat2 = Material(
            materialID=2,
            Vfrac=vfrac_2,
            S=zeros.copy(),
            theta=zeros.copy(),
            psi=zeros.copy(),
            energies=[285.0],
            opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
            name="mat2",
        )
        return Morphology(
            2,
            materials={1: mat1, 2: mat2},
            PhysSize=5.0,
            backend="cupy-rsoxs",
            backend_options={"mixed_precision_mode": "reduced_morphology_bit_depth"},
            input_policy="strict",
            create_cy_object=False,
        )

    morph_ok = _build(9.5e-4)
    morph_ok.check_materials(quiet=True)

    morph_bad = _build(1.5e-3)
    with pytest.raises(AssertionError, match="voxelwise closure budget"):
        morph_bad.check_materials(quiet=True)


@pytest.mark.gpu
def test_cupy_mixed_precision_device_resident_accepts_authoritative_cupy_float16_inputs():
    """Ensure device-resident mixed_precision_mode accepts strict authoritative CuPy float16 arrays."""
    morph = None
    try:
        morph = _build_two_material_isotropic_block_morphology(
            backend="cupy-rsoxs",
            backend_options={"mixed_precision_mode": "reduced_morphology_bit_depth"},
            resident_mode="device",
            field_namespace="cupy",
            isotropic_representation="legacy_zero_array",
            array_dtype=np.float16,
        )

        assert morph.backend_dtype == "float16"
        assert morph.runtime_dtype == "float16"
        for material in morph.materials.values():
            for field_name in ("Vfrac", "S", "theta", "psi"):
                arr = getattr(material, field_name)
                assert str(arr.dtype) == "float16"
                assert inspect_array(arr)["namespace"] == "cupy"
        morph.check_materials(quiet=True)
    finally:
        if morph is not None:
            try:
                morph.release_runtime()
            except Exception:
                pass
        _release_cupy_memory()


@pytest.mark.backend_specific
@pytest.mark.cpu
def test_morphology_strict_input_policy_rejects_required_backend_coercion(nrss_backend):
    """Ensure input_policy='strict' fails early when backend normalization would copy or cast."""
    shape = (1, 4, 4)
    vfrac_1 = np.asfortranarray(np.ones(shape, dtype=np.float64))
    vfrac_2 = np.asfortranarray(np.zeros(shape, dtype=np.float64))
    zeros = np.asfortranarray(np.zeros(shape, dtype=np.float64))

    mat1 = Material(
        materialID=1,
        Vfrac=vfrac_1,
        S=zeros.copy(order="F"),
        theta=zeros.copy(order="F"),
        psi=zeros.copy(order="F"),
        energies=[285.0],
        opt_constants={285.0: [1e-4, 2e-4, 1e-4, 2e-4]},
        name="mat1",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=vfrac_2,
        S=zeros.copy(order="F"),
        theta=zeros.copy(order="F"),
        psi=zeros.copy(order="F"),
        energies=[285.0],
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="mat2",
    )

    with pytest.raises(TypeError, match="input_policy='strict'"):
        Morphology(
            2,
            materials={1: mat1, 2: mat2},
            PhysSize=5.0,
            backend=nrss_backend,
            input_policy="strict",
            create_cy_object=False,
        )


@pytest.mark.gpu
def test_cupy_host_resident_strict_rejects_cupy_authoritative_inputs():
    """Ensure the default host-resident strict contract rejects CuPy authoritative inputs."""
    cp = _import_cupy_required()
    try:
        shape = (1, 4, 4)
        zeros = cp.zeros(shape, dtype=cp.float32)

        mat1 = Material(
            materialID=1,
            Vfrac=cp.ones(shape, dtype=cp.float32),
            S=zeros.copy(),
            theta=zeros.copy(),
            psi=zeros.copy(),
            energies=[285.0],
            opt_constants={285.0: [1e-4, 2e-4, 1e-4, 2e-4]},
            name="mat1",
        )
        mat2 = Material(
            materialID=2,
            Vfrac=zeros.copy(),
            S=zeros.copy(),
            theta=zeros.copy(),
            psi=zeros.copy(),
            energies=[285.0],
            opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
            name="mat2",
        )

        with pytest.raises(TypeError, match="resident_mode='host'"):
            Morphology(
                2,
                materials={1: mat1, 2: mat2},
                PhysSize=5.0,
                backend="cupy-rsoxs",
                input_policy="strict",
                create_cy_object=False,
            )
    finally:
        _release_cupy_memory()


@pytest.mark.backend_specific
@pytest.mark.cpu
def test_cyrsoxs_backend_rejects_non_default_dtype_option():
    """Ensure the legacy cyrsoxs backend rejects unsupported dtype options up front."""
    with pytest.raises(BackendOptionError, match="does not support dtype"):
        Morphology(
            1,
            backend="cyrsoxs",
            backend_options={"dtype": "float16"},
            create_cy_object=False,
        )


@pytest.mark.backend_specific
@pytest.mark.cpu
def test_morphology_construction_rejects_unrecognized_array_types(nrss_backend):
    """Ensure unsupported array types fail cleanly during Morphology construction."""
    shape = (1, 4, 4)
    zeros = np.zeros(shape, dtype=np.float32)

    mat1 = Material(
        materialID=1,
        Vfrac=object(),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=[285.0],
        opt_constants={285.0: [1e-4, 2e-4, 1e-4, 2e-4]},
        name="mat1",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=zeros.copy(),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=[285.0],
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="mat2",
    )

    with pytest.raises(TypeError, match="Unsupported array type"):
        Morphology(
            2,
            materials={1: mat1, 2: mat2},
            PhysSize=5.0,
            backend=nrss_backend,
            create_cy_object=False,
        )


@pytest.mark.gpu
@pytest.mark.cyrsoxs_only
def test_cyrsoxs_morphology_normalizes_cupy_inputs_to_host_contract():
    """Ensure CuPy material fields are normalized to the legacy NumPy host contract for cyrsoxs."""
    cp = _import_cupy_required()
    try:
        energies = [285.0]
        shape = (1, 8, 8)
        zeros = cp.asfortranarray(cp.zeros(shape, dtype=cp.float64))

        mat1 = Material(
            materialID=1,
            Vfrac=cp.asfortranarray(cp.ones(shape, dtype=cp.float64)),
            S=zeros.copy(),
            theta=zeros.copy(),
            psi=zeros.copy(),
            energies=energies,
            opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
            name="vacuum_1",
        )
        mat2 = Material(
            materialID=2,
            Vfrac=cp.asfortranarray(cp.zeros(shape, dtype=cp.float64)),
            S=zeros.copy(),
            theta=zeros.copy(),
            psi=zeros.copy(),
            energies=energies,
            opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
            name="vacuum_2",
        )

        config = {
            "CaseType": 0,
            "MorphologyType": 0,
            "Energies": energies,
            "EAngleRotation": [0.0, 0.0, 0.0],
            "RotMask": 1,
            "WindowingType": 0,
            "AlgorithmType": 0,
            "ReferenceFrame": 1,
            "EwaldsInterpolation": 1,
        }

        morph = Morphology(
            2,
            materials={1: mat1, 2: mat2},
            PhysSize=5.0,
            config=config,
            backend="cyrsoxs",
            create_cy_object=True,
        )

        for material in morph.materials.values():
            for field_name in ("Vfrac", "S", "theta", "psi"):
                arr = getattr(material, field_name)
                info = inspect_array(arr)
                assert info["namespace"] == "numpy"
                assert str(arr.dtype) == "float32"
                assert info["c_contiguous"]

        assert any(
            plan.transfer == "device_to_host"
            for plan in morph.construction_backend_coercion_report
            if plan.original_namespace == "cupy"
        )
        morph.check_materials(quiet=True)
        morph.validate_all(quiet=True)
        assert morph.inputData is not None
        assert morph.OpticalConstants is not None
        assert morph.voxelData is not None
    finally:
        _release_cupy_memory()


@pytest.mark.gpu
def test_cupy_host_resident_runtime_stages_authoritative_numpy_fields_to_device():
    """Ensure default host-resident cupy-rsoxs stages temporary CuPy arrays at runtime."""
    cp = _import_cupy_required()
    shape = (2, 4, 4)
    energies = [285.0]
    zeros = np.zeros(shape, dtype=np.float32)

    mat1 = Material(
        materialID=1,
        Vfrac=np.ones(shape, dtype=np.float32),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="vacuum_1",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=zeros.copy(),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="vacuum_2",
    )
    config = {
        "CaseType": 0,
        "MorphologyType": 0,
        "Energies": energies,
        "EAngleRotation": [0.0, 0.0, 0.0],
        "RotMask": 0,
        "WindowingType": 0,
        "AlgorithmType": 0,
        "ReferenceFrame": 1,
        "EwaldsInterpolation": 1,
    }

    morph = None
    try:
        morph = Morphology(
            2,
            materials={1: mat1, 2: mat2},
            PhysSize=5.0,
            config=config,
            backend="cupy-rsoxs",
            input_policy="strict",
            create_cy_object=True,
        )
        assert morph.resident_mode == "host"
        assert morph.backend_array_contract["namespace"] == "numpy"
        result = morph.run(stdout=False, stderr=False, return_xarray=False)
        cp.cuda.Stream.null.synchronize()
        assert list(result.to_backend_array().shape) == [1, 4, 4]
        assert morph.last_runtime_staging_report
        assert all(
            plan.target_namespace == "cupy"
            for plan in morph.last_runtime_staging_report
            if plan.original_namespace != "missing"
        )
        assert any(
            plan.transfer == "host_to_device"
            for plan in morph.last_runtime_staging_report
            if plan.original_namespace == "numpy"
        )
    finally:
        if morph is not None:
            try:
                morph.release_runtime()
            except Exception:
                pass
        _release_cupy_memory()


@pytest.mark.gpu
def test_cupy_mixed_precision_host_runtime_stages_numpy_float16_fields_to_device_without_widening():
    """Ensure host-resident mixed mode stages authoritative NumPy float16 fields to CuPy float16."""
    cp = _import_cupy_required()
    morph = None
    try:
        morph = _build_two_material_isotropic_block_morphology(
            backend="cupy-rsoxs",
            backend_options={"mixed_precision_mode": "reduced_morphology_bit_depth"},
            resident_mode="host",
            field_namespace="numpy",
            isotropic_representation="legacy_zero_array",
            array_dtype=np.float16,
        )
        result = morph.run(stdout=False, stderr=False, return_xarray=False)
        cp.cuda.Stream.null.synchronize()
        assert list(result.to_backend_array().shape) == [1, 16, 16]
        assert morph.last_runtime_staging_report
        assert all(
            plan.target_namespace == "cupy" and plan.target_dtype == "float16"
            for plan in morph.last_runtime_staging_report
            if plan.original_namespace != "missing"
        )
        assert any(
            plan.transfer == "host_to_device"
            for plan in morph.last_runtime_staging_report
            if plan.original_namespace == "numpy"
        )
    finally:
        if morph is not None:
            try:
                morph.release_runtime()
            except Exception:
                pass
        _release_cupy_memory()


@pytest.mark.gpu
def test_cupy_host_resident_runtime_skips_orientation_staging_for_explicit_isotropic_contract_materials():
    """Ensure enum-backed isotropic materials stage only Vfrac into the runtime CuPy contract."""
    cp = _import_cupy_required()
    morph = None
    try:
        morph = _build_two_material_isotropic_block_morphology(
            backend="cupy-rsoxs",
            resident_mode="host",
            field_namespace="numpy",
            isotropic_representation="enum_contract",
        )
        result = morph.run(stdout=False, stderr=False, return_xarray=False)
        cp.cuda.Stream.null.synchronize()
        assert list(result.to_backend_array().shape) == [1, 16, 16]
        staged_fields = sorted(
            (plan.material_id, plan.field_name)
            for plan in morph.last_runtime_staging_report
            if plan.original_namespace != "missing"
        )
        assert staged_fields == [(1, "Vfrac"), (2, "Vfrac")]
        assert all(
            plan.transfer == "host_to_device"
            for plan in morph.last_runtime_staging_report
            if plan.original_namespace != "missing"
        )
    finally:
        if morph is not None:
            try:
                morph.release_runtime()
            except Exception:
                pass
        _release_cupy_memory()


@pytest.mark.gpu
def test_cupy_device_resident_runtime_skips_orientation_staging_for_explicit_isotropic_contract_materials():
    """Ensure enum-backed isotropic materials also skip orientation staging in device-resident mode."""
    cp = _import_cupy_required()
    morph = None
    try:
        morph = _build_two_material_isotropic_block_morphology(
            backend="cupy-rsoxs",
            resident_mode="device",
            field_namespace="cupy",
            isotropic_representation="enum_contract",
        )
        result = morph.run(stdout=False, stderr=False, return_xarray=False)
        cp.cuda.Stream.null.synchronize()
        assert list(result.to_backend_array().shape) == [1, 16, 16]
        staged_fields = sorted(
            (plan.material_id, plan.field_name)
            for plan in morph.last_runtime_staging_report
            if plan.original_namespace != "missing"
        )
        assert staged_fields == [(1, "Vfrac"), (2, "Vfrac")]
        assert all(
            plan.transfer == "none"
            for plan in morph.last_runtime_staging_report
            if plan.original_namespace != "missing"
        )
    finally:
        if morph is not None:
            try:
                morph.release_runtime()
            except Exception:
                pass
        _release_cupy_memory()


@pytest.mark.gpu
def test_cupy_host_resident_direct_polarization_shortcuts_legacy_zero_array_fields():
    """Ensure host-resident direct_polarization legacy-zero fields now stage only Vfrac."""
    cp = _import_cupy_required()
    morph = None
    try:
        morph = _build_two_material_isotropic_block_morphology(
            backend="cupy-rsoxs",
            backend_options={"execution_path": "direct_polarization"},
            resident_mode="host",
            field_namespace="numpy",
            isotropic_representation="legacy_zero_array",
        )
        result = morph.run(stdout=False, stderr=False, return_xarray=False)
        cp.cuda.Stream.null.synchronize()
        assert list(result.to_backend_array().shape) == [1, 16, 16]
        staged_fields = sorted(
            (plan.material_id, plan.field_name)
            for plan in morph.last_runtime_staging_report
            if plan.original_namespace != "missing"
        )
        assert staged_fields == [
            (1, "Vfrac"),
            (2, "Vfrac"),
        ]
    finally:
        if morph is not None:
            try:
                morph.release_runtime()
            except Exception:
                pass
        _release_cupy_memory()


@pytest.mark.gpu
def test_cupy_host_resident_tensor_coeff_shortcuts_legacy_zero_array_fields():
    """Ensure host-resident tensor_coeff legacy-zero fields now stage only Vfrac."""
    cp = _import_cupy_required()
    morph = None
    try:
        morph = _build_two_material_isotropic_block_morphology(
            backend="cupy-rsoxs",
            backend_options={"execution_path": "tensor_coeff"},
            resident_mode="host",
            field_namespace="numpy",
            isotropic_representation="legacy_zero_array",
        )
        result = morph.run(stdout=False, stderr=False, return_xarray=False)
        cp.cuda.Stream.null.synchronize()
        assert list(result.to_backend_array().shape) == [1, 16, 16]
        staged_fields = sorted(
            (plan.material_id, plan.field_name)
            for plan in morph.last_runtime_staging_report
            if plan.original_namespace != "missing"
        )
        assert staged_fields == [
            (1, "Vfrac"),
            (2, "Vfrac"),
        ]
    finally:
        if morph is not None:
            try:
                morph.release_runtime()
            except Exception:
                pass
        _release_cupy_memory()


@pytest.mark.gpu
def test_cupy_device_resident_tensor_coeff_keeps_staging_all_legacy_zero_array_fields():
    """Ensure the tensor_coeff legacy-zero shortcut remains scoped to host-resident staging."""
    cp = _import_cupy_required()
    morph = None
    try:
        morph = _build_two_material_isotropic_block_morphology(
            backend="cupy-rsoxs",
            backend_options={"execution_path": "tensor_coeff"},
            resident_mode="device",
            field_namespace="cupy",
            isotropic_representation="legacy_zero_array",
        )
        result = morph.run(stdout=False, stderr=False, return_xarray=False)
        cp.cuda.Stream.null.synchronize()
        assert list(result.to_backend_array().shape) == [1, 16, 16]
        staged_fields = sorted(
            (plan.material_id, plan.field_name)
            for plan in morph.last_runtime_staging_report
            if plan.original_namespace != "missing"
        )
        assert staged_fields == [
            (1, "S"),
            (1, "Vfrac"),
            (1, "psi"),
            (1, "theta"),
            (2, "S"),
            (2, "Vfrac"),
            (2, "psi"),
            (2, "theta"),
        ]
    finally:
        if morph is not None:
            try:
                morph.release_runtime()
            except Exception:
                pass
        _release_cupy_memory()


@pytest.mark.gpu
def test_cupy_execution_paths_and_isotropic_representations_match_on_fully_isotropic_morphology():
    """Ensure execution-path parity holds for both isotropic representations and that enum and legacy outputs match."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for isotropic execution-path comparison.")

    execution_paths = ("tensor_coeff", "direct_polarization")
    outputs = {}
    for isotropic_representation in ("legacy_zero_array", "enum_contract"):
        outputs[isotropic_representation] = {}
        for execution_path in execution_paths:
            morph = None
            try:
                morph = _build_two_material_isotropic_block_morphology(
                    backend="cupy-rsoxs",
                    backend_options={"execution_path": execution_path},
                    resident_mode="device",
                    field_namespace="cupy",
                    isotropic_representation=isotropic_representation,
                )
                outputs[isotropic_representation][execution_path] = (
                    morph.run(stdout=False, stderr=False, return_xarray=True).values.copy()
                )
            finally:
                if morph is not None:
                    try:
                        morph.release_runtime()
                    except Exception:
                        pass
                _release_cupy_memory()

    for isotropic_representation in ("legacy_zero_array", "enum_contract"):
        np.testing.assert_allclose(
            outputs[isotropic_representation]["direct_polarization"],
            outputs[isotropic_representation]["tensor_coeff"],
            rtol=0.0,
            atol=1e-6,
        )

    for execution_path in execution_paths:
        np.testing.assert_allclose(
            outputs["legacy_zero_array"][execution_path],
            outputs["enum_contract"][execution_path],
            rtol=0.0,
            atol=1e-6,
        )


@pytest.mark.gpu
def test_cupy_direct_polarization_matches_tensor_coeff_on_anisotropic_sphere():
    """Ensure direct_polarization remains numerically aligned with tensor_coeff on a small anisotropic sphere."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for anisotropic execution-path comparison.")

    for eangle_rotation in ([0.0, 0.0, 0.0], [0.0, 5.0, 165.0]):
        outputs = {}
        for execution_path in ("tensor_coeff", "direct_polarization"):
            morph = None
            try:
                morph = _build_two_material_sphere_morphology(
                    energies=[285.0],
                    eangle_rotation=eangle_rotation,
                    backend="cupy-rsoxs",
                    backend_options={"execution_path": execution_path},
                    resident_mode="host",
                    input_policy="strict",
                    ownership_policy="borrow",
                )
                outputs[execution_path] = morph.run(
                    stdout=False,
                    stderr=False,
                    return_xarray=True,
                ).values.copy()
            finally:
                if morph is not None:
                    try:
                        morph.release_runtime()
                    except Exception:
                        pass
                _release_cupy_memory()

        np.testing.assert_allclose(
            outputs["direct_polarization"],
            outputs["tensor_coeff"],
            rtol=1e-4,
            atol=5e-2,
        )


@pytest.mark.gpu
def test_cupy_z_collapse_direct_polarization_matches_tensor_coeff_on_anisotropic_sphere():
    """Ensure z-collapsed direct_polarization stays aligned with z-collapsed tensor_coeff on a small anisotropic sphere."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for anisotropic execution-path comparison.")

    for eangle_rotation in ([0.0, 0.0, 0.0], [0.0, 5.0, 165.0]):
        outputs = {}
        for execution_path in ("tensor_coeff", "direct_polarization"):
            morph = None
            try:
                morph = _build_two_material_sphere_morphology(
                    energies=[285.0],
                    eangle_rotation=eangle_rotation,
                    backend="cupy-rsoxs",
                    backend_options={
                        "execution_path": execution_path,
                        "z_collapse_mode": "mean",
                    },
                    resident_mode="host",
                    input_policy="strict",
                    ownership_policy="borrow",
                )
                outputs[execution_path] = morph.run(
                    stdout=False,
                    stderr=False,
                    return_xarray=True,
                ).values.copy()
            finally:
                if morph is not None:
                    try:
                        morph.release_runtime()
                    except Exception:
                        pass
                _release_cupy_memory()

        np.testing.assert_allclose(
            outputs["direct_polarization"],
            outputs["tensor_coeff"],
            rtol=1e-4,
            atol=5e-2,
        )


@pytest.mark.gpu
def test_cupy_mixed_precision_direct_polarization_matches_tensor_coeff_on_anisotropic_sphere():
    """Ensure mixed mode preserves execution-path parity on a small anisotropic sphere."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for mixed-precision anisotropic execution-path comparison.")

    outputs = {}
    for execution_path in ("tensor_coeff", "direct_polarization"):
        morph = None
        try:
            morph = _build_two_material_sphere_morphology(
                energies=[285.0],
                eangle_rotation=[0.0, 5.0, 165.0],
                backend="cupy-rsoxs",
                backend_options={
                    "execution_path": execution_path,
                    "mixed_precision_mode": "reduced_morphology_bit_depth",
                },
                resident_mode="host",
                input_policy="coerce",
                ownership_policy="borrow",
                field_namespace="numpy",
                array_dtype=np.float16,
            )
            outputs[execution_path] = morph.run(
                stdout=False,
                stderr=False,
                return_xarray=True,
            ).values.copy()
        finally:
            if morph is not None:
                try:
                    morph.release_runtime()
                except Exception:
                    pass
            _release_cupy_memory()

    np.testing.assert_allclose(
        outputs["direct_polarization"],
        outputs["tensor_coeff"],
        rtol=3e-3,
        atol=8e-2,
    )


@pytest.mark.gpu
def test_cupy_private_segment_timing_is_opt_in_and_subsettable():
    """Ensure cupy-rsoxs segment timing is disabled by default and records only requested segments."""
    cp = _import_cupy_required()
    shape = (2, 4, 4)
    energies = [285.0]
    zeros = cp.zeros(shape, dtype=cp.float32)

    def build_morphology():
        mat1 = Material(
            materialID=1,
            Vfrac=cp.ones(shape, dtype=cp.float32),
            S=zeros.copy(),
            theta=zeros.copy(),
            psi=zeros.copy(),
            energies=energies,
            opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
            name="vacuum_1",
        )
        mat2 = Material(
            materialID=2,
            Vfrac=zeros.copy(),
            S=zeros.copy(),
            theta=zeros.copy(),
            psi=zeros.copy(),
            energies=energies,
            opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
            name="vacuum_2",
        )
        config = {
            "CaseType": 0,
            "MorphologyType": 0,
            "Energies": energies,
            "EAngleRotation": [0.0, 0.0, 0.0],
            "RotMask": 0,
            "WindowingType": 0,
            "AlgorithmType": 0,
            "ReferenceFrame": 1,
            "EwaldsInterpolation": 1,
        }
        return Morphology(
            2,
            materials={1: mat1, 2: mat2},
            PhysSize=5.0,
            config=config,
            backend="cupy-rsoxs",
            resident_mode="device",
            input_policy="strict",
            ownership_policy="borrow",
            create_cy_object=True,
        )

    morph = None
    timed = None
    try:
        morph = build_morphology()
        result = morph.run(stdout=False, stderr=False, return_xarray=False)
        cp.cuda.Stream.null.synchronize()
        assert list(result.to_backend_array().shape) == [1, 4, 4]
        assert morph.backend_timings == {}
        assert all(
            plan.transfer == "none"
            for plan in morph.last_runtime_staging_report
            if plan.original_namespace != "missing"
        )
        morph.release_runtime()

        timed = build_morphology()
        timed._set_private_backend_timing_segments(("B", "D", "F"))
        timed_result = timed.run(stdout=False, stderr=False, return_xarray=False)
        cp.cuda.Stream.null.synchronize()
        assert list(timed_result.to_backend_array().shape) == [1, 4, 4]
        timings = timed.backend_timings
        assert timings["measurement"] == "cuda_event"
        assert timings["selected_segments"] == ["B", "D", "F"]
        assert set(timings["segment_seconds"]) == {"B", "D", "F"}
        assert all(float(value) >= 0.0 for value in timings["segment_seconds"].values())
    finally:
        if timed is not None:
            try:
                timed._clear_private_backend_timing_segments()
            except Exception:
                pass
            try:
                timed.release_runtime()
            except Exception:
                pass
        if morph is not None:
            try:
                morph.release_runtime()
            except Exception:
                pass
        _release_cupy_memory()


@pytest.mark.gpu
def test_cupy_private_segment_timing_records_runtime_staging_as_a2():
    """Ensure cupy-rsoxs records runtime morphology staging as the private A2 segment."""
    cp = _import_cupy_required()
    shape = (2, 4, 4)
    energies = [285.0]
    zeros = np.zeros(shape, dtype=np.float32)

    mat1 = Material(
        materialID=1,
        Vfrac=np.ones(shape, dtype=np.float32),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="vacuum_1",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=zeros.copy(),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="vacuum_2",
    )
    config = {
        "CaseType": 0,
        "MorphologyType": 0,
        "Energies": energies,
        "EAngleRotation": [0.0, 0.0, 0.0],
        "RotMask": 0,
        "WindowingType": 0,
        "AlgorithmType": 0,
        "ReferenceFrame": 1,
        "EwaldsInterpolation": 1,
    }

    morph = None
    try:
        morph = Morphology(
            2,
            materials={1: mat1, 2: mat2},
            PhysSize=5.0,
            config=config,
            backend="cupy-rsoxs",
            input_policy="strict",
            create_cy_object=True,
        )
        morph._set_private_backend_timing_segments(("A2", "B"))
        result = morph.run(stdout=False, stderr=False, return_xarray=False)
        cp.cuda.Stream.null.synchronize()
        assert list(result.to_backend_array().shape) == [1, 4, 4]
        timings = morph.backend_timings
        assert timings["measurement"] == "mixed"
        assert timings["selected_segments"] == ["A2", "B"]
        assert set(timings["segment_seconds"]) == {"A2", "B"}
        assert timings["segment_measurements"] == {
            "A2": "wall_clock",
            "B": "cuda_event",
        }
        assert all(float(value) >= 0.0 for value in timings["segment_seconds"].values())
    finally:
        if morph is not None:
            try:
                morph._clear_private_backend_timing_segments()
            except Exception:
                pass
            try:
                morph.release_runtime()
            except Exception:
                pass
        _release_cupy_memory()


@pytest.mark.gpu
def test_cupy_release_runtime_unlocks_mutation_and_allows_rerun():
    """Ensure release_runtime clears cupy-rsoxs transient state and permits rerun after mutation."""
    cp = _import_cupy_required()
    shape = (2, 4, 4)
    energies = [285.0]
    zeros = np.zeros(shape, dtype=np.float32)

    mat1 = Material(
        materialID=1,
        Vfrac=np.ones(shape, dtype=np.float32),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="vacuum_1",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=zeros.copy(),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="vacuum_2",
    )
    config = {
        "CaseType": 0,
        "MorphologyType": 0,
        "Energies": energies,
        "EAngleRotation": [0.0, 0.0, 0.0],
        "RotMask": 0,
        "WindowingType": 0,
        "AlgorithmType": 0,
        "ReferenceFrame": 1,
        "EwaldsInterpolation": 1,
    }

    morph = None
    try:
        morph = Morphology(
            2,
            materials={1: mat1, 2: mat2},
            PhysSize=5.0,
            config=config,
            backend="cupy-rsoxs",
            input_policy="strict",
            create_cy_object=True,
        )
        result = morph.run(stdout=False, stderr=False, return_xarray=False)
        cp.cuda.Stream.null.synchronize()

        assert list(result.to_backend_array().shape) == [1, 4, 4]
        assert morph.simulated
        assert morph._results_locked

        with pytest.raises(RuntimeError, match="Cannot mutate Morphology\\.PhysSize"):
            morph.PhysSize = 6.0
        with pytest.raises(RuntimeError, match="Cannot mutate Material\\.S"):
            morph.materials[1].S = np.full(shape, 0.25, dtype=np.float32)

        morph.release_runtime()

        assert not morph.simulated
        assert not morph._results_locked
        assert morph.scatteringPattern is None
        assert morph._backend_result is None
        assert morph.backend_timings == {}

        morph.PhysSize = 6.0
        morph.materials[1].S = np.full(shape, 0.25, dtype=np.float32)

        rerun = morph.run(stdout=False, stderr=False, return_xarray=False)
        cp.cuda.Stream.null.synchronize()

        assert list(rerun.to_backend_array().shape) == [1, 4, 4]
        assert morph.simulated
        assert morph._results_locked
    finally:
        if morph is not None:
            try:
                morph.release_runtime()
            except Exception:
                pass
        _release_cupy_memory()


@pytest.mark.backend_agnostic_contract
def test_tiny_deterministic_white_noise_kernel():
    """Check deterministic numeric fingerprint for a tiny white-noise smoothing kernel."""
    rng = np.random.default_rng(seed=20260317)
    white_noise = rng.standard_normal((16, 16))
    smoothed = _tiny_smoothing_kernel(white_noise)

    assert smoothed.shape == (16, 16)
    assert np.isfinite(smoothed).all()
    assert np.isclose(float(smoothed.mean()), 0.03638822141137503, atol=1e-12)
    assert np.isclose(float(smoothed.std()), 0.3639495657715855, atol=1e-12)
    assert np.isclose(float(smoothed[0, 0]), 0.15636713796507423, atol=1e-12)


@pytest.mark.backend_agnostic_contract
@pytest.mark.cpu
def test_small_hdf5_roundtrip(tmp_path: Path):
    """Validate basic HDF5 write/read roundtrip integrity."""
    arr = np.arange(64, dtype=np.float64).reshape(8, 8)
    test_h5 = tmp_path / "smoke_roundtrip.h5"

    with h5py.File(test_h5, "w") as h5f:
        h5f.create_dataset("arr", data=arr)

    with h5py.File(test_h5, "r") as h5f:
        loaded = h5f["arr"][()]

    assert np.array_equal(loaded, arr)


@pytest.mark.backend_agnostic_contract
@pytest.mark.cpu
def test_write_and_read_config_roundtrip(tmp_path: Path):
    """Ensure NRSS config writer/reader roundtrip preserves key fields."""
    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        write_config(
            Energies=[280.0, 281.0],
            EAngleRotation=[0.0, 0.0, 0.0],
            CaseType=0,
            MorphologyType=0,
            NumThreads=1,
            AlgorithmType=0,
            DumpMorphology=False,
            ScatterApproach=0,
            WindowingType=0,
            RotMask=False,
            EwaldsInterpolation=1,
        )
        parsed = read_config("config.txt")
    finally:
        os.chdir(cwd)

    assert parsed["CaseType"] == 0
    assert parsed["MorphologyType"] == 0
    assert parsed["NumThreads"] == 1
    assert parsed["Energies"] == [280.0, 281.0]
    assert parsed["EAngleRotation"] == [0.0, 0.0, 0.0]


@pytest.mark.cyrsoxs_only
def test_pybind_morphology_object_lifecycle_smoke():
    """Exercise pybind object creation/update/validation without full simulation workflow."""
    # Tiny deterministic morphology to exercise the pybind wiring without launching a full simulation.
    energies = [285.0]
    shape = (1, 8, 8)
    zeros = np.zeros(shape, dtype=np.float32)

    mat1 = Material(
        materialID=1,
        Vfrac=np.ones(shape, dtype=np.float32),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="vacuum_1",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=zeros.copy(),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="vacuum_2",
    )

    config = {
        "CaseType": 0,
        "MorphologyType": 0,
        "Energies": energies,
        "EAngleRotation": [0.0, 0.0, 0.0],
        "RotMask": 1,
        "WindowingType": 0,
        "AlgorithmType": 0,
        "ReferenceFrame": 1,
        "EwaldsInterpolation": 1,
    }

    morph = Morphology(
        2,
        materials={1: mat1, 2: mat2},
        PhysSize=5.0,
        config=config,
        create_cy_object=True,
    )

    morph.create_update_cy()

    assert morph.inputData is not None
    assert morph.OpticalConstants is not None
    assert morph.voxelData is not None
    assert morph.inputData.validate()
    assert morph.OpticalConstants.validate()
    assert morph.voxelData.validate()


@pytest.mark.gpu
@pytest.mark.cyrsoxs_only
def test_cyrsoxs_release_runtime_unlocks_mutation_and_rebuilds_pybind_objects():
    """Ensure release_runtime clears pybind runtime state and permits clean reruns after mutation."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for pybind lifecycle reset smoke test.")

    energies = [285.0]
    shape = (1, 8, 8)
    zeros = np.zeros(shape, dtype=np.float32)

    mat1 = Material(
        materialID=1,
        Vfrac=np.ones(shape, dtype=np.float32),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [1e-4, 2e-4, 1e-4, 2e-4]},
        name="poly",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=zeros.copy(),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="vacuum",
    )
    config = {
        "CaseType": 0,
        "MorphologyType": 0,
        "Energies": energies,
        "EAngleRotation": [0.0, 0.0, 0.0],
        "RotMask": 1,
        "WindowingType": 0,
        "AlgorithmType": 0,
        "ReferenceFrame": 1,
        "EwaldsInterpolation": 1,
    }

    morph = Morphology(
        2,
        materials={1: mat1, 2: mat2},
        PhysSize=5.0,
        config=config,
        backend="cyrsoxs",
        create_cy_object=True,
    )
    try:
        scattering = morph.run(stdout=False, stderr=False, return_xarray=True)

        assert scattering.shape == (1, 8, 8)
        assert morph.simulated
        assert morph._results_locked
        assert morph.scatteringPattern is not None
        assert morph.inputData is not None
        assert morph.OpticalConstants is not None
        assert morph.voxelData is not None

        with pytest.raises(RuntimeError, match="Cannot mutate Morphology\\.PhysSize"):
            morph.PhysSize = 6.0
        with pytest.raises(RuntimeError, match="Cannot mutate Material\\.S"):
            morph.materials[1].S = np.full(shape, 0.25, dtype=np.float32)

        morph.release_runtime()

        assert not morph.simulated
        assert not morph._results_locked
        assert morph.scatteringPattern is None
        assert morph.inputData is None
        assert morph.OpticalConstants is None
        assert morph.voxelData is None
        assert morph.backend_timings == {}

        morph.PhysSize = 6.0
        morph.materials[1].S = np.full(shape, 0.25, dtype=np.float32)

        rerun = morph.run(stdout=False, stderr=False, return_xarray=True)

        assert rerun.shape == (1, 8, 8)
        assert morph.simulated
        assert morph._results_locked
        assert morph.scatteringPattern is not None
        assert morph.inputData is not None
        assert morph.OpticalConstants is not None
        assert morph.voxelData is not None
    finally:
        morph.release_runtime()


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff", "cupy_direct_polarization")
def test_cupy_return_xarray_includes_minimal_reduction_metadata(nrss_path: ComputationPath):
    """Ensure detached cupy-rsoxs xarray results carry the minimal attrs needed for reduction."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for cupy xarray metadata smoke test.")

    morph = _build_two_material_isotropic_block_morphology(
        backend=nrss_path.backend,
        backend_options=nrss_path.backend_options,
        resident_mode=nrss_path.resident_mode,
        field_namespace=nrss_path.field_namespace,
        isotropic_representation="enum_contract",
    )
    try:
        scattering = morph.run(stdout=False, stderr=False, return_xarray=True)
        assert scattering.attrs == {"phys_size_nm": 5.0, "z_dim": 4}
        assert scattering.dims == ("energy", "qy", "qx")
    finally:
        morph.release_runtime()
        _release_cupy_memory()


@pytest.mark.gpu
@pytest.mark.cyrsoxs_only
def test_cyrsoxs_return_xarray_includes_minimal_reduction_metadata():
    """Ensure detached legacy cyrsoxs xarray results carry the minimal attrs needed for reduction."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for cyrsoxs xarray metadata smoke test.")

    energies = [285.0]
    shape = (1, 8, 8)
    zeros = np.zeros(shape, dtype=np.float32)

    mat1 = Material(
        materialID=1,
        Vfrac=np.ones(shape, dtype=np.float32),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [1e-4, 2e-4, 1e-4, 2e-4]},
        name="poly",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=zeros.copy(),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="vacuum",
    )
    config = {
        "CaseType": 0,
        "MorphologyType": 0,
        "Energies": energies,
        "EAngleRotation": [0.0, 0.0, 0.0],
        "RotMask": 1,
        "WindowingType": 0,
        "AlgorithmType": 0,
        "ReferenceFrame": 1,
        "EwaldsInterpolation": 1,
    }

    morph = Morphology(
        2,
        materials={1: mat1, 2: mat2},
        PhysSize=5.0,
        config=config,
        backend="cyrsoxs",
        create_cy_object=True,
    )
    try:
        scattering = morph.run(stdout=False, stderr=False, return_xarray=True)
        assert scattering.attrs == {"phys_size_nm": 5.0, "z_dim": 1}
        assert scattering.dims == ("energy", "qy", "qx")
    finally:
        morph.release_runtime()


@pytest.mark.cyrsoxs_only
def test_vacuum_named_matches_explicit_zero_constants():
    """Verify named vacuum optical constants match explicit all-zero constants."""
    energies = [285.0, 286.0]
    shape = (1, 4, 4)
    zeros = np.zeros(shape, dtype=np.float32)
    ones = np.ones(shape, dtype=np.float32)

    mat1 = Material(
        materialID=1,
        Vfrac=ones.copy(),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={e: [1e-4, 2e-4, 1e-4, 2e-4] for e in energies},
        name="poly",
    )
    vacuum_named = Material(
        materialID=2,
        Vfrac=zeros.copy(),
        energies=energies,
        name="vacuum",
    )
    vacuum_explicit = Material(
        materialID=2,
        Vfrac=zeros.copy(),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={e: [0.0, 0.0, 0.0, 0.0] for e in energies},
        name="vacuum_explicit",
    )

    assert vacuum_named.S is SFieldMode.ISOTROPIC
    assert vacuum_named.theta is None
    assert vacuum_named.psi is None
    assert vacuum_named.opt_constants == vacuum_explicit.opt_constants

    config = {
        "CaseType": 0,
        "MorphologyType": 0,
        "Energies": energies,
        "EAngleRotation": [0.0, 0.0, 0.0],
        "RotMask": 1,
        "WindowingType": 0,
        "AlgorithmType": 0,
        "ReferenceFrame": 1,
        "EwaldsInterpolation": 1,
    }

    morph_named = Morphology(
        2,
        materials={1: mat1.copy(), 2: vacuum_named},
        PhysSize=5.0,
        config=config,
        create_cy_object=True,
    )
    morph_explicit = Morphology(
        2,
        materials={1: mat1.copy(), 2: vacuum_explicit},
        PhysSize=5.0,
        config=config,
        create_cy_object=True,
    )

    morph_named.check_materials(quiet=True)
    morph_explicit.check_materials(quiet=True)
    morph_named.validate_all(quiet=True)
    morph_explicit.validate_all(quiet=True)

    assert morph_named.materials[2].S is SFieldMode.ISOTROPIC
    assert morph_named.materials[2].theta is None
    assert morph_named.materials[2].psi is None
    assert morph_named.materials[2]._explicit_isotropic_contract is True

    for e in energies:
        named_vals = morph_named.materials[2].opt_constants[e]
        explicit_vals = morph_explicit.materials[2].opt_constants[e]
        assert np.allclose(named_vals, explicit_vals, atol=0.0)


@pytest.mark.backend_agnostic_contract
def test_vacuum_named_ignores_supplied_orientation_and_forces_explicit_isotropic_contract():
    """Ensure named vacuum always resolves to the explicit isotropic contract and ignores orientation fields."""
    shape = (1, 4, 4)
    zeros = np.zeros(shape, dtype=np.float32)

    with pytest.warns(UserWarning, match="name='vacuum'.*SFieldMode\\.ISOTROPIC") as caught:
        vacuum = Material(
            materialID=2,
            Vfrac=zeros.copy(),
            S=zeros.copy(),
            theta=zeros.copy(),
            psi=zeros.copy(),
            energies=[285.0],
            name="vacuum",
        )

    messages = [str(item.message) for item in caught]
    assert len(messages) == 1
    assert "S, theta, and psi" in messages[0]
    assert vacuum.S is SFieldMode.ISOTROPIC
    assert vacuum.theta is None
    assert vacuum.psi is None
    assert vacuum._explicit_isotropic_contract is True


@pytest.mark.backend_agnostic_contract
@pytest.mark.cpu
def test_optical_constants_calc_and_load_matfile_smoke(tmp_path: Path):
    """Smoke-test optical constant interpolation and MaterialX.txt parsing."""
    reference_data = {
        "Energy": [100.0, 200.0],
        "DeltaPara": [1.0, 3.0],
        "BetaPara": [2.0, 4.0],
        "DeltaPerp": [5.0, 7.0],
        "BetaPerp": [6.0, 8.0],
    }
    energies = [100.0, 150.0, 200.0]
    calc = OpticalConstants.calc_constants(energies, reference_data, name="interp")

    assert np.allclose(calc.opt_constants[100.0], [1.0, 2.0, 5.0, 6.0], atol=1e-12)
    assert np.allclose(calc.opt_constants[150.0], [2.0, 3.0, 6.0, 7.0], atol=1e-12)
    assert np.allclose(calc.opt_constants[200.0], [3.0, 4.0, 7.0, 8.0], atol=1e-12)

    matfile = tmp_path / "Material1.txt"
    matfile.write_text(
        "\n".join(
            [
                "Energy = 100.0;",
                "BetaPara = 2.0;",
                "BetaPerp = 6.0;",
                "DeltaPara = 1.0;",
                "DeltaPerp = 5.0;",
                "Energy = 200.0;",
                "BetaPara = 4.0;",
                "BetaPerp = 8.0;",
                "DeltaPara = 3.0;",
                "DeltaPerp = 7.0;",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = OpticalConstants.load_matfile(str(matfile), name="from_file")
    assert loaded.energies == [100.0, 200.0]
    assert np.allclose(loaded.opt_constants[100.0], [1.0, 2.0, 5.0, 6.0], atol=1e-12)
    assert np.allclose(loaded.opt_constants[200.0], [3.0, 4.0, 7.0, 8.0], atol=1e-12)


@pytest.mark.backend_agnostic_contract
@pytest.mark.cpu
def test_morphology_validation_fails_when_total_vfrac_exceeds_one():
    """Assert morphology validator rejects voxels where total volume fraction exceeds one."""
    energies = [285.0]
    shape = (1, 4, 4)
    zeros = np.zeros(shape, dtype=np.float32)
    vfrac_1 = np.full(shape, 0.6, dtype=np.float32)
    vfrac_2 = np.full(shape, 0.6, dtype=np.float32)

    assert np.all(vfrac_1 <= 1.0)
    assert np.all(vfrac_2 <= 1.0)
    assert np.all(vfrac_1 + vfrac_2 > 1.0)

    mat1 = Material(
        materialID=1,
        Vfrac=vfrac_1,
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [1e-4, 2e-4, 1e-4, 2e-4]},
        name="mat1",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=vfrac_2,
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="mat2",
    )

    morph = Morphology(
        2,
        materials={1: mat1, 2: mat2},
        PhysSize=5.0,
        create_cy_object=False,
    )

    with pytest.raises(
        AssertionError,
        match="Total material volume fractions do not sum to 1",
    ):
        morph.check_materials(quiet=True)


@pytest.mark.backend_agnostic_contract
@pytest.mark.cpu
def test_morphology_validation_fails_on_nan_field():
    """Assert morphology validator rejects NaN values in material fields."""
    energies = [285.0]
    shape = (1, 4, 4)
    zeros = np.zeros(shape, dtype=np.float32)
    ones = np.ones(shape, dtype=np.float32)
    bad_s = zeros.copy()
    bad_s[0, 0, 0] = np.nan

    mat1 = Material(
        materialID=1,
        Vfrac=ones.copy(),
        S=bad_s,
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [1e-4, 2e-4, 1e-4, 2e-4]},
        name="mat1",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=zeros.copy(),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="mat2",
    )
    morph = Morphology(2, materials={1: mat1, 2: mat2}, PhysSize=5.0, create_cy_object=False)

    with pytest.raises(AssertionError, match="NaNs are present in Material 1 S"):
        morph.check_materials(quiet=True)


@pytest.mark.backend_agnostic_contract
@pytest.mark.cpu
def test_morphology_construction_coerces_non_float_field_to_float():
    """Assert Morphology eagerly coerces backend-supported non-float arrays to float32."""
    energies = [285.0]
    shape = (1, 4, 4)
    zeros_f = np.zeros(shape, dtype=np.float32)
    zeros_i = np.zeros(shape, dtype=np.int32)
    ones = np.ones(shape, dtype=np.float32)

    mat1 = Material(
        materialID=1,
        Vfrac=ones.copy(),
        S=zeros_f.copy(),
        theta=zeros_i,
        psi=zeros_f.copy(),
        energies=energies,
        opt_constants={285.0: [1e-4, 2e-4, 1e-4, 2e-4]},
        name="mat1",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=zeros_f.copy(),
        S=zeros_f.copy(),
        theta=zeros_f.copy(),
        psi=zeros_f.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="mat2",
    )
    morph = Morphology(2, materials={1: mat1, 2: mat2}, PhysSize=5.0, create_cy_object=False)

    assert morph.materials[1].theta.dtype == np.float32
    morph.check_materials(quiet=True)


@pytest.mark.backend_agnostic_contract
@pytest.mark.cpu
def test_morphology_validation_fails_on_negative_s():
    """Assert morphology validator rejects aligned-fraction values below zero."""
    energies = [285.0]
    shape = (1, 4, 4)
    zeros = np.zeros(shape, dtype=np.float32)
    ones = np.ones(shape, dtype=np.float32)
    bad_s = zeros.copy()
    bad_s[0, 0, 0] = -1e-3

    mat1 = Material(
        materialID=1,
        Vfrac=ones.copy(),
        S=bad_s,
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [1e-4, 2e-4, 1e-4, 2e-4]},
        name="mat1",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=zeros.copy(),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="mat2",
    )
    morph = Morphology(2, materials={1: mat1, 2: mat2}, PhysSize=5.0, create_cy_object=False)

    with pytest.raises(
        AssertionError,
        match="Material 1 S value\\(s\\) does not lie between 0 and 1",
    ):
        morph.check_materials(quiet=True)


@pytest.mark.backend_agnostic_contract
@pytest.mark.cpu
def test_morphology_validation_fails_on_s_above_one():
    """Assert morphology validator rejects aligned-fraction values above one."""
    energies = [285.0]
    shape = (1, 4, 4)
    zeros = np.zeros(shape, dtype=np.float32)
    ones = np.ones(shape, dtype=np.float32)
    bad_s = zeros.copy()
    bad_s[0, 0, 0] = 1.001

    mat1 = Material(
        materialID=1,
        Vfrac=ones.copy(),
        S=bad_s,
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [1e-4, 2e-4, 1e-4, 2e-4]},
        name="mat1",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=zeros.copy(),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="mat2",
    )
    morph = Morphology(2, materials={1: mat1, 2: mat2}, PhysSize=5.0, create_cy_object=False)

    with pytest.raises(
        AssertionError,
        match="Material 1 S value\\(s\\) does not lie between 0 and 1",
    ):
        morph.check_materials(quiet=True)


@pytest.mark.backend_agnostic_contract
@pytest.mark.cpu
def test_morphology_validation_fails_on_negative_vfrac():
    """Assert morphology validator rejects negative volume fractions."""
    energies = [285.0]
    shape = (1, 4, 4)
    zeros = np.zeros(shape, dtype=np.float32)
    ones = np.ones(shape, dtype=np.float32)
    bad_vfrac = ones.copy()
    bad_vfrac[0, 0, 0] = -1e-3
    matrix_vfrac = zeros.copy()
    matrix_vfrac[0, 0, 0] = 1.001

    mat1 = Material(
        materialID=1,
        Vfrac=bad_vfrac,
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [1e-4, 2e-4, 1e-4, 2e-4]},
        name="mat1",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=matrix_vfrac,
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="mat2",
    )
    morph = Morphology(2, materials={1: mat1, 2: mat2}, PhysSize=5.0, create_cy_object=False)

    with pytest.raises(
        AssertionError,
        match="Material 1 Vfrac value\\(s\\) does not lie between 0 and 1",
    ):
        morph.check_materials(quiet=True)


@pytest.mark.backend_agnostic_contract
@pytest.mark.cpu
def test_morphology_validation_accepts_small_closure_drift_within_allclose_defaults():
    """Pin the current closure tolerance implied by numpy allclose defaults."""
    energies = [285.0]
    shape = (1, 4, 4)
    zeros = np.zeros(shape, dtype=np.float64)
    vfrac_1 = np.full(shape, 0.500005, dtype=np.float64)
    vfrac_2 = np.full(shape, 0.5, dtype=np.float64)

    assert np.allclose(vfrac_1 + vfrac_2, 1.0)

    mat1 = Material(
        materialID=1,
        Vfrac=vfrac_1,
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [1e-4, 2e-4, 1e-4, 2e-4]},
        name="mat1",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=vfrac_2,
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="mat2",
    )
    morph = Morphology(
        2,
        materials={1: mat1, 2: mat2},
        PhysSize=5.0,
        create_cy_object=False,
    )

    morph.check_materials(quiet=True)


@pytest.mark.backend_agnostic_contract
@pytest.mark.cpu
def test_visualizer_two_material_outputs_and_summary_are_consistent(capsys, monkeypatch):
    """Check two-material visualizer output count/shape and printed summary consistency."""
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    energies = [285.0]
    shape = (1, 16, 16)
    zeros = np.zeros(shape, dtype=np.float32)

    zz, yy, xx = np.indices(shape)
    vfrac_1 = ((zz + yy + xx) % 2).astype(np.float32)
    vfrac_2 = 1.0 - vfrac_1

    mat1 = Material(
        materialID=1,
        Vfrac=vfrac_1,
        S=(0.5 * vfrac_1).astype(np.float32),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [2e-4, 1e-4, 2e-4, 1e-4]},
        name="mat1",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=vfrac_2,
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="mat2",
    )

    morph = Morphology(
        2,
        materials={1: mat1, 2: mat2},
        PhysSize=5.0,
        create_cy_object=False,
    )

    # Ensure generated morphology matches the intended two-material construction.
    assert np.allclose(morph.materials[1].Vfrac + morph.materials[2].Vfrac, 1.0)
    expected_line_1 = (
        f"Material 1 Vfrac. Min: {morph.materials[1].Vfrac.min()} "
        f"Max: {morph.materials[1].Vfrac.max()}"
    )
    expected_line_2 = (
        f"Material 2 Vfrac. Min: {morph.materials[2].Vfrac.min()} "
        f"Max: {morph.materials[2].Vfrac.max()}"
    )

    images = morph.visualize_materials(
        z_slice=0,
        subsample=16,
        outputmat=[1, 2],
        outputplot=["vfrac", "S"],
        outputaxes=True,
        runquiet=False,
        batchMode=True,
    )
    stdout = capsys.readouterr().out

    assert "Number of Materials: 2" in stdout
    assert expected_line_1 in stdout
    assert expected_line_2 in stdout
    assert len(images) == 4
    for image in images:
        assert isinstance(image, np.ndarray)
        assert image.ndim == 3
        assert image.shape[0] > 0
        assert image.shape[1] > 0
        assert image.shape[2] == 4


@lru_cache(maxsize=1)
def _has_visible_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0 and "GPU " in result.stdout


def _build_two_material_sphere_morphology(
    energies: list[float],
    shape: tuple[int, int, int] = (32, 32, 32),
    sphere_diameter_vox: int = 16,
    eangle_rotation: list[float] | None = None,
    config_overrides: dict | None = None,
    backend: str = "cyrsoxs",
    backend_options: dict | None = None,
    resident_mode: str | None = None,
    input_policy: str = "coerce",
    ownership_policy: str | None = None,
    field_namespace: str = "numpy",
    array_dtype=np.float32,
) -> Morphology:
    zz, yy, xx = np.indices(shape)
    cz = (shape[0] - 1) / 2.0
    cy = (shape[1] - 1) / 2.0
    cx = (shape[2] - 1) / 2.0
    dz = zz - cz
    dy = yy - cy
    dx = xx - cx

    radius = sphere_diameter_vox / 2.0
    sphere_mask = (dx * dx + dy * dy + dz * dz) <= (radius * radius)
    vfrac_1 = sphere_mask.astype(np.float32)
    vfrac_2 = 1.0 - vfrac_1

    theta = np.arctan2(np.sqrt(dx * dx + dy * dy), dz)
    psi = np.arctan2(dy, dx)
    theta = (theta.astype(np.float32) * vfrac_1).astype(np.float32)
    psi = (psi.astype(np.float32) * vfrac_1).astype(np.float32)
    S_1 = (0.7 * vfrac_1).astype(np.float32)
    zeros = np.zeros(shape, dtype=np.float32)

    vfrac_1 = _to_backend_namespace(vfrac_1, field_namespace, dtype=array_dtype)
    vfrac_2 = _to_backend_namespace(vfrac_2.astype(np.float32), field_namespace, dtype=array_dtype)
    theta = _to_backend_namespace(theta, field_namespace, dtype=array_dtype)
    psi = _to_backend_namespace(psi, field_namespace, dtype=array_dtype)
    S_1 = _to_backend_namespace(S_1, field_namespace, dtype=array_dtype)
    zeros = _to_backend_namespace(zeros, field_namespace, dtype=array_dtype)

    mat1 = Material(
        materialID=1,
        Vfrac=vfrac_1,
        S=S_1,
        theta=theta,
        psi=psi,
        energies=energies,
        opt_constants={e: [2e-4, 1e-4, 2e-4, 1e-4] for e in energies},
        name="sphere_material",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=vfrac_2,
        S=_clone_backend_array(zeros),
        theta=_clone_backend_array(zeros),
        psi=_clone_backend_array(zeros),
        energies=energies,
        opt_constants={e: [0.0, 0.0, 0.0, 0.0] for e in energies},
        name="matrix_vacuum",
    )

    if eangle_rotation is None:
        eangle_rotation = [0.0, 0.0, 0.0]

    config = {
        "CaseType": 0,
        "MorphologyType": 0,
        "Energies": energies,
        "EAngleRotation": eangle_rotation,
        "RotMask": 1,
        "WindowingType": 0,
        "AlgorithmType": 0,
        "ReferenceFrame": 1,
        "EwaldsInterpolation": 1,
    }
    if config_overrides:
        config.update(config_overrides)

    morph = Morphology(
        2,
        materials={1: mat1, 2: mat2},
        PhysSize=5.0,
        config=config,
        create_cy_object=True,
        backend=backend,
        backend_options=backend_options,
        resident_mode=resident_mode,
        input_policy=input_policy,
        ownership_policy=ownership_policy,
    )
    morph.check_materials(quiet=True)
    morph.validate_all(quiet=True)
    return morph


def _run_tiny_pybind_simulation(
    energies: list[float] | None = None,
    return_xarray: bool = False,
    shape: tuple[int, int, int] = (32, 32, 32),
    sphere_diameter_vox: int = 16,
    runtime_kwargs: dict | None = None,
):
    # Pin smoke runtime to one GPU for better stability on multi-GPU hosts.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    if energies is None:
        energies = [285.0]
    morph = _build_two_material_sphere_morphology(
        energies=energies,
        shape=shape,
        sphere_diameter_vox=sphere_diameter_vox,
        **(runtime_kwargs or {}),
    )
    scattering = morph.run(stdout=False, stderr=False, return_xarray=True)
    if return_xarray:
        return scattering
    return scattering.values.copy()


def _run_cli_from_serialized_morphology(morph: Morphology, run_path: Path) -> np.ndarray:
    from PyHyperScattering.load import cyrsoxsLoader

    run_path.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    os.chdir(run_path)
    try:
        morphology_fname = "morphology.hdf5"
        morph.write_to_file(morphology_fname, author="NRSS smoke test")
        morph.write_constants(path=".")

        # Ensure NRSS writes CLI optical-constant files with expected headings.
        for mat_id in range(1, morph.numMaterial + 1):
            material_text = (run_path / f"Material{mat_id}.txt").read_text(
                encoding="utf-8"
            )
            for token in (
                "EnergyData0",
                "Energy = ",
                "BetaPara = ",
                "BetaPerp = ",
                "DeltaPara = ",
                "DeltaPerp = ",
            ):
                assert token in material_text

        write_config(
            Energies=morph.Energies,
            EAngleRotation=morph.EAngleRotation,
            CaseType=int(morph.CaseType),
            MorphologyType=int(morph.MorphologyType),
            NumThreads=1,
            AlgorithmType=int(morph.AlgorithmType),
            DumpMorphology=False,
            ScatterApproach=0,
            WindowingType=int(morph.WindowingType),
            RotMask=bool(morph.RotMask),
            EwaldsInterpolation=int(morph.EwaldsInterpolation),
        )

        result = subprocess.run(
            ["CyRSoXS", morphology_fname],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"CyRSoXS CLI failed with return code {result.returncode}. "
            f"stdout tail: {' | '.join(result.stdout.splitlines()[-10:])} "
            f"stderr tail: {' | '.join(result.stderr.splitlines()[-10:])}"
        )

        loaded = cyrsoxsLoader(profile_time=False).loadDirectory(run_path)
        cli_vals = np.asarray(loaded.values).copy()
    finally:
        os.chdir(cwd)

    # cyrsoxsLoader returns (qx, qy, energy); align to pybind's (energy, qy, qx).
    cli_vals = np.moveaxis(cli_vals, -1, 0)
    cli_vals = np.swapaxes(cli_vals, 1, 2)
    return cli_vals.copy()


def _sanitize_scattering(arr: np.ndarray, clip_percentile: float = 99.9) -> np.ndarray:
    safe = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    safe = np.where(safe < 0.0, 0.0, safe)
    cap = float(np.percentile(safe, clip_percentile))
    if cap > 0.0:
        safe = np.clip(safe, 0.0, cap)
    return safe


def _assert_scattering_parity(
    pybind_vals: np.ndarray,
    cli_vals: np.ndarray,
    *,
    min_finite_ratio: float = 0.99,
    rtol_scalar: float = 1e-3,
    rtol_max: float | None = None,
    p95_abs_max: float = 8e-7,
    max_abs_max: float = 3e-5,
    p95_log_max: float = 8e-2,
    max_log_max: float = 1e-1,
) -> None:
    assert pybind_vals.shape == cli_vals.shape
    assert float(np.isfinite(pybind_vals).mean()) >= min_finite_ratio
    assert float(np.isfinite(cli_vals).mean()) >= min_finite_ratio
    if rtol_max is None:
        rtol_max = rtol_scalar

    pybind_safe = _sanitize_scattering(pybind_vals)
    cli_safe = _sanitize_scattering(cli_vals)
    abs_diff = np.abs(pybind_safe - cli_safe)
    signal_mask = np.logical_and(pybind_safe > 1e-12, cli_safe > 1e-12)

    assert signal_mask.any()
    log_pybind = np.log10(pybind_safe[signal_mask])
    log_cli = np.log10(cli_safe[signal_mask])
    log_abs = np.abs(log_pybind - log_cli)

    assert np.isclose(
        float(pybind_safe.sum()),
        float(cli_safe.sum()),
        rtol=rtol_scalar,
        atol=1e-12,
    )
    assert np.isclose(
        float(pybind_safe.max()),
        float(cli_safe.max()),
        rtol=rtol_max,
        atol=1e-12,
    )
    assert float(np.percentile(abs_diff, 95)) <= p95_abs_max
    assert float(abs_diff.max()) <= max_abs_max
    assert float(np.percentile(log_abs, 95)) <= p95_log_max
    assert float(log_abs.max()) <= max_log_max


def _assert_scattering_similarity(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    sum_rtol: float = 1e-4,
    max_rtol: float = 1e-4,
    p95_abs_max: float = 1e-6,
    max_abs_max: float = 1e-4,
) -> None:
    lhs_safe = _sanitize_scattering(lhs)
    rhs_safe = _sanitize_scattering(rhs)
    assert lhs_safe.shape == rhs_safe.shape
    assert np.isfinite(lhs_safe).all()
    assert np.isfinite(rhs_safe).all()
    assert np.isclose(float(lhs_safe.sum()), float(rhs_safe.sum()), rtol=sum_rtol, atol=1e-12)
    assert np.isclose(float(lhs_safe.max()), float(rhs_safe.max()), rtol=max_rtol, atol=1e-12)
    abs_diff = np.abs(lhs_safe - rhs_safe)
    assert float(np.percentile(abs_diff, 95)) <= p95_abs_max
    assert float(abs_diff.max()) <= max_abs_max


def _build_two_material_asymmetric_lobed_morphology(
    energies: list[float],
    eangle_rotation: list[float],
    *,
    backend: str = "cyrsoxs",
    backend_options: dict | None = None,
    resident_mode: str | None = None,
    ownership_policy: str | None = None,
    field_namespace: str = "numpy",
) -> Morphology:
    shape = (32, 32, 32)
    zz, yy, xx = np.indices(shape)
    cz = (shape[0] - 1) / 2.0
    cy = (shape[1] - 1) / 2.0
    cx = (shape[2] - 1) / 2.0
    dz = zz - cz
    dy = yy - cy
    dx = xx - cx

    # Asymmetric morphology: one side has ~8 voxel diameter, the other ~7.
    local_radius = np.where(dx <= 0.0, 4.0, 3.5)
    vfrac_1 = ((dx * dx + dy * dy + dz * dz) <= (local_radius * local_radius)).astype(
        np.float32
    )
    vfrac_2 = 1.0 - vfrac_1

    # Intentionally non-uniform orientation fields to make E-angle sampling effects visible.
    theta_raw = (np.pi / 4.0) + 0.25 * (dx / (shape[2] / 2.0)) + 0.15 * (dy / (shape[1] / 2.0))
    theta = (np.clip(theta_raw, 0.0, np.pi).astype(np.float32) * vfrac_1).astype(np.float32)
    psi = (np.arctan2(dy + 0.3, dx - 0.2).astype(np.float32) * vfrac_1).astype(np.float32)
    s_1 = ((0.55 + 0.25 * ((yy % 3) == 0).astype(np.float32)) * vfrac_1).astype(np.float32)
    zeros = np.zeros(shape, dtype=np.float32)

    vfrac_1 = _to_backend_namespace(vfrac_1, field_namespace)
    vfrac_2 = _to_backend_namespace(vfrac_2.astype(np.float32), field_namespace)
    theta = _to_backend_namespace(theta, field_namespace)
    psi = _to_backend_namespace(psi, field_namespace)
    s_1 = _to_backend_namespace(s_1, field_namespace)
    zeros = _to_backend_namespace(zeros, field_namespace)

    mat1 = Material(
        materialID=1,
        Vfrac=vfrac_1,
        S=s_1,
        theta=theta,
        psi=psi,
        energies=energies,
        opt_constants={e: [2e-4, 1e-4, 2e-4, 1e-4] for e in energies},
        name="asymmetric_lobe",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=vfrac_2,
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={e: [0.0, 0.0, 0.0, 0.0] for e in energies},
        name="vacuum",
    )

    config = {
        "CaseType": 0,
        "MorphologyType": 0,
        "Energies": energies,
        "EAngleRotation": eangle_rotation,
        "RotMask": 1,
        "WindowingType": 0,
        "AlgorithmType": 0,
        "ReferenceFrame": 1,
        "EwaldsInterpolation": 1,
    }
    morph = Morphology(
        2,
        materials={1: mat1, 2: mat2},
        PhysSize=5.0,
        config=config,
        create_cy_object=True,
        backend=backend,
        backend_options=backend_options,
        resident_mode=resident_mode,
        ownership_policy=ownership_policy,
    )
    morph.check_materials(quiet=True)
    morph.validate_all(quiet=True)
    return morph


def _radial_asymmetry_score(arr: np.ndarray) -> float:
    """Compute ring-wise azimuthal CV; lower score means more radial symmetry."""
    img = _sanitize_scattering(arr)[0].astype(np.float64)
    ny, nx = img.shape
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0
    yy, xx = np.indices((ny, nx))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    r_min = 2.0
    r_max = min(cy, cx)
    bins = np.linspace(r_min, r_max, 12)
    cvs = []
    weights = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = np.logical_and(rr >= lo, rr < hi)
        if int(mask.sum()) < 12:
            continue
        ring = img[mask]
        ring_mean = float(ring.mean())
        if ring_mean <= 1e-12:
            continue
        cvs.append(float(ring.std()) / ring_mean)
        weights.append(float(mask.sum()))

    assert cvs
    return float(np.average(np.asarray(cvs), weights=np.asarray(weights)))


@pytest.mark.gpu
@pytest.mark.path_matrix
def test_pybind_runtime_tiny_deterministic_pattern(nrss_path: ComputationPath):
    """Run a tiny GPU sphere simulation and assert deterministic scalar/log similarity."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for pybind runtime smoke test.")

    runtime_kwargs = _path_runtime_kwargs(nrss_path)
    arr_1 = _run_tiny_pybind_simulation(runtime_kwargs=runtime_kwargs)
    arr_2 = _run_tiny_pybind_simulation(runtime_kwargs=runtime_kwargs)
    finite_ratio_1 = float(np.isfinite(arr_1).mean())
    finite_ratio_2 = float(np.isfinite(arr_2).mean())
    arr_1_safe = _sanitize_scattering(arr_1)
    arr_2_safe = _sanitize_scattering(arr_2)

    assert arr_1.shape == (1, 32, 32)
    assert arr_2.shape == (1, 32, 32)
    assert finite_ratio_1 >= 0.99
    assert finite_ratio_2 >= 0.99
    # Pinned single-GPU repeat runs are bitwise-stable here; keep a small fixed margin.
    assert np.isfinite(arr_1_safe).all()
    assert np.isfinite(arr_2_safe).all()
    assert np.isclose(float(arr_1_safe.sum()), float(arr_2_safe.sum()), rtol=1e-4, atol=1e-12)
    assert np.isclose(float(arr_1_safe.max()), float(arr_2_safe.max()), rtol=1e-4, atol=1e-12)
    signal_mask = np.logical_and(arr_1_safe > 1e-12, arr_2_safe > 1e-12)
    assert signal_mask.any()
    log_1 = np.log10(arr_1_safe[signal_mask])
    log_2 = np.log10(arr_2_safe[signal_mask])
    log_abs = np.abs(log_1 - log_2)
    assert float(np.percentile(log_abs, 95)) <= 1e-4
    assert float(log_abs.max()) <= 1e-3


@pytest.mark.gpu
@pytest.mark.path_matrix
def test_pyhyperscattering_integrator_to_xarray_smoke(nrss_path: ComputationPath):
    """Run NRSS-to-PyHyperScattering integration and verify xarray/remesh invariants."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for PyHyperScattering smoke test.")

    from PyHyperScattering.integrate import WPIntegrator

    energies = [285.0, 286.0]
    data = _run_tiny_pybind_simulation(
        energies=energies,
        return_xarray=True,
        runtime_kwargs=_path_runtime_kwargs(nrss_path),
    )

    assert hasattr(data, "dims")
    assert "energy" in data.dims
    assert data.sizes["energy"] == len(energies)

    integrator = WPIntegrator(use_chunked_processing=False)
    remeshed = integrator.integrateImageStack(data)
    remeshed_vals = np.asarray(remeshed.values)

    assert hasattr(remeshed, "dims")
    assert "energy" in remeshed.dims
    assert remeshed.sizes["energy"] == len(energies)
    assert "chi" in remeshed.dims
    assert any(dim.startswith("q") or dim == "q" for dim in remeshed.dims)
    # The tiny remesh path can pick up a thin NaN fringe at the edge; keep this
    # as a broad sanity check rather than a tight determinism threshold.
    assert float(np.isfinite(remeshed_vals).mean()) >= 0.89


@pytest.mark.gpu
@pytest.mark.path_matrix
def test_pybind_runtime_2d_disk_smoke(nrss_path: ComputationPath):
    """Run a 2D (1x32x32) pybind morphology to cover the 2D computation pathway."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for 2D runtime smoke test.")

    data = _run_tiny_pybind_simulation(
        energies=[285.0],
        return_xarray=True,
        shape=(1, 32, 32),
        sphere_diameter_vox=16,
        runtime_kwargs=_path_runtime_kwargs(nrss_path),
    )
    arr = data.values
    arr_safe = _sanitize_scattering(arr)

    assert arr.shape == (1, 32, 32)
    assert float(np.isfinite(arr).mean()) >= 0.99
    assert float(arr_safe.max()) > 1e-5
    assert float(arr_safe.sum()) > 1e-4


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_tensor_coeff")
def test_cupy_tensor_coeff_host_and_device_residency_parity(nrss_path: ComputationPath):
    """Check a simple maintained tensor_coeff case matches between host and device residency."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for CuPy tensor_coeff parity smoke test.")

    host = _run_tiny_pybind_simulation(
        runtime_kwargs={
            "backend": nrss_path.backend,
            "backend_options": nrss_path.backend_options,
            "resident_mode": "host",
            "ownership_policy": "borrow",
            "field_namespace": "numpy",
        }
    )
    device = _run_tiny_pybind_simulation(runtime_kwargs=_path_runtime_kwargs(nrss_path))
    _assert_scattering_similarity(host, device)


@pytest.mark.gpu
@pytest.mark.path_subset("cupy_direct_polarization")
def test_cupy_direct_polarization_host_and_device_residency_parity(nrss_path: ComputationPath):
    """Check a simple maintained direct_polarization case matches between host and device residency."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for CuPy direct-path parity smoke test.")

    host = _run_tiny_pybind_simulation(
        runtime_kwargs={
            "backend": nrss_path.backend,
            "backend_options": nrss_path.backend_options,
            "resident_mode": "host",
            "ownership_policy": "borrow",
            "field_namespace": "numpy",
        }
    )
    device = _run_tiny_pybind_simulation(runtime_kwargs=_path_runtime_kwargs(nrss_path))
    _assert_scattering_similarity(host, device)


@pytest.mark.gpu
@pytest.mark.cyrsoxs_only
@pytest.mark.reference_parity
def test_cli_serialized_run_matches_pybind_smoke(tmp_path: Path):
    """Compare pybind output to CLI output from serialized morphology/constants/config."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for CLI-vs-pybind smoke test.")
    if shutil.which("CyRSoXS") is None:
        pytest.skip("CyRSoXS CLI executable not found on PATH.")

    energies = [285.0]
    morph = _build_two_material_sphere_morphology(
        energies=energies,
        shape=(32, 32, 32),
        sphere_diameter_vox=16,
    )

    # Intentionally use a single Morphology instance:
    # 1) run pybind, then 2) serialize that same object for CLI parity.
    pybind_vals = morph.run(stdout=False, stderr=False, return_xarray=True).values.copy()
    run_path = tmp_path / "cli_serialized_from_pybind_morph"
    cli_vals = _run_cli_from_serialized_morphology(morph, run_path=run_path)

    assert (run_path / "morphology.hdf5").exists()
    assert (run_path / "config.txt").exists()
    assert (run_path / "HDF5").exists()

    _assert_scattering_parity(pybind_vals, cli_vals)


@pytest.mark.gpu
@pytest.mark.cyrsoxs_only
@pytest.mark.reference_parity
def test_cli_serialized_multi_energy_matches_pybind_smoke(tmp_path: Path):
    """Compare pybind/CLI parity for a small multi-energy simulation."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for multi-energy parity smoke test.")
    if shutil.which("CyRSoXS") is None:
        pytest.skip("CyRSoXS CLI executable not found on PATH.")

    energies = [285.0, 286.0]
    morph = _build_two_material_sphere_morphology(energies=energies, shape=(32, 32, 32))
    pybind_vals = morph.run(stdout=False, stderr=False, return_xarray=True).values.copy()
    cli_vals = _run_cli_from_serialized_morphology(
        morph, run_path=tmp_path / "cli_multi_energy_parity"
    )

    assert pybind_vals.shape == (2, 32, 32)
    assert cli_vals.shape == (2, 32, 32)
    _assert_scattering_parity(pybind_vals, cli_vals)


@pytest.mark.gpu
@pytest.mark.cyrsoxs_only
@pytest.mark.reference_parity
def test_cli_serialized_2d_disk_matches_pybind_smoke(tmp_path: Path):
    """Compare pybind/CLI parity for a 2D (1x32x32) morphology."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for 2D CLI parity smoke test.")
    if shutil.which("CyRSoXS") is None:
        pytest.skip("CyRSoXS CLI executable not found on PATH.")

    morph = _build_two_material_sphere_morphology(
        energies=[285.0],
        shape=(1, 32, 32),
        sphere_diameter_vox=16,
    )
    pybind_vals = morph.run(stdout=False, stderr=False, return_xarray=True).values.copy()
    cli_vals = _run_cli_from_serialized_morphology(morph, run_path=tmp_path / "cli_2d_parity")

    assert pybind_vals.shape == (1, 32, 32)
    assert cli_vals.shape == (1, 32, 32)
    _assert_scattering_parity(
        pybind_vals,
        cli_vals,
        # Full GPU-smoke runs showed occasional 2D scalar-sum drift and a
        # single-pixel hotspot delta while the bulk/log-shape checks stayed tight.
        # Keep the percentile/log-shape guards tight and let the absolute-hotspot
        # bound control this one center-pixel outlier. The max-log guard only
        # needs to exclude pathological reshaping, not this known hotspot wobble.
        rtol_scalar=1.2e-1,
        rtol_max=1e-3,
        p95_abs_max=1e-7,
        max_abs_max=1.2e-4,
        p95_log_max=8e-2,
        max_log_max=8e-1,
    )


@pytest.mark.gpu
@pytest.mark.cyrsoxs_only
@pytest.mark.parametrize(
    "config_overrides",
    [
        {"AlgorithmType": 0, "WindowingType": 0, "EwaldsInterpolation": 1, "RotMask": 1},
        {"AlgorithmType": 1, "WindowingType": 0, "EwaldsInterpolation": 1, "RotMask": 1},
        {"AlgorithmType": 0, "WindowingType": 1, "EwaldsInterpolation": 0, "RotMask": 1},
    ],
)
def test_gpu_config_switch_matrix_smoke(config_overrides: dict):
    """Verify key GPU config switches execute and return finite non-trivial output."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for config-switch smoke test.")

    morph = _build_two_material_sphere_morphology(
        energies=[285.0],
        shape=(32, 32, 32),
        config_overrides=config_overrides,
    )
    arr = morph.run(stdout=False, stderr=False, return_xarray=True).values.copy()
    arr_safe = _sanitize_scattering(arr)

    assert arr.shape == (1, 32, 32)
    assert float(np.isfinite(arr).mean()) >= 0.99
    assert float(arr_safe.max()) > 1e-3
    assert float(arr_safe.sum()) > 1e-2


@pytest.mark.gpu
@pytest.mark.path_matrix
@pytest.mark.reference_parity
def test_eangle_rotation_endpoint_behavior_smoke(nrss_path: ComputationPath):
    """Validate endpoint semantics and expected radial-symmetry trend for E-angle averaging."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for EAngleRotation endpoint smoke test.")
    # Pin to a single visible GPU for more stable endpoint comparisons.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    runtime_kwargs = _path_runtime_kwargs(nrss_path)
    eangle_0 = _build_two_material_asymmetric_lobed_morphology(
        energies=[285.0], eangle_rotation=[0.0, 0.0, 0.0], **runtime_kwargs
    ).run(stdout=False, stderr=False, return_xarray=True).values.copy()
    eangle_165 = _build_two_material_asymmetric_lobed_morphology(
        energies=[285.0], eangle_rotation=[0.0, 15.0, 165.0], **runtime_kwargs
    ).run(stdout=False, stderr=False, return_xarray=True).values.copy()
    eangle_1799 = _build_two_material_asymmetric_lobed_morphology(
        energies=[285.0], eangle_rotation=[0.0, 15.0, 179.9], **runtime_kwargs
    ).run(stdout=False, stderr=False, return_xarray=True).values.copy()
    eangle_180 = _build_two_material_asymmetric_lobed_morphology(
        energies=[285.0], eangle_rotation=[0.0, 15.0, 180.0], **runtime_kwargs
    ).run(stdout=False, stderr=False, return_xarray=True).values.copy()
    asym_0 = _radial_asymmetry_score(eangle_0)
    asym_165 = _radial_asymmetry_score(eangle_165)
    asym_1799 = _radial_asymmetry_score(eangle_1799)
    asym_180 = _radial_asymmetry_score(eangle_180)

    # Expected for current CyRSoXS behavior (v1.1.8.0):
    # numAnglesRotation = round((end-start)/increment + 1)
    # sampled angle_i = start + i*increment
    # so [0,15,165] -> 0..165 and [0,15,179.9]/[0,15,180] -> 0..180.
    # Fixed thresholds were chosen from repeat-run inspection on pinned single-GPU runs.
    # The image-level L1 differences are noisy, but the ring-asymmetry trend is stable and
    # directly encodes the endpoint-semantics behavior we care about.
    #
    # 179.9 and 180 both round to the same 0..180 sampling and should therefore agree.
    assert abs(asym_1799 - asym_180) <= 0.01
    # 180 includes the endpoint-equivalent orientation while 165 does not, which measurably
    # changes the averaged anisotropy.
    assert (asym_180 - asym_165) >= 0.005
    # Rotation averaging should be substantially more radially symmetric than no rotation.
    assert asym_0 > (1.5 * asym_165)
    assert asym_0 > (1.5 * asym_180)
