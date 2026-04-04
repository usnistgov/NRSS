#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Development studies should default to one visible GPU unless the caller has
# already pinned the runtime.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from NRSS.backends import format_backend_availability, get_backend_info
from NRSS.morphology import Material, Morphology
from tests.validation.dev.cupy_rsoxs_optimization.run_cupy_rsoxs_optimization_matrix import (
    CORE_SHELL_SINGLE_ENERGIES,
    CORE_SHELL_TRIPLE_ENERGIES,
    EANGLE_FULL,
    EANGLE_LIMITED,
    EANGLE_OFF,
    ISOTROPIC_MATERIAL_REPRESENTATIONS,
    SIZE_SPECS,
    SUMMARY_NAME,
    _build_isotropic_representation_comparisons,
    _centered_core_shell_energies,
    _default_morphology_config,
    _energy_list_label_fragment,
    _isotropic_representation_comparison_summary_line,
    _isotropic_representation_label_suffix,
    _json_default,
    _parse_csv_labels,
    _parse_cuda_prewarm_mode,
    _parse_energy_lists,
    _parse_positive_int_csv,
    _parse_rotation_specs,
    _parse_worker_warmup_runs,
    _rotation_label_fragment,
    _timestamp,
    build_scaled_core_shell_materials,
)
from tests.validation.lib.core_shell import has_visible_gpu
from tests.validation.lib.core_shell import release_runtime_memory


OUT_ROOT = REPO_ROOT / "test-reports" / "cyrsoxs-timing-dev"
PRIMARY_TIMING_BOUNDARY = (
    "Morphology(...) -> completed run(return_xarray=False) with results manifested on the "
    "Morphology object"
)
DEFAULT_BACKEND = "cyrsoxs"
DEFAULT_RESIDENT_MODE = "host"
DEFAULT_FIELD_NAMESPACE = "numpy"
DEFAULT_INPUT_POLICY = "strict"
DEFAULT_OWNERSHIP_POLICY = "borrow"
_CYRSOXS_STREAM_NOISE_SNIPPETS = (
    "CyRSoXS",
    "Thanks for using CyRSoXS",
    "Maximum Number Of Material",
    "Size of Real",
    "Version   :",
    "Git patch :",
    "Number of CUDA devices:",
    "[STAT] Executing:",
    "[STAT] Energy =",
    "[INFO] [GPU =",
)


@dataclass(frozen=True)
class BenchmarkCase:
    label: str
    family: str
    shape_label: str
    energies_ev: tuple[float, ...]
    eangle_rotation: tuple[float, float, float]
    isotropic_representation: str
    cuda_prewarm_mode: str
    backend: str = DEFAULT_BACKEND
    resident_mode: str = DEFAULT_RESIDENT_MODE
    field_namespace: str = DEFAULT_FIELD_NAMESPACE
    input_policy: str = DEFAULT_INPUT_POLICY
    ownership_policy: str = DEFAULT_OWNERSHIP_POLICY
    create_cy_object: bool = True
    worker_warmup_runs: int = 0
    notes: str | None = None


@contextmanager
def _suppress_process_streams(*, stdout: bool = True, stderr: bool = True):
    saved_fds: list[tuple[int, int]] = []
    sys.stdout.flush()
    sys.stderr.flush()
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    try:
        for target_fd, enabled in ((1, stdout), (2, stderr)):
            if not enabled:
                continue
            saved_fd = os.dup(target_fd)
            os.dup2(devnull_fd, target_fd)
            saved_fds.append((target_fd, saved_fd))
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        for target_fd, saved_fd in reversed(saved_fds):
            os.dup2(saved_fd, target_fd)
            os.close(saved_fd)
        os.close(devnull_fd)


def _case_note(case: BenchmarkCase) -> str:
    size_spec = SIZE_SPECS.get(case.shape_label)
    if case.family == "core_shell":
        return (
            f"CoreShell {case.shape_label} {size_spec.shape} PhysSize={size_spec.phys_size_nm} "
            f"EAngleRotation={list(case.eangle_rotation)} energies={list(case.energies_ev)} "
            f"resident_mode={case.resident_mode} field_namespace={case.field_namespace} "
            f"isotropic_representation={case.isotropic_representation} "
            f"cuda_prewarm_mode={case.cuda_prewarm_mode} "
            f"worker_warmup_runs={case.worker_warmup_runs} "
            f"input_policy={case.input_policy} ownership_policy={case.ownership_policy}"
        )
    raise AssertionError(f"Unsupported benchmark family for the timing harness: {case.family}")


def _result_summary_line(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return f"{result['label']}: {result.get('status')} ({result.get('error_type', 'unknown')})"
    return f"{result['label']}: primary {float(result['primary_seconds']):.3f}s"


def _prepare_core_shell_case_inputs(case: BenchmarkCase) -> dict[str, Any]:
    if case.family != "core_shell":
        raise AssertionError(f"Unsupported timing family: {case.family}")
    size_spec = SIZE_SPECS[case.shape_label]
    return {
        "num_material": 3,
        "materials": build_scaled_core_shell_materials(
            size_spec=size_spec,
            energies_ev=case.energies_ev,
            field_namespace=case.field_namespace,
            isotropic_representation=case.isotropic_representation,
        ),
        "phys_size": float(size_spec.phys_size_nm),
        "config": _default_morphology_config(case.energies_ev, case.eangle_rotation),
    }


def _construct_morphology_for_timing_case(case: BenchmarkCase, prepared: dict[str, Any]) -> Morphology:
    return Morphology(
        prepared["num_material"],
        materials=prepared["materials"],
        PhysSize=prepared["phys_size"],
        config=prepared["config"],
        create_cy_object=case.create_cy_object,
        backend=case.backend,
        resident_mode=case.resident_mode,
        input_policy=case.input_policy,
        ownership_policy=case.ownership_policy,
    )


def _build_prewarm_morphology() -> Morphology:
    energies = [285.0]
    shape = (1, 4, 4)
    zeros = np.zeros(shape, dtype=np.float32)
    ones = np.ones(shape, dtype=np.float32)
    zero_constants = {285.0: [0.0, 0.0, 0.0, 0.0]}
    materials = {
        1: Material(
            materialID=1,
            Vfrac=ones.copy(),
            S=zeros.copy(),
            theta=zeros.copy(),
            psi=zeros.copy(),
            energies=energies,
            opt_constants=zero_constants,
            name="prewarm_core",
        ),
        2: Material(
            materialID=2,
            Vfrac=zeros.copy(),
            S=zeros.copy(),
            theta=zeros.copy(),
            psi=zeros.copy(),
            energies=energies,
            opt_constants=zero_constants,
            name="prewarm_matrix",
        ),
    }
    config = {
        "CaseType": 0,
        "MorphologyType": 0,
        "Energies": energies,
        "EAngleRotation": [0.0, 0.0, 0.0],
        "AlgorithmType": 0,
        "WindowingType": 0,
        "RotMask": 0,
        "ReferenceFrame": 1,
        "EwaldsInterpolation": 1,
    }
    return Morphology(
        2,
        materials=materials,
        PhysSize=5.0,
        config=config,
        create_cy_object=True,
        backend=DEFAULT_BACKEND,
        resident_mode=DEFAULT_RESIDENT_MODE,
        input_policy=DEFAULT_INPUT_POLICY,
        ownership_policy=DEFAULT_OWNERSHIP_POLICY,
    )


def _prewarm_cyrsoxs_runtime() -> float:
    started = time.perf_counter()
    morph = None
    try:
        with _suppress_process_streams():
            morph = _build_prewarm_morphology()
            morph.run(stdout=False, stderr=False, return_xarray=False)
    finally:
        if morph is not None:
            try:
                morph.release_runtime()
            except Exception:
                pass
    return time.perf_counter() - started


def _resolve_case_cuda_prewarm(case: BenchmarkCase) -> tuple[str, str | None]:
    if case.cuda_prewarm_mode == "off":
        return "off", "CyRSoXS prewarm disabled."
    if case.cuda_prewarm_mode == "before_prepare_inputs":
        return (
            "before_prepare_inputs",
            (
                "Best-effort CyRSoXS import and tiny launch warmup before primary_start. "
                "Kept for parity with the host-resident cupy-rsoxs dev harness."
            ),
        )
    raise AssertionError(f"Unsupported cuda_prewarm mode: {case.cuda_prewarm_mode!r}")


def _manifested_results_state(morphology: Morphology) -> dict[str, bool]:
    return {
        "simulated": bool(morphology.simulated),
        "results_locked": bool(getattr(morphology, "_results_locked", False)),
        "scatteringPattern": morphology.scatteringPattern is not None,
    }


def _expected_panel_shape(morphology: Morphology) -> list[int]:
    return [
        len(tuple(float(energy) for energy in morphology.Energies)),
        int(morphology.NumZYX[1]),
        int(morphology.NumZYX[2]),
    ]


def _sanitize_worker_output(text: str) -> str:
    kept_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if any(snippet in stripped for snippet in _CYRSOXS_STREAM_NOISE_SNIPPETS):
            continue
        if set(stripped) <= {"=", "_", "-", "|"}:
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines).strip()


def _run_case_once(
    *,
    case: BenchmarkCase,
    prepared_inputs: dict[str, Any],
) -> dict[str, Any]:
    morphology = None
    try:
        with _suppress_process_streams():
            primary_start = time.perf_counter()
            morphology = _construct_morphology_for_timing_case(case, prepared_inputs)
            result = {
                "primary_seconds": time.perf_counter() - primary_start,
                "resolved_backend_options": dict(morphology.backend_options),
            }
            morphology.run(stdout=False, stderr=False, return_xarray=False)
            result["primary_seconds"] = time.perf_counter() - primary_start

        manifested = _manifested_results_state(morphology)
        result["results_manifested_on_object"] = manifested
        if not all(manifested.values()):
            raise AssertionError(
                "CyRSoXS run returned without manifesting results on the Morphology object."
            )

        result["runtime_objects_present"] = {
            "inputData": morphology.inputData is not None,
            "OpticalConstants": morphology.OpticalConstants is not None,
            "voxelData": morphology.voxelData is not None,
            "scatteringPattern": morphology.scatteringPattern is not None,
        }
        result["backend_timing_details"] = morphology.backend_timings
        result["panel_shape"] = _expected_panel_shape(morphology)
        return result
    finally:
        if morphology is not None:
            try:
                morphology.release_runtime()
            except Exception:
                pass
        del morphology


def _worker_main(case_path: Path, result_path: Path) -> int:
    case = BenchmarkCase(**json.loads(case_path.read_text()))
    result: dict[str, Any] = {
        "label": case.label,
        "family": case.family,
        "backend": case.backend,
        "shape_label": case.shape_label,
        "resident_mode": case.resident_mode,
        "field_namespace": case.field_namespace,
        "isotropic_representation": case.isotropic_representation,
        "cuda_prewarm_requested_mode": case.cuda_prewarm_mode,
        "input_policy": case.input_policy,
        "ownership_policy": case.ownership_policy,
        "energies_ev": list(case.energies_ev),
        "eangle_rotation": list(case.eangle_rotation),
        "timing_boundary": PRIMARY_TIMING_BOUNDARY,
        "note": case.notes or _case_note(case),
        "worker_warmup_runs_requested": int(case.worker_warmup_runs),
        "status": "error",
    }

    if case.backend != DEFAULT_BACKEND:
        raise AssertionError(f"The cyrsoxs timing harness only supports backend={DEFAULT_BACKEND!r}.")

    prepared_inputs = None
    morphology = None
    try:
        prewarm_mode, prewarm_note = _resolve_case_cuda_prewarm(case)
        result["cuda_prewarm_applied_mode"] = prewarm_mode
        if prewarm_note is not None:
            result["cuda_prewarm_note"] = prewarm_note
        if prewarm_mode == "before_prepare_inputs":
            result["cuda_prewarm_seconds"] = _prewarm_cyrsoxs_runtime()

        prepared_inputs = _prepare_core_shell_case_inputs(case)
        warmup_seconds_total = 0.0
        warmup_runs_completed = 0
        for _ in range(case.worker_warmup_runs):
            warmup_started = time.perf_counter()
            warmup_result = _run_case_once(case=case, prepared_inputs=prepared_inputs)
            warmup_seconds_total += time.perf_counter() - warmup_started
            warmup_runs_completed += 1
            if "resolved_backend_options" not in result:
                result["resolved_backend_options"] = warmup_result["resolved_backend_options"]
            del warmup_result

        result["worker_warmup_runs_completed"] = warmup_runs_completed
        if warmup_runs_completed:
            result["worker_warmup_seconds_total"] = warmup_seconds_total

        timed_result = _run_case_once(case=case, prepared_inputs=prepared_inputs)
        result.update(timed_result)
        result["status"] = "ok"

    except BaseException as exc:  # noqa: BLE001 - worker must serialize failures
        result["status"] = "error"
        result["error_type"] = exc.__class__.__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        del morphology, prepared_inputs
        release_runtime_memory()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2, default=_json_default) + "\n")
    return 0


def _run_case_subprocess(
    *,
    case: BenchmarkCase,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cyrsoxs_timing_", dir=output_dir) as tmp_dir:
        tmp_path = Path(tmp_dir)
        case_path = tmp_path / "case.json"
        result_path = tmp_path / "result.json"
        case_path.write_text(json.dumps(asdict(case), indent=2, default=_json_default) + "\n")

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-case-path",
            str(case_path),
            "--worker-result-path",
            str(result_path),
        ]

        started = time.perf_counter()
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - started

        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            result = {
                "label": case.label,
                "backend": case.backend,
                "family": case.family,
                "shape_label": case.shape_label,
                "status": "subprocess_failed",
                "error_type": "SubprocessFailure",
                "error": "Worker exited before writing a result file.",
            }

        result["subprocess_returncode"] = int(completed.returncode)
        result["subprocess_seconds"] = elapsed
        if result.get("status") != "ok":
            stdout_text = _sanitize_worker_output(completed.stdout)
            stderr_text = _sanitize_worker_output(completed.stderr)
            if stdout_text:
                result["worker_stdout"] = stdout_text[-4000:]
            if stderr_text:
                result["worker_stderr"] = stderr_text[-4000:]
        return result


def _timing_cases(
    *,
    isotropic_representations: tuple[str, ...],
    cuda_prewarm_mode: str,
    size_labels: tuple[str, ...],
    include_triple_no_rotation: bool,
    include_triple_limited: bool,
    include_full_small_check: bool,
    no_rotation_energy_counts: tuple[int, ...],
    rotation_specs: tuple[tuple[float, float, float], ...],
    energy_lists: tuple[tuple[float, ...], ...],
    worker_warmup_runs: int,
) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    include_representation_in_label = (
        len(isotropic_representations) > 1
        or isotropic_representations[0] != "legacy_zero_array"
    )
    lane_note = (
        "Legacy pybind host lane with NumPy authoritative morphology fields, "
        "input_policy='strict', and ownership_policy='borrow' to mirror the default "
        "cupy-rsoxs host-resident dev contract."
    )
    for isotropic_representation in isotropic_representations:
        representation_suffix = _isotropic_representation_label_suffix(
            isotropic_representation,
            include_suffix=include_representation_in_label,
        )

        def append_case(
            *,
            label: str,
            size_label: str,
            energies_ev: tuple[float, ...],
            eangle_rotation: tuple[float, float, float],
            notes: str,
        ) -> None:
            cases.append(
                BenchmarkCase(
                    label=label,
                    family="core_shell",
                    shape_label=size_label,
                    energies_ev=energies_ev,
                    eangle_rotation=eangle_rotation,
                    isotropic_representation=isotropic_representation,
                    cuda_prewarm_mode=cuda_prewarm_mode,
                    worker_warmup_runs=worker_warmup_runs,
                    notes=notes,
                )
            )

        for size_label in size_labels:
            append_case(
                label=f"core_shell_{size_label}_single_no_rotation_host_cyrsoxs{representation_suffix}",
                size_label=size_label,
                energies_ev=CORE_SHELL_SINGLE_ENERGIES,
                eangle_rotation=EANGLE_OFF,
                notes=(
                    f"{lane_note} Isotropic-material representation={isotropic_representation}. "
                    "Primary no-rotation tuning lane."
                ),
            )

            if include_triple_no_rotation:
                append_case(
                    label=f"core_shell_{size_label}_triple_no_rotation_host_cyrsoxs{representation_suffix}",
                    size_label=size_label,
                    energies_ev=CORE_SHELL_TRIPLE_ENERGIES,
                    eangle_rotation=EANGLE_OFF,
                    notes=(
                        f"{lane_note} Isotropic-material representation={isotropic_representation}. "
                        "Secondary three-energy no-rotation checkpoint lane."
                    ),
                )

            for energy_count in no_rotation_energy_counts:
                if energy_count == 1:
                    continue
                append_case(
                    label=(
                        f"core_shell_{size_label}_{energy_count}energy_no_rotation_"
                        f"host_cyrsoxs{representation_suffix}"
                    ),
                    size_label=size_label,
                    energies_ev=_centered_core_shell_energies(energy_count),
                    eangle_rotation=EANGLE_OFF,
                    notes=(
                        f"{lane_note} Isotropic-material representation={isotropic_representation}. "
                        f"Development-only centered-energy no-rotation lane with {energy_count} "
                        "energies."
                    ),
                )

            if include_triple_limited:
                append_case(
                    label=(
                        f"core_shell_{size_label}_triple_limited_rotation_host_cyrsoxs"
                        f"{representation_suffix}"
                    ),
                    size_label=size_label,
                    energies_ev=CORE_SHELL_TRIPLE_ENERGIES,
                    eangle_rotation=EANGLE_LIMITED,
                    notes=(
                        f"{lane_note} Isotropic-material representation={isotropic_representation}. "
                        "Secondary limited-EAngle checkpoint lane."
                    ),
                )

            for eangle_rotation in rotation_specs:
                if eangle_rotation == EANGLE_OFF:
                    continue
                rotation_label = _rotation_label_fragment(eangle_rotation)
                append_case(
                    label=(
                        f"core_shell_{size_label}_single_{rotation_label}_host_cyrsoxs"
                        f"{representation_suffix}"
                    ),
                    size_label=size_label,
                    energies_ev=CORE_SHELL_SINGLE_ENERGIES,
                    eangle_rotation=eangle_rotation,
                    notes=(
                        f"{lane_note} Isotropic-material representation={isotropic_representation}. "
                        f"Custom single-energy rotation lane with EAngleRotation={list(eangle_rotation)} "
                        "([StartAngle, IncrementAngle, EndAngle])."
                    ),
                )

            for energies_ev in energy_lists:
                if energies_ev == CORE_SHELL_SINGLE_ENERGIES:
                    continue
                energy_label = _energy_list_label_fragment(energies_ev)
                append_case(
                    label=(
                        f"core_shell_{size_label}_{energy_label}_no_rotation_host_cyrsoxs"
                        f"{representation_suffix}"
                    ),
                    size_label=size_label,
                    energies_ev=energies_ev,
                    eangle_rotation=EANGLE_OFF,
                    notes=(
                        f"{lane_note} Isotropic-material representation={isotropic_representation}. "
                        f"Custom explicit-energy no-rotation lane with energies={list(energies_ev)}."
                    ),
                )

            for eangle_rotation in rotation_specs:
                if eangle_rotation == EANGLE_OFF:
                    continue
                rotation_label = _rotation_label_fragment(eangle_rotation)
                for energies_ev in energy_lists:
                    if energies_ev == CORE_SHELL_SINGLE_ENERGIES:
                        continue
                    energy_label = _energy_list_label_fragment(energies_ev)
                    append_case(
                        label=(
                            f"core_shell_{size_label}_{energy_label}_{rotation_label}_"
                            f"host_cyrsoxs{representation_suffix}"
                        ),
                        size_label=size_label,
                        energies_ev=energies_ev,
                        eangle_rotation=eangle_rotation,
                        notes=(
                            f"{lane_note} Isotropic-material representation={isotropic_representation}. "
                            f"Custom explicit-energy plus custom rotation lane with "
                            f"energies={list(energies_ev)} and EAngleRotation={list(eangle_rotation)} "
                            "([StartAngle, IncrementAngle, EndAngle])."
                        ),
                    )

        if include_full_small_check and "small" in size_labels:
            append_case(
                label=f"core_shell_small_triple_full_rotation_host_cyrsoxs{representation_suffix}",
                size_label="small",
                energies_ev=CORE_SHELL_TRIPLE_ENERGIES,
                eangle_rotation=EANGLE_FULL,
                notes=(
                    f"{lane_note} Isotropic-material representation={isotropic_representation}. "
                    "Occasional expensive checkpoint for the full parity-style rotation loop."
                ),
            )
    return cases


def _write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")


def run_matrix(args: argparse.Namespace) -> int:
    backend_info = get_backend_info(DEFAULT_BACKEND)
    if not backend_info.available:
        raise SystemExit(
            f"Requested NRSS backend {DEFAULT_BACKEND!r} is unavailable.\n"
            f"{format_backend_availability()}"
        )
    if not has_visible_gpu():
        raise SystemExit("No visible NVIDIA GPU found for the cyrsoxs timing study.")

    run_label = args.label or _timestamp()
    isotropic_representations = _parse_isotropic_material_representations(
        args.isotropic_material_representation
    )
    cuda_prewarm_mode = _parse_cuda_prewarm_mode(args.cuda_prewarm)
    size_labels = _parse_csv_labels(args.size_labels)
    unknown = [label for label in size_labels if label not in SIZE_SPECS]
    if unknown:
        raise SystemExit(f"Unsupported size_labels entries: {unknown!r}")
    no_rotation_energy_counts = _parse_positive_int_csv(args.no_rotation_energy_counts)
    rotation_specs = _parse_rotation_specs(args.rotation_specs)
    energy_lists = _parse_energy_lists(args.energy_lists)
    worker_warmup_runs = _parse_worker_warmup_runs(args.worker_warmup_runs)

    run_dir = OUT_ROOT / run_label
    output_dir = run_dir / "cases"
    summary: dict[str, Any] = {
        "label": run_label,
        "created_utc": _timestamp(),
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "backend": DEFAULT_BACKEND,
        "backend_import_target": backend_info.import_target,
        "timing_boundary": PRIMARY_TIMING_BOUNDARY,
        "resident_mode": DEFAULT_RESIDENT_MODE,
        "field_namespace": DEFAULT_FIELD_NAMESPACE,
        "input_policy": DEFAULT_INPUT_POLICY,
        "ownership_policy": DEFAULT_OWNERSHIP_POLICY,
        "isotropic_material_representations": list(isotropic_representations),
        "cuda_prewarm_mode": cuda_prewarm_mode,
        "worker_warmup_runs": worker_warmup_runs,
        "size_labels": list(size_labels),
        "include_triple_no_rotation": bool(args.include_triple_no_rotation),
        "include_triple_limited": bool(args.include_triple_limited),
        "no_rotation_energy_counts": list(no_rotation_energy_counts),
        "rotation_specs": [list(spec) for spec in rotation_specs],
        "explicit_energy_lists": [list(energies) for energies in energy_lists],
        "timing_cases": {},
        "isotropic_representation_comparisons": {},
    }

    run_dir.mkdir(parents=True, exist_ok=True)

    print("Running cyrsoxs timing cases...", flush=True)
    for case in _timing_cases(
        isotropic_representations=isotropic_representations,
        cuda_prewarm_mode=cuda_prewarm_mode,
        size_labels=size_labels,
        include_triple_no_rotation=args.include_triple_no_rotation,
        include_triple_limited=args.include_triple_limited,
            include_full_small_check=args.include_full_small_check,
            no_rotation_energy_counts=no_rotation_energy_counts,
            rotation_specs=rotation_specs,
            energy_lists=energy_lists,
            worker_warmup_runs=worker_warmup_runs,
        ):
        result = _run_case_subprocess(case=case, output_dir=output_dir)
        summary["timing_cases"][case.label] = result
        print(_result_summary_line(result), flush=True)

    summary["isotropic_representation_comparisons"] = _build_isotropic_representation_comparisons(
        summary["timing_cases"]
    )
    for comparison_label, comparison in summary["isotropic_representation_comparisons"].items():
        print(
            _isotropic_representation_comparison_summary_line(comparison_label, comparison),
            flush=True,
        )

    _write_summary(run_dir, summary)
    print(f"Wrote {run_dir / SUMMARY_NAME}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only cyrsoxs timing matrix for legacy pybind speed studies. "
            "Default run is the small host-style single-energy no-rotation lane with "
            "resident_mode='host', NumPy authoritative fields, input_policy='strict', "
            "and ownership_policy='borrow'. Primary timing starts at Morphology(...) "
            "construction and ends after run(return_xarray=False) completes with results "
            "manifested on the Morphology object."
        )
    )
    parser.add_argument("--label", default=None, help="Output subdirectory label under test-reports.")
    parser.add_argument(
        "--isotropic-material-representation",
        default="legacy_zero_array",
        help=(
            "Isotropic-material representation to use for the CoreShell core/matrix materials. "
            "Supported values: legacy_zero_array, enum_contract, or both."
        ),
    )
    parser.add_argument(
        "--cuda-prewarm",
        default="off",
        help=(
            "Optional best-effort CyRSoXS import/launch prewarm outside the primary timing "
            "boundary. Supported values: off or before_prepare_inputs."
        ),
    )
    parser.add_argument(
        "--size-labels",
        default="small",
        help="Comma-separated subset of size labels to run, for example 'small,medium'.",
    )
    parser.add_argument(
        "--worker-warmup-runs",
        type=int,
        default=0,
        help=(
            "Development-only fully-hot mode. Run this many untimed identical warm-up passes "
            "inside each worker subprocess before the timed boundary. Default: 0."
        ),
    )
    parser.add_argument(
        "--include-triple-no-rotation",
        action="store_true",
        help="Include the no-rotation three-energy CoreShell lane.",
    )
    parser.add_argument(
        "--include-triple-limited",
        action="store_true",
        help="Include the limited-rotation three-energy CoreShell lane.",
    )
    parser.add_argument(
        "--no-rotation-energy-counts",
        default="",
        help=(
            "Optional comma-separated extra no-rotation energy counts for development sweeps, "
            "for example '2,4,8'. Energies are selected as a centered contiguous subset "
            "from the CoreShell optics table."
        ),
    )
    parser.add_argument(
        "--rotation-specs",
        default="",
        help=(
            "Optional comma-separated explicit rotation specs in start:increment:end order, "
            "for example '0:15:165,0:5:165'. Each spec maps to "
            "EAngleRotation=[StartAngle, IncrementAngle, EndAngle]."
        ),
    )
    parser.add_argument(
        "--energy-lists",
        default="",
        help=(
            "Optional comma-separated explicit energy groups. Quote this argument because each "
            "group uses '|' separators, for example '284.7|285.0|285.2,284.9|285.0|285.1|285.2'."
        ),
    )
    parser.add_argument(
        "--include-full-small-check",
        action="store_true",
        help="Include the expensive full-rotation small CoreShell case.",
    )
    parser.add_argument("--worker-case-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result-path", default=None, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.worker_case_path:
        return _worker_main(
            case_path=Path(args.worker_case_path),
            result_path=Path(args.worker_result_path),
        )
    return run_matrix(args)


def _parse_isotropic_material_representations(raw: str) -> tuple[str, ...]:
    cleaned = raw.strip().lower()
    if cleaned == "both":
        return ISOTROPIC_MATERIAL_REPRESENTATIONS

    requested = tuple(dict.fromkeys(part.strip().lower() for part in raw.split(",") if part.strip()))
    if not requested:
        raise SystemExit(
            "isotropic_material_representation must select at least one representation."
        )
    unknown = tuple(
        representation
        for representation in requested
        if representation not in ISOTROPIC_MATERIAL_REPRESENTATIONS
    )
    if unknown:
        raise SystemExit(
            "Unsupported isotropic material representation(s) "
            f"{unknown!r}. Valid values: {ISOTROPIC_MATERIAL_REPRESENTATIONS!r} or 'both'."
        )
    return requested


if __name__ == "__main__":
    raise SystemExit(main())
