# NRSS Repo Agent Guide

This file is repo-local guidance for work in this checkout.
It holds workflow, tutorial, test, and implementation details that should version with this repo.

## Repo Map
- Package metadata: `pyproject.toml`
- Core code: `src/NRSS`
- Tutorials: `src/NRSS_tutorials`
- Docs: `docs/source`
- Smoke tests: `tests/smoke`
- Validation tests: `tests/validation`
- Repo upgrade plan: `REPO_UPGRADE_PLAN.md`
- `cupy-rsoxs` backend spec: `optimization/cupy_rsoxs/backend_spec.md`
- `cupy-rsoxs` optimization index: `optimization/cupy_rsoxs/README.md`
- `cupy-rsoxs` validation/program status: `optimization/cupy_rsoxs/validation_and_status.md`
- `cupy-rsoxs` optimization archive: `optimization/cupy_rsoxs/archive/`

## Source And Edit Map
- `src/NRSS/morphology.py`: main `Morphology` / `Material` API, config handling, CyRSoXS object wiring, and run path.
- `src/NRSS/reader.py`: config and morphology/material loading helpers.
- `src/NRSS/writer.py`: config writing, HDF5 writing, and serialization helpers.
- `src/NRSS/visualizer.py`: visualization path used for morphology inspection and pre-run checks.
- `src/NRSS/checkH5.py`: validation helpers and legacy morphology checks.
- `tests/smoke`: fast regression lane and CI-aligned runtime checks.
- `tests/validation`: physics-facing and analytical comparison tests.
- `docs/source`: editable docs source.

## Validation Layout
- `tests/validation/test_*.py`: maintained pytest-facing validation tests.
- `tests/validation/lib/`: maintained reusable helpers for validation tests.
- `tests/validation/data/`: minimal vendored inputs/references needed by maintained validation tests.
- `tests/validation/dev/`: development-only sweeps, falsification probes, and benchmarks; do not treat this as the maintained pytest surface.

## Runtime Assumptions
- Preferred install path is a conda-installed `NRSS` package.
- Current maintained runtime requires CuPy and defaults to the `cupy-rsoxs` backend when both `cupy-rsoxs` and `cyrsoxs` are available.
- `CyRSoXS` remains a supported legacy/reference backend rather than a required default runtime dependency.
- Acceptable alternative when needed: `pip install nrss` plus environment-managed GPU/runtime dependencies.
- Do not assume any machine-local environment name.
- For repo verification, align with CI/local test entry points:
  - editable install path used in CI: `python -m pip install -e . pytest`
  - CPU smoke path used in CI: `python -m pytest tests/smoke -m "not gpu" -v`
  - extended local report path: `scripts/run_local_test_report.sh`
- Tests default to one visible GPU when `CUDA_VISIBLE_DEVICES` is otherwise unset; respect any explicit user or CI pinning.

## Do Not Edit Generated Or Artifact Files
- Do not hand-edit `src/NRSS/_version.py`; it is generated version metadata.
- Do not edit generated build artifacts under `build/`, `dist/`, or `*.egg-info/`.
- Do not edit generated docs output under `docs/build/`; edit sources under `docs/source/` instead.
- Do not edit `test-reports/`, `CyRSoXS.log`, or generated validation output directories under `tests/validation/`.
- Avoid committing notebook-output churn, checkpoint directories, logs, or other ignored artifacts unless the task explicitly requires regenerated artifacts.
- Do not commit or leave behind generated caches such as `__pycache__`, `.nbc`, or `.nbi`, especially under `tests/validation/dev/`.

## Establish Local Context
- Inspect `pyproject.toml`, relevant builders in `src/NRSS`, and local scripts before making workflow changes.
- Confirm environment/dependencies and accelerator visibility before expensive runs.
- Prefer `rg` for codebase search and `pytest -k` for focused iteration.

## Docs And Tutorial Policy
- Edit docs in `docs/source/`.
- Do not edit generated docs output in `docs/build/`.

## Current Source-Coupled Behavior

### Validator details
- `check_materials` currently checks closure with `np.allclose(Vfrac_sum, 1)` using numpy defaults (`rtol=1e-05`, `atol=1e-08`).
- `check_materials` currently hard-bounds only `S` and `Vfrac`; it also requires float dtype and rejects NaNs.
- `validate_all` currently chains:
  - `check_materials(...)`,
  - `inputData.validate()`,
  - `OpticalConstants.validate()`,
  - `voxelData.validate()`.

### Euler details tied to current docs/source
- The repo docs define Euler morphology with a ZYZ convention.
- The current docs/visualization path treat `psi % (2*pi)` and `0..2*pi` as the canonical presentation even though the validator does not hard-bound psi.

### Results lifetime safety
- For the `cyrsoxs` backend, if results are passed out of a function while the
  owning parent object is garbage-collected or deleted, results can be
  deallocated and later access may crash with a `SIGSEGV`, often without a
  Python exception.

## Optional Test/Report Path
- Use smoke tests to guard key behavior and integration points.
- For quick iteration, run focused subsets with `pytest -k`.
- For PR-ready local evidence, run:

```bash
scripts/run_local_test_report.sh
```

- Keep smoke/regression coverage aligned with:
  - morphology validator checks,
  - tiny deterministic pybind runs,
  - `WPIntegrator` and xarray-compatible downstream paths,
  - CLI-vs-pybind parity only for legacy compatibility,
  - `EAngleRotation` semantics,
  - geometry/orientation invariants for the morphology under test.
- Physics test docstrings are surfaced in the local report summary.
- Experimental-validation tests should include a concise provenance/citation block in the docstring, consistent with the maintained core-shell validation style.

## EAngleRotation Semantics In This Repo
- Treat `EAngleRotation` as `[StartAngle, IncrementAngle, EndAngle]`. Spell the argument order out explicitly whenever you discuss it because users frequently get it wrong.

Physics-level intent:
- `EAngleRotation` samples multiple in-plane orientations of the electric-field vector relative to model azimuth in the `YX` plane and averages the resulting full energy panels.

## Runtime And Backend Notes Tied To Current Work
- Current pybind simulation pathways may still require host-side transfers, so GPU-built morphology is not automatically an end-to-end on-device workflow.
- Keep backend behavior parity-oriented by default: alternate backends should mimic trusted pybind/CyRSoXS behavior before optimization changes.
- Prefer backend/input/output policy-explicit workflows where supported.
- For debugging, validation, and reproducible comparisons, prefer pinning to a single GPU by default.
- Be cautious with multi-GPU CyRSoXS energy broadcasting: it can expose instability/segfault behavior in current workflows, so only use multi-GPU when that path is explicitly being exercised.

## CLI Compatibility Rule
- Do not recommend CLI as the preferred path for modern workflows; CLI only for explicit historical reproduction.

## Backend Upgrade Context
- For repo-wide modernization goals, validation status, packaging direction, and historical prep status, use `REPO_UPGRADE_PLAN.md`.
- For stable `cupy-rsoxs` backend contract and execution semantics, use `optimization/cupy_rsoxs/backend_spec.md`.
- For `cupy-rsoxs` optimization work, always start with `optimization/cupy_rsoxs/README.md`.
- For maintained `cupy-rsoxs` validation/path-matrix/program status, use `optimization/cupy_rsoxs/validation_and_status.md`.
- Do not read the whole `optimization/cupy_rsoxs/` documentation tree by default; use its routing table to choose the one specific markdown file needed for the current task.
- Do not open `optimization/cupy_rsoxs/archive/` unless the compact optimization docs are insufficient for a specific historical question.
