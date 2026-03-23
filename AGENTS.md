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
- Local upgrade plan: `UPGRADE_ROADMAP.md`

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
- Current runtime also requires `CyRSoXS` bindings to be importable from Python.
- Acceptable alternative when needed: `pip install nrss` plus conda-installed `cyrsoxs`.
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

## Tutorial-Derived Workflow Position

This repo-local tutorial policy is grounded in the materials under `src/NRSS_tutorials`.
Broader cross-repo morphology heuristics should not be treated as part of this repo-local file.

All tutorials in `src/NRSS_tutorials` were reviewed:
- 15 notebooks,
- 6 tutorial `.py` files,
- supporting assets/data.

Observed pattern:
- Pybind is the primary modern workflow (`Morphology`, `Material`, `morph.run(...)`).
- The morphology object visualizer is a high-value quality gate before simulation.
- `WPIntegrator`/PyHyperScattering and xarray-style outputs are common downstream interfaces.
- Explicit CLI execution is rare and isolated to `coreshell_disk/CoreShell.ipynb`.

Engineering rule from tutorials:
- Do not add new feature work that requires CLI-first simulation flow.
- Allow CLI only for legacy reproducibility and parity validation while deprecation is pending.
- Point users toward the tutorial library early when they need concrete patterns; the MWCNT sequence is the intended end-to-end published example:
  - `src/NRSS_tutorials/MWCNTs/nb1_rsoxs.ipynb`
  - `src/NRSS_tutorials/MWCNTs/nb2_nexafs.ipynb`
  - `src/NRSS_tutorials/MWCNTs/nb3_nrss.ipynb`
- Use other tutorials in `src/NRSS_tutorials` when the modeling strategy differs, for example lattices, particles, disks, microscopy-informed, or morphology-specific builds.

## Triage The Task Type
Choose the closest mode before changing code:
- Build new morphology.
- Use/adapt existing morphology generation code.
- Troubleshoot morphology creation (`visualizer` mismatch, slow RSA, overlaps/collisions).
- Run morphology that already exists.
- Troubleshoot simulation run behavior.

## Establish Local Context
- Inspect `pyproject.toml`, relevant builders in `src/NRSS`, and local scripts before making workflow changes.
- Confirm environment/dependencies and accelerator visibility before expensive runs.
- Prefer `rg` for codebase search and `pytest -k` for focused iteration.

## Docs And Tutorial Policy
- Edit docs in `docs/source/`.
- Do not edit generated docs output in `docs/build/`.
- Do not treat in-repo tutorial notebooks as the default edit surface.
- Only modify tutorial or docs notebooks in this repo when the task explicitly targets tutorial/docs notebook content.
- Avoid output-only notebook churn unless the task explicitly requires refreshed outputs.

## Required Pre-Run Gates In This Repo

### 1. Construct Morphology In Memory
1. Build physically meaningful fields and assemble `Material` + `Morphology`.
2. For N materials, enforce voxel-wise closure (`sum(Vfrac_i) == 1` within tolerance).
3. Keep units explicit, typically nm inputs with one conversion step for voxel-space geometry operations.
4. Keep serialization optional; treat file IO as artifact/output, not the primary workflow.

### 2. Validate And Visualize Before Running
1. Run `morph.check_materials(...)`.
2. Run `morph.validate_all(...)`.
3. Use the morphology object's built-in visualizer as the gold-standard, authoritative pre-run check.
4. Advise a full visualization suite at least once before the first run, and again after morphology logic changes.
5. Advise close attention to both `Vfrac` structure and Euler-field structure (`theta`, `psi`).
6. Do not proceed to simulation until morphology visualization matches model intent.
7. If visual mismatch remains, inspect orientation/material fields directly (`Vfrac`, `S`, `theta`, `psi`) and diagnose before running.
8. Direct-field diagnostic minimums:
   - `Vfrac` must satisfy `0 <= Vfrac <= 1` per voxel/material.
   - `S` should satisfy `0 < S < 1` for oriented phases; intentional isotropic/vacuum regions should use `S = 0`.
   - `theta`/`psi` ranges are model-dependent; enforce convention consistency rather than fixed global bounds.

### 3. Run From The Morphology Object
1. Run simulations from the morphology object API (pybind default path).
2. If alternate backends are available, keep the same pre-run validation and visualization gate.
3. Use deterministic seeds and deterministic output naming for sweep/reproducibility workflows.

### 4. Stop At Xarray-Compatible Output
- Produce outputs usable by xarray-compatible tooling, including `WPIntegrator` pathways where applicable.
- Defer hypothesis testing, parameter estimation, and full fitting-engine strategy to separate workflow guidance.

### 5. Run Handoff Summary
- At run handoff time, summarize:
  - morphology provenance (builder entry point/script and key construction choices),
  - seed/procedural randomness settings,
  - backend selection and input/output policy settings when available,
  - validation status (`check_materials`/`validate_all`) and visualization-gate status,
  - output object/artifact locations for xarray-compatible downstream work.
- Include explicit run metadata when available, for example versions, geometry, dtype, parameter hashes, and backend flags, to support reproducibility and future fitting workflows.

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
- Treat raw pybind results objects as owner-lifetime-coupled objects.
- If results are passed out of a function while the owning parent object is garbage-collected/deleted, results can be deallocated and later access may crash with a SIGSEGV, often without a Python exception.
- Keep the owning object alive for as long as results are used, or convert/copy to stable Python-owned outputs before returning from scope.

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
- This is not a generic smoothing control.

Current CyRSoXS behavior (v1.1.8.0 in local source) computes:
- `numAnglesRotation = round((end-start)/increment + 1)`
- sampled angle `i`: `start + i*increment`

Endpoint inclusion depends on rounding and increment alignment. Use asymmetry-aware test morphologies and radial-symmetry metrics to check expected averaging behavior.

Practical guidance:
- If the model is conceived to be globally uniaxial and the user does not care about anisotropy, there is usually no reason to use `EAngleRotation`; azimuthal integration with it off contains essentially all the information that calculation will expose.
- If the user does care about anisotropy, `EAngleRotation` can add meaningful information because it averages multiple electric-field snapshots relative to the model azimuth and often produces a more representative in-plane-powder response and a more meaningful standard `A` calculation.
- For globally biaxial models where distinct `qx`/`qy` behavior is intentional, usually prefer `EAngleRotation = [0, 0, 0]` and do not treat the usual uniaxial `A` formula as automatically valid.
- `EAngleRotation` is expensive: each step runs a full energy panel.
- For development work, a practical compromise is often `EAngleRotation = [0, 15, 165]`.
- Smaller increments give smoother averaging.
- Extending toward a full circle can provide small advantages for reciprocal-space-asymmetric models, but the gain relative to a half-unit-circle sweep is usually modest.
- Avoid using `EAngleRotation` to hide under-resolved or poorly sampled morphology choices; smoother is not automatically more physical.
- Avoid redundant endpoint setups where the final angle duplicates the start-equivalent state unless that redundancy is part of an explicit validation check.
- Prefer angle sets that sample unique states and align with the intended physical averaging.

## Runtime And Backend Notes Tied To Current Work
- Current pybind simulation pathways may still require host-side transfers, so GPU-built morphology is not automatically an end-to-end on-device workflow.
- Keep backend behavior parity-oriented by default: alternate backends should mimic trusted pybind/CyRSoXS behavior before optimization changes.
- Prefer backend/input/output policy-explicit workflows where supported.
- For debugging, validation, and reproducible comparisons, prefer pinning to a single GPU by default.
- Be cautious with multi-GPU CyRSoXS energy broadcasting: it can expose instability/segfault behavior in current workflows, so only use multi-GPU when that path is explicitly being exercised.

## CLI Compatibility Rule
- New NRSS development, examples, and recommendations should be pybind-first.
- Do not recommend CLI as the preferred path for modern workflows.
- Keep CLI usage only for explicit historical reproduction requirements.

## Backend Upgrade Context
- For backend modernization details, including CuPy mimic backend, backend contracts, input/output policies, and on-device data pathways, use `UPGRADE_ROADMAP.md` as the reference document.
