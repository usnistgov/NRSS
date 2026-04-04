# NRSS Repo Upgrade Plan

This document is the authoritative repo-wide modernization plan for NRSS.
It covers repository-level goals, the validation/test program, packaging and
environment direction, workflow policy, phased delivery, and historical
implementation status that still matters when resuming the work.

Related planning documents:

- `CUPY_RSOXS_BACKEND_SPEC.md`
  - stable `cupy-rsoxs` backend contract, physics target, backend seams, and
    implementation-facing risks/checklists
- `CUPY_RSOXS_OPTIMIZATION_LEDGER.md`
  - speed-optimization methodology, accepted/rejected experiments, and current
    benchmark authority

Legacy mixed-role documents now redirect to these three files:

- `UPGRADE_ROADMAP.md`
- `BACKEND_UPGRADE_GUIDE.md`

## Goal

Build a new NRSS backend architecture that:

1. Preserves trusted physics behavior (CyRSoXS parity first).
2. Enables a CuPy-native simulation path to avoid avoidable GPU->CPU->GPU
   transfers.
3. Supports optional future backends (PyTorch, JAX) without hard dependencies.
4. Improves robustness with deterministic, reproducible regression testing.
5. Provides durable value even if alternate backend implementation is delayed
   (test hardening alone is a success milestone).

## Current Repo-Wide State

1. The test-hardening milestone is implemented and remains a meaningful
   modernization outcome even independent of backend speed work.
2. The backend-preparation stages from the historical prep guide are complete
   for the current registry/routing/reporting scope.
3. `cupy-rsoxs` compute/runtime work now exists in-repo; stable backend
   behavior is documented in `CUPY_RSOXS_BACKEND_SPEC.md`.
4. `cupy-rsoxs` is now the intended default backend when both `cupy-rsoxs` and
   `cyrsoxs` are installed and runnable.
5. CuPy is now a required maintained runtime dependency rather than an optional
   backend extra.
6. Speed optimization is now documented separately in
   `CUPY_RSOXS_OPTIMIZATION_LEDGER.md` so timing history no longer dominates the
   repo-wide plan.

## Expert-Only Approximate 2D Z-Collapse Mode (Proposed April 4, 2026)

This section records a proposed expert-only approximation mode for
`cupy-rsoxs`. It is intentionally documented before implementation so a fresh
context can resume the work without reconstructing the design discussion.

### Purpose

The goal is to provide a fast approximate-physics path for arbitrary `3D`
morphologies by collapsing the locally composed field through `z` before the
FFT and then continuing the run as an effective `z=1` problem.

This is not a parity target. It is explicitly an opt-in approximation intended
for expert use when the user wants a faster qualitative or semi-quantitative
view of the scattering and accepts that:

1. deviations should increase when the morphology has stronger heterogeneity
   through `z`,
2. deviations should generally grow at higher `q`,
3. and agreement with the maintained full `3D` path is something to measure on
   representative cases rather than assume.

### Decisions already made

The following design decisions were explicitly selected on April 4, 2026 and
should be treated as the resume baseline:

1. This should be a new expert-only approximation option rather than a new
   `execution_path`.
2. The approximation should be available for arbitrary `3D` boxes, not just
   native `z=1` morphologies.
3. The initial collapse rule should be `mean` through `z`.
   - `sum` was discussed and may still be interesting later, but `mean` is the
     current working choice.
4. The collapse should happen before detector windowing / FFT work.
5. After the collapse, the simulation should proceed with effective-`2D`
   semantics as if the input problem were `z=1`.
6. The approximation should be available regardless of `EAngleRotation`.
7. The approximation is desired for both maintained `cupy-rsoxs` execution
   paths:
   - `tensor_coeff`
   - `direct_polarization`
8. Validation support should remain exploratory at first.
   - Do not add maintained tests for this mode until its deviation is judged
     useful on representative cases such as CoreShell and at least one
     form-factor / `I(q)` comparison surface.

### Corrected current implementation detail for effective `2D`

One earlier hypothesis was that the current `z=1` path might effectively
"just provide the FFT." That is not the current backend behavior.

What is true today:

1. the Hann factor in `z` is identity for `z=1`,
2. but the backend still runs detector-projection math on the single `qz=0`
   slice rather than returning a raw FFT panel.

This matters because the proposed approximation should initially be thought of
as:

1. compose the `3D` local field,
2. collapse it through `z`,
3. then continue with the current effective-`z=1` detector semantics,

not as "collapse through `z` and return the raw `2D` FFT."

### Intended option surface

Tentative preferred naming:

1. `backend_options={"z_collapse_mode": "mean"}`

Current design intent:

1. default remains off / `None`,
2. `z_collapse_mode` should remain orthogonal to `execution_path`,
3. `z_collapse_mode` should remain orthogonal to `mixed_precision_mode`,
4. and this mode should be documented as expert-only and approximation-only.

### Path-specific implementation intent

For `execution_path='tensor_coeff'`:

1. build the usual energy-specific local tensor coefficients,
2. collapse the required `Nt` component fields through `z` by `mean`,
3. treat the collapsed result as a reusable `2D` tensor field,
4. then continue through FFT and detector projection with effective-`z=1`
   semantics.

This is currently the cleaner first prototype target because it preserves the
existing reuse story of the maintained `tensor_coeff` architecture.

For `execution_path='direct_polarization'`:

1. build the usual angle-specific `p_x`, `p_y`, `p_z` fields,
2. collapse those polarization fields through `z` by `mean`,
3. then continue through FFT and detector projection with effective-`z=1`
   semantics.

This is still desired, but it is a slightly less clean first prototype because
the direct path remains more angle-local and therefore gains less reuse from
the collapsed intermediate.

### Recommended implementation order

If this proposal is resumed in a fresh context, the intended order is:

1. prototype the approximation in `tensor_coeff` first,
2. measure detector-image and `I(q)` deviation on representative cases,
3. then add the same approximation surface to `direct_polarization`,
4. and only after that decide whether the mode deserves maintained validation
   support or should remain a dev-only exploratory path.

### Separate but related optimization goal

The approximate `z`-collapse mode should not be conflated with the distinct
"effective `2D` detector simplification" goal.

That separate goal is:

1. detect when the input to detector projection is already effectively `2D`
   (`z=1` natively, and later also the collapsed approximation mode),
2. and route it through a simpler detector routine instead of the current more
   general detector-projection work.

That detector simplification should be treated as its own optimization thread,
but as a low-priority one for now, because it is potentially useful even
without the approximation mode and should preserve current effective-`2D`
semantics rather than redefine them.

## Test-First Program (Highest Priority)

### Immediate objective

Maintain a path-first `tests/validation/` pytest surface that runs the
maintained physics checks through explicit NRSS morphology/backend execution
instead of legacy CLI serialization or hidden backend defaults.

### Maintained validation cases

1. Analytical sphere form factor.
2. Sphere contrast scaling.
3. Sphere orientational contrast scaling.
4. Analytical 2D disk form factor.
5. 2D disk contrast scaling.
6. 2D and 3D Bragg lattice peak-position validation.
7. Core-shell.
8. MWCNT.

### Completed peer-path test refactor (April 4, 2026)

The maintained validation and smoke surfaces now treat the three prioritized
computation paths as first-class peers:

1. `legacy_cyrsoxs`
   - `backend="cyrsoxs"`
2. `cupy_tensor_coeff`
   - `backend="cupy-rsoxs"`
   - `backend_options={"execution_path": "tensor_coeff"}`
3. `cupy_direct_polarization`
   - `backend="cupy-rsoxs"`
   - `backend_options={"execution_path": "direct_polarization"}`

Implemented changes:

1. Added shared path metadata and fixtures for the maintained pytest surface,
   centered on `tests/path_matrix.py` plus the `nrss_path` fixture.
2. Extended test-routing plumbing so plain `pytest`, the local report, and the
   physics runner can execute per-path lanes rather than only a single
   `NRSS_BACKEND`-selected lane.
3. Converted maintained validation builders so they accept explicit
   `backend` / `backend_options` inputs anywhere a test is intended to run on
   more than one computation path.
4. Replaced backend-first maintained routing with path-oriented routing while
   retaining explicit legacy-only compatibility markers where scientifically
   justified.
5. Default path-matrix behavior now expands to all three peer paths.
6. `--nrss-path` / `NRSS_PATH` and `--nrss-backend` / `NRSS_BACKEND`
   disagreements now fail fast.
7. The maintained CuPy peer lanes use device residency after lightweight host
   vs device parity checks, with `ownership_policy="borrow"` for the standard
   CuPy matrix.

Completed maintained migration outcomes:

1. Current `backend_specific` tests are now split into one of two groups:
   - path-matrix tests that run on all three prioritized computation paths,
   - true compatibility tests that remain intentionally legacy-only.
2. `tests/validation/test_bragg_2d_lattice.py`,
   `tests/validation/test_bragg_3d_lattice.py`, and
   `tests/validation/test_mwcnt_reference.py` should become path-matrix
   physics tests. Their morphology builders should accept explicit
   `backend` / `backend_options` arguments and should be exercised in the
   `legacy_cyrsoxs`, `cupy_tensor_coeff`, and
   `cupy_direct_polarization` lanes. This is now complete.
3. `tests/validation/test_core_shell_reference.py` already has partial path
   plumbing and should be normalized into the same path-matrix structure so
   the maintained `cupy_tensor_coeff` and `cupy_direct_polarization` workflows
   are represented symmetrically rather than as a default-plus-extra pattern.
   This is now complete.
4. Tests that are currently marked `cyrsoxs_only` should be reviewed
   individually:
   - if they express general physics expectations, convert them into
     path-matrix tests with per-path thresholds if needed,
   - if they exercise true CyRSoXS-only compatibility behavior, move them into
     an explicit legacy-compatibility lane rather than leaving them in the
     principal peer-path physics matrix.
   This review is complete for the maintained validation surface.
5. Keep CLI-vs-pybind checks as explicit legacy compatibility tests. They
   remain valuable, but they should not define the maintained peer structure
   for the three primary computation paths.

Newly converted maintained physics modules:

1. `tests/validation/test_analytical_sphere_form_factor.py`
   - now a path-matrix physics test,
   - runs explicit NRSS `Morphology` backends on
     `legacy_cyrsoxs`, `cupy_tensor_coeff`, and
     `cupy_direct_polarization`,
   - compares the simulated flat-detector sphere signal against the analytical
     flat-detector reference through the maintained PyHyperScattering
     reduction, with pointwise and minima-alignment thresholds calibrated to
     the current maintained morphology runner.
2. `tests/validation/test_sphere_contrast_scaling.py`
   - now a path-matrix physics test,
   - keeps the one-morph, multi-energy sphere contrast-scaling design,
   - validates beta-only, delta-only, mixed, and split-material scaling on all
     three peer paths through backend-explicit morphology execution.
3. `tests/validation/test_sphere_orientational_contrast_scaling.py`
   - now a path-matrix physics test,
   - keeps the one-morph, multi-energy sphere orientational-contrast design,
   - validates the helper-predicted orientational ratios on all three peer
     paths through backend-explicit morphology execution.
4. `tests/validation/test_analytical_2d_disk_form_factor.py`
   - now a path-matrix physics test,
   - keeps the direct analytical 2D disk comparison through the maintained
     PyHyperScattering reduction,
   - runs the maintained explicit disk-versus-vacuum morphology on all three
     peer paths.
5. `tests/validation/test_2d_disk_contrast_scaling.py`
   - now a path-matrix physics test,
   - keeps the one-morph, multi-energy 2D contrast-scaling design,
   - runs the maintained backend-explicit morphology on all three peer paths.

Current maintained validation surface:

1. All maintained validation modules now participate in the peer-path
   matrix. The surface is nine pytest modules covering eight maintained
   validation cases:
   - analytical sphere form factor,
   - sphere contrast scaling,
   - sphere orientational contrast scaling,
   - analytical 2D disk form factor,
   - 2D disk contrast scaling,
   - 2D Bragg lattice,
   - 3D Bragg lattice,
   - CoreShell,
   - MWCNT.
2. The maintained validation surface comprises `14` physics tests per path.
3. There are no remaining maintained CuPy skips on the principal physics
   matrix.

### Path-matrix refactor details for resumption

This refactor is now implemented. The details below remain as the authoritative
design description for how the maintained path-first surface is structured.

The goal of this refactor is not just to add more parametrization. It is to
make the maintained test program structurally path-first so that a fresh
resumer can see, run, and extend the three prioritized computation paths
without reconstructing hidden defaults from the current backend-first routing.

#### Canonical path definition

Add a maintained shared path definition module, for example
`tests/path_matrix.py`, with one canonical object per peer computation path.

Recommended shape:

1. `id`
2. `backend`
3. `backend_options`
4. `category`
5. `supports_cli`
6. `supports_reference_parity`

Initial path set:

1. `legacy_cyrsoxs`
   - `backend="cyrsoxs"`
   - `backend_options={}`
   - `category="legacy"`
2. `cupy_tensor_coeff`
   - `backend="cupy-rsoxs"`
   - `backend_options={"execution_path": "tensor_coeff"}`
   - `category="cupy"`
3. `cupy_direct_polarization`
   - `backend="cupy-rsoxs"`
   - `backend_options={"execution_path": "direct_polarization"}`
   - `category="cupy"`

#### Required routing changes

Current test routing is centered on `tests/conftest.py` plus
`--nrss-backend` / `NRSS_BACKEND`. The refactor should preserve backend
selection compatibility while making the maintained route path-aware.

Planned changes:

1. Add `--nrss-path` to `tests/conftest.py`.
2. Add `NRSS_PATH` environment support, checked before implicit default
   backend resolution.
3. If no path selector is provided, expand maintained path-matrix tests to all
   three peer paths by default rather than collapsing to the backend default.
4. If `--nrss-path` / `NRSS_PATH` and `--nrss-backend` / `NRSS_BACKEND`
   disagree, fail fast instead of silently choosing one selector.
5. Add a session fixture `nrss_path`.
6. Add a parametrized maintained-path fixture for path-matrix tests so a plain
   `pytest` run expands those tests across all three peer paths.
7. Keep `nrss_backend` temporarily as a compatibility fixture during the
   migration, but treat it as a derived field from `nrss_path`.
8. Update pytest report headers so they show both the selected path id and the
   effective backend/backend-options pair.
9. Update collection-time skip logic so path-only and legacy-only tests route
   cleanly without relying on `backend_specific` as the main concept.

#### Scripts that must be updated

The path-first model is incomplete unless the standard scripts can run path
lanes directly. The following files are part of the required edit set:

1. `tests/conftest.py`
   - add path option/env parsing and `nrss_path` fixture
2. `scripts/run_local_test_report.sh`
   - add `--nrss-path`
   - support running repeated standard lanes per path
   - emit path-specific metadata and artifacts
3. `scripts/run_physics_validation_suite.py`
   - add `--nrss-path`
   - pass path metadata to row-level pytest invocations
4. optionally `pyproject.toml`
   - add/adjust markers if new maintained path markers are introduced

#### Maintained file migration order

To minimize risk, convert the maintained shared builders first, then their
tests, then the report scripts.

Recommended implementation order:

1. `tests/path_matrix.py` new shared path-definition module.
2. `tests/conftest.py` path fixture and routing compatibility layer.
3. `tests/validation/lib/bragg.py`
   - add `backend` / `backend_options` plumbing to both Bragg builders
4. `tests/validation/lib/mwcnt.py`
   - add `backend` / `backend_options` plumbing to the maintained MWCNT
     builder/run helpers
5. `tests/validation/lib/core_shell.py`
   - normalize existing partial path support into the same interface used by
     Bragg and MWCNT
6. `tests/validation/test_bragg_2d_lattice.py`
7. `tests/validation/test_bragg_3d_lattice.py`
8. `tests/validation/test_mwcnt_reference.py`
9. `tests/validation/test_core_shell_reference.py`
10. `tests/smoke/test_smoke.py`
11. `scripts/run_physics_validation_suite.py`
12. `scripts/run_local_test_report.sh`

#### Marker migration policy

Current markers encode the old model and should be reinterpreted explicitly.

1. `backend_specific`
   - stop using this as the primary maintained routing label
   - replace with one of:
     - path-matrix test,
     - path-subset test,
     - backend-contract test,
     - legacy-compatibility test
2. `cyrsoxs_only`
   - keep only for true legacy-compatibility behavior during migration
   - for general physics checks, convert away from this marker
3. `reference_parity`
   - keep this meaning intact
   - path conversion should not weaken reference-parity expectations
4. `physics_validation`, `experimental_validation`, and
   `toolchain_validation`
   - keep these categories intact
   - they describe test intent, not route selection

#### Test-by-test migration guidance

Validation surface:

1. `tests/validation/test_bragg_2d_lattice.py`
   - convert to a path-matrix physics test
   - run on `legacy_cyrsoxs`, `cupy_tensor_coeff`, and
     `cupy_direct_polarization`
   - accept `nrss_path` and pass explicit path settings into the Bragg builder
2. `tests/validation/test_bragg_3d_lattice.py`
   - same conversion as 2D Bragg
3. `tests/validation/test_mwcnt_reference.py`
   - convert to a path-matrix physics test
   - use shared maintained observables for all three peer paths
   - keep path-specific thresholds only if empirical variance justifies them
4. `tests/validation/test_core_shell_reference.py`
   - collapse the current asymmetry between the default cupy path and the
     explicit direct-path case
   - make the maintained experimental/sim-regression tests run through the
     path matrix
   - drop the extra strict/borrow CuPy stress cases unless they still provide
     unique coverage not already represented in the principal matrix

Smoke surface:

1. `test_pybind_runtime_tiny_deterministic_pattern`
   - convert from `backend_specific` to path-matrix smoke
2. `test_pyhyperscattering_integrator_to_xarray_smoke`
   - convert from `backend_specific` to path-matrix smoke
3. `test_pybind_runtime_2d_disk_smoke`
   - convert from `backend_specific` to path-matrix smoke
4. `test_eangle_rotation_endpoint_behavior_smoke`
   - convert from `backend_specific` to path-matrix smoke
5. CuPy backend-option contract tests
   - keep as implementation-specific tests
   - relabel as backend-contract or CuPy-only rather than peer-path tests
6. CLI-vs-pybind smoke tests
   - keep as legacy-compatibility tests
7. CyRSoXS object-lifecycle or pybind-ownership tests
   - keep as legacy-compatibility unless a parallel cupy-specific lifecycle
     requirement emerges

#### CuPy residency policy for peer-path lanes

The two maintained `cupy-rsoxs` peer paths should use device residency for the
standard smoke and physics matrix after lightweight parity checks establish
that host-resident and device-resident execution agree on the maintained
observables.

Required policy:

1. Add at least two simple parity checks before switching the shared CuPy lanes
   to device residency:
   - host vs device parity for `cupy_tensor_coeff`
   - host vs device parity for `cupy_direct_polarization`
2. Keep backend-contract tests that explicitly exercise host-side staging in
   host residency; do not erase host-resident coverage.
3. Do not reuse mutable `Morphology` instances across tests as a speed
   optimization. Runtime reuse may occur only at the pytest-process / imported
   CuPy-runtime level, with each test owning its own morphology and releasing
   runtime state normally.
4. Prefer CuPy-native fields plus `ownership_policy="borrow"` for maintained
   CuPy lanes when the builder can supply them without distorting the test
   intent.
5. When a builder still originates fields on the host, `resident_mode="device"`
   is still acceptable for the maintained CuPy matrix if construction performs
   a one-time transfer and the resulting lane remains deterministic.
6. Note the current limitation that `scripts/run_physics_validation_suite.py`
   executes each physics row in a separate pytest process, so this residency
   speedup applies within a row/runtime rather than across all maintained
   physics modules.

#### Expected lane structure after refactor

The standard report should expose the maintained program as:

1. one shared environment snapshot
2. one shared CPU contract lane
3. `legacy_cyrsoxs` GPU smoke lane
4. `legacy_cyrsoxs` physics-validation lane
5. `cupy_tensor_coeff` GPU smoke lane
6. `cupy_tensor_coeff` physics-validation lane
7. `cupy_direct_polarization` GPU smoke lane
8. `cupy_direct_polarization` physics-validation lane
9. optional legacy-compatibility lane for CLI-only checks if kept outside the
   main path matrix

Artifact layout should avoid lane collisions. Use per-path directories or
path-prefixed filenames under `test-reports/`.

#### Acceptance criteria

The refactor is only complete when the following are true:

1. a developer can select a maintained path directly via pytest and the two
   report scripts
2. all four maintained shared physics modules
   - `Bragg 2D`,
   - `Bragg 3D`,
   - `CoreShell`,
   - `MWCNT`
   run through the path matrix rather than through hidden default backend
   resolution
3. the maintained shared smoke surface has at least one deterministic 3D
   run, one deterministic 2D run, one PyHyperScattering reduction run, and one
   `EAngleRotation` run on each peer path
4. direct-polarization is no longer represented only by ad hoc extra tests;
   it appears as its own first-class standard lane
5. legacy CLI checks still exist, but they are clearly separated from the
   peer-path matrix
6. `pytest --collect-only` output and the local report summary make the active
   path visible without inspecting environment variables

#### Resume-here checklist

If this work is resumed in a fresh context, start from this exact sequence:

1. read `tests/conftest.py`
2. read `scripts/run_local_test_report.sh`
3. read `scripts/run_physics_validation_suite.py`
4. read `tests/validation/lib/core_shell.py`
5. read `tests/validation/lib/bragg.py`
6. read `tests/validation/lib/mwcnt.py`
7. convert the path fixture and path-definition module first
8. convert Bragg, MWCNT, and CoreShell builders next
9. convert the validation tests
10. convert smoke and report plumbing last

Do not start by mass-editing markers across the suite before the shared path
fixture and builder interfaces exist; that would create a partially migrated
state with ambiguous routing semantics.

### Required test qualities

1. Deterministic fixtures and fixed RNG seeds if randomness appears anywhere.
2. Explicit metadata capture (versions, geometry, dtype, parameter hashes,
   backend flags).
3. Machine-readable golden references generated from trusted pybind runs.
4. CPU smoke is intentionally limited to validator, I/O, and API-contract
   behavior; it is not a CPU physics-parity lane.

### Parity metrics (layered)

1. Objective scalar parity (for fitting-style workflows): target <= 1% relative
   error.
2. Radial I(q) parity with q-window-specific rtol/atol.
3. Peak-position parity with absolute q tolerance.
4. Optional image-space checks on masked finite support.

Initial threshold table is intentionally provisional; calibrate from empirical
baseline variance before final gating.

### Golden data governance

1. Generate now from trusted current reference.
2. Regenerate only when confirmed physics/scientific bug fixes intentionally
   change expected output.
3. Require changelog note + reviewer signoff for any golden update.

### Analytical guardrail track (projected sphere)

1. This track is now implemented as
   `tests/validation/test_analytical_sphere_form_factor.py`.
2. Current implementation uses pybind execution plus PyHyperScattering
   reduction, compares against a flat-detector analytical reference, and
   evaluates both pointwise agreement and all-minima alignment.
3. Geometry is currently fixed at `512^3`, `PhysSize = 1.0 nm`, diameters
   `70 nm` and `128 nm`, with optional superresolution support retained for
   future follow-up.
4. Treat this as a guardrail rather than strict equality because
   discretization and finite resolution still perturb high-q behavior.

### Implemented smoke harness (March 17, 2026)

1. Added `tests/smoke/test_smoke.py` for deterministic environment/import
   checks, morphology validation checks, pybind runtime coverage,
   PyHyperScattering integration, CLI-vs-pybind parity smoke, and GPU
   config/E-angle semantics smoke.
2. Added `scripts/run_local_test_report.sh` to standardize local execution and
   emit timestamped metadata/log/summary artifacts under `test-reports/`.
   - Default conda env is now `nrss-dev` unless overridden with `-e/--env` or
     `NRSS_TEST_ENV`.
   - Standard lanes can be skipped with `--skip-defaults`.
   - Explicit `--cmd` entries can be repeated with `--repeat N` for brittleness
     sweeps and injected-build validation.
3. Added pytest marker declarations in `pyproject.toml` for `smoke`, `cpu`,
   `gpu`, `slow`, `physics_validation`, `experimental_validation`, and
   `toolchain_validation`.
4. Added `tests/conftest.py` to default tests to a single visible GPU when the
   environment is otherwise unset, improving reproducibility and avoiding known
   CyRSoXS multi-GPU instability during energy fan-out.

Latest run evidence:

1. Command: `bash scripts/run_local_test_report.sh --stop-on-fail`
2. Timestamp (UTC): `20260320T134227Z`
3. Result: `4/4` steps passed
4. CPU smoke: `12 passed, 10 deselected`
5. GPU smoke: `10 passed, 12 deselected`
6. Physics validation: this early snapshot is superseded by the later expanded
   lane below.

### Implemented physics validation layer (March 20, 2026)

1. Added `tests/validation/test_analytical_sphere_form_factor.py`:
   - now a maintained path-matrix physics test,
   - runs explicit NRSS morphology execution on
     `legacy_cyrsoxs`, `cupy_tensor_coeff`, and
     `cupy_direct_polarization`,
   - compares against the flat-detector analytical sphere reference through the
     maintained PyHyperScattering reduction,
   - uses the currently calibrated pointwise and minima-alignment thresholds
     for the maintained morphology runner,
   - optional plot writing gated by `NRSS_WRITE_VALIDATION_PLOTS=1`.
2. Added `tests/validation/test_sphere_contrast_scaling.py`:
   - now a maintained path-matrix physics test,
   - one-morph, multi-energy contrast-scaling validation,
   - 24 close-energy scenarios covering beta-only, delta-only, mixed, and
     split-material families,
   - integrated-intensity checks over a fixed q window with fixed empirical
     thresholds on all three peer paths.
3. Added `tests/validation/lib/orientational_contrast.py`:
   - reusable tensor-based helper that turns para/perp delta/beta channels plus
     Euler angles and `S` into inspectable effective indices, induced
     polarization vectors, and Eq. 15/16-style far-field contrast predictions,
   - explicitly documents the How-to-RSoXS citation plus the rotation /
     far-field projection path used for expectations.
4. Added `tests/validation/test_sphere_orientational_contrast_scaling.py`:
   - now a maintained path-matrix physics test,
   - one-morph, multi-energy orientational-contrast validation for a sphere in
     vacuum,
   - `128 x 128 x 128`, `PhysSize = 2.0 nm`, `Diameter = 32 nm`,
   - close-energy pure-delta, pure-beta, and mixed dichroic families,
   - high-symmetry `theta` and `psi` coverage, low-symmetry coupled Euler
     cases, and an `S` series including `S=0`,
   - helper-driven expected ratios plus direct detector-annulus observed
     ratios on all three peer paths,
   - optional plot writing through `NRSS_WRITE_VALIDATION_PLOTS=1`.
5. Added `tests/validation/test_analytical_2d_disk_form_factor.py`:
   - now a maintained path-matrix physics test,
   - direct analytical 2D disk comparison through the maintained
     PyHyperScattering reduction,
   - `1 x 2048 x 2048`, `PhysSize = 1.0 nm`, diameters `70 nm` and `128 nm`,
   - pointwise and minima-alignment metrics with fixed empirical thresholds,
   - explicit disk-versus-vacuum morphology,
   - fixed `sr=1` only as the assertion anchor while retaining the maintained
     superresolution loop for diagnostics,
   - optional plot writing gated by `NRSS_WRITE_VALIDATION_PLOTS=1`.
6. Added `tests/validation/test_2d_disk_contrast_scaling.py`:
   - now a maintained path-matrix physics test,
   - one-morph, multi-energy contrast-scaling validation for the 2D pathway,
   - `1 x 2048 x 2048`, `PhysSize = 1.0 nm`,
   - 24 close-energy scenarios covering beta-only, delta-only, mixed, and
     split-material families,
   - integrated-intensity checks over a fixed q window with fixed empirical
     thresholds on all three peer paths.
7. Added `tests/validation/lib/bragg.py`:
   - shared deterministic lattice builders and reciprocal-space prediction
     helpers for Bragg validation,
   - supports square/hexagonal 2D disk lattices and simple-cubic/HCP 3D sphere
     lattices,
   - keeps explicit vacuum as the second material and uses float-center local
     stamping for morphology construction.
8. Added `tests/validation/test_bragg_2d_lattice.py`:
   - deterministic square (`a = 30 nm`) and hexagonal (`a = 45 nm`) disk
     lattices at `1 x 2048 x 2048`, `PhysSize = 1.0 nm`,
   - now runs as a maintained path-matrix physics test on all three peer
     paths,
   - validates detector-peak locations and quasi-powder shell locations through
     the maintained PyHyperScattering reduction,
   - includes verbose diagnostic plots with full predicted-shell overlays.
9. Added `tests/validation/test_bragg_3d_lattice.py`:
   - deterministic simple-cubic (`a = 30 nm`) and ideal HCP (`a = 45 nm`)
     sphere lattices at `256 x 1024 x 1024`, `PhysSize = 1.0 nm`,
   - now runs as a maintained path-matrix physics test on all three peer
     paths,
   - validates detector-visible 3D Bragg peak locations plus azimuthally
     averaged shell locations,
   - uses explicit flat-detector geometry handling for shell prediction and
     includes verbose diagnostic plots with visibility-class overlays.
10. Added `tests/validation/lib/core_shell.py` plus
    `tests/validation/test_core_shell_reference.py`:
   - maintained CoreShell baseline workflow through explicit NRSS morphology
     execution + PyHyperScattering `WPIntegrator` + manual A-wedge reduction,
   - now runs as a maintained path-matrix physics test on all three peer
     paths,
   - experimental PGN RSoXS golden as the scientific gate,
   - parallel sim-derived golden as a tight regression guard,
   - `experimental_validation` marker applied to the experimental-reference
     test,
    - falsification/subterfuge scenarios intentionally kept only in the
      development diagnostic, not in the principal `tests/validation` surface.
11. Added `tests/validation/lib/mwcnt.py` plus
    `tests/validation/test_mwcnt_reference.py`:
   - maintained deterministic MWCNT workflow through explicit NRSS morphology
     execution + PyHyperScattering `WPIntegrator` + anisotropy-observable
     reduction,
   - now runs as a maintained path-matrix physics test on all three peer
     paths,
   - periodic field construction is now the maintained default and the legacy
     field path remains available as an explicit switch,
    - maintained simulation defaults are `WindowingType=0` and
      `EAngleRotation=[0, 20, 340]`,
    - experimental reduced `A(E)` / `A(q)` observables derived from the
      tutorial/manuscript workflow are the scientific gate,
    - `experimental_validation` marker applied to the official MWCNT test,
    - manuscript Table I provenance plus realized fixed-seed geometry
      statistics are exposed in the maintained validation plots and helper
      metadata,
    - development-only MWCNT falsification/threshold probes stay under
      `tests/validation/dev/`, not in the principal `tests/validation`
      surface.
12. Archived one-off exploratory validation code under
    `scripts/validation_diagnostics/` so it remains available for future
    archaeology without polluting pytest collection.
    - this directory now also holds
      `orientational_contrast_tiny_diagnostic.py`, the development-only
      preserved `64^3` probe that preceded the official orientational test,
    - and `sphere_orientational_contrast_diagnostic.py`, an opt-in artifact
      generator that writes orientational ratio plots plus TSV summaries under
      `test-reports/sphere-orientational-contrast-dev/`,
    - and `core_shell_reference_diagnostic.py`, the opt-in CoreShell artifact
      generator that also owns the falsification/subterfuge comparisons.
13. Extended `scripts/run_local_test_report.sh` to include the marker-based
    `physics_validation` lane in the standard local report, while also
    supporting `--skip-defaults` plus repeated explicit `--cmd` runs for
    targeted validation and stochastic-failure checks. Newly added physics
    modules are therefore included automatically.
    - physics-test report summaries now retain full docstring descriptions
      rather than only the first line,
    - and targeted custom physics commands now resolve per-test statuses in the
      markdown report instead of falling back to `DESELECTED`,
    - the report now also captures imported NRSS module resolution plus a
      hashed manifest of vendored validation-reference artifacts under
      `tests/validation/data/`.
14. Targeted local validation against an injected fixed CyRSoXS pybind build
    removed the prior same-process 2D analytical disk stochastic failure in
    local testing:
    - one-process back-to-back `70 nm` then `128 nm` analytical 2D disk
      validation passed `20/20` repeated runs on a single visible GPU,
    - the shipped pytest module also passed cleanly against the injected build,
    - interpret this as local evidence that the 2D-path failure was upstream to
      NRSS rather than a remaining deterministic NRSS harness issue.
15. Latest default local report evidence for the expanded suite:
    - command:
      `CUDA_VISIBLE_DEVICES=0 bash scripts/run_local_test_report.sh --stop-on-fail`,
    - timestamp/report: `20260322T140310Z` / `test-reports/20260322T140310Z`,
    - result: `4/4` steps passed,
    - physics-validation lane inside the report: `14 passed`, including both
      CoreShell tests and the new MWCNT experimental test,
    - the generated markdown summary lists both experimental-reference tests
      with their `experimental_validation` markers and full scientific citation
      blocks.
16. A targeted local CoreShell-only run confirmed the new official module
    passes cleanly on its own:
    - command:
      `CUDA_VISIBLE_DEVICES=0 /home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/validation/test_core_shell_reference.py -v`,
    - result: `2 passed`.
17. A targeted local MWCNT-only run confirmed the new official module passes
    cleanly on its own:
    - command:
      `CUDA_VISIBLE_DEVICES=0 /home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/validation/test_mwcnt_reference.py -v`,
    - result: `1 passed`,
    - a development-only threshold probe showed that nearby radius
      falsifications fail the maintained thresholds while a moderate orientation
      broadening can still pass, so no threshold tightening was applied.
18. A targeted installed-build report run also confirmed that the new
    orientational module is described correctly in `summary.md`:
    - command:
      `bash scripts/run_local_test_report.sh --skip-defaults --cmd "python -m pytest tests/validation/test_sphere_orientational_contrast_scaling.py -m physics_validation -v"`,
    - timestamp/report: `20260321T190619Z` / `test-reports/20260321T190619Z`,
    - result: `1 passed`,
    - the "Physics Tests" section now includes the full orientational test
      description and a `PASSED` status.
19. Installed-package cross-check for the earlier Bragg coverage also passed:
    - command:
      `CUDA_VISIBLE_DEVICES=1 /home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/validation/test_bragg_2d_lattice.py tests/validation/test_bragg_3d_lattice.py -v`,
    - result: `4 passed in 126.03s`,
    - installed package resolved to `CyRSoXS 1.1.8.0`, patch `9d45790`.
20. Completed the remaining path-matrix conversion of the five formerly
    legacy-only maintained physics modules:
    - `test_analytical_sphere_form_factor.py`,
    - `test_sphere_contrast_scaling.py`,
    - `test_sphere_orientational_contrast_scaling.py`,
    - `test_analytical_2d_disk_form_factor.py`,
    - `test_2d_disk_contrast_scaling.py`.
21. Verified the newly enabled maintained path-matrix tests on all three peer
    computation paths after threshold review:
    - `legacy_cyrsoxs`: `7 passed`,
    - `cupy_tensor_coeff`: `7 passed`,
    - `cupy_direct_polarization`: `7 passed`,
    - total for the newly enabled set: `21/21 passed`.
22. Current maintained physics-matrix status:
    - the principal validation surface is now `14` tests per path,
    - no maintained physics tests remain skipped for the CuPy peer paths,
    - path-matrix tests expand across all three peer paths by default when no
      explicit selector is provided.

### Remaining test-hardening gaps

1. Add CI gating policy for CPU smoke and GPU smoke/parity/physics lanes.
2. Keep the CPU smoke lane focused on validator, I/O, and API-contract behavior
   rather than CPU physics parity.
3. Add backend-contract tests only when alternate backend implementation work
   begins.

## Packaging and Environment Direction

Current implemented state:

1. CuPy is a required maintained runtime dependency because `cupy-rsoxs` is the
   default backend and the primary pure-Python backend implementation.
2. `pyproject.toml` now carries `cupy-cuda12x` as the maintained base CuPy
   dependency line.
3. `environment.yml` also pins `cupy-cuda12x` in the maintained conda
   environment.
4. `cyrsoxs` remains a supported legacy/reference backend, but it is no longer
   the default backend when both backends are available.
5. If the project later needs to move to a different supported CuPy package
   line, update `pyproject.toml`, `environment.yml`, and install docs together.
6. Historical notes later in this document still record the earlier
   extras-based packaging state as a point-in-time archive; they are not the
   current policy.

## Multi-GPU Execution Model

Primary production pattern for this project:

1. Model-parallel execution (one model per GPU worker), typically via Ray.
2. Persistent workers for memory/plan reuse and lower startup overhead.
3. Keep internal energy-parallel multi-GPU optional and non-default.

Rationale:

1. Matches current objective-evaluation workload.
2. Avoids known instability concerns in CyRSoXS internal multi-GPU energy
   splitting.

Phase-1 parity lock-in:

1. `cupy-rsoxs` parity development and required parity tests target single-GPU
   execution only.
2. Internal multi-GPU energy fan-out is explicitly deferred until after parity
   because current CyRSoXS multi-GPU behavior is known to be unstable in some
   workflows.

## Runtime Observability and Safety Rails

Repo-wide expectations:

1. Log explicit host<->device transfers at NRSS-controlled boundaries.
2. Add strict-mode warnings for policy-driven conversions and make
   resident-mode assumptions visible in development diagnostics.
3. Keep development instrumentation opt-in and fully disableable for non-dev
   runs.
4. Treat structural memory ownership as the first-line control mechanism:
   reuse scratch buffers, release dead intermediates promptly, and use pool
   trimming only as an explicit lifecycle action rather than as a hot-path
   substitute.
5. Continue memory-observability follow-up for both:
   - post-run cleanup behavior,
   - peak and per-stage resident usage during compute, especially around
     polarization, FFT, and Ewald/result stages.

Detailed `cupy-rsoxs` timing segmentation and benchmark methodology now live in
`CUPY_RSOXS_OPTIMIZATION_LEDGER.md`.

## Development Workflow

### Source control

1. Use feature branches.
2. Keep commits small and frequent.
3. Merge milestone-sized PRs.

### Environment practice

1. Keep stable scientific environment untouched.
2. Use dedicated dev environment(s) with editable NRSS install.
3. Pin versions during parity development.

### Notebook and staging workflow

1. Use notebooks for exploratory visualization and diagnostics.
2. Keep an external staging workspace for baseline generation and comparison
   artifacts.
3. Require script-based reproducibility for official golden generation and
   CI-bound regression assets.

## Phased Delivery Plan

1. Test-hardening milestone: pybind golden baselines + deterministic harness.
2. Phase 1: CuPy backend that mimics CyRSoXS algorithmic flow.
   - Phase 1a: low-memory (`AlgorithmType=1`) implementation first, because it
     is the best immediate path for GPU headroom learning and memory
     discipline.
   - Phase 1b: communication-minimizing (`AlgorithmType=0`) implementation to
     parity using lessons from the low-memory path.
   - Phase 1 completion requires both algorithm paths to be runnable and
     parity-tested, even if one lands first.
   - Phase 1 parity scope is Euler-only, `ScatterApproach::PARTIAL` only, and
     single-GPU only.
3. Phase 2: Internal math cleanup (tensor-character refactor) while preserving
   parity tests.
4. Phase 3: Additional backends behind shared backend contract.

Independent success criterion:

1. Completion of the test-hardening milestone is a meaningful modernization
   outcome even if backend phases are delayed.

## Explicit Non-Goals (Initial Phases)

1. Immediate production biaxial feature release.
2. Optimization-first changes before parity harness exists.
3. Simultaneous full parity for every legacy execution mode on day one.

## Open Decisions for Follow-Up

1. Final parity threshold table by metric and q-region.
2. Golden dataset size/retention strategy in repository vs external artifacts.
3. GPU CI strategy and minimum gating matrix.
4. Release performance gates (what is measured and acceptable drift).
5. Detailed objective-function API and on-device return contract for fitting
   pipelines.

## Historical Backend-Prep Implementation Stages

The original `BACKEND_UPGRADE_GUIDE.md` was explicitly prep-only. It set the
architecture for multiple backends and the first NRSS-native backend, but it
did not initially implement the new backend compute engine itself.

Historical prep-only non-goals:

1. Do not implement `cupy-rsoxs` compute kernels or full simulation math here.
2. Do not redesign the science model beyond what is needed for backend
   decoupling.
3. Do not require multiple backend bindings per `Morphology` instance.
4. Do not force parity-with-CyRSoXS as a permanent rule for every future
   backend.
5. Do not overhaul all output handling yet; only add enough scaffolding to
   avoid blocking future GPU-native outputs.

Historical prep stages:

### Stage 0. Planning and handoff

Deliverables:

1. this guide
2. explicit decisions captured in-repo

Status:

1. required before all subsequent work

### Stage 1. Import-safe backend discovery

Deliverables:

1. new backend registry module(s)
2. public functions to inspect known/available backends
3. import-safe NRSS package even when CyRSoXS is absent

Success criteria:

1. `import NRSS` no longer requires CyRSoXS to be installed
2. `available_backends()` returns structured availability information

### Stage 2. Backend-selected `Morphology`

Deliverables:

1. `Morphology(..., backend=..., input_policy=...)`
2. default backend resolution using the locked rules above
3. eager normalization of material fields for the selected backend
4. clear failure for unknown/unavailable backends

Success criteria:

1. existing default `cyrsoxs` behavior still works
2. `Morphology` construction no longer depends on top-level CyRSoXS import

### Stage 3. CyRSoXS compatibility preservation

Deliverables:

1. preserve current `cyrsoxs` methods/attrs for existing users
2. route CyRSoXS-only work through lazy imports and backend checks

Success criteria:

1. current smoke/validation paths continue to work with `backend="cyrsoxs"`

### Stage 4. Backend-aware test routing

Deliverables:

1. pytest backend option/env
2. skip/routing rules for `cyrsoxs_only` and `reference_parity`
3. test-marker updates in smoke/validation suites
4. avoid top-level hard import failures in test modules that currently import
   `CyRSoXS` directly

Success criteria:

1. backend selection can route tests naturally
2. selecting a non-CyRSoXS backend does not crash collection because a module
   imported `CyRSoXS` too early

### Stage 5. Report-script integration

Deliverables:

1. `run_local_test_report.sh` backend option/env support
2. `run_physics_validation_suite.py` backend option/env support
3. report metadata includes selected backend

Success criteria:

1. developers can run targeted validation for a specific backend with the same
   report tooling

### Stage 6. Future work after prep

Not part of the prep milestone, but enabled by it:

1. implement `cupy-rsoxs`
2. add parity tests between `cupy-rsoxs` and CyRSoXS pybind
3. expand output policy
4. add transfer-observability tests
5. add additional backends

## Historical Backend-Prep Milestone Snapshot (2026-03-22)

### Completed stages

1. Stage 0: complete
   - this guide was created as the resumable backend-prep handoff
2. Stage 1: complete
   - `src/NRSS/backends/registry.py` now provides import-safe backend discovery
   - `src/NRSS/backends/__init__.py` and `src/NRSS/__init__.py` expose public
     backend-inspection helpers
   - `import NRSS` no longer requires CyRSoXS to be importable
3. Stage 2: substantially complete for prep
   - `Morphology(..., backend=..., input_policy=..., output_policy=...)` now
     resolves backend selection without top-level CyRSoXS import
   - material fields are now normalized eagerly at `Morphology` construction
     time for the selected backend contract
   - `Morphology(..., backend_options=...)` now validates and normalizes
     backend-specific options at construction time
   - backend-specific option handling is now part of the explicit contract layer
     instead of being hardcoded inside array coercion; this should support
     narrow surfaces like `execution_path` and the named
     `mixed_precision_mode`, and the old generic backend `dtype` knob should be
     considered stale
   - `input_policy='strict'` now fails early when normalization would require
     dtype/layout/device coercion
   - unsupported input array types now fail cleanly during `Morphology`
     construction instead of surfacing as opaque lower-level failures later
4. Stage 3: complete for current `cyrsoxs` preservation scope
   - `src/NRSS/backends/runtime.py` now defines the internal backend runtime
     seam for `prepare`, `run`, and `validate_all`
   - `src/NRSS/backends/cyrsoxs.py` now implements the legacy `cyrsoxs`
     runtime adapter and centralizes CyRSoXS import/helper logic
   - `Morphology.prepare()`, `run()`, and `validate_all()` now dispatch through
     the runtime seam rather than hardcoded backend branches
   - `create_inputData`, `create_optical_constants`, `create_voxel_data`,
     `create_update_cy`, `validate_all`, and `run` remain available for the
     `cyrsoxs` backend
   - CyRSoXS imports are lazy and backend-gated
5. Stage 4: complete for the current test suite routing scope
   - `tests/conftest.py` now supports `--nrss-backend` and `NRSS_BACKEND`
   - backend-aware skip/routing is implemented for `backend_specific`,
     `cyrsoxs_only`, and `reference_parity`
   - validation modules that previously broke collection due to eager marker
     tuples were fixed
   - lazy CyRSoXS import helper added at
     `tests/validation/lib/lazy_cyrsoxs.py`
6. Stage 5: complete
   - `scripts/run_local_test_report.sh` now threads backend selection through
     local report steps
   - `scripts/run_physics_validation_suite.py` now accepts and propagates
     `--nrss-backend`

### Files changed in the landed prep work

1. `BACKEND_UPGRADE_GUIDE.md`
2. `pyproject.toml`
3. `environment.yml`
4. `scripts/run_local_test_report.sh`
5. `scripts/run_physics_validation_suite.py`
6. `src/NRSS/__init__.py`
7. `src/NRSS/backends/__init__.py`
8. `src/NRSS/backends/arrays.py`
9. `src/NRSS/backends/contracts.py`
10. `src/NRSS/backends/cyrsoxs.py`
11. `src/NRSS/backends/registry.py`
12. `src/NRSS/backends/runtime.py`
13. `src/NRSS/morphology.py`
14. `tests/conftest.py`
15. `tests/smoke/test_smoke.py`
16. `tests/validation/lib/lazy_cyrsoxs.py`
17. `tests/validation/test_2d_disk_contrast_scaling.py`
18. `tests/validation/test_analytical_2d_disk_form_factor.py`
19. `tests/validation/test_analytical_sphere_form_factor.py`
20. `tests/validation/test_bragg_2d_lattice.py`
21. `tests/validation/test_bragg_3d_lattice.py`
22. `tests/validation/test_core_shell_reference.py`
23. `tests/validation/test_mwcnt_reference.py`
24. `tests/validation/test_sphere_contrast_scaling.py`
25. `tests/validation/test_sphere_orientational_contrast_scaling.py`

### Verification completed

1. `python -m py_compile` on modified runtime/test modules
2. baseline full local report:
   - `./scripts/run_local_test_report.sh -e nrss-dev --nrss-backend cyrsoxs`
   - result: `4/4 steps passed`
   - CPU smoke: `22 passed, 10 deselected`
   - GPU smoke: `10 passed, 22 deselected`
   - physics validation: `14 passed`
3. post-change full local report:
   - `./scripts/run_local_test_report.sh -e nrss-dev --nrss-backend cyrsoxs --no-plots`
   - result: `4/4 steps passed`
   - CPU smoke: `22 passed, 12 deselected`
   - GPU smoke: `12 passed, 22 deselected`
   - physics validation: `14 passed`
4. authoritative post-backend-options full local report:
   - `./scripts/run_local_test_report.sh -e nrss-dev --nrss-backend cyrsoxs --no-plots`
   - report dir: `test-reports/20260322T195347Z`
   - result: `4/4 steps passed`
   - CPU smoke: `23 passed, 13 deselected`
   - GPU smoke: `13 passed, 23 deselected`
   - physics validation: `14 passed`
5. `/home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/smoke -m 'not gpu' -q`
   - result: `23 passed, 13 deselected`
6. `/home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/smoke -m 'gpu' -q`
   - result: `13 passed, 23 deselected`
7. `/home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/smoke/test_smoke.py -m 'backend_agnostic_contract and not gpu' -q`
   - result: `16 passed, 16 deselected`
8. `/home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/smoke/test_smoke.py -m 'cyrsoxs_only and not gpu' -q`
   - result: `3 passed, 29 deselected`
9. `/home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/smoke/test_smoke.py -k 'backend_registry_reports_known_backends or planned_cupy_backend_array_contract or cyrsoxs_backend_rejects_non_default_dtype_option or normalizes_material_arrays_eagerly_for_selected_backend' -q`
   - result: `5 passed, 31 deselected`
10. `/home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/smoke/test_smoke.py -k 'normalizes_material_arrays_eagerly or strict_input_policy or unrecognized_array_types or coerces_non_float_field' -q`
    - result: `4 passed, 28 deselected`
11. `/home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/smoke/test_smoke.py -k 'cupy_import_available or planned_cupy_backend_array_contract or cyrsoxs_morphology_normalizes_cupy_inputs' -q`
    - result: `5 passed, 31 deselected`
12. `/home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/smoke/test_smoke.py -k 'backend_registry_reports_known_backends or pybind_morphology_object_lifecycle_smoke or pybind_runtime_tiny_deterministic_pattern or cyrsoxs_morphology_normalizes_cupy_inputs_to_host_contract' -q`
    - result: `4 passed, 32 deselected`
13. These command fragments are historical verification snapshots only.
    They are not the authority for the current `cupy-rsoxs` backend-option
    surface, which is documented in `CUPY_RSOXS_BACKEND_SPEC.md`.
14. `/home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/validation --collect-only -q`
    - result: `14 tests collected`
15. latest full compatibility report after the runtime-interface refactor:
    - `./scripts/run_local_test_report.sh -e nrss-dev --nrss-backend cyrsoxs --no-plots`
    - report dir: `test-reports/20260322T202539Z`
    - result: `4/4 steps passed`
    - CPU smoke: `23 passed, 13 deselected`
    - GPU smoke: `13 passed, 23 deselected`
    - physics validation: `14 passed`

### Important note

1. `test-reports/20260322T194742Z` should be treated as stale and ignored for
   status purposes.
2. That run started before the CuPy memory cleanup fix in the GPU smoke path
   and is not the authoritative post-change report.
3. `test-reports/cupy-rsoxs-optimization-dev/verify_cli_small_postcleanup/summary.json`
   is the first authoritative post-cleanup timing snapshot for resumed
   optimization work.

### Remaining intentionally deferred or unresolved items

1. `cupy-rsoxs` compute/runtime implementation now exists; timing cleanup is in
   place for the current optimization loop, and the open work is resident-mode
   refinement, segment-targeted optimization, export timing, and deeper memory
   instrumentation follow-up. See `CUPY_RSOXS_OPTIMIZATION_LEDGER.md` for the
   current detailed state.
   - current next-pass optimization focus is Segment `E` rotation and angle
     accumulation on the maintained `tensor_coeff` path,
   - accepted Segment `E` changes now require the maintained CoreShell
     sim-regression physics gates on both the default host-resident and device
     strict/borrow `cupy-rsoxs` workflows because that validation exercises
     the maintained full-rotation CoreShell workflow.
2. The approved mixed-precision implementation plan now lives in:
   - `CUPY_RSOXS_BACKEND_SPEC.md` for the stable option surface and execution
     contract,
   - `CUPY_RSOXS_OPTIMIZATION_LEDGER.md` for the implementation order, timing
     interpretation, and CoreShell graphical-abstract plan.
   - current lock-ins:
     - no exposed generic backend `dtype` option for `cupy-rsoxs`,
     - `mixed_precision_mode` remains orthogonal to `execution_path`,
     - both `tensor_coeff` and `direct_polarization` are expected to carry
       strict authoritative `float16` morphology inputs through authoritative
       normalization and runtime staging,
     - the dev harnesses must be able to generate the strict `float16` inputs
       required by that mode.
   - progress recorded on April 4, 2026:
     - the initial runtime/contracts/tests implementation pass is now landed,
     - the public `cupy-rsoxs` reduced-precision surface is
       `mixed_precision_mode='reduced_morphology_bit_depth'`,
     - mixed mode now enforces strict authoritative `float16` inputs and the
       mixed-mode voxelwise closure budget,
     - both supported execution paths now implement the intended morphology
       precision ladder through FFT ingress compute,
     - current documented `tensor_coeff` implementation widens half inputs
       during `Nt` construction and FFTs the resulting `complex64 Nt`,
     - that exact internal widening point is documented implementation state,
       not yet a separately maintained test-backed contract,
     - the maintained smoke suite passes with the new mixed-mode coverage,
     - maintained validation-surface expansion and graphical-abstract/dev-harness
       work remain intentionally deferred to a later phase.
3. The principal cross-backend primary-time comparison now lives at
   `tests/validation/dev/core_shell_backend_performance/run_primary_backend_speed_comparison.py`
   and emits a combined summary, TSV, and PNG table for the fixed single-energy
   CoreShell comparison panel.
4. The legacy full-energy backend-comparison harness under
   `tests/validation/dev/core_shell_backend_performance/` is no longer the
   authoritative timing harness for optimization work.
5. Backend-native result/output policy is still scaffolding only; current run
   behavior remains xarray/NumPy-oriented for parity and comparison workflows.
6. Serialization and write helpers remain effectively NumPy/CyRSoXS-oriented,
   which is acceptable for prep but will need review once a non-CyRSoXS runtime
   starts emitting backend-native arrays.
7. Resident-mode control now ships as the public `resident_mode` API surface
   for choosing host-resident vs device-resident authoritative morphology
   behavior.
8. Package/dependency policy for CuPy was later tightened after this historical
   snapshot:
   - this March 22, 2026 snapshot still reflects the older extras-based
     metadata state,
   - the current repo-wide policy is documented above in
     "Packaging and Environment Direction",
   - keep this note only as historical context for why older prep-era text may
     still mention extras or packaging-standard limitations
9. If a future environment has no runnable backend at all, `Morphology`
   construction still fails cleanly rather than creating a backend-less object;
   this is consistent with the current one-backend-per-instance decision.
