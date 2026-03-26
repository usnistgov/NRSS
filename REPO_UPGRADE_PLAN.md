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

## Test-First Program (Highest Priority)

### Immediate objective

Convert `tests/validation/` legacy scripts into robust pytest suites using
pybind CyRSoXS execution where applicable (no CLI serialization bottleneck),
while establishing a first stable physics-validation lane before backend
refactors.

### Maintained validation cases

1. Analytical sphere form factor.
2. Sphere contrast scaling.
3. Sphere orientational contrast scaling.
4. Analytical 2D disk form factor.
5. 2D disk contrast scaling.
6. 2D and 3D Bragg lattice peak-position validation.
7. Core-shell.
8. MWCNT.

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
   - flat-detector analytical sphere comparison through the pybind-to-PyHyper
     workflow,
   - pointwise and minima-alignment metrics with fixed empirical thresholds,
   - explicit sphere-versus-vacuum morphology,
   - optional plot writing gated by `NRSS_WRITE_VALIDATION_PLOTS=1`.
2. Added `tests/validation/test_sphere_contrast_scaling.py`:
   - one-morph, multi-energy contrast-scaling validation,
   - 24 close-energy scenarios covering beta-only, delta-only, mixed, and
     split-material families,
   - integrated-intensity checks over a fixed q window with fixed empirical
     thresholds.
3. Added `tests/validation/lib/orientational_contrast.py`:
   - reusable tensor-based helper that turns para/perp delta/beta channels plus
     Euler angles and `S` into inspectable effective indices, induced
     polarization vectors, and Eq. 15/16-style far-field contrast predictions,
   - explicitly documents the How-to-RSoXS citation plus the rotation /
     far-field projection path used for expectations.
4. Added `tests/validation/test_sphere_orientational_contrast_scaling.py`:
   - one-morph, multi-energy orientational-contrast validation for a sphere in
     vacuum,
   - `128 x 128 x 128`, `PhysSize = 2.0 nm`, `Diameter = 32 nm`,
   - close-energy pure-delta, pure-beta, and mixed dichroic families,
   - high-symmetry `theta` and `psi` coverage, low-symmetry coupled Euler
     cases, and an `S` series including `S=0`,
   - helper-driven expected ratios plus direct detector-annulus observed
     ratios,
   - optional plot writing through `NRSS_WRITE_VALIDATION_PLOTS=1`.
5. Added `tests/validation/test_analytical_2d_disk_form_factor.py`:
   - direct analytical 2D disk comparison through the pybind-to-PyHyper
     workflow,
   - `1 x 2048 x 2048`, `PhysSize = 1.0 nm`, diameters `70 nm` and `128 nm`,
   - pointwise and minima-alignment metrics with fixed empirical thresholds,
   - explicit disk-versus-vacuum morphology,
   - fixed `sr=1` only, mirroring the sphere test's assertion anchor while
     avoiding extra 2D-path variability,
   - optional plot writing gated by `NRSS_WRITE_VALIDATION_PLOTS=1`.
6. Added `tests/validation/test_2d_disk_contrast_scaling.py`:
   - one-morph, multi-energy contrast-scaling validation for the 2D pathway,
   - `1 x 2048 x 2048`, `PhysSize = 1.0 nm`,
   - 24 close-energy scenarios covering beta-only, delta-only, mixed, and
     split-material families,
   - integrated-intensity checks over a fixed q window with fixed empirical
     thresholds.
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
   - validates detector-peak locations and quasi-powder shell locations through
     the pybind-to-PyHyper workflow,
   - includes verbose diagnostic plots with full predicted-shell overlays.
9. Added `tests/validation/test_bragg_3d_lattice.py`:
   - deterministic simple-cubic (`a = 30 nm`) and ideal HCP (`a = 45 nm`)
     sphere lattices at `256 x 1024 x 1024`, `PhysSize = 1.0 nm`,
   - validates detector-visible 3D Bragg peak locations plus azimuthally
     averaged shell locations,
   - uses explicit flat-detector geometry handling for shell prediction and
     includes verbose diagnostic plots with visibility-class overlays.
10. Added `tests/validation/lib/core_shell.py` plus
    `tests/validation/test_core_shell_reference.py`:
    - maintained CoreShell baseline workflow through pybind +
      PyHyperScattering `WPIntegrator` + manual A-wedge reduction,
    - experimental PGN RSoXS golden as the scientific gate,
    - parallel sim-derived golden as a tight regression guard,
    - `experimental_validation` marker applied to the experimental-reference
      test,
    - falsification/subterfuge scenarios intentionally kept only in the
      development diagnostic, not in the principal `tests/validation` surface.
11. Added `tests/validation/lib/mwcnt.py` plus
    `tests/validation/test_mwcnt_reference.py`:
    - maintained deterministic MWCNT workflow through pybind +
      PyHyperScattering `WPIntegrator` + anisotropy-observable reduction,
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
     narrow surfaces like `execution_path` and the planned named
     mixed-precision mode rather than a generic backend `dtype` knob
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
13. `/home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/validation --collect-only -q`
    - result: `14 tests collected`
14. latest full compatibility report after the runtime-interface refactor:
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
2. The principal cross-backend primary-time comparison now lives at
   `tests/validation/dev/core_shell_backend_performance/run_primary_backend_speed_comparison.py`
   and emits a combined summary, TSV, and PNG table for the fixed single-energy
   CoreShell comparison panel.
3. The legacy full-energy backend-comparison harness under
   `tests/validation/dev/core_shell_backend_performance/` is no longer the
   authoritative timing harness for optimization work.
4. Backend-native result/output policy is still scaffolding only; current run
   behavior remains xarray/NumPy-oriented for parity and comparison workflows.
5. Serialization and write helpers remain effectively NumPy/CyRSoXS-oriented,
   which is acceptable for prep but will need review once a non-CyRSoXS runtime
   starts emitting backend-native arrays.
6. Resident-mode control now ships as the public `resident_mode` API surface
   for choosing host-resident vs device-resident authoritative morphology
   behavior.
7. Package/dependency policy for CuPy was later tightened after this historical
   snapshot:
   - this March 22, 2026 snapshot still reflects the older extras-based
     metadata state,
   - the current repo-wide policy is documented above in
     "Packaging and Environment Direction",
   - keep this note only as historical context for why older prep-era text may
     still mention extras or packaging-standard limitations
8. If a future environment has no runnable backend at all, `Morphology`
   construction still fails cleanly rather than creating a backend-less object;
   this is consistent with the current one-backend-per-instance decision.
