# NRSS Backend Upgrade Guide

This document is the primary resumable handoff for the NRSS backend-preparation
project. It is intended to be sufficient for a fresh context to continue the
work without relying on chat history.

This guide is prep-only. It sets the architecture for multiple backends and the
first NRSS-native backend, but it does not implement the new backend compute
engine itself.

## 1. Scope And Intent

### 1.1 Immediate objective

Prepare NRSS to support multiple execution backends while preserving current
behavior for existing users.

### 1.2 First planned new backend

The first NRSS-native backend is planned to be named `cupy-rsoxs`.

This backend is not implemented in the prep milestone. The prep milestone
creates the registry, selection, input negotiation, and test routing needed so
that `cupy-rsoxs` can be developed later without forcing a larger refactor.

### 1.3 Current backend reality

At the start of this work, NRSS has one runtime backend:

- `cyrsoxs`, accessed through the CyRSoXS Python bindings.

The current implementation is tightly coupled:

- `src/NRSS/morphology.py` imports `CyRSoXS` at module import time.
- `Morphology` owns CyRSoXS object creation and launch directly.
- many tests either import `CyRSoXS` directly or assume CyRSoXS as the runtime.

The current CyRSoXS pybind path is not compatible with end-to-end GPU-resident
morphology handling. The binding accepts host arrays and the engine performs
explicit host-to-device transfers internally.

## 2. Hard Decisions Locked In

The following decisions were made explicitly and should be treated as the
current project contract unless superseded by a later repo edit.

### 2.1 One backend per `Morphology` instance

- A `Morphology` instance is associated with exactly one backend.
- Different `Morphology` instances may target different backends.
- The same object should not keep multiple live backend bindings.

### 2.2 Backend preference semantics

- `Morphology.backend` means the preferred backend for that instance.
- `run(backend=...)` may override backend selection for a single run in the
  future if desired, but the prep milestone only needs the instance-level
  backend selection and default resolution.

### 2.3 Input normalization timing

- Input suitability should be checked at `Morphology` creation time.
- Material field arrays should be normalized eagerly for the chosen backend.
- The project should not preserve arbitrary original array types and defer
  conversion until much later unless a later design explicitly changes this.

Rationale:

- users get early feedback on unsupported input types,
- expensive host/device conversions are surfaced immediately,
- the backend contract becomes explicit at object construction.

### 2.4 Supported input/output array namespaces for the prep path

For the current prep work:

- accepted input namespaces: `numpy`, `cupy`
- supported output policy targets: CPU-oriented current behavior now, plus
  scaffolding for future backend-native output

Unknown array types must be recognized and rejected with a clear error. They
must not crash NRSS with opaque low-level failures.

### 2.5 Default backend resolution

Default backend resolution is:

1. if `backend=` is explicit, require that backend to be known and available;
2. if omitted, prefer `cyrsoxs` when available, for backward compatibility;
3. otherwise prefer `cupy-rsoxs` when it exists and is available;
4. if no runnable backend is available, NRSS import should still succeed, but
   backend-dependent workflows should fail cleanly with an availability report.

### 2.6 Packaging policy

- It is acceptable for NRSS to require CuPy in the main package metadata.
- The project is not trying to preserve a lightweight CPU-only install as a
  practical production path.
- More exotic future backend stacks such as PyTorch/JAX may still remain
  optional later.

### 2.7 Validation and parity policy

- CLI CyRSoXS vs pybind CyRSoXS parity remains part of the test suite.
- The first NRSS-native backend (`cupy-rsoxs`) is expected to have a parity lane
  against pybind CyRSoXS.
- CyRSoXS parity is not a universal rule for all future backends. Some future
  backends may intentionally diverge because they add new physics beyond the
  CyRSoXS/Born-equivalent model.

## 3. Explicit Non-Goals For This Milestone

- Do not implement `cupy-rsoxs` compute kernels or full simulation math here.
- Do not redesign the science model beyond what is needed for backend
  decoupling.
- Do not require multiple backend bindings per `Morphology` instance.
- Do not force parity-with-CyRSoXS as a permanent rule for every future backend.
- Do not overhaul all output handling yet; only add enough scaffolding to avoid
  blocking future GPU-native outputs.

## 4. Current Code Touchpoints

These are the high-value files for backend preparation.

### 4.1 Runtime

- `src/NRSS/morphology.py`
  - current Morphology/Material API
  - current CyRSoXS object creation and launch
  - current validator boundary
- `src/NRSS/reader.py`
  - config/material parsing helpers
- `src/NRSS/writer.py`
  - CLI/HDF5 compatibility helpers
- `src/NRSS/__init__.py`
  - public export surface

### 4.2 Tests

- `tests/conftest.py`
  - pytest-wide backend selection and skipping logic
- `tests/smoke/test_smoke.py`
  - fast contract tests and current runtime checks
- `tests/validation/test_*.py`
  - physics validation modules
- `tests/validation/lib/`
  - shared validation builders/reducers

### 4.3 Local reporting/orchestration

- `scripts/run_local_test_report.sh`
  - local report orchestration
- `scripts/run_physics_validation_suite.py`
  - row-by-row physics runner

### 4.4 Existing planning documents

- `UPGRADE_ROADMAP.md`
  - broader backend modernization notes

This file is intended to be the practical implementation handoff for the
backend-prep phase. `UPGRADE_ROADMAP.md` remains useful for broader future
direction.

## 5. Architecture Direction

### 5.1 Backend registry

NRSS needs an explicit backend registry that can answer:

- what backends are known,
- what backends are currently available,
- why a backend is unavailable,
- what capabilities a backend advertises.

Initial known backend ids:

- `cyrsoxs`
- `cupy-rsoxs`

Initial registry responsibilities:

- lazy availability detection
- no hard import of CyRSoXS at NRSS module import time
- backend metadata/capability reporting

Useful initial capability fields:

- `available`
- `implemented`
- `supports_cli`
- `supports_reference_parity`
- `supports_device_input`
- `supports_backend_native_output`

### 5.2 `Morphology` responsibilities after prep

`Morphology` should remain the user-facing container for:

- materials
- config
- selected backend
- input normalization state
- simulation state

`Morphology` should stop being inherently synonymous with the CyRSoXS object
graph.

Implementation-phase direction now locked:

- `Morphology` remains the main user-facing simulation container for parity work.
- phase-1 `cupy-rsoxs` parity scope is Euler-only and single-GPU-only.
- phase-1 `cupy-rsoxs` parity scope uses the default partial-scatter path only.
- the parity target is the CyRSoXS math path, not the pybind host-ingestion pathway.

### 5.3 Runtime object ownership

Current CyRSoXS compatibility requires preserving legacy attributes such as:

- `inputData`
- `OpticalConstants`
- `voxelData`
- `scatteringPattern`
- `create_cy_object`
- `create_update_cy()`

Prep-path policy:

- keep these working for the `cyrsoxs` backend,
- do not make them the universal abstraction for all future backends,
- move toward generic backend-neutral preparation/run entry points.

Implementation-phase direction now locked:

- do not model `cupy-rsoxs` as a thin clone of the CyRSoXS pybind object graph;
  model it as a backend-owned runtime/session with backend-neutral entry points.
- do not use a singleton-style runtime object for `cupy-rsoxs` state that owns
  scratch buffers, plans, or results.
- prefer per-`Morphology` runtime/session ownership for:
  - FFT plans,
  - scratch/intermediate buffers,
  - memory instrumentation state,
  - result handles,
  - explicit release/cleanup hooks.
- globally cached compiled kernels/modules are acceptable later if they remain
  stateless and separate from morphology-owned runtime state.

### 5.4 Input negotiation

Input negotiation is one of the most important prep tasks.

It is not just "dtype conversion." It must track:

- namespace (`numpy`, `cupy`, unknown)
- device class (host vs device)
- dtype
- contiguity/layout
- whether a copy occurred
- whether a host<->device transfer occurred
- whether the conversion was backend-required

The prep milestone only needs `numpy` and `cupy`, but the design should not
block future adapters for PyTorch/JAX or other CUDA-array producers.

Current prep implementation status:

- backend array contracts now live in `src/NRSS/backends/contracts.py`
- `Morphology(..., backend_options=...)` now normalizes backend options at
  construction time
- the current normalized option surface is intentionally small:
  - `dtype` for `cyrsoxs`
  - `dtype` for `cupy-rsoxs`
- current contract table:
  - `cyrsoxs`: authoritative/runtime namespace `numpy`, device `cpu`,
    default dtype `float32`, supported dtypes `float32`
  - `cupy-rsoxs`: default authoritative namespace `numpy` in
    `resident_mode='host'`, runtime namespace `cupy`, optional authoritative
    namespace `cupy` in `resident_mode='device'`, default dtype `float32`,
    supported dtypes `float32`

These contracts are prep scaffolding, not a claim that future backends should
share the same option set. The important design point is that backend-specific
input requirements now have an explicit normalization/validation layer instead
of being spread implicitly through `Morphology`.

Implementation-phase direction now locked:

- phase-1 `cupy-rsoxs` parity uses `float32` morphology/runtime normalization by
  default; `float16` is deferred until after parity and is currently disabled
  in the public backend option contract.
- add an explicit `ownership_policy` surface in v1 with `borrow` and `copy`
  modes.
- `cupy-rsoxs` development/parity work should use `ownership_policy='borrow'`
  as the default path.
- `cyrsoxs` should retain copy-oriented construction semantics by default for
  backward compatibility unless explicitly overridden later.
- explicit tracking of contiguity remains important because device-side layout
  changes can be materially expensive.
- publish backend-preferred CuPy layout/contiguity guidance early for direct
  CuPy morphology builders, and provide helper inspection/normalization
  utilities rather than relying on silent coercion.
- recommended parity contract for CuPy morphology fields is:
  - shape/order semantics: ZYX voxel indexing,
  - memory layout: C-contiguous,
  - dtype: `float32`,
  - one array per field (`Vfrac`, `S`, `theta`, `psi`) per material.
- strict-mode rejection should state this expected contract explicitly so users
  know what to build.
- strict-mode tests should be used to catch stealth transfers/copies during
  `cupy-rsoxs` bring-up.
- avoiding unnecessary copies is a project goal, but changing default
  `Material`/`Morphology` ownership semantics can be deferred if it would slow
  parity work.
- borrowed construction in `cupy-rsoxs` means incoming morphology arrays are
  used in place after contract validation/coercion rules are satisfied; if a
  coercion is required under `input_policy='coerce'`, the coerced array becomes
  the backend-owned borrowed array for that run.
- ownership/borrowing should be designed as an explicit contract rather than an
  implicit backend side effect.

Optical-constants direction for parity:

- keep the public `OpticalConstants` contract host-friendly for phase-1 parity.
- material optical constants do not need to live on device persistently inside
  `Morphology`.
- at compute time, the backend may stage the small per-energy optical-constant
  tensors into CuPy arrays when they are needed for math with device-resident
  morphology data.

### 5.5 Output policy

Output policy is lower priority than input negotiation for the prep milestone.

Direction:

- preserve current default behavior for existing users
- add a backend-policy surface so future backends can return backend-native
  arrays without forcing immediate device-to-host copies
- a lightweight result wrapper is acceptable later if it does not impose
  meaningful overhead on the hot path

Implementation-phase direction now locked:

- parity output remains xarray-compatible and NumPy/PyHyperScattering-friendly.
- for `cupy-rsoxs`, prefer lazy host conversion at result-read time instead of
  eager device-to-host transfer during compute if practical.
- backend-native/on-device result access is deferred until after parity, but the
  result abstraction should be designed now so it can be added without breaking
  the parity API.
- users should be able to keep results while allowing the larger runtime/morph
  state to be released when possible.
- a lightweight backend-owned result wrapper is now recommended rather than
  baking xarray conversion directly into the hot compute path.
- that result wrapper should support an explicit release path so users can drop
  morphology/runtime working state while keeping the scattering result alive.

### 5.6 Visualization and host-view utilities

The current validator path is already partly backend-aware, but the visualizer
and several convenience/reporting surfaces are still NumPy/matplotlib-oriented.

Implementation-phase direction now locked:

- keep `check_materials(...)` and related scalar checks backend-aware.
- add backend-neutral host-view helpers for:
  - scalar reductions,
  - selected-slice conversion to NumPy,
  - lightweight histogram/debug extraction.
- do not require the visualizer to understand CuPy arrays directly at every call
  site; centralize host conversion in a narrow utility layer.
- avoid accidental full-volume device-to-host transfers when visualization only
  needs a slice or summary.

## 6. Test Architecture Direction

The current suite is already strong scientifically, but it is not yet routed
for multi-backend development.

### 6.1 Required test lanes

Introduce or formalize these test categories:

- `backend_agnostic_contract`
  - API-level and validator-level behavior not tied to a specific backend
- `cyrsoxs_only`
  - tests that directly depend on CyRSoXS pybind/CLI or legacy CyRSoXS-specific
    object semantics
- `reference_parity`
  - tests whose assertions encode current CyRSoXS-compatible behavior and should
    only run on parity-oriented backends
- `backend_specific`
  - tests meant to exercise the selected backend pathway
- existing physics/validation markers remain useful and should coexist with the
  backend-routing markers

### 6.2 Important distinction

`cyrsoxs_only` is not the same as `reference_parity`.

Examples:

- CLI-vs-pybind parity is `cyrsoxs_only`
- a future parity test comparing `cupy-rsoxs` against CyRSoXS would be
  `reference_parity` but not `cyrsoxs_only`

### 6.3 Routing mechanism

Pytest should support backend selection through:

- `--nrss-backend`
- `NRSS_BACKEND`

The local report runner should thread this selection through all commands so
that backend development can target:

- a single test,
- one module,
- a marker lane,
- or the default local report.

## 7. Implementation Stages

The prep work should be done in the following order.

### Stage 0. Planning and handoff

Deliverables:

- this guide
- explicit decisions captured in-repo

Status:

- required before all subsequent work

### Stage 1. Import-safe backend discovery

Deliverables:

- new backend registry module(s)
- public functions to inspect known/available backends
- import-safe NRSS package even when CyRSoXS is absent

Success criteria:

- `import NRSS` no longer requires CyRSoXS to be installed
- `available_backends()` returns structured availability information

### Stage 2. Backend-selected `Morphology`

Deliverables:

- `Morphology(..., backend=..., input_policy=...)`
- default backend resolution using the locked rules above
- eager normalization of material fields for the selected backend
- clear failure for unknown/unavailable backends

Success criteria:

- existing default `cyrsoxs` behavior still works
- `Morphology` construction no longer depends on top-level CyRSoXS import

### Stage 3. CyRSoXS compatibility preservation

Deliverables:

- preserve current `cyrsoxs` methods/attrs for existing users
- route CyRSoXS-only work through lazy imports and backend checks

Success criteria:

- current smoke/validation paths continue to work with `backend="cyrsoxs"`

### Stage 4. Backend-aware test routing

Deliverables:

- pytest backend option/env
- skip/routing rules for `cyrsoxs_only` and `reference_parity`
- test-marker updates in smoke/validation suites
- avoid top-level hard import failures in test modules that currently import
  `CyRSoXS` directly

Success criteria:

- backend selection can route tests naturally
- selecting a non-CyRSoXS backend does not crash collection because a module
  imported `CyRSoXS` too early

### Stage 5. Report-script integration

Deliverables:

- `run_local_test_report.sh` backend option/env support
- `run_physics_validation_suite.py` backend option/env support
- report metadata includes selected backend

Success criteria:

- developers can run targeted validation for a specific backend with the same
  report tooling

### Stage 6. Future work after prep

Not part of this milestone, but enabled by it:

- implement `cupy-rsoxs`
- add parity tests between `cupy-rsoxs` and CyRSoXS pybind
- expand output policy
- add transfer-observability tests
- add additional backends

## 8. Current Recommended File Layout

The prep path should add a backend package under `src/NRSS/backends/`.

Recommended initial contents:

- `src/NRSS/backends/__init__.py`
- `src/NRSS/backends/registry.py`
- `src/NRSS/backends/arrays.py`
- `src/NRSS/backends/contracts.py`
- `src/NRSS/backends/runtime.py`
- `src/NRSS/backends/cyrsoxs.py`

The first prep implementation does not need a complete backend adapter class
hierarchy yet. It is acceptable to keep the current CyRSoXS execution code in
`morphology.py` while removing import-time coupling and adding backend-aware
selection/normalization.

## 9. Risks To Watch

### 9.1 Backward compatibility risk

The highest risk is breaking users who currently rely on:

- default CyRSoXS execution,
- `create_cy_object`,
- direct access to `inputData` / `voxelData` / `scatteringPattern`,
- current return behavior from `run()`.

### 9.2 Test collection risk

Some validation modules import `CyRSoXS` at module scope. Those imports must be
made lazy or guarded, otherwise backend-selection work will still fail during
test collection.

### 9.3 Hidden conversion risk

Normalization at `Morphology` creation can hide expensive transfers if it is not
reported clearly. The coercion layer must record when it performed:

- dtype casts
- layout copies
- host->device transfers
- device->host transfers

### 9.4 Scope creep risk

Do not drift from prep into implementation of `cupy-rsoxs` math kernels during
this phase.

### 9.5 Runtime-state lifetime risk

For `cupy-rsoxs`, a cached singleton runtime would make it easy to create:

- stale scratch buffers,
- hidden cross-run coupling,
- unclear cleanup boundaries,
- memory retention that is hard to reason about.

The implementation phase should treat runtime/session ownership as explicit and
per-morphology unless a narrower shared cache is clearly proven safe.

### 9.6 Result-lifetime and mutation risk

Once backend-native results and lazy conversion exist, NRSS must define what
happens if morphology data is mutated after:

- preparation,
- run,
- result conversion/export.

Recommended direction:

- phase-1 should take the safer default and forbid morphology mutation after a
  result has been materialized unless and until an explicit invalidation API is
  implemented.
- this freeze should begin at successful `run()` completion, not only at later
  xarray/export access.
- if later mutation is re-enabled, treat post-run mutation as a
  result-invalidation event at minimum.

### 9.7 Large-box parity risk

Large morphologies are one of the main reasons this backend exists, but they
also make it easy to accidentally introduce policy drift while trying to save
memory.

Recommended direction:

- for parity, large-box behavior should initially mirror current CyRSoXS
  policy/limits rather than introducing new chunking or alternate projection
  policy at the same time as the backend rewrite.
- once parity and memory instrumentation are in place, large-box-specific
  optimizations can be evaluated as explicit follow-up work with separate
  thresholds.

## 10. Resumption Checklist

If resuming from a fresh context, do the following first:

1. Read this document.
2. Inspect:
   - `src/NRSS/morphology.py`
   - `tests/conftest.py`
   - `tests/smoke/test_smoke.py`
   - `scripts/run_local_test_report.sh`
   - `scripts/run_physics_validation_suite.py`
3. Check whether `src/NRSS/backends/` exists and what has been implemented.
4. Run focused smoke tests first, then targeted backend-aware tests.
5. Confirm whether `cyrsoxs` remains the default resolved backend.

### 10.1 Preferred parity-validation shape

When `cupy-rsoxs` implementation starts, prefer this minimum targeted test set:

- a CoreShell workflow variant that builds NumPy morphology fields and runs with
  `backend='cupy-rsoxs'`, `input_policy='coerce'`, and the parity ownership
  default. This validates the supported NumPy-to-CuPy input contract.
- a second CoreShell workflow variant that builds CuPy-native morphology fields
  in backend-preferred layout and runs with `backend='cupy-rsoxs'`,
  `ownership_policy='borrow'`, and `input_policy='strict'`. This validates the
  no-stealth-transfer borrowed path.

## 11. Implementation Status Notes

When updating this document in later work, append the current status here:

- what stage is complete,
- what files were changed,
- what tests were run,
- what remains blocked.

### Status snapshot: 2026-03-22

Completed stages:

- Stage 0: complete
  - this guide was created as the resumable backend-prep handoff
- Stage 1: complete
  - `src/NRSS/backends/registry.py` now provides import-safe backend discovery
  - `src/NRSS/backends/__init__.py` and `src/NRSS/__init__.py` expose public
    backend-inspection helpers
  - `import NRSS` no longer requires CyRSoXS to be importable
- Stage 2: substantially complete for prep
  - `Morphology(..., backend=..., input_policy=..., output_policy=...)` now
    resolves backend selection without top-level CyRSoXS import
  - material fields are now normalized eagerly at `Morphology` construction
    time for the selected backend contract
  - `Morphology(..., backend_options=...)` now validates and normalizes
    backend-specific options at construction time
  - backend dtype selection is now part of the explicit contract layer instead
    of being hardcoded inside array coercion
  - `input_policy='strict'` now fails early when normalization would require
    dtype/layout/device coercion
  - unsupported input array types now fail cleanly during `Morphology`
    construction instead of surfacing as opaque lower-level failures later
- Stage 3: complete for current `cyrsoxs` preservation scope
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

Planning decisions additionally locked after the prep milestone:

- `cupy-rsoxs` parity targets the CyRSoXS math path rather than the pybind
  ingestion path
- phase-1 parity scope is Euler-only
- phase-1 parity scope uses `ScatterApproach::PARTIAL` only
- phase-1 parity scope is single-GPU only
- literal rotation / angle-accumulation semantics are required for parity;
  alternate leaner/faster variants belong in later optimization notes
- low-memory path may be implemented first, but parity is not complete until
  both algorithm paths are supported
- `ownership_policy` should be exposed in v1 with `borrow` and `copy`, and
  `cupy-rsoxs` parity work should use `borrow`
- `cupy-rsoxs` should support controllable resident modes:
  - host-resident staged mode is the default guidance for public workflows
  - device-resident direct mode is an opt-in path for already-CuPy morphology
    fields
- the implemented public API name for this control is `resident_mode`
- resident mode is a separate concept from `input_policy` and
  `ownership_policy`; these should not be conflated in docs or benchmarks
- parity output remains xarray-compatible; backend-native/on-device output is a
  later extension
- `float16` is deferred until after parity and is currently disabled in the
  backend option contract
- future `cupy-rsoxs` runtime state should be per-morphology/per-session rather
  than a singleton backend adapter

- large-box behavior should initially follow current CyRSoXS policy for parity
- phase-1 should prefer freezing morphology mutation after result creation until
  an explicit invalidation contract exists
- the mutation freeze should begin at successful `run()` completion
- preferred direct-CuPy morphology contract for parity is ZYX-shaped,
  C-contiguous, `float32` arrays for each material field
- initial `cupy-rsoxs` parity validation should include both:
  - a NumPy-input contract case using `input_policy='coerce'`
  - a CuPy-native borrowed case using `ownership_policy='borrow'` and
    `input_policy='strict'`
- internal `cupy-rsoxs` optimization timing is now implemented:
  - primary timing runs from immediately before `Morphology(...)`
    construction to synchronized `run(return_xarray=False)` completion
  - upstream field generation and export are excluded from the primary metric
  - private `Morphology._set_private_backend_timing_segments(...)` and
    `Morphology._clear_private_backend_timing_segments()` drive opt-in segment
    timing
  - Segment `A1` is measured in the dev harness
  - Segment `A2` is measured in `cupy-rsoxs` via private wall-clock timing for
    runtime staging
  - Segments `B-F` are measured in `cupy-rsoxs` via CUDA events, and Segment
    `G` is deferred
  - when timing is not enabled, `backend_timings` stays empty and the timing
    event path is skipped
- the default optimization harness now targets the common host-resident public
  workflow first:
  - `resident_mode='host'`,
  - single energy,
  - `EAngleRotation=[0, 0, 0]`,
  - NumPy authoritative fields generated directly in contract shape/dtype
- the opt-in device-resident benchmark should not be treated as a true
  end-to-end GPU-native morphology-generation benchmark because fields are
  still created in NumPy before preconversion to CuPy
- in that opt-in device-resident lane, the harness synchronizes the default
  stream before starting the timer so upstream CuPy preparation is excluded
- the live dev optimization harness now also supports optional untimed CUDA
  prewarm for host-resident steady-state studies:
  - `--cuda-prewarm off` preserves the default cold subprocess behavior
  - `--cuda-prewarm before_prepare_inputs` performs a tiny NumPy -> CuPy
    staging touch inside the worker before `_prepare_core_shell_case_inputs(...)`
  - this models many-morph single-process workflows without changing backend
    allocator/pool refresh behavior
  - device-resident cases record the mode as redundant because that lane
    already touches CuPy before `primary_start`
- the limited-rotation triple-energy checkpoint remains available as an opt-in
  lane for either resident-mode variant, but it is not part of the default
  optimization loop
- in the default host-resident lane, host-to-device staging is included in the
  primary wall-clock metric and is now isolated as private Segment `A2`
- fresh host-resident `A2` measurements may still include first-touch
  CUDA/CuPy bring-up, so cold-process `A2` should not be overinterpreted as
  pure transfer cost
- Segment `A` is now nominally complete for the common workflow:
  - Segment `A1` constructor work is already small,
  - the accepted host-resident Segment `A2` staging improvement is in place,
  - workflows that intentionally keep morphology fields on GPU should use
    `resident_mode='device'` and are expected to be faster because Segment
    `A2` largely disappears in that use case
- the authoritative isotropic fast path is now the explicit
  `SFieldMode.ISOTROPIC` material contract:
  - named `vacuum` always resolves to that contract
  - supplied `theta` / `psi` are ignored with warning under that contract
  - no inferred all-zero scan remains in Segment `A2`
  - legacy zero-array isotropic inputs remain supported for compatibility but
    do not receive inferred isotropic optimization
- default future speed work should focus on Segments `B` and `D`
- repeated-run host-resident staging reuse is recorded only as a low-priority
  niche possibility; if persistent GPU residency is the real use case, prefer
  `resident_mode='device'`
- current optimization tuning should default to single-energy lanes; full-energy
  studies are legacy/historical comparison artifacts or milestone confirmation

- Stage 4: complete for the current test suite routing scope
  - `tests/conftest.py` now supports `--nrss-backend` and `NRSS_BACKEND`
  - backend-aware skip/routing is implemented for `backend_specific`,
    `cyrsoxs_only`, and `reference_parity`
  - validation modules that previously broke collection due to eager marker
    tuples were fixed
  - lazy CyRSoXS import helper added at `tests/validation/lib/lazy_cyrsoxs.py`
- Stage 5: complete
  - `scripts/run_local_test_report.sh` now threads backend selection through
    local report steps
  - `scripts/run_physics_validation_suite.py` now accepts and propagates
    `--nrss-backend`

Files changed in the landed prep work:

- `BACKEND_UPGRADE_GUIDE.md`
- `pyproject.toml`
- `environment.yml`
- `scripts/run_local_test_report.sh`
- `scripts/run_physics_validation_suite.py`
- `src/NRSS/__init__.py`
- `src/NRSS/backends/__init__.py`
- `src/NRSS/backends/arrays.py`
- `src/NRSS/backends/contracts.py`
- `src/NRSS/backends/cyrsoxs.py`
- `src/NRSS/backends/registry.py`
- `src/NRSS/backends/runtime.py`
- `src/NRSS/morphology.py`
- `tests/conftest.py`
- `tests/smoke/test_smoke.py`
- `tests/validation/lib/lazy_cyrsoxs.py`
- `tests/validation/test_2d_disk_contrast_scaling.py`
- `tests/validation/test_analytical_2d_disk_form_factor.py`
- `tests/validation/test_analytical_sphere_form_factor.py`
- `tests/validation/test_bragg_2d_lattice.py`
- `tests/validation/test_bragg_3d_lattice.py`
- `tests/validation/test_core_shell_reference.py`
- `tests/validation/test_mwcnt_reference.py`
- `tests/validation/test_sphere_contrast_scaling.py`
- `tests/validation/test_sphere_orientational_contrast_scaling.py`

Verification completed:

- `python -m py_compile` on modified runtime/test modules
- baseline full local report:
  - `./scripts/run_local_test_report.sh -e nrss-dev --nrss-backend cyrsoxs`
  - result: `4/4 steps passed`
  - CPU smoke: `22 passed, 10 deselected`
  - GPU smoke: `10 passed, 22 deselected`
  - physics validation: `14 passed`
- post-change full local report:
  - `./scripts/run_local_test_report.sh -e nrss-dev --nrss-backend cyrsoxs --no-plots`
  - result: `4/4 steps passed`
  - CPU smoke: `22 passed, 12 deselected`
  - GPU smoke: `12 passed, 22 deselected`
  - physics validation: `14 passed`
- authoritative post-backend-options full local report:
  - `./scripts/run_local_test_report.sh -e nrss-dev --nrss-backend cyrsoxs --no-plots`
  - report dir: `test-reports/20260322T195347Z`
  - result: `4/4 steps passed`
  - CPU smoke: `23 passed, 13 deselected`
  - GPU smoke: `13 passed, 23 deselected`
  - physics validation: `14 passed`
- `/home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/smoke -m 'not gpu' -q`
  - result: `23 passed, 13 deselected`
- `/home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/smoke -m 'gpu' -q`
  - result: `13 passed, 23 deselected`
- `/home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/smoke/test_smoke.py -m 'backend_agnostic_contract and not gpu' -q`
  - result: `16 passed, 16 deselected`
- `/home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/smoke/test_smoke.py -m 'cyrsoxs_only and not gpu' -q`
  - result: `3 passed, 29 deselected`
- `/home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/smoke/test_smoke.py -k 'backend_registry_reports_known_backends or planned_cupy_backend_array_contract or cyrsoxs_backend_rejects_non_default_dtype_option or normalizes_material_arrays_eagerly_for_selected_backend' -q`
  - result: `5 passed, 31 deselected`
- `/home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/smoke/test_smoke.py -k 'normalizes_material_arrays_eagerly or strict_input_policy or unrecognized_array_types or coerces_non_float_field' -q`
  - result: `4 passed, 28 deselected`
- `/home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/smoke/test_smoke.py -k 'cupy_import_available or planned_cupy_backend_array_contract or cyrsoxs_morphology_normalizes_cupy_inputs' -q`
  - result: `5 passed, 31 deselected`
- `/home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/smoke/test_smoke.py -k 'backend_registry_reports_known_backends or pybind_morphology_object_lifecycle_smoke or pybind_runtime_tiny_deterministic_pattern or cyrsoxs_morphology_normalizes_cupy_inputs_to_host_contract' -q`
  - result: `4 passed, 32 deselected`
- `/home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/validation --collect-only -q`
  - result: `14 tests collected`
- latest full compatibility report after the runtime-interface refactor:
  - `./scripts/run_local_test_report.sh -e nrss-dev --nrss-backend cyrsoxs --no-plots`
  - report dir: `test-reports/20260322T202539Z`
  - result: `4/4 steps passed`
  - CPU smoke: `23 passed, 13 deselected`
  - GPU smoke: `13 passed, 23 deselected`
  - physics validation: `14 passed`

Important note:

- `test-reports/20260322T194742Z` should be treated as stale and ignored for
  status purposes
- that run started before the CuPy memory cleanup fix in the GPU smoke path and
  is not the authoritative post-change report
- `test-reports/cupy-rsoxs-optimization-dev/verify_cli_small_postcleanup/summary.json`
  is the first authoritative post-cleanup timing snapshot for resumed
  optimization work

Remaining intentionally deferred or unresolved items:

- `cupy-rsoxs` compute/runtime implementation now exists; timing cleanup is in
  place for the current optimization loop, and the open work is resident-mode
  refinement, segment-targeted optimization, export timing, and deeper memory
  instrumentation follow-up. See `UPGRADE_ROADMAP.md` for the current detailed
  state.
- the principal cross-backend primary-time comparison now lives at
  `tests/validation/dev/core_shell_backend_performance/run_primary_backend_speed_comparison.py`
  and emits a combined summary, TSV, and PNG table for the fixed single-energy
  CoreShell comparison panel
- the legacy full-energy backend-comparison harness under
  `tests/validation/dev/core_shell_backend_performance/` is no longer the
  authoritative timing harness for optimization work
- backend-native result/output policy is still scaffolding only; current run
  behavior remains xarray/NumPy-oriented for parity and comparison workflows
- serialization and write helpers remain effectively NumPy/CyRSoXS-oriented,
  which is acceptable for prep but will need review once a non-CyRSoXS runtime
  starts emitting backend-native arrays
- resident-mode control now ships as the public `resident_mode` API surface for
  choosing host-resident vs device-resident authoritative morphology behavior
- package/dependency policy for CuPy is improved but not perfectly expressible
  in standard metadata:
  - the default conda env now pins `cupy-cuda12x` in `environment.yml`
  - `pyproject.toml` now exposes supported CuPy install lines as extras:
    `cupy`, `cupy-cuda12x`, and `cupy-cuda13x`
  - smoke coverage now requires `import cupy` to succeed in supported dev/test
    environments
  - base PEP 621 dependencies still cannot encode a hard "install any one of
    these three packages" rule, so the remaining gap is a packaging-standard
    limitation rather than a missing repo hook
- if a future environment has no runnable backend at all, `Morphology`
  construction still fails cleanly rather than creating a backend-less object;
  this is consistent with the current one-backend-per-instance decision
