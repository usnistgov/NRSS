# `cupy-rsoxs` Backend Spec

This document is the authoritative stable specification for the `cupy-rsoxs`
backend and the backend-neutral seams required to support it. It consolidates
the stable backend contract that was previously spread across
`UPGRADE_ROADMAP.md` and `BACKEND_UPGRADE_GUIDE.md`.

Related planning documents:

- `REPO_UPGRADE_PLAN.md`
  - repo-wide goals, validation/test program, packaging direction, workflow
    policy, and historical prep/status notes
- `CUPY_RSOXS_OPTIMIZATION_LEDGER.md`
  - timing methodology, resident-mode benchmark interpretation, and accepted or
    rejected optimization experiments

## Current Backend Reality and Scope

At the start of this work, NRSS had one runtime backend:

1. `cyrsoxs`, accessed through the CyRSoXS Python bindings.

The original implementation was tightly coupled:

1. `src/NRSS/morphology.py` imported `CyRSoXS` at module import time.
2. `Morphology` owned CyRSoXS object creation and launch directly.
3. Many tests either imported `CyRSoXS` directly or assumed CyRSoXS as the
   runtime.
4. The current CyRSoXS pybind path was not compatible with end-to-end
   GPU-resident morphology handling because the binding accepted host arrays and
   the engine performed explicit host-to-device transfers internally.

Current scope:

1. `cupy-rsoxs` is the first NRSS-native backend.
2. This document specifies the stable backend contract and implementation scope
   for that backend.
3. Detailed performance work and timing methodology are intentionally separated
   into `CUPY_RSOXS_OPTIMIZATION_LEDGER.md`.

## Goal

Build a new NRSS backend architecture that:

1. Preserves trusted physics behavior (CyRSoXS parity first).
2. Enables a CuPy-native simulation path to avoid avoidable GPU->CPU->GPU
   transfers.
3. Supports optional future backends (PyTorch, JAX) without hard dependencies.
4. Improves robustness with deterministic, reproducible regression testing.
5. Provides durable value even if alternate backend implementation is delayed
   (test hardening alone is a success milestone).

## Hard Decisions Locked In

### Instance and backend selection

1. A `Morphology` instance is associated with exactly one backend.
2. Different `Morphology` instances may target different backends.
3. The same object should not keep multiple live backend bindings.
4. `Morphology.backend` means the preferred backend for that instance.
5. `run(backend=...)` may override backend selection for a single run in the
   future if desired, but the prep milestone only needed the instance-level
   backend selection and default resolution.

### Input normalization and supported array namespaces

1. Input suitability should be checked at `Morphology` creation time.
2. Material field arrays should be normalized eagerly for the chosen backend.
3. The project should not preserve arbitrary original array types and defer
   conversion until much later unless a later design explicitly changes this.
4. Accepted input namespaces for the current backend contract are `numpy` and
   `cupy`.
5. Unknown array types must be recognized and rejected with a clear error.
   They must not crash NRSS with opaque low-level failures.

### Default backend resolution

1. If `backend=` is explicit, require that backend to be known and available.
2. If omitted, prefer `cupy-rsoxs` when available.
3. Otherwise prefer `cyrsoxs` when it is available as the legacy/reference
   fallback.
4. If no runnable backend is available, `import NRSS` should still succeed, but
   backend-dependent workflows should fail cleanly with an availability report.

### Validation and parity policy

1. CLI CyRSoXS vs pybind CyRSoXS parity remains part of the test suite.
2. The first NRSS-native backend (`cupy-rsoxs`) is expected to have a parity
   lane against pybind CyRSoXS.
3. CyRSoXS parity is not a universal rule for all future backends. Some future
   backends may intentionally diverge because they add new physics beyond the
   CyRSoXS/Born-equivalent model.

### `cupy-rsoxs` phase-1 lock-ins

1. Parity target is the CyRSoXS math path, not the pybind host-ingestion path.
2. Phase-1 parity is Euler-only.
3. Phase-1 parity uses `ScatterApproach::PARTIAL` only.
4. Phase-1 parity is single-GPU only.
5. Start with literal rotation and angle-accumulation semantics for parity; any
   leaner/faster alternatives are optimization-track follow-ups only.
6. Low-memory path may be implemented first, but parity is not declared
   complete until both algorithm paths are supported.
7. `cupy-rsoxs` should expose controllable resident modes.
8. Host-resident staged mode is the default guidance for public workflows.
9. Device-resident direct mode is an opt-in path for already-CuPy morphology
   fields and is expected to use more GPU memory.
10. Resident mode, `input_policy`, and `ownership_policy` are distinct concepts
    and should remain separately documented.
11. Parity output contract is xarray-compatible and
    NumPy/PyHyperScattering-friendly.
12. Backend-native/on-device result access is deferred until after parity, but
    the result abstraction should preserve that path.
13. `float16` is deferred until after parity; parity-sensitive compute remains
    `float32/complex64`.
14. Runtime instrumentation for timing/memory is desirable in development but
    must be fully disableable.
15. The primary optimization wall metric starts immediately before
    `Morphology(...)` construction and ends immediately after synchronized
    `run(return_xarray=False)` completion.
16. Large-box behavior should initially mirror current CyRSoXS policy during
    parity.
17. Phase-1 should prefer freezing morphology mutation after result creation
    until an explicit invalidation contract exists.
18. The mutation freeze should begin at successful `run()` completion.
19. Public optical constants may remain host-oriented for parity; backend
    staging to CuPy happens when math requires it.
20. Preferred direct-CuPy morphology contract for parity is ZYX-shaped,
    C-contiguous, `float32` arrays for each material field.
21. Initial `cupy-rsoxs` parity validation should include both:
    - a NumPy-input contract case using `input_policy='coerce'`,
    - a CuPy-native borrowed case using `ownership_policy='borrow'` and
      `input_policy='strict'`.

## Physics Reference (Ground Truth Equations)

These equations define the numerical target. Phase 1 (CuPy mimic) follows
current CyRSoXS implementation details and conventions.

1. Far-field scattering intensity (detector projection of Fourier-space
   polarization):

   dσ/dΩ is proportional to |k² (I - r_hat r_hat) · p(q)|².

2. Fourier transform of induced polarization:

   p(q) = ∫ exp(i q·r) p(r) d³r.

3. Scattering vector magnitude from wavelength and scattering angle:

   The magnitude of q is |q| = (4π/λ) sin(θ/2).

4. Local induced polarization from susceptibility tensor and incident field:

   p(r) = ε₀ χ(r) · Ê.

5. Uniaxial constitutive model (principal frame view):

   The local susceptibility is χ_local = diag(χ_ord, χ_ord, χ_ext).

6. Lab-frame tensor is obtained by rotating local-frame tensor with morphology
   orientation (Euler in current workflows), then applying
   composition/volume-fraction weighting per material and energy.

Notes:

1. Exact Euler convention and rotation order must match CyRSoXS for parity.
2. The CuPy mimic backend is required to copy CyRSoXS math order, not
   "improve" it initially.

## CyRSoXS-Mimic Computational Pipeline (Required Mapping)

This section is the implementation handoff for Phase 1.

### Stage A: Input normalization and policy resolution

1. Accept morphology fields (`vfrac`, `S`, `theta`, `psi`; optionally vector
   inputs).
2. Resolve backend (`cyrsoxs`, `cupy-rsoxs`, future backends), `input_policy`,
   and `output_policy`.
3. Normalize dtypes and device placement.

Memory slimming:

1. Do not duplicate morphology arrays unless policy requires copy.
2. Keep one authoritative device-resident view for each field.

GPU tuning opportunities:

1. Use zero-copy where possible (`__cuda_array_interface__`, DLPack
   boundaries).
2. Fuse trivial cast/scale ops into downstream kernels.

### Stage B: Orientation decode

1. Convert Euler representation to orientation direction field `n(r)` using
   CyRSoXS-consistent convention.
2. If vector input is supplied, skip decode and validate normalization/shape.

Memory slimming:

1. Avoid persisting both Euler-derived vectors and equivalent expanded tensors
   unless needed.
2. Reuse scratch buffers across energies/angles.

GPU tuning opportunities:

1. Use fused elementwise kernels for trig + normalization.
2. Keep decode on device; avoid host round-trips for validation.

### Stage C: Local polarization field composition

1. Build local susceptibility/polarization components from morphology + optical
   constants + incident polarization.
2. Produce component fields consumed by FFT stage (current logic mimic first).

Memory slimming:

1. Stream by energy/angle chunks; avoid materializing a full
   `[energy, angle, xyz, components]` tensor at once.
2. Free/recycle polarization component buffers immediately after FFT
   contribution is consumed.

GPU tuning opportunities:

1. Fuse composition math in custom kernels to reduce global-memory traffic.
2. Prefer SoA-style component layout when it improves coalesced reads in
   subsequent FFT prep.

### Stage D: FFT and reciprocal-space conversion

1. Transform spatial polarization fields to `p(q)` via cuFFT-backed operations.
2. Apply required shift/DC handling consistent with CyRSoXS.

Memory slimming:

1. Reuse cuFFT work buffers and plans.
2. Perform in-place transforms where safe.
3. Do not retain pre-FFT buffers once transformed data is consumed.

GPU tuning opportunities:

1. Plan caching and batched FFT shapes.
2. Keep FFT input/working tensors in `float32/complex64` for parity-sensitive
   runs.
3. Optimize layout/strides to avoid internal transposes.

### Stage E: Detector projection / Ewald handling

1. Apply the projection operator `(I - r_hat r_hat)` to `p(q)`.
2. Compute detector intensity contribution for each required geometry.

Memory slimming:

1. Accumulate directly into output/objective buffers; avoid temporary full-size
   detector stacks when not required.
2. Chunk detector or q-regions if peak memory is dominated by result tensors.

GPU tuning opportunities:

1. Kernel fusion for projection + norm-squared accumulation.
2. Use read-only cached loads for geometry tables reused across voxels.

### Stage F: Rotation, angle accumulation, and export

1. Rotate/accumulate across sample angles and energies per current semantics.
2. Emit output by policy (`numpy`, backend-native, objective-only).

Memory slimming:

1. Drop intermediate polarization fields before final result tensor growth.
2. Prefer objective-only mode for fitting loops to minimize resident memory.
3. Allow streaming writes/checkpointing for large result tensors.

GPU tuning opportunities:

1. Keep reduction operations on device.
2. Use backend-native reductions and avoid host synchronization inside hot
   loops.

## Peak Memory Model (Planning Equations)

Define:

1. `N = Nx * Ny * Nz` voxels.
2. `M = detector_nx * detector_ny` detector pixels.
3. `E = n_energies`, `A = n_angles`, and `P = n_polarizations`.

Approximate resident memory terms:

1. Morphology storage:
   `B_morph ~ N * C_morph * b_morph`, where `C_morph = 5` for (`vfrac`, `S`,
   `theta`, `psi`, plus optional mask/material-index representation).
2. Polarization working set (worst-case):
   `B_p ~ N * C_p * b_p`, with `C_p = 3` vector components (real/complex by
   stage).
3. FFT workspace: `B_fft ~ α * B_p`, where `α` depends on cuFFT plan and shape.
4. Result tensor (dominant in many workflows):
   `B_res ~ M * E * A * P * b_res`.

Operational policy:

1. Optimize for the peak footprint, `B_morph + B_p + B_fft + B_res`.
2. Release `B_p` and `B_fft` aggressively before `B_res` expansion.
3. Introduce chunk order default: energy -> angle -> detector/q when needed.
4. Permit high utilization (up to ~95% of 48 GB) only under explicit guardrail
   configuration.

## Precision Policy

1. Default compute precision: `float32` / `complex64`.
2. Morphology storage in `float16`/`bfloat16` is allowed as a compression path.
3. Decode/cast should occur in device kernels near use sites.
4. FFT/q-space and projection math remain `float32/complex64` for
   parity-sensitive runs.
5. Avoid full pipeline compute in 16-bit when parity is required (high-q
   deviations known risk).

Note on cast cost:

1. GPU decode/cast is typically memory-bandwidth-bound and usually cheap
   relative to 3D FFT cost.
2. Cast overhead still needs profiling in end-to-end runs, but it is generally
   not the dominant term.

## Representation Roadmap (Euler, Vector, Tensor)

### Phase 1 behavior

1. Keep Euler-first compatibility (dominant workflows).
2. Accept vector input for backward compatibility.
3. Internals mimic CyRSoXS logic first.

### Phase 2 refactor target

1. Move to cleaner tensor-character internals after parity lock.
2. Candidate uniaxial order tensor form:

   `Q = S (n ⊗ n - I/3)`.

3. Candidate susceptibility decomposition:

   `χ = χ_iso I + Δχ Q` (or equivalent project-specific parameterization).

### Symmetric tensor storage compression

1. Full `3x3` tensor has 9 entries, 6 unique for symmetric form.
2. Store symmetric tensors in packed 6-component form
   (`xx, yy, zz, xy, xz, yz`) to reduce memory.
3. Reconstruct needed matrix elements in fused kernels instead of materializing
   dense `3x3` arrays globally.

### Biaxial scaffold

1. Add constitutive interface now so biaxial models can be added without
   backend redesign.
2. Biaxial may initially exist only in alternate backends if CyRSoXS parity
   path remains uniaxial.

## Future-Backend Context and Deferred Variants

Recommended implementation order for future backend expansion:

1. CuPy: best parity path with current CUDA/cuFFT workflow and minimal
   conceptual translation.
2. PyTorch: good GPU kernel ecosystem and deployment maturity for later
   integration.
3. JAX: strong for compiled/fused workflows, but higher complexity for this
   parity-first migration.

Important backend variant to preserve in planning:

1. Approximate but accelerated computation could be done by computing a 3D
   model up to the point of the 3D `p`-fields, then collapsing the Z-axis by
   sum or mean (Z-axis projection), then continuing on the 2D FFT computation
   track to results.
2. This may be accurate enough for many users and could be significantly faster
   for some jobs.
3. The legacy `cyrsoxs` engine does not have this capability.
4. Ideally, this could be incorporated into future implementations with a flag
   that is ignored with warning by backends that do not support it (like
   `cyrsoxs`).

TensorFlow is deprioritized for this project.

## Architecture Direction

### Backend registry

NRSS needs an explicit backend registry that can answer:

1. what backends are known,
2. what backends are currently available,
3. why a backend is unavailable,
4. what capabilities a backend advertises.

Initial known backend ids:

1. `cyrsoxs`
2. `cupy-rsoxs`

Initial registry responsibilities:

1. lazy availability detection
2. no hard import of CyRSoXS at NRSS module import time
3. backend metadata/capability reporting

Useful initial capability fields:

1. `available`
2. `implemented`
3. `supports_cli`
4. `supports_reference_parity`
5. `supports_device_input`
6. `supports_backend_native_output`

### `Morphology` responsibilities

`Morphology` should remain the user-facing container for:

1. materials
2. config
3. selected backend
4. input normalization state
5. simulation state

`Morphology` should stop being inherently synonymous with the CyRSoXS object
graph.

Implementation-phase direction:

1. `Morphology` remains the main user-facing simulation container for parity
   work.
2. Do not model `cupy-rsoxs` as a thin clone of the CyRSoXS pybind object
   graph; model it as a backend-owned runtime/session with backend-neutral
   entry points.
3. Preserve legacy attributes and methods such as `inputData`,
   `OpticalConstants`, `voxelData`, `scatteringPattern`, `create_cy_object`,
   and `create_update_cy()` for the `cyrsoxs` backend.
4. Do not make those CyRSoXS-specific objects the universal abstraction for all
   future backends.

### Runtime object ownership

1. Do not use a singleton-style runtime object for `cupy-rsoxs` state that
   owns scratch buffers, plans, or results.
2. Prefer per-`Morphology` runtime/session ownership for:
   - FFT plans,
   - scratch/intermediate buffers,
   - memory instrumentation state,
   - result handles,
   - explicit release/cleanup hooks.
3. Globally cached compiled kernels/modules are acceptable later if they remain
   stateless and separate from morphology-owned runtime state.

### Input negotiation and ownership

Input negotiation must track:

1. namespace (`numpy`, `cupy`, unknown)
2. device class (host vs device)
3. dtype
4. contiguity/layout
5. whether a copy occurred
6. whether a host<->device transfer occurred
7. whether the conversion was backend-required

Current prep/implementation status to preserve:

1. Backend array contracts live in `src/NRSS/backends/contracts.py`.
2. `Morphology(..., backend_options=...)` normalizes backend options at
   construction time.
3. The current normalized option surface is intentionally small:
   - `dtype` for `cyrsoxs`
   - `dtype` for `cupy-rsoxs`
4. Current contract table:
   - `cyrsoxs`: authoritative/runtime namespace `numpy`, device `cpu`, default
     dtype `float32`, supported dtypes `float32`
   - `cupy-rsoxs`: default authoritative namespace `numpy` in
     `resident_mode='host'`, runtime namespace `cupy`, optional authoritative
     namespace `cupy` in `resident_mode='device'`, default dtype `float32`,
     supported dtypes `float32`

Implementation-phase direction:

1. Add an explicit `ownership_policy` surface in v1 with `borrow` and `copy`
   modes.
2. `cupy-rsoxs` development/parity work should use `ownership_policy='borrow'`
   as the default path.
3. `cyrsoxs` should retain copy-oriented construction semantics by default for
   backward compatibility unless explicitly overridden later.
4. Explicit tracking of contiguity remains important because device-side layout
   changes can be materially expensive.
5. Publish backend-preferred CuPy layout/contiguity guidance early for direct
   CuPy morphology builders, and provide helper inspection/normalization
   utilities rather than relying on silent coercion.
6. Strict-mode rejection should state the expected direct-CuPy contract
   explicitly so users know what to build.
7. Strict-mode tests should be used to catch stealth transfers/copies during
   `cupy-rsoxs` bring-up.
8. Avoiding unnecessary copies is a project goal, but changing default
   `Material`/`Morphology` ownership semantics can be deferred if it would slow
   parity work.
9. Borrowed construction in `cupy-rsoxs` means incoming morphology arrays are
   used in place after contract validation/coercion rules are satisfied; if a
   coercion is required under `input_policy='coerce'`, the coerced array
   becomes the backend-owned borrowed array for that run.
10. Ownership/borrowing should be designed as an explicit contract rather than
    an implicit backend side effect.
11. The prep milestone only needs `numpy` and `cupy`, but the design should not
    block future adapters for PyTorch/JAX or other CUDA-array producers.

Optical-constants direction for parity:

1. Keep the public `OpticalConstants` contract host-friendly for phase-1
   parity.
2. Material optical constants do not need to live on device persistently inside
   `Morphology`.
3. At compute time, the backend may stage the small per-energy optical-constant
   tensors into CuPy arrays when they are needed for math with device-resident
   morphology data.

### Output policy and result ownership

Output policy direction:

1. Preserve current default behavior for existing users.
2. Add an explicit backend-policy surface so future backends can return
   backend-native arrays without forcing immediate device-to-host copies.
3. Support objective-only returns for fitting workflows so GPU-resident
   objective results do not require forced host copies.
4. A lightweight result wrapper is acceptable later if it does not impose
   meaningful overhead on the hot path.

Implementation-phase direction:

1. Parity output remains xarray-compatible and NumPy/PyHyperScattering-friendly.
2. For `cupy-rsoxs`, prefer lazy host conversion at result-read time instead of
   eager device-to-host transfer during compute if practical.
3. Backend-native/on-device result access is deferred until after parity, but
   the result abstraction should be designed now so it can be added without
   breaking the parity API.
4. Users should be able to keep results while allowing the larger runtime/morph
   state to be released when possible.
5. A lightweight backend-owned result wrapper is recommended rather than baking
   xarray conversion directly into the hot compute path.
6. That result wrapper should support an explicit release path so users can
   drop morphology/runtime working state while keeping the scattering result
   alive.
7. Preserve current default behavior for existing users while also exposing
   explicit conversion methods such as `to_xarray()`, NumPy views, and later
   backend-native accessors.

### Visualization and host-view utilities

1. Keep `check_materials(...)` and related scalar checks backend-aware.
2. Add backend-neutral host-view helpers for:
   - scalar reductions,
   - selected-slice conversion to NumPy,
   - lightweight histogram/debug extraction.
3. Do not require the visualizer to understand CuPy arrays directly at every
   call site; centralize host conversion in a narrow utility layer.
4. Avoid accidental full-volume device-to-host transfers when visualization
   only needs a slice or summary.

## Test Architecture and Preferred Parity Validation Shape

The current suite is already strong scientifically, but it must remain routed
for multi-backend development.

Required test lanes:

1. `backend_agnostic_contract`
   - API-level and validator-level behavior not tied to a specific backend
2. `cyrsoxs_only`
   - tests that directly depend on CyRSoXS pybind/CLI or legacy
     CyRSoXS-specific object semantics
3. `reference_parity`
   - tests whose assertions encode current CyRSoXS-compatible behavior and
     should only run on parity-oriented backends
4. `backend_specific`
   - tests meant to exercise the selected backend pathway
5. Existing physics/validation markers remain useful and should coexist with
   the backend-routing markers.

Important distinction:

1. `cyrsoxs_only` is not the same as `reference_parity`.
2. CLI-vs-pybind parity is `cyrsoxs_only`.
3. A future parity test comparing `cupy-rsoxs` against CyRSoXS would be
   `reference_parity` but not `cyrsoxs_only`.

Routing mechanism:

1. Pytest should support backend selection through `--nrss-backend`.
2. Pytest should support backend selection through `NRSS_BACKEND`.
3. The local report runner should thread this selection through all commands so
   that backend development can target:
   - a single test,
   - one module,
   - a marker lane,
   - or the default local report.

Preferred parity-validation shape when `cupy-rsoxs` implementation work is
active:

1. A CoreShell workflow variant that builds NumPy morphology fields and runs
   with `backend='cupy-rsoxs'`, `input_policy='coerce'`, and the parity
   ownership default. This validates the supported NumPy-to-CuPy input
   contract.
2. A second CoreShell workflow variant that builds CuPy-native morphology
   fields in backend-preferred layout and runs with `backend='cupy-rsoxs'`,
   `ownership_policy='borrow'`, and `input_policy='strict'`. This validates the
   no-stealth-transfer borrowed path.

## Runtime Observability and Safety Rails

1. Log explicit host<->device transfers at NRSS-controlled boundaries.
2. Add strict-mode warnings for policy-driven conversions and make
   resident-mode assumptions visible in development diagnostics.
3. The current stage-level timing model remains organized around the following
   serial optimization segments:
   - Segment A1: `Morphology` construction and contract normalization,
   - Segment A2: runtime morphology staging into backend compute space,
   - Segment B: n-field / tensor-character assembly,
   - Segment C: FFT, reorder, scratch reuse, and plan behavior,
   - Segment D: Ewald / scatter / projection math,
   - Segment E: rotation and angle accumulation,
   - Segment F: result-buffer assembly and retention,
   - Segment G: export and host conversion as a separate non-primary metric.
4. The internal timing control surface is private-only:
   - `Morphology._set_private_backend_timing_segments(...)`
   - `Morphology._clear_private_backend_timing_segments()`
   - do not expose this as public API without an explicit design decision.
5. Existing peak-memory monitoring remains acceptable for the current pass;
   per-stage memory instrumentation is deferred to follow-up work.
6. Best-effort only for third-party implicit copies; complete interception may
   not be possible.
7. Structural memory control comes before allocator tricks:
   - reuse scratch buffers,
   - delete/release intermediates as soon as they are dead,
   - only use allocator/pool trimming as an explicit lifecycle action, not as a
     hot-path substitute for sound ownership.
8. When interpreting peak GPU usage, treat resident morphology fields and live
   compute tensors as distinct from allocator/pool retention. Device-side
   residency can be an intentional policy choice rather than an allocator bug.
9. For large morphologies, phase-1 behavior should follow current CyRSoXS
   policy first; large-box-specific chunking or alternate projection strategies
   are optimization-stage follow-up work rather than parity-stage policy
   changes.

Detailed timing methodology, benchmark caveats, and experiment history live in
`CUPY_RSOXS_OPTIMIZATION_LEDGER.md`.

## Multi-GPU Execution Policy

Primary production pattern for this project:

1. Model-parallel execution (one model per GPU worker), typically via Ray.
2. Persistent workers for memory/plan reuse and lower startup overhead.
3. Keep internal energy-parallel multi-GPU optional and non-default.

Rationale:

1. Matches current objective-evaluation workload.
2. Avoids known instability concerns in CyRSoXS internal multi-GPU energy
   splitting.

## Current Implementation Touchpoints

High-value files for backend preparation and implementation:

### Runtime

1. `src/NRSS/morphology.py`
   - current `Morphology` / `Material` API
   - current CyRSoXS object creation and launch
   - current validator boundary
2. `src/NRSS/reader.py`
   - config/material parsing helpers
3. `src/NRSS/writer.py`
   - CLI/HDF5 compatibility helpers
4. `src/NRSS/__init__.py`
   - public export surface

### Tests

1. `tests/conftest.py`
   - pytest-wide backend selection and skipping logic
2. `tests/smoke/test_smoke.py`
   - fast contract tests and current runtime checks
3. `tests/validation/test_*.py`
   - physics validation modules
4. `tests/validation/lib/`
   - shared validation builders/reducers

### Local reporting/orchestration

1. `scripts/run_local_test_report.sh`
   - local report orchestration
2. `scripts/run_physics_validation_suite.py`
   - row-by-row physics runner

Recommended backend package layout under `src/NRSS/backends/`:

1. `src/NRSS/backends/__init__.py`
2. `src/NRSS/backends/registry.py`
3. `src/NRSS/backends/arrays.py`
4. `src/NRSS/backends/contracts.py`
5. `src/NRSS/backends/runtime.py`
6. `src/NRSS/backends/cyrsoxs.py`

Historical prep note worth preserving:

1. The first prep implementation did not need a complete backend adapter class
   hierarchy immediately.
2. It was acceptable to keep the current CyRSoXS execution code in
   `morphology.py` while removing import-time coupling and adding backend-aware
   selection/normalization.

## Risks To Watch

### Backward compatibility risk

The highest risk is breaking users who currently rely on:

1. default CyRSoXS execution,
2. `create_cy_object`,
3. direct access to `inputData` / `voxelData` / `scatteringPattern`,
4. current return behavior from `run()`.

### Test collection risk

1. Some validation modules import `CyRSoXS` at module scope.
2. Those imports must be made lazy or guarded, otherwise backend-selection work
   will still fail during test collection.

### Hidden conversion risk

Normalization at `Morphology` creation can hide expensive transfers if it is
not reported clearly. The coercion layer must record when it performed:

1. dtype casts
2. layout copies
3. host->device transfers
4. device->host transfers

### Scope creep risk

1. Do not mix parity bring-up with broader science-model redesign unless that
   redesign is required for backend decoupling.
2. Do not let optimization-first changes outrun the parity and validation
   contract.

### Runtime-state lifetime risk

For `cupy-rsoxs`, a cached singleton runtime would make it easy to create:

1. stale scratch buffers,
2. hidden cross-run coupling,
3. unclear cleanup boundaries,
4. memory retention that is hard to reason about.

The implementation phase should treat runtime/session ownership as explicit and
per-morphology unless a narrower shared cache is clearly proven safe.

### Result-lifetime and mutation risk

Once backend-native results and lazy conversion exist, NRSS must define what
happens if morphology data is mutated after:

1. preparation,
2. run,
3. result conversion/export.

Recommended direction:

1. Phase-1 should take the safer default and forbid morphology mutation after a
   result has been materialized unless and until an explicit invalidation API is
   implemented.
2. This freeze should begin at successful `run()` completion, not only at later
   xarray/export access.
3. If later mutation is re-enabled, treat post-run mutation as a
   result-invalidation event at minimum.

### Large-box parity risk

Large morphologies are one of the main reasons this backend exists, but they
also make it easy to accidentally introduce policy drift while trying to save
memory.

Recommended direction:

1. For parity, large-box behavior should initially mirror current CyRSoXS
   policy/limits rather than introducing new chunking or alternate projection
   policy at the same time as the backend rewrite.
2. Once parity and memory instrumentation are in place, large-box-specific
   optimizations can be evaluated as explicit follow-up work with separate
   thresholds.

## Resumption Checklist

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
5. Confirm whether `cupy-rsoxs` remains the default resolved backend when both
   runnable backends are installed.

## Explicit Non-Goals (Initial Phases)

1. Immediate production biaxial feature release.
2. Optimization-first changes before parity harness exists.
3. Simultaneous full parity for every legacy execution mode on day one.
