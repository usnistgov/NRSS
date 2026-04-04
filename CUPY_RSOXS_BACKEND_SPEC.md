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
- `CUPY_RSOXS_DIRECT_POLARIZATION_OPTIMIZATION.md`
  - path-specific optimization note for
    `execution_path='direct_polarization'`

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
13. Reduced-precision morphology handling is not modeled as a generic backend
    `dtype` knob.
14. The reduced-precision surface is a named mixed-precision mode, distinct
    from `execution_path`, and parity-sensitive compute remains
    `float32/complex64`.
15. Runtime instrumentation for timing/memory is desirable in development but
    must be fully disableable.
16. The primary optimization wall metric starts immediately before
    `Morphology(...)` construction and ends immediately after synchronized
    `run(return_xarray=False)` completion.
17. Large-box behavior should initially mirror current CyRSoXS policy during
    parity.
18. Phase-1 should prefer freezing morphology mutation after result creation
    until an explicit invalidation contract exists.
19. The mutation freeze should begin at successful `run()` completion.
20. Public optical constants may remain host-oriented for parity; backend
    staging to CuPy happens when math requires it.
21. Default parity contracts remain ZYX-shaped, C-contiguous, `float32`
    morphology arrays for each material field unless the mixed-precision mode
    is explicitly selected.
22. Initial `cupy-rsoxs` parity validation should include both:
    - a NumPy-input contract case using `input_policy='coerce'`,
    - a CuPy-native borrowed case using `ownership_policy='borrow'` and
      `input_policy='strict'`.
23. The mixed-precision mode is expert-only and double-gated:
    - the user must opt into the named mode explicitly,
    - and the submitted authoritative morphology arrays must already satisfy the
      mode's strict namespace and dtype contract.

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
3. Normalize namespace, mixed-precision contract, and device placement.

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
2. `cupy-rsoxs` does not expose a generic backend `dtype` knob.
3. Reduced-precision work is exposed only through a named mixed-precision mode
   whose purpose is morphology storage / transfer compression rather than a
   general backend compute-depth toggle.
4. In the mixed-precision mode, authoritative morphology storage is `float16`
   while FFT/q-space and projection math remain `float32` / `complex64`.
5. Avoid full pipeline compute in 16-bit when parity is required; the intended
   ladder is reduced-precision morphology handling followed by promotion into
   parity-sensitive FFT-ingress compute.

Note on cast cost:

1. GPU decode/cast is typically memory-bandwidth-bound and usually cheap
   relative to 3D FFT cost.
2. Cast overhead still needs profiling in end-to-end runs, but it is generally
   not the dominant term.

## Mixed-Precision Mode

This section defines the intended reduced-precision morphology path.

Public surface:

1. The option name should reflect that this is a specific mixed-precision mode
   rather than a backend dtype toggle. Preferred naming:
   `mixed_precision_mode`.
2. The default mode is `None` (off).
3. The first supported expert mode is
   `mixed_precision_mode='reduced_morphology_bit_depth'`.
4. `execution_path` and `mixed_precision_mode` are orthogonal backend-option
   surfaces.
5. The legacy backend `dtype` option should be removed rather than broadened.
6. Any attempt to use a generic backend `dtype` option for `cupy-rsoxs`
   should fail with a migration-focused error that points users to
   `mixed_precision_mode`.

Authoritative input contract:

1. This mode overrides the usual `input_policy` behavior and acts as strict
   regardless of the user's `input_policy` value.
2. No silent coercion should activate this mode.
3. Host-resident mode requires authoritative morphology fields to already be
   ZYX-shaped, C-contiguous, `numpy.float16` arrays.
4. Device-resident mode requires authoritative morphology fields to already be
   ZYX-shaped, C-contiguous, `cupy.float16` arrays.
5. Inputs that do not satisfy the namespace + dtype contract must fail early
   with a mode-specific error rather than being downcast automatically.

Runtime-compute intent:

1. The mixed-precision mode is a morphology-storage / transfer optimization.
2. It is not a declaration that the backend compute dtype is `float16`.
3. In this mode, authoritative morphology fields remain `float16` through
   `Morphology(...)` construction, validator checks, and backend runtime
   staging.
4. For both `execution_path='tensor_coeff'` and
   `execution_path='direct_polarization'`, the intended fast path carries those
   `float16` morphology inputs through the pre-FFT orientation decode and local
   field-composition work.
5. Current implementation detail:
   - this document describes the current code path, but that exact promotion
     boundary is not yet treated as a stable, maintained test-backed contract,
   - for `execution_path='tensor_coeff'`, the authoritative `float16`
     morphology inputs are decoded from half inside the `Nt` accumulation raw
     kernel,
   - that kernel writes the parity-sensitive tensor field directly into
     `complex64 Nt`,
   - the FFT then runs on that `complex64 Nt` field rather than on a separate
     `float16` or `float32` pre-FFT morphology buffer.
6. FFT/q-space and detector/projection math remain `float32` / `complex64` for
   parity-sensitive runs.

Validation contract:

1. The physics-relevant closure invariant is voxelwise closure, not a
   morphology-global total.
2. Mixed-precision closure should therefore be expressed explicitly as a
   voxelwise absolute-error bound on `abs(sum_i Vfrac_i - 1)`.
3. The mixed-precision closure check should operate in the authoritative dtype
   of the mode rather than allocating a widened full-volume validation buffer
   by default.
4. The initial mixed-precision closure budget is an expert-mode absolute
   tolerance of `1e-3` per voxel.
5. Other validator checks remain the same in spirit: populated fields, shape
   agreement, float dtype, finite values, and `0 <= S,Vfrac <= 1`.

## Mixed-Precision Implementation Plan

This section is the current approved implementation plan for the first
maintained mixed-precision fast path.

Implementation status as of April 4, 2026:

1. Completed in code for the initial runtime/contracts/tests pass:
   - removed the exposed generic `dtype` option from the public
     `cupy-rsoxs` backend-option surface,
   - implemented `mixed_precision_mode='reduced_morphology_bit_depth'`,
   - made mixed mode override `input_policy` and behave as strict,
   - implemented the mixed-mode voxelwise closure budget
     `abs(sum_i Vfrac_i - 1) <= 1e-3`,
   - kept authoritative and staged morphology handling at `float16` for both
     supported execution paths,
   - preserved `float32/complex64` FFT-ingress and parity-sensitive compute,
   - updated the maintained smoke suite to cover the new option surface and
     runtime behavior.
2. Additional implementation detail now documented as of April 4, 2026:
   - this is descriptive implementation-state documentation rather than a new
     stable parity contract,
   - in the current `tensor_coeff` mixed path, `float16` morphology arrays are
     not widened in a separate pass immediately before `cp.fft.fftn(...)`,
   - instead, the half-input raw kernels decode `Vfrac`, `S`, `theta`, and
     `psi` from half while accumulating directly into `complex64 Nt`,
   - the FFT-ingress object is therefore `complex64 Nt`, and the widening work
     is fused into `Nt` construction rather than split into a later
     full-volume conversion pass.
3. Intentionally deferred from this first implementation pass:
   - maintained validation-surface expansion,
   - CoreShell dev-harness updates for mixed-mode input generation,
   - graphical-abstract production and interpretation.
4. The initial implementation was verified with the full maintained smoke lane
   in `nrss-dev`.

Option-surface refactor:

1. Remove the exposed generic backend `dtype` option from the public
   `cupy-rsoxs` surface.
2. Keep `execution_path` as the execution-algorithm selector.
3. Add or preserve `mixed_precision_mode` as the only reduced-precision option
   surface for `cupy-rsoxs`.
4. Keep `execution_path` and `mixed_precision_mode` orthogonal.

Contract separation:

1. Do not model this feature as one backend-wide dtype.
2. Track at least three precision concepts explicitly:
   - authoritative morphology precision,
   - staged runtime morphology precision,
   - FFT-ingress / parity-sensitive compute precision.
3. Default mode keeps those surfaces at `float32`.
4. `mixed_precision_mode='reduced_morphology_bit_depth'` uses authoritative and
   staged morphology precision `float16`.
5. Current `tensor_coeff` implementation:
   - half morphology inputs are promoted during `Nt` construction,
   - `Nt` is accumulated directly as `complex64`,
   - and FFT ingress therefore begins at `complex64 Nt`.
6. This precise promotion boundary is currently documented implementation
   state, not a separately maintained test-backed contract.

Execution-path scope:

1. The first maintained mixed-precision pass must cover both supported
   execution paths:
   - `tensor_coeff`,
   - `direct_polarization`.
2. Neither path should widen morphology inputs back to `float32` during
   authoritative normalization or runtime staging.
3. The current mixed-precision execution-path ladder in maintained code is:
   - strict authoritative `float16` morphology inputs,
   - `float16` authoritative normalization and runtime staging,
   - path-local decode/composition work that consumes half inputs without
     widening the authoritative morphology arrays,
   - `float32/complex64` FFT and post-FFT math.
4. Current `tensor_coeff` implementation detail:
   - the half-to-float promotion occurs inside `Nt` construction, not in a
     standalone buffer-conversion step immediately before FFT.
5. The exact widening boundary in `direct_polarization` is allowed to evolve
   independently from `tensor_coeff` while the mixed-precision surface remains
   one public mode.

## Proposed Expert Approximation: `z_collapse_mode`

This section records the current expert-only approximation mode status.

As of April 4, 2026:

1. `z_collapse_mode` is now part of the normalized `cupy-rsoxs` option
   surface.
2. implemented mode:
   - `backend_options={"z_collapse_mode": "mean"}`
3. currently implemented for both maintained execution paths:
   - `execution_path='tensor_coeff'`
   - `execution_path='direct_polarization'`
4. the current `tensor_coeff` implementation collapses during `Nt`
   construction and therefore avoids materializing the full `3D` `Nt` tensor.
5. the current `direct_polarization` implementation collapses angle-specific
   `p_x`, `p_y`, and `p_z` fields during direct-field construction and avoids
   materializing the full active-path `3D` polarization volumes.
6. the current implementation intentionally rejects combination with
   `mixed_precision_mode`; that half-input combination remains future work.
7. maintained validation now includes a cupy-only analytical sphere collapse
   check in `tests/validation/test_analytical_sphere_form_factor.py`:
   - the collapse assertions are validated against direct analytical sphere
     `I(q)`,
   - not against the analytical flat-detector sphere surface,
   - and the maintained thresholds are tuned separately for the collapsed lane.

For the detailed implementation history, exploratory validation results, and
the recommended next-step plan, see `CUPY_RSOXS_Z_COLLAPSE_PROPOSAL.md`.

### Intent

Provide an opt-in fast approximate-physics path for arbitrary `3D`
morphologies by collapsing the locally composed field through `z` before FFT
and then continuing as an effective `z=1` problem.

For the full resumption blueprint, including first-pass implementation scope,
internal helper refactor recommendations, and the analytical-sphere
full-`3D`-versus-collapsed-`3D` validation proposal, see
`CUPY_RSOXS_Z_COLLAPSE_PROPOSAL.md`.

This mode is explicitly:

1. an approximation,
2. expert-only,
3. not a parity promise,
4. and expected to diverge more strongly as `z` heterogeneity and `q`
   increase.

### Tentative public surface

Preferred current naming:

1. `backend_options={"z_collapse_mode": "mean"}`

Current design intent:

1. default is off / `None`,
2. the first supported reduction would be `"mean"`,
3. `z_collapse_mode` should remain orthogonal to `execution_path`,
4. the original design intended orthogonality with `mixed_precision_mode`,
   but the current implementation deliberately rejects that combination,
5. and this should be documented as an approximation mode rather than an
   alternate exact execution algorithm.

### Semantics locked in for the current proposal

The following design choices were explicitly selected on April 4, 2026:

1. The mode should work on arbitrary `3D` boxes, not only native `z=1`
   morphologies.
2. The initial reduction should be `mean` through `z`.
3. The collapse should occur after local-field composition and before
   detector-windowing / FFT work.
4. After collapse, downstream execution should proceed as an effective `z=1`
   simulation.
5. The mode should be available for any `EAngleRotation` scheme.
6. The mode should be available for both maintained execution paths.

### Path-specific implementation sketch

For `execution_path='tensor_coeff'`:

1. current implementation:
   - accumulate directly into collapsed `(components, 1, y, x)` `Nt`
   - do not materialize full `3D` `Nt`
2. then continue with effective-`z=1` FFT and detector logic.

For `execution_path='direct_polarization'`:

1. current implementation:
   - build angle-specific collapsed `p_x`, `p_y`, `p_z` fields without
   materializing the full `3D` polarization volumes,
2. then continue with effective-`z=1` FFT and detector logic,
3. native-`z=1` identity coverage exists in the maintained smoke suite,
4. and maintained analytical sphere collapse validation now exists for the
   cupy-only paths.

### Current effective-`2D` behavior note

The current backend behavior for native `z=1` inputs is important context:

1. the Hann factor in `z` is identity for `z=1`,
2. but the backend does not simply return a raw `2D` FFT,
3. instead, it still evaluates detector-projection math on the single
   `qz=0` slice.

The proposed `z_collapse_mode` should therefore initially be understood as
"collapse to an effective `z=1` problem and reuse the current effective-`2D`
detector semantics," not as "collapse and expose the raw FFT."

### Validation and acceptance posture

This mode remains an expert-only approximation and not an exactness promise,
but it is no longer purely dev-only from a test posture standpoint.

Current planned evaluation order:

1. both maintained execution-path implementations now exist,
2. maintained analytical sphere form-factor tests pass with the normal paths
   unchanged,
3. maintained cupy-only analytical sphere collapse validation now exists
   against direct analytical `I(q)`,
4. the current exploratory comparison surface is still the full-`3D` versus
   collapsed-`3D` sphere `I(q)` dev harness plus generated plots,
5. and broader support claims or broader maintained-test promotion should
   still be revisited cautiously rather than assumed complete.

### Separate optimization thread: effective-`2D` detector simplification

Do not conflate `z_collapse_mode` with the separate optimization goal of a
simpler detector routine for effective-`2D` inputs.

That detector simplification should be treated as a distinct project because:

1. it should apply to native `z=1` morphologies even without approximation,
2. it may later also serve collapsed `z_collapse_mode` inputs,
3. and it should preserve current effective-`2D` semantics rather than
   redefining the approximation.

For current planning, treat this as low priority relative to the maintained
validation/test-first work and the more central backend upgrade items.

Implementation order:

1. Refactor backend option normalization and remove the old `dtype` option.
2. Update `Morphology(...)` normalization so mixed mode behaves as strict even
   if the user requested `input_policy='coerce'`.
3. Narrow the validator change to the mixed-mode closure rule.
4. Refactor runtime staging so host-resident `numpy.float16` morphology is
   transferred to `cupy.float16` without widening on CPU.
5. Implement the `tensor_coeff` mixed path so authoritative/staged morphology
   remains `float16` and any widening is fused into the last useful pre-FFT
   composition step rather than paying for a separate full-volume conversion
   pass.
6. Implement the `direct_polarization` mixed path with the same FFT-ingress
   promotion boundary.
7. Leave detector/projection math unchanged at `float32/complex64` in the
   first maintained pass.

Testing and harness plan:

1. Update the smoke suite so `dtype`-option rejection tests are replaced by
   mixed-mode option, strict-input, and closure-budget tests.
2. Extend the maintained CoreShell helper and the `cupy-rsoxs` optimization
   matrix harness so they can emit strict authoritative `numpy.float16` and
   `cupy.float16` morphology inputs on demand.
3. The optimization harness should compare, at minimum:
   - standard host,
   - mixed host,
   - standard device,
   - mixed device,
   across both `tensor_coeff` and `direct_polarization`.
4. Runtime reports should expose enough metadata to confirm that the submitted
   authoritative morphology stayed `float16` through normalization and staging,
   and to document where the current implementation widens into FFT-ingress
   compute.

Graphical-abstract plan:

1. Produce a CoreShell graphical abstract that compares the standard
   `tensor_coeff` path against the mixed-precision `tensor_coeff` path.
2. The figure should be built around direct user inspection of precision loss,
   not only summary metrics.
3. Recommended figure content:
   - timing comparison panels across representative CoreShell sizes,
   - `A(E)` overlay,
   - `A(q)` overlays at `284.7` and `285.2 eV` with residual inset,
   - a heatmap of relative drift metrics,
   - at least one detector-image delta panel for a canonical small case.
4. The same harness may optionally emit a parallel `direct_polarization`
   comparison, but the required published abstract for this plan is the
   standard-versus-mixed `tensor_coeff` comparison.

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
   - no backend `dtype` knob,
   - `execution_path` for `cupy-rsoxs`,
   - a named `mixed_precision_mode` for the approved reduced-precision
     morphology plan in `cupy-rsoxs`.
4. Current contract table:
   - `cyrsoxs`: authoritative/runtime namespace `numpy`, device `cpu`, default
     dtype `float32`
   - `cupy-rsoxs`: default authoritative namespace `numpy` in
     `resident_mode='host'`, runtime namespace `cupy`, optional authoritative
     namespace `cupy` in `resident_mode='device'`, default authoritative
     morphology precision `float32`, parity-sensitive runtime compute
     `float32/complex64`, and an expert-only mixed-precision mode that requires
     authoritative `float16` inputs in the correct namespace

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
