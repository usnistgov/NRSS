# NRSS Backend Modernization Spec (Superseding + Handoff)

This document supersedes prior roadmap content in this file and is written to be resumable in a fresh context.

## 1. Goal

Build a new NRSS backend architecture that:

1. Preserves trusted physics behavior (CyRSoXS parity first).
2. Enables a CuPy-native simulation path to avoid avoidable GPU->CPU->GPU transfers.
3. Supports optional future backends (PyTorch, JAX) without hard dependencies.
4. Improves robustness with deterministic, reproducible regression testing.
5. Provides durable value even if alternate backend implementation is delayed (test hardening alone is a success milestone).

## 2. Physics Reference (Ground Truth Equations)

These equations define the numerical target. Phase 1 (CuPy mimic) follows current CyRSoXS implementation details and conventions.

1. Far-field scattering intensity (detector projection of Fourier-space polarization):

   dσ/dΩ is proportional to |k² (I - r_hat r_hat) · p(q)|².

2. Fourier transform of induced polarization:

   p(q) = ∫ exp(i q·r) p(r) d³r.

3. Scattering vector magnitude from wavelength and scattering angle:

   The magnitude of q is |q| = (4π/λ) sin(θ/2).

4. Local induced polarization from susceptibility tensor and incident field:

   p(r) = ε₀ χ(r) · Ê.

5. Uniaxial constitutive model (principal frame view):

   The local susceptibility is χ_local = diag(χ_ord, χ_ord, χ_ext).

6. Lab-frame tensor is obtained by rotating local-frame tensor with morphology orientation (Euler in current workflows), then applying composition/volume-fraction weighting per material and energy.

Notes:

1. Exact Euler convention and rotation order must match CyRSoXS for parity.
2. The CuPy mimic backend is required to copy CyRSoXS math order, not "improve" it initially.

## 3. CyRSoXS-Mimic Computational Pipeline (Required Mapping)

This section is the implementation handoff for Phase 1.

### 3.1 Stage A: Input normalization and policy resolution

1. Accept morphology fields (`vfrac`, `S`, `theta`, `psi`; optionally vector inputs).
2. Resolve backend (`cyrsoxs`, `cupy`, future backends), `input_policy`, `output_policy`.
3. Normalize dtypes and device placement.

Memory slimming:

1. Do not duplicate morphology arrays unless policy requires copy.
2. Keep one authoritative device-resident view for each field.

GPU tuning opportunities:

1. Use zero-copy where possible (`__cuda_array_interface__`, DLPack boundaries).
2. Fuse trivial cast/scale ops into downstream kernels.

### 3.2 Stage B: Orientation decode

1. Convert Euler representation to orientation direction field n(r) using CyRSoXS-consistent convention.
2. If vector input is supplied, skip decode and validate normalization/shape.

Memory slimming:

1. Avoid persisting both Euler-derived vectors and equivalent expanded tensors unless needed.
2. Reuse scratch buffers across energies/angles.

GPU tuning opportunities:

1. Use fused elementwise kernels for trig + normalization.
2. Keep decode on device; avoid host round-trips for validation.

### 3.3 Stage C: Local polarization field composition

1. Build local susceptibility/polarization components from morphology + optical constants + incident polarization.
2. Produce component fields consumed by FFT stage (current logic mimic first).

Memory slimming:

1. Stream by energy/angle chunks; avoid materializing a full [energy, angle, xyz, components] tensor at once.
2. Free/recycle polarization component buffers immediately after FFT contribution is consumed.

GPU tuning opportunities:

1. Fuse composition math in custom kernels to reduce global-memory traffic.
2. Prefer SoA-style component layout when it improves coalesced reads in subsequent FFT prep.

### 3.4 Stage D: FFT and reciprocal-space conversion

1. Transform spatial polarization fields to p(q) via cuFFT-backed operations.
2. Apply required shift/DC handling consistent with CyRSoXS.

Memory slimming:

1. Reuse cuFFT work buffers and plans.
2. Perform in-place transforms where safe.
3. Do not retain pre-FFT buffers once transformed data is consumed.

GPU tuning opportunities:

1. Plan caching and batched FFT shapes.
2. Keep FFT input/working tensors in `float32/complex64` for parity-sensitive runs.
3. Optimize layout/strides to avoid internal transposes.

### 3.5 Stage E: Detector projection / Ewald handling

1. Apply the projection operator (I - r_hat r_hat) to p(q).
2. Compute detector intensity contribution for each required geometry.

Memory slimming:

1. Accumulate directly into output/objective buffers; avoid temporary full-size detector stacks when not required.
2. Chunk detector or q-regions if peak memory is dominated by result tensors.

GPU tuning opportunities:

1. Kernel fusion for projection + norm-squared accumulation.
2. Use read-only cached loads for geometry tables reused across voxels.

### 3.6 Stage F: Rotation, angle accumulation, and export

1. Rotate/accumulate across sample angles and energies per current semantics.
2. Emit output by policy (`numpy`, backend-native, objective-only).

Memory slimming:

1. Drop intermediate polarization fields before final result tensor growth.
2. Prefer objective-only mode for fitting loops to minimize resident memory.
3. Allow streaming writes/checkpointing for large result tensors.

GPU tuning opportunities:

1. Keep reduction operations on device.
2. Use backend-native reductions and avoid host synchronization inside hot loops.

## 4. Peak Memory Model (Planning Equations)

Define:

1. N = Nx * Ny * Nz voxels.
2. M = detector_nx * detector_ny detector pixels.
3. E = n_energies, A = n_angles, and P = n_polarizations.

Approximate resident memory terms:

1. Morphology storage: B_morph ~ N * C_morph * b_morph, where C_morph = 5 for (`vfrac`, `S`, `theta`, `psi`, plus optional mask/material-index representation).
2. Polarization working set (worst-case): B_p ~ N * C_p * b_p, with C_p = 3 vector components (real/complex by stage).
3. FFT workspace: B_fft ~ α * B_p, where α depends on cuFFT plan and shape.
4. Result tensor (dominant in many workflows): B_res ~ M * E * A * P * b_res.

Operational policy:

1. Optimize for the peak footprint, B_morph + B_p + B_fft + B_res.
2. Release B_p and B_fft aggressively before B_res expansion.
3. Introduce chunk order default: energy -> angle -> detector/q when needed.
4. Permit high utilization (up to ~95% of 48 GB) only under explicit guardrail configuration.

## 5. Precision Policy

1. Default compute precision: `float32` / `complex64`.
2. Morphology storage in `float16`/`bfloat16` is allowed as a compression path.
3. Decode/cast should occur in device kernels near use sites.
4. FFT/q-space and projection math remain `float32/complex64` for parity-sensitive runs.
5. Avoid full pipeline compute in 16-bit when parity is required (high-q deviations known risk).

Note on cast cost:

1. GPU decode/cast is typically memory-bandwidth-bound and usually cheap relative to 3D FFT cost.
2. Cast overhead still needs profiling in end-to-end runs, but it is generally not the dominant term.

## 6. Representation Roadmap (Euler, Vector, Tensor)

### 6.1 Phase 1 behavior

1. Keep Euler-first compatibility (dominant workflows).
2. Accept vector input for backward compatibility.
3. Internals mimic CyRSoXS logic first.

### 6.2 Phase 2 refactor target

1. Move to cleaner tensor-character internals after parity lock.
2. Candidate uniaxial order tensor form:

   Q = S (n ⊗ n - I/3).

3. Candidate susceptibility decomposition:

   χ = χ_iso I + Δχ Q (or equivalent project-specific parameterization).

### 6.3 Symmetric tensor storage compression

1. Full 3x3 tensor has 9 entries, 6 unique for symmetric form.
2. Store symmetric tensors in packed 6-component form (xx, yy, zz, xy, xz, yz) to reduce memory.
3. Reconstruct needed matrix elements in fused kernels instead of materializing dense 3x3 arrays globally.

### 6.4 Biaxial scaffold

1. Add constitutive interface now so biaxial models can be added without backend redesign.
2. Biaxial may initially exist only in alternate backends if CyRSoXS parity path remains uniaxial.

## 7. Backend Strategy and Ranking

Recommended implementation order:

1. CuPy: best parity path with current CUDA/cuFFT workflow and minimal conceptual translation.
2. PyTorch: good GPU kernel ecosystem and deployment maturity for later integration.
3. JAX: strong for compiled/fused workflows, but higher complexity for this parity-first migration.

Important backend variant(s):
approximate but accelerated computation could be done by computing a 3D model up to the point of the 3D p-fields, then collapsing the Z-axis by sum or mean (Z-axis projection), then continuing on the 2D FFT computation track to results. This may be accurate enough for many users and could be significantly faster for some jobs. The legacy cyrsoxs engine does not have this capability. Ideally, this could be incorporated into the above implementations with a flag that is ignored with warning by backends that don't support it (like cyrsoxs)

TensorFlow is deprioritized for this project.

## 8. Test-First Program (Highest Priority)

### 8.1 Immediate objective

Convert `tests/validation/` legacy scripts into robust pytest suites using pybind CyRSoXS execution where applicable (no CLI serialization bottleneck), while establishing a first stable physics-validation lane before backend refactors.

### 8.2 Maintained validation cases

1. Analytical sphere form factor.
2. Sphere contrast scaling.
3. Sphere orientational contrast scaling.
4. Analytical 2D disk form factor.
5. 2D disk contrast scaling.
6. 2D and 3D Bragg lattice peak-position validation.
7. Core-shell.
8. MWCNT.

### 8.3 Required test qualities

1. Deterministic fixtures and fixed RNG seeds if randomness appears anywhere.
2. Explicit metadata capture (versions, geometry, dtype, parameter hashes, backend flags).
3. Machine-readable golden references generated from trusted pybind runs.
4. CPU smoke is intentionally limited to validator, I/O, and API-contract behavior; it is not a CPU physics-parity lane.

### 8.4 Parity metrics (layered)

1. Objective scalar parity (for fitting-style workflows): target <= 1% relative error.
2. Radial I(q) parity with q-window-specific rtol/atol.
3. Peak-position parity with absolute q tolerance.
4. Optional image-space checks on masked finite support.

Initial threshold table is intentionally provisional; calibrate from empirical baseline variance before final gating.

### 8.5 Golden data governance

1. Generate now from trusted current reference.
2. Regenerate only when confirmed physics/scientific bug fixes intentionally change expected output.
3. Require changelog note + reviewer signoff for any golden update.

### 8.6 Analytical guardrail track (projected sphere)

1. This track is now implemented as `tests/validation/test_analytical_sphere_form_factor.py`.
2. Current implementation uses pybind execution plus PyHyperScattering reduction, compares against a flat-detector analytical reference, and evaluates both pointwise agreement and all-minima alignment.
3. Geometry is currently fixed at `512^3`, `PhysSize = 1.0 nm`, diameters `70 nm` and `128 nm`, with optional superresolution support retained for future follow-up.
4. Treat this as a guardrail rather than strict equality because discretization and finite resolution still perturb high-q behavior.

### 8.7 Implemented smoke harness (March 17, 2026)

1. Added `tests/smoke/test_smoke.py` for deterministic environment/import checks, morphology validation checks, pybind runtime coverage, PyHyperScattering integration, CLI-vs-pybind parity smoke, and GPU config/E-angle semantics smoke.
2. Added `scripts/run_local_test_report.sh` to standardize local execution and emit timestamped metadata/log/summary artifacts under `test-reports/`.
   - Default conda env is now `nrss-dev` unless overridden with `-e/--env` or `NRSS_TEST_ENV`.
   - Standard lanes can be skipped with `--skip-defaults`.
   - Explicit `--cmd` entries can be repeated with `--repeat N` for brittleness sweeps and injected-build validation.
3. Added pytest marker declarations in `pyproject.toml` for `smoke`, `cpu`, `gpu`, `slow`, `physics_validation`, `experimental_validation`, and `toolchain_validation`.
4. Added `tests/conftest.py` to default tests to a single visible GPU when the environment is otherwise unset, improving reproducibility and avoiding known CyRSoXS multi-GPU instability during energy fan-out.

Latest run evidence:

1. Command: `bash scripts/run_local_test_report.sh --stop-on-fail`
2. Timestamp (UTC): `20260320T134227Z`
3. Result: `4/4` steps passed
4. CPU smoke: `12 passed, 10 deselected`
5. GPU smoke: `10 passed, 12 deselected`
6. Physics validation: this early snapshot is superseded by the later expanded lane below.

### 8.8 Implemented physics validation layer (March 20, 2026)

1. Added `tests/validation/test_analytical_sphere_form_factor.py`:
   - flat-detector analytical sphere comparison through the pybind-to-PyHyper workflow,
   - pointwise and minima-alignment metrics with fixed empirical thresholds,
   - explicit sphere-versus-vacuum morphology,
   - optional plot writing gated by `NRSS_WRITE_VALIDATION_PLOTS=1`.
2. Added `tests/validation/test_sphere_contrast_scaling.py`:
   - one-morph, multi-energy contrast-scaling validation,
   - 24 close-energy scenarios covering beta-only, delta-only, mixed, and split-material families,
   - integrated-intensity checks over a fixed q window with fixed empirical thresholds.
3. Added `tests/validation/lib/orientational_contrast.py`:
   - reusable tensor-based helper that turns para/perp delta/beta channels plus Euler angles and `S` into inspectable effective indices, induced polarization vectors, and Eq. 15/16-style far-field contrast predictions,
   - explicitly documents the How-to-RSoXS citation plus the rotation / far-field projection path used for expectations.
4. Added `tests/validation/test_sphere_orientational_contrast_scaling.py`:
   - one-morph, multi-energy orientational-contrast validation for a sphere in vacuum,
   - `128 x 128 x 128`, `PhysSize = 2.0 nm`, `Diameter = 32 nm`,
   - close-energy pure-delta, pure-beta, and mixed dichroic families,
   - high-symmetry `theta` and `psi` coverage, low-symmetry coupled Euler cases, and an `S` series including `S=0`,
   - helper-driven expected ratios plus direct detector-annulus observed ratios,
   - optional plot writing through `NRSS_WRITE_VALIDATION_PLOTS=1`.
5. Added `tests/validation/test_analytical_2d_disk_form_factor.py`:
   - direct analytical 2D disk comparison through the pybind-to-PyHyper workflow,
   - `1 x 2048 x 2048`, `PhysSize = 1.0 nm`, diameters `70 nm` and `128 nm`,
   - pointwise and minima-alignment metrics with fixed empirical thresholds,
   - explicit disk-versus-vacuum morphology,
   - fixed `sr=1` only, mirroring the sphere test’s assertion anchor while avoiding extra 2D-path variability,
   - optional plot writing gated by `NRSS_WRITE_VALIDATION_PLOTS=1`.
6. Added `tests/validation/test_2d_disk_contrast_scaling.py`:
   - one-morph, multi-energy contrast-scaling validation for the 2D pathway,
   - `1 x 2048 x 2048`, `PhysSize = 1.0 nm`,
   - 24 close-energy scenarios covering beta-only, delta-only, mixed, and split-material families,
   - integrated-intensity checks over a fixed q window with fixed empirical thresholds.
7. Added `tests/validation/lib/bragg.py`:
   - shared deterministic lattice builders and reciprocal-space prediction helpers for Bragg validation,
   - supports square/hexagonal 2D disk lattices and simple-cubic/HCP 3D sphere lattices,
   - keeps explicit vacuum as the second material and uses float-center local stamping for morphology construction.
8. Added `tests/validation/test_bragg_2d_lattice.py`:
   - deterministic square (`a = 30 nm`) and hexagonal (`a = 45 nm`) disk lattices at `1 x 2048 x 2048`, `PhysSize = 1.0 nm`,
   - validates detector-peak locations and quasi-powder shell locations through the pybind-to-PyHyper workflow,
   - includes verbose diagnostic plots with full predicted-shell overlays.
9. Added `tests/validation/test_bragg_3d_lattice.py`:
   - deterministic simple-cubic (`a = 30 nm`) and ideal HCP (`a = 45 nm`) sphere lattices at `256 x 1024 x 1024`, `PhysSize = 1.0 nm`,
   - validates detector-visible 3D Bragg peak locations plus azimuthally averaged shell locations,
   - uses explicit flat-detector geometry handling for shell prediction and includes verbose diagnostic plots with visibility-class overlays.
10. Added `tests/validation/lib/core_shell.py` plus `tests/validation/test_core_shell_reference.py`:
   - maintained CoreShell baseline workflow through pybind + PyHyperScattering `WPIntegrator` + manual A-wedge reduction,
   - experimental PGN RSoXS golden as the scientific gate,
   - parallel sim-derived golden as a tight regression guard,
   - `experimental_validation` marker applied to the experimental-reference test,
   - falsification/subterfuge scenarios intentionally kept only in the development diagnostic, not in the principal `tests/validation` surface.
11. Added `tests/validation/lib/mwcnt.py` plus `tests/validation/test_mwcnt_reference.py`:
   - maintained deterministic MWCNT workflow through pybind + PyHyperScattering `WPIntegrator` + anisotropy-observable reduction,
   - periodic field construction is now the maintained default and the legacy field path remains available as an explicit switch,
   - maintained simulation defaults are `WindowingType=0` and `EAngleRotation=[0, 20, 340]`,
   - experimental reduced `A(E)` / `A(q)` observables derived from the tutorial/manuscript workflow are the scientific gate,
   - `experimental_validation` marker applied to the official MWCNT test,
   - manuscript Table I provenance plus realized fixed-seed geometry statistics are exposed in the maintained validation plots and helper metadata,
   - development-only MWCNT falsification/threshold probes stay under `tests/validation/dev/`, not in the principal `tests/validation` surface.
12. Archived one-off exploratory validation code under `scripts/validation_diagnostics/` so it remains available for future archaeology without polluting pytest collection.
    - this directory now also holds `orientational_contrast_tiny_diagnostic.py`, the development-only preserved `64^3` probe that preceded the official orientational test,
    - and `sphere_orientational_contrast_diagnostic.py`, an opt-in artifact generator that writes orientational ratio plots plus TSV summaries under `test-reports/sphere-orientational-contrast-dev/`.
    - and `core_shell_reference_diagnostic.py`, the opt-in CoreShell artifact generator that also owns the falsification/subterfuge comparisons.
13. Extended `scripts/run_local_test_report.sh` to include the marker-based `physics_validation` lane in the standard local report, while also supporting `--skip-defaults` plus repeated explicit `--cmd` runs for targeted validation and stochastic-failure checks. Newly added physics modules are therefore included automatically.
    - physics-test report summaries now retain full docstring descriptions rather than only the first line,
    - and targeted custom physics commands now resolve per-test statuses in the markdown report instead of falling back to `DESELECTED`.
    - the report now also captures imported NRSS module resolution plus a hashed manifest of vendored validation-reference artifacts under `tests/validation/data/`.
14. Targeted local validation against an injected fixed CyRSoXS pybind build removed the prior same-process 2D analytical disk stochastic failure in local testing:
   - one-process back-to-back `70 nm` then `128 nm` analytical 2D disk validation passed `20/20` repeated runs on a single visible GPU,
   - the shipped pytest module also passed cleanly against the injected build,
   - interpret this as local evidence that the 2D-path failure was upstream to NRSS rather than a remaining deterministic NRSS harness issue.
15. Latest default local report evidence for the expanded suite:
   - command: `CUDA_VISIBLE_DEVICES=0 bash scripts/run_local_test_report.sh --stop-on-fail`,
   - timestamp/report: `20260322T140310Z` / `test-reports/20260322T140310Z`,
   - result: `4/4` steps passed,
   - physics-validation lane inside the report: `14 passed`, including both CoreShell tests and the new MWCNT experimental test,
   - the generated markdown summary lists both experimental-reference tests with their `experimental_validation` markers and full scientific citation blocks.
16. A targeted local CoreShell-only run confirmed the new official module passes cleanly on its own:
   - command: `CUDA_VISIBLE_DEVICES=0 /home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/validation/test_core_shell_reference.py -v`,
   - result: `2 passed`.
17. A targeted local MWCNT-only run confirmed the new official module passes cleanly on its own:
   - command: `CUDA_VISIBLE_DEVICES=0 /home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/validation/test_mwcnt_reference.py -v`,
   - result: `1 passed`,
   - a development-only threshold probe showed that nearby radius falsifications fail the maintained thresholds while a moderate orientation broadening can still pass, so no threshold tightening was applied.
18. A targeted installed-build report run also confirmed that the new orientational module is described correctly in `summary.md`:
   - command: `bash scripts/run_local_test_report.sh --skip-defaults --cmd "python -m pytest tests/validation/test_sphere_orientational_contrast_scaling.py -m physics_validation -v"`,
   - timestamp/report: `20260321T190619Z` / `test-reports/20260321T190619Z`,
   - result: `1 passed`,
   - the “Physics Tests” section now includes the full orientational test description and a `PASSED` status.
19. Installed-package cross-check for the earlier Bragg coverage also passed:
   - command: `CUDA_VISIBLE_DEVICES=1 /home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/validation/test_bragg_2d_lattice.py tests/validation/test_bragg_3d_lattice.py -v`,
   - result: `4 passed in 126.03s`,
   - installed package resolved to `CyRSoXS 1.1.8.0`, patch `9d45790`.

### 8.9 Remaining test-hardening gaps

1. Add CI gating policy for CPU smoke and GPU smoke/parity/physics lanes.
2. Keep the CPU smoke lane focused on validator, I/O, and API-contract behavior rather than CPU physics parity.
3. Add backend-contract tests only when alternate backend implementation work begins.

## 9. Input/Output Contract and Compatibility

1. `backend`: selects execution backend.
2. `input_policy`: governs host/device acceptance and conversion behavior.
3. `output_policy`: `numpy`, backend-native arrays, or objective-only.

Contract requirements:

1. Preserve current default behavior for existing users.
2. Provide explicit conversion methods (`as_numpy`, backend-native accessors).
3. Support GPU-resident objective returns for fitting workflows to avoid forced host copies.
4. Avoid implicit copies; when unavoidable, emit diagnostics in strict mode.
5. Expose explicit ownership behavior for morphology inputs rather than hiding
   copy-vs-borrow inside backend internals.
6. Publish and enforce a backend-preferred morphology-array contract for direct
   CuPy inputs.

Phase-1 lock-ins for `cupy-rsoxs`:

1. Parity target output remains xarray-compatible and NumPy/PyHyperScattering-friendly.
2. Host transfer for parity output should happen lazily at result read/conversion time rather than eagerly during compute if practical.
3. Backend-native/device-resident result access is deferred until after parity, but the result abstraction should be designed so it can be added without breaking the parity API.
4. The result object should be allowed to outlive the rest of the morphology/runtime state when feasible; do not require users to keep the full morphology alive just to retain scattering output.
5. Add a lightweight result-wrapper direction now:
   - backend-owned internal result storage,
   - explicit `to_xarray()` for parity workflows,
   - later `to_backend_array()` / device-resident accessors.
6. Provide an explicit release path so users can keep only the result object and
   drop morphology/runtime working state when desired.
7. Add an explicit `ownership_policy` surface in v1 with `borrow` and `copy`.
8. `cupy-rsoxs` parity development should use `ownership_policy='borrow'`.
9. Preferred direct-CuPy morphology contract for parity is ZYX-shaped,
   C-contiguous, `float32` arrays for each material field.

## 10. Optional Dependency and Packaging Plan

1. Base NRSS install must not force GPU backend stacks.
2. Use extras for backend deps (for example `nrss[cupy]`, `nrss[torch]`, `nrss[jax]`).
3. Implement lazy backend imports with actionable error messages.
4. Keep CPU-only/base path installable and testable independently.
5. Treat conda CUDA compatibility fragility as an external risk; do not tie core installability to one GPU package line.

## 11. Multi-GPU Execution Model

Primary production pattern for this project:

1. Model-parallel execution (one model per GPU worker), typically via Ray.
2. Persistent workers for memory/plan reuse and lower startup overhead.
3. Keep internal energy-parallel multi-GPU optional and non-default.

Rationale:

1. Matches current objective-evaluation workload.
2. Avoids known instability concerns in CyRSoXS internal multi-GPU energy splitting.

Phase-1 parity lock-in:

1. `cupy-rsoxs` parity development and required parity tests target single-GPU execution only.
2. Internal multi-GPU energy fan-out is explicitly deferred until after parity because current CyRSoXS multi-GPU behavior is known to be unstable in some workflows.

## 12. Runtime Observability and Safety Rails

1. Log explicit host<->device transfers at NRSS-controlled boundaries.
2. Add strict mode warnings for policy-driven conversions and make resident-mode
   assumptions visible in dev diagnostics.
3. The implemented primary optimization wall metric is:
   - start immediately before `Morphology(...)` construction, after upstream
     field arrays already exist,
   - end immediately after synchronized `run(return_xarray=False)` completion.
4. The implemented primary optimization wall metric excludes:
   - morph-field generation before object creation,
   - result export such as `to_xarray()`,
   - downstream A-wedge generation, plotting, or analysis.
5. For CuPy-native upstream workflows, the dev timing harness forces a default
   stream synchronize immediately before the start timestamp so unfinished
   upstream GPU work is not counted inside morphology timing.
6. The current timing pass records:
   - Segment `A1` in the harness with wall-clock timing for `Morphology(...)`
     construction / contract normalization,
   - Segment `A2` inside `cupy-rsoxs` with private wall-clock timing for
     runtime morphology staging,
   - Segments `B-F` inside `cupy-rsoxs` with private CUDA-event timing,
   - no timing payload at all unless timing is explicitly enabled,
   - no Segment `G` / export timing in this pass.
7. The internal timing control surface is private-only:
   - `Morphology._set_private_backend_timing_segments(...)`
   - `Morphology._clear_private_backend_timing_segments()`
   - do not expose this as public API without an explicit design decision.
8. Existing peak-memory monitoring remains acceptable for this pass; per-stage
   memory instrumentation is deferred to follow-up work.
9. Best-effort only for third-party implicit copies; complete interception may
   not be possible.

Additional phase-1 requirements:

1. Instrumentation is now opt-in/internal and must remain disableable with
   effectively zero extra synchronization in non-dev runs.
2. Memory observability follow-up should cover both:
   - post-run cleanup behavior,
   - peak and per-stage resident usage during compute, especially around polarization, FFT, and Ewald/result stages.
3. Stage-level timing work should continue to be organized around the following
   serial optimization segments:
   - Segment A1: `Morphology` construction and contract normalization,
   - Segment A2: runtime morphology staging into backend compute space,
   - Segment B: n-field / tensor-character assembly,
   - Segment C: FFT, reorder, scratch reuse, and plan behavior,
   - Segment D: Ewald / scatter / projection math,
   - Segment E: rotation and angle accumulation,
   - Segment F: result-buffer assembly and retention,
   - Segment G: export and host conversion as a separate non-primary metric,
     intentionally deferred in the current timing pass.
4. The parity implementation should prefer structural memory control first:
   - reuse scratch buffers,
   - delete/release intermediates as soon as they are dead,
   - only use allocator/pool trimming as an explicit lifecycle action, not as a hot-path substitute for sound ownership.
5. When interpreting peak GPU usage, treat resident morphology fields and
   live compute tensors as distinct from allocator/pool retention. Device-side
   residency can be an intentional policy choice rather than an allocator bug.
6. For large morphologies, phase-1 behavior should follow current CyRSoXS
   policy first; large-box-specific chunking or alternate projection strategies
   are optimization-stage follow-up work rather than parity-stage policy
   changes.
7. Optical constants may remain host-oriented in the public API for parity, but
   the backend should materialize the small per-energy tensors onto device when
   needed for device-side math.

## 13. Development Workflow

### 13.1 Source control

1. Use feature branches.
2. Keep commits small and frequent.
3. Merge milestone-sized PRs.

### 13.2 Environment practice

1. Keep stable scientific environment untouched.
2. Use dedicated dev environment(s) with editable NRSS install.
3. Pin versions during parity development.

### 13.3 Notebook and staging workflow

1. Use notebooks for exploratory visualization and diagnostics.
2. Keep an external staging workspace for baseline generation and comparison artifacts.
3. Require script-based reproducibility for official golden generation and CI-bound regression assets.

## 14. Phased Delivery Plan

1. Test-hardening milestone: pybind golden baselines + deterministic harness.
2. Phase 1: CuPy backend that mimics CyRSoXS algorithmic flow.
   - Phase 1a: low-memory (`AlgorithmType=1`) implementation first, because it is the best immediate path for GPU headroom learning and memory discipline.
   - Phase 1b: communication-minimizing (`AlgorithmType=0`) implementation to parity using lessons from the low-memory path.
   - Phase 1 completion requires both algorithm paths to be runnable and parity-tested, even if one lands first.
   - Phase 1 parity scope is Euler-only, `ScatterApproach::PARTIAL` only, and single-GPU only.
3. Phase 2: Internal math cleanup (tensor-character refactor) while preserving parity tests.
4. Phase 3: Additional backends behind shared backend contract.

Independent success criterion:

1. Completion of the test-hardening milestone is a meaningful modernization outcome even if backend phases are delayed.

## 15. Explicit Non-Goals (Initial Phases)

1. Immediate production biaxial feature release.
2. Optimization-first changes before parity harness exists.
3. Simultaneous full parity for every legacy execution mode on day one.

## 16. Open Decisions for Follow-Up Interview

1. Final parity threshold table by metric and q-region.
2. Golden dataset size/retention strategy in repository vs external artifacts.
3. GPU CI strategy and minimum gating matrix.
4. Release performance gates (what is measured and acceptable drift).
5. Detailed objective-function API and on-device return contract for fitting pipelines.

## 17. Locked Decisions For `cupy-rsoxs` Implementation Start

The following are now considered locked unless later planning explicitly reopens them:

1. Parity target is the CyRSoXS math path, not the pybind ingestion path.
2. Phase-1 parity is Euler-only.
3. Phase-1 parity uses `ScatterApproach::PARTIAL` only.
4. Phase-1 parity is single-GPU only.
5. Start with literal rotation and angle-accumulation semantics for parity; any leaner/faster alternatives are optimization-track follow-ups only.
6. Low-memory path may be implemented first, but parity is not declared complete until both algorithm paths are supported.
7. `cupy-rsoxs` should expose controllable resident modes.
8. Host-resident staged mode is the default guidance for public workflows.
9. Device-resident direct mode is an opt-in path for already-CuPy morphology
   fields and is expected to use more GPU memory.
10. Resident mode, `input_policy`, and `ownership_policy` are distinct concepts
    and should remain separately documented.
11. Parity output contract is xarray-compatible and NumPy/PyHyperScattering-friendly.
12. Backend-native/on-device result access is deferred until after parity, but the result abstraction should preserve that path.
13. `float16` is deferred until after parity; parity-sensitive compute remains `float32/complex64`.
14. Runtime instrumentation for timing/memory is desirable in development but must be fully disableable.
15. The primary optimization wall metric starts immediately before `Morphology(...)` construction and ends immediately after synchronized `run(return_xarray=False)` completion.
16. Large-box behavior should initially mirror current CyRSoXS policy during parity.
17. Phase-1 should prefer freezing morphology mutation after result creation until an explicit invalidation contract exists.
18. The mutation freeze should begin at successful `run()` completion.
19. Public optical constants may remain host-oriented for parity; backend staging to CuPy happens when math requires it.
20. Preferred direct-CuPy morphology contract for parity is ZYX-shaped, C-contiguous, `float32` arrays for each material field.
21. Initial `cupy-rsoxs` parity validation should include both:
    - a NumPy-input contract case using `input_policy='coerce'`,
    - a CuPy-native borrowed case using `ownership_policy='borrow'` and `input_policy='strict'`.

## 18. Post-Parity Optimization Guidance And Ledger For `cupy-rsoxs`

This section replaces the earlier optimization-inventory framing with a more
resumable guidance-and-ledger record. It is intended to let a fresh context
recover:
- the current accepted backend state,
- the measurement caveats,
- the official resident-mode guidance,
- the optimization campaign strategy,
- and the work that has already been tried, accepted, or rejected.

### 18.1 Current state and authoritative artifacts

1. The current accepted tuned backend state is the behavior reached after the
   first optimization campaign:
   - dead `Nt[5]` removed from the supported Euler-only / `PARTIAL` path,
   - FFT storage reused so separate `fft_nt` residency is reduced,
   - Igor reorder implemented with a cached `RawKernel`.
2. The authoritative optimization timing harness now lives at:
   - `tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py`
   - supporting notes at `tests/validation/dev/cupy_rsoxs_optimization/README.md`
3. That harness is now explicitly cupy-only and timing-only:
   - primary timing starts immediately before `Morphology(...)`,
   - primary timing ends immediately after synchronized `run(return_xarray=False)`,
   - upstream field generation is excluded,
   - export timing is excluded,
   - Segment `A1` is measured in the harness and Segments `A2-F` are measured
     in `cupy-rsoxs`,
   - Segment `G` remains reserved for future export timing.
4. The reusable full-energy backend-comparison harness lives at:
   - `tests/validation/dev/core_shell_backend_performance/run_core_shell_backend_performance_abstract.py`
   - supporting notes at `tests/validation/dev/core_shell_backend_performance/README.md`
5. The full-energy backend-comparison harness is now legacy/historical only for
   optimization work:
   - it is not the authoritative timing harness,
   - its old mixed workflow metric should not be used to rank current speed
     work,
   - no new `cyrsoxs` timing lanes should be added to the default optimization
     loop.
6. Current development artifacts that should be treated as the authoritative
   resumable timing record root are:
   - `test-reports/cupy-rsoxs-optimization-dev/`
   - the current harness writes fresh per-run `summary.json` files under that
     root using host/device resident-mode labels such as
     `core_shell_small_single_no_rotation_host`
   - historical snapshots such as
     `test-reports/cupy-rsoxs-optimization-dev/verify_cli_small_postcleanup/summary.json`
     remain useful for context, but they predate the resident-mode/default-lane
     refactor and should not be treated as the naming or workflow authority
7. Maintained parity and physics guardrails still live in the official test
   suite. The dev harnesses above are for optimization and comparison work, not
   for replacing the maintained parity suite.

### 18.2 Timing apparatus status (implemented March 23, 2026)

1. The timing repair requested in earlier versions of this roadmap is now
   complete for the current optimization loop.
2. The primary optimization wall metric is:
   - start immediately before `Morphology(...)` construction,
   - end immediately after synchronized `run(return_xarray=False)` completion.
3. The primary optimization wall metric excludes:
   - field generation before object creation,
   - result export such as `to_xarray()`,
   - downstream A-wedge generation, plotting, and analysis.
4. The implemented internal-only control path is:
   - `Morphology._set_private_backend_timing_segments(...)`
   - `Morphology._clear_private_backend_timing_segments()`
   - this is intentionally private and should remain internal unless a later
     API review explicitly promotes it.
5. Implemented segment coverage is:
   - Segment `A1`: harness wall time for morphology construction /
     contract normalization inside the primary timing boundary,
   - Segment `A2`: `cupy-rsoxs` private wall time for runtime staging,
   - Segments `B-F`: `cupy-rsoxs` private CUDA-event timings,
   - Segment `G`: deferred for a later export-timing pass.
6. The backend timing payload now has the shape:
   - `{"measurement": "...", "selected_segments": [...], "segment_seconds": {...}, "segment_measurements": {...}}`
7. When timing is not explicitly enabled:
   - `morphology.backend_timings` remains `{}`,
   - `cupy-rsoxs` does not enter the CUDA-event timing path,
   - production runs therefore avoid development-only synchronization overhead.
8. The old dev-harness `workflow_seconds` metric is intentionally discarded and
   should not be revived as an optimization authority.
9. Segment totals are not expected to sum exactly to `primary_seconds`:
   - Segments `A1` and `A2` are wall-clock,
   - Segments `B-F` are CUDA-event timings,
   - residual host/launch overhead remains in the primary wall metric.
10. Verification completed in `nrss-dev` for this pass:
    - smoke coverage now includes
      `test_cupy_private_segment_timing_is_opt_in_and_subsettable`,
    - direct single-energy / no-rotation CoreShell worker timing passed,
    - subset selection such as `("B", "D")` returned only the requested
      backend segments,
    - CLI timing harness run `verify_cli_small_postcleanup` passed.

### 18.3 Current evidence summary from the accepted backend state

1. The first optimization campaign retained three changes and rejected four
   first-pass implementations. See Section 18.6 for the ledger.
2. Historical full-energy `cyrsoxs` comparison reports remain useful as context
   for what was explored, but they are no longer the authoritative timing basis
   for resumed optimization work.
3. The first post-cleanup timing snapshot remains:
   - `test-reports/cupy-rsoxs-optimization-dev/verify_cli_small_postcleanup/summary.json`
4. That snapshot is still useful as historical evidence that the repaired
   timing boundary was working, but it predates the current resident-mode
   strategy and current case naming.
   - it uses older labels such as
     `core_shell_small_single_no_rotation_cupy_borrow`
   - the current harness now emits resident-mode-explicit labels such as
     `core_shell_small_single_no_rotation_host`,
     `core_shell_small_single_no_rotation_device`,
     and opt-in limited-rotation cases for either variant
5. Historical timing numbers from that old post-cleanup snapshot are still
   worth keeping as rough context for the then-accepted device-resident lane:
   - `core_shell_small_single_no_rotation_cupy_borrow`
     - `primary_seconds`: about `0.2363s`,
     - Segment `A`: about `0.0028s`,
     - Segment `B`: about `0.1319s`,
     - Segment `C`: about `0.0122s`,
     - Segment `D`: about `0.0694s`,
     - Segment `E`: about `0.0034s`,
     - Segment `F`: about `0.00047s`
   - `core_shell_small_triple_limited_rotation_cupy_borrow`
     - `primary_seconds`: about `0.4480s`,
     - Segment `B`: about `0.1936s`,
     - Segment `D`: about `0.2001s`,
     - Segment `E`: about `0.0153s`
6. Current interpretation from the repaired timing apparatus:
   - on the primary small single-energy lane, current latency is dominated by
     Segments `B` and `D`,
   - Segment `C` is visible but materially smaller,
   - Segment `A1` is currently minor in that lane,
   - host-resident `A2` can still be material and should be interpreted
     separately from constructor overhead,
   - Segments `E` and `F` are currently minor contributors in that lane.
7. Important benchmark caveats for the harness going forward:
   - the default host-resident lane is meant to resemble the most common public
     workflow and therefore includes host-resident morphology handling in the
     primary wall-clock metric,
   - in host mode, host-to-device staging happens inside `run()` and is counted
     in total wall time,
   - that staging is now isolated as Segment `A2` inside the private timing
     breakdown.
8. Additional caveat for the opt-in device-resident lane:
   - the `cupy -> cupy-rsoxs (device/borrow/strict)` path is contract-clean at
     the `Morphology` boundary,
   - but it is not a true end-to-end GPU-native morphology-generation
     benchmark,
   - the current CoreShell builder still creates fields in NumPy and then
     preconverts them to CuPy before `Morphology(...)` timing starts.
9. Latest Segment `A` evidence from the current accepted state:
   - `segA12_probe_20260323` established that Segment `A1` constructor work is
     already minor while host-resident Segment `A2` staging is the transfer
     cost center,
   - `a2_exp1_empty_set_20260323` replaced the host-resident NumPy -> CuPy
     staging fast path with `cp.empty(...); out.set(host)` and improved the
     small single-energy host lane by about `7.5%` on primary wall time and
     about `7.0%` on Segment `A2` versus the split-only baseline,
   - in the corresponding device-resident lane, Segment `A2` is effectively
     zero because the morphology fields already satisfy the backend-preferred
     device contract before `run()` begins.
10. Practical prioritization interpretation from that evidence:
   - Segment `A` is nominally complete for the common workflow,
   - if a workflow expects morphology fields to remain on GPU,
     `resident_mode='device'` is the intended faster tradeoff and should
     remain visibly faster than host-resident staging,
   - default future speed work should focus on Segments `B` and `D` unless new
     timing evidence changes the ranking.
11. Maintained parity remains a post-optimization check through the test suite
   rather than through a `cyrsoxs` timing harness.

### 18.4 Official resident-mode guidance

1. Official guidance for `cupy-rsoxs` is now to support controllable resident
   modes, with CPU / host-resident behavior as the default.
2. Host-resident staged mode is the default guidance for public workflows.
   - authoritative morphology fields remain on CPU,
   - the backend materializes temporary CuPy arrays when math requires them,
   - temporary device arrays should be deleted or replaced aggressively as the
     pipeline advances,
   - this mode is expected to lower GPU memory pressure at the cost of at least
     one host-to-device transfer.
3. Device-resident direct mode is an opt-in path for already-CuPy morphology
   fields.
   - authoritative morphology fields already satisfy the backend-preferred
     CuPy contract,
   - the backend may borrow and use those arrays directly,
   - this mode is expected to use more GPU memory but may help GPU-native
     workflows or later on-device chaining.
4. Resident mode must be documented separately from `input_policy` and
   `ownership_policy`.
   - `input_policy='strict'` means no coercion is allowed under the chosen
     contract,
   - `input_policy='coerce'` means coercion is allowed under the chosen
     contract,
   - `ownership_policy='borrow'` means incoming material objects are not copied,
   - none of those concepts alone imply end-to-end zero-copy execution.
5. The implemented public API name for resident-mode control is
   `resident_mode`.
   - `resident_mode='host'` keeps authoritative morphology fields on CPU,
   - `resident_mode='device'` keeps authoritative morphology fields on GPU
     when the backend supports that mode.

### 18.5 Optimization campaign strategy going forward

1. The optimization goal is to make `cupy-rsoxs` materially faster while
   preserving physics parity through the maintained test suite.
2. The default inner-loop timing command is now:
   - `/home/deand/mambaforge/envs/nrss-dev/bin/python tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py --label <label> --size-labels small --resident-modes host --timing-segments all`
3. Future optimization campaigns should focus on one segment at a time using
   the segmentation defined in Section 18.2 rather than trying to optimize the
   whole backend at once.
4. The default optimization benchmark ladder should use single-energy cases
   rather than full-energy runs.
   - energy iteration mostly wraps the same loop structure,
   - for non-parity tuning, full-energy runs mainly add runtime and statistics
     rather than exposing a fundamentally different algorithmic path.
5. Recommended default tuning ladder:
   - primary latency lane: `small`, single energy, `EAngleRotation=[0, 0, 0]`,
     `resident_mode='host'`, with morphology fields generated directly as the
     host-resident authoritative NumPy contract before timing starts,
   - secondary latency lane: `medium`, single energy, `EAngleRotation=[0, 0, 0]`,
     `resident_mode='host'`, with the same direct NumPy contract generation,
   - device-regression lane: selected `small` or `medium`, single energy,
     `EAngleRotation=[0, 0, 0]`, `resident_mode='device'`, with morphology
     fields preconverted to the CuPy contract and the default stream
     synchronized before the timer starts,
   - rotation-focused lane: use opt-in `--include-triple-limited` on either or
     both resident-mode variants when the built-in
     `EAngleRotation=[0, 15, 165]` checkpoint is sufficient, or use
     `--rotation-specs start:increment:end[, ...]` when the exact rotation set
     is the point of the measurement,
   - energy-focused lane: use `--no-rotation-energy-counts` for centered
     contiguous CoreShell subsets, or `--energy-lists 'E1|E2|...'[, ...]` when
     an explicit energy list is the point of the measurement,
   - memory / throughput guardrail lane: `large`, single energy, selected
     resident mode, dense angle sweep when needed.
6. When chasing a hotspot, narrow `--timing-segments` to the relevant subset
   instead of recording everything on every run.
   - Segment `A` is nominally complete for the common workflow:
     - `A1` constructor overhead is already minor,
     - the accepted host-resident `A2` staging improvement is in place,
     - the device-resident lane is an intentionally faster alternate use case
       and should remain visibly faster when morphology is already on device,
   - current evidence makes Segments `B` and `D` the default focal points,
   - re-expand to `all` when checking for regression spillover into neighboring
     stages.
7. Do not add `cyrsoxs` timing lanes to the default optimization harness.
   `cyrsoxs` timing is not part of the current optimization loop.
8. Use `--include-triple-limited` only when rotation-sensitive behavior needs a
   checkpoint outside the default no-rotation loop.
9. Use `--include-full-small-check` only as an occasional expensive checkpoint
   for the full rotation path.
10. If a single-energy lane is too fast or too noisy to rank consistently,
   repeat it enough times to stabilize the comparison rather than escalating
   directly to full-energy runs.
11. Full-energy runs should be reserved for:
    - milestone confirmation after accepted optimization changes,
    - final comparison graphics if they are later needed,
    - and maintained parity / correctness rechecks.
12. Future optimization guidance should be treated as open-ended. Many more
   opportunities likely exist beyond the ones already enumerated or tried. The
   document should not be read as an exhaustive list of remaining ideas.

### 18.5.1 Current Segment B/D campaign plan (March 24, 2026)

This subsection is the live plan and outcome ledger for the current Segment `B`
/ `D` optimization block. Update it as each step completes so a fresh context
can recover both the intended order and the measured outcomes.

Status key:
- `planned`: not started yet,
- `in_progress`: active implementation or measurement work,
- `completed`: implemented and measured,
- `rejected`: attempted and not kept.

Current campaign steps:

1. `plan01_document_campaign` - `completed`
   - write the full Segment `B` / `D` plan into this roadmap and the dev timing
     README before code changes,
   - define the execution-path terminology and the benchmark matrix to be used
     for the campaign,
   - outcome:
     - completed on March 24, 2026 by adding this live campaign subsection and
       the matching dev-harness README notes.
2. `plan02_execution_path_surface` - `completed`
   - revive `backend_options` as the backend-specific runtime-behavior surface
     for `cupy-rsoxs`,
   - add an explicit `execution_path` option with default
     `execution_path='tensor_coeff'`,
   - initial intended values:
     - `tensor_coeff`: current accepted `Nt -> FFT(Nt) -> projection-coefficient`
       route,
     - `direct_polarization`: CyRSoXS `AlgorithmType=0`
       communication-minimizing analog,
     - `nt_polarization`: CyRSoXS `AlgorithmType=1`
       memory-minimizing analog,
   - outcome:
     - completed on March 24, 2026,
     - `cupy-rsoxs` now accepts `backend_options["execution_path"]` with
       supported values `tensor_coeff`, `direct_polarization`, and
       `nt_polarization`,
     - aliases now normalize as:
       - `default -> tensor_coeff`,
       - `tensor -> tensor_coeff`,
       - `direct -> direct_polarization`,
       - `nt -> nt_polarization`,
     - default behavior remains unchanged at `execution_path='tensor_coeff'`.
3. `plan03_core_shell_backend_options_plumbing` - `completed`
   - thread `backend_options` through the maintained CoreShell construction and
     backend-run helpers so execution-path validation can use the official
     maintained morphology path with reasonable defaults,
   - outcome:
     - completed on March 24, 2026,
     - maintained CoreShell helpers now accept `backend_options` for both
       morphology construction and backend execution so dormant-path validation
       can use the official maintained morphology lane rather than ad hoc
       builders.
4. `plan04_harness_execution_path_matrix` - `completed`
   - extend the authoritative subprocess timing harness to accept
     execution-path-specific sweeps,
   - ensure labels and summaries include both primary timing and
     segment-by-segment timing per execution path,
   - keep segment tracking available for all execution paths using the existing
     `A1,A2,B,C,D,E,F` segmentation,
   - outcome:
     - completed on March 24, 2026,
     - the authoritative subprocess harness now accepts
       `--execution-paths ...`,
     - the same harness now also accepts explicit
       `--rotation-specs start:increment:end[, ...]` and
       `--energy-lists 'E1|E2|...'[, ...]` inputs for targeted
       rotation-sensitive or energy-sensitive studies,
     - benchmark labels now carry the execution-path suffix,
     - requested and resolved backend options are persisted into the summary
       artifact for each case,
     - segment tracking remains available across all surfaced execution paths,
       even though the exact work contained inside Segments `B/C/D/E` differs
       by execution path,
     - when both explicit rotation and explicit energy lists are supplied, the
       harness emits combined cases as well as the rotation-only and
       energy-only variants,
     - focused verification artifact:
       `test-reports/cupy-rsoxs-optimization-dev/harness_rotation_energy_smoke_20260324/summary.json`
       confirmed baseline, rotation-only, energy-only, and combined case
       emission on both host and device resident modes.
5. `plan05_execution_path_baselines` - `in_progress`
   - benchmark the current `tensor_coeff` path plus the two dormant paths on
     the official no-rotation host/device small lanes before new math changes,
   - run CoreShell correctness validation for the newly surfaced dormant paths
     before treating their timings as optimization guidance,
   - current measured timing baseline from
     `execution_path_surface_smoke_20260324`:
     - host / `tensor_coeff`:
       `primary 2.833s`, `A1 0.003`, `A2 2.482`, `B 0.139`, `C 0.010`,
       `D 0.072`, `E 0.112`, `F 0.001`,
     - host / `direct_polarization`:
       `primary 2.609s`, `A1 0.003`, `A2 2.401`, `B 0.143`, `C 0.009`,
       `D 0.033`, `E 0.002`, `F 0.001`,
     - host / `nt_polarization`:
       `primary 2.607s`, `A1 0.003`, `A2 2.408`, `B 0.130`, `C 0.016`,
       `D 0.034`, `E 0.002`, `F 0.000`,
     - device / `tensor_coeff`:
       `primary 0.237s`, `A1 0.003`, `A2 0.000`, `B 0.131`, `C 0.010`,
       `D 0.072`, `E 0.004`, `F 0.000`,
     - device / `direct_polarization`:
       `primary 0.204s`, `A1 0.003`, `A2 0.000`, `B 0.138`, `C 0.009`,
       `D 0.033`, `E 0.002`, `F 0.000`,
     - device / `nt_polarization`:
       `primary 0.205s`, `A1 0.003`, `A2 0.000`, `B 0.134`, `C 0.015`,
       `D 0.033`, `E 0.002`, `F 0.000`,
   - current interpretation:
     - both surfaced dormant routes are materially faster than
       `tensor_coeff` on the small single-energy no-rotation lane,
     - their speed advantage is driven mostly by much smaller Segment `D` and
       Segment `E` work rather than by a clear Segment `B` win,
     - however, a later limited multi-angle comparison on the same small
       CoreShell model showed the ranking reverses once angle iteration
       matters:
       - source artifact:
         `test-reports/cupy-rsoxs-optimization-dev/execution_path_multiangle_5_vs_15_20260324/summary.json`,
       - host / `EAngleRotation=[0, 15, 165]`:
         `tensor_coeff 2.742s`, `nt_polarization 3.319s`,
         `direct_polarization 3.742s`,
       - host / `EAngleRotation=[0, 5, 165]`:
         `tensor_coeff 2.921s`, `nt_polarization 4.376s`,
         `direct_polarization 4.524s`,
       - device / `EAngleRotation=[0, 15, 165]`:
         `tensor_coeff 0.419s`, `nt_polarization 0.441s`,
         `direct_polarization 0.556s`,
       - device / `EAngleRotation=[0, 5, 165]`:
         `tensor_coeff 0.412s`, `nt_polarization 0.926s`,
         `direct_polarization 1.559s`,
       - interpretation:
         for angle-heavy workloads, `tensor_coeff` is the practical winner and
         should remain the primary optimization target even though the dormant
         routes are still useful no-rotation reference points,
     - quick raw CoreShell comparisons against `tensor_coeff` on the official
       maintained morphology path show the dormant routes are numerically very
       close overall but not bitwise-identical:
        `max_abs 0.078125`, `rmse 1.60801e-4`, `p95_abs 3.8147e-06`,
      - after the accepted plan06/plan07 steps, the same official maintained
        morphology raw comparison remained close:
        `max_abs 0.046875`, `rmse 9.72605e-05`, `p95_abs 3.8147e-06`,
      - the maintained official sim-regression test for the accepted default
        `cupy-rsoxs` path passed on March 24, 2026:
        `pytest tests/validation/test_core_shell_reference.py -k "sim_regression_cupy_borrow_strict" --nrss-backend cupy-rsoxs -v`,
      - full dormant-path A-wedge / sim-reference validation was attempted on
        the official CoreShell helper path but remained too expensive for the
        current inner-loop campaign, so dormant-path timing guidance should
        still be treated as timing evidence plus raw-maintained-morphology
        similarity rather than as fully closed reference validation.
6. `plan06_isotropic_material_fast_path` - `completed`
   - add a full-material isotropic fast path for materials with
     `S == 0` everywhere,
   - skip Euler decode and off-diagonal tensor work for those materials across
     all execution paths,
   - expected to help common vacuum / matrix / isotropic-additive cases,
   - outcome from `plan06_isotropic_material_fast_path_clean_20260324`:
     - completed on March 24, 2026,
     - exact-zero isotropic materials now:
       - skip runtime staging of `S`, `theta`, and `psi` in the host-resident
         path,
       - skip Euler decode and off-diagonal tensor work in Segment `B` across
         all execution paths,
     - focused smoke checks added:
       - host-resident isotropic staging now confirms only `Vfrac` is staged to
         CuPy for fully isotropic materials,
       - all three execution paths now agree on a fully isotropic synthetic
         morphology in the maintained smoke lane,
     - measured timing deltas versus the execution-path baseline on the
       official small single-energy no-rotation CoreShell lane:
       - host / `tensor_coeff`:
         `primary 2.833s -> 2.519s`, `A2 2.482 -> 2.305`, `B 0.139 -> 0.110`,
       - host / `direct_polarization`:
         `primary 2.609s -> 2.613s`, `B 0.143 -> 0.112`,
       - host / `nt_polarization`:
         `primary 2.607s -> 2.706s`, `B 0.130 -> 0.113`,
       - device / `tensor_coeff`:
         `primary 0.237s -> 0.231s`, `A2 0.000 -> 0.112`, `B 0.131 -> 0.014`,
       - device / `direct_polarization`:
         `primary 0.204s -> 0.210s`, `A2 0.000 -> 0.109`, `B 0.138 -> 0.015`,
       - device / `nt_polarization`:
         `primary 0.205s -> 0.201s`, `A2 0.000 -> 0.116`, `B 0.134 -> 0.014`,
     - interpretation:
       - the CoreShell lane confirms the intended Segment `B` win strongly,
         especially in the device-resident path where two of the three
         materials are fully isotropic,
       - total latency impact is mixed because exact-zero isotropic detection is
         currently performed during runtime staging for device-resident inputs,
         which moves about `0.11s` into Segment `A2` on this small benchmark,
       - the host-default `tensor_coeff` lane still improves materially on the
         primary wall metric and remains worth keeping,
       - the current implementation is therefore accepted as the baseline for
         the next step, with the device-path `A2` caveat recorded rather than
         treated as a blocker.
7. `plan07_axis_family_fast_path` - `completed`
   - add the high-value axis-family special cases for electric-field rotations
     congruent to `0°/180°` and `90°/270°`,
   - first target the single-angle / fully axis-aligned cases where the
     savings reach beyond Segment `E`,
   - note the analogous downstream pruning opportunity in Segment `D`:
     axis-aligned cases can avoid `proj_xy` and one projection-family branch,
   - outcome from `plan07_axis_family_fast_path_clean_20260324`:
     - completed on March 24, 2026,
     - fully axis-aligned angle sets now use an explicit angle-family plan:
       - `0°/180°` use the x-family subset,
       - `90°/270°` use the y-family subset,
       - general angles continue to use the existing full path,
     - implementation details:
       - `tensor_coeff` now skips `proj_xy` and the unused x/y projection
         family on fully axis-aligned angle sets,
       - `nt_polarization` now computes only the `Nt` component subset needed
         by the aligned angle family,
       - `direct_polarization` now uses the aligned-field specialization rather
         than the general `mx*sx + my*sy` branch,
       - identity rotations now bypass the affine-transform path in Segment
         `E`,
     - focused smoke/regression checks remained green, including the existing
       endpoint-semantics smoke and the execution-path smoke subset,
     - measured deltas versus the post-plan06 baseline on the official small
       single-energy no-rotation CoreShell lane:
       - host / `tensor_coeff`:
         `D 0.073 -> 0.035`, `E 0.004 -> 0.002`,
       - host / `direct_polarization`:
         `primary 2.613s -> 2.530s`, `E 0.002 -> 0.000`,
       - host / `nt_polarization`:
         `primary 2.706s -> 2.537s`, `D 0.040 -> 0.033`,
         `E 0.002 -> 0.000`,
       - device / `tensor_coeff`:
         `primary 0.231s -> 0.196s`, `D 0.072 -> 0.035`,
         `E 0.004 -> 0.002`,
       - device / `direct_polarization`:
         `primary 0.210s -> 0.198s`, `D 0.051 -> 0.033`,
         `E 0.002 -> 0.000`,
       - device / `nt_polarization`:
         `primary 0.201s -> 0.193s`, `E 0.002 -> 0.000`,
     - interpretation:
       - the aligned-angle specialization is a clear `D`/`E` win on the
         default `0°` lane and produces the intended overall device-lane speedup,
       - host-lane `A2` variance still makes wall-clock interpretation noisier
         than the backend-segment metrics, so the step should be read primarily
         as a segment-specific improvement rather than as a universal primary
         win on every surfaced path,
       - an isolated `90°` spot check showed the same low-`D` / near-zero-`E`
         shape for the y-family branch, so the optimization is not only
         exercising the `0°` case.
8. `plan08_segment_b_algebraic_rewrite` - `rejected`
   - simplify Segment `B` tensor assembly with shared-term factoring, smaller
     scratch reuse, and more aggressive dead-intermediate deletion,
   - this is explicitly not the abandoned multi-energy cache idea:
     it should reduce temporary live-set pressure rather than persist large
     per-energy/per-material tensors,
   - outcome from `plan08_segment_b_algebraic_rewrite_20260324`:
     - attempted on March 24, 2026 with a fixed-size scratch-buffer
       formulation for Segment `B`,
     - this implementation did *not* recreate the abandoned cache-memory risk:
       it used only per-call scratch buffers rather than any persistent
       per-energy/per-material cache,
     - however it materially regressed the default `tensor_coeff` Segment `B`
       lane and therefore did not clear the acceptance bar,
     - most important measured regression versus the post-plan07 baseline:
       - host / `tensor_coeff`:
         `B 0.107 -> 0.192` and `primary 2.690s -> 2.619s` still failed the
         Segment `B` objective because the targeted segment became
         substantially slower,
     - other surfaced paths were mixed to nearly flat rather than strongly
       improved,
     - the implementation was reverted and the accepted baseline therefore
       remains the post-plan07 axis-family state.
9. `plan09_rebenchmark_and_regression_check` - `completed`
   - rerun the official subprocess timing matrix after each accepted step,
   - use the maintained smoke/reference checks as regression guards before the
     campaign is closed out,
   - outcome from `plan09_final_rebenchmark_accepted_state_20260324`:
     - completed on March 24, 2026,
     - final accepted state is the combination of:
       - execution-path surfacing,
       - full-material isotropic fast path,
       - axis-family fast path,
       - with the plan08 scratch experiment and the plan11
         `ElementwiseKernel` experiment both reverted,
     - final accepted-state timing snapshot on the official small
       single-energy no-rotation CoreShell lane:
       - host / `tensor_coeff`:
         `primary 2.515s`, `A2 2.339`, `B 0.109`, `D 0.035`, `E 0.002`,
       - host / `direct_polarization`:
         `primary 2.541s`, `A2 2.364`, `B 0.113`, `D 0.034`, `E 0.000`,
       - host / `nt_polarization`:
         `primary 2.519s`, `A2 2.344`, `B 0.108`, `D 0.033`, `E 0.000`,
       - device / `tensor_coeff`:
         `primary 0.205s`, `A2 0.122`, `B 0.010`, `D 0.039`, `E 0.002`,
       - device / `direct_polarization`:
         `primary 0.198s`, `A2 0.116`, `B 0.014`, `D 0.034`, `E 0.000`,
       - device / `nt_polarization`:
         `primary 0.207s`, `A2 0.115`, `B 0.010`, `D 0.035`, `E 0.001`,
     - maintained regression checks completed in this close-out pass:
       - focused smoke subset: `9 passed`,
       - accepted-path maintained CoreShell sim regression:
         `1 passed` with `--nrss-backend cupy-rsoxs`,
     - final interpretation:
       - the accepted state materially improves the default host
         `tensor_coeff` lane versus the original execution-path baseline,
         mostly through lower `A2`, `B`, `D`, and `E`,
       - the device lane keeps the strong Segment `B` and aligned-angle
         `D/E` wins, with the known caveat that isotropic detection now lands
         in `A2`.
10. `plan10_float16_followup_scaffold` - `completed`
    - do not implement the mixed-precision campaign in this block,
    - only leave the execution-path surface and roadmap notes in a form that
      allows an orthogonal future backend-options extension for reduced
      storage/transfer precision,
    - outcome:
      - completed on March 24, 2026 with no float16 compute-path work landed,
      - `backend_options["execution_path"]` and the accompanying roadmap notes
        now provide the intended light scaffold for a future orthogonal reduced
        precision option surface,
      - the existing float16-rejection smoke continues to pass in this state.
11. `plan11_elementwise_kernel_experiment` - `rejected`
    - try one last `ElementwiseKernel` implementation aimed at Segment `B`,
    - scope it narrowly to the aligned-angle/default-lane path first rather
      than broadening it across every route immediately,
    - accept it only if the default-lane gain is large enough to justify the
      added maintenance burden; otherwise reject and revert it,
    - outcome from `plan11_elementwise_kernel_experiment_20260324`:
      - attempted on March 24, 2026 by replacing the aligned-angle `Nt`
        subset path with a narrow `ElementwiseKernel` implementation,
      - this did reduce device-resident aligned-angle `B` time for the
        `tensor_coeff` / `nt_polarization` subset:
        - device / `tensor_coeff`: `B 0.010 -> 0.005`,
        - device / `nt_polarization`: `B 0.010 -> 0.005`,
      - but it materially regressed the default host-resident
        `tensor_coeff` path:
        - host / `tensor_coeff`: `B 0.107 -> 0.246`,
      - because the default host lane remains the primary public-workflow
        authority and the gain was not broad enough to offset the extra kernel
        maintenance burden, the experiment was rejected and reverted.

Precision and option-surface notes for this campaign:

1. `backend_options` is being resurrected as the explicit backend-specific
   runtime-behavior surface rather than overloading `AlgorithmType` directly in
   `cupy-rsoxs`.
2. The float16 plan is orthogonal to `execution_path`.
   - intended future direction:
     - reduced storage / host->device transfer precision as the first target,
     - optional Segment `B` low-precision experiment if validation permits,
     - cast to `float32` / `complex64` before FFT ingress for parity-sensitive
       math.
3. No float16 compute-path implementation should be accepted in this campaign
   beyond the light option-surface scaffolding needed so a later block can add
   it cleanly.
4. Segment `D` should keep a note for future pruning and scratch-reuse work:
   - axis-family cases may avoid building `proj_xy` and one of the x/y
     projection families,
   - algebraic factoring and scratch reuse around basis/projection assembly may
     be worthwhile there after the Segment `B` pass is measured.

### 18.6 Campaign ledger from work already done

1. Current accepted optimization state:
   - `opt01_drop_nt5`
     - removed dead `Nt[5]` from the supported Euler-only / `PARTIAL` path,
     - improved run time by about `6.8%` to `32.3%` versus the first matrix
       baseline,
     - improved large limited-rotation free memory after run from about
       `0.66 GiB` to `4.04 GiB`.
   - `opt02_fft_reuse`
     - reused `nt` storage for FFT output and removed the separate `fft_nt`
       live set,
     - improved run time by an additional `0.8%` to `6.7%` versus `opt01`,
     - improved large limited-rotation free memory after run from about
       `4.04 GiB` to `6.99 GiB`,
     - reduced large limited-rotation pool total from about `43.24 GiB` to
       `40.29 GiB`.
   - `opt04_igor_kernel`
     - replaced the advanced-indexing Igor reorder with a cached `RawKernel`,
     - improved run time by an additional `1.1%` to `5.6%` versus `opt03`,
     - improved run time by about `12.8%` to `36.4%` versus the original matrix
       baseline,
     - preserved the ad hoc validation metrics already seen in the baseline
       run.
   - `opt08_segment_a_split`
     - split historical Segment `A` into Segment `A1` constructor timing and
       Segment `A2` runtime staging timing,
     - made host-resident transfer cost separately measurable from
       `Morphology(...)` construction,
     - changed the accepted measurement basis for resumed work so Segment `A`
       no longer hides transfer and constructor behavior inside one number.
   - `opt09_stage_empty_set`
     - replaced host-resident NumPy -> CuPy staging from `cp.asarray(...)`
       with `cp.empty(...); out.set(host)`,
     - improved the small single-energy host lane by about `7.5%` on primary
       wall time and about `7.0%` on Segment `A2` versus the split-only
       baseline,
     - device-resident Segment `A2` remains effectively zero when morphology
       fields are already on device before `run()` begins,
     - this makes Segment `A` nominally complete for the common workflow, with
       future default tuning focus shifted to Segments `B` and `D`.
2. Explored but not accepted in the first campaign:
   - `opt03_prealloc_result_scratch`
     - mixed result: about `-2.3%` to `+0.4%` versus `opt02`,
     - no meaningful memory-headroom improvement in the measured matrix,
     - rejected and reverted.
   - `opt05_proj_coeff_fuse`
     - preserved validation, but regressed run time by about `5.7%` to `38.6%`
       versus `opt04`,
     - rejected and reverted.
   - `opt06_stream_projection`
     - preserved validation, but regressed run time heavily versus the accepted
       state and did not produce a practical memory-headroom win in this
       matrix,
     - rejected and reverted.
   - `opt07_texture_affine`
     - implemented a `cupyx.scipy.ndimage.affine_transform(...,
       texture_memory=True)` candidate using a homogeneous transform in texture
       mode,
     - full-matrix result was effectively flat to slightly worse:
       about `+0.07%` to `+1.9%` versus `opt04`,
     - large-case memory footprint was unchanged relative to `opt04`,
     - ad hoc validation remained within the same tolerance band,
     - rejected and reverted because it did not clear the significant-gain
       bar.
3. Interpretation rule for the ledger:
   - an accepted item has demonstrated a material win under measured conditions
     and remains part of the current baseline,
   - a rejected item means that specific implementation did not clear the bar,
     not that the whole idea class is permanently closed.

### 18.7 Explicit experiments and deferred directions

1. `float16` and mixed-precision work remain deferred until after the next
   speed campaigns; parity-sensitive compute remains `float32/complex64`.
   - near-term intended shape of that work:
     - `backend_options` should carry reduced-precision storage/runtime flags
       orthogonally to `execution_path`,
     - the first target is host/device storage and transfer reduction rather
       than end-to-end low-precision FFT/projection math,
     - the expected precision ladder is reduced-precision storage or Segment
       `B` staging followed by promotion to `float32` / `complex64` before FFT
       ingress.
2. Reduced angle sampling, alternate interpolation rules, and multi-GPU fan-out
   remain outside the current exact-tuning track.
3. Host-resident staged mode creates room for explicit experiments with deeper
   CPU-side precompute, including testing whether some early field math is
   faster on CPU before transfer.
4. Deeper CPU-side precompute should be treated as an explicit experiment, not
   as the default plan.
   - it may help some host-resident workflows,
   - but it may also become transfer-dominated if large per-energy
     intermediates are moved to GPU,
   - therefore it must be measured before being promoted into recommended
     architecture.
5. Export timing / host-conversion timing remains intentionally deferred as
   Segment `G`.
6. Per-stage memory instrumentation remains deferred; the existing coarse peak
   memory monitoring is enough for the current pass.
7. Host-resident repeated-run staging reuse remains explicitly deferred as a
   low-priority niche possibility.
   - examples include registered-host buffers, reusable staged device mirrors,
     or other host-lane transfer caches,
   - this repo does not expect many repeated-run workflows to justify making
     that the default direction,
   - if a workflow benefits from persistent GPU morphology residency, prefer
     `resident_mode='device'` rather than adding host-resident caching by
     default.
8. Resume rule for a fresh optimization context:
   - start from the current accepted backend state in this repo, including the
     Segment `A1/A2` split and the accepted host-resident `A2` staging fast
     path,
   - use `tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py`
     as the default inner-loop harness,
   - begin with the primary lane:
     `/home/deand/mambaforge/envs/nrss-dev/bin/python tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py --label <label> --size-labels small --resident-modes host --timing-segments all`,
   - narrow `--timing-segments` when focusing on a specific segment,
   - use the single-energy benchmark ladder for inner-loop tuning,
   - for explicit rotation-sensitive or energy-sensitive studies, the harness
     now supports `--rotation-specs` and `--energy-lists`, and it emits
     combined cases when both are supplied,
   - recheck `--resident-modes device` periodically as a regression lane for
     direct-CuPy workflows,
   - add `--include-triple-limited` only when the fixed
     `EAngleRotation=[0, 15, 165]` checkpoint is sufficient for the question,
   - otherwise use `--rotation-specs` and `--energy-lists` to benchmark the
     exact angle or energy sets under discussion,
   - rerun maintained parity checks after promising optimization changes,
   - treat the legacy full-energy backend-comparison harness as optional
     historical context rather than a required step in the default loop.
