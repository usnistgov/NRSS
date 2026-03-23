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
   - Segment `A` in the harness with wall-clock timing,
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
   - Segment A: `Morphology` construction, contract normalization, and data staging,
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
   - Segment `A` is measured in the harness and Segments `B-F` are measured in
     `cupy-rsoxs`,
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
   resumable timing record are:
   - `test-reports/cupy-rsoxs-optimization-dev/`
   - especially `test-reports/cupy-rsoxs-optimization-dev/verify_cli_small_postcleanup/summary.json`
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
   - Segment `A`: harness wall time for morphology construction / staging
     inside the primary timing boundary,
   - Segments `B-F`: `cupy-rsoxs` private CUDA-event timings,
   - Segment `G`: deferred for a later export-timing pass.
6. The backend timing payload now has the shape:
   - `{"measurement": "cuda_event", "selected_segments": [...], "segment_seconds": {...}}`
7. When timing is not explicitly enabled:
   - `morphology.backend_timings` remains `{}`,
   - `cupy-rsoxs` does not enter the CUDA-event timing path,
   - production runs therefore avoid development-only synchronization overhead.
8. The old dev-harness `workflow_seconds` metric is intentionally discarded and
   should not be revived as an optimization authority.
9. Segment totals are not expected to sum exactly to `primary_seconds`:
   - Segment `A` is wall-clock,
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
3. The first authoritative post-cleanup timing snapshot is:
   - `test-reports/cupy-rsoxs-optimization-dev/verify_cli_small_postcleanup/summary.json`
4. That snapshot confirms the repaired timing boundary and current segment
   profile on the accepted backend state:
   - `core_shell_small_single_no_rotation_cupy_borrow`
     - `primary_seconds`: about `0.2363s`,
     - Segment `A`: about `0.0028s`,
     - Segment `B`: about `0.1319s`,
     - Segment `C`: about `0.0122s`,
     - Segment `D`: about `0.0694s`,
     - Segment `E`: about `0.0034s`,
     - Segment `F`: about `0.00047s`.
5. The same snapshot also provides a post-cleanup limited-rotation checkpoint:
   - `core_shell_small_triple_limited_rotation_cupy_borrow`
     - `primary_seconds`: about `0.4480s`,
     - Segment `B`: about `0.1936s`,
     - Segment `D`: about `0.2001s`,
     - Segment `E`: about `0.0153s`.
6. Current interpretation from the repaired timing apparatus:
   - on the primary small single-energy lane, current latency is dominated by
     Segments `B` and `D`,
   - Segment `C` is visible but materially smaller,
   - Segments `A`, `E`, and `F` are currently minor contributors in that lane.
7. Important benchmark caveat:
   - the current `cupy -> cupy-rsoxs (borrow/strict)` path is contract-clean at
     the `Morphology` boundary,
   - but it is not a true end-to-end GPU-native morphology-generation
     benchmark,
   - the current CoreShell builder still creates fields in NumPy and then
     preconverts them to CuPy before `Morphology(...)` timing starts.
8. Consequence of that caveat:
   - current timings isolate backend speed, not morphology generation speed,
   - this is intentional for the present optimization campaign.
9. Maintained parity remains a post-optimization check through the test suite
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
5. Public API naming for resident-mode control does not need to be finalized in
   this document, but the conceptual split and the CPU-default policy are now
   considered official guidance.

### 18.5 Optimization campaign strategy going forward

1. The optimization goal is to make `cupy-rsoxs` materially faster while
   preserving physics parity through the maintained test suite.
2. The default inner-loop timing command is now:
   - `/home/deand/mambaforge/envs/nrss-dev/bin/python tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py --label <label> --size-labels small --timing-segments all`
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
   - secondary latency lane: `medium`, single energy, `EAngleRotation=[0, 0, 0]`,
   - rotation-focused lane: `medium`, single energy, limited or dense angle
     sweep to isolate rotation/accumulation cost,
   - memory / throughput guardrail lane: `large`, single energy, dense angle
     sweep when needed.
6. When chasing a hotspot, narrow `--timing-segments` to the relevant subset
   instead of recording everything on every run.
   - current evidence makes Segments `B` and `D` the first sensible focal
     points,
   - re-expand to `all` when checking for regression spillover into neighboring
     stages.
7. Do not add `cyrsoxs` timing lanes to the default optimization harness.
   `cyrsoxs` timing is not part of the current optimization loop.
8. Use `--include-full-small-check` only as an occasional expensive checkpoint
   for the full rotation path.
9. If a single-energy lane is too fast or too noisy to rank consistently,
   repeat it enough times to stabilize the comparison rather than escalating
   directly to full-energy runs.
10. Full-energy runs should be reserved for:
    - milestone confirmation after accepted optimization changes,
    - final comparison graphics if they are later needed,
    - and maintained parity / correctness rechecks.
11. Future optimization guidance should be treated as open-ended. Many more
   opportunities likely exist beyond the ones already enumerated or tried. The
   document should not be read as an exhaustive list of remaining ideas.

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
       run,
     - this is the current accepted code state for resumed optimization work.
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
7. Resume rule for a fresh optimization context:
   - start from the current accepted `opt04`-equivalent behavior plus the
     repaired timing apparatus in this repo state,
   - use `tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py`
     as the default inner-loop harness,
   - begin with the primary lane:
     `/home/deand/mambaforge/envs/nrss-dev/bin/python tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py --label <label> --size-labels small --timing-segments all`,
   - narrow `--timing-segments` when focusing on a specific segment,
   - use the single-energy benchmark ladder for inner-loop tuning,
   - rerun maintained parity checks after promising optimization changes,
   - treat the legacy full-energy backend-comparison harness as optional
     historical context rather than a required step in the default loop.
