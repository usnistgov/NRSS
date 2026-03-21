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

TensorFlow is deprioritized for this project.

## 8. Test-First Program (Highest Priority)

### 8.1 Immediate objective

Convert `tests/validation/` legacy scripts into robust pytest suites using pybind CyRSoXS execution where applicable (no CLI serialization bottleneck), while establishing a first stable physics-validation lane before backend refactors.

### 8.2 Canonical initial cases

1. Analytical sphere form factor.
2. Sphere contrast scaling.
3. Core-shell.
4. Circle lattice.

### 8.3 Required test qualities

1. Deterministic fixtures and fixed RNG seeds if randomness appears anywhere.
2. Explicit metadata capture (versions, geometry, dtype, parameter hashes, backend flags).
3. Machine-readable golden references generated from trusted pybind runs.

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
3. Added pytest marker declarations in `pyproject.toml` for `smoke`, `cpu`, `gpu`, `slow`, `physics_validation`, `toolchain_validation`, and `phase0`.
4. Added `tests/conftest.py` to default tests to a single visible GPU when the environment is otherwise unset, improving reproducibility and avoiding known CyRSoXS multi-GPU instability during energy fan-out.

Latest run evidence:

1. Command: `bash scripts/run_local_test_report.sh --stop-on-fail`
2. Timestamp (UTC): `20260320T134227Z`
3. Result: `4/4` steps passed
4. CPU smoke: `12 passed, 10 deselected`
5. GPU smoke: `10 passed, 12 deselected`
6. Physics validation: `6 passed, 2 deselected`

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
3. Added `tests/validation/test_analytical_2d_disk_form_factor.py`:
   - direct analytical 2D disk comparison through the pybind-to-PyHyper workflow,
   - `1 x 2048 x 2048`, `PhysSize = 1.0 nm`, diameters `70 nm` and `128 nm`,
   - pointwise and minima-alignment metrics with fixed empirical thresholds,
   - explicit disk-versus-vacuum morphology,
   - fixed `sr=1` only, mirroring the sphere test’s assertion anchor while avoiding extra 2D-path variability,
   - optional plot writing gated by `NRSS_WRITE_VALIDATION_PLOTS=1`.
4. Added `tests/validation/test_2d_disk_contrast_scaling.py`:
   - one-morph, multi-energy contrast-scaling validation for the 2D pathway,
   - `1 x 2048 x 2048`, `PhysSize = 1.0 nm`,
   - 24 close-energy scenarios covering beta-only, delta-only, mixed, and split-material families,
   - integrated-intensity checks over a fixed q window with fixed empirical thresholds.
5. Added `tests/validation/lib/bragg.py`:
   - shared deterministic lattice builders and reciprocal-space prediction helpers for Bragg validation,
   - supports square/hexagonal 2D disk lattices and simple-cubic/HCP 3D sphere lattices,
   - keeps explicit vacuum as the second material and uses float-center local stamping for morphology construction.
6. Added `tests/validation/test_bragg_2d_lattice.py`:
   - deterministic square (`a = 30 nm`) and hexagonal (`a = 45 nm`) disk lattices at `1 x 2048 x 2048`, `PhysSize = 1.0 nm`,
   - validates detector-peak locations and quasi-powder shell locations through the pybind-to-PyHyper workflow,
   - includes verbose diagnostic plots with full predicted-shell overlays.
7. Added `tests/validation/test_bragg_3d_lattice.py`:
   - deterministic simple-cubic (`a = 30 nm`) and ideal HCP (`a = 45 nm`) sphere lattices at `256 x 1024 x 1024`, `PhysSize = 1.0 nm`,
   - validates detector-visible 3D Bragg peak locations plus azimuthally averaged shell locations,
   - uses explicit flat-detector geometry handling for shell prediction and includes verbose diagnostic plots with visibility-class overlays.
8. Archived one-off exploratory validation code under `scripts/validation_diagnostics/` so it remains available for future archaeology without polluting pytest collection.
9. Extended `scripts/run_local_test_report.sh` to include the marker-based `physics_validation` lane in the standard local report, while also supporting `--skip-defaults` plus repeated explicit `--cmd` runs for targeted validation and stochastic-failure checks. Newly added Bragg pytest modules are therefore included automatically.
10. Targeted local validation against an injected fixed CyRSoXS pybind build removed the prior same-process 2D analytical disk stochastic failure in local testing:
   - one-process back-to-back `70 nm` then `128 nm` analytical 2D disk validation passed `20/20` repeated runs on a single visible GPU,
   - the shipped pytest module also passed cleanly against the injected build,
   - interpret this as local evidence that the 2D-path failure was upstream to NRSS rather than a remaining deterministic NRSS harness issue.
11. Latest injected-build physics-lane evidence for the expanded suite:
   - command: `bash scripts/run_local_test_report.sh --skip-defaults --cyrsoxs-cli-dir /homes/deand/dev/cyrsoxs/build --cyrsoxs-pybind-dir /homes/deand/dev/cyrsoxs/build-pybind --cmd "python -m pytest tests/validation -m physics_validation -v"`,
   - timestamp/report: `20260321T104515Z` / `test-reports/20260321T104515Z`,
   - result: `10 passed, 2 deselected` in the physics-validation lane.
12. Installed-package cross-check for the new Bragg coverage also passed:
   - command: `CUDA_VISIBLE_DEVICES=1 /home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/validation/test_bragg_2d_lattice.py tests/validation/test_bragg_3d_lattice.py -v`,
   - result: `4 passed in 126.03s`,
   - installed package resolved to `CyRSoXS 1.1.8.0`, patch `9d45790`.

### 8.9 Remaining Phase 0 test-hardening gaps

1. Complete migration of remaining legacy validation workflows to pytest-native modules (especially core-shell and circle lattice).
2. Decide which remaining cases should use analytical references, golden references, or both.
3. Add CI gating policy for CPU smoke and GPU smoke/parity/physics lanes.
4. Revisit follow-on physics coverage such as additional two-material mixed beta/delta equivalence cases if needed.

## 9. Input/Output Contract and Compatibility

1. `backend`: selects execution backend.
2. `input_policy`: governs host/device acceptance and conversion behavior.
3. `output_policy`: `numpy`, backend-native arrays, or objective-only.

Contract requirements:

1. Preserve current default behavior for existing users.
2. Provide explicit conversion methods (`as_numpy`, backend-native accessors).
3. Support GPU-resident objective returns for fitting workflows to avoid forced host copies.
4. Avoid implicit copies; when unavoidable, emit diagnostics in strict mode.

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

## 12. Runtime Observability and Safety Rails

1. Log explicit host<->device transfers at NRSS-controlled boundaries.
2. Add strict mode warnings for policy-driven conversions.
3. Record per-stage timings and peak memory for benchmark/parity runs.
4. Best-effort only for third-party implicit copies; complete interception may not be possible.

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

1. Phase 0: Test hardening + pybind golden baselines + deterministic harness.
2. Phase 1: CuPy backend that mimics CyRSoXS algorithmic flow.
3. Phase 2: Internal math cleanup (tensor-character refactor) while preserving parity tests.
4. Phase 3: Additional backends behind shared backend contract.

Independent success criterion:

1. Completion of Phase 0 is a meaningful modernization outcome even if backend phases are delayed.

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
