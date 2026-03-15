# NRSS Backend Modernization Spec (Superseding)

This document supersedes prior roadmap content in this file.

## 1. Goal

Build a new NRSS backend architecture that:

1. Preserves current trusted physics behavior (CyRSoXS parity first).
2. Enables a CuPy-native simulation backend to avoid unnecessary GPU/CPU/GPU transfers.
3. Supports optional future backends (PyTorch, JAX) without hard dependencies.
4. Improves robustness via deterministic, reproducible regression testing.
5. Delivers durable value even if alternate backend implementation is paused or never completed.

## 2. Guiding Principles

1. Test-first: improve validation harness before backend rewrites.
2. One unknown at a time: first CuPy backend should mimic existing CyRSoXS logic as closely as possible.
3. Preserve backward compatibility by default.
4. Keep scientific workflows practical (notebook-friendly), but ensure reproducibility through scripted pipelines.
5. Treat test-harness modernization as a standalone product-quality objective, not only as a dependency for backend work.

## 3. Phase 0 (Immediate): Validation Harness Overhaul

The existing validation scripts in `tests/validation/` are useful source material but are not robust pytest regression tests. They must be converted into a deterministic test suite.

### 3.1 Baseline Strategy

Use current CyRSoXS **pybind workflow** as reference backend for ground truth generation.

1. Pin trusted environment and versions.
2. Generate curated golden reference artifacts for canonical morphologies.
3. Store compact reference data + metadata (versions, parameters, shapes, dtype).
4. Compare all future backend results against these references.

### 3.2 Canonical Cases

Minimum initial set:

1. Projected sphere case (analytical-informed behavior).
2. Core-shell case (energy-dependent anisotropy behavior).
3. Circle lattice case (peak-position behavior).

### 3.2.1 Analytical Guardrail

In addition to pybind golden references, maintain an analytical comparison track for the projected sphere form factor.

Notes:

1. This is an approximate guardrail because discretization and voxel resolution affect agreement.
2. Use high-resolution runs for stronger agreement checks.
3. Initial implementation can be lightweight and expanded later as notebook-derived methodology is formalized.

### 3.3 Parity Metrics (not one global rule)

Do **not** use one global `%` tolerance across all pixels.

Use layered acceptance:

1. Objective scalar parity (if used in fitting): target <= 1% relative error.
2. Radial `I(q)` parity with q-window masks and `rtol+atol`.
3. Peak-position parity (absolute q tolerance).
4. Optional image-level checks on masked finite support.

High-q tails should be treated with separate tolerances because relative error is unstable there.

Exact threshold tables are intentionally deferred until the test interview/specification pass.

## 4. Phase 1: CuPy Backend (CyRSoXS-Mimic)

Implement CuPy backend with algorithmic flow matching current CyRSoXS stages:

1. Morphology ingestion.
2. Polarization field computation.
3. FFT/shift/DC handling.
4. Scatter + Ewald projection.
5. Rotation/accumulation.
6. Result export.

Scope priority:

1. Calculation parity first.
2. Performance second.
3. Clean mathematical refactors later.

### 4.1 Initial Feature Scope

1. Euler workflows are the primary target (dominant real-world usage).
2. Vector morphology input should remain accepted for backward compatibility.
3. Phase 1 does not require immediate canonical tensor-internal conversion; that is a refactor phase concern.
4. Phase 2 may unify internal representation (tensor/tensor-like) once parity is established.

### 4.2 Constitutive Scaffolding for Future Biaxial Support

Even before implementing biaxial physics, Phase 1 should introduce a constitutive interface boundary so future biaxial support can be implemented in alternate backends without redesigning the full pipeline.

## 5. Input/Output Contract and Backward Compatibility

Default behavior remains compatible with current NRSS usage.

Planned explicit controls:

1. `backend`: backend selector.
2. `input_policy`: host/device handling policy.
3. `output_policy`: `numpy`, backend-native device arrays, or objective-only.

Result API should support conversion methods (`as_numpy`, backend-native access) and avoid implicit copies where possible.

## 6. Optional Dependency Model

Backend libraries must be optional, not mandatory.

1. Base install should not force all backend stacks.
2. Use backend-specific optional extras (for example, `cupy`, `torch`, `jax`).
3. Backend imports should be lazy and fail with actionable error messages.

### 6.1 Packaging Considerations

1. Keep pip extras and conda packaging concerns decoupled in design docs.
2. Anticipate CUDA-version fragility in conda-forge builds and avoid coupling core NRSS installability to any single GPU backend package.
3. Ensure CPU-only/base usage remains installable and testable without GPU backend packages.

## 7. Multi-GPU Execution Model

Primary target workload is model comparison/fitting with expensive objective evaluation.

Preferred pattern:

1. Model-parallel scheduling (one model per GPU worker).
2. Persistent worker processes (for allocation/plan reuse).
3. Avoid dependence on CyRSoXS internal multi-GPU-energy splitting.

## 8. Memory and Precision Policy

### 8.1 Memory policy

Optimize for **peak memory**, not just per-step slimness.

1. Aggressively release/reuse transient buffers (especially polarization fields).
2. Support chunked/streamed output to avoid large resident result tensors.
3. Prefer objective-only output mode for fitting workflows.
4. Use configurable memory guardrails; high utilization (including up to ~95% of 48 GB) is allowed when explicitly configured.
5. Define chunking order defaults: chunk `energy` first, then `angle`, then `k`.

### 8.2 Precision policy

1. Core compute target: `float32` / `complex64`.
2. 16-bit is acceptable for storage/compression of morphology fields.
3. Do not run FFT/q-space pipeline in 16-bit for parity-sensitive runs.
4. If using 16-bit storage, decode/cast inside consuming kernels.

## 9. Input Matrix and Determinism Requirements

### 9.1 Input/Backend Parity Matrix

Phase 0/1 tests must explicitly cover:

1. `numpy` inputs -> CyRSoXS reference backend.
2. `cupy` inputs -> CyRSoXS reference backend (where applicable via conversion path).
3. `numpy` inputs -> CuPy backend.
4. `cupy` inputs -> CuPy backend.

### 9.2 Determinism

1. If any test generation path uses randomness, all RNG seeds must be fixed and recorded.
2. Test fixtures must include metadata sufficient to reproduce arrays and simulation settings.
3. Determinism checks are required for golden generation pipelines.

## 10. Development Workflow

### 10.1 Source control

1. Develop on feature branches in main repo.
2. Keep commits small and frequent.
3. Merge in milestone-sized PRs.

### 10.2 Environments

1. Keep stable scientific env untouched.
2. Use dedicated dev env(s), install NRSS editable.
3. Pin versions for parity work.

### 10.3 Notebook + staging workflow

Use notebooks for exploration/visualization in a staging workspace, but require script-based reproducibility for official baselines and regression assets.

## 11. Runtime Observability

Provide best-effort transfer/memory observability:

1. Log explicit host<->device transfers in backend code paths.
2. Add strict/diagnostic mode to warn when conversions are triggered by policy (`input_policy`, `output_policy`).
3. Record peak memory and stage timings for parity/benchmark runs.

Note: complete detection of all implicit third-party transfers may not be possible in every backend/runtime, but instrumentation should cover NRSS-controlled boundaries.

## 12. Phased Plan

1. Phase 0: robust pybind-based regression suite + golden datasets.
2. Phase 1: CuPy mimic backend with parity gates.
3. Phase 2: CuPy math cleanup/refactor (vector/tensor formulation) with unchanged tests.
4. Phase 3: Additional backends (PyTorch/JAX) via shared backend contract.

### 12.1 Independent Success Path

If backend phases are delayed, halted, or descoped, completion of Phase 0 is still considered a successful modernization milestone for NRSS quality and release safety.

## 13. Explicit Non-Goals (Initial Phases)

1. Immediate biaxial implementation.
2. Immediate full feature parity across every legacy mode before core parity path is stable.
3. Optimization-first changes before parity harness exists.

## 14. Golden Data Governance

1. Golden data is generated now from trusted pybind reference runs.
2. Golden datasets are regenerated only when a known scientific/physics bug is confirmed and corrected (or when intentionally changing physical behavior).
3. Golden updates require explicit changelog notes and reviewer signoff.

## 15. Open Items to Decide During Implementation

1. Exact parity thresholds by metric and q-region.
2. Initial feature subset for CuPy mimic backend.
3. Golden dataset size/retention policy in repository.
4. Automation strategy for GPU-required tests vs CPU-only smoke checks.
5. Minimal acceptable performance tracking thresholds for release gating.
