# Shared Reusable Staging Branch Plan

Use this file when the task is to resume or implement the next shared
`cupy-rsoxs` optimization branch:

- host-resident reusable staging in `A2`,
- stronger isotropic / anisotropic runtime separation,
- multi-energy speedup work centered on `B`.

Start from `README.md` for routing.
Open `remaining_untried_ideas.md` first for the compact ranking context, then
open this file for the detailed resumable branch plan.

This document is intentionally a working-branch handoff, not a maintained-state
record. If this work lands, compress the accepted outcome back into:

- `accepted_state.md`
- `remaining_untried_ideas.md`
- path-specific minimal history docs as needed

## Status

- Branch state: planned, not yet attempted in this documented form
- Priority: next major shared optimization attempt
- Primary target: host-resident float32 multi-energy throughput
- Primary lane: CoreShell triple energy with `EAngleRotation=[0, 15, 165]`

## Goal

Move morphology-only reusable work out of per-energy `B` and into one-time `A2`
for host residency, while keeping peak memory flat enough to preserve the
default-path memory posture.

Reusable fields of interest:

- anisotropic path:
  - `phi_a = Vfrac * S`
  - `sx = cos(psi) * sin(theta)`
  - `sy = sin(psi) * sin(theta)`
  - `sz = cos(theta)`
- isotropic path:
  - `Vfrac`

Current repeated work lives in:

- runtime staging: `src/NRSS/backends/cupy_rsoxs.py::_runtime_material_views`
- tensor float32 hot path:
  - `_compute_nt_components`
  - `_compute_nt_components_collapsed_mean`
  - `_nt_accumulate_anisotropic_float32_kernel`
- direct float32 hot path:
  - `_compute_direct_polarization`
  - `_compute_direct_polarization_collapsed_mean`
  - `_direct_generic_kernel_float32`

## Locked Scope Decisions

- First attempt is `resident_mode='host'` only.
- Device-resident behavior stays unchanged in the prototype round.
- Half-input / mixed-precision support is out of scope for the first attempt.
- `z_collapse_mode='mean'` is out of scope for the first attempt.
- The slow maintained physics suite is not the first gate for this work.
- Temporary fast parity checks are the first regression gate.

## Important Constraint: Keep Anisotropic `Vfrac` For Round 1

Do not assume anisotropic runtime entries can drop `Vfrac` in the first round.

Reason:

- both maintained execution paths still use `Vfrac` for the isotropic
  contribution each energy,
- removing anisotropic `Vfrac` would require a deeper algebra or staging change
  than "precompute morphology-only reusables."

Round-1 runtime split should therefore be:

- isotropic runtime entries:
  - `materialID`
  - optical constants
  - `Vfrac`
- anisotropic runtime entries:
  - `materialID`
  - optical constants
  - `Vfrac`
  - `phi_a`
  - `sx`
  - `sy`
  - `sz`

Round 1 should remove staged raw `S`, `theta`, and `psi` from the anisotropic
runtime view after `A2`.

## Runtime Separation To Try

The first branch should try more than one degree of iso / aniso separation.
The variants are ordered to isolate one variable at a time.

### Variant A: host CPU precompute plus fused `B` kernels

What changes:

- in `A2`, compute anisotropic reusables on CPU from authoritative NumPy
  morphology fields before staging them to GPU,
- stage only final runtime arrays for anisotropic materials,
- keep explicit isotropic materials on the existing `Vfrac`-only path,
- add new float32 anisotropic kernels that consume `Vfrac + phi_a + sx + sy + sz`
  directly,
- keep isotropic and anisotropic accumulation fused inside each anisotropic
  kernel.

Purpose:

- isolate the benefit of moving trig and `Vfrac * S` work out of per-energy `B`
  without also changing the compute site for the reusables,
- avoid introducing GPU-side `A2` peak-memory concerns in the first comparison.

Expected tradeoff:

- lower `B`,
- possibly higher host preprocessing cost before staging,
- likely simple memory story.

### Variant B: host GPU precompute plus fused `B` kernels

What changes:

- same runtime layout as Variant A,
- same float32 anisotropic kernels as Variant A,
- only the compute site for the reusables changes,
- in `A2`, stage raw anisotropic fields to GPU one material at a time,
  precompute reusables on GPU, delete raw staged fields immediately, and retain
  only `Vfrac + phi_a + sx + sy + sz`.

Purpose:

- compare where reusable computation should happen,
- test whether GPU-side `A2` precompute wins enough to justify the extra
  staging complexity.

Expected tradeoff:

- lower `B`,
- possibly lower or higher `A2` than Variant A depending on transfer balance,
- more sensitive to temporary allocation peaks.

### Variant C: winner of A/B plus split isotropic and anisotropic `B` work

What changes:

- start from whichever of A or B wins,
- keep anisotropic runtime entries as `Vfrac + phi_a + sx + sy + sz`,
- route all isotropic contribution through isotropic-only kernels,
- route all anisotropic contribution through smaller anisotropic-only kernels,
- the anisotropic kernels no longer handle isotropic accumulation.

Purpose:

- test whether greater iso / aniso path separation reduces register pressure,
  simplifies hot kernels, or improves maintained timing.

Expected tradeoff:

- more kernel launches,
- potentially smaller anisotropic kernels,
- potentially clearer memory and compute ownership by path.

If Variant C loses, keep the winning fused-`B` variant from A/B.

## Specific Kernel Work To Try

### New float32 `B` kernels to add

Tensor path:

- precomputed-fused anisotropic kernel
  - inputs:
    - `vfrac`
    - `phi_a`
    - `sx`
    - `sy`
    - `sz`
    - optical scalars
    - `need0..need4`
  - behavior:
    - keep isotropic contribution fused in the kernel for Variants A/B

Direct path:

- precomputed-fused anisotropic kernel
  - inputs:
    - `vfrac`
    - `phi_a`
    - `sx`
    - `sy`
    - `sz`
    - optical scalars
    - `mx`
    - `my`
  - behavior:
    - keep isotropic contribution fused in the kernel for Variants A/B

### Optional float32 `B` kernels for Variant C

Tensor path:

- precomputed anisotropic-only kernel
  - inputs:
    - `phi_a`
    - `sx`
    - `sy`
    - `sz`
    - optical scalars
    - `need0..need4`

Direct path:

- precomputed anisotropic-only kernel
  - inputs:
    - `phi_a`
    - `sx`
    - `sy`
    - `sz`
    - optical scalars
    - `mx`
    - `my`

### New `A2` precompute kernel to add for Variant B

Prefer one fused float32 precompute kernel rather than CuPy expression chains.

Inputs:

- `vfrac`
- `s`
- `theta`
- `psi`

Outputs:

- `phi_a`
- `sx`
- `sy`
- `sz`

Reason:

- avoid materializing `sin(theta)` or other intermediate arrays,
- keep the peak-memory story legible,
- avoid CuPy expression-temporary ambiguity when measuring allocator high-water
  mark.

### Existing kernels to keep

- keep the current float32 isotropic kernels for both paths,
- keep all half-input kernels unchanged in round 1,
- keep device-resident path kernels unchanged in round 1.

## `A2` Compute-Site Comparison

The reusable-compute site is a first-class comparison, not an implementation
detail to lock in early.

### CPU-side reusable computation

Pros:

- simplest memory story,
- no transient GPU overlap between raw staged orientation fields and final
  reusables,
- likely easiest to reason about for host default workflows.

Cons:

- more host compute before transfer,
- sends more final arrays over PCIe instead of fewer raw arrays.

### GPU-side reusable computation in host mode

Pros:

- may reduce host-side preprocessing cost,
- may shift the reusable build to the faster device math path.

Cons:

- memory high-water mark can easily regress if raw staged fields and final
  reusables overlap too long,
- requires tighter lifetime discipline.

Do not choose between these on intuition alone. Compare Variant A vs Variant B
on the same authority lane.

## GPU-Side Memory Minimization Strategy For Variant B

If reusables are computed on device in host mode, use all of the following:

1. Stage one anisotropic material at a time.
2. Do not stage raw anisotropic fields for all materials at once.
3. Use one fused precompute kernel that writes final outputs directly.
4. Do not materialize `sin(theta)` as a standalone array.
5. Delete staged raw `S`, `theta`, and `psi` immediately after the fused
   precompute completes for that material.
6. Keep explicit isotropic materials on the existing `Vfrac`-only path.
7. Measure with the external peak-memory observer, not pool-internal intuition.

Note:

- CuPy pool retention means "deleted" arrays still matter if they were present
  at the simultaneous-allocation peak.
- The relevant authority is the external baseline-subtracted peak observer.

## Fast Parity Strategy

The maintained test suite is intentionally not the first gate for this work.

Use a temporary fast parity surface first:

1. same execution path, before vs candidate
2. small CoreShell
3. triple energy:
   - `284.7`
   - `285.0`
   - `285.2`
4. `EAngleRotation=[0, 15, 165]`
5. compare both maintained paths:
   - `tensor_coeff`
   - `direct_polarization`

Primary fast parity rule:

- candidate must match the current baseline closely on the same path.

Secondary sanity rule:

- compare candidate `tensor_coeff` vs candidate `direct_polarization`, but do
  not use that as the primary pass/fail authority because this branch touches
  both paths.

Use reduced summary metrics plus direct detector-panel comparison.

If a candidate survives the temporary gate and looks materially promising, only
then escalate to the slower maintained physics surface.

## Benchmark Plan

### Ranking lane

- family:
  - CoreShell
- size:
  - small first
- energies:
  - triple energy `284.7, 285.0, 285.2`
- rotation:
  - `EAngleRotation=[0, 15, 165]`
- residency:
  - host
- dtype:
  - float32
- warm state:
  - host hot / prewarmed worker
- execution paths:
  - `tensor_coeff`
  - `direct_polarization`

### Measurements

Collect:

- `A2`
- `B`
- primary time

Use the existing private backend timing segments.

### Memory lane

Use a focused dev memory recheck patterned after the current memory recheck
scripts:

- same small CoreShell
- same triple-energy rotated lane
- host hot
- external warmed CuPy observer
- baseline-subtracted peak GPU memory
- parent-side RSS polling when already part of the harness pattern

## Acceptance Gates

The branch should be considered promising only if it passes all of:

1. speed:
   - at least 5% primary-time improvement on the authority lane
2. memory:
   - no meaningful peak-memory increase
   - use 5% as the practical guardrail for the prototype ranking pass
3. temporary parity:
   - no obvious regression on the fast same-path comparison

If speed wins but memory loses materially, do not treat it as the default-path
winner.

## Breadcrumbs For Future Resume

Open in this order:

1. `README.md`
2. `remaining_untried_ideas.md`
3. this file
4. code seams:
   - `src/NRSS/backends/cupy_rsoxs.py`
   - `src/NRSS/material_contracts.py`
   - `src/NRSS/backends/contracts.py`
5. benchmark authority:
   - `optimization/cupy_rsoxs/benchmarking_guide.md`
   - `tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py`
   - `tests/validation/dev/core_shell_backend_performance/README.md`

Questions already resolved for this branch:

- first implementation leaves `resident_mode='device'` unchanged,
- any later device cache / lower-memory mode is a note only for now,
- `0:15:165` is sufficient as the authority rotation lane,
- temporary fast parity is the first regression gate,
- maintained full-suite validation is deferred until a promising prototype
  exists.

## Explicitly Deferred

- device-resident default-mode redesign
- public option-surface changes for device cache vs low-memory behavior
- half-input support updates
- mixed-precision updates
- `z_collapse_mode='mean'` updates
- deeper algebra intended to remove anisotropic `Vfrac`

