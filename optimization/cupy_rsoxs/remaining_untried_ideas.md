# Remaining Untried Ideas

Open this file when the task is "what is still left to try?".

This file intentionally excludes already-landed ideas.
It also separates truly untried work from partially tried work that is not yet
cleanly closed.

## Truly Untried

| Path | Idea | Primary Goal | Segment | Note |
| --- | --- | --- | --- | --- |
| `shared` | host-resident derivative staging plus isotropic/anisotropic runtime split | speed | `A2/B` | next major untried attempt; for host residency, stage `phi_a`, `sx`, `sy`, `sz` in `A2`, replace repeated per-energy trig/product work in `B`, and split isotropic vs anisotropic material paths so anisotropic kernels no longer need raw `S/theta/psi` and may not need `Vfrac`; device residency remains a separate higher-memory fast-path question because staged arrays may alias authoritative CuPy inputs |
| `tensor_coeff` | `planE05_tensor_coeff_projection_rotation_fusion` | speed | `E` | combine `x/y/xy` weighting with rotation accumulation and avoid the intermediate per-angle detector image |
| `tensor_coeff` | `planE06_exact_quarter_turn_fast_path` | speed | `E` | use exact `90/180/270` transforms when rotation lands on the detector grid |
| `tensor_coeff` | `planD06_segment_d_scratch_reuse` | possible memory / allocator relief | `D` | documented but never implemented; low priority because the remaining `D` surface was not convincingly allocator-limited |
| `direct_polarization` | fuse windowing into direct polarization generation | speed / memory traffic | `B/C` boundary | skipped rather than rejected; only relevant when windowing is active |
| `direct_polarization` | packed voxel-style runtime staging plus device-owned material loop | speed / memory / launch pressure | `B` | structural CyRSoXS-like redesign; still deferred |
| `direct_polarization` | narrower orientation-only cache mode | throughput with smaller memory tradeoff than full cache mode | multi-angle `B` | documented but not attempted |

### Resumption Notes: host-resident derivative staging plus isotropic/anisotropic runtime split

This is the next major untried optimization to attempt.

Current repeated work that motivates the idea:

- `A2` currently stages runtime material views once per run, but it stages raw
  morphology fields (`Vfrac`, `S`, `theta`, `psi`) rather than derivative
  products.
- `tensor_coeff` `B` currently recomputes `phi_a = Vfrac * S` and orientation
  components `sx`, `sy`, `sz` for every energy.
- `direct_polarization` `B` currently recomputes the same orientation-derived
  quantities for every energy, and repeats per-angle work inside the direct
  path.
- Multi-energy sweeps therefore continue paying for repeated trig / product
  work even though these quantities depend only on morphology, not energy.

Proposed host-resident fast path:

- Restrict the first attempt to `resident_mode='host'`.
- During `A2`, after staging CuPy runtime arrays from authoritative NumPy
  morphology inputs, build derivative runtime fields for anisotropic materials:
  - `phi_a = Vfrac * S`
  - `sx = cos(psi) * sin(theta)`
  - `sy = sin(psi) * sin(theta)`
  - `sz = cos(theta)`
- Keep `Vfrac` available for isotropic work.
- Stop passing raw `S`, `theta`, `psi` into anisotropic `B` kernels once the
  staged derivative path exists.

Runtime-layout idea:

- Split runtime materials into isotropic and anisotropic families rather than
  carrying one mixed view shape.
- Isotropic runtime entries should retain only the fields needed for isotropic
  accumulation, primarily `Vfrac` plus optical constants / material id.
- Anisotropic runtime entries should retain the derivative fields needed by the
  hot `B` kernels:
  - `phi_a`
  - `sx`
  - `sy`
  - `sz`
- For the anisotropic path, `Vfrac` may become unnecessary after `A2`; confirm
  this separately for each execution path before removing it from the runtime
  view.

Why this is attractive:

- It moves morphology-only math out of per-energy `B` and into one-time `A2`.
- It should help both maintained execution paths.
- For host residency, the staged CuPy arrays are runtime copies, not the
  authoritative NumPy fields, so replacing raw staged orientation fields with
  derivative fields is semantically safe.
- The memory tradeoff is modest relative to larger cache-mode ideas because the
  proposal reuses the existing runtime-staging concept rather than creating an
  energy-batched tensor cache.

Device-residency caveat:

- Do not assume the same in-place strategy is safe for `resident_mode='device'`.
- In the current runtime staging flow, arrays that already satisfy the runtime
  contract may be returned unchanged, so staged arrays can alias authoritative
  CuPy morphology inputs.
- `cupy-rsoxs` also defaults to borrowed ownership, so overwriting staged
  device arrays can corrupt authoritative morphology state.
- Device residency may still benefit from a related fast path, but that should
  be treated as a separate higher-memory cache mode or explicit copy mode, not
  as the same default host-resident optimization.

Likely implementation shape for the first attempt:

- Extend runtime staging in `A2` to optionally produce derivative anisotropic
  views for host residency.
- Introduce explicit runtime-view types or fields for isotropic vs
  anisotropic materials rather than overloading `theta`/`psi` semantics.
- Update `tensor_coeff` float32 `B` kernels to consume derivative anisotropic
  inputs directly.
- Update `direct_polarization` float32 `B` kernels to consume derivative
  anisotropic inputs directly.
- Leave half-input / mixed-precision support out of the first attempt unless
  the float32 path lands cleanly.
- Leave device-resident morphology support behavior unchanged in the first
  attempt.

Recommended first benchmark / acceptance shape:

- Start with host-resident float32 morphology only.
- Compare one-morphology multi-energy sweeps before and after the change.
- Measure at least `A2`, `B`, and end-to-end runtime with the private backend
  timing segments.
- Run both execution paths:
  - `tensor_coeff`
  - `direct_polarization`
- Include at least one angle-light and one angle-heavy case, because the direct
  path repeats `B` work inside the angle loop.
- Require no physics regressions against the maintained validation surface
  before treating the change as accepted.

## Partially Tried Or Still Live

| Path | Idea | Current State | Why Still Open |
| --- | --- | --- | --- |
| `tensor_coeff` | `planE04_rotate_accumulate_kernel_fusion` | exploratory device-only screen | never carried through host-prewarmed comparison or required physics gates |
| `direct_polarization` | higher-memory multi-angle cache mode | attempted, not accepted | timing looked interesting, but the rotation-parity check crashed, so it is a retry candidate rather than a clean rejection |

## Not Included Here

- Already accepted items are in `accepted_state.md`.
- Retried or rejected ideas are compressed into:
  - `tensor_coeff_minimal_history.md`
  - `direct_polarization_minimal_history.md`
