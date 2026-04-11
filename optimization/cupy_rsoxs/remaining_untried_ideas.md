# Remaining Untried Ideas

Open this file when the task is "what is still left to try?".

This file intentionally excludes already-landed ideas.
It also separates truly untried work from partially tried work that is not yet
cleanly closed.

If the task is to resume the shared host-resident reusable-staging branch from a
cold start, open `shared_reusable_staging_branch_plan.md` after this file.

## Truly Untried

| Path | Idea | Primary Goal | Segment | Note |
| --- | --- | --- | --- | --- |
| `shared` | direct-path isotropic/anisotropic rawkernel split after reusable staging | speed | `B` | host-resident float32 GPU reusable staging is now the maintained standard path, but the float32 `direct_polarization` precomputed kernel still reads `Vfrac` only to pay for the isotropic term; split isotropic and anisotropic direct kernels so anisotropic-precomputed work consumes only `phi_a`, `sx`, `sy`, `sz` |
| `tensor_coeff` | `planE05_tensor_coeff_projection_rotation_fusion` | speed | `E` | combine `x/y/xy` weighting with rotation accumulation and avoid the intermediate per-angle detector image |
| `tensor_coeff` | `planE06_exact_quarter_turn_fast_path` | speed | `E` | use exact `90/180/270` transforms when rotation lands on the detector grid |
| `tensor_coeff` | `planD06_segment_d_scratch_reuse` | possible memory / allocator relief | `D` | documented but never implemented; low priority because the remaining `D` surface was not convincingly allocator-limited |
| `direct_polarization` | fuse windowing into direct polarization generation | speed / memory traffic | `B/C` boundary | skipped rather than rejected; only relevant when windowing is active |
| `direct_polarization` | packed voxel-style runtime staging plus device-owned material loop | speed / memory / launch pressure | `B` | structural CyRSoXS-like redesign; still deferred |
| `direct_polarization` | narrower orientation-only cache mode | throughput with smaller memory tradeoff than full cache mode | multi-angle `B` | documented but not attempted |

### Resumption Notes: direct-path isotropic/anisotropic rawkernel split after reusable staging

The CPU-vs-GPU reusable-staging comparison is now closed.

Historical branch plan and implementation notes:

- `shared_reusable_staging_branch_plan.md`

Completed outcome:

- CPU-side reusable computation was rejected:
  - slower than the incumbent host path on both maintained execution paths,
  - higher peak GPU memory than baseline,
  - no reason to keep it as a maintained or benchmarked branch surface.
- GPU-side reusable computation in host-resident float32 mode was accepted:
  - `A2` stages `Vfrac`, `S`, `theta`, `psi` one anisotropic material at a time,
  - GPU precompute builds `phi_a`, `sx`, `sy`, `sz`,
  - raw staged `S`, `theta`, `psi` are dropped immediately,
  - steady-state runtime keeps `Vfrac + phi_a + sx + sy + sz` for anisotropic
    host-resident float32 materials,
  - this is now the maintained standard host-resident float32 path rather than
    an experiment switch.

Authority artifacts from the completed comparison:

- speed comparison summary:
  - `test-reports/cupy-rsoxs-optimization-dev/codex_shared_reusable_authority_20260410a/summary.json`
- peak-memory summary:
  - `test-reports/core-shell-backend-performance-dev/codex_shared_reusable_memprobe_20260410T170810Z/shared_reusable_memory_summary.json`

What remains open is narrower than the original branch plan:

- `tensor_coeff` benefited from reusable staging and is already on the
  intended structure.
- `direct_polarization` did not show a meaningful end-to-end multi-energy win
  on the heavier host-resident lane that should have challenged it:
  - artifact:
    - `test-reports/core-shell-backend-performance-dev/codex_dp_20energy_0_15_165_exactband_20260410T172052Z/dp_20energy_0_15_165_exactband_summary.json`
  - lane:
    - 20 energies using exact CoreShell optics keys from `280.0` to `289.5` eV
      in `0.5` eV steps,
    - `EAngleRotation=[0,15,165]`
  - result:
    - baseline median primary `1.7327 s`,
    - reusable-GPU median primary `1.7289 s`,
    - effectively flat at `-0.22%`.

Why the direct path likely stayed flat:

- The incumbent float32 `direct_polarization` path was already a fused raw
  kernel, not an unfused Python/CuPy expression chain.
- The precomputed float32 direct kernel still reads `Vfrac` because isotropic
  and anisotropic accumulation remain fused in one kernel.
- In the current precomputed direct kernel, `Vfrac` is used only for the
  isotropic diagonal term; the anisotropic contribution itself uses only
  `phi_a`, `sx`, `sy`, `sz`.
- So the current direct precompute path trades inline trig / product work for:
  - an extra one-time `A2` GPU precompute kernel,
  - an extra steady-state global read of `phi_a`,
  - while still retaining the steady-state `Vfrac` read for isotropic work.
- That means the direct-path precomputed kernel has not yet reached the cleaner
  anisotropic-only memory shape implied by the original branch idea.

Evidence that the anisotropic split is mathematically valid:

- The collapsed-mean direct path already computes isotropic work from `Vfrac`
  separately and anisotropic work from `phi_a + sx/sy/sz`.
- So the remaining issue is kernel layout, not physics semantics.

Recommended next implementation:

- Keep the accepted reusable-staging runtime layout.
- For float32 `direct_polarization`, split anisotropic-material work into:
  - isotropic-only accumulation kernel using `Vfrac`,
  - anisotropic-precomputed kernel using only `phi_a`, `sx`, `sy`, `sz`.
- Do not change device-resident behavior in the first pass.
- Do not expand the first pass to half-input / mixed precision.

Recommended resume benchmark shape:

- Measure at least `A2`, `B`, and primary end-to-end timing.
- Use host-resident float32 only.
- Keep the heavy direct-path challenge lane:
  - 20 exact CoreShell energies from `280.0` to `289.5` eV,
  - `EAngleRotation=[0,15,165]`
- Recheck the noisier angle-heavy lane if needed:
  - triple-energy `0:5:175`
- Require no physics regressions before treating the split as accepted.

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
