# Remaining Untried Ideas

Open this file when the task is "what is still left to try?".

This file intentionally excludes already-landed ideas.
It also separates truly untried work from partially tried work that is not yet
cleanly closed.

## Truly Untried

| Path | Idea | Primary Goal | Segment | Note |
| --- | --- | --- | --- | --- |
| `tensor_coeff` | `planE05_tensor_coeff_projection_rotation_fusion` | speed | `E` | combine `x/y/xy` weighting with rotation accumulation and avoid the intermediate per-angle detector image |
| `tensor_coeff` | `planE06_exact_quarter_turn_fast_path` | speed | `E` | use exact `90/180/270` transforms when rotation lands on the detector grid |
| `tensor_coeff` | `planD06_segment_d_scratch_reuse` | possible memory / allocator relief | `D` | documented but never implemented; low priority because the remaining `D` surface was not convincingly allocator-limited |
| `direct_polarization` | fuse windowing into direct polarization generation | speed / memory traffic | `B/C` boundary | skipped rather than rejected; only relevant when windowing is active |
| `direct_polarization` | packed voxel-style runtime staging plus device-owned material loop | speed / memory / launch pressure | `B` | structural CyRSoXS-like redesign; still deferred |
| `direct_polarization` | narrower orientation-only cache mode | throughput with smaller memory tradeoff than full cache mode | multi-angle `B` | documented but not attempted |

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
