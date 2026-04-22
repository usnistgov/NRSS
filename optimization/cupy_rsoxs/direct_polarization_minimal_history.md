# `direct_polarization` Minimal History

Open this file only when you need compact provenance for the direct path.

Do not open the archive unless one of the rows below fails to answer a specific
historical question.

## Accepted And Retained

| Idea | Status | Minimal Note |
| --- | --- | --- |
| materially different detector-plane direct projection path | accepted | direct-path `D` now projects FFT polarization directly on the detector grid |
| detector projection kernels on `nvcc`-preferred path | accepted | removed the earlier cold-process memory penalty on the maintained environment |
| constructor-time preload plus mixed backend-family defaults | accepted | maintained direct-path defaults are `a1` preload, `igor_shift=nvcc`, `direct_polarization_generic=nvrtc` |
| true in-place direct-path `Segment C` | accepted | cuFFT and Igor-order swap now operate in place |
| fused float32 isotropic accumulation | accepted | removed the direct float32 isotropic cache-plus-temporary structure and lowered peak memory |
| runtime zero-field shortcut for exact-zero legacy materials | accepted | exact-zero `legacy_zero_array` compatibility inputs now stage only `Vfrac` and route through isotropic handling in both supported residency modes, without rewriting the authoritative material contract |

## Rejected Or Closed

| Idea | Status | Why Not Retry Blindly |
| --- | --- | --- |
| scalar hoist / material split refactors | rejected | no material authority-lane win; keep only as possible scaffolding for deeper rewrites |
| pure-CuPy `Segment B` scratch reuse | rejected | regressed maintained host-prewarmed timing |
| explicit `x` / `y` / general-angle Python-CuPy split | rejected | regressed maintained host-prewarmed timing |
| aligned-family custom kernels as default-path change | rejected | hot no-rotation win existed, but cold authority timing regressed badly |
| early detector-plane direct projection formulation | rejected | only the later materially different formulation was worth keeping |
| memory-cleanup items `1-3` from April 6 | rejected | failed the direct-hot peak-memory gate on this environment |
| bucketed material-loop follow-up after runtime zero-field shortcut | rejected | no extra memory win over the simpler accepted shortcut |
| device-resident reusable-field precompute opt-in | closed | medium-model direct-path HOT checks showed no speedup, slight slowdown, and modest runtime-memory growth, so this branch was intentionally discontinued |

## Still Open

| Idea | State | Next Place To Look |
| --- | --- | --- |
| higher-memory multi-angle cache mode | retry candidate | `remaining_untried_ideas.md` |
| packed voxel-style staging plus device-owned material loop | deferred | `remaining_untried_ideas.md` |
| narrower orientation-only cache mode | untried | `remaining_untried_ideas.md` |
| fuse windowing into direct generation | skipped | `remaining_untried_ideas.md` |
