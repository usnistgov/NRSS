# `tensor_coeff` Minimal History

Open this file only when you need compact provenance for `tensor_coeff`.

Do not open the archive unless one of the rows below is insufficient.

## Accepted And Retained

| Idea | Status | Minimal Note |
| --- | --- | --- |
| detector / projection geometry cache scaffolding | retained | useful scaffolding, but not treated as the main accepted speed win by itself |
| detector-grid helper projection path for general-angle and aligned `x` / `y` families | accepted | replaced extra `scatter3d` materialization in the wrapper |
| float32 fused isotropic accumulation (`mem09`) | accepted | removed the float32 `isotropic_term` temporary and materially improved host-hot `Segment B` plus peak memory |
| legacy-zero shortcut (`mem10`) | accepted | exact-zero `legacy_zero_array` inputs now stage only `Vfrac` and route through isotropic handling in both supported residency modes, without rewriting the authoritative material contract |

## Rejected Or Closed

| Idea | Status | Why Not Retry Blindly |
| --- | --- | --- |
| direct detector kernel path (`planD03`) | rejected | cold first-use cost was too large on the authority surface |
| static `D` algebra prefactor cache (`planD04`) | rejected | trimmed targeted `D` work slightly, but primary time stayed effectively flat |
| scale hoist (`planD05`) | rejected | general-angle primary timing regressed |
| Segment `D` scratch reuse (`planD06`) | closed without implementation | after `planD02`, the remaining `D` surface no longer looked allocator-limited enough to justify the pass |
| rotation accumulation scratch reuse (`planE01`) | rejected | non-material on the maintained explicit-rotation surface |
| texture affine transform probe (`planE02`) | rejected | host-prewarmed `0:15:165` lane regressed |
| rotmask-zero fast path (`planE03`) | rejected | rebuilding the strict mask surface was too expensive |

## Still Open

| Idea | State | Next Place To Look |
| --- | --- | --- |
| rotate-accumulate kernel fusion (`planE04`) | partially tried | `remaining_untried_ideas.md` |
| projection-plus-rotation fusion (`planE05`) | planned | `remaining_untried_ideas.md` |
| exact quarter-turn fast path (`planE06`) | planned | `remaining_untried_ideas.md` |
