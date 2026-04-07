# Fast Approximations

Open this file only for approximation-mode questions.

If the task is ordinary optimization of the maintained exact paths, do not open
this file; start from `accepted_state.md` or `remaining_untried_ideas.md`
instead.

## Scope

This file covers expert-only approximation modes that trade fidelity or support
breadth for speed or memory relief.

## `z_collapse_mode='mean'`

- Implemented on both maintained execution paths.
- Remains expert-only rather than a generally recommended default.
- The backend keeps effective `z=1` downstream FFT and detector semantics after
  collapse.
- The public morphology shape is not mutated.
- The remaining active work is mostly support posture and effective-`2D`
  detector cleanup, not another major collapse implementation pass.

## Half-Input / Mixed-Precision Path

- Half-input support remains part of the maintained runtime on the normal
  non-collapsed paths.
- `z_collapse_mode='mean'` does not support the half-input mixed-precision path.
- Treat `z_collapse_mode` plus half-input as an unresolved redesign question,
  not as a routine combination to reopen casually.

## Routing

- If the question is:
  - "should I use collapse?" stay in this file.
  - "what is the exact contract?" open `backend_spec.md`.
  - "how was the collapse feature originally implemented?" open the archive only
    if this file and the backend spec are insufficient.

## Archive Boundary

- Historical design detail lives at:
  - `archive/CUPY_RSOXS_Z_COLLAPSE_PROPOSAL.md`
- Do not open that archive unless you are redesigning the approximation,
  revisiting validation breadth, or reconstructing a specific old decision.
