# Validation And Status

Open this file when you need the current maintained `cupy-rsoxs` validation,
path-matrix, and implementation-status picture without reading the archived
repo plan.

If you need benchmark commands or optimization acceptance gates, use
`benchmarking_guide.md` instead.

## Current Repo-Level `cupy-rsoxs` Status

- `cupy-rsoxs` now exists in-repo as the maintained default backend when both
  `cupy-rsoxs` and `cyrsoxs` are available.
- CuPy is a required maintained runtime dependency.
- The backend-preparation milestone is complete for the current
  registry/routing/reporting scope.
- Test hardening is implemented and remains valuable even independent of
  backend speed work.

## Maintained Validation Surface

- The maintained validation surface now runs as a peer-path matrix across:
  - `legacy_cyrsoxs`
  - `cupy_tensor_coeff`
  - `cupy_direct_polarization`
- Maintained validation cases:
  - analytical sphere form factor
  - sphere contrast scaling
  - sphere orientational contrast scaling
  - analytical 2D disk form factor
  - 2D disk contrast scaling
  - 2D Bragg lattice
  - 3D Bragg lattice
  - CoreShell
  - MWCNT
- Current maintained physics status:
  - `14` tests per path
  - no maintained CuPy skips on the principal physics matrix

## Smoke And Routing Status

- `tests/smoke` and `tests/validation` are now routed through maintained
  explicit path selection rather than relying on hidden backend defaults.
- The shared maintained routing model is path-first rather than backend-first.
- CLI-vs-pybind checks remain legacy compatibility checks rather than defining
  the principal maintained path matrix.

## Residency And Runtime Policy

- The maintained CuPy peer paths use device residency for the standard smoke
  and physics matrix after lightweight host-vs-device parity coverage.
- Host-resident coverage is still retained for backend-contract and staging
  behavior where it matters.
- In host-resident float32 mode, anisotropic materials now use standard GPU
  reusable staging during `A2`; the older CPU-side comparison branch has been
  removed from the maintained and dev-harness surfaces.
- Single-GPU execution remains the maintained parity target.

## Current Open Repo-Level Work

- resident-mode refinement
- segment-targeted optimization
- export timing follow-up
- deeper memory instrumentation
- GPU CI strategy and final gating matrix

## Fast Approximation Status

- Expert-only `z_collapse_mode='mean'` exists on both maintained execution
  paths.
- Approximation-specific support posture and half-input caveats live in
  `fast_approximations.md`.

## Archive Boundary

- The long-form pre-cleanup repo plan is preserved at:
  - `archive/REPO_UPGRADE_PLAN_pre_cleanup.md`
- Do not open that archive unless you need historical implementation detail
  that is intentionally absent from this compact status view.
