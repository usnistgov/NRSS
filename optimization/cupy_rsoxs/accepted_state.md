# Accepted `cupy-rsoxs` State

Open this file when you need the current maintained optimization state.

If you need untried work, go to `remaining_untried_ideas.md`.
If you need benchmark commands or acceptance gates, go to `benchmarking_guide.md`.

## Maintained Execution Paths

- `tensor_coeff`
  - maintained default execution path
  - primary optimization target for angle-heavy workloads
- `direct_polarization`
  - maintained alternate path
  - useful when lower memory footprint or direct-path behavior is the point

## Current Accepted Optimizations

### Shared

- Exact-zero `legacy_zero_array` materials now stage only `Vfrac` on both
  maintained execution paths in both supported residency modes:
  - `resident_mode='host'`
  - `resident_mode='device'`
- The device-resident runtime zero-field shortcut is a runtime-view
  optimization only:
  - authoritative `Material.S/theta/psi` fields remain concrete device arrays,
  - the shortcut does not rewrite the material contract to the explicit
    isotropic enum.
- Host-resident float32 anisotropic materials now use standard GPU reusable
  staging in `A2` on both maintained execution paths:
  - stage raw `Vfrac`, `S`, `theta`, `psi`,
  - build `phi_a`, `sx`, `sy`, `sz` on GPU,
  - drop raw staged `S`, `theta`, `psi`,
  - keep `Vfrac + phi_a + sx + sy + sz` as the steady-state runtime layout.
- Detector / projection geometry caching remains part of the accepted runtime.

### `tensor_coeff`

- Detector-grid helper path is accepted for both:
  - general-angle projection families
  - aligned `x` / `y` families
- Float32 `Segment B` uses fused isotropic accumulation rather than
  materializing a full-volume `isotropic_term` temporary.

### `direct_polarization`

- Detector-plane direct projection is accepted in the maintained path.
- Detector projection kernels prefer `nvcc` when available and fall back to
  `nvrtc`.
- Constructor-time kernel preload is the accepted default for this path.
- Accepted direct-path defaults are:
  - `kernel_preload_stage='a1'`
  - `igor_shift_backend='nvcc'`
  - `direct_polarization_backend='nvrtc'`
- `Segment C` uses in-place cuFFT plus in-place Igor-order swap.
- Float32 direct-path isotropic work is fused into direct accumulation.
- The current float32 direct-path reusable staging kernel still keeps `Vfrac`
  for fused isotropic accumulation; anisotropic-only kernel splitting remains a
  separate open optimization rather than part of the accepted state.

## Fast Approximation State

- `z_collapse_mode='mean'` exists on both maintained execution paths.
- It remains expert-only rather than a generally recommended default.
- `z_collapse_mode='mean'` does not support the half-input mixed-precision path.
- Approximation-specific details live in `fast_approximations.md`.

## Current Caveats That Still Matter

- `tensor_coeff` remains the maintained default for multi-angle work.
- Cross-backend GPU-memory claims should use host-resident `cupy-rsoxs` as the
  comparison authority.
- Old long-form optimization notes have been archived; do not open them unless
  the compact docs fail to answer a specific historical question.
