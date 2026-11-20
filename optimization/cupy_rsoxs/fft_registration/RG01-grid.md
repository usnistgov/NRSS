# RG01 · Register the native FFT bins

**Deliverable.** One native-bin geometry for both unpadded CuPy paths, verified at the public detector output on known Fourier modes.

1. Record the original branch SHA and its full panel report (an archived report at that SHA and environment is sufficient). Run the supplied geometry and grating tests on the baseline. Capture the old full-size panels following RG01B. Save the failures and the already-passing amplitude replacement checks.
2. Trace `_detector_geometry` and `_detector_projection_geometry` in `src/NRSS/backends/cupy_rsoxs.py`, including their full-volume, selected-slice, multi-energy compact and fallback consumers in `_cupy_energy_basis.py`. Build physical axes as `(arange(N) - N//2) * (2π/(N*d))`. Preserve the existing Igor Fourier-index reversal and the established complex DC replacement. The public qx and qy already use shifted FFT frequencies. Check projector inputs and labels together. Version any cache or serialized metadata which could mix old and corrected geometries.
3. Use the forward root `qz = -q_perp²/(k + sqrt(k² - q_perp²))` on its physical domain. Locate it at `Nz//2 + qz/Δqz`. Preserve endpoint-intensity interpolation and the single-plane convention at `Nz=1`. An exact hit at the last valid bin can be handled with one neighbor. In particular `Nz=2` must retain the direct beam at the upper bin. Mask truly unsupported samples before clipping. Add tests for lower/upper endpoint hits and out-of-support queries.
4. Verify full complex and compact real-FFT basis outputs including the existing DC treatment. A real cosine fixes axial registration but cannot detect a global Fourier sign reversal. If changing the Igor mapping, include a signed direct-DFT comparison using two contrasts with different complex phases in an asymmetric arrangement. Inspect the last lateral row/column mask and a marked pixel rotated by zero and 90 degrees on odd, even and rectangular panels. Classify any rotation-pivot or border effect independently before changing that policy.

```bash
python -m pytest -q tests/smoke/test_ewald_fft_grid.py
grating=tests/validation/test_bragg_3d_lattice.py::test_bragg_off_axis_registration
python -m pytest -q tests/validation/test_ewald_fft_registration.py "$grating" --nrss-path cupy_tensor_coeff
python -m pytest -q tests/validation/test_ewald_fft_registration.py "$grating" --nrss-path cupy_direct_polarization
```

**Acceptance.** Both CuPy routes recover the absolute grating intensity at the known Fourier mode and the expected interpolation at a half step. The single-plane and two-plane cases have valid forward pixels. Record native indices, weights, mask and actual route. Preserve the observed value of the treated DC coefficient and the other coefficients in its plane. Report any remaining boundary or sign discrepancy before qualifying the full-size peaks in RG01B and migrating the CoreShell golden in RG02.

For the supplied 128 × 1024 × 9 grating, a 1 nm voxel and lateral order 64 produce an exact axial hit at 314.803625 eV. The normalized intensity there is 1. Halfway to the next empty bin the existing linear-intensity interpolator gives 0.5. The continuous finite grating gives about 0.405 there. Virtual padding will permit a separate test of that interpolation error.
