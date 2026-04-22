# Memory Footprint remaining options

Readout
  The device-resident exact-zero `legacy_zero_array` runtime shortcut has now landed, so device runs no longer need to keep zero `S/theta/psi` runtime staging alive just to reach the isotropic compute branches. That item is no longer part of the remaining-options list.

  The latest medium report makes the remaining problem pretty specific. Rotation growth is basically a tensor_coeff problem, not a direct_polarization problem: host-hot tensor_coeff rises 6574 -> 8110 MiB at 0:5:165, and device-hot rises 7500 -> 9036 MiB, while direct_polarization stays flat with rotation at 4014 MiB host and 4824 MiB device. The only
  measured 1 -> 3 energy growth in this matrix is device-only and is +512 MiB on both paths. See dev/NRSS/test-reports/core-shell-backend-performance-dev/comprehensive_medium/comprehensive_backend_comparison_report.md:47.

  These benchmark lanes are still using isotropic_representation=legacy_zero_array, not an explicit isotropic contract, for both the host and device tensor cases. See dev/NRSS/test-reports/core-shell-backend-performance-dev/comprehensive_medium/comprehensive_backend_comparison_summary.json:106 and dev/NRSS/test-reports/core-shell-backend-performance-dev/
  comprehensive_medium/comprehensive_backend_comparison_summary.json:1073.

  Why

  - cupy-rsoxs keeps staged runtime_materials alive across the whole run and allocates a device result tensor cp.empty((len(energies), ...)), so it has an explicit GPU energy term that cyrsoxs does not. See dev/NRSS/src/NRSS/backends/cupy_rsoxs.py:283 and dev/cyrsoxs/include/PyClass/ScatteringPattern.h:43.
  - tensor_coeff always allocates Nt as 5 full complex64 3D volumes, then materializes full-size basis_x and basis_y copies from fft_nt * scalar. At medium size, one float32 3D volume is 256 MiB and one complex64 3D volume is 512 MiB, so Nt alone is about 2560 MiB, basis_x is 1536 MiB, and basis_y adds another 1536 MiB. That extra 1536 MiB matches the
  observed rotation step almost exactly. See dev/NRSS/src/NRSS/backends/cupy_rsoxs.py:1195 and dev/NRSS/src/NRSS/backends/cupy_rsoxs.py:1889.
  - direct_polarization is flatter because it only allocates p_x/p_y/p_z, processes one angle at a time, projects to 2D, and immediately accumulates. There is no basis_x/basis_y analogue. See dev/NRSS/src/NRSS/backends/cupy_rsoxs.py:2081 and dev/NRSS/src/NRSS/backends/cupy_rsoxs.py:2179.
  - The closest C++ template is CyRSoXS MemoryMinizing: keep d_Nt, stream morphology in per energy, free voxel staging before allocating polarization/projection work buffers, and keep final results on host. See dev/cyrsoxs/include/Datatypes.h:128, dev/cyrsoxs/src/pymain.cpp:105, and dev/cyrsoxs/src/cudaMain.cu:1199.

# Possible Experiments

  1. Add a result_residency='host' or stream_results_to_host mode. For return_xarray=True, allocate the final (energy, qy, qx) buffer on CPU and copy each 2D panel off device per energy. This is the lowest-risk way to make GPU memory truly energy-flat. For medium size, a full 101-energy panel would otherwise be about 404 MiB on device just from the explicit
  result buffer. See dev/NRSS/src/NRSS/backends/cupy_rsoxs.py:302 and dev/cyrsoxs/include/PyClass/ScatteringPattern.h:43.
  2. Add a tensor_coeff_low_memory projection mode that avoids full-volume basis_x/basis_y. The right experiment is a fused detector-projection kernel that reads fft_nt[0..4] directly and writes the 2D projection, instead of first materializing 3D basis triplets. This is the main rotation-fix experiment; it should remove the extra 1536 MiB general-angle step
  and likely lower the no-rotation peak too. See dev/NRSS/src/NRSS/backends/cupy_rsoxs.py:1875.
  3. Add a cyrsoxs-style memory_mode='low' for host residency. Stage and accumulate per energy, then release staged morphology arrays before FFT/projection. If needed, go one material at a time. This targets the current host staging footprint, about 1792 MiB on medium, at a predictable transfer/speed cost. See dev/NRSS/src/NRSS/backends/cupy_rsoxs.py:744 and
  dev/cyrsoxs/src/cudaMain.cu:1231.
  4. Diagnose the shared device-only +512 MiB energy step with FFT-plan instrumentation. The explicit device result buffer is too small to explain that jump in the 1 -> 3 energy report, so this looks more like cuFFT plan/workspace or pool high-water than live result data. Run A/B with explicit plan creation, plan-cache clear between energies, and pool
  metrics after energy 1 vs 2.
  5. Do not spend more time first on allocator tricks or Segment C-only in-place work. The tensor in-place Segment C recheck already failed to lower peak memory, and pool-off only cut about 56 MiB while slowing the direct path by about 2.2x to 2.4x. See dev/NRSS/tests/validation/dev/core_shell_backend_performance/README.md:116 and dev/NRSS/optimization/
  cupy_rsoxs/archive/CUPY_RSOXS_DIRECT_POLARIZATION_OPTIMIZATION.md:2249.

  If I were doing this in order, I would implement host result streaming first, then a fused low-memory tensor_coeff detector projection mode, then the host-side low-memory staging mode. Those directly target the remaining measured pathologies without depending on allocator behavior.
