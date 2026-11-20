# CPU replay on the registered detector grid

Rerun on 2026-11-20 using the numerical lattice routines from `3c4c14d51ecb7ff63e47968b2a323f140f4937ef` and the `fft_bins_v1` reciprocal coordinates. These are scalar, normal-incidence CPU results in double precision. They quantify axial sampling error and provide targets for the GPU experiment.

## Finite-film high-q intensity

Both specimens were 128 × 1024 × 1024 at 1 nm per voxel, with 11 nm spheres, supersampling 2, azimuth 27°, tilts (9°,5°), no window and x-polarized illumination. The HCP raster contained 1,847 centers and BCC contained 8,799. At each physical detector pixel the independent reference sums **complex amplitudes** over the occupied planes at the continuous Ewald qz and applies the transverse projection. The gridded calculation uses ordinary axial FFTs of the selected XY columns, projects at the two endpoints, and interpolates their **intensities**. A direct full XYZ sum checks one pixel per specimen, energy and band.

The table gives relative L2 error of detector intensities against the continuous reference. The 570 eV outer band contains 179,416 valid pixels. Each common band contains 55,104. Strong peaks have the greatest weight in this metric.

| Finite film | Energy | q⊥ band (nm⁻¹) | M=128 | M=256 | M=512 | M=1024 |
|---|---:|---:|---:|---:|---:|---:|
| HCP | 570 eV | 1.90–2.40 | 32.57% | 11.72% | 3.00% | **0.67%** |
| BCC | 570 eV | 1.90–2.40 | 36.30% | 11.98% | 3.30% | **0.72%** |
| HCP | 285 eV | 0.95–1.25 | 43.69% | 11.80% | 3.48% | 0.80% |
| BCC | 285 eV | 0.95–1.25 | 44.56% | 13.96% | 2.07% | 0.65% |
| HCP | 570 eV | 0.95–1.25 | 38.86% | 10.88% | 2.86% | 0.78% |
| BCC | 570 eV | 0.95–1.25 | 35.16% | 10.48% | 3.27% | 0.71% |

The original November 19 pilot compared four coordinate conventions. Its FFT-bin arm is the one applicable to this parent. This replay uses that registered convention throughout and evaluates the same fixed films at each M. Per-reflection peak claims, full NRSS tensor panels, wall time and peak VRAM are measured in ZP02–03. [1024-wide results](cpu_pilot_1024.json) · [192-wide results](cpu_pilot.json) · [Scalar replay](../../../tests/validation/dev/cupy_rsoxs_virtual_zpad/lattice_probe.py)

## Independent Fourier and signed-grating checks

The [NumPy check](../../../tests/validation/dev/cupy_rsoxs_virtual_zpad/check_fft.py) compares all native complex coefficients on small arrays against literal vacuum padding. In 72 comparisons the worst relative L2 errors were 6.46 × 10⁻¹⁵ for selected axial sums and 6.02 × 10⁻¹⁵ for the residue-FFT method. The largest absolute DC-neighbor mean discrepancy divided by the largest Fourier amplitude was 9.41 × 10⁻¹⁶.

A further 12 signed grating cases check orders ±64, two physical axial depths and three logical depths. At half of a bin past the bright native order, `M=128` gives normalized intensity 0.500000 on that order and zero on its opposite. For `M=256` and `1024` the same pixel samples an exact padded bin, giving 0.40530508 and 0.01623175. Both axial evaluators agree with the dense complex coefficients and the grating powers agree with the finite-sum predictions. These cases would detect an implementation that quietly returned the native-depth image for a padding request.

Run the commands in [README_FIRST.md](README_FIRST.md) to regenerate these results. The packaged CPU replay used NumPy 2.3.5, SciPy 1.17.0 and the checkout’s numerical geometry and rasterization functions. The installed-NRSS command imports those functions through the regular validation module. The replay in this workspace supplied placeholders for the module’s unused simulation-type imports. No CUDA execution was involved.
