# Numerical evidence at the reviewed revision

## Native FFT registration

An even, centered transform of length N has one sampled origin at displayed index N/2. With voxel size d its coordinates run from −π/d up to π/d − 2π/(Nd). The two Nyquist endpoints alias. A detector whose beam geometrically straddles four pixels is also meaningful and needs its own pixel-to-q sampling. The present NRSS array reads one Fourier column per output XY index and exports the shifted FFT coordinates.

At the pinned revision, Igor reordering places native DC at the displayed integer origin. The historical inclusive Ewald axes assign zero to the point between the two central indices of each even axis. The established complex-amplitude replacement of the native DC coefficient is present in **both** of the plotted calculations. The off-axis grating selects a different lateral column.

| Unpadded grating at 314.803625 eV | Inclusive XYZ | Only axial registration corrected | FFT bins XYZ | Exact mode |
| --- | ---: | ---: | ---: | ---: |
| Lateral order +64, matched negative axial order | 0.489994 | 0.982040 | 0.999996 | 1 |
| Lateral order −64, mirrored grating with matched negative axial order | 0.521561 | 0.986145 | 0.999996 | 1 |

These are normalized single-mode intensities. A NumPy execution used the branch's geometry, Igor orders, complex replacement and endpoint projector. Supplying physical bin coordinates tests the proposed convention without requiring a changed CUDA build. In the figure the scan varies k at fixed lateral order +64, so the horizontal coordinate is the true Ewald depth. The black curve is the finite slab's continuous Fourier intensity. The orange curve interpolates intensities between unpadded FFT planes. They coincide at exact grating modes and differ between modes.

## CoreShell on the existing experimental q axis

The CPU calculation uses the branch's fixed CoreShell field construction and three optical-constant tables. The unpadded shape is 32 × 512 × 512, with 2.5 nm voxels. It forms the independent real tensor basis fields in 3D, uses the existing six-neighbor DC prescription, projects the two transverse polarization bases at the two bounding axial planes, and interpolates the endpoint intensities. Both arms use the same 361 polarization/rotation samples and the historical WPIntegrator horizontal and vertical sectors. The q coordinate shown is the historical reduced detector-plane q_perp. No model parameters were fit.

All 101 energies and 77 retained radii were evaluated, covering 0.0245463–0.3976503 nm⁻¹ in the requested 0.02–0.40 interval. The legacy replay reproduced the frozen A-wedge simulation within 0.00001462 everywhere in this window. At the two anchor energies an independently evaluated angle-by-angle detector rotation and polar remesh reproduced the fast composition's metrics within 1e−12. In this particular window the two wedges have exactly zero weight on the central XY pixel.

| Observable | Corrected vs frozen sim, max / RMSE | Sim limits | Corrected vs measured, max / RMSE | Measurement limits |
| --- | --- | --- | --- | --- |
| Mean A versus energy | 0.001704 / 0.000400 | 0.0025 / 0.0010 | 0.021510 / 0.013522 | 0.030 / 0.020 |
| A(q), 284.7 eV | 0.005995 / 0.001478 | 0.008 / 0.003 | 0.031294 / 0.014045 | 0.050 / 0.020 |
| A(q), 285.2 eV | 0.022243 / 0.005004 | 0.008 / 0.003 | 0.060936 / 0.027896 | 0.055 / 0.030 |

Four of 77 corrected points at 285.2 eV exceed the experimental maximum absolute error, at q_perp of 0.382923, 0.387832, 0.392741 and 0.397650 nm⁻¹. Seven exceed the old simulation limit, starting at 0.368195 nm⁻¹. At the largest radius, measured A is −0.117232, the legacy replay gives −0.155922 and FFT bins give −0.178168. The 285.2 eV legacy maximum error against experiment is 0.041995, whereas at 284.7 eV the correction improves the maximum error from 0.037289 to 0.031294. The plotted experimental points and the simulation regression are different evidence with different tolerances.

The live Euler contrast annulus on a 128 × 128 detector (2 nm voxels) changes from 5,108 to 5,188 selected pixels. 107 enter and 27 leave. Preserve its independent scaling expectations when migrating its mask. These are counts of geometric membership before image-specific finite masks.

The official three-path CUDA report, full MWCNT panel and corrected production golden are local-agent deliverables. The supplied grating and CPU replay quantify the issue and assess the migration.

## Reference identities

| File | SHA-256 |
| --- | --- |
| CoreShell experimental HDF5 | `b13449fdc84cac37233c44c806569503d98147ec25cdea90654f8d127ec91a5d` |
| CoreShell original experimental netCDF | `3747803bc1157be8f890b5e90fceb8c759511a223e554d1853e2a58245fd04b3` |
| CoreShell historical simulation | `d7df94ffd396d212b584c66bb66f7b481ff6e40a53b2dcf693e9ddcb690e272b` |
| MWCNT experimental HDF5 | `72cc62d3169a09264ed3b82239bb80d8de14ef9a84870f86e0a1c478e111255a` |

## Pinned source

- [CuPy detector geometry and interpolation](https://github.com/usnistgov/NRSS/blob/eb7d7bf1e6d97a383ded0cef29c0d0acf2ab646c/src/NRSS/backends/cupy_rsoxs.py#L4778-L4881)
- [CuPy native Fourier DC replacement](https://github.com/usnistgov/NRSS/blob/eb7d7bf1e6d97a383ded0cef29c0d0acf2ab646c/src/NRSS/backends/cupy_rsoxs.py#L4355-L4361)
- [CuPy compact real-FFT basis](https://github.com/usnistgov/NRSS/blob/eb7d7bf1e6d97a383ded0cef29c0d0acf2ab646c/src/NRSS/backends/_cupy_energy_basis.py#L198-L215)
- [CyRSoXS native Fourier DC replacement](https://github.com/usnistgov/cyrsoxs/blob/9bc714aeb678f26e5b3074e2696a711fe33ac1ee/src/cudaMain.cu#L123-L175) and [Ewald lookup](https://github.com/usnistgov/cyrsoxs/blob/9bc714aeb678f26e5b3074e2696a711fe33ac1ee/include/uniaxial.h#L835-L869)
- [CoreShell reference tests](https://github.com/usnistgov/NRSS/blob/eb7d7bf1e6d97a383ded0cef29c0d0acf2ab646c/tests/validation/test_core_shell_reference.py#L78-L185) and [thresholds](https://github.com/usnistgov/NRSS/blob/eb7d7bf1e6d97a383ded0cef29c0d0acf2ab646c/tests/validation/lib/core_shell.py#L59-L75)
- [Euler contrast test annular mask](https://github.com/usnistgov/NRSS/blob/eb7d7bf1e6d97a383ded0cef29c0d0acf2ab646c/tests/validation/test_sphere_orientational_contrast_scaling.py#L209-L245)
- [WPIntegrator polar reduction](https://github.com/usnistgov/PyHyperScattering/blob/caeb18e6320998bfe3df558dd507ad6a9eb445d0/src/PyHyperScattering/WPIntegrator.py)

## Bragg peak comparison

[BRAGG_RESULTS.md](BRAGG_RESULTS.md) and [RG01B](RG01B-bragg.md) cover the existing 3D lattice suite, the integrated off-axis registration assertion, full-size CPU evidence and the required public GPU comparison.
