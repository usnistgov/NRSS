# Where the cutoff saves work

The published native compact path constructs a dense demand bitmap, transforms each real geometry field once with `rfftn`, gathers its samples into a reusable basis, and projects each requested energy. The virtual Z and XYZ paths collect a sparse union before their axial calculations. Native can retain the exact full FFT while reducing the gather, optical contraction, angle rotation, output buffers and host copies. Z sum and XYZ can additionally evaluate fewer axial coefficients. The residue path still transforms each occupied residue class unless all of its demanded entries disappear. Preserve the actual local ZP15 winners when running these comparisons. [Compact basis](https://github.com/usnistgov/NRSS/blob/bdf69e111a0e00d0adb2241067d949648d01e552/src/NRSS/backends/_cupy_energy_basis.py#L170-L275) · [Z evaluators](https://github.com/usnistgov/NRSS/blob/bdf69e111a0e00d0adb2241067d949648d01e552/src/NRSS/backends/_cupy_virtual_z.py#L89-L166) · [XYZ evaluator](https://github.com/usnistgov/NRSS/blob/bdf69e111a0e00d0adb2241067d949648d01e552/src/NRSS/backends/_cupy_virtual_xyz.py#L57-L102)

The public runtime currently builds the basis before obtaining the reduction plan. `CupyScatteringResult.to_xarray` constructs qx/qy using the full logical shape. A smaller detector buffer needs an explicit global index offset and axes sliced from the full logical grid. The compact angle kernel uses a bilinear inverse mapping. Reduced layouts sample the rotated Cartesian image on 360 rays and can use another interpolation for shared q. [Run order](https://github.com/usnistgov/NRSS/blob/bdf69e111a0e00d0adb2241067d949648d01e552/src/NRSS/backends/cupy_rsoxs.py#L581-L710) · [Raw detector axes](https://github.com/usnistgov/NRSS/blob/bdf69e111a0e00d0adb2241067d949648d01e552/src/NRSS/backends/cupy_rsoxs.py#L386-L418) · [Radial plan](https://github.com/usnistgov/NRSS/blob/bdf69e111a0e00d0adb2241067d949648d01e552/src/NRSS/backends/cupy_rsoxs.py#L952-L1050) · [Compact rotation](https://github.com/usnistgov/NRSS/blob/bdf69e111a0e00d0adb2241067d949648d01e552/src/NRSS/backends/_cupy_compact_rotation.py)

## A geometry-only sizing example

On a 256 × 1024 × 1024 grid with 1.5 nm voxels, 101 energies from 282.55 to 292.55 eV, and a physical cutoff of 0.8 nm⁻¹, the following counts apply to any material on that logical grid. The four-pixel private margin is conservative for this square example. It is separate from the returned detector. The calculated output bytes refer to one FP32 intensity panel per energy. Process RSS and VRAM include other allocations and must be measured independently.

| Quantity | Unbounded | Limited |
|---|---:|---:|
| Returned rectangular detector | 1024 × 1024 | 377 × 377 |
| Host or device output payload for 101 FP32 frames | 404.0 MiB | 54.8 MiB |
| Valid pixel visits summed over energies | 40,270,985 | 11,702,929 with margin |
| Unique logical Fourier endpoints including DC repair | 2,017,475 | 263,519 with margin |
| Native common radial bins, if a reduced output was requested | 725 | 297 |

In this case 193 of 513 native half-spectrum kx columns and 58,291 of 525,312 (ky,kx) columns are demanded with the same margin. At 287.55 eV, |q| = 0.8 nm⁻¹ corresponds to transverse q ≈ 0.76927 nm⁻¹. The full endpoint count reproduces the earlier GPU ledger. [Machine-readable CPU results](cpu-evidence.json) · [Prior measurement geometry](https://github.com/usnistgov/NRSS/blob/bdf69e111a0e00d0adb2241067d949648d01e552/optimization/cupy_rsoxs/zpad_tickets/ZP05-09-RESULTS.md#L113-L132)

Two small probes affect implementation choices. On a 77 × 169 rotated panel a four-pixel physical margin lost 115 of 189 originally finite azimuthal values at a boundary ring. Tracing actual neighbors preserved the complete result bitwise. In the separate NumPy complex128 native FFT experiment, the maximum error divided by peak was about 4 × 10⁻¹⁶, yet 31,539 of 42,840 coefficients and mixtures had different bits in this environment. This illustrates why a tolerance cannot qualify an alternate FFT. These examples are CPU calculations. The packet supplies GPU timing and memory workers for the local agent.
