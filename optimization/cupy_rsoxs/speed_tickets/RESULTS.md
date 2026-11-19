# Decisions and measured results

S00 passed on Wheatley. The isolated baseline is `c29129a`, and each run plan records the source hash of the CuPy backend for each arm.

| Ticket / variant | Incumbent → candidate revision/options | Workload, GPU, result layout | Median primary A/B (s), speedup, favorable pairs | Sampled peak A/B (MiB) | Accuracy and report location | Decision / next action |
|---|---|---|---|---|---|---|
| S00 A/A, one direction | `c29129a` → same | Small, direct polarization, device input, host detector, Wheatley RTX A6000 | 0.01908 / 0.01853, 1.033×, 3/3 favorable | | [A/A summary](../../../test-reports/cupy-speed-tickets/S00-wheatley-LeghgmlO/aa-one/20261118T194013Z-one-zxemb9ak/summary.json) | screened, 3.26% median absolute paired deviation |
| S00 A/A, 34 directions | `c29129a` → same | Small, tensor coefficients, device input, host detector, Wheatley RTX A6000 | 0.03629 / 0.03613, 0.991×, 1/3 favorable | | [A/A summary](../../../test-reports/cupy-speed-tickets/S00-wheatley-LeghgmlO/aa-angles/20261118T194039Z-angles-phij4bdz/summary.json) | screened, 1.63% median absolute paired deviation |
| S00 instrumentation, small | `c29129a` → S00 | 20 energies, tensor coefficients, host input and detector, Wheatley RTX A6000 | 0.30225 / 0.30339, 0.996×, 1/3 favorable | | [Primary](../../../test-reports/cupy-speed-tickets/S00-wheatley-LeghgmlO/small-primary/20261118T194100Z-energy20-q6o3kn4u/summary.json) · [Stages](../../../test-reports/cupy-speed-tickets/S00-wheatley-LeghgmlO/small-stages/20261118T194122Z-energy20-y0x8dkj5/r01-A.result.json) | screened, 100 complex FFTs in 20 single-energy tiles |
| S00 instrumentation, medium | `c29129a` → S00 | One energy and direction, tensor coefficients, host input and detector, Wheatley RTX A6000 | 0.51638 / 0.51424, 1.004×, 2/3 favorable | | [Primary](../../../test-reports/cupy-speed-tickets/S00-wheatley-LeghgmlO/medium-primary/20261118T194126Z-depth-y2xi7mg0/summary.json) · [Stages](../../../test-reports/cupy-speed-tickets/S00-wheatley-LeghgmlO/medium-stages/20261118T194237Z-depth-4ilf6ev1/r01-A.result.json) | screened, three complex FFTs in one tile |
| S00 memory observer | `c29129a` → S00 | Small, one energy and direction, tensor coefficients, host input and detector, Wheatley RTX A6000 | | 1079.3 / 1079.3 MiB peak delta | [Memory pass](../../../test-reports/cupy-speed-tickets/S00-wheatley-LeghgmlO/memory/20261118T194300Z-one-k11x2gdr/summary.json) | screened, 10 ms sample interval and 48,266 MiB free at launch for both arms |
| S02 real FFT, medium | `c29129a` → `compact_r2c` | 20 energies, 12 directions, host input and detector, Wheatley RTX A6000 | 1.942 / 0.634, 3.06×, 5/5 favorable | 8215.3 / 2849.3 MiB | [Primary](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/medium-20-speed/20261118T194848Z-energy20-4owfzbwn/summary.json) · [Accuracy](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/medium-20-accuracy/20261118T195111Z-energy20-0b7v86n4/summary.json) · [Memory](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/medium-20-memory/20261118T195048Z-energy20-oxa5k1dt/summary.json) | screened, 100 → 9 FFTs |
| S02 real FFT, large | `c29129a` → `compact_r2c` | 20 energies, 12 directions, host input and detector, Wheatley RTX A6000 | 6.363 / 1.802, 3.53×, 5/5 favorable | 27063.3 / 9793.3 MiB | [Primary](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/large-20-speed/20261118T195223Z-energy20-i9_ggv89/summary.json) · [Accuracy](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/large-20-accuracy/20261118T195817Z-energy20-z271xyr1/summary.json) · [Memory](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/large-20-memory/20261118T195720Z-energy20-strlf6an/summary.json) | screened, 100 → 9 FFTs |
| S02 real FFT, medium | `c29129a` → `compact_r2c` | 101 energies, 12 directions, host input and detector, Wheatley RTX A6000 | 7.818 / 1.306, 5.98×, 3/3 favorable | | [Primary](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/medium-101-speed/20261118T200640Z-energy101-sbp3_ww_/summary.json) · [Accuracy](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/medium-101-accuracy/20261118T200821Z-energy101-9mhje605/summary.json) | screened |
| S02 real FFT, large | `c29129a` → `compact_r2c` | 101 energies, 12 directions, host input and detector, Wheatley RTX A6000 | 25.470 / 2.792, 9.12×, 3/3 favorable | 27065.3 / 9795.3 MiB | [Primary](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/large-101-speed/20261118T200849Z-energy101-48erwawa/summary.json) · [Accuracy](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/large-101-accuracy/20261118T201331Z-energy101-zc3zorns/summary.json) · [Memory](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/large-101-memory/20261118T201450Z-energy101-l0yyjit6/summary.json) | screened |
| S02 real FFT, medium I/A | `c29129a` → `compact_r2c` | 20 energies, 12 directions, host input and I/A results, Wheatley RTX A6000 | 2.658 / 1.362, 1.95×, 3/3 favorable | | [Primary](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/medium-20-ia/20261118T200355Z-energy20-_m6sxljv/summary.json) · [Accuracy](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/medium-20-ia-accuracy/20261118T200509Z-energy20-1meu8b14/summary.json) | screened |

Use `screened`, `rejected`, `inconclusive`, `selected` or `push-ready`. Record the selector/fallback and actual FFT counts for energy and detector variants. Append one row for the combined implementation at S06. At S07 attach the revision and full report path, including candidate-enabled supplemental panels. Keep large outputs under the ignored `test-reports/` tree.

## S00 stage split

Both tensor cases used float32 morphology and complex64 FFTs on an RTX A6000 with NVIDIA driver 580.178.04, CUDA runtime 12.9, cuFFT 11.4.1 and CuPy 14.0.1. The stage passes recorded no fallback and zero basis builds. The small scan used 242,028 valid columns and 562,540 distinct detector endpoints across 17 z planes. The medium case used 271,464 columns and 542,928 endpoints across 20 planes. [Run log](../../../test-reports/cupy-speed-tickets/S00-wheatley-LeghgmlO/gate.log)

| Workload | Primary B median (ms) | A2 (ms) | B (ms) | C+D (ms) | E (ms) | D2H (ms) | Host copy waits (ms) | F (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small, 20 energies | 303.39 | 51.45 | 54.47 | 135.00 | 61.92 | 1.94 | 0.12 | 0.02 |
| Medium, one energy | 514.24 | 465.26 | 14.85 | 32.55 | 0.02 | 0.33 | 18.65 | 0.02 |

C+D accounted for about 44.5% and 6.3% of the matched primary times. Eliminating that entire range gives idealized speed ceilings of 1.80× and 1.07×, respectively. The medium wait overlaps device work. Host staging (A2) accounted for about 90% of the medium single-energy primary time.

## S00 accuracy and rejection checks

The generic fixture matched exactly across the full three-energy detector output and the screened 20-energy I/A output, including coordinates and finite/NaN masks. The I/A comparison exercised host result chunking, windowing and rotation masking. The three-energy detector fixture included a weak-contrast panel with peak intensity about 3.73 × 10⁻⁹. The full comparison of sector intensities with device-resident results also matched exactly. [Detector accuracy](../../../test-reports/cupy-speed-tickets/S00-wheatley-LeghgmlO/detector-accuracy/20261118T194248Z-three-8xsd02jr/summary.json) · [I/A accuracy](../../../test-reports/cupy-speed-tickets/S00-wheatley-LeghgmlO/ia-accuracy/20261118T194253Z-energy20-xxct0lon/summary.json) · [Device sector accuracy](../../../test-reports/cupy-speed-tickets/S00-wheatley-LeghgmlO/device-sector-accuracy.log)

The launcher rejected both an intentionally wrong route expectation and a 1% intensity perturbation applied to the first detector panel in a temporary checkout. A separate geometry probe recorded material shape 1 × 33 × 35 and detector demand on a requested transform grid of 9 × 33 × 35. [Route rejection](../../../test-reports/cupy-speed-tickets/S00-wheatley-LeghgmlO/rejected-route.log) · [Intensity rejection](../../../test-reports/cupy-speed-tickets/S00-wheatley-LeghgmlO/rejected-output-forced/20261118T194355Z-three-uwd_o44r/summary.json) · [Depth probe](../../../test-reports/cupy-speed-tickets/S00-wheatley-LeghgmlO/material-depth/20261118T194310Z-depth-8zd717e_/r01-A.result.json)

The CPU smoke lane passed 82 tests with three skips. Six focused GPU smoke tests passed on Wheatley, covering timing opt-in, host staging, device results and the union q grid. [GPU smoke log](../../../test-reports/cupy-speed-tickets/S00-wheatley-LeghgmlO/gpu-focused-smoke.log)

## S02 compact energy basis

The 20-energy runs retained nine basis fields at 710,222 medium endpoints (54.2 MiB) and 711,368 large endpoints (54.3 MiB). Basis build and gather took 62 ms and 185 ms in the separate diagnostic passes. The existing direct-polarization path took 10.19 s and 34.91 s in single primary runs of the same 12-direction workloads. The energy angles use `[0, 15, 165]` degrees in `[StartAngle, IncrementAngle, EndAngle]` order. [Medium stages](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/medium-20-stages/20261118T195134Z-energy20-q_se4aki/r01-B.result.json) · [Large stages](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/large-20-stages/20261118T195903Z-energy20-38xsz1ii/r01-B.result.json) · [Medium direct path](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/medium-20-direct/20261118T195154Z-energy20-ewr9uas3/summary.json) · [Large direct path](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/large-20-direct/20261118T195957Z-energy20-cbiow3pz/summary.json)

The medium complex FFT variant took 0.632 s for the 20-energy scan, within the timing spread of the real FFT variant's 0.634 s. The larger assessment used the real FFT variant. [Complex FFT timing](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/medium-20-c2c/20261118T200126Z-energy20-msw58_cs/summary.json)

For medium with one electric-field direction and 20 energies, the tensor and direct incumbents took 1.325 s and 1.291 s, respectively. The real FFT option took 0.512 s, about 2.52× faster than the faster incumbent. The small 3-, 20- and 101-energy lanes measured 1.06×, 1.39× and 1.57× paired speedups across three pairs each. [Medium one direction](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/medium-20-off/20261118T200224Z-energy20-756ngd4l/summary.json) · [Direct control](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/medium-20-off-direct/20261118T200319Z-energy20-fb7vhfe8/summary.json) · [Run log](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/assessment.log)

Both FFT variants passed the generic detector comparison with energies in nonmonotonic order, windowing and rotation masking. The real FFT also passed the unwindowed weak-contrast case with normalized L2 error 3.81 × 10⁻⁴ and maximum absolute intensity difference 1.83 × 10⁻¹² at 285 eV. Complex amplitudes at sampled endpoints and translation phase were checked at material depths 1 and 9 on Wheatley's RTX 8000. The single-energy control used the incumbent route and recorded its fallback reason. [Preflight](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/preflight.log) · [Weak contrast](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/small-accuracy-weak/20261118T194842Z-three-0qjeaw17/summary.json) · [Endpoint checks](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/endpoint-check.log)

The medium and large detector spot checks across 20 and 101 energies had worst normalized L2 error 2.07 × 10⁻⁷. Peak VRAM was sampled at 10 ms in separate passes with 48,266 MiB free at each launch. The large scan's peak grew by about 2 MiB from 20 to 101 energies in each arm. The S02 runs evaluated `compact_r2c` through its explicit option. [Run log](../../../test-reports/cupy-speed-tickets/S02-wheatley-JkL6FqfO/assessment.log)

## S07 energy-basis qualification

Qualified and promoted locally. The automatic route uses one real FFT basis for eligible FP32 scans with at least 20 distinct energies, 32 material planes, 8,388,608 voxels and fourfold estimated FFT-count reuse. Explicit direct polarization and the explicit `energy_basis_mode=None` setting use the per-energy route.

All maintained smoke and physics lanes and both forced Fourier variants passed with the expected case applicability. Full panel comparisons included every pixel and energy of the small, medium and large 101-energy scans.

- [baseline report](../../../test-reports/S07-energy-basis-EgYPVG20/baseline/20261118T220332Z/summary.md)
- [candidate report](../../../test-reports/S07-energy-basis-EgYPVG20/candidate/20261118T220345Z/summary.md)
- [compact_r2c report](../../../test-reports/S07-energy-basis-EgYPVG20/compact_r2c/20261118T223347Z/summary.md)
- [compact_c2c report](../../../test-reports/S07-energy-basis-EgYPVG20/compact_c2c/20261118T223745Z/summary.md)

| Workload | Incumbent (s) | Automatic (s) | Paired speedup |
|---|---:|---:|---:|
| [medium-20-host](../../../test-reports/S07-energy-basis-EgYPVG20/timing/medium-20-host/20261118T224438Z-energy20-hrcngn0t/summary.json) | 1.944 | 0.634 | 3.07× |
| [large-101-host](../../../test-reports/S07-energy-basis-EgYPVG20/timing/large-101-host/20261118T224627Z-energy101-8bhrb2d9/summary.json) | 25.449 | 2.785 | 9.14× |
| [medium-20-default](../../../test-reports/S07-energy-basis-EgYPVG20/timing/medium-20-default/20261118T225619Z-energy20-nhn_jrw_/summary.json) | 1.235 | 0.541 | 2.28× |

[Full qualification evidence](../../../test-reports/S07-energy-basis-EgYPVG20/gate-result.json)

## S07b direct-polarization energy basis

The `direct_r2c` screen uses one fixed electric-field direction on Wheatley's RTX A6000 with host input and host detector output. At 37°, `EAngleRotation` is `[37, 0, 37]` degrees in `[StartAngle, IncrementAngle, EndAngle]` order. The kernel mixes the complex polarization amplitudes, projects the two endpoints and interpolates intensity in one launch.

| Workload | Incumbent → candidate | Median A/B (s), paired speedup | Peak A/B (MiB) | Evidence |
|---|---|---|---|---|
| Medium, 20 energies at 37° | Per-energy direct → `direct_r2c` | 1.211 / 0.454, 2.67×, 3/3 favorable | 4119.3 / 2841.3 | [Primary](../../../test-reports/cupy-speed-tickets/S07b-wheatley-vYGncYJw/medium-direct-speed/20261119T004605Z-energy20-z82z8od3/summary.json) · [Memory](../../../test-reports/cupy-speed-tickets/S07b-wheatley-vYGncYJw/medium-direct-memory/20261119T004750Z-energy20-7gtujvy8/summary.json) |
| Medium, 20 energies at 37° | Tensor `compact_r2c` → `direct_r2c` | 0.523 / 0.450, 1.16×, 3/3 favorable | 2849.3 / 2841.3 | [Primary](../../../test-reports/cupy-speed-tickets/S07b-wheatley-vYGncYJw/medium-tensor-speed/20261119T004700Z-energy20-t986fx5_/summary.json) · [Memory](../../../test-reports/cupy-speed-tickets/S07b-wheatley-vYGncYJw/medium-tensor-memory/20261119T004812Z-energy20-5jvyk40v/summary.json) |
| Medium, 20 energies at 0° | Tensor `compact_r2c` → `direct_r2c` | | 2849.3 / 2841.3 | [Memory](../../../test-reports/cupy-speed-tickets/S07b-wheatley-vYGncYJw/medium-aligned-memory/20261119T004835Z-energy20-ej63vn9f/summary.json) |
| Large, 20 energies at 37° | Tensor `compact_r2c` → `direct_r2c` | 1.767 / 1.656, 1.07×, 3/3 favorable | 9793.3 / 9785.3 | [Primary](../../../test-reports/cupy-speed-tickets/S07b-wheatley-vYGncYJw/large-tensor-speed/20261119T005005Z-energy20-9toncjfk/summary.json) · [Memory](../../../test-reports/cupy-speed-tickets/S07b-wheatley-vYGncYJw/large-tensor-memory/20261119T005252Z-energy20-yagh19vw/summary.json) |
| Large, 20 energies at 37° | Per-energy direct → `direct_r2c` | | 13239.3 / 9785.3 | [Memory](../../../test-reports/cupy-speed-tickets/S07b-wheatley-vYGncYJw/large-direct-memory/20261119T005339Z-energy20-_dxib389/summary.json) |

The direct basis passed every-energy comparison for the medium detector and I/A cases, with worst normalized L2 errors of 3.48 × 10⁻⁷ and 4.59 × 10⁻⁷ respectively. The generic 37° check included nonmonotonic energies and produced a worst normalized L2 error of 5.39 × 10⁻⁴ at the weak-contrast energy. The single-plane window/mask case passed. The single-energy and multi-direction controls recorded per-energy direct polarization as the actual route. [Medium detector](../../../test-reports/cupy-speed-tickets/S07b-wheatley-vYGncYJw/medium-oblique-accuracy/20261119T004544Z-energy20-9d8bz7c4/summary.json) · [Medium I/A](../../../test-reports/cupy-speed-tickets/S07b-wheatley-vYGncYJw/medium-ia-accuracy/20261119T004855Z-energy20-7vi2r9vw/summary.json) · [Preflight](../../../test-reports/cupy-speed-tickets/S07b-wheatley-vYGncYJw/preflight.log)

The large comparison passed at the first, middle and last energies with worst normalized L2 error 1.63 × 10⁻⁷. The large 37° scan retained 711,368 endpoints and used seven FFTs in `direct_r2c` compared with nine for tensor reuse. Its cache dropped from 54.3 to 43.4 MiB. The sampled peak was 8 MiB below tensor reuse in each tested size. The large detector projection stage took 78 ms with tensor reuse and 13 ms with direct reuse. Each memory arm was sampled at 10 ms with 48,266 MiB free at launch. [Large detector](../../../test-reports/cupy-speed-tickets/S07b-wheatley-vYGncYJw/large-oblique-accuracy/20261119T004912Z-energy20-paw_cj6d/summary.json) · [Stage counts](../../../test-reports/cupy-speed-tickets/S07b-wheatley-vYGncYJw/large-oblique-stages/20261119T005443Z-energy20-smw5p5ac/summary.json)

**Decision.** Screened as an opt-in single-direction route. The CPU smoke lane passed 86 tests with three skips, and six focused GPU smoke tests passed. [GPU smoke log](../../../test-reports/cupy-speed-tickets/S07b-wheatley-vYGncYJw/gpu-smoke.log)

## Single-energy compact tensor screen

Measured one energy at 285 eV with host-resident CoreShell inputs and host detector output on Wheatley's RTX A6000. The trial checkout allowed `compact_r2c` to execute with one distinct energy. Its compact calculation used the same Fourier and projection code as the qualified multi-energy route. Memory passes used the 10 ms observer on the same GPU.

| Workload | Dense tensor median (s) | Compact tensor median (s) | Paired speedup | Peak VRAM dense tensor / per-energy direct / compact tensor (GiB) |
|---|---:|---:|---:|---:|
| Medium, 37° | 0.555 | 0.466 | 1.20× | 8.02 / 4.02 / 2.78 |
| Large, 37° | 1.976 | 1.678 | 1.18× | 26.43 / 12.93 / 9.56 |

The compact path's sampled peak matched the 20-energy tensor-reuse measurements at both sizes. The baseline at 0° on medium used 6.52 GiB for dense tensor and 4.02 GiB for per-energy direct. [Medium timing](../../../test-reports/cupy-speed-tickets/single-energy-memory-F1ZU0Zg6/medium-compact-oblique-speed/20261119T010334Z-one-4erpm8um/summary.json) · [Large timing](../../../test-reports/cupy-speed-tickets/single-energy-memory-F1ZU0Zg6/large-compact-oblique-speed/20261119T010537Z-one-9uuca4yr/summary.json) · [Medium peak](../../../test-reports/cupy-speed-tickets/single-energy-memory-F1ZU0Zg6/medium-compact-oblique-memory/20261119T010311Z-one-dgomldbw/summary.json) · [Large peak](../../../test-reports/cupy-speed-tickets/single-energy-memory-F1ZU0Zg6/large-compact-oblique-memory/20261119T010438Z-one-zhr75dwo/summary.json) · [Direct and aligned controls](../../../test-reports/cupy-speed-tickets/single-energy-memory-F1ZU0Zg6/dense.log)

The medium 12-direction scan used `EAngleRotation = [0, 15, 165]` degrees in `[StartAngle, IncrementAngle, EndAngle]` order. Compact tensor took 0.470 s versus 0.562 s for dense tensor, with peak VRAM of 2.78 versus 8.02 GiB. Its complete detector panel passed with normalized L2 error 2.99 × 10⁻⁸. For the medium single direction at 37°, per-energy direct and compact tensor took 0.468 and 0.473 s respectively. The diagnostic pass confirmed one compact basis build and nine FFTs. [Sweep timing](../../../test-reports/cupy-speed-tickets/single-energy-memory-F1ZU0Zg6/medium-compact-sweep-speed/20261119T010846Z-one-38ey6zde/summary.json) · [Sweep accuracy](../../../test-reports/cupy-speed-tickets/single-energy-memory-F1ZU0Zg6/medium-compact-sweep-accuracy/20261119T010950Z-one-ohi3guev/summary.json) · [Direct timing comparison](../../../test-reports/cupy-speed-tickets/single-energy-memory-F1ZU0Zg6/medium-direct-vs-compact/20261119T011033Z-one-5w0wagrw/summary.json) · [Actual route](../../../test-reports/cupy-speed-tickets/single-energy-memory-F1ZU0Zg6/medium-compact-stage/20261119T011011Z-one-e42gg2ip/r01-B.result.json)

**Decision.** Screened compact tensor for single-energy medium and large volumes. The sampled results support qualifying it as the tensor route at these sizes regardless of whether the workload has one energy or a longer scan.

## Compact tensor default at all energy counts

Promoted compact real FFT tensor for full-depth float32 CuPy simulations at all nonempty energy counts. The complete smoke and physics report passed, and the recorded runs exercised compact tensor at both one energy and multiple energies. The explicit direct path remains available.

Matched Wheatley RTX 8000 comparisons used direct polarization in A and the public compact default in B. The 12- and 34-direction cases use `EAngleRotation` values `[0, 15, 165]` and `[0, 5, 165]` in `[StartAngle, IncrementAngle, EndAngle]` order.

| Workload | Direct median (s) | Compact median (s) | Paired speedup |
|---|---:|---:|---:|
| [Small, 1 energy, 1 direction](../../../test-reports/S07-compact-default-7sq1vs72/comparisons/small-one-direction/20261119T011504Z-one-t2sdywdm/summary.json) | 0.020 | 0.030 | 0.67× |
| [Small, 1 energy, 34 directions](../../../test-reports/S07-compact-default-7sq1vs72/comparisons/small-34-direction/20261119T011540Z-angles-n_g6bi4h/summary.json) | 0.238 | 0.044 | 5.42× |
| [Medium, 1 energy, 12 directions](../../../test-reports/S07-compact-default-7sq1vs72/comparisons/medium-12-direction/20261119T011635Z-one-7j3cegde/summary.json) | 1.038 | 0.491 | 2.11× |
| [Large, 1 energy, 12 directions](../../../test-reports/S07-compact-default-7sq1vs72/comparisons/large-12-direction/20261119T011854Z-one-fk5ai_p7/summary.json) | 3.768 | 1.682 | 2.24× |
| [Medium, 20 energies, 12 directions](../../../test-reports/S07-compact-default-7sq1vs72/comparisons/medium-20-energy-12-direction/20261119T012256Z-energy20-drnhvc1a/summary.json) | 12.721 | 0.650 | 19.65× |

[Full report](../../../test-reports/S07-compact-default-7sq1vs72/validation/20261119T011547Z/summary.md) · [Qualification evidence](../../../test-reports/S07-compact-default-7sq1vs72/qualification.json)

## Compact tensor memory by material count

Matched small synthetic morphologies span one to eight materials with closed, nonuniform volume fractions and spatially varying Euler orientation fields. Both tensor paths used full-depth float32 host inputs and host detector output on the Wheatley RTX 8000. Each arm ran in a separate process with one warm-up and a fresh morphology. NVML sampled every 10 ms and found 48,231 MiB free at launch. Peak is the sampled increase over idle. The retained cache includes Fourier values, indices, and coefficients. The 20-energy scans cover 280 to 289.5 eV in 0.5 eV steps. Single-energy runs use 285 eV.

| Shape and voxel size | Anisotropic + isotropic | Energies | E direction | Endpoints / voxels | Dense peak (MiB) | Compact peak (MiB) | Retained basis (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32 × 256 × 256, 2.5 nm | 1 + 0 | 1 | 37° | 5.7% | 414 | 258 | 7.3 |
| 32 × 256 × 256, 2.5 nm | 2 + 0 | 1 | 37° | 5.7% | 454 | 312 | 13.6 |
| 32 × 256 × 256, 2.5 nm | 4 + 0 | 1 | 37° | 5.7% | 534 | 418 | 26.4 |
| 32 × 256 × 256, 2.5 nm | 6 + 0 | 1 | 37° | 5.7% | 614 | 518 | 39.1 |
| 32 × 256 × 256, 2.5 nm | 8 + 0 | 1 | 37° | 5.7% | 694 | 616 | 51.9 |
| 32 × 256 × 256, 2.5 nm | 2 + 6 | 1 | 37° | 5.7% | 490 | 368 | 19.1 |
| 32 × 256 × 256, 2.5 nm | 0 + 8 | 1 | 37° | 5.7% | 426 | 282 | 8.2 |
| 1 × 1024 × 1024, 5 nm | 1 + 0 | 1 | 37° | 99.8% | 362 | 428 | 63.9 |
| 1 × 1024 × 1024, 5 nm | 2 + 0 | 1 | 37° | 99.8% | 382 | 560 | 119.8 |
| 1 × 1024 × 1024, 5 nm | 4 + 0 | 1 | 37° | 99.8% | 422 | 824 | 231.5 |
| 1 × 1024 × 1024, 5 nm | 6 + 0 | 1 | 37° | 99.8% | 462 | 1088 | 343.3 |
| 1 × 1024 × 1024, 5 nm | 8 + 0 | 1 | 37° | 99.8% | 502 | 1352 | 455.1 |
| 1 × 1024 × 1024, 5 nm | 2 + 6 | 1 | 37° | 99.8% | 406 | 680 | 167.7 |
| 1 × 1024 × 1024, 5 nm | 0 + 8 | 1 | 37° | 99.8% | 374 | 456 | 71.9 |
| 1 × 1024 × 1024, 1.25 nm | 4 + 0 | 1 | 37° | 25.9% | 422 | 438 | 60.1 |
| 1 × 1024 × 1024, 1.25 nm | 8 + 0 | 1 | 37° | 25.9% | 502 | 634 | 118.1 |
| 32 × 256 × 256, 2.5 nm | 8 + 0 | 20 | 37° | 6.7% | 694 | 1118 | 61.0 |
| 1 × 1024 × 1024, 5 nm | 8 + 0 | 20 | 37° | 99.8% | 514 | 1354 | 455.2 |
| 1 × 1024 × 1024, 5 nm | 8 + 0 | 1 | 0° | 99.8% | 478 | 1080 | 327.4 |

All 19 pairs had matching finite detector masks and passed the full-panel comparison. The worst normalized L2 difference was 4.81e-07. Staging reports confirmed only `Vfrac` for enum isotropic materials, versus `Vfrac`, `S`, `theta`, and `psi` for the anisotropic fields. The compact basis used one, five, or seven rows per material, as predicted by material type and the active tensor components.

[Measurements](../../../test-reports/cupy-speed-tickets/material-count-memory-pq_fb7kc/results.json) · [Fixture and runner](../../../test-reports/cupy-speed-tickets/material-count-memory-pq_fb7kc/benchmark.py) · [User guide](../../../docs/source/user_guide/cupy_rsoxs_memory.rst)

### Allocation growth across the energy scan

The follow-up repeated the 32 × 256 × 256, 2.5 nm scan at 37° and recorded CuPy pool occupancy immediately after each compact projection. In the fixed-geometry control, 20 distinct energies were within 0.0000002 eV of 285 eV and produced the same float32 projection geometry. The trim control used the original energy list and released free blocks immediately after each projection, after synchronizing the device.

| Anisotropic + isotropic | Scan | Dense peak (MiB) | Compact peak (MiB) | Basis (MiB) | Compact pool total at first → last projection (MiB) |
|---:|---|---:|---:|---:|---:|
| 1 + 0 | 280–289.5 eV | 414 | 258 | 8.6 | 72 → 72 |
| 2 + 0 | 280–289.5 eV | 454 | 314 | 16.1 | 127 → 127 |
| 4 + 0 | 280–289.5 eV | 534 | 688 | 31.0 | 235 → 477 |
| 6 + 0 | 280–289.5 eV | 614 | 904 | 46.0 | 336 → 699 |
| 8 + 0 | 280–289.5 eV | 694 | 1118 | 61.0 | 437 → 921 |
| 2 + 6 | 280–289.5 eV | 490 | 592 | 22.5 | 210 → 383 |
| 0 + 8 | 280–289.5 eV | 426 | 316 | 9.6 | 125 → 125 |
| 8 + 0 | Fixed detector geometry | 694 | 616 | 51.9 | 428 → 428 |
| 8 + 0 | Pool trim control | 694 | 624 | 61.0 | 437 → 96 |

For eight anisotropic materials on the ordinary scan, CuPy pool memory in use just after projection was 62.1 MiB at the first energy and 62.3 MiB at the last, while reserved pool memory rose by 484 MiB. The fixed-geometry and trim controls isolated allocator reservation as the main source of the peak increase. In the trim control, pool reservation dropped to about 68 MiB after each trim, and the full-run sampled peak fell from 1118 to 624 MiB with the same 61 MiB retained basis. Pool trimming was used only in this diagnostic run.

All 9 follow-up pairs passed full-panel accuracy with worst normalized L2 error 7.49e-07. [Measurements and pool traces](../../../test-reports/cupy-speed-tickets/material-count-scan-lldrb8km/results.json) · [Fixture and runner](../../../test-reports/cupy-speed-tickets/material-count-scan-lldrb8km/benchmark.py)

## Compact-route screens for S03, S04, and S05

### S03 · Fused compact tensor projection

The CUDA projector contracts directly from the retained Fourier basis, then projects and interpolates the endpoint intensity. It removes the temporary matrix of gathered basis fields. The tiling prototype uses the same kernel at widths 1, 2, 4, and 8 and preserves output chunking.

| Matched workload | Incumbent median (s) | Candidate median (s) | Paired speedup | Favorable pairs |
|---|---:|---:|---:|---:|
| [Small, 1 energy, 1 direction, width 1](../../../test-reports/S03-compact-yDaZfK/screen/small-single/20261119T015307Z-one-v7z6wpc0/summary.json) | 0.0290 | 0.0254 | 1.14× | 3/3 |
| [Small, 3 energies, width 1](../../../test-reports/S03-compact-yDaZfK/screen/small-3-w1/20261119T015327Z-three-l8vgmjem/summary.json) | 0.0511 | 0.0404 | 1.28× | 3/3 |
| [Small, 20 energies, width 1](../../../test-reports/S03-compact-yDaZfK/screen/small-20-w1/20261119T015348Z-energy20-tsuw12nf/summary.json) | 0.1801 | 0.1178 | 1.53× | 3/3 |
| [Medium, 20 energies, width 1](../../../test-reports/S03-compact-yDaZfK/screen/medium-20-w1/20261119T015618Z-energy20-68ska8av/summary.json) | 0.6563 | 0.5941 | 1.10× | 3/3 |
| [Medium, 20 energies, width 8](../../../test-reports/S03-compact-yDaZfK/screen/medium-20-w8/20261119T015932Z-energy20-didjpp_o/summary.json) | 0.6493 | 0.5876 | 1.10× | 3/3 |
| [Small, 101 energies, width 1](../../../test-reports/S03-compact-yDaZfK/extended/small-101-w1/20261119T020039Z-energy101-l_b1xf4n/summary.json) | 0.8108 | 0.4909 | 1.66× | 5/5 |
| [Small, 101 energies, width 8](../../../test-reports/S03-compact-yDaZfK/extended/small-101-w8/20261119T020145Z-energy101-oqclko5v/summary.json) | 0.8058 | 0.4939 | 1.64× | 5/5 |
| [Small, 101 energies, output chunk 4, width 1](../../../test-reports/S03-compact-yDaZfK/extended/small-101-chunk4/20261119T020312Z-energy101-kwj6tgsk/summary.json) | 0.8197 | 0.5007 | 1.59× | 3/3 |
| [Small, 101 energies, reduced I/A, width 1](../../../test-reports/S03-compact-yDaZfK/extended/small-101-i-a/20261119T020340Z-energy101-0qgp8vmg/summary.json) | 4.4236 | 4.0234 | 1.10× | 3/3 |
| [Medium, 20 energies, device input, width 1](../../../test-reports/S03-compact-yDaZfK/extended/medium-20-device/20261119T020526Z-energy20-hhkg9685/summary.json) | 0.2846 | 0.2248 | 1.27× | 3/3 |

The medium 20-energy projection stage fell from 70.8 to 10.0 ms with nine basis FFTs in both arms. Wider tiles matched scalar end-to-end timing on the 101-energy case. The full scans and reduced I/A checks passed. Worst normalized L2 error across the S03 detector and reduced comparisons was 4.60e-07. [Stage sample](../../../test-reports/S03-compact-yDaZfK/screen/medium-20-stage-1/20261119T015222Z-energy20-wxiltrmp/summary.json)

- The eight-anisotropic-material 32-plane, 20-energy model fell from 1221 to 691 MiB on the RTX A6000. [Memory traces](../../../test-reports/S03-compact-yDaZfK/memory/results.json)
- The eight-anisotropic-material single-plane, one-energy model fell from 1455 to 959 MiB on the RTX A6000. [Memory traces](../../../test-reports/S03-compact-yDaZfK/memory/results.json)
- Medium CoreShell, 20 energies on the RTX 8000 used 2746 and 2738 MiB respectively. [Memory pass](../../../test-reports/S03-compact-yDaZfK/extended/medium-20-memory/20261119T020505Z-energy20-wxc4uy_z/summary.json)

**Selected for combination.** Use scalar fused projection. The measured wider tiles offered too little end-to-end improvement to justify extra allocations and dispatch complexity.

### S04 · XY FFT followed by selective z evaluation

The candidate performs a real FFT on each XY slice, then evaluates the axial series at the native z modes required by the compact endpoint union. It preserves Hermitian reflection and the six-neighbor DC replacement. The one-plane control uses the existing real transform.

| Matched workload | Incumbent median (s) | Candidate median (s) | Paired speedup | Favorable pairs |
|---|---:|---:|---:|---:|
| [Small, 1 energy, 1 direction](../../../test-reports/S04-selected-z-rtB3m1/screen/small-one/20261119T015315Z-one-dvveomd2/summary.json) | 0.0272 | 0.0260 | 1.05× | 3/3 |
| [Small, 1 energy, 34 directions](../../../test-reports/S04-selected-z-rtB3m1/screen/small-angles/20261119T015334Z-angles-4selbjwp/summary.json) | 0.0419 | 0.0403 | 1.01× | 2/3 |
| [Medium, 1 energy, 12 directions](../../../test-reports/S04-selected-z-rtB3m1/screen/medium-12/20261119T015449Z-one-6c07nsxa/summary.json) | 0.4447 | 0.4394 | 1.01× | 2/3 |
| [Medium, 20 energies, 12 directions](../../../test-reports/S04-selected-z-rtB3m1/screen/medium-20/20261119T015541Z-energy20-1pxhj7gs/summary.json) | 0.6126 | 0.6058 | 1.01× | 3/3 |
| [128 × 1024 × 1024, 1 energy, 1 direction](../../../test-reports/S04-selected-z-rtB3m1/screen/medium-z128/20261119T015652Z-depth-togy1prv/summary.json) | 0.8582 | 0.8446 | 1.01× | 3/3 |

The medium 20-energy basis build fell from 61.9 to 56.7 ms. Sampled peaks were 2849 and 2849 MiB at depth 64, and 5409 and 5409 MiB at depth 128. Worst normalized L2 error was 5.02e-07. [Depth 128 memory](../../../test-reports/S04-selected-z-rtB3m1/memory/medium-z128/20261119T020325Z-depth-_o_opae3/summary.json)

**Deferred.** The measured XY-first evaluator stayed below the primary speed threshold on the tested workloads. A z-first variant remains unmeasured.

### S05 · Fused tensor angle weighting, rotation, and accumulation

The fused kernel uses the same inverse affine matrices, bilinear interpolation bounds, per-angle weights, and valid-count averaging. Odd rectangular and even grids passed, including quarter turns and both rotation mask states. The prototype applies fusion to multiple electric-field directions.

| Matched workload | Incumbent median (s) | Candidate median (s) | Paired speedup | Favorable pairs |
|---|---:|---:|---:|---:|
| [Small, 34 directions, rotation only](../../../test-reports/S05-rotation-nQjkRW/screen/small-34-alone/20261119T015908Z-angles-luxzkv5d/summary.json) | 0.0433 | 0.0314 | 1.39× | 3/3 |
| [Small, 34 directions, with fused projection in both arms](../../../test-reports/S05-rotation-nQjkRW/screen/small-34-incremental/20261119T020020Z-angles-dyz7eyg1/summary.json) | 0.0381 | 0.0267 | 1.42× | 3/3 |
| [Medium, 20 energies, rotation only](../../../test-reports/S05-rotation-nQjkRW/screen/medium-20-alone/20261119T015929Z-energy20-0xk9rd0b/summary.json) | 0.6104 | 0.5304 | 1.15× | 3/3 |
| [Medium, 20 energies, with fused projection in both arms](../../../test-reports/S05-rotation-nQjkRW/screen/medium-20-incremental/20261119T020040Z-energy20-zd6sioir/summary.json) | 0.5344 | 0.4589 | 1.16× | 3/3 |
| [Small, 34 directions, projection and rotation](../../../test-reports/S05-rotation-nQjkRW/screen/small-34-combined/20261119T020147Z-angles-pw3s8ynr/summary.json) | 0.0427 | 0.0275 | 1.56× | 3/3 |
| [Medium, 20 energies, projection and rotation](../../../test-reports/S05-rotation-nQjkRW/screen/medium-20-combined/20261119T020206Z-energy20-5m7xapox/summary.json) | 0.6049 | 0.4627 | 1.31× | 3/3 |

With fused projection in both arms, the medium 20-energy rotation stage fell from 85.9 to 9.4 ms. Worst normalized L2 error was 3.55e-07 in the full detector comparisons. [Stage sample](../../../test-reports/S05-rotation-nQjkRW/screen/medium-20-stages/20261119T020130Z-energy20-ezcgxbqg/summary.json)

**Selected for combination.** Use fused weighting, rotation, and accumulation for more than one electric-field direction.

## S06 · Combined compact projection and rotation

The proposed compact tensor path uses the fused projection on every energy and the fused angle kernel on multiple directions. The actual default entry point was compared with the qualified compact default on the same Wheatley RTX A6000. The experiment builds one basis with nine real FFTs for the CoreShell fixture and keeps compute tile width one.

| Workload | Incumbent median (s) | Candidate median (s) | Paired speedup | Favorable pairs |
|---|---:|---:|---:|---:|
| [Small, one energy, one direction](../../../test-reports/S06-combined-LqHImM/gate/small-single/20261119T020538Z-one-nz_bls2e/summary.json) | 0.0285 | 0.0250 | 1.13× | 5/5 |
| [Small, one energy, 34 directions](../../../test-reports/S06-combined-LqHImM/gate/small-34/20261119T020611Z-angles-r7w4cz_0/summary.json) | 0.0418 | 0.0276 | 1.52× | 5/5 |
| [Medium, one energy, 12 directions](../../../test-reports/S06-combined-LqHImM/gate/medium-12/20261119T020646Z-one-twm_4jfd/summary.json) | 0.4804 | 0.4709 | 1.02× | 4/5 |
| [Medium, 20 energies, 12 directions, host input](../../../test-reports/S06-combined-LqHImM/gate/medium-20/20261119T020857Z-energy20-9kls2aro/summary.json) | 0.6510 | 0.5015 | 1.29× | 5/5 |
| [Medium, 20 energies, 12 directions, device input](../../../test-reports/S06-combined-LqHImM/gate/medium-20-device/20261119T021036Z-energy20-yc6v7p1_/summary.json) | 0.2713 | 0.1299 | 2.09× | 5/5 |
| [Large, 20 energies, 12 directions, host input](../../../test-reports/S06-combined-LqHImM/gate/large-20/20261119T021219Z-energy20-i4rf4hyq/summary.json) | 1.9309 | 1.7228 | 1.12× | 5/5 |
| [Small, 101 energies, 12 directions, host input](../../../test-reports/S06-combined-LqHImM/gate/small-101/20261119T021701Z-energy101-rm_bgc1y/summary.json) | 0.8178 | 0.1949 | 4.19× | 5/5 |

- Medium, 20 energies, sampled peak 2849 → 2841 MiB. [Observer result](../../../test-reports/S06-combined-LqHImM/gate/medium-20-memory/20261119T021804Z-energy20-3c7e21qj/summary.json)
- Large, 20 energies, sampled peak 9793 → 9785 MiB. [Observer result](../../../test-reports/S06-combined-LqHImM/gate/large-20-memory/20261119T021828Z-energy20-6h9qnwfm/summary.json)
- Eight-anisotropic-material 32-plane, 20-energy model on RTX 8000, sampled peak 1118 → 588 MiB. [Observer result](../../../test-reports/S06-combined-LqHImM/material-memory/results.json)
- Eight-anisotropic-material single-plane, one-energy model on RTX 8000, sampled peak 1352 → 856 MiB. [Observer result](../../../test-reports/S06-combined-LqHImM/material-memory/results.json)

The fast CPU and tensor GPU smoke tests passed. Full accuracy comparisons passed at all 101 energies for a single direction, the 12-direction scan, and the reduced I/A result with result chunk size four. Additional controls cover nonmonotonic energies with Hann window and rotation mask enabled, a single plane, and the medium 20-energy scan. Worst normalized L2 was 4.60e-07. [Gate evidence](../../../test-reports/S06-combined-LqHImM/gate-evidence.json)

The combined candidate proceeds to the standard S07 qualification.

## S05 follow-up on the combined compact candidate

Profiled the small 101-energy CoreShell scan with 12 electric-field directions and host input and output on the Wheatley RTX 8000. These are single diagnostic runs. The integrated and reduced layouts use the existing intersection q grid by default.

| Result layout | Output chunk | Wall time (s) | Reduction (s) | Device-to-host copy (s) |
|---|---:|---:|---:|---:|
| [`detector`](../../../test-reports/S06-combined-LqHImM/promotion-evidence/profiles/detector-1/20261119T022711Z-energy101-ffm64zc8/r01-A.result.json) | 1 | 0.2178 | 0.0000 | 0.0084 |
| [`detector`](../../../test-reports/S06-combined-LqHImM/promotion-evidence/profiles/detector-4/20261119T022715Z-energy101-d_wp2pf1/r01-A.result.json) | 4 | 0.2144 | 0.0000 | 0.0082 |
| [`i_a`](../../../test-reports/S06-combined-LqHImM/promotion-evidence/profiles/i_a-1/20261119T022719Z-energy101-ivpqk0j_/r01-A.result.json) | 1 | 3.8295 | 3.5582 | 0.0019 |
| [`i_a`](../../../test-reports/S06-combined-LqHImM/promotion-evidence/profiles/i_a-4/20261119T022731Z-energy101-o6ysyx1r/r01-A.result.json) | 4 | 3.6225 | 3.3842 | 0.0005 |

The measured target is reduction. The current common-q remap calls CuPy interpolation once per azimuthal row. A batched remap can reuse the interpolation indices and weights across all rows for an energy. Screen against the same reduced product and verify the q coordinates, sector weights, empty bins, and masks for both intersection and union grids.

For the medium host-input 20-energy detector run, staging took 421 ms in a 500 ms profiled run. A host-staging experiment would be a separate follow-up for compact tensor. [Medium profile](../../../test-reports/S06-combined-LqHImM/gate/medium-20-stages)

## S07 · Combined compact projection and rotation qualification

The combined compact candidate passed the eight maintained report stages with matching applicability against the promoted compact reference. The physics validation entered the compact path at one and multiple energies. The tested source is identified by its SHA-256 manifest.

The report shell lost its filesystem handle after the final passing test stage. The summary was regenerated with the standard renderer and verified against the saved case tables, logs, and three graphics archives.

[Full report](../../../test-reports/S06-combined-LqHImM/validation/20261119T022340Z/summary.md) · [Qualification evidence](../../../test-reports/S06-combined-LqHImM/qualification.json)

Promoted the qualified compact projector and angle kernel on `perf/pre-kin-speed`. The promoted checkout passed 35 tensor GPU smoke tests. [Memory measurements for the current default](../../../test-reports/S06-combined-LqHImM/promotion-evidence/results.json)

## S05 · Batched common-q interpolation

Compute the two source bins for each energy and common-q coordinate once at result-plan construction. Apply all of the row interpolations in one CUDA launch per output chunk, writing into the existing chunk buffer when present. Matched timing uses the same fused compact default, layout, and host inputs and outputs on the Wheatley RTX 8000.

| Matched workload | Per-row (s) | Batched (s) | Paired speedup | Faster pairs |
|---|---:|---:|---:|---:|
| [Small, 101 energies, I/A, output chunk 1](../../../test-reports/S05-common-q-zDoMZT/timing/small-101/20261119T132321Z-energy101-kadlk3y5/summary.json) | 3.7302 | 0.5343 | 7.03× | 5/5 |
| [Small, 101 energies, I/A, output chunk 4](../../../test-reports/S05-common-q-zDoMZT/timing/small-101-chunk4/20261119T132438Z-energy101-470zdvvg/summary.json) | 3.5395 | 0.2909 | 12.09× | 5/5 |
| [Medium, 20 energies, I/A](../../../test-reports/S05-common-q-zDoMZT/timing/medium-20/20261119T132627Z-energy20-6t9r676b/summary.json) | 1.2060 | 0.5645 | 2.14× | 3/3 |
| [Small, 20 energies, integrated](../../../test-reports/S05-common-q-zDoMZT/timing/small-20-integrated/20261119T132756Z-energy20-v2r1iyyf/summary.json) | 0.7780 | 0.1283 | 6.06× | 3/3 |
| [Small, one energy, I/A](../../../test-reports/S05-common-q-zDoMZT/timing/small-single-ia/20261119T132821Z-one-k8lrb_mh/summary.json) | 0.0747 | 0.0748 | 1.01× | 2/3 |
| [Small, 101 energies, detector](../../../test-reports/S05-common-q-zDoMZT/timing/small-101-detector/20261119T132843Z-energy101-wfm3j1jv/summary.json) | 0.2153 | 0.2126 | 1.01× | 3/3 |

- Small, 101 energies, output chunk 1, I/A, sampled peak 546 → 546 MiB. [Observer pass](../../../test-reports/S05-common-q-zDoMZT/timing/small-101-memory/20261119T132551Z-energy101-1__edvo6/summary.json)
- Small, 101 energies, output chunk 4, I/A, sampled peak 546 → 546 MiB. [Observer pass](../../../test-reports/S05-common-q-zDoMZT/timing/small-101-chunk4-memory/20261119T132609Z-energy101-phap0wfm/summary.json)
- Medium, 20 energies, output chunk 1, I/A, sampled peak 2746 → 2746 MiB. [Observer pass](../../../test-reports/S05-common-q-zDoMZT/timing/medium-20-memory/20261119T132731Z-energy20-nmg10jqu/summary.json)

- The 101-energy reduction stage fell from 3.509 to 0.310 s at chunk size 1. [Stage profile](../../../test-reports/S05-common-q-zDoMZT/timing/small-101-stage/20261119T132251Z-energy101-9qw4k9c3/summary.json)
- The 101-energy reduction stage fell from 3.276 to 0.101 s at chunk size 4. [Stage profile](../../../test-reports/S05-common-q-zDoMZT/timing/small-101-chunk4-stage/20261119T132306Z-energy101-twzx9h5n/summary.json)

All seven matched accuracy lanes returned identical coordinates, finite values, and NaN masks. The screen covers all 101 I/A energies, medium integrated output, a nonmonotonic soft/hard union grid, sectors with a 47° wedge and device results, a single-plane control, and the standard detector layout. The CPU and both CuPy GPU smoke lanes passed. [Matched accuracy evidence](../../../test-reports/S05-common-q-zDoMZT/accuracy) · [Smoke evidence](../../../test-reports/S05-common-q-zDoMZT/smoke.log)

A separate RTX A6000 profile times the polar mapping, batched common-q remap, and final chi and channel reduction. [Reduction breakdown](../../../test-reports/S05-common-q-zDoMZT/reduction-profile.json)

## S07 · Batched common-q qualification

All eight maintained smoke and physics stages passed. The exact 101-energy detector comparisons passed at a fixed electric-field direction and across the usual angle sweep. The reduced comparisons matched shared q coordinates, values, and masks exactly. The tested source was promoted on `perf/pre-kin-speed`. The active checkout passed 5 reduced-output smoke tests.

[Full report](../../../test-reports/S05-common-q-zDoMZT/validation/20261119T133246Z/summary.md) · [Qualification evidence](../../../test-reports/S05-common-q-zDoMZT/qualification.json)

## S05 · Separable Hann window in compact tensor basis

The windowed compact tensor basis uses separable Hann factors. Generation multiplies the final window weight into the scalar or orientation-product field in a single CUDA operation. The multiplication order matches the full-window path. The matched benchmarks ran on the Wheatley RTX 8000 with the same Fourier basis, angles, energies, and detector results.

| Workload | Full window (s) | Fused (s) | Paired speedup | Favorable pairs |
|---|---:|---:|---:|---:|
| [Small, one energy, device inputs](../../../test-reports/S05-hann-qHFShU/screen/small-one/20261119T143518Z-one-qdma6yv8/summary.json) | 0.0297 | 0.0280 | 1.06× | 2/3 |
| [Medium, one energy, device inputs](../../../test-reports/S05-hann-qHFShU/screen/medium-one-device/20261119T143539Z-one-p1iaxdeg/summary.json) | 0.1227 | 0.1092 | 1.12× | 5/5 |
| [Medium, 20 energies, device inputs](../../../test-reports/S05-hann-qHFShU/screen/medium-20-device/20261119T143721Z-energy20-x7zp5ffp/summary.json) | 0.1579 | 0.1386 | 1.14× | 5/5 |
| [Medium, 20 energies, host inputs, unbound](../../../test-reports/S05-hann-qHFShU/screen/medium-20-host/20261119T143850Z-energy20-ak9v01nd/summary.json) | 0.4165 | 0.5218 | 0.80× | 1/3 |
| [Large, 20 energies, host inputs, unbound](../../../test-reports/S05-hann-qHFShU/screen/large-20-host/20261119T144211Z-energy20-ldz38fi3/summary.json) | 1.2177 | 1.1405 | 1.06× | 3/3 |
| [Medium, 20 energies, host inputs, GPU-local NUMA node](../../../test-reports/S05-hann-qHFShU/screen/medium-20-local/20261119T144822Z-energy20-f3nqdj0z/summary.json) | 0.4967 | 0.4795 | 1.03× | 5/5 |
| [Large, 20 energies, host inputs, GPU-local NUMA node](../../../test-reports/S05-hann-qHFShU/screen/large-20-local/20261119T144945Z-energy20-vc2s2vrp/summary.json) | 1.6829 | 1.6323 | 1.03× | 3/3 |
| [Large, 20 energies, device inputs](../../../test-reports/S05-hann-qHFShU/screen/large-20-device/20261119T144441Z-energy20-0it6xi3a/summary.json) | 0.4639 | 0.4002 | 1.16× | 3/3 |
| [Medium, 20 energies, device inputs, window off](../../../test-reports/S05-hann-qHFShU/screen/medium-20-window-off/20261119T143954Z-energy20-6xoqeruh/summary.json) | 0.1399 | 0.1395 | 1.00× | 2/3 |

- Medium, 20 energies with host inputs, sampled peak 2994 → 2738 MiB. [Memory pass](../../../test-reports/S05-hann-qHFShU/screen/medium-20-memory/20261119T144056Z-energy20-yu5qgenu/summary.json)
- Large, 20 energies with host inputs, sampled peak 10546 → 9682 MiB. [Memory pass](../../../test-reports/S05-hann-qHFShU/screen/large-20-memory/20261119T144118Z-energy20-ejcf2r37/summary.json)

On medium with device inputs, window setup plus basis build took 96.5 ms and 78.1 ms with nine real FFTs in both arms. [Stage profile](../../../test-reports/S05-hann-qHFShU/screen/medium-20-stage/20261119T143456Z-energy20-69pqblsq/summary.json)

Unbound host-input runs on Wheatley contained two wall-time bands. The follow-up pinned both arms to the RTX 8000's local CPU and memory node. Median medium staging was 392.6 and 390.1 ms, with 0.9% median absolute paired variation in the local A/A control. [Local stage profile](../../../test-reports/S05-hann-qHFShU/screen/medium-20-local-stage/20261119T144748Z-energy20-uy4nb469/summary.json) · [A/A control](../../../test-reports/S05-hann-qHFShU/screen/medium-20-local-aa/20261119T144659Z-energy20-8a8atoy_/summary.json)

All ten accuracy comparisons matched coordinates, finite outputs, and masks exactly. They covered odd and even grids, single and two-plane windows, 101-energy detector output at one and multiple electric-field directions, reduced I/A on shared q grids, and a window-off control. The forced allocation-fallback smoke test produced the same output as the per-energy direct reference. [Accuracy screen](../../../test-reports/S05-hann-qHFShU/accuracy) · [Full scans](../../../test-reports/S05-hann-qHFShU/supplement) · [Fallback test](../../../test-reports/S05-hann-qHFShU/accuracy.log)

## S07 · Separable Hann window qualification

All eight maintained smoke and physics stages passed. Full windowed detector and I/A comparisons matched the full-window reference at every energy. The active source matched the qualification manifest after installation on `perf/pre-kin-speed` and passed 9 basis smoke tests.

[Full validation report](../../../test-reports/S05-hann-qHFShU/validation/20261119T143738Z/summary.md) · [Qualification evidence](../../../test-reports/S05-hann-qHFShU/qualification.json)
