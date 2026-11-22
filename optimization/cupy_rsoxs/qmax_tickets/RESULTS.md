# Qmax qualification results

## QM01 public output contract

The parent is post-ZP15 commit `bb152f0e31d9ea09668fc8fa3522132c1c759bc8` in a detached worktree. Both sides used Python 3.14.3 and CuPy 14.0.1. The synthetic A/A controls ran on the RTX A6000 with the same unbounded parent in both arms. The small `9 × 33 × 35` specimen used a logical `36 × 64 × 72` target, three energies, and twelve E directions. Every route built one compact basis per scan. [Parent checkout](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax-parent) · [A/A executions and timings](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM01.d4Dnff/aa)

| Parent route | A/A samples checked | Bit mismatches | Median paired warm ratio |
| --- | ---: | ---: | ---: |
| Native compact tensor | 4,069 | 0 | 1.010 |
| Virtual Z sum | 4,069 | 0 | 1.005 |
| Virtual Z residues | 4,069 | 0 | 0.997 |
| Virtual XYZ | 4,069 | 0 | 1.037 |

The first candidate selects the requested Cartesian window before allocating or transferring the public result. It uses the global detector coordinates and the cutoff at each energy. Reduced layouts select positions from the full registered radial grid and retain the corresponding auxiliary coordinates. The candidate keeps the inclusive boundary, zero cutoff, supported masks, and the matching full result on the native allocation fallback. The baseline comparisons exercised the compact default and all three padded evaluators. [Contract test log](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM01.d4Dnff/qwindow-pytest.log) · [Contract tree](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM01.d4Dnff/qm01-contract-tree.txt)

The independent oracle compared 166,692 samples with zero bit mismatches across eight A/B runs on the Quadro RTX 8000. The comparison covers omitted cutoff, 0.8 nm⁻¹, zero on device, a cutoff beyond support, and all four reduced layouts. The first GPU fixture revision inherited its material energy schedule and assumed the fallback was tensor-based. The final cases set the material energies explicitly and verify the actual incumbent fallback, with 22 passes. CPU smoke passed 143 cases with seven skips. [A/B results](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM01.d4Dnff/ab) · [CPU smoke](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM01.d4Dnff/cpu-smoke.log) · [Initial fixture diagnostic](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM01.d4Dnff/fixture-diagnostic.log)

The dependency probe confirms that the off-axis `77 × 169` detector loses 115 of its 189 finite rays with a fixed four-pixel guard across twelve directions. Explicitly tracing the dependencies preserves every ray. The CPU FFT experiment also found 31,539 changed complex coefficients across 42,840 samples after reordering transform stages. These results motivate planning from the original affine dependencies and retaining the incumbent transform when qualifying restricted work. [Dependency probe](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM01.d4Dnff/dependency-probe.json) · [FFT probe](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM01.d4Dnff/native-fft-probe.json)

## QM02 radial interpolation demand

The first computation change plans the reduced q window before building the basis and retains the native radial samples through the last required interpolation bracket, including the upper neighbor at an exact lower-bin match. The polar remesh computes only that prefix. It restores the original radial stride for the existing interpolation and chi reduction so the cuBLAS reduction sees the same dimensions. In the `36 × 64 × 73` logical example at 0.8 nm⁻¹, all four routes remeshed 22 of 49 radial bins per azimuth and returned 1,008 bytes for three-energy I/A. The latest checks covered all four reduced layouts, a zero-cutoff device return, XY Hann, and both common-grid modes across smoke and public comparisons. There were 92,004 additional checked values and zero bit mismatches. The cutoff smoke, padding smoke and q-grid tests passed 131 cases on the RTX A6000. CPU smoke passed 143 with seven skips. [Paired radial checks and diagnostic](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM01.d4Dnff/polar) · [Focused tests](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM01.d4Dnff/focused-smoke.log) · [CPU smoke](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM01.d4Dnff/cpu-final.log)

At this checkpoint, detector projection and Fourier gathering still covered the full grid.

## QM02 detector and endpoint demand

The compact projection path bounds its output before it builds the Fourier basis. The source rectangle includes the global inverse-affine dependencies and the polar interpolation neighbors. Both GPU kernels use the original logical coordinates, rounding and angle order. A single off-axis direction retains the existing affine path and applies the cutoff at the result boundary. The native bitmap and the padded union builders now use the bounded source demand. The virtual XYZ path derives the incumbent X-column set from the full detector support and the DC stencil, preserving its FFT sizes and tiling. [Projection-only cases](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM02.jfEQUf/projection-pairs) · [Endpoint cases](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM02.jfEQUf/endpoint-pairs)

The independent enumeration verified that the rectangles cover every required input and output pixel. The count comparison shows where exact packing could reduce additional work. These rows use 0.8 nm⁻¹ and twelve E directions. [Dependency enumeration](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM02.jfEQUf/dependency-coverage.json) · [Probe source](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM02.jfEQUf/enumerate-dependencies.py)

| Logical detector YX | Product | Full detector cells | Source rectangle cells | Enumerated source cells |
| --- | --- | ---: | ---: | ---: |
| 77 × 169 | Detector | 13,013 | 8,778 | 7,178 |
| 77 × 169 | I/A | 13,013 | 9,163 | 7,525 |
| 512 × 512 | Detector | 262,144 | 196,249 | 78,281 |
| 512 × 512 | I/A | 262,144 | 201,601 | 80,025 |

On the odd rectangular detector, limiting the source reduced native endpoints from 28,396 to 19,521 and padded endpoints from 28,397 to 19,522. The 20-energy, 1024² device XYZ test at 0.5 nm⁻¹ reduced endpoint count from 1,241,139 to 730,275 and projected cells per energy from 1,048,576 to 305,809. Both arms retained 370 X columns for the factored FFT. All 2,451,600 limited output samples and 20,971,520 unbounded control samples matched bit for bit. The initial smoke run passed the new geometry argument through unbounded calls and broke existing literal-Fourier test hooks. The revised call passes it only when the cutoff restricts the Fourier demand, keeping the original tests and assertions. [Odd detector diagnostic](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM02.jfEQUf/endpoint-pairs/diagnose-fw586coq/summary.json) · [Dense XYZ diagnostic](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM02.jfEQUf/endpoint-pairs/diagnose-g80_hpwr/summary.json) · [Initial smoke diagnostic](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM02.jfEQUf/endpoint-smoke.log)

The corrected focused GPU panel passed 140 tests. CPU smoke passed 146 tests with seven skips, including the full-column support check for one-, two- and 17-row detectors. [GPU smoke](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM02.jfEQUf/endpoint-smoke-final.log) · [CPU smoke](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM02.jfEQUf/cpu-final-verified.log)

## QM03 native screen

The first 20-energy `128 × 512 × 512` host detector run used 0.8 nm⁻¹ and twelve directions on the A6000. The CPU-mask implementation returned 7.84 MB in place of 20.97 MB and reduced peak host RSS from 1,169.95 to 1,127.62 MiB. Peak VRAM was 1,684 MiB in each arm. The warm median moved from 219.45 to 233.58 ms, a 6.4% increase. Median A/A ratio was 0.997. The FFT count stayed at eight and the diagnostic counted 575,410 retained endpoints out of 828,474. All 1,542,948 selected samples matched. [First native timing, memory and checks](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM02.jfEQUf/native-screen)

The selected implementation transfers the Cartesian radius once per scan and fuses the per-energy mask with the output write. It uses the original double-precision operations without contraction to preserve the inclusive boundary. The updated native warm times were within the A/A drift at both cutoffs. Each row uses three matched worker pairs and five warm calls per worker. The first-call median includes any setup performed inside the timed call. [Matched native timing and memory](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM02.jfEQUf/device-mask-native)

| Cutoff (nm⁻¹) | First (ms), parent / limited | Warm (ms), parent / limited | Peak host RSS (MiB), parent / limited | Peak GPU VRAM (MiB), parent / limited | Returned bytes, parent / limited | Samples checked |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.80 | 328.24 / 344.41 | 204.95 / 205.86 | 1,171.90 / 1,126.68 | 1,684 / 1,684 | 20,971,520 / 7,837,520 | 1,542,948 |
| 0.25 | 326.25 / 335.93 | 201.19 / 199.77 | 1,171.67 / 1,113.61 | 1,684 / 1,684 | 20,971,520 / 816,080 | 161,700 |

The 0.25 nm⁻¹ case retained 41,990 of 828,474 Fourier endpoints. Both cutoffs kept all eight native FFTs. Every listed sample matched bit for bit, as did 2,501,996 samples in additional odd, broad-cutoff and 1024² device checks. The final q-window smoke run passed 28 cases. [Additional raw detector checks](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM02.jfEQUf/device-mask-correctness) · [Q-window smoke and worker launch](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM02.jfEQUf/device-mask-native.log)

The instrumented 0.8 nm⁻¹ pair isolated 176.64 / 177.10 ms for native material staging in 219.69 / 222.75 ms public calls. The eight FFTs took 9.73 / 9.74 ms inside the basis stage. Rotation fell from 6.62 to 3.04 ms and host copy from 1.68 to 0.65 ms. Per-energy geometry and coefficient projection rose from 3.61 to 7.38 ms, leaving the complete energy stage at 10.35 / 10.50 ms. The timing brackets overlap for the basis, FFTs, and energy subphases. [Parent profile](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM04-timing.RpCctd/native-uncapped-profile.json) · [Cutoff profile](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM04-timing.RpCctd/native-limited-profile.json)

## QM04 paired padding timings

Each row compares three fresh worker pairs with five warm calls per worker and a separate NVML memory pass. The water-filled vesicle kept the 52³ stored crop, 1024³ logical shape, 101 energies, 1.5 nm pitch and one direction on the RTX A6000. The Z-only fixtures used a `32 × 512 × 512` input and a `256 × 512 × 512` target, 20 energies and twelve directions on the Quadro RTX 8000. All rows use host output and a cutoff of 0.8 nm⁻¹. [Vesicle matched runs](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM04-timing.RpCctd/summary.json) · [Vesicle chunk-four runs](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM04-timing.RpCctd/chunk4/summary.json) · [Virtual Z workers](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM04-timing.RpCctd/z)

| Workload | First (ms), full / limited | Warm (ms), full / limited | Peak host RSS (MiB), full / limited | Peak VRAM (MiB), full / limited | Returned bytes, full / limited |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vesicle XYZ, native I/A, chunk 1 | 741.9 / 763.4 | 401.5 / 423.0 | 717.0 / 713.9 | 1,042 / 458 | 1,171,600 / 479,952 |
| Vesicle XYZ, native I/A, chunk 4 | 508.5 / 515.3 | 156.0 / 162.1 | 723.0 / 718.7 | 1,042 / 458 | 1,171,600 / 479,952 |
| Virtual Z sum, detector | 253.2 / 268.7 | 92.2 / 86.2 | 749.2 / 706.7 | 602 / 578 | 20,971,520 / 7,837,520 |
| Virtual Z residues, detector | 348.2 / 361.8 | 113.9 / 109.1 | 757.8 / 714.0 | 670 / 646 | 20,971,520 / 7,837,520 |

The vesicle window lowered peak VRAM by 584 MiB at each chunk size. Median paired warm speedups were 0.949× and 0.962× for chunks one and four. The uncapped A/A chunk-one ratios spanned 0.983–1.002×. The window reduced basis work and projection on the vesicle. In the chunk-four diagnostic, axial accumulation fell from 9.68 to 1.49 ms and the projection kernel from 17.44 to 5.67 ms. The native I/A reduction rose from 88.50 to 108.33 ms. All 29,997 retained values in each of I and A matched for each chunk size. The full chunk-one replay also matched both complete saved virtual-XYZ arrays. [Vesicle A/A](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM04-timing.RpCctd/vesicle-aa.json) · [Vesicle chunk-one diagnostic](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM04-timing.RpCctd/summary.json) · [Vesicle chunk-four diagnostic](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM04-timing.RpCctd/chunk4/summary.json)

Virtual Z sum and residues improved by 1.055× and 1.044× in the paired warm measurements. Their A/A median ratios were 1.003× and 1.009×. Each retained 760,491 of 1,170,071 endpoints with zero bit mismatches across 1,542,948 selected pixels. The two evaluators used eight and 72 FFT calls respectively in both arms. [Virtual Z timing](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM04-timing.RpCctd/z/time-yarhg8g9/summary.json) · [Virtual Z parity](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM04-timing.RpCctd/z/check-jvwqyhvt/summary.json) · [Virtual Z diagnostic](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM04-timing.RpCctd/z/diagnose-nebv8xow/summary.json)

### Vesicle host I/A chunk sweep

The same 101-energy, 1024³ virtual vesicle ran on the A6000 with matched uncapped and 0.8 nm⁻¹ arms. Each chunk size used three fresh worker pairs with five synchronized warm calls per worker. Batch sizes and A/B order were interleaved. Peak VRAM came from separate 5 ms NVML passes. [Source identification and run order](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/vesicle-chunks.yAG6nG/source.json) · [Complete timing, peak RSS and VRAM records](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/vesicle-chunks.yAG6nG/summary.json)

| Chunk size | First (ms), full / limited | Warm (ms), full / limited | Peak VRAM (MiB), full / limited |
| ---: | ---: | ---: | ---: |
| 4 | 537.63 / 538.60 | 155.56 / 160.93 | 1,042 / 458 |
| 8 | 484.77 / 485.59 | 123.11 / 115.84 | 1,042 / 490 |
| 16 | 467.24 / 466.45 | 112.13 / 96.52 | 1,170 / 596 |
| 32 | 466.90 / 451.43 | 105.60 / 86.65 | 1,618 / 832 |
| 64 | 462.53 / 452.90 | 101.19 / 80.01 | 2,514 / 1,290 |
| 101 | 457.62 / 444.48 | 99.73 / 77.26 | 3,370 / 1,826 |

With the cutoff, chunk 32 was 1.86× faster than chunk 4. Chunk 101 saved another 9.38 ms and used 994 MiB more peak VRAM than chunk 32. The matched cutoff speedup at chunk 101 was 1.29×. All 29,997 retained values in each of I and A matched bit for bit within each chunk size. Comparing chunk sizes, A stayed bitwise identical and I differed by at most `1.03e-12` in absolute units, or `1.29e-15` of the intensity peak. [Per-chunk bitwise comparisons](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/vesicle-chunks.yAG6nG/parity.json)

## Final vesicle comparison

The explicit `256 × 1024 × 1024` panel uses the saved NRSS-native I/A data and its measured warm time of 2.220 s and peak VRAM of 11,550 MiB. The virtual-XYZ panel uses the frozen `52³` crop with a `1024³` logical grid, the same 101 energies from 282.55 to 292.55 eV, one E direction, and host I/A output. Both panels display the shared 0.06–0.80 nm⁻¹ fit band with one anisotropy scale. [Saved incumbent data](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/vesicle-101-explicit-vs-1024cube.2ObrIJ/explicit-i-a.nc) · [Saved benchmark](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/vesicle-101-explicit-vs-1024cube.2ObrIJ/summary.json)

The QM02 cutoff output matched all 29,997 retained I and 29,997 retained A values in the saved full virtual-XYZ result on the original A6000. A check on the Quadro RTX 8000 first showed low-bit differences in I, so the comparison was repeated on the original GPU using the saved reference. The cutoff ran projection for 143,641 of 1,048,576 cells per energy and remeshed 190 of 725 polar bins. It retained 363 factored X columns. This was a correctness check. [Matched vesicle result](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM02.jfEQUf/lipid-bit-check.json) · [Cross-GPU diagnostic](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM02.jfEQUf/cross-gpu/lipid-bit-check.log)

The virtual-XYZ panel with `q_max=0.8` and chunk 101 took 77.26 ms warm with 1,826 MiB peak VRAM. The 28.7× speedup from the saved baseline combines virtual padding, the cutoff, and chunk tuning. The matched chunk-101 pair isolates the cutoff at 99.73 → 77.26 ms and 3,370 → 1,826 MiB. [Final figure](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/vesicle-chunks.yAG6nG/incumbent-comparison/vesicle-101-energy-incumbent-vs-qmax-chunk101.pdf)

## QM05 opt-in qualification

Set `backend_options={"q_max": 0.8}` to request q ≤ 0.8 nm⁻¹. Omitting the option or setting it to `None` uses the incumbent accessible q range for the selected layout. The synchronized timings pair the uncapped parent at `bb152f0e31d9ea09668fc8fa3522132c1c759bc8` with the opt-in candidate. [User guide](../../../docs/source/user_guide/cupy_rsoxs_result_q_grids.rst) · [Candidate source fingerprints](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM05.s21EA7/qualification-source.sha256)

The maintained reports ran with Python 3.14.3, pytest 9.0.2 and CuPy 14.0.1. The validation reference manifests are identical.

| Lane | Incumbent | Candidate |
| --- | --- | --- |
| CPU smoke | 132 passed, 7 skipped | 146 passed, 7 skipped |
| Default compact tensor GPU smoke | 110 passed, 10 skipped, 1 failed | 124 passed, 10 skipped, 1 failed |
| Default compact tensor physics | 23 passed, 2 failed | 23 passed, 2 failed |
| Optional direct GPU smoke | 31 passed, 89 skipped, 1 failed | 31 passed, 103 skipped, 1 failed |
| Optional direct physics | 19 passed, 4 skipped, 2 failed | 19 passed, 4 skipped, 2 failed |

The shared failures were the multi-energy comparison to PyHyperScattering's `NRSSIntegrator` and the 70 and 128 nm sphere tests using that integrator. Both sphere sizes pass the existing thresholds with NRSS native integration. Parent and candidate produced identical native-integration metrics. [Incumbent report](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM05.s21EA7/report-parent/20261122T201324Z/summary.md) · [Candidate report](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM05.s21EA7/report-candidate/20261122T201324Z/summary.md) · [Incumbent native sphere checks](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM05.s21EA7/parent-native-sphere-theory.json) · [Candidate native sphere checks](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM05.s21EA7/candidate-native-sphere-theory.json)

The optional direct report reproduced the same three failures through its execution path. Its 14 additional skips are the new compact tensor GPU tests. The incumbent direct report was reused from the ZP15 qualification and its runtime source hashes match the pinned parent. The focused candidate q-window suite passed all 28 tests. [Incumbent direct report](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/direct-candidate/20261122T173714Z/summary.md) · [Candidate direct report](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM05.s21EA7/report-direct-candidate/20261122T202941Z/summary.md) · [Test outcome comparison](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM05.s21EA7/full-report-parity.json) · [Focused test log](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM05.s21EA7/q-window-final-smoke.log)

The independent cutoff oracle checked 94,125 samples across the signed off-axis grating on three routes, the tilted HCP lattice, and the anisotropic CoreShell sweep, with zero bit mismatches. The CoreShell comparisons include raw compact tensor and direct polarization, native I/A, a dark channel, and the full-range result with explicit `None`. The oracle verifies the retained coordinates and requested values, including existing invalid pixels, and checks the output mask. [Per-energy physics checks](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM05.s21EA7/physics-cutoff/summary.json)

Selected host detector comparisons use twelve E directions. Each timing uses three fresh worker pairs and five warm calls per worker. A separate set of workers measured peak process RSS and sampled per-process VRAM. [All timings, peaks and output sizes](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM05.s21EA7/comparison.csv) · [Per-energy detector counts and bit checks](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM05.s21EA7/checked-by-energy.json)

| Route and logical shape (ZYX) | Energies | Cutoff (nm⁻¹) | First (ms), full / limited | Warm (ms), full / limited | Peak RSS (MiB), full / limited | Peak VRAM (MiB), full / limited |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native, 128 × 512² | 101 | 0.80 | 488.60 / 508.44 | 311.49 / 345.37 | 1,362.94 / 1,239.41 | 1,684 / 1,684 |
| Native, 128 × 1024² | 20 | 0.80 | 1,004.23 / 962.05 | 828.33 / 809.97 | 3,378.18 / 3,378.23 | 5,908 / 5,908 |
| Virtual Z sum, 256 × 512² | 20 | 0.50 | 258.31 / 263.16 | 93.46 / 80.05 | 749.81 / 697.68 | 602 / 530 |
| Virtual Z residues, 256 × 512² | 20 | 0.50 | 354.51 / 359.01 | 115.60 / 103.40 | 757.71 / 706.20 | 670 / 598 |
| Virtual XYZ, 512³ | 20 | 0.50 | 205.27 / 218.30 | 35.65 / 20.90 | 581.21 / 526.82 | 534 / 358 |

Dense native staging and FFTs remain full-size. The 101-energy case reduced the returned payload from 101.0 to 37.7 MiB and lowered host peak RSS by 123.5 MiB. Its warm time rose by 33.9 ms. The larger native case ran 1.025× faster in paired warm timing and had effectively equal host and GPU peaks. The 20-energy virtual XYZ case at 0.50 nm⁻¹ ran 1.638× faster and saved 176 MiB of peak VRAM. Its unbounded A/A warm ratios spanned 0.999–1.049×.

In the single-energy, one-direction virtual XYZ run on a 256 × 4096² logical grid, the 0.80 nm⁻¹ cutoff lowered peak VRAM by 2,808 MiB (57.5%) while peak host RSS rose by 55.95 MiB (8.5%). The median paired warm speedup was 1.471× with zero mismatches across 4,937,973 checked samples. [4096² run](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM05.s21EA7/matrix-padding/xyz1-4096)

The 40 timed comparisons cover native, Z sum, Z residues and XYZ, with separate host and device returns, one and twelve E directions, and 1, 20 and 101 energies. An additional 22 checks cover mixed soft and hard energies, odd and rectangular grids, XY Hann with rotation masking, identity padding, a singleton-Z plane, and uncapped full-result parity. The 62 comparisons checked 97,385,976 values with zero bit mismatches. [Geometry controls](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-qmax/test-reports/qmax/QM05.s21EA7/geometry)
