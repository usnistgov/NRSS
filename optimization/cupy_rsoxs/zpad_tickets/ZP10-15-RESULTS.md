# Padded scan speed results

The [ZP10–ZP15 packet](ZP10-15-START.md) was installed on 2026-11-22 in `perf/virtual-pad-speed`. The clean local `perf/virtual-z-pad` tip and GitHub branch both resolved to `bdf69e111a0e00d0adb2241067d949648d01e552`. The detached comparator is pinned to that revision. The supplied patch was applied to the two development harness files. [Comparator](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed-parent)

The archive passed its CRC check. All 15 members were extracted without overwriting files. Its SHA-256 is `633915568e17330e00a27e400e3c1026308b6a956c191c2cc4b9788cb09c8796`. [Archive](/resdata/DeLongchamp/NRSS-opt-data/NRSS_Padded_Scan_Speed_Tickets_20261122.zip)

## ZP10 startup

| Experiment | State | Result |
| --- | --- | --- |
| CPU endpoint probe | `screened` | Exact self-check on three geometries, including extra bins from gapped energy support. The 101-energy fixture has 412,705 valid columns, 31,128 bit-identical FP32 squared radii, 80,541,970 endpoint visits, and a 2,017,475-bin exact union including DC repair. A contiguous envelope adds 64 bins. [Self-test](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP10-start.xBRNhw/probe-self-test.log) · [Counts](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP10-start.xBRNhw/radius-count.json) |
| CPU endpoint mixing and factored XY | `screened` | Shared endpoint loads gave identical outputs for all three channels in native energy order. The 64 signed complex128 comparisons against a literal 3D FFT gave a maximum peak-normalized error of `1.46e-14`. [Endpoint mixing](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP10-start.xBRNhw/shared-endpoint-mixing.log) · [Factored XY](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP10-start.xBRNhw/factored-xy.log) |
| CPU smoke | `screened` | 132 passed, 7 skipped, 104 deselected in 4.59 s with CUDA hidden. [Pytest](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP10-start.xBRNhw/cpu-smoke.log) |
| Generic XYZ harness | `screened` | Two-material nonseparable orientation fixture, 20 energies from 280 to 289.5 eV, one direction, material 2 exterior, host detector. Both arms used the same 9 × 33 × 35 source and 256 × 512 × 512 logical geometry. All 20 full panels matched exactly across checkouts. [A/B plan](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP10-start.xBRNhw/generic-ab-plan.json) · [A/A plan](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP10-start.xBRNhw/generic-aa-plan.json) · [GPU log](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP10-start.xBRNhw/generic.log) |

The generic A/A comparison alternated worker order over five pairs with one warmup per worker. The two arm medians were 58.75 and 65.61 ms, including the harness construction boundary. Median absolute within-pair deviation was 10.46%. A separate 10 ms NVML pass recorded 852 MiB peak per process in both arms, with 48,266 MiB available at launch and 554 MiB peak host RSS. Instrumented runs recorded one basis build, 1,151,123 union endpoints, and ten nine-plane XY FFT batches per call. [A/A timings](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP10-start.xBRNhw/generic-aa/20261122T162903Z-energy20-gbfuhz3b/summary.json) · [A stage trace](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP10-start.xBRNhw/generic-stages/20261122T162928Z-energy20-s1b320n0/r01-A.result.json) · [A memory](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP10-start.xBRNhw/generic-memory/20261122T162933Z-energy20-pqsrj2w4/r01-A.result.json)

The frozen ZP09 vesicle manifest resolves to a 52³ occupied crop at 1.5 nm, with head, tail, and water as the three materials. The crop and optical-source SHA-256 checks passed. The optical NPZ matches the manifest's scan energies and the 287.55 eV values saved in the occupied crop. [Frozen fixture](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-vesicle-energies.vrbPPq/fixture.json)

The wide 250/285/570/850 eV CPU control at 1.5 nm reproduces 5,061,138 excess envelope bins on the ticket's 256 × 512 × 512 grid. At the vesicle's 256 × 1024 × 1024 grid the same energies would add 20,299,536 bins. [512² count](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP10-start.xBRNhw/radius-wide-512-count.json) · [1024² count](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP10-start.xBRNhw/radius-wide-count.json)

## ZP10 frozen vesicle timing

**Screened.** Three matched worker pairs in alternating order on Wheatley's RTX A6000 with Python 3.14.3 and CuPy 14.0.1. Each worker timed its first call and five subsequent calls. The table takes the median of the worker medians for after-first timings. Each timing synchronizes around `Morphology.run` after construction. Each call returns a host FP32 detector at 256 × 1024 × 1024 logical ZYX, with one electric-field direction and water fill. The 1-, 20-, and 101-energy runs reproduced every intensity value, finite mask, and detector coordinate in the corresponding subset of the archived virtual 101-energy detector in both arms. [Benchmark runner](../../../tests/validation/dev/cupy_rsoxs_virtual_3dpad/run_padded_scan_baseline.py) · [Wheatley commands](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP10-start.xBRNhw/run-frozen.sh) · [Per-worker results](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP10-start.xBRNhw/frozen) · [Archived detector](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-vesicle-energies.vrbPPq/virtual-101-chunk1-detector.npy)

| Energies | Default first (ms) | Default warm (ms) | Compact-pinned first (ms) | Compact-pinned warm (ms) | Median of paired warm ratios (default / pinned) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 168.35 | 20.53 | 180.32 | 20.52 | 0.998× |
| 20 | 295.54 | 87.00 | 292.07 | 87.12 | 1.000× |
| 101 | 728.36 | 360.26 | 727.90 | 360.01 | 1.003× |

The default reproduces the prior warm medians of 20.5, 86.5 and 361.4 ms. Its first call after the initial run takes 20.81, 130.55 and 560.92 ms respectively. The five-call distributions preserve that transient. Both paths build the same seven-field basis once per public call and execute `virtual_3d_sum`. Per call, 28 XY FFT batches transform 364 source planes across the seven fields at every energy count. The endpoint union grows from 797,371 to 1,937,851 to 2,017,475 bins as the scan grows. Corresponding basis sizes including endpoint indices are 48.7, 118.3, and 123.2 MiB. [Earlier results](ZP05-09-RESULTS.md)

A separate default worker collected each instrumented call after six uninstrumented calls. The ranges below can nest or overlap. Detector projection is the existing D range, which includes geometry and mixing. The GPU copy range reports asynchronous device-to-host transfer. [1 energy](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP10-start.xBRNhw/frozen/1-default-stages-warmed-A.json) · [20 energies](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP10-start.xBRNhw/frozen/20-default-stages-warmed-A.json) · [101 energies](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP10-start.xBRNhw/frozen/101-default-stages-warmed-A.json)

| Energies | Wall (ms) | Endpoint planning (ms) | XY FFT (ms) | Axial accumulation (ms) | Detector projection (ms) | D2H (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 21.94 | 4.87 | 12.38 | 2.07 | 0.35 | 0.32 |
| 20 | 87.44 | 59.29 | 12.19 | 3.50 | 9.54 | 6.48 |
| 101 | 361.09 | 290.82 | 12.25 | 3.58 | 48.65 | 32.58 |

Endpoint planning increased by 231.53 ms over the 20-to-101-energy step, the largest observed change among these ranges. The planner range covers the per-energy endpoint union and the index work that follows it. ZP10 still needs counters and timers to quantify geometry, sorting, and merging within that range.

## ZP10 sphere resolution screen

**Screened.** On the same RTX A6000, the clean parent reran the established 128³ sphere at 1.0 nm and 285 eV, using one electric-field direction and a complete host detector. Times are medians of five warm synchronized calls and include the sphere runner's xarray return. VRAM is the sampled per-process peak from a separate run. Every case passed the discrete sphere anchors. The worst peak-normalized anchor error across the eight cases was `9.79e-7`. [Vacuum](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP10-start.xBRNhw/resolution-vacuum/20261122T163124Z-s_jus36e/summary.json) · [Material 2](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP10-start.xBRNhw/resolution-2/20261122T163217Z-u6ll4j9e/summary.json)

| Logical cubic size | Vacuum warm (ms) | Material 2 warm (ms) | Peak VRAM, either fill (MiB) |
| ---: | ---: | ---: | ---: |
| 1024 | 18.65 | 17.93 | 476 |
| 2048 | 58.50 | 58.27 | 584 |
| 4096 | 275.42 | 274.41 | 1,076 |
| 8192 | 1,214.60 | 1,215.21 | 3,428 |

On the separate instrumented vacuum calls at 4096³ and 8192³, XY FFT time grew from 191.57 to 831.65 ms and axial accumulation from 54.71 to 216.74 ms. Both sizes transformed one plane per batch for 256 FFT calls across two fields. The basis grew from 127.5 to 510.0 MiB. These first-call traces include setup. [4096 stages](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP10-start.xBRNhw/resolution-vacuum/20261122T163124Z-s_jus36e/M4096/accuracy.json) · [8192 stages](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP10-start.xBRNhw/resolution-vacuum/20261122T163124Z-s_jus36e/M8192/accuracy.json)

Dean accepted the ZP10 incumbent baseline on 2026-11-22. The optimization comparisons carry forward the additional output residency, native reduction, geometry, and scaling controls as relevant to each candidate.

## ZP11 direct endpoint addressing

Three private XYZ planner arms use the same scalar projection and 3D evaluator. The legacy arm merges endpoint IDs on each energy. The bitmap arm collects the exact union in one bit per logical bin, with a 128 MiB bitmap workspace limit, then sorts the emitted IDs with the DC stencil once. Both use the same sorted-ID search. The span arm groups basis rows by detector column and uses a 64-bit prefix offset plus the axial bin within the span. The span includes the exact DC stencil. Its cost screen declines if the estimated additional coefficient and ID storage exceeds 32 MiB or one eighth of the detector endpoint count with an allowance for the stencil. The public route continues to select the legacy planner during these experiments. [Source fingerprints](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP11.sNjywh/source.sha256)

The three arms each reproduced the matching archived detector values, finite masks, and registered q coordinates for the 1-, 20-, and 101-energy vesicle. The 101-energy span evaluates 2,017,539 bins, 64 more than the 2,017,475-bin exact union. The 20-point subset has 24,204 additional bins. [Full-panel checks and stage traces](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP11.sNjywh/pilot) · [101-energy repeat on the final source](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP11.sNjywh/final-parity)

The stage traces split planning geometry, endpoint merging, projection geometry, and the scalar projection kernel. On the 101-energy vesicle, the legacy endpoint merge took 310.9 ms with the added counters enabled. The bitmap and span merges took 4.6 and 4.8 ms. The scalar projection kernel took 27.35 ms for legacy and 25.77 ms for spans. The instrumented ranges can nest or overlap. The timing comparisons use a separate uninstrumented run.

**Radius table screened.** A separate geometry-only check on Wheatley's Quadro RTX 8000 compared every validity bit, lower bin, upper bin, and interpolation fraction across all 101 energies. The incumbent geometry took 29.09 ms. A fused per-column geometry kernel took 3.86 ms. Building the table of 31,128 exact squared-radius keys took 2.94 ms, and expanding per-energy geometry took another 4.02 ms. The persistent table and column map within a call occupied 43.0 MiB. These measurements end at the geometry arrays and leave projection for an end-to-end comparison. [Radius screen](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP11.sNjywh/radius.json)

The frozen vesicle benchmark used three fresh workers per arm in alternating order on the RTX A6000, with five synchronized warm calls per worker. The table reports medians across workers. VRAM is the sampled peak per worker in a separate NVML pass with 10 ms sampling. [Timings](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP11.sNjywh/timing) · [VRAM traces](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP11.sNjywh/memory)

| Energies | Legacy first / warm (ms) | Bitmap first / warm (ms) | Spans first / warm (ms) | Peak VRAM legacy / bitmap / spans (MiB) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 180.75 / 20.55 | 186.41 / 18.20 | 170.91 / 19.91 | 526 / 536 / 546 |
| 20 | 295.64 / 87.10 | 260.69 / 35.42 | 238.50 / 35.71 | 882 / 660 / 680 |
| 101 | 734.40 / 359.69 | 478.59 / 97.92 | 455.15 / 98.15 | 986 / 664 / 682 |

The exact bitmap is a `keep` candidate for eligible grids. It achieved a 3.66× median paired speedup and used 322 MiB less GPU memory at 101 energies. The direct span plan is `conditional` for ZP12's scalar control. Its 101-energy warm timing matches the bitmap within run-to-run variation, with 18 MiB more peak VRAM. The geometry-only radius table is a slower option in the measured build-and-expand comparison. Retain the fused per-column geometry for a separate end-to-end ablation.

The rotated wide scan used a 256 × 512 × 512 target, 12 directions, XY Hann, an oriented fill, out-of-order energies at 850, 250, 570, and 285 eV, and a duplicate 250 eV panel. All three arms produced identical values, coordinates, order, and masks. The exact bitmap has 1,325,339 bins. The candidate span would add 5,061,138 bins and fell back to sparse addressing. The CPU smoke had 132 passes and seven skips. The focused GPU suite had 95 passes and checks independent endpoint enumeration, one-plane geometry, signed coefficients and DC repair, identity, and native reduced output. [Wide scan](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP11.sNjywh/wide.json) · [CPU smoke](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP11.sNjywh/cpu-smoke-final.log) · [GPU smoke](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP11.sNjywh/gpu-smoke.log)

The sphere screen on the RTX A6000 used a complete host panel and five warm calls per worker. Both plans reproduced the discrete sphere anchors at 4096³ and 8192³. The bitmap's workspace cap excludes these sizes. The span metadata raises peak VRAM by 29% at 4096³ and 36% at 8192³ for about 5% lower wall time. Carry the legacy plan forward as the large-grid control for ZP13. [Vacuum legacy](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP11.sNjywh/sphere-vacuum-legacy/20261122T164429Z-4_ua6mfq/summary.json) · [Vacuum spans](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP11.sNjywh/sphere-vacuum-spans/20261122T164507Z-osp2igd_/summary.json) · [Material 2 legacy](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP11.sNjywh/sphere-2-legacy/20261122T164620Z-lmdykj7t/summary.json) · [Material 2 spans](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP11.sNjywh/sphere-2-spans/20261122T164544Z-evjk8msy/summary.json)

| Logical size and fill | Legacy warm (ms) | Span warm (ms) | Legacy / span peak VRAM (MiB) |
| --- | ---: | ---: | ---: |
| 4096³ vacuum | 275.05 | 261.07 | 1,076 / 1,388 |
| 4096³ material 2 | 275.29 | 261.71 | 1,076 / 1,388 |
| 8192³ vacuum | 1,214.31 | 1,156.59 | 3,428 / 4,666 |
| 8192³ material 2 | 1,217.71 | 1,156.41 | 3,428 / 4,666 |

## ZP12 projection-only screen

**Screened.** The CUDA prototype projects the full 101-energy frozen vesicle from the real seven-field packed basis, with one electric-field direction, on Wheatley's Quadro RTX 8000. To isolate reuse, the reload and share arms use the same two-endpoint tensor contraction. The share arm retains complex samples only when their endpoint positions match across energies. Geometry is prepared before timing. Each timed pass includes launches for the full scan and terminal stream synchronization. The three-stage comparison is the incumbent-style separate-endpoint contraction at width 1, the paired-endpoint contraction with fresh loads, and the paired-endpoint contraction that can share loads. [CUDA screen](../../../tests/validation/dev/cupy_rsoxs_virtual_3dpad/screen_energy_tile.py) · [Raw timings and errors](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP11.sNjywh/zp12-screen.json)

| Energy tile width | Reload every energy (ms) | Share matching loads (ms) |
| ---: | ---: | ---: |
| 1 | 20.83 | 20.80 |
| 2 | 19.76 | 11.95 |
| 4 | 19.14 | 7.76 |
| 8 | 18.92 | 5.84 |

The separate-endpoint contraction took 32.51 ms at width 1. At width 8 the scan has a five-energy final tile, and sharing was 3.24× faster than reloading at the same width. CUDA reports 72 registers per thread for reloading and 96 for sharing, with zero local bytes per thread in both cases. Every variant matched the local scalar finite masks, with worst peak-normalized absolute error `1.18e-7` and relative L2 `5.08e-8` across all 101 panels. The local scalar reference differed from the earlier RTX A6000 archive by up to `8.97e-7` of a panel's peak. The next ZP12 checkpoint measures bounded per-tile geometry and the actual public result pipeline, including multi-angle channels, native reduction, and host and device output.

## ZP12 public XYZ pipeline

The private projector integrates sorted-ID and span addressing with `Morphology.run`. The tiled kernels calculate each energy's Ewald geometry in registers and reuse the scalar dipole projection. Tile width controls the number of coefficient planes made together, separately from the result-copy chunk size. The per-tile output cap is 96 MiB. When a tile allocation fails, scalar projection resumes from the first unprocessed energy. The host stream, rotation, and native reduction continue using the existing result code. [Backend source fingerprints](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP12.qt0GOw/source.sha256)

The RTX A6000 pilot checked all 101 complete one-direction detector panels at widths 1, 2, 4, and 8, for both addressing formats and with and without Fourier sample reuse. Coordinates and finite masks agreed with the archive. Worst peak-normalized error was `1.18e-7`. Span width 8 reduced the projector range from 12.25 to 3.38 ms with sharing. The same scan spent about 34 ms on device-to-host copies. Span width 2 and width 8 had warm end-to-end times of 84.29 and 84.50 ms in this pilot. The main interleaved comparison used width 2. [Pilot profiles](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP12.qt0GOw/pilot) · [Width selection](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP12.qt0GOw/selection.json)

The rolling two-position cache requests 563,793,790 complex basis samples on the frozen 101-energy scan at width 1 and 80,182,802 at width 8. These counts enumerate exact FP32 bins with the same cache rule. The wide 850/250/570/285/250 eV scan needs almost all of the original loads, a 1.03× reduction at width 8. On the Quadro RTX 8000, the wide test used a noninteger 255 × 512 × 517 target, XY Hann, an oriented fill, and both one and 12 directions. Every method preserved the duplicate, order, q coordinates, and masks. The fastest wide results used the fused width-1 reload kernel at 10.05 and 13.44 ms. Width-8 sharing took 10.34 and 14.46 ms. The register and local-storage counts come from CUDA kernel attributes. Nsight hardware cache and DRAM counters were unavailable on Wheatley. [Requested loads](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP12.qt0GOw/load-counts.json) · [Wide scan](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP12.qt0GOw/wide.json)

CPU smoke passed 132 tests with seven skips. GPU padding smoke passed all 100 tests, including host and device output, repeated energies, one or 12 directions, rotated native I/A, and a forced allocation failure after a completed tile. [CPU smoke](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP12.qt0GOw/cpu-smoke.log) · [GPU smoke](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP12.qt0GOw/gpu-smoke.log)

The 27 full-pipeline checks covered 1, 20, and 101 energies, one or 12 directions, the 101-energy non-monotone order, host chunks 1 and 4, retained device detectors, and native I/A. Worst detector peak-normalized error was `2.80e-7` and relative L2 was `5.68e-8`, with exact coordinates and finite-pixel masks. The native results also reproduced all coordinates and masks. The largest absolute anisotropy error in the 0.06–0.80 nm⁻¹ band was `1.62e-8`. [Full-pipeline checks](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP12.qt0GOw/accuracy)

The primary timing panel uses the one-direction vesicle, a complete host detector with copy chunk 1, and three interleaved fresh workers per arm with five warm calls each. A uses the pinned `bdf69e1` incumbent. B uses the exact bitmap and scalar projection. C and D both use spans with fused geometry at width 2. C reloads and D shares Fourier samples. Times include the synchronized `Morphology.run` call after construction. [Paired timings](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP12.qt0GOw/timing) · [NVML memory passes](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP12.qt0GOw/memory)

| Energies | A warm (ms) | B warm (ms) | C warm (ms) | D warm (ms) | Median paired A/D speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20.55 | 18.21 | 19.22 | 19.28 | 1.07× |
| 20 | 86.56 | 35.53 | 33.00 | 32.84 | 2.64× |
| 101 | 360.76 | 113.07 | 84.56 | 84.65 | 4.28× |

The 101-energy bitmap workers ranged from 97.99 to 113.20 ms. Reloading and sharing at width 2 were effectively tied for host detector delivery. The 20-to-101 marginal times for A through D were 3.385, 0.957, 0.637, and 0.640 ms per added energy. At 101 energies the first-call medians were 728.97, 503.27, 451.20, and 455.32 ms, and the peak GPU memory samples were 992, 664, 682, and 682 MiB. The profile recorded one basis build and 28 XY FFT batches per call for one direction and 36 for 12 directions.

The width-8 screen holds span addressing, fused geometry, and the requested copy chunk of 4 fixed. Each entry is a separate warm worker on the RTX A6000. All variants also checked absolute intensity and masks against the archive. Both tiled variants processed 12 full tiles and a five-energy tail. Width-8 sharing uses 80 registers per thread versus 53 for reloading, with zero local bytes per thread in either kernel. [Width-8 screens](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP12.qt0GOw/layout-widths) · [Combined data](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP12.qt0GOw/summary.json)

| 101-energy result | Reload warm (ms) | Share warm (ms) | Share speedup |
| --- | ---: | ---: | ---: |
| Host detector, chunk 4 | 92.94 | 82.57 | 1.13× |
| Retained device detector | 62.38 | 53.95 | 1.16× |
| Native host I/A, chunk 4 | 151.83 | 149.44 | 1.02× |

**Keep** fused geometry and the paired-endpoint projector for the combined experiments. **Conditional** for shared reads on narrow scans, with replicated dispatch and width-8 memory evidence at the promotion gate. The 101-energy width-2 peak VRAM is 682 MiB for a host detector and 1,490 MiB for a retained device detector. The width-8 detector uses a 32 MiB output tile. Single-energy and wide runs retain the bitmap or fused-width-1 controls.

## ZP13 zero optical weight screen

**Screened.** A development worker left the vacuum sphere's input materials and XYZ geometry intact and excluded the exactly zero optical material from Fourier basis construction. The contributing field retained the existing compact FFT and axial evaluator. A matched Quadro RTX 8000 comparison used five warm synchronized calls per worker and separate 10 ms NVML passes. Each optimized result matched its complete incumbent detector, including the finite mask, exactly. This screen covers the zero-weight vacuum case. [Prototype and campaign](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP12.qt0GOw/screen-zero-optics.py) · [Timings and anchors](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP12.qt0GOw/zp13-screen/summary.json)

| Logical size | Incumbent warm (ms) | Skip zero field (ms) | XY FFT calls before / after | Peak VRAM before / after (MiB) |
| ---: | ---: | ---: | ---: | ---: |
| 4096³ | 343.45 | 198.99 | 256 / 128 | 1,044 / 1,042 |
| 8192³ | 1,348.63 | 775.26 | 256 / 128 | 3,588 / 3,586 |

## ZP13 XY batch depth screen

**Screened.** The material-2 sphere at 8192³ used both contributing basis fields, one direction, and the full host detector. The Quadro RTX 8000 processed the requested depth at all four settings and reproduced the complete detector values, coordinates, masks, and sphere anchors. More planes per batch reduced axial accumulation time from 262 ms at depth 1 to 96 ms at depth 8. XY FFT time stayed between 897 and 903 ms. Depth 2 is the smaller memory tradeoff for later dispatch tests. Depth 8 added 3,072 MiB over depth 4 for 8.81 ms of warm time. [Batch-depth worker](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP12.qt0GOw/screen-xy-batches.py) · [Timings and full-panel comparisons](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP12.qt0GOw/zp13-xy/summary.json)

| XY planes per batch | XY FFT calls | Warm (ms) | Peak VRAM (MiB) |
| ---: | ---: | ---: | ---: |
| 1 | 256 | 1,351.97 | 3,588 |
| 2 | 128 | 1,236.20 | 4,356 |
| 4 | 64 | 1,186.60 | 5,892 |
| 8 | 32 | 1,177.79 | 8,964 |

## ZP13 exact optical row pruning

**Keep** the private row-level implementation for combined qualification. The compact XYZ builder assembles the same FP32 optical coefficients before planning its basis. A row is omitted if its coefficient is exactly zero for every requested energy and active component. The check uses no near-zero threshold. Before dropping a row the builder checks the staged fields needed for that row. The incumbent row set is used if this check encounters unbounded or nonfinite input. The all-zero case produces the same zero intensity and detector mask with zero FFTs. [Source fingerprints](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP13.zBJuws/source.sha256)

Tests used an oriented inclusion with nonzero scattering at 340 eV and zero contrast at 285 eV, isotropic optical constants on an oriented field, an all-zero scan, and an unvalidated nonfinite field. Every output and coordinate matched the incumbent exactly. The mixed-energy case retained the later contributing rows. The isotropic optical case omitted seven rows and retained two. CPU smoke passed 132 tests with seven skips. GPU padding smoke passed all 104 tests. A separate coefficient layout check reproduced the original tensor and direct coefficient arrays exactly. [Focused cases](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP13.zBJuws/prune-smoke.log) · [CPU smoke](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP13.zBJuws/cpu-smoke.log) · [GPU smoke](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP13.zBJuws/gpu-smoke.log) · [Coefficient layout](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP13.zBJuws/coefficient-layout.log)

The 128³ sphere used five synchronized warm calls and separate per-process NVML observations. All eight complete detectors matched their controls exactly. The vacuum ladder used the RTX A6000. The material-2 ladder used the Quadro RTX 8000 for both arms. The vacuum run omitted one row at each size and halved its XY FFT batches. The material-2 run retained both rows. [Vacuum timings and full-panel comparison](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP13.zBJuws/sphere-vacuum-comparison.json) · [Material-2 timings and full-panel comparison](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP13.zBJuws/sphere-2-comparison.json)

| Logical size | Vacuum incumbent / pruned (ms) | Vacuum incumbent / pruned peak VRAM (MiB) | Material 2 incumbent / checked (ms) |
| ---: | ---: | ---: | ---: |
| 1024³ | 17.89 / 13.91 | 476 / 468 | 21.72 / 21.65 |
| 2048³ | 58.81 / 38.50 | 584 / 584 | 74.61 / 74.18 |
| 4096³ | 274.58 / 163.05 | 1,076 / 1,010 | 343.44 / 344.25 |
| 8192³ | 1,214.46 / 700.34 | 3,428 / 3,170 | 1,350.37 / 1,349.14 |

The 52³ vesicle kept every row across all six independent XY and Z checks. The 20-energy 256 × 2048 × 2048 full detectors matched exactly in one and 12 directions, as did the single-energy 1024 and 4096 lateral checkpoints at Z = 256 and the 1024-depth checkpoint at fixed 1024². Median before / after warm times at the 20-energy 2048² checkpoint were 389.12 / 389.41 ms and 475.44 / 477.93 ms on the Quadro RTX 8000. [Vesicle controls](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP13.zBJuws/cross)

## ZP14 selected-column X/Y FFTs

**Screened.** The private candidate transforms the occupied Y rows in X, packs the required canonical X bins contiguously, and transforms those bins in Y. It uses the same Z accumulation, finite exterior, placement phase, Hermitian reflection, and DC repair. The signed complex128 literal checks passed at two mixed-parity rectangular targets, including XY Hann and nonzero fill. Complete detector checks also passed with vacuum and oriented fill. [Signed tensor checks](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP13.zBJuws/zp14-smoke.log) · [Candidate source fingerprint](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP13.zBJuws/zp14-source.sha256)

The RTX A6000 pilot used a complete host panel and five warm calls per worker. Both vacuum arms used the verified ZP13 row pruning. At 2048³ the split path transforms 471 of the 1,025 native X columns in Y. The compared detector coordinates and finite masks match. The largest absolute difference divided by the control peak was `5.50e-7`. [1024 and 2048 full-panel comparisons](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP13.zBJuws/zp14-comparison.json)

| Logical size and fill | Full XY warm (ms) | Split XY warm (ms) | Full / split peak VRAM (MiB) |
| --- | ---: | ---: | ---: |
| 1024³ vacuum | 13.91 | 11.69 | 468 / 408 |
| 1024³ material 2 | 17.92 | 13.09 | 476 / 416 |
| 2048³ vacuum | 38.32 | 25.65 | 584 / 484 |
| 2048³ material 2 | 58.82 | 32.74 | 584 / 514 |

The high-resolution controls used the RTX A6000 for the vacuum sphere and the Quadro RTX 8000 for the material-2 sphere. Each row compares complete host results on the same GPU. At 8192³ the split path transformed 1,884 of 4,097 X columns in Y. All full finite masks and coordinates matched. Worst detector peak-normalized difference across the high-resolution cases was `9.66e-7`, and worst relative L2 was `2.17e-7`. [4096 and 8192 full-panel comparisons](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP13.zBJuws/zp14-large-comparison.json)

| Logical size and fill | Full XY warm (ms) | Split XY warm (ms) | Full / split peak VRAM (MiB) |
| --- | ---: | ---: | ---: |
| 4096³ vacuum with ZP13 pruning | 163.26 | 103.85 | 1,010 / 1,010 |
| 8192³ vacuum with ZP13 pruning | 698.47 | 388.39 | 3,170 / 3,170 |
| 4096³ material 2 | 343.10 | 169.23 | 1,044 / 976 |
| 8192³ material 2 | 1,348.82 | 642.51 | 3,588 / 3,328 |

The staged source also passed the frozen vesicle on the RTX A6000. At 256 × 1024 × 1024 it cut the single-energy detector from 20.54 to 15.55 ms, and the 20-energy 12-direction detector from 104.94 to 101.98 ms. On the 257 × 640 × 768 grid, every one of the 20 panels matched exactly and warm runtime moved from 50.61 to 49.76 ms. The largest peak-normalized vesicle error was `4.67e-7`. The six focused integrated checks and CPU smoke passed. [Vesicle comparisons](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP13.zBJuws/zp14-vesicle) · [Integrated GPU checks](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP13.zBJuws/integrated-smoke.log) · [Integrated CPU smoke](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP13.zBJuws/cpu-smoke-integrated.log)

**Keep** the selected-column candidate for combined qualification. The integer-ratio residue comparison and combined dispatch measurements are the next checkpoints.

## ZP15 automatic XYZ dispatch

The qualification compares the pinned parent `bdf69e1` with candidate tree `f062764a564d62508f1ab3ef2cdacc14c1ec9da2`. Private benchmark profiles isolate endpoint/projection changes and resolution changes. A standard `Morphology.run` call uses automatic selection for actual compact-tensor XYZ extensions. [Source fingerprints](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/qualification-source.sha256) · [Benchmark environment](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/environment.sh)

- The exact bitmap planner is considered within a 128 MiB bitmap cap and available workspace. Short, densely sampled energy ranges can use column spans subject to their carrying cost. The fallback order is spans, bitmap, then the sparse union. The planner report records the layout actually used.
- The fused Ewald projector is considered for multiple energies within the supported field, geometry, and workspace bounds. Densely sampled detector results can share loads in width-eight tiles with device residency or host copy chunks of four or more. Other dense scans use width-two reloading. Wide scans use width-one reloading. The separate host-copy chunk and the requested energy order remain the public result policy.
- XYZ elides exactly zero optical basis rows after checking the fields involved. For large grids and strong lateral extension, split X/Y FFTs are considered when the required transverse support occupies at most three quarters of the native X half-plane. The retry path decreases batch depth and can replay a field with full-plane FFTs if the split buffer cannot fit.

The all-bin signed Fourier oracle explicitly requests the reference endpoint planner. The endpoint-format unit test uses a scalar projection in each arm. These selectors retain the original numerical checks. The new resource regression injects both a bitmap allocation failure and a split-FFT failure. The focused test gate passed all 116 cases. [Focused gate](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/short-gate.log)

| Default panel | Parent | Candidate |
| --- | --- | --- |
| CPU smoke | 132 passed, 7 skipped | 132 passed, 7 skipped |
| Compact GPU smoke | 93 passed, 10 skipped, 1 failed | 110 passed, 10 skipped, 1 failed |
| Compact physics | 23 passed, 2 failed | 23 passed, 2 failed |

The same three installed PyHyperScattering checks fail in the parent and candidate. They are the integrated smoke test and the 70 nm and 128 nm sphere form factors. The CoreShell, MWCNT, Bragg registration, disk, and orientational contrast cases pass on both. [Parent report](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/report-parent/20261122T171755Z/summary.md) · [Candidate report](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/report-candidate/20261122T172133Z/summary.md)

The forced report has the same test dispositions as the default candidate. The audit recorded 215 successful literal comparisons in 39 pytest cases and 17 processes, with 215 exclusions logged by reason. The 109 XYZ and 106 material-filled Z evaluations include both padding routes through the new small CoreShell direct-polarization comparison, accounting for the two additions since ZP09. Maximum relative L2 was `5.50e-6` and maximum peak-normalized detector error was `6.25e-6` across all results. All 109 XYZ replays used split FFTs, 96 pruned at least one row, 64 used bitmap lookup, 45 used spans, and 48 executed shared tiles on more than one energy. The site bootstrap captured inherited physics subprocesses. [Forced report](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/report-forced/20261122T172223Z/summary.md) · [Per-case outcomes and exclusions](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/forced.jsonl)

### Vesicle performance

The four ablation arms are A pinned parent, B endpoint and projection selection, C optical-row pruning and split FFTs, and D automatic combined selection. The input is the original 52³ crop in a 256 × 1024 × 1024 grid at 1.5 nm pitch, with one direction and water fill. All arms return a complete host detector with copy chunk 1. Three interleaved pairs per energy count each timed the first call and five warm calls on the RTX A6000. [Paired worker results](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/vesicle)

| Energies | A first / warm (ms) | B first / warm (ms) | C first / warm (ms) | D first / warm (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 175.03 / 20.60 | 195.83 / 18.29 | 215.69 / 15.39 | 217.29 / 13.11 |
| 20 | 293.32 / 87.00 | 247.65 / 33.62 | 340.35 / 85.73 | 264.74 / 29.67 |
| 101 | 786.26 / 359.90 | 517.03 / 85.05 | 841.54 / 358.23 | 536.72 / 80.81 |

The single-energy case gains about 7.5 ms per warm call and incurs about 42 ms more on a fresh worker's first call. Paired warm speedups were 1.57×, 2.93× and 4.45× for 1, 20 and 101 energies. The marginal warm cost across 20 to 101 energies fell from 3.37 to 0.63 ms per additional energy. The main timing and parity arms finished before the benchmark harness hit a duplicated reference-capture assertion on the 2048² case. Reference capture now uses the command-line validation of the legacy scalar profile. The mixed case and memory passes resumed with the same backend source. [Capture diagnostic](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/vesicle.log) · [Updated harness fingerprint](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/resume-harness.sha256)

| Control | Parent / automatic warm (ms) | Parent / automatic peak VRAM (MiB) |
| --- | ---: | ---: |
| 1 energy, host detector | 20.60 / 13.11 | 526 / 498 |
| 20 energies, host detector | 87.00 / 29.67 | 882 / 642 |
| 101 energies, host detector | 359.90 / 80.81 | 980 / 644 |
| 20 energies, 12 directions, host detector | 104.15 / 42.61 | 912 / 688 |
| 101 energies, host detector, chunk 4 | 425.16 / 79.27 | |
| 101 energies, permuted request order | 428.48 / 79.65 | |
| 101 energies, retained device detector | 420.67 / 50.28 | 1,788 / 1,452 |
| 101 energies, native host I/A, chunk 4 | 454.07 / 153.87 | 988 / 652 |
| 20 energies, 12 directions, 2048² host detector | 394.16 / 165.44 | 1,620 / 1,606 |

The first three timing rows use the paired warm medians. The remaining rows use instrumented worker medians. Peak VRAM comes from separate per-process NVML passes. All 101 detector panels, the requested order, coordinates, and finite masks matched at the packet tolerances. The largest vesicle detector error divided by its panel peak was `9.73e-7`. Native host I/A preserved its axes and support, with maximum anisotropy error `2.82e-7` in the fit band. [Control workers](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/vesicle) · [NVML workers](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/vesicle-memory)

The 101-energy host scan retained seven Fourier rows. The packed layout increased the endpoint count from 2,017,475 to 2,017,539, while reducing the planning stage from 289.86 to 33.77 ms. Splitting the XY FFT doubled the transform call count from 28 to 56 and cut total XY transform time from 12.27 to 6.07 ms. XY depth stayed at 16 cropped planes per batch. Scalar projection on the same optimized basis took 92.05 ms for host chunk 4 and 89.47 ms for the retained device detector, compared with 79.27 and 50.28 ms using shared width-eight tiles. The device and host-chunk-four projectors each used a 32 MiB tile. The full host result still transfers 423,624,704 bytes. Device retention transfers no detector bytes, and native I/A transfers 1,171,600 result bytes. On the automatic route, planning and host transfer dominate the host detector, planning dominates the retained detector, and native reduction dominates I/A. [Stage and output byte counts](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/qualification-results.json)

### Sphere resolution and independent reference

The Quadro RTX 8000 workers used the fixed 128³ sphere crop, 1 nm pitch, one energy, and complete host detectors. Each timing and memory worker has a separate accuracy worker at the same logical resolution. Independent CPU voxel-DTFT panels were generated once per fill and supplied to the established accuracy worker’s full-voxel comparison. The worker continued to apply its original sphere, analytic, and high-q checks. All 14 parent and 14 automatic detectors passed those checks. [Vacuum workers](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/sphere-vacuum-D) · [Material-2 workers](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/sphere-2-D) · [CPU fixture provenance](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/voxel-hooks/sitecustomize.py)

| Logical grid | Vacuum warm (ms) | Vacuum peak VRAM (MiB) | Material 2 warm (ms) | Material 2 peak VRAM (MiB) |
| ---: | ---: | ---: | ---: | ---: |
| 128³ | 6.40 / 6.47 | 212 / 212 | 6.31 / 6.50 | 212 / 212 |
| 256³ | 7.29 / 7.85 | 284 / 288 | 7.45 / 7.04 | 284 / 286 |
| 512³ | 9.90 / 8.99 | 356 / 368 | 9.84 / 9.66 | 356 / 368 |
| 1024³ | 21.60 / 12.02 | 378 / 370 | 21.48 / 14.35 | 378 / 370 |
| 2048³ | 74.18 / 29.54 | 486 / 386 | 74.32 / 38.38 | 486 / 416 |
| 4096³ | 343.27 / 112.08 | 1,044 / 912 | 343.51 / 169.72 | 1,044 / 976 |
| 8192³ | 1,351.77 / 422.45 | 3,588 / 3,072 | 1,347.76 / 642.37 | 3,588 / 3,328 |

Each cell is parent / automatic. At 8192³ both fills have 11,140,041 valid pixels. Throughput rose from 8.24 to 26.37 million valid pixels/s in vacuum and from 8.27 to 17.34 million in material 2. Warm ranges were 1,348–1,353 / 421–424 ms in vacuum and 1,348–1,349 / 641–645 ms in material 2. Peak host RSS rose from about 1,065 to 1,105 MiB for each fill. The sphere’s first minimum for both fills was `0.128397 nm⁻¹` after quadratic refinement, with analytic value `0.128383 nm⁻¹`. The largest parent-relative detector peak error across the ladder was `6.27e-7`. Maximum full-voxel peak error was `1.02e-6`, and maximum full-voxel high-q relative L2 was `7.18e-7`. Analytic band, first-minimum, and absolute anchor measurements are in each accuracy record. [Ladder and error data](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/qualification-results.json) · [8192³ vacuum accuracy](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/sphere-vacuum-D/20261122T173139Z-uafd5y3d/M8192/accuracy.json) · [8192³ material accuracy](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/sphere-2-D/20261122T173311Z-7pcjk18y/M8192/accuracy.json)

At 128³ the requested logical extent equals the occupied extent, so the benchmark exercises identity. The 256³ vacuum result showed a small crossover. Reversed pairs on the RTX A6000 measured 7.37 / 7.41 ms for vacuum and 8.21 / 6.96 ms for material 2 at 256³. At 512³ the same worker blocks measured 8.97 / 8.08 ms and 9.35 / 8.68 ms. Median first calls took 21–28 ms longer in these four small-grid cases. The 256–512³ Quadro samples added 2–12 MiB of peak VRAM. [Interleaved small-grid workers](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/small-rungs)

### Regression controls

All three compact reports hashed the same 16 validation inputs, and the candidate and parent had identical NRSS-native results on the 70 and 128 nm sphere tests at their original limits. The three installed external-reducer failures remain visible in the reports. Native 1/20/101 and all four Z-only sum and residue window controls produced bitwise-equal detectors. Early timings varied by process, so a second campaign used reversed parent/candidate pairs and parent/parent controls. The resulting medians were 5.24 / 5.19, 25.64 / 25.56 and 111.83 / 111.74 ms for native runs. The Z-only checks ranged from 30.93 to 39.26 ms on the parent and 30.99 to 38.90 ms on the automatic branch. The explicit three-energy direct R2C output was also bitwise equal. [Native sphere results](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/candidate-native-sphere-theory.json) · [Validation input manifest](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/report-candidate/20261122T172133Z/validation_reference_manifest.tsv) · [Correctness controls](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/aux) · [Warm timing repeats](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/steady)

Each native and Z-only correctness fixture used 274 MiB peak VRAM on both revisions in separate NVML workers. [Regression memory passes](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/memory-controls)

The opt-in direct-polarization report qualifies the shared coefficient builder. Each revision passed 31 GPU smoke cases and 19 physics cases. The parent and candidate skipped 72 and 89 GPU cases respectively, including the tensor-specific additions on the candidate. Both skipped four physics cases and reproduced the same external integrated smoke failure and the two external sphere failures. The five default and opt-in reports used identical reference manifests. [Direct parent](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/direct-parent/20261122T173714Z/summary.md) · [Direct candidate](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/direct-candidate/20261122T173714Z/summary.md) · [Paired verification](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/qualification-supplemental.json)

The wide and permuted four-energy scan with oriented material fill and XY Hann selected bitmap lookup and a fused width-one projector. Warm time was 42.48 / 34.29 ms. The initial benchmark diagnostic used bitwise equality. Rechecking with the ticket’s full-detector criterion gave `8.51e-14` maximum relative L2 and `1.65e-13` maximum peak-normalized error. [Wide scan comparisons](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/aux/wide-parity) · [Wide scan routes](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/aux/wide-stages)

The integer-ratio Y-residue experiment reproduced the needed Fourier coefficients within `2.54e-7` of peak. Its median FFT times were 213, 167, 165 and 585 µs at 1024, 2048, 4096 and 8192. Standard factored Y took 230, 161, 157 and 575 µs at those sizes. Standard factored Y earned the production route. The larger XY slab depths remain experimental given their peak VRAM cost at 8192³. [Residue screen](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/y-residues.json)

### Promotion

Promoted the bounded bitmap and column-span planners, fused endpoint projection and eligible shared energy tiles, exact-zero optical row pruning, and selected-column XY transforms to automatic compact tensor XYZ dispatch. The native-integrator criteria passed at their original thresholds. The two primary workloads exceeded the 20% warm-time reduction target, with 77.5% lower runtime for the 101-energy host detector and 68.7% and 52.3% lower runtime for the 8192³ vacuum and material-2 spheres. The allocation retries and the small-grid and first-call measurements bound the tradeoffs of the default policy. [Qualification results](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-zp-speed/test-reports/padded-scan-speed/ZP15.5xqLJl/qualification-results.json)
