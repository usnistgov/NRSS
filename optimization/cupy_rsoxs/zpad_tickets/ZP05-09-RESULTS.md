# Virtual XYZ padding and material fill results

This ledger follows the [ZP05–ZP09 packet](ZP05-09-START.md). Installed on 2026-11-22 in `perf/virtual-z-pad` at `74b6d09f81d56441a2c4aeaaa8246a0dce6999d2`. The inspected revision, local tip, and public GitHub branch tip matched. Before the first fixture edits, a clean detached checkout was created at the same SHA. [Baseline checkout](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-3dpad-baseline)

Archive SHA-256 is `c178c479d62e15aeaf6826f524e8a37c129551eec7b25fd82f100433d55c9e38`. The CRC check passed and all 15 entries were installed without replacing existing files. [Original archive](/resdata/DeLongchamp/NRSS-opt-data/NRSS_3D_Virtual_Padding_Tickets_20261122.zip)

| Ticket | State | Work |
| --- | --- | --- |
| ZP05 | `qualified` | CPU oracles and maintained literal material boxes passed. Public Z and XYZ options, literal references and signed Fourier comparisons passed. |
| ZP06 | `qualified` | Material-filled Z extensions pass literal detector, Fourier, reuse and compatibility checks. |
| ZP07 | `qualified` | Signed coefficients, logical detector layouts, allocation retry and the full XYZ detector passed. |
| ZP08 | `qualified` | Both seven-rung sphere ladders and all 18 matched incumbent comparisons passed. |
| ZP09 | `qualified` | Native and forced padding validation passed. All paired default reports agree on prior failing and passing cases. |

Dean accepted the ZP07 and ZP08 results on 2026-11-22.

## ZP05 startup evidence

Python 3.14.3, NumPy 2.4.3, SciPy 1.17.1, pytest 9.0.2 and CuPy 14.0.1. The CPU checks ran with the GPU hidden. The focused CuPy suite ran on Wheatley's RTX A6000 with the tensor-coefficient path selected.

| Check | Result | Evidence |
| --- | --- | --- |
| `check_material_fill.py` | 96 complex tensor cases before or after DC replacement passed for each formulation. Peak-normalized errors ≤ `6.45e-15` against a literal box, under the `5e-13` threshold. | [Material check](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP05-start.vQIQmh/material.log) |
| `check_literal_sphere.py --sizes 128 256` | Vacuum and material 2 agreed with the complex128 literal reference at both sizes. Worst panel relative L2 `3.151e-7`, under `2e-5`. | [128³](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP05-start.vQIQmh/literal.log) · [256³](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP05-start.vQIQmh/literal-256.log) |
| `run_sphere_ladder.py --plan` | Seven sizes, 128–8192. | [Plan](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP05-start.vQIQmh/plan.log) |
| `run_sphere_ladder.py --preflight`, repeated with `--fill 2` | Ten independent CPU sphere anchors at each of seven sizes for each fill. | [Vacuum](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP05-start.vQIQmh/preflight.log) · [Material 2](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP05-start.vQIQmh/preflight-matrix.log) |
| `pytest tests/smoke -m "not gpu"` | 116 passed, 3 skipped, 72 deselected. Includes the five new literal-reference cases. | [CPU smoke](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP05-start.vQIQmh/cpu-smoke.log) |
| `pytest tests/smoke/test_cupy_virtual_z.py --nrss-path cupy_tensor_coeff` | 28 passed. Includes incumbent Z Fourier coefficients, detector and reduction parity, identity, window and memory-failure checks. | [Wheatley GPU smoke](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP05-start.vQIQmh/wheatley-virtual-z.log) |

The maintained helper builds centered literal ZYX boxes with either a numbered fill or vacuum. The tests use a `(5, 7, 8) → (14, 16, 19)` fixture with a matrix-filled hole. They verify the centered placement, exterior `S=0` and finite Euler angles, isotropic sentinels, optical tables, original interior fields and voxel fraction closure. One case selects a material with zero volume fraction throughout the supplied crop. The incumbent Z-only vacuum helper delegates to the same literal builder. [Reference fixture](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/tests/validation/lib/virtual_z_padding.py) · [Maintained tests](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/tests/smoke/test_cupy_virtual_z.py)

The packet calls for exposing the new options alongside runnable backend routes. The recorded 8192 checks are CPU anchor preflight. Full production panels enter qualification in ZP08.


## ZP06 material-filled Z gate

Implemented `virtual_pad_material` on the two existing axial evaluators. The selected material's real volume-fraction basis row transforms its contrast with a uniform box. The implementation adds the logical XY-Hann box terms before DC repair and mixes the isotropic optical scalar with the interior orientation rows at each energy. The two extra window FFTs and occupancy correction are recorded under `zpad_fill`.

The real specimen is a single centered spherical vesicle rebuilt from the [accepted WB05 fitting registry](/homes/deand/lipid_vesicle/fitting/accepted_joint_fit.json) using the accepted midplane radius 34.10089 nm, tail order 0.601960, bilayer profile, and measured head, tail and water optical constants. The 64³ grid at 1.5 nm pitch follows the project's single-object visualization build. Both geometry and optics were frozen before any comparisons. The volume fractions sum to one at every voxel and all six faces are pure water. The NRSS visualizer confirmed the two headgroup bands and the radial tail orientation. [Fixture metadata](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP06-vesicle.fOLHXi/vesicle-fixture.json) · [Build script](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP06-vesicle.fOLHXi/build_fixture.py) · [Tail orientation preview](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP06-vesicle.fOLHXi/nrss-check-7.png)

At 284.7 and 287.55 eV, compared the public compact path with a literal box occupied by material 3 (water). Both evaluators passed at 128 planes with both window settings. The sum evaluator also passed at 129 planes with both windows. All 12 full panels have identical coordinates and finite masks. The largest relative L2 is `2.29e-07` and the largest absolute difference divided by the literal peak is `3.37e-07`, both below `2e-5`. Vacuum comparisons passed on both evaluators. [Per-panel results](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP06-vesicle.fOLHXi/vesicle-parity.json) · [Full detector data](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP06-vesicle.fOLHXi/vesicle-detectors.h5) · [Comparison script](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP06-vesicle.fOLHXi/check_vesicle.py)

The maintained suite checks every signed bin of a small absorbing tensor specimen before and after six-neighbor DC replacement against a complex128 Fourier transform. It also covers both evaluators and windows with native vacuum, selected zero optics, isotropic fill, and oriented fill, plus same-instance changes, identity padding, the homogeneous-box limit and NetCDF metadata. CPU smoke reported 126 passed and 3 skipped. The focused Wheatley suite reported 61 passed. [CPU smoke](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP06-vesicle.fOLHXi/cpu-tests-final.log) · [GPU smoke](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP06-vesicle.fOLHXi/gpu-tests-final.log)

### Vesicle timings

Two energies, XY Hann, RTX A6000, five warm synchronized calls per arm after two warmups. Stage times come from separate instrumented calls. Each run builds one basis and reuses it across both energies. The saved detector, coordinate and metadata hashes agree exactly between the pinned parent and candidate vacuum arms. [Parent timings](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP06-vesicle.fOLHXi/timing-parent.json) · [Candidate timings](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP06-vesicle.fOLHXi/timing-candidate.json) · [Timing script](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP06-vesicle.fOLHXi/bench_vesicle.py)

| Evaluator | Target Z | Parent vacuum (ms) | Candidate vacuum (ms) | Virtual water (ms) | Literal water (ms) | Fill preparation (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sum | 65 | 7.57 | 7.63 | 8.21 | 7.13 | 0.57 |
| residues | 128 | 13.31 | 13.26 | 13.86 | 8.27 | 0.55 |
| sum | 512 | 7.68 | 7.65 | 8.35 | 16.73 | 0.58 |
| residues | 512 | 27.19 | 27.45 | 28.02 | 16.75 | 0.55 |

The frozen vesicle fields also supply a material-filled literal control for the next XYZ comparison.


## ZP07 single-vesicle ladder

The accepted fit manifest records a 256 × 1024 × 1024 grid at 1.5 nm pitch. We used that grid for the explicit compact-tensor reference on the pinned parent at `74b6d09`. Starting with the ZP06 frozen vesicle, the smallest centered even crop containing every non-water material fraction is 52³. All 48,008 voxels with non-water fractions were retained. Every removed voxel was pure water; its stored tail angles and order carry zero tail volume fraction. All runs use the same 284.7 and 287.55 eV optics, the same water material and windowing off. [Crop provenance](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP07-vesicle-ladder.IGZTvv/minimal-crop.json) · [Benchmark script](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP07-vesicle-ladder.IGZTvv/benchmark.py)

The matched 256 × 1024 × 1024 virtual run returned exactly the explicit detector coordinates and finite masks. Its worst detector relative L2 was `6.07e-07` and its worst absolute error divided by the reference peak was `6.79e-06`. The greatest `|ΔA|` over the fitting q band was `9.05e-07`. The three cubic grids use the same stored crop and deliver complete detector panels. [Full results](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP07-vesicle-ladder.IGZTvv/vesicle-ladder-results.json) · [Anisotropy ladder](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP07-vesicle-ladder.IGZTvv/vesicle-anisotropy-ladder.pdf) · [Fringe zoom](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP07-vesicle-ladder.IGZTvv/vesicle-fringe-zoom.pdf)

Warm detector times are medians of three synchronized calls after one warmup on the RTX A6000. Peak VRAM is the NVML per-process maximum from a separate run and includes CUDA context and cached allocations. Values cover both energies.

| Route | Logical ZYX | Δqz (nm⁻¹) | Δq⊥ (nm⁻¹) | Detector time (s) | Peak VRAM (MiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| Explicit | 256 × 1024 × 1024 | 0.016362 | 0.004091 | 1.72515 | 11,540 |
| Virtual XYZ | 256 × 1024 × 1024 | 0.016362 | 0.004091 | 0.02426 | 586 |
| Virtual XYZ | 1024 × 1024 × 1024 | 0.004091 | 0.004091 | 0.02609 | 628 |
| Virtual XYZ | 2048 × 2048 × 2048 | 0.002045 | 0.002045 | 0.10866 | 1,194 |
| Virtual XYZ | 4096 × 4096 × 4096 | 0.001023 | 0.001023 | 0.68608 | 3,442 |

The XYZ route evaluates the required axial modes from padded XY FFT batches of the occupied planes. The bounded tile retries with fewer planes after an allocation failure. Small tests cover rectangular and odd targets, both windows, isotropic and oriented fills, detector and integrated output, Z-only equivalence, large 64-bit addresses, and the retry. CPU smoke had 136 passes and 3 skips. The focused Z/XYZ GPU suite had 76 passes. [CPU smoke](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP07-vesicle-ladder.IGZTvv/cpu-smoke.log) · [Wheatley smoke](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP07-vesicle-ladder.IGZTvv/padding-smoke-final.log)


## ZP07 acceptance checks

The expanded ZP07 suite passed 29 tests on Wheatley's Quadro RTX 8000. Two arbitrary ZYX targets, both windows, and original and laterally translated absorbing inclusions match every complex tensor component against a literal complex128 FFT before and after DC repair. The same controls compare complete detector panels. The rectangular 12-angle tests cover out-of-order energies, masks, host and device output, detector and all reduced layouts, intersection and union, and serialization. Same-process changes to the target and material reproduced fresh reference boxes. An induced allocation failure fell back to one XY plane and retained full parity. [XYZ checks](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP07-08-gates.omcG9F/xyz-gates.log) · [CPU smoke](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP07-08-gates.omcG9F/cpu-smoke.log)

The XYZ route batches occupied XY FFT planes and evaluates the selected axial endpoints with 64-bit IDs. This applies to integer and noninteger targets. The ZP08 complete-panel check uses a separate, integer-ratio CPU residue implementation. The production working set includes the occupied fields, tiled FFT workspace, endpoint indices and basis amplitudes, and the logical detector. At 8192³ and one energy it retained 22,280,083 endpoint rows in a 534,722,040-byte basis and processed one FFT plane per tile. [Source revision](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP07-08-gates.omcG9F/checkout-head.txt) · [Backend source hashes](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP07-08-gates.omcG9F/source.sha256)

## ZP08 production sphere ladder

All seven sizes completed in each of the vacuum and synthetic material-2 campaigns on the same RTX A6000. The same 128³ voxel sphere and optical settings were used at each rung for a given fill. Each row returns a complete host FP32 detector. The independent CPU residue detector validated all pixels and exact masks. [Vacuum figure](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP07-08-gates.omcG9F/figures-vacuum/sphere_ladder.png) · [Material-2 figure](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP07-08-gates.omcG9F/figures-2/sphere_ladder.png) · [Campaign logs](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP07-08-gates.omcG9F/sphere-campaign.log)

| Logical cube | Vacuum warm (s) | Vacuum peak (MiB) | Material 2 warm (s) | Material 2 peak (MiB) |
| ---: | ---: | ---: | ---: | ---: |
| 128³ | 0.0070 | 310 | 0.0071 | 310 |
| 256³ | 0.0075 | 382 | 0.0075 | 382 |
| 512³ | 0.0091 | 454 | 0.0092 | 454 |
| 1024³ | 0.0179 | 476 | 0.0179 | 476 |
| 2048³ | 0.0583 | 584 | 0.0583 | 584 |
| 4096³ | 0.2743 | 1076 | 0.2747 | 1076 |
| 8192³ | 1.2165 | 3428 | 1.2154 | 3428 |

Times are medians of five synchronized warm calls. GPU peaks use separate NVML workers. At 8192³, 11,140,041 detector pixels were finite, and full-panel relative L2 against the voxel reference was `4.48e-7` for vacuum and `5.00e-7` for material 2. The transverse `+qy` check evaluates the same voxels at continuous qz. At q ≥ 1 nm⁻¹, it shows 11.48% relative L2 versus the continuum, with axial endpoint interpolation contributing about 0.0025% along that line. Tail intensity there peaks at about `4.3e-6 I0`. [Continuous-qz diagnostic](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP07-08-gates.omcG9F/sphere-continuous-line.json)


### Matched incumbent controls

Both fills have literal-cube controls through 1024 and laterally explicit plus virtual-Z controls through 4096. All 18 compare full panels and exact finite masks to XYZ. Their largest relative L2 is `3.49e-07`. Across the 14 production panels the largest relative L2 versus the independent voxel calculation is `5.05e-07`. The repeated 512³ literal vacuum control differed from its first warm median by `-0.50%`. The contrast-normalized vacuum and material-2 complete panels differed by at most `1.87e-07`. [All timings and matched controls](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP07-08-gates.omcG9F/sphere-qualification-results.json) · [Runtime and VRAM comparison](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP07-08-gates.omcG9F/sphere-matched-resources.png)

| Vacuum control | Logical grid | Control warm (s) | XYZ warm (s) | Control peak (MiB) | XYZ peak (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Literal cube | 1024³ | 1.840 | 0.0179 | 17,738 | 476 |
| Explicit XY + virtual Z | 4096³ | 3.873 | 0.2743 | 41,894 | 1,076 |

Two FP32 material volumes alone would need 64 GiB in a literal 2048³ cube or in a 128 × 8192 × 8192 laterally explicit crop. Both exceed this GPU's capacity before allocating FFT workspace. The complete Z/XYZ smoke file passed 90 tests, and CPU smoke passed 136 with 3 skips. [Padding suite](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP07-08-gates.omcG9F/padding-smoke.log) · [CPU smoke](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP07-08-gates.omcG9F/cpu-smoke.log)


## ZP09 lipid vesicle energy scan

Both arms use the frozen 52³ vesicle at 1.5 nm pitch and the same 256 × 1024 × 1024 logical box, filled by water and run with windowing off. The explicit parent at `74b6d09` stores the full physical box. The XYZ candidate stores the crop. The compact tensor settings, FP32 host detector output and optical tables are shared. The single energy is 287.55 eV, the fitted energy. The 101 energies cover 282.55–292.55 eV in 0.10 eV steps with the 20-point subset recorded in the fixture. The optical tables were interpolated from the same hash-verified headgroup, oriented tail and water sources as ZP06. At 287.55 eV the frozen optical constants reproduced exactly. [Fixture and spectral provenance](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-vesicle-energies.vrbPPq/fixture.json) · [Runtime snapshot](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-vesicle-energies.vrbPPq/snapshot.json) · [Candidate source hashes](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-vesicle-energies.vrbPPq/candidate-backend.sha256)

The benchmark uses one polarization direction and reports the first synchronized call plus the median of five warm calls on Wheatley's RTX A6000. All times include the full host detector output. Peak GPU memory is the NVML per-process maximum and peak host RSS includes stored material fields. The memory runs use separate workers. [Benchmark script](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-vesicle-energies.vrbPPq/benchmark.py) · [All timing samples and counters](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-vesicle-energies.vrbPPq/vesicle-energy-results.json)

| Energies | Result chunk | Explicit first / warm (s) | XYZ first / warm (s) | Warm speedup | Peak GPU explicit / XYZ (MiB) | Peak host explicit / XYZ (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1.872 / 1.730 | 0.171 / 0.0205 | 84.5× | 11,540 / 526 | 6,622 / 529 |
| 20 | 1 | 1.900 / 1.749 | 0.289 / 0.0865 | 20.2× | 11,540 / 894 | 6,870 / 776 |
| 101 | 1 | 2.176 / 1.856 | 0.728 / 0.3614 | 5.14× | 11,540 / 980 | 7,638 / 1,544 |
| 101 | 4 | 2.170 / 1.855 | 0.796 / 0.3630 | 5.11× | 11,540 / 1,032 | 7,637 / 1,543 |

Each scan constructs one geometry basis. The explicit path makes seven 3D FFT calls and the XYZ path makes 28 batched XY FFT calls over the occupied planes, independent of energy count. XYZ uses up to 16 planes per XY batch. It includes one detector endpoint mask per distinct energy in the union of sampled Fourier indices. That union grows from 797,371 endpoints at one energy to 1,937,851 at 20 and 2,017,475 at 101, including the DC stencil. Result chunking changes the host transfer buffers. Both 101-energy chunk sizes use the same FFT count and endpoint union.

All 101 full detector panels match in energy order, q coordinates and finite-pixel masks. The largest relative L2 is `7.57e-7` and the largest absolute pixel error divided by the panel peak is `8.60e-6`, both below `2e-5`. The public `i_a` outputs agree in all coordinates and finite masks. Across q = 0.06–0.80 nm⁻¹, the largest absolute difference in anisotropy is `9.26e-7` and its RMS is `1.34e-7`. The maximum across all q is `5.15e-5` at 1.948 nm⁻¹ and 284.95 eV, where the radial intensity is `1.78e-7` of that energy's peak. [Heatmap PNG](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-vesicle-energies.vrbPPq/vesicle-101-energy-anisotropy.png) · [Heatmap PDF](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-vesicle-energies.vrbPPq/vesicle-101-energy-anisotropy.pdf) · [Explicit reduced dataset](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-vesicle-energies.vrbPPq/explicit-101-chunk1-i-a.nc) · [XYZ reduced dataset](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-vesicle-energies.vrbPPq/virtual-101-chunk1-i-a.nc)

The 20-energy subset also passed with 12 polarization directions using `EAngleRotation=[0, 15, 165]` degrees in `[StartAngle, IncrementAngle, EndAngle]` order. Explicit first and warm times were 1.970 s and 1.803 s with 11,540 MiB peak GPU. XYZ took 0.315 s and 0.105 s with 924 MiB peak GPU. Each route built one basis, with nine 3D FFT calls for explicit padding and 36 batched XY FFT calls for XYZ. XYZ used the same 20 endpoint masks and union size as the one-direction run. The complete panels matched q and finite masks with worst relative L2 `5.90e-7`. A non-monotone permutation of all 101 energies at chunk four reproduced each route's chunk-one detector values bit for bit after restoring the order.

Serial diagnostics reused a fixed morphology object and changed only the requested energy before each public run, keeping the process CUDA caches warm between calls. The detector stack assembled by each 101-call loop matched its own batched stack bit for bit, and both 20-call loops matched the corresponding frames from the 101-energy batch. The first 101-call loop took 177.181 s for explicit and 2.344 s for XYZ, with 11,540 and 580 MiB peak GPU respectively. The two 20-call loops took 34.899 and 34.762 s for explicit and 0.612 and 0.451 s for XYZ. The 101-call loop was measured once per path. [Rotation, energy order and loop results](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-vesicle-energies.vrbPPq/extra-results.json)

The lipid subgate passes.

### Native and workflow compatibility

The speed-ticket harness compared the pinned parent and candidate on CoreShell at 32 × 512 × 512 on the RTX A6000. The single-energy case uses one polarization direction. The 20- and 101-energy cases use `EAngleRotation=[0, 15, 165]` degrees in `[StartAngle, IncrementAngle, EndAngle]` order. Each full FP32 host detector stack matched bit for bit, including its q coordinates and finite mask. Both arms ran the compact tensor route and built one basis per call. [Paired measurements and diagnostics](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/compatibility-results.json)

| Energies | Parent / candidate first median (ms) | Parent / candidate warm median (ms) | Peak GPU parent / candidate (MiB) | Absolute median A/A drift |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 149.49 / 150.45 | 24.90 / 25.26 | 756 / 756 | 3.57% |
| 20 | 176.00 / 176.33 | 55.27 / 54.93 | 868 / 868 | 1.01% |
| 101 | 327.81 / 331.16 | 170.11 / 170.81 | 1,110 / 1,110 | 0.18% |

The installed PyHyperScattering `NRSSIntegrator` gives a different q step than the built-in reduction in the same integration smoke assertion on the parent and candidate. Dean directed that NRSS's built-in integrator be the reference for this qualification. The paired reports retain the shipped dependency and its test dispositions. The forced workflow audit compares full absolute-intensity detector panels and NRSS-generated sector datasets from virtual and literal boxes on the same logical grids. Maintained thresholds and q reference files retain their existing assertions. [Parent dependency reproduction](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/parent-installed-radial.log) · [Candidate dependency reproduction](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/audit-pilot.log) · [Workflow audit](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/tests/validation/dev/cupy_rsoxs_virtual_zpad/workflow_padding_audit.py)

The 90 padding smoke tests passed on the Quadro RTX 8000. The audited DC-replacement and off-axis Bragg registration pilot had four pytest passes and 18 literal comparisons. Both virtual routes matched complete native detector and sector outputs. [Smoke log](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/native-padding-smoke.log) · [Audit pilot](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/audit-native-pilot.log) · [Per-evaluation metrics](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/audit-native-pilot.jsonl)

For the `(128, 256, 256) → (256, 288, 320)` HCP and BCC films, the tilted HCP uses XY Hann and the tilted BCC is unwindowed. At 285 and 570 eV all four complete XYZ detector comparisons matched exact coordinates and finite masks, and their worst relative L2 was `2.77e-7`. The 343 matched peak measurements were assigned to the same q bins by the explicit detector, XYZ, and independent continuous-qz voxel calculation. The largest fractional explicit-versus-XYZ peak discrepancy for those measurements was `1.13e-6`. [High-q matched controls and peak data](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/high-q-xyz/results.json)

The 20-energy vesicle and 12 polarization directions were repeated on the same 256 × 1024 × 1024 target with vacuum and with material 2 as the exterior. Tail material 2 has Euler orientation fields and nonzero interior order. Its exterior uses its isotropic response. For both fills, virtual XYZ and the literal box matched detector and native sector coordinates and masks. Worst detector relative L2 is `1.42e-7` and worst native sector relative L2 is `1.71e-7`. [Native vesicle detector and sector parity](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/lipid-fills-native-results.json)

On the A6000, the 12 × 33 × 35 anisotropic film was padded to 48 Z planes. All eight vacuum-Z combinations matched the pinned parent bit for bit, including both evaluators, windowing off and XY Hann, and one and 12 polarization directions. Each path built one basis and returned the full host detector with 274 MiB peak per-process VRAM in each arm. The initial 12-direction Hann residues and 101-energy native cases had paired median slowdowns of 1.91% and 0.35%. The timing confirmations below use the same A6000. [Vacuum-Z and native run data](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/compatibility-results.json)

The maintained CoreShell and MWCNT morphology builders also ran directly through the audit in a standalone native check. Both the XYZ vacuum and material-filled Z routes matched their literal boxes in the full detector and native sector channels. CoreShell's largest relative L2 across the four comparisons was `1.84e-6` and MWCNT's was `8.56e-7`, below `2e-5`. [Reference morphology parity](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/native-reference-builders.json)

The two 512³ maintained sphere fixtures also ran through NRSS's native `i_only` layout. At the asserted supersampling, both diameters passed every original form-factor bound in each CuPy path on both checkouts. For the 70 nm sphere, the tensor-path 95th-percentile log error was `0.09475` against a limit of `0.120`. For the 128 nm sphere, the mean absolute minimum-position error was `0.0002572 nm⁻¹` against a limit of `0.0004500 nm⁻¹`. The two checkouts returned identical q and form-factor samples within each path. [Parent native theory check](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/parent-native-sphere-theory.json) · [Candidate native theory check](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/candidate-native-sphere-theory.json)

### Full forced padding audit

On the A6000, 213 literal-box comparisons passed across 38 test instances. There were 108 XYZ vacuum evaluations and 105 material-filled Z evaluations. The comparisons included 167 full detectors, four native integrated results and 42 native sector datasets. The largest relative L2 was `5.51e-6` and the largest peak-normalized absolute error was `6.26e-6`, below the `2e-5` limits. The audit logged ineligible or dedicated cases with their reasons. [Per-evaluation record](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/forced-padding.jsonl) · [Coverage and numeric summary](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/compatibility-results.json)

The pytest runner recorded 136 CPU passes with three skips, 92 GPU smoke passes with ten skips and one failure, and 23 physics passes with two failures. The integration smoke failure used the installed PyHyperScattering `NRSSIntegrator`. The 70 nm and 128 nm sphere assertions on that reducer's output reproduced the same failing values on both checkouts on the Quadro RTX 8000. Both sphere fixtures pass the same analytical limits on NRSS's native q grid. The original test dispositions and graphics are retained. [Full forced report](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/forced-padding-report/20261122T141732Z/summary.md) · [Parent external sphere diagnostic](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/parent-external-sphere.log) · [Candidate external sphere diagnostic](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/candidate-external-sphere.log)

### Timing confirmations and handoff

For the 12-direction XY-Hann residues case, the reversed-order candidate slowdown was `0.12%` against `1.25%` same-checkout drift. The 101-energy reverse run initially showed a `13.20%` candidate slowdown. Individual parent-only workers also switched between the 167–171 ms and 191–215 ms timing bands. An interleaved follow-up used eight worker blocks on the same A6000, with three warmups and eight measurements per block and no concurrent memory sampling. Its first three paired block deltas were between `−0.31%` and `+0.09%`. Its fourth parent block slowed to a 189.56 ms median. The 32-call pooled medians were 168.35 ms parent and 167.74 ms candidate, which did not reproduce the 101-energy slowdown. [Recheck samples and spread](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/compatibility-results.json) · [Steady worker blocks](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/native-101-steady-results.json)

## ZP09 final paired reports

The parent and candidate completed all three report lanes on Wheatley. Each pair ran on the same physical GPU and used the installed environment with plot harvesting. The columns show passed, failed, and skipped counts.

| Lane | Parent GPU smoke | Candidate GPU smoke | Parent physics | Candidate physics | Reports |
| --- | ---: | ---: | ---: | ---: | --- |
| `legacy_cyrsoxs` | 36/0/25 | 36/0/67 | 14/0/11 | 14/0/11 | [Parent](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/default-parent-legacy_cyrsoxs/20261122T141215Z/summary.md) · [Candidate](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/default-candidate-legacy_cyrsoxs/20261122T142555Z/summary.md) |
| `cupy_tensor_coeff` | 50/1/10 | 92/1/10 | 23/2/0 | 23/2/0 | [Parent](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/default-parent-cupy_tensor_coeff/20261122T143914Z/summary.md) · [Candidate](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/default-candidate-cupy_tensor_coeff/20261122T144328Z/summary.md) |
| `cupy_direct_polarization` | 30/1/30 | 30/1/72 | 19/2/4 | 19/2/4 | [Parent](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/default-parent-cupy_direct_polarization/20261122T143034Z/summary.md) · [Candidate](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/default-candidate-cupy_direct_polarization/20261122T144827Z/summary.md) |

The CuPy lanes reproduce the installed PyHyperScattering q-grid smoke failure and the two external spherical form-factor failures on both checkouts. NRSS's native sphere checks pass the existing tolerances. All prior passing cases retain that disposition, and all 213 independent padding comparisons passed. This closes the local padding qualification using the NRSS native integrator and preserves the external-reduction failures in the reports. [Paired case comparison](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-compatibility.Lk45wF/compatibility-results.json)
