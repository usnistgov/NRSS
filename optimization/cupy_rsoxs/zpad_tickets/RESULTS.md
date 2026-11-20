# Virtual Z padding results

The packet branch `perf/virtual-z-pad` and its detached timing control start from `3c4c14d51ecb7ff63e47968b2a323f140f4937ef`, verified against the public `fix/fft-grid-registration` ref. The two worktrees live under `/homes/deand/dev/NRSS_speed/.worktrees`.

The published parent carries the CoreShell and radial-label dispositions in [README_FIRST.md](README_FIRST.md). The separate local FFT-grid follow-up with its companion PyHyperScattering patch passed 8/8 gates. Its results are recorded separately from this pinned parent. [FFT-grid follow-up](/homes/deand/dev/NRSS_speed/test-reports/fft-registration-followup.ArgvS1/review/VALIDATION_GATES.md)

The reference replays used Python 3.14.3, NumPy 2.4.3 and SciPy 1.17.1. The GPU runs used CuPy 14.0.1 with CUDA 12.9.

| Check | Result | Evidence |
| --- | --- | --- |
| Fourier oracle | 72 complex-coefficient comparisons and 12 signed grating cases passed. Worst relative L2 across the two evaluators was `6.47e-15`. | [JSON](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/fourier-and-grating.json) |
| CPU HCP and BCC replays at 192² and 1024² | The per-band errors reproduced the packaged values exactly at both sizes. | [192²](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/pilot-192.json) · [1024²](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/pilot-1024.json) |
| Explicit centered vacuum and registered detector controls | Six checks passed on each CuPy path on Wheatley’s Quadro RTX 8000. This includes the DC test, the native grating, and the literal M=256 and M=1024 grating cases with signed intensities, coordinates and full finite masks checked. | [pytest log](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/wheatley-zp01-registered.log) |
| GPU-hidden NRSS smoke | 93 passed, 3 skipped. | [pytest log](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/cpu-smoke.log) |
| Registered A/A timing at 128 × 1024² | RTX A6000, one energy, one E direction, host input and output. Median warm times were 0.9151 s and 0.9127 s across three paired samples, with 0.22% median absolute paired variation. | [Timing summary](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/parent-aa/20261120T181348Z-depth-jja_i4dm/summary.json) |

For the grating at half of a native axial bin, the explicit box gives normalized bright and opposite intensities of approximately 0.405305 and 0.016232 for both signs. The native run gives 0.5 and 0. The new test checks the literal box through `Morphology.run`, with the existing `fft_bins_v1` transverse coordinates and detector support.

The S04 selected-z prototype is available in the local historical checkout at `/homes/deand/dev/NRSS/test-reports/S04-selected-z-rtB3m1/candidate-source`. The initial reference established the control for the virtual detector comparisons below.

## Virtual padding GPU gate

The Hann measurements in this section use the earlier target-box XYZ weights and establish explicit-box compatibility. The ZP04 section records the final XY-only film behavior.

Implementation revision `26ee3423c3947dedb0a4c006fa3b9044dd47b02c` on `perf/virtual-z-pad`. The paired clean parent is `3c4c14d`. The selected sum supports noninteger pad ratios. The residue FFT supports integer ratios. Both build one occupied XY Fourier basis across all energies, use a sparse endpoint union, and evaluate and phase the target-grid DC stencil before replacement.

### Full-width timing and peak VRAM

The CoreShell-derived workload occupied 128 × 1024 × 1024 voxels at 1.25 nm, with one energy and one electric-field direction on an RTX A6000. The explicit arm included literal vacuum outside the fixed specimen. Warm medians cover the synchronized public morphology call. An independent 10 ms NVML sampling pass measured peak worker VRAM.

| Transform depth | Execution | Warm time (s) | Peak worker VRAM (MiB) | Paired reference |
| ---: | --- | ---: | ---: | --- |
| 128 | Native | 0.914 | 5,396 | [Three native/1024 pairs](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/next-gate.UZPccu/medium-1024-sum/20261120T182800Z-depth-ddbxodpr/summary.json) |
| 256 | Explicit | 2.125 | 11,540 | [Matched timings](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/next-gate.UZPccu/medium-256-sum/20261120T182510Z-depth-f8ah3bpk/summary.json) |
| 256 | Selected sum | 0.911 | 5,396 | |
| 512 | Explicit | 4.291 | 22,806 | [Matched timings](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/next-gate.UZPccu/medium-512-matched-sum/20261120T184917Z-depth-bw11cxkd/summary.json) |
| 512 | Selected sum | 0.902 | 5,396 | |
| 1024 | Explicit | 8.643 | 45,338 | [Matched timings](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/next-gate.UZPccu/medium-1024-matched-sum/20261120T184132Z-depth-1qcutb_1/summary.json) |
| 1024 | Selected sum | 0.909 | 5,396 | |

The median of five alternating 1024-plane explicit/virtual pairs was **9.51× faster** with the selected sum and peak VRAM was **88.1% lower**. Median first calls after CUDA prewarm were 8.101 s explicit and 1.043 s virtual. A separate native/1024 comparison put the virtual route at 1.012× native time. The 1024-plane residue route took 1.126 s with 6,424 MiB of worker VRAM.

On the Quadro RTX 8000, the full-width 20-energy, 12-direction Hann case ran in 0.959 s with the 256-plane selected sum and 2.095 s with explicit padding. At 1024 planes the selected sum took 1.021 s versus 0.896 s at native depth. The selected sum and native arm each peaked at 5,298 MiB. [256-plane pairs](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/next-gate.UZPccu/full20-256-sum/20261120T182606Z-energy20-dxv0cvde/summary.json) · [Native/1024 pairs](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/next-gate.UZPccu/full20-1024-sum/20261120T182742Z-energy20-m3z313ns/summary.json)

These runs timed both evaluators on the smaller CoreShell control at three pad factors, including an independent device-resident screen. The selected sum was faster than the residue route in the measured timing comparisons. The stage records include actual route, FFT calls and counts, phase/layout/axial/gather costs where applicable, detector endpoint counts, and retained basis bytes.

[All benchmark summaries and stage data](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/next-gate.UZPccu/evidence.json) · [Device-resident timing](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/next-gate.UZPccu/small-device20-timing/20261120T184201Z-energy20-eq1hw6uy/summary.json) · [Largest CuPy allocation checks](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/next-gate.UZPccu/allocation-profile)

### Accuracy and smoke

| Comparison | Worst relative L2 | Evidence |
| --- | ---: | --- |
| 101-energy anisotropic Hann, selected sum | 1.63e-07 | [Full detector comparison](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/next-gate.UZPccu/scan101-sum/20261120T182102Z-energy101-tj752jw1/summary.json) |
| 101-energy anisotropic Hann, residue FFT | 2.86e-07 | [Full detector comparison](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/next-gate.UZPccu/scan101-residues/20261120T182113Z-energy101-8cuys70y/summary.json) |
| Full-width 20-energy Hann at 256 planes, selected sum | 7.37e-07 | [Full detector comparison](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/next-gate.UZPccu/full20-parity-sum/20261120T182533Z-energy20-zxignjxj/summary.json) |
| Full-width 20-energy Hann at 256 planes, residue FFT | 1.71e-07 | [Full detector comparison](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/next-gate.UZPccu/full20-parity-residues/20261120T182902Z-energy20-8mmkmadf/summary.json) |
| Full-width single energy at 1024 planes, selected sum | 2.11e-10 | [Full detector comparison](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/next-gate.UZPccu/medium-1024-parity-sum/20261120T183750Z-depth-2jrahmjj/summary.json) |
| Full-width single energy at 1024 planes, residue FFT | 1.45e-10 | [Full detector comparison](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/next-gate.UZPccu/medium-1024-parity-residues/20261120T183920Z-depth-8mywicfv/summary.json) |
| 512² CoreShell 101-energy Hann at 128 planes, selected sum | 1.26e-07 | [Full detector comparison](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/next-gate.UZPccu/core101-parity-sum/20261120T183448Z-energy101-7iv0q13d/summary.json) |
| 1024² CoreShell 101-energy Hann at 1024 planes, selected sum | 1.47e-07 | [Full detector comparison](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/next-gate.UZPccu/wide101-parity-sum/20261120T183459Z-energy101-28x14tdk/summary.json) |

All accuracy comparisons required exact detector coordinates, energy ordering and finite masks. The full-width 101-energy maximum frame error occurred at 288.6 eV, with the largest absolute residual at `(qx, qy) = (0, −0.00490874) nm⁻¹`. The signed padded grating passed on both evaluators at 256 and 1024 planes. CPU smoke passed with 102 tests and 3 skips. GPU smoke passed with 45 on compact tensor and 31 on direct polarization.

[Worst residual locations](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/next-gate.UZPccu/residual-locations.json) · [CPU smoke](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/cpu-smoke-candidate.log) · [Compact GPU smoke](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/next-gate.UZPccu/smoke-cupy_tensor_coeff.log) · [Direct GPU smoke](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/next-gate.UZPccu/smoke-cupy_direct_polarization.log)

The public opt-in is `backend_options={"virtual_z_pad_to": 1024}` with the eligible compact tensor path. The next validation covers indexed high-q HCP/BCC accuracy and the full promotion panel in [ZP03](ZP03-high-q.md).

## ZP03 and ZP04 · Indexed films and the manual XY window

The ZP04 implementation is `8f1fd82`. The high-q and matched-box evaluation is in `3db18f3`. For an extended film, omitted `WindowingType` and explicit `0` both give unit weights. Explicit `1` applies lateral Hann weights before the XY transform and uses unit axial weights on every occupied plane. Identity padding retains the native window.

The submitted NumPy oracle passed 40 literal FFT, 40 selected-sum and 32 residue comparisons, including odd depth 257. The updated GPU coefficient and detector checks passed through both evaluators. At the prescribed scatterer position the non-DC intensity factor was 1/16 on either face and the middle plane. The two-face complex coefficients reproduced the independent thickness-fringe calculation. Public edge-matched scalar and discontinuous anisotropic fixtures gave identical results for omission and explicit `0`. Their explicit `1` results agreed with the literal-box XY reference with both projectors exercised. [CPU oracle](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/zp03-zp04/film-window-cpu.json) · [GPU checks](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/zp03-zp04/film-window-gpu.log)

### Indexed Bragg accuracy

The tests used fixed 128 × 1024² specimens at 1 nm pitch. HCP and BCC each ran at both tilts with the window off. The tilted HCP Hann specimen used matched XY weights in the native, explicit, virtual and continuous-reference arms. Recorded axes, mask counts, anchor indices, indexed intensities, centroid residuals, integrated intensities and RMS widths are available at every transform depth. Both maintained 256 × 1024² anchor cases were replayed at native depth. [All bands](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/zp03-zp04/review/band-overview.csv) · [Indexed counts at all three floors](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/zp03-zp04/review/indexed-counts.csv) · [Per-case captures](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/zp03-zp04/high-q)

| 570 eV, q⊥ 1.90–2.40 nm⁻¹ | Native error | 512-plane error | 1024-plane error | Native → 1024 matched peaks |
| --- | ---: | ---: | ---: | ---: |
| Tilted HCP | 32.57% | 3.00% | 0.67% | 257 → 280 of 280 |
| Tilted BCC | 36.30% | 3.30% | 0.72% | 10 → 12 of 12 |
| Tilted HCP with XY Hann | 35.74% | 3.00% | 0.73% | 261 → 288 of 288 |

Errors are relative L2 in detector intensity against the continuous Ewald sum. Counts use the 10⁻⁴ floor relative to each direct anchor. The BCC anchor is a verified allowed member of {110} for each tilt and energy. BCC odd-only candidates are tabulated separately. In the tilted BCC outer band the largest odd-only maximum was 3.49 × 10⁻⁸ of the allowed anchor in the continuous reference and 3.45 × 10⁻⁸ at 1024 planes. Both tilted unwindowed cases had zero lost reference peaks and zero unmatched detections at 1024 planes at this floor. Tilted HCP already matched all 280 peaks at 512, with two unmatched detections. Its median absolute peak-height error fell from 20.17% at native depth to 0.55% at 1024 planes.

[Depth versus error](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/zp03-zp04/review/high-q-depth-error.pdf) · [Tilted HCP maps](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/zp03-zp04/review/hcp_tilted-outer-band.pdf) · [Tilted BCC maps](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/zp03-zp04/review/bcc_tilted-outer-band.pdf) · [XY-Hann HCP maps](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/zp03-zp04/review/hcp_hann-outer-band.pdf) · [Near-bin and between-bin cuts](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/zp03-zp04/review/hcp_tilted-signed-cuts.pdf)

### Matched XY-Hann performance and parity

The 128 × 1024² CoreShell-derived model used 1024 transform planes on the A6000 with the same lateral weights and unit axial weight in both arms. Over three alternating pairs at 20 energies and 12 directions, warm medians were 8.943 s explicit and 1.036 s virtual, an 8.64× median paired speedup. A separate 10 ms NVML pass measured 45,338 MiB explicit and 5,396 MiB virtual peak worker VRAM. [Timing](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/zp03-zp04/film-wide-20-hann-timing/20261120T191145Z-energy20-qluhtf88/summary.json) · [Memory](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/zp03-zp04/film-wide-20-hann-memory/20261120T191833Z-energy20-vh183tzg/summary.json)

The full 101-energy 1024-plane XY-Hann comparison passed with exact detector coordinates and finite masks. Its worst relative L2 error was 7.39 × 10⁻⁷ at 283.3 eV. Separate small anisotropic 101-energy checks passed with both window choices and with one or 12 directions. [101-energy XY-Hann details](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/zp03-zp04/film-wide-101-hann/20261120T190825Z-energy101-e9r3f6hf/summary.json)

The direct-polarization report generator hit an NFS stale-file-handle error after all four steps had been recorded. Its Markdown summary was regenerated with the repository's summary writer from the completed records. [Original run log](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/zp03-zp04/default-cupy_direct_polarization.log)

Across all five film configurations, the largest full-detector relative L2 difference between the explicit and virtual paths was 3.93e-07. All requested depths passed the exact detector-coordinate and finite-mask comparisons.

### Promotion panels

The default report used the installed PyHyperScattering and the pinned FFT-grid parent conventions. The opt-in audit replayed eligible compact-tensor morphologies at twice their occupied depth through both public padding routes. It kept the native returns for the maintained assertions and logged the padded comparisons independently. The GPU-hidden CPU smoke ran 102 tests with 3 skips. The split native report also ran the path-specific CPU lanes and environment checks.

| Panel | GPU smoke | Physics | Report |
| --- | --- | --- | --- |
| Legacy CyRSoXS | 36 passed, 25 skipped | 14 passed, 11 skipped | [Summary](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/zp03-zp04/default-legacy_cyrsoxs/20261120T192057Z/summary.md) |
| Default compact tensor | 51 passed, 10 skipped | 21 passed, 4 failed | [Summary](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/zp03-zp04/default-cupy_tensor_coeff/20261120T191236Z/summary.md) |
| Default direct polarization | 31 passed, 30 skipped | 17 passed, 4 failed, 4 skipped | [Summary](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/zp03-zp04/default-direct/20261120T191946Z/summary.md) |
| Compact tensor with opt-in parity audit | 51 passed, 10 skipped | 21 passed, 4 failed | [Summary](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/zp03-zp04/opt-in-film-report/20261120T193416Z/summary.md) |

All 92 padded comparisons passed. They covered raw detector results, integrated results, and the three observable layouts. The worst relative L2 error was 1.04e-06 on MWCNT. The inherited native failures are the two analytical sphere sizes and the two CoreShell reference checks. The reviewed radial-coordinate and CoreShell fixes live in the separate FFT-grid follow-up. Promotion remains gated on their integration. The film option remains opt-in. [Padded audit](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/zp03-zp04/opt-in-film.jsonl) · [Audit summary](/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-zpad/zp03-zp04/review/film-audit-summary.json) · [Reviewed FFT-grid follow-up](/homes/deand/dev/NRSS_speed/test-reports/fft-registration-followup.ArgvS1/review/VALIDATION_GATES.md)
