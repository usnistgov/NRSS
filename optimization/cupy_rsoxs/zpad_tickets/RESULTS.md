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
