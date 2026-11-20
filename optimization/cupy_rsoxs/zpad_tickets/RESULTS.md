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

The S04 selected-z prototype is available in the local historical checkout at `/homes/deand/dev/NRSS/test-reports/S04-selected-z-rtB3m1/candidate-source`. The execution options in the matched harness, virtual-versus-explicit tests, GPU timing and VRAM sweep, and indexed high-q validation are open.
