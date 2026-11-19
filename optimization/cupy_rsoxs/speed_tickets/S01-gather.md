# S01 · Gather detector samples before scaling or reordering entire volumes

**Question.** How much time does the current dense FFT spend moving values that never reach a returned detector pixel? Compare the following changes individually and together against the current dense cuFFT baseline.

| Variant | Intervention | Reason to time separately |
|---|---|---|
| Gather first | On `tensor_coeff` gather each unique component at the current z0/z1 endpoints, then multiply by `1/(4π)` and project. Share Nxy between x and y, respecting the angle family. | `_projection_coefficients_from_fft_nt` currently scales full-volume triplets before gathering. This change still computes every Fourier frequency. |
| Native order | Keep the full FFT in native order. Map each detector read to the index returned by the incumbent Igor permutation on **each** axis. Try direct and tensor with each route's original normalization. | Removes full-volume shifts, including three in-place swaps per direct angle, at the expense of more complex reads. |
| Combined | Native-order endpoint gathering and compact projection. First use the readable gather, then compare a fused kernel if the gather dominates. | Supplies a common endpoint representation for S02 and S04. |

Inspect `_compute_fft_nt_components`, `_projection_coefficients_from_fft_pair`, `_fft_polarization_fields`, `_igor_axis_order` and `_replace_dc_component` in `src/NRSS/backends/cupy_rsoxs.py`. The Igor lookup is `(floor(n/2) − i) mod n` for an output index `i`. Apply the actual DC neighbor average in native order before serving a DC read. The source uses four neighbors for Z=1 and six for Z>1. Preserve the original endpoint projectors, intensity interpolation, invalid border, affine rotation, and angle weighting. The aligned axis and single-slice fast paths merit explicit attention.

**Run.** Use `one` and `angles` on both paths. Start small, then run `depth` and medium `angles`. Use one diagnostic C+D comparison and three uninstrumented pairs per arm. Compare a host-prewarmed 12-angle case for a finalist. Optional tensor component streaming in tiles of 1, 2 or all required components is a separate row if full component volumes still dominate. Count repeated material reads and include their time.

**Spot check.** Use a small asymmetric odd-XY case and an even case with complex contrast, include DC plus neighbors, one border and one nonzero z interpolation fraction. Check a whole small panel. Transposed axes or a stock `fftshift` approximation can pass a radial average.

**Decision.** Retain individual wins and the fastest equivalent combined path for subsequent matched runs. If native mapping slows projection enough to erase the gain, retain the incumbent permutation for that route. Record the speed and memory tradeoff separately.
