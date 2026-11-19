# S04 · Evaluate only the Fourier z samples needed by today's detector

For the current default, compare the selective evaluator with the real 3D
FFT used to build the compact tensor basis. Use the same endpoint union,
energy coefficients, projection and output in both arms.

**Question.** For a thin film and a full returned detector, can selective evaluation beat a dense cuFFT once geometry setup, launches and gathers are included? Use the existing integer z endpoints and interpolation for all A/B runs.

The following CPU counts evaluate the branch's `_detector_geometry` and `_detector_projection_geometry` logic in float32 using NumPy. They count distinct valid (z,y,x) interpolation endpoints before E-angle image rotation for the entire panel. Auxiliary DC neighbors are excluded. Confirm counts on the local GPU when registering each shape.

| Geometry | Energies | Unique valid (y,x) columns | Unique endpoints over scan | Different z planes anywhere | Fraction of FFT grid |
|---|---:|---:|---:|---:|---:|
| Small | 1 at 285 eV | 239,424 | 478,848 | 17 of 32 | 5.708% |
| Small | 20 exact keys | 242,028 | 562,540 | 17 of 32 | 6.706% |
| Small | 101 exact keys | 242,296 | 567,024 | 17 of 32 | 6.759% |
| Medium | 1 at 285 eV | 271,464 | 542,928 | 20 of 64 | 0.809% |
| Medium | 20 exact keys | 280,108 | 710,222 | 20 of 64 | 1.058% |
| Medium | 101 exact keys | 280,996 | 730,214 | 20 of 64 | 1.088% |

At medium size the 101-energy scan needs about 2.60 distinct z nodes per active column. Computing whole z planes would still cover 20/64 of the volume. The counts budget **output storage**. Both approaches must read the material and perform substantial in-plane Fourier work.

| Candidate | Implementation | Likely controlling variable |
|---|---|---|
| Control | Dense 3D cuFFT plus the fastest **equivalent** endpoint gather from S01. Keep the released incumbent as an anchor. | FFT size and device bandwidth |
| XY first | Batched 2D cuFFT on the actual z slices. For each active native (kx,ky), sum the short complex z series only at its required integer modes. Fuse indexing, phase application and accumulation in a GPU launch over columns. | Distinct modes **per column**, material depth, coalescing and the price of staging/transposes |
| z first | Compute selected integer z modes across the source planes, then a 2D FFT per retained mode. Gather only the endpoints required at each (kx,ky). | Number of **global** z planes; attractive when a genuinely requested detector region uses few modes |
| Conditional | Pruned or fused z FFT with a detector-only store, or phase-interleaved z FFT for an actual padding workload. Try only if the simpler measurements justify the implementation. | cuFFT throughput versus launch/workspace cost |

Use the incumbent native mode from the Igor mapping in the exponent `exp(−2πi k_z n/N_z)` and preserve window, normalization, DC neighbors and signs. Add an asymmetric two-layer fixture, including odd sizes and `N_z=1`. First compare at the exact integer nodes and then run the current endpoint intensity interpolation. A direct sum at continuously varying qz would change the numerical target and belongs to later instrument work.

**Run.** Small and medium `one`/`angles`, then fixed 1024 × 1024 XY at the medium voxel size with Z=32,64,128. Explicitly distinguish a physically thicker/extended material from a padded transform. Where a 32-plane fixture would clip the medium CoreShell shell, use a deliberately positioned, contained fixture and describe its z dependence. Charge plan construction, mapping and compaction to the primary result. Compare sequential versus tiled GPU launches only when both return the same panel. For multi-energy basis mode, use the energy union and record its growth. Compare S02 with dense basis FFTs against S02 with the selected evaluator before attributing their gains to z selection.

**Optional padding test.** Only if padding is used for a candidate case, compare material depth N evaluated on target transform depth M with the same explicit zero-padded *scattering source* on the same target grid. Keep physical voxel size, window support, placement, DC policy and detector coordinates identical. No silent geometry compression or `z_collapse_mode='mean'`. A cropped ROI is a separately named, matched workload with all rotation/reduction interpolation source pixels included. Full detector results determine a general speed claim.

**Decision.** Report the faster evaluator by (shape, energy count, requested nodes) and the point at which dense cuFFT wins. Capacity-only gains can support a separate opt-in use case.
