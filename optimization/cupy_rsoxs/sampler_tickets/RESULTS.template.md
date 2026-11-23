# FS02 qualification record

**GPU qualification is pending.** Fill this record from the tested revisions.

| Provenance | Measured value |
| --- | --- |
| Merge target, baseline SHA, candidate SHA and source hashes | |
| Host, physical GPU UUID, driver, CUDA/cuFFT, CuPy, Python | |
| Pinned optical, coordinate and crop hashes | |
| Engineering margin chosen before timing, fresh pairs, warms, NVML interval | |
| Full parent and candidate reports, API oracle, complete command plans | |

Give primary warm times in milliseconds. In the CI columns, report median percent difference and interval with B/A as the timing ratio. Repeat the timing table for first calls or link its complete machine-readable equivalent. Keep per-row linked evidence for A/A variation, any fallbacks and timing-order trends.

| Configuration | Warm A / B, ms | Warm B/A change and interval | First B/A change and interval | A/A interval, warm / first | Peak VRAM A / B, MiB | Peak RSS A / B, MiB | Bit mismatches | Basis builds and FFTs A / B | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Native small, one energy | | | | | | | | | |
| Native medium, one energy | | | | | | | | | |
| Native large, one energy | | | | | | | | | |
| Native small, 101 energies | | | | | | | | | |
| Native medium, 101 energies | | | | | | | | | |
| Native large, 101 energies | | | | | | | | | |
| Lipid detector, chunk 1 | | | | | | | | | |
| Lipid detector, chunk 4 | | | | | | | | | |
| Lipid I/A, chunk 101, full | | | | | | | | | |
| Lipid I/A, chunk 101, 0.8 nm⁻¹ | | | | | | | | | |

Record numerical oracle results, complete stage-signature comparisons, any pre-existing regression-suite failures, missing measurements, and their evidence links. State whether the target branch still points to the tested baseline. Finish with one of three recommendations. Mark the branch ready for review, hold for regression, or inconclusive.
