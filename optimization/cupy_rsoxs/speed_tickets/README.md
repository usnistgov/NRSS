# Current-workflow CuPy speed tickets

Campaign opened 2026-11-18 against `backend/cupy-cyrsoxs-clone`, commit `8aa24ec0c701b00a7b41f60196c5e72d3ba9ddd3`. The objective is a faster returned NRSS result for existing normal-incidence workflows on one GPU. Wall time on the user's full calculation determines priority. Peak memory constrains which implementations can run. Variable incident direction and its new detector/E-field physics start after the campaign closes.

Use the [measurement protocol](MEASUREMENTS.md) and the S00 launcher for matched comparisons. Record measurements in [RESULTS.md](RESULTS.md).

## Current priority

The qualified `compact_r2c` tensor default uses scalar fused projection
(S03) and fused angle weighting, rotation, and accumulation for multiple
directions (S05). S04's XY-first selective z evaluator passed accuracy and
yielded too little speedup for selection. The S05 batched common-q remap
passed S07 and now runs for multi-energy reduced results with a shared q grid.
The compact tensor basis also fuses Hann-window application when windowing is enabled. [Hann qualification](RESULTS.md#s07--separable-hann-window-qualification)
[Screen results](RESULTS.md#compact-route-screens-for-s03-s04-and-s05) ·
[Combined gate](RESULTS.md#s06--combined-compact-projection-and-rotation) ·
[Qualification](RESULTS.md#s07--combined-compact-projection-and-rotation-qualification) ·
[Batched common-q qualification](RESULTS.md#s07--batched-common-q-qualification)

S01 is optional and deferred. The default already gathers detector endpoints
from native FFT order. The remaining S01 work targets the dense routes.
The dense-only batching arm of S03, the direct-only work in S05, and
further qualification of S07b are also optional and deferred.

| Original order | Ticket | What the comparison decides | Depends on |
|---|---|---|---|
| 0 | [S00 · Establish controls and extend the harness](S00-harness.md) | How much time each public workflow takes, how repeatable it is, and whether the intended route ran | Shipped launcher |
| 1 | [S01 · Gather the detector endpoints in native FFT order](S01-gather.md) | Time saved by avoiding full-volume scaling and reordering, measured separately | S00 |
| 2 | [S02 · Reuse morphology transforms across energy](S02-energy-basis.md) | Crossover in energy count for optical-constant-independent FFTs | S00. Use S01's sampler if it is ready. A dense FFT plus reference gather also works. |
| 3 | [S03 · Broadcast the energy calculation in bounded tiles](S03-energy-tiles.md) | Incremental benefit of batching versus transform reuse, and the best tile size for full and reduced outputs | S00 for the tile-only control. S02 for the combined comparison. |
| 4 | [S04 · Compute just the z modes the detector needs](S04-selected-z.md) | Where selective Fourier evaluation beats cuFFT on full returned panels | S00 and an endpoint map from S01 or the incumbent |
| 5 | [S05 · Address the remaining measured bottleneck](S05-residual.md) | Whether rotation, direct polarization, or output handling merits a dedicated kernel | Timings from S01–S04 |
| 6 | [S06 · Combine winners and choose crossover rules](S06-combine.md) | Which workloads benefit from each route on realistic inputs | Screened candidates |
| 7 | [S07 · Run the full panels before a push](S07-promotion.md) | Whether an exact revision is ready for promotion | S06 and candidate-enabled validation |
| 7b | [S07b · Reuse Fourier transforms in direct polarization](S07b-direct-energy-basis.md) | Whether a fixed-direction basis reduces scan time or peak VRAM relative to tensor reuse | S02 and S07 measurement controls |

After every ticket, record the incumbent, result and next action. Negative results close experiments.

## Source and scope

The current source loops over energies in [the backend](../../../src/NRSS/backends/cupy_rsoxs.py). `result_chunk_size` already controls output buffering and some integration. The current direct path repeats three component FFTs for each E direction at each energy. The tensor path needs three components for one aligned direction and up to five for a general angle family, reused across its E-angle sweep. The CuPy default uses compact tensor for full-depth FP32 inputs at any energy count. The runtime [contract](../../../src/NRSS/backends/contracts.py) configures `direct_polarization` as the per-energy fallback. The old optimization matrix defaults to `tensor_coeff` and a legacy isotropic input representation. All comparison cases set the path and representation explicitly.

This board supersedes the open CuPy speed queues. The [backend specification](../backend_spec.md), [accepted state](../accepted_state.md), [validation record](../validation_and_status.md), and path histories retain reference value. Previously accepted host FP32 reusable staging and host result streaming are starting conditions. The old CPU staging experiment, device-only orientation cache retry, allocator tuning and expert z-collapse are outside the first round. The agent can reopen an item with new timing evidence. No candidate here requires CyRSoXS as its implementation model.
