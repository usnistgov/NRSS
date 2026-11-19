# S03 · Measure bounded broadcasting across photon energy

Start with arm C, the current `compact_r2c` default. Measure a fused scalar
projection that consumes cached Fourier samples directly, then test bounded
energy tiles with equivalent inputs and outputs. Check the material-count
scan where variable detector support increased peak VRAM. Arm B is optional
and deferred.

**Question.** Once the Fourier samples are available, how quickly can NRSS turn them into a scan? The compact route currently projects one energy per iteration of the Python loop. The existing `result_chunk_size` buffers completed images for host transfer and can batch their integration. Compute tiling batches the projection itself.

Run this four-way comparison on the **same geometry and returned product**. Use the same native-order/gather implementation across arms if available. If one arm needs a different sampler, measure its single-energy tile control and disclose the difference.

| Fourier inputs | Compute one energy at a time | Compute a bounded tile of energies |
|---|---|---|
| Incumbent energy-specific tensor FFTs | A · Existing production control | B · Gather compact endpoints per energy, then batch the downstream projector, angle weighting or output stages. Count the same source FFTs as A. |
| Reused real morphology basis | C · S02 scalar recombination | D · Broadcast the complex optical coefficient table across compact endpoint samples, then use each energy's geometry and projector. Keep the source FFT count equal to C. |

A→C measures transform reuse. C→D measures compute broadcasting on top of reuse. A→B measures batching with conventional Fourier inputs. Compare D with the fastest existing production route for the complete workflow. Tiling the expensive dense transforms themselves is an optional separate B variant, starting at width 2 and charging cuFFT workspace.

Form the small coefficient table with axes (energy, material, coefficient) from the current helper before entering the device loop. For the first D kernel, map a tile of (energy, detector pixel, endpoint) to the cached basis samples, contract the complex coefficients and form the projected x/y/xy intensity terms. Each energy carries its own k, z0/z1, interpolation fractions, masks and q corrections. Try a CuPy broadcast as a reference and a fused kernel if it avoids large temporary arrays. Preserve order, per-energy outputs, event ownership and the final partial tile. The intended intermediates scale with tile width × detector support. Report bytes if a variant introduces a tile × full-volume buffer.

**Run.** First `three` and `energy20` with compute widths 1, 2, 4 and 8 where they fit, keeping `result_chunk_size=1` and a host detector result. Rank the few best settings on `energy101`, whose tail exercises partial tiles. Use widths 2 or 4 on the three-energy test to catch boundary handling. Independently vary result chunk size, e.g. 1 and 4, at fixed compute width for the finalist. Repeat on `i_a` with identical q-grid mode, mask, reduction and output residency in both arms. Include a nonmonotonic energy selection in the quick correctness pass. Stop increasing width at the first sustained plateau or memory pressure. Bandwidth, Python launches or return transfers may now dominate.

**Finish.** Report first result latency if output streaming matters, total scan seconds, energies/s, basis builds, FFTs, kernels/tiles and last tile length, output-copy time and externally observed peak. If batching adds little once the basis is reused, a scalar loop over compact fields is a valid winner. Record crossover for 1, 3, 20 and 101 energies before setting any automatic policy.
