# S05 · Optimize the remaining measured stage

Re-profile one single-energy and one 20-energy full run on the leading compact
implementation after the projection screen. Choose a stage that occupies
enough primary time to deliver a meaningful workflow speedup. Use Amdahl's
bound from nonoverlapping wall time. The direct-accumulation row is optional
and deferred. Leave the ticket closed as deferred if no target qualifies.

After fused projection and rotation, reduced I/A is the remaining measured
target. The 101-energy profile spent 3.56 of 3.83 seconds in reduction at
output chunk size one. Start with the common-q remap, which calls
interpolation separately for each azimuthal row even though the rows share
the source and output q coordinates for an energy.
[Profile](RESULTS.md#s05-follow-up-on-the-combined-compact-candidate)

The batched common-q candidate passed the matched screen. At output chunk
size four, small 101-energy I/A took 0.29 seconds versus 3.54 seconds
for the per-row implementation, with identical output and sampled peak
VRAM. The candidate passed S07 and was promoted. [Qualification](RESULTS.md#s07--batched-common-q-qualification)
[Screen results](RESULTS.md#s05--batched-common-q-interpolation)

Separable Hann fusion passed S07 and was promoted for windowed compact tensor runs. [Qualification](RESULTS.md#s07--separable-hann-window-qualification)

| Observation | Experiment | Workload / check |
|---|---|---|
| E-angle rotation and accumulation dominate tensor time | Fuse x/y/xy weighting, affine sampling, mask accounting and detector accumulation or test the smaller rotation-plus-accumulation change first. Consider exact quarter turns separately where actual center and interpolation match. | `angles` versus one E direction, then `energy20` with 12 angles. Compare pre-average images on asymmetric odd/even grids and rectangular shapes, fill values and valid counts for both RotMask states. |
| Host FP32 direct path spends substantial time in B after reusable staging | Compare existing fused direct accumulation, existing optional `direct_isotropic_mode="cached_base"` and an anisotropic-only consumer of `φ,u` with independently accumulated isotropic terms. Charge extra kernels, initialization and staging. | Host `energy20` with 12 angles plus `one`. Measure A2+B and primary seconds. Prior host reusable staging alone moved a historical 20-energy median from 1.7327 s to 1.7289 s. That small change motivates profiling the current source first. |
| A real Hann-window workflow spends time allocating/multiplying the full window | Apply separable Hann factors in an existing field or basis generation kernel. Match endpoint behavior, especially size one, and keep the value convention of `_window_tensor`. | Same window on in both arms and a window-off control. |
| Long scan time shifts to integration or D2H | Time actual reduction, pinned buffers and stream joins. Vary the existing output chunk at fixed compute width. Compare optional fused weighting/reduction only for a supported reduced product. | `energy101` with `detector` versus its detector baseline, `i_a` versus its own baseline. Check common q coordinates, sector weights, empty bins and asynchronous lifetime. |

Start with one diagnostic pass per variant and the three-pair screen. Update only the winning implementation. The earlier CPU-precompute, allocator-only, half-precision and z-collapse ideas answer different questions. Evidence from an actual FP32 full-workflow bottleneck is the condition for reopening a historical speed experiment.
