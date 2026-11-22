# ZP13 · Reduce full-plane transform and axial traffic at high resolution

**First resolution experiment.** Retain the current cuFFT + exact axial-sum algorithm. The 8192 sphere takes 1.2165 s for a complete 8192² host panel. It processes a single occupied plane per XY batch and visits a union of 22.28 million endpoints on every batch, per captured field. This is a different cost from producing a bigger detector for each energy.

**Separate Z and XY.** Run the common 128³ sphere at its seven qualified sizes. For development use 1024, 2048, 4096 and 8192 first, with both vacuum and nonzero matrix. For the 52³ crop, hold `(My,Mx)=(1024,1024)` and vary `Mz`, then hold `Mz=256` and increase lateral sampling to 2048 and 4096. Keep physical crop, pitch, energy, E angle, window and returned layout fixed. Report wall time, valid and nominal Mpixel/s, bytes of the returned panel and effective working bytes. A constant throughput would yield 4× more time upon doubling both lateral dimensions. Compare against that scale as well as the incumbent at the **same** resolution.

1. **Tile-depth ablation.** Measure the current 64 MiB real-plane rule, private fixed depths 1/2/4/8 where feasible at the upper rungs, and one workspace-aware choice. Include the actual RFFT output bytes, cuFFT workspace/plan cache, union/basis bytes, active field count, free memory, output residency and pool high-water. At 8192 each real plane takes 256 MiB even at tile depth 1. Track allocation retries and release. Reuse bounded planning/buffers when eligible and time first and warm calls. A larger slab can reduce launches and global partial writes. Confirm whether its FFT and workspace cost offsets that saving. Preserve the existing smaller-tile retry and clear failure when one plane cannot fit.
2. **Axial gather and accumulation.** Distinguish XY-transform time from axial kernel time and global partial-sum traffic. Increasing the tile reduces traversals of the union. Benchmark one endpoint/thread versus grouping endpoints from the same canonical native XY column so several demanded kz bins share the XY spectra. Compare global ID decoding/modulo with predecoded relative modes only if saved arithmetic pays for extra memory traffic. Profile non-coalesced loads and writes. On the single-energy sphere the typical logical column needs two axial samples. On a mixed-energy column it can need many. Keep the DC stencil and reflected native modes in the measured counts.
3. **Avoid Fourier fields with no contribution.** Count current basis rows and identify rows with coefficients exactly zero across *all* requested energies and active tensor components. Compare skipping those transforms with the unmodified field builder. The vacuum sphere provides a clean case. Add a material whose contrast is zero at one energy and nonzero later to prevent accidental pruning. A separate optional experiment recognizes an exactly complementary two-material particle field and derives the second amplitude using the finite-box, window and DC terms. Use complementarity only when the fixture or input representation guarantees it, or after explicitly verifying it at the required precision. The validator permits approximate volume closure. Leave the general anisotropic, absorbing and nonzero-fill calculations fully supported. Account for the verification cost.
4. **Band-limited internal work.** The current projector writes a complete panel even though many pixels are invalid. An implementation may restrict the *internal* launch to valid detector support and initialize the rest according to the requested mask. Count full-output initialization, rotations across masked borders, transfers and reductions. Probe this as a separate ablation only if ZP10 attributes substantial time to these operations.

**Decision.** Publish the sphere ladder for retained variants and report the 2048² 20-energy cross-case plus the one-direction and 12-direction vesicle. Keep a simple cuFFT control for shapes where it wins. Match logical resolution, public q range, energy count and output layout in each detector speed comparison. Retain parent-accurate pixels and the qualified memory advantage over the dense box.

The repository sphere driver can screen the qualified route before experimental variants are added. Run from the candidate worktree. The example measures the parent, then the candidate. Reverse order on the repeat.

```bash
sphere=tests/validation/dev/cupy_rsoxs_virtual_3dpad/run_sphere_ladder.py
out=test-reports/padded-scan-speed/sphere-screen
python "$sphere" --plan --sizes 1024 2048 4096 8192
python "$sphere" --run --gpu 0 --nrss-root ../NRSS-zp-speed-parent --arm xyz --fill vacuum \
    --sizes 1024 2048 4096 8192 --repeats 5 --out "$out/parent"
python "$sphere" --run --gpu 0 --nrss-root "$PWD" --arm xyz --fill vacuum \
    --sizes 1024 2048 4096 8192 --repeats 5 --out "$out/candidate"
```

For the final qualification run both `--fill vacuum` and `--fill 2`, omit `--sizes` to use the full ladder and include `--verify-full-voxel` on the accuracy worker via the campaign command. The driver runs timing, memory and accuracy in separate workers.
