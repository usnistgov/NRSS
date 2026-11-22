# ZP10 · Reproduce the scan and separate planning, computation and delivery

**Question.** Which measured work explains the extra 274.9 ms from 20 to 101 energies on the qualified XYZ vesicle? Which stages account for the 4096→8192 sphere increase? Deliver a matched A/A control and a timing budget that will set the order of subsequent experiments.

**Primary fixture.** On Wheatley use the ZP09 `fixture.json`, `benchmark.py` and `vesicle-energy-results.json` in `test-reports/virtual-3dpad/ZP09-vesicle-energies.vrbPPq/` under the original worktree. The exact original paths are in the [qualification ledger](ZP05-09-RESULTS.md). Recover the frozen 52³ crop, optical tables/hash, water ID 3, accepted energy subset and original benchmark scope. Make the fixture selectable by path/manifest and the checkout and output roots configurable. Verify hashes, shape, pitch and pre-existing control intensities. Record an explicit “frozen fixture unavailable” if local artifacts cannot be located. The checked-in `generic` case is an additional reproducible control. Its different optics, dimensions and pitch deserve its own row.

| Screen | Required matched cases |
|---|---|
| Multi-energy first | Frozen vesicle, `(256,1024,1024)` target, one direction, 1/20/101 energies, same morphology and energy list across arms. First run and five or more warm calls. Actual default plus a compact-pinned comparison |
| Use-case controls | 20 energies with 12 directions, all 101 energies in non-monotone order, water and oriented exterior material 2, and 101-energy built-in `i_a`. Record one batched scan and a loop of single-energy public calls with the same return policy |
| Resolution first | Existing 128³ sphere at 285 eV in vacuum and material 2. Screen 1024/2048/4096/8192 and use all seven rungs for promotion. Add the 52³ vesicle at fixed XY with different Mz and fixed Mz with different XY. Include a 20-energy 2048² detector as a cross-check of both changes |
| Adverse geometry | Widely spaced soft/hard energies, a noninteger rectangular XYZ pad and the existing vacuum Z-only and native compact paths. The geometric probe's four-energy case is a useful stress test for gapped bins |

**Instrument outside primary timing.** Extend `_ExecutionReport`, `_CUPY_SEGMENT_MEASUREMENTS` and the harness `STAGES` list together. Existing `basis_build` includes `zpad_plan`, XY and axial captures. Stage D includes some geometry creation. Stage E combines angle rotation/accumulation. Nested and overlapping CUDA events and CPU copy waits cannot simply be summed. Use disjoint subranges plus an uninstrumented wall timer, and capture one representative stream timeline if needed. Collect the following measurements.

- Runtime input staging, source basis count, active fields, distinct energies, requested/valid pixels, requested and distinct endpoints, unique radius keys, layout of the union, number/bytes of sorts or merges, geometry constructions and cache hits in planning versus projection. Count actual calls in the timed region, excluding the optional post-run `--geometry` host diagnostic. Log coordinate geometry and mask construction separately.
- Per-field XY FFT calls **and transformed planes**, tile depth and retries, cuFFT workspace if exposed, time gathering/axially accumulating the union, and basis bytes. Separate window and material-fill work.
- Projection lookup/mixing, angle warp, native integration, pinned host allocation, D2H bytes and event waits. Compare host detector, device detector retained without `to_xarray`, and built-in `i_a` individually. Calibrate same-size device-to-pinned-host copies independently (101 × 1024² FP32 = 404 MiB). Document return type and terminal synchronization. Equal output bytes are a condition for claims about compute speed.

**Harness entry points.** The supplied [development patch](ZP10-shared-benchmark.patch) adds shared XYZ settings and corrects the prior worker's Z-only reporting of an XYZ request. From the branch root after the [startup procedure](ZP10-15-START.md):

```bash
speed=tests/validation/dev/cupy_rsoxs_optimization/run_speed_tickets.py
parent=../NRSS-zp-speed-parent
out=test-reports/padded-scan-speed
common=(--fixture generic --case energy20 --path default --rotation off --resident host
        --xyz-pad-to 256 512 512 --xyz-pad-material 2 --gpu 0)
python "$speed" "${common[@]}" --baseline-root "$parent" --aa --repeats 5 --out-root "$out/aa"
python "$speed" "${common[@]}" --baseline-root "$parent" --candidate-root "$PWD" \
    --stages --expect-route virtual_3d_sum --repeats 1 --out-root "$out/counters"
python "$speed" "${common[@]}" --baseline-root "$parent" --candidate-root "$PWD" \
    --memory --memory-poll-interval 0.01 --repeats 1 --out-root "$out/memory"
```

Use a dedicated physical GPU, identical CPU/NUMA placement, environment and source hashes. Interleave A/B then B/A workers. Report paired medians, variability, warm distributions and first-call medians from fresh workers. Run staged and memory measurements separately. In the generic worker `primary_seconds` includes its own construction boundary. For the primary frozen fixture keep the original synchronized `Morphology.run` timing after construction. On a memory pass report sampled worker VRAM, available VRAM, allocator/pool and host RSS. Freeze the baseline before implementing changes and keep the earlier literal comparison as context.
