# FS02 · Qualify the isolated branch and issue a merge verdict

Compare the pinned clone-branch head A with the final FS01 candidate B through public `Morphology.run`. First establish that two identical A workers repeat on the same physical GPU. Use the supplied [matrix launcher](../../../tests/validation/dev/cupy_rsoxs_sampler/run_matrix.py) for A/A and A/B. Each pair has identical inputs, backend options, output shape, result chunk size and residency. A cutoff comparison means A and B **both** use the cutoff.

## Required timing rows

Shapes are Z × Y × X. Each 101-energy workload runs in **one** public call. The three single-energy controls measure the setup cost with minimal opportunity for reuse.

| ID in launcher | Morphology / logical grid | Energies and E directions | Host return |
| --- | --- | --- | --- |
| `core-small-one`, `core-medium-one`, `core-large-one` | The three native CoreShell sizes below | 285 eV, one direction | Unbounded FP32 detector, chunk 1 |
| `core-small` | CoreShell `32 × 512 × 512`, 2.5 nm | 280.0–290.0 eV in 0.1 eV steps, 12 directions | Unbounded FP32 detector, chunk 1 |
| `core-medium` | CoreShell `64 × 1024 × 1024`, 1.25 nm | Same scan and directions | Same |
| `core-large` | CoreShell `96 × 1536 × 1536`, 2.5/3 nm | Same scan and directions | Same |
| `detector-1` | Frozen `52³` lipid crop → `256 × 1024 × 1024` | 282.55–292.55 eV in 0.1 eV steps, one direction | Unbounded FP32 detector, chunk 1 |
| `detector-4` | Same crop and detector grid | Same lipid scan | Unbounded FP32 detector, chunk 4 |
| `ia-full` | Same crop → `1024³` | Same lipid scan | Unbounded FP64 native I/A, chunk 101 |
| `ia-qmax` | Same crop → `1024³` | Same lipid scan | Native I/A, `q_max=0.8` nm⁻¹, chunk 101 |

CoreShell uses vendored spectra, host input, full material depth, FP32, `enum_contract`, host detector output, window off and rotation mask off. `EAngleRotation` is `[start, increment, end]` in degrees. The 12-direction schedule is `[0,15,165]`. The lipid scan uses `[0,0,0]`, 1.5 nm pitch, head/tail/water material IDs 1/2/3, and water as the virtual fill. The worker validates the frozen crop hash, optical source hashes, anchor energy and scan order. It records the optical NPZ hash. Use the ZP09 fixture on the project's GPU host. Generated generic morphologies cannot qualify a lipid row.

The two lipid detector chunks exercise the automatic multi-energy projection choices. At this revision the dense host detector scan is eligible for reloaded tiles at chunk 1 and shared tiles at chunk 4. Record actual dispatch and workspace fallbacks. The cube rows protect the fitting path at its qualified large chunk size. Hold that chunk fixed when checking the cutoff. The prior campaign found low-bit changes in I between chunk sizes and between physical GPUs. Z-only, fill and window variants run in the correctness suite. The performance decision covers six native and four lipid configurations.

## Measurement and controls

- Pin the physical GPU, driver/runtime/cuFFT/CuPy versions, Python environment, CPU affinity, input hashes and both Git SHAs. Record installed NRSS source paths and source hashes for each worker. Keep clocks/power state comparable, use the same host and avoid competing GPU jobs. On large runs, record free device memory and CPU memory availability.
- The CoreShell launcher uses the established construction-to-synchronized-return boundary with one warmup on a fresh morphology in each fresh worker. Its first-call value is `first_call_seconds` from that warmup's primary timer. The warm result is `primary_seconds` from the timed new morphology. The frozen lipid harness measures synchronized `run(return_xarray=False)` on an already constructed morphology. It records the first call and five subsequent calls. Reduce those five to one warm median **per worker**. All requested host transfers are inside the respective timing boundaries. Treat the two workflow families separately.
- Alternate A/B then B/A worker order. Use eight independent worker pairs per A/A and A/B timing row as a starting sample. Optionally add another interleaved batch on a quiet GPU if dispersion prevents a decision. When combining batches, keep the exact revision, fixture, hardware and policy fixed. Decide the additional sample count before looking at the next batch. Save every raw sample and its order. Check the time ordering for drift.
- Keep correctness, profiling and memory observation out of timing workers. In the stages pass compare selected routes, fallbacks, basis builds, FFT method/shape/dtype/count, compact basis bytes/rows, union endpoints, selected X columns, planner layout, tile widths/counts and any `q_max` demand counters that exist on the measured route. Expect one compact basis per scan and the same actual work as A. The launcher checks execution reports for each row. Inspect segment times only to explain a discrepancy.
- The memory pass uses fresh workers and separate NVML process observation at 5 ms. Record peak per-worker GPU MiB, host peak RSS, sample count and interval, free memory at launch and any additional live device buffers or larger workspaces. For CoreShell, match the worker PID in the observer process map (`worker_pid` or the PID in the run evidence). The raw process peak is separate from whole-device peak delta and ending CuPy pool reservation. Repeat a suspicious increase with matched A/A controls. Hold if there is a reproducible peak increase beyond the A/A spread and polling resolution, a new allocation of the dense grid, or any new capacity failure. Missing process-peak evidence is inconclusive for memory.

## Equivalence and merge decision

The proposed engineering margin is **±2%**, configurable with `--margin-percent` **before B timings**. It is a sensitivity target for this refactor. A noisy host does not widen it automatically. For each row use the within-pair ratio `T_B/T_A`, its median and a two-sided distribution-free median interval with at least 90% coverage (binomial order statistics, at most 5% in each tail). Ratios above one mean slower. Run the same analysis for first and warm results. The launcher saves `timing-decisions.json` for each timing batch.

| Finding for one timing row | Disposition |
| --- | --- |
| A/A interval includes 1 and lies inside the margin. A/B interval does the same. | Neutral within the stated sensitivity |
| A/A has drift, or either interval is too wide to establish neutrality | Inconclusive. Diagnose order, contention and run variance, then remeasure |
| A/B interval lies wholly above or below 1 with calibrated A/A | Report the resolved effect in percent even if small. Hold a slowdown. Review a speedup alongside identical work counters and the diff before recommending merge |

The timing phase returns code 2 after completing all selected rows if any decision needs review. A failed execution or parity check returns an error and retains completed evidence. An inconclusive result supplies no basis for a claim of neutrality. Use actual paired output for each size and pad setting. Never pool ratios across configurations to hide a slow case. When pooling valid batches for one row, use `verify_results.py timing --aa <A/A folders> --ab <A/B folders> --metric primary_seconds` or `--metric first_seconds`.

## Numerical and regression gates

1. A/A then A/B on every returned value at every requested energy. Require identical dimensions, channel names, dtypes, masks, coordinates, and value bits including signed zero and NaN payloads. Native CoreShell's zero-tolerance numeric summary alone does not compare signed zeros or NaN payloads. The included verifier checks the NPZ arrays' underlying bytes. The lipid runner uses the repository's byte oracle on complete, identically configured products. Keep per-energy mismatch counts and first offending indices/bit patterns.
2. On the candidate, run the standalone Fourier oracle against the public sampler. Run the focused CPU/GPU suites and full maintained reports from each tree in the same environment. The maintained reports exercise physics and integration beyond the timing fixtures. Preserve the test names, outcomes, tolerances, reference hashes and plots. A previously known failure requires the same current-parent outcome and diagnostic; investigate any new failure. The previous Qmax report identified PyHyperScattering-integrated sphere and multi-energy discrepancies that make a current parent comparison important.
3. Compare geometry-sensitive outputs including registered DC, off-axis/Bragg, mixed odd/even shape, singleton depth, explicit versus virtual padding at equal logical shape, material fill under XY Hann, Z sum/residues, identity padding and q-window boundary/upper-neighbor behavior. Confirm full panel parity on the same algorithm and hardware. Literal-versus-virtual oracles use their existing numerical tolerances; byte parity applies between revisions on the **same route**.
4. Ensure `nrss_sals` is unnecessary for installing, importing, and running NRSS. The candidate's public method is exercised by an external caller and continues to sample phase correctly. Confirm no new GPU kernels, transfers, workspace or changed default selectors in production RSoXS.

```bash
for tree in "$A" "$B"; do
    (cd "$tree" && PYTHONPATH="$tree/src:$tree" CUDA_VISIBLE_DEVICES="$GPU" \
        python -m pytest -q tests/smoke/test_cupy_energy_basis.py tests/smoke/test_cupy_virtual_z.py \
        tests/smoke/test_cupy_q_window.py tests/smoke/test_ewald_fft_grid.py \
        tests/validation/test_ewald_fft_registration.py --nrss-path cupy_tensor_coeff)
done
for arm in parent candidate; do
    if [[ "$arm" == parent ]]; then tree="$A"; else tree="$B"; fi
    (cd "$tree" && PYTHONPATH="$tree/src:$tree" CUDA_VISIBLE_DEVICES="$GPU" \
        scripts/run_local_test_report.sh -e "$CONDA_DEFAULT_ENV" -r "$OUT/report-$arm")
done
```

Run the candidate's new maintained public-API tests as well. If shared code affects `compact_c2c` or `direct_r2c`, run the focused compatibility tests for those modes and the applicable optional direct report on both sides. Record a pre-existing failure by paired evidence. If the target moves, retest the final candidate against the updated merge parent.

## Qualification record

Record A/B SHAs and GPU UUID/stack once. Provide one row per timed configuration with A and B first/warm milliseconds, paired delta and interval, A/A interval, sampled peak RSS/VRAM, mismatch count, actual route, basis and FFT counts, and links to raw artifacts. Attach the public sampler oracle, source diff and full-report comparisons. Name every unmeasured case and why. The verdict is **ready for review**, **hold for regression** or **inconclusive**. The merge decision follows this evidence.

[CoreShell sizes and optics schedules](../../../tests/validation/dev/cupy_rsoxs_optimization/run_speed_tickets.py) · [Frozen fixture and reference checks](../../../tests/validation/dev/cupy_rsoxs_virtual_3dpad/run_padded_scan_baseline.py) · [QM05 byte oracle](../../../tests/validation/dev/cupy_rsoxs_qmax/qmax_reference.py)
