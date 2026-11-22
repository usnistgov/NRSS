# ZP09 · Existing workflows, multi-energy cost and promotion

Run the parent and candidate in the same CUDA/conda environment on the same physical GPU. Identify both SHAs, the imported paths of `NRSS` and `CyRSoXS`, GPU model, VRAM, driver, CUDA, CuPy, host RAM, dtype, residency, material fractions, energies, angles, window, output layout and logical detector. Set aside a quiet benchmark interval. Use public `Morphology.run` calls for scientific and end-to-end results, and separate diagnostic runs for route counters.

## Current workflows and multi-energy reuse

Use [the speed harness](../../../tests/validation/dev/cupy_rsoxs_optimization/run_speed_tickets.py) for paired native and Z regression runs. Its `--case energy20` and `energy101` cases exercise the earlier energy optimizations. The [ZP08 program](../../../tests/validation/dev/cupy_rsoxs_virtual_3dpad/run_sphere_ladder.py) supplies the single-energy matched sphere fixture and three calculation arms. Extend the speed-harness worker or sphere driver with a shared multi-energy fixture and exposure of both arms' pad choices for the following scan. Plainly record which harness produced each row.

| Matched workload | Measurements and verification |
|---|---|
| Parent native vs candidate native, 1/20/101 energies | Paired A/A and A/B, warm and first synchronized times plus peak memory. Full output parity with parent on existing grids |
| Parent vacuum Z vs candidate vacuum Z, both current methods | Single direction and routine 12-angle schedules, XY unwindowed and Hann, same occupied film and logical depth. Verify eligible route calls and preserved axes |
| Literal XYZ and laterally explicit Z controls vs the new XYZ route | ZP08 matched sphere on every feasible common target. Give first/warm ratios and process-peak VRAM ratios for matching shapes, separate from the native 128 control |
| Fixed particle crop plus matrix, 1/20/101 energies | At least one 20-energy medium case with a routine rotation schedule and one small full 101-energy comparison. Use common input geometry and identical energy-dependent optics in all arms. Compare vacuum and nonzero selected fill on the same schedule, plus one non-enum exterior with genuinely anisotropic interior |
| Detector, integrated and derived observables | At manageable shapes compare host/device and all returned channels, masks, intersection/union q axes and energy order. Exercise a nontrivial angle on a rectangular padded detector and an off-axis registered reflection |

For the scan, request all energies through one public call and compare with a loop of single-energy public calls on the same fixed object. Time each end-to-end including the same output policy. Report first/warm total seconds, milliseconds per energy, peak worker VRAM and peak host RSS. Use distinct optical constants across energies and a non-monotone requested order as an additional correctness case. Confirm fixed-geometry basis FFT and fill-mask counts track actual tiles, not one rebuild per energy or per E angle when the geometry and tile are shared. Keep one energy per result chunk for a common baseline, then measure one larger supported chunk. Record union endpoint count, tile count and FFT work versus energy. Large multi-energy cubes may need throughput-oriented streamed or integrated output. Compare like layouts. The fixed 8192 sphere qualification remains a complete single-energy detector.

For example, from the candidate worktree, after setting `baseline` to the pristine parent path:

```bash
speed=tests/validation/dev/cupy_rsoxs_optimization/run_speed_tickets.py
baseline=../NRSS-3dpad-baseline
python "$speed" --case energy20 --path default --rotation 12 --baseline-root "$baseline" --aa --repeats 5 --memory
python "$speed" --case energy20 --path default --rotation 12 --baseline-root "$baseline" --candidate-root "$PWD" --repeats 5 --accuracy full --memory
python "$speed" --case energy101 --path default --baseline-root "$baseline" --candidate-root "$PWD" --accuracy full
```

Keep all times and spreads together with the actual route, shape and energy count. Compare any apparent regression with same-checkout A/A drift, then repeat with reversed or interleaved arm order. A small workload can favor an incumbent FFT. Identify the measured crossover per GPU, along with any large speed/memory tradeoff. A lower memory number obtained by shrinking the output does not constitute a matched comparison.

## Full correctness panels before a push

1. **Baseline and final candidate.** On each checkout, from its root, in the same configured local environment, run `python -c 'import NRSS; print(NRSS.__file__)'` followed by the report below. Retain the environment snapshot, CPU smoke, GPU smoke and full physics panels for `legacy_cyrsoxs`, `cupy_tensor_coeff` and `cupy_direct_polarization`. Use the report driver defaults for panels and plots. Compare per-test names, statuses and historical dispositions to the actual parent commit. The branch includes an FFT-grid follow-up after the historical ZP04 report.
2. **Force the candidate through every eligible public fixture.** Adapt [the film parity audit](../../../tests/validation/dev/cupy_rsoxs_virtual_zpad/film_parity_audit.py) into a test-only opt-in that reaches each eligible compact fixture, handling helpers that resolve their own computation path or start a subprocess. The current `--nrss-speed-variant compact_r2c` alone selects a basis, not XYZ padding. Keep default assertions on native output. Independently run and compare literal versus XYZ, and literal versus material-filled Z. Use modest rectangular extensions for large fixtures and bounded streams where required. For each test and route record executed shapes, method, material, tolerance, reference, result or pre-existing ineligibility. Require nonzero candidate evaluations of each claimed configuration. Also run tests aimed specifically at the new public controls, returning the padded result to their assertions.
3. **Science checks.** Cover the existing sphere, disk, scalar and material-Euler quadratic contrast panels, high-q HCP/BCC and off-axis registration, CoreShell and MWCNT. Use absolute intensity, full q coordinates, energy order, full finite/NaN masks and streamed panel comparisons. Physically changed padded sampling uses its own literal target and analytical/grid tests. Preserve versioned native goldens and their tolerances. For HCP/BCC compare assigned peak q on the registered physical axes, same energy, material and window. A finer grid can resolve previously unresolved peaks. Assess peak positions and intensity against an independent matching grid.
4. **Promotion evidence.** Require the complete seven-rung vacuum and numbered-material ZP08 production results, measured speed/memory and matched incumbent comparisons, the material/XYZ tests, and the default and forced-route panels on the same final executable revision. A shorter smoke/spot run is sufficient between tickets. A failed case, missed candidate route, incomplete 8192 output, missing measured resources or a new required skip leaves qualification open. Resolve and rerun affected diagnostics and the final full reports after executable changes. Record the local promotion decision in the new ledger. Add a user-facing example of each pad selector, a numbered oriented material, and the default vacuum case, with the XY window, unwindowed Z and full-detector sampling described.

```bash
env -u NRSS_TEST_BACKEND -u NRSS_BACKEND -u NRSS_TEST_PATH -u NRSS_PATH -u NRSS_TEST_SPEED_VARIANT \
    scripts/run_local_test_report.sh -e "$CONDA_DEFAULT_ENV" -r test-reports/virtual-3dpad/default
```

Before the final handoff follow [S07](../speed_tickets/S07-promotion.md) for comparing full output paths on the same revision. Record explicit dense controls that exceeded available resources with their attempted/estimated allocation and available VRAM. The mandatory *virtual* rows carry actual measurements.
