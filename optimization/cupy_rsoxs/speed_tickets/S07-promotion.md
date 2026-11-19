# S07 · Complete validation on the revision intended for a push

Quick checks from S00–S06 authorize experiments. Before any optimization is promoted for a push, run the complete maintained smoke and physics panels on the combined candidate and compare the **full numerical outputs** of the workflows used to sell the change.

**1. Run through the actual user entry point.** Commit or otherwise identify the precise local revision and resolve the import of NRSS to this checkout. Record the GPU, CuPy/CUDA, result contracts and selectors. Exercise the proposed production dispatch, plus every new selectable route on its supported inputs. Show route counters for basis build, FFT method, energy tile and fallback. Some maintained tests explicitly select execution paths, obtain options again by path ID inside helpers, or cache references by path. Account for those routes when forcing a candidate. A clean default report with zero candidate calls cannot validate an opt-in change.

**2. Run the complete repository report.** Use a local conda/mamba environment with this revision, CuPy, the legacy toolchain and the vendored references. From the repo root, in the selected environment:

```bash
python -c 'import NRSS; print(NRSS.__file__)'
env -u NRSS_TEST_BACKEND -u NRSS_BACKEND -u NRSS_TEST_PATH -u NRSS_PATH \
    scripts/run_local_test_report.sh -e "$CONDA_DEFAULT_ENV"
```

The [report driver](../../../scripts/run_local_test_report.sh) runs the environment snapshot, CPU smoke, GPU smoke and full physics for `legacy_cyrsoxs`, `cupy_tensor_coeff` and `cupy_direct_polarization`. Keep the default tests and panel/plot harvesting enabled. Inspect `summary.md`, `steps.tsv`, per-path case statuses and the graphical-abstract output. Compare expected path-specific applicability to the baseline. An unavailable GPU/reference/toolchain, a failed test, a newly skipped required case, or an entirely skipped path leaves promotion pending. Use the active catalog, which includes parameterized tests. An old test-count number is insufficient.

For a candidate selected only by an option, run the **entire applicable** CuPy smoke and physics suites again with that option routed through the test harness. S00/S07 should add a test-only selector (e.g. `--nrss-speed-variant`) when needed, propagate it to relevant builders/subprocesses, and keep API-default contract checks on their declared defaults. For example, once that selector exists:

```bash
python -m pytest tests/smoke tests/validation -m gpu --nrss-path cupy_tensor_coeff \
    --nrss-speed-variant "$VARIANT" -v
```

Repeat for `cupy_direct_polarization` when affected. Record which workloads deliberately exercise fallback. Add a representative high-energy-count test that enters the new route if maintained fixtures cannot. Keep existing tolerances and references. In particular, run the orientational and scalar contrast scaling, sphere/disk form factors, Bragg, CoreShell and MWCNT validations with their existing selections.

**3. Compare output panels, outside all timing.** For the small 101-energy case compare every detector pixel at every energy against the matching incumbent, streaming one panel at a time where needed. Compare dimensions, energy order, coordinates, finite/NaN masks, absolute intensity and predeclared normalized error limits, and summarize the worst energy. Include both a single direction and the routine rotated result in representative comparisons. Check a medium guard case, the generic anisotropic spectrum fixture, one enabled mask/window and z=1 if affected. For changed reduction or broadcasting compare the full reported q arrays, I/A or sector channels and NaN masks on the same support, including intersection/union behavior if touched. Check tails, DC and borders. Preserve coherence and numerical scale; document genuine FP32 summation-order differences.

**4. Record the promotion decision.** Require the full report, candidate-enabled evidence, full output comparisons and final paired speed/memory table from S06 for the same executable revision. Fix failures and repeat the affected checks plus the complete report on the final promotion revision if the executable changes. Put report paths and the selected defaults/options in [RESULTS.md](RESULTS.md); keep generated reports outside version control. Mark `push-ready` only after all required evidence passes. Record rejected/conditional routes and the accepted current-workflow baseline before opening variable incident direction work.
