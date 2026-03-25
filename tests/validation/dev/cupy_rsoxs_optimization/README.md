This directory is for development-only `cupy-rsoxs` optimization studies.

Nothing here is part of the maintained pytest validation surface.

For the principal cross-backend comparison panel, use:

- `tests/validation/dev/core_shell_backend_performance/run_primary_backend_speed_comparison.py`
  - that orchestrator consumes this harness for the `cupy-rsoxs` host/device
    comparison rows and writes the combined summary, TSV, and PNG table.

Current contents:

- `run_cupy_rsoxs_optimization_matrix.py`
  - reusable cupy-only timing driver for the staged `cupy-rsoxs`
    optimization campaign,
  - uses scaled CoreShell morphologies for timing trends across box size,
  - excludes upstream field generation from the primary metric,
  - defaults to the common public-workflow lane:
    - `resident_mode='host'`,
    - single energy,
    - `EAngleRotation=[0, 0, 0]`,
    - NumPy authoritative fields generated directly in contract shape/dtype,
  - supports an opt-in device-resident regression lane:
    - `resident_mode='device'`,
    - CuPy authoritative fields prepared before timing starts,
    - default stream synchronized before the timer starts so upstream GPU work
      is excluded,
  - keeps the no-rotation triple-energy lane available as an opt-in secondary
    checkpoint for either resident-mode variant,
  - keeps the limited-rotation triple-energy lane available as an opt-in
    secondary checkpoint for either resident-mode variant,
  - supports optional centered-energy no-rotation sweeps for arbitrary energy
    counts via `--no-rotation-energy-counts`,
  - supports arbitrary explicit rotation sweeps via `--rotation-specs`,
    - syntax is comma-separated `start:increment:end` triples,
    - each triple maps directly to
      `EAngleRotation=[StartAngle, IncrementAngle, EndAngle]`,
  - supports arbitrary explicit energy-list sweeps via `--energy-lists`,
    - quote the argument because each group uses `|` separators,
    - each comma-separated group becomes one explicit energy case,
  - if both `--rotation-specs` and `--energy-lists` are supplied, the harness
    also emits combined explicit-energy plus explicit-rotation cases,
  - execution-path sweep support is now live for the current Segment `B` / `D`
    campaign,
    - use `backend_options["execution_path"]` through `--execution-paths`,
    - supported execution-path values are:
      - `tensor_coeff` for the current accepted route,
      - `direct_polarization` for the CyRSoXS communication-minimizing analog,
      - `nt_polarization` for the CyRSoXS memory-minimizing analog,
  - defines the primary timing boundary as:
    - start immediately before `Morphology(...)`,
    - end immediately after synchronized `run(return_xarray=False)`,
  - supports internal segment timing for segments `A1`, `A2`, and `B-F`,
  - alias `A` expands to `A1,A2` for convenience,
  - supports isotropic-material representation selection for the CoreShell
    isotropic materials:
    - `legacy_zero_array` keeps concrete zero `S/theta/psi` fields for the
      isotropic core/matrix materials,
    - `enum_contract` uses the explicit `SFieldMode.ISOTROPIC` contract for
      those isotropic materials,
    - `both` emits paired cases so the same timing sweep compares both
      representations directly,
    - when `both` is selected, stdout and `summary.json` also emit paired
      `isotropic_representation_comparisons` entries that surface `primary`
      wall time side by side for legacy zero arrays versus the enum contract,
  - supports optional untimed CUDA prewarm for host-resident steady-state
    timing studies:
    - `off` preserves the current cold-subprocess behavior and remains the
      default,
    - `before_prepare_inputs` performs a tiny NumPy -> CuPy staging touch
      inside the worker before `_prepare_core_shell_case_inputs(...)` so
      first-touch CUDA/CuPy bring-up can be absorbed before the primary timer
      starts,
    - device-resident cases record this as redundant because that lane already
      touches CuPy before `primary_start`,
    - the prewarm mode does not change allocator/pool refresh behavior,
  - runs each benchmark case in a subprocess so crashes or OOMs are isolated to
    the individual case.
- `run_cupy_rsoxs_optimization_matrix_legacy_pre_isotropic_contract.py`
  - frozen pre-enum snapshot of the timing harness kept so the older CLI and
    case-construction behavior remain available unchanged if needed.

Recommended workflow:

Segment `A` is nominally complete for the common workflow.
- Default new speed work should focus on Segments `B` and `D`.
- `resident_mode='device'` is expected to be faster when morphology fields
  already live on GPU, and that distinction should remain visible in the
  timing results.
- Repeated-run host-resident staging reuse inside the backend remains only a
  low-priority niche future idea.
- If the goal is to model many morphologies inside one already-warm subprocess,
  use `--cuda-prewarm before_prepare_inputs` in this dev harness rather than
  changing backend residency or pool-refresh behavior.
- Current Segment `B` / `D` campaign focus:
  - establish execution-path baselines before changing math,
  - keep timing results execution-path-specific,
  - validate dormant paths with the maintained CoreShell helper before using
    them as optimization guidance,
  - treat future float16 work as orthogonal to execution path rather than as a
    replacement for it.
- Current Segment `D` continuation focus for the next pass:
  - use `execution_path='tensor_coeff'` as the maintained default target,
  - use device-resident timing only for the inner-loop ranking pass,
  - keep the aligned no-rotation device lane as a regression guard,
  - rank proposed `D` changes on at least one explicit general-angle device
    case such as `--rotation-specs 0:15:165`,
  - require a parity/correctness check before accepting any measured speed win,
  - update `CUPY_RSOXS_OPTIMIZATION_LEDGER.md` after each attempted step or any
    important intermediate result.

Current prioritized Segment `D` plan:

1. Cache detector / Ewald geometry and interpolation tables that are currently
   rebuilt inside `_q_axes(...)` and `_project_scatter3d(...)`.
   - current status:
     - implemented and retained as scaffolding,
     - standalone gain on the general-angle device lane was only modest/noisy,
       so do not count it as the first accepted March 25 Segment `D` win by
       itself.
2. For `tensor_coeff`, reduce repeated general-angle detector-projection work
   in `_projection_coefficients_from_fft_nt(...)`.
   - current status:
     - accepted and later extended to the aligned `x` / `y` families,
     - the general-angle `tensor_coeff` path now builds `proj_x`, `proj_y`,
       and `proj_xy` directly on the detector grid from the two FFT basis
       families instead of calling the generic 3D projector three times.
     - aligned `x` / `y` families now also use the detector-grid helper
       instead of the old `scatter3d` materialization route inside this
       wrapper,
     - current accepted end-state artifacts:
       - `test-reports/cupy-rsoxs-optimization-dev/planD02b_aligned_detector_projection_no_rotation/summary.json`
       - `test-reports/cupy-rsoxs-optimization-dev/planD02b_aligned_detector_projection_general/summary.json`
       - use the dedicated no-rotation artifact as the aligned regression
         authority because the explicit-rotation harness still emits its
         built-in no-rotation companion case.
3. Replace full `scatter3d` materialization with a direct detector-output path
   if the lower-risk caching/algebra steps are not enough.
   - current status:
     - rejected for this pass,
     - a raw-kernel prototype was much faster warm but much worse on the cold
       first call that defines the current subprocess timing surface.
4. Precompute and reuse static `D` algebra built from detector geometry.
   - current status:
     - rejected as non-material on the authority surface,
     - it trimmed the targeted general-angle `D` segment slightly but did not
       produce a meaningful primary-time win.
5. Test whether constant projection-family scaling can be hoisted out of the
   hot full-array path.
   - current status:
     - rejected,
     - the targeted general-angle primary time regressed.
6. Only then test reusable Segment `D` scratch buffers if the simplified math
   path still appears allocator-sensitive.
   - current status:
     - closed without implementation for this pass,
     - after the accepted `planD02` refactor, the remaining cold timing surface
       is no longer convincingly allocator-limited.

Latest execution-path baseline snapshot:

- Source artifact:
  - `test-reports/cupy-rsoxs-optimization-dev/execution_path_surface_smoke_20260324/summary.json`
- Host no-rotation single-energy small lane:
  - `tensor_coeff`: `primary 2.833s`, `B 0.139`, `D 0.072`, `E 0.112`
  - `direct_polarization`: `primary 2.609s`, `B 0.143`, `D 0.033`, `E 0.002`
  - `nt_polarization`: `primary 2.607s`, `B 0.130`, `D 0.034`, `E 0.002`
- Device no-rotation single-energy small lane:
  - `tensor_coeff`: `primary 0.237s`, `B 0.131`, `D 0.072`, `E 0.004`
  - `direct_polarization`: `primary 0.204s`, `B 0.138`, `D 0.033`, `E 0.002`
  - `nt_polarization`: `primary 0.205s`, `B 0.134`, `D 0.033`, `E 0.002`
- Quick correctness checkpoint on the official maintained CoreShell morphology
  path versus `tensor_coeff`:
  - `direct_polarization` and `nt_polarization` both showed
    `max_abs 0.078125`, `rmse 1.60801e-4`, `p95_abs 3.8147e-06` on the raw
    no-rotation single-energy scattering output,
  - heavier A-wedge validation should still be finished before treating the
    dormant paths as validated optimization guidance.

Latest accepted optimization step:

- `planD02b_aligned_detector_projection_extension`
  - after the accepted `planD02` general-angle fusion, the aligned
    `tensor_coeff` `x` / `y` families were moved onto the same detector-grid
    projection helper instead of materializing full `scatter3d` volumes,
  - benchmark artifacts:
    - `test-reports/cupy-rsoxs-optimization-dev/planD02b_aligned_detector_projection_no_rotation/summary.json`
    - `test-reports/cupy-rsoxs-optimization-dev/planD02b_aligned_detector_projection_general/summary.json`
  - maintained parity:
    - `pytest tests/validation/test_core_shell_reference.py -k "test_core_shell_sim_regression_cupy_borrow_strict" --nrss-backend cupy-rsoxs -v`
      passed,
    - `pytest tests/validation/test_core_shell_reference.py -k "test_core_shell_sim_regression_pybind" --nrss-backend cupy-rsoxs -v`
      passed,
  - notable deltas versus the initial March 25 device baseline:
    - aligned no-rotation device lane:
      `primary 0.219s -> 0.177s`, `D 0.035 -> 0.020`,
    - explicit general-angle device lane:
      `primary 0.248s -> 0.185s`, `D 0.074 -> 0.020`.

Current isotropic-contract note:

- the older exact-zero detection path from `plan06` has been retired,
- the explicit `enum_contract` path is now the only isotropic-material route
  that skips `S/theta/psi` staging in `A2` and skips the Euler/off-diagonal
  work in Segment `B`,
- named `vacuum` materials now always resolve to that explicit isotropic
  contract and ignore any supplied `S/theta/psi` fields with warning,
- `legacy_zero_array` remains in the harness as the historical comparison lane
  and does not receive inferred isotropic optimization,
- for explicit `enum_contract` cases, device-resident `A2` again represents
  runtime staging rather than a hidden `S` scan.

Current host-prewarm note:

- artifacts:
  - cold host comparison:
    `test-reports/cupy-rsoxs-optimization-dev/host_iso_cold_prewarm_compare_20260324/summary.json`
  - prewarmed host comparison:
    `test-reports/cupy-rsoxs-optimization-dev/host_iso_warm_prewarm_compare_20260324/summary.json`
  - device-resident redundancy smoke:
    `test-reports/cupy-rsoxs-optimization-dev/device_prewarm_redundant_smoke_20260324/summary.json`
- primary-lane totals on the small single-energy host / `tensor_coeff`
  CoreShell comparison:
  - cold / `legacy_zero_array`: `primary 2.546s`, `A2 2.363`
  - cold / `enum_contract`: `primary 2.545s`, `A2 2.369`
  - prewarmed / `legacy_zero_array`: `primary 0.282s`, `A2 0.0887`
  - prewarmed / `enum_contract`: `primary 0.226s`, `A2 0.0461`
- interpretation:
  - fresh host subprocesses still pay first-touch CUDA/CuPy bring-up inside
    `A2`, so cold-process totals are not the right evidence for steady-state
    isotropic staging gains,
  - on this rerun the cold host totals were effectively tied while `A2`
    remained startup-dominated, which is exactly why cold-process totals are a
    poor authority for the isotropic contract itself,
  - once that startup is prewarmed outside the primary boundary, the explicit
    enum contract improves host primary time by about `19.8%` and cuts `A2`
    by about `48.0%`,
  - device-resident cases already touch CuPy before timing starts, so the
    harness reports the prewarm mode as redundant there.

Latest accepted-state verification:

- final accepted-state benchmark artifacts:
  - `test-reports/cupy-rsoxs-optimization-dev/planD02b_aligned_detector_projection_no_rotation/summary.json`
  - `test-reports/cupy-rsoxs-optimization-dev/planD02b_aligned_detector_projection_general/summary.json`
- final accepted-state snapshot on the March 25 Segment `D` device authority
  lanes:
  - aligned no-rotation device / `tensor_coeff`:
    `primary 0.177s`, `D 0.020`
  - explicit general-angle device / `tensor_coeff`:
    `primary 0.185s`, `D 0.020`
- maintained reference close-out:
  - `pytest tests/validation/test_core_shell_reference.py -k "test_core_shell_sim_regression_cupy_borrow_strict" --nrss-backend cupy-rsoxs -v`
    passed,
  - `pytest tests/validation/test_core_shell_reference.py -k "test_core_shell_sim_regression_pybind" --nrss-backend cupy-rsoxs -v`
    passed.
- latest principal cross-backend speed snapshot:
  - artifact root:
    `test-reports/core-shell-backend-performance-dev/principal_cross_backend_20260325_planD02b/`
  - summary:
    - host speedup versus legacy `cyrsoxs` ranged from about `1.9x` to `4.5x`,
    - device speedup versus legacy pre-warm ranged from about `2.6x` to
      `13.3x`,
    - medium/large device rows now land around `9.2x-13.3x` faster than the
      matching legacy pre-warm rows.

Latest harness-extension verification:

- artifact:
  - `test-reports/cupy-rsoxs-optimization-dev/harness_rotation_energy_smoke_20260324/summary.json`
- focused verification scope:
  - `resident_modes=host,device`,
  - `execution_paths=tensor_coeff`,
  - `rotation_specs=[[0, 15, 165]]`,
  - `explicit_energy_lists=[[284.7, 285.0, 285.2]]`,
- confirmed emitted case families for both host and device:
  - baseline single-energy no-rotation,
  - explicit single-energy rotation,
  - explicit energy-list no-rotation,
  - explicit energy-list plus explicit rotation.

Recommended inner-loop command for the next Segment `D` device-only pass:

```bash
/home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py \
  --label d_focus \
  --size-labels small \
  --resident-modes device \
  --execution-paths tensor_coeff \
  --rotation-specs 0:15:165 \
  --timing-segments D
```

Recommended aligned-angle device regression command for the same pass:

```bash
/home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py \
  --label d_device_regression \
  --size-labels small \
  --resident-modes device \
  --execution-paths tensor_coeff \
  --timing-segments D
```

Initial device-only Segment `D` baseline for this continuation pass:

- artifact:
  - `test-reports/cupy-rsoxs-optimization-dev/planD00_device_baseline_20260325_193556/summary.json`
- measured before any new Segment `D` code changes:
  - `core_shell_small_single_no_rotation_device_tensor_coeff`:
    `primary 0.219s`, `D 0.035s`
  - `core_shell_small_single_rot_0_15_165_device_tensor_coeff`:
    `primary 0.248s`, `D 0.074s`
- use the explicit general-angle device case as the main ranking lane because
  the accepted aligned-angle work already removed much of the easy `D` cost
  from the `0°` path.

Latest Segment `D` continuation outcomes:

- `planD01_detector_geometry_cache`
  - authoritative artifact:
    `test-reports/cupy-rsoxs-optimization-dev/planD01_detector_geometry_cache_seq_20260325_194304/summary.json`
  - parity/smoke:
    - maintained device strict/borrow CoreShell regression passed,
    - maintained host-resident CoreShell sim-regression smoke also passed,
  - timing versus the initial device baseline:
    - no-rotation device lane:
      `primary 0.219s -> 0.190s`, `D 0.035 -> 0.035`,
    - explicit general-angle device lane:
      `primary 0.248s -> 0.244s`, `D 0.074 -> 0.070`,
  - interpretation:
    - keep it as low-risk detector-geometry scaffolding,
    - do not count it as a standalone material speed win.
- `planD02_tensor_coeff_general_angle_projection_fusion`
  - authoritative artifacts:
    - `test-reports/cupy-rsoxs-optimization-dev/planD02_tensor_coeff_general_angle_projection_fusion_fix/summary.json`,
    - `test-reports/cupy-rsoxs-optimization-dev/planD02b_aligned_detector_projection_no_rotation/summary.json`,
    - `test-reports/cupy-rsoxs-optimization-dev/planD02b_aligned_detector_projection_general/summary.json`,
    - use the dedicated no-rotation artifact as the aligned regression
      authority because the explicit-rotation harness still emits its
      companion no-rotation case,
  - superseded intermediates:
    - `test-reports/cupy-rsoxs-optimization-dev/planD02_tensor_coeff_general_angle_projection_fusion/summary.json`,
    - `test-reports/cupy-rsoxs-optimization-dev/planD02_tensor_coeff_general_angle_projection_fusion_rerun/summary.json`,
    - those two runs predated removal of an accidentally retained legacy
      `proj_x` projection call in the general-angle wrapper and should not be
      used as the final `planD02` authority,
  - current-tree regression repros that motivated the aligned-family
    follow-on:
    - `test-reports/cupy-rsoxs-optimization-dev/planD02_repro_current_no_rotation/summary.json`
    - `test-reports/cupy-rsoxs-optimization-dev/planD02_repro_current_general/summary.json`
    - those reruns showed the aligned device guard had drifted back to about
      `primary 0.243-0.256s`, `D 0.055-0.076` while the targeted general-angle
      lane stayed fast,
  - maintained parity:
    - `pytest tests/validation/test_core_shell_reference.py -k "test_core_shell_sim_regression_cupy_borrow_strict" --nrss-backend cupy-rsoxs -v`
      passed,
    - `pytest tests/validation/test_core_shell_reference.py -k "test_core_shell_sim_regression_pybind" --nrss-backend cupy-rsoxs -v`
      passed,
  - timing versus the initial device baseline:
    - aligned no-rotation device lane:
      `primary 0.219s -> 0.177s`, `D 0.035 -> 0.020`,
    - explicit general-angle device lane:
      `primary 0.248s -> 0.185s`, `D 0.074 -> 0.020`,
  - interpretation:
    - accept this as the first material March 25 Segment `D` speed win because
      the targeted general-angle device lane improved strongly,
    - the later aligned-family detector-grid follow-on removed the renewed
      no-rotation regression and now defines the final accepted March 25
      Segment `D` end state.
- `planD03_direct_detector_kernel`
  - interpretation:
    - reject for this pass,
    - a dedicated raw-kernel detector path reached about `0.10ms` warm for the
      helper itself but its first cold call was about `59ms`, which is worse
      than the accepted elementwise helper on the authority timing surface.
- `planD04_static_algebra_prefactor_cache`
  - authoritative artifact:
    `test-reports/cupy-rsoxs-optimization-dev/planD04_static_algebra_prefactor_cache/summary.json`
  - timing versus the accepted `planD02` state:
    - aligned no-rotation device lane:
      `primary 0.186s -> 0.196s`, `D 0.034 -> 0.035`,
    - explicit general-angle device lane:
      `primary 0.186s -> 0.188s`, `D 0.020 -> 0.019`,
  - interpretation:
    - reject as non-material.
- `planD05_tensor_coeff_scale_hoist`
  - authoritative artifact:
    `test-reports/cupy-rsoxs-optimization-dev/planD05_tensor_coeff_scale_hoist/summary.json`
  - timing versus the accepted `planD02` state:
    - aligned no-rotation device lane:
      `primary 0.186s -> 0.185s`, `D 0.034 -> 0.034`,
    - explicit general-angle device lane:
      `primary 0.186s -> 0.205s`, `D 0.020 -> 0.021`,
  - interpretation:
    - reject because the targeted primary metric regressed.
- `planD06_segment_d_scratch_reuse`
  - interpretation:
    - close without implementation for this pass,
    - the remaining Segment `D` surface after `planD02` is dominated more by
      first-call kernel setup than by obvious allocator churn.

Latest CUDA-prewarm verification:

- focused non-GPU smoke:
  - `pytest tests/smoke/test_smoke.py -k "cuda_prewarm or explicit_isotropic_contract or vacuum_named" -m "not gpu" -v`
  - `7 passed`
- verified behaviors:
  - parser and summary plumbing for `--cuda-prewarm`,
  - named `vacuum` forcing the explicit isotropic contract,
  - warning behavior when orientation is supplied to `vacuum` or other
    explicit isotropic materials.

Latest multi-angle execution-path comparison:

- artifact:
  - `test-reports/cupy-rsoxs-optimization-dev/execution_path_multiangle_5_vs_15_20260324/summary.json`
- measurement scope:
  - small CoreShell,
  - single energy `285.0`,
  - `resident_modes=host,device`,
  - `execution_paths=tensor_coeff,direct_polarization,nt_polarization`,
  - rotation sets equivalent to `--rotation-specs '0:15:165,0:5:165'`,
- result:
  - once multi-angle work matters, `tensor_coeff` is the clear winner,
  - host / `0:15:165`:
    - `tensor_coeff 2.742s`,
    - `nt_polarization 3.319s`,
    - `direct_polarization 3.742s`,
  - host / `0:5:165`:
    - `tensor_coeff 2.921s`,
    - `nt_polarization 4.376s`,
    - `direct_polarization 4.524s`,
  - device / `0:15:165`:
    - `tensor_coeff 0.419s`,
    - `nt_polarization 0.441s`,
    - `direct_polarization 0.556s`,
  - device / `0:5:165`:
    - `tensor_coeff 0.412s`,
    - `nt_polarization 0.926s`,
    - `direct_polarization 1.559s`,
- interpretation:
  - the dormant paths still matter as no-rotation references,
  - but for angle-heavy workloads the default `tensor_coeff` path is the one
    to optimize and benchmark first.

Rejected late experiments:

- `plan08_segment_b_algebraic_rewrite`
  - fixed-size scratch reuse did not blow up memory, but it regressed the
    default `tensor_coeff` Segment `B` path and was reverted
- `plan11_elementwise_kernel_experiment`
  - the aligned-angle `ElementwiseKernel` cut device `B` for the `Nt` subset,
    but it regressed the default host `tensor_coeff` `B` path badly enough that
    the added maintenance burden was not justified

1. Iterate on `cupy-rsoxs` with the primary no-rotation lane first:
   - for new work, start with Segments `B` or `D` unless Segment `A` is being
     revisited for a specific reason,
   - rerun the default host-resident single-energy small-box timing case after
     each optimization step,
   - narrow `--timing-segments` when focusing on one segment,
   - keep only optimizations that materially improve the primary timing metric
     or the targeted segment metric.
2. Recheck the opt-in device-resident lane periodically as a regression guard
   for direct CuPy workflows.
3. Use the limited-rotation triple-energy lane only when the built-in
   `EAngleRotation=[0, 15, 165]` checkpoint is sufficient.
4. Use `--rotation-specs` when the exact angle set is the point of the
   measurement, and `--energy-lists` when the exact energy list is the point of
   the measurement.
5. If both energy list and rotation set matter, pass both options together so
   the harness records the combined cases explicitly.
6. Return to the maintained test suite after the optimization block to confirm
   there was no physical drift.

Typical commands:

```bash
/home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py \
  --label baseline \
  --size-labels small
```

```bash
/home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py \
  --label isotropic-representation-compare \
  --size-labels small \
  --resident-modes host,device \
  --execution-paths tensor_coeff \
  --isotropic-material-representation both \
  --timing-segments A2,B
```

```bash
/home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py \
  --label isotropic-representation-compare-prewarmed \
  --size-labels small \
  --resident-modes host \
  --execution-paths tensor_coeff \
  --isotropic-material-representation both \
  --timing-segments A2,B \
  --cuda-prewarm before_prepare_inputs
```

```bash
/home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py \
  --label device-regression \
  --size-labels small \
  --resident-modes device
```

```bash
/home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py \
  --label multienergy-no-rotation \
  --size-labels small \
  --resident-modes host,device \
  --include-triple-no-rotation \
  --no-rotation-energy-counts 2,4,8 \
  --timing-segments B
```

```bash
/home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py \
  --label explicit-rotation \
  --size-labels small \
  --resident-modes host,device \
  --execution-paths tensor_coeff,direct_polarization,nt_polarization \
  --rotation-specs '0:15:165,0:5:165' \
  --timing-segments B,C,D,E
```

```bash
/home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py \
  --label explicit-energy-and-rotation \
  --size-labels small \
  --resident-modes host,device \
  --execution-paths tensor_coeff \
  --rotation-specs '0:15:165' \
  --energy-lists '284.7|285.0|285.2,284.9|285.0|285.1|285.2' \
  --timing-segments B,C,D,E
```

```bash
/home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py \
  --label rotation-check \
  --size-labels small \
  --resident-modes host,device \
  --include-triple-limited \
  --timing-segments B,C,D,E
```
