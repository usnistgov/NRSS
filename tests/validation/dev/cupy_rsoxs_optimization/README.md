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

- `plan07_axis_family_fast_path`
  - fully axis-aligned angle sets now use explicit x-family / y-family
    specialization for `0°/180°` and `90°/270°`,
  - benchmark artifact:
    - `test-reports/cupy-rsoxs-optimization-dev/plan07_axis_family_fast_path_clean_20260324/summary.json`
  - notable deltas versus the post-plan06 baseline:
    - device / `tensor_coeff`: `primary 0.231s -> 0.196s`,
      `D 0.072 -> 0.035`, `E 0.004 -> 0.002`,
    - device / `direct_polarization`: `primary 0.210s -> 0.198s`,
      `D 0.051 -> 0.033`, `E 0.002 -> 0.000`,
    - host / `tensor_coeff`: `D 0.073 -> 0.035`, `E 0.004 -> 0.002`
      even though host `A2` noise still makes the wall-clock delta less
      stable than the backend-segment deltas,
  - isolated `90°` spot checks showed the same low-`D` / near-zero-`E` shape
    for the y-family branch.

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

- final accepted-state benchmark artifact:
  - `test-reports/cupy-rsoxs-optimization-dev/plan09_final_rebenchmark_accepted_state_20260324/summary.json`
- final accepted-state snapshot on the primary small single-energy no-rotation
  lane:
  - host / `tensor_coeff`: `primary 2.515s`, `A2 2.339`, `B 0.109`,
    `D 0.035`, `E 0.002`
  - device / `tensor_coeff`: `primary 0.205s`, `A2 0.122`, `B 0.010`,
    `D 0.039`, `E 0.002`
- focused smoke close-out:
  - `9 passed` across execution-path option handling, isotropic staging,
    aligned-angle endpoint behavior, and private segment timing checks
- maintained reference close-out:
  - `pytest tests/validation/test_core_shell_reference.py -k "sim_regression_cupy_borrow_strict" --nrss-backend cupy-rsoxs -v`
    passed for the accepted default path
- dormant-path correctness note:
  - official maintained-morphology raw scattering comparisons versus
    `tensor_coeff` remained close after the accepted plan06/plan07 changes:
    `max_abs 0.046875`, `rmse 9.72605e-05`, `p95_abs 3.8147e-06`,
  - full dormant-path A-wedge validation was attempted but remained too
    expensive for this inner-loop campaign, so treat dormant-path timing
    results as timing evidence plus raw-scattering similarity rather than as
    fully closed reference validation.

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
