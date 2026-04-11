# `cupy-rsoxs` Benchmarking Guide

Open this file only when you are ranking, accepting, or rejecting an
optimization candidate.

If you only need the current state, use `accepted_state.md` instead.

## Primary Rule

- Use the compact harnesses and maintained authority lanes.
- Do not use the archived long-form experiment notes as the primary benchmark
  authority.

## Primary Timing Harness

- Main timing harness:
  - `tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py`
- Maintained `cupy-rsoxs` authority lanes should pass:
  - `--isotropic-material-representation enum_contract`
- Treat `legacy_zero_array` as a compatibility / recheck surface, not the
  default maintained authority surface.
- Segment timing surface:
  - `A1`, `A2`, `B`, `C`, `D`, `E`, `F`
- Companion cross-backend comparison entry point:
  - `tests/validation/dev/core_shell_backend_performance/run_primary_backend_speed_comparison.py`

## `tensor_coeff` Authority Surface

- Speed ranking:
  - small CoreShell
  - explicit rotation sets, especially `0:15:165` and `0:5:165`
  - device-resident first for quick ranking
  - `isotropic_representation=enum_contract`
- Required host steady-state follow-up:
  - host resident
  - `--cuda-prewarm before_prepare_inputs`
  - `--isotropic-material-representation enum_contract`
- Required physics gates before acceptance:
  - `pytest tests/validation/test_core_shell_reference.py -k "test_core_shell_sim_regression_pybind" --nrss-backend cupy-rsoxs -v`
  - `pytest tests/validation/test_core_shell_reference.py -k "test_core_shell_sim_regression_cupy_borrow_strict" --nrss-backend cupy-rsoxs -v`

## `direct_polarization` Authority Surface

- Maintained acceptance lane:
  - small CoreShell
  - `resident_mode='device'`
  - single energy
  - `EAngleRotation=[0, 5, 165]`
  - `isotropic_representation=enum_contract`
  - `--worker-warmup-runs 1`
- Required companion checks:
  - device-hot no-rotation companion
  - external whole-worker peak GPU memory on the same `0:5:165` lane
- Use host-resident follow-up when the claim is about shared-GPU memory or
  compatibility-lane staging, not just device-hot throughput.

## Memory Claim Rules

- For cross-backend GPU-memory comparisons against legacy `cyrsoxs`, use the
  host-resident `cupy-rsoxs` lane as the comparison authority.
- Do not confuse CuPy allocator-retained free memory with still-live working
  set.
- If a candidate wins on speed but materially loses on the relevant memory
  authority surface, do not accept it as a maintained default-path change.

## What Not To Use As Primary Authority

- The archived full-energy comparison harness
- Old mixed workflow metrics
- Archive prose that predates the current resident-mode and acceptance-lane
  conventions

## If More Detail Is Needed

- Historical harness detail:
  - `archive/TESTS_VALIDATION_DEV_CUPY_RSOXS_OPTIMIZATION_README.md`
- Current accepted state:
  - `accepted_state.md`
