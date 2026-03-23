This directory is for development-only `cupy-rsoxs` optimization studies.

Nothing here is part of the maintained pytest validation surface.

Current contents:

- `run_cupy_rsoxs_optimization_matrix.py`
  - reusable benchmark/validation driver for the staged `cupy-rsoxs`
    optimization campaign,
  - uses scaled CoreShell morphologies for timing trends across box size,
  - uses small sphere-in-vacuum morphologies for fast polarization-sensitive
    ad hoc validation during optimization work,
  - runs each benchmark case in a subprocess so `cyrsoxs` crashes or OOMs are
    isolated to the individual case.

Recommended workflow:

1. First capture trusted references:
   - generate the ad hoc validation baselines from `cyrsoxs`,
   - capture one `cyrsoxs` timing run per CoreShell size.
2. Iterate on `cupy-rsoxs`:
   - rerun the default `cupy-rsoxs` timing matrix after each optimization step,
   - rerun the ad hoc validation cases against the saved `cyrsoxs` baselines,
   - keep only optimizations that materially improve runtime or unlock larger
     cases without introducing validation drift.
3. Return to the maintained CoreShell pytest suite after the optimization block
   to confirm there was no physical drift in the official parity lane.

Typical commands:

```bash
/home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py \
  --label baseline \
  --refresh-validation-baselines \
  --include-cyrsoxs-timing
```

```bash
/home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py \
  --label opt-step-01 \
  --baseline-dir test-reports/cupy-rsoxs-optimization-dev/baseline/baselines
```
