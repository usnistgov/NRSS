This directory is for development-only `cupy-rsoxs` optimization studies.

Nothing here is part of the maintained pytest validation surface.

Current contents:

- `run_cupy_rsoxs_optimization_matrix.py`
  - reusable cupy-only timing driver for the staged `cupy-rsoxs`
    optimization campaign,
  - uses scaled CoreShell morphologies for timing trends across box size,
  - excludes upstream field generation from the primary metric,
  - defines the primary timing boundary as:
    - start immediately before `Morphology(...)`,
    - end immediately after synchronized `run(return_xarray=False)`,
  - supports internal segment timing for segments `A-F`,
  - runs each benchmark case in a subprocess so crashes or OOMs are isolated to
    the individual case.

Recommended workflow:

1. Iterate on `cupy-rsoxs` with the primary no-rotation lane first:
   - rerun the default `cupy-rsoxs` timing matrix after each optimization step,
   - narrow `--timing-segments` when focusing on one segment,
   - keep only optimizations that materially improve the primary timing metric
     or the targeted segment metric.
2. Return to the maintained test suite after the optimization block to confirm
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
  --label opt-step-01 \
  --size-labels small \
  --timing-segments B,C,D
```
