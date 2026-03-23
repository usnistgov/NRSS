This directory is for development-only `cupy-rsoxs` optimization studies.

Nothing here is part of the maintained pytest validation surface.

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
  - keeps the limited-rotation triple-energy lane available as an opt-in
    secondary checkpoint for either resident-mode variant,
  - defines the primary timing boundary as:
    - start immediately before `Morphology(...)`,
    - end immediately after synchronized `run(return_xarray=False)`,
  - supports internal segment timing for segments `A1`, `A2`, and `B-F`,
  - alias `A` expands to `A1,A2` for convenience,
  - runs each benchmark case in a subprocess so crashes or OOMs are isolated to
    the individual case.

Recommended workflow:

Segment `A` is nominally complete for the common workflow.
- Default new speed work should focus on Segments `B` and `D`.
- `resident_mode='device'` is expected to be faster when morphology fields
  already live on GPU, and that distinction should remain visible in the
  timing results.
- Repeated-run host-resident staging reuse remains only a low-priority niche
  future idea.

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
3. Use the limited-rotation triple-energy lane only when rotation-sensitive
   changes need confirmation.
4. Return to the maintained test suite after the optimization block to confirm
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
  --label device-regression \
  --size-labels small \
  --resident-modes device
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
