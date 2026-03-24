This directory is for development-only `cyrsoxs` timing studies.

Nothing here is part of the maintained pytest validation surface.

This is a sibling harness to `tests/validation/dev/cupy_rsoxs_optimization/`.
It exists for legacy backend speed studies and should remain separate from the
default `cupy-rsoxs` optimization loop.

For the principal cross-backend comparison panel, use:

- `tests/validation/dev/core_shell_backend_performance/run_primary_backend_speed_comparison.py`
  - that orchestrator consumes this harness as the legacy backend component and
    writes the combined summary, TSV, and PNG comparison table.

Current contents:

- `run_cyrsoxs_timing_matrix.py`
  - reusable subprocess timing driver for the legacy `cyrsoxs` pybind backend,
  - uses scaled CoreShell morphologies for timing trends across box size,
  - excludes upstream field generation from the primary metric,
  - forces the host-style contract used for comparison against the default
    `cupy-rsoxs` host lane:
    - `backend='cyrsoxs'`,
    - `resident_mode='host'`,
    - NumPy authoritative morphology fields,
    - `input_policy='strict'`,
    - `ownership_policy='borrow'`,
    - `create_cy_object=True`,
  - defines the primary timing boundary as:
    - start immediately before `Morphology(...)`,
    - end immediately after `run(return_xarray=False)` returns, with results
      already manifested on the `Morphology` object,
  - supports the same CoreShell energy and rotation case-matrix surface as the
    current `cupy-rsoxs` dev harness,
  - keeps the isotropic-material representation switch:
    - `legacy_zero_array`,
    - `enum_contract`,
    - `both`,
  - keeps the optional `--cuda-prewarm before_prepare_inputs` switch as a
    best-effort legacy import/launch warmup outside the primary boundary,
  - suppresses `CyRSoXS` banner and status spam inside worker subprocesses,
  - runs each benchmark case in a subprocess so crashes or GPU failures are
    isolated to the individual case.

Recommended workflow:

1. Run the default small single-energy no-rotation lane:

```bash
/home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/cyrsoxs_timing/run_cyrsoxs_timing_matrix.py \
  --label initial
```

2. If you want to probe steady-state startup sensitivity, rerun with the
   best-effort warmup switch:

```bash
/home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/cyrsoxs_timing/run_cyrsoxs_timing_matrix.py \
  --label prewarmed \
  --cuda-prewarm before_prepare_inputs
```

Artifacts are written under:

- `test-reports/cyrsoxs-timing-dev/<label>/summary.json`
