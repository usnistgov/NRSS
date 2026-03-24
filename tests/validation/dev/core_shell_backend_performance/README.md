This directory is for development-only CoreShell backend comparison studies.

It is not the authoritative optimization timing harness for `cupy-rsoxs`.
Use `tests/validation/dev/cupy_rsoxs_optimization/` for current optimization timing work.

Nothing here is part of the maintained pytest surface.

Principal cross-backend comparison:

- `run_primary_backend_speed_comparison.py`
  - this is the principal cross-backend comparison entry point for backend dev work,
  - runs the fixed single-energy primary-time panel across:
    - legacy `cyrsoxs` cold,
    - legacy `cyrsoxs` pre-warm,
    - `cupy-rsoxs` host cold,
    - `cupy-rsoxs` host pre-warm,
    - `cupy-rsoxs` device steady-state,
  - uses the scaled CoreShell ladder:
    - `small`: `(32, 512, 512)`,
    - `medium`: `(64, 1024, 1024)`,
    - `large`: `(96, 1536, 1536)`,
  - uses single-energy cases only,
  - uses the two comparison rotations only:
    - `no rotation`: `[0, 0, 0]`,
    - `some rotation`: `[0, 15, 165]`,
  - reports a combined table with:
    - legacy primary time,
    - `cupy-rsoxs` host primary time,
    - host speedup versus the matching legacy startup state,
    - `cupy-rsoxs` device primary time on pre-warm rows,
    - device speedup versus the matching legacy pre-warm row,
  - writes a combined summary, TSV, and PNG table,
  - reuses the component timing summaries automatically if they already exist for the selected label,
  - supports `--plot-only` to regenerate the TSV and PNG from the combined summary.

Recommended workflow:

1. Run the principal cross-backend comparison panel:

```bash
/home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/core_shell_backend_performance/run_primary_backend_speed_comparison.py \
  --label principal_cross_backend
```

2. If only the table styling changes, regenerate from the saved combined summary:

```bash
/home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/core_shell_backend_performance/run_primary_backend_speed_comparison.py \
  --label principal_cross_backend \
  --plot-only
```

Artifacts are written under:

- `test-reports/core-shell-backend-performance-dev/<label>/primary_backend_speed_comparison_summary.json`
- `test-reports/core-shell-backend-performance-dev/<label>/primary_backend_speed_comparison_table.tsv`
- `test-reports/core-shell-backend-performance-dev/<label>/primary_backend_speed_comparison_table.png`

Historical full-energy comparison:

- `run_core_shell_backend_performance_abstract.py`
  - older serial, subprocess-isolated comparison runner for:
    - `numpy -> cyrsoxs`,
    - `numpy -> cupy-rsoxs (coerce)`,
    - `cupy -> cupy-rsoxs (borrow/strict)`,
  - uses the full `101`-energy CoreShell baseline,
  - writes timing summaries, compact A-wedge artifacts, and a graphical abstract,
  - remains useful as historical context and as a deeper full-energy / A-wedge study,
    but it is no longer the principal cross-backend comparison path.
