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

- `run_comprehensive_backend_comparison.py`
  - dev-only small-CoreShell comprehensive comparison tier,
  - keeps the maintained default benchmark unchanged,
  - supports an opt-in z-collapse extension via `--include-z-collapse`,
  - runs separate speed and memory passes over:
    - host `warm`: `cyrsoxs`, `cupy-rsoxs tensor_coeff`, `cupy-rsoxs direct_polarization`,
    - host `hot`: the same three host paths with one untimed identical warm-up run inside each worker,
    - device `steady`: `cupy-rsoxs tensor_coeff`, `cupy-rsoxs direct_polarization`,
    - device `hot`: the same two device paths with one untimed identical warm-up run inside each worker,
  - with `--include-z-collapse`, adds:
    - host `warm` / `hot`: `cupy-rsoxs tensor_coeff` with `z_collapse_mode="mean"`,
    - device `steady` / `hot`: the same collapsed `tensor_coeff` path,
  - uses single energy only,
  - accepts `--size-label` to choose the CoreShell ladder entry, for example
    `small`, `medium`, or `large`,
  - uses the two requested rotation schemes only:
    - `no rotation`: `[0, 0, 0]`,
    - `0:5:165`: `[0, 5, 165]`,
  - records speed from the backend-specific maintained timing boundaries,
  - records memory in a separate pass with:
    - a warmed same-GPU CuPy observer subprocess using `cupy.cuda.runtime.memGetInfo()`,
    - baseline-subtracted peak GPU memory for the worker lifetime,
    - and process RSS polling from the parent orchestrator,
  - writes a combined summary plus separate speed and memory TSVs,
  - includes a speedup column on each `cupy-rsoxs` row against the comparable
    legacy `cyrsoxs` run,
  - tags collapsed rows separately in the TSV and Markdown report with
    `z collapse = mean` so they do not alias plain `tensor_coeff`,
  - treats device `steady` as comparable to legacy `warm`, and device `hot` as
    comparable to legacy `hot`,
  - writes a simple human-readable Markdown report table in the same
    `test-reports` run directory.

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

3. If you want the small-CoreShell comprehensive table with the opt-in
   `tensor_coeff` z-collapse lane included:

```bash
mamba run -n nrss-dev python \
  tests/validation/dev/core_shell_backend_performance/run_comprehensive_backend_comparison.py \
  --label comprehensive_z_collapse \
  --include-z-collapse
```

Targeted tensor-coeff memory rechecks:

- `run_tensor_coeff_inplace_segment_c_recheck.py`
  - dev-only recheck for the direct-path-inspired true in-place `Segment C`
    tensor analogue,
  - current accepted disposition: rejected,
  - the measured host-hot tensor lanes showed no peak-memory reduction and a
    small `Segment C` slowdown.
- `run_tensor_coeff_fused_isotropic_recheck.py`
  - dev-only recheck for the accepted `mem09` tensor-coeff fused-isotropic
    `Segment B` change,
  - current accepted disposition: accepted,
  - accepted artifact root:
    `test-reports/core-shell-backend-performance-dev/tc_mem09_fused_isotropic_20260407/`.
- `run_tensor_coeff_legacy_zero_recheck.py`
  - dev-only recheck for the accepted `mem10` host-resident tensor-coeff
    `legacy_zero_array` compatibility shortcut,
  - current accepted disposition: accepted,
  - accepted artifact root:
    `test-reports/core-shell-backend-performance-dev/tc_mem10_legacy_zero_20260407/`.

Artifacts are written under:

- `test-reports/core-shell-backend-performance-dev/<label>/primary_backend_speed_comparison_summary.json`
- `test-reports/core-shell-backend-performance-dev/<label>/primary_backend_speed_comparison_table.tsv`
- `test-reports/core-shell-backend-performance-dev/<label>/primary_backend_speed_comparison_table.png`

Latest snapshot tied to the accepted March 25 Segment `D` end state:

- artifact root:
  - `test-reports/core-shell-backend-performance-dev/principal_cross_backend_20260325_planD02b/`
- headline results from the combined table:
  - host speedup versus legacy `cyrsoxs`: about `1.9x` to `4.5x`,
  - device speedup versus legacy pre-warm: about `2.6x` to `13.3x`,
  - medium/large device rows: about `9.2x` to `13.3x` faster than the
    matching legacy pre-warm rows.

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

Mixed-precision sim-regression comparison:

- `run_core_shell_mixed_precision_abstract.py`
  - dev-only CoreShell sim-regression comparison focused on the maintained
    vendored sim golden at `tests/validation/data/core_shell/CS_sim_reference.h5`,
  - runs four subprocess-isolated cases:
    - `tensor_coeff` default,
    - `tensor_coeff` mixed precision,
    - `direct_polarization` default,
    - `direct_polarization` mixed precision,
  - keeps the maintained CoreShell morphology, full energy panel, and A-wedge
    reduction path from `tests/validation/lib/core_shell.py`,
  - writes cached A-wedge artifacts, a JSON summary, a TSV table, and one
    graphical abstract per execution path with:
    - sim golden overlays,
    - default-path overlays,
    - mixed-path overlays,
    - residuals and timing/metric summaries,
  - supports `--plot-only` to restyle from a saved summary without rerunning
    the simulations.

Recommended workflow:

```bash
mamba run -n nrss-dev python \
  tests/validation/dev/core_shell_backend_performance/run_core_shell_mixed_precision_abstract.py \
  --label mixed_precision_core_shell
```

Z-collapse sim-regression comparison:

- `run_core_shell_z_collapse_abstract.py`
  - dev-only CoreShell sim-regression comparison focused on the maintained
    vendored sim golden at `tests/validation/data/core_shell/CS_sim_reference.h5`,
  - runs one subprocess-isolated `cupy-rsoxs` case:
    - `tensor_coeff` with `z_collapse_mode="mean"`,
  - keeps the maintained CoreShell morphology, full energy panel, and A-wedge
    reduction path from `tests/validation/lib/core_shell.py`,
  - writes a cached A-wedge artifact, a JSON summary, and a graphical abstract
    with:
    - sim golden overlays,
    - collapsed-path overlays,
    - residuals and timing/metric summaries,
  - supports `--plot-only` to restyle from a saved summary without rerunning
    the simulation.
  - this dev harness directly informed the relaxed maintained CoreShell
    collapse regression lane now recorded in
    `tests/validation/test_core_shell_reference.py`,
  - and its role is now primarily figure generation / threshold inspection
    rather than proving implementation existence.

Recommended workflow:

```bash
mamba run -n nrss-dev python \
  tests/validation/dev/core_shell_backend_performance/run_core_shell_z_collapse_abstract.py \
  --label z_collapse_core_shell
```

Current status note:

- the effective-`2D` collapse implementation is now basically complete from a
  maintained-validation standpoint,
- both maintained cupy execution paths have relaxed CoreShell collapse
  sim-regression coverage,
- the main remaining backend-specific work is the separate effective-`2D`
  detector cleanup / simplification thread already tracked in the backend
  spec and proposal docs.
