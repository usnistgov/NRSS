This directory is for development-only CoreShell backend performance studies.

Nothing here is part of the maintained pytest surface.

Current contents:

- `run_core_shell_backend_performance_abstract.py`
  - serial, subprocess-isolated comparison runner for:
    - `numpy -> cyrsoxs`,
    - `numpy -> cupy-rsoxs (coerce)`,
    - `cupy -> cupy-rsoxs (borrow/strict)`,
  - uses the scaled CoreShell ladder:
    - `small`: `(32, 512, 512)`,
    - `medium`: `(64, 1024, 1024)`,
    - `large`: `(96, 1536, 1536)`,
  - uses the full `101`-energy CoreShell baseline,
  - sweeps unique-state `EAngleRotation` settings:
    - `off`: `[0, 0, 0]`,
    - `30deg`: `[0, 30, 330]`,
    - `15deg`: `[0, 15, 345]`,
    - `5deg`: `[0, 5, 355]`,
  - writes timing summaries, compact A-wedge artifacts, and a graphical abstract,
  - supports a separate peak-memory pass on the densest angle setting so the
    headline timing pass stays minimally instrumented.

Recommended workflow:

1. Run the full serial comparison study:

```bash
/home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/core_shell_backend_performance/run_core_shell_backend_performance_abstract.py \
  --label initial
```

2. If interrupted, rerun the same command.
   Existing completed case results are reused automatically.

3. If only the plot styling changes, regenerate from saved results:

```bash
/home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/core_shell_backend_performance/run_core_shell_backend_performance_abstract.py \
  --label initial \
  --plot-only
```

4. If the sizes were run as separate labels, merge them into one combined report:

```bash
/home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/core_shell_backend_performance/run_core_shell_backend_performance_abstract.py \
  --label full_energy_all \
  --merge-labels full_energy_small,full_energy_medium,full_energy_large
```

Artifacts are written under:

- `test-reports/core-shell-backend-performance-dev/<label>/summary.json`
- `test-reports/core-shell-backend-performance-dev/<label>/performance_table.tsv`
- `test-reports/core-shell-backend-performance-dev/<label>/core_shell_backend_performance_graphical_abstract.png`
