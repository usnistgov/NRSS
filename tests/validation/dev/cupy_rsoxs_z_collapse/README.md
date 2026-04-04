`cupy-rsoxs` `z_collapse_mode` sphere comparison

This dev-only lane evaluates the exploratory `z_collapse_mode="mean"` fast path
against the maintained full `3D` `tensor_coeff` path on the analytical sphere
form-factor geometry.

Run from the repo root with the NRSS development environment:

```bash
mamba run -n nrss-dev python tests/validation/dev/cupy_rsoxs_z_collapse/run_sphere_z_collapse_comparison.py
```

Default outputs land under:

`test-reports/cupy-rsoxs-z-collapse-sphere`

Per-diameter artifacts include:

- `*_comparison.png`: graphical comparison of normalized `I(q)` curves and log residuals
- `*_summary.json`: pointwise/minima metrics and runtime summaries

The default run covers the two diameters called out in the proposal:

- `70 nm`
- `128 nm`
