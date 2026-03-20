# Validation Diagnostics Archive

This directory preserves one-off diagnostic scripts that were useful while rebuilding the analytical sphere validation into a robust pytest suite.

These files are intentionally **not** under `tests/` anymore:
- they were exploratory rather than stable regression tests,
- they often used large ad hoc geometries and custom plotting paths,
- several probe overlapping hypotheses and would be noisy to maintain as part of the normal suite.

The stable replacement that came out of this work is:
- `tests/validation/test_analytical_sphere_form_factor.py`

Use these archived scripts only when re-opening one of the specific investigation threads below.

## Archived Files

- `analytical_disk_form_factor_diagnostic.py`
  - Purpose: tested whether a 2D disk/circle could be a cheaper analytical comparison surrogate when the 3D sphere showed persistent high-q disagreement.
  - Historical question: whether smaller `PhysSize` in 2D would expose a discretization trend more clearly than the 3D sphere.

- `sphere_centering_diagnostic.py`
  - Purpose: compared half-voxel centering to exact voxel-centered placement.
  - Historical question: whether the sphere minima shift came from a single-voxel centering convention error.

- `sphere_diameter_fit_diagnostic.py`
  - Purpose: fit an effective analytical diameter to the simulated trace.
  - Historical question: whether the disagreement could be explained as a simple effective-radius mismatch.

- `sphere_fftigor_qmap_diagnostic.py`
  - Purpose: reproduced the FFTIgor-style detector q mapping in Python and compared it to the NRSS/PyHyper reduction path.
  - Historical question: whether the reported q axis, rather than the scattering itself, explained the extrema offset.

- `sphere_flat_detector_analytic_diagnostic.py`
  - Purpose: evaluated the analytical sphere form factor on the flat detector plane before azimuthal reduction.
  - Historical question: whether the main discrepancy came from comparing the simulation to a curved-detector or direct `I(q)` analytical expression.
  - Outcome: this line of inquiry directly motivated the finalized flat-detector analytical comparison in the stable test.

- `sphere_minima_scaling_diagnostic.py`
  - Purpose: measured how simulated and analytical extrema positions diverged across the q range.
  - Historical question: whether the offset behaved like a constant multiplicative q-scale error.

- `sphere_projection_diagnostics.py`
  - Purpose: compared different `Nz` values and interpolation settings in the 3D far-field projection path.
  - Historical question: whether the high-q disagreement was dominated by detector projection details instead of morphology discretization.

- `sphere_scatter_approach_diagnostic.py`
  - Purpose: compared alternate CyRSoXS scatter approaches on the same spherical morphology.
  - Historical question: whether engine-side scattering mode choices materially changed the sphere discrepancy.

## Guidance For Future Reuse

- Treat these as archived notebooks-in-code, not as authoritative tests.
- If one of these themes becomes relevant again, copy the relevant logic into a fresh diagnostic script rather than reviving the whole file uncritically.
- If a future backend rewrite changes the sphere behavior substantially, the most relevant starting points are:
  - `sphere_flat_detector_analytic_diagnostic.py`
  - `sphere_fftigor_qmap_diagnostic.py`
  - `sphere_centering_diagnostic.py`
