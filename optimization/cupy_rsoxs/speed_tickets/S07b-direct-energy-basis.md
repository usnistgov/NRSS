# S07b · Reuse Fourier transforms in direct polarization

**Question.** Can an energy-independent basis for direct polarization reduce peak device memory or returned scan time compared with the qualified tensor basis?

For one electric-field direction `e`, let `v = Vfrac`, `φ = vS`, and `u` be the local director. The spatial factors in direct polarization are `v`, `φ`, and the three components of `φ u (u·e)`. Apply the incumbent optical coefficients after the FFT and combine the complex amplitudes at the detector endpoints. The existing direct projection intensity equation, endpoint interpolation, rotation and output contract apply to each energy.

The experimental `direct_r2c` option transforms the five real fields for each anisotropic material and one field for each isotropic material. For a single oblique direction, the tensor basis uses seven fields per anisotropic material. For a single aligned direction both bases use five. The direct kernel combines the cached amplitudes and projects one panel in the same launch. Measure the retained cache and the actual peak under matched conditions.

## Screen

- Select `execution_path="direct_polarization"` with `energy_basis_mode="direct_r2c"` for the candidate. Compare with both per-energy direct polarization and tensor `compact_r2c` at the same angle, energies, resident mode and returned layout. Use the medium and large 20-energy cases first.
- Include an aligned direction and an oblique direction. Use `EAngleRotation` in `[StartAngle, IncrementAngle, EndAngle]` order. The launcher represents `--angle 37` as `[37, 0, 37]` degrees and accepts `--candidate-path` for a matched comparison across execution paths.
- Run separate primary timing, stage counts, 10 ms GPU memory observation and output comparisons. Include the generic tilted two-material fixture, the weak-contrast energy, nonmonotonic energy ordering, a single z plane and a window/mask case. Check the incumbent fallback for a single distinct energy and for multiple directions.

## Multiple directions

For an angle sweep, a direct projector can form `P = N e` from the cached tensor samples. That shares FFTs across angles and runs the direct projection for each angle. Compare it against tensor `compact_r2c`, which shares the projected coefficients across angles, if the fixed-direction screen identifies a relevant memory or timing advantage.

Record the decision and measurements in [RESULTS.md](RESULTS.md). A default change requires the full qualification in [S07](S07-promotion.md).
