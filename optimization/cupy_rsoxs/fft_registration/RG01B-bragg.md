# RG01B · Off-axis registration and high-q Bragg reflections

**Run between RG01 and RG02.** Install the packet before changing the production lookup so both revisions can be captured. The normal 3D Bragg suite gets one small, exact registration test. A separate paired measurement determines which weak reflections merit a permanent assertion.

## Permanent physics test

The supplied `bragg-tests.patch` places `test_bragg_off_axis_registration` directly inside `tests/validation/test_bragg_3d_lattice.py`. Both signs and both energy conditions run in one test per CuPy path, using `Morphology.run`. The first packet's standalone grating is migrated to this node. The forward-pixel check retains its original node. The report discovers the explicit `physics_validation` marker and harvests the path-specific plot from the existing Bragg plot directory. The historical CyRSoXS path retains its existing lattice coverage. Its non-applicability to this corrected-grid assertion is expected.

```bash
reg=optimization/cupy_rsoxs/fft_registration
grating=tests/validation/test_bragg_3d_lattice.py::test_bragg_off_axis_registration
python -m pytest --collect-only -q "$grating"
NRSS_WRITE_VALIDATION_PLOTS=1 python -m pytest -q "$grating" --nrss-path cupy_tensor_coeff
NRSS_WRITE_VALIDATION_PLOTS=1 python -m pytest -q "$grating" --nrss-path cupy_direct_polarization
```

Require bright-mode normalized intensities 1 and 0.5 at axial depths 1 and 1.5 bins, respectively, on each lateral sign. Require the empty conjugate column on this Ewald branch to stay dark. The half-bin value follows the current interpolation of endpoint intensities. The continuous finite transform has a different prediction between bins. Use the analytical expectations for promotion and the plotted legacy values to document migration.

## Paired measurement on existing fixtures

1. Run the full-size `simple_cubic_a30` and `hcp_a45` cases with their fixed optical contrast, 256 × 1024 × 1024 occupied voxels, 1 nm spacing and window off. Keep all existing lattice assertions. BCC is a separate fixture proposed for ZP03. Use the 2D square/hex cases as single-plane controls.
2. At 285 and 570 eV compare legacy and corrected within the same energy. Use the common band `0.95 ≤ q_perp ≤ 1.25 nm⁻¹` at both energies and `1.90 ≤ q_perp ≤ 2.40 nm⁻¹` at 570 eV. Report detector XY spots and radial shells separately. For 3D radial lines use `q_abs_detector` and NRSSIntegrator. The supplied patch threads the case's energy, physical shape and voxel size through the 3D remesh. For the diagnostic plots show the whole band. The existing 1.2 nm⁻¹ radial plotting limit hides part of it.
3. Capture both production CuPy paths on the old code, then again on the corrected code. Give each geometry and path its own output directory. Use the same CPU reference on both runs. Select `tag=legacy` on the original code and `tag=corrected` on the corrected code. These labels only name output folders. Record the checkout SHA, environment and path for each run.

```bash
reg=optimization/cupy_rsoxs/fft_registration
out="$PWD/test-reports/fft-registration/bragg"
ref="$reg/evidence/bragg"
for path in cupy_tensor_coeff cupy_direct_polarization; do
  python "$reg/bragg_high_q.py" public --repo . --path "$path" \
    --reference "$ref" --out "$out/$tag/$path"
done
```

If using separate worktrees, set `out` to the same absolute directory in both. After both captures, run the paired summary. The paired summary writes bragg_summary.csv and bragg_matched_spots.csv with contributor indices, detection flags, intensities and measured positions. The public command saves native raw panels, original finite masks, radial curves and indexed peak tables. Inspect the full images for strong unindexed maxima as in the existing test. Compare raw masks and count valid candidate windows before interpreting a changed detection.

```bash
for path in cupy_tensor_coeff cupy_direct_polarization; do
  python "$reg/report_bragg_high_q.py" "$ref" \
    --legacy "$out/legacy/$path" --corrected "$out/corrected/$path" \
    --out "$out/paired/$path"
done
```

4. Use the common candidate list from the reciprocal lattice, signed excitation error, form factor and basis structure factor. The broad original `3.5 Δqz` window can place multiple l orders at the same detector XY. Store all contributing indices and count one observed spot. Tabulate reference-confirmed gains/losses and above-floor calls absent in the reference. Compare signed positions, peak and local background, and measured radial profiles for any order proposed for promotion. One observed maximum can represent several projected indices or one combined radial shell.
5. The CPU survey uses a local maximum within three pixels, the existing peak/background cut (15 for HCP, 100 for simple cubic) and a common absolute floor relative to the direct (1,0,0) peak. It reports `10⁻³`, `10⁻⁴` and `10⁻⁵`. Treat those sensitivity results as diagnostics. Promote individually identified spots only after direct finite-voxel Fourier evidence, stable public results on both corrected paths, adequate separation and common support. Include at least one weak/missed case as a control when reviewing an expanded peak set. A higher total count alone gives no acceptance criterion.

**Before push.** RG02's full, unfiltered three-path report must cover the new grating on both CuPy paths, the unchanged full 2D/3D lattice tests, and all existing physics. Archive the paired high-q comparison and the graphical abstracts. Carry the resulting unpadded data into ZP03 for a matched padding comparison. The CPU survey and its regeneration commands are in [BRAGG_RESULTS.md](BRAGG_RESULTS.md).
