# High-q Bragg evidence on the unpadded fixtures

The maintained 3D suite uses `simple_cubic_a30` and `hcp_a45` at 285 eV. It applies a 1% floor to the calculated sphere-form-factor intensity times the basis structure factor squared, relative to the strongest geometrically accepted reflection. In the full 285 eV datasets this and the axial tolerance select the asserted spots. In the common high-q band the predictor returns **32** simple-cubic orders and **156** HCP orders at its permissive `10⁻¹²` setting. The assertion list includes zero orders in that band for either fixture. The HCP orders project onto 90 distinct transverse locations. The 3D unexpected-peak check consults the broad predicted list, and the radial evidence assertion tests only the selected shells. The existing pipeline can pass when individual higher-order spots change visibility.

## Measured on the CPU fixture replay

Both grids use the exact same rasterized sphere lattice, energies, isotropic contrast, X polarization and endpoint-intensity interpolation. The direct calculation uses the voxelized field's 2D Fourier coefficients at the reported XY bins and sums the complex amplitudes along z at the continuous physical Ewald depth, then projects and squares. The selected high-q columns are away from DC. This is a finite-specimen reference for these scalar, normal-incidence fixtures.

The search counts a local detector maximum within three pixels and applies the suite's per-case contrast criterion against the median of its 4–9 pixel annulus. A fixed absolute floor of `10⁻⁴` times the direct (1,0,0) peak applies to every method for that fixture and energy. The dense target regions are fully within the valid mask in these runs. A projected spot is counted once. A candidate measured in the same window by the reference and by the test calculation is a matched detection. The paired spot table gives the observed coordinates. At the stated floor the largest separation between a reference maximum and a matched maximum in any of the six fixture/band comparisons is √5 transverse bins.

| Fixture | Energy | q_perp band (nm⁻¹) | Reference above floor | Matched legacy → corrected | Confirmed gains / losses | Above-floor calls absent in reference, legacy → corrected |
| --- | ---: | --- | ---: | --- | --- | --- |
| HCP | 285 eV | 0.95–1.25 | 70 | 58 → 66 | 10 / 2 | 3 → 2 |
| Simple cubic | 285 eV | 0.95–1.25 | 24 | 4 → 8 | 4 / 0 | 0 → 0 |
| HCP | 570 eV | 0.95–1.25 | 86 | 76 → 76 | 0 / 0 | 0 → 0 |
| Simple cubic | 570 eV | 0.95–1.25 | 28 | 28 → 28 | 0 / 0 | 4 → 8 |
| HCP | 570 eV | 1.90–2.40 | 58 | 41 → 54 | 13 / 0 | 22 → 8 |
| Simple cubic | 570 eV | 1.90–2.40 | 20 | 2 → 12 | 10 / 0 | 1 → 0 |

The HCP result at 285 eV is threshold dependent. At `10⁻⁵` the reference confirms 77 legacy and 74 corrected spots out of 78. At `10⁻³` the counts are 23 and 24 out of 30. At 570 eV in the outer band, the raw HCP total at `10⁻⁴` actually falls from 63 to 62 even as the number confirmed by the direct reference rises from 41 to 54. Several weak features are sensitive to the residual interpolation between adjacent axial planes under either registration convention.

The plotted HCP (8,5,−6) cut was selected to illustrate a change of classification. It crosses the same qy row and includes the 2D maximum of all three methods. Its peak intensity is approximately 0.57, 1.45 and 2.00 × 10⁻⁴ of the direct (1,0,0) reference for the legacy, corrected and direct calculations respectively. The corrected maximum is one detector pixel to the side of the direct maximum.

## Reproduce or examine individual reflections

```bash
reg=optimization/cupy_rsoxs/fft_registration
out=test-reports/fft-registration/bragg-cpu
python "$reg/report_bragg_high_q.py" "$reg/evidence/bragg" --out "$out/supplied"
python "$reg/bragg_high_q.py" cpu --repo . --out "$out/recomputed"
python "$reg/report_bragg_high_q.py" "$out/recomputed" --out "$out/redrawn"
```

The packet includes per-coordinate CSVs for both fixtures and both energies, the direct anchor intensities, the plotted line cut, the catalog inventory, a threshold sensitivity summary and a joined spot-by-spot table (bragg_matched_spots.csv). The azimuthally averaged radial profile belongs to the public-run comparison in RG01B. This replay measures individual detector spots.

The vectorized discovery catalog is cross-checked against the repository predictor for both the broad and 1% lists. An independent, complex128 sum over all 268,435,456 voxel locations at one HCP pixel and one simple-cubic pixel verifies the reference calculation to better than 1 × 10⁻⁷ relative in intensity. The two checks are recorded in `evidence/bragg/catalog_verification.json` and `evidence/bragg/spot_verification.json`.

- [3D Bragg fixtures and current thresholds](https://github.com/usnistgov/NRSS/blob/eb7d7bf1e6d97a383ded0cef29c0d0acf2ab646c/tests/validation/test_bragg_3d_lattice.py#L31-L94)
- [Prediction, broad/strong selection and reduced-shell comparison](https://github.com/usnistgov/NRSS/blob/eb7d7bf1e6d97a383ded0cef29c0d0acf2ab646c/tests/validation/test_bragg_3d_lattice.py#L546-L620)
- [Independent lattice and Ewald predictor](https://github.com/usnistgov/NRSS/blob/eb7d7bf1e6d97a383ded0cef29c0d0acf2ab646c/tests/validation/lib/bragg.py#L503-L586)
