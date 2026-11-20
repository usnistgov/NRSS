# RG02 · Migrate the simulation regression and run the panels

**Deliverable.** A versioned corrected CuPy simulation reference, the historical reference on the legacy path, and a reviewed assessment of both measured fixtures.

| Existing check | Treatment |
| --- | --- |
| CoreShell `CS_reference.nc` and `CS_reference.h5`, MWCNT `MWCNT_reference_A.h5` | Preserve all measured data, axes, optical inputs and reduction settings. Apply the current experimental limits in the first full comparison. |
| `CS_sim_reference.h5` | Keep the historical file and its test on `legacy_cyrsoxs`. After RG01, RG01B and independent validation, use a separately named `CS_sim_reference_fft_bins_v1.h5` for both corrected CuPy paths. This file contains reduced A-wedge values. |
| CoreShell `z_collapse_mode="mean"` | Compare the approximate calculation to corrected **full depth**, using its existing approximation limits initially. Retain the single-plane identity checks. |
| Material Euler scaling | `_detector_annulus_mask` in `tests/validation/test_sphere_orientational_contrast_scaling.py` uses inclusive endpoints. Derive the corrected annulus from reported coordinates and key any separately retained historical mask by convention. Preserve the 18 orientations, materials and existing scaling tolerance. |
| Sphere/disk and 2D/3D Bragg tests | Preserve the existing form factors, predictions and acceptance windows. Attach the [indexed high-q Bragg comparison](RG01B-bragg.md) and the grating graphical abstracts. Select any added weak-peak assertion from independently verified reflections. Compare appropriate quantities (q_perp for disks, scattering-vector magnitude for applicable 3D peaks). Trace active call sites. The sphere's `_flat_detector_analytic_image` helper is outside its maintained evaluation path. |
| Computation equivalence | Keep the strict same-convention tolerances on raw panels, coordinates, finite masks, host/device, compact/fallback, multi-energy and reduction parity. The CLI/pybind check still runs on the legacy backend. |

1. Select the simulation reference explicitly by convention in `tests/validation/lib/core_shell.py` and `tests/validation/test_core_shell_reference.py`. Names ending in `_pybind` can expand across all three paths. Key cached products by actual path/options, geometry and reducer where applicable. The `reference_parity` marker also protects analytic physics tests. Preserve its coverage. The current `core_shell_reference_diagnostic.py --write-sim-reference` writes the historical file. Give the writer an explicit output destination before generating a second reference.
2. Assert coordinate values and order, dimensions, finite masks, and comparison counts before subtracting. `compute_awedge_metrics` currently uses positional values and NaN-aware statistics. For old versus corrected raw panels report common-support max and normalized L2 per energy, plus counts newly valid and newly invalid. Capture masks before sanitizing invalid samples. For matched algorithms retain the strict equality requirements of the current speed harness (`--accuracy full`).
3. Use the historical WPIntegrator reduction and record its version and CPU/GPU route. CoreShell has 101 energies, 361 angles, A(E) over 0.02–0.40 nm⁻¹ and A(q) at 284.7 and 285.2 eV. MWCNT has 55 energies, A(E) over 0.6–0.7 nm⁻¹, and A(q) at 285 and 292 eV over 0.20–0.95 nm⁻¹. Inspect raw intensity, horizontal/vertical sector means and measured-minus-simulated curves at changed points. A native NRSSIntegrator comparison exercises a separate reduction contract.
4. Reproduce the CoreShell 285.2 eV discrepancy on the workstation. If confirmed, investigate its energy and q dependence, fixed morphology/optics, reduction, and experimental uncertainty. Use independent experimental evidence for any revised validation criterion. A regenerated simulation file cannot decide that question. Hold promotion while the experimental gate is unresolved.
5. After the independent checks and the two full CuPy runs agree, generate the versioned 101 × 363 simulation reference from the pinned full-depth calculation and the historical reducer. Cross-check with the other CuPy route and the two CPU anchor energies. Record file hash, input hashes, geometry, code SHA, GPU and library versions, reduction path, axes, masks, and before/after panels. Run the complete report with the versioned file. Use `--stages --expect-route` for an instrumented benchmark if needed to establish which multi-energy route executed. Keep uninstrumented timing runs for speed comparisons.

```bash
mkdir -p test-reports/fft-registration
export NRSS_TEST_ENERGY_BASIS_LOG="$PWD/test-reports/fft-registration/energy-basis.jsonl"
env -u NRSS_TEST_PATH -u NRSS_PATH -u NRSS_TEST_BACKEND -u NRSS_BACKEND \
  scripts/run_local_test_report.sh -e "$CONDA_DEFAULT_ENV"
```

**Before push.** Inspect `summary.md`, `steps.tsv`, per-case statuses, all three paths, new CuPy cases, and the graphical-abstract outputs. Account for every changed physics result and any new skip. After acceptance, record the corrected unpadded SHA as the numerical parent for ZP01. Generate the literal-padded and virtual-padded controls under that convention. The former speed-series SHA remains a historical timing reference.
