# ZP08 · Full-detector sphere ladder through 8192

The experiment holds one physical sphere, voxel pitch, energy and polarization fixed. Only the logical transform span changes. Measure every row below in the production backend. The first row is a native control. The other six use a centered XYZ pad.

| Logical cube side | Complete detector pixels | $\Delta q$ (nm⁻¹) | One FP32 detector (MiB) | One literal FP32 volume (GiB) |
|---:|---:|---:|---:|---:|
| 128 | 16,384 | 0.0490874 | 0.0625 | 0.0078125 |
| 256 | 65,536 | 0.0245437 | 0.25 | 0.0625 |
| 512 | 262,144 | 0.0122718 | 1 | 0.5 |
| 1024 | 1,048,576 | 0.00613592 | 4 | 4 |
| 2048 | 4,194,304 | 0.00306796 | 16 | 32 |
| 4096 | 16,777,216 | 0.00153398 | 64 | 256 |
| 8192 | 67,108,864 | 0.000766990 | 256 | 2048 |

## Three references with different jobs

**Continuum sphere.** Use a centered 70 nm diameter sphere on the same $128^3$ occupied array for all seven runs, $d=1$ nm, 285.0 eV, $E\parallel X$, `EAngleRotation=[0,0,0]`, `RotMask=0` and `WindowingType=0`. The boundary has $4^3$ subvoxel occupancy samples. The fixed SHA-256 and optical tables are in [the runnable reference](../../../tests/validation/dev/cupy_rsoxs_virtual_3dpad/sphere_reference.py) and [CPU evidence](ZP08-cpu-evidence.md). The second material inside the crop is the complementary matrix. In the vacuum campaign it has vacuum optics. In the medium campaign it has nonzero synthetic isotropic optics and the exterior selects the same ID 2. This is an unwindowed particle embedded in a homogeneous finite matrix. The constant whole-box term occupies raw DC, which NRSS repairs.

Write $u=R|\mathbf q|$, $k=2\pi E/1239.84197$ in nm⁻¹, $V=4\pi R^3/3$ and $\Delta\chi=\chi_{\rm sphere}-\chi_{\rm matrix}$. On the forward Ewald surface at each valid pixel,

$$
q_z=-\frac{q_x^2+q_y^2}{k+\sqrt{k^2-q_x^2-q_y^2}}, \qquad
\frac{I_{\rm sphere}}{I_0}=\left(1-\frac{q_x^2}{k^2}\right)
\left[\frac{3(\sin u-u\cos u)}{u^3}\right]^2,\qquad
I_0=\left[\frac{k^2(V/d^3)|\Delta\chi|}{4\pi}\right]^2 .
$$

The forward amplitude limit is one. The neutral `+qy` line has unit dipole factor. Record the replaced DC separately and compare it to the discrete prescription. Use the fixed, **unrepaired** $I_0$ for normalization across the ladder. Compare non-DC full-detector power, $0<qR_g\le1$, $2\le qR\le14$, and $q\ge1$ nm⁻¹ independently. Use band-normalized L2 with counts. Around a true zero, report absolute intensity/$I_0$, bin q and the three-point interpolated minimum, including an unresolved flag on coarse grids. The first three minima are 0.1283831, 0.2207215 and 0.3115463 nm⁻¹. The theoretical $R_g=\sqrt{3/5}R=27.11088$ nm. Report the fitted slope, point count and fixed fit interval. A finite Guinier fit can have a small systematic bias even for a continuum sphere. The existing [sphere validation](../../../tests/validation/test_analytical_sphere_form_factor.py) and its established thresholds also run unaltered. [Standard sphere form factor](https://www.sasview.org/docs/user/models/sphere.html)

**Physical voxel reference.** For every rung compare fixed signed off-axis and near-DC pixels with directly evaluated complex128 sums of the original voxels. Compute both registered axial endpoints, replace DC from its logical neighbors, project them separately, then interpolate their powers. Expand the full dense comparison at the manageable rungs and use the independent CPU construction for larger complete panels. The supplied program scores *every valid pixel* against the analytic law and verifies the full mask. For a stronger large-panel acceptance test compare candidate pixels to the CPU pilot at every valid pixel, streamed by rung/row, including the radial/angular tails. The CPU pilot has no NRSS imports. Its scalar simplification is valid for this specific unwindowed matched matrix. ZP06 supplies the general finite-box reference. For this sphere the numerical comparator applies the known optical precision factor $|\Delta\chi_{\rm FP32}/\Delta\chi_{\rm tabulated}|^2$, derived from the branch’s stored optical coefficients before inspecting the output. The driver records it and the uncorrected anchor discrepancy separately. The continuum uses the tabulated constants at full precision. On NumPy 2.3.5 this factor is 0.99999994 for vacuum and 1.00029226 for the synthetic medium. Fitting a scalar to the panel would conceal errors in absolute intensity. Use the GPU runner option `--verify-full-voxel` for this comparison in the accuracy worker.

**Incumbent resource controls.** Measure three implementations for the same specimen, window, pitch and logical detector wherever the machine can hold them. A is a literal $M^3$ box through the incumbent native backend. B is the usual `(128,M,M)` laterally explicit specimen with incumbent virtual Z extended to `M`. C is XYZ from $128^3$. Place the same matrix in the lateral exterior and Z exterior in B. A/B are identical calculations to C at this shape, making speed ratios and numerical deltas meaningful. For the nonzero-medium B arm use the **ZP06** material-aware Z implementation and identify its revision. In the vacuum arm B can use the pinned parent. The supplied driver builds A/B/C from the same compact voxel array. The parent `--material-z` control alone changes the specimen. A reasonable initial set is literal native A at 128 and 256, then 512 and larger when safely feasible, and B at least through 1024 where feasible. Record peak or capacity limit per control and all sizes actually run. Log allocation estimates separately from measured peaks. A literal 8192 cube needs 2 TiB for just one float field. The complete virtual 8192 measurement remains required. A dedicated GPU or internal tiling is appropriate if the first attempt exceeds available memory.

## Run on the candidate

```bash
lab=tests/validation/dev/cupy_rsoxs_virtual_3dpad
out=test-reports/virtual-3dpad/sphere
python "$lab/run_sphere_ladder.py" --run --gpu 0 --fill vacuum --verify-full-voxel --repeats 5 --out "$out/vacuum"
python "$lab/run_sphere_ladder.py" --run --gpu 0 --fill 2 --verify-full-voxel --repeats 5 --out "$out/material-2"
```

Each invocation prints its unique result directory. Give that directory to `plot_sphere_ladder.py RUN_DIRECTORY --out FIGURE_DIRECTORY` for a per-rung CSV and PNG. During development, `--sizes 128 256` selects a quick pair. The qualification run takes all seven sizes in **both** campaigns. Compare ID 2 to the vacuum case in absolute units and after contrast normalization. Its geometry is identical. Any measured time or memory premium is attributable to the material path within benchmark variability.

For the overlapping control sizes, from the candidate checkout, use its driver to import the pinned parent for the physical cube and vacuum film. Increase the upper control sizes after checking capacity. The nonzero-medium film control imports the candidate so it has the ZP06 material feature.

```bash
baseline=../NRSS-3dpad-baseline
python "$lab/run_sphere_ladder.py" --run --gpu 0 --arm literal --nrss-root "$baseline" --sizes 128 256 --fill vacuum --save-panel --out "$out/literal-vacuum"
python "$lab/run_sphere_ladder.py" --run --gpu 0 --arm literal --nrss-root "$baseline" --sizes 128 256 --fill 2 --save-panel --out "$out/literal-material-2"
python "$lab/run_sphere_ladder.py" --run --gpu 0 --arm film --nrss-root "$baseline" --sizes 256 512 1024 --fill vacuum --save-panel --out "$out/film-vacuum"
python "$lab/run_sphere_ladder.py" --run --gpu 0 --arm film --sizes 256 512 1024 --fill 2 --save-panel --out "$out/film-material-2"
```

Use `--save-panel` on matching XYZ diagnostic runs for a direct A/B/C array comparison. Each arm also accepts `--verify-full-voxel`. Add `--z-method residues` when benchmarking that Z method. The literal target embeds the medium explicitly and runs on the parent for both fill campaigns.

The [driver](../../../tests/validation/dev/cupy_rsoxs_virtual_3dpad/run_sphere_ladder.py) uses distinct fresh workers for end-to-end timing, memory, and accuracy. Record synchronized first `Morphology.run` and five warm calls, individual values, median and spread. Morphology construction and file I/O are outside the reported run time. The canonical result is a host xarray detector and includes its return transfer. For memory retain per-process NVML peak, device baseline/peak/delta, 10 ms polling, host peak RSS, GPU identity and versions. A polled peak is a sampled peak. Instrument CuPy live/reserved pool or allocator high-water and cuFFT workspace where available to catch shorter transients. Measure device-resident output separately if reported. Record valid pixels/s and scaling with $M$, geometry/FFT/axial gather/assembly stage counters in a separate diagnostic trial. Use `--stages` on representative cases and add the new segment names to the candidate speed harness. Include baseline A/A variability alongside matched comparisons. Save arrays or reproducible tiled comparisons and logs. OOM, an incomplete panel, a silent native fallback or an unavailable peak leaves that rung open.

## Acceptance and interpretation

- Exact output dimensions, logical FFT-bin axes, Ewald/NaN mask, energy, radial metadata and expected executed route at every rung. The CPU pilot counts 11,140,041 valid pixels out of 67,108,864 on the largest panel with these settings.
- Finite voxel agreement for fully matched panels at ZP06 tolerances. Discrete anchors, including repaired DC, must also satisfy max absolute error/$I_0\le 2\times10^{-5}$. Apply an additional full-voxel high-q relative L2 threshold of 0.002 for this sphere. Track differences relative to the reference's own band norms near minima. For this scalar, unwindowed fixture use band-error differences within `max(2e-4, 0.005 × CPU band error)`, on populated bands, as an additional review threshold. Report any disagreement even if global error passes.
- For `M≥2048` require each resolved first-three minimum within one local q interval of its analytic root. Compare the GPU and CPU fitted $R_g$ within 0.1 nm on the same bins when fit exists. The CPU high-resolution fit is about 27.41 nm for this chosen interval. At smaller rungs report missing fit points and ambiguous minima explicitly. Include full-q and first-minimum figures plus timing/memory curves. A band error need not decrease at every increment.
- The CPU pilot found that the upper-q band levels off around 11% relative L2 on this fixture. Report that alongside its low absolute intensity and plot the tail on a logarithmic scale. To attribute the floor, halve pitch at fixed physical radius in a separate targeted diagnostic and/or evaluate directly at continuous $q_z$. Keep that separately labeled from the fixed-rung speed series. Score the observed change without extending the detector q range. Dense voxel equivalence is the test of the padding implementation.
