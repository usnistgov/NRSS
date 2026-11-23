# Prepared Fourier sampler validation gate

**Hold for regression.** Candidate `91b790b` matched the parent numerically and reproduced an increase in peak VRAM in the memory rerun. The tested parent was `0727a5948af63ba35e9bcf12722aabd424ebd5d2` on `backend/cupy-cyrsoxs-clone`. The separate repair `49f7030` passed the targeted checks below.

| Measured artifact | SHA or SHA-256 |
| --- | --- |
| Parent revision | `0727a5948af63ba35e9bcf12722aabd424ebd5d2` |
| Candidate revision | `91b790bf0797e00aa3b7a8b37844cf5cd0b0b857` |
| Parent NRSS source fingerprint | `88addb777f5fa23b2e7ac77dd97154a38cbe29d9b5ee49986bdde183e93220d2` |
| Candidate NRSS source fingerprint | `09f502b955047305eb9599f67fbdffdbe62e23cfe751977dfb7b14b5f63022fe` |
| Frozen lipid manifest | `c0ec325de1874d03d17d7e631a7cff38926db937998b75b70ce4335b2782f00f` |
| Frozen lipid fields | `74f32cf7c69fd5242a26749866b849b4da4c25aa9eddb36d607a36cde4f31ac2` |
| Frozen lipid scan optics | `e08bdabf601a9d2095c4e8bddcbd640f131635f3b4c8ea0c249f3814234b7537` |

The paired measurements use the cases and controls specified in FS02. The run directory `prepared-sampler.P6Ag3t` under the validation checkout's ignored `test-reports/` retains worker outputs, per-energy checks, memory traces and maintained test reports. [FS02 workload definitions](FS02-qualification.md) · [Measured source diff](https://github.com/usnistgov/NRSS/compare/0727a5948af63ba35e9bcf12722aabd424ebd5d2...91b790bf0797e00aa3b7a8b37844cf5cd0b0b857)

## Numerical and regression checks

The public Fourier oracle passed 16 captures on four geometries. All six API tests passed, including validation of input/output types, DC and stream ordering. The installed wheel imported with CuPy unloaded and sampled a field in an environment without `nrss_sals`.

The nine internal cases compared 53,772 complex basis coefficients and complete public results bit for bit. Twelve registered panels and both native sphere curves also matched bit for bit. The standalone native sphere checks met the existing thresholds. The ten primary workloads checked 586,373,022 values and coordinates, including both q-windowed and uncapped I/A. All ten A/A and A/B comparisons matched.

Both trees passed 154 focused tests. The maintained reports reproduced the same three PyHyperScattering failures across revisions, with matching test outcomes for every existing test and identical reference manifests. The sphere metrics and multi-energy integration diagnostic remain recorded in the reports.

| Lane | Parent | Candidate |
| --- | --- | --- |
| CPU smoke | 146 passed, 7 skipped | 147 passed, 7 skipped |
| Default GPU smoke | 124 passed, 10 skipped, 1 failed | 129 passed, 10 skipped, 1 failed |
| Default physics | 23 passed, 2 failed | 23 passed, 2 failed |
| Optional direct GPU smoke | 31 passed, 103 skipped, 1 failed | 36 passed, 103 skipped, 1 failed |
| Optional direct physics | 19 passed, 4 skipped, 2 failed | 19 passed, 4 skipped, 2 failed |

## Paired timing

All ten rows ran on the Wheatley RTX A6000 with Python 3.14.3, CuPy 14.0.1, CUDA runtime 12.9 and cuFFT 11.4.1. There were eight fresh A/A and eight alternating A/B pairs for each row. Ratios are B/A, with distribution-free median intervals at 92.97% coverage and the preregistered ±2% margin. The CoreShell primary boundary includes construction and synchronized public return. The lipid primary value is the median of five warmed public returns after construction.

| Case | Warm A / B (ms) | Paired change and A/B interval | A/A interval | Decision |
| --- | ---: | --- | --- | --- |
| core-small-one | 57.02 / 56.14 | -0.28% [-1.98%, +1.04%] | [-0.84%, +3.55%] | A/A inconclusive |
| core-medium-one | 335.23 / 334.55 | -0.19% [-0.33%, +1.22%] | [-0.80%, +0.45%] | neutral |
| core-large-one | 1231.44 / 1240.28 | +0.53% [+0.11%, +1.05%] | [-0.13%, +2.15%] | A/A inconclusive |
| core-small | 180.92 / 180.51 | -0.48% [-7.54%, +2.14%] | [-0.73%, +1.02%] | A/B inconclusive |
| core-medium | 492.14 / 493.71 | -0.04% [-0.71%, +0.58%] | [-0.29%, +0.52%] | neutral |
| core-large | 1467.14 / 1477.44 | +0.59% [+0.45%, +1.06%] | [-0.48%, +0.05%] | slower |
| detector-1 | 81.57 / 81.59 | +0.03% [-0.40%, +0.24%] | [-0.20%, +12.24%] | A/A inconclusive |
| detector-4 | 79.83 / 79.91 | -0.22% [-1.08%, +0.47%] | [-0.84%, +0.03%] | neutral |
| ia-full | 100.53 / 110.57 | +9.65% [-0.39%, +10.72%] | [-0.93%, +0.36%] | A/B inconclusive |
| ia-qmax | 77.64 / 78.03 | +0.55% [+0.15%, +0.90%] | [-1.10%, +0.71%] | slower |

| Case | First call A / B (ms) | Paired change and A/B interval | A/A interval | Decision |
| --- | ---: | --- | --- | --- |
| core-small-one | 199.71 / 199.91 | +0.27% [-2.42%, +3.00%] | [-4.58%, +8.22%] | A/A inconclusive |
| core-medium-one | 479.17 / 484.90 | +0.35% [-0.20%, +2.91%] | [-0.53%, +1.44%] | A/B inconclusive |
| core-large-one | 1319.69 / 1326.12 | +0.31% [-0.48%, +1.11%] | [-0.57%, +1.33%] | neutral |
| core-small | 403.60 / 405.65 | +0.83% [-0.54%, +2.31%] | [-0.66%, +1.43%] | A/B inconclusive |
| core-medium | 944.74 / 946.07 | +0.17% [-1.08%, +1.21%] | [-0.94%, +1.94%] | neutral |
| core-large | 2163.43 / 2173.46 | +0.18% [-0.21%, +0.89%] | [-1.08%, +0.14%] | neutral |
| detector-1 | 591.44 / 596.50 | +0.82% [-0.73%, +1.49%] | [-1.87%, +3.47%] | A/A inconclusive |
| detector-4 | 596.66 / 603.07 | +0.81% [+0.06%, +2.09%] | [-0.24%, +1.15%] | slower |
| ia-full | 462.96 / 473.50 | +1.70% [+0.16%, +4.46%] | [-0.41%, +0.68%] | slower |
| ia-qmax | 437.93 / 444.20 | +0.59% [-0.94%, +2.86%] | [-3.50%, +0.75%] | A/A inconclusive |

Four metrics show resolved slowdowns. Median changes were +0.59% for the large CoreShell warm run, +0.81% for the lipid detector chunk-four first call, +1.70% for the full I/A first call, and +0.55% for the cutoff I/A warm run. Full I/A warm timing was +9.65% at the median with an interval from -0.39% to +10.72%, an inconclusive result.

## Execution and peak memory

Each measurement uses fresh processes and separate 5 ms NVML observation. The full execution reports matched for each A/A and A/B pair, including planned endpoints, basis rows and bytes, FFT definitions, fallbacks, projector tiles and q demand.

| Case | Route | Basis / FFTs | Peak GPU A / B (MiB) | Peak RSS A / B (MiB) | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: |
| core-small-one | compact_r2c | 1 / 7 | 628.00 / 628.00 | 981.95 / 981.27 | 0 |
| core-medium-one | compact_r2c | 1 / 7 | 2,836.00 / 2,836.00 | 4,936.86 / 4,936.67 | 0 |
| core-large-one | compact_r2c | 1 / 7 | 9,780.00 / 9,780.00 | 15,713.37 / 15,712.43 | 0 |
| core-small | compact_r2c | 1 / 9 | 636.00 / 668.00 | 982.73 / 981.87 | 0 |
| core-medium | compact_r2c | 1 / 9 | 2,836.00 / 2,836.00 | 4,936.14 / 4,936.34 | 0 |
| core-large | compact_r2c | 1 / 9 | 9,780.00 / 9,780.00 | 15,712.61 / 15,711.00 | 0 |
| detector-1 | virtual_3d_sum | 1 / 56 | 644.00 / 692.00 | 1,535.59 / 1,535.73 | 0 |
| detector-4 | virtual_3d_sum | 1 / 56 | 644.00 / 692.00 | 1,536.95 / 1,536.85 | 0 |
| ia-full | virtual_3d_sum | 1 / 56 | 3,370.00 / 3,458.00 | 718.31 / 718.76 | 0 |
| ia-qmax | virtual_3d_sum | 1 / 56 | 1,826.00 / 1,834.00 | 713.93 / 714.85 | 0 |

| Case with a memory repeat | Repeat peak GPU A / B (MiB) | Repeat peak RSS A / B (MiB) |
| --- | ---: | ---: |
| core-small | 636.00 / 668.00 | 981.96 / 981.41 |
| core-medium | 2,836.00 / 2,836.00 | 4,935.54 / 4,935.98 |
| detector-1 | 644.00 / 692.00 | 1,536.14 / 1,535.72 |
| detector-4 | 644.00 / 692.00 | 1,535.98 / 1,536.90 |
| ia-full | 3,370.00 / 3,458.00 | 718.62 / 715.66 |
| ia-qmax | 1,826.00 / 1,834.00 | 712.62 / 714.57 |

The RTX A6000 rerun reproduced all five peak VRAM increases. The increases were 32 MiB for small multi-energy CoreShell, 48 MiB for each lipid detector chunk, 88 MiB for full lipid I/A and 8 MiB for cutoff lipid I/A. Peak RSS increases did not reproduce beyond the paired A/A variation.

## Address preparation repair

The refactor held three original coordinate arrays on the caller stack while the new helper built Hermitian addresses. Pool traces on the Quadro RTX 8000 show that the extra reservation appeared before the first virtual capture. Restoring the incumbent address calculation alone returned peak pool reservation from 409.17 to 362.99 MiB on lipid detector chunk 1 and from 3,148.57 to 3,063.20 MiB on full lipid I/A. Both values match the parent.

The separate local repair `49f7030` also restores the native gather's original temporary lifetime. It passed the Fourier oracle and all nine internal cases on the Quadro RTX 8000.

| Targeted repair case | Parent / repaired peak VRAM (MiB) |
| --- | ---: |
| core-small | 538.00 / 538.00 |
| detector-1 | 546.00 / 546.00 |
| detector-4 | 546.00 / 546.00 |
| ia-full | 3,272.00 / 3,272.00 |
| ia-qmax | 1,728.00 / 1,728.00 |

The full timing comparisons use the frozen `91b790b` candidate. The repair has targeted correctness, execution and memory evidence on the Quadro.

<details>
<summary>Diff of the separately tested repair</summary>

```diff
diff --git a/src/NRSS/backends/_cupy_energy_basis.py b/src/NRSS/backends/_cupy_energy_basis.py
index 286eab6..0be0a72 100644
--- a/src/NRSS/backends/_cupy_energy_basis.py
+++ b/src/NRSS/backends/_cupy_energy_basis.py
@@ -9 +9 @@ import numpy as np
-from ..fourier import _PreparedCapture, _r2c_addresses
+from ..fourier import _PreparedCapture
@@ -242 +242,6 @@ def _native_indices(runtime, cp, shape, indices, real_transform):
-    return _r2c_addresses(cp, shape, z, y, x)
+    # The omitted half is Hermitian in all three dimensions for real fields.
+    reflect = x > nx // 2
+    z = cp.where(reflect, (-z) % nz, z)
+    y = cp.where(reflect, (-y) % ny, y)
+    x = cp.where(reflect, (-x) % nx, x)
+    return (z * ny + y) * (nx // 2 + 1) + x, reflect
diff --git a/src/NRSS/fourier.py b/src/NRSS/fourier.py
index 2581075..bc52b60 100644
--- a/src/NRSS/fourier.py
+++ b/src/NRSS/fourier.py
@@ -58 +58,2 @@ class _PreparedCapture:
-        out[...] = cp.where(self.reflect, samples.conj(), samples)
+        samples = cp.where(self.reflect, samples.conj(), samples)
+        out[...] = samples
```

</details>

GitHub's `backend/cupy-cyrsoxs-clone` tip still matched the tested parent when the gate closed.
