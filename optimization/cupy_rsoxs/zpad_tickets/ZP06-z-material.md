# ZP06 · Add the selected material to the existing Z pad

First demonstrate the physics with fixed lateral dimensions. Preserve the `sum` and `residues` routes in [the Z evaluator](../../../src/NRSS/backends/_cupy_virtual_z.py). A user selects one numbered material to fill both exterior slabs. A vacuum request should run the current fast path. Output for old calls is a baseline comparison.

## Optical and Fourier convention

For material $j$ at energy $E$, read the same optical constants as `_material_optical_scalars`. With $n_a=1-\delta_a+i\beta_a$, the branch's isotropic response before its polarization prefactor is

$$
\chi_{j,0}(E)=\left[\frac{n_{\parallel,j}(E)+2n_{\perp,j}(E)}{3}\right]^2-1 .
$$

The existing `SFieldMode.ISOTROPIC` and exterior `S=0` for an oriented material both use that scalar. The interior follows its own Euler field and order, as originally supplied. The isotropic material is diagonal in the polarization tensor. Apply the backend's existing $1/(4\pi)$ convention exactly once. The test optics in ZP05 include unequal complex eigenvalues large enough to distinguish this expression from a weighted average of squared indices.

Let $B$ be the logical box, $C$ its centered occupied crop, and $W$ the logical amplitude window. For each energy the answer is the coherent sum

$$
\widehat{W P_{\rm crop}}+
P_{j,0}(E)\,[\,\widehat{W 1_B}-\widehat{W 1_C}\,] .
$$

Here the first field has its centered placement phase. The bracket is the real-space exterior mask transformed at the required modes. It is energy-independent. A practical insertion point is the selected material's real `Vfrac` Fourier row in [the compact energy basis](../../../src/NRSS/backends/_cupy_energy_basis.py). Add the exterior amplitude there and reuse the existing optical mixer, E-angle projection and energy tiling. Keep all oriented rows unchanged. Coherent addition before projection and squaring retains environmental interference with all interior materials. Apply the existing DC operator to the **combined** field or apply the same linear operator to both contributions before addition. For real geometry rows the replacement is the real average of the logical neighbors. Complex optical weights applied later preserve absorption. Secure complex-tensor comparisons before and after repair. For genuine 3D use six neighbors. Preserve native rules for an identity on degenerate geometry.

**A cheaper implementation to try first.** In the selected material’s *temporary* real basis row, transform `Vfrac_j - 1` inside the crop with the same window and placement phase. Add the analytically transformed entire logical box to this one row. Algebraically this is $\widehat{W(P_{\rm crop}-P_{j,0})1_C}+P_{j,0}\widehat{W1_B}$. It builds all interior contrasts with the same energy-independent geometry as the existing backend. The stored fractions and the oriented material rows keep their physical values. With an unwindowed box its uniform contribution is confined to raw DC and disappears under the prescribed DC replacement. A raw-coefficient diagnostic can reconstruct that single bin exactly. For XY Hann, the full-box term lives on the $q_z=0$ plane and carries the exact lateral Hann spectrum. It also contributes through the neighboring bins used in the repaired DC. This arrangement avoids cancellation between a near-uniform exterior and interior and can eliminate a separate general-purpose exterior-mask array.

The windowed box and windowed centered crop each have separable 1D sums. Evaluate the difference only at required endpoint bins, including the DC stencil. Axis modes containing zero need exact limits. For an unwindowed full box use its exact discrete delta. At large pads avoid subtracting nearly equal FP32 trigonometric sums. One option is disjoint slabs with a product of short interval sums. Unit tests should include first neighbors, tail bins, odd offsets and a nearly full logical box. Applying real-field Hermitian symmetry before mixing optics is safe. Conjugating an already mixed absorbing material can reverse the optical phase.

## Acceptance

1. Extend [the public GPU Z tests](../../../tests/smoke/test_cupy_virtual_z.py) for both methods and both XY windows, each with a matching explicit control. Run vacuum omitted, explicit `None`, registered zero optics, a nonzero isotropic ID, and an oriented ID whose exterior is forced to `S=0`. Exercise two dissimilar energies and multiple E angles, including an orientation producing off-diagonal components. Selecting the oriented ID must give the same exterior as a separately declared isotropic material with its $S=0$ response and the same interior.
2. Compare raw compact coefficients to a literal complex128 FFT at signed bins on and off the axes. Use the branch coefficient criterion `rtol=2e-6, atol=2e-6 * reference_peak` on each informative component. Compare full detector coordinates and finite masks exactly, relative L2 and max absolute error/peak ≤ `2e-5`. Use absolute bounds derived from the fixture scale for exactly dark components. A homogeneous matched, unwindowed box has only raw DC and should be dark after the established neighbor replacement. With XY Hann it has the spectrum of the window. An absorbing two-material off-axis fixture must resolve interference.
3. Alternate fills, energies, optics and window types in the same process. Include repeated calls, release/rebuild, identity padding, and an internal hole. Compare each to a fresh independent instance. Backwards-compatible Z result attributes should persist. Add serializable fill and window metadata from ZP07.
4. Record both Z methods against parent vacuum and dense material-fill controls. Distinguish one-time mask preparation from warm multi-energy reuse in the private diagnostics. Include one just-over-crop target and one large factor. Hand the verified geometric correction to ZP07.
