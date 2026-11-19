# S02 · Transform real morphology fields once per scan

**Priority.** This is the first algorithmic experiment for multi-energy workflows. The incumbent builds complex tensor components at every energy. Preserve NRSS's `Vfrac` and `S` mixing exactly and move the energy-dependent coefficients through the linear Fourier transform. Build this first for the FP32 tensor path, with one energy at a time in the recombination loop. That control isolates reuse from batching. Compare with the direct path as a separate existing workflow.

For material `m`, set `v=Vfrac`, `φ=vS` and `u=(cosψ sinθ, sinψ sinθ, cosθ)`. Obtain the three **complex** coefficients from the existing `_material_optical_scalars`. In mathematical notation,

$$
n_o=(n_\parallel+2n_\perp)/3,\qquad
a=n_o^2-1,\quad b=n_\perp^2-n_o^2,\quad c=n_\parallel^2-n_\perp^2 .
$$

The five tensor components used at current incidence obey

$$
N_{ij}(E)=\sum_m \left[\delta_{ij}(a_m(E)v_m+b_m(E)\phi_m)
             +c_m(E)\phi_m u_{mi}u_{mj}\right],\quad ij=xx,xy,xz,yy,yz .
$$

Use the helper's existing precision/cast order for the actual coefficients. For each anisotropic material form the seven real fields $v,\ \phi,\ \phi u_x^2,\ \phi u_xu_y,\ \phi u_xu_z,\ \phi u_y^2,\ \phi u_yu_z$. An explicit isotropic material contributes only `v` weighted by `a`, even when its optical table contains distinct parallel and perpendicular values. For each required Fourier endpoint assemble the **complex amplitudes** across all materials and terms, use the existing normalization and the energy-specific outgoing projector, compute the squared norms/cross coefficient, then interpolate the two endpoint intensity coefficients as the current detector code does. Preserve the cross terms and the E-angle average.

**Implementation comparison.**

A CPU algebra check on a 7 × 9 × 12 asymmetric grid used the branch's complex optical-scalar and DC helpers, fractional order, a Hann window and three distinct test energies. Building all five complex tensor fields before their FFTs agreed with recombining their separately transformed real bases to a worst relative complex L2 difference of 1.05 × 10⁻⁷. Both calculations used the same mixing coefficients. This checks the factorization and gives no GPU timing or detector-panel tolerance.

1. Reference the active `_compute_nt_components`, `_compute_fft_nt_components` and the current projector. Produce a dense complex FFT of each real basis as an easily checked first implementation if helpful. Measure a scan with the basis built **once inside** the timed boundary. A dense cache is a diagnostic control only where it fits.
2. Transform a basis at a time. Gather its values at the **union of required (z,y,x) indices** for the scan and cache compact complex values with per-energy gather maps. Then recombine one energy at a time. An alternative streams a bounded group of energies per basis pass. If this repeats FFTs across groups, charge and count those repeats. Prefer a shared union when it fits.
3. Test R2C as a separate variant. For a missing negative-frequency sample conjugate the real basis transform at the reflected index on **all three axes**. Recombine with complex optical coefficients afterward. DC replacement and the common window are linear, so apply them consistently to each basis. Keep uniform/window terms and Hermitian edge cases. Cross-call caching requires immutable/versioned morphology or explicit invalidation including fields/order/window, shape, pixel size and the endpoint energy/grid plan. A per-run cache is sufficient for the first experiment.

CoreShell has one anisotropic shell and two isotropic material representations. Its shell and matrix optical tables are identical at all 101 keys, although their orientation fields differ. Use distinct spectra for the generic check before claiming generality. For general E directions the conservative count is nine real-basis transforms instead of five complex transforms **per energy**. At 101 energies that is 9 versus 505 if the basis is transformed once. The direct path takes 3 × 12 × 101 component transforms on the 12-angle case. These are counts. Measure full host/device delivery, the cost of building the basis, gathering, mixing, and the scan-size crossover. Compare against the faster of the two real incumbents before recommending a user-facing switch.

**Run.** Use `one` as the protected low-reuse control, `three` for correctness and short scans, `energy20` first for ranking, and `energy101` for confirmation. Include both a single E direction and the 12-angle routine. Do the first long run on small CoreShell, then confirm the crossover on medium and on one representative reduced-output workflow. Record actual transform counts, basis bytes, union size, build time and scan throughput. An optional fixed-morphology optical-refit measurement must include first-build time and a changed spectrum on the second scan.

**Spot check.** Compare first/middle/last detector panels plus a small generic case with distinct complex spectra and fractional order. At one endpoint compare a mixed complex tensor component before squaring. Reorder the requested energy list and verify outputs follow it. Preserve the complex amplitude from a translated pattern and cancellation near a contrast match. The specific output of each energy still requires that energy's Ewald weights and projector. Use the full S07 gate for a retained candidate.

## Experimental implementations

Select `energy_basis_mode` with `--candidate-options` and `--path tensor_coeff`. The `compact_c2c` mode transforms each real field in a reusable complex cuFFT buffer. The `compact_r2c` mode uses a real FFT and conjugates samples from the reflected three-dimensional frequency index. Both retain complex values at the union of the detector's integer endpoints for the duration of one scan. Each energy's gather indices are looked up as needed. The amplitudes are assembled with the existing optical scalars and passed to the same tensor projector and rotation average used by the incumbent. A single distinct energy, a direct-path request, or reduced-precision or collapsed-depth inputs use the incumbent route with a recorded reason.

For the host-resident medium and large scans, `--shape-label medium` selects 64 × 1024 × 1024 at 1.25 nm and `--shape-label large` selects 96 × 1536 × 1536 at about 0.833 nm. Use `--case energy20` or `--case energy101` with each size. The routine rotation uses `EAngleRotation = [0, 15, 165]` degrees in `[StartAngle, IncrementAngle, EndAngle]` order. Use `--rotation off` for a single direction at the same shape and energies.

```bash
RUN=tests/validation/dev/cupy_rsoxs_optimization/run_speed_tickets.py
python "$RUN" --baseline-root /homes/deand/dev/NRSS_speed --candidate-root /homes/deand/dev/NRSS \
    --case energy20 --shape-label large --path tensor_coeff --resident host \
    --candidate-options '{"energy_basis_mode":"compact_r2c"}' --repeats 5
```

Run `--memory` separately with the same conditions. The worker reports sampled device-wide peak usage above the observer's idle baseline. The stage report includes the compact cache size and its endpoint count. [Measurements](RESULTS.md)
