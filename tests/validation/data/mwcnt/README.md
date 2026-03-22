This directory contains the minimal vendored inputs needed for the MWCNT experimental validation test.

Files:
- `MWCNT_reference_A.h5`: reduced experimental anisotropy observables used by the test.
- `MWCNT_opts.csv`: interpolated MWCNT optical-constant source table used by the maintained morphology builder.
- `mwcnt_seed12345_cnts.csv`: fixed-seed CNT geometry table used to build the deterministic test morphology.

Geometry provenance:
- The vendored geometry corresponds to the tutorial/manuscript MWCNT parameterization with:
  - `theta_mu = pi/2`
  - `theta_sigma = 1/(2*pi)` rad
  - `hollow_fraction = 0.325`
- The tutorial RSA generator exposes the radius distribution in lognormal sampling parameters:
  - `radius_mu = 2.225`
  - `radius_sigma = 0.23`
- After the tutorial's `2x` downscale, those generator settings produce an effective radius distribution close to manuscript Table I:
  - manuscript Table I target: mean radius `4.60 nm`, std `1.03 nm`
  - vendored fixed-seed realization: mean radius `4.495 nm`, std `1.014 nm`

The experimental reference was reduced from the tutorial WAXS dataset to the following maintained observables:
- `A(E)` averaged over `q = 0.6-0.7 nm^-1`
- `A(q)` at `285 eV`, restricted to `q = 0.20-0.95 nm^-1`
- `A(q)` at `292 eV`, restricted to `q = 0.20-0.95 nm^-1`

Published provenance:

Dudenas, P. J.; Flagg, L. Q.; Goetz, K.; Shapturenka, P.; Fagan, J. A.; Gann, E.; DeLongchamp, D. M. *J. Chem. Phys.* **2025**, *163* (6), 061501. https://doi.org/10.1063/5.0267709.
