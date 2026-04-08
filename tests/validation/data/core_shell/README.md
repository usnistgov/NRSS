Core-shell validation assets used by the legacy and maintained pytest-native
CoreShell reference work.

Files:
- `LoG_coord.csv`: legacy particle coordinate list used to stamp the morphology
- `Material1.txt`, `Material2.txt`, `Material3.txt`: vendored optical constants
- `config.txt`: legacy CLI-era run settings kept for provenance
- `CS_reference.nc`: original vendored xarray/netCDF-style reference artifact
- `CS_reference.h5`: resilient HDF5 copy for environments without optional
  netCDF backends
- `CS_sim_reference.h5`: sim-derived regression golden generated from the
  maintained pybind + WPIntegrator + manual A-wedge baseline workflow

Historical reduction note:
- The maintained CoreShell pytest path intentionally stays on
  `WPIntegrator` because the vendored experimental and sim-regression
  references were built from that detector-plane `q_perp` workflow.
- This path is maintained for historical comparability and regression
  continuity. It should not be treated as the recommended reduction pattern for
  new analytical NRSS validations, which should prefer `NRSSIntegrator` when
  the target comparison is expressed in geometry-aware `|q|`.

The maintained CoreShell migration should prefer `CS_reference.h5` for loading
the golden data because `nrss-dev` does not currently ship `netCDF4` or
`h5netcdf`.

Reference provenance:
- `CS_reference.h5` is the experimental PGN RSoXS reference used for scientific
  validation.
- `CS_sim_reference.h5` is a secondary regression artifact used to catch future
  implementation drift without claiming experimental truth.
- Experimental reference citation:
  Subhrangsu Mukherjee, Jason K. Streit, Eliot Gann, Kumar Saurabh, Daniel F.
  Sunday, Adarsh Krishnamurthy, Baskar Ganapathysubramanian, Lee J. Richter,
  Richard A. Vaia, and Dean M. DeLongchamp, "Polarized X-ray scattering
  measures molecular orientation in polymer-grafted nanoparticles," Nature
  Communications 12, 4896 (2021), doi:10.1038/s41467-021-25176-4.
