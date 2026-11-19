# CuPy RSoXS benchmarking

Start with the [measurement protocol and working commands](speed_tickets/MEASUREMENTS.md), then the [ticket for the active experiment](speed_tickets/README.md). Use [S07](speed_tickets/S07-promotion.md) for the full smoke/physics and output-panel requirements before a push.

The current development worker is `tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py`. The ticket launcher adds paired runs and supplies `enum_contract`, explicit execution paths and matching outputs. For terminal-specific work use `run_energy_progress_bar_overhead.py` in the same directory. For cross-backend memory claims use the host-input comparison through `tests/validation/dev/core_shell_backend_performance/run_comprehensive_backend_comparison.py`.

For a quick CoreShell regression, select the applicable path on the maintained test:

```bash
python -m pytest tests/validation/test_core_shell_reference.py -k test_core_shell_sim_regression_pybind --nrss-path cupy_tensor_coeff -v
python -m pytest tests/validation/test_core_shell_reference.py -k test_core_shell_sim_regression_pybind --nrss-path cupy_direct_polarization -v
```
