# Reusable staging and direct polarization

The GPU precompute for host FP32 inputs is implemented in the current backend. Its remaining direct-kernel speed experiment is conditional on the timings in [S05](speed_tickets/S05-residual.md). Follow the [active ticket board](speed_tickets/README.md). For provenance, the original design is available via `git show 8aa24ec0:optimization/cupy_rsoxs/shared_reusable_staging_branch_plan.md`.
