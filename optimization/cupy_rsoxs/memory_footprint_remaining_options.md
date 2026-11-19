# GPU memory and current-workflow speed

Use the [measurement protocol](speed_tickets/MEASUREMENTS.md) for external peak-memory comparisons. Detector gathering is in [S01](speed_tickets/S01-gather.md), energy cache size in [S02](speed_tickets/S02-energy-basis.md), and selective Fourier storage in [S04](speed_tickets/S04-selected-z.md). The backend already supports host output streaming. Historical memory measurements are available via `git show 8aa24ec0:optimization/cupy_rsoxs/memory_footprint_remaining_options.md`.
