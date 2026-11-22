# Virtual XYZ padding and material fill · ZP05–ZP09

Extend the local working branch `perf/virtual-z-pad`. The reference checkout inspected for this packet is `74b6d09f81d56441a2c4aeaaa8246a0dce6999d2`. Its centered, vacuum-filled Z pad, FFT-bin registration and sampled radial coordinates are the starting point. A successful result will let a user simulate a particle supplied in a $128^3$ crop, receive an $8192\times8192$ detector on the corresponding $8192^3$ Fourier grid, and choose a registered material for the exterior in either padding mode. The public detector covers the normal NRSS range at the original voxel pitch.

[Inspected GitHub revision](https://github.com/usnistgov/NRSS/tree/74b6d09f81d56441a2c4aeaaa8246a0dce6999d2) · [Working branch](https://github.com/usnistgov/NRSS/tree/perf/virtual-z-pad)

## Unpack in the NRSS repository

Start in a local NRSS clone with the usual CuPy environment. If the branch is already checked out in a worktree, use that worktree. Keep existing work there. In a fresh clone the first two commands select the published branch.

```bash
git fetch origin
git switch perf/virtual-z-pad
git status --short --branch
git rev-parse HEAD
archive="$HOME/Downloads/NRSS_3D_Virtual_Padding_Tickets_20261122.zip"
unzip -t "$archive"
unzip -n "$archive"
lab=tests/validation/dev/cupy_rsoxs_virtual_3dpad
python "$lab/check_material_fill.py"
OPENBLAS_NUM_THREADS=1 python "$lab/check_literal_sphere.py" --sizes 128
python "$lab/run_sphere_ladder.py" --plan
OPENBLAS_NUM_THREADS=1 python "$lab/run_sphere_ladder.py" --preflight
```

Compare the working revision with the inspected commit. If upstream has advanced, preserve it and record the actual parent SHA. Before changing executable code, keep a clean local baseline for comparisons, e.g. `git worktree add --detach ../NRSS-3dpad-baseline HEAD`. Reuse an existing clean worktree of that SHA if available. Record any earlier uncommitted executable changes separately so a reported baseline can be reproduced. The archive adds paths alongside ZP01–ZP04. `unzip -n` also preserves local edits on re-extraction.

| Order | Local ticket | Evidence required to close |
|---|---|---|
| 1 | [ZP05 · API, geometry and references](ZP05-contract.md) | Validated options, literal material-box fixture, complex pre- and post-DC agreement |
| 2 | [ZP06 · Material fill for the Z pad](ZP06-z-material.md) | Both incumbent Z methods reproduce a physical material-filled film, including two energies and oriented interior |
| 3 | [ZP07 · Full XYZ detector](ZP07-xyz-detector.md) | Rectangular, odd and large logical grids retain registration, polarization, rotations, reductions and bounded storage |
| 4 | [ZP08 · Sphere through 8192](ZP08-sphere-ladder.md) | Seven complete panels, continuum and voxel comparisons, synchronized timing and measured peak GPU and host memory |
| 5 | [ZP09 · Compatibility and promotion](ZP09-qualification.md) | Complete maintained panels, candidate-enabled panels, matched performance comparisons and final SHA |

The [CPU reference measurements](ZP08-cpu-evidence.md) and [standalone programs](../../../tests/validation/dev/cupy_rsoxs_virtual_3dpad) can be used before implementation. The production GPU runner uses the proposed public API and the repository's NVML observer. Execute its `--run` mode on the local CUDA machine after ZP07. The plan, CPU evidence and spot checks are distinct from a production qualification.

## Handoff between tickets

Use a new result ledger such as `optimization/cupy_rsoxs/zpad_tickets/ZP05-09-RESULTS.md`. For each ticket record state (`open`, `in_progress`, `review_ready` or `qualified`), revision, environment, commands, output locations, thresholds and any issue to resolve. Keep generated panels, traces and full logs in `test-reports/virtual-3dpad/`. The handoff from ZP09 is a local promotion assessment. The original [ZP series](README_FIRST.md), its [film window policy](ZP04-film-window.md) and the [repository promotion procedure](../speed_tickets/S07-promotion.md) remain the references for incumbent behavior.
