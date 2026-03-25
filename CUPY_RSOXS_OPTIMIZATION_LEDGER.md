# `cupy-rsoxs` Optimization Ledger

This document is the authoritative speed-optimization ledger for `cupy-rsoxs`. It records timing methodology, accepted and rejected experiments, current benchmark authority, and deferred optimization directions.

Stable backend contract and phase-1 behavior live in `CUPY_RSOXS_BACKEND_SPEC.md`. Repo-wide upgrade planning and test-program status live in `REPO_UPGRADE_PLAN.md`.

## Current Optimization Guidance And Ledger

This section replaces the earlier optimization-inventory framing with a more
resumable guidance-and-ledger record. It is intended to let a fresh context
recover:
- the current accepted backend state,
- the measurement caveats,
- the official resident-mode guidance,
- the optimization campaign strategy,
- and the work that has already been tried, accepted, or rejected.

### Current state and authoritative artifacts

1. The current accepted tuned backend state is the behavior reached after the
   first optimization campaign:
   - dead `Nt[5]` removed from the supported Euler-only / `PARTIAL` path,
   - FFT storage reused so separate `fft_nt` residency is reduced,
   - Igor reorder implemented with a cached `RawKernel`.
2. The authoritative optimization timing harness now lives at:
   - `tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py`
   - supporting notes at `tests/validation/dev/cupy_rsoxs_optimization/README.md`
3. That harness is now explicitly cupy-only and timing-only:
   - primary timing starts immediately before `Morphology(...)`,
   - primary timing ends immediately after synchronized `run(return_xarray=False)`,
   - upstream field generation is excluded,
   - export timing is excluded,
   - Segment `A1` is measured in the harness and Segments `A2-F` are measured
     in `cupy-rsoxs`,
   - Segment `G` remains reserved for future export timing.
4. The backend comparison dev library now has a principal cross-backend
   primary-time comparison entry point at:
   - `tests/validation/dev/core_shell_backend_performance/run_primary_backend_speed_comparison.py`
   - supporting notes at `tests/validation/dev/core_shell_backend_performance/README.md`
   - output artifacts:
     - `primary_backend_speed_comparison_summary.json`
     - `primary_backend_speed_comparison_table.tsv`
     - `primary_backend_speed_comparison_table.png`
5. The reusable full-energy backend-comparison harness still lives at:
   - `tests/validation/dev/core_shell_backend_performance/run_core_shell_backend_performance_abstract.py`
   - supporting notes at `tests/validation/dev/core_shell_backend_performance/README.md`
6. The full-energy backend-comparison harness is now legacy/historical only for
   optimization work:
   - it is not the authoritative timing harness,
   - its old mixed workflow metric should not be used to rank current speed
     work,
   - no new `cyrsoxs` timing lanes should be added to the default optimization
     loop.
7. Current development artifacts that should be treated as the authoritative
   resumable timing record root are:
   - `test-reports/cupy-rsoxs-optimization-dev/`
   - the current harness writes fresh per-run `summary.json` files under that
     root using host/device resident-mode labels such as
     `core_shell_small_single_no_rotation_host`
   - historical snapshots such as
     `test-reports/cupy-rsoxs-optimization-dev/verify_cli_small_postcleanup/summary.json`
     remain useful for context, but they predate the resident-mode/default-lane
     refactor and should not be treated as the naming or workflow authority
8. Maintained parity and physics guardrails still live in the official test
   suite. The dev harnesses above are for optimization and comparison work, not
   for replacing the maintained parity suite.

### Timing apparatus status (implemented March 23, 2026)

1. The timing repair requested in earlier versions of this roadmap is now
   complete for the current optimization loop.
2. The primary optimization wall metric is:
   - start immediately before `Morphology(...)` construction,
   - end immediately after synchronized `run(return_xarray=False)` completion.
3. The primary optimization wall metric excludes:
   - field generation before object creation,
   - result export such as `to_xarray()`,
   - downstream A-wedge generation, plotting, and analysis.
   - in a fresh host subprocess, Segment `A2` may still include first-touch
     CUDA/CuPy bring-up and therefore is not always a pure transfer metric.
4. The implemented internal-only control path is:
   - `Morphology._set_private_backend_timing_segments(...)`
   - `Morphology._clear_private_backend_timing_segments()`
   - this is intentionally private and should remain internal unless a later
     API review explicitly promotes it.
5. Implemented segment coverage is:
   - Segment `A1`: harness wall time for morphology construction /
     contract normalization inside the primary timing boundary,
   - Segment `A2`: `cupy-rsoxs` private wall time for runtime staging,
   - Segments `B-F`: `cupy-rsoxs` private CUDA-event timings,
   - Segment `G`: deferred for a later export-timing pass.
6. The backend timing payload now has the shape:
   - `{"measurement": "...", "selected_segments": [...], "segment_seconds": {...}, "segment_measurements": {...}}`
7. When timing is not explicitly enabled:
   - `morphology.backend_timings` remains `{}`,
   - `cupy-rsoxs` does not enter the CUDA-event timing path,
   - production runs therefore avoid development-only synchronization overhead.
8. The old dev-harness `workflow_seconds` metric is intentionally discarded and
   should not be revived as an optimization authority.
9. Segment totals are not expected to sum exactly to `primary_seconds`:
   - Segments `A1` and `A2` are wall-clock,
   - Segments `B-F` are CUDA-event timings,
   - residual host/launch overhead remains in the primary wall metric.
10. The live dev harness now also supports an optional untimed CUDA prewarm
    mode for steady-state many-morph comparisons:
    - `--cuda-prewarm before_prepare_inputs` performs a tiny host->CuPy touch
      inside the worker before `_prepare_core_shell_case_inputs(...)`,
    - default remains `off` to preserve cold subprocess measurements,
    - device-resident cases record `redundant_device_prepare`,
    - allocator/pool refresh behavior is intentionally unchanged.
11. Verification completed in `nrss-dev` for this pass:
    - smoke coverage now includes
      `test_cupy_private_segment_timing_is_opt_in_and_subsettable`,
    - direct single-energy / no-rotation CoreShell worker timing passed,
    - subset selection such as `("B", "D")` returned only the requested
      backend segments,
    - CLI timing harness run `verify_cli_small_postcleanup` passed.

### Current evidence summary from the accepted backend state

1. The first optimization campaign retained three changes and rejected four
   first-pass implementations. See "Campaign ledger from work already done"
   below.
2. Historical full-energy `cyrsoxs` comparison reports remain useful as context
   for what was explored, but they are no longer the authoritative timing basis
   for resumed optimization work.
3. The first post-cleanup timing snapshot remains:
   - `test-reports/cupy-rsoxs-optimization-dev/verify_cli_small_postcleanup/summary.json`
4. That snapshot is still useful as historical evidence that the repaired
   timing boundary was working, but it predates the current resident-mode
   strategy and current case naming.
   - it uses older labels such as
     `core_shell_small_single_no_rotation_cupy_borrow`
   - the current harness now emits resident-mode-explicit labels such as
     `core_shell_small_single_no_rotation_host`,
     `core_shell_small_single_no_rotation_device`,
     and opt-in limited-rotation cases for either variant
5. Historical timing numbers from that old post-cleanup snapshot are still
   worth keeping as rough context for the then-accepted device-resident lane:
   - `core_shell_small_single_no_rotation_cupy_borrow`
     - `primary_seconds`: about `0.2363s`,
     - Segment `A`: about `0.0028s`,
     - Segment `B`: about `0.1319s`,
     - Segment `C`: about `0.0122s`,
     - Segment `D`: about `0.0694s`,
     - Segment `E`: about `0.0034s`,
     - Segment `F`: about `0.00047s`
   - `core_shell_small_triple_limited_rotation_cupy_borrow`
     - `primary_seconds`: about `0.4480s`,
     - Segment `B`: about `0.1936s`,
     - Segment `D`: about `0.2001s`,
     - Segment `E`: about `0.0153s`
6. Current interpretation from the repaired timing apparatus:
   - on the primary small single-energy lane, current latency is dominated by
     Segments `B` and `D`,
   - Segment `C` is visible but materially smaller,
   - Segment `A1` is currently minor in that lane,
   - host-resident `A2` can still be material and should be interpreted
     separately from constructor overhead,
   - Segments `E` and `F` are currently minor contributors in that lane.
7. Important benchmark caveats for the harness going forward:
   - the default host-resident lane is meant to resemble the most common public
     workflow and therefore includes host-resident morphology handling in the
     primary wall-clock metric,
   - in host mode, host-to-device staging happens inside `run()` and is counted
     in total wall time,
   - that staging is now isolated as Segment `A2` inside the private timing
     breakdown.
8. Additional caveat for the opt-in device-resident lane:
   - the `cupy -> cupy-rsoxs (device/borrow/strict)` path is contract-clean at
     the `Morphology` boundary,
   - but it is not a true end-to-end GPU-native morphology-generation
     benchmark,
   - the current CoreShell builder still creates fields in NumPy and then
     preconverts them to CuPy before `Morphology(...)` timing starts.
9. Latest Segment `A` evidence from the current accepted state:
   - `segA12_probe_20260323` established that Segment `A1` constructor work is
     already minor while host-resident Segment `A2` staging is the transfer
     cost center,
   - `a2_exp1_empty_set_20260323` replaced the host-resident NumPy -> CuPy
     staging fast path with `cp.empty(...); out.set(host)` and improved the
     small single-energy host lane by about `7.5%` on primary wall time and
     about `7.0%` on Segment `A2` versus the split-only baseline,
   - in the corresponding device-resident lane, Segment `A2` is effectively
     zero because the morphology fields already satisfy the backend-preferred
     device contract before `run()` begins,
   - later steady-state host-prewarm comparisons showed that fresh-subprocess
     startup dominates the cold host lane:
     - cold host / `legacy_zero_array`: `primary 2.546s`, `A2 2.363`,
     - cold host / `enum_contract`: `primary 2.545s`, `A2 2.369`,
     - prewarmed host / `legacy_zero_array`: `primary 0.282s`, `A2 0.0887`,
     - prewarmed host / `enum_contract`: `primary 0.226s`, `A2 0.0461`.
10. Practical prioritization interpretation from that evidence:
   - Segment `A` is nominally complete for the common workflow,
   - cold host `A2` can be startup-dominated, so steady-state host comparisons
     should use the prewarm option when the workflow model is many morphs per
     process,
   - if a workflow expects morphology fields to remain on GPU,
     `resident_mode='device'` is the intended faster tradeoff and should
     remain visibly faster than host-resident staging,
   - default future speed work should focus on Segments `B` and `D` unless new
     timing evidence changes the ranking.
11. Maintained parity remains a post-optimization check through the test suite
   rather than through a `cyrsoxs` timing harness.

### Official resident-mode guidance

1. Official guidance for `cupy-rsoxs` is now to support controllable resident
   modes, with CPU / host-resident behavior as the default.
2. Host-resident staged mode is the default guidance for public workflows.
   - authoritative morphology fields remain on CPU,
   - the backend materializes temporary CuPy arrays when math requires them,
   - temporary device arrays should be deleted or replaced aggressively as the
     pipeline advances,
   - this mode is expected to lower GPU memory pressure at the cost of at least
     one host-to-device transfer.
3. Device-resident direct mode is an opt-in path for already-CuPy morphology
   fields.
   - authoritative morphology fields already satisfy the backend-preferred
     CuPy contract,
   - the backend may borrow and use those arrays directly,
   - this mode is expected to use more GPU memory but may help GPU-native
     workflows or later on-device chaining.
4. Resident mode must be documented separately from `input_policy` and
   `ownership_policy`.
   - `input_policy='strict'` means no coercion is allowed under the chosen
     contract,
   - `input_policy='coerce'` means coercion is allowed under the chosen
     contract,
   - `ownership_policy='borrow'` means incoming material objects are not copied,
   - none of those concepts alone imply end-to-end zero-copy execution.
5. The implemented public API name for resident-mode control is
   `resident_mode`.
   - `resident_mode='host'` keeps authoritative morphology fields on CPU,
   - `resident_mode='device'` keeps authoritative morphology fields on GPU
     when the backend supports that mode.

### Optimization campaign strategy going forward

1. The optimization goal is to make `cupy-rsoxs` materially faster while
   preserving physics parity through the maintained test suite.
2. The default inner-loop timing command is now:
   - `/home/deand/mambaforge/envs/nrss-dev/bin/python tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py --label <label> --size-labels small --resident-modes host --timing-segments all`
3. Future optimization campaigns should focus on one segment at a time using
   the segmentation defined in "Timing apparatus status" above rather than
   trying to optimize the whole backend at once.
4. The default optimization benchmark ladder should use single-energy cases
   rather than full-energy runs.
   - energy iteration mostly wraps the same loop structure,
   - for non-parity tuning, full-energy runs mainly add runtime and statistics
     rather than exposing a fundamentally different algorithmic path.
5. Recommended default tuning ladder:
   - primary latency lane: `small`, single energy, `EAngleRotation=[0, 0, 0]`,
     `resident_mode='host'`, with morphology fields generated directly as the
     host-resident authoritative NumPy contract before timing starts,
   - secondary latency lane: `medium`, single energy, `EAngleRotation=[0, 0, 0]`,
     `resident_mode='host'`, with the same direct NumPy contract generation,
   - device-regression lane: selected `small` or `medium`, single energy,
     `EAngleRotation=[0, 0, 0]`, `resident_mode='device'`, with morphology
     fields preconverted to the CuPy contract and the default stream
     synchronized before the timer starts,
   - rotation-focused lane: use opt-in `--include-triple-limited` on either or
     both resident-mode variants when the built-in
     `EAngleRotation=[0, 15, 165]` checkpoint is sufficient, or use
     `--rotation-specs start:increment:end[, ...]` when the exact rotation set
     is the point of the measurement,
   - energy-focused lane: use `--no-rotation-energy-counts` for centered
     contiguous CoreShell subsets, or `--energy-lists 'E1|E2|...'[, ...]` when
     an explicit energy list is the point of the measurement,
   - memory / throughput guardrail lane: `large`, single energy, selected
     resident mode, dense angle sweep when needed.
6. When chasing a hotspot, narrow `--timing-segments` to the relevant subset
   instead of recording everything on every run.
   - Segment `A` is nominally complete for the common workflow:
     - `A1` constructor overhead is already minor,
     - the accepted host-resident `A2` staging improvement is in place,
     - the device-resident lane is an intentionally faster alternate use case
       and should remain visibly faster when morphology is already on device,
   - current evidence makes Segments `B` and `D` the default focal points,
   - re-expand to `all` when checking for regression spillover into neighboring
     stages.
7. Do not add `cyrsoxs` timing lanes to the default optimization harness.
   `cyrsoxs` timing is not part of the current optimization loop.
8. Use `--include-triple-limited` only when rotation-sensitive behavior needs a
   checkpoint outside the default no-rotation loop.
9. Use `--include-full-small-check` only as an occasional expensive checkpoint
   for the full rotation path.
10. If a single-energy lane is too fast or too noisy to rank consistently,
   repeat it enough times to stabilize the comparison rather than escalating
   directly to full-energy runs.
11. Full-energy runs should be reserved for:
    - milestone confirmation after accepted optimization changes,
    - final comparison graphics if they are later needed,
    - and maintained parity / correctness rechecks.
12. Future optimization guidance should be treated as open-ended. Many more
   opportunities likely exist beyond the ones already enumerated or tried. The
   document should not be read as an exhaustive list of remaining ideas.

### Current Segment B/D campaign plan (March 24, 2026)

This subsection is the live plan and outcome ledger for the current Segment `B`
/ `D` optimization block. Update it as each step completes so a fresh context
can recover both the intended order and the measured outcomes.

Status key:
- `planned`: not started yet,
- `in_progress`: active implementation or measurement work,
- `completed`: implemented and measured,
- `rejected`: attempted and not kept.

Current campaign steps:

1. `plan01_document_campaign` - `completed`
   - write the full Segment `B` / `D` plan into this roadmap and the dev timing
     README before code changes,
   - define the execution-path terminology and the benchmark matrix to be used
     for the campaign,
   - outcome:
     - completed on March 24, 2026 by adding this live campaign subsection and
       the matching dev-harness README notes.
2. `plan02_execution_path_surface` - `completed`
   - revive `backend_options` as the backend-specific runtime-behavior surface
     for `cupy-rsoxs`,
   - add an explicit `execution_path` option with default
     `execution_path='tensor_coeff'`,
   - initial intended values:
     - `tensor_coeff`: current accepted `Nt -> FFT(Nt) -> projection-coefficient`
       route,
     - `direct_polarization`: CyRSoXS `AlgorithmType=0`
       communication-minimizing analog,
     - `nt_polarization`: CyRSoXS `AlgorithmType=1`
       memory-minimizing analog,
   - outcome:
     - completed on March 24, 2026,
     - `cupy-rsoxs` now accepts `backend_options["execution_path"]` with
       supported values `tensor_coeff`, `direct_polarization`, and
       `nt_polarization`,
     - aliases now normalize as:
       - `default -> tensor_coeff`,
       - `tensor -> tensor_coeff`,
       - `direct -> direct_polarization`,
       - `nt -> nt_polarization`,
     - default behavior remains unchanged at `execution_path='tensor_coeff'`.
3. `plan03_core_shell_backend_options_plumbing` - `completed`
   - thread `backend_options` through the maintained CoreShell construction and
     backend-run helpers so execution-path validation can use the official
     maintained morphology path with reasonable defaults,
   - outcome:
     - completed on March 24, 2026,
     - maintained CoreShell helpers now accept `backend_options` for both
       morphology construction and backend execution so dormant-path validation
       can use the official maintained morphology lane rather than ad hoc
       builders.
4. `plan04_harness_execution_path_matrix` - `completed`
   - extend the authoritative subprocess timing harness to accept
     execution-path-specific sweeps,
   - ensure labels and summaries include both primary timing and
     segment-by-segment timing per execution path,
   - keep segment tracking available for all execution paths using the existing
     `A1,A2,B,C,D,E,F` segmentation,
   - outcome:
     - completed on March 24, 2026,
     - the authoritative subprocess harness now accepts
       `--execution-paths ...`,
     - the same harness now also accepts explicit
       `--rotation-specs start:increment:end[, ...]` and
       `--energy-lists 'E1|E2|...'[, ...]` inputs for targeted
       rotation-sensitive or energy-sensitive studies,
     - benchmark labels now carry the execution-path suffix,
     - requested and resolved backend options are persisted into the summary
       artifact for each case,
     - segment tracking remains available across all surfaced execution paths,
       even though the exact work contained inside Segments `B/C/D/E` differs
       by execution path,
     - when both explicit rotation and explicit energy lists are supplied, the
       harness emits combined cases as well as the rotation-only and
       energy-only variants,
     - focused verification artifact:
       `test-reports/cupy-rsoxs-optimization-dev/harness_rotation_energy_smoke_20260324/summary.json`
       confirmed baseline, rotation-only, energy-only, and combined case
       emission on both host and device resident modes.
5. `plan05_execution_path_baselines` - `in_progress`
   - benchmark the current `tensor_coeff` path plus the two dormant paths on
     the official no-rotation host/device small lanes before new math changes,
   - run CoreShell correctness validation for the newly surfaced dormant paths
     before treating their timings as optimization guidance,
   - current measured timing baseline from
     `execution_path_surface_smoke_20260324`:
     - host / `tensor_coeff`:
       `primary 2.833s`, `A1 0.003`, `A2 2.482`, `B 0.139`, `C 0.010`,
       `D 0.072`, `E 0.112`, `F 0.001`,
     - host / `direct_polarization`:
       `primary 2.609s`, `A1 0.003`, `A2 2.401`, `B 0.143`, `C 0.009`,
       `D 0.033`, `E 0.002`, `F 0.001`,
     - host / `nt_polarization`:
       `primary 2.607s`, `A1 0.003`, `A2 2.408`, `B 0.130`, `C 0.016`,
       `D 0.034`, `E 0.002`, `F 0.000`,
     - device / `tensor_coeff`:
       `primary 0.237s`, `A1 0.003`, `A2 0.000`, `B 0.131`, `C 0.010`,
       `D 0.072`, `E 0.004`, `F 0.000`,
     - device / `direct_polarization`:
       `primary 0.204s`, `A1 0.003`, `A2 0.000`, `B 0.138`, `C 0.009`,
       `D 0.033`, `E 0.002`, `F 0.000`,
     - device / `nt_polarization`:
       `primary 0.205s`, `A1 0.003`, `A2 0.000`, `B 0.134`, `C 0.015`,
       `D 0.033`, `E 0.002`, `F 0.000`,
   - current interpretation:
     - both surfaced dormant routes are materially faster than
       `tensor_coeff` on the small single-energy no-rotation lane,
     - their speed advantage is driven mostly by much smaller Segment `D` and
       Segment `E` work rather than by a clear Segment `B` win,
     - however, a later limited multi-angle comparison on the same small
       CoreShell model showed the ranking reverses once angle iteration
       matters:
       - source artifact:
         `test-reports/cupy-rsoxs-optimization-dev/execution_path_multiangle_5_vs_15_20260324/summary.json`,
       - host / `EAngleRotation=[0, 15, 165]`:
         `tensor_coeff 2.742s`, `nt_polarization 3.319s`,
         `direct_polarization 3.742s`,
       - host / `EAngleRotation=[0, 5, 165]`:
         `tensor_coeff 2.921s`, `nt_polarization 4.376s`,
         `direct_polarization 4.524s`,
       - device / `EAngleRotation=[0, 15, 165]`:
         `tensor_coeff 0.419s`, `nt_polarization 0.441s`,
         `direct_polarization 0.556s`,
       - device / `EAngleRotation=[0, 5, 165]`:
         `tensor_coeff 0.412s`, `nt_polarization 0.926s`,
         `direct_polarization 1.559s`,
       - interpretation:
         for angle-heavy workloads, `tensor_coeff` is the practical winner and
         should remain the primary optimization target even though the dormant
         routes are still useful no-rotation reference points,
     - quick raw CoreShell comparisons against `tensor_coeff` on the official
       maintained morphology path show the dormant routes are numerically very
       close overall but not bitwise-identical:
        `max_abs 0.078125`, `rmse 1.60801e-4`, `p95_abs 3.8147e-06`,
      - after the accepted plan06/plan07 steps, the same official maintained
        morphology raw comparison remained close:
        `max_abs 0.046875`, `rmse 9.72605e-05`, `p95_abs 3.8147e-06`,
      - the maintained official sim-regression test for the accepted default
        `cupy-rsoxs` path passed on March 24, 2026:
        `pytest tests/validation/test_core_shell_reference.py -k "sim_regression_cupy_borrow_strict" --nrss-backend cupy-rsoxs -v`,
      - full dormant-path A-wedge / sim-reference validation was attempted on
        the official CoreShell helper path but remained too expensive for the
        current inner-loop campaign, so dormant-path timing guidance should
        still be treated as timing evidence plus raw-maintained-morphology
        similarity rather than as fully closed reference validation.
6. `plan06_isotropic_material_fast_path` - `completed`
   - add a full-material isotropic fast path for materials with
     `S == 0` everywhere,
   - skip Euler decode and off-diagonal tensor work for those materials across
     all execution paths,
   - expected to help common vacuum / matrix / isotropic-additive cases,
   - outcome from `plan06_isotropic_material_fast_path_clean_20260324`:
     - completed on March 24, 2026,
     - exact-zero isotropic materials now:
       - skip runtime staging of `S`, `theta`, and `psi` in the host-resident
         path,
       - skip Euler decode and off-diagonal tensor work in Segment `B` across
         all execution paths,
     - focused smoke checks added:
       - host-resident isotropic staging now confirms only `Vfrac` is staged to
         CuPy for fully isotropic materials,
       - all three execution paths now agree on a fully isotropic synthetic
         morphology in the maintained smoke lane,
     - measured timing deltas versus the execution-path baseline on the
       official small single-energy no-rotation CoreShell lane:
       - host / `tensor_coeff`:
         `primary 2.833s -> 2.519s`, `A2 2.482 -> 2.305`, `B 0.139 -> 0.110`,
       - host / `direct_polarization`:
         `primary 2.609s -> 2.613s`, `B 0.143 -> 0.112`,
       - host / `nt_polarization`:
         `primary 2.607s -> 2.706s`, `B 0.130 -> 0.113`,
       - device / `tensor_coeff`:
         `primary 0.237s -> 0.231s`, `A2 0.000 -> 0.112`, `B 0.131 -> 0.014`,
       - device / `direct_polarization`:
         `primary 0.204s -> 0.210s`, `A2 0.000 -> 0.109`, `B 0.138 -> 0.015`,
       - device / `nt_polarization`:
         `primary 0.205s -> 0.201s`, `A2 0.000 -> 0.116`, `B 0.134 -> 0.014`,
     - interpretation:
       - the CoreShell lane confirms the intended Segment `B` win strongly,
         especially in the device-resident path where two of the three
         materials are fully isotropic,
       - total latency impact is mixed because exact-zero isotropic detection is
         currently performed during runtime staging for device-resident inputs,
         which moves about `0.11s` into Segment `A2` on this small benchmark,
       - the host-default `tensor_coeff` lane still improves materially on the
         primary wall metric and remains worth keeping,
       - the current implementation is therefore accepted as the baseline for
         the next step, with the device-path `A2` caveat recorded rather than
         treated as a blocker,
       - this inferred exact-zero path was later superseded by
         `plan12_explicit_isotropic_contract`, which removed the runtime
         all-zero scan entirely and attached the isotropic fast path to an
         explicit material contract instead.
7. `plan07_axis_family_fast_path` - `completed`
   - add the high-value axis-family special cases for electric-field rotations
     congruent to `0°/180°` and `90°/270°`,
   - first target the single-angle / fully axis-aligned cases where the
     savings reach beyond Segment `E`,
   - note the analogous downstream pruning opportunity in Segment `D`:
     axis-aligned cases can avoid `proj_xy` and one projection-family branch,
   - outcome from `plan07_axis_family_fast_path_clean_20260324`:
     - completed on March 24, 2026,
     - fully axis-aligned angle sets now use an explicit angle-family plan:
       - `0°/180°` use the x-family subset,
       - `90°/270°` use the y-family subset,
       - general angles continue to use the existing full path,
     - implementation details:
       - `tensor_coeff` now skips `proj_xy` and the unused x/y projection
         family on fully axis-aligned angle sets,
       - `nt_polarization` now computes only the `Nt` component subset needed
         by the aligned angle family,
       - `direct_polarization` now uses the aligned-field specialization rather
         than the general `mx*sx + my*sy` branch,
       - identity rotations now bypass the affine-transform path in Segment
         `E`,
     - focused smoke/regression checks remained green, including the existing
       endpoint-semantics smoke and the execution-path smoke subset,
     - measured deltas versus the post-plan06 baseline on the official small
       single-energy no-rotation CoreShell lane:
       - host / `tensor_coeff`:
         `D 0.073 -> 0.035`, `E 0.004 -> 0.002`,
       - host / `direct_polarization`:
         `primary 2.613s -> 2.530s`, `E 0.002 -> 0.000`,
       - host / `nt_polarization`:
         `primary 2.706s -> 2.537s`, `D 0.040 -> 0.033`,
         `E 0.002 -> 0.000`,
       - device / `tensor_coeff`:
         `primary 0.231s -> 0.196s`, `D 0.072 -> 0.035`,
         `E 0.004 -> 0.002`,
       - device / `direct_polarization`:
         `primary 0.210s -> 0.198s`, `D 0.051 -> 0.033`,
         `E 0.002 -> 0.000`,
       - device / `nt_polarization`:
         `primary 0.201s -> 0.193s`, `E 0.002 -> 0.000`,
     - interpretation:
       - the aligned-angle specialization is a clear `D`/`E` win on the
         default `0°` lane and produces the intended overall device-lane speedup,
       - host-lane `A2` variance still makes wall-clock interpretation noisier
         than the backend-segment metrics, so the step should be read primarily
         as a segment-specific improvement rather than as a universal primary
         win on every surfaced path,
       - an isolated `90°` spot check showed the same low-`D` / near-zero-`E`
         shape for the y-family branch, so the optimization is not only
         exercising the `0°` case.
8. `plan08_segment_b_algebraic_rewrite` - `rejected`
   - simplify Segment `B` tensor assembly with shared-term factoring, smaller
     scratch reuse, and more aggressive dead-intermediate deletion,
   - this is explicitly not the abandoned multi-energy cache idea:
     it should reduce temporary live-set pressure rather than persist large
     per-energy/per-material tensors,
   - outcome from `plan08_segment_b_algebraic_rewrite_20260324`:
     - attempted on March 24, 2026 with a fixed-size scratch-buffer
       formulation for Segment `B`,
     - this implementation did *not* recreate the abandoned cache-memory risk:
       it used only per-call scratch buffers rather than any persistent
       per-energy/per-material cache,
     - however it materially regressed the default `tensor_coeff` Segment `B`
       lane and therefore did not clear the acceptance bar,
     - most important measured regression versus the post-plan07 baseline:
       - host / `tensor_coeff`:
         `B 0.107 -> 0.192` and `primary 2.690s -> 2.619s` still failed the
         Segment `B` objective because the targeted segment became
         substantially slower,
     - other surfaced paths were mixed to nearly flat rather than strongly
       improved,
     - the implementation was reverted and the accepted baseline therefore
       remains the post-plan07 axis-family state.
9. `plan09_rebenchmark_and_regression_check` - `completed`
   - rerun the official subprocess timing matrix after each accepted step,
   - use the maintained smoke/reference checks as regression guards before the
     campaign is closed out,
   - outcome from `plan09_final_rebenchmark_accepted_state_20260324`:
     - completed on March 24, 2026,
     - final accepted state is the combination of:
       - execution-path surfacing,
       - full-material isotropic fast path,
       - axis-family fast path,
       - with the plan08 scratch experiment and the plan11
         `ElementwiseKernel` experiment both reverted,
     - final accepted-state timing snapshot on the official small
       single-energy no-rotation CoreShell lane:
       - host / `tensor_coeff`:
         `primary 2.515s`, `A2 2.339`, `B 0.109`, `D 0.035`, `E 0.002`,
       - host / `direct_polarization`:
         `primary 2.541s`, `A2 2.364`, `B 0.113`, `D 0.034`, `E 0.000`,
       - host / `nt_polarization`:
         `primary 2.519s`, `A2 2.344`, `B 0.108`, `D 0.033`, `E 0.000`,
       - device / `tensor_coeff`:
         `primary 0.205s`, `A2 0.122`, `B 0.010`, `D 0.039`, `E 0.002`,
       - device / `direct_polarization`:
         `primary 0.198s`, `A2 0.116`, `B 0.014`, `D 0.034`, `E 0.000`,
       - device / `nt_polarization`:
         `primary 0.207s`, `A2 0.115`, `B 0.010`, `D 0.035`, `E 0.001`,
     - maintained regression checks completed in this close-out pass:
       - focused smoke subset: `9 passed`,
       - accepted-path maintained CoreShell sim regression:
         `1 passed` with `--nrss-backend cupy-rsoxs`,
     - final interpretation:
       - the accepted state materially improves the default host
         `tensor_coeff` lane versus the original execution-path baseline,
         mostly through lower `A2`, `B`, `D`, and `E`,
       - the device lane keeps the strong Segment `B` and aligned-angle
         `D/E` wins, with the known caveat that isotropic detection now lands
         in `A2`.
10. `plan10_float16_followup_scaffold` - `completed`
    - do not implement the mixed-precision campaign in this block,
    - only leave the execution-path surface and roadmap notes in a form that
      allows an orthogonal future backend-options extension for reduced
      storage/transfer precision,
    - outcome:
      - completed on March 24, 2026 with no float16 compute-path work landed,
      - `backend_options["execution_path"]` and the accompanying roadmap notes
        now provide the intended light scaffold for a future orthogonal reduced
        precision option surface,
      - the existing float16-rejection smoke continues to pass in this state.
11. `plan11_elementwise_kernel_experiment` - `rejected`
    - try one last `ElementwiseKernel` implementation aimed at Segment `B`,
    - scope it narrowly to the aligned-angle/default-lane path first rather
      than broadening it across every route immediately,
    - accept it only if the default-lane gain is large enough to justify the
      added maintenance burden; otherwise reject and revert it,
    - outcome from `plan11_elementwise_kernel_experiment_20260324`:
      - attempted on March 24, 2026 by replacing the aligned-angle `Nt`
        subset path with a narrow `ElementwiseKernel` implementation,
      - this did reduce device-resident aligned-angle `B` time for the
        `tensor_coeff` / `nt_polarization` subset:
        - device / `tensor_coeff`: `B 0.010 -> 0.005`,
        - device / `nt_polarization`: `B 0.010 -> 0.005`,
      - but it materially regressed the default host-resident
        `tensor_coeff` path:
        - host / `tensor_coeff`: `B 0.107 -> 0.246`,
      - because the default host lane remains the primary public-workflow
        authority and the gain was not broad enough to offset the extra kernel
        maintenance burden, the experiment was rejected and reverted.
12. `plan12_explicit_isotropic_contract` - `completed`
    - replace inferred exact-zero isotropic detection with an explicit
      enum-backed full-material contract,
    - accept `SFieldMode.ISOTROPIC` for fully isotropic materials only,
    - expect `theta` and `psi` to be `None` for that contract and ignore them
      with warning if the caller still supplies arrays,
    - remove the Segment `A2` all-zero scan entirely so the optimization is
      attached only to the explicit contract,
    - keep legacy surfaces working by synthesizing concrete zero `S`, `theta`,
      and `psi` arrays only at the legacy boundary,
    - extend the live optimization harness with an isotropic-representation
      comparison option while preserving a frozen pre-change harness snapshot,
    - outcome from `plan12_explicit_isotropic_contract_20260324`:
      - completed on March 24, 2026,
      - `Material.S` now accepts the boolean-backed enum
        `SFieldMode.ISOTROPIC` as the explicit full-material isotropic
        contract,
      - explicit isotropic materials now validate against `Vfrac`,
        optical constants, and shape consistency without requiring concrete
        `S/theta/psi` arrays,
      - named `vacuum` materials now always resolve to the explicit isotropic
        contract and ignore any supplied `S/theta/psi` fields with warning,
      - non-`None` `theta` / `psi` values under that contract are warned about
        and normalized away before runtime use,
      - `cupy-rsoxs` Segment `A2` now keys isotropic staging only from the
        explicit contract and performs no inferred all-zero scan in either
        resident mode,
      - legacy zero-array isotropic inputs remain supported but no longer
        receive inferred isotropic optimization,
      - CyRSoXS voxel handoff, HDF5 export, and visualization synthesize
        concrete zero `S/theta/psi` arrays only when those legacy surfaces need
        them,
      - focused smoke coverage now includes explicit-contract validation,
        warning behavior, host/device staging behavior, representation parity,
        and legacy-boundary materialization,
      - the live optimization harness now supports
        `--isotropic-material-representation {legacy_zero_array, enum_contract, both}`
        and emits paired `primary` / segment comparison entries in
        `summary.json` when both representations are requested,
      - that isotropic comparison surface now composes directly with
        `--cuda-prewarm {off, before_prepare_inputs}` so the contract can be
        measured in both cold-process and steady-state host modes,
      - standard small CoreShell no-rotation measurements after the explicit
        contract plus named-vacuum cleanup showed:
        - host / `tensor_coeff`: `primary 2.534s -> 2.503s`,
          `A2 2.352 -> 2.326`, `B 0.119 -> 0.111`,
        - device / `tensor_coeff`: `primary 0.182s -> 0.180s`,
          `A2 0.000154 -> 0.000131`, `B 0.116 -> 0.111`,
        and the pre-change harness snapshot is kept as
        `run_cupy_rsoxs_optimization_matrix_legacy_pre_isotropic_contract.py`.
13. `plan13_dev_harness_cuda_context_prewarm` - `completed`
    - extend only the live dev optimization harness with optional in-process
      CUDA context prewarm,
    - keep default `--cuda-prewarm off` as the current cold subprocess
      behavior,
    - add `before_prepare_inputs` so the worker can perform a tiny untimed
      host->CuPy touch before `_prepare_core_shell_case_inputs(...)`,
    - leave allocator/pool refresh behavior unchanged,
    - record both requested and applied prewarm modes in per-case results and
      isotropic comparison summaries,
    - treat device-resident prewarm as explicitly redundant rather than as a
      separate timing lane,
    - outcome from `plan13_dev_harness_cuda_context_prewarm_20260324`:
      - completed on March 24, 2026,
      - the live harness now accepts
        `--cuda-prewarm {off, before_prepare_inputs}`,
      - host cases record `cuda_prewarm_seconds` outside the primary timing
        boundary when `before_prepare_inputs` is selected,
      - device-resident cases record
        `cuda_prewarm_applied_mode='redundant_device_prepare'`,
      - focused smoke coverage now includes parser/summary checks for the new
        option,
      - small CoreShell host / `tensor_coeff` totals now show:
        - cold / `legacy_zero_array`: `primary 2.546s`,
        - cold / `enum_contract`: `primary 2.545s`,
        - prewarmed / `legacy_zero_array`: `primary 0.282s`,
        - prewarmed / `enum_contract`: `primary 0.226s`,
      - interpretation:
        - cold host timings remain dominated by first-touch CUDA/CuPy bring-up
          and are not the right authority for steady-state host staging,
        - the prewarmed host lane is the better model for many-morph
          single-process workflows,
        - in that prewarmed host lane, the explicit enum contract improves
          primary time by about `19.8%` versus legacy zero arrays and cuts
          `A2` by about `48.0%`.

Precision and option-surface notes for this campaign:

1. `backend_options` is being resurrected as the explicit backend-specific
   runtime-behavior surface rather than overloading `AlgorithmType` directly in
   `cupy-rsoxs`.
2. The float16 plan is orthogonal to `execution_path`.
   - intended future direction:
     - reduced storage / host->device transfer precision as the first target,
     - optional Segment `B` low-precision experiment if validation permits,
     - cast to `float32` / `complex64` before FFT ingress for parity-sensitive
       math.
3. No float16 compute-path implementation should be accepted in this campaign
   beyond the light option-surface scaffolding needed so a later block can add
   it cleanly.
4. Segment `D` should keep a note for future pruning and scratch-reuse work:
   - axis-family cases may avoid building `proj_xy` and one of the x/y
     projection families,
   - algebraic factoring and scratch reuse around basis/projection assembly may
     be worthwhile there after the Segment `B` pass is measured.

### Campaign ledger from work already done

1. Current accepted optimization state:
   - `opt01_drop_nt5`
     - removed dead `Nt[5]` from the supported Euler-only / `PARTIAL` path,
     - improved run time by about `6.8%` to `32.3%` versus the first matrix
       baseline,
     - improved large limited-rotation free memory after run from about
       `0.66 GiB` to `4.04 GiB`.
   - `opt02_fft_reuse`
     - reused `nt` storage for FFT output and removed the separate `fft_nt`
       live set,
     - improved run time by an additional `0.8%` to `6.7%` versus `opt01`,
     - improved large limited-rotation free memory after run from about
       `4.04 GiB` to `6.99 GiB`,
     - reduced large limited-rotation pool total from about `43.24 GiB` to
       `40.29 GiB`.
   - `opt04_igor_kernel`
     - replaced the advanced-indexing Igor reorder with a cached `RawKernel`,
     - improved run time by an additional `1.1%` to `5.6%` versus `opt03`,
     - improved run time by about `12.8%` to `36.4%` versus the original matrix
       baseline,
     - preserved the ad hoc validation metrics already seen in the baseline
       run.
   - `opt08_segment_a_split`
     - split historical Segment `A` into Segment `A1` constructor timing and
       Segment `A2` runtime staging timing,
     - made host-resident transfer cost separately measurable from
       `Morphology(...)` construction,
     - changed the accepted measurement basis for resumed work so Segment `A`
       no longer hides transfer and constructor behavior inside one number.
   - `opt09_stage_empty_set`
     - replaced host-resident NumPy -> CuPy staging from `cp.asarray(...)`
       with `cp.empty(...); out.set(host)`,
     - improved the small single-energy host lane by about `7.5%` on primary
       wall time and about `7.0%` on Segment `A2` versus the split-only
       baseline,
     - device-resident Segment `A2` remains effectively zero when morphology
       fields are already on device before `run()` begins,
     - this makes Segment `A` nominally complete for the common workflow, with
       future default tuning focus shifted to Segments `B` and `D`.
2. Explored but not accepted in the first campaign:
   - `opt03_prealloc_result_scratch`
     - mixed result: about `-2.3%` to `+0.4%` versus `opt02`,
     - no meaningful memory-headroom improvement in the measured matrix,
     - rejected and reverted.
   - `opt05_proj_coeff_fuse`
     - preserved validation, but regressed run time by about `5.7%` to `38.6%`
       versus `opt04`,
     - rejected and reverted.
   - `opt06_stream_projection`
     - preserved validation, but regressed run time heavily versus the accepted
       state and did not produce a practical memory-headroom win in this
       matrix,
     - rejected and reverted.
   - `opt07_texture_affine`
     - implemented a `cupyx.scipy.ndimage.affine_transform(...,
       texture_memory=True)` candidate using a homogeneous transform in texture
       mode,
     - full-matrix result was effectively flat to slightly worse:
       about `+0.07%` to `+1.9%` versus `opt04`,
     - large-case memory footprint was unchanged relative to `opt04`,
     - ad hoc validation remained within the same tolerance band,
     - rejected and reverted because it did not clear the significant-gain
       bar.
3. Interpretation rule for the ledger:
   - an accepted item has demonstrated a material win under measured conditions
     and remains part of the current baseline,
   - a rejected item means that specific implementation did not clear the bar,
     not that the whole idea class is permanently closed.

### Explicit experiments and deferred directions

1. `float16` and mixed-precision work remain deferred until after the next
   speed campaigns; parity-sensitive compute remains `float32/complex64`.
   - near-term intended shape of that work:
     - `backend_options` should carry reduced-precision storage/runtime flags
       orthogonally to `execution_path`,
     - the first target is host/device storage and transfer reduction rather
       than end-to-end low-precision FFT/projection math,
     - the expected precision ladder is reduced-precision storage or Segment
       `B` staging followed by promotion to `float32` / `complex64` before FFT
       ingress.
2. Reduced angle sampling, alternate interpolation rules, and multi-GPU fan-out
   remain outside the current exact-tuning track.
3. Host-resident staged mode creates room for explicit experiments with deeper
   CPU-side precompute, including testing whether some early field math is
   faster on CPU before transfer.
4. Deeper CPU-side precompute should be treated as an explicit experiment, not
   as the default plan.
   - it may help some host-resident workflows,
   - but it may also become transfer-dominated if large per-energy
     intermediates are moved to GPU,
   - therefore it must be measured before being promoted into recommended
     architecture.
5. Export timing / host-conversion timing remains intentionally deferred as
   Segment `G`.
6. Per-stage memory instrumentation remains deferred; the existing coarse peak
   memory monitoring is enough for the current pass.
7. Host-resident repeated-run staging reuse remains explicitly deferred as a
   low-priority niche possibility.
   - examples include registered-host buffers, reusable staged device mirrors,
     or other host-lane transfer caches,
   - this repo does not expect many repeated-run workflows to justify making
     that the default direction,
   - if a workflow benefits from persistent GPU morphology residency, prefer
     `resident_mode='device'` rather than adding host-resident caching by
     default.
8. Resume rule for a fresh optimization context:
   - start from the current accepted backend state in this repo, including the
     Segment `A1/A2` split and the accepted host-resident `A2` staging fast
     path,
   - use `tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py`
     as the default inner-loop harness,
   - begin with the primary lane:
     `/home/deand/mambaforge/envs/nrss-dev/bin/python tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py --label <label> --size-labels small --resident-modes host --timing-segments all`,
   - narrow `--timing-segments` when focusing on a specific segment,
   - use the single-energy benchmark ladder for inner-loop tuning,
   - for explicit rotation-sensitive or energy-sensitive studies, the harness
     now supports `--rotation-specs` and `--energy-lists`, and it emits
     combined cases when both are supplied,
   - recheck `--resident-modes device` periodically as a regression lane for
     direct-CuPy workflows,
   - add `--include-triple-limited` only when the fixed
     `EAngleRotation=[0, 15, 165]` checkpoint is sufficient for the question,
   - otherwise use `--rotation-specs` and `--energy-lists` to benchmark the
     exact angle or energy sets under discussion,
   - rerun maintained parity checks after promising optimization changes,
   - use
     `tests/validation/dev/core_shell_backend_performance/run_primary_backend_speed_comparison.py`
     when a principal cross-backend primary-time snapshot is needed,
   - treat the legacy full-energy backend-comparison harness as optional
     historical context rather than a required step in the default loop.
