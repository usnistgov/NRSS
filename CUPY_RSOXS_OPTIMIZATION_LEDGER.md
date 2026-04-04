# `cupy-rsoxs` Optimization Ledger

This document is the authoritative speed-optimization ledger for `cupy-rsoxs`. It records timing methodology, accepted and rejected experiments, current benchmark authority, and deferred optimization directions.

Stable backend contract and phase-1 behavior live in `CUPY_RSOXS_BACKEND_SPEC.md`. Repo-wide upgrade planning and test-program status live in `REPO_UPGRADE_PLAN.md`. Path-specific `direct_polarization` optimization notes now live in `CUPY_RSOXS_DIRECT_POLARIZATION_OPTIMIZATION.md`.

Current supported `cupy-rsoxs` execution paths are `tensor_coeff` and
`direct_polarization`. Historical `nt_polarization` references below are kept
as archival benchmark context from the earlier three-path optimization campaign.

## Current Optimization Guidance And Ledger

This section replaces the earlier optimization-inventory framing with a more
resumable guidance-and-ledger record. It is intended to let a fresh context
recover:
- the current accepted backend state,
- the measurement caveats,
- the official resident-mode guidance,
- the optimization campaign strategy,
- and the work that has already been tried, accepted, or rejected.

### Path-specific companion notes

1. This ledger remains the authoritative backend-wide optimization record.
2. When one execution path becomes a distinct optimization thread with its own
   timing caveats and priorities, keep the accepted backend-wide conclusions
   here and track the path-specific campaign in a companion note.
3. Current companion note:
   - `CUPY_RSOXS_DIRECT_POLARIZATION_OPTIMIZATION.md`
     - current authoritative `direct_polarization` timing surface,
     - host-resident peak-memory context,
     - historical execution-path benchmark context transferred from the
      March 24 campaign,
     - and the current ranked direct-path speed priorities.
4. Latest backend-wide direct-path cross-reference:
   - April 3, 2026 accepted a fused custom-kernel Segment `B` rewrite for
     `execution_path='direct_polarization'`
   - accepted lane:
     - small host-prewarmed `EAngleRotation=[0, 5, 165]`
   - accepted evidence lives in:
     - `CUPY_RSOXS_DIRECT_POLARIZATION_OPTIMIZATION.md`
   - interpretation:
     - this materially improves the maintained multi-angle direct-path timing
       surface while preserving the low-memory identity of that path,
     - but it does not replace `tensor_coeff` as the maintained default
       execution path

### Current state and authoritative artifacts

1. The current accepted tuned backend state is the behavior reached after the
   first optimization campaign plus the accepted March 25 Segment `D`
   continuation:
   - dead `Nt[5]` removed from the supported Euler-only / `PARTIAL` path,
   - FFT storage reused so separate `fft_nt` residency is reduced,
   - Igor reorder implemented with a cached `RawKernel`,
   - detector / projection geometry cached in backend runtime state,
   - and `tensor_coeff` projection-family evaluation now uses detector-grid
     helpers for both general-angle and aligned `x` / `y` families instead of
     materializing extra `scatter3d` volumes in that wrapper.
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
9. Latest principal cross-backend primary-time snapshot for the accepted March
   25 Segment `D` end state:
   - artifact root:
     `test-reports/core-shell-backend-performance-dev/principal_cross_backend_20260325_planD02b/`
   - panel scope:
     - single-energy CoreShell only,
     - sizes `small`, `medium`, `large`,
     - rotations `[0, 0, 0]` and `[0, 15, 165]`,
     - legacy `cyrsoxs` cold and pre-warm,
     - `cupy-rsoxs` host cold and pre-warm,
     - `cupy-rsoxs` device compared against legacy pre-warm rows,
   - headline results:
     - host speedup versus legacy ranged from about `1.9x` to `4.5x`,
     - device speedup versus legacy pre-warm ranged from about `2.6x` on
       `small` to about `13.3x` on `large`,
     - medium/large device rows now sit around `9.2x-13.3x` faster than the
       matching legacy pre-warm rows,
   - interpretation:
     - the March 25 Segment `D` end state materially improves the already-fast
       `cupy-rsoxs` device lane and preserves a clear overall cross-backend
       speed advantage over legacy `cyrsoxs`.

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
11. Path-specific follow-up note:
   - `execution_path='direct_polarization'` is now tracked in
     `CUPY_RSOXS_DIRECT_POLARIZATION_OPTIMIZATION.md`,
   - use that companion note for the current authoritative direct-path timing
     surface, current memory observations, historical execution-path context,
     and ranked direct-path experiment list.
12. Maintained parity remains a post-optimization check through the test suite
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

### Memory concerns and March 26, 2026 host-resident comparison probe

1. Recent user feedback raised a different question from the post-run
   allocator-retention issue:
   - the main concern is sustained GPU usage during long runs in shared-GPU
     environments,
   - the relevant comparison is in-run and mid-run device pressure rather than
     only post-run cleanup.
2. Comparison rule for memory claims:
   - when comparing `cupy-rsoxs` against legacy `cyrsoxs` for GPU-memory
     footprint, use `resident_mode='host'` for `cupy-rsoxs`,
   - this is the directly comparable lane because both workflows begin from
     host-resident authoritative morphology and stage GPU data only for
     compute,
   - do not treat `resident_mode='device'` as the fairness authority for
     cross-backend memory claims, because that lane intentionally keeps the
     authoritative morphology arrays on GPU and therefore models a different
     workflow class.
3. Ad hoc comparison probe run on March 26, 2026:
   - host machine / device lane:
     - `CUDA_VISIBLE_DEVICES=0`,
     - GPU: `Quadro RTX 8000`,
   - morphology source:
     - maintained CoreShell helper
       `tests.validation.lib.core_shell.build_core_shell_morphology(...)`,
     - `scenario='baseline'`,
     - shape `SHAPE = (32, 512, 512)`,
     - `create_cy_object=False`,
   - run conditions:
     - single energy set to the midpoint CoreShell energy (`285.0 eV`),
     - `EAngleRotation=[0.0, 0.0, 0.0]`,
     - `return_xarray=False`,
     - `WindowingType=0`,
   - legacy lane:
     - `backend='cyrsoxs'`,
     - authoritative NumPy fields,
   - host-resident CuPy lane:
     - `backend='cupy-rsoxs'`,
     - `resident_mode='host'`,
     - `input_policy='coerce'`,
     - `ownership_policy='borrow'`,
     - `field_namespace='numpy'`,
   - non-comparable device-resident context lane:
     - `backend='cupy-rsoxs'`,
     - `resident_mode='device'`,
     - `input_policy='strict'`,
     - `ownership_policy='borrow'`,
     - `field_namespace='cupy'`,
   - measurement method:
     - clear the CuPy default memory pool, pinned pool, and FFT plan cache
       before the baseline snapshot,
     - sample `cupy.cuda.runtime.memGetInfo()` every `5 ms` in a side thread
       during `run(...)`,
     - for CuPy runs also record post-run `pool.total_bytes()` and
       `pool.used_bytes()` to separate live arrays from allocator-retained free
       blocks.
4. Measured results on this probe:
   - `cupy-rsoxs` host-resident lane:
     - baseline driver-used: about `165 MB`,
     - peak driver-used during `run(...)`: about `1227 MB`,
     - peak delta versus baseline: about `+1062 MB`,
     - post-run driver-used before release: about `1227 MB`,
     - post-run CuPy pool state before release:
       - `pool.total_bytes() ~= 1056 MB`,
       - `pool.used_bytes() ~= 5.5 MB`,
     - post-run after `release_runtime()`:
       - driver-used fell to about `203 MB`,
       - interpretation: nearly all of the extra post-run host-lane residency
         was allocator-retained free memory rather than still-live arrays.
   - `cyrsoxs` legacy lane:
     - baseline driver-used: about `165 MB`,
     - peak driver-used during `run(...)`: about `759 MB`,
     - peak delta versus baseline: about `+594 MB`,
     - post-run driver-used: about `171 MB`.
   - non-comparable device-resident context lane:
     - baseline driver-used: about `165 MB`,
     - post-construction driver-used: about `641 MB`,
     - peak driver-used during `run(...)`: about `1253 MB`,
     - peak delta versus baseline: about `+1088 MB`,
     - post-run after `release_runtime()`: about `557 MB`,
     - interpretation: device-resident mode intentionally retains the
       authoritative morphology arrays on GPU and should not be used as the
       fairness baseline versus `cyrsoxs`.
   - interpretation:
     - the host-resident `cupy-rsoxs` lane shows a real in-run peak-memory gap
       versus `cyrsoxs` on this maintained CoreShell case of about `468 MB`,
     - this gap is therefore not only a post-run allocator-retention artifact,
     - however the large post-run apparent residency in host mode is still
       mostly pool-retained free memory and should not be confused with
       steady-state live arrays.
5. Current lifetime interpretation from code inspection for
   `execution_path='tensor_coeff'`:
   - host-resident staged morphology (`runtime_materials`, including staged
     `Vfrac`, `S`, `theta`, and `psi`) is created before the per-energy loop
     and is currently kept alive until the end of the outer run scope,
   - in the maintained `tensor_coeff` path, those staged morphology arrays are
     only needed through Segment `B` while `_compute_nt_components(...)` is
     building `nt`,
   - after Segment `B`, later Segments `C`, `D`, and `E` operate on `nt`,
     FFT-domain `nt`, detector/projection geometry, and projection families
     rather than on the staged morphology fields,
   - the `nt` buffer allocated in Segment `B` is transformed in place by
     Segment `C`,
   - therefore there is no separate real-space `nt` live set after Segment
     `C`,
   - the FFT-domain `nt` alias remains needed through Segment `D`,
   - after Segment `D`, Segment `E` uses `proj_x`, `proj_y`, and `proj_xy`, so
     `nt` / `fft_nt` is no longer needed,
   - `proj_x`, `proj_y`, and `proj_xy` are still needed through Segment `E`
     and are not early-release candidates in this pass.
6. Resulting memory hypotheses for the maintained `tensor_coeff` host lane:
   - candidate early release point 1:
     - drop host-staged `runtime_materials` immediately after Segment `B`,
     - if the goal is shared-GPU relief rather than same-process reuse, pair
       that drop with an explicit
       `cp.cuda.Stream.null.synchronize()` plus pool trim
       (`cp.get_default_memory_pool().free_all_blocks()` and
       `cp.get_default_pinned_memory_pool().free_all_blocks()`) rather than
       relying on Python scope exit alone.
   - candidate early release point 2:
     - drop `nt` / FFT-domain `nt` immediately after Segment `D` and before
       Segment `E`,
     - again treat allocator trimming as a separate experimental control rather
       than assuming reference deletion alone will return memory to other
       processes.
7. Tensor-coeff-only experiment plan for the next memory pass:
   - scope rule:
     - limit the pass to `execution_path='tensor_coeff'`,
     - do not broaden this pass to `direct_polarization` or
       `nt_polarization`,
     - keep host-resident `cupy-rsoxs` as the comparison authority against
       legacy `cyrsoxs`.
   - baseline measurement:
     - rerun the March 26 small single-energy no-rotation host lane with the
       current code,
     - record:
       - primary wall time,
       - segments `B`, `C`, `D`, and `E`,
       - peak driver-used memory,
       - explicit snapshots immediately after Segment `B`, immediately after
         Segment `D`, and after `run(...)`,
       - CuPy `pool.total_bytes()` and `pool.used_bytes()` at the same
         checkpoints when available.
   - experiment `mem01_drop_runtime_materials_no_trim`:
     - delete / release `runtime_materials` immediately after Segment `B`,
     - do not synchronize or trim the pool,
     - purpose:
       - distinguish same-process allocator reuse from actual driver-visible
         memory return.
   - experiment `mem02_drop_runtime_materials_sync_only`:
     - delete `runtime_materials` after Segment `B`,
     - explicitly call `cp.cuda.Stream.null.synchronize()`,
     - do not trim the pool,
     - purpose:
       - test whether queued work or stream semantics are hiding the usable
         lifetime boundary.
   - experiment `mem03_drop_runtime_materials_sync_free_pool`:
     - delete `runtime_materials` after Segment `B`,
     - call `cp.cuda.Stream.null.synchronize()`,
     - call `cp.get_default_memory_pool().free_all_blocks()` and
       `cp.get_default_pinned_memory_pool().free_all_blocks()`,
     - purpose:
       - test whether the long `C` / `D` / `E` window can materially reduce
         driver-visible memory in shared-GPU conditions,
       - quantify the latency penalty of the synchronize plus trim boundary.
   - experiment `mem04_drop_fft_nt_no_trim`:
     - delete `nt` / FFT-domain `nt` immediately after Segment `D`,
     - do not synchronize or trim the pool,
     - purpose:
       - distinguish same-process allocator reuse from actual driver-visible
         memory return in the `D -> E` boundary.
   - experiment `mem05_drop_fft_nt_sync_only`:
     - delete `nt` / FFT-domain `nt` after Segment `D`,
     - call `cp.cuda.Stream.null.synchronize()`,
     - do not trim the pool,
     - purpose:
       - test whether queued work delays the usable release boundary after
         Segment `D`.
   - experiment `mem06_drop_fft_nt_sync_free_pool`:
     - delete `nt` / FFT-domain `nt` after Segment `D`,
     - call `cp.cuda.Stream.null.synchronize()`,
     - call `cp.get_default_memory_pool().free_all_blocks()` and
       `cp.get_default_pinned_memory_pool().free_all_blocks()`,
     - purpose:
       - measure whether Segment `E` can run with materially lower
         driver-visible memory,
       - quantify the latency penalty of the `D -> E` synchronize plus trim
         boundary.
   - experiment `mem07_combined_low_memory_tensor_coeff`:
     - combine the best-performing Segment `B` and Segment `D` early-release
       candidates in one host-resident `tensor_coeff` run,
     - measure whether the combined strategy reduces the long-lived footprint
       enough to matter on shared GPUs without an unacceptable primary-time
       regression.
   - escalation rule:
     - only if a small-case host result is promising, repeat the winning
       candidate on:
       - the `medium` single-energy host lane,
       - and one explicit general-angle host lane such as
         `EAngleRotation=[0, 15, 165]`,
     - keep `cyrsoxs` comparison on the same host-authoritative single-energy
       lane rather than comparing device-resident `cupy-rsoxs` against legacy.
8. Acceptance criteria for this memory pass:
   - prioritize sustained driver-visible memory reduction during the long later
     segments rather than post-run cleanup alone,
   - require the comparison table to show both:
     - memory effect:
       - peak driver-used delta,
       - after-`B` and after-`D` driver-used deltas,
       - and post-run driver-used / pool totals for interpretation,
     - time effect:
       - primary wall time,
       - and segment deltas around the inserted synchronize / trim points,
   - if a candidate only moves pool-retained free memory without materially
     lowering driver-visible usage during the later segments, treat it as
     insufficient for the shared-GPU problem,
   - if a candidate materially lowers sustained driver-visible usage but
     regresses primary time too heavily, record it and reject it or keep it as
     an opt-in low-memory policy rather than silently adopting it as the
     default.

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

### Current Segment D continuation plan (March 25, 2026)

This subsection records the prioritized next-pass plan for continuing Segment
`D` work after the accepted March 24 state. It is intentionally narrower than
the earlier mixed Segment `B` / `D` block:

1. the execution path for this pass is `tensor_coeff`,
2. the timing focus is device-resident only for inner-loop ranking,
3. every candidate optimization must clear a physics-parity check before it is
   accepted into the maintained tuned state,
4. and the optimization ledger should be updated after each attempted step or
   any important intermediate result.

Priority order and current outcomes for the next Segment `D` pass:

1. `planD01_detector_geometry_cache` - `implemented; retained as scaffolding`
   - cache the detector / Ewald geometry tables that are rebuilt today inside
     `_q_axes(...)` and `_project_scatter3d(...)`,
   - expected cache contents:
     - `qx`, `qy`, `qz`,
     - detector pixel index grids,
     - per-energy interpolation/support tables such as `valid`, `safe_z0`,
       `safe_z1`, and `frac`,
   - cache location should be `morphology._backend_runtime_state` so the
     lifetime matches the existing angle-plan and transform caches and is
     cleared on backend release,
   - rationale:
     - this is the lowest-risk way to reduce repeated Segment `D` setup work
     across all execution paths,
     - and it should help both the no-rotation and explicit-rotation device
     lanes without changing the core projection math.
   - outcome:
     - authoritative timing artifact:
       `test-reports/cupy-rsoxs-optimization-dev/planD01_detector_geometry_cache_seq_20260325_194304/summary.json`,
     - parity/smoke status:
       - maintained device strict/borrow CoreShell regression passed,
       - maintained host-resident CoreShell sim-regression smoke also passed,
     - measured deltas versus the initial device baseline:
       - aligned no-rotation device lane stayed effectively flat:
         `primary 0.219s -> 0.190s`, `D 0.035 -> 0.035`,
       - explicit general-angle device lane improved only modestly on that run:
         `primary 0.248s -> 0.244s`, `D 0.074 -> 0.070`,
     - later reruns did not reproduce a clean standalone material win on the
       general-angle lane, so this step is kept as low-risk detector-geometry
       scaffolding rather than counted as a standalone accepted speed gain.
2. `planD02_tensor_coeff_general_angle_projection_fusion` - `accepted; later extended to aligned x/y families`
   - target the current `tensor_coeff` general-angle path in
     `_projection_coefficients_from_fft_nt(...)`,
   - try to compute the x / y / xy detector-projection family with one shared
     detector walk rather than three largely repeated calls through
     `_projection_from_fft_polarization(...)`,
   - rationale:
     - the accepted aligned-angle work already cut the default `0°` lane,
     - so the remaining high-value `tensor_coeff` Segment `D` target is the
       general-angle path where projection-family work is still repeated.
   - outcome:
     - initial implementation:
       - the general-angle `tensor_coeff` path now computes `proj_x`,
         `proj_y`, and `proj_xy` directly on the detector grid from the two
         FFT basis families rather than calling the generic 3D
         `_projection_from_fft_polarization(...)` path three separate times,
     - follow-on implementation after a current-tree regression repro:
       - aligned `x` / `y` families now also route through the same
         detector-grid helper instead of calling
         `_projection_from_fft_polarization(...)`,
       - mixed aligned `x+y` angle sets reuse the paired helper directly,
       - single-family aligned sets use same-basis detector evaluation to avoid
         materializing a temporary `scatter3d` volume,
     - initial authoritative timing artifact:
       - `test-reports/cupy-rsoxs-optimization-dev/planD02_tensor_coeff_general_angle_projection_fusion_fix/summary.json`,
     - current accepted end-state artifacts:
       - `test-reports/cupy-rsoxs-optimization-dev/planD02b_aligned_detector_projection_no_rotation/summary.json`,
       - `test-reports/cupy-rsoxs-optimization-dev/planD02b_aligned_detector_projection_general/summary.json`,
       - when the explicit-rotation harness is used, it still emits its built-in
         no-rotation companion case; use the dedicated no-rotation artifact as
         the aligned regression authority,
     - superseded intermediate artifacts:
       - `test-reports/cupy-rsoxs-optimization-dev/planD02_tensor_coeff_general_angle_projection_fusion/summary.json`,
       - `test-reports/cupy-rsoxs-optimization-dev/planD02_tensor_coeff_general_angle_projection_fusion_rerun/summary.json`,
       - those two runs were taken before removing an accidentally retained
         legacy `proj_x` projection call from the general-angle wrapper and
         should not be treated as the final authority for `planD02`,
     - current-tree regression repro artifacts that motivated the aligned
       follow-on:
       - `test-reports/cupy-rsoxs-optimization-dev/planD02_repro_current_no_rotation/summary.json`,
       - `test-reports/cupy-rsoxs-optimization-dev/planD02_repro_current_general/summary.json`,
       - those reruns showed the aligned device guard had drifted back to
         about `primary 0.243-0.256s`, `D 0.055-0.076` while the targeted
         general-angle lane remained fast,
     - measured deltas versus the initial device baseline:
       - aligned no-rotation device lane:
         `primary 0.219s -> 0.177s`, `D 0.035 -> 0.020`,
       - explicit general-angle device lane:
         `primary 0.248s -> 0.185s`, `D 0.074 -> 0.020`,
     - interpretation:
       - the targeted general-angle lane is the ranking authority for this
         continuation pass and the win there is large enough to accept,
       - the later aligned-family follow-on removes the renewed no-rotation
         guard inflation while keeping the general-angle win intact, so this
         now defines the final accepted March 25 Segment `D` end state,
     - parity status:
       - maintained device strict/borrow CoreShell regression passed on the
         final aligned-family extension,
       - maintained host-resident CoreShell sim-regression smoke also passed
         on the same final code.
3. `planD03_direct_detector_kernel` - `rejected for this pass`
   - replace the current full `scatter3d` materialization plus later detector
     sampling with a direct detector-output implementation that computes the
     needed `z0` / `z1` contributions and interpolates intensity without first
     storing the full 3D scatter volume,
   - preserve current CyRSoXS-mimic math order and interpolation semantics,
   - rationale:
     - this is the largest potential Segment `D` win,
     - but it is also the highest-risk change in this block and should follow
       the lower-risk cache and algebra work above.
   - outcome:
     - a dedicated raw-kernel prototype for the `tensor_coeff` detector-pair
       helper reached about `0.10ms` warm versus about `3.7ms` for the
       accepted elementwise helper,
     - however its first cold call on the actual subprocess timing surface was
       about `59ms`, far worse than the accepted helper's cold cost,
     - because the current optimization authority for this pass is the cold
       device timing harness rather than a many-call steady-state kernel loop,
       this path is rejected for now.
4. `planD04_static_algebra_prefactor_cache` - `rejected`
   - precompute and reuse the static Segment `D` algebra built from detector
     geometry, for example `a`, `b`, `a^2`, `b^2`, `ab`, and other reusable
     q-space terms,
   - keep per-energy `c`-dependent terms separate when needed,
   - rationale:
     - this is the explicit continuation of the "algebraic factoring and
       scratch reuse around basis/projection assembly" note left open in the
       March 24 ledger,
     - and it composes naturally with geometry caching.
   - outcome:
     - prototype profiling showed the accepted helper itself could be nudged
       from about `3.4ms` warm to about `3.1ms` warm and from about `8.1ms`
       cold to about `2.8ms` cold,
     - implemented timing artifact:
       `test-reports/cupy-rsoxs-optimization-dev/planD04_static_algebra_prefactor_cache/summary.json`,
     - measured device-lane outcome:
       - aligned no-rotation device lane:
         `primary 0.186s -> 0.196s`, `D 0.034 -> 0.035`,
       - explicit general-angle device lane:
         `primary 0.186s -> 0.188s`, `D 0.020 -> 0.019`,
     - because the targeted general-angle primary time was effectively flat and
       the aligned regression lane got slightly worse, the change is rejected
       as non-material on the actual authority surface.
5. `planD05_tensor_coeff_scale_hoist` - `rejected`
   - test whether the constant `1 / (4*pi)` scaling applied while building
     projection families can be hoisted out of the hot full-array path and
     applied later as an output-scale factor without changing the accepted
     numerical target,
   - rationale:
     - expected gain is smaller than the items above,
     - but it is a simple change that could shave repeated full-array math from
       the maintained `tensor_coeff` route.
   - outcome:
     - implemented timing artifact:
       `test-reports/cupy-rsoxs-optimization-dev/planD05_tensor_coeff_scale_hoist/summary.json`,
     - measured device-lane outcome:
       - aligned no-rotation device lane stayed flat:
         `primary 0.186s -> 0.185s`, `D 0.034 -> 0.034`,
       - explicit general-angle device lane regressed on the primary metric:
         `primary 0.186s -> 0.205s`, `D 0.020 -> 0.021`,
     - this change is therefore rejected.
6. `planD06_segment_d_scratch_reuse` - `closed without implementation`
   - only after the geometry/algebra path is measured, test reusable scratch
     buffers for Segment `D` intermediates such as detector projections or
     direct-detector temporaries,
   - rationale:
     - scratch reuse may help allocator churn,
     - but the current code appears more geometry/computation heavy than
       allocator limited, so this should not be the first move.
   - outcome:
     - after the accepted `planD02` refactor, the remaining warm
       `tensor_coeff` detector-pair helper is already only a few milliseconds
       and the cold timing surface is dominated more by first-call kernel setup
       than by obvious allocator churn,
     - given that `planD03` was already rejected on cold first-use cost and
       `planD04`/`planD05` failed to produce material authority-surface gains,
       scratch reuse does not have a credible path to a material win in this
       pass,
     - close it out without implementation and treat the current accepted state
       as the stopping point for this Segment `D` continuation.

Acceptance protocol for this block:

1. Inner-loop timing authority for this pass is the device-resident
   `tensor_coeff` lane.
2. Use both:
   - the aligned no-rotation small single-energy device lane as the regression
     guard,
   - and at least one explicit general-angle device lane such as
     `--rotation-specs 0:15:165` when ranking Segment `D` changes, because the
     accepted aligned-angle work already removed much of the easy `D` cost from
     the `0°` lane.
3. A candidate Segment `D` change must show a material speed gain on the
   targeted device lane before it can proceed to acceptance review.
4. Before accepting that gain, rerun maintained parity/correctness checks and
   confirm there is no unacceptable physics drift.
5. If a candidate is rejected, record the measured outcome here rather than
   silently dropping the attempt.

Initial device-only Segment `D` baseline for this continuation pass:

- artifact:
  - `test-reports/cupy-rsoxs-optimization-dev/planD00_device_baseline_20260325_193556/summary.json`
- measured on March 25, 2026 before any new Segment `D` code changes:
  - aligned no-rotation device lane:
    - `core_shell_small_single_no_rotation_device_tensor_coeff`
    - `primary 0.219s`, `D 0.035s`
  - explicit general-angle device lane:
    - `core_shell_small_single_rot_0_15_165_device_tensor_coeff`
    - `primary 0.248s`, `D 0.074s`
- interpretation:
  - the accepted aligned-angle work from March 24 still keeps the `0°` device
    lane relatively cheap in Segment `D`,
  - the general-angle device lane remains the more useful ranking surface for
    continued Segment `D` optimization.

Current maintained parity route for this continuation pass:

- use the in-repo CoreShell validation surface rather than legacy CLI parity:
  - device strict/borrow gate:
    `pytest tests/validation/test_core_shell_reference.py -k "test_core_shell_sim_regression_cupy_borrow_strict" --nrss-backend cupy-rsoxs -v`,
  - host-resident regression smoke:
    `pytest tests/validation/test_core_shell_reference.py -k "test_core_shell_sim_regression_pybind" --nrss-backend cupy-rsoxs -v`.

### Current Segment E continuation plan (March 25, 2026)

This subsection records the prioritized next-pass plan for Segment `E`
rotation and angle accumulation after the accepted March 25 Segment `D`
state:

1. the execution path for this pass is `tensor_coeff`,
2. the quick-ranking lane is the small device-resident explicit-rotation
   CoreShell case family because the default `0°` lane already bypasses the
   affine path,
3. accepted changes must also clear host-resident steady-state regression
   checks with `--cuda-prewarm before_prepare_inputs`,
4. accepted changes must clear maintained CoreShell sim-regression physics
   gates on both the default host-resident and device strict/borrow
   `cupy-rsoxs` paths because that validation uses the maintained
   full-rotation `EAngleRotation=[0.0, 1.0, 360.0]` CoreShell workflow,
5. and the optimization ledger should be updated after each attempted Segment
   `E` step or any important intermediate result.

Current Segment `E` ranking baseline from the latest explicit multi-angle
artifact:

1. source artifact:
   `test-reports/cupy-rsoxs-optimization-dev/execution_path_multiangle_5_vs_15_20260324/summary.json`
2. ranking lanes for `tensor_coeff`:
   - host / `EAngleRotation=[0, 15, 165]`:
     `primary 2.742s`, `E 0.00734`,
   - host / `EAngleRotation=[0, 5, 165]`:
     `primary 2.921s`, `E 0.01559`,
   - device / `EAngleRotation=[0, 15, 165]`:
     `primary 0.419s`, `E 0.01466`,
   - device / `EAngleRotation=[0, 5, 165]`:
     `primary 0.412s`, `E 0.02937`.
3. interpretation:
   - Segment `E` is still minor on the no-rotation lane,
   - but it scales roughly with the number of sampled angles and is visible on
     the maintained explicit-rotation `tensor_coeff` workload,
   - therefore explicit multi-angle `tensor_coeff` device timing is the right
     first ranking surface for this pass.

Priority order for the Segment `E` pass:

1. `planE01_rotation_accumulation_scratch_reuse` - `rejected`
   - preallocate Segment `E` accumulation buffers and avoid repeated
     projection-average reallocations inside the per-angle loop,
   - use `output=` where supported in the CuPy affine path,
   - keep this step low-risk and semantics-preserving.
   - outcome:
     - authoritative artifacts:
       - current-tree baseline roots:
         `test-reports/cupy-rsoxs-optimization-dev/planE00_device_baseline_20260325/summary.json`,
         `test-reports/cupy-rsoxs-optimization-dev/planE00_host_warm_20260325/summary.json`,
       - attempt roots:
         `test-reports/cupy-rsoxs-optimization-dev/planE01_rotation_accumulation_scratch_reuse/summary.json`,
         `test-reports/cupy-rsoxs-optimization-dev/planE01_rotation_accumulation_scratch_reuse_host_warm/summary.json`,
     - measured host-prewarmed outcome versus the current-tree host baseline:
       - no-rotation host lane improved on primary only:
         `primary 0.254s -> 0.219s`, `E 0.00158 -> 0.00168`,
       - explicit `EAngleRotation=[0, 15, 165]` host lane stayed flat:
         `primary 0.272s -> 0.273s`, `E 0.00706 -> 0.00717`,
       - explicit `EAngleRotation=[0, 5, 165]` host lane stayed effectively flat:
         `primary 0.285s -> 0.282s`, `E 0.01387 -> 0.01400`,
     - device interpretation:
       - the first explicit-rotation device baseline in this continuation pass
         clearly paid a large cold first-use cost, so the later lower device
         `E` timings from this attempt are not enough on their own to justify
         acceptance,
       - the host-prewarmed explicit-rotation lanes are therefore the
         acceptance authority for this step.
     - interpretation:
       - this allocator/scratch-only rewrite did not materially improve the
         maintained explicit-rotation `tensor_coeff` authority surface,
       - reject it and proceed to the next Segment `E` priority step.
2. `planE02_texture_affine_transform_probe` - `rejected`
   - probe CuPy `ndimage.affine_transform(..., texture_memory=True)` on the
     2D projection path using cached float32 homogeneous transforms,
   - rank it on both cold and steady-state surfaces before accepting it.
   - outcome:
     - authoritative artifacts:
       - attempt roots:
         `test-reports/cupy-rsoxs-optimization-dev/planE02_texture_affine_transform_probe/summary.json`,
         `test-reports/cupy-rsoxs-optimization-dev/planE02_texture_affine_transform_probe_host_warm/summary.json`,
       - compare against the current-tree host baseline at
         `test-reports/cupy-rsoxs-optimization-dev/planE00_host_warm_20260325/summary.json`,
     - measured host-prewarmed outcome versus the current-tree host baseline:
       - no-rotation host lane stayed flat:
         `primary 0.254s -> 0.256s`, `E 0.00158 -> 0.00154`,
       - explicit `EAngleRotation=[0, 15, 165]` host lane regressed materially:
         `primary 0.272s -> 0.303s`, `E 0.00706 -> 0.00815`,
       - explicit `EAngleRotation=[0, 5, 165]` host lane stayed near-flat on
         primary but regressed on Segment `E`:
         `primary 0.285s -> 0.282s`, `E 0.01387 -> 0.01825`,
     - device interpretation:
       - the explicit `EAngleRotation=[0, 15, 165]` device lane also regressed
         sharply on this cold surface:
         `primary 0.409s`, `E 0.153`,
       - the texture path therefore does not offer a compelling cold-surface
         or steady-state advantage in this pass.
     - interpretation:
       - reject the texture-backed affine path for this Segment `E` campaign
         and return to the accepted non-texture baseline before the next step.
3. `planE03_rotmask_zero_fast_path` - `rejected`
   - separate the common `RotMask=0` path from the `RotMask=1` valid-count
     averaging path so the default no-mask lane does not pay for per-angle
     validity bookkeeping,
   - preserve the current `RotMask=1` semantics, which average over valid
     angles rather than applying only a final post hoc crop.
   - outcome:
     - authoritative artifact:
       `test-reports/cupy-rsoxs-optimization-dev/planE03_rotmask_zero_fast_path/summary.json`,
     - measured device-lane outcome:
       - no-rotation device lane regressed heavily:
         `primary 0.165s -> 0.340s`, `E 0.00146 -> 0.171`,
       - explicit `EAngleRotation=[0, 15, 165]` device lane regressed:
         `primary 0.181s -> 0.378s`, `E 0.00672 -> 0.195`,
       - explicit `EAngleRotation=[0, 5, 165]` device lane also regressed:
         `primary 0.224s -> 0.198s`, `E 0.01444 -> 0.0250`,
     - interpretation:
       - rebuilding the strict no-mask validity surface by rotating a support
         mask adds too much extra Segment `E` work to be viable on this
         authority surface,
       - reject without host or physics follow-up.
4. `planE04_rotate_accumulate_kernel_fusion` - `not fully tried`
   - fuse rotate, validity handling, and accumulation into one GPU helper for
     the shared projection path if the lower-risk steps do not materially cut
     Segment `E`.
   - partial outcome:
     - exploratory device artifact:
       `test-reports/cupy-rsoxs-optimization-dev/planE04_rotate_accumulate_kernel_fusion/summary.json`,
     - the exploratory device screen showed a promising explicit-rotation
       reduction but also changed unrelated surface timing enough that the
       implementation needed one more cleanup pass before host-prewarmed
       acceptance timing or physics gates,
     - the backend code was therefore reverted before the change was carried
       through the required host-resident and CoreShell sim-regression gates.
5. `planE05_tensor_coeff_projection_rotation_fusion` - `planned`
   - if needed, target the maintained `tensor_coeff` path more aggressively by
     eliminating the intermediate per-angle detector projection image and
     combining x/y/xy weighting with rotation accumulation.
6. `planE06_exact_quarter_turn_fast_path` - `planned`
   - add exact `90°/180°/270°` image-space transforms for rotation sets that
     land exactly on the detector grid so those cases avoid generic affine
     interpolation.

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
   - historical initial values for that March 24, 2026 surface:
     - `tensor_coeff`: current accepted `Nt -> FFT(Nt) -> projection-coefficient`
       route,
     - `direct_polarization`: CyRSoXS `AlgorithmType=0`
       communication-minimizing analog,
     - `nt_polarization`: CyRSoXS `AlgorithmType=1`
       memory-minimizing analog,
   - outcome:
     - completed on March 24, 2026,
     - at that time `cupy-rsoxs` accepted
       `backend_options["execution_path"]` with
       supported values `tensor_coeff`, `direct_polarization`, and
       `nt_polarization`,
     - aliases now normalize as:
       - `default -> tensor_coeff`,
       - `tensor -> tensor_coeff`,
       - `direct -> direct_polarization`,
       - `nt -> nt_polarization`,
     - default behavior remained unchanged at `execution_path='tensor_coeff'`,
     - current supported values are `tensor_coeff` and
       `direct_polarization`; `nt_polarization` has since been removed from the
       maintained surface.
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
10. `plan10_mixed_precision_followup_scaffold` - `completed`
    - do not implement the mixed-precision campaign in this block,
    - only leave the execution-path surface and roadmap notes in a form that
      allows an orthogonal future backend-options extension for a named
      mixed-precision mode,
    - outcome:
      - completed on March 24, 2026 with no mixed-precision execution-path work
        landed,
      - `execution_path` remains orthogonal to future mixed-precision work,
      - the old generic backend `dtype` framing should be considered stale and
        replaced by a named mixed-precision mode,
      - no reduced-precision compute-path work was accepted in this state,
      - this scaffold is historical only and is superseded by the approved
        mixed-precision implementation plan recorded later in this ledger and
        in `CUPY_RSOXS_BACKEND_SPEC.md`.
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
2. The mixed-precision plan is orthogonal to `execution_path`.
   - approved implementation direction:
     - remove the exposed generic backend `dtype` option entirely,
     - use a named `mixed_precision_mode`,
     - treat reduced morphology storage / host->device transfer precision as the
       first target,
     - carry `float16` morphology inputs through pre-FFT work in both
       `tensor_coeff` and `direct_polarization`,
     - widen into parity-sensitive FFT-ingress compute without widening the
       authoritative morphology arrays during normalization/staging,
     - keep FFT/projection math parity-sensitive.
3. Reduced precision should not be exposed as a generic backend `dtype` knob.
   - the agreed option surface is a named mixed-precision mode,
   - the mode overrides `input_policy` and behaves as strict regardless of the
     user's `input_policy` value,
   - host-resident mode requires authoritative `numpy.float16` morphology
     inputs,
   - device-resident mode requires authoritative `cupy.float16` morphology
     inputs,
   - non-conforming inputs fail immediately rather than being silently
     downcast.
4. The validator change should be narrow and physics-driven:
   - closure remains a voxelwise invariant,
   - the mixed-precision closure rule should be expressed as
     `abs(sum_i Vfrac_i - 1) <= 1e-3` per voxel,
   - this rule should operate in the authoritative dtype of the mixed-precision
     mode by default,
   - other validator checks remain materially unchanged.
5. Segment `D` should keep a note for future pruning and scratch-reuse work:
   - axis-family cases may avoid building `proj_xy` and one of the x/y
     projection families,
   - algebraic factoring and scratch reuse around basis/projection assembly may
     be worthwhile there after the Segment `B` pass is measured.
6. Immediate continuation order for the next Segment `D` block is now:
   - detector/Ewald geometry caching,
   - `tensor_coeff` general-angle projection-family fusion,
   - direct detector-output kernel work,
   - static algebra/prefactor reuse,
   - then optional scratch reuse after the math path is simplified.

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

### Explicit experiments and approved directions

1. Mixed-precision morphology handling is now an approved implementation track;
   parity-sensitive compute remains `float32/complex64`.
   - required public surface:
     - `backend_options={"mixed_precision_mode": "reduced_morphology_bit_depth"}`
       plus the existing `execution_path`,
     - no exposed generic backend `dtype` option.
   - required strict input contract:
     - host-resident mode requires authoritative `numpy.float16` morphology
       arrays,
     - device-resident mode requires authoritative `cupy.float16` morphology
       arrays,
     - mixed mode overrides `input_policy` and behaves as strict.
   - required runtime precision ladder:
     - keep authoritative and staged morphology handling at `float16`,
     - in both `tensor_coeff` and `direct_polarization`, carry `float16`
       morphology inputs through the pre-FFT decode/composition work,
     - widen into `float32/complex64` FFT-ingress compute without a required
       standalone pre-FFT conversion pass,
     - keep FFT/q-space and detector/projection math at `float32/complex64`.
   - required validation change:
     - mixed-mode closure budget remains
       `abs(sum_i Vfrac_i - 1) <= 1e-3` per voxel in the authoritative dtype.
   - required harness change:
     - extend the maintained CoreShell helper and the
       `run_cupy_rsoxs_optimization_matrix.py` dev harness so they can emit the
       strict `float16` inputs required by this mode,
     - compare standard versus mixed host/device lanes across both supported
       execution paths.
   - required graphical-abstract output:
     - produce a CoreShell graphical abstract comparing the standard
       `tensor_coeff` path against the mixed-precision `tensor_coeff` path so
       precision loss can be inspected directly.
   - implementation status on April 4, 2026:
     - completed the initial runtime/contracts/tests pass in the maintained
       codebase,
     - removed the exposed generic `dtype` option from `cupy-rsoxs`,
     - implemented the public surface
       `backend_options={"mixed_precision_mode": "reduced_morphology_bit_depth"}`,
     - implemented strict authoritative `float16` host/device input handling,
     - implemented the mixed-mode closure budget
       `abs(sum_i Vfrac_i - 1) <= 1e-3`,
     - implemented mixed authoritative/staged morphology handling for both
       `tensor_coeff` and `direct_polarization`,
     - preserved `float32/complex64` FFT/post-FFT math,
     - added smoke coverage for the new contracts and runtime behavior,
     - verified the maintained smoke suite in `nrss-dev`,
     - current documented `tensor_coeff` implementation widens half inputs
       during `Nt` construction and writes `complex64 Nt` directly, so the FFT
       runs on `complex64 Nt` rather than after a separate full-volume
       pre-FFT cast,
     - this exact widening boundary is documented implementation state, not a
       separately maintained test-backed contract,
     - intentionally deferred any maintained validation-surface expansion tied
       specifically to that internal promotion boundary to a later context.
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
