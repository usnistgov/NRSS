# `cupy-rsoxs` `direct_polarization` Optimization Notes

This document is the path-specific optimization note for
`execution_path='direct_polarization'` in `cupy-rsoxs`.

It is a companion to:

- `CUPY_RSOXS_OPTIMIZATION_LEDGER.md`
  - the authoritative backend-wide speed and timing ledger
- `CUPY_RSOXS_BACKEND_SPEC.md`
  - the stable backend contract and execution-path scope
- `tests/validation/dev/cupy_rsoxs_optimization/README.md`
  - harness usage, current timing workflow, and development-only run notes

This note exists because the main optimization ledger is now centered on the
accepted `tensor_coeff` state and the March 25 Segment `D` continuation, while
`direct_polarization` has become a distinct optimization thread with different
hotspots, tradeoffs, and benchmark caveats.

## Scope

This note tracks:

1. the current authoritative timing and memory surface for
   `direct_polarization`,
2. the most relevant historical benchmark context that was previously mixed
   into the main ledger,
3. the current interpretation of its bottlenecks,
4. and the ranked next optimization candidates for this path.

This note does not replace the backend-wide ledger. If a path-specific result
changes the accepted backend-wide optimization state, record it in
`CUPY_RSOXS_OPTIMIZATION_LEDGER.md` as well.

Current supported `cupy-rsoxs` execution paths are `tensor_coeff` and
`direct_polarization`. Historical comparisons against `nt_polarization` remain
in this note as archival context from the earlier three-path campaign.

## Path Identity

Current execution-path mapping:

1. `tensor_coeff`
   - maintained default `cupy-rsoxs` execution path
2. `direct_polarization`
   - CyRSoXS `AlgorithmType=0` analog
   - communication-minimizing / polarization-first analog

`direct_polarization` remains useful because it:

1. is the closest conceptual match to the legacy polarization-first path,
2. keeps a lower GPU memory footprint than `tensor_coeff` on the maintained
   host-resident CoreShell lane,
3. and is the cleanest place to study low-memory direct-field computation
   without reusing `Nt` across angles.

## Current Authoritative Benchmarks

### Current clean timing authority

The current clean timing authority for the small, single-energy, no-rotation,
host-prewarmed lane is:

- `test-reports/cupy-rsoxs-optimization-dev/crosspath_speed_recheck_20260328_host_prewarm_now/summary.json`

That artifact records:

1. morphology:
   - maintained CoreShell helper
   - shape `(32, 512, 512)`
   - `PhysSize=2.5`
2. run conditions:
   - `resident_mode='host'`
   - `field_namespace='numpy'`
   - `input_policy='strict'`
   - `ownership_policy='borrow'`
   - `cuda_prewarm_mode='before_prepare_inputs'`
   - single energy `[285.0]`
   - `EAngleRotation=[0.0, 0.0, 0.0]`
3. `direct_polarization` timing:
   - `primary_seconds ~= 0.3128 s`
   - `A2 ~= 0.0889 s`
   - `B ~= 0.1563 s`
   - `C ~= 0.00875 s`
   - `D ~= 0.0437 s`
   - `E ~= 0`

Current interpretation from that clean recheck:

1. the current dominant hotspot is Segment `B`,
2. Segment `D` is real but secondary,
3. and the path is no longer showing the earlier suspicious extreme `D`
   outlier.

### Current host-resident peak-memory authority

The current host-resident peak-memory authority for the same lane is:

- `test-reports/cupy-rsoxs-optimization-dev/crosspath_speed_mem_recheck_20260402_host_prewarm/summary.json`

That artifact was produced with an external `nvidia-smi` polling probe. The
polling perturbs wall time enough that its `primary_seconds` values should not
be used as timing authority. Use it only for peak-memory observations.

Current peak-memory results from that artifact:

1. `tensor_coeff`
   - peak GPU memory: `1229 MiB`
2. `direct_polarization`
   - peak GPU memory: `1097 MiB`
3. `nt_polarization`
   - peak GPU memory: `1485 MiB`

Current interpretation:

1. `direct_polarization` remains the lowest-footprint `cupy-rsoxs`
   computation path on this maintained host-resident lane,
2. but it is not the fastest path,
3. and its remaining speed gap is now best explained by Segment `B`, not by an
   obviously pathological Segment `D`.

### Important non-authoritative artifact

Treat the older artifact below as historical context only, not as the current
authority:

- `test-reports/cupy-rsoxs-optimization-dev/crosspath_speed_20260328_host_prewarm/summary.json`

That saved summary reported a host-prewarmed `direct_polarization` anomaly of
approximately:

1. `primary ~= 1.564 s`
2. `D ~= 1.330 s`

That result was later checked against:

1. a fresh three-path timing rerun,
2. direct backend private-segment timing on the current tree,
3. and isolated `_compute_scatter3d(...)` plus `_project_scatter3d(...)`
   probes,

and the anomaly did not reproduce. Current documentation should therefore
treat that artifact as a stale or bad run, not as the current execution-path
authority.

## Historical Benchmark Context Kept For This Path

### March 24 execution-path surfacing baseline

Source artifact:

- `test-reports/cupy-rsoxs-optimization-dev/execution_path_surface_smoke_20260324/summary.json`

Small, single-energy, no-rotation baseline at path surfacing time:

1. host / `direct_polarization`
   - `primary 2.609 s`
   - `A2 2.401`
   - `B 0.143`
   - `D 0.033`
   - `E 0.002`
2. device / `direct_polarization`
   - `primary 0.204 s`
   - `A2 0.000`
   - `B 0.138`
   - `D 0.033`
   - `E 0.002`

Initial interpretation:

1. `direct_polarization` and `nt_polarization` were immediately competitive on
   the surfaced no-rotation lane,
2. both were materially faster than the then-current `tensor_coeff`
   no-rotation `D`/`E` path,
3. but neither had yet been elevated to the maintained optimized default.

### March 24 accepted-state rebenchmark

Source artifact:

- `test-reports/cupy-rsoxs-optimization-dev/plan09_final_rebenchmark_accepted_state_20260324/summary.json`

Accepted-state no-rotation snapshot:

1. host / `direct_polarization`
   - `primary 2.541 s`
   - `A2 2.364`
   - `B 0.113`
   - `D 0.034`
   - `E ~0`
2. device / `direct_polarization`
   - `primary 0.198 s`
   - `A2 0.116`
   - `B 0.014`
   - `D 0.034`
   - `E ~0`

Important context:

1. those March 24 host totals were not host-prewarmed,
2. and they should not be compared directly against the later host-prewarmed
   March 28/April 2 timing lanes without calling out the prewarm difference.

### March 24 multi-angle context

Source artifact:

- `test-reports/cupy-rsoxs-optimization-dev/execution_path_multiangle_5_vs_15_20260324/summary.json`

Historical `direct_polarization` results:

1. host, `15°` step sweep
   - `3.742 s`
2. host, `5°` step sweep
   - `4.524 s`
3. device, `15°` step sweep
   - `0.556 s`
4. device, `5°` step sweep
   - `1.559 s`

Interpretation:

1. the direct path scales poorly with dense angle sweeps because it rebuilds
   polarization per angle,
2. so it should not be expected to converge on the reuse advantage of
   `tensor_coeff` or `nt_polarization` in multi-angle workloads unless it
   starts caching additional reusable intermediates.

## Current Implementation Interpretation

The current clean profile for `direct_polarization` is:

1. Segment `B` dominates,
2. Segment `D` is secondary,
3. Segment `C` is small,
4. Segment `E` is negligible on the no-rotation lane.

Current code path:

1. Segment `B`
   - `_compute_direct_polarization(...)`
2. Segment `C`
   - `_fft_polarization_fields(...)`
3. Segment `D`
   - `_projection_from_fft_polarization(...)`
   - `_compute_scatter3d(...)`
   - `_project_scatter3d(...)`

Current direct-path bottleneck explanation:

1. `_compute_direct_polarization(...)` still builds several large
   per-material temporaries:
   - `isotropic_term`
   - `phi_a`
   - `sx`, `sy`, `sz`
   - `field_projection`
2. those temporaries are large enough that memory traffic dominates the
   remaining direct-path speed gap on the maintained no-rotation lane.

Current `D` interpretation:

1. `direct_polarization` still materializes a full `scatter3d` volume before
   detector projection,
2. unlike the accepted `tensor_coeff` detector-grid helper path,
3. but the clean recheck shows that this is now a secondary optimization
   target for the maintained small no-rotation lane.

## Ranked Next Optimization Candidates

Current recommended priority order:

1. Fuse `_compute_direct_polarization(...)` into custom kernels.
   - goal:
     - remove the large per-material temporary traffic in Segment `B`
   - expected benefit:
     - highest direct-path speed upside on the clean authority lane
   - risk:
     - moderate implementation complexity
2. Split Segment `B` into specialized aligned-family kernels.
   - goal:
     - keep distinct `x`, `y`, and general-angle paths rather than forcing all
       cases through one broader algebra route
   - expected benefit:
     - modest-to-material `B` savings on aligned and no-rotation lanes
   - risk:
     - moderate complexity, but lower than a fully generic fused route
3. Add a detector-plane direct projection path for `direct_polarization`.
   - goal:
     - avoid `scatter3d` materialization in Segment `D`
   - expected benefit:
     - real speed and memory improvement
   - current ranking:
     - second tier because the current clean `D` cost is only about `44 ms`
4. Add Segment `B` scratch reuse if a full fused rewrite is deferred.
   - goal:
     - reuse one or two scratch volumes for `phi_a`, orientation products, and
       field projection
   - expected benefit:
     - lower-risk partial reduction in direct-path temporary pressure
5. Pre-split isotropic and anisotropic material lists once per run.
   - goal:
     - remove repeated hot-loop branch checks on `material.is_full_isotropic`
   - expected benefit:
     - smaller, but cheap
6. Cache `sx`, `sy`, `sz` across angles or energies only if a path-specific
   higher-memory mode is explicitly desired.
   - goal:
     - trade GPU memory for multi-angle or multi-energy direct-path speed
   - current ranking:
     - low because it conflicts with the low-memory appeal of this path

## Current Blocking Questions And Decision Points

Before accepting major direct-path work, keep these decision points explicit:

1. Is the main target:
   - single-angle / no-rotation latency,
   - or multi-angle throughput?
2. Are custom CUDA kernels acceptable in this backend for a direct-path
   Segment `B` rewrite?
3. Should `direct_polarization` preserve its current lower-memory character,
   or is a higher-memory cached fast path acceptable as an opt-in mode?
4. Is numerical equivalence sufficient for any detector-plane `D` rewrite, or
   is bitwise stability required on the current parity surface?
5. Should direct-path optimization focus on the aligned-angle common case even
   if the general-angle route remains materially slower?

Current working bias:

1. optimize the single-angle / no-rotation lane first,
2. keep the path low-memory by default,
3. and treat a fused Segment `B` implementation as the first serious speed
   candidate.

## Eight Recorded Potential Improvements

The following eight items are now the recorded potential-improvement catalog
for `direct_polarization`.

These should remain on the table for a future higher-reasoning retry even
though this pass did not produce an accepted implementation change.

### 1. Hoist cheap per-energy / per-run work out of the hot angle loop

- idea:
  - precompute direct-path optical scalars and any cheap material splits once
    per energy rather than inside `_compute_direct_polarization(...)` for
    every angle
- this pass:
  - attempted
- current disposition:
  - rejected for the maintained authority surface
- reason:
  - no-rotation authority timing did not improve materially, and the mixed
    angle result showed a suspicious `E` collapse that should not be credited
    to this refactor without a cleaner re-check

### 2. Add pure-CuPy Segment `B` scratch reuse

- idea:
  - reuse a small scratch set for `phi_a`, orientation products, and field
    projection to reduce temporary-allocation traffic without changing the math
- this pass:
  - attempted
- current disposition:
  - rejected for the maintained authority surface
- reason:
  - materially regressed both no-rotation and `0:5:165` host-prewarmed timing

### 3. Split Segment `B` into explicit `x`, `y`, and general-angle Python/CuPy routes

- idea:
  - keep aligned families on narrower algebra paths rather than branching
    inside the inner loop
- this pass:
  - attempted
- current disposition:
  - rejected for the maintained authority surface
- reason:
  - materially regressed both no-rotation and `0:5:165` host-prewarmed timing

### 4. Add aligned-family custom kernels for Segment `B`

- idea:
  - use custom CUDA kernels only for the common aligned `x` / `y` families as
    a lower-risk precursor to a fully generic fused rewrite
- this pass:
  - attempted as an aligned-family raw-kernel prototype
- current disposition:
  - rejected for the maintained authority surface
- reason:
  - cold subprocess timing regressed badly enough that the prototype is not
    acceptable on the current benchmark boundary

### 5. Fuse `_compute_direct_polarization(...)` fully into custom kernels

- idea:
  - remove large per-material temporary traffic in Segment `B` with a generic
    fused kernel route
- this pass:
  - not attempted as a full generic rewrite
- current disposition:
  - still a potential improvement
- reason:
  - the aligned-family kernel probe showed that cold compile/load cost must be
    treated as a first-class acceptance risk on the current subprocess timing
    surface

### 6. Add a detector-plane direct projection path for `direct_polarization`

- idea:
  - bypass full `scatter3d` materialization in Segment `D` by projecting the
    FFT polarization vector directly on the detector grid
- this pass:
  - attempted
- current disposition:
  - rejected for the maintained authority surface
- reason:
  - no-rotation authority timing regressed, and the mixed-angle timing again
    showed an implausible `E` collapse that should not be treated as a clean
    accepted win

### 7. Add a higher-memory multi-angle cache mode

- idea:
  - cache `phi_a` plus orientation components across the angle loop so the
    direct path can trade memory for multi-angle throughput
- this pass:
  - attempted
- current disposition:
  - rejected for now, but remains the most interesting retry candidate
- reason:
  - the mixed-angle timing looked plausibly better in Segment `B`, but the
    isolated rotation parity check crashed, so it failed the acceptance gate

### 8. Add a narrower orientation-only cache mode

- idea:
  - cache only `sx`, `sy`, `sz` across angles or energies as a smaller-memory
    compromise relative to the fuller cache mode
- this pass:
  - not attempted
- current disposition:
  - still a potential improvement
- reason:
  - it remains a reasonable follow-on experiment if a future retry wants a
    lower-memory variant of item `7`

### Current retry posture

After this pass:

1. none of the attempted items were accepted,
2. all eight items remain recorded as potential improvements,
3. and any next pass should be run as an explicitly higher-reasoning
   investigation rather than as a blind continuation of the same ranking loop.

## April 2 2026 Resume Proposal

This section records the next proposed retry plan after a fresh code and
artifact review on April 2, 2026.

Treat this section as the authoritative resume package for the next
`direct_polarization` pass.

The earlier "Ranked Next Optimization Candidates" section above remains useful
as historical context, but the order below should be used for the next retry.

### Current code-grounded interpretation

Current review of `src/NRSS/backends/cupy_rsoxs.py` indicates:

1. Segment `B` is still the primary direct-path optimization target on the
   maintained small host-prewarmed no-rotation authority lane.
2. The current `direct_polarization` implementation still recomputes
   per-material optical scalars inside `_compute_direct_polarization(...)` for
   every angle:
   - `_material_optical_scalars(...)`
   - `_material_optics(...)`
3. The current implementation also recomputes orientation trigonometry for
   every anisotropic material on every angle:
   - `_orientation_components(...)`
4. Inside `_compute_direct_polarization(...)`, the hot loop still materializes
   several large temporaries per material per angle:
   - `isotropic_term`
   - `phi_a`
   - `sx`, `sy`, `sz`
   - `field_projection`
5. Segment `D` still materializes full `scatter3d` before detector projection,
   but on the current no-rotation authority surface it is secondary rather
   than primary.

### Methodology caveat from the April 2 review

The previous raw-kernel false-negative risk remains real.

Current harness behavior:

1. `--cuda-prewarm before_prepare_inputs` only absorbs first-touch
   host-to-CuPy bring-up before the timed boundary.
2. It does not precompile custom `RawKernel` code.
3. Therefore the current harness is not a fully-hot authority surface for
   kernel-heavy experiments.
4. Any retry of item `4` or item `5` below should not be rejected on the basis
   of cold compile/load cost alone without a separate fully-hot measurement
   surface.

### Environment blocker observed during the April 2 review

Before resuming optimization work, first confirm the active GPU toolchain is
internally consistent.

Observed April 2, 2026 facts:

1. visible GPUs:
   - three Quadro RTX 8000 devices
2. active Python env used for the review:
   - `/home/deand/mambaforge/envs/nrss-dev/bin/python`
3. CuPy version in that env:
   - `14.0.1`
4. direct GPU parity probing in that env failed before runtime physics checks:
   - first with `failed to open libnvrtc-builtins.so.11.8`
   - then, after injecting an older CUDA library path, with
     `CUDA versions below 12 are not supported`

Interpretation:

1. the current shell-visible NVRTC runtime is not aligned with the CuPy 14
   expectations of `nrss-dev`,
2. so do not treat direct-path parity or timing failures from this broken
   state as optimization evidence,
3. and repair or explicitly select a working CUDA/NVRTC runtime before the
   next optimization pass.

### April 3 2026 progress update

Current status after resuming work on April 3, 2026:

1. the April 2 runtime blocker is no longer active:
   - `/home/deand/mambaforge/envs/nrss-dev/bin/python`
   - CuPy `14.0.1`
   - three visible Quadro RTX 8000 GPUs
   - tiny CuPy arithmetic plus a tiny JIT-backed `ElementwiseKernel` probe
     both succeeded in `nrss-dev`
2. the dev harness now has a fully-hot measurement mode:
   - `--worker-warmup-runs N`
   - this keeps the existing cold-subprocess authority path unchanged at
     `N=0`
   - for kernel-heavy experiments, `N=1` performs one untimed identical warm-up
     run inside the worker subprocess before the timed boundary
3. maintained CoreShell sim-regression coverage for
   `backend_options={'execution_path': 'direct_polarization'}` has been wired
   into `tests/validation/test_core_shell_reference.py`
   - the maintained direct-path gate now targets the device-resident borrowed
     CuPy CoreShell surface because the host-resident direct-path regression
     was too slow to use as a practical iterative acceptance command
4. current policy for this direct-path pass is stricter than the April 2 retry
   list:
   - keep `direct_polarization` low-memory by default
   - defer any experiment that caches `3D` or larger arrays, including
     `phi_a`, `sx/sy/sz`, or fuller multi-angle volume caches
   - keep custom kernels in scope

### April 3 2026 first A/B experiment

Experiment:

1. narrowed item `1`
   - pre-split isotropic versus anisotropic materials once per run
   - precompute per-energy optical scalars once per energy
   - no `3D` cache added

Same-GPU A/B artifacts on `CUDA_VISIBLE_DEVICES=2`:

1. baseline cold:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_ab_baseline_cold_gpu2_seq_20260403/summary.json`
2. experiment cold:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_ab_exp1_cold_gpu2_seq_20260403/summary.json`
3. baseline hot:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_ab_baseline_hot_gpu2_seq_20260403/summary.json`
4. experiment hot:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_ab_exp1_hot_gpu2_seq_20260403/summary.json`

Measured outcome:

1. cold no-rotation lane:
   - `primary 0.197 s -> 0.208 s`
   - regression of about `+5.6%`
2. cold `0:5:165` lane:
   - `primary 1.236 s -> 1.229 s`
   - effectively flat at about `-0.6%`
3. hot no-rotation lane:
   - `primary 0.118 s -> 0.117 s`
   - effectively flat at about `-0.8%`
4. hot `0:5:165` lane:
   - `primary 1.148 s -> 1.135 s`
   - effectively flat at about `-1.1%`
5. hot `B` segment on `0:5:165`:
   - `0.725 s -> 0.725 s`
   - no meaningful change

Disposition:

1. reject the narrowed item `1` experiment
2. keep the backend code at the pre-experiment direct-path baseline
3. do not spend more time on pure scalar-hoist / material-split refactors
   unless a later kernel rewrite can reuse them as scaffolding
4. next active speed candidate under the current low-memory policy is:
   - fully fused Segment `B` custom-kernel work measured on the fully-hot
     harness surface

### April 3 2026 second A/B experiment

Experiment:

1. detector-plane direct projection path for Segment `D`
   - direct-path-only rewrite
   - no `3D` cache added
   - bypass full `scatter3d` materialization in the direct path

Same-GPU A/B artifacts on `CUDA_VISIBLE_DEVICES=2`:

1. baseline cold:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_ab_baseline_cold_gpu2_seq_20260403/summary.json`
2. experiment cold:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_cand_d_directproj_cold_gpu2_seq_20260403/summary.json`
3. baseline hot:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_ab_baseline_hot_gpu2_seq_20260403/summary.json`
4. experiment hot:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_cand_d_directproj_hot_gpu2_seq_20260403/summary.json`

Measured outcome:

1. cold no-rotation lane:
   - `primary 0.197 s -> 0.202 s`
   - regression of about `+2.5%`
2. cold `0:5:165` lane:
   - `primary 1.236 s -> 1.232 s`
   - effectively flat at about `-0.3%`
3. hot no-rotation lane:
   - `primary 0.118 s -> 0.121 s`
   - regression of about `+2.5%`
4. hot `0:5:165` lane:
   - `primary 1.148 s -> 1.156 s`
   - regression of about `+0.7%`
5. hot `D` segment on `0:5:165`:
   - `0.223 s -> 0.229 s`
   - regression of about `+2.7%`

Disposition:

1. reject the detector-plane direct projection experiment
2. keep the backend code at the pre-experiment direct-path baseline
3. do not spend more time on detector-plane `D` work for `direct_polarization`
   unless a materially different formulation is available
4. next active low-memory candidates remain:
   - aligned-family custom kernels for Segment `B`
   - generic fused Segment `B` custom-kernel work

### April 3 2026 third A/B experiment

Experiment:

1. aligned-family custom kernels for Segment `B`
   - direct-path-only raw-kernel rewrite for aligned `x` / `y` families
   - general-angle route kept on the existing Python/CuPy path
   - no `3D` cache added

Same-GPU A/B artifacts on `CUDA_VISIBLE_DEVICES=2`:

1. baseline cold:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_ab_baseline_cold_gpu2_seq_20260403/summary.json`
2. experiment cold:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_cand_b_alignedkernel_cold_gpu2_seq_20260403/summary.json`
3. baseline hot:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_ab_baseline_hot_gpu2_seq_20260403/summary.json`
4. experiment hot:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_cand_b_alignedkernel_hot_gpu2_seq_20260403/summary.json`

Measured outcome:

1. cold no-rotation lane:
   - `primary 0.197 s -> 0.258 s`
   - regression of about `+31.0%`
2. cold `0:5:165` lane:
   - `primary 1.236 s -> 1.282 s`
   - regression of about `+3.7%`
3. hot no-rotation lane:
   - `primary 0.118 s -> 0.108 s`
   - improvement of about `-8.5%`
4. hot `0:5:165` lane:
   - `primary 1.148 s -> 1.125 s`
   - improvement of about `-2.0%`
5. hot `B` segment on no-rotation:
   - `0.021 s -> 0.007 s`
   - improvement of about `-66.7%`
6. hot `B` segment on `0:5:165`:
   - `0.725 s -> 0.697 s`
   - improvement of about `-3.9%`

Disposition:

1. reject the aligned-family kernel experiment as a default-path change
2. reason:
   - it is a real fully-hot no-rotation win,
   - but it regresses the maintained cold authority lane badly enough that it
     does not qualify for unconditional adoption
3. no parity run was used for acceptance because the timing gate already failed
   for the maintained default surface
4. this remains useful evidence that fused direct-path kernels can pay off once
   compile/load cost is isolated or amortized
5. next active candidate:
   - generic fused Segment `B` custom-kernel work

### April 3 2026 fourth A/B experiment

Experiment:

1. generic fused Segment `B` custom-kernel rewrite
   - direct-path-only raw-kernel rewrite for the anisotropic contribution
   - no `3D` cache added
   - isotropic terms remain on the existing path

Same-GPU A/B artifacts on `CUDA_VISIBLE_DEVICES=2`:

1. baseline cold:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_ab_baseline_cold_gpu2_seq_20260403/summary.json`
2. experiment cold:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_cand_b_generickernel_cold_gpu2_seq_20260403/summary.json`
3. baseline hot:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_ab_baseline_hot_gpu2_seq_20260403/summary.json`
4. experiment hot:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_cand_b_generickernel_hot_gpu2_seq_20260403/summary.json`

Measured outcome:

1. cold no-rotation lane:
   - `primary 0.197 s -> 0.261 s`
   - regression of about `+32.5%`
2. cold `0:5:165` lane:
   - `primary 1.236 s -> 0.786 s`
   - improvement of about `-36.4%`
3. hot no-rotation lane:
   - `primary 0.118 s -> 0.108 s`
   - improvement of about `-8.5%`
4. hot `0:5:165` lane:
   - `primary 1.148 s -> 0.701 s`
   - improvement of about `-38.9%`
5. hot `B` segment on `0:5:165`:
   - `0.725 s -> 0.275 s`
   - improvement of about `-62.1%`

Parity evidence used for this pass:

1. maintained isotropic execution-path smoke:
   - `pytest tests/smoke/test_smoke.py -k "test_cupy_execution_paths_and_isotropic_representations_match_on_fully_isotropic_morphology" -v`
   - passed
2. new anisotropic execution-path smoke:
   - `pytest tests/smoke/test_smoke.py -k "test_cupy_direct_polarization_matches_tensor_coeff_on_anisotropic_sphere" -v`
   - passed
   - covers both `EAngleRotation=[0, 0, 0]` and `EAngleRotation=[0, 5, 165]`
3. additional CoreShell helper comparison against `tensor_coeff` on the active
   timing surface:
   - no-rotation:
     - `max_abs 0.046875`
     - `max_rel ~= 5.36e-05`
     - `mean_abs ~= 1.71e-06`
   - `0:5:165`:
     - `max_abs 0.03125`
     - `max_rel ~= 2.15e-06`
     - `mean_abs ~= 1.48e-06`
4. note on the slower maintained direct-path CoreShell regression:
   - the dedicated pytest-facing direct-path sim-regression hook is wired, but
     remained too slow to serve as a practical iterative acceptance command
     during this pass

Disposition:

1. accept the generic fused Segment `B` custom-kernel rewrite
2. reason:
   - it exceeds the `5%` speed gate decisively on the maintained
     host-prewarmed `0:5:165` lane,
   - it also improves the fully-hot no-rotation lane,
   - and the parity evidence above did not reveal a meaningful physics drift
3. interpretation:
   - this is an accepted `0:5:165`-oriented speed win for the low-memory
     direct path,
   - it is not a clean cold no-rotation win because first-use compile/load
     cost is still visible there
4. current next-step implication:
   - with this kernel accepted, no higher-priority low-memory direct-path
     candidate remains from the current list

### Revised retry order for the next pass

The next retry should use the following order.

#### 0. Add a fully-hot timing mode to the dev harness

- status:
  - new prerequisite, not one of the recorded eight experiments
- goal:
  - separate true steady-state kernel performance from first-use compile/load
    cost
- why first:
  - current raw-kernel evidence is confounded by the current subprocess timing
    boundary
- recommended implementation shape:
  - keep the current cold-subprocess authority path intact
  - add a development-only mode that performs one untimed identical warm-up
    run inside the worker subprocess before `primary_start`
  - use this mode only for evaluating kernel-heavy direct-path candidates

#### 1. Retry recorded item `1` in a narrower form

- original item:
  - hoist cheap per-energy / per-run work out of the hot angle loop
- revised focus:
  - pre-split isotropic and anisotropic material lists once per run
  - precompute per-energy optical scalars once per energy
- why it moved up:
  - current code still pays both costs inside the hot angle loop
  - this is low-risk and does not require large additional arrays
- expected upside:
  - modest but credible speedup on both `0°` and `0:5:165`

#### 2. Add a new `phi_a` cache experiment

- status:
  - new opportunity identified during the April 2 review
- goal:
  - cache `phi_a = Vfrac * S` once per anisotropic material per energy rather
    than once per angle
- why it moved near the top:
  - current code performs this full-volume multiply inside every angle loop
  - this should be cheaper than a full orientation cache while still removing
    meaningful Segment `B` work
- expected upside:
  - moderate on both `0°` and `0:5:165`
- risk:
  - low to moderate

#### 3. Move recorded item `8` up

- original item:
  - narrower orientation-only cache mode
- revised interpretation:
  - cache `sx`, `sy`, `sz` across angles, and possibly across energies only if
    energy reuse is still clean
- why it moved up:
  - current code recomputes trigonometric orientation components for every
    angle even though they are morphology-derived and angle-independent
- expected upside:
  - material Segment `B` savings, especially on `0:5:165`
- tradeoff:
  - memory increase, but smaller than the fuller item `7` cache

#### 4. Retry recorded item `7`

- original item:
  - higher-memory multi-angle cache mode
- revised interpretation:
  - cache `phi_a` plus orientation components across the angle loop as an
    explicit higher-memory fast mode
- why it stays high:
  - this remains the most promising direct-path throughput idea for
    `0:5:165`
- why it is not first:
  - lower-memory wins should be tested first
  - previous attempt failed parity by crash, so correctness risk remains real

#### 5. Retry recorded item `5` only after the fully-hot harness exists

- original item:
  - fully fused custom-kernel rewrite for `_compute_direct_polarization(...)`
- why it stays important:
  - this still has the highest theoretical Segment `B` upside
- why it is not first:
  - the current evidence is contaminated by compile/load cost on the existing
    cold subprocess timing surface

#### 6. Retry recorded item `6` after the `B` retries

- original item:
  - detector-plane direct projection path
- why it moved down:
  - current clean no-rotation authority timing says `D` is secondary
- retry guidance:
  - if retried, prefer adapting the accepted detector-grid helper style from
    `tensor_coeff` rather than inventing a wholly separate projection algebra
    path

#### 7. Push recorded items `2` and `3` down

- item `2`:
  - pure-CuPy Segment `B` scratch reuse
- item `3`:
  - explicit `x` / `y` / general-angle Python-CuPy route split
- why both moved down:
  - both already regressed on the relevant maintained host-prewarmed lanes
  - current code review does not reveal a compelling reason to expect a
    different outcome before the higher-value cache/hoist ideas are tested

#### 8. Keep recorded item `4` below the generic fused-kernel retry

- original item:
  - aligned-family custom kernels
- why it moved down:
  - the aligned-family prototype already suffered the exact cold-kernel timing
    pathology we now suspect
  - if a fully-hot kernel retry is going to happen, the generic Segment `B`
    rewrite is the more important one

### Proposed testing series

Use the following sequence for the next retry campaign.

#### Phase 1. Repair and confirm the runtime

1. verify GPU visibility with:
   - `nvidia-smi -L`
2. verify the selected Python env can import CuPy and report device count
3. verify a tiny CuPy operation and a tiny JIT/NVRTC-backed operation both run
   successfully in the exact env that will be used for optimization work
4. do not begin optimization experiments until the NVRTC mismatch described
   above is resolved

#### Phase 2. Capture fresh baselines

After the runtime is repaired, capture fresh `direct_polarization` baselines on
both:

1. the maintained small host-prewarmed lanes, for continuity with the earlier
   path note,
2. and the maintained small device-hot lane, which is now the acceptance
   authority for new direct-path code paths.

Required baseline command:

```bash
CUDA_VISIBLE_DEVICES=0 /home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py \
  --label dp_baseline_resume \
  --size-labels small \
  --resident-modes host \
  --execution-paths direct_polarization \
  --cuda-prewarm before_prepare_inputs \
  --rotation-specs '0:5:165' \
  --timing-segments all
```

Interpret the resulting `summary.json` as the new baseline for:

1. small host-prewarmed no-rotation
2. small host-prewarmed `0:5:165`

Optional quick-rank companion:

```bash
CUDA_VISIBLE_DEVICES=0 /home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py \
  --label dp_device_rank_resume \
  --size-labels small \
  --resident-modes device \
  --execution-paths direct_polarization \
  --cuda-prewarm before_prepare_inputs \
  --rotation-specs '0:5:165' \
  --timing-segments all
```

Required device-hot acceptance baseline:

```bash
CUDA_VISIBLE_DEVICES=0 /home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py \
  --label dp_device_hot_acceptance_baseline \
  --size-labels small \
  --resident-modes device \
  --execution-paths direct_polarization \
  --rotation-specs '0:5:165' \
  --timing-segments all \
  --worker-warmup-runs 1
```

Recommended no-rotation device-hot companion:

```bash
CUDA_VISIBLE_DEVICES=0 /home/deand/mambaforge/envs/nrss-dev/bin/python \
  tests/validation/dev/cupy_rsoxs_optimization/run_cupy_rsoxs_optimization_matrix.py \
  --label dp_device_hot_acceptance_baseline_no_rotation \
  --size-labels small \
  --resident-modes device \
  --execution-paths direct_polarization \
  --rotation-specs '0:0:0' \
  --timing-segments all \
  --worker-warmup-runs 1
```

Required memory baseline methodology:

1. record peak memory on the same small device-hot `0:5:165` surface in a
   separate pass using the external GPU-polling method already implemented in
   `tests/validation/dev/core_shell_backend_performance/run_comprehensive_backend_comparison.py`
2. use the `device hot` direct-path row from that report as the acceptance
   memory baseline for any candidate direct-path code change
3. compare candidate peak GPU memory against the current maintained
   `direct_polarization` device-hot row, not against legacy `cyrsoxs` and not
   against `tensor_coeff`

#### Phase 3. Add missing parity coverage before accepting wins

The next retry should not rely only on raw image spot-checks.

Required parity gates:

1. maintained CoreShell sim-regression surface:
   - run the existing CoreShell regression with
     `backend_options={'execution_path': 'direct_polarization'}`
   - this should be wired into a maintained pytest-facing path before major
     direct-path optimization changes are accepted
2. explicit `0:5:165` rotation parity surface:
   - add a direct-path versus `tensor_coeff` parity check on a deliberately
     asymmetric morphology
   - use a nontrivial `EAngleRotation`, at minimum `0:5:165`
   - this is needed because the maintained CoreShell A-wedge workflow uses
     `EAngleRotation=[0.0, 1.0, 360.0]`, while the optimization acceptance lane
     for this pass includes `0:5:165`
3. crash-free execution is part of parity:
   - any experiment that crashes on the rotation parity surface fails
     acceptance even if timing looks good

#### Phase 4. Run experiments in this order

Run one implementation change at a time.

1. narrowed item `1`
   - per-run isotropic/anisotropic split plus per-energy optical-scalar hoist
2. new `phi_a` cache experiment
3. item `8`
   - orientation-only cache
4. item `7`
   - higher-memory multi-angle cache mode
5. item `5`
   - fully fused Segment `B` kernel rewrite
6. item `6`
   - detector-plane direct projection path

Defer unless the above disappoint:

1. item `2`
2. item `3`
3. item `4`

## Fresh Optimization Opportunities From The April 4 Code Comparison

This section supersedes the older retry ranking above when resuming from a new
context.

The list below was derived from a direct comparison of:

1. the maintained NRSS `direct_polarization` path in
   `src/NRSS/backends/cupy_rsoxs.py`,
2. the maintained NRSS `tensor_coeff` path in the same file,
3. and the legacy CyRSoXS C++ / CUDA direct and low-memory tensor routes in
   `src/cudaMain.cu` and `include/uniaxial.h`.

The implementation references below are the specific code anchors that
motivated each opportunity.

### 1. Complete the remaining float32 Segment `B` fusion

- current NRSS state:
  - the accepted generic fused direct-path kernel only fuses the anisotropic
    contribution for the float32 path
  - the float32 path still materializes `isotropic_term = vfrac *
    isotropic_diag` and still updates `p_x` / `p_y` from the outer Python
    material loop
  - current anchors:
    - `src/NRSS/backends/cupy_rsoxs.py::_compute_direct_polarization(...)`
    - `src/NRSS/backends/cupy_rsoxs.py::_direct_generic_kernel_float32(...)`
- inspiration:
  - the existing half-input direct kernels already split isotropic and
    anisotropic work into dedicated kernels:
    - `src/NRSS/backends/cupy_rsoxs.py::_direct_isotropic_kernel_float16(...)`
    - `src/NRSS/backends/cupy_rsoxs.py::_direct_anisotropic_kernel_float16(...)`
- resume guidance:
  - add float32 isotropic-only and float32 isotropic-plus-anisotropic kernels
  - the first target is to eliminate the float32 `isotropic_term` full-volume
    temporary without changing the current low-memory default character
  - this is the lowest-risk remaining Segment `B` cleanup because it requires
    no persistent cache

### 2. Fuse across materials per voxel in the direct path, CyRSoXS-style

- current NRSS state:
  - `_compute_direct_polarization(...)` loops over materials from Python and
    repeatedly updates the same full-volume `p_x`, `p_y`, and `p_z` arrays
- inspiration from CyRSoXS:
  - `include/uniaxial.h::computePolarizationEulerAngles(...)` loops over
    `NUM_MATERIAL` inside the device computation for one voxel
  - the kernel accumulates in registers and writes one final polarization
    vector per voxel
  - execution path entry:
    - `src/cudaMain.cu::computePolarization(...)`
- resume guidance:
  - prototype a direct-path kernel that owns the material loop internally
  - stage per-energy optical constants into compact device arrays and have one
    voxel thread accumulate all material contributions before one write-back
  - this is the closest structural emulation of the default CyRSoXS direct
    path and is the most important missing compute strategy not yet mirrored in
    NRSS
  - acceptance should treat this as a direct-path low-memory candidate unless
    it requires explicit cached intermediates

### 3. Add a per-energy isotropic base-field cache

- current NRSS state:
  - isotropic work is recomputed inside every angle loop even though the only
    angle dependence is the final multiplication by `mx` / `my`
  - current anchor:
    - `src/NRSS/backends/cupy_rsoxs.py::_compute_direct_polarization(...)`
- inspiration:
  - `tensor_coeff` already invests in building angle-independent fields once
    per energy before the angle-family projection and rotation steps:
    - `src/NRSS/backends/cupy_rsoxs.py::_run_single_energy_tensor_coeff(...)`
    - `src/NRSS/backends/cupy_rsoxs.py::_compute_nt_components(...)`
- resume guidance:
  - cache one angle-independent complex64 isotropic field per energy:
    - `p_iso(r) = sum_m Vfrac_m(r) * isotropic_diag_m(E)`
  - then each direct-path angle can apply:
    - `p_x += mx * p_iso`
    - `p_y += my * p_iso`
  - this is smaller than caching orientation fields and specifically targets
    direct-path work that the current float32 fused kernel still leaves on the
    hot path
  - expected memory impact is one extra complex64 `3D` field per energy while
    the angle loop is active

### 4. Fuse windowing into direct polarization generation

- current NRSS state:
  - the direct path applies the FFT window in `_fft_polarization_fields(...)`
    as three separate full-volume multiplies over `p_x`, `p_y`, and `p_z`
- inspiration from CyRSoXS:
  - `src/cudaMain.cu::computePolarization(...)` applies the Hanning weights
    inside the polarization kernel before the FFT
- resume guidance:
  - keep the current no-window fast path untouched
  - when windowing is active, either:
    - multiply the final voxel contributions by the precomputed scalar window
      inside the direct kernel,
    - or emit a dedicated windowed direct kernel variant
  - the goal is to remove one extra read/write pass over all three
    polarization volumes
  - this should be treated as a low-memory opportunity because it swaps a
    separate pass for fused arithmetic rather than adding a cache

### 5. Revisit detector-plane direct projection only with a materially different formulation

- current NRSS state:
  - the maintained direct path still does:
    - `_projection_from_fft_polarization(...)`
    - `_compute_scatter3d(...)`
    - `_project_scatter3d(...)`
  - the April 3 detector-plane direct-projection attempt regressed and was
    rejected
- inspiration from CyRSoXS:
  - the `ScatterApproach::PARTIAL` route projects directly from FFT
    polarization on the detector grid and never materializes a separate
    `scatter3D` volume
  - anchors:
    - `src/cudaMain.cu` branch that calls
      `peformEwaldProjectionGPU(d_projection, d_polarizationX, d_polarizationY, d_polarizationZ, ...)`
    - `include/uniaxial.h::computeEwaldProjectionGPU(...)` overload taking
      polarization arrays
- additional inspiration from `tensor_coeff`:
  - the accepted detector-grid helper path:
    - `src/NRSS/backends/cupy_rsoxs.py::_projection_coefficients_from_fft_nt(...)`
    - `src/NRSS/backends/cupy_rsoxs.py::_projection_coefficients_from_fft_pair(...)`
- resume guidance:
  - do not retry the earlier rejected direct-projection idea as another broad
    CuPy algebra rewrite
  - if retried, either:
    - port the CyRSoXS partial-projection kernel structure more literally,
    - or adapt the accepted `tensor_coeff` detector-grid helper style to the
      direct polarization basis
  - this opportunity remains valid because it is a real CyRSoXS compute
    strategy, but only a materially different implementation should be tested

### 6. Add angle tiling so one morphology pass serves multiple angles

- current NRSS state:
  - `_project_from_direct_polarization(...)` rebuilds polarization, FFTs, and
    projects one angle at a time
  - the direct-path note already shows poor scaling on dense angle sweeps
- inspiration:
  - `tensor_coeff` wins multi-angle work by reusing angle-independent fields
    and only recombining angle families later
  - CyRSoXS still loops per angle, but its direct voxel kernel already owns the
    material accumulation, which makes tiling more plausible there than in the
    current NRSS outer-Python material loop
- resume guidance:
  - test a tiled direct kernel that computes a small fixed set of angles per
    voxel pass, for example `2-8` angles per launch
  - the goal is to amortize:
    - `Vfrac/S/theta/psi` reads
    - trigonometric decode
    - per-material optical arithmetic
  - keep the tile size small enough to limit register pressure and avoid
    exploding resident memory
  - this is a fresh multi-angle throughput idea distinct from the previously
    recorded persistent morphology caches

### 7. If a higher-memory mode is allowed, prefer `phi_a` cache before full orientation cache

- current NRSS state:
  - the direct path recomputes `phi_a = Vfrac * S` inside each anisotropic
    angle pass
  - the note already identified this as a promising deferred cache candidate
- inspiration:
  - the current code makes it clear `phi_a` is angle-independent within an
    energy:
    - `src/NRSS/backends/cupy_rsoxs.py::_compute_direct_polarization(...)`
    - `src/NRSS/backends/cupy_rsoxs.py::_compute_direct_polarization_collapsed_mean(...)`
  - `tensor_coeff` similarly benefits from building angle-independent fields
    before the angle-family path
- resume guidance:
  - keep this as a staged cache ladder:
    1. cache `phi_a` only
    2. only if needed, expand to the narrower orientation cache in item `8`
    3. only after that, consider fuller multi-angle caches
  - this should be treated as an explicit higher-memory candidate, not as the
    default low-memory direct path
  - for any accepted version, record peak GPU memory against the maintained
    direct baseline on the same device-hot lane

### 8. Add an orientation cache mode before any fuller multi-angle cache

- current NRSS state:
  - `_orientation_components(...)` recomputes `sin(theta)`, `cos(theta)`,
    `cos(psi)`, and `sin(psi)` for every anisotropic material on every angle
- inspiration:
  - the note already recognized the value of caching `sx`, `sy`, and `sz`
  - `tensor_coeff` effectively cashes in on the same idea by forming
    angle-independent tensor ingredients once and reusing them across all angle
    families
- resume guidance:
  - start with the narrowest useful cache:
    - `sx`, `sy`, `sz`
  - only if that is still too expensive, consider caching the even more direct
    basis products used by both direct and tensor routes:
    - `sx*sx`
    - `sx*sy`
    - `sx*sz`
    - `sy*sy`
    - `sy*sz`
  - this mode should remain opt-in because its memory cost is real even though
    it is smaller than caching full per-angle polarization or full multi-angle
    intermediates

### 9. Treat kernel warm-up and Segment `C` buffer reuse as first-class direct-path opportunities

- current NRSS state:
  - direct-path kernel-heavy experiments are confounded by first-use
    compile/load cost on the cold subprocess surface
  - the direct path also still allocates fresh FFT outputs and shifted arrays
    in `_fft_polarization_fields(...)`
- inspiration:
  - the note already shows aligned and generic direct kernels become much more
    attractive on the fully-hot surface
  - `tensor_coeff` already reuses storage more aggressively in
    `_compute_fft_nt_components(...)` by shifting into `nt[idx]`
- resume guidance:
  - always rank custom-kernel direct-path candidates on the fully-hot device
    surface with `--worker-warmup-runs 1`
  - separately, test whether direct-path Segment `C` can reuse one or more
    existing polarization or FFT buffers the same way `tensor_coeff` reuses the
    `nt` storage
  - do not count warm-up alone as a math optimization, but do treat:
    - eager kernel initialization,
    - pre-JIT of maintained kernels,
    - and direct-path FFT/shift buffer reuse
    as real implementation opportunities when deciding which direct-path code
    path should be maintained

### Recommended experiment order for the nine fresh opportunities

Use the following order for the next direct-path pass unless a new benchmark or
correctness result clearly changes the ranking:

1. item `1`
   - complete the remaining float32 isotropic fusion
2. item `2`
   - CyRSoXS-style all-material fused voxel kernel
3. item `3`
   - per-energy isotropic base-field cache
4. item `4`
   - fuse windowing into Segment `B`
5. item `6`
   - angle tiling for multi-angle throughput
6. item `7`
   - `phi_a` cache mode
7. item `8`
   - orientation cache mode
8. item `5`
   - materially different direct detector-plane projection retry
9. item `9`
   - direct-path kernel warm-up / eager-init plus Segment `C` buffer reuse

Interpretation notes:

1. items `1`, `2`, `4`, and the buffer-reuse part of item `9` are the best
   low-memory default-path candidates
2. items `3`, `7`, and `8` are memory-tradeoff candidates and should be
   treated as explicit opt-in modes unless they clear the memory gate below
   comfortably enough to justify default adoption
3. item `5` should be retried only if the implementation is recognizably
   closer to either:
   - the CyRSoXS partial-projection kernel shape,
   - or the accepted `tensor_coeff` detector-grid helper style
4. item `6` is the main fresh idea for `0:5:165` throughput that does not rely
   on a persistent morphology cache

### Deferred experiment: precompile maintained kernels before `A2`

This is a recorded deferred hypothesis, not an accepted change.

Question to test later:

1. can whole-worker peak GPU memory be reduced further by forcing
   compilation/loading of known compilable kernels before the heavy direct-path
   `A2` through `D` work begins,
2. while preserving or improving the maintained device-hot `0:5:165` primary
   timing lane?

Current motivation:

1. the April 5-6 detector-projection work strongly suggested that some of the
   earlier cold-process peak came from first-use compile/load activity rather
   than from steady-state direct-path working set,
2. CuPy pool reuse alone does not prove the worker-lifetime peak will fall,
   because a whole-worker external probe still records the maximum point across
   the process lifetime,
3. therefore an early precompile step is only promising if it reduces overlap
   between compile-time transients and the later large `B-D` allocations, or if
   the compile-time transient can be made to subside before those later
   allocations begin.

Important caveat:

1. an explicit `cp.cuda.Stream.null.synchronize()` before the heavy path may be
   needed to establish a clean boundary, but synchronization latency is itself
   a real speed risk and must be measured rather than assumed away,
2. similarly, freeing pool blocks after precompile may help the external peak,
   but it can also perturb latency enough that it should be treated as a test
   dimension, not as an assumed default,
3. without such a boundary, a precompile step may merely move the compile spike
   earlier in the worker lifetime rather than lower the measured peak.

Scope to test later:

1. start with maintained custom-kernel paths that are explicitly compilable and
   already known to matter on the direct path:
   - direct detector projection kernels,
   - direct polarization fused kernels,
   - Igor shift kernel,
2. do not initially broaden the experiment to every possible CuPy JIT path,
   because generic ufunc / elementwise JIT is harder to pre-stage
   deterministically and would blur the result.

Recorded follow-up priority:

1. after the detector-projection `nvcc` win, explicitly examine the other
   maintained direct-path custom kernels to see whether they also behave better
   under `nvcc` than under `nvrtc`,
2. rank that work below the currently accepted detector-projection `nvcc`
   change, but above broad speculative kernel-JIT cleanup,
3. do not assume the answer generalizes from one kernel family to another;
   record speed and whole-worker peak-memory results per kernel family.

Recommended experiment matrix for a future pass:

1. baseline:
   - current maintained worker behavior
2. variant `P1`:
   - startup precompile of the maintained direct-path custom kernels before the
     timed run
   - no explicit synchronize
   - no pool release
3. variant `P2`:
   - same precompile step
   - explicit synchronize before entering the heavy path
4. variant `P3`:
   - same as `P2`
   - then free CuPy default and pinned pools before entering the heavy path

Acceptance evidence to capture if this is resumed:

1. maintained device-hot acceptance lane:
   - small CoreShell
   - `resident_mode='device'`
   - `execution_path='direct_polarization'`
   - `EAngleRotation=[0, 5, 165]`
   - `--worker-warmup-runs 1`
2. device-hot no-rotation companion
3. external whole-worker peak GPU memory on the same `0:5:165` lane
4. if speed and memory disagree, also capture a warmed repeated-run probe in
   one long-lived subprocess to separate:
   - whole-worker cold peak,
   - from warmed steady-state working set

Decision rule if resumed:

1. keep it only if the measured whole-worker peak GPU memory actually falls on
   the maintained `0:5:165` lane,
2. and do not keep it if any required synchronize / pool-release boundary gives
   back too much primary-time improvement,
3. do not assume success just because compile-time allocations are expected to
   return to the CuPy pool.

### April 6 2026 preload plus per-kernel-backend follow-up

The deferred preload question and the per-kernel-family `nvcc` versus `nvrtc`
question were both resumed on April 6, 2026.

Artifact root:

- `test-reports/core-shell-backend-performance-dev/kernel_preload_backend_matrix_20260406/`

Measured matrix:

1. maintained hot authority lane:
   - small CoreShell
   - `resident_mode='device'`
   - single energy
   - `EAngleRotation=[0, 5, 165]`
   - `--worker-warmup-runs 1`
2. explicit preload-stage variants:
   - `off`
   - `A2`
   - `A1`
3. direct-path kernel-family backend variants:
   - `igor_shift`: `nvrtc` or `nvcc`
   - `direct_polarization_generic`: `nvrtc` or `nvcc`
4. detector projection kernels were not reopened as a backend-choice question:
   - they remained on the already accepted `nvcc`-preferred path with
     `nvrtc` fallback
5. separate external whole-worker peak GPU memory pass was run on the same
   matrix
6. a subprocess-isolated hot no-rotation companion was also captured for the
   shortlisted winner

Current direct-path ranking outcome from that matrix:

1. absolute fastest hot `0:5:165` row:
   - `A1 / igor nvcc / direct nvcc`
   - `primary 0.27088 s`
   - peak GPU memory about `679 MiB`
2. accepted maintained winner:
   - `A1 / igor nvcc / direct nvrtc`
   - `primary 0.27103 s`
   - peak GPU memory about `623 MiB`
3. old direct-path baseline for comparison:
   - `off / igor nvrtc / direct nvrtc`
   - `primary 0.27215 s`
   - peak GPU memory about `677 MiB`

Interpretation:

1. constructor-time preload is now justified for the maintained direct path
   because it preserved hot-lane speed while lowering whole-worker peak memory
2. `igor_shift` behaves better for the direct path under `nvcc` than under
   `nvrtc`
3. the default `float32` `direct_polarization_generic` kernel should stay on
   `nvrtc`
4. the detector-projection kernels should remain on the already accepted
   `nvcc`-preferred path
5. this is a path-specific mixed backend-family result rather than evidence
   that every maintained custom kernel should move to `nvcc`

Accepted implementation defaults after this pass:

1. plain `backend_options={'execution_path': 'direct_polarization'}` now
   resolves to:
   - `kernel_preload_stage='a1'`
   - `igor_shift_backend='nvcc'`
   - `direct_polarization_backend='nvrtc'`
2. if `nvcc` is unavailable, the maintained custom-kernel factories fall back
   to `nvrtc`
3. the direct detector-projection kernels still prefer `nvcc` when it is
   discoverable and otherwise fall back to `nvrtc`

No-rotation companion result:

1. old direct-path baseline:
   - `primary 0.01224 s`
2. accepted winner:
   - `primary 0.01058 s`

Validation completed for the accepted winner:

1. `PYTHONPATH=/homes/deand/dev/NRSS mamba run -n nrss-dev python -m pytest tests/smoke/test_smoke.py -k 'test_cupy_direct_polarization_matches_tensor_coeff_on_anisotropic_sphere or test_cupy_direct_polarization_host_and_device_residency_parity or test_cupy_tensor_coeff_host_and_device_residency_parity' -v`
   - passed
2. `PYTHONPATH=/homes/deand/dev/NRSS mamba run -n nrss-dev python -m pytest tests/validation/test_core_shell_reference.py -k 'test_core_shell_sim_regression_pybind' --nrss-backend cupy-rsoxs -v`
   - passed for both `cupy_tensor_coeff` and `cupy_direct_polarization`

Current disposition after this follow-up:

1. accept the constructor-time preload plus mixed backend-family default for
   `direct_polarization`
2. close the earlier deferred preload experiment as an open default-choice
   question for the current direct path
3. do not generalize the result blindly to `tensor_coeff`
4. if preload work is revisited again, the next credible reason would be a new
   kernel family or a materially different memory authority surface

### Acceptance rule for the next pass

Keep a new direct-path code path only if all conditions below hold:

1. speed:
   - achieve at least about `5%` improvement on the maintained device-hot
     direct-path acceptance lane:
     - `resident_mode='device'`
     - `execution_path='direct_polarization'`
     - small CoreShell
     - `EAngleRotation=[0, 5, 165]`
     - `--worker-warmup-runs 1`
2. physics:
   - pass the parity gates above
3. memory:
   - compare peak GPU memory against the current maintained
     `direct_polarization` baseline on the same small device-hot `0:5:165`
     surface using the existing external polling methodology
   - accept only if the candidate:
     - decreases peak GPU memory,
     - or increases peak GPU memory by no more than about `5%`

Additional interpretation rules:

1. Report the device-hot no-rotation companion as a regression guard, but that
   companion is not the primary acceptance authority.
2. Report the maintained host-prewarmed lanes for continuity with the earlier
   direct-path note, but host-prewarmed wins alone are no longer sufficient for
   acceptance.
3. Suspicious segment collapses, especially implausible `E` changes, should be
   treated as recheck triggers rather than accepted evidence.
4. If a candidate only clears the speed gate by taking a larger memory hit than
   the `+5%` ceiling, reject it as an accepted code path and keep it only as an
   unaccepted research note.
5. If a candidate wins on the device-hot acceptance lane but is intended to be
   a higher-memory expert mode, still require the same parity and memory gates;
   explicit mode status does not waive them.

### Resume checklist for a fresh context

A fresh context should resume in this order:

1. read this file first
2. confirm the runtime toolchain is fixed
3. rerun the fresh host-prewarmed and device-hot speed baselines in Phase 2
4. refresh the device-hot peak-memory baseline using the existing external
   polling methodology
5. implement the parity scaffolding in Phase 3 if it does not already exist
6. use the nine fresh opportunities as the active experiment list unless a new
   benchmark result clearly changes the order
   - deferred add-on to consider only if memory remains a cold-process issue:
     - the precompile-before-`A2` experiment described above
7. update this file after every attempted step with:
   - artifact paths
   - measured device-hot `0:5:165` delta
   - measured device-hot no-rotation delta
   - measured no-rotation delta
   - measured `0:5:165` delta
   - measured peak-memory delta
   - parity outcome
   - keep/reject decision
8. if a change affects the backend-wide story, add a brief cross-reference in
   `CUPY_RSOXS_OPTIMIZATION_LEDGER.md`

## April 4 2026 execution results for fresh opportunities `1-5`

The user explicitly asked that this pass be judged on the `0:5:165` rotation
surface because that lane emphasizes the current direct-path issues.

Treat the following as the current recorded result set for the first five fresh
opportunities listed above.

Acceptance surface used in this pass:

1. speed authority:
   - small CoreShell
   - `resident_mode='device'`
   - `execution_path='direct_polarization'`
   - `EAngleRotation=[0, 5, 165]`
   - `--worker-warmup-runs 1`
   - baseline artifact:
     - `test-reports/cupy-rsoxs-optimization-dev/dp_apr4_baseline_gpu2_hot_rot/summary.json`
2. explicit parity gate used in this pass:
   - `PYTHONPATH=/homes/deand/dev/NRSS mamba run -n nrss-dev python -m pytest tests/smoke/test_smoke.py -k 'test_cupy_direct_polarization_matches_tensor_coeff_on_anisotropic_sphere' -v`
3. additional parity gate used for idea `5` because it rewired the direct-path
   detector projection:
   - `PYTHONPATH=/homes/deand/dev/NRSS mamba run -n nrss-dev python -m pytest tests/smoke/test_smoke.py -k 'test_cupy_direct_polarization_host_and_device_residency_parity' -v`
4. peak-memory baseline used in this pass:
   - external GPU polling on `CUDA_VISIBLE_DEVICES=2` over the same small
     device-hot `0:5:165` command
   - maintained baseline peak observed:
     - about `945 MiB`

### Opportunity `1`: complete the remaining float32 Segment `B` fusion

- implementation shape:
  - split the float32 direct path into dedicated isotropic and anisotropic
    kernels so the float32 path no longer materializes `isotropic_term`
    separately in Python
- artifacts:
  - timing artifact:
    - `test-reports/cupy-rsoxs-optimization-dev/dp_apr4_idea1_gpu2_hot_rot_rerun/summary.json`
- measured outcome versus the maintained baseline:
  - device-hot no-rotation:
    - `primary 0.0221 s -> 0.0235 s`
    - regression of about `+6.4%`
  - device-hot `0:5:165`:
    - `primary 0.6530 s -> 0.6310 s`
    - improvement of about `-3.4%`
  - device-hot `B` on `0:5:165`:
    - `0.2757 s -> 0.1346 s`
  - device-hot `D` on `0:5:165`:
    - `0.2593 s -> 0.3789 s`
- parity:
  - not pursued for acceptance after the speed gate failed
- disposition:
  - rejected
- reason:
  - the `0:5:165` primary lane did not clear the `5%` speed gate, and the
    no-rotation companion regressed

### Opportunity `2`: fuse across materials per voxel in the direct path

- implementation shape:
  - prototype a float32 direct-path kernel that owned the material loop per
    voxel and read compact device arrays of per-material scalars plus raw field
    pointers
- artifacts:
  - timing artifact:
    - `test-reports/cupy-rsoxs-optimization-dev/dp_apr4_idea2_gpu2_hot_rot_rerun/summary.json`
- measured outcome versus the maintained baseline:
  - device-hot no-rotation:
    - `primary 0.0221 s -> 0.0239 s`
    - regression of about `+8.6%`
  - device-hot `0:5:165`:
    - `primary 0.6530 s -> 0.6271 s`
    - improvement of about `-4.0%`
  - device-hot `B` on `0:5:165`:
    - `0.2757 s -> 0.0742 s`
  - device-hot `D` on `0:5:165`:
    - `0.2593 s -> 0.4352 s`
- parity:
  - not pursued for acceptance after the speed gate failed
- disposition:
  - rejected
- reason:
  - this again shifted time out of `B` and into later work without clearing
    the `0:5:165` primary acceptance threshold

### Opportunity `3`: add a per-energy isotropic base-field cache

- implementation shape:
  - build one angle-independent `complex64` isotropic base field per energy
    and reuse it across the direct-path angle loop
  - keep anisotropic work on the existing low-memory path
- artifacts:
  - timing artifact:
    - `test-reports/cupy-rsoxs-optimization-dev/dp_apr4_idea3_gpu2_hot_rot/summary.json`
- measured outcome versus the maintained baseline:
  - device-hot no-rotation:
    - `primary 0.0221 s -> 0.0235 s`
    - regression of about `+6.5%`
  - device-hot `0:5:165`:
    - `primary 0.6530 s -> 0.6203 s`
    - improvement of about `-5.0%`
  - device-hot `B` on `0:5:165`:
    - `0.2757 s -> 0.1504 s`
  - device-hot `D` on `0:5:165`:
    - `0.2593 s -> 0.3517 s`
  - peak GPU memory on the same surface:
    - about `945 MiB -> 879 MiB`
    - improvement of about `-7.0%`
- parity:
  - explicit anisotropic execution-path smoke passed
- disposition:
  - accepted
- reason:
  - this was the first candidate to clear the `0:5:165` speed gate while also
    improving measured peak GPU memory
- implementation status:
  - retained in `src/NRSS/backends/cupy_rsoxs.py`

### Opportunity `4`: fuse windowing into direct polarization generation

- status in this pass:
  - skipped by explicit user direction
- reason:
  - the user required this pass to be evaluated on the maintained
    `0:5:165` issue-emphasizing surface, and the maintained CoreShell
    acceptance morphology on that lane uses `WindowingType = 0`
  - therefore opportunity `4` is not meaningfully exercised on that authority
    surface
- disposition:
  - skipped / not evaluated in this pass

### Opportunity `5`: materially different direct detector-plane projection retry

- implementation shape:
  - adapt the accepted `tensor_coeff` detector-grid-helper style to the direct
    polarization basis
  - project directly from `(fft_x, fft_y, fft_z)` on the detector grid and
    bypass full `scatter3d` materialization
- artifacts:
  - timing artifact:
    - `test-reports/cupy-rsoxs-optimization-dev/dp_apr4_idea5_gpu2_hot_rot_rerun/summary.json`
- measured outcome versus accepted opportunity `3`:
  - device-hot no-rotation:
    - `primary 0.0235 s -> 0.0109 s`
    - improvement of about `-53.6%`
  - device-hot `0:5:165`:
    - `primary 0.6203 s -> 0.2898 s`
    - improvement of about `-53.3%`
  - device-hot `D` on `0:5:165`:
    - `0.3517 s -> 0.0220 s`
  - peak GPU memory on the same surface:
    - about `945 MiB -> 1071 MiB` versus the maintained baseline
    - increase of about `+13.3%`
- parity:
  - explicit anisotropic execution-path smoke passed
  - direct-path host/device residency parity smoke passed
- disposition:
  - rejected
- reason:
  - despite the large `0:5:165` speed win, it failed the memory gate by a wide
    margin
- implementation status:
  - historical April 4 status only:
    - reverted at the end of that day
    - see the April 5-6 follow-up below for the current maintained state

### April 5-6 2026 follow-up on opportunity `5`

The April 4 rejection above was based on coarse external peak-memory polling of
the first kernelized detector-plane attempt across the entire worker lifetime.

That was enough to reject the change under the earlier strict memory rule, but
it did not distinguish:

1. first-use kernel compile/load/startup overhead,
2. from warmed steady-state detector execution working set.

Current follow-up implementation:

1. keep accepted opportunity `3`,
2. replace the earlier temporary-heavy detector-grid helper attempt with a
   fused direct detector-projection RawKernel path,
3. and measure both:
   - full worker-lifetime peak memory,
   - and warmed steady-state memory after one untimed warm-up run inside a
     long-lived subprocess.

Current speed artifact:

1. `test-reports/cupy-rsoxs-optimization-dev/dp_apr5_idea5_kernel_gpu2_hot_rot/summary.json`

Measured speed versus accepted opportunity `3`:

1. device-hot no-rotation:
   - `primary 0.0235 s -> 0.0100 s`
2. device-hot `0:5:165`:
   - `primary 0.6203 s -> 0.2686 s`
   - improvement of about `-56.7%`
3. device-hot `D` on `0:5:165`:
   - `0.3517 s -> 0.00145 s`

Parity on the fused-kernel version:

1. explicit anisotropic execution-path smoke passed
2. direct-path host/device residency parity smoke passed

Peak-memory interpretation from the follow-up:

1. tighter external whole-worker peak polling still shows a cold/startup peak
   increase:
   - maintained accepted-`3` baseline on the same method:
     - about `871 MiB`
   - fused-kernel opportunity `5` on the same method:
     - about `1135 MiB`
2. however, a warmed steady-state probe in one long-lived subprocess showed:
   - one untimed warm-up run first,
   - then five consecutive timed `0:5:165` runs,
   - warmed baseline at the start of the timed series:
     - about `879 MiB`
   - warmed peak during the five timed runs:
     - still about `879 MiB`
   - no additional timed-run peak above the warmed baseline was observed
3. interpretation:
   - the large extra peak appears to be first-use kernel compile/load/startup
     overhead rather than steady-state detector-execution working set

Disposition after the April 5 follow-up (historical interim state):

1. retain opportunity `5` in the maintained backend code
2. current caveat:
   - cold worker-lifetime peak memory is still materially higher
3. current interpretation:
   - if the relevant workflow is a long-lived warmed subprocess, this path
     captures the detector-plane speed win without a warmed steady-state memory
     increase
4. user direction for this pass:
   - if the remaining memory increase proved to be real even after deeper
     investigation, keep opportunity `5` anyway because the speed win is too
     large to ignore

### April 6 2026 compile-backend mitigation on opportunity `5`

After the April 5 result narrowed the memory issue to first-use
compile/load/startup overhead, the remaining credible mitigation was to keep
the fused detector kernel but switch only the detector projection RawKernels
to the `nvcc` compile path when an `nvcc` binary is available.

Current April 6 implementation shape:

1. keep accepted opportunity `3`
2. keep the fused detector-projection RawKernel path from April 5
3. prefer `backend='nvcc'` for the detector projection RawKernels when `nvcc`
   is discoverable, and fall back to `nvrtc` otherwise
4. refresh CuPy's cached `nvcc` lookup when automatic detection succeeds so
   the worker does not require a manually exported `NVCC` environment variable

April 6 artifacts:

1. explicit-`NVCC` speed confirmation:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_apr6_idea5_kernel_nvcc_env_gpu2_hot_rot/summary.json`
2. final self-configuring speed confirmation:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_apr6_idea5_kernel_auto_nvcc_cachefix_gpu2_hot_rot/summary.json`
3. explicit-`NVCC` external whole-worker memory probe paired with the same
   command:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_apr6_idea5_kernel_nvcc_env_gpu2_hot_rot_memprobe/summary.json`
   - observed peak:
     - about `868 MiB`
4. final self-configuring external whole-worker memory probe paired with the
   same command:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_apr6_idea5_kernel_auto_nvcc_cachefix_gpu2_hot_rot_memprobe/summary.json`
   - observed peak on the recheck:
     - about `674 MiB`

Measured April 6 speed versus accepted opportunity `3`:

1. explicit-`NVCC` speed confirmation:
   - device-hot no-rotation:
     - `primary 0.0235 s -> 0.010 s`
   - device-hot `0:5:165`:
     - `primary 0.6203 s -> 0.269 s`
2. final self-configuring code:
   - device-hot no-rotation:
     - `primary 0.0235 s -> 0.010 s`
   - device-hot `0:5:165`:
     - `primary 0.6203 s -> 0.269 s`
   - device-hot `D` on `0:5:165`:
     - `0.3517 s -> 0.001 s`

April 6 parity on the final self-configuring code:

1. `PYTHONPATH=/homes/deand/dev/NRSS mamba run -n nrss-dev python -m pytest tests/smoke/test_smoke.py -k 'test_cupy_direct_polarization_matches_tensor_coeff_on_anisotropic_sphere' -v`
   - passed
2. `PYTHONPATH=/homes/deand/dev/NRSS mamba run -n nrss-dev python -m pytest tests/smoke/test_smoke.py -k 'test_cupy_direct_polarization_host_and_device_residency_parity' -v`
   - passed

April 6 peak-memory interpretation:

1. first `nvcc`-backed whole-worker probe on the same external method:
   - accepted-`3` baseline reference:
     - about `871 MiB`
   - `nvcc`-backed opportunity `5`:
     - about `868 MiB`
2. repeated recheck after the `nvcc`-backed detector kernel cache was
   populated:
   - observed whole-worker peak:
     - about `674 MiB`
3. interpretation:
   - on the current development environment, switching the detector
     projection RawKernels to `nvcc` eliminates the earlier cold-process
     peak regression while preserving the April 5 speed win
   - if a target environment cannot discover `nvcc`, the code falls back to
     `nvrtc`, so the earlier cold-peak caveat may still apply there

Final disposition after the April 6 follow-up:

1. accept opportunity `5`
2. retain it in the maintained backend code alongside accepted opportunity `3`
3. current accepted interpretation:
   - on the maintained `nrss-dev` environment, idea `5` now captures the
     detector-plane speed win without the earlier large memory penalty

## April 6 2026 direct-path memory-lifetime cleanup pass

This pass evaluated six concrete memory-lifetime cleanup items on top of the
current maintained direct-path state.

Acceptance rule used for this pass:

1. keep a candidate if:
   - direct-hot small CoreShell `0:5:165` primary time does not regress by
     `>= 5%`,
   - the simple direct-vs-tensor anisotropic parity smoke passes,
   - and peak GPU memory on the matching direct-hot memory probe does not rise
     by `>= 5%`
2. reject otherwise

Authority surfaces used in this pass:

1. direct-hot timing baseline:
   - artifact:
     - `test-reports/cupy-rsoxs-optimization-dev/dp_memcleanup_baseline_speed_20260406/summary.json`
   - measured baseline:
     - no rotation:
       - `primary 0.01045 s`
     - `0:5:165`:
       - `primary 0.27320 s`
2. parity gate:
   - `PYTHONPATH=/homes/deand/dev/NRSS mamba run -n nrss-dev python -m pytest tests/smoke/test_smoke.py -k 'test_cupy_direct_polarization_matches_tensor_coeff_on_anisotropic_sphere' -v`
3. direct-hot memory baseline:
   - same external polling method from
     `tests/validation/dev/core_shell_backend_performance/run_comprehensive_backend_comparison.py`
   - narrowed to the single `device / hot / direct_polarization / 0:5:165`
     worker case through the harness's own `_cupy_case(...)` plus
     `_run_case_subprocess(...)` path
   - baseline artifact:
     - `test-reports/core-shell-backend-performance-dev/dp_memcleanup_item1_baseline_singlecase_20260406/single_case_memory_summary.json`
   - measured baseline:
     - peak GPU memory:
       - `679 MiB`

### Item `1`: delete FFT polarization volumes immediately after Segment `D`

Implementation shape:

1. move `del fft_x, fft_y, fft_z` from after Segment `E` to immediately after
   `_projection_from_fft_polarization(...)` returns

Artifacts:

1. timing:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_memcleanup_item1_speed_20260406/summary.json`
2. memory:
   - `test-reports/core-shell-backend-performance-dev/dp_memcleanup_item1_memprobe_20260406/single_case_memory_summary.json`

Measured outcome versus the maintained baseline:

1. direct-hot `0:5:165`:
   - `primary 0.27320 s -> 0.27243 s`
2. peak GPU memory:
   - `679 MiB -> 743 MiB`

Parity:

1. passed

Disposition:

1. rejected
2. reverted in code

Reason:

1. the direct-hot peak-memory probe rose by about `+9.4%`

### Item `2`: preallocate result storage and stop retaining a projection list

Implementation shape:

1. replace `projections.append(...)` plus final `cp.stack(...)` with direct
   writes into a preallocated result tensor

Artifacts:

1. timing:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_memcleanup_item2_speed_20260406/summary.json`
2. memory:
   - `test-reports/core-shell-backend-performance-dev/dp_memcleanup_item2_memprobe_20260406/single_case_memory_summary.json`

Measured outcome versus the maintained baseline:

1. direct-hot `0:5:165`:
   - `primary 0.27320 s -> 0.27131 s`
2. peak GPU memory:
   - `679 MiB -> 871 MiB`

Parity:

1. passed

Disposition:

1. rejected
2. reverted in code

Reason:

1. the direct-hot peak-memory probe rose by about `+28.3%`

### Item `3`: reuse dead polarization buffers as Segment `C` IGOR-shift outputs

Implementation shape:

1. FFT one polarization volume at a time and reuse `p_x`, `p_y`, and `p_z` as
   the shifted outputs instead of allocating fresh shifted arrays

Artifacts:

1. timing:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_memcleanup_item3_speed_20260406/summary.json`
2. memory:
   - `test-reports/core-shell-backend-performance-dev/dp_memcleanup_item3_memprobe_20260406/single_case_memory_summary.json`

Measured outcome versus the maintained baseline:

1. direct-hot `0:5:165`:
   - `primary 0.27320 s -> 0.27070 s`
2. peak GPU memory:
   - `679 MiB -> 871 MiB`

Parity:

1. passed

Disposition:

1. rejected
2. reverted in code

Reason:

1. the direct-hot peak-memory probe rose by about `+28.3%`

### Item `4`: evict detector projection geometry after each completed energy

Implementation shape:

1. do **not** remove within-energy reuse
2. instead, preserve angle-loop reuse and discard the cached
   `detector_projection_geometry_current` entry in `_run_single_energy(...)`
   after that energy finishes

Artifacts:

1. timing:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_memcleanup_item4_speed_20260406/summary.json`
2. memory:
   - `test-reports/core-shell-backend-performance-dev/dp_memcleanup_item4_memprobe_20260406/single_case_memory_summary.json`

Measured outcome versus the maintained baseline:

1. direct-hot `0:5:165`:
   - `primary 0.27320 s -> 0.27173 s`
2. peak GPU memory:
   - `679 MiB -> 679 MiB`

Parity:

1. passed

Disposition:

1. accepted
2. retained in code

### Item `5`: tighten `z_collapse_mode='mean'` direct-path temporaries

Implementation shape:

1. reduce `contrib_x`, `contrib_y`, and `contrib_z` sequentially in
   `_compute_direct_polarization_collapsed_mean(...)`
2. delete `sx`, `sy`, and `sz` as soon as each becomes dead

Artifacts:

1. timing:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_memcleanup_item5_speed_20260406/summary.json`
2. first memory probe:
   - `test-reports/core-shell-backend-performance-dev/dp_memcleanup_item5_memprobe_20260406/single_case_memory_summary.json`
3. recheck after the suspicious first probe:
   - `test-reports/core-shell-backend-performance-dev/dp_memcleanup_item5_memprobe_rerun_20260406/single_case_memory_summary.json`

Measured outcome versus the item-`4` retained state:

1. direct-hot `0:5:165`:
   - `primary 0.27173 s -> 0.27076 s`
2. first peak-memory probe:
   - `679 MiB -> 871 MiB`
3. recheck peak-memory probe:
   - `679 MiB -> 679 MiB`

Parity:

1. passed

Disposition:

1. accepted
2. retained in code

Interpretation:

1. this item only touched the collapsed direct path, so the first `871 MiB`
   reading was treated as a recheck trigger rather than as trustworthy
   evidence
2. the rerun returned to the `679 MiB` maintained baseline on the actual
   direct-hot authority lane

### Item `6`: in-place direct-path rotation accumulation and final averaging

Implementation shape:

1. accumulate `valid_counts` with `cp.add(..., out=...)`
2. zero invalid rotated pixels in place with `cp.nan_to_num(..., copy=False)`
3. accumulate `projection_average` with `cp.add(..., out=...)`
4. finalize the average in place rather than building a new `cp.where(...)`
   output

Artifacts:

1. timing:
   - `test-reports/cupy-rsoxs-optimization-dev/dp_memcleanup_item6_speed_rerun_20260406/summary.json`
2. memory:
   - `test-reports/core-shell-backend-performance-dev/dp_memcleanup_item6_memprobe_20260406/single_case_memory_summary.json`

Measured outcome versus the retained item-`4` plus item-`5` state:

1. direct-hot `0:5:165`:
   - `primary 0.27076 s -> 0.27191 s`
2. peak GPU memory:
   - `679 MiB -> 679 MiB`

Parity:

1. passed after fixing an initial CuPy `divide(..., where=...)` incompatibility
   in the first attempt

Disposition:

1. accepted
2. retained in code

Current retained end state after this pass:

1. keep item `4`
2. keep item `5`
3. keep item `6`
4. item `1`, item `2`, and item `3` were informative but are not retained
   because they failed the direct-hot peak-memory gate on this environment

### April 6 2026 fast-delta CuPy observer recheck of rejected items `1-3`

The user requested a direct recheck of the three earlier rejects above because
the first-pass memory decisions relied on single coarse external probes and
the observed deltas looked suspiciously stochastic.

Recheck methodology:

1. orchestrator:
   - `tests/validation/dev/core_shell_backend_performance/run_direct_polarization_memcleanup_recheck.py`
2. artifact:
   - `test-reports/core-shell-backend-performance-dev/dp_memcleanup_fastdelta_recheck_20260406/direct_polarization_memcleanup_recheck_summary.json`
3. authority surface:
   - small CoreShell
   - `resident_mode='device'`
   - `startup_mode='hot'`
   - `execution_path='direct_polarization'`
   - speed repeats on:
     - no rotation
     - `0:5:165`
   - memory repeats on:
     - `0:5:165`
4. repeated-run settings:
   - `5` runs per variant
   - same-GPU warmed CuPy observer
   - `cupy.cuda.runtime.memGetInfo()` delta versus the stabilized observer
     baseline
   - observer sampling cadence:
     - `0.001 s`
5. interpretation rule used for this recheck:
   - use the repeated median on the fast-delta method as the memory gate
     authority for this retry rather than the earlier single coarse probe

Repeated fast-delta baseline on the new method:

1. no rotation median primary:
   - `0.010579 s`
2. `0:5:165` median primary:
   - `0.278519 s`
3. `0:5:165` median peak GPU delta:
   - `1132 MiB`
4. stability note:
   - all five baseline memory repeats returned the same `1132 MiB` peak delta,
     so this recheck did not show baseline-method stochasticity

Repeated fast-delta outcomes:

1. item `1`:
   - no rotation median primary:
     - `0.010507 s`
   - `0:5:165` median primary:
     - `0.276418 s`
   - `0:5:165` median peak GPU delta:
     - `1132 MiB`
   - disposition:
     - pass
2. item `2`:
   - no rotation median primary:
     - `0.010923 s`
   - `0:5:165` median primary:
     - `0.272532 s`
   - `0:5:165` median peak GPU delta:
     - `1132 MiB`
   - disposition:
     - pass
3. item `3`:
   - no rotation median primary:
     - `0.010782 s`
   - `0:5:165` median primary:
     - `0.275258 s`
   - `0:5:165` median peak GPU delta:
     - `940 MiB`
   - disposition:
     - pass

Combined-state parity after retaining items `1-3`:

1. `PYTHONPATH=/homes/deand/dev/NRSS mamba run -n nrss-dev python -m pytest tests/smoke/test_smoke.py -k 'test_cupy_direct_polarization_matches_tensor_coeff_on_anisotropic_sphere or test_cupy_direct_polarization_host_and_device_residency_parity' -v`
   - passed

Updated retained end state after the fast-delta recheck:

1. keep item `1`
2. keep item `2`
3. keep item `3`
4. keep item `4`
5. keep item `5`
6. keep item `6`

### April 6 2026 CuPy pool-off control on small direct-hot `direct_polarization`

The user also requested a high-risk allocator control: disable the CuPy device
and pinned memory pools inside the worker and compare the resulting small
direct-hot `direct_polarization` lane against the current maintained state.

Method:

1. same development recheck harness shape as the fast-delta pass above
2. artifact:
   - `test-reports/core-shell-backend-performance-dev/dp_cupy_pool_off_20260406/direct_polarization_memcleanup_recheck_summary.json`
3. authority surface:
   - small CoreShell
   - `resident_mode='device'`
   - `startup_mode='hot'`
   - `execution_path='direct_polarization'`
   - speed repeats on:
     - no rotation
     - `0:5:165`
   - memory repeats on:
     - `0:5:165`
4. repeated-run settings:
   - `5` runs per variant
   - same-GPU warmed CuPy observer
   - `cupy.cuda.runtime.memGetInfo()` delta versus the stabilized observer
     baseline

Results versus the maintained baseline:

1. baseline:
   - no rotation median primary:
     - `0.010642 s`
   - `0:5:165` median primary:
     - `0.276261 s`
   - `0:5:165` median peak GPU delta:
     - `940 MiB`
2. CuPy pool off:
   - no rotation median primary:
     - `0.025799 s`
   - `0:5:165` median primary:
     - `0.608152 s`
   - `0:5:165` median peak GPU delta:
     - `884 MiB`

Interpretation:

1. disabling the pools lowered the observed peak GPU delta by only about
   `56 MiB` or `6.0%`
2. but it slowed the direct path by about:
   - `2.42x` on no rotation
   - `2.20x` on `0:5:165`
3. parent-process RSS also rose materially on this environment during the
   pool-off runs

Disposition:

1. rejected
2. do not retain any allocator-pool-disable mode in tests or backend code
3. treat this as an informative control showing that the current small
   direct-hot peak is not worth trading for no-pool execution on this
   environment
4. if allocator behavior is revisited later, prefer targeted lifetime reuse or
   release controls over globally disabling CuPy pooling

### April 6 2026 medium host-resident Segment `C` and isotropic-cache follow-up

The user then requested two separate medium-shape experiments on the fair
cross-backend lane after the direct/CUDA code review:

1. promote a true CyRSoXS-like in-place `Segment C`
2. try removing the persistent `isotropic_base_field`

Authority surface:

1. shape:
   - medium CoreShell
   - `(64, 1024, 1024)`
2. residency:
   - `resident_mode='host'`
3. startup:
   - warmed worker
4. path:
   - `execution_path='direct_polarization'`
5. rotations:
   - no rotation
   - `0:5:165`
6. artifact summary:
   - `test-reports/core-shell-backend-performance-dev/direct_polarization_medium_experiments_20260406/summary.json`

Parity gate used for this follow-up:

1. small host-resident no-rotation direct-path parity versus the maintained
   baseline
2. both candidates returned:
   - `max_abs_diff = 0.0`
   - `sum_abs_diff = 0.0`

Measured outcome on the medium host-resident authority lane:

1. maintained pre-promotion baseline:
   - no rotation:
     - `primary 0.74166 s`
     - peak GPU delta `6144 MiB`
   - `0:5:165`:
     - `primary 2.73686 s`
     - peak GPU delta `6144 MiB`
2. true in-place `Segment C`:
   - implementation shape:
     - cached cuFFT `C2C` plan
     - in-place cuFFT on `p_x / p_y / p_z`
     - in-place Igor-order swap kernel modeled on the CyRSoXS swap logic
   - no rotation:
     - `primary 0.74166 s -> 0.75149 s`
     - peak GPU delta `6144 MiB -> 5632 MiB`
   - `0:5:165`:
     - `primary 2.73686 s -> 2.77645 s`
     - peak GPU delta `6144 MiB -> 5632 MiB`
   - disposition:
     - accepted
     - retained in code as the new maintained baseline
   - interpretation:
     - this consistently removed one medium `complex64` full-volume block, about
       `512 MiB`, with only about `1.3%` to `1.4%` slowdown
3. no persistent `isotropic_base_field`:
   - implementation shape:
     - do not build the per-energy cached isotropic base field
     - fall back to the per-material direct-path isotropic accumulation already
       present in `_compute_direct_polarization(...)`
   - no rotation:
     - `primary 0.74166 s -> 0.75614 s`
     - peak GPU delta `6144 MiB -> 5634 MiB`
   - `0:5:165`:
     - `primary 2.73686 s -> 3.70781 s`
     - peak GPU delta `6144 MiB -> 6146 MiB`
   - disposition:
     - rejected for the maintained baseline
   - interpretation:
     - this helps only the one-angle case
     - on the maintained multi-angle lane the speed loss was about `35%`
     - and the memory win disappeared because the path still materializes the
       per-material `isotropic_term = vfrac * isotropic_diag` temporary

Updated maintained interpretation after this follow-up:

1. keep the accepted per-energy isotropic base-field cache
2. make true in-place direct-path `Segment C` the new maintained baseline
3. treat the rejected no-isotropic-base result as evidence that a future
   isotropic-memory reduction must remove both:
   - the persistent cached base field
   - and the per-material `isotropic_term` temporary
4. note the cross-path implication carefully:
   - this accepted direct-path in-place FFT result may also point to a future
     `tensor_coeff` Segment `C` memory experiment, because
     `_compute_fft_nt_components(...)` still does an out-of-place
     `cp.fft.fftn(...)` before shifting back into `nt[idx]`
   - but that possibility is still unmeasured on the maintained
     `tensor_coeff` lanes and should not be generalized from direct
     polarization without a separate parity, timing, and peak-memory pass

### Deferred future exploration ideas `3-6` after the medium follow-up

These ideas were not implemented in this follow-up and remain the next
medium-memory candidates if direct-path memory is revisited again.

#### Idea `3`: fused isotropic accumulation kernel

1. implementation shape:
   - replace both:
     - the persistent per-energy `isotropic_base_field`
     - and the per-material `isotropic_term = vfrac * isotropic_diag`
       temporary
   - with a small RawKernel that writes the isotropic contribution directly
     into `p_x` and `p_y`
2. why it remains attractive:
   - the rejected no-isotropic-base control showed that removing only the cache
     is not enough on the maintained multi-angle lane
   - this fused kernel is the direct next step that could remove both known
     isotropic-memory costs at once
3. expected payoff:
   - lower direct-path peak memory without giving back the accepted multi-angle
     speed benefit of the isotropic cache

#### Idea `4`: packed voxel-style runtime staging plus device-owned material loop

1. implementation shape:
   - pack direct-path runtime material fields into a CyRSoXS-like voxel record,
     for example `(S, theta, psi, Vfrac)` plus a compact material-class tag
   - let one direct-path kernel own the material loop internally, as the legacy
     CUDA path does
2. why it remains attractive:
   - current direct-path runtime staging still keeps four separate arrays per
     anisotropic material and loops over materials in Python
   - this is the closest structural match to the CyRSoXS path
3. expected payoff:
   - reduced staging pressure
   - lower launch overhead
   - and less allocator churn during `Segment B`

#### Idea `5`: legacy zero-array field slimdown

1. implementation shape:
   - add a runtime zero-field contract for historical `legacy_zero_array`
     materials so all-zero `S / theta / psi` fields can be represented
     compactly instead of being staged as concrete arrays
2. why it remains attractive:
   - the explicit isotropic contract already avoids this staging, but the
     historical compatibility lane intentionally keeps those arrays alive
   - medium direct-path memory complaints still encounter that compatibility
     surface in practice
3. expected payoff:
   - lower host-to-device staging volume
   - lower steady GPU footprint for isotropic-heavy morphologies

#### Idea `6`: zero-field-aware execution fast paths

1. implementation shape:
   - bucket materials by field class, for example:
     - fully isotropic
     - legacy-zero isotropic
     - anisotropic with nonzero aligned content
   - specialize the direct-path work so known-zero branches skip unnecessary
     loads and arithmetic
2. why it remains attractive:
   - once zero fields are represented explicitly, direct-path kernels can avoid
     paying full anisotropic work for field classes that cannot contribute to
     those terms
3. expected payoff:
   - memory-traffic reduction in `Segment B`
   - plus speed wins on isotropic-heavy or sparse-anisotropy workloads

### April 7 2026 fused float32 isotropic accumulation follow-up

This pass executed deferred idea `3` on top of the retained April 6 direct
path state.

Implementation shape:

1. remove the float32 direct-path `isotropic_base_field`
2. remove the fallback per-material `isotropic_term = vfrac * isotropic_diag`
   temporary
3. add:
   - a float32 isotropic kernel for fully isotropic materials
   - a fused float32 anisotropic kernel that writes both the isotropic and
     anisotropic contributions directly into `p_x / p_y / p_z`

Artifacts:

1. dev recheck runner:
   - `tests/validation/dev/core_shell_backend_performance/run_direct_polarization_fused_isotropic_recheck.py`
2. summary:
   - `test-reports/core-shell-backend-performance-dev/dp_fused_iso_run_20260407/direct_polarization_fused_isotropic_recheck_summary.json`

Parity:

1. small host no-rotation parity versus the maintained baseline:
   - `max_abs = 0`
   - `rmse = 0`
   - `p95_abs = 0`

Measured outcome versus the pre-pass maintained state:

1. small device-hot `0:5:165`:
   - `primary 0.273831 s -> 0.257478 s`
   - peak GPU delta `876 MiB -> 760 MiB`
2. medium host-hot no rotation:
   - `primary 0.870253 s -> 0.898388 s`
   - peak GPU delta `5806 MiB -> 4824 MiB`
3. medium host-hot `0:5:165`:
   - `primary 2.797279 s -> 2.752121 s`
   - peak GPU delta `5806 MiB -> 4824 MiB`

Disposition:

1. accepted
2. promoted into `src/NRSS/backends/cupy_rsoxs.py`

Interpretation:

1. this is the first direct-path isotropic-memory candidate that removed both
   known float32 isotropic costs at once
2. it reduced peak GPU memory by:
   - about `116 MiB` or `13.2%` on the small direct-hot authority lane
   - about `982 MiB` or `16.9%` on the medium host-hot lane
3. it also improved the maintained multi-angle primary time on both measured
   `0:5:165` surfaces
4. the medium no-rotation host lane regressed by about `3.2%`, which remained
   well inside the current direct-path acceptance gate

### April 7 2026 host-resident `legacy_zero_array` runtime zero-field follow-up

This pass executed deferred idea `5` and a narrow follow-up for idea `6`.

Implementation shape:

1. detect host-resident direct-path `legacy_zero_array` materials whose
   `S / theta / psi` fields are all zero
2. stage only `Vfrac` for those materials
3. route them through the direct-path isotropic handling
4. compare that simpler contract against an extra bucketed-material-loop
   variant

Artifacts:

1. dev recheck runner:
   - `tests/validation/dev/core_shell_backend_performance/run_direct_polarization_legacy_zero_recheck.py`
2. summary:
   - `test-reports/core-shell-backend-performance-dev/dp_legacy_zero_run_20260407/direct_polarization_legacy_zero_recheck_summary.json`

Parity:

1. small host no-rotation parity versus the maintained legacy baseline:
   - `max_abs = 0`
   - `rmse = 0`
   - `p95_abs = 0`

Measured outcome versus the post-fused-isotropic baseline:

1. accepted runtime zero-field contract:
   - small host no rotation:
     - `primary 0.094900 s -> 0.084919 s`
     - `A2 0.084897 s -> 0.075542 s`
     - `B 0.004157 s -> 0.003363 s`
     - peak GPU delta `760 MiB -> 568 MiB`
   - small host `0:5:165`:
     - `primary 0.354417 s -> 0.313630 s`
     - `A2 0.096546 s -> 0.084159 s`
     - `B 0.134353 s -> 0.106169 s`
     - peak GPU delta `760 MiB -> 568 MiB`
   - medium host no rotation:
     - `primary 0.810066 s -> 0.736386 s`
     - `A2 0.749655 s -> 0.682963 s`
     - `B 0.031988 s -> 0.025205 s`
     - peak GPU delta `4824 MiB -> 3288 MiB`
   - medium host `0:5:165`:
     - `primary 2.681578 s -> 2.400912 s`
     - `A2 0.741228 s -> 0.686065 s`
     - `B 1.062546 s -> 0.837708 s`
     - peak GPU delta `4824 MiB -> 3288 MiB`
2. bucketed follow-up:
   - matched the accepted contract on peak GPU memory:
     - `568 MiB` on both small host cases
     - `3288 MiB` on both medium host cases
   - but did not provide an additional memory win
   - and only produced mixed speed differences versus the simpler contract

Disposition:

1. accepted:
   - the runtime zero-field contract
2. rejected:
   - the extra bucketed-material-loop follow-up
3. promoted accepted scope into `src/NRSS/backends/cupy_rsoxs.py`
4. added focused smoke coverage for:
   - the direct-path shortcut on host-resident legacy-zero materials
   - the deliberate non-expansion of that shortcut to `tensor_coeff`

Interpretation:

1. deferred idea `5` was real and high leverage on the measured host-resident
   compatibility surface
2. the accepted contract reduced peak GPU memory by:
   - about `192 MiB` or `25.3%` on the small host legacy lane
   - about `1536 MiB` or `31.8%` on the medium host legacy lane
3. deferred idea `6` did not justify a separate maintained implementation once
   the zero-field contract already routed those materials through the isotropic
   path
4. the maintained scope is intentionally narrow:
   - `execution_path='direct_polarization'`
   - `resident_mode='host'`
   - historical `legacy_zero_array` compatibility inputs only

## Update Rule

When `direct_polarization` work produces either:

1. a new authoritative path-specific timing result,
2. an accepted implementation change,
3. or a rejected but informative experiment,

update this note and add a brief cross-reference in
`CUPY_RSOXS_OPTIMIZATION_LEDGER.md` if the result changes the backend-wide
story.
