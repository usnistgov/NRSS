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

## Path Identity

Current execution-path mapping:

1. `tensor_coeff`
   - maintained default `cupy-rsoxs` execution path
2. `direct_polarization`
   - CyRSoXS `AlgorithmType=0` analog
   - communication-minimizing / polarization-first analog
3. `nt_polarization`
   - CyRSoXS `AlgorithmType=1` analog
   - lower-memory `Nt`-first analog

`direct_polarization` remains useful because it:

1. is the closest conceptual match to the legacy polarization-first path,
2. keeps a lower GPU memory footprint than `tensor_coeff` and
   `nt_polarization` on the maintained host-resident CoreShell lane,
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
the maintained small host-prewarmed lanes with all timing segments enabled.

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

### Acceptance rule for the next pass

Keep a change only if both conditions hold:

1. speed:
   - achieve about `5%` or better improvement in either
     - the maintained no-rotation lane
     - or the maintained `0:5:165` lane
2. physics:
   - pass the parity gates above

Additional interpretation rules:

1. A win on either lane is sufficient for acceptance if parity holds.
2. The other lane should still be reported, but a flat result there is not an
   automatic rejection.
3. Suspicious segment collapses, especially implausible `E` changes, should be
   treated as recheck triggers rather than accepted evidence.
4. If a candidate only wins on the fully-hot kernel surface but loses on the
   current cold subprocess authority surface, document both results explicitly
   and decide whether that candidate belongs in:
   - the default maintained path
   - or an explicit higher-performance opt-in mode

### Resume checklist for a fresh context

A fresh context should resume in this order:

1. read this file first
2. confirm the runtime toolchain is fixed
3. rerun the fresh baselines in Phase 2
4. implement the parity scaffolding in Phase 3 if it does not already exist
5. start with the Phase 4 experiment order exactly as listed
6. update this file after every attempted step with:
   - artifact paths
   - measured no-rotation delta
   - measured `0:5:165` delta
   - parity outcome
   - keep/reject decision
7. if a change affects the backend-wide story, add a brief cross-reference in
   `CUPY_RSOXS_OPTIMIZATION_LEDGER.md`

## Update Rule

When `direct_polarization` work produces either:

1. a new authoritative path-specific timing result,
2. an accepted implementation change,
3. or a rejected but informative experiment,

update this note and add a brief cross-reference in
`CUPY_RSOXS_OPTIMIZATION_LEDGER.md` if the result changes the backend-wide
story.
