# `cupy-rsoxs` `z_collapse_mode` Proposal

Status: exploratory design, partially implemented

Date baseline: April 4, 2026

This document is the canonical resume note for the proposed expert-only
approximation mode that collapses the composed field through `z` before FFT and
then continues with the current effective-`z=1` detector semantics.

## Implementation update

Current implementation status after the April 4, 2026 prototype pass:

1. normalized backend option support is implemented:
   - `backend_options={"z_collapse_mode": "mean"}`
2. `Morphology.z_collapse_mode` is implemented.
3. internal effective-shape handling is implemented for:
   - `_window_tensor(...)`
   - `_detector_geometry(...)`
   - `_detector_projection_geometry(...)`
4. `execution_path="tensor_coeff"` support is implemented.
5. `execution_path="direct_polarization"` support is not implemented yet.
6. the current `tensor_coeff` implementation now collapses during `Nt`
   construction rather than building full `3D` `Nt` and reducing afterward.
7. `z_collapse_mode` is currently mutually exclusive with
   `mixed_precision_mode`:
   - this was an intentional scope cut so the half-input path remains
     untouched for now.

Implemented validation/results from this pass:

1. smoke coverage now includes:
   - option normalization,
   - unknown-option rejection,
   - explicit rejection of `z_collapse_mode` combined with
     `mixed_precision_mode`,
   - native `z=1` exact identity with collapse on versus off.
2. maintained analytical sphere form-factor tests still pass unchanged.
3. a dev comparison harness now exists at:
   - `tests/validation/dev/cupy_rsoxs_z_collapse/run_sphere_z_collapse_comparison.py`
4. the current recommended graphical comparison outputs are:
   - `test-reports/cupy-rsoxs-z-collapse-sphere/sphere_d70_sr1_comparison.png`
   - `test-reports/cupy-rsoxs-z-collapse-sphere/sphere_d128_sr1_comparison.png`

Current exploratory summary from the dev runner:

1. `70 nm` sphere:
   - `rms_log(collapsed/full) ~= 0.151`
2. `128 nm` sphere:
   - `rms_log(collapsed/full) ~= 0.078`
3. runtime ratios remain variable run-to-run, so current evidence should be
   treated as exploratory rather than publishable speed claims.

Use this document together with:

1. `CUPY_RSOXS_BACKEND_SPEC.md` for the current backend contract and current
   implementation state.
2. `REPO_UPGRADE_PLAN.md` for repo-wide prioritization and maintained
   validation posture.

## Intent

Provide an opt-in approximate fast path for arbitrary `3D` morphologies:

1. compose the usual local field in `3D`,
2. collapse that field through `z`,
3. and then continue as an effective `z=1` simulation.

This is explicitly:

1. an approximation,
2. expert-only,
3. not a parity promise,
4. and expected to diverge more as `z` heterogeneity and `q` increase.

## Decisions already locked in

The following are the baseline decisions and should not be re-litigated unless
new evidence appears.

1. Public surface:
   - `backend_options={"z_collapse_mode": "mean"}`
2. Default:
   - off / `None`
3. Orthogonality:
   - `z_collapse_mode` remains orthogonal to `execution_path`
   - original design intent was orthogonality with `mixed_precision_mode`
   - current implementation deliberately does **not** allow this combination
     yet; it should be revisited only when the half-input path is updated
4. Scope:
   - arbitrary `3D` boxes are in scope
   - not limited to native `z=1` morphologies
5. Reduction rule:
   - initial supported reduction is `mean` through `z`
6. Collapse boundary:
   - collapse after local-field composition
   - collapse before detector windowing / FFT work
7. Downstream semantics:
   - continue as an effective `z=1` simulation
   - do not reinterpret this as "return the raw 2D FFT"
8. Execution-path scope:
   - desired eventually for both `tensor_coeff` and `direct_polarization`
9. Prototype order:
   - implement `tensor_coeff` first
   - add `direct_polarization` only after inspecting the first prototype
10. Separation rule:
   - keep this distinct from the separate effective-`2D` detector
     simplification project

## Current backend behavior that matters

For native `z=1` inputs, current `cupy-rsoxs` behavior is already:

1. identity Hann factor in `z`,
2. then detector projection on the single `qz=0` slice,
3. not "skip detector work and expose a raw 2D FFT panel."

That means the intended approximation is:

1. build the full local field,
2. collapse through `z`,
3. and reuse current effective-`z=1` detector semantics.

## First implementation scope

The first implementation should stay narrow.

1. Add normalized backend option support only.
2. Implement only `execution_path="tensor_coeff"`.
3. Keep `direct_polarization` unchanged in the first pass.
4. Keep the feature expert-only and approximation-only.
5. Do not treat the first implementation as maintained validation contract.
6. Do not combine this work with the separate effective-`2D` detector
   simplification project.

## Why `tensor_coeff` is the first target

`tensor_coeff` is the cleaner first prototype because its architecture already
has an energy-local reusable intermediate:

1. Segment `B` composes energy-specific `Nt` component fields.
2. Segment `C` applies windowing and FFT.
3. Segment `D` projects the FFT result onto the detector.

The proposed approximation fits naturally here:

1. build `Nt` normally,
2. collapse the required `Nt` components through `z`,
3. and then feed the collapsed field into the existing FFT/projection pipeline.

This preserves the current reuse story better than the angle-local
`direct_polarization` path.

## Proposed implementation details

### 1. Option normalization

Update `src/NRSS/backends/contracts.py`.

1. Add `z_collapse_mode` to the `cupy-rsoxs` supported backend options.
2. Support:
   - `None`
   - `"mean"`
3. Support the same style of aliases as existing backend options where useful:
   - `""`, `"none"`, `"off"`, `"default"` -> `None`
4. Reject unknown values up front with `BackendOptionError`.

Update `src/NRSS/morphology.py`.

1. Add a `z_collapse_mode` property alongside `mixed_precision_mode`.
2. No change to input-policy semantics is expected from this option alone.

### 2. Runtime placement

Update `src/NRSS/backends/cupy_rsoxs.py`.

Keep the current segment meaning intact:

1. Segment `B` remains local-field composition.
2. The collapse happens at the start of Segment `C`.
3. Segment `D` remains detector projection.

This keeps timing comparisons interpretable against the existing backend.

### 3. Shape policy

Do not collapse to a raw `(components, y, x)` layout.

Instead:

1. keep collapsed `Nt` as `(components, 1, y, x)`,
2. keep collapsed polarization fields later as `(1, y, x)` if/when the direct
   path is added,
3. and let the rest of the code continue to see a `z` axis of length `1`.

This avoids unnecessary special casing in:

1. FFT entry,
2. DC replacement,
3. Igor shift,
4. detector projection,
5. and rotation handling.

### 4. Do not mutate the public morphology

Do not rewrite `morphology.NumZYX` or any public morphology metadata.

Instead introduce internal effective-shape handling, for example:

1. local helper arguments such as `shape_override`,
2. or a small internal shape/geometry descriptor cached in
   `_backend_runtime_state`.

The approximation is a backend execution choice, not a mutation of the
authoritative morphology object.

### 5. Helper refactor needed for the first pass

The current backend still infers detector/window semantics directly from
`morphology.NumZYX`, so the first pass needs a small internal refactor.

Update the helpers that currently assume the authoritative morphology shape:

1. `_window_tensor(...)`
2. `_detector_geometry(...)`
3. `_detector_projection_geometry(...)`
4. any cache keys derived from the shape above

The collapsed path must be able to request effective `z=1` behavior while the
underlying morphology remains `z>1`.

### 6. `tensor_coeff` flow for the first pass

The originally intended first-pass flow was:

1. compute the normal required `Nt` components,
2. if `z_collapse_mode is None`, run the existing path unchanged,
3. if `z_collapse_mode == "mean"`, apply `cp.mean(nt, axis=1, keepdims=True)`,
4. build a window tensor for the effective collapsed shape,
5. FFT the collapsed `Nt`,
6. run the existing detector projection helper on effective `z=1` geometry,
7. then continue through the usual angle-rotation averaging logic.

Actual implemented refinement:

1. the initial prototype was implemented this way,
2. but the current implementation now collapses during `Nt` construction for
   `tensor_coeff`,
3. so the backend does not materialize the full `3D` `Nt` tensor in the
   active collapse path,
4. which is the preferred design for memory reduction and likely the right
   template for any future `direct_polarization` pass.

### 7. Recommended internal helpers

The following helper split would keep the implementation readable.

1. `_z_collapse_mode(morphology) -> str | None`
2. `_collapse_nt_components(nt, cp, mode)`
3. `_effective_detector_geometry(morphology, cp, shape_override=None)`
4. `_window_tensor(morphology, cp, shape_override=None)`

The exact helper names are flexible; the important point is to avoid scattering
implicit `z==1` special cases across unrelated code paths.

## Expected performance behavior

The first pass is expected to help primarily in:

1. Segment `C`
2. Segment `D`

It is not expected to materially reduce the dominant cost of Segment `B`
because the full `3D` `Nt` field still exists before the collapse.

Similarly, peak memory may improve after the collapse boundary, but the first
pass is not a deep memory rewrite.

## Validation strategy

Validation should stay exploratory in the first implementation.

### A. Smoke / contract checks

Add smoke tests for:

1. backend option normalization and rejection,
2. orthogonality with `execution_path`,
3. orthogonality with `mixed_precision_mode`,
4. native `z=1` exact identity with collapse on versus off.

The native `z=1` identity check is important because for a single-slice input
the `mean` collapse is a no-op and should preserve current output exactly.

### B. Main approximation check: full `3D` sphere vs collapsed `3D` sphere

The main validation surface for the first prototype should be the analytical
sphere infrastructure in `tests/validation/test_analytical_sphere_form_factor.py`.

This is the preferred comparison, not the disk.

The intended comparison is:

1. build the same full `3D` sphere morphology,
2. run the maintained full `3D` path,
3. run the same morphology with `backend_options={"z_collapse_mode": "mean"}`,
4. reduce both through the same maintained `I(q)` path,
5. compare `full_3d_sphere` versus `collapsed_3d_sphere`.

This keeps the approximation check focused on the kind of `3D` morphology the
feature is actually meant to accelerate.

### C. Analytical sphere references remain secondary diagnostics

After comparing `collapsed_3d_sphere` to `full_3d_sphere`, compare both against
the two analytical sphere references already used by the maintained test:

1. flat-detector analytical reference
2. direct analytical sphere form factor

The current maintained sphere test already documents why the flat-detector
reference is the correct authority for the maintained detector/remeshing path
and why the direct analytical curve remains a useful secondary comparison.

For this approximation proposal, use those references to diagnose where the
collapsed path lands, not as the primary evidence of approximation quality.

## Sphere comparison details

The exploratory sphere evaluation should reuse the existing maintained helpers:

1. `_run_sphere_backend(...)`
2. `_pyhyper_iq_by_energy(...)`
3. `_analytic_sphere_form_factor_binned_iq(...)`
4. `_flat_detector_analytic_image(...)`
5. `_pointwise_metrics(...)`
6. `_minima_alignment_metrics(...)`

The intended outputs for each diameter are:

1. full-vs-collapsed pointwise metrics:
   - `rms_log`
   - `p95_log_abs`
2. full-vs-collapsed minima metrics:
   - `mae_abs_dq`
   - `rmse_abs_dq`
   - `max_abs_dq`
3. collapsed-vs-flat analytical metrics
4. collapsed-vs-direct analytical metrics
5. runtime summary:
   - baseline wall time
   - collapsed wall time
   - ratio

The primary approximation signal should be:

1. `full_3d` vs `collapsed_3d` `I(q)` agreement

The analytical references should answer:

1. whether the collapsed path remains physically plausible,
2. and whether any shift looks more like a detector/remeshing effect or a true
   collapse-induced deviation.

## Recommended implementation order

If resumed in a fresh context, use this order.

1. Add option normalization and smoke tests.
2. Refactor the internal shape-aware window/detector helpers.
3. Implement `tensor_coeff` collapse at the Segment `C` boundary.
4. Add native `z=1` identity coverage.
5. Add the exploratory analytical-sphere comparison:
   - full `3D` sphere
   - collapsed `3D` sphere
   - flat analytical sphere
   - direct analytical sphere
6. Inspect timing and error surfaces on the 70 nm and 128 nm sphere cases.
7. Only after that decide whether:
   - to keep the feature as dev-only,
   - to add `direct_polarization`,
   - or to promote some coverage into maintained validation.

## Acceptance gates for the first pass

The first implementation should be considered successful only if:

1. native `z=1` outputs are unchanged,
2. the option surface is normalized and tested,
3. `tensor_coeff` collapse works on arbitrary `3D` boxes,
4. the analytical-sphere exploratory comparison shows a useful speed/accuracy
   tradeoff,
5. and the work remains clearly separated from the effective-`2D` detector
   simplification thread.

Do not promote the approximation to maintained validation or user-facing
recommended workflow status until those checks are complete.

## Explicit non-goals for the first pass

1. No new public claim of exactness.
2. No detector simplification work.
3. No `direct_polarization` implementation yet.
4. No attempt to change maintained analytical-sphere thresholds.
5. No attempt to make this the default execution path.

## Resume note

If a fresh context resumes this work, the shortest correct summary is:

1. add `z_collapse_mode="mean"` as a normalized `cupy-rsoxs` option,
2. keep the current implementation only in `tensor_coeff`,
3. the current `tensor_coeff` path already collapses during `Nt` construction
   and uses effective `z=1` FFT/detector semantics,
4. preserve current effective-`z=1` detector semantics,
5. do not mutate the public morphology shape,
6. keep `mixed_precision_mode` incompatible with `z_collapse_mode` until the
   half-input path is intentionally redesigned,
7. the next implementation step should be the same in-construction collapse
   strategy for `execution_path='direct_polarization'`,
8. validate that direct-path implementation with the same style of:
   - native `z=1` identity checks,
   - full `3D` sphere versus collapsed `3D` sphere `I(q)` comparison,
   - analytical sphere secondary diagnostics,
9. and only after that decide whether to broaden support claims or keep the
   feature dev-only.
