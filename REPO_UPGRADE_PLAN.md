# NRSS Repo Upgrade Plan

This document is the authoritative repo-level modernization summary for NRSS.

It intentionally stays narrow:

- repo-wide goals
- current high-level status
- packaging and workflow policy
- phased delivery and open decisions

For detailed `cupy-rsoxs` backend contract, optimization status, benchmark
methodology, and validation-program detail, use the optimization tree:

- `optimization/cupy_rsoxs/backend_spec.md`
- `optimization/cupy_rsoxs/README.md`
- `optimization/cupy_rsoxs/validation_and_status.md`

The previous long-form version of this plan is preserved at:

- `optimization/cupy_rsoxs/archive/REPO_UPGRADE_PLAN_pre_cleanup.md`

## Goal

Build and maintain an NRSS backend architecture that:

1. Preserves trusted physics behavior with CyRSoXS parity-first discipline.
2. Enables a CuPy-native simulation path that avoids avoidable GPU-host-GPU
   transfers.
3. Supports future backend growth without hard-coupling the public API to one
   implementation.
4. Keeps regression testing deterministic and reproducible.
5. Delivers value even when backend speed work is still evolving.

## Current Repo-Wide State

1. The test-hardening milestone is implemented.
2. The backend-preparation milestone is complete for the current
   registry/routing/reporting scope.
3. `cupy-rsoxs` now exists in-repo and is the intended default backend when
   both `cupy-rsoxs` and `cyrsoxs` are available.
4. CuPy is a required maintained runtime dependency.
5. Optimization-specific status and experiment history no longer live in this
   document.

## Current `cupy-rsoxs` Document Routing

1. Stable backend contract:
   - `optimization/cupy_rsoxs/backend_spec.md`
2. Optimization routing and compact history:
   - `optimization/cupy_rsoxs/README.md`
3. Validation/path-matrix/program status:
   - `optimization/cupy_rsoxs/validation_and_status.md`
4. Fast approximation support posture:
   - `optimization/cupy_rsoxs/fast_approximations.md`

## Test And Validation Program

Repo-level policy:

1. Keep the maintained validation surface path-first and backend-explicit.
2. Keep CPU smoke focused on validator, I/O, and API-contract behavior rather
   than CPU physics parity.
3. Treat CLI-vs-pybind checks as legacy compatibility coverage, not as the
   primary maintained route.
4. Keep deterministic fixtures, explicit metadata capture, and controlled
   golden-data updates.

Current maintained `cupy-rsoxs` validation/program status now lives in:

- `optimization/cupy_rsoxs/validation_and_status.md`

## Packaging And Environment Direction

1. CuPy is a required maintained runtime dependency.
2. `cyrsoxs` remains a supported legacy/reference backend.
3. If the maintained CuPy package line changes, update packaging,
   environment, and install guidance together.

## Multi-GPU And Runtime Policy

1. Primary production pattern remains model-parallel execution, typically one
   model per GPU worker.
2. Persistent workers are preferred when memory/plan reuse matters.
3. Single-GPU execution remains the maintained parity target.
4. Internal multi-GPU energy fan-out is still explicitly non-default.

## Development Workflow

1. Use small, focused commits on feature branches.
2. Keep stable scientific environments separate from dev environments.
3. Use notebooks for exploration, but require script-based reproducibility for
   official goldens and maintained diagnostics.

## Phased Delivery Plan

1. Test-hardening milestone:
   - complete
2. Backend-preparation milestone:
   - complete for the current contract/routing/reporting scope
3. Maintained `cupy-rsoxs` backend evolution:
   - active
4. Internal math cleanup and future backend expansion:
   - deferred follow-on work

## Explicit Non-Goals

1. Optimization-first work without maintained regression coverage.
2. Immediate full parity for every historical legacy execution mode.
3. Collapsing repo-wide planning, backend contract, and optimization history
   back into one document.

## Open Decisions For Follow-Up

1. Final parity threshold table by metric and `q` region.
2. Golden dataset size and retention strategy.
3. GPU CI strategy and minimum gating matrix.
4. Release performance gates and acceptable drift.
5. Objective-function API and on-device return contract for fitting workflows.

## Historical Note

The backend-prep stages, landed prep-work file list, and long-form historical
verification record were removed from this hot-path document to reduce context
bloat. They remain preserved in:

- `optimization/cupy_rsoxs/archive/REPO_UPGRADE_PLAN_pre_cleanup.md`
