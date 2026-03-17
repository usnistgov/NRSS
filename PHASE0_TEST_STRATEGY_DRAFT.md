# NRSS Phase 0 Test Strategy (Draft)

Status: In progress (smoke harness landed; GPU-smoke verified)  
Branch: `test/phase0-pytest-hardening`

## 1. Why this exists

Phase 0 in the upgrade roadmap is explicitly about test hardening before backend refactors.
The goal is to make scientific behavior verifiable and stable so later changes can move fast without breaking trusted physics.

## 2. What "good" looks like

A good Phase 0 test program should:

1. Run reliably in `mar2025` with one command.
2. Separate fast checks from expensive GPU checks.
3. Be deterministic (same inputs -> same pass/fail outcome).
4. Compare against trusted references with documented tolerances.
5. Fail with clear messages when physics changes.

## 3. Current state (as of March 17, 2026)

Current smoke assets:

1. `tests/smoke/test_smoke.py` (22 tests total: 12 CPU/non-GPU, 10 GPU-marked).
2. `scripts/run_local_test_report.sh` (environment snapshot + CPU smoke + GPU smoke + markdown summary).
3. `pyproject.toml` pytest markers (`smoke`, `cpu`, `gpu`, `slow`, `phase0`).

Latest local evidence (GPU-enabled host):

1. Command: `scripts/run_local_test_report.sh`
2. Timestamp (UTC): `20260317T170618Z`
3. Report directory: `test-reports/20260317T170618Z`
4. Result: `3/3` steps passed
5. CPU smoke: `12 passed, 10 deselected`
6. GPU smoke: `10 passed, 12 deselected`

Observed caveats:

1. GPU tests are hardware-dependent and still require a visible NVIDIA GPU.
2. GPU smoke emitted one upstream `PendingDeprecationWarning` from `PyHyperScattering` (`GroupBy.apply`).
3. Legacy validation scripts in `tests/validation/` are not yet fully migrated into robust pytest fixtures/parity gates.

## 4. Proposed test layers

### Layer A: Smoke tests (fast, always run)

Purpose: catch broken imports/environment quickly.

Status: Implemented in `tests/smoke/test_smoke.py`.

Coverage examples:

1. Package import succeeds.
2. Required libraries import (`numpy`, `h5py`, `xarray`).
3. Optional GPU libraries are detected and reported.

### Layer B: Unit tests (fast, CPU)

Purpose: validate deterministic helper math and I/O utilities.

Status: Partially represented by deterministic smoke checks; dedicated unit modules still pending.

Targets:

1. Sphere geometry helper behavior.
2. q-grid/remesh helper output shape and monotonicity.
3. Material-file interpolation utilities.

### Layer C: Integration tests (medium)

Purpose: run full mini-workflows and validate expected outputs exist and are parseable.

Status: Partially represented in smoke via pybind runtime, PyHyperScattering integration, and CLI-vs-pybind parity checks.

Targets:

1. Projected sphere workflow.
2. Core-shell workflow.
3. Circle lattice workflow.

### Layer D: Physics parity tests (slow/GPU)

Purpose: compare simulation outputs against trusted analytical/golden references.

Status: Pending as strict golden/analytical gating; smoke parity currently acts as early guardrail.

Targets:

1. Sphere analytical guardrail (radial I(q) trend and tolerance table).
2. Core-shell parity against `CS_reference.nc`.
3. Circle-lattice peak-position parity.

## 5. Pytest architecture

### 5.1 File and naming structure

Use pytest-discoverable names:

1. Rename `circle-lattice-test.py` -> `test_circle_lattice.py`.
2. Keep existing test module names for sphere and core-shell.

### 5.2 Shared fixtures

Create `tests/validation/conftest.py` with fixtures for:

1. `work_dir` (temporary deterministic workspace).
2. `generated_sphere_case`.
3. `generated_coreshell_case`.
4. `generated_circle_lattice_case`.
5. `gpu_available` probe (`nvidia-smi` or CuPy runtime query).

### 5.3 Markers

Define and use:

1. `@pytest.mark.cpu`
2. `@pytest.mark.gpu`
3. `@pytest.mark.slow`
4. `@pytest.mark.phase0`

Default local run can skip slow/GPU unless explicitly requested.

## 6. Determinism rules

1. No timestamp-based folder names inside assertions.
2. Fixed seeds where randomness is used.
3. Tolerances are explicit in code and documented.
4. Golden data updates require a short changelog note.

## 7. Reference data policy

1. Prefer local repo references for Phase 0 (portable execution).
2. Keep references machine-readable (`.nc`, `.h5`, or compressed numpy/xarray formats).
3. Hardcoded personal/home paths are not allowed.

## 8. Execution commands (draft)

From repository root:

```bash
scripts/run_local_test_report.sh
conda run -n mar2025 python -m pytest tests/smoke -m "not gpu" -v
conda run -n mar2025 python -m pytest tests/smoke -m "gpu" -v
conda run -n mar2025 python -m pytest -m "not slow" -q
conda run -n mar2025 python -m pytest -m "phase0" -q
conda run -n mar2025 python -m pytest -m "gpu and phase0" -q
```

## 9. Initial acceptance criteria for Phase 0

1. All Phase 0 tests run under pytest without manual SLURM steps.
2. CPU-marked tests pass on a CPU-only machine.
3. GPU-marked tests run and pass when GPU is present.
4. Analytical/golden comparisons have documented tolerances.
5. CI (or equivalent scripted run) executes Phase 0 suite reproducibly.

Current status snapshot:

1. Criteria 2-3 are satisfied for the smoke layer on `2026-03-17`.
2. Criteria 1/4/5 remain open for full Phase 0 coverage (especially legacy validation migration + CI gating).

## 10. Proposed implementation order

1. Harness conversion: fixtures + naming + markers.
2. Path hardening: remove hardcoded external paths.
3. Fix known circle-lattice generation bug.
4. Re-enable parity assertions under pytest flow.
5. Add smoke/unit tests around helper utilities.
6. Add basic performance/memory logging (non-gating initially).

## 11. Open questions for your review

1. Should GPU tests be required on every commit, or only nightly/label-triggered?
2. Are we comfortable committing golden datasets in-repo, or should they live externally?
3. Do we want strict parity gating immediately, or start with warning-only thresholds?
