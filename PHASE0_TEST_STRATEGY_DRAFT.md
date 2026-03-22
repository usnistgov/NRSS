# NRSS Phase 0 Test Strategy (Draft)

Status: In progress (smoke harness landed; Bragg-inclusive 3D+2D physics-validation layer landed; sphere orientational-contrast physics validation landed; CoreShell experimental+sim-reference validation landed; MWCNT experimental validation landed; latest local physics lane green)  
Branch: `test/phase0-pytest-hardening`

## 1. Why this exists

Phase 0 in the upgrade roadmap is explicitly about test hardening before backend refactors.
The goal is to make scientific behavior verifiable and stable so later changes can move fast without breaking trusted physics.

## 2. What "good" looks like

A good Phase 0 test program should:

1. Run reliably in the maintained development environment with one command.
2. Separate fast checks from expensive GPU checks.
3. Be deterministic (same inputs -> same pass/fail outcome).
4. Compare against trusted references with documented tolerances.
5. Fail with clear messages when physics changes.

## 3. Current state (as of March 22, 2026)

Current implemented Phase 0 assets:

1. `tests/smoke/test_smoke.py` (22 tests total: 12 CPU/non-GPU, 10 GPU-marked).
2. `tests/conftest.py` (default single-GPU pinning for reproducibility and to avoid known CyRSoXS multi-GPU instability during energy fan-out).
3. `scripts/run_local_test_report.sh` (default `nrss-dev` environment, environment snapshot + CPU smoke + GPU smoke + physics validation + markdown summary, plus `--skip-defaults` and `--repeat N` for targeted command sweeps).
4. `tests/validation/lib/orientational_contrast.py` (reusable tensor-based para/perp-to-far-field helper for inspectable anisotropic contrast predictions).
5. `tests/validation/test_analytical_sphere_form_factor.py` (flat-detector analytical sphere guardrail through the pybind-to-PyHyper workflow).
6. `tests/validation/test_sphere_contrast_scaling.py` (quadratic contrast-scaling validation across beta/delta/mixed/split families).
7. `tests/validation/test_sphere_orientational_contrast_scaling.py` (helper-driven orientational-contrast validation across `theta`, `psi`, low-symmetry Euler combinations, and `S`, including the explicit `S=0` isotropic endpoint).
8. `tests/validation/test_analytical_2d_disk_form_factor.py` (direct analytical 2D disk guardrail through the pybind-to-PyHyper workflow).
9. `tests/validation/test_2d_disk_contrast_scaling.py` (quadratic contrast-scaling validation for the 2D disk pathway).
10. `tests/validation/lib/bragg.py` (shared deterministic Bragg lattice morphology/prediction helpers for 2D and 3D validation).
11. `tests/validation/test_bragg_2d_lattice.py` (deterministic square and hexagonal 2D Bragg peak-position validation through the pybind-to-PyHyper workflow).
12. `tests/validation/test_bragg_3d_lattice.py` (deterministic simple-cubic and HCP 3D Bragg peak-position validation through the pybind-to-PyHyper workflow).
13. `tests/validation/lib/core_shell.py` (shared maintained CoreShell baseline morphology/reduction/reference helper for the official pytest module; falsification scenarios stay under `scripts/validation_diagnostics/`).
14. `tests/validation/test_core_shell_reference.py` (official CoreShell physics validation against the experimental A-wedge reference plus a parallel sim-derived regression golden).
15. `tests/validation/lib/mwcnt.py` (shared maintained deterministic MWCNT morphology/reduction/reference helper with periodic field construction, `WindowingType=0`, published Table I provenance, and vendored reduced experimental observables).
16. `tests/validation/test_mwcnt_reference.py` (official MWCNT physics validation against the experimental anisotropy observables derived from the tutorial/manuscript workflow).
17. `tests/validation/dev/` (development-only MWCNT studies for RSA benchmarking, EAngleRotation sweeps, windowing/periodicity comparisons, and threshold/falsification probes; kept out of pytest collection).
18. `scripts/validation_diagnostics/` (archived one-off development diagnostics plus opt-in orientational-contrast/CoreShell plot probes, including the CoreShell falsification scenarios, kept out of pytest collection).
19. `pyproject.toml` pytest markers (`smoke`, `cpu`, `gpu`, `slow`, `physics_validation`, `experimental_validation`, `toolchain_validation`, `phase0`).

Latest local evidence (GPU-enabled host):

1. Default local report command: `CUDA_VISIBLE_DEVICES=0 bash scripts/run_local_test_report.sh --stop-on-fail`
2. Timestamp (UTC): `20260322T140310Z`
3. Report directory: `test-reports/20260322T140310Z`
4. Result: `4/4` steps passed
5. Physics validation lane inside the report: `14 passed`
6. The standard physics lane in `scripts/run_local_test_report.sh` auto-discovers both the CoreShell and MWCNT experimental-reference modules because it runs `python -m pytest tests/validation -m physics_validation -v`.
7. The markdown report carries physics-test docstrings and marker metadata in the “Physics Tests” section, so both the CoreShell and MWCNT provenance/citation blocks and `experimental_validation` tags flow into future summaries.
8. MWCNT-only command: `CUDA_VISIBLE_DEVICES=0 /home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/validation/test_mwcnt_reference.py -v`
9. MWCNT-only result: `1 passed`
10. CoreShell-only command: `CUDA_VISIBLE_DEVICES=0 /home/deand/mambaforge/envs/nrss-dev/bin/python -m pytest tests/validation/test_core_shell_reference.py -v`
11. CoreShell-only result: `2 passed`
12. Development-only MWCNT falsification evidence now lives under `tests/validation/dev/mwcnt_threshold/` and `test-reports/mwcnt-threshold-dev/`; current maintained thresholds already reject nearby radius falsifications, so no tightening was applied.
13. Opt-in CoreShell diagnostic artifacts were regenerated successfully via `python scripts/validation_diagnostics/core_shell_reference_diagnostic.py --write-sim-reference` under `test-reports/core-shell-dev/`.
14. Opt-in orientational artifacts remain available via `python scripts/validation_diagnostics/sphere_orientational_contrast_diagnostic.py` under `test-reports/sphere-orientational-contrast-dev/`.

Observed caveats:

1. GPU tests are hardware-dependent and still require a visible NVIDIA GPU.
2. GPU smoke emitted one upstream `PendingDeprecationWarning` from `PyHyperScattering` (`GroupBy.apply`).
3. Circle-lattice validation migration still remains open. CoreShell now has official pytest coverage on the maintained pybind + PyHyper + manual A-wedge path, and MWCNT now has official pytest coverage on the maintained pybind + PyHyper + anisotropy-observable path against the published experimental reference.
4. Validation plot writing remains opt-in; the orientational sphere and CoreShell cases keep dedicated manual diagnostics under `scripts/validation_diagnostics/`, while MWCNT development diagnostics and falsification probes now live under `tests/validation/dev/` rather than the maintained pytest surface.

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

Status: Initial 3D+2D layer implemented, including the CoreShell and MWCNT experimental-reference migrations; further case migration still pending.

Implemented:

1. Analytical sphere form-factor comparison:
   - pybind execution only (no CLI serialization path),
   - `512 x 512 x 512`, `PhysSize = 1.0 nm`,
   - diameters `70 nm` and `128 nm`,
   - explicit sphere plus explicit vacuum matrix,
   - PyHyperScattering radial reduction,
   - flat-detector analytical comparison,
   - separate pointwise and minima-alignment metrics with fixed thresholds,
   - superresolution machinery retained (`1x/2x/3x/4x`) with assertions anchored on `sr=1`.
2. Sphere contrast-scaling guardrail:
   - one `70 nm` sphere morphology reused across energies,
   - 24 close-energy contrast scenarios,
   - beta-only, positive-delta-only, negative-delta-only, mixed, and split-material families,
   - integrated intensity metrics over `q in [0.06, 1.0] nm^-1`,
   - fixed thresholds for weighted/unweighted contrast scaling and family pairing consistency.
3. Sphere orientational-contrast guardrail:
   - one `128 x 128 x 128` sphere-in-vacuum morphology at `PhysSize = 2.0 nm`, `Diameter = 32 nm`,
   - contrast varied only through uniaxial tensor orientation and aligned fraction rather than scalar OC magnitude,
   - helper-driven expectations from `tests/validation/lib/orientational_contrast.py`,
   - three close-energy optical-constant families: pure delta dichroism, pure beta dichroism, and mixed delta+beta,
   - high-symmetry `theta` coverage across `[0, pi]`,
   - high-symmetry `psi` coverage across `[0, 2*pi]`,
   - low-symmetry coupled Euler cases plus an `S` sweep including `S=0`,
   - direct detector-annulus ratio checks with a fixed empirical threshold.
4. Analytical 2D disk form-factor comparison:
   - pybind execution only,
   - `1 x 2048 x 2048`, `PhysSize = 1.0 nm`,
   - diameters `70 nm` and `128 nm`,
   - explicit disk plus explicit vacuum matrix,
   - PyHyperScattering radial reduction,
   - direct analytical disk comparison on the PyHyper q bins,
   - separate pointwise and minima-alignment metrics with fixed thresholds,
   - fixed `sr=1` to match the established sphere-harness assertion mode while exercising the distinct 2D compute path.
5. 2D disk contrast-scaling guardrail:
   - one `70 nm` 2D disk morphology reused across energies,
   - `1 x 2048 x 2048`, `PhysSize = 1.0 nm`,
   - 24 close-energy contrast scenarios,
   - beta-only, delta-only, mixed, and split-material families,
   - integrated intensity metrics over `q in [0.06, 1.0] nm^-1`,
   - fixed thresholds for weighted/unweighted contrast scaling and family pairing consistency.

Next targets:

1. Circle-lattice peak-position parity.
2. Two-material mixed beta/delta equivalence cases beyond the current split-family contrast checks.
3. Shared validation fixtures/helpers for cases that now reuse the same pybind-to-PyHyper comparison pattern.

## 5. Pytest architecture

### 5.1 File and naming structure

Use pytest-discoverable names:

1. Keep active pytest modules under `tests/validation/test_*.py`.
2. Legacy development-era cases that are not pytest-ready now live under `scripts/validation_diagnostics/legacy_validation_tests/` so they cannot pollute normal collection.
3. When circle-lattice returns to the maintained suite, reintroduce it as a fresh pytest-native module rather than moving the legacy script back unchanged.

### 5.2 Shared fixtures

Current shared infrastructure:

1. `tests/conftest.py` pins to a single visible GPU by default unless the environment already specifies `CUDA_VISIBLE_DEVICES`.

Still worth adding in `tests/validation/conftest.py` or equivalent helpers:

1. `work_dir` (temporary deterministic workspace).
2. `generated_coreshell_case`.
3. `generated_circle_lattice_case`.
4. shared PyHyper reduction helpers.
5. shared analytical/golden metric helpers.
6. `gpu_available` probe (`nvidia-smi` or CuPy runtime query) if more granular skipping is needed.

### 5.3 Markers

Define and use:

1. `@pytest.mark.cpu`
2. `@pytest.mark.gpu`
3. `@pytest.mark.slow`
4. `@pytest.mark.physics_validation`
5. `@pytest.mark.experimental_validation`
6. `@pytest.mark.toolchain_validation`
7. `@pytest.mark.phase0`

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
bash scripts/run_local_test_report.sh --stop-on-fail
bash scripts/run_local_test_report.sh --skip-defaults --repeat 20 \
  --cmd "python -m pytest tests/validation/test_analytical_2d_disk_form_factor.py -q"
python scripts/validation_diagnostics/orientational_contrast_tiny_diagnostic.py
python scripts/validation_diagnostics/sphere_orientational_contrast_diagnostic.py
conda run -n nrss-dev python -m pytest tests/smoke -m "not gpu" -v
conda run -n nrss-dev python -m pytest tests/smoke -m "gpu" -v
conda run -n nrss-dev python -m pytest tests/validation -m "physics_validation" -v
conda run -n nrss-dev python -m pytest -m "not slow" -q
conda run -n nrss-dev python -m pytest -m "phase0" -q
conda run -n nrss-dev python -m pytest -m "gpu and phase0" -q
```

## 9. Initial acceptance criteria for Phase 0

1. All Phase 0 tests run under pytest without manual SLURM steps.
2. CPU-marked tests pass on a CPU-only machine.
3. GPU-marked tests run and pass when GPU is present.
4. Analytical/golden comparisons have documented tolerances.
5. CI (or equivalent scripted run) executes Phase 0 suite reproducibly.

Current status snapshot:

1. Criteria 1/3/4 are satisfied locally on a GPU-enabled host as of `2026-03-20`.
2. Criterion 2 is likely satisfied by marker separation, but it has not yet been re-confirmed on a true CPU-only host in this workstream.
3. Criterion 5 remains open (CI/nightly policy still needs to be established).

## 10. Proposed implementation order

1. Finish migration of the remaining circle-lattice legacy case if it still warrants maintained coverage.
2. Add shared validation fixtures/helpers for PyHyper reduction and reference-metric computation.
3. Add any missing two-material contrast-family follow-ons if physics coverage warrants it.
5. Add smoke/unit tests around helper utilities.
6. Add basic performance/memory logging (non-gating initially).

## 11. Open questions for your review

1. Should GPU tests be required on every commit, or only nightly/label-triggered?
2. Are we comfortable committing golden datasets in-repo, or should they live externally?
3. Do we want strict parity gating immediately, or start with warning-only thresholds?
