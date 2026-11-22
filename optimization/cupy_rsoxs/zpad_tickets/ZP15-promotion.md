# ZP15 · Combine winning changes and qualify the complete public workflow

**Ablate the interaction.** On the same GPU and the same input bytes, compare the pinned qualified XYZ baseline, the best energy-only candidate, the best resolution-only candidate, and their combination. With multiple winners within a group also report their incremental gain. Keep native explicit and Z-only matched controls where previously feasible. Use public automatic dispatch and capture actual routes. Recheck any crossover after combining. A useful review target is a 20% lower warm time on 101-energy lipid and on the 8192 single-energy sphere relative to today's matched rerun. This is an engineering target. Smaller reproducible gains can justify a simpler change, and unsuccessful experiments should be recorded.

**Required performance evidence.**

- Frozen lipid 1/20/101 energies, water, one direction, common `(256,1024,1024)` target, first/warm times and absolute/paired speedups vs current XYZ. Repeat 20 energies with 12 directions, 101 in permuted requested order, a true scalar-lookup control, authoritative native `i_a`, host and device detector controls. Report 20→101 slope, fields, FFT count/planes, endpoint/lookup counters and D2H bytes. Include a wide-energy schedule and one non-enum fill (`S=0` exterior, native orientation inside).
- Complete 128–8192 sphere ladder for both fills. Use the existing `--verify-full-voxel` accuracy worker. Record valid pixel counts, timing spread, peak worker VRAM, host RSS, throughput and absolute versus analytical band/first-minimum results on the same fixed-pitch fixture. Use the established full-voxel reference at 8192³. Include a representative mixed case (20 energies, 2048² detector, with matched response and residency). A 101×8192² host stack alone is 25.25 GiB. The two primary campaigns isolate the two effects independently.
- Native (unpadded) 1/20/101 and existing Z-only sum/residues with both windows as regression controls. Compare at unchanged output chunking and precision. Repeat uncertain timing results in interleaved worker blocks with A/A and reversed order. Include a separate memory run and document each peak-memory tradeoff. Avoid promoting a heuristic that simply transfers work to the CPU, increases VRAM beyond the practical budget, or pays an extra setup cost on every short scan without a measured gain.

**Correctness and promotion matrix.** Retain the parent registration, pitch, positive/negative FFT-bin edges for odd/even sizes, DC neighbor repair, stable Ewald root, projection at lower and upper endpoints before intensity interpolation, Euler physics, manual XY Hann or none, unwindowed Z for an extension, identity behavior, centered finite material fill and FP32 semantics. Check numerical equality against the same pinned XYZ path and matched literal controls. For a required full detector check per-energy relative L2 and max absolute error/reference peak ≤ `2e-5`, exact coordinates, energy order and finite masks. Include the first/last energy, boundaries, repaired DC and dark components with a specified absolute scale. Use existing tighter complex-coefficient and sphere tail checks. Keep all 101 panels, native integration axes, absolute I, `i_para_i_perp` and `i_a` on the same support, including `union`/`intersection` and the documented vesicle anisotropy band. Preserve external goldens and thresholds. Include anisotropic off-axis tests, quadratic Euler scaling, HCP/BCC, disk/sphere, CoreShell and MWCNT. Test both externally visible fill modes, supported shapes, release/reuse and forced low-memory retry.

On this branch the unqualified report runs environment, CPU smoke and the compact tensor GPU smoke/full physics suite. From both parent and final candidate worktrees run the complete default panel, plus forced candidate replay through the eligible maintained tests. The command below loads the existing [workflow audit](../../../tests/validation/dev/cupy_rsoxs_virtual_zpad/workflow_padding_audit.py) in pytest and inherited Python test workers. Bind experiments to actual execution counters and update the audit if a candidate specializes the expected `virtual_3d_sum` route. Require at least one invocation of every variant being promoted and record case-level exclusions. Confirm the same-executable full panel after the last computational change.

```bash
env -u NRSS_TEST_BACKEND -u NRSS_BACKEND -u NRSS_TEST_PATH -u NRSS_PATH -u NRSS_TEST_SPEED_VARIANT \
    scripts/run_local_test_report.sh -e "$CONDA_DEFAULT_ENV" -r test-reports/padded-scan-speed/compact
```

Run the first command in each worktree. Run the forced replay separately from the candidate worktree with the packet unpacked, on the same revision used for the default panel. Use a fresh report directory so each JSONL corresponds to one run.

```bash
mkdir -p test-reports/padded-scan-speed
audit_root=$(mktemp -d "$PWD/test-reports/padded-scan-speed/forced.XXXXXX")
mkdir -p "$audit_root/hooks"
cat > "$audit_root/hooks/sitecustomize.py" <<'PY'
import os
if os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get("NRSS_PADDING_SPEED_AUDIT_LOG"):
    from tests.validation.dev.cupy_rsoxs_virtual_3dpad.padding_speed_audit_plugin import install_once
    install_once()
PY
audit_plugin=tests.validation.dev.cupy_rsoxs_virtual_3dpad.padding_speed_audit_plugin
NRSS_PADDING_SPEED_AUDIT_LOG="$audit_root/replay.jsonl" \
PYTEST_PLUGINS="${PYTEST_PLUGINS:+$PYTEST_PLUGINS,}$audit_plugin" \
PYTHONPATH="$audit_root/hooks:$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
    scripts/run_local_test_report.sh -e "$CONDA_DEFAULT_ENV" --nrss-path cupy_tensor_coeff \
    -r "$audit_root/report"
test -s "$audit_root/replay.jsonl"
```

Summarize passed, failed and excluded replays per test, route and fill, and reconcile subprocess coverage against ZP09. Attribute any count change to the actual test selection, exclusions or newly covered cases. Use explicit candidate counters and the native/XYZ production benchmarks to verify the chosen energy and resolution variants also ran on their intended shapes.

Direct and legacy lanes are explicitly selectable at this SHA. Include the full relevant optional lane when shared code can affect it, or to reproduce the complete previous three-lane qualification use `--nrss-path cupy_direct_polarization` and `--nrss-path legacy_cyrsoxs` in separate report directories. Keep their frozen configurations. The earlier report documented installed PyHyperScattering q-grid differences on both revisions. Follow the accepted NRSS native integrator decision, retain historical failures visibly, and compare dispositions against a fresh parent. The [ZP09 ledger](ZP05-09-RESULTS.md) documents the external-reducer discrepancies, the 213-comparison audit and the accepted native checks.

**Handoff.** Record the final candidate SHA, benchmark/fixture hashes and environment, ledger row for each variant, full reports and the qualified dispatch rule, or retain the current route. Identify the operations limiting the best case after the changes. A spot test authorizes further experiments. Promotion to a push requires the full panels and end-to-end evidence on that revision.
