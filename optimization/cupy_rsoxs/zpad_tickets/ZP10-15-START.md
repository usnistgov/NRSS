# Faster multi-energy and high-resolution virtual padding · ZP10–ZP15

Start from `usnistgov/NRSS` branch `perf/virtual-z-pad`, reviewed at `bdf69e111a0e00d0adb2241067d949648d01e552`. The XYZ implementation landed in `fe42c7d`. The tip subsequently changed which validation lanes run by default. ZP05–ZP09 are qualified. This addendum targets the qualified XYZ calculation as the speed baseline. All proposed production changes are experiments. [Pinned source](https://github.com/usnistgov/NRSS/tree/bdf69e111a0e00d0adb2241067d949648d01e552) · [Prior GPU results](ZP05-09-RESULTS.md) · [Source assessment and CPU evidence](ZP10-15-EVIDENCE.md)

| Priority | Ticket | First decision |
|---|---|---|
| 1 | [ZP10 · Reproduce and attribute](ZP10-profile.md) | How much time do planning, projection, copy and XY/axial work actually consume? |
| 2 | [ZP11 · Address the endpoint basis directly](ZP11-endpoint-plan.md) | Can a compact per-column lookup replace repeated sorting and binary searches, preserving energy-specific geometry? |
| 3 | [ZP12 · Share Fourier loads across energies](ZP12-energy-projector.md) | Does a kernel that shares actual basis loads outperform the fastest scalar projector? |
| 4 | [ZP13 · Tune the current full-panel XYZ evaluator](ZP13-logical-grid.md) | Where do tile depth, basis rows and axial memory traffic limit throughput at 2048–8192? |
| 5 | [ZP14 · Factor the lateral transform](ZP14-factored-xy.md) | Can exact transforms restricted to required Fourier columns beat the best full-plane cuFFT path? |
| 6 | [ZP15 · Combine and qualify](ZP15-promotion.md) | Which changes improve real calls on the final revision while preserving qualified outputs? |

Finish ZP10 before selecting candidates. ZP12 has a scalar control even if ZP11 loses. ZP13 and ZP14 both have independent high-resolution controls. Carry forward winning changes separately so their interaction can be measured. The public product remains the full registered detector or the caller's existing native reduction. Keep new tuning controls private while investigating.

## Unpack and start

Start in the existing branch worktree and record the local and remote SHAs and working state. To keep development independent, the following example creates a sibling worktree for the series and a second, clean worktree for the comparator. If the local branch has advanced, preserve it, identify the new starting SHA and repeat the baseline measurements there. Reuse matching worktrees when resuming.

```bash
git status --short --branch
git rev-parse HEAD
git fetch origin perf/virtual-z-pad
base_sha=$(git rev-parse FETCH_HEAD)
git show -s --format='%H %s' "$base_sha"
git worktree add --detach ../NRSS-zp-speed-parent "$base_sha"
git worktree add -b perf/virtual-pad-speed ../NRSS-zp-speed "$base_sha"
cd ../NRSS-zp-speed
archive="$HOME/Downloads/NRSS_Padded_Scan_Speed_Tickets_20261122.zip"
unzip -t "$archive"
unzip -n "$archive"
tickets=optimization/cupy_rsoxs/zpad_tickets
lab=tests/validation/dev/cupy_rsoxs_virtual_3dpad
git apply --check "$tickets/ZP10-shared-benchmark.patch"
git apply "$tickets/ZP10-shared-benchmark.patch"
python "$lab/probe_energy_endpoints.py" --self-test
python "$lab/probe_energy_endpoints.py" --count-radii
python "$lab/check_shared_endpoint_mixing.py"
python "$lab/check_factored_xy.py"
```

The patch applies only to the development harness. It puts identical XYZ geometry and exterior on both arms, including A/A, and records actual XYZ dimensions in worker results. Apply it once. If the tree already contains these changes, start from the verified dry run below. In a checkout using sparse paths, include `optimization/cupy_rsoxs`, `tests`, `scripts` and `src/NRSS`.

```bash
speed=tests/validation/dev/cupy_rsoxs_optimization/run_speed_tickets.py
baseline=../NRSS-zp-speed-parent
python "$speed" --dry-run --case energy20 --fixture generic --path default --rotation off \
    --xyz-pad-to 256 512 512 --xyz-pad-material 2 --resident host \
    --baseline-root "$baseline" --candidate-root "$PWD"
```

Inspect the two conditions in the printed plan. Each must contain the same target, fill ID, energy list, layout, residency and angles. Drop `--dry-run` to time the public calculation on the local GPU. The generic check starts immediately from repository fixtures. ZP10 identifies the exact frozen lipid benchmark for the primary claim. Its 20-point subset comes from the frozen manifest. Keep the development harness import and each worker's imported NRSS checkout in the provenance.

Write a new ledger, `optimization/cupy_rsoxs/zpad_tickets/ZP10-15-RESULTS.md`, as work proceeds. Record each experiment as `screened`, `keep`, `conditional` or `reject`, together with the evidence, revision and operating range. Keep raw logs and traces under `test-reports/padded-scan-speed/`. On completing ZP15, hand off the qualification results and chosen deployment scope to Dean.
