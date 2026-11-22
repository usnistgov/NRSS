# Compute fewer detector pixels · QM01–QM05

Add one opt-in CuPy backend option, `q_max` in nm⁻¹, to native compact tensor with energy reuse, virtual Z sum/residues, and virtual XYZ. The primary product is the Cartesian detector. Acceptance requires **zero bit mismatches inside the requested window**. Measure elapsed time from public call to synchronized return and the **peak process host RAM and GPU VRAM**. The existing reduced layouts share the option and get compatibility checks. Synthetic inputs cover the full series. The 0.8 nm⁻¹ example sets one useful measurement point.

| Order | Ticket | Output |
|---|---|---|
| 1 | [QM01 · Freeze the reference and output contract](QM01-contract.md) | Matched baselines and an executable bitwise oracle |
| 2 | [QM02 · Plan the actual pixels](QM02-demand.md) | Public crop, rotation and reduction neighbors, and energy-aware Fourier demand |
| 3 | [QM03 · Native compact tensor](QM03-native.md) | A qualified implementation using the incumbent FFT and fewer detector pixels |
| 4 | [QM04 · Virtual Z and XYZ](QM04-padding.md) | Independent results for axial sums, residues and the current XYZ evaluator |
| 5 | [QM05 · Measure and qualify](QM05-qualification.md) | Speed and peak-memory tables, complete in-window bit checks and full pre-push report |

The first candidate preserves the incumbent FFT and per-pixel arithmetic. The experiment is whether touching fewer pixels pays. An optional transform change comes last and faces the same zero-mismatch gate. [Source notes and CPU counts](EVIDENCE.md)

## Unpack at the final speed-series revision

Start in the local worktree containing the completed and tested ZP10–ZP15 changes. The public `perf/virtual-z-pad` tip visible during preparation was `bdf69e111a0e00d0adb2241067d949648d01e552`. Use the **actual local post-ZP15 commit** as the parent for all timers, memory workers and reference pixels. If that work is still in progress, the CPU probes can run now. Take the comparison SHA from the finished worktree. The example uses Bash and an activated NRSS environment.

```bash
git status --short --branch
git log -1 --format='%H %s'
speed_sha=$(git rev-parse HEAD)
git worktree add --detach ../NRSS-qmax-parent "$speed_sha"
git worktree add -b perf/compact-qmax ../NRSS-qmax "$speed_sha"
cd ../NRSS-qmax
archive="$HOME/Downloads/NRSS_Qmax_Optimization_Tickets_20261122.zip"
unzip -t "$archive"
unzip -n "$archive"
lab=tests/validation/dev/cupy_rsoxs_qmax
parent=../NRSS-qmax-parent
python "$lab/work_counts.py"
python "$lab/probe_dependencies.py"
python "$lab/qmax_reference.py" --self-test
common=(--parent-root "$parent" --candidate-root "$PWD" --gpu 0)
python "$lab/run_public_pair.py" "${common[@]}" --phase plan
```

The archive contains additive tickets and development tools. The default probe builds a deterministic 9 × 33 × 35 morphology and represents it on the same 36 × 64 × 72 logical grid through all four routes. `--source` and `--target` scale the synthetic inputs. Material 1 is isotropic. Material 2 is oriented in the specimen and uses its isotropic response outside it when selected as fill.

## Run the public workflow

The A/A commands work on the parent today. Both workers run the same unbounded calculation. The check worker selects the reference pixels afterward. Once QM02–QM04 expose `q_max`, run A/B. Only the candidate receives the cutoff. The default product is a host-resident detector with 12 E directions.

```bash
python "$lab/run_public_pair.py" "${common[@]}" --phase check --aa
python "$lab/run_public_pair.py" "${common[@]}" --phase time --aa --warms 5 --pairs 3
python "$lab/run_public_pair.py" "${common[@]}" --phase check --q-max 0.8
python "$lab/run_public_pair.py" "${common[@]}" --phase diagnose --q-max 0.8
python "$lab/run_public_pair.py" "${common[@]}" --phase time --q-max 0.8 --warms 5 --pairs 3
python "$lab/run_public_pair.py" "${common[@]}" --phase memory --q-max 0.8 --poll-ms 10
```

Timing and memory workers measure the unmodified parent return and the limited candidate return directly. Full-product slicing, host materialization of device results and bitwise comparison happen in dedicated check workers. `--residency device` measures a synchronized GPU-resident result. `--uncapped` compares the complete parent and candidate with the option omitted. Every GPU phase writes a fresh directory under `test-reports/qmax/`.

Use the [measurement matrix](QM05-qualification.md) for meaningful sizes. Record results in `optimization/cupy_rsoxs/qmax_tickets/RESULTS.md`. A clean A/A and a small synthetic spot check permit further development. The final push requires the full maintained panels plus exhaustive in-window comparisons on the candidate revision.

Collect completed A/B results in one CSV. Blank fields identify pending measurements. Each row links its worker directories.

```bash
python "$lab/summarize_runs.py" test-reports/qmax > test-reports/qmax/comparison.csv
```
