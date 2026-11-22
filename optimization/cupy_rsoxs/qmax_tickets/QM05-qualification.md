# QM05 · Report wall time, peak RAM and bitwise detector parity

**Primary comparison.** On the final candidate SHA, compare a full unbounded parent return with the candidate's limited return. Use the same physical inputs, logical grid, window, angles, energy order, result residency and GPU. In a separate accuracy worker select the requested pixels from the full parent, then compare their original dtype and bytes. Use `--uncapped` to compare full parent and candidate behavior. An A/A pair measures noise and verifies that this GPU/driver/parent is repeatable.

| Workload | Inputs / logical target, ZYX | Cutoff and energy controls |
|---|---|---|
| Default native compact, dense film | `128×512×512` native, then `128×1024×1024` where feasible | Q of 0.25, 0.5, 0.8 and a broad cutoff. One and 101 energies at 0.8. Use 20 energies for the sweep |
| Virtual Z, sum and residues | `32×512×512 → 256×512×512` | Same core sweep, independently for both evaluators |
| Virtual XYZ | `64³ → 512³`. Also test `52³ → 256×1024×1024` | Same core sweep. Extend one energy to 2048² and 4096² for memory scaling |
| Boundary and registration | Small `9×33×35 → 36×64×72` plus native odd and rectangular targets. Add an identity extension and singleton-Z native | Zero, exact-bin, one ULP either side, broad limit and omitted option. Include one mixed soft/hard scan or permutation |
| Existing consumers | Same small synthetic inputs, all reduced layouts. Exercise single-energy, intersection, union and a dark channel | Compare complete retained variables, axes, finite masks and bits. No application data are required |

Use raw `detector` output for the main table. Include one direction and the 12 directions (`EAngleRotation=[0,15,165]` in start, increment, end order), host and device result residency at 0.8, and a host chunk-size control. The rest of the sweep can use the regular 12 angles and host output. Check full raw regions for **every energy** in the stated speed claims. If the output is too large to retain twice, check per-energy blocks against a streamed copy. Preserve the original scan/dispatch rather than splitting energies into different public calls. Store per-energy mismatch counts. Use independently recorded first differences to diagnose failures.

**Measurements.** Time fresh worker pairs in alternating order, at least five warm calls per worker and three matched pairs for results, plus first calls and A/A drift. Synchronize the GPU before and after the public return. Run topology counters, allocations and snapshots separately. For a host result D2H is included. For a device result time the terminal device synchronization and defer host materialization to the check workers. Log actual routes, setup, full versus bounded projection pixel counts, output cells and bytes, FFT and basis counters. If speedup is inside the A/A variability, report it as unresolved.

The memory phase writes **absolute process peak host RSS** and **per-worker peak GPU VRAM** from a separate NVML observer, along with polling interval, sample count, worker PID, pool state and host high-water before the call. Use an isolated physical GPU and fresh workers with the same source generation. Report absolute peaks in MiB, the difference in MiB and relative savings. An unchanged peak is a valid result. Sampled VRAM can miss short transients. Repeat near the crossover or at a faster supported rate when needed. The size of the output array and CuPy's ending pool reservation each have their own column. Neither is a peak-RAM measurement.

```bash
big=(--route xyz --source 64 64 64 --target 256 4096 4096 --energy-count 1 --angles 1)
python "$lab/run_public_pair.py" "${common[@]}" "${big[@]}" --phase time --q-max 0.8 --pairs 3 --warms 5
python "$lab/run_public_pair.py" "${common[@]}" "${big[@]}" --phase memory --q-max 0.8
python "$lab/run_public_pair.py" "${common[@]}" --phase check --uncapped
python "$lab/run_public_pair.py" "${common[@]}" --phase check --layout i_a --grid union --q-max 0.8
```

**Push gate.** On the same tested parent and final candidate, run the maintained full default report, including full physics panels. Run the full cutoff oracle on eligible representative maintained inputs through the public route, including an off-axis/Bragg panel and an anisotropic angle sweep. Check the incumbent's actual retained outputs and every retained coordinate bit. Confirm the unbounded defaults still pass. If shared code changes any optional direct or legacy lane, include its normal report. Record inherited failures with their paired evidence. Keep the existing physics goldens and tolerances for those existing tests. The new q-window gate requires **zero mismatched bits**. An alternative that passes only a tolerance stays out of the promoted dispatch.

```bash
env -u NRSS_TEST_BACKEND -u NRSS_BACKEND -u NRSS_TEST_PATH -u NRSS_PATH \
    scripts/run_local_test_report.sh -e "$CONDA_DEFAULT_ENV" -r test-reports/qmax/full-report
```

**Handoff.** Deliver one row per route, shape, cutoff, energies and residency with full/limited first and warm time, paired speedup, host peak pair, GPU peak pair, output byte pair, number of requested samples checked and bit mismatches. Link raw reports and exact SHAs. State where the option pays, where a fixed FFT/staging cost dominates and where RSS or VRAM stays flat. Record any unmeasured configuration explicitly.
