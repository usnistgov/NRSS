# Expose the prepared CuPy Fourier sampler

Implement one production change on `refactor/prepared-fourier-sampler`. Give an external caller access to selected complex Fourier coefficients of a real voxel field, using the same numerical implementation as the compact RSoXS backend. Qualify the change against the exact parent before requesting a merge to `backend/cupy-cyrsoxs-clone`.

| Ticket | Deliverable |
| --- | --- |
| [FS01 · Extract the sampler](FS01-prepared-sampler.md) | A physics-independent public function to prepare a sampler, a reusable capture operation, focused tests, and a short API reference. The existing compact path calls the shared implementation. |
| [FS02 · Qualify the branch](FS02-qualification.md) | Bitwise parent/candidate parity, matched execution and memory evidence, and paired timing decisions on the three CoreShell sizes and the frozen 101-energy lipid scan. |

[FS02 validation results and repair evidence](RESULTS.md)

## Start in an isolated worktree

The reviewed branch head on November 23, 2026 was `0727a5948af63ba35e9bcf12722aabd424ebd5d2` (the opt-in CuPy `q_max` commit). Start in a local NRSS clone. Set `ZIP` to the downloaded archive's absolute path. The following Bash commands create two sibling worktrees and apply the tickets only to the candidate. Set `ZIP` before running the block.

```bash
git fetch origin refs/heads/backend/cupy-cyrsoxs-clone
BASE=$(git rev-parse FETCH_HEAD)
ROOT=$(git rev-parse --show-toplevel)
A=$(dirname "$ROOT")/NRSS-sampler-parent
B=$(dirname "$ROOT")/NRSS-sampler-candidate
git worktree add --detach "$A" "$BASE"
git worktree add -b refactor/prepared-fourier-sampler "$B" "$BASE"
unzip -n "$ZIP" -d "$B"
cd "$B"
LAB="$B/tests/validation/dev/cupy_rsoxs_sampler"
OUT="$B/test-reports/prepared-sampler"
printf '%s\n' "$BASE"
```

If the fetched SHA has advanced, verify the extraction points in FS01 against that revision and record it as A. Use a candidate descended from A. Keep A read-only and qualify the final candidate commit, including its source fingerprints. If the target advances before merge, qualify the rebased candidate against that new target before claiming parity. Use one interpreter and the same installed GPU stack for both worktrees.

## Run the qualification

Use the established GPU host and its existing frozen fixture. The historical Wheatley manifest is below. If the job moved, set `LIPID` to its equivalent manifest with the same crop, optics and scan. The lipid worker verifies those inputs. Pin one idle physical GPU using its allocated index or UUID.

```bash
LIPID=/homes/deand/dev/NRSS_speed/.worktrees/NRSS-virtual-zpad/test-reports/virtual-3dpad/ZP09-vesicle-energies.vrbPPq/fixture.json
GPU=0
common=(--parent-root "$A" --candidate-root "$B" --manifest "$LIPID" --out "$OUT" --gpu "$GPU")

python "$LAB/check_sampling.py"
python "$LAB/run_matrix.py" --phase plan "${common[@]}"
```

Implement the public sampler per FS01, then run the focused and maintained tests in FS02. Commit the candidate before recording the qualification evidence.

```bash
CUDA_VISIBLE_DEVICES="$GPU" python "$LAB/check_sampling.py" --candidate-root "$B"
python "$LAB/run_matrix.py" --phase check  "${common[@]}"
python "$LAB/run_matrix.py" --phase stages "${common[@]}"
python "$LAB/run_matrix.py" --phase memory "${common[@]}"
python "$LAB/run_matrix.py" --phase time   "${common[@]}" --pairs 8 --warms 5 --margin-percent 2
```

Each invocation makes a new evidence directory. A native-only run, e.g. `--case core-small core-medium core-large`, can proceed independently of the external vesicle fixture. A missing frozen input leaves the lipid rows pending. Keep raw worker logs, arrays and GPU traces under `test-reports/`. Fill the [qualification record](RESULTS.template.md) from the final runs. Submit the tested branch for review after qualifying it.

Use `--pairing aa` for an incumbent-only batch and `--pairing ab` for a
separate candidate batch. When timing batches run separately, combine the
per-case run directories with `verify_results.py timing` as described in FS02.

## Provenance

The protocol builds on the *Assess variable k_in accuracy* work. The maintained CoreShell driver sets the native timing boundary. The ZP10–ZP15 vesicle replay defines the raw detector control. QM05 supplies the cube, the 0.8 nm⁻¹ cutoff and the matching-chunk I/A comparisons. Historical timings are reference points for identifying workloads. Each performance decision uses freshly measured A/A and A/B runs at the selected commit.

[Current source](https://github.com/usnistgov/NRSS/tree/0727a5948af63ba35e9bcf12722aabd424ebd5d2) · [Benchmark protocol](../speed_tickets/MEASUREMENTS.md) · [Frozen lipid and padded-scan qualification](../zpad_tickets/ZP10-15-RESULTS.md) · [q cutoff and 101-energy I/A qualification](../qmax_tickets/RESULTS.md)
