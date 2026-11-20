# Virtual Z padding on the registered NRSS CuPy grid

Build `perf/virtual-z-pad` as a new child of `fix/fft-grid-registration`. The parent inspected on 2026-11-20 was `3c4c14d51ecb7ff63e47968b2a323f140f4937ef`. It already registers the CuPy Ewald sampler on `fft_bins_v1`, tests the two off-axis grating orders, and remeshes the Bragg cases at their actual energy. [Parent branch](https://github.com/usnistgov/NRSS/tree/fix/fft-grid-registration) · [Inspected commit and validation record](https://github.com/usnistgov/NRSS/commit/3c4c14d51ecb7ff63e47968b2a323f140f4937ef)

**Objective.** For a fixed 128 × 1024 × 1024 film, reproduce the detector image of the same film centered in a 1024³ vacuum box while storing only the occupied real-space planes and the Fourier coefficients the detector needs. Measure the end-to-end time and actual peak VRAM. Quantify what the finer axial sampling buys at high q. Offer padding by explicit request. These tickets use normal incidence.

## Start in a local NRSS checkout

Use your normal NRSS development environment. From an existing checkout with `origin` pointing to `usnistgov/NRSS`, run the following once. Set `archive` to the downloaded packet. The two worktrees leave the starting checkout intact. The detached worktree supplies the same-parent baseline for the timing harness.

```bash
archive="$HOME/Downloads/NRSS_Virtual_Z_Padding_Tickets_20261119.zip"
git fetch origin refs/heads/fix/fft-grid-registration
parent=$(git rev-parse FETCH_HEAD)
git show -s --format='%h %s' "$parent"
git worktree add --detach ../NRSS-zpad-baseline "$parent"
git worktree add -b perf/virtual-z-pad ../NRSS-virtual-zpad "$parent"
cd ../NRSS-virtual-zpad
unzip -n "$archive"
git merge-base --is-ancestor "$parent" HEAD
```

If resuming an existing worktree, use `git worktree list` and verify its ancestry first. If the parent has advanced, record the actual starting SHA and its panel status. The November 19 archive name is retained for this revised packet. The README inside this version names the registration branch above.

```bash
lab=tests/validation/dev/cupy_rsoxs_virtual_zpad
speed=tests/validation/dev/cupy_rsoxs_optimization/run_speed_tickets.py
base=$(git -C ../NRSS-zpad-baseline rev-parse --show-toplevel)
out="$PWD/test-reports/virtual-zpad"
mkdir -p "$out"
python "$lab/check_fft.py" > "$out/fourier-and-grating.json"
python "$lab/lattice_probe.py" --repo . --out "$out/pilot-192.json"
python "$lab/lattice_probe.py" --repo . --xy-size 1024 --out "$out/pilot-1024.json"
python "$speed" --baseline-root "$base" --path default --case depth --material-z 128 --resident host --aa --repeats 3 --out-root "$out/parent-aa"
```

## Ticket order and decision

| Ticket | Work | Evidence to return |
|---|---|---|
| [ZP01](ZP01-reference.md) | Establish the explicit-vacuum and independent Fourier references, then extend the signed Bragg fixture. | Coefficient errors, analytic grating intensities, raw masks and detector coordinates |
| [ZP02](ZP02-gpu.md) | Prototype two ways to evaluate the required axial bins. Preserve the existing reuse across energies. | Paired speed and peak VRAM for the current workflow, explicit pad and virtual pad |
| [ZP03](ZP03-high-q.md) | Measure HCP/BCC peak recovery and complete the promotion panels. | Indexed peak evidence, high-q error, full compatibility results and opt-in decision |

| Arm | Occupied planes | Transform depth | What it establishes |
|---|---:|---:|---|
| A · registered incumbent | N | N | Existing fidelity and cost on the clean parent |
| B · explicit vacuum | N | M | Numerical target for C, using ordinary 3D FFTs |
| C · virtual vacuum | N | M | Opt-in implementation using slice FFTs and the requested axial bins |
| D · independent Fourier sum | N | Continuous qz | Physical sampling error at the reported detector pixels |

Use the same voxelized specimen and the same detector convention in all four arms. Measure C/B parity with matched arithmetic and postprocessing. Measure A/D and C/D to establish whether a change improves the simulated measurement. The [rerun CPU study](CPU_PILOT.md) gives initial results at 128 × 1024². GPU performance and production physics panels are local-agent deliverables.

**Inherited validation status.** The parent commit reports that the FFT-bin and Bragg gates passed on Wheatley. Its historical clean baseline passed 8/8 report steps and the corrected result passed 6/8 after smoke rechecks. The 285.2 eV CoreShell experimental residual exceeded the existing 0.055 limit. The sphere test exposed a radial-label discrepancy in `NRSSIntegrator`. The corrected CoreShell reference is held. Record the parent results separately from new failures. These two findings allow development and raw-detector comparisons to proceed. Promotion to a push requires the complete panels and resolution or explicit maintainer disposition of outstanding results. Preserve the current thresholds and golden provenance.

Keep generated inputs, plots, timings and raw panels under `test-reports/virtual-zpad/`. Maintain a short results ledger next to the tickets with parent and candidate SHAs, package versions, GPU, evidence paths and disposition. If the registration branch is still unmerged when the work qualifies, target it as the PR base. Start with ZP01.
