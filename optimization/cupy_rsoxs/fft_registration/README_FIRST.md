# FFT grid registration before virtual Z padding

**Predecessor to ZP01.** Correct the unpadded CuPy lookup and establish its scientific regression baseline. Use that revision to evaluate the zero-padded and virtual-padded transforms. Both the historical calculation and the corrected calculation in this packet use the established DC-neighbor replacement.

Reviewed `usnistgov/NRSS`, branch `backend/cupy-cyrsoxs-clone`, commit `eb7d7bf1e6d97a383ded0cef29c0d0acf2ab646c`. This revision adds RG01B, the off-axis grating in the maintained 3D Bragg module, and a paired high-q lattice survey. The local agent implements the change and runs the GPU panels.

## Start in an NRSS checkout

Set `packet` to the absolute path of the downloaded zip. Activate the CUDA-enabled NRSS development environment. For a new checkout create the worktree below. For an existing registration worktree start at the staging commands from its root. If the branch has advanced, record the new SHA and repeat the baseline.

```bash
git fetch origin backend/cupy-cyrsoxs-clone
git worktree add -b fix/fft-grid-registration ../NRSS-registration FETCH_HEAD
cd ../NRSS-registration
git rev-parse HEAD
```

Stage and install the current packet. The installer handles both a fresh worktree and unmodified files from the earlier packet. It preserves local edits and applies the small tracked Bragg test patch once.

```bash
stage_dir="$(mktemp -d)"
unzip -q "$packet" -d "$stage_dir"
python "$stage_dir/optimization/cupy_rsoxs/fft_registration/install.py" --repo "$PWD"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -c 'from pathlib import Path; import NRSS; print(Path(NRSS.__file__).resolve())'
```

Verify that the printed NRSS module is in this worktree. Capture the old CuPy Bragg panels as directed in [RG01B](RG01B-bragg.md) before changing the grid. Run [RG01](RG01-grid.md), complete [RG01B](RG01B-bragg.md), then run [RG02](RG02-regression.md). The supplied geometry and grating tests should expose the issue on the historical CuPy implementation.

## Reproduce the evidence

Open the CoreShell, grating and Bragg PDFs in `evidence/`. See [the indexed Bragg results](BRAGG_RESULTS.md) for the candidate catalogs, threshold study and regeneration commands. The NPZ contains the common q and energy axes and four 101 × 77 anisotropy tables. [EVIDENCE.md](EVIDENCE.md) gives the limits, validation status and input hashes. To redraw the plots or repeat the CPU replay from the repository fixtures, run from the worktree root.

```bash
reg=optimization/cupy_rsoxs/fft_registration
out=test-reports/fft-registration/replay
python "$reg/plot_registration.py" "$reg/evidence/core_shell_curves.npz" --out "$out/supplied"
python "$reg/core_shell_replay.py" --repo . --out "$out"
python "$reg/plot_registration.py" "$out/core_shell_curves.npz" --out "$out/local"
```

The replay generates diagnostics in the output directory. A release reference requires the full production calculation and the existing experimental reduction.
