# QM04 · Fewer pixels on virtual Z and XYZ

**Deliverable.** Apply the same public cutoff and backward demand plan to each qualified local padding evaluator. Compare each candidate with its **own unbounded post-ZP15 route**, at the same occupied shape, logical shape, fill, energy schedule and output. Bitwise parity against that incumbent is the acceptance reference. A mathematically equivalent explicit box is a useful existing physics control and can have different bits.

- **Virtual Z sum.** Restrict the gathered XY columns, axial endpoint dots, per-energy projection and transfers. Initially preserve the XY transforms and the original occupied-Z summation order. Record axial dots, endpoint union and bytes separately from plane FFTs.
- **Virtual Z residues.** Select only required rows and omit a residue class if it has no demanded endpoints or DC neighbors. Count classes actually transformed and elements gathered. A smaller window may still touch every residue. First preserve cuFFT batch dimensions and the arithmetic for retained columns.
- **Virtual XYZ.** Feed the restricted union into the locally promoted evaluator. Keep its winning energy reuse, centering, fill, XY window and bounded Z tiling. A smaller union may change automatic tile depth, FFT planning, or accumulation, potentially altering bits. Compare the parent's exact tile/plan policy with a candidate-adaptive policy separately. Record transformed XY planes/columns, axial endpoints and partial writes. A reduced public detector must also use a smaller return buffer.
- **Geometry and material coverage.** Check vacuum, registered isotropic material, and an oriented material filled using S=0 outside the crop. Preserve manual XY Hann/unwindowed selection, unwindowed Z on an actual extension, identity requests, odd centering, noninteger XYZ ratios and DC replacement. Include one reuse/release sequence and a low-memory retry. Keep a bit-preserving incumbent-transform fallback where needed and disclose its timing.

**Run.** The regular Z example has a genuine eightfold axial extension. The XYZ example uses a compact 64³ morphology. Each command can also run `--phase check` and `--phase diagnose`.

```bash
zpad=(--route z-sum z-residues --source 32 512 512 --target 256 512 512)
xyz=(--route xyz --source 64 64 64 --target 512 512 512)
python "$lab/run_public_pair.py" "${common[@]}" "${zpad[@]}" --phase time --energy-count 20 --q-max 0.8
python "$lab/run_public_pair.py" "${common[@]}" "${xyz[@]}" --phase memory --energy-count 20 --q-max 0.8
```

Include the noninteger rectangular XYZ case and the larger 2048/4096 detector controls in QM05. The principal comparison is end-to-end speed and memory for the same requested pixels. Count improvements from smaller projection and transfer even where the FFT stage is unchanged.
