# ZP04 · Window a finite film during virtual Z padding

**Addendum to the submitted ZP01–ZP03 series.** Apply this ticket in the existing virtual-padding worktree. ZP04 specifies the user-facing film window and adds a regression before push qualification. Completed window-off results carry forward. Record earlier XYZ Hann results as explicit-box compatibility evidence. The earlier tickets and reports can stay in place.

## Physical and API contract

Model one finite film in vacuum. For occupied depth `N` and requested transform depth `M>N`, place its first plane at `o=(M-N)//2`. An odd remainder gives the high-index side one extra vacuum plane. At 128→1024 the configuration is 448 vacuum planes, 128 occupied planes and 448 vacuum planes. At each axial frequency, sum the occupied film once. Its Fourier transform is sampled at intervals `2π/(M d)`. The periodic continuation implicit in a DFT is a representation of the box. A physical stack would require a separate coherent sum over film positions.

| Request | Lateral amplitude weight | Axial amplitude weight |
|---|---|---|
| Virtual film, `WindowingType=0` (existing configuration default) | 1 | 1 on every occupied plane |
| Virtual film, `WindowingType=1` | `hanning(Ny)[:,None] * hanning(Nx)[None,:]` | 1 on every occupied plane |
| Ordinary call, or identity pad `M=N` through the existing API | Existing parent behavior | Existing parent behavior |

The caller selects XY manually through `WindowingType`. Omission resolves to the fixed NRSS configuration default of `0` for every morphology. Use the requested setting regardless of apparent seams, particle positions, or pad depth. “Periodic XY” describes a possible reason for the user to choose an unwindowed transform. Any future edge assessment belongs to a separately requested, explicit `auto` mode. The geometry and window selectors are independent. Preserve the centering phase for the parent’s complex-neighbor DC replacement.

**Window placement.** Multiply each scalar or tensor scattering basis field by the lateral amplitude weight once before the XY FFT. Use the same weight on every occupied plane and across energies and polarizations. Preserve physical volume fractions and absolute scale. In the dense explicit-vacuum reference, apply precisely that XY weight with the backend’s existing XYZ Hann disabled. On a public reference call, use a reference window hook at the field/basis stage if needed. For a native-depth XY-Hann control, use the same explicitly weighted field at `M=N`. An ordinary native-depth run with `WindowingType=1` is the legacy XYZ-Hann control. Use each for its stated purpose.

## Acceptance and integration

1. **Verify where the weight acts.** Reuse [the small CPU oracle](../../../tests/validation/dev/cupy_rsoxs_virtual_zpad/check_film_window.py). A unit scatterer at lateral index (2,3) on a 9 × 13 grid has weight 1 with no window and ¼ with XY Hann. Its intensity at a non-DC Fourier bin must therefore be 1 or 1/16 whether it occupies the first, middle or last film plane. For two identical scatterers on the two faces, verify the thickness fringes against the coherent two-plane sum. The oracle includes the odd padding case `M=257` and checks complex phases. Run the analogous assertions on the actual GPU window/basis path and compare public detector outputs to B with matched optics, masks and projection.
2. **Keep the two questions separate in the results.** Use the same XY window for all depths in a physics sweep and for the direct continuous-qz reference D. Use unwindowed XY for the principal HCP/BCC comparisons already assigned. Add one representative tilted high-q case with XY Hann and compare its indexed peaks and absolute intensity to its own weighted D. Existing full-box Hann checks in ZP01 remain useful as explicit reference calculations. A selectable axial taper can await a scientific use case.
3. **Qualify the implemented route.** Record physical and transform shapes, offset and resolved XY/Z weights beside timings and results. From the public entry point, run both manual settings on both a laterally edge-matched specimen and one with an obvious lateral discontinuity. In both fixtures verify that omission matches explicit `0`, and that explicit `1` gives the prescribed XY factor. Include a scalar case, a complex anisotropic case and a multi-energy scan. Exercise fused and unfused paths where supported. Use the broadcast XY factor or fuse it into the occupied fields. Compare timings and peak VRAM under matched window choices. Apply the full ZP03/S07 promotion panels and the existing golden-data dispositions on the final revision. If ZP01–ZP03 already finished, append the new evidence to their ledger.

## Local execution

From the current virtual-padding branch root, unpack the two new files. Set `archive` to the downloaded addendum. The CPU reference runs standalone. The final command also checks the two NumPy transform methods from ZP01.

```bash
archive="$HOME/Downloads/NRSS_Virtual_Z_Padding_ZP04_Film_Window_Addendum.zip"
unzip -n "$archive"
lab=tests/validation/dev/cupy_rsoxs_virtual_zpad
python "$lab/check_film_window.py"
python "$lab/check_film_window.py" --with-zp01
```

Record CPU and GPU results in `test-reports/virtual-zpad/` and identify the executable revision in the ticket ledger. [Submitted series](README_FIRST.md) · [Pre-push qualification](ZP03-high-q.md)
