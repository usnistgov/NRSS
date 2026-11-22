# QM03 · Manifest fewer detector pixels in native compact tensor

**Priority.** Measure the default, unpadded, full-depth FP32 compact-tensor calculation with energy reuse. Use a dense occupied morphology with the same physical and logical shape. Pin the parent GPU, source, dtype, optical tables, angles, input residency, output residency and chunk size. Exercise automatic dispatch, then check the recorded route and one basis build per scan.

| Arm | What changes | What to measure |
|---|---|---|
| A | Unbounded post-ZP15 parent | Full call, full returned bytes, basis, FFTs, projection, memory |
| B | Candidate with the option omitted | Full-output bitwise regression and timing drift |
| C | Same native material-field preparation and `rfftn`. Gather only requested Fourier endpoints, launch projection/rotation over demanded source/output pixels, allocate and transfer the public rectangle | Detector speedup and host/GPU peaks attributable to fewer pixels |
| D, optional | Limit selected FFT stages for narrow windows | Incremental benefit and bitwise gate against A and C |

Start with C. The existing native demand builder in the published source allocates a bitmap for the full `Nz × Ny × Nx` space. Test sparse or bounded column planning if diagnostics show meaningful cost. Keep the parent's shape, axes, plan, precision, field order, GPU math and DC replacement. A smaller endpoint table may use the post-ZP15 direct index layout. Projection reads must address the same complex amplitudes in the same operation order.

For D, a separable candidate runs X real FFTs, selects kx groups for Y, then transforms Z only for required (ky,kx) columns. cuFFT can select a different plan for different batch shapes or strides. The [CPU probe](../../../tests/validation/dev/cupy_rsoxs_qmax/probe_native_fft.py) demonstrates a bit-level discrepancy even when coefficients agree to numerical roundoff. Screen a non-identity cutoff against the complete parent before investing in tuning. **Any retained-output bit mismatch disqualifies that implementation for the affected dispatch.** Keep C available. An experiment that changes transforms is unnecessary if C provides the desired gain.

**Run.** From the candidate root, with the README's `lab` and `common` variables:

```bash
native=(--route native --source 128 512 512 --target 128 512 512 --angles 12 --layout detector)
python "$lab/run_public_pair.py" "${common[@]}" "${native[@]}" --phase check --energy-count 20 --q-max 0.8
python "$lab/run_public_pair.py" "${common[@]}" "${native[@]}" --phase time --energy-count 20 --q-max 0.8
python "$lab/run_public_pair.py" "${common[@]}" "${native[@]}" --phase memory --energy-count 20 --q-max 0.8
```

Pair one-energy and 101-energy cases as specified in QM05. Record delivered detector dimensions, host-resident and device-resident elapsed time, and both memory peaks. A host speedup accompanied by identical device timing still has value when the source is reduced host transfer. Report that attribution explicitly.
