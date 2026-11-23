# FS01 · Extract the prepared Fourier sampler

**Deliverable.** Expose the scalar field sampler in `NRSS.fourier` and route the existing compact real-basis build through that implementation. The downstream consumer will supply optical physics, incident polarization, detector rays, interpolation weights and any orientation averaging. This ticket concerns integer Fourier-grid samples. Keep changes to the NRSS production tree in one independently reviewable refactor.

## Proposed public contract

```python
from NRSS.fourier import prepare_sampler

plan = prepare_sampler(
    shape_zyx=field.shape,
    logical_shape_zyx=(256, 1024, 1024),
    modes_zyx=modes,                 # (M, 3) signed integer FFT modes
    dc="raw",
)
plan.capture_into(field, out=amplitudes)
```

| Item | Contract |
| --- | --- |
| Fields and result | For this first API, accept a C-contiguous CuPy FP32 scalar field in ZYX order and a caller-owned, same-device, contiguous `complex64` vector of length M. Validate domain, dtype, shape and device. The input can be reused. Write directly into `out` and document any return value. |
| Grid and origin | `logical_shape_zyx` defaults to the field shape. For virtual zero extension, place the field at integer offsets `(logical_shape − field_shape) // 2` per axis. Use the negative-exponential forward DFT, unnormalized. No voxel-volume factor. At each signed mode `m`, return the FFT of that centered logical grid at native index `m mod N`. |
| Mode list | Accept integer triples in ZYX order, supplied on host or on the selected device at preparation. Define the signed interval per axis as `[-floor(N/2), ceil(N/2)-1]`, including negative Nyquist on even axes. Preserve caller order and repetitions. Convert external requests to native R2C half-grid locations once. On Hermitian reflection negate all three indices before conjugation. Internal RSoXS endpoint IDs have their own registered Igor convention. |
| Forward bin | `dc="raw"` returns the actual zero mode, equal to the sum of the real field. `dc="nrss"` requests the incumbent neighbor-repaired value. The NRSS caller explicitly preserves the legacy policy and its operation order. A public request can omit DC or ask for DC alone. Resolve any internal stencil points during preparation, then return only the requested points. |
| Lifetime and cost | Bind the prepared plan to its geometry, point order, pad convention, DC policy and device. Support repeated calls with different fields and output rows on the creating stream. Honor CuPy stream ordering. Keep GPU arrays on-device during capture. A CPU has no need to import CuPy at top-level just to import NRSS or its documentation. |

A downstream complex field uses two real captures and combines their complex results as `F(real) + 1j*F(imag)`. Hermitian reconstruction applies separately to each real capture. A plain public call uses zero exterior and the supplied field values. RSoXS retains its existing material-fill arithmetic and window preparation. The interface contains no photon energy, morphology, optical constants, reciprocal-space rotation, DDA solver, or detector model.

## Source map at the reviewed commit

| Existing code | Implementation detail to preserve |
| --- | --- |
| `src/NRSS/backends/_cupy_energy_basis.py`, `build_energy_basis` | The nested `capture(row)` reads a prepared work buffer and fills `values[row]`. `capture_scalar` can call the virtual evaluator directly for a fill row. Extract both handoffs as needed. Retain fused scalar/tensor XY windowing and material-row pruning ahead of capture. |
| Same module, `_native_indices` and `_transform_basis` | Existing native indices reflect negative X in Z, Y and X. The native R2C helper currently performs DC repair unconditionally. Make raw DC possible through the public sampler and retain the legacy arithmetic on RSoXS calls. The optional in-place C2C route operates on a private work buffer. |
| `_cupy_virtual_z.py` and `_cupy_virtual_xyz.py` | The virtual evaluator already supports `repair_dc`. Preserve native versus logical offsets, fill-box correction, XY Hann, selected columns, Z sum/residues, tiling and allocation fallbacks. Keep evaluator and recorder reuse intact. |
| `_cupy_xyz_endpoints.py`, `_cupy_q_window.py`, `_cupy_xyz_policy.py` | NRSS already planned its demand. Provide an internal construction path that adopts those index arrays, spans, stencil positions, full-support `transform_kx` and chosen evaluator by reference. Reconstructing an `M × 3` external mode list on this path would cost time and memory. |
| `cupy_rsoxs.py`, `_cupy_energy_basis.py`, `_cupy_energy_tiles.py` | Keep complex samples in the compact basis buffer for existing across-energy reuse. The existing fused projection, tile selection, E-angle summation and output chunking consume that buffer as before. |

Use one capture implementation behind public preparation and internal adoption of a prepared plan. Keep mode validation, deduplication, registration, and convenience allocation outside the RSoXS per-row loop. In the common RSoXS path, the implementation should issue the same FFT calls, gathers, kernels and device transfers into the same preallocated row. Preserve native dense R2C where selected. A factored virtual job should retain its compact representation. Leave material mixing and detector intensity interpolation in their existing positions.

## Acceptance

1. Fresh external code can import the public function and sample two independent FP32 fields using one plan and reusable output. It needs only array geometry and modes. Include a short API example giving axes, sign, origin, DC, normalization, shape/device requirements and stream lifetime.
2. Add small GPU regression tests through the public API. The bundled `check_sampling.py` provides an independent complex128 direct sum and a literal padded FFT over odd/even, native/Z/XYZ and singleton-Z geometries, mixed-sign modes, Nyquist, repeated modes, phase-sensitive translation, raw DC and two-field complex reconstruction. Its scaled FP32 bound applies to the new mathematical oracle. Also check a DC-only raw request, an off-center list with no DC, and, for supported geometry, `nrss`-policy DC against the parent's actual axial-neighbor rule. Exercise clear rejection of invalid mode/dtype/output shapes.
3. Compare the internal legacy samples for representative native and virtual rows bit for bit with the parent on the same GPU. Include fill, isotropic `S` sentinel, nontrivial director, XY Hann, a nonidentity E-angle on an odd rectangular grid, and `q_max` demand including the last interpolation neighbor. Existing focused tests cover several of these. Use the repository's CPU and GPU physics regressions as specified in FS02.
4. Provide a call-site diff showing the real RSoXS hot path consumes the shared sampler. Show that it adopts existing device buffers and has no per-field plan rebuild, extra full-grid allocation, unconditional synchronizing validation/reduction, host round trip, or extra detector intermediate. Use FS02 for the actual performance decision.

[Capture and fill paths](https://github.com/usnistgov/NRSS/blob/0727a5948af63ba35e9bcf12722aabd424ebd5d2/src/NRSS/backends/_cupy_energy_basis.py#L435-L513) · [Native registration and DC](https://github.com/usnistgov/NRSS/blob/0727a5948af63ba35e9bcf12722aabd424ebd5d2/src/NRSS/backends/_cupy_energy_basis.py#L231-L265) · [Virtual DC and fill](https://github.com/usnistgov/NRSS/blob/0727a5948af63ba35e9bcf12722aabd424ebd5d2/src/NRSS/backends/_cupy_virtual_z.py#L78-L116)
