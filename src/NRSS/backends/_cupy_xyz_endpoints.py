"""Endpoint layouts for the virtual XYZ Fourier basis."""

from dataclasses import dataclass
from typing import Any

import numpy as np

XYZ_ENDPOINT_PLAN_KEY = "_private_xyz_endpoint_plan"
_BITMAP_LIMIT_BYTES = 128 * 1024**2
_EXTRA_BASIS_LIMIT_BYTES = 32 * 1024**2
_KERNELS = None


@dataclass(frozen=True)
class ColumnSpans:
    first: Any
    offsets: Any

    def positions(self, endpoints):
        bins = endpoints // self.first.size
        columns = endpoints % self.first.size
        return self.offsets[columns] + bins - self.first[columns]

    @property
    def nbytes(self):
        return self.first.nbytes + self.offsets.nbytes


def _endpoint_kernels(cp):
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = (
            cp.ElementwiseKernel(
                "bool valid, int32 lower, int32 upper", "int32 first, int32 last, int32 count",
                """
                if (valid) {
                    count += max(0, upper - max(last, lower - 1));
                    first = min(first, lower);
                    last = max(last, upper);
                }
                """, "nrss_xyz_update_spans",
            ),
            cp.ElementwiseKernel(
                "int32 first, raw int64 offsets, int64 plane", "raw int64 endpoints",
                """
                for (long long pos = offsets[i]; pos < offsets[i + 1]; ++pos) {
                    const long long bin = first + (pos - offsets[i]);
                    endpoints[pos] = bin * plane + i;
                }
                """, "nrss_xyz_expand_spans",
            ),
            cp.ElementwiseKernel(
                "bool valid, int32 lower, int32 upper, int64 plane", "raw uint32 words",
                """
                if (valid) {
                    words[(long long)(lower / 32) * plane + i] |= 1u << (lower % 32);
                    words[(long long)(upper / 32) * plane + i] |= 1u << (upper % 32);
                }
                """, "nrss_xyz_update_bitmap",
            ),
            cp.ElementwiseKernel(
                "raw uint32 words, int32 depth, int64 plane", "int32 count",
                """
                count = 0;
                for (int z = 0; z < depth; ++z) {
                    count += __popc(words[(long long)z * plane + i]);
                }
                """, "nrss_xyz_count_bitmap",
            ),
            cp.ElementwiseKernel(
                "raw uint32 words, raw int64 offsets, int32 depth, int64 plane",
                "raw int64 endpoints",
                """
                long long pos = offsets[i];
                for (int z = 0; z < depth; ++z) {
                    unsigned int word = words[(long long)z * plane + i];
                    while (word) {
                        const long long bin = (long long)z * 32 + __ffs(word) - 1;
                        endpoints[pos++] = bin * plane + i;
                        word &= word - 1;
                    }
                }
                """, "nrss_xyz_expand_bitmap",
            ),
        )
    return _KERNELS


def _accumulate(runtime, morphology, cp, energies, detector, shape, recorder, update, pixel_region=None):
    region_options = {} if pixel_region is None else {"pixel_region": pixel_region}
    for energy in sorted(set(energies)):
        try:
            projection = recorder.measure(
                "zpad_plan_geometry",
                lambda: runtime._detector_projection_geometry(
                    morphology, energy, cp, detector, shape_override=shape, **region_options,
                ),
            )
            if projection.safe_z0 is None:
                lower = upper = np.int32(0)
            else:
                lower = projection.safe_z0.ravel()
                upper = projection.safe_z1.ravel()
            with recorder.range("zpad_endpoint_merge"):
                update(projection.valid.ravel(), lower, upper)
        finally:
            runtime._discard_detector_projection_geometry(morphology, energy, shape_override=shape)


def _offsets(cp, counts):
    offsets = cp.empty(counts.size + 1, dtype=cp.int64)
    offsets[0] = 0
    cp.cumsum(counts, dtype=cp.int64, out=offsets[1:])
    return offsets


def _workspace_available(cp, size):
    free_bytes, _ = cp.cuda.runtime.memGetInfo()
    reusable_bytes = cp.get_default_memory_pool().free_bytes()
    return size <= (free_bytes + reusable_bytes) // 4


def _plan_spans(runtime, morphology, cp, energies, detector, shape, stencil, fields, recorder, pixel_region=None):
    nz, ny, nx = shape
    plane = ny * nx
    if not _workspace_available(cp, 24 * plane + 8):
        return None, None, {"fallback": "span workspace"}
    update, expand, *_ = _endpoint_kernels(cp)
    first = cp.full(plane, nz, dtype=cp.int32)
    last = cp.full(plane, -1, dtype=cp.int32)
    exact_counts = cp.zeros(plane, dtype=cp.int32)
    if pixel_region is None:
        update_spans = lambda valid, lo, hi: update(valid, lo, hi, first, last, exact_counts)
    else:
        views = tuple(values.reshape(ny, nx)[pixel_region.slices] for values in (first, last, exact_counts))

        def update_spans(valid, lo, hi):
            if nz > 1:
                lo, hi = lo.reshape(pixel_region.shape), hi.reshape(pixel_region.shape)
            update(valid.reshape(pixel_region.shape), lo, hi, *views)

    _accumulate(runtime, morphology, cp, energies, detector, shape, recorder, update_spans, pixel_region)

    with recorder.range("zpad_endpoint_merge"):
        # Combine stencil entries per column before assigning on the GPU.
        # Repeated IDs still remain in the DC repair stencil.
        bins, columns = np.divmod(stencil, plane)
        touched, inverse = np.unique(columns, return_inverse=True)
        low = np.full(touched.size, nz, dtype=np.int32)
        high = np.full(touched.size, -1, dtype=np.int32)
        np.minimum.at(low, inverse, bins)
        np.maximum.at(high, inverse, bins)
        touched = cp.asarray(touched)
        first[touched] = cp.minimum(first[touched], cp.asarray(low))
        last[touched] = cp.maximum(last[touched], cp.asarray(high))
        counts = cp.maximum(last - first + 1, 0)
        offsets = _offsets(cp, counts)
        detector_endpoints, span_endpoints = map(int, cp.asnumpy(cp.stack((
            cp.sum(exact_counts, dtype=cp.int64), offsets[-1],
        ))))

        # On the forward branch both endpoints advance with increasing
        # energy. If rounding ever moves an endpoint backward, the incremental
        # count can undercount. The independent min/max still cover every bin
        # and the resulting conservative cost estimate favors the fallback.
        extra = span_endpoints - detector_endpoints
        extra_bytes = extra * (np.dtype(np.complex64).itemsize * fields + np.dtype(np.int64).itemsize)
        planner_bytes = first.nbytes + last.nbytes + exact_counts.nbytes + counts.nbytes + offsets.nbytes
        stats = dict(detector_endpoints=detector_endpoints, span_endpoints=span_endpoints,
                     extra_endpoints_upper_bound=extra, extra_basis_bytes_upper_bound=extra_bytes,
                     planner_arrays_bytes=planner_bytes)
        if extra_bytes > _EXTRA_BASIS_LIMIT_BYTES or extra > max(len(stencil), detector_endpoints // 8):
            return None, None, {**stats, "fallback": "span overhead"}
        indices = cp.empty(span_endpoints, dtype=cp.int64)
        expand(first, offsets, np.int64(plane), indices)
    return indices, ColumnSpans(first, offsets), stats


def _plan_bitmap(runtime, morphology, cp, energies, detector, shape, recorder, pixel_region=None):
    nz, ny, nx = shape
    logical_plane = ny * nx
    plane = logical_plane if pixel_region is None else pixel_region.size
    depth = (nz + 31) // 32
    size = depth * plane * np.dtype(np.uint32).itemsize
    if size > _BITMAP_LIMIT_BYTES or not _workspace_available(cp, size + 12 * plane + 8):
        return None, {"bitmap_bytes": size, "fallback": "bitmap workspace"}
    _, _, update, count_bits, expand = _endpoint_kernels(cp)
    words = cp.zeros((depth, plane), dtype=cp.uint32)
    _accumulate(runtime, morphology, cp, energies, detector, shape, recorder,
                lambda valid, lo, hi: update(valid, lo, hi, np.int64(plane), words), pixel_region)
    with recorder.range("zpad_endpoint_merge"):
        counts = cp.empty(plane, dtype=cp.int32)
        count_bits(words, np.int32(depth), np.int64(plane), counts)
        offsets = _offsets(cp, counts)
        total = int(offsets[-1])
        indices = cp.empty(total, dtype=cp.int64)
        expand(words, offsets, np.int32(depth), np.int64(plane), indices, size=plane)
        if pixel_region is not None:
            bins, columns = cp.divmod(indices, plane)
            indices = bins * logical_plane + pixel_region.global_columns(columns, nx)
    return indices, dict(detector_endpoints=total, bitmap_bytes=size,
                         planner_arrays_bytes=words.nbytes + counts.nbytes + offsets.nbytes)


def xyz_endpoints(
    runtime, morphology, cp, energies, detector, shape, stencil, fields, recorder, *,
    requested=None, pixel_region=None,
):
    if requested is None:
        requested = morphology._backend_runtime_state.get(XYZ_ENDPOINT_PLAN_KEY, "legacy")
    if requested not in {"legacy", "bitmap", "spans"}:
        raise ValueError(f"Unknown private XYZ endpoint plan {requested!r}.")
    stats = {"requested": requested, "layout": "legacy"}
    region_options = {} if pixel_region is None else {"pixel_region": pixel_region}
    if requested == "spans":
        try:
            indices, spans, details = _plan_spans(
                runtime, morphology, cp, energies, detector, shape, stencil, fields, recorder,
                **region_options,
            )
        except cp.cuda.memory.OutOfMemoryError:
            details = {"fallback": "span allocation failed"}
        else:
            if spans is not None:
                return indices, spans, {**stats, **details, "layout": "spans"}
        stats["span_screen"] = details

    if requested in {"bitmap", "spans"}:
        try:
            indices, details = _plan_bitmap(runtime, morphology, cp, energies, detector, shape, recorder, **region_options)
            if indices is not None:
                with recorder.range("zpad_endpoint_merge"):
                    indices = cp.union1d(indices, cp.asarray(stencil))
                return indices, None, {**stats, **details, "layout": "bitmap"}
        except cp.cuda.memory.OutOfMemoryError:
            indices = None
            details = {"fallback": "bitmap allocation failed"}
        stats.update(details)

    from ._cupy_virtual_z import sparse_endpoint_union

    indices = sparse_endpoint_union(
        runtime, morphology, cp, energies, detector, shape, recorder, **region_options,
    )
    with recorder.range("zpad_endpoint_merge"):
        return cp.union1d(indices, cp.asarray(stencil)), None, stats
