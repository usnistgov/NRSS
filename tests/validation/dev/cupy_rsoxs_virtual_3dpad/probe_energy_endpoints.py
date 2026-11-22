"""Measure exact detector endpoint overlap for a proposed scan using only NumPy.

Reproduces the registered forward geometry from the pinned branch. Counts are
CPU planning evidence. There is no GPU simulation or morphology construction.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def axis(length, pitch):
    step = 2 * np.pi / (length * pitch)
    return (np.arange(length, dtype=np.float64) - length // 2) * step


def geometry(radius_sq, border, energy, depth, pitch):
    k = 2 * np.pi / (1239.84197 / float(energy))
    dz = 2 * np.pi / (depth * pitch)
    mid = depth // 2
    limit = min(k, mid * dz) if mid else k
    valid = border & (radius_sq <= np.float32(limit * (2*k-limit)))
    if depth == 1:
        lo = np.zeros(valid.shape, dtype=np.int32)
        return valid, lo, lo
    root = np.sqrt(np.maximum(np.float32(k*k) - radius_sq, np.float32(0)))
    pos = (-radius_sq / (np.float32(k) + root)) / np.float32(dz)
    lo = np.floor(np.maximum(pos, np.float32(-mid))).astype(np.int32) + mid
    return valid, lo, np.minimum(lo+1, depth-1)


def dc_stencil(shape):
    center = np.asarray(shape) // 2
    steps = np.vstack((np.zeros((1, 3), dtype=int), np.eye(3, dtype=int), -np.eye(3, dtype=int)))
    return {tuple(map(int, (center + s) % shape)) for s in steps}


def increments(low, high, previous_high):
    """Count new positions on sorted, nondecreasing lower/upper Z runs."""
    return np.maximum(0, high - np.maximum(low-1, previous_high))


def probe(shape, pitch, requested, widths=(1, 2, 4, 8), rows_per_block=16, count_radii=False):
    nz, ny, nx = shape
    if min(shape) < 1 or not np.isfinite(pitch) or pitch <= 0 or rows_per_block < 1:
        raise ValueError("Use positive shape, pitch and row count")
    if not requested or any(not np.isfinite(e) or e <= 0 for e in requested):
        raise ValueError("Provide at least one finite positive energy")
    if not widths or min(widths) < 1:
        raise ValueError("Positive tile widths required")
    energies = sorted(set(requested))
    qx, qy = axis(nx, pitch), axis(ny, pitch)
    stencil = dc_stencil(shape)
    seen_stencil = set()
    used = envelope = padded_envelope = touched = gaps = valid_total = 0
    comparable = same_pair = any_shared = 0
    tile_totals = {w:0 for w in widths}
    radius_keys = set()
    for start in range(0, ny, rows_per_block):
        stop = min(ny, start+rows_per_block)
        radius_sq = (qx[None, :]**2 + qy[start:stop, None]**2).astype(np.float32)
        border = (np.arange(start,stop)[:,None] != ny-1) & (np.arange(nx)[None,:] != nx-1)
        last_lo = np.full(radius_sq.shape, -1, dtype=np.int32)
        last_hi = np.full_like(last_lo, -1)
        first_lo = np.full_like(last_lo, nz)
        counts = np.zeros_like(last_lo)
        tile_hi = {w:np.full_like(last_lo, -1) for w in widths}
        local_stencil = [(z,y-start,x) for z,y,x in stencil if start <= y < stop]
        for i, energy in enumerate(energies):
            valid, low, high = geometry(radius_sq, border, energy, nz, pitch)
            previous = last_hi >= 0
            if np.any(previous & (~valid | (low < last_lo) | (high < last_hi))):
                raise ValueError("Sorted masks/bins violated monotonicity. Use an exact general union on these inputs.")
            valid_total += int(valid.sum())
            comparable += int(previous.sum())
            same_pair += int(np.count_nonzero(previous & (low == last_lo) & (high == last_hi)))
            any_shared += int(np.count_nonzero(previous & (low <= last_hi)))
            counts += np.where(valid, increments(low, high, last_hi), 0)
            for w in widths:
                if i % w == 0:
                    tile_hi[w].fill(-1)
                addition = increments(low, high, tile_hi[w])
                tile_totals[w] += int(np.sum(addition, where=valid, dtype=np.int64))
                np.copyto(tile_hi[w], high, where=valid)
            np.copyto(first_lo, low, where=(~previous & valid))
            np.copyto(last_lo, low, where=valid)
            np.copyto(last_hi, high, where=valid)
            for z, y, x in local_stencil:
                if valid[y,x] and low[y,x] <= z <= high[y,x]:
                    seen_stencil.add((z,y+start,x))
        touched += int(np.count_nonzero(counts))
        if count_radii:
            radius_keys.update(map(int, np.unique(radius_sq[counts > 0].view(np.uint32))))
        span = np.maximum(0, last_hi-first_lo+1)
        used += int(counts.sum(dtype=np.int64))
        envelope += int(span.sum(dtype=np.int64))
        gaps += int(np.count_nonzero(span > counts))
        for z,y,x in local_stencil:
            first_lo[y,x] = min(first_lo[y,x], z)
            last_hi[y,x] = max(last_hi[y,x], z)
        padded_envelope += int(np.maximum(0,last_hi-first_lo+1).sum(dtype=np.int64))
    requests = valid_total * (1 if nz == 1 else 2)
    def ratio(num, den):
        return num / den if den else None
    result = {
        "shape_zyx":list(shape), "pitch_nm":pitch, "requested_energies_eV":list(requested),
        "sorted_distinct_energies":len(energies), "envelope_support_columns":touched,
        "endpoint_visits_distinct_energies":requests, "union_before_dc":used,
        "union_including_dc":used + len(stencil - seen_stencil), "missing_dc_bins_added":len(stencil-seen_stencil),
        "envelope_before_dc":envelope, "envelope_including_dc":padded_envelope,
        "excess_envelope_bins_including_dc":padded_envelope - (used+len(stencil-seen_stencil)),
        "columns_with_gaps_before_dc":gaps, "adjacent_jointly_valid_pixels":comparable,
        "same_pair_fraction":ratio(same_pair, comparable), "at_least_one_shared_fraction":ratio(any_shared, comparable),
        "tile_distinct_endpoints_per_group_summed":tile_totals,
        "potential_endpoint_read_reuse_per_tile":{w:ratio(requests,total) for w,total in tile_totals.items()},
    }
    if count_radii:
        result["distinct_valid_fp32_radius_sq"] = len(radius_keys)
        result["columns_per_distinct_valid_radius"] = ratio(touched, len(radius_keys))
    return result


def literal_check(shape, pitch, requested):
    qx, qy = axis(shape[2],pitch), axis(shape[1],pitch)
    radius = (qx[None,:]**2 + qy[:,None]**2).astype(np.float32)
    border = np.ones(radius.shape, bool)
    border[-1], border[:,-1] = False, False
    masks, endpoints = [], set()
    for energy in sorted(set(requested)):
        valid, lo, hi = geometry(radius, border, energy, shape[0], pitch)
        arm = {(z, int(y), int(x)) for y,x in np.argwhere(valid) for z in range(int(lo[y,x]),int(hi[y,x])+1)}
        masks.append(arm)
        endpoints.update(arm)
    enriched = endpoints | dc_stencil(shape)
    columns = {}
    for z,y,x in enriched:
        columns.setdefault((y,x),set()).add(z)
    span = sum(max(v)-min(v)+1 for v in columns.values())
    result = probe(shape,pitch,requested,rows_per_block=3)
    assert result["union_before_dc"] == len(endpoints)
    assert result["union_including_dc"] == len(enriched)
    assert result["envelope_including_dc"] == span
    for w, actual in result["tile_distinct_endpoints_per_group_summed"].items():
        assert actual == sum(len(set().union(*masks[i:i+w])) for i in range(0,len(masks),w))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=3, type=int, default=(256,1024,1024), metavar=("Z","Y","X"))
    parser.add_argument("--pitch", type=float, default=1.5, help="Isotropic voxel pitch in nm")
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--energies", type=float, nargs="+")
    group.add_argument("--energies-json", type=Path, help="File containing a JSON list of energies in eV")
    group.add_argument("--self-test", action="store_true")
    parser.add_argument("--block-rows", type=int, default=16)
    parser.add_argument("--count-radii", action="store_true", help="Count bit-identical FP32 squared radii on the union support")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.self_test:
        trial = [285., 690., 120., 1700., 282., 285.]
        results = [literal_check(shape,1.5,trial) for shape in ((1,8,7),(12,15,16),(201,17,20))]
        assert any(v["excess_envelope_bins_including_dc"] > 0 for v in results)
        data = {"checked_shapes":[v["shape_zyx"] for v in results],
                "wide_scan_extra_bins":[v["excess_envelope_bins_including_dc"] for v in results]}
    else:
        if args.energies_json is not None:
            energies = json.loads(args.energies_json.read_text())
        else:
            energies = args.energies or [(28255+10*i)/100 for i in range(101)]
        data = probe(tuple(args.shape),args.pitch,energies,rows_per_block=args.block_rows,count_radii=args.count_radii)
    rendered = json.dumps(data, indent=2, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
