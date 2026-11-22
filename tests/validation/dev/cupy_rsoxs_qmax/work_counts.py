"""Count geometric work for the published lipid benchmark using only NumPy."""
import argparse
import json

import numpy as np


def count(limit, margin):
    nz, ny, nx = 256, 1024, 1024
    pitch = 1.5
    dq = 2 * np.pi / (nx * pitch)
    dz = 2 * np.pi / (nz * pitch)
    qx = (np.arange(nx) - nx // 2) * dq
    radius_sq = (qx[None, :] ** 2 + qx[:, None] ** 2).astype(np.float32)
    # The source projector excludes the last detector row and column.
    radii, multiplicity = np.unique(radius_sq[:-1, :-1], return_counts=True)
    bins = int(np.ceil(np.hypot(ny / 2, nx / 2)))
    native_qr = np.arange(bins) * (np.hypot(ny / 2 * dq, nx / 2 * dq) / bins)
    mid = nz // 2
    dc = {(mid, ny // 2, nx // 2)}
    for dim in range(3):
        for sign in (-1, 1):
            neighbor = [mid, ny // 2, nx // 2]
            neighbor[dim] += sign
            dc.add(tuple(neighbor))
    def count_union(seen):
        touched = int(seen.sum(axis=0, dtype=np.int64) @ multiplicity)
        extra = sum(not seen[z, np.searchsorted(radii, radius_sq[y, x])] for z, y, x in dc)
        return touched + extra

    seen = {name: np.zeros((nz, len(radii)), dtype=bool) for name in ("full", "cut", "guard")}
    visits = dict.fromkeys(seen, 0)
    one = {}
    upper_q = []
    envelope_sq = 0.
    returned_radius = 0.
    for cents in range(28255, 29256, 10):
        k = 2 * np.pi / (1239.84197 / (cents / 100))
        depth = min(k, mid * dz)
        supported = radii <= np.float32(depth * (2 * k - depth))
        root = np.sqrt(np.maximum(np.float32(k * k) - radii, np.float32(0.)))
        offset = -radii / (np.float32(k) + root) / np.float32(dz)
        lower = np.floor(np.maximum(offset, np.float32(-mid))).astype(np.int32) + mid
        upper = np.minimum(lower + 1, nz - 1)
        # The forward Ewald branch tops out at |q| = sqrt(2) k.
        cap = min(limit, np.sqrt(2.) * k)
        transverse = cap * np.sqrt(1 - (cap / (2 * k)) ** 2)
        returned_radius = max(returned_radius, transverse)
        guarded_sq = (transverse + margin * dq) ** 2
        envelope_sq = max(envelope_sq, min(guarded_sq, depth * (2 * k - depth)))
        regions = {"full": supported, "cut": supported & (radii <= transverse ** 2),
                   "guard": supported & (radii <= guarded_sq)}
        for name, valid in regions.items():
            idx = np.flatnonzero(valid)
            seen[name][lower[idx], idx] = True
            seen[name][upper[idx], idx] = True
            pixels = int(multiplicity[idx].sum())
            visits[name] += pixels
            if cents == 28755:
                frame = np.zeros_like(seen[name])
                frame[lower[idx], idx] = frame[upper[idx], idx] = True
                one[name] = {"valid_pixels": pixels, "bins_with_dc": count_union(frame)}
        r = native_qr[native_qr <= k]
        q_abs = np.hypot(r, np.sqrt(k * k - r * r) - k)
        upper_q.append(q_abs[-1])
    shared = np.linspace(0., min(upper_q), bins)
    kept = shared[shared <= limit]
    # Fold only the real geometry fields into the native half-spectrum.
    ys, xs = np.nonzero(radius_sq[:-1, :-1] <= envelope_sq)
    ys = np.r_[ys, [y for _, y, _ in dc]]
    xs = np.r_[xs, [x for _, _, x in dc]]
    ky = (ny // 2 - ys) % ny
    kx = (nx // 2 - xs) % nx
    reflect = kx > nx // 2
    kx = np.where(reflect, -kx, kx) % nx
    ky = np.where(reflect, -ky, ky) % ny
    pairs = np.unique(ky * (nx // 2 + 1) + kx)
    side = int(np.count_nonzero(np.abs(qx) <= returned_radius))
    return {
        "q_max_nm_inv": limit, "guard_pixels_this_square_fixture": margin,
        "logical_zyx": [nz, ny, nx], "voxel_nm": pitch, "energies": 101,
        "one_energy_287_55": one,
        "scan": {key: {"pixel_visits": visits[key], "bins_with_dc": count_union(value)}
                 for key, value in seen.items()},
        "reduced_grid": {"total": bins, "kept": len(kept), "last": float(kept[-1])},
        "detector_return": {"full_yx": [ny, nx], "limited_yx": [side, side],
                            "full_mib": 101*ny*nx*4 / 2**20, "limited_mib": 101*side*side*4 / 2**20},
        "guarded_native_fft": {"kx_kept": len(np.unique(kx)), "kx_total": nx // 2 + 1,
                               "ky_kx_kept": len(pairs), "ky_kx_total": ny * (nx // 2 + 1)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-max", type=float, default=0.8)
    parser.add_argument("--guard-pixels", type=int, default=4)
    args = parser.parse_args()
    if not np.isfinite(args.q_max) or args.q_max < 0 or args.guard_pixels < 0:
        parser.error("Use a finite nonnegative cutoff and nonnegative margin")
    result = count(args.q_max, args.guard_pixels)
    if (args.q_max, args.guard_pixels) == (0.8, 4):
        assert result["scan"]["full"] == {"pixel_visits": 40270985, "bins_with_dc": 2017475}
        assert result["scan"]["guard"] == {"pixel_visits": 11702929, "bins_with_dc": 263519}
        assert result["reduced_grid"]["kept"] == 297
        assert result["detector_return"]["limited_yx"] == [377, 377]
        assert result["guarded_native_fft"]["ky_kx_kept"] == 58291
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
