"""Tiny physical-box cross-check of the CPU voxel reference used by the sphere ladder."""
import argparse
import json

import numpy as np
from scipy import fft

from residue_pilot import full_detector
from sphere_reference import (CROP, K, MEDIUM, PARTICLE, VACUUM, axis, compare_voxel_panels,
                              discrete_anchors, forward_scale, material_response, sphere_fraction)


def literal_detector(fraction, size, fill_id):
    mid = size // 2
    offset = (size - CROP) // 2
    inside = slice(offset, offset + CROP)
    sphere = material_response(PARTICLE) / (4*np.pi)
    matrix = material_response(MEDIUM if fill_id == 2 else VACUUM) / (4*np.pi)
    field = np.full((size, size, size), matrix, dtype=np.complex128)
    field[inside, inside, inside] = sphere*fraction.astype(np.float64) + matrix*(1-fraction.astype(np.float64))
    transformed = fft.fftn(field, overwrite_x=True, workers=1)
    neighbors = ((0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0))
    transformed[0, 0, 0] = np.mean([transformed[point] for point in neighbors])

    q = axis(size)
    qy, qx = np.meshgrid(q, q, indexing="ij")
    radius_sq = qx**2 + qy**2
    valid = (radius_sq.astype(np.float32) <= np.float32(K*K))
    valid[-1, :] = False
    valid[:, -1] = False
    y, x = np.nonzero(valid)
    qx, qy = q[x], q[y]
    dq = q[mid+1]
    zpos = -radius_sq[y, x] / (K + np.sqrt(np.maximum(K*K-radius_sq[y, x], 0))) / dq
    low = np.floor(zpos).astype(int)
    frac = zpos - low
    result = np.full((size, size), np.nan)
    intensity = np.zeros(len(x))
    for z, weight in ((low, 1-frac), (low+1, frac)):
        # The Igor coordinate uses the negative native FFT index.
        amplitude = transformed[(-z) % size, (mid-y) % size, (mid-x) % size]
        projected = np.stack(((K*K-qx*qx)*amplitude, -qx*qy*amplitude,
                              -qx*(K+dq*z)*amplitude), axis=-1)
        intensity += weight * np.sum(abs(projected)**2, axis=-1)
    result[y, x] = intensity
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, choices=(128, 256), default=(128, 256))
    args = parser.parse_args()
    fraction = sphere_fraction()
    records = []
    for size in args.sizes:
        for fill_id in (None, 2):
            literal = literal_detector(fraction, size, fill_id)
            compact = full_detector(fraction, size, fill_id)
            stats = compare_voxel_panels(compact, literal, size, fill_id)
            pixels, direct = discrete_anchors(fraction, size, fill_id)
            anchor_error = float(np.max(abs(literal[tuple(pixels.T)]-direct)) / forward_scale(fill_id))
            stats.update(M=size, fill_id=fill_id, literal_vs_direct_anchor_over_I0=anchor_error)
            if max(stats["max_abs_over_peak"], stats["relative_l2"], anchor_error) > 2e-5:
                raise AssertionError(stats)
            records.append(stats)
    print(json.dumps(records, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
