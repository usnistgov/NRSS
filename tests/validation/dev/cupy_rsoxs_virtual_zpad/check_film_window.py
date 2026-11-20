"""NumPy reference for a centered film with selectable XY and unit Z weights.

Run with --with-zp01 in the same directory as the submitted check_fft.py to
also compare the selected-sum and residue-FFT implementations.
"""
import argparse
import json

import numpy as np


NZ, NY, NX = 128, 9, 13
Y, X = 2, 3
KY, KX = 1, 2  # Native transverse bin away from the replaced DC coefficient.


def face_fixture(depth, planes, lateral):
    offset = (depth - NZ) // 2
    padded = np.zeros((depth, NY, NX), dtype=float)
    for local_z in planes:
        padded[offset + local_z, Y, X] = 1.0
    if lateral == "hann":
        padded *= np.hanning(NY)[None, :, None] * np.hanning(NX)[None, None, :]
    return padded, offset


def expected_coefficients(depth, offset, planes, lateral):
    # At the chosen voxel Hann(9)[2] = Hann(13)[3] = 1/2.
    amplitude = 0.25 if lateral == "hann" else 1.0
    m = np.arange(depth)
    lateral_phase = KY * Y / NY + KX * X / NX
    expected = np.zeros(depth, dtype=complex)
    for local_z in planes:
        phase = m * (offset + local_z) / depth + lateral_phase
        expected += amplitude * np.exp(-2j * np.pi * phase)
    return expected


def check(with_zp01=False):
    if with_zp01:
        from check_fft import selected

    counts = {"dense": 0, "sum": 0, "residues": 0}
    worst = {}
    for depth in (128, 256, 257, 512, 1024):
        modes = np.arange(depth)
        points = np.column_stack((modes, np.full(depth, KY), np.full(depth, KX)))
        for lateral in ("none", "hann"):
            for planes in ((0,), (NZ//2,), (NZ-1,), (0, NZ-1)):
                padded, offset = face_fixture(depth, planes, lateral)
                expected = expected_coefficients(depth, offset, planes, lateral)
                calculated = {"dense": np.fft.fftn(padded)[:, KY, KX]}
                if with_zp01:
                    occupied = padded[offset:offset+NZ]
                    calculated["sum"] = selected(occupied, depth, points, offset, "sum")
                    if depth % NZ == 0:
                        calculated["residues"] = selected(occupied, depth, points, offset, "residues")
                for route, values in calculated.items():
                    np.testing.assert_allclose(values, expected, atol=3e-12, rtol=2e-12,
                        err_msg=f"{route}, M={depth}, XY={lateral}, occupied planes={planes}")
                    error = float(np.max(np.abs(values - expected)))
                    worst[route] = max(worst.get(route, 0.0), error)
                    counts[route] += 1
    return {"depths": [128, 256, 257, 512, 1024], "field_shape_zyx": [NZ, NY, NX],
            "native_transverse_bin_yx": [KY, KX], "comparisons": counts,
            "max_absolute_complex_error": worst}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-zp01", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check(args.with_zp01), indent=2))
