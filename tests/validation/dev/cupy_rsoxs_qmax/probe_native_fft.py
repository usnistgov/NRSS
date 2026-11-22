"""Check the native candidate that limits Y and Z FFT columns after the X RFFT."""
import json

import numpy as np


def partial_fft(real_field, modes, width):
    nz, ny, nx = real_field.shape
    kz, ky, kx = (modes % real_field.shape).T.copy()
    reflect = kx > nx // 2
    for index, length in ((kz, nz), (ky, ny), (kx, nx)):
        index[reflect] = -index[reflect] % length
    x_fft = np.fft.rfft(real_field, axis=2)
    result = np.empty(len(kx), dtype=np.complex128)
    columns = np.unique(kx)
    for start in range(0, len(columns), width):
        block = columns[start:start + width]
        ids = np.flatnonzero(np.isin(kx, block))
        y_fft = np.fft.fft(x_fft[:, :, block], axis=1)
        local_x = np.searchsorted(block, kx[ids])
        pairs = ky[ids] * len(block) + local_x
        unique, inverse = np.unique(pairs, return_inverse=True)
        selected = y_fft[:, unique // len(block), unique % len(block)]
        z_fft = np.fft.fft(selected, axis=0)
        result[ids] = z_fft[kz[ids], inverse]
    result[reflect] = result[reflect].conj()
    return result


def main():
    rng = np.random.default_rng(20261122)
    count = 0
    largest = 0.
    differing_values = 0
    total_values = 0
    for shape in ((8, 12, 14), (7, 13, 11), (8, 11, 13), (1, 10, 9)):
        shape = np.array(shape)
        center = shape // 2
        steps = np.eye(3, dtype=int)
        if shape[0] == 1:
            steps = steps[1:]
        neighbors = np.r_[steps, -steps]
        sample = rng.integers(-center, center + 1, size=(60, 3))
        modes = np.vstack(([0, 0, 0], neighbors, center, sample, -sample)) % shape
        dc = np.all(modes == 0, axis=1)
        fields = rng.normal(size=(3, *shape))
        weights = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
        for hann in (False, True):
            window = 1.0
            if hann:
                wz, wy, wx = (np.hanning(n) for n in shape)
                window = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]
            windowed = fields * window
            dense = np.fft.fftn(windowed, axes=(1, 2, 3))
            ref = dense[(slice(None), *modes.T)]
            for width in (1, 3, 9):
                measured = np.stack([partial_fft(field, modes, width) for field in windowed])
                for repair in (False, True):
                    expected = ref.copy()
                    observed = measured.copy()
                    if repair:
                        neighbor_rows = slice(1, 1 + len(neighbors))
                        expected[:, dc] = ref[:, neighbor_rows].real.mean(axis=1, keepdims=True)
                        observed[:, dc] = measured[:, neighbor_rows].real.mean(axis=1, keepdims=True)
                    for a, b in ((observed, expected), (weights @ observed, weights @ expected)):
                        raw_a = np.ascontiguousarray(a).view(np.uint64).reshape(-1, 2)
                        raw_b = np.ascontiguousarray(b).view(np.uint64).reshape(-1, 2)
                        differing_values += int(np.count_nonzero(np.any(raw_a != raw_b, axis=1)))
                        total_values += a.size
                        fraction = np.max(np.abs(a - b)) / np.max(np.abs(b))
                        assert fraction < 3e-13, (shape, width, hann, fraction)
                        largest = max(largest, float(fraction))
                        count += 1
    print(json.dumps({"comparisons": count, "complex_values": total_values,
                      "complex_values_with_different_bits": differing_values, "worst_error_over_peak": largest,
                      "precision": "NumPy complex128", "timed_gpu": False}, indent=2))


if __name__ == "__main__":
    main()
