"""Exercise the proposed public sampler with a NumPy direct-sum oracle."""
import argparse
from pathlib import Path
import sys

import numpy as np


def direct(field, target, modes):
    offset = (np.array(target) - field.shape) // 2
    positions = np.indices(field.shape).reshape(3, -1).T + offset
    phases = -2j * np.pi * ((positions / target) @ modes.T)
    return field.ravel() @ np.exp(phases)


def literal_reference(field, target, modes):
    """Independent literal, centered zero extension and dense 3-D FFT."""
    grid = np.zeros(target, dtype=field.dtype)
    offset = (np.array(target) - field.shape) // 2
    grid[tuple(slice(int(i), int(i+n)) for i, n in zip(offset, field.shape))] = field
    return np.fft.fftn(grid)[tuple((modes % target).T)]


def check(cp=None, make_plan=None):
    rng = np.random.default_rng(20261123)
    cases = [
        ((3, 4, 5), (3, 4, 5)),       # native, mixed parity
        ((3, 4, 5), (8, 4, 5)),       # centered Z extension with odd gap
        ((3, 4, 5), (8, 9, 10)),      # XYZ; includes lateral Nyquist
        ((1, 5, 6), (1, 5, 6)),       # singleton Z
    ]
    calls = 0
    for shape, target in cases:
        real = rng.normal(size=shape).astype(np.float32)
        imag = rng.normal(size=shape).astype(np.float32)
        impulse = np.zeros(shape, np.float32)
        impulse[(0, 1, 2)] = 1
        z = int(target[0] > 1)
        modes = np.array([(0, 0, 0), (z, 1, -2), (-z, -1, 2),
                          (0, 0, 0), (0, -(target[1] // 2), -(target[2] // 2)),
                          (z, -1, 1), (z, 1, -2)], dtype=np.int64)
        if cp is not None:
            plan = make_plan(shape_zyx=shape, logical_shape_zyx=target,
                             modes_zyx=modes, dc="raw")
            out = cp.empty(len(modes), dtype=cp.complex64)
        amplitudes = []
        for field in (real, imag, impulse, np.ones(shape, np.float32)):
            ref = direct(field, target, modes)
            literal = literal_reference(field.astype(np.float64), target, modes)
            np.testing.assert_allclose(literal, ref, atol=5e-13, rtol=1e-13)
            if cp is not None:
                inp = cp.asarray(field)
                # A poisoned caller-owned buffer detects any unwritten endpoint.
                out.fill(complex(np.nan, np.nan))
                plan.capture_into(inp, out=out)
                got = cp.asnumpy(out)
                np.testing.assert_array_equal(cp.asnumpy(inp), field)
                # Per-row bound scaled by the sum of the input magnitudes.
                np.testing.assert_allclose(got, ref, rtol=0, atol=5e-6*np.abs(field).sum())
                np.testing.assert_array_equal(got[[0, 1]], got[[3, 6]])
            else:
                got = literal
            amplitudes.append(got)
            calls += 1
        source = real.astype(np.complex128) + 1j * imag
        np.testing.assert_allclose(amplitudes[0] + 1j*amplitudes[1],
                                   direct(source, target, modes),
                                   rtol=0, atol=1e-5*np.abs(source).sum())
    print(f"{len(cases)} geometries, {calls} captures, raw DC and complex reconstruction passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, help="Run the CuPy public API of this checkout")
    args = parser.parse_args()
    if args.candidate_root:
        checkout = args.candidate_root.resolve()
        sys.path.insert(0, str(checkout / "src"))
        import cupy
        import NRSS
        from NRSS.fourier import prepare_sampler
        if Path(NRSS.__file__).resolve() != checkout / "src/NRSS/__init__.py":
            raise RuntimeError("NRSS was imported from a different checkout")
        check(cupy, prepare_sampler)
    else:
        check()
