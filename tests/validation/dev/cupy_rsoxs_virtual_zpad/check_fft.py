"""Small NumPy oracle for exact implicit axial padding. No NRSS or GPU required."""
import json

import numpy as np


def selected(field, target_z, points, offset, method):
    """Unnormalized FFT values, queried as native unsigned (kz, ky, kx) bins."""
    nz, ny, nx = field.shape
    if target_z < nz or not 0 <= offset <= target_z - nz:
        raise ValueError("source must lie inside the target transform")
    kz, ky, kx = np.asarray(points, dtype=int).T.copy()
    reflect = np.zeros(kx.shape, dtype=bool)
    if np.isrealobj(field):
        g = np.fft.rfft2(field, axes=(-2, -1))
        reflect = kx > nx // 2
        kz[reflect] = -kz[reflect] % target_z
        ky[reflect] = -ky[reflect] % ny
        kx[reflect] = -kx[reflect] % nx
    else:
        g = np.fft.fft2(field, axes=(-2, -1))
    z = np.arange(nz)
    if method == "sum":
        phase = np.exp((-2j * np.pi / target_z) * z[:, None] * kz)
        result = np.sum(g[:, ky, kx] * phase, axis=0)
    elif method == "residues":
        p, remainder = divmod(target_z, nz)
        if remainder:
            raise ValueError("the residue identity requires M=p*N")
        result = np.empty(kz.size, dtype=complex)
        for residue in np.unique(kz % p):
            keep = (kz % p) == residue
            phase = np.exp((-2j * np.pi * residue / target_z) * z)
            transformed = np.fft.fft(g * phase[:, None, None], axis=0)
            result[keep] = transformed[kz[keep] // p, ky[keep], kx[keep]]
    else:
        raise ValueError(method)
    result *= np.exp((-2j * np.pi * offset / target_z) * kz)
    result[reflect] = result[reflect].conj()
    return result


def relative_error(values, reference):
    scale = np.linalg.norm(reference)
    difference = np.linalg.norm(values - reference)
    return float(difference / scale if scale else difference)


def check_off_axis_grating():
    """Compare native-bin interpolation and literal padding of both signed orders."""
    nz, ny = 128, 1024
    z, y = np.ogrid[:nz, :ny]
    amplitude_scale = nz * ny / 4  # qx = 0, so constant x and the transverse E factor cancel.
    records = []
    for order in (-64, 64):
        field = ((1 + np.cos(2*np.pi*(order*y/ny - z/nz))) / 2)[:, :, None]
        for target in (nz, 2*nz, 8*nz):
            offset = (target - nz)//2
            padded = np.zeros((target, ny))
            padded[offset:offset+nz] = field[:, :, 0]
            dense = np.fft.fft2(padded)
            for depth in (1., 1.5):
                t = -depth*target/nz  # signed physical bin in units of the target z spacing
                lower = int(np.floor(t))
                fraction = t - lower
                # Igor reverses the signed physical bins relative to the native FFT.
                points = [(-s % target, -v % ny, 0) for v in (order, -order)
                          for s in (lower, lower + 1)]
                indices = np.array(points)
                reference = dense[indices[:, 0], indices[:, 1]]
                for method in ("sum", "residues"):
                    got = selected(field, target, points, offset, method)
                    np.testing.assert_allclose(got/amplitude_scale, reference/amplitude_scale,
                                               atol=2e-13, rtol=2e-12)
                endpoint_power = abs(reference.reshape(2, 2) / amplitude_scale)**2
                observed = endpoint_power @ np.array([1 - fraction, fraction])
                if target == nz:
                    expected = [1. if depth == 1. else .5, 0.]
                else:
                    detuning = np.array([depth - 1., depth + 1.])
                    expected = (np.sinc(detuning) / np.sinc(detuning/nz))**2
                np.testing.assert_allclose(observed, expected, atol=2e-13, rtol=2e-12)
                records.append({"order": order, "target_z": target, "native_depth": depth,
                                "bright": float(observed[0]), "opposite": float(observed[1])})
    return records


def check():
    rng = np.random.default_rng(1924)
    errors = {"sum": [], "residues": [], "dc": []}
    count = 0
    for nz, ny, nx, target in [(5, 7, 9, 5), (7, 6, 8, 56), (5, 8, 9, 20),
                               (3, 7, 8, 11), (1, 5, 6, 8), (1, 5, 6, 1)]:
        # Small grids allow checking every coefficient, including both Hermitian halves.
        points = np.indices((target, ny, nx)).reshape(3, -1).T
        neighbors = [(0, 0, 1), (0, 1, 0), (0, 0, nx - 1), (0, ny - 1, 0)]
        if target > 1:
            neighbors += [(1, 0, 0), (target - 1, 0, 0)]
        for real in (True, False):
            raw = rng.standard_normal((nz, ny, nx))
            if not real:
                raw = raw + 1j * rng.standard_normal(raw.shape)
            for offset in sorted({0, (target - nz) // 2}):
                for window in (False, True):
                    padded = np.zeros((target, ny, nx), dtype=raw.dtype)
                    padded[offset:offset + nz] = raw
                    source = raw
                    if window:
                        xy_window = np.hanning(ny)[:, None] * np.hanning(nx)
                        wz = np.hanning(target)
                        padded *= wz[:, None, None] * xy_window
                        source = raw * (wz[offset:offset + nz, None, None] * xy_window)
                    dense = np.fft.fftn(padded)
                    exact = dense.ravel()
                    for method in ("sum", "residues"):
                        if method == "residues" and target % nz:
                            continue
                        result = selected(source, target, points, offset, method)
                        errors[method].append(relative_error(result, exact))
                        dc = selected(source, target, neighbors, offset, method).mean()
                        ref_dc = dense[tuple(np.array(neighbors).T)].mean()
                        errors["dc"].append(float(abs(dc-ref_dc) / max(np.max(abs(exact)), 1e-30)))
                        count += 1
    worst = {name: max(series) for name, series in errors.items()}
    if not all(np.all(np.isfinite(series)) for series in errors.values()) or max(worst.values()) > 2e-12:
        raise AssertionError(worst)
    print(json.dumps({"comparisons": count, "worst": worst,
                      "off_axis_grating": check_off_axis_grating()}, indent=2))


if __name__ == "__main__":
    check()
