"""CPU axial-sampling study for fft_bins_v1 on finite sphere lattices, in nm and eV.

Run from an NRSS development checkout. The scalar probe uses the repo's lattice
rasterizer and a single x-polarized input. It does not call the simulation backend.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def make_field(bragg, kind, xy_size):
    shape = (128, xy_size, xy_size)
    if kind == "hcp":
        lattice, basis = bragg.lattice_vectors_3d_nm("hcp", 45.0, 27.0, 9.0, 5.0)
    else:
        lattice, _ = bragg.lattice_vectors_3d_nm("simple_cubic", 30.0, 27.0, 9.0, 5.0)
        basis = np.array([[0., 0., 0.], [.5, .5, .5]])
    field, centers = bragg._sphere_lattice_vfrac_3d(
        shape=shape, phys_size_nm=1., primitive_vectors_nm=lattice, basis_frac=basis,
        diameter_nm=11., superresolution=2)
    return field, len(centers)


def axial_q(k, qx, qy):
    q2 = qx**2 + qy**2
    return -q2 / (k + np.sqrt(np.maximum(k*k - q2, 0.)))


def weight(k, qx, qy, qz):
    # |(k² I - kout koutᵀ) e_x|² / k⁴. Common scalar factors cancel in the error.
    return ((k*k - qx*qx)**2 + (qx*qy)**2 + (qx*(k + qz))**2) / k**4


def exact_xyz(field, qx, qy, qz):
    nz, ny, nx = field.shape
    x = np.exp(1j * qx * np.arange(nx))
    y = np.exp(1j * qy * np.arange(ny))
    z = np.exp(1j * qz * np.arange(nz))
    xy_phase = y[:, None] * x[None, :]
    planes = [np.einsum("yx,yx->", plane, xy_phase) for plane in field]
    return np.dot(z, planes)


def energy_band(field, g, energy, low, high):
    nz, ny, nx = field.shape
    k = 2*np.pi * energy / 1239.84197
    # fft_bins_v1 for unit voxel spacing. Igor maps physical q to -q in the native FFT.
    qx_axis = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(nx))
    qy_axis = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(ny))
    radius = np.hypot(qy_axis[:, None], qx_axis[None, :])
    # All requested N and M are even. The negative axial edge has magnitude pi.
    # Match the registered forward support and the existing transverse border.
    depth_limit = min(k, np.pi)
    valid = radius**2 <= depth_limit * (2*k - depth_limit)
    valid[-1, :] = False
    valid[:, -1] = False
    iy, ix = np.nonzero((radius >= low) & (radius < high) & valid)
    qx, qy = qx_axis[ix], qy_axis[iy]
    if not ix.size:
        raise ValueError("The chosen comparison band has no valid pixels")
    depths = [nz, 2*nz, 4*nz, 8*nz]
    sq_error = np.zeros(len(depths))
    ref_squared = 0.
    n = np.arange(nz)[:, None]
    for start in range(0, ix.size, 192):
        stop = min(start + 192, ix.size)
        cut = slice(start, stop)
        gx, gy = qx[cut], qy[cut]
        gz = axial_q(k, gx, gy)
        values = g[:, (ny//2 - iy[cut]) % ny, (nx//2 - ix[cut]) % nx]
        amp = np.sum(values * np.exp(1j * n * gz), axis=0)
        if start == 0:
            xyz = exact_xyz(field, gx[0], gy[0], gz[0])
            np.testing.assert_allclose(amp[0], xyz, atol=1e-9, rtol=1e-10)
        ref = abs(amp)**2 * weight(k, gx, gy, gz)
        ref_squared += float(np.dot(ref, ref))
        cols = np.arange(ref.size)
        for j, depth in enumerate(depths):
            # Dense padded z FFT on a small group of columns, used only by this oracle.
            spectrum = np.fft.fft(values, n=depth, axis=0)
            step = 2*np.pi/depth
            fractional_bin = gz/step
            lower = np.floor(fractional_bin).astype(int)
            upper = np.minimum(lower + 1, depth - 1 - depth//2)
            fraction = fractional_bin - lower
            lo = abs(spectrum[-lower % depth, cols])**2 * weight(k, gx, gy, lower*step)
            hi = abs(spectrum[-upper % depth, cols])**2 * weight(k, gx, gy, upper*step)
            diff = (1-fraction)*lo + fraction*hi - ref
            sq_error[j] += np.dot(diff, diff)
    return {"energy_eV": energy, "q_perp_band_nm_inv": [low, high], "pixels": int(ix.size),
            "target_z": depths, "rel_l2_intensity": np.sqrt(sq_error/ref_squared).tolist()}


def run(bragg, xy_size=192):
    results = {}
    for kind in ("hcp", "bcc"):
        field, n_centers = make_field(bragg, kind, xy_size)
        # Limit scratch to one plane when repeating the study at 1024².
        g = np.empty(field.shape, dtype=np.complex128)
        for index, plane in enumerate(field):
            g[index] = np.fft.fft2(plane.astype(np.float64))
        results[kind] = {"shape_zyx": list(field.shape), "sphere_count": n_centers, "bands": [
            energy_band(field, g, energy, low, high)
            for energy, low, high in ((285., .95, 1.25), (570., .95, 1.25), (570., 1.9, 2.4))]}
        del g, field
    return {"detector_grid_convention": "fft_bins_v1", "arithmetic": "float64/complex128",
            "reference": "finite voxel sum at continuous Ewald qz", "lattices": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--xy-size", type=int, choices=(192, 1024), default=192)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.repo.resolve()
    sys.path[:0] = [str(root / "tests/validation/lib"), str(root / "src")]
    import bragg
    report = json.dumps(run(bragg, args.xy_size), indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n")
    else:
        print(report)
