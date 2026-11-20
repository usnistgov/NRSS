"""Isotropic finite-mode fixtures used by the native FFT registration tests."""
import numpy as np
import pytest

from NRSS import SFieldMode
from NRSS.morphology import Material, Morphology

HC_EV_NM = 1239.84197
DELTA = 1 / 32
CHI = (1 - DELTA) ** 2 - 1


def detector_run(fraction, energies, nrss_path, *, explicit_z=None):
    cp = pytest.importorskip("cupy")
    fraction = cp.ascontiguousarray(cp.asarray(fraction, dtype=cp.float32))
    occupied_z = fraction.shape[0]
    if explicit_z is not None and explicit_z != occupied_z:
        if explicit_z < occupied_z:
            raise ValueError("The explicit vacuum box must contain all occupied planes.")
        offset = (explicit_z - occupied_z) // 2
        padded = cp.zeros((explicit_z, *fraction.shape[1:]), dtype=fraction.dtype)
        padded[offset:offset + occupied_z] = fraction
        fraction = padded
    materials = {
        i: Material(i, Vfrac=field, S=SFieldMode.ISOTROPIC, energies=energies,
                    opt_constants={e: [delta, 0., delta, 0.] for e in energies})
        for i, field, delta in ((1, fraction, DELTA), (2, 1-fraction, 0.))
    }
    config = dict(Morphology.config_default, Energies=energies, EAngleRotation=[0.,0.,0.],
                  RotMask=0, WindowingType=0)
    morph = Morphology(2, materials=materials, PhysSize=1., config=config,
                       backend=nrss_path.backend, backend_options=dict(nrss_path.backend_options),
                       resident_mode="device", input_policy="strict", ownership_policy="borrow")
    try:
        return morph.run(stdout=False, stderr=False, return_xarray=True).copy(deep=True)
    finally:
        morph.release_runtime()


def intensity(energy, amplitude):
    # E along x and selected qx=0 give an exactly transverse polarization.
    k = 2 * np.pi * energy / HC_EV_NM
    return k**4 * (CHI * amplitude / (4*np.pi))**2


def grating_measurements(nrss_path, *, explicit_z=None):
    nz, ny, nx = 128, 1024, 9
    z,y = np.ogrid[:nz,:ny]
    measurements = []
    for signed_order in (-64,64):
        fraction = (1 + np.cos(2*np.pi*(signed_order*y/ny - z/nz))) / 2
        field = np.broadcast_to(fraction[:,:,None],(nz,ny,nx))
        qy = 2*np.pi*signed_order/ny
        depths = np.array([1., 1.5])
        qz = -depths*2*np.pi/nz
        energies = (-(qy**2+qz**2)/(2*qz) * HC_EV_NM/(2*np.pi)).tolist()
        panel = detector_run(field, energies, nrss_path, explicit_z=explicit_z)
        expected_qx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx))
        expected_qy = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny))
        np.testing.assert_array_equal(panel.energy.values, energies)
        np.testing.assert_allclose(panel.qx.values, expected_qx, rtol=0, atol=1e-8)
        np.testing.assert_allclose(panel.qy.values, expected_qy, rtol=0, atol=1e-8)
        assert panel.attrs["detector_grid_convention"] == "fft_bins_v1"
        assert panel.attrs["z_dim"] == (nz if explicit_z is None else explicit_z)
        iy,ix = ny//2+signed_order, nx//2
        opposite = ny//2-signed_order
        for depth, energy, expected in zip(depths,energies,(1.,.5)):
            frame = panel.sel(energy=energy).transpose("qy","qx")
            scale = intensity(energy,field.size/4)
            k = 2 * np.pi * energy / HC_EV_NM
            # Here k < π at unit voxel pitch, so the forward Ewald support
            # lies within the negative axial FFT edge at all tested depths.
            assert k < np.pi
            valid = expected_qy[:, None] ** 2 + expected_qx[None, :] ** 2 <= k ** 2
            valid[-1, :] = False
            valid[:, -1] = False
            np.testing.assert_array_equal(np.isfinite(frame.values), valid)
            expected_dark = 0.
            if explicit_z is not None and explicit_z > nz:
                target_bin = depth * explicit_z / nz
                lower = np.floor(target_bin)
                sampled_depths = np.array([lower, lower + 1]) * nz / explicit_z
                detuning = sampled_depths - np.array([[1.], [-1.]])
                weights = np.array([1 - (target_bin - lower), target_bin - lower])
                expected, expected_dark = ((np.sinc(detuning) / np.sinc(detuning / nz)) ** 2) @ weights
            measurements.append(dict(order=signed_order, depth=depth, energy=energy,
                                     qx=float(frame.qx[ix]), qy=float(frame.qy[iy]),
                                     expected_qy=qy, expected=expected, expected_dark=expected_dark,
                                     bright=float(frame.values[iy,ix]/scale),
                                     dark=float(frame.values[opposite,ix]/scale)))
    return measurements


def plot_grating(measurements, path_id, directory):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4,3.9),layout="constrained")
    ax.plot([1.,1.5],[1.,.5],color="black",linestyle=":",label="Analytic bin interpolation")
    for order,marker in ((-64,"o"),(64,"x")):
        rows = [m for m in measurements if m["order"]==order]
        ax.scatter([m["depth"] for m in rows],[m["bright"] for m in rows],
                   marker=marker,s=68,label=f"Observed order {order:+d}")
    ax.set(xlabel="Axial Ewald depth (FFT bins)", ylabel="Intensity / exact on-bin intensity",
           xticks=[1.,1.5],xlim=(.9,1.6),ylim=(0,1.1),title="Off-axis Bragg registration")
    ax.legend(frameon=False)
    directory.mkdir(parents=True,exist_ok=True)
    fig.savefig(directory/f"{path_id}__bragg_off_axis_registration.png",dpi=160)
    plt.close(fig)
