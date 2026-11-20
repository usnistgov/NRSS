"""Isotropic finite-mode fixtures used by the native FFT registration tests."""
import numpy as np
import pytest

from NRSS import SFieldMode
from NRSS.morphology import Material, Morphology

HC_EV_NM = 1239.84197
DELTA = 1 / 32
CHI = (1 - DELTA) ** 2 - 1


def detector_run(fraction, energies, nrss_path):
    cp = pytest.importorskip("cupy")
    fraction = cp.ascontiguousarray(cp.asarray(fraction, dtype=cp.float32))
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


def grating_measurements(nrss_path):
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
        panel = detector_run(field, energies, nrss_path)
        iy,ix = ny//2+signed_order, nx//2
        opposite = ny//2-signed_order
        for depth, energy, expected in zip(depths,energies,(1.,.5)):
            frame = panel.sel(energy=energy).transpose("qy","qx")
            scale = intensity(energy,field.size/4)
            measurements.append(dict(order=signed_order, depth=depth, energy=energy,
                                     qx=float(frame.qx[ix]), qy=float(frame.qy[iy]),
                                     expected_qy=qy, expected=expected,
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
