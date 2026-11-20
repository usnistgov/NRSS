"""Literal vacuum controls for the opt-in virtual z padding path."""

import numpy as np

from NRSS import SFieldMode
from NRSS.morphology import Material


def explicit_vacuum_materials(materials, transform_z, xp=np):
    first = next(iter(materials.values()))
    occupied_z, ny, nx = first.Vfrac.shape
    if transform_z < occupied_z:
        raise ValueError("The vacuum box must contain all occupied planes.")
    if transform_z == occupied_z:
        return materials

    vacuum = next(
        (key for key, material in materials.items() if not np.any(list(material.opt_constants.values()))),
        None,
    )
    offset = (transform_z - occupied_z) // 2
    occupied = slice(offset, offset + occupied_z)
    shape = (transform_z, ny, nx)

    def embed(values, outside=0):
        if values is None or isinstance(values, SFieldMode):
            return values
        padded = xp.full(shape, outside, dtype=values.dtype)
        padded[occupied] = values
        return padded

    padded = {
        key: Material(
            key, Vfrac=embed(material.Vfrac, int(key == vacuum)),
            S=embed(material.S), theta=embed(material.theta), psi=embed(material.psi),
            energies=material.energies, opt_constants=material.opt_constants, name=material.name,
        )
        for key, material in materials.items()
    }
    if vacuum is None:
        key = max(materials) + 1
        fraction = xp.ones(shape, dtype=first.Vfrac.dtype)
        fraction[occupied] = 0
        padded[key] = Material(
            key, Vfrac=fraction, S=SFieldMode.ISOTROPIC, energies=first.energies,
            opt_constants={energy: [0.] * 4 for energy in first.energies}, name="vacuum",
        )
    return padded


def xy_hann_reference(morphology):
    """Weight literal-box scattering fields in XY with the backend XYZ window disabled."""
    if morphology.WindowingType != 0:
        raise ValueError("Disable the normal XYZ window for the explicit XY Hann reference.")

    def window(morph, cp, shape_override=None, *, separable=False, **_options):
        _, ny, nx = shape_override or morph.NumZYX
        wy = cp.asarray(np.hanning(ny), dtype=cp.float32)[None, :, None]
        wx = cp.asarray(np.hanning(nx), dtype=cp.float32)[None, None, :]
        return (wy, wx) if separable else wy * wx

    morphology._backend_runtime._window_tensor = window
