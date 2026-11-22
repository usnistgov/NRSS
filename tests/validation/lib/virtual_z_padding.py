"""Physical finite boxes for checking virtual padding."""

import numpy as np

from NRSS import SFieldMode
from NRSS.morphology import Material


def explicit_vacuum_materials(materials, transform_z, xp=np):
    """Build the incumbent Z-only vacuum reference."""
    _, ny, nx = next(iter(materials.values())).Vfrac.shape
    return explicit_padded_materials(materials, (transform_z, ny, nx), xp=xp)


def explicit_padded_materials(materials, transform_zyx, *, fill_material=None, xp=np):
    """Center the supplied crop in a literal box with isotropic exterior fill."""
    first = next(iter(materials.values()))
    shape = tuple(transform_zyx)
    source_shape = first.Vfrac.shape
    if len(shape) != len(source_shape) or any(
        isinstance(target, (bool, np.bool_))
        or not isinstance(target, (int, np.integer))
        or target < length
        for target, length in zip(shape, source_shape)
    ):
        raise ValueError("The target ZYX box must contain the occupied crop.")
    if fill_material is not None and fill_material not in materials:
        raise ValueError(f"Fill material {fill_material!r} is not registered.")
    if shape == source_shape:
        return materials

    if fill_material is None:
        fill_material = next(
            (key for key, material in materials.items() if not np.any(list(material.opt_constants.values()))),
            None,
        )
    offsets = tuple((target - length) // 2 for target, length in zip(shape, source_shape))
    occupied = tuple(slice(offset, offset + length) for offset, length in zip(offsets, source_shape))

    def embed(values, outside=0):
        if values is None or isinstance(values, SFieldMode):
            return values
        padded = xp.full(shape, outside, dtype=values.dtype)
        padded[occupied] = values
        return padded

    padded = {
        key: Material(
            key, Vfrac=embed(material.Vfrac, int(key == fill_material)),
            S=embed(material.S), theta=embed(material.theta), psi=embed(material.psi),
            energies=material.energies, opt_constants=material.opt_constants, name=material.name,
        )
        for key, material in materials.items()
    }
    if fill_material is None:
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
