from __future__ import annotations

from enum import Enum


class SFieldMode(Enum):
    """Special contract values accepted for ``Material.S``.

    ``SFieldMode.ISOTROPIC`` declares that the material is explicitly isotropic,
    so orientation fields are not needed. When this contract is used correctly,
    NRSS can avoid carrying full ``S``, ``theta``, and ``psi`` arrays for that
    material through the backend, which can dramatically reduce memory footprint
    and improve runtime.
    """

    ISOTROPIC = False


def is_isotropic_s_field_mode(value) -> bool:
    return value is SFieldMode.ISOTROPIC


__all__ = ["SFieldMode", "is_isotropic_s_field_mode"]
