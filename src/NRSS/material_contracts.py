from __future__ import annotations

from enum import Enum


class SFieldMode(Enum):
    """Special contract values accepted for Material.S."""

    ISOTROPIC = False


def is_isotropic_s_field_mode(value) -> bool:
    return value is SFieldMode.ISOTROPIC


__all__ = ["SFieldMode", "is_isotropic_s_field_mode"]
