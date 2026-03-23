"""Reusable orientational-contrast helpers for NRSS validation tests.

Reference:
    P. J. Dudenas, L. Q. Flagg, K. Goetz, P. Shapturenka, J. A. Fagan,
    E. Gann, and D. M. DeLongchamp, "How to RSoXS," J. Chem. Phys. 163,
    061501 (2025), https://doi.org/10.1063/5.0267709.

This module follows the tensor path discussed around Eq. (13), Eq. (15),
and Eq. (16). The helper keeps the intermediate tensors, the induced
polarization vector, and the far-field projection explicit so the expected
contrast calculation stays inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


HOW_TO_RSOXS_CITATION = (
    "P. J. Dudenas, L. Q. Flagg, K. Goetz, P. Shapturenka, "
    "J. A. Fagan, E. Gann, and D. M. DeLongchamp, "
    "\"How to RSoXS,\" J. Chem. Phys. 163, 061501 (2025), "
    "https://doi.org/10.1063/5.0267709."
)


@dataclass(frozen=True)
class UniaxialOpticalState:
    """Single-material uniaxial optical state in the measurement frame.

    The para channel is the extraordinary axis of the local uniaxial tensor.
    The first Euler angle is unused for a uniaxial tensor, so callers only
    supply the second and third ZYZ angles used by NRSS/CyRSoXS.
    """

    delta_para: float
    beta_para: float
    delta_perp: float
    beta_perp: float
    theta: float = 0.0
    psi: float = 0.0
    S: float = 0.0

    @classmethod
    def vacuum(cls) -> "UniaxialOpticalState":
        return cls(
            delta_para=0.0,
            beta_para=0.0,
            delta_perp=0.0,
            beta_perp=0.0,
            theta=0.0,
            psi=0.0,
            S=0.0,
        )


@dataclass(frozen=True)
class UniaxialMaterialResponse:
    """Inspectable intermediate values for a single uniaxial material."""

    state: UniaxialOpticalState
    incident_polarization_xyz: np.ndarray
    far_field_direction_xyz: np.ndarray
    refractive_index_local_xyz: np.ndarray
    rotation_matrix_xyz: np.ndarray
    refractive_index_rotated_xyz: np.ndarray
    isotropic_index: complex
    refractive_index_effective_xyz: np.ndarray
    effective_scalar_index: complex
    effective_scalar_delta: float
    effective_scalar_beta: float
    induced_polarization_xyz: np.ndarray
    far_field_projector_xyz: np.ndarray
    far_field_response_xyz: np.ndarray

    @property
    def far_field_contrast_sq(self) -> float:
        return float(np.vdot(self.far_field_response_xyz, self.far_field_response_xyz).real)


@dataclass(frozen=True)
class TwoMaterialFarFieldContrast:
    """Eq. 15/16-style contrast prediction between two materials."""

    material_1: UniaxialMaterialResponse
    material_2: UniaxialMaterialResponse
    induced_polarization_difference_xyz: np.ndarray
    far_field_difference_xyz: np.ndarray
    far_field_contrast_sq: float


def _as_unit_vector(vec: tuple[float, float, float] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a length-3 vector, got shape {arr.shape!r}.")
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} must have finite nonzero norm, got {arr!r}.")
    return arr / norm


def _validate_state(state: UniaxialOpticalState) -> None:
    for field_name in (
        "delta_para",
        "beta_para",
        "delta_perp",
        "beta_perp",
        "theta",
        "psi",
        "S",
    ):
        value = getattr(state, field_name)
        if not np.isfinite(value):
            raise ValueError(f"{field_name} must be finite, got {value!r}.")
    if not (0.0 <= state.S <= 1.0):
        raise ValueError(f"S must lie on [0, 1], got {state.S!r}.")


def predict_uniaxial_material_response(
    state: UniaxialOpticalState,
    *,
    incident_polarization_xyz: tuple[float, float, float] | np.ndarray = (1.0, 0.0, 0.0),
    far_field_direction_xyz: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 1.0),
) -> UniaxialMaterialResponse:
    """Predict a uniaxial material response from para/perp optical constants.

    This follows the tensor path described in the How to RSoXS tutorial:

    - Eq. (13): rotate the local diagonal refractive-index tensor into the
      measurement frame with ``N_rot = R @ N_local @ R.T``.
    - Eq. (15): contrast is proportional to the magnitude squared of the
      induced-polarization difference vector.
    - Eq. (16): the induced-polarization vector is built from
      ``(N @ N - I) @ E``.

    The index tensor is first built in a local XYZ frame as
    ``diag(n_perp, n_perp, n_para)``, where the extraordinary direction is
    the local +Z axis. The NRSS/CyRSoXS Euler convention is extrinsic ZYZ,
    and for a uniaxial tensor the first Euler angle is unused, so the SciPy
    call is ``Rotation.from_euler("zyz", [0, theta, psi])``.

    ``S`` is treated as the aligned fraction. The remaining ``(1-S)`` part is
    mixed with the isotropic average index ``(n_para + 2 * n_perp) / 3``.

    For consistency with current NRSS terminology, the projector
    ``I - k_hat k_hat.T`` is called the "far field" projection here. It removes
    the component of the induced polarization parallel to the scattered
    propagation direction. The Eq. (15) photon-energy prefactor is not included
    because current validation use is monochromatic and ratio-based.
    """

    _validate_state(state)
    incident_polarization = _as_unit_vector(
        incident_polarization_xyz,
        name="incident_polarization_xyz",
    )
    far_field_direction = _as_unit_vector(
        far_field_direction_xyz,
        name="far_field_direction_xyz",
    )

    n_para = complex(1.0 - state.delta_para, state.beta_para)
    n_perp = complex(1.0 - state.delta_perp, state.beta_perp)
    refractive_index_local = np.diag([n_perp, n_perp, n_para]).astype(np.complex128)

    # Lowercase 'zyz' is SciPy's extrinsic ZYZ convention, which matches the
    # measurement-frame rotation discussion in How to RSoXS Eq. (13).
    rotation_matrix = Rotation.from_euler(
        "zyz",
        [0.0, state.theta, state.psi],
    ).as_matrix()
    refractive_index_rotated = (
        rotation_matrix @ refractive_index_local @ rotation_matrix.T
    ).astype(np.complex128)

    isotropic_index = (n_para + 2.0 * n_perp) / 3.0
    refractive_index_effective = (
        state.S * refractive_index_rotated
        + (1.0 - state.S) * isotropic_index * np.eye(3, dtype=np.complex128)
    )

    effective_scalar_index = complex(
        incident_polarization @ refractive_index_effective @ incident_polarization
    )
    effective_scalar_delta = float(1.0 - np.real(effective_scalar_index))
    effective_scalar_beta = float(np.imag(effective_scalar_index))

    identity = np.eye(3, dtype=np.complex128)
    induced_polarization = (
        refractive_index_effective @ refractive_index_effective - identity
    ) @ incident_polarization

    # In the far field, the measured field must be transverse to the scattered
    # propagation direction. This projector is the measurement-geometry step.
    far_field_projector = np.eye(3, dtype=np.complex128) - np.outer(
        far_field_direction,
        far_field_direction,
    )
    far_field_response = far_field_projector @ induced_polarization

    return UniaxialMaterialResponse(
        state=state,
        incident_polarization_xyz=incident_polarization,
        far_field_direction_xyz=far_field_direction,
        refractive_index_local_xyz=refractive_index_local,
        rotation_matrix_xyz=rotation_matrix,
        refractive_index_rotated_xyz=refractive_index_rotated,
        isotropic_index=isotropic_index,
        refractive_index_effective_xyz=refractive_index_effective,
        effective_scalar_index=effective_scalar_index,
        effective_scalar_delta=effective_scalar_delta,
        effective_scalar_beta=effective_scalar_beta,
        induced_polarization_xyz=induced_polarization,
        far_field_projector_xyz=far_field_projector,
        far_field_response_xyz=far_field_response,
    )


def predict_two_material_far_field_contrast(
    material_1: UniaxialOpticalState,
    material_2: UniaxialOpticalState,
    *,
    incident_polarization_xyz: tuple[float, float, float] | np.ndarray = (1.0, 0.0, 0.0),
    far_field_direction_xyz: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 1.0),
) -> TwoMaterialFarFieldContrast:
    """Predict Eq. 15/16 contrast between two uniaxial materials."""

    response_1 = predict_uniaxial_material_response(
        material_1,
        incident_polarization_xyz=incident_polarization_xyz,
        far_field_direction_xyz=far_field_direction_xyz,
    )
    response_2 = predict_uniaxial_material_response(
        material_2,
        incident_polarization_xyz=incident_polarization_xyz,
        far_field_direction_xyz=far_field_direction_xyz,
    )

    induced_polarization_difference = (
        response_1.induced_polarization_xyz - response_2.induced_polarization_xyz
    )
    far_field_difference = (
        response_1.far_field_projector_xyz @ induced_polarization_difference
    )
    far_field_contrast_sq = float(np.vdot(far_field_difference, far_field_difference).real)

    return TwoMaterialFarFieldContrast(
        material_1=response_1,
        material_2=response_2,
        induced_polarization_difference_xyz=induced_polarization_difference,
        far_field_difference_xyz=far_field_difference,
        far_field_contrast_sq=far_field_contrast_sq,
    )


def predict_uniaxial_vacuum_far_field_contrast(
    material: UniaxialOpticalState,
    *,
    incident_polarization_xyz: tuple[float, float, float] | np.ndarray = (1.0, 0.0, 0.0),
    far_field_direction_xyz: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 1.0),
) -> TwoMaterialFarFieldContrast:
    """Convenience wrapper for a material contrasted against vacuum."""

    return predict_two_material_far_field_contrast(
        material,
        UniaxialOpticalState.vacuum(),
        incident_polarization_xyz=incident_polarization_xyz,
        far_field_direction_xyz=far_field_direction_xyz,
    )


__all__ = [
    "HOW_TO_RSOXS_CITATION",
    "TwoMaterialFarFieldContrast",
    "UniaxialMaterialResponse",
    "UniaxialOpticalState",
    "predict_two_material_far_field_contrast",
    "predict_uniaxial_material_response",
    "predict_uniaxial_vacuum_far_field_contrast",
]
