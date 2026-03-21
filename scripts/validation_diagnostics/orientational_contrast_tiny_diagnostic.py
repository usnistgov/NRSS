"""Development-only tiny orientational-contrast probe.

This preserves the small 64^3 check that was useful while building the
reusable tensor-based helper and the official 128^3 physics validation. It is
kept outside pytest collection on purpose: the stable regression now lives in
``tests/validation/test_sphere_orientational_contrast_scaling.py``.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import CyRSoXS as cy
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.validation.lib.orientational_contrast import (  # noqa: E402
    HOW_TO_RSOXS_CITATION,
    UniaxialOpticalState,
    predict_uniaxial_vacuum_far_field_contrast,
)


TINY_SHAPE = (64, 64, 64)
TINY_PHYS_SIZE_NM = 2.0
TINY_DIAMETER_NM = 32.0
TINY_ENERGY_EV = 285.0
Q_INTEGRATE_MIN = 0.06
Q_INTEGRATE_MAX = 1.0
TINY_RATIO_REL_ERR_MAX = 0.06
TINY_ZERO_RATIO_ABS_ERR_MAX = 5e-3

REFERENCE_STATE = UniaxialOpticalState(
    delta_para=1e-4,
    beta_para=0.0,
    delta_perp=0.0,
    beta_perp=0.0,
    theta=np.pi / 2.0,
    psi=0.0,
    S=1.0,
)


@lru_cache(maxsize=1)
def _has_visible_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0 and "GPU " in result.stdout


def _release_runtime_memory() -> None:
    gc.collect()
    try:
        import cupy as cp
    except ImportError:
        return

    for action in (
        lambda: cp.cuda.runtime.deviceSynchronize(),
        lambda: cp.get_default_memory_pool().free_all_blocks(),
        lambda: cp.get_default_pinned_memory_pool().free_all_blocks(),
    ):
        try:
            action()
        except Exception:
            pass


def _cyrsoxs_detector_axis(n: int, phys_size_nm: float) -> np.ndarray:
    if int(n) < 2:
        raise AssertionError(f"CyRSoXS detector axis needs at least 2 points, got n={n}.")
    start = -np.pi / float(phys_size_nm)
    step = (2.0 * np.pi / float(phys_size_nm)) / float(int(n) - 1)
    return start + np.arange(int(n), dtype=np.float64) * step


def _detector_annulus_intensity(detector_image: np.ndarray) -> float:
    img = np.asarray(detector_image, dtype=np.float64)
    if img.shape != (TINY_SHAPE[1], TINY_SHAPE[2]):
        raise AssertionError(f"Unexpected detector image shape {img.shape!r}.")

    qy = _cyrsoxs_detector_axis(TINY_SHAPE[1], TINY_PHYS_SIZE_NM)
    qx = _cyrsoxs_detector_axis(TINY_SHAPE[2], TINY_PHYS_SIZE_NM)
    qx_grid, qy_grid = np.meshgrid(qx, qy)
    q = np.sqrt(qx_grid * qx_grid + qy_grid * qy_grid)

    mask = np.logical_and.reduce(
        [
            q >= Q_INTEGRATE_MIN,
            q <= Q_INTEGRATE_MAX,
            np.isfinite(q),
            np.isfinite(img),
            img >= 0.0,
        ]
    )
    if int(np.count_nonzero(mask)) < 64:
        raise AssertionError("Insufficient detector pixels for tiny orientational-contrast annulus.")
    return float(np.sum(img[mask], dtype=np.float64))


def _tiny_sphere_and_vacuum_vfrac() -> tuple[np.ndarray, np.ndarray]:
    nz, ny, nx = TINY_SHAPE
    radius_vox = float(TINY_DIAMETER_NM) / (2.0 * TINY_PHYS_SIZE_NM)
    cz = (nz - 1) / 2.0
    cy0 = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0

    zz, yy, xx = np.indices(TINY_SHAPE, dtype=np.float32)
    dist2 = (zz - cz) ** 2 + (yy - cy0) ** 2 + (xx - cx) ** 2
    sphere = (dist2 <= np.float32(radius_vox * radius_vox)).astype(np.float32)
    vacuum = (1.0 - sphere).astype(np.float32)
    return sphere, vacuum


def _run_tiny_orientational_sphere(state: UniaxialOpticalState) -> float:
    sphere_vfrac, vacuum_vfrac = _tiny_sphere_and_vacuum_vfrac()
    zeros = np.zeros_like(sphere_vfrac, dtype=np.float32)

    input_data = cy.InputData(NumMaterial=2)
    input_data.setEnergies([TINY_ENERGY_EV])
    input_data.setERotationAngle(StartAngle=0.0, EndAngle=0.0, IncrementAngle=0.0)
    input_data.setPhysSize(TINY_PHYS_SIZE_NM)
    input_data.setDimensions(TINY_SHAPE, cy.MorphologyOrder.ZYX)
    input_data.setCaseType(cy.CaseType.Default)
    input_data.setMorphologyType(cy.MorphologyType.EulerAngles)
    input_data.setAlgorithm(AlgorithmID=0, MaxStreams=1)
    input_data.interpolationType = cy.InterpolationType.Linear
    input_data.windowingType = cy.FFTWindowing.NoPadding
    input_data.rotMask = True
    input_data.referenceFrame = 1
    if not input_data.validate():
        raise AssertionError("CyRSoXS InputData validation failed for tiny orientational sphere.")

    optical_constants = cy.RefractiveIndex(input_data)
    optical_constants.addData(
        OpticalConstants=[
            [state.delta_para, state.beta_para, state.delta_perp, state.beta_perp],
            [0.0, 0.0, 0.0, 0.0],
        ],
        Energy=TINY_ENERGY_EV,
    )
    if not optical_constants.validate():
        raise AssertionError("CyRSoXS optical-constants validation failed for tiny orientational sphere.")

    voxel_data = cy.VoxelData(InputData=input_data)
    voxel_data.addVoxelData(
        S=(state.S * sphere_vfrac).astype(np.float32),
        Theta=(state.theta * sphere_vfrac).astype(np.float32),
        Psi=(state.psi * sphere_vfrac).astype(np.float32),
        Vfrac=sphere_vfrac,
        MaterialID=1,
    )
    voxel_data.addVoxelData(
        S=zeros,
        Theta=zeros,
        Psi=zeros,
        Vfrac=vacuum_vfrac,
        MaterialID=2,
    )
    if not voxel_data.validate():
        raise AssertionError("CyRSoXS VoxelData validation failed for tiny orientational sphere.")

    scattering_pattern = cy.ScatteringPattern(InputData=input_data)
    with cy.ostream_redirect(stdout=False, stderr=False):
        cy.launch(
            VoxelData=voxel_data,
            RefractiveIndexData=optical_constants,
            InputData=input_data,
            ScatteringPattern=scattering_pattern,
        )

    detector_image = np.array(
        scattering_pattern.writeAllToNumpy(kID=0)[0],
        dtype=np.float64,
        copy=True,
    )
    del scattering_pattern, voxel_data, optical_constants, input_data
    del sphere_vfrac, vacuum_vfrac, zeros
    _release_runtime_memory()
    return _detector_annulus_intensity(detector_image)


def _run_closed_form_checks() -> None:
    if "How to RSoXS" not in HOW_TO_RSOXS_CITATION:
        raise AssertionError("Citation string is missing the How to RSoXS reference.")

    ref = predict_uniaxial_vacuum_far_field_contrast(REFERENCE_STATE)
    theta_pi4 = predict_uniaxial_vacuum_far_field_contrast(
        UniaxialOpticalState(1e-4, 0.0, 0.0, 0.0, np.pi / 4.0, 0.0, 1.0)
    )
    psi_pi4 = predict_uniaxial_vacuum_far_field_contrast(
        UniaxialOpticalState(1e-4, 0.0, 0.0, 0.0, np.pi / 2.0, np.pi / 4.0, 1.0)
    )
    s_half = predict_uniaxial_vacuum_far_field_contrast(
        UniaxialOpticalState(1e-4, 0.0, 0.0, 0.0, np.pi / 2.0, 0.0, 0.5)
    )

    ref_scalar_delta = ref.material_1.effective_scalar_delta
    if not np.isclose(ref_scalar_delta, 1e-4, atol=1e-12):
        raise AssertionError("Reference scalar delta check failed.")
    if not np.isclose(theta_pi4.material_1.effective_scalar_delta / ref_scalar_delta, 0.5, atol=1e-12):
        raise AssertionError("Theta pi/4 scalar closed form failed.")
    if not np.isclose(psi_pi4.material_1.effective_scalar_delta / ref_scalar_delta, 0.5, atol=1e-12):
        raise AssertionError("Psi pi/4 scalar closed form failed.")
    if not np.isclose(s_half.material_1.effective_scalar_delta / ref_scalar_delta, 2.0 / 3.0, atol=1e-12):
        raise AssertionError("S=0.5 scalar closed form failed.")
    if not np.isclose(theta_pi4.far_field_contrast_sq / ref.far_field_contrast_sq, 0.25, atol=1e-12):
        raise AssertionError("Theta pi/4 far-field closed form failed.")
    if not np.isclose(psi_pi4.far_field_contrast_sq / ref.far_field_contrast_sq, 0.5, atol=1e-12):
        raise AssertionError("Psi pi/4 far-field closed form failed.")
    if not np.isclose(s_half.far_field_contrast_sq / ref.far_field_contrast_sq, 4.0 / 9.0, rtol=1e-4, atol=1e-12):
        raise AssertionError("S=0.5 far-field closed form failed.")


def main() -> int:
    _run_closed_form_checks()
    if not _has_visible_gpu():
        print("No visible NVIDIA GPU found; closed-form helper checks passed, pybind probe skipped.")
        return 0

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    ref_pred = predict_uniaxial_vacuum_far_field_contrast(REFERENCE_STATE).far_field_contrast_sq
    ref_sim = _run_tiny_orientational_sphere(REFERENCE_STATE)
    if ref_pred <= 0.0 or ref_sim <= 0.0:
        raise AssertionError("Reference tiny orientational contrast must be positive.")

    cases = [
        ("theta_zero", UniaxialOpticalState(1e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
        ("theta_pi4", UniaxialOpticalState(1e-4, 0.0, 0.0, 0.0, np.pi / 4.0, 0.0, 1.0)),
        ("psi_pi4", UniaxialOpticalState(1e-4, 0.0, 0.0, 0.0, np.pi / 2.0, np.pi / 4.0, 1.0)),
        ("psi_pi2", UniaxialOpticalState(1e-4, 0.0, 0.0, 0.0, np.pi / 2.0, np.pi / 2.0, 1.0)),
        ("mixed_pi4_pi4", UniaxialOpticalState(1e-4, 0.0, 0.0, 0.0, np.pi / 4.0, np.pi / 4.0, 1.0)),
        ("mixed_pi3_pi6", UniaxialOpticalState(1e-4, 0.0, 0.0, 0.0, np.pi / 3.0, np.pi / 6.0, 1.0)),
        ("s_half", UniaxialOpticalState(1e-4, 0.0, 0.0, 0.0, np.pi / 2.0, 0.0, 0.5)),
        ("mixed_delta_beta", UniaxialOpticalState(1e-4, 1e-4, 0.0, 0.0, np.pi / 3.0, np.pi / 6.0, 1.0)),
    ]

    print(f"tiny reference intensity = {ref_sim:.12f}")
    print("label\tpred_ratio\tsim_ratio\trel_err")
    for label, state in cases:
        pred_ratio = (
            predict_uniaxial_vacuum_far_field_contrast(state).far_field_contrast_sq / ref_pred
        )
        sim_ratio = _run_tiny_orientational_sphere(state) / ref_sim
        if pred_ratio <= 1e-12:
            abs_err = abs(sim_ratio - pred_ratio)
            print(f"{label}\t{pred_ratio:.9f}\t{sim_ratio:.9f}\t{abs_err:.6f} (abs)")
            if abs_err >= TINY_ZERO_RATIO_ABS_ERR_MAX:
                raise AssertionError(
                    f"{label} zero-ratio mismatch too large: pred={pred_ratio:.6f} sim={sim_ratio:.6f}"
                )
            continue

        rel_err = abs(sim_ratio - pred_ratio) / pred_ratio
        print(f"{label}\t{pred_ratio:.9f}\t{sim_ratio:.9f}\t{rel_err:.6f}")
        if rel_err >= TINY_RATIO_REL_ERR_MAX:
            raise AssertionError(
                f"{label} relative error too large: pred={pred_ratio:.6f} sim={sim_ratio:.6f} rel={rel_err:.6f}"
            )

    print("Tiny orientational diagnostic passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
