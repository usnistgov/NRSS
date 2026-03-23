from __future__ import annotations

import importlib

from .registry import BackendUnavailableError
from .runtime import BackendRuntime


def require_cyrsoxs_module():
    errors = []
    for name in ("CyRSoXS", "cyrsoxs"):
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - exercised only when unavailable
            errors.append(f"{name}: {exc.__class__.__name__}({exc})")

    raise BackendUnavailableError(
        "CyRSoXS backend is unavailable. "
        f"Import attempts failed: {'; '.join(errors)}"
    )


def cyrsoxs_input_mapping(cy_module):
    return {
        'CaseType': ['setCaseType', [cy_module.CaseType.Default, cy_module.CaseType.BeamDivergence, cy_module.CaseType.GrazingIncidence]],
        'MorphologyType': ['setMorphologyType', [cy_module.MorphologyType.EulerAngles, cy_module.MorphologyType.VectorMorphology]],
        'EwaldsInterpolation': ['interpolationType', [cy_module.InterpolationType.NearestNeighour, cy_module.InterpolationType.Linear]],
        'WindowingType': ['windowingType', [cy_module.FFTWindowing.NoPadding, cy_module.FFTWindowing.Hanning]],
        'RotMask': ['rotMask', [False, True]],
        'AlgorithmType': ['setAlgorithm', [0, 1]],
        'ReferenceFrame': ['referenceFrame', [0, 1]],
    }


class CyrsoxsBackendRuntime(BackendRuntime):
    """Runtime adapter for the legacy CyRSoXS pybind backend."""

    name = "cyrsoxs"

    def prepare(self, morphology) -> None:
        morphology.create_update_cy()

    def run(
        self,
        morphology,
        *,
        stdout: bool = True,
        stderr: bool = True,
        return_xarray: bool = True,
        print_vec_info: bool = False,
        validate: bool = False,
    ):
        if (
            validate
            or morphology.inputData is None
            or morphology.OpticalConstants is None
            or morphology.voxelData is None
        ):
            self.prepare(morphology)

        cy = require_cyrsoxs_module()
        if not morphology.scatteringPattern:
            morphology.scatteringPattern = cy.ScatteringPattern(morphology.inputData)

        with cy.ostream_redirect(stdout=stdout, stderr=stderr):
            cy.launch(
                VoxelData=morphology.voxelData,
                RefractiveIndexData=morphology.OpticalConstants,
                InputData=morphology.inputData,
                ScatteringPattern=morphology.scatteringPattern,
            )

        morphology._simulated = True
        morphology._lock_results()
        if return_xarray:
            return morphology.scattering_to_xarray(
                return_xarray=return_xarray,
                print_vec_info=print_vec_info,
            )

    def validate_all(self, morphology, *, quiet: bool = True) -> None:
        morphology.check_materials(quiet=quiet)
        if (
            morphology.inputData is None
            or morphology.OpticalConstants is None
            or morphology.voxelData is None
        ):
            self.prepare(morphology)

        input_check = morphology.inputData.validate()
        opt_const_check = morphology.OpticalConstants.validate()
        voxel_check = morphology.voxelData.validate()
        assert input_check, 'CyRSoXS object inputData validation has failed'
        assert opt_const_check, 'CyRSoXS object OpticalConstants validation has failed'
        assert voxel_check, 'CyRSoXS object voxelData validation has failed'
        if not quiet:
            print('All objects have been validated successfully. You can run your simulation')

    def release(self, morphology) -> None:
        morphology.scatteringPattern = None
        morphology.voxelData = None
        morphology.OpticalConstants = None
        morphology.inputData = None
