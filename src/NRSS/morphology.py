import h5py
# import pathlib
import warnings
from .checkH5 import check_NumMat
from .material_contracts import SFieldMode, is_isotropic_s_field_mode
from .reader import read_material, read_config
from .writer import write_opts, write_hdf5
from .visualizer import morphology_visualizer
from .backends import (
    BackendUnavailableError,
    assess_array_for_backend,
    coerce_array_for_backend,
    get_namespace_module,
    get_backend_info,
    inspect_array,
    normalize_backend_options,
    normalize_resident_mode,
    resolve_backend_name,
    resolve_backend_array_contract,
    resolve_backend_runtime_contract,
    to_python_bool,
)
from .backends.cyrsoxs import (
    cyrsoxs_input_mapping as _cyrsoxs_input_mapping,
    require_cyrsoxs_module as _require_cyrsoxs_module,
)
from .backends.runtime import get_backend_runtime

import numpy as np
import xarray as xr
import sys
import os
import copy

from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


class _LegacyInputDataProxy:
    """Backend-neutral compatibility shim for legacy ``morph.inputData`` usage."""

    _ATTRIBUTE_TO_CONFIG = {
        "windowingType": "WindowingType",
        "interpolationType": "EwaldsInterpolation",
        "referenceFrame": "ReferenceFrame",
        "rotMask": "RotMask",
    }

    def __init__(self, morphology):
        object.__setattr__(self, "_morphology", morphology)

    def __bool__(self):
        # Keep internal ``if self.inputData`` checks behaving as if no pybind
        # object exists for non-CyRSoXS backends.
        return False

    def __repr__(self):
        return (
            f"<LegacyInputDataProxy backend={self._morphology.backend!r} "
            f"NumMaterial={self._morphology.numMaterial}>"
        )

    def __getattr__(self, name):
        if name in self._ATTRIBUTE_TO_CONFIG:
            return getattr(self._morphology, self._ATTRIBUTE_TO_CONFIG[name])
        if name == "NumMaterial":
            return self._morphology.numMaterial
        raise AttributeError(
            f"{type(self).__name__!s} does not provide attribute {name!r}. "
            "Only the legacy config-like InputData surface is emulated for "
            "non-CyRSoXS backends."
        )

    def __setattr__(self, name, value):
        if name in self._ATTRIBUTE_TO_CONFIG:
            setattr(self._morphology, self._ATTRIBUTE_TO_CONFIG[name], value)
            return
        raise AttributeError(
            f"{type(self).__name__!s} does not support setting attribute {name!r}. "
            "Only the legacy config-like InputData surface is emulated for "
            "non-CyRSoXS backends."
        )

    def setCaseType(self, value):
        self._morphology.CaseType = value

    def setMorphologyType(self, value):
        self._morphology.MorphologyType = value

    def setAlgorithm(self, AlgorithmID, MaxStreams=1):
        del MaxStreams
        self._morphology.AlgorithmType = AlgorithmID

    def setEnergies(self, energies):
        self._morphology.Energies = energies

    def setERotationAngle(self, *args, **kwargs):
        if args and kwargs:
            raise TypeError("setERotationAngle accepts either positional or keyword arguments, not both.")
        if kwargs:
            try:
                start_angle = kwargs["StartAngle"]
                end_angle = kwargs["EndAngle"]
                increment_angle = kwargs["IncrementAngle"]
            except KeyError as exc:
                raise TypeError(
                    "setERotationAngle requires StartAngle, EndAngle, and IncrementAngle."
                ) from exc
        elif len(args) == 3:
            start_angle, end_angle, increment_angle = args
        else:
            raise TypeError(
                "setERotationAngle requires exactly three values: "
                "StartAngle, EndAngle, IncrementAngle."
            )
        self._morphology.EAngleRotation = [start_angle, increment_angle, end_angle]

    def setPhysSize(self, value):
        self._morphology.PhysSize = value

    def setDimensions(self, dimensions, order=None):
        if order is not None and str(order) != "ZYX":
            raise ValueError("Only ZYX morphology ordering is supported.")
        self._morphology.NumZYX = tuple(int(v) for v in dimensions)

    def validate(self):
        try:
            self._morphology.validate_all(quiet=True)
        except Exception:
            return False
        return True

    def print(self):
        print(
            "Legacy InputData compatibility proxy\n"
            f"backend = {self._morphology.backend}\n"
            f"CaseType = {self._morphology.CaseType}\n"
            f"Energies = {self._morphology.Energies}\n"
            f"EAngleRotation = {self._morphology.EAngleRotation}\n"
            f"MorphologyType = {self._morphology.MorphologyType}\n"
            f"AlgorithmType = {self._morphology.AlgorithmType}\n"
            f"WindowingType = {self._morphology.WindowingType}\n"
            f"RotMask = {self._morphology.RotMask}\n"
            f"ReferenceFrame = {self._morphology.ReferenceFrame}\n"
            f"EwaldsInterpolation = {self._morphology.EwaldsInterpolation}\n"
            f"PhysSize = {self._morphology.PhysSize}\n"
            f"NumZYX = {self._morphology.NumZYX}"
        )


def wraps(wrapper: Callable[P, T]) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    A decorator to preserve the original function's docstring.

    Args:
        wrapper: The wrapper function.

    Returns:
        A decorator that preserves the original function's docstring.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        """
        A decorator to preserve the original function's docstring.

        Args:
            func: The original function.

        Returns:
            The original function with its original docstring.
        """
        func.__doc__ = wrapper.__doc__
        return func

    return decorator

def _validate_input_policy(policy):
    if policy not in {"coerce", "strict"}:
        raise ValueError(
            f"Unsupported NRSS input_policy {policy!r}. "
            "Supported policies are 'coerce' and 'strict'."
        )
    return policy


def _validate_output_policy(policy):
    normalized = "backend-native" if policy == "backend" else policy
    if normalized not in {"numpy", "backend-native"}:
        raise ValueError(
            f"Unsupported NRSS output_policy {policy!r}. "
            "Supported policies are 'numpy', 'backend-native', and alias 'backend'."
        )
    return normalized


def _validate_resident_mode(policy, backend_name):
    return normalize_resident_mode(backend_name, policy)


def _validate_ownership_policy(policy, backend_name):
    if policy is None:
        return "borrow" if backend_name == "cupy-rsoxs" else "copy"
    if policy not in {"copy", "borrow"}:
        raise ValueError(
            f"Unsupported NRSS ownership_policy {policy!r}. "
            "Supported policies are 'copy', 'borrow', or None."
        )
    return policy

class Morphology:
    '''
    Object used to hold the components necessary for a complete NRSS morphology
    and its backend execution contract.

    Attributes
    ----------
    input_mapping : dict
        A dictionary to handle specific configuration input types

    numMaterial : int
        Number of materials present in the morphology

    materials : dict
        A dictionary of Material objects

    PhysSize : float
        The physical size of each cubic voxel's three dimensions

    NumZYX : tuple or list
        Number of voxels in the Z, Y, and X directions (NumZ, NumY, NumX)

    config : dict
        A dictionary of configuration parameters for CyRSoXS

    create_cy_object : bool
        Boolean value that decides if the CyRSoXS objects are created upon instantiation

    simulated : bool
        Boolean value that tracks whether or not the simulation has been run

    Methods
    -------
    load_morph_hdf5(hdf5_file, create_cy_object=True)
        Class method that creates a Morphology object from a morphology HDF5 file

    create_inputData()
        Creates a CyRSoXS InputData object and populates it with parameters from self.config

    create_optical_constants()
        Creates a CyRSoXS RefractiveIndex object and populates it with optical constants from the materials dict

    create_voxel_data()
        Creates a CyRSoXS VoxelData object and populates it with the voxel information from the materials dict

    run(stdout=True,stderr=True, return_xarray=True, print_vec_info=False)
        Creates a CyRSoXS ScatteringPattern object if not already created, and submits all CyRSoXS objects to 
        run the simulation

    scattering_to_xarray(return_xarray=True,print_vec_info=False)
        Copies the CyRSoXS ScatteringPattern arrays to an xarray in the format used by PyHyperScattering for 
        further analysis
    '''

    config_default = {'CaseType': 0, 'Energies': [270.0], 'EAngleRotation': [0.0, 1.0, 0.0],
                      'MorphologyType': 0, 'AlgorithmType': 0, 'WindowingType': 0,
                      'RotMask': 0,
                      'ReferenceFrame': 1,
                      'EwaldsInterpolation': 1}

    def __init__(self, numMaterial, materials=None, PhysSize=None,
                 config={'CaseType': 0, 'MorphologyType': 0, 'Energies': [270.0], 'EAngleRotation': [0.0, 1.0, 0.0]},
                 create_cy_object=True,
                 backend=None,
                 backend_options=None,
                 resident_mode=None,
                 input_policy='coerce',
                 output_policy='numpy',
                 ownership_policy=None):
        """
        Create a morphology and bind it to an NRSS backend.

        Parameters
        ----------
        numMaterial : int
            Number of materials in the morphology.
        materials : dict[int, Material] or None, default None
            Mapping from material ID to :class:`Material` objects. If omitted,
            empty ``Material`` placeholders are created.
        PhysSize : float or None, default None
            Physical voxel size.
        config : dict, default {"CaseType": 0, "MorphologyType": 0,
                                "Energies": [270.0],
                                "EAngleRotation": [0.0, 1.0, 0.0]}
            Simulation configuration. Missing keys are filled from
            ``Morphology.config_default``. In particular, the effective defaults
            also include ``AlgorithmType=0``, ``WindowingType=0``,
            ``RotMask=0``, ``ReferenceFrame=1``, and
            ``EwaldsInterpolation=1``.

            ``EAngleRotation`` is ordered as
            ``[StartAngle, IncrementAngle, EndAngle]``.
        create_cy_object : bool, default True
            If ``True``, prepare backend/runtime objects during construction.
        backend : {"cupy-rsoxs", "cyrsoxs"} or None, default None
            Preferred backend. If omitted, NRSS resolves the default backend
            automatically, preferring ``"cupy-rsoxs"`` when available and
            otherwise falling back to ``"cyrsoxs"``.
        backend_options : mapping or None, default None
            Backend-specific options. These are normalized during construction
            and are exposed later via :attr:`backend_options`.

            ``cyrsoxs`` options:
            - ``dtype``: only ``"float32"`` is supported.

            ``cupy-rsoxs`` options:
            - ``execution_path``: ``"direct_polarization"`` (default) or
              ``"tensor_coeff"``.
              Selects the maintained CuPy execution path.
              ``"direct_polarization"`` is now the default public path.
              ``"tensor_coeff"`` remains the maintained alternate path.
            - ``mixed_precision_mode``: ``None`` (default) or
              ``"reduced_morphology_bit_depth"``.
              Expert-only reduced-precision morphology-input mode. This is the
              approved half-input path for ``cupy-rsoxs``; it is not a generic
              dtype knob. Runtime compute remains ``float32`` / ``complex64``.
            - ``z_collapse_mode``: ``None`` (default) or ``"mean"``.
              Expert-only effective-2D approximation. The public morphology
              shape is not mutated. This mode is currently incompatible with
              ``mixed_precision_mode``.
            - ``kernel_preload_stage``: ``"a1"`` by default on the
              ``"direct_polarization"`` path, otherwise ``"off"`` unless set
              explicitly. Supported values are ``"off"``, ``"a1"``, and
              ``"a2"``.
            - ``igor_shift_backend``: RawKernel compiler selection for the
              Igor-shift kernel family. Defaults to ``"nvcc"`` on the
              ``"direct_polarization"`` path and ``"nvrtc"`` otherwise.
              Supported values are ``"auto"``, ``"nvcc"``, and ``"nvrtc"``.
            - ``direct_polarization_backend``: RawKernel compiler selection for
              maintained direct-polarization kernels. Defaults to ``"nvrtc"``.
              Supported values are ``"auto"``, ``"nvcc"``, and ``"nvrtc"``.
        resident_mode : {"host", "device"} or None, default None
            Location of the authoritative morphology arrays. ``"cupy-rsoxs"``
            defaults to ``"host"`` and also supports ``"device"``.
            ``"cyrsoxs"`` supports only ``"host"``.
        input_policy : {"coerce", "strict"}, default "coerce"
            How input arrays are validated for the selected backend.
            ``"coerce"`` converts arrays as needed. ``"strict"`` requires
            arrays to already satisfy the backend contract.
        output_policy : {"numpy", "backend-native", "backend"}, default "numpy"
            Output array policy for simulation results. ``"backend"`` is
            accepted as an alias for ``"backend-native"``.
        ownership_policy : {"copy", "borrow"} or None, default None
            How input material arrays are retained by the morphology. If
            omitted, this defaults to ``"borrow"`` for ``"cupy-rsoxs"`` and
            ``"copy"`` for ``"cyrsoxs"``.

        Notes
        -----
        ``backend_options`` are normalized at construction time. As a result,
        :attr:`backend_options` may include defaults that were not passed
        explicitly. For example, with ``backend="cupy-rsoxs"`` and no explicit
        ``execution_path``, the normalized defaults include
        ``execution_path="direct_polarization"``,
        ``kernel_preload_stage="a1"``,
        ``igor_shift_backend="nvcc"``, and
        ``direct_polarization_backend="nvrtc"``.
        """

        self._numMaterial = numMaterial
        self._PhysSize = PhysSize
        self.NumZYX = None
        self._backend = resolve_backend_name(backend)
        self._backend_info = get_backend_info(self._backend)
        self._backend_runtime = get_backend_runtime(self._backend)
        self._resident_mode = _validate_resident_mode(resident_mode, self._backend)
        self._backend_options = normalize_backend_options(self._backend, backend_options)
        self._backend_array_contract = resolve_backend_array_contract(
            self._backend,
            self._backend_options,
            resident_mode=self._resident_mode,
        )
        self._runtime_compute_contract = resolve_backend_runtime_contract(
            self._backend,
            self._backend_options,
        )
        self._input_policy = _validate_input_policy(input_policy)
        self._output_policy = _validate_output_policy(output_policy)
        self._ownership_policy = _validate_ownership_policy(ownership_policy, self._backend)
        self._results_locked = False
        self._backend_result = None
        self._backend_runtime_state = {}
        self._backend_timings = {}
        self.input_compatibility_report = []
        self.construction_backend_coercion_report = []
        self.last_backend_coercion_report = []
        self.last_runtime_staging_report = []
        self.last_kernel_backend_report = {}
        self.last_kernel_preload_report = {}
        self.inputData = None
        if self._backend != "cyrsoxs":
            self.inputData = _LegacyInputDataProxy(self)
        self.OpticalConstants = None
        self.voxelData = None
        self.scatteringPattern = None
        # add config keys and values to class dict
        for key in self.config_default:
            if key in config:
                self.__dict__['_'+key] = config[key]
            else:
                self.__dict__['_'+key] = self.config_default[key]

        # add materials
        self.materials = {}
        for i in range(1, self._numMaterial+1):
            if materials is None:
                self.materials[i] = Material(materialID=i)
            else:
                try:
                    if self._ownership_policy == "copy":
                        self.materials[i] = materials[i].copy()
                    else:
                        self.materials[i] = materials[i]
                    if i == 1:
                        self._Energies = materials[i].energies
                        self.NumZYX = materials[i].NumZYX
                except KeyError:
                    warnings.warn('numMaterial is greater than number of Material objects passed in. Creating empty Material')
                    self.materials[i] = Material(materialID=i)

        self._bind_material_owners()
        self._normalize_material_contracts()
        self.normalize_materials_for_backend(report_attr='construction_backend_coercion_report')

        # flag denoting if Morphology has been simulated
        self._simulated = False

        if create_cy_object:
            self.prepare()

    def __repr__(self):
        return f'Morphology (NumMaterial : {self.numMaterial}, PhysSize : {self.PhysSize})'

    def _bind_material_owners(self):
        for material in self.materials.values():
            material._owner_morphology = self

    @staticmethod
    def _material_is_explicit_isotropic(material):
        return bool(getattr(material, "_explicit_isotropic_contract", False))

    def _normalize_material_contract(self, material):
        explicit_isotropic = is_isotropic_s_field_mode(material.S)
        object.__setattr__(material, "_explicit_isotropic_contract", explicit_isotropic)
        if not explicit_isotropic:
            return

        material_id = getattr(material, "materialID", "unknown")
        for field_name in ("theta", "psi"):
            if getattr(material, field_name) is None:
                continue
            warnings.warn(
                f"Material {material_id} uses SFieldMode.ISOTROPIC, so {field_name} is ignored and will be treated as None.",
                stacklevel=2,
            )
            object.__setattr__(material, field_name, None)

    def _normalize_material_contracts(self):
        for material in self.materials.values():
            self._normalize_material_contract(material)

    def _material_contract_field_names(self, material, *, include_effective_euler=False):
        if self._material_is_explicit_isotropic(material) and not include_effective_euler:
            return ("Vfrac",)
        return ("Vfrac", "S", "theta", "psi")

    def _material_effective_field(self, material, field_name):
        if field_name == "Vfrac":
            return material.Vfrac

        if not self._material_is_explicit_isotropic(material):
            return getattr(material, field_name)

        vfrac = material.Vfrac
        if vfrac is None:
            return None

        info = inspect_array(vfrac)
        if info["namespace"] == "numpy":
            return np.zeros_like(np.asarray(vfrac), dtype=vfrac.dtype)
        if info["namespace"] == "cupy":
            xp = get_namespace_module("cupy")
            return xp.zeros_like(vfrac, dtype=vfrac.dtype)
        raise TypeError(
            "Explicit isotropic material contract requires Vfrac to be a NumPy or CuPy array "
            f"before synthesizing effective field {field_name!r}."
        )

    def _material_effective_fields(self, material):
        return {
            field_name: self._material_effective_field(material, field_name)
            for field_name in ("Vfrac", "S", "theta", "psi")
        }

    def _assert_mutation_allowed(self, target="morphology"):
        if self._results_locked:
            raise RuntimeError(
                f"Cannot mutate {target} after a simulation result has been created for this "
                "Morphology. Create a new Morphology or release/invalidate results first."
            )

    def _lock_results(self):
        self._results_locked = True

    def _reset_result_state(self):
        self._results_locked = False
        self._backend_result = None
        self.scatteringPattern = None
        self._simulated = False
        self._backend_timings = {}
        if hasattr(self, "scattering_data"):
            self.scattering_data = None

    @property
    def ownership_policy(self):
        return self._ownership_policy

    @property
    def resident_mode(self):
        return self._resident_mode

    @property
    def backend_timings(self):
        return dict(self._backend_timings)

    def _set_private_backend_timing_segments(self, segments):
        unique_segments = tuple(dict.fromkeys(str(segment) for segment in segments))
        self._backend_runtime_state["_private_backend_timing_segments"] = unique_segments
        self._backend_timings = {}

    def _clear_private_backend_timing_segments(self):
        self._backend_runtime_state.pop("_private_backend_timing_segments", None)
        self._backend_timings = {}

    def release_runtime(self):
        try:
            if hasattr(self._backend_runtime, "release"):
                self._backend_runtime.release(self)
        finally:
            self._backend_runtime_state.clear()
            self._reset_result_state()

    @property
    def CaseType(self):
        return self._CaseType

    @CaseType.setter
    def CaseType(self, casevalue):
        self._assert_mutation_allowed("Morphology.CaseType")
        if (casevalue != 0) & (casevalue != 1) & (casevalue !=2):
            raise ValueError('CaseType must be 0, 1, or 2')
        else:
            self._CaseType = casevalue

            if self.inputData:
                cy = _require_cyrsoxs_module()
                self.inputData.setCaseType(_cyrsoxs_input_mapping(cy)['CaseType'][1][casevalue])

    @property
    def Energies(self):
        return self._Energies

    @Energies.setter
    def Energies(self, Elist):
        self._assert_mutation_allowed("Morphology.Energies")
        self._Energies = Elist

        if self.inputData:
            self.inputData.setEnergies(Elist)

    @property
    def EAngleRotation(self):
        return self._EAngleRotation

    @EAngleRotation.setter
    def EAngleRotation(self, anglelist):
        self._assert_mutation_allowed("Morphology.EAngleRotation")
        self._EAngleRotation = anglelist

        if self.inputData:
            self.inputData.setERotationAngle(StartAngle=anglelist[0],
                                             EndAngle=anglelist[2],
                                             IncrementAngle=anglelist[1])

    @property
    def MorphologyType(self):
        return self._MorphologyType

    @MorphologyType.setter
    def MorphologyType(self, value):
        self._assert_mutation_allowed("Morphology.MorphologyType")
        if value != 0:
            raise ValueError('Only Euler Morphology is currently supported')
        else:
            self._MorphologyType = value

            if self.inputData:
                cy = _require_cyrsoxs_module()
                self.inputData.setMorphologyType(_cyrsoxs_input_mapping(cy)['MorphologyType'][1][value])

    @property
    def AlgorithmType(self):
        return self._AlgorithmType

    @AlgorithmType.setter
    def AlgorithmType(self, value):
        self._assert_mutation_allowed("Morphology.AlgorithmType")
        if (value != 0) & (value != 1):
            raise ValueError('AlgorithmType must be 0 (communication minimizing) or 1 (memory minimizing).')
        else:
            self._AlgorithmType = value
            if self.inputData:
                self.inputData.setAlgorithm(AlgorithmID=value,MaxStreams=1)

    @property
    def WindowingType(self):
        return self._WindowingType

    @WindowingType.setter
    def WindowingType(self, value):
        self._assert_mutation_allowed("Morphology.WindowingType")
        if (value != 0) & (value != 1):
            raise ValueError('WindowingType must be 0 (None) or 1 (Hanning).')
        else:
            self._WindowingType = value

            if self.inputData:
                cy = _require_cyrsoxs_module()
                self.inputData.windowingType = _cyrsoxs_input_mapping(cy)['WindowingType'][1][value]

    @property
    def RotMask(self):
        return self._RotMask

    @RotMask.setter
    def RotMask(self, value):
        self._assert_mutation_allowed("Morphology.RotMask")
        if (value != 0) & (value != 1):
            raise ValueError('RotMask must be 0 (False) or 1 (True).')
        else:
            self._RotMask = value

            if self.inputData:
                self.inputData.rotMask = self._RotMask

    @property
    def EwaldsInterpolation(self):
        return self._EwaldsInterpolation

    @EwaldsInterpolation.setter
    def EwaldsInterpolation(self, value):
        self._assert_mutation_allowed("Morphology.EwaldsInterpolation")
        if (value != 0) & (value != 1):
            raise ValueError('EwaldsInterpolation must be 0 (Nearest Neighbor) or 1 (Trilinear).')
        else:
            self._EwaldsInterpolation = value

            if self.inputData:
                cy = _require_cyrsoxs_module()
                self.inputData.interpolationType = _cyrsoxs_input_mapping(cy)['EwaldsInterpolation'][1][value]

    @property
    def ReferenceFrame(self):
        return self._ReferenceFrame

    @ReferenceFrame.setter
    def ReferenceFrame(self, value):
        self._assert_mutation_allowed("Morphology.ReferenceFrame")
        if (value != 0) & (value != 1):
            raise ValueError('ReferenceFrame must be 0 (Material Frame) or 1 (Lab Frame - Default).')
        else:
            self._ReferenceFrame = value

            if self.inputData:
                cy = _require_cyrsoxs_module()
                self.inputData.referenceFrame = _cyrsoxs_input_mapping(cy)['ReferenceFrame'][1][value]

    @property
    def simulated(self):
        return self._simulated

    @property
    def backend(self):
        return self._backend

    @property
    def backend_info(self):
        return self._backend_info

    @property
    def backend_options(self):
        return dict(self._backend_options)

    @property
    def mixed_precision_mode(self):
        return self._backend_options.get("mixed_precision_mode")

    @property
    def z_collapse_mode(self):
        return self._backend_options.get("z_collapse_mode")

    @property
    def backend_array_contract(self):
        return self.authoritative_array_contract

    @property
    def authoritative_array_contract(self):
        return dict(self._backend_array_contract)

    @property
    def runtime_compute_contract(self):
        return dict(self._runtime_compute_contract)

    @property
    def backend_dtype(self):
        return self._backend_array_contract["dtype"]

    @property
    def runtime_dtype(self):
        return self._runtime_compute_contract["dtype"]

    @property
    def runtime_compute_dtype(self):
        return self._runtime_compute_contract["runtime_compute_dtype"]

    @property
    def input_policy(self):
        return self._input_policy

    def _effective_input_policy(self):
        if self.backend == "cupy-rsoxs" and self.mixed_precision_mode is not None:
            return "strict"
        return self.input_policy

    @property
    def output_policy(self):
        return self._output_policy

    @property
    def PhysSize(self):
        return self._PhysSize

    @PhysSize.setter
    def PhysSize(self, val):
        self._assert_mutation_allowed("Morphology.PhysSize")
        if val < 0:
            raise ValueError('PhysSize must be greater than 0')
        self._PhysSize = float(val)
        # update inputData object
        if self.inputData:
            self.inputData.setPhysSize(self._PhysSize)

    @property
    def numMaterial(self):
        return self._numMaterial

    def _require_backend(self, backend_name, operation):
        if self.backend != backend_name:
            raise BackendUnavailableError(
                f"{operation} requires backend {backend_name!r}, but this Morphology "
                f"is configured for backend {self.backend!r}."
            )

    def _refresh_backend_info(self):
        self._backend_info = get_backend_info(self.backend)

    def _refresh_backend_contract(self):
        self._backend_array_contract = resolve_backend_array_contract(
            self.backend,
            self._backend_options,
            resident_mode=self.resident_mode,
        )
        self._runtime_compute_contract = resolve_backend_runtime_contract(
            self.backend,
            self._backend_options,
        )

    def _collect_backend_assessment(self):
        self._normalize_material_contracts()
        contract = self._backend_array_contract
        backend_name = self._backend
        resident_mode = self._resident_mode
        backend_options = self._backend_options
        reports = []
        for material_id, mat in self.materials.items():
            for field_name in self._material_contract_field_names(mat):
                reports.append(
                    assess_array_for_backend(
                        getattr(mat, field_name),
                        backend_name=backend_name,
                        field_name=field_name,
                        material_id=material_id,
                        backend_options=backend_options,
                        resident_mode=resident_mode,
                        contract=contract,
                    )
                )
        return reports

    @staticmethod
    def _plan_requires_coercion(plan):
        return (
            plan.transfer != 'none'
            or plan.requires_dtype_cast
            or plan.requires_layout_copy
        )

    def _format_backend_input_error(self, plans, *, strict):
        header = (
            f"Morphology backend input normalization failed for backend {self.backend!r}."
            if not strict
            else (
                f"Morphology backend input normalization would require coercion under "
                f"input_policy='strict' for backend {self.backend!r} with "
                f"resident_mode={self.resident_mode!r}."
            )
        )
        if strict:
            namespace_label = "NumPy" if self._backend_array_contract["namespace"] == "numpy" else "CuPy"
            header += (
                f" Expected {self.resident_mode}-resident {namespace_label} inputs to be "
                f"ZYX-shaped, C-contiguous, {self.backend_dtype} arrays for each material field."
            )
            if self.input_policy != "strict" and self._effective_input_policy() == "strict":
                header += (
                    f" backend_options['mixed_precision_mode']={self.mixed_precision_mode!r} "
                    "overrides input_policy and behaves as strict."
                )
        details = []
        for plan in plans:
            material_label = "unknown" if plan.material_id is None else str(plan.material_id)
            details.append(
                f"- material {material_label} field {plan.field_name}: {plan.reason} "
                f"(from {plan.original_namespace}/{plan.original_dtype} "
                f"to {plan.target_namespace}/{plan.target_dtype}; transfer={plan.transfer})"
            )
        return "\n".join([header, *details])

    def refresh_backend_assessment(self):
        reports = self._collect_backend_assessment()
        self.input_compatibility_report = reports
        self._refresh_backend_contract()
        self._refresh_backend_info()
        return reports

    def normalize_materials_for_backend(self, report_attr='last_backend_coercion_report'):
        self._normalize_material_contracts()
        reports = self._collect_backend_assessment()
        unsupported = [plan for plan in reports if not plan.supported]
        if unsupported:
            raise TypeError(self._format_backend_input_error(unsupported, strict=False))

        if self._effective_input_policy() == 'strict':
            required = [
                plan for plan in reports
                if plan.original_namespace != 'missing' and self._plan_requires_coercion(plan)
            ]
            if required:
                raise TypeError(self._format_backend_input_error(required, strict=True))
            setattr(self, report_attr, reports)
            self.input_compatibility_report = reports
            return self.input_compatibility_report

        contract = self._backend_array_contract
        backend_name = self._backend
        resident_mode = self._resident_mode
        backend_options = self._backend_options
        normalized_reports = []
        for material_id, mat in self.materials.items():
            for field_name in self._material_contract_field_names(mat):
                plan = assess_array_for_backend(
                    getattr(mat, field_name),
                    backend_name=backend_name,
                    field_name=field_name,
                    material_id=material_id,
                    backend_options=backend_options,
                    resident_mode=resident_mode,
                    contract=contract,
                )
                normalized_reports.append(plan)
                if getattr(mat, field_name) is not None:
                    setattr(mat, field_name, coerce_array_for_backend(getattr(mat, field_name), plan))

        setattr(self, report_attr, normalized_reports)
        self.input_compatibility_report = self._collect_backend_assessment()
        return self.input_compatibility_report

    def _coerce_material_field(self, value, field_name, material_id):
        plan = assess_array_for_backend(
            value,
            backend_name=self._backend,
            field_name=field_name,
            material_id=material_id,
            backend_options=self._backend_options,
            resident_mode=self._resident_mode,
            contract=self._backend_array_contract,
        )
        self.last_backend_coercion_report.append(plan)
        if value is None:
            return None
        return coerce_array_for_backend(value, plan)

    # @numMaterial.setter
    # def numMaterial(self, val):
    #     if val < 0:
    #         raise ValueError('numMaterial must be greater than 0')
    #     self._numMaterial = int(val)
    #     # if we change the number of materials and we have an inputData object, we need to recreate it with the new number of materials
    #     if self.inputData:
    #         self.create_inputData()
    #     if self.OpticalConstants:
    #         self.update_optical_constants()

    @property
    def config(self):
        return {key: self.__dict__['_'+key] for key in self.config_default}

    @config.setter
    def config(self, dict1):
        self._assert_mutation_allowed("Morphology.config")
        for key in dict1:
            if key in self.config_default:
                self.__dict__['_'+key] = dict1[key]
            else:
                warnings.warn(f'Key {key} not supported')

    @classmethod
    def load_morph_hdf5(
        cls,
        hdf5_file,
        create_cy_object=False,
        backend=None,
        backend_options=None,
        resident_mode=None,
        input_policy='coerce',
        output_policy='numpy',
        ownership_policy=None,
    ):
        with h5py.File(hdf5_file, 'r') as f:
            if 'Euler_Angles' not in f.keys():
                raise KeyError('Only the Euler Angle convention is currently supported')
            # get number of materials in HDF5
            numMat = check_NumMat(f, morphology_type=0)
            PhysSize = f['Morphology_Parameters/PhysSize'][()]
            materials = dict()

            for i in range(numMat):
                materialID = i + 1
                # Load arrays directly without type conversion
                materials[materialID] = Material(
                    materialID=materialID,
                    Vfrac=f[f'Euler_Angles/Mat_{i+1}_Vfrac'][()],
                    S=f[f'Euler_Angles/Mat_{i+1}_S'][()],
                    theta=f[f'Euler_Angles/Mat_{i+1}_Theta'][()],
                    psi=f[f'Euler_Angles/Mat_{i+1}_Psi'][()],
                    NumZYX=f[f'Euler_Angles/Mat_{i+1}_Vfrac'][()].shape)

        return cls(
            numMat,
            materials=materials,
            PhysSize=PhysSize,
            create_cy_object=create_cy_object,
            backend=backend,
            backend_options=backend_options,
            resident_mode=resident_mode,
            input_policy=input_policy,
            output_policy=output_policy,
            ownership_policy=ownership_policy,
        )

    def load_config(self, config_file):
        self.config = read_config(config_file)

    def load_matfile(self, matfile):
        return read_material(matfile)

    def prepare(self):
        return self._backend_runtime.prepare(self)

    def create_inputData(self):
        self._require_backend('cyrsoxs', 'create_inputData')
        cy = _require_cyrsoxs_module()
        self.inputData = cy.InputData(NumMaterial=self._numMaterial)
        # parse config dictionary and assign to appropriate places in inputData object
        self.config_to_inputData()

        if self.NumZYX is None:
            self.NumZYX = self.materials[1].NumZYX

        # only support ZYX ordering at the moment
        self.inputData.setDimensions(self.NumZYX, cy.MorphologyOrder.ZYX)

        if self.PhysSize is not None:
            self.inputData.setPhysSize(self.PhysSize)

        if not self.inputData.validate():
            warnings.warn('Validation failed. Double check inputData values')

    def create_optical_constants(self):
        self._require_backend('cyrsoxs', 'create_optical_constants')
        cy = _require_cyrsoxs_module()
        self.OpticalConstants = cy.RefractiveIndex(self.inputData)
        self.update_optical_constants()        
        if not self.OpticalConstants.validate():
            warnings.warn('Validation failed. Double check optical constant values')

    def update_optical_constants(self):
        # Pre-allocate list to avoid repeated allocations
        all_constants = [None] * self._numMaterial
        for energy in self.Energies:
            # Update list in-place
            for ID in range(1, self.numMaterial+1):
                all_constants[ID-1] = self.materials[ID].opt_constants[energy]
            self.OpticalConstants.addData(OpticalConstants=all_constants, Energy=energy)

    def create_voxel_data(self):
        self._require_backend('cyrsoxs', 'create_voxel_data')
        cy = _require_cyrsoxs_module()
        self.voxelData = cy.VoxelData(InputData=self.inputData)
        self.update_voxel_data()
        if not self.voxelData.validate():
            warnings.warn('Validation failed. Double check voxel data values')

    def update_voxel_data(self):
        self._require_backend('cyrsoxs', 'update_voxel_data')
        self._normalize_material_contracts()
        self.last_backend_coercion_report = []
        for ID in range(1, self.numMaterial+1):
            mat = self.materials[ID]
            effective_fields = self._material_effective_fields(mat)
            s = self._coerce_material_field(effective_fields['S'], field_name='S', material_id=ID)
            theta = self._coerce_material_field(effective_fields['theta'], field_name='theta', material_id=ID)
            psi = self._coerce_material_field(effective_fields['psi'], field_name='psi', material_id=ID)
            vfrac = self._coerce_material_field(effective_fields['Vfrac'], field_name='Vfrac', material_id=ID)
            
            self.voxelData.addVoxelData(
                S=s,
                Theta=theta,
                Psi=psi,
                Vfrac=vfrac,
                MaterialID=ID)

    def config_to_inputData(self):
        self._require_backend('cyrsoxs', 'config_to_inputData')
        cy = _require_cyrsoxs_module()
        input_mapping = _cyrsoxs_input_mapping(cy)
        for key in self.config:
            if key == "Energies":
                self.inputData.setEnergies(self.config[key])
            elif key == 'EAngleRotation':
                angles = self.config[key]
                self.inputData.setERotationAngle(StartAngle=float(angles[0]),
                                                 EndAngle=float(angles[2]),
                                                 IncrementAngle=float(angles[1]))
            elif key == 'AlgorithmType':
                self.inputData.setAlgorithm(AlgorithmID=self.config[key], MaxStreams=1)
            # if the key corresponds to one of the idiosyncratic methods, use this
            elif key in input_mapping.keys():
                func = getattr(self.inputData, input_mapping[key][0])
                if callable(func):
                    func(input_mapping[key][1][self.config[key]])
                # if the attribute is not callable, use input_mapping to set the attribute
                else:
                    setattr(self.inputData,
                            input_mapping[key][0],
                            input_mapping[key][1][self.config[key]])
            else:
                warnings.warn(f'{key} is currently not implemented')

    def create_update_cy(self):
        self._require_backend('cyrsoxs', 'create_update_cy')
        self.refresh_backend_assessment()
        # create or update all CyRSoXS objects
        if self.inputData:
            self.config_to_inputData()
        else:
            self.create_inputData()

        # create or update OpticalConstants
        if self.OpticalConstants:
            self.update_optical_constants()            
        else:
            self.create_optical_constants()

        # create or udpate voxelData
        if self.voxelData:
            self.voxelData.reset()
            self.update_voxel_data()
        else:
            self.create_voxel_data()

    def write_to_file(self, fname, author='NIST'):
        self._normalize_material_contracts()
        _ = write_hdf5(
            [
                [
                    self._material_effective_field(self.materials[i], field_name)
                    for field_name in ('Vfrac', 'S', 'theta', 'psi')
                ]
                for i in self.materials
            ],
            self.PhysSize,
            fname,
            self.MorphologyType,
            ordering='ZYX',
            author=author,
        )

    # TODO : function to write a config.txt file from config dict
    def write_config(self,):
        pass

    def write_constants(self, path=None):
        for i in range(1, self._numMaterial+1):
            write_opts(self.materials[i].opt_constants, i, path)

    # submit to CyRSoXS
    def run(self, stdout=True, stderr=True, return_xarray=True, print_vec_info=False, validate=False):
        return self._backend_runtime.run(
            self,
            stdout=stdout,
            stderr=stderr,
            return_xarray=return_xarray,
            print_vec_info=print_vec_info,
            validate=validate,
        )

    def scattering_to_xarray(self, return_xarray=True, print_vec_info=False):
        if self.backend != "cyrsoxs":
            if self._backend_result is None:
                warnings.warn('You haven\'t run your simulation yet')
                return None
            scattering_data = self._backend_result.to_xarray()
            if return_xarray:
                return scattering_data
            self.scattering_data = scattering_data
            return None

        if self.simulated:
            if not print_vec_info:
                old_stdout = sys.stdout
                f = open(os.devnull, 'w')
                sys.stdout = f

            try:
                # Get data in [energy, NumY, NumX] shape
                scattering_data = self.scatteringPattern.writeAllToNumpy(kID=0)
                
                # Pre-compute FFT frequencies once
                ny, nx = self.NumZYX[1:]
                d = self.PhysSize
                
                # Calculate q-vectors
                qy = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(ny, d=d))
                qx = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(nx, d=d))
                
                # Create DataArray with pre-computed coordinates
                scattering_data = xr.DataArray(
                    scattering_data,
                    dims=['energy', 'qy', 'qx'],
                    coords={'qy': qy, 'qx': qx, 'energy': self.config['Energies']},
                    attrs={
                        'phys_size_nm': float(self.PhysSize),
                        'z_dim': int(self.NumZYX[0]),
                    },
                )
                
                if return_xarray:
                    return scattering_data
                else:
                    self.scattering_data = scattering_data
                    
            finally:
                if not print_vec_info:
                    sys.stdout = old_stdout
                    f.close()
        else:
            warnings.warn('You haven\'t run your simulation yet')

    # TODO : restructure to have a single checkH5 engine for both NRSS and
    # command line formats
    def check_materials(self, quiet=True):
        self._normalize_material_contracts()
        coerced_vfracs = []
        reference_shape = None

        # Check each material's properties
        for i, mat in self.materials.items():
            field_names = self._material_contract_field_names(mat)
            material_shape = None
            # Validate value ranges and types in one pass
            for name in field_names:
                arr = getattr(mat, name)
                if arr is None:
                    raise AssertionError(f'Material {i} {name} is not populated')

                plan = assess_array_for_backend(
                    arr,
                    backend_name=self.backend,
                    field_name=name,
                    material_id=i,
                    backend_options=self.backend_options,
                    resident_mode=self.resident_mode,
                )
                if not plan.supported:
                    raise AssertionError(plan.reason)

                validation_arr = arr
                xp = get_namespace_module(plan.original_namespace)
                if plan.original_namespace == "cupy" and str(arr.dtype) == "float16":
                    validation_arr = np.asarray(arr.get(), dtype=np.float16)
                    xp = np

                # Check for NaNs and float type in one operation
                if not np.issubdtype(validation_arr.dtype, np.floating):
                    raise AssertionError(f'Material {i} {name} is not of type float')
                
                if to_python_bool(xp.any(xp.isnan(validation_arr))):
                    raise AssertionError(f'NaNs are present in Material {i} {name}')

                field_shape = tuple(int(dim) for dim in validation_arr.shape)
                if material_shape is None:
                    material_shape = field_shape
                elif field_shape != material_shape:
                    raise AssertionError(
                        f'Material {i} field {name} shape {field_shape} does not match '
                        f'the material reference shape {material_shape}'
                    )
                
                # Check bounds for S and Vfrac
                if name in ('S', 'Vfrac'):
                    if not to_python_bool(xp.all((validation_arr >= 0) & (validation_arr <= 1))):
                        raise AssertionError(f'Material {i} {name} value(s) does not lie between 0 and 1')

            if reference_shape is None:
                reference_shape = material_shape
            elif material_shape != reference_shape:
                raise AssertionError(
                    f'Material {i} Vfrac shape {material_shape} does not match '
                    f'the morphology reference shape {reference_shape}'
                )

            vfrac_plan = assess_array_for_backend(
                mat.Vfrac,
                backend_name=self.backend,
                field_name='Vfrac',
                material_id=i,
                backend_options=self.backend_options,
                resident_mode=self.resident_mode,
            )
            if vfrac_plan.original_namespace == "cupy" and str(mat.Vfrac.dtype) == "float16":
                coerced_vfracs.append(np.asarray(mat.Vfrac.get(), dtype=np.float16))
            else:
                coerced_vfracs.append(coerce_array_for_backend(mat.Vfrac, vfrac_plan))

        # Vectorized sum of volume fractions in backend-compatible space
        Vfrac_sum = coerced_vfracs[0]
        for arr in coerced_vfracs[1:]:
            Vfrac_sum = Vfrac_sum + arr

        sum_namespace = inspect_array(Vfrac_sum)['namespace']
        sum_xp = get_namespace_module(sum_namespace)
        if self.mixed_precision_mode == "reduced_morphology_bit_depth":
            closure_error = sum_xp.abs(Vfrac_sum - Vfrac_sum.dtype.type(1.0))
            assert to_python_bool(sum_xp.all(closure_error <= Vfrac_sum.dtype.type(1e-3))), (
                'Total material volume fractions do not satisfy the mixed-precision '
                'voxelwise closure budget of abs(sum_i Vfrac_i - 1) <= 1e-3'
            )
        else:
            assert to_python_bool(sum_xp.allclose(Vfrac_sum, 1)), 'Total material volume fractions do not sum to 1'

        if not quiet:
            print('All material checks have passed')

    @wraps(morphology_visualizer)
    def visualize_materials(self, *args,**kwargs):
        return morphology_visualizer(self, *args,**kwargs)
    visualize_materials.__doc__ = morphology_visualizer.__doc__


    def validate_all(self, quiet=True):
        return self._backend_runtime.validate_all(self, quiet=quiet)


class OpticalConstants:
    '''
    Object to hold dielectric optical constants in a format compatible with CyRSoXS

    Attributes
    ----------

    energies : list or array
        List of energies
    opt_constants : dict
        Dictionary of optical constants, where each energy is a key in the dict
    name : str
        String identifying the element or material for these optical constants

    Methods
    -------
    calc_constants(energies, reference_data, name='unkown')
        Interpolates optical constant data to the list of energies provided
    load_matfile(matfile, name='unknown')
        Creates an OpticalConstants object from a previously written MaterialX.txt file
    create_vacuum(energies)
        Convenience function to populate zeros for all optical constants

    '''

    def __init__(self, energies, opt_constants=None, name='unknown'):
        self.energies = energies
        self.opt_constants = opt_constants
        self.name = name
        if self.name == 'vacuum':
            self.create_vacuum(energies)

    def __repr__(self):
        return f'OpticalConstants (Material : {self.name}, Number of Energies : {len(self.energies)})'

    @classmethod
    def calc_constants(cls, energies, reference_data, name='unknown'):
        deltabeta = dict()
        for energy in energies:
            dPara = np.interp(energy, reference_data['Energy'], reference_data['DeltaPara'])
            bPara = np.interp(energy, reference_data['Energy'], reference_data['BetaPara'])
            dPerp = np.interp(energy, reference_data['Energy'], reference_data['DeltaPerp'])
            bPerp = np.interp(energy, reference_data['Energy'], reference_data['BetaPerp'])
            deltabeta[energy] = [dPara, bPara, dPerp, bPerp]
        return cls(energies, deltabeta, name=name)

    @classmethod
    def load_matfile(cls, matfile, name='unknown'):
        energies, deltabeta = read_material(matfile)
        return cls(energies, deltabeta, name=name)

    def create_vacuum(self, energies):
        deltabeta = dict()
        for energy in energies:
            deltabeta[energy] = [0.0, 0.0, 0.0, 0.0]
        self.energies = energies
        self.opt_constants = deltabeta


class Material(OpticalConstants):
    '''
    Object to hold the voxel-level data for an NRSS morphology. Inherits from
    the OpticalConstants class.

    Attributes
    ----------
    materialID : int
        Integer value denoting the material number. Used in CyRSoXS
    Vfrac : ndarray
        Volume fractions for a Material
    S : ndarray or SFieldMode
        Orientational order parameter field, or SFieldMode.ISOTROPIC for the
        explicit full-material isotropic contract
    theta : ndarray
        The second Euler angle (ZYZ convention)
    psi : ndarray
        The third Euler angle (ZYZ convention)
    NumZYX : tuple or list
        Dimensions of the Material arrays (NumZ, NumY, NumX)
    name : str
        Name of the Material (e.g. 'Polystyrene')

    '''

    _guarded_attributes = {
        "Vfrac",
        "S",
        "theta",
        "psi",
        "NumZYX",
        "energies",
        "opt_constants",
        "name",
    }

    def __init__(self, materialID=1, Vfrac=None, S=None, theta=None, psi=None,
                 NumZYX=None, energies=None, opt_constants=None, name=None):
        """
        Create a material field bundle and optional optical-constant mapping.

        Parameters
        ----------
        materialID : int, default 1
            Integer material identifier used by NRSS.
        Vfrac : ndarray or None, default None
            Volume-fraction field for this material.
        S : ndarray or SFieldMode or None, default None
            Orientational order parameter field. Use
            ``SFieldMode.ISOTROPIC`` to declare an explicitly isotropic
            material contract.
        theta : ndarray or None, default None
            Second Euler angle field in the NRSS ZYZ convention.
        psi : ndarray or None, default None
            Third Euler angle field in the NRSS ZYZ convention.
        NumZYX : tuple[int, int, int] or None, default None
            Material array shape as ``(NumZ, NumY, NumX)``. If omitted and
            ``Vfrac`` is array-like, the shape is inferred from ``Vfrac``.
        energies : sequence[float] or None, default None
            Energy grid for the optical constants.
        opt_constants : dict or None, default None
            Optical constants keyed by energy.
        name : str or None, default None
            Optional material name. ``name="vacuum"`` is treated as an explicit
            isotropic convenience contract.

        Notes
        -----
        The isotropic formalism is important enough to document explicitly:

        - ``SFieldMode.ISOTROPIC`` means this material should be treated as
          fully isotropic.
        - Under this contract, ``theta`` and ``psi`` are ignored and are
          normalized to ``None``.
        - Using this contract correctly can dramatically reduce the memory
          footprint of the calculation and improve runtime, because NRSS does
          not need to carry full orientation fields for that material.
        - ``name="vacuum"`` automatically opts into this explicit isotropic
          contract. If conflicting ``S``, ``theta``, or ``psi`` values are
          provided, they are ignored with a warning.

        This isotropic contract is semantic, not just shorthand for filling
        orientation arrays with zeros.
        """
        self._owner_morphology = None
        self._explicit_isotropic_contract = False
        self.materialID = materialID
        # Store arrays as-is, type conversion will happen when needed
        self.Vfrac = Vfrac
        self.S = S
        self.theta = theta
        self.psi = psi
        self.NumZYX = NumZYX
        self.name = name
        self.energies = energies
        self.opt_constants = opt_constants
        if self.NumZYX is None:
            try:
                self.NumZYX = Vfrac.shape
            except AttributeError:
                pass

        if (energies is None) & (opt_constants is not None):
            self.energies = list(opt_constants.keys())

        self._normalize_named_vacuum_contract()
        super().__init__(self.energies, self.opt_constants, name=name)

    def __repr__(self):
        return f'Material (Name : {self.name}, ID : {self.materialID}, Shape : {self.NumZYX})'

    @staticmethod
    def _format_ignored_field_names(field_names):
        if len(field_names) == 1:
            return field_names[0]
        if len(field_names) == 2:
            return f"{field_names[0]} and {field_names[1]}"
        return ", ".join(field_names[:-1]) + f", and {field_names[-1]}"

    def _normalize_named_vacuum_contract(self):
        if self.__dict__.get("name") != "vacuum":
            return

        ignored_fields = []
        s_value = self.__dict__.get("S")
        if s_value is not None and not is_isotropic_s_field_mode(s_value):
            ignored_fields.append("S")

        for field_name in ("theta", "psi"):
            if self.__dict__.get(field_name) is not None:
                ignored_fields.append(field_name)

        if ignored_fields:
            ignored_field_names = self._format_ignored_field_names(tuple(ignored_fields))
            verb = "is" if len(ignored_fields) == 1 else "are"
            warnings.warn(
                f"Material {self.materialID} uses name='vacuum', so {ignored_field_names} "
                f"{verb} ignored and the explicit isotropic contract "
                "SFieldMode.ISOTROPIC will be used.",
                stacklevel=3,
            )

        object.__setattr__(self, "S", SFieldMode.ISOTROPIC)
        object.__setattr__(self, "theta", None)
        object.__setattr__(self, "psi", None)
        object.__setattr__(self, "_explicit_isotropic_contract", True)

    def __setattr__(self, key, value):
        owner = self.__dict__.get("_owner_morphology")
        if owner is not None and key in self._guarded_attributes:
            owner._assert_mutation_allowed(f"Material.{key}")
        super().__setattr__(key, value)
        if key in {"name", "S", "theta", "psi"}:
            self._normalize_named_vacuum_contract()
        if owner is not None and key in {"name", "S", "theta", "psi"}:
            owner._normalize_material_contract(self)

    @staticmethod
    def _copy_field(value):
        if value is None:
            return None
        if hasattr(value, 'copy'):
            try:
                return value.copy()
            except TypeError:
                pass
        return copy.copy(value)

    def __copy__(self):
        return Material(materialID=self.materialID,
                        Vfrac=self._copy_field(self.Vfrac),
                        S=self._copy_field(self.S),
                        theta=self._copy_field(self.theta),
                        psi=self._copy_field(self.psi),
                        NumZYX=self.NumZYX,
                        energies=self.energies,
                        opt_constants=self.opt_constants,
                        name=self.name)

    def copy(self):
        return copy.copy(self)
