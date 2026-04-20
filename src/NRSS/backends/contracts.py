from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class BackendOptionError(ValueError):
    """Raised when backend options are invalid for the selected backend."""


class ResidentModeError(ValueError):
    """Raised when resident_mode is invalid for the selected backend."""


_BACKEND_ARRAY_CONTRACTS = {
    "cyrsoxs": {
        "default_resident_mode": "host",
        "supported_resident_modes": ("host",),
        "resident_modes": {
            "host": {
                "namespace": "numpy",
                "device": "cpu",
            },
        },
        "runtime_namespace": "numpy",
        "runtime_device": "cpu",
        "default_dtype": "float32",
        "supported_dtypes": ("float32",),
        "supported_backend_options": ("dtype",),
        "default_mixed_precision_mode": None,
        "supported_mixed_precision_modes": (None,),
        "runtime_compute_dtype": "float32",
        "runtime_complex_dtype": "complex64",
    },
    "cupy-rsoxs": {
        "default_resident_mode": "host",
        "supported_resident_modes": ("host", "device"),
        "resident_modes": {
            "host": {
                "namespace": "numpy",
                "device": "cpu",
            },
            "device": {
                "namespace": "cupy",
                "device": "gpu",
            },
        },
        "runtime_namespace": "cupy",
        "runtime_device": "gpu",
        "default_dtype": "float32",
        "supported_dtypes": ("float32", "float16"),
        "default_execution_path": "direct_polarization",
        "supported_execution_paths": (
            "tensor_coeff",
            "direct_polarization",
        ),
        "default_mixed_precision_mode": None,
        "supported_mixed_precision_modes": (
            None,
            "reduced_morphology_bit_depth",
        ),
        "default_z_collapse_mode": None,
        "supported_z_collapse_modes": (
            None,
            "mean",
        ),
        "default_kernel_preload_stage": "off",
        "supported_kernel_preload_stages": (
            "off",
            "a1",
            "a2",
        ),
        "default_igor_shift_backend": "nvrtc",
        "default_direct_polarization_backend": "nvrtc",
        "default_direct_isotropic_mode": None,
        "supported_direct_isotropic_modes": (
            None,
            "cached_base",
        ),
        "default_energy_progress_bar": True,
        "default_result_residency": "host",
        "default_result_chunk_size": 1,
        "default_result_layout": "detector",
        "default_total_chi_wedge_deg": 90.0,
        "supported_rawkernel_backends": (
            "auto",
            "nvcc",
            "nvrtc",
        ),
        "supported_result_residencies": (
            "host",
            "device",
        ),
        "supported_result_layouts": (
            "detector",
            "integrated",
            "i_only",
            "i_para_i_perp",
            "i_a",
        ),
        "supported_backend_options": (
            "execution_path",
            "mixed_precision_mode",
            "z_collapse_mode",
            "kernel_preload_stage",
            "igor_shift_backend",
            "direct_polarization_backend",
            "direct_isotropic_mode",
            "energy_progress_bar",
            "result_residency",
            "result_chunk_size",
            "result_layout",
            "total_chi_wedge_deg",
        ),
        "runtime_compute_dtype": "float32",
        "runtime_complex_dtype": "complex64",
    },
}


def _backend_contract_spec(backend_name: str) -> dict[str, Any]:
    try:
        return _BACKEND_ARRAY_CONTRACTS[backend_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported backend contract lookup for {backend_name!r}."
        ) from exc


def normalize_dtype_name(dtype: Any) -> str:
    try:
        return np.dtype(dtype).name
    except TypeError as exc:
        raise BackendOptionError(
            f"Unsupported backend dtype option {dtype!r}."
        ) from exc


def normalize_mixed_precision_mode_name(mode: Any) -> str | None:
    if mode is None:
        return None

    cleaned = str(mode).strip().lower()
    aliases = {
        "": None,
        "none": None,
        "off": None,
        "default": None,
        "reduced-morphology-bit-depth": "reduced_morphology_bit_depth",
    }
    return aliases.get(cleaned, cleaned)


def normalize_z_collapse_mode_name(mode: Any) -> str | None:
    if mode is None:
        return None

    cleaned = str(mode).strip().lower()
    aliases = {
        "": None,
        "none": None,
        "off": None,
        "default": None,
    }
    return aliases.get(cleaned, cleaned)


def normalize_kernel_preload_stage_name(stage: Any) -> str:
    if stage is None:
        return "off"

    cleaned = str(stage).strip().lower()
    aliases = {
        "": "off",
        "none": "off",
        "default": "off",
        "constructor": "a1",
        "prepare": "a1",
        "staging": "a2",
    }
    return aliases.get(cleaned, cleaned)


def normalize_rawkernel_backend_name(backend: Any) -> str:
    if backend is None:
        return "nvrtc"

    cleaned = str(backend).strip().lower()
    aliases = {
        "": "nvrtc",
        "default": "nvrtc",
    }
    return aliases.get(cleaned, cleaned)


def normalize_direct_isotropic_mode_name(mode: Any) -> str | None:
    if mode is None:
        return None

    cleaned = str(mode).strip().lower()
    aliases = {
        "": None,
        "none": None,
        "off": None,
        "default": None,
        "cached-base": "cached_base",
    }
    return aliases.get(cleaned, cleaned)


def normalize_energy_progress_bar_name(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False

    cleaned = str(value).strip().lower()
    aliases = {
        "": False,
        "0": False,
        "1": True,
        "default": False,
        "false": False,
        "none": False,
        "no": False,
        "off": False,
        "on": True,
        "true": True,
        "yes": True,
    }
    if cleaned not in aliases:
        raise BackendOptionError(
            "Backend 'cupy-rsoxs' does not support energy_progress_bar="
            f"{value!r}. Supported values: True, False, 'on', 'off'."
        )
    return aliases[cleaned]


def normalize_result_residency_name(value: Any) -> str:
    if value is None:
        return "host"

    cleaned = str(value).strip().lower()
    aliases = {
        "": "host",
        "cpu": "host",
        "default": "host",
        "gpu": "device",
    }
    return aliases.get(cleaned, cleaned)


def normalize_result_chunk_size(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"", "auto", "default", "none", "off"}:
            return None
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise BackendOptionError(
            "Backend 'cupy-rsoxs' result_chunk_size must be a positive integer "
            "or one of: auto, default, none, off."
        ) from exc
    if normalized <= 0:
        raise BackendOptionError(
            "Backend 'cupy-rsoxs' result_chunk_size must be a positive integer."
        )
    return normalized


def normalize_result_layout_name(value: Any) -> str:
    if value is None:
        return "detector"

    cleaned = str(value).strip().lower()
    aliases = {
        "": "detector",
        "default": "detector",
        "raw": "detector",
        "scattering": "detector",
        "detector": "detector",
        "integrated": "integrated",
        "reduced": "integrated",
        "polar": "integrated",
        "i-only": "i_only",
        "i_only": "i_only",
        "i": "i_only",
        "i-para-i-perp": "i_para_i_perp",
        "i_para_i_perp": "i_para_i_perp",
        "ipara_iperp": "i_para_i_perp",
        "i-and-a": "i_a",
        "i_a": "i_a",
        "ia": "i_a",
    }
    return aliases.get(cleaned, cleaned)


def normalize_total_chi_wedge_deg(value: Any) -> float:
    if value is None:
        return 90.0
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise BackendOptionError(
            "Backend 'cupy-rsoxs' total_chi_wedge_deg must be a finite float in (0, 180]."
        ) from exc
    if not np.isfinite(normalized) or normalized <= 0.0 or normalized > 180.0:
        raise BackendOptionError(
            "Backend 'cupy-rsoxs' total_chi_wedge_deg must be a finite float in (0, 180]."
        )
    return normalized


def normalize_resident_mode(
    backend_name: str,
    resident_mode: str | None = None,
) -> str:
    spec = _backend_contract_spec(backend_name)
    if resident_mode is None:
        return spec["default_resident_mode"]

    normalized = str(resident_mode).strip().lower()
    aliases = {
        "cpu": "host",
        "gpu": "device",
    }
    normalized = aliases.get(normalized, normalized)

    if normalized not in spec["supported_resident_modes"]:
        raise ResidentModeError(
            f"Unsupported resident_mode {resident_mode!r} for backend {backend_name!r}. "
            f"Supported resident modes: {', '.join(spec['supported_resident_modes'])}."
        )
    return normalized


def normalize_backend_options(
    backend_name: str,
    backend_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec = _backend_contract_spec(backend_name)
    options = {} if backend_options is None else dict(backend_options)

    if backend_name == "cupy-rsoxs" and "dtype" in options:
        raise BackendOptionError(
            "Backend 'cupy-rsoxs' does not expose a generic dtype option. "
            "Remove backend_options['dtype']; for the approved reduced-precision "
            "path use backend_options={'mixed_precision_mode': "
            "'reduced_morphology_bit_depth'}."
        )

    unknown = sorted(set(options) - set(spec["supported_backend_options"]))
    if unknown:
        raise BackendOptionError(
            f"Unsupported backend option(s) for {backend_name!r}: {', '.join(unknown)}. "
            f"Supported options are: {', '.join(spec['supported_backend_options'])}."
        )

    normalized_options = {}
    if "dtype" in spec["supported_backend_options"]:
        normalized_dtype = normalize_dtype_name(options.get("dtype", spec["default_dtype"]))
        if normalized_dtype not in spec["supported_dtypes"]:
            raise BackendOptionError(
                f"Backend {backend_name!r} does not support dtype {normalized_dtype!r}. "
                f"Supported dtypes: {', '.join(spec['supported_dtypes'])}."
            )
        normalized_options["dtype"] = normalized_dtype

    if "execution_path" in spec["supported_backend_options"]:
        execution_path = str(
            options.get("execution_path", spec["default_execution_path"])
        ).strip().lower()
        aliases = {
            "default": spec["default_execution_path"],
            "tensor": "tensor_coeff",
            "direct": "direct_polarization",
        }
        execution_path = aliases.get(execution_path, execution_path)
        if execution_path not in spec["supported_execution_paths"]:
            raise BackendOptionError(
                f"Backend {backend_name!r} does not support execution_path {execution_path!r}. "
                "Supported execution paths: "
                f"{', '.join(spec['supported_execution_paths'])}."
            )
        normalized_options["execution_path"] = execution_path
    if "mixed_precision_mode" in spec["supported_backend_options"]:
        mixed_precision_mode = normalize_mixed_precision_mode_name(
            options.get("mixed_precision_mode", spec["default_mixed_precision_mode"])
        )
        if mixed_precision_mode not in spec["supported_mixed_precision_modes"]:
            supported_modes = tuple(
                "None" if mode is None else mode
                for mode in spec["supported_mixed_precision_modes"]
            )
            raise BackendOptionError(
                f"Backend {backend_name!r} does not support mixed_precision_mode "
                f"{mixed_precision_mode!r}. Supported modes: {', '.join(supported_modes)}."
            )
        normalized_options["mixed_precision_mode"] = mixed_precision_mode
    if "z_collapse_mode" in spec["supported_backend_options"]:
        z_collapse_mode = normalize_z_collapse_mode_name(
            options.get("z_collapse_mode", spec.get("default_z_collapse_mode"))
        )
        if z_collapse_mode not in spec.get("supported_z_collapse_modes", (None,)):
            supported_modes = tuple(
                "None" if mode is None else mode
                for mode in spec.get("supported_z_collapse_modes", (None,))
            )
            raise BackendOptionError(
                f"Backend {backend_name!r} does not support z_collapse_mode "
                f"{z_collapse_mode!r}. Supported modes: {', '.join(supported_modes)}."
            )
        normalized_options["z_collapse_mode"] = z_collapse_mode

    if (
        backend_name == "cupy-rsoxs"
        and normalized_options.get("z_collapse_mode") is not None
        and normalized_options.get("mixed_precision_mode") is not None
    ):
        raise BackendOptionError(
            "Backend 'cupy-rsoxs' does not yet support combining z_collapse_mode "
            "with mixed_precision_mode. Disable one of those expert options."
        )

    if "kernel_preload_stage" in spec["supported_backend_options"]:
        default_kernel_preload_stage = spec.get("default_kernel_preload_stage", "off")
        if backend_name == "cupy-rsoxs" and normalized_options.get("execution_path") == "direct_polarization":
            default_kernel_preload_stage = "a1"
        kernel_preload_stage = normalize_kernel_preload_stage_name(
            options.get("kernel_preload_stage", default_kernel_preload_stage)
        )
        if kernel_preload_stage not in spec.get("supported_kernel_preload_stages", ("off",)):
            raise BackendOptionError(
                f"Backend {backend_name!r} does not support kernel_preload_stage "
                f"{kernel_preload_stage!r}. Supported stages: "
                f"{', '.join(spec.get('supported_kernel_preload_stages', ('off',)))}."
            )
        normalized_options["kernel_preload_stage"] = kernel_preload_stage

    for option_name in ("igor_shift_backend", "direct_polarization_backend"):
        if option_name not in spec["supported_backend_options"]:
            continue
        default_backend = spec.get(f"default_{option_name}", "nvrtc")
        if backend_name == "cupy-rsoxs":
            execution_path = normalized_options.get("execution_path")
            if option_name == "igor_shift_backend" and execution_path == "direct_polarization":
                default_backend = "nvcc"
            if option_name == "direct_polarization_backend" and execution_path == "direct_polarization":
                default_backend = "nvrtc"
        normalized_backend = normalize_rawkernel_backend_name(
            options.get(option_name, default_backend)
        )
        if normalized_backend not in spec.get("supported_rawkernel_backends", ("nvrtc",)):
            raise BackendOptionError(
                f"Backend {backend_name!r} does not support {option_name}={normalized_backend!r}. "
                f"Supported values: {', '.join(spec.get('supported_rawkernel_backends', ('nvrtc',)))}."
            )
        normalized_options[option_name] = normalized_backend

    if "direct_isotropic_mode" in spec["supported_backend_options"]:
        direct_isotropic_mode = normalize_direct_isotropic_mode_name(
            options.get("direct_isotropic_mode", spec.get("default_direct_isotropic_mode"))
        )
        if direct_isotropic_mode not in spec.get("supported_direct_isotropic_modes", (None,)):
            supported_modes = tuple(
                "None" if mode is None else mode
                for mode in spec.get("supported_direct_isotropic_modes", (None,))
            )
            raise BackendOptionError(
                f"Backend {backend_name!r} does not support direct_isotropic_mode "
                f"{direct_isotropic_mode!r}. Supported modes: {', '.join(supported_modes)}."
            )
        normalized_options["direct_isotropic_mode"] = direct_isotropic_mode

    if "energy_progress_bar" in spec["supported_backend_options"]:
        normalized_options["energy_progress_bar"] = normalize_energy_progress_bar_name(
            options.get("energy_progress_bar", spec.get("default_energy_progress_bar", False))
        )

    if "result_residency" in spec["supported_backend_options"]:
        result_residency = normalize_result_residency_name(
            options.get("result_residency", spec.get("default_result_residency", "host"))
        )
        if result_residency not in spec.get("supported_result_residencies", ("host",)):
            raise BackendOptionError(
                f"Backend {backend_name!r} does not support result_residency "
                f"{result_residency!r}. Supported values: "
                f"{', '.join(spec.get('supported_result_residencies', ('host',)))}."
            )
        normalized_options["result_residency"] = result_residency

    if "result_chunk_size" in spec["supported_backend_options"]:
        result_chunk_size = normalize_result_chunk_size(
            options.get("result_chunk_size", spec.get("default_result_chunk_size"))
        )
        if result_chunk_size is None:
            result_chunk_size = int(spec.get("default_result_chunk_size", 1))
        normalized_options["result_chunk_size"] = result_chunk_size

    if "result_layout" in spec["supported_backend_options"]:
        result_layout = normalize_result_layout_name(
            options.get("result_layout", spec.get("default_result_layout", "detector"))
        )
        if result_layout not in spec.get("supported_result_layouts", ("detector",)):
            raise BackendOptionError(
                f"Backend {backend_name!r} does not support result_layout "
                f"{result_layout!r}. Supported values: "
                f"{', '.join(spec.get('supported_result_layouts', ('detector',)))}."
            )
        normalized_options["result_layout"] = result_layout

    if "total_chi_wedge_deg" in spec["supported_backend_options"]:
        normalized_options["total_chi_wedge_deg"] = normalize_total_chi_wedge_deg(
            options.get("total_chi_wedge_deg", spec.get("default_total_chi_wedge_deg", 90.0))
        )

    return normalized_options


def _resolve_precision_contract(
    backend_name: str,
    spec: Mapping[str, Any],
    normalized_options: Mapping[str, Any],
) -> dict[str, str | None]:
    mixed_precision_mode = normalized_options.get("mixed_precision_mode")
    authoritative_dtype = spec["default_dtype"]
    runtime_dtype = spec["default_dtype"]
    if backend_name == "cupy-rsoxs" and mixed_precision_mode == "reduced_morphology_bit_depth":
        authoritative_dtype = "float16"
        runtime_dtype = "float16"

    return {
        "mixed_precision_mode": mixed_precision_mode,
        "authoritative_dtype": authoritative_dtype,
        "runtime_dtype": runtime_dtype,
        "runtime_compute_dtype": spec["runtime_compute_dtype"],
        "runtime_complex_dtype": spec["runtime_complex_dtype"],
    }


def resolve_backend_array_contract(
    backend_name: str,
    backend_options: Mapping[str, Any] | None = None,
    resident_mode: str | None = None,
) -> dict[str, Any]:
    spec = _backend_contract_spec(backend_name)
    normalized_options = normalize_backend_options(backend_name, backend_options)
    precision = _resolve_precision_contract(backend_name, spec, normalized_options)
    normalized_resident_mode = normalize_resident_mode(
        backend_name,
        resident_mode,
    )
    resident_spec = spec["resident_modes"][normalized_resident_mode]
    return {
        "namespace": resident_spec["namespace"],
        "device": resident_spec["device"],
        "resident_mode": normalized_resident_mode,
        "default_resident_mode": spec["default_resident_mode"],
        "supported_resident_modes": spec["supported_resident_modes"],
        "default_dtype": spec["default_dtype"],
        "supported_dtypes": spec["supported_dtypes"],
        "default_execution_path": spec.get("default_execution_path"),
        "supported_execution_paths": spec.get("supported_execution_paths", ()),
        "default_mixed_precision_mode": spec.get("default_mixed_precision_mode"),
        "supported_mixed_precision_modes": spec.get("supported_mixed_precision_modes", (None,)),
        "default_z_collapse_mode": spec.get("default_z_collapse_mode"),
        "supported_z_collapse_modes": spec.get("supported_z_collapse_modes", (None,)),
        "supported_backend_options": spec["supported_backend_options"],
        "dtype": precision["authoritative_dtype"],
        "authoritative_dtype": precision["authoritative_dtype"],
        "runtime_dtype": precision["runtime_dtype"],
        "runtime_compute_dtype": precision["runtime_compute_dtype"],
        "runtime_complex_dtype": precision["runtime_complex_dtype"],
        "mixed_precision_mode": precision["mixed_precision_mode"],
        "z_collapse_mode": normalized_options.get("z_collapse_mode"),
        "options": normalized_options,
    }


def resolve_backend_runtime_contract(
    backend_name: str,
    backend_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec = _backend_contract_spec(backend_name)
    normalized_options = normalize_backend_options(backend_name, backend_options)
    precision = _resolve_precision_contract(backend_name, spec, normalized_options)
    return {
        "namespace": spec["runtime_namespace"],
        "device": spec["runtime_device"],
        "resident_mode": "runtime-compute",
        "default_resident_mode": spec["default_resident_mode"],
        "supported_resident_modes": spec["supported_resident_modes"],
        "default_dtype": spec["default_dtype"],
        "supported_dtypes": spec["supported_dtypes"],
        "default_execution_path": spec.get("default_execution_path"),
        "supported_execution_paths": spec.get("supported_execution_paths", ()),
        "default_mixed_precision_mode": spec.get("default_mixed_precision_mode"),
        "supported_mixed_precision_modes": spec.get("supported_mixed_precision_modes", (None,)),
        "default_z_collapse_mode": spec.get("default_z_collapse_mode"),
        "supported_z_collapse_modes": spec.get("supported_z_collapse_modes", (None,)),
        "supported_backend_options": spec["supported_backend_options"],
        "dtype": precision["runtime_dtype"],
        "authoritative_dtype": precision["authoritative_dtype"],
        "runtime_dtype": precision["runtime_dtype"],
        "runtime_compute_dtype": precision["runtime_compute_dtype"],
        "runtime_complex_dtype": precision["runtime_complex_dtype"],
        "mixed_precision_mode": precision["mixed_precision_mode"],
        "z_collapse_mode": normalized_options.get("z_collapse_mode"),
        "options": normalized_options,
    }
