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
        "supported_dtypes": ("float32",),
        "default_execution_path": "tensor_coeff",
        "supported_execution_paths": (
            "tensor_coeff",
            "direct_polarization",
        ),
        "supported_backend_options": ("dtype", "execution_path"),
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

    unknown = sorted(set(options) - set(spec["supported_backend_options"]))
    if unknown:
        raise BackendOptionError(
            f"Unsupported backend option(s) for {backend_name!r}: {', '.join(unknown)}. "
            f"Supported options are: {', '.join(spec['supported_backend_options'])}."
        )

    normalized_dtype = normalize_dtype_name(options.get("dtype", spec["default_dtype"]))
    if normalized_dtype not in spec["supported_dtypes"]:
        raise BackendOptionError(
            f"Backend {backend_name!r} does not support dtype {normalized_dtype!r}. "
            f"Supported dtypes: {', '.join(spec['supported_dtypes'])}."
        )

    normalized_options = {"dtype": normalized_dtype}
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

    return normalized_options


def resolve_backend_array_contract(
    backend_name: str,
    backend_options: Mapping[str, Any] | None = None,
    resident_mode: str | None = None,
) -> dict[str, Any]:
    spec = _backend_contract_spec(backend_name)
    normalized_options = normalize_backend_options(backend_name, backend_options)
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
        "supported_backend_options": spec["supported_backend_options"],
        "dtype": normalized_options["dtype"],
        "options": normalized_options,
    }


def resolve_backend_runtime_contract(
    backend_name: str,
    backend_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec = _backend_contract_spec(backend_name)
    normalized_options = normalize_backend_options(backend_name, backend_options)
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
        "supported_backend_options": spec["supported_backend_options"],
        "dtype": normalized_options["dtype"],
        "options": normalized_options,
    }
