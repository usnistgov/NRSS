from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np


class BackendOptionError(ValueError):
    """Raised when backend options are invalid for the selected backend."""


_BACKEND_ARRAY_CONTRACTS = {
    "cyrsoxs": {
        "namespace": "numpy",
        "device": "cpu",
        "default_dtype": "float32",
        "supported_dtypes": ("float32",),
        "supported_backend_options": ("dtype",),
    },
    "cupy-rsoxs": {
        "namespace": "cupy",
        "device": "gpu",
        "default_dtype": "float32",
        "supported_dtypes": ("float16", "float32"),
        "supported_backend_options": ("dtype",),
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

    return {"dtype": normalized_dtype}


def resolve_backend_array_contract(
    backend_name: str,
    backend_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec = deepcopy(_backend_contract_spec(backend_name))
    normalized_options = normalize_backend_options(backend_name, backend_options)
    spec["dtype"] = normalized_options["dtype"]
    spec["options"] = normalized_options
    return spec
