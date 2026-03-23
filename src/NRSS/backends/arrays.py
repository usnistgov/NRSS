from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from typing import Any

import numpy as np

from .contracts import (
    resolve_backend_array_contract,
    resolve_backend_runtime_contract,
)


try:
    import cupy as cp
except ImportError:  # pragma: no cover - exercised only when CuPy is absent
    cp = None


@dataclass(frozen=True)
class ArrayPlan:
    material_id: int | None
    field_name: str
    supported: bool
    original_namespace: str
    target_namespace: str
    original_device: str
    target_device: str
    original_dtype: str
    target_dtype: str
    shape: tuple[int, ...] | None
    requires_dtype_cast: bool
    requires_layout_copy: bool
    transfer: str
    reason: str


def inspect_array(arr: Any) -> dict[str, Any]:
    if arr is None:
        return {
            "recognized": True,
            "namespace": "missing",
            "device": "unknown",
            "dtype": "unknown",
            "shape": None,
            "c_contiguous": True,
        }

    if isinstance(arr, np.ndarray):
        return {
            "recognized": True,
            "namespace": "numpy",
            "device": "cpu",
            "dtype": str(arr.dtype),
            "shape": tuple(arr.shape),
            "c_contiguous": bool(arr.flags.c_contiguous),
        }

    if cp is not None and isinstance(arr, cp.ndarray):
        return {
            "recognized": True,
            "namespace": "cupy",
            "device": "gpu",
            "dtype": str(arr.dtype),
            "shape": tuple(arr.shape),
            "c_contiguous": bool(arr.flags.c_contiguous),
        }

    return {
        "recognized": False,
        "namespace": type(arr).__module__,
        "device": "unknown",
        "dtype": getattr(getattr(arr, "dtype", None), "name", "unknown"),
        "shape": tuple(getattr(arr, "shape", ())) or None,
        "c_contiguous": False,
    }


def get_namespace_module(namespace: str):
    if namespace == "numpy":
        return np
    if namespace == "cupy":
        if cp is None:  # pragma: no cover - exercised only when CuPy is absent
            raise RuntimeError("CuPy is not importable.")
        return cp
    raise ValueError(f"Unsupported array namespace {namespace!r}.")


def assess_array_for_backend(
    arr: Any,
    backend_name: str,
    field_name: str,
    material_id: int | None = None,
    backend_options: Mapping[str, Any] | None = None,
    resident_mode: str | None = None,
    contract: Mapping[str, Any] | None = None,
) -> ArrayPlan:
    info = inspect_array(arr)
    if contract is None:
        contract = resolve_backend_array_contract(
            backend_name,
            backend_options,
            resident_mode=resident_mode,
        )
    return _assess_array_against_contract(
        arr=arr,
        field_name=field_name,
        material_id=material_id,
        contract=contract,
        info=info,
    )


def assess_array_for_backend_runtime(
    arr: Any,
    backend_name: str,
    field_name: str,
    material_id: int | None = None,
    backend_options: Mapping[str, Any] | None = None,
    contract: Mapping[str, Any] | None = None,
) -> ArrayPlan:
    info = inspect_array(arr)
    if contract is None:
        contract = resolve_backend_runtime_contract(backend_name, backend_options)
    return _assess_array_against_contract(
        arr=arr,
        field_name=field_name,
        material_id=material_id,
        contract=contract,
        info=info,
    )


def _assess_array_against_contract(
    *,
    arr: Any,
    field_name: str,
    material_id: int | None,
    contract: Mapping[str, Any],
    info: Mapping[str, Any],
) -> ArrayPlan:
    target_namespace = contract["namespace"]
    target_device = contract["device"]
    target_dtype = contract["dtype"]

    if info["namespace"] == "missing":
        return ArrayPlan(
            material_id=material_id,
            field_name=field_name,
            supported=True,
            original_namespace="missing",
            target_namespace=target_namespace,
            original_device="unknown",
            target_device=target_device,
            original_dtype="unknown",
            target_dtype=target_dtype,
            shape=None,
            requires_dtype_cast=False,
            requires_layout_copy=False,
            transfer="none",
            reason="Field is not populated.",
        )

    if not info["recognized"]:
        return ArrayPlan(
            material_id=material_id,
            field_name=field_name,
            supported=False,
            original_namespace=info["namespace"],
            target_namespace=target_namespace,
            original_device=info["device"],
            target_device=target_device,
            original_dtype=str(info["dtype"]),
            target_dtype=target_dtype,
            shape=info["shape"],
            requires_dtype_cast=False,
            requires_layout_copy=False,
            transfer="unsupported",
            reason=(
                f"Unsupported array type for material {material_id} field {field_name}: "
                f"{type(arr)!r}. Supported namespaces are numpy and cupy."
            ),
        )

    if target_namespace == "cupy" and cp is None:
        return ArrayPlan(
            material_id=material_id,
            field_name=field_name,
            supported=False,
            original_namespace=info["namespace"],
            target_namespace=target_namespace,
            original_device=info["device"],
            target_device=target_device,
            original_dtype=str(info["dtype"]),
            target_dtype=target_dtype,
            shape=info["shape"],
            requires_dtype_cast=False,
            requires_layout_copy=False,
            transfer="unsupported",
            reason="CuPy is not importable.",
        )

    transfer = "none"
    if info["namespace"] == "numpy" and target_namespace == "cupy":
        transfer = "host_to_device"
    elif info["namespace"] == "cupy" and target_namespace == "numpy":
        transfer = "device_to_host"

    return ArrayPlan(
        material_id=material_id,
        field_name=field_name,
        supported=True,
        original_namespace=info["namespace"],
        target_namespace=target_namespace,
        original_device=info["device"],
        target_device=target_device,
        original_dtype=str(info["dtype"]),
        target_dtype=target_dtype,
        shape=info["shape"],
        requires_dtype_cast=str(info["dtype"]) != target_dtype,
        requires_layout_copy=not bool(info["c_contiguous"]),
        transfer=transfer,
        reason=(
            "No conversion required."
            if transfer == "none" and str(info["dtype"]) == target_dtype and bool(info["c_contiguous"])
            else "Backend coercion required."
        ),
    )


def coerce_array_for_backend(arr: Any, plan: ArrayPlan):
    if arr is None:
        return None
    if not plan.supported:
        raise TypeError(plan.reason)
    if (
        plan.transfer == "none"
        and not plan.requires_dtype_cast
        and not plan.requires_layout_copy
    ):
        return arr

    if plan.target_namespace == "numpy":
        np_dtype = np.dtype(plan.target_dtype)
        if plan.original_namespace == "numpy":
            out = np.asarray(arr, dtype=np_dtype)
            return np.ascontiguousarray(out)
        if plan.original_namespace == "cupy":
            host = cp.asnumpy(arr)
            host = np.asarray(host, dtype=np_dtype)
            return np.ascontiguousarray(host)

    if plan.target_namespace == "cupy":
        if cp is None:  # pragma: no cover - gated by plan.supported
            raise RuntimeError("CuPy is not importable.")
        cp_dtype = cp.dtype(plan.target_dtype)
        if plan.original_namespace == "numpy":
            np_dtype = np.dtype(plan.target_dtype)
            host = np.asarray(arr, dtype=np_dtype)
            host = np.ascontiguousarray(host)
            out = cp.empty(host.shape, dtype=cp_dtype)
            out.set(host)
            return out
        if plan.original_namespace == "cupy":
            out = cp.asarray(arr, dtype=cp_dtype)
            return cp.ascontiguousarray(out)

    raise TypeError(plan.reason)


def to_python_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)
