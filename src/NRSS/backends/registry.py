from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import os

from .contracts import resolve_backend_array_contract


KNOWN_BACKEND_ORDER = ("cupy-rsoxs", "cyrsoxs")
DEFAULT_BACKEND_ORDER = ("cupy-rsoxs", "cyrsoxs")


class BackendSelectionError(RuntimeError):
    """Base error for backend resolution and availability problems."""


class UnknownBackendError(BackendSelectionError):
    """Raised when a backend id is not recognized by NRSS."""


class BackendUnavailableError(BackendSelectionError):
    """Raised when a known backend is not currently runnable."""


@dataclass(frozen=True)
class BackendInfo:
    name: str
    available: bool
    implemented: bool
    import_target: str | None
    reason: str | None
    supports_cli: bool
    supports_reference_parity: bool
    supports_device_input: bool
    supports_backend_native_output: bool
    default_resident_mode: str
    supported_resident_modes: tuple[str, ...]
    default_dtype: str
    supported_dtypes: tuple[str, ...]
    supported_backend_options: tuple[str, ...]
    description: str


@lru_cache(maxsize=None)
def _find_import_target(*candidates: str) -> str | None:
    for candidate in candidates:
        try:
            if importlib.util.find_spec(candidate) is not None:
                return candidate
        except (ImportError, ValueError, AttributeError):
            continue
    return None


def _detect_cyrsoxs() -> BackendInfo:
    contract = resolve_backend_array_contract("cyrsoxs")
    import_target = _find_import_target("CyRSoXS", "cyrsoxs")
    if import_target is None:
        return BackendInfo(
            name="cyrsoxs",
            available=False,
            implemented=True,
            import_target=None,
            reason="CyRSoXS Python bindings are not importable.",
            supports_cli=True,
            supports_reference_parity=True,
            supports_device_input=False,
            supports_backend_native_output=False,
            default_resident_mode=contract["default_resident_mode"],
            supported_resident_modes=contract["supported_resident_modes"],
            default_dtype=contract["default_dtype"],
            supported_dtypes=contract["supported_dtypes"],
            supported_backend_options=contract["supported_backend_options"],
            description="Legacy CyRSoXS backend accessed through Python bindings.",
        )

    return BackendInfo(
        name="cyrsoxs",
        available=True,
        implemented=True,
        import_target=import_target,
        reason=None,
        supports_cli=True,
        supports_reference_parity=True,
        supports_device_input=False,
        supports_backend_native_output=False,
        default_resident_mode=contract["default_resident_mode"],
        supported_resident_modes=contract["supported_resident_modes"],
        default_dtype=contract["default_dtype"],
        supported_dtypes=contract["supported_dtypes"],
        supported_backend_options=contract["supported_backend_options"],
        description="Legacy CyRSoXS backend accessed through Python bindings.",
    )


def _detect_cupy_rsoxs() -> BackendInfo:
    contract = resolve_backend_array_contract("cupy-rsoxs")
    backend_spec = _find_import_target("NRSS.backends.cupy_rsoxs")
    if backend_spec is None:
        return BackendInfo(
            name="cupy-rsoxs",
            available=False,
            implemented=False,
            import_target=None,
            reason="CuPy backend module is not importable from this NRSS installation.",
            supports_cli=False,
            supports_reference_parity=True,
            supports_device_input=True,
            supports_backend_native_output=True,
            default_resident_mode=contract["default_resident_mode"],
            supported_resident_modes=contract["supported_resident_modes"],
            default_dtype=contract["default_dtype"],
            supported_dtypes=contract["supported_dtypes"],
            supported_backend_options=contract["supported_backend_options"],
            description="Pure-Python CuPy-native NRSS backend.",
        )

    cupy_spec = _find_import_target("cupy")
    if cupy_spec is None:
        return BackendInfo(
            name="cupy-rsoxs",
            available=False,
            implemented=True,
            import_target=backend_spec,
            reason="CuPy is not importable.",
            supports_cli=False,
            supports_reference_parity=True,
            supports_device_input=True,
            supports_backend_native_output=True,
            default_resident_mode=contract["default_resident_mode"],
            supported_resident_modes=contract["supported_resident_modes"],
            default_dtype=contract["default_dtype"],
            supported_dtypes=contract["supported_dtypes"],
            supported_backend_options=contract["supported_backend_options"],
            description="Pure-Python CuPy-native NRSS backend.",
        )

    return BackendInfo(
        name="cupy-rsoxs",
        available=True,
        implemented=True,
        import_target=backend_spec,
        reason=None,
        supports_cli=False,
        supports_reference_parity=True,
        supports_device_input=True,
        supports_backend_native_output=True,
        default_resident_mode=contract["default_resident_mode"],
        supported_resident_modes=contract["supported_resident_modes"],
        default_dtype=contract["default_dtype"],
        supported_dtypes=contract["supported_dtypes"],
        supported_backend_options=contract["supported_backend_options"],
        description="Pure-Python CuPy-native NRSS backend.",
    )


@lru_cache(maxsize=None)
def get_backend_info(name: str) -> BackendInfo:
    if name == "cyrsoxs":
        return _detect_cyrsoxs()
    if name == "cupy-rsoxs":
        return _detect_cupy_rsoxs()
    raise UnknownBackendError(
        f"Unknown NRSS backend {name!r}. Known backends: {', '.join(KNOWN_BACKEND_ORDER)}."
    )


def known_backends() -> tuple[str, ...]:
    return KNOWN_BACKEND_ORDER


def available_backends(include_unavailable: bool = True) -> tuple[BackendInfo, ...]:
    infos = tuple(get_backend_info(name) for name in KNOWN_BACKEND_ORDER)
    if include_unavailable:
        return infos
    return tuple(info for info in infos if info.available)


def format_backend_availability() -> str:
    rows = []
    for info in available_backends(include_unavailable=True):
        status = "available" if info.available else "unavailable"
        reason = "" if info.reason is None else f" ({info.reason})"
        rows.append(f"- {info.name}: {status}{reason}")
    return "\n".join(rows)


def resolve_backend_name(preferred: str | None = None) -> str:
    if preferred is None:
        env_backend = os.environ.get("NRSS_BACKEND")
        if env_backend is not None and env_backend.strip():
            preferred = env_backend.strip()

    if preferred is not None and preferred.strip():
        candidate = preferred.strip()
        if candidate == "default":
            preferred = None
        else:
            info = get_backend_info(candidate)
            if not info.available:
                raise BackendUnavailableError(
                    f"Requested NRSS backend {candidate!r} is unavailable.\n"
                    f"{format_backend_availability()}"
                )
            return candidate

    for candidate in DEFAULT_BACKEND_ORDER:
        info = get_backend_info(candidate)
        if info.available:
            return candidate

    raise BackendUnavailableError(
        "No runnable NRSS backend is currently available.\n"
        f"{format_backend_availability()}"
    )
