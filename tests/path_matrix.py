from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComputationPath:
    id: str
    backend: str
    backend_options: dict[str, object]
    category: str
    supports_cli: bool
    supports_reference_parity: bool
    resident_mode: str | None = None
    field_namespace: str = "numpy"
    ownership_policy: str | None = None


PEER_PATHS: tuple[ComputationPath, ...] = (
    ComputationPath(
        id="legacy_cyrsoxs",
        backend="cyrsoxs",
        backend_options={},
        category="legacy",
        supports_cli=True,
        supports_reference_parity=True,
        resident_mode=None,
        field_namespace="numpy",
        ownership_policy=None,
    ),
    ComputationPath(
        id="cupy_tensor_coeff",
        backend="cupy-rsoxs",
        backend_options={"execution_path": "tensor_coeff"},
        category="cupy",
        supports_cli=False,
        supports_reference_parity=True,
        resident_mode="device",
        field_namespace="cupy",
        ownership_policy="borrow",
    ),
    ComputationPath(
        id="cupy_direct_polarization",
        backend="cupy-rsoxs",
        backend_options={"execution_path": "direct_polarization"},
        category="cupy",
        supports_cli=False,
        supports_reference_parity=True,
        resident_mode="device",
        field_namespace="cupy",
        ownership_policy="borrow",
    ),
)

PATHS_BY_ID = {path.id: path for path in PEER_PATHS}


def get_computation_path(path_id: str) -> ComputationPath:
    try:
        return PATHS_BY_ID[path_id]
    except KeyError as exc:
        known = ", ".join(PATHS_BY_ID)
        raise ValueError(f"Unknown NRSS computation path {path_id!r}. Known paths: {known}.") from exc


def peer_paths_for_backend(backend: str) -> tuple[ComputationPath, ...]:
    return tuple(path for path in PEER_PATHS if path.backend == backend)


def default_path_for_backend(backend: str) -> ComputationPath:
    matches = peer_paths_for_backend(backend)
    if not matches:
        raise ValueError(f"No maintained computation path maps to backend {backend!r}.")
    return matches[0]
