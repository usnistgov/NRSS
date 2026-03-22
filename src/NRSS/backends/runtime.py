from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .registry import BackendUnavailableError


if TYPE_CHECKING:
    from NRSS.morphology import Morphology


class BackendRuntime(ABC):
    """Internal runtime interface for backend-dispatched Morphology operations."""

    name: str

    @abstractmethod
    def prepare(self, morphology: "Morphology") -> None:
        """Prepare backend-owned runtime objects for the morphology."""

    @abstractmethod
    def run(
        self,
        morphology: "Morphology",
        *,
        stdout: bool = True,
        stderr: bool = True,
        return_xarray: bool = True,
        print_vec_info: bool = False,
        validate: bool = False,
    ) -> Any:
        """Execute the backend runtime for the morphology."""

    @abstractmethod
    def validate_all(self, morphology: "Morphology", *, quiet: bool = True) -> None:
        """Run backend-aware validation for the morphology."""

    def release(self, morphology: "Morphology") -> None:
        """Release backend-owned runtime state for the morphology."""
        return None

def get_backend_runtime(backend_name: str) -> BackendRuntime:
    if backend_name == "cyrsoxs":
        from .cyrsoxs import CyrsoxsBackendRuntime

        return CyrsoxsBackendRuntime()

    if backend_name == "cupy-rsoxs":
        from .cupy_rsoxs import CupyRsoxsBackendRuntime

        return CupyRsoxsBackendRuntime()

    raise BackendUnavailableError(
        f"Backend {backend_name!r} does not implement an NRSS runtime adapter yet."
    )
