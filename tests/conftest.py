import os
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from NRSS.backends import (
    BackendUnavailableError,
    format_backend_availability,
    get_backend_info,
    resolve_backend_name,
)


# Default test execution to a single visible GPU for reproducibility and to avoid
# known multi-GPU CyRSoXS instability during energy fan-out. Respect any explicit
# user or CI pinning that is already present in the environment.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def pytest_addoption(parser):
    parser.addoption(
        "--nrss-backend",
        action="store",
        default=None,
        help="Select the NRSS backend for backend-aware tests.",
    )


def _determine_selected_backend(config):
    requested = config.getoption("--nrss-backend")
    if requested is None:
        requested = os.environ.get("NRSS_BACKEND")

    if requested is not None and requested.strip():
        try:
            selected = resolve_backend_name(requested)
        except BackendUnavailableError as exc:
            raise pytest.UsageError(str(exc)) from exc
        return selected, True

    try:
        selected = resolve_backend_name(None)
    except BackendUnavailableError:
        return None, False
    return selected, False


def pytest_configure(config):
    selected_backend, explicit = _determine_selected_backend(config)
    backend_info = get_backend_info(selected_backend) if selected_backend is not None else None

    config._nrss_backend = selected_backend
    config._nrss_backend_explicit = explicit
    config._nrss_backend_info = backend_info

    if selected_backend is not None:
        os.environ["NRSS_BACKEND"] = selected_backend
    else:
        os.environ.pop("NRSS_BACKEND", None)


def pytest_report_header(config):
    selected_backend = getattr(config, "_nrss_backend", None)
    if selected_backend is None:
        return "NRSS backend: none available\n" + format_backend_availability()
    return f"NRSS backend: {selected_backend}"


def pytest_collection_modifyitems(config, items):
    selected_backend = getattr(config, "_nrss_backend", None)
    backend_info = getattr(config, "_nrss_backend_info", None)

    skip_no_backend = pytest.mark.skip(reason="No runnable NRSS backend is available.")
    skip_cyrsoxs_only = pytest.mark.skip(
        reason=(
            f"Test requires backend 'cyrsoxs'; selected backend is "
            f"{selected_backend!r}."
        )
    )
    skip_reference_parity = pytest.mark.skip(
        reason=(
            f"Selected backend {selected_backend!r} does not advertise "
            "reference-parity support."
        )
    )

    for item in items:
        if selected_backend is None:
            if (
                item.get_closest_marker("backend_specific")
                or item.get_closest_marker("cyrsoxs_only")
                or item.get_closest_marker("reference_parity")
                or item.get_closest_marker("physics_validation")
                or item.get_closest_marker("toolchain_validation")
                or item.get_closest_marker("gpu")
            ):
                item.add_marker(skip_no_backend)
            continue

        if item.get_closest_marker("cyrsoxs_only") and selected_backend != "cyrsoxs":
            item.add_marker(skip_cyrsoxs_only)
            continue

        if item.get_closest_marker("reference_parity"):
            if backend_info is None or not backend_info.supports_reference_parity:
                item.add_marker(skip_reference_parity)


@pytest.fixture(scope="session")
def nrss_backend(pytestconfig):
    return getattr(pytestconfig, "_nrss_backend", None)


@pytest.fixture(scope="session")
def nrss_backend_info(pytestconfig):
    return getattr(pytestconfig, "_nrss_backend_info", None)
