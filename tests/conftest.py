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
from tests.path_matrix import (
    PEER_PATHS,
    ComputationPath,
    default_path_for_backend,
    get_computation_path,
    peer_paths_for_backend,
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
    parser.addoption(
        "--nrss-path",
        action="store",
        default=None,
        help="Select a maintained NRSS computation path for path-aware tests.",
    )


def _explicit_backend_request(config) -> str | None:
    requested = config.getoption("--nrss-backend")
    if requested is None:
        requested = os.environ.get("NRSS_BACKEND")
    if requested is None or not requested.strip():
        return None
    try:
        return resolve_backend_name(requested)
    except BackendUnavailableError as exc:
        raise pytest.UsageError(str(exc)) from exc


def _explicit_path_request(config) -> ComputationPath | None:
    requested = config.getoption("--nrss-path")
    if requested is None:
        requested = os.environ.get("NRSS_PATH")
    if requested is None or not requested.strip():
        return None
    try:
        path = get_computation_path(requested.strip())
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc
    info = get_backend_info(path.backend)
    if not info.available:
        raise pytest.UsageError(
            f"Requested NRSS computation path {path.id!r} is unavailable because backend "
            f"{path.backend!r} is unavailable.\n{format_backend_availability()}"
        )
    return path


def _determine_default_backend() -> tuple[str | None, bool]:
    try:
        return resolve_backend_name(None), False
    except BackendUnavailableError:
        return None, False


def _resolve_path_matrix_selection(config) -> tuple[ComputationPath | None, str | None, str | None]:
    explicit_path = _explicit_path_request(config)
    explicit_backend = _explicit_backend_request(config)

    if explicit_path is not None and explicit_backend is not None and explicit_path.backend != explicit_backend:
        raise pytest.UsageError(
            "NRSS selector conflict: "
            f"--nrss-path/NRSS_PATH resolved to backend {explicit_path.backend!r}, "
            f"but --nrss-backend/NRSS_BACKEND resolved to {explicit_backend!r}."
        )

    selected_backend = explicit_backend
    if explicit_path is not None:
        selected_backend = explicit_path.backend
        return explicit_path, selected_backend, "explicit_path"

    if explicit_backend is not None:
        return None, explicit_backend, "explicit_backend"

    default_backend, _ = _determine_default_backend()
    return None, default_backend, "default"


def _matrix_paths_for_config(config) -> tuple[ComputationPath, ...]:
    explicit_path = getattr(config, "_nrss_explicit_path", None)
    selected_backend = getattr(config, "_nrss_backend", None)
    selection_mode = getattr(config, "_nrss_selection_mode", "default")

    if explicit_path is not None:
        return (explicit_path,)
    if selection_mode == "explicit_backend" and selected_backend is not None:
        return peer_paths_for_backend(selected_backend)
    return PEER_PATHS


def _paths_from_subset_marker(marker) -> tuple[ComputationPath, ...]:
    if marker is None:
        return ()
    paths: list[ComputationPath] = []
    for arg in marker.args:
        if not isinstance(arg, str):
            raise pytest.UsageError(
                "pytest.mark.path_subset expects one or more computation-path id strings."
            )
        paths.append(get_computation_path(arg))
    return tuple(paths)


def _parametrize_nrss_path(metafunc, paths: tuple[ComputationPath, ...]) -> None:
    if not paths:
        metafunc.parametrize(
            "nrss_path",
            [
                pytest.param(
                    None,
                    id="no_matching_path",
                    marks=pytest.mark.skip(
                        reason="Test does not apply to the selected NRSS computation path."
                    ),
                )
            ],
        )
        return

    params = []
    for path in paths:
        info = get_backend_info(path.backend)
        if info.available:
            params.append(pytest.param(path, id=path.id))
        else:
            params.append(
                pytest.param(
                    path,
                    id=path.id,
                    marks=pytest.mark.skip(
                        reason=(
                            f"NRSS computation path {path.id!r} is unavailable because backend "
                            f"{path.backend!r} is unavailable."
                        )
                    ),
                )
            )
    metafunc.parametrize("nrss_path", params)


def pytest_generate_tests(metafunc):
    if "nrss_path" not in metafunc.fixturenames:
        return

    subset_marker = metafunc.definition.get_closest_marker("path_subset")
    matrix_marker = metafunc.definition.get_closest_marker("path_matrix")

    if subset_marker is not None:
        paths = _paths_from_subset_marker(subset_marker)
        if not paths:
            raise pytest.UsageError("pytest.mark.path_subset requires at least one path id.")
        explicit_path = getattr(metafunc.config, "_nrss_explicit_path", None)
        if explicit_path is not None:
            paths = tuple(path for path in paths if path.id == explicit_path.id)
        _parametrize_nrss_path(metafunc, paths)
        return

    if matrix_marker is not None:
        _parametrize_nrss_path(metafunc, _matrix_paths_for_config(metafunc.config))
        return


def pytest_configure(config):
    explicit_path, selected_backend, selection_mode = _resolve_path_matrix_selection(config)
    backend_info = get_backend_info(selected_backend) if selected_backend is not None else None

    config._nrss_explicit_path = explicit_path
    config._nrss_backend = selected_backend
    config._nrss_backend_explicit = selection_mode in {"explicit_path", "explicit_backend"}
    config._nrss_backend_info = backend_info
    config._nrss_selection_mode = selection_mode

    config.addinivalue_line(
        "markers",
        "path_matrix: maintained peer-path test that expands across the selected computation paths",
    )
    config.addinivalue_line(
        "markers",
        "path_subset(*path_ids): maintained path-aware test limited to the named computation paths",
    )

    if explicit_path is not None:
        os.environ["NRSS_PATH"] = explicit_path.id
        os.environ["NRSS_BACKEND"] = explicit_path.backend
    elif selection_mode == "explicit_backend" and selected_backend is not None:
        os.environ.pop("NRSS_PATH", None)
        os.environ["NRSS_BACKEND"] = selected_backend
    else:
        os.environ.pop("NRSS_PATH", None)
        os.environ.pop("NRSS_BACKEND", None)


def pytest_report_header(config):
    selected_backend = getattr(config, "_nrss_backend", None)
    explicit_path = getattr(config, "_nrss_explicit_path", None)
    selection_mode = getattr(config, "_nrss_selection_mode", "default")

    if selected_backend is None:
        return "NRSS backend: none available\n" + format_backend_availability()

    lines = [f"NRSS backend: {selected_backend}"]
    if explicit_path is not None:
        lines.append(
            "NRSS path selection: "
            f"{explicit_path.id} (backend={explicit_path.backend}, backend_options={explicit_path.backend_options})"
        )
    elif selection_mode == "explicit_backend":
        compatible = ", ".join(path.id for path in peer_paths_for_backend(selected_backend))
        lines.append(
            "NRSS path selection: "
            f"compatible peer paths for backend {selected_backend} -> {compatible}"
        )
    else:
        peer_ids = ", ".join(path.id for path in PEER_PATHS)
        lines.append(f"NRSS path selection: all peer paths by default -> {peer_ids}")
    return "\n".join(lines)


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
                or item.get_closest_marker("path_matrix")
                or item.get_closest_marker("path_subset")
            ):
                item.add_marker(skip_no_backend)
            continue

        if item.get_closest_marker("cyrsoxs_only") and selected_backend != "cyrsoxs":
            item.add_marker(skip_cyrsoxs_only)
            continue

        if item.get_closest_marker("reference_parity"):
            if backend_info is None or not backend_info.supports_reference_parity:
                item.add_marker(skip_reference_parity)


@pytest.fixture
def nrss_path(request, pytestconfig):
    if hasattr(request, "param"):
        return request.param

    explicit_path = getattr(pytestconfig, "_nrss_explicit_path", None)
    if explicit_path is not None:
        return explicit_path

    backend = getattr(pytestconfig, "_nrss_backend", None)
    if backend is None:
        pytest.skip("No runnable NRSS backend is available.")
    return default_path_for_backend(backend)


@pytest.fixture
def nrss_backend(nrss_path):
    return nrss_path.backend


@pytest.fixture
def nrss_backend_options(nrss_path):
    return dict(nrss_path.backend_options)


@pytest.fixture
def nrss_backend_info(nrss_path):
    return get_backend_info(nrss_path.backend)
