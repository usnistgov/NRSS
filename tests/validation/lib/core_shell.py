from __future__ import annotations

import gc
import importlib
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from NRSS.morphology import Material, Morphology, OpticalConstants


DATA_DIR = REPO_ROOT / "tests" / "validation" / "data" / "core_shell"
EXPERIMENTAL_REFERENCE_PATH = DATA_DIR / "CS_reference.h5"
SIM_REFERENCE_PATH = DATA_DIR / "CS_sim_reference.h5"

SHAPE = (32, 512, 512)
PHYS_SIZE_NM = 2.5
CORE_RADIUS_VOX = 4.0
SHELL_THICKNESS_VOX = 2.94
PHI_ISO = 0.46
DECAY_ORDER = 0.42
CENTER_Z_VOX = 15.0
AWEDGE_HALF_WIDTH_DEG = 10.0
HORIZONTAL_CHI_CENTER_DEG = 180.0
VERTICAL_CHI_CENTER_DEG = 90.0
Q_COMPARE_MIN = 0.02
Q_COMPARE_MAX = 0.4
ENERGY_CHECKS_EV = (284.7, 285.2)

EXPERIMENTAL_REFERENCE_LABEL = "Experimental reference"
SIM_REFERENCE_LABEL = "Sim-derived regression golden"
EXPERIMENTAL_REFERENCE_CITATION = (
    "Subhrangsu Mukherjee, Jason K. Streit, Eliot Gann, Kumar Saurabh, "
    "Daniel F. Sunday, Adarsh Krishnamurthy, Baskar Ganapathysubramanian, "
    "Lee J. Richter, Richard A. Vaia, and Dean M. DeLongchamp, "
    "\"Polarized X-ray scattering measures molecular orientation in "
    "polymer-grafted nanoparticles,\" Nature Communications 12, 4896 "
    "(2021), doi:10.1038/s41467-021-25176-4."
)

EXPERIMENTAL_THRESHOLDS = {
    "a_vs_energy": {"max_abs_diff": 0.03, "rmse": 0.02},
    "a_vs_q_284p7": {"max_abs_diff": 0.05, "rmse": 0.02},
    "a_vs_q_285p2": {"max_abs_diff": 0.055, "rmse": 0.03},
}

SIM_THRESHOLDS = {
    "a_vs_energy": {"max_abs_diff": 0.0025, "rmse": 0.0010},
    "a_vs_q_284p7": {"max_abs_diff": 0.0080, "rmse": 0.003},
    "a_vs_q_285p2": {"max_abs_diff": 0.0080, "rmse": 0.003},
}


@dataclass(frozen=True)
class CoreShellScenario:
    scenario_id: str
    display_name: str
    shell_s_mode: str
    shell_s_scale: float
    shell_orientation_mode: str
    description: str


BASELINE_SCENARIO = CoreShellScenario(
    scenario_id="baseline",
    display_name="Baseline legacy-radial shell",
    shell_s_mode="legacy_decay",
    shell_s_scale=1.0,
    shell_orientation_mode="radial",
    description="Legacy radial shell orientation with the original decaying aligned fraction.",
)


@lru_cache(maxsize=1)
def has_visible_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0 and "GPU " in result.stdout


def release_runtime_memory() -> None:
    gc.collect()
    try:
        import cupy as cp
    except ImportError:
        return

    for action in (
        lambda: cp.cuda.runtime.deviceSynchronize(),
        lambda: cp.get_default_memory_pool().free_all_blocks(),
        lambda: cp.get_default_pinned_memory_pool().free_all_blocks(),
    ):
        try:
            action()
        except Exception:
            pass


def _load_reference_awedge(path: Path, name: str) -> xr.DataArray:
    with h5py.File(path, "r") as f:
        data = np.asarray(f["A"][()], dtype=np.float64)
        energy = np.asarray(f["energy"][()], dtype=np.float64)
        q = np.asarray(f["q"][()], dtype=np.float64)
    return xr.DataArray(
        data,
        dims=("energy", "q"),
        coords={"energy": energy, "q": q},
        name=name,
    ).sortby("energy")


def load_experimental_reference_awedge() -> xr.DataArray:
    return _load_reference_awedge(EXPERIMENTAL_REFERENCE_PATH, name="A_experimental_reference")


def load_sim_reference_awedge() -> xr.DataArray:
    return _load_reference_awedge(SIM_REFERENCE_PATH, name="A_sim_reference")


def write_awedge_reference(
    awedge: xr.DataArray,
    path: Path,
    source_kind: str,
    description: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("A", data=np.asarray(awedge.values, dtype=np.float64))
        f.create_dataset("energy", data=np.asarray(awedge.coords["energy"].values, dtype=np.float64))
        f.create_dataset("q", data=np.asarray(awedge.coords["q"].values, dtype=np.float64))
        f.attrs["source_kind"] = source_kind
        f.attrs["description"] = description
        f.attrs["shape"] = str(SHAPE)
        f.attrs["phys_size_nm"] = PHYS_SIZE_NM
        f.attrs["eangle_rotation"] = str([0.0, 1.0, 360.0])


def _load_optical_constants() -> dict[int, OpticalConstants]:
    constants = {
        1: OpticalConstants.load_matfile(str(DATA_DIR / "Material1.txt"), name="core"),
        2: OpticalConstants.load_matfile(str(DATA_DIR / "Material2.txt"), name="shell"),
        3: OpticalConstants.load_matfile(str(DATA_DIR / "Material3.txt"), name="matrix"),
    }
    energies = tuple(constants[1].energies)
    for material_id, opt in constants.items():
        if tuple(opt.energies) != energies:
            raise AssertionError(f"Material {material_id} energies differ from Material 1.")
    return constants


def _local_bounds(center: float, radius: float, size: int) -> tuple[int, int]:
    start = max(0, int(np.floor(center - radius)))
    stop = min(size, int(np.ceil(center + radius)) + 1)
    return start, stop


def _shell_orientation_angles(
    shell_mask: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    orientation_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if orientation_mode == "radial":
        ox = dx
        oy = dy
        oz = dz
    elif orientation_mode == "tangential_latitude":
        ox = -dy
        oy = dx
        oz = np.zeros_like(dz, dtype=np.float32)
    else:
        raise AssertionError(f"Unsupported orientation mode: {orientation_mode}")

    ox = np.where(shell_mask, ox, 0.0).astype(np.float32, copy=False)
    oy = np.where(shell_mask, oy, 0.0).astype(np.float32, copy=False)
    oz = np.where(shell_mask, oz, 0.0).astype(np.float32, copy=False)
    theta = np.arctan2(np.sqrt(ox * ox + oy * oy, dtype=np.float32), oz).astype(np.float32)
    psi = np.arctan2(oy, ox).astype(np.float32)
    return theta, psi


def _build_core_shell_fields(scenario: CoreShellScenario) -> dict[str, np.ndarray]:
    coords = np.genfromtxt(DATA_DIR / "LoG_coord.csv", delimiter=",", skip_header=1)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise AssertionError("Unexpected CoreShell coordinate table shape.")

    nz, ny, nx = SHAPE
    a_b = np.zeros(SHAPE, dtype=bool)
    b_b = np.zeros(SHAPE, dtype=bool)
    radial_x = np.zeros(SHAPE, dtype=np.float32)
    radial_y = np.zeros(SHAPE, dtype=np.float32)
    radial_z = np.zeros(SHAPE, dtype=np.float32)

    shell_radius = CORE_RADIUS_VOX + SHELL_THICKNESS_VOX
    z0, z1 = _local_bounds(CENTER_Z_VOX, shell_radius, nz)
    z = np.arange(z0, z1, dtype=np.float32)[:, None, None]

    for row in coords:
        px = float(row[0])
        py = float(row[1])
        y0, y1 = _local_bounds(py, shell_radius, ny)
        x0, x1 = _local_bounds(px, shell_radius, nx)

        y = np.arange(y0, y1, dtype=np.float32)[None, :, None]
        x = np.arange(x0, x1, dtype=np.float32)[None, None, :]

        mf = (x - px) ** 2 + (y - py) ** 2 + (z - CENTER_Z_VOX) ** 2
        core_mask = mf <= np.float32(CORE_RADIUS_VOX ** 2)
        shell_radius_mask = mf <= np.float32(shell_radius ** 2)

        a_view = a_b[z0:z1, y0:y1, x0:x1]
        a_view |= core_mask

        shell_mask = np.logical_and(~a_view, shell_radius_mask)
        b_view = b_b[z0:z1, y0:y1, x0:x1]
        b_view |= shell_mask

        dx = (x - px).astype(np.float32)
        dy = (y - py).astype(np.float32)
        dz = (z - CENTER_Z_VOX).astype(np.float32)

        rx_view = radial_x[z0:z1, y0:y1, x0:x1]
        ry_view = radial_y[z0:z1, y0:y1, x0:x1]
        rz_view = radial_z[z0:z1, y0:y1, x0:x1]

        rx_view += dx * shell_mask * (rx_view == 0)
        ry_view += dy * shell_mask * (ry_view == 0)
        rz_view += dz * shell_mask * (rz_view == 0)

    b_b = np.logical_and(~a_b, b_b)
    radial_x *= b_b
    radial_y *= b_b
    radial_z *= b_b
    c_b = np.logical_and(~a_b, ~b_b)

    radial_norm = np.sqrt(
        radial_x * radial_x + radial_y * radial_y + radial_z * radial_z,
        dtype=np.float32,
    )
    ratio = np.divide(
        np.float32(CORE_RADIUS_VOX) * b_b.astype(np.float32),
        radial_norm,
        out=np.zeros_like(radial_norm, dtype=np.float32),
        where=radial_norm > 0,
    )

    vf_a = a_b.astype(np.float32)
    vf_b = b_b.astype(np.float32)
    vf_c = c_b.astype(np.float32)

    shell_s_legacy = vf_b * np.float32(1.0 - PHI_ISO) * np.power(
        ratio,
        np.float32(DECAY_ORDER),
        dtype=np.float32,
    )
    shell_s_legacy = np.nan_to_num(shell_s_legacy, copy=False)

    if scenario.shell_s_mode == "legacy_decay":
        shell_s = shell_s_legacy
    elif scenario.shell_s_mode == "scaled_legacy_decay":
        shell_s = shell_s_legacy * np.float32(scenario.shell_s_scale)
    else:
        raise AssertionError(f"Unsupported shell S mode: {scenario.shell_s_mode}")

    theta_b, psi_b = _shell_orientation_angles(
        shell_mask=b_b,
        dx=radial_x,
        dy=radial_y,
        dz=radial_z,
        orientation_mode=scenario.shell_orientation_mode,
    )

    zeros = np.zeros(SHAPE, dtype=np.float32)
    return {
        "mat1_vfrac": vf_a,
        "mat1_s": zeros.copy(),
        "mat1_theta": zeros.copy(),
        "mat1_psi": zeros.copy(),
        "mat2_vfrac": vf_b,
        "mat2_s": shell_s.astype(np.float32),
        "mat2_theta": theta_b,
        "mat2_psi": psi_b,
        "mat3_vfrac": vf_c,
        "mat3_s": zeros.copy(),
        "mat3_theta": zeros.copy(),
        "mat3_psi": zeros.copy(),
    }


def _convert_fields_namespace(
    fields: dict[str, np.ndarray],
    field_namespace: str,
    array_dtype: np.dtype | type[np.floating] = np.float32,
) -> dict[str, np.ndarray]:
    target_dtype = np.dtype(array_dtype)
    if field_namespace == "numpy":
        return {
            key: np.ascontiguousarray(np.asarray(value, dtype=target_dtype))
            for key, value in fields.items()
        }
    if field_namespace != "cupy":
        raise AssertionError(f"Unsupported field namespace: {field_namespace}")

    cp = importlib.import_module("cupy")
    converted = {}
    for key, value in fields.items():
        converted[key] = cp.ascontiguousarray(cp.asarray(value, dtype=target_dtype))
    return converted


def build_core_shell_morphology(
    scenario: CoreShellScenario | str = "baseline",
    create_cy_object: bool = True,
    *,
    backend: str | None = None,
    backend_options: dict[str, Any] | None = None,
    resident_mode: str | None = None,
    input_policy: str = "coerce",
    ownership_policy: str | None = None,
    field_namespace: str = "numpy",
    array_dtype: np.dtype | type[np.floating] = np.float32,
) -> Morphology:
    if isinstance(scenario, str):
        if scenario != "baseline":
            raise AssertionError(f"Unsupported named CoreShell scenario: {scenario}")
        scenario = BASELINE_SCENARIO

    constants = _load_optical_constants()
    energies = list(map(float, constants[1].energies))
    fields = _convert_fields_namespace(
        _build_core_shell_fields(scenario=scenario),
        field_namespace=field_namespace,
        array_dtype=array_dtype,
    )

    materials = {
        1: Material(
            materialID=1,
            Vfrac=fields["mat1_vfrac"],
            S=fields["mat1_s"],
            theta=fields["mat1_theta"],
            psi=fields["mat1_psi"],
            energies=energies,
            opt_constants=constants[1].opt_constants,
            name="core",
        ),
        2: Material(
            materialID=2,
            Vfrac=fields["mat2_vfrac"],
            S=fields["mat2_s"],
            theta=fields["mat2_theta"],
            psi=fields["mat2_psi"],
            energies=energies,
            opt_constants=constants[2].opt_constants,
            name="shell",
        ),
        3: Material(
            materialID=3,
            Vfrac=fields["mat3_vfrac"],
            S=fields["mat3_s"],
            theta=fields["mat3_theta"],
            psi=fields["mat3_psi"],
            energies=energies,
            opt_constants=constants[3].opt_constants,
            name="matrix",
        ),
    }

    config = {
        "CaseType": 0,
        "MorphologyType": 0,
        "Energies": energies,
        "EAngleRotation": [0.0, 1.0, 360.0],
        "AlgorithmType": 0,
        "WindowingType": 0,
        "RotMask": 0,
        "ReferenceFrame": 1,
        "EwaldsInterpolation": 1,
    }

    morph = Morphology(
        3,
        materials=materials,
        PhysSize=PHYS_SIZE_NM,
        config=config,
        create_cy_object=create_cy_object,
        backend=backend,
        backend_options=backend_options,
        resident_mode=resident_mode,
        input_policy=input_policy,
        ownership_policy=ownership_policy,
    )
    morph.check_materials(quiet=True)
    if create_cy_object:
        morph.validate_all(quiet=True)
    return morph


def run_core_shell_pybind(scenario: CoreShellScenario | str = "baseline") -> xr.DataArray:
    morph = build_core_shell_morphology(
        scenario=scenario,
        create_cy_object=True,
        backend="cyrsoxs",
    )
    try:
        scattering = morph.run(stdout=False, stderr=False, return_xarray=True)
        if scattering is None:
            raise AssertionError("Core-shell pybind run returned no scattering data.")
        return scattering.copy(deep=True)
    finally:
        del morph
        release_runtime_memory()


def run_core_shell_backend(
    scenario: CoreShellScenario | str = "baseline",
    *,
    backend: str,
    backend_options: dict[str, Any] | None = None,
    resident_mode: str | None = None,
    input_policy: str = "coerce",
    ownership_policy: str | None = None,
    field_namespace: str = "numpy",
) -> tuple[xr.DataArray, dict[str, float]]:
    morph = build_core_shell_morphology(
        scenario=scenario,
        create_cy_object=True,
        backend=backend,
        backend_options=backend_options,
        resident_mode=resident_mode,
        input_policy=input_policy,
        ownership_policy=ownership_policy,
        field_namespace=field_namespace,
    )
    started = time.perf_counter()
    try:
        scattering = morph.run(stdout=False, stderr=False, return_xarray=True)
        if scattering is None:
            raise AssertionError("Core-shell backend run returned no scattering data.")
        timings = dict(morph.backend_timings)
        timings.setdefault("wall_seconds", time.perf_counter() - started)
        return scattering.copy(deep=True), timings
    finally:
        morph.release_runtime()
        del morph
        release_runtime_memory()


def _angular_distance_deg(angles_deg: np.ndarray, center_deg: float) -> np.ndarray:
    return ((angles_deg - float(center_deg) + 180.0) % 360.0) - 180.0


def scattering_to_awedge(scattering: xr.DataArray) -> xr.DataArray:
    from PyHyperScattering.integrate import WPIntegrator

    integrator = WPIntegrator(use_chunked_processing=False)
    remeshed = integrator.integrateImageStack(scattering)
    if "energy" not in remeshed.dims or "chi" not in remeshed.dims:
        raise AssertionError("PyHyperScattering output missing expected energy/chi dimensions.")
    qdim = next((dim for dim in remeshed.dims if dim == "q" or dim.startswith("q")), None)
    if qdim is None:
        raise AssertionError("PyHyperScattering output missing q dimension.")

    chi = np.mod(np.asarray(remeshed.coords["chi"].values, dtype=np.float64), 360.0)
    horiz_idx = np.flatnonzero(
        np.abs(_angular_distance_deg(chi, HORIZONTAL_CHI_CENTER_DEG)) < AWEDGE_HALF_WIDTH_DEG
    )
    vert_idx = np.flatnonzero(
        np.abs(_angular_distance_deg(chi, VERTICAL_CHI_CENTER_DEG)) < AWEDGE_HALF_WIDTH_DEG
    )
    if horiz_idx.size == 0 or vert_idx.size == 0:
        raise AssertionError("Failed to resolve legacy A-wedge chi sectors.")

    horiz = remeshed.isel(chi=horiz_idx).mean("chi")
    vert = remeshed.isel(chi=vert_idx).mean("chi")
    with np.errstate(divide="ignore", invalid="ignore"):
        awedge = -(vert - horiz) / (vert + horiz)
    if qdim != "q":
        awedge = awedge.rename({qdim: "q"})
    return awedge.sortby("energy")


def awedge_comparison_slices(
    awedge: xr.DataArray,
    reference: xr.DataArray,
) -> dict[str, xr.DataArray]:
    q_window = dict(q=slice(Q_COMPARE_MIN, Q_COMPARE_MAX))
    comparison = {
        "awedge": awedge.sel(**q_window),
        "reference": reference.sel(**q_window),
    }
    comparison["a_vs_energy"] = comparison["awedge"].mean("q")
    comparison["reference_a_vs_energy"] = comparison["reference"].mean("q")
    for energy_ev in ENERGY_CHECKS_EV:
        key = f"a_vs_q_{str(energy_ev).replace('.', 'p')}"
        comparison[key] = comparison["awedge"].sel(energy=float(energy_ev), method="nearest")
        comparison[f"reference_{key}"] = comparison["reference"].sel(energy=float(energy_ev), method="nearest")
    return comparison


def compute_awedge_metrics(comparison: dict[str, xr.DataArray]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for key in ("a_vs_energy", "a_vs_q_284p7", "a_vs_q_285p2"):
        ref_key = "reference_a_vs_energy" if key == "a_vs_energy" else f"reference_{key}"
        delta = np.asarray(comparison[key].values - comparison[ref_key].values, dtype=np.float64)
        metrics[key] = {
            "max_abs_diff": float(np.nanmax(np.abs(delta))),
            "rmse": float(np.sqrt(np.nanmean(delta * delta))),
        }
    return metrics


def metrics_within_thresholds(
    metrics: dict[str, dict[str, float]],
    thresholds: dict[str, dict[str, float]],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for key, limits in thresholds.items():
        for metric_name, limit in limits.items():
            value = metrics[key][metric_name]
            if not np.isfinite(value) or value > float(limit):
                failures.append(f"{key} {metric_name}={value:.6g} exceeds limit {float(limit):.6g}")
    return len(failures) == 0, failures


def plot_core_shell_validation_panel(
    comparison: dict[str, xr.DataArray],
    metrics: dict[str, dict[str, float]],
    out_path: Path,
    scenario: CoreShellScenario,
    reference_label: str,
    reference_citation: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(
        comparison["reference_a_vs_energy"].coords["energy"].values,
        comparison["reference_a_vs_energy"].values,
        color="black",
        linewidth=1.7,
        label=reference_label,
    )
    ax.plot(
        comparison["a_vs_energy"].coords["energy"].values,
        comparison["a_vs_energy"].values,
        color="#d62728",
        linewidth=1.8,
        linestyle="--",
        label="Pybind + WPIntegrator",
    )
    ax.set_xlabel("Energy [eV]")
    ax.set_ylabel("A(E)")
    ax.set_title("A-wedge average vs energy")
    ax.set_xticks([280, 282, 284, 286, 288, 290])
    ax.legend(loc="best")

    ax = axes[0, 1]
    ax.plot(
        comparison["reference_a_vs_q_284p7"].coords["q"].values,
        comparison["reference_a_vs_q_284p7"].values,
        color="black",
        linewidth=1.6,
        label="284.7 eV reference",
    )
    ax.plot(
        comparison["reference_a_vs_q_285p2"].coords["q"].values,
        comparison["reference_a_vs_q_285p2"].values,
        color="#1f77b4",
        linewidth=1.6,
        label="285.2 eV reference",
    )
    ax.plot(
        comparison["a_vs_q_284p7"].coords["q"].values,
        comparison["a_vs_q_284p7"].values,
        color="black",
        linewidth=1.8,
        linestyle="--",
        label="284.7 eV pybind",
    )
    ax.plot(
        comparison["a_vs_q_285p2"].coords["q"].values,
        comparison["a_vs_q_285p2"].values,
        color="#1f77b4",
        linewidth=1.8,
        linestyle="--",
        label="285.2 eV pybind",
    )
    ax.set_xlabel(r"q [nm$^{-1}$]")
    ax.set_ylabel("A(q)")
    ax.set_xlim(Q_COMPARE_MIN, Q_COMPARE_MAX)
    ax.set_ylim(-0.31, 0.31)
    ax.set_title("A-wedge slices vs q")
    ax.legend(loc="best", fontsize=9)

    ax = axes[1, 0]
    residual_2847 = comparison["a_vs_q_284p7"] - comparison["reference_a_vs_q_284p7"]
    residual_2852 = comparison["a_vs_q_285p2"] - comparison["reference_a_vs_q_285p2"]
    ax.plot(
        residual_2847.coords["q"].values,
        residual_2847.values,
        color="black",
        linewidth=1.6,
        label="284.7 eV residual",
    )
    ax.plot(
        residual_2852.coords["q"].values,
        residual_2852.values,
        color="#1f77b4",
        linewidth=1.6,
        label="285.2 eV residual",
    )
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.set_xlabel(r"q [nm$^{-1}$]")
    ax.set_ylabel(r"$\Delta A(q)$")
    ax.set_xlim(Q_COMPARE_MIN, Q_COMPARE_MAX)
    ax.set_title("Residuals vs q")
    ax.legend(loc="best")

    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        scenario.display_name,
        "",
        f"Scenario: {scenario.scenario_id}",
        scenario.description,
        "",
        f"Shape: {SHAPE}",
        f"PhysSize: {PHYS_SIZE_NM} nm",
        f"EAngleRotation: [0.0, 1.0, 360.0]",
        f"q range: [{Q_COMPARE_MIN:.2f}, {Q_COMPARE_MAX:.2f}] nm^-1",
        "",
        "Metrics:",
        f"A(E) max abs {metrics['a_vs_energy']['max_abs_diff']:.5f}",
        f"A(E) RMSE {metrics['a_vs_energy']['rmse']:.5f}",
        f"A(q) 284.7 max abs {metrics['a_vs_q_284p7']['max_abs_diff']:.5f}",
        f"A(q) 284.7 RMSE {metrics['a_vs_q_284p7']['rmse']:.5f}",
        f"A(q) 285.2 max abs {metrics['a_vs_q_285p2']['max_abs_diff']:.5f}",
        f"A(q) 285.2 RMSE {metrics['a_vs_q_285p2']['rmse']:.5f}",
        "",
        "Reference:",
        reference_citation,
    ]
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=9,
        wrap=True,
    )

    fig.suptitle("CoreShell A-wedge validation", fontsize=15)
    fig.savefig(
        out_path,
        format="png",
        dpi=160,
        bbox_inches="tight",
        metadata={
            "Title": f"CoreShell A-wedge validation: {scenario.scenario_id}",
            "Description": (
                f"{scenario.display_name}. {scenario.description} Reference: "
                f"{reference_citation}"
            ),
            "Author": "OpenAI Codex",
            "Software": "matplotlib",
        },
    )
    plt.close(fig)
