from __future__ import annotations

import csv
import gc
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from NRSS.morphology import Material, Morphology
from tests.path_matrix import ComputationPath
from tests.validation.lib.orientational_contrast import (
    UniaxialOpticalState,
    predict_uniaxial_vacuum_far_field_contrast,
)


pytestmark = [pytest.mark.path_matrix, pytest.mark.reference_parity]


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = REPO_ROOT / "test-reports" / "sphere-orientational-contrast-dev"
WRITE_VALIDATION_PLOTS = os.environ.get("NRSS_WRITE_VALIDATION_PLOTS", "").strip() == "1"

SHAPE = (128, 128, 128)
PHYS_SIZE_NM = 2.0
DIAMETER_NM = 32.0
Q_INTEGRATE_MIN = 0.06
Q_INTEGRATE_MAX = 1.0
RATIO_REL_ERR_MAX = 0.011

FAMILY_COLORS = {
    "delta": "#d62728",
    "beta": "#1f77b4",
    "mixed": "#2ca02c",
}

SERIES_MARKERS = {
    "reference": "x",
    "theta": "o",
    "psi": "s",
    "low_sym": "^",
    "S": "D",
}

SERIES_LABELS = {
    "reference": "Reference",
    "theta": r"$\theta$ sweep",
    "psi": r"$\psi$ sweep",
    "low_sym": "Low symmetry",
    "S": "S sweep",
}


@dataclass(frozen=True)
class OrientationalScenario:
    label: str
    series: str
    theta: float
    psi: float
    S: float


@dataclass(frozen=True)
class FamilySpec:
    name: str
    energy_ev: float
    delta_para: float
    beta_para: float
    delta_perp: float
    beta_perp: float

    def build_state(self, scenario: OrientationalScenario) -> UniaxialOpticalState:
        return UniaxialOpticalState(
            delta_para=self.delta_para,
            beta_para=self.beta_para,
            delta_perp=self.delta_perp,
            beta_perp=self.beta_perp,
            theta=scenario.theta,
            psi=scenario.psi,
            S=scenario.S,
        )


REFERENCE_SCENARIO = OrientationalScenario(
    label="ref_theta_pi2_psi0_s1",
    series="reference",
    theta=np.pi / 2.0,
    psi=0.0,
    S=1.0,
)


ORIENTATION_SCENARIOS = (
    # High-symmetry theta coverage across the full [0, pi] range.
    OrientationalScenario("theta_0", "theta", 0.0, 0.0, 1.0),
    OrientationalScenario("theta_pi6", "theta", np.pi / 6.0, 0.0, 1.0),
    OrientationalScenario("theta_pi3", "theta", np.pi / 3.0, 0.0, 1.0),
    OrientationalScenario("theta_2pi3", "theta", 2.0 * np.pi / 3.0, 0.0, 1.0),
    OrientationalScenario("theta_5pi6", "theta", 5.0 * np.pi / 6.0, 0.0, 1.0),
    OrientationalScenario("theta_pi", "theta", np.pi, 0.0, 1.0),
    # High-symmetry psi coverage spanning the [0, 2*pi] range.
    OrientationalScenario("psi_pi6", "psi", np.pi / 2.0, np.pi / 6.0, 1.0),
    OrientationalScenario("psi_pi3", "psi", np.pi / 2.0, np.pi / 3.0, 1.0),
    OrientationalScenario("psi_pi2", "psi", np.pi / 2.0, np.pi / 2.0, 1.0),
    OrientationalScenario("psi_3pi2", "psi", np.pi / 2.0, 3.0 * np.pi / 2.0, 1.0),
    OrientationalScenario("psi_11pi6", "psi", np.pi / 2.0, 11.0 * np.pi / 6.0, 1.0),
    # Low-symmetry coupled Euler selections.
    OrientationalScenario("lowsym_a", "low_sym", 3.0 * np.pi / 10.0, np.pi / 11.0, 1.0),
    OrientationalScenario("lowsym_b", "low_sym", 7.0 * np.pi / 10.0, 13.0 * np.pi / 11.0, 1.0),
    OrientationalScenario("lowsym_c", "low_sym", 9.0 * np.pi / 10.0, 5.0 * np.pi / 3.0, 1.0),
    # S sweep at fixed low symmetry, including the explicit isotropic endpoint.
    OrientationalScenario("lowsym_a_s0", "S", 3.0 * np.pi / 10.0, np.pi / 11.0, 0.0),
    OrientationalScenario("lowsym_a_s025", "S", 3.0 * np.pi / 10.0, np.pi / 11.0, 0.25),
    OrientationalScenario("lowsym_a_s05", "S", 3.0 * np.pi / 10.0, np.pi / 11.0, 0.5),
    OrientationalScenario("lowsym_a_s075", "S", 3.0 * np.pi / 10.0, np.pi / 11.0, 0.75),
)


FAMILY_SPECS = (
    FamilySpec(
        name="delta",
        energy_ev=285.01,
        delta_para=2e-4,
        beta_para=0.0,
        delta_perp=1e-4,
        beta_perp=0.0,
    ),
    FamilySpec(
        name="beta",
        energy_ev=285.02,
        delta_para=0.0,
        beta_para=2e-4,
        delta_perp=0.0,
        beta_perp=1e-4,
    ),
    FamilySpec(
        name="mixed",
        energy_ev=285.03,
        delta_para=2e-4,
        beta_para=2e-4,
        delta_perp=1e-4,
        beta_perp=1e-4,
    ),
)


@lru_cache(maxsize=1)
def _has_visible_gpu() -> bool:
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


def _release_runtime_memory() -> None:
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


def _to_backend_namespace(array: np.ndarray, field_namespace: str):
    if field_namespace == "numpy":
        return np.ascontiguousarray(array.astype(np.float32, copy=False))
    if field_namespace != "cupy":
        raise AssertionError(f"Unsupported field namespace {field_namespace!r}.")
    import cupy as cp

    return cp.ascontiguousarray(cp.asarray(array, dtype=cp.float32))


def _path_runtime_kwargs(nrss_path: ComputationPath) -> dict[str, object]:
    return {
        "backend": nrss_path.backend,
        "backend_options": nrss_path.backend_options,
        "resident_mode": nrss_path.resident_mode,
        "input_policy": "strict" if nrss_path.category == "cupy" else "coerce",
        "ownership_policy": nrss_path.ownership_policy,
        "field_namespace": nrss_path.field_namespace,
    }


def _cyrsoxs_detector_axis(n: int, phys_size_nm: float) -> np.ndarray:
    if int(n) < 2:
        raise AssertionError(f"CyRSoXS detector axis needs at least 2 points, got n={n}.")
    start = -np.pi / float(phys_size_nm)
    step = (2.0 * np.pi / float(phys_size_nm)) / float(int(n) - 1)
    return start + np.arange(int(n), dtype=np.float64) * step


@lru_cache(maxsize=1)
def _detector_annulus_mask() -> np.ndarray:
    qy = _cyrsoxs_detector_axis(SHAPE[1], PHYS_SIZE_NM)
    qx = _cyrsoxs_detector_axis(SHAPE[2], PHYS_SIZE_NM)
    qx_grid, qy_grid = np.meshgrid(qx, qy)
    q = np.sqrt(qx_grid * qx_grid + qy_grid * qy_grid)
    mask = np.logical_and.reduce(
        [
            q >= Q_INTEGRATE_MIN,
            q <= Q_INTEGRATE_MAX,
            np.isfinite(q),
        ]
    )
    if int(np.count_nonzero(mask)) < 256:
        raise AssertionError("Insufficient detector pixels for sphere orientational-contrast annulus.")
    return mask


def _detector_annulus_intensity(detector_image: np.ndarray) -> float:
    img = np.asarray(detector_image, dtype=np.float64)
    if img.shape != (SHAPE[1], SHAPE[2]):
        raise AssertionError(f"Unexpected detector image shape {img.shape!r}.")

    mask = np.logical_and.reduce(
        [
            _detector_annulus_mask(),
            np.isfinite(img),
            img >= 0.0,
        ]
    )
    return float(np.sum(img[mask], dtype=np.float64))


@lru_cache(maxsize=1)
def _sphere_and_vacuum_vfrac() -> tuple[np.ndarray, np.ndarray]:
    nz, ny, nx = SHAPE
    radius_vox = float(DIAMETER_NM) / (2.0 * PHYS_SIZE_NM)
    cz = (nz - 1) / 2.0
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0

    zz, yy, xx = np.indices(SHAPE, dtype=np.float32)
    dist2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    sphere = (dist2 <= np.float32(radius_vox * radius_vox)).astype(np.float32)
    vacuum = (1.0 - sphere).astype(np.float32)
    return sphere, vacuum


def _run_sphere_scenario(
    scenario: OrientationalScenario,
    runtime_kwargs: dict[str, object],
) -> dict[str, float]:
    sphere_vfrac_host, vacuum_vfrac_host = _sphere_and_vacuum_vfrac()
    field_namespace = str(runtime_kwargs["field_namespace"])
    sphere_vfrac = _to_backend_namespace(sphere_vfrac_host, field_namespace)
    vacuum_vfrac = _to_backend_namespace(vacuum_vfrac_host, field_namespace)
    s_field = _to_backend_namespace((scenario.S * sphere_vfrac_host).astype(np.float32), field_namespace)
    theta = _to_backend_namespace((scenario.theta * sphere_vfrac_host).astype(np.float32), field_namespace)
    psi = _to_backend_namespace((scenario.psi * sphere_vfrac_host).astype(np.float32), field_namespace)

    energies = [family.energy_ev for family in FAMILY_SPECS]
    mat1 = Material(
        materialID=1,
        Vfrac=sphere_vfrac,
        S=s_field,
        theta=theta,
        psi=psi,
        energies=energies,
        opt_constants={
            family.energy_ev: [
                family.delta_para,
                family.beta_para,
                family.delta_perp,
                family.beta_perp,
            ]
            for family in FAMILY_SPECS
        },
        name="sphere",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=vacuum_vfrac,
        S=None,
        theta=None,
        psi=None,
        energies=energies,
        opt_constants={family.energy_ev: [0.0, 0.0, 0.0, 0.0] for family in FAMILY_SPECS},
        name="vacuum",
    )
    config = {
        "CaseType": 0,
        "MorphologyType": 0,
        "Energies": energies,
        "EAngleRotation": [0.0, 0.0, 0.0],
        "RotMask": 1,
        "WindowingType": 0,
        "AlgorithmType": 0,
        "ReferenceFrame": 1,
        "EwaldsInterpolation": 1,
    }
    morph = Morphology(
        2,
        materials={1: mat1, 2: mat2},
        PhysSize=PHYS_SIZE_NM,
        config=config,
        backend=str(runtime_kwargs["backend"]),
        backend_options=dict(runtime_kwargs["backend_options"]),
        resident_mode=runtime_kwargs["resident_mode"],
        input_policy=str(runtime_kwargs["input_policy"]),
        ownership_policy=runtime_kwargs["ownership_policy"],
        create_cy_object=True,
    )
    try:
        scattering = morph.run(stdout=False, stderr=False, return_xarray=True)
        if scattering is None:
            raise AssertionError("Orientational contrast backend run returned no scattering data.")
        intensities = {}
        for family in FAMILY_SPECS:
            detector_image = np.asarray(scattering.sel(energy=float(family.energy_ev)).values, dtype=np.float64)
            intensities[family.name] = _detector_annulus_intensity(detector_image)
        return intensities
    finally:
        morph.release_runtime()
        del morph
        del sphere_vfrac, vacuum_vfrac, s_field, theta, psi
        _release_runtime_memory()


def evaluate_orientational_ratio_rows(nrss_path: ComputationPath) -> list[dict[str, float | str]]:
    runtime_kwargs = _path_runtime_kwargs(nrss_path)
    observed_by_label = {
        REFERENCE_SCENARIO.label: _run_sphere_scenario(REFERENCE_SCENARIO, runtime_kwargs),
    }
    for scenario in ORIENTATION_SCENARIOS:
        observed_by_label[scenario.label] = _run_sphere_scenario(scenario, runtime_kwargs)

    rows: list[dict[str, float | str]] = []
    for family in FAMILY_SPECS:
        ref_expected = predict_uniaxial_vacuum_far_field_contrast(
            family.build_state(REFERENCE_SCENARIO)
        ).far_field_contrast_sq
        ref_observed = observed_by_label[REFERENCE_SCENARIO.label][family.name]
        if ref_expected <= 0.0 or ref_observed <= 0.0:
            raise AssertionError(f"Reference contrast must be positive for family={family.name}.")

        rows.append(
            {
                "family": family.name,
                "energy_ev": float(family.energy_ev),
                "label": REFERENCE_SCENARIO.label,
                "series": REFERENCE_SCENARIO.series,
                "theta": float(REFERENCE_SCENARIO.theta),
                "psi": float(REFERENCE_SCENARIO.psi),
                "S": float(REFERENCE_SCENARIO.S),
                "expected_ratio": 1.0,
                "observed_ratio": 1.0,
                "rel_err": 0.0,
            }
        )

        for scenario in ORIENTATION_SCENARIOS:
            expected_ratio = (
                predict_uniaxial_vacuum_far_field_contrast(
                    family.build_state(scenario)
                ).far_field_contrast_sq
                / ref_expected
            )
            observed_ratio = observed_by_label[scenario.label][family.name] / ref_observed
            rel_err = abs(observed_ratio - expected_ratio) / expected_ratio
            rows.append(
                {
                    "family": family.name,
                    "energy_ev": float(family.energy_ev),
                    "label": scenario.label,
                    "series": scenario.series,
                    "theta": float(scenario.theta),
                    "psi": float(scenario.psi),
                    "S": float(scenario.S),
                    "expected_ratio": float(expected_ratio),
                    "observed_ratio": float(observed_ratio),
                    "rel_err": float(rel_err),
                }
            )

    return rows


def _series_rows(
    rows: list[dict[str, float | str]],
    family_name: str,
    series_name: str,
) -> list[dict[str, float | str]]:
    series_rows = [
        row
        for row in rows
        if row["family"] == family_name and row["series"] == series_name
    ]
    if series_name == "theta":
        series_rows.sort(key=lambda row: float(row["theta"]))
    elif series_name == "psi":
        series_rows.sort(key=lambda row: float(row["psi"]))
    elif series_name == "S":
        series_rows.sort(key=lambda row: float(row["S"]))
    else:
        series_rows.sort(key=lambda row: str(row["label"]))
    return series_rows


def _write_ratio_rows_tsv(rows: list[dict[str, float | str]], path_id: str) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / f"{path_id}__orientational_ratio_rows.tsv"
    fieldnames = [
        "family",
        "energy_ev",
        "label",
        "series",
        "theta",
        "psi",
        "S",
        "expected_ratio",
        "observed_ratio",
        "rel_err",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out


def _write_family_plot(
    family_name: str,
    rows: list[dict[str, float | str]],
    path_id: str,
) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    family_rows = [row for row in rows if row["family"] == family_name]
    non_ref_rows = [row for row in family_rows if row["series"] != "reference"]
    color = FAMILY_COLORS[family_name]

    theta_rows = _series_rows(rows, family_name, "theta")
    psi_rows = _series_rows(rows, family_name, "psi")
    low_rows = _series_rows(rows, family_name, "low_sym")
    s_rows = _series_rows(rows, family_name, "S")
    ref_row = next(row for row in family_rows if row["series"] == "reference")
    max_rel_err = max(float(row["rel_err"]) for row in non_ref_rows)

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.8))
    ax_scatter, ax_theta, ax_psi, ax_s, ax_low, ax_text = axes.flatten()

    lim = max(
        1.05,
        1.05 * max(
            max(float(row["expected_ratio"]), float(row["observed_ratio"]))
            for row in family_rows
        ),
    )
    ax_scatter.plot([0.0, lim], [0.0, lim], color="gray", linestyle=":", linewidth=1.0)
    for series_name in ("theta", "psi", "low_sym", "S"):
        srows = [row for row in non_ref_rows if row["series"] == series_name]
        if not srows:
            continue
        ax_scatter.scatter(
            [float(row["expected_ratio"]) for row in srows],
            [float(row["observed_ratio"]) for row in srows],
            marker=SERIES_MARKERS[series_name],
            s=55,
            alpha=0.9,
            color=color,
            label=SERIES_LABELS[series_name],
        )
    ax_scatter.scatter([1.0], [1.0], marker="x", s=75, color="black", label="Reference")
    ax_scatter.set_xlim(0.0, lim)
    ax_scatter.set_ylim(0.0, lim)
    ax_scatter.set_xlabel("Expected ratio")
    ax_scatter.set_ylabel("Observed ratio")
    ax_scatter.set_title(f"{family_name}: Expected vs Observed")
    ax_scatter.grid(alpha=0.25)
    ax_scatter.legend(loc="best", fontsize=8.5)

    ax_theta.plot(
        [float(row["theta"]) / np.pi for row in theta_rows],
        [float(row["expected_ratio"]) for row in theta_rows],
        color="black",
        linestyle="--",
        marker="o",
        label="Expected",
    )
    ax_theta.plot(
        [float(row["theta"]) / np.pi for row in theta_rows],
        [float(row["observed_ratio"]) for row in theta_rows],
        color=color,
        marker="o",
        label="Observed",
    )
    ax_theta.set_xlabel(r"$\theta / \pi$")
    ax_theta.set_ylabel("Ratio")
    ax_theta.set_title(rf"{family_name}: $\theta$ sweep at $\psi=0$, $S=1$")
    ax_theta.grid(alpha=0.25)
    ax_theta.legend(loc="best", fontsize=8.5)

    ax_psi.plot(
        [float(row["psi"]) / np.pi for row in psi_rows],
        [float(row["expected_ratio"]) for row in psi_rows],
        color="black",
        linestyle="--",
        marker="s",
        label="Expected",
    )
    ax_psi.plot(
        [float(row["psi"]) / np.pi for row in psi_rows],
        [float(row["observed_ratio"]) for row in psi_rows],
        color=color,
        marker="s",
        label="Observed",
    )
    ax_psi.set_xlabel(r"$\psi / \pi$")
    ax_psi.set_ylabel("Ratio")
    ax_psi.set_title(rf"{family_name}: $\psi$ sweep at $\theta=\pi/2$, $S=1$")
    ax_psi.grid(alpha=0.25)
    ax_psi.legend(loc="best", fontsize=8.5)

    ax_s.plot(
        [float(row["S"]) for row in s_rows],
        [float(row["expected_ratio"]) for row in s_rows],
        color="black",
        linestyle="--",
        marker="D",
        label="Expected",
    )
    ax_s.plot(
        [float(row["S"]) for row in s_rows],
        [float(row["observed_ratio"]) for row in s_rows],
        color=color,
        marker="D",
        label="Observed",
    )
    ax_s.set_xlabel("S")
    ax_s.set_ylabel("Ratio")
    ax_s.set_title(f"{family_name}: S sweep at low symmetry")
    ax_s.grid(alpha=0.25)
    ax_s.legend(loc="best", fontsize=8.5)

    ax_low.plot(
        range(len(low_rows)),
        [float(row["expected_ratio"]) for row in low_rows],
        color="black",
        linestyle="--",
        marker="^",
        label="Expected",
    )
    ax_low.plot(
        range(len(low_rows)),
        [float(row["observed_ratio"]) for row in low_rows],
        color=color,
        marker="^",
        label="Observed",
    )
    ax_low.set_xticks(range(len(low_rows)))
    ax_low.set_xticklabels([str(row["label"]).replace("lowsym_", "") for row in low_rows])
    ax_low.set_ylabel("Ratio")
    ax_low.set_title(f"{family_name}: low-symmetry coupled cases")
    ax_low.grid(alpha=0.25)
    ax_low.legend(loc="best", fontsize=8.5)

    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                f"family = {family_name}",
                f"energy = {float(ref_row['energy_ev']):.2f} eV",
                "reference = (theta=pi/2, psi=0, S=1)",
                (
                    "OC = "
                    f"para(delta={next(f for f in FAMILY_SPECS if f.name == family_name).delta_para:.1e}, "
                    f"beta={next(f for f in FAMILY_SPECS if f.name == family_name).beta_para:.1e}), "
                    f"perp(delta={next(f for f in FAMILY_SPECS if f.name == family_name).delta_perp:.1e}, "
                    f"beta={next(f for f in FAMILY_SPECS if f.name == family_name).beta_perp:.1e})"
                ),
                f"max relative error = {max_rel_err:.5f}",
                "",
                "series included:",
                "- theta sweep",
                "- psi sweep",
                "- low-symmetry coupled Euler cases",
                "- S sweep including S=0",
            ]
        ),
        ha="left",
        va="top",
        fontsize=9.5,
        family="monospace",
    )

    fig.suptitle("Sphere Orientational Contrast Scaling", fontsize=14)
    fig.tight_layout()
    out = PLOT_DIR / f"{path_id}__{family_name}_orientational_contrast.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def write_orientational_validation_artifacts(
    rows: list[dict[str, float | str]],
    path_id: str,
) -> list[Path]:
    outputs = [_write_ratio_rows_tsv(rows, path_id=path_id)]
    for family in FAMILY_SPECS:
        outputs.append(_write_family_plot(family.name, rows, path_id=path_id))
    return outputs


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.physics_validation
@pytest.mark.toolchain_validation
def test_sphere_orientational_contrast_scaling_pybind(nrss_path: ComputationPath):
    """Validate helper-predicted orientational contrast ratios for a sphere in vacuum.

    This physics test keeps the sphere form factor fixed and varies contrast only
    through the uniaxial optical-tensor orientation (`theta`, `psi`) and the
    aligned fraction `S`. It exercises three close-energy optical-constant
    families (pure delta dichroism, pure beta dichroism, and mixed delta+beta)
    and compares every simulated detector-annulus ratio against the reusable
    How-to-RSoXS Eq. 15/16 far-field helper prediction for the same scenario.

    The scenario matrix intentionally spans:
    - high-symmetry `theta` selections across `[0, pi]`,
    - high-symmetry `psi` selections across `[0, 2*pi]`,
    - low-symmetry coupled Euler selections,
    - an `S` series that includes the explicit isotropic endpoint `S = 0`.

    All ratios are referenced to the common `(theta=pi/2, psi=0, S=1)` state so
    the geometry and form factor stay fixed while only the orientational
    contrast changes.
    """
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for sphere orientational-contrast validation.")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    assert len(ORIENTATION_SCENARIOS) == 18

    rows = evaluate_orientational_ratio_rows(nrss_path)
    if WRITE_VALIDATION_PLOTS:
        write_orientational_validation_artifacts(rows, path_id=nrss_path.id)

    non_ref_rows = [row for row in rows if row["series"] != "reference"]
    max_rel_err = 0.0
    worst_case = None
    for row in non_ref_rows:
        rel_err = float(row["rel_err"])
        if rel_err > max_rel_err:
            max_rel_err = rel_err
            worst_case = row
        assert rel_err < RATIO_REL_ERR_MAX, (
            f"family={row['family']} scenario={row['label']} ratio mismatch too large: "
            f"expected={float(row['expected_ratio']):.9f} "
            f"observed={float(row['observed_ratio']):.9f} "
            f"rel={rel_err:.6f}"
        )

    if worst_case is not None:
        print(
            "worst_case "
            f"family={worst_case['family']} "
            f"scenario={worst_case['label']} "
            f"expected={float(worst_case['expected_ratio']):.9f} "
            f"observed={float(worst_case['observed_ratio']):.9f} "
            f"rel={float(worst_case['rel_err']):.6f}"
        )
