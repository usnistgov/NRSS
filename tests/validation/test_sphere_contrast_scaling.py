import gc
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import CyRSoXS as cy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = REPO_ROOT / "test-reports" / "sphere-contrast-scaling-dev"
WRITE_VALIDATION_PLOTS = os.environ.get("NRSS_WRITE_VALIDATION_PLOTS", "").strip() == "1"

SHAPE = (512, 512, 512)
PHYS_SIZE_NM = 1.0
DIAMETER_NM = 70.0
ENERGY_START_EV = 285.0
ENERGY_STEP_EV = 0.01
CONTRAST_MAGNITUDES = (1e-4, 2e-4, 3e-4)
Q_INTEGRATE_MIN = 0.06
Q_INTEGRATE_MAX = 1.0
CONTRAST_WEIGHTED_REL_ERR_MAX = 0.005
CONTRAST_UNWEIGHTED_REL_ERR_MAX = 0.005
CONTRAST_PAIRING_REL_ERR_MAX = 0.005
CONTRAST_INTEGRAL_CONSISTENCY_REL_MAX = 5e-4

FAMILY_COLORS = {
    "beta_vac": "#1f77b4",
    "delta_pos_vac": "#d62728",
    "delta_neg_vac": "#2ca02c",
    "mixed_vac": "#ff7f0e",
    "beta_split": "#8c564b",
    "delta_pos_split": "#e377c2",
    "delta_neg_split": "#17becf",
    "mixed_split": "#bcbd22",
}


@dataclass(frozen=True)
class ContrastScenario:
    family: str
    label: str
    energy_ev: float
    magnitude: float
    sphere_oc: tuple[float, float, float, float]
    matrix_oc: tuple[float, float, float, float]
    delta_beta: float
    delta_delta: float

    @property
    def expected_contrast_sq(self) -> float:
        return self.delta_beta * self.delta_beta + self.delta_delta * self.delta_delta


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


def _cyrsoxs_detector_axis(n: int, phys_size_nm: float) -> np.ndarray:
    if int(n) < 2:
        raise AssertionError(f"CyRSoXS detector axis needs at least 2 points, got n={n}.")
    start = -np.pi / float(phys_size_nm)
    step = (2.0 * np.pi / float(phys_size_nm)) / float(int(n) - 1)
    return start + np.arange(int(n), dtype=np.float64) * step


def _sphere_and_matrix_vfrac() -> tuple[np.ndarray, np.ndarray]:
    nz, ny, nx = SHAPE
    radius_vox = float(DIAMETER_NM) / (2.0 * PHYS_SIZE_NM)
    cz = (nz - 1) / 2.0
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0

    pad = radius_vox + 2.0
    z0 = max(0, int(np.floor(cz - pad)))
    z1 = min(nz, int(np.ceil(cz + pad)) + 1)
    y0 = max(0, int(np.floor(cy - pad)))
    y1 = min(ny, int(np.ceil(cy + pad)) + 1)
    x0 = max(0, int(np.floor(cx - pad)))
    x1 = min(nx, int(np.ceil(cx + pad)) + 1)

    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    dz = zz.astype(np.float32) - np.float32(cz)
    dy = yy.astype(np.float32) - np.float32(cy)
    dx = xx.astype(np.float32) - np.float32(cx)
    dist2 = dx * dx + dy * dy + dz * dz

    sphere = np.zeros(SHAPE, dtype=np.float32)
    local_sphere = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=np.float32)
    local_sphere[dist2 <= np.float32(radius_vox * radius_vox)] = 1.0
    sphere[z0:z1, y0:y1, x0:x1] = local_sphere
    matrix = (1.0 - sphere).astype(np.float32)
    return sphere, matrix


def _scattering_to_xarray(scattering_pattern, energies_eV: list[float]) -> xr.DataArray:
    scattering_data = scattering_pattern.writeAllToNumpy(kID=0)
    qy = _cyrsoxs_detector_axis(SHAPE[1], PHYS_SIZE_NM)
    qx = _cyrsoxs_detector_axis(SHAPE[2], PHYS_SIZE_NM)
    return xr.DataArray(
        scattering_data,
        dims=["energy", "qy", "qx"],
        coords={"energy": list(map(float, energies_eV)), "qy": qy, "qx": qx},
    )


def _run_sphere_pybind(scenarios: list[ContrastScenario]) -> xr.DataArray:
    sphere_vfrac, matrix_vfrac = _sphere_and_matrix_vfrac()
    energies_eV = [scenario.energy_ev for scenario in scenarios]

    input_data = cy.InputData(NumMaterial=2)
    input_data.setEnergies(list(map(float, energies_eV)))
    input_data.setERotationAngle(StartAngle=0.0, EndAngle=0.0, IncrementAngle=0.0)
    input_data.setPhysSize(PHYS_SIZE_NM)
    input_data.setDimensions(SHAPE, cy.MorphologyOrder.ZYX)
    input_data.setCaseType(cy.CaseType.Default)
    input_data.setMorphologyType(cy.MorphologyType.EulerAngles)
    input_data.setAlgorithm(AlgorithmID=0, MaxStreams=1)
    input_data.interpolationType = cy.InterpolationType.Linear
    input_data.windowingType = cy.FFTWindowing.NoPadding
    input_data.rotMask = True
    input_data.referenceFrame = 1
    if not input_data.validate():
        raise AssertionError("CyRSoXS InputData validation failed.")

    optical_constants = cy.RefractiveIndex(input_data)
    for scenario in scenarios:
        optical_constants.addData(
            OpticalConstants=[list(scenario.sphere_oc), list(scenario.matrix_oc)],
            Energy=float(scenario.energy_ev),
        )
    if not optical_constants.validate():
        raise AssertionError("CyRSoXS optical-constants validation failed.")

    voxel_data = cy.VoxelData(InputData=input_data)
    voxel_data.addVoxelData(Vfrac=sphere_vfrac, MaterialID=1)
    voxel_data.addVoxelData(Vfrac=matrix_vfrac, MaterialID=2)
    if not voxel_data.validate():
        raise AssertionError("CyRSoXS VoxelData validation failed.")

    scattering_pattern = cy.ScatteringPattern(InputData=input_data)
    with cy.ostream_redirect(stdout=False, stderr=False):
        cy.launch(
            VoxelData=voxel_data,
            RefractiveIndexData=optical_constants,
            InputData=input_data,
            ScatteringPattern=scattering_pattern,
        )

    data = _scattering_to_xarray(scattering_pattern, energies_eV=energies_eV).copy(deep=True)

    del scattering_pattern, voxel_data, optical_constants, input_data
    del sphere_vfrac, matrix_vfrac
    gc.collect()
    return data


def _pyhyper_iq_by_energy(scattering) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    from PyHyperScattering.integrate import WPIntegrator

    integrator = WPIntegrator(use_chunked_processing=False)
    remeshed = integrator.integrateImageStack(scattering)

    if "energy" not in remeshed.dims or "chi" not in remeshed.dims:
        raise AssertionError("PyHyperScattering output missing expected energy/chi dimensions.")
    qdim = next((d for d in remeshed.dims if d == "q" or d.startswith("q")), None)
    if qdim is None:
        raise AssertionError("PyHyperScattering output missing q dimension.")

    iq_by_energy = {}
    for energy in np.asarray(remeshed.coords["energy"].values, dtype=np.float64):
        iq = remeshed.sel(energy=float(energy)).mean("chi")
        q = np.asarray(iq.coords[qdim].values, dtype=np.float64)
        sim_iq = np.asarray(iq.values, dtype=np.float64)
        iq_by_energy[round(float(energy), 5)] = (q, sim_iq)
    return iq_by_energy


def _integrated_intensity_metrics(q: np.ndarray, iq: np.ndarray) -> dict[str, float]:
    mask = np.logical_and.reduce(
        [
            q >= Q_INTEGRATE_MIN,
            q <= Q_INTEGRATE_MAX,
            np.isfinite(q),
            np.isfinite(iq),
            iq >= 0.0,
        ]
    )
    if int(np.count_nonzero(mask)) < 20:
        raise AssertionError("Insufficient q points for contrast-scaling integration.")
    q_use = np.asarray(q[mask], dtype=np.float64)
    iq_use = np.asarray(iq[mask], dtype=np.float64)
    return {
        "weighted_qiq": float(np.trapezoid(q_use * iq_use, q_use)),
        "unweighted_iq": float(np.trapezoid(iq_use, q_use)),
        "n_comp": int(q_use.size),
    }


def _build_scenarios() -> list[ContrastScenario]:
    scenarios = []
    family_order = [
        "beta_vac",
        "delta_pos_vac",
        "delta_neg_vac",
        "mixed_vac",
        "beta_split",
        "delta_pos_split",
        "delta_neg_split",
        "mixed_split",
    ]
    energy_index = 0
    for family in family_order:
        for magnitude in CONTRAST_MAGNITUDES:
            energy_ev = round(ENERGY_START_EV + ENERGY_STEP_EV * energy_index, 5)
            energy_index += 1
            a = float(magnitude)
            if family == "beta_vac":
                sphere_oc = (0.0, a, 0.0, a)
                matrix_oc = (0.0, 0.0, 0.0, 0.0)
                delta_beta = a
                delta_delta = 0.0
            elif family == "delta_pos_vac":
                sphere_oc = (a, 0.0, a, 0.0)
                matrix_oc = (0.0, 0.0, 0.0, 0.0)
                delta_beta = 0.0
                delta_delta = a
            elif family == "delta_neg_vac":
                sphere_oc = (-a, 0.0, -a, 0.0)
                matrix_oc = (0.0, 0.0, 0.0, 0.0)
                delta_beta = 0.0
                delta_delta = -a
            elif family == "mixed_vac":
                sphere_oc = (a, a, a, a)
                matrix_oc = (0.0, 0.0, 0.0, 0.0)
                delta_beta = a
                delta_delta = a
            elif family == "beta_split":
                sphere_oc = (0.0, 1.5 * a, 0.0, 1.5 * a)
                matrix_oc = (0.0, 0.5 * a, 0.0, 0.5 * a)
                delta_beta = a
                delta_delta = 0.0
            elif family == "delta_pos_split":
                sphere_oc = (0.5 * a, 0.0, 0.5 * a, 0.0)
                matrix_oc = (-0.5 * a, 0.0, -0.5 * a, 0.0)
                delta_beta = 0.0
                delta_delta = a
            elif family == "delta_neg_split":
                sphere_oc = (-0.5 * a, 0.0, -0.5 * a, 0.0)
                matrix_oc = (0.5 * a, 0.0, 0.5 * a, 0.0)
                delta_beta = 0.0
                delta_delta = -a
            elif family == "mixed_split":
                sphere_oc = (0.5 * a, 1.5 * a, 0.5 * a, 1.5 * a)
                matrix_oc = (-0.5 * a, 0.5 * a, -0.5 * a, 0.5 * a)
                delta_beta = a
                delta_delta = a
            else:
                raise AssertionError(f"Unknown family: {family}")
            scenarios.append(
                ContrastScenario(
                    family=family,
                    label=f"{family}_a{int(round(a * 1e4))}",
                    energy_ev=energy_ev,
                    magnitude=a,
                    sphere_oc=sphere_oc,
                    matrix_oc=matrix_oc,
                    delta_beta=delta_beta,
                    delta_delta=delta_delta,
                )
            )
    return scenarios


def _write_family_plot(
    family: str,
    rows: list[dict[str, float]],
) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    expected = np.asarray([row["expected_ratio"] for row in rows], dtype=np.float64)
    weighted = np.asarray([row["weighted_ratio"] for row in rows], dtype=np.float64)
    unweighted = np.asarray([row["unweighted_ratio"] for row in rows], dtype=np.float64)

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.4),
        gridspec_kw={"height_ratios": [3.0, 1.6]},
    )
    color = FAMILY_COLORS[family]
    lim = max(1.05 * np.max(expected), 1.05 * np.max(weighted), 1.05 * np.max(unweighted), 1.1)
    ax0.plot([0.0, lim], [0.0, lim], color="gray", linestyle=":", linewidth=1.0)
    ax0.scatter(expected, weighted, color=color, s=55, alpha=0.9, label=r"$\int q I(q)\,dq$")
    ax0.scatter(
        expected,
        unweighted,
        facecolors="none",
        edgecolors=color,
        s=55,
        alpha=0.95,
        label=r"$\int I(q)\,dq$",
    )
    for row in rows:
        ax0.annotate(
            f"{int(round(row['magnitude'] * 1e4))}",
            (row["expected_ratio"], row["weighted_ratio"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    ax0.set_xlim(0.0, lim)
    ax0.set_ylim(0.0, lim)
    ax0.set_xlabel("Expected ratio vs beta_vac 1e-4")
    ax0.set_ylabel("Observed ratio")
    ax0.set_title(f"Sphere Contrast Scaling: {family}")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best", fontsize=8.5)

    ax1.axis("off")
    col_labels = [
        "mag",
        "energy",
        "exp",
        "qIq",
        "Iq",
        "qIq err",
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                f"{row['magnitude']:.4g}",
                f"{row['energy_ev']:.2f}",
                f"{row['expected_ratio']:.3f}",
                f"{row['weighted_ratio']:.3f}",
                f"{row['unweighted_ratio']:.3f}",
                f"{row['weighted_rel_err']:.3f}",
            ]
        )
    table = ax1.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)

    fig.tight_layout()
    out = PLOT_DIR / f"{family}_contrast_scaling.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _evaluate_contrast_family_rows() -> dict[str, list[dict[str, float]]]:
    scenarios = _build_scenarios()
    data = _run_sphere_pybind(scenarios)
    iq_by_energy = _pyhyper_iq_by_energy(data)
    del data
    gc.collect()

    scenario_rows = []
    for scenario in scenarios:
        q, iq = iq_by_energy[round(float(scenario.energy_ev), 5)]
        metrics = _integrated_intensity_metrics(q, iq)
        scenario_rows.append(
            {
                "family": scenario.family,
                "label": scenario.label,
                "energy_ev": scenario.energy_ev,
                "magnitude": scenario.magnitude,
                "expected_contrast_sq": scenario.expected_contrast_sq,
                **metrics,
            }
        )

    ref_label = "beta_vac_a1"
    ref_row = next((row for row in scenario_rows if row["label"] == ref_label), None)
    if ref_row is None:
        raise AssertionError(f"Reference scenario {ref_label} not found.")

    ref_weighted = ref_row["weighted_qiq"]
    ref_unweighted = ref_row["unweighted_iq"]
    ref_expected = ref_row["expected_contrast_sq"]
    if ref_weighted <= 0.0 or ref_unweighted <= 0.0 or ref_expected <= 0.0:
        raise AssertionError("Reference integrated intensity must be positive.")

    family_rows = {}
    for row in scenario_rows:
        row["expected_ratio"] = row["expected_contrast_sq"] / ref_expected
        row["weighted_ratio"] = row["weighted_qiq"] / ref_weighted
        row["unweighted_ratio"] = row["unweighted_iq"] / ref_unweighted
        row["weighted_rel_err"] = abs(row["weighted_ratio"] - row["expected_ratio"]) / row["expected_ratio"]
        row["unweighted_rel_err"] = abs(row["unweighted_ratio"] - row["expected_ratio"]) / row["expected_ratio"]
        if row["weighted_ratio"] > 0.0:
            row["integral_consistency_rel"] = abs(row["unweighted_ratio"] - row["weighted_ratio"]) / row["weighted_ratio"]
        else:
            row["integral_consistency_rel"] = np.inf
        family_rows.setdefault(row["family"], []).append(row)

    for rows in family_rows.values():
        rows.sort(key=lambda row: row["magnitude"])
    return family_rows


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.physics_validation
@pytest.mark.toolchain_validation
def test_sphere_contrast_scaling_pybind():
    """Validate quadratic contrast scaling for a 70 nm sphere across beta, delta, mixed, and split-material cases."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for sphere contrast-scaling validation.")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    assert len(_build_scenarios()) == 24
    family_rows = _evaluate_contrast_family_rows()

    for family, rows in family_rows.items():
        if WRITE_VALIDATION_PLOTS:
            _write_family_plot(family, rows)
        magnitudes = [row["magnitude"] for row in rows]
        assert magnitudes == sorted(magnitudes)
        weighted_ratios = [row["weighted_ratio"] for row in rows]
        assert weighted_ratios[0] < weighted_ratios[1] < weighted_ratios[2]

    pairings = [
        ("delta_pos_vac", "delta_neg_vac"),
        ("delta_pos_split", "delta_neg_split"),
        ("beta_vac", "beta_split"),
        ("delta_pos_vac", "delta_pos_split"),
        ("delta_neg_vac", "delta_neg_split"),
        ("mixed_vac", "mixed_split"),
    ]
    for left_family, right_family in pairings:
        left_rows = family_rows[left_family]
        right_rows = family_rows[right_family]
        if len(left_rows) != len(right_rows):
            raise AssertionError(f"Family length mismatch: {left_family} vs {right_family}")
        for left_row, right_row in zip(left_rows, right_rows):
            rel = abs(left_row["weighted_ratio"] - right_row["weighted_ratio"]) / left_row["weighted_ratio"]
            assert rel < CONTRAST_PAIRING_REL_ERR_MAX, (
                f"{left_family} vs {right_family} mismatch at magnitude {left_row['magnitude']}: {rel:.3f}"
            )

    for family, rows in family_rows.items():
        for row in rows:
            assert row["weighted_rel_err"] < CONTRAST_WEIGHTED_REL_ERR_MAX, f"{row['label']} weighted ratio error too large"
            assert row["unweighted_rel_err"] < CONTRAST_UNWEIGHTED_REL_ERR_MAX, f"{row['label']} unweighted ratio error too large"
            assert (
                row["integral_consistency_rel"] < CONTRAST_INTEGRAL_CONSISTENCY_REL_MAX
            ), f"{row['label']} weighted/unweighted mismatch too large"

    for family in sorted(family_rows):
        print(f"family={family}")
        for row in family_rows[family]:
            print(
                "  "
                f"mag={row['magnitude']:.4g} "
                f"exp={row['expected_ratio']:.3f} "
                f"qIq={row['weighted_ratio']:.3f} "
                f"Iq={row['unweighted_ratio']:.3f} "
                f"qIq_err={row['weighted_rel_err']:.3f} "
                f"Iq_err={row['unweighted_rel_err']:.3f}"
            )
