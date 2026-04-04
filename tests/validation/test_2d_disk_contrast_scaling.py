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
import xarray as xr

from NRSS import SFieldMode
from NRSS.morphology import Material, Morphology
from tests.path_matrix import ComputationPath


pytestmark = [pytest.mark.path_matrix, pytest.mark.reference_parity]


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = REPO_ROOT / "test-reports" / "disk-2d-contrast-scaling-dev"
WRITE_VALIDATION_PLOTS = os.environ.get("NRSS_WRITE_VALIDATION_PLOTS", "").strip() == "1"

SHAPE = (1, 2048, 2048)
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
    disk_oc: tuple[float, float, float, float]
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


def _disk_and_matrix_vfrac() -> tuple[np.ndarray, np.ndarray]:
    _, ny, nx = SHAPE
    radius_vox = float(DIAMETER_NM) / (2.0 * PHYS_SIZE_NM)
    cy0 = (ny - 1) / 2.0
    cx0 = (nx - 1) / 2.0

    pad = radius_vox + 2.0
    y0 = max(0, int(np.floor(cy0 - pad)))
    y1 = min(ny, int(np.ceil(cy0 + pad)) + 1)
    x0 = max(0, int(np.floor(cx0 - pad)))
    x1 = min(nx, int(np.ceil(cx0 + pad)) + 1)

    yy, xx = np.ogrid[y0:y1, x0:x1]
    dy = yy.astype(np.float32) - np.float32(cy0)
    dx = xx.astype(np.float32) - np.float32(cx0)
    dist2 = dx * dx + dy * dy

    disk = np.zeros(SHAPE, dtype=np.float32)
    local_disk = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    local_disk[dist2 <= np.float32(radius_vox * radius_vox)] = 1.0
    disk[0, y0:y1, x0:x1] = local_disk
    matrix = (1.0 - disk).astype(np.float32)
    return disk, matrix


def _run_disk_backend(
    scenarios: list[ContrastScenario],
    runtime_kwargs: dict[str, object],
) -> xr.DataArray:
    disk_vfrac, matrix_vfrac = _disk_and_matrix_vfrac()
    energies_eV = [scenario.energy_ev for scenario in scenarios]
    field_namespace = str(runtime_kwargs["field_namespace"])
    disk_vfrac = _to_backend_namespace(disk_vfrac, field_namespace)
    matrix_vfrac = _to_backend_namespace(matrix_vfrac, field_namespace)

    mat1 = Material(
        materialID=1,
        Vfrac=disk_vfrac,
        S=SFieldMode.ISOTROPIC,
        theta=None,
        psi=None,
        energies=energies_eV,
        opt_constants={float(s.energy_ev): list(s.disk_oc) for s in scenarios},
        name="disk",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=matrix_vfrac,
        S=SFieldMode.ISOTROPIC,
        theta=None,
        psi=None,
        energies=energies_eV,
        opt_constants={float(s.energy_ev): list(s.matrix_oc) for s in scenarios},
        name="matrix",
    )
    config = {
        "CaseType": 0,
        "MorphologyType": 0,
        "Energies": energies_eV,
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
        data = morph.run(stdout=False, stderr=False, return_xarray=True)
        if data is None:
            raise AssertionError("2D disk contrast-scaling backend run returned no scattering data.")
        return data.copy(deep=True)
    finally:
        morph.release_runtime()
        del morph
        del disk_vfrac, matrix_vfrac
        _release_runtime_memory()


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
        raise AssertionError("Insufficient q points for 2D disk contrast-scaling integration.")
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
                disk_oc = (0.0, a, 0.0, a)
                matrix_oc = (0.0, 0.0, 0.0, 0.0)
                delta_beta = a
                delta_delta = 0.0
            elif family == "delta_pos_vac":
                disk_oc = (a, 0.0, a, 0.0)
                matrix_oc = (0.0, 0.0, 0.0, 0.0)
                delta_beta = 0.0
                delta_delta = a
            elif family == "delta_neg_vac":
                disk_oc = (-a, 0.0, -a, 0.0)
                matrix_oc = (0.0, 0.0, 0.0, 0.0)
                delta_beta = 0.0
                delta_delta = -a
            elif family == "mixed_vac":
                disk_oc = (a, a, a, a)
                matrix_oc = (0.0, 0.0, 0.0, 0.0)
                delta_beta = a
                delta_delta = a
            elif family == "beta_split":
                disk_oc = (0.0, 1.5 * a, 0.0, 1.5 * a)
                matrix_oc = (0.0, 0.5 * a, 0.0, 0.5 * a)
                delta_beta = a
                delta_delta = 0.0
            elif family == "delta_pos_split":
                disk_oc = (0.5 * a, 0.0, 0.5 * a, 0.0)
                matrix_oc = (-0.5 * a, 0.0, -0.5 * a, 0.0)
                delta_beta = 0.0
                delta_delta = a
            elif family == "delta_neg_split":
                disk_oc = (-0.5 * a, 0.0, -0.5 * a, 0.0)
                matrix_oc = (0.5 * a, 0.0, 0.5 * a, 0.0)
                delta_beta = 0.0
                delta_delta = -a
            elif family == "mixed_split":
                disk_oc = (0.5 * a, 1.5 * a, 0.5 * a, 1.5 * a)
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
                    disk_oc=disk_oc,
                    matrix_oc=matrix_oc,
                    delta_beta=delta_beta,
                    delta_delta=delta_delta,
                )
            )
    return scenarios


def _scenario_batches(scenarios: list[ContrastScenario]) -> list[list[ContrastScenario]]:
    reference_label = "beta_vac_a1"
    reference = next((scenario for scenario in scenarios if scenario.label == reference_label), None)
    if reference is None:
        raise AssertionError(f"Reference scenario {reference_label} not found.")

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
    batches = []
    for family in family_order:
        batch = [scenario for scenario in scenarios if scenario.family == family]
        if not batch:
            raise AssertionError(f"No scenarios found for family {family}.")
        if family != "beta_vac":
            batch = [reference] + batch
        batches.append(batch)
    return batches


def _scenario_to_payload(scenario: ContrastScenario) -> dict[str, object]:
    return {
        "family": scenario.family,
        "label": scenario.label,
        "energy_ev": float(scenario.energy_ev),
        "magnitude": float(scenario.magnitude),
        "disk_oc": list(scenario.disk_oc),
        "matrix_oc": list(scenario.matrix_oc),
        "delta_beta": float(scenario.delta_beta),
        "delta_delta": float(scenario.delta_delta),
    }


def _scenario_from_payload(payload: dict[str, object]) -> ContrastScenario:
    return ContrastScenario(
        family=str(payload["family"]),
        label=str(payload["label"]),
        energy_ev=float(payload["energy_ev"]),
        magnitude=float(payload["magnitude"]),
        disk_oc=tuple(float(v) for v in payload["disk_oc"]),
        matrix_oc=tuple(float(v) for v in payload["matrix_oc"]),
        delta_beta=float(payload["delta_beta"]),
        delta_delta=float(payload["delta_delta"]),
    )


def _write_family_plot(
    family: str,
    rows: list[dict[str, float]],
    path_id: str,
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
    ax0.set_title(f"2D Disk Contrast Scaling: {family}")
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
        "Iq err",
        "agree",
    ]
    table_rows = []
    for row in rows:
        agree = (
            row["weighted_rel_err"] < CONTRAST_WEIGHTED_REL_ERR_MAX
            and row["unweighted_rel_err"] < CONTRAST_UNWEIGHTED_REL_ERR_MAX
            and row["integral_consistency_rel"] < CONTRAST_INTEGRAL_CONSISTENCY_REL_MAX
        )
        table_rows.append(
            [
                f"{row['magnitude']:.4g}",
                f"{row['energy_ev']:.2f}",
                f"{row['expected_ratio']:.3f}",
                f"{row['weighted_ratio']:.3f}",
                f"{row['unweighted_ratio']:.3f}",
                f"{row['weighted_rel_err']:.3f}",
                f"{row['unweighted_rel_err']:.3f}",
                "pass" if agree else "fail",
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
    out = PLOT_DIR / f"{path_id}__{family}_contrast_scaling.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _write_summary_table_plot(
    family_rows: dict[str, list[dict[str, float]]],
    path_id: str,
) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.5, 9.0))
    ax.axis("off")

    col_labels = [
        "family",
        "mag",
        "energy",
        "exp",
        "qIq",
        "Iq",
        "qIq err",
        "Iq err",
        "cons err",
        "agree",
    ]
    table_rows = []
    row_colors = []
    for family in sorted(family_rows):
        for row in family_rows[family]:
            agree = (
                row["weighted_rel_err"] < CONTRAST_WEIGHTED_REL_ERR_MAX
                and row["unweighted_rel_err"] < CONTRAST_UNWEIGHTED_REL_ERR_MAX
                and row["integral_consistency_rel"] < CONTRAST_INTEGRAL_CONSISTENCY_REL_MAX
            )
            table_rows.append(
                [
                    family,
                    f"{row['magnitude']:.4g}",
                    f"{row['energy_ev']:.2f}",
                    f"{row['expected_ratio']:.3f}",
                    f"{row['weighted_ratio']:.3f}",
                    f"{row['unweighted_ratio']:.3f}",
                    f"{row['weighted_rel_err']:.3f}",
                    f"{row['unweighted_rel_err']:.3f}",
                    f"{row['integral_consistency_rel']:.3f}",
                    "pass" if agree else "fail",
                ]
            )
            row_colors.append("#e8f5e9" if agree else "#ffebee")

    table = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)

    for col in range(len(col_labels)):
        table[(0, col)].set_facecolor("#d9e6f2")
    for row_idx, color in enumerate(row_colors, start=1):
        for col in range(len(col_labels)):
            table[(row_idx, col)].set_facecolor(color)

    fig.suptitle("2D Disk Contrast Scaling Agreement Summary")
    fig.text(
        0.01,
        0.02,
        (
            f"Agreement thresholds: qIq err < {CONTRAST_WEIGHTED_REL_ERR_MAX:.3f}, "
            f"Iq err < {CONTRAST_UNWEIGHTED_REL_ERR_MAX:.3f}, "
            f"consistency err < {CONTRAST_INTEGRAL_CONSISTENCY_REL_MAX:.4f}"
        ),
        fontsize=9,
        va="bottom",
        ha="left",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    out = PLOT_DIR / f"{path_id}__contrast_scaling_agreement_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _evaluate_contrast_batch_rows(
    batch: list[ContrastScenario],
    nrss_path: ComputationPath,
) -> list[dict[str, float]]:
    data = _run_disk_backend(batch, runtime_kwargs=_path_runtime_kwargs(nrss_path))
    iq_by_energy = _pyhyper_iq_by_energy(data)
    del data
    _release_runtime_memory()

    batch_rows = []
    for scenario in batch:
        q, iq = iq_by_energy[round(float(scenario.energy_ev), 5)]
        metrics = _integrated_intensity_metrics(q, iq)
        batch_rows.append(
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
    ref_row = next((row for row in batch_rows if row["label"] == ref_label), None)
    if ref_row is None:
        raise AssertionError(f"Reference scenario {ref_label} not found.")

    ref_weighted = ref_row["weighted_qiq"]
    ref_unweighted = ref_row["unweighted_iq"]
    ref_expected = ref_row["expected_contrast_sq"]
    if ref_weighted <= 0.0 or ref_unweighted <= 0.0 or ref_expected <= 0.0:
        raise AssertionError("Reference integrated intensity must be positive.")

    for row in batch_rows:
        row["expected_ratio"] = row["expected_contrast_sq"] / ref_expected
        row["weighted_ratio"] = row["weighted_qiq"] / ref_weighted
        row["unweighted_ratio"] = row["unweighted_iq"] / ref_unweighted
        row["weighted_rel_err"] = abs(row["weighted_ratio"] - row["expected_ratio"]) / row["expected_ratio"]
        row["unweighted_rel_err"] = abs(row["unweighted_ratio"] - row["expected_ratio"]) / row["expected_ratio"]
        if row["weighted_ratio"] > 0.0:
            row["integral_consistency_rel"] = abs(row["unweighted_ratio"] - row["weighted_ratio"]) / row["weighted_ratio"]
        else:
            row["integral_consistency_rel"] = np.inf
    return batch_rows


def _evaluate_contrast_family_rows(nrss_path: ComputationPath) -> dict[str, list[dict[str, float]]]:
    scenarios = _build_scenarios()
    scenario_rows = []
    for batch_index, batch in enumerate(_scenario_batches(scenarios)):
        batch_rows = _evaluate_contrast_batch_rows(batch, nrss_path)
        for row in batch_rows:
            if row["label"] == "beta_vac_a1" and batch_index > 0:
                continue
            scenario_rows.append(row)

    family_rows = {}
    for row in scenario_rows:
        family_rows.setdefault(row["family"], []).append(row)

    for rows in family_rows.values():
        rows.sort(key=lambda row: row["magnitude"])
    return family_rows


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.physics_validation
@pytest.mark.toolchain_validation
def test_2d_disk_contrast_scaling_pybind(nrss_path: ComputationPath):
    """Validate quadratic contrast scaling for a 70 nm 2D disk across beta, delta, mixed, and split-material cases."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for 2D disk contrast-scaling validation.")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    assert len(_build_scenarios()) == 24
    family_rows = _evaluate_contrast_family_rows(nrss_path)

    for family, rows in family_rows.items():
        if WRITE_VALIDATION_PLOTS:
            _write_family_plot(family, rows, path_id=nrss_path.id)
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

    if WRITE_VALIDATION_PLOTS:
        _write_summary_table_plot(family_rows, path_id=nrss_path.id)

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
