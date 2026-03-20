import importlib
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from NRSS.reader import read_config
from NRSS.morphology import Material, Morphology, OpticalConstants
from NRSS.writer import write_config


pytestmark = pytest.mark.smoke


def _import_required(module_name: str):
    mod = importlib.import_module(module_name)
    assert mod is not None
    return mod


def _import_cyrsoxs_required():
    # CyRSoXS is required for the current backend.
    # If/when NRSS supports multiple independent backends, this check can become optional.
    errors = []
    for name in ("CyRSoXS", "cyrsoxs"):
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - exercised when import fails
            errors.append(f"{name}: {exc.__class__.__name__}({exc})")
    raise AssertionError(
        "CyRSoXS import failed for listed attempts below. "
        f"Attempts: {'; '.join(errors)}"
    )


def _tiny_smoothing_kernel(arr: np.ndarray) -> np.ndarray:
    return (
        0.25 * arr
        + 0.125 * np.roll(arr, 1, axis=0)
        + 0.125 * np.roll(arr, -1, axis=0)
        + 0.125 * np.roll(arr, 1, axis=1)
        + 0.125 * np.roll(arr, -1, axis=1)
        + 0.0625 * np.roll(np.roll(arr, 1, axis=0), 1, axis=1)
        + 0.0625 * np.roll(np.roll(arr, 1, axis=0), -1, axis=1)
        + 0.0625 * np.roll(np.roll(arr, -1, axis=0), 1, axis=1)
        + 0.0625 * np.roll(np.roll(arr, -1, axis=0), -1, axis=1)
    )


def test_required_imports():
    """Verify core runtime dependencies import, including required CyRSoXS bindings."""
    for name in (
        "NRSS",
        "pytest",
        "numpy",
        "scipy",
        "pandas",
        "h5py",
        "xarray",
        "PyHyperScattering",
    ):
        _import_required(name)
    _import_cyrsoxs_required()


def test_optional_cupy_is_non_blocking():
    """Confirm missing CuPy does not fail smoke tests because it is optional."""
    try:
        importlib.import_module("cupy")
    except Exception:
        # cupy is currently optional for smoke; absence should not fail this suite.
        pass


def test_tiny_deterministic_white_noise_kernel():
    """Check deterministic numeric fingerprint for a tiny white-noise smoothing kernel."""
    rng = np.random.default_rng(seed=20260317)
    white_noise = rng.standard_normal((16, 16))
    smoothed = _tiny_smoothing_kernel(white_noise)

    assert smoothed.shape == (16, 16)
    assert np.isfinite(smoothed).all()
    assert np.isclose(float(smoothed.mean()), 0.03638822141137503, atol=1e-12)
    assert np.isclose(float(smoothed.std()), 0.3639495657715855, atol=1e-12)
    assert np.isclose(float(smoothed[0, 0]), 0.15636713796507423, atol=1e-12)


def test_small_hdf5_roundtrip(tmp_path: Path):
    """Validate basic HDF5 write/read roundtrip integrity."""
    arr = np.arange(64, dtype=np.float64).reshape(8, 8)
    test_h5 = tmp_path / "smoke_roundtrip.h5"

    with h5py.File(test_h5, "w") as h5f:
        h5f.create_dataset("arr", data=arr)

    with h5py.File(test_h5, "r") as h5f:
        loaded = h5f["arr"][()]

    assert np.array_equal(loaded, arr)


def test_write_and_read_config_roundtrip(tmp_path: Path):
    """Ensure NRSS config writer/reader roundtrip preserves key fields."""
    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        write_config(
            Energies=[280.0, 281.0],
            EAngleRotation=[0.0, 0.0, 0.0],
            CaseType=0,
            MorphologyType=0,
            NumThreads=1,
            AlgorithmType=0,
            DumpMorphology=False,
            ScatterApproach=0,
            WindowingType=0,
            RotMask=False,
            EwaldsInterpolation=1,
        )
        parsed = read_config("config.txt")
    finally:
        os.chdir(cwd)

    assert parsed["CaseType"] == 0
    assert parsed["MorphologyType"] == 0
    assert parsed["NumThreads"] == 1
    assert parsed["Energies"] == [280.0, 281.0]
    assert parsed["EAngleRotation"] == [0.0, 0.0, 0.0]


def test_pybind_morphology_object_lifecycle_smoke():
    """Exercise pybind object creation/update/validation without full simulation workflow."""
    # Tiny deterministic morphology to exercise the pybind wiring without launching a full simulation.
    energies = [285.0]
    shape = (1, 8, 8)
    zeros = np.zeros(shape, dtype=np.float32)

    mat1 = Material(
        materialID=1,
        Vfrac=np.ones(shape, dtype=np.float32),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="vacuum_1",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=zeros.copy(),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="vacuum_2",
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
        PhysSize=5.0,
        config=config,
        create_cy_object=True,
    )

    morph.create_update_cy()

    assert morph.inputData is not None
    assert morph.OpticalConstants is not None
    assert morph.voxelData is not None
    assert morph.inputData.validate()
    assert morph.OpticalConstants.validate()
    assert morph.voxelData.validate()


def test_vacuum_named_matches_explicit_zero_constants():
    """Verify named vacuum optical constants match explicit all-zero constants."""
    energies = [285.0, 286.0]
    shape = (1, 4, 4)
    zeros = np.zeros(shape, dtype=np.float32)
    ones = np.ones(shape, dtype=np.float32)

    mat1 = Material(
        materialID=1,
        Vfrac=ones.copy(),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={e: [1e-4, 2e-4, 1e-4, 2e-4] for e in energies},
        name="poly",
    )
    vacuum_named = Material(
        materialID=2,
        Vfrac=zeros.copy(),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        name="vacuum",
    )
    vacuum_explicit = Material(
        materialID=2,
        Vfrac=zeros.copy(),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={e: [0.0, 0.0, 0.0, 0.0] for e in energies},
        name="vacuum_explicit",
    )

    assert vacuum_named.opt_constants == vacuum_explicit.opt_constants

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

    morph_named = Morphology(
        2,
        materials={1: mat1.copy(), 2: vacuum_named},
        PhysSize=5.0,
        config=config,
        create_cy_object=True,
    )
    morph_explicit = Morphology(
        2,
        materials={1: mat1.copy(), 2: vacuum_explicit},
        PhysSize=5.0,
        config=config,
        create_cy_object=True,
    )

    morph_named.check_materials(quiet=True)
    morph_explicit.check_materials(quiet=True)
    morph_named.validate_all(quiet=True)
    morph_explicit.validate_all(quiet=True)

    for e in energies:
        named_vals = morph_named.materials[2].opt_constants[e]
        explicit_vals = morph_explicit.materials[2].opt_constants[e]
        assert np.allclose(named_vals, explicit_vals, atol=0.0)


def test_optical_constants_calc_and_load_matfile_smoke(tmp_path: Path):
    """Smoke-test optical constant interpolation and MaterialX.txt parsing."""
    reference_data = {
        "Energy": [100.0, 200.0],
        "DeltaPara": [1.0, 3.0],
        "BetaPara": [2.0, 4.0],
        "DeltaPerp": [5.0, 7.0],
        "BetaPerp": [6.0, 8.0],
    }
    energies = [100.0, 150.0, 200.0]
    calc = OpticalConstants.calc_constants(energies, reference_data, name="interp")

    assert np.allclose(calc.opt_constants[100.0], [1.0, 2.0, 5.0, 6.0], atol=1e-12)
    assert np.allclose(calc.opt_constants[150.0], [2.0, 3.0, 6.0, 7.0], atol=1e-12)
    assert np.allclose(calc.opt_constants[200.0], [3.0, 4.0, 7.0, 8.0], atol=1e-12)

    matfile = tmp_path / "Material1.txt"
    matfile.write_text(
        "\n".join(
            [
                "Energy = 100.0;",
                "BetaPara = 2.0;",
                "BetaPerp = 6.0;",
                "DeltaPara = 1.0;",
                "DeltaPerp = 5.0;",
                "Energy = 200.0;",
                "BetaPara = 4.0;",
                "BetaPerp = 8.0;",
                "DeltaPara = 3.0;",
                "DeltaPerp = 7.0;",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = OpticalConstants.load_matfile(str(matfile), name="from_file")
    assert loaded.energies == [100.0, 200.0]
    assert np.allclose(loaded.opt_constants[100.0], [1.0, 2.0, 5.0, 6.0], atol=1e-12)
    assert np.allclose(loaded.opt_constants[200.0], [3.0, 4.0, 7.0, 8.0], atol=1e-12)


def test_morphology_validation_fails_when_total_vfrac_exceeds_one():
    """Assert morphology validator rejects voxels where total volume fraction exceeds one."""
    energies = [285.0]
    shape = (1, 4, 4)
    zeros = np.zeros(shape, dtype=np.float32)
    vfrac_1 = np.full(shape, 0.6, dtype=np.float32)
    vfrac_2 = np.full(shape, 0.6, dtype=np.float32)

    assert np.all(vfrac_1 <= 1.0)
    assert np.all(vfrac_2 <= 1.0)
    assert np.all(vfrac_1 + vfrac_2 > 1.0)

    mat1 = Material(
        materialID=1,
        Vfrac=vfrac_1,
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [1e-4, 2e-4, 1e-4, 2e-4]},
        name="mat1",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=vfrac_2,
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="mat2",
    )

    morph = Morphology(
        2,
        materials={1: mat1, 2: mat2},
        PhysSize=5.0,
        create_cy_object=False,
    )

    with pytest.raises(
        AssertionError,
        match="Total material volume fractions do not sum to 1",
    ):
        morph.check_materials(quiet=True)


def test_morphology_validation_fails_on_nan_field():
    """Assert morphology validator rejects NaN values in material fields."""
    energies = [285.0]
    shape = (1, 4, 4)
    zeros = np.zeros(shape, dtype=np.float32)
    ones = np.ones(shape, dtype=np.float32)
    bad_s = zeros.copy()
    bad_s[0, 0, 0] = np.nan

    mat1 = Material(
        materialID=1,
        Vfrac=ones.copy(),
        S=bad_s,
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [1e-4, 2e-4, 1e-4, 2e-4]},
        name="mat1",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=zeros.copy(),
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="mat2",
    )
    morph = Morphology(2, materials={1: mat1, 2: mat2}, PhysSize=5.0, create_cy_object=False)

    with pytest.raises(AssertionError, match="NaNs are present in Material 1 S"):
        morph.check_materials(quiet=True)


def test_morphology_validation_fails_on_non_float_field():
    """Assert morphology validator rejects non-float material arrays."""
    energies = [285.0]
    shape = (1, 4, 4)
    zeros_f = np.zeros(shape, dtype=np.float32)
    zeros_i = np.zeros(shape, dtype=np.int32)
    ones = np.ones(shape, dtype=np.float32)

    mat1 = Material(
        materialID=1,
        Vfrac=ones.copy(),
        S=zeros_f.copy(),
        theta=zeros_i,
        psi=zeros_f.copy(),
        energies=energies,
        opt_constants={285.0: [1e-4, 2e-4, 1e-4, 2e-4]},
        name="mat1",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=zeros_f.copy(),
        S=zeros_f.copy(),
        theta=zeros_f.copy(),
        psi=zeros_f.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="mat2",
    )
    morph = Morphology(2, materials={1: mat1, 2: mat2}, PhysSize=5.0, create_cy_object=False)

    with pytest.raises(AssertionError, match="Material 1 theta is not of type float"):
        morph.check_materials(quiet=True)


@pytest.mark.cpu
def test_visualizer_two_material_outputs_and_summary_are_consistent(capsys, monkeypatch):
    """Check two-material visualizer output count/shape and printed summary consistency."""
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    energies = [285.0]
    shape = (1, 16, 16)
    zeros = np.zeros(shape, dtype=np.float32)

    zz, yy, xx = np.indices(shape)
    vfrac_1 = ((zz + yy + xx) % 2).astype(np.float32)
    vfrac_2 = 1.0 - vfrac_1

    mat1 = Material(
        materialID=1,
        Vfrac=vfrac_1,
        S=(0.5 * vfrac_1).astype(np.float32),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [2e-4, 1e-4, 2e-4, 1e-4]},
        name="mat1",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=vfrac_2,
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={285.0: [0.0, 0.0, 0.0, 0.0]},
        name="mat2",
    )

    morph = Morphology(
        2,
        materials={1: mat1, 2: mat2},
        PhysSize=5.0,
        create_cy_object=False,
    )

    # Ensure generated morphology matches the intended two-material construction.
    assert np.allclose(morph.materials[1].Vfrac + morph.materials[2].Vfrac, 1.0)
    expected_line_1 = (
        f"Material 1 Vfrac. Min: {morph.materials[1].Vfrac.min()} "
        f"Max: {morph.materials[1].Vfrac.max()}"
    )
    expected_line_2 = (
        f"Material 2 Vfrac. Min: {morph.materials[2].Vfrac.min()} "
        f"Max: {morph.materials[2].Vfrac.max()}"
    )

    images = morph.visualize_materials(
        z_slice=0,
        subsample=16,
        outputmat=[1, 2],
        outputplot=["vfrac", "S"],
        outputaxes=True,
        runquiet=False,
        batchMode=True,
    )
    stdout = capsys.readouterr().out

    assert "Number of Materials: 2" in stdout
    assert expected_line_1 in stdout
    assert expected_line_2 in stdout
    assert len(images) == 4
    for image in images:
        assert isinstance(image, np.ndarray)
        assert image.ndim == 3
        assert image.shape[0] > 0
        assert image.shape[1] > 0
        assert image.shape[2] == 4


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


def _build_two_material_sphere_morphology(
    energies: list[float],
    shape: tuple[int, int, int] = (32, 32, 32),
    sphere_diameter_vox: int = 16,
    eangle_rotation: list[float] | None = None,
    config_overrides: dict | None = None,
) -> Morphology:
    zz, yy, xx = np.indices(shape)
    cz = (shape[0] - 1) / 2.0
    cy = (shape[1] - 1) / 2.0
    cx = (shape[2] - 1) / 2.0
    dz = zz - cz
    dy = yy - cy
    dx = xx - cx

    radius = sphere_diameter_vox / 2.0
    sphere_mask = (dx * dx + dy * dy + dz * dz) <= (radius * radius)
    vfrac_1 = sphere_mask.astype(np.float32)
    vfrac_2 = 1.0 - vfrac_1

    theta = np.arctan2(np.sqrt(dx * dx + dy * dy), dz)
    psi = np.arctan2(dy, dx)
    theta = (theta.astype(np.float32) * vfrac_1).astype(np.float32)
    psi = (psi.astype(np.float32) * vfrac_1).astype(np.float32)
    S_1 = (0.7 * vfrac_1).astype(np.float32)
    zeros = np.zeros(shape, dtype=np.float32)

    mat1 = Material(
        materialID=1,
        Vfrac=vfrac_1,
        S=S_1,
        theta=theta,
        psi=psi,
        energies=energies,
        opt_constants={e: [2e-4, 1e-4, 2e-4, 1e-4] for e in energies},
        name="sphere_material",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=vfrac_2,
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={e: [0.0, 0.0, 0.0, 0.0] for e in energies},
        name="matrix_vacuum",
    )

    if eangle_rotation is None:
        eangle_rotation = [0.0, 0.0, 0.0]

    config = {
        "CaseType": 0,
        "MorphologyType": 0,
        "Energies": energies,
        "EAngleRotation": eangle_rotation,
        "RotMask": 1,
        "WindowingType": 0,
        "AlgorithmType": 0,
        "ReferenceFrame": 1,
        "EwaldsInterpolation": 1,
    }
    if config_overrides:
        config.update(config_overrides)

    morph = Morphology(
        2,
        materials={1: mat1, 2: mat2},
        PhysSize=5.0,
        config=config,
        create_cy_object=True,
    )
    morph.check_materials(quiet=True)
    morph.validate_all(quiet=True)
    return morph


def _run_tiny_pybind_simulation(
    energies: list[float] | None = None,
    return_xarray: bool = False,
    shape: tuple[int, int, int] = (32, 32, 32),
    sphere_diameter_vox: int = 16,
):
    # Pin smoke runtime to one GPU for better stability on multi-GPU hosts.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    if energies is None:
        energies = [285.0]
    morph = _build_two_material_sphere_morphology(
        energies=energies,
        shape=shape,
        sphere_diameter_vox=sphere_diameter_vox,
    )
    scattering = morph.run(stdout=False, stderr=False, return_xarray=True)
    if return_xarray:
        return scattering
    return scattering.values.copy()


def _run_cli_from_serialized_morphology(morph: Morphology, run_path: Path) -> np.ndarray:
    from PyHyperScattering.load import cyrsoxsLoader

    run_path.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    os.chdir(run_path)
    try:
        morphology_fname = "morphology.hdf5"
        morph.write_to_file(morphology_fname, author="NRSS smoke test")
        morph.write_constants(path=".")

        # Ensure NRSS writes CLI optical-constant files with expected headings.
        for mat_id in range(1, morph.numMaterial + 1):
            material_text = (run_path / f"Material{mat_id}.txt").read_text(
                encoding="utf-8"
            )
            for token in (
                "EnergyData0",
                "Energy = ",
                "BetaPara = ",
                "BetaPerp = ",
                "DeltaPara = ",
                "DeltaPerp = ",
            ):
                assert token in material_text

        write_config(
            Energies=morph.Energies,
            EAngleRotation=morph.EAngleRotation,
            CaseType=int(morph.CaseType),
            MorphologyType=int(morph.MorphologyType),
            NumThreads=1,
            AlgorithmType=int(morph.AlgorithmType),
            DumpMorphology=False,
            ScatterApproach=0,
            WindowingType=int(morph.WindowingType),
            RotMask=bool(morph.RotMask),
            EwaldsInterpolation=int(morph.EwaldsInterpolation),
        )

        result = subprocess.run(
            ["CyRSoXS", morphology_fname],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"CyRSoXS CLI failed with return code {result.returncode}. "
            f"stdout tail: {' | '.join(result.stdout.splitlines()[-10:])} "
            f"stderr tail: {' | '.join(result.stderr.splitlines()[-10:])}"
        )

        loaded = cyrsoxsLoader(profile_time=False).loadDirectory(run_path)
        cli_vals = np.asarray(loaded.values).copy()
    finally:
        os.chdir(cwd)

    # cyrsoxsLoader returns (qx, qy, energy); align to pybind's (energy, qy, qx).
    cli_vals = np.moveaxis(cli_vals, -1, 0)
    cli_vals = np.swapaxes(cli_vals, 1, 2)
    return cli_vals.copy()


def _sanitize_scattering(arr: np.ndarray, clip_percentile: float = 99.9) -> np.ndarray:
    safe = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    safe = np.where(safe < 0.0, 0.0, safe)
    cap = float(np.percentile(safe, clip_percentile))
    if cap > 0.0:
        safe = np.clip(safe, 0.0, cap)
    return safe


def _assert_scattering_parity(
    pybind_vals: np.ndarray,
    cli_vals: np.ndarray,
    *,
    min_finite_ratio: float = 0.99,
    rtol_scalar: float = 1e-3,
    p95_abs_max: float = 8e-7,
    max_abs_max: float = 3e-5,
    p95_log_max: float = 8e-2,
    max_log_max: float = 1e-1,
) -> None:
    assert pybind_vals.shape == cli_vals.shape
    assert float(np.isfinite(pybind_vals).mean()) >= min_finite_ratio
    assert float(np.isfinite(cli_vals).mean()) >= min_finite_ratio

    pybind_safe = _sanitize_scattering(pybind_vals)
    cli_safe = _sanitize_scattering(cli_vals)
    abs_diff = np.abs(pybind_safe - cli_safe)
    signal_mask = np.logical_and(pybind_safe > 1e-12, cli_safe > 1e-12)

    assert signal_mask.any()
    log_pybind = np.log10(pybind_safe[signal_mask])
    log_cli = np.log10(cli_safe[signal_mask])
    log_abs = np.abs(log_pybind - log_cli)

    assert np.isclose(
        float(pybind_safe.sum()),
        float(cli_safe.sum()),
        rtol=rtol_scalar,
        atol=1e-12,
    )
    assert np.isclose(
        float(pybind_safe.max()),
        float(cli_safe.max()),
        rtol=rtol_scalar,
        atol=1e-12,
    )
    assert float(np.percentile(abs_diff, 95)) <= p95_abs_max
    assert float(abs_diff.max()) <= max_abs_max
    assert float(np.percentile(log_abs, 95)) <= p95_log_max
    assert float(log_abs.max()) <= max_log_max


def _build_two_material_asymmetric_lobed_morphology(
    energies: list[float],
    eangle_rotation: list[float],
) -> Morphology:
    shape = (32, 32, 32)
    zz, yy, xx = np.indices(shape)
    cz = (shape[0] - 1) / 2.0
    cy = (shape[1] - 1) / 2.0
    cx = (shape[2] - 1) / 2.0
    dz = zz - cz
    dy = yy - cy
    dx = xx - cx

    # Asymmetric morphology: one side has ~8 voxel diameter, the other ~7.
    local_radius = np.where(dx <= 0.0, 4.0, 3.5)
    vfrac_1 = ((dx * dx + dy * dy + dz * dz) <= (local_radius * local_radius)).astype(
        np.float32
    )
    vfrac_2 = 1.0 - vfrac_1

    # Intentionally non-uniform orientation fields to make E-angle sampling effects visible.
    theta_raw = (np.pi / 4.0) + 0.25 * (dx / (shape[2] / 2.0)) + 0.15 * (dy / (shape[1] / 2.0))
    theta = (np.clip(theta_raw, 0.0, np.pi).astype(np.float32) * vfrac_1).astype(np.float32)
    psi = (np.arctan2(dy + 0.3, dx - 0.2).astype(np.float32) * vfrac_1).astype(np.float32)
    s_1 = ((0.55 + 0.25 * ((yy % 3) == 0).astype(np.float32)) * vfrac_1).astype(np.float32)
    zeros = np.zeros(shape, dtype=np.float32)

    mat1 = Material(
        materialID=1,
        Vfrac=vfrac_1,
        S=s_1,
        theta=theta,
        psi=psi,
        energies=energies,
        opt_constants={e: [2e-4, 1e-4, 2e-4, 1e-4] for e in energies},
        name="asymmetric_lobe",
    )
    mat2 = Material(
        materialID=2,
        Vfrac=vfrac_2,
        S=zeros.copy(),
        theta=zeros.copy(),
        psi=zeros.copy(),
        energies=energies,
        opt_constants={e: [0.0, 0.0, 0.0, 0.0] for e in energies},
        name="vacuum",
    )

    config = {
        "CaseType": 0,
        "MorphologyType": 0,
        "Energies": energies,
        "EAngleRotation": eangle_rotation,
        "RotMask": 1,
        "WindowingType": 0,
        "AlgorithmType": 0,
        "ReferenceFrame": 1,
        "EwaldsInterpolation": 1,
    }
    morph = Morphology(
        2,
        materials={1: mat1, 2: mat2},
        PhysSize=5.0,
        config=config,
        create_cy_object=True,
    )
    morph.check_materials(quiet=True)
    morph.validate_all(quiet=True)
    return morph


def _radial_asymmetry_score(arr: np.ndarray) -> float:
    """Compute ring-wise azimuthal CV; lower score means more radial symmetry."""
    img = _sanitize_scattering(arr)[0].astype(np.float64)
    ny, nx = img.shape
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0
    yy, xx = np.indices((ny, nx))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    r_min = 2.0
    r_max = min(cy, cx)
    bins = np.linspace(r_min, r_max, 12)
    cvs = []
    weights = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = np.logical_and(rr >= lo, rr < hi)
        if int(mask.sum()) < 12:
            continue
        ring = img[mask]
        ring_mean = float(ring.mean())
        if ring_mean <= 1e-12:
            continue
        cvs.append(float(ring.std()) / ring_mean)
        weights.append(float(mask.sum()))

    assert cvs
    return float(np.average(np.asarray(cvs), weights=np.asarray(weights)))


@pytest.mark.gpu
def test_pybind_runtime_tiny_deterministic_pattern():
    """Run a tiny GPU sphere simulation and assert deterministic scalar/log similarity."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for pybind runtime smoke test.")

    arr_1 = _run_tiny_pybind_simulation()
    arr_2 = _run_tiny_pybind_simulation()
    finite_ratio_1 = float(np.isfinite(arr_1).mean())
    finite_ratio_2 = float(np.isfinite(arr_2).mean())
    arr_1_safe = _sanitize_scattering(arr_1)
    arr_2_safe = _sanitize_scattering(arr_2)

    assert arr_1.shape == (1, 32, 32)
    assert arr_2.shape == (1, 32, 32)
    assert finite_ratio_1 >= 0.99
    assert finite_ratio_2 >= 0.99
    # Pinned single-GPU repeat runs are bitwise-stable here; keep a small fixed margin.
    assert np.isfinite(arr_1_safe).all()
    assert np.isfinite(arr_2_safe).all()
    assert np.isclose(float(arr_1_safe.sum()), float(arr_2_safe.sum()), rtol=1e-4, atol=1e-12)
    assert np.isclose(float(arr_1_safe.max()), float(arr_2_safe.max()), rtol=1e-4, atol=1e-12)
    signal_mask = np.logical_and(arr_1_safe > 1e-12, arr_2_safe > 1e-12)
    assert signal_mask.any()
    log_1 = np.log10(arr_1_safe[signal_mask])
    log_2 = np.log10(arr_2_safe[signal_mask])
    log_abs = np.abs(log_1 - log_2)
    assert float(np.percentile(log_abs, 95)) <= 1e-4
    assert float(log_abs.max()) <= 1e-3


@pytest.mark.gpu
def test_pyhyperscattering_integrator_to_xarray_smoke():
    """Run NRSS-to-PyHyperScattering integration and verify xarray/remesh invariants."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for PyHyperScattering smoke test.")

    from PyHyperScattering.integrate import WPIntegrator

    energies = [285.0, 286.0]
    data = _run_tiny_pybind_simulation(energies=energies, return_xarray=True)

    assert hasattr(data, "dims")
    assert "energy" in data.dims
    assert data.sizes["energy"] == len(energies)

    integrator = WPIntegrator(use_chunked_processing=False)
    remeshed = integrator.integrateImageStack(data)
    remeshed_vals = np.asarray(remeshed.values)

    assert hasattr(remeshed, "dims")
    assert "energy" in remeshed.dims
    assert remeshed.sizes["energy"] == len(energies)
    assert "chi" in remeshed.dims
    assert any(dim.startswith("q") or dim == "q" for dim in remeshed.dims)
    # The tiny remesh path shows host/run variability, but repeat runs stayed well above 0.95.
    assert float(np.isfinite(remeshed_vals).mean()) >= 0.95


@pytest.mark.gpu
def test_pybind_runtime_2d_disk_smoke():
    """Run a 2D (1x32x32) pybind morphology to cover the 2D computation pathway."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for 2D runtime smoke test.")

    data = _run_tiny_pybind_simulation(
        energies=[285.0],
        return_xarray=True,
        shape=(1, 32, 32),
        sphere_diameter_vox=16,
    )
    arr = data.values
    arr_safe = _sanitize_scattering(arr)

    assert arr.shape == (1, 32, 32)
    assert float(np.isfinite(arr).mean()) >= 0.99
    assert float(arr_safe.max()) > 1e-5
    assert float(arr_safe.sum()) > 1e-4


@pytest.mark.gpu
def test_cli_serialized_run_matches_pybind_smoke(tmp_path: Path):
    """Compare pybind output to CLI output from serialized morphology/constants/config."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for CLI-vs-pybind smoke test.")
    if shutil.which("CyRSoXS") is None:
        pytest.skip("CyRSoXS CLI executable not found on PATH.")

    energies = [285.0]
    morph = _build_two_material_sphere_morphology(
        energies=energies,
        shape=(32, 32, 32),
        sphere_diameter_vox=16,
    )

    # Intentionally use a single Morphology instance:
    # 1) run pybind, then 2) serialize that same object for CLI parity.
    pybind_vals = morph.run(stdout=False, stderr=False, return_xarray=True).values.copy()
    run_path = tmp_path / "cli_serialized_from_pybind_morph"
    cli_vals = _run_cli_from_serialized_morphology(morph, run_path=run_path)

    assert (run_path / "morphology.hdf5").exists()
    assert (run_path / "config.txt").exists()
    assert (run_path / "HDF5").exists()

    _assert_scattering_parity(pybind_vals, cli_vals)


@pytest.mark.gpu
def test_cli_serialized_multi_energy_matches_pybind_smoke(tmp_path: Path):
    """Compare pybind/CLI parity for a small multi-energy simulation."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for multi-energy parity smoke test.")
    if shutil.which("CyRSoXS") is None:
        pytest.skip("CyRSoXS CLI executable not found on PATH.")

    energies = [285.0, 286.0]
    morph = _build_two_material_sphere_morphology(energies=energies, shape=(32, 32, 32))
    pybind_vals = morph.run(stdout=False, stderr=False, return_xarray=True).values.copy()
    cli_vals = _run_cli_from_serialized_morphology(
        morph, run_path=tmp_path / "cli_multi_energy_parity"
    )

    assert pybind_vals.shape == (2, 32, 32)
    assert cli_vals.shape == (2, 32, 32)
    _assert_scattering_parity(pybind_vals, cli_vals)


@pytest.mark.gpu
def test_cli_serialized_2d_disk_matches_pybind_smoke(tmp_path: Path):
    """Compare pybind/CLI parity for a 2D (1x32x32) morphology."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for 2D CLI parity smoke test.")
    if shutil.which("CyRSoXS") is None:
        pytest.skip("CyRSoXS CLI executable not found on PATH.")

    morph = _build_two_material_sphere_morphology(
        energies=[285.0],
        shape=(1, 32, 32),
        sphere_diameter_vox=16,
    )
    pybind_vals = morph.run(stdout=False, stderr=False, return_xarray=True).values.copy()
    cli_vals = _run_cli_from_serialized_morphology(morph, run_path=tmp_path / "cli_2d_parity")

    assert pybind_vals.shape == (1, 32, 32)
    assert cli_vals.shape == (1, 32, 32)
    _assert_scattering_parity(
        pybind_vals,
        cli_vals,
        rtol_scalar=1e-3,
        p95_abs_max=1e-7,
        max_abs_max=5e-7,
        p95_log_max=8e-2,
        max_log_max=1e-1,
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "config_overrides",
    [
        {"AlgorithmType": 0, "WindowingType": 0, "EwaldsInterpolation": 1, "RotMask": 1},
        {"AlgorithmType": 1, "WindowingType": 0, "EwaldsInterpolation": 1, "RotMask": 1},
        {"AlgorithmType": 0, "WindowingType": 1, "EwaldsInterpolation": 0, "RotMask": 1},
    ],
)
def test_gpu_config_switch_matrix_smoke(config_overrides: dict):
    """Verify key GPU config switches execute and return finite non-trivial output."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for config-switch smoke test.")

    morph = _build_two_material_sphere_morphology(
        energies=[285.0],
        shape=(32, 32, 32),
        config_overrides=config_overrides,
    )
    arr = morph.run(stdout=False, stderr=False, return_xarray=True).values.copy()
    arr_safe = _sanitize_scattering(arr)

    assert arr.shape == (1, 32, 32)
    assert float(np.isfinite(arr).mean()) >= 0.99
    assert float(arr_safe.max()) > 1e-3
    assert float(arr_safe.sum()) > 1e-2


@pytest.mark.gpu
def test_eangle_rotation_endpoint_behavior_smoke():
    """Validate endpoint semantics and expected radial-symmetry trend for E-angle averaging."""
    if not _has_visible_gpu():
        pytest.skip("No visible NVIDIA GPU found for EAngleRotation endpoint smoke test.")
    # Pin to a single visible GPU for more stable endpoint comparisons.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    eangle_0 = _build_two_material_asymmetric_lobed_morphology(
        energies=[285.0], eangle_rotation=[0.0, 0.0, 0.0]
    ).run(stdout=False, stderr=False, return_xarray=True).values.copy()
    eangle_165 = _build_two_material_asymmetric_lobed_morphology(
        energies=[285.0], eangle_rotation=[0.0, 15.0, 165.0]
    ).run(stdout=False, stderr=False, return_xarray=True).values.copy()
    eangle_1799 = _build_two_material_asymmetric_lobed_morphology(
        energies=[285.0], eangle_rotation=[0.0, 15.0, 179.9]
    ).run(stdout=False, stderr=False, return_xarray=True).values.copy()
    eangle_180 = _build_two_material_asymmetric_lobed_morphology(
        energies=[285.0], eangle_rotation=[0.0, 15.0, 180.0]
    ).run(stdout=False, stderr=False, return_xarray=True).values.copy()
    asym_0 = _radial_asymmetry_score(eangle_0)
    asym_165 = _radial_asymmetry_score(eangle_165)
    asym_1799 = _radial_asymmetry_score(eangle_1799)
    asym_180 = _radial_asymmetry_score(eangle_180)

    # Expected for current CyRSoXS behavior (v1.1.8.0):
    # numAnglesRotation = round((end-start)/increment + 1)
    # sampled angle_i = start + i*increment
    # so [0,15,165] -> 0..165 and [0,15,179.9]/[0,15,180] -> 0..180.
    # Fixed thresholds were chosen from repeat-run inspection on pinned single-GPU runs.
    # The image-level L1 differences are noisy, but the ring-asymmetry trend is stable and
    # directly encodes the endpoint-semantics behavior we care about.
    #
    # 179.9 and 180 both round to the same 0..180 sampling and should therefore agree.
    assert abs(asym_1799 - asym_180) <= 0.01
    # 180 includes the endpoint-equivalent orientation while 165 does not, which measurably
    # changes the averaged anisotropy.
    assert (asym_180 - asym_165) >= 0.005
    # Rotation averaging should be substantially more radially symmetric than no rotation.
    assert asym_0 > (1.5 * asym_165)
    assert asym_0 > (1.5 * asym_180)
