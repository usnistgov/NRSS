"""Compare registered film detectors to a continuous finite-voxel projection."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from NRSS import SFieldMode
from NRSS.morphology import Material, Morphology
from optimization.cupy_rsoxs.fft_registration import bragg_high_q as registered
from tests.validation.dev.cupy_rsoxs_virtual_zpad.lattice_probe import axial_q, exact_xyz, weight
from tests.validation.lib import bragg
from tests.validation.lib.virtual_z_padding import explicit_vacuum_materials, xy_hann_reference


ENERGIES = (285., 570.)
BCC_BASIS = np.array([[0., 0., 0.], [.5, .5, .5]])
PEAK_FLOORS = (1.e-3, 1.e-4, 1.e-5)
# Freeze the BCC contrast criterion before measuring any detector arm. The
# preceding SC screen used the same contrast for a conventional cubic cell.
CONTRAST = {"hcp": 15., "bcc": 100., "simple_cubic": 100.}


def case_info(name, xy):
    if name in ("hcp_a45", "simple_cubic_a30"):
        from tests.validation.test_bragg_3d_lattice import CASES_BY_ID

        case = CASES_BY_ID[name]
        return case, case.lattice_kind, False
    kind, tilt = name.split("_")
    tilted = tilt in ("tilted", "hann")
    case = bragg.Bragg3DCase(
        name, "simple_cubic" if kind == "bcc" else kind, 30. if kind == "bcc" else 45.,
        11., 27., 9. if tilted else 0., 5. if tilted else 0., (128, xy, xy),
    )
    return case, kind, tilt == "hann"


def specimen(case, kind, specimen_path):
    lattice, basis = bragg.lattice_vectors_3d_nm(
        case.lattice_kind, case.lattice_constant_nm, case.azimuth_deg, case.tilt_x_deg, case.tilt_y_deg,
    )
    if kind == "bcc":
        basis = BCC_BASIS
    count_path = specimen_path.with_suffix(".json")
    if specimen_path.is_file() and count_path.is_file():
        return np.load(specimen_path, mmap_mode="r"), json.loads(count_path.read_text()), basis
    field, positions = bragg._sphere_lattice_vfrac_3d(
        case.shape, case.phys_size_nm, lattice, basis, case.particle_diameter_nm, case.superresolution,
    )
    if not (float(field.min()) >= 0. and float(field.max()) <= 1.):
        raise ValueError("The frozen scalar specimen needs volume fractions in [0, 1].")
    digest = hashlib.sha256(memoryview(field).cast("B")).hexdigest()
    meta = {"occupied_shape": case.shape, "particles": len(positions), "sha256_float32": digest}
    np.save(specimen_path, field)
    count_path.write_text(json.dumps(meta, indent=2) + "\n")
    return field, meta, basis


def merge_unresolved(groups, pixel):
    """Combine reciprocal orders whose transverse positions are within one pixel."""
    centers = [(group["qx"], group["qy"]) for group in groups]
    parents = list(range(len(groups)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, j in cKDTree(centers).query_pairs(pixel):
        parents[root(j)] = root(i)
    components = {}
    for i, group in enumerate(groups):
        components.setdefault(root(i), []).append(group)
    merged = []
    for members in components.values():
        strongest = max(members, key=lambda g: max(order[2] for order in g["orders"]))
        merged.append(dict(strongest, orders=[order for g in members for order in g["orders"]]))
    return merged


def catalog(case, kind, basis):
    groups, (qy, qx) = registered.catalog(
        case, basis_override=basis, include_extinctions=kind == "bcc",
    )
    if kind == "hcp":
        groups = merge_unresolved(groups, min(qx[1] - qx[0], qy[1] - qy[0]))
    bands = {name: limits for name, limits in registered.BANDS.items()
             if name == "common" or case.energy_eV >= 570}
    targets = [(name, g) for name, (low, high) in bands.items()
               for g in groups if low <= g["q_perp"] < high]
    if kind == "bcc":
        anchors = [g for g in groups if any(sorted(map(abs, hkl)) == [0, 1, 1] for hkl, _dz, _w in g["orders"])]
    else:
        anchors = [g for g in groups if any(hkl == (1, 0, 0) for hkl, _dz, _w in g["orders"])]
    anchors = [g for g in anchors if g["q_perp"] > 3 * (qx[1] - qx[0])]
    if not anchors:
        raise ValueError(f"No accessible independent anchor for {case.case_id}.")
    targets.extend(("anchor", g) for g in anchors)
    patches = []
    for _band, g in targets:
        row = int(np.argmin(abs(qy - g["qy"])))
        col = int(np.argmin(abs(qx - g["qx"])))
        yy, xx = row + registered.DY, col + registered.DX
        patches.append(np.ravel_multi_index((yy, xx), (len(qy), len(qx))))
    return targets, np.stack(patches), bands, qy, qx


def independent_map(field, case, qy, qx, ids):
    """Evaluate true forward-Ewald intensity directly from the occupied voxels."""
    nz, ny, nx = field.shape
    iy, ix = np.unravel_index(ids, (ny, nx))
    k = bragg.flat_detector_k_nm_inv(case.energy_eV)
    qy_at, qx_at = qy[iy], qx[ix]
    q2 = qy_at**2 + qx_at**2
    qz_limit = min(k, (nz // 2) * 2 * np.pi / (nz * case.phys_size_nm))
    supported = (q2 <= qz_limit * (2 * k - qz_limit)) & (iy < ny - 1) & (ix < nx - 1)
    ids = ids[supported]
    x, y = qx_at[supported], qy_at[supported]
    z = axial_q(k, x, y)
    xy = registered.spectrum_columns(field, ids, double_precision=True)
    image = np.full((ny, nx), np.nan)
    z_planes = (np.arange(nz) * case.phys_size_nm)[:, None]
    first_checked = False
    for start in range(0, ids.size, 2048):
        sl = slice(start, start + 2048)
        amp = np.sum(xy[:, sl] * np.exp(-1j * z_planes * z[sl]), axis=0)
        if not first_checked and case.phys_size_nm == 1.:
            phase_sum = exact_xyz(field, x[start], y[start], z[start])
            np.testing.assert_allclose(amp[0].conj(), phase_sum, atol=2.e-8, rtol=1.e-9)
            first_checked = True
        image.ravel()[ids[sl]] = abs(amp)**2 * weight(k, x[sl], y[sl], z[sl]) * registered.optical_scale(k)
    return image


def prepare_reference(field, case, kind, hann, basis):
    targets, patches, bands, qy, qx = catalog(case, kind, basis)
    q_perp = np.hypot(qy[:, None], qx[None, :])
    mask = {name: (q_perp >= low) & (q_perp < high) for name, (low, high) in bands.items()}
    ids = np.unique(np.concatenate([patches.ravel(), *[np.flatnonzero(v) for v in mask.values()]]))
    source = field
    if hann:
        xy = np.hanning(field.shape[1])[:, None] * np.hanning(field.shape[2])[None, :]
        source = field.astype(np.float64) * xy
    direct = independent_map(source, case, qy, qx, ids)
    anchors = [idx for idx, (name, _g) in enumerate(targets) if name == "anchor"]
    anchor_rows = registered.measure(
        [targets[idx] for idx in anchors], patches[anchors], qy, qx,
        {"direct": direct.ravel()[patches[anchors]]}, 1.,
    )
    anchor_row = max((row for row in anchor_rows if row["local_peak"]), key=lambda row: row["peak"])
    anchor = anchor_row["peak"]
    assert anchor > 0.
    anchor_id = [i for i, row in zip(anchors, anchor_rows) if row is anchor_row][0]
    allowed_anchor = (1, 0, 0) if kind != "bcc" else next(
        hkl for hkl, _dz, _w in targets[anchor_id][1]["orders"] if sorted(map(abs, hkl)) == [0, 1, 1]
    )
    return targets, patches, qy, qx, mask, direct, anchor, allowed_anchor


def detector(field, depth, arm, hann):
    import cupy as cp

    energies = list(ENERGIES)
    materials = {
        index: Material(
            index, Vfrac=values, S=SFieldMode.ISOTROPIC, energies=energies,
            opt_constants={energy: [0., beta, 0., beta] for energy in energies},
        )
        for index, values, beta in ((1, field, 2.e-4), (2, 1 - field, 0.))
    }
    options = {"execution_path": "tensor_coeff"}
    if arm == "explicit":
        materials = explicit_vacuum_materials(materials, depth)
    elif arm == "virtual":
        options["virtual_z_pad_to"] = depth
    config = dict(Morphology.config_default, Energies=energies, EAngleRotation=[0., 0., 0.],
                  WindowingType=int(hann and arm == "virtual"), RotMask=0)
    morph = Morphology(
        len(materials), materials=materials, PhysSize=1., config=config,
        backend="cupy-rsoxs", backend_options=options, resident_mode="host",
        input_policy="strict", ownership_policy="borrow",
    )
    if hann and arm != "virtual":
        xy_hann_reference(morph)
    try:
        morph._set_private_backend_timing_segments(("energy",))
        started = perf_counter()
        result = morph.run(stdout=False, stderr=False, return_xarray=True).copy(deep=True)
        elapsed = perf_counter() - started
        assert result.attrs["detector_grid_convention"] == "fft_bins_v1"
        assert result.attrs["z_dim"] == depth
        routes = morph.backend_timings["execution"]["routes"]
        assert routes == (["virtual_z_sum"] if arm == "virtual" else ["compact_r2c"])
        if arm == "virtual":
            assert result.attrs["occupied_z"] == field.shape[0]
            assert result.attrs["window_xy"] == ("hann" if hann else "none")
            assert result.attrs["window_z"] == "none"
        return result, {"wall_seconds_with_diagnostics": elapsed, "routes": routes,
                        "shape_zyx": [depth, *field.shape[1:]], "window_xy": "hann" if hann else "none",
                        "window_z": "none", "offset": (depth - field.shape[0]) // 2}
    finally:
        morph.release_runtime()
        bragg.release_runtime_memory()
        cp.get_default_memory_pool().free_all_blocks()


def relative_error(test, ref, valid):
    a, b = test[valid].astype(np.float64), ref[valid].astype(np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def compare_box_panels(explicit, virtual):
    for axis in ("energy", "qy", "qx"):
        np.testing.assert_array_equal(explicit.coords[axis], virtual.coords[axis])
    a, b = explicit.values, virtual.values
    valid = np.isfinite(a)
    np.testing.assert_array_equal(valid, np.isfinite(b))
    by_energy = []
    for energy, x, y, support in zip(ENERGIES, a, b, valid):
        err = relative_error(y, x, support)
        over_peak = float(np.max(abs(x[support] - y[support])) / np.max(abs(x[support])))
        if err > 2.e-5 or over_peak > 2.e-5:
            raise AssertionError((energy, err, over_peak))
        by_energy.append({"energy": energy, "relative_l2": err, "max_difference_over_peak": over_peak,
                          "finite_pixels": int(support.sum())})
    return by_energy


def peak_profiles(rows, targets, patches, images, qy, qx, case, kind):
    pitch = min(qx[1] - qx[0], qy[1] - qy[0])
    dz = 2 * np.pi / (case.shape[0] * case.phys_size_nm)
    k = bragg.flat_detector_k_nm_inv(case.energy_eV)
    modes = list(images)
    for idx, row in enumerate(rows):
        n = idx // len(modes)
        g = targets[n][1]
        native_bin = axial_q(k, row["qx"], row["qy"]) / dz
        row["fraction_to_nearest_axial_bin"] = float(abs(native_bin - np.round(native_bin)))
        row["bcc_odd_only"] = kind == "bcc" and all(sum(hkl) % 2 for hkl, _dz, _weight in g["orders"])
        row.update(centroid_qx=np.nan, centroid_qy=np.nan, rms_width_nm_inv=np.nan, integrated_intensity=np.nan)
        if not row["local_peak"]:
            continue
        yy, xx = np.unravel_index(patches[n], (len(qy), len(qx)))
        image = images[row["mode"]][n]
        core = np.hypot(qx[xx] - row["qx"], qy[yy] - row["qy"]) <= 3 * pitch
        weights = np.maximum(image[core] - row["background"], 0.)
        total = weights.sum()
        if total > 0:
            x, y = qx[xx[core]], qy[yy[core]]
            center_x = float(np.dot(weights, x) / total)
            center_y = float(np.dot(weights, y) / total)
            width = np.sqrt(np.dot(weights, (x - center_x)**2 + (y - center_y)**2) / total)
            row.update(centroid_qx=center_x, centroid_qy=center_y,
                       rms_width_nm_inv=float(width), integrated_intensity=float(total * pitch**2))


def tally(rows, kind):
    modes = tuple(dict.fromkeys(row["mode"] for row in rows))
    table = {mode: {key: row for key, row in enumerate(r for r in rows if r["mode"] == mode)}
             for mode in modes}
    out = []
    for band in registered.BANDS:
        candidates = {i for i, r in table["direct"].items() if r["band"] == band and not r["bcc_odd_only"]}
        valid = {i for i in candidates if all(table[m][i]["support"] for m in modes)}
        if not candidates:
            continue
        for floor in PEAK_FLOORS:
            hits = {m: {i for i in valid if table[m][i]["local_peak"]
                        and table[m][i]["peak_over_anchor"] >= floor
                        and table[m][i]["peak_to_background"] >= CONTRAST[kind]}
                    for m in modes}
            truth, native = hits["direct"], hits["native"]
            for mode in modes:
                found = hits[mode]
                out.append(dict(band=band, floor=floor, mode=mode, contrast=CONTRAST[kind],
                                candidates=len(candidates), common_support=len(valid), reference=len(truth),
                                matched=len(found & truth), recovered=len((found - native) & truth),
                                lost=len((native - found) & truth), false=len(found - truth),
                                missed=len(truth - found)))
    return out


def save_outputs(case, kind, out, expected, results):
    targets, patches, qy, qx, masks, direct, anchor, anchor_hkl = expected
    panels = {"direct": direct}
    energy_index = ENERGIES.index(case.energy_eV)
    for mode, frame in results.items():
        np.testing.assert_allclose(frame.qy, qy, rtol=0, atol=1.e-14)
        np.testing.assert_allclose(frame.qx, qx, rtol=0, atol=1.e-14)
        panels[mode] = frame.values[energy_index]
    images = {mode: value.ravel()[patches] for mode, value in panels.items()}
    rows = registered.measure(targets, patches, qy, qx, images, anchor)
    peak_profiles(rows, targets, patches, images, qy, qx, case, kind)
    for start in range(0, len(rows), len(images)):
        direct_row = rows[start]
        for row in rows[start:start + len(images)]:
            row["centroid_error_nm_inv"] = float(np.hypot(
                row["centroid_qx"] - direct_row["centroid_qx"], row["centroid_qy"] - direct_row["centroid_qy"],
            ))
            row["relative_peak_error"] = (row["peak"] - direct_row["peak"]) / direct_row["peak"]
    out.mkdir(parents=True, exist_ok=True)
    registered.write_rows(out / "peaks.csv", rows)
    registered.write_rows(out / "counts.csv", tally(rows, kind))
    band_errors = []
    for band, mask in masks.items():
        common = mask & np.logical_and.reduce([np.isfinite(v) for v in panels.values()])
        for mode, value in panels.items():
            band_errors.append(dict(
                band=band, mode=mode, pixels=int(mask.sum()), common_support=int(common.sum()),
                unsupported=int((mask & ~np.isfinite(value)).sum()), relative_l2=relative_error(value, direct, common),
            ))
    registered.write_rows(out / "bands.csv", band_errors)
    (out / "anchor.json").write_text(json.dumps({"hkl": anchor_hkl, "intensity": anchor, "contrast": CONTRAST[kind]}) + "\n")
    np.savez_compressed(out / "detector.npz", qx=qx, qy=qy, **panels)


def run_case(name, xy, targets, output):
    case, kind, hann = case_info(name, xy)
    occupied = case.shape[0]
    out = output / name
    out.mkdir(parents=True, exist_ok=True)
    unique_name = "hcp_tilted" if name == "hcp_hann" else name
    field, specimen_meta, basis = specimen(case, kind, output / (unique_name + ".npy"))
    configs = [replace(case, energy_eV=e) for e in ENERGIES]
    references = [prepare_reference(field, c, kind, hann, basis) for c in configs]
    results = {}
    results["native"], native_meta = detector(field, occupied, "native", hann)
    meta = dict(specimen=specimen_meta, native=native_meta, matched={}, declared_contrast=CONTRAST[kind])
    for target in targets:
        if target == occupied:
            continue
        literal, explicit_meta = detector(field, target, "explicit", hann)
        virtual, virtual_meta = detector(field, target, "virtual", hann)
        results[f"explicit_{target}"] = literal
        results[f"virtual_{target}"] = virtual
        meta["matched"][target] = dict(
            explicit=explicit_meta, virtual=virtual_meta, parity=compare_box_panels(literal, virtual),
        )
        print(name, target, meta["matched"][target]["parity"], flush=True)
    for config, expected in zip(configs, references):
        save_outputs(config, kind, out / f"{config.energy_eV:g}", expected, results)
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=(
        "hcp_flat", "hcp_tilted", "bcc_flat", "bcc_tilted", "hcp_hann", "hcp_a45", "simple_cubic_a30",
    ))
    parser.add_argument("--xy-size", type=int, default=1024)
    parser.add_argument("--depths", type=int, nargs="+", default=(128, 256, 512, 1024))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run_case(args.case, args.xy_size, args.depths, args.out)
