"""Reference product obtained by selecting from a completed NRSS xarray result."""
import argparse
import json
from numbers import Real

import numpy as np
import xarray as xr


def cutoff_value(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("q_max must be a finite nonnegative real in inverse nm")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("q_max must be a finite nonnegative real in inverse nm")
    return value


def detector_mask(transverse, energy, is_plane, limit):
    if is_plane:
        return transverse <= limit
    k = 2.0 * np.pi / (1239.84197 / energy)
    values = k * k - transverse * transverse
    valid = values >= 0.0
    root = np.sqrt(values, where=valid, out=np.full_like(values, np.nan))
    physical = np.sqrt(transverse * transverse + (root - k) ** 2)
    return valid & (physical <= limit)


def select_product(full, q_max):
    """Public reference. Keep coordinates and positional auxiliary-coordinate slices."""
    if q_max is None:
        return full
    limit = cutoff_value(q_max)
    if "q" in full.dims:
        if full.attrs.get("radial_coordinate_mode") == "per_slice_q_abs":
            physical = full.coords["q_abs"]
            allowed = np.isfinite(physical) & (physical <= limit)
            indices = np.flatnonzero(allowed.any("energy").values)
            return full.isel(q=indices).where(allowed.isel(q=indices))
        axis = full.coords["q"].values
        indices = np.flatnonzero(np.isfinite(axis) & (axis <= limit))
        return full.isel(q=indices)

    # Detector frames share a Cartesian array across energies. Build 2D masks
    # one frame at a time and allocate the output mask only on the selected box.
    qy = np.asarray(full.qy, dtype=np.float64)
    qx = np.asarray(full.qx, dtype=np.float64)
    energies = np.asarray(full.energy, dtype=np.float64)
    radius = np.hypot(qy[:, None], qx[None, :])
    is_plane = int(full.attrs["z_dim"]) == 1

    union = np.zeros(radius.shape, dtype=bool)
    for energy in energies:
        union |= detector_mask(radius, energy, is_plane, limit)
    rows = np.flatnonzero(union.any(axis=1))
    cols = np.flatnonzero(union.any(axis=0))
    if not rows.size or not cols.size:
        raise ValueError("A registered detector with DC must have a pixel at q=0")
    iy = slice(rows[0], rows[-1] + 1)
    ix = slice(cols[0], cols[-1] + 1)
    cropped = full.isel(qy=iy, qx=ix)
    masks = np.stack([detector_mask(radius[iy, ix], energy, is_plane, limit) for energy in energies])
    return cropped.where(xr.DataArray(masks, dims=("energy", "qy", "qx")))


def channels(obj):
    return dict(obj.data_vars) if isinstance(obj, xr.Dataset) else {"__array__": obj}


def bit_matches(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        raise AssertionError(f"Shape or dtype changed: {a.shape} {a.dtype}, {b.shape} {b.dtype}")
    if a.dtype.hasobject:
        raise TypeError("Bit comparisons require numeric arrays")
    return np.all(a.view(np.uint8).reshape(a.size, a.dtype.itemsize) ==
                  b.view(np.uint8).reshape(b.size, b.dtype.itemsize), axis=1).reshape(a.shape)


def compare(reference, result, q_max=None):
    """Require identical bits at every requested sample, including NaNs and signed zero."""
    if type(reference) is not type(result) or dict(reference.sizes) != dict(result.sizes):
        raise AssertionError("Result kind or dimension lengths changed")
    if set(reference.coords) != set(result.coords):
        raise AssertionError("Coordinate names changed")
    for name, coord in reference.coords.items():
        other = result.coords[name]
        if coord.dims != other.dims or not np.all(bit_matches(coord.values, other.values)):
            raise AssertionError(f"Coordinate {name!r} changed")
    expected, candidate = channels(reference), channels(result)
    if set(expected) != set(candidate):
        raise AssertionError("Channel names changed")
    if isinstance(reference, xr.DataArray) and reference.name != result.name:
        raise AssertionError("DataArray name changed")
    for key, value in reference.attrs.items():
        if key not in result.attrs:
            raise AssertionError(f"Missing original attribute {key}")
        np.testing.assert_equal(result.attrs[key], value, err_msg=f"attribute {key}")
    radial = reference.attrs.get("radial_coordinate_mode") == "per_slice_q_abs"
    transverse = None
    if q_max is not None and "qy" in reference.dims:
        transverse = np.hypot(np.asarray(reference.qy, dtype=np.float64)[:, None],
                              np.asarray(reference.qx, dtype=np.float64)[None, :])
    counts = {}
    for name, array in expected.items():
        other = candidate[name]
        if array.dims != other.dims:
            raise AssertionError(f"Dimension order changed for {name}")
        per_energy = []
        for i, energy in enumerate(reference.energy.values):
            ref, got = array.isel(energy=i), other.isel(energy=i)
            same = bit_matches(ref.values, got.values)
            if transverse is not None:
                mask = detector_mask(transverse, energy, reference.attrs["z_dim"] == 1, q_max)
            elif radial and q_max is not None:
                axis = np.asarray(reference.q_abs.isel(energy=i))
                mask = np.broadcast_to(np.isfinite(axis) & (axis <= q_max), ref.shape)
            else:
                mask = np.ones(ref.shape, dtype=bool)
            bad = mask & ~same
            if np.any(bad):
                at = tuple(np.argwhere(bad)[0])
                old = np.asarray(ref.values[at]).tobytes().hex()
                new = np.asarray(got.values[at]).tobytes().hex()
                raise AssertionError(f"{name}, E={energy}, {at}: {int(bad.sum())} bit mismatches, {old} -> {new}")
            if not np.all(np.isnan(got.values[~mask])):
                raise AssertionError(f"Exposed non-NaN outside the requested region in {name}")
            per_energy.append({"energy_ev": float(energy), "in_window_values_checked": int(mask.sum()),
                               "bit_mismatches": 0})
        counts[name] = {"in_window_values_checked": sum(row["in_window_values_checked"] for row in per_energy),
                        "bit_mismatches": 0, "per_energy": per_energy}
    return counts


def self_test():
    q = np.array([0., 0.25, 0.5, 0.75, 1.])
    energies = [285., 280.]
    values = np.column_stack((q, 2 * q)).T.astype("float32")
    labels = {"q": q, "energy": energies, "q_abs": (("energy", "q"), [q, q * 1.1])}
    full = xr.Dataset({"I": (("energy", "q"), values), "A": (("energy", "q"), -values)}, coords=labels)
    expected = full.isel(q=[0, 1, 2])
    compare(expected, select_product(full, 0.5), 0.5)
    assert select_product(full, np.nextafter(0.5, 0.)).sizes["q"] == 2
    assert select_product(full, None) is full
    per = full.assign_coords(q=np.arange(q.size)).assign_attrs(radial_coordinate_mode="per_slice_q_abs")
    selected = select_product(per, 0.5)
    assert np.isnan(selected.I.values[1, -1]) and selected.I.values[0, -1] == values[0, 2]
    detector = xr.DataArray(np.zeros((2, 11, 13), dtype="float32"), dims=("energy", "qy", "qx"),
                            coords={"energy": [400., 160.], "qy": np.arange(-5, 6) * 0.2,
                                    "qx": np.arange(-6, 7) * 0.15}, attrs={"z_dim": 13})
    cropped = select_product(detector, 0.8)
    assert cropped.sizes["qx"] < detector.sizes["qx"] and cropped.isnull().any().item()
    assert np.isfinite(cropped[0]).sum() > np.isfinite(cropped[1]).sum()
    assert select_product(detector, 0.).shape == (2, 1, 1)
    plane = select_product(detector.assign_attrs(z_dim=1), 0.8)
    assert np.isfinite(plane[0]).sum() == np.isfinite(plane[1]).sum()
    detector.values.view(np.uint32)[0, 5, 6] = 0x7fc00003
    center_only = select_product(detector, 0.)
    assert center_only.values.view(np.uint32)[0, 0, 0] == 0x7fc00003
    compare(center_only, center_only.copy(deep=True), 0.)
    signed = expected.copy(deep=True)
    signed["I"].values[0, 0] = np.float32(-0.)
    ulp = expected.copy(deep=True)
    ulp["I"].values[0, 1] = np.nextafter(ulp["I"].values[0, 1], np.float32(np.inf))
    nan_ref = expected.copy(deep=True)
    nan_ref["I"].values.view(np.uint32)[0, 2] = 0x7fc00001
    nan_other = nan_ref.copy(deep=True)
    nan_other["I"].values.view(np.uint32)[0, 2] = 0x7fc00002
    try:
        compare(nan_ref, nan_other, 0.5)
    except AssertionError:
        pass
    else:
        raise AssertionError("NaN payload difference passed")
    for invalid in (-1., np.inf, np.nan, True, "0.8"):
        try:
            select_product(full, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted {invalid!r}")
    for bad in (signed, ulp, expected.assign_coords(q=expected.q + 0.001), expected.assign(A=expected.A + 0.1),
                expected.assign(I=expected.I.where(expected.I > 0))):
        try:
            compare(expected, bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("Comparator accepted a damaged reference")
    print("Reference selections, boundary neighbors, masks, coordinates and failure checks passed.")


def load_product(path):
    try:
        return xr.load_dataarray(path)
    except ValueError:
        return xr.load_dataset(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--full")
    parser.add_argument("--limited")
    parser.add_argument("--q-max", type=float, default=0.8)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.full and args.limited:
        print(json.dumps(compare(select_product(load_product(args.full), args.q_max),
                                 load_product(args.limited), args.q_max), indent=2))
    else:
        parser.error("Use --self-test or supply --full and --limited NetCDF files")


if __name__ == "__main__":
    main()
