"""Compare a literal, windowed material box with independent compact Fourier sums (NumPy)."""
from itertools import product
import json

import numpy as np


DIAGONAL = np.array([1, 0, 0, 1, 0])  # xx, xy, xz, yy, yz
ENERGIES = (285.0, 300.0)
# NRSS order: delta_parallel, beta_parallel, delta_perp, beta_perp.
TABLE = {2: ((.09, .035, -.04, .012), (-.08, .01, .11, .046)),
         1: ((.031, .022, .031, .022), (.018, .005, .018, .005)),
         3: ((0., 0., 0., 0.), (0., 0., 0., 0.))}
NEIGHBORS = tuple(tuple(v) for v in np.vstack((np.eye(3, dtype=int), -np.eye(3, dtype=int))))


def indices(optics):
    dp, bp, dt, bt = optics
    return complex(1 - dp, bp), complex(1 - dt, bt)


def explicit_s0(optics):
    dp, bp, dt, bt = optics
    mean_index = complex(1 - (dp + 2 * dt) / 3, (bp + 2 * bt) / 3)
    return mean_index**2 - 1


def fourier_matrix(length, indices_in_box):
    modes = np.arange(length)[:, None]
    return np.exp((-2j * np.pi / length) * ((modes * indices_in_box) % length))


def separable_transform(field, matrices):
    z, y, x = matrices
    return np.einsum("az,by,cx,zyxk->abck", z, y, x, field, optimize=True)


def separable_mask(factors):
    z, y, x = (np.fft.fft(a) for a in factors)
    return z[:, None, None] * y[None, :, None] * x[None, None, :]


def replace_dc(amplitudes):
    amplitudes[0, 0, 0] = np.mean([amplitudes[index] for index in NEIGHBORS], axis=0)


def verify():
    shape = (5, 7, 8)
    rng = np.random.default_rng(2701)
    fraction, order, theta, psi = rng.random((4, *shape))
    theta *= np.pi
    psi *= 2 * np.pi
    x, y, z = np.cos(psi)*np.sin(theta), np.sin(psi)*np.sin(theta), np.cos(theta)
    director = np.stack((x*x, x*y, x*z, y*y, y*z), axis=-1)
    worst = 0.0
    worst_contrast_first = 0.0
    count = 0
    saw_interference = False
    for target, use_hann, fill, ei in product(
        ((17, 7, 8), (14, 16, 19), (5, 14, 16)), (False, True), (None, 1, 2, 3), range(2)
    ):
        offset = tuple((m-n)//2 for m, n in zip(target, shape))
        crop = tuple(slice(o, o+n) for o, n in zip(offset, shape))
        weights = [np.ones(m) for m in target]
        if use_hann:
            weights[1:] = [np.hanning(m) for m in target[1:]]
        lateral = weights[1][None, :, None] * weights[2][None, None, :]
        npar, nper = indices(TABLE[2][ei])
        iso = explicit_s0(TABLE[2][ei])
        aligned = (nper**2-1)*DIAGONAL + (npar**2-nper**2)*director
        specimen = (fraction[..., None] * ((1-order[..., None])*iso*DIAGONAL + order[..., None]*aligned)
                    + (1-fraction[..., None])*explicit_s0(TABLE[1][ei])*DIAGONAL)
        literal_fill = 0j if fill is None else explicit_s0(TABLE[fill][ei])
        literal = np.empty((*target, 5), dtype=np.complex128)
        literal[:] = literal_fill * DIAGONAL
        literal[crop] = specimen
        expected = np.fft.fftn(literal*lateral[..., None], axes=(0, 1, 2))

        selected_fill = 0j
        if fill is not None:
            par, perp = indices(TABLE[fill][ei])
            selected_fill = ((par + 2*perp)/3)**2 - 1
            if fill == 2:
                assert abs(selected_fill - ((par**2 + 2*perp**2)/3 - 1)) > 1e-3
        matrices = [fourier_matrix(m, np.arange(n)+o) for m, n, o in zip(target, shape, offset)]
        crop_window = weights[1][crop[1]][None, :, None] * weights[2][crop[2]][None, None, :]
        specimen_amp = separable_transform(specimen*crop_window[..., None], matrices)
        inside_weights = [np.zeros(m) for m in target]
        for restricted, full, occupied in zip(inside_weights, weights, crop):
            restricted[occupied] = full[occupied]
        exterior = separable_mask(weights) - separable_mask(inside_weights)
        fill_amp = selected_fill * exterior[..., None] * DIAGONAL
        calculated = specimen_amp + fill_amp
        contrast_crop = (specimen - selected_fill*DIAGONAL)*crop_window[..., None]
        contrast_first = separable_transform(contrast_crop, matrices)
        contrast_first += selected_fill*separable_mask(weights)[..., None]*DIAGONAL
        incoherent = abs(specimen_amp)**2 + abs(fill_amp)**2
        if fill == 2 and use_hann:
            saw_interference |= np.max(abs(abs(calculated)**2 - incoherent)) > .001*np.max(abs(calculated)**2)
        for repair in (False, True):
            if repair:
                replace_dc(expected)
                replace_dc(calculated)
                replace_dc(contrast_first)
            error = np.max(abs(calculated-expected)) / np.max(abs(expected))
            contrast_error = np.max(abs(contrast_first-expected)) / np.max(abs(expected))
            if max(error, contrast_error) > 5e-13:
                raise AssertionError((target, use_hann, fill, ei, repair, error, contrast_error))
            worst = max(worst, float(error))
            worst_contrast_first = max(worst_contrast_first, float(contrast_error))
            count += 1
    assert saw_interference
    return {"complex_tensor_comparisons": count, "max_abs_error_over_peak": worst,
            "contrast_first_comparisons": count, "contrast_first_max_abs_error_over_peak": worst_contrast_first,
            "energies_eV": ENERGIES, "tested_ids": [None, 1, 2, 3],
            "reference": "literal finite target, independent of NRSS or GPU"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
