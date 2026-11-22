"""Numerical edge test for planning before polar interpolation and E-angle rotation."""
import json

import numpy as np


def neighbors(y, x):
    iy, ix = np.floor(y).astype(int), np.floor(x).astype(int)
    fy, fx = y - iy, x - ix
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            yield iy + dy, ix + dx, wy * wx


def pixels(y, x, shape):
    result = np.zeros(shape, dtype=bool)
    for iy, ix, _ in neighbors(y, x):
        inside = (iy >= 0) & (iy < shape[0]) & (ix >= 0) & (ix < shape[1])
        result[iy[inside], ix[inside]] = True
    return result


def sample(a, y, x, *, skip_zero):
    values = np.zeros_like(y, dtype=float)
    inside = (y >= 0) & (y <= a.shape[0] - 1) & (x >= 0) & (x <= a.shape[1] - 1)
    for iy, ix, weight in neighbors(y, x):
        use = inside & (weight != 0) if skip_zero else inside
        values[use] += weight[use] * a[iy[use].clip(0, a.shape[0]-1), ix[use].clip(0, a.shape[1]-1)]
    values[~inside] = np.nan
    return values


def exercise(shape, angles):
    ny, nx = shape
    y, x = np.indices(shape)
    pitch, limit = 1.5, 0.8
    k = 2 * np.pi / (1239.84197 / 287.55)
    dqy, dqx = 2 * np.pi / (pitch * np.array(shape))
    center = np.array(shape) // 2
    angular_center = np.array(shape) / 2
    extent = np.array(shape) - center
    count = int(np.ceil(np.hypot(*extent)))
    step = np.hypot(*(extent * [dqy, dqx])) / count
    qr = np.arange(count) * step
    valid_r = qr < k
    qa = np.hypot(qr[valid_r], np.sqrt(k*k - qr[valid_r]**2) - k)
    # Exercise a requested shared-grid point at the edge of the window.
    upper = np.searchsorted(qa, limit, side="right")
    lower = upper - 1
    frac = (limit - qa[lower]) / (qa[upper] - qa[lower])
    theta = np.deg2rad(np.arange(360))[:, None]
    radial = qr[[lower, upper]][None, :]
    polar_y = center[0] + radial * np.sin(theta) / dqy
    polar_x = center[1] + radial * np.cos(theta) / dqx
    after_rotation = pixels(polar_y, polar_x, shape)
    sy, sx = np.nonzero(after_rotation)
    def inverse(yy, xx, angle):
        cosine, sine = np.cos(angle), np.sin(angle)
        yy, xx = yy - angular_center[0], xx - angular_center[1]
        return (angular_center[0] + cosine * yy + sine * xx,
                angular_center[1] - sine * yy + cosine * xx)

    demand = np.zeros(shape, dtype=bool)
    for angle in angles:
        demand |= pixels(*inverse(sy, sx, angle), shape)
    coeffs = np.random.default_rng(19).uniform(0.2, 1.0, (3, ny, nx))
    radius = np.hypot((y-center[0])*dqy, (x-center[1])*dqx)
    cut_r = limit * np.sqrt(1 - (limit/(2*k))**2)

    def evaluate(mask):
        physical = (radius <= k) & (y < ny - 1) & (x < nx - 1)
        planes = np.where(mask & physical, coeffs, np.nan)
        total = np.zeros(shape)
        for angle in angles:
            c, s = np.cos(angle), np.sin(angle)
            field = c*c*planes[0] + s*s*planes[1] + c*s*planes[2]
            total += sample(field, *inverse(y, x, angle), skip_zero=True)
        ring = sample(total/len(angles), polar_y, polar_x, skip_zero=False)
        return (1 - frac)*ring[:, 0] + frac*ring[:, 1]

    dense = evaluate(np.ones(shape, dtype=bool))
    sparse = evaluate(demand)
    np.testing.assert_array_equal(dense.view(np.uint64), sparse.view(np.uint64))
    finite = np.isfinite(dense)
    assert finite.any()
    clipped = evaluate(radius <= cut_r)
    margin = 4 * max(dqy, dqx)
    heuristic = evaluate(radius <= cut_r + margin)
    return {"shape": list(shape), "angles_deg": np.rint(np.rad2deg(angles)).astype(int).tolist(),
            "finite_rays": int(finite.sum()), "exact_dependencies_match": True,
            "rays_lost_by_hard_cut": int(np.count_nonzero(finite & ~np.isfinite(clipped))),
            "rays_lost_by_four_pixel_guard": int(np.count_nonzero(finite & ~np.isfinite(heuristic)))}


if __name__ == "__main__":
    angles = np.deg2rad(np.arange(0, 180, 15))
    examples = [exercise(shape, angles) for shape in ((128, 128), (77, 169))]
    assert examples[0]["rays_lost_by_four_pixel_guard"] == 0
    assert examples[1]["rays_lost_by_four_pixel_guard"] > 0
    assert all(example["rays_lost_by_hard_cut"] > 0 for example in examples)
    print(json.dumps(examples, indent=2))
