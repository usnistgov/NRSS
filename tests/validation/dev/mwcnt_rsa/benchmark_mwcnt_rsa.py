from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

try:
    from numba import njit
except ModuleNotFoundError:
    def njit(*args, **kwargs):  # type: ignore[no-redef]
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def _decorator(fn):
            return fn

        return _decorator


REPO_ROOT = Path(__file__).resolve().parents[4]
TUTORIAL_PATH = REPO_ROOT / "src" / "NRSS_tutorials" / "MWCNTs"
FIBRIL_MODELS_SRC = Path("/homes/deand/dev/fibril_models/src")
OUT_DIR = REPO_ROOT / "test-reports" / "mwcnt-rsa-dev"
REFERENCE_GEOMETRY_PATH = REPO_ROOT / "tests" / "validation" / "data" / "mwcnt" / "mwcnt_seed12345_cnts.csv"

if str(TUTORIAL_PATH) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_PATH))
if str(FIBRIL_MODELS_SRC) not in sys.path:
    sys.path.insert(0, str(FIBRIL_MODELS_SRC))


NUM_TRIALS = 20_000
RADIUS_MU = 2.225
RADIUS_SIGMA = 0.23
THETA_MU = math.pi / 2.0
THETA_SIGMA = 1 / 2 / math.pi
LENGTH_LOWER = 75.0
LENGTH_UPPER = 300.0
BOX_XY = 2048
BOX_Z = 256
SEED = 12345
BOX_DIMS = np.asarray([BOX_Z, BOX_XY, BOX_XY], dtype=np.float64)
HALF_BOX_DIMS = BOX_DIMS / 2.0


@dataclass(frozen=True)
class Candidate:
    idx: int
    x_center: float
    y_center: float
    z_center: float
    psi: float
    theta: float
    length: float
    radius: float
    points_zyx: np.ndarray


def _candidate_row(candidate: Candidate) -> tuple[float, ...]:
    return (
        float(np.float32(candidate.x_center)),
        float(np.float32(candidate.y_center)),
        float(np.float32(candidate.z_center)),
        float(np.float32(candidate.psi)),
        float(np.float32(candidate.theta)),
        float(np.float32(candidate.length)),
        float(np.float32(candidate.radius)),
    )


def candidate_rows(candidates: list[Candidate]) -> list[tuple[float, ...]]:
    return [_candidate_row(candidate) for candidate in candidates]


def _rng_candidate_stream(
    seed: int = SEED,
    num_trials: int = NUM_TRIALS,
    radius_mu: float = RADIUS_MU,
    radius_sigma: float = RADIUS_SIGMA,
    theta_mu: float = THETA_MU,
    theta_sigma: float = THETA_SIGMA,
    length_lower: float = LENGTH_LOWER,
    length_upper: float = LENGTH_UPPER,
) -> list[Candidate]:
    rng = np.random.RandomState(seed)
    candidates: list[Candidate] = []
    total = num_trials + 1
    for idx in range(total):
        psi = float(rng.random_sample() * np.pi * 2.0)
        theta = float(rng.normal(theta_mu, theta_sigma))
        radius = float(rng.lognormal(radius_mu, sigma=radius_sigma))
        length = float(rng.uniform(length_lower, length_upper))
        x_center = float(rng.random_sample() * BOX_XY)
        y_center = float(rng.random_sample() * BOX_XY)
        z_center = float(rng.random_sample() * BOX_Z)

        r = np.linspace(
            -length / 2.0,
            length / 2.0,
            int(abs(length / 2.0 / radius + 1.0)),
            dtype=np.float64,
        )
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        x = (x_center + r * sin_theta * cos_psi) % BOX_XY
        y = (y_center + r * sin_theta * sin_psi) % BOX_XY
        z = (z_center + r * cos_theta) % BOX_Z
        points_zyx = np.column_stack((z, y, x))

        candidates.append(
            Candidate(
                idx=idx,
                x_center=x_center,
                y_center=y_center,
                z_center=z_center,
                psi=psi,
                theta=theta,
                length=length,
                radius=radius,
                points_zyx=points_zyx,
            )
        )
    return candidates


def generate_candidates(
    seed: int = SEED,
    num_trials: int = NUM_TRIALS,
    radius_mu: float = RADIUS_MU,
    radius_sigma: float = RADIUS_SIGMA,
    theta_mu: float = THETA_MU,
    theta_sigma: float = THETA_SIGMA,
    length_lower: float = LENGTH_LOWER,
    length_upper: float = LENGTH_UPPER,
) -> list[Candidate]:
    return _rng_candidate_stream(
        seed=seed,
        num_trials=num_trials,
        radius_mu=radius_mu,
        radius_sigma=radius_sigma,
        theta_mu=theta_mu,
        theta_sigma=theta_sigma,
        length_lower=length_lower,
        length_upper=length_upper,
    )


def _periodic_diffs(test_points: np.ndarray, ref_points: np.ndarray) -> np.ndarray:
    diff = test_points[:, None, :] - ref_points[None, :, :]
    diff = np.where(np.abs(diff) > HALF_BOX_DIMS, np.where(diff > 0.0, diff - BOX_DIMS, diff + BOX_DIMS), diff)
    return diff


def _candidate_center_zyx(candidate: Candidate) -> np.ndarray:
    return np.asarray([candidate.z_center, candidate.y_center, candidate.x_center], dtype=np.float64)


def _occupied_point_capacity(candidates: list[Candidate]) -> int:
    return sum(int(candidate.points_zyx.shape[0]) for candidate in candidates)


def _periodic_center_sq_dists(center_zyx: np.ndarray, ref_centers_zyx: np.ndarray) -> np.ndarray:
    if ref_centers_zyx.size == 0:
        return np.empty((0,), dtype=np.float64)
    diff = np.abs(ref_centers_zyx - center_zyx[None, :])
    diff = np.where(diff > HALF_BOX_DIMS, BOX_DIMS - diff, diff)
    return np.sum(diff * diff, axis=1, dtype=np.float64)


def _pair_min_allowed(candidate_radius: float, ref_radius: float) -> float:
    return float(int(candidate_radius + ref_radius))


@njit(cache=True)
def _reject_by_points_numba(
    candidate_points: np.ndarray,
    candidate_radius: float,
    ref_points: np.ndarray,
    ref_radii: np.ndarray,
) -> bool:
    if ref_points.shape[0] == 0:
        return False

    for i in range(candidate_points.shape[0]):
        z1 = candidate_points[i, 0]
        y1 = candidate_points[i, 1]
        x1 = candidate_points[i, 2]

        for j in range(ref_points.shape[0]):
            dz = abs(z1 - ref_points[j, 0])
            if dz > BOX_Z / 2.0:
                dz = BOX_Z - dz

            dy = abs(y1 - ref_points[j, 1])
            if dy > BOX_XY / 2.0:
                dy = BOX_XY - dy

            dx = abs(x1 - ref_points[j, 2])
            if dx > BOX_XY / 2.0:
                dx = BOX_XY - dx

            min_allowed = float(int(candidate_radius + ref_radii[j]))
            if (dz * dz + dy * dy + dx * dx) < (min_allowed * min_allowed):
                return True
    return False


def _reject_candidate_pair(candidate: Candidate, ref_candidate: Candidate) -> bool:
    min_allowed = _pair_min_allowed(candidate.radius, ref_candidate.radius)
    diff = _periodic_diffs(candidate.points_zyx, ref_candidate.points_zyx)
    sq_dist = np.sum(diff * diff, axis=2, dtype=np.float64)
    return bool(np.any(sq_dist < (min_allowed * min_allowed)))


def _reject_by_broadcast(candidate: Candidate, occ_points: np.ndarray, occ_radii: np.ndarray) -> bool:
    if occ_points.size == 0:
        return False
    diff = _periodic_diffs(candidate.points_zyx, occ_points)
    sq_dist = np.sum(diff * diff, axis=2, dtype=np.float64)
    min_allowed = np.trunc(float(candidate.radius) + occ_radii[None, :]).astype(np.float64, copy=False)
    return bool(np.any(sq_dist < (min_allowed * min_allowed)))


def _reject_by_points(candidate: Candidate, ref_points: np.ndarray, ref_radii: np.ndarray) -> bool:
    if ref_points.size == 0:
        return False
    return _reject_by_broadcast(candidate, ref_points, ref_radii)


def _reject_by_kdtree(candidate: Candidate, tree: cKDTree | None, occ_radii: np.ndarray) -> bool:
    if tree is None or occ_radii.size == 0:
        return False
    search_radius = float(candidate.radius) + float(np.max(occ_radii))
    neighbor_lists = tree.query_ball_point(candidate.points_zyx, r=search_radius)
    for point, neighbor_idx in zip(candidate.points_zyx, neighbor_lists):
        if not neighbor_idx:
            continue
        ref_points = tree.data[np.asarray(neighbor_idx, dtype=np.int32)]
        ref_r = occ_radii[np.asarray(neighbor_idx, dtype=np.int32)]
        diff = point[None, :] - ref_points
        box_dims = np.asarray([BOX_Z, BOX_XY, BOX_XY], dtype=np.float64)
        half_box = box_dims / 2.0
        diff = np.where(np.abs(diff) > half_box, np.where(diff > 0.0, diff - box_dims, diff + box_dims), diff)
        sq_dist = np.sum(diff * diff, axis=1, dtype=np.float64)
        min_allowed = np.trunc(float(candidate.radius) + ref_r).astype(np.float64, copy=False)
        if np.any(sq_dist < (min_allowed * min_allowed)):
            return True
    return False


def _load_cupy():
    try:
        import cupy as cp  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("cupy is not importable in the active environment.") from exc

    device_count = cp.cuda.runtime.getDeviceCount()
    if device_count < 1:
        raise RuntimeError("No CUDA devices visible for cupy exact-broadcast benchmark.")
    return cp


def _append_points(
    occ_points: np.ndarray,
    occ_radii: np.ndarray,
    candidate: Candidate,
) -> tuple[np.ndarray, np.ndarray]:
    new_points = candidate.points_zyx
    new_radii = np.full(new_points.shape[0], float(candidate.radius), dtype=np.float64)
    if occ_points.size == 0:
        return new_points.copy(), new_radii
    return np.vstack((occ_points, new_points)), np.concatenate((occ_radii, new_radii))


def run_broadcast_exact(candidates: list[Candidate]) -> dict[str, object]:
    accepted: list[Candidate] = []
    occ_points = np.empty((0, 3), dtype=np.float64)
    occ_radii = np.empty((0,), dtype=np.float64)
    for candidate in candidates:
        if _reject_by_broadcast(candidate, occ_points, occ_radii):
            continue
        accepted.append(candidate)
        occ_points, occ_radii = _append_points(occ_points, occ_radii, candidate)
    return {
        "accepted": accepted,
        "occ_point_count": int(occ_points.shape[0]),
    }


def run_kdtree_rebuild_each(candidates: list[Candidate]) -> dict[str, object]:
    accepted: list[Candidate] = []
    occ_points = np.empty((0, 3), dtype=np.float64)
    occ_radii = np.empty((0,), dtype=np.float64)
    tree: cKDTree | None = None
    for candidate in candidates:
        if _reject_by_kdtree(candidate, tree, occ_radii):
            continue
        accepted.append(candidate)
        occ_points, occ_radii = _append_points(occ_points, occ_radii, candidate)
        tree = cKDTree(occ_points, boxsize=[BOX_Z, BOX_XY, BOX_XY])
    return {
        "accepted": accepted,
        "occ_point_count": int(occ_points.shape[0]),
    }


def run_kdtree_dual(candidates: list[Candidate], rebuild_threshold_points: int = 512) -> dict[str, object]:
    accepted: list[Candidate] = []
    tree_points = np.empty((0, 3), dtype=np.float64)
    tree_radii = np.empty((0,), dtype=np.float64)
    recent_points = np.empty((0, 3), dtype=np.float64)
    recent_radii = np.empty((0,), dtype=np.float64)
    tree: cKDTree | None = None
    points_since_rebuild = 0

    for candidate in candidates:
        reject = _reject_by_kdtree(candidate, tree, tree_radii)
        if not reject and recent_points.size:
            reject = _reject_by_points(candidate, recent_points, recent_radii)
        if reject:
            continue

        accepted.append(candidate)
        new_points = candidate.points_zyx
        new_radii = np.full(new_points.shape[0], float(candidate.radius), dtype=np.float64)
        recent_points = new_points.copy() if recent_points.size == 0 else np.vstack((recent_points, new_points))
        recent_radii = new_radii if recent_radii.size == 0 else np.concatenate((recent_radii, new_radii))
        points_since_rebuild += int(new_points.shape[0])

        if points_since_rebuild >= rebuild_threshold_points:
            tree_points = recent_points.copy() if tree_points.size == 0 else np.vstack((tree_points, recent_points))
            tree_radii = recent_radii.copy() if tree_radii.size == 0 else np.concatenate((tree_radii, recent_radii))
            tree = cKDTree(tree_points, boxsize=[BOX_Z, BOX_XY, BOX_XY])
            recent_points = np.empty((0, 3), dtype=np.float64)
            recent_radii = np.empty((0,), dtype=np.float64)
            points_since_rebuild = 0

    if recent_points.size:
        tree_points = recent_points.copy() if tree_points.size == 0 else np.vstack((tree_points, recent_points))
        tree_radii = recent_radii if tree_radii.size == 0 else np.concatenate((tree_radii, recent_radii))

    return {
        "accepted": accepted,
        "occ_point_count": int(tree_points.shape[0]),
    }


def run_cupy_exact_batched(candidates: list[Candidate], occ_batch_points: int = 16_384) -> dict[str, object]:
    cp = _load_cupy()
    accepted: list[Candidate] = []
    capacity = _occupied_point_capacity(candidates)
    occ_points = cp.empty((capacity, 3), dtype=cp.float64)
    occ_radii = cp.empty((capacity,), dtype=cp.float64)
    occ_count = 0
    box_dims_cp = cp.asarray(BOX_DIMS, dtype=cp.float64)
    half_box_cp = cp.asarray(HALF_BOX_DIMS, dtype=cp.float64)

    for candidate in candidates:
        candidate_points = cp.asarray(candidate.points_zyx, dtype=cp.float64)
        reject = False

        if occ_count:
            for start in range(0, occ_count, occ_batch_points):
                stop = min(start + occ_batch_points, occ_count)
                ref_points = occ_points[start:stop]
                ref_radii = occ_radii[start:stop]
                diff = candidate_points[:, None, :] - ref_points[None, :, :]
                diff = cp.where(
                    cp.abs(diff) > half_box_cp,
                    cp.where(diff > 0.0, diff - box_dims_cp, diff + box_dims_cp),
                    diff,
                )
                sq_dist = cp.sum(diff * diff, axis=2, dtype=cp.float64)
                min_allowed = cp.trunc(float(candidate.radius) + ref_radii[None, :])
                if bool(cp.any(sq_dist < (min_allowed * min_allowed)).item()):
                    reject = True
                    break

        if reject:
            continue

        accepted.append(candidate)
        point_count = int(candidate.points_zyx.shape[0])
        occ_points[occ_count : occ_count + point_count] = candidate_points
        occ_radii[occ_count : occ_count + point_count] = float(candidate.radius)
        occ_count += point_count

    cp.cuda.Stream.null.synchronize()
    return {
        "accepted": accepted,
        "occ_point_count": int(occ_count),
    }


def _cell_index(point_zyx: np.ndarray, cell_size: float, grid_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    return (
        int(point_zyx[0] / cell_size) % grid_shape[0],
        int(point_zyx[1] / cell_size) % grid_shape[1],
        int(point_zyx[2] / cell_size) % grid_shape[2],
    )


def _reject_by_cell_list(
    candidate: Candidate,
    occ_points: np.ndarray,
    occ_radii: np.ndarray,
    grid: dict[tuple[int, int, int], list[int]],
    cell_size: float,
    grid_shape: tuple[int, int, int],
    max_occ_radius: float,
) -> bool:
    if not grid:
        return False

    search_radius = float(candidate.radius) + max_occ_radius
    reach = max(1, int(math.ceil(search_radius / cell_size)))
    candidate_neighbor_indices: set[int] = set()

    for point in candidate.points_zyx:
        cell_z, cell_y, cell_x = _cell_index(point, cell_size, grid_shape)
        for dz in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                for dx in range(-reach, reach + 1):
                    idx_list = grid.get(
                        (
                            (cell_z + dz) % grid_shape[0],
                            (cell_y + dy) % grid_shape[1],
                            (cell_x + dx) % grid_shape[2],
                        )
                    )
                    if idx_list:
                        candidate_neighbor_indices.update(idx_list)

    if not candidate_neighbor_indices:
        return False

    ref_idx = np.fromiter(candidate_neighbor_indices, dtype=np.int64)
    return _reject_by_points(candidate, occ_points[ref_idx], occ_radii[ref_idx])


def run_cell_list_exact(candidates: list[Candidate], cell_size: float = 16.0) -> dict[str, object]:
    accepted: list[Candidate] = []
    capacity = _occupied_point_capacity(candidates)
    occ_points = np.empty((capacity, 3), dtype=np.float64)
    occ_radii = np.empty((capacity,), dtype=np.float64)
    occ_count = 0
    max_occ_radius = 0.0
    grid_shape = (
        int(math.ceil(BOX_Z / cell_size)),
        int(math.ceil(BOX_XY / cell_size)),
        int(math.ceil(BOX_XY / cell_size)),
    )
    grid: dict[tuple[int, int, int], list[int]] = defaultdict(list)

    for candidate in candidates:
        if _reject_by_cell_list(
            candidate,
            occ_points[:occ_count],
            occ_radii[:occ_count],
            grid,
            cell_size,
            grid_shape,
            max_occ_radius,
        ):
            continue

        accepted.append(candidate)
        point_count = int(candidate.points_zyx.shape[0])
        start = occ_count
        stop = occ_count + point_count
        occ_points[start:stop] = candidate.points_zyx
        occ_radii[start:stop] = float(candidate.radius)

        for point_idx in range(start, stop):
            grid[_cell_index(occ_points[point_idx], cell_size, grid_shape)].append(point_idx)

        occ_count = stop
        max_occ_radius = max(max_occ_radius, float(candidate.radius))

    return {
        "accepted": accepted,
        "occ_point_count": int(occ_count),
    }


def run_center_prefilter_tree(candidates: list[Candidate]) -> dict[str, object]:
    accepted: list[Candidate] = []
    accepted_centers = np.empty((len(candidates), 3), dtype=np.float64)
    accepted_half_lengths = np.empty((len(candidates),), dtype=np.float64)
    accepted_radii = np.empty((len(candidates),), dtype=np.float64)
    center_tree: cKDTree | None = None
    accepted_count = 0
    max_half_length = 0.0
    max_radius = 0.0
    occ_point_count = 0

    for candidate in candidates:
        reject = False
        candidate_center = _candidate_center_zyx(candidate)

        if center_tree is not None:
            search_radius = (
                float(candidate.length) / 2.0
                + max_half_length
                + float(int(float(candidate.radius) + max_radius))
            )
            neighbor_idx = center_tree.query_ball_point(candidate_center, r=search_radius)
            if neighbor_idx:
                neighbor_idx_array = np.asarray(neighbor_idx, dtype=np.int64)
                center_sq_dist = _periodic_center_sq_dists(candidate_center, accepted_centers[neighbor_idx_array])
                exact_thresholds = (
                    float(candidate.length) / 2.0
                    + accepted_half_lengths[neighbor_idx_array]
                    + np.trunc(float(candidate.radius) + accepted_radii[neighbor_idx_array])
                )
                possible_idx = neighbor_idx_array[center_sq_dist < (exact_thresholds * exact_thresholds)]
                for ref_idx in possible_idx:
                    if _reject_candidate_pair(candidate, accepted[int(ref_idx)]):
                        reject = True
                        break

        if reject:
            continue

        accepted.append(candidate)
        accepted_centers[accepted_count] = candidate_center
        accepted_half_lengths[accepted_count] = float(candidate.length) / 2.0
        accepted_radii[accepted_count] = float(candidate.radius)
        accepted_count += 1
        occ_point_count += int(candidate.points_zyx.shape[0])
        max_half_length = max(max_half_length, float(candidate.length) / 2.0)
        max_radius = max(max_radius, float(candidate.radius))
        center_tree = cKDTree(accepted_centers[:accepted_count], boxsize=[BOX_Z, BOX_XY, BOX_XY])

    return {
        "accepted": accepted,
        "occ_point_count": int(occ_point_count),
    }


def run_kdtree_dual_numba_recent(candidates: list[Candidate], rebuild_threshold_points: int = 256) -> dict[str, object]:
    accepted: list[Candidate] = []
    tree_points = np.empty((0, 3), dtype=np.float64)
    tree_radii = np.empty((0,), dtype=np.float64)
    recent_points = np.empty((0, 3), dtype=np.float64)
    recent_radii = np.empty((0,), dtype=np.float64)
    tree: cKDTree | None = None
    points_since_rebuild = 0

    for candidate in candidates:
        reject = _reject_by_kdtree(candidate, tree, tree_radii)
        if not reject and recent_points.size:
            reject = bool(
                _reject_by_points_numba(
                    candidate.points_zyx,
                    float(candidate.radius),
                    recent_points,
                    recent_radii,
                )
            )
        if reject:
            continue

        accepted.append(candidate)
        new_points = candidate.points_zyx
        new_radii = np.full(new_points.shape[0], float(candidate.radius), dtype=np.float64)
        recent_points = new_points.copy() if recent_points.size == 0 else np.vstack((recent_points, new_points))
        recent_radii = new_radii if recent_radii.size == 0 else np.concatenate((recent_radii, new_radii))
        points_since_rebuild += int(new_points.shape[0])

        if points_since_rebuild >= rebuild_threshold_points:
            tree_points = recent_points.copy() if tree_points.size == 0 else np.vstack((tree_points, recent_points))
            tree_radii = recent_radii.copy() if tree_radii.size == 0 else np.concatenate((tree_radii, recent_radii))
            tree = cKDTree(tree_points, boxsize=[BOX_Z, BOX_XY, BOX_XY])
            recent_points = np.empty((0, 3), dtype=np.float64)
            recent_radii = np.empty((0,), dtype=np.float64)
            points_since_rebuild = 0

    if recent_points.size:
        tree_points = recent_points.copy() if tree_points.size == 0 else np.vstack((tree_points, recent_points))

    return {
        "accepted": accepted,
        "occ_point_count": int(tree_points.shape[0]),
    }


def _dynamic_rebuild_threshold(tree_point_count: int, min_points: int, max_points: int) -> int:
    scaled = int(min_points + 0.75 * math.sqrt(max(tree_point_count, 0)))
    return max(min_points, min(max_points, scaled))


def run_kdtree_dual_dynamic(
    candidates: list[Candidate],
    min_rebuild_points: int = 128,
    max_rebuild_points: int = 1024,
) -> dict[str, object]:
    accepted: list[Candidate] = []
    tree_points = np.empty((0, 3), dtype=np.float64)
    tree_radii = np.empty((0,), dtype=np.float64)
    recent_points = np.empty((0, 3), dtype=np.float64)
    recent_radii = np.empty((0,), dtype=np.float64)
    tree: cKDTree | None = None
    points_since_rebuild = 0
    direct_recent_time = 0.0
    rebuild_time_estimate = 0.0

    for candidate in candidates:
        reject = _reject_by_kdtree(candidate, tree, tree_radii)
        if not reject and recent_points.size:
            recent_check_start = time.perf_counter()
            reject = bool(
                _reject_by_points_numba(
                    candidate.points_zyx,
                    float(candidate.radius),
                    recent_points,
                    recent_radii,
                )
            )
            direct_recent_time += time.perf_counter() - recent_check_start
        if reject:
            continue

        accepted.append(candidate)
        new_points = candidate.points_zyx
        new_radii = np.full(new_points.shape[0], float(candidate.radius), dtype=np.float64)
        recent_points = new_points.copy() if recent_points.size == 0 else np.vstack((recent_points, new_points))
        recent_radii = new_radii if recent_radii.size == 0 else np.concatenate((recent_radii, new_radii))
        points_since_rebuild += int(new_points.shape[0])

        threshold = _dynamic_rebuild_threshold(int(tree_points.shape[0]), min_rebuild_points, max_rebuild_points)
        should_rebuild = points_since_rebuild >= threshold
        if not should_rebuild and rebuild_time_estimate > 0.0 and points_since_rebuild >= min_rebuild_points:
            should_rebuild = direct_recent_time >= rebuild_time_estimate

        if should_rebuild:
            rebuild_start = time.perf_counter()
            tree_points = recent_points.copy() if tree_points.size == 0 else np.vstack((tree_points, recent_points))
            tree_radii = recent_radii.copy() if tree_radii.size == 0 else np.concatenate((tree_radii, recent_radii))
            tree = cKDTree(tree_points, boxsize=[BOX_Z, BOX_XY, BOX_XY])
            rebuild_elapsed = time.perf_counter() - rebuild_start
            rebuild_time_estimate = rebuild_elapsed if rebuild_time_estimate == 0.0 else 0.5 * (
                rebuild_time_estimate + rebuild_elapsed
            )
            recent_points = np.empty((0, 3), dtype=np.float64)
            recent_radii = np.empty((0,), dtype=np.float64)
            points_since_rebuild = 0
            direct_recent_time = 0.0

    if recent_points.size:
        tree_points = recent_points.copy() if tree_points.size == 0 else np.vstack((tree_points, recent_points))

    return {
        "accepted": accepted,
        "occ_point_count": int(tree_points.shape[0]),
    }


def build_dynamic_geometry_rows(
    seed: int = SEED,
    num_trials: int = NUM_TRIALS,
    radius_mu: float = RADIUS_MU,
    radius_sigma: float = RADIUS_SIGMA,
    theta_mu: float = THETA_MU,
    theta_sigma: float = THETA_SIGMA,
    length_lower: float = LENGTH_LOWER,
    length_upper: float = LENGTH_UPPER,
) -> list[tuple[float, ...]]:
    candidates = generate_candidates(
        seed=seed,
        num_trials=num_trials,
        radius_mu=radius_mu,
        radius_sigma=radius_sigma,
        theta_mu=theta_mu,
        theta_sigma=theta_sigma,
        length_lower=length_lower,
        length_upper=length_upper,
    )
    result = run_kdtree_dual_dynamic(candidates)
    accepted = result["accepted"]
    return candidate_rows(accepted)  # type: ignore[arg-type]


def write_geometry_csv(rows: list[tuple[float, ...]], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = "x_center,y_center,z_center,psi,theta,length,radius"
    array = np.asarray(rows, dtype=np.float64)
    np.savetxt(out_path, array, delimiter=",", header=header, comments="")
    return out_path


def _load_create_all_cnt() -> Callable[..., object]:
    try:
        from fiberRSA.fiberRSA import create_all_CNT  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "tutorial_baseline requires the tutorial dependencies (including numba/cupy) "
            "to be importable in the active environment."
        ) from exc
    return create_all_CNT


def run_tutorial_baseline(seed: int = SEED, num_trials: int = NUM_TRIALS) -> dict[str, object]:
    create_all_CNT = _load_create_all_cnt()
    np.random.seed(seed)
    cnts = create_all_CNT(
        num_trials,
        RADIUS_MU,
        RADIUS_SIGMA,
        THETA_MU,
        THETA_SIGMA,
        LENGTH_LOWER,
        LENGTH_UPPER,
        BOX_XY,
        BOX_Z,
    )
    rows = [
        (
            float(cnt.x_center),
            float(cnt.y_center),
            float(cnt.z_center),
            float(cnt.psi),
            float(cnt.theta),
            float(cnt.length),
            float(cnt.radius),
        )
        for cnt in cnts
    ]
    return {
        "rows": rows,
        "accepted_count": len(rows),
    }


def load_reference_rows() -> list[tuple[float, ...]]:
    data = np.genfromtxt(REFERENCE_GEOMETRY_PATH, delimiter=",", names=True, dtype=np.float64)
    return [
        (
            float(row["x_center"]),
            float(row["y_center"]),
            float(row["z_center"]),
            float(row["psi"]),
            float(row["theta"]),
            float(row["length"]),
            float(row["radius"]),
        )
        for row in data
    ]


def rows_match_reference(rows: list[tuple[float, ...]], reference_rows: list[tuple[float, ...]], atol: float = 1e-5) -> bool:
    if len(rows) != len(reference_rows):
        return False
    lhs = np.asarray(rows, dtype=np.float64)
    rhs = np.asarray(reference_rows, dtype=np.float64)
    return bool(np.allclose(lhs, rhs, atol=atol, rtol=0.0))


def _benchmark(
    name: str,
    fn: Callable[[], dict[str, object]],
    reference_rows: list[tuple[float, ...]],
) -> dict[str, object]:
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start

    if "accepted" in result:
        rows = [_candidate_row(candidate) for candidate in result["accepted"]]  # type: ignore[index]
        accepted_count = len(rows)
    else:
        rows = result["rows"]  # type: ignore[index]
        accepted_count = int(result["accepted_count"])  # type: ignore[index]

    return {
        "name": name,
        "elapsed_sec": elapsed,
        "accepted_count": accepted_count,
        "matches_reference_seed12345": rows_match_reference(rows, reference_rows),
        "first_row": rows[0] if rows else None,
        "last_row": rows[-1] if rows else None,
        "occ_point_count": result.get("occ_point_count"),
    }


def warm_tutorial_baseline() -> None:
    create_all_CNT = _load_create_all_cnt()
    np.random.seed(0)
    create_all_CNT(
        2,
        RADIUS_MU,
        RADIUS_SIGMA,
        THETA_MU,
        THETA_SIGMA,
        LENGTH_LOWER,
        LENGTH_UPPER,
        BOX_XY,
        BOX_Z,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark alternate MWCNT RSA geometry-generation variants.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "tutorial_baseline",
            "kdtree_dual_128",
            "kdtree_dual_256",
            "kdtree_dual_512",
            "cupy_exact_batched",
            "cell_list_exact",
            "center_prefilter_tree",
            "kdtree_dual_numba_recent_256",
            "kdtree_dual_dynamic",
        ],
        help="Variant names to run.",
    )
    parser.add_argument("--num-trials", type=int, default=NUM_TRIALS, help="Number of post-initial RSA trials.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for candidate generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reference_rows = load_reference_rows()
    candidates = _rng_candidate_stream(seed=args.seed, num_trials=args.num_trials)

    benchmark_map: dict[str, Callable[[], dict[str, object]]] = {
        "tutorial_baseline": lambda: run_tutorial_baseline(args.seed, args.num_trials),
        "broadcast_exact": lambda: run_broadcast_exact(candidates),
        "kdtree_rebuild_each": lambda: run_kdtree_rebuild_each(candidates),
        "kdtree_dual_128": lambda: run_kdtree_dual(candidates, rebuild_threshold_points=128),
        "kdtree_dual_256": lambda: run_kdtree_dual(candidates, rebuild_threshold_points=256),
        "kdtree_dual_512": lambda: run_kdtree_dual(candidates, rebuild_threshold_points=512),
        "kdtree_dual_1024": lambda: run_kdtree_dual(candidates, rebuild_threshold_points=1024),
        "kdtree_dual_2048": lambda: run_kdtree_dual(candidates, rebuild_threshold_points=2048),
        "cupy_exact_batched": lambda: run_cupy_exact_batched(candidates),
        "cell_list_exact": lambda: run_cell_list_exact(candidates),
        "center_prefilter_tree": lambda: run_center_prefilter_tree(candidates),
        "kdtree_dual_numba_recent_256": lambda: run_kdtree_dual_numba_recent(candidates, rebuild_threshold_points=256),
        "kdtree_dual_dynamic": lambda: run_kdtree_dual_dynamic(candidates),
    }
    unknown = [name for name in args.variants if name not in benchmark_map]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}")
    if "tutorial_baseline" in args.variants:
        warm_tutorial_baseline()

    results = []
    for name in args.variants:
        print(f"Running {name}...", flush=True)
        result = _benchmark(name, benchmark_map[name], reference_rows)
        results.append(result)
        print(
            f"Finished {name}: time={result['elapsed_sec']:.3f}s "
            f"accepted={result['accepted_count']} "
            f"match_ref={result['matches_reference_seed12345']}",
            flush=True,
        )
    results_sorted = sorted(results, key=lambda item: item["elapsed_sec"])

    summary = {
        "seed": args.seed,
        "num_trials": args.num_trials,
        "params": {
            "radius_mu": RADIUS_MU,
            "radius_sigma": RADIUS_SIGMA,
            "theta_mu": THETA_MU,
            "theta_sigma": THETA_SIGMA,
            "length_lower": LENGTH_LOWER,
            "length_upper": LENGTH_UPPER,
            "box_xy": BOX_XY,
            "box_z": BOX_Z,
        },
        "variants_run": args.variants,
        "results": results_sorted,
    }

    json_path = OUT_DIR / "mwcnt_rsa_benchmark_results.json"
    json_path.write_text(json.dumps(summary, indent=2))

    lines = [
        "# MWCNT RSA Benchmark",
        "",
        f"- Seed: `{args.seed}`",
        f"- Candidate count: `{args.num_trials + 1}`",
        f"- Parameters: `radius_mu={RADIUS_MU}`, `radius_sigma={RADIUS_SIGMA}`, "
        f"`theta_mu={THETA_MU}`, `theta_sigma={THETA_SIGMA}`, "
        f"`length=[{LENGTH_LOWER}, {LENGTH_UPPER}]`, `box=[{BOX_Z}, {BOX_XY}, {BOX_XY}]`",
        "",
        "| Variant | Time (s) | Accepted | Matches seed12345 reference | Occupied points |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for item in results_sorted:
        lines.append(
            f"| `{item['name']}` | {item['elapsed_sec']:.3f} | {item['accepted_count']} | "
            f"{item['matches_reference_seed12345']} | {item['occ_point_count'] or '-'} |"
        )
    md_path = OUT_DIR / "mwcnt_rsa_benchmark_results.md"
    md_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    for item in results_sorted:
        print(
            f"{item['name']:>22s}  "
            f"time={item['elapsed_sec']:.3f}s  "
            f"accepted={item['accepted_count']}  "
            f"match_ref={item['matches_reference_seed12345']}"
        )


if __name__ == "__main__":
    main()
