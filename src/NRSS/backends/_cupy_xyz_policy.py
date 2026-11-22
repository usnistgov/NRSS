"""Select the bounded compact-tensor paths for virtual XYZ padding."""

from dataclasses import dataclass
import math


XYZ_PERFORMANCE_PROFILE_KEY = "_private_xyz_performance_profile"


@dataclass(frozen=True)
class XYZPerformancePlan:
    endpoint_plan: str = "legacy"
    energy_tile: tuple[int, bool] | None = None
    prune_zero_fields: bool = False
    factored_xy: bool | str = False


def xyz_performance_plan(state, occupied_shape, target, energies, layout, residency, chunk):
    profile = state.get(XYZ_PERFORMANCE_PROFILE_KEY, "auto")
    if profile not in {"auto", "legacy", "energy", "resolution"}:
        raise ValueError(f"Unknown private XYZ performance profile {profile!r}.")
    endpoint, tile = "legacy", None
    _, ny, nx = target
    if profile in {"auto", "energy"}:
        endpoint = "bitmap"
        distinct = sorted(set(energies))
        if len(distinct) >= 2 and all(math.isfinite(e) and e > 0 for e in distinct):
            # Short, densely sampled scans reuse nearby bins. Use the compact
            # span lookup only where its per-column metadata stays bounded.
            dense = (len(distinct) >= 8 and distinct[-1] <= 1.05 * distinct[0]
                     and max(b - a for a, b in zip(distinct, distinct[1:])) <= 0.01 * distinct[0])
            if dense and ny * nx <= 2048**2:
                endpoint = "spans"
            if dense and layout == "detector" and (residency == "device" or chunk >= 4):
                tile = (8, True)
            else:
                tile = (2, False) if dense else (1, False)
    resolution = profile in {"auto", "resolution"}
    oversampled = min(ny, nx) >= 1024 and ny >= 4 * occupied_shape[1] and nx >= 4 * occupied_shape[2]
    return XYZPerformancePlan(
        endpoint_plan=endpoint, energy_tile=tile, prune_zero_fields=resolution,
        factored_xy="auto" if resolution and oversampled else False,
    )
