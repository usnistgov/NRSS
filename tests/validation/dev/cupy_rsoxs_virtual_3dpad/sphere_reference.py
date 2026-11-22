"""Full-panel sphere reference plus selected literal-voxel Fourier sums (ZYX, nm)."""
import hashlib
import numpy as np


SIZES = (128, 256, 512, 1024, 2048, 4096, 8192)
CROP = 128
D = 1.0
RADIUS = 35.0
ENERGY = 285.0
K = 2 * np.pi * ENERGY / 1239.84197
ROOTS = np.array([4.493409457909064, 7.725251836937707, 10.90412165942890, 14.06619391283147])
PARTICLE = (0., 2e-4, 0., 2e-4)
MEDIUM = (.0003, .00006, .0003, .00006)  # synthetic isotropic test medium
VACUUM = (0., 0., 0., 0.)
SPHERE_SHA256 = "25548b6bd66496101501e5ebdc911417e31bc0f7d9679166753d805e928ede85"


def sphere_fraction():
    """Four midpoint samples per voxel axis at the boundary, full occupancy in the interior."""
    coord = (np.arange(CROP) - (CROP-1)/2) * D
    z, y, x = np.ogrid[:CROP, :CROP, :CROP]
    dist = np.sqrt(coord[z]**2 + coord[y]**2 + coord[x]**2)
    half_diagonal = np.sqrt(3.) * D / 2
    fraction = (dist <= RADIUS - half_diagonal).astype(np.float32)
    izb, iyb, ixb = np.where(abs(dist - RADIUS) <= half_diagonal)
    offsets = (np.arange(4) + .5) * D / 4 - D/2
    inside = np.zeros(len(izb), dtype=np.int16)
    for dz in offsets:
        for dy in offsets:
            for dx in offsets:
                rr = (coord[izb] + dz)**2 + (coord[iyb] + dy)**2 + (coord[ixb] + dx)**2
                inside += rr <= RADIUS**2
    fraction[izb, iyb, ixb] = inside / 64
    return fraction


def fingerprint(fraction):
    digest = hashlib.sha256(fraction.astype("<f4", copy=False).tobytes()).hexdigest()
    if digest != SPHERE_SHA256:
        raise ValueError(f"Sphere occupancy differs from the fixed benchmark: {digest}")
    return digest


def material_response(optics):
    dp, bp, dt, bt = optics
    return ((complex(1-dp, bp) + 2*complex(1-dt, bt))/3)**2 - 1


def contrast(fill_id):
    return material_response(PARTICLE) - material_response(MEDIUM if fill_id == 2 else VACUUM)


def branch_intensity_scale(fill_id):
    """Predicted FP32 optical-coefficient effect relative to tabulated continuum contrast."""
    def stored_s0(optics):
        dp, bp, dt, bt = optics
        parallel = np.complex64(complex(1-dp, bp))
        transverse = np.complex64(complex(1-dt, bt))
        numerator = np.complex64((parallel + 2.0*transverse)**2)
        return complex(np.complex64(numerator/np.float32(9.) - np.complex64(1.)))
    matrix = MEDIUM if fill_id == 2 else VACUUM
    stored = stored_s0(PARTICLE) - stored_s0(matrix)
    return abs(stored/contrast(fill_id))**2


def sphere_amplitude(x):
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    small = abs(x) < 1e-3
    result[small] = 1 - x[small]**2/10 + x[small]**4/280
    u = x[~small]
    result[~small] = 3*(np.sin(u) - u*np.cos(u))/u**3
    return result


def forward_scale(fill_id):
    voxel_volume_count = (4*np.pi/3) * (RADIUS/D)**3
    return (K**2 * abs(contrast(fill_id)) * voxel_volume_count / (4*np.pi))**2


def axis(size):
    return 2*np.pi*np.fft.fftshift(np.fft.fftfreq(size, d=D))


def q_magnitude(qx, qy):
    transverse_sq = qx*qx + qy*qy
    qz = -transverse_sq / (K + np.sqrt(np.maximum(K*K - transverse_sq, 0.)))
    return np.sqrt(transverse_sq + qz*qz)


def continuum(qx, qy, fill_id):
    q = q_magnitude(qx, qy)
    dipole = np.maximum(1 - (qx/K)**2, 0.)
    return forward_scale(fill_id) * sphere_amplitude(q*RADIUS)**2 * dipole


def line_metrics(panel, size, fill_id):
    """Along +qy, E parallel to X gives a unity normalized dipole factor."""
    mid = size//2
    qy = axis(size)[mid:]
    valid = qy < K
    q = q_magnitude(0., qy[valid])
    intensity = panel[mid:, mid][valid] / forward_scale(fill_id)
    rg = np.sqrt(3./5) * RADIUS
    fit = (q > 0) & (q*rg <= 1) & (intensity > 0)
    rg_fit = None
    if fit.sum() >= 3:
        slope = np.polyfit(q[fit]**2, np.log(intensity[fit]), 1)[0]
        rg_fit = float(np.sqrt(-3*slope)) if slope < 0 else None
    minima = []
    for i, root in enumerate(ROOTS[:3]):
        left = (ROOTS[i-1]+root)/2 if i else root*.65
        right = (root+ROOTS[i+1])/2
        points = np.flatnonzero((q*RADIUS > left) & (q*RADIUS < right))
        minimum = None
        if len(points) >= 3:
            j = points[np.argmin(intensity[points])]
            if points[0] < j < points[-1]:
                ids = slice(j-1, j+2)
                a, b, _ = np.polyfit(q[ids]-q[j], intensity[ids], 2)
                vertex = q[j] - b/(2*a) if a > 0 else np.nan
                if q[j-1] <= vertex <= q[j+1]:
                    minimum = {"q_bin": float(q[j]), "q_quadratic": float(vertex),
                               "I_bin_over_I0": float(intensity[j])}
        minima.append({"q_exact": float(root/RADIUS), "resolved_minimum": minimum})
    return {"guinier_nonzero_points": int(fit.sum()), "Rg_reference_nm": rg, "Rg_fit_nm": rg_fit,
            "minima": minima}


def compare_panel(panel, size, fill_id, block_rows=64):
    """Evaluate every pixel against physical Ewald q and the X-polarized continuum sphere."""
    if panel.shape != (size,size):
        raise ValueError(f"Expected {(size,size)}, received {panel.shape}")
    qaxis = axis(size)
    limit = min(K, (size//2)*2*np.pi/(size*D))
    limit_sq = np.float32(limit*(2*K-limit))
    totals = {name: np.zeros(3) for name in ("all_non_dc", "guinier", "lobes", "high_q")}
    count = 0
    for start in range(0,size,block_rows):
        end = min(size,start+block_rows)
        qy, qx = qaxis[start:end,None], qaxis[None,:]
        support = ((qx*qx + qy*qy).astype(np.float32) <= limit_sq)
        support[:, -1] = False
        if end == size:
            support[-1, :] = False
        block = np.asarray(panel[start:end], dtype=np.float64)
        if not np.array_equal(np.isfinite(block), support) or np.isinf(block).any():
            raise AssertionError(f"Finite support differs in rows {start}..{end-1}")
        count += int(support.sum())
        yy, xx = np.nonzero(support)
        qxs, qys = qaxis[xx], qaxis[start+yy]
        q = q_magnitude(qxs,qys)
        ideal = continuum(qxs,qys,fill_id)
        measured = block[yy,xx]
        bands = (q>0, (q>0)&(q*np.sqrt(.6)*RADIUS<=1), (q*RADIUS>=2)&(q*RADIUS<=14), q>=1.)
        for values, select in zip(totals.values(), bands):
            values += (select.sum(), np.sum((measured[select]-ideal[select])**2),np.sum(ideal[select]**2))
    metrics = {key: {"pixels": int(n), "relative_l2": float(np.sqrt(err/ref)) if ref else None}
               for key,(n,err,ref) in totals.items()}
    return {"valid_pixels": count, "bands": metrics, "dq_nm_inv": 2*np.pi/(size*D),
            "dc_simulated": float(panel[size//2,size//2]), "unrepaired_I0": forward_scale(fill_id),
            **line_metrics(panel,size,fill_id)}


def discrete_anchors(fraction, size, fill_id):
    """Exact voxel sums at signed array bins, using the registered Igor order and repaired DC."""
    mid = size//2
    dq = 2*np.pi/(size*D)
    pos = (np.arange(CROP)+(size-CROP)//2)/size
    # Each pair is (signed y, signed x). Include low q, two signs, and fixed physical wavevectors.
    offsets = {(0,0),(1,0),(0,-1),(0,2),(-2,1)}
    for y,x in ((.08,.11),(.19,-.07),(-.35,.16),(.51,.33),(.77,-.81)):
        offsets.add((round(y/dq), round(x/dq)))
    points = np.array(sorted(offsets), dtype=int)
    qy,qx = points.T * dq
    rsq = (qx*qx+qy*qy).astype(np.float32)
    qz = -rsq/(np.float32(K)+np.sqrt(np.maximum(np.float32(K*K)-rsq,0.)))
    pos_z = qz/np.float32(dq)
    lower = np.floor(pos_z).astype(int)
    frac = pos_z-lower
    qxphase = np.exp(2j*np.pi*pos[:,None]*points[None,:,1])
    qyphase = np.exp(2j*np.pi*pos[:,None]*points[None,:,0])
    xy = np.einsum("zyp,yp->zp", fraction @ qxphase, qyphase)
    # The +/- pair on each axis has a real sum for a real volume fraction.
    dc = 0.
    for dim in range(3):
        others = tuple(i for i in range(3) if i != dim)
        dc += fraction.sum(axis=others, dtype=np.float64) @ np.cos(2*np.pi*pos) / 3
    intensities = []
    for zbin in (lower,lower+1):
        amp = np.sum(xy * np.exp(2j*np.pi*pos[:,None]*zbin[None,:]),axis=0)
        amp[(points==0).all(axis=1) & (zbin==0)] = dc
        # Uniform material occupying the whole box contributes only raw DC; repair removes it.
        weight = (K*K-qx*qx)**2 + (qx*qy)**2 + (qx*(K+dq*zbin))**2
        intensities.append(abs(amp*contrast(fill_id)/(4*np.pi))**2 * weight)
    values = (1-frac)*intensities[0] + frac*intensities[1]
    return points+mid, values


def compare_voxel_panels(actual, reference, size, fill_id, block_rows=64):
    """Compare two complete finite-object panels, in small host row blocks."""
    if actual.shape != reference.shape or actual.shape != (size, size):
        raise AssertionError("Different logical detector shapes")
    qaxis = axis(size)
    peak = worst = sum_delta_sq = sum_ref_sq = tail_delta_sq = tail_ref_sq = 0.
    for start in range(0, size, block_rows):
        stop = min(size, start + block_rows)
        target = reference[start:stop]
        test = actual[start:stop]
        valid = np.isfinite(target)
        if not np.array_equal(valid, np.isfinite(test)) or np.isinf(test).any():
            raise AssertionError(f"Different support in rows {start}..{stop-1}")
        iy, ix = np.nonzero(valid)
        if not iy.size:
            continue
        ref = target[iy, ix].astype(np.float64)
        diff = test[iy, ix].astype(np.float64) - ref
        peak = max(peak, abs(ref).max())
        worst = max(worst, abs(diff).max())
        sum_ref_sq += np.sum(ref**2)
        sum_delta_sq += np.sum(diff**2)
        tail = q_magnitude(qaxis[ix], qaxis[start+iy]) >= 1.
        tail_ref_sq += np.sum(ref[tail]**2)
        tail_delta_sq += np.sum(diff[tail]**2)
    return {"relative_l2": float(np.sqrt(sum_delta_sq/sum_ref_sq)),
            "max_abs_over_peak": float(worst/peak),
            "max_abs_over_I0": float(worst/forward_scale(fill_id)),
            "high_q_relative_l2": float(np.sqrt(tail_delta_sq/tail_ref_sq))}


def assess_full_reference(actual_metrics, reference_metrics, differences, size):
    failures = []
    for name in ("relative_l2", "max_abs_over_peak"):
        if differences[name] > 2e-5:
            failures.append(f"voxel panel {name} > 2e-5")
    if differences["high_q_relative_l2"] > 2e-3:
        failures.append("voxel tail relative L2 > 2e-3")
    for band, observed in actual_metrics["bands"].items():
        ref = reference_metrics["bands"][band]
        if observed["pixels"] != ref["pixels"]:
            failures.append(f"different {band} pixel count")
        if ref["relative_l2"] is not None:
            limit = max(2e-4, .005 * ref["relative_l2"])
            if abs(observed["relative_l2"] - ref["relative_l2"]) > limit:
                failures.append(f"different continuum {band} error")
    ref_rg, rg = reference_metrics["Rg_fit_nm"], actual_metrics["Rg_fit_nm"]
    if (rg is None) != (ref_rg is None) or (rg is not None and abs(rg - ref_rg) > .1):
        failures.append("Guinier radius differs from voxel reference")
    if size >= 2048:
        positive = q_magnitude(0., axis(size)[size//2:])
        for minimum in actual_metrics["minima"]:
            q = minimum["q_exact"]
            i = int(np.searchsorted(positive, q))
            resolved = minimum["resolved_minimum"]
            if resolved is None or abs(resolved["q_quadratic"] - q) > positive[i] - positive[i-1]:
                failures.append(f"unresolved or displaced minimum at {q:g}")
    return failures
