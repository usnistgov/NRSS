"""Replay the three maintained CoreShell A-wedge metrics on CPU.

Run in an installed NRSS development checkout. This independent diagnostic
uses the vendored morphology, optics and files. It emulates the interior of
the historic GPU WPIntegrator remesh. It never writes a reference.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import fft, sparse


def read_inputs(repo):
    sys.path[:0] = [str(repo / "tests/validation/lib"), str(repo / "src")]
    import core_shell as cs

    material = cs._build_core_shell_fields(cs.BASELINE_SCENARIO)
    constants = cs._load_optical_constants()
    optics = [constants[i].opt_constants for i in (1, 2, 3)]
    def table(array):
        array = array.transpose("energy", "q").sortby("energy")
        return {key: np.asarray(value) for key, value in
                (("energy", array.energy.values), ("q", array.q.values), ("A", array.values))}
    return material, optics, table(cs.load_sim_reference_awedge()), table(cs.load_experimental_reference_awedge())


def basis_fields(material, half):
    nz, ny, nx = material["mat1_vfrac"].shape
    offset = np.arange(-half, half+1)
    def transform(field):
        g = fft.fftn(field, workers=2)
        g[0, 0, 0] = (g[0, 0, 1] + g[0, 0, -1] + g[0, 1, 0] + g[0, -1, 0]
                      + g[1, 0, 0] + g[-1, 0, 0]) / 6
        return g[:, (-offset[:, None]) % ny, (-offset) % nx].astype(np.complex128)
    fields = [transform(material[f"mat{i}_vfrac"]) for i in (1, 2, 3)]
    phi = material["mat2_vfrac"] * material["mat2_s"]
    theta, psi = material["mat2_theta"], material["mat2_psi"]
    u = (np.sin(theta)*np.cos(psi), np.sin(theta)*np.sin(psi), np.cos(theta))
    fields.append(transform(phi))
    fields.extend(transform(phi*u[i]*u[j]) for i,j in ((0,0),(0,1),(0,2),(1,1),(1,2)))
    return fields


def optical_weights(table, energy):
    dp,bp,dt,bt = table[energy]
    par, perp = complex(1-dp,bp), complex(1-dt,bt)
    iso = ((par+2*perp)/3)**2
    return iso-1, perp**2-iso, par**2-perp**2


def endpoint_panels(fields, optics, shape, half, energy, corrected):
    nz, ny, nx = shape
    d = 2.5
    k = 2*np.pi*energy/1239.84197
    offsets = np.arange(-half, half+1)
    def axis(length, index):
        if corrected:
            return (index - length//2) * 2*np.pi/(length*d)
        return -np.pi/d + index*2*np.pi/(d*(length-1))
    a = axis(nx, nx//2+offsets)[None, :]
    b = axis(ny, ny//2+offsets)[:, None]
    z_axis = axis(nz, np.arange(nz))
    radius2 = a*a + b*b
    qz = -radius2/(k+np.sqrt(k*k-radius2))
    position = (qz-z_axis[0])/(z_axis[1]-z_axis[0])
    lower = np.floor(position).astype(int)
    frac = position - lower
    assert lower.min() >= 0 and (lower+1).max() < nz
    rows, cols = np.indices(radius2.shape)
    coeff = [optical_weights(table, energy) for table in optics]
    def endpoint(iz):
        indices = ((nz//2-iz) % nz, rows, cols)
        values = [g[indices] for g in fields]
        diagonal = sum(v*w[0] for v,w in zip(values[:3], coeff)) + coeff[1][1]*values[3]
        xx, xy, xz, yy, yz = [coeff[1][2]*v for v in values[4:]]
        xx += diagonal
        yy += diagonal
        direction = (a, b, k+z_axis[iz])
        def project(vec):
            dot = sum(p*v for p,v in zip(direction, vec))
            return [k*k*v-p*dot for p,v in zip(direction, vec)]
        px = project((xx,xy,xz))
        py = project((xy,yy,yz))
        # Global intensity factors cancel in (H-V)/(H+V).
        return np.array([sum(abs(v)**2 for v in px), sum(abs(v)**2 for v in py),
                         2*sum((x*y.conj()).real for x,y in zip(px,py))])
    return (1-frac)*endpoint(lower) + frac*endpoint(lower+1)


def wedge_operator(half, radial_indices):
    """Compose two bilinear interpolations on the interior, then average each chi wedge."""
    width = 2*half + 1
    theta = np.deg2rad(np.arange(361))[:,None]  # baseline EAngleRotation 0,1,360
    ct, st = np.cos(theta), np.sin(theta)
    polarization = (ct*ct, st*st, ct*st)
    rows = []
    chi = np.linspace(-179.5, 179.5, 360)
    for center in (180., 90.):
        members = np.flatnonzero(abs((chi-center+180.)%360.-180.) < 10)
        azimuth = np.deg2rad(members)[None, :]
        for radius in radial_indices:
            # In the historical WP GPU path the radial step is one detector pixel.
            y, x = np.sin(azimuth)*radius, np.cos(azimuth)*radius
            iy, ix = np.floor(y), np.floor(x)
            fy, fx = y-iy, x-ix
            total = np.zeros((3, width*width), dtype=np.float64)
            for dy, wy in ((0,1-fy),(1,fy)):
                for dx, wx in ((0,1-fx),(1,fx)):
                    # Inverse detector rotation in (y,x) pixel coordinates.
                    sy = ct*(iy+dy) + st*(ix+dx) + half
                    sx = -st*(iy+dy) + ct*(ix+dx) + half
                    y0, x0 = np.floor(sy).astype(int), np.floor(sx).astype(int)
                    ty, tx = sy-y0, sx-x0
                    assert min(y0.min(),x0.min()) >= 0 and max(y0.max(),x0.max())+1 < width
                    for oy, vy in ((0,1-ty),(1,ty)):
                        for ox, vx in ((0,1-tx),(1,tx)):
                            ids = ((y0+oy)*width+x0+ox).ravel()
                            weight = wy*wx*vy*vx/(theta.size*members.size)
                            for j, pol in enumerate(polarization):
                                total[j] += np.bincount(ids, weights=(weight*pol).ravel(), minlength=width*width)
            rows.append(sparse.csr_matrix(total.reshape(1, -1)))
    return sparse.vstack(rows, format="csr")


def difference(source, target, coord):
    delta = np.asarray(source) - np.asarray(target)
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite A in the maintained comparison window")
    i = np.argmax(abs(delta))
    return {"max_abs":float(abs(delta[i])), "rmse":float(np.sqrt(np.mean(delta**2))),
            "worst_coordinate":float(coord[i]), "signed_at_worst":float(delta[i])}


def run(material, optics, sim, exp):
    energies, q_axis = sim["energy"], sim["q"]
    assert np.array_equal(energies, exp["energy"]) and np.array_equal(q_axis, exp["q"])
    shape = material["mat1_vfrac"].shape
    assert shape == (32, 512, 512)
    indices = np.flatnonzero((q_axis >= .02) & (q_axis <= .4))
    q = q_axis[indices]
    half = int(indices[-1]+5)
    fields = basis_fields(material, half)
    op = wedge_operator(half, indices)
    size = (2*half+1)**2
    dc = half*(2*half+1) + half
    dc_weight = float(np.abs(op[:, [dc, size+dc, 2*size+dc]]).sum())
    assert dc_weight == 0.
    curves = {key:np.empty((len(energies),len(indices))) for key in ("legacy","corrected")}
    for ie, e in enumerate(energies):
        for label, curve in curves.items():
            panels = endpoint_panels(fields, optics, shape, half, float(e), label=="corrected")
            horiz, vert = (op @ panels.ravel()).reshape(2, -1)
            curve[ie] = (horiz-vert)/(horiz+vert)
    curves["saved_sim"] = sim["A"][:,indices]
    curves["measured"] = exp["A"][:,indices]
    summary = {"q_points":len(q), "q_min":float(q[0]), "q_max":float(q[-1]),
               "energy_count":len(energies), "central_pixel_weight_in_wedges":dc_weight,
               "q_window_full_map_legacy_vs_saved_max_abs":float(np.max(abs(curves["legacy"]-curves["saved_sim"])))}
    for lhs,rhs in [("legacy","saved_sim"),("corrected","saved_sim"),("corrected","legacy"),
                    ("legacy","measured"),("corrected","measured")]:
        summary[f"{lhs}_vs_{rhs}"] = {"A_E": difference(curves[lhs].mean(1),curves[rhs].mean(1),energies)}
        for e in (284.7, 285.2):
            i = np.flatnonzero(np.isclose(energies,e,rtol=0,atol=1e-9)).item()
            summary[f"{lhs}_vs_{rhs}"][str(e)] = difference(curves[lhs][i],curves[rhs][i],q)
    return {"energy":energies,"q":q,**curves}, summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=Path.cwd())
    parser.add_argument("--out",type=Path,default=Path("test-reports/fft-registration"))
    args=parser.parse_args()
    arrays, summary = run(*read_inputs(args.repo.resolve()))
    args.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out/"core_shell_curves.npz",**arrays)
    (args.out/"core_shell_metrics.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
