"""Evaluate indexed high-q peaks on the maintained unpadded Bragg fixtures.

Run from an installed NRSS checkout. The CPU calculation uses the real voxel
fixture and a continuous axial Fourier sum at the same transverse FFT bins.
The public command measures Morphology.run with the same catalog and cuts.
"""
import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from scipy import fft, ndimage

SHARED = (0.95, 1.25)
EXTENDED = (1.90, 2.40)
BOX = 10
OFFSETS = np.arange(-BOX, BOX + 1)
DY, DX = np.meshgrid(OFFSETS, OFFSETS, indexing="ij")
BANDS = {"common": SHARED, "570-only": EXTENDED}


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def catalog(case, *, basis_override=None, include_extinctions=False):
    """Vectorized equivalent of predict_bragg_spots_3d for these fixtures."""
    from tests.validation.lib import bragg

    axes = [2 * np.pi * np.fft.fftshift(np.fft.fftfreq(n, case.phys_size_nm))
            for n in case.shape]
    az, ay, ax = axes
    a, basis = bragg.lattice_vectors_3d_nm(case.lattice_kind, case.lattice_constant_nm,
                                           case.azimuth_deg, case.tilt_x_deg, case.tilt_y_deg)
    if basis_override is not None:
        basis = basis_override
    reciprocal = bragg.reciprocal_vectors_nm_inv(a)
    beam = bragg.flat_detector_k_nm_inv(case.energy_eV)
    limit = int(np.ceil(2 * beam / np.linalg.norm(reciprocal, axis=1).min())) + 2
    kk, ll = np.meshgrid(np.arange(-limit,limit+1), np.arange(-limit,limit+1), indexing="ij")
    kl = np.column_stack((kk.ravel(),ll.ravel()))
    margin_x, margin_y = 2*(ax[1]-ax[0]), 2*(ay[1]-ay[0])
    rows = []
    for h in range(-limit, limit+1):
        hkl = np.column_stack((np.full(len(kl),h), kl))
        gx, gy, gz = (hkl @ reciprocal).T
        qz = bragg.flat_detector_qz_nm_inv(gx, gy, case.energy_eV)
        good = ((gx >= ax[0]-margin_x) & (gx <= ax[-1]+margin_x)
                & (gy >= ay[0]-margin_y) & (gy <= ay[-1]+margin_y)
                & ((abs(gx) > margin_x) | (abs(gy) > margin_y))
                & (abs(qz-gz) <= 3.5*(az[1]-az[0])))
        qperp = np.hypot(gx[good], gy[good])
        qabs = np.hypot(qperp, qz[good])
        form = bragg.sphere_form_factor_amplitude(qabs, case.particle_diameter_nm)
        for idx, p, q, f in zip(np.flatnonzero(good), qperp, qabs, form):
            indices = tuple(map(int,hkl[idx]))
            amplitude = abs(bragg.structure_factor_amplitude(hkl[idx], basis))
            weight = float(f*f*amplitude**2)
            rows.append((indices, gx[idx], gy[idx], p, q, (gz[idx]-qz[idx])/(az[1]-az[0]), weight, amplitude < 1.e-10))
    strongest = max(row[-2] for row in rows)
    groups = {}
    for hkl, gx, gy, p, q, dz, strength, extinct in rows:
        relative = strength/strongest
        if relative < 1e-12 and not (include_extinctions and extinct):
            continue
        key = (round(float(gx),9), round(float(gy),9))
        if key not in groups:
            groups[key] = dict(qx=gx, qy=gy, q_perp=p, q_abs=q, orders=[])
        groups[key]["orders"].append((hkl,float(dz),relative))
    return list(groups.values()), (ay,ax)


def select_targets(case):
    groups, axes = catalog(case)
    ay, ax = axes
    targets, inventory = [], []
    for name, (lo, hi) in BANDS.items():
        if name == "570-only" and case.energy_eV < 570:
            continue
        group = [g for g in groups if lo <= g["q_perp"] <= hi]
        n_all = sum(len(g["orders"]) for g in group)
        n_required = sum(s >= .01 for g in group for _,_,s in g["orders"])
        inventory.append(dict(case=case.case_id, energy=case.energy_eV, band=name,
                              allowed_hkl=n_all, distinct_xy=len(group), asserted_hkl=n_required))
        targets.extend((name,g) for g in group)
    anchor = next(g for g in groups if any(hkl==(1,0,0) for hkl,_,_ in g["orders"]))
    targets.append(("anchor",anchor))
    patches = []
    for band, group in targets:
        col, row = np.argmin(abs(ax-group["qx"])), np.argmin(abs(ay-group["qy"]))
        yy, xx = row+DY, col+DX
        if yy.min()<0 or xx.min()<0 or yy.max()>=len(ay) or xx.max()>=len(ax):
            raise ValueError("Peak ROI reaches the FFT boundary")
        patches.append(np.ravel_multi_index((yy,xx),(len(ay),len(ax))))
    return targets, np.stack(patches), inventory


def measure(targets, patches, ay, ax, images, anchor):
    """Use an actual local maximum and a common finite background for all arms."""
    # At zero tilt distinct XY centers are much farther apart than the 3-pixel match radius.
    # Multiple axial orders can contribute to the same XY center.
    rows = []
    pix = min(ax[1]-ax[0], ay[1]-ay[0])
    for n,(band,g) in enumerate(targets):
        yy, xx = np.unravel_index(patches[n], (len(ay),len(ax)))
        distance = np.hypot(ay[yy]-g["qy"], ax[xx]-g["qx"])/pix
        support = np.logical_and.reduce([np.isfinite(img[n]) for img in images.values()])
        core = (distance<=3) & ndimage.minimum_filter(support, size=3, mode="constant", cval=0)
        bg = (distance>=4) & (distance<=9) & support
        orders = sorted(g["orders"],key=lambda item:abs(item[1]))
        primary, dz, _ = orders[0]
        for mode, images_by_target in images.items():
            image = images_by_target[n]
            valid = bool(support.all() and bg.any() and core.any())
            finite = np.where(support,image,-np.inf)
            maxima = core & (finite==ndimage.maximum_filter(finite,size=3))
            found = bool(np.any(maxima) and valid)
            peak = ratio = normalized = dq = found_x = found_y = background = np.nan
            if found:
                loc = np.unravel_index(np.argmax(np.where(maxima,image,-np.inf)), image.shape)
                peak = float(image[loc])
                background = float(np.median(image[bg]))
                ratio = peak/max(background,1e-30)
                normalized, dq = peak/anchor, float(distance[loc])
                found_x, found_y = ax[xx[loc]], ay[yy[loc]]
            rows.append(dict(band=band, mode=mode, h=primary[0], k=primary[1], l=primary[2],
                             contributors="|".join(",".join(map(str,x[0])) for x in orders),
                             q_perp=g["q_perp"], q_abs=g["q_abs"], qx=g["qx"], qy=g["qy"],
                             signed_mismatch_dz=dz, max_ideal_relative=max(x[2] for x in orders),
                             support=valid, local_peak=found, dq_pixels=dq, peak=peak,
                             qx_found=found_x, qy_found=found_y, background=background,
                             peak_to_background=ratio, peak_over_anchor=normalized))
    return rows


def spectrum_columns(field, ids, *, double_precision=False):
    nz,ny,nx = field.shape
    iy,ix = np.unravel_index(ids,(ny,nx))
    output = np.empty((nz,len(ids)),np.complex128 if double_precision else np.complex64)
    native_y, native_x = (iy-ny//2)%ny, (ix-nx//2)%nx
    for z in range(0,nz,4):
        data = field[z:z+4]
        if double_precision:
            data = data.astype(np.float64)
        planes = fft.fft2(data,axes=(-2,-1),workers=3)
        output[z:z+len(planes)] = planes[:,native_y,native_x]
    return output


def optical_scale(beam):
    susceptibility = (1+2.e-4j)**2-1
    return beam**4 * abs(susceptibility/(4*np.pi))**2


def sample_columns(columns, ids, case):
    """Scalar x-polarized response with the production endpoint projection."""
    nz,ny,nx = case.shape
    iy,ix = np.unravel_index(ids,(ny,nx))
    d = case.phys_size_nm
    beam = 2*np.pi*case.energy_eV/1239.84197
    axial_spectrum = fft.fft(columns,axis=0,workers=3)
    results = {}
    for mode in ("legacy","corrected"):
        def axis(n,index):
            if mode=="legacy":
                return -np.pi/d + index*2*np.pi/(d*(n-1))
            return (index-n//2)*2*np.pi/(n*d)
        x, y, az = axis(nx,ix), axis(ny,iy), axis(nz,np.arange(nz))
        p2 = x*x+y*y
        propagates = p2 <= beam*beam
        qz = -p2/(beam+np.sqrt(np.maximum(0,beam*beam-p2)))
        pos = (qz-az[0])/(az[1]-az[0])
        lo = np.floor(pos).astype(int)
        valid = propagates & (lo>=0) & (lo+1<nz) & (ix<nx-1) & (iy<ny-1)
        if not np.all(valid):
            raise ValueError("The proposed benchmark bands require a common valid ROI")
        def endpoint(i):
            value = axial_spectrum[(i-nz//2)%nz,np.arange(len(ids))]
            proj = (1-(x/beam)**2)**2 + (x*y/beam**2)**2 + (x*(beam+az[i])/beam**2)**2
            return np.abs(value.astype(np.complex128))**2 * proj
        frac = pos-lo
        results[mode] = optical_scale(beam)*((1-frac)*endpoint(lo)+frac*endpoint(lo+1))
    # x,y,qz now correspond to the reported physical FFT-bin detector.
    direct = np.empty(len(ids))
    z = np.arange(nz)[:,None]*d
    for start in range(0,len(ids),2048):
        sl = slice(start,start+2048)
        amp = np.sum(columns[:,sl]*np.exp(-1j*z*qz[sl]),axis=0)
        direct[sl] = abs(amp)**2*(1-(x[sl]/beam)**2)
    results["direct"] = optical_scale(beam)*direct
    return results


def save(path, name, rows):
    path.mkdir(parents=True,exist_ok=True)
    write_rows(path/name,rows)


def run_cpu(case, energies, out, field=None):
    from tests.validation.lib import bragg

    configs = [replace(case,energy_eV=e) for e in energies]
    items = [select_targets(c) for c in configs]
    ids = np.unique(np.concatenate([p.ravel() for _,p,_ in items]))
    if field is None:
        a,b = bragg.lattice_vectors_3d_nm(case.lattice_kind,case.lattice_constant_nm,case.azimuth_deg,
                                          case.tilt_x_deg,case.tilt_y_deg)
        field,_ = bragg._sphere_lattice_vfrac_3d(case.shape,case.phys_size_nm,a,b,
                                                 case.particle_diameter_nm,case.superresolution)
    columns = spectrum_columns(field,ids)
    del field
    for c,(targets,patches,inventory) in zip(configs,items):
        dirpath = out/c.case_id/f"{c.energy_eV:g}"
        ay,ax = [np.fft.fftshift(np.fft.fftfreq(n,c.phys_size_nm))*2*np.pi for n in c.shape[1:]]
        local_ids = np.unique(patches)
        windows = np.searchsorted(local_ids,patches)
        values = sample_columns(columns[:,np.searchsorted(ids,local_ids)],local_ids,c)
        images = {k:v[windows] for k,v in values.items()}
        anchor_target = targets[-1][1]
        yy,xx = np.unravel_index(patches[-1], (len(ay),len(ax)))
        radius = np.hypot(ay[yy]-anchor_target["qy"],ax[xx]-anchor_target["qx"])/(ax[1]-ax[0])
        anchor = float(images["direct"][-1][radius<=3].max())
        save(dirpath,"anchor.csv",[{"direct_100_peak":anchor}])
        save(dirpath,"inventory.csv",inventory)
        peak_rows = measure(targets,patches,ay,ax,images,anchor)
        save(dirpath,"peaks.csv",peak_rows)
        if c.case_id=="hcp_a45" and c.energy_eV==285.:
            index = next(i for i,(_,g) in enumerate(targets)
                         if any(order==(8,5,-6) for order,_,_ in g["orders"]))
            yy,xx = np.unravel_index(patches[index],(len(ay),len(ax)))
            target = targets[index][1]
            core = np.hypot(ay[yy]-target["qy"],ax[xx]-target["qx"])<=3*(ax[1]-ax[0])
            row,_ = np.unravel_index(np.argmax(np.where(core,images["direct"][index],-np.inf)),core.shape)
            line = [dict(qx=ax[xx[row,i]], qy=ay[yy[row,i]], delta_qx=ax[xx[row,i]]-target["qx"],
                         **{mode:float(img[index,row,i]/anchor) for mode,img in images.items()})
                    for i in range(xx.shape[1])]
            save(dirpath,"illustrative_cut.csv",line)


def run_public(case, path_id, out, reference):
    from PyHyperScattering.integrate import NRSSIntegrator
    from tests.path_matrix import get_computation_path
    from tests.validation.lib.bragg import build_bragg_3d_case_morphology, release_runtime_memory

    path = get_computation_path(path_id)
    dest = out/case.case_id/f"{case.energy_eV:g}"
    dest.mkdir(parents=True,exist_ok=True)
    with (reference/case.case_id/f"{case.energy_eV:g}"/"anchor.csv").open() as stream:
        anchor = float(next(csv.DictReader(stream))["direct_100_peak"])
    targets,patches,_ = select_targets(case)
    morph,_ = build_bragg_3d_case_morphology(case,backend=path.backend,backend_options=path.backend_options,
        resident_mode=path.resident_mode,ownership_policy=path.ownership_policy,field_namespace=path.field_namespace)
    try:
        panel = morph.run(stdout=False,stderr=False,return_xarray=True)
        frame = panel.sel(energy=case.energy_eV).transpose("qy","qx")
        y,x = np.asarray(frame.qy), np.asarray(frame.qx)
        expected = [np.fft.fftshift(np.fft.fftfreq(n,case.phys_size_nm))*2*np.pi for n in case.shape[1:]]
        for actual,want in zip((y,x),expected):
            np.testing.assert_allclose(actual,want,rtol=2e-7,atol=1e-8)
        raw = np.asarray(frame.values)
        # Expose original values and NaNs before the maintained test's sanitization.
        np.savez_compressed(dest/"detector.npz",qx=x,qy=y,values=raw,finite=np.isfinite(raw))
        remesh = NRSSIntegrator(use_chunked_processing=False,force_np_backend=True).integrateImageStack(
            panel,phys_size_nm=case.phys_size_nm,shape_zyx=case.shape,energy_ev=case.energy_eV)
        iq = remesh.sel(energy=case.energy_eV).mean("chi")
        np.savez_compressed(dest/"radial.npz",q=np.asarray(iq.coords[iq.dims[0]]),values=np.asarray(iq),
                            semantics=str(remesh.attrs["radial_semantics"]))
        save(dest,"peaks.csv",measure(targets,patches,y,x,{path_id:raw.ravel()[patches]},anchor))
    finally:
        morph.release_runtime()
        release_runtime_memory()


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("cpu","public"))
    p.add_argument("--repo",type=Path,default=Path.cwd())
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--cases",nargs="+",default=["simple_cubic_a30","hcp_a45"])
    p.add_argument("--energies",nargs="+",type=float,default=[285.,570.])
    p.add_argument("--path",choices=["cupy_tensor_coeff","cupy_direct_polarization","legacy_cyrsoxs"])
    p.add_argument("--reference",type=Path,help="CPU output directory containing anchor.csv")
    args = p.parse_args()
    if args.mode=="public" and (args.path is None or args.reference is None):
        p.error("public requires --path and --reference")
    sys.path.insert(0,str(args.repo.resolve()))
    from tests.validation.test_bragg_3d_lattice import CASES_BY_ID
    for case_id in args.cases:
        base = CASES_BY_ID[case_id]
        if args.mode=="cpu":
            run_cpu(base,args.energies,args.out)
        else:
            for e in args.energies:
                run_public(replace(base,energy_eV=e),args.path,args.out,args.reference)
