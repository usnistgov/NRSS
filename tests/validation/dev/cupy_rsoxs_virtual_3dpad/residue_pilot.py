"""CPU construction of a complete padded detector from a compact sphere. Development reference."""
import argparse
import json
import resource
from pathlib import Path
import time

import numpy as np
from scipy import fft

from sphere_reference import (CROP, D, K, SIZES, axis, compare_panel, contrast,
                              discrete_anchors, forward_scale, sphere_fraction)


def full_detector(volume, size, fill_id):
    """One XY residue group at a time. The Z sum uses only the two detector endpoints."""
    n = volume.shape[0]
    factor, remainder = divmod(size, n)
    if remainder or volume.shape != (n,n,n):
        raise ValueError("This pilot uses an integer cubic ratio. See ZP07 for general shapes.")
    phase = np.exp(-2j*np.pi*np.arange(size)/size).astype(np.complex64)
    plane_indices = np.arange(n)
    offset = (size-n)//2
    modes = np.arange(size)
    lateral_shift = phase[(modes*offset) % size]
    axial_phase = phase[(modes[:,None]*(plane_indices+offset)) % size]
    ramp = phase[(np.arange(factor)[:,None]*plane_indices) % size]
    signed = np.arange(size)-size//2
    native = (-signed) % size
    qaxis = axis(size)
    dq = 2*np.pi/(size*D)
    scale = abs(contrast(fill_id)/(4*np.pi))**2
    panel = np.full((size,size),np.nan,dtype=np.float32)
    dc = 0.
    for dim in range(3):
        other = tuple(j for j in range(3) if j!=dim)
        projected = volume.sum(axis=other,dtype=np.float64)
        dc += projected @ np.cos(2*np.pi*(plane_indices+offset)/size) / 3
    for ry in range(factor):
        yids = np.flatnonzero(native%factor == ry)
        for rx in range(factor):
            xids = np.flatnonzero(native%factor == rx)
            rsq = (qaxis[yids,None]**2 + qaxis[None,xids]**2).astype(np.float32)
            valid = (rsq<=np.float32(K*K)) & (yids[:,None]<size-1) & (xids[None,:]<size-1)
            iy,ix = np.nonzero(valid)
            if not len(iy):
                continue
            y,x = yids[iy],xids[ix]
            my,mx = native[y],native[x]
            root = np.sqrt(np.maximum(np.float32(K*K)-rsq[iy,ix],0.))
            pos = -rsq[iy,ix]/(np.float32(K)+root)/np.float32(dq)
            low = np.floor(pos).astype(int)
            frac = pos-low
            field = volume * (ramp[ry,:,None]*ramp[rx,None,:])
            transformed = fft.fft2(field,axes=(1,2),overwrite_x=True,workers=1)
            columns = transformed[:,my//factor,mx//factor].T
            lateral = lateral_shift[my] * lateral_shift[mx]
            power = np.zeros(len(x))
            for zbin, weight in ((low,1-frac),(low+1,frac)):
                amp = lateral*np.einsum("pz,pz->p",columns,axial_phase[(-zbin)%size])
                amp[(my==0)&(mx==0)&(zbin==0)] = dc
                qx,qy = qaxis[x],qaxis[y]
                dipole = (K*K-qx*qx)**2+(qx*qy)**2+(qx*(K+dq*zbin))**2
                power += weight*scale*abs(amp)**2*dipole
            panel[y,x] = power
    return panel


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size",choices=SIZES,type=int,default=256)
    parser.add_argument("--fill",choices=("vacuum","2"),default="vacuum")
    parser.add_argument("--evidence-dir", type=Path, help="Write scalar metrics and the positive qy line here")
    args = parser.parse_args()
    fill = 2 if args.fill=="2" else None
    fraction = sphere_fraction()
    start = time.perf_counter()
    panel = full_detector(fraction,args.size,fill)
    elapsed = time.perf_counter()-start
    compute_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    stats = compare_panel(panel,args.size,fill)
    idx, ref = discrete_anchors(fraction,args.size,fill)
    err = float(np.max(abs(panel[tuple(idx.T)]-ref))/forward_scale(fill))
    if err > 2e-5:
        raise ValueError(f"Voxel anchor mismatch: {err}")
    report = {"kind":"standalone CPU, one scalar field, one trial", "M":args.size,
                      "fill":fill, "crop":CROP, "calculation_s":elapsed,
                      "peak_process_MiB_through_calculation":compute_peak,
                      "peak_process_MiB_including_scoring":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                      "anchor_max_abs_over_I0":err, **stats}
    rendered = json.dumps(report,indent=2,allow_nan=False)
    if args.evidence_dir:
        args.evidence_dir.mkdir(parents=True,exist_ok=True)
        (args.evidence_dir/"cpu.json").write_text(rendered+"\n")
        np.save(args.evidence_dir/"positive_qy.npy",panel[args.size//2:,args.size//2])
    print(rendered)
