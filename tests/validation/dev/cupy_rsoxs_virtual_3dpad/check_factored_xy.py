"""Small complex128 check of the split, support-aware XY transform in ZP14."""
import json

import numpy as np


def split_samples(occupied, target, offsets, modes, x_block):
    """Native unshifted full-DFT (z,y,x) modes of a centered real crop."""
    mz, my, mx = target
    z, y, x = (modes % target).T.copy()
    reflected = x > mx // 2
    z[reflected] = -z[reflected] % mz
    y[reflected] = -y[reflected] % my
    x[reflected] = -x[reflected] % mx
    available_x = np.unique(x)
    x_transform = np.fft.rfft(occupied, n=mx, axis=2)
    result = np.empty(len(modes), np.complex128)
    depth_indices = np.arange(occupied.shape[0])
    for start in range(0,len(available_x),x_block):
        keys = available_x[start:start+x_block]
        where = np.flatnonzero(np.isin(x, keys))
        xy = np.fft.fft(x_transform[:,:,keys], n=my, axis=1)
        spectra = xy[:,y[where],np.searchsorted(keys,x[where])]
        z_phase = np.exp(-2j*np.pi*depth_indices[:,None]*z[where]/mz)
        result[where] = np.sum(spectra*z_phase, axis=0)
    oz, oy, ox = offsets
    phase = np.exp(-2j*np.pi*(z*oz/mz + y*oy/my + x*ox/mx))
    result *= phase
    result[reflected] = result[reflected].conj()
    return result


def check():
    rng = np.random.default_rng(4172)
    source = rng.normal(size=(5,7,6))
    cases = 0
    worst = 0.
    for target in ((16,24,32),(13,27,31),(13,9,7),(5,14,12)):
        shape = np.array(target)
        offsets = (shape-source.shape)//2
        crop = tuple(slice(a,a+n) for a,n in zip(offsets,source.shape))
        interior_modes = rng.integers(0, shape, size=(51,3))
        stencil = np.vstack([np.zeros(3,dtype=int),np.eye(3,dtype=int),-np.eye(3,dtype=int)]) % shape
        modes = np.vstack((stencil,shape//2,interior_modes,(-interior_modes)%shape,interior_modes[:4]))
        for windowed in (False, True):
            wy, wx = (np.hanning(n) if windowed else np.ones(n) for n in target[1:])
            for exterior in (0., .63):
                dense = np.full(target, exterior)
                dense[crop] = source
                dense *= wy[None,:,None]*wx[None,None,:]
                ft = np.fft.fftn(dense)
                ref = ft[tuple(modes.T)]
                contrast = (source-exterior)*wy[crop[1]][None,:,None]*wx[crop[2]][None,None,:]
                # The unwindowed Z factor of the logical full box is a discrete delta.
                box = ((modes[:,0] == 0)*shape[0]*np.fft.fft(wy)[modes[:,1]]*
                       np.fft.fft(wx)[modes[:,2]])
                for x_block in (1,4):
                    actual = split_samples(contrast,shape,offsets,modes,x_block) + exterior*box
                    for repair in (False,True):
                        expected = ref.copy()
                        if repair:
                            zero = np.all(modes==0, axis=1)
                            expected[zero] = ref[1:7].real.mean()
                            actual[zero] = actual[1:7].real.mean()
                        relative = np.max(abs(actual-expected))/np.max(abs(expected))
                        if relative > 2e-13:
                            raise AssertionError((target,x_block,windowed,exterior,repair,relative))
                        worst = max(worst,float(relative))
                        cases += 1
    return {"signed_coefficient_comparisons":cases, "max_abs_over_peak":worst,
            "scope":"CPU complex128 against literal full 3D FFT"}


if __name__ == "__main__":
    print(json.dumps(check(),indent=2))
