import numpy as np
import pyFAI,pyFAI.azimuthalIntegrator
import warnings
import xarray as xr
import h5py
from skimage.transform import warp_polar

# global constants for this file
h = 4.135667696e-15 #Planck's constant [eV*s]
c = 299792458 # speed of light [m/s]

def remesh(img):
    remeshed = warp_polar(img,center=(img.shape[0]//2, img.shape[1]//2))
    q = np.sqrt(img.Qy**2+img.Qx**2)
    output_q = np.linspace(0,np.amax(q),remeshed.shape[1])
    output_chi = np.linspace(0.5,359.5,360)
    da_remeshed = xr.DataArray(remeshed,dims=['chi','q'],coords=dict(q=output_q,chi=output_chi))
    return da_remeshed

def read_img_h5(fname,PhysSize=5):
    with h5py.File(fname,'r') as h5:
        img = h5['K0']['projection'][()]
    Qx = 2.0*np.pi*np.fft.fftshift(np.fft.fftfreq(img.shape[1],d=PhysSize))
    Qy = 2.0*np.pi*np.fft.fftshift(np.fft.fftfreq(img.shape[0],d=PhysSize))
    da_img = xr.DataArray(img,dims=['Qy','Qx'],coords=dict(Qx=Qx,Qy=Qy))
    return da_img


def read_config(fname):
    config = {}
    with open(fname) as f:
        for line in f:
            key,value = line.split('=')
            key = key.strip()
            value = value.split(';')[0].strip()
            if key in ['NumX','NumY','NumZ','NumThreads','EwaldsInterpolation','WindowingType']:
                value = int(value)
            elif key in ['RotMask','WriteVTI']:
                value = bool(value)
            else:
                value = float(value)
            config[key] = value
    return config

