from lib.util import center_sphere, write_sphere_hdf5, write_config2D, write_slurm
from lib.generateConstants import *
from lib.reduce import *
import pathlib
import os
import datetime
import subprocess
import shlex
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import xarray as xr
import glob
import sys
import numpy as np


def generate_sphere_test(basePath, return_folderpath=False, slurm_file=0, **slurm_kwargs):
    # default input parameters
    NumXY = 2048
    NumZ = 1
    radius = 50
    PhysSize = 5.0 #nm
    NumMat = 2

    # create projection of 3D sphere
    projection = center_sphere(NumXY, PhysSize, radius)
#     projection = np.reshape(projection,(1,512,512)) #ZYX convention

    # write to file
    filename = write_sphere_hdf5(projection,NumXY, PhysSize, radius)


    # write config and material files
    startEnergy = 270.0
    endEnergy = 310.0
    incrementEnergy = 0.2

    numThreads = 4;        #number of threads for execution
    #Files corresponding to Each material.
    dict={'Material0':'PEOlig2018.txt',
          'Material1':'vacuum'}


    # Label of energy to look for
    labelEnergy={"BetaPara":0,
                 "BetaPerp":1,
                 "DeltaPara":2,
                 "DeltaPerp":3,
                 "Energy":6}

    ens = np.round(np.arange(startEnergy,endEnergy+incrementEnergy,incrementEnergy),2)
    write_config2D(NumXY,PhysSize,ens,euler=True)

    # renamed "main" in generateConstants.py to something more descriptive
    write_materials(startEnergy,endEnergy,incrementEnergy,dict,labelEnergy,len(dict))

    # Create folder stamped with current datetime
    time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = 'spheretest_' + time_now
    os.mkdir(folder_name)
    subprocess.run(["mv","Material1.txt",f"./{folder_name}"])
    subprocess.run(["mv","Material2.txt",f"./{folder_name}"])
    subprocess.run(["mv","config.txt",f"./{folder_name}"])
    subprocess.run(["mv",f"{filename}",f"./{folder_name}"])

    if slurm_file == 1:
        write_slurm(filename, **slurm_kwargs)
        subprocess.run(["mv","job.slurm",f"./{folder_name}"])
        
        
    if return_folderpath:
        return pathlib.Path(basePath,folder_name)
        
def run_slurm(folder_path):
    os.chdir(folder_path)
    subprocess.run(['sbatch','job.slurm'])
    

rho = complex(-0.0001569718509412237,0.0001268706792520359) #284.98651 eV
h          = 4.135667696e-15 #eV*s
c          = 299792458 #m/s
wavelength =  h*c/284.98651
SLD = rho*2*np.pi/(wavelength*1e9)**2

def analytical_sphere(q, radius, NumXY, PhysSize):
    vol = 4/3*np.pi*(radius)**3
    scale = vol/(NumXY/PhysSize)**3
    I = scale/vol*np.abs(3*vol*SLD*(np.sin(q*radius)-q*radius*np.cos(q*radius))/(q*radius)**3)**2
    return I
    
def test_vs_analytical(folder_path,savepng=False):
    h5_path = pathlib.Path(folder_path,'HDF5')
    h5s = sorted(list(h5_path.glob('*h5')))
    energy = np.zeros(len(h5s))
    data = read_img_h5(h5s[0], PhysSize=5)
    remeshed_data = remesh(data)
    energy[0] = h5s[0].name.split('_')[1][:-3]
    all_remeshed = np.zeros((len(h5s),remeshed_data.values.shape))
    all_remeshed[0,...] = remeshed_data.values
    for i in range(1,len(h5s)):
        energy[i] = h5s[i].name.split('_')[1][:-3]
        tmp = read_img_h5(h5s[i],PhysSize=5)
        tmp_remesh = remesh(tmp)
        all_remeshed[i,...] = tmp_remesh.values
    
    all_remeshed = xr.DataArray(all_remeshed,dims=['energy','chi','q'],coords={'energy':energy,'chi':tmp_remesh.chi,'q':tmp_remesh.q})
    
    I_simulation = remeshed_data.mean('chi',)
    I_simulation = I_simulation.where((I_simulation.q < 0.6) & (I_simulation.q > 2e-3),drop=True)
    I_analytical = analytical_sphere(I_simulation.q.values,50,2048,5)
#     avg_scaling = np.nanmean(I_analytical/I_simulation)
    I_simulation = I_simulation*I_analytical[0]/I_simulation[0]
    
    if savepng:

        plt.figure()
        plt.loglog(I_simulation.q,I_analytical,label='Analytical')
        I_simulation.plot(yscale='log',xscale='log',label='Simulation',linestyle='--')
        plt.legend()
        plt.xlabel(r'q [nm$^{-1}$]')
        plt.ylabel('I(q)')
        plt.savefig(pathlib.Path(folder_path,'analytical_v_sim.png'),format='png',dpi=140,bbox_inches='tight')
        plt.close()

        plt.figure()
        remeshed_data.plot(norm=LogNorm(1e-10,1e-3))
        plt.title('Remeshed CyRSoXS Output')
        plt.xlabel(r'q [nm$^{-1}$]')
        plt.ylabel(r'$\chi$ [Degrees]')
        plt.savefig(pathlib.Path(folder_path,'remeshed.png'),format='png',dpi=140,bbox_inches='tight')
        plt.close()

        plt.figure()
        data.plot(norm=LogNorm(1e-10,1e-3))
        plt.title('Raw CyRSoXS output')
        plt.xlabel(r'q$_x$ [nm$^{-1}$]')
        plt.ylabel(r'q$_y$ [nm$^{-1}$]')
        plt.savefig(pathlib.Path(folder_path,'raw.png'),format='png',dpi=140,bbox_inches='tight')
        plt.close()
        
        plt.figure()
        plt.loglog(I_simulation.q, np.abs(I_simulation - I_analytical))
        plt.title(r'(I$_{simulation}$ - I$_{analytical}$)')
        plt.xlabel(r'q [nm$^{-1}$]')        
        plt.ylabel('I(q)')
        plt.savefig(pathlib.Path(folder_path,'difference.png'),format='png',dpi=140,bbox_inches='tight')
        plt.close()
        
        plt.figure()
        all_remeshed.mean('chi').sum('q').plot()
        plt.xlabel('Energy [eV]')
        plt.ylabel('ISI')
        plt.savefig(pathlib.path(folder_path,'ISI.png'),format='png',dpi=140,bbox_inches='tight')
        plt.close()
        
        
    
#     mean_diff = np.abs(np.mean(I_simulation - I_analytical))
    assert(np.allclose(I_simulation,I_analytical,1e-7))
    
    return I_simulation, I_analytical
    
if __name__ == '__main__':
    version = 'latest'
    slurm_file = 0
    
    for arg in sys.argv:
        if arg.startswith('slurm'):
            slurm_file = int(arg.split('=')[-1])
        if arg.startswith('version'):
            version = arg.split('=')[-1]

    basePath = pathlib.Path('.').absolute()
    
    workpath = generate_sphere_test(basePath,return_folderpath=True, slurm_file=slurm_file,cyrsoxs_version=version)

    if slurm_file == 1:
        import time
        run_slurm(workpath)
        hdf5path = pathlib.Path(workpath,'HDF5')
        while len(list(hdf5path.glob('*.h5'))) < 201:
            time.sleep(10)
#         test_vs_analytical(workpath,savepng=True)
        I_simulation, I_analytical = test_vs_analytical(workpath, savepng=True)