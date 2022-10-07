from lib.util import center_sphere, write_sphere_hdf5, write_config2D, write_slurm
from lib.generateConstants import *
from lib.reduce import *
import pathlib
import subprocess
import cupy as cp
import os
import h5py
import datetime
from numpy import genfromtxt
import shutil
from shutil import copyfile
import sys
import xarray as xr
import time
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':14})

#Generates Euler morphology from scratch
def coreshell_spheres_decay_euler(basePath, r=512,d=32,vx_size=2.5,ra=4,t=2.94,phi_iso=0.46,decay_order=0.42,numMaterial=3,
                                  filename="coreshelltest.hdf5",mformat='ZYX', 
                                  return_folderpath=False, hdfcompress=True, slurm_file=0, cyrsoxs_version='latest'):
    
    # a b c are boolean masks for material presence
    a_b = cp.full((d, r, r), False)  # core
    b_b = cp.full((d, r, r), False)  # shell
    c_b = cp.full((d, r, r), False)  # everything else

    #orientation components of shell
    b_x = cp.zeros([d, r, r])  # shell
    b_y = cp.zeros([d, r, r])  # shell
    b_z = cp.zeros([d, r, r])  # shell

    #volume fractions of each material - euler
    vf_a = cp.zeros([d, r, r])  # core
    vf_b = cp.zeros([d, r, r])  # shell
    vf_c = cp.zeros([d, r, r])  # everything else

    #euler angles and aligned fraction - euler morphology - ZXZ rotations
    SE_b = cp.zeros([d, r, r])
    theta_b = cp.zeros([d, r, r]) #theta of shell
    psi_b = cp.zeros([d, r, r]) #psi of shell

    #intermediate orientation components of shell
    b_xi = cp.zeros([d, r, r])  # shell
    b_yi = cp.zeros([d, r, r])  # shell
    b_zi = cp.zeros([d, r, r])  # shell

    #read list of coordinates
    l = genfromtxt('./lib/CoreShell/LoG_coord.csv', delimiter=',', skip_header=1)

    z,y,x = cp.ogrid[0:d:1,0:r:1, 0:r:1]

    for p in l:
        #print(str(p[0]) + " " + str(p[1]))
        # these are the particle coordinate locations
        #this mask defines each sphere
        #mf is a volume matrix (same shape as morphology) of the squared radius out from center
        mf = (x-p[0])**2 + (y-p[1])**2 +(z-15)**2 
        #create a boolean matrix (same shape as morphology) that is true if mf is less than squared radius
        mask = mf <= ra**2
        #add that sphere to material a boolean mask containing all the material a (core) spheres
        a_b = cp.logical_or(a_b, mask)
        #calculate the shell using the same mf matrix    
        mask = (a_b == False) & (mf <= (ra+t)**2)
        #add that shell to material b boolean mask containing all the material b shells
        b_b = cp.logical_or(b_b, mask)

        #these 3 statements calculate orientation vector pieces
        b_xi = ((x-p[0]) + y*0 + z*0)*mask
        b_yi = (x*0 + (y-p[1]) + z*0)*mask
        b_zi = (x*0 + y*0 + (z-15))*mask

        b_x = b_x + b_xi*(b_x==0)
        b_y = b_y + b_yi*(b_y==0)
        b_z = b_z + b_zi*(b_z==0)

    #ensure shell where there is no core and where there is shell
    b_b = (a_b == False) & (b_b == True)
    #orientation in shell only where there is shell
    b_x = b_x*b_b
    b_y = b_y*b_b
    b_z = b_z*b_b

    #matrix - 3rd component is everywhere that core and shell are not
    c_b = (a_b == False) & (b_b == False)

    #calculate the magnitude of the orientation vector in shell for later normalization
    b_t = (b_x**2 + b_y**2+b_z**2)**0.5

    #make each of the components of orientation normalized so that their squared sum is equal to aligned volume fraction
    #this part needs to be adjusted for decay; in principle we already have all the pieces... instead of dividing by b_t (hypotenuse length) we can add a ratio of (1st_shell/b_t)^decay_order, which will always be <=1  
    #what should 1st shell be? It needs to be a matrix of value ra
    b_1 = (ra)*b_b

    #total and aligned volume fractions
    vf_b = 1*b_b  #total volume fraction of shell = 1
    SE_b = b_b*(1-phi_iso)*(b_1/b_t)**(decay_order)  #aligned fraction of shell
    vf_a = 1*a_b  # total volume fraction of core = 1
    vf_c = 1*c_b  # total volume fraction of everything else = 1

    # convert the handful of NaN's arising from divide by zero to zeros
    SE_b = cp.nan_to_num(SE_b)

    # Euler angle calculations
    theta_b = cp.arctan2(((b_x)**2+(b_y)**2)**0.5, b_z)  #theta of shell - calculated from vector morphology
    psi_b = cp.arctan2(b_y, b_x)

    with h5py.File(filename, "w") as f:
        f.create_dataset('igor_parameters/igorefield', data="0,1")
        f.create_dataset('igor_parameters/igormaterialnum', data=numMaterial)
        f.create_dataset('igor_parameters/igornum', data=r)
        f.create_dataset('igor_parameters/igorthickness', data=d)
        f.create_dataset('igor_parameters/igorvoxelsize', data=vx_size)

        f.create_dataset('Morphology_Parameters/creation_date', data=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")) # datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        f.create_dataset('Morphology_Parameters/film_normal', data=[1,0,0])
        f.create_dataset('Morphology_Parameters/morphology_creator', data="")
        f.create_dataset('Morphology_Parameters/name', data=filename)
        f.create_dataset('Morphology_Parameters/version', data="")#data=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        f.create_dataset('Morphology_Parameters/PhysSize', data=vx_size)
        f.create_dataset('Morphology_Parameters/NumMaterial',data=numMaterial)
        f.create_dataset('Morphology_Parameters/Parameters',data="r,d,ra,t,phi_iso,decay_order")
        f.create_dataset('Morphology_Parameters/Parameter_values',data=[r,d,ra,t,phi_iso,decay_order])


        mat1vf = f.create_dataset('Euler_Angles/Mat_1_Vfrac', data=cp.asnumpy(vf_a.astype(cp.float64)), compression="gzip", compression_opts=9)
        mat1s = f.create_dataset('Euler_Angles/Mat_1_S', data=np.zeros([d, r, r]), compression="gzip", compression_opts=9)
        mat1theta = f.create_dataset('Euler_Angles/Mat_1_Theta', data=np.zeros([d, r, r]), compression="gzip", compression_opts=9)
        mat1psi = f.create_dataset('Euler_Angles/Mat_1_Psi', data=np.zeros([d, r, r]), compression="gzip", compression_opts=9)
        mat2vf = f.create_dataset('Euler_Angles/Mat_2_Vfrac', data=cp.asnumpy(vf_b.astype(cp.float64)), compression="gzip", compression_opts=9)
        mat2s = f.create_dataset('Euler_Angles/Mat_2_S', data=cp.asnumpy(SE_b), compression="gzip", compression_opts=9)
        mat2theta = f.create_dataset('Euler_Angles/Mat_2_Theta', data=cp.asnumpy(theta_b), compression="gzip", compression_opts=9)
        mat2psi = f.create_dataset('Euler_Angles/Mat_2_Psi', data=cp.asnumpy(psi_b), compression="gzip", compression_opts=9)      
        mat3vf = f.create_dataset('Euler_Angles/Mat_3_Vfrac', data=cp.asnumpy(vf_c.astype(cp.float64)), compression="gzip", compression_opts=9)
        mat3s = f.create_dataset('Euler_Angles/Mat_3_S', data=np.zeros([d, r, r]), compression="gzip", compression_opts=9)
        mat3theta = f.create_dataset('Euler_Angles/Mat_3_Theta', data=np.zeros([d, r, r]), compression="gzip", compression_opts=9)     
        mat3psi = f.create_dataset('Euler_Angles/Mat_3_Psi', data=np.zeros([d, r, r]), compression="gzip", compression_opts=9)

        for i in range(1,numMaterial+1):
            for j in range(0,3):
                locals()[f"mat{i}vf"].dims[j].label = mformat[j]
                locals()[f"mat{i}s"].dims[j].label = mformat[j]
                locals()[f"mat{i}theta"].dims[j].label = mformat[j]
                locals()[f"mat{i}psi"].dims[j].label = mformat[j]
    f.close()
    
    # Create folder stamped with current datetime
    time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = 'CoreShelltest_' + time_now
    os.mkdir(folder_name)
    subprocess.run(["cp","./lib/CoreShell/Material1.txt",f"./{folder_name}"])
    subprocess.run(["cp","./lib/CoreShell/Material2.txt",f"./{folder_name}"])
    subprocess.run(["cp","./lib/CoreShell/Material3.txt",f"./{folder_name}"])
    subprocess.run(["cp","./lib/CoreShell/config.txt",f"./{folder_name}"])
    subprocess.run(["mv",f"{filename}",f"./{folder_name}"])

    if slurm_file == 1:
        write_slurm(filename, cyrsoxs_version=cyrsoxs_version)
        subprocess.run(["mv","job.slurm",f"./{folder_name}"])
        
    
    if return_folderpath:
        return pathlib.Path(basePath,folder_name)

    
    
def run_slurm(folder_path):
    os.chdir(folder_path)
    result = subprocess.run(['sbatch','job.slurm'])
    return result


def test_vs_reference(folder_path,cyrsoxs_version='latest', savepng=False):
    h5_path = pathlib.Path(folder_path,'HDF5')
    h5list = sorted(list(h5_path.glob('*h5')))
    while len(h5list) < 101:
        time.sleep(0.5)
        h5list = sorted(list(h5_path.glob('*h5')))
        
    horz = np.zeros((len(h5list),363))
    vert = horz.copy()
    for i, file in enumerate(h5list):
        data = read_img_h5(file,PhysSize=2.5)
        remeshed = remesh(data)
        horz[i,:] = remeshed.sel(chi=slice(170,190)).mean('chi')
        vert[i,:] = remeshed.sel(chi=slice(80,100)).mean('chi')
        
    with np.errstate(invalid='ignore'):
        A_np = -(vert-horz)/(vert+horz)
    
    A = xr.DataArray(A_np, dims=['energy','q'],coords={'q':remeshed.q,'energy':np.round(np.arange(280.0,290.01,0.1),1)}).sortby('energy')
    
    A_reference = xr.open_dataarray('/home/pjd1/CyRSoXS-Processing/pete/test_directory/lib/CoreShell/CS_reference.nc')
    A_reference = A_reference.interp(energy=np.round(np.arange(280.0,290.01,0.1),1)).sortby('energy')
    if savepng:
        plt.figure()
#         (A_reference.sel(q=slice(0.02,0.4)).mean('q') - A.sel(q=slice(0.02,0.4)).mean('q')).plot(color='k',label='Reference',xscale='log')
        A_reference.sel(q=slice(0.02,0.4)).mean('q').plot(color='k',label='Reference',xscale='log')
        A.sel(q=slice(0.02,0.4)).mean('q').plot(linestyle='--',label='CyRSoXS',xscale='log')
        plt.legend()
        plt.xlabel('Energy [eV]')
        plt.ylabel('A(E)')
        plt.title('')
        plt.xticks(ticks=[280,282,284,286,288,290],labels=[280,282,284,286,288,290])
        plt.savefig(pathlib.Path(folder_path,'AvE.png'),format='png',dpi=140,bbox_inches='tight')
        plt.close()
        
        plt.figure()
#         (A_reference.sel(energy=284.7,q=slice(0.02,0.4)) - A.sel(energy=284.7,q=slice(0.02,0.4))).plot(xscale='log',color='k',label=f'284.7 eV Reference')
#         (A_reference.sel(energy=285.2,q=slice(0.02,0.4)) -  A.sel(energy=285.2,q=slice(0.02,0.4))).plot(xscale='log',color='r',label=f'285.2 eV Reference')
        A_reference.sel(energy=284.7,q=slice(0.02,0.4)).plot(xscale='log',color='k',label=f'284.7 eV Reference')
        A_reference.sel(energy=285.2,q=slice(0.02,0.4)).plot(xscale='log',color='r',label=f'285.2 eV Reference')
        A.sel(energy=284.7,q=slice(0.02,0.4)).plot(xscale='log',linestyle='--',label=f'284.7 eV CyRSoXS')
        A.sel(energy=285.2,q=slice(0.02,0.4)).plot(xscale='log',linestyle='--',label=f'285.2 eV CyRSoXS')
        plt.xlim(0.02,0.41)
        plt.ylim(-0.3,0.3)
        plt.legend()
        plt.xlabel(r'q [nm$^{-1}$]')
        plt.ylabel('A(q)')
        plt.title('')
        plt.savefig(pathlib.Path(folder_path,'AvQ.png'),format='png',dpi=140,bbox_inches='tight')
        plt.close()

    
    assert(np.allclose(A_reference.sel(q=slice(0.02,0.4)).mean('q').values,A.sel(q=slice(0.02,0.4)).mean('q').values,atol=2e-6,equal_nan=True))
    assert(np.allclose(A_reference.sel(energy=284.7,q=slice(0.02,0.4)).values,A.sel(energy=284.7,q=slice(0.02,0.4)).values,atol=1e-4,equal_nan=True))
    assert(np.allclose(A_reference.sel(energy=285.2,q=slice(0.02,0.4)).values,A.sel(energy=285.2,q=slice(0.02,0.4)).values,atol=1e-4,equal_nan=True))
    
if __name__ == '__main__':
    version = 'latest'
    slurm_file = 0
    
    for arg in sys.argv:
        if arg.startswith('slurm'):
            slurm_file = int(arg.split('=')[-1])
        if arg.startswith('version'):
            version = arg.split('=')[-1]

    basePath = pathlib.Path('.').absolute()
    
    workpath = coreshell_spheres_decay_euler(basePath, filename="coreshelltest.hdf5",mformat='ZYX',return_folderpath=True, hdfcompress=True, slurm_file=slurm_file, cyrsoxs_version=version)

    if slurm_file == 1:
        import time
        result = run_slurm(workpath)
        if result.returncode == 0:
            test_vs_reference(workpath,cyrsoxs_version=version, savepng=True)