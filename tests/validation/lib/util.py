import numpy as np
import warnings
import h5py
import datetime
from skimage import transform


def projected_sphere(X, Y, xc, yc, radius):
    '''

    Arguments
    ---------
    X,Y: np.ndarray (M,N)
        2D meshgrids of X and Y positions of real-space grid

    xc,yc: float
        floating point positions of the x and y center of the sphere

    radius: float
        radius of the sphere

    Returns
    -------
    P: np.ndarray(M,N)
        2D image of sphere density

    Example
    --------
    ```python
    NumXY = 2048
    PhysSize=5.0
    Radius=500

    x = y = np.linspace(-PhysSize*NumXY/2,PhysSize*NumXY/2,NumXY)
    X, Y = np.meshgrid(x,y,indexing='ij')
    Mat_1_unaligned = tyler.lib.util.projected_sphere(X,Y,0.0,0.0,Radius)
    Mat_2_unaligned = 1.0 - Mat_1_unaligned
    plt.imshow(Mat_1_unaligned)

    #reshape arrays for CyRSoXS
    Mat_1_unaligned = Mat_1_unaligned[np.newaxis,:,:]
    Mat_2_unaligned = Mat_2_unaligned[np.newaxis,:,:]
    ```


    '''
    Xc2 = np.square(X-xc)
    Yc2 = np.square(Y-yc)
    r2 = radius*radius
    mask = ((Xc2 + Yc2)<(r2))
    P = np.zeros_like(X)
    P[mask] = 2*np.sqrt(r2 - Xc2[mask] - Yc2[mask])
    P /= np.amax(P)
    return P


def center_sphere(NumXY, PhysSize, radius):
    if radius > PhysSize*NumXY/2:
        warnings.warn('Sphere diameter is greater than simulation box width')

    x = y = np.linspace(-PhysSize*NumXY/2, PhysSize*NumXY/2, NumXY)
    X, Y = np.meshgrid(x, y, indexing='ij')
    proj_sphere = projected_sphere(X, Y, 0, 0, radius)

    return proj_sphere


def center_sphere3D(NumXY, NumZ, PhysSize, radius, interpolation=False, scale=2):
    if (radius > PhysSize*NumXY/2) | (radius > PhysSize*NumZ/2):
        warnings.warn('Sphere diameter is greater than a simulation box dimension')
    if interpolation:
        x = y = z = np.linspace(-int(radius/PhysSize), int(radius/PhysSize), int(scale*2*radius/PhysSize))
        Z, Y, X = np.meshgrid(z, y, x,indexing='ij')
        Xc2 = np.square(X)
        Yc2 = np.square(Y)
        Zc2 = np.square(Z)
        r2 = (radius/PhysSize)**2
        mask = ((Xc2 + Yc2 + Zc2) <= (r2))
        P_subset = np.zeros_like(X)
        P_subset[mask] = 1
        P_subset = transform.downscale_local_mean(P_subset, (scale, scale, scale))
        P = np.zeros((NumZ, NumXY, NumXY))
        lowXY = int(P.shape[1]/2 - P_subset.shape[1]/2)
        highXY = int(P.shape[1]/2 + P_subset.shape[1]/2)
        lowZ = int(P.shape[0]/2 - P_subset.shape[0]/2)
        highZ = int(P.shape[0]/2 + P_subset.shape[0]/2)
        P[lowZ:highZ, lowXY:highXY, lowXY:highXY] = P_subset
    else:
        x = y = z = np.linspace(-int(radius/PhysSize), int(radius/PhysSize), int(2*radius/PhysSize))
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        Xc2 = np.square(X)
        Yc2 = np.square(Y)
        Zc2 = np.square(Z)
        r2 = (radius/PhysSize)**2
        mask = ((Xc2 + Yc2 + Zc2) <= (r2))
        P_subset = np.zeros_like(X)
        P_subset[mask] = 1
        P = np.zeros((NumZ, NumXY, NumXY))
        low = int(P.shape[0]/2 - P_subset.shape[0]/2)
        high = int(P.shape[0]/2 + P_subset.shape[0]/2)
        P[low:high, low:high, low:high] = P_subset
    return P


def write_sphere_hdf5(phi1, NumXY, PhysSize, radius, euler=True, author='PJD'):
    s1 = s2 = np.zeros((1, NumXY, NumXY, 3))
    phi1_out = phi1[np.newaxis, :, :]
    phi2_out = 1 - phi1_out

    if euler:
        # placeholder zeros since there is no alignment in this example
        psi1 = psi2 = np.zeros((1, NumXY, NumXY))
        theta1 = theta2 = psi1.copy()
        S1 = S2 = psi1.copy()
    phys_str = str(PhysSize)
    phys_str = phys_str.replace('.', 'p')
    radius_str = str(radius)
    radius_str = radius_str.replace('.', 'p')
    fname = f'SingleSphere-{NumXY}-{phys_str}-{radius_str}.hdf5'
    print(f'--> Marking {fname}')

    array_ordering = 'ZYX'
    with h5py.File(fname, 'w') as f:
        if euler:
            f.create_dataset("Euler_Angles/Mat_1_Psi", data=psi1, compression='gzip', compression_opts=9)
            f.create_dataset("Euler_Angles/Mat_1_Theta", data=theta1, compression='gzip', compression_opts=9)
            f.create_dataset("Euler_Angles/Mat_1_S", data=S1, compression='gzip', compression_opts=9)
            f.create_dataset("Euler_Angles/Mat_1_Vfrac", data=phi1_out, compression='gzip', compression_opts=9)

            f.create_dataset("Euler_Angles/Mat_2_Psi", data=psi2, compression='gzip', compression_opts=9)
            f.create_dataset("Euler_Angles/Mat_2_Theta", data=theta2, compression='gzip', compression_opts=9)
            f.create_dataset("Euler_Angles/Mat_2_S", data=S2, compression='gzip', compression_opts=9)
            f.create_dataset("Euler_Angles/Mat_2_Vfrac", data=phi2_out, compression='gzip', compression_opts=9)
            for key in f['Euler_Angles'].keys():
                for i in range(3):
                    f['Euler_Angles'][key].dims[i].label = array_ordering[i]

        else:
            f.create_dataset("vector_morphology/Mat_1_alignment", data=s1, compression='gzip', compression_opts=9)
            f.create_dataset("vector_morphology/Mat_2_alignment", data=s2, compression='gzip', compression_opts=9)
            f.create_dataset("vector_morphology/Mat_1_unaligned", data=phi1_out, compression='gzip', compression_opts=9)
            f.create_dataset("vector_morphology/Mat_2_unaligned", data=phi2_out, compression='gzip', compression_opts=9)
            for key in f['vector_morphology'].keys():
                for i in range(3):
                    f['vector_morphology'][key].dims[i].label = array_ordering[i]

        f.create_dataset('Morphology_Parameters/NumMaterial', data=2)
        f.create_dataset('Morphology_Parameters/PhysSize', data=PhysSize)
        f.create_dataset('Morphology_Parameters/creation_date', data=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        f.create_dataset('Morphology_Parameters/film_normal', data=[1, 0, 0])
        f.create_dataset('Morphology_Parameters/morphology_creator', data=author)
        f.create_dataset('Morphology_Parameters/name', data=author)
        f.create_dataset('Morphology_Parameters/version', data=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
#             f.create_dataset('Morphology_Parameters/voxel_size_nm', data=PhysSize)

        f.create_dataset('igor_parameters/igorefield', data="0,1")
        f.create_dataset('igor_parameters/igormaterials', data="PEOlig2018,vac")
        f.create_dataset('igor_parameters/igormodelname', data="SingleSphere")
        f.create_dataset('igor_parameters/igormovie', data=0)
        f.create_dataset('igor_parameters/igorname', data="perp001")
        f.create_dataset('igor_parameters/igornum', data=0)
        f.create_dataset('igor_parameters/igorparamstring', data="n/a")
        f.create_dataset('igor_parameters/igorpath', data="n/a")
        f.create_dataset('igor_parameters/igorrotation', data=0)
        f.create_dataset('igor_parameters/igorthickness', data=1)
        f.create_dataset('igor_parameters/igorvoxelsize', data=1)
    return fname


def write_config2D(NumXY, PhysSize,
                   energies, euler=False, referenceFrame=1):
    f = open("config.txt", "w")
    f.write('Energies = ' + str(list(energies)) + ';\n')
    f.write('CaseType=0;  # leave it as it is \n')
    f.write('EAngleRotation=[0.0,1.0,360.0]; # Angle Rotation \n')

    f.write('MorphologyType=' + str(1-int(euler)) + ';  # 0: Euler 1: Vector\n')
    f.write('Algorithm=0;  # Leave it as it is\n')
    f.write("NumThreads = 4;\n")
    f.write("NumX = " + str(NumXY) + ";\n")
    f.write("NumY = " + str(NumXY) + ";\n")
    f.write("NumZ = " + str(1) + ";\n")
    f.write("PhysSize = " + str(PhysSize) + ";\n")
    f.write("ReferenceFrame = " + str(referenceFrame) + ";\n")
    f.close()


def write_slurm(filename, cyrsoxs_version='latest'):

    f = open("job.slurm","w")
    f.write('#!/bin/bash -l\n')
    f.write('#SBATCH --job-name=CYRSOXS      # Job name\n')
    f.write('#SBATCH --output=cyrsoxs.%j.out # Stdout (%j expands to jobId)\n')
    f.write('#SBATCH --error=cyrsoxs.%j.out  # Stderr (%j expands to jobId)\n')
    f.write('#SBATCH --time=04:00:00         # walltime\n')
    f.write('#SBATCH --nodes=1               # Number of nodes requested (only one node on glados)\n')
    f.write('#SBATCH --ntasks=1              # Number of tasks(processes) (tasks distributed across nodes)\n')
    f.write('#SBATCH --ntasks-per-node=1     # Tasks per node\n')
    f.write('#SBATCH --cpus-per-task=1       # Threads per task (all cpus will be on same node)\n')
    f.write('#SBATCH --gres=gpu:turing:1     # number and type of GPUS to use (glados only has turing)\n')
    f.write('#SBATCH --gres-flags=enforce-binding\n')
    f.write('#SBATCH --partition=gpu\n')
    f.write('\n')
    f.write('set -e\n')
    f.write('\n')
    f.write('if [ x$SLURM_CPUS_PER_TASK == x ]; then\n')
    f.write('    export OMP_NUM_THREADS=1\n')
    f.write('else\n')
    f.write('    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n')
    f.write('fi\n')
    f.write('\n')
    f.write('# load necessary modules\n')
    f.write('source ~/.typyEnv CLEAN\n')
    f.write(f'typyEnv --add cyrsoxs/{cyrsoxs_version}\n')
    f.write('\n')
    f.write('## RUN YOUR PROGRAM ##\n')
    f.write('echo "RUNNING ON GPU"${CUDA_VISIBLE_DEVICES}\n')
    f.write(f'srun CyRSoXS {filename}')
    f.close()


def write_sphere3D_hdf5(phi1, NumXYZ, PhysSize, radius, author='PJD'):
    s1 = s2 = np.zeros((NumXYZ, NumXYZ, NumXYZ, 3))
    phi1_out = phi1[np.newaxis, :, :]
    phi2_out = 1 - phi1_out

    fname = f'SingleSphere-{NumXYZ}-{PhysSize}-{radius}.hdf5'
    print(f'--> Marking {fname}')
    with h5py.File(fname, 'w') as f:
        f.create_dataset("igor_parameters/igormaterialnum", data=2.0)
        f.create_dataset("vector_morphology/Mat_1_alignment", data=s1, compression='gzip', compression_opts=9)
        f.create_dataset("vector_morphology/Mat_2_alignment", data=s2, compression='gzip', compression_opts=9)
        f.create_dataset("vector_morphology/Mat_1_unaligned", data=phi1_out, compression='gzip', compression_opts=9)
        f.create_dataset("vector_morphology/Mat_2_unaligned", data=phi2_out, compression='gzip', compression_opts=9)

        f.create_dataset('morphology_variables/creation_date', data=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        f.create_dataset('morphology_variables/film_normal', data=[1, 0, 0])
        f.create_dataset('morphology_variables/morphology_creator', data=author)
        f.create_dataset('morphology_variables/name', data=author)
        f.create_dataset('morphology_variables/version', data=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        f.create_dataset('morphology_variables/voxel_size_nm', data=PhysSize)

        f.create_dataset('igor_parameters/igorefield', data="0,1")
        f.create_dataset('igor_parameters/igormaterials', data="PEOlig2018,vac")
        f.create_dataset('igor_parameters/igormodelname', data="SingleSphere")
        f.create_dataset('igor_parameters/igormovie', data=0)
        f.create_dataset('igor_parameters/igorname', data="perp001")
        f.create_dataset('igor_parameters/igornum', data=0)
        f.create_dataset('igor_parameters/igorparamstring', data="n/a")
        f.create_dataset('igor_parameters/igorpath', data="n/a")
        f.create_dataset('igor_parameters/igorrotation', data=0)
        f.create_dataset('igor_parameters/igorthickness', data=1)
        f.create_dataset('igor_parameters/igorvoxelsize', data=1)
    return fname


def write_config3D(phi1, PhysSize, numXYZ,
                   startEnergy, endEnergy, incrementEnergy,
                   startAngle=0.0, endAngle=360.0, incrementAngle=2.0,
                   numThreads=4):
    f = open("config.txt", "w")
    f.write("StartEnergy = " + str(startEnergy) + ";\n")
    f.write("EndEnergy = " + str(endEnergy) + ";\n")
    f.write("IncrementEnergy = " + str(incrementEnergy) + ";\n")
    f.write("StartAngle = " + str(startAngle) + ";\n")
    f.write("EndAngle = " + str(endAngle) + ";\n")
    f.write("IncrementAngle = " + str(incrementAngle) + ";\n")
    f.write("NumThreads = " + str(numThreads) + ";\n")
    f.write("NumX = " + str(numXYZ) + ";\n")
    f.write("NumY = " + str(numXYZ) + ";\n")
    f.write("NumZ = " + str(numXYZ) + ";\n")
    f.write("PhysSize = " + str(PhysSize) + ";\n")
    f.close()


def write_grating_hdf5(s1, s2, phi1_out, phi2_out, PhysSize, width, author='PJD'):
    NumX = phi1_out.shape[1]
    NumY = phi1_out.shape[2]
    NumZ = phi1_out.shape[0]

    fname = f'SingleGrating-{NumX}-{NumY}-{NumZ}-{PhysSize}-{width}.hdf5'
    print(f'--> Marking {fname}')
    with h5py.File(fname, 'w') as f:
        f.create_dataset("igor_parameters/igormaterialnum", data=2.0)
        f.create_dataset("vector_morphology/Mat_1_alignment", data=s1, compression='gzip', compression_opts=9)
        f.create_dataset("vector_morphology/Mat_2_alignment", data=s2, compression='gzip', compression_opts=9)
        f.create_dataset("vector_morphology/Mat_1_unaligned", data=phi1_out, compression='gzip', compression_opts=9)
        f.create_dataset("vector_morphology/Mat_2_unaligned", data=phi2_out, compression='gzip', compression_opts=9)

        f.create_dataset('morphology_variables/creation_date', data=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        f.create_dataset('morphology_variables/film_normal', data=[1, 0, 0])
        f.create_dataset('morphology_variables/morphology_creator', data=author)
        f.create_dataset('morphology_variables/name', data=author)
        f.create_dataset('morphology_variables/version', data=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        f.create_dataset('morphology_variables/voxel_size_nm', data=PhysSize)

        f.create_dataset('igor_parameters/igorefield', data="0,1")
        f.create_dataset('igor_parameters/igormaterials', data="PEOlig2018,vac")
        f.create_dataset('igor_parameters/igormodelname', data="SingleGrating")
        f.create_dataset('igor_parameters/igormovie', data=0)
        f.create_dataset('igor_parameters/igorname', data="perp001")
        f.create_dataset('igor_parameters/igornum', data=0)
        f.create_dataset('igor_parameters/igorparamstring', data="n/a")
        f.create_dataset('igor_parameters/igorpath', data="n/a")
        f.create_dataset('igor_parameters/igorrotation', data=0)
        f.create_dataset('igor_parameters/igorthickness', data=1)
        f.create_dataset('igor_parameters/igorvoxelsize', data=1)
    return fname
