import numpy as np
import h5py
import datetime
import warnings
import pathlib


def write_hdf5(material_list, PhysSize, fname, MorphologyType=0, ordering='ZYX', author='NIST'):
    '''
    Writes Euler or Vector Morphology format into CyRSoXS-compatible HDF5 file and returns the hdf5 filename

    Parameters
    ----------
        material_list : lists
            List of material lists. 
            Euler Ex. [[Mat_1_Vfrac, Mat_1_S, Mat_1_Theta, Mat_1_Psi],[Mat_2_Vfrac, Mat_2_S, Mat_2_Theta, Mat_2_Psi]]
            Vector Ex. [[Mat_1_alignment, Mat_1_unaligned],[Mat_2_alignment, Mat_2_unaligned]]
        PhysSize : float
            Voxel size
        fname : str or path
            name of hdf5 file to write
        MorphologyType : int
            0 - Euler
            1 - Vector
        ordering : str
            String denoting the axes ordering. 'ZYX' or 'XYZ'
        author : str
            Name of author writing the morphology
    
    Returns
    -------
        fname
            Name of hdf5 file written
    '''

    print(f'--> Marking {fname}')
    with h5py.File(fname, 'w') as f:
        num_mat = len(material_list)
        if MorphologyType == 0:
            i = 1
            for material in material_list:
                f.create_dataset(f"Euler_Angles/Mat_{i}_Vfrac", data=material[0], compression="gzip", compression_opts=9)
                f.create_dataset(f"Euler_Angles/Mat_{i}_S", data=material[1], compression="gzip", compression_opts=9)
                f.create_dataset(f"Euler_Angles/Mat_{i}_Theta", data=material[2], compression="gzip", compression_opts=9)
                f.create_dataset(f"Euler_Angles/Mat_{i}_Psi", data=material[3], compression="gzip", compression_opts=9)
                for j in range(3):
                    f[f"Euler_Angles/Mat_{i}_Vfrac"].dims[j].label = ordering[j]
                    f[f"Euler_Angles/Mat_{i}_S"].dims[j].label = ordering[j]
                    f[f"Euler_Angles/Mat_{i}_Theta"].dims[j].label = ordering[j]
                    f[f"Euler_Angles/Mat_{i}_Psi"].dims[j].label = ordering[j]
                i += 1
        
        elif MorphologyType == 1:
            i = 1
            for material in material_list:
                f.create_dataset(f"Vector_Morphology/Mat_{i}_alignment", data=material[0], compression="gzip", compression_opts=9)
                f.create_dataset(f"Vector_Morphology/Mat_{i}_unalignment", data=material[1], compression="gzip", compression_opts=9)
                for j in range(3):
                    f[f"Vector_Morphology/Mat_{i}_Vfrac"].dims[j].label = ordering[j]
                    f[f"Vector_Morphology/Mat_{i}_S"].dims[j].label = ordering[j]
                    f[f"Vector_Morphology/Mat_{i}_Theta"].dims[j].label = ordering[j]
                    f[f"Vector_Morphology/Mat_{i}_Psi"].dims[j].label = ordering[j]
                i += 1
        else:
            warnings.warn('Check MorphologyType. Should be 0 (Euler) or 1 (Vector)', stacklevel=2)

        f.create_dataset('Morphology_Parameters/creation_date', data=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        f.create_dataset('Morphology_Parameters/morphology_creator', data=author)
        f.create_dataset('Morphology_Parameters/PhysSize', data=PhysSize)
        f.create_dataset('Morphology_Parameters/NumMaterial', data=num_mat)

    return fname


def write_config(Energies, EAngleRotation, CaseType=0, MorphologyType=0,
                 NumThreads=4, AlgorithmType=0, DumpMorphology=False,
                 ScatterApproach=0, WindowingType=0, RotMask=False,
                 EwaldsInterpolation=1):
    """
    Writes config.txt file for CyRSoXS simulations

    Parameters
    ----------
        energies : list
            List of energies to be simulated
        EAngleRotation : list
            List of angle rotations in degrees. Format is [AngleStart, AngleIncrement, AngleStop]
        CaseType : int
            Scattering configuration. 
            0 - Default
            1 - Beam Divergence
            2 - Grazing Incidence
        MorphologyType : int
            0 - Euler Morphology (default)
            1 - Vector Morphology
        NumThreads : int
            Number of OpenMP threads to use. Must be >= number of GPUs
        AlgorithmType : int
            0 - Communication minimizing (default)
            1 - Memory minimizing
        DumpMorphology : bool
            Boolean flag to write morphology to file, as seen by CyRSoXS, after necessary conversions are performed
        ScatterApproach : int
            Flag to explicitly calculate the differential scattering cross-section before the Ewald Sphere projection
            0 - Do not compute
            1 - Compute
        WindowingType : int
            Type of FFT window to apply
            0 - None
            1 - Hanning
        RotMask : bool
            Boolean flag to include values outside valid range for all angles
            False - Writes NaNs for pixels that are not valid at every EAngleRotation
            True - Replaces NaNs with values averaged over valid EAngleRotations
        EwaldInterpolation : int
            Type of interpolation for Ewald Sphere projection
            0 - Nearest Neighbor
            1 - Trilinear

    Returns
    -------
        None
    """
    f = open('config.txt', "w")

    # required
    f.write("CaseType = " + str(CaseType) + ";\n")
    f.write("Energies = " + str(Energies) + ";\n")
    f.write("EAngleRotation = " + str(EAngleRotation) + ";\n")
    f.write("MorphologyType = " + str(MorphologyType) + ";\n")

    # optional, but written by default
    f.write("AlgorithmType = " + str(AlgorithmType) + ";\n")
    f.write("NumThreads = " + str(NumThreads) + ";\n")
    f.write("DumpMorphology = " + str(DumpMorphology) + ";\n")
    f.write("ScatterApproach = " + str(ScatterApproach) + ";\n")
    f.write("WindowingType = " + str(WindowingType) + ";\n")
    f.write("RotMask = " + str(RotMask) + ";\n")
    f.write("EwaldsInterpolation = " + str(EwaldsInterpolation) + ";\n")        
    f.close()


# NIST specific
def write_slurm(hdf5_filename, cyrsoxs_version='latest'):
    """
    Writes job.slurm file for use on the NIST glados and wheatley workstations

    Parameters
    ----------
        filename : str or pathlib object
            Name of morphology HDF5 file to be simulated. Can include path
        cyrsoxs_version : str
            String denoting which version of CyRSoXS to run

    Returns
    -------
        None
    """

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
    f.write(f'srun CyRSoXS {hdf5_filename}')
    f.close()


# Set of functions originally authored by Kumar Saurabh @maksbh 
def find_nearest(array, value):
    """
    Function to find the nearest index 

    Parameters
    ----------

    array : ndarray
    value : value of energy

    Returns
    -------
    idx : Integer
        index location corresponding to the closest location
    """
    idx = (np.abs(array - value)).argmin()
    return idx


def get_interpolated_value(array, value, nearest_id, energy_id):
    """
    Function to get the interpolated value

    Parameters
    ----------

    array : ndarray
    value : value of energy
    nearest_id : id corresponding to the nearest value

    Returns
    -------
    valArray : ndarray
            array of the interpolated values
    """
    valArray = np.zeros(array.shape[1])
    if (array[nearest_id][energy_id] > value):
        xp = [array[nearest_id - 1][energy_id], array[nearest_id][energy_id]]
        for i in range(0, array.shape[1]):
            yp = [array[nearest_id - 1][i], array[nearest_id][i]]
            valArray[i] = np.interp(value, xp, yp)

    elif (array[nearest_id][energy_id] < value):
        xp = [array[nearest_id][energy_id], array[nearest_id + 1][energy_id]]
        for i in range(0, array.shape[1]):
            yp = [array[nearest_id][i], array[nearest_id + 1][i]]
            valArray[i] = np.interp(value, xp, yp)

    else:
        for i in range(0, len(valArray)):
            valArray[i] = array[nearest_id][i]

    return valArray


def removeDuplicates(Data, energy_id):
    """
    Function to remove duplicate energies

    Parameters
    ----------

    Data : ndarray
    energy_id : int

    Returns
    -------

    listOut : ndarray

    """
    listIn = Data.tolist()
    listOut = []
    listOut.append(listIn[0])
    currEnergy = listIn[0][energy_id]
    duplicateFound = False
    for i in range(1, len(listIn)):
        if (listIn[i][energy_id] == currEnergy):
            duplicateFound = True
            continue
        else:
            listOut.append(listIn[i])
            currEnergy = listIn[i][energy_id]

    if (duplicateFound):
        print('Duplicates in Energy found. Removing it')
    return (np.array(listOut))


def dump_dataVacuum(index, energy, f):
    """
    Function to write vacuum optical contants (n = 1 + i0)

    Parameters
    ----------

    index : int
    energy : float

    """
    Header = "EnergyData" + str(index) + ":\n{\n"
    f.write(Header)
    Energy = "Energy = " + str(energy) + ";\n"
    f.write(Energy)
    BetaPara = "BetaPara = " + str(0.0) + ";\n"
    f.write(BetaPara)
    BetaPerp = "BetaPerp = " + str(0.0) + ";\n"
    f.write(BetaPerp)
    DeltaPara = "DeltaPara = " + str(0.0) + ";\n"
    f.write(DeltaPara)
    DeltaPerp = "DeltaPerp = " + str(0.0) + ";\n"
    f.write(DeltaPerp)
    f.write("}\n")


def dump_data(valArray, index, labelEnergy, f):
    """
    Function to write material optical constants to file

    Parameters
    ----------

    valArray : ndarray
    index : int
    labelEnergy : dict
    f : file object


    """
    Header = "EnergyData" + str(index) + ":\n{\n";
    f.write(Header)
    Energy = "Energy = " + str(valArray[labelEnergy["Energy"]]) + ";\n"
    f.write(Energy)
    BetaPara = "BetaPara = " + str(valArray[labelEnergy["BetaPara"]]) + ";\n"
    f.write(BetaPara)
    BetaPerp = "BetaPerp = " + str(valArray[labelEnergy["BetaPerp"]]) + ";\n"
    f.write(BetaPerp)
    DeltaPara = "DeltaPara = " + str(valArray[labelEnergy["DeltaPara"]]) + ";\n"
    f.write(DeltaPara)
    DeltaPerp = "DeltaPerp = " + str(valArray[labelEnergy["DeltaPerp"]]) + ";\n"
    f.write(DeltaPerp)
    f.write("}\n")


def writeList(name: str, value: list, file):
    """
    Function to write list to file

    Parameters
    ----------

    name : str
    value : list
    file : file object

    """
    valStr: str = name + "["
    for i in range(len(value) - 1):
        valStr = valStr + str(value[i]) + ","
    valStr = valStr + str(value[len(value) - 1])
    file.write(valStr + "];\n")


def write_materials(energies, materialdict, labelEnergy, numMaterial):
    """
    Function to write optical constants for all energies supplied

    Parameters
    ----------

    energies : ndarray
    materialdict : dict
    labelEnergy : dict
    numMaterial : int


    """
    NumEnergy = len(energies)

    for numMat in range(1, numMaterial+1):
        f = open("Material" + str(numMat) + ".txt", "w")
        fname = materialdict["Material" + str(numMat)]
        if (fname != 'vacuum'):
            Data = np.loadtxt(fname, skiprows=1)
            Data = Data[Data[:, labelEnergy["Energy"]].argsort()]
            Data = removeDuplicates(Data, labelEnergy["Energy"])
            for i in range(0, NumEnergy):
                currentEnergy = energies[i]
                nearest_id = find_nearest(Data[:, labelEnergy["Energy"]], currentEnergy)
                ValArray = get_interpolated_value(Data, currentEnergy, nearest_id, labelEnergy["Energy"])
                dump_data(ValArray, i, labelEnergy, f)

        else:
            for i in range(0, NumEnergy):
                currentEnergy = energies[i]
                dump_dataVacuum(i, currentEnergy, f)
        f.close()


def write_opts(optical_constants, material_num, path):
    # pybind ordering is deltaPara, betaPara, deltaPerp, betaPerp
    # CLI ordering is betaPara, betaPerp, deltaPara, deltaPerp
    if path:
        fname = pathlib.Path(path, "Material" + str(material_num) + ".txt")
    else:
        fname = "Material" + str(material_num) + ".txt"
    f = open(fname, "w")
    for i, key in enumerate(optical_constants):
        Header = "EnergyData" + str(i) + ":\n{\n"
        f.write(Header)
        Energy = "Energy = " + str(key) + ";\n"
        f.write(Energy)
        BetaPara = "BetaPara = " + str(optical_constants[key][1]) + ";\n"
        f.write(BetaPara)
        BetaPerp = "BetaPerp = " + str(optical_constants[key][3]) + ";\n"
        f.write(BetaPerp)
        DeltaPara = "DeltaPara = " + str(optical_constants[key][0]) + ";\n"
        f.write(DeltaPara)
        DeltaPerp = "DeltaPerp = " + str(optical_constants[key][2]) + ";\n"
        f.write(DeltaPerp)
        f.write("}\n")
    f.close()
