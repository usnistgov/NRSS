import h5py
# import pathlib
import CyRSoXS as cy
import warnings
from .checkH5 import check_NumMat
from .reader import read_material, read_config
from .writer import write_opts, write_hdf5
from .visualizer import morphology_visualizer

import numpy as np
import xarray as xr
import sys
import os
import copy

from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
T = TypeVar("T")

def wraps(wrapper: Callable[P, T]) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    A decorator to preserve the original function's docstring.

    Args:
        wrapper: The wrapper function.

    Returns:
        A decorator that preserves the original function's docstring.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        """
        A decorator to preserve the original function's docstring.

        Args:
            func: The original function.

        Returns:
            The original function with its original docstring.
        """
        func.__doc__ = wrapper.__doc__
        return func

    return decorator

class Morphology:
    '''
    Object used to hold all the components necessary for a complete CyRSoXS morphology

    Attributes
    ----------
    input_mapping : dict
        A dictionary to handle specific configuration input types

    numMaterial : int
        Number of materials present in the morphology

    materials : dict
        A dictionary of Material objects

    PhysSize : float
        The physical size of each cubic voxel's three dimensions

    NumZYX : tuple or list
        Number of voxels in the Z, Y, and X directions (NumZ, NumY, NumX)

    config : dict
        A dictionary of configuration parameters for CyRSoXS

    create_cy_object : bool
        Boolean value that decides if the CyRSoXS objects are created upon instantiation

    simulated : bool
        Boolean value that tracks whether or not the simulation has been run

    Methods
    -------
    load_morph_hdf5(hdf5_file, create_cy_object=True)
        Class method that creates a Morphology object from a morphology HDF5 file

    create_inputData()
        Creates a CyRSoXS InputData object and populates it with parameters from self.config

    create_optical_constants()
        Creates a CyRSoXS RefractiveIndex object and populates it with optical constants from the materials dict

    create_voxel_data()
        Creates a CyRSoXS VoxelData object and populates it with the voxel information from the materials dict

    run(stdout=True,stderr=True, return_xarray=True, print_vec_info=False)
        Creates a CyRSoXS ScatteringPattern object if not already created, and submits all CyRSoXS objects to 
        run the simulation

    scattering_to_xarray(return_xarray=True,print_vec_info=False)
        Copies the CyRSoXS ScatteringPattern arrays to an xarray in the format used by PyHyperScattering for 
        further analysis
    '''

    # dict to deal with specific CyRSoXS input objects. dict structure inspired from David Ackerman's cyrsoxs-framework
    input_mapping = {'CaseType': ['setCaseType',[cy.CaseType.Default, cy.CaseType.BeamDivergence, cy.CaseType.GrazingIncidence]],
                     'MorphologyType': ['setMorphologyType', [cy.MorphologyType.EulerAngles, cy.MorphologyType.VectorMorphology]],
                     'EwaldsInterpolation': ['interpolationType', [cy.InterpolationType.NearestNeighour, cy.InterpolationType.Linear]],
                     'WindowingType': ['windowingType', [cy.FFTWindowing.NoPadding, cy.FFTWindowing.Hanning]],
                     'RotMask': ['rotMask', [False, True]],
                     'AlgorithmType': ['setAlgorithm', [0, 1]],
                     'ReferenceFrame': ['referenceFrame', [0, 1]]}

    config_default = {'CaseType': 0, 'Energies': [270.0], 'EAngleRotation': [0.0, 1.0, 0.0],
                      'MorphologyType': 0, 'AlgorithmType': 0, 'WindowingType': 0,
                      'RotMask': 0,
                      'ReferenceFrame': 1,
                      'EwaldsInterpolation': 1}

    def __init__(self, numMaterial, materials=None, PhysSize=None,
                 config={'CaseType': 0, 'MorphologyType': 0, 'Energies': [270.0], 'EAngleRotation': [0.0, 1.0, 0.0]}, create_cy_object=True):

        self._numMaterial = numMaterial
        self._PhysSize = PhysSize
        self.NumZYX = None
        self.inputData = None
        self.OpticalConstants = None
        self.voxelData = None
        self.scatteringPattern = None
        # add config keys and values to class dict
        for key in self.config_default:
            if key in config:
                self.__dict__['_'+key] = config[key]
            else:
                self.__dict__['_'+key] = self.config_default[key]

        # add materials
        self.materials = {}
        for i in range(1, self._numMaterial+1):
            if materials is None:
                self.materials[i] = Material(materialID=i)
            else:
                try:
                    self.materials[i] = materials[i].copy()
                    if i == 1:
                        self._Energies = materials[i].energies
                        self.NumZYX = materials[i].NumZYX
                except KeyError:
                    warnings.warn('numMaterial is greater than number of Material objects passed in. Creating empty Material')
                    self.materials[i] = Material(materialID=i)

        # flag denoting if Morphology has been simulated
        self._simulated = False

        if create_cy_object:
            self.create_update_cy()

    def __repr__(self):
        return f'Morphology (NumMaterial : {self.numMaterial}, PhysSize : {self.PhysSize})'

    @property
    def CaseType(self):
        return self._CaseType

    @CaseType.setter
    def CaseType(self, casevalue):
        if (casevalue != 0) & (casevalue != 1) & (casevalue !=2):
            raise ValueError('CaseType must be 0, 1, or 2')
        else:
            self._CaseType = casevalue

            if self.inputData:
                self.inputData.setCaseType(self.input_mapping['CaseType'][1][casevalue])

    @property
    def Energies(self):
        return self._Energies

    @Energies.setter
    def Energies(self, Elist):
        self._Energies = Elist

        if self.inputData:
            self.inputData.setEnergies(Elist)

    @property
    def EAngleRotation(self):
        return self._EAngleRotation

    @EAngleRotation.setter
    def EAngleRotation(self, anglelist):
        self._EAngleRotation = anglelist

        if self.inputData:
            self.inputData.setERotationAngle(StartAngle=anglelist[0],
                                             EndAngle=anglelist[2],
                                             IncrementAngle=anglelist[1])

    @property
    def MorphologyType(self):
        return self._MorphologyType

    @MorphologyType.setter
    def MorphologyType(self, value):
        if value != 0:
            raise ValueError('Only Euler Morphology is currently supported')
        else:
            self._MorphologyType = value

            if self.inputData:
                self.inputData.setMorphologyType(self.input_mapping['MorphologyType'][1][value])

    @property
    def AlgorithmType(self):
        return self._AlgorithmType

    @AlgorithmType.setter
    def AlgorithmType(self, value):
        if (value != 0) & (value != 1):
            raise ValueError('AlgorithmType must be 0 (communication minimizing) or 1 (memory minimizing).')
        else:
            self._AlgorithmType = value
            if self.inputData:
                self.inputData.setAlgorithm(AlgorithmID=value,MaxStreams=1)

    @property
    def WindowingType(self):
        return self._WindowingType

    @WindowingType.setter
    def WindowingType(self, value):
        if (value != 0) & (value != 1):
            raise ValueError('WindowingType must be 0 (None) or 1 (Hanning).')
        else:
            self._WindowingType = value

            if self.inputData:
                self.inputData.windowingType = self.input_mapping['WindowingType'][1][value]

    @property
    def RotMask(self):
        return self._RotMask

    @RotMask.setter
    def RotMask(self, value):
        if (value != 0) & (value != 1):
            raise ValueError('RotMask must be 0 (False) or 1 (True).')
        else:
            self._RotMask = value

            if self.inputData:
                self.inputData.rotMask = self._RotMask

    @property
    def EwaldsInterpolation(self):
        return self._EwaldsInterpolation

    @EwaldsInterpolation.setter
    def EwaldsInterpolation(self, value):
        if (value != 0) & (value != 1):
            raise ValueError('EwaldsInterpolation must be 0 (Nearest Neighbor) or 1 (Trilinear).')
        else:
            self._EwaldsInterpolation = value

            if self.inputData:
                self.inputData.interpolationType = self.input_mapping['EwaldsInterpolation'][1][value]

    @property
    def ReferenceFrame(self):
        return self._ReferenceFrame

    @ReferenceFrame.setter
    def ReferenceFrame(self, value):
        if (value != 0) & (value != 1):
            raise ValueError('ReferenceFrame must be 0 (Material Frame) or 1 (Lab Frame - Default).')
        else:
            self._ReferenceFrame = value

            if self.inputData:
                self.inputData.referenceFrame = self.input_mapping['ReferenceFrame'][1][value]

    @property
    def simulated(self):
        return self._simulated

    @property
    def PhysSize(self):
        return self._PhysSize

    @PhysSize.setter
    def PhysSize(self, val):
        if val < 0:
            raise ValueError('PhysSize must be greater than 0')
        self._PhysSize = float(val)
        # update inputData object
        if self.inputData:
            self.inputData.setPhysSize(self._PhysSize)

    @property
    def numMaterial(self):
        return self._numMaterial

    # @numMaterial.setter
    # def numMaterial(self, val):
    #     if val < 0:
    #         raise ValueError('numMaterial must be greater than 0')
    #     self._numMaterial = int(val)
    #     # if we change the number of materials and we have an inputData object, we need to recreate it with the new number of materials
    #     if self.inputData:
    #         self.create_inputData()
    #     if self.OpticalConstants:
    #         self.update_optical_constants()

    @property
    def config(self):
        return {key: self.__dict__['_'+key] for key in self.config_default}

    @config.setter
    def config(self, dict1):
        for key in dict1:
            if key in self.config_default:
                self.__dict__['_'+key] = dict1[key]
            else:
                warnings.warn(f'Key {key} not supported')

    @classmethod
    def load_morph_hdf5(cls, hdf5_file, create_cy_object=False):
        with h5py.File(hdf5_file, 'r') as f:
            if 'Euler_Angles' not in f.keys():
                raise KeyError('Only the Euler Angle convention is currently supported')
            # get number of materials in HDF5
            numMat = check_NumMat(f, morphology_type=0)
            PhysSize = f['Morphology_Parameters/PhysSize'][()]
            materials = dict()

            for i in range(numMat):
                materialID = i + 1
                Vfrac = f[f'Euler_Angles/Mat_{i+1}_Vfrac'][()]
                S = f[f'Euler_Angles/Mat_{i+1}_S'][()]
                theta = f[f'Euler_Angles/Mat_{i+1}_Theta'][()]
                psi = f[f'Euler_Angles/Mat_{i+1}_Psi'][()]
                materials[materialID] = Material(materialID=materialID, 
                                                 Vfrac=Vfrac,
                                                 S=S,
                                                 theta=theta,
                                                 psi=psi,
                                                 NumZYX=Vfrac.shape)

        return cls(numMat, materials=materials, PhysSize=PhysSize, create_cy_object=create_cy_object)

    def load_config(self, config_file):
        self.config = read_config(config_file)

    def load_matfile(self, matfile):
        return read_material(matfile)

    def create_inputData(self):
        self.inputData = cy.InputData(NumMaterial=self._numMaterial)
        # parse config dictionary and assign to appropriate places in inputData object
        self.config_to_inputData()

        if self.NumZYX is None:
            self.NumZYX = self.materials[1].NumZYX

        # only support ZYX ordering at the moment
        self.inputData.setDimensions(self.NumZYX, cy.MorphologyOrder.ZYX)

        if self.PhysSize is not None:
            self.inputData.setPhysSize(self.PhysSize)

        if not self.inputData.validate():
            warnings.warn('Validation failed. Double check inputData values')

    def create_optical_constants(self):
        self.OpticalConstants = cy.RefractiveIndex(self.inputData)
        self.update_optical_constants()        
        if not self.OpticalConstants.validate():
            warnings.warn('Validation failed. Double check optical constant values')

    def update_optical_constants(self):
        for energy in self.Energies:
            all_constants = []
            for ID in range(1, self.numMaterial+1):
                all_constants.append(self.materials[ID].opt_constants[energy])
            self.OpticalConstants.addData(OpticalConstants=all_constants, Energy=energy)

    def create_voxel_data(self):
        self.voxelData = cy.VoxelData(InputData=self.inputData)
        self.update_voxel_data()
        if not self.voxelData.validate():
            warnings.warn('Validation failed. Double check voxel data values')

    def update_voxel_data(self):
        for ID in range(1, self.numMaterial+1):
            self.voxelData.addVoxelData(S=self.materials[ID].S.astype(np.single),
                                        Theta=self.materials[ID].theta.astype(np.single),
                                        Psi=self.materials[ID].psi.astype(np.single),
                                        Vfrac=self.materials[ID].Vfrac.astype(np.single),
                                        MaterialID=ID)

    def config_to_inputData(self):
        for key in self.config:
            if key == "Energies":
                self.inputData.setEnergies(self.config[key])
            elif key == 'EAngleRotation':
                angles = self.config[key]
                self.inputData.setERotationAngle(StartAngle=float(angles[0]),
                                                 EndAngle=float(angles[2]),
                                                 IncrementAngle=float(angles[1]))
            elif key == 'AlgorithmType':
                self.inputData.setAlgorithm(AlgorithmID=self.config[key], MaxStreams=1)
            # if the key corresponds to one of the idiosyncratic methods, use this
            elif key in self.input_mapping.keys():
                func = getattr(self.inputData,self.input_mapping[key][0])
                if callable(func):
                    func(self.input_mapping[key][1][self.config[key]])
                # if the attribute is not callable, use input_mapping to set the attribute
                else:
                    setattr(self.inputData,
                            self.input_mapping[key][0],
                            self.input_mapping[key][1][self.config[key]])
            else:
                warnings.warn(f'{key} is currently not implemented')

    def create_update_cy(self):
        # create or update all CyRSoXS objects
        if self.inputData:
            self.config_to_inputData()
        else:
            self.create_inputData()

        # create or update OpticalConstants
        if self.OpticalConstants:
            self.update_optical_constants()            
        else:
            self.create_optical_constants()

        # create or udpate voxelData
        if self.voxelData:
            self.voxelData.reset()
            self.update_voxel_data()
        else:
            self.create_voxel_data()

    def write_to_file(self, fname, author='NIST'):
        _ = write_hdf5([[self.materials[i].Vfrac,
                        self.materials[i].S,
                        self.materials[i].theta,
                        self.materials[i].psi] for i in self.materials],
                        self.PhysSize, fname, self.MorphologyType, ordering='ZYX', author=author)

    # TODO : function to write a config.txt file from config dict
    def write_config(self,):
        pass

    def write_constants(self, path=None):
        for i in range(1, self._numMaterial+1):
            write_opts(self.materials[i].opt_constants, i, path)

    # submit to CyRSoXS
    def run(self, stdout=True, stderr=True, return_xarray=True, print_vec_info=False, validate=False):
        if validate:
            self.create_update_cy()

        # if we haven't created a ScatteringPattern object, create it now
        if not self.scatteringPattern:
            self.scatteringPattern = cy.ScatteringPattern(self.inputData)
        with cy.ostream_redirect(stdout=stdout, stderr=stderr):
            cy.launch(VoxelData=self.voxelData,
                      RefractiveIndexData=self.OpticalConstants,
                      InputData=self.inputData,
                      ScatteringPattern=self.scatteringPattern)
        self._simulated = True
        if return_xarray:
            return self.scattering_to_xarray(return_xarray=return_xarray, print_vec_info=print_vec_info)

    def scattering_to_xarray(self, return_xarray=True, print_vec_info=False):
        if self.simulated:
            if not print_vec_info:
                old_stdout = sys.stdout
                f = open(os.devnull, 'w')
                sys.stdout = f

            # data will be returned in shape [energy, NumY, NumX]
            scattering_data = self.scatteringPattern.writeAllToNumpy(kID=0)
            qy = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(self.NumZYX[1], d=self.PhysSize))
            qx = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(self.NumZYX[2], d=self.PhysSize))
            scattering_data = xr.DataArray(scattering_data,
                                           dims=['energy', 'qy', 'qx'],
                                           coords={'qy': qy, 'qx': qx, 'energy': self.config['Energies']})

            if not print_vec_info:
                sys.stdout = old_stdout
                f.close()

            if return_xarray:
                return scattering_data
            else:
                self.scattering_data = scattering_data
        else:
            warnings.warn('You haven\'t run your simulation yet')

    # TODO : restructure to have a single checkH5 engine for both NRSS and
    # command line formats
    def check_materials(self, quiet=True):
        Vfrac_test = np.zeros(self.materials[1].Vfrac.shape)
        for i in range(1, self._numMaterial+1):
            Vfrac_test += self.materials[i].Vfrac

            # test that S and Vfrac lie between 0 and 1 for each material
            assert np.all((self.materials[i].S >= 0) & (self.materials[i].S <= 1)), f'Material {i} S value(s) does not lie between 0 and 1'
            assert np.all((self.materials[i].Vfrac >= 0) & (self.materials[i].Vfrac <= 1)), f'Material {i} Vfrac value(s) does not lie between 0 and 1'

            # test for NaNs
            for name in ['S', 'Vfrac', 'theta', 'psi']:
                assert np.all(~np.isnan(getattr(self.materials[i], name))), f'NaNs are present in Material {i} {name}'
                assert ('float' in getattr(self.materials[i], name).dtype.name), f'Material {i} {name} is not of type float' 

        assert np.allclose(Vfrac_test, 1), 'Total material volume fractions do not sum to 1'

        # delete Vfrac_test after validation
        del Vfrac_test

        if not quiet:
            print('All material checks have passed')

    @wraps(morphology_visualizer)
    def visualize_materials(self, *args,**kwargs):
        return morphology_visualizer(self, *args,**kwargs)
    visualize_materials.__doc__ = morphology_visualizer.__doc__


    def validate_all(self, quiet=True):
        self.check_materials(quiet=quiet)
        input_check = self.inputData.validate()
        opt_const_check = self.OpticalConstants.validate()
        voxel_check = self.voxelData.validate()
        assert (input_check), 'CyRSoXS object inputData validation has failed'
        assert (opt_const_check), 'CyRSoXS object OpticalConstants validation has failed'
        assert (voxel_check), 'CyRSoXS object voxelData validation has failed'
        if not quiet:
            print('All objects have been validated successfully. You can run your simulation')


class OpticalConstants:
    '''
    Object to hold dielectric optical constants in a format compatible with CyRSoXS

    Attributes
    ----------

    energies : list or array
        List of energies
    opt_constants : dict
        Dictionary of optical constants, where each energy is a key in the dict
    name : str
        String identifying the element or material for these optical constants

    Methods
    -------
    calc_constants(energies, reference_data, name='unkown')
        Interpolates optical constant data to the list of energies provided
    load_matfile(matfile, name='unknown')
        Creates an OpticalConstants object from a previously written MaterialX.txt file
    create_vacuum(energies)
        Convenience function to populate zeros for all optical constants

    '''

    def __init__(self, energies, opt_constants=None, name='unknown'):
        self.energies = energies
        self.opt_constants = opt_constants
        self.name = name
        if self.name == 'vacuum':
            self.create_vacuum(energies)

    def __repr__(self):
        return f'OpticalConstants (Material : {self.name}, Number of Energies : {len(self.energies)})'

    @classmethod
    def calc_constants(cls, energies, reference_data, name='unknown'):
        deltabeta = dict()
        for energy in energies:
            dPara = np.interp(energy, reference_data['Energy'], reference_data['DeltaPara'])
            bPara = np.interp(energy, reference_data['Energy'], reference_data['BetaPara'])
            dPerp = np.interp(energy, reference_data['Energy'], reference_data['DeltaPerp'])
            bPerp = np.interp(energy, reference_data['Energy'], reference_data['BetaPerp'])
            deltabeta[energy] = [dPara, bPara, dPerp, bPerp]
        return cls(energies, deltabeta, name=name)

    @classmethod
    def load_matfile(cls, matfile, name='unknown'):
        energies, deltabeta = read_material(matfile)
        return cls(energies, deltabeta, name=name)

    def create_vacuum(self, energies):
        deltabeta = dict()
        for energy in energies:
            deltabeta[energy] = [0.0, 0.0, 0.0, 0.0]
        self.energies = energies
        self.opt_constants = deltabeta


class Material(OpticalConstants):
    '''
    Object to hold the voxel-level data for a CyRSoXS morphology. Inherits from the OpticalConstants class.

    Attributes
    ----------
    materialID : int
        Integer value denoting the material number. Used in CyRSoXS
    Vfrac : ndarray
        Volume fractions for a Material
    theta : ndarray
        The second Euler angle (ZYZ convention)
    psi : ndarray
        The third Euler angle (ZYZ convention)
    NumZYX : tuple or list
        Dimensions of the Material arrays (NumZ, NumY, NumX)
    name : str
        Name of the Material (e.g. 'Polystyrene')

    '''

    def __init__(self, materialID=1, Vfrac=None, S=None, theta=None, psi=None,
                 NumZYX=None, energies=None, opt_constants=None, name=None):
        self.materialID = materialID
        self.Vfrac = Vfrac
        self.S = S
        self.theta = theta
        self.psi = psi
        self.NumZYX = NumZYX
        self.name = name
        self.energies = energies
        self.opt_constants = opt_constants
        if self.NumZYX is None:
            try:
                self.NumZYX = Vfrac.shape
            except AttributeError:
                pass

        if (energies is None) & (opt_constants is not None):
            self.energies = list(opt_constants.keys())

        super().__init__(self.energies, self.opt_constants, name=name)

    def __repr__(self):
        return f'Material (Name : {self.name}, ID : {self.materialID}, Shape : {self.NumZYX})'

    def __copy__(self):
        return Material(materialID=self.materialID,
                        Vfrac=self.Vfrac.copy(),
                        S=self.S.copy(),
                        theta=self.theta.copy(),
                        psi=self.psi.copy(),
                        NumZYX=self.NumZYX,
                        energies=self.energies,
                        opt_constants=self.opt_constants,
                        name=self.name)

    def copy(self):
        return copy.copy(self)
