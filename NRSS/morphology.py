import h5py
import pathlib
import CyRSoXS as cy
import warnings
from .checkH5 import check_NumMat
from .reader import read_material, read_config
import numpy as np
import xarray as xr
import sys
import os

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
    
    create_CyObject : bool
        Boolean value that decides if the CyRSoXS objects are created upon instantiation
    
    simulated : bool
        Boolean value that tracks whether or not the simulation has been run

    Methods
    -------
    load_morph_hdf5(hdf5_file, create_CyObject=True)
        Class method that creates a Morphology object from a morphology HDF5 file

    create_inputData()
        Creates a CyRSoXS InputData object and populates it with parameters from self.config
    
    create_OpticalConstants()
        Creates a CyRSoXS RefractiveIndex object and populates it with optical constants from the materials dict
    
    create_voxelData()
        Creates a CyRSoXS VoxelData object and populates it with the voxel information from the materials dict
    
    run(stdout=True,stderr=True, return_xarray=True, print_vec_info=False)
        Creates a CyRSoXS ScatteringPattern object if not already created, and submits all CyRSoXS objects to run the simulation
    
    scattering_to_xarray(return_xarray=True,print_vec_info=False)
        Copies the CyRSoXS ScatteringPattern arrays to an xarray in the format used by PyHyperScattering for further analysis
    '''

    # dict to deal with specific CyRSoXS input objects. dict structure inspired from David Ackerman's cyrsoxs-framework
    input_mapping = {'CaseType':['setCaseType',[cy.CaseType.Default,cy.CaseType.BeamDivergence,cy.CaseType.GrazingIncidence]]
                    ,'MorphologyType':['setMorphologyType',[cy.MorphologyType.EulerAngles, cy.MorphologyType.VectorMorphology]]
                    ,'EwaldsInterpolation':['interpolationType',[cy.InterpolationType.NearestNeighour, cy.InterpolationType.Linear]]
                    ,'WindowingType':['windowingType',[cy.FFTWindowing.NoPadding, cy.FFTWindowing.Hanning]]
                    }


    
    def __init__(self, numMaterial, materials=None, PhysSize=None, NumZYX=None, 
                config = {'CaseType':0, 'MorphologyType': 0, 'Energies': [270.0], 'EAngleRotation':[0.0, 1.0, 0.0]}, create_CyObject=False):
        self._numMaterial = numMaterial
        self._PhysSize = PhysSize
        self.NumZYX = NumZYX
        self._config = config.copy()
        if materials is not None:
            self.materials = materials
            self._config['Energies'] = materials[1].energies
        else:
            self.materials = None
        self.inputData = None
        self.simulated = False
        if create_CyObject:
            self.create_InputData()
            if self.materials:
                self.create_OpticalConstants()
                self.create_voxelData()

    
    def __repr__(self):
        return f'Morphology (NumMaterial : {self.numMaterial}, PhysSize : {self.PhysSize})'

    @property
    def PhysSize(self):
        return self._PhysSize
    
    @PhysSize.setter
    def PhysSize(self,val):
        if val < 0:
            raise ValueError('PhysSize must be greater than 0')
        self._PhysSize = float(val)
        # update inputData object
        if self.inputData:
            self.inputData.setPhysSize(self._PhysSize)

    @property
    def numMaterial(self):
        return self._numMaterial
    
    @numMaterial.setter
    def numMaterial(self, val):
        if val < 0:
            raise ValueError('numMaterial must be greater than 0')
        self._numMaterial = int(val)
        # if we change the number of materials and we have an inputData object, we need to recreate it with the new number of materials
        if self.inputData:
            self.create_InputData()
        if self.OpticalConstants:
            self.updateOpticalConstants()
    
    @property
    def config(self):
        return self._config

    @config.setter
    def config(self,dict1):
        self._config = dict1
        # update inputData to reflect config
        if self.inputData:
            self.config_to_inputData()

    @classmethod
    def load_morph_hdf5(cls,hdf5_file, create_CyObject=True):
        with h5py.File(hdf5_file,'r') as f:
            if 'Euler_Angles' not in f.keys():
                raise KeyError('Only the Euler Angle convention is currently supported')
            # get number of materials in HDF5
            numMat = check_NumMat(f)
            PhysSize = f['Morphology_Parameters/PhysSize']
            materials = dict()

            for i in range(numMat):
                materialID = i + 1
                Vfrac= f[f'Euler_Angles/Mat_{i+1}_Vfrac']
                S = f[f'Euler_Angles/Mat_{i+1}_S']
                theta = f[f'Euler_Angles/Mat_{i+1}_Theta']
                psi = f[f'Euler_Angles/Mat_{i+1}_Psi']
                materials[materialID] = Material(materialID=materialID, 
                                                 Vfrac=Vfrac,
                                                 S=S,
                                                 theta=theta,
                                                 psi=psi,
                                                 NumZYX=Vfrac.shape)

        return cls(numMat, materials=materials, PhysSize=PhysSize, NumZYX=materials[materialID].NumZYX, create_CyObject=True)
            
    
    def load_config(self, config_file):
        self._config = read_config(config_file)
        
    
    def load_matfile(self, matfile):
        return read_material(matfile)

    def create_InputData(self):
        self.inputData = cy.InputData(NumMaterial=self._numMaterial)
        # parse config dictionary and assign to appropriate places in inputData object
        self.config_to_inputData()
        
        if self.NumZYX is None:
            self.NumZYX = self.materials[1].NumZYX
        
        #only support ZYX ordering at the moment
        self.inputData.setDimensions(self.NumZYX, cy.MorphologyOrder.ZYX)

        if self.PhysSize is not None:
            self.inputData.setPhysSize(self.PhysSize)
        
        if not self.inputData.validate():
            warnings.warn('Validation failed. Double check inputData values')
    
    def create_OpticalConstants(self):
        self.OpticalConstants = cy.RefractiveIndex(self.inputData)
        self.updateOpticalConstants()        
        if not self.OpticalConstants.validate():
            warnings.warn('Validation failed. Double check optical constant values')


    def updateOpticalConstants(self):
        for energy in self._config['Energies']:
            all_constants = []
            for ID in range(1,self.numMaterial+1):
                all_constants.append(self.materials[ID].opt_constants[energy])
            self.OpticalConstants.addData(OpticalConstants=all_constants, Energy=energy)

    def create_voxelData(self):
        self.voxelData = cy.VoxelData(InputData = self.inputData)
        for ID in range(1, self.numMaterial+1):
            self.voxelData.addVoxelData(S=self.materials[ID].S.astype(np.single),
                                        Theta=self.materials[ID].theta.astype(np.single),
                                        Psi=self.materials[ID].psi.astype(np.single),
                                        Vfrac=self.materials[ID].Vfrac.astype(np.single),
                                        MaterialID=ID)
        if not self.voxelData.validate():
            warnings.warn('Validation failed. Double check voxel data values')
        
    def config_to_inputData(self):
        for key in self._config:
            if key == "Energies":
                self.inputData.setEnergies(self._config[key])
            elif key == 'EAngleRotation':
                angles = self._config[key]
                self.inputData.setERotationAngle(StartAngle = float(angles[0]), 
                                                    EndAngle = float(angles[2]), 
                                                    IncrementAngle = float(angles[1]))
            # if the key corresponds to one of the idiosyncratic methods, use this
            elif key in self.input_mapping.keys():
                func = getattr(self.inputData,self.input_mapping[key][0])
                if callable(func):
                    func(self.input_mapping[key][1][self._config[key]])
                # if the attribute is not callable, use input_mapping to set the attribute
                else:
                    setattr(self.inputData,
                            self.input_mapping[key][0],
                            self.input_mapping[key][1][self._config[key]])
            else:
                warnings.warn(f'{key} is currently not implemented')    
            
    #TODO : function to write morphology to HDF5
    def write_hdf5(self,):
        pass
    
    #TODO : function to write a config.txt file from config dict
    def write_config(self,):
        pass
    
    #TODO : function to write constants to MaterialX.txt files
    def write_constants(self,):
        pass
    
    #submit to CyRSoXS
    def run(self,stdout=True,stderr=True, return_xarray=True, print_vec_info=False):
        # run one more time to make sure everything has been updated
        self.config_to_inputData()
        # if we haven't created a ScatteringPattern object, create it now
        try:
            self.scatteringPattern
        except AttributeError:
            self.scatteringPattern = cy.ScatteringPattern(self.inputData)
        with cy.ostream_redirect(stdout=stdout,stderr=stderr):
            cy.launch(VoxelData=self.voxelData, 
                      RefractiveIndexData=self.OpticalConstants, 
                      InputData=self.inputData,
                      ScatteringPattern=self.scatteringPattern)
        self.simulated = True
        if return_xarray:
            return self.scattering_to_xarray(return_xarray=return_xarray,print_vec_info=print_vec_info)

    def scattering_to_xarray(self,return_xarray=True,print_vec_info=False):
        if self.simulated:
            if not print_vec_info:
                old_stdout = sys.stdout
                f = open(os.devnull,'w')
                sys.stdout = f

            scattering_data = np.zeros((self.NumZYX[1],self.NumZYX[2],len(self._config['Energies'])))

            for i,energy in enumerate(self._config['Energies']):
                scattering_data[:,:,i] = self.scatteringPattern.dataToNumpy(energy,0)
            qy = np.fft.fftshift(np.fft.fftfreq(self.NumZYX[1],d=self.PhysSize))
            qx = np.fft.fftshift(np.fft.fftfreq(self.NumZYX[2],d=self.PhysSize))
            scattering_data = xr.DataArray(scattering_data,
                                            dims=['qy','qx','energy'],
                                            coords={'qy':qy,'qx':qx,'energy':self._config['Energies']})
            
            if not print_vec_info:
                sys.stdout = old_stdout
                f.close()
            
            if return_xarray:
                return scattering_data
            else:
                self.scattering_data = scattering_data
        else:
            warnings.warn('You haven\'t run your simulation yet')

    #TODO : call checkH5 to validate morphology
    def check_materials(self,):
        pass

    def validate_all(self):
        passed = 0
        if self.inputData.validate():
            print('inputData has passed validation.')
            passed += 1
        if self.OpticalConstants.validate():
            print('OpticalConstants has passed validation.')
            passed += 1
        if self.voxelData.validate():
            print('voxelData has passed validation.')
            passed += 1
        if passed == 3:
            print('\n')
            print('All CyRSoXS objects have passed validation, you can run your simulation.')


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

    def __init__(self, energies, opt_constants=None, name = 'unknown'):
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
            dPara = np.interp(energy,reference_data['Energy'],reference_data['DeltaPara'])
            bPara = np.interp(energy,reference_data['Energy'],reference_data['BetaPara'])
            dPerp = np.interp(energy,reference_data['Energy'],reference_data['DeltaPerp'])
            bPerp = np.interp(energy,reference_data['Energy'],reference_data['BetaPerp'])
            deltabeta[energy] = [dPara, bPara, dPerp, bPerp]
        return cls(energies,deltabeta,name=name)

    @classmethod
    def load_matfile(cls, matfile,name='unknown'):
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


    def __init__(self, materialID=1, Vfrac=None, S=None, theta=None, psi = None, 
                    NumZYX=None, energies = None, opt_constants = None, name=None):
        self.materialID = materialID
        self.Vfrac = Vfrac
        self.S = S
        self.theta = theta
        self.psi = psi
        self.NumZYX = NumZYX
        self.name = name
        if self.NumZYX is None:
            try:
                self.NumZYX = Vfrac.shape
            except AttributeError:
                pass
        
        super().__init__(energies, opt_constants, name=name)

    def __repr__(self):
        return f'Material (Name : {self.name}, ID : {self.materialID}, Shape : {self.NumZYX})'

