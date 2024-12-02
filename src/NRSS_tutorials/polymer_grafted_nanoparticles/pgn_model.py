from copy import copy
import numpy as np
import cupy as cp
from NRSS.morphology import Material, Morphology, wraps
from PyHyperScattering.integrate import WPIntegrator
import pickle, lzma
from pathlib import Path
import glob
from tqdm import tqdm
from pprint import pprint
from matplotlib import colormaps, gridspec, rc, ticker
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from xarray import DataArray
from dataclasses import dataclass, field


@dataclass
class Arguments():
    """
    Dataclass for storing parameters for the core-shell spheres decay simulation
    """
    r: int = 512
    """
    Number of voxels in YX dimension of the simulation box
    """
    d: int = 32
    """
    Number of voxels in Z dimension of the simulation box
    """
    vx_size: float = 2.5
    """
    Size of each voxel in the simulation box in nm
    """
    ra: float = 9.9
    """
    Radius of the core particle
    """
    t: float = 7.3
    """
    Thickness of the oriented corona
    """
    S_0: float = 0.11
    """
    orientation immediately near the core particle
    """
    S_range_max: float = 0.11
    """
    orientation magnitude maximum sed for S visualizations
    """
    S_cmap: str = "afmhot"
    """
    orientation immediately near the core particle
    """
    decay_order: float = 0.421
    """
    Decay order for the core-shell spheres
    """
    energies: np.ndarray = field(default_factory=lambda: np.array([260, 270, 285]))
    """
    Energies at which to simulate the scattering
    """
    coord_list: list = field(default_factory=list)
    """
    List of coordinates for the core-shell spheres
    """
    return_2d: bool = False
    """
    whether to return 2D pattern simulations in result object
    """

@dataclass
class Visualization():
    """
    Dataclass for storing visualization parameters and results
    """
    outputmat: list = field(default_factory=list)
    """
    List of materials to output in the visualization
    """
    outputplot: list = field(default_factory=list)
    """
    List of plots to output in the visualization
    """
    img_dict: dict = field(default_factory=dict)
    """
    List of images output from the visualization
    """
    
    @wraps(Morphology.visualize_materials)
    def get_visualizations(self, morph, **kwargs):
        """
        Get visualization results for the given morphology
        
        Parameters
        ----------
        morph : Morphology
            Morphology object to visualize
        **kwargs
            Additional keyword arguments to pass to `morph.visualize_materials`
        
        Returns
        -------
        self : Visualization
            The same Visualization object with the results stored as attributes
        """
        kwargs["runquiet"] = True
        kwargs["outputmat"] = self.outputmat
        kwargs["outputplot"] = self.outputplot

        visualization_results = morph.visualize_materials(
        **kwargs
        )
        self.img_dict = {}
        for i, mat in enumerate(self.outputmat):
            for j, vizplot in enumerate(self.outputplot):
                self.img_dict[f"mat_{mat}_{vizplot}"] = visualization_results[i*len(self.outputplot) + j]
        return self

@dataclass
class Results():
    twod: DataArray = field(default_factory=DataArray) # 2D image data from the simulation
    I: DataArray = field(default_factory=DataArray) # Intensity data from the simulation
    A: DataArray  = field(default_factory=DataArray) # Anisotropy data from the simulation
    viz: list = field(default_factory=list)  # Visualization data from the simulation


def coreshell_spheres_decay_euler_old(args):

    r = args.r
    d = args.d
    vx_size = args.vx_size
    ra_vx = args.ra / args.vx_size
    t_vx = args.t / args.vx_size
    phi_iso = 1 - args.S_0
    decay_order = args.decay_order
    l = args.coord_list

    # Initialize boolean masks for material presence
    a_b = cp.full((d, r, r), False)  # core
    b_b = cp.full((d, r, r), False)  # shell
    c_b = cp.full((d, r, r), False)  # everything else

    # Initialize orientation components of shell
    b_x = cp.zeros([d, r, r])  # shell
    b_y = cp.zeros([d, r, r])  # shell
    b_z = cp.zeros([d, r, r])  # shell

    # Initialize volume fractions of each material - euler
    vf_a = cp.zeros([d, r, r])  # core
    vf_b = cp.zeros([d, r, r])  # shell
    vf_c = cp.zeros([d, r, r])  # everything else

    # Initialize euler angles and aligned fraction - euler morphology - ZXZ rotations
    SE_b = cp.zeros([d, r, r])
    theta_b = cp.zeros([d, r, r])  # theta of shell
    psi_b = cp.zeros([d, r, r])  # psi of shell

    # Initialize intermediate orientation components of shell
    b_xi = cp.zeros([d, r, r])  # shell
    b_yi = cp.zeros([d, r, r])  # shell
    b_zi = cp.zeros([d, r, r])  # shell

    z, y, x = cp.ogrid[0:d:1, 0:r:1, 0:r:1]

    for p in l:
        mf = (x - p[0]) ** 2 + (y - p[1]) ** 2 + (z - 15) ** 2 
        mask = mf <= ra_vx ** 2
        a_b = cp.logical_or(a_b, mask)
        mask = (a_b == False) & (mf <= (ra_vx + t_vx) ** 2)
        b_b = cp.logical_or(b_b, mask)

        b_xi = ((x - p[0]) + y * 0 + z * 0) * mask
        b_yi = (x * 0 + (y - p[1]) + z * 0) * mask
        b_zi = (x * 0 + y * 0 + (z - 15)) * mask

        b_x = b_x + b_xi * (b_x == 0)
        b_y = b_y + b_yi * (b_y == 0)
        b_z = b_z + b_zi * (b_z == 0)

    b_b = (a_b == False) & (b_b == True)
    b_x = b_x * b_b
    b_y = b_y * b_b
    b_z = b_z * b_b

    c_b = (a_b == False) & (b_b == False)

    b_t = (b_x ** 2 + b_y ** 2 + b_z ** 2) ** 0.5

    b_1 = (ra_vx) * b_b

    vf_b = 1 * b_b
    SE_b = b_b * (1 - phi_iso) * (b_1 / b_t) ** (decay_order)
    vf_a = 1 * a_b
    vf_c = 1 * c_b

    SE_b = cp.nan_to_num(SE_b)

    theta_b = cp.arctan2(((b_x) ** 2 + (b_y) ** 2) ** 0.5, b_z)
    psi_b = cp.arctan2(b_y, b_x)

    vf_a = cp.asnumpy(vf_a)
    vf_b = cp.asnumpy(vf_b)
    vf_c = cp.asnumpy(vf_c)
    SE_b = cp.asnumpy(SE_b)
    theta_b = cp.asnumpy(theta_b)
    psi_b = cp.asnumpy(psi_b)

    a_zeros = np.zeros(vf_a.shape, dtype=np.float32)

    mat1_poly = Material(
        materialID=1,
        Vfrac=(vf_b).astype(np.float32),
        S=SE_b.astype(np.float32),
        theta=theta_b.astype(np.float32),
        psi=psi_b.astype(np.float32),
        NumZYX=vf_a.shape,
        energies=args.energies,
        opt_constants=args.oc_polymer.opt_constants,
        name="PS",
    )

    mat2_particle = Material(
        materialID=2,
        Vfrac=vf_a.astype(np.float32),
        S=a_zeros,
        theta=a_zeros,
        psi=a_zeros,
        NumZYX=vf_a.shape,
        energies=args.energies,
        opt_constants=args.oc_particle.opt_constants,
        name="particle",
    )

    # u = np.random.uniform(size = vf_c.shape)
    # v = np.random.uniform(size =vf_c.shape)
    # theta_rand = 2*np.pi*u
    # psi_rand = np.arccos(2*v-1)

    mat3_poly = Material(
        materialID=3,
        Vfrac=(vf_c).astype(np.float32),
        S=a_zeros.astype(np.float32),
        theta=a_zeros.astype(np.float32),
        psi=a_zeros.astype(np.float32),
        NumZYX=a_zeros.shape,
        energies=args.energies,
        opt_constants=args.oc_polymer.opt_constants,
        name="PS",
    )

    morph = Morphology(
        3,
        {1: mat1_poly, 2: mat2_particle, 3: mat3_poly},
        PhysSize=args.vx_size,
    )

    # Clean up
    del(a_b, b_b, c_b, b_x, b_y, b_z, b_xi, b_yi, b_zi, b_t, b_1, SE_b, theta_b, psi_b, vf_a, vf_b, vf_c, a_zeros, mat1_poly, mat2_particle)

    return morph


def coreshell_spheres_decay_euler(args):
    """
    Creates a morphology with a core-shell particle and a polymer matrix.

    Parameters
    ----------
    args : Arguments
        A dataclass containing the parameters for the morphology

    Returns
    -------
    morph : Morphology
        The created morphology
    """
    # Initialize boolean masks for material presence
    core_mask = cp.zeros((args.d, args.r, args.r), dtype=bool)
    shell_mask = cp.zeros((args.d, args.r, args.r), dtype=bool)
    polymer_mask = cp.ones((args.d, args.r, args.r), dtype=bool)

    # Initialize orientation components of shell
    shell_x = cp.zeros((args.d, args.r, args.r))
    shell_y = cp.zeros((args.d, args.r, args.r))
    shell_z = cp.zeros((args.d, args.r, args.r))

    # Initialize volume fractions of each material
    core_volume_fraction = cp.zeros((args.d, args.r, args.r))
    shell_volume_fraction = cp.zeros((args.d, args.r, args.r))
    polymer_volume_fraction = cp.zeros((args.d, args.r, args.r))

    # Initialize euler angles and aligned fraction - euler morphology - ZXZ rotations
    shell_S = cp.zeros((args.d, args.r, args.r))
    shell_theta = cp.zeros((args.d, args.r, args.r))
    shell_psi = cp.zeros((args.d, args.r, args.r))

    # Initialize intermediate orientation components of shell
    shell_xi = cp.zeros((args.d, args.r, args.r))
    shell_yi = cp.zeros((args.d, args.r, args.r))
    shell_zi = cp.zeros((args.d, args.r, args.r))

    z, y, x = cp.ogrid[0 : args.d : 1, 0 : args.r : 1, 0 : args.r : 1]

    for point in args.coord_list:
        squared_distance_from_point = (
            (x - point[0]) ** 2 + (y - point[1]) ** 2 + (z - 15) ** 2
        )  # calculate the squared distance from each point in the grid to the current particle
        mask = squared_distance_from_point <= (args.ra / args.vx_size) ** 2  # create a mask for the particle core
        core_mask = cp.logical_or(core_mask, mask)  # update the core mask
        mask = (core_mask == False) & (
            squared_distance_from_point
            <= (args.ra / args.vx_size + args.t / args.vx_size) ** 2
        )  # create a mask for the particle shell
        shell_mask = cp.logical_or(shell_mask, mask)  # update the shell mask

        shell_xi = ((x - point[0]) + y * 0 + z * 0) * mask  # calculate the x-component of the shell orientation
        shell_yi = (x * 0 + (y - point[1]) + z * 0) * mask  # calculate the y-component of the shell orientation
        shell_zi = (x * 0 + y * 0 + (z - 15)) * mask  # calculate the z-component of the shell orientation

        shell_x += shell_xi * (shell_x == 0)  # update the x-component of the shell orientation
        shell_y += shell_yi * (shell_y == 0)  # update the y-component of the shell orientation
        shell_z += shell_zi * (shell_z == 0)  # update the z-component of the shell orientation

    shell_mask = (core_mask == False) & (shell_mask == True)
    shell_x *= shell_mask
    shell_y *= shell_mask
    shell_z *= shell_mask

    polymer_mask = (core_mask == False) & (shell_mask == False)

    shell_t = (shell_x**2 + shell_y**2 + shell_z**2) ** 0.5  # calculate the magnitude of the shell orientation

    shell_1 = (args.ra / args.vx_size) * shell_mask  # calculate the radial distance from the particle center to the shell

    shell_volume_fraction = 1 * shell_mask  # calculate the volume fraction of the shell
    shell_S = shell_mask * args.S_0 * (shell_1 / shell_t) ** (args.decay_order)  # calculate the aligned fraction of the shell
    core_volume_fraction = 1 * core_mask  # calculate the volume fraction of the particle core
    polymer_volume_fraction = 1 * polymer_mask  # calculate the volume fraction of the polymer matrix

    shell_S = cp.nan_to_num(shell_S)

    shell_theta = cp.arctan2(((shell_x) ** 2 + (shell_y) ** 2) ** 0.5, shell_z)  # calculate the theta angle of the shell orientation
    shell_psi = cp.arctan2(shell_y, shell_x)  # calculate the psi angle of the shell orientation

    core_volume_fraction = cp.asnumpy(core_volume_fraction)
    shell_volume_fraction = cp.asnumpy(shell_volume_fraction)
    polymer_volume_fraction = cp.asnumpy(polymer_volume_fraction)
    shell_S = cp.asnumpy(shell_S)
    shell_theta = cp.asnumpy(shell_theta)
    shell_psi = cp.asnumpy(shell_psi)

    a_zeros = np.zeros(core_volume_fraction.shape, dtype=np.float32)

    mat1_poly = Material(
        materialID=1,
        Vfrac=(shell_volume_fraction).astype(np.float32),
        S=shell_S.astype(np.float32),
        theta=shell_theta.astype(np.float32),
        psi=shell_psi.astype(np.float32),
        NumZYX=core_volume_fraction.shape,
        energies=args.energies,
        opt_constants=args.oc_polymer.opt_constants,
        name="PS",
    )

    mat2_particle = Material(
        materialID=2,
        Vfrac=core_volume_fraction.astype(np.float32),
        S=a_zeros,
        theta=a_zeros,
        psi=a_zeros,
        NumZYX=core_volume_fraction.shape,
        energies=args.energies,
        opt_constants=args.oc_particle.opt_constants,
        name="particle",
    )

    mat3_poly = Material(
        materialID=3,
        Vfrac=(polymer_volume_fraction).astype(np.float32),
        S=a_zeros.astype(np.float32),
        theta=a_zeros.astype(np.float32),
        psi=a_zeros.astype(np.float32),
        NumZYX=a_zeros.shape,
        energies=args.energies,
        opt_constants=args.oc_polymer.opt_constants,
        name="PS",
    )

    morph = Morphology(
        3,
        {1: mat1_poly, 2: mat2_particle, 3: mat3_poly},
        PhysSize=args.vx_size,
    )

    del (
        core_mask,
        shell_mask,
        polymer_mask,
        shell_x,
        shell_y,
        shell_z,
        shell_xi,
        shell_yi,
        shell_zi,
        core_volume_fraction,
        shell_volume_fraction,
        polymer_volume_fraction,
        shell_S,
        shell_theta,
        shell_psi,
        a_zeros,
        mat1_poly,
        mat2_particle,
        mat3_poly,
    )
    # a cupy helper function to ensure that unused memory is freed
    cp._default_memory_pool.free_all_blocks()
    return morph

def coreshell_spheres_decay_euler_16bit(args):
    """
    Creates a morphology with a core-shell particle and a polymer matrix.

    Parameters
    ----------
    args : Arguments
        A dataclass containing the parameters for the morphology

    Returns
    -------
    morph : Morphology
        The created morphology
    """
    # Initialize boolean masks for material presence
    core_mask = cp.zeros((args.d, args.r, args.r), dtype=bool)
    shell_mask = cp.zeros((args.d, args.r, args.r), dtype=bool)
    polymer_mask = cp.ones((args.d, args.r, args.r), dtype=bool)

    # Initialize orientation components of shell
    shell_x = cp.zeros((args.d, args.r, args.r), dtype=cp.float16)
    shell_y = cp.zeros((args.d, args.r, args.r), dtype=cp.float16)
    shell_z = cp.zeros((args.d, args.r, args.r), dtype=cp.float16)

    # Initialize volume fractions of each material
    core_volume_fraction = cp.zeros((args.d, args.r, args.r), dtype=cp.float16)
    shell_volume_fraction = cp.zeros((args.d, args.r, args.r), dtype=cp.float16)
    polymer_volume_fraction = cp.zeros((args.d, args.r, args.r), dtype=cp.float16)

    # Initialize euler angles and aligned fraction - euler morphology - ZXZ rotations
    shell_S = cp.zeros((args.d, args.r, args.r), dtype=cp.float16)
    shell_theta = cp.zeros((args.d, args.r, args.r), dtype=cp.float16)
    shell_psi = cp.zeros((args.d, args.r, args.r), dtype=cp.float16)

    # Initialize intermediate orientation components of shell
    shell_xi = cp.zeros((args.d, args.r, args.r), dtype=cp.float16)
    shell_yi = cp.zeros((args.d, args.r, args.r), dtype=cp.float16)
    shell_zi = cp.zeros((args.d, args.r, args.r), dtype=cp.float16)

    z,y,x = cp.ix_(cp.arange(args.d, dtype=cp.float16), cp.arange(args.r, dtype = cp.float16), cp.arange(args.r, dtype = cp.float16))

    for point in args.coord_list:
        squared_distance_from_point = (
            (x - cp.float16(point[0])) ** 2 + (y - cp.float16(point[1])) ** 2 + (z - 15) ** 2
        )  # calculate the squared distance from each point in the grid to the current particle
        mask = squared_distance_from_point <= (cp.float16(args.ra / args.vx_size)) ** 2  # create a mask for the particle core
        core_mask = cp.logical_or(core_mask, mask)  # update the core mask
        mask = (core_mask == False) & (
            squared_distance_from_point
            <= cp.float16(args.ra / args.vx_size + args.t / args.vx_size) ** 2
        )  # create a mask for the particle shell
        shell_mask = cp.logical_or(shell_mask, mask)  # update the shell mask

        shell_xi = ((x - cp.float16(point[0])) + y * 0 + z * 0) * mask  # calculate the x-component of the shell orientation
        shell_yi = (x * 0 + (y - cp.float16(point[1])) + z * 0) * mask  # calculate the y-component of the shell orientation
        shell_zi = (x * 0 + y * 0 + (z - 15)) * mask  # calculate the z-component of the shell orientation

        shell_x += shell_xi * (shell_x == 0)  # update the x-component of the shell orientation
        shell_y += shell_yi * (shell_y == 0)  # update the y-component of the shell orientation
        shell_z += shell_zi * (shell_z == 0)  # update the z-component of the shell orientation

    shell_mask = (core_mask == False) & (shell_mask == True)
    shell_x *= shell_mask
    shell_y *= shell_mask
    shell_z *= shell_mask

    polymer_mask = (core_mask == False) & (shell_mask == False)

    shell_t = (shell_x**2 + shell_y**2 + shell_z**2) ** 0.5  # calculate the magnitude of the shell orientation

    shell_1 = (args.ra / args.vx_size) * shell_mask  # calculate the radial distance from the particle center to the shell

    shell_volume_fraction = 1 * shell_mask  # calculate the volume fraction of the shell
    shell_S = shell_mask * cp.float16(args.S_0) * (shell_1 / shell_t) ** cp.float16(args.decay_order)  # calculate the aligned fraction of the shell
    core_volume_fraction = 1 * core_mask  # calculate the volume fraction of the particle core
    polymer_volume_fraction = 1 * polymer_mask  # calculate the volume fraction of the polymer matrix

    shell_S = cp.nan_to_num(shell_S)

    shell_theta = cp.arctan2(((shell_x) ** 2 + (shell_y) ** 2) ** 0.5, shell_z)  # calculate the theta angle of the shell orientation
    shell_psi = cp.arctan2(shell_y, shell_x)  # calculate the psi angle of the shell orientation

    core_volume_fraction = cp.asnumpy(core_volume_fraction)
    shell_volume_fraction = cp.asnumpy(shell_volume_fraction)
    polymer_volume_fraction = cp.asnumpy(polymer_volume_fraction)
    shell_S = cp.asnumpy(shell_S)
    shell_theta = cp.asnumpy(shell_theta)
    shell_psi = cp.asnumpy(shell_psi)

    a_zeros = np.zeros(core_volume_fraction.shape, dtype=np.float32)

    mat1_poly = Material(
        materialID=1,
        Vfrac=(shell_volume_fraction).astype(np.float32),
        S=shell_S.astype(np.float32),
        theta=shell_theta.astype(np.float32),
        psi=shell_psi.astype(np.float32),
        NumZYX=core_volume_fraction.shape,
        energies=args.energies,
        opt_constants=args.oc_polymer.opt_constants,
        name="PS",
    )

    mat2_particle = Material(
        materialID=2,
        Vfrac=core_volume_fraction.astype(np.float32),
        S=a_zeros,
        theta=a_zeros,
        psi=a_zeros,
        NumZYX=core_volume_fraction.shape,
        energies=args.energies,
        opt_constants=args.oc_particle.opt_constants,
        name="particle",
    )

    mat3_poly = Material(
        materialID=3,
        Vfrac=(polymer_volume_fraction).astype(np.float32),
        S=a_zeros.astype(np.float32),
        theta=a_zeros.astype(np.float32),
        psi=a_zeros.astype(np.float32),
        NumZYX=a_zeros.shape,
        energies=args.energies,
        opt_constants=args.oc_polymer.opt_constants,
        name="PS",
    )

    morph = Morphology(
        3,
        {1: mat1_poly, 2: mat2_particle, 3: mat3_poly},
        PhysSize=args.vx_size,
    )

    del (
        core_mask,
        shell_mask,
        polymer_mask,
        shell_x,
        shell_y,
        shell_z,
        shell_xi,
        shell_yi,
        shell_zi,
        core_volume_fraction,
        shell_volume_fraction,
        polymer_volume_fraction,
        shell_S,
        shell_theta,
        shell_psi,
        a_zeros,
        mat1_poly,
        mat2_particle,
        mat3_poly,
    )
    # a cupy helper function to ensure that unused memory is freed
    cp._default_memory_pool.free_all_blocks()
    return morph

def coreshell_spheres_run(args):

    # notice how we're passing args to the adsorbed_polymer_morphology function
    # we could also have args that affect how the model is run (but we don't right now)

    # we start right away with the adsorbed_polymer_morphology function
    # Create morphology object
    morph = coreshell_spheres_decay_euler(args)
    
    #this creates a single-particle model to aid visualization
    args_mdl = copy(args) # args.copy()
    args_mdl.coord_list = [[args.r/2, args.r/2]]
    
    morph_mdl = coreshell_spheres_decay_euler(args_mdl)    

    # FFT "window" parameter
    morph.inputData.windowingType = 0 #cy.FFTWindowing.Hanning

    morph.EAngleRotation = [0.0, 2.0, 360.0]

    # we will keep the validator in here. Always good to check the morphology your function made
    morph.validate_all(quiet=False)

    # we will use a dataclass for holding the results
    results = Results()

    # this actually runs the model
    data = morph.run(stdout=True, stderr=False)

    if args.return_2d:    
        results.twod = data.copy() # note that this gets deleted, so copy is appropriate

    # the PyHyperScattering part for interpreting the model result
    integrator = WPIntegrator(use_chunked_processing=False)
    remeshed_data = integrator.integrateImageStack(data)


        
    results.I = remeshed_data.mean(dim="chi")
    results.A = remeshed_data.rsoxs.AR(chi_width=5)

    # note, in addition to the model output, you could also return other things like the visualization results (which you might want to see, for parameter sweeps)
    # morphology.visualize_materials has a huge number of potential arguments to tailor what it shows about the model and how it shows it

    results.viz = Visualization(outputmat = [1], outputplot=['vfrac', 'S', 'psi', 'theta'])
    results.viz.get_visualizations(
        morph_mdl,
        subsample = 32,
        z_slice= 16,
        outputmat = [1],
        outputplot=['vfrac', 'S', 'psi', 'theta'],
        S_range=[[0, 0.2]],
        runquiet=True,
        outputaxes = False,
        S_cmap = args.S_cmap,
    )

    # also, it's a great idea to store the args dictionary in the results
    # that creates a durable record of what the parameters were when you simulated
    results.args = args

    # you could choose to return a variety of information.
    # for example, you could return the data object and/or the remeshed_data object
    # if you wanted to see full 2D patterns
    # I'm choosing only I and A, which are reductions of the data, but ones focused on our inquiry
    # I'm also returning the visualizations created above

    # note we return the results dictionary, which has I, A, and visualizations stored in it
    del(data, args_mdl, morph, morph_mdl, integrator, remeshed_data)
    cp._default_memory_pool.free_all_blocks()
    return results

def run_sweep(args, swept_arg, swept_range):
    """
    Runs the model for a range of values for a single argument and saves the results to pickle files.

    Parameters:
        args (dict): A dictionary containing the arguments for running the model.
        swept_arg (str): The name of the argument to sweep. Choose from "t", "S_0", and "decay_order" 
        swept_range (list): A list of values to sweep over.
    """
    # loop through the range of values
    for i, arg_value in enumerate(swept_range):
        # set the value of the argument to the current value in the range
        setattr(args, swept_arg, arg_value)
        # run the model with the current argument
        results = coreshell_spheres_run(args)
        # create a filename for the pickle file
        filename = f"{swept_arg}_{i:03d}.pkl"
        # create a path to the pickle directory
        pickle_path = Path.cwd() / "pickles"
        # create the full path and name of the pickle file
        savefile_path_and_name = pickle_path / filename
        # save the results to the pickle file
        with lzma.open(savefile_path_and_name, "wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

def import_pickles(picklePath, sstring):
    # below is a "docstring" and it is used to describe the function
    """
    Imports pickle files from a specified directory that match a given string.

    Parameters:
        picklePath (str): The path to the directory containing the pickle files.
        sstring (str): The string that the file names must contain.

    Returns:
        list: A list of objects representing the loaded pickle data.
    """
    # initialize an empty list to store the results
    results = []
    # use glob to find all the files in the directory that match the string
    # note the wildcard "*" in the glob function
    file_list = glob.glob(f"{picklePath}/{sstring}*")
    # show us those filenames
    pprint(file_list)
    # sort them - this will put them in the order they were created
    file_list.sort()
    # loop through the files. Notice how tqdm works - it actually is a function around the list that is iterated through
    for file_name in tqdm(file_list, colour = "blue"):
        # open the file using lzma
        with lzma.open(file_name, "rb") as pickle_file:
            # load the pickle file and append it to the results list
            results.append(pickle.load(pickle_file))
    # return the whole results list
    return results


def plot_sweep(swept_arg):
    """
    Plots the results of a sweep of the argument `swept_arg`.

    Parameters:
        swept_arg (str): The name of the argument to sweep. Choose from "t", "S_0", and "decay_order" The results should previously have been saved to pickle files using the `run_sweep` function.

    Returns:
        None
    """
    results = import_pickles(Path.cwd() / "pickles", swept_arg)
    num_viz = len(results)
    viz_col = np.ceil(num_viz / 3).astype(int)

    # font = {
    #     "family": "sans-serif",
    #     "sans-serif": "Arial",
    #     "weight": "regular",
    #     "size": 8,
    # }

    # rc("font", **font)

    # fig, _ = plt.subplots()
    fig, _ = plt.subplots(figsize=(7,8), dpi = 100)

    gs = gridspec.GridSpec(
        nrows=2,
        ncols=1,
        figure=fig,
        width_ratios=[1],
        height_ratios=[0.6, 0.4],
        wspace=0.1,
        hspace=0.15,
    )
    ax1 = plt.subplot(gs[0:1])

    gs_lower =  gridspec.GridSpecFromSubplotSpec(
        nrows=1,
        ncols=3,
        subplot_spec=gs[1:2],
        width_ratios=[0.9, 0.02, 0.08],
        wspace=0,
        hspace=0,
    )

    ax_cbar = plt.subplot(gs_lower[1:2])
    
    ax_backdrop = plt.subplot(gs_lower[0:1])
    ax_backdrop.set_facecolor("black")
    ax_backdrop.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    
    gs_viz =  gridspec.GridSpecFromSubplotSpec(
        nrows=3,
        ncols=viz_col,
        subplot_spec=gs_lower[0:1],
        wspace=0,
        hspace=0,
    )
    # axs = plt.subplots(1,18, gs[1:2])

    norm = Normalize(vmin=0, vmax=results[0].args.S_range_max)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=results[0].args.S_cmap),
                cax=ax_cbar, orientation='vertical', label='S', pad = 0.3)
    cbar.set_label("S", rotation=270, labelpad = 10)
    # fig, ax = plt.subplots(figsize=(7,5), dpi = 600)
    trace_cmap = 'viridis'
    num_traces = len(results) -1
    for i, result in enumerate(results):
        if swept_arg == "decay_order":
            # format the decay order as a float with 2 decimal places
            swept_arg_format = ".2f"
            swept_arg_unit = ""
            unformatted_swept_arg_value = result.args.__dict__[swept_arg]
            formatted_swept_arg_value = f"{unformatted_swept_arg_value:{swept_arg_format}}"
            label_string = f"n = {formatted_swept_arg_value}"
            ymin, ymax = -0.5, 0.15
            plt_title = f"sweeping decay order n"
        elif swept_arg == "S_0":
            # format the S_0 as a float with 2 decimal places
            swept_arg_format = ".2f"
            swept_arg_unit = ""
            unformatted_swept_arg_value = result.args.__dict__[swept_arg]
            formatted_swept_arg_value = f"{unformatted_swept_arg_value:{swept_arg_format}}"
            label_string = f"$S_0$ = {formatted_swept_arg_value}"
            ymin, ymax = -0.4, 0.15
            plt_title = f"sweeping orientation near particle $S_0$"
        else:
            # format the thickness as a float with 1 decimal place
            swept_arg_format = ".1f"
            swept_arg_unit = "nm"
            unformatted_swept_arg_value = result.args.__dict__[swept_arg]
            formatted_swept_arg_value = f"{unformatted_swept_arg_value:{swept_arg_format}}"
            label_string = f"{swept_arg} = {formatted_swept_arg_value} {swept_arg_unit}"
            ymin, ymax = -0.45, 0.15
            plt_title = f"sweeping thickness t (nm) of oriented corona"
        result.A.sel(energy=284.7, method = 'nearest').plot(ax = ax1, lw = 1, color = colormaps[trace_cmap](i/num_traces),label = label_string)
        result.A.sel(energy=285.2, method = 'nearest').plot(ax = ax1, lw = 1, color = colormaps[trace_cmap](i/num_traces))
        ax = fig.add_subplot(gs_viz[i // viz_col, i % viz_col: i % viz_col+ 1])
        ax.imshow(result.viz.img_dict["mat_1_S"], origin="lower")
        ax.set_axis_off()
        ax.text(0.5, 0.1, f"{formatted_swept_arg_value} {swept_arg_unit}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color = "white")
    txt = ax1.annotate("284.7 eV", (0.18, 0.05),xycoords = "data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color = "black")
    txt.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='none'))
    txt = ax1.annotate("285.2 eV", (0.18, -0.05), xycoords = "data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color = "black")
    txt.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax1.tick_params(axis = 'y',which = 'major',  direction = 'in')
    ax1.tick_params(axis = 'y',which = 'minor',  direction = 'in')
    ax1.tick_params(axis = 'x',which = 'major',  direction = 'in')
    ax1.tick_params(axis = 'x',which = 'minor',  direction = 'in')
    
    ax1.set_title(plt_title)
    ax1.hlines(0,0.05,0.4,linestyle = 'dotted',color = 'black', zorder = -100, lw = 1)
    ax1.set_xlim(0.05,0.4)
    ax1.set_ylim(ymin,ymax)
    ax1.xaxis.set_major_locator(ticker.AutoLocator())
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.yaxis.set_major_locator(ticker.AutoLocator())
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # ax1.plot(expt[0],-expt[1],lw = 1, color = 'salmon',linestyle = 'dotted',label = '284.7 eV expt')
    # ax1.plot(expt[0],-expt[2],lw = 1, color = 'dodgerblue',linestyle = 'dotted',label = '285.2 eV expt')
    # ax1.set_title("")
    ax1.set_ylabel("Anisotropy (a.u.)")
    ax1.set_xlabel("q (nm$^{-1})$")
    ax1.legend(loc = "center left")
    plt.savefig(f"{swept_arg}_sweep.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
