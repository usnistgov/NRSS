from enum import Enum
import numpy as np
from NRSS.morphology import Material, Morphology
from PyHyperScattering.integrate import WPIntegrator
import pickle, lzma
from pathlib import Path
import glob
from tqdm import tqdm
from pprint import pprint
from matplotlib import colormaps, gridspec, rc
from matplotlib import pyplot as plt

# FROM NOTEBOOK 2

# class syntax
class EulerStyle(Enum):
    """
    a class for qualitative model changes to Euler angle style relative to particle
    """
    RADIAL = 1
    """
    creates extraordindary index orientation that is radial to the core particle center
    """
    TANGENTIAL_LAT = 2
    """
    creates extraordindary index orientation that is tangential to the core particle center and latitudinal such that all orientations are parallel to the XY plane
    """
    TANGENTIAL_LONG = 3
    """
    creates extraordindary index orientation that is tangential to the core particle center and longitudinal such that all orientations are not parallel to the XY plane except at the north and south poles
    """
def adsorbed_polymer_morphology(args):
    """
    Generates a morphology object for an adsorbed polymer on a sphere.

    Args:
        args (dict): A dictionary containing the following keys:
            - "vd" (int): The vertical dimension of the grid.
            - "ld" (int): The lateral dimension of the grid.
            - "radius_nm" (float): The radius of the sphere in nanometers.
            - "PhysSize_nm_per_voxel" (float): The physical size of each voxel in nanometers.
            - "euler_style" (str): The style of Euler angles to use. From an enumerated type: EulerStyle.RADIAL, EulerStyle.TANGENTIAL_LAT, or EulerStyle.TANGENTIAL_LONG.
            - "S0" (float): The initial value of the S-field next to the nanoparticle.
            - "S_slope_per_nm" (float): The slope of the S-field in nanometers.
            - "energies" (list): A list of energies for the model.
            - "oc_polymer" (object): The opt_constants object for the polymer.
            - "oc_particle" (object): The opt_constants object for the particle.

    Returns:
        morph (Morphology): A Morphology object representing the adsorbed polymer on a sphere.
    """
    # make grid
    z, y, x = np.ogrid[0 : args["vd"], 0 : args["ld"], 0 : args["ld"]]
    # distance from center
    dist_from_ctr = np.sqrt(
        (x - x.mean()) ** 2
        + (y - y.mean()) ** 2
        + (z - z.mean()) ** 2
    )
    # boolean sphere
    v_sphere = dist_from_ctr < args["radius_nm"] / args["PhysSize_nm_per_voxel"]

    # Euler angle math
    delta_x = x - args["ld"] / 2
    delta_y = y - args["ld"] / 2
    delta_z = z - args["vd"] / 2

    # the psi and theta for the radial case
    psi = np.arctan2(delta_y, delta_x) + 0 * delta_z
    theta = np.arctan2(np.sqrt(delta_x**2 + delta_y**2), delta_z)

    # match - case is similar to a series of if - elseif statements
    match args["euler_style"]:
        case EulerStyle.RADIAL:
            pass
            # do nothing because psi and theta are already correct for radial
        case EulerStyle.TANGENTIAL_LAT:
            # modify the psi and theta for tangential_lat
            theta = np.full_like(theta, np.pi / 2)
            psi = psi - np.pi / 2
        case EulerStyle.TANGENTIAL_LONG:
            # modify the psi and theta for tangential_lon
            psi = psi
            theta = np.where(theta < np.pi / 2, theta + np.pi / 2, theta - np.pi / 2)
        case _:
            # raise a value error if this doesn't come in as one of the enumerated types
            raise ValueError("invalid euler style")
        
    # S-field math
    S_field = (
        args["S0"]
        + (dist_from_ctr - args["radius_nm"] / args["PhysSize_nm_per_voxel"])
        * args["S_slope_per_nm"]
    )
    S_field = np.maximum(0, S_field)
    S_field = np.minimum(args["S0"], S_field)
    S_field *= v_sphere == False

    # constructing the model object
    a_zeros = np.zeros(v_sphere.shape, dtype=np.float32)

    mat1_poly = Material(
        materialID=1,
        Vfrac=(1 - v_sphere).astype(np.float32),
        S=S_field.astype(np.float32),
        theta=theta.astype(np.float32),
        psi=psi.astype(np.float32),
        NumZYX=v_sphere.shape,
        energies=args["energies"],
        opt_constants=args["oc_polymer"].opt_constants,
        name="polymer",
    )

    mat2_particle = Material(
        materialID=2,
        Vfrac=v_sphere.astype(np.float32),
        S=a_zeros,
        theta=a_zeros,
        psi=a_zeros,
        NumZYX=v_sphere.shape,
        energies=args["energies"],
        opt_constants=args["oc_particle"].opt_constants,
        name="particle",
    )

    morph = Morphology(
        2,
        {1: mat1_poly, 2: mat2_particle},
        PhysSize=args["PhysSize_nm_per_voxel"],
    )
    # the return statement is what the function delivers as an output
    return morph

# FROM NOTEBOOK 2
def adsorbed_polymer_run(args):
    """
    Run the adsorbed polymer model.

    Args:
        args (dict): A dictionary containing the arguments for running the model.

    Returns:
        results (dict): A dictionary containing the results of the model run. The dictionary has the following keys:
            - "I" (numpy.ndarray): The mean of the remeshed data along the "chi" dimension.
            - "A" (numpy.ndarray): The result of the rsoxs.AR function applied to the remeshed data with a chi_width of 45.
            - "visualizations" (dict): A dictionary containing the visualizations created by the morphology.visualize_materials function.
            - "args" (dict): A copy of the input arguments.

    Description:
        This function runs the adsorbed polymer model using the provided arguments. It creates a morphology object using the adsorbed_polymer_morphology function and then runs the model using the morphology.run method. The resulting data is then remeshed using the WPIntegrator.integrateImageStack method. The mean of the remeshed data along the "chi" dimension is calculated and stored in the "I" key of the results dictionary. The result of the rsoxs.AR function applied to the remeshed data with a chi_width of 45 is stored in the "A" key of the results dictionary. Visualizations of the model are created using the morphology.visualize_materials function and stored in the "visualizations" key of the results dictionary. The input arguments are stored in the "args" key of the results dictionary. The results dictionary is then returned.
    """    
    # notice how we're passing args to the adsorbed_polymer_morphology function
    # we could also have args that affect how the model is run (but we don't right now)

    # we start right away with the adsorbed_polymer_morphology function
    # Create morphology object
    morph = adsorbed_polymer_morphology(args)

    # FFT "window" parameter
    morph.inputData.windowingType = 0  # cy.FFTWindowing.Hanning


    # that rotation function we don't need bc your model is radially symmetric
    morph.EAngleRotation = [0.0, 0.0, 0.0]

    # we will keep the validator in here. Always good to check the morphology your function made
    morph.validate_all(quiet=False)

    # this actually runs the model
    data = morph.run(stdout=True, stderr=False)

    # the PyHyperScattering part for interpreting the model result
    integrator = WPIntegrator(use_chunked_processing=False)
    remeshed_data = integrator.integrateImageStack(data)

    # we will use a pythin dictionary for holding the results
    results = {}

    results["I"] = remeshed_data.mean(dim="chi")
    results["A"] = remeshed_data.rsoxs.AR(chi_width=45)

    # note, in addition to the model output, you could also return other things like the visualization results (which you might want to see, for parameter sweeps)
    # morphology.visualize_materials has a huge number of potential arguments to tailor what it shows about the model and how it shows it
    results["visualizations"] = morph.visualize_materials(
        z_slice=args["vd"] // 2,
        subsample=256,
        outputplot=["vfrac", "S", "psi", "theta"],
        S_range=[[0, 1]],
        outputmat=[1],
        runquiet=True,
    )

    # also, it's a great idea to store the args dictionary in the results
    # that creates a durable record of what the parameters were when you simulated
    results["args"] = args

    # you could choose to return a variety of information.
    # for example, you could return the data object and/or the remeshed_data object
    # if you wanted to see full 2D patterns
    # I'm choosing only I and A, which are reductions of the data, but ones focused on our inquiry
    # I'm also returning the visualizations created above

    # note we return the results dictionary, which has I, A, and visualizations stored in it
    return results


# FROM NOTEBOOK 2
def adsorbed_polymer_run_save(args):
    """
    Saves the results of the adsorbed polymer run to a pickle file.

    Args:
        args (dict): A dictionary containing the arguments for running the model.

    Returns:
        None
    """
    # we jump straight to the results dictionary
    results = adsorbed_polymer_run(args)

    # now we save the results to a pickle file
    # note that we need a file name! We can use a new entry in the args dictionary
    # also we need  a directory to save the file in.
    # For convenience, I've hardcoded a "pickles" subdirectory in the current working directory

    # this syntax is using the pathlib module. Some older styles might just use strings for this
    # Path.cwd() returns the current working directory

    savefile_path_and_name = Path.cwd() / "pickles" / args["filename"]

    # lzma is a hardcore compression library we're using with pickle to keep the file size down
    with lzma.open(savefile_path_and_name, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # note there is nothing to return here. The results are stored in the pickled results dictionary

# FROM NOTEBOOK 3
def adsorbed_polymer_sweep(args, swept_arg, values, file_prefix = ""):
    """
    Runs the adsorbed polymer model for multiple values of a swept argument and saves the results to pickle files.

    Args:
        args (dict): A dictionary containing the arguments for running the model.
        swept_arg (str): The name of the argument to sweep.
        values (list): A list of values to sweep over.
        file_prefix (str, optional): A prefix to add to the filename. Defaults to an empty string.

    Returns:
        None
    """
        
    # note that the default value of the file_prefix is an empty string
    # this prefix is here just if you want to mark the filenames with a special marker like the date,
    # or other values you might have set outside this function like euler style

    # add a couple of values here to args including the swept args and values
    # this is mostly to save for after-simulation analysis
    args["swept_arg"] = swept_arg
    args["swept_values"] = values

    # enumerate provides two values, the index (0,1,2,3, etc) and the value.
    # we use that here so that the variable i can keep track of which value we're on
    # and we'll use that in the filename
    for i, value in enumerate(values):
        # set the swept argument in the args dictionary to the current value    
        args[swept_arg] = value
        
        # set the filename to the concatenation of the prefix, the swept argument, and the index
        # the index is zero padded to 2 digits so that the filename is always the same length
        args["filename"] = file_prefix + swept_arg + "_" + str(i).zfill(2) + ".pkl"
        
        #run the model and pickle the result
        adsorbed_polymer_run_save(args)
        
# FROM NOTEBOOK 5
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

#FROM NOTEBOOK 5

def plot_result(result):
    """
    Plots the result of a polymer adsorption simulation using matplotlib.

    Parameters:
        result (dict): A dictionary containing the following keys:
            - "I" (numpy.ndarray): The mean of the remeshed data along the "chi" dimension.
            - "A" (numpy.ndarray): The result of the rsoxs.AR function applied to the remeshed data.
            - "visualizations" (dict): A dictionary containing visualizations of the simulation result.
            - "args" (dict): A copy of the arguments used to run the simulation.

    Returns:
        numpy.ndarray: A 3D RGB array representing the plot.

    Example usage:
        >>> result = adsorbed_polymer_run(args)
        >>> rgb_array = plot_result(result)
    """
    rc('font', size=14)

    dpi = 72
    h_size = 1920 / dpi
    v_size = 1080 / dpi

    fig = plt.subplots(figsize=(h_size, v_size), dpi=dpi)

    gs = gridspec.GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.2,
        hspace=0.2,
    )

    ax1 = plt.subplot(gs[0:1, 0:1])
    ax2 = plt.subplot(gs[0:1, 1:2])
    ax3 = plt.subplot(gs[1:2, 0:1])
    ax4 = plt.subplot(gs[1:2, 1:2])

    qmax = 1
    
    result["I"].sel(energy = 284.7, method = 'nearest').plot(ax = ax1, yscale = "log", xlim = (0.05, 1.4))

    ax1.set_title(f"Intensity vs. q")
    ax1.set_ylim(top = 1E8, bottom = 1)
    ax1.set_xlim(left = 0, right = qmax)
    ax1.set_xlabel("q ($nm^{-1}$)")
    ax1.set_ylabel("I (a.u.) at 284.7 eV")
    
    annotation = f"radius_nm = " + "{:0.1f}".format(result['args']["radius_nm"]) + " nm\n"
    annotation += f"S0 = " + "{:0.3f}".format(result['args']["S0"]) + "\n"
    annotation += f"S_slope_per_nm = " + "{:0.3f}".format(result['args']["S_slope_per_nm"])
    ax1.annotate(annotation, xy = (0.7, 0.8), xycoords = "axes fraction")
    
    ax2.imshow(result["visualizations"][1])
    ax2.axis("off")

    ax3.axhline(y = 0, color = "black", linestyle = "dotted") 
    result["A"].sel(energy = 284.7, method = 'nearest').plot(ax = ax3, ylim = (-1, 1), color = "green", label = "284.7 eV")
    result["A"].sel(energy = 285.2, method = 'nearest').plot(ax = ax3, ylim = (-1, 1), color = "magenta", label = "285.2 eV")
    ax3.set_xlim(left = 0, right = qmax)
    ax3.set_title(f"Radial simulation result, A")
    ax3.set_xlabel("q ($nm^{-1}$)")
    ax3.set_ylabel("A")
    ax3.legend()

    result["A"].plot(xlim=(0,1.0), vmin = -1, vmax = 1, cmap = 'seismic', ax = ax4)
    ax4.set_xlim(left = 0, right = qmax)
    ax4.set_title(f"Radial simulation result, A")
    ax4.set_xlabel("q ($nm^{-1}$)")
    ax4.set_ylabel("energy (eV)")

    plt.subplots_adjust(left=0.05, bottom=0.10, right=0.95, top=0.95, wspace=0, hspace=0)
    fig[0].canvas.draw()
    width, height = fig[0].canvas.get_width_height()
    data = np.frombuffer(fig[0].canvas.tostring_rgb(), dtype=np.uint8)
    rgb_array = data.reshape(height, width, 3)
    return rgb_array

import subprocess, shlex

#FROM NOTEBOOK 5
def encode_video_loop(
    frames, output_file, frame_rate=3, still_frames=5, loops=1
):
    """
    Encode a video from a sequence of frames, with a specified number
    of loops and still frames in between each loop.

    Parameters
    ----------
    frames : list of numpy arrays
        Frames to be encoded into the video
    output_file : str
        Path to the output video file
    frame_rate : int, optional
        Frame rate of the output video. Defaults to 3.
    still_frames : int, optional
        Number of still frames to write in between each loop. Defaults to 5.
    loops : int, optional
        Number of loops to write. Defaults to 1.
    """
    width, height = frames[0].shape[1], frames[0].shape[0]
    cmd = f'ffmpeg -y -s {width}x{height} -pixel_format rgb24 -f rawvideo -r {frame_rate} -i pipe: -vcodec libx264 -pix_fmt yuv420p -crf 15 {output_file}'
    process = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
    
    for _ in range(loops):
        for frame in frames:
            process.stdin.write(frame.tobytes())
        for _ in range(still_frames):
            process.stdin.write(frames[-1].tobytes())
        for frame in reversed(frames):
            process.stdin.write(frame.tobytes())
        for _ in range(still_frames):
            process.stdin.write(frames[0].tobytes())
    
    process.stdin.close()
    process.wait()
    process.terminate()