import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc, gridspec
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import h5py
import datetime
import warnings


def check_NumMat(f, morphology_type):
    """
    Checks and validates the number of materials in the HDF5 file

    Parameters
    ----------

    f : file object
    morphology_type : int
        0 - Euler, 1 - Vector

    Returns
    -------

    num_mat : int
        Number of materials in file

    """
    morphology_num = f['Morphology_Parameters/NumMaterial'][()]

    if morphology_type == 0:
        num_mat = 0
        while f'Euler_Angles/Mat_{num_mat + 1}_Vfrac' in f.keys():
            num_mat += 1
    elif morphology_type == 1:
        num_mat = 0
        while f'Vector_Morphology/Mat_{num_mat + 1}_unaligned' in f.keys():
            num_mat += 1

    assert morphology_num == num_mat, 'Number of materials does not match manual count of materials. Recheck hdf5'

    return num_mat


def readH5_vector(filename):
    """
    Reads in Vector Morphology from HDF5 file and converts to Euler Morphology. Checks that total material volume fractions sum to 1 for all voxels.

    Parameters
    ----------

    filename : str or path

    Returns
    -------
        Vfrac : ndarray
            Volume fraction of all materials. [Material, Z, Y, X]
        S : ndarray
            Fraction of alignment for each material. [Material, Z, Y, X]
        theta : ndarray
            Second Euler angle (ZYZ convention). [Material, Z, Y, X]
        psi : ndarray
            Third Euler angle (ZYZ convention). [Material, Z, Y, X]

    """
    with h5py.File(filename, 'r') as f:
        num_mat = check_NumMat(f, morphology_type=1)

        ds = f['Vector_Morphology/Mat_1_unaligned'][()]

        unaligned = np.zeros((num_mat, *ds.shape))
        alignment = np.zeros((num_mat, *ds.shape, 3))

        for i in range(0, num_mat):
            unaligned[i, ...] = f[f'Vector_Morphology/Mat_{i+1}_unaligned'][()]
            alignment[i, ...] = f[f'Vector_Morphology/Mat_{i+1}_alignment'][()]

        # calculate total material
        total_material = np.sum(unaligned, axis=0) + np.sum(alignment**2, axis=(0, -1))

    # assert that the entire morphology has total material equal to 1
    assert np.allclose(total_material, 1), 'Not all voxels in morphology have Total Material equal to 1'

    # convert vector to Euler for visualization purposes
    S = np.zeros((num_mat, *ds.shape))
    Vfrac = S.copy()
    theta = S.copy()
    psi = S.copy()

    s2 = np.sum(alignment**2, axis=-1)
    Vfrac = unaligned + s2
    S = s2/Vfrac
    np.nan_to_num(S, copy=False)
    S = np.round(S, 6)
    # calculate theta and psi from vectors
    with np.errstate(invalid='ignore'):
        normed_vectors = alignment/np.sqrt(s2[..., np.newaxis])
    np.nan_to_num(normed_vectors, copy=False)
    theta = np.arccos(normed_vectors[..., 2])
    psi = np.arctan2(normed_vectors[..., 1], normed_vectors[..., 0])

    return Vfrac, S, theta, psi


def readH5_euler(filename):
    """
    Reads in Euler Morphology from HDF5 file. Checks that total material volume fractions sum to 1 for all voxels.

    Parameters
    ----------

    filename : str or path

    Returns
    -------
        Vfrac : ndarray
            Volume fraction of all materials. [Material, Z, Y, X]
        S : ndarray
            Fraction of alignment for each material. [Material, Z, Y, X]
        theta : ndarray
            Second Euler angle (ZYZ convention). [Material, Z, Y, X]
        psi : ndarray
            Third Euler angle (ZYZ convention). [Material, Z, Y, X]

    """
    with h5py.File(filename, 'r') as f:
        num_mat = check_NumMat(f, morphology_type=0)

        ds = f['Euler_Angles/Mat_1_Vfrac'][()]

        Vfrac = np.zeros((num_mat, *ds.shape))
        S = Vfrac.copy()
        theta = Vfrac.copy()
        psi = Vfrac.copy()

        # 'Mat_1_Psi', 'Mat_1_S', 'Mat_1_Theta', 'Mat_1_Vfrac'

        for i in range(0, num_mat):
            Vfrac[i, :, :, :] = f[f'Euler_Angles/Mat_{i+1}_Vfrac']
            S[i, :, :, :] = f[f'Euler_Angles/Mat_{i+1}_S']
            theta[i, :, :, :] = f[f'Euler_Angles/Mat_{i+1}_Theta']
            psi[i, :, :, :] = f[f'Euler_Angles/Mat_{i+1}_Psi']
            psi = np.mod(psi, np.pi)
            # calculate total material
            total_material = np.sum(Vfrac, axis=0)

    # assert that the entire morphology has total material equal to 1
    assert np.allclose(total_material, 1), 'Not all voxels in morphology have Total Material equal to 1' 

    return Vfrac, S, theta, psi


def checkH5(filename='perp82.hd5', z_slice=0, subsample=None, outputmat=None, runquiet=False, plotstyle='light'):
    """
    Reads in morphology HDF5 file and checks that the format is consistent for CyRSoXS. Optionally plots and returns select quantities.

    Parameters
    ----------

        filename : str or path
            Name of HDF5 morphology file to check
        z_slice : int
            Which z-slice of the array to plot.
        subsample : int
            Number of voxels to display in X and Y
        outputmat : int
            Number of which material to return
        runquiet : bool
            Boolean flag for running without plotting or outputting to console
        plotstyle : str
            Use a light or dark background for plots. 'dark' - dark, 'light' - light

    Returns
    -------
        Vfrac : ndarray
            Volume fraction of selected material. [ Z, Y, X]
        S : ndarray
            Fraction of alignment for selected material. [Z, Y, X]
        theta : ndarray
            Second Euler angle (ZYZ convention). [Z, Y, X]
        psi : ndarray
            Third Euler angle (ZYZ convention). [Z, Y, X]
    """
    style_dict = {'dark': 'dark_background',
                  'light': 'default'}

    begin_time = datetime.datetime.now()

    with h5py.File(filename, 'r') as f:
        # check morphology type
        if 'Euler_Angles' in f.keys():
            morphology_type = 0
        elif 'Vector_Morphology' in f.keys():
            morphology_type = 1
        else:
            raise KeyError('Neither \"Euler_Angles\" or \"Vector_Morphology\" group detected in hdf5')
            # morphology_type = None

    if morphology_type == 0:
        Vfrac, S, theta, psi = readH5_euler(filename)
        num_mat, zdim, ydim, xdim = Vfrac.shape
    elif morphology_type == 1:
        Vfrac, S, theta, psi = readH5_vector(filename)
        num_mat, zdim, ydim, xdim = Vfrac.shape

    # check to make sure arrays are float
    improper_types = []
    for array, array_name in zip([Vfrac, S, theta, psi], ['Vfrac', 'S', 'theta', 'psi']):
        if 'float' not in array.dtype.name:
            warnings.warn(f'{array_name} dtype is {array.dtype.name} and incompatible with CyRSoXS. Array dtype must be some variant of float')
            improper_types.append([array_name, array.dtype.name])

    # raise exception after going through all material arrays so user can fix all of them at once
    if len(improper_types) != 0:
        for val in improper_types:
            print(val[0], val[1])  
        raise TypeError('One or more of the material arrays dtype is not float. See above for details')

    if subsample is None:
        subsample = ydim # y dimension

    if not runquiet:
        print(f'Dataset dimensions (Z, Y, X): {zdim} x {ydim} x {xdim}')
        print(f'Number of Materials: {num_mat}')
        print('')

    if z_slice > (zdim-1):
        print(f'Error: z_slice of {z_slice} is greater than the maximum index of {zdim-1}. Using z_slice = {zdim-1} instead.')
        z_slice = zdim-1

    if not runquiet:

#         if plotstyle == 'dark':
        plt.style.use(style_dict[plotstyle])
        font = {'family': 'sans-serif',
                'sans-serif': 'DejaVu Sans',
                'weight': 'regular',
                'size': 18}

        rc('font', **font)

        for i in range(0,num_mat):    
            print(f'Material {i+1} Vfrac. Min: {np.amin(Vfrac[i,:,:,:])} Max: {np.amax(Vfrac[i,:,:,:])}')
            print(f'Material {i+1} S. Min: {np.amin(S[i,:,:,:])} Max: {np.amax(S[i,:,:,:])}')
            print(f'Material {i+1} theta. Min: {np.amin(theta[i,:,:,:])} Max: {np.amax(theta[i,:,:,:])}')
            print(f'Material {i+1} psi. Min: {np.amin(psi[i,:,:,:])} Max: {np.amax(psi[i,:,:,:])}')

            fig = plt.figure(figsize=(22, 33))

            gs = gridspec.GridSpec(nrows=5,
                                   ncols=2,
                                   figure=fig,
                                   width_ratios=[1, 1],
                                   height_ratios=[3, 1, 0.1, 3, 1],
                                   wspace=0.3,
                                   hspace=0.2)

            start = int(ydim/2)-int(subsample/2)
            end = int(ydim/2)+int(subsample/2)

            ax1 = plt.subplot(gs[0, 0])
            Vfracplot = ax1.imshow(Vfrac[i, z_slice, start:end, start:end], cmap=plt.get_cmap('winter'), origin='lower', interpolation='none')
            ax1.set_ylabel('Y index')
            ax1.set_xlabel('X index')

            ax1.set_title(f'Material {i+1} Vfrac')
            Vfrac_cbar = plt.colorbar(Vfracplot, fraction=0.040)
            Vfrac_cbar.set_label('Vfrac: volume fraction', rotation=270, labelpad=22)

            ax2 = plt.subplot(gs[0, 1])
            Splot = ax2.imshow(S[i, z_slice, start:end, start:end], cmap=plt.get_cmap('nipy_spectral'), origin='lower', interpolation='none')
            ax2.set_ylabel('Y index')
            ax2.set_xlabel('X index')
            ax2.set_title(f'Material {i+1} S')
            S_cbar = plt.colorbar(Splot, fraction=0.040)
            S_cbar.set_label('S: orientational order parameter', rotation=270, labelpad=22)

            ax3 = plt.subplot(gs[1, 0])
            ax3.hist(Vfrac[i, :, :, :].flatten())
            ax3.set_title(f'Material {i+1} Vfrac')
            ax3.set_xlim(left=0)
            ax3.set_xlabel('Vfrac: volume fraction')
            ax3.set_ylabel('num voxels')
            ax3.set_yscale('log')

            ax4 = plt.subplot(gs[1, 1])
            ax4.hist(S[i, :, :, :].flatten())
            ax4.set_title(f'Material {i+1} S')
            ax4.set_xlim(left=0)
            ax4.set_xlabel('S: orientational order parameter')
            ax4.set_ylabel('num voxels')
            ax4.set_yscale('log')

            ax5 = plt.subplot(gs[3, 0])
            norm = matplotlib.colors.Normalize(vmin=0, vmax=np.pi, clip=False)
            thetaplot = ax5.imshow(np.ma.masked_array(theta[i, z_slice, start:end, start:end], np.logical_or(Vfrac[i, z_slice, start:end, start:end] < 0.01, S[i, z_slice, start:end, start:end] < 0.01)),
                                   cmap=plt.get_cmap('jet'), norm=norm, origin='lower', interpolation='none')
            ax5.set_ylabel('Y index')
            ax5.set_xlabel('X index')            
            ax5.set_title(f'Material {i+1} theta')
            theta_cbar = plt.colorbar(thetaplot, fraction=0.040)
            theta_cbar.set_label('theta in radians', rotation=270, labelpad=22)

            ax6 = plt.subplot(gs[3, 1])
            norm = matplotlib.colors.Normalize(vmin=0, vmax=np.pi, clip=False)
            psiplot = ax6.imshow(np.ma.masked_array(psi[i, z_slice, start:end, start:end], np.logical_or(Vfrac[i, z_slice, start:end, start:end] < 0.01, S[i, z_slice, start:end, start:end] < 0.01)),
                                 cmap=plt.get_cmap('hsv'), norm=norm, origin='lower', interpolation='none')
            ax6.set_ylabel('Y index')
            ax6.set_xlabel('X index')            
            ax6.set_title(f'Material {i+1} psi')
            psi_cbar = plt.colorbar(psiplot, fraction=0.040)
            psi_cbar.set_label('psi in radians', labelpad=22)       

            ax7 = plt.subplot(gs[4, 0])
            ax7.hist(theta[i, :, :, :].flatten())
            ax7.set_title(f'Material {i+1} theta')
            ax7.set_xlim(left=0)
            ax7.set_xlabel('theta in radians')
            ax7.set_ylabel('num voxels')
            ax7.set_yscale('log')

            ax8 = plt.subplot(gs[4, 1])
            ax8.hist(psi[i, :, :, :].flatten())
            ax8.set_title(f'Material {i+1} psi')
            ax8.set_xlim(left=0)
            ax8.set_xlabel('psi in radians')
            ax8.set_ylabel('num voxels')
            ax8.set_yscale('log')

            ax4i = inset_axes(ax6,  axes_class=matplotlib.projections.get_projection_class('polar'), width=2, height=2, axes_kwargs={"alpha": 0})

            ax4i.grid(False)
            # creates inset legend for orientation
            azimuths = np.deg2rad(np.arange(0, 360, 1))
            zeniths = np.linspace(5, 10, 50)
            values = np.mod(azimuths, np.pi) * np.ones((50, 360))
            ax4i.plot(0, 0, 'o', ms=75, mec='none', mfc=plt.rcParams['axes.facecolor'], mew=2, zorder=0, alpha=0.7)
            ax4i.pcolormesh(azimuths, zeniths, values, cmap=cm.hsv, shading='auto')
            ax4i.set_axis_off()
            ax4i.arrow(0, 0, 0, 4, width=0.015, head_width=0.25, head_length=0.5)  #,facecolor='k',edgecolor='k')
            ax4i.arrow(np.pi/2, 0, 0, 4, width=0.015, head_width=0.25, head_length=0.5)  #,facecolor='k',edgecolor='k')
            ax4i.text(2*np.pi-np.deg2rad(40), 3.5, 'X')
            ax4i.text(np.pi/2+np.deg2rad(40), 3.5, 'Y')

            ax5i = inset_axes(ax5,  axes_class=matplotlib.projections.get_projection_class('polar'), width=2, height=2, axes_kwargs={"alpha": 0})

            ax5i.grid(False)
            # creates inset legend for orientation
            azimuths_t = np.deg2rad(np.arange(-90, 90, 1))
            zeniths_t = np.linspace(5, 10, 50)
            values_t = np.mod(np.pi/2-azimuths_t, np.pi) * np.ones((50, 180))
            ax5i.plot(0, 0, 'o', ms=75, mec='none', mfc=plt.rcParams['axes.facecolor'], mew=2, zorder=0, alpha=0.7)
            ax5i.pcolormesh(azimuths_t, zeniths_t, values_t, cmap=cm.jet, shading='auto')
            ax5i.set_axis_off()
            ax5i.arrow(0, 0, 0, 4, width=0.015, head_width=0.25, head_length=0.5)  #,facecolor='k',edgecolor='k')
            ax5i.arrow(np.pi/2, 0, 0, 4, width=0.015, head_width=0.25, head_length=0.5)  #, facecolor='k',edgecolor='k')
            ax5i.text(2*np.pi-np.deg2rad(40), 3.5, 'X')
            ax5i.text(np.pi/2+np.deg2rad(40), 3.5, 'Z')

            plt.show()

    plt.rcParams.update(plt.rcParamsDefault)

    if not runquiet:
        print(f'Total Vfrac whole model. Min: {np.amin(np.sum(Vfrac, axis=0))} Max: {np.amax(np.sum(Vfrac, axis=0))}')
        print(datetime.datetime.now() - begin_time)
    if not (outputmat is None):
        start = int(ydim/2)-int(subsample/2)
        end = int(ydim/2)+int(subsample/2)
        return Vfrac[outputmat, z_slice, start:end, start:end], S[outputmat, z_slice, start:end, start:end], psi[outputmat, z_slice, start:end, start:end], theta[outputmat, z_slice, start:end, start:end]
