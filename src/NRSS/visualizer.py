import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc, gridspec
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import warnings
import pathlib


def morphology_visualizer(
    morphology,
    z_slice: int = 0,
    subsample: int = None,
    translate_x: int = None,
    translate_y: int = None,
    screen_euler: bool = True,
    add_quiver: bool = False,
    quiver_bw: bool = True,
    outputmat: list = None,
    outputplot: list = None,
    outputaxes: bool = True,
    vfrac_range: list = None,
    S_range: list = None,
    vfrac_cmap: str = None,
    S_cmap: str = None,
    runquiet: bool = False,
    batchMode: bool = False,
    plotstyle: str = "light",
    dpi: int = 300,
    exportDir: str = None,
    exportParams: dict = None,
):
    """
    Reads in morphology HDF5 file and checks that the format is consistent for CyRSoXS. Optionally plots and returns select quantities.

    Parameters
    ----------

        z_slice : int
            Which z-slice of the array to plot.
        subsample : int
            Number of voxels to display in X and Y
        translate_x : int
            Number of voxels to translate image in x; meant for use with subsample
        translate_y : int
            Number of voxels to translate image in y; meant for use with subsample
        screen_euler : bool
            Suppress visualization of euler angles where vfrac < 0.05 or S < 0.05; intended to hilight edges
        add_quiver : bool
            Adds lines to every voxel on the psi plot that indicate in-plane direction. Not recommended for resolutions larger than 128x128, best for resolutions 64x64 or lower.
        quiver_bw : bool
            Intended to be used when add_quiver == True, when quiver_bw is True, the quiver arrows will be black and white instead of colored.
        outputmat : list of ints
            Number of which materials to return
        outputplot : list of strings
            Number of which plots to return, can include 'vfrac', 'S', 'theta', 'psi'
        outputaxes : bool
            If a plot is returned, include its axes
        vfrac_range: list of tuples as [float, float]
            A custom range for vfrac colorbar
        S_range: list of tuples as [float, float]
            A custom range for S colorbar
        vfrac_cmap: str
            A custom substitution for vfrac colormap
        S_cmap: str
            A custom substitution for vfrac colormap
        runquiet : bool
            Boolean flag for running without plotting or outputting to console
        batchMode : bool
            if true, prints console output and generates plots but doesnt show (provide exportDir for export)
        plotstyle : str
            Use a light or dark background for plots. 'dark' - dark, 'light' - light
        dpi : int
            The dpi at which the plot is generated. Per-material plot dimensions are 8.5" x 12.75"
        exportDir : str, optional
            if provided, export directory to save any generated figures into,
            by default, will respect dpi and save as png, use exportParams to override
        exportParams : dict, optional
            additional params to unpack into matplotlib.pyplot.savefig. Overrides existing params.
            ex: exportParams = {'dpi':600, format='svg'}
    Returns
    -------
        If outputmat and outputplot are correctly entered, will return an index list of images of the selected material and plot. Each list element will be  a numpy array in RGB format that be displayed with imshow

    """
    style_dict = {"dark": "dark_background", "light": "default"}

    rgb_return_list = []

    if subsample is None:
        subsample = morphology.NumZYX[1]  # y dimension

    if not runquiet:
        print(
            f"Dataset dimensions (Z, Y, X): {morphology.NumZYX[0]} x {morphology.NumZYX[1]} x"
            f" {morphology.NumZYX[2]}"
        )
        print(f"Number of Materials: {morphology._numMaterial}")
        print("")

    if z_slice > (morphology.NumZYX[0] - 1):
        warnings.warn(
            f"z_slice of {z_slice} is greater than the maximum index of {morphology.NumZYX[0]-1}."
            f" Using z_slice = {morphology.NumZYX[0]-1} instead."
        )
        z_slice = morphology.NumZYX[0] - 1

    #         if plotstyle == 'dark':
    plt.style.use(style_dict[plotstyle])
    font = {
        "family": "sans-serif",
        "sans-serif": "DejaVu Sans",
        "weight": "regular",
        "size": 8,
    }

    rc("font", **font)

    cwdPath = pathlib.Path(__file__).resolve().parent
    psi_cmap = matplotlib.colors.ListedColormap(np.load(cwdPath / "cmap/infinitydouble_cmap.npy"))

    backend_ = matplotlib.get_backend()

    try:
        if runquiet:
            matplotlib.use("Agg")  # Prevent showing stuff

        for i in range(1, morphology._numMaterial + 1):
            fig = plt.figure(figsize=(8.5, 12.75), dpi=dpi)
            if runquiet == False:
                print(
                    f"Material {i} Vfrac. Min: {morphology.materials[i].Vfrac.min()} Max:"
                    f" {morphology.materials[i].Vfrac.max()}"
                )
                print(
                    f"Material {i} S. Min: {morphology.materials[i].S.min()} Max:"
                    f" {morphology.materials[i].S.max()}"
                )
                print(
                    f"Material {i} theta. Min: {morphology.materials[i].theta.min()} Max:"
                    f" {morphology.materials[i].theta.max()}"
                )
                print(
                    f"Material {i} psi. Min: {morphology.materials[i].psi.min()} Max:"
                    f" {morphology.materials[i].psi.max()}"
                )

            if (morphology.materials[i].theta.min() < 0) or (
                morphology.materials[i].theta.max() > (np.pi)
            ):
                warnings.warn(
                    "Visualization expects theta to have bounds of [0,pi]. This model has theta"
                    " outside those bounds and visualization may be incorrect."
                )

            # run if you don't want runquiet or run if you've selected this material for output
            if (runquiet is not True) or ((outputmat is not None) and (i in outputmat)):
                gs = gridspec.GridSpec(
                    nrows=5,
                    ncols=2,
                    figure=fig,
                    width_ratios=[1, 1],
                    height_ratios=[3, 1, 0.1, 3, 1],
                    wspace=0.3,
                    hspace=0.65,
                )

                start = int(morphology.NumZYX[1] / 2) - int(subsample / 2)
                end = int(morphology.NumZYX[1] / 2) + int(subsample / 2)

                if translate_x:
                    start_x = max(0, start + translate_x)
                    end_x = min(end + translate_x, morphology.NumZYX[1])
                else:
                    start_x = start
                    end_x = end

                if translate_y:
                    start_y = max(0, start + translate_y)
                    end_y = min(end + translate_y, morphology.NumZYX[1])
                else:
                    start_y = start
                    end_y = end

                if (runquiet is not True) or ("vfrac" in outputplot):
                    ax1 = plt.subplot(gs[0, 0])
                    if vfrac_range and (len(vfrac_range) >= i) and (len(vfrac_range[i - 1]) == 2):
                        norm = matplotlib.colors.Normalize(
                            vmin=vfrac_range[i - 1][0], vmax=vfrac_range[i - 1][1]
                        )
                    else:
                        norm = "linear"

                    if vfrac_cmap:
                        cmap = plt.get_cmap(vfrac_cmap)
                    else:
                        cmap = plt.get_cmap("winter")

                    Vfracplot = ax1.imshow(
                        morphology.materials[i].Vfrac[z_slice, :, :],
                        cmap=cmap,
                        origin="lower",
                        interpolation="none",
                        norm=norm,
                    )
                    ax1.set_ylabel("Y index", labelpad=0)
                    ax1.set_xlabel("X index")
                    ax1.set_title(f"Mat {i} {morphology.materials[i].name} Vfrac")
                    ax1.set_xlim(start_x, end_x)
                    ax1.set_ylim(start_y, end_y)
                    Vfrac_cbar = plt.colorbar(Vfracplot, ax=ax1, fraction=0.040)
                    # Vfrac_cbar.set_label(
                    #     "Vfrac: volume fraction", rotation=270, labelpad=22
                    # )

                if (runquiet is not True) or ("S" in outputplot):
                    ax2 = plt.subplot(gs[0, 1])
                    if S_range and (len(S_range) >= i) and (len(S_range[i - 1]) == 2):
                        norm = matplotlib.colors.Normalize(
                            vmin=S_range[i - 1][0], vmax=S_range[i - 1][1]
                        )
                    else:
                        norm = "linear"

                    if S_cmap:
                        cmap = plt.get_cmap(S_cmap)
                    else:
                        cmap = plt.get_cmap("nipy_spectral")

                    Splot = ax2.imshow(
                        morphology.materials[i].S[z_slice, :, :],
                        cmap=cmap,
                        origin="lower",
                        interpolation="none",
                        norm=norm,
                    )
                    ax2.set_ylabel("Y index", labelpad=0)
                    ax2.set_xlabel("X index")
                    ax2.set_title(f"Mat {i} {morphology.materials[i].name} S")
                    ax2.set_xlim(start_x, end_x)
                    ax2.set_ylim(start_y, end_y)
                    S_cbar = plt.colorbar(Splot, fraction=0.040)
                    # S_cbar.set_label(
                    #     "S: orientational order parameter", rotation=270, labelpad=22
                    # )

                # only do this if not runquiet; these plots are not outputted
                if runquiet is not True:
                    ax3 = plt.subplot(gs[1, 0])
                    ax3.hist(morphology.materials[i].Vfrac.flatten())
                    ax3.set_title(f"Mat {i} {morphology.materials[i].name} Vfrac")
                    ax3.set_xlim(left=0)
                    ax3.set_xlabel("Vfrac: volume fraction")
                    ax3.set_ylabel("num voxels")
                    ax3.set_yscale("log")

                    ax4 = plt.subplot(gs[1, 1])
                    ax4.hist(morphology.materials[i].S.flatten())
                    ax4.set_title(f"Mat {i} {morphology.materials[i].name} S")
                    ax4.set_xlim(left=0)
                    ax4.set_xlabel("S: orientational order parameter")
                    ax4.set_ylabel("num voxels")
                    ax4.set_yscale("log")

                if (runquiet is not True) or ("theta" in outputplot):
                    ax5 = plt.subplot(gs[3, 0])
                    norm = matplotlib.colors.Normalize(vmin=0, vmax=np.pi, clip=False)
                    if screen_euler:
                        thetaplot = ax5.imshow(
                            np.ma.masked_array(
                                morphology.materials[i].theta[z_slice, :, :] % np.pi,
                                np.logical_or(
                                    morphology.materials[i].Vfrac[z_slice, :, :] < 0.01,
                                    morphology.materials[i].S[z_slice, :, :] < 0.01,
                                ),
                            ),
                            cmap=plt.get_cmap("jet"),
                            norm=norm,
                            origin="lower",
                            interpolation="none",
                        )

                    else:
                        thetaplot = ax5.imshow(
                            morphology.materials[i].theta[z_slice, :, :] % np.pi,
                            cmap=plt.get_cmap("jet"),
                            norm=norm,
                            origin="lower",
                            interpolation="none",
                        )
                    ax5.set_ylabel("Y index")
                    ax5.set_xlabel("X index")
                    ax5.set_title(f"Mat {i} {morphology.materials[i].name} theta")
                    ax5.set_xlim(start_x, end_x)
                    ax5.set_ylim(start_y, end_y)
                    theta_cbar = plt.colorbar(thetaplot, fraction=0.040)
                    # theta_cbar.set_label("theta in radians", rotation=270, labelpad=22)

                    ax5i = inset_axes(
                        ax5,
                        axes_class=matplotlib.projections.get_projection_class("polar"),
                        width=0.7,
                        height=0.7,
                        axes_kwargs={"alpha": 0},
                    )

                    ax5i.grid(False)
                    # creates inset legend for orientation
                    azimuths_t = np.deg2rad(np.arange(-90, 90, 1))
                    zeniths_t = np.linspace(5, 10, 50)
                    values_t = np.mod(np.pi / 2 - azimuths_t, np.pi) * np.ones((50, 180))
                    ax5i.plot(
                        0,
                        0,
                        "o",
                        ms=75,
                        mec="none",
                        mfc=plt.rcParams["axes.facecolor"],
                        mew=2,
                        zorder=0,
                        alpha=0.7,
                    )
                    ax5i.pcolormesh(azimuths_t, zeniths_t, values_t, cmap=cm.jet, shading="auto")
                    ax5i.set_axis_off()
                    ax5i.arrow(
                        0, 0, 0, 4, width=0.005, head_width=0.2, head_length=0.6, lw=0.5
                    )  # ,facecolor='k',edgecolor='k')
                    ax5i.arrow(
                        np.pi / 2,
                        0,
                        0,
                        4,
                        width=0.005,
                        head_width=0.2,
                        head_length=0.6,
                        lw=0.5,
                    )  # , facecolor='k',edgecolor='k')
                    ax5i.text(2 * np.pi - np.deg2rad(40), 3.5, "X", fontsize=6)
                    ax5i.text(np.pi / 2 + np.deg2rad(40), 3.5, "Z", fontsize=6)

                if (runquiet is not True) or ("psi" in outputplot):
                    ax6 = plt.subplot(gs[3, 1])
                    norm = matplotlib.colors.Normalize(vmin=0, vmax=2 * np.pi, clip=False)
                    if screen_euler:
                        screen_mask = np.logical_or(
                            morphology.materials[i].Vfrac[z_slice, :, :] < 0.01,
                            morphology.materials[i].S[z_slice, :, :] < 0.01,
                        )
                        psiplot = ax6.imshow(
                            np.ma.masked_array(
                                morphology.materials[i].psi[z_slice, :, :] % (2 * np.pi),
                                screen_mask,
                            ),
                            cmap=psi_cmap,  # plt.get_cmap("hsv"),
                            norm=norm,
                            origin="lower",
                            interpolation="none",
                        )
                        if add_quiver:
                            screen_white = np.logical_or(
                                screen_mask,
                                (morphology.materials[i].psi[z_slice, :, :] % (2 * np.pi)) > np.pi,
                            )
                            screen_black = np.logical_or(
                                screen_mask,
                                (morphology.materials[i].psi[z_slice, :, :] % (2 * np.pi)) < np.pi,
                            )
                            sin_psi = np.sin(morphology.materials[i].psi[z_slice, :, :])
                            cos_psi = np.cos(morphology.materials[i].psi[z_slice, :, :])
                            len_scale = np.maximum(np.abs(sin_psi), np.abs(cos_psi))
                            if quiver_bw:
                                ax6.quiver(
                                    np.ma.masked_array(
                                        cos_psi / len_scale,
                                        screen_white,
                                    ),
                                    np.ma.masked_array(
                                        sin_psi / len_scale,
                                        screen_white,
                                    ),
                                    angles="xy",
                                    scale=1,
                                    pivot="mid",
                                    headaxislength=0,
                                    headlength=0,
                                    scale_units="xy",
                                    color="white",
                                )
                            else:
                                ax6.quiver(
                                    np.ma.masked_array(
                                        cos_psi / len_scale,
                                        screen_white,
                                    ),
                                    np.ma.masked_array(
                                        sin_psi / len_scale,
                                        screen_white,
                                    ),
                                    np.ma.masked_array(
                                        (morphology.materials[i].psi[z_slice, :, :] + np.pi)
                                        % (2 * np.pi),
                                        screen_white,
                                    ),
                                    cmap=psi_cmap,  # plt.get_cmap("hsv"),
                                    norm=norm,
                                    angles="xy",
                                    scale=1,
                                    pivot="mid",
                                    headaxislength=0,
                                    headlength=0,
                                    scale_units="xy",
                                )
                            if quiver_bw:
                                ax6.quiver(
                                    np.ma.masked_array(
                                        cos_psi / len_scale,
                                        screen_black,
                                    ),
                                    np.ma.masked_array(
                                        sin_psi / len_scale,
                                        screen_black,
                                    ),
                                    angles="xy",
                                    scale=1,
                                    pivot="mid",
                                    headaxislength=0,
                                    headlength=0,
                                    scale_units="xy",
                                    color="black",
                                )
                            else:
                                ax6.quiver(
                                    np.ma.masked_array(
                                        cos_psi / len_scale,
                                        screen_black,
                                    ),
                                    np.ma.masked_array(
                                        sin_psi / len_scale,
                                        screen_black,
                                    ),
                                    np.ma.masked_array(
                                        (morphology.materials[i].psi[z_slice, :, :] + np.pi)
                                        % (2 * np.pi),
                                        screen_black,
                                    ),
                                    cmap=psi_cmap,  # plt.get_cmap("hsv"),
                                    norm=norm,
                                    angles="xy",
                                    scale=1,
                                    pivot="mid",
                                    headaxislength=0,
                                    headlength=0,
                                    scale_units="xy",
                                )
                    else:
                        psiplot = ax6.imshow(
                            morphology.materials[i].psi[z_slice, :, :] % (2 * np.pi),
                            cmap=psi_cmap,  # plt.get_cmap("hsv"),
                            norm=norm,
                            origin="lower",
                            interpolation="none",
                        )
                        if add_quiver:
                            screen_white = (
                                morphology.materials[i].psi[z_slice, :, :] % (2 * np.pi) > np.pi
                            )
                            screen_black = (
                                morphology.materials[i].psi[z_slice, :, :] % (2 * np.pi) < np.pi
                            )
                            sin_psi = np.sin(morphology.materials[i].psi[z_slice, :, :])
                            cos_psi = np.cos(morphology.materials[i].psi[z_slice, :, :])
                            len_scale = np.maximum(np.abs(sin_psi), np.abs(cos_psi))
                            ax6.quiver(
                                np.ma.masked_array(
                                    cos_psi / len_scale,
                                    screen_white,
                                ),
                                np.ma.masked_array(
                                    sin_psi / len_scale,
                                    screen_white,
                                ),
                                angles="xy",
                                scale=1,
                                pivot="mid",
                                headaxislength=0,
                                headlength=0,
                                scale_units="xy",
                                color="white",
                            )
                            ax6.quiver(
                                np.ma.masked_array(
                                    cos_psi / len_scale,
                                    screen_black,
                                ),
                                np.ma.masked_array(
                                    sin_psi / len_scale,
                                    screen_black,
                                ),
                                angles="xy",
                                scale=1,
                                pivot="mid",
                                headaxislength=0,
                                headlength=0,
                                scale_units="xy",
                                color="black",
                            )

                    ax6.set_ylabel("Y index")
                    ax6.set_xlabel("X index")
                    ax6.set_title(f"Mat {i} {morphology.materials[i].name} psi")
                    ax6.set_xlim(start_x, end_x)
                    ax6.set_ylim(start_y, end_y)
                    psi_cbar = plt.colorbar(psiplot, fraction=0.040)
                    # psi_cbar.set_label("psi in radians", labelpad=22)
                    ax4i = inset_axes(
                        ax6,
                        axes_class=matplotlib.projections.get_projection_class("polar"),
                        width=0.7,
                        height=0.7,
                        axes_kwargs={"alpha": 0},
                    )

                    ax4i.grid(False)
                    # creates inset legend for orientation
                    azimuths = np.deg2rad(np.arange(0, 360, 1))
                    zeniths = np.linspace(5, 10, 50)
                    values = np.mod(azimuths, 2 * np.pi) * np.ones((50, 360))
                    ax4i.plot(
                        0,
                        0,
                        "o",
                        ms=75,
                        mec="none",
                        mfc=plt.rcParams["axes.facecolor"],
                        mew=2,
                        zorder=0,
                        alpha=0.7,
                    )
                    ax4i.pcolormesh(azimuths, zeniths, values, cmap=psi_cmap, shading="auto")
                    ax4i.set_axis_off()
                    ax4i.arrow(
                        0, 0, 0, 4, width=0.005, head_width=0.2, head_length=0.6, lw=0.5
                    )  # ,facecolor='k',edgecolor='k')
                    ax4i.arrow(
                        np.pi / 2,
                        0,
                        0,
                        4,
                        width=0.005,
                        head_width=0.2,
                        head_length=0.6,
                        lw=0.5,
                    )  # ,facecolor='k',edgecolor='k')
                    ax4i.text(2 * np.pi - np.deg2rad(40), 3.5, "X", fontsize=6)
                    ax4i.text(np.pi / 2 + np.deg2rad(40), 3.5, "Y", fontsize=6)

                if runquiet is not True:
                    ax7 = plt.subplot(gs[4, 0])
                    ax7.hist(morphology.materials[i].theta.flatten())
                    ax7.set_title(f"Mat {i} {morphology.materials[i].name} theta")
                    ax7.set_xlim(left=0)
                    ax7.set_xlabel("theta in radians")
                    ax7.set_ylabel("num voxels")
                    ax7.set_yscale("log")

                    ax8 = plt.subplot(gs[4, 1])
                    ax8.hist(morphology.materials[i].psi.flatten())
                    ax8.set_title(f"Mat {i} {morphology.materials[i].name} psi")
                    ax8.set_xlim(left=0)
                    ax8.set_xlabel("psi in radians")
                    ax8.set_ylabel("num voxels")
                    ax8.set_yscale("log")

            if runquiet is False:  # Show plot and/or export to file
                # Exporting plots
                if exportDir is not None:
                    # Attempt to export image
                    outDir = pathlib.Path(exportDir)
                    outPath = outDir / f"Mat{i}_viz"

                    if exportParams is None:  # No user provided kwargs
                        plt.savefig(fname=(str(outPath) + ".png"), dpi=dpi, format="png")
                    else:  # Apply user provided kwargs
                        # Grab format, if provided, else do png
                        format = exportParams.get("format", "png")
                        plt.savefig(fname=(str(outPath) + f".{format}"), **exportParams)
                if batchMode is True:  # Dont show plots, but do export them
                    pass
                else:
                    plt.show()

            if outputmat and (i in outputmat):
                fig.canvas.draw()
                rgb_return = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                rgb_return = rgb_return.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                if outputplot and ("vfrac" in outputplot):
                    if outputaxes:
                        fig_xl = ax1.get_tightbbox().intervalx[0].astype(int)
                        fig_xr = Vfrac_cbar.ax.get_tightbbox().intervalx[1].astype(int)

                        fig_yt = rgb_return.shape[0] - ax1.get_tightbbox().intervaly[1].astype(int)
                        fig_yb = rgb_return.shape[0] - ax1.get_tightbbox().intervaly[0].astype(int)
                    else:
                        fig_yt = rgb_return.shape[0] - ax1.get_window_extent().intervaly[1].astype(
                            int
                        )
                        fig_yb = rgb_return.shape[0] - ax1.get_window_extent().intervaly[0].astype(
                            int
                        )
                        fig_xl = ax1.get_window_extent().intervalx[0].astype(int)
                        fig_xr = ax1.get_window_extent().intervalx[1].astype(int)

                    rgb_return_list.append(rgb_return[fig_yt:fig_yb, fig_xl:fig_xr])
                if outputplot and "S" in outputplot:
                    if outputaxes:
                        fig_yt = rgb_return.shape[0] - ax2.get_tightbbox().intervaly[1].astype(int)
                        fig_yb = rgb_return.shape[0] - ax2.get_tightbbox().intervaly[0].astype(int)
                        fig_xl = ax2.get_tightbbox().intervalx[0].astype(int)
                        fig_xr = S_cbar.ax.get_tightbbox().intervalx[1].astype(int)
                    else:
                        fig_yt = rgb_return.shape[0] - ax2.get_window_extent().intervaly[1].astype(
                            int
                        )
                        fig_yb = rgb_return.shape[0] - ax2.get_window_extent().intervaly[0].astype(
                            int
                        )
                        fig_xl = ax2.get_window_extent().intervalx[0].astype(int)
                        fig_xr = ax2.get_window_extent().intervalx[1].astype(int)
                    rgb_return_list.append(rgb_return[fig_yt:fig_yb, fig_xl:fig_xr])
                if outputplot and "theta" in outputplot:
                    if outputaxes:
                        fig_yt = rgb_return.shape[0] - ax5.get_tightbbox().intervaly[1].astype(int)
                        fig_yb = rgb_return.shape[0] - ax5.get_tightbbox().intervaly[0].astype(int)
                        fig_xl = ax5.get_tightbbox().intervalx[0].astype(int)
                        fig_xr = theta_cbar.ax.get_tightbbox().intervalx[1].astype(int)
                    else:
                        fig_yt = rgb_return.shape[0] - ax5.get_window_extent().intervaly[1].astype(
                            int
                        )
                        fig_yb = rgb_return.shape[0] - ax5.get_window_extent().intervaly[0].astype(
                            int
                        )
                        fig_xl = ax5.get_window_extent().intervalx[0].astype(int)
                        fig_xr = ax5.get_window_extent().intervalx[1].astype(int)
                    rgb_return_list.append(rgb_return[fig_yt:fig_yb, fig_xl:fig_xr])
                if outputplot and "psi" in outputplot:
                    if outputaxes:
                        fig_yt = rgb_return.shape[0] - ax6.get_tightbbox().intervaly[1].astype(int)
                        fig_yb = rgb_return.shape[0] - ax6.get_tightbbox().intervaly[0].astype(int)
                        fig_xl = ax6.get_tightbbox().intervalx[0].astype(int)
                        fig_xr = psi_cbar.ax.get_tightbbox().intervalx[1].astype(int)
                    else:
                        fig_yt = rgb_return.shape[0] - ax6.get_window_extent().intervaly[1].astype(
                            int
                        )
                        fig_yb = rgb_return.shape[0] - ax6.get_window_extent().intervaly[0].astype(
                            int
                        )
                        fig_xl = ax6.get_window_extent().intervalx[0].astype(int)
                        fig_xr = ax6.get_window_extent().intervalx[1].astype(int)
                    rgb_return_list.append(rgb_return[fig_yt:fig_yb, fig_xl:fig_xr])
        if not runquiet:
            plt.show()
        plt.clf()
        plt.close(fig)
    finally:
        # this code can hijack the backend and "agg" will not show figures, so this needs always to run, regardless of whether there are errors in the arguments.
        matplotlib.rc_file_defaults()
        matplotlib.use(backend_)
    return rgb_return_list
