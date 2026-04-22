import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc, gridspec
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import warnings
import pathlib

DEFAULT_NOTEBOOK_FIGSIZE = (9.1, 13.65)
DEFAULT_NOTEBOOK_DPI = 240
DEFAULT_HISTOGRAM_SAMPLE_SIZE = 100_000
DEFAULT_HISTOGRAM_SAMPLE_THRESHOLD = 100_000


def _resolve_figure_size():
    rc_figsize = tuple(float(value) for value in matplotlib.rcParams.get("figure.figsize", []))
    default_rc_figsize = tuple(
        float(value) for value in matplotlib.rcParamsDefault.get("figure.figsize", [])
    )
    if rc_figsize and rc_figsize != default_rc_figsize:
        return rc_figsize
    return DEFAULT_NOTEBOOK_FIGSIZE


def _compute_axis_window(axis_size: int, window_size: int, translate: int = None):
    window_size = min(int(window_size), int(axis_size))
    start = max(0, (axis_size - window_size) // 2)
    if translate is not None:
        start += int(translate)
    start = min(max(0, start), axis_size - window_size)
    end = start + window_size
    return start, end


def _is_cupy_array(array) -> bool:
    return hasattr(array, "get") and array.__class__.__module__.startswith("cupy")


def _histogram_values(
    array,
    mode: str,
    sample_size: int,
    sample_threshold: int,
):
    flat = array.ravel()
    total_size = int(flat.size)

    if mode not in {"auto", "full", "sample"}:
        raise ValueError("histogram_mode must be one of 'auto', 'full', or 'sample'.")

    use_sample = mode == "sample" or (mode == "auto" and total_size > int(sample_threshold))
    selected_count = min(int(sample_size), total_size) if use_sample else total_size

    if use_sample and selected_count < total_size:
        if _is_cupy_array(flat):
            import cupy as cp

            rng = cp.random.default_rng(0)
            sample_idx = rng.choice(total_size, size=selected_count, replace=False)
            values = cp.asnumpy(flat[sample_idx])
        else:
            rng = np.random.default_rng(0)
            sample_idx = rng.choice(total_size, size=selected_count, replace=False)
            values = np.asarray(flat[sample_idx])
        return values, True, selected_count

    if _is_cupy_array(flat):
        import cupy as cp

        return cp.asnumpy(flat), False, total_size
    return np.asarray(flat), False, total_size


def _resolve_slice_plane(num_zyx, z_slice, y_slice, x_slice):
    requested = {
        "xy": z_slice if (y_slice is None and x_slice is None) else None,
        "xz": y_slice,
        "yz": x_slice,
    }
    selected = [(plane, index) for plane, index in requested.items() if index is not None]

    if len(selected) > 1:
        raise ValueError("Specify at most one of z_slice, y_slice, or x_slice.")

    if not selected:
        plane = "xy"
        index = 0
    else:
        plane, index = selected[0]

    axis_name = {"xy": "z_slice", "xz": "y_slice", "yz": "x_slice"}[plane]
    axis_index = {"xy": 0, "xz": 1, "yz": 2}[plane]
    axis_size = int(num_zyx[axis_index])

    if index < 0:
        warnings.warn(
            f"{axis_name} of {index} is less than 0. Using {axis_name} = 0 instead."
        )
        index = 0
    if index > axis_size - 1:
        warnings.warn(
            f"{axis_name} of {index} is greater than the maximum index of {axis_size - 1}."
            f" Using {axis_name} = {axis_size - 1} instead."
        )
        index = axis_size - 1

    return plane, int(index)


def _plane_display_config(num_zyx, plane, subsample, translate_x, translate_y):
    if plane == "xy":
        vertical_axis = 1
        horizontal_axis = 2
        vertical_label = "Y index"
        horizontal_label = "X index"
    elif plane == "xz":
        vertical_axis = 0
        horizontal_axis = 2
        vertical_label = "Z index"
        horizontal_label = "X index"
    else:
        vertical_axis = 0
        horizontal_axis = 1
        vertical_label = "Z index"
        horizontal_label = "Y index"

    if subsample is None:
        vertical_window = int(num_zyx[vertical_axis])
        horizontal_window = int(num_zyx[horizontal_axis])
    else:
        vertical_window = int(subsample)
        horizontal_window = int(subsample)

    vertical_start, vertical_end = _compute_axis_window(
        num_zyx[vertical_axis],
        vertical_window,
        translate_y,
    )
    horizontal_start, horizontal_end = _compute_axis_window(
        num_zyx[horizontal_axis],
        horizontal_window,
        translate_x,
    )

    return {
        "plane": plane,
        "vertical_axis": vertical_axis,
        "horizontal_axis": horizontal_axis,
        "vertical_label": vertical_label,
        "horizontal_label": horizontal_label,
        "vertical_start": vertical_start,
        "vertical_end": vertical_end,
        "horizontal_start": horizontal_start,
        "horizontal_end": horizontal_end,
    }


def _extract_plane_slice(array, plane, slice_index, display_config):
    vertical_start = display_config["vertical_start"]
    vertical_end = display_config["vertical_end"]
    horizontal_start = display_config["horizontal_start"]
    horizontal_end = display_config["horizontal_end"]

    if plane == "xy":
        return array[slice_index, vertical_start:vertical_end, horizontal_start:horizontal_end]
    if plane == "xz":
        return array[vertical_start:vertical_end, slice_index, horizontal_start:horizontal_end]
    return array[vertical_start:vertical_end, horizontal_start:horizontal_end, slice_index]


def _crop_slice_to_square(slice_array):
    height, width = slice_array.shape[:2]
    if height == width:
        return slice_array
    if width > height:
        start_x = (width - height) // 2
        end_x = start_x + height
        return slice_array[:, start_x:end_x]
    start_y = (height - width) // 2
    end_y = start_y + width
    return slice_array[start_y:end_y, :]


def morphology_visualizer(
    morphology,
    z_slice: int = 0,
    y_slice: int = None,
    x_slice: int = None,
    subsample: int = None,
    translate_x: int = None,
    translate_y: int = None,
    vertical_slice_aspect: str = "full",
    screen_euler: bool = True,
    screen_euler_vfrac: float = 0.05,
    screen_euler_s: float = 0.05,
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
    dpi: int = DEFAULT_NOTEBOOK_DPI,
    histograms: bool = True,
    histogram_mode: str = "auto",
    histogram_sample_size: int = DEFAULT_HISTOGRAM_SAMPLE_SIZE,
    histogram_sample_threshold: int = DEFAULT_HISTOGRAM_SAMPLE_THRESHOLD,
    exportDir: str = None,
    exportParams: dict = None,
):
    """
    Reads in morphology HDF5 file and checks that the format is consistent for CyRSoXS. Optionally plots and returns select quantities.

    Parameters
    ----------

        z_slice : int
            Which z-slice of the array to plot for an XY view. Mutually exclusive with
            y_slice and x_slice.
        y_slice : int
            Which y-slice of the array to plot for an XZ view. Mutually exclusive with
            z_slice and x_slice.
        x_slice : int
            Which x-slice of the array to plot for a YZ view. Mutually exclusive with
            z_slice and y_slice.
        subsample : int
            Number of voxels to display along the horizontal and vertical axes of the
            selected plane.
        translate_x : int
            Number of voxels to translate the displayed window along the horizontal axis
            of the selected plane; meant for use with subsample.
        translate_y : int
            Number of voxels to translate the displayed window along the vertical axis
            of the selected plane; meant for use with subsample.
        vertical_slice_aspect : str
            Aspect handling for XZ and YZ views. Use 'full' to preserve the full returned
            panel including its original colorbar-inclusive width, or 'square' to center-crop
            the slice data to a square before plotting. XY views ignore this option.
        screen_euler : bool
            Suppress visualization of euler angles where vfrac < screen_euler_vfrac or S < screen_euler_s; intended to hilight edges
            screen_euler_vfrac : float
            screen_euler_s : float
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
            The dpi at which the plot is generated. The default figure size is 9.1" x 13.65"
            unless overridden via Matplotlib rcParams.
        histograms : bool
            When True, include the four full-volume histogram panels in interactive display mode.
        histogram_mode : str
            Histogram calculation mode: 'full' always uses all voxels, 'sample' always uses a
            sample, and 'auto' switches to sampling when the voxel count exceeds
            histogram_sample_threshold.
        histogram_sample_size : int
            Number of voxels to use when histogram_mode resolves to sampled histograms.
        histogram_sample_threshold : int
            Voxel-count threshold above which histogram_mode='auto' switches to sampled
            histograms.
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

    plane, slice_index = _resolve_slice_plane(morphology.NumZYX, z_slice, y_slice, x_slice)
    display_config = _plane_display_config(
        morphology.NumZYX,
        plane,
        subsample,
        translate_x,
        translate_y,
    )
    plane_name = {"xy": "XY", "xz": "XZ", "yz": "YZ"}[plane]
    slice_label = {"xy": "z_slice", "xz": "y_slice", "yz": "x_slice"}[plane]

    if not runquiet:
        print(
            f"Dataset dimensions (Z, Y, X): {morphology.NumZYX[0]} x {morphology.NumZYX[1]} x"
            f" {morphology.NumZYX[2]}"
        )
        print(f"Number of Materials: {morphology._numMaterial}")
        print(f"Viewing {plane_name} plane at {slice_label} = {slice_index}")
        print("")

    font = {
        "family": "sans-serif",
        "sans-serif": "DejaVu Sans",
        "weight": "regular",
        "size": 8,
    }

    cwdPath = pathlib.Path(__file__).resolve().parent
    psi_cmap = matplotlib.colors.ListedColormap(np.load(cwdPath / "cmap/infinitydouble_cmap.npy"))
    requested_plots = set(outputplot or [])
    figure_size = _resolve_figure_size()

    if histogram_sample_size <= 0:
        raise ValueError("histogram_sample_size must be a positive integer.")
    if histogram_sample_threshold <= 0:
        raise ValueError("histogram_sample_threshold must be a positive integer.")
    if vertical_slice_aspect not in {"full", "square"}:
        raise ValueError("vertical_slice_aspect must be one of 'full' or 'square'.")

    with plt.style.context(style_dict[plotstyle]), matplotlib.rc_context(
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.weight": "regular",
            "font.size": 8,
        }
    ):
        rc("font", **font)

        for i in range(1, morphology._numMaterial + 1):
            material = morphology.materials[i]
            show_full_summary = not runquiet
            show_histograms = show_full_summary and histograms
            return_material = outputmat is not None and i in outputmat
            render_material = show_full_summary or return_material

            if not render_material:
                continue

            show_vfrac = show_full_summary or "vfrac" in requested_plots
            show_s = show_full_summary or "S" in requested_plots
            show_theta = show_full_summary or "theta" in requested_plots
            show_psi = show_full_summary or "psi" in requested_plots
            show_quiver = show_psi and add_quiver

            if show_quiver and plane != "xy":
                raise ValueError("add_quiver is only supported for XY views selected with z_slice.")

            field_cache = {}

            def get_field(field_name):
                if field_name not in field_cache:
                    field_cache[field_name] = morphology._material_effective_field(material, field_name)
                return field_cache[field_name]

            vfrac = get_field("Vfrac") if (show_vfrac or show_theta or show_psi) else None
            s_field = get_field("S") if (show_s or show_theta or show_psi) else None
            theta_field = get_field("theta") if show_theta else None
            psi_field = get_field("psi") if (show_psi or show_quiver) else None

            if show_full_summary:
                print(f"Material {i} Vfrac. Min: {vfrac.min()} Max: {vfrac.max()}")
                print(f"Material {i} S. Min: {s_field.min()} Max: {s_field.max()}")
                print(f"Material {i} theta. Min: {theta_field.min()} Max: {theta_field.max()}")
                print(f"Material {i} psi. Min: {psi_field.min()} Max: {psi_field.max()}")

            if theta_field is not None and (
                (theta_field.min() < 0) or (theta_field.max() > np.pi)
            ):
                warnings.warn(
                    "Visualization expects theta to have bounds of [0,pi]. This model has theta"
                    " outside those bounds and visualization may be incorrect."
                )

            fig = plt.figure(figsize=figure_size, dpi=dpi)
            try:
                if show_histograms:
                    gs = gridspec.GridSpec(
                        nrows=5,
                        ncols=2,
                        figure=fig,
                        width_ratios=[1, 1],
                        height_ratios=[3, 1, 0.1, 3, 1],
                        wspace=0.3,
                        hspace=0.65,
                    )
                else:
                    gs = gridspec.GridSpec(
                        nrows=3,
                        ncols=2,
                        figure=fig,
                        width_ratios=[1, 1],
                        height_ratios=[3, 0.1, 3],
                        wspace=0.3,
                        hspace=0.4,
                    )

                vfrac_slice = None
                s_slice = None
                theta_slice = None
                psi_slice = None
                screen_mask = None

                if vfrac is not None:
                    vfrac_slice = _extract_plane_slice(vfrac, plane, slice_index, display_config)
                if s_field is not None:
                    s_slice = _extract_plane_slice(s_field, plane, slice_index, display_config)
                if theta_field is not None:
                    theta_slice = _extract_plane_slice(theta_field, plane, slice_index, display_config)
                if psi_field is not None:
                    psi_slice = _extract_plane_slice(psi_field, plane, slice_index, display_config)
                if plane != "xy" and vertical_slice_aspect == "square":
                    if vfrac_slice is not None:
                        vfrac_slice = _crop_slice_to_square(vfrac_slice)
                    if s_slice is not None:
                        s_slice = _crop_slice_to_square(s_slice)
                    if theta_slice is not None:
                        theta_slice = _crop_slice_to_square(theta_slice)
                    if psi_slice is not None:
                        psi_slice = _crop_slice_to_square(psi_slice)
                if screen_euler and (show_theta or show_psi):
                    screen_mask = np.logical_or(
                        vfrac_slice < screen_euler_vfrac,
                        s_slice < screen_euler_s,
                    )

                if show_vfrac:
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
                        vfrac_slice,
                        cmap=cmap,
                        origin="lower",
                        interpolation="none",
                        norm=norm,
                    )
                    ax1.set_ylabel(display_config["vertical_label"], labelpad=0)
                    ax1.set_xlabel(display_config["horizontal_label"])
                    ax1.set_title(f"Mat {i} {material.name} Vfrac ({plane_name})")
                    Vfrac_cbar = plt.colorbar(Vfracplot, ax=ax1, fraction=0.040)

                if show_s:
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
                        s_slice,
                        cmap=cmap,
                        origin="lower",
                        interpolation="none",
                        norm=norm,
                    )
                    ax2.set_ylabel(display_config["vertical_label"], labelpad=0)
                    ax2.set_xlabel(display_config["horizontal_label"])
                    ax2.set_title(f"Mat {i} {material.name} S ({plane_name})")
                    S_cbar = plt.colorbar(Splot, fraction=0.040)

                if show_histograms:
                    vfrac_hist, vfrac_sampled, vfrac_hist_count = _histogram_values(
                        vfrac,
                        histogram_mode,
                        histogram_sample_size,
                        histogram_sample_threshold,
                    )
                    s_hist, s_sampled, s_hist_count = _histogram_values(
                        s_field,
                        histogram_mode,
                        histogram_sample_size,
                        histogram_sample_threshold,
                    )
                    theta_hist, theta_sampled, theta_hist_count = _histogram_values(
                        theta_field,
                        histogram_mode,
                        histogram_sample_size,
                        histogram_sample_threshold,
                    )
                    psi_hist, psi_sampled, psi_hist_count = _histogram_values(
                        psi_field,
                        histogram_mode,
                        histogram_sample_size,
                        histogram_sample_threshold,
                    )

                    ax3 = plt.subplot(gs[1, 0])
                    ax3.hist(vfrac_hist)
                    vfrac_hist_title = f"Mat {i} {material.name} Vfrac"
                    if vfrac_sampled:
                        vfrac_hist_title += f" sampled ({vfrac_hist_count})"
                    ax3.set_title(vfrac_hist_title)
                    ax3.set_xlim(left=0)
                    ax3.set_xlabel("Vfrac: volume fraction")
                    ax3.set_ylabel("num voxels")
                    ax3.set_yscale("log")

                    ax4 = plt.subplot(gs[1, 1])
                    ax4.hist(s_hist)
                    s_hist_title = f"Mat {i} {material.name} S"
                    if s_sampled:
                        s_hist_title += f" sampled ({s_hist_count})"
                    ax4.set_title(s_hist_title)
                    ax4.set_xlim(left=0)
                    ax4.set_xlabel("S: orientational order parameter")
                    ax4.set_ylabel("num voxels")
                    ax4.set_yscale("log")

                if show_theta:
                    theta_row = 3 if show_histograms else 2
                    ax5 = plt.subplot(gs[theta_row, 0])
                    norm = matplotlib.colors.Normalize(vmin=0, vmax=np.pi, clip=False)
                    if screen_euler:
                        theta_image = np.ma.masked_array(theta_slice % np.pi, screen_mask)
                    else:
                        theta_image = theta_slice % np.pi
                    thetaplot = ax5.imshow(
                        theta_image,
                        cmap=plt.get_cmap("jet"),
                        norm=norm,
                        origin="lower",
                        interpolation="none",
                    )
                    ax5.set_ylabel(display_config["vertical_label"])
                    ax5.set_xlabel(display_config["horizontal_label"])
                    ax5.set_title(f"Mat {i} {material.name} theta ({plane_name})")
                    theta_cbar = plt.colorbar(thetaplot, fraction=0.040)

                    ax5i = inset_axes(
                        ax5,
                        axes_class=matplotlib.projections.get_projection_class("polar"),
                        width=0.7,
                        height=0.7,
                        axes_kwargs={"alpha": 0},
                    )

                    ax5i.grid(False)
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
                    ax5i.arrow(0, 0, 0, 4, width=0.005, head_width=0.2, head_length=0.6, lw=0.5)
                    ax5i.arrow(
                        np.pi / 2,
                        0,
                        0,
                        4,
                        width=0.005,
                        head_width=0.2,
                        head_length=0.6,
                        lw=0.5,
                    )
                    ax5i.text(2 * np.pi - np.deg2rad(40), 3.5, "X", fontsize=6)
                    ax5i.text(np.pi / 2 + np.deg2rad(40), 3.5, "Z", fontsize=6)

                if show_psi:
                    psi_row = 3 if show_histograms else 2
                    ax6 = plt.subplot(gs[psi_row, 1])
                    norm = matplotlib.colors.Normalize(vmin=0, vmax=2 * np.pi, clip=False)
                    psi_phase = psi_slice % (2 * np.pi)
                    if screen_euler:
                        psi_image = np.ma.masked_array(psi_phase, screen_mask)
                    else:
                        psi_image = psi_phase
                    psiplot = ax6.imshow(
                        psi_image,
                        cmap=psi_cmap,
                        norm=norm,
                        origin="lower",
                        interpolation="none",
                    )
                    if show_quiver:
                        if screen_euler:
                            screen_white = np.logical_or(screen_mask, psi_phase > np.pi)
                            screen_black = np.logical_or(screen_mask, psi_phase < np.pi)
                        else:
                            screen_white = psi_phase > np.pi
                            screen_black = psi_phase < np.pi
                        sin_psi = np.sin(psi_slice)
                        cos_psi = np.cos(psi_slice)
                        len_scale = np.maximum(np.abs(sin_psi), np.abs(cos_psi))
                        if quiver_bw:
                            ax6.quiver(
                                np.ma.masked_array(cos_psi / len_scale, screen_white),
                                np.ma.masked_array(sin_psi / len_scale, screen_white),
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
                                np.ma.masked_array(cos_psi / len_scale, screen_white),
                                np.ma.masked_array(sin_psi / len_scale, screen_white),
                                np.ma.masked_array((psi_slice + np.pi) % (2 * np.pi), screen_white),
                                cmap=psi_cmap,
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
                                np.ma.masked_array(cos_psi / len_scale, screen_black),
                                np.ma.masked_array(sin_psi / len_scale, screen_black),
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
                                np.ma.masked_array(cos_psi / len_scale, screen_black),
                                np.ma.masked_array(sin_psi / len_scale, screen_black),
                                np.ma.masked_array((psi_slice + np.pi) % (2 * np.pi), screen_black),
                                cmap=psi_cmap,
                                norm=norm,
                                angles="xy",
                                scale=1,
                                pivot="mid",
                                headaxislength=0,
                                headlength=0,
                                scale_units="xy",
                            )

                    ax6.set_ylabel(display_config["vertical_label"])
                    ax6.set_xlabel(display_config["horizontal_label"])
                    ax6.set_title(f"Mat {i} {material.name} psi ({plane_name})")
                    psi_cbar = plt.colorbar(psiplot, fraction=0.040)
                    ax4i = inset_axes(
                        ax6,
                        axes_class=matplotlib.projections.get_projection_class("polar"),
                        width=0.7,
                        height=0.7,
                        axes_kwargs={"alpha": 0},
                    )

                    ax4i.grid(False)
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
                    ax4i.arrow(0, 0, 0, 4, width=0.005, head_width=0.2, head_length=0.6, lw=0.5)
                    ax4i.arrow(
                        np.pi / 2,
                        0,
                        0,
                        4,
                        width=0.005,
                        head_width=0.2,
                        head_length=0.6,
                        lw=0.5,
                    )
                    ax4i.text(2 * np.pi - np.deg2rad(40), 3.5, "X", fontsize=6)
                    ax4i.text(np.pi / 2 + np.deg2rad(40), 3.5, "Y", fontsize=6)

                if show_histograms:
                    ax7 = plt.subplot(gs[4, 0])
                    ax7.hist(theta_hist)
                    theta_hist_title = f"Mat {i} {material.name} theta"
                    if theta_sampled:
                        theta_hist_title += f" sampled ({theta_hist_count})"
                    ax7.set_title(theta_hist_title)
                    ax7.set_xlim(left=0)
                    ax7.set_xlabel("theta in radians")
                    ax7.set_ylabel("num voxels")
                    ax7.set_yscale("log")

                    ax8 = plt.subplot(gs[4, 1])
                    ax8.hist(psi_hist)
                    psi_hist_title = f"Mat {i} {material.name} psi"
                    if psi_sampled:
                        psi_hist_title += f" sampled ({psi_hist_count})"
                    ax8.set_title(psi_hist_title)
                    ax8.set_xlim(left=0)
                    ax8.set_xlabel("psi in radians")
                    ax8.set_ylabel("num voxels")
                    ax8.set_yscale("log")

                if return_material:
                    fig.canvas.draw()
                    rgb_return = np.array(fig.canvas.buffer_rgba(), dtype=np.uint8)
                    rgb_return = rgb_return.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                    if "vfrac" in requested_plots:
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
                    if "S" in requested_plots:
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
                    if "theta" in requested_plots:
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
                    if "psi" in requested_plots:
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

                if show_full_summary and exportDir is not None:
                    outDir = pathlib.Path(exportDir)
                    outPath = outDir / f"Mat{i}_viz"

                    if exportParams is None:
                        plt.savefig(fname=(str(outPath) + ".png"), dpi=dpi, format="png")
                    else:
                        format = exportParams.get("format", "png")
                        savefig_params = dict(exportParams)
                        savefig_params.pop("figsize", None)
                        plt.savefig(fname=(str(outPath) + f".{format}"), **savefig_params)

                if show_full_summary and not batchMode:
                    plt.show()
            finally:
                plt.close(fig)
    return rgb_return_list
