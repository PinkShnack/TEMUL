import hyperspy
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
from decimal import Decimal
import colorcet as cc
from matplotlib_scalebar.scalebar import ScaleBar
from temul.signal_plotting import (
    get_polar_2d_colorwheel_color_list,
    _make_color_wheel)


# good to have an example of getting atom_positions_A and B from sublattice
def find_polarisation_vectors(atom_positions_A, atom_positions_B,
                              save=None):
    """
    Calculate the vectors from atom_positions_A to atom_positions_B.

    Parameters
    ----------
    atom_positions_A, atom_positions_B : list
        Atom positions list in the form [[x1,y1], [x2,y2], [x3,y3]...].
    save : string, default None
        If set to a string, the array will be saved.

    Returns
    -------
    two lists: u and v components of the vector from A to B

    Examples
    --------
    >>> from temul.topotem.polarisation import find_polarisation_vectors
    >>> pos_A = [[1,2], [3,4], [5,8], [5,2]]
    >>> pos_B = [[1,1], [5,2], [3,1], [6,2]]
    >>> u, v = find_polarisation_vectors(pos_A, pos_B, save=None)

    convert to the [[u1,v1], [u2,v2], [u3,v3]...] format

    >>> import numpy as np
    >>> vectors = np.asarray([u, v]).T

    """
    if len(atom_positions_A) != len(atom_positions_B):
        raise ValueError("atom_positions_A and atom_positions_B must be the "
                         "same length")

    atom_positions_A_list_x = [row[0] for row in atom_positions_A]
    atom_positions_A_list_y = [row[1] for row in atom_positions_A]

    atom_positions_B_list_x = [row[0] for row in atom_positions_B]
    atom_positions_B_list_y = [row[1] for row in atom_positions_B]

    # Create a list of dx and dy by simple subtraction
    u_v_component_list = []
    i = 0
    delta = 1
    while i < len(atom_positions_B):
        u_v_component = (
            atom_positions_B_list_x[i] - atom_positions_A_list_x[i],
            atom_positions_B_list_y[i] - atom_positions_A_list_y[i])

        u_v_component_list.append(u_v_component)
        i = i + delta

    # Separate the created list into u (u=dx) and v (v=dy)
    #   u and v are notation used by the ax.quiver plotting tool
    u = [row[0] for row in u_v_component_list]
    v = [row[1] for row in u_v_component_list]

    if save is not None:
        np.save(save + '.npy', u_v_component_list)

    return (u, v)


def corrected_vectors_via_average(u, v):
    u, v = np.asarray(u), np.asarray(v)
    u_av_corr = u - np.mean(u)
    v_av_corr = v - np.mean(v)
    return (u_av_corr, v_av_corr)


def _calc_2D_center_of_mass(u, v):
    u, v = np.asarray(u), np.asarray(v)
    r = (u ** 2 + v ** 2) ** 0.5
    u_com = np.sum(u * r) / np.sum(r)
    v_com = np.sum(v * r) / np.sum(r)
    return (u_com, v_com)


def corrected_vectors_via_center_of_mass(u, v):
    u_com, v_com = _calc_2D_center_of_mass(u, v)
    u_com_corr = u - u_com
    v_com_corr = v - v_com
    return (u_com_corr, v_com_corr)


def correct_off_tilt_vectors(u, v, method="com"):
    """ Useful if your image is off-tilt (electron beam is not perfectly
    parallel to the atomic columns).

    Parameters
    ----------
    u, v : 1D numpy arrays
        horizontal and vertical components of the (polarisation) vectors.
    method : string, default "com"
        method used to correct the vector components. "com" is via the center
        of mass of the vectors. "av" is via the average vector.

    Returns
    -------
    u_corr, v_corr : corrected 1D numpy arrays

    Examples
    --------
    >>> from temul.topotem.polarisation import correct_off_tilt_vectors
    >>> from temul.dummy_data import get_polarisation_dummy_dataset
    >>> atom_lattice = get_polarisation_dummy_dataset()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeB = atom_lattice.sublattice_list[1]
    >>> sublatticeA.construct_zone_axes()
    >>> za0, za1 = sublatticeA.zones_axis_average_distances[0:2]
    >>> s_p = sublatticeA.get_polarization_from_second_sublattice(
    ...     za0, za1, sublatticeB, color='blue')
    >>> vector_list = s_p.metadata.vector_list
    >>> x, y = [i[0] for i in vector_list], [i[1] for i in vector_list]
    >>> u, v = [i[2] for i in vector_list], [i[3] for i in vector_list]

    Correct for some tilt using the correct_off_tilt_vectors function:

    >>> u_com, v_com = correct_off_tilt_vectors(u, v, method="com")

    Use the average vector instead: (be careful that you're not just applying
    this on previously corrected data!)

    >>> u_av, v_av = correct_off_tilt_vectors(u, v, method="av")

    """
    if "com" in method.lower():
        u_corr, v_corr = corrected_vectors_via_center_of_mass(u, v)
    if "av" in method.lower():
        u_corr, v_corr = corrected_vectors_via_average(u, v)
    return (u_corr, v_corr)


def plot_polarisation_vectors(
        x, y, u, v, image, sampling=None, units='pix',
        plot_style='vector', overlay=True, unit_vector=False,
        vector_rep='magnitude', degrees=False, angle_offset=None,
        save='polarisation_image', title="", color='yellow',
        cmap=None, alpha=1.0, image_cmap='gray', monitor_dpi=96,
        no_axis_info=True, invert_y_axis=True, ticks=None, scalebar=False,
        antialiased=False, levels=20, remove_vectors=False,
        quiver_units='width', pivot='middle', angles='xy',
        scale_units='xy', scale=None, headwidth=3.0, headlength=5.0,
        headaxislength=4.5, width=None, minshaft=1, minlength=1):
    """
    Plot the polarisation vectors. These can be found with
    ``find_polarisation_vectors()`` or Atomap's
    ``get_polarization_from_second_sublattice()`` function.

    Parameters
    ----------
    See matplotlib's quiver function for more details.

    x, y : list or 1D NumPy array
        xy coordinates of the vectors on the image.
    u, v : list or 1D NumPy array
        uv vector components.
    image : 2D NumPy array
        image is used to fit the image. Will flip the y axis, as used for
        electron microscopy data (top left point is (0, 0) coordinate).
    sampling : float, default None
        Pixel sampling (pixel size) of the image for calibration.
    units : string, default "pix"
        Units used to display the magnitude of the vectors.
    plot_style : string, default "vector"
        Options are "vector", "colormap", "contour", "colorwheel",
        "polar_colorwheel". Note that "colorwheel" will automatically plot the
        colorbar as an angle. Also note that "polar_colorwheel" will
        automatically generate a 2D RGB (HSV) list of colors that match with
        the vector components (uv).
    overlay : bool, default True
        If set to True, the ``image`` will be plotting behind the arrows
    unit_vector : bool, default False
        Change the vectors magnitude to unit vectors for plotting purposes.
        Magnitude will still be displayed correctly for colormaps etc.
    vector_rep : str, default "magnitude"
        How the vectors are represented. This can be either their ``magnitude``
        or ``angle``. One may want to use ``angle`` when plotting a
        contour map, i.e. view the contours in terms of angles which can be
        useful for visualising regions of different polarisation.
    degrees : bool, default False
        Change between degrees and radian. Default is radian.
        If ``plot_style="colorwheel"``, then setting ``degrees=True``
        will convert the angle unit to degree from the default radians.
    angle_offset : float, default None
        If using ``vector_rep="angle"``   or ``plot_style="contour"``, this
        angle will rotate the vector angles displayed by the given amount.
        Useful when you want to offset the angle of the atom planes relative
        to the polarisation.
    save : string
        If set to ``save=None``, the image will not be saved.
    title : string, default ""
        Title of the plot.
    color : string, default "r"
        Color of the arrows when ``plot_style="vector"`` or ``"contour"``.
    cmap : matplotlib colormap, default ``"viridis"``
        Matplotlib cmap used for the vector arrows.
    alpha : float, default 1.0
        Transparency of the matplotlib ``cmap``. For ``plot_style="colormap"``
        and ``plot_style="colorwheel"``, this alpha applies to the vector
        arrows. For ``plot_style="contour"`` this alpha applies to the
        tricontourf map.
    image_cmap : matplotlib colormap, default 'gray'
        Matplotlib cmap that will be used for the overlay image.
    monitor_dpi : int, default 96
        The DPI of the monitor, generally 96 pixels. Used to scale the image
        so that large images render correctly. Use a smaller value to enlarge
        too-small images. ``monitor_dpi=None`` will ignore this param.
    no_axis_info :  bool, default True
        This will remove the x and y axis labels and ticks from the plot if set
        to True.
    invert_y_axis : bool, default True
        If set to true, this will flip the y axis, effectively setting the top
        left corner of the image as the (0, 0) origin, as in scanning electron
        microscopy images.
    ticks : colorbar ticks, default None
        None or list of ticks or Locator If None, ticks are determined
        automatically from the input.
    scalebar : bool or dict, default False
        Add a matplotlib-scalebar to the plot. If set to True the scalebar will
        appear similar to that given by Hyperspy's ``plot()`` function. A
        custom scalebar can be included as a dictionary and more custom
        options can be found in the matplotlib-scalebar package. See below
        for an example.
    antialiased : bool, default False
        Applies only to ``plot_style="contour"``. Essentially removes the
        border between regions in the tricontourf map.
    levels : int, default 20
        Number of Matplotlib tricontourf levels to be used.
    remove_vectors : bool, default False
        Applies only to ``plot_style="contour"``. If set to True, do not plot
        the vector arrows.
    quiver_units : string, default 'width'
        The units parameter from the matplotlib quiver function, not to be
        confused with the ``units`` parameter above for the image units.
    ax.quiver parameters
        See matplotlib's quiver function for the remaining parameters.

    Returns
    -------
    ax : Axes
        Matplotlib Axes object

    Examples
    --------
    >>> import temul.api as tml
    >>> from temul.dummy_data import get_polarisation_dummy_dataset
    >>> atom_lattice = get_polarisation_dummy_dataset()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeB = atom_lattice.sublattice_list[1]
    >>> sublatticeA.construct_zone_axes()
    >>> za0, za1 = sublatticeA.zones_axis_average_distances[0:2]
    >>> s_p = sublatticeA.get_polarization_from_second_sublattice(
    ...     za0, za1, sublatticeB, color='blue')
    >>> vector_list = s_p.metadata.vector_list
    >>> x, y = [i[0] for i in vector_list], [i[1] for i in vector_list]
    >>> u, v = [i[2] for i in vector_list], [i[3] for i in vector_list]

    vector plot with red arrows:

    >>> ax = tml.plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           unit_vector=False, save=None,
    ...                           plot_style='vector', color='r',
    ...                           overlay=False, title='Vector Arrows',
    ...                           monitor_dpi=50)

    vector plot with red arrows overlaid on the image:

    >>> ax = tml.plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           unit_vector=False, save=None,
    ...                           plot_style='vector', color='r',
    ...                           overlay=True, monitor_dpi=50)

    vector plot with colormap viridis:

    >>> ax = tml.plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           unit_vector=False, save=None,
    ...                           plot_style='colormap', monitor_dpi=50,
    ...                           overlay=False, cmap='viridis')

    vector plot with colormap viridis, with ``vector_rep="angle"``:

    >>> ax = tml.plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           unit_vector=False, save=None,
    ...                           plot_style='colormap', monitor_dpi=50,
    ...                           overlay=False, cmap='cet_colorwheel',
    ...                           vector_rep="angle", degrees=True)

    colormap arrows with sampling applied and with scalebar:

    >>> ax = tml.plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           sampling=3.0321, units='pm', monitor_dpi=50,
    ...                           unit_vector=False, plot_style='colormap',
    ...                           overlay=True, save=None, cmap='viridis',
    ...                           scalebar=True)

    vector plot with colormap viridis and unit vectors:

    >>> ax = tml.plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           unit_vector=True, save=None, monitor_dpi=50,
    ...                           plot_style='colormap', color='r',
    ...                           overlay=False, cmap='viridis')

    Change the vectors to unit vectors on a tricontourf map:

    >>> ax = tml.plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           unit_vector=True, plot_style='contour',
    ...                           overlay=False, pivot='middle', save=None,
    ...                           color='darkgray', cmap='plasma',
    ...                           monitor_dpi=50)

    Plot a partly transparent angle tricontourf map with vector arrows:

    >>> ax = tml.plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           unit_vector=False, plot_style='contour',
    ...                           overlay=True, pivot='middle', save=None,
    ...                           color='red', cmap='cet_colorwheel',
    ...                           monitor_dpi=50, remove_vectors=False,
    ...                           vector_rep="angle", alpha=0.5, levels=9,
    ...                           antialiased=True, degrees=True,
    ...                           ticks=[180, 90, 0, -90, -180])

    Plot a partly transparent angle tricontourf map with no vector arrows:

    >>> ax = tml.plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           unit_vector=True, plot_style='contour',
    ...                           overlay=True, pivot='middle', save=None,
    ...                           cmap='cet_colorwheel',
    ...                           monitor_dpi=50, remove_vectors=True,
    ...                           vector_rep="angle", alpha=0.5,
    ...                           antialiased=True, degrees=True)

    "colorwheel" plot of the vectors, useful for vortexes:

    >>> import colorcet as cc
    >>> ax = tml.plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           unit_vector=True, plot_style="colorwheel",
    ...                           vector_rep="angle",
    ...                           overlay=False, cmap=cc.cm.colorwheel,
    ...                           degrees=True, save=None, monitor_dpi=50,
    ...                           ticks=[180, 90, 0, -90, -180])

    "polar_colorwheel" plot showing a 2D polar color wheel:

    >>> ax = tml.plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           plot_style="polar_colorwheel",
    ...                           unit_vector=False, overlay=False,
    ...                           save=None, monitor_dpi=50)

    Plot with a custom scalebar, for example here we need it to be dark, see
    matplotlib-scalebar for more custom features.

    >>> scbar_dict = {"dx": 3.0321, "units": "pm", "location": "lower left",
    ...               "box_alpha":0.0, "color": "black", "scale_loc": "top"}
    >>> ax = tml.plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           sampling=3.0321, units='pm', monitor_dpi=50,
    ...                           unit_vector=False, plot_style='colormap',
    ...                           overlay=False, save=None, cmap='viridis',
    ...                           scalebar=scbar_dict)

    Plot a contourf for quadrant visualisation using a custom matplotlib cmap:

    >>> import temul.api as tml
    >>> from matplotlib.colors import from_levels_and_colors
    >>> zest = tml.hex_to_rgb(tml.color_palettes('zesty'))
    >>> zest.append(zest[0])  # make the -180 and 180 degree colour the same
    >>> expanded_zest = tml.expand_palette(zest, [1,2,2,2,1])
    >>> custom_cmap, _ = from_levels_and_colors(
    ...     levels=range(9), colors=tml.rgb_to_dec(expanded_zest))
    >>> ax = tml.plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           unit_vector=False, plot_style='contour',
    ...                           overlay=False, pivot='middle', save=None,
    ...                           cmap=custom_cmap, levels=9, monitor_dpi=50,
    ...                           vector_rep="angle", alpha=0.5, color='r',
    ...                           antialiased=True, degrees=True,
    ...                           ticks=[180, 90, 0, -90, -180])

    """

    if isinstance(image, np.ndarray):
        pass
    elif isinstance(image, hyperspy._signals.signal2d.Signal2D):
        sampling = image.axes_manager[-1].scale
        units = image.axes_manager[-1].units
        image = image.data
    else:
        raise ValueError("``image`` must be a 2D numpy array or 2D Hyperspy "
                         "Signal")

    u, v = np.array(u), np.array(v)

    if sampling is not None:
        u, v = u * sampling, v * sampling

    # get the magnitude or angle representation
    if vector_rep == "magnitude":
        vector_rep_val = get_vector_magnitudes(u, v)
    elif vector_rep == "angle":
        # -v because in STEM the origin is top left
        vector_rep_val = get_angles_from_uv(u, -v, degrees=degrees,
                                            angle_offset=angle_offset)

    vector_label = angle_label(
        vector_rep=vector_rep, units=units, degrees=degrees)

    if plot_style == "polar_colorwheel":
        color_list = get_polar_2d_colorwheel_color_list(u, -v)

    # change all vector magnitudes to the same size
    if unit_vector:
        u_norm = u / np.sqrt(u ** 2.0 + v ** 2.0)
        v_norm = v / np.sqrt(u ** 2.0 + v ** 2.0)
        u = u_norm
        v = v_norm

    # setting up norm and cmap for colorbar scalar mappable
    if vector_rep == "angle":
        if degrees:
            min_val, max_val = -180, 180 + 0.0001  # fix display issues
        elif not degrees:
            min_val, max_val = -np.pi, np.pi
    elif vector_rep == "magnitude":
        min_val = np.min(vector_rep_val)
        max_val = np.max(vector_rep_val) + 0.00000001
    norm = colors.Normalize(vmin=min_val, vmax=max_val)

    if monitor_dpi is not None:
        fig, ax = plt.subplots(figsize=[image.shape[1] / monitor_dpi,
                                        image.shape[0] / monitor_dpi])
    else:
        fig, ax = plt.subplots()
    ax.set_title(title, loc='left', fontsize=20)

    # plot_style options
    if plot_style == "vector":
        Q = ax.quiver(
            x, y, u, v, units=quiver_units, color=color, pivot=pivot,
            angles=angles, scale_units=scale_units, scale=scale,
            headwidth=headwidth, headlength=headlength, minshaft=minshaft,
            headaxislength=headaxislength, width=width, minlength=minlength)
        length = np.max(np.hypot(u, v))
        ax.quiverkey(Q, 0.8, 1.025, length,
                     label='{:.2E} {}'.format(Decimal(length), units),
                     labelpos='E', coordinates='axes')

    elif plot_style == "colormap":

        if cmap is None:
            cmap = 'viridis'
        ax.quiver(
            x, y, u, v, vector_rep_val, color=color, cmap=cmap, norm=norm,
            units=quiver_units, pivot=pivot, angles=angles,
            scale_units=scale_units, scale=scale, headwidth=headwidth,
            alpha=alpha, headlength=headlength, headaxislength=headaxislength,
            width=width, minshaft=minshaft, minlength=minlength)

        # norm = colors.Normalize(vmin=min_val, vmax=max_val)
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])
        # cbar = plt.colorbar(mappable=sm, fraction=0.046, pad=0.04,
        #                     drawedges=False)
        # cbar.set_ticks(ticks)
        # cbar.ax.set_ylabel(vector_label)

    elif plot_style == "colorwheel":

        if vector_rep != "angle":
            raise ValueError("`vector_rep`` must be set to 'angle' when "
                             "`plot_style`` is set to 'colorwheel'.")
        if cmap is None:
            cmap = cc.cm.colorwheel

        Q = ax.quiver(
            x, y, u, v, vector_rep_val, cmap=cmap, norm=norm, alpha=alpha,
            pivot=pivot, angles=angles, scale_units=scale_units,
            scale=scale, headwidth=headwidth, headlength=headlength,
            headaxislength=headaxislength, units=quiver_units, width=width,
            minshaft=minshaft, minlength=minlength)

    elif plot_style == "contour":

        if cmap is None:
            cmap = 'viridis'

        if isinstance(levels, list):
            levels_list = levels
        elif isinstance(levels, int):
            if vector_rep == "angle":
                levels_list = np.linspace(min_val, max_val, levels)
            elif vector_rep == "magnitude":
                levels_list = np.linspace(min_val, max_val, levels)

        plt.tricontourf(
            x, y, vector_rep_val, cmap=cmap, norm=norm, alpha=alpha,
            antialiased=antialiased, levels=levels_list)

        if not remove_vectors:
            ax.quiver(
                x, y, u, v, color=color, pivot=pivot, units=quiver_units,
                angles=angles, scale_units=scale_units,
                scale=scale, headwidth=headwidth, width=width,
                headlength=headlength, headaxislength=headaxislength,
                minshaft=minshaft, minlength=minlength)

        # cbar = plt.colorbar(mappable=contour_map, fraction=0.046, pad=0.04,
        #                     drawedges=False)
        # cbar.ax.tick_params(labelsize=14)
        # cbar.set_ticks(ticks)
        # cbar.ax.set_ylabel(vector_label, fontsize=14)

    elif plot_style == "polar_colorwheel":

        ax.quiver(
            x, y, u, v, color=color_list, pivot=pivot, units=quiver_units,
            angles=angles, scale_units=scale_units, scale=scale,
            headwidth=headwidth, width=width, headlength=headlength,
            headaxislength=headaxislength, minshaft=minshaft,
            minlength=minlength)

    else:
        raise NameError("The plot_style you have chosen is not available.")

    if invert_y_axis:
        ax.set(aspect='equal')
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)

    if overlay:
        plt.imshow(image, cmap=image_cmap)

    if no_axis_info:
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)

    if scalebar is True:
        scbar = ScaleBar(sampling, units, location="lower left", box_alpha=0.0,
                         color="white", scale_loc="top")
        plt.gca().add_artist(scbar)
    elif isinstance(scalebar, dict):
        scbar = ScaleBar(**scalebar)
        plt.gca().add_artist(scbar)

    # colorbars
    if (plot_style == "colormap" or plot_style == "colorwheel" or
            plot_style == "contour"):

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(mappable=sm, fraction=0.046, pad=0.04,
                            drawedges=False)
        cbar.set_ticks(ticks)
        cbar.ax.set_ylabel(vector_label)

    elif plot_style == "polar_colorwheel":
        ax2 = fig.add_subplot(444)
        _make_color_wheel(ax2, rotation=None)
        ax2.set_axis_off()

    # plt.tight_layout()
    if isinstance(save, str):
        plt.savefig(fname=save + '_' + plot_style + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)
    return ax


def get_angles_from_uv(u, v, degrees=False, angle_offset=None):
    """
    Calculate the angle of a vector given the uv components.

    Parameters
    ----------
    u,v  : list or 1D NumPy array
    degrees : bool, default False
        Change between degrees and radian. Default is radian.
    angle_offset : float, default None
        Rotate the angles by the given amount. The function assumes that if you
        set ``degrees=False`` then the provided ``angle_offset`` is in radians,
        and if you set ``degrees=True`` then the provided ``angle_offset`` is
        in degrees.

    Returns
    -------
    1D NumPy array
    """

    u_comp = np.array(u)
    v_comp = np.array(v).T

    vector_angles = np.arctan2(v_comp, u_comp)

    if angle_offset is not None:
        # all here is calculated in rad
        if degrees:
            # assumes angle_offset has also been given in degrees
            # so change to rad
            angle_offset = angle_offset * np.pi / 180
        vector_angles += angle_offset
        # refactor so that all angles still lie between -180 and 180
        a = vector_angles.copy()
        b = np.where(a > np.pi, a - (2 * np.pi), a)
        c = np.where(b < -np.pi, b + (2 * np.pi), b)
        vector_angles = c.copy()

    if degrees:
        vector_angles = vector_angles * 180 / np.pi

    return (vector_angles)


def get_vector_magnitudes(u, v, sampling=None):
    """
    Calculate the magnitude of a vector given the uv components.

    Parameters
    ----------
    u,v  : list or 1D NumPy array
    sampling : float, default None
        If sampling is set, the vector magnitudes (in pix) will be scaled
        by sampling (nm/pix).

    Returns
    -------
    1D NumPy array

    Examples
    --------
    >>> from temul.topotem.polarisation import get_vector_magnitudes
    >>> import numpy as np
    >>> u, v = [4,3,2,5,6], [8,5,2,1,1] # list input
    >>> vector_mags = get_vector_magnitudes(u,v)
    >>> u, v = np.array(u), np.array(v) # numpy input also works
    >>> vector_mags = get_vector_magnitudes(u,v)
    >>> sampling = 0.0321
    >>> vector_mags = get_vector_magnitudes(u,v, sampling=sampling)

    """

    # uv_vector_comp_list = [list(uv) for uv in uv_vector_comp]
    # u = [row[0] for row in uv_vector_comp_list]
    # v = [row[1] for row in uv_vector_comp_list]

    u_comp = np.array(u)
    v_comp = np.array(v).T

    vector_mags = (u_comp ** 2 + v_comp ** 2) ** 0.5

    if sampling is not None:
        vector_mags = vector_mags * sampling

    return (vector_mags)


def delete_atom_planes_from_sublattice(sublattice,
                                       zone_axis_index=0,
                                       atom_plane_tolerance=0.5,
                                       divisible_by=3,
                                       offset_from_zero=0,
                                       opposite=False):
    """
    Delete atom_planes from a zone axis. Can choose whether to delete
    every second, third etc. atom plane, and the offset from the zero index.

    Parameters
    ----------
    sublattice : Atomap Sublattice object
    zone_axis_index : int, default 0
        The zone axis you wish to specify. You are indexing
        sublattice.zones_axis_average_distances[zone_axis_index]
    atom_plane_tolerance : float, default 0.5
        float between 0.0 and 1.0. Closer to 1 means it will find more zones.
        See sublattice.construct_zone_axes() for more information.
    divisible_by : int, default 3
        If divisible_by is 2, every second atom_plane is deleted,
        If divisible_by is 4, every fourth atom_plane is deleted, etc.
    offset_from_zero : int, default 0
        The atom_plane from which you start deleting.
        If offset_from_zero is 4, the fourth atom_plane will be
        the first deleted.
    opposite : bool, default False
        If this is set to True, the atom_plane specified by divisible_by
        will be kept and all others deleted.

    Examples
    --------
    >>> from temul.topotem.polarisation import (
    ...     delete_atom_planes_from_sublattice)
    >>> import atomap.dummy_data as dd
    >>> atom_lattice = dd.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeA.construct_zone_axes()
    >>> zone_vec_list = sublatticeA.zones_axis_average_distances[0:2]

    Plot the planes visually using, it may take some time in large images:
    zones01_A = sublatticeA.get_all_atom_planes_by_zone_vector(zone_vec_list)
    zones01_A.plot()

    >>> delete_atom_planes_from_sublattice(
    ...         sublatticeA, zone_axis_index=0,
    ...         divisible_by=3, offset_from_zero=1)

    zones01_B = sublatticeA.get_all_atom_planes_by_zone_vector(zone_vec_list)
    zones01_A.plot()

    """
    sublattice.construct_zone_axes(atom_plane_tolerance=atom_plane_tolerance)

    zone_vec_needed = sublattice.zones_axis_average_distances[zone_axis_index]

    atom_plane_index_delete = []
    opposite_list = []
    for i, _ in enumerate(
            sublattice.atom_planes_by_zone_vector[zone_vec_needed]):
        if i % divisible_by == 0:
            atom_plane_index_delete.append(i)
        if opposite:
            opposite_list.append(i)
    # print(atom_plane_index_delete)
    # print(opposite_list)
    # atom_plane_index_delete = [0, 3, 6, 9]
    # offset_from_zero = 2
    atom_plane_index_delete = [offset_from_zero +
                               index for index in atom_plane_index_delete]
    atom_plane_index_delete = [
        index for index in atom_plane_index_delete
        if index < len(sublattice.atom_planes_by_zone_vector[zone_vec_needed])]

    if opposite:
        opposite_list = [
            index for index in opposite_list
            if index not in atom_plane_index_delete]
        atom_plane_index_delete = opposite_list
    # reversal needed because first it will delete 0, then 1 will become 0.
    # Then it will delete 3, which is the wrong one! (should have been 2)
    atom_plane_index_delete.sort(reverse=True)
    # print(atom_plane_index_delete)
    # print(opposite_list)
    for i in atom_plane_index_delete:
        del sublattice.atom_planes_by_zone_vector[zone_vec_needed][i]


def _fit_line_clusters(arr, n, second_fit_rigid=True, plot=False):
    """
    Fits the data in an array to two straight lines using the
    first ``n`` and second (last) ``n`` array value pairs.

    The slope of the first fitting will be used for the
    second fitting. Setting ``second_fit_rigid`` = False will reverse this
    behaviour.

    Parameters
    ----------
    arr : 2D array-like
        Array-like (e.g., NumPy 2D array) in the form [[x1, y1], [x2, y2]...]
    n : int
        The number of arr value pairs used at the beginning and end of the arr
        to fit a straight line.
    second_fit_rigid : bool, default True
        Used to decide whether the first or second fitting's slope will be
        rigid during fitting. With ``second_fit_rigid`` = True, the slope of
        the second fitting will be defined as the slope as the first fitting.
        The y-intercept is free to move.
    plot : bool, default False
        Whether to plot the arr data, first and second fitting, and the
        line constructed halfway between the two.

    Returns
    -------
    x, line_C, m_fit
        x arr values, halfway fitting line, slope of fit
    x, line_C, m_fit, line_A, line_B, c_fit, c_fit_rigid
        If plot is True, the above will be returned along with the first line
        fitting, second line fitting, first y-intercept, second y-intercept.

    See Also
    --------
    get_xyuv_from_line_fit : uses _fit_line_clusters to get xyuv from arr
    atom_deviation_from_straight_line_fit : Gets xyuv for Sublattice object

    Examples
    --------
    >>> arr = np.array([[1, 2], [2, 2], [3, 2], [4, 2], [5, 2.05],
    ...                 [6, 1], [7, 1], [8, 1], [9, 1], [10, 0.75]])
    >>> fittings = _fit_line_clusters(
    ...     arr, n=5, second_fit_rigid=True, plot=False)

    Use the lower (second) cluster to fit the data, making the first cluster
    rigid

    >>> fittings = _fit_line_clusters(
    ...     arr, n=5, second_fit_rigid=False, plot=False)

    """
    x = arr[:, 0]
    y = arr[:, 1]

    x_i = x[0:n]
    y_i = y[0:n]

    x_f = x[-n:len(x)]
    y_f = y[-n:len(y)]

    if second_fit_rigid:
        m_fit, c_fit = np.polyfit(x_i, y_i, 1)
        c_fit_rigid = np.mean(y_f - m_fit * x_f)
    elif not second_fit_rigid:
        m_fit, c_fit = np.polyfit(x_f, y_f, 1)
        c_fit_rigid = np.mean(y_i - m_fit * x_i)

    # use the first slope to get the intercept of the second
    line_A = m_fit * x + c_fit
    line_B = m_fit * x + c_fit_rigid
    c_halfway = (c_fit + c_fit_rigid) / 2
    line_C = m_fit * x + c_halfway

    if plot:
        plt.figure()
        plt.scatter(x, y)
        plt.plot(x, line_A, c='orange', label='first fit')
        plt.plot(x, line_B, c='green', label='second fit')
        plt.plot(x, line_C, c='k', label='halfway fit')
        plt.legend(loc='upper right')
        plt.show()

    if plot:
        return (x, line_C, m_fit, line_A, line_B, c_fit, c_fit_rigid)
    else:
        return (x, line_C, m_fit)


def get_xyuv_from_line_fit(arr, n, second_fit_rigid=True, plot=False):
    """
    Fits the data in an array to two straight lines using the
    first ``n`` and second (last) ``n`` array value pairs.
    Computes the distance of
    each array value pair from the line halfway between the two fitted lines.

    The slope of the first fitting will be used for the
    second fitting. Setting ``second_fit_rigid`` = False will reverse this
    behaviour.

    Parameters
    ----------
    arr : 2D array-like
        Array-like (e.g., NumPy 2D array) in the form [[x1, y1], [x2, y2]...]
    n : int
        The number of arr value pairs used at the beginning and end of the arr
        to fit a straight line.
    second_fit_rigid : bool, default True
        Used to decide whether the first or second fitting's slope will be
        rigid during fitting. With ``second_fit_rigid=True``, the slope of the
        second fitting will be defined as the slope as the first fitting. The
        y-intercept is free to move.
    plot : bool, default False
        Whether to plot the arr data, first and second fitting, and the
        line constructed halfway between the two.

    Returns
    -------
    x, y, u, v : lists of equal length
        x, y are the original arr coordinates. u, v are the vector components
        pointing towards the halfway line from the arr coordinates.
        These can be input to ``plot_polarisation_vectors()`` for
        visualisation.

    See Also
    --------
    atom_deviation_from_straight_line_fit : Gets xyuv for Sublattice object

    Examples
    --------
    >>> arr = np.array([[1, 2], [2, 2], [3, 2], [4, 2], [5, 2.05],
    ...                 [6, 1], [7, 1], [8, 1], [9, 1], [10, 0.75]])
    >>> x, y, u, v = get_xyuv_from_line_fit(
    ...     arr, n=5, second_fit_rigid=True, plot=False)

    Use the lower (second) cluster to fit the data, making the first cluster
    rigid

    >>> x, y, u, v = get_xyuv_from_line_fit(
    ...     arr, n=5, second_fit_rigid=False, plot=False)

    """
    arr = np.asarray(arr)
    fittings = _fit_line_clusters(arr=arr, n=n,
                                  second_fit_rigid=second_fit_rigid, plot=plot)

    x, line_C, m_fit = fittings[0], fittings[1], fittings[2]

    slope_neg_inv = -(1 / m_fit)
    angle = np.arctan(slope_neg_inv)  # * (180/np.pi)

    # start xy coord for straight line fit
    p1 = np.array((x[0], line_C[0]), ndmin=2)
    # end xy coord for straight line fit
    p2 = np.array((x[-1], line_C[-1]), ndmin=2)

    x_list = []
    y_list = []
    u_list = []
    v_list = []
    for original_pos in arr:
        distance = np.cross(p2 - p1, original_pos -
                            p1) / np.linalg.norm(p2 - p1)
        distance = float(distance)
        u = distance * np.cos(angle)
        v = distance * np.sin(angle)
        x_list.append(original_pos[0])
        y_list.append(original_pos[1])
        u_list.append(u)
        v_list.append(v)

    return (x_list, y_list, u_list, v_list)


def atom_deviation_from_straight_line_fit(
        sublattice, axis_number, n, second_fit_rigid=True, plot=False):
    """
    Fits the atomic columns in an atom plane to two straight lines using the
    first ``n`` and second (last) ``n`` atomic columns. Computes the distance
    of each atomic column from the line halfway between the two fitted lines,
    as described in [1]_. This is done for every sublattice atom plane along
    the chosen ``axis_number``.

    The slope of the first fitting will be used for the
    second fitting. Setting ``second_fit_rigid`` = False will reverse this
    behaviour.

    Parameters
    ----------
    sublattice : Atomap Sublattice object
    axis_number : int
        The index of the zone axis (translation symmetry) found by the Atomap
        function ``construct_zone_axes()``. For sublattices with heavily
        deviating atomic columns, you may need to use
        sublattice.construct_zone_axes(atom_plane_tolerance=1).
    n : int
        The number of columns used at the beginning and end of each atom plane
        to fit a straight line.
    second_fit_rigid : bool, default True
        Used to decide whether the first or second fitting's slope will be
        rigid during fitting. With ``second_fit_rigid=True``, the slope of the
        second fitting will be defined as the slope as the first fitting. The
        y-intercept is free to move.
    plot : bool, default False
        Whether to plot the atom plane data, first and second fitting, and the
        line constructed halfway between the two.

    Returns
    -------
    x, y, u, v : lists of equal length
        x, y are the original atom position coordinates
        ``sublattice.x_position`` and ``sublattice.y_position`` for
        the coordinates included in the chosen
        ``axis_number``. u, v are the polarisation vector components pointing
        towards the halfway line from the atom position coordinates.
        These can be input to ``plot_polarisation_vectors()``
        for visualisation.

    See Also
    --------
    get_xyuv_from_line_fit : uses ``_fit_line_clusters`` to get xyuv from arr

    References
    ----------
    .. [1] Reference: Julie Gonnissen, Dmitry Batuk, Guillaume F. Nataf,
           Lewys Jones, Artem M. Abakumov, Sandra Van Aert, Dominique
           Schryvers, Ekhard K. H. Salje, Direct Observation of Ferroelectric
           Domain Walls in LiNbO3: Wall‐Meanders, Kinks, and Local Electric
           Charges, 26, 42, 2016, DOI: 10.1002/adfm.201603489

    Examples
    --------
    >>> import temul.api as tml
    >>> import temul.dummy_data as dd
    >>> sublattice = dd.get_polarised_single_sublattice()
    >>> sublattice.construct_zone_axes(atom_plane_tolerance=1)

    Choose ``n`: how many atom columns should be used to fit the line on each
    side of the atom planes. If ``n`` is too large, the fitting will appear
    incorrect.

    >>> n = 5
    >>> x, y, u, v = tml.atom_deviation_from_straight_line_fit(
    ...     sublattice, 0, n)
    >>> ax = tml.plot_polarisation_vectors(x, y, u, v, image=sublattice.image,
    ...                               unit_vector=False, save=None,
    ...                               plot_style='vector', color='r',
    ...                               overlay=True, monitor_dpi=50)

    Plot with angle and up/down. Note that the data ranges from -90 to +90
    degrees, so the appropriate diverging cmap should be chosen.

    >>> ax = tml.plot_polarisation_vectors(x, y, u, v, image=sublattice.image,
    ...                       vector_rep='angle', save=None, degrees=True,
    ...                       plot_style='colormap', cmap='cet_coolwarm',
    ...                       overlay=True, monitor_dpi=50)

    Let's look at some rotated data

    >>> sublattice = dd.get_polarised_single_sublattice_rotated(
    ...     image_noise=True, rotation=45)
    >>> sublattice.construct_zone_axes(atom_plane_tolerance=0.9)
    >>> # sublattice.plot_planes()
    >>> n = 3  # plot the sublattice to see why 3 is suitable here!
    >>> x, y, u, v = tml.atom_deviation_from_straight_line_fit(
    ...     sublattice, 0, n)
    >>> ax = tml.plot_polarisation_vectors(x, y, u, v, image=sublattice.image,
    ...                       vector_rep='angle', save=None, degrees=True,
    ...                       plot_style='colormap', cmap='cet_coolwarm',
    ...                       overlay=True, monitor_dpi=50)

    """

    if sublattice.zones_axis_average_distances is None:
        raise Exception(
            "zones_axis_average_distances is empty. "
            "Has sublattice.construct_zone_axes() been run? You may need to "
            "use sublattice.construct_zone_axes(atom_plane_tolerance=1).")
    else:
        zon_vec_needed = sublattice.zones_axis_average_distances[axis_number]

    x_list = []
    y_list = []
    u_list = []
    v_list = []
    for i, atom_plane in enumerate(sublattice.atom_plane_list):

        if sublattice.atom_plane_list[i].zone_vector == zon_vec_needed:
            arr = np.array([atom_plane.get_x_position_list(),
                            atom_plane.get_y_position_list()]).T

            x, y, u, v = get_xyuv_from_line_fit(
                arr=arr, n=n, second_fit_rigid=second_fit_rigid, plot=plot)
            x_list.extend(x)
            y_list.extend(y)
            u_list.extend(u)
            v_list.extend(v)

    return (x_list, y_list, u_list, v_list)


def _truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def full_atom_plane_deviation_from_straight_line_fit(sublattice,
                                                     axis_number: int = 0,
                                                     save: str = ''):
    """
    Fit the atoms in an atom plane to a straight line and find the deviation
    of each atom position from that straight line fit.
    To plot all zones see ``plot_atom_deviation_from_all_zone_axes()``.

    Parameters
    ----------
    sublattice : Atomap Sublattice object
    axis_number : int, default 0
        The index of the zone axis (translation symmetry) found by the Atomap
        function ``construct_zone_axes()``.
    save : string, default ''
        If set to ``save=None``, the array will not be saved.

    Returns
    -------
    Four lists: x, y, u, and v where x,y are the original atom position
    coordinates (simply sublattice.x_position, sublattice.y_position) and
    u,v are the polarisation vector components pointing to the new coordinate.
    These can be input to ``plot_polarisation_vectors()`` for visualisation.

    Examples
    --------
    >>> import atomap.api as am
    >>> from temul.topotem.polarisation import (
    ...     full_atom_plane_deviation_from_straight_line_fit,
    ...     plot_polarisation_vectors)
    >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeA.find_nearest_neighbors()
    >>> _ = sublatticeA.refine_atom_positions_using_center_of_mass()
    >>> sublatticeA.construct_zone_axes()
    >>> x,y,u,v = full_atom_plane_deviation_from_straight_line_fit(
    ...     sublatticeA, save=None)
    >>> ax = plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           unit_vector=False, save=None, monitor_dpi=50)

    """

    if sublattice.zones_axis_average_distances is None:
        raise Exception(
            "zones_axis_average_distances is empty. "
            "Has sublattice.construct_zone_axes() been run?")
    else:
        zon_vec_needed = sublattice.zones_axis_average_distances[axis_number]
    original_atom_pos_list = []
    new_atom_pos_list = []
    new_atom_diff_list = []

    # this loop creates two arrays.
    # the original array contains all the original atom positions
    # the new array contains all the xy positions on the fitted straight
    # lines the new array positions are the point at which the original
    # position is perpendicular to the fitted line.
    for i, atom_plane in enumerate(sublattice.atom_plane_list):

        if sublattice.atom_plane_list[i].zone_vector == zon_vec_needed:
            original_atoms_list = []
            for atom_pos in sublattice.atom_plane_list[i].atom_list:
                original_atoms_list.append(
                    [atom_pos.pixel_x, atom_pos.pixel_y])

            original_atoms_array = np.array(original_atoms_list)

            slope, intercept = scipy.polyfit(
                original_atoms_array[:, 0], original_atoms_array[:, 1], 1)

            slope_neg_inv = -(1 / slope)
            angle = np.arctan(slope_neg_inv)  # * (180/np.pi)

            x1 = atom_plane.start_atom.pixel_x
            y1 = slope * x1 + intercept
            x2 = atom_plane.end_atom.pixel_x
            y2 = slope * x2 + intercept

            p1 = np.array((x1, y1), ndmin=2)
            # end xy coord for straight line fit
            p2 = np.array((x2, y2), ndmin=2)

            atoms_on_plane_list = []
            atom_dist_diff_list = []
            # original_atom position, point an arrow towards it by using
            # original_atom_pos_array and new_atom_diff_array,
            # or away using new_atom_pos_array and -new_atom_diff_array
            for original_atom in original_atoms_array:
                distance = np.cross(p2 - p1, original_atom -
                                    p1) / np.linalg.norm(p2 - p1)
                distance = float(distance)
                x_diff = distance * np.cos(angle)
                y_diff = distance * np.sin(angle)

                x_on_plane = original_atom[0] + x_diff
                y_on_plane = original_atom[1] + y_diff

                atoms_on_plane_list.append([x_on_plane, y_on_plane])
                atom_dist_diff_list.append([x_diff, y_diff])
                # atoms_not_on_plane_list.append([original_atom])

            original_atom_pos_list.extend(original_atoms_list)
            new_atom_pos_list.extend(atoms_on_plane_list)
            new_atom_diff_list.extend(atom_dist_diff_list)

    original_atom_pos_array = np.array(original_atom_pos_list)
    new_atom_pos_array = np.array(new_atom_pos_list)
    distance_diff_array = np.array(new_atom_diff_list)

    if save is not None:
        np.save(save + '_sublattice_xy', original_atom_pos_array)
        np.save(save + '_new_atom_positions_xy', new_atom_pos_array)
        np.save(save + '_vector_uv', distance_diff_array)

    # this is the difference between the original position and the point on
    # the fitted atom plane line. To get the actual shift direction, just
    # use -new_atom_diff_array. (negative of it!)

    x = [row[0] for row in original_atom_pos_list]
    y = [row[1] for row in original_atom_pos_list]
    u = [row[0] for row in new_atom_diff_list]
    v = [row[1] for row in new_atom_diff_list]

    return (x, y, u, v)


# need to add the truncated colormap version: divergent plot.
def plot_atom_deviation_from_all_zone_axes(
        sublattice, image=None, sampling=None, units='pix',
        plot_style='vector', overlay=True, unit_vector=False, degrees=False,
        save='atom_deviation', title="", color='yellow', cmap=None,
        pivot='middle', angles='xy', scale_units='xy', scale=None,
        headwidth=3.0, headlength=5.0, headaxislength=4.5, monitor_dpi=96,
        no_axis_info=True, scalebar=False):
    """
    Plot the atom deviation from a straight line fit for all zone axes
    constructed by an Atomap sublattice object.

    Parameters
    ----------
    sublattice : Atomap Sublattice object

    For all other parameters see ``plot_polarisation_vectors()``.

    Examples
    --------
    >>> import atomap.dummy_data as dd
    >>> import temul.api as tml
    >>> atom_lattice = dd.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeA.find_nearest_neighbors()
    >>> _ = sublatticeA.refine_atom_positions_using_center_of_mass()
    >>> sublatticeA.construct_zone_axes()
    >>> tml.plot_atom_deviation_from_all_zone_axes(
    ...     sublatticeA, save=None)

    """

    if image is None:
        image = sublattice.image

    if sublattice.zones_axis_average_distances is None:
        raise Exception(
            "zones_axis_average_distances is empty. "
            "Has sublattice.construct_zone_axes() been run?")
    else:
        pass

    for axis_number in range(len(sublattice.zones_axis_average_distances)):
        x, y, u, v = full_atom_plane_deviation_from_straight_line_fit(
            sublattice=sublattice, axis_number=axis_number,
            save=None)

        plot_polarisation_vectors(
            u=u, v=v, x=x, y=y, image=image, sampling=sampling, units=units,
            plot_style=plot_style, overlay=overlay, unit_vector=unit_vector,
            degrees=degrees, save=save, title=title, color=color, cmap=cmap,
            pivot=pivot, angles=angles, scale_units=scale_units, scale=scale,
            headwidth=headwidth, headlength=headlength,
            headaxislength=headaxislength, monitor_dpi=monitor_dpi,
            no_axis_info=no_axis_info, scalebar=scalebar)


def combine_atom_deviations_from_zone_axes(
        sublattice, image=None, axes=None, sampling=None, units='pix',
        plot_style='vector', overlay=True, unit_vector=False, degrees=False,
        save='atom_deviation', title="", color='yellow', cmap=None,
        pivot='middle', angles='xy', scale_units='xy', scale=None,
        headwidth=3.0, headlength=5.0, headaxislength=4.5, monitor_dpi=96,
        no_axis_info=True, scalebar=False):
    """
    Combine the atom deviations of each atom for all zone axes.
    Good for plotting vortexes and seeing the total deviation from a
    perfect structure.

    Parameters
    ----------
    sublattice : Atomap Sublattice object

    For the remaining parameters see ``plot_polarisation_vectors()``.

    Returns
    -------
    Four lists: x, y, u, and v where x,y are the original atom position
    coordinates (simply sublattice.x_position, sublattice.y_position) and
    u,v are the polarisation vector components pointing to the new coordinate.
    These can be input to ``plot_polarisation_vectors()`` for visualisation.

    Examples
    --------

    >>> import atomap.api as am
    >>> from temul.topotem.polarisation import (plot_polarisation_vectors,
    ...     combine_atom_deviations_from_zone_axes)
    >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeA.find_nearest_neighbors()
    >>> _ = sublatticeA.refine_atom_positions_using_center_of_mass()
    >>> sublatticeA.construct_zone_axes()
    >>> x,y,u,v = combine_atom_deviations_from_zone_axes(
    ...     sublatticeA, save=None)
    >>> ax = plot_polarisation_vectors(x, y, u, v, save=None,
    ...     image=sublatticeA.image)

    You can also choose the axes:

    >>> x,y,u,v = combine_atom_deviations_from_zone_axes(
    ...     sublatticeA, axes=[0,1], save=None)
    >>> ax = plot_polarisation_vectors(x, y, u, v, save=None,
    ...     image=sublatticeA.image)

    """

    all_atoms_in_planes_xy = []
    all_atoms_in_planes_uv = []

    if sublattice.zones_axis_average_distances is None:
        raise Exception(
            "zones_axis_average_distances is empty. "
            "Has sublattice.construct_zone_axes() been run?")
    else:
        pass

    if axes is None:
        axes_list = range(len(sublattice.zones_axis_average_distances))
    else:
        axes_list = axes

    for axis_number in axes_list:
        x, y, u, v = full_atom_plane_deviation_from_straight_line_fit(
            sublattice=sublattice, axis_number=axis_number,
            save=None)

        xy_array = np.array([x, y]).T
        atoms_in_plane_xy = [list(i) for i in xy_array]
        uv_array = np.array([u, v]).T
        atoms_in_plane_uv = [list(i) for i in uv_array]

        all_atoms_in_planes_xy.extend(atoms_in_plane_xy)
        all_atoms_in_planes_uv.extend(atoms_in_plane_uv)

    sublattice_xy = []
    for atom in sublattice.atom_list:
        sublattice_xy.append([atom.pixel_x, atom.pixel_y])

    atoms_not_found = []
    combined_vectors = []
    for atom_xy in sublattice_xy:
        individual_vectors = []
        for atom_along_plane_xy, atom_along_plane_uv in zip(
                all_atoms_in_planes_xy, all_atoms_in_planes_uv):

            if atom_xy == atom_along_plane_xy:
                individual_vectors.append(atom_along_plane_uv)

        if len(individual_vectors) != 0:
            calc_combined_vectors = list(sum(np.array(individual_vectors)))
            combined_vectors.append(calc_combined_vectors)
        else:
            atoms_not_found.append(atom_xy)

    if len(atoms_not_found) != 0:
        print("This sublattice_xy atom isn't included in the "
              f"axes given, removing atoms: {atoms_not_found}.")
    for atom in atoms_not_found:
        sublattice_xy.remove(atom)

    if len(sublattice_xy) != len(combined_vectors):
        raise ValueError("len(sublattice_xy) != len(combined_vectors)")

    x = [row[0] for row in sublattice_xy]
    y = [row[1] for row in sublattice_xy]
    u = [row[0] for row in combined_vectors]
    v = [row[1] for row in combined_vectors]

    if save is not None:
        np.save(save + '_sublattice_xy', np.array(sublattice_xy))
        np.save(save + '_vector_uv', np.array(combined_vectors))

        if image is None:
            image = sublattice.image

        plot_polarisation_vectors(
            u=u, v=v, x=x, y=y, image=image, sampling=sampling, units=units,
            plot_style=plot_style, overlay=overlay, unit_vector=unit_vector,
            degrees=degrees, save=save, title=title, color=color, cmap=cmap,
            pivot=pivot, angles=angles, scale_units=scale_units, scale=scale,
            headwidth=headwidth, headlength=headlength,
            headaxislength=headaxislength, monitor_dpi=monitor_dpi,
            no_axis_info=no_axis_info, scalebar=scalebar)

    return (x, y, u, v)


def get_divide_into(sublattice, averaging_by, sampling,
                    zone_axis_index_A, zone_axis_index_B):
    """
    Calculate the ``divide_into`` required to get an averaging of
    ``averaging_by``. ``divide_into`` can then be used in
    temul.topotem.polarisation.get_average_polarisation_in_regions.
    Also finds unit cell size and the number of unit cells in the (square)
    image along the x axis.

    Parameters
    ----------
    sublattice : Atomap Sublattice
    averaging_by : int or float
        How many unit cells should be averaged. If ``averaging_by=2``, 2x2 unit
        cells will be averaged when passing ``divide_into`` to
        temul.topotem.polarisation.get_average_polarisation_in_regions.
    sampling : float
        Pixel sampling of the image for calibration.
    zone_axis_index_A, zone_axis_index_B : int
        Sublattice zone axis indices which should represent the sides of the
        unit cell.

    Returns
    -------
    divide_into, unit_cell_size, num_unit_cells

    Examples
    --------

    >>> from temul.topotem.polarisation import get_divide_into
    >>> from atomap.dummy_data import get_simple_cubic_sublattice
    >>> sublattice = get_simple_cubic_sublattice()
    >>> sublattice.construct_zone_axes()
    >>> cell_info = get_divide_into(sublattice, averaging_by=2, sampling=0.1,
    ...                 zone_axis_index_A=0, zone_axis_index_B=1)
    >>> divide_into = cell_info[0]
    >>> unit_cell_size = cell_info[1]
    >>> num_unit_cells = cell_info[2]
    >>> sublattice.plot()  # You can count the unit cells to check

    """

    if sublattice.zones_axis_average_distances is None:
        raise Exception(
            "zones_axis_average_distances is empty. "
            "Has sublattice.construct_zone_axes() been run?")

    zone_vector_index_list = sublattice._get_zone_vector_index_list(
        zone_vector_list=None)

    # zone A
    zone_index_A, zone_vector_A = zone_vector_index_list[zone_axis_index_A]
    _, _, xy_dist_A = \
        sublattice.get_atom_distance_list_from_zone_vector(zone_vector_A)

    # zone B
    zone_index_B, zone_vector_B = zone_vector_index_list[zone_axis_index_B]
    _, _, xy_dist_B = \
        sublattice.get_atom_distance_list_from_zone_vector(zone_vector_B)

    # average unit cell size in zone A and zone B
    p_A = np.mean(xy_dist_A) * sampling
    p_B = np.mean(xy_dist_B) * sampling

    unit_cell_size = (p_A + p_B) / 2
    image_size_x = sublattice.signal.axes_manager[0].size * sampling

    num_unit_cells = image_size_x / unit_cell_size

    divide_into = num_unit_cells / averaging_by

    return (divide_into, unit_cell_size, num_unit_cells)


def get_average_polarisation_in_regions(x, y, u, v, image, divide_into=8):
    """
    This function splits the image into the given number of regions and
    averages the polarisation of each region.

    Parameters
    ----------
    x, y : list or 1D NumPy array
        xy coordinates on the image
    u, v : list or 1D NumPy array
        uv vector components
    image : 2D NumPy array
    divide_into : int, default 8
        The number used to divide the image up. If 8, then the image will be
        split into an 8x8 grid.

    Returns
    -------
    x_new, y_new, u_new, v_new : lists of equal length
        x_new and y_new are the central coordinates of the divided regions.
        u_new and v_new are the averaged polarisation vectors.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import atomap.api as am
    >>> from temul.topotem.polarisation import (
    ...    combine_atom_deviations_from_zone_axes,
    ...    plot_polarisation_vectors, get_average_polarisation_in_regions)
    >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeA.find_nearest_neighbors()
    >>> _ = sublatticeA.refine_atom_positions_using_center_of_mass()
    >>> sublatticeA.construct_zone_axes()

    Get and plot the original polarisation vectors:

    >>> x, y, u, v = combine_atom_deviations_from_zone_axes(sublatticeA,
    ...     save=None)
    >>> ax = plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                   unit_vector=False, save=None,
    ...                   plot_style='vector', color='r',
    ...                   overlay=False, title='Actual Vector Arrows',
    ...                   monitor_dpi=50)

    Get and plot the new, averaged polarisation vectors

    >>> x_new, y_new, u_new, v_new = get_average_polarisation_in_regions(
    ...     x, y, u, v, image=sublatticeA.image, divide_into=8)
    >>> ax = plot_polarisation_vectors(x_new, y_new, u_new, v_new,
    ...         monitor_dpi=50, image=sublatticeA.image, save=None, color='r',
    ...         overlay=False, title='Averaged Vector Arrows')

    """

    if divide_into >= np.sqrt(len(x)):
        raise ValueError(
            "divide_into ({}) cannot be greater than the number of "
            "vector coordinates in each dimension ({})".format(
                divide_into, np.sqrt(len(x))))

    # divide the image into sections
    image_x_max, image_y_max = image.shape[-1], image.shape[-2]
    x_region_length = image_x_max // divide_into
    y_region_length = image_y_max // divide_into

    all_x_region_lengths = []
    all_y_region_lengths = []
    for i in range(divide_into):
        all_x_region_lengths.append(
            [i * x_region_length, (i + 1) * x_region_length])
        all_y_region_lengths.append(
            [i * y_region_length, (i + 1) * y_region_length])

    # get the new x, y coords
    x_new, y_new = [], []
    for x_length in all_x_region_lengths:
        for y_length in all_y_region_lengths:
            x_new.append(x_length[1] - ((x_length[1] - x_length[0]) / 2))
            y_new.append(y_length[1] - ((y_length[1] - y_length[0]) / 2))

    # get the new averaged u, v components
    u_new, v_new = [], []
    for x_length in all_x_region_lengths:
        for y_length in all_y_region_lengths:
            u_area, v_area = [], []
            for (x_i, y_i, u_i, v_i) in zip(x, y, u, v):
                if x_length[0] < x_i < x_length[1] and \
                        y_length[0] < y_i < y_length[1]:
                    u_area.append(u_i)
                    v_area.append(v_i)

            u_mean = np.mean(np.array(u_area))
            u_new.append(u_mean)
            v_mean = np.mean(np.array(v_area))
            v_new.append(v_mean)

    return (x_new, y_new, u_new, v_new)


def get_average_polarisation_in_regions_square(x, y, u, v, image,
                                               divide_into=4):
    """
    Same as ``get_average_polarisation_in_regions()`` but works for non-square
    (rectangular) images.

    Parameters
    ----------
    x, y : list or 1D NumPy array
        xy coordinates on the image
    u, v : list or 1D NumPy array
        uv vector components
    image : 2D NumPy array
    divide_into : int, default 8
        The number used to divide the image up. If 8, then the image will be
        split into an 8x8 grid.

    Returns
    -------
    x_new, y_new, u_new, v_new : lists of equal length
        x_new and y_new are the central coordinates of the divided regions.
        u_new and v_new are the averaged polarisation vectors.

    Examples
    --------
    >>> import atomap.api as am
    >>> from temul.topotem.polarisation import (
    ...     combine_atom_deviations_from_zone_axes, plot_polarisation_vectors,
    ...     get_average_polarisation_in_regions_square)
    >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeA.find_nearest_neighbors()
    >>> _ = sublatticeA.refine_atom_positions_using_center_of_mass()
    >>> sublatticeA.construct_zone_axes()

    Get and plot the original polarisation vectors of a non-square image

    >>> image = sublatticeA.image[0:200]
    >>> x, y, u, v = combine_atom_deviations_from_zone_axes(sublatticeA,
    ...     save=None)
    >>> ax = plot_polarisation_vectors(x, y, u, v, image=image, save=None,
    ...                   color='r', overlay=False, monitor_dpi=50,
    ...                   title='Actual Vector Arrows')

    Get and plot the new, averaged polarisation vectors for a non-square image

    >>> coords = get_average_polarisation_in_regions_square(
    ...     x, y, u, v, image=image, divide_into=8)
    >>> x_new, y_new, u_new, v_new = coords
    >>> ax = plot_polarisation_vectors(x_new, y_new, u_new, v_new, image=image,
    ...                   color='r', overlay=False, monitor_dpi=50,
    ...                   title='Averaged Vector Arrows', save=None)
    """

    if divide_into >= np.sqrt(len(x)):
        raise ValueError(
            "divide_into ({}) cannot be greater than the number of "
            "vector coordinates in each dimension ({})".format(
                divide_into, np.sqrt(len(x))))

    # divide the image into sections
    image_x_max, image_y_max = image.shape[-1], image.shape[-2]

    region_length = image_x_max // divide_into
    divide_other_into = image_y_max // region_length
    # other_region_length = image_other_axis_max // divide_other_into

    all_x_region_lengths = []
    all_y_region_lengths = []

    for i in range(divide_into):
        all_x_region_lengths.append(
            [i * region_length, (i + 1) * region_length])

    for i in range(divide_other_into):
        all_y_region_lengths.append(
            [i * region_length, (i + 1) * region_length])

    # get the new x, y coords
    x_new, y_new = [], []
    for x_length in all_x_region_lengths:
        for y_length in all_y_region_lengths:
            x_new.append(x_length[1] - ((x_length[1] - x_length[0]) / 2))
            y_new.append(y_length[1] - ((y_length[1] - y_length[0]) / 2))

    # get the new averaged u, v components
    u_new, v_new = [], []
    for x_length in all_x_region_lengths:
        for y_length in all_y_region_lengths:
            u_area, v_area = [], []
            for (x_i, y_i, u_i, v_i) in zip(x, y, u, v):
                if x_length[0] < x_i < x_length[1] and \
                        y_length[0] < y_i < y_length[1]:
                    u_area.append(u_i)
                    v_area.append(v_i)

            u_mean = np.mean(np.array(u_area))
            u_new.append(u_mean)
            v_mean = np.mean(np.array(v_area))
            v_new.append(v_mean)

    return (x_new, y_new, u_new, v_new)


def get_strain_map(sublattice, zone_axis_index, theoretical_value,
                   sampling=None, units='pix', vmin=None, vmax=None,
                   cmap='inferno', title='Strain Map', plot=False,
                   filename=None, return_x_y_z=False, **kwargs):
    """
    Calculate the strain across a zone axis of a sublattice.

    Parameters
    ----------
    sublattice : Atomap Sublattice object
    zone_axis_index : int
        The zone axis you wish to specify. You are indexing
        ``sublattice.zones_axis_average_distances[zone_axis_index]``.
    theoretical_value : float
        The theoretical separation between the atoms across (not along) the
        specified zone.
    sampling : float, default None
        Pixel sampling of the image for calibration.
    units : string, default "pix"
        Units of the sampling.
    vmin, vmax, cmap : see Matplotlib for details
    title : string, default "Strain Map"
    plot : bool
        Set to true if the figure should be plotted.
    filename : string, optional
        If filename is set, the strain signal and plot will be saved.
        ``plot`` must be set to True.
    return_x_y_z : bool, default False
        If this is set to True, the ``strain_signal`` (map), as well as
        separate lists of the x, y, and strain values.
    **kwargs : Matplotlib keyword arguments passed to ``imshow()``.

    Returns
    -------
    Strain map as a Hyperspy Signal2D

    Examples
    --------
    >>> import atomap.api as am
    >>> from temul.topotem.polarisation import get_strain_map
    >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeA.find_nearest_neighbors()
    >>> _ = sublatticeA.refine_atom_positions_using_center_of_mass()
    >>> sublatticeA.construct_zone_axes()
    >>> strain_map = get_strain_map(sublatticeA, zone_axis_index=0, units='nm',
    ...                             theoretical_value=1.9, sampling=0.1)
    """

    zone_vector_index_list = sublattice._get_zone_vector_index_list(
        zone_vector_list=None)
    _, zone_vector = zone_vector_index_list[zone_axis_index]
    zone_data = sublattice.get_monolayer_distance_list_from_zone_vector(
        zone_vector)
    x_position, y_position, xy_distance = zone_data

    xy_distance = [i * sampling for i in xy_distance]
    xy_distance = [(i - theoretical_value) / theoretical_value for
                   i in xy_distance]
    xy_distance = [i * 100 for i in xy_distance]

    strain_signal = sublattice.get_property_map(
        x_position, y_position, xy_distance, upscale_map=1)
    if sampling is not None:
        strain_signal.axes_manager[0].scale = sampling
        strain_signal.axes_manager[1].scale = sampling
    strain_signal.axes_manager[0].units = units
    strain_signal.axes_manager[1].units = units

    if plot:
        if vmax == 'max':
            vmax = np.max(xy_distance)
        if vmin == 'min':
            vmin = np.min(xy_distance)

        strain_signal.plot(vmin=vmin, vmax=vmax, cmap=cmap,
                           colorbar=False, **kwargs)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.title("{}_{}".format(title, zone_axis_index))
        cbar = ScalarMappable(cmap=cmap)
        cbar.set_array(xy_distance)
        cbar.set_clim(vmin, vmax)
        plt.colorbar(cbar, fraction=0.046, pad=0.04,
                     label="Strain (% above {} {})".format(
                         theoretical_value, units))
        plt.tight_layout()

        if filename is not None:
            plt.savefig(
                fname="{}_{}_{}.png".format(filename, title, zone_axis_index),
                transparent=True, frameon=False, bbox_inches='tight',
                pad_inches=None, dpi=300, labels=False)
            strain_signal.save("{}_{}_{}.hspy".format(
                filename, title, zone_axis_index))

    if return_x_y_z:
        return (strain_signal, x_position, y_position, xy_distance)
    else:
        return (strain_signal)


# Improvement would be to distinguish between horizontal angle e.g., 5 and 175
# degrees. Also 'deg' and 'rad' should be degrees=True/False
def rotation_of_atom_planes(sublattice, zone_axis_index, angle_offset=None,
                            degrees=False, sampling=None, units='pix',
                            vmin=None, vmax=None, cmap='inferno',
                            title='Rotation Map', plot=False, filename=None,
                            return_x_y_z=False, **kwargs):
    """
    Calculate a map of the angles between each atom along the atom planes of a
    zone axis.

    Parameters
    ----------
    sublattice : Atomap Sublattice object
    zone_axis_index : int
        The zone axis you wish to specify. You are indexing
        ``sublattice.zones_axis_average_distances[zone_axis_index]``.
    angle_offset : float, default None
        The angle which can be considered zero degrees. Useful when the atomic
        planes are at an angle.
    degrees : bool, default False
        Setting to False will return angle values in radian. Setting to True
        will return angle values in degrees.
    sampling : float, default None
        Pixel sampling of the image for calibration.
    units : string, default "pix"
        Units of the sampling.
    vmin, vmax, cmap : see Matplotlib for details
    title : string, default "Rotation Map"
    plot : bool
        Set to true if the figure should be plotted.
    filename : string, optional
        If filename is set, the signal and plot will be saved.
        ``plot`` must be set to True.
    return_x_y_z : bool, default False
        If this is set to True, the rotation_signal (map), as well as separate
        lists of the x, y, and angle values.
    **kwargs : Matplotlib keyword arguments passed to ``imshow()``.

    Returns
    -------
    Rotation map as a Hyperspy Signal2D

    Examples
    --------
    >>> import atomap.api as am
    >>> from temul.topotem.polarisation import rotation_of_atom_planes
    >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[1]
    >>> sublatticeA.find_nearest_neighbors()
    >>> _ = sublatticeA.refine_atom_positions_using_center_of_mass()
    >>> sublatticeA.construct_zone_axes()
    >>> rotation_map = rotation_of_atom_planes(sublatticeA, 3, degrees=True)

    Use ``angle_offset`` to effectively change the angle of the horizontal axis
    when calculating angles:

    >>> rotation_map = rotation_of_atom_planes(sublatticeA, 3, angle_offset=45,
    ...                                        degrees=True)

    Use the return_x_y_z parameter when you want to either plot with a
    different style (e.g., contour map), or you want the angle information:

    >>> rotation_map, x, y, angles = rotation_of_atom_planes(
    ...     sublatticeA, 3, degrees=True, return_x_y_z=True)
    >>> mean_angle = np.mean(angles)  # useful for offsetting polar. plots

    """

    zone_vector_index_list = sublattice._get_zone_vector_index_list(
        zone_vector_list=None)
    _, zone_vector = zone_vector_index_list[zone_axis_index]

    angles_list_rad = []
    x_list, y_list = [], []
    for atom_plane in sublattice.atom_plane_list:
        if atom_plane.zone_vector == zone_vector:
            pos_distance = atom_plane.position_distance_to_neighbor()
            x_half_pos = pos_distance[:, 0]
            y_half_pos = pos_distance[:, 1]
            angle = atom_plane.get_angle_to_horizontal_axis()

            x_list.append(x_half_pos)
            y_list.append(y_half_pos)
            angles_list_rad.append(angle)

    # flatten the lists
    x_list = [i for sublist in x_list for i in sublist]
    y_list = [i for sublist in y_list for i in sublist]
    angles_list_rad = [i for sublist in angles_list_rad for i in sublist]

    if degrees:
        angles_list_deg = [np.degrees(i) for i in angles_list_rad]
        angles_list = angles_list_deg
    elif not degrees:
        angles_list = angles_list_rad

    bar_label = angle_label("angle", degrees=degrees)

    if angle_offset is not None:
        angles_list = [i + angle_offset for i in angles_list]

    rotation_signal = sublattice.get_property_map(
        x_list, y_list, angles_list, upscale_map=1)
    if sampling is not None:
        rotation_signal.axes_manager[0].scale = sampling
        rotation_signal.axes_manager[1].scale = sampling
    rotation_signal.axes_manager[0].units = units
    rotation_signal.axes_manager[1].units = units

    if plot:
        if vmax == 'max':
            vmax = np.max(angles_list)
        if vmin == 'min':
            vmin = np.min(angles_list)

        rotation_signal.plot(vmin=vmin, vmax=vmax, cmap=cmap,
                             colorbar=False, **kwargs)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.title("{}_{}".format(title, zone_axis_index))
        cbar = ScalarMappable(cmap=cmap)
        cbar.set_array(angles_list)
        cbar.set_clim(vmin, vmax)
        plt.colorbar(cbar, fraction=0.046, pad=0.04,
                     label=bar_label)
        plt.tight_layout()

        if filename is not None:
            plt.savefig(
                fname="{}_{}_{}.png".format(filename, title, zone_axis_index),
                transparent=True, frameon=False, bbox_inches='tight',
                pad_inches=None, dpi=300, labels=False)
            rotation_signal.save("{}_{}_{}.hspy".format(
                filename, title, zone_axis_index))

    if return_x_y_z:
        return (rotation_signal, x_list, y_list, angles_list)
    else:
        return (rotation_signal)


def ratio_of_lattice_spacings(sublattice, zone_axis_index_A, zone_axis_index_B,
                              ideal_ratio_one=True, sampling=1, units='pix',
                              vmin=None, vmax=None, cmap='inferno',
                              title='Spacings Map', filename=None, **kwargs):
    """
    Create a ratio map between two zone axes. Useful to see the tetragonality
    or shearing of unit cells.

    Parameters
    ----------
    sublattice : Atomap Sublattice object
    zone_axis_index_A, zone_axis_index_B : int
        The zone axes you wish to specify. You are indexing
        ``sublattice.zones_axis_average_distances[zone_axis_index]``.
        The signal created from zone_axis_index_A will be divided by the signal
        created from zone_axis_index_B.
    ideal_ratio_one : bool, default True
        If set to true this will force negative ratio values to be positive.
        Useful for seeing the overall tetragonality of a lattice.
    sampling : float, default None
        Pixel sampling of the image for calibration.
    units : string, default "pix"
        Units of the sampling.
    vmin, vmax, cmap : see Matplotlib for details
    title : string, default "Spacings Map"
    filename : string, optional
        If filename is set, the strain signal and plot will be saved.
    **kwargs : Matplotlib keyword arguments passed to ``imshow()``.

    Returns
    -------
    Ratio of lattice spacings map as a Hyperspy Signal2D. It will also plot the
    two lattice spacing maps.

    Examples
    --------
    >>> import atomap.api as am
    >>> from temul.topotem.polarisation import ratio_of_lattice_spacings
    >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeA.find_nearest_neighbors()
    >>> _ = sublatticeA.refine_atom_positions_using_center_of_mass()
    >>> sublatticeA.construct_zone_axes()
    >>> ratio_map = ratio_of_lattice_spacings(sublatticeA, 0, 1)

    Use ``ideal_ratio_one=False`` to view the direction of tetragonality

    >>> ratio_map = ratio_of_lattice_spacings(sublatticeA, 0, 1,
    ...                                       ideal_ratio_one=False)

    """

    zone_vector_index_list = sublattice._get_zone_vector_index_list(
        zone_vector_list=None)

    # spacing A
    zone_index_A, zone_vector_A = zone_vector_index_list[zone_axis_index_A]
    x_list_A, y_list_A, xy_dist_A = \
        sublattice.get_atom_distance_list_from_zone_vector(zone_vector_A)

    signal_spacing_A = sublattice.get_property_map(
        x_list_A, y_list_A, xy_dist_A, upscale_map=1)
    if sampling is not None:
        signal_spacing_A.axes_manager[0].scale = sampling
        signal_spacing_A.axes_manager[1].scale = sampling
    signal_spacing_A.axes_manager[0].units = units
    signal_spacing_A.axes_manager[1].units = units

    signal_spacing_A.plot(vmin=vmin, vmax=vmax, cmap=cmap,
                          colorbar=False, **kwargs)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.title("{}_{}".format(title, zone_index_A))
    cbar = ScalarMappable(cmap=cmap)
    cbar.set_array(xy_dist_A)
    cbar.set_clim(vmin, vmax)
    plt.colorbar(cbar, fraction=0.046, pad=0.04,
                 label="Spacing of Atoms (pix)")
    plt.tight_layout()

    # spacing B
    zone_index_B, zone_vector_B = zone_vector_index_list[zone_axis_index_B]
    x_list_B, y_list_B, xy_dist_B = \
        sublattice.get_atom_distance_list_from_zone_vector(zone_vector_B)

    signal_spacing_B = sublattice.get_property_map(
        x_list_B, y_list_B, xy_dist_B, upscale_map=1)
    if sampling is not None:
        signal_spacing_B.axes_manager[0].scale = sampling
        signal_spacing_B.axes_manager[1].scale = sampling
    signal_spacing_B.axes_manager[0].units = units
    signal_spacing_B.axes_manager[1].units = units

    signal_spacing_B.plot(vmin=vmin, vmax=vmax, cmap=cmap,
                          colorbar=False, **kwargs)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.title("{}_{}".format(title, zone_index_B))
    cbar = ScalarMappable(cmap=cmap)
    cbar.set_array(xy_dist_B)
    cbar.set_clim(vmin, vmax)
    plt.colorbar(cbar, fraction=0.046, pad=0.04,
                 label="Spacing of Atoms (pix)")
    plt.tight_layout()

    # Get the A/B ratio
    ratio_signal = signal_spacing_A / signal_spacing_B
    ratio_signal_data = ratio_signal.data

    if ideal_ratio_one:
        ratio_signal.data = np.where(
            ratio_signal_data < 1, 1 / ratio_signal_data, ratio_signal_data)

    ratio_signal.plot(vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.title("Ratio of {}_{}/{}".format(title, zone_index_A, zone_index_B))
    plt.tight_layout()

    if filename is not None:
        plt.savefig(fname="{}_{}_{}{}.png".format(
            filename, title, zone_axis_index_A, zone_axis_index_B),
            transparent=True, frameon=False, bbox_inches='tight',
            pad_inches=None, dpi=300, labels=False)
        ratio_signal.save("{}_{}_{}{}.hspy".format(
            filename, title, zone_axis_index_A, zone_axis_index_B))

    return (ratio_signal)


def angle_label(vector_rep="magnitude", units='pix', degrees=False):
    if vector_rep == "magnitude":
        vector_label = "Magnitude ({})".format(units)
    elif vector_rep == "angle":
        if degrees:
            vector_label = "Angle (deg)"
        else:
            vector_label = "Angle (rad)"
    else:
        raise ValueError(
            "`vector_rep`` must be either 'magnitude' or 'angle'.")
    return (vector_label)


def atom_to_atom_distance_grouped_mean(sublattice, zone_axis_index,
                                       aggregation_axis="y",
                                       slice_thickness=10,
                                       sampling=None, units="pix"):
    """
    Average the atom to atom distances along the chosen zone_axis_index
    parallel to the chosen axis ('x' or 'y').

    Parameters
    ----------
    sublattice : Atomap Sublattice object
    zone_axis_index : int
        The zone axes you wish to average along.
    aggregation_axis : string, default "y"
        Axis parallel to which the atom to atom distances will be averaged.
    slice_thickness : float, default 10
        thickness of the slices used for aggregation.
    sampling : float, default None
        Pixel sampling of the image for calibration.
    units : string, default "pix"
        Units of the sampling.

    Returns
    -------
    Slice thickness groupings and means of each grouping. Groupings can be
    thought of as bins in a histogram of grouped means.

    Example
    -------

    >>> import numpy as np
    >>> from atomap.dummy_data import get_distorted_cubic_sublattice
    >>> import matplotlib.pyplot as plt
    >>> from temul.topotem.polarisation import (
    ...     atom_to_atom_distance_grouped_mean)
    >>> sublatticeA = get_distorted_cubic_sublattice()
    >>> sublatticeA.construct_zone_axes(atom_plane_tolerance=1)
    >>> sublatticeA.plot()
    >>> groupings, grouped_means = atom_to_atom_distance_grouped_mean(
    ...     sublatticeA, 0, 'y', 40)

    You can then plot these as below:
    plt.figure()
    plt.plot(groupings, grouped_means, 'k.')
    plt.show()

    Average parallel to the x axis instead:

    >>> groupings, grouped_means = atom_to_atom_distance_grouped_mean(
    ...     sublatticeA, 0, 'x', 40)

    You can then plot these as below:
    plt.figure()
    plt.plot(groupings, grouped_means, 'k.')
    plt.show()

    """
    zone_vector_index_list = sublattice._get_zone_vector_index_list(
        zone_vector_list=None)
    _, zone_vector = zone_vector_index_list[zone_axis_index]

    x, y, dist = sublattice.get_atom_distance_list_from_zone_vector(
        zone_vector)

    if sampling is not None:
        x, y, dist = x * sampling, y * sampling, dist * sampling

    image_size_yx = sublattice.image.shape

    if aggregation_axis == 'x':
        image_size = image_size_yx[1]
        travel_axis = y
    elif aggregation_axis == 'y':
        image_size = image_size_yx[0]
        travel_axis = x

    groupings = np.arange(0, image_size * 1.25, slice_thickness)

    grouped_means = []
    for group in groupings:
        grouped_region = []
        for travel_axis_i, dist_i in zip(travel_axis, dist):
            if group < travel_axis_i < group + slice_thickness:
                grouped_region.append(dist_i)
        grouped_region = np.asarray(grouped_region)
        grouped_means.append(np.mean(grouped_region))

    return (groupings, grouped_means)
