
import hyperspy
import numpy as np
import scipy
from scipy.optimize import curve_fit
from scipy.misc import derivative
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
from decimal import Decimal
import colorcet as cc
from matplotlib_scalebar.scalebar import ScaleBar
from temul.signal_processing import sine_wave_function_strain_gradient


# good to have an example of getting atom_positions_A and B from sublattice
def find_polarisation_vectors(atom_positions_A, atom_positions_B,
                              save=None):
    '''
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
    >>> from temul.polarisation import find_polarisation_vectors
    >>> pos_A = [[1,2], [3,4], [5,8], [5,2]]
    >>> pos_B = [[1,1], [5,2], [3,1], [6,2]]
    >>> u, v = find_polarisation_vectors(pos_A, pos_B, save=None)

    convert to the [[u1,v1], [u2,v2], [u3,v3]...] format

    >>> import numpy as np
    >>> vectors = np.asarray([u, v]).T

    '''
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

    return(u, v)


def plot_polarisation_vectors(
        x, y, u, v, image, sampling=None, units='pix',
        plot_style='vector', vector_rep='magnitude',
        overlay=True, unit_vector=False, degrees=False, angle_offset=None,
        save='polarisation_image', title="",
        color='yellow', cmap=None, alpha=1.0,
        monitor_dpi=96, pivot='middle', angles='xy',
        scale_units='xy', scale=None, headwidth=3.0, headlength=5.0,
        headaxislength=4.5, no_axis_info=True, ticks=None, scalebar=False,
        antialiased=False, levels=20, remove_vectors=False):
    '''
    Plot the polarisation vectors. These can be found with
    `find_polarisation_vectors()` or Atomap's
    `get_polarization_from_second_sublattice()` function.

    Parameters
    ----------
    See matplotlib's quiver function for more details.

    x, y : list or 1D NumPy array
        xy coordinates on the image
    u, v : list or 1D NumPy array
        uv vector components
    image : 2D NumPy array
    sampling : float, default None
        Pixel sampling of the image for calibration.
    units : string, default "pix"
        Units used to display the magnitude of the vectors.
    plot_style : string, default "vector"
        Options are "vector", "colormap", "contour", "colorwheel". Note that
        "colorwheel" will automatically plot the colorbar as an angle.
    vector_rep : str, default "magnitude"
        How the vectors are represented. This can be either their `magnitude`
        or `angle`. One may want to use `angle` when plotting a contour map,
        i.e., view the contours in terms of angles which can be useful for
        visualising regions of different polarisation.
    overlay : Bool, default True
        If set to True, the `image` will be plotting behind the arrows
    unit_vector : Bool, default False
        Change the vectors magnitude to unit vectors for plotting purposes.
        Magnitude will still be displayed correctly for colormaps etc.
    degrees : Bool, default False
        Change between degrees and radian. Default is radian.
        If `plot_style="colorwheel"`, then setting `degrees=True` will convert
        the angle unit to degree from the default radians.
    angle_offset : float, default None
        If using `vector_rep="angle"` or `plot_style="contour"`, this angle
        will rotate the vector angles displayed by the given amount. Useful
        when you want to offset the angle of the atom planes relative to the
        polarisation.
    save : string, default "polarisation_image"
        If set to `save=None`, the array will not be saved.
    title : string, default ""
        Title of the plot.
    color : string, default "r"
        Color of the arrows when `plot_style="vector" or "contour".
    cmap : matplotlib colormap, default "viridis"
    alpha : float, default 1.0
        Transparency of the matplotlib `cmap`. For `plot_style="colormap"` and
        `plot_style="colorwheel"`, this alpha applies to the vector arrows.
        For `plot_style="contour"` this alpha applies to the tricontourf map.
    monitor_dpi : int, default 96
        The DPI of the monitor, generally 96 pixels. Used to scale the image
        so that large images render correctly. Use a smaller value or
        `monitor_dpi=None` to enlarge too-small images.
    no_axis_info :  Bool, default True
        This will remove the x and y axis labels and ticks from the plot if set
        to True.
    ticks : colorbar ticks, default None
        None or list of ticks or Locator If None, ticks are determined
        automatically from the input.
    scalebar : Bool or dict, default False
        Add a matplotlib-scalebar to the plot. If set to True the scalebar will
        appear similar to that given by Hyperspy's `plot()` function. A custom
        scalebar can be included as a dictionary and more custom options can be
        found in the matplotlib-scalebar package. See below for an example.
    antialiased : Bool, default False
        Applies only to `plot_style="contour"`. Essentially removes the
        border between regions in the tricontourf map.
    levels : int, default 20
        Number of Matplotlib tricontourf levels to be used.
    remove_vectors : Bool, default False
        Applies only to `plot_style="contour"`. If set to True, do not plot
        the vector arrows.
    See matplotlib's quiver function for the remaining parameters.

    Examples
    --------
    >>> from temul.polarisation import plot_polarisation_vectors
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

    >>> plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           unit_vector=False, save=None,
    ...                           plot_style='vector', color='r',
    ...                           overlay=False, title='Vector Arrows',
    ...                           monitor_dpi=50)

    vector plot with red arrows overlaid on the image:

    >>> plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           unit_vector=False, save=None,
    ...                           plot_style='vector', color='r',
    ...                           overlay=True, monitor_dpi=50)

    vector plot with colormap viridis:

    >>> plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           unit_vector=False, save=None,
    ...                           plot_style='colormap', monitor_dpi=50,
    ...                           overlay=False, cmap='viridis')

    vector plot with colormap viridis, with `vector_rep="angle"`:

    >>> plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           unit_vector=False, save=None,
    ...                           plot_style='colormap', monitor_dpi=50,
    ...                           overlay=False, cmap='cet_colorwheel',
    ...                           vector_rep="angle", degrees=True)

    colormap arrows with sampling applied and with scalebar:

    >>> plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           sampling=3.0321, units='pm', monitor_dpi=50,
    ...                           unit_vector=False, plot_style='colormap',
    ...                           overlay=True, save=None, cmap='viridis',
    ...                           scalebar=True)

    vector plot with colormap viridis and unit vectors:

    >>> plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           unit_vector=True, save=None, monitor_dpi=50,
    ...                           plot_style='colormap', color='r',
    ...                           overlay=False, cmap='viridis')

    Change the vectors to unit vectors on a tricontourf map:

    >>> plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           unit_vector=True, plot_style='contour',
    ...                           overlay=False, pivot='middle', save=None,
    ...                           color='darkgray', cmap='viridis',
    ...                           monitor_dpi=50)

    Plot a partly transparent angle tricontourf map with vector arrows:

    >>> plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           unit_vector=False, plot_style='contour',
    ...                           overlay=True, pivot='middle', save=None,
    ...                           color='red', cmap='cet_colorwheel',
    ...                           monitor_dpi=50, remove_vectors=False,
    ...                           vector_rep="angle", alpha=0.5, levels=9,
    ...                           antialiased=True, degrees=True,
    ...                           ticks=[180, 90, 0, -90, -180])

    Plot a partly transparent angle tricontourf map with no vector arrows:

    >>> plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           unit_vector=True, plot_style='contour',
    ...                           overlay=True, pivot='middle', save=None,
    ...                           cmap='cet_colorwheel',
    ...                           monitor_dpi=50, remove_vectors=True,
    ...                           vector_rep="angle", alpha=0.5,
    ...                           antialiased=True, degrees=True)

    "colorwheel" plot of the vectors, useful for vortexes:

    >>> import colorcet as cc
    >>> plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           unit_vector=True, plot_style="colorwheel",
    ...                           vector_rep="angle",
    ...                           overlay=False, cmap=cc.cm.colorwheel,
    ...                           degrees=True, save=None, monitor_dpi=50)

    Plot with a custom scalebar, for example here we need it to be dark, see
    matplotlib-scalebar for more custom features.

    >>> scbar_dict = {"dx": 3.0321, "units": "pm", "location": "lower left",
    ...               "box_alpha":0.0, "color": "black", "scale_loc": "top"}
    >>> plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           sampling=3.0321, units='pm', monitor_dpi=50,
    ...                           unit_vector=False, plot_style='colormap',
    ...                           overlay=False, save=None, cmap='viridis',
    ...                           scalebar=scbar_dict)

    Plot a contourf for quadrant visualisation using a custom matplotlib cmap:

    >>> import temul.signal_plotting as tmlplot
    >>> from matplotlib.colors import from_levels_and_colors
    >>> zest = tmlplot.hex_to_rgb(tmlplot.color_palettes('zesty'))
    >>> zest.append(zest[0])  # make the -180 and 180 degree colour the same
    >>> expanded_zest = tmlplot.expand_palette(zest, [1,2,2,2,1])
    >>> custom_cmap, _ = from_levels_and_colors(
    ...     levels=range(9), colors=tmlplot.rgb_to_dec(expanded_zest))
    >>> plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           unit_vector=False, plot_style='contour',
    ...                           overlay=False, pivot='middle', save=None,
    ...                           cmap=custom_cmap, levels=9, monitor_dpi=50,
    ...                           vector_rep="angle", alpha=0.5, color='r',
    ...                           antialiased=True, degrees=True,
    ...                           ticks=[180, 90, 0, -90, -180])

    '''

    if isinstance(image, np.ndarray):
        pass
    elif isinstance(image, hyperspy._signals.signal2d.Signal2D):
        sampling = image.axes_manager[-1].scale
        units = image.axes_manager[-1].units
        image = image.data
    else:
        raise ValueError("`image` must be a 2D numpy array or 2D Hyperspy "
                         "Signal")

    u, v = np.array(u), np.array(v)

    if sampling is not None:
        u, v = u * sampling, v * sampling

    if vector_rep == "magnitude":
        vector_rep_val = get_vector_magnitudes(u, v)
    elif vector_rep == "angle":
        # -v because in STEM the origin is top left
        vector_rep_val = get_angles_from_uv(u, -v, degrees=degrees,
                                            angle_offset=angle_offset)

    vector_label = angle_label(
            vector_rep=vector_rep, units=units, degrees=degrees)

    if unit_vector:
        # Normalise the data for uniform arrow size
        u_norm = u / np.sqrt(u ** 2.0 + v ** 2.0)
        v_norm = v / np.sqrt(u ** 2.0 + v ** 2.0)
        u = u_norm
        v = v_norm

    if monitor_dpi is not None:
        _, ax = plt.subplots(figsize=[image.shape[1] / monitor_dpi,
                                      image.shape[0] / monitor_dpi])
    else:
        _, ax = plt.subplots()

    ax.set_title(title, loc='left', fontsize=20)
    # plot_style options
    if plot_style == "vector":
        Q = ax.quiver(
            x, y, u, v, color=color, pivot=pivot, angles=angles,
            scale_units=scale_units, scale=scale, headwidth=headwidth,
            headlength=headlength, headaxislength=headaxislength)
        length = np.max(np.hypot(u, v)) / 2
        ax.quiverkey(Q, 0.9, 1.025, length,
                     label='{:.0E} {}'.format(Decimal(length), units),
                     labelpos='E', coordinates='axes')

    elif plot_style == "colormap":

        if cmap is None:
            cmap = 'viridis'
        ax.quiver(
            x, y, u, v, vector_rep_val, color=color, cmap=cmap,
            pivot=pivot, angles=angles, scale_units=scale_units,
            scale=scale, headwidth=headwidth, alpha=alpha,
            headlength=headlength, headaxislength=headaxislength)

        norm = colors.Normalize(vmin=np.min(vector_rep_val),
                                vmax=np.max(vector_rep_val))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(mappable=sm)
        cbar.ax.set_ylabel(vector_label)

    elif plot_style == "colorwheel":

        if vector_rep != "angle":
            raise ValueError("`vector_rep` must be set to 'angle' when "
                             "`plot_style` is set to 'colorwheel'.")
        if cmap is None:
            cmap = cc.cm.colorwheel

        Q = ax.quiver(
            x, y, u, v, vector_rep_val, cmap=cmap, alpha=alpha,
            pivot=pivot, angles=angles, scale_units=scale_units,
            scale=scale, headwidth=headwidth,
            headlength=headlength, headaxislength=headaxislength)
        plt.colorbar(Q, label=vector_label)

    elif plot_style == "contour":

        if cmap is None:
            cmap = 'viridis'

        if vector_rep == "angle":
            if degrees:
                min_angle, max_angle = -180, 180 + 0.0001  # fixes display issues
            elif not degrees:
                min_angle, max_angle = -np.pi, np.pi

        if isinstance(levels, list):
            levels_list = levels
        elif isinstance(levels, int):
            if vector_rep == "angle":
                levels_list = np.linspace(min_angle, max_angle, levels)
            elif vector_rep == "magnitude":
                levels_list = np.linspace(np.min(vector_rep_val),
                                          np.max(vector_rep_val)+0.00001,
                                          levels)

        contour_map = plt.tricontourf(x, y, vector_rep_val, cmap=cmap,
                                      alpha=alpha, antialiased=antialiased,
                                      levels=levels_list)

        if not remove_vectors:
            ax.quiver(
                x, y, u, v, color=color, pivot=pivot,
                angles=angles, scale_units=scale_units,
                scale=scale, headwidth=headwidth,
                headlength=headlength, headaxislength=headaxislength)

    ax.set(aspect='equal')
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)

    if plot_style == 'contour':
        cbar = plt.colorbar(mappable=contour_map, fraction=0.046, pad=0.04,
                            drawedges=False)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_ticks(ticks)
        cbar.ax.set_ylabel(vector_label, fontsize=14)

    if overlay:
        plt.imshow(image)

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

    # plt.tight_layout()
    if save is not None:
        plt.savefig(fname=save + '_' + plot_style + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)


def get_angles_from_uv(u, v, degrees=False, angle_offset=None):
    '''
    Calculate the angle of a vector given the uv components.

    Parameters
    ----------
    u,v  : list or 1D NumPy array
    degrees : Bool, default False
        Change between degrees and radian. Default is radian.
    angle_offset : float, default None
        Rotate the angles by the given amount. The function assumes that if you
        set `degrees=False` then the provided `angle_offset` is in radians, and
        if you set `degrees=True` then the provided `angle_offset` is in
        degrees.

    Returns
    -------
    1D NumPy array
    '''

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

    return(vector_angles)


def get_vector_magnitudes(u, v, sampling=None):
    '''
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
    >>> from temul.polarisation import get_vector_magnitudes
    >>> import numpy as np
    >>> u, v = [4,3,2,5,6], [8,5,2,1,1] # list input
    >>> vector_mags = get_vector_magnitudes(u,v)
    >>> u, v = np.array(u), np.array(v) # numpy input also works
    >>> vector_mags = get_vector_magnitudes(u,v)
    >>> sampling = 0.0321
    >>> vector_mags = get_vector_magnitudes(u,v, sampling=sampling)

    '''

    # uv_vector_comp_list = [list(uv) for uv in uv_vector_comp]
    # u = [row[0] for row in uv_vector_comp_list]
    # v = [row[1] for row in uv_vector_comp_list]

    u_comp = np.array(u)
    v_comp = np.array(v).T

    vector_mags = (u_comp ** 2 + v_comp ** 2) ** 0.5

    if sampling is not None:
        vector_mags = vector_mags * sampling

    return(vector_mags)


def delete_atom_planes_from_sublattice(sublattice,
                                       zone_axis_index=0,
                                       atom_plane_tolerance=0.5,
                                       divisible_by=3,
                                       offset_from_zero=0,
                                       opposite=False):
    '''
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
    opposite : Bool, default False
        If this is set to True, the atom_plane specified by divisible_by
        will be kept and all others deleted.

    Examples
    --------
    >>> from temul.polarisation import delete_atom_planes_from_sublattice
    >>> import atomap.dummy_data as dd
    >>> atom_lattice = dd.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeA.construct_zone_axes()
    >>> zone_vec_list = sublatticeA.zones_axis_average_distances[0:2]
    >>> sublatticeA.get_all_atom_planes_by_zone_vector(zone_vec_list).plot()
    >>> delete_atom_planes_from_sublattice(
    ...         sublatticeA, zone_axis_index=0,
    ...         divisible_by=3, offset_from_zero=1)
    >>> sublatticeA.get_all_atom_planes_by_zone_vector(zone_vec_list).plot()

    '''
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
    atom_plane_index_delete = [index for index in atom_plane_index_delete
                               if index < len(
                                   sublattice.atom_planes_by_zone_vector[
                                       zone_vec_needed])]

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


# atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
# sublatticeA = atom_lattice.sublattice_list[0]
# delete_atom_planes_from_sublattice(sublattice=sublatticeA,
#                                    zone_axis_index=0,
#                                    divisible_by=3,
#                                    offset_from_zero=0,
#                                    opposite=True)
# sublatticeA.plot_planes()


def _truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def atom_deviation_from_straight_line_fit(sublattice,
                                          axis_number: int = 0,
                                          save: str = ''):
    '''
    Fit the atoms in an atom plane to a straight line and find the deviation
    of each atom position from that straight line fit.
    To plot all zones see `plot_atom_deviation_from_all_zone_axes()`.

    Parameters
    ----------
    sublattice : Atomap Sublattice object
    axis_number : int, default 0
        The index of the zone axis (translation symmetry) found by the Atomap
        function `construct_zone_axes()`.
    save : string, default ''
        If set to `save=None`, the array will not be saved.

    Returns
    -------
    Four lists: x, y, u, and v where x,y are the original atom position
    coordinates (simply sublattice.x_position, sublattice.y_position) and
    u,v are the polarisation vector components pointing to the new coordinate.
    These can be input to `plot_polarisation_vectors()` for visualisation.

    Examples
    --------
    >>> import atomap.api as am
    >>> from temul.polarisation import (atom_deviation_from_straight_line_fit,
    ...                                 plot_polarisation_vectors)
    >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeA.find_nearest_neighbors()
    >>> sublatticeA.refine_atom_positions_using_center_of_mass()
    >>> sublatticeA.construct_zone_axes()
    >>> x,y,u,v = atom_deviation_from_straight_line_fit(sublatticeA, save=None)
    >>> plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           unit_vector=False, save=None, monitor_dpi=50)

    '''

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

    return(x, y, u, v)


# need to add the truncated colormap version: divergent plot.
def plot_atom_deviation_from_all_zone_axes(
        sublattice, image=None, sampling=None, units='pix',
        plot_style='vector', overlay=True, unit_vector=False, degrees=False,
        save='atom_deviation', title="", color='yellow', cmap=None,
        pivot='middle', angles='xy', scale_units='xy', scale=None,
        headwidth=3.0, headlength=5.0, headaxislength=4.5, monitor_dpi=96,
        no_axis_info=True, scalebar=False):
    '''
    Plot the atom deviation from a straight line fit for all zone axes
    constructed by an Atomap sublattice object.

    Parameters
    ----------
    sublattice : Atomap Sublattice object

    For all other parameters see `plot_polarisation_vectors()`.

    Examples
    --------
    >>> import atomap.api as am
    >>> from temul.polarisation import plot_atom_deviation_from_all_zone_axes
    >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeA.find_nearest_neighbors()
    >>> sublatticeA.refine_atom_positions_using_center_of_mass()
    >>> sublatticeA.construct_zone_axes()
    >>> plot_atom_deviation_from_all_zone_axes(sublatticeA, save=None)

    '''

    if image is None:
        image = sublattice.image

    if sublattice.zones_axis_average_distances is None:
        raise Exception(
            "zones_axis_average_distances is empty. "
            "Has sublattice.construct_zone_axes() been run?")
    else:
        pass

    for axis_number in range(len(sublattice.zones_axis_average_distances)):

        x, y, u, v = atom_deviation_from_straight_line_fit(
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
    '''
    Combine the atom deviations of each atom for all zone axes.
    Good for plotting vortexes and seeing the total deviation from a
    perfect structure.

    Parameters
    ----------
    sublattice : Atomap Sublattice object

    For the remaining parameters see `plot_polarisation_vectors()`.

    Returns
    -------
    Four lists: x, y, u, and v where x,y are the original atom position
    coordinates (simply sublattice.x_position, sublattice.y_position) and
    u,v are the polarisation vector components pointing to the new coordinate.
    These can be input to `plot_polarisation_vectors()` for visualisation.

    Examples
    --------

    >>> import atomap.api as am
    >>> from temul.polarisation import (plot_polarisation_vectors,
    ...     combine_atom_deviations_from_zone_axes)
    >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeA.find_nearest_neighbors()
    >>> sublatticeA.refine_atom_positions_using_center_of_mass()
    >>> sublatticeA.construct_zone_axes()
    >>> x,y,u,v = combine_atom_deviations_from_zone_axes(
    ...     sublatticeA, save=None)
    >>> plot_polarisation_vectors(x, y, u, v, save=None,
    ...     image=sublatticeA.image)

    You can also choose the axes:

    >>> x,y,u,v = combine_atom_deviations_from_zone_axes(
    ...     sublatticeA, axes=[0,1], save=None)
    >>> plot_polarisation_vectors(x, y, u, v, save=None,
    ...     image=sublatticeA.image)

    '''

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

        x, y, u, v = atom_deviation_from_straight_line_fit(
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
              "axes given, removing atoms: {}".format(
                  atoms_not_found))
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

    return(x, y, u, v)


def get_divide_into(sublattice, averaging_by, sampling,
                    zone_axis_index_A, zone_axis_index_B):
    '''
    Calculate the `divide_into` required to get an averaging of `averaging_by`.
    `divide_into` can then be used in
    temul.polarisation.get_average_polarisation_in_regions.
    Also finds unit cell size and the number of unit cells in the (square)
    image along the x axis.

    Parameters
    ----------
    sublattice : Atomap Sublattice
    averaging_by : int or float
        How many unit cells should be averaged. If `averaging_by=2`, 2x2 unit
        cells will be averaged when passing `divide_into` to
        temul.polarisation.get_average_polarisation_in_regions.
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

    >>> from temul.polarisation import get_divide_into
    >>> from atomap.dummy_data import get_simple_cubic_sublattice
    >>> sublattice = get_simple_cubic_sublattice()
    >>> sublattice.construct_zone_axes()
    >>> cell_info = get_divide_into(sublattice, averaging_by=2, sampling=0.1,
    ...                 zone_axis_index_A=0, zone_axis_index_B=1)
    >>> divide_into = cell_info[0]
    >>> unit_cell_size = cell_info[1]
    >>> num_unit_cells = cell_info[2]
    >>> sublattice.plot()  # You can count the unit cells to check

    '''

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

    unit_cell_size = (p_A + p_B)/2
    image_size_x = sublattice.signal.axes_manager[0].size * sampling

    num_unit_cells = image_size_x / unit_cell_size

    divide_into = num_unit_cells / averaging_by

    return(divide_into, unit_cell_size, num_unit_cells)


def get_average_polarisation_in_regions(x, y, u, v, image, divide_into=8):
    '''
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
    Four lists: x_new, y_new, u_new, v_new.
    x_new and y_new are the central coordinates of the divided regions.
    u_new and v_new are the averaged polarisation vectors.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import atomap.api as am
    >>> from temul.polarisation import (
    ...    combine_atom_deviations_from_zone_axes,
    ...    plot_polarisation_vectors, get_average_polarisation_in_regions)
    >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeA.find_nearest_neighbors()
    >>> sublatticeA.refine_atom_positions_using_center_of_mass()
    >>> sublatticeA.construct_zone_axes()

    Get and plot the original polarisation vectors:

    >>> x, y, u, v = combine_atom_deviations_from_zone_axes(sublatticeA,
    ...     save=None)
    >>> plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                   unit_vector=False, save=None,
    ...                   plot_style='vector', color='r',
    ...                   overlay=False, title='Actual Vector Arrows',
    ...                   monitor_dpi=50)

    Get and plot the new, averaged polarisation vectors

    >>> x_new, y_new, u_new, v_new = get_average_polarisation_in_regions(
    ...     x, y, u, v, image=sublatticeA.image, divide_into=8)
    >>> plot_polarisation_vectors(x_new, y_new, u_new, v_new, monitor_dpi=50,
    ...                   image=sublatticeA.image, save=None, color='r',
    ...                   overlay=False, title='Averaged Vector Arrows')
    '''

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

    return(x_new, y_new, u_new, v_new)


def get_average_polarisation_in_regions_square(x, y, u, v, image,
                                               divide_into=4):
    '''
    Same as `get_average_polarisation_in_regions()` but works for non-square
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
    Four lists: x_new, y_new, u_new, v_new.
    x_new and y_new are the central coordinates of the divided regions.
    u_new and v_new are the averaged polarisation vectors.

    Examples
    --------
    >>> import atomap.api as am
    >>> from temul.polarisation import (
    ...     combine_atom_deviations_from_zone_axes, plot_polarisation_vectors,
    ...     get_average_polarisation_in_regions_square)
    >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeA.find_nearest_neighbors()
    >>> sublatticeA.refine_atom_positions_using_center_of_mass()
    >>> sublatticeA.construct_zone_axes()

    Get and plot the original polarisation vectors of a non-square image

    >>> image = sublatticeA.image[0:200]
    >>> x, y, u, v = combine_atom_deviations_from_zone_axes(sublatticeA,
    ...     save=None)
    >>> plot_polarisation_vectors(x, y, u, v, image=image, save=None,
    ...                   color='r', overlay=False, monitor_dpi=50,
    ...                   title='Actual Vector Arrows')

    Get and plot the new, averaged polarisation vectors for a non-square image

    >>> coords = get_average_polarisation_in_regions_square(
    ...     x, y, u, v, image=image, divide_into=8)
    >>> x_new, y_new, u_new, v_new = coords
    >>> plot_polarisation_vectors(x_new, y_new, u_new, v_new, image=image,
    ...                   color='r', overlay=False, monitor_dpi=50,
    ...                   title='Averaged Vector Arrows', save=None)
    '''

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

    return(x_new, y_new, u_new, v_new)


def get_strain_map(sublattice, zone_axis_index, theoretical_value,
                   sampling=None, units='pix', vmin=None, vmax=None,
                   cmap='inferno', title='Strain Map', filename=None,
                   return_x_y_z=False, **kwargs):
    '''
    Calculate the strain across a zone axis of a sublattice.

    Parameters
    ----------
    sublattice : Atomap Sublattice object
    zone_axis_index : int
        The zone axis you wish to specify. You are indexing
        `sublattice.zones_axis_average_distances[zone_axis_index]`.
    theoretical_value : float
        The theoretical separation between the atoms across (not along) the
        specified zone.
    sampling : float, default None
        Pixel sampling of the image for calibration.
    units : string, default "pix"
        Units of the sampling.
    vmin, vmax, cmap : see Matplotlib for details
    title : string, default "Strain Map"
    filename : string, optional
        If filename is set, the strain signal and plot will be saved.
    return_x_y_z : Bool, default False
        If this is set to True, the `strain_signal` (map), as well as separate
        lists of the x, y, and strain values.
    **kwargs : Matplotlib keyword arguments passed to `imshow()`.

    Returns
    -------
    Strain map as a Hyperspy Signal2D

    Examples
    --------
    >>> import atomap.api as am
    >>> from temul.polarisation import get_strain_map
    >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeA.find_nearest_neighbors()
    >>> sublatticeA.refine_atom_positions_using_center_of_mass()
    >>> sublatticeA.construct_zone_axes()
    >>> strain_map = get_strain_map(sublatticeA, zone_axis_index=0, units='nm',
    ...                             theoretical_value=1.9, sampling=0.1)
    '''

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
        return(strain_signal, x_position, y_position, xy_distance)
    else:
        return(strain_signal)


# Improvement would be to distinguish between horizontal angle e.g., 5 and 175
# degrees. Also 'deg' and 'rad' should be degrees=True/False
def rotation_of_atom_planes(sublattice, zone_axis_index, angle_offset=None,
                            degrees=False, sampling=None, units='pix',
                            vmin=None, vmax=None, cmap='inferno',
                            title='Rotation Map', filename=None,
                            return_x_y_z=False, **kwargs):
    '''
    Calculate a map of the angles between each atom along the atom planes of a
    zone axis.

    Parameters
    ----------
    sublattice : Atomap Sublattice object
    zone_axis_index : int
        The zone axis you wish to specify. You are indexing
        `sublattice.zones_axis_average_distances[zone_axis_index]`.
    angle_offset : float, default None
        The angle which can be considered zero degrees. Useful when the atomic
        planes are at an angle.
    degrees : Bool, default False
        Setting to False will return angle values in radian. Setting to True
        will return angle values in degrees.
    sampling : float, default None
        Pixel sampling of the image for calibration.
    units : string, default "pix"
        Units of the sampling.
    vmin, vmax, cmap : see Matplotlib for details
    title : string, default "Rotation Map"
    filename : string, optional
        If filename is set, the strain signal and plot will be saved.
    return_x_y_z : Bool, default False
        If this is set to True, the rotation_signal (map), as well as separate
        lists of the x, y, and angle values.
    **kwargs : Matplotlib keyword arguments passed to `imshow()`.

    Returns
    -------
    Rotation map as a Hyperspy Signal2D

    Examples
    --------
    >>> import atomap.api as am
    >>> from temul.polarisation import rotation_of_atom_planes
    >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[1]
    >>> sublatticeA.find_nearest_neighbors()
    >>> sublatticeA.refine_atom_positions_using_center_of_mass()
    >>> sublatticeA.construct_zone_axes()
    >>> rotation_map = rotation_of_atom_planes(sublatticeA, 3, degrees=True)

    Use `angle_offset` to effectively change the angle of the horizontal axis
    when calculating angles:

    >>> rotation_map = rotation_of_atom_planes(sublatticeA, 3, angle_offset=45,
    ...                                        degrees=True)

    Use the return_x_y_z parameter when you want to either plot with a
    different style (e.g., contour map), or you want the angle information:

    >>> rotation_map, x, y, angles = rotation_of_atom_planes(
    ...     sublatticeA, 3, degrees=True, return_x_y_z=True)
    >>> mean_angle = np.mean(angles)  # useful for offsetting polar. plots

    '''

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
        return(rotation_signal, x_list, y_list, angles_list)
    else:
        return(rotation_signal)


def ratio_of_lattice_spacings(sublattice, zone_axis_index_A, zone_axis_index_B,
                              ideal_ratio_one=True, sampling=1, units='pix',
                              vmin=None, vmax=None, cmap='inferno',
                              title='Spacings Map', filename=None, **kwargs):
    '''
    Create a ratio map between two zone axes. Useful to see the tetragonality
    or shearing of unit cells.

    Parameters
    ----------
    sublattice : Atomap Sublattice object
    zone_axis_index_A, zone_axis_index_B : int
        The zone axes you wish to specify. You are indexing
        `sublattice.zones_axis_average_distances[zone_axis_index]`.
        The signal created from zone_axis_index_A will be divided by the signal
        created from zone_axis_index_B.
    ideal_ratio_one : Bool, default True
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
    **kwargs : Matplotlib keyword arguments passed to `imshow()`.

    Returns
    -------
    Ratio of lattice spacings map as a Hyperspy Signal2D. It will also plot the
    two lattice spacing maps.

    Examples
    --------
    >>> import atomap.api as am
    >>> from temul.polarisation import ratio_of_lattice_spacings
    >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeA.find_nearest_neighbors()
    >>> sublatticeA.refine_atom_positions_using_center_of_mass()
    >>> sublatticeA.construct_zone_axes()
    >>> ratio_map = ratio_of_lattice_spacings(sublatticeA, 0, 1)

    Use `ideal_ratio_one=False` to view the direction of tetragonality

    >>> ratio_map = ratio_of_lattice_spacings(sublatticeA, 0, 1,
    ...                                       ideal_ratio_one=False)

    '''

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

    return(ratio_signal)


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
                "`vector_rep` must be either 'magnitude' or 'angle'.")
    return(vector_label)


def atom_to_atom_distance_grouped_mean(sublattice, zone_axis_index,
                                       aggregation_axis="y",
                                       slice_thickness=10,
                                       sampling=None, units="pix"):
    '''
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
    >>> from temul.polarisation import atom_to_atom_distance_grouped_mean
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

    '''
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

    return(groupings, grouped_means)


"""
def atom_deviation_from_straight_line_fit(sublattice, save_name='example'):

    for axis_number in range(len(sublattice.zones_axis_average_distances)):

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

                slope_neg_inv = -(1/slope)
                angle = np.arctan(slope_neg_inv)  # * (180/np.pi)

                x1 = atom_plane.start_atom.pixel_x
                y1 = slope*x1 + intercept
                x2 = atom_plane.end_atom.pixel_x
                y2 = slope*x2 + intercept

                p1 = np.array((x1, y1), ndmin=2)
                # end xy coord for straight line fit
                p2 = np.array((x2, y2), ndmin=2)

                atoms_on_plane_list = []
                atom_dist_diff_list = []
                # original_atom position, point an arrow towards it by using
                # original_atom_pos_array and new_atom_diff_array,
                # or away using new_atom_pos_array and -new_atom_diff_array
                for original_atom in original_atoms_array:

                    distance = np.cross(p2-p1, original_atom -
                                        p1) / np.linalg.norm(p2-p1)
                    distance = float(distance)
                    x_diff = distance*np.cos(angle)
                    y_diff = distance*np.sin(angle)

                    x_on_plane = original_atom[0] + x_diff
                    y_on_plane = original_atom[1] + y_diff

                    atoms_on_plane_list.append([x_on_plane, y_on_plane])
                    atom_dist_diff_list.append([x_diff, y_diff])
        #            atoms_not_on_plane_list.append([original_atom])

                original_atom_pos_list.extend(original_atoms_list)
                new_atom_pos_list.extend(atoms_on_plane_list)
                new_atom_diff_list.extend(atom_dist_diff_list)

        original_atom_pos_array = np.array(original_atom_pos_list)
        # new_atom_pos_array = np.array(new_atom_pos_list)

        # this is the difference between the original position and the point on
        # the fitted atom plane line. To get the actual shift direction, just
        # use -new_atom_diff_array. (negative of it!)
        new_atom_diff_array = np.array(new_atom_diff_list)

        '''
Divergent scale beautifying:
    Below we divide the vectors(arrows) into the ones going upward and
    downward. We then want to plot them on a divergent colorbar scale.

    We create two separate color maps with the data from the vector arrows,
    truncated so that the top(darkest) colors aren't included.

    Then plot the downward arrows, with that colorbar,
    plot the upward arrows, with that colorbar.
    Put the colorbar in the right place.
    '''
        arrows_downward = []
        arrows_upward = []
        original_downward = []
        original_upward = []
        for i, component in enumerate(new_atom_diff_array):
            # >0 because the y-axis is flipped in hyperspy data!
            if component[1] > 0:
                arrows_downward.append(component)
                original_downward.append(original_atom_pos_array[i, :])
            else:
                arrows_upward.append(component)
                original_upward.append(original_atom_pos_array[i, :])

        arrows_downward = np.array(arrows_downward)
        arrows_upward = np.array(arrows_upward)
        original_downward = np.array(original_downward)
        original_upward = np.array(original_upward)  # plot the results

        # downward
        color_chart_downward = np.hypot(
            arrows_downward[:, 0], arrows_downward[:, 1])
        color_cmap_downward = plt.get_cmap('Blues')
        color_cmap_downward = _truncate_colormap(
            color_cmap_downward, 0.0, 0.75)

        # upward
        color_chart_upward = np.hypot(arrows_upward[:, 0], arrows_upward[:, 1])
        color_cmap_upward = plt.get_cmap('Reds')
        color_cmap_upward = _truncate_colormap(color_cmap_upward, 0.0, 0.75)

        fig, ax = plt.subplots(figsize=(7, 5))
        plt.title(save_name + '_%i' % axis_number)
        downward = ax.quiver(
            original_downward[:, 0],
            original_downward[:, 1],
            arrows_downward[:, 0],
            arrows_downward[:, 1],
            color_chart_downward,
            cmap=color_cmap_downward,
            angles='xy',
            scale_units='xy',
            scale=None,
            headwidth=7.0,
            headlength=5.0,
            headaxislength=4.5,
            pivot='middle')
        ax.set(aspect='equal')

        # plt.imshow(sublattice.image)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)

        cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.4])
        cbar_downward = plt.colorbar(downward, cax=cbaxes, extend='max',
                                     use_gridspec=False, anchor=(0.0, 0.0),
                                     ticks=[0.4, 0.8, 1.2, 1.6])
        cbar_downward.ax.invert_yaxis()
        cbar_downward.outline.set_visible(False)

        upward = ax.quiver(
            original_upward[:, 0],
            original_upward[:, 1],
            arrows_upward[:, 0],
            arrows_upward[:, 1],
            color_chart_upward,
            cmap=color_cmap_upward,
            angles='xy',
            scale_units='xy',
            scale=None,
            headwidth=7.0,
            headlength=5.0,
            headaxislength=4.5,
            pivot='middle')
        ax.set(aspect='equal')

        cbaxes_upward = fig.add_axes([0.8, 0.5, 0.03, 0.4])
        cbar_upward = plt.colorbar(upward, cax=cbaxes_upward, extend='max',
                                   use_gridspec=False, anchor=(0.0, 1.0),
                                   ticks=[0.0, 0.4, 0.8, 1.2, 1.6])
        cbar_upward.outline.set_visible(False)

        ax.set_xlim(0, sublattice.image.shape[1])
        ax.set_ylim(sublattice.image.shape[0], 0)

        plt.savefig(fname=save_name + '_%i.png' % axis_number,
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)
"""
