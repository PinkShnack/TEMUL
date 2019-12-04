
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from decimal import Decimal


def find_polarisation_vectors(atom_positions_A, atom_positions_B,
                              save='uv_vectors_array'):
    '''
    Calculate the vectors from atom_positions_A to atom_positions_B.

    Parameters
    ----------
    atom_positions_A, atom_positions_B : list
        Atom positions list in the form [[x1,y1], [x2,y2], [x3,y3]...].
    save : string, default 'uv_vectors_array'
        If set to `save=None`, the array will not be saved.

    Returns
    -------
    two lists: u components and v components.

    Examples
    --------

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


def plot_polarisation_vectors(x, y, u, v, image,
                              sampling=None, units='pix',
                              plot_style='vector',
                              overlay=True, normalise=False,
                              save='polarisation_image', title="",
                              color='yellow', cmap=None, monitor_dpi=96,
                              pivot='middle', angles='xy', scale_units='xy',
                              scale=None, headwidth=3.0, headlength=5.0,
                              headaxislength=4.5):
    '''
    Plot the polarisation vectors.

    Parameters
    ----------
    See matplotlib's quiver() function for more details.

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
        Options are "vector", "colormap", "contour"
    overlay : Bool, default True
        If set to True, the `image` will be plotting behind the arrows
    normalise : Bool, default False
        Normalise the vectors to unit vectors for plotting purposes.
        Magnitude will still be displayed correctly.
    save : string, default "polarisation_image"
        If set to `save=None`, the array will not be saved.
    title : string, default ""
        Title of the plot.
    color : string, default "r"
        Color of the arrows when `plot_style="vector" or "contour".
    cmap : matplotlib colormap, default "viridis"
    monitor_dpi : int, default 96
        The DPI of the monitor, generally 96 pixels. Used to scale the image
        so that large images render correctly. Use a smaller value or
        `monitor_dpi=None` to enlarge too-small images.

    See matplotlib's quiver function for the remaining parameters.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import atomap.api as am
    >>> from temul.polarisation import (
    ...    combine_atom_deviations_from_zone_axes,
    ...    plot_polarisation_vectors)
    >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeA.find_nearest_neighbors()
    >>> sublatticeA.refine_atom_positions_using_center_of_mass()
    >>> sublatticeA.construct_zone_axes()
    >>> x, y, u, v = combine_atom_deviations_from_zone_axes(sublatticeA,
    ...     save=None)

    vector plot with red arrows:

    >>> plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           normalise=False, save=None,
    ...                           plot_style='vector', color='r',
    ...                           overlay=False, title='Vector Arrows',
    ...                           monitor_dpi=50)

    vector plot with red arrows overlaid on the image:

    >>> plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           normalise=False, save=None,
    ...                           plot_style='vector', color='r',
    ...                           overlay=True, monitor_dpi=50)

    vector plot with colormap viridis:

    >>> plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           normalise=False, save=None,
    ...                           plot_style='colormap',
    ...                           overlay=False, cmap='viridis')

    colormap arrows with sampling applied:

    >>> plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           sampling=3.0321, units='pm',
    ...                           normalise=False, plot_style='colormap',
    ...                           overlay=False, save=None, cmap='viridis')

    vector plot with colormap viridis and unit vectors:

    >>> plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           normalise=True, save=None,
    ...                           plot_style='colormap', color='r',
    ...                           overlay=False, cmap='viridis')

    normalised vectors on a contourf map:

    >>> plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           normalise=True, plot_style='contour',
    ...                           overlay=False, pivot='middle',
    ...                           color='darkgray', cmap='viridis', save=None)

    '''

    u, v = np.array(u), np.array(v)

    if sampling is not None:
        u, v = u * sampling, v * sampling

    # for ax.quiver optional C paramater, we need to set this to
    # something. None doesn't work!
    vector_mags = get_vector_magnitudes(u, v, sampling=sampling)

    if normalise:
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
            x, y, u, v, vector_mags, color=color, cmap=cmap,
            pivot=pivot, angles=angles, scale_units=scale_units,
            scale=scale, headwidth=headwidth,
            headlength=headlength, headaxislength=headaxislength)

        norm = colors.Normalize(vmin=np.min(vector_mags),
                                vmax=np.max(vector_mags))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(mappable=sm)
        cbar.ax.set_ylabel('Vector Magnitude ({})'.format(units))

    elif plot_style == 'contour':
        if cmap is None:
            cmap = 'viridis'

        contour_map = plt.tricontourf(x, y, vector_mags, cmap=cmap)

        ax.quiver(
            x, y, u, v, color=color, pivot=pivot,
            angles=angles, scale_units=scale_units,
            scale=scale, headwidth=headwidth,
            headlength=headlength, headaxislength=headaxislength)

    ax.set(aspect='equal')
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)

    if plot_style == 'contour':
        cbar = plt.colorbar(mappable=contour_map)
        cbar.ax.set_ylabel('Vector Magnitude ({})'.format(units))
    if overlay:
        plt.imshow(image)

    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    # plt.tight_layout()
    if save is not None:
        plt.savefig(fname=save + '_' + plot_style + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)


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
    >>> u, v = [4,3,2,5,6], [8,5,2,1,1] # list input
    >>> vector_mags = get_vector_magnitudes(u,v)
    >>> u, v = np.array(u), np.array(v) # np input
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
    delete atom_planes from a zone axis. Can choose whether to delete
    every second, third etc., and the offset from the zero index.

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
    >>> import atomap.api as am
    >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> delete_atom_planes_from_sublattice(sublattice=sublatticeA,
    ...                        zone_axis_index=0,
    ...                         divisible_by=3,
    ...                         offset_from_zero=1)
    >>> sublatticeA.plot_planes()

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


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def atom_deviation_from_straight_line_fit(sublattice,
                                          axis_number: int = 0,
                                          save: str = ''):
    '''
    delete atom_planes from a zone axis. Can choose whether to delete
    every second, third etc., and the offset from the zero index.

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
    >>> from temul.polarisation import atom_deviation_from_straight_line_fit
    >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeA.find_nearest_neighbors()
    >>> sublatticeA.refine_atom_positions_using_center_of_mass()
    >>> sublatticeA.construct_zone_axes()
    >>> x,y,u,v = atom_deviation_from_straight_line_fit(sublatticeA, save=None)

    This polarisation can then be visualised in plot_polarisation_vectors()

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
    #            atoms_not_on_plane_list.append([original_atom])

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


def plot_atom_deviation_from_all_zone_axes(
        sublattice, image=None, sampling=None, units='pix',
        plot_style='vector', overlay=True, normalise=False,
        save='atom_deviation', title="", color='yellow', cmap=None,
        pivot='middle', angles='xy', scale_units='xy', scale=None,
        headwidth=3.0, headlength=5.0, headaxislength=4.5, monitor_dpi=96):
    '''
    # need to add the truncated colormap version: divergent plot.

    Plot the atom deviation from a straight line fit for all zone axes
    constructed by an Atomap sublattice object.

    Parameters
    ----------
    sublattice : Atomap Sublattice object
    For all other parameters see plot_polarisation_vectors()

    Examples
    --------
    >>> import atomap.api as am
    >>> from temul.polarisation import plot_atom_deviation_from_all_zone_axes
    >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeA.find_nearest_neighbors()
    >>> sublatticeA.refine_atom_positions_using_center_of_mass()
    >>> sublatticeA.construct_zone_axes()
    >>> plot_atom_deviation_from_all_zone_axes(sublatticeA,
    ...     plot_style='vector', save=None)

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
            plot_style=plot_style, overlay=overlay, normalise=normalise,
            save=save, title=title, color=color, cmap=cmap, pivot=pivot,
            angles=angles, scale_units=scale_units, scale=scale,
            headwidth=headwidth, headlength=headlength,
            headaxislength=headaxislength, monitor_dpi=monitor_dpi)


def combine_atom_deviations_from_zone_axes(
        sublattice, image=None, sampling=None, units='pix',
        plot_style='vector', overlay=True, normalise=False,
        save='atom_deviation', title="", color='yellow', cmap=None,
        pivot='middle', angles='xy', scale_units='xy', scale=None,
        headwidth=3.0, headlength=5.0, headaxislength=4.5, monitor_dpi=96):
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
    >>> from temul.polarisation import combine_atom_deviations_from_zone_axes
    >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeA.find_nearest_neighbors()
    >>> sublatticeA.refine_atom_positions_using_center_of_mass()
    >>> sublatticeA.construct_zone_axes()
    >>> x,y,u,v = combine_atom_deviations_from_zone_axes(
    ...     sublatticeA, save=None)
    >>> plot_polarisation_vectors(u, v, x, y, save=None,
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

    for axis_number in range(len(sublattice.zones_axis_average_distances)):

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
            plot_style=plot_style, overlay=overlay, normalise=normalise,
            save=save, title=title, color=color, cmap=cmap, pivot=pivot,
            angles=angles, scale_units=scale_units, scale=scale,
            headwidth=headwidth, headlength=headlength,
            headaxislength=headaxislength, monitor_dpi=monitor_dpi)

    return(x, y, u, v)


def get_average_polarisation_in_regions(x, y, u, v, image, divide_into=8):
    '''
    This function splits the image into a certain number of regions and
    averages and plots the polarisation vectors in those regions.

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
    x_new and y_new are the certain coordinates of the regions.
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
    ...                   normalise=False, save=None,
    ...                   plot_style='vector', color='r',
    ...                   overlay=False, title='Actual Vector Arrows',
    ...                   monitor_dpi=50)

    Get and plot the new, averaged polarisation vectors

    >>> x_new, y_new, u_new, v_new = get_average_polarisation_in_regions(
    ...     x, y, u, v, image=sublatticeA.image, divide_into=8)
    >>> plot_polarisation_vectors(x_new, y_new, u_new, v_new,
    ...                   image=sublatticeA.image,
    ...                   normalise=False, save=None,
    ...                   plot_style='vector', color='r',
    ...                   overlay=False, title='Averaged Vector Arrows',
    ...                   monitor_dpi=50)
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
        color_cmap_downward = truncate_colormap(color_cmap_downward, 0.0, 0.75)

        # upward
        color_chart_upward = np.hypot(arrows_upward[:, 0], arrows_upward[:, 1])
        color_cmap_upward = plt.get_cmap('Reds')
        color_cmap_upward = truncate_colormap(color_cmap_upward, 0.0, 0.75)

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
