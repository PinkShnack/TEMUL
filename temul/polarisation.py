
import numpy as np
import matplotlib.pyplot as plt


def find_polarisation_vectors(atom_positions_A, atom_positions_B,
                              save='uv_vectors_array'):
    '''

    other params: unit vectoring,
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


def plot_polarisation_vectors(u, v, sublattice, image=None,
                              plot_style=['overlay'],
                              save='polarisation_image',
                              pivot='middle', color='yellow',
                              angles='xy', scale_units='xy',
                              scale=None, headwidth=3.0,
                              width=0.005, headlength=5.0,
                              headaxislength=4.5, title=""):

    if image is None:
        image = sublattice.image
    else:
        image = image

    if "vectors" in plot_style:

        _, ax = plt.subplots()
        ax.quiver(
            sublattice.x_position,
            sublattice.y_position,
            u,
            v,
            angles=angles,
            scale_units=scale_units,
            scale=scale,
            width=width,
            headwidth=headwidth,
            headlength=headlength,
            headaxislength=headaxislength,
            pivot=pivot,
            color=color)
        ax.set(aspect='equal')
        ax.set_xlim(0, image.data[1])
        ax.set_ylim(image.data[0], 0)
        plt.title(title)
        plt.tight_layout()
        if save is not None:
            plt.savefig(fname=save + '.png',
                        transparent=True, frameon=False, bbox_inches='tight',
                        pad_inches=None, dpi=300, labels=False)

    if "overlay" in plot_style:
        _, ax = plt.subplots()
        ax.quiver(
            sublattice.x_position,
            sublattice.y_position,
            u,
            v,
            angles=angles,
            scale_units=scale_units,
            scale=scale,
            width=width,
            headwidth=headwidth,
            headlength=headlength,
            headaxislength=headaxislength,
            pivot=pivot,
            color=color)
        ax.set(aspect='equal')
        ax.set_xlim(0, image.data[1])
        ax.set_ylim(image.data[0], 0)
        plt.imshow(sublattice.image)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.title(title)
        plt.tight_layout()
        if save is not None:
            plt.savefig(fname=save + '.png',
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
