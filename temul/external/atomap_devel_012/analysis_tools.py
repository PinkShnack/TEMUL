import numpy as np
import temul.external.atomap_devel_012.tools as to


def get_neighbor_middle_position(atom, za0, za1):
    """Find the middle point between four neighboring atoms.

    The neighbors are found by moving one step along the atom planes
    belonging to za0 and za1.

    So atom planes must be constructed first.

    Parameters
    ----------
    atom : Atom_Position object
    za0 : tuple
    za1 : tuple

    Return
    ------
    middle_position : tuple
        If the atom is at the edge by being the last atom in the
        atom plane, False is returned.

    Examples
    --------
    >>> import temul.external.atomap_devel_012.analysis_tools as an
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.construct_zone_axes()
    >>> za0 = sublattice.zones_axis_average_distances[0]
    >>> za1 = sublattice.zones_axis_average_distances[1]
    >>> atom = sublattice.atom_list[33]
    >>> middle_position = an.get_neighbor_middle_position(atom, za0, za1)

    """
    atom00 = atom
    atom01 = atom.get_next_atom_in_zone_vector(za0)
    atom10 = atom.get_next_atom_in_zone_vector(za1)
    middle_position = False
    if not (atom01 is False):
        if not (atom10 is False):
            atom11 = atom10.get_next_atom_in_zone_vector(za0)
            if not (atom11 is False):
                middle_position = to.get_point_between_four_atoms((
                    atom00, atom01, atom10, atom11))
    return middle_position


def get_middle_position_list(sublattice, za0, za1):
    """Find the middle point between all four neighboring atoms.

    The neighbors are found by moving one step along the atom planes
    belonging to za0 and za1.

    So atom planes must be constructed first.

    Parameters
    ----------
    sublattice : Sublattice object
    za0 : tuple
    za1 : tuple

    Return
    ------
    middle_position_list : list

    Examples
    --------
    >>> import temul.external.atomap_devel_012.analysis_tools as an
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.construct_zone_axes()
    >>> za0 = sublattice.zones_axis_average_distances[0]
    >>> za1 = sublattice.zones_axis_average_distances[1]
    >>> middle_position_list = an.get_middle_position_list(
    ...     sublattice, za0, za1)

    """
    middle_position_list = []
    for atom in sublattice.atom_list:
        middle_pos = get_neighbor_middle_position(atom, za0, za1)
        if not (middle_pos is False):
            middle_position_list.append(middle_pos)
    return middle_position_list


def get_vector_shift_list(sublattice, position_list):
    """Find the atom shifts from a central position.

    Useful for finding polarization in B-cations in a perovskite structure.

    Parameters
    ----------
    sublattice : Sublattice object
    position_list : list
        [[x0, y0], [x1, y1], ...]

    Returns
    -------
    vector_list : list
        In the form [[x0, y0, dx0, dy0], [x1, y1, dx1, dy1]...]

    Example
    -------
    >>> import temul.external.atomap_devel_012.analysis_tools as an
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.construct_zone_axes()
    >>> za0 = sublattice.zones_axis_average_distances[0]
    >>> za1 = sublattice.zones_axis_average_distances[1]
    >>> middle_position_list = an.get_middle_position_list(
    ...     sublattice, za0, za1)
    >>> vector_list = an.get_vector_shift_list(
    ...     sublattice, middle_position_list)

    """
    vector_list = []
    for position in position_list:
        dist = np.hypot(
            np.array(sublattice.x_position) - position[0],
            np.array(sublattice.y_position) - position[1])
        atom_b = sublattice.atom_list[dist.argmin()]
        vector = (position[0], position[1],
                  position[0] - atom_b.pixel_x,
                  position[1] - atom_b.pixel_y)
        vector_list.append(vector)
    return vector_list
