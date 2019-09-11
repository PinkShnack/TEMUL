
import atomap.api as am


def delete_atom_planes_from_sublattice(sublattice,
                                       zone_axis_index=0,
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
import atomap.api as am
atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
sublatticeA = atom_lattice.sublattice_list[0]
delete_atom_planes_from_sublattice(sublattice=sublatticeA,
                            zone_axis_index=0,
                            divisible_by=3,
                            offset_from_zero=1)
sublatticeA.plot_planes()

    '''
    sublattice.construct_zone_axes()

    zone_vec_needed = sublattice.zones_axis_average_distances[zone_axis_index]

    atom_plane_index_delete = []
    opposite_list = []
    for i, _ in enumerate(sublattice.atom_planes_by_zone_vector[zone_vec_needed]):
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
                               if index < len(sublattice.atom_planes_by_zone_vector[zone_vec_needed])]

    if opposite:
        opposite_list = [
            index for index in opposite_list if index not in atom_plane_index_delete]
        atom_plane_index_delete = opposite_list
    # reversal needed because first it will delete 0, then 1 will become 0.
    # Then it will delete 3, which is the wrong one! (should have been 2)
    atom_plane_index_delete.sort(reverse=True)
    # print(atom_plane_index_delete)
    # print(opposite_list)
    for i in atom_plane_index_delete:
        del sublattice.atom_planes_by_zone_vector[zone_vec_needed][i]


atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
sublatticeA = atom_lattice.sublattice_list[0]
delete_atom_planes_from_sublattice(sublattice=sublatticeA,
                                   zone_axis_index=0,
                                   divisible_by=3,
                                   offset_from_zero=0,
                                   opposite=True)
sublatticeA.plot_planes()
