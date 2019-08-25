# author: Eoghan O'Connell

import atomap.api as am
import numpy as np
import my_code_functions_all as temul


def return_z_coordinates(z_thickness,
                         z_bond_length,
                         number_atoms_z=None,
                         fractional_coordinates=True,
                         centered_atoms=True):
    '''
    Produce fractional z-dimension coordinates for a given thickness and bond
    length.

    Parameters
    ----------
    z_thickness : number
        Size (Angstrom) of the z-dimension.
    z_bond_length : number
        Size (Angstrom) of the bond length between adjacent atoms in the
        z-dimension.
    number_atoms_z : integer, default None
        number of atoms in the z-dimension. If this is set to an interger
        value, it will override the use of z_thickness.
    centered_atoms : Bool, default True
        If set to True, the z_coordinates will be centered about 0.5.
        If set to False, the z_coordinate will start at 0.0.

    Returns
    -------
    1D numpy array

    Examples
    --------
    >>> Au_NP_z_coord = return_z_coordinates(z_thickness=20, z_bond_length=1.5)

    '''

    if number_atoms_z is not None:
        # print("number_atoms_z has been specified, using number_atoms_z instead\
        #     of z_thickness")
        z_thickness = number_atoms_z * z_bond_length

    z_coords = np.arange(start=0,
                         stop=z_thickness,
                         step=z_bond_length)

    if fractional_coordinates:
        z_coords = z_coords/z_thickness

    # for centered particles (atoms centered around 0.5 in z)
    if centered_atoms:
        z_coords = z_coords + (1-z_coords.max())/2

    return(z_coords)


'''
add intensity used for getting number, and index for later reference with 
sublattice
'''


def return_xyz_coordintes(x, y,
                          z_thickness, z_bond_length,
                          number_atoms_z=None,
                          fractional_coordinates=True,
                          centered_atoms=True):
    '''
    Produce xyz coordinates for an xy coordinate given the z-dimension
    information. 

    Parameters
    ----------

    x, y : float
        atom position coordinates.

    for other parameters see return_z_coordinates()

    Returns
    -------
    2D numpy array with columns x, y, z

    Examples
    --------
    >>> x, y = 2, 3
    >>> atom_coords = return_xyz_coordintes(x, y,
    ...                         z_thickness=10, 
    ...                         z_bond_length=1.5,
    ...                         number_atoms_z=5)

    '''

    z_coords = return_z_coordinates(number_atoms_z=number_atoms_z,
                                    z_thickness=z_thickness,
                                    z_bond_length=z_bond_length,
                                    fractional_coordinates=True,
                                    centered_atoms=True)

    # One z for each atom in number_atoms for each xy pair
    atom_coords = []
    for z in z_coords:
        atom_coords.append([x, y, z])

    return(np.array(atom_coords))


def convert_numpy_z_coords_to_z_height_string(z_coords):
    """
    Convert from the output of return_z_coordinates(), which is a 1D numpy
    array, to a long string, with which I have set up 
    sublattice.atom_list[i].z_height.

    Examples
    --------

    >>> Au_NP_z_coord = return_z_coordinates(z_thickness=20, z_bond_length=1.5)
    >>> Au_NP_z_height_string = convert_numpy_z_coords_to_z_height_string(Au_NP_z_coord)

    """
    z_string = ""
    for coord in z_coords:

        if z_string == "":
            z_string = z_string + "{0:.{1}f}".format(coord, 6)
        else:
            z_string = z_string + "," + "{0:.{1}f}".format(coord, 6)
        # str(coord).format('.6f')

    return(z_string)


def assign_z_height_to_sublattice(sublattice,
                                  material=None):

    # if material == 'Au_NP_'
        # z_bond_length = function_to_get_material_info
    z_bond_length = 1.5
    z_thickness = 1

    for i in range(0, len(sublattice.atom_list)):
        # if i < 10:
        #     print(str(i) + ' ' + sublattice.atom_list + ' ' + str(i))

        number_atoms_z = temul.split_and_sort_element(
            element=sublattice.atom_list[i].elements)[0][2]

        z_coords = return_z_coordinates(
            number_atoms_z=number_atoms_z,
            z_thickness=z_thickness,
            z_bond_length=z_bond_length,
            fractional_coordinates=True,
            centered_atoms=True)

        z_height = convert_numpy_z_coords_to_z_height_string(z_coords)
        sublattice.atom_list[i].z_height = z_height


'''
# Working Example

sublattice = am.dummy_data.get_simple_cubic_sublattice()
sublattice

element_list = ['Au_5']
elements_in_sublattice = temul.sort_sublattice_intensities(
    sublattice, element_list=element_list)

assign_z_height_to_sublattice(sublattice)

temul.print_sublattice_elements(sublattice)


Au_NP_df = temul.create_dataframe_for_cif(sublattice_list=[sublattice],
                         element_list=element_list)

temul.write_cif_from_dataframe(dataframe=Au_NP_df,
                         filename="Au_NP_test_01",
                         chemical_name_common="Au_NP",
                         cell_length_a=20,
                         cell_length_b=20,
                         cell_length_c=5,
                         cell_angle_alpha=90,
                         cell_angle_beta=90,
                         cell_angle_gamma=90,
                         space_group_name_H_M_alt='P 1',
                         space_group_IT_number=1)

Au_NP_z_height_string.split(",")

for height in Au_NP_z_height_string.split(",")):
    print(i)
'''
