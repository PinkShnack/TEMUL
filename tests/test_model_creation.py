from collections import Counter

import numpy as np
import pandas as pd
import pytest

import temul.external.atomap_devel_012.dummy_data as atomap_dd
import temul.model_creation as mc


def test_count_element_in_pandas_df_handles_stacked_columns():
    dataframe = pd.DataFrame(
        data=[[1, 2], [3, 4]],
        columns=['Mo_1.S_2', 'S_1'])

    counts = mc.count_element_in_pandas_df('S', dataframe)

    assert counts == Counter({1: 10, 0: 4})


def test_count_all_individual_elements_returns_dict_of_counters():
    dataframe = pd.DataFrame(
        data=[[9, 4], [8, 6]],
        columns=['Se_1', 'Mo_1'])

    counts = mc.count_all_individual_elements(['Mo', 'Ti'], dataframe)

    assert counts['Mo'] == Counter({1: 6, 0: 4})
    assert counts['Ti'] == Counter()


def test_count_atoms_in_sublattice_list_counts_elements():
    atom_lattice = atomap_dd.get_simple_atom_lattice_two_sublattices()
    sub1, sub2 = atom_lattice.sublattice_list

    for atom in sub1.atom_list:
        atom.elements = 'Ti_2'
    for atom in sub2.atom_list:
        atom.elements = 'Cl_1'

    counts = mc.count_atoms_in_sublattice_list([sub1, sub2])

    assert counts == Counter(
        {'Ti_2': len(sub1.atom_list), 'Cl_1': len(sub2.atom_list)})


def test_compare_count_atoms_in_sublattice_list_true_false_and_invalid():
    assert mc.compare_count_atoms_in_sublattice_list(
        [Counter({'Mo': 2}), Counter({'Mo': 2})]) is True
    assert mc.compare_count_atoms_in_sublattice_list(
        [Counter({'Mo': 2}), Counter({'Mo': 3})]) is False
    with pytest.raises(ValueError, match="must be 2"):
        mc.compare_count_atoms_in_sublattice_list([Counter({'Mo': 2})])


def test_auto_generate_sublattice_element_list_for_single_element_column():
    element_list = mc.auto_generate_sublattice_element_list(
        material_type='single_element_column',
        elements='Au',
        max_number_atoms_z=3)

    assert element_list == ['Au_0', 'Au_1', 'Au_2', 'Au_3']


def test_return_z_coordinates_for_top_and_center_layouts():
    top = mc.return_z_coordinates(
        z_thickness=8,
        z_bond_length=2,
        number_atoms_z=2,
        max_number_atoms_z=4,
        fractional_coordinates=True,
        atom_layout='top')
    centered = mc.return_z_coordinates(
        z_thickness=8,
        z_bond_length=2,
        number_atoms_z=2,
        max_number_atoms_z=4,
        fractional_coordinates=True,
        atom_layout='center')

    assert np.allclose(top, np.array([0.5, 0.75]))
    assert np.allclose(centered, np.array([0.375, 0.625]))


def test_return_z_coordinates_rejects_invalid_requests():
    with pytest.raises(ValueError, match="greater than max_number_atoms_z"):
        mc.return_z_coordinates(
            z_thickness=8,
            z_bond_length=2,
            number_atoms_z=5,
            max_number_atoms_z=4)

    with pytest.raises(ValueError, match="Only 'bot', 'center' and 'top'"):
        mc.return_z_coordinates(
            z_thickness=8,
            z_bond_length=2,
            number_atoms_z=2,
            max_number_atoms_z=4,
            atom_layout='invalid')


def test_convert_numpy_z_coords_to_z_height_string():
    z_string = mc.convert_numpy_z_coords_to_z_height_string(
        np.array([0.25, 0.5, 0.75]))

    assert z_string == '0.250000,0.500000,0.750000'


def test_get_max_number_atoms_z():
    sublattice = atomap_dd.get_simple_cubic_sublattice()
    for index, atom in enumerate(sublattice.atom_list):
        atom.elements = 'Mo_2' if index % 2 == 0 else 'Mo_5'

    assert mc.get_max_number_atoms_z(sublattice) == 5


def test_get_most_common_sublattice_element_and_z_height():
    sublattice = atomap_dd.get_simple_cubic_sublattice()
    for index, atom in enumerate(sublattice.atom_list):
        if index % 3 == 0:
            atom.elements = 'Ti_3'
            atom.z_height = '0.3,0.6,0.9'
        else:
            atom.elements = 'Ti_2'
            atom.z_height = '0.3,0.6'

    assert mc.get_most_common_sublattice_element(sublattice) == 'Ti_2'
    assert mc.get_most_common_sublattice_element(
        sublattice, info='z_height'
    ) == '0.3,0.6'


def test_change_sublattice_atoms_via_intensity_updates_element_and_z_height():
    sublattice = atomap_dd.get_simple_cubic_sublattice()
    for atom in sublattice.atom_list:
        atom.elements = 'Mo_2'
        atom.z_height = [0.25, 0.75]

    mc.change_sublattice_atoms_via_intensity(
        sublattice=sublattice,
        image_diff_array=np.array([[0, 0, 0, 1.0]]),
        darker_or_brighter=0,
        element_list=['Mo_0', 'Mo_1', 'Mo_2'],
    )

    assert sublattice.atom_list[0].elements == 'Mo_1'
    assert sublattice.atom_list[0].z_height == [0.5]


def test_change_sublattice_atoms_via_intensity_fills_missing_elements():
    sublattice = atomap_dd.get_simple_cubic_sublattice()
    for atom in sublattice.atom_list:
        atom.elements = 'Mo_1'
        atom.z_height = '0.5'
    sublattice.atom_list[0].elements = ''
    sublattice.atom_list[0].z_height = ''

    mc.change_sublattice_atoms_via_intensity(
        sublattice=sublattice,
        image_diff_array=np.array([]),
        darker_or_brighter=1,
        element_list=['Mo_0', 'Mo_1', 'Mo_2'],
    )

    assert sublattice.atom_list[0].elements == 'Mo_1'


def test_change_sublattice_atoms_via_intensity_rejects_unknown_element():
    sublattice = atomap_dd.get_simple_cubic_sublattice()
    sublattice.atom_list[0].elements = 'Ti_9'
    sublattice.atom_list[0].z_height = '0.5'

    with pytest.raises(ValueError, match="isn't in the element_list"):
        mc.change_sublattice_atoms_via_intensity(
            sublattice=sublattice,
            image_diff_array=np.array([[0, 0, 0, 1.0]]),
            darker_or_brighter=1,
            element_list=['Mo_0', 'Mo_1', 'Mo_2'],
        )


def test_find_middle_and_edge_intensities_with_numeric_standard():
    sublattice = atomap_dd.get_simple_cubic_sublattice()
    middle, limits = mc.find_middle_and_edge_intensities(
        sublattice=sublattice,
        element_list=['Mo_1', 'Mo_2', 'Mo_3'],
        standard_element=42.0,
        scaling_exponent=1.0,
    )

    assert middle == [1.0, 2.0, 3.0]
    assert limits == [0.0, 1.5, 2.5, 3.5]


def test_find_middle_and_edge_intensities_for_background():
    middle, limits = mc.find_middle_and_edge_intensities_for_background(
        elements_from_sub1=['Mo_1', 'Mo_2'],
        elements_from_sub2=['S_1'],
        sub1_mode=2.0,
        sub2_mode=3.0,
        element_list_sub1=['Mo_1', 'Mo_2'],
        element_list_sub2=['S_1'],
        middle_intensity_list_sub1=[1.0, 2.0],
        middle_intensity_list_sub2=[0.5],
    )

    assert middle == [0.0, 1.5, 2.0, 4.0]
    assert limits == [0.0, 0.75, 1.75, 3.0, 5.0]


def test_sort_sublattice_intensities_assigns_elements_without_scaling(
        monkeypatch):
    sublattice = atomap_dd.get_simple_cubic_sublattice()
    monkeypatch.setattr(
        mc,
        'get_sublattice_intensity',
        lambda **kwargs: np.array([0.25] * 200 + [0.75] * 200),
    )

    elements = mc.sort_sublattice_intensities(
        sublattice=sublattice,
        element_list=['Mo_1', 'Mo_2'],
        middle_intensity_list=[0.25, 0.75],
        limit_intensity_list=[0.0, 0.5, 1.0],
        scalar_method=1.0,
    )

    assert elements.count('Mo_1') == 200
    assert elements.count('Mo_2') == 200


def test_correct_background_elements_and_print_sublattice_elements():
    sublattice = atomap_dd.get_simple_cubic_sublattice()
    sublattice.find_nearest_neighbors()
    sublattice.get_atom_column_amplitude_max_intensity(percent_to_nn=0.4)
    sublattice.get_atom_column_amplitude_min_intensity(percent_to_nn=0.4)
    for index, atom in enumerate(sublattice.atom_list):
        atom.elements = 'Mo_1' if index < 3 else 'S_1'
        atom.z_height = '0.5'
        atom.amplitude_mean_intensity = atom.amplitude_max_intensity
        atom.amplitude_total_intensity = atom.amplitude_max_intensity

    mc.correct_background_elements(sublattice)
    printed = mc.print_sublattice_elements(sublattice, number_of_lines=2)

    assert sublattice.atom_list[3].elements == 'H_0'
    assert len(printed) == 2
    assert printed[0][0] == 'Mo_1'


def test_return_xyz_coordinates_and_assign_z_height_to_sublattice():
    coords = mc.return_xyz_coordinates(
        2,
        3,
        z_thickness=8,
        z_bond_length=2,
        number_atoms_z=3,
    )
    assert coords.shape == (3, 3)
    assert np.allclose(coords[:, :2], np.array([[2, 3], [2, 3], [2, 3]]))

    sublattice = atomap_dd.get_simple_cubic_sublattice()
    for index, atom in enumerate(sublattice.atom_list):
        atom.elements = 'Mo_1' if index % 2 == 0 else 'Mo_2'
    mc.assign_z_height_to_sublattice(sublattice, z_bond_length=0.5)

    assert sublattice.atom_list[0].z_height == '0.000000'
    assert sublattice.atom_list[1].z_height == '0.000000,0.500000'


def test_create_dataframe_for_cif_and_change_sublattice_pseudo_inplace():
    sublattice = atomap_dd.get_simple_cubic_sublattice()
    sublattice.atom_list[0].elements = 'Mo_2'
    sublattice.atom_list[0].z_height = '0.25,0.75'
    for atom in sublattice.atom_list[1:]:
        atom.elements = 'Mo_0'
        atom.z_height = '0.5'

    dataframe = mc.create_dataframe_for_cif([sublattice], ['Mo_0', 'Mo_2'])
    assert len(dataframe) == 2
    assert dataframe['_atom_site_label'].tolist() == ['Mo', 'Mo']

    sublattice.atom_list[0].elements = 'Ti_1'
    new_sublattice = mc.change_sublattice_pseudo_inplace(
        [[4, 4], [5, 5]],
        sublattice,
    )
    assert new_sublattice.atom_list[0].elements == 'Ti_1'
    assert len(new_sublattice.atom_list) == len(sublattice.atom_list) + 2
