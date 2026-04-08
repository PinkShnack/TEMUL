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
