from collections import Counter
import sys
import types

import pandas as pd
from hyperspy.signals import Signal2D

import temul.external.atomap_devel_012.dummy_data as atomap_dd


def _build_refiner():
    sys.modules.setdefault('pyprismatic', types.SimpleNamespace())
    from temul.model_refiner import Model_Refiner

    atom_lattice = atomap_dd.get_simple_atom_lattice_two_sublattices()
    sub1, sub2 = atom_lattice.sublattice_list

    for atom in sub1.atom_list:
        atom.elements = 'Ti_2'
    for atom in sub2.atom_list:
        atom.elements = 'Cl_1'

    return Model_Refiner(
        {
            sub1: ['Ti_0', 'Ti_1', 'Ti_2', 'Ti_3'],
            sub2: ['Cl_0', 'Cl_1', 'Cl_2', 'Cl_3'],
        },
        comparison_image=Signal2D(atom_lattice.image.copy()),
        name='example',
    )


def test_model_refiner_element_count_dataframes():
    refiner = _build_refiner()

    df_configs = refiner.get_element_count_as_dataframe()
    df_individual = refiner.get_individual_elements_as_dataframe()
    df_combined = refiner.combine_individual_and_element_counts_as_dataframe()

    assert isinstance(df_configs, pd.DataFrame)
    assert list(df_configs.index) == ['0 Initial State']
    assert 'Ti_2' in df_configs.columns
    assert 'Cl_1' in df_configs.columns
    assert 'Ti' in df_individual.columns
    assert 'Cl' in df_individual.columns
    assert set(df_configs.columns).issubset(df_combined.columns)
    assert set(df_individual.columns).issubset(df_combined.columns)


def test_model_refiner_history_updates_and_comparison():
    refiner = _build_refiner()

    assert refiner.compare_latest_element_counts() is False

    refiner.update_element_count_and_refinement_history('repeat')
    assert refiner.compare_latest_element_counts() is True

    refiner.element_count_history_list.append(Counter({'Ti_2': 1}))
    assert refiner.compare_latest_element_counts() is False
