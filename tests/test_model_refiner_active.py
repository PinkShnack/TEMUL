from collections import Counter
import sys
import types

import numpy as np
import pandas as pd
import pytest
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


def test_model_refiner_comparison_image_validation_and_calibration_helpers(
        monkeypatch):
    refiner = _build_refiner()

    refiner.set_calibration_area([[2, 3], [4, 5]])
    refiner.set_calibration_separation(9)

    assert refiner.calibration_area == [[2, 3], [4, 5]]
    assert refiner.calibration_separation == 9

    monkeypatch.setattr(
        'temul.model_refiner.choose_points_on_image',
        lambda image: [[10, 11], [12, 13]],
    )
    refiner.set_calibration_area()
    assert refiner.calibration_area == [[10, 11], [12, 13]]

    refiner.comparison_image = None
    with pytest.raises(ValueError, match='comparison_image attribute has not'):
        refiner._comparison_image_warning(error_message=['None'])

    refiner.comparison_image = Signal2D(np.zeros((2, 2), dtype=float))
    with pytest.raises(ValueError, match='must have the same shape'):
        refiner._comparison_image_warning(error_message=['wrong_size'])


def test_model_refiner_create_simulation_calls_expected_pipeline(monkeypatch):
    refiner = _build_refiner()
    calls = {}

    def fake_create_dataframe_for_xyz(**kwargs):
        calls['create_dataframe_for_xyz'] = kwargs

    def fake_simulate_with_prismatic(**kwargs):
        calls['simulate_with_prismatic'] = kwargs

    def fake_load_prismatic_mrc_with_hyperspy(filename, save_name=None):
        calls['load_prismatic_mrc_with_hyperspy'] = {
            'filename': filename,
            'save_name': save_name,
        }
        return Signal2D(refiner.reference_image.data.copy())

    def fake_compare_two_image_and_create_filtered_image(**kwargs):
        calls['compare_two_image_and_create_filtered_image'] = kwargs
        return kwargs['image_to_filter']

    def fake_calibrate_intensity_distance_with_sublattice_roi(**kwargs):
        calls['calibrate_intensity_distance_with_sublattice_roi'] = kwargs

    monkeypatch.setattr(
        'temul.model_refiner.create_dataframe_for_xyz',
        fake_create_dataframe_for_xyz,
    )
    monkeypatch.setattr(
        'temul.model_refiner.simulate_with_prismatic',
        fake_simulate_with_prismatic,
    )
    monkeypatch.setattr(
        'temul.model_refiner.load_prismatic_mrc_with_hyperspy',
        fake_load_prismatic_mrc_with_hyperspy,
    )
    monkeypatch.setattr(
        'temul.model_refiner.compare_two_image_and_create_filtered_image',
        fake_compare_two_image_and_create_filtered_image,
    )
    monkeypatch.setattr(
        'temul.model_refiner.calibrate_intensity_distance_with_sublattice_roi',
        fake_calibrate_intensity_distance_with_sublattice_roi,
    )

    refiner.create_simulation(
        filter_image=True,
        calibrate_image=True,
        filename='unit_test_refiner',
        mask_radius='auto',
    )

    assert calls['create_dataframe_for_xyz']['filename'] == (
        'unit_test_refiner_xyz_file'
    )
    assert calls['create_dataframe_for_xyz']['x_size'] == pytest.approx(
        refiner.image_xyz_sizes[0]
    )
    assert calls['simulate_with_prismatic']['xyz_filename'] == (
        'unit_test_refiner_xyz_file.xyz'
    )
    assert calls['simulate_with_prismatic']['filename'] == (
        'unit_test_refiner_mrc_file'
    )
    assert calls['load_prismatic_mrc_with_hyperspy']['filename'] == (
        'prism_2Doutput_unit_test_refiner_mrc_file.mrc'
    )
    assert calls['compare_two_image_and_create_filtered_image'][
        'mask_radius'
    ] == pytest.approx(np.mean(refiner.auto_mask_radius))
    assert calls['calibrate_intensity_distance_with_sublattice_roi'][
        'image'
    ] is refiner.comparison_image
    assert len(refiner.error_between_images_history) == 1


def test_model_refiner_calibrate_comparison_image_uses_auto_mask_radius(
        monkeypatch):
    refiner = _build_refiner()
    calls = {}

    def fake_calibrate_intensity_distance_with_sublattice_roi(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(
        'temul.model_refiner.calibrate_intensity_distance_with_sublattice_roi',
        fake_calibrate_intensity_distance_with_sublattice_roi,
    )

    refiner.calibrate_comparison_image(mask_radius='auto', refine=False)

    assert calls['image'] is refiner.comparison_image
    assert calls['mask_radius'] == pytest.approx(
        np.mean(refiner.auto_mask_radius)
    )
    assert calls['refine'] is False
