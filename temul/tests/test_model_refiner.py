'''
import atomap_devel_012.api as am
import model_refiner as mr
import numpy as np
from collections import Counter
from scipy.ndimage import gaussian_filter
import hyperspy
import hyperspy.api as hs

atom_lattice = am.dummy_data.get_simple_atom_lattice_two_sublattices()
sub1 = atom_lattice.sublattice_list[0]
sub2 = atom_lattice.sublattice_list[1]
for i in range(0, len(sub1.atom_list)):
    sub1.atom_list[i].elements = 'Ti_2'
for i in range(0, len(sub2.atom_list)):
    sub2.atom_list[i].elements = 'Cl_1'
sub1_element_list = ['Ti_0', 'Ti_1', 'Ti_2', 'Ti_3']
sub2_element_list = ['Cl_0', 'Cl_1', 'Cl_2', 'Cl_3']

sub1.atom_list[2].elements = 'Ti_1'
sub1.atom_list[5].elements = 'Ti_1'

refiner_dict = {sub1: sub1_element_list,
                sub2: sub2_element_list}


def test_comparison_image():

    blurred_image = hyperspy._signals.signal2d.Signal2D(
        gaussian_filter(atom_lattice.image, 3))
    atom_lattice_refiner = Model_Refiner(refiner_dict, blurred_image)

    assert isinstance(atom_lattice_refiner.comparison_image,
                      hyperspy._signals.signal2d.Signal2D)


def test_repeating_intensity_refinement_no_change():
    atom_lattice = am.dummy_data.get_simple_atom_lattice_two_sublattices(
        image_noise=True
    )
    sub1 = atom_lattice.sublattice_list[0]
    sub2 = atom_lattice.sublattice_list[1]

    for i in range(0, len(sub1.atom_list)):
        sub1.atom_list[i].elements = 'Ti_2'
    for i in range(0, len(sub2.atom_list)):
        sub2.atom_list[i].elements = 'Cl_1'

    sub1_element_list = ['Ti_0', 'Ti_1', 'Ti_2', 'Ti_3']
    sub2_element_list = ['Cl_0', 'Cl_1', 'Cl_2', 'Cl_3']

    refiner_dict = {sub1: sub1_element_list,
                    sub2: sub2_element_list}

    blurred_image = gaussian_filter(atom_lattice.image, 3)
    atom_lattice_refiner = Model_Refiner(refiner_dict, blurred_image)

    dataframe_before = atom_lattice_refiner.get_element_count_as_dataframe()

    atom_lattice_refiner.repeating_intensity_refinement(
        n=5,
        sublattices='all',
        comparison_image='default',
        change_sublattice=True,
        percent_to_nn=0.40,
        mask_radius=None,
        filename=None)

    dataframe_after = atom_lattice_refiner.get_element_count_as_dataframe()

    dataframe_before.append(dataframe_before[0:])

    assert dataframe_after.equals(dataframe_before.append(
        dataframe_before[0:], ignore_index=True))


def test_image_difference_intensity_model_refiner():
    # create a fake simulated image
    sim_sublattice = am.dummy_data.get_simple_cubic_sublattice()
    sim_image = sim_sublattice.signal

    # use the positions to create a vacancy filled experimental sublattice
    vac_sublattice = am.dummy_data.get_simple_cubic_with_vacancies_sublattice()
    sublattice = am.Sublattice(np.array(sim_sublattice.atom_positions).T,
                               vac_sublattice.image)

    for i in range(0, len(sublattice.atom_list)):
        sublattice.atom_list[i].elements = 'Mo_1'
        sublattice.atom_list[i].z_height = [0.5]
    element_list = ['H_0', 'Mo_0', 'Mo_1', 'Mo_2']

    # create a Model_Refiner object
    refiner_dict = {sublattice: element_list}
    atom_lattice_refiner = Model_Refiner(refiner_dict, sim_image,
                                         name='simple example')

    dataframe_before = atom_lattice_refiner.get_element_count_as_dataframe()
    atom_lattice_refiner.image_difference_intensity_model_refiner(
        sublattices='all',
        comparison_image='fault',
        change_sublattice=True,
        percent_to_nn=0.40,
        mask_radius=None,
        filename=None)
    dataframe_after = atom_lattice_refiner.get_element_count_as_dataframe()

    # next simulation would return this:
    # next_loop_simulation = vac_sublattice
    # next_loop_simulation.plot()de

    assert dataframe_after.equals(dataframe_before.append(
        Counter({'Mo_1': 397, 'Mo_0': 3}), ignore_index=True).fillna(0))
'''
