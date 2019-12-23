
from atomap.testing_tools import MakeTestData
import numpy as np
import atomap.api as am
import hyperspy
from temul.model_refinement_class import model_refiner
from temul.model_creation import auto_generate_sublattice_element_list
from temul.element_tools import split_and_sort_element
from scipy.ndimage import gaussian_filter

# changes to atomap code:


def _make_simple_cubic_testdata(image_noise=False, amplitude=1,
                                with_vacancies=False):
    """
    Parameters
    ----------
    image_noise : default False
        If True, will add Gaussian noise to the image.
    amplitude : int, list of ints, default 1
        If amplitude is set to an int, that int will be applied to all atoms in
        the sublattice.
        If amplitude is set to a list, the atoms will be a distribution set by
        np.random.randint between the min and max int.
    """
    simple_cubic = MakeTestData(300, 300)
    x, y = np.mgrid[10:290:20j, 10:290:20j]
    if type(amplitude) == list:
        amplitude = np.random.randint(
            np.min(amplitude), np.max(amplitude),
            size=(len(x), len(y))).flatten()

    x, y = x.flatten().tolist(), y.flatten().tolist()
    if with_vacancies:
        for i in [15, 34, 54, 71, 82, 123, 167, 193, 201, 222, 253, 264]:
            x.pop(i)
            y.pop(i)
    simple_cubic.add_atom_list(x, y, sigma_x=3, sigma_y=3,
                               amplitude=amplitude)
    if image_noise:
        simple_cubic.add_image_noise(mu=0, sigma=0.002)
    return simple_cubic


def get_simple_cubic_signal(image_noise=False, amplitude=1,
                            with_vacancies=False):
    """Generate a test image signal of a simple cubic atomic structure.

    Parameters
    ----------
    image_noise : default False
        If True, will add Gaussian noise to the image.
    amplitude : int, list of ints, default 1
        If amplitude is set to an int, that int will be applied to all atoms in
        the sublattice.
        If amplitude is set to a list, the atoms will be a distribution set by
        np.random.randint between the min and max int.

    Returns
    -------
    signal : HyperSpy 2D

    Examples
    --------
    >>> import atomap.api as am
    >>> s = am.dummy_data.get_simple_cubic_signal()
    >>> s.plot()

    """
    test_data = _make_simple_cubic_testdata(image_noise=image_noise,
                                            amplitude=amplitude,
                                            with_vacancies=with_vacancies)
    return test_data.signal


def get_simple_cubic_sublattice(image_noise=False, amplitude=1,
                                with_vacancies=False):
    """Generate a test sublattice of a simple cubic atomic structure.

    Parameters
    ----------
    image_noise : default False
        If True, will add Gaussian noise to the image.
    amplitude : int, list of ints, default 1
        If amplitude is set to an int, that int will be applied to all atoms in
        the sublattice.
        If amplitude is set to a list, the atoms will be a distribution set by
        np.random.randint between the min and max int.

    Returns
    -------
    sublattice : Atomap Sublattice

    Examples
    --------
    >>> import atomap.api as am
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.plot()

    # If the amplitude list is:
    sublattice = tml.dummy_data.get_simple_cubic_sublattice(
        amplitude=[2, 3])
    # then only amplitudes of 2 will be used, see numpy.random.randint

    """
    test_data = _make_simple_cubic_testdata(image_noise=image_noise,
                                            amplitude=amplitude,
                                            with_vacancies=with_vacancies)
    return test_data.sublattice


def get_model_refiner_two_sublattices():

    atom_lattice = am.dummy_data.get_simple_atom_lattice_two_sublattices()
    sub1 = atom_lattice.sublattice_list[0]
    sub2 = atom_lattice.sublattice_list[1]
    for i in range(0, len(sub1.atom_list)):
        if i//2 == 0:
            sub1.atom_list[i].elements = 'Ti_2'
        else:
            sub1.atom_list[i].elements = 'Ti_1'
    for i in range(0, len(sub2.atom_list)):
        if i//2 == 0:
            sub2.atom_list[i].elements = 'Cl_2'
        else:
            sub1.atom_list[i].elements = 'Cl_3'

    sub1.atom_list[2].elements = 'Ti_3'
    sub1.atom_list[5].elements = 'Ti_3'
    sub2.atom_list[3].elements = 'Cl_1'
    sub2.atom_list[4].elements = 'Cl_1'
    sub1_element_list = ['Ti_0', 'Ti_1', 'Ti_2', 'Ti_3']
    sub2_element_list = ['Cl_0', 'Cl_1', 'Cl_2', 'Cl_3']

    refiner_dict = {sub1: sub1_element_list,
                    sub2: sub2_element_list}
    blurred_image = hyperspy._signals.signal2d.Signal2D(
        gaussian_filter(atom_lattice.image, 3))
    atom_lattice_refiner = model_refiner(refiner_dict, blurred_image)
    return atom_lattice_refiner


def get_model_refiner_two_sublattices_refined(n=4):

    refined_model = get_model_refiner_two_sublattices()
    refined_model.repeating_intensity_refinement(
        n=n,
        ignore_element_count_comparison=True)

    return refined_model


def get_model_refiner_one_sublattice_varying_amp(
        image_noise=False, amplitude=[0, 5],
        test_element='Ti_2'):

    sub1 = get_simple_cubic_sublattice(image_noise=image_noise,
                                       amplitude=amplitude)
    for i in range(0, len(sub1.atom_list)):
        sub1.atom_list[i].elements = test_element

    test_element_info = split_and_sort_element(test_element)
    sub1_element_list = auto_generate_sublattice_element_list(
        material_type='single_element_column',
        elements=test_element_info[0][1],
        max_number_atoms_z=np.max(amplitude))

    refiner_dict = {sub1: sub1_element_list}
    comparison_image = get_simple_cubic_signal(
        image_noise=image_noise,
        amplitude=test_element_info[0][2])
    refiner = model_refiner(refiner_dict, comparison_image)
    return refiner


def get_model_refiner_one_sublattice_3_vacancies(
        image_noise=True, test_element='Ti_2'):

    # make one with more vacancies maybe
    sub1 = am.dummy_data.get_simple_cubic_with_vacancies_sublattice(
        image_noise=image_noise)
    for i in range(0, len(sub1.atom_list)):
        sub1.atom_list[i].elements = test_element

    test_element_info = split_and_sort_element(test_element)
    sub1_element_list = auto_generate_sublattice_element_list(
        material_type='single_element_column',
        elements=test_element_info[0][1],
        max_number_atoms_z=test_element_info[0][2]+3)

    refiner_dict = {sub1: sub1_element_list}
    comparison_image = am.dummy_data.get_simple_cubic_signal(
        image_noise=image_noise)
    refiner = model_refiner(refiner_dict, comparison_image)
    return refiner


def get_model_refiner_one_sublattice_12_vacancies(
        image_noise=True, test_element='Ti_2'):

    # make one with more vacancies maybe
    sublattice = get_simple_cubic_sublattice(
        image_noise=image_noise,
        with_vacancies=False)
    sub1_atom_positions = np.array(sublattice.atom_positions).T
    sub1_image = get_simple_cubic_signal(
        image_noise=image_noise,
        with_vacancies=True)
    sub1 = am.Sublattice(sub1_atom_positions, sub1_image)
    for i in range(0, len(sub1.atom_list)):
        sub1.atom_list[i].elements = test_element

    test_element_info = split_and_sort_element(test_element)
    sub1_element_list = auto_generate_sublattice_element_list(
        material_type='single_element_column',
        elements=test_element_info[0][1],
        max_number_atoms_z=test_element_info[0][2]+3)

    refiner_dict = {sub1: sub1_element_list}
    comparison_image = sublattice.signal
    refiner = model_refiner(refiner_dict, comparison_image)
    return refiner


def get_model_refiner_with_12_vacancies_refined(
        image_noise=True, test_element='Ti_2', filename=None):

    refined_model = get_model_refiner_one_sublattice_12_vacancies(
        image_noise=image_noise, test_element=test_element)
    refined_model.image_difference_intensity_model_refiner(filename=filename)
    refined_model.image_difference_intensity_model_refiner(filename=filename)

    # now that the vacancies are correctly labelled as Ti_0, we should use
    # a comparison image that actually represents what might be simulated.
    refined_model.comparison_image = refined_model.sublattice_list[0].signal
    refined_model.image_difference_intensity_model_refiner(filename=filename)

    return refined_model


def get_model_refiner_with_3_vacancies_refined(
        image_noise=True, test_element='Ti_2', filename=None):
    '''
    >>> refiner = get_model_refiner_with_3_vacancies_refined()
    >>> history = refiner.get_element_count_as_dataframe()
    '''
    refiner = get_model_refiner_one_sublattice_3_vacancies(
        image_noise=image_noise, test_element=test_element)

    refiner.image_difference_position_model_refiner(
        pixel_threshold=10, filename=filename)

    refiner.image_difference_intensity_model_refiner(filename=filename)
    refiner.image_difference_intensity_model_refiner(filename=filename)

    return refiner
