
from atomap.testing_tools import MakeTestData
import numpy as np
import atomap.api as am
import hyperspy
from temul.model_refinement_class import model_refiner
from scipy.ndimage import gaussian_filter

# changes to atomap code:


def _make_simple_cubic_testdata(image_noise=False, amplitude=1):
    simple_cubic = MakeTestData(300, 300)
    x, y = np.mgrid[10:290:20j, 10:290:20j]
    if type(amplitude) == list:
        amplitude = np.random.randint(
            amplitude[0], amplitude[1], size=(len(x), len(x))).flatten()
    simple_cubic.add_atom_list(x.flatten(), y.flatten(), sigma_x=3, sigma_y=3,
                               amplitude=amplitude)
    if image_noise:
        simple_cubic.add_image_noise(mu=0, sigma=0.002)
    return simple_cubic


def get_simple_cubic_signal(image_noise=False, amplitude=1):
    """Generate a test image signal of a simple cubic atomic structure.

    Parameters
    ----------
    image_noise : default False
        If True, will add Gaussian noise to the image.

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
                                            amplitude=amplitude)
    return test_data.signal


def get_simple_cubic_sublattice(image_noise=False, amplitude=1):
    """Generate a test sublattice of a simple cubic atomic structure.

    Parameters
    ----------
    image_noise : default False
        If True, will add Gaussian noise to the image.

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
                                            amplitude=amplitude)
    return test_data.sublattice


def get_model_refiner_two_sublattices():

    atom_lattice = am.dummy_data.get_simple_atom_lattice_two_sublattices()
    sub1 = atom_lattice.sublattice_list[0]
    sub2 = atom_lattice.sublattice_list[1]
    for i in range(0, len(sub1.atom_list)):
        sub1.atom_list[i].elements = 'Ti_2'
    for i in range(0, len(sub2.atom_list)):
        sub2.atom_list[i].elements = 'Cl_1'
    sub1.atom_list[2].elements = 'Ti_1'
    sub1.atom_list[5].elements = 'Ti_1'
    sub2.atom_list[3].elements = 'Cl_2'
    sub2.atom_list[4].elements = 'Cl_2'

    sub1_element_list = ['Ti_0', 'Ti_1', 'Ti_2', 'Ti_3']
    sub2_element_list = ['Cl_0', 'Cl_1', 'Cl_2', 'Cl_3']

    refiner_dict = {sub1: sub1_element_list,
                    sub2: sub2_element_list}
    blurred_image = hyperspy._signals.signal2d.Signal2D(
        gaussian_filter(atom_lattice.image, 3))
    atom_lattice_refiner = model_refiner(refiner_dict, blurred_image)
    return atom_lattice_refiner
