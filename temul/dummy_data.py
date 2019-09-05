
from atomap.testing_tools import MakeTestData
import numpy as np


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
########
