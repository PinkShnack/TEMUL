
from atomap.testing_tools import MakeTestData
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc


# adapted from Atomap:

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


def _make_distorted_cubic_testdata_adjustable(y_offset=2, image_noise=False):
    test_data = MakeTestData(240, 240)
    x, y = np.mgrid[30:212:40, 30:222:20]
    x, y = x.flatten(), y.flatten()
    test_data.add_atom_list(x, y)
    x, y = np.mgrid[50:212:40, 30.0:111:20]
    x, y = x.flatten(), y.flatten()
    test_data.add_atom_list(x, y)
    x, y = np.mgrid[50:212:40, 130 + y_offset:222:20]
    x, y = x.flatten(), y.flatten()
    test_data.add_atom_list(x, y)
    if image_noise:
        test_data.add_image_noise(mu=0, sigma=0.002)
    return test_data


def get_distorted_cubic_signal_adjustable(y_offset=2, image_noise=False):
    """Generate a test image signal of a distorted cubic atomic structure.

    Parameters
    ----------
    image_noise : default False
        If True, will add Gaussian noise to the image.

    Returns
    -------
    signal : HyperSpy 2D

    Examples
    --------
    >>> from temul.dummy_data import get_distorted_cubic_signal_adjustable
    >>> s = get_distorted_cubic_signal_adjustable(y_offset=2)
    >>> s.plot()

    """
    test_data = _make_distorted_cubic_testdata_adjustable(
        y_offset=y_offset, image_noise=image_noise)
    return test_data.signal


def get_distorted_cubic_sublattice_adjustable(y_offset=2, image_noise=False):
    """Generate a test sublattice of a distorted cubic atomic structure.

    Parameters
    ----------
    image_noise : default False
        If True, will add Gaussian noise to the image.

    Returns
    -------
    sublattice : Atomap Sublattice

    Examples
    --------
    >>> from temul.dummy_data import get_distorted_cubic_sublattice_adjustable
    >>> sublattice = get_distorted_cubic_sublattice_adjustable(y_offset=2)
    >>> sublattice.plot()

    """
    test_data = _make_distorted_cubic_testdata_adjustable(
        y_offset=y_offset, image_noise=image_noise)
    return test_data.sublattice


def polarisation_colorwheel_test_dataset(cmap=cc.cm.colorwheel, plot_XY=True,
                                         degrees=False, normalise=False):
    """
    Check how the arrows will be plotted on a colorwheel.
    Note that for STEM images, the y axis is reversed. This is taken into
    account in the plot_polarisation_vectors function.

    Parameters
    ----------
    image_noise : default False
        If True, will add Gaussian noise to the image.

    Returns
    -------
    sublattice : Atomap Sublattice

    Examples
    --------
    >>> from temul.dummy_data import polarisation_colorwheel_test_dataset

    Use a cyclic colormap for better understanding of vectors

    >>> polarisation_colorwheel_test_dataset(cmap='hsv')

    For more cyclic colormap options (and better colormaps), use colorcet

    >>> import colorcet as cc
    >>> polarisation_colorwheel_test_dataset(cmap=cc.cm.colorwheel)

    Plot with degrees rather than the default radians

    >>> polarisation_colorwheel_test_dataset(degrees=True)

    To just plot the top and bottom arrows as "diverging" the middle of the
    colormap should be the same as the edges, such as colorcet's CET_C4.

    >>> polarisation_colorwheel_test_dataset(cmap=cc.cm.CET_C4)

    To just plot the left and right arrows as "diverging" the halfway points of
    the colormap should be the same as the edges, such as colorcet's CET_C4s.

    >>> polarisation_colorwheel_test_dataset(cmap=cc.cm.CET_C4s)

    """

    x, y = np.meshgrid(np.arange(0, 2 * np.pi, .2),
                       np.arange(0, 2 * np.pi, .2))
    u = np.cos(x)
    v = np.sin(y)

    if normalise:
        u_norm = u / np.sqrt(u ** 2.0 + v ** 2.0)
        v_norm = v / np.sqrt(u ** 2.0 + v ** 2.0)
        u = u_norm
        v = v_norm

    angles = np.arctan2(v, u)
    if degrees:
        angles = angles * (180 / np.pi)

    _, ax = plt.subplots()
    Q = ax.quiver(x, y, u, v, angles, cmap=cmap)
    if plot_XY is True:
        plt.scatter(x, y, color='k', alpha=0.7, s=2)
    plt.colorbar(Q)
