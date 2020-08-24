
from atomap.testing_tools import MakeTestData
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
from temul.external.atomap_devel_012.sublattice import Sublattice


# some of the below have been adapted from Atomap:

def _make_simple_cubic_testdata(image_noise=False, amplitude=1,
                                with_vacancies=False):
    """
    Parameters
    ----------
    image_noise : Bool, default False
        If set to True, will add Gaussian noise to the image.
    amplitude : int, list of ints, default 1
        If amplitude is set to an int, that int will be applied to all atoms in
        the sublattice.
        If amplitude is set to a list, the atoms will be a distribution set by
        np.random.randint between the min and max int.
    with_vacancies : Bool, default False
        If set to True, the returned signal or sublattice will have some
        vacancies.
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
    image_noise : Bool, default False
        If set to True, will add Gaussian noise to the image.
    amplitude : int, list of ints, default 1
        If amplitude is set to an int, that int will be applied to all atoms in
        the sublattice.
        If amplitude is set to a list, the atoms will be a distribution set by
        np.random.randint between the min and max int.
    with_vacancies : Bool, default False
        If set to True, the returned signal or sublattice will have some
        vacancies.

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
    image_noise : Bool, default False
        If set to True, will add Gaussian noise to the image.
    amplitude : int, list of ints, default 1
        If amplitude is set to an int, that int will be applied to all atoms in
        the sublattice.
        If amplitude is set to a list, the atoms will be a distribution set by
        np.random.randint between the min and max int.
    with_vacancies : Bool, default False
        If set to True, the returned signal or sublattice will have some
        vacancies.

    Returns
    -------
    Atomap Sublattice object

    Examples
    --------
    >>> from temul.dummy_data import get_simple_cubic_sublattice
    >>> sublattice = get_simple_cubic_sublattice()
    >>> sublattice.plot()

    If you want different atom amplitudes, use `amplitude`

    >>> sublattice = get_simple_cubic_sublattice(
    ...     amplitude=[1, 5])

    Do not set `amplitude` to two consecutive numbers, as only amplitudes of
    the lower number (2 below) will be set, see numpy.random.randint for info.

    >>> sublattice = get_simple_cubic_sublattice(
    ...     amplitude=[2,3])

    """

    test_data = _make_simple_cubic_testdata(image_noise=image_noise,
                                            amplitude=amplitude,
                                            with_vacancies=with_vacancies)
    return test_data.sublattice


def get_simple_cubic_sublattice_positions_on_vac(image_noise=False):
    '''
    Create a simple cubic structure similar to `get_simple_cubic_sublattice`
    above but the atom positions are also overlaid on the vacancy positions.
    '''

    temp_sub = _make_simple_cubic_testdata(image_noise=image_noise,
                                           with_vacancies=False).sublattice
    temp_pos = np.asarray([temp_sub.x_position, temp_sub.y_position]).T
    image = _make_simple_cubic_testdata(image_noise=image_noise,
                                        with_vacancies=True).signal
    sublattice = Sublattice(temp_pos, image.data)

    return sublattice


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


def get_distorted_cubic_signal_adjustable(image_noise=False, y_offset=2):
    """Generate a test image signal of a distorted cubic atomic structure.

    Parameters
    ----------
    image_noise : default False
        If True, will add Gaussian noise to the image.
    y_offset : int, default 2
        The magnitude of distortion of the cubic signal.

    Returns
    -------
    HyperSpy Signal2D

    Examples
    --------
    >>> from temul.dummy_data import get_distorted_cubic_signal_adjustable
    >>> s = get_distorted_cubic_signal_adjustable(y_offset=2)
    >>> s.plot()

    """
    test_data = _make_distorted_cubic_testdata_adjustable(
        y_offset=y_offset, image_noise=image_noise)
    return test_data.signal


def get_distorted_cubic_sublattice_adjustable(image_noise=False, y_offset=2):
    """Generate a test sublattice of a distorted cubic atomic structure.

    Parameters
    ----------
    image_noise : default False
        If True, will add Gaussian noise to the image.
    y_offset : int, default 2
        The magnitude of distortion of the cubic signal.

    Returns
    -------
    Atomap Sublattice object

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
    account in the plot_polarisation_vectors function, but not here.

    Parameters
    ----------
    image_noise : default False
        If True, will add Gaussian noise to the image.

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
