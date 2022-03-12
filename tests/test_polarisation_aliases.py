

import temul.topotem.polarisation as pol


def test_find_polarisation_vectors(get_dummy_xyuv):
    """find_polarisation_vectors check"""
    pass


def test_find_polarization_vectors(get_dummy_xyuv):
    """find_polarisation_vectors check"""
    pass


def test_plot_polarisation_vectors(get_dummy_xyuv, handle_plots):
    """plot_polarisation_vectors check"""
    sublatticeA, sublatticeB, x, y, u, v = get_dummy_xyuv
    _ = pol.plot_polarisation_vectors(
        x, y, u, v, image=sublatticeA.image, save=None)


def test_plot_polarization_vectors(get_dummy_xyuv, handle_plots):
    """plot_polarization_vectors check"""
    sublatticeA, sublatticeB, x, y, u, v = get_dummy_xyuv
    _ = pol.plot_polarization_vectors(
        x, y, u, v, image=sublatticeA.image, save=None)


def test_get_average_polarisation_in_regions():
    """find_polarisation_vectors check"""
    pass


def test_get_average_polarization_in_regions():
    """find_polarisation_vectors check"""
    pass


def test_get_average_polarisation_in_regions_square():
    """find_polarisation_vectors check"""
    pass


def test_get_average_polarization_in_regions_square():
    """find_polarisation_vectors check"""
    pass
