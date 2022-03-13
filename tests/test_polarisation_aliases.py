import temul.topotem.polarisation as pol


def test_find_polarisation_vectors(get_dummy_xyuv):
    """find_polarisation_vectors check"""
    sublatticeA = get_dummy_xyuv[0]
    atom_positions_A = sublatticeA.atom_positions
    atom_positions_B = sublatticeA.atom_positions + 2
    assert atom_positions_B[0][0] == atom_positions_A[0][0] + 2
    assert atom_positions_B[-1][0] == atom_positions_A[-1][0] + 2

    u, v = pol.find_polarisation_vectors(
        atom_positions_A, atom_positions_B, save=None)

    assert [isinstance(i, list) for i in [u, v]]
    assert len(u) == len(v)


def test_find_polarization_vectors(get_dummy_xyuv):
    """find_polarization_vectors check"""
    sublatticeA = get_dummy_xyuv[0]
    atom_positions_A = sublatticeA.atom_positions
    atom_positions_B = sublatticeA.atom_positions + 2
    assert atom_positions_B[0][0] == atom_positions_A[0][0] + 2
    assert atom_positions_B[-1][0] == atom_positions_A[-1][0] + 2

    u, v = pol.find_polarization_vectors(
        atom_positions_A, atom_positions_B, save=None)

    assert [isinstance(i, list) for i in [u, v]]
    assert len(u) == len(v)


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


def test_get_average_polarisation_in_regions(get_dummy_xyuv):
    """get_average_polarisation_in_regions check"""
    sublatticeA, sublatticeB, x, y, u, v = get_dummy_xyuv
    x_new, y_new, u_new, v_new = pol.get_average_polarisation_in_regions(
        x, y, u, v, image=sublatticeA.image, divide_into=8)

    assert [isinstance(i, list) for i in [x_new, y_new, u_new, v_new]]
    assert len(x_new) == len(y_new) == len(u_new) == len(v_new)


def test_get_average_polarization_in_regions(get_dummy_xyuv):
    """get_average_polarization_in_regions check"""
    sublatticeA, sublatticeB, x, y, u, v = get_dummy_xyuv
    x_new, y_new, u_new, v_new = pol.get_average_polarization_in_regions(
        x, y, u, v, image=sublatticeA.image, divide_into=8)

    assert [isinstance(i, list) for i in [x_new, y_new, u_new, v_new]]
    assert len(x_new) == len(y_new) == len(u_new) == len(v_new)


def test_get_average_polarisation_in_regions_square(get_dummy_xyuv):
    """get_average_polarisation_in_regions_square check"""
    sublatticeA, sublatticeB, x, y, u, v = get_dummy_xyuv
    x_new, y_new, u_new, v_new = pol.get_average_polarisation_in_regions_square(  # noqa: E501
        x, y, u, v, image=sublatticeA.image, divide_into=8)

    assert [isinstance(i, list) for i in [x_new, y_new, u_new, v_new]]
    assert len(x_new) == len(y_new) == len(u_new) == len(v_new)


def test_get_average_polarization_in_regions_square(get_dummy_xyuv):
    """get_average_polarization_in_regions_square check"""
    sublatticeA, sublatticeB, x, y, u, v = get_dummy_xyuv
    x_new, y_new, u_new, v_new = pol.get_average_polarization_in_regions_square(  # noqa: E501
        x, y, u, v, image=sublatticeA.image, divide_into=8)

    assert [isinstance(i, list) for i in [x_new, y_new, u_new, v_new]]
    assert len(x_new) == len(y_new) == len(u_new) == len(v_new)
