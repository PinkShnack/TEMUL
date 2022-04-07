import pytest
import numpy as np
import matplotlib.pyplot as plt

import temul.topotem.polarisation as pol
import temul.dummy_data as dd


def test_find_polarisation_vectors_basic_01():
    """Check basic use of find_polarisation_vectors."""
    pos_A = [[1, 2], [3, 4]]
    pos_B = [[1, 1], [5, 2]]
    u, v = pol.find_polarisation_vectors(pos_A, pos_B)

    assert u == [0, 2]
    assert v == [-1, -2]
    vectors = np.asarray([u, v]).T
    assert np.allclose(vectors, np.array([[0, -1], [2, -2]]))


def test_corrected_vectors_via_average():
    pos_A = [[1, 2], [3, 4], [5, 8], [5, 2]]
    pos_B = [[1, 1], [5, 2], [3, 1], [6, 2]]
    u, v = pol.find_polarisation_vectors(pos_A, pos_B)
    u, v = np.asarray(u), np.asarray(v)
    u_av_expt, v_av_expt = u - np.mean(u), v - np.mean(v)

    u_av_corr, v_av_corr = pol.corrected_vectors_via_average(u, v)
    assert np.allclose(u_av_corr, u_av_expt)
    assert np.allclose(v_av_corr, v_av_expt)


def test_corrected_vectors_via_center_of_mass():
    pos_A = [[1, 2], [3, 4], [5, 8], [5, 2]]
    pos_B = [[1, 1], [5, 2], [3, 1], [6, 2]]
    u, v = pol.find_polarisation_vectors(pos_A, pos_B)
    u, v = np.asarray(u), np.asarray(v)
    r = (u ** 2 + v ** 2) ** 0.5
    u_com = np.sum(u * r) / np.sum(r)
    v_com = np.sum(v * r) / np.sum(r)
    u_com_expt, v_com_expt = u - u_com, v - v_com

    u_com_corr, v_com_corr = pol.corrected_vectors_via_center_of_mass(u, v)
    assert np.allclose(u_com_corr, u_com_expt)
    assert np.allclose(v_com_corr, v_com_expt)


def test_get_angles_from_uv_degree():
    u, v = np.array([1, 0, -1, 0]), np.array([0, 1, 0, -1])
    angles_expt = np.arctan2(v, u)
    angles_expt = angles_expt * 180 / np.pi
    assert np.allclose(angles_expt, [0, 90, 180, -90])

    angles = pol.get_angles_from_uv(u, v, degrees=True)
    assert np.allclose(angles, angles_expt)
    # the angles are kept between -180 and 180
    assert np.allclose(angles, [0, 90, 180, -90])


def test_get_angles_from_uv_radian():
    u, v = np.array([1, 0, -1, 0]), np.array([0, 1, 0, -1])
    angles_expt = np.arctan2(v, u)

    angles = pol.get_angles_from_uv(u, v, degrees=False)
    assert np.allclose(angles, angles_expt)


@pytest.mark.parametrize(
    "plot_style, vector_rep",
    [
        ("vector", "magnitude"), ("vector", "angle"),
        ("colormap", "magnitude"), ("colormap", "angle"),
        ("contour", "magnitude"), ("contour", "angle"),
        ("colorwheel", "angle"),  # magnitude not allowed for colorwheel
        ("polar_colorwheel", "magnitude"), ("polar_colorwheel", "angle"),
    ]
)
def test_plot_polarisation_vectors_plot_style_vector_rep(
        plot_style, vector_rep, get_dummy_xyuv, handle_plots):
    """Check the available plot_style."""
    sublatticeA, sublatticeB, x, y, u, v = get_dummy_xyuv
    _ = pol.plot_polarisation_vectors(
        x, y, u, v, image=sublatticeA.image, save=None,
        plot_style=plot_style, vector_rep=vector_rep)


@pytest.mark.parametrize(
    "overlay, unit_vector",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]
)
def test_plot_polarisation_vectors_overlay_unit_vector(
        overlay, unit_vector, get_dummy_xyuv, handle_plots):
    """Check the available overlay, unit_vector."""
    sublatticeA, sublatticeB, x, y, u, v = get_dummy_xyuv
    _ = pol.plot_polarisation_vectors(
        x, y, u, v, image=sublatticeA.image, save=None,
        overlay=overlay, unit_vector=unit_vector)


@pytest.mark.parametrize(
    "sampling, units",
    [
        (None, 'pix'),
        (0.05, 'um'),
        (0.0034, 'nm'),
    ]
)
def test_plot_polarisation_vectors_sampling_units(
        sampling, units, get_dummy_xyuv, handle_plots):
    """Check the available sampling, units."""
    sublatticeA, sublatticeB, x, y, u, v = get_dummy_xyuv
    _ = pol.plot_polarisation_vectors(
        x, y, u, v, image=sublatticeA.image, save=None,
        sampling=sampling, units=units)


@pytest.mark.parametrize(
    "vector_rep, degrees, angle_offset",
    [
        ('magnitude', False, None),
        ('angle', False, None),
        ('angle', True, None),
        ('angle', True, 10),  # degrees
        ('angle', False, 0.02),  # radians
    ]
)
def test_plot_polarisation_vectors_degrees_angle_offset(
        vector_rep, degrees, angle_offset, get_dummy_xyuv, handle_plots):
    """Check the available sampling, units."""
    sublatticeA, sublatticeB, x, y, u, v = get_dummy_xyuv
    _ = pol.plot_polarisation_vectors(
        x, y, u, v, image=sublatticeA.image, save=None,
        vector_rep=vector_rep, degrees=degrees, angle_offset=angle_offset)


@pytest.mark.parametrize(
    "vector_rep, cbar_vmin, cbar_vmax",
    [
        ("magnitude", None, None),
        ("magnitude", 2.0, 4.0),
        ("magnitude", None, 3.0),
        ("magnitude", 3.0, None),
    ]
)
def test_plot_polarisation_vectors_plot_style_cbar_limits(
        vector_rep, cbar_vmin, cbar_vmax, get_dummy_xyuv, handle_plots):
    """Check the available plot_style."""
    sublatticeA, sublatticeB, x, y, u, v = get_dummy_xyuv
    _ = pol.plot_polarisation_vectors(
        x, y, u, v, image=sublatticeA.image, save=None,
        plot_style='colormap', vector_rep=vector_rep,
        cbar_vmin=cbar_vmin, cbar_vmax=cbar_vmax)
    plt.show()


def test_atom_deviation_from_straight_line_fit_01_basic():
    """Simple check to make sure outputted data types are correct."""
    sublattice = dd.get_polarised_single_sublattice()
    sublattice.construct_zone_axes(atom_plane_tolerance=1)

    x, y, u, v = pol.atom_deviation_from_straight_line_fit(
        sublattice, axis_number=0, n=5, second_fit_rigid=True, plot=False,
        return_individual_atom_planes=False)

    assert isinstance(x, list)
    assert isinstance(x[0], float)
    assert all([len(i) == 256 for i in [x, y, u, v]])

    assert all([isinstance(i, list) for i in [x, y, u, v]])
    assert all([isinstance(i[0], float) for i in [x, y, u, v]])


def test_atom_deviation_from_straight_line_fit_02_basic():
    """Simple check to make sure outputted data types are correct for
    when ``return_individual_atom_planes=True``."""
    sublattice = dd.get_polarised_single_sublattice()
    sublattice.construct_zone_axes(atom_plane_tolerance=1)
    return_individual_atom_planes = True

    x, y, u, v = pol.atom_deviation_from_straight_line_fit(
        sublattice, axis_number=0, n=5, second_fit_rigid=True, plot=False,
        return_individual_atom_planes=return_individual_atom_planes)

    assert isinstance(x, list)
    assert isinstance(x[0], list)  # it is now a list of lists!
    assert len(x) == 16  # 16 atomic columns

    # look in each sublist
    x_first_atomic_column = x[0]
    assert isinstance(x_first_atomic_column, list)
    assert isinstance(x_first_atomic_column[0], float)
    # just so happens to be 16x16 in our test example!
    assert len(x_first_atomic_column) == 16  # 16 atoms in this atomic column
    assert isinstance(x_first_atomic_column[0], float)
    assert x_first_atomic_column[0] == 5.0

    # look at each sublist
    assert all([len(i) == 16 for i in [x, y, u, v]])
    assert all([isinstance(i, list) for i in [x, y, u, v]])
    assert all([isinstance(i[0], list) for i in [x, y, u, v]])


def test_atom_deviation_from_straight_line_fit_plot_rumpling(
        handle_plots):
    """Simple check to make sure outputted data types are correct for
    when ``return_individual_atom_planes=True``."""
    sublattice = dd.get_polarised_single_sublattice()
    sublattice.construct_zone_axes(atom_plane_tolerance=1)
    return_individual_atom_planes = True

    x, y, u, v = pol.atom_deviation_from_straight_line_fit(
        sublattice, axis_number=0, n=5, second_fit_rigid=True, plot=False,
        return_individual_atom_planes=return_individual_atom_planes)

    # look at first atomic plane rumpling
    plt.figure()
    plt.plot(range(len(v[0])), v[0], 'ro')
    plt.title("First atomic plane v (vertical) rumpling.")
    plt.xlabel("Atomic plane #")
    plt.ylabel("Vertical deviation (rumpling) (a.u.)")
    plt.show()
