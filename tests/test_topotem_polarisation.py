import pytest
import numpy as np

import temul.topotem.polarisation as pol


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
