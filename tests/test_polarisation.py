
import numpy as np

from temul import polarisation as tmlp


def test_find_polarisation_vectors():
    pos_A = [[1,2], [3,4]]
    pos_B = [[1,1], [5,2]]
    u, v = tmlp.find_polarisation_vectors(pos_A, pos_B)
    assert u == [0, 2]
    assert v == [-1, -2]

    vectors = np.asarray([u, v]).T
    assert np.allclose(vectors, np.array([[0, -1], [2, -2]]))


def test_corrected_vectors_via_average():
    pos_A = [[1,2], [3,4], [5,8], [5,2]]
    pos_B = [[1,1], [5,2], [3,1], [6,2]]
    u, v = tmlp.find_polarisation_vectors(pos_A, pos_B)
    u, v = np.asarray(u), np.asarray(v)
    u_av_expt, v_av_expt = u - np.mean(u), v - np.mean(v)

    u_av_corr, v_av_corr = tmlp.corrected_vectors_via_average(u, v)
    assert np.allclose(u_av_corr, u_av_expt)
    assert np.allclose(v_av_corr, v_av_expt)


def test_corrected_vectors_via_center_of_mass():
    pos_A = [[1,2], [3,4], [5,8], [5,2]]
    pos_B = [[1,1], [5,2], [3,1], [6,2]]
    u, v = tmlp.find_polarisation_vectors(pos_A, pos_B)
    u, v = np.asarray(u), np.asarray(v)
    r = (u**2 + v**2) ** 0.5
    u_com = np.sum(u*r) / np.sum(r)
    v_com = np.sum(v*r) / np.sum(r)
    u_com_expt, v_com_expt = u - u_com, v - v_com

    u_com_corr, v_com_corr = tmlp.corrected_vectors_via_center_of_mass(u, v)
    assert np.allclose(u_com_corr, u_com_expt)
    assert np.allclose(v_com_corr, v_com_expt)


def test_get_angles_from_uv_degree():
    u, v = np.array([1, 0, -1, 0]), np.array([0, 1, 0, -1])
    angles_expt = np.arctan2(v, u)
    angles_expt = angles_expt * 180 / np.pi
    assert np.allclose(angles_expt, [0, 90, 180, -90])


    angles = tmlp.get_angles_from_uv(u, v, degrees=True)
    assert np.allclose(angles, angles_expt)
    # the angles are kept between -180 and 180
    assert np.allclose(angles, [0, 90, 180, -90])


def test_get_angles_from_uv_radian():
    u, v = np.array([1, 0, -1, 0]), np.array([0, 1, 0, -1])
    angles_expt = np.arctan2(v, u)

    angles = tmlp.get_angles_from_uv(u, v, degrees=False)
    assert np.allclose(angles, angles_expt)



