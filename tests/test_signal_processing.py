import numpy as np
import pytest

from temul.dummy_data import (
    get_simple_cubic_signal,
    get_simple_cubic_sublattice_positions_on_vac,
)
import temul.signal_processing as sp


def test_get_xydata_from_list_of_intensities_histogram_shapes():
    intensities = np.linspace(0, 1, 100)

    xdata, ydata = sp.get_xydata_from_list_of_intensities(
        intensities, hist_bins=10)

    assert xdata.shape == (10,)
    assert ydata.shape == (10,)
    assert ydata.sum() == 100


def test_fit_1d_gaussian_to_data_peak_near_mu():
    xdata = np.linspace(8, 12, 1001)
    ydata = sp.fit_1D_gaussian_to_data(xdata, amp=10, mu=10, sigma=0.5)

    assert ydata.shape == xdata.shape
    assert xdata[np.argmax(ydata)] == pytest.approx(10, abs=1e-2)


def test_return_fitting_of_1d_gaussian_recovers_parameters():
    xdata = np.linspace(8, 12, 200)
    ydata = sp.fit_1D_gaussian_to_data(xdata, amp=10, mu=10, sigma=0.5)

    popt, pcov = sp.return_fitting_of_1D_gaussian(
        sp.fit_1D_gaussian_to_data, xdata, ydata, amp=9, mu=9.8, sigma=0.6)

    assert popt[0] == pytest.approx(10, rel=1e-2)
    assert popt[1] == pytest.approx(10, rel=1e-2)
    assert popt[2] == pytest.approx(0.5, rel=1e-2)
    assert pcov.shape == (3, 3)


def test_measure_image_errors_for_identical_images():
    image = get_simple_cubic_signal().data

    mse_number, ssm_number = sp.measure_image_errors(image, image.copy())

    assert mse_number == pytest.approx(0.0)
    assert ssm_number == pytest.approx(1.0)


def test_make_gaussian_center_and_symmetry():
    gaussian = sp.make_gaussian(15, 5)

    assert gaussian.shape == (15, 15)
    assert gaussian[7, 7] == pytest.approx(gaussian.max())
    assert np.allclose(gaussian, np.flipud(gaussian))
    assert np.allclose(gaussian, np.fliplr(gaussian))


def test_make_gaussian_pos_neg_returns_opposite_signals():
    positive, negative = sp.make_gaussian_pos_neg(
        size=11, fwhm_neg=3, fwhm_pos=5, neg_min=0.9)

    assert positive.data.shape == (11, 11)
    assert negative.data.shape == (11, 11)
    assert positive.data.max() > 0
    assert negative.data.min() < 0


def test_get_cell_image_returns_image_and_validates_inputs():
    sublattice = get_simple_cubic_sublattice_positions_on_vac()
    cell_image = sp.get_cell_image(
        sublattice.image,
        sublattice.x_position,
        sublattice.y_position,
        show_progressbar=False)

    assert cell_image.shape == sublattice.image.shape
    assert np.isfinite(cell_image).all()

    with pytest.raises(ValueError, match="at least 2 dimensions"):
        sp.get_cell_image(np.array([1, 2, 3]), [0], [0], show_progressbar=False)

    with pytest.raises(NotImplementedError, match="unimplemented method"):
        sp.get_cell_image(
            sublattice.image,
            sublattice.x_position,
            sublattice.y_position,
            method='invalid',
            show_progressbar=False)


def test_distance_vector_returns_euclidean_distance():
    assert sp.distance_vector(0, 0, 3, 4) == pytest.approx(5.0)


def test_mean_and_std_nearest_neighbour_distances_scaling():
    sublattice = get_simple_cubic_sublattice_positions_on_vac()

    mean_list, std_list = sp.mean_and_std_nearest_neighbour_distances(
        sublattice, nearest_neighbours=5)
    mean_scaled, std_scaled = sp.mean_and_std_nearest_neighbour_distances(
        sublattice, nearest_neighbours=5, sampling=0.5)

    assert len(mean_list) == len(std_list) == len(sublattice.atom_list)
    assert len(mean_scaled) == len(std_scaled) == len(sublattice.atom_list)
    assert np.allclose(mean_scaled, np.asarray(mean_list) * 0.5)
    assert np.allclose(std_scaled, np.asarray(std_list) * 0.5)
