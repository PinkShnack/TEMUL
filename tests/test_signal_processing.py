import numpy as np
import pytest
from hyperspy.signals import Signal2D

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


def test_get_scaled_middle_limit_intensity_list():
    middle, limits = sp.get_scaled_middle_limit_intensity_list(
        sublattice=None,
        middle_intensity_list=[0.5, 1.0],
        limit_intensity_list=[0.0, 0.75, 1.25],
        sublattice_scalar=2.0,
    )

    assert middle == [1.0, 2.0]
    assert limits == [0.0, 1.5, 2.5]


def test_get_fitting_tools_for_plotting_gaussians_and_validation():
    fitting_tools = sp.get_fitting_tools_for_plotting_gaussians(
        element_list=['Mo_1', 'Mo_2'],
        scaled_middle_intensity_list=[1.0, 2.0],
        scaled_limit_intensity_list=[0.0, 1.5, 3.0],
    )

    assert fitting_tools[0][0] == 'Mo_2'
    assert fitting_tools[1][0] == 'Mo_1'

    with pytest.raises(ValueError, match='length one greater'):
        sp.get_fitting_tools_for_plotting_gaussians(
            ['Mo_1'], [1.0], [0.0, 1.0, 2.0]
        )

    with pytest.raises(ValueError, match='same length as middle list'):
        sp.get_fitting_tools_for_plotting_gaussians(
            ['Mo_1', 'Mo_2'], [1.0], [0.0, 2.0]
        )


def test_mse_and_measure_image_errors_with_filename(tmp_path):
    import os

    image_a = np.zeros((8, 8), dtype=np.uint8)
    image_b = np.ones((8, 8), dtype=np.float64)

    assert sp.mse(image_a, image_b) == pytest.approx(1.0)

    cwd = tmp_path.cwd()
    try:
        os.chdir(tmp_path)
        mse_number, ssm_number = sp.measure_image_errors(
            image_a,
            image_b,
            filename='unit',
        )
    finally:
        os.chdir(cwd)

    assert mse_number == pytest.approx(1.0)
    assert ssm_number < 1.0
    assert (tmp_path / 'MSE_SSM_single_image_unit.png').exists()


def test_compare_two_image_and_create_filtered_image_selects_sigma(
        monkeypatch, handle_plots):
    image_to_filter = Signal2D(np.ones((5, 5), dtype=float))
    reference_image = Signal2D(np.ones((5, 5), dtype=float))
    calls = {'sigmas': []}

    monkeypatch.setattr(
        sp,
        'calibrate_intensity_distance_with_sublattice_roi',
        lambda **kwargs: None,
    )

    def fake_measure_image_errors(imageA, imageB, filename=None):
        sigma_estimate = float(np.mean(imageB) - 1.0)
        calls['sigmas'].append(round(sigma_estimate, 2))
        return abs(sigma_estimate - 1.0), 1.0 - abs(sigma_estimate)

    monkeypatch.setattr(sp, 'measure_image_errors', fake_measure_image_errors)
    monkeypatch.setattr(
        sp,
        'gaussian_filter',
        lambda data, sigma: data + sigma,
    )

    filtered, ideal_sigma = sp.compare_two_image_and_create_filtered_image(
        image_to_filter=image_to_filter,
        reference_image=reference_image,
        delta_image_filter=0.5,
        max_sigma=1.0,
        refine=False,
    )

    assert calls['sigmas'] == [0.0, 0.5, 1.0]
    assert ideal_sigma == pytest.approx(0.5)
    assert np.allclose(filtered.data, np.ones((5, 5)) + 0.5)


def test_toggle_atom_refine_position_automatically_and_invalid_range_type():
    sublattice = get_simple_cubic_sublattice_positions_on_vac()
    sublattice.find_nearest_neighbors()
    false_list = sp.toggle_atom_refine_position_automatically(
        sublattice,
        min_cut_off_percent=0.75,
        max_cut_off_percent=1.25,
        range_type='internal',
        method='mean',
        percent_to_nn=0.05,
    )

    assert len(false_list) > 0
    assert any(atom.refine_position is False for atom in sublattice.atom_list)

    with pytest.raises(TypeError, match='only options for range_type'):
        sp.toggle_atom_refine_position_automatically(
            sublattice,
            min_cut_off_percent=0.75,
            max_cut_off_percent=1.25,
            range_type='invalid',
            method='mean',
            percent_to_nn=0.05,
        )


def test_remove_image_intensity_in_data_slice_reduces_local_maximum():
    sublattice = get_simple_cubic_sublattice_positions_on_vac()
    sublattice.find_nearest_neighbors()
    atom = sublattice.atom_list[0]
    image = sublattice.image.copy()
    before = image.max()

    sp.remove_image_intensity_in_data_slice(atom, image, percent_to_nn=0.4)

    assert image.max() <= before
