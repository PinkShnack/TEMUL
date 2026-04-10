import pytest

import numpy as np

import temul.api as tml
import temul.intensity_tools as it
from temul.dummy_data import get_simple_cubic_sublattice

sub1 = get_simple_cubic_sublattice()


def test_get_sublattice_intensity_default():
    """Simple check of the default params for
    ``temul.intensity_tools.get_sublattice_intensity``.
    """
    intensities = tml.get_sublattice_intensity(sub1)

    assert isinstance(intensities, np.ndarray)
    assert len(sub1.atom_list) == intensities.shape[0]
    assert isinstance(intensities[0], float)
    assert np.allclose(intensities[0], 0.0176, rtol=1e-02)


@pytest.mark.parametrize(
    'intensity_type, remove_background_method, background_sub',
    [('max', None, None),
     ('min', None, None),
     ('max', 'average', sub1),
     ])
def test_get_sublattice_intensity_params_01(
        intensity_type, remove_background_method, background_sub):
    """Check of some params for
    ``temul.intensity_tools.get_sublattice_intensity``.
    """
    intensities = tml.get_sublattice_intensity(
        sub1, intensity_type, remove_background_method, background_sub)

    assert isinstance(intensities, np.ndarray)
    assert len(sub1.atom_list) == intensities.shape[0]
    assert isinstance(intensities[0], float)


@pytest.mark.xfail
@pytest.mark.parametrize(
    'intensity_type, num_points',
    [('max', 3),
     ('max', 5),
     ('max', 2),
     ])
def test_get_sublattice_intensity_rmbck_local(
        intensity_type, num_points):
    """Check of some params for
    ``temul.intensity_tools.get_sublattice_intensity``.
    """
    intensities = tml.get_sublattice_intensity(
        sub1, intensity_type=intensity_type,
        remove_background_method='local',
        background_sub=sub1, num_points=num_points)

    assert isinstance(intensities, np.ndarray)
    assert len(sub1.atom_list) == intensities.shape[0]
    assert isinstance(intensities[0], float)


def test_get_sublattice_intensity_params_03():
    """Check of some params for
    ``temul.intensity_tools.get_sublattice_intensity``.
    """
    with pytest.raises(ValueError):
        _ = tml.get_sublattice_intensity(
            sub1, intensity_type='max',
            remove_background_method='local',
            background_sub=sub1, num_points=0)


def test_get_sublattice_intensity_rmbkgnd_type():
    """Check of some params for
    ``temul.intensity_tools.get_sublattice_intensity``.
    """
    with pytest.raises(ValueError):
        _ = tml.get_sublattice_intensity(
            sub1, intensity_type='min',
            remove_background_method='local')


def test_get_sublattice_intensity_all_returns_two_columns():
    intensities = tml.get_sublattice_intensity(sub1, intensity_type='all')

    assert intensities.shape == (len(sub1.atom_list), 2)
    assert np.isfinite(intensities).all()


def test_remove_average_background_matches_manual_difference():
    sub1.find_nearest_neighbors()
    raw = tml.get_sublattice_intensity(sub1, intensity_type='max')
    background = tml.get_sublattice_intensity(sub1, intensity_type='min')

    corrected = it.remove_average_background(
        sublattice=sub1,
        background_sub=sub1,
        intensity_type='max')

    assert np.allclose(corrected, raw - np.mean(background))


def test_remove_average_background_mean_matches_manual_difference():
    sub1.find_nearest_neighbors()
    sub1.get_atom_column_amplitude_mean_intensity(percent_to_nn=0.4)
    raw_mean = np.array(sub1.atom_amplitude_mean_intensity)
    background = tml.get_sublattice_intensity(sub1, intensity_type='min')

    corrected = it.remove_average_background(
        sublattice=sub1,
        background_sub=sub1,
        intensity_type='mean',
    )

    assert np.allclose(corrected, raw_mean - np.mean(background))


def test_remove_average_background_rejects_total_intensity():
    with pytest.raises(ValueError, match="doesn't work with total intensity"):
        it.remove_average_background(
            sublattice=sub1,
            background_sub=sub1,
            intensity_type='total',
        )


def test_get_sublattice_intensity_rejects_unknown_type():
    with pytest.raises(ValueError, match="choose an intensity_type"):
        tml.get_sublattice_intensity(sub1, intensity_type='median')


def test_get_pixel_count_from_image_slice_returns_positive_count():
    sub1.find_nearest_neighbors()
    atom0 = sub1.atom_list[0]

    count = it.get_pixel_count_from_image_slice(atom0, sub1.image)

    assert isinstance(count, int)
    assert count > 0
