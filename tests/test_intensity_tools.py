import pytest

import numpy as np

import temul.api as tml
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
