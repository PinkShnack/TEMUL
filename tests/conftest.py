
import pytest
import matplotlib.pyplot as plt

from temul.dummy_data import get_polarisation_dummy_dataset


@pytest.fixture
def get_dummy_xyuv():
    """Example dummy data."""
    atom_lattice = get_polarisation_dummy_dataset()
    sublatticeA = atom_lattice.sublattice_list[0]
    sublatticeB = atom_lattice.sublattice_list[1]
    sublatticeA.construct_zone_axes()
    za0, za1 = sublatticeA.zones_axis_average_distances[0:2]
    s_p = sublatticeA.get_polarization_from_second_sublattice(
        za0, za1, sublatticeB, color='blue')
    vector_list = s_p.metadata.vector_list
    x, y = [i[0] for i in vector_list], [i[1] for i in vector_list]
    u, v = [i[2] for i in vector_list], [i[3] for i in vector_list]
    return sublatticeA, sublatticeB, x, y, u, v


@pytest.fixture()
def handle_plots():
    """Clean up images for tests."""
    # before test
    plt.ion()
    # during
    yield
    # after
    plt.close('all')
    plt.ioff()
