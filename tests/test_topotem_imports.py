"""
Test whether the topotem imports work from temul.api
"""

import temul.api as tml
import temul.topotem as tt


def test_topotem_polarisation_functions():
    pos_a = [[1, 2], [3, 4], [5, 8], [5, 2]]
    pos_b = [[1, 1], [5, 2], [3, 1], [6, 2]]
    u, v = tt.find_polarisation_vectors(pos_a, pos_b, save=None)
    assert isinstance(u, list)


def test_tml_polarisation_functions():
    pos_a = [[1, 2], [3, 4], [5, 8], [5, 2]]
    pos_b = [[1, 1], [5, 2], [3, 1], [6, 2]]
    u, v = tml.find_polarisation_vectors(pos_a, pos_b, save=None)
    assert isinstance(u, list)
