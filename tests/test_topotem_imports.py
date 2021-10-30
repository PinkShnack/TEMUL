"""
Test whether the topotem imports work from temul.api
"""

import pytest


def test_polarisation_old_import():
    with pytest.raises(DeprecationWarning):
        import temul.polarisation  # noqa: F401


def test_lattice_structure_tools_old_import():
    with pytest.raises(DeprecationWarning):
        import temul.lattice_structure_tools  # noqa: F401
