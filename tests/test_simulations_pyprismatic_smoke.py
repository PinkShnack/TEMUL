import importlib

import pytest


def test_pyprismatic_import_and_temul_simulations_module_load():
    pyprismatic = pytest.importorskip('pyprismatic')

    assert hasattr(pyprismatic, 'Metadata')

    simulations = importlib.import_module('temul.simulations')

    assert simulations.pr is pyprismatic
    assert hasattr(simulations, 'simulate_with_prismatic')


def test_pyprismatic_metadata_can_be_instantiated_with_filename(tmp_path):
    pyprismatic = pytest.importorskip('pyprismatic')
    xyz_file = tmp_path / 'smoke.xyz'
    xyz_file.write_text('placeholder', encoding='utf-8')

    metadata = pyprismatic.Metadata(filenameAtoms=str(xyz_file))

    assert metadata is not None
    assert getattr(metadata, 'filenameAtoms') == str(xyz_file)
