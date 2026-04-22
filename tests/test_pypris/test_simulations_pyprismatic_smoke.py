import importlib

import pytest


def _import_real_pyprismatic():
    pyprismatic = pytest.importorskip('pyprismatic')
    if not hasattr(pyprismatic, 'Metadata'):
        pytest.skip('real pyprismatic is not active in this test process')
    return pyprismatic


def _write_minimal_prismatic_xyz(path):
    path.write_text(
        "\n".join([
            "TEMUL pyprismatic integration test",
            "4.000000 4.000000 4.000000",
            "14 2.000000 2.000000 2.000000 1.0 0.1",
            "-1",
        ]) + "\n",
        encoding='utf-8',
    )


def test_pyprismatic_import_and_temul_simulations_module_load():
    pyprismatic = _import_real_pyprismatic()

    simulations = importlib.import_module('temul.simulations')

    assert simulations.pr is pyprismatic
    assert hasattr(simulations, 'simulate_with_prismatic')


def test_pyprismatic_metadata_can_be_instantiated_with_filename(tmp_path):
    pyprismatic = _import_real_pyprismatic()
    xyz_file = tmp_path / 'smoke.xyz'
    _write_minimal_prismatic_xyz(xyz_file)

    metadata = pyprismatic.Metadata(filenameAtoms=str(xyz_file))

    assert metadata is not None
    assert getattr(metadata, 'filenameAtoms') == str(xyz_file)


def test_simulate_with_prismatic_creates_mrc_output(tmp_path, monkeypatch):
    _import_real_pyprismatic()
    simulations = importlib.import_module('temul.simulations')

    xyz_file = tmp_path / 'integration.xyz'
    _write_minimal_prismatic_xyz(xyz_file)
    monkeypatch.chdir(tmp_path)

    simulations.simulate_with_prismatic(
        xyz_filename=str(xyz_file),
        filename='integration_output',
        reference_image=None,
        probeStep=1.0,
        interpolationFactor=4,
        realspacePixelSize=0.2,
        numFP=1,
        scanWindowMin=0.0,
        scanWindowMax=0.125,
        numThreads=1,
        algorithm='prism',
    )

    mrc_candidates = sorted(tmp_path.glob('*integration_output*.mrc'))

    assert mrc_candidates
    assert any(path.stat().st_size > 0 for path in mrc_candidates)
