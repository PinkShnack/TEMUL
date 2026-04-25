import importlib
import sys
import types

import numpy as np
import pytest
from hyperspy.signals import Signal2D


class FakeMetadata:
    instances = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.go_called = False
        FakeMetadata.instances.append(self)

    def go(self):
        self.go_called = True


def _load_simulations_module(monkeypatch):
    FakeMetadata.instances.clear()
    monkeypatch.setitem(
        sys.modules,
        'pyprismatic',
        types.SimpleNamespace(Metadata=FakeMetadata),
    )
    import temul.simulations as simulations
    return importlib.reload(simulations)


def test_simulate_with_prismatic_raises_for_missing_xyz(monkeypatch, tmp_path):
    simulations = _load_simulations_module(monkeypatch)

    with pytest.raises(OSError, match='XYZ file not found'):
        simulations.simulate_with_prismatic(
            xyz_filename=str(tmp_path / 'missing.xyz'),
            filename='ignored_output',
            probeStep=1.0,
        )


def test_simulate_with_prismatic_raises_without_probe_info(
        monkeypatch, tmp_path):
    simulations = _load_simulations_module(monkeypatch)
    xyz_file = tmp_path / 'model.xyz'
    xyz_file.write_text('placeholder', encoding='utf-8')

    with pytest.raises(ValueError, match='Both reference_image and probeStep'):
        simulations.simulate_with_prismatic(
            xyz_filename=str(xyz_file),
            filename='ignored_output',
            reference_image=None,
            probeStep=None,
        )


def test_simulate_with_prismatic_uses_probe_step_and_runs_metadata(
        monkeypatch, tmp_path):
    simulations = _load_simulations_module(monkeypatch)
    xyz_file = tmp_path / 'model.xyz'
    xyz_file.write_text('placeholder', encoding='utf-8')

    simulations.simulate_with_prismatic(
        xyz_filename=str(xyz_file),
        filename='simulated_output',
        reference_image=None,
        probeStep=1.25,
        interpolationFactor=8,
        realspacePixelSize=0.12,
        numThreads=4,
        algorithm='multislice',
    )

    metadata = FakeMetadata.instances[-1]
    assert metadata.init_kwargs['filenameAtoms'] == str(xyz_file)
    assert metadata.probeStepX == pytest.approx(1.25)
    assert metadata.probeStepY == pytest.approx(1.25)
    assert metadata.interpolationFactorX == 8
    assert metadata.interpolationFactorY == 8
    assert metadata.realspacePixelSizeX == pytest.approx(0.12)
    assert metadata.realspacePixelSizeY == pytest.approx(0.12)
    assert metadata.numThreads == 4
    assert metadata.algorithm == 'multislice'
    assert metadata.filenameOutput == 'simulated_output.mrc'
    assert metadata.save2DOutput is True
    assert metadata.save3DOutput is False
    assert metadata.go_called is True


def test_simulate_with_prismatic_uses_reference_image_sampling(
        monkeypatch, tmp_path):
    simulations = _load_simulations_module(monkeypatch)
    xyz_file = tmp_path / 'model.xyz'
    xyz_file.write_text('placeholder', encoding='utf-8')
    reference_image = Signal2D(np.zeros((4, 4), dtype=float))
    reference_image.axes_manager[-1].scale = 0.5
    reference_image.axes_manager[-2].scale = 0.5
    reference_image.axes_manager[-1].units = 'nm'
    reference_image.axes_manager[-2].units = 'nm'

    simulations.simulate_with_prismatic(
        xyz_filename=str(xyz_file),
        filename='simulated_output',
        reference_image=reference_image,
        probeStep=9.0,
    )

    metadata = FakeMetadata.instances[-1]
    assert metadata.probeStepX == pytest.approx(5.000005)
    assert metadata.probeStepY == pytest.approx(5.000005)


def test_simulate_with_prismatic_sets_cell_and_tile_dimensions(
        monkeypatch, tmp_path):
    simulations = _load_simulations_module(monkeypatch)
    xyz_file = tmp_path / 'model.xyz'
    xyz_file.write_text('placeholder', encoding='utf-8')

    simulations.simulate_with_prismatic(
        xyz_filename=str(xyz_file),
        filename='simulated_output',
        probeStep=1.0,
        cellDimXYZ=(4.0, 5.0, 6.0),
        tileXYZ=(2, 3, 4),
    )

    metadata = FakeMetadata.instances[-1]
    assert metadata.cellDimX == 4.0
    assert metadata.cellDimY == 5.0
    assert metadata.cellDimZ == 6.0
    assert metadata.tileX == 2
    assert metadata.tileY == 3
    assert metadata.tileZ == 4


def test_simulate_and_calibrate_with_prismatic_calls_pipeline(monkeypatch):
    simulations = _load_simulations_module(monkeypatch)
    calls = {}
    expected_signal = Signal2D(np.ones((3, 3), dtype=float))

    def fake_simulate_with_prismatic(**kwargs):
        calls['simulate_with_prismatic'] = kwargs

    def fake_load_prismatic_mrc_with_hyperspy(filename, save_name=None):
        calls['load_prismatic_mrc_with_hyperspy'] = {
            'filename': filename,
            'save_name': save_name,
        }
        return expected_signal

    def fake_calibrate_intensity_distance_with_sublattice_roi(**kwargs):
        calls['calibrate_intensity_distance_with_sublattice_roi'] = kwargs

    monkeypatch.setattr(
        simulations, 'simulate_with_prismatic', fake_simulate_with_prismatic)
    monkeypatch.setattr(
        simulations,
        'load_prismatic_mrc_with_hyperspy',
        fake_load_prismatic_mrc_with_hyperspy,
    )
    monkeypatch.setattr(
        simulations,
        'calibrate_intensity_distance_with_sublattice_roi',
        fake_calibrate_intensity_distance_with_sublattice_roi,
    )

    reference_image = Signal2D(np.zeros((3, 3), dtype=float))
    result = simulations.simulate_and_calibrate_with_prismatic(
        xyz_filename='model.xyz',
        filename='example',
        reference_image=reference_image,
        calibration_area=[[0, 0], [2, 2]],
        calibration_separation=3,
        refine=False,
    )

    assert result is expected_signal
    assert calls['simulate_with_prismatic']['filename'] == 'example'
    assert calls['load_prismatic_mrc_with_hyperspy']['filename'] == (
        'prism_2Doutput_example.mrc'
    )
    assert calls['calibrate_intensity_distance_with_sublattice_roi'][
        'image'
    ] is expected_signal
    assert calls['calibrate_intensity_distance_with_sublattice_roi'][
        'refine'
    ] is False


def test_simulate_and_filter_and_calibrate_with_prismatic_calls_filter(
        monkeypatch):
    simulations = _load_simulations_module(monkeypatch)
    calls = {}
    loaded_signal = Signal2D(np.ones((3, 3), dtype=float))
    filtered_signal = Signal2D(np.full((3, 3), 2.0, dtype=float))

    monkeypatch.setattr(
        simulations,
        'simulate_with_prismatic',
        lambda **kwargs: calls.setdefault('simulate_with_prismatic', kwargs),
    )
    monkeypatch.setattr(
        simulations,
        'load_prismatic_mrc_with_hyperspy',
        lambda filename, save_name=None: loaded_signal,
    )

    def fake_compare_two_image_and_create_filtered_image(**kwargs):
        calls['compare_two_image_and_create_filtered_image'] = kwargs
        return filtered_signal

    monkeypatch.setattr(
        simulations,
        'compare_two_image_and_create_filtered_image',
        fake_compare_two_image_and_create_filtered_image,
    )

    reference_image = Signal2D(np.zeros((3, 3), dtype=float))
    result = simulations.simulate_and_filter_and_calibrate_with_prismatic(
        xyz_filename='model.xyz',
        filename='example',
        reference_image=reference_image,
        calibration_area=[[0, 0], [2, 2]],
        calibration_separation=3,
        delta_image_filter=0.2,
        refine=False,
    )

    assert result is filtered_signal
    assert calls['compare_two_image_and_create_filtered_image'][
        'image_to_filter'
    ] is loaded_signal
    assert calls['compare_two_image_and_create_filtered_image'][
        'reference_image'
    ] is reference_image


def test_simulation_helpers_reject_invalid_calibration_area(monkeypatch):
    simulations = _load_simulations_module(monkeypatch)
    reference_image = Signal2D(np.zeros((3, 3), dtype=float))

    with pytest.raises(ValueError, match='calibration_area must be two points'):
        simulations.simulate_and_calibrate_with_prismatic(
            xyz_filename='model.xyz',
            filename='example',
            reference_image=reference_image,
            calibration_area=[[0, 0]],
            calibration_separation=3,
        )

    with pytest.raises(ValueError, match='calibration_area must be two points'):
        simulations.simulate_and_filter_and_calibrate_with_prismatic(
            xyz_filename='model.xyz',
            filename='example',
            reference_image=reference_image,
            calibration_area=[[0, 0]],
            calibration_separation=3,
        )
