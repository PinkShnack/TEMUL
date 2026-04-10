from pathlib import Path

import hyperspy.api as hs
import numpy as np
import pytest

import temul.external.atomap_devel_012.dummy_data as atomap_dd
from temul.io import (
    convert_vesta_xyz_to_prismatic_xyz,
    create_dataframe_for_xyz,
    load_data_and_sampling,
    load_prismatic_mrc_with_hyperspy,
    save_individual_images_from_image_stack,
)


def test_convert_vesta_xyz_to_prismatic_xyz_returns_expected_dataframe(
        tmp_path):
    output_path = tmp_path / "converted_prismatic.xyz"
    dataframe = convert_vesta_xyz_to_prismatic_xyz(
        "temul/example_data/prismatic/example_MoS2_vesta_xyz.xyz",
        str(output_path),
        delimiter="   |    |  ",
        header=None,
        skiprows=[0, 1],
        engine="python",
        occupancy=1.0,
        rms_thermal_vib=0.05,
        header_comment="Let's do this!",
        save=True,
    )

    assert output_path.exists()
    assert dataframe.iloc[0, 0] == "Let's do this!"
    assert float(dataframe.iloc[1, 1]) > 0
    assert dataframe.iloc[-1, 0] == -1

    atom_rows = dataframe.iloc[2:-1]
    assert atom_rows["_atom_site_Z_number"].map(
        lambda value: isinstance(value, (int, np.integer))).all()
    assert np.isclose(
        atom_rows["_atom_site_occupancy"].astype(float), 1.0).all()
    assert np.isclose(
        atom_rows["_atom_site_RMS_thermal_vib"].astype(float), 0.05).all()


def test_create_dataframe_for_xyz_returns_header_atom_rows_and_footer():
    sublattice = atomap_dd.get_simple_cubic_sublattice()
    for atom in sublattice.atom_list:
        atom.elements = "Mo_1"
        atom.z_height = "0.5"

    dataframe = create_dataframe_for_xyz(
        [sublattice],
        ["Mo_0", "Mo_1", "Mo_2"],
        x_size=50,
        y_size=50,
        z_size=5,
        filename=None,
        header_comment="Here is an Example",
    )

    assert dataframe.iloc[0, 0] == "Here is an Example"
    assert float(dataframe.iloc[1, 1]) == 50.0
    assert dataframe.iloc[-1, 0] == -1

    atom_rows = dataframe.iloc[2:-1]
    assert len(atom_rows) == len(sublattice.atom_list)
    assert atom_rows["_atom_site_Z_number"].eq(42).all()


def test_load_prismatic_mrc_with_hyperspy_returns_signal2d():
    signal = load_prismatic_mrc_with_hyperspy(
        "temul/example_data/prismatic/prism_2Doutput_prismatic_simulation.mrc",
        save_name=None,
    )

    assert signal.data.ndim == 2
    assert signal.data.shape == (773, 1182)


def test_load_data_and_sampling_sets_pixel_units_without_saving(tmp_path):
    signal = hs.signals.Signal2D(np.arange(16, dtype=float).reshape(4, 4))
    signal.axes_manager[-1].scale = 1
    signal.axes_manager[-2].scale = 1
    filename = tmp_path / 'image.hspy'
    signal.save(filename)

    loaded, sampling = load_data_and_sampling(
        str(filename), invert_image=False, save_image=False)

    assert sampling == 1
    assert loaded.axes_manager[-1].units == 'pixels'
    assert loaded.axes_manager[-2].units == 'pixels'


def test_load_data_and_sampling_inverts_image_without_saving(tmp_path):
    signal = hs.signals.Signal2D(np.full((3, 3), 2.0))
    signal.axes_manager[-1].scale = 0.5
    signal.axes_manager[-2].scale = 0.5
    filename = tmp_path / 'invert.hspy'
    signal.save(filename)

    loaded, sampling = load_data_and_sampling(
        str(filename), invert_image=True, save_image=False)

    assert sampling == 0.5
    assert np.allclose(loaded.data, 0.5)
    assert loaded.axes_manager[-1].units == 'nm'
    assert loaded.axes_manager[-2].units == 'nm'


def test_save_individual_images_from_image_stack_writes_all_frames(tmp_path):
    stack = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    output_dir = tmp_path / 'individual_images'

    save_individual_images_from_image_stack(
        stack,
        output_folder=str(output_dir),
    )

    files = sorted(output_dir.glob('images_aligned_*.tif'))
    assert len(files) == 4
    assert all(Path(file).is_file() for file in files)


def test_convert_vesta_xyz_to_prismatic_xyz_applies_edge_padding(tmp_path):
    output_path = tmp_path / "padded_prismatic.xyz"
    unpadded = convert_vesta_xyz_to_prismatic_xyz(
        "temul/example_data/prismatic/example_MoS2_vesta_xyz.xyz",
        str(tmp_path / "unpadded_prismatic.xyz"),
        delimiter="   |    |  ",
        header=None,
        skiprows=[0, 1],
        engine="python",
        save=False,
    )
    dataframe = convert_vesta_xyz_to_prismatic_xyz(
        "temul/example_data/prismatic/example_MoS2_vesta_xyz.xyz",
        str(output_path),
        delimiter="   |    |  ",
        header=None,
        skiprows=[0, 1],
        engine="python",
        edge_padding=(1, 1, 2),
        save=False,
    )

    unit_cell_row = dataframe.iloc[1]
    unpadded_unit_cell_row = unpadded.iloc[1]
    assert float(unit_cell_row["_atom_site_fract_x"]) == pytest.approx(
        float(unpadded_unit_cell_row["_atom_site_fract_x"])
    )
    assert float(unit_cell_row["_atom_site_fract_y"]) == pytest.approx(
        float(unpadded_unit_cell_row["_atom_site_fract_y"])
    )
    assert float(unit_cell_row["_atom_site_fract_z"]) == pytest.approx(
        float(unpadded_unit_cell_row["_atom_site_fract_z"]) * 2
    )


def test_convert_vesta_xyz_to_prismatic_xyz_rejects_bad_edge_padding(
        tmp_path):
    with pytest.raises(ValueError, match="tuple of length 3"):
        convert_vesta_xyz_to_prismatic_xyz(
            "temul/example_data/prismatic/example_MoS2_vesta_xyz.xyz",
            str(tmp_path / "bad_padding.xyz"),
            delimiter="   |    |  ",
            header=None,
            skiprows=[0, 1],
            engine="python",
            edge_padding=(1, 2),
            save=False,
        )


def test_create_dataframe_for_xyz_expands_multi_atom_element_entries():
    sublattice = atomap_dd.get_simple_cubic_sublattice()
    atom = sublattice.atom_list[0]
    atom.elements = "Mo_2"
    atom.z_height = "0.25,0.75"
    for other_atom in sublattice.atom_list[1:]:
        other_atom.elements = "Mo_0"
        other_atom.z_height = "0.5"

    dataframe = create_dataframe_for_xyz(
        [sublattice],
        ["Mo_0", "Mo_1", "Mo_2"],
        x_size=20,
        y_size=20,
        z_size=10,
        filename=None,
    )

    atom_rows = dataframe.iloc[2:-1]
    assert len(atom_rows) == 2
    assert atom_rows["_atom_site_Z_number"].eq(42).all()
    assert atom_rows["_atom_site_fract_z"].tolist() == ["2.500000", "7.500000"]
