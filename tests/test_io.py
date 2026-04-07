import numpy as np

import temul.external.atomap_devel_012.dummy_data as atomap_dd
from temul.io import (
    convert_vesta_xyz_to_prismatic_xyz,
    create_dataframe_for_xyz,
    load_prismatic_mrc_with_hyperspy,
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
    assert np.isclose(atom_rows["_atom_site_occupancy"].astype(float), 1.0).all()
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
