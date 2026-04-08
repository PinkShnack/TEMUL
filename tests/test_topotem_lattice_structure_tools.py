import numpy as np

from temul.dummy_data import sine_wave_sublattice
import temul.topotem.lattice_structure_tools as lst


def test_second_derivative_matches_quadratic():
    def quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c

    x_values = np.linspace(-2, 2, 11)
    derivative = lst._second_derivative(
        quadratic, x_values, params=(3.0, -1.0, 5.0))

    assert np.allclose(derivative, np.full_like(x_values, 6.0), atol=5e-3)


def test_sine_wave_function_strain_gradient_output_shape():
    x_values = np.linspace(0, 10, 25)
    y_values = lst.sine_wave_function_strain_gradient(x_values, 2.0, 1.0, 5.0,
                                                      3.0)

    assert y_values.shape == x_values.shape
    assert np.isfinite(y_values).all()


def test_calculate_atom_plane_curvature_returns_signal_and_fits():
    sublattice = sine_wave_sublattice()
    sublattice.construct_zone_axes(atom_plane_tolerance=1)

    curvature_map, fits = lst.calculate_atom_plane_curvature(
        sublattice,
        zone_vector_index=0,
        atom_planes=(0, 3),
        sampling=0.05,
        units="nm",
        return_fits=True,
        p0=[2, 1, 1, 15],
    )

    assert curvature_map.data.ndim == 2
    assert curvature_map.axes_manager[0].scale == 0.05
    assert curvature_map.axes_manager[1].scale == 0.05
    assert curvature_map.axes_manager[0].units == "nm"
    assert curvature_map.axes_manager[1].units == "nm"
    assert len(fits) == 3
    assert all(len(params) == 4 for params in fits)
    assert np.isfinite(curvature_map.data).any()


def test_calculate_atom_plane_curvature_with_default_function():
    sublattice = sine_wave_sublattice()
    sublattice.construct_zone_axes(atom_plane_tolerance=1)

    curvature_map = lst.calculate_atom_plane_curvature(
        sublattice,
        zone_vector_index=0,
        sampling=0.05,
        units="nm",
    )

    assert curvature_map.data.ndim == 2
    assert curvature_map.axes_manager[0].units == "nm"
