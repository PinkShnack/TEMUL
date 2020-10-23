
import numpy as np
from scipy.optimize import curve_fit
from scipy.misc import derivative
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from temul.signal_processing import sine_wave_function_strain_gradient


def calculate_atom_plane_curvature(sublattice, zone_vector_index,
                                   func='strain_grad', atom_planes=None,
                                   sampling=None, units='pix', vmin=None,
                                   vmax=None, cmap='inferno',
                                   title='Curvature Map', filename=None,
                                   plot_and_return_fits=False, **kwargs):
    """
    Calculates the curvature of the sublattice atom planes along the direction
    given by `zone_vector_index`. In the case of [1] below, the curvature is
    the inverse of the radius of curvature, and is effectively equal to the
    second derivative of the displacement direction of the atoms. Because the
    first derivative is negligible, the curvature can be calculated as the
    strain gradient [2].
    With the parameter func="strain_grad", this function calculates the strain
    gradient of the atom planes of a Atomap Sublattice object.

    Parameters
    ----------
    sublattice : Atomap Sublattice object
    zone_vector_index : int
        The index of the zone axis (translation symmetry) found by the Atomap
        function `construct_zone_axes()`.
    func : 'strain_grad' or function
        Function that can be used by `scipy.optimize.curve_fit`. If
        func='strain_grad', then the
        `temul.signal_processing.sine_wave_function_strain_gradient` function
        will be used.
    atom_planes : tuple, optional
        The starting and ending atom plane to be computed. Useful if only a
        section of the image should be fitted with sine waves. Given in the
        form e.g., (0, 3).
    sampling : float, optional
        sampling of an image in units of units/pix
    units : string, default "pix"
        Units of sampling, for display purposes.
    vmin, vmax, cmap : see Matplotlib documentation, default None
    title : string, default 'Curvature Map'
    filename : string, default None
        Name of the file to be saved.
    plot_and_return_fits : Bool, default False
        If set to True, each atom plane fitting will be plotted along with its
        respective atom positions. The fitting parameters (popt) will be
        returned as a list.
    **kwargs
        keyword arguments to be passed to `scipy.optimize.curve_fit`.

    Examples
    --------
    >>> from temul.dummy_data import sine_wave_sublattice
    >>> from temul.lattice_structure_tools import (
    ...     calculate_atom_plane_curvature)
    >>> sublattice = sine_wave_sublattice()
    >>> sublattice.construct_zone_axes(atom_plane_tolerance=1)
    >>> sublattice.plot()
    >>> sampling = 0.05 #  nm/pix
    >>> cmap='bwr'
    >>> curvature_map = calculate_atom_plane_curvature(sublattice,
    ...         zone_vector_index=0, sampling=sampling, units='nm', cmap=cmap)

    Just compute several atom planes:

    >>> curvature_map = calculate_atom_plane_curvature(sublattice, 0,
    ...         atom_planes=(0,3), sampling=sampling, units='nm', cmap=cmap)

    You can also provide initial fitting estimations via scipy's curve_fit:

    >>> p0 = [2, 1, 1, 15]
    >>> kwargs = {'p0': p0}
    >>> curvature_map, fittings = calculate_atom_plane_curvature(sublattice,
    ...         zone_vector_index=0, atom_planes=(0,3), sampling=sampling,
    ...         units='nm', cmap=cmap, **kwargs, plot_and_return_fits=True)

    Returns
    -------
    Curvature Map as a Hyperspy Signal2D

    References
    ----------
    .. [1] Reference: Function adapted from a script written by
    Dr. Marios Hadjimichael, and used in paper_name. The original MATLAB script
    can be found in TEMUL/publication_examples/PTO_marios_hadj
    .. [2] Reference: Landau and Lifshitz, Theory of Elasticity, Vol 7,
    pp 47-49, 1981

    """

    if sublattice.zones_axis_average_distances is None:
        raise Exception(
            "zones_axis_average_distances is empty. "
            "Has sublattice.construct_zone_axes() been run?")
    else:
        zone_vector = sublattice.zones_axis_average_distances[
            zone_vector_index]

    atom_plane_list = sublattice.atom_planes_by_zone_vector[zone_vector]

    if atom_planes is not None:
        atom_plane_list = atom_plane_list[atom_planes[0]:atom_planes[1]]

    if func == 'strain_grad':
        func = sine_wave_function_strain_gradient

    curvature = []
    x_list, y_list = [], []
    fittings_list = []
    for atom_plane in atom_plane_list:
        # fit a sine wave to the atoms in the atom_plane
        params, _ = curve_fit(func, atom_plane.x_position,
                              atom_plane.y_position, **kwargs)

        # calculate the second derivative of the sine wave
        #   with respect to x analytically (to extract the strain gradient)
        second_der = derivative(func,
                                np.asarray(atom_plane.x_position),
                                dx=1e-6, n=2, args=(params))

        if plot_and_return_fits:
            fittings_list.append(params)
            plt.figure()
            plt.scatter(atom_plane.x_position, atom_plane.y_position)
            plt.plot(atom_plane.x_position,
                     func(atom_plane.x_position, *params), 'r-',
                     label=f'fit params: {params}')
            plt.legend(loc='lower left')
            plt.show()

        x_list.extend(atom_plane.x_position)
        y_list.extend(atom_plane.y_position)
        curvature.extend(list(second_der))

    if sampling is not None:
        curvature = [i * sampling for i in curvature]
    curvature_map = sublattice.get_property_map(
        x_list, y_list, curvature, upscale_map=1)
    if sampling is not None:
        curvature_map.axes_manager[0].scale = sampling
        curvature_map.axes_manager[1].scale = sampling
    curvature_map.axes_manager[0].units = units
    curvature_map.axes_manager[1].units = units

    curvature_map.plot(vmin=vmin, vmax=vmax, cmap=cmap,
                             colorbar=False)
    # need to put in colorbar axis units like in get_strain_map
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.title("{} of Index {}".format(title, zone_vector_index))
    cbar = ScalarMappable(cmap=cmap)
    cbar.set_array(curvature)
    cbar.set_clim(vmin, vmax)
    plt.colorbar(cbar, fraction=0.046, pad=0.04,
                 label=f"Curvature (1/{units})")
    plt.tight_layout()

    if filename is not None:
        plt.savefig(fname="{}_{}_{}.png".format(
            filename, title, zone_vector_index),
            transparent=True, frameon=False, bbox_inches='tight',
            pad_inches=None, dpi=300, labels=False)
        curvature_map.save("{}_{}_{}.hspy".format(
            filename, title, zone_vector_index))

    if plot_and_return_fits:
        return(curvature_map, fittings_list)
    else:
        return(curvature_map)
