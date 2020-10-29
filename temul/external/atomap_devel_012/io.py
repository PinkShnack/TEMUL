"""Module containing functions to save and load Atom_Lattice objects."""
import h5py
import os
from temul.external.atomap_devel_012.atom_lattice import Atom_Lattice
from temul.external.atomap_devel_012.sublattice import Sublattice
from temul.external.atomap_devel_012.atom_finding_refining import construct_zone_axes_from_sublattice
import numpy as np
####
import ast
####


def load_atom_lattice_from_hdf5(filename, construct_zone_axes=True):
    """
    Load an Atomap HDF5-file, restoring a saved Atom_Lattice.

    Parameters
    ----------
    filename : string
        Filename of the HDF5-file.
    construct_zone_axes : bool
        If True, find relations between atomic positions by
        constructing atomic planes. Default True.

    Returns
    -------
    Atomap Atom_Lattice object

    """
    h5f = h5py.File(filename, 'r')
    atom_lattice = Atom_Lattice()
    sublattice_list = []
    sublattice_index_list = []
    for group_name in h5f:
        if ('atom_lattice' in group_name) or ('sublattice' in group_name):
            sublattice_set = h5f[group_name]
            modified_image_data = sublattice_set['modified_image_data'][:]
            original_image_data = sublattice_set['original_image_data'][:]
            atom_position_array = sublattice_set['atom_positions'][:]

            if 'sublattice_index' in sublattice_set.attrs.keys():
                sublattice_index_list.append(
                    sublattice_set.attrs['sublattice_index'])

            sublattice = Sublattice(
                atom_position_array,
                modified_image_data)
            sublattice.original_image = original_image_data

            if 'sigma_x' in sublattice_set.keys():
                sigma_x_array = sublattice_set['sigma_x'][:]
                for atom, sigma_x in zip(
                        sublattice.atom_list,
                        sigma_x_array):
                    atom.sigma_x = sigma_x
            if 'sigma_y' in sublattice_set.keys():
                sigma_y_array = sublattice_set['sigma_y'][:]
                for atom, sigma_y in zip(
                        sublattice.atom_list,
                        sigma_y_array):
                    atom.sigma_y = sigma_y
            if 'rotation' in sublattice_set.keys():
                rotation_array = sublattice_set['rotation'][:]
                for atom, rotation in zip(
                        sublattice.atom_list,
                        rotation_array):
                    atom.rotation = rotation
####
            if 'amplitude_max_intensity' in sublattice_set.keys():
                amplitude_max_intensity_array = sublattice_set['amplitude_max_intensity'][:]
                for atom, amplitude_max_intensity in zip(
                        sublattice.atom_list,
                        amplitude_max_intensity_array):
                    atom.amplitude_max_intensity = amplitude_max_intensity

            if 'amplitude_mean_intensity' in sublattice_set.keys():
                amplitude_mean_intensity_array = sublattice_set['amplitude_mean_intensity'][:]
                for atom, amplitude_mean_intensity in zip(
                        sublattice.atom_list,
                        amplitude_mean_intensity_array):
                    atom.amplitude_mean_intensity = amplitude_mean_intensity

            if 'amplitude_min_intensity' in sublattice_set.keys():
                amplitude_min_intensity_array = sublattice_set['amplitude_min_intensity'][:]
                for atom, amplitude_min_intensity in zip(
                        sublattice.atom_list,
                        amplitude_min_intensity_array):
                    atom.amplitude_min_intensity = amplitude_min_intensity

            if 'amplitude_total_intensity' in sublattice_set.keys():
                amplitude_total_intensity_array = sublattice_set['amplitude_total_intensity'][:]
                for atom, amplitude_total_intensity in zip(
                        sublattice.atom_list,
                        amplitude_total_intensity_array):
                    atom.amplitude_total_intensity = amplitude_total_intensity

            if 'elements' in sublattice_set.keys():
                elements_array = sublattice_set['elements'][:]
                for atom, elements in zip(
                        sublattice.atom_list,
                        elements_array):
                    atom.elements = elements

            if 'z_height' in sublattice_set.keys():
                z_height_array = sublattice_set['z_height'][:]
                # z_height_array_2 = [] # first loop needed because i can't eval() the z_height itself, don't really know why
                # for i in range(0, len(z_height_array)):
                #   z_h = ast.literal_eval(z_height_array[i])
                #  z_height_array_2.append(z_h)
                for atom, z_height in zip(
                        sublattice.atom_list,
                        z_height_array):
                    atom.z_height = z_height
####
            sublattice.pixel_size = sublattice_set.attrs['pixel_size']

            if 'tag' in sublattice_set.attrs.keys():
                sublattice.name = sublattice_set.attrs['tag']
            elif 'name' in sublattice_set.attrs.keys():
                sublattice.name = sublattice_set.attrs['name']
            else:
                sublattice.name = ''

            if type(sublattice.name) == bytes:
                sublattice.name = sublattice.name.decode()

            sublattice._plot_color = sublattice_set.attrs['plot_color']

            if type(sublattice._plot_color) == bytes:
                sublattice._plot_color = sublattice._plot_color.decode()

            if 'pixel_separation' in sublattice_set.attrs.keys():
                sublattice._pixel_separation = sublattice_set.attrs[
                    'pixel_separation']
            else:
                sublattice._pixel_separation = 0.0

            if construct_zone_axes:
                construct_zone_axes_from_sublattice(sublattice)

            if 'zone_axis_names_byte' in sublattice_set.keys():
                zone_axis_list_byte = sublattice_set.attrs[
                    'zone_axis_names_byte']
                zone_axis_list = []
                for zone_axis_name_byte in zone_axis_list_byte:
                    zone_axis_list.append(zone_axis_name_byte.decode())
                sublattice.zones_axis_average_distances_names = zone_axis_list

            sublattice_list.append(sublattice)

        if group_name == 'image_data0':
            atom_lattice.image0 = h5f[group_name][:]
            atom_lattice.image = atom_lattice.image0
        if group_name == 'image_data1':
            atom_lattice.image1 = h5f[group_name][:]

    sorted_sublattice_list = []
    if not sublattice_index_list:  # Support for older hdf5 files
        sublattice_index_list = range(len(sublattice_list))
    for sublattice_index in sublattice_index_list:
        sorted_sublattice_list.append(sublattice_list[sublattice_index])
    atom_lattice.sublattice_list.extend(sorted_sublattice_list)
    if 'name' in h5f.attrs.keys():
        atom_lattice.name = h5f.attrs['name']
    elif 'path_name' in h5f.attrs.keys():
        atom_lattice.name = h5f.attrs['path_name']
    if 'pixel_separation' in h5f.attrs.keys():
        atom_lattice._pixel_separation = h5f.attrs['pixel_separation']
    else:
        atom_lattice._pixel_separation = 0.8 / sublattice.pixel_size
    if type(atom_lattice.name) == bytes:
        atom_lattice.name = atom_lattice.name.decode()
    h5f.close()
    return(atom_lattice)


def save_atom_lattice_to_hdf5(atom_lattice, filename, overwrite=False):
    """Store an Atom_Lattice object as an HDF5-file.

    Parameters
    ----------
    atom_lattice : Atomap Atom_Lattice object
    filename : string
    overwrite : bool, default False

    """
    if os.path.isfile(filename) and not overwrite:
        raise FileExistsError(
            "The file %s already exist, either change the name or "
            "use overwrite=True")
    elif os.path.isfile(filename) and overwrite:
        os.remove(filename)

    h5f = h5py.File(filename, 'w')
    for index, sublattice in enumerate(atom_lattice.sublattice_list):
        subgroup_name = sublattice.name + "_sublattice"
        if subgroup_name in h5f:
            subgroup_name = str(index) + subgroup_name

        modified_image_data = sublattice.image
        original_image_data = sublattice.original_image

        # Atom position data
        atom_positions = np.array([
            sublattice.x_position,
            sublattice.y_position]).swapaxes(0, 1)

#            atom_positions = np.array(sublattice._get_atom_position_list())
        sigma_x = np.array(sublattice.sigma_x)
        sigma_y = np.array(sublattice.sigma_y)
        rotation = np.array(sublattice.rotation)
####
        amplitude_max_intensity = np.array(
            sublattice.atom_amplitude_max_intensity)
        amplitude_mean_intensity = np.array(
            sublattice.atom_amplitude_mean_intensity)
        amplitude_min_intensity = np.array(
            sublattice.atom_amplitude_min_intensity)
        amplitude_total_intensity = np.array(
            sublattice.atom_amplitude_total_intensity)
        elements = np.array(sublattice.elements, dtype=np.dtype('O'))
        z_height = np.array(sublattice.z_height, dtype=np.dtype('O'))

####
        h5f.create_dataset(
            subgroup_name + "/modified_image_data",
            data=modified_image_data,
            chunks=True,
            compression='gzip')
        h5f.create_dataset(
            subgroup_name + "/original_image_data",
            data=original_image_data,
            chunks=True,
            compression='gzip')

        h5f.create_dataset(
            subgroup_name + "/atom_positions",
            data=atom_positions,
            chunks=True,
            compression='gzip')
        h5f.create_dataset(
            subgroup_name + "/sigma_x",
            data=sigma_x,
            chunks=True,
            compression='gzip')
        h5f.create_dataset(
            subgroup_name + "/sigma_y",
            data=sigma_y,
            chunks=True,
            compression='gzip')
        h5f.create_dataset(
            subgroup_name + "/rotation",
            data=rotation,
            chunks=True,
            compression='gzip')
####
        h5f.create_dataset(
            subgroup_name + "/amplitude_max_intensity",
            data=amplitude_max_intensity,
            chunks=True,
            compression='gzip')
        h5f.create_dataset(
            subgroup_name + "/amplitude_mean_intensity",
            data=amplitude_mean_intensity,
            chunks=True,
            compression='gzip')
        h5f.create_dataset(
            subgroup_name + "/amplitude_min_intensity",
            data=amplitude_min_intensity,
            chunks=True,
            compression='gzip')
        h5f.create_dataset(
            subgroup_name + "/amplitude_total_intensity",
            data=amplitude_total_intensity,
            chunks=True,
            compression='gzip')
        h5f.create_dataset(
            subgroup_name + "/elements",
            data=elements,
            chunks=True,
            compression='gzip',
            dtype=h5py.special_dtype(vlen=str))
        h5f.create_dataset(
            subgroup_name + "/z_height",
            data=z_height,
            chunks=True,
            compression='gzip',
            dtype=h5py.special_dtype(vlen=str))
####
        h5f[subgroup_name].attrs['pixel_size'] = sublattice.pixel_size
        h5f[subgroup_name].attrs[
            'pixel_separation'] = sublattice._pixel_separation
        h5f[subgroup_name].attrs['name'] = sublattice.name
        h5f[subgroup_name].attrs['plot_color'] = sublattice._plot_color
        h5f[subgroup_name].attrs['sublattice_index'] = index

        # HDF5 does not supporting saving a list of strings, so converting
        # them to bytes
        zone_axis_names = sublattice.zones_axis_average_distances_names
        zone_axis_names_byte = []
        for zone_axis_name in zone_axis_names:
            zone_axis_names_byte.append(zone_axis_name.encode())
        h5f[subgroup_name].attrs[
            'zone_axis_names_byte'] = zone_axis_names_byte

    h5f.create_dataset(
        "image_data0",
        data=atom_lattice.image0,
        chunks=True,
        compression='gzip')
    if hasattr(atom_lattice, 'image1'):
        h5f.create_dataset(
            "image_data1",
            data=atom_lattice.image1,
            chunks=True,
            compression='gzip')
    h5f.attrs['name'] = atom_lattice.name
    h5f.attrs['pixel_separation'] = atom_lattice._pixel_separation

    h5f.close()
