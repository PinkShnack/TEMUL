
from temul.element_tools import (
    get_and_return_element, split_and_sort_element)

import matplotlib.pyplot as plt
from tifffile import imwrite
import numpy as np
import hyperspy.api as hs
import CifFile
import pandas as pd
from glob import glob
import os


def batch_convert_emd_to_image(extension_to_save,
                               top_level_directory,
                               glob_search="**/*",
                               overwrite=True):
    """
    Convert all .emd files to the chosen extension_to_save file format in the
    specified directory and all subdirectories.

    Parameters
    ----------
    extension_to_save : string
        the image file extension to be used for saving the image.
        See Hyperspy documentation for information on file writing extensions
        available: http://hyperspy.org/hyperspy-doc/current/user_guide/io.html
    top_level_directory : string
        The top-level directory in which the emd files exist. The default
        glob_search will search this directory and all subdirectories.
    glob_search : string, default "**/*"
        Glob search string, see glob for more details:
        https://docs.python.org/2/library/glob.html
        Default will search this directory and all subdirectories.
    overwrite : Bool, default True
        Overwrite if the extension_to_save file already exists.

    """

    if "." not in extension_to_save:
        extension_to_save = "." + extension_to_save

    os.chdir(top_level_directory)

    filenames = glob(glob_search + ".emd", recursive=True)

    for filename in filenames:
        if 'EDS' in filename:
            pass
        elif ('-DF4-' or '-DF2-' or '-BF-') in filename:
            pass

        elif '.emd' in filename:

            s = hs.load(filename)
            print('Processing image: ' + filename)

            if type(s) == list:
                aligned_image = s[0].inav[-3:-2]
                s = aligned_image.sum('Time')

            elif len(s.axes_manager._axes) > 2:
                pass
            else:
                filename = filename[:-4]
                s.change_dtype("float32")
                s.save(filename + extension_to_save, overwrite=True)


def load_data_and_sampling(filename, file_extension=None,
                           invert_image=False, save_image=True):

    if '.' in filename:
        s = hs.load(filename)
    else:
        s = hs.load(filename + file_extension)
    # s.plot()

    # Get correct xy units and sampling
    if s.axes_manager[-1].scale == 1:
        real_sampling = 1
        s.axes_manager[-1].units = 'pixels'
        s.axes_manager[-2].units = 'pixels'
        print("WARNING: Image calibrated to pixels, you should "
              "calibrate to distance")
    elif s.axes_manager[-1].scale != 1:
        real_sampling = s.axes_manager[-1].scale
        s.axes_manager[-1].units = 'nm'
        s.axes_manager[-2].units = 'nm'

    # real_sampling =
    # physical_image_size = real_sampling * len(s.data)
    save_name = filename[:-4]

    if invert_image is True:
        s.data = np.divide(1, s.data)

        if save_image is True:

            s.plot()
            plt.title(save_name, fontsize=20)
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.tight_layout()
            plt.savefig(fname=save_name + '.png',
                        transparent=True, frameon=False, bbox_inches='tight',
                        pad_inches=None, dpi=300, labels=False)
            plt.close()
        else:
            pass

    else:
        if save_image is True:
            s.plot()
            plt.title(save_name, fontsize=20)
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.tight_layout()
            plt.savefig(fname=save_name + '.png',
                        transparent=True, frameon=False, bbox_inches='tight',
                        pad_inches=None, dpi=300, labels=False)
            plt.close()
        else:
            pass

    return s, real_sampling


def convert_vesta_xyz_to_prismatic_xyz(vesta_xyz_filename,
                                       prismatic_xyz_filename,
                                       delimiter='   |    |  ',
                                       header=None,
                                       skiprows=[0, 1],
                                       engine='python',
                                       occupancy=1.0,
                                       rms_thermal_vib=0.05,
                                       edge_padding=None,
                                       header_comment="Let's make a file!",
                                       save=True):
    '''
    Convert from Vesta outputted xyz file format to the prismatic-style xyz
    format.
    Lose some information from the .cif or .vesta file but okay for now.
    Develop your own converter if you need rms and occupancy! Lots to do.

    delimiter='      |       |  ' # ase xyz
    delimiter='   |    |  ' # vesta xyz

    Parameters
    ----------
    vesta_xyz_filename : string
        name of the vesta outputted xyz file. See vesta > export > xyz
    prismatic_xyz_filename : string
        name to be given to the outputted prismatic xyz file
    delimiter, header, skiprows, engine : pandas.read_csv input parameters
        See pandas.read_csv for documentation
        Note that the delimiters here are only available if you use
        engine='python'
    occupancy, rms_thermal_vib : see prismatic documentation
        if you want a file format that will retain these atomic attributes,
        use a format other than vesta xyz. Maybe .cif or .vesta keeps these?
    header_comment : string
        header comment for the file.
    save : Bool, default True
        whether to output the file as a prismatic formatted xyz file with the
        name of the file given by "prismatic_xyz_filename".

    Returns
    -------
    The converted file format as a pandas dataframe

    Examples
    --------

    See example_data for the vesta xyz file.

    >>> from temul.io import convert_vesta_xyz_to_prismatic_xyz
    >>> prismatic_xyz = convert_vesta_xyz_to_prismatic_xyz(
    ...     'temul/example_data/prismatic/example_MoS2_vesta_xyz.xyz',
    ...     'temul/example_data/prismatic/MoS2_hex_prismatic.xyz',
    ...     delimiter='   |    |  ', header=None, skiprows=[0, 1],
    ...     engine='python', occupancy=1.0, rms_thermal_vib=0.05,
    ...     header_comment="Let's do this!", save=True)

    '''

    file = pd.read_csv(vesta_xyz_filename,
                       delimiter=delimiter,
                       header=header,
                       skiprows=skiprows,
                       engine=engine)

    # check if there are nans, happens when the file wasn't read correctly
    for i in file.values:
        for value in i:
            if 'nan' in str(value):
                print("ERROR: nans present, file not read correctly. "
                      "Try changing the delimiters! "
                      "See: https://stackoverflow.com/"
                      "questions/51195299/python-reading-a-data-text-file"
                      "-with-different-delimiters")

    file.columns = ['_atom_site_Z_number',
                    '_atom_site_fract_x',
                    '_atom_site_fract_y',
                    '_atom_site_fract_z']

    # change all elements to atomic number
    for i, element_symbol in enumerate(file.loc[:, '_atom_site_Z_number']):
        element = get_and_return_element(element_symbol=element_symbol)
        file.loc[i, '_atom_site_Z_number'] = element.number

    # add occupancy and rms values
    file['_atom_site_occupancy'] = occupancy
    file['_atom_site_RMS_thermal_vib'] = rms_thermal_vib

    # add unit cell dimensions in angstroms
    axis_column_names = [file.columns[1],
                         file.columns[2],
                         file.columns[3]]
    unit_cell_dimen = []
    for name in axis_column_names:
        # round to 4 decimal places
        file[name] = file[name].round(6)

        axis_values_list = [
            x for x in file.loc[0:file.shape[0], name].values
            if not isinstance(x, str)]
        min_axis = min(axis_values_list)
        max_axis = max(axis_values_list)
        unit_cell_dimen_axis = max_axis - min_axis
        unit_cell_dimen_axis = format(unit_cell_dimen_axis, '.6f')
        unit_cell_dimen.append(unit_cell_dimen_axis)

    # print(unit_cell_dimen)
    unit_cell_dimen = [float(unit_cell) for unit_cell in unit_cell_dimen]
    # should match the vesta values (or be slightly larger)

    if edge_padding is not None:
        if type(edge_padding) is tuple and len(edge_padding) == 3:
            for i, padding in enumerate(edge_padding):
                unit_cell_dimen[i] *= padding

            # x, y, z translations
            file.loc[:, '_atom_site_fract_x'] += unit_cell_dimen[0] / 4
            file.loc[:, '_atom_site_fract_y'] += unit_cell_dimen[1] / 4
            file.loc[:, '_atom_site_fract_z'] += unit_cell_dimen[2] / 4

        else:
            raise ValueError(
                "edge_padding must be a tuple of length 3."
                "Example: (1, 1, 2) will double the _atom_site_fract_z, "
                "adding 0.5 of _atom_site_fract_z to each side.")

    # print(unit_cell_dimen)

    file.loc[-1] = ['', unit_cell_dimen[0],
                    unit_cell_dimen[1],
                    unit_cell_dimen[2], '', '']
    file.index = file.index + 1  # shifts from last to first
    file.sort_index(inplace=True)

    # add header line
    header = header_comment
    file.loc[-1] = [header, '', '', '', '', '']
    file.index = file.index + 1  # shifts from last to first
    file.sort_index(inplace=True)

    # add -1 to end file
    file.loc[file.shape[0]] = [int(-1), '', '', '', '', '']

    if save is True:

        if '.xyz' not in prismatic_xyz_filename:
            file.to_csv(prismatic_xyz_filename + '.xyz',
                        sep=' ', header=False, index=False)
        else:
            file.to_csv(prismatic_xyz_filename,
                        sep=' ', header=False, index=False)

    return file


# cif writing
def write_cif_from_dataframe(dataframe,
                             filename,
                             chemical_name_common,
                             cell_length_a,
                             cell_length_b,
                             cell_length_c,
                             cell_angle_alpha=90,
                             cell_angle_beta=90,
                             cell_angle_gamma=90,
                             space_group_name_H_M_alt='P 1',
                             space_group_IT_number=1):
    """
    Write a cif file from a Pandas Dataframe. This Dataframe can be created
    with temul.model_creation.create_dataframe_for_cif().

    Parameters
    ----------
    dataframe : dataframe object
        pandas dataframe containing rows of atomic position information
    chemical_name_common : string
        name of chemical
    cell_length_a, _cell_length_b, _cell_length_c : float
        lattice dimensions in angstrom
    cell_angle_alpha, cell_angle_beta, cell_angle_gamma : float
        lattice angles in degrees
    space_group_name_H-M_alt : string
        space group name
    space_group_IT_number : float

    """

    # create cif
    cif_file = CifFile.CifFile()

    # create block to hold values
    params = CifFile.CifBlock()

    cif_file['block_1'] = params

    # set unit cell properties
    params.AddItem('_chemical_name_common', chemical_name_common)
    params.AddItem('_cell_length_a', format(cell_length_a, '.6f'))
    params.AddItem('_cell_length_b', format(cell_length_b, '.6f'))
    params.AddItem('_cell_length_c', format(cell_length_c, '.6f'))
    params.AddItem('_cell_angle_alpha', cell_angle_alpha)
    params.AddItem('_cell_angle_beta', cell_angle_beta)
    params.AddItem('_cell_angle_gamma', cell_angle_gamma)
    params.AddItem('_space_group_name_H-M_alt', space_group_name_H_M_alt)
    params.AddItem('_space_group_IT_number', space_group_IT_number)

    # loop 1 - _space_group_symop_operation_xyz
    params.AddCifItem(([['_space_group_symop_operation_xyz']],

                       [[['x, y, z']]]))

    # [[['x, y, z',
    # 'x, y, z+1/2']]]))

    # loop 2 - individual atom positions and properties
    params.AddCifItem(([['_atom_site_label',
                         '_atom_site_occupancy',
                         '_atom_site_fract_x',
                         '_atom_site_fract_y',
                         '_atom_site_fract_z',
                         '_atom_site_adp_type',
                         '_atom_site_B_iso_or_equiv',
                         '_atom_site_type_symbol']],

                       [[dataframe['_atom_site_label'],
                         dataframe['_atom_site_occupancy'],
                         dataframe['_atom_site_fract_x'],
                         dataframe['_atom_site_fract_y'],
                         dataframe['_atom_site_fract_z'],
                         dataframe['_atom_site_adp_type'],
                         dataframe['_atom_site_B_iso_or_equiv'],
                         dataframe['_atom_site_type_symbol']]]))

    # put it all together in a cif
    outFile = open(filename + ".cif", "w")
    outFile.write(str(cif_file))
    outFile.close()


# create dataframe function for single atom lattice for .xyz
def create_dataframe_for_xyz(sublattice_list,
                             element_list,
                             x_size,
                             y_size,
                             z_size,
                             filename,
                             header_comment='top_level_comment'):
    """
    Creates a Pandas Dataframe and a .xyz file (usable with Prismatic) from the
    inputted sublattice(s).

    Parameters
    ----------
    sublattice_list : list of Atomap Sublattice objects
    element_list : list of strings
        Each string must be an element symbol from the periodic table.
    x_size, y_size, z_size : floats
        Dimensions of the x,y,z axes in Angstrom.
    filename : string
        Name with which the .xyz file will be saved.
    header_comment : string, default 'top_level_comment'

    Example
    -------
    >>> import temul.external.atomap_devel_012.dummy_data as dummy_data
    >>> sublattice = dummy_data.get_simple_cubic_sublattice()
    >>> for i in range(0, len(sublattice.atom_list)):
    ...     sublattice.atom_list[i].elements = 'Mo_1'
    ...     sublattice.atom_list[i].z_height = '0.5'
    >>> element_list = ['Mo_0', 'Mo_1', 'Mo_2']
    >>> x_size, y_size = 50, 50
    >>> z_size = 5
    >>> dataframe = create_dataframe_for_xyz([sublattice], element_list,
    ...                          x_size, y_size, z_size,
    ...                          filename='dataframe',
    ...                          header_comment='Here is an Example')

    """
    df_xyz = pd.DataFrame(columns=['_atom_site_Z_number',
                                   '_atom_site_fract_x',
                                   '_atom_site_fract_y',
                                   '_atom_site_fract_z',
                                   '_atom_site_occupancy',
                                   '_atom_site_RMS_thermal_vib'])

    # add header sentence
    df_xyz = df_xyz.append({'_atom_site_Z_number': header_comment,
                            '_atom_site_fract_x': '',
                            '_atom_site_fract_y': '',
                            '_atom_site_fract_z': '',
                            '_atom_site_occupancy': '',
                            '_atom_site_RMS_thermal_vib': ''},
                           ignore_index=True)

    # add unit cell dimensions
    df_xyz = df_xyz.append({'_atom_site_Z_number': '',
                            '_atom_site_fract_x': format(x_size, '.6f'),
                            '_atom_site_fract_y': format(y_size, '.6f'),
                            '_atom_site_fract_z': format(z_size, '.6f'),
                            '_atom_site_occupancy': '',
                            '_atom_site_RMS_thermal_vib': ''},
                           ignore_index=True)

    for sublattice in sublattice_list:
        # denomiator could also be: sublattice.signal.axes_manager[0].size

        for i in range(0, len(sublattice.atom_list)):
            if sublattice.atom_list[i].elements in element_list:
                # value = 0
                # this loop cycles through the length of the split element eg,
                # 2 for 'Se_1.S_1' and
                #   outputs an atom label for each
                for k in range(0, len(split_and_sort_element(
                        sublattice.atom_list[i].elements))):
                    if split_and_sort_element(
                            sublattice.atom_list[i].elements)[k][2] >= 1:
                        atomic_number = split_and_sort_element(
                            sublattice.atom_list[i].elements)[k][3]

                        if "," in sublattice.atom_list[i].z_height:
                            atom_z_height = float(
                                sublattice.atom_list[i].z_height.split(",")[
                                    k])
                        else:
                            atom_z_height = float(
                                sublattice.atom_list[i].z_height)

                        # this loop controls the  z_height
                        # len(sublattice.atom_list[i].z_height)):
                        for p in range(0, split_and_sort_element(
                                sublattice.atom_list[i].elements)[k][2]):
                            # could use ' ' + value to get an extra space
                            # between columns!
                            # nans could be better than ''
                            # (len(sublattice.image)-

                            if "," in sublattice.atom_list[
                                    i].z_height and split_and_sort_element(
                                    sublattice.atom_list[i].elements)[
                                    k][2] > 1:
                                atom_z_height = float(
                                    sublattice.atom_list[
                                        i].z_height.split(",")[p])
                            else:
                                pass

                            df_xyz = df_xyz.append(
                                {'_atom_site_Z_number': atomic_number,
                                 '_atom_site_fract_x': format(
                                     sublattice.atom_list[i].pixel_x * (
                                         x_size / len(sublattice.image[
                                             0, :])), '.6f'),
                                 '_atom_site_fract_y': format(
                                     sublattice.atom_list[i].pixel_y * (
                                         y_size / len(sublattice.image[
                                             :, 0])), '.6f'),
                                 # this is a fraction already, which is why we
                                 # don't divide as in x and y
                                 '_atom_site_fract_z': format(
                                     atom_z_height * z_size, '.6f'),
                                 '_atom_site_occupancy': 1.0,
                                 '_atom_site_RMS_thermal_vib': 0.1},
                                ignore_index=True)  # insert row

    df_xyz = df_xyz.append({'_atom_site_Z_number': int(-1),
                            '_atom_site_fract_x': '',
                            '_atom_site_fract_y': '',
                            '_atom_site_fract_z': '',
                            '_atom_site_occupancy': '',
                            '_atom_site_RMS_thermal_vib': ''},
                           ignore_index=True)

    if filename is not None:
        df_xyz.to_csv(filename + '.xyz', sep=' ', header=False, index=False)

    return(df_xyz)


def dm3_stack_to_tiff_stack(loading_file,
                            loading_file_extension='.dm3',
                            saving_file_extension='.tif',
                            crop=False,
                            crop_start=20.0,
                            crop_end=80.0):
    '''
    Save an image stack filetype to a different filetype, e.g., dm3 to tiff.

    Parameters
    ----------
    filename : string
        Name of the image stack file
    loading_file_extension : string
        file extension of the filename
    saving_file_extension : string
        file extension you wish to save as
    crop : bool, default False
        if True, the image will be cropped in the navigation space,
        defined by the frames given in crop_start and crop_end
    crop_start, crop_end : float, default 20.0, 80.0
        the start and end frame of the crop

    '''
    if '.' in loading_file:
        file = loading_file
        filename = loading_file[:-4]
    else:
        file = loading_file + loading_file_extension
        filename = loading_file

    s = hs.load(file)

    if crop is True:
        s = s.inav[crop_start:crop_end]

        # In the form: '20.0:80.0'

    # Save the dm3 file as a tiff stack. Allows us to use below
    # analysis without editing!
    saving_file = filename + saving_file_extension
    s.save(saving_file)
    # These two lines normalize the hyperspy loaded file. Do Not
    # so if you are also normalizing below
    # stack.change_dtype('float')
    # stack.data /= stack.data.max()


def save_individual_images_from_image_stack(
        image_stack, output_folder='individual_images'):
    '''
    Save each image in an image stack. The images are saved in a new folder.
    Useful for after running an image series through Rigid Registration.

    Parameters
    ----------
    image_stack : rigid registration image stack object
    output_folder : string
        Name of the folder in which all individual images from
        the stack will be saved.

    '''
    # Save each image as a 32 bit tiff )cqn be displayed in DM
    image_stack_32bit = np.float32(image_stack)
    folder = os.mkdir(output_folder)
    # Find the number of images, change to an integer for the loop.
    for i in range(int(image_stack_32bit[0, 0, :].shape[0])):
        im = image_stack_32bit[:, :, i]
        imwrite(os.path.join(folder, f'images_aligned_{i:04}.tif'), im)


def load_prismatic_mrc_with_hyperspy(
        prismatic_mrc_filename,
        save_name='calibrated_data_'):
    '''
    We are aware this is currently producing errors with new versions of
    Prismatic.

    Open a prismatic .mrc file and save as a hyperspy object.
    Also plots saves a png.

    Parameters
    ----------
    prismatic_mrc_filename : string
        name of the outputted prismatic .mrc file.

    Returns
    -------
    Hyperspy Signal 2D

    Examples
    --------

    >>> from temul.io import load_prismatic_mrc_with_hyperspy
    >>> load_prismatic_mrc_with_hyperspy("temul/example_data/prismatic/"
    ...         "prism_2Doutput_prismatic_simulation.mrc")
    <Signal2D, title: , dimensions: (|1182, 773)>

    '''

    # if '.mrc' not in prismatic_mrc_filename:
    #     prismatic_mrc_filename = prismatic_mrc_filename + 'mrc'
    # if 'prism_2Doutput_' not in prismatic_mrc_filename:
    #     prismatic_mrc_filename = 'prism_2Doutput' + prismatic_mrc_filename

    simulation = hs.load(prismatic_mrc_filename)
    simulation.axes_manager[0].name = 'extra_dimension'
    simulation = simulation.sum('extra_dimension')

    if save_name is not None:
        simulation.save(save_name, overwrite=True)
        simulation.plot()
        plt.title(save_name, fontsize=20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname=save_name + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)
        # plt.close()

    return simulation
