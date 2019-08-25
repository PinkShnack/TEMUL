# author: Eoghan O'Connell

import hyperspy.api as hs
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

    Example
    -------

    >>> batch_convert_emd_to_image(extension_to_save='.png',
    ...         top_level_directory='G:/Titan Images/08-10-19_MHEOC_SampleImaging stem',
    ...         glob_search="**/*",
    ...         overwrite=True)

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
