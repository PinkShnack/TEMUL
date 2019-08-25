# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 21:19:13 2019

@author: Eoghan O'Connell, Michael Hennessy
"""

from atomap.atom_finding_refining import _make_circular_mask
from matplotlib import gridspec
import rigidregistration
from tifffile import imread, imwrite, TiffWriter
from collections import Counter
import warnings
from time import time
from pyprismatic.fileio import readMRC
import pyprismatic as pr
from glob import glob
from atomap.atom_finding_refining import normalize_signal
from atomap.tools import remove_atoms_from_image_using_2d_gaussian
import os
from skimage.measure import compare_ssim as ssm
# from atomap.atom_finding_refining import get_atom_positions_in_difference_image
from scipy.ndimage.filters import gaussian_filter
import collections
from atomap.atom_finding_refining import subtract_average_background
from numpy import mean
import matplotlib.pyplot as plt
import hyperspy.api as hs
import atomap.api as am
import numpy as np
from numpy import log
import CifFile
import pandas as pd
import scipy
import periodictable as pt
import matplotlib
# matplotlib.use('Agg')

'''
Remove the local background from a sublattice intensity using
a background sublattice. 

Parameters
----------

Returns
-------

Examples
--------
>>> 

'''

# need to add "radius" for where to get intensity from. Do we though?
#   until we find a way of defining it in the image, radius should be left alone. Radius can be accessed in the
#   periodictable package anyway.
# need to add remove backgroun locally or with a third sublattice
#        sublattice.find_nearest_neighbors()


def get_sublattice_intensity(sublattice, intensity_type='max', remove_background_method=None,
                             background_sublattice=None, num_points=3, percent_to_nn=0.4, mask_radius=None):
    '''
    Finds the intensity for each atomic column using either max, mean, 
    min, total or all of them at once.

    The intensity values are taken from the area defined by 
    percent_to_nn.

    Results are stored in each Atom_Position object as 
    amplitude_max_intensity, amplitude_mean_intensity, 
    amplitude_min_intensity and/or amplitude_total_intensity 
    which can most easily be accessed through the sublattice object. 
    See the examples in get_atom_column_amplitude_max_intensity.

    Parameters
    ----------

    sublattice : sublattice object
        The sublattice whose intensities you are finding.
    intensity_type : string, default 'max'
        Determines the method used to find the sublattice intensities.
        The available methods are 'max', 'mean', 'min', 'total' and
        'all'. 
    remove_background_method : string, default None
        Determines the method used to remove the background_sublattice
        intensities from the image. Options are 'average' and 'local'.
    background_sublattice : sublattice object, default None
        The sublattice used if remove_background_method is used.
    num_points : int, default 3
        If remove_background_method='local', num_points is the number 
        of nearest neighbour values averaged from background_sublattice
    percent_to_nn : float, default 0.40
        Determines the boundary of the area surrounding each atomic 
        column, as fraction of the distance to the nearest neighbour.

    Returns
    -------
    2D numpy array

    Examples
    --------

    >>> import numpy as np
    >>> import atomap.api as am
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.find_nearest_neighbors()
    >>> intensities_all = get_sublattice_intensity(sublattice=sublattice, 
                                                   intensity_type='all', 
                                                   remove_background_method=None,
                                                   background_sublattice=None)

    >>> intensities_total = get_sublattice_intensity(sublattice=sublattice, 
                                                   intensity_type='total', 
                                                   remove_background_method=None,
                                                   background_sublattice=None)

    >>> intensities_total_local = get_sublattice_intensity(sublattice=sublattice, 
                                                   intensity_type='total', 
                                                   remove_background_method='local',
                                                   background_sublattice=sublattice)

    >>> intensities_max_average = get_sublattice_intensity(sublattice=sublattice, 
                                                   intensity_type='max', 
                                                   remove_background_method='average',
                                                   background_sublattice=sublattice)

    '''
    if percent_to_nn is not None:
        sublattice.find_nearest_neighbors()
    else:
        pass

    if remove_background_method == None and background_sublattice == None:
        if intensity_type == 'all':
            sublattice.get_atom_column_amplitude_max_intensity(
                percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_max_intensity_list = []
            sublattice_max_intensity_list.append(
                sublattice.atom_amplitude_max_intensity)
            sublattice_max_intensity_list = sublattice_max_intensity_list[0]
            max_intensities = np.array(sublattice_max_intensity_list)

            sublattice.get_atom_column_amplitude_mean_intensity(
                percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_mean_intensity_list = []
            sublattice_mean_intensity_list.append(
                sublattice.atom_amplitude_mean_intensity)
            sublattice_mean_intensity_list = sublattice_mean_intensity_list[0]
            mean_intensities = np.array(sublattice_mean_intensity_list)

            sublattice.get_atom_column_amplitude_min_intensity(
                percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_min_intensity_list = []
            sublattice_min_intensity_list.append(
                sublattice.atom_amplitude_min_intensity)
            sublattice_min_intensity_list = sublattice_min_intensity_list[0]
            min_intensities = np.array(sublattice_min_intensity_list)

            sublattice.get_atom_column_amplitude_total_intensity(
                percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_total_intensity_list = []
            sublattice_total_intensity_list.append(
                sublattice.atom_amplitude_total_intensity)
            sublattice_total_intensity_list = sublattice_total_intensity_list[0]
            total_intensities = np.array(sublattice_total_intensity_list)


#            sublattice_total_intensity_list, _, _ = sublattice.integrate_column_intensity() # maxradius should be changed to percent_to_nn!
#            total_intensities = np.array(sublattice_total_intensity_list)

            sublattice_intensities = np.column_stack(
                (max_intensities, mean_intensities, min_intensities, total_intensities))
            return(sublattice_intensities)
          #  return max_intensities, mean_intensities, min_intensities, total_intensities

        elif intensity_type == 'max':
            sublattice.get_atom_column_amplitude_max_intensity(
                percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_max_intensity_list = []
            sublattice_max_intensity_list.append(
                sublattice.atom_amplitude_max_intensity)
            sublattice_max_intensity_list = sublattice_max_intensity_list[0]
            max_intensities = np.array(sublattice_max_intensity_list)

            return(max_intensities)

        elif intensity_type == 'mean':
            sublattice.get_atom_column_amplitude_mean_intensity(
                percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_mean_intensity_list = []
            sublattice_mean_intensity_list.append(
                sublattice.atom_amplitude_mean_intensity)
            sublattice_mean_intensity_list = sublattice_mean_intensity_list[0]
            mean_intensities = np.array(sublattice_mean_intensity_list)

            return(mean_intensities)

        elif intensity_type == 'min':
            sublattice.get_atom_column_amplitude_min_intensity(
                percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_min_intensity_list = []
            sublattice_min_intensity_list.append(
                sublattice.atom_amplitude_min_intensity)
            sublattice_min_intensity_list = sublattice_min_intensity_list[0]
            min_intensities = np.array(sublattice_min_intensity_list)

            return(min_intensities)

        elif intensity_type == 'total':
            sublattice.get_atom_column_amplitude_total_intensity(
                percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_total_intensity_list = []
            sublattice_total_intensity_list.append(
                sublattice.atom_amplitude_total_intensity)
            sublattice_total_intensity_list = sublattice_total_intensity_list[0]
            total_intensities = np.array(sublattice_total_intensity_list)

#            sublattice_total_intensity_list, _, _ = sublattice.integrate_column_intensity()
#            total_intensities = np.array(sublattice_total_intensity_list)
            return(total_intensities)

        else:
            raise ValueError('You must choose an intensity_type')

    elif remove_background_method == 'average':

        sublattice_intensity_list_average_bksubtracted = remove_average_background(sublattice=sublattice,
                                                                                   background_sublattice=background_sublattice,
                                                                                   intensity_type=intensity_type,
                                                                                   percent_to_nn=percent_to_nn,
                                                                                   mask_radius=mask_radius)
        return(sublattice_intensity_list_average_bksubtracted)

    elif remove_background_method == 'local':

        sublattice_intensity_list_local_bksubtracted = remove_local_background(sublattice=sublattice,
                                                                               background_sublattice=background_sublattice,
                                                                               intensity_type=intensity_type,
                                                                               num_points=num_points,
                                                                               percent_to_nn=percent_to_nn,
                                                                               mask_radius=mask_radius)
        return(sublattice_intensity_list_local_bksubtracted)

    else:
        pass


def get_pixel_count_from_image_slice(
        self,
        image_data,
        percent_to_nn=0.40):
    """
    Fid the number of pixels in an area when calling
    _get_image_slice_around_atom()

    Parameters
    ----------

    image_data : Numpy 2D array
    percent_to_nn : float, default 0.40
        Determines the boundary of the area surrounding each atomic 
        column, as fraction of the distance to the nearest neighbour.

    Returns
    -------
    The number of pixels in the image_slice

    Examples
    --------

    >>> import atomap.api as am
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.find_nearest_neighbors()
    >>> atom0 = sublattice.atom_list[0]
    >>> pixel_count = atom0.get_pixel_count_from_image_slice(sublattice.image)

    """
    closest_neighbor = self.get_closest_neighbor()

    slice_size = closest_neighbor * percent_to_nn * 2
    # data_slice, x0, y0 - see atomap documentation
    data_slice, _, _ = self._get_image_slice_around_atom(
        image_data, slice_size)

    pixel_count = len(data_slice[0]) * len(data_slice[0])

    return(pixel_count)


def remove_average_background(sublattice, intensity_type,
                              background_sublattice, percent_to_nn=0.40,
                              mask_radius=None):
    '''
    Remove the average background from a sublattice intensity using
    a background sublattice. 

    Parameters
    ----------

    sublattice : sublattice object
        The sublattice whose intensities are of interest.
    intensity_type : string
        Determines the method used to find the sublattice intensities.
        The available methods are 'max', 'mean', 'min' and 'all'. 
    background_sublattice : sublattice object
        The sublattice used to find the average background.
    percent_to_nn : float, default 0.4
        Determines the boundary of the area surrounding each atomic 
        column, as fraction of the distance to the nearest neighbour.

    Returns
    -------
    2D numpy array

    Examples
    --------

    >>> import numpy as np
    >>> import atomap.api as am
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.find_nearest_neighbors()
    >>> intensities_all = remove_average_background(sublattice, intensity_type='all',
                                                    background_sublattice=sublattice)
    >>> intensities_max = remove_average_background(sublattice, intensity_type='max',
                                                background_sublattice=sublattice)

    '''
    background_sublattice.find_nearest_neighbors()
    background_sublattice.get_atom_column_amplitude_min_intensity(
        percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    background_sublattice_min = []
    background_sublattice_min.append(
        background_sublattice.atom_amplitude_min_intensity)
    background_sublattice_mean_of_min = np.mean(background_sublattice_min)

    if intensity_type == 'all':
        sublattice.get_atom_column_amplitude_max_intensity(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_max_intensity_list = []
        sublattice_max_intensity_list.append(
            sublattice.atom_amplitude_max_intensity)
        sublattice_max_intensity_list = sublattice_max_intensity_list[0]
        max_intensities = np.array(
            sublattice_max_intensity_list) - background_sublattice_mean_of_min

        sublattice.get_atom_column_amplitude_mean_intensity(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_mean_intensity_list = []
        sublattice_mean_intensity_list.append(
            sublattice.atom_amplitude_mean_intensity)
        sublattice_mean_intensity_list = sublattice_mean_intensity_list[0]
        mean_intensities = np.array(
            sublattice_mean_intensity_list) - background_sublattice_mean_of_min

        sublattice.get_atom_column_amplitude_min_intensity(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_min_intensity_list = []
        sublattice_min_intensity_list.append(
            sublattice.atom_amplitude_min_intensity)
        sublattice_min_intensity_list = sublattice_min_intensity_list[0]
        min_intensities = np.array(
            sublattice_min_intensity_list) - background_sublattice_mean_of_min

#        sublattice.get_atom_column_amplitude_total_intensity(percent_to_nn=percent_to_nn)
#        sublattice_total_intensity_list = []
#        sublattice_total_intensity_list.append(sublattice.atom_amplitude_total_intensity)
#        sublattice_total_intensity_list = sublattice_total_intensity_list[0]
#        total_intensities = np.array(sublattice_total_intensity_list) - background_sublattice_mean_of_min

        sublattice_intensities = np.column_stack(
            (max_intensities, mean_intensities, min_intensities))
        return sublattice_intensities

    elif intensity_type == 'max':
        sublattice.get_atom_column_amplitude_max_intensity(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_max_intensity_list = []
        sublattice_max_intensity_list.append(
            sublattice.atom_amplitude_max_intensity)
        sublattice_max_intensity_list = sublattice_max_intensity_list[0]
        max_intensities = np.array(
            sublattice_max_intensity_list) - background_sublattice_mean_of_min

        return max_intensities

    elif intensity_type == 'mean':
        sublattice.get_atom_column_amplitude_mean_intensity(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_mean_intensity_list = []
        sublattice_mean_intensity_list.append(
            sublattice.atom_amplitude_mean_intensity)
        sublattice_mean_intensity_list = sublattice_mean_intensity_list[0]
        mean_intensities = np.array(
            sublattice_mean_intensity_list) - background_sublattice_mean_of_min

        return mean_intensities

    elif intensity_type == 'min':
        sublattice.get_atom_column_amplitude_min_intensity(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_min_intensity_list = []
        sublattice_min_intensity_list.append(
            sublattice.atom_amplitude_min_intensity)
        sublattice_min_intensity_list = sublattice_min_intensity_list[0]
        min_intensities = np.array(
            sublattice_min_intensity_list) - background_sublattice_mean_of_min

        return min_intensities

    elif intensity_type == 'total':
        raise ValueError(
            "Average background removal doesn't work with total intensity, yet")
#        sublattice.get_atom_column_amplitude_total_intensity(percent_to_nn=percent_to_nn)
#        sublattice_total_intensity_list = []
#        sublattice_total_intensity_list.append(sublattice.atom_amplitude_total_intensity)
#        sublattice_total_intensity_list = sublattice_total_intensity_list[0]
#        total_intensities = np.array(sublattice_total_intensity_list) - background_sublattice_mean_of_min
#
#        return total_intensities

    else:
        pass

#
#sublattice0 = am.dummy_data.get_simple_cubic_sublattice()
#
# inten = get_sublattice_intensity(sublattice=sublattice0, intensity_type='max', remove_background_method='local',
#                         background_sublattice=sublattice0, num_points=3, percent_to_nn=0.3)


# can make the mean/mode option better:
#   code blocks aren't needed, just put the if statement lower down where the change is...


def remove_local_background(sublattice, background_sublattice, intensity_type,
                            num_points=3, percent_to_nn=0.40, mask_radius=None):
    '''
    Remove the local background from a sublattice intensity using
    a background sublattice. 

    Parameters
    ----------

    sublattice : sublattice object
        The sublattice whose intensities are of interest.
    intensity_type : string
        Determines the method used to find the sublattice intensities.
        The available methods are 'max', 'mean', 'min', 'total' and 'all'. 
    background_sublattice : sublattice object
        The sublattice used to find the local backgrounds.
    num_points : int, default 3
        The number of nearest neighbour values averaged from 
        background_sublattice
    percent_to_nn : float, default 0.40
        Determines the boundary of the area surrounding each atomic 
        column, as fraction of the distance to the nearest neighbour.

    Returns
    -------
    2D numpy array

    Examples
    --------

    >>> import numpy as np
    >>> import atomap.api as am
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.find_nearest_neighbors()
    >>> intensities_total = remove_local_background(sublattice, intensity_type='total',
                                                    background_sublattice=sublattice)
    >>> intensities_max = remove_local_background(sublattice, intensity_type='max',
                                                  background_sublattice=sublattice)

    '''
    # get background_sublattice intensity list

    if percent_to_nn is not None:
        sublattice.find_nearest_neighbors()
        background_sublattice.find_nearest_neighbors()
    else:
        pass

    background_sublattice.get_atom_column_amplitude_min_intensity(
        percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    background_sublattice_min_intensity_list = []
    background_sublattice_min_intensity_list.append(
        background_sublattice.atom_amplitude_min_intensity)
    background_sublattice_min_intensity_list = background_sublattice_min_intensity_list[0]
    if intensity_type == 'all':
        raise ValueError(
            "All intensities has not yet been implemented. Use max, mean or total instead")

    if num_points == 0:
        raise ValueError(
            "num_points cannot be 0 if you wish to locally remove the background")

    if intensity_type == 'max':
        # get list of sublattice and background_sublattice atom positions
        # np.array().T will not be needed in newer versions of atomap
        sublattice_atom_pos = np.array(sublattice.atom_positions).T
        background_sublattice_atom_pos = np.array(
            background_sublattice.atom_positions).T

        # get sublattice intensity list
        # could change to my function, which allows choice of intensity type
        sublattice.get_atom_column_amplitude_max_intensity(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_max_intensity_list = []
        sublattice_max_intensity_list.append(
            sublattice.atom_amplitude_max_intensity)
        sublattice_max_intensity_list = sublattice_max_intensity_list[0]

        # create list which will be output
        # therefore the original data is not changed!
        sublattice_max_intensity_list_bksubtracted = []

        # for each sublattice atom position, calculate the nearest
        #   background_sublattice atom positions.
        for p in range(0, len(sublattice_atom_pos)):

            xy_distances = background_sublattice_atom_pos - \
                sublattice_atom_pos[p]

            # put all distances in this array with this loop
            vector_array = []
            for i in range(0, len(xy_distances)):
                # get distance from sublattice position to every background_sublattice position
                vector = np.sqrt(
                    (xy_distances[i][0]**2) + (xy_distances[i][1]**2))
                vector_array.append(vector)
            # convert to numpy array
            vector_array = np.array(vector_array)

            # sort through the vector_array and find the 1st to kth smallest distance and find the
            #   corressponding index
            # num_points is the number of nearest points from which the background will be averaged
            k = num_points
            min_indices = list(np.argpartition(vector_array, k)[:k])
            # sum the chosen intensities and find the mean (or median - add this)
            local_background = 0
            for index in min_indices:
                local_background += background_sublattice.atom_amplitude_min_intensity[index]

            local_background_mean = local_background / k

            # subtract this mean local background intensity from the sublattice
            #   atom position intensity
            # indexing here is the loop digit p
            sublattice_bksubtracted_atom = np.array(sublattice.atom_amplitude_max_intensity[p]) - \
                np.array(local_background_mean)

            sublattice_max_intensity_list_bksubtracted.append(
                [sublattice_bksubtracted_atom])

        sublattice_max_intensity_list_bksubtracted = np.array(
            sublattice_max_intensity_list_bksubtracted)

        return(sublattice_max_intensity_list_bksubtracted[:, 0])

    elif intensity_type == 'mean':
        # get list of sublattice and background_sublattice atom positions
        # np.array().T will not be needed in newer versions of atomap
        sublattice_atom_pos = np.array(sublattice.atom_positions).T
        background_sublattice_atom_pos = np.array(
            background_sublattice.atom_positions).T

        # get sublattice intensity list
        # could change to my function, which allows choice of intensity type
        sublattice.get_atom_column_amplitude_mean_intensity(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_mean_intensity_list = []
        sublattice_mean_intensity_list.append(
            sublattice.atom_amplitude_mean_intensity)
        sublattice_mean_intensity_list = sublattice_mean_intensity_list[0]

        # create list which will be output
        # therefore the original data is not changed!
        sublattice_mean_intensity_list_bksubtracted = []

        # for each sublattice atom position, calculate the nearest
        #   background_sublattice atom positions.
        for p in range(0, len(sublattice_atom_pos)):

            xy_distances = background_sublattice_atom_pos - \
                sublattice_atom_pos[p]

            # put all distances in this array with this loop
            vector_array = []
            for i in range(0, len(xy_distances)):
                # get distance from sublattice position to every background_sublattice position
                vector = np.sqrt(
                    (xy_distances[i][0]**2) + (xy_distances[i][1]**2))
                vector_array.append(vector)
            # convert to numpy array
            vector_array = np.array(vector_array)

            # sort through the vector_array and find the 1st to kth smallest distance and find the
            #   corressponding index
            # num_points is the number of nearest points from which the background will be averaged
            k = num_points
            min_indices = list(np.argpartition(vector_array, k)[:k])
            # sum the chosen intensities and find the mean (or median - add this)
            local_background = 0
            for index in min_indices:
                local_background += background_sublattice.atom_amplitude_min_intensity[index]

            local_background_mean = local_background / k

            # subtract this mean local background intensity from the sublattice
            #   atom position intensity
            # indexing here is the loop digit p
            sublattice_bksubtracted_atom = np.array(sublattice.atom_amplitude_mean_intensity[p]) - \
                local_background_mean

            sublattice_mean_intensity_list_bksubtracted.append(
                [sublattice_bksubtracted_atom])

        sublattice_mean_intensity_list_bksubtracted = np.array(
            sublattice_mean_intensity_list_bksubtracted)

        return(sublattice_mean_intensity_list_bksubtracted[:, 0])

    elif intensity_type == 'total':
        # get list of sublattice and background_sublattice atom positions
        # np.array().T will not be needed in newer versions of atomap
        sublattice_atom_pos = np.array(sublattice.atom_positions).T
        background_sublattice_atom_pos = np.array(
            background_sublattice.atom_positions).T

        # get sublattice intensity list
        # could change to my function, which allows choice of intensity type
        sublattice.get_atom_column_amplitude_total_intensity(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_total_intensity_list = []
        sublattice_total_intensity_list.append(
            sublattice.atom_amplitude_total_intensity)
        sublattice_total_intensity_list = sublattice_total_intensity_list[0]

        # create list which will be output
        # therefore the original data is not changed!
        sublattice_total_intensity_list_bksubtracted = []

        # for each sublattice atom position, calculate the nearest
        #   background_sublattice atom positions.
        for p in range(0, len(sublattice_atom_pos)):

            xy_distances = background_sublattice_atom_pos - \
                sublattice_atom_pos[p]

            # put all distances in this array with this loop
            vector_array = []
            for i in range(0, len(xy_distances)):
                # get distance from sublattice position to every background_sublattice position
                vector = np.sqrt(
                    (xy_distances[i][0]**2) + (xy_distances[i][1]**2))
                vector_array.append(vector)
            # convert to numpy array
            vector_array = np.array(vector_array)

            # sort through the vector_array and find the 1st to kth smallest distance and find the
            #   corressponding index
            # num_points is the number of nearest points from which the background will be averaged
            k = num_points
            min_indices = list(np.argpartition(vector_array, range(k))[:k])
            # if you want the values rather than the indices, use:
            # vector_array[np.argpartition(vector_array, range(k))[:k]]
            # sum the chosen intensities and find the total (or median - add this)
            local_background = 0
            for index in min_indices:
                local_background += background_sublattice.atom_amplitude_min_intensity[index]

            local_background_mean = local_background / k

            # for summing pixels around atom
            if mask_radius is None:
                pixel_count_in_region = get_pixel_count_from_image_slice(sublattice.atom_list[p],
                                                                         sublattice.image,
                                                                         percent_to_nn)
            elif mask_radius is not None:
                mask = _make_circular_mask(centerX=sublattice.atom_list[p].pixel_x,
                                           centerY=sublattice.atom_list[p].pixel_y,
                                           imageSizeX=sublattice.image.shape[0],
                                           imageSizeY=sublattice.image.shape[1],
                                           radius=mask_radius)

                pixel_count_in_region = len(sublattice.image[mask])

            local_background_mean_summed = pixel_count_in_region * local_background_mean

            # subtract this mean local background intensity from the sublattice
            #   atom position intensity
            # indexing here is the loop digit p
            sublattice_bksubtracted_atom = np.array(sublattice.atom_amplitude_total_intensity[p]) - \
                local_background_mean_summed

            sublattice_total_intensity_list_bksubtracted.append(
                [sublattice_bksubtracted_atom])

        sublattice_total_intensity_list_bksubtracted = np.array(
            sublattice_total_intensity_list_bksubtracted)

        return(sublattice_total_intensity_list_bksubtracted[:, 0])

    else:
        raise ValueError(
            "You must choose a valid intensity_type. Try max, mean or total")


# split_symbol must be a list
# splitting an element

def split_and_sort_element(element, split_symbol=['_', '.']):
    '''
    Extracts info from input atomic column element configuration
    Split an element and its count, then sort the element for 
    use with other functions.

    Parameters
    ----------

    element : string, default None
        element species and count must be separated by the
        first string in the split_symbol list.
        separate elements must be separated by the second 
        string in the split_symbol list.
    split_symbol : list of strings, default ['_', '.']
        The symbols used to split the element into its name
        and count.
        The first string '_' is used to split the name and count 
        of the element.
        The second string is used to split different elements in
        an atomic column configuration.

    Returns
    -------
    list with element_split, element_name, element_count, and
    element_atomic_number

    Examples
    --------
    >>> import periodictable as pt
    >>> single_element = split_and_sort_element(element='S_1')
    >>> complex_element = split_and_sort_element(element='O_6.Mo_3.Ti_5')

    '''
    splitting_info = []

    if '.' in element:
        # if len(split_symbol) > 1:

        if split_symbol[1] == '.':

            stacking_element = element.split(split_symbol[1])
            for i in range(0, len(stacking_element)):
                element_split = stacking_element[i].split(split_symbol[0])
                element_name = element_split[0]
                element_count = int(element_split[1])
                element_atomic_number = pt.elements.symbol(element_name).number
                splitting_info.append(
                    [element_split, element_name, element_count, element_atomic_number])
        else:
            raise ValueError(
                "To split a stacked element use: split_symbol=['_', '.']")

    # elif len(split_symbol) == 1:
    elif '.' not in element:
        element_split = element.split(split_symbol[0])
        element_name = element_split[0]
        element_count = int(element_split[1])
        element_atomic_number = pt.elements.symbol(element_name).number
        splitting_info.append(
            [element_split, element_name, element_count, element_atomic_number])

    else:
        raise ValueError(
            "You must include a split_symbol. Use '_' to separate element and count. Use '.' to separate elements in the same xy position")

    return(splitting_info)


# scaling method
# Limited to single elements at the moment. Need to figure out maths to expand it to more.
def scaling_z_contrast(numerator_sublattice, numerator_element,
                       denominator_sublattice, denominator_element,
                       intensity_type, method, remove_background_method,
                       background_sublattice, num_points,
                       percent_to_nn=0.40, mask_radius=None, split_symbol='_'):
    # Make sure that the intensity_type input has been chosen. Could
    #   make this more flexible, so that 'all' could be calculated in one go
    #   simple loop should do that.
    if intensity_type == 'all':
        TypeError
        print('intensity_type must be "max", "mean", or "min"')
    else:
        pass

    sublattice0 = numerator_sublattice
    sublattice1 = denominator_sublattice

    # use the get_sublattice_intensity() function to get the mean/mode intensities of
    #   each sublattice
    if type(mask_radius) is list:
        sublattice0_intensity = get_sublattice_intensity(
            sublattice0, intensity_type, remove_background_method, background_sublattice,
            num_points, percent_to_nn=percent_to_nn, mask_radius=mask_radius[0])

        sublattice1_intensity = get_sublattice_intensity(
            sublattice1, intensity_type, remove_background_method, background_sublattice,
            num_points, percent_to_nn=percent_to_nn, mask_radius=mask_radius[1])
    else:
        sublattice0_intensity = get_sublattice_intensity(
            sublattice0, intensity_type, remove_background_method, background_sublattice,
            num_points, percent_to_nn=percent_to_nn, mask_radius=mask_radius)

        sublattice1_intensity = get_sublattice_intensity(
            sublattice1, intensity_type, remove_background_method, background_sublattice,
            num_points, percent_to_nn=percent_to_nn, mask_radius=mask_radius)

    if method == 'mean':
        sublattice0_intensity_method = np.mean(sublattice0_intensity)
        sublattice1_intensity_method = np.mean(sublattice1_intensity)
    elif method == 'mode':
        sublattice0_intensity_method = scipy.stats.mode(
            np.round(sublattice0_intensity, decimals=2))[0][0]
        sublattice1_intensity_method = scipy.stats.mode(
            np.round(sublattice1_intensity, decimals=2))[0][0]

    # Calculate the scaling ratio and exponent for Z-contrast images
    scaling_ratio = sublattice0_intensity_method / sublattice1_intensity_method

    numerator_element_split = split_and_sort_element(
        element=numerator_element, split_symbol=split_symbol)
    denominator_element_split = split_and_sort_element(
        element=denominator_element, split_symbol=split_symbol)

    if len(numerator_element_split) == 1:
        scaling_exponent = log(denominator_element_split[0][2]*scaling_ratio) / (
            log(numerator_element_split[0][3]) - log(denominator_element_split[0][3]))
    else:
        pass  # need to include more complicated equation to deal with multiple elements as the e.g., numerator

    return scaling_ratio, scaling_exponent, sublattice0_intensity_method, sublattice1_intensity_method


def auto_generate_sublattice_element_list(material_type,
                                          elements='Au',
                                          max_number_atoms_z=10):
    """
    Example
    -------

    >>> element_list = auto_generate_sublattice_element_list(
    ...                     material_type='nanoparticle',
    ...                     elements='Au',
    ...                     max_number_atoms_z=10)

    """

    element_list = []

    if material_type == 'TMD_pristine':
        pass
    elif material_type == 'TMD_modified':
        pass
    elif material_type == 'nanoparticle':
        if isinstance(elements, str):

            for i in range(0, max_number_atoms_z+1):
                element_list.append(elements + '_' + str(i))

        elif isinstance(elements, list):
            pass

    return(element_list)


'''
# manipulating the adatoms. Issue here is that if we just look for
# #  vacancies, Se_1 and Mo_1, then we'd just get more Se_1.
# #  need to find a way of "locking in" those limits..

# Calculate the middle point and limits of the distribution for a given element_list.
# Need to add Mike's histogram display
'''


def find_middle_and_edge_intensities(sublattice,
                                     element_list,
                                     standard_element,
                                     scaling_exponent,
                                     split_symbol=['_', '.']):
    """
    Create a list which represents the peak points of the
    intensity distribution for each atom.

    works for nanoparticles as well, doesn't matter what 
    scaling_exponent you use for nanoparticle. Figure this out!
    """

    middle_intensity_list = []
    limit_intensity_list = [0.0]

    if isinstance(standard_element, str) == True:
        standard_split = split_and_sort_element(
            element=standard_element, split_symbol=split_symbol)
        standard_element_value = 0.0
        for i in range(0, len(standard_split)):
            standard_element_value += standard_split[i][2] * \
                (pow(standard_split[i][3], scaling_exponent))
    else:
        standard_element_value = standard_element
    # find the values for element_lists
    for i in range(0, len(element_list)):
        element_split = split_and_sort_element(
            element=element_list[i], split_symbol=split_symbol)
        element_value = 0.0
        for p in range(0, len(element_split)):
            element_value += element_split[p][2] * \
                (pow(element_split[p][3], scaling_exponent))
        atom = element_value / standard_element_value
        middle_intensity_list.append(atom)

    middle_intensity_list.sort()

    for i in range(0, len(middle_intensity_list)-1):
        limit = (middle_intensity_list[i] + middle_intensity_list[i+1])/2
        limit_intensity_list.append(limit)

    if len(limit_intensity_list) <= len(middle_intensity_list):
        max_limit = middle_intensity_list[-1] + \
            (middle_intensity_list[-1]-limit_intensity_list[-1])
        limit_intensity_list.append(max_limit)
    else:
        pass

    return middle_intensity_list, limit_intensity_list


# choosing the percent_to_nn for this seems dodgy atm...
def find_middle_and_edge_intensities_for_background(elements_from_sub1,
                                                    elements_from_sub2,
                                                    sub1_mode,
                                                    sub2_mode,
                                                    element_list_sub1,
                                                    element_list_sub2,
                                                    middle_intensity_list_sub1,
                                                    middle_intensity_list_sub2):

    middle_intensity_list_background = [0.0]

    # it is neccessary to scale the background_sublattice intensities here already because otherwise
    #   the background_sublattice has no reference atom to base its mode intensity on. eg. in MoS2, first sub has Mo
    #   as a standard atom, second sub has S2 as a standard reference.

    for i in elements_from_sub1:
        index = element_list_sub1.index(i)
        middle = middle_intensity_list_sub1[index] * sub1_mode
        middle_intensity_list_background.append(middle)

    for i in elements_from_sub2:
        index = element_list_sub2.index(i)
        middle = middle_intensity_list_sub2[index] * sub2_mode
        middle_intensity_list_background.append(middle)

    middle_intensity_list_background.sort()

    limit_intensity_list_background = [0.0]
    for i in range(0, len(middle_intensity_list_background)-1):
        limit = (
            middle_intensity_list_background[i] + middle_intensity_list_background[i+1])/2
        limit_intensity_list_background.append(limit)

    if len(limit_intensity_list_background) <= len(middle_intensity_list_background):
        max_limit = middle_intensity_list_background[-1] + (
            middle_intensity_list_background[-1]-limit_intensity_list_background[-1])
        limit_intensity_list_background.append(max_limit)
    else:
        pass

    return middle_intensity_list_background, limit_intensity_list_background


#
#
#sub2_ints = get_sublattice_intensity(sub2, intensity_type='max', remove_background_method=None)
#
# min(sub2_ints)
# sub2_ints.sort()
#
#sub2_mode = scipy.stats.mode(np.round(sub2_ints, decimals=2))[0][0]
#
#limit_numbers = []
# for i in limit_intensity_list_sub2:
#    limit_numbers.append(i*sub2_mode)
#
#
# elements_of_sub2 = sort_sublattice_intensities(sub2, 'max', middle_intensity_list_sub2,
#                                               limit_intensity_list_sub2, element_list_sub2,
#                                               method='mode', remove_background_method=None,
#                                               percent_to_nn=0.2)
#
#sublattice = sub2
# intensity_type='max'
# middle_intensity_list=middle_intensity_list_sub2
# limit_intensity_list=limit_intensity_list_sub2
# element_list=element_list_sub2
# method='mode'
# remove_background=None
# percent_to_nn=0.2

# Place each atom intensity into their element variations according
# to the middle and limit points

def sort_sublattice_intensities(sublattice,
                                intensity_type=None,
                                middle_intensity_list=None,
                                limit_intensity_list=None,
                                element_list=[],
                                scalar_method='mode',
                                remove_background_method=None,
                                background_sublattice=None,
                                num_points=3,
                                intensity_list_real=False,
                                percent_to_nn=0.40, mask_radius=None):

    # intensity_list_real is asking whether the intensity values in your intensity_list for the current sublattice
    #   are scaled. Scaled meaning already multiplied by the mean or mode of said sublattice.
    #   Set to Tru for background sublattices. For more details see "find_middle_and_edge_intensities_for_background()"
    #   You can see that the outputted lists are scaled by the mean or mode, whereas in
    #   "find_middle_and_edge_intensities()", they are not.

    # For testing and quickly assigning a sublattice some elements.
    if middle_intensity_list is None:
        elements_of_sublattice = []
        for i in range(0, len(sublattice.atom_list)):
            sublattice.atom_list[i].elements = element_list[0]
            elements_of_sublattice.append(sublattice.atom_list[i].elements)

    else:
        sublattice_intensity = get_sublattice_intensity(
            sublattice, intensity_type, remove_background_method, background_sublattice,
            num_points, percent_to_nn=percent_to_nn, mask_radius=mask_radius)

        for i in sublattice_intensity:
            if i < 0:
                i = 0.0000000001
                #raise ValueError("You have negative intensity. Bad Vibes")

        if intensity_list_real == False:

            if scalar_method == 'mean':
                scalar = np.mean(sublattice_intensity)
            elif scalar_method == 'mode':
                scalar = scipy.stats.mode(
                    np.round(sublattice_intensity, decimals=2))[0][0]
            elif isinstance(scalar_method, (int, float)):
                scalar = scalar_method

            if len(element_list) != len(middle_intensity_list):
                raise ValueError(
                    'elements_list length does not equal middle_intensity_list length')
            else:
                pass

            elements_of_sublattice = []
            for p in range(0, (len(limit_intensity_list)-1)):
                for i in range(0, len(sublattice.atom_list)):
                    if limit_intensity_list[p]*scalar < sublattice_intensity[i] < limit_intensity_list[p+1]*scalar:
                        sublattice.atom_list[i].elements = element_list[p]
                        elements_of_sublattice.append(
                            sublattice.atom_list[i].elements)

        elif intensity_list_real == True:
            if len(element_list) != len(middle_intensity_list):
                raise ValueError(
                    'elements_list length does not equal middle_intensity_list length')
            else:
                pass

            elements_of_sublattice = []
            for p in range(0, (len(limit_intensity_list)-1)):
                for i in range(0, len(sublattice.atom_list)):
                    if limit_intensity_list[p] < sublattice_intensity[i] < limit_intensity_list[p+1]:
                        sublattice.atom_list[i].elements = element_list[p]
                        elements_of_sublattice.append(
                            sublattice.atom_list[i].elements)

        for i in range(0, len(sublattice.atom_list)):
            if sublattice.atom_list[i].elements == '':
                sublattice.atom_list[i].elements = 'H_0'
                elements_of_sublattice.append(sublattice.atom_list[i].elements)
            else:
                pass

    return(elements_of_sublattice)


# sublattice=sub2
# for i in range(0, len(sublattice.atom_list)):
#    if sublattice.atom_list[i].elements == 'S_0':
#        print(i)
#
#sublattice.atom_list[36].elements = 'S_1'
# i=36
#

#whatareyou = split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2]
#
# if whatareyou == 0:
#    print('arg')
# else:
#    print('nope')


# if chalcogen = True, give positions as...
   #   currently "chalcogen" is relevant to our TMDC work

def assign_z_height(sublattice, lattice_type, material):
    for i in range(0, len(sublattice.atom_list)):
        if material == 'mose2_one_layer':
            if lattice_type == 'chalcogen':
                if len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1:
                    sublattice.atom_list[i].z_height = '0.758'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 2:
                    sublattice.atom_list[i].z_height = '0.242, 0.758'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 2:
                    sublattice.atom_list[i].z_height = '0.242, 0.758, 0.9'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[1][2] == 1:
                    sublattice.atom_list[i].z_height = '0.242, 0.758'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 1:
                    sublattice.atom_list[i].z_height = '0.242, 0.758, 0.9'
                else:
                    sublattice.atom_list[i].z_height = '0.758'
                    #raise ValueError("z_height is limited to only a handful of positions")
            elif lattice_type == 'transition_metal':
                if len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1:
                    sublattice.atom_list[i].z_height = '0.5'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 2:
                    sublattice.atom_list[i].z_height = '0.5, 0.95'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 2:
                    sublattice.atom_list[i].z_height = '0.5, 0.95, 1'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[1][2] == 1:
                    sublattice.atom_list[i].z_height = '0.5, 0.95'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 1:
                    sublattice.atom_list[i].z_height = '0.5, 0.95, 1'
                else:
                    sublattice.atom_list[i].z_height = '0.5'
                    #raise ValueError("z_height is limited to only a handful of positions")
            elif lattice_type == 'background':
                # if sublattice.atom_list[i].elements == 'H_0' or sublattice.atom_list[i].elements == 'vacancy':
                sublattice.atom_list[i].z_height = '0.95'
                # elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1:
                #    sublattice.atom_list[i].z_height = [0.9]
                # else:
                #    sublattice.atom_list[i].z_height = []

            else:
                print(
                    "You must include a suitable lattice_type. This feature is limited")

        if material == 'mos2_one_layer':
            if lattice_type == 'chalcogen':
                if len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1:
                    # from L Mattheis, PRB, 1973
                    sublattice.atom_list[i].z_height = '0.757'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 2:
                    sublattice.atom_list[i].z_height = '0.242, 0.757'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 2:
                    sublattice.atom_list[i].z_height = '0.242, 0.757, 0.95'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[1][2] == 1:
                    sublattice.atom_list[i].z_height = '0.242, 0.757'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 1:
                    sublattice.atom_list[i].z_height = '0.242, 0.757, 0.95'
                else:
                    sublattice.atom_list[i].z_height = '0.757'
                    #raise ValueError("z_height is limited to only a handful of positions")
            elif lattice_type == 'transition_metal':
                if len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1:
                    sublattice.atom_list[i].z_height = '0.5'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 2:
                    sublattice.atom_list[i].z_height = '0.5, 0.95'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 2:
                    sublattice.atom_list[i].z_height = '0.5, 0.95, 1'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[1][2] == 1:
                    sublattice.atom_list[i].z_height = '0.5, 0.95'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 1:
                    sublattice.atom_list[i].z_height = '0.5, 0.95, 1'
                else:
                    sublattice.atom_list[i].z_height = '0.5'
                    #raise ValueError("z_height is limited to only a handful of positions")
            elif lattice_type == 'background':
                # if sublattice.atom_list[i].elements == 'H_0' or sublattice.atom_list[i].elements == 'vacancy':
                sublattice.atom_list[i].z_height = '0.95'
                # elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1:
                #    sublattice.atom_list[i].z_height = [0.9]
                # else:
                #    sublattice.atom_list[i].z_height = []

            else:
                print(
                    "You must include a suitable lattice_type. This feature is limited")

        if material == 'mos2_two_layer':  # from L Mattheis, PRB, 1973
            if lattice_type == 'TM_top':
                if len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 3:
                    sublattice.atom_list[i].z_height = '0.1275, 0.3725, 0.75'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2:
                    sublattice.atom_list[i].z_height = '0.1275, 0.3725, 0.75'
                else:
                    sublattice.atom_list[i].z_height = '0.95'
                    #raise ValueError("z_height is limited to only a handful of positions")
            elif lattice_type == 'TM_bot':
                if len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 3:
                    sublattice.atom_list[i].z_height = '0.25, 0.6275, 0.8725'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2:
                    sublattice.atom_list[i].z_height = '0.25, 0.6275, 0.8725'
                else:
                    sublattice.atom_list[i].z_height = '0.95'
                    #raise ValueError("z_height is limited to only a handful of positions")
            elif lattice_type == 'background':
                # if sublattice.atom_list[i].elements == 'H_0' or sublattice.atom_list[i].elements == 'vacancy':
                sublattice.atom_list[i].z_height = '0.95'
                # elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1:
                #    sublattice.atom_list[i].z_height = [0.9]
                # else:
                #    sublattice.atom_list[i].z_height = []

            else:
                print(
                    "You must include a suitable lattice_type. This feature is limited")
# MoS2 0.2429640475, 0.7570359525


def print_sublattice_elements(sublattice):
    elements_of_sublattice = []
    for i in range(0, len(sublattice.atom_list)):
        sublattice.atom_list[i].elements
        sublattice.atom_list[i].z_height  # etc.
        elements_of_sublattice.append([sublattice.atom_list[i].elements,
                                       sublattice.atom_list[i].z_height,
                                       sublattice.atom_amplitude_max_intensity[i],
                                       sublattice.atom_amplitude_mean_intensity[i],
                                       sublattice.atom_amplitude_min_intensity[i],
                                       sublattice.atom_amplitude_total_intensity[i]
                                       ])
    return elements_of_sublattice


#
# scaling_ratio, scaling_exponent = scaling_z_contrast(numerator_sublattice=sub1, numerator_element='Mo_1',
#                                                     denominator_sublattice=sub2, denominator_element='S_2',
#                                                     intensity_type='mean', method='mode')


# needs to be able to deal with eg: Mo_1.S_1.Se_1

# create dataframe function for atom lattice for cif

def create_dataframe_for_cif(sublattice_list, element_list):
    """
    Parameters
    ----------

    """
    dfObj = pd.DataFrame(columns=['_atom_site_label',
                                  '_atom_site_occupancy',
                                  '_atom_site_fract_x',
                                  '_atom_site_fract_y',
                                  '_atom_site_fract_z',
                                  '_atom_site_adp_type',
                                  '_atom_site_B_iso_or_equiv',
                                  '_atom_site_type_symbol'])

    # Start with the first sublattice in the list of sublattices given
    for sublattice in sublattice_list:
            # Go through each atom_list index one by one
        for i in range(0, len(sublattice.atom_list)):
                # check if the element is in the given element list
            if sublattice.atom_list[i].elements in element_list:
                    # this loop cycles through the length of the split element eg, 2 for 'Se_1.S_1' and
                    #   outputs an atom label and z_height for each
                for k in range(0, len(split_and_sort_element(sublattice.atom_list[i].elements))):
                    if split_and_sort_element(sublattice.atom_list[i].elements)[k][2] >= 1:
                        atom_label = split_and_sort_element(
                            sublattice.atom_list[i].elements)[k][1]

                        if "," in sublattice.atom_list[i].z_height:
                            atom_z_height = float(
                                sublattice.atom_list[i].z_height.split(",")[k])
                        else:
                            atom_z_height = float(
                                sublattice.atom_list[i].z_height)

                        # this loop checks the number of atoms that share
                        # the same x and y coords.
                        # len(sublattice.atom_list[i].z_height)):
                        for p in range(0, split_and_sort_element(sublattice.atom_list[i].elements)[k][2]):

                            if "," in sublattice.atom_list[i].z_height and split_and_sort_element(sublattice.atom_list[i].elements)[k][2] > 1:
                                atom_z_height = float(
                                    sublattice.atom_list[i].z_height.split(",")[p])
                            else:
                                pass

                            dfObj = dfObj.append({'_atom_site_label': atom_label,
                                                  '_atom_site_occupancy': 1.0,
                                                  '_atom_site_fract_x': format(sublattice.atom_list[i].pixel_x/len(sublattice.image[0, :]), '.6f'),
                                                  '_atom_site_fract_y': format((len(sublattice.image[:, 0])-sublattice.atom_list[i].pixel_y)/len(sublattice.image[:, 0]), '.6f'),
                                                  # great touch
                                                  '_atom_site_fract_z': format(atom_z_height, '.6f'),
                                                  '_atom_site_adp_type': 'Biso',
                                                  '_atom_site_B_iso_or_equiv': format(1.0, '.6f'),
                                                  '_atom_site_type_symbol': atom_label},
                                                 ignore_index=True)  # insert row

                            #value += split_and_sort_element(sublattice.atom_list[i].elements)[k][2]
    # need an option to save to the cuurent directory should be easy
#        dfObj.to_pickle('atom_lattice_atom_position_table.pkl')
#        dfObj.to_csv('atom_lattice_atom_position_table.csv', sep=',', index=False)
    return dfObj

#element_list = ['S_0', 'S_1', 'S_2', 'S_2.C_1', 'S_2.C_2', 'Mo_1', 'Mo_0']
#example_df = create_dataframe_for_cif(atom_lattice, element_list)

# '_atom_site_fract_z' : format( (sublattice.atom_list[i].z_height)[p+(k*k)], '.6f'), #great touch


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
    outFile = open(filename+".cif", "w")
    outFile.write(str(cif_file))
    outFile.close()


# write_cif_from_dataframe(dataframe=example_df,
#                         filename='simulation_Se_1.S_1',
#                         chemical_name_common='MoS2_sim',
#                         cell_length_a=30,
#                         cell_length_b=30,
#                         cell_length_c=6.3)


# sublattice=sub2
# i=14
# k=1
# p=1
#sublattice_list = [sub1]


# create dataframe function for single atom lattice for .xyz
def create_dataframe_for_xyz(sublattice_list,
                             element_list,
                             x_distance,
                             y_distance,
                             z_distance,
                             filename,
                             header_comment='top_level_comment'):
    """
    Parameters
    ----------

    Example
    -------

    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> for i in range(0, len(sublattice.atom_list)):
            sublattice.atom_list[i].elements = 'Mo_1'
            sublattice.atom_list[i].z_height = '0.5'
    >>> element_list = ['Mo_0', 'Mo_1', 'Mo_2']
    >>> x_distance, y_distance = 50, 50
    >>> z_distance = 5
    >>> dataframe = create_dataframe_for_xyz([sublattice], element_list,
                                 x_distance, y_distance, z_distance,
                                 save='dataframe',
                                 header_comment='Here is an Example')

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
                            '_atom_site_fract_x': format(x_distance, '.6f'),
                            '_atom_site_fract_y': format(y_distance, '.6f'),
                            '_atom_site_fract_z': format(z_distance, '.6f'),
                            '_atom_site_occupancy': '',
                            '_atom_site_RMS_thermal_vib': ''},
                           ignore_index=True)

    for sublattice in sublattice_list:
        # denomiator could also be: sublattice.signal.axes_manager[0].size

        for i in range(0, len(sublattice.atom_list)):
            if sublattice.atom_list[i].elements in element_list:
                #value = 0
                # this loop cycles through the length of the split element eg, 2 for 'Se_1.S_1' and
                #   outputs an atom label for each
                for k in range(0, len(split_and_sort_element(sublattice.atom_list[i].elements))):
                    if split_and_sort_element(sublattice.atom_list[i].elements)[k][2] >= 1:
                        atomic_number = split_and_sort_element(
                            sublattice.atom_list[i].elements)[k][3]

                        if "," in sublattice.atom_list[i].z_height:
                            atom_z_height = float(
                                sublattice.atom_list[i].z_height.split(",")[k])
                        else:
                            atom_z_height = float(
                                sublattice.atom_list[i].z_height)

                        # this loop controls the  z_height
                        # len(sublattice.atom_list[i].z_height)):
                        for p in range(0, split_and_sort_element(sublattice.atom_list[i].elements)[k][2]):
                            # could use ' ' + value to get an extra space between columns!
                            # nans could be better than ''
                            # (len(sublattice.image)-

                            if "," in sublattice.atom_list[i].z_height and split_and_sort_element(sublattice.atom_list[i].elements)[k][2] > 1:
                                atom_z_height = float(
                                    sublattice.atom_list[i].z_height.split(",")[p])
                            else:
                                pass

                            df_xyz = df_xyz.append({'_atom_site_Z_number': atomic_number,
                                                    '_atom_site_fract_x': format(sublattice.atom_list[i].pixel_x * (x_distance / len(sublattice.image[0, :])), '.6f'),
                                                    '_atom_site_fract_y': format(sublattice.atom_list[i].pixel_y * (y_distance / len(sublattice.image[:, 0])), '.6f'),
                                                    # this is a fraction already, which is why we don't divide as in x and y
                                                    '_atom_site_fract_z': format(atom_z_height * z_distance, '.6f'),
                                                    '_atom_site_occupancy': 1.0,  # might need to loop through the vancancies here?
                                                    '_atom_site_RMS_thermal_vib': 0.05},
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

#element_list = ['S_0', 'S_1', 'S_2', 'S_2.C_1', 'S_2.C_2', 'Mo_1', 'Mo_0']
#example_df = create_dataframe_for_cif(atom_lattice, element_list)


'''
old Z_height with lists

def assign_z_height(sublattice, lattice_type, material):
    for i in range(0, len(sublattice.atom_list)):
        if material == 'mose2_one_layer':
            if lattice_type == 'chalcogen':
                if len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1:
                    sublattice.atom_list[i].z_height = [0.758]
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 2:
                    sublattice.atom_list[i].z_height = [0.242, 0.758]
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 2:
                    sublattice.atom_list[i].z_height = [0.242, 0.758, 0.9]
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[1][2] == 1:
                    sublattice.atom_list[i].z_height = [0.242, 0.758]
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 1:
                    sublattice.atom_list[i].z_height = [0.242, 0.758, 0.9]
                else:
                    sublattice.atom_list[i].z_height = [0.758]
                    #raise ValueError("z_height is limited to only a handful of positions")
            elif lattice_type == 'transition_metal':
                if len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1:
                    sublattice.atom_list[i].z_height = [0.5]
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 2:
                    sublattice.atom_list[i].z_height = [0.5, 0.95]
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 2:
                    sublattice.atom_list[i].z_height = [0.5, 0.95, 1]
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[1][2] == 1:
                    sublattice.atom_list[i].z_height = [0.5, 0.95]
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 1:
                    sublattice.atom_list[i].z_height = [0.5, 0.95, 1]
                else:
                    sublattice.atom_list[i].z_height = [0.5]
                    #raise ValueError("z_height is limited to only a handful of positions")
            elif lattice_type == 'background':
                #if sublattice.atom_list[i].elements == 'H_0' or sublattice.atom_list[i].elements == 'vacancy':
                sublattice.atom_list[i].z_height = [0.95]
                #elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1:
                #    sublattice.atom_list[i].z_height = [0.9]
                #else:
                #    sublattice.atom_list[i].z_height = []
                
            else:
                print("You must include a suitable lattice_type. This feature is limited")
                
                
        if material == 'mos2_one_layer':
            if lattice_type == 'chalcogen':
                if len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1:
                    sublattice.atom_list[i].z_height = [0.757] # from L Mattheis, PRB, 1973
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 2:
                    sublattice.atom_list[i].z_height = [0.242, 0.757]
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 2:
                    sublattice.atom_list[i].z_height = [0.242, 0.757, 0.95]
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[1][2] == 1:
                    sublattice.atom_list[i].z_height = [0.242, 0.757]
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 1:
                    sublattice.atom_list[i].z_height = [0.242, 0.757, 0.95]
                else:
                    sublattice.atom_list[i].z_height = [0.757]
                    #raise ValueError("z_height is limited to only a handful of positions")
            elif lattice_type == 'transition_metal':
                if len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1:
                    sublattice.atom_list[i].z_height = [0.5]
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 2:
                    sublattice.atom_list[i].z_height = [0.5, 0.95]
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 2:
                    sublattice.atom_list[i].z_height = [0.5, 0.95, 1]
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[1][2] == 1:
                    sublattice.atom_list[i].z_height = [0.5, 0.95]
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 1:
                    sublattice.atom_list[i].z_height = [0.5, 0.95, 1]
                else:
                    sublattice.atom_list[i].z_height = [0.5]
                    #raise ValueError("z_height is limited to only a handful of positions")
            elif lattice_type == 'background':
                #if sublattice.atom_list[i].elements == 'H_0' or sublattice.atom_list[i].elements == 'vacancy':
                sublattice.atom_list[i].z_height = [0.95]
                #elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1:
                #    sublattice.atom_list[i].z_height = [0.9]
                #else:
                #    sublattice.atom_list[i].z_height = []
                
            else:
                print("You must include a suitable lattice_type. This feature is limited")

# MoS2 0.2429640475, 0.7570359525   
'''


# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:21:53 2019

@author: eoghan.oconnell
"""


'''Load the Dataset'''


def load_data_and_sampling(filename, file_extension=None, invert_image=False, save_image=True):

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
        print('WARNING: Image calibrated to pixels, you should calibrate to distance')
    elif s.axes_manager[-1].scale != 1:
        real_sampling = s.axes_manager[-1].scale
        s.axes_manager[-1].units = 'nm'
        s.axes_manager[-2].units = 'nm'

    # real_sampling =
#    physical_image_size = real_sampling * len(s.data)
    save_name = filename[:-4]

    if invert_image == True:
        s.data = np.divide(1, s.data)

        if save_image == True:

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
        if save_image == True:
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


def get_and_return_element(element_symbol):
    '''
    From the elemental symbol, e.g., 'H' for Hydrogen, returns Hydrogen as
    a periodictable.core.Element object for further use.

    Parameters
    ----------

    element_symbol : string, default None
        Symbol of an element from the periodic table of elements

    Returns
    -------
    A periodictable.core.Element object

    Examples
    --------
    >>> Moly = get_and_return_element(element_symbol='Mo')
    >>> print(Moly.covalent_radius)
    >>> print(Moly.symbol)
    >>> print(Moly.number)

    '''

    for pt_element in pt.elements:
        if pt_element.symbol == element_symbol:
            chosen_element = pt_element

    return(chosen_element)


def atomic_radii_in_pixels(sampling, element_symbol):
    '''
    Get the atomic radius of an element in pixels, scaled by an image sampling

    Parameters
    ----------
    sampling : float, default None
        sampling of an image in units of nm/pix
    element_symbol : string, default None
        Symbol of an element from the periodic table of elements

    Returns
    -------
    Half the colavent radius of the input element in pixels

    Examples
    --------

    >>> import atomap.api as am
    >>> image = am.dummy_data.get_simple_cubic_signal()
    >>> # pretend it is a 5x5 nm image
    >>> image_sampling = 5/len(image.data) # units nm/pix
    >>> radius_pix_Mo = atomic_radii_in_pixels(image_sampling, 'Mo')
    >>> radius_pix_S = atomic_radii_in_pixels(image_sampling, 'S')

    '''

    element = get_and_return_element(element_symbol=element_symbol)

    # mult by 0.5 to get correct distance (google image of covalent radius)
    # divided by 10 to get nm
    radius_nm = (element.covalent_radius*0.5)/10

    radius_pix = radius_nm/sampling

    return(radius_pix)


def toggle_atom_refine_position_automatically(sublattice,
                                              filename,
                                              min_cut_off_percent,
                                              max_cut_off_percent,
                                              range_type='internal',
                                              method='mode',
                                              percent_to_nn=0.05,
                                              mask_radius=None):
    '''
    Sets the 'refine_position' attribute of each Atom Position in a 
    sublattice using a range of intensities.

    Parameters
    ----------
    sublattice : Atomap Sublattice object, default None

    min_cut_off_percent : float, default None
        The lower end of the intensity range is defined as
        min_cut_off_percent * modal value of max intensity list of
        sublattice.
    max_cut_off_percent : float, default None
        The upper end of the intensity range is defined as
        max_cut_off_percent * modal value of max intensity list of
        sublattice.
    range_type : string, default 'internal'
        'internal' returns the 'refine_position' attribute for each
        Atom Position as True if the intensity of that Atom Position
        lies between the lower and upper limits defined by min_cut_off_percent
        and max_cut_off_percent.
        'external' returns the 'refine_position' attribute for each
        Atom Position as True if the intensity of that Atom Position
        lies outside the lower and upper limits defined by min_cut_off_percent
        and max_cut_off_percent.
    save_image : Bool, default False
        Save the 'sublattice.toggle_atom_refine_position_with_gui()'
        image.
    percent_to_nn : float, default 0.40
        Determines the boundary of the area surrounding each atomic 
        column, as fraction of the distance to the nearest neighbour.

    Returns
    -------
    calibrated image data
    Example
    -------

    >>> min_cut_off_percent = 0.75
    >>> max_cut_off_percent = 1.25
    >>> sublattice = am.dummy_data.get_simple_cubic_with_vacancies_sublattice(image_noise=True)
    >>> sublattice.find_nearest_neighbors()
    >>> sublattice.plot()
    >>> false_list_sublattice =  toggle_atom_refine_position_automatically(
                                    sublattice=sublattice,
                                    min_cut_off_percent=min_cut_off_percent,
                                    max_cut_off_percent=max_cut_off_percent, 
                                    range_type='internal',
                                    method='mode',
                                    save_image=False,
                                    percent_to_nn=0.05)

    >>> # Check which atoms will not be refined (red dots)
    >>> sublattice.toggle_atom_refine_position_with_gui()
    '''

    sublattice.get_atom_column_amplitude_max_intensity(
        percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    sublattice_vacancy_check_list = sublattice.atom_amplitude_max_intensity

    if method == 'mode':
        sublattice_scalar = scipy.stats.mode(
            np.round(a=sublattice_vacancy_check_list, decimals=2))[0]
    elif method == 'mean':
        sublattice_scalar = np.mean(sublattice_vacancy_check_list)

    sublattice_min_cut_off = min_cut_off_percent*sublattice_scalar
    sublattice_max_cut_off = max_cut_off_percent*sublattice_scalar

    if range_type == 'internal':

        for i in range(0, len(sublattice.atom_list)):
            if sublattice_min_cut_off < sublattice.atom_amplitude_max_intensity[i] < sublattice_max_cut_off:
                sublattice.atom_list[i].refine_position = True
            else:
                sublattice.atom_list[i].refine_position = False

    elif range_type == 'external':
        for i in range(0, len(sublattice.atom_list)):
            if sublattice.atom_amplitude_max_intensity[i] > sublattice_max_cut_off or sublattice_min_cut_off > sublattice.atom_amplitude_max_intensity[i]:
                sublattice.atom_list[i].refine_position = True
            else:
                sublattice.atom_list[i].refine_position = False

    else:
        raise TypeError(
            "'internal' and 'external' are the only options for range_type")

    # checking we have some falses
    false_list_sublattice = []
    for i in range(0, len(sublattice.atom_list)):
        if sublattice.atom_list[i].refine_position == False:
            false_list_sublattice.append(
                sublattice.atom_list[i].refine_position)

    if len(false_list_sublattice) == 0:
        print("false_list_sublattice is empty")

    if filename is not None:
        sublattice.toggle_atom_refine_position_with_gui()
        plt.title('Toggle Atom Refine ' + sublattice.name +
                  ' ' + filename + '\n Red=False', fontsize=20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname='toggle_atom_refine_' + sublattice.name + '_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)
        plt.close()

    return(false_list_sublattice)


#cropping_area = am.add_atoms_with_gui(image.data)


def calibrate_intensity_distance_with_sublattice_roi(image,
                                                     cropping_area,
                                                     separation,
                                                     filename=None,
                                                     reference_image=None,
                                                     percent_to_nn=0.2,
                                                     mask_radius=None,
                                                     refine=True,
                                                     scalebar_true=False):  # add max mean min etc.
    ''' 
    Calibrates the intensity of an image by using a sublattice, found with some
    atomap functions. The mean intensity of that sublattice is set to 1

    Parameters
    ----------
    image : HyperSpy 2D signal, default None
        The signal can be distance calibrated. If it is, set
        scalebar_true=True
    cropping_area : list of 2 floats, default None
        The best method of choosing the area is by using the atomap
        function "add_atoms_with_gui(image.data)". Choose two points on the 
        image. First point is top left of area, second point is bottom right.
    percent_to_nn : float, default 0.40
        Determines the boundary of the area surrounding each atomic 
        column, as fraction of the distance to the nearest neighbour.
    scalebar_true : Bool, default False
        Set to True if the scale of the image is calibrated to a distance unit.

    Returns
    -------
    calibrated image data

    Example
    -------

    >>> image = am.dummy_data.get_simple_cubic_with_vacancies_signal()
    >>> image.plot()
    >>> cropping_area = am.add_atoms_with_gui(image.data) # choose two points
    >>> calibrate_intensity_distance_with_sublattice_roi(image, cropping_area)
    >>> image.plot()

    '''
    llim, tlim = cropping_area[0]
    rlim, blim = cropping_area[1]

    if image.axes_manager[0].scale != image.axes_manager[1].scale:
        raise ValueError("x & y scales don't match!")

    if scalebar_true == True:
        llim *= image.axes_manager[0].scale
        tlim *= image.axes_manager[0].scale
        rlim *= image.axes_manager[0].scale
        blim *= image.axes_manager[0].scale
    else:
        pass

    cal_area = hs.roi.RectangularROI(
        left=llim, right=rlim, top=tlim, bottom=blim)(image)
    atom_positions = am.get_atom_positions(
        cal_area, separation=separation, pca=True)
    #atom_positions = am.add_atoms_with_gui(cal_area, atom_positions)
    calib_sub = am.Sublattice(atom_positions, cal_area, color='r')
#    calib_sub.plot()
    if refine == True:
        calib_sub.find_nearest_neighbors()
        calib_sub.refine_atom_positions_using_center_of_mass(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        calib_sub.refine_atom_positions_using_2d_gaussian(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    else:
        pass
    # calib_sub.plot()
    calib_sub.get_atom_column_amplitude_max_intensity(
        percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    calib_sub_max_list = calib_sub.atom_amplitude_max_intensity
    calib_sub_scalar = mean(a=calib_sub_max_list)
    image.data = image.data/calib_sub_scalar

    if reference_image is not None:
        image.axes_manager = reference_image.axes_manager

    if filename is not None:
        save_name = 'calibrated_data_'
        image.save(save_name + filename, overwrite=True)
        image.plot()
        plt.title(save_name + filename, fontsize=20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname=save_name + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)
        # plt.close()


'''Image Cropping (optional!)'''

# Reload image if you need to
#s = hs.load('Filtered Image.hspy')


# cropping done in the scale, so nm, pixel, or whatever you have

def crop_image_hs(image, cropping_area, save_image=True, save_variables=True,
                  scalebar_true=True):
    '''
    Example
    -------

    >>> image = am.dummy_data.get_simple_cubic_with_vacancies_signal()
    >>> image.plot()
    >>> cropping_area = am.add_atoms_with_gui(image.data) # choose two points
    '''

    llim, tlim = cropping_area[0]
    rlim, blim = cropping_area[1]

    unit = image.axes_manager[0].units
#    image_name = image.metadata.General.original_filename

    if image.axes_manager[0].scale != image.axes_manager[1].scale:
        raise ValueError("x & y scales don't match!")

    if scalebar_true == True:
        llim *= image.axes_manager[0].scale
        tlim *= image.axes_manager[0].scale
        rlim *= image.axes_manager[0].scale
        blim *= image.axes_manager[0].scale
    else:
        pass

    roi = hs.roi.RectangularROI(left=llim, right=rlim, top=tlim, bottom=blim)
    image.plot()
    image_crop = roi.interactive(image)

    if save_image == True:
        plt.title('Cropped region highlighted', fontsize=20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname='Cropped region highlighted.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)
        plt.close()
    else:
        plt.close()

    image_crop.plot()

    image_crop
    physical_image_crop_size_x = image_crop.axes_manager[0].scale * \
        image_crop.axes_manager[0].size
    physical_image_crop_size_y = image_crop.axes_manager[1].scale * \
        image_crop.axes_manager[1].size

    if save_image == True:
        image_crop.save('Cropped Image.hspy')
        image_crop.plot()
        plt.title('Cropped Image', fontsize=20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname='Cropped Image.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)
        plt.close()
    else:
        plt.close()

    if save_variables == True:
        ''' Saving the Variables for the image and filtered Image '''
        Cropping_Variables = collections.OrderedDict()
#        Cropping_Variables['Image Name'] = [image_name]
        Cropping_Variables['left (%s)' % unit] = [llim]
        Cropping_Variables['right (%s)' % unit] = [rlim]
        Cropping_Variables['top (%s)' % unit] = [tlim]
        Cropping_Variables['bottom (%s)' % unit] = [blim]
        Cropping_Variables['physical_image_size X axis (%s)' % unit] = [
            physical_image_crop_size_x]
        Cropping_Variables['physical_image_size Y axis (%s)' % unit] = [
            physical_image_crop_size_y]
        Cropping_Variables['Unit'] = [unit]
        Cropping_Variables_Table = pd.DataFrame(Cropping_Variables)
        Cropping_Variables_Table
        Cropping_Variables_Table.to_pickle('Cropping_Variables_Table.pkl')
        Cropping_Variables_Table.to_csv(
            'Cropping_Variables_Table.csv', sep=',', index=False)

    else:
        pass

    return image_crop


def make_gaussian(size, fwhm, center):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    arr = []  # output numpy array
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    arr.append(np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2))

    return(arr)


def DG_filter(image, filename, d_inner, d_outer, delta, real_space_sampling, units='nm'):
    # Folder: G:/SuperStem visit/Feb 2019 data/2019_02_18_QMR_S1574_MoS2-Se-15eV

    # Accuracy of calculation. Smaller = more accurate.
    #   0.01 means it will fit until intensity is 0.01 away from 0
    # 0.01 is a good starting value
    #    delta=0.01

    # Find the FWHM for both positive (outer) and negative (inner) gaussians
    # d_1 is the inner reflection diameter in units of 1/nm (or whatever unit you're working with)
    # I find these in gatan, should be a way of doing automatically.
    #    d_1 = 7.48
    #    d_outer = 14.96

    # image.plot()
    #    image.save('Original Image Data', overwrite=True)
    #    image_name = image.metadata.General.original_filename

    physical_image_size = real_space_sampling * len(image.data)
    reciprocal_sampling = 1/physical_image_size

    # Get radius
    reciprocal_d_inner = (d_inner/2)
    reciprocal_d_outer = (d_outer/2)
    reciprocal_d_inner_pix = reciprocal_d_inner/reciprocal_sampling
    reciprocal_d_outer_pix = reciprocal_d_outer/reciprocal_sampling

    fwhm_neg_gaus = reciprocal_d_inner_pix
    fwhm_pos_gaus = reciprocal_d_outer_pix

    #s = normalize_signal(subtract_average_background(s))
    image.axes_manager[0].scale = real_space_sampling
    image.axes_manager[1].scale = real_space_sampling
    image.axes_manager[0].units = units
    image.axes_manager[1].units = units
    #image.save('Calibrated Image Data', overwrite=True)

#    image.plot()
#    plt.title('Calibrated Image', fontsize = 20)
#    plt.gca().axes.get_xaxis().set_visible(False)
#    plt.gca().axes.get_yaxis().set_visible(False)
#    plt.tight_layout()
#    plt.savefig(fname='Calibrated Image.png',
#                transparent=True, frameon=False, bbox_inches='tight',
#                pad_inches=None, dpi=300, labels=False)
#    plt.close()

    # Get FFT of the image
    image_fft = image.fft(shift=True)
    # image_fft.plot()

    # Get the absolute value for viewing purposes
    # image_amp = image_fft.amplitude

    # image_amp.plot(norm='log')
    '''Plot the dataset'''
    # image.plot()
    # plt.close()
    # Get the sampling of the real and reciprocal space

    # Positive Gaussian
    arr = make_gaussian(size=len(image.data), fwhm=fwhm_pos_gaus, center=None)
    nD_Gaussian = hs.signals.Signal2D(np.array(arr))
    # nD_Gaussian.plot()
    # plt.close()

    # negative gauss
    arr_neg = make_gaussian(size=len(image.data),
                            fwhm=fwhm_neg_gaus, center=None)
    # Note that this step isn't actually neccessary for the computation,
    #   we could just subtract when making the double gaussian below.
    #   However, we do it this way so that we can save a plot of the negative gaussian!
    #np_arr_neg = np_arr_neg
    nD_Gaussian_neg = hs.signals.Signal2D(np.array(arr_neg))
    # nD_Gaussian_neg.plot()

    neg_gauss_amplitude = 0.0
    int_and_gauss_array = []

    for neg_gauss_amplitude in np.arange(0, 1+delta, delta):

        # while neg_gauss_amplitude <= 1:
        nD_Gaussian_neg_scaled = nD_Gaussian_neg*-1 * \
            neg_gauss_amplitude  # NEED TO FIGURE out best number here!
        # nD_Gaussian_neg.plot()
        # plt.close()

        # Double Gaussian
        DGFilter = nD_Gaussian + nD_Gaussian_neg_scaled
        # DGFilter.plot()
        # plt.close()

        '''
        # Remove background intensity and normalize
        DGFilter = normalize_signal(subtract_average_background(DGFilter))
        DGFilter.plot()
        '''
        # Multiply the 2-D Gaussian with the FFT. This low pass filters the FFT.
        convolution = image_fft*DGFilter
        # convolution.plot(norm='log')
        #convolution_amp = convolution.amplitude
        # convolution_amp.plot(norm='log')

        # Create the inverse FFT, which is your filtered image!
        convolution_ifft = convolution.ifft()
        # convolution_ifft.plot()
        minimum_intensity = convolution_ifft.data.min()
        maximum_intensity = convolution_ifft.data.max()

        int_and_gauss_array.append(
            [neg_gauss_amplitude, minimum_intensity, maximum_intensity])

        #neg_gauss_amplitude = neg_gauss_amplitude + delta

    np_arr_2 = np.array(int_and_gauss_array)
    x_axis = np_arr_2[:, 0]
    y_axis = np_arr_2[:, 1]
    zero_line = np.zeros_like(x_axis)
    idx = np.argwhere(np.diff(np.sign(zero_line-y_axis))).flatten()
    neg_gauss_amplitude_calculated = x_axis[idx][0]

    ''' Filtering the Image with the Chosen Negative Amplitude '''
    # positive gauss
    nD_Gaussian.axes_manager[0].scale = reciprocal_sampling
    nD_Gaussian.axes_manager[1].scale = reciprocal_sampling
    nD_Gaussian.axes_manager[0].units = '1/' + units
    nD_Gaussian.axes_manager[1].units = '1/' + units

    # negative gauss
    nD_Gaussian_neg_used = nD_Gaussian_neg*-1 * \
        neg_gauss_amplitude_calculated  # NEED TO FIGURE out best number here!
    nD_Gaussian_neg_used.axes_manager[0].scale = reciprocal_sampling
    nD_Gaussian_neg_used.axes_manager[1].scale = reciprocal_sampling
    nD_Gaussian_neg_used.axes_manager[0].units = '1/' + units
    nD_Gaussian_neg_used.axes_manager[1].units = '1/' + units

    # Double Gaussian
    DGFilter_extra_dimension = nD_Gaussian + nD_Gaussian_neg_used
    DGFilter_extra_dimension.axes_manager[0].name = 'extra_dimension'

    '''how to change to just the 2 dimensiuons'''
    DGFilter = DGFilter_extra_dimension.sum('extra_dimension')

    DGFilter.axes_manager[0].scale = reciprocal_sampling
    DGFilter.axes_manager[1].scale = reciprocal_sampling
    DGFilter.axes_manager[0].units = '1/' + units
    DGFilter.axes_manager[1].units = '1/' + units

    # Multiply the 2-D Gaussian with the FFT. This filters the FFT.
    convolution = image_fft * DGFilter
    convolution_amp = convolution.amplitude

    # Create the inverse FFT, which is your filtered image!
    image_filtered = convolution.ifft()
    #s = normalize_signal(subtract_average_background(convolution_ifft))

    image_filtered.axes_manager[0].scale = real_space_sampling
    image_filtered.axes_manager[1].scale = real_space_sampling
    image_filtered.axes_manager[0].units = units
    image_filtered.axes_manager[1].units = units

    if filename is not None:
        plt.figure()
        plt.plot(x_axis, y_axis)
        plt.plot(x_axis, zero_line)
        plt.plot(x_axis[idx], y_axis[idx], 'ro')
        plt.xlabel('Negative Gaussian Amplitude', fontsize=16)
        plt.ylabel('Minimum Image Intensity', fontsize=16)
        plt.title('Finding the Best DG Filter \n NG Amp = %' +
                  filename % x_axis[idx][0], fontsize=20)
        plt.legend(labels=('Neg. Gauss. Amp.', 'y = 0',), fontsize=14)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.savefig(fname='minimising_negative_gaussian_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300)
        plt.close()

        #    if filename is not None:
        nD_Gaussian_neg_used.save(
            'negative_gaussian_' + filename, overwrite=True)
        nD_Gaussian_neg_used.plot()
        plt.title('Negative Gaussian ' + filename, fontsize=20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname='negative_gaussian_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)
        plt.close()

        nD_Gaussian.save('positive_gaussian_' + filename, overwrite=True)
        nD_Gaussian.plot()
        plt.title('Positive Gaussian ' + filename, fontsize=20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname='positive_gaussian_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)
        plt.close()

        DGFilter.save('double_gaussian_filter_' + filename,
                      overwrite=True)  # Save the .hspy file

        DGFilter.plot()
        plt.title('Double Gaussian Filter ' + filename, fontsize=20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname='double_gaussian_filter_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)
        plt.close()

        convolution_amp.plot(norm='log')
        plt.title('FFT and Filter Convolved ' + filename, fontsize=20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname='FFT_and_filter_convolved_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=600, labels=False)
        plt.close()

        image_filtered.save('filtered_image_' + filename, overwrite=True)
        image_filtered.plot()
        plt.title('DG Filtered Image ' + filename, fontsize=20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname='DG_filtered_image_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=600, labels=False)
        plt.close()

        ''' Saving the Variables for the image and filtered Image '''
        Filtering_Variables = collections.OrderedDict()
        Filtering_Variables['filename'] = [filename]
        Filtering_Variables['Image Size (nm)'] = [physical_image_size]
        Filtering_Variables['Image Size (pix)'] = [len(image.data)]
        Filtering_Variables['Real Space Sampling (nm/pix)'] = [
            real_space_sampling]
        Filtering_Variables['Reciprocal Space Sampling (1/nm/pix)'] = [
            reciprocal_sampling]
        Filtering_Variables['First Diffraction Ring (Diameter) (1/nm)'] = [
            d_inner]
        Filtering_Variables['Second Diffraction Ring (Diameter) (1/nm)'] = [
            d_outer]
        Filtering_Variables['First Diffraction Ring (Radius) (1/nm)'] = [
            reciprocal_d_inner]
        Filtering_Variables['Second Diffraction Ring (Radius) (1/nm)'] = [
            reciprocal_d_outer]
        Filtering_Variables['First Diffraction Ring (Radius) (pix)'] = [
            reciprocal_d_inner_pix]
        Filtering_Variables['Second Diffraction Ring (Radius) (pix)'] = [
            reciprocal_d_outer_pix]
        Filtering_Variables['Positive Gaussian FWHM (pix)'] = [fwhm_pos_gaus]
        Filtering_Variables['Negative Gaussian FWHM (pix)'] = [fwhm_neg_gaus]
        Filtering_Variables['Positive Gaussian Size'] = [len(image.data)]
        Filtering_Variables['Negative Gaussian Size'] = [len(image.data)]
        Filtering_Variables['Positive Gaussian FWHM'] = [fwhm_pos_gaus]
        Filtering_Variables['Negative Gaussian FWHM'] = [fwhm_neg_gaus]
        Filtering_Variables['Negative Gaussian Amplitude'] = [
            neg_gauss_amplitude_calculated]
        Filtering_Variables['Delta used for Calculation'] = [delta]
        Filtering_Variables_Table = pd.DataFrame(Filtering_Variables)
        Filtering_Variables_Table
        Filtering_Variables_Table.to_pickle(
            'filtering_variables_table_' + filename + '.pkl')
        #Filtering_Variables_Table.to_csv('Filtering_Variables_Table.csv', sep=',', index=False)

    return(image_filtered)


def compare_image_to_filtered_image(image_to_filter,
                                    reference_image,
                                    filename,
                                    delta_image_filter,
                                    cropping_area,
                                    separation,
                                    max_sigma=6,
                                    percent_to_nn=0.4,
                                    mask_radius=None,
                                    refine=False):
    '''
    Gaussian blur an image for comparison with a reference image. 
    Good for finding the best gaussian blur for a simulation by 
    comparing to an experimental image.
    See measure_image_errors() and load_and_compare_images()

    >>> new_sim_data = compare_image_to_filtered_image(
                                    image_to_filter=simulation, 
                                    reference_image=atom_lattice_max)


    '''
    image_to_filter_data = image_to_filter.data
    reference_image_data = reference_image.data

    mse_number_list = []
    ssm_number_list = []

    for i in np.arange(0, max_sigma+delta_image_filter, delta_image_filter):

        image_to_filter_data_filtered = gaussian_filter(image_to_filter_data,
                                                        sigma=i)
        temp_image_filtered = hs.signals.Signal2D(
            image_to_filter_data_filtered)
#        temp_image_filtered.plot()
        calibrate_intensity_distance_with_sublattice_roi(image=temp_image_filtered,
                                                            cropping_area=cropping_area,
                                                            separation=separation,
                                                            percent_to_nn=percent_to_nn,
                                                            mask_radius=mask_radius,
                                                            refine=refine)

        mse_number, ssm_number = measure_image_errors(
            imageA=reference_image_data,
            imageB=temp_image_filtered.data,
            filename=None)

        mse_number_list.append([mse_number, i])
        ssm_number_list.append([ssm_number, i])

    mse = [mse[:1] for mse in mse_number_list]
    mse_indexing = [indexing[1:2] for indexing in mse_number_list]
    ssm = [ssm[:1] for ssm in ssm_number_list]
    ssm_indexing = [indexing[1:2] for indexing in ssm_number_list]

    ideal_mse_number_index = mse.index(min(mse))
    ideal_mse_number = float(
        format(mse_number_list[ideal_mse_number_index][1], '.1f'))

    ideal_ssm_number_index = ssm.index(max(ssm))
    ideal_ssm_number = float(
        format(ssm_number_list[ideal_ssm_number_index][1], '.1f'))

    # ideal is halway between mse and ssm indices
    ideal_sigma = (ideal_mse_number + ideal_ssm_number)/2
    ideal_sigma_y_coord = (float(min(mse)[0]) + float(max(ssm)[0]))/2

    image_to_filter_filtered = gaussian_filter(image_to_filter_data,
                                               sigma=ideal_sigma)

    image_filtered = hs.signals.Signal2D(image_to_filter_filtered)

    calibrate_intensity_distance_with_sublattice_roi(image=image_filtered,
                                                        cropping_area=cropping_area,
                                                        separation=separation,
                                                        percent_to_nn=percent_to_nn,
                                                        mask_radius=mask_radius,
                                                        refine=refine)

    if filename is not None:

        plt.figure()
        plt.scatter(x=ssm_indexing, y=ssm, label='ssm',
                    marker='x', color='magenta')
        plt.scatter(x=mse_indexing, y=mse, label='mse', marker='o', color='b')
        plt.scatter(x=ideal_sigma, y=ideal_sigma_y_coord, label='\u03C3 = ' +
                    str(round(ideal_sigma, 2)), marker='D', color='k')
        plt.title("MSE & SSM vs. Gauss Blur " + filename, fontsize=20)
        plt.xlabel("\u03C3 (Gaussian Blur)", fontsize=16)
        plt.ylabel("MSE (0) and SSM (1)", fontsize=16)
        plt.legend()
        plt.tight_layout
        plt.show()
        plt.savefig(fname='MSE_SSM_gaussian_blur_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)

    return(image_filtered)


#from all_atom_position_functions import *

'''
This is the first loop to refine the simulation:
refining the atom_position's assigned atoms without
changing the atom_position location. Using intensity of current positions only
'''

# aim hereis to change the elements in the sublattice to something that will
#   make the simulation agree more with the experiment


def change_sublattice_atoms_via_intensity(sublattice, image_diff_array, darker_or_brighter,
                                          element_list):
    # get the index in sublattice from the image_difference_intensity() output,
    #   which is the image_diff_array input here.
    # then, depending on whether the image_diff_array is for atoms that should
    # be brighter or darker, set a new element to that sublattice atom_position
    '''
    Change the elements in a sublattice object to a higher or lower combined
    atomic (Z) number.
    The aim is to change the sublattice elements so that the experimental image
    agrees with the simulated image in a realistic manner.
    See image_difference_intensity()

    Parameters
    ----------

    sublattice : Atomap Sublattice object
        the elements of this sublattice will be changed
    image_diff_array : Numpy 2D array
        Contains (p, x, y, intensity) where
        p = index of Atom_Position in sublattice
        x = Atom_Position.pixel_x
        y = Atom_Position.pixel_y
        intensity = calculated intensity of atom in sublattice.image
    darker_or_brighter : int
        if the element should have a lower combined atomic Z number,
        darker_or_brighter = 0.
        if the element should have a higher combined atomic Z number,
        darker_or_brighter = 1
        In other words, the image_diff_array will change the given 
        sublattice elements to darker or brighter spots by choosing 0 and 1,
        respectively.
    element_list : list
        list of element configurations

    Returns
    -------

    n/a - changes sublattice elements inplace

    Examples
    --------

    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> for i in range(0, len(sublattice.atom_list)):
            sublattice.atom_list[i].elements = 'Mo_1'
            sublattice.atom_list[i].z_height = [0.5]
    >>> element_list = ['H_0', 'Mo_1', 'Mo_2']
    >>> image_diff_array = np.array([5, 2, 2, 20])
    >>> # This will change the 5th atom in the sublattice to a lower atomic Z 
    >>> # number, i.e., 'H_0' in the given element_list
    >>> change_sublattice_atoms_via_intensity(sublattice=sublattice, 
                                      image_diff_array=image_diff_array, 
                                      darker_or_brighter=0,
                                      element_list=element_list)


    '''
    if image_diff_array.size == 0:
        pass
    else:
        print('Changing some atoms')
        for p in image_diff_array[:, 0]:
            # could be a better way to do this within image_difference_intensity()
            p = int(p)

            elem = sublattice.atom_list[p].elements
            if elem in element_list:
                atom_index = element_list.index(elem)

                if darker_or_brighter == 0:
                    if '_0' in elem:
                        pass
                    else:
                        new_atom_index = atom_index - 1
                        if len(sublattice.atom_list[p].z_height) == 2:
                            z_h = sublattice.atom_list[p].z_height
                            sublattice.atom_list[p].z_height = [
                                (z_h[0] + z_h[1])/2]
                        else:
                            pass
                        new_atom = element_list[new_atom_index]

                elif darker_or_brighter == 1:
                    new_atom_index = atom_index + 1
                    if len(sublattice.atom_list[p].z_height) == 2:
                        z_h = sublattice.atom_list[p].z_height
                        sublattice.atom_list[p].z_height = [
                            (z_h[0] + z_h[1])/2]
                    else:
                        pass
                        new_atom = element_list[new_atom_index]

                elif new_atom_index < 0:
                    raise ValueError("You don't have any smaller atoms")
                elif new_atom_index >= len(element_list):
                    raise ValueError("You don't have any bigger atoms")

#                new_atom = element_list[new_atom_index]

            elif elem == '':
                raise ValueError("No element assigned for atom %s. Note that this \
                                 error only picks up first instance of fail" % p)
            elif elem not in element_list:
                raise ValueError("This element isn't in the element_list")

            try:
                new_atom
            except NameError:
                pass
            else:
                sublattice.atom_list[p].elements = new_atom

#            sublattice.atom_list[p].elements = new_atom


def image_difference_intensity(sublattice,
                               simulation_image,
                               element_list,
                               filename=None,
                               percent_to_nn=0.40,
                               mask_radius=None,
                               change_sublattice=False):
    ''' 
    Find the differences in a sublattice's atom_position intensities. 
    Change the elements of these atom_positions depending on this difference of
    intensities.

    The aim is to change the sublattice elements so that the experimental image
    agrees with the simulated image in a realistic manner.

    Parameters
    ----------

    sublattice : Atomap Sublattice object
        Elements of this sublattice will be refined
    simulation_image : HyperSpy 2D signal
        The image you wish to refine with, usually an image simulation of the
        sublattice.image
    element_list : list
        list of element configurations, used for refinement
    filename : string, default None
        name with which the image will be saved
    percent_to_nn : float, default 0.40
        Determines the boundary of the area surrounding each atomic 
        column, as fraction of the distance to the nearest neighbour.
    mask_radius : float, default None
        Radius of the mask around each atom. If this is not set,
        the radius will be the distance to the nearest atom in the
        same sublattice times the `percent_to_nn` value.
        Note: if `mask_radius` is not specified, the Atom_Position objects
        must have a populated nearest_neighbor_list.
    change_sublattice : bool, default False
        If change_sublattice is set to True, all incorrect element assignments
        will be corrected inplace.


    Returns
    -------
    Nothing - changes the elements within the sublattice object.

    Example
    -------

    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> simulation_image = am.dummy_data.get_simple_cubic_with_vacancies_signal()
    >>> for i in range(0, len(sublattice.atom_list)):
            sublattice.atom_list[i].elements = 'Mo_1'
            sublattice.atom_list[i].z_height = [0.5]
    >>> element_list = ['H_0', 'Mo_1', 'Mo_2']
    >>> image_difference_intensity(sublattice=sublattice,
                                   simulation_image=simulation_image,
                                   element_list=element_list)

    with some image noise and plotting the images
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice(image_noise=True)
    >>> simulation_image = am.dummy_data.get_simple_cubic_with_vacancies_signal()
    >>> for i in range(0, len(sublattice.atom_list)):
            sublattice.atom_list[i].elements = 'Mo_1'
            sublattice.atom_list[i].z_height = [0.5]
    >>> element_list = ['H_0', 'Mo_1', 'Mo_2']
    >>> image_difference_intensity(sublattice=sublattice,
                                   simulation_image=simulation_image,
                                   element_list=element_list,
                                   plot_details=True)

    '''

    # np.array().T needs to be taken away for newer atomap versions
    sublattice_atom_positions = np.array(sublattice.atom_positions).T

    diff_image = hs.signals.Signal2D(sublattice.image - simulation_image.data)

    # create sublattice for the 'difference' data
    diff_sub = am.Sublattice(
        atom_position_list=sublattice_atom_positions, image=diff_image)

    if percent_to_nn is not None:
        sublattice.find_nearest_neighbors()
        diff_sub.find_nearest_neighbors()
    else:
        pass

    # Get the intensities of these sublattice positions
    diff_sub.get_atom_column_amplitude_mean_intensity(
        percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    diff_mean_ints = np.array(
        diff_sub.atom_amplitude_mean_intensity, ndmin=2).T
    #diff_mean_ints = np.array(diff_mean_ints, ndmin=2).T

    # combine the sublattice_atom_positions and the intensities for
    # future indexing
    positions_intensities_list = np.append(sublattice_atom_positions,
                                           diff_mean_ints, axis=1)
    # find the mean and std dev of this distribution of intensities
    mean_ints = np.mean(diff_mean_ints)
    std_dev_ints = np.std(diff_mean_ints)

    # plot the mean and std dev on each side of intensities histogram
    std_from_mean = np.array([mean_ints-std_dev_ints, mean_ints+std_dev_ints,
                              mean_ints-(2*std_dev_ints), mean_ints +
                              (2*std_dev_ints),
                              mean_ints-(3*std_dev_ints), mean_ints +
                              (3*std_dev_ints),
                              mean_ints-(4*std_dev_ints), mean_ints +
                              (4*std_dev_ints)
                              ], ndmin=2).T
    y_axis_std = np.array([len(diff_mean_ints)/100] * len(std_from_mean),
                          ndmin=2).T
    std_from_mean_array = np.concatenate((std_from_mean, y_axis_std), axis=1)
    std_from_mean_array = np.append(std_from_mean, y_axis_std, axis=1)

    # if the intensity if outside 3 sigma, give me those atom positions
    # and intensities (and the index!)
    outliers_bright, outliers_dark = [], []
    for p in range(0, len(positions_intensities_list)):
        x, y = positions_intensities_list[p,
                                          0], positions_intensities_list[p, 1]
        intensity = positions_intensities_list[p, 2]

        if positions_intensities_list[p, 2] > std_from_mean_array[7, 0]:
            outliers_bright.append([p, x, y, intensity])
        elif positions_intensities_list[p, 2] < std_from_mean_array[6, 0]:
            outliers_dark.append([p, x, y, intensity])
    # Now we have the details of the not correct atom_positions
    outliers_bright = np.array(outliers_bright)
    outliers_dark = np.array(outliers_dark)

    if change_sublattice == True:
        # Now make the changes to the sublattice for both bright and dark arrays
        change_sublattice_atoms_via_intensity(sublattice=sublattice,
                                              image_diff_array=outliers_bright,
                                              darker_or_brighter=1,
                                              element_list=element_list)
        change_sublattice_atoms_via_intensity(sublattice=sublattice,
                                              image_diff_array=outliers_dark,
                                              darker_or_brighter=0,
                                              element_list=element_list)

    else:
        pass

    if filename is not None:
        #        sublattice.plot()
        #        simulation_image.plot()
        #        diff_image.plot()
        diff_sub.plot()
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.title("Image Differences with " +
                  sublattice.name + " Overlay", fontsize=16)
        plt.savefig(fname="Image Differences with " + sublattice.name + "Overlay.png",
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)

        plt.figure()
        plt.hist(diff_mean_ints, bins=50, color='b', zorder=-1)
        plt.scatter(mean_ints, len(diff_mean_ints)/50, c='red', zorder=1)
        plt.scatter(
            std_from_mean_array[:, 0], std_from_mean_array[:, 1], c='green', zorder=1)
        plt.title("Histogram of " + sublattice.name +
                  " Intensities", fontsize=16)
        plt.xlabel("Intensity (a.u.)", fontsize=16)
        plt.ylabel("Count", fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.savefig(fname="Histogram of " + sublattice.name + " Intensities.png",
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)

    else:
        pass


def image_difference_position(sublattice_list,
                              simulation_image,
                              pixel_threshold,
                              filename=None,
                              percent_to_nn=0.40,
                              mask_radius=None,
                              num_peaks=5,
                              add_sublattice=False,
                              sublattice_name='sub_new'):
    '''
    Find new atomic coordinates by comparing experimental to simulated image. 
    Create a new sublattice to store the new atomic coordinates.

    The aim is to change the sublattice elements so that the experimental image
    agrees with the simulated image in a realistic manner.

    Parameters
    ----------

    sublattice_list : list of atomap sublattice objects
    simulation_image : simulated image used for comparison with sublattice image
    pixel_threshold : int
        minimum pixel distance from current sublattice atoms. If the new atomic
        coordinates are greater than this distance, they will be created.
        Choose a pixel_threshold that will not create new atoms in unrealistic
        positions.
    filename : string, default None
        name with which the image will be saved
    percent_to_nn : float, default 0.40
        Determines the boundary of the area surrounding each atomic 
        column, as fraction of the distance to the nearest neighbour.
    mask_radius : float, default None
        Radius of the mask around each atom. If this is not set,
        the radius will be the distance to the nearest atom in the
        same sublattice times the `percent_to_nn` value.
        Note: if `mask_radius` is not specified, the Atom_Position objects
        must have a populated nearest_neighbor_list.
    num_peaks : int, default 5
        number of new atoms to add
    add_sublattice : bool, default False
        If set to True, a new sublattice will be created and returned.
        The reason it is set to False is so that one can check if new atoms
        would be added with the given parameters.
    sublattice_name : string, default 'sub_new'
        the outputted sublattice object name and sublattice.name the new 
        sublattice will be given

    Returns
    -------
    Atomap sublattice object

    Examples
    --------

    >>> sublattice = am.dummy_data.get_simple_cubic_with_vacancies_sublattice(
                                                    image_noise=True)
    >>> simulation_image = am.dummy_data.get_simple_cubic_signal()
    >>> for i in range(0, len(sublattice.atom_list)):
                sublattice.atom_list[i].elements = 'Mo_1'
                sublattice.atom_list[i].z_height = '0.5'
    >>> # Check without adding a new sublattice
    >>> image_difference_position(sublattice_list=[sublattice],
                                  simulation_image=simulation_image,
                                  pixel_threshold=1,
                                  filename='',
                                  mask_radius=5,
                                  num_peaks=5,
                                  add_sublattice=False)
    >>> # Add a new sublattice
    >>> # if you have problems with mask_radius, increase it!
    >>> # Just a gaussian fitting issue, could turn it off!
    >>> sub_new = image_difference_position(sublattice_list=[sublattice],
                                          simulation_image=simulation_image,
                                          pixel_threshold=10,
                                          filename='',
                                          mask_radius=10,
                                          num_peaks=5,
                                          add_sublattice=True)
    '''
    image_for_sublattice = sublattice_list[0]
    diff_image = hs.signals.Signal2D(
        image_for_sublattice.image - simulation_image.data)
    diff_image_inverse = hs.signals.Signal2D(
        simulation_image.data - image_for_sublattice.image)

    # below function edit of get_atom_positions. Just allows num_peaks from
    # sklearn>find_local_maximum
    atom_positions_diff_image = atomap.atom_finding_refining.get_atom_positions_in_difference_image(
        diff_image, num_peaks=num_peaks)
    atom_positions_diff_image_inverse = atomap.atom_finding_refining.get_atom_positions_in_difference_image(
        diff_image_inverse, num_peaks=num_peaks)

    diff_image_sub = am.Sublattice(atom_positions_diff_image, diff_image)
    diff_image_sub.refine_atom_positions_using_center_of_mass(
        percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    diff_image_sub.refine_atom_positions_using_2d_gaussian(
        percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    atom_positions_sub_diff = np.array(diff_image_sub.atom_positions).T

    # sublattice.plot()

    diff_image_sub_inverse = am.Sublattice(atom_positions_diff_image_inverse,
                                           diff_image_inverse)
    diff_image_sub_inverse.refine_atom_positions_using_center_of_mass(
        percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    diff_image_sub_inverse.refine_atom_positions_using_2d_gaussian(
        percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    atom_positions_sub_diff_inverse = np.array(
        diff_image_sub_inverse.atom_positions).T

    # these should be inputs for the image_list below

    atom_positions_diff_all = np.concatenate(
        (atom_positions_sub_diff, atom_positions_sub_diff_inverse))

    atom_positions_sub_new = []

    for atom in range(0, len(atom_positions_diff_all)):

        new_atom_distance_list = []

        for sublattice in sublattice_list:
            sublattice_atom_pos = np.array(sublattice.atom_positions).T

            for p in range(0, len(sublattice_atom_pos)):

                xy_distances = atom_positions_diff_all[atom] - \
                    sublattice_atom_pos[p]

                # put all distances in this array with this loop
                vector_array = []
                vector = np.sqrt((xy_distances[0]**2) +
                                 (xy_distances[1]**2))
                vector_array.append(vector)

                new_atom_distance_list.append(
                    [vector_array, atom_positions_diff_all[atom],
                     sublattice])

        # use list comprehension to get the distances on their own, the [0] is
        # changing the list of lists to a list of floats
        new_atom_distance_sublist = [sublist[:1][0] for sublist in
                                     new_atom_distance_list]
        new_atom_distance_min = min(new_atom_distance_sublist)

        new_atom_distance_min_index = new_atom_distance_sublist.index(
            new_atom_distance_min)

        new_atom_index = new_atom_distance_list[new_atom_distance_min_index]

        if new_atom_index[0][0] > pixel_threshold:  # greater than 10 pixels

            if len(atom_positions_sub_new) == 0:
                atom_positions_sub_new = [np.ndarray.tolist(new_atom_index[1])]
            else:
                atom_positions_sub_new.extend(
                    [np.ndarray.tolist(new_atom_index[1])])
        else:
            pass

    if len(atom_positions_sub_new) == 0:
        print("No New Atoms")
    elif len(atom_positions_sub_new) != 0 and add_sublattice == True:
        print("New Atoms Found! Adding to a new sublattice")

        sub_new = am.Sublattice(atom_positions_sub_new, sublattice_list[0].image,
                                name=sublattice_name, color='cyan')
#        sub_new.refine_atom_positions_using_center_of_mass(
#           percent_to_nn=percent_to_nn, mask_radius=mask_radius)
#        sub_new.refine_atom_positions_using_2d_gaussian(
#           percent_to_nn=percent_to_nn, mask_radius=mask_radius)

    else:
        pass

    try:
        sub_new
    except NameError:
        sub_new_exists = False
    else:
        sub_new_exists = True

    if filename is not None:
        '''
        diff_image.plot()
        diff_image_sub.plot()
        diff_image_inverse.plot()
        diff_image_sub_inverse.plot()
        '''

        plt.figure()
        plt.suptitle('Image Difference Position' + filename)

        plt.subplot(1, 2, 1)
        plt.imshow(diff_image.data)
        plt.title('diff')
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(diff_image_inverse.data)
        plt.title('diff_inv')
        plt.axis("off")
        plt.show()

        plt.savefig(fname='pos_diff_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)

        if sub_new_exists == True:
            sub_new.plot()
            plt.title(sub_new.name + filename, fontsize=20)
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.tight_layout()
            plt.savefig(fname='pos_diff_' + sub_new.name + filename + '.png',
                        transparent=True, frameon=False, bbox_inches='tight',
                        pad_inches=None, dpi=300, labels=False)

    return sub_new if sub_new_exists == True else None


# https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def measure_image_errors(imageA, imageB, filename):
    ''' 
    Measure the Mean Squared Error (mse) and Structural Similarity Index (ssm)
    between two images.

    Parameters
    ----------

    imageA, imageB : 2D NumPy array, default None
        Two images between which to measure mse and ssm
    filename : string, default None
        name with which the image will be saved

    Returns
    -------

    mse_number, ssm_number : float
    returned subtracted image is imageA - imageB

    Example
    -------

    >>> imageA = am.dummy_data.get_simple_cubic_signal().data
    >>> imageB = am.dummy_data.get_simple_cubic_with_vacancies_signal().data
    >>> mse_number, ssm_number = measure_image_errors(imageA, imageB,
                                                      plot_details=True)

    Showing the ideal case of both images being exactly equal   
    >>> imageA = am.dummy_data.get_simple_cubic_signal().data
    >>> imageB = am.dummy_data.get_simple_cubic_signal().data
    >>> mse_number, ssm_number = measure_image_errors(imageA, imageA,
                                                      plot_details=True)

    '''

    mse_number = mse(imageA, imageB)
    ssm_number = ssm(imageA, imageB)

    if filename is not None:
        plt.figure()
        plt.suptitle("MSE: %.6f, SSIM: %.6f" %
                     (mse_number, ssm_number) + filename)

        plt.subplot(2, 2, 1)
        plt.imshow(imageA)
        plt.title('imageA')
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.imshow(imageB)
        plt.title('imageB')
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.imshow(imageA - imageB)
        plt.title('imageA - imageB')
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.scatter(mse_number.size, mse_number, color='r',
                    marker='x', label='mse')
        plt.scatter(ssm_number.size, ssm_number, color='b',
                    marker='o', label='ssm')
        plt.title('MSE & SSM')
        plt.legend()
        plt.show()

        plt.savefig(fname='MSE_SSM_single_image_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)

    return(mse_number, ssm_number)


#imageA = am.dummy_data.get_simple_cubic_signal().data
#imageB = am.dummy_data.get_simple_cubic_with_vacancies_signal().data
# mse_number, ssm_number = measure_image_errors(imageA, imageB,
#                                                      plot_details=True)


def load_and_compare_images(imageA, imageB, filename=None):
    '''
    Load two images and compare their mean standard error and structural
    simularity index.

    Parameters
    ----------

    imageA, imageB : string
        filename of the images to be loaded and compared
    filename : string, default None
        name with which the image will be saved

    Returns
    -------
    mean standard error and structural simularity index

    Examples
    --------

    >>> imageA = am.dummy_data.get_simple_cubic_signal(image_noise=True)
    >>> imageB = am.dummy_data.get_simple_cubic_signal()
    >>> load_and_compare_images(imageA, imageB, filename=None)

    '''
    imageA = hs.load(imageA)
    imageB = hs.load(imageB)

    mse_number, ssm_number = measure_image_errors(imageA, imageB, filename)

    return(mse_number, ssm_number)


def dm3_stack_to_tiff_stack(loading_file,
                            loading_file_extension='.dm3',
                            saving_file_extension='.tif',
                            crop=False,
                            crop_start=20.0,
                            crop_end=80.0):
    '''
    Save an image stack filetype to a different filetype.
    For example dm3 to tiff

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

    Returns
    -------
    n/a

    Examples
    --------

    >>> directory = os.chdir('C:/Users/Eoghan.OConnell/Documents/Documents/Eoghan UL/PHD/Experimental/Ion implantation experiments/Feb 2019 SStem data')
    >>> filename = '003_HAADF_movie_300_4nm_MC'
    >>> dm3_stack_to_tiff_stack(filename=filename, crop=True, crop_start=20.0, crop_end=30.0)


    '''
    if '.' in loading_file:
        file = loading_file
        filename = loading_file[:-4]
    else:
        file = loading_file + loading_file_extension
        filename = loading_file

    s = hs.load(file)

    if crop == True:
        s = s.inav[crop_start:crop_end]

        # In the form: '20.0:80.0'

    # Save the dm3 file as a tiff stack. Allows us to use below analysis without editing!
    saving_file = filename + saving_file_extension
    s.save(saving_file)
    # These two lines normalize the hyperspy loaded file. Do Not so if you are also normalizing below
    # stack.change_dtype('float')
    #stack.data /= stack.data.max()

#dm3_stack_to_tiff_stack(loading_file = loading_file, crop=True, crop_start=50.0, crop_end=54.0)


# for after rigid registration
def save_individual_images(image_stack, output_folder='individual_images'):
    '''
    Save each image in an image stack. The images are saved in a new folder.

    Parameters
    ----------

    image_stack : rigid registration image stack object

    output_folder : string
        Name of the folder in which all individual images from 
        the stack will be saved.

    Returns
    -------

    n/a

    Examples
    --------

    '''

    # Save each image as a 32 bit tiff )cqn be displayed in DM
    image_stack_32bit = np.float32(image_stack)
    folder = './' + output_folder + '/'
    create_new_folder(create_new_folder)
    i = 0
    delta = 1
    # Find the number of images, change to an integer for the loop.
    while i < int(image_stack_32bit[0, 0, :].shape[0]):
        im = image_stack_32bit[:, :, i]
        i_filled = str(i).zfill(4)
        imwrite(folder + 'images_aligned_%s.tif' % i_filled, im)
        i = i+delta

# save_individual_images(image_stack=s.stack_registered)


def create_new_folder(new_folder_name):
    '''
    Create a folder in the given directory

    Parameters
    ----------

    new_folder_name : string
        name of the new folder. It will be created in the current directory

    Returns
    -------
    Nothing

    Examples
    --------

    >>> create_new_folder('test_folder')

    '''
    try:
        if not os.path.exists(new_folder_name):
            os.makedirs('./' + new_folder_name + '/')
    except OSError:
        print('Could not create directory ' + new_folder_name)


def atomic_positions_load_and_refine(image,
                                     filename,
                                     atom_positions_1_original,
                                     atom_positions_2_original,
                                     atom_positions_3_original,
                                     percent_to_nn,
                                     percent_to_nn_remove_atoms,
                                     min_cut_off_percent,
                                     max_cut_off_percent,
                                     min_cut_off_percent_sub3,
                                     max_cut_off_percent_sub3,
                                     mask_radius_sub1,
                                     mask_radius_sub2,
                                     sub1_colour='blue',
                                     sub2_colour='yellow',
                                     sub3_colour='green',
                                     sub1_name='sub1',
                                     sub2_name='sub2',
                                     sub3_name='sub3',
                                     sub3_inverse_name='sub3_inverse'):

    #    atom_positions_1_original = am.get_atom_positions(image, calibration_separation, pca=True)
    sub1 = am.Sublattice(atom_positions_1_original, image,
                         name=sub1_name, color=sub1_colour)
    sub1.find_nearest_neighbors()

    false_list_sub1 = toggle_atom_refine_position_automatically(
        sublattice=sub1,
        filename=filename,
        min_cut_off_percent=min_cut_off_percent,
        max_cut_off_percent=max_cut_off_percent,
        range_type='internal',
        percent_to_nn=percent_to_nn,
        mask_radius=mask_radius_sub1)
    '''
    sub1.toggle_atom_refine_position_with_gui()
    plt.title(sub1_name + '_refine_toggle', fontsize = 20)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(fname= sub1_name + '_refine_toggle.png', 
                transparent=True, frameon=False, bbox_inches='tight', 
                pad_inches=None, dpi=300, labels=False)
    plt.close()
    '''
    sub1.refine_atom_positions_using_center_of_mass(
        percent_to_nn=percent_to_nn, mask_radius=mask_radius_sub1)
    sub1.refine_atom_positions_using_2d_gaussian(
        percent_to_nn=percent_to_nn, mask_radius=mask_radius_sub1)
    # sub1.plot()

    atom_positions_1_refined = np.array(sub1.atom_positions).T
    np.save(file='atom_positions_1_refined_' +
            filename, arr=atom_positions_1_refined)

    sub1.get_atom_list_on_image(markersize=2, color=sub1_colour).plot()
    plt.title(sub1_name, fontsize=20)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(fname=sub1_name + '_' + filename + '.png',
                transparent=True, frameon=False, bbox_inches='tight',
                pad_inches=None, dpi=300, labels=False)
    plt.close()

    ''' SUBLATTICE 2 '''
    # remove first sublattice
    sub1_atoms_removed = remove_atoms_from_image_using_2d_gaussian(sub1.image, sub1,
                                                                   percent_to_nn=percent_to_nn_remove_atoms)
    sub1_atoms_removed = hs.signals.Signal2D(sub1_atoms_removed)
    '''
    sub1.construct_zone_axes(atom_plane_tolerance=0.5)
#    sub1.plot_planes()
    zone_number = 5
    zone_axis_001 = sub1.zones_axis_average_distances[zone_number]
    atom_positions_2_original = sub1.find_missing_atoms_from_zone_vector(zone_axis_001, vector_fraction=vector_fraction_sub2)
    '''

#    am.get_feature_separation(sub1_atoms_removed).plot()
#    atom_positions_2 = am.get_atom_positions(sub1_atoms_removed, 19)
#    atom_positions_2_original = am.add_atoms_with_gui(sub1_atoms_removed, atom_positions_2)
#    np.save(file='atom_positions_2_original', arr = atom_positions_2_original)

    sub2_refining = am.Sublattice(atom_positions_2_original, sub1_atoms_removed,
                                  name=sub2_name, color=sub2_colour)
    sub2_refining.find_nearest_neighbors()

    # Auto
    false_list_sub2 = toggle_atom_refine_position_automatically(
        sublattice=sub2_refining,
        filename=filename,
        min_cut_off_percent=min_cut_off_percent,
        max_cut_off_percent=max_cut_off_percent,
        range_type='internal',
        percent_to_nn=percent_to_nn,
        mask_radius=mask_radius_sub2)

    # sub2_refining.toggle_atom_refine_position_with_gui()
    #plt.title(sub2_name + '_refine_toggle', fontsize = 20)
    # plt.gca().axes.get_xaxis().set_visible(False)
    # plt.gca().axes.get_yaxis().set_visible(False)
    # plt.tight_layout()
    # plt.savefig(fname= sub2_name + '_refine_toggle.png',
    #            transparent=True, frameon=False, bbox_inches='tight',
    #            pad_inches=None, dpi=300, labels=False)
    # plt.close()

    sub2_refining.refine_atom_positions_using_center_of_mass(
        percent_to_nn=percent_to_nn, mask_radius=mask_radius_sub2)
    sub2_refining.refine_atom_positions_using_2d_gaussian(
        percent_to_nn=percent_to_nn, mask_radius=mask_radius_sub2)
    # sub2_refining.plot()

    sub2_refining.get_atom_list_on_image(
        markersize=2, color=sub2_colour).plot()
    plt.title(sub2_name + '_refining', fontsize=20)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(fname=sub2_name + '_refining_' + filename + '.png',
                transparent=True, frameon=False, bbox_inches='tight',
                pad_inches=None, dpi=300, labels=False)
    plt.close()

    atom_positions_2_refined = np.array(sub2_refining.atom_positions).T
    np.save(file='atom_positions_2_refined_' +
            filename, arr=atom_positions_2_refined)

    sub2 = am.Sublattice(atom_positions_2_refined, image,
                         name=sub2_name, color=sub2_colour)
    sub2.find_nearest_neighbors()
#    sub2.plot()

    sub2.get_atom_list_on_image(markersize=2, color=sub2_colour).plot()
    plt.title(sub2_name, fontsize=20)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(fname=sub2_name + '_' + filename + '.png',
                transparent=True, frameon=False, bbox_inches='tight',
                pad_inches=None, dpi=300, labels=False)
    plt.close()

    ''' SUBLATTICE 3 '''
    '''
    atom_positions_3_original = sub1.find_missing_atoms_from_zone_vector(
            zone_axis_001, 
            vector_fraction=vector_fraction_sub3)
    
    #am.get_feature_separation(sub1_atoms_removed).plot()
    #atom_positions_2 = am.get_atom_positions(sub1_atoms_removed, 19)
#    atom_positions_3_original = am.add_atoms_with_gui(s, atom_positions_3_original)
#    np.save(file='atom_positions_3_original', arr = atom_positions_3_original)
    '''
    s_inverse = image
    s_inverse.data = np.divide(1, s_inverse.data)
    # s_inverse.plot()

    sub3_inverse = am.Sublattice(
        atom_positions_3_original, s_inverse, name=sub3_inverse_name, color=sub3_colour)
    sub3_inverse.find_nearest_neighbors()
    # sub3_inverse.plot()

    # get_sublattice_intensity(sublattice=sub3_inverse, intensity_type=intensity_type, remove_background_method=None,
    #                             background_sublattice=None, num_points=3, percent_to_nn=percent_to_nn,
    #                             mask_radius=radius_pix_S)

    false_list_sub3_inverse = toggle_atom_refine_position_automatically(
        sublattice=sub3_inverse,
        filename=filename,
        min_cut_off_percent=min_cut_off_percent,
        max_cut_off_percent=max_cut_off_percent,
        range_type='internal',
        method='mean',
        percent_to_nn=percent_to_nn,
        mask_radius=mask_radius_sub2)

    # sub3_inverse.toggle_atom_refine_position_with_gui()

    sub3_inverse.refine_atom_positions_using_center_of_mass(
        percent_to_nn=percent_to_nn, mask_radius=mask_radius_sub2)
    sub3_inverse.refine_atom_positions_using_2d_gaussian(
        percent_to_nn=percent_to_nn, mask_radius=mask_radius_sub2)
    # sub3_inverse.plot()

    atom_positions_3_refined = np.array(sub3_inverse.atom_positions).T
    np.save(file='atom_positions_3_refined_' +
            filename, arr=atom_positions_3_refined)

    image.data = np.divide(1, image.data)
    # s.plot()

    sub3 = am.Sublattice(atom_positions_3_refined, image,
                         name=sub3_name, color=sub3_colour)
    sub3.find_nearest_neighbors()
    # sub3.plot()

    # Now re-refine the adatom locations for the original data
    false_list_sub3 = toggle_atom_refine_position_automatically(
        sublattice=sub3,
        filename=filename,
        min_cut_off_percent=min_cut_off_percent_sub3,
        max_cut_off_percent=max_cut_off_percent_sub3,
        range_type='external',
        percent_to_nn=percent_to_nn,
        mask_radius=mask_radius_sub2)

    # sub3.toggle_atom_refine_position_with_gui()

    if any(false_list_sub3):
        sub3.refine_atom_positions_using_center_of_mass(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius_sub1)
        sub3.refine_atom_positions_using_2d_gaussian(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius_sub2)
    # sub3.plot()

        atom_positions_3_refined = np.array(sub3.atom_positions).T
        np.save(file='atom_positions_3_refined_' +
                filename, arr=atom_positions_3_refined)

    sub3.get_atom_list_on_image(markersize=2).plot()
    plt.title(sub3_name, fontsize=20)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(fname=sub3_name + '_' + filename + '.png',
                transparent=True, frameon=False, bbox_inches='tight',
                pad_inches=None, dpi=300, labels=False)
    plt.close()

    # Now we have the correct, refined positions of the Mo, S and bksubs

    return(sub1, sub2, sub3)


# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:12:44 2019

@author: eoghan.oconnell
"""

warnings.simplefilter("ignore", UserWarning)


'''
directory = 'G:/SuperStem visit/Feb 2019 data/2019_02_18_QMR_S1262_MoS2-Se-10eV/005_rigid_reg/Try_5/32bit/images_aligned_0' 
os.chdir(directory)

image_refine_via_intensity_loop(atom_lattice_name='Atom_Lattice_total.hdf5',
                               change_sublattice=True,
                                   plot_details=False,
                                   intensity_type='total',
                                   intensity_refine_name = 'intensity_refine_',
                                   folder_name = 'refinement_of_intensity')
'''


def image_refine_via_intensity_loop(atom_lattice,
                                    change_sublattice,
                                    calibration_separation,
                                    calibration_area,
                                    percent_to_nn,
                                    mask_radius,
                                    element_list,
                                    image_sampling,
                                    iterations,
                                    delta_image_filter,
                                    simulation_filename,
                                    filename,
                                    intensity_type,
                                    intensity_refine_name='intensity_refine_',
                                    folder_name='refinement_of_intensity'):

    for sub in atom_lattice.sublattice_list:
        if sub.name == 'sub1':
            sub1 = sub
        elif sub.name == 'sub2':
            sub2 = sub
        elif sub.name == 'sub3':
            sub3 = sub
        else:
            pass

    if len(atom_lattice.image) == 1:
        image_pixel_x = len(atom_lattice.image.data[0, :])
        image_pixel_y = len(atom_lattice.image.data[:, 0])
        atom_lattice_data = atom_lattice.image.data
        atom_lattice_signal = atom_lattice.image
    elif len(atom_lattice.image) > 1:
        image_pixel_x = len(atom_lattice.image[0, :])
        image_pixel_y = len(atom_lattice.image[:, 0])
        atom_lattice_data = atom_lattice.image
        atom_lattice_signal = atom_lattice.signal

    '''
    Image Intensity Loop
    '''

    if len(calibration_area) != 2:
        raise ValueError('calibration_area_simulation must be two points')

    df_inten_refine = pd.DataFrame(columns=element_list)

    real_sampling_exp_angs = image_sampling*10

    if str(real_sampling_exp_angs)[-1] == '5':
        real_sampling_sim_angs = real_sampling_exp_angs + 0.000005
    else:
        pass

    t0 = time()

    for suffix in range(1, iterations):

        loading_suffix = '_' + str(suffix)

        saving_suffix = '_' + str(suffix+1)

        if '.xyz' in simulation_filename:
            pass
        else:
            simulation_filename = simulation_filename + '.xyz'

        file_exists = os.path.isfile(simulation_filename)
        if file_exists:
            pass
        else:
            raise OSError('XYZ file not found, stopping refinement')

        file = pr.Metadata(filenameAtoms=simulation_filename, E0=60e3)

        file.integrationAngleMin = 0.085
        file.integrationAngleMax = 0.186

        file.interpolationFactorX = file.interpolationFactorY = 16
        file.realspacePixelSizeX = file.realspacePixelSizeY = 0.0654
    #    file.probeStepX = file.cellDimX/atom_lattice_data.shape[1]
    #    file.probeStepY = file.cellDimY/atom_lattice_data.shape[0]
        file.probeStepX = round(real_sampling_sim_angs, 6)
        file.probeStepY = round(real_sampling_sim_angs, 6)
        file.numFP = 1
        file.probeSemiangle = 0.030
        file.alphaBeamMax = 0.032  # in rads
        file.detectorAngleStep = 0.001
        file.scanWindowXMin = file.scanWindowYMin = 0.0
        file.scanWindowYMax = file.scanWindowXMax = 1.0
        file.algorithm = "prism"
        file.numThreads = 2
        file.save3DOutput = False

        file.filenameOutput = intensity_refine_name + loading_suffix + ".mrc"

        file.go()

        simulation = hs.load('prism_2Doutput_' + file.filenameOutput)
        simulation.axes_manager[0].name = 'extra_dimension'
        simulation = simulation.sum('extra_dimension')
        simulation.axes_manager[0].scale = image_sampling
        simulation.axes_manager[1].scale = image_sampling

        calibrate_intensity_distance_with_sublattice_roi(image=simulation,
                                                         cropping_area=calibration_area,
                                                         separation=calibration_separation,
                                                         filename=intensity_refine_name + "Simulation" + loading_suffix,
                                                         percent_to_nn=percent_to_nn,
                                                         mask_radius=mask_radius,
                                                         scalebar_true=True)

        # simulation.plot()

        # Filter the image with Gaussian noise to get better match with experiment
        simulation_new = compare_image_to_filtered_image(
            image_to_filter=simulation,
            reference_image=atom_lattice_signal,
            delta_image_filter=delta_image_filter,
            cropping_area=calibration_area,
            separation=calibration_separation,
            percent_to_nn=percent_to_nn,
            mask_radius=mask_radius,
            refine=False,
            filename=filename)

        simulation = simulation_new

        simulation.save(intensity_refine_name +
                        'Filt_Simulation' + loading_suffix + '.hspy')

        simulation.plot()
        plt.title('Filt_Simulation' + loading_suffix, fontsize=20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname=intensity_refine_name + 'Filt_Simulation' + loading_suffix + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)
        plt.close()

        '''
        Need to add the intensity type to the image_difference_intensity algorithm!
        '''

        counter_before_refinement = count_atoms_in_sublattice_list(
            sublattice_list=atom_lattice.sublattice_list,
            filename=intensity_refine_name + 'Elements' + loading_suffix)

        if suffix == 1:
            df_inten_refine = df_inten_refine.append(
                counter_before_refinement, ignore_index=True).fillna(0)
        else:
            pass

        ''' Sub1 '''
        image_difference_intensity(sublattice=sub1,
                                   simulation_image=simulation,
                                   element_list=element_list,
                                   percent_to_nn=percent_to_nn,
                                   mask_radius=mask_radius,
                                   change_sublattice=change_sublattice,
                                   filename=filename)

        #sub1_info_refined = print_sublattice_elements(sub1)

        ''' Sub2 '''
        image_difference_intensity(sublattice=sub2,
                                   simulation_image=simulation,
                                   element_list=element_list,
                                   percent_to_nn=percent_to_nn,
                                   mask_radius=mask_radius,
                                   change_sublattice=change_sublattice,
                                   filename=filename)

        #sub2_info_refined = print_sublattice_elements(sub2)

        ''' Sub3 '''
        image_difference_intensity(sublattice=sub3,
                                   simulation_image=simulation,
                                   element_list=element_list,
                                   percent_to_nn=percent_to_nn,
                                   mask_radius=mask_radius,
                                   change_sublattice=change_sublattice,
                                   filename=filename)

        #sub3_info_refined = print_sublattice_elements(sub3)
        counter_after_refinement = count_atoms_in_sublattice_list(
            sublattice_list=atom_lattice.sublattice_list,
            filename=intensity_refine_name + 'Elements' + saving_suffix)

        df_inten_refine = df_inten_refine.append(
            counter_after_refinement, ignore_index=True).fillna(0)

        compare_sublattices = compare_count_atoms_in_sublattice_list(
            counter_list=[counter_before_refinement, counter_after_refinement])

        if compare_sublattices is True:
            print('Finished Refinement! No more changes.')
            break

        if suffix > 4:
            if df_inten_refine.diff(periods=2)[-4:].all(axis=1).all() == False:
                # if statement steps above:
                # .diff(periods=2) gets the difference between each row, and the row two above it
                # [-4:] slices this new difference df to get the final four rows
                # .all(axis=1) checks if all row elements are zero or NaN and returns False
                # .all() check if all four of these results are False
                # Basically checking that the intensity refinement is repeating every second iteration
                print('Finished Refinement! Repeating every second iteration.')
                break

        ''' Remake XYZ file for further refinement'''
        # loading_suffix is now saving_suffix

        dataframe = create_dataframe_for_xyz(sublattice_list=atom_lattice.sublattice_list,
                                             element_list=element_list,
                                             x_distance=image_size_x_nm*10,
                                             y_distance=image_size_y_nm*10,
                                             z_distance=image_size_z_nm*10,
                                             filename=image_name + saving_suffix,
                                             header_comment='Something Something Something Dark Side')

        dataframe_intensity = create_dataframe_for_xyz(sublattice_list=atom_lattice.sublattice_list,
                                                       element_list=element_list,
                                                       x_distance=image_size_x_nm*10,
                                                       y_distance=image_size_y_nm*10,
                                                       z_distance=image_size_z_nm*10,
                                                       filename=intensity_refine_name + image_name + saving_suffix,
                                                       header_comment='Something Something Something Dark Side')

        # when ready:
        example_df_cif = create_dataframe_for_cif(
            sublattice_list=atom_lattice.sublattice_list, element_list=element_list)

        write_cif_from_dataframe(dataframe=example_df_cif,
                                 filename=intensity_refine_name + image_name + saving_suffix,
                                 chemical_name_common='MoSx-1Sex',
                                 cell_length_a=image_size_x_nm*10,
                                 cell_length_b=image_size_y_nm*10,
                                 cell_length_c=image_size_z_nm*10)

    df_inten_refine.to_pickle(intensity_refine_name + 'df_inten_refine.pkl')
    df_inten_refine.to_csv(intensity_refine_name +
                           'df_inten_refine.csv', sep=',', index=False)

    # https://python-graph-gallery.com/124-spaghetti-plot/
    # https://stackoverflow.com/questions/8931268/using-colormaps-to-set-color-of-line-in-matplotlib
    plt.figure()
    palette = plt.get_cmap('tab20')
    # plt.style.use('seaborn-darkgrid')
    # multiple line plot
    color_num = 0
    for df_column in df_inten_refine:
        #    print(df_column)
        color_num += 1
        plt.plot(df_inten_refine.index, df_inten_refine[df_column],
                 marker='', color=palette(color_num), linewidth=1, alpha=0.9,
                 label=df_column)

    plt.xlim(0, len(df_inten_refine.index)+1)
    plt.legend(loc=5, ncol=1, fontsize=10,
               fancybox=True, frameon=True, framealpha=1)
    plt.title("Refinement of Atoms via Intensity \nAll Elements",
              loc='left', fontsize=16, fontweight=0)
    plt.xlabel("Refinement Iteration", fontsize=16, fontweight=0)
    plt.ylabel("Count of Element", fontsize=16, fontweight=0)
    plt.tight_layout()
    plt.savefig(fname=intensity_refine_name + 'inten_refine_all.png',
                transparent=True, frameon=False, bbox_inches='tight',
                pad_inches=None, dpi=300, labels=False)
    plt.close()

    atom_of_interest = 'Mo_1'
    # Highlight Plot
    text_position = (len(df_inten_refine.index)+0.2)-1
    plt.figure()
    # plt.style.use('seaborn-darkgrid')
    # multiple line plot
    for df_column in df_inten_refine:
        plt.plot(df_inten_refine.index, df_inten_refine[df_column],
                 marker='', color='grey', linewidth=1, alpha=0.4)

    plt.plot(df_inten_refine.index,
             df_inten_refine[atom_of_interest], marker='', color='orange', linewidth=4, alpha=0.7)

    plt.xlim(0, len(df_inten_refine.index)+1)

    # Let's annotate the plot
    num = 0
    for i in df_inten_refine.values[len(df_inten_refine.index)-2][1:]:
        num += 1
        name = list(df_inten_refine)[num]
        if name != atom_of_interest:
            plt.text(text_position, i, name,
                     horizontalalignment='left', size='small', color='grey')

    plt.text(text_position, df_inten_refine.Mo_1.tail(1), 'Moly',
             horizontalalignment='left', size='medium', color='orange')

    plt.title("Refinement of Atoms via Intensity",
              loc='left', fontsize=16, fontweight=0)
    plt.xlabel("Refinement Iteration", fontsize=16, fontweight=0)
    plt.ylabel("Count of Element", fontsize=16, fontweight=0)
    plt.tight_layout()
    plt.savefig(fname=intensity_refine_name + 'inten_refine.png',
                transparent=True, frameon=False, bbox_inches='tight',
                pad_inches=None, dpi=300, labels=False)
    plt.close()

    ''' ATOM LATTICE with simulation refinement '''

    atom_lattice_int_ref_name = 'Atom_Lattice_' + \
        intensity_type + '_refined' + saving_suffix

    atom_lattice_int_ref = am.Atom_Lattice(image=atom_lattice_signal,
                                           name=atom_lattice_int_ref_name,
                                           sublattice_list=atom_lattice.sublattice_list)
    atom_lattice_int_ref.save(filename=intensity_refine_name +
                              atom_lattice_int_ref_name + ".hdf5", overwrite=True)
    atom_lattice_int_ref.save(
        filename=atom_lattice_int_ref_name + "_intensity.hdf5", overwrite=True)

    atom_lattice_int_ref.get_sublattice_atom_list_on_image(markersize=2).plot()
    plt.title(atom_lattice_int_ref_name, fontsize=20)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(fname=intensity_refine_name + atom_lattice_int_ref_name + '.png',
                transparent=True, frameon=False, bbox_inches='tight',
                pad_inches=None, dpi=300, labels=False)
    plt.close()

    t = time()-t0

    create_new_folder('./' + folder_name + '/')
    intensity_refine_filenames = glob('*' + intensity_refine_name + '*')
    for intensity_refine_file in intensity_refine_filenames:
        #    print(position_refine_file, position_refine_name + '/' + position_refine_file)
        os.rename(intensity_refine_file, folder_name +
                  '/' + intensity_refine_file)


# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:33:48 2019

@author: Eoghan.OConnell
"""

warnings.simplefilter("ignore", UserWarning)

'''
directory = 'G:/SuperStem visit/Feb 2019 data/2019_02_18_QMR_S1262_MoS2-Se-10eV/005_rigid_reg/Try_5/32bit/images_aligned_0' 
os.chdir(directory)
image_refine_via_position_loop(atom_lattice_name='Atom_Lattice_total.hdf5',
                               add_sublattice=True,
                                   plot_details=False,
                                   intensity_type='total',
                                   pixel_threshold=15,
                                   position_refine_name = 'position_refine_',
                                   folder_name = 'refinement_of_position')
'''


def image_refine_via_position_loop(image,
                                   sublattice_list,
                                   filename,
                                   xyz_filename,
                                   add_sublattice,
                                   pixel_threshold,
                                   num_peaks,
                                   image_size_x_nm,
                                   image_size_y_nm,
                                   image_size_z_nm,
                                   calibration_area,
                                   calibration_separation,
                                   element_list,
                                   element_list_new_sub,
                                   middle_intensity_list,
                                   limit_intensity_list,
                                   delta_image_filter=0.5,
                                   intensity_type='max',
                                   method='mode',
                                   remove_background_method=None,
                                   background_sublattice=None,
                                   num_points=3,
                                   percent_to_nn=0.4,
                                   mask_radius=None,
                                   iterations=10,
                                   max_sigma=10,
                                   E0=60e3,
                                   integrationAngleMin=0.085,
                                   integrationAngleMax=0.186,
                                   interpolationFactor=16,
                                   realspacePixelSize=0.0654,
                                   numFP=1,
                                   probeSemiangle=0.030,
                                   alphaBeamMax=0.032,
                                   scanWindowMin=0.0,
                                   scanWindowMax=1.0,
                                   algorithm="prism",
                                   numThreads=2
                                   ):
    ''' Image Position Loop '''

    df_position_refine = pd.DataFrame(columns=element_list)
    new_subs = []

    create_dataframe_for_xyz(
        sublattice_list=sublattice_list,
        element_list=element_list,
        x_distance=image_size_x_nm*10,
        y_distance=image_size_y_nm*10,
        z_distance=image_size_z_nm*10,
        filename=xyz_filename + '01',
        header_comment=filename)

    for suffix in range(1, iterations):

        loading_suffix = '_' + str(suffix).zfill(2)
        saving_suffix = '_' + str(suffix+1).zfill(2)
        simulation_filename = xyz_filename + loading_suffix + '.XYZ'

        simulate_and_calibrate_with_prismatic(reference_image=image,
                                              xyz_filename=simulation_filename,
                                              calibration_area=calibration_area,
                                              calibration_separation=calibration_separation,
                                              filename=filename,
                                              percent_to_nn=percent_to_nn,
                                              mask_radius=mask_radius,
                                              E0=E0,
                                              integrationAngleMin=integrationAngleMin,
                                              integrationAngleMax=integrationAngleMax,
                                              interpolationFactor=interpolationFactor,
                                              realspacePixelSize=realspacePixelSize,
                                              numFP=numFP,
                                              probeSemiangle=probeSemiangle,
                                              alphaBeamMax=alphaBeamMax,
                                              scanWindowMin=scanWindowMin,
                                              scanWindowMax=scanWindowMax,
                                              algorithm=algorithm,
                                              numThreads=numThreads)

        # Filter the image with Gaussian noise to get better match with experiment
        simulation_new = compare_image_to_filtered_image(
            image_to_filter=simulation,
            reference_image=image,
            filename=filename + loading_suffix,
            delta_image_filter=delta_image_filter,
            cropping_area=calibration_area,
            separation=calibration_separation,
            percent_to_nn=percent_to_nn,
            mask_radius=mask_radius,
            max_sigma=max_sigma,
            refine=False)

        simulation = simulation_new

        simulation.save('filt_sim_' + filename + loading_suffix + '.hspy')

        # simulation.plot()
        # plt.title('Filtered_Simulation' + filename +
        #           loading_suffix, fontsize=20)
        # plt.gca().axes.get_xaxis().set_visible(False)
        # plt.gca().axes.get_yaxis().set_visible(False)
        # plt.tight_layout()
        # plt.savefig(fname='filt_sim_' + filename + loading_suffix + '.png',
        #             transparent=True, frameon=False, bbox_inches='tight',
        #             pad_inches=None, dpi=300, labels=False)
        # plt.close()

        counter_before_refinement = count_atoms_in_sublattice_list(
            sublattice_list=sublattice_list,
            filename=None)

        if suffix == 1:
            df_position_refine = df_position_refine.append(
                counter_before_refinement, ignore_index=True).fillna(0)
        else:
            pass

        sub_new = image_difference_position(sublattice_list=sublattice_list,
                                            simulation_image=simulation,
                                            pixel_threshold=pixel_threshold,
                                            filename=None,
                                            mask_radius=mask_radius,
                                            num_peaks=num_peaks,
                                            add_sublattice=add_sublattice,
                                            sublattice_name='sub_new' + loading_suffix)

        if type(sub_new) == type(sublattice_list[0]):

            new_subs.append(sub_new)
            sublattice_list += [new_subs[-1]]

            sub_new.get_atom_list_on_image(markersize=2).plot()
            plt.title(sub_new.name, fontsize=20)
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.tight_layout()
            plt.savefig(fname=sub_new.name + filename + '.png',
                        transparent=True, frameon=False, bbox_inches='tight',
                        pad_inches=None, dpi=300, labels=False)
            plt.close()

            sort_sublattice_intensities(sublattice=sub_new,
                                        intensity_type=intensity_type,
                                        middle_intensity_list=middle_intensity_list,
                                        limit_intensity_list=limit_intensity_list,
                                        element_list=element_list_new_sub,
                                        method=method,
                                        remove_background_method=remove_background_method,
                                        background_sublattice=background_sublattice,
                                        num_points=num_points,
                                        percent_to_nn=percent_to_nn,
                                        mask_radius=mask_radius)
            '''
            need to make mask_radius for this too, for the new sublattice
            '''

            assign_z_height(sublattice=sub_new,
                            lattice_type='chalcogen',
                            material='mos2_one_layer')

    #        sub_new_info = print_sublattice_elements(sub_new)

        elif sub_new is None:
            print('All new sublattices have been added!')
            break

        counter_after_refinement = count_atoms_in_sublattice_list(
            sublattice_list=sublattice_list,
            filename=None)

        df_position_refine = df_position_refine.append(
            counter_after_refinement, ignore_index=True).fillna(0)

        ''' Remake XYZ file for further refinement'''
        # loading_suffix is now saving_suffix

        create_dataframe_for_xyz(
            sublattice_list=sublattice_list,
            element_list=element_list,
            x_distance=image_size_x_nm*10,
            y_distance=image_size_y_nm*10,
            z_distance=image_size_z_nm*10,
            filename=xyz_filename + filename + saving_suffix,
            header_comment=filename)

    df_position_refine.to_csv(filename + '.csv', sep=',', index=False)

    '''Save Atom Lattice Object'''
    atom_lattice = am.Atom_Lattice(image=image.data,
                                   name='All Sublattices ' + filename,
                                   sublattice_list=sublattice_list)
    atom_lattice.save(filename="Atom_Lattice_" +
                      filename + ".hdf5", overwrite=True)

    folder_name = filename + "_pos_ref_data"
    create_new_folder('./' + folder_name + '/')
    position_refine_filenames = glob('*' + filename + '*')
    for position_refine_file in position_refine_filenames:
        os.rename(position_refine_file, folder_name +
                  '/' + position_refine_file)


#element_config = 'Mo_2.S_4'


def count_element_in_pandas_df(element, dataframe):
    '''
    Count the number of a single element in a dataframe

    Parameters
    ----------

    element : string 
        element symbol

    dataframe : pandas dataframe
        The dataframe must have column headers as elements or element 
        configurations

    Returns
    -------

    Counter object

    Examples
    --------

    >>> Mo_count = count_element_in_pandas_df(element='Mo', dataframe=df)

    '''
    count_of_element = Counter()

    for element_config in dataframe.columns:
        #        print(element_config)
        if element in element_config:
            split_element = split_and_sort_element(element_config)

            for split in split_element:
                # split=split_element[1]
                if element in split[1]:
                    #                    print(element + ":" + str(split[2]*dataframe.loc[:, element_config]))
                    count_of_element += split[2] * \
                        dataframe.loc[:, element_config]

    return(count_of_element)


def count_all_individual_elements(individual_element_list, dataframe):
    '''
    Perform count_element_in_pandas_df() for all elements in a dataframe.
    Specify the elements you wish to count in the individual_element_list.

    Parameters
    ----------

    individual_element_list : list

    dataframe : pandas dataframe
        The dataframe must have column headers as elements or element 
        configurations

    Returns
    -------

    dict object with each individual element as dict.key and their 
    count as dict.value

    Examples
    --------

    >>> individual_element_list = ['Mo', 'S', 'Se']
    >>> element_count = count_all_individual_elements(individual_element_list, dataframe=df)
    >>> element_count
    '''

    element_count_dict = {}

    for element in individual_element_list:

        element_count = count_element_in_pandas_df(
            element=element, dataframe=dataframe)

        element_count_dict[element] = element_count

    return(element_count_dict)


def count_atoms_in_sublattice_list(sublattice_list, filename=None):
    '''
    Count the elements in a list of Atomap sublattices

    Parameters
    ----------

    sublattice_list : list of atomap sublattice(s)

    filename : string, default None
        name with which the image will be saved

    Returns
    -------

    Counter object

    Examples
    --------

    >>> from collections import Counter
    >>> import atomap.api as am
    >>> import matplotlib.pyplot as plt
    >>> atom_lattice = am.dummy_data.get_simple_atom_lattice_two_sublattices()
    >>> sub1 = atom_lattice.sublattice_list[0]
    >>> sub2 = atom_lattice.sublattice_list[1]

    >>> for i in range(0, len(sub1.atom_list)):
    >>>     sub1.atom_list[i].elements = 'Ti_2'
    >>> for i in range(0, len(sub2.atom_list)):
    >>>     sub2.atom_list[i].elements = 'Cl_1'

    >>> added_atoms = count_atoms_in_sublattice_list(
    >>>     sublattice_list=[sub1, sub2],
    >>>     filename=atom_lattice.name)

    Compare before and after
    >>> atom_lattice_before = am.dummy_data.get_simple_atom_lattice_two_sublattices()
    >>> no_added_atoms = count_atoms_in_sublattice_list(
    >>>     sublattice_list=atom_lattice_before.sublattice_list,
    >>>     filename=atom_lattice_before.name)

    >>> if added_atoms == no_added_atoms:
    >>>     print('They are the same, you can stop refining')
    >>> else:
    >>>     print('They are not the same!')

    '''
    count_of_sublattice = Counter()
    for sublattice in sublattice_list:

        sublattice_info = print_sublattice_elements(sublattice)
        elements_in_sublattice = [atoms[0:1]
                                  for atoms in sublattice_info]  # get just chemical info
        elements_in_sublattice = [
            y for x in elements_in_sublattice for y in x]  # flatten to a list
        # count each element
        count_of_sublattice += Counter(elements_in_sublattice)

        # count_of_sublattice.most_common()

    if filename is not None:
        plt.figure()
        plt.scatter(x=count_of_sublattice.keys(),
                    y=count_of_sublattice.values())
        plt.title('Elements in ' + filename, fontsize=16)
        plt.xlabel('Elements', fontsize=16)
        plt.ylabel('Count of Elements', fontsize=16)
        plt.tight_layout()
        plt.savefig(fname='element_count_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)
        plt.close()
    else:
        pass

    return(count_of_sublattice)


def compare_count_atoms_in_sublattice_list(counter_list):
    '''

    Compare the count of atomap elements in two counter_lists gotten by
    count_atoms_in_sublattice_list()

    If the counters are the same, then the original atom_lattice is the
    same as the refined atom_lattice. It will return the boolean value
    True. This can be used to  stop refinement loops if neccessary.

    Parameters
    ----------

    counter_list : list of two Counter objects

    Returns
    -------

    Boolean True if the counters are equal,
    boolean False is the counters are not equal.

    Examples
    --------

    >>> from collections import Counter
    >>> import atomap.api as am
    >>> import matplotlib.pyplot as plt
    >>> atom_lattice = am.dummy_data.get_simple_atom_lattice_two_sublattices()
    >>> sub1 = atom_lattice.sublattice_list[0]
    >>> sub2 = atom_lattice.sublattice_list[1]

    >>> for i in range(0, len(sub1.atom_list)):
    >>>     sub1.atom_list[i].elements = 'Ti_2'
    >>> for i in range(0, len(sub2.atom_list)):
    >>>     sub2.atom_list[i].elements = 'Cl_1'

    >>> added_atoms = count_atoms_in_sublattice_list(
    >>>     sublattice_list=[sub1, sub2],
    >>>     filename=atom_lattice.name)

    >>> atom_lattice_before = am.dummy_data.get_simple_atom_lattice_two_sublattices()
    >>> no_added_atoms = count_atoms_in_sublattice_list(
    >>>     sublattice_list=atom_lattice_before.sublattice_list,
    >>>     filename=atom_lattice_before.name)

    >>> compare_count_atoms_in_sublattice_list([added_atoms, no_added_atoms])

    # To stop a refinement loop
    # >>> if compare_count_atoms_in_sublattice_list(counter_list) is True:
    # >>>    break
    '''
    if len(counter_list) == 2:

        counter0 = counter_list[0]
        counter1 = counter_list[1]
    else:
        raise ValueError('len(counter_list) must be 2')

    return True if counter0 == counter1 else False


# Remove spikes
#directory = os.chdir('G:/SuperStem visit/Feb 2019 data/2019_02_18_QMR_S1262_MoS2-Se-10eV/005_rigid_reg/Try_5')
#file = '005_HAADF_movie_351_4nm_MC.dm3'
# s_series, real_sampling = load_data_and_sampling(filename='005_HAADF_movie_351_4nm_MC',
#          file_extension='.dm3',
#          invert_image=False, save_image=False)
#dm3_stack_to_tiff_stack(filename=file, crop=True, crop_start=20.0, crop_end=40.0)


def rigid_registration(file, masktype='hann', n=4, findMaxima='gf'):
    ''' 
    Perform image registraion with the rigid registration package

    Parameters
    ----------

    file : stack of tiff images

    masktype : filtering method, default 'hann'
        See https://github.com/bsavitzky/rigidRegistration for 
        more information

    n : width of filter, default 4
        larger numbers mean smaller filter width
        See https://github.com/bsavitzky/rigidRegistration for 
        more information

    findMaxima : image matching method, default 'gf'
        'pixel' and 'gf' options, See 
        https://github.com/bsavitzky/rigidRegistration for 
        more information

    Returns
    -------
    Outputs of
    report of the image registration
    aligned and stacked image with and without crop
    creates a folder and places all uncropped aligned images in it


    Examples
    --------

    >>>


    '''

    # Read tiff file. Rearrange axes so final axis iterates over images
    stack = np.rollaxis(imread(file), 0, 3)
    stack = stack[:, :, :]/float(2**16)        # Normalize data between 0 and 1

    s = rigidregistration.stackregistration.imstack(stack)
    s.getFFTs()

    # Choose Mask and cutoff frequency
    s.makeFourierMask(mask=masktype, n=n)     # Set the selected Fourier mask
    # s.show_Fourier_mask(i=0,j=5)             # Display the results

    # Calculate image shifts using gaussian fitting
    findMaxima = findMaxima
    s.setGaussianFitParams(num_peaks=3, sigma_guess=3, window_radius=4)

    # Find shifts.  Set verbose=True to print the correlation status to screen
    s.findImageShifts(findMaxima=findMaxima, verbose=False)

    # Identify outliers using nearest neighbors to enforce "smoothness"
    s.set_nz(0, s.nz)
    s.get_outliers_NN(max_shift=8)
    # s.show_Rij(mask=True)

    s.make_corrected_Rij()    # Correct outliers using the transitivity relations
    # s.show_Rij_c()            # Display the corrected shift matrix

    # Create registered image stack and average
    # To skip calculation of image shifts, or correcting the shift matrix, pass the function
    s.get_averaged_image()
    # get_shifts=False, or correct_Rij=False

    s.get_all_aligned_images()
    # s.show()

    # Display report of registration procedure
    # s.show_report()

    # Save report of registration procedure
    s.save_report("registration_report.pdf")

    # Save the average image
    s.save("average_image.tif")

    # Save the average image, including outer areas. Be careful when analysis outer regions of this file
    s.save("average_image_no_crop.tif", crop=False)

    # creates a folder and put all the individual images in there
    save_individual_images(image_stack=s.stack_registered)


def simulate_and_calibrate_with_prismatic(
        xyz_filename,
        filename,
        reference_image,
        calibration_area,
        calibration_separation,
        percent_to_nn=0.4,
        mask_radius=None,
        refine=True,
        scalebar_true=False,
        probeStep=None,
        E0=60e3,
        integrationAngleMin=0.085,
        integrationAngleMax=0.186,
        interpolationFactor=16,
        realspacePixelSize=0.0654,
        numFP=1,
        probeSemiangle=0.030,
        alphaBeamMax=0.032,
        scanWindowMin=0.0,
        scanWindowMax=1.0,
        algorithm="prism",
        numThreads=2):
    ''' 
    Simulate an xyz coordinate model with pyprismatic
    fast simulation software.

    Parameters
    ----------

    xyz_filename : string
        filename of the xyz coordinate model. Must be in the prismatic format.
        See http://prism-em.com/docs-inputs/ for more information.
    filename : string, default None
        name with which the image will be saved
    reference_image : hyperspy signal 2D
        image from which calibration information is taken, such
        as sampling, pixel height and pixel width
    calibration_area : list
        xy pixel coordinates of the image area to be used to calibrate 
        the intensity. In the form [[0,0], [512,512]]
        See calibrate_intensity_distance_with_sublattice_roi()
    calibration_separation : int
        pixel separation used for the intensity calibration. 
        See calibrate_intensity_distance_with_sublattice_roi()
    percent_to_nn : float, default 0.40
        Determines the boundary of the area surrounding each atomic 
        column, as fraction of the distance to the nearest neighbour.
    mask_radius : float, default None
            Radius of the mask around each atom. If this is not set,
            the radius will be the distance to the nearest atom in the
            same sublattice times the `percent_to_nn` value.
            Note: if `mask_radius` is not specified, the Atom_Position objects
            must have a populated nearest_neighbor_list.
    refine, scalebar_true
        See function calibrate_intensity_distance_with_sublattice_roi()
    probeStep, E0 ... etc.
        See function simulate_with_prismatic()


    Returns
    -------
    Simulated image as a hyperspy object

    Examples
    --------

    >>> simulate_and_calibrate_with_prismatic()
    ######## need to include an example reference_image here

    '''

    if len(calibration_area) != 2:
        raise ValueError('calibration_area must be two points')

    simulate_with_prismatic(xyz_filename=xyz_filename,
                            filename=filename,
                            reference_image=reference_image,
                            probeStep=probeStep,
                            E0=E0,
                            integrationAngleMin=integrationAngleMin,
                            integrationAngleMax=integrationAngleMax,
                            interpolationFactor=interpolationFactor,
                            realspacePixelSize=realspacePixelSize,
                            numFP=numFP,
                            probeSemiangle=probeSemiangle,
                            alphaBeamMax=alphaBeamMax,
                            scanWindowMin=scanWindowMin,
                            scanWindowMax=scanWindowMax,
                            algorithm=algorithm,
                            numThreads=numThreads)

    simulation = hs.load('prism_2Doutput_' + filename + '.mrc')
    simulation.axes_manager[0].name = 'extra_dimension'
    simulation = simulation.sum('extra_dimension')
    # simulation.axes_manager[0].scale = real_sampling
    # simulation.axes_manager[1].scale = real_sampling

    calibrate_intensity_distance_with_sublattice_roi(
        image=simulation,
        cropping_area=calibration_area,
        separation=calibration_separation,
        filename=filename,
        reference_image=reference_image,
        percent_to_nn=percent_to_nn,
        mask_radius=mask_radius,
        refine=refine,
        scalebar_true=scalebar_true)

    return(simulation)


'''
os.chdir('C:/Users/Eoghan.OConnell/Documents/Documents/Eoghan UL/PHD/Python Files/scripts/Functions/private_development_git/private_development')
print(os.getcwd())
'''


def simulate_with_prismatic(xyz_filename,
                            filename,
                            reference_image=None,
                            probeStep=1.0,
                            E0=60e3,
                            integrationAngleMin=0.085,
                            integrationAngleMax=0.186,
                            interpolationFactor=16,
                            realspacePixelSize=0.0654,
                            numFP=1,
                            cellDimXYZ=None,
                            tileXYZ=None,
                            probeSemiangle=0.030,
                            alphaBeamMax=0.032,
                            scanWindowMin=0.0,
                            scanWindowMax=1.0,
                            algorithm="prism",
                            numThreads=2):
    ''' 
    Simulate an xyz coordinate model with pyprismatic
    fast simulation software.

    Parameters
    ----------

    xyz_filename : string
        filename of the xyz coordinate model. Must be in the prismatic format.
        See http://prism-em.com/docs-inputs/ for more information.
    filename : string, default None
        name with which the image will be saved
    reference_image : hyperspy signal 2D
        image from which calibration information is taken, such
        as sampling, pixel height and pixel width
    probeStep : float, default 1.0
        Should be the sampling of the image, where
        sampling = length (in angstrom)/pixels
        If you want the probeStep to be calculated from the reference image,
        set this to None.
    E0, numThreads etc.,  : Prismatic parameters
        See http://prism-em.com/docs-params/
    cellDimXYZ : tuple, default None
        A tuple of length 3. Example (2.3, 4.5, 5.6).
        If this is set to None, the cell dimension values from the .xyz file 
        will be used (default). If it is specified, it will overwrite the .xyz 
        file values.
    tileXYZ : tuple, deault None
        A tuple of length 3. Example (5, 5, 2) would multiply the model in x 
        and y by 5, and z by 2.
        Default of None is just set to (1, 1, 1)

    Returns
    -------
    Simulated image as a 2D mrc file

    Examples
    --------

    >>> simulate_with_prismatic(
            xyz_filename='MoS2_hex_prismatic.xyz',
            filename='prismatic_simulation',
            probeStep=1.0,
            reference_image=None,
            E0=60e3,
            integrationAngleMin=0.085,
            integrationAngleMax=0.186,
            interpolationFactor=16,
            realspacePixelSize=0.0654,
            numFP=1,
            probeSemiangle=0.030,
            alphaBeamMax=0.032,
            scanWindowMin=0.0,
            scanWindowMax=1.0,
            algorithm="prism",
            numThreads=2)

    '''
    print("Simulating: " + simulation_filename)

    if '.xyz' not in xyz_filename:
        simulation_filename = xyz_filename + '.XYZ'
    else:
        simulation_filename = xyz_filename
    print("Simulating: " + simulation_filename)
    file_exists = os.path.isfile(simulation_filename)
    if file_exists:
        pass
    else:
        raise OSError('XYZ file not found in directory, stopping refinement')

    # param inputs, feel free to add more!!

    pr_sim = pr.Metadata(filenameAtoms=simulation_filename)
    print("Simulating: " + simulation_filename)
    # use the reference image to get the probe step if given

    if reference_image is None and probeStep is None:
        raise ValueError("Both reference_image and probeStep are None.\
            Either choose a reference image, from which a probe step can\
            be calculated, or choose a probeStep.")
    elif reference_image is not None and probeStep is not None:
        print("Note: Both reference_image and probeStep have been specified.\
            .. probeStep will be used.")

    if reference_image is not None:
        real_sampling = reference_image.axes_manager[0].scale
        real_sampling_exp_angs = real_sampling*10
        if str(real_sampling_exp_angs)[-1] == '5':
            real_sampling_sim_angs = real_sampling_exp_angs + 0.000005
        pr_sim.probeStepX = pr_sim.probeStepY = round(
            real_sampling_sim_angs, 6)
    else:
        pr_sim.probeStepX = pr_sim.probeStepY = probeStep

    # if you specify cellDimXYZ, you overwrite the values from the xyz file
    if cellDimXYZ is not None:
        pr_sim.cellDimX, pr_sim.cellDimX, pr_sim.cellDimX = cellDimXYZ
    if tileXYZ is not None:
        pr_sim.tileX, pr_sim.tileY, pr_sim.tileZ = tileXYZ

    #    pr_sim.probeStepX = pr_sim.cellDimX/atom_lattice_data.shape[1]
    #    pr_sim.probeStepY = pr_sim.cellDimY/atom_lattice_data.shape[0]
    pr_sim.detectorAngleStep = 0.001
    pr_sim.save3DOutput = False

    pr_sim.E0 = E0
    pr_sim.integrationAngleMin = integrationAngleMin
    pr_sim.integrationAngleMax = integrationAngleMax
    pr_sim.interpolationFactorX = pr_sim.interpolationFactorY = interpolationFactor
    pr_sim.realspacePixelSizeX = pr_sim.realspacePixelSizeY = realspacePixelSize
    pr_sim.numFP = numFP
    pr_sim.probeSemiangle = probeSemiangle
    pr_sim.alphaBeamMax = alphaBeamMax  # in rads
    pr_sim.scanWindowXMin = pr_sim.scanWindowYMin = scanWindowMin
    pr_sim.scanWindowYMax = pr_sim.scanWindowXMax = scanWindowMax
    pr_sim.algorithm = algorithm
    pr_sim.numThreads = numThreads
    pr_sim.filenameOutput = filename + '.mrc'
    print("Simulating: " + simulation_filename)
    pr_sim.go()


def load_prismatic_mrc_with_hyperspy(
        prismatic_mrc_filename,
        save_name='calibrated_data_'):
    ''' 
    Open a prismatic .mrc file and save as a hyperspy object.
    Also plots save saves a png.

    Parameters
    ----------

    prismatic_mrc_filename : string
        name of the outputted prismatic .mrc file.

    Returns
    -------
    Hyperspy Signal 2D of the simulation

    Examples
    --------

    >>> load_prismatic_mrc_with_hyperspy(
                prismatic_mrc_filename='prism_2Doutput_prismatic_simulation.mrc',
                save_name='calibrated_data_')

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


# refine with and plot gaussian fitting
def get_xydata_from_sublattice_intensities(
        sublattice_intensity_list,
        hist_bins=100):
    '''
    Output x and y data for a histogram of intensities

    Parameters
    ----------
    sublattice_intensity_list : list
        See get_subattice_intensity() for more information
    hist_bins : int, default 100
        number of bins to sort the intensities into
        must be a better way of doing this? maybe automate the binning choice

    Returns
    -------
    Two numpy 1D arrays corressponding to the x and y values of a histogram
    of the sublattice intensities.

    Examples
    --------

    >>> amp, mu, sigma = 10, 10, 0.5
    >>> sub1_inten = np.random.normal(mu, sigma, 1000)
    >>> xdata, ydata = get_xydata_from_sublattice_intensities(
                            sub1_inten, hist_bins=50)
    '''

    hist_bins = hist_bins
    y_array, x_array = np.histogram(sublattice_intensity_list,
                                    bins=hist_bins)
    x_array = np.delete(x_array, [0])

    # set the x_ values so that they are at the middle of each hist bin
    x_separation = (x_array.max()-x_array.min())/hist_bins
    x_array = x_array - (x_separation/2)

    return(x_array, y_array)


# 1D single Gaussian
def _1Dgaussian(xdata, amp, mu, sigma):
    '''
    Fitting function for a single 1D gaussian distribution

    Parameters
    ----------
    xdata : numpy 1D array
        values input as the x coordinates of the gaussian distribution
    amp : float
        amplitude of the gaussian in y-axis
    mu : float
        mean value of the gaussianin x-axis, corresponding to y-axis 
        amplitude.
    sigma : float
        standard deviation of the gaussian distribution

    Returns
    -------
    gaussian distibution of xdata array

    Examples
    --------

    >>> amp, mu, sigma = 10, 10, 0.5
    >>> sub1_inten = np.random.normal(mu, sigma, 1000)
    >>> xdata, ydata = get_xydata_from_sublattice_intensities(
                            sub1_inten, hist_bins=50)
    >>> gauss_fit_01 = _1Dgaussian(xdata, amp, mu, sigma)    
    '''

    return amp*(1/(sigma*(np.sqrt(2*np.pi))))*(np.exp(-((xdata-mu)**2)/((2*sigma)**2)))


# Fit gaussian to element
def refine_element_histogram_intensity_with_gaussian(
        function,
        xdata,
        ydata,
        amp, mu, sigma):
    '''
    Use the initially found centre (mean/mode) value of a sublattice
    histogram (e.g., Mo_1 in an Mo sublattice) as an input mean for a 
    gaussian fit of the data. 

    Parameters
    ----------
    xdata, ydata : see scipy.optimize.curve_fit
    amp, mu, sigma : see _1Dgaussian() for more details

    Returns
    -------
    optimised parameters (popt) and estimated covariance (pcov) of the 
    fitted gaussian function.

    Examples
    --------

    >>> amp, mu, sigma = 10, 10, 0.5
    >>> sub1_inten = np.random.normal(mu, sigma, 1000)
    >>> xdata, ydata = get_xydata_from_sublattice_intensities(
                            sub1_inten, hist_bins=50)
    >>> popt_gauss, _ = refine_element_histogram_intensity_with_gaussian(
                            function=_1Dgaussian,
                            xdata=xdata,
                            ydata=ydata,
                            p0=[amp, mu, sigma])
    >>> print("calculated mean: " + str(round(np.mean(xdata),3)) + "\n"
              + "fitted mean: " + str(round(popt_gauss[1],3)))

    '''

    popt_gauss, pcov_gauss = scipy.optimize.curve_fit(
        f=function,
        xdata=xdata,
        ydata=ydata,
        p0=[amp, mu, sigma])
    # p0 = [amp, mu, sigma]

    return(popt_gauss, pcov_gauss)


# plot single gauss fit

def plot_gaussian_fit(xdata, ydata, function, amp, mu, sigma,
                      gauss_art='r--', gauss_label='Gauss Fit',
                      plot_data=True,
                      data_art='ko', data_label='Data Points',
                      plot_fill=False,
                      facecolor='r', alpha=0.5):
    # save_image/filename maybe?
    '''
    >>> amp, mu, sigma = 10, 10, 0.5
    >>> sub1_inten = np.random.normal(mu, sigma, 1000)
    >>> xdata, ydata = get_xydata_from_sublattice_intensities(
                            sub1_inten, hist_bins=50)
    >>> popt_gauss, _ = refine_element_histogram_intensity_with_gaussian(
                            function=_1Dgaussian,
                            xdata=xdata,
                            ydata=ydata,
                            p0=[amp, mu, sigma])
    >>> plot_gaussian_fit(xdata, ydata, function=_1Dgaussian, 
                  amp=popt_gauss[0], mu=popt_gauss[1], sigma=popt_gauss[2],
                  gauss_art='r--', gauss_label='Gauss Fit',
                  plot_data=True, data_art='ko', data_label='Data Points',
                  plot_fill=True, facecolor='r', alpha=0.5)
    '''

    _gaussian_fit = function(xdata=xdata, amp=amp, mu=mu, sigma=sigma)

    fig = plt.figure(figsize=(4, 3))
    gs = gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(xdata, _gaussian_fit, gauss_art, label=gauss_label)

    if plot_data:
        ax1.plot(xdata, ydata, data_art, label=data_label)
    if plot_fill:
        ax1.fill_between(xdata, _gaussian_fit.min(),
                         _gaussian_fit,
                         facecolor=facecolor,
                         alpha=alpha)
    plt.xlabel("Intensity (a.u.)")
    plt.ylabel("Count")
    plt.title("Gaussian Fit")  # filename input here
    plt.legend(loc='upper right')
    plt.tight_layout()


def plot_gauss_fit_only(f, array, popt_gauss):

    ax1.plot(x, f(x, *popt_gauss), 'k--', label='Fit')


def plot_gaussian_fitting_all_elements(sub_ints_all,
                                       fitting_tools_all_subs,
                                       element_list_all_subs,
                                       marker_list,
                                       hist_bins,
                                       filename):

    # set up cyclers for plotting gaussian fits
    cycler_sub1 = plt.cycler(c=['lightblue', 'blue', 'darkviolet', 'violet', 'navy', 'darkslateblue'],
                             linestyle=[':', '--', '-.', '-', ':', '--'])
    cycler_sub2 = plt.cycler(c=['wheat', 'gold', 'forestgreen', 'greenyellow', 'darkgreen', 'darkolivegreen', 'y'],
                             linestyle=[':', '--', '-.', '-', ':', '--', '-.'])

    cyclers_all = [cycler_sub1, cycler_sub2]

    if len(cyclers_all) != len(element_list_all_subs) != len(fitting_tools_all_subs):
        raise ValueError(
            "len(cyclers_all) != len(element_list_all) != len(fitting_tools_all_subs)")

    for cycler, element, fitting in zip(cyclers_all, element_list_all_subs, fitting_tools_all_subs):
        if len(cycler) != len(element) != len(fitting):
            raise ValueError("len(cycler) != len(element) != len(fitting)")

    plt.rcParams.update({'xtick.labelsize': 'x-large',
                         'ytick.labelsize': 'x-large'})
    plt.rc('font', family='Arial')

    fig, (ax1, ax2) = plt.subplots(figsize=(16, 9), nrows=2, sharex=True,
                                   gridspec_kw={'height_ratios': [2, 0.5]})
    plt.subplots_adjust(hspace=0)

    #fig.suptitle("Fit of all Elements with Residuals", family="serif", fontsize=20)
    ax2.set_xlabel("Intensity (a.u.)", family="serif",  fontsize=20)
    ax1.set_ylabel("Counts", family="serif",  fontsize=20)
    ax2.set_ylabel("Res.", family="serif",  fontsize=20)

    sub_residual_gauss_list = []
    for sublattice_array, fitting_tools_sub, cycler_sub, marker, in zip(sub_ints_all, fitting_tools_all_subs, cyclers_all, marker_list):
        array = get_2d_distribution_from_sublattice_intensities(sublattice_array,
                                                                hist_bins=hist_bins)

        x_array = array[:, 0]
        y_array = array[:, 1]
        sub_data = ax1.plot(x_array, y_array, color='grey',
                            label=marker[0] + ' Data',
                            marker=marker[1],
                            linestyle='',
                            markersize=4,
                            alpha=0.75)

        for fitting_tools, kwargs in zip(fitting_tools_sub, cycler_sub):
            sliced_array = []
            for atom_int, atom_count in zip(x_array, y_array):
                if fitting_tools[2] < atom_int < fitting_tools[3]:
                    sliced_array.append([atom_int, atom_count])
            sliced_array = np.array(sliced_array)
            if sliced_array.size != 0:
                x = sliced_array[:, 0]
                y = sliced_array[:, 1]

                try:
                    popt_gauss, pcov_gauss = scipy.optimize.curve_fit(
                        f=_1gaussian,
                        xdata=x,
                        ydata=y,
                        p0=[fitting_tools[4], fitting_tools[5],
                            fitting_tools[6]])
                    individual_gauss = _1gaussian(x, *popt_gauss)
                    sub_gauss = ax1.plot(x, individual_gauss, **kwargs)
                    sub_gauss_fill = ax1.fill_between(x,
                                                      individual_gauss.min(),
                                                      individual_gauss,
                                                      facecolor=kwargs['c'],
                                                      alpha=0.5)

                    sub_residual_gauss = abs(y - (_1gaussian(x, *popt_gauss)))
                    sub_gauss_hl = ax1.plot(x, _1gaussian(x, *popt_gauss),
                                            label=r"$\bf{" + fitting_tools[0] + "}$" + ': ' +
                                            str(round(
                                                sum(abs(sub_residual_gauss)), 1)),
                                            linewidth=1.5,
                                            **kwargs)

                    sub_residual_gauss_list.append([fitting_tools[0],
                                                    sub_residual_gauss])
                    sub_resid = ax2.plot(x, sub_residual_gauss, marker=marker[1],
                                         color='grey',
                                         linestyle='',
                                         markersize=4,
                                         alpha=0.75,
                                         label=fitting_tools[0] + ': ' +
                                         str(round(sum(abs(sub_residual_gauss)), 1)))

                except OptimizeWarning:
                    print("Warning - Covariance could not be estimated for " +
                          fitting_tools[0] + ", skipping...")
                except RuntimeError:
                    print("Error - curve_fit failed for " +
                          fitting_tools[0] + ", skipping...")

    legend1 = ax1.legend(
        loc="best", prop={'size': 10}, ncol=2, edgecolor='grey')
    for line in legend1.get_lines():
        line.set_linewidth(1.5)

    #ax1.hist(sub1_ints, bins=500)
    #ax1.hist(sub2_ints, bins=500)
    if filename is not None:
        plt.savefig(fname=filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=900, labels=False)


def convert_vesta_xyz_to_prismatic_xyz(vesta_xyz_filename,
                                       prismatic_xyz_filename,
                                       delimiter='   |    |  ',
                                       header=None,
                                       skiprows=[0, 1],
                                       engine='python',
                                       occupancy=1.0,
                                       rms_thermal_vib=0.05,
                                       header_comment="Let's make a file!",
                                       save=True):
    '''
    Convert from Vesta outputted xyz file format to the prismatic-style xyz
    format. 
    Lose some information from the .cif or .vesta file but okay for now.
    Develop your own converter if you need rms and occupancy! Lots to do.

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
    >>> prismatic_xyz = convert_vesta_xyz_to_prismatic_xyz(
                vesta_xyz_filename='MoS2_hex_vesta_xyz.xyz',
                prismatic_xyz_filename='MoS2_hex_prismatic.xyz',
                delimiter='   |    |  ',
                header=None,
                skiprows=[0, 1],
                engine='python',
                occupancy=1.0,
                rms_thermal_vib=0.05,
                header_comment="Let's make a file!",
                save=True)

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
                print('ERROR: nans present, file not read correctly. Try changes the delimiters! See: https://stackoverflow.com/questions/51195299/python-reading-a-data-text-file-with-different-delimiters')

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
            x for x in file.loc[0:file.shape[0], name].values if not isinstance(x, str)]
        min_axis = min(axis_values_list)
        max_axis = max(axis_values_list)
        unit_cell_dimen_axis = max_axis-min_axis
        unit_cell_dimen_axis = format(unit_cell_dimen_axis, '.6f')
        unit_cell_dimen.append(unit_cell_dimen_axis)
    # should match the vesta values (or be slightly larger)
    print(unit_cell_dimen)

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

    if save == True:

        if '.xyz' not in prismatic_xyz_filename:
            file.to_csv(prismatic_xyz_filename + '.xyz',
                        sep=' ', header=False, index=False)
        else:
            file.to_csv(prismatic_xyz_filename,
                        sep=' ', header=False, index=False)

    return file
