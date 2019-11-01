
import numpy as np
from atomap.atom_finding_refining import _make_circular_mask


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


# need to add "radius" for where to get intensity from. Do we though?
#   until we find a way of defining it in the image, radius should be left alone. Radius can be accessed in the
#   periodictable package anyway.
# need to add remove backgroun locally or with a third sublattice
#        sublattice.find_nearest_neighbors()


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