
from temul.intensity_tools import get_sublattice_intensity
from temul.element_tools import split_and_sort_element

import atomap.api as am
from atomap.atom_finding_refining import get_atom_positions_in_difference_image

import matplotlib.pyplot as plt
import scipy
import numpy as np
from numpy import log
import hyperspy.api as hs
import pandas as pd
from collections import Counter


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
    # >>> from temul.model_creation import count_element_in_pandas_df
    # >>> Mo_count = count_element_in_pandas_df(element='Mo', dataframe=df)

    '''
    count_of_element = Counter()

    for element_config in dataframe.columns:
        #        print(element_config)
        if element in element_config:
            split_element = split_and_sort_element(element_config)

            for split in split_element:
                # split=split_element[1]
                if element in split[1]:
                    # print(element + ":" + str(split[2]*dataframe.loc[
                    # :, element_config]))
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
    # >>> from temul.model_creation import count_all_individual_elements
    # >>> individual_element_list = ['Mo', 'S', 'Se']
    # >>> element_count = count_all_individual_elements(
    # ...     individual_element_list, dataframe=df)
    # >>> element_count
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

    >>> from temul.model_creation import count_atoms_in_sublattice_list
    >>> import atomap.api as am
    >>> atom_lattice = am.dummy_data.get_simple_atom_lattice_two_sublattices()
    >>> sub1 = atom_lattice.sublattice_list[0]
    >>> sub2 = atom_lattice.sublattice_list[1]

    >>> for i in range(0, len(sub1.atom_list)):
    ...     sub1.atom_list[i].elements = 'Ti_2'
    >>> for i in range(0, len(sub2.atom_list)):
    ...     sub2.atom_list[i].elements = 'Cl_1'

    >>> added_atoms = count_atoms_in_sublattice_list(
    ...     sublattice_list=[sub1, sub2])

    Compare before and after 
    >>> at_lat_before = am.dummy_data.get_simple_atom_lattice_two_sublattices()
    >>> no_added_atoms = count_atoms_in_sublattice_list(
    ...    sublattice_list=at_lat_before.sublattice_list)

    '''
    count_of_sublattice = Counter()
    for sublattice in sublattice_list:

        sublattice_info = print_sublattice_elements(sublattice)
        elements_in_sublattice = [atoms[0:1]
                                  for atoms in sublattice_info]
        # get just chemical info
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

    >>> from temul.model_creation import (
    ...     count_atoms_in_sublattice_list,
    ...     compare_count_atoms_in_sublattice_list)
    >>> import atomap.api as am
    >>> atom_lattice = am.dummy_data.get_simple_atom_lattice_two_sublattices()
    >>> sub1 = atom_lattice.sublattice_list[0]
    >>> sub2 = atom_lattice.sublattice_list[1]

    >>> for i in range(0, len(sub1.atom_list)):
    ...     sub1.atom_list[i].elements = 'Ti_2'
    >>> for i in range(0, len(sub2.atom_list)):
    ...     sub2.atom_list[i].elements = 'Cl_1'

    >>> added_atoms = count_atoms_in_sublattice_list(
    ...     sublattice_list=[sub1, sub2],
    ...     filename=atom_lattice.name)

    >>> at_lat_before = am.dummy_data.get_simple_atom_lattice_two_sublattices()
    >>> no_added_atoms = count_atoms_in_sublattice_list(
    ...     sublattice_list=at_lat_before.sublattice_list,
    ...     filename=at_lat_before.name)

    >>> compare_count_atoms_in_sublattice_list([added_atoms, no_added_atoms])
    False

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


'''
This is the first loop to refine the simulation:
refining the atom_position's assigned atoms without
changing the atom_position location. Using intensity of current positions only
'''

# aim hereis to change the elements in the sublattice to something that will
#   make the simulation agree more with the experiment


def change_sublattice_atoms_via_intensity(
        sublattice,
        image_diff_array,
        darker_or_brighter,
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
    >>> from temul.model_creation import change_sublattice_atoms_via_intensity
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> for i in range(0, len(sublattice.atom_list)):
    ...     sublattice.atom_list[i].elements = 'Mo_1'
    ...     sublattice.atom_list[i].z_height = [0.5]
    >>> element_list = ['H_0', 'Mo_1', 'Mo_2']
    >>> image_diff_array = np.array([[5, 2, 2, 20],[1, 2, 4, 7]])
    >>> # This will change the 5th atom in the sublattice to a lower atomic Z
    >>> # number, i.e., 'H_0' in the given element_list
    >>> change_sublattice_atoms_via_intensity(sublattice=sublattice,
    ...                               image_diff_array=image_diff_array,
    ...                               darker_or_brighter=0,
    ...                               element_list=element_list)
    Changing some atoms


    '''
    if image_diff_array.size == 0:
        pass
    else:
        print('Changing some atoms')
        for p in image_diff_array[:, 0]:
            # could be a better way to do this within
            # image_difference_intensity()
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
                raise ValueError("No element assigned for atom %s. Note that "
                                 "this error only picks up first instance of "
                                 "fail" % p)
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
                               sim_image,
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
    sim_image : HyperSpy 2D signal
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
    >>> sim_image = am.dummy_data.get_simple_cubic_with_vacancies_signal()
    >>> for i in range(0, len(sublattice.atom_list)):
    ...     sublattice.atom_list[i].elements = 'Mo_1'
    ...     sublattice.atom_list[i].z_height = [0.5]
    >>> element_list = ['H_0', 'Mo_1', 'Mo_2']
    >>> image_difference_intensity(sublattice=sublattice,
    ...                            sim_image=sim_image,
    ...                            element_list=element_list)

    with some image noise and plotting the images
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice(
    ...     image_noise=True)
    >>> sim_image = am.dummy_data.get_simple_cubic_with_vacancies_signal()
    >>> for i in range(0, len(sublattice.atom_list)):
    ...     sublattice.atom_list[i].elements = 'Mo_1'
    ...     sublattice.atom_list[i].z_height = [0.5]
    >>> element_list = ['H_0', 'Mo_1', 'Mo_2']
    >>> image_difference_intensity(sublattice=sublattice,
    ...                            sim_image=sim_image,
    ...                            element_list=element_list)

    '''

    # np.array().T needs to be taken away for newer atomap versions
    sublattice_atom_positions = np.array(sublattice.atom_positions).T

    diff_image = hs.signals.Signal2D(sublattice.image - sim_image.data)

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
    # diff_mean_ints = np.array(diff_mean_ints, ndmin=2).T

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

    if change_sublattice is True:
        # Now make the changes to the sublattice for both bright
        # and dark arrays
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
        #        sim_image.plot()
        #        diff_image.plot()
        diff_sub.plot()
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.title("Image Differences with " +
                  sublattice.name + " Overlay", fontsize=16)
        plt.savefig(
            fname="Image Differences with " + sublattice.name + "Overlay.png",
            transparent=True, frameon=False, bbox_inches='tight',
            pad_inches=None, dpi=300, labels=False)

        plt.figure()
        plt.hist(diff_mean_ints, bins=50, color='b', zorder=-1)
        plt.scatter(mean_ints, len(diff_mean_ints)/50, c='red', zorder=1)
        plt.scatter(
            std_from_mean_array[:, 0], std_from_mean_array[:, 1], c='green',
            zorder=1)
        plt.title("Histogram of " + sublattice.name +
                  " Intensities", fontsize=16)
        plt.xlabel("Intensity (a.u.)", fontsize=16)
        plt.ylabel("Count", fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.savefig(
            fname="Histogram of " + sublattice.name + " Intensities.png",
            transparent=True, frameon=False, bbox_inches='tight',
            pad_inches=None, dpi=300, labels=False)

    else:
        pass


def image_difference_position(sublattice_list,
                              sim_image,
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
    sim_image : simulated image used for comparison with sublattice image
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

    >>> from temul.model_creation import image_difference_position
    >>> import atomap.api as am
    >>> sublattice = am.dummy_data.get_simple_cubic_with_vacancies_sublattice(
    ...                                             image_noise=True)
    >>> sim_image = am.dummy_data.get_simple_cubic_signal()
    >>> for i in range(0, len(sublattice.atom_list)):
    ...         sublattice.atom_list[i].elements = 'Mo_1'
    ...         sublattice.atom_list[i].z_height = '0.5'
    >>> # Check without adding a new sublattice
    >>> image_difference_position(sublattice_list=[sublattice],
    ...                           sim_image=sim_image,
    ...                           pixel_threshold=1,
    ...                           percent_to_nn=None,
    ...                           mask_radius=5,
    ...                           num_peaks=5,
    ...                           add_sublattice=False)
    >>> # Add a new sublattice
    >>> # if you have problems with mask_radius, increase it!
    >>> # Just a gaussian fitting issue, could turn it off!
    >>> sub_new = image_difference_position(sublattice_list=[sublattice],
    ...                                   sim_image=sim_image,
    ...                                   pixel_threshold=10,
    ...                                   num_peaks=5,
    ...                                   add_sublattice=True)
    New Atoms Found! Adding to a new sublattice
    '''

    image_for_sublattice = sublattice_list[0]
    diff_image = hs.signals.Signal2D(
        image_for_sublattice.image - sim_image.data)
    diff_image_inverse = hs.signals.Signal2D(
        sim_image.data - image_for_sublattice.image)

    # below function edit of get_atom_positions. Just allows num_peaks from
    # sklearn>find_local_maximum
    atom_positions_diff_image = get_atom_positions_in_difference_image(
        diff_image, num_peaks=num_peaks)
    atom_positions_diff_image_inverse = get_atom_positions_in_difference_image(
        diff_image_inverse, num_peaks=num_peaks)

    diff_image_sub = am.Sublattice(atom_positions_diff_image, diff_image)
    # diff_image_sub.find_nearest_neighbors()
    # diff_image_sub.refine_atom_positions_using_center_of_mass(
    #     percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    # diff_image_sub.refine_atom_positions_using_2d_gaussian(
    #     percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    atom_positions_sub_diff = np.array(diff_image_sub.atom_positions).T

    # sublattice.plot()

    diff_image_sub_inverse = am.Sublattice(atom_positions_diff_image_inverse,
                                           diff_image_inverse)
    # diff_image_sub_inverse.find_nearest_neighbors()
    # diff_image_sub_inverse.refine_atom_positions_using_center_of_mass(
    #     percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    # diff_image_sub_inverse.refine_atom_positions_using_2d_gaussian(
    #     percent_to_nn=percent_to_nn, mask_radius=mask_radius)
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
    elif len(atom_positions_sub_new) != 0 and add_sublattice is True:
        print("New Atoms Found! Adding to a new sublattice")

        sub_new = am.Sublattice(
            atom_positions_sub_new, sublattice_list[0].image,
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

        if sub_new_exists is True:
            sub_new.plot()
            plt.title(sub_new.name + filename, fontsize=20)
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.tight_layout()
            plt.savefig(fname='pos_diff_' + sub_new.name + filename + '.png',
                        transparent=True, frameon=False, bbox_inches='tight',
                        pad_inches=None, dpi=300, labels=False)

    return sub_new if sub_new_exists is True else None


# scaling method
# Limited to single elements at the moment. Need to figure out maths to expand
# it to more.
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

    # use the get_sublattice_intensity() function to get the mean/mode
    # intensities of
    #   each sublattice
    if type(mask_radius) is list:
        sublattice0_intensity = get_sublattice_intensity(
            sublattice0, intensity_type, remove_background_method,
            background_sublattice,
            num_points, percent_to_nn=percent_to_nn,
            mask_radius=mask_radius[0])

        sublattice1_intensity = get_sublattice_intensity(
            sublattice1, intensity_type, remove_background_method,
            background_sublattice,
            num_points, percent_to_nn=percent_to_nn,
            mask_radius=mask_radius[1])
    else:
        sublattice0_intensity = get_sublattice_intensity(
            sublattice0, intensity_type, remove_background_method,
            background_sublattice,
            num_points, percent_to_nn=percent_to_nn,
            mask_radius=mask_radius)

        sublattice1_intensity = get_sublattice_intensity(
            sublattice1, intensity_type, remove_background_method,
            background_sublattice,
            num_points, percent_to_nn=percent_to_nn,
            mask_radius=mask_radius)

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
        scaling_exponent = log(
            denominator_element_split[0][2]*scaling_ratio) / (
            log(numerator_element_split[0][3]) -
            log(denominator_element_split[0][3]))
    else:
        pass
    # need to include more complicated equation to deal with
    # multiple elements as the e.g., numerator

    return(scaling_ratio, scaling_exponent, sublattice0_intensity_method,
           sublattice1_intensity_method)


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
    elif material_type == 'single_element_column':
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

# Calculate the middle point and limits of the distribution for a
# given element_list.
# Need to add Mike's histogram display
'''


def find_middle_and_edge_intensities(sublattice,
                                     element_list,
                                     standard_element,
                                     scaling_exponent,
                                     largest_element_intensity=None,
                                     split_symbol=['_', '.']):
    """
    Create a list which represents the peak points of the
    intensity distribution for each atom.

    works for nanoparticles as well, doesn't matter what
    scaling_exponent you use for nanoparticle. Figure this out!

    If the max_element_intensity is set, then the program assumes
    that the standard element is the largest available element
    combination, and scales the middle and limit intensity lists
    so that the middle_intensity_list[-1] == max_element_intensity

    """

    middle_intensity_list = []
    limit_intensity_list = [0.0]

    if isinstance(standard_element, str) is True:
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

    if largest_element_intensity is not None:
        ratio = sublattice.image.max() / largest_element_intensity
        middle_intensity_list = [
            middle/ratio for middle in middle_intensity_list]
        limit_intensity_list = [limit/ratio for limit in limit_intensity_list]

    return middle_intensity_list, limit_intensity_list


# choosing the percent_to_nn for this seems dodgy atm...
def find_middle_and_edge_intensities_for_background(
        elements_from_sub1,
        elements_from_sub2,
        sub1_mode,
        sub2_mode,
        element_list_sub1,
        element_list_sub2,
        middle_intensity_list_sub1,
        middle_intensity_list_sub2):

    middle_intensity_list_background = [0.0]

    # it is neccessary to scale the background_sublattice intensities here
    # already because otherwise
    #   the background_sublattice has no reference atom to base its mode
    # intensity on. eg. in MoS2, first sub has Mo
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
            middle_intensity_list_background[i] +
            middle_intensity_list_background[i+1])/2
        limit_intensity_list_background.append(limit)

    if len(limit_intensity_list_background) <= len(
            middle_intensity_list_background):
        max_limit = middle_intensity_list_background[-1] + (
            middle_intensity_list_background[-1] -
            limit_intensity_list_background[-1])
        limit_intensity_list_background.append(max_limit)
    else:
        pass

    return middle_intensity_list_background, limit_intensity_list_background


#
#
# sub2_ints = get_sublattice_intensity(sub2, intensity_type='max',
# remove_background_method=None)
#
# min(sub2_ints)
# sub2_ints.sort()
#
# sub2_mode = scipy.stats.mode(np.round(sub2_ints, decimals=2))[0][0]
#
# limit_numbers = []
# for i in limit_intensity_list_sub2:
#    limit_numbers.append(i*sub2_mode)
#
#
# elements_of_sub2 = sort_sublattice_intensities(
#   sub2, 'max', middle_intensity_list_sub2,
#   limit_intensity_list_sub2, element_list_sub2,
#   method='mode', remove_background_method=None,
#   percent_to_nn=0.2)
#
# sublattice = sub2
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
                                intensity_type='max',
                                element_list=[],
                                scalar_method='mode',
                                middle_intensity_list=None,
                                limit_intensity_list=None,
                                remove_background_method=None,
                                background_sublattice=None,
                                num_points=3,
                                intensity_list_real=False,
                                percent_to_nn=0.40, mask_radius=None):

    # intensity_list_real is asking whether the intensity values in your
    # intensity_list for the current sublattice
    #   are scaled. Scaled meaning already multiplied by the mean or mode
    # of said sublattice.
    #   Set to Tru for background sublattices. For more details see
    # "find_middle_and_edge_intensities_for_background()"
    #   You can see that the outputted lists are scaled by the mean or mode,
    # whereas in
    #   "find_middle_and_edge_intensities()", they are not.

    # For testing and quickly assigning a sublattice some elements.
    if middle_intensity_list is None:
        elements_of_sublattice = []
        for i in range(0, len(sublattice.atom_list)):
            sublattice.atom_list[i].elements = element_list[0]
            elements_of_sublattice.append(sublattice.atom_list[i].elements)

    else:
        sublattice_intensity = get_sublattice_intensity(
            sublattice, intensity_type, remove_background_method,
            background_sublattice,
            num_points, percent_to_nn=percent_to_nn,
            mask_radius=mask_radius)

        for i in sublattice_intensity:
            if i < 0:
                i = 0.0000000001
                # raise ValueError("You have negative intensity. Bad Vibes")

        if intensity_list_real is False:

            if scalar_method == 'mean':
                scalar = np.mean(sublattice_intensity)
            elif scalar_method == 'mode':
                scalar = scipy.stats.mode(
                    np.round(sublattice_intensity, decimals=2))[0][0]
            elif isinstance(scalar_method, (int, float)):
                scalar = scalar_method

            if len(element_list) != len(middle_intensity_list):
                raise ValueError(
                    "element_list length does not equal "
                    "middle_intensity_list length")
            else:
                pass

            elements_of_sublattice = []
            for p in range(0, (len(limit_intensity_list)-1)):
                for i in range(0, len(sublattice.atom_list)):
                    if limit_intensity_list[p]*scalar < sublattice_intensity[i] < limit_intensity_list[p+1]*scalar:
                        sublattice.atom_list[i].elements = element_list[p]
                        elements_of_sublattice.append(
                            sublattice.atom_list[i].elements)

        elif intensity_list_real is True:
            if len(element_list) != len(middle_intensity_list):
                raise ValueError(
                    "element_list length does not equal middle_intensity_list "
                    "length")
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
# sublattice.atom_list[36].elements = 'S_1'
# i=36
#

# whatareyou = split_and_sort_element(
#   element=sublattice.atom_list[i].elements)[0][2]
#
# if whatareyou == 0:
#    print('arg')
# else:
#    print('nope')


# if chalcogen = True, give positions as...
# currently "chalcogen" is relevant to our TMDC work

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
                    # raise ValueError("z_height is limited to only a handful of positions")
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
                    # raise ValueError("z_height is limited to only a handful of positions")
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
                # raise ValueError("z_height is limited to only a handful of positions")
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
                # raise ValueError("z_height is limited to only a handful of positions")
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
                    # raise ValueError("z_height is limited to only a handful of positions")
            elif lattice_type == 'TM_bot':
                if len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 3:
                    sublattice.atom_list[i].z_height = '0.25, 0.6275, 0.8725'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2:
                    sublattice.atom_list[i].z_height = '0.25, 0.6275, 0.8725'
                else:
                    sublattice.atom_list[i].z_height = '0.95'
                # raise ValueError("z_height is limited to only a handful of positions")
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


def print_sublattice_elements(sublattice, number_of_lines='all'):

    if number_of_lines == 'all':
        number_of_lines_end = len(sublattice.atom_list)
    else:
        number_of_lines_end = number_of_lines

    elements_of_sublattice = []
    for i in range(0, number_of_lines_end):
        sublattice.atom_list[i].elements
        sublattice.atom_list[i].z_height  # etc.
        elements_of_sublattice.append([
            sublattice.atom_list[i].elements,
            sublattice.atom_list[i].z_height,
            sublattice.atom_amplitude_max_intensity[i],
            sublattice.atom_amplitude_mean_intensity[i],
            sublattice.atom_amplitude_min_intensity[i],
            sublattice.atom_amplitude_total_intensity[i]]
        )

    return elements_of_sublattice


def return_z_coordinates(z_thickness,
                         z_bond_length,
                         number_atoms_z=None,
                         max_number_atoms_z=None,
                         fractional_coordinates=True,
                         atom_layout='bot'):
    '''
    Produce fractional z-dimension coordinates for a given thickness and bond
    length.

    Parameters
    ----------
    z_thickness : number
        Size (Angstrom) of the z-dimension.
    z_bond_length : number
        Size (Angstrom) of the bond length between adjacent atoms in the
        z-dimension.
    number_atoms_z : integer, default None
        number of atoms in the z-dimension. If this is set to an interger
        value, it will override the use of z_thickness.
    centered_atoms : Bool, default True
        If set to True, the z_coordinates will be centered about 0.5.
        If set to False, the z_coordinate will start at 0.0.

    Returns
    -------
    1D numpy array

    Examples
    --------
    # >>> from temul.model_creation import return_z_coordinates
    # >>> Au_NP_z_coord = return_z_coordinates(z_thickness=20, z_bond_length=1.5)

    '''

    if max_number_atoms_z is not None:
        # print("number_atoms_z has been specified, using
        # number_atoms_z instead\
        #     of z_thickness")
        z_thickness = max_number_atoms_z * z_bond_length

    z_coords_all = np.arange(start=0,
                             stop=z_thickness,
                             step=z_bond_length)

    if number_atoms_z > max_number_atoms_z:
        raise ValueError("number_atoms_z is greater than max_number_atoms_z."
                         "Not allowed.")
    elif number_atoms_z == max_number_atoms_z:
        z_coords = z_coords_all
    elif number_atoms_z < max_number_atoms_z:
        if atom_layout in ('bot', 'center'):
            z_coords = z_coords_all[:number_atoms_z]
        elif atom_layout == 'top':
            z_coords = z_coords_all[-1*number_atoms_z:]
        else:
            raise ValueError("Only 'bot', 'center' and 'top' are allowed.")

    if fractional_coordinates:
        z_coords = z_coords/z_thickness

    # for centered particles (atoms centered around 0.5 in z)
    if atom_layout == 'center':
        # adds half the distance from the top atom to the top of the unit cell
        # to each atom coordinate.
        z_coords = z_coords + (1-z_coords.max())/2

    return(z_coords)


'''
add intensity used for getting number, and index for later reference with
sublattice
'''


def return_xyz_coordinates(x, y,
                           z_thickness, z_bond_length,
                           number_atoms_z=None,
                           fractional_coordinates=True,
                           atom_layout='bot'):
    '''
    Produce xyz coordinates for an xy coordinate given the z-dimension
    information.

    Parameters
    ----------

    x, y : float
        atom position coordinates.

    for other parameters see return_z_coordinates()

    Returns
    -------
    2D numpy array with columns x, y, z

    Examples
    --------
    # >>> from temul.model_creation import return_xyz_coordinates
    # >>> x, y = 2, 3
    # >>> atom_coords = return_xyz_coordinates(x, y,
    # ...                         z_thickness=10,
    # ...                         z_bond_length=1.5,
    # ...                         number_atoms_z=5)

    '''

    z_coords = return_z_coordinates(
        number_atoms_z=number_atoms_z,
        z_thickness=z_thickness,
        z_bond_length=z_bond_length,
        fractional_coordinates=fractional_coordinates,
        atom_layout=atom_layout)

    # One z for each atom in number_atoms for each xy pair
    atom_coords = []
    for z in z_coords:
        atom_coords.append([x, y, z])

    return(np.array(atom_coords))


def convert_numpy_z_coords_to_z_height_string(z_coords):
    """
    Convert from the output of return_z_coordinates(), which is a 1D numpy
    array, to a long string, with which I have set up
    sublattice.atom_list[i].z_height.

    Examples
    --------
    # >>> from temul.model_creation import (
    # ...     return_z_coordinates,
    # ...     convert_numpy_z_coords_to_z_height_string)
    # >>> Au_NP_z_coord = return_z_coordinates(z_thickness=20, z_bond_length=1.5)
    # >>> Au_NP_z_height_string = convert_numpy_z_coords_to_z_height_string(
    # ...     Au_NP_z_coord)

    """
    z_string = ""
    for coord in z_coords:

        if z_string == "":
            z_string = z_string + "{0:.{1}f}".format(coord, 6)
        else:
            z_string = z_string + "," + "{0:.{1}f}".format(coord, 6)
        # str(coord).format('.6f')

    return(z_string)


def get_max_number_atoms_z(sublattice):

    max_number_atoms_z_list = []

    for i in range(0, len(sublattice.atom_list)):

        max_number_atoms_z_list.append(split_and_sort_element(
            element=sublattice.atom_list[i].elements)[0][2])

    max_number_atoms_z = max(max_number_atoms_z_list)

    return(max_number_atoms_z)


def assign_z_height_to_sublattice(sublattice,
                                  z_bond_length,
                                  material=None,
                                  fractional_coordinates=True,
                                  atom_layout='bot'):
    '''
    Set the z_heights for each atom position in a sublattice.

    Parameters
    ----------
    sublattice : Atomap Sublattice object

    Examples
    --------
    See example_scripts : Model Creation Example

    '''

    # if material == 'Au_NP_'
    # z_bond_length = function_to_get_material_info

    z_thickness = 1

    max_number_atoms_z = get_max_number_atoms_z(
        sublattice=sublattice)

    for i in range(0, len(sublattice.atom_list)):
        # if i < 10:
        #     print(str(i) + ' ' + sublattice.atom_list + ' ' + str(i))

        number_atoms_z = split_and_sort_element(
            element=sublattice.atom_list[i].elements)[0][2]

        z_coords = return_z_coordinates(
            number_atoms_z=number_atoms_z,
            max_number_atoms_z=max_number_atoms_z,
            z_thickness=z_thickness,
            z_bond_length=z_bond_length,
            fractional_coordinates=fractional_coordinates,
            atom_layout=atom_layout)

        z_height = convert_numpy_z_coords_to_z_height_string(z_coords)
        sublattice.atom_list[i].z_height = z_height


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

    # value += split_and_sort_element(sublattice.atom_list[i].elements)[k][2]
    # need an option to save to the cuurent directory should be easy
    #        dfObj.to_pickle('atom_lattice_atom_position_table.pkl')
    #        dfObj.to_csv('atom_lattice_atom_position_table.csv', sep=',', index=False)
    return dfObj

    # element_list = ['S_0', 'S_1', 'S_2', 'S_2.C_1', 'S_2.C_2', 'Mo_1', 'Mo_0']
    # example_df = create_dataframe_for_cif(atom_lattice, element_list)

    # '_atom_site_fract_z' : format( (sublattice.atom_list[i].z_height)[p+(k*k)], '.6f'), #great touch
