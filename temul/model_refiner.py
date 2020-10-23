
from temul.model_creation import (
    count_atoms_in_sublattice_list,
    compare_count_atoms_in_sublattice_list,
    image_difference_intensity,
    image_difference_position,
    count_all_individual_elements,
    get_positions_from_sublattices,
    get_most_common_sublattice_element,
    auto_generate_sublattice_element_list
)
from temul.element_tools import (
    get_individual_elements_from_element_list,
    combine_element_lists,
    split_and_sort_element,
    atomic_radii_in_pixels
)
from temul.io import (create_dataframe_for_xyz,
                      load_prismatic_mrc_with_hyperspy)

from temul.signal_processing import (
    compare_two_image_and_create_filtered_image,
    calibrate_intensity_distance_with_sublattice_roi,
    measure_image_errors
)
from temul.simulations import (
    simulate_with_prismatic
)
from temul.dummy_data import (
    get_simple_cubic_sublattice,
    get_simple_cubic_signal
)
import numpy as np
import pandas as pd
import hyperspy
import temul.external.atomap_devel_012.api as am_dev
from temul.external.atomap_devel_012.initial_position_finding import (
    add_atoms_with_gui as choose_points_on_image)
import matplotlib.pyplot as plt
import copy
from scipy.ndimage import gaussian_filter


class Model_Refiner():
    def __init__(self, sublattice_and_elements_dict,
                 comparison_image=None, sampling=None,
                 thickness=10, name=''):
        '''
        Object which is used to refine the elements in a list of
        Atomap Sublattice objects. There are currently two refinement methods:
        1. Refine with position `image_difference_position_model_refiner`
        2. Refine with intensity `image_difference_intensity_model_refiner`.

        Parameters
        ----------
        sublattice_and_elements_dict : dictionary
            A dictionary of the structure `{sublattice: element_list,}` where
            each sublattice is an Atomap sublattice object and each
            element_list is a list of elements of the form given in the
            examples of `get_individual_elements_from_element_list` in the
            element_tools.py module.
        comparison_image : Hyperspy Signal2D, default None
            This is the image with which the first `sublattice` image will be
            compared to refine the list of sublattices' elements.
            If None is given when the Model Refiner is created, or if the
            `comparison_image` is not the same shape as the first sublattice
            image, a warning will be returned.
            To set a `comparison_image`, you can either use
            `model_refiner.create_simulation` or by setting the
            `model_refiner.comparison_image` to an image.
        sampling : float, default None
            The real space sampling of the `sublattice` images in Angstrom.
            The sampling is defined as: sampling = angstrom/pixel. This
            sampling will be identical for the `comparison_image`. If it is set
            to None, the sampling will automatically be set by the
            `sublattice.signal` object.
        thickness : float, default 10
            Physical thickness of the sample in Angstrom. This will be used for
            the simulation.
        name : string, default ''
            Name of the Model Refiner.


        Attributes
        ----------
        sublattice_and_elements_dict: dictionary
        sublattice_list: list
        name: string
        element_list: list of strings, or lists of lists of strings
        flattened_element_list: list of strings
        _element_count: Counter
        element_count_history_list: list of Counters
        refinement_history: list of strings
        reference_image: Hyperspy Signal2D
        calibration_area: 2D array-like; list of 2 lists
        calibration_separation: int
        sampling: float
        thickness: float
        image_xyz_sizes: list
        previous_refiner_instance: Model Refiner object
        error_between_comparison_and_reference_image: list of floats
        error_between_images_history: list of lists

        Examples
        --------
        For now, see `image_difference_intensity_model_refiner` and
        `image_difference_position_model_refiner` below.
        '''

        self.sublattice_and_elements_dict = sublattice_and_elements_dict
        self.sublattice_list = list(sublattice_and_elements_dict.keys())
        self.element_list = list(sublattice_and_elements_dict.values())
        self.flattened_element_list = combine_element_lists(
            self.element_list)

        self.reference_image = self.sublattice_list[0].signal
        self._comparison_image_init(comparison_image)
        self.name = name
        self._element_count = count_atoms_in_sublattice_list(
            self.sublattice_list, filename=None)
        self.element_count_history_list = []
        if len(self.element_count_history_list) == 0:
            self.element_count_history_list.append(self._element_count)

        self.refinement_history = []
        if len(self.refinement_history) == 0:
            self.refinement_history.append("Initial State")

        self.calibration_area = [[1, 1],
                                 [self.reference_image.data.shape[-1] - 1,
                                  self.reference_image.data.shape[-2] - 1]]
        self.calibration_separation = 12
        # maybe have a _sampling_init function
        if sampling is None:
            if sampling == 1:
                print("Sampling (axes_manager.scale) is 1, you may need to "
                      "set this.")
            else:
                self.sampling = self.reference_image.axes_manager[-1].scale
        else:
            self.sampling = sampling
        if self.reference_image.axes_manager[-1].units == 'nm':
            self.sampling = self.sampling * 10
        self._reference_image_init()

        self._auto_mask_radii_init()

        self.thickness = thickness
        self.image_xyz_sizes = [
            self.sampling * self.reference_image.data.shape[-1],
            self.sampling * self.reference_image.data.shape[-2],
            self.thickness]

        self.previous_refiner_instance = None
        self.update_sublattices_positions(self.sublattice_list)

        self.error_between_comparison_and_reference_image = []
        self.error_between_images_history = []
        if len(self.error_between_comparison_and_reference_image) != 0:
            self.error_between_images_history.append(
                self.error_between_comparison_and_reference_image)

    def _reference_image_init(self):
        axes = self.reference_image.axes_manager
        axes[-1].scale = axes[-2].scale = self.sampling
        axes[-1].units = axes[-2].units = 'A'

    def _comparison_image_init(self, comparison_image):

        if comparison_image is None:
            print("Warning: "
                  "comparison_image is set to None. You will not be able to "
                  "refine the model until a comparison_image is set. You can "
                  "do this via Model_refiner.create_simulation() or by "
                  "setting the Model_Refiner.comparison_image to an image.")
        else:
            if not isinstance(comparison_image,
                              hyperspy._signals.signal2d.Signal2D):
                raise ValueError(
                    "comparison_image must be a 2D Hyperspy signal of type "
                    "hyperspy._signals.signal2d.Signal2D. The current "
                    "incorrect type is {}".format(str(type(comparison_image))))

            for sublattice in self.sublattice_list:
                if not comparison_image.data.shape == sublattice.image.shape:
                    print("Warning: "
                          "comparison_image must have the same shape as each "
                          "sublattice image. comparison_image shape is {}, "
                          "while sublattice '{}' is {}. This will stop you "
                          "from refining your model.".format(
                              comparison_image.data.shape,
                              sublattice.name,
                              sublattice.image.data))

            comparison_image.axes_manager = self.reference_image.axes_manager

        self.comparison_image = comparison_image

    def _comparison_image_warning(self, error_message=['None', 'wrong_size']):
        if 'None' in error_message:
            if self.comparison_image is None:
                raise ValueError(
                    "The comparison_image attribute has not been "
                    "set. You will not be able to "
                    "refine the model until a comparison_image is set. You "
                    "can do this via Model_refiner.create_simulation() or by "
                    "setting the Model_Refiner.comparison_image to an image.")

        if 'wrong_size' in error_message:
            for sublattice in self.sublattice_list:
                if not self.comparison_image.data.shape == \
                        sublattice.image.shape:
                    raise ValueError(
                        "comparison_image must have the same shape as each "
                        "sublattice image. comparison_image shape is {}, "
                        "while sublattice '{}' is {}".format(
                            self.comparison_image.data.shape,
                            sublattice.name,
                            sublattice.image.data))

    def __repr__(self):
        return '<%s, %s (sublattices:%s,element_list:%s)>' % (
            self.__class__.__name__,
            self.name,
            len(self.sublattice_list),
            len(self.element_list),
        )

    def _auto_mask_radii_init(self):
        '''
        Automatically set the mask_radius for each sublattice.
        Use mask_radius='auto' when calling a Model Refiner method.
        Ideally you could have the mask_radius/percent_to_nn change depending
        on the atom, but for now depending on the sublattice is okay.
        '''

        mask_radii = []
        for sublattice in self.sublattice_list:
            element_config = get_most_common_sublattice_element(
                sublattice, info='element')
            chemical = split_and_sort_element(element_config)[0][1]
            radius = atomic_radii_in_pixels(self.sampling, chemical)
            mask_radii.append(radius)

        self.auto_mask_radius = mask_radii

    @property
    def element_count(self):
        self._element_count = count_atoms_in_sublattice_list(
            self.sublattice_list, filename=None)
        return self._element_count

    def update_element_count_history(self):
        self.element_count
        self.element_count_history_list.append(self._element_count)

    def update_refinement_history(self, refinement_method):
        self.refinement_history.append(refinement_method)

    def update_element_count_and_refinement_history(self, refinement_method):
        self.update_element_count_history()
        self.update_refinement_history(refinement_method)

    def compare_latest_element_counts(self):

        if len(self.element_count_history_list) < 2:
            return False
            # raise ValueError("element_count_history must have at least two "
            #                  "element_counts for comparison")
        else:
            return(compare_count_atoms_in_sublattice_list(
                self.element_count_history_list[-2:]))

    def get_element_count_as_dataframe(self):
        '''
        Allows you to view the element count as a Pandas Dataframe.
        See `image_difference_intensity_model_refiner` for an example.

        Returns
        -------
        Pandas Dataframe
        '''

        elements_ = [i for sublist in self.element_list for i in sublist]
        elements_ = list(set(elements_))
        elements_.sort()

        df = pd.DataFrame(columns=elements_)
        for element_in_history in self.element_count_history_list:
            df = df.append(element_in_history, ignore_index=True).fillna(0)
        for i, refinement_name in enumerate(self.refinement_history):
            df.rename(index={i: str(i) + " " + refinement_name}, inplace=True)

        return df

    def get_individual_elements_as_dataframe(self, split_symbol=['_', '.']):

        df_all = self.get_element_count_as_dataframe()
        indiv_element_list = get_individual_elements_from_element_list(
            self.element_list, split_symbol=split_symbol)
        indiv_element_counts = count_all_individual_elements(
            indiv_element_list, df_all)
        df = pd.DataFrame.from_dict(indiv_element_counts)
        return df

    def combine_individual_and_element_counts_as_dataframe(
            self, split_symbol=['_', '.']):

        df_configs = self.get_element_count_as_dataframe()
        df_indiv = self.get_individual_elements_as_dataframe(
            split_symbol=['_', '.'])

        df_combined = pd.concat([df_configs, df_indiv], axis=1)
        return df_combined

    def plot_element_count_as_bar_chart(self, element_configs=0,
                                        flip_colrows=True,
                                        title="Refinement of Elements",
                                        fontsize=16, split_symbol=['_', '.']):
        '''
        Allows you to plot the element count as a bar chart.
        See `image_difference_intensity_model_refiner` for an example.

        Parameters
        ----------
        element_configs : int, default 0
            Change the plotted data.
            `element_configs=0` will plot the element configuration counts,
            `element_configs=1` will plot the individual element counts, and
            `element_configs=2` will plot the both 0 and 1.
        flip_colrows : Bool, default True
            Change how the data is plotted.
            `flip_colrows=True` will take the transpose of the dataframe.
        title : string, default "Refinement of Elements"
            Title of the bar chart.
        fontsize : int, default 16
        split_symbol : list, default ['_', '.']
            See temul.element_tools.split_and_sort_element for details.
        '''

        if element_configs == 0:  # only element configs
            df = self.get_element_count_as_dataframe()
        elif element_configs == 1:  # only individual elements
            df = self.get_individual_elements_as_dataframe(
                split_symbol=['_', '.'])
        elif element_configs == 2:  # both element configs + individual
            df = self.combine_individual_and_element_counts_as_dataframe(
                split_symbol=['_', '.'])
        else:
            raise ValueError(
                "element_configs can only be 0, 1, or 2. "
                "0 returns only the element configurations given in "
                "self.element_list ({}). "
                "1 returns the individual elements ({}). "
                "2 returns a combination of 1 and 2.".format(
                    self.element_list,
                    get_individual_elements_from_element_list(
                        self.element_list,
                        split_symbol=split_symbol)))

        if flip_colrows:
            df = df.T
        df.plot.bar(fontsize=fontsize)
        plt.title(title, fontsize=fontsize + 4)
        plt.ylabel('Element Count', fontsize=fontsize)
        plt.legend(loc=0, fontsize=fontsize - 4)
        # plt.gca().axes.get_xaxis().set_visible(False)
        plt.tight_layout()

    def set_calibration_area(self, manual_list=None):
        '''
        Area that will be used to calibrate a simulation. The pixel separation
        can be set with set_calibration_separation. The average intensity of
        the atoms chosen by this separation within the area will be set to 1.
        The idea is to calibrate the experimental and simulated images with a
        known intensity (e.g., single Mo atom in MoS2 is relatively
        consistant).

        reference_image can be changed via the Model_refiner.reference_image
        attribute.

        manual_list must be a list of two lists, each of length two.
        For example: [ [0,0], [50, 50] ]
        '''

        if isinstance(manual_list, list):
            self.calibration_area = manual_list
        else:
            self.calibration_area = choose_points_on_image(
                self.reference_image)

    def set_calibration_separation(self, pixel_separation):
        self.calibration_separation = pixel_separation

    def update_error_between_comparison_and_reference_image(self):

        mse_number, ssm_number = measure_image_errors(
            imageA=self.reference_image.data,
            imageB=self.comparison_image.data,
            filename=None)

        self.error_between_comparison_and_reference_image = [
            mse_number, ssm_number]
        self.error_between_images_history.append(
            self.error_between_comparison_and_reference_image)

    def plot_error_between_comparison_and_reference_image(self, style='plot'):

        errors = self.error_between_images_history

        mse = [i[0] for i in errors]
        ssm = [i[1] for i in errors]
        x = range(0, len(errors))

        plt.figure()
        if style == 'plot':
            plt.plot(x, mse, color='b', marker='o',
                     linestyle='-', label='Mean Square Error')
            plt.plot(x, ssm, color='r', marker='^',
                     linestyle='--', label='Struc. Sim. Index')
        elif style == 'scatter':
            plt.scatter(x=x, y=mse, color='b', marker='o',
                        label='Mean Square Error')
            plt.scatter(x=x, y=ssm, color='r', marker='^',
                        label='Struc. Sim. Index')
        plt.title("Reference and Comparison Image Diff.", fontsize=16)
        plt.xlabel("Simulation Order")
        plt.ylabel("MSE and SSM Value")
        plt.legend()

    def update_sublattices_positions(self, sublattice_list):

        positions = get_positions_from_sublattices(sublattice_list)
        self._sublattices_positions = positions

    def set_thickness(self, thickness):
        self.thickness = thickness

    def set_image_xyz_sizes(self, xyz_list):
        self.image_xyz_sizes = xyz_list

    # tools for reverting to the previous version of the refiner
    def _save_refiner_instance(self):

        self.previous_refiner_instance = copy.deepcopy(self)

    def revert_to_previous_refiner_instance(self):
        print("This doesn't seem to work as intended."
              "If you wish to revert to the previous version of the refiner, "
              "you can create a new object. For example: "
              "refiner_2 = refiner_1.previous_refiner_instance")
        self = self.previous_refiner_instance

    # tools for using refinement algorithm
    def image_difference_intensity_model_refiner(
            self,
            sublattices='all',
            comparison_image='default',
            change_sublattice=True,
            filename=None,
            verbose=False,
            refinement_method="Intensity"):
        '''
        Change the elements for sublattice atom positions that don't match with
        comparison_image.

        See image_difference_intensity for details.

        Parameters
        ----------
        See image_difference_intensity for parameter information.
        refinement_method : string, default "Intensity"
            Name passed to self.refinement_history for tracking purposes.

        Examples
        --------
        >>> from temul.model_refiner import (
        ...     get_model_refiner_with_12_vacancies_refined)
        >>> refiner = get_model_refiner_with_12_vacancies_refined(
        ...     image_noise=True)
        Changing some atoms
        Changing some atoms

        >>> history = refiner.element_count_history_list
        >>> history_df = refiner.get_element_count_as_dataframe()
        >>> refiner.combine_individual_and_element_counts_as_dataframe()
                         Ti_0  Ti_1   Ti_2  Ti_3  Ti_4  Ti_5     Ti
        0 Initial State   0.0   0.0  400.0   0.0   0.0   0.0  800.0
        1 Intensity       0.0  12.0  388.0   0.0   0.0   0.0  788.0
        2 Intensity      12.0   0.0  388.0   0.0   0.0   0.0  776.0
        3 Intensity      12.0   0.0  388.0   0.0   0.0   0.0  776.0

        >>> refiner.plot_element_count_as_bar_chart(
        ...     element_configs=2, flip_colrows=True, fontsize=24)

        Flip the view of the data

        >>> refiner.plot_element_count_as_bar_chart(
        ...     element_configs=2, flip_colrows=False, fontsize=24)
        >>> refiner.sublattice_list[0].plot()
        >>> refiner.comparison_image.plot()

        '''

        self._comparison_image_warning()
        self._save_refiner_instance()

        # define variables for refinement
        if 'all' in sublattices:
            sublattice_list = self.sublattice_list
            element_list = self.element_list
            mask_radius_list = self.auto_mask_radius
        elif isinstance(sublattices, list):
            sublattice_list = [self.sublattice_list[i] for i in sublattices]
            element_list = [self.element_list[i] for i in sublattices]
            mask_radius_list = [self.auto_mask_radius[i] for i in sublattices]

        if 'default' in comparison_image:
            comparison_image = self.comparison_image

        for sublattice, element_list_i, mask_radius in zip(
                sublattice_list, element_list, mask_radius_list):

            image_difference_intensity(
                sublattice=sublattice,
                sim_image=comparison_image,
                element_list=element_list_i,
                filename=filename,
                percent_to_nn=None,
                mask_radius=mask_radius,
                change_sublattice=change_sublattice,
                verbose=verbose)

        self.update_element_count_and_refinement_history(refinement_method)

    def repeating_intensity_refinement(self, n=5,
                                       sublattices='all',
                                       comparison_image='default',
                                       change_sublattice=True,
                                       filename=None,
                                       verbose=False,
                                       ignore_element_count_comparison=False):

        self._comparison_image_warning()
        self._save_refiner_instance()

        for i in range(n):
            self.image_difference_intensity_model_refiner(
                sublattices=sublattices,
                comparison_image=comparison_image,
                change_sublattice=change_sublattice,
                filename=filename,
                verbose=verbose)

            if not ignore_element_count_comparison:

                if self.compare_latest_element_counts():
                    print("The latest refinement did not change the model. "
                          "Exiting the refinement after {} loops. To ignore "
                          "this, set ignore_element_count_comparison=True."
                          .format(i + 1))
                    break

    # chosen_sublattice should be less ambiguous

    def image_difference_position_model_refiner(
            self,
            chosen_sublattice,
            sublattices='all',
            comparison_sublattice_list='auto',
            comparison_image='default',
            pixel_threshold=5,
            num_peaks=5,
            inplace=True,
            filename=None,
            refinement_method="Position"):
        '''
        Find new atom positions that were perhaps missed by the initial
        position finding steps. Each sublattice will be updated with new atom
        positions if new atom positions are found.

        See image_difference_position for details.

        Parameters
        ----------
        See image_difference_position for parameter information.
        refinement_method : string, default "Position"
            Name passed to self.refinement_history for tracking purposes.

        Examples
        --------
        >>> from temul.model_refiner import (
        ...     get_model_refiner_one_sublattice_3_vacancies)
        >>> refiner = get_model_refiner_one_sublattice_3_vacancies()
        >>> refiner.sublattice_list[0].plot()
        >>> refiner.comparison_image.plot()
        >>> refiner.image_difference_position_model_refiner(
        ...     pixel_threshold=10,
        ...     chosen_sublattice=refiner.sublattice_list[0])
        3 new atoms found! Adding new atom positions.

        >>> refiner.sublattice_list[0].plot()

        Combination of Refinements (cont.)

        >>> refiner.image_difference_intensity_model_refiner()
        Changing some atoms

        >>> refiner.image_difference_intensity_model_refiner()
        Changing some atoms

        >>> history_df = refiner.get_element_count_as_dataframe()
        >>> refiner.plot_element_count_as_bar_chart(
        ...     element_configs=2, flip_colrows=True, fontsize=24)

        '''

        self._comparison_image_warning()
        self._save_refiner_instance()

        # define variables for refinement
        if 'all' in sublattices:
            sublattice_list = self.sublattice_list
            mask_radius_list = self.auto_mask_radius
        elif isinstance(sublattices, list):
            sublattice_list = [self.sublattice_list[i] for i in sublattices]
            print(
                "Warning: Setting `sublattices` to a list can cause overwrite"
                "errors. Best use `sublattice='all'` until this is fixed")
            mask_radius_list = [self.auto_mask_radius[i] for i in sublattices]

        # elif isinstance(positions_from_sublattices, list):
        #     positions_from_sublattices = [
        #         positions_from_sublattices[i] for i in
        #         positions_from_sublattices] # not correct, would be nice to
        #         have

        if 'default' in comparison_image:
            comparison_image = self.comparison_image

        # if you set sublattices=[1], you get an error that the line
        # self.sublattice_list[i] = sublattice will overwrite the first
        # sublattice with the sublattice_list[1] sublattice! could just not
        # have sublattice as an option! see print warning above!
        for i, (sublattice, mask_radius) in enumerate(zip(
                sublattice_list, mask_radius_list)):

            if sublattice == chosen_sublattice:

                # update the positions_from_sublattices before running the next
                # sublattice if chosen
                if comparison_sublattice_list == 'auto':

                    comparison_sublattice_list = self.sublattice_list

                elif comparison_sublattice_list == 'each':
                    comparison_sublattice_list = []
                    comparison_sublattice_list.append(self.sublattice_list[i])

                sublattice = image_difference_position(
                    sublattice=sublattice,
                    sim_image=comparison_image,
                    pixel_threshold=pixel_threshold,
                    comparison_sublattice_list=comparison_sublattice_list,
                    filename=filename,
                    percent_to_nn=None,
                    mask_radius=mask_radius,
                    num_peaks=num_peaks,
                    inplace=inplace)

                self.sublattice_list[i] = sublattice

        self.update_element_count_and_refinement_history(refinement_method)

    def create_simulation(
            self,
            sublattices='all',
            filter_image=False,
            calibrate_image=True,
            xyz_sizes=None,
            header_comment='example',
            filename='refiner_simulation',
            reference_image='auto',
            probeStep=1.0,
            E0=60e3,
            integrationAngleMin=0.085,
            integrationAngleMax=0.186,
            detectorAngleStep=0.001,
            interpolationFactor=16,
            realspacePixelSize=0.0654,
            numFP=1,
            cellDimXYZ=None,
            tileXYZ=None,
            probeSemiangle=0.030,
            alphaBeamMax=0.032,
            scanWindowMin=0.0,
            scanWindowMax=1.0,
            numThreads=2,
            algorithm="prism",
            delta_image_filter=0.5,
            max_sigma=6,
            mask_radius='auto',
            percent_to_nn=None,
            refine=True,
            sigma_offset=0):
        """
        Create and simulate a .xyz file from the sublattice information.
        Uses the pyprismatic prism algorithm by default.

        Note: The error resulting from filter=True with percent_to_nn set can
        be fixed by setting mask_radius instead of percent_to_nn.
        Error is: "TypeError: 'NoneType' object is not iterable"

        Parameters
        ----------
        sublattices : Atomap Sublattices, default 'all'
            If set to 'all', sublattices that exist in the `Model_Refiner` will
            all be used. The `sublattice` indexes can be specified in a list.
            For example [0, 2] will select the first and third sublattices.
            A list of `sublattice` objects can instead be used.
        filter_image : Bool, default False
            Choose whether to filter the simulation with a Gaussian to match
            the `reference_image`.
        calibrate_image : Bool, default True
            Choose whether to calibrate the simulation with a set
            `calibration_area` and `calibration_separation`.
        xyz_sizes : list, default None
            List of the x, y, z physical size of the reference_image in
            angstrom. If None is set, the sizes will be automatically taken
            from the `image_xyz_sizes` attribute.
        header_comment : string, default 'example'
            The first line comment for the .xyz file.
        filename : string, default 'refiner_simulation'
            filename with which the .xyz file and simulated .mrc file will be
            saved.
        delta_image_filter : float, default 0.5
            The change in sigma for the `filter` Gaussian filter. Small values
            will slow the `filter` down.
        max_sigma : float, default 6
            The maximum sigma used for the `filter` Gaussian filter. Large
            values will slow the `filter` down.
        percent_to_nn : float, default None
            The percentage distance to the nearest neighbour atom used for
            atomap atom position refinement.
        mask_radius : float, default 'auto'
            The pixel distance to the nearest neighbour atom used for atomap
            atom position refinement
        refine : Bool, default True
            Whether to refine the atom positions for the `calibrate` parameter.

        Note: The error resulting from filter=True with percent_to_nn set can
        be fixed by setting mask_radius instead of percent_to_nn.

        Examples
        --------


        Returns
        -------
        Updates the comparison_image attribute with the newly simulated model.
        """

        if 'all' in sublattices:
            sublattice_list = self.sublattice_list
            element_list = self.flattened_element_list
        elif isinstance(sublattices, list):
            sublattice_list = [self.sublattice_list[i] for i in sublattices]
            element_list = [self.element_list[i] for i in sublattices]
            element_list = combine_element_lists(element_list)

        if xyz_sizes is None:
            x_size = self.image_xyz_sizes[0]
            y_size = self.image_xyz_sizes[1]
            z_size = self.image_xyz_sizes[2]

        create_dataframe_for_xyz(
            sublattice_list=sublattice_list,
            element_list=element_list,
            x_size=x_size,
            y_size=y_size,
            z_size=z_size,
            filename=filename + '_xyz_file',
            header_comment=header_comment)

        if reference_image == 'auto':
            reference_image = self.reference_image

        simulate_with_prismatic(
            xyz_filename=filename + '_xyz_file.xyz',
            filename=filename + '_mrc_file',
            reference_image=reference_image,
            probeStep=probeStep,
            E0=E0,
            integrationAngleMin=integrationAngleMin,
            integrationAngleMax=integrationAngleMax,
            detectorAngleStep=detectorAngleStep,
            interpolationFactor=interpolationFactor,
            realspacePixelSize=realspacePixelSize,
            numFP=numFP,
            cellDimXYZ=cellDimXYZ,
            tileXYZ=tileXYZ,
            probeSemiangle=probeSemiangle,
            alphaBeamMax=alphaBeamMax,
            scanWindowMin=scanWindowMin,
            scanWindowMax=scanWindowMax,
            algorithm=algorithm,
            numThreads=numThreads)

        simulation = load_prismatic_mrc_with_hyperspy(
            'prism_2Doutput_' + filename + '_mrc_file.mrc', save_name=None)

        if mask_radius == 'auto':
            mask_radius = np.mean(self.auto_mask_radius)

        if filter_image:
            # set filename as not None - good to output the mse ssm plot
            simulation = compare_two_image_and_create_filtered_image(
                image_to_filter=simulation,
                reference_image=self.reference_image,
                delta_image_filter=delta_image_filter,
                cropping_area=self.calibration_area,
                separation=self.calibration_separation,
                filename=None,
                max_sigma=max_sigma,
                percent_to_nn=percent_to_nn,
                mask_radius=mask_radius,
                refine=False,
                sigma_offset=sigma_offset)

        if calibrate_image:
            calibrate_intensity_distance_with_sublattice_roi(
                image=simulation,
                cropping_area=self.calibration_area,
                separation=self.calibration_separation,
                filename=None,
                reference_image=self.reference_image,
                percent_to_nn=percent_to_nn,
                mask_radius=mask_radius,
                refine=refine,
                scalebar_true=True)

        self._save_refiner_instance()
        self._comparison_image_init(simulation)
        self.update_error_between_comparison_and_reference_image()

    def calibrate_comparison_image(self,
                                   filename=None,
                                   percent_to_nn=None,
                                   mask_radius='auto',
                                   refine=True):

        self._comparison_image_warning()

        if mask_radius == 'auto':
            mask_radius = np.mean(self.auto_mask_radius)

        calibrate_intensity_distance_with_sublattice_roi(
            image=self.comparison_image,
            cropping_area=self.calibration_area,
            separation=self.calibration_separation,
            filename=filename,
            reference_image=self.reference_image,
            percent_to_nn=percent_to_nn,
            mask_radius=mask_radius,
            refine=refine,
            scalebar_true=True)

    def plot_reference_and_comparison_images(
            self, title_reference='Reference', title_comparison='Comparison'):
        _, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(self.reference_image.data)
        ax2.imshow(self.comparison_image.data)
        ax1.set_title(title_reference)
        ax2.set_title(title_comparison)
        plt.show()

# a_list = Counter({'Mo': 1, 'F': 5, 'He': 9})
# b_list = ['init', 'one', 'two']
# for i, (a, b) in enumerate(zip(a_list, b_list)):

#     print(i)
#     print(a)
#     print(b)


# df = pd.DataFrame(columns=a_list)

# i = 0
# a = 'Mo'
# b = 'one'

# for i, (a, b) in enumerate(zip(a_list, b_list)):

#     df = df.append({'Mo': 3}, ignore_index=True).fillna(0)

# for i, b in enumerate(b_list):
#     df.rename(index={i: b}, inplace=True)
# df


''' Model Refiner Dummy Data examples '''


def get_model_refiner_two_sublattices():

    atom_lattice = am_dev.dummy_data.get_simple_atom_lattice_two_sublattices()
    sub1 = atom_lattice.sublattice_list[0]
    sub2 = atom_lattice.sublattice_list[1]
    for i in range(0, len(sub1.atom_list)):
        if i // 2 == 0:
            sub1.atom_list[i].elements = 'Ti_2'
        else:
            sub1.atom_list[i].elements = 'Ti_1'
    for i in range(0, len(sub2.atom_list)):
        if i // 2 == 0:
            sub2.atom_list[i].elements = 'Cl_2'
        else:
            sub1.atom_list[i].elements = 'Cl_3'

    sub1.atom_list[2].elements = 'Ti_3'
    sub1.atom_list[5].elements = 'Ti_3'
    sub2.atom_list[3].elements = 'Cl_1'
    sub2.atom_list[4].elements = 'Cl_1'
    sub1_element_list = ['Ti_0', 'Ti_1', 'Ti_2', 'Ti_3']
    sub2_element_list = ['Cl_0', 'Cl_1', 'Cl_2', 'Cl_3']

    refiner_dict = {sub1: sub1_element_list,
                    sub2: sub2_element_list}
    blurred_image = hyperspy._signals.signal2d.Signal2D(
        gaussian_filter(atom_lattice.image, 3))
    atom_lattice_refiner = Model_Refiner(refiner_dict, blurred_image)
    return atom_lattice_refiner


def get_model_refiner_two_sublattices_refined(n=4):

    refined_model = get_model_refiner_two_sublattices()
    refined_model.repeating_intensity_refinement(
        n=n,
        ignore_element_count_comparison=True)

    return refined_model


def get_model_refiner_one_sublattice_varying_amp(
        image_noise=False, amplitude=[0, 5],
        test_element='Ti_2'):

    sub1 = get_simple_cubic_sublattice(image_noise=image_noise,
                                       amplitude=amplitude)
    for i in range(0, len(sub1.atom_list)):
        sub1.atom_list[i].elements = test_element

    test_element_info = split_and_sort_element(test_element)
    sub1_element_list = auto_generate_sublattice_element_list(
        material_type='single_element_column',
        elements=test_element_info[0][1],
        max_number_atoms_z=np.max(amplitude))

    refiner_dict = {sub1: sub1_element_list}
    comparison_image = get_simple_cubic_signal(
        image_noise=image_noise,
        amplitude=test_element_info[0][2])
    refiner = Model_Refiner(refiner_dict, comparison_image)
    return refiner


def get_model_refiner_one_sublattice_3_vacancies(
        image_noise=True, test_element='Ti_2'):

    # make one with more vacancies maybe
    sub1 = am_dev.dummy_data.get_simple_cubic_with_vacancies_sublattice(
        image_noise=image_noise)
    for i in range(0, len(sub1.atom_list)):
        sub1.atom_list[i].elements = test_element

    test_element_info = split_and_sort_element(test_element)
    sub1_element_list = auto_generate_sublattice_element_list(
        material_type='single_element_column',
        elements=test_element_info[0][1],
        max_number_atoms_z=test_element_info[0][2] + 3)

    refiner_dict = {sub1: sub1_element_list}
    comparison_image = am_dev.dummy_data.get_simple_cubic_signal(
        image_noise=image_noise)
    refiner = Model_Refiner(refiner_dict, comparison_image)
    refiner.auto_mask_radius = [4]

    return refiner


def get_model_refiner_one_sublattice_12_vacancies(
        image_noise=True, test_element='Ti_2'):

    # make one with more vacancies maybe
    sublattice = get_simple_cubic_sublattice(
        image_noise=image_noise,
        with_vacancies=False)
    sub1_atom_positions = np.array(sublattice.atom_positions).T
    sub1_image = get_simple_cubic_signal(
        image_noise=image_noise,
        with_vacancies=True)
    sub1 = am_dev.Sublattice(sub1_atom_positions, sub1_image)
    for i in range(0, len(sub1.atom_list)):
        sub1.atom_list[i].elements = test_element

    test_element_info = split_and_sort_element(test_element)
    sub1_element_list = auto_generate_sublattice_element_list(
        material_type='single_element_column',
        elements=test_element_info[0][1],
        max_number_atoms_z=test_element_info[0][2] + 3)

    refiner_dict = {sub1: sub1_element_list}
    comparison_image = sublattice.signal
    refiner = Model_Refiner(refiner_dict, comparison_image)
    return refiner


def get_model_refiner_with_12_vacancies_refined(
        image_noise=True, test_element='Ti_2', filename=None):

    refined_model = get_model_refiner_one_sublattice_12_vacancies(
        image_noise=image_noise, test_element=test_element)
    refined_model.auto_mask_radius = [4]
    refined_model.image_difference_intensity_model_refiner(filename=filename)
    refined_model.image_difference_intensity_model_refiner(filename=filename)

    # now that the vacancies are correctly labelled as Ti_0, we should use
    # a comparison image that actually represents what might be simulated.
    refined_model.comparison_image = refined_model.sublattice_list[0].signal
    refined_model.image_difference_intensity_model_refiner(filename=filename)

    return refined_model


def get_model_refiner_with_3_vacancies_refined(
        image_noise=True, test_element='Ti_2', filename=None):
    '''
    >>> from temul.model_refiner import (
    ...     get_model_refiner_with_3_vacancies_refined)
    >>> refiner = get_model_refiner_with_3_vacancies_refined()
    3 new atoms found! Adding new atom positions.
    Changing some atoms
    Changing some atoms
    >>> history = refiner.get_element_count_as_dataframe()
    '''
    refiner = get_model_refiner_one_sublattice_3_vacancies(
        image_noise=image_noise, test_element=test_element)
    refiner.auto_mask_radius = [4]

    refiner.image_difference_position_model_refiner(
        pixel_threshold=10, filename=filename,
        chosen_sublattice=refiner.sublattice_list[0])

    refiner.image_difference_intensity_model_refiner(filename=filename)
    refiner.image_difference_intensity_model_refiner(filename=filename)

    return refiner
