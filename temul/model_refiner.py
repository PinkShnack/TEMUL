
from temul.model_creation import (
    count_atoms_in_sublattice_list,
    compare_count_atoms_in_sublattice_list,
    image_difference_intensity,
    image_difference_position,
    count_all_individual_elements
)
from temul.element_tools import (
    get_individual_elements_from_element_list,
    combine_element_lists
)
from temul.io import create_dataframe_for_xyz
from temul.signal_processing import (
    compare_two_image_and_create_filtered_image,
    calibrate_intensity_distance_with_sublattice_roi
)
from temul.simulations import (
    simulate_with_prismatic,
    load_prismatic_mrc_with_hyperspy
)
import pandas as pd
import hyperspy
from atomap.initial_position_finding import add_atoms_with_gui
import matplotlib.pyplot as plt


class Model_Refiner():
    def __init__(self, sublattice_and_elements_dict,
                 comparison_image=None, sampling=None,
                 thickness=10, name=''):
        '''
        Object which is used to refine the elements in a
        sublattice object

        thickness of sample in angstrom
        sampling in angstrom per pixel

        do:
        auto atomic radii:
        mask_radius_sub2 = atomic_radii_in_pixels(real_sampling, 'S')

        '''

        self.sublattice_and_elements_dict = sublattice_and_elements_dict
        self.sublattice_list = list(sublattice_and_elements_dict.keys())
        self.element_list = list(sublattice_and_elements_dict.values())
        self.flattened_element_list = combine_element_lists(
            self.element_list)
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

        self.calibration_area = []
        self.calibration_separation = 12

        self.reference_image = self.sublattice_list[0].signal

        # maybe have a _sampling_init function
        if sampling is None:
            self.sampling = self.reference_image.axes_manager[-1].scale
        else:
            self.sampling = sampling
        if self.reference_image.axes_manager[-1].units == 'nm':
            self.sampling = self.sampling * 10

        self._reference_image_init()

        self.thickness = thickness
        self.image_xyz_sizes = [
            self.sampling * self.reference_image.data.shape[-1],
            self.sampling * self.reference_image.data.shape[-2],
            self.thickness]

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
                    "hyperspy._signals.signal2d.Signal2D. The current incorrect "
                    "type is {}".format(str(type(comparison_image))))

            for sublattice in self.sublattice_list:
                if not comparison_image.data.shape == sublattice.image.shape:
                    print("Warning: "
                          "comparison_image must have the same shape as each "
                          "sublattice image. comparison_image shape is {}, "
                          "while sublattice '{}' is {}. This will stop you from "
                          "refining your model.".format(
                              comparison_image.data.shape,
                              sublattice.name,
                              sublattice.image.data))

            comparison_image.axes_manager = self.reference_image.axes_manager

        self.comparison_image = comparison_image

    def comparison_image_warning(self, error_message=['None', 'wrong_size']):
        if 'None' in error_message:
            if self.comparison_image is None:
                raise ValueError(
                    "The comparison_image attribute has not been "
                    "set. You will not be able to "
                    "refine the model until a comparison_image is set. You can "
                    "do this via Model_refiner.create_simulation() or by "
                    "setting the Model_Refiner.comparison_image to an image.")

        if 'wrong_size' in error_message:
            for sublattice in self.sublattice_list:
                if not self.comparison_image.data.shape == sublattice.image.shape:
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

    def set_calibration_area(self):
        '''
        Area that will be used to calibrate a simulation. The pixel separation
        can be set with set_calibration_separation. The average intensity of
        the atoms chosen by this separation within the area will be set to 1.
        The idea is to calibrate the experimental and simulated images with a
        known intensity (e.g., single Mo atom in MoS2 is relatively
        consistant).

        reference_image can be changed via the Model_refiner.reference_image
        attribute.
        '''

        self.calibration_area = add_atoms_with_gui(self.reference_image)

    def set_calibration_separation(self, pixel_separation):
        self.calibration_separation = pixel_separation

    def image_difference_intensity_model_refiner(
            self,
            sublattices='all',
            comparison_image='default',
            change_sublattice=True,
            percent_to_nn=0.40,
            mask_radius=None,
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
        >>> from temul.dummy_data import (
        ...     get_model_refiner_with_12_vacancies_refined)
        >>> refiner = get_model_refiner_with_12_vacancies_refined(
        ...     image_noise=True)
        Changing some atoms
        Changing some atoms
        >>> history = refiner.element_count_history_list
        >>> refiner.combine_individual_and_element_counts_as_dataframe()
                         Ti_0  Ti_1   Ti_2  Ti_3  Ti_4  Ti_5     Ti
        0 Initial State   0.0   0.0  400.0   0.0   0.0   0.0  800.0
        1 Intensity       0.0  12.0  388.0   0.0   0.0   0.0  788.0
        2 Intensity      12.0   0.0  388.0   0.0   0.0   0.0  776.0
        3 Intensity      12.0   0.0  388.0   0.0   0.0   0.0  776.0
        >>> refiner.plot_element_count_as_bar_chart(
        ...     element_configs=2, flip_colrows=True, fontsize=24)
        >>> refiner.plot_element_count_as_bar_chart(
        ...     element_configs=2, flip_colrows=False, fontsize=24)
        >>> refiner.sublattice_list[0].plot()
        >>> refiner.comparison_image.plot()

        '''

        self.comparison_image_warning()

        # define variables for refinement
        if 'all' in sublattices:
            sublattice_list = self.sublattice_list
            element_list = self.element_list
        elif isinstance(sublattices, list):
            sublattice_list = [self.sublattice_list[i] for i in sublattices]
            element_list = [self.element_list[i] for i in sublattices]

        if 'default' in comparison_image:
            comparison_image = self.comparison_image

        for sublattice, element_list_i in zip(
                sublattice_list, element_list):

            image_difference_intensity(
                sublattice=sublattice,
                sim_image=comparison_image,
                element_list=element_list_i,
                filename=filename,
                percent_to_nn=percent_to_nn,
                mask_radius=mask_radius,
                change_sublattice=change_sublattice,
                verbose=verbose)

        self.update_element_count_and_refinement_history(refinement_method)

    def repeating_intensity_refinement(self, n=5,
                                       sublattices='all',
                                       comparison_image='default',
                                       change_sublattice=True,
                                       percent_to_nn=0.40,
                                       mask_radius=None,
                                       filename=None,
                                       verbose=False,
                                       ignore_element_count_comparison=False):

        self.comparison_image_warning()

        for i in range(n):
            self.image_difference_intensity_model_refiner(
                sublattices=sublattices,
                comparison_image=comparison_image,
                change_sublattice=change_sublattice,
                percent_to_nn=percent_to_nn,
                mask_radius=mask_radius,
                filename=filename,
                verbose=verbose)

            if not ignore_element_count_comparison:

                if self.compare_latest_element_counts():
                    print("The latest refinement did not change the model. "
                          "Exiting the refinement after {} loops. To ignore "
                          "this, set ignore_element_count_comparison=True."
                          .format(i + 1))
                    break

    def image_difference_position_model_refiner(self,
                                                sublattices='all',
                                                comparison_image='default',
                                                pixel_threshold=5,
                                                num_peaks=5,
                                                inplace=True,
                                                percent_to_nn=0.40,
                                                mask_radius=None,
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
        >>> from temul.dummy_data import (
        ...     get_model_refiner_one_sublattice_3_vacancies)
        >>> refiner = get_model_refiner_one_sublattice_3_vacancies()
        >>> refiner.sublattice_list[0].plot()
        >>> refiner.comparison_image.plot()
        >>> refiner.image_difference_position_model_refiner(
        ...     pixel_threshold=10)
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

        self.comparison_image_warning()

        # define variables for refinement
        if 'all' in sublattices:
            sublattice_list = self.sublattice_list
        elif isinstance(sublattices, list):
            sublattice_list = [self.sublattice_list[i] for i in sublattices]

        if 'default' in comparison_image:
            comparison_image = self.comparison_image

        for i, sublattice in enumerate(sublattice_list):

            sublattice = image_difference_position(
                sublattice=sublattice,
                sim_image=comparison_image,
                pixel_threshold=pixel_threshold,
                filename=filename,
                percent_to_nn=percent_to_nn,
                mask_radius=mask_radius,
                num_peaks=num_peaks,
                inplace=inplace)

            self.sublattice_list[i] = sublattice

        self.update_element_count_and_refinement_history(refinement_method)

    def create_simulation(
            self,
            sublattices='all',
            filter_image=True,
            calibrate_image=True,
            xyz_sizes=None,
            header_comment='example',
            filename='refiner_simulation',
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
            percent_to_nn=0.4,
            mask_radius=None,
            refine=True):

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

        simulate_with_prismatic(
            xyz_filename=filename + '_xyz_file.xyz',
            filename=filename + '_mrc_file',
            reference_image=self.reference_image,
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

        if filter_image:
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
                refine=False)

        if calibrate_image:
            calibrate_intensity_distance_with_sublattice_roi(
                image=simulation,
                cropping_area=self.calibration_area,
                separation=self.calibration_separation,
                filename=filename,
                reference_image=self.reference_image,
                percent_to_nn=percent_to_nn,
                mask_radius=mask_radius,
                refine=refine,
                scalebar_true=True)
        
        self._comparison_image_init(simulation)

    def calibrate_comparison_image(
            self, filename=None, percent_to_nn=0.4, mask_radius=None,
            refine=True):

        self.comparison_image_warning()

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
