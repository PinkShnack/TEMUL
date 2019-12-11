
from atomap.api import dummy_data
import atomap.api as am
from temul.model_creation import (count_atoms_in_sublattice_list,
                                  compare_count_atoms_in_sublattice_list,
                                  image_difference_intensity)
from collections import Counter
import pandas as pd
import hyperspy
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


class model_refiner():
    def __init__(self, sublattice_and_elements_dict,
                 comparison_image, name=''):
        '''

        counts_int_ref = pd.DataFrame(columns=element_list)
        count_atoms = temul.count_atoms_in_sublattice_list([sub1, sub2, sub3, sub_new], filename=None)
        counts_int_ref = counts_int_ref.append(count_atoms, ignore_index=True).fillna(0)

        indiv_elems = temul.count_all_individual_elements(individual_element_list, counts_int_ref)
        indiv_elems_int_ref = pd.DataFrame.from_dict(indiv_elems)
        counts_int_ref = pd.concat([indiv_elems_int_ref, counts_int_ref], axis=1)

        Object which is used to refine the elements in a
        sublattice object

        can i have sublattice_and_element_list as a dict?

        if you call model_refiner.count_elements it would store the count in
        the object and return a df. You could call a plot elements method too.
        Might make the refine functions easier to use over several sublattices
        as each would normally need the function run separately for each one
        (because different elements etc.).
        You create a model_refiner object, input being the sublattice,
        element_list. just a list/dict of those.

        dict = {sub1 : element_list1,
                sub2 : element_list2, ...}

        IF the counter attribute isn't empy, append! then you simply have a list
        of the history.
        '''

        self.sublattice_list = list(sublattice_and_elements_dict.keys())
        self.element_list = list(sublattice_and_elements_dict.values())
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

    def _comparison_image_init(self, comparison_image):

        if not isinstance(comparison_image, hyperspy._signals.signal2d.Signal2D):
            raise ValueError(
                "comparison_image must be a 2D Hyperspy signal of type "
                "hyperspy._signals.signal2d.Signal2D. The current incorrect "
                "type is {}".format(str(type(comparison_image))))

        for sublattice in self.sublattice_list:
            if not comparison_image.data.shape == sublattice.image.shape:
                raise ValueError(
                    "comparison_image must have the same shape as each "
                    "sublattice image. comparison_image shape is {}, while "
                    "sublattice '{}' is {}".format(comparison_image.data.shape,
                                                   sublattice.name,
                                                   sublattice.image.data))

        self.comparison_image = comparison_image

    def __repr__(self):
        return '<%s, %s (sublattices:%s,element_lists:%s)>' % (
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
            df.rename(index={i: refinement_name}, inplace=True)

        return df

    def plot_element_count_as_bar_chart(self, flip_colrows=True,
                                        title="Refinement of Elements",
                                        fontsize=16):
        df = self.get_element_count_as_dataframe()
        if flip_colrows:
            df = df.T
        df.plot.bar(fontsize=fontsize)
        plt.title(title, fontsize=fontsize)
        plt.ylabel('Element Count', fontsize=fontsize)
        # plt.gca().axes.get_xaxis().set_visible(False)
        plt.tight_layout()

    def image_difference_intensity_model_refiner(
            self,
            sublattices='all',
            comparison_image='default',
            change_sublattice=True,
            percent_to_nn=0.40,
            mask_radius=None,
            filename=None,
            refinement_method="Intensity"):

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
                change_sublattice=change_sublattice)

        self.update_element_count_and_refinement_history(refinement_method)

    def repeating_intensity_refinement(self, n=5,
                                       sublattices='all',
                                       comparison_image='default',
                                       change_sublattice=True,
                                       percent_to_nn=0.40,
                                       mask_radius=None,
                                       filename=None,
                                       ignore_element_count_comparison=False):

        for i in range(n):
            self.image_difference_intensity_model_refiner(
                sublattices=sublattices,
                comparison_image=comparison_image,
                change_sublattice=change_sublattice,
                percent_to_nn=percent_to_nn,
                mask_radius=mask_radius,
                filename=filename)

            if not ignore_element_count_comparison:

                if self.compare_latest_element_counts():
                    print("The latest refinement did not change the model. "
                          "Exiting the refinement after {} loops. To ignore "
                          "this, set ignore_element_count_comparison=True."
                          .format(i+1))
                    # include number of the loop in "exciting the loop here... with .format(n)"
                    break


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
