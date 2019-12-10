
from scipy.ndimage import gaussian_filter
from atomap.api import dummy_data
import atomap.api as am
from temul.model_creation import (count_atoms_in_sublattice_list,
                                  compare_count_atoms_in_sublattice_list,
                                  image_difference_intensity)
from collections import Counter
import pandas as pd
import hyperspy.api as hs
import numpy as np


class model_refiner():
    def __init__(self, sublattice_and_elements_dict,
                 comparison_image, name=''):
        '''
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

        self._comparison_image_init(comparison_image)
        self.sublattice_list = list(sublattice_and_elements_dict.keys())
        self.element_list = list(sublattice_and_elements_dict.values())
        self.name = name

        self._element_count = count_atoms_in_sublattice_list(
            self.sublattice_list, filename=None)

        self.element_count_history_list = []
        if self.element_count_history_list is not None:
            self.element_count_history_list.append(self._element_count)

    def _comparison_image_init(self, comparison_image):
        # if not type(  image, hspy 2Dsignal):
        # give an error

        # if size of sublattice.image != comparison image:
        # give an error

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

    def get_element_count_as_dataframe(self):
        # for individ sublattices, need a sublattices param, and have it
        # simply index the self.element_list as in image_difference_intensity_model_refiner

        elements_ = [i for sublist in self.element_list for i in sublist]
        elements_ = list(set(elements_))

        df = pd.DataFrame(columns=elements_)
        for changed_elements in self.element_count_history_list:
            df = df.append(changed_elements, ignore_index=True).fillna(0)
        return df

    def compare_latest_element_counts(self):

        if len(self.element_count_history_list) < 2:
            return False
            # raise ValueError("element_count_history must have at least two "
            #                  "element_counts for comparison")
        else:
            return(compare_count_atoms_in_sublattice_list(
                self.element_count_history_list[-2:]))

    def image_difference_intensity_model_refiner(
            self,
            sublattices='all',
            comparison_image='default',
            change_sublattice=True,
            percent_to_nn=0.40,
            mask_radius=None,
            filename=None):

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

        self.update_element_count_history()

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
                    print("The latest refinement did not change the model."
                          "Exciting the refinement. To ignore this, set "
                          "ignore_element_count_comparison=True")
                    # include number of the loop in "exciting the loop here... with .format(n)"
                    break


'''
listy = [9,8,7,6,4]
indexing_list = [0,8]

new_listy = [listy[i] for i in indexing_list]


test_counter1 = Counter('111')
test_counter2 = Counter('222')
test_counter3 = Counter('333')
test_counter4 = Counter('444')

element_count_history = []
element_count_history.append(test_counter1)
element_count_history.append(test_counter2)
element_count_history.append(test_counter3)
element_count_history.append(test_counter4)

element_list = ['1', '2', '3', '4']

element_count_df = pd.DataFrame(columns=element_list)
element_count_df = pd.DataFrame(columns=element_list).append(
    element_count_history, ignore_index=True).fillna(0)
'''

"""
atom_lattice = am.dummy_data.get_simple_atom_lattice_two_sublattices(
    image_noise=True
)
sub1 = atom_lattice.sublattice_list[0]
sub2 = atom_lattice.sublattice_list[1]

for i in range(0, len(sub1.atom_list)):
    sub1.atom_list[i].elements = 'Ti_2'
for i in range(0, len(sub2.atom_list)):
    sub2.atom_list[i].elements = 'Cl_1'

sub1_element_list = ['Ti_0', 'Ti_1', 'Ti_2', 'Ti_3']
sub2_element_list = ['Cl_0', 'Cl_1', 'Cl_2', 'Cl_3']

refiner_dict = {sub1: sub1_element_list,
                sub2: sub2_element_list}

# testing:

blurry_image = gaussian_filter(atom_lattice.image, 3)
atom_lattice_refiner = model_refiner(refiner_dict, blurry_image,
                                     name='mad refine')

atom_lattice_refiner.element_count_history_list
dataframe_before = atom_lattice_refiner.get_element_count_as_dataframe()


atom_lattice_refiner.image_difference_intensity_model_refiner(
            sublattices='all',
            comparison_image='default',
            change_sublattice=True,
            percent_to_nn=0.40,
            mask_radius=None,
            filename=None)

dataframe_after = atom_lattice_refiner.get_element_count_as_dataframe()


atom_lattice_refiner.repeating_intensity_refinement(n=5,
                                       sublattices='all',
                                       comparison_image='default',
                                       change_sublattice=True,
                                       percent_to_nn=0.40,
                                       mask_radius=None,
                                       filename=None)

dataframe_after_repeat = atom_lattice_refiner.get_element_count_as_dataframe()

"""
