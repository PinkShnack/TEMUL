
# import all here and change name to ~ api

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







"""
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
"""