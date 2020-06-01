
'''
from temul.model_creation import get_max_number_atoms_z
from temul.signal_processing import (get_xydata_from_list_of_intensities,
                                     return_fitting_of_1D_gaussian,
                                     fit_1D_gaussian_to_data,
                                     plot_gaussian_fit,
                                     get_fitting_tools_for_plotting_gaussians,
                                     plot_gaussian_fitting_for_multiple_fits)
from temul.dummy_data import get_simple_cubic_sublattice
import atomap.api as am
import hyperspy.api as hs
import numpy as np

from ase.visualize import view
from ase.io import read, write
from ase.cluster.cubic import FaceCenteredCubic
import ase

import os

import matplotlib.pyplot as plt

import rigidregistration


plt.style.use('default')
# %matplotlib qt


# example fitting of a gaussian to data
amp, mu, sigma = 10, 10, 0.5
sub1_inten = np.random.normal(mu, sigma, 1000)
xdata, ydata = get_xydata_from_list_of_intensities(sub1_inten, hist_bins=50)
popt_gauss, _ = return_fitting_of_1D_gaussian(
    function=fit_1D_gaussian_to_data, xdata=xdata, ydata=ydata,
    amp=amp, mu=mu, sigma=sigma)

# choice of hist or scatter data plot!
plot_gaussian_fit(xdata, ydata, function=fit_1D_gaussian_to_data,
                  amp=popt_gauss[0], mu=popt_gauss[1], sigma=popt_gauss[2],
                  gauss_art='r--', gauss_label='Gauss Fit',
                  plot_data=True, data_art='ko', data_label='Data Points',
                  plot_fill=True, facecolor='r', alpha=0.5)


# fitting to test sample, single atom element.
sublattice = get_simple_cubic_sublattice(
    image_noise=True,
    amplitude=[5, 10])

sublattice.image /= sublattice.image.max()
# sublattice.plot()

sub1_inten = tml.get_sublattice_intensity(sublattice, 'max')
# sub1_inten = np.concatenate((sub1_inten, sub1_inten+0.001, sub1_inten-0.001))

plt.figure()
plt.hist(sub1_inten, bins=150)
plt.show()

plt.figure()
plt.imshow(sublattice.image)
plt.show()

# fit single plot
amp, mu, sigma = 10, 0.27, 0.005
xdata, ydata = get_xydata_from_list_of_intensities(
    sub1_inten, hist_bins=150)
popt_gauss, _ = return_fitting_of_1D_gaussian(
    function=fit_1D_gaussian_to_data, xdata=xdata, ydata=ydata,
    amp=amp, mu=mu, sigma=sigma)

plot_gaussian_fit(xdata, ydata, function=fit_1D_gaussian_to_data,
                  amp=popt_gauss[0], mu=popt_gauss[1], sigma=popt_gauss[2],
                  gauss_art='r--', gauss_label='Gauss Fit',
                  plot_data=True, data_art='ko', data_label='Data Points',
                  plot_fill=True, facecolor='r', alpha=0.5)


# fit all elements in the sublattice.
element_list = tml.auto_generate_sublattice_element_list(
    material_type='single_element_column',
    elements='Cu', max_number_atoms_z=10)

middle_list, limit_list = tml.find_middle_and_edge_intensities(
    sublattice, element_list=element_list,
    standard_element=element_list[-1],
    scaling_exponent=1.0,
    largest_element_intensity=0.96)

fitting_tools = get_fitting_tools_for_plotting_gaussians(
    element_list,
    scaled_middle_intensity_list=middle_list,
    scaled_limit_intensity_list=limit_list,
    gaussian_amp=5,
    gauss_sigma_division=2)

# allow choice of kwargs for displaying, no fill, etc
plot_gaussian_fitting_for_multiple_fits(sub_ints_all=[sub1_inten],
                                        fitting_tools_all_subs=[fitting_tools],
                                        element_list_all_subs=[element_list],
                                        marker_list=[['Sub1', '.']],
                                        hist_bins=100,
                                        filename='Fit of Intensities')

# plot a second fake sublattice
sub2_inten = sub1_inten + 0.01
plot_gaussian_fitting_for_multiple_fits(sub_ints_all=[sub1_inten, sub2_inten],
                                        fitting_tools_all_subs=[
                                            fitting_tools, fitting_tools],
                                        element_list_all_subs=[
                                            element_list, element_list],
                                        marker_list=[
                                            ['Sub1', '.'], ['Sub2', 'x']],
                                        hist_bins=150,
                                        filename='Fit of 2 Intensities',
                                        mpl_cmaps_list=['viridis', 'cividis'])


######## Model Creation Example - Cubic Variation ########

sublattice = tml.dummy_data.get_simple_cubic_sublattice(amplitude=[2, 10])

# could have this happen automatically
# when you choose "normalised" for the sort_sublattice_intensity
# paramater scaler_method
sublattice.image /= sublattice.image.max()

sublattice.plot()

element_list = tml.auto_generate_sublattice_element_list(material_type='single_element_column',
                                                         elements='Au', max_number_atoms_z=10)

middle_list, limit_list = tml.find_middle_and_edge_intensities(
    sublattice, element_list=element_list,
    standard_element=element_list[-1],
    scaling_exponent=1.5)

elements_in_sublattice = tml.sort_sublattice_intensities(
    sublattice, intensity_type='max',
    element_list=element_list, scalar_method=1,
    middle_intensity_list=middle_list,
    limit_intensity_list=limit_list)

tml.assign_z_height_to_sublattice(sublattice,
                                  z_bond_length=1.5,
                                  atom_layout='top')

tml.print_sublattice_elements(sublattice, 10)

df = tml.create_dataframe_for_cif(sublattice_list=[sublattice],
                                  element_list=element_list)

max_number_atoms_z = get_max_number_atoms_z(sublattice=sublattice)
bond_length = 1.5
z_thickness = max_number_atoms_z * bond_length

cif_filename = "sublattice_variation_amp"
tml.write_cif_from_dataframe(dataframe=df,
                             filename=cif_filename,
                             chemical_name_common="sublattice_variation_amp",
                             cell_length_a=40,
                             cell_length_b=40,
                             cell_length_c=z_thickness,
                             cell_angle_alpha=90,
                             cell_angle_beta=90,
                             cell_angle_gamma=90,
                             space_group_name_H_M_alt='P 1',
                             space_group_IT_number=1)

sublattice_structure = read(cif_filename + '.cif')
view(sublattice_structure)


######## Model Simulation Example - Cu NP ########

ase_xyz_filename = "Cu_NP_example.xyz"
prismatic_xyz_filename = "Cu_NP_example_pris.xyz"

surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
layers = [6, 9, 5]
lc = 3.61000
Cu_NP = FaceCenteredCubic('Cu', surfaces, layers, latticeconstant=lc)

view(Cu_NP)

write(filename=ase_xyz_filename, images=Cu_NP)

prismatic_xyz = tml.convert_vesta_xyz_to_prismatic_xyz(
    vesta_xyz_filename=ase_xyz_filename,
    prismatic_xyz_filename=prismatic_xyz_filename,
    edge_padding=(2, 2, 1),
    delimiter='      |       |  ')


tml.simulate_with_prismatic(xyz_filename=prismatic_xyz_filename,
                            filename=prismatic_xyz_filename,
                            reference_image=None,
                            probeStep=0.1,
                            E0=60e3,
                            integrationAngleMin=0.085,
                            integrationAngleMax=0.186,
                            detectorAngleStep=0.001,
                            interpolationFactor=8,
                            realspacePixelSize=0.0654,
                            numFP=1,
                            cellDimXYZ=None,
                            tileXYZ=None,
                            probeSemiangle=0.030,
                            alphaBeamMax=0.032,
                            scanWindowMin=0.0,
                            scanWindowMax=1.0,
                            algorithm="prism",
                            numThreads=2)


simulation = tml.load_prismatic_mrc_with_hyperspy(
    prismatic_mrc_filename='prism_2Doutput_' + prismatic_xyz_filename + '.mrc',
    save_name=prismatic_xyz_filename[:-4])

simulation.plot()


######## Model Creation Example - simulated Cu NP ########

s = tml.example_data.load_example_Cu_nanoparticle()
s.plot()

atom_positions = am.get_atom_positions(s, separation=10, pca=True)
# atom_positions = am.add_atoms_with_gui(image=s, atom_list=atom_positions)
# np.save("Au_NP_atom_positions", atom_positions)
# atom_positions = np.load("Au_NP_atom_positions.npy")

sublattice = am.Sublattice(atom_position_list=atom_positions, image=s)
sublattice.refine_atom_positions_using_center_of_mass()
sublattice.plot()

atom_lattice = am.Atom_Lattice(image=s, name="Cu_NP_sim",
                               sublattice_list=[sublattice])
atom_lattice.save(filename="Cu_NP_sim_Atom_Lattice.hdf5", overwrite=True)


sublattice.image /= sublattice.image.max()

element_list = tml.auto_generate_sublattice_element_list(
    material_type='single_element_column',
    elements='Cu', max_number_atoms_z=7)

middle_list, limit_list = tml.find_middle_and_edge_intensities(
    sublattice, element_list=element_list,
    standard_element=element_list[-1],
    scaling_exponent=1.5)

elements_in_sublattice = tml.sort_sublattice_intensities(
    sublattice, intensity_type='max',
    element_list=element_list, scalar_method=1,
    middle_intensity_list=middle_list,
    limit_intensity_list=limit_list)

z_bond_length = 3.61000
tml.assign_z_height_to_sublattice(sublattice,
                                  z_bond_length=z_bond_length,
                                  atom_layout='center')

tml.print_sublattice_elements(sublattice, 10)

df = tml.create_dataframe_for_cif(sublattice_list=[sublattice],
                                  element_list=element_list)

max_number_atoms_z = get_max_number_atoms_z(sublattice=sublattice)
z_thickness = max_number_atoms_z * z_bond_length

cif_filename = "Create_Cu_NP"

tml.write_cif_from_dataframe(dataframe=df,
                             filename=cif_filename,
                             chemical_name_common=cif_filename,
                             cell_length_a=s.axes_manager['x'].size / 10,
                             cell_length_b=s.axes_manager['y'].size / 10,
                             cell_length_c=z_thickness)

sublattice_structure = read(cif_filename + '.cif')
view(sublattice_structure)

# Simulate created NP
tml.create_dataframe_for_xyz(sublattice_list=[sublattice],
                             element_list=element_list,
                             x_distance=s.axes_manager['x'].size / 10,
                             y_distance=s.axes_manager['y'].size / 10,
                             z_distance=z_thickness,
                             filename=cif_filename,
                             header_comment='top_level_comment')

prismatic_xyz_filename = cif_filename
tml.simulate_with_prismatic(xyz_filename=prismatic_xyz_filename,
                            filename=prismatic_xyz_filename,
                            reference_image=None,
                            probeStep=0.1,
                            E0=60e3,
                            integrationAngleMin=0.085,
                            integrationAngleMax=0.186,
                            detectorAngleStep=0.001,
                            interpolationFactor=4,
                            realspacePixelSize=0.0654,
                            numFP=1,
                            cellDimXYZ=None,
                            tileXYZ=None,
                            probeSemiangle=0.030,
                            alphaBeamMax=0.032,
                            scanWindowMin=0.0,
                            scanWindowMax=1.0,
                            algorithm="prism",
                            numThreads=2)


simulation = tml.load_prismatic_mrc_with_hyperspy(
    prismatic_mrc_filename='prism_2Doutput_' + prismatic_xyz_filename + '.mrc',
    save_name=prismatic_xyz_filename[:-4])

simulation.plot()

# Refine created NP


######## Model Creation Example - Au NP ########

s = tml.example_data.load_example_Au_nanoparticle()
s.plot()

cropping_area = am.add_atoms_with_gui(s.data)

s_crop = tml.crop_image_hs(image=s, cropping_area=cropping_area,
                           save_image=True, save_variables=True,
                           scalebar_true=True)

am.get_feature_separation(signal=s_crop,
                          separation_range=(10, 15), pca=True).plot()

atom_positions = am.get_atom_positions(s_crop, separation=12, pca=True)
atom_positions = am.add_atoms_with_gui(image=s_crop, atom_list=atom_positions)
# np.save("Au_NP_atom_positions", atom_positions)
# atom_positions = np.load("Au_NP_atom_positions.npy")

sublattice = am.Sublattice(atom_position_list=atom_positions, image=s_crop)
sublattice.plot()

atom_lattice = am.Atom_Lattice(image=s_crop, name="Au_NP_1",
                               sublattice_list=[sublattice])
atom_lattice.save(filename="Au_NP_Atom_Lattice.hdf5", overwrite=True)


######## Model Creation Example ########
# Simple 20x20x5 Au supercell

sublattice = am.dummy_data.get_simple_cubic_sublattice()
sublattice

element_list = ['Au_5']

elements_in_sublattice = tml.sort_sublattice_intensities(
    sublattice, element_list=element_list)

tml.assign_z_height_to_sublattice(sublattice, z_bond_length=0.5)

tml.print_sublattice_elements(sublattice, 10)

Au_NP_df = tml.create_dataframe_for_cif(sublattice_list=[sublattice],
                                        element_list=element_list)

tml.write_cif_from_dataframe(dataframe=Au_NP_df,
                             filename="Au_NP_test_01",
                             chemical_name_common="Au_NP",
                             cell_length_a=20,
                             cell_length_b=20,
                             cell_length_c=5,
                             cell_angle_alpha=90,
                             cell_angle_beta=90,
                             cell_angle_gamma=90,
                             space_group_name_H_M_alt='P 1',
                             space_group_IT_number=1)


######## Simulate MoS2 with Prismatic Example ########


'''
'''
Steps

1. get the path to the example data
2. convert from a vesta xyz file to a prismatic xyz file
3. simulate the prismatic xyz file. This outputs a mrc file
4. save the simulated file as a png and hyperspy file

You can add lots to this. For example if you're simulating an experimental
image, see the function simulate_and_calibrate_with_prismatic() for allowing
the experimental(reference) image to calculate the probeStep(sampling).
Only works if the reference image is loaded into python as a hyperspy 2D signal
and if the image is calibrated.
'''


'''

# Step 1

vesta_xyz_filename = tml.example_data.path_to_example_data_MoS2_vesta_xyz()
# print(vesta_xyz_filename)

# set the filenames for opening and closing
prismatic_xyz_filename = 'MoS2_hex_prismatic_2.xyz'
mrc_filename = 'prismatic_simulation_2'
simulated_filename = 'calibrated_data_2_electric_boogaloo'

# Step 2
prismatic_xyz = tml.convert_vesta_xyz_to_prismatic_xyz(
    vesta_xyz_filename=vesta_xyz_filename,
    prismatic_xyz_filename=prismatic_xyz_filename,
    delimiter='   |    |  ',
    header=None,
    skiprows=[0, 1],
    engine='python',
    occupancy=1.0,
    rms_thermal_vib=0.05,
    header_comment="Let's make a file 2, Electric Boogaloo!",
    save=True)

# Step 3
tml.simulate_with_prismatic(
    xyz_filename=prismatic_xyz_filename,
    filename=mrc_filename,
    reference_image=None,
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
    algorithm="prism",
    numThreads=2)

# Step 4
simulation = tml.load_prismatic_mrc_with_hyperspy(
    prismatic_mrc_filename='prism_2Doutput_' + mrc_filename + '.mrc',
    save_name=simulated_filename)

simulation.plot()


"""
Thesis example for monolayer Se implanted MoS2
"""

import hyperspy.api as hs
import numpy as np
import atomap.api as am
import scipy
import matplotlib.pyplot as plt
from atomap.atom_finding_refining import subtract_average_background
from atomap.atom_finding_refining import normalize_signal
from atomap.tools import remove_atoms_from_image_using_2d_gaussian

import temul.example_data as example_data
from temul.element_tools import (
    atomic_radii_in_pixels,
    combine_element_lists
)
from temul.signal_processing import (
    double_gaussian_fft_filter,
    calibrate_intensity_distance_with_sublattice_roi,
    crop_image_hs,
    toggle_atom_refine_position_automatically)
from temul.model_creation import (
    scaling_z_contrast,
    find_middle_and_edge_intensities,
    sort_sublattice_intensities,
    assign_z_height,
    print_sublattice_elements,
    find_middle_and_edge_intensities_for_background,
    correct_background_elements,
    create_dataframe_for_cif,
)
from temul.intensity_tools import get_sublattice_intensity
from temul.io import (
    write_cif_from_dataframe,
    create_dataframe_for_xyz
)

s_original = example_data.load_Se_implanted_MoS2_data()
real_sampling = s_original.axes_manager[-1].scale
s_original.plot()

# define the image name
# put in model refiner for auto saving purposes?
image_name = s_original.metadata.General.original_filename

percent_to_nn = None
percent_to_nn_remove_atoms = 0.4
percent_to_nn_toggle = 0.05
min_cut_off_percent = 0.75
max_cut_off_percent = 1.25
min_cut_off_percent_sub3 = 0.0
max_cut_off_percent_sub3 = 3
mask_radius_sub1 = atomic_radii_in_pixels(real_sampling, 'Mo')
mask_radius_sub2 = atomic_radii_in_pixels(real_sampling, 'S')
d_inner = 7.48
d_outer = 14.96

s_filtered = normalize_signal(subtract_average_background(s_original))

# filter the image
s_filtered = double_gaussian_fft_filter(
    image=s_filtered,
    filename=None,
    d_inner=d_inner,
    d_outer=d_outer,
    delta=0.01,
    real_space_sampling=real_sampling,
    units='nm')

s_filtered.plot()

# Choose a feature separation that only includes the Mo atoms.
# This will be used for calibration (setting average Mo intensity to 1)
features = am.get_feature_separation(
    s_filtered, separation_range=(8, 14), pca=True)
features.plot()
calibration_separation = 11

# choose the area from which to calibrate the image intensity
calibration_area = am.add_atoms_with_gui(s_filtered.data)
# calibration_area = [[431.75, 445.79], [643.14, 635.66]] # example

# calibrate the image
calibrate_intensity_distance_with_sublattice_roi(image=s_filtered,
                                                 cropping_area=calibration_area,
                                                 separation=calibration_separation,
                                                 filename=None,
                                                 percent_to_nn=percent_to_nn,
                                                 mask_radius=mask_radius_sub1,
                                                 scalebar_true=True)

s_filtered.plot()
s = s_filtered

# s.plot()
# Crop the image if you want to analyse a small section
cropping_area = am.add_atoms_with_gui(s.data)
# cropping_area = [[288.71, 64.79], [635.54, 631.86]] # example

s_crop = crop_image_hs(image=s,
                       cropping_area=cropping_area,
                       save_image=False,
                       save_variables=False,
                       scalebar_true=True)
s_crop.plot()
s = s_crop

# Set the fractions for finding the other sublattices and colours
vector_fraction_sub2 = 0.6703
vector_fraction_sub3 = 0.3303
sub1_colour = 'blue'
sub2_colour = 'yellow'
sub3_colour = 'green'
sub1_name = 'sub1'
sub2_name = 'sub2'
sub3_name = 'sub3'
sub3_inverse_name = 'sub3_inverse'


# SUBLATTICE 1
am.get_feature_separation(s, separation_range=(3, 12), pca=True).plot()

separation_sub1 = 7
atom_positions_1_original = am.get_atom_positions(s, separation_sub1, pca=True)
# remove some Se atoms:
atom_positions_1_original = am.add_atoms_with_gui(s, atom_positions_1_original)

np.save(file='atom_positions_1_original', arr=atom_positions_1_original)

sub1 = am.Sublattice(atom_positions_1_original, s,
                     name=sub1_name, color=sub1_colour)
sub1.find_nearest_neighbors()

# Do not allow certain atoms from being refined if they are vacancies etc.
# Automatically :
false_list_sub1 = toggle_atom_refine_position_automatically(
    sublattice=sub1,
    min_cut_off_percent=min_cut_off_percent,
    max_cut_off_percent=max_cut_off_percent * 1.5,
    range_type='internal',
    filename=None,
    percent_to_nn=percent_to_nn,
    mask_radius=mask_radius_sub1)

# Manually & to check:
sub1.toggle_atom_refine_position_with_gui()

sub1.refine_atom_positions_using_center_of_mass(
    percent_to_nn=percent_to_nn, mask_radius=mask_radius_sub1)
sub1.refine_atom_positions_using_2d_gaussian(
    percent_to_nn=percent_to_nn, mask_radius=mask_radius_sub1)
# sub1.plot()

atom_positions_1_refined = np.array(sub1.atom_positions).T
np.save(file='atom_positions_1_refined', arr=atom_positions_1_refined)

sub1.get_atom_list_on_image(markersize=2, color=sub1_colour).plot()
plt.title(sub1_name, fontsize=20)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig(fname=sub1_name + '.png',
            transparent=True, frameon=False, bbox_inches='tight',
            pad_inches=None, dpi=300, labels=False)
plt.close()


# SUBLATTICE 2
# remove first sublattice
sub1_atoms_removed = remove_atoms_from_image_using_2d_gaussian(
    sub1.image, sub1, percent_to_nn=percent_to_nn_remove_atoms)
sub1_atoms_removed = hs.signals.Signal2D(sub1_atoms_removed)
sub1_atoms_removed.plot()

sub1.construct_zone_axes(atom_plane_tolerance=0.5)
sub1.plot_planes()
zone_number = 4
zone_axis_001 = sub1.zones_axis_average_distances[zone_number]
atom_positions_2 = sub1.find_missing_atoms_from_zone_vector(
    zone_axis_001, vector_fraction=vector_fraction_sub2)


# am.get_feature_separation(sub1_atoms_removed).plot()
#atom_positions_2 = am.get_atom_positions(sub1_atoms_removed, 19)
atom_positions_2_original = am.add_atoms_with_gui(
    sub1_atoms_removed, atom_positions_2)
np.save(file='atom_positions_2_original', arr=atom_positions_2_original)

sub2_refining = am.Sublattice(atom_positions_2_original, sub1_atoms_removed,
                              name=sub2_name, color=sub2_colour)
sub2_refining.find_nearest_neighbors()

# Auto
false_list_sub2 = toggle_atom_refine_position_automatically(
    sublattice=sub2_refining,
    min_cut_off_percent=min_cut_off_percent,
    max_cut_off_percent=max_cut_off_percent * 1.5,
    range_type='internal',
    filename=None,
    percent_to_nn=percent_to_nn,
    mask_radius=mask_radius_sub2)

sub2_refining.toggle_atom_refine_position_with_gui()


sub2_refining.refine_atom_positions_using_center_of_mass(
    percent_to_nn=percent_to_nn, mask_radius=mask_radius_sub2)
sub2_refining.refine_atom_positions_using_2d_gaussian(
    percent_to_nn=percent_to_nn, mask_radius=mask_radius_sub2)
# sub2_refining.plot()

sub2_refining.get_atom_list_on_image(markersize=2, color=sub2_colour).plot()
plt.title(sub2_name + '_refining', fontsize=20)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig(fname=sub2_name + '_refining.png',
            transparent=True, frameon=False, bbox_inches='tight',
            pad_inches=None, dpi=300, labels=False)
plt.close()


atom_positions_2_refined = np.array(sub2_refining.atom_positions).T
np.save(file='atom_positions_2_refined', arr=atom_positions_2_refined)

sub2 = am.Sublattice(atom_positions_2_refined, s,
                     name=sub2_name, color=sub2_colour)
sub2.find_nearest_neighbors()
sub2.plot()

sub2.get_atom_list_on_image(markersize=2, color=sub2_colour).plot()
plt.title(sub2_name, fontsize=20)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig(fname=sub2_name + '.png',
            transparent=True, frameon=False, bbox_inches='tight',
            pad_inches=None, dpi=300, labels=False)
plt.close()


# SUBLATTICE 3

atom_positions_3_original = sub1.find_missing_atoms_from_zone_vector(
    zone_axis_001, vector_fraction=vector_fraction_sub3)

# am.get_feature_separation(sub1_atoms_removed).plot()
#atom_positions_2 = am.get_atom_positions(sub1_atoms_removed, 19)
atom_positions_3_original = am.add_atoms_with_gui(s, atom_positions_3_original)
np.save(file='atom_positions_3_original', arr=atom_positions_3_original)

s_inverse = s
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
    min_cut_off_percent=min_cut_off_percent / 2,
    max_cut_off_percent=max_cut_off_percent * 2,
    range_type='internal',
    method='mean',
    filename=None,
    percent_to_nn=percent_to_nn,
    mask_radius=mask_radius_sub2)

sub3_inverse.toggle_atom_refine_position_with_gui()

sub3_inverse.refine_atom_positions_using_center_of_mass(
    percent_to_nn=percent_to_nn, mask_radius=mask_radius_sub2)
sub3_inverse.refine_atom_positions_using_2d_gaussian(
    percent_to_nn=percent_to_nn, mask_radius=mask_radius_sub2)
# sub3_inverse.plot()

atom_positions_3_refined = np.array(sub3_inverse.atom_positions).T
np.save(file='atom_positions_3_refined', arr=atom_positions_3_refined)

s.data = np.divide(1, s.data)
s.plot()

sub3 = am.Sublattice(atom_positions_3_refined, s,
                     name=sub3_name, color=sub3_colour)
sub3.find_nearest_neighbors()
# sub3.plot()

# Now re-refine the adatom locations for the original data
false_list_sub3 = toggle_atom_refine_position_automatically(
    sublattice=sub3,
    min_cut_off_percent=min_cut_off_percent_sub3,
    max_cut_off_percent=max_cut_off_percent_sub3,
    range_type='external',
    filename=None,
    percent_to_nn=percent_to_nn,
    mask_radius=mask_radius_sub2)

sub3.toggle_atom_refine_position_with_gui()

sub3.refine_atom_positions_using_center_of_mass(
    percent_to_nn=percent_to_nn, mask_radius=mask_radius_sub1)
sub3.refine_atom_positions_using_2d_gaussian(
    percent_to_nn=percent_to_nn, mask_radius=mask_radius_sub2)
# sub3.plot()

atom_positions_3_refined = np.array(sub3.atom_positions).T
np.save(file='atom_positions_3_refined', arr=atom_positions_3_refined)

sub3.get_atom_list_on_image(markersize=2).plot()
plt.title(sub3_name, fontsize=20)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig(fname=sub3_name + '.png',
            transparent=True, frameon=False, bbox_inches='tight',
            pad_inches=None, dpi=300, labels=False)
plt.close()

# Now we have the correct, refined positions of the Mo, S and bksubs

# ATOM LATTICE

atom_lattice = am.Atom_Lattice(image=s,
                               name=image_name,
                               sublattice_list=[sub1, sub2, sub3])
atom_lattice.save(filename="Atom_Lattice.hdf5", overwrite=True)

atom_lattice.get_sublattice_atom_list_on_image(markersize=2).plot()
plt.title('Atom Lattice', fontsize=20)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig(fname='Atom_Lattice.png',
            transparent=True, frameon=False, bbox_inches='tight',
            pad_inches=None, dpi=300, labels=False)
plt.close()


# atom_lattice = am.load_atom_lattice_from_hdf5('Run7_max/Atom_Lattice.hdf5')
# sub1 = atom_lattice.sublattice_list[0]
# sub2 = atom_lattice.sublattice_list[1]
# sub3 = atom_lattice.sublattice_list[2]


# assign elements
mask_radius_both = [mask_radius_sub1, mask_radius_sub2]
intensity_type = 'max'
num_points = 3
method = 'mode'
remove_background_method = None
background_sublattice = None
bins_for_hist = 15
image_size_x_nm = real_sampling * s.data.shape[-1]
image_size_y_nm = real_sampling * s.data.shape[-2]
image_size_z_nm = 1.2294 / 2

element_list_sub1 = ['Mo_0', 'Mo_1', 'Mo_1.S_1', 'Mo_1.Se_1', 'Mo_2']
standard_element_sub1 = 'Mo_1'

element_list_sub2 = ['S_0', 'S_1', 'S_2', 'Se_1', 'Se_1.S_1', 'Se_2']
standard_element_sub2 = 'S_2'

element_list_sub3 = ['H_0', 'S_1', 'Se_1', 'Mo_1', ]
elements_from_sub1 = ['Mo_1']
elements_from_sub2 = ['S_1', 'Se_1']

scaling_ratio, scaling_exponent, sub1_mode, sub2_mode = scaling_z_contrast(
    numerator_sublattice=sub1,
    numerator_element='Mo_1',
    denominator_sublattice=sub2,
    denominator_element='S_2',
    intensity_type=intensity_type,
    method=method,
    remove_background_method=remove_background_method,
    background_sublattice=background_sublattice,
    num_points=num_points, percent_to_nn=percent_to_nn,
    mask_radius=mask_radius_both)


# reset the elements column example
# def reset_elements_and_z_height(sublattice_list):
#    for sublattice in sublattice_list:
#        for i in range(0, len(sublattice.atom_list)):
#            sublattice.atom_list[i].elements = ''
#            sublattice.atom_list[i].z_height = ''
#
#reset_elements_and_z_height(sublattice_list=[sub1, sub2, sub3])


# SUB1

# sub1_ints = get_sublattice_intensity(sub1, intensity_type=intensity_type,
#                                     remove_background_method=remove_background_method,
#                                     background_sublattice=background_sublattice,
#                                     num_points=num_points,
#                                     percent_to_nn=percent_to_nn,
#                                     mask_radius=mask_radius_sub1)
#
#scipy.stats.mode(np.round(sub1_ints, decimals=2))[0][0]
# sub1_ints.mean()
# plt.figure()
#plt.hist(sub1_ints, bins=bins_for_hist)
# plt.show()
# sub1.plot()


middle_intensity_list_sub1, limit_intensity_list_sub1 = find_middle_and_edge_intensities(
    sublattice=sub1,
    element_list=element_list_sub1,
    standard_element=standard_element_sub1,
    scaling_exponent=scaling_exponent)

elements_of_sub1 = sort_sublattice_intensities(
    sub1, intensity_type, element_list_sub1, method,
    middle_intensity_list_sub1, limit_intensity_list_sub1,
    remove_background_method=remove_background_method,
    background_sublattice=background_sublattice,
    num_points=num_points,
    percent_to_nn=percent_to_nn,
    mask_radius=mask_radius_sub1)

assign_z_height(sub1, lattice_type='transition_metal',
                material='mos2_one_layer')

sub1_info = print_sublattice_elements(sub1)


# SUB2
# sub2_ints = get_sublattice_intensity(
#     sub2, intensity_type,
#     remove_background_method=remove_background_method,
#     background_sub=background_sublattice,
#     num_points=num_points,
#     percent_to_nn=percent_to_nn,
#     mask_radius=mask_radius_sub2)

# scipy.stats.mode(np.round(sub2_ints, decimals=2))[0][0]
# sub2_ints.mean()
# plt.figure()
# plt.hist(sub2_ints, bins=bins_for_hist)
# plt.show()


middle_intensity_list_sub2, limit_intensity_list_sub2 = find_middle_and_edge_intensities(
    sublattice=sub2,
    element_list=element_list_sub2,
    standard_element=standard_element_sub2,
    scaling_exponent=scaling_exponent)

elements_of_sub2 = sort_sublattice_intensities(
    sub2, intensity_type, element_list_sub2, method,
    middle_intensity_list_sub2, limit_intensity_list_sub2,
    remove_background_method=remove_background_method,
    background_sublattice=background_sublattice,
    num_points=num_points, percent_to_nn=percent_to_nn,
    mask_radius=mask_radius_sub2)

assign_z_height(sub2, lattice_type='chalcogen', material='mos2_one_layer')

sub2_info = print_sublattice_elements(sub2)


# SUB3

# sub3_ints = get_sublattice_intensity(sub3,
#                                     intensity_type=intensity_type,
#                                     remove_background_method=remove_background_method,
#                                     background_sublattice=background_sublattice,
#                                     num_points=num_points,
#                                     percent_to_nn=percent_to_nn,
#                                     mask_radius=mask_radius_sub2)
#
#scipy.stats.mode(np.round(sub3_ints, decimals=2))[0][0]
# sub3_ints.mean()
# plt.figure()
#plt.hist(sub3_ints, bins=15)
# plt.show()


middle_intensity_list_sub3, limit_intensity_list_sub3 = find_middle_and_edge_intensities_for_background(
    elements_from_sub1=elements_from_sub1,
    elements_from_sub2=elements_from_sub2,
    sub1_mode=sub1_mode,
    sub2_mode=sub2_mode,
    element_list_sub1=element_list_sub1,
    element_list_sub2=element_list_sub2,
    middle_intensity_list_sub1=middle_intensity_list_sub1,
    middle_intensity_list_sub2=middle_intensity_list_sub2)


elements_of_sub3 = sort_sublattice_intensities(
    sub3, intensity_type, element_list_sub3, method,
    middle_intensity_list_sub3,
    limit_intensity_list_sub3,
    remove_background_method=remove_background_method,
    background_sublattice=background_sublattice,
    num_points=num_points,
    intensity_list_real=True,
    percent_to_nn=percent_to_nn,
    mask_radius=mask_radius_sub2)

correct_background_elements(sub3)

assign_z_height(sub3, lattice_type='background', material='mos2_one_layer')

sub3_info = print_sublattice_elements(sub3)


# create .CIF file
element_list = combine_element_lists([element_list_sub1,
                                      element_list_sub2,
                                      element_list_sub3])


example_df_cif = create_dataframe_for_cif(
    sublattice_list=[sub1, sub2, sub3], element_list=element_list)

write_cif_from_dataframe(dataframe=example_df_cif,
                         filename=image_name,
                         chemical_name_common='MoSx-1Sex',
                         cell_length_a=image_size_x_nm * 10,
                         cell_length_b=image_size_y_nm * 10,
                         cell_length_c=image_size_z_nm * 10)

dataframe = create_dataframe_for_xyz(sublattice_list=[sub1, sub2, sub3],
                                     element_list=element_list,
                                     x_distance=image_size_x_nm * 10,
                                     y_distance=image_size_y_nm * 10,
                                     z_distance=image_size_z_nm * 10,
                                     filename=image_name,
                                     header_comment='Selenium implanted MoS2')


# Save Atom Lattice with intensity_type model 

sublattice_list = [sub1, sub2, sub3]
atom_lattice_name = 'Atom_Lattice_' + intensity_type

atom_lattice = am.Atom_Lattice(image=atom_lattice.image,
                               name=atom_lattice_name,
                               sublattice_list=sublattice_list)
atom_lattice.save(filename=atom_lattice_name + ".hdf5", overwrite=True)

atom_lattice.get_sublattice_atom_list_on_image(markersize=2).plot()
plt.title(atom_lattice_name, fontsize=20)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig(fname=atom_lattice_name + '.png',
            transparent=True, frameon=False, bbox_inches='tight',
            pad_inches=None, dpi=300, labels=False)
plt.close()


from temul.model_refiner import Model_Refiner
import atomap.api as am
import temul.example_data as example_data

s_original = example_data.load_Se_implanted_MoS2_data()
real_sampling = s_original.axes_manager[-1].scale

atom_lattice = am.load_atom_lattice_from_hdf5('Atom_Lattice_max.hdf5')
sub1 = atom_lattice.sublattice_list[0]
sub2 = atom_lattice.sublattice_list[1]
sub3 = atom_lattice.sublattice_list[2]

# Refine the Sublattice elements 
element_list_sub1 = ['Mo_0', 'Mo_1', 'Mo_1.S_1', 'Mo_1.Se_1', 'Mo_2']
element_list_sub2 = ['S_0', 'S_1', 'S_2', 'Se_1', 'Se_1.S_1', 'Se_2']
element_list_sub3 = ['H_0', 'S_1', 'Se_1', 'Mo_1', ]

sub_dict = {sub1: element_list_sub1,
            sub2: element_list_sub2,
            sub3: element_list_sub3}

image_size_z_nm = 1.2294 / 2

refiner = Model_Refiner(sub_dict,
                        sampling=real_sampling * 10,
                        thickness=image_size_z_nm * 10,
                        name='Se Implanted MoS2')
refiner.get_element_count_as_dataframe()
refiner.plot_element_count_as_bar_chart(2)

refiner.sublattice_list
for sub in refiner.sublattice_list:
    sub.plot()

refiner.image_difference_intensity_model_refiner()

refiner.sublattice_and_elements_dict
refiner.sublattice_list
refiner.element_list
refiner.flattened_element_list
refiner.sublattice_list[0].signal.axes_manager
refiner.sampling
refiner.name

refiner.auto_mask_radius
# refiner.auto_mask_radius[2] = 0.5

# refiner.thickness
# refiner.image_xyz_sizes

# refiner.set_thickness = 6.10
# refiner.image_xyz_sizes

# refiner.set_image_xyz_sizes = [5, 10, 6.2]
# refiner.image_xyz_sizes

# refiner.image_xyz_sizes[2] = 10
# refiner.image_xyz_sizes


# pick the top-left and bot-right of clean homogenous area
refiner.set_calibration_area()
refiner.calibration_area


# use atomap to get the pixel separation for the atoms you will use for
# calibration
# features = am.get_feature_separation(
#     signal=refiner.reference_image,
#     separation_range=(8, 15), pca=True)
# features.plot()
refiner.set_calibration_separation(11)
refiner.calibration_separation

refiner.comparison_image
refiner.comparison_image.axes_manager
refiner.reference_image.axes_manager

positions = refiner._sublattices_positions
# len(refiner.sublattice_list[0].atom_list) + \
# len(refiner.sublattice_list[1].atom_list) + \
# len(refiner.sublattice_list[2].atom_list)

refiner.set_calibration_area(
    manual_list=[[159.05087400067845, 409.82096276271284],
                 [331.0900946589779, 546.9873684227083]])

refiner.create_simulation(sublattices='all',
                          filter_image=True,
                          calibrate_image=True,
                          filename='sim',
                          interpolationFactor=200)

refiner.image_difference_intensity_model_refiner()

refiner.plot_reference_and_comparison_images()

refiner.comparison_image.plot()
refiner.reference_image.plot()


refiner.image_difference_intensity_model_refiner()

refiner.previous_refiner_instance
refiner = refiner.previous_refiner_instance

refiner.repeating_intensity_refinement(n=7)

refiner.image_difference_intensity_model_refiner()
refiner.get_element_count_as_dataframe()

refiner.image_difference_position_model_refiner(
    sublattices='all', pixel_threshold=14,
    filename='example', num_peaks=10)

refiner.image_difference_intensity_model_refiner()

refiner.get_element_count_as_dataframe()

refiner.sublattice_list[0].plot()

refiner.plot_element_count_as_bar_chart(2, flip_colrows=False)


refiner.error_between_comparison_and_reference_image
refiner.error_between_images_history
refiner.plot_error_between_comparison_and_reference_image()
refiner.plot_error_between_comparison_and_reference_image(style='scatter')

####################

import temul.polarisation as tmlpol
'''


# Image Registration

# def rigid_registration(file, masktype='hann', n=4, findMaxima='gf'):
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
'''

    # Read tiff file. Rearrange axes so final axis iterates over images
    stack = np.rollaxis(imread(file), 0, 3)
    # Normalize data between 0 and 1
    stack = stack[:, :, :] / float(2**16)

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

    s.make_corrected_Rij()
    # Correct outliers using the transitivity relations
    # s.show_Rij_c()
    # Display the corrected shift matrix
    # Create registered image stack and average
    # To skip calculation of image shifts, or correcting the shift matrix, pass
    # the function
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

    # Save the average image, including outer areas. Be careful when analysis
    # outer regions of this file
    s.save("average_image_no_crop.tif", crop=False)

    # creates a folder and put all the individual images in there
    save_individual_images_from_image_stack(image_stack=s.stack_registered)
'''
