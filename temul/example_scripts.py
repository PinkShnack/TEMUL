
import ase
from ase.cluster.cubic import FaceCenteredCubic
from ase.io import read, write
from ase.visualize import view
from temul.model_creation import get_max_number_atoms_z
import temul.api as tml
import os
import atomap.api as am
import hyperspy.api as hs
import numpy as np

%matplotlib qt


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
                             cell_length_a=s.axes_manager['x'].size/10,
                             cell_length_b=s.axes_manager['y'].size/10,
                             cell_length_c=z_thickness)

sublattice_structure = read(cif_filename + '.cif')
view(sublattice_structure)

# Simulate created NP
tml.create_dataframe_for_xyz(sublattice_list=[sublattice],
                             element_list=element_list,
                             x_distance=s.axes_manager['x'].size/10,
                             y_distance=s.axes_manager['y'].size/10,
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
# ...


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

tml.assign_z_height_to_sublattice(sublattice)

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
Steps

1. get the path to the example data
2. convert from a vesta xyz file to a prismatic xyz file
3. simulate the prismatic xyz file. This outputs a mrc file
4. save the simulated file as a png and hyperspy file

You can add lots to this. For example if you're simulating an experimental
image, see the function simulate_and_calibrate_with_prismatic() for allowing
the experimental (reference) image to calculate the probeStep (sampling).
Only works if the reference image is loaded into python as a hyperspy 2D signal
and if the image is calibrated.

'''

# Step 1

vesta_xyz_filename = tml.example_data.path_to_example_data_MoS2_vesta_xyz()
# print(vesta_xyz_filename)

# set the filenames for opening and closing...
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


# choose directory
# directory = 'G:/Titan Images/08-10-19_MHEOC_SampleImaging stem/Cross Grating for STEM alignment/Au NP'
# os.chdir(directory)


# '''
# Au NP example
# '''
# # open file
# s_raw, sampling = temul.load_data_and_sampling(
#     'STEM 20190813 HAADF 1732.emd', save_image=False)

# cropping_area = am.add_atoms_with_gui(s_raw.data)

# cropping_area = [[1, 2], [1, 2]]

# s_crop = temul.crop_image_hs(s_raw, cropping_area)

# roi = hs.roi.RectangularROI(left=1, right=5, top=1, bottom=5)
# s_raw.plot()
# s_crop = roi.interactive(s_raw)
# s_crop.plot()
