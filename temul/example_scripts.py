
import temul.example_data as example_data
import temul.my_code_functions_all
import os
import my_code_functions_all as temul
import atomap.api as am
import hyperspy.api as hs
# choose directory

directory = 'G:/Titan Images/08-10-19_MHEOC_SampleImaging stem/Cross Grating for STEM alignment/Au NP'
os.chdir(directory)


'''
Au NP example
'''
# open file
s_raw, sampling = temul.load_data_and_sampling(
    'STEM 20190813 HAADF 1732.emd', save_image=False)

cropping_area = am.add_atoms_with_gui(s_raw.data)

cropping_area = [[1, 2], [1, 2]]

s_crop = temul.crop_image_hs(s_raw, cropping_area)

roi = hs.roi.RectangularROI(left=1, right=5, top=1, bottom=5)
s_raw.plot()
s_crop = roi.interactive(s_raw)
s_crop.plot()


########################################################

'''
Simulate MoS2 with Prismatic Example
'''

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

vesta_xyz_filename = example_data.path_to_example_data_vesta_MoS2_vesta_xyz()
# print(vesta_xyz_filename)

# set the filenames for opening and closing...
prismatic_xyz_filename = 'MoS2_hex_prismatic.xyz'
mrc_filename = 'prismatic_simulation'
simulated_filename = 'calibrated_data_2_electric_boogaloo'

# Step 2
prismatic_xyz = temul.convert_vesta_xyz_to_prismatic_xyz(
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
temul.simulate_with_prismatic(
    xyz_filename=prismatic_xyz_filename,
    filename=mrc_filename,
    probeStep=0.01,
    reference_image=None,
    E0=60e3,
    integrationAngleMin=0.085,
    integrationAngleMax=0.186,
    interpolationFactor=4,
    realspacePixelSize=0.0654,
    numFP=1,
    probeSemiangle=0.030,
    alphaBeamMax=0.032,
    scanWindowMin=0.0,
    scanWindowMax=0.5,
    algorithm="prism",
    numThreads=2)

# Step 4
simulation = temul.load_prismatic_mrc_with_hyperspy(
    prismatic_mrc_filename='prism_2Doutput_' + mrc_filename + '.mrc',
    save_name=simulated_filename)

simulation.plot()
