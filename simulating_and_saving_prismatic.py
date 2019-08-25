
'''
Steps 

1. set the directory to where your file be do
2. convert from a vesta xyz file to a prismatic xyz file 
3. simulate the prismatic xyz file. This outputs a mrc file
4. save the simulated file as a png and hyperspy file

You can add lots to this. For example if you're simulating an experimental
image, see the function simulate_and_calibrate_with_prismatic() for allowing 
the experimental (reference) image to calculate the probeStep (sampling).
Only works if the reference image is loaded into python as a hyperspy 2D signal
and if the image is calibrated.

'''

import os
import hyperspy.api as hs
import my_code_functions_all as temul

# Step 1
# input the directory of your downloaded files here, pointing to the example_data!
os.chdir('C:/Users/Eoghan.OConnell/Documents/Documents/Eoghan UL/PHD/Python Files/scripts/Functions/private_development_git/private_development/example_data/prismatic')

# Step 2
prismatic_xyz = temul.convert_vesta_xyz_to_prismatic_xyz(
    vesta_xyz_file='MoS2_hex_vesta_xyz.xyz',
    prismatic_xyz_filename='MoS2_hex_prismatic.xyz',
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
    xyz_filename='MoS2_hex_prismatic.xyz',
    filename='prismatic_simulation',
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
    prismatic_mrc_filename='prism_2Doutput_prismatic_simulation.mrc',
    save_name='calibrated_data_2_electric_boogaloo')

simulation.plot()
