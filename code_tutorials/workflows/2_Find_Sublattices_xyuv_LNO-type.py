
import numpy as np
import atomap.api as am
import temul.api as tml

''' This file will find the sublattices in your image. Check out Atomap.org
    for lots of information on how to use atomap.

Note: this file is not guaranteed to find what you want. It varies from image
to image. Check out the other "2_Find_Sublattices..." files for different
types of materials.
'''




''' Get the Pixel separation between the atoms in your first sublattice

Note: If you have used the image filter file, then replace "image" with
"image_ifft" or whatever you called your filtered file.
'''

# look at the different pixel separations available, this can take some time.
feat = am.get_feature_separation(image, separation_range=(5, 20), pca=True)
feat.plot()


''' If you're happy with one of the pixel separations, then set it below. If
    you're not, you may need to change your separation_range or filter your
    image with the "1b_Filter_Data.py" file.

Note: If you have used the image filter file, then replace "image" with
"image_ifft" or whatever you called your filtered file.    
'''

sep = 5  # just an example
atom_positions1 = am.get_atom_positions(image, separation=sep, pca=True)

# save these original sub1 positions!
np.save('atom_positions1.npy', arr=atom_positions1)

# how to reload this file format
# example_positions = np.load('atom_positions1.npy')


''' Create the first sublattice, which we call sub1. '''

sub1 = am.Sublattice(atom_position_list=atom_positions1,
                     image=image, color='red')
sub1.find_nearest_neighbors()

# you can plot the sublattice easily with:
# sub1.plot()

# You can refine the sublattice easily with a 2D Gaussian or COM algorithm:
sub1.refine_atom_positions_using_2d_gaussian(percent_to_nn=0.2)
# sub1.refine_atom_positions_using_center_of_mass(percent_to_nn=0.2)

np.save('atom_positions1_refined.npy', [sub1.x_position, sub1.y_position])

# Use atom_plane_tolerance=1 for these deformed structures
sub1.construct_zone_axes(atom_plane_tolerance=1)
# sub1.plot_planes()  # this can take a long time for large images!


''' Create and save the Atom Lattice Object - This contains our sublattice.
'''

atom_lattice = am.Atom_Lattice(image=image.data,
                               name='LNO-type structure',
                               sublattice_list=[sub1])

atom_lattice.save(filename="Atom_Lattice.hdf5", overwrite=True)

atom_lattice.plot()


''' Now we need to get the (x, y) and (u, v) data for the polarisation vectors.
    For the LNO-type material, we need to use just the first sublattice to
    find the polarisation vectors.
    
    We can use the temul toolkit to do this easily with the
    atom_deviation_from_straight_line_fit function. 
'''

n = 5  # example
zone_axis = 0  # example
x, y, u, v = tml.atom_deviation_from_straight_line_fit(
    sub1, zone_axis, n, plot=True)
