
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

sep = 9  # just an example
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


# this is needed to create the second sublattice:
sub1.construct_zone_axes()
# sub1.plot_planes()  # this can take a long time for large images!

''' Create the second sublattice - sub2.

We have to choose the zone_axes that will give you the correct atoms positions
along the atom plane lines. To visualise the atom planes, use
sub1.plot_planes() which can take a long time for certain images.
'''

zone_axis_A = sub1.zones_axis_average_distances[2]

# use this function to choose a position between atoms defined by the
# vector_fraction
atom_positions2 = sub1.find_missing_atoms_from_zone_vector(
    zone_axis_A, vector_fraction=0.5)

# save these positions 
np.save('atom_positions2_ideal.npy', arr=atom_positions2)


sub2 = am.Sublattice(atom_position_list=atom_positions2,
                     image=image, color='blue')
sub2.find_nearest_neighbors()
# sub2.plot()

sub2.refine_atom_positions_using_2d_gaussian(percent_to_nn=0.4)
# sub2.refine_atom_positions_using_center_of_mass(percent_to_nn=0.2)

np.save('atom_positions2_refined.npy', [sub2.x_position, sub2.y_position])



''' Create and save the Atom Lattice Object - This contains our two
    sublattices.
'''

atom_lattice = am.Atom_Lattice(image=image.data,
                               name='PTO-type structure',
                               sublattice_list=[sub1, sub2])

atom_lattice.save(filename="Atom_Lattice.hdf5", overwrite=True)

atom_lattice.plot()


''' Now we need to get the (x, y) and (u, v) data for the polarisation vectors.
    This requires the relevant sublattice's original "ideal" positions and
    refined "actual" positions.
    
    We can use Atomap or the Temul toolkit to do this easily with the save information
    from above.
'''


# Atomap method (if this doesn't work, try the Temul method below)

za0, za1 = sub1.zones_axis_average_distances[0:2]
s_p = sub1.get_polarization_from_second_sublattice(
    za0, za1, sub2, color='cyan')
vector_list = s_p.metadata.vector_list
x, y = [i[0] for i in vector_list], [i[1] for i in vector_list]
u, v = [i[2] for i in vector_list], [i[3] for i in vector_list]
x, y, u, v = np.asarray(x), np.asarray(y), np.asarray(u), np.asarray(v)



# Temul method, only use this if the above Atomap method doesn't work
atom_positions_A = np.load('atom_positions2_ideal.npy')
atom_positions_B = np.load('atom_positions2_refined.npy').T
x, y = atom_positions_A[:, 0], atom_positions_A[:, 1]

u, v = tml.find_polarisation_vectors(atom_positions_A=atom_positions_A,
                                     atom_positions_B=atom_positions_B)
u, v = np.asarray(u), np.asarray(v)
