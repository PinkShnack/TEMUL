
import os
import numpy as np
import atomap.api as am
import hyperspy.api as hs
import matplotlib.pyplot as plt
import temul.api as tml

''' This file will find the sublattices in your image. Check out Atomap.org
    for lots of information on how to use atomap.

Note: this file is not guaranteed to find what you want. It varies from image
to image. Check out the other "2_Find_Sublattices..." files for different
types of materials.
'''




'''
Pixel separation
'''
# have a look at the different pixel separations available (you can change them! Go mad)
am.get_feature_separation(
    s_ifft, separation_range=(9, 20), pca=True).plot()

# choose the best pixel separation
# 12 for 2004, 9 for 2048
atom_positions1 = am.get_atom_positions(s_ifft, separation=9, pca=True)
atom_positions1 = am.add_atoms_with_gui(s, atom_list=atom_positions1)

# save these original positions!
np.save('atom_positions1', arr=atom_positions1)
# positions = np.load('atom_positions1_overfull.npy') # how to reload this file format


''' Create the first sublattice - sub1 '''

sub1 = am.Sublattice(
    atom_position_list=atom_positions1, image=s, color='red')
sub1.find_nearest_neighbors()
#sub1.plot()

sub1.refine_atom_positions_using_2d_gaussian(percent_to_nn=0.2)
# sub1.refine_atom_positions_using_center_of_mass(percent_to_nn=0.2)


# change the marker size (can be useful for big images)
sub1.get_atom_list_on_image(markersize=2).plot()

plt.title('sub1', fontsize=20)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig(fname='sub1.png',
            transparent=True, frameon=False, bbox_inches='tight',
            pad_inches=None, dpi=300, labels=False)
plt.close()


sub1.construct_zone_axes()
#sub1.plot_planes()  # this can take a long time for large images!


# # side note: You can remove these atoms from the image if you like!
# im_atoms_subtracted = amtools.remove_atoms_from_image_using_2d_gaussian(
#     s.data, sub1)

# # convert this numpy array to a hyperspy image object
# im_atoms_subtracted = hs.signals.Signal2D(im_atoms_subtracted)
# im_atoms_subtracted.plot()
# im_atoms_subtracted.save('im_atoms_subtracted_im.hspy')


''' Part 2 - finding the other sublattices '''

# choose the zone_axes that will give you the correct atoms positions along the lines
zone_axis_001 = sub1.zones_axis_average_distances[0]

# use this function to choose a position between atoms defined by the vector_fraction
atom_positions2_part1 = sub1.find_missing_atoms_from_zone_vector(
    zone_axis_001, vector_fraction=0.5)

# for this structrue, you'll need to do it again in the perperdicular direction!
zone_axis_002 = sub1.zones_axis_average_distances[1]
atom_positions2_part2 = sub1.find_missing_atoms_from_zone_vector(
    zone_axis_002, vector_fraction=0.5)

atom_positions2 = atom_positions2_part1 + atom_positions2_part2

# atom_positions2 = am.add_atoms_with_gui(s, atom_list=atom_positions2)

np.save('atom_positions2_ideal', arr=atom_positions2)

''' Create the second sublattice - sub2 '''
'''
image_atoms_removed = at.remove_atoms_from_image_using_2d_gaussian(
    image=s.data,
    sublattice=sub1,
    percent_to_nn=0.3)

hs.signals.Signal2D(image_atoms_removed).plot()

sub2 = am.Sublattice(atom_positions2, image=s, color='blue')
sub2.find_nearest_neighbors()
# sub2.plot()

# t0 = time()
sub2.refine_atom_positions_using_2d_gaussian(image_data=image_atoms_removed,
                                             percent_to_nn=0.3)
# t1 = time()-t0
# print("This refinement took {:.2f} seconds".format(t1))

sub2.refine_atom_positions_using_center_of_mass(image_data=image_atoms_removed,
                                                percent_to_nn=0.4)
'''

sub2 = am.Sublattice(atom_positions2, image=s, color='blue')
sub2.find_nearest_neighbors()
#sub2.plot()

sub2.refine_atom_positions_using_2d_gaussian(percent_to_nn=0.2)
# sub2.refine_atom_positions_using_center_of_mass(percent_to_nn=0.3)

np.save('atom_positions2_refined', [sub2.x_position, sub2.y_position])

sub2.get_atom_list_on_image(markersize=2).plot()

plt.title('sub2', fontsize=20)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig(fname='sub2.png',
            transparent=True, frameon=False, bbox_inches='tight',
            pad_inches=None, dpi=300, labels=False)
plt.close()


'''Save Atom Lattice Object - This contains our two sublattice'''
atom_lattice = am.Atom_Lattice(image=s.data,
                               name='Both Sublattices UL Boracite First Test!',
                               sublattice_list=[sub1, sub2])

atom_lattice.save(filename="Atom_Lattice.hdf5", overwrite=True)

atom_lattice.plot(markersize=4)
plt.title('Atom Lattice', fontsize=20)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig(fname='Atom Lattice.png',
            transparent=True, frameon=False, bbox_inches='tight',
            pad_inches=None, dpi=300, labels=False)

