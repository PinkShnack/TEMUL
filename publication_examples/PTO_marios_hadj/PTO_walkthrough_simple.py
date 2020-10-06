
from temul.polarisation import get_strain_gradient
import atomap.api as am
import hyperspy.api as hs

# Open the original PTO/SRO dataset
image = hs.load("publication_examples/data"
                "/STEM 20191121 HAADF 3.3 Mx 0300.emd")

# Open the pre-made PTO atom lattice. 
# See the extended walkthrough for how to make this atom lattice.
atom_lattice = am.load_atom_lattice_from_hdf5("data/Atom_Lattice.hdf5")
sublattice1 = atom_lattice.sublattice_list[0] #  PTO Sublattice
sublattice2 = atom_lattice.sublattice_list[1] #  SRO Sublattice

# Plot the sublattice planes to see which zone_vector_index we use
sublattice2.construct_zone_axes()
sublattice2.plot_planes()

# Set up parameters for get_strain_gradient
zone_vector_index = 0
atom_planes = (14, 19) #  chooses the starting and ending atom planes
sampling = image.axes_manager[-1].scale
units = image.axes_manager[-1].units
vmin, vmax = 1, 4
cmap = 'inferno' #  see matplotlib and colorcet for more colormaps
title = 'Strain Gradient Map'
filename = None #  Set to a string if you want to save the map


# We want to see the strain gradient in the SRO Sublattice
str_grad_map = get_strain_gradient(sublattice2, zone_vector_index, atom_planes,
                                   sampling, units, cmap, title)
