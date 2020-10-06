
from temul.polarisation import get_strain_gradient
import atomap.api as am
import hyperspy.api as hs

path_to_data = os.path.join(os.path.dirname(__file__), "data") 

# Open the PTO/SRO dataset
image = hs.load(os.path.join(path_to_data, 'PTO-SRO_Aligned-Series.hspy'))

sampling = image.axes_manager[-1].scale #  nm/pix
units = image.axes_manager[-1].units
image.plot()

# Open the pre-made PTO-SRO atom lattice. 
atom_lattice = am.load_atom_lattice_from_hdf5(os.path.join(path_to_data,
                                              "Atom_Lattice.hdf5"))
sublattice1 = atom_lattice.sublattice_list[0] #  Pb-Sr Sublattice
sublattice2 = atom_lattice.sublattice_list[1] #  Ti-Ru Sublattice

# Plot the sublattice planes to see which zone_vector_index we use
sublattice2.construct_zone_axes(atom_plane_tolerance=1)
sublattice1.construct_zone_axes(atom_plane_tolerance=1)
sublattice2.plot_planes()

# Set up parameters for get_strain_gradient
zone_vector_index = 1
atom_planes = (5, 10) #  chooses the starting and ending atom planes
vmin, vmax = 1, 2
cmap = 'bwr' #  see matplotlib and colorcet for more colormaps
title = 'Strain Gradient Map'
filename = None #  Set to a string if you want to save the map


# We want to see the strain gradient in the SRO Sublattice
str_grad_map = get_strain_gradient(sublattice2, zone_vector_index,
                                   sampling=sampling, units=units,
                                   cmap=cmap, title=title, atom_planes=atom_planes)
