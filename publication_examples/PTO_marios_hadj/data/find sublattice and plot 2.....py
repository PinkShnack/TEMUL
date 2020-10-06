# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:22:22 2020

@author: Michele.Conroy
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:26:20 2018

@author: eoghan.oconnell
"""


import os
import numpy as np
import atomap.api as am
from atomap.tools import remove_atoms_from_image_using_2d_gaussian
import hyperspy.api as hs
import matplotlib.pyplot as plt

#cwd = os.getcwd()
#cwd
#os.chdir(cwd + '/PTO_scripts/vortex/0258/Image 1,15')

os.chdir('C:/Users/Eoghan.OConnell/Documents/Documents/Eoghan UL/PHD'
         '/Python Files/development/private_development/PTO_scripts/vortex/0228'
         '/filtered_image')

s_orig = hs.load('../STEM 20191121 HAADF 3.3 Mx 0228.emd')
s = hs.load("filter 3 of 0228 3.3 Mx.tif")

s.axes_manager = s_orig.axes_manager
#s = hs.load('original_image_1,15.hspy')

# s = s[1]
# s = s.inav[15]

# s.save("original_image_1,15.hspy", overwrite=True)

s.plot()
image_name = "Original Image"
plt.title(image_name, fontsize=20)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig(fname=image_name + '.png',
            transparent=True, frameon=False, bbox_inches='tight',
            pad_inches=None, dpi=300, labels=False)
plt.close()


''' crop and save the image '''
roi = hs.roi.RectangularROI(left=1, right=3, top=1, bottom=3)
s.plot()
s_crop = roi.interactive(s)

s_crop.save("cropped_region.hspy")
s_crop.axes_manager

s=s_crop

plt.title('Cropped region highlighted', fontsize=20)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig(fname='Cropped region highlighted.png',
            transparent=True, frameon=False, bbox_inches='tight',
            pad_inches=None, dpi=300, labels=False)
plt.close()

plt.title('Cropped region', fontsize=20)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig(fname='Cropped region.png',
            transparent=True, frameon=False, bbox_inches='tight',
            pad_inches=None, dpi=300, labels=False)
plt.close()


# crop offset
axes = s_crop.axes_manager
sampling = axes[-1].scale

x_offset_nm, y_offset_nm = axes[-2].offset, axes[-1].offset
x_offset_pix, y_offset_pix = x_offset_nm/sampling, y_offset_nm/sampling

offsets = np.array([[x_offset_nm, y_offset_nm], [x_offset_pix, y_offset_pix]])
np.save('xy_offset_nm and pix.npy', offsets)



#am.get_feature_separation(s, separation_range=(8,11), pca=True).plot()
atom_positions1 = am.get_atom_positions(s, separation=11, pca=True)
atom_positions1 = am.add_atoms_with_gui(s, atom_list=atom_positions1)
np.save('atom_positions1', arr=atom_positions1)

''' Sublattice1 '''
sublattice1 = am.Sublattice(
    atom_position_list=atom_positions1, image=s, color='blue')
sublattice1.find_nearest_neighbors()

sublattice1.refine_atom_positions_using_2d_gaussian()

np.save('atom_positions1_refined',
        [sublattice1.x_position, sublattice1.y_position])

# sublattice1.plot()
# Plot using markersize to make the dots small enough
sublattice1.get_atom_list_on_image(markersize=2).plot()

plt.title('Sublattice1', fontsize=20)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig(fname='sublattice1.png',
            transparent=True, frameon=False, bbox_inches='tight',
            pad_inches=None, dpi=300, labels=False)
plt.close()



sublattice1.construct_zone_axes(atom_plane_tolerance=1)
#sublattice1.plot_planes()


''' Sublattice2 '''
#sublattice1.construct_zone_axes()
# sublattice1.plot_planes()
zone_axis_001 = sublattice1.zones_axis_average_distances[3]
#zone_axis_002 = sublattice1.zones_axis_average_distances[1]

atom_positions2 = sublattice1.find_missing_atoms_from_zone_vector(
    zone_axis_001, vector_fraction=0.5)
#atom_positions2_part2 = sublattice1.find_missing_atoms_from_zone_vector(zone_axis_002, vector_fraction=0.5)

#atom_positions2 = atom_positions2_part1 + atom_positions2_part2


#atom_positions2 = am.add_atoms_with_gui(s, atom_list=atom_positions2)
np.save('atom_positions2', arr=atom_positions2)

#image_atoms_subtracted = remove_atoms_from_image_using_2d_gaussian(
#    sublattice1.image, sublattice1, percent_to_nn=0.4)

sublattice2 = am.Sublattice(atom_position_list=atom_positions2,
                            image=s, color='red')

sublattice2.find_nearest_neighbors()

#sublattice2.plot()

sublattice2.refine_atom_positions_using_2d_gaussian(percent_to_nn=0.25)
sublattice2.refine_atom_positions_using_2d_gaussian(percent_to_nn=0.25)

np.save('atom_positions2_refined',
        [sublattice2.x_position, sublattice2.y_position])

sublattice2.get_atom_list_on_image(markersize=2).plot()

plt.title('Sublattice2', fontsize=20)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig(fname='sublattice2.png',
            transparent=True, frameon=False, bbox_inches='tight',
            pad_inches=None, dpi=300, labels=False)
plt.close()


'''Save Atom Lattice Object'''
atom_lattice = am.Atom_Lattice(image=s.data,
                               name='Both Sublattices',
                               sublattice_list=[sublattice1, sublattice2])
atom_lattice.save(filename="Atom_Lattice.hdf5", overwrite=True)

atom_lattice.plot(markersize=2)
plt.title('Atom Lattice', fontsize=20)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig(fname='Atom Lattice.png',
            transparent=True, frameon=False, bbox_inches='tight',
            pad_inches=None, dpi=300, labels=False)
plt.close()


## Loading Atom Lattice object
#atom_lattice = am.load_atom_lattice_from_hdf5('Atom_Lattice.hdf5')
#atom_lattice.sublattice_list
#atom_lattice.plot()

#plotting Ti

atom_lattice = am.load_atom_lattice_from_hdf5('Atom_Lattice.hdf5')
atom_lattice.sublattice_list
# atom_lattice.plot()

sublattice1 = atom_lattice.sublattice_list[0]
sublattice2 = atom_lattice.sublattice_list[1]

sublattice1.construct_zone_axes()
sublattice2.construct_zone_axes()

za0 = sublattice1.zones_axis_average_distances[0]
za1 = sublattice1.zones_axis_average_distances[1]
s_polarization = sublattice1.get_polarization_from_second_sublattice(
    za0, za1, sublattice2)

vector_list = s_polarization.metadata.vector_list
len(vector_list)

x = [i[0] for i in vector_list]
y = [i[1] for i in vector_list]
u = [i[2] for i in vector_list]
v = [i[3] for i in vector_list]

plt.figure()
plt.hist(u, bins=50)
plt.hist(v, bins=50)

# clean up data to remove large vectors
indexes_to_remove = []
for i, _ in enumerate(vector_list):
    if vector_list[i][2] > 10 or vector_list[i][2] < -10:
        indexes_to_remove.append(i)
    if vector_list[i][3] > 10 or vector_list[i][3] < -10:
        indexes_to_remove.append(i)

for index_to_remove in sorted(indexes_to_remove, reverse=True):
    del vector_list[index_to_remove]

len(vector_list)

x = [i[0] for i in vector_list]
y = [i[1] for i in vector_list]
u = [i[2] for i in vector_list]
v = [i[3] for i in vector_list]

plt.figure()
plt.hist(u, bins=50)
plt.hist(v, bins=50)



tml_pol.plot_polarisation_vectors(
        x=x, y=y, u=u, v=v, image=atom_lattice.image,
        sampling=sampling, units=units,
        normalise=True, overlay=True,
        save=None, color='yellow',
        plot_style='vector', title='Ti Polarisation',
        pivot='middle', angles='xy', scale_units='xy', scale=None,
        headwidth=3.5, headlength=5.0, headaxislength=4.5, monitor_dpi=96)







u = [-1*i for i in u]
v = [-1*i for i in v]


plot_style_list = ['vector', 'vector', 'vector', 'colormap',
                   'colormap', 'contour']
normal_list = [False, True, False, False, False, True]
overlay_list = [True, True, False, True, False, True]
save_list = ['', '_normal', '_no_overlay', '', '_no_overlay', '_normal']
color_list = ['yellow', 'yellow', 'red', 'red', 'red', 'gray']
cmap_list = [None, None, None, None, 'inferno', 'viridis']

for plot_style, normal, overlay, save, color, cmap in zip(
        plot_style_list, normal_list, overlay_list, save_list,
        color_list, cmap_list):

    tml_pol.plot_polarisation_vectors(
        x=x, y=y, u=u, v=v, image=atom_lattice.image,
        sampling=sampling, units=units,
        normalise=normal, overlay=overlay, cmap=cmap,
        save="Ti Polarisation" + save, color=color,
        plot_style=plot_style, title='Ti Polarisation',
        pivot='middle', angles='xy', scale_units='xy', scale=None,
        headwidth=3.5, headlength=5.0, headaxislength=4.5, monitor_dpi=96)




import numpy as np
from temul.polarisation import (
    find_polarisation_vectors, get_average_polarisation_in_regions,
    get_average_polarisation_in_regions_square)
#
#atom_positions_A = np.array(sublattice2.atom_positions).T
#x = atom_positions_A[:, 0]
#y = atom_positions_A[:, 1]
#np.save('refined_atom_positions2', atom_positions_A)
#
##atom_positions_B = np.load('atom_positions2.npy')
#atom_positions_B = atom_positions2
#
#u, v = find_polarisation_vectors(atom_positions_B, atom_positions_A,
#                                 save='uv_vectors_array')


divide_into_list = [4,8,10,12,16,20,24,28,32,36,40]

for divide_into in divide_into_list:

    x_new, y_new, u_new, v_new = get_average_polarisation_in_regions_square(
        x, y, u, v, image=atom_lattice.image, divide_into=divide_into)
        
    tml_pol.plot_polarisation_vectors(
            x=x_new, y=y_new, u=u_new, v=v_new, image=atom_lattice.image,
            sampling=sampling, units=units,
            normalise=True, overlay=True,
            save='Averaged Ti Polarisation {}'.format(divide_into), color='yellow',
            plot_style='vector', title='Averaged Ti Polarisation',
            pivot='middle', angles='xy', scale_units='xy', scale=None,
            headwidth=3.5, headlength=5.0, headaxislength=4.5, monitor_dpi=96)



#x_new, y_new, u_new, v_new = get_average_polarisation_in_regions_square(
#        x, y, u, v, image=atom_lattice.image, divide_into=8)
#
#tml_pol.plot_polarisation_vectors(
#        x=x_new, y=y_new, u=u_new, v=v_new, image=atom_lattice.image,
#        sampling=sampling, units=units,
#        normalise=True, overlay=True,
#        save=None, color='yellow',
#        plot_style='colormap', title='Ti Polarisation',
#        pivot='middle', angles='xy', scale_units='xy', scale=None,
#        headwidth=3.5, headlength=5.0, headaxislength=4.5, monitor_dpi=96)






# offset plotting for main image
s_orig = hs.load('STEM 20191121 HAADF 3.3 Mx 0228.emd')
sampling = s_orig.axes_manager[-1].scale
units = s_orig.axes_manager[-1].units
image = s_orig.data
monitor_dpi=200

offsets = np.load("xy_offset_nm and pix.npy")
x_offset_pix= offsets[1][0]
y_offset_pix= offsets[1][1]
x_os = [i+x_offset_pix for i in x]
y_os = [i+y_offset_pix for i in y]


tml_pol.plot_polarisation_vectors(
        x=x_os, y=y_os, u=u, v=v, image=image,
        sampling=sampling, units=units,
        normalise=True, overlay=True,
        save=None, color='yellow',
        plot_style='vector', title='Ti Polarisation',
        pivot='middle', angles='xy', scale_units='xy', scale=None,
        headwidth=3.5, headlength=5.0, headaxislength=4.5, monitor_dpi=monitor_dpi)



plot_style_list = ['vector', 'vector', 'vector', 'colormap',
                   'colormap', 'contour']
normal_list = [False, True, False, False, False, True]
overlay_list = [True, True, False, True, False, True]
save_list = ['', '_normal', '_no_overlay', '', '_no_overlay', '_normal']
color_list = ['yellow', 'yellow', 'red', 'red', 'red', 'gray']
cmap_list = [None, None, None, None, 'inferno', 'viridis']

for plot_style, normal, overlay, save, color, cmap in zip(
        plot_style_list, normal_list, overlay_list, save_list,
        color_list, cmap_list):

    tml_pol.plot_polarisation_vectors(
        x=x_os, y=y_os, u=u, v=v, image=image,
        sampling=sampling, units=units,
        normalise=normal, overlay=overlay, cmap=cmap,
        save="Ti Polarisation Full Image" + save, color=color,
        plot_style=plot_style, title='Ti Polarisation',
        pivot='middle', angles='xy', scale_units='xy', scale=None,
        headwidth=3.5, headlength=5.0, headaxislength=4.5, monitor_dpi=monitor_dpi)



divide_into_list = [4,8,10,12,16,20,24,28,32,36,40]

x = [i[0] for i in vector_list]
y = [i[1] for i in vector_list]
u = [i[2] for i in vector_list]
v = [i[3] for i in vector_list]
u = [-1*i for i in u]
v = [-1*i for i in v]

for divide_into in divide_into_list:
        
    x_new, y_new, u_new, v_new = get_average_polarisation_in_regions_square(
        x, y, u, v, image=atom_lattice.image, divide_into=divide_into)
    
    x_os = [i+x_offset_pix for i in x_new]
    y_os = [i+y_offset_pix for i in y_new]
    
    tml_pol.plot_polarisation_vectors(
            x=x_os, y=y_os, u=u_new, v=v_new, image=image,
            sampling=sampling, units=units,
            normalise=True, overlay=True,
            save='Averaged Ti Polarisation Full Image {}'.format(divide_into), color='yellow',
            plot_style='vector', title='Averaged Ti Polarisation',
            pivot='middle', angles='xy', scale_units='xy', scale=None,
            headwidth=3.5, headlength=5.0, headaxislength=4.5, monitor_dpi=monitor_dpi)











''' polarisation '''

'''
s_orig = hs.load('original_image_1,15.hspy')
sampling = s_orig.axes_manager[-1].scale
units = s_orig.axes_manager[-1].units

os.chdir('./Bottom section')

atom_lattice = am.load_atom_lattice_from_hdf5('Atom_Lattice.hdf5')
atom_lattice.sublattice_list
# atom_lattice.plot()

sublattice1 = atom_lattice.sublattice_list[0]
sublattice2 = atom_lattice.sublattice_list[1]
'''
sublattice1.construct_zone_axes()
sublattice2.construct_zone_axes()

za0 = sublattice1.zones_axis_average_distances[0]
za1 = sublattice1.zones_axis_average_distances[1]
s_polarization = sublattice1.get_polarization_from_second_sublattice(
    za0, za1, sublattice2)

vector_list = s_polarization.metadata.vector_list
len(vector_list)

x = [i[0] for i in vector_list]
y = [i[1] for i in vector_list]
u = [i[2] for i in vector_list]
v = [i[3] for i in vector_list]

plt.figure()
plt.hist(u, bins=50)
plt.hist(v, bins=50)

# clean up data to remove large vectors
indexes_to_remove = []
for i, _ in enumerate(vector_list):
    if vector_list[i][2] > 10 or vector_list[i][2] < -10:
        indexes_to_remove.append(i)
    if vector_list[i][3] > 10 or vector_list[i][3] < -10:
        indexes_to_remove.append(i)

for index_to_remove in sorted(indexes_to_remove, reverse=True):
    del vector_list[index_to_remove]

len(vector_list)

x = [i[0] for i in vector_list]
y = [i[1] for i in vector_list]
u = [i[2] for i in vector_list]
v = [i[3] for i in vector_list]

plt.figure()
plt.hist(u, bins=50)
plt.hist(v, bins=50)



tml_pol.plot_polarisation_vectors(
        x=x, y=y, u=u, v=v, image=atom_lattice.image,
        sampling=sampling, units=units,
        normalise=True, overlay=True,
        save=None, color='yellow',
        plot_style='vector', title='Ti Polarisation',
        pivot='middle', angles='xy', scale_units='xy', scale=None,
        headwidth=3.5, headlength=5.0, headaxislength=4.5, monitor_dpi=96)







u = [-1*i for i in u]
v = [-1*i for i in v]


plot_style_list = ['vector', 'vector', 'vector', 'colormap',
                   'colormap', 'contour']
normal_list = [False, True, False, False, False, True]
overlay_list = [True, True, False, True, False, True]
save_list = ['', '_normal', '_no_overlay', '', '_no_overlay', '_normal']
color_list = ['yellow', 'yellow', 'red', 'red', 'red', 'gray']
cmap_list = [None, None, None, None, 'inferno', 'viridis']

for plot_style, normal, overlay, save, color, cmap in zip(
        plot_style_list, normal_list, overlay_list, save_list,
        color_list, cmap_list):

    tml_pol.plot_polarisation_vectors(
        x=x, y=y, u=u, v=v, image=atom_lattice.image,
        sampling=sampling, units=units,
        normalise=normal, overlay=overlay, cmap=cmap,
        save="Ti Polarisation" + save, color=color,
        plot_style=plot_style, title='Ti Polarisation',
        pivot='middle', angles='xy', scale_units='xy', scale=None,
        headwidth=3.5, headlength=5.0, headaxislength=4.5, monitor_dpi=96)




import numpy as np
from temul.polarisation import (
    find_polarisation_vectors, get_average_polarisation_in_regions,
    get_average_polarisation_in_regions_square)
#
#atom_positions_A = np.array(sublattice2.atom_positions).T
#x = atom_positions_A[:, 0]
#y = atom_positions_A[:, 1]
#np.save('refined_atom_positions2', atom_positions_A)
#
##atom_positions_B = np.load('atom_positions2.npy')
#atom_positions_B = atom_positions2
#
#u, v = find_polarisation_vectors(atom_positions_B, atom_positions_A,
#                                 save='uv_vectors_array')


divide_into_list = [4,8,10,12,16,20,24,28,32,36,40]

for divide_into in divide_into_list:

    x_new, y_new, u_new, v_new = get_average_polarisation_in_regions_square(
        x, y, u, v, image=atom_lattice.image, divide_into=divide_into)
        
    tml_pol.plot_polarisation_vectors(
            x=x_new, y=y_new, u=u_new, v=v_new, image=atom_lattice.image,
            sampling=sampling, units=units,
            normalise=True, overlay=True,
            save='Averaged Ti Polarisation {}'.format(divide_into), color='yellow',
            plot_style='vector', title='Averaged Ti Polarisation',
            pivot='middle', angles='xy', scale_units='xy', scale=None,
            headwidth=3.5, headlength=5.0, headaxislength=4.5, monitor_dpi=96)



#x_new, y_new, u_new, v_new = get_average_polarisation_in_regions_square(
#        x, y, u, v, image=atom_lattice.image, divide_into=8)
#
#tml_pol.plot_polarisation_vectors(
#        x=x_new, y=y_new, u=u_new, v=v_new, image=atom_lattice.image,
#        sampling=sampling, units=units,
#        normalise=True, overlay=True,
#        save=None, color='yellow',
#        plot_style='colormap', title='Ti Polarisation',
#        pivot='middle', angles='xy', scale_units='xy', scale=None,
#        headwidth=3.5, headlength=5.0, headaxislength=4.5, monitor_dpi=96)






# offset plotting for main image
s_orig = hs.load('STEM 20191121 HAADF 3.3 Mx 0228.emd')
sampling = s_orig.axes_manager[-1].scale
units = s_orig.axes_manager[-1].units
image = s_orig.data
monitor_dpi=200

offsets = np.load("xy_offset_nm and pix.npy")
x_offset_pix= offsets[1][0]
y_offset_pix= offsets[1][1]
x_os = [i+x_offset_pix for i in x]
y_os = [i+y_offset_pix for i in y]


tml_pol.plot_polarisation_vectors(
        x=x_os, y=y_os, u=u, v=v, image=image,
        sampling=sampling, units=units,
        normalise=True, overlay=True,
        save=None, color='yellow',
        plot_style='vector', title='Ti Polarisation',
        pivot='middle', angles='xy', scale_units='xy', scale=None,
        headwidth=3.5, headlength=5.0, headaxislength=4.5, monitor_dpi=monitor_dpi)



plot_style_list = ['vector', 'vector', 'vector', 'colormap',
                   'colormap', 'contour']
normal_list = [False, True, False, False, False, True]
overlay_list = [True, True, False, True, False, True]
save_list = ['', '_normal', '_no_overlay', '', '_no_overlay', '_normal']
color_list = ['yellow', 'yellow', 'red', 'red', 'red', 'gray']
cmap_list = [None, None, None, None, 'inferno', 'viridis']

for plot_style, normal, overlay, save, color, cmap in zip(
        plot_style_list, normal_list, overlay_list, save_list,
        color_list, cmap_list):

    tml_pol.plot_polarisation_vectors(
        x=x_os, y=y_os, u=u, v=v, image=image,
        sampling=sampling, units=units,
        normalise=normal, overlay=overlay, cmap=cmap,
        save="Ti Polarisation Full Image" + save, color=color,
        plot_style=plot_style, title='Ti Polarisation',
        pivot='middle', angles='xy', scale_units='xy', scale=None,
        headwidth=3.5, headlength=5.0, headaxislength=4.5, monitor_dpi=monitor_dpi)



divide_into_list = [4,8,10,12,16,20,24,28,32,36,40]

x = [i[0] for i in vector_list]
y = [i[1] for i in vector_list]
u = [i[2] for i in vector_list]
v = [i[3] for i in vector_list]
u = [-1*i for i in u]
v = [-1*i for i in v]

for divide_into in divide_into_list:
        
    x_new, y_new, u_new, v_new = get_average_polarisation_in_regions_square(
        x, y, u, v, image=atom_lattice.image, divide_into=divide_into)
    
    x_os = [i+x_offset_pix for i in x_new]
    y_os = [i+y_offset_pix for i in y_new]
    
    tml_pol.plot_polarisation_vectors(
            x=x_os, y=y_os, u=u_new, v=v_new, image=image,
            sampling=sampling, units=units,
            normalise=True, overlay=True,
            save='Averaged Ti Polarisation Full Image {}'.format(divide_into), color='yellow',
            plot_style='vector', title='Averaged Ti Polarisation',
            pivot='middle', angles='xy', scale_units='xy', scale=None,
            headwidth=3.5, headlength=5.0, headaxislength=4.5, monitor_dpi=monitor_dpi)

