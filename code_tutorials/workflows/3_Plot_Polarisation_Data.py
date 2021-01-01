
import temul.api as tml

''' Plot the polarisation vectors. Set the parameters below.

Don't be overwhelmed, plot_style is the most important input! Many of the rest
you can ignore.

Note: image, sampling, units are loaded from the initial loading step, but you
      can do that here.
Note: u, v, x, y are all loaded from the previous step
'''

save = None  # save = 'example_name' will save the image

# plot_style options are:
#   'vector', 'colormap', 'contour, 'colorwheel', 'polar_colorwheel'
# for the DPC-style colorwheel just use plot_style = 'polar_colorwheel'
plot_style = 'vector'
overlay = True  # the vectors will be plotted on the image
unit_vector = False  # formerly called normalise
vector_rep = 'magnitude'  # 'magnitude' or 'angle'
degrees = False  # Set to True for degrees, False for radians
angle_offset = None
title = ""
color = 'yellow'  # may be ignored depending on the plot_style
cmap = 'viridis'  # may be ignored depending on the plot_style
alpha = 1.0  # transparency of image or vectors, depending on plot_style
image_cmap = 'gray'
ticks = None
scalebar = False
monitor_dpi = None  # set to ~200 to make too-large images a bit smaller
no_axis_info = True  
invert_y_axis = True
antialiased = False  # relevant for the contour mapping
levels = 20  # relevant for the contour mapping
remove_vectors = False
scale = None  # set to 0.001-0.01 to change arrow size
width = None  # set to ~0.005 for chunky (thicker) arrows
minshaft = 1
minlength = 1
headwidth = 3.0
headlength = 5.0
headaxislength = 4.5
quiver_units = 'width'
pivot = 'middle'
angles = 'xy'
scale_units = 'xy'


# plot the vectors!
ax_vectors = tml.plot_polarisation_vectors(
    x, y, u, v, image.data, sampling=sampling, units=units,
    plot_style=plot_style, overlay=overlay, unit_vector=unit_vector,
    vector_rep=vector_rep, degrees=degrees, angle_offset=angle_offset,
    save=save, title=title, color=color, cmap=cmap, alpha=alpha,
    image_cmap=image_cmap, monitor_dpi=monitor_dpi,
    no_axis_info=no_axis_info, invert_y_axis=invert_y_axis, ticks=ticks,
    scalebar=scalebar, antialiased=antialiased, levels=levels,
    remove_vectors=remove_vectors, quiver_units=quiver_units, pivot=pivot,
    angles=angles, scale_units=scale_units, scale=scale, headwidth=headwidth,
    headlength=headlength, headaxislength=headaxislength, width=width,
    minshaft=minshaft, minlength=minlength)

