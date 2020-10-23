.. _polarisation_vectors_tutorial:

.. include:: define_roles.rst


==========================================
Plotting Polarisation and Movement Vectors
==========================================

The :python:`temul.polarisation` module allows one to visualise the
polarisation/movement of atoms in an atomic resolution image. In this tutorial,
we will use a dummy dataset to show the different ways the :python:`plot_polarisation_vectors`
function can display data. In future, tutorials on published experimental data
will also be available.


Prepare and Plot the dummy dataset
----------------------------------

.. code-block:: python

    >>> from temul.polarisation import plot_polarisation_vectors
    >>> from temul.dummy_data import get_polarisation_dummy_dataset
    >>> atom_lattice = get_polarisation_dummy_dataset(image_noise=True)
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeB = atom_lattice.sublattice_list[1]
    >>> image = sublatticeA.signal
    >>> image.plot()

.. image:: tutorial_images/polarisation_vectors_tutorial/image_uncalibrated.png
    :scale: 60 %

It is best when the image is calibrated. Your image may already be calibrated,
but if not, use Hyperspy's :python:`axes_manager` for calibration.

.. code-block:: python

    >>> sampling = 0.1  # example of 0.1 nm/pix
    >>> units = 'nm'
    >>> image.axes_manager[-1].scale = sampling
    >>> image.axes_manager[-2].scale = sampling
    >>> image.axes_manager[-1].units = units
    >>> image.axes_manager[-2].units = units
    >>> image.plot()

.. image:: tutorial_images/polarisation_vectors_tutorial/image_calibrated.png
    :scale: 60 %

Zoom in on the image to see how the atoms look in the different regions.

.. image:: tutorial_images/polarisation_vectors_tutorial/image_calibrated_labelled_atoms.png
    :scale: 60 %

Find the Vector Coordinates using Atomap
----------------------------------------

Using the `Atomap <https://atomap.org/>`_ package, we can easily get the
polarisation vectors for regular structures.

    >>> sublatticeA.construct_zone_axes()
    >>> za0, za1 = sublatticeA.zones_axis_average_distances[0:2]
    >>> s_p = sublatticeA.get_polarization_from_second_sublattice(
    ...     za0, za1, sublatticeB, color='blue')
    >>> vector_list = s_p.metadata.vector_list
    >>> x, y = [i[0] for i in vector_list], [i[1] for i in vector_list]
    >>> u, v = [i[2] for i in vector_list], [i[3] for i in vector_list]

Now we can display all of the variations that :python:`plot_polarisation_vectors`
gives us! You can specify sampling (scale) and units, or use a calibrated image
so that they are automatically set.

Vector magnitude plot with red arrows:

.. code-block:: python

    >>> plot_polarisation_vectors(x, y, u, v, image=image,
    ...                           unit_vector=False, save=None,
    ...                           plot_style='vector', color='r',
    ...                           overlay=False, title='Vector Arrows',
    ...                           monitor_dpi=50)

.. image:: tutorial_images/polarisation_vectors_tutorial/vectors_red.png
    :scale: 60 %

Vector magnitude plot with red arrows overlaid on the image, no title:

.. code-block:: python

    >>> plot_polarisation_vectors(x, y, u, v, image=image,
    ...                           unit_vector=False, save=None,
    ...                           plot_style='vector', color='r',
    ...                           overlay=True, monitor_dpi=50)

.. image:: tutorial_images/polarisation_vectors_tutorial/vectors_red_overlay.png
    :scale: 60 %


Vector magnitude plot with colormap viridis:

.. code-block:: python

    >>> plot_polarisation_vectors(x, y, u, v, image=image,
    ...                           unit_vector=False, save=None,
    ...                           plot_style='colormap', monitor_dpi=50,
    ...                           overlay=False, cmap='viridis')


.. image:: tutorial_images/polarisation_vectors_tutorial/colormap_magnitude.png
    :scale: 60 %


Vector angle plot with colormap viridis (:python:`vector_rep='angle'`):

.. code-block:: python

    >>> plot_polarisation_vectors(x, y, u, v, image=image,
    ...                           unit_vector=False, save=None,
    ...                           plot_style='colormap', monitor_dpi=50,
    ...                           overlay=False, cmap='cet_colorwheel',
    ...                           vector_rep="angle", degrees=True)

.. image:: tutorial_images/polarisation_vectors_tutorial/colormap_angle.png
    :scale: 60 %


Colormap arrows with sampling specified in the parameters and with scalebar:

.. code-block:: python

    >>> plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           sampling=3.0321, units='pm', monitor_dpi=50,
    ...                           unit_vector=False, plot_style='colormap',
    ...                           overlay=True, save=None, cmap='viridis',
    ...                           scalebar=True)

.. image:: tutorial_images/polarisation_vectors_tutorial/colormap_magnitude_overlay_sb_pm.png
    :scale: 60 %


Vector plot with colormap viridis and unit vectors:

.. code-block:: python

    >>> plot_polarisation_vectors(x, y, u, v, image=image,
    ...                           unit_vector=True, save=None, monitor_dpi=50,
    ...                           plot_style='colormap', color='r',
    ...                           overlay=False, cmap='viridis')

.. image:: tutorial_images/polarisation_vectors_tutorial/colormap_unitvectors.png
    :scale: 60 %


Change the vectors to unit vectors on a Matplotlib tricontourf map:

.. code-block:: python

    >>> plot_polarisation_vectors(x, y, u, v, image=image, unit_vector=True,
    ...                           plot_style='contour', overlay=False,
    ...                           pivot='middle', save=None, monitor_dpi=50,
    ...                           color='darkgray', cmap='viridis')

.. image:: tutorial_images/polarisation_vectors_tutorial/contour_magnitude_unitvectors.png
    :scale: 60 %


Plot a partly transparent angle tricontourf map with specified colorbar ticks
and vector arrows:

.. code-block:: python

    >>> plot_polarisation_vectors(x, y, u, v, image=image,
    ...                           unit_vector=False, plot_style='contour',
    ...                           overlay=True, pivot='middle', save=None,
    ...                           color='red', cmap='cet_colorwheel',
    ...                           monitor_dpi=50, remove_vectors=False,
    ...                           vector_rep="angle", alpha=0.5, levels=9,
    ...                           antialiased=True, degrees=True,
    ...                           ticks=[180, 90, 0, -90, -180])

.. image:: tutorial_images/polarisation_vectors_tutorial/contour_angle_trans_overlay_vectors.png
    :scale: 60 %


Plot a partly transparent angle tricontourf map with no vector arrows:

.. code-block:: python

    >>> plot_polarisation_vectors(x, y, u, v, image=image, remove_vectors=True,
    ...                           unit_vector=True, plot_style='contour',
    ...                           overlay=True, pivot='middle', save=None,
    ...                           cmap='cet_colorwheel', alpha=0.5,
    ...                           monitor_dpi=50, vector_rep="angle",
    ...                           antialiased=True, degrees=True)

.. image:: tutorial_images/polarisation_vectors_tutorial/contour_angle_trans_overlay.png
    :scale: 60 %


"colorwheel" plot of the vectors, useful for visualising vortexes:

.. code-block:: python

    >>> import colorcet as cc
    >>> plot_polarisation_vectors(x, y, u, v, image=image,
    ...                           unit_vector=True, plot_style="colorwheel",
    ...                           vector_rep="angle",
    ...                           overlay=False, cmap=cc.cm.colorwheel,
    ...                           degrees=True, save=None, monitor_dpi=50)

.. image:: tutorial_images/polarisation_vectors_tutorial/colorwheel_angle.png
    :scale: 60 %


Plot with a custom scalebar. In this example, we need it to be dark, see
matplotlib-scalebar for more custom features.

.. code-block:: python

    >>> scbar_dict = {"dx": 3.0321, "units": "pm", "location": "lower left",
    ...               "box_alpha":0.0, "color": "black", "scale_loc": "top"}
    >>> plot_polarisation_vectors(x, y, u, v, image=sublatticeA.image,
    ...                           sampling=3.0321, units='pm', monitor_dpi=50,
    ...                           unit_vector=False, plot_style='colormap',
    ...                           overlay=False, save=None, cmap='viridis',
    ...                           scalebar=scbar_dict)

.. image:: tutorial_images/polarisation_vectors_tutorial/colormap_magnitude_custom_sb.png
    :scale: 60 %


Plot a tricontourf for quadrant visualisation using a custom matplotlib cmap:

.. code-block:: python

    >>> import temul.signal_plotting as tmlplot
    >>> from matplotlib.colors import from_levels_and_colors
    >>> zest = tmlplot.hex_to_rgb(tmlplot.color_palettes('zesty'))
    >>> zest.append(zest[0])  # make the -180 and 180 degree colour the same
    >>> expanded_zest = tmlplot.expand_palette(zest, [1,2,2,2,1])
    >>> custom_cmap, _ = from_levels_and_colors(
    ...     levels=range(9), colors=tmlplot.rgb_to_dec(expanded_zest))
    >>> plot_polarisation_vectors(x, y, u, v, image=image,
    ...                           unit_vector=False, plot_style='contour',
    ...                           overlay=False, pivot='middle', save=None,
    ...                           cmap=custom_cmap, levels=9, monitor_dpi=50,
    ...                           vector_rep="angle", alpha=0.5, color='r',
    ...                           antialiased=True, degrees=True,
    ...                           ticks=[180, 90, 0, -90, -180])

.. image:: tutorial_images/polarisation_vectors_tutorial/contour_angle_custom_cmap_vectors.png
    :scale: 60 %
