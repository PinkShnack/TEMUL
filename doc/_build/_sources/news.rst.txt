.. _news:

.. include:: define_roles.rst


News
----

.. new version additions:


16/02/2021: Version 0.1.4 released
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The polarisation, structure tools and fft mapping has now
been refactored into the topotem module. The temul
functionality remains the same i.e. ``import temul.api as tml``.


16/02/2021: Version 0.1.3 released
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First articles uses and citations for the TEMUL Toolkit!
This version updated the Publication Examples folder with two newly published
articles. The folder contains interactive and raw code on how to reproduce the
data in the publications. Congrats to those involved!
    * M. Hadjimichael, Y. Li *et al*, `Metal-ferroelectric supercrystals with periodically curved metallic layers, Nature Materials 2020 <https://www.nature.com/articles/s41563-020-00864-6>`_
    * K. Moore *et al* `Highly charged 180 degree head-to-head domain walls in lead titanate, Nature Communications Physics 2020 <https://www.nature.com/articles/s42005-020-00488-x>`_
If you have a question or issue with using the publication examples, please make
an issue on `GitHub <https://github.com/PinkShnack/TEMUL/issues>`_.

Code changes in this version:
    * The :python:`atom_deviation_from_straight_line_fit` function has been **corrected** and expanded. For a use case, see :ref:`Finding Polarisation Vectors <polarisation_vectors_tutorial>`
    * Corrected the :python:`plot_polarisation_vectors` function's vector quiver key.
    * Created the "polar_colorwheel" :python:`plot_style` for :python:`plot_polarisation_vectors` by using a HSV to RGB 2D colorwheel and mapping the angles and magnitudes to these values. Used code from `PixStem <https://pixstem.org/>`_ for colorwheel visualisation. 
    * Fixed :python:`norm` and :python:`cmap` scaling for the colorbar for the "contour", "colorwheel" and "colormap" :python:`plot_styles`. Now each of these :python:`plot_styles` scale nicely, and colorbar :python:`ticks` may be specified.
    * Added invert_y_axis param for plot_polarisation_vectors function, useful for testing if angles are displaying correctly.
    * :python:`plot_polarisation_vectors` function now returns a Matplotlib :python:`Axes` object, which can be used to further edit the layout/presentation of the plotted map.
    * Added functions to correct for possible off-zone tilt in the atomic columns. Use with caution.

Documentation changes in this version:
    * Added documentation for :ref:`how to find the polarisation vectors <polarisation_vectors_tutorial>`.
    * Added "code_tutorials" ipynb (interactive Jupyter Notebook) examples. See the `GitHub repository <https://github.com/PinkShnack/TEMUL>`_ for downloads.
    * The :ref:`workflows <workflows>` folder in "code_tutorials/workflows" also contains starting workflows for analysis of different materials. See the `GitHub repository <https://github.com/PinkShnack/TEMUL>`_ for downloads.
    * Added "publication_examples" tutorial ipynb (interactive Jupyter Notebook) examples. See the `GitHub repository <https://github.com/PinkShnack/TEMUL>`_ for downloads.


03/11/2020: Version 0.1.2 released
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This version contains minor changes from the 0.1.1 release. It removes pyCifRW
as a dependency.


02/11/2020: Version 0.1.1 released
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This version contains many changes to the TEMUL Toolkit.

    * More parameters have been added to the polarisation module's :python:`plot_polarisation_vectors` function. Check out the walkthrough :ref:`here <polarisation_vectors_tutorial>` for more info!
    * :ref:`Interactive double Gaussian filtering <dg_visualiser_tutorial>` with the :python:`visualise_dg_filter` function in the signal_processing module. Thanks to `Michael Hennessy <https://github.com/michaelhennessyjr>`_ for the help!
    * The :python:`calculate_atom_plane_curvature` function has been added, creating the lattice_structure_tools module.
    * Strain, rotation, and c/a mapping can now be done :ref:`here <structure_map_tutorial>`.
    * Masked FFT filtering to obtain iFFTs. See :ref:`this guide <masked_fft_tutorial>` to see some code!
    * Example walk-throughs for many features of the TEMUL Toolkit are now on this website! Check out the menu on the left to get started!
