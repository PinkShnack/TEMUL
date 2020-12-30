.. _install:

.. include:: define_roles.rst


News
----

.. new version additions:
    Corrected atom_deviation_from_straight_line_fit function.
    Added documentation for how to find the polarisation vectors
    Added "code_tutorials" ipynb examples
    Corrected plot_polarisation_vectors vector quiver key
    Created "polar_colorwheel" `plot_style` for plot_polarisation_vectors by
        using a HSV to RGB 2D colorwheel and mapping the angles and magnitudes to
        these values. Used code from PixStem for colorwheel visualisation. 
    Fixed norm and cmap scaling for the colorbar for "contour", "colorwheel" and
        "colormap" plot_styles. Now they scale nicely and ticks can be input to all three.
    Added invert_y_axis param for plot_polarisation_vectors function
    plot_polarisation_vectors function now returns the Axes object
    

**03/11/2020: Version 0.1.2 released**

This version contains minor changes from the 0.1.1 release. It removes pyCifRW
as a dependency.


**02/11/2020: Version 0.1.1 released**

This version contains many changes to the TEMUL Toolkit.

    * More parameters have been added to the polarisation module's :python:`plot_polarisation_vectors` function. Check out the walkthrough :ref:`here <polarisation_vectors_tutorial>` for more info!
    * :ref:`Interactive double Gaussian filtering <dg_visualiser_tutorial>` with the :python:`visualise_dg_filter` function in the signal_processing module. Thanks to `Michael Hennessy <https://github.com/michaelhennessyjr>`_ for the help!
    * The :python:`calculate_atom_plane_curvature` function has been added, creating the lattice_structure_tools module.
    * Strain, rotation, and c/a mapping can now be done :ref:`here <structure_map_tutorial>`.
    * Masked FFT filtering to obtain iFFTs. See :ref:`this guide <masked_fft_tutorial>` to see some code!
    * Example walk-throughs for many features of the TEMUL Toolkit are now on this website! Check out the menu on the left to get started!
