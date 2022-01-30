.. _line_profile_tutorial:

.. include:: define_roles.rst


==================================
Line Intensity Profile Comparisons
==================================

The :py:mod:`temul.signal_plotting` module allows one to draw line
intensity profiles over images. The
:py:func`temul.signal_plotting.compare_images_line_profile_one_image` can be used to
draw two line profiles on one image for comparison. In future we hope to expand
this function to allow for multiple line profiles on one image. The
:py:func:`temul.signal_plotting.compare_images_line_profile_two_images`
function allows you to draw a line
profile on an image, and apply that same profile to another image (of the same shape).
This can be useful for comparing subsequent images in series or comparing experimental
and simulated images.

Check out the examples below for each comparison method.



Load the Example Images
-----------------------

Here we load some dummy data using a variation of Atomap's
:py:func:`temul.dummy_data.get_simple_cubic_signal` function.

.. code-block:: python

    >>> from temul.dummy_data import get_simple_cubic_signal
    >>> imageA = get_simple_cubic_signal(image_noise=True, amplitude=[1, 5])
    >>> imageA.plot()
    >>> imageB = get_simple_cubic_signal(image_noise=True, amplitude=[3, 7])
    >>> imageB.plot()
    >>> sampling, units = 0.1, 'nm'

.. image:: tutorial_images/line_profile_tutorial/imageA.png
    :scale: 50 %

.. image:: tutorial_images/line_profile_tutorial/imageB.png
    :scale: 50 %


Compare two Line Profiles in one Image
--------------------------------------

As with the :ref:`Masked FFT and iFFT <masked_fft_tutorial>` tutorial, we can
choose points on the image. This time we use
:py:func:`temul.topotem.fft_mapping.choose_points_on_image`.
We need to choose four points for the
:py:func:`temul.signal_plotting.compare_images_line_profile_one_image`
function, as it draws two line profiles over one image.

.. code-block:: python

    >>> import temul.api as tml
    >>> line_profile_positions = tml.choose_points_on_image(imageA)
    >>> line_profile_positions
    [[61.75132848177407, 99.25182885155715],
     [178.97030854763057, 96.60281235289372],
     [61.75132848177407, 186.0071191827843],
     [177.64580029829887, 184.6826109334526]]

.. image:: tutorial_images/line_profile_tutorial/choose_points_of_image_4points.gif
    :scale: 60 %
    :align: center

Now run the comparison function to display the two line intensity profiles.

.. code-block:: python

    >>> tml.compare_images_line_profile_one_image(
    ...             imageA, line_profile_positions, linewidth=5,
    ...             sampling=sampling, units=units, arrow='h', linetrace=1)

.. image:: tutorial_images/line_profile_tutorial/comparison1.png
    :scale: 70 %
    :align: center


Compare two Images with Line Profile
------------------------------------

Using :py:func:`temul.topotem.fft_mapping.choose_points_on_image`,
we now choose two points on one image.
Then, we plot this line intensity profile over the same position in two images.

.. code-block:: python

    >>> line_profile_positions = tml.choose_points_on_image(imageA)
    >>> line_profile_positions
    [[127.31448682369383, 46.93375300295452],
     [127.97674094835968, 176.7355614374623]]

.. image:: tutorial_images/line_profile_tutorial/choose_points_of_image_2points.gif
    :scale: 60 %
    :align: center

.. code-block:: python

    >>> import numpy as np
    >>> tml.compare_images_line_profile_two_images(imageA, imageB,
    ...             line_profile_positions, linewidth=5, reduce_func=np.mean,
    ...             sampling=sampling, units=units, crop_offset=50,
    ...             imageA_title="Image A", imageB_title="Image B")

.. image:: tutorial_images/line_profile_tutorial/comparison2.png
    :scale: 70 %
    :align: center
