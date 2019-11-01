
import numpy as np
import hyperspy.api as hs
from skimage.measure import profile_line

import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots_adjust

# line_profile_positions = am.add_atoms_with_gui(s)


def compare_images_line_profile_one_image(image,
                                          line_profile_positions,
                                          linewidth=1,
                                          image_sampling='Auto',
                                          arrow=None,
                                          linetrace=None):
    '''
    Plots two line profiles on one image with the line profile intensities
    in a subfigure.
    See skimage PR PinkShnack for details on implementing profile_line
    in skimage

    Parameters
    ----------

    image : 2D Hyperspy signal
    line_profile_positions : list of lists
        two line profile coordinates. Use atomap's am.add_atoms_with_gui()
        function to get these. The first two dots will trace the first line
        profile etc.
        Could be extended to n positions with a basic loop.
    linewidth : int, default 1
        see profile_line for parameter details.
    image_sampling :  float, default 'Auto'
        if set to 'Auto' the function will attempt to find the sampling of
        image from image.axes_manager[0].scale.
     arrow : string, default None
        If set, arrows will be plotting on the image. Options are 'h' and
        'v' for horizontal and vertical arrows, respectively.
    linetrace : int, default None
        If set, the line profile will be plotted on the image.
        The thickness of the linetrace will be linewidth*linetrace.
        Name could be improved maybe.

    Returns
    -------

    Examples
    --------

    Include PTO example from paper
    '''

    if image_sampling == 'Auto':
        if image.axes_manager[0].scale != 1.0:
            image_sampling = image.axes_manager[0].scale
        else:
            raise ValueError("The image sampling cannot be computed."
                             "The image should either be calibrated "
                             "or the image_sampling provided")

    # image.axes_manager[0].scale = 1
    # image.axes_manager[1].scale = 1
    # image.axes_manager[0].units = scale_unit
    # image.axes_manager[1].units = scale_unit

    # llim, tlim = cropping_area[0]
    # rlim, blim = cropping_area[1]

    # for many line plots:
    # for i, line_profile in enumerate(line_profile_positions):
    #     print(i)
    #     print(line_profile)
    #   expand

    x0, y0 = line_profile_positions[0]
    x1, y1 = line_profile_positions[1]

    x2, y2 = line_profile_positions[2]
    x3, y3 = line_profile_positions[3]

    profile_y_1 = profile_line(image=image.data, src=[y0, x0], dst=[y1, x1],
                               linewidth=linewidth)
    profile_x_1 = np.arange(0, len(profile_y_1), 1)
    profile_x_1 = profile_x_1*image_sampling

    profile_y_2 = profile_line(image=image.data, src=[y2, x2], dst=[y3, x3],
                               linewidth=linewidth)
    profile_x_2 = np.arange(0, len(profile_y_2), 1)
    profile_x_2 = profile_x_2*image_sampling

    # -- Plot the line profile comparisons
    _, (ax1, ax2) = plt.subplots(nrows=2)  # figsize=(12, 4)
    subplots_adjust(wspace=0.3)

    ax1.imshow(image.data)
    # ax1.plot([x0, x1], [y0, y1], color='r', marker='v', markersize=10,
    # alpha=0.5)

    alpha = 0.5
    color_1 = 'r'
    color_2 = 'b'

    if linetrace is not None:

        ax1.plot([x0, x1], [y0, y1], color=color_1,
                 lw=linewidth*linetrace, alpha=alpha)
        ax1.plot([x2, x3], [y2, y3], color=color_2, ls='-',
                 lw=linewidth*linetrace, alpha=alpha)

    if arrow is not None:
        if arrow == 'v':
            plot_arrow_1 = '^'  # r'$\uparrow$' #u'$/U+2190$'
            plot_arrow_2 = 'v'  # r'$\downarrow$' #u'$/u2193$'

        if arrow == 'h':
            plot_arrow_1 = '>'  # r'$\rightarrow$' #u'$/u2192$'
            plot_arrow_2 = '<'  # r'$\leftarrow$' #u'$/u2194$'

        arrow_markersize = 10

        ax1.plot(x0, y0, color=color_1, marker=plot_arrow_1,
                 markersize=arrow_markersize, alpha=alpha)
        ax1.plot(x1, y1, color=color_1, marker=plot_arrow_2,
                 markersize=arrow_markersize, alpha=alpha)

        ax1.plot(x2, y2, color=color_2, marker=plot_arrow_1,
                 markersize=arrow_markersize, alpha=alpha)
        ax1.plot(x3, y3, color=color_2, marker=plot_arrow_2,
                 markersize=arrow_markersize, alpha=alpha)

    ax1.yaxis.set_ticks([])
    ax1.xaxis.set_ticks([])

    ax2.plot(profile_x_1, profile_y_1, color='r',
             label='1')
    ax2.plot(profile_x_2, profile_y_2, color='b',
             ls='--', label='2')
    ax2.set_title('Intensity Profile')
    ax2.set_ylabel('Intensity (a.u.)')
    ax2.set_xlabel('Distance (nm)')
    # ax2.set_ylim(max(profile_y_1), min(profile_y_1))
    ax2.legend(fancybox=True, loc='upper right')
    # ax2.yaxis.set_ticks([])
    # ax2.xaxis.set_ticks([])
    # ax2.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()

    # plt.savefig(fname='Line Profile Example.png',
    #             transparent=True, frameon=False, bbox_inches='tight',
    #             pad_inches=None, dpi=300)


# line_profile_positions = am.add_atoms_with_gui(s)


def compare_images_line_profile_two_images(imageA,
                                           imageB,
                                           line_profile_positions,
                                           linewidth=1,
                                           image_sampling='Auto'):
    '''
    Plots two line profiles on two images separately with the line
    profile intensities in a subfigure.
    See skimage PR PinkShnack for details on implementing profile_line
    in skimage

    Parameters
    ----------

    imageAm, imageB : 2D Hyperspy signal
    line_profile_positions : list of lists
        one line profile coordinate. Use atomap's am.add_atoms_with_gui()
        function to get these. The two dots will trace the line profile.
    linewidth : int, default 1
        see profile_line for parameter details.
    image_sampling :  float, default 'Auto'
        if set to 'Auto' the function will attempt to find the sampling of
        image from image.axes_manager[0].scale.

    Returns
    -------

    Examples
    --------

    Include PTO example from paper
    '''

    # if image_sampling == 'Auto':
    # compute the sampling

    image_sampling = 0.1  # nm
    scale_unit = 'pix'
    linewidth = 10
    offset = 40

    plot_arrow_up = '^'  # r'$\uparrow$' #u'$/u2191$'
    plot_arrow_down = 'v'  # r'$\downarrow$' #u'$/u2193$'

    # plot_arrow_right = r'$\rightarrow$' #u'$/u2192$'
    # plot_arrow_left = r'$\leftarrow$' #u'$/u2194$'
    arrow_markersize = 10

    imageA.axes_manager[0].scale = 1
    imageA.axes_manager[1].scale = 1
    imageA.axes_manager[0].units = scale_unit
    imageA.axes_manager[1].units = scale_unit

    # llim, tlim = cropping_area[0]
    # rlim, blim = cropping_area[1]

    x0, y0 = line_profile_positions[0]
    x1, y1 = line_profile_positions[1]

    profile_exp = profile_line(image=imageA.data, src=[y0, x0], dst=[y1, x1],
                               linewidth=linewidth)
    profile_y = np.arange(0, len(profile_exp), 1)
    profile_y = profile_y*image_sampling

    crop_left = x0 - offset
    crop_right = x1 + offset
    crop_top = y0 - offset
    crop_bot = y1 + offset

    imageA_crop = hs.roi.RectangularROI(left=crop_left, right=crop_right,
                                        top=crop_top, bottom=crop_bot)(imageA)

    # for the final plot, set the marker positions
    crop_x0 = offset
    crop_y0 = offset
    crop_x1 = offset + (x1 - x0)
    crop_y1 = offset + (y1 - y0)

    ''' Simulation '''
    imageB.axes_manager[0].scale = 1
    imageB.axes_manager[1].scale = 1
    imageB.axes_manager[0].units = scale_unit
    imageB.axes_manager[1].units = scale_unit

    profile_sim = profile_line(image=imageB.data, src=[y0, x0], dst=[y1, x1],
                               linewidth=linewidth)
    profile_y_sim = np.arange(0, len(profile_sim), 1)
    profile_y_sim = profile_y_sim*image_sampling

    imageB_crop = hs.roi.RectangularROI(left=crop_left, right=crop_right,
                                        top=crop_top, bottom=crop_bot)(imageB)

    # -- Plot the line profile comparisons
    _, (ax1, ax2, ax3) = plt.subplots(ncols=3)  # figsize=(12, 4)
    subplots_adjust(wspace=0.3)
    ax1.imshow(imageA_crop)
    # ax1.plot([x0, x1], [y0, y1], color='r', marker='v', markersize=10,
    # alpha=0.5)
    ax1.plot(crop_x0, crop_y0, color='r', marker=plot_arrow_down,
             markersize=arrow_markersize, alpha=1)
    ax1.plot(crop_x1, crop_y1, color='r', marker=plot_arrow_up,
             markersize=arrow_markersize, alpha=1)
    ax1.set_title('Experiment')
    ax1.yaxis.set_ticks([])
    ax1.xaxis.set_ticks([])

    ax2.imshow(imageB_crop)
    ax2.plot(crop_x0, crop_y0, color='b', marker=plot_arrow_down,
             markersize=arrow_markersize, alpha=1)
    ax2.plot(crop_x1, crop_y1, color='b', marker=plot_arrow_up,
             markersize=arrow_markersize, alpha=1)
    ax2.set_title('Simulation')
    ax2.yaxis.set_ticks([])
    ax2.xaxis.set_ticks([])

    ax3.plot(profile_exp, profile_y, color='r', label='Exp')
    ax3.plot(profile_sim, profile_y_sim, color='b',
             linestyle='--', label='Sim')
    ax3.set_title('Intensity Profile')
    ax3.set_xlabel('Intensity (a.u.)')
    ax3.set_ylabel('Distance (nm)')
    ax3.set_ylim(max(profile_y), min(profile_y))
    ax3.legend(fancybox=True, loc='upper right')
    # ax3.yaxis.set_ticks([])
    # ax3.xaxis.set_ticks([])
    # ax3.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()

    plt.savefig(fname='Line Profile Example.png',
                transparent=True, frameon=False, bbox_inches='tight',
                pad_inches=None, dpi=300)
