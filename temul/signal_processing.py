
from temul.element_tools import split_and_sort_element
from temul.io import save_individual_images_from_image_stack
from temul.external.atomap_devel_012.initial_position_finding import (
    add_atoms_with_gui as choose_points_on_image)

import atomap.api as am
from atomap.atom_finding_refining import _make_circular_mask

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

from tifffile import imread
from skimage.metrics import structural_similarity as ssm
import scipy
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import OptimizeWarning
from math import sqrt
import numpy as np
from numpy import mean
import hyperspy.api as hs
import pandas as pd
import copy
from tqdm import trange
import collections
import warnings
warnings.simplefilter("error", OptimizeWarning)


def get_xydata_from_list_of_intensities(
        sublattice_intensity_list,
        hist_bins=100):
    '''
    Output x and y data for a histogram of intensities

    Parameters
    ----------
    sublattice_intensity_list : list
        See get_subattice_intensity() for more information
    hist_bins : int, default 100
        number of bins to sort the intensities into
        must be a better way of doing this? maybe automate the binning choice

    Returns
    -------
    Two numpy 1D arrays corressponding to the x and y values of a histogram
    of the sublattice intensities.

    Examples
    --------

    >>> from temul.signal_processing import get_xydata_from_list_of_intensities
    >>> amp, mu, sigma = 10, 10, 0.5
    >>> sub1_inten = np.random.normal(mu, sigma, 1000)
    >>> xdata, ydata = get_xydata_from_list_of_intensities(sub1_inten,
    ...     hist_bins=50)
    '''

    hist_bins = hist_bins
    y_array, x_array = np.histogram(sublattice_intensity_list,
                                    bins=hist_bins)
    x_array = np.delete(x_array, [0])

    # set the x_ values so that they are at the middle of each hist bin
    x_separation = (x_array.max() - x_array.min()) / hist_bins
    x_array = x_array - (x_separation / 2)

    return(x_array, y_array)

# 1D single Gaussian
# test


def fit_1D_gaussian_to_data(xdata, amp, mu, sigma):
    '''
    Fitting function for a single 1D gaussian distribution

    Parameters
    ----------
    xdata : numpy 1D array
        values input as the x coordinates of the gaussian distribution
    amp : float
        amplitude of the gaussian in y-axis
    mu : float
        mean value of the gaussianin x-axis, corresponding to y-axis
        amplitude.
    sigma : float
        standard deviation of the gaussian distribution

    Returns
    -------
    gaussian distibution of xdata array

    Examples
    --------

    >>> from temul.signal_processing import (
    ...     get_xydata_from_list_of_intensities,
    ...     fit_1D_gaussian_to_data)
    >>> amp, mu, sigma = 10, 10, 0.5
    >>> sub1_inten = np.random.normal(mu, sigma, 1000)
    >>> xdata, ydata = get_xydata_from_list_of_intensities(sub1_inten,
    ...     hist_bins=50)
    >>> gauss_fit_01 = fit_1D_gaussian_to_data(xdata, amp, mu, sigma)
    '''

    return(amp * (1 / (sigma * (np.sqrt(2 * np.pi)))) * (
        np.exp(-((xdata - mu)**2) / ((2 * sigma)**2))))


# Fit gaussian to element
def return_fitting_of_1D_gaussian(
        function,
        xdata,
        ydata,
        amp, mu, sigma):
    '''
    Use the initially found centre (mean/mode) value of a sublattice
    histogram (e.g., Mo_1 in an Mo sublattice) as an input mean for a
    gaussian fit of the data.

    Parameters
    ----------
    xdata, ydata : see scipy.optimize.curve_fit
    amp, mu, sigma : see fit_1D_gaussian_to_data() for more details

    Returns
    -------
    optimised parameters (popt) and estimated covariance (pcov) of the
    fitted gaussian function.

    Examples
    --------

    >>> from temul.signal_processing import (
    ...     get_xydata_from_list_of_intensities,
    ...     return_fitting_of_1D_gaussian,
    ...     fit_1D_gaussian_to_data)
    >>> amp, mu, sigma = 10, 10, 0.5
    >>> sub1_inten = np.random.normal(mu, sigma, 1000)
    >>> xdata, ydata = get_xydata_from_list_of_intensities(sub1_inten,
    ...     hist_bins=50)
    >>> popt_gauss, _ = return_fitting_of_1D_gaussian(
    ...                     fit_1D_gaussian_to_data,
    ...                     xdata, ydata,
    ...                     amp, mu, sigma)

    # print("Calculated Mean: " + str(round(np.mean(xdata),3))
    # + "\n Fitted Mean: " + str(round(popt_gauss[1],3)))
    '''

    popt_gauss, pcov_gauss = scipy.optimize.curve_fit(
        f=function,
        xdata=xdata,
        ydata=ydata,
        p0=[amp, mu, sigma])
    # p0 = [amp, mu, sigma]

    return(popt_gauss, pcov_gauss)


# plot single gauss fit

def plot_gaussian_fit(xdata, ydata, function, amp, mu, sigma,
                      gauss_art='r--', gauss_label='Gauss Fit',
                      plot_data=True,
                      data_art='ko', data_label='Data Points',
                      plot_fill=False,
                      facecolor='r', alpha=0.5):
    # save_image/filename maybe?
    '''
    >>> from temul.signal_processing import (fit_1D_gaussian_to_data,
    ...                                 plot_gaussian_fit,
    ...                                 return_fitting_of_1D_gaussian)
    >>> amp, mu, sigma = 10, 10, 0.5
    >>> sub1_inten = np.random.normal(mu, sigma, 1000)
    >>> xdata, ydata = get_xydata_from_list_of_intensities(sub1_inten,
    ...     hist_bins=50)
    >>> popt_gauss, _ = return_fitting_of_1D_gaussian(
    ...                     fit_1D_gaussian_to_data,
    ...                     xdata, ydata, amp, mu, sigma)
    >>> plot_gaussian_fit(xdata, ydata, function=fit_1D_gaussian_to_data,
    ...           amp=popt_gauss[0], mu=popt_gauss[1], sigma=popt_gauss[2],
    ...           gauss_art='r--', gauss_label='Gauss Fit',
    ...           plot_data=True, data_art='ko', data_label='Data Points',
    ...           plot_fill=True, facecolor='r', alpha=0.5)
    '''

    _gaussian_fit = function(xdata=xdata, amp=amp, mu=mu, sigma=sigma)

    fig = plt.figure(figsize=(4, 3))
    gs = gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(xdata, _gaussian_fit, gauss_art, label=gauss_label)

    if plot_data:
        ax1.plot(xdata, ydata, data_art, label=data_label)
    if plot_fill:
        ax1.fill_between(xdata, _gaussian_fit.min(),
                         _gaussian_fit,
                         facecolor=facecolor,
                         alpha=alpha)
    plt.xlabel("Intensity (a.u.)")
    plt.ylabel("Count")
    plt.title("Gaussian Fit")  # filename input here
    plt.legend(loc='upper right')
    plt.tight_layout()


# def plot_gauss_fit_only(f, array, popt_gauss):

#     ax1.plot(x, f(x, *popt_gauss), 'k--', label='Fit')


def get_scaled_middle_limit_intensity_list(sublattice,
                                           middle_intensity_list,
                                           limit_intensity_list,
                                           sublattice_scalar):
    '''
    Returns the middle and limit lists scaled to the actual intensities in the
    sublattice. Useful for get_fitting_tools_for_plotting_gaussians().
    '''
    middle_intensity_list_real = []
    limit_intensity_list_real = []

    for middle in middle_intensity_list:
        middle_real = middle * sublattice_scalar
        middle_intensity_list_real.append(middle_real)
    for limit in limit_intensity_list:
        limit_real = limit * sublattice_scalar
        limit_intensity_list_real.append(limit_real)

    return(middle_intensity_list_real, limit_intensity_list_real)


def get_fitting_tools_for_plotting_gaussians(element_list,
                                             scaled_middle_intensity_list,
                                             scaled_limit_intensity_list,
                                             fit_bright_first=True,
                                             gaussian_amp=5,
                                             gauss_sigma_division=100):
    '''
    Creates a list of parameters and details for fitting the intensities of a
    sublattice with multiple Gaussians.
    '''

    if len(scaled_middle_intensity_list) + 1 != len(
            scaled_limit_intensity_list):
        raise ValueError(
            "limit list must have a length one greater than middle list")

    if len(element_list) != len(scaled_middle_intensity_list):
        raise ValueError(
            "element list must be the same length as middle list")

    fitting_tools = []
    for i, (element, middle) in enumerate(zip(
            element_list, scaled_middle_intensity_list)):
        element_name = element
        middle_int = middle
        lower_int = scaled_limit_intensity_list[i]
        upper_int = scaled_limit_intensity_list[i + 1]
        gauss_amp = gaussian_amp
        gauss_mu = middle
        gauss_sigma = (upper_int - lower_int) / gauss_sigma_division
        fitting_tools.append([element_name, middle_int, lower_int, upper_int,
                              gauss_amp, gauss_mu, gauss_sigma])

    if fit_bright_first:
        fitting_tools.sort(reverse=True)
    return(fitting_tools)


def plot_gaussian_fitting_for_multiple_fits(sub_ints_all,
                                            fitting_tools_all_subs,
                                            element_list_all_subs,
                                            marker_list,
                                            hist_bins=150,
                                            plotting_style='hist',
                                            filename='Fit of Intensities',
                                            mpl_cmaps_list=['viridis']):
    '''
    plots Gaussian distributions for intensities of a sublattice, over the
    given parameters (fitting tools).

    Example
    -------

    sub_ints_all = [sub1_ints, sub2_ints]
    marker_list = [['Sub1', '.'],['Sub2', 'x']]

    middle_intensity_list_real_sub1, limit_intensity_list_real_sub1 = \
        make_middle_limit_intensity_list_real(
            sublattice=sub1,
            middle_intensity_list=middle_intensity_list_sub1,
            limit_intensity_list=limit_intensity_list_sub1,
            method=method,
            sublattice_scalar=sub1_mode)

    middle_intensity_list_real_sub2, limit_intensity_list_real_sub2 = \
        make_middle_limit_intensity_list_real(
            sublattice=sub2,
            middle_intensity_list=middle_intensity_list_sub2,
            limit_intensity_list=limit_intensity_list_sub2,
            method=method,
            sublattice_scalar=sub2_mode)


    element_list_all_subs = [element_list_sub1, element_list_sub2]

    fitting_tools_all_subs = [
        get_fitting_tools_for_plotting_gaussians(
            element_list_sub1,
            middle_intensity_list_real_sub1,
            limit_intensity_list_real_sub1),
        get_fitting_tools_for_plotting_gaussians(
            element_list_sub2,
            middle_intensity_list_real_sub2,
            limit_intensity_list_real_sub2)
            ]


    plot_gaussian_fitting_for_multiple_fits(sub_ints_all,
                                    fitting_tools_all_subs,
                                    element_list_all_subs,
                                    marker_list,
                                    hist_bins=500,
                                    filename='Fit of Intensities900')

    '''
    # set up cyclers for plotting gaussian fits

    # need to set up a loop here to create as many cyclers as sublattices
    # can be done by appending all to an empty cyclers_all list
    cyclers_all = []
    for i, _ in enumerate(sub_ints_all):
        mpl_cmap = matplotlib.cm.get_cmap(mpl_cmaps_list[i])
        colormap_list = []
        linestyle_list = []
        for j in np.arange(0, 1, 1 / len(element_list_all_subs[0])):
            colormap_list.append(mpl_cmap(j))
            linestyle_list.append('-')

        cycler_sub = plt.cycler(c=colormap_list,
                                linestyle=linestyle_list)

        cyclers_all.append(cycler_sub)

    if len(cyclers_all) != len(element_list_all_subs) != len(
            fitting_tools_all_subs):
        raise ValueError(
            "len(cyclers_all) != len(element_list_all) != "
            "len(fitting_tools_all_subs), "
            + str(len(cyclers_all)) + ', ' +
            str(len(element_list_all_subs)) + ', '
            + str(len(fitting_tools_all_subs)))

    for cycler, element, fitting in zip(
            cyclers_all, element_list_all_subs, fitting_tools_all_subs):
        if len(cycler) != len(element) != len(fitting):
            raise ValueError("len(cycler) != len(element) != len(fitting)")

    plt.rcParams.update({'xtick.labelsize': 'x-large',
                         'ytick.labelsize': 'x-large'})
    plt.rc('font', family='Arial')

    _, (ax1, ax2) = plt.subplots(figsize=(16, 9), nrows=2, sharex=True,
                                 gridspec_kw={'height_ratios': [2, 0.5]})
    plt.subplots_adjust(hspace=0)

    # fig.suptitle("Fit of all Elements with Residuals", family="serif",
    # fontsize=20)
    ax2.set_xlabel("Intensity (a.u.)", family="serif", fontsize=20)
    ax1.set_ylabel("Counts", family="serif", fontsize=20)
    ax2.set_ylabel("Res.", family="serif", fontsize=20)

    sub_residual_gauss_list = []
    for sublattice_array, fitting_tools_sub, cycler_sub, marker, in zip(
            sub_ints_all, fitting_tools_all_subs, cyclers_all, marker_list):
        x_array, y_array = get_xydata_from_list_of_intensities(
            sublattice_array,
            hist_bins=hist_bins)

        if plotting_style == 'scatter':
            ax1.plot(x_array, y_array, color='grey',
                     label=marker[0] + ' Data',
                     marker=marker[1],
                     linestyle='',
                     markersize=4,
                     alpha=0.75)
        elif plotting_style == 'hist':
            ax1.hist(sublattice_array,
                     bins=hist_bins,
                     color='grey',
                     label=marker[0] + 'Data',
                     alpha=0.75)

        for fitting_tools, kwargs in zip(fitting_tools_sub, cycler_sub):
            # label for plotting
            label_info = split_and_sort_element(
                fitting_tools[0])
            label_name = label_info[0][1] + '_{' + str(label_info[0][2]) + '}'

            sliced_array = []
            for atom_int, atom_count in zip(x_array, y_array):
                if fitting_tools[2] < atom_int < fitting_tools[3]:
                    sliced_array.append([atom_int, atom_count])
            sliced_array = np.array(sliced_array)
            if sliced_array.size != 0:
                x = sliced_array[:, 0]
                y = sliced_array[:, 1]

                try:
                    popt_gauss, _ = scipy.optimize.curve_fit(
                        f=fit_1D_gaussian_to_data,
                        xdata=x,
                        ydata=y,
                        p0=[fitting_tools[4], fitting_tools[5],
                            fitting_tools[6]])
                    individual_gauss = fit_1D_gaussian_to_data(x, *popt_gauss)
                    _ = ax1.plot(x, individual_gauss, **kwargs)
                    _ = ax1.fill_between(x,
                                         individual_gauss.min(),
                                         individual_gauss,
                                         facecolor=kwargs['c'],
                                         alpha=0.5)

                    sub_residual_gauss = abs(
                        y - (fit_1D_gaussian_to_data(x, *popt_gauss)))
                    _ = ax1.plot(
                        x, fit_1D_gaussian_to_data(x, *popt_gauss),
                        label=r"$\bf{%s}$ : " % label_name +
                        str(round(
                            sum(abs(sub_residual_gauss)), 1)),
                        linewidth=1.5,
                        **kwargs)

                    sub_residual_gauss_list.append([fitting_tools[0],
                                                    sub_residual_gauss])
                    _ = ax2.plot(
                        x, sub_residual_gauss, marker=marker[1],
                        color='grey',
                        linestyle='',
                        markersize=4,
                        alpha=0.75,
                        label=fitting_tools[0] + ': ' +
                        str(round(sum(abs(sub_residual_gauss)), 1)))

                except OptimizeWarning:
                    print("Warning - Covariance could not be estimated for " +
                          fitting_tools[0] + ", skipping...")
                except RuntimeError:
                    print("Error - curve_fit failed for " +
                          fitting_tools[0] + ", skipping...")
                except TypeError:
                    print("Error (see leastsq in scipy/optimize/minpack) - " +
                          "Not enough data for fitting of " +
                          fitting_tools[0] + ", skipping...")
    # https://stackoverflow.com/questions/48637960/improper-input-n-3-must-
    #   not-exceed-m-1-error-trying-to-fit-a-gaussian-function?rq=1

    legend1 = ax1.legend(
        loc="best", prop={'size': 10}, ncol=2, edgecolor='grey')
    for line in legend1.get_lines():
        line.set_linewidth(1.5)

    # ax1.hist(sub1_ints, bins=500)
    # ax1.hist(sub2_ints, bins=500)
    if filename is not None:
        plt.savefig(fname=filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=900, labels=False)


'''
Image Comparison
# https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
'''


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def measure_image_errors(imageA, imageB, filename=None):
    '''
    Measure the Mean Squared Error (mse) and Structural Similarity Index (ssm)
    between two images.

    Parameters
    ----------

    imageA, imageB : 2D NumPy array, default None
        Two images between which to measure mse and ssm
    filename : string, default None
        name with which the image will be saved

    Returns
    -------

    mse_number, ssm_number : float
    returned subtracted image is imageA - imageB

    Example
    -------

    >>> imageA = am.dummy_data.get_simple_cubic_signal().data
    >>> imageB = am.dummy_data.get_simple_cubic_with_vacancies_signal().data
    >>> mse_number, ssm_number = measure_image_errors(imageA, imageB,
    ...                                               filename=None)

    Showing the ideal case of both images being exactly equal
    >>> imageA = am.dummy_data.get_simple_cubic_signal().data
    >>> imageB = am.dummy_data.get_simple_cubic_signal().data
    >>> mse_number, ssm_number = measure_image_errors(imageA, imageA,
    ...                                               filename=None)

    '''
    if imageA.dtype is not imageB.dtype:
        imageA = imageA.astype('float64')
        imageB = imageB.astype('float64')

    mse_number = mse(imageA, imageB)
    ssm_number = ssm(imageA, imageB)

    if filename is not None:
        plt.figure()
        plt.suptitle("MSE: %.6f, SSIM: %.6f" %
                     (mse_number, ssm_number) + filename)

        plt.subplot(2, 2, 1)
        plt.imshow(imageA)
        plt.title('imageA')
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.imshow(imageB)
        plt.title('imageB')
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.imshow(imageA - imageB)
        plt.title('imageA - imageB')
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.scatter(mse_number.size, mse_number, color='r',
                    marker='x', label='mse')
        plt.scatter(ssm_number.size, ssm_number, color='b',
                    marker='o', label='ssm')
        plt.title('MSE & SSM')
        plt.legend()
        plt.show()

        plt.savefig(fname='MSE_SSM_single_image_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)

    return(mse_number, ssm_number)


# imageA = am.dummy_data.get_simple_cubic_signal().data
# imageB = am.dummy_data.get_simple_cubic_with_vacancies_signal().data
# mse_number, ssm_number = measure_image_errors(imageA, imageB,


def load_and_compare_images(imageA, imageB, filename=None):
    '''
    Load two images and compare their mean standard error and structural
    simularity index.

    Parameters
    ----------

    imageA, imageB : string
        filename of the images to be loaded and compared
    filename : string, default None
        name with which the image will be saved

    Returns
    -------
    mean standard error and structural simularity index

    Examples
    --------
    # >>> # has to be a name for hyperspy to open with hs.load!!!
    # >>> imageA = am.dummy_data.get_simple_cubic_signal(image_noise=True)
    # >>> imageB = am.dummy_data.get_simple_cubic_signal()
    # >>> load_and_compare_images(imageA, imageB, filename=None)

    '''
    imageA = hs.load(imageA)
    imageB = hs.load(imageB)

    mse_number, ssm_number = measure_image_errors(
        imageA,
        imageB,
        filename=filename)

    return(mse_number, ssm_number)


def compare_two_image_and_create_filtered_image(
        image_to_filter,
        reference_image,
        delta_image_filter,
        cropping_area,
        separation,
        filename=None,
        max_sigma=6,
        percent_to_nn=0.4,
        mask_radius=None,
        refine=False):
    '''
    Gaussian blur an image for comparison with a reference image.
    Good for finding the best gaussian blur for a simulation by
    comparing to an experimental image.
    See measure_image_errors() and load_and_compare_images()

    Examples
    --------

    >>> from temul.signal_processing import (
    ...     compare_two_image_and_create_filtered_image)
    >>> import temul.example_data as example_data
    >>> experiment = example_data.load_Se_implanted_MoS2_data()

    # get a simulation from desktop
    #>>> simulation = example_data.load_Se_implanted_MoS2_simulation()
    #>>> filtered_image = compare_two_image_and_create_filtered_image(
    #...     simulation, experiment, 0.5, cropping_area=[[5,5], [20, 20]],
    #...     separation=15, mask_radius=4, percent_to_nn=None)

    '''
    image_to_filter_data = image_to_filter.data
    reference_image_data = reference_image.data

    mse_number_list = []
    ssm_number_list = []

    for i in np.arange(0, max_sigma + delta_image_filter, delta_image_filter):

        image_to_filter_data_filtered = gaussian_filter(image_to_filter_data,
                                                        sigma=i)
        temp_image_filtered = hs.signals.Signal2D(
            image_to_filter_data_filtered)
#        temp_image_filtered.plot()
        calibrate_intensity_distance_with_sublattice_roi(
            image=temp_image_filtered,
            cropping_area=cropping_area,
            separation=separation,
            percent_to_nn=percent_to_nn,
            mask_radius=mask_radius,
            refine=refine,
            filename=None)

        mse_number, ssm_number = measure_image_errors(
            imageA=reference_image_data,
            imageB=temp_image_filtered.data,
            filename=None)

        mse_number_list.append([mse_number, i])
        ssm_number_list.append([ssm_number, i])

    mse = [mse[:1] for mse in mse_number_list]
    mse_indexing = [indexing[1:2] for indexing in mse_number_list]
    ssm = [ssm[:1] for ssm in ssm_number_list]
    ssm_indexing = [indexing[1:2] for indexing in ssm_number_list]

    ideal_mse_number_index = mse.index(min(mse))
    ideal_mse_number = float(
        format(mse_number_list[ideal_mse_number_index][1], '.1f'))

    ideal_ssm_number_index = ssm.index(max(ssm))
    ideal_ssm_number = float(
        format(ssm_number_list[ideal_ssm_number_index][1], '.1f'))

    # ideal is halway between mse and ssm indices
    ideal_sigma = (ideal_mse_number + ideal_ssm_number) / 2
    ideal_sigma_y_coord = (float(min(mse)[0]) + float(max(ssm)[0])) / 2

    image_to_filter_filtered = gaussian_filter(image_to_filter_data,
                                               sigma=ideal_sigma)

    image_filtered = hs.signals.Signal2D(image_to_filter_filtered)

    # calibrate_intensity_distance_with_sublattice_roi(
    #     image=image_filtered,
    #     cropping_area=cropping_area,
    #     separation=separation,
    #     percent_to_nn=percent_to_nn,
    #     mask_radius=mask_radius,
    #     refine=refine,
    #     filename=None)

    if filename is not None:

        plt.figure()
        plt.scatter(x=ssm_indexing, y=ssm, label='ssm',
                    marker='x', color='magenta')
        plt.scatter(x=mse_indexing, y=mse, label='mse', marker='o', color='b')
        plt.scatter(x=ideal_sigma, y=ideal_sigma_y_coord, label='\u03C3 = ' +
                    str(round(ideal_sigma, 2)), marker='D', color='k')
        plt.title("MSE & SSM vs. Gauss Blur " + filename, fontsize=20)
        plt.xlabel("\u03C3 (Gaussian Blur)", fontsize=16)
        plt.ylabel("MSE (0) and SSM (1)", fontsize=16)
        plt.legend()
        plt.tight_layout
        plt.show()
        plt.savefig(fname='MSE_SSM_gaussian_blur_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)

    return(image_filtered)


'''
Image Filtering
'''


def make_gaussian(size, fwhm, center):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    arr = []  # output numpy array
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    arr.append(np.exp(-4 * np.log(2) * ((x - x0)**2 + (y - y0)**2) / fwhm**2))

    return(arr)


def double_gaussian_fft_filter(image, filename,
                               d_inner, d_outer, real_space_sampling,
                               delta=0.05, units='nm'):
    # Folder: G:/SuperStem visit/Feb 2019 data/2019_02_18_QMR_S1574_MoS2-
    # Se-15eV

    # Accuracy of calculation. Smaller = more accurate.
    #   0.01 means it will fit until intensity is 0.01 away from 0
    # 0.01 is a good starting value
    #    delta=0.01

    # Find the FWHM for both positive (outer) and negative (inner) gaussians
    # d_1 is the inner reflection diameter in units of 1/nm (or whatever unit
    # you're working with)
    # I find these in gatan, should be a way of doing automatically.
    #    d_1 = 7.48
    #    d_outer = 14.96

    # image.plot()
    #    image.save('Original Image Data', overwrite=True)
    #    image_name = image.metadata.General.original_filename
    '''
    Example d_inner, d_outer:
    MoS2: d_1 = 7.7, d_2 = 14
    '''

    physical_image_size = real_space_sampling * len(image.data)
    reciprocal_sampling = 1 / physical_image_size

    # Get radius
    reciprocal_d_inner = (d_inner / 2)
    reciprocal_d_outer = (d_outer / 2)
    reciprocal_d_inner_pix = reciprocal_d_inner / reciprocal_sampling
    reciprocal_d_outer_pix = reciprocal_d_outer / reciprocal_sampling

    fwhm_neg_gaus = reciprocal_d_inner_pix
    fwhm_pos_gaus = reciprocal_d_outer_pix

    # s = normalize_signal(subtract_average_background(s))
    image.axes_manager[0].scale = real_space_sampling
    image.axes_manager[1].scale = real_space_sampling
    image.axes_manager[0].units = units
    image.axes_manager[1].units = units
    # image.save('Calibrated Image Data', overwrite=True)

    #    image.plot()
    #    plt.title('Calibrated Image', fontsize = 20)
    #    plt.gca().axes.get_xaxis().set_visible(False)
    #    plt.gca().axes.get_yaxis().set_visible(False)
    #    plt.tight_layout()
    #    plt.savefig(fname='Calibrated Image.png',
    #                transparent=True, frameon=False, bbox_inches='tight',
    #                pad_inches=None, dpi=300, labels=False)
    #    plt.close()

    # Get FFT of the image
    image_fft = image.fft(shift=True)
    # image_fft.plot()

    # Get the absolute value for viewing purposes
    # image_amp = image_fft.amplitude

    # image_amp.plot(norm='log')
    '''Plot the dataset'''
    # image.plot()
    # plt.close()
    # Get the sampling of the real and reciprocal space

    # Positive Gaussian
    arr = make_gaussian(size=len(image.data), fwhm=fwhm_pos_gaus, center=None)
    nD_Gaussian = hs.signals.Signal2D(np.array(arr))
    # nD_Gaussian.plot()
    # plt.close()

    # negative gauss
    arr_neg = make_gaussian(size=len(image.data),
                            fwhm=fwhm_neg_gaus, center=None)
    # Note that this step isn't actually neccessary for the computation,
    #   we could just subtract when making the double gaussian below.
    #   However, we do it this way so that we can save a plot of the negative
    # gaussian!
    # np_arr_neg = np_arr_neg
    nD_Gaussian_neg = hs.signals.Signal2D(np.array(arr_neg))
    # nD_Gaussian_neg.plot()

    neg_gauss_amplitude = 0.0
    int_and_gauss_array = []

    for neg_gauss_amplitude in np.arange(0, 1 + delta, delta):

        # while neg_gauss_amplitude <= 1:
        nD_Gaussian_neg_scaled = nD_Gaussian_neg * -1 * \
            neg_gauss_amplitude  # NEED TO FIGURE out best number here!
        # nD_Gaussian_neg.plot()
        # plt.close()

        # Double Gaussian
        DGFilter = nD_Gaussian + nD_Gaussian_neg_scaled
        # DGFilter.plot()
        # plt.close()

        '''
        # Remove background intensity and normalize
        DGFilter = normalize_signal(subtract_average_background(DGFilter))
        DGFilter.plot()
        '''
        # Multiply the 2-D Gaussian with the FFT. This low pass filters the
        # FFT.
        convolution = image_fft * DGFilter
        # convolution.plot(norm='log')
        # convolution_amp = convolution.amplitude
        # convolution_amp.plot(norm='log')

        # Create the inverse FFT, which is your filtered image!
        convolution_ifft = convolution.ifft()
        # convolution_ifft.plot()
        minimum_intensity = convolution_ifft.data.min()
        maximum_intensity = convolution_ifft.data.max()

        int_and_gauss_array.append(
            [neg_gauss_amplitude, minimum_intensity, maximum_intensity])

        # neg_gauss_amplitude = neg_gauss_amplitude + delta

    np_arr_2 = np.array(int_and_gauss_array)
    x_axis = np_arr_2[:, 0]
    y_axis = np_arr_2[:, 1]
    zero_line = np.zeros_like(x_axis)
    idx = np.argwhere(np.diff(np.sign(zero_line - y_axis))).flatten()
    neg_gauss_amplitude_calculated = x_axis[idx][0]

    ''' Filtering the Image with the Chosen Negative Amplitude '''
    # positive gauss
    nD_Gaussian.axes_manager[0].scale = reciprocal_sampling
    nD_Gaussian.axes_manager[1].scale = reciprocal_sampling
    nD_Gaussian.axes_manager[0].units = '1/' + units
    nD_Gaussian.axes_manager[1].units = '1/' + units

    # negative gauss
    nD_Gaussian_neg_used = nD_Gaussian_neg * -1 * \
        neg_gauss_amplitude_calculated  # NEED TO FIGURE out best number here!
    nD_Gaussian_neg_used.axes_manager[0].scale = reciprocal_sampling
    nD_Gaussian_neg_used.axes_manager[1].scale = reciprocal_sampling
    nD_Gaussian_neg_used.axes_manager[0].units = '1/' + units
    nD_Gaussian_neg_used.axes_manager[1].units = '1/' + units

    # Double Gaussian
    DGFilter_extra_dimension = nD_Gaussian + nD_Gaussian_neg_used
    DGFilter_extra_dimension.axes_manager[0].name = 'extra_dimension'

    '''how to change to just the 2 dimensiuons'''
    DGFilter = DGFilter_extra_dimension.sum('extra_dimension')

    DGFilter.axes_manager[0].scale = reciprocal_sampling
    DGFilter.axes_manager[1].scale = reciprocal_sampling
    DGFilter.axes_manager[0].units = '1/' + units
    DGFilter.axes_manager[1].units = '1/' + units

    # Multiply the 2-D Gaussian with the FFT. This filters the FFT.
    convolution = image_fft * DGFilter
    convolution_amp = convolution.amplitude

    # Create the inverse FFT, which is your filtered image!
    image_filtered = convolution.ifft()
    # s = normalize_signal(subtract_average_background(convolution_ifft))

    image_filtered.axes_manager[0].scale = real_space_sampling
    image_filtered.axes_manager[1].scale = real_space_sampling
    image_filtered.axes_manager[0].units = units
    image_filtered.axes_manager[1].units = units

    if filename is not None:
        plt.figure()
        plt.plot(x_axis, y_axis)
        plt.plot(x_axis, zero_line)
        plt.plot(x_axis[idx], y_axis[idx], 'ro')
        plt.xlabel('Negative Gaussian Amplitude', fontsize=16)
        plt.ylabel('Minimum Image Intensity', fontsize=16)
        plt.title('Finding the Best DG Filter \n NG Amp = %' +
                  filename % x_axis[idx][0], fontsize=20)
        plt.legend(labels=('Neg. Gauss. Amp.', 'y = 0',), fontsize=14)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.savefig(fname='minimising_negative_gaussian_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300)
        plt.close()

        #    if filename is not None:
        nD_Gaussian_neg_used.save(
            'negative_gaussian_' + filename, overwrite=True)
        nD_Gaussian_neg_used.plot()
        plt.title('Negative Gaussian ' + filename, fontsize=20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname='negative_gaussian_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)
        plt.close()

        nD_Gaussian.save('positive_gaussian_' + filename, overwrite=True)
        nD_Gaussian.plot()
        plt.title('Positive Gaussian ' + filename, fontsize=20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname='positive_gaussian_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)
        plt.close()

        DGFilter.save('double_gaussian_filter_' + filename,
                      overwrite=True)  # Save the .hspy file

        DGFilter.plot()
        plt.title('Double Gaussian Filter ' + filename, fontsize=20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname='double_gaussian_filter_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)
        plt.close()

        convolution_amp.plot(norm='log')
        plt.title('FFT and Filter Convolved ' + filename, fontsize=20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname='FFT_and_filter_convolved_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=600, labels=False)
        plt.close()

        image_filtered.save('filtered_image_' + filename, overwrite=True)
        image_filtered.plot()
        plt.title('DG Filtered Image ' + filename, fontsize=20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname='DG_filtered_image_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=600, labels=False)
        plt.close()

        ''' Saving the Variables for the image and filtered Image '''
        Filtering_Variables = collections.OrderedDict()
        Filtering_Variables['filename'] = [filename]
        Filtering_Variables['Image Size (nm)'] = [physical_image_size]
        Filtering_Variables['Image Size (pix)'] = [len(image.data)]
        Filtering_Variables['Real Space Sampling (nm/pix)'] = [
            real_space_sampling]
        Filtering_Variables['Reciprocal Space Sampling (1/nm/pix)'] = [
            reciprocal_sampling]
        Filtering_Variables['First Diffraction Ring (Diameter) (1/nm)'] = [
            d_inner]
        Filtering_Variables['Second Diffraction Ring (Diameter) (1/nm)'] = [
            d_outer]
        Filtering_Variables['First Diffraction Ring (Radius) (1/nm)'] = [
            reciprocal_d_inner]
        Filtering_Variables['Second Diffraction Ring (Radius) (1/nm)'] = [
            reciprocal_d_outer]
        Filtering_Variables['First Diffraction Ring (Radius) (pix)'] = [
            reciprocal_d_inner_pix]
        Filtering_Variables['Second Diffraction Ring (Radius) (pix)'] = [
            reciprocal_d_outer_pix]
        Filtering_Variables['Positive Gaussian FWHM (pix)'] = [fwhm_pos_gaus]
        Filtering_Variables['Negative Gaussian FWHM (pix)'] = [fwhm_neg_gaus]
        Filtering_Variables['Positive Gaussian Size'] = [len(image.data)]
        Filtering_Variables['Negative Gaussian Size'] = [len(image.data)]
        Filtering_Variables['Positive Gaussian FWHM'] = [fwhm_pos_gaus]
        Filtering_Variables['Negative Gaussian FWHM'] = [fwhm_neg_gaus]
        Filtering_Variables['Negative Gaussian Amplitude'] = [
            neg_gauss_amplitude_calculated]
        Filtering_Variables['Delta used for Calculation'] = [delta]
        Filtering_Variables_Table = pd.DataFrame(Filtering_Variables)
        Filtering_Variables_Table
        Filtering_Variables_Table.to_pickle(
            'filtering_variables_table_' + filename + '.pkl')
        # Filtering_Variables_Table.to_csv('Filtering_Variables_Table.csv',
        # sep=',', index=False)

    return(image_filtered)


'''
Cropping and Calibrating
'''

# cropping done in the scale, so nm, pixel, or whatever you have
# cropping_area = choose_points_on_image(image.data)


def crop_image_hs(image, cropping_area, save_image=True, save_variables=True,
                  scalebar_true=True):
    '''
    Example
    -------

    >>> image = am.dummy_data.get_simple_cubic_with_vacancies_signal()
    >>> image.plot()
    >>> cropping_area = choose_points_on_image(image.data) # choose two points
    '''

    llim, tlim = cropping_area[0]
    rlim, blim = cropping_area[1]

    unit = image.axes_manager[0].units
#    image_name = image.metadata.General.original_filename

    if image.axes_manager[0].scale != image.axes_manager[1].scale:
        raise ValueError("x & y scales don't match!")

    if scalebar_true is True:
        llim *= image.axes_manager[0].scale
        tlim *= image.axes_manager[0].scale
        rlim *= image.axes_manager[0].scale
        blim *= image.axes_manager[0].scale
    else:
        pass

    roi = hs.roi.RectangularROI(left=llim, right=rlim, top=tlim, bottom=blim)
    image.plot()
    image_crop = roi.interactive(image)

    if save_image is True:
        plt.title('Cropped region highlighted', fontsize=20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname='Cropped region highlighted.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)
        plt.close()
    else:
        plt.close()

    image_crop.plot()

    image_crop
    physical_image_crop_size_x = image_crop.axes_manager[0].scale * \
        image_crop.axes_manager[0].size
    physical_image_crop_size_y = image_crop.axes_manager[1].scale * \
        image_crop.axes_manager[1].size

    if save_image is True:
        image_crop.save('Cropped Image.hspy')
        image_crop.plot()
        plt.title('Cropped Image', fontsize=20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname='Cropped Image.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)
        plt.close()
    else:
        plt.close()

    if save_variables is True:
        ''' Saving the Variables for the image and filtered Image '''
        Cropping_Variables = collections.OrderedDict()
#        Cropping_Variables['Image Name'] = [image_name]
        Cropping_Variables['left (%s)' % unit] = [llim]
        Cropping_Variables['right (%s)' % unit] = [rlim]
        Cropping_Variables['top (%s)' % unit] = [tlim]
        Cropping_Variables['bottom (%s)' % unit] = [blim]
        Cropping_Variables['physical_image_size X axis (%s)' % unit] = [
            physical_image_crop_size_x]
        Cropping_Variables['physical_image_size Y axis (%s)' % unit] = [
            physical_image_crop_size_y]
        Cropping_Variables['Unit'] = [unit]
        Cropping_Variables_Table = pd.DataFrame(Cropping_Variables)
        Cropping_Variables_Table
        Cropping_Variables_Table.to_pickle('Cropping_Variables_Table.pkl')
        Cropping_Variables_Table.to_csv(
            'Cropping_Variables_Table.csv', sep=',', index=False)

    else:
        pass

    return image_crop


# cropping_area = choose_points_on_image(image.data)


def calibrate_intensity_distance_with_sublattice_roi(image,
                                                     cropping_area,
                                                     separation,
                                                     filename=None,
                                                     reference_image=None,
                                                     percent_to_nn=0.2,
                                                     mask_radius=None,
                                                     refine=True,
                                                     scalebar_true=False):
    # add max mean min etc.
    '''
    Calibrates the intensity of an image by using a sublattice, found with some
    atomap functions. The mean intensity of that sublattice is set to 1

    Parameters
    ----------
    image : HyperSpy 2D signal, default None
        The signal can be distance calibrated. If it is, set
        scalebar_true=True
    cropping_area : list of 2 floats, default None
        The best method of choosing the area is by using the function
        "choose_points_on_image(image.data)". Choose two points on the
        image. First point is top left of area, second point is bottom right.
    percent_to_nn : float, default 0.40
        Determines the boundary of the area surrounding each atomic
        column, as fraction of the distance to the nearest neighbour.
    scalebar_true : Bool, default False
        Set to True if the scale of the image is calibrated to a distance unit.
        *** is there any point to this? if scale=1, then multiplying has no
        *** effect, and if it is scaled to nm or angstrom, multiplying is
        *** good. so keep the code, remove the parameter option!

    Returns
    -------
    calibrated image data

    Example
    -------

    >>> image = am.dummy_data.get_simple_cubic_with_vacancies_signal()
    >>> # image.plot()
    >>> cropping_area = [[10,10],[100,100]]
    >>> # cropping_area = choose_points_on_image(image.data) # manually
    >>> calibrate_intensity_distance_with_sublattice_roi(image,
    ...             cropping_area, separation=10)
    >>> # image.plot()

    '''
    llim, tlim = cropping_area[0]
    rlim, blim = cropping_area[1]

    if image.axes_manager[0].scale != image.axes_manager[1].scale:
        raise ValueError("x & y scales don't match!")

    if scalebar_true is True:
        llim *= image.axes_manager[0].scale
        tlim *= image.axes_manager[0].scale
        rlim *= image.axes_manager[0].scale
        blim *= image.axes_manager[0].scale
    else:
        pass

    cal_area = hs.roi.RectangularROI(
        left=llim, right=rlim, top=tlim, bottom=blim)(image)
    atom_positions = am.get_atom_positions(
        cal_area, separation=separation, pca=True)
    # atom_positions = am.add_atoms_with_gui(cal_area, atom_positions)
    calib_sub = am.Sublattice(atom_positions, cal_area, color='r')
    # calib_sub.plot()
    if refine is True:
        calib_sub.find_nearest_neighbors()
        calib_sub.refine_atom_positions_using_center_of_mass(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius,
            show_progressbar=False)
        calib_sub.refine_atom_positions_using_2d_gaussian(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius,
            show_progressbar=False)
    else:
        pass
    # calib_sub.plot()
    calib_sub.get_atom_column_amplitude_max_intensity(
        percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    calib_sub_max_list = calib_sub.atom_amplitude_max_intensity
    calib_sub_scalar = mean(a=calib_sub_max_list)
    image.data = image.data / calib_sub_scalar

    if reference_image is not None:
        image.axes_manager = reference_image.axes_manager

    if filename is not None:
        save_name = 'calibrated_data_'
        image.save(save_name + filename, overwrite=True)
        image.plot()
        plt.title(save_name + filename, fontsize=20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname=save_name + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)
        # plt.close()


'''
Atomap extensions
'''


def toggle_atom_refine_position_automatically(sublattice,
                                              min_cut_off_percent,
                                              max_cut_off_percent,
                                              filename=None,
                                              range_type='internal',
                                              method='mode',
                                              percent_to_nn=0.05,
                                              mask_radius=None):
    '''
    Sets the 'refine_position' attribute of each Atom Position in a
    sublattice using a range of intensities.

    Parameters
    ----------
    sublattice : Atomap Sublattice object, default None

    min_cut_off_percent : float, default None
        The lower end of the intensity range is defined as
        min_cut_off_percent * modal value of max intensity list of
        sublattice.
    max_cut_off_percent : float, default None
        The upper end of the intensity range is defined as
        max_cut_off_percent * modal value of max intensity list of
        sublattice.
    range_type : string, default 'internal'
        'internal' returns the 'refine_position' attribute for each
        Atom Position as True if the intensity of that Atom Position
        lies between the lower and upper limits defined by min_cut_off_percent
        and max_cut_off_percent.
        'external' returns the 'refine_position' attribute for each
        Atom Position as True if the intensity of that Atom Position
        lies outside the lower and upper limits defined by min_cut_off_percent
        and max_cut_off_percent.
    save_image : Bool, default False
        Save the 'sublattice.toggle_atom_refine_position_with_gui()'
        image.
    percent_to_nn : float, default 0.40
        Determines the boundary of the area surrounding each atomic
        column, as fraction of the distance to the nearest neighbour.

    Returns
    -------
    calibrated image data
    Example
    -------

    >>> min_cut_off_percent = 0.75
    >>> max_cut_off_percent = 1.25
    >>> sublattice = am.dummy_data.get_simple_cubic_with_vacancies_sublattice(
    ...     image_noise=True)
    >>> sublattice.find_nearest_neighbors()
    >>> sublattice.plot()
    >>> false_list_sublattice =  toggle_atom_refine_position_automatically(
    ...                             sublattice=sublattice,
    ...                             min_cut_off_percent=min_cut_off_percent,
    ...                             max_cut_off_percent=max_cut_off_percent,
    ...                             range_type='internal',
    ...                             method='mode',
    ...                             percent_to_nn=0.05)

    >>> # Check which atoms will not be refined (red dots)
    >>> sublattice.toggle_atom_refine_position_with_gui()
    '''

    sublattice.get_atom_column_amplitude_max_intensity(
        percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    sublattice_vacancy_check_list = sublattice.atom_amplitude_max_intensity

    if method == 'mode':
        sublattice_scalar = scipy.stats.mode(
            np.round(a=sublattice_vacancy_check_list, decimals=2))[0]
    elif method == 'mean':
        sublattice_scalar = np.mean(sublattice_vacancy_check_list)

    sublattice_min_cut_off = min_cut_off_percent * sublattice_scalar
    sublattice_max_cut_off = max_cut_off_percent * sublattice_scalar

    if range_type == 'internal':

        for i in range(0, len(sublattice.atom_list)):
            if sublattice_min_cut_off < \
                sublattice.atom_amplitude_max_intensity[
                    i] < sublattice_max_cut_off:
                sublattice.atom_list[i].refine_position = True
            else:
                sublattice.atom_list[i].refine_position = False

    elif range_type == 'external':
        for i in range(0, len(sublattice.atom_list)):
            if sublattice.atom_amplitude_max_intensity[
                i] > sublattice_max_cut_off or sublattice_min_cut_off > \
                    sublattice.atom_amplitude_max_intensity[i]:
                sublattice.atom_list[i].refine_position = True
            else:
                sublattice.atom_list[i].refine_position = False

    else:
        raise TypeError(
            "'internal' and 'external' are the only options for range_type")

    # checking we have some falses
    false_list_sublattice = []
    for i in range(0, len(sublattice.atom_list)):
        if sublattice.atom_list[i].refine_position is False:
            false_list_sublattice.append(
                sublattice.atom_list[i].refine_position)

    if len(false_list_sublattice) == 0:
        print("false_list_sublattice is empty")

    if filename is not None:
        sublattice.toggle_atom_refine_position_with_gui()
        plt.title('Toggle Atom Refine ' + sublattice.name +
                  ' ' + filename + '\n Red=False', fontsize=20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(
            fname='toggle_atom_refine_' + sublattice.name + '_' + filename +
                  '.png',
            transparent=True, frameon=False, bbox_inches='tight',
            pad_inches=None, dpi=300, labels=False)
        plt.close()

    return(false_list_sublattice)


def remove_image_intensity_in_data_slice(atom,
                                         image_data,
                                         percent_to_nn=0.50):
    """
    Remove intensity from the area around an atom in a
    sublattice
    """
    closest_neighbor = atom.get_closest_neighbor()

    slice_size = closest_neighbor * percent_to_nn * 2
    _remove_image_slice_around_atom(atom,
                                    image_data, slice_size)


def _remove_image_slice_around_atom(
        self,
        image_data,
        slice_size):
    """
    Return a square slice of the image data.

    The atom is in the center of this slice.

    Parameters
    ----------
    image_data : Numpy 2D array
    slice_size : int
        Width and height of the square slice

    Returns
    -------
    2D numpy array

    """
    x0 = self.pixel_x - slice_size / 2
    x1 = self.pixel_x + slice_size / 2
    y0 = self.pixel_y - slice_size / 2
    y1 = self.pixel_y + slice_size / 2

    if x0 < 0.0:
        x0 = 0
    if y0 < 0.0:
        y0 = 0
    if x1 > image_data.shape[1]:
        x1 = image_data.shape[1]
    if y1 > image_data.shape[0]:
        y1 = image_data.shape[0]
    x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
    data_slice = copy.deepcopy(image_data[y0:y1, x0:x1])

    data_slice_max = data_slice.max()

    image_data[y0:y1, x0:x1] = image_data[y0:y1, x0:x1] - \
        data_slice_max


def get_cell_image(s, points_x, points_y, method='Voronoi', max_radius='Auto',
                   reduce_func=np.min,
                   show_progressbar=True):
    '''
    The same as atomap's integrate, except instead of summing the
    region around an atom, it removes the value from all pixels in
    that region.
    For example, with this you can remove the local background intensity
    by setting reduce_func=np.min.

    Parameters
    ----------
    See atomap's integrate function.

    reduce_func : ufunc, default np.min
        function used to reduce the pixel values around each atom
        to a float.

    Examples
    --------
    #### add PTO example from paper

    Returns
    -------

    Numpy array with the same shape as s
    '''
    image = s.__array__()
    if len(image.shape) < 2:
        raise ValueError("s must have at least 2 dimensions")
    intensity_record = np.zeros_like(image, dtype=float)
    currentFeature = np.zeros_like(image.T, dtype=float)
    point_record = np.zeros(image.shape[0:2][::-1], dtype=int)
    integrated_intensity = np.zeros_like(sum(sum(currentFeature.T)))
    integrated_intensity = np.dstack(
        integrated_intensity for i in range(len(points_x)))
    integrated_intensity = np.squeeze(integrated_intensity.T)
    points = np.array((points_y, points_x))
    # Setting max_radius to the width of the image, if none is set.
    if method == 'Voronoi':
        if max_radius == 'Auto':
            max_radius = max(point_record.shape)
        elif max_radius <= 0:
            raise ValueError("max_radius must be higher than 0.")
        distance_log = np.zeros_like(points[0])

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):

                # For every pixel the distance to all points must be
                # calculated.
                distance_log = ((points[0] - float(i))**2 +
                                (points[1] - float(j))**2)**0.5

                # Next for that pixel the minimum distance to and point should
                # be checked and discarded if too large:
                distMin = np.min(distance_log)
                minIndex = np.argmin(distance_log)

                if distMin >= max_radius:
                    point_record[j][i] = 0
                else:
                    point_record[j][i] = minIndex + 1

    else:
        raise NotImplementedError(
            "Oops! You have asked for an unimplemented method.")
    point_record -= 1

    for point in trange(points[0].shape[0], desc='Integrating',
                        disable=not show_progressbar):
        currentMask = (point_record == point)
        currentFeature = currentMask * image.T
        # integrated_intensity[point] = sum(sum(currentFeature.T)).T
        # my shite
        # remove zeros from array (needed for np.mean, np.min)
        integrated_intensity[point] = reduce_func(
            currentFeature[currentFeature != 0])

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if currentMask.T[i][j]:
                    intensity_record[i][j] = integrated_intensity[point]

    # return (integrated_intensity, s_intensity_record, point_record.T)
    return(intensity_record)


def distance_vector(x1, y1, x2, y2):
    distance_vector = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))
    return(distance_vector)


def mean_and_std_nearest_neighbour_distances(sublattice,
                                             nearest_neighbours=5,
                                             sampling=None):
    '''
    Calculates mean and standard deviation of the distance from each atom to
    its nearest neighbours.

    Parameters
    ----------
    sublattice : Atomap Sublattice object
    nearest_neighbours : int, default 5
        The number of nearest neighbours used to calculate the mean distance
        from an atom. As in atomap, choosing 5 gets the 4 nearest
        neighbours.

    Returns
    -------
    2 lists: list of mean distances, list of standard deviations.

    Examples
    --------
    >>> from temul.dummy_data import get_simple_cubic_sublattice
    >>> import temul.signal_processing as tmlsp
    >>> sub1 = get_simple_cubic_sublattice()
    >>> mean, std = tmlsp.mean_and_std_nearest_neighbour_distances(sub1)
    >>> mean_scaled,_ = tmlsp.mean_and_std_nearest_neighbour_distances(sub1,
    ...     nearest_neighbours=5,
    ...     sampling=0.0123)

    '''
    # 5 will get the nearest 4 atoms
    sublattice.find_nearest_neighbors(
        nearest_neighbors=nearest_neighbours)

    # atom_nn_list = []
    mean_list = []
    std_dev_list = []
    # variance_list = []
    for atom in sublattice.atom_list:
        atom_nns = atom.nearest_neighbor_list
        x1 = atom.pixel_x
        y1 = atom.pixel_y

        distance_list = []
        for i in range(0, len(atom_nns)):
            x2 = atom_nns[i].pixel_x
            y2 = atom_nns[i].pixel_y
            distance = distance_vector(x1, y1, x2, y2)
            distance_list.append(distance)

        mean_distance = sum(distance_list) / len(distance_list)
        mean_list.append(mean_distance)
        std_dev = np.std(distance_list, dtype=np.float64)
        std_dev_list.append(std_dev)
        # variance = np.var(distance_list, dtype=np.float64)
        # variance_list.append(variance)

    if sampling is not None:
        mean_list = [k * sampling for k in mean_list]
        std_dev_list = [k * sampling for k in std_dev_list]

    return(mean_list, std_dev_list)


def choose_mask_coordinates(image, norm='log'):
    '''
    Pick the mask locations for an FFT. See get_masked_ifft() and
    commit 5ba307b5af0b598bedc0284aa989d44e23fdde4d on Atomap

    Parameters
    ----------
    image : Hyperspy 2D Signal
    norm : string, default 'log'
        How to scale the intensity value for the displayed image.
        Options are 'linear' and 'log'

    Returns
    -------
    mask_coords : list of pixel coordinates

    Examples
    --------
    See get_masked_ifft() for example.
    '''

    fft = image.fft(shift=True)
    fft_amp = fft.amplitude

    mask_coords = choose_points_on_image(
        fft_amp.data, norm=norm)

    return(mask_coords)


def get_masked_ifft(image, mask_coords, mask_radius=10, space="real",
                    keep_masked_area=True, plot_masked_fft=False):
    '''
    loop through each mask_coords and mask the fft. Then return
    an ifft of the image.
    To Do: calibration of units automatically.

    Masks a fast Fourier transform (FFT) and returns the inverse FFT.
    Use choose_mask_coordinates() to manually choose mask coordinates in the
    FFT.

    Parameters
    ----------
    image : Hyperspy 2D Signal
    mask_coords : list of lists
        Pixel coordinates of the masking locations. See the example below for
        two simple coordinates.
    mask_radius : int, default 10
        Radius in pixels of the mask.
    space : string, default "real"
        If the input image is a Fourier transform already, set space="fourier"
    keep_masked_area : Bool, default True
        If True, this will set the mask at the mask_coords.
        If False, this will set the mask as everything other than the
        mask_coords.
    plot_masked_fft : Bool, default False
        If True, the mask used to filter the FFT will be plotted. Good for
        checking that the mask is doing what you want.

    Returns
    -------
    Inverse FFT as a Hyperspy 2D Signal

    Examples
    --------
    >>> from temul.dummy_data import get_simple_cubic_signal
    >>> import temul.signal_processing as tmlsp
    >>> image = get_simple_cubic_signal()
    >>> mask_coords = [[170.2, 170.8],[129.8, 130]]
    >>> # mask_coords = tmlsp.choose_mask_coordinates(image=image, norm='log')

    Use the defaults:

    >>> image_ifft = tmlsp.get_masked_ifft(
    ...     image=image, mask_coords=mask_coords)
    >>> image_ifft.plot()

    Plot the masked fft:

    >>> image_ifft = tmlsp.get_masked_ifft(
    ...     image=image, mask_coords=mask_coords,
    ...     plot_masked_fft=True)
    >>> image_ifft.plot()

    Use unmasked fft area and plot the masked fft:

    >>> image_ifft = tmlsp.get_masked_ifft(
    ...     image=image, mask_coords=mask_coords,
    ...     plot_masked_fft=True, keep_masked_area=False)
    >>> image_ifft.plot()

    If the input image is already a Fourier transform:

    >>> fft_image = image.fft(shift=True)
    >>> image_ifft = tmlsp.get_masked_ifft(
    ...     image=fft_image, mask_coords=mask_coords,
    ...     space='fourier')
    >>> image_ifft.plot()

    '''
    if space == 'real':
        fft = image.fft(shift=True)
    elif space == 'fourier':
        fft = image

    for mask_coord in mask_coords:

        x_pix = mask_coord[0]
        y_pix = mask_coord[1]

        # Create a mask over that location and transpose the axes
        mask = _make_circular_mask(centerX=x_pix, centerY=y_pix,
                                   imageSizeX=image.data.shape[1],
                                   imageSizeY=image.data.shape[0],
                                   radius=mask_radius).T

        # check if the combined masked_fft exists
        try:
            mask_combined
        except NameError:
            mask_combined = mask
        else:
            mask_combined += mask

    # the tilda ~ inverses the mask
    # fill in nothings with 0
    if keep_masked_area:
        masked_fft = np.ma.array(fft.data, mask=~mask_combined).filled(0)
    elif not keep_masked_area:
        masked_fft = np.ma.array(fft.data, mask=mask_combined).filled(0)

    masked_fft_image = hs.signals.ComplexSignal2D(masked_fft)
    if plot_masked_fft:
        masked_fft_image.amplitude.plot(norm="log")
    # sort out units here
    image_ifft = masked_fft_image.ifft()
    image_ifft = np.absolute(image_ifft)

    return(image_ifft)
