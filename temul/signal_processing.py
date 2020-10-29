
from temul.element_tools import split_and_sort_element
from temul.signal_plotting import choose_points_on_image

from temul.external.atomap_devel_012.sublattice import Sublattice
from temul.external.atomap_devel_012.atom_finding_refining import (
    get_atom_positions, _make_circular_mask)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

from skimage.metrics import structural_similarity as ssm
import scipy
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import OptimizeWarning
from math import sqrt
import numpy as np
from numpy import mean
import hyperspy.api as hs
from hyperspy.roi import RectangularROI
from hyperspy._signals.complex_signal2d import ComplexSignal2D
from hyperspy._signals.signal2d import Signal2D
import pandas as pd
import copy
from tqdm import trange
import collections
import warnings
from matplotlib.widgets import Slider, Button
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

    Examples
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
    '''
    Measure the mean squared error between two images of the same shape.

    Parameters
    ----------
    imageA, imageB : array-like
        The images must be the same shape.

    Returns
    -------
    Mean squared error

    '''
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
    filename : str, default None
        name with which the image will be saved

    Returns
    -------
    two floats (mse_number, ssm_number)

    Examples
    -------
    >>> from temul.dummy_data import get_simple_cubic_signal
    >>> imageA = get_simple_cubic_signal().data
    >>> imageB = get_simple_cubic_signal(image_noise=True).data
    >>> mse_number, ssm_number = measure_image_errors(imageA, imageB)

    Showing the ideal case of both images being exactly equal:

    >>> imageB = imageA
    >>> mse_number, ssm_number = measure_image_errors(imageA, imageA)
    >>> print("MSE: {} and SSM: {}".format(mse_number, ssm_number))
    MSE: 0.0 and SSM: 1.0

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


def load_and_compare_images(imageA, imageB, filename=None):
    '''
    Load two images with hyperspy and compare their mean square error and
    structural simularity index.

    Parameters
    ----------
    imageA, imageB : str, path to file
        filename of the images to be loaded and compared
    filename : str, default None
        name with which the image will be saved

    Returns
    -------
    Two floats (mean standard error and structural simularity index)

    '''
    imageA = hs.load(imageA)
    imageB = hs.load(imageB)

    mse_number, ssm_number = measure_image_errors(
        imageA,
        imageB,
        filename=filename)

    return(mse_number, ssm_number)


def compare_two_image_and_create_filtered_image(
        image_to_filter, reference_image, delta_image_filter, max_sigma=6,
        cropping_area=[[0, 0], [50, 50]], separation=8, filename=None,
        percent_to_nn=0.4, mask_radius=None, refine=False):
    '''
    Gaussian blur an image for comparison with a reference image.
    Good for finding the best gaussian blur for a simulation by
    comparing to an experimental image.
    See measure_image_errors() and load_and_compare_images().

    Parameters
    ----------
    image_to_filter : Hyperspy Signal2D
        Image you wish to automatically filter.
    reference_image : Hyperspy Signal2D
        Image with which `image_to_filter` is compared.
    delta_image_filter : float
        The increment of the Gaussian sigma used.
    max_sigma : float, default 6
        The largest (limiting) Gaussian sigma used.
    cropping_area : list of 2 floats, default [[0,0], [50,50]]
        The best method of choosing the area is by using the function
        "choose_points_on_image(image.data)". Choose two points on the
        image. First point is top left of area, second point is bottom right.
    separation : int, default 8
        Pixel separation between atoms as used by Atomap.
    filename : str, default None
        If set to a string, the plotted and filtered image will be saved.
    percent_to_nn : float, default 0.4
        Determines the boundary of the area surrounding each atomic
        column, as fraction of the distance to the nearest neighbour.
    mask_radius : int, default None
        Radius in pixels of the mask. If set, then set `percent_to_nn=None`.
    refine : Bool, default False
        If set to True, the `calibrate_intensity_distance_with_sublattice_roi`
        calibration will refine the atom positions for each calibration. May
        make the function very slow depending on the size of `image_to_filter`
        and `cropping_area`.

    Returns
    -------
    Hyperspy Signal2D (filtered image) and float (ideal Gaussian sigma).

    Examples
    --------
    >>> from scipy.ndimage.filters import gaussian_filter
    >>> import temul.example_data as example_data
    >>> import matplotlib.pyplot as plt
    >>> from temul.signal_processing import (
    ...     compare_two_image_and_create_filtered_image)
    >>> experiment = example_data.load_Se_implanted_MoS2_data() # example
    >>> experiment.data = gaussian_filter(experiment.data, sigma=4)
    >>> simulation = example_data.load_Se_implanted_MoS2_data()

    filt_image, ideal_sigma = compare_two_image_and_create_filtered_image(
         simulation, experiment, 0.25, cropping_area=[[5,5], [200, 200]],
         separation=11, mask_radius=4, percent_to_nn=None, max_sigma=10)

    '''

    image_to_filter_data = image_to_filter.data
    reference_image_data = reference_image.data

    mse_number_list = []
    ssm_number_list = []

    for i in np.arange(0, max_sigma + delta_image_filter, delta_image_filter):

        image_to_filter_data_filtered = gaussian_filter(image_to_filter_data,
                                                        sigma=i)
        temp_image_filtered = Signal2D(
            image_to_filter_data_filtered)

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

    image_filtered = Signal2D(image_to_filter_filtered)

    # calibrate_intensity_distance_with_sublattice_roi(
    #     image=image_filtered,
    #     cropping_area=cropping_area,
    #     separation=separation,
    #     percent_to_nn=percent_to_nn,
    #     mask_radius=mask_radius,
    #     refine=refine,
    #     filename=None)

    plt.figure()
    plt.scatter(x=ssm_indexing, y=ssm, label='ssm',
                marker='x', color='magenta')
    plt.scatter(x=mse_indexing, y=mse, label='mse', marker='o', color='b')
    plt.scatter(x=ideal_sigma, y=ideal_sigma_y_coord, label='\u03C3 = ' +
                str(round(ideal_sigma, 2)), marker='D', color='k')
    plt.title("MSE & SSM vs. Gauss Blur", fontsize=20)
    plt.xlabel("\u03C3 (Gaussian Blur)", fontsize=16)
    plt.ylabel("MSE (0) and SSM (1)", fontsize=16)
    plt.legend()
    plt.tight_layout
    plt.show()

    if filename is not None:
        plt.savefig(fname='MSE_SSM_gaussian_blur_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)

    return(image_filtered, ideal_sigma)


def make_gaussian(size, fwhm, center=None):
    """ Make a square gaussian kernel.

    Parameters
    ----------
    size : int
        The length of a side of the square
    fwhm : float
        The full-width-half-maximum of the Gaussian, which can be thought of as
        an effective radius.
    center : array, default None
        The location of the center of the Gaussian. None will set it to the
        center of the array.

    Returns
    -------
    2D Numpy array

    Examples
    --------
    >>> from temul.signal_processing import make_gaussian
    >>> import matplotlib.pyplot as plt
    >>> array = make_gaussian(15, 5)
    >>> im = plt.imshow(array)

    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    arr = np.array((np.exp(-4 * np.log(2) * ((x - x0)**2 +
                                             (y - y0)**2) / fwhm**2)))

    return(arr)


def make_gaussian_pos_neg(size, fwhm_neg, fwhm_pos, neg_min=0.9, center=None):
    ''' See double_gaussian_fft_filter for details '''
    arr_pos = make_gaussian(size, fwhm=fwhm_pos, center=center)
    nD_Gaussian_pos = Signal2D(arr_pos)

    arr_neg = make_gaussian(size, fwhm=fwhm_neg, center=center)
    nD_Gaussian_neg = Signal2D(arr_neg) * -1 * neg_min

    return(nD_Gaussian_pos, nD_Gaussian_neg)


def double_gaussian_fft_filter(image, fwhm_neg, fwhm_pos, neg_min=0.9):
    '''
    Filter an image with a bandpass-like filter.

    Parameters
    ----------
    image : Hyperspy Signal2D
    fwhm_neg, fwhm_pos : float
        Initial guess in pixels of full width at half maximum (fwhm) of
        inner (negative) and outer (positive) Gaussian to be applied to fft,
        respectively.
        Use the visualise_dg_filter function to find the optimium values.
    neg_min : float, default 0.9
        Effective amplitude of the negative Gaussian.

    Examples
    --------
    >>> import temul.signal_processing as tmlsig
    >>> from temul.example_data import load_Se_implanted_MoS2_data
    >>> image = load_Se_implanted_MoS2_data()

    Use the visualise_dg_filter to find suitable FWHMs

    >>> tmlsig.visualise_dg_filter(image)

    then use these values to carry out the double_gaussian_fft_filter

    >>> filtered_image = tmlsig.double_gaussian_fft_filter(image, 50, 150)
    >>> image.plot()
    >>> filtered_image.plot()

    '''
    image_fft = image.fft(shift=True)
    fft_data = image_fft.amplitude.data

    nD_Gaussian_pos, nD_Gaussian_neg = make_gaussian_pos_neg(
        fft_data.shape[-1], fwhm_neg, fwhm_pos, neg_min)
    dg_filter = nD_Gaussian_pos + nD_Gaussian_neg

    convolution = image_fft * dg_filter
    convolution_ifft = convolution.ifft()
    convolution_ifft.axes_manager = image.axes_manager
    convolution_ifft.metadata.General.title = "Filtered Image"
    return convolution_ifft


def double_gaussian_fft_filter_optimised(image, d_inner, d_outer, delta=0.05,
                                         sampling=None, units=None,
                                         filename=None):
    '''
    Filter an image with an double Gaussian (band-pass) filter. The function
    will automatically find the optimum magnitude of the negative inner
    Gaussian.

    Parameters
    ----------
    image : Hyperspy Signal2D
        Image to be filtered.
    d_inner : float
        Inner diameter of the FFT spots. Effectively the diameter of the
        negative Gaussian.
    d_outer : float
        Outer diameter of the FFT spots. Effectively the diameter of the
        positive Gaussian.
    delta : float, default 0.05
        Increment of the automatic filtering with the negative Gaussian.
        Setting this very small will slow down the function, but too high will
        not allow the function to calculate negative Gaussians near zero.
    sampling : float
        image sampling in units/pixel. If set to None, the image.axes_manager
        will be used.
    units : str
        Real space units. `sampling` should then be the value of
        these units/pixel. If set to None, the image.axes_manager
        will be used.
    filename : str, default None
        If set to a string, the following files will be plotted and saved:
        negative Gaussian optimumisation, negative Gaussian, positive Gaussian,
        double Gaussian, FFT and double Gaussian convolution, filtered image,
        filtered variables table.

    Returns
    -------
    Hyperspy Signal2D

    Examples
    --------
    >>> import temul.example_data as example_data
    >>> from temul.signal_processing import (
    ...     double_gaussian_fft_filter)
    >>> experiment = example_data.load_Se_implanted_MoS2_data()
    >>> experiment.plot()
    >>> filtered_image = double_gaussian_fft_filter(experiment, 7.48, 14.96)
    >>> filtered_image.plot()

    '''

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
    # Example d_inner, d_outer:
    # MoS2: d_1 = 7.7, d_2 = 14

    if sampling is None:
        sampling = image.axes_manager[-1].scale
    else:
        image.axes_manager[0].scale = sampling
        image.axes_manager[1].scale = sampling

    if units is None:
        units = image.axes_manager[-1].units
    else:
        image.axes_manager[0].units = units
        image.axes_manager[1].units = units

    physical_image_size = sampling * len(image.data)
    reciprocal_sampling = 1 / physical_image_size

    # Get radius
    reciprocal_d_inner = (d_inner / 2)
    reciprocal_d_outer = (d_outer / 2)
    reciprocal_d_inner_pix = reciprocal_d_inner / reciprocal_sampling
    reciprocal_d_outer_pix = reciprocal_d_outer / reciprocal_sampling

    fwhm_neg_gaus = reciprocal_d_inner_pix
    fwhm_pos_gaus = reciprocal_d_outer_pix

    # Get FFT of the image
    image_fft = image.fft(shift=True)

    nD_Gaussian, nD_Gaussian_neg = make_gaussian_pos_neg(
        len(image.data), fwhm_pos_gaus, fwhm_neg_gaus, 1, center=None)

    neg_gauss_amplitude = 0.0
    int_and_gauss_array = []
    for neg_gauss_amplitude in np.arange(0, 1 + delta, delta):

        # while neg_gauss_amplitude <= 1:
        nD_Gaussian_neg_scaled = nD_Gaussian_neg * \
            neg_gauss_amplitude  # NEED TO FIGURE out best number here!

        # Double Gaussian
        DGFilter = nD_Gaussian + nD_Gaussian_neg_scaled

        # Multiply the 2-D Gaussian with the FFT. This low pass filters the
        # FFT.
        convolution = image_fft * DGFilter

        # Create the inverse FFT, which is your filtered image!
        convolution_ifft = convolution.ifft()
        # convolution_ifft.plot()
        minimum_intensity = convolution_ifft.data.min()
        maximum_intensity = convolution_ifft.data.max()

        int_and_gauss_array.append(
            [neg_gauss_amplitude, minimum_intensity, maximum_intensity])

        # neg_gauss_amplitude = neg_gauss_amplitude + delta

    int_and_gauss_array = np.array(int_and_gauss_array)
    x_axis = int_and_gauss_array[:, 0]
    y_axis = int_and_gauss_array[:, 1]
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

    image_filtered.axes_manager[0].scale = sampling
    image_filtered.axes_manager[1].scale = sampling
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
            sampling]
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
        Filtering_Variables_Table.to_csv('Filtering_Variables_Table.csv',
                                         sep=',', index=False)

    return(image_filtered)


def visualise_dg_filter(image, d_inner=7.7, d_outer=21, slider_min=0.1,
                        slider_max=300, slider_step=0.1, plot_lims=(0, 1),
                        figsize=(15, 7)):
    '''

    Parameters
    ----------
    image : Hyperspy Signal2D
        This image.axes_manager scale should be calibrated.
    d_inner : float, default 7.7
        Initial 'guess' of full width at half maximum (fwhm) of
        inner (negative) gaussian to be applied to fft.
        Can be changed with sliders during visualisation.
    d_outer : float, default 14
        Initial 'guess' of full width at half maximum (fwhm) of
        outer (positive) gaussian to be applied to fft.
        Can be changed with sliders during visualisation.
    slider_min : float, default 0.1
        Minimum value on sliders
    slider_max : float, default 300
        Maximum value on sliders
    slider_step : float, default 0.1
        Step size on sliders
    plot_lims : tuple, default (0, 1)
        Used to plot a smaller section of the FFT image, which can be useful if
        the information is very small (far away!). Default plots the whole
        image.

    Examples
    --------
    >>> import temul.signal_processing as tmlsig
    >>> from temul.example_data import load_Se_implanted_MoS2_data
    >>> image = load_Se_implanted_MoS2_data()
    >>> tmlsig.visualise_dg_filter(image)

    '''

    # Get FFT of the image
    image_fft = image.fft(shift=True)
    fft_data = image_fft.amplitude.data
    sampling = image.axes_manager[0].scale
    # units = image.axes_manager[0].units

    physical_image_size = sampling * len(image.data)
    fourier_sampling = 1 / physical_image_size

    # Get radius
    fwhm_neg_gaus = d_inner / fourier_sampling
    fwhm_pos_gaus = d_outer / fourier_sampling
    r_fwhm_neg_gaus = fwhm_neg_gaus / 2
    r_fwhm_pos_gaus = fwhm_pos_gaus / 2

    # Plot circles to represent d_inner and d_outer
    # circles only represent fwhm of two gaussians (inner -ve, outer +ve)
    # definitely a better way of plotting these
    half_image_len = len(fft_data) / 2
    inner_color = 'r'
    outer_color = 'b'
    alpha = 0.4
    inner_circle = plt.Circle((half_image_len, half_image_len),
                              r_fwhm_neg_gaus, color=inner_color, alpha=alpha)
    outer_circle = plt.Circle((half_image_len, half_image_len),
                              r_fwhm_pos_gaus, color=outer_color, alpha=alpha)

    # Make a subplot to show DG filter
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    plt.subplots_adjust(bottom=0.25)
    # fig.set_figheight(100)
    # fig.set_figwidth(100)
    ax1.imshow(np.log(fft_data))
    plt.xlim(image.data.shape[-1] * plot_lims[0],
             image.data.shape[-1] * plot_lims[1])
    plt.ylim(image.data.shape[-2] * plot_lims[0],
             image.data.shape[-2] * plot_lims[1])

    ax1.add_artist(outer_circle)
    ax1.add_artist(inner_circle)

    # Make sliders
    axcolor = 'lightgoldenrodyellow'
    ax_d_inner = plt.axes([0.25, 0.15, 0.5, 0.03], facecolor=axcolor)
    ax_d_outer = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor=axcolor)

    d_inner_slider = Slider(ax_d_inner, 'Inner FWHM (pix)',
                            slider_min, slider_max, color=inner_color,
                            alpha=alpha,
                            valinit=r_fwhm_neg_gaus, valstep=slider_step)
    d_outer_slider = Slider(ax_d_outer, 'Outer FWHM (pix)',
                            slider_min, slider_max, color=outer_color,
                            alpha=alpha,
                            valinit=r_fwhm_pos_gaus, valstep=slider_step)

    def update(val):
        outer_circle.radius = d_outer_slider.val
        inner_circle.radius = d_inner_slider.val

        plt.draw()

    d_inner_slider.on_changed(update)
    d_outer_slider.on_changed(update)

    # Setup reset button
    resetax = plt.axes([0.8, 0.025, 0.1, 0.05])  # left, bottom, width, height
    reset_button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    def reset(event):
        d_inner_slider.reset()
        d_outer_slider.reset()

        outer_circle.radius = d_outer_slider.valinit
        inner_circle.radius = d_inner_slider.valinit

        plt.draw()

    reset_button.on_clicked(reset)

    # Setup filter button
    filterax = plt.axes([0.1, 0.025, 0.1, 0.05])
    filter_button = Button(filterax, 'Filter',
                           color=axcolor, hovercolor='0.975')

    def filter_image(event):
        # image.plot()
        d_inner = d_inner_slider.val
        d_outer = d_outer_slider.val
        # array = make_gaussian(fft_data.shape[-1], d_outer*100)

        nD_Gaussian_pos, nD_Gaussian_neg = make_gaussian_pos_neg(
            fft_data.shape[-1], d_inner, d_outer, 0.9, center=None)
        dg_filter = nD_Gaussian_pos + nD_Gaussian_neg

        convolution = image_fft * dg_filter

        ax2.imshow(np.log(convolution.amplitude.data))

        convolution_ifft = convolution.ifft()
        convolution_ifft.axes_manager = image.axes_manager
        ax3.imshow(convolution_ifft.data)
        plt.show()

    for ax in [ax1, ax2, ax3]:
        ax.set_axis_off()
    ax1.set_title("FFT Widget")
    ax2.set_title("Convolution")
    ax3.set_title("Filtered Image")

    filter_button.on_clicked(filter_image)
    resetax._button = reset_button
    filterax._button = filter_button


'''
Cropping and Calibrating
'''

# cropping done in the scale, so nm, pixel, or whatever you have
# cropping_area = choose_points_on_image(image.data)


def crop_image_hs(image, cropping_area, scalebar_true=True, filename=None):
    '''
    Crop a Hyperspy Signal2D by providing the `cropping_area`. See the example
    below.

    Parameters
    ----------
    image : Hyperspy Signal2D
        Image you wish to crop
    cropping_area : list of 2 floats
        The best method of choosing the area is by using the function
        "choose_points_on_image(image.data)". Choose two points on the
        image. First point is top left of area, second point is bottom right.
    scalebar_true : Bool, default True
        If set to True, the function assumes that `image.axes_manager` is
        calibrated to a unit other than pixel.
    filename : str, default None
        If set to a string, the images and cropping variables will be saved.

    Returns
    -------
    Hyperspy Signal2D

    Examples
    -------
    >>> from temul.dummy_data import get_simple_cubic_signal
    >>> from temul.signal_processing import (
    ...     choose_points_on_image, crop_image_hs)
    >>> import matplotlib.pyplot as plt
    >>> image = get_simple_cubic_signal()
    >>> image.plot()
    >>> cropping_area = choose_points_on_image(image.data) # choose two points
    >>> cropping_area = [[5,5],[50,50]] # use above line if trying yourself!
    >>> # image_cropped = crop_image_hs(image, cropping_area, False)
    >>> # image_cropped.plot()

    '''

    llim, tlim = cropping_area[0]
    rlim, blim = cropping_area[1]

    unit = image.axes_manager[0].units

    if image.axes_manager[0].scale != image.axes_manager[1].scale:
        raise ValueError("x & y scales don't match!")

    if scalebar_true is True:
        llim *= image.axes_manager[0].scale
        tlim *= image.axes_manager[0].scale
        rlim *= image.axes_manager[0].scale
        blim *= image.axes_manager[0].scale

    roi = RectangularROI(left=llim, right=rlim, top=tlim, bottom=blim)
    image.plot()
    image_crop = roi.interactive(image)
    plt.title('Cropped region highlighted', fontsize=20)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(fname=f'Cropped region highlighted_{filename}.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)

    image_crop.plot()
    plt.title('Cropped Image', fontsize=20)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(fname=f'Cropped Image_{filename}.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)
        image_crop.save(f'Cropped Image_{filename}.hspy')

        physical_image_crop_size_x = image_crop.axes_manager[0].scale * \
            image_crop.axes_manager[0].size
        physical_image_crop_size_y = image_crop.axes_manager[1].scale * \
            image_crop.axes_manager[1].size

        ''' Saving the Variables for the image and filtered Image '''
        Cropping_Variables = collections.OrderedDict()
        # Cropping_Variables['Image Name'] = [image_name]
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
        Cropping_Variables_Table.to_csv(
            f'Cropping_Variables_Table_{filename}.csv', sep=',', index=False)

    return image_crop


# cropping_area = choose_points_on_image(image.data)
def calibrate_intensity_distance_with_sublattice_roi(image,
                                                     cropping_area,
                                                     separation,
                                                     reference_image=None,
                                                     scalebar_true=False,
                                                     percent_to_nn=0.2,
                                                     mask_radius=None,
                                                     refine=True,
                                                     filename=None):
    # add max mean min etc.
    '''
    Calibrates the intensity of an image by using the brightest sublattice.
    The mean intensity of that sublattice is set to 1.

    Parameters
    ----------
    image : Hyperspy Signal2D
        Image you wish to calibrate.
    cropping_area : list of 2 floats
        The best method of choosing the area is by using the function
        "choose_points_on_image(image.data)". Choose two points on the
        image. First point is top left of area, second point is bottom right.
    separation : int, default 8
        Pixel separation between atoms as used by Atomap.
    reference_image : Hyperspy Signal2D
        Image with which `image` is compared.
    scalebar_true : Bool, default True
        If set to True, the function assumes that `image.axes_manager` is
        calibrated to a unit other than pixel.
    mask_radius : int, default None
        Radius in pixels of the mask.
    percent_to_nn : float, default 0.2
        Determines the boundary of the area surrounding each atomic
        column, as fraction of the distance to the nearest neighbour.
    refine : Bool, default False
        If set to True, the atom positions found for the calibration will be
        refined.
    filename : str, default None
        If set to a string, the image will be saved.

    Returns
    -------
    Nothing, but the mean intensity of the brightest sublattice is set to 1.

    Examples
    -------
    >>> from temul.dummy_data import get_simple_cubic_signal
    >>> from temul.signal_processing import (choose_points_on_image,
    ...             calibrate_intensity_distance_with_sublattice_roi)
    >>> import matplotlib.pyplot as plt
    >>> image = get_simple_cubic_signal()
    >>> image.plot()
    >>> crop_a = choose_points_on_image(image.data) # manually
    >>> crop_a = [[10,10],[100,100]] #use above line if trying yourself!

    calibrate_intensity_distance_with_sublattice_roi(image, crop_a, 10)
    image.plot()

    '''
    llim, tlim = cropping_area[0]
    rlim, blim = cropping_area[1]

    if image.axes_manager[0].scale != image.axes_manager[1].scale:
        raise ValueError("x & y scales don't match!")

    if scalebar_true:
        llim *= image.axes_manager[0].scale
        tlim *= image.axes_manager[0].scale
        rlim *= image.axes_manager[0].scale
        blim *= image.axes_manager[0].scale

    cal_area = RectangularROI(
        left=llim, right=rlim, top=tlim, bottom=blim)(image)
    atom_positions = get_atom_positions(
        cal_area, separation=separation, pca=True)
    calib_sub = Sublattice(atom_positions, cal_area, color='r')

    if refine:
        calib_sub.find_nearest_neighbors()
        calib_sub.refine_atom_positions_using_center_of_mass(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius,
            show_progressbar=False)
        calib_sub.refine_atom_positions_using_2d_gaussian(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius,
            show_progressbar=False)

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


# Atomap extensions
def toggle_atom_refine_position_automatically(sublattice,
                                              min_cut_off_percent,
                                              max_cut_off_percent,
                                              range_type='internal',
                                              method='mode',
                                              percent_to_nn=0.05,
                                              mask_radius=None,
                                              filename=None):
    '''
    Sets the 'refine_position' attribute of each Atom Position in a
    sublattice using a range of intensities.

    Parameters
    ----------
    sublattice : Atomap Sublattice object
    min_cut_off_percent : float, default None
        The lower end of the intensity range is defined as
        `min_cut_off_percent` * `method` value of max intensity list of
        `sublattice`.
    max_cut_off_percent : float, default None
        The upper end of the intensity range is defined as
        `max_cut_off_percent` * `method` value of max intensity list of
        `sublattice`.
    range_type : str, default 'internal'
        "internal" provides the `refine_position` attribute for each
        `Atom Position` as True if the intensity of that Atom Position
        lies between the lower and upper limits defined by
        `min_cut_off_percent` and `max_cut_off_percent`.
        "external" provides the `refine_position` attribute for each
        `Atom Position` as True if the intensity of that Atom Position
        lies outside the lower and upper limits defined by
        `min_cut_off_percent` and `max_cut_off_percent`.
    method : str, default 'mode'
        The method used to aggregate the intensity of the sublattice positions
        max intensity list. Options are "mode" and "mean"
    percent_to_nn : float, default 0.05
        Determines the boundary of the area surrounding each atomic
        column, as fraction of the distance to the nearest neighbour.
    mask_radius : int, default None
        Radius in pixels of the mask.
    filename : str, default None
        If set to a string, the Atomap `refine_position` image will be saved.

    Returns
    -------
    list of the `AtomPosition.refine_position=False` attribute.

    Examples
    -------
    >>> from temul.dummy_data import (
    ...     get_simple_cubic_sublattice_positions_on_vac)
    >>> from temul.signal_processing import (
    ...     toggle_atom_refine_position_automatically)
    >>> sublattice = get_simple_cubic_sublattice_positions_on_vac()
    >>> sublattice.find_nearest_neighbors()
    >>> sublattice.plot()
    >>> min_cut_off_percent = 0.75
    >>> max_cut_off_percent = 1.25
    >>> false_list_sublattice =  toggle_atom_refine_position_automatically(
    ...         sublattice, min_cut_off_percent, max_cut_off_percent,
    ...         range_type='internal', method='mode', percent_to_nn=0.05)
    >>> len(false_list_sublattice) # check how many atoms will not be refined
    12

    Visually check which atoms will not be refined (red dots)

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


# atomap adaption
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


# atomap adaption
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


def get_cell_image(s, points_x, points_y, method='Voronoi',
                   max_radius='Auto', reduce_func=np.min,
                   show_progressbar=True):
    '''
    The same as atomap's integrate, except instead of summing the
    region around an atom, it removes the value from all pixels in
    that region.
    For example, with this you can remove the local background intensity
    by setting reduce_func=np.min.

    Parameters
    ----------
    reduce_func : ufunc, default np.min
        function used to reduce the pixel values around each atom
        to a float.
    For the other parameters see Atomap's `integrate` function.

    Returns
    -------
    Numpy array with the same shape as s

    Examples
    --------
    >>> from temul.dummy_data import (
    ...     get_simple_cubic_sublattice_positions_on_vac)
    >>> from temul.signal_processing import get_cell_image
    >>> sublattice = get_simple_cubic_sublattice_positions_on_vac()
    >>> cell_image = get_cell_image(sublattice.image, sublattice.x_position,
    ...     sublattice.y_position)

    Plot the `cell_image` which shows, in this case, the background intensity

    >>> import matplotlib.pyplot as plt
    >>> im = plt.imshow(cell_image)

    Convert it to a Hyperspy Signal2D object:

    >>> import hyperspy.api as hs
    >>> cell_image = hs.signals.Signal2D(cell_image)
    >>> cell_image.plot()

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
    sampling : float, default None
        The image sampling in units/pixel. If set to None then the values
        returned are given in pixels.
        This may be changed in future versions if Atomap's Sublattice pixel
        attribute is updated.

    Returns
    -------
    Two lists: list of mean distances, list of standard deviations.

    Examples
    --------
    >>> from temul.dummy_data import get_simple_cubic_sublattice
    >>> from temul.signal_processing import (
    ...     mean_and_std_nearest_neighbour_distances)
    >>> sublattice = get_simple_cubic_sublattice()
    >>> mean, std = mean_and_std_nearest_neighbour_distances(sublattice)
    >>> mean_scaled, _ = mean_and_std_nearest_neighbour_distances(sublattice,
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


def choose_mask_coordinates(image, norm="log"):
    '''
    Pick the mask locations for an FFT. See get_masked_ifft() for examples and
    commit 5ba307b5af0b598bedc0284aa989d44e23fdde4d on Atomap for more details.

    Parameters
    ----------
    image : Hyperspy 2D Signal
    norm : str, default "log"
        How to scale the intensity value for the displayed image.
        Options are "linear" and "log".

    Returns
    -------
    list of pixel coordinates

    '''

    fft = image.fft(shift=True)
    fft_amp = fft.amplitude

    mask_coords = choose_points_on_image(
        fft_amp.data, norm=norm)

    return(mask_coords)


def get_masked_ifft(image, mask_coords, mask_radius=10, image_space="real",
                    keep_masked_area=True, plot_masked_fft=False):
    '''
    Creates an inverse fast Fourier transform (iFFT) from an image and mask
    coordinates. Use `choose_mask_coordinates` to manually choose mask
    coordinates in the FFT.

    Parameters
    ----------
    image : Hyperspy 2D Signal
    mask_coords : list of pixel coordinates
        Pixel coordinates of the masking locations. See the example below for
        two simple coordinates found using `choose_mask_coordinates`.
    mask_radius : int, default 10
        Radius in pixels of the mask.
    image_space : str, default "real"
        If the input image is in Fourier/diffraction/reciprocal space already,
        set space="fourier".
    keep_masked_area : Bool, default True
        If True, this will set the mask at the mask_coords.
        If False, this will set the mask as everything other than the
        mask_coords. Can be thought of as inversing the mask.
    plot_masked_fft : Bool, default False
        If True, the mask used to filter the FFT will be plotted. Good for
        checking that the mask is doing what you want. Can fail sometimes due
        to matplotlib plotting issues.

    Returns
    -------
    Inverse FFT as a Hyperspy 2D Signal

    Examples
    --------
    >>> from temul.dummy_data import get_simple_cubic_signal
    >>> from temul.signal_processing import (
    ...     choose_mask_coordinates, get_masked_ifft)
    >>> image = get_simple_cubic_signal()
    >>> mask_coords = choose_mask_coordinates(image) # use this on the image!
    >>> mask_coords = [[170.2, 170.8],[129.8, 130]]
    >>> image_ifft = get_masked_ifft(image, mask_coords)
    >>> image_ifft.plot()

    Plot the masked fft:

    >>> image_ifft = get_masked_ifft(image, mask_coords, plot_masked_fft=True)

    Use unmasked fft area and plot the masked fft:

    >>> image_ifft = get_masked_ifft(image, mask_coords, plot_masked_fft=True,
    ...     keep_masked_area=False)
    >>> image_ifft.plot()

    If the input image is already a Fourier transform:

    >>> fft_image = image.fft(shift=True)
    >>> image_ifft = get_masked_ifft(fft_image, mask_coords,
    ...     image_space='fourier')
    >>> image_ifft.plot()

    '''
    if image_space == 'real':
        fft = image.fft(shift=True)
    elif image_space == 'fourier':
        fft = image

    if len(mask_coords) == 0:
        raise ValueError("`mask_coords` has not been set.")
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

    masked_fft_image = ComplexSignal2D(masked_fft)
    if plot_masked_fft:
        masked_fft_image.amplitude.plot(norm="log")
    # sort out units here
    image_ifft = masked_fft_image.ifft()
    image_ifft = np.absolute(image_ifft)
    image_ifft.axes_manager = image.axes_manager

    return(image_ifft)


def sine_wave_function_strain_gradient(x, a, b, c, d):
    return a * np.sin((2 * np.pi * (x + b)) / c) + d
