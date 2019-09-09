

from atomap.atom_finding_refining import _make_circular_mask
from matplotlib import gridspec
import rigidregistration
from tifffile import imread, imwrite, TiffWriter
from collections import Counter
from time import time
from glob import glob
from atomap.atom_finding_refining import normalize_signal
from atomap.tools import remove_atoms_from_image_using_2d_gaussian
import os
from skimage.measure import compare_ssim as ssm
# from atomap.atom_finding_refining import get_atom_positions_in_difference_image
from scipy.ndimage.filters import gaussian_filter
import collections
from atomap.atom_finding_refining import subtract_average_background
from numpy import mean
import matplotlib
import matplotlib.pyplot as plt
import hyperspy.api as hs
import atomap.api as am
import numpy as np
from numpy import log
import CifFile
import pandas as pd
import scipy
import periodictable as pt
import matplotlib
# matplotlib.use('Agg')

import warnings
from scipy.optimize import OptimizeWarning
# warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("error", OptimizeWarning)


# refine with and plot gaussian fitting
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

    >>> amp, mu, sigma = 10, 10, 0.5
    >>> sub1_inten = np.random.normal(mu, sigma, 1000)
    >>> xdata, ydata = get_xydata_from_list_of_intensities(sub1_inten, hist_bins=50)
    '''

    hist_bins = hist_bins
    y_array, x_array = np.histogram(sublattice_intensity_list,
                                    bins=hist_bins)
    x_array = np.delete(x_array, [0])

    # set the x_ values so that they are at the middle of each hist bin
    x_separation = (x_array.max()-x_array.min())/hist_bins
    x_array = x_array - (x_separation/2)

    return(x_array, y_array)

# 1D single Gaussian


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

    >>> amp, mu, sigma = 10, 10, 0.5
    >>> sub1_inten = np.random.normal(mu, sigma, 1000)
    >>> xdata, ydata = get_xydata_from_list_of_intensities(sub1_inten, hist_bins=50)
    >>> gauss_fit_01 = _(xdata, amp, mu, sigma)    
    '''

    return amp*(1/(sigma*(np.sqrt(2*np.pi))))*(np.exp(-((xdata-mu)**2)/((2*sigma)**2)))


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

    >>> amp, mu, sigma = 10, 10, 0.5
    >>> sub1_inten = np.random.normal(mu, sigma, 1000)
    >>> xdata, ydata = get_xydata_from_list_of_intensities(sub1_inten, hist_bins=50)
    >>> popt_gauss, _ = return_fitting_of_1D_gaussian(
                            function=fit_1D_gaussian_to_data,
                            xdata=xdata,
                            ydata=ydata,
                            p0=[amp, mu, sigma])
    >>> print("calculated mean: " + str(round(np.mean(xdata),3)) + "\n"
              + "fitted mean: " + str(round(popt_gauss[1],3)))

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
    >>> amp, mu, sigma = 10, 10, 0.5
    >>> sub1_inten = np.random.normal(mu, sigma, 1000)
    >>> xdata, ydata = get_xydata_from_list_of_intensities(sub1_inten, hist_bins=50)
    >>> popt_gauss, _ = return_fitting_of_1D_gaussian(
                            function=fit_1D_gaussian_to_data,
                            xdata=xdata,
                            ydata=ydata,
                            p0=[amp, mu, sigma])
    >>> plot_gaussian_fit(xdata, ydata, function=_, 
                  amp=popt_gauss[0], mu=popt_gauss[1], sigma=popt_gauss[2],
                  gauss_art='r--', gauss_label='Gauss Fit',
                  plot_data=True, data_art='ko', data_label='Data Points',
                  plot_fill=True, facecolor='r', alpha=0.5)
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
        middle_real = middle*sublattice_scalar
        middle_intensity_list_real.append(middle_real)
    for limit in limit_intensity_list:
        limit_real = limit*sublattice_scalar
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

    if len(scaled_middle_intensity_list)+1 != len(scaled_limit_intensity_list):
        raise ValueError(
            "limit list must have a length one greater than middle list")

    if len(element_list) != len(scaled_middle_intensity_list):
        raise ValueError(
            "element list must be the same length as middle list")

    fitting_tools = []
    for i, (element, middle) in enumerate(zip(element_list, scaled_middle_intensity_list)):
        element_name = element
        middle_int = middle
        lower_int = scaled_limit_intensity_list[i]
        upper_int = scaled_limit_intensity_list[i+1]
        gauss_amp = gaussian_amp
        gauss_mu = middle
        gauss_sigma = (upper_int - lower_int)/gauss_sigma_division
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

    middle_intensity_list_real_sub1, limit_intensity_list_real_sub1 = make_middle_limit_intensity_list_real(
                                        sublattice=sub1,
                                        middle_intensity_list=middle_intensity_list_sub1, 
                                        limit_intensity_list=limit_intensity_list_sub1,
                                        method=method,
                                        sublattice_scalar=sub1_mode)

    middle_intensity_list_real_sub2, limit_intensity_list_real_sub2 = make_middle_limit_intensity_list_real(
                                        sublattice=sub2,
                                        middle_intensity_list=middle_intensity_list_sub2, 
                                        limit_intensity_list=limit_intensity_list_sub2,
                                        method=method,
                                        sublattice_scalar=sub2_mode)


    element_list_all_subs = [element_list_sub1, element_list_sub2]

    fitting_tools_all_subs = [get_fitting_tools_for_plotting_gaussians(element_list_sub1, 
                                                middle_intensity_list_real_sub1,
                                                limit_intensity_list_real_sub1),
                            get_fitting_tools_for_plotting_gaussians(element_list_sub2, 
                                                middle_intensity_list_real_sub2,
                                                limit_intensity_list_real_sub2)]


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
        for j in np.arange(0,1,1/len(element_list_all_subs[0])):
            colormap_list.append(mpl_cmap(j))
            linestyle_list.append('-')

        cycler_sub = plt.cycler(c=colormap_list,
                        linestyle=linestyle_list)

        cyclers_all.append(cycler_sub)

    if len(cyclers_all) != len(element_list_all_subs) != len(fitting_tools_all_subs):
        raise ValueError(
            "len(cyclers_all) != len(element_list_all) != len(fitting_tools_all_subs), "
            + str(len(cyclers_all)) + ', ' +
            str(len(element_list_all_subs)) + ', '
            + str(len(fitting_tools_all_subs)))

    for cycler, element, fitting in zip(cyclers_all, element_list_all_subs, fitting_tools_all_subs):
        if len(cycler) != len(element) != len(fitting):
            raise ValueError("len(cycler) != len(element) != len(fitting)")

    plt.rcParams.update({'xtick.labelsize': 'x-large',
                         'ytick.labelsize': 'x-large'})
    plt.rc('font', family='Arial')

    _, (ax1, ax2) = plt.subplots(figsize=(16, 9), nrows=2, sharex=True,
                                   gridspec_kw={'height_ratios': [2, 0.5]})
    plt.subplots_adjust(hspace=0)

    #fig.suptitle("Fit of all Elements with Residuals", family="serif", fontsize=20)
    ax2.set_xlabel("Intensity (a.u.)", family="serif",  fontsize=20)
    ax1.set_ylabel("Counts", family="serif",  fontsize=20)
    ax2.set_ylabel("Res.", family="serif",  fontsize=20)

    sub_residual_gauss_list = []
    for sublattice_array, fitting_tools_sub, cycler_sub, marker, in zip(sub_ints_all, fitting_tools_all_subs, cyclers_all, marker_list):
        x_array, y_array = get_xydata_from_list_of_intensities(sublattice_array,
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
                    sub_gauss = ax1.plot(x, individual_gauss, **kwargs)
                    sub_gauss_fill = ax1.fill_between(x,
                                                      individual_gauss.min(),
                                                      individual_gauss,
                                                      facecolor=kwargs['c'],
                                                      alpha=0.5)

                    sub_residual_gauss = abs(
                        y - (fit_1D_gaussian_to_data(x, *popt_gauss)))
                    sub_gauss_hl = ax1.plot(x, fit_1D_gaussian_to_data(x, *popt_gauss),
                                            label=r"$\bf{" + fitting_tools[0] + "}$" + ': ' +
                                            str(round(
                                                sum(abs(sub_residual_gauss)), 1)),
                                            linewidth=1.5,
                                            **kwargs)

                    sub_residual_gauss_list.append([fitting_tools[0],
                                                    sub_residual_gauss])
                    sub_resid = ax2.plot(x, sub_residual_gauss, marker=marker[1],
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
                    # https://stackoverflow.com/questions/48637960/improper-input-n-3-must-not-exceed-m-1-error-trying-to-fit-a-gaussian-function?rq=1

    legend1 = ax1.legend(
        loc="best", prop={'size': 10}, ncol=2, edgecolor='grey')
    for line in legend1.get_lines():
        line.set_linewidth(1.5)

    #ax1.hist(sub1_ints, bins=500)
    #ax1.hist(sub2_ints, bins=500)
    if filename is not None:
        plt.savefig(fname=filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=900, labels=False)


######## Image Registration ########


def rigid_registration(file, masktype='hann', n=4, findMaxima='gf'):
    ''' 
    Perform image registraion with the rigid registration package

    Parameters
    ----------

    file : stack of tiff images

    masktype : filtering method, default 'hann'
        See https://github.com/bsavitzky/rigidRegistration for 
        more information

    n : width of filter, default 4
        larger numbers mean smaller filter width
        See https://github.com/bsavitzky/rigidRegistration for 
        more information

    findMaxima : image matching method, default 'gf'
        'pixel' and 'gf' options, See 
        https://github.com/bsavitzky/rigidRegistration for 
        more information

    Returns
    -------
    Outputs of
    report of the image registration
    aligned and stacked image with and without crop
    creates a folder and places all uncropped aligned images in it


    Examples
    --------

    >>>


    '''

    # Read tiff file. Rearrange axes so final axis iterates over images
    stack = np.rollaxis(imread(file), 0, 3)
    stack = stack[:, :, :]/float(2**16)        # Normalize data between 0 and 1

    s = rigidregistration.stackregistration.imstack(stack)
    s.getFFTs()

    # Choose Mask and cutoff frequency
    s.makeFourierMask(mask=masktype, n=n)     # Set the selected Fourier mask
    # s.show_Fourier_mask(i=0,j=5)             # Display the results

    # Calculate image shifts using gaussian fitting
    findMaxima = findMaxima
    s.setGaussianFitParams(num_peaks=3, sigma_guess=3, window_radius=4)

    # Find shifts.  Set verbose=True to print the correlation status to screen
    s.findImageShifts(findMaxima=findMaxima, verbose=False)

    # Identify outliers using nearest neighbors to enforce "smoothness"
    s.set_nz(0, s.nz)
    s.get_outliers_NN(max_shift=8)
    # s.show_Rij(mask=True)

    s.make_corrected_Rij()    # Correct outliers using the transitivity relations
    # s.show_Rij_c()            # Display the corrected shift matrix

    # Create registered image stack and average
    # To skip calculation of image shifts, or correcting the shift matrix, pass the function
    s.get_averaged_image()
    # get_shifts=False, or correct_Rij=False

    s.get_all_aligned_images()
    # s.show()

    # Display report of registration procedure
    # s.show_report()

    # Save report of registration procedure
    s.save_report("registration_report.pdf")

    # Save the average image
    s.save("average_image.tif")

    # Save the average image, including outer areas. Be careful when analysis outer regions of this file
    s.save("average_image_no_crop.tif", crop=False)

    # creates a folder and put all the individual images in there
    save_individual_images(image_stack=s.stack_registered)


######## Image Comparison ########

# https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def measure_image_errors(imageA, imageB, filename):
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
                                                      plot_details=True)

    Showing the ideal case of both images being exactly equal   
    >>> imageA = am.dummy_data.get_simple_cubic_signal().data
    >>> imageB = am.dummy_data.get_simple_cubic_signal().data
    >>> mse_number, ssm_number = measure_image_errors(imageA, imageA,
                                                      plot_details=True)

    '''

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


#imageA = am.dummy_data.get_simple_cubic_signal().data
#imageB = am.dummy_data.get_simple_cubic_with_vacancies_signal().data
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

    >>> imageA = am.dummy_data.get_simple_cubic_signal(image_noise=True)
    >>> imageB = am.dummy_data.get_simple_cubic_signal()
    >>> load_and_compare_images(imageA, imageB, filename=None)

    '''
    imageA = hs.load(imageA)
    imageB = hs.load(imageB)

    mse_number, ssm_number = measure_image_errors(imageA, imageB, filename)

    return(mse_number, ssm_number)


def compare_two_image_and_create_filtered_image(
        image_to_filter,
        reference_image,
        filename,
        delta_image_filter,
        cropping_area,
        separation,
        max_sigma=6,
        percent_to_nn=0.4,
        mask_radius=None,
        refine=False):
    '''
    Gaussian blur an image for comparison with a reference image. 
    Good for finding the best gaussian blur for a simulation by 
    comparing to an experimental image.
    See measure_image_errors() and load_and_compare_images()

    >>> new_sim_data = compare_two_image(
                                    image_to_filter=simulation, 
                                    reference_image=atom_lattice_max)


    '''
    image_to_filter_data = image_to_filter.data
    reference_image_data = reference_image.data

    mse_number_list = []
    ssm_number_list = []

    for i in np.arange(0, max_sigma+delta_image_filter, delta_image_filter):

        image_to_filter_data_filtered = gaussian_filter(image_to_filter_data,
                                                        sigma=i)
        temp_image_filtered = hs.signals.Signal2D(
            image_to_filter_data_filtered)
#        temp_image_filtered.plot()
        calibrate_intensity_distance_with_sublattice_roi(image=temp_image_filtered,
                                                         cropping_area=cropping_area,
                                                         separation=separation,
                                                         percent_to_nn=percent_to_nn,
                                                         mask_radius=mask_radius,
                                                         refine=refine)

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
    ideal_sigma = (ideal_mse_number + ideal_ssm_number)/2
    ideal_sigma_y_coord = (float(min(mse)[0]) + float(max(ssm)[0]))/2

    image_to_filter_filtered = gaussian_filter(image_to_filter_data,
                                               sigma=ideal_sigma)

    image_filtered = hs.signals.Signal2D(image_to_filter_filtered)

    calibrate_intensity_distance_with_sublattice_roi(image=image_filtered,
                                                     cropping_area=cropping_area,
                                                     separation=separation,
                                                     percent_to_nn=percent_to_nn,
                                                     mask_radius=mask_radius,
                                                     refine=refine)

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


######## Image Filtering ########


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

    arr.append(np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2))

    return(arr)


def double_gaussian_fft_filter(image, filename, d_inner, d_outer, delta, real_space_sampling, units='nm'):
    # Folder: G:/SuperStem visit/Feb 2019 data/2019_02_18_QMR_S1574_MoS2-Se-15eV

    # Accuracy of calculation. Smaller = more accurate.
    #   0.01 means it will fit until intensity is 0.01 away from 0
    # 0.01 is a good starting value
    #    delta=0.01

    # Find the FWHM for both positive (outer) and negative (inner) gaussians
    # d_1 is the inner reflection diameter in units of 1/nm (or whatever unit you're working with)
    # I find these in gatan, should be a way of doing automatically.
    #    d_1 = 7.48
    #    d_outer = 14.96

    # image.plot()
    #    image.save('Original Image Data', overwrite=True)
    #    image_name = image.metadata.General.original_filename

    physical_image_size = real_space_sampling * len(image.data)
    reciprocal_sampling = 1/physical_image_size

    # Get radius
    reciprocal_d_inner = (d_inner/2)
    reciprocal_d_outer = (d_outer/2)
    reciprocal_d_inner_pix = reciprocal_d_inner/reciprocal_sampling
    reciprocal_d_outer_pix = reciprocal_d_outer/reciprocal_sampling

    fwhm_neg_gaus = reciprocal_d_inner_pix
    fwhm_pos_gaus = reciprocal_d_outer_pix

    #s = normalize_signal(subtract_average_background(s))
    image.axes_manager[0].scale = real_space_sampling
    image.axes_manager[1].scale = real_space_sampling
    image.axes_manager[0].units = units
    image.axes_manager[1].units = units
    #image.save('Calibrated Image Data', overwrite=True)

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
    #   However, we do it this way so that we can save a plot of the negative gaussian!
    #np_arr_neg = np_arr_neg
    nD_Gaussian_neg = hs.signals.Signal2D(np.array(arr_neg))
    # nD_Gaussian_neg.plot()

    neg_gauss_amplitude = 0.0
    int_and_gauss_array = []

    for neg_gauss_amplitude in np.arange(0, 1+delta, delta):

        # while neg_gauss_amplitude <= 1:
        nD_Gaussian_neg_scaled = nD_Gaussian_neg*-1 * \
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
        # Multiply the 2-D Gaussian with the FFT. This low pass filters the FFT.
        convolution = image_fft*DGFilter
        # convolution.plot(norm='log')
        #convolution_amp = convolution.amplitude
        # convolution_amp.plot(norm='log')

        # Create the inverse FFT, which is your filtered image!
        convolution_ifft = convolution.ifft()
        # convolution_ifft.plot()
        minimum_intensity = convolution_ifft.data.min()
        maximum_intensity = convolution_ifft.data.max()

        int_and_gauss_array.append(
            [neg_gauss_amplitude, minimum_intensity, maximum_intensity])

        #neg_gauss_amplitude = neg_gauss_amplitude + delta

    np_arr_2 = np.array(int_and_gauss_array)
    x_axis = np_arr_2[:, 0]
    y_axis = np_arr_2[:, 1]
    zero_line = np.zeros_like(x_axis)
    idx = np.argwhere(np.diff(np.sign(zero_line-y_axis))).flatten()
    neg_gauss_amplitude_calculated = x_axis[idx][0]

    ''' Filtering the Image with the Chosen Negative Amplitude '''
    # positive gauss
    nD_Gaussian.axes_manager[0].scale = reciprocal_sampling
    nD_Gaussian.axes_manager[1].scale = reciprocal_sampling
    nD_Gaussian.axes_manager[0].units = '1/' + units
    nD_Gaussian.axes_manager[1].units = '1/' + units

    # negative gauss
    nD_Gaussian_neg_used = nD_Gaussian_neg*-1 * \
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
    #s = normalize_signal(subtract_average_background(convolution_ifft))

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
        #Filtering_Variables_Table.to_csv('Filtering_Variables_Table.csv', sep=',', index=False)

    return(image_filtered)


######## Cropping and Calibrating ########


# cropping done in the scale, so nm, pixel, or whatever you have

#cropping_area = am.add_atoms_with_gui(image.data)

def crop_image_hs(image, cropping_area, save_image=True, save_variables=True,
                  scalebar_true=True):
    '''
    Example
    -------

    >>> image = am.dummy_data.get_simple_cubic_with_vacancies_signal()
    >>> image.plot()
    >>> cropping_area = am.add_atoms_with_gui(image.data) # choose two points
    '''

    llim, tlim = cropping_area[0]
    rlim, blim = cropping_area[1]

    unit = image.axes_manager[0].units
#    image_name = image.metadata.General.original_filename

    if image.axes_manager[0].scale != image.axes_manager[1].scale:
        raise ValueError("x & y scales don't match!")

    if scalebar_true == True:
        llim *= image.axes_manager[0].scale
        tlim *= image.axes_manager[0].scale
        rlim *= image.axes_manager[0].scale
        blim *= image.axes_manager[0].scale
    else:
        pass

    roi = hs.roi.RectangularROI(left=llim, right=rlim, top=tlim, bottom=blim)
    image.plot()
    image_crop = roi.interactive(image)

    if save_image == True:
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

    if save_image == True:
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

    if save_variables == True:
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


#cropping_area = am.add_atoms_with_gui(image.data)


def calibrate_intensity_distance_with_sublattice_roi(image,
                                                     cropping_area,
                                                     separation,
                                                     filename=None,
                                                     reference_image=None,
                                                     percent_to_nn=0.2,
                                                     mask_radius=None,
                                                     refine=True,
                                                     scalebar_true=False):  # add max mean min etc.
    ''' 
    Calibrates the intensity of an image by using a sublattice, found with some
    atomap functions. The mean intensity of that sublattice is set to 1

    Parameters
    ----------
    image : HyperSpy 2D signal, default None
        The signal can be distance calibrated. If it is, set
        scalebar_true=True
    cropping_area : list of 2 floats, default None
        The best method of choosing the area is by using the atomap
        function "add_atoms_with_gui(image.data)". Choose two points on the 
        image. First point is top left of area, second point is bottom right.
    percent_to_nn : float, default 0.40
        Determines the boundary of the area surrounding each atomic 
        column, as fraction of the distance to the nearest neighbour.
    scalebar_true : Bool, default False
        Set to True if the scale of the image is calibrated to a distance unit.

    Returns
    -------
    calibrated image data

    Example
    -------

    >>> image = am.dummy_data.get_simple_cubic_with_vacancies_signal()
    >>> image.plot()
    >>> cropping_area = am.add_atoms_with_gui(image.data) # choose two points
    >>> calibrate_intensity_distance_with_sublattice_roi(image, cropping_area)
    >>> image.plot()

    '''
    llim, tlim = cropping_area[0]
    rlim, blim = cropping_area[1]

    if image.axes_manager[0].scale != image.axes_manager[1].scale:
        raise ValueError("x & y scales don't match!")

    if scalebar_true == True:
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
    #atom_positions = am.add_atoms_with_gui(cal_area, atom_positions)
    calib_sub = am.Sublattice(atom_positions, cal_area, color='r')
#    calib_sub.plot()
    if refine == True:
        calib_sub.find_nearest_neighbors()
        calib_sub.refine_atom_positions_using_center_of_mass(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        calib_sub.refine_atom_positions_using_2d_gaussian(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    else:
        pass
    # calib_sub.plot()
    calib_sub.get_atom_column_amplitude_max_intensity(
        percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    calib_sub_max_list = calib_sub.atom_amplitude_max_intensity
    calib_sub_scalar = mean(a=calib_sub_max_list)
    image.data = image.data/calib_sub_scalar

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


######## Atomap extensions ########

def toggle_atom_refine_position_automatically(sublattice,
                                              filename,
                                              min_cut_off_percent,
                                              max_cut_off_percent,
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
    >>> sublattice = am.dummy_data.get_simple_cubic_with_vacancies_sublattice(image_noise=True)
    >>> sublattice.find_nearest_neighbors()
    >>> sublattice.plot()
    >>> false_list_sublattice =  toggle_atom_refine_position_automatically(
                                    sublattice=sublattice,
                                    min_cut_off_percent=min_cut_off_percent,
                                    max_cut_off_percent=max_cut_off_percent, 
                                    range_type='internal',
                                    method='mode',
                                    save_image=False,
                                    percent_to_nn=0.05)

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

    sublattice_min_cut_off = min_cut_off_percent*sublattice_scalar
    sublattice_max_cut_off = max_cut_off_percent*sublattice_scalar

    if range_type == 'internal':

        for i in range(0, len(sublattice.atom_list)):
            if sublattice_min_cut_off < sublattice.atom_amplitude_max_intensity[i] < sublattice_max_cut_off:
                sublattice.atom_list[i].refine_position = True
            else:
                sublattice.atom_list[i].refine_position = False

    elif range_type == 'external':
        for i in range(0, len(sublattice.atom_list)):
            if sublattice.atom_amplitude_max_intensity[i] > sublattice_max_cut_off or sublattice_min_cut_off > sublattice.atom_amplitude_max_intensity[i]:
                sublattice.atom_list[i].refine_position = True
            else:
                sublattice.atom_list[i].refine_position = False

    else:
        raise TypeError(
            "'internal' and 'external' are the only options for range_type")

    # checking we have some falses
    false_list_sublattice = []
    for i in range(0, len(sublattice.atom_list)):
        if sublattice.atom_list[i].refine_position == False:
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
        plt.savefig(fname='toggle_atom_refine_' + sublattice.name + '_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight',
                    pad_inches=None, dpi=300, labels=False)
        plt.close()

    return(false_list_sublattice)


# need to add "radius" for where to get intensity from. Do we though?
#   until we find a way of defining it in the image, radius should be left alone. Radius can be accessed in the
#   periodictable package anyway.
# need to add remove backgroun locally or with a third sublattice
#        sublattice.find_nearest_neighbors()


def get_sublattice_intensity(sublattice, intensity_type='max', remove_background_method=None,
                             background_sublattice=None, num_points=3, percent_to_nn=0.4, mask_radius=None):
    '''
    Finds the intensity for each atomic column using either max, mean, 
    min, total or all of them at once.

    The intensity values are taken from the area defined by 
    percent_to_nn.

    Results are stored in each Atom_Position object as 
    amplitude_max_intensity, amplitude_mean_intensity, 
    amplitude_min_intensity and/or amplitude_total_intensity 
    which can most easily be accessed through the sublattice object. 
    See the examples in get_atom_column_amplitude_max_intensity.

    Parameters
    ----------

    sublattice : sublattice object
        The sublattice whose intensities you are finding.
    intensity_type : string, default 'max'
        Determines the method used to find the sublattice intensities.
        The available methods are 'max', 'mean', 'min', 'total' and
        'all'. 
    remove_background_method : string, default None
        Determines the method used to remove the background_sublattice
        intensities from the image. Options are 'average' and 'local'.
    background_sublattice : sublattice object, default None
        The sublattice used if remove_background_method is used.
    num_points : int, default 3
        If remove_background_method='local', num_points is the number 
        of nearest neighbour values averaged from background_sublattice
    percent_to_nn : float, default 0.40
        Determines the boundary of the area surrounding each atomic 
        column, as fraction of the distance to the nearest neighbour.

    Returns
    -------
    2D numpy array

    Examples
    --------

    >>> import numpy as np
    >>> import atomap.api as am
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.find_nearest_neighbors()
    >>> intensities_all = get_sublattice_intensity(sublattice=sublattice, 
                                                   intensity_type='all', 
                                                   remove_background_method=None,
                                                   background_sublattice=None)

    >>> intensities_total = get_sublattice_intensity(sublattice=sublattice, 
                                                   intensity_type='total', 
                                                   remove_background_method=None,
                                                   background_sublattice=None)

    >>> intensities_total_local = get_sublattice_intensity(sublattice=sublattice, 
                                                   intensity_type='total', 
                                                   remove_background_method='local',
                                                   background_sublattice=sublattice)

    >>> intensities_max_average = get_sublattice_intensity(sublattice=sublattice, 
                                                   intensity_type='max', 
                                                   remove_background_method='average',
                                                   background_sublattice=sublattice)

    '''
    if percent_to_nn is not None:
        sublattice.find_nearest_neighbors()
    else:
        pass

    if remove_background_method == None and background_sublattice == None:
        if intensity_type == 'all':
            sublattice.get_atom_column_amplitude_max_intensity(
                percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_max_intensity_list = []
            sublattice_max_intensity_list.append(
                sublattice.atom_amplitude_max_intensity)
            sublattice_max_intensity_list = sublattice_max_intensity_list[0]
            max_intensities = np.array(sublattice_max_intensity_list)

            sublattice.get_atom_column_amplitude_mean_intensity(
                percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_mean_intensity_list = []
            sublattice_mean_intensity_list.append(
                sublattice.atom_amplitude_mean_intensity)
            sublattice_mean_intensity_list = sublattice_mean_intensity_list[0]
            mean_intensities = np.array(sublattice_mean_intensity_list)

            sublattice.get_atom_column_amplitude_min_intensity(
                percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_min_intensity_list = []
            sublattice_min_intensity_list.append(
                sublattice.atom_amplitude_min_intensity)
            sublattice_min_intensity_list = sublattice_min_intensity_list[0]
            min_intensities = np.array(sublattice_min_intensity_list)

            sublattice.get_atom_column_amplitude_total_intensity(
                percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_total_intensity_list = []
            sublattice_total_intensity_list.append(
                sublattice.atom_amplitude_total_intensity)
            sublattice_total_intensity_list = sublattice_total_intensity_list[0]
            total_intensities = np.array(sublattice_total_intensity_list)


#            sublattice_total_intensity_list, _, _ = sublattice.integrate_column_intensity() # maxradius should be changed to percent_to_nn!
#            total_intensities = np.array(sublattice_total_intensity_list)

            sublattice_intensities = np.column_stack(
                (max_intensities, mean_intensities, min_intensities, total_intensities))
            return(sublattice_intensities)
          #  return max_intensities, mean_intensities, min_intensities, total_intensities

        elif intensity_type == 'max':
            sublattice.get_atom_column_amplitude_max_intensity(
                percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_max_intensity_list = []
            sublattice_max_intensity_list.append(
                sublattice.atom_amplitude_max_intensity)
            sublattice_max_intensity_list = sublattice_max_intensity_list[0]
            max_intensities = np.array(sublattice_max_intensity_list)

            return(max_intensities)

        elif intensity_type == 'mean':
            sublattice.get_atom_column_amplitude_mean_intensity(
                percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_mean_intensity_list = []
            sublattice_mean_intensity_list.append(
                sublattice.atom_amplitude_mean_intensity)
            sublattice_mean_intensity_list = sublattice_mean_intensity_list[0]
            mean_intensities = np.array(sublattice_mean_intensity_list)

            return(mean_intensities)

        elif intensity_type == 'min':
            sublattice.get_atom_column_amplitude_min_intensity(
                percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_min_intensity_list = []
            sublattice_min_intensity_list.append(
                sublattice.atom_amplitude_min_intensity)
            sublattice_min_intensity_list = sublattice_min_intensity_list[0]
            min_intensities = np.array(sublattice_min_intensity_list)

            return(min_intensities)

        elif intensity_type == 'total':
            sublattice.get_atom_column_amplitude_total_intensity(
                percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_total_intensity_list = []
            sublattice_total_intensity_list.append(
                sublattice.atom_amplitude_total_intensity)
            sublattice_total_intensity_list = sublattice_total_intensity_list[0]
            total_intensities = np.array(sublattice_total_intensity_list)

#            sublattice_total_intensity_list, _, _ = sublattice.integrate_column_intensity()
#            total_intensities = np.array(sublattice_total_intensity_list)
            return(total_intensities)

        else:
            raise ValueError('You must choose an intensity_type')

    elif remove_background_method == 'average':

        sublattice_intensity_list_average_bksubtracted = remove_average_background(sublattice=sublattice,
                                                                                   background_sublattice=background_sublattice,
                                                                                   intensity_type=intensity_type,
                                                                                   percent_to_nn=percent_to_nn,
                                                                                   mask_radius=mask_radius)
        return(sublattice_intensity_list_average_bksubtracted)

    elif remove_background_method == 'local':

        sublattice_intensity_list_local_bksubtracted = remove_local_background(sublattice=sublattice,
                                                                               background_sublattice=background_sublattice,
                                                                               intensity_type=intensity_type,
                                                                               num_points=num_points,
                                                                               percent_to_nn=percent_to_nn,
                                                                               mask_radius=mask_radius)
        return(sublattice_intensity_list_local_bksubtracted)

    else:
        pass


def get_pixel_count_from_image_slice(
        self,
        image_data,
        percent_to_nn=0.40):
    """
    Fid the number of pixels in an area when calling
    _get_image_slice_around_atom()

    Parameters
    ----------

    image_data : Numpy 2D array
    percent_to_nn : float, default 0.40
        Determines the boundary of the area surrounding each atomic 
        column, as fraction of the distance to the nearest neighbour.

    Returns
    -------
    The number of pixels in the image_slice

    Examples
    --------

    >>> import atomap.api as am
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.find_nearest_neighbors()
    >>> atom0 = sublattice.atom_list[0]
    >>> pixel_count = atom0.get_pixel_count_from_image_slice(sublattice.image)

    """
    closest_neighbor = self.get_closest_neighbor()

    slice_size = closest_neighbor * percent_to_nn * 2
    # data_slice, x0, y0 - see atomap documentation
    data_slice, _, _ = self._get_image_slice_around_atom(
        image_data, slice_size)

    pixel_count = len(data_slice[0]) * len(data_slice[0])

    return(pixel_count)


def remove_average_background(sublattice, intensity_type,
                              background_sublattice, percent_to_nn=0.40,
                              mask_radius=None):
    '''
    Remove the average background from a sublattice intensity using
    a background sublattice. 

    Parameters
    ----------

    sublattice : sublattice object
        The sublattice whose intensities are of interest.
    intensity_type : string
        Determines the method used to find the sublattice intensities.
        The available methods are 'max', 'mean', 'min' and 'all'. 
    background_sublattice : sublattice object
        The sublattice used to find the average background.
    percent_to_nn : float, default 0.4
        Determines the boundary of the area surrounding each atomic 
        column, as fraction of the distance to the nearest neighbour.

    Returns
    -------
    2D numpy array

    Examples
    --------

    >>> import numpy as np
    >>> import atomap.api as am
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.find_nearest_neighbors()
    >>> intensities_all = remove_average_background(sublattice, intensity_type='all',
                                                    background_sublattice=sublattice)
    >>> intensities_max = remove_average_background(sublattice, intensity_type='max',
                                                background_sublattice=sublattice)

    '''
    background_sublattice.find_nearest_neighbors()
    background_sublattice.get_atom_column_amplitude_min_intensity(
        percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    background_sublattice_min = []
    background_sublattice_min.append(
        background_sublattice.atom_amplitude_min_intensity)
    background_sublattice_mean_of_min = np.mean(background_sublattice_min)

    if intensity_type == 'all':
        sublattice.get_atom_column_amplitude_max_intensity(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_max_intensity_list = []
        sublattice_max_intensity_list.append(
            sublattice.atom_amplitude_max_intensity)
        sublattice_max_intensity_list = sublattice_max_intensity_list[0]
        max_intensities = np.array(
            sublattice_max_intensity_list) - background_sublattice_mean_of_min

        sublattice.get_atom_column_amplitude_mean_intensity(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_mean_intensity_list = []
        sublattice_mean_intensity_list.append(
            sublattice.atom_amplitude_mean_intensity)
        sublattice_mean_intensity_list = sublattice_mean_intensity_list[0]
        mean_intensities = np.array(
            sublattice_mean_intensity_list) - background_sublattice_mean_of_min

        sublattice.get_atom_column_amplitude_min_intensity(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_min_intensity_list = []
        sublattice_min_intensity_list.append(
            sublattice.atom_amplitude_min_intensity)
        sublattice_min_intensity_list = sublattice_min_intensity_list[0]
        min_intensities = np.array(
            sublattice_min_intensity_list) - background_sublattice_mean_of_min

#        sublattice.get_atom_column_amplitude_total_intensity(percent_to_nn=percent_to_nn)
#        sublattice_total_intensity_list = []
#        sublattice_total_intensity_list.append(sublattice.atom_amplitude_total_intensity)
#        sublattice_total_intensity_list = sublattice_total_intensity_list[0]
#        total_intensities = np.array(sublattice_total_intensity_list) - background_sublattice_mean_of_min

        sublattice_intensities = np.column_stack(
            (max_intensities, mean_intensities, min_intensities))
        return sublattice_intensities

    elif intensity_type == 'max':
        sublattice.get_atom_column_amplitude_max_intensity(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_max_intensity_list = []
        sublattice_max_intensity_list.append(
            sublattice.atom_amplitude_max_intensity)
        sublattice_max_intensity_list = sublattice_max_intensity_list[0]
        max_intensities = np.array(
            sublattice_max_intensity_list) - background_sublattice_mean_of_min

        return max_intensities

    elif intensity_type == 'mean':
        sublattice.get_atom_column_amplitude_mean_intensity(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_mean_intensity_list = []
        sublattice_mean_intensity_list.append(
            sublattice.atom_amplitude_mean_intensity)
        sublattice_mean_intensity_list = sublattice_mean_intensity_list[0]
        mean_intensities = np.array(
            sublattice_mean_intensity_list) - background_sublattice_mean_of_min

        return mean_intensities

    elif intensity_type == 'min':
        sublattice.get_atom_column_amplitude_min_intensity(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_min_intensity_list = []
        sublattice_min_intensity_list.append(
            sublattice.atom_amplitude_min_intensity)
        sublattice_min_intensity_list = sublattice_min_intensity_list[0]
        min_intensities = np.array(
            sublattice_min_intensity_list) - background_sublattice_mean_of_min

        return min_intensities

    elif intensity_type == 'total':
        raise ValueError(
            "Average background removal doesn't work with total intensity, yet")
#        sublattice.get_atom_column_amplitude_total_intensity(percent_to_nn=percent_to_nn)
#        sublattice_total_intensity_list = []
#        sublattice_total_intensity_list.append(sublattice.atom_amplitude_total_intensity)
#        sublattice_total_intensity_list = sublattice_total_intensity_list[0]
#        total_intensities = np.array(sublattice_total_intensity_list) - background_sublattice_mean_of_min
#
#        return total_intensities

    else:
        pass

#
#sublattice0 = am.dummy_data.get_simple_cubic_sublattice()
#
# inten = get_sublattice_intensity(sublattice=sublattice0, intensity_type='max', remove_background_method='local',
#                         background_sublattice=sublattice0, num_points=3, percent_to_nn=0.3)


# can make the mean/mode option better:
#   code blocks aren't needed, just put the if statement lower down where the change is...


def remove_local_background(sublattice, background_sublattice, intensity_type,
                            num_points=3, percent_to_nn=0.40, mask_radius=None):
    '''
    Remove the local background from a sublattice intensity using
    a background sublattice. 

    Parameters
    ----------

    sublattice : sublattice object
        The sublattice whose intensities are of interest.
    intensity_type : string
        Determines the method used to find the sublattice intensities.
        The available methods are 'max', 'mean', 'min', 'total' and 'all'. 
    background_sublattice : sublattice object
        The sublattice used to find the local backgrounds.
    num_points : int, default 3
        The number of nearest neighbour values averaged from 
        background_sublattice
    percent_to_nn : float, default 0.40
        Determines the boundary of the area surrounding each atomic 
        column, as fraction of the distance to the nearest neighbour.

    Returns
    -------
    2D numpy array

    Examples
    --------

    >>> import numpy as np
    >>> import atomap.api as am
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.find_nearest_neighbors()
    >>> intensities_total = remove_local_background(sublattice, intensity_type='total',
                                                    background_sublattice=sublattice)
    >>> intensities_max = remove_local_background(sublattice, intensity_type='max',
                                                  background_sublattice=sublattice)

    '''
    # get background_sublattice intensity list

    if percent_to_nn is not None:
        sublattice.find_nearest_neighbors()
        background_sublattice.find_nearest_neighbors()
    else:
        pass

    background_sublattice.get_atom_column_amplitude_min_intensity(
        percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    background_sublattice_min_intensity_list = []
    background_sublattice_min_intensity_list.append(
        background_sublattice.atom_amplitude_min_intensity)
    background_sublattice_min_intensity_list = background_sublattice_min_intensity_list[0]
    if intensity_type == 'all':
        raise ValueError(
            "All intensities has not yet been implemented. Use max, mean or total instead")

    if num_points == 0:
        raise ValueError(
            "num_points cannot be 0 if you wish to locally remove the background")

    if intensity_type == 'max':
        # get list of sublattice and background_sublattice atom positions
        # np.array().T will not be needed in newer versions of atomap
        sublattice_atom_pos = np.array(sublattice.atom_positions).T
        background_sublattice_atom_pos = np.array(
            background_sublattice.atom_positions).T

        # get sublattice intensity list
        # could change to my function, which allows choice of intensity type
        sublattice.get_atom_column_amplitude_max_intensity(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_max_intensity_list = []
        sublattice_max_intensity_list.append(
            sublattice.atom_amplitude_max_intensity)
        sublattice_max_intensity_list = sublattice_max_intensity_list[0]

        # create list which will be output
        # therefore the original data is not changed!
        sublattice_max_intensity_list_bksubtracted = []

        # for each sublattice atom position, calculate the nearest
        #   background_sublattice atom positions.
        for p in range(0, len(sublattice_atom_pos)):

            xy_distances = background_sublattice_atom_pos - \
                sublattice_atom_pos[p]

            # put all distances in this array with this loop
            vector_array = []
            for i in range(0, len(xy_distances)):
                # get distance from sublattice position to every background_sublattice position
                vector = np.sqrt(
                    (xy_distances[i][0]**2) + (xy_distances[i][1]**2))
                vector_array.append(vector)
            # convert to numpy array
            vector_array = np.array(vector_array)

            # sort through the vector_array and find the 1st to kth smallest distance and find the
            #   corressponding index
            # num_points is the number of nearest points from which the background will be averaged
            k = num_points
            min_indices = list(np.argpartition(vector_array, k)[:k])
            # sum the chosen intensities and find the mean (or median - add this)
            local_background = 0
            for index in min_indices:
                local_background += background_sublattice.atom_amplitude_min_intensity[index]

            local_background_mean = local_background / k

            # subtract this mean local background intensity from the sublattice
            #   atom position intensity
            # indexing here is the loop digit p
            sublattice_bksubtracted_atom = np.array(sublattice.atom_amplitude_max_intensity[p]) - \
                np.array(local_background_mean)

            sublattice_max_intensity_list_bksubtracted.append(
                [sublattice_bksubtracted_atom])

        sublattice_max_intensity_list_bksubtracted = np.array(
            sublattice_max_intensity_list_bksubtracted)

        return(sublattice_max_intensity_list_bksubtracted[:, 0])

    elif intensity_type == 'mean':
        # get list of sublattice and background_sublattice atom positions
        # np.array().T will not be needed in newer versions of atomap
        sublattice_atom_pos = np.array(sublattice.atom_positions).T
        background_sublattice_atom_pos = np.array(
            background_sublattice.atom_positions).T

        # get sublattice intensity list
        # could change to my function, which allows choice of intensity type
        sublattice.get_atom_column_amplitude_mean_intensity(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_mean_intensity_list = []
        sublattice_mean_intensity_list.append(
            sublattice.atom_amplitude_mean_intensity)
        sublattice_mean_intensity_list = sublattice_mean_intensity_list[0]

        # create list which will be output
        # therefore the original data is not changed!
        sublattice_mean_intensity_list_bksubtracted = []

        # for each sublattice atom position, calculate the nearest
        #   background_sublattice atom positions.
        for p in range(0, len(sublattice_atom_pos)):

            xy_distances = background_sublattice_atom_pos - \
                sublattice_atom_pos[p]

            # put all distances in this array with this loop
            vector_array = []
            for i in range(0, len(xy_distances)):
                # get distance from sublattice position to every background_sublattice position
                vector = np.sqrt(
                    (xy_distances[i][0]**2) + (xy_distances[i][1]**2))
                vector_array.append(vector)
            # convert to numpy array
            vector_array = np.array(vector_array)

            # sort through the vector_array and find the 1st to kth smallest distance and find the
            #   corressponding index
            # num_points is the number of nearest points from which the background will be averaged
            k = num_points
            min_indices = list(np.argpartition(vector_array, k)[:k])
            # sum the chosen intensities and find the mean (or median - add this)
            local_background = 0
            for index in min_indices:
                local_background += background_sublattice.atom_amplitude_min_intensity[index]

            local_background_mean = local_background / k

            # subtract this mean local background intensity from the sublattice
            #   atom position intensity
            # indexing here is the loop digit p
            sublattice_bksubtracted_atom = np.array(sublattice.atom_amplitude_mean_intensity[p]) - \
                local_background_mean

            sublattice_mean_intensity_list_bksubtracted.append(
                [sublattice_bksubtracted_atom])

        sublattice_mean_intensity_list_bksubtracted = np.array(
            sublattice_mean_intensity_list_bksubtracted)

        return(sublattice_mean_intensity_list_bksubtracted[:, 0])

    elif intensity_type == 'total':
        # get list of sublattice and background_sublattice atom positions
        # np.array().T will not be needed in newer versions of atomap
        sublattice_atom_pos = np.array(sublattice.atom_positions).T
        background_sublattice_atom_pos = np.array(
            background_sublattice.atom_positions).T

        # get sublattice intensity list
        # could change to my function, which allows choice of intensity type
        sublattice.get_atom_column_amplitude_total_intensity(
            percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_total_intensity_list = []
        sublattice_total_intensity_list.append(
            sublattice.atom_amplitude_total_intensity)
        sublattice_total_intensity_list = sublattice_total_intensity_list[0]

        # create list which will be output
        # therefore the original data is not changed!
        sublattice_total_intensity_list_bksubtracted = []

        # for each sublattice atom position, calculate the nearest
        #   background_sublattice atom positions.
        for p in range(0, len(sublattice_atom_pos)):

            xy_distances = background_sublattice_atom_pos - \
                sublattice_atom_pos[p]

            # put all distances in this array with this loop
            vector_array = []
            for i in range(0, len(xy_distances)):
                # get distance from sublattice position to every background_sublattice position
                vector = np.sqrt(
                    (xy_distances[i][0]**2) + (xy_distances[i][1]**2))
                vector_array.append(vector)
            # convert to numpy array
            vector_array = np.array(vector_array)

            # sort through the vector_array and find the 1st to kth smallest distance and find the
            #   corressponding index
            # num_points is the number of nearest points from which the background will be averaged
            k = num_points
            min_indices = list(np.argpartition(vector_array, range(k))[:k])
            # if you want the values rather than the indices, use:
            # vector_array[np.argpartition(vector_array, range(k))[:k]]
            # sum the chosen intensities and find the total (or median - add this)
            local_background = 0
            for index in min_indices:
                local_background += background_sublattice.atom_amplitude_min_intensity[index]

            local_background_mean = local_background / k

            # for summing pixels around atom
            if mask_radius is None:
                pixel_count_in_region = get_pixel_count_from_image_slice(sublattice.atom_list[p],
                                                                         sublattice.image,
                                                                         percent_to_nn)
            elif mask_radius is not None:
                mask = _make_circular_mask(centerX=sublattice.atom_list[p].pixel_x,
                                           centerY=sublattice.atom_list[p].pixel_y,
                                           imageSizeX=sublattice.image.shape[0],
                                           imageSizeY=sublattice.image.shape[1],
                                           radius=mask_radius)

                pixel_count_in_region = len(sublattice.image[mask])

            local_background_mean_summed = pixel_count_in_region * local_background_mean

            # subtract this mean local background intensity from the sublattice
            #   atom position intensity
            # indexing here is the loop digit p
            sublattice_bksubtracted_atom = np.array(sublattice.atom_amplitude_total_intensity[p]) - \
                local_background_mean_summed

            sublattice_total_intensity_list_bksubtracted.append(
                [sublattice_bksubtracted_atom])

        sublattice_total_intensity_list_bksubtracted = np.array(
            sublattice_total_intensity_list_bksubtracted)

        return(sublattice_total_intensity_list_bksubtracted[:, 0])

    else:
        raise ValueError(
            "You must choose a valid intensity_type. Try max, mean or total")
