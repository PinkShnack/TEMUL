"""

NOTHING IMPLEMENTED YET

from atomap.atom_finding_refining import _make_circular_mask
from matplotlib import gridspec
import rigidregistration
from tifffile import imread, imwrite, TiffWriter
from collections import Counter
import warnings
from time import time
from glob import glob
from atomap.atom_finding_refining import normalize_signal
from atomap.tools import remove_atoms_from_image_using_2d_gaussian
import os
from skimage.measure import compare_ssim as ssm
from atomap.atom_finding_refining import get_atom_positions_in_difference_image
from scipy.ndimage.filters import gaussian_filter
import collections
from atomap.atom_finding_refining import subtract_average_background
from numpy import mean
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


def Eff_Beta(E_0, E_p, alpha, beta):
    '''This function is adapted from the MATLAB code at the back of
    Egerton's book.
    it calculates:
    - effective collection angle as a function of the collection semi-angle,
    convergence angle and excitation energy. B*, bstar
    - Bethe-Ridge Angle, the angle beyond which the dipole approximation is no
    longer applicable. Theta_Br
    - plasmon mean free path length (pmfp)
    - inelastic mean free path length (imfp)
    - total mean free path length (tmfp)

    *kw arguments
    E_0 = The microscope High Tension or electron accelerating voltage. (keV)
    E_p = The excitation energy of the plasmon in the sample, or another
    excitation. (eV)
    alpha = probe convergence semi-angle. 0 for TEM mode, parallel beam (mrad)
    beta = detector collection semi-angle. (mrad)
    '''

    F = (1+(E_0/1022))/(1+E_0/511)**2
    Fg = (1+E_0/1022)/(1+E_0/511)
    T = E_0*F  # keV
    tgt = 2*Fg*E_0
    a0 = 0.0529  # nm

    theta_E = (E_p+1e-6)/tgt
    a2 = (alpha*alpha*1e-6) + 1e-10  # radians^2, avoiding inf for alpha=0
    b2 = beta*beta*1e-6  # radians^2
    t2 = theta_E*theta_E*1e-6  # radians^2
    eta1 = (((a2+b2+t2)**2-4*a2*b2)**0.5)-a2-b2-t2
    eta2 = 2*b2*log(0.5/t2*((((a2+t2-b2)**2+4.*b2*t2)**0.5)+a2+t2-b2))
    eta3 = 2*a2*log(0.5/t2*(((b2+t2-a2)**2+4*a2*t2)**0.5+b2+t2-a2))
    eta = (eta1+eta2+eta3)/a2/log(4/t2)
    f1 = (eta1+eta2+eta3)/2/a2/log(1+b2/t2)
    f2 = f1

    if (alpha/beta) > 1:
        f2 = f1*(a2/b2)

    bstar = theta_E*(exp(f2*log(1+b2/t2))-1)**0.5
    theta_br = 1000*(E_p/E_0/1000)**0.5

    if bstar < theta_br:
        pmfp = 4000*a0*T/E_p/log(1+bstar**2/theta_E**2)
        imfp = 106*F*E_0/E_p/log(2*bstar*E_0/E_p)
        return bstar, theta_br, pmfp, imfp
    else:
        print('Dipole range is exceeded')
        tmfp = 4000*a0*T/E_p/log(1+theta_br**2/theta_E**2)
        x = 0
        return bstar, theta_br, tmfp, x


def sum_spectra(s, xrange=[0, s.axes_manager[1].size-1], yrange=[
    0, s.axes_manager[1].size-1]):
    '''
    Sums spectra in spectrum image together using the inputs xrange and y
    range. If these inputs are left
    blank then the all the spectra in the SI will be summed.
    If you want to sum SI along a navigational Axis you can use the function
    hs.
    kw arguments
        xrange: from the hyperspy spectrum imaging tool pick the starting and
        ending index and write as [start,end]
        yrange: from the hyperspy spectrum imaging tool pick the starting and
        ending index and write as [start,end]

        i.e. x = sum_spectra(s,xrange[2,5],yrange=[0,8])

        default range is the full spectrum image
    '''

    '''convert SI into a easier to use form'''
    spectrum = s.data
    '''sum the spectra over the appropriate navigation dimensions'''
    summed_spectrum = np.zeros([np.shape(spectrum)[2], 1])
    for x in range(np.shape(spectrum)[2]):
        summed_spectrum[x] = np.sum(
            spectrum[yrange[0]:yrange[1]+1, xrange[0]:xrange[1]+1, x])
    '''write the summed spectrum back as a hyperspy signal'''
    sample_s = s.inav[0, 0]
    spectrum_sum = hs.signals.EELSSpectrum(
        summed_spectrum, axes=[
            {'name': 'Energy Loss','size':sample_s.axes_manager[0].size,
            'units':'eV', 'scale':sample_s.axes_manager[0].scale,
            'offset':sample_s.axes_manager[0].offset}])
    spectrum_sum.original_metadata = s.original_metadata
    return spectrum_sum

"""
