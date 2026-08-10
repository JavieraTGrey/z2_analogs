"""
This script contains series of useful functions to be used in the course
of the magE analogs research proyect.

Created by BN.
Creation date: Unknown. Last edit: 08-03-2022
"""

import os
import scipy.optimize as so
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
import uncertainties.umath as uum
from uncertainties import ufloat
from uncertainties import unumpy
from .spec.spec_classes import *


"""

The following functions were made for Investigación Dirigida II in
the Spring semester of 2020. They are for continuous extraction
(for line fitting) and data treatment.
"""


def lineData(data, order, norm, interval):
    """
    Extract a dataframe of emission line data in an interval

    Params
    ------
        data : MageMultispec object
        order : order of the spectrum
        norm : normalization factor of the spectrum
        interval : wavelength interval of the line/lines

    Output
    ------
        new_dF : pandas DataFrame.
    """
    dF = pd.DataFrame({'Wavelength': data.wavelength(order), 
                       'Flux': data.flux(order)*norm,
                      'Noise': data.noise(order)*norm})
    
    cond = np.logical_and(interval[0] < dF['Wavelength'], 
                          dF['Wavelength'] < interval[1])
    
    mask = dF.mask(cond, True).mask(np.logical_not(cond), False)
    mask_wav = mask['Wavelength']
    new_dF = pd.DataFrame({'Wavelength': dF['Wavelength'][mask_wav],
                           'Flux': dF['Flux'][mask['Flux']], 
                           'Noise': dF['Noise'][mask['Noise']]})
    return new_dF


def model(wl, c):
    """
    
    Constant model for continuous fit.
    
    Params
    ------
        wl : wavelength
        c : parameter
    
    Output
    ------
        c : float
    """
    return c


def optimalContinuous(wl, flux, noise, p0):
    """
    
    Return optimal parameter for a constant model of the
    continuum of the spectrum.
    
    Params
    ------
        wl : wavelength
        flux : flux
        noise : flux error
        p0 : initial guess for optimal parameter
        
    Output
    ------
        opt : numpy.ndarray containing the optimal parameters
        std : numpy.ndarray containing the uncertainties of the parameters
    """
    
    opt, var = so.curve_fit(f=model, xdata=wl, ydata=flux, 
                            p0=p0, sigma=noise)
    std = np.sqrt(np.diag(var))
    return opt, std


def contExtract(dF, low_limit, up_limit, p0):
    """
    
    This function extract a window of 50 Angstrom around emission lines
    to fit a 0th order polynomial to the continuum and then it gets rid
    of it giving a clean spectrum.
    
    Params
    ------
        df : dataframe of the line spectrum (from lineData)
        low_limit : left window
        up_limit :  right window
        p0 :  initial guess for continuous polinomial
    
    Output:
        new_dF : dataFrame with the continuum extracted spectrum
    """
    lower = np.logical_and(low_limit[0] < dF['Wavelength'], 
                           dF['Wavelength'] < low_limit[1])
    upper = np.logical_and(up_limit[0] < dF['Wavelength'], 
                           dF['Wavelength'] < up_limit[1])
    within_cond = np.logical_or(lower, upper)
    
    mask = dF['Wavelength'].mask(within_cond, True)
    mask = mask.mask(np.logical_not(within_cond), False)
    opt, std = optimalContinuous(dF['Wavelength'][mask], 
                                 dF['Flux'][mask], 
                                 dF['Noise'][mask], p0=p0)
    cFlux = dF['Flux'] - opt
    dF = dF.assign(Flux=cFlux)
    line_cond = np.logical_and(low_limit[0] < dF['Wavelength'], 
                               dF['Wavelength'] < up_limit[1])
    line_mask = dF.mask(line_cond, True)
    line_mask = line_mask.mask(np.logical_not(line_cond), False)
    line_mask_wav = line_mask['Wavelength']
    new_dF = pd.DataFrame({'Wavelength': dF['Wavelength'][line_mask_wav],
                           'Flux': dF['Flux'][line_mask['Flux']],
                           'Noise': dF['Noise'][line_mask['Noise']]})
    return new_dF


"""

The following functions report characteristics of the results of line
fitting.
"""


def vel(sigma, err, rest_frame_wl):
    """
    
    Returns the dispersion velocity of an emission line component.
    
    Params
    ------
        sigma : velocity dispersion from the emission line fit.
        err : uncertainty of sigma
        rest_frame_wl : wavelength center of the em. line.

    Output
    ------
        value : velocity dispersion from emission line
        err : uncertainty of the velocity
    """

    FWHM = 2*sigma
    FWHM_err = 2*sigma*np.sqrt((err/sigma)**2)
    sol_km_s = c / 1000
    factor = sol_km_s / rest_frame_wl
    value = factor*FWHM
    err = value*np.sqrt((FWHM_err/FWHM)**2)
    return value, err


def reportVel(narrow, nar_err, broad, brd_err, rest_frame_wl):
    """
    
    This function print the results of vel() function.

    Params
    ------
        narrow : narrow velocity component of an emission line
        nar_err : uncertainty of narrow velocity
        broad : broad velocity component of the same emission line
        brd_err : uncertainty of broad velocity
        rest_frame_wl : wavelength center of the em. line.
        
    Output
    ------
        None
    """
    nar, err_nar = vel(narrow, nar_err, rest_frame_wl)
    brd, err_brd = vel(broad, brd_err, rest_frame_wl)
    print('Narrow component (FWHM): {:.2f} +/- {:.2f} km/s'.format(nar, 
                                                                   err_nar))
    print('Broad component (FWHM): {:.2f} +/- {:.2f} km/s'.format(brd, 
                                                                  err_brd))


def reportIntensities(par, unc, nlines=1):
    """
    
    Reports the intensity values of a free fit, given a set of
    estimated parameters.
    
    Params
    ------
        par: estimated parameters from curve_fit routines
        unc: uncertainty of each parameter, estimated by the covariance
        matrix of curve_fit routines.
        nlines: {1, 2, 3} number of lines of the fit.
    
    Output
    ------
        None
    """
    print('---------------------------')
    print('Line intensities report')

    if nlines == 1:
        print('Narrow: {:.3f} +/- {:.3f}'.format(par[2], unc[2]))
        print('Broad: {:.3f} +/- {:.3f}'.format(par[3], unc[3]))
    if nlines == 2:
        print('Narrow left: {:.3f} +/- {:.3f}'.format(par[0], unc[0]),
              '| Broad left: {:.3f} +/- {:.3f}'.format(par[1], unc[1]))
        print('Narrow right: {:.3f} +/- {:.3f}'.format(par[2], unc[2]),
              '| Broad right: {:.3f} +/- {:.3f}'.format(par[3], unc[3]))
    if nlines == 3:
        print('Narrow left: {:.3f} +/- {:.3f}'.format(par[0], unc[0]),
              '| Broad left: {:.3f} +/- {:.3f}'.format(par[1], unc[1]))
        print('Narrow middle: {:.3f} +/- {:.3f}'.format(par[2], unc[2]),
              '| Broad middle: {:.3f} +/- {:.3f}'.format(par[3], unc[3]))
        print('Narrow right: {:.3f} +/- {:.3f}'.format(par[4], unc[4]),
              '| Broad right: {:.3f} +/- {:.3f}'.format(par[5], unc[5]))


#########################################################################

"""

The following functions are gaussian composite models to fit different
cases of emission lines (embedded lines, for example) with different
number of components (two component gaussian fit for broad lines,
for example).
"""

def free2CompFit(data, z, mu):
    """
    
    This is a model that fits 1 emission line using 2 gaussians
    all its parameters set as free.
    
    Params
    ------
        data : DataFrame
        z : galaxy redshift
        mu : rest frame wavelength
        
    Output
    ------
        par : numpy.ndarray with the values of the optimal parameters
        std : numpy.ndarray with the uncertainties of each parameter
    """
    wl = data['Wavelength']
    fl = data['Flux']
    err = data['Noise']
    maxFl = max(fl)/2

    par, cov = so.curve_fit(f=lambda wl, z1, z2, A1, A2, o1, o2: \
                            freeDoubleComp(wl, z1, z2, A1, A2, o1, o2, \
                            mu=mu), xdata=wl, ydata=fl, p0=[z, z, maxFl, \
                            maxFl/2, 1, 3], sigma=err,
                            bounds=(0, np.inf))
    std = np.sqrt(np.diag(cov))
    return par, std


def singleLine(wl, pars):
    """
    
    This function is just to reproduce the best model using the best
    parameters found. This is used mainly for plotting.
    
    Params
    ------
        wl : wavelength
        pars : parameters of the best model
        
    Output
    ------
        output : array or scalar with the values of the evaluated model
    """
    mu, A, sigma = pars
    
    output = A * np.exp(-(wl - mu)**2 / 2 / sigma)
    return output


def tripleComp(wl, z1, z2, z3, A1, A2, A3, o1, o2, o3, mu):
    """
    
    This is a model with all its parameters set as free. It is for
    fitting: 1 Emission line, 3 Gaussians.
    
    Params
    ------
        wl : wavelength
        mu : the center of the gaussian uncorrected by redshift
        z : redshift
        F1, F2, F3 : component amplitudes
        o1, o2, o3 : standard deviation of each component (sigma)
    
    Output
    ------
        comp1 + comp2 + comp3 : Sum of the composite model.
    """
    cMu1 = mu * (1 + z1)
    cMu2 = mu * (1 + z2)
    cMu3 = mu * (1 + z3)
    
    comp1 = A1 * np.exp(-(wl - cMu1)**2 / 2 / o1**2)
    comp2 = A2 * np.exp(-(wl - cMu2)**2 / 2 / o2**2)
    comp3 = A3 * np.exp(-(wl - cMu3)**2 / 2 / o3**2)
    return comp1 + comp2 + comp3


def freeDoubleComp(wl, z1, z2, A1, A2, o1, o2, mu):
    """
    
    This is a model with all its parameters set as free. It is for
    fitting: 1 Emission line, 2 Gaussians.
    
    Params
    ------
        wl : wavelength
        mu : the center of the gaussian uncorrected by redshift
        z1, z1 : redshift
        A1, A2 : component amplitudes
        o1, o2 : standard deviation of each component (sigma)
        
    Output
    ------
        comp1 + comp2 : sum of the composite model
    """
    cMu1 = mu * (1 + z1)
    cMu2 = mu * (1 + z2)
    
    comp1 = A1 * np.exp(-(wl - cMu1)**2 / 2 / o1**2)
    comp2 = A2 * np.exp(-(wl - cMu2)**2 / 2 / o2**2)
    
    return comp1 + comp2


def consDoubleComp(wl, A1, A2, z1, z2, o1, o2, mu):
    """
    
    This is a model with their parameters thought to be fixed. It is for
    fitting: 1 Emission line, 2 Gaussians.
    
    Params
    ------
        wl : wavelength
        A1, A2 : component amplitudes
        z1, z1 : redshift
        o1, o2 : standard deviation of each component (sigma)
        mu : the center of the gaussian uncorrected by redshift
        
    Output
    ------
        comp1 + comp2 : sum of the composite model
    """
    cMu1 = mu * (1 + z1)
    cMu2 = mu * (1 + z2)
    
    comp1 = A1 * np.exp(-(wl - cMu1)**2 / 2 / o1**2)
    comp2 = A2 * np.exp(-(wl - cMu2)**2 / 2 / o2**2)
    
    return comp1 + comp2


def doubleLine(wl, A1_1, A1_2, A2_1, A2_2, z1, z2, o1_1, o1_2, o2_1,
               o2_2, mu1, mu2):
    """
    
    This is a model with all its parameters as free. It is for
    fitting: 2 Emission line, 2 Gaussians.
    
    Params
    -------
        wl: wavelength
        Ai_j : the ith component flux of the jth gaussian
        z1, z2 : redshift of each gaussian component
        oi_j : the ith component sigma of the jth gaussian
        mu1, mu2 : center of each emission line
    
    Output
    ------
        line1 + line2 : sum of the two double gaussians
    """

    line1 = freeDoubleComp(wl, z1, z2, A1_1, A1_2, o1_1, o1_2, mu1)
    line2 = freeDoubleComp(wl, z1, z2, A2_1, A2_2, o2_1, o2_2, mu2)
    
    return line1 + line2


def tripleLine(wl, A1_1, A1_2, A2_1, A2_2, A3_1, A3_2, z1, z2, o1_1, \
               o1_2, o2_1, o2_2, o3_1, o3_2, mu1, mu2, mu3):
    """
    
    This is a model with all its parameters as free. It is for
    fitting: 3 Emission line, 2 Gaussians.
    
    Params
    ------
        wl: wavelength
        Ai_j : the ith component flux of the jth gaussian
        z1, z2 : redshift of each gaussian component
        oi_j : the ith component sigma of the jth gaussian
        mu1, mu2, mu3 : center of each emission line
    
    Output
    ------
        line1 + line2 + line3 : sum of the three double gaussians
    """

    line1 = freeDoubleComp(wl, z1, z2, A1_1, A1_2, o1_1, o1_2, mu1)
    line2 = freeDoubleComp(wl, z1, z2, A2_1, A2_2, o2_1, o2_2, mu2)
    line3 = freeDoubleComp(wl, z1, z2, A3_1, A3_2, o3_1, o3_2, mu3)
    
    return line1 + line2 + line3


def freeDoubleLine2CompFit(data, z, mu1, mu2):
    """
    
    Free double component gaussian fit for two consecutive emission lines.
    
    Params
    ------
        data : DataFrame of the object
        z : redshift
        mu1 : rest-frame wavelength of the first line (Lower wavelength)
        mu2 : rest-frame wavelength of the second line (Higher wavelength)
    
    Output
    ------
        par : nupmy.ndarray with the values of the optimal parameters
        std : numpy.ndarray with the uncertainties of each parameter
    """
    
    wl = data['Wavelength']
    fl = data['Flux']
    err = data['Noise']
    maxFl = max(fl)/2

    par, cov = so.curve_fit(f = lambda wl, A1_1, A1_2, A2_1, A2_2,
                            z1, z2, o1_1, o1_2, o2_1, o2_2:
                            doubleLine(wl, A1_1, A1_2, A2_1, A2_2,
                            z1, z2, o1_1, o1_2, o2_1, o2_2, 
                                       mu1=mu1, mu2=mu2),
                            xdata=wl, ydata=fl,
                            p0=[maxFl, maxFl/2, maxFl, 
                                maxFl/2, z, z, 1, 3, 1, 3],
                            sigma=err, bounds=(0, np.inf))
    std = np.sqrt(np.diag(cov))
    return par, std

def freeTripleLine2CompFit(data, z, mu1, mu2, mu3):
    """
    
    Free double component gaussian fit for three consecutive
    emission lines.
    
    Params
    ------
        data : DataFrame of the object
        z : redshift
        mu1 : rest-frame wavelength of the first line (Lower wavelength)
        mu2 : rest-frame wavelength of the second line (Middle wavelength)
        mu3 : rest-frame wavelength of the third line (Higher wavelength)
        
    Output
    ------
        par : nupmy.ndarray with the values of the optimal parameters
        std : numpy.ndarray with the uncertainties of each parameter
    """
    
    wl = data['Wavelength']
    fl = data['Flux']
    err = data['Noise']
    maxFl = max(fl)/2

    par, cov = so.curve_fit(f = lambda wl, A1_1, A1_2, A2_1, A2_2,
                            A3_1, A3_2, z1, z2, o1_1, o1_2, o2_1, 
                            o2_2, o3_1, o3_2: tripleLine(wl, A1_1, 
                            A1_2, A2_1, A2_2, A3_1, A3_2, z1, z2,
                            o1_1, o1_2, o2_1, o2_2, o3_1, o3_2,
                            mu1=mu1, mu2=mu2, mu3=mu3),
                            xdata=wl, ydata=fl,
                            p0=[maxFl/4, maxFl/4, maxFl, maxFl/2, maxFl/4, 
                                maxFl/4, z, z, 1, 3, 1, 3, 1, 3],
                            sigma=err, bounds=(0, np.inf))
    std = np.sqrt(np.diag(cov))
    return par, std

#########################################################################

"""

Other functions from 2022
"""

def lineOutput(sourceName, I, I_err, I_label):
    """
    
    This function process the line intensities values and error,
    producing a file containing the relevant information.
    
    Params
    ------
        sourceName : name of the galaxy
        I : array of emission line intensities
        I_err : array of uncertainties of intensities
        I_label : name of the line at which the intensities correspond
        
    Output
    ------
        None
    """
    
    import os
    df = {}
    df['ID'] = sourceName
    for i in range(len(I)):
        df[I_label[i]] = I[i]
        df['e' + I_label[i]] = I_err[i]
    out = pd.DataFrame(df, index=[0])
    out.set_index('ID')
    path = os.getcwd()
    out.to_csv(path + '/' + str(sourceName) + '.csv', 
               float_format=np.float64)
    print('File saved at ' + path)


def mergeSpec(wl1, fl1, err1, wl2, fl2, err2):
    """
    Takes two spectrum and gives one as a result of merging the two inputs, 
    considering overlap. It assumes that wl1 has lower wavelengths than wl2.
    It computes the error weighted mean between fluxes in the overlapping
    zone.
    
    Parameters
    ----------
        wl1, wl2 : ndarray
            Array of wavelengths of each spectrum
        fl1, fl2 : ndarray
            Array of fluxes of each spectrum
        err1, err2 : ndarray
            Array of uncertainties of each spectrum

    Output
    ------
        wl : ndarray
            Array of new wavelengths
        fl : ndarray
            Array of new fluxes
        err : ndarray
            Array of new errors.
    """
    mask_wl1 = wl1 > wl2[0]
    mask_wl2 = wl2 < wl1[-1]
    wl = np.concatenate((wl1[wl1 < wl2[0]], wl2))
    
    new_fl, new_err = spectres(wl2[mask_wl2], wl1[mask_wl1],
                              fl1[mask_wl1], err1[mask_wl1],
                              verbose=False)
    
    new_fl2 = unumpy.uarray(fl2[mask_wl2], err2[mask_wl2])
    new_fl = unumpy.uarray(new_fl, new_err)
    
    num = (new_fl * unumpy.std_devs(new_fl)**-2) + \
             (new_fl2 * unumpy.std_devs(new_fl2)**-2)
    
    den = unumpy.std_devs(new_fl)**-2 + unumpy.std_devs(new_fl2)**-2
    res_fl = unumpy.nominal_values(num/den)
    res_err = unumpy.std_devs(num/den)
    
    fl = np.concatenate((fl1[wl1 < wl2[0]], res_fl, fl2[wl2 > wl1[-1]]))
    err = np.concatenate((err1[wl1 < wl2[0]], res_err, err2[wl2 > wl1[-1]]))
    
    return wl, fl, err

def air(vac):
    '''
    
    Convert vacuum wavelength into air wavelengths.
    
    Params
    ------
    vac : float or ndarray
        Vacuum wavelength
    
    Return
    ------
    output : float or ndarray
        Air wavelength
    
    '''
    a = 1.0 + 2.735182e-4
    b = 131.4182 / (vac**2)
    c = 2.76249e8 / (vac**4)
    return vac / (a + b + c)


def gaussian(x, A, mu, z, sigma):
    '''
    Single gaussian with redshifted center.
    '''
    amplitude = A / np.sqrt(2 * np.pi) / sigma
    exp = np.exp(-(x - mu * (1 + z))**2 / 2 / sigma**2 )
    return amplitude * exp


def gaussian2(x, A, mu, sigma):
    '''
    Single classical gaussian model.
    '''
    amplitude = A / np.sqrt(2 * np.pi) / sigma
    exp = np.exp(-(x - mu)**2 / 2 / sigma**2 )
    return amplitude * exp


def gaussianC(x, A, mu, sigma, c):
    '''
    Single classical gaussian model plus continuum
    level for stellar spectra.
    '''
    amplitude = A / np.sqrt(2 * np.pi) / sigma
    exp = np.exp(-(x - mu)**2 / 2 / sigma**2 )
    return amplitude * exp + c


def to_vel(wl_array, z, wl_line):
    '''
    Function to convert from wavelength to velocities
    '''
    c = 299792.458 # in km/s
    v_c = wl_array / ((1 + z) * wl_line)
    return c * v_c

def FWHM(sigma, wl):
    '''
    Function to convert from sigma to FWHM
    '''
    c = 299792.458 # in km/s
    fwhm = c * (2.354 * sigma) / wl
    return fwhm

def dV(z1, z0):
    '''
    Function to study the velocity difference between 
    broad (z1) and narrow (z0) component
    '''
    c = 299792.458 # in km/s
    dZ = (z1 - z0) / (1 + z0)
    return c * dZ


def save_catalog(galname, catalog, path):
    '''
    Function to save and/or update the emission line, equivalent width and
    velocity catalogs taken from the fluxes notebooks for each galaxy.
    '''
    
    catalog_exist = os.path.exists(path)
    if catalog_exist:
        print('Existent catalog (.csv) file.')

        catalog_all = pd.read_csv(path, on_bad_lines='skip')
        if np.sum(catalog_all['galname'] == galname) == 1:
            print('Existent data for ' + galname + ', overwriting...')
            index = np.where(catalog_all['galname'] == galname)[0][0]
            catalog_all.loc[index] = np.array(catalog.loc[0])
            catalog_all.to_csv(path, index=None)
            print('Done.')

        else: 
            print('Non existent data for ' + galname + ', adding new data...')
            catalog.to_csv(path, mode='a',
                         header=False, index=None)
            print('Done.')
    else:
        print('File does not exist, creating new catalog...')
        catalog.to_csv(path, sep=',', index=None)
        print('Done.')
        
        

        

import uncertainties.umath as uum
from uncertainties import ufloat
from uncertainties import unumpy as upy

# Mock unc float to make tests
unc_test = ufloat(1, 0.1)
unc_test_func = upy.log10([unc_test, unc_test])

def is_unc(obj):
    '''
    Return True if the obj is an uncertainty array and False if not.
    '''
    try:
        boolean = (type(obj[0]) == type(unc_test))
        if not boolean:
            boolean = (type(obj[0]) == type(unc_test_func[0]))
    except:
        boolean = (type(obj) == type(unc_test))
        if not boolean:
            boolean = (type(obj) == type(unc_test_func[0]))
    return boolean


        
def TO3(oiiia, oiiib, oiiic):
    RO3 = (oiiia + oiiib) / oiiic
    t_OIII = 0.784 - (1.357e-4 * RO3) + (48.44 / RO3)
    return t_OIII


# For the density n([SII]), polynomials as a function of temperature

def a0_t(t):
    return 16.054 - (7.79 / t) - (11.32 * t)


def a1_t(t):
    return -22.66 + (11.08 / t) + (16.02 * t)


def b0_t(t):
    return -21.61 + (11.89 / t) + (14.59 * t)
    
    
def b1_t(t):
    return 9.17 - (5.09 / t) - (6.18 * t)

# For the low excitation zone temperature [OII], polynomials as a function of
# the density


def a0_n(n):
    return 0.2526 - (3.57e-4 * n) - (0.43 / n)


def a1_n(n):
    return 1.36e-3 + (5.42e-6 * n) + (4.81e-3 / n)


def a2_n(n):
    return 35.624 - (0.0172 * n) + (25.12 / n)


def ne(siia, siib, to3):
    RS2 = siib / siia
    ne = 1e3 * ((RS2 * a0_t(to3)) + a1_t(to3)) / ((RS2 * b0_t(to3)) + b1_t(to3))
    #if np.sum(upy.nominal_values(ne) < 0) > 0:
    #    print('WARNING: Negative electron density is unphysical.')
    return ne


def TO2(oiia, oiib, oiic, oiid, n, IO2_R):
    # Assuming the correction is already multiplied by Hb
    den = (oiia + oiib) - IO2_R
    RO2 = (oiic + oiid) / den
    return a0_n(n) + (a1_n(n) * RO2) + (a2_n(n) / RO2)


def TN2(niia, niib, niic, IN2_R):
    den = (niic - IN2_R)
    RN2 = (niia + niib) / den
    t_NII = 0.6153 - (1.529e-4 * RN2) + (35.3641 / RN2)
    return t_NII


def TS3(siiia, siiib, siiic):
    RS3 = (siiia + siiib) / siiic
    tS3 = 0.5147 + (3.187e-4 * RS3) + (23.64041 / RS3)
    return tS3


def calzetti2000(wave, params = [1e4, 2.659, 4.05]):
    '''
    Calzetti extinction curve, function defined to calculate it for single
    values
    
    Params
    ------
    wave : float
        Wavelength to calculate the extinction
    params : list of floats (optional)
        Parameters to define the curve as PyNeb does. The first element is
        normalization to go from Angstrom to microns. The second parameter
        is 2.569 that multiplies everything and the last one is Rv, that is
        equal to 4.05 in the case of starburst galaxies.
        
    Output
    ------
    ext : float
        Extinction in magnitudes of the input wavelength.
    '''
    wl = wave / params[0] # from angstrom to microns
    if wl >= 0.63:
        ext = params[1] * (-1.857 + (1.040 / wl)) + params[2]
    elif wl < 0.63:
        ext = params[1] * (-2.156 + (1.509 / wl) - (0.198 / (wl**2)) \
                           + (0.011 / (wl**3)) ) + params[2]
    return ext





# OXYGEN

def single_log_OH(oiic, oiid, hb, tl, n):
    if is_unc(oiic) or is_unc(oiid) or is_unc(hb) or is_unc(tl) or is_unc(n):
        oii = oiic + oiid
        log = upy.log10(oii / hb)
        s1 = 5.887 + (1.641 / tl) - (0.543 * upy.log10(tl)) + (1.14e-4 * n)
        return log + s1
    else:
        oii = oiic + oiid
        log = np.log10(oii / hb)
        s1 = 5.887 + (1.641 / tl) - (0.543 * np.log10(tl)) + (1.14e-4 * n)
        return log + s1


def double_log_OH(oiiia, oiiib, hb, th):
    if is_unc(oiiia) or is_unc(oiiib) or is_unc(hb) or is_unc(th):
        oiii = oiiia + oiiib
        log = upy.log10(oiii / hb)
        s1 = 6.1868 + (1.2491 / th) - (0.5816 * upy.log10(th))
        return log + s1
    else:
        oiii = oiiia + oiiib
        log = np.log10(oiii / hb)
        s1 = 6.1868 + (1.2491 / th) - (0.5816 * np.log10(th))
        return log + s1




# HELIUM

def emissivities(th, ne):
    F4471 = (2.0301 + (1.5e-5 * ne)) * (th**(0.1463 - (0.0005 * ne)))
    F5875 = (0.745 - (5.1e-5 * ne)) * (th**(0.226 - (0.0011 * ne)))
    F6678 = (2.612 - (1.46e-4 * ne)) * (th**(0.2355 - (0.0016 * ne)))
    F7065 = (4.329 - (2.4e-3 * ne)) * (th**(-0.368 - (0.0017 * ne)))
    return F4471, F5875, F6678, F7065


def optical_depth_function(th, ne, tau, line='AAAA'):
    if line == '4471':
        term = (tau / 2) * (0.00274 + (th * (8.81e-4 - (1.21e-6 * ne))))
        return 1 + term
    elif line == '5875':
        term = (tau / 2) * (0.00470 + (th * (2.23e-3 - (2.51e-6 * ne))))
        return 1 + term
    elif line == '6678':
        return 1
    elif line == '7065':
        term = (tau / 2) * (0.359 + (th * (-3.46e-2 - (1.84e-4 * ne) + (3.039e-7 * (ne**2)))))
        return 1 + term


def yplus(hei, hb, F, f):
    ratio = hei / hb
    em_od = F / f
    return ratio * em_od


def y2plus(heii_x, hb, th):
    ratio = heii_x / hb
    y2 = ratio * 0.0416 * (th**(-0.146))
    return y2




# SULFUR

def single_log_S(siia, siib, hb, tl):
    if is_unc(siia) or is_unc(siib) or is_unc(hb) or is_unc(tl):
        sii = siia + siib
        log = upy.log10(sii / hb)
        s1 = 5.463 + (0.941 / tl) - (0.37 * upy.log10(tl))
        return log + s1
    else:
        sii = siia + siib
        log = np.log10(sii / hb)
        s1 = 5.463 + (0.941 / tl) - (0.37 * np.log10(tl))
        return log + s1


def double_log_S(siiia, siiib, hb, tm):
    if is_unc(siiia) or is_unc(siiib) or is_unc(hb) or is_unc(tm):
        siii = siiia + siiib
        log = upy.log10(siii / hb)
        s1 = 5.983 + (0.661 / tm) - (0.527 * upy.log10(tm))
        return log + s1
    else:
        siii = siiia + siiib
        log = np.log10(siii / hb)
        s1 = 5.983 + (0.661 / tm) - (0.527 * np.log10(tm))
        return log + s1


def double_log_S_noline(siiic, hb, tm):
    if is_unc(siiic) or is_unc(hb) or is_unc(tm):
        log = upy.log10(siiic / hb)
        s1 = 6.695 + (1.664 / tm) - (0.513 * upy.log10(tm))
        return log + s1
    else:
        log = np.log10(siiic / hb)
        s1 = 6.695 + (1.664 / tm) - (0.513 * np.log10(tm))
        return log + s1


def ICF_S(O2plus, Oplus, alpha=3.27):
    term = O2plus / (Oplus + O2plus)
    return (1 - (term**alpha))**(-1/alpha)




# NITROGEN

def log_NH_12(niia, niib, hb, tl):
    if is_unc(niia) or is_unc(niib) or is_unc(hb) or is_unc(tl):
        n_ratio = (niia + niib) / hb
        log = upy.log10(n_ratio)
        s1 = 6.291 + (0.90221 / tl) - (0.5511 * upy.log10(tl))
        return log + s1
    else:
        n_ratio = (niia + niib) / hb
        log = np.log10(n_ratio)
        s1 = 6.291 + (0.90221 / tl) - (0.5511 * np.log10(tl))
        return log + s1


def ICF_N(Oplus, O):
    return (O / Oplus)


def log_NO(niia, oiic, oiid, tl):
    if is_unc(niia) or is_unc(oiic) or is_unc(oiid) or is_unc(tl):
        oii = oiic + oiid
        log = upy.log10(niia / oii)
        s1 = 0.493 - (0.025 * tl) - (0.687 / tl) + (0.1621 * upy.log10(tl))
        return log + s1
    else:
        oii = oiic + oiid
        log = np.log10(niia / oii)
        s1 = 0.493 - (0.025 * tl) - (0.687 / tl) + (0.1621 * np.log10(tl))
        return log + s1




# NEON 

def log_Ne_12(neiiia, hb, th):
    if is_unc(neiiia) or is_unc(hb) or is_unc(th):
        log = upy.log10(neiiia / hb)
        s1 = 6.947 + (1.614 / th) - (0.4291 * upy.log10(th))
        return log + s1
    else:
        log = np.log10(neiiia / hb)
        s1 = 6.947 + (1.614 / th) - (0.4291 * np.log10(th))
        return log + s1

def ICF_Ne(O2plus, Oplus):
    x = O2plus / (O2plus + Oplus)
    pol = 0.753 + (0.142 * x) + (0.171 / x)
    return pol





# ARGON

def double_log_Ar(ariiia, hb, tm):
    if is_unc(ariiia) or is_unc(hb) or is_unc(tm):
        log = upy.log10(ariiia / hb)
        s1 = 6.1 + (0.86 / tm) - (0.404 * upy.log10(tm))
        return log + s1
    else:
        log = np.log10(ariiia / hb)
        s1 = 6.1 + (0.86 / tm) - (0.404 * np.log10(tm))
        return log + s1
    

def triple_log_Ar(ariva, hb, th):
    if is_unc(ariva) or is_unc(hb) or is_unc(th):
        log = upy.log10(ariva / hb)
        s1 = 6.306 + (1.232 / th) - (0.703 * upy.log10(th))
        return log + s1
    else:
        log = np.log10(ariva / hb)
        s1 = 6.306 + (1.232 / th) - (0.703 * np.log10(th))
        return log + s1


def ICF_double_Ar(O2plus, Oplus):
    x = O2plus / (O2plus + Oplus)
    pol = 0.596 + (0.967 * (1 - x)) + (0.077 / (1 - x))
    return pol

def ICF_double_triple_Ar(O2plus, Oplus):
    x = O2plus / (O2plus + Oplus)
    pol = 0.928 + (0.364 * (1 - x)) + (0.006 / (1 - x))
    return pol



# IRON

def log_Fe_12(feiiia, hb, th):
    if is_unc(feiiia) or is_unc(hb) or is_unc(th):
        log = upy.log10(feiiia / hb)
        s1 = 6.288 + (1.408 / th) - (0.203 * upy.log10(th))
        return log + s1
    else:
        log = np.log10(feiiia / hb)
        s1 = 6.288 + (1.408 / th) - (0.203 * np.log10(th))
        return log + s1

def ICF_Fe(O2plus, Oplus):
    x = O2plus / Oplus
    return (x**(-0.09)) * (1 + x)

# IONIZATION PARAMETER

def log_q(oiiia, oiid, log_OH, U=False):
    sol = 29979245800.0 # in cm/s
    
    if is_unc(oiiia) or is_unc(oiid) or is_unc(log_OH):
        y = upy.log10(oiiia / oiid)
    else:
        y = np.log10(oiiia / oiid)
    
    ip1 = 32.81 - (1.153 * (y**2)) + (log_OH * (-3.396 - (0.025 * y) + (0.1444*(y**2))))
    ip2 = 4.603 - (0.3119 * y) - (0.163 * (y**2)) + (log_OH * (-0.48 + (0.0271 * y) + (0.02037 * (y**2))))
    
    log_ip = ip1 / ip2
    
    # q = cU
    # U = q / c
    if U:
        if is_unc(oiiia) or is_unc(oiid) or is_unc(log_OH):
            log_U = upy.log10((10**log_ip) / sol)
        else:
            log_U = np.log10(((10**log_ip) / sol).astype(float))
            return log_U
    else:
        return log_ip
        