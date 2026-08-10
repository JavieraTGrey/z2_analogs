"""
Created on Mon Nov 7 11:42 2022

@author: bnavarrete

This script takes a galaxy spectrum and explore the velocity
profile of the lines for data reduced with pypeit.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
magePath = '/home/benjamin/Documents/analogs/'
os.sys.path.append(magePath)
from magE.functions import air, gaussian2, gaussianC

## Light speed
c = 299792.458 # km/s

## Used ionization lines for each velocity profile

low_ion = ['Hg', 'Hb', '[OI]6300', 'Ha', '[SII]6716', '[SII]6730']
high_ion = ['[NeIII]3869', '[OIII]4363', '[OIII]4959', '[OIII]5007']

## Wavelengths (air) from NIST

low_ion_wl = [4340.472, 4861.363, 6300.304, 6562.801, 6716.44, 6730.82]
high_ion_wl = [3868.76, 4363.209, 4958.911, 5006.843]

## Functions


def dataframe(wave, flux, noise):
    '''
    Recieves a wavelenght, flux, and noise array and
    return a dataframe with this information.
    '''
    dF = pd.DataFrame({'wave': wave,
                        'flux': flux,
                        'noise': noise})
    return dF


def cut_line(line_wl, spec, z=0, window=30):
    
    line_center = line_wl * (1 + z)
    mask_low = (spec['wave'] > line_center - window)
    mask_high = (spec['wave'] < line_center + window)
    mask_line = mask_low & mask_high
    wl = spec['wave'][mask_line]
    fl = spec['flux'][mask_line]
    err = spec['noise'][mask_line]
    return dataframe(wl, fl, err)


def fit_line(line_wl, spec, line_width, z=0):
    '''
    Fit the line and gives its center wavelength
    
    Return: float
    '''
    line_center = line_wl * (1 + z)
    pp, pcov = curve_fit(gaussianC, xdata=spec['wave'], ydata=spec['flux'],
                         p0=[np.max(spec['flux']), line_center, line_width,
                            1e-16],
                         sigma=spec['noise'])
    return pp[1], pp[3]


def wl_to_vel(spec, center, cont):
    '''
    Recieves a spectrum dataset of a line, the located center
    of the line and an estimation of the continuum. Transforms
    the wavelengths into velocity an gives a continuum subtracted
    flux array.
    
    Return: velocity and flux data for the emission line.
    '''
    wl = spec['wave']
    v = c * (wl - center) / center
    flux = spec['wave'] - cont
    return v, flux


def resample_and_normalize(vel, flux):
    '''
    Takes a set of velocities and fluxes and resample both
    arrays into a uniform velocity array. This is necessary to
    take means in the averaged profile. It also normalizes
    the lines for consistency.
    
    Return: resampled velocities and fluxes
    '''
    newvel = np.arange(-600, 600, 20)
    new_flux = []
    for i in range(len(newvel)):
        bot = newvel[i]-10
        top = newvel[i]+10
        new_flux.append(np.nanmean(flux[(vel > bot) & (vel < top)]/max(flux)))
    return newvel, np.array(new_flux)


if __nam
