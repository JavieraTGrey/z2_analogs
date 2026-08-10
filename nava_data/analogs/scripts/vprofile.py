"""
Created on Thu Oct 6 16:04 2022

@author: bnavarrete

This script takes a galaxy spectrum and explore the velocity
profile of the lines. 
"""

import os
import sys
magePath = '/home/benjamin/Documents/analogs/'
os.sys.path.append(magePath)
from magE.functions import air, gaussian2, gaussianC
from magE.spec.spec_classes import MageMultispec, SimpleSpectrum

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

## Light speed
c = 299792.458 # km/s

## Used ionization lines for each velocity profile

low_ion = ['Hg', 'Hb', '[OI]6300', 'Ha', '[SII]6716', '[SII]6730']
high_ion = ['[NeIII]3869', '[OIII]4363', '[OIII]4959', '[OIII]5007']

## Wavelengths (air) from NIST

low_ion_wl = [4340.472, 4861.363, 6300.304, 6562.801, 6716.44, 6730.82]
high_ion_wl = [3868.76, 4363.209, 4958.911, 5006.843]

## Functions


  
def locate_orders(line_wl, spec, z=0):
    '''
    This function locate the orders in which there is the
    specified emission line "line_wl".
    
    Return: [order1, order2, ...]
    '''
    line_center = line_wl * (1 + z)
    found_orders = []
    for i in range(spec.n_orders):
        order = i + spec.first_order
        mask_low = line_center > spec.wavelength(order)[0]
        mask_high = line_center < spec.wavelength(order)[-1]
        inside = (mask_low & mask_high)
        if inside:
            found_orders.append(order)
    return found_orders


def cut_line(line_wl, spec, orders, z=0, window=36):
    '''
    Takes a spec and prepares the data for making the velocity
    profile.
    
    Return: list of SimpleSpectrum object
    '''
    line_center = line_wl * (1 + z)
    specs = []
    for order in orders:
        mask_low = (spec.wavelength(order) > line_center - window)
        mask_high = (spec.wavelength(order) < line_center + window)
        wl = spec.wavelength(order)[mask_low & mask_high]
        flux = spec.flux(order)[mask_low & mask_high]
        noise = spec.noise(order)[mask_low & mask_high]
        line_spec = SimpleSpectrum(lda=wl,
                                   flux=flux,
                                   e_flux=noise)
        specs.append(line_spec)
    
    return specs


def signal_to_noise(specs):
    '''
    Recieves a list of spectrums and determine what is the
    one that has highest signal_to_noise.
    INPUT HAS TO BE FROM CUT_LINES.

    Return: integer (index)
    '''
    if len(specs) == 1:
        return 0
    else:
        s_n = []
        for spec in specs:
            s_n.append(np.sum(spec.flux/spec.e_flux))
        return np.argmax(s_n)


def fit_line(line_wl, spec, line_width, z=0):
    '''
    Fit the line and gives its center wavelength
    
    Return: float
    '''
    line_center = line_wl * (1 + z)
    pp, pcov = curve_fit(gaussianC, xdata=spec.lda, ydata=spec.flux,
                         p0=[np.max(spec.flux), line_center, line_width,
                            1e-16],
                         sigma=spec.e_flux)
    return pp[1], pp[3]


def wl_to_vel(spec, center, cont):
    '''
    Recieves a spectrum dataset of a line, the located center
    of the line and an estimation of the continuum. Transforms
    the wavelengths into velocity an gives a continuum subtracted
    flux array.
    
    Return: velocity and flux data for the emission line.
    '''
    wl = spec.lda
    v = c * (wl - center) / center
    flux = spec.flux - cont
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


def vprofile(vels, fluxes):
    '''
    Calculates the averaged velocity profile of a set of emission lines.
    
    Return: array of velocities and average normalized profile.
    '''
    if len(vels) == 1:
        return np.array(vels[0]), np.array(fluxes[0] / max(fluxes[0]))
    else:
        norm_fluxes = []
        vel = np.arange(-600, 600, 20)
        for i in range(len(vels)):
            norm_fluxes.append(resample_and_normalize(vels[i],
                                                      fluxes[i])[1].tolist())
        mean = np.nanmean(np.array(norm_fluxes), axis=0)
        return vel, mean


def plot_vprofile(vels, fluxes, labels, figname):
    '''
    Plot the velocity profiles, for each line in the back
    transparent curves and the mean in the solid line.
    Saves the figure at the end.
    
    Return: None
    '''
    color='purple'
    color_back='blue'
    if 'Ha' in labels:
        color='orangered'
        color_back='orange'
    
    if len(vels) == 1:
        plt.figure(figsize=(8, 6))
        vel, mean = vprofile(vels, fluxes)
        plt.plot(vel, mean, label='Average', color=color)
        plt.axvline(0, ls='--', color='gray')
        plt.xlim(-600, 600)
        plt.legend(frameon=False)
        plt.ylabel('Normalized flux')
        plt.xlabel(r'$v$ [km/s]')
        plt.savefig(figname + '.png', facecolor='white', bbox_inches='tight')
        plt.close()
    else:
        plt.figure(figsize=(8, 6))
        vel, mean = vprofile(vels, fluxes)
        for i in range(len(vels)):
            plt.plot(vels[i], fluxes[i]/max(fluxes[i]), label=labels[i],
                     color=color_back, alpha=0.3)
        plt.plot(vel, mean, label='Average', color=color)
        plt.axvline(0, ls='--', color='gray')
        plt.xlim(-600, 600)
        plt.legend(frameon=False)
        plt.ylabel('Normalized flux')
        plt.xlabel(r'$v$ [km/s]')
        plt.savefig(figname + '.png', facecolor='white', bbox_inches='tight')
        plt.close()


def singleProfile(line_wl, spec, window, z=0):
    '''
    Does the complete location, selection and preparation
    for a single emission line in a galaxy spectrum to let it
    ready for the velocity profile
    
    Return: velocity and flux array data for the line
    '''
    orders = locate_orders(line_wl, spec, z)
    specs = cut_line(line_wl, spec, orders, z, window)
    final_spec = specs[signal_to_noise(specs)]
    center, cont = fit_line(line_wl, final_spec, 1.0, z)
    vel_line, flux_line = wl_to_vel(final_spec, center, cont)
    return vel_line, flux_line


if __name__=='__main__':

    # galaxy parameters
    z = 0.034 # redshift (aprox.)
    
    # read data
    galname = 'j2119+0052'
    dataPath = '/home/benjamin/Documents/analogs/specs/fits/'
    spec = MageMultispec.from_fname(dataPath + galname + '.fits')
    
    window = 36 # aproximated width in Angstrom for isolated lines
    nwindow = 20 # narrow window to avoid moderately close lines
    ewindow = 10 # extremely narrow window to avoid close lines

    # Prepares all the velocity arrays and fluxes to be normalized
    # and averaged to see in the analysis
    
    ## Low ionization lines
    
    vels_low = []
    fluxes_low = []
    for i in range(len(low_ion)):
        if low_ion[i][:5] == '[SII]' or low_ion[i][:4] == '[OI]':
            single_line = singleProfile(low_ion_wl[i], spec,
                                        ewindow, z)
        else:
            single_line = singleProfile(low_ion_wl[i], spec,
                                        window, z)
        vels_low.append(single_line[0])
        fluxes_low.append(single_line[1])
        
    ## High ionization lines
    
    vels_high = []
    fluxes_high = []
    for i in range(len(high_ion)):
        if high_ion[i] == '[OIII]4363':
            single_line = singleProfile(high_ion_wl[i], spec,
                                        nwindow, z)
        else:
            single_line = singleProfile(high_ion_wl[i], spec,
                                        window, z)
        vels_high.append(single_line[0])
        fluxes_high.append(single_line[1])
    
    # Save a plot for each profile
    plot_vprofile(vels_low, fluxes_low, low_ion,
                  'vprofilelow_' + galname)
    plot_vprofile(vels_high, fluxes_high, high_ion,
                  'vprofilehigh_' + galname)
