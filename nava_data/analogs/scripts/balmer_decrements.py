# Created on Mon Dec 19th, 2023
# @author: bnavarrete

# This script uses Balmer decrements to find the extinction normalization
# E(B-V) and then correct the emission lines according to such extinction.
# Both Calzetti et al. 2000 and the SMC are used.

import numpy as np
import pandas as pd
import pyneb as pn
import corner
import matplotlib.pyplot as plt
from astropy.io import fits
import uncertainties.unumpy as upy
from uncertainties import ufloat
from lmfit import minimize, Parameters

import os
import sys
magePath = '/home/benjamin/Documents/analogs/'
os.sys.path.append(magePath)

from magE.functions import save_catalog
from magE.plotutils import * 
from magE.constutils import *

# =============================================================================
#
# Global data, variables, etc.
#
# =============================================================================

# Set fixed random seed to not vary the fits
np.random.seed(286)

# If correcting the total, narrow or broad fluxes.
mode = 'lines'

# Set the 

# Read the data
# WE NEED TO USE THIS FILE BECAUSE IT HAS THE FLUXES WITHOUT DUST CORRECTION
el_cat = pd.read_csv(magePath+'results/em_'+mode+'.csv')

# Intrinsic Balmer decrements
# Fixed for all galaxies. Assuming Case B, ne = 100 cm^-3 and Te = 10000 K
# It is ordered by wavelength, the last element is Halpha
R_int = np.array([0.259, 0.468, 1.0, 2.86])

# Path of the catalog with normalizations
path_norm = magePath + '/results/dust_norm_cal_lines.csv'

# =============================================================================
#
# Functions
#
# =============================================================================


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


def calzetti2000_arr(wave, params = [1e4, 2.659, 4.05]):
    '''
    Same as calzetti2000, but this one is made to calculate it for arrays.
    '''
    wl = wave / params[0] # from angstrom to microns
    mask_up = (0.63 <= wl)
    mask_down = (0.63 > wl)
    ext_up = params[1] * (-1.857 + (1.040 / wl[mask_up])) + params[2]
    ext_down = params[1] * (-2.156 + (1.509 / wl[mask_down]) \
                            - (0.198 / (wl[mask_down]**2)) \
                            + (0.011 / (wl[mask_down]**3)) ) + params[2]
    return np.concatenate((ext_down, ext_up))


def SMC(wave):
    return wave


# =============================================================================
#
# Program
#
# =============================================================================

if __name__ == '__main__':
    
    # Extinction values with respect to Hb for Calzetti ext. curve
    x_cal = [calzetti2000(wave_cat['Hd']) - calzetti2000(wave_cat['Hb']),
             calzetti2000(wave_cat['Hg']) - calzetti2000(wave_cat['Hb']), 
             calzetti2000(wave_cat['Hb']) - calzetti2000(wave_cat['Hb']), 
             calzetti2000(wave_cat['Ha']) - calzetti2000(wave_cat['Hb'])]
    x_cal = np.array(x_cal)
    
    # Extinction values with respect to Hb for SMC ext. curve
    x_SMC = []
    x_SMC = np.array(x_SMC)
    
    # For each galaxy in our sample
    for galaxy in el_cat['galname']:
        
        
        '''
        # If there is already a catalog containing the reddening constant
        # for our galaxies, just apply the correction
        if os.path.exists(path_norm):
            print('Existent dust reddening factor, correcting flux...')
            norm_cat = pd.read_csv(path_norm)
            
            # Locate the index of the galaxy in the dF
            index = np.argmax(el_cat['galname'] == galaxy)
            
            # Dust reddening constant
            E_BV_cal = norm_cat.iloc[index]['E_BV']
            
            # Dictionary to store the dust corrected emission line fluxes
            dust_corr_lines_cal = {'galname': galaxy}
            
            # For each line available
            for line in el_cat.keys()[1:][::2]:

                # Read the line
                f_obs = ufloat(el_cat.iloc[index][line],
                               el_cat.iloc[index][line+'_err'])

                # Apply the dust correction
                f_int_cal = f_obs * 10**(0.4 * E_BV_cal \
                                     * calzetti2000(wave_cat[line]))
                #f_int_SMC = f_obs * 10**(0.4 * E_BV_SMC \
                #                     * SMC(wave_cat[line]))

                # Save the corrected fluxes
                dust_corr_lines_cal[line] = f_int_cal.nominal_value
                
                # If the flux is an upper limit, keep it as dust corrected
                # upper limit
                if f_obs.std_dev == 0.0:
                    dust_corr_lines_cal[line+'_err'] = 0.0
                else:
                    dust_corr_lines_cal[line+'_err'] = f_int_cal.std_dev

                #dust_corr_lines_SMC[line] = f_int_SMC.nominal_value
                #dust_corr_lines_SMC[line+'_err'] = f_int_SMC.std_dev

            # Convert to dataframe
            dust_corr_lines_cal_pd = pd.DataFrame(data=dust_corr_lines_cal,
                                                  index=[0])
            #dust_corr_lines_SMC_pd = pd.DataFrame(data=dust_corr_lines_SMC,
            #                                      index=[0])

            # Saving catalog
            save_catalog(galaxy, dust_corr_lines_cal_pd,
                         magePath + '/results/em_'+mode+'_corr_cal.csv')
            #save_catalog(galaxy, dust_corr_lines_SMC_pd,
            #             magePath + '/results/em_'+mode+'_corr_SMC.csv')

            print('Saved corrected lines.')
            print('==============================================================')
            '''
        #else:
        # Locate the index of the galaxy in the dF
        index = np.argmax(el_cat['galname'] == galaxy)

        # Balmer line fluxes observed
        balmer_obs = upy.uarray([el_cat.iloc[index]['Hd'],
                                el_cat.iloc[index]['Hg'],
                                el_cat.iloc[index]['Hb'],
                                el_cat.iloc[index]['Ha']],
                               [el_cat.iloc[index]['Hd_err'],
                                el_cat.iloc[index]['Hg_err'],
                                el_cat.iloc[index]['Hb_err'],
                                el_cat.iloc[index]['Ha_err']])

        # Hb flux, to calculate the ratios
        hb = ufloat(el_cat.iloc[index]['Hb'], el_cat.iloc[index]['Hb_err'])

        # Balmer decrements
        R_obs = balmer_obs / hb
        y = upy.log10(R_obs / R_int)


        # Initialize fitting
        # Initial guess for the slope of the decrements
        mcal_init = (upy.nominal_values(y)[-1] - upy.nominal_values(y)[0]) \
                    / (x_cal[-1] - x_cal[0])
        #mSMC_init = (upy.nominal_values(y)[-1] - upy.nominal_values(y)[0]) \
        #            / (x_SMC[-1] - x_SMC[0])

        # Fit parameters
        params_cal = Parameters()
        params_cal.add('m', value=mcal_init)

        #params_SMC = Parameters()
        #params_SMC.add('m', value=mSMC_init)

        # Function to fit
        def dust_fit(params_dust, x, data, uncertainty):
            m = params_dust['m']
            model = x * m
            return (upy.nominal_values(y) - model) / upy.std_devs(y)

        print('Fitting Balmer decrements for '+galaxy+'...')
        print('Using Calzetti extinction curve.')
        out_cal = minimize(dust_fit, params_cal, args=(x_cal, 1, 1),
                              method='emcee', burn=1500, steps=2500, thin=10)
        #print('Using SMC extinction curve.')
        #out_SMC = minimize(dust_fit, params_SMC, args=(x_SMC, 1, 1),
        #                      method='emcee', burn=1500, steps=2500, thin=10)

        # Resulting slope
        mcal = out_cal.flatchain.m.mean()
        mcal_std = out_cal.flatchain.m.std()
        print('Value of the slope from the fit')
        print(mcal)

        #mSMC = out_SMC.flatchain.m.mean()
        #mSMC_std = out_SMC.flatchain.m.std()

        # Normalizations
        E_BV_cal = ufloat(mcal, mcal_std) / -0.4
        #E_BV_SMC = ufloat(mSMC, mSMC_std) / -0.4

        cHb_cal = 0.4 * E_BV_cal * calzetti2000(wave_cat['Hb'])
        #cHb_SMC = 0.4 * E_BV_SMC * SMC(wave_cat['Hb'])

        print('Saving figure...')
        texPlot('on')
        plt.figure()
        plt.errorbar(x_cal, upy.nominal_values(y), yerr=upy.std_devs(y),
                    marker='o', ls='', color='black', capsize=3)
        #plt.errorbar(x_SMC, upy.nominal_values(y), yerr=upy.std_devs(y),
        #            marker='o', ls='', color='black', label='SMC')
        plt.plot(x_cal, mcal * x_cal, ls='--', color='red')
                 #label=r'$E(B-V)_{Cal}  = $'+str(round(E_BV_cal.nominal_value,
                 #                                      3))+' ± '\
                 #+str(round(E_BV_cal.std_dev, 3)))
        #plt.plot(x_SMC, mSMC * x_SMC, ls='--', color='red',
        #         label=r'$E(B-V)_{SMC}  = $'+str(round(E_BV_SMC.nominal_value,
        #                                               3))+' ± '\
        #         +str(round(E_BV_SMC.std_dev, 3)))
        plt.ylabel(r'log(R$_{obs}$/R$_{int}$)')
        plt.xlabel(r'$k(\lambda) - k($H$\beta)$')
        plt.title(r'$E(B-V)_{Cal}  =  $'+str(round(E_BV_cal.nominal_value,
                                                       3))+' ± '\
                 +str(round(E_BV_cal.std_dev, 3)))
        plt.savefig(magePath+'imges/decrements/'+galaxy+'_'+mode+'.pdf',
                   bbox_inches='tight')
        plt.close()

        # Print results
        print('Calzetti E(B-V) = '+str(round(E_BV_cal.nominal_value,
                                             3))+' ± '\
                 +str(round(E_BV_cal.std_dev, 3)))
        #print('SMC E(B-V) = '+str(round(E_BV_SMC.nominal_value, 3))+' ± '\
        #         +str(round(E_BV_SMC.std_dev, 3)))

        # Dictionary to store the dust corrected emission line fluxes and
        # extinction normalizations
        dust_corr_lines_cal = {'galname': galaxy}
        #dust_corr_lines_SMC = {'galname': galaxy}
        dust_norm_cal = {'galname': galaxy,
                         'E_BV': E_BV_cal.nominal_value,
                         'E_BV_err': E_BV_cal.std_dev,
                         'cHb': cHb_cal.nominal_value,
                         'cHb_err': cHb_cal.std_dev}
        #dust_norm_SMC = {'galname': galaxy,
        #                 'E_BV': E_BV_SMC.nominal_value,
        #                 'E_BV_err': E_BV_SMC.std_dev,
        #                 'cHb': cHb_SMC.nominal_value,
        #                 'cHb_err': cHb_SMC.std_dev}

        # For each line available
        for line in el_cat.keys()[1:][::2]:

            # Read the line
            f_obs = ufloat(el_cat.iloc[index][line],
                           el_cat.iloc[index][line+'_err'])

            # Check if there is data for this line
            if f_obs.std_dev == 0.0:
                f_int_cal = f_obs
                #f_int_SMC = f_obs

            # If not, make the dust correction
            else:
                f_int_cal = f_obs * 10**(0.4 * E_BV_cal \
                                     * calzetti2000(wave_cat[line]))
                #f_int_SMC = f_obs * 10**(0.4 * E_BV_SMC \
                #                     * SMC(wave_cat[line]))

            # Save the corrected fluxes
            dust_corr_lines_cal[line] = f_int_cal.nominal_value
            dust_corr_lines_cal[line+'_err'] = f_int_cal.std_dev

            #dust_corr_lines_SMC[line] = f_int_SMC.nominal_value
            #dust_corr_lines_SMC[line+'_err'] = f_int_SMC.std_dev

        # Convert to dataframe
        dust_corr_lines_cal_pd = pd.DataFrame(data=dust_corr_lines_cal,
                                              index=[0])
        #dust_corr_lines_SMC_pd = pd.DataFrame(data=dust_corr_lines_SMC,
        #                                      index=[0])
        dust_norm_cal_pd = pd.DataFrame(data=dust_norm_cal, index=[0])
        #dust_norm_SMC_pd = pd.DataFrame(data=dust_norm_SMC, index=[0])

        # Saving catalog
        save_catalog(galaxy, dust_corr_lines_cal_pd,
                     magePath + '/results/em_'+mode+'_corr_cal.csv')
        #save_catalog(galaxy, dust_corr_lines_SMC_pd,
        #             magePath + '/results/em_'+mode+'_corr_SMC.csv')

        save_catalog(galaxy, dust_norm_cal_pd,
                     magePath + '/results/dust_norm_cal_'+mode+'.csv')

        #save_catalog(galaxy, dust_norm_SMC_pd,
        #             magePath + '/results/dust_norm_SMC_'+mode+'.csv')

        print('Saved corrected lines.')
        print('==============================================================')