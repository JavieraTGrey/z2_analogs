"""
Created on Tue May 23, 2023

@author: bnavarre

Update on Wed May 1, 2024
The blue reduction of night 19A now is available thanks to version
1.15.0 of pypeit. The script has been modified to join such results.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from spectres import spectres
from constutils import source_dict

magePath = '/home/benjamin/Documents/analogs/'
os.sys.path.append(magePath)
from magE.functions import air
from magE.plotutils import *

# os.environ["PATH"] += os.pathsep + '/usr/bin'

# run the script typing the script name, night of observation, galaxy name
# and extraction you want to merge and rebin.
# e.g. >> python merge_and_rebin.py 19a J0950 BOX

# ==========================================================================
#
# Useful info
#
# ==========================================================================

limits = {#'J2225': ['18', 4258, 'BOX'], # boxcar
          #'J2215': ['18', 4756, 'BOX'], # boxcar
          #'J0240': ['18', 4922, 'BOX'], # boxcar
          #'J0021': ['18', 4976, 'BOX'], # boxcar
          
          #'J0950': ['19a', 5050, 'BOX'], # boxcar
          'J1146': ['19a', 5100, 'BOX'], # boxcar
          #'J1226': ['19a', 5000, 'BOX'], # boxcar
          #'J1444': ['19a', 4850, 'BOX'], # boxcar
          #'J1448': ['19a', 4850, 'BOX'], # boxcar
          #'J2101': ['19a', 5050, 'BOX'], # boxcar
          #'J2119': ['19a', 4950, 'BOX'], # boxcar
          
          #'J0136': ['19b', 4826, 'BOX'], # to be tested
          #'J0252': ['19b', 4564, 'BOX'], # to be tested
          #'J0305': ['19b', 4788, 'BOX'], # to be tested
          #'J2337': ['19b', 4626, 'BOX'], # to be tested
          
          #'J0023': ['23', 5464, 'BOX'],
          #'J1624': ['23', 4783, 'BOX'],
          #'J2212': ['23', 5539, 'BOX'],
          #'J2212_comp': ['23', 5463, 'OPT']
          
}

waves = [3726, 4861.363, 4363, 4958.911,
         5006.843, 6562.801, 6716.44, 7319]

lines = ['[OII]3727,29', r'H$\beta$', '[OIII]4363', '[OIII]4959',
         '[OIII]5007', r'H$\alpha$', '[SII]6716,30', '[OII]7319']

# ==========================================================================
#
# Functions
#
# ==========================================================================


def merge_spec(blue, red, limit=4500):
    """
    This function takes the good part of the blue and red spectrum,
    concatenates the wavelength, flux and inverse variance.
    
    """
    
    blue_mask = (blue['wave'] <= limit)
    gp_blue = blue['mask'].astype(bool)
    
    red_mask = (red['wave'] > limit)
    gp_red = red['mask'].astype(bool)
    
    wave = np.concatenate([blue['wave'][blue_mask*gp_blue],
                           red['wave'][red_mask*gp_red]])
    
    flux = np.concatenate([blue['flux'][blue_mask*gp_blue],
                           red['flux'][red_mask*gp_red]])
    
    ivar = np.concatenate([blue['ivar'][blue_mask*gp_blue],
                           red['ivar'][red_mask*gp_red]])
    
    #dl = red['wave'][-1] - red['wave'][-2]
    worst_dl = red['wave'][-1] - red['wave'][-2]
    best_dl = blue['wave'][1] - blue['wave'][0]
    
    if worst_dl < best_dl:
        worst_dl = blue['wave'][1] - blue['wave'][0]
        best_dl = red['wave'][-1] - red['wave'][-2]
        print('Careful: This source has worst dlambda in blue orders.')
    
    return wave, flux, ivar, best_dl, worst_dl


def plot_red_blue(redspec, bluespec, galname, z, limit):
    """
    This function is to plot the resulting merged spectrum.
    """
    blue_mask = (bluespec['wave'] <= limit)
    red_mask = (redspec['wave'] > limit)
    texPlot("on")
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 4)

    ax.plot(bluespec['wave'][blue_mask]/(1 + z),
             bluespec['flux'][blue_mask]*1e-17,
             color=czp[1], label='blue spec')
    ax.plot(redspec['wave'][red_mask]/(1 + z),
             redspec['flux'][red_mask]*1e-17,
             color=czd[0], label='red spec')
    
    ax.fill_between(bluespec['wave'][blue_mask]/(1 + z),
                 bluespec['flux'][blue_mask]*1e-17 - np.sqrt(1/bluespec['ivar'][blue_mask])*1e-17,
                 bluespec['flux'][blue_mask]*1e-17 + np.sqrt(1/bluespec['ivar'][blue_mask])*1e-17,
                 color=czp[1], label='blue noise', ls='--')
    ax.fill_between(redspec['wave'][red_mask]/(1 + z),
                 redspec['flux'][red_mask]*1e-17 - np.sqrt(1/redspec['ivar'][red_mask])*1e-17,
                 redspec['flux'][red_mask]*1e-17 + np.sqrt(1/redspec['ivar'][red_mask])*1e-17,
                 color=czd[0], label='red noise', ls='--')
    
    for i in range(len(waves)):
            ax.axvline(waves[i], ls='--', color='black', lw=0.5)
            ax.text(waves[i], 1.05e-15, lines[i], rotation=60)
    ax.set_title(galname+' (z = '+str(z)+')')
    ax.set_ylabel('Flux [erg / s / cm$^2$]', fontsize=15)
    ax.set_xlabel('$\lambda$ [\AA]', fontsize=15)
    #ax.set_yscale('log')
    
    fig.savefig(magePath+'results/merge_and_rebin/merged_'+galname+'.pdf',
                bbox_inches='tight')
    plt.close()
    texPlot('off')

# ==========================================================================
#
# Program
#
# ==========================================================================


if __name__=='__main__':
    
    for galname in limits.keys():

        parentpath = '/home/benjamin/Documents/'
        # Parameters from constutils. Name, redshift, etc.
        igal = np.argmax(source_dict['name'] == galname)
        z = source_dict['z_BN'][igal]
        ext_galname = source_dict['ext_name'][igal]
        
        print('Merging '+galname+' spectrum...')
        
        # Read parameters to select data products from the reduction
        night = limits[galname][0]
        limit = limits[galname][1]
        extraction = limits[galname][2]

        # If you want to use telluric corrected data, set tellcorr=True
        tellcorr = True
        suffix = '.fits'
        if tellcorr:
            suffix = '_tellcorr.fits'
        
        # None indicates now that the source does not have red-blue
        # spectrum, only one.
        if limit == None:
            # Explicative messages
            print('This source only has red (all) reduction.')
            print('No merging performed, only processing...')

            # Reading spectrum
            allpath = 'pypeit' + night + '/redux_' + galname + \
            '/all/magellan_mage_A/'
            spec = fits.open(parentpath + allpath \
                             + galname + '_all' + suffix)[1].data
            wave = spec['wave']
            flux = spec['flux']
            ivar = spec['ivar']
            
            # Processing errors
            print('Masking errors...')
            for i in range(len(ivar)):
                if ivar[i] == 0:
                    ivar[i] = np.nan
                    
            # Flux and noise units
            flux = flux*1e-17
            noise = (np.sqrt(ivar)**-1)*1e-17
            
            # Save the concatenated spectrum
            # Wavelengths are saved in air
            concat_spec = pd.DataFrame({'wave': wave,
                                      'flux': flux,
                                      'noise': noise})

            # Plotting
            texPlot("on")
    
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(10, 4)
        
            ax.plot(spec['wave']/(1 + z),
                    flux, color=czp[1], label='Spec')
            
            ax.fill_between(spec['wave']/(1 + z),
                            flux - noise, flux + noise,
                            color=czp[1], label='Noise', ls='--')
            
            for i in range(len(waves)):
                    ax.axvline(waves[i], ls='--', color='black', lw=0.5)
                    ax.text(waves[i], 1.05e-15, lines[i], rotation=60)
            ax.set_title(galname+' (z = '+str(z)+')')
            ax.set_ylabel('Flux [erg / s / cm$^2$]', fontsize=15)
            ax.set_xlabel('$\lambda$ [\AA]', fontsize=15)
            
            fig.savefig(magePath+'results/merge_and_rebin/merged_'+galname+'.pdf',
                        bbox_inches='tight')
            plt.close()
            texPlot('off')
    
            print('Spectrum properly processed.')
            concat_spec.to_csv(parentpath+'analogs/specs/pypeit/'+galname+'.csv',
                              index=None, sep=',')
            print('Saved at '+parentpath+'analogs/specs/pypeit/'+galname+'.csv')
            print('========================================================')
            continue

        if extraction == 'BOX':
            bluepath = 'pypeit' + night + '/redux_' + galname + \
            '/blue/magellan_mage_A/'
            redpath = 'pypeit' + night + '/redux_' + galname + \
            '/red/magellan_mage_A/'

        elif extraction == 'OPT':
            bluepath = 'pypeit' + night + '/redux_' + galname \
            + '/blue/magellan_mage_A/'
            redpath = 'pypeit' + night + '/redux_' + galname \
            + '/red/magellan_mage_A/'
    
        print('Joining spectra in '+extraction+' mode.')
        print('Red spectrum loaded from: '+redpath+galname+'_red_1'+suffix)
        print('Blue spectrum loaded from: '+bluepath+galname+'_blue'+suffix)
        
        # Read the blue and red reductions
        bluespec = fits.open(parentpath + bluepath \
                             + galname + '_blue' + suffix)[1].data

        redspec = fits.open(parentpath + redpath \
                             + galname + '_red_1' + suffix)[1].data
        
        # Plot the merged spectrum.
        plot_red_blue(redspec, bluespec, galname, z, limit)
        
        # Concatenated spectrum.
        wave, flux, ivar, b_dlam, w_dlam = merge_spec(bluespec, redspec,
                                                      limit)

        # Replace 0 values in inverse variance with NaN (otherwise it cause
        # problems with sigma calculation by dividing by 0)

        print('Masking errors...')
        for i in range(len(ivar)):
            if ivar[i] == 0:
                ivar[i] = np.nan

        # Flux and noise units
        flux = flux*1e-17
        noise = (np.sqrt(ivar)**-1)*1e-17

        # Save the concatenated spectrum
        # Wavelengths are saved in air
        concat_spec = pd.DataFrame({'wave': wave,
                                  'flux': flux,
                                  'noise': noise})

        print('Spectra joined successfully.')

        concat_spec.to_csv(parentpath+'analogs/specs/pypeit/'+galname+'_new.csv',
                          index=None, sep=',')
        
        print('Saved at '+parentpath+'analogs/specs/pypeit/'+galname+'_new.csv')
        print('Preparing input spectrum for pPXF.')

        # Perform rebinning and processing for pPXF.
        pPXF = False
        if not pPXF:
            print('No rebinning nor saving spec for pPXF fitting.')
            print('========================================================')
        else:
            # Rebin with the worst dlambda for wavelength uniformity with ppxf
            new_wave = np.arange(min(wave), max(wave), w_dlam)
            
            # Resample and save
            new_flux, new_noise = spectres(new_wave, wave, flux, noise)
            rebinned_spectres = pd.DataFrame({'wave': new_wave,
                                      'flux': new_flux,
                                      'noise': new_noise})
            rebinned_spectres.to_csv(parentpath + 'analogs/specs/pypeit_for_ppxf/' \
                               + galname + '.csv', index=None, sep=',')
            print('Rebinned spectra saved!')
            
            texPlot('on')
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(10, 4)
            
            ax.plot(new_wave/(1 + z), new_flux, lw=0.5, color=czp[1])
            ax.fill_between(new_wave/(1 + z), new_flux - new_noise,
                            new_flux + new_noise, color=czp[1],  alpha=0.7)
            ax.set_yscale('log')
            ax.set_title(galname+' (z = '+str(z)+', d$\lambda$ = ' \
                         +str(round(w_dlam, 2))+')')
            ax.set_ylabel('Flux [erg / s / cm$^2$]', fontsize=15)
            ax.set_xlabel('$\lambda$ [\AA]', fontsize=15)
            fig.savefig(magePath+'results/merge_and_rebin/rebinned_'+galname+'.pdf',
                    bbox_inches='tight')
            plt.close()
            texPlot('off')
