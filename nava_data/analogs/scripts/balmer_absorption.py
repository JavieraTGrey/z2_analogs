# Created on Aug 18th 12:29 AM
# @author : bnavarre
#
# This script takes the best model SED fit for the local analogs and measure
# the Balmer absorption from the stellar spectra.

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
from lmfit import minimize, Parameters
from astropy.io import fits
from uncertainties import ufloat

import os
import sys
import glob
docPath = '/home/benjamin/Documents/'
magePath = '/home/benjamin/Documents/analogs/'
os.sys.path.append(magePath)

from magE.directMethod import directT, run_all
from magE.functions import mergeSpec, gaussian
from magE.plotutils import * 
from magE.constutils import *
PATH = '/home/benjamin/Documents/analogs/specs/pypeit/'
os.chdir(PATH)

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)


# =============================================================================
#
# Functions
#
# =============================================================================


def prep_model_specs(sed_model, z, lum_dist):
    '''
    Gives the wavelength and flux arrays of Halpha and Hbeta absorptions in the
    SED model spectrum to be fitted. It is normalized to 1e-17.
    '''
    
    # H alpha
    mask_ha = (sed_model[1].data['wavelength']*10 > (wave_cat['Ha'] * (1 + z) - 200)) & \
              (sed_model[1].data['wavelength']*10 <= (wave_cat['Ha'] * (1 + z) + 200))
    wave_ha_abs = (sed_model[1].data['wavelength']*10)[mask_ha]  # in Angstrom
    flux_ha_abs = 1e17 * (1 + z) * (sed_model[1].data['stellar.young'][mask_ha] + sed_model[1].data['stellar.old'][mask_ha] + sed_model[1].data['attenuation.stellar.young'][mask_ha] + sed_model[1].data['attenuation.stellar.old'][mask_ha]) * 1e6 / (4 * np.pi * (lum_dist.value**2))

    # H beta
    mask_hb = (sed_model[1].data['wavelength']*10 > (wave_cat['Hb'] * (1 + z) - 200)) & \
              (sed_model[1].data['wavelength']*10 <= (wave_cat['Hb'] * (1 + z) + 200))
    wave_hb_abs = (sed_model[1].data['wavelength']*10)[mask_hb]  # in Angstrom
    flux_hb_abs = 1e17 * (1 + z) * (sed_model[1].data['stellar.young'][mask_hb] + sed_model[1].data['stellar.old'][mask_hb] + sed_model[1].data['attenuation.stellar.young'][mask_hb] + sed_model[1].data['attenuation.stellar.old'][mask_hb]) * 1e6 / (4 * np.pi * (lum_dist.value**2))
    
    return wave_ha_abs, flux_ha_abs, wave_hb_abs, flux_hb_abs


def residual_ha_abs(params_ha_abs, x, data, uncertainty):

    z = params_ha_abs['z']
    sigma = params_ha_abs['sigma']
    ha_flux = params_ha_abs['Ha_flux']
    bkg_m = params_ha_abs['bkg_m']
    bkg_n = params_ha_abs['bkg_n']
    
    x = wave_ha_abs
    model = bkg_m * x + bkg_n + gaussian(x, ha_flux, wave_cat['Ha'], z, sigma)
    noise = 0.01 * np.min(flux_ha_abs)
    return (model - flux_ha_abs) / noise


def residual_hb_abs(params_hb_abs, x, data, uncertainty):
    
    z = params_hb_abs['z']
    sigma = params_hb_abs['sigma']
    hb_flux = params_hb_abs['Hb_flux']
    bkg_m = params_hb_abs['bkg_m']
    bkg_n = params_hb_abs['bkg_n']
    
    x = wave_hb_abs
    model = (bkg_m * x) + bkg_n + gaussian(x, hb_flux, wave_cat['Hb'], z, sigma)
    noise = 0.05 * np.max(flux_ha_abs)
    return (model - flux_hb_abs) / noise


# =============================================================================
#
# Program
#
# =============================================================================

if __name__ == '__main__':
    
    np.random.seed(999)
    # Read galnames
    dataPath = magePath + 'specs/final_spec/'
    galaxy_spectra = glob.glob(dataPath+'*.csv')
    
    # fit_mode defines what type of fit do you want to consider
    # when reading the sed fitting results.
    fit_mode = 'old_pop_late_burst/'
    
    gals_to_dict = []
    ha_abs_to_dict = []
    hb_abs_to_dict = []
    ha_ew_to_dict = []
    ha_ew_err_to_dict = []
    hb_ew_to_dict = []
    hb_ew_err_to_dict = []
    
    for galspec in galaxy_spectra:
        
        if galspec[-5] == 'p':
            galname = galspec[-14:-4]
        else:
            galname = galspec[-9:-4]
            
        # Galaxy redshto
        igal = np.argmax(np.array(galnames) == galname)
        ext_name = source_dict['ext_name'][igal]
        z_gal = z_BN[igal]
        
        print('Fitting ' + ext_name + ' absorption. Redshift:', z_gal)

        # Read bestfit SED model
        SED_model = fits.open(docPath  + 'sed_fitting/' + fit_mode \
                              + ext_name + '_best_model.fits')
        # Luminosity distance
        lum_distance = cosmo.luminosity_distance(z_gal) * \
                        (1 * u.cm / (3.24078e-25 * u.Mpc))
        
        # Arrays to fit
        wave_ha_abs, flux_ha_abs, wave_hb_abs, flux_hb_abs = prep_model_specs(SED_model,
                                                             z_gal,
                                                             lum_distance)
        
        print('Fitting H alpha absorption...')
        # Initial guess Halpha
        y1_ha, y2_ha = flux_ha_abs[0], flux_ha_abs[-1]
        x1_ha, x2_ha = wave_ha_abs[0], wave_ha_abs[-1]
        m_ha = (y2_ha - y1_ha) / (x2_ha - x1_ha)
        n_ha = y1_ha - (m_ha * x1_ha)
        
        params_ha_abs = Parameters()
        params_ha_abs.add('z', value=z_gal)
        params_ha_abs.add('sigma', value=10, min=0.)
        params_ha_abs.add('Ha_flux', value=-3 * min(flux_ha_abs), max=0.)
        params_ha_abs.add('bkg_m', value=m_ha, max=0.)
        params_ha_abs.add('bkg_n', value=n_ha, min=0.)
        
        # Fitting Ha absorption
        out_ha_abs = minimize(residual_ha_abs, params_ha_abs, args=(1, 1, 1),
                      method='emcee', burn=1000, steps=2500, thin=10)
        ha_abs_flux = out_ha_abs.flatchain.Ha_flux.mean()
        ha_abs_std = out_ha_abs.flatchain.Ha_flux.std()
        bkg_ha_abs_m = out_ha_abs.flatchain.bkg_m.mean()
        bkg_ha_abs_n = out_ha_abs.flatchain.bkg_n.mean()
        z_ha_abs = out_ha_abs.flatchain.z.mean()
        sigma_ha_abs = out_ha_abs.flatchain.sigma.mean()
        
        # Equivalent width
        continuum_ha = bkg_ha_abs_m * wave_ha_abs + bkg_ha_abs_n
        bkg_ha = np.mean(continuum_ha)
        disp_ha = np.std(continuum_ha)
        print('Background Ha: ', bkg_ha)
        print('Bkg disp: ', disp_ha)
        
        spectrum_ha = continuum_ha + gaussian(wave_ha_abs, ha_abs_flux,
                               wave_cat['Ha'], z_ha_abs, sigma_ha_abs)
        
        #EW_Ha = np.sum(1 - (spectrum_ha/continuum_ha))
        EW_Ha = np.abs(ufloat(ha_abs_flux, ha_abs_std) / ufloat(bkg_ha,
                                                   disp_ha))
        
        print('Fitting H beta absorption...')
        # Initial guess Hbeta
        y1_hb, y2_hb = flux_hb_abs[0], flux_hb_abs[-1]
        x1_hb, x2_hb = wave_hb_abs[0], wave_hb_abs[-1]
        m_hb = (y2_hb - y1_hb) / (x2_hb - x1_hb)
        n_hb = y1_hb - (m_hb * x1_hb)
        
        params_hb_abs = Parameters()
        params_hb_abs.add('z', value=z_gal)
        params_hb_abs.add('sigma', value=10, min=0.)
        params_hb_abs.add('Hb_flux', value=-4 * min(flux_hb_abs), max=0.)
        params_hb_abs.add('bkg_m', value=m_hb, max=0.)
        params_hb_abs.add('bkg_n', value=n_hb, min=0.)
        
        # Fitting Hb absorption
        out_hb_abs = minimize(residual_hb_abs, params_hb_abs, args=(1, 1, 1),
                      method='emcee', burn=1000, steps=2500, thin=10)
        hb_abs_flux = out_hb_abs.flatchain.Hb_flux.mean()
        hb_abs_std = out_hb_abs.flatchain.Hb_flux.std()
        bkg_hb_abs_m = out_hb_abs.flatchain.bkg_m.mean()
        bkg_hb_abs_n = out_hb_abs.flatchain.bkg_n.mean()
        z_hb_abs = out_hb_abs.flatchain.z.mean()
        sigma_hb_abs = out_hb_abs.flatchain.sigma.mean()
        
        # Equivalent width
        continuum_hb = bkg_hb_abs_m * wave_hb_abs + bkg_hb_abs_n
        bkg_hb = np.mean(continuum_hb)
        disp_hb = np.std(continuum_hb)
        print('Background Hb: ', bkg_hb)
        print('Bkg disp: ', disp_hb)
        
        spectrum_hb = gaussian(wave_hb_abs, hb_abs_flux,
                               wave_cat['Hb'], z_hb_abs, sigma_hb_abs)
        
        # EW_Hb = np.sum(1 - (spectrum_hb/continuum_hb))
        EW_Hb = np.abs(ufloat(hb_abs_flux, hb_abs_std) / ufloat(bkg_hb,
                                                                disp_hb))
        
        print('Results:')
        print('Ha abs flux:', np.abs(ha_abs_flux) * 1e-17)
        print('Hb abs flux:', np.abs(hb_abs_flux) * 1e-17)
        
        #print('EW_Ha (now): ', np.abs(ha_abs_flux / bkg_ha))
        #print('EW_Hb (now): ', np.abs(hb_abs_flux / bkg_hb))
        print('EW(Ha):', EW_Ha)
        print('EW(Hb):', EW_Hb)
        
        print('Saving plots...')
        
        # Plotting
        texPlot('on')
        fig = plt.figure(1)
        fig.set_size_inches(6, 4)
        fig.suptitle(ext_name)
        ax = fig.add_axes((0, .2, 1, .8))
        ax1 = fig.add_axes((0, 0, 1, .2))

        ax2 = fig.add_axes((1.2, .2, 1, .8))
        ax3 = fig.add_axes((1.2, 0, 1, .2))

        ax.errorbar(wave_ha_abs, flux_ha_abs / np.max(flux_ha_abs), 0.02,
                    color=c[0], markersize=2, label='Data')
        ha_abs = gaussian(wave_ha_abs, ha_abs_flux, wave_cat['Ha'], z_ha_abs, sigma_ha_abs)
        ax.plot(wave_ha_abs, ((bkg_ha_abs_m * wave_ha_abs) + bkg_ha_abs_n + ha_abs) / np.max(flux_ha_abs),
                color='green', lw=2, label='Fit')
        ax1.plot(wave_ha_abs, flux_ha_abs - (bkg_ha_abs_m * wave_ha_abs) - bkg_ha_abs_n - ha_abs, color=c[3])

        ax.set_ylabel(r'Relative flux', fontsize=15)
        ax1.set_xlabel(r'Rest wavelength (\AA)', fontsize=15)
        #ax.set_yscale('log')
        ax.tick_params(axis='y', direction='in', length=8)
        ax1.tick_params(axis='x', direction='in', length=8)
        ax1.tick_params(axis='y', direction='in', length=8)
        ax.set_xticks([])
        ax.legend(loc='lower left')
        ax.set_title(r'H$\alpha$')

        ax2.errorbar(wave_hb_abs, flux_hb_abs / np.max(flux_hb_abs), 0.02,
                    color=c[0], markersize=2, label='Data')
        hb_abs = gaussian(wave_hb_abs, hb_abs_flux, wave_cat['Hb'], z_hb_abs, sigma_hb_abs)
        ax2.plot(wave_hb_abs, ((bkg_hb_abs_m * wave_hb_abs) + bkg_hb_abs_n + hb_abs) / np.max(flux_hb_abs),
                 color='green', lw=2, label='Fit')
        ax3.plot(wave_hb_abs, flux_hb_abs - (bkg_hb_abs_m * wave_hb_abs) - bkg_hb_abs_n - hb_abs, color=c[3])
        
        ax2.set_ylabel(r'Relative flux', fontsize=15)
        ax3.set_xlabel(r'Rest wavelength (\AA)', fontsize=15)
        #ax2.set_yscale('log')
        ax2.tick_params(axis='y', direction='in', length=8)
        ax3.tick_params(axis='x', direction='in', length=8)
        ax3.tick_params(axis='y', direction='in', length=8)
        ax2.set_xticks([])
        ax2.set_title(r'H$\beta$')
        fig.savefig(magePath+'results/balmer_absorption/'+ext_name+'_bal_abs.pdf', bbox_inches='tight')
        plt.close()
        
        gals_to_dict.append(galname)
        ha_abs_to_dict.append(np.abs(ha_abs_flux)*1e-17)
        hb_abs_to_dict.append(np.abs(hb_abs_flux)*1e-17)
        ha_ew_to_dict.append(EW_Ha.nominal_value)
        ha_ew_err_to_dict.append(EW_Ha.std_dev)
        hb_ew_to_dict.append(EW_Hb.nominal_value)
        hb_ew_err_to_dict.append(EW_Hb.std_dev)
        
        print('Saved absorption values.')
        print('===============================================================')
    abs_dict = pd.DataFrame({'Galaxy': gals_to_dict,
                             'Ha_abs': ha_abs_to_dict,
                             'EW_Ha': ha_ew_to_dict,
                             'EW_Ha_err': ha_ew_err_to_dict,
                             'Hb_abs': hb_abs_to_dict,
                             'EW_Hb': hb_ew_to_dict,
                             'EW_Hb_err': hb_ew_err_to_dict})
    abs_dict.to_csv(magePath+'results/balmer_absorption/balmer_abs_oplb_we.csv',
                    index=None)
    
        