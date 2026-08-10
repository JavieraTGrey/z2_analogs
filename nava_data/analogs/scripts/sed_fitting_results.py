'''
Created on Tue Sep 26th 15:04
@author: bnavarre

This script processes the results from the SED fitting to give a plot
of the best model SED and the resulting properties in the same plot.

'''

# Imports
import os
import sys
magePath = '/home/benjamin/Documents/analogs/'
os.sys.path.append(magePath)

from astropy.io import fits, ascii
import matplotlib.pyplot as plt
import numpy as np
from magE.constutils import *
from magE.functions import gaussian
import uncertainties.umath as uum
from uncertainties import ufloat
from uncertainties import unumpy

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

# Defining the type of fitting
fit_mode = 'old_pop_late_burst'

# Reading necessary files
cigalePath = '/home/benjamin/Documents/sed_fitting/'

input_data = ascii.read(cigalePath+'input_new.txt')
results = fits.open(cigalePath+fit_mode+'/results.fits')

# Important data
# Pivot wavelength of ugriz filters in Angstrom taken from the SVO filter system
center = 10000 *  np.array([0.356505, 0.470033, 0.617448,
                            0.753363, 0.878169])
filts = ['u', 'g', 'r', 'i', 'z'] # SDSS filters

# For all galaxies in the sample
for gal_id in results[1].data['id']:
    
    # Process the data
    mask = results[1].data['id'] == gal_id
    
    flux_model = []
    flux_obs = []
    
    flux_model_mJy = []
    flux_obs_mJy = []
    
    print('=======================================================')
    print('Procesing the data for:', gal_id)
    # Reading Age, M, Av, SFR and Chi Squared
    age = ufloat(results[1].data['bayes.sfh.age'][mask][0],
                 results[1].data['bayes.sfh.age_err'][mask][0]) / 1000
    mass = unumpy.log10(ufloat(results[1].data['bayes.stellar.m_star'][mask][0],
                               results[1].data['bayes.stellar.m_star_err'][mask][0]))
    att = 3.1 * ufloat(results[1].data['bayes.attenuation.E_BVs'][mask][0],
                       results[1].data['bayes.attenuation.E_BVs_err'][mask][0])
    sfr = ufloat(results[1].data['bayes.sfh.sfr'][mask][0],
                 results[1].data['bayes.sfh.sfr_err'][mask][0])
    red_chi2 = round(results[1].data['best.reduced_chi_square'][mask][0], 2)
    
    print('SED fitting results:')
    print('Age:', age)
    print('Mass:', mass)
    print('Av:', att)
    print('SFR:', sfr)
    print('reduced Chi squared:', red_chi2)
    
    # Luminosity distance
    igal = np.argmax(np.array(galnames_ext) == gal_id)
    z_gal = z_BN[igal]
    lum_distance = cosmo.luminosity_distance(z_gal) * \
                        (1 * u.cm / (3.24078e-25 * u.Mpc))
    
    for i in range(len(filts)):
        fl_model_mJy = results[1].data['bayes.'+filts[i]+'_prime'][mask][0]
        fl_obs_mJy = input_data[filts[i]+'_prime'][mask]
        flux_model_mJy.append(fl_model_mJy)
        flux_obs_mJy.append(fl_obs_mJy)
        flux_model.append(1e-18 * fl_model_mJy * 299792458 * 100 / (center[i]**2))
        flux_obs.append(1e-18 * float(fl_obs_mJy) * 299792458 * 100 / (center[i]**2))


    # Plot the SED
    print('Saving SED plot...')
    galaxy = fits.open(cigalePath+fit_mode+'/'+gal_id+'_best_model.fits')
    
    plt.figure()
    plt.plot(galaxy[1].data['wavelength']*10,
             galaxy[1].data['Fnu'],
             color='gray', alpha=0.6, lw=1, label='Best-fit spectrum'
    )
    plt.plot(galaxy[1].data['wavelength']*10,
             (galaxy[1].data['stellar.young']+galaxy[1].data['stellar.old']) * 1e6 / (4 * np.pi * (lum_distance**2)) * 1e18 * (galaxy[1].data['wavelength']*10)**2 / 299792458 / 100,
             lw=2, label='Stellar unattenuated'
    )

    plt.text(7500, 3e-2, r'$\chi^{2}_{\nu}:$ '+str(red_chi2))
    plt.text(7500, 2e-2, 'Age: '+ str(round(age.nominal_value, 2)) + ' ± ' + str(round(age.std_dev, 2)) +' Gyr')
    plt.text(7500, 1.3e-2, 'log(M/M*): '+str(round(float(unumpy.nominal_values(mass)), 2)) + ' ± ' + str(round(float(unumpy.std_devs(mass)), 2)))
    plt.text(7500, 8e-3, 'SFR: '+ str(round(sfr.nominal_value, 2)) + ' ± ' + str(round(sfr.std_dev, 2)) + r' M$_{\odot}$/yr')
    plt.text(7500, 5e-3, 'Av: '+ str(round(att.nominal_value, 2)) + ' ± ' + str(round(att.std_dev, 2)))
    plt.ylabel(r'$F_{\nu}$ (mJy)')

    plt.xlabel('Observed $\lambda$ ($\AA$)')
    plt.scatter(center, flux_model_mJy, color='orangered', label='Model')
    plt.scatter(center, flux_obs_mJy, color='green', facecolor='none', label='Observed')
    plt.yscale('log')
    plt.ylim(3e-3, 16)
    plt.xlim(3000, 10000)
    plt.legend(loc='upper left')
    plt.title(gal_id)
    plt.savefig(magePath+'results/sed_fitting/'+fit_mode+'/'+gal_id+'_SEDplot.pdf',
                bbox_inches='tight')
    plt.close()
    print('Done! Plot saved for', gal_id)