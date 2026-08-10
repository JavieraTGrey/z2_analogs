# Created on Tue Jul 25th, 2023 9:03 am

# @author : bnavarre

# This script takes the magnitudes and errors of the list of local analogs
# downloaded in the fits files from the SQL CrossID search in SDSS and prepare
# the data input file to run SED fitting with CIGALE.

import numpy as np
import pandas as pd
from astropy.io import fits, ascii
from uncertainties import unumpy
from astropy.table import Table

import os
import sys
magePath = '/home/benjamin/Documents/analogs/'
os.sys.path.append(magePath)
from magE.constutils import *


# =============================================================================
# Functions

def mag_to_flux(mag, b):
    '''
    Takes an SDSS magnitude and the softening parameter b to calculate the
    flux density in mJy.
    '''
    f_f0 = 2 * b * unumpy.sinh(- np.log(b) - (mag * np.log(10) / 2.5 ))
    S = 3631 * 1e3 * f_f0
    return S

# =============================================================================

# Useful variables
homedir = '/home/benjamin/Documents/'

# Read the data catalog
mag_catalog = fits.open(homedir+'sed_fitting/photo_sdss_new.fits')

# Prepare columns
ID = np.array(mag_catalog[1].data['name'])
redshift = source_dict['z_SDSS']

# Create dictionary

cigale_input = {'id': ID, 'redshift': redshift}

# SDSS bands and softening parameters
sdss_bands = ['u', 'g', 'r', 'i', 'z']
b = np.array([1.4, 0.9, 1.2, 1.8, 7.4]) * 1e-10


if __name__ == '__main__':
    
    # Create table
    for band in sdss_bands:
        # Gather the data
        band_mag = unumpy.uarray(mag_catalog[1].data['cModelMag_'+band],
                                 mag_catalog[1].data['cModelMagerr_'+band])

        # Corrections according to Flux Calibration from SDSS
        if band == 'u':
            band_mag = band_mag - 0.04

        if band == 'z':
            band_mag = band_mag + 0.02
        
        # Gather the corresponding b factor
        b_band = b[np.argmax(sdss_bands == band)]
        
        # Convert magnitudes to flux density in mJy
        band_flux = mag_to_flux(band_mag, b_band)
        
        # Writing to the input file
        cigale_input[band+'_prime'] = unumpy.nominal_values(band_flux)
        cigale_input[band+'_prime_err'] = unumpy.std_devs(band_flux)
    
    # Create table and write to input.txt
    final_input = Table(cigale_input)
    savepath = homedir + 'sed_fitting/input_new.txt'
    final_input.write(savepath, format='ascii', overwrite=True)
    print('Cigale input file saved to', savepath)