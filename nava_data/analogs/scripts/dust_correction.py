# Created on Mon May 22th, 2023
# @author: bnavarrete

# This script makes the extinction for all galaxies using PyNeb and the
# reddening estimated from https://irsa.ipac.caltech.edu/applications/DUST/
# for each galaxy in our sample.

import os
import sys
import glob
magePath = '/home/benjamin/Documents/analogs/'
os.sys.path.append(magePath)

from magE.plotutils import *
from magE.constutils import source_dict
import pandas as pd
import numpy as np
import pyneb as pn
import matplotlib.pyplot as plt

# =============================================================================
#
# Functions
#
# =============================================================================


def dust_pyneb(wl, fl, err, E_BV, rel_Hb=False):
    '''
    Receives a spectrum (wl, fl, err) and a reddening constant E_BV and returns
    the dust corrected spectra using PyNeb. If rel_Hb=True, the correction is
    done using the extinction curve normalized by the extinction in Hb. The MW
    extinction curve from CCM89 is used (see PyNeb list of extinctions curve).
    
    The spectra MUST be in rest-frame wavelength.
    '''
    
    rc = pn.RedCorr(E_BV=E_BV, R_V=3.1, law='CCM89')
    
    if rel_Hb:
        dcorr_fl = fl * rc.getCorrHb(wl)
        dcorr_err = err * rc.getCorrHb(wl)
        print('Extinction correction done! Used Hb normalized curve.')
    else:
        dcorr_fl = fl * rc.getCorr(wl)
        dcorr_err = err * rc.getCorr(wl)
        print('Extinction correction done! Used total curve.')
    return wl, dcorr_fl, dcorr_err


def plot_comparison(gal_df, gal_id, igal, ppxf=False):
    
    texPlot("on")
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 3)
    
    ax.plot(gal_df['rf_wl'], gal_df['flux'], color=ccd[3],
            lw=0.5, label='Observed flux')
    ax.plot(gal_df['rf_wl'], gal_df['dcorr_flux'], color=cz[0],
            alpha=0.8, lw=0.5, label='Dust corrected')
    
    #ax.set_yscale('log')
    ax.set_ylabel(r'Flux (erg / s / cm$^{2}$)', fontsize=15)
    ax.set_xlabel(r'$\lambda$ (Angstrom)', fontsize=15)
    
    ax.set_title(r'Object: '+source_dict['ext_name'][igal]+', '+'z = '+str(z)+\
                ', '+'E$_{B - V}$ = '+str(E_BV))
    if ppxf:
        fig.savefig(magePath+'dust/dcorr_'+gal_id+'_for_ppxf.pdf',
               bbox_inches='tight')
        print('Saved figure at: '+magePath+'dust/dcorr'+gal_id+'_for_ppxf.pdf')
    else:
        fig.savefig(magePath+'dust/dcorr_pypeit_'+gal_id+'.pdf',
                   bbox_inches='tight')
        print('Saved figure at: '+magePath+'dust/dcorr_pypeit_'+gal_id+'.pdf')


# =============================================================================
#
# Program
#
# =============================================================================
    

if __name__ == '__main__':
    
    # Read all spectra
    ppxf = False
    if ppxf:
        dataPath = magePath+'specs/pypeit_for_ppxf/'
        print('Correction for pPXF input specra.')
    else:
        dataPath = magePath+'specs/pypeit/'
        
    #galaxy_spectra = glob.glob(dataPath+'*_new.csv')
    galaxy_spectra = ['/home/benjamin/Documents/analogs/specs/pypeit/J1146_new.csv']
    print('Spectra succesfully read.')
        
    
    # Read extinction table
    extinction = pd.read_table(magePath+'dust/extinction_new.tbl',
                             comment='#', delim_whitespace=True)
    
    for key in extinction.keys():
        extinction.rename(columns={key: key[1:]}, inplace=True)
    extinction.drop([0, 1], axis=0, inplace=True)
    print('Extinction rable read.')
    
    # Dust extinction correction for all galaxies
    for element in galaxy_spectra:
        #element = '/home/benjamin/Documents/analogs/specs/pypeit/J0950.csv'

        # Read data
        spec = pd.read_csv(element)
        wl = spec['wave']
        fl = spec['flux']
        err = spec['noise']

        # Properties
        if ppxf:
            gal_id = element[55:60]
        else:
            if element[51] != '.':
                gal_id = element[46:56]
            else:
                gal_id = element[46:51]
        print('Performing dust correction for '+gal_id)

        igal = np.argmax(source_dict['name'] == gal_id)
        z = source_dict['z_SDSS'][igal]
        E_BV = float(extinction['E_B_V_SandF'][igal + 2])

        # dust correction
        wl2, fl2, err2 = dust_pyneb(wl/(1 + z), fl, err, E_BV=E_BV)

        # store in the dataframe
        final_spec = pd.DataFrame({'wave': wl,
                                   'rf_wl': wl2,
                                   'flux': fl2,
                                   'noise': err2})
        spec['rf_wl'] = wl2
        spec['dcorr_flux'] = fl2
        spec['dcorr_err'] = err2

        # Plot and save comparison between observed and intrinsic spectrum
        plot_comparison(spec, gal_id, igal, ppxf=ppxf)
        final_spec.to_csv(magePath+'specs/final_spec/'+gal_id+'.csv', index=None)
        print('Saved new corrected spectrum at: '+magePath+'specs/final_spec/'\
              +gal_id+'.csv')
