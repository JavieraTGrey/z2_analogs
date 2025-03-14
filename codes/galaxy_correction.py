import os

from magE.plotutils import *
from magE.constutils import source_dict
import pandas as pd
import numpy as np
import pyneb as pn
import matplotlib.pyplot as plt
from LineClass import REDUC_LINES

magePath = '/Users/javieratoro/Desktop/proyecto 2024-2'
os.sys.path.append(magePath)

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


def plot_comparison(gal_df, gal_id, igal):

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 3)

    ax.plot(gal_df['rf_wl'], gal_df['flux'], color=ccd[3],
            lw=0.5, label='Observed flux')
    ax.plot(gal_df['rf_wl'], gal_df['dcorr_flux'], color=cz[0],
            alpha=0.8, lw=0.5, label='Dust corrected')

    ax.set_ylabel(r'Flux (erg / s / cm$^{2}$)', fontsize=15)
    ax.set_xlabel(r'$\lambda$ (Angstrom)', fontsize=15)
    title = f"Object: {source_dict['ext_name'][igal]}, z = {str(z)}"
    ax.set_title(title + ' E$_{B - V}$ = '+str(E_BV))

    fig.savefig(f'{magePath}dust/dcorr_{gal_id}.pdf',
                bbox_inches='tight')
    print(f'Saved figure at: {magePath}dust/dcorr_{gal_id}.pdf')


# =============================================================================
#
# Program
#
# =============================================================================


if __name__ == '__main__':

    # Creating the galaxy object
    J0328_ = {
              'DIR': '/Users/javieratoro/Desktop/BAADE_DATA/testing/35-J0328/',
              'FILES': ['no_very_flats/J0328_NO_VERY_FLATS_tellcorr.fits'],
              'redshift': 0.086,
              'names': ['J0328+0031'],
              'mass': 9.8}
    J0328 = REDUC_LINES(J0328_)

    print('Spectra succesfully read.')

    # Read extinction table
    extinction = pd.read_table(magePath+'CSV_files/extinction.tbl',
                               comment='#', delim_whitespace=True)

    # Clean up the table
    for key in extinction.keys():
        extinction.rename(columns={key: key[1:]}, inplace=True)
    extinction.drop([0, 1], axis=0, inplace=True)
    print('Extinction rable read.')

    # Dust extinction correction for the galaxy

    # Read data
    wave, flux, err, _= J0328.datas[0]
    gal_id = J0328.names[0][:5]
    print('Performing dust correction for ' + gal_id)

    igal = np.argmax(extinction['objname'] == gal_id)
    z = J0328.redshift
    E_BV = float(extinction['E_B_V_SandF'][igal + 2])

    # dust correction
    wl2, fl2, err2 = dust_pyneb(wave/(1 + z), flux, err, E_BV=E_BV)

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
    final_spec.to_csv(magePath+'specs/final_spec/'+gal_id+'.csv',
                        index=None)
    print('Saved new corrected spectrum at: '+magePath+'specs/final_spec/'+ gal_id +'.csv')
