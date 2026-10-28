"""
Milky Way (Galactic) dust correction

Correction (Cardelli, Clayton & Mathis 1989, R_V = 3.1, at OBSERVED
wavelengths, E(B-V) from the IRSA table, Schlafly & Finkbeiner 2011):
"""

# Dependencies
import os
import shutil
import numpy as np
import pandas as pd
import GALAXIES as gal
import matplotlib as mpl
import matplotlib.pyplot as plt

# Dust Attenuation Law
from dust_laws import k_ccm

# Constants and general data
LN10 = np.log(10.0)
proj_DIR = gal.PROJ_DIR

mpl.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"      # Computer Modern, like LaTeX
plt.rcParams["font.serif"] = ["cmr10", "DejaVu Serif"]
plt.rcParams["axes.formatter.use_mathtext"] = True


# ================================================================
# E(B-V) of each sightline
# ================================================================

def read_extinction(path, calib_frac=0.0):
    """
    E(B-V)_MW and its uncertainty per galaxy from the IRSA table.
    """
    rows = []
    with open(path) as fh:
        for lin in fh:
            if lin.startswith('#') or lin.startswith('|') or not lin.strip():
                continue
            parts = lin.split()
            rows.append({'gal': parts[0],
                         'E_MW': float(parts[4]),
                         'stdev': float(parts[6])})
    t = pd.DataFrame(rows).set_index('gal')
    t['sigma_E'] = np.sqrt(t['stdev']**2 + (calib_frac * t['E_MW'])**2)
    if calib_frac == 0:
        print('sigma_E = map scatter only (calib_frac = 0)')
    return t


# ================================================================
# Correct one spectrum
# ================================================================


def correct_spectrum(wave, flux, sigma, E_mw, sigma_E, Rv=3.1):
    k_value = k_ccm(wave, Rv)
    Corr = 10**(0.4 * E_mw * k_value)
    out = pd.DataFrame({'wave': wave,
                        'flux': flux * Corr,
                        'sigma': sigma * Corr,
                        'C': Corr,
                        'k': k_value,
                        'C_rel_err': 0.4 * LN10 * k_value * sigma_E})
    return out


def save_corr(gal, t, E_mw, sigma_E):
    folder = f'{proj_DIR}new_results/MW_dust/{gal}'
    os.makedirs(folder, exist_ok=True)
    path = f'{folder}/dcorr_{gal}.csv'

    # Check file doesnt get rewritten, save as old
    old = f'{folder}/dcorr_{gal}_old.csv'
    if os.path.exists(path) and not os.path.exists(old):
        shutil.copy(path, old)
        print(f'Old file kept as dcorr_{gal}_old.csv')

    # Save t
    t[['wave', 'flux', 'sigma']].to_csv(path, index=False)
    side = t[['wave', 'C', 'k', 'C_rel_err']].copy()
    side['E_MW'] = E_mw
    side['sigma_E'] = sigma_E
    side.to_csv(f'{folder}/dcorr_{gal}_mw.csv', index=False)

    print(f'{gal}: E(B-V)_MW = {E_mw:.4f} +- {sigma_E:.4f}; '
          f'correction {t["C"].min():.3f} (red) to {t["C"].max():.3f} (blue); '
          f'its error {100*t["C_rel_err"].min():.2f}% to '
          f'{100*t["C_rel_err"].max():.2f}%')
    print(f'Saved {path} and dcorr_{gal}_mw.csv')


def run(class_, ext_path=None, calib_frac=0.10, Rv=3.1,
        plot=True):

    # Read table
    if ext_path is None:
        ext_path = f'{proj_DIR}CSV_files/extinction.tbl'
    ext = read_extinction(ext_path, calib_frac=calib_frac)

    # Data from galaxy
    gal = class_.gal_id
    E_mw = ext.loc[gal, 'E_MW']
    sigma_E = ext.loc[gal, 'sigma_E']

    wave, flux, sigma, _ = class_.datas[0]
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    t = correct_spectrum(wave, flux, sigma, E_mw, sigma_E, Rv)

    # Saving data
    save_corr(gal, t, E_mw, sigma_E)

    if plot:
        folder = f'{proj_DIR}new_results/MW_dust/{gal}'
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax1.plot(wave, flux, lw=0.4, color='0.6', label='observed')
        ax1.plot(wave, t['flux'], lw=0.4, color='C0', label='MW corrected')
        ax1.set_ylim(0, np.nanpercentile(t['flux'], 99) * 1.5)
        ax1.set_ylabel(r'$F_\lambda$ [$10^{-17}$ erg s$^{-1}$ '
                       r'cm$^{-2}$ $\AA^{-1}$]')
        ax1.legend(frameon=False)
        ax1.set_title(f'{gal}: E(B-V)$_{{MW}}$ = {E_mw:.4f}'
                      f'$\\pm$ {sigma_E:.4f}')
        ax2.plot(wave, t['C'], color='C3', label='correction C')
        ax2.fill_between(wave, t['C'] * (1 - t['C_rel_err']),
                         t['C'] * (1 + t['C_rel_err']), color='C3', alpha=0.3)
        ax2.set_xlabel(r'Observed wavelength ($\AA$)')
        ax2.set_ylabel('C = F$_{corr}$ / F$_{obs}$')
        fig.savefig(f'{folder}/MWcorr_{gal}.png', dpi=130, bbox_inches='tight')
    return t, E_mw, sigma_E
