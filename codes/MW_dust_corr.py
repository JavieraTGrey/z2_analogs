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

# Dust Attenuation Law
from dust_laws import k_ccm


# Constants and general data
LN10 = np.log(10.0)
PROJ_DIR = gal.PROJ_DIR


# ================================================================
# E(B-V) of each sightline
# ================================================================

def read_extinction(path, calib_frac=0.0):
    """
    E(B-V)_MW and its uncertainty per galaxy from the IRSA table.
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') or line.startswith('|') or not line.strip():
                continue
            parts = line.split()
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
