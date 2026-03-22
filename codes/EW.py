# Equivalent width calculation and plotting functions

import astropy.constants as const
import numpy as np
import matplotlib.pyplot as plt
import lmfit as lm
import os
import pandas as pd

from GaussianFitting import fitSpectrum

proj_DIR = '/Users/javieratoro/Desktop/thesis/proyecto 2024-2/'


def read_data(class_):
    """
    Reads MW duct corrected data if it exist, otherwise reads the
    uncorrected data.
    """
    gal_id = class_.names[0][:5]
    print(f'Reading data for {gal_id}')
    file_path = f'{proj_DIR}dust/{gal_id}/dcorr_{gal_id}.csv'
    if os.path.exists(file_path):
        print("Using MW dust corrected data")
        dcorr = pd.read_csv(file_path)
        arrays = dcorr.to_numpy().T
        wave, flux, sigma = arrays

    else:
        print("Using uncorrected data")
        wave, flux, sigma, _ = class_.datas[0]
    return wave, flux, sigma


def first_sigma_est(class_, data, plot=False):
    """
    Get the velocity dispersion (sigma) of the narrow and/or broad
    Gaussian components from the spectral fitting.

    Add plot=True to visualize the spectral fitting model
    """
    wave, flux, sigma = data

    fit = fitSpectrum(wave, flux, sigma,
                      linelist=class_.linelist_dict,
                      z_init=class_.redshift,
                      weights=1/sigma**2,
                      showPlot=plot,
                      broad=True, nfev=1000)

    sigma_narrow = fit.params['sigma_v_narrow']
    sigma_broad = fit.params['sigma_v_broad']

    return sigma_narrow, sigma_broad


def isolate_emission_line(class_, line, window, datas):
    """
    Isolate the emission line from the spectral data.

    Parameters:
    - class_: The spectral data class.
    - line_center: The central wavelength of the emission line to isolate.
    - window: The width around the line center to consider for
              isolation (in Angstroms).

    Returns:
    - isolated_wave: Wavelength array of the isolated emission line region.
    - isolated_flux: Flux array of the isolated emission line region.
    - isolated_sigma: Sigma array of the isolated emission line region.
    - estimated_redshift: Estimated redshift based on the peak wavelength.
    - peak_wave: Observed wavelength at the peak flux.
    - peak_flux: Peak flux value of the emission line.
    """
    wave, flux, sigma = datas

    # Define the line center
    line_center = class_.linelist_dict[line] * (1 + class_.redshift)

    # Define the range for isolation
    lower_bound = line_center - window
    upper_bound = line_center + window

    # Create a mask to isolate the emission line
    mask = (wave >= lower_bound) & (wave <= upper_bound)

    # Isolate the data
    isolated_wave = wave[mask]
    isolated_flux = flux[mask]
    isolated_sigma = sigma[mask]

    # Estimate redshift from peak
    peak_idx = np.argmax(isolated_flux)
    peak_wave = isolated_wave[peak_idx]
    peak_flux = isolated_flux[peak_idx]
    vacuum_wave = class_.linelist_dict[line]
    estimated_redshift = (peak_wave - vacuum_wave) / vacuum_wave

    return (isolated_wave, isolated_flux, isolated_sigma,
            estimated_redshift, peak_wave, peak_flux)


def mask_nonuse_emission_line(class_, sigmas, line, datas):
    """
    Mask the non-use emission line from the spectral data.

    Parameters:
    - class_: The spectral data class.
    - wave: The wavelength array of the spectrum.
    - flux: The flux array of the spectrum.

    Returns:
    - masked_wave: The wavelength array with non-use emission lines masked.
    - masked_flux: The flux array with non-use emission lines masked.
    """
    # Define the wavelength ranges for masking (example ranges)
    wave, flux, err = class_.datas[0][0], class_.datas[0][1], class_.datas[0][2]

    bright_lines = ['O2_3725', 'O2_3727', 'H_alpha', 'H_beta', 'H_gamma',
                    'O3_5008', 'O3_4959', 'N2_6550', 'N2_6585', 'S2_6716',
                    'S2_6730']

    sigma_narr, sigma_broad = sigmas

    def get_sigma(line):
        center = class_.linelist_dict[line] * (1 + class_.redshift)
        if line in bright_lines:
            sigma = (center / const.c.to('km/s').value) * sigma_broad.value
        else:
            sigma = (center / const.c.to('km/s').value) * sigma_narr.value
        return sigma

    # Mask every emission line 3 sigma from center
    cte = (1 + class_.redshift)
    masked_flux = flux.copy()
    masked_sigma = err.copy()

    for label in class_.linelist_dict:
        if label == line:
            continue
        else:
            center = class_.linelist_dict[label] * cte
            sigma = get_sigma(label)

            size = 2.5
            mask_below = (wave > center - size*sigma)
            mask_up = (wave < center + size*sigma)
            mask_line = mask_below & mask_up

            # Every line to nan
            masked_flux[mask_line] = np.nan
            masked_sigma[mask_line] = 0.0

    return wave, masked_flux, masked_sigma



# =============================================================================
#
# Program
#
# =============================================================================
