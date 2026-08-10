# Equivalent width calculation and plotting functions

import astropy.constants as const
from corr import REDSHIFT, SPECTRALDATA
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameter
from lmfit.models import GaussianModel, PolynomialModel
import os
import pandas as pd

from GaussianFitting import fitSpectrum

proj_DIR = '/Users/javieratoro/Desktop/thesis/proyecto 2024-2/'
balmer_lines = ['H_gamma', 'H_delta',
                'H_epsilon', 'H_8', 'H_9', 'H_10', 'H_11', 'H_12']


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


def get_sigma(line, class_, sigmas):
    sigma_narr, sigma_broad = sigmas
    bright_lines = ['O2_3725', 'O2_3727', 'H_alpha', 'H_beta', 'H_gamma',
                    'O3_5008', 'O3_4959', 'N2_6550', 'N2_6585', 'S2_6716',
                    'S2_6730']
    center = class_.linelist_dict[line] * (1 + class_.redshift)
    if line in bright_lines:
        sigma = (center / const.c.to('km/s').value) * sigma_broad.value
    else:
        sigma = (center / const.c.to('km/s').value) * sigma_narr.value
    return sigma


def mask_nonuse_emission_line(class_, sigmas, line, data):
    """
    Mask the non-use emission line from the spectral data.

    Parameters:
    - class_: The spectral data class.
    - sigmas: A tuple containing the velocity dispersions (sigma) for
                narrow and broad components.
    - line: The emission line to keep (e.g., 'H_alpha').
    - data: A tuple containing the wavelength, flux, and sigma arrays.

    Returns:
    - masked_wave: The wavelength array with non-use emission lines masked.
    - masked_flux: The flux array with non-use emission lines masked.
    """
    # Define the wavelength ranges for masking (example ranges)
    wave, flux, err = data

    # Mask every emission line 3 sigma from center
    cte = (1 + class_.redshift)
    masked_flux = flux.copy()
    masked_sigma = err.copy()

    for label in class_.linelist_dict:
        if label == line:
            continue
        else:
            center = class_.linelist_dict[label] * cte
            sigma = get_sigma(label, class_, sigmas)

            size = 2.5
            mask_below = (wave > center - size*sigma)
            mask_up = (wave < center + size*sigma)
            mask_line = mask_below & mask_up

            # Every line to nan
            masked_flux[mask_line] = np.nan
            masked_sigma[mask_line] = 0.0

    return wave, masked_flux, masked_sigma


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

    # plt.figure(figsize=(10, 6))
    # plt.step(wave, flux, label='Data', color='black')
    # plt.step(isolated_wave, isolated_flux, label='Isolated Line',
    #          color='red')
    # plt.xlabel('Wavelength (Å)')
    # plt.ylabel('Flux')
    # plt.title(f'Isolated Emission Line: {line}')
    # plt.legend()
    # plt.show()
    return (isolated_wave, isolated_flux, isolated_sigma)


def model(class_, label, datas, sigmas):
    """
    Create the model for the absoption with a negative gaussian
    and a 1-degree polynomial
    """
    wave, flux_, err = datas

    # mask emission line
    lam = class_.linelist_dict[label]
    center = lam * (1 + class_.redshift)

    sigma = get_sigma(label, class_, sigmas)

    size = 2.5
    mask_below = (wave > center - size*sigma)
    mask_up = (wave < center + size*sigma)
    mask_line = mask_below & mask_up

    # Every line to nan
    flux = flux_.copy()
    flux[mask_line] = np.nan

    # Gaussian model for balmer abs
    gaussian = GaussianModel(prefix=label+'_')

    # Create a polynomial model for the continuum
    polydeg = 1
    polynomial = PolynomialModel(degree=polydeg)

    comp_mult = gaussian + polynomial
    pars_mult = comp_mult.make_params()

    pars_mult.add(name='z', value=class_.redshift,
                  vary=False)

    pars_mult.add(name='sigma_v', value=550, vary=False)
    # velocity dispersion at 450 km/s

    # Loop through emission lines to define parameters
    # for narrow and broad

    for param in ['center', 'amplitude', 'sigma']:
        narrow_key = f'{label}_{param}'
        if param == 'center':
            value = lam
            vary_ = False
            min_ = None
            max_ = None
            expr = f'{lam:6.2f}*(1+z)'
        elif param == 'amplitude':
            value = -80
            vary_ = True
            min_ = -100
            max_ = 0
            expr = None
        elif param == 'sigma':
            value = None
            vary_ = False
            min_ = None
            max_ = None
            expr = f'(sigma_v/3e5)*{label}_center'
        pars_mult[narrow_key] = Parameter(name=narrow_key,
                                          value=value,
                                          vary=vary_, expr=expr,
                                          min=min_, max=max_)
    for i in range(polydeg+1):
        pars_mult[f'c{i:1.0f}'].set(value=0)

    out_comp_mult = comp_mult.fit(flux, pars_mult,
                                  x=wave, weights=1/err**2,
                                  nan_policy='omit',
                                  max_nfev=1000)

    return out_comp_mult


def model_mcmc(class_, label, stamp):

    wave, flux, f_err = stamp[0], stamp[1], stamp[2]
    z, w_center = class_.redshift, class_.linelist_dict[label]

    mu = w_center * (1 + z)
    sigma = w_center * (1 + z) * 550 / 3e5

    step = 0.5*sigma
    mask = np.isfinite(flux) & ((wave < mu - step) | (wave > mu + step))

    def absorption(w, A, m, n):
        output = -A * np.exp(-((mu - w) ** 2) / (2 * sigma**2)) + m * w + n
        return output

    #  Test one fit
    p0 = 5.0, 0.0, 5.0
    popt, pcov = curve_fit(absorption, wave[mask], flux[mask], p0)

    #
    # Do a MC
    #

    N_mc = 100
    f_mc = np.random.normal(loc=flux[mask],
                            scale=f_err[mask], size=(N_mc, mask.sum()))
    popt_mc = np.zeros((N_mc, 3))
    for i in range(N_mc):
        popt, pcov = curve_fit(absorption, wave[mask], f_mc[i], p0)
        popt_mc[i] = popt

    A_mean = np.median(popt_mc[:, 0])
    A_err = np.std(popt_mc[:, 0])

    EW_mc = (
        (popt_mc[:, 0] * np.sqrt(2 * np.pi * sigma**2))
        / (popt_mc[:, 1] * w_center * (1 + z) + popt_mc[:, 2])
        / (1 + z)
    )

    return A_mean, A_err, np.median(EW_mc), np.std(EW_mc), mask, popt_mc


def neg_gauss(class_, line, w, A):
    w_center = class_.linelist_dict[line]
    z = class_.redshift
    mu = w_center * (1 + z)
    sigma = w_center * (1 + z) * 550 / 3e5
    output = -A * np.exp(-((mu - w) ** 2) / (2 * sigma**2))
    return output


def absorption_func(class_, line, w, A, m, n):
    gauss = neg_gauss(class_, line, w, A)
    output = gauss + m * w + n
    return output


def neg_gauss_EW(class_, line, w, emcee, EW_, num):
    w_center = class_.linelist_dict[line]
    z = class_.redshift
    mu = w_center * (1 + z)
    sigma = w_center * (1 + z) * 550 / 3e5
    popt_mc = emcee[5]

    m = popt_mc[num, 1]
    n = popt_mc[num, 2]

    continuum_at_mu = m * mu + n
    A = (EW_ * (1 + class_.redshift)
        * continuum_at_mu
        / (np.sqrt(2*np.pi) * sigma) )
    output1 = -A * np.exp(-((mu - w) ** 2) / (2 * sigma**2))
    return output1


def save_corrected_data(class_, data, new_flux):
    """
    Saves the corrected data to a CSV file,
    creating the directory if needed.
    """
    # Define the directory and file path
    save_dir = f'{proj_DIR}bal_abs'
    save_path = f'{save_dir}/bcorr_{class_.gal_id}_new.csv'

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create and save DataFrame
    df = pd.DataFrame({'wave': data[0], 'flux': new_flux,
                       'sigma': data[2]})
    df.to_csv(save_path, index=False)

    print(f'Saved Balmer Absorption corrected data to: {save_path}')


def balmer_absorption_correction(info):
    """
    Applies the Balmer absorption correction to the flux data using the
    MCMC results from the model fitting.

    Parameters:
    - class_: The spectral data class.
    - data: The input data.
    - sigmas: The sigma values.
    - emcee: The MCMC results.

    Returns:
    - A message indicating the correction has been applied and results saved.
    """
    # Implementation for applying Balmer absorption correction
    class_ = SPECTRALDATA(info)
    _ = REDSHIFT(class_)
    balmer_lines = ['H_gamma', 'H_delta',
                    'H_epsilon', 'H_8', 'H_9', 'H_10', 'H_11', 'H_12', 'H_13', 'H_14']

    windows = [50, 50, 50, 50, 50, 40, 40, 15, 40, 40]
    fits, comps, stamps, emcee = [], [], [], []
    EW_lines = []
    data = read_data(class_)
    flux = data[1]
    corrected_flux = flux.copy()
    sigmas = first_sigma_est(class_, data, plot=True)
    for line, window in zip(balmer_lines, windows):
        masked_data = mask_nonuse_emission_line(class_, sigmas, line,
                                                [data[0], corrected_flux,
                                                 data[2]])
        isolated_data = isolate_emission_line(class_, line, window=window,
                                              datas=masked_data)
        stamps.append(isolated_data)

        fit = model(class_, line, isolated_data, sigmas)
        fits.append(fit)
        comp = fit.eval_components(x=isolated_data[0])
        comps.append(comp)

        model_ = model_mcmc(class_, line, isolated_data)
        emcee.append(model_)

    for i, _ in enumerate(balmer_lines):
        EW_line = emcee[i][2]
        EW_lines.append(EW_line)

    EW_med = np.median(EW_lines[:5])

    for i, line in enumerate(balmer_lines):
        gaus_ew = neg_gauss_EW(class_, line, data[0], emcee[i], EW_med, i)
        gaus_2  = neg_gauss_EW(class_, line, data[0], emcee[i], EW_lines[i], i)
        if np.min(gaus_2) < np.min(gaus_ew):
            param_key = f"{line}_height"
            A_val = fits[i].params[param_key].value
            corrected_flux -= gaus_2
        else:
            corrected_flux -= gaus_ew

    absorption_beta = neg_gauss_EW(class_, 'H_beta', data[0], emcee[0], EW_med, 0)
    absorption_alpha = neg_gauss_EW(class_, 'H_alpha', data[0], emcee[0], EW_med, 0)
    absorption = absorption_alpha + absorption_beta

    corrected_flux -= absorption

    plt.figure(figsize=(14, 4*len(balmer_lines)))

    for i, line in enumerate(balmer_lines[:5]):
        param_key = f"{line}_height"

        A_val = -fits[i].params[param_key].value
        A_err = fits[i].params[param_key].stderr

        wave = stamps[i][0]
        flux = stamps[i][1]
        mask = emcee[i][4]
        popt_mc = emcee[i][5]

        wave_to_plot = np.linspace(np.min(stamps[i][0]),
                                   np.max(stamps[i][0]),
                                   100)

        # Left panel: main fit
        plt.subplot(len(balmer_lines[:5]), 2, 2*i + 1)

        plt.step(wave, flux, label='Data', color='black')

        plt.plot(
            wave,
            comps[i][line + '_'] + comps[i]['polynomial'],
            label='Gaussian Fit',
            color='red'
        )

        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Flux')


        if A_err is None:
            A_text = f"A: {A_val:.2f} ± N/A"
        else:
            A_text = f"A: {A_val:.2f} ± {A_err:.2f}"

        plt.title(f"Absorption Line: {line}\n{A_text}")
        plt.legend()
        plt.ylim(np.min(comps[i][line + '_'] + comps[i]['polynomial']) - 2,
                 np.max(comps[i][line + '_'] + comps[i]['polynomial']) + 2)

        # Right panel: MC fits
        plt.subplot(len(balmer_lines), 2, 2*i + 2)

        plt.plot(wave[mask], flux[mask], "rx", label='Masked data')

        for j in range(100):
            plt.plot(
                wave_to_plot,
                absorption_func(class_, line, wave_to_plot, *popt_mc[j]),
                "k-",
                alpha=0.05
            )

        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Flux')
        plt.title(
            f"A = {emcee[i][0]:.2f} ± {emcee[i][1]:.2f}\n"
            f"EW = {emcee[i][2]:.2f} ± {emcee[i][3]:.2f} Å"
        )

        plt.legend()

    plt.tight_layout()
    # plt.savefig(f'{proj_DIR}bal_abs/balmer_fits_{class_.gal_id}_.pdf',
    #             format='pdf')
    plt.show()

    # absorption_beta = neg_gauss_EW(class_, 'H_beta', data[0], emcee[0], EW_med, 0)
    # absorption_alpha = neg_gauss_EW(class_, 'H_alpha', data[0], emcee[0], EW_med, 0)
    # absorption = absorption_alpha + absorption_beta

    # corrected_flux += absorption

    # save_corrected_data(class_, data, corrected_flux)

    return f'Balmer absorption correction applied and results saved for {class_.gal_id}.'


# =============================================================================
#
# Program
#
# =============================================================================

# get EW estimated from other hydrogen lunes, with fixed sigma
# with that you can create the gaussian fro h alpha and h beta
# correct the spectra
# make some estimations about the change in fluxes for h alpha and h beta
if __name__ == '__main__':

    J0328_SPEC = {
                 'DIR': '/Users/javieratoro/Desktop/thesis/BAADE_DATA/testing/35-J0328/',
                 'FILES': ['no_very_flats/J0328_NO_VERY_FLATS_tellcorr.fits'],
                 'redshift': 0.086,
                 'names': ['J0328+0031'],
                 'mass': 9.8
                 }

    J0020_SPECTRA= {
                    'DIR':'/Users/javieratoro/Desktop/thesis/BAADE_DATA/testing/10-J0020/',
                    'FILES': ['no_very_flats/J0020_NO_VERY_FLATS_tellcorr.fits'],
                    'redshift': 0.106,
                    'names':['J0020+0030'],
                    'mass':9.6
    }

    J0203_SPECTRA= {
    'DIR':'/Users/javieratoro/Desktop/thesis/BAADE_DATA/testing/25-J0203/',
    'FILES': ['no_very_flats/J0203_NO_VERY_FLATS_tellcorr.fits'],# 'twilight/J0203_TWILIGHT_tellcorr.fits'],
     'redshift': 0.156,
     'names':['J0203+0035'],
     'mass':9.96
    }

    J0243_SPECTRA= {
    'DIR':'/Users/javieratoro/Desktop/thesis/BAADE_DATA/testing/28-J0243/',
    'FILES': ['no_very_flats/J0243_NO_VERY_FLATS_tellcorr.fits'],# 'twilight/J0243_TWILIGHT_tellcorr.fits'],
     'redshift': 0.134,
     'names':['J0243+0111'],
     'mass':9.7
    }

    J0333_SPECTRA= {
        'DIR':'/Users/javieratoro/Desktop/thesis/BAADE_DATA/testing/36-J0033/',
        'FILES': ['no_very_flats/J0033_NO_VERY_FLATS_tellcorr.fits'], #'twilight/J0033_TWILIGHT_tellcorr.fits'],
        'redshift': 0.194,
        'names':['J0333+0017'],
        'mass':9.9
    }

    J0404_SPECTRA= {
        'DIR':'/Users/javieratoro/Desktop/thesis/BAADE_DATA/testing/38-J0404/',
        'FILES': ['no_very_flats/J0404_NO_VERY_FLATS_tellcorr.fits', 'twilight/J0404_TWILIGHT_tellcorr.fits'],
        'redshift': 0.066,
        'names':['J0404+0538'],
        'mass':10.2
    }

    J2204_SPECTRA= {
        'DIR':'/Users/javieratoro/Desktop/thesis/BAADE_DATA/testing/2-J2204/',
        'FILES': ['no_very_flats/J2204_NO_VERY_tellcorr.fits', 'twilight/J2204_TWILIGHT_tellcorr.fits'],
        'redshift': 0.185,
        'names':['J2204+0058'],
        'mass':10.16
    }

    J2258_SPECTRA= {
                    'DIR': '/Users/javieratoro/Desktop/thesis/BAADE_DATA/testing/6-J2258/',
                    'FILES': ['twilight/J2258_TWILIGHT_tellcorr.fits'], #'no_very_flats/J2258_NO_VERY_FLATS_tellcorr.fits'],
                    'redshift': 0.094,
                    'names': ['J2258+0056'],
                    'mass': 9.6
    }

    J2336_SPECTRA= {
        'DIR':'/Users/javieratoro/Desktop/thesis/BAADE_DATA/testing/7-J2336/',
        'FILES': ['no_very_flats/J2336_NO_VERY_BLUE_tellcorr.fits', 'twilight/J2336_TWILIGHT_tellcorr.fits'],
        'redshift':0.17047114835326904,
        'names':['J2336-0042'],
        'mass':9.9
    }
    # balmer_absorption_correction(J0328_SPEC)
    # balmer_absorption_correction(J0020_SPECTRA)
    # balmer_absorption_correction(J0203_SPECTRA)
    # balmer_absorption_correction(J0243_SPECTRA)
    # balmer_absorption_correction(J0333_SPECTRA)
    # balmer_absorption_correction(J0404_SPECTRA)
    # balmer_absorption_correction(J2204_SPECTRA)
    # balmer_absorption_correction(J2258_SPECTRA)
    # balmer_absorption_correction(J2336_SPECTRA)
