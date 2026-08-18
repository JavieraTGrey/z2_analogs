# Equivalent width calculation and plotting functions

import astropy.constants as const
from corr import REDSHIFT, SPECTRALDATA
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameter
from lmfit.models import GaussianModel, PolynomialModel
import os
import pandas as pd
from GALAXIES import get_galaxy, list_galaxies
from GaussianFitting import fitSpectrum

proj_DIR = '/Users/javieratoro/Desktop/thesis/proyecto 2024-2/'
balmer_lines = ['H1_4340A', 'H1_4102A',
                'H1_3970A', 'H1_3889A', 'H1_3835A', 'H1_3798A', 'H1_3771A', 'H1_3750A']


# Read data for galaxies

def read_data(class_):
    """
    Reads MW dust corrected data if it exists, otherwise reads the
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


# SIgma estimation for line masking
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
                      weights=1 / sigma**2,
                      showPlot=plot,
                      broad=True, nfev=1000)

    sigma_narrow = fit.params['sigma_v_narrow']
    sigma_broad = fit.params['sigma_v_broad']

    return sigma_narrow, sigma_broad


def get_sigma(line, class_, sigmas):
    sigma_narr, sigma_broad = sigmas
    bright_lines = ['O2_3726A', 'O2_3729A', 'H1_6563A', 'H1_4861A', 'H1_4340A',
                    'O3_5007A', 'O3_4959A', 'N2_6548A', 'N2_6583A', 'S2_6716A',
                    'S2_6731A']
    center = class_.linelist_dict[line] * (1 + class_.redshift)
    if line in bright_lines:
        sigma = (center / const.c.to('km/s').value) * sigma_broad.value
    else:
        sigma = (center / const.c.to('km/s').value) * sigma_narr.value
    return sigma


# Mask emission lines other than Hidriogen lines
def mask_nonuse_emission_line(class_, sigmas, line, data):
    """
    Mask every emission line other than `line`, 2.5 sigma around its
    center, setting flux to NaN and sigma to 0 in that window.

    Parameters:
    - class_: The spectral data class.
    - sigmas: A tuple containing the velocity dispersions (sigma) for
                narrow and broad components.
    - line: The emission line to keep (e.g., 'H_alpha').
    - data: A tuple containing the wavelength, flux, and sigma arrays.

    Returns:
    - masked_wave, masked_flux, masked_sigma
    """
    wave, flux, err = data

    cte = (1 + class_.redshift)
    masked_flux = flux.copy()
    masked_sigma = err.copy()

    for label in class_.linelist_dict:
        if label == line:
            continue
        center = class_.linelist_dict[label] * cte
        sigma = get_sigma(label, class_, sigmas)

        size = 4
        mask_line = (wave > center - size * sigma) & \
                    (wave < center + size * sigma)

        masked_flux[mask_line] = np.nan
        masked_sigma[mask_line] = 0.0

    return wave, masked_flux, masked_sigma


# Isolate hidrogen line that we are studying
def isolate_emission_line(class_, line, window, datas):
    """
    Isolate the emission line from the spectral data.

    Parameters:
    - class_: The spectral data class.
    - line: Name of the line to isolate (key into class_.linelist_dict).
    - window: The width around the line center to consider for
              isolation (in Angstroms).
    - datas: (wave, flux, sigma) tuple.

    Returns:
    - isolated_wave, isolated_flux, isolated_sigma
    """
    wave, flux, sigma = datas

    line_center = class_.linelist_dict[line] * (1 + class_.redshift)
    lower_bound = line_center - window
    upper_bound = line_center + window

    mask = (wave >= lower_bound) & (wave <= upper_bound)

    return wave[mask], flux[mask], sigma[mask]


def model(class_, label, datas, sigmas):
    """
    Create the model for the absorption with a negative gaussian
    and a 1-degree polynomial continuum, and fit it with lmfit.
    """
    wave, flux_, err = datas

    lam = class_.linelist_dict[label]
    center = lam * (1 + class_.redshift)
    sigma = get_sigma(label, class_, sigmas)

    size = 3.5
    mask_line = (wave > center - size * sigma) & (wave < center + size * sigma)

    flux = flux_.copy()
    flux[mask_line] = np.nan

    gaussian = GaussianModel(prefix=label + '_')

    polydeg = 1
    polynomial = PolynomialModel(degree=polydeg)

    comp_mult = gaussian + polynomial
    pars_mult = comp_mult.make_params()

    pars_mult.add(name='z', value=class_.redshift, vary=False)
    pars_mult.add(name='sigma_v', value=550, vary=False)

    for param in ['center', 'amplitude', 'sigma']:
        narrow_key = f'{label}_{param}'
        if param == 'center':
            value, vary_, min_, max_ = lam, False, None, None
            expr = f'{lam:6.2f}*(1+z)'
        elif param == 'amplitude':
            value, vary_, min_, max_ = -80, True, -100, 0
            expr = None
        else:  # sigma
            value, vary_, min_, max_ = None, False, None, None
            expr = f'(sigma_v/3e5)*{label}_center'
        pars_mult[narrow_key] = Parameter(name=narrow_key, value=value,
                                          vary=vary_, expr=expr,
                                          min=min_, max=max_)
    for i in range(polydeg + 1):
        pars_mult[f'c{i:1.0f}'].set(value=0)

    out_comp_mult = comp_mult.fit(flux, pars_mult,
                                  x=wave, weights=1 / err**2,
                                  nan_policy='omit',
                                  max_nfev=1000)

    return out_comp_mult


def model_mcmc(class_, label, stamp, n_mc=100, rng=None):

    wave, flux, f_err = stamp[0], stamp[1], stamp[2]
    z, w_center = class_.redshift, class_.linelist_dict[label]

    mu = w_center * (1 + z)
    sigma = w_center * (1 + z) * 550 / 3e5

    step = 1 * sigma
    mask = np.isfinite(flux) & ((wave < mu - step) | (wave > mu + step))

    w = wave[mask]
    gauss_term = -np.exp(-((mu - w) ** 2) / (2 * sigma**2))
    design = np.column_stack([gauss_term, w, np.ones_like(w)])

    # best-fit on the actual data
    popt, *_ = np.linalg.lstsq(design, flux[mask], rcond=None)

    # Monte Carlo over flux uncertainties, all realizations solved at once
    rng = np.random.default_rng() if rng is None else rng
    f_mc = rng.normal(loc=flux[mask], scale=f_err[mask],
                      size=(n_mc, mask.sum()))
    popt_mc, *_ = np.linalg.lstsq(design, f_mc.T, rcond=None)
    popt_mc = popt_mc.T

    A_mean = np.median(popt_mc[:, 0])
    A_err = np.std(popt_mc[:, 0])

    EW_mc = (
        (popt_mc[:, 0] * np.sqrt(2 * np.pi * sigma**2))
        / (popt_mc[:, 1] * w_center * (1 + z) + popt_mc[:, 2])
        / (1 + z)
    )

    return A_mean, A_err, np.median(EW_mc), np.std(EW_mc), mask, popt_mc


def save_corrected_data(class_, data, new_flux):
    """
    Saves the corrected data to a CSV file, creating the directory if
    needed.
    """
    save_dir = f'{proj_DIR}bal_abs'
    save_path = f'{save_dir}/bcorr_{class_.gal_id}_new.csv'

    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame({'wave': data[0], 'flux': new_flux,
                       'sigma': data[2]})
    df.to_csv(save_path, index=False)

    print(f'Saved Balmer Absorption corrected data to: {save_path}')


def refit_lines_on_flux(class_, lines, windows, flux, data, sigmas):
    """
    Run the isolate -> model -> model_mcmc

    Returns (fits, comps, stamps, emcee).
    """
    fits, comps, stamps, emcee = [], [], [], []

    for line, window in zip(lines, windows):
        masked_data = mask_nonuse_emission_line(
            class_, sigmas, line, [data[0], flux, data[2]])
        isolated_data = isolate_emission_line(
            class_, line, window=window, datas=masked_data)
        stamps.append(isolated_data)

        fit = model(class_, line, isolated_data, sigmas)
        fits.append(fit)
        comps.append(fit.eval_components(x=isolated_data[0]))

        emcee.append(model_mcmc(class_, line, isolated_data))

    return fits, comps, stamps, emcee


def neg_gauss(class_, line, w, A):
    w_center = class_.linelist_dict[line]
    z = class_.redshift
    mu = w_center * (1 + z)
    sigma = w_center * (1 + z) * 550 / 3e5
    return -A * np.exp(-((mu - w) ** 2) / (2 * sigma**2))


def absorption_func(class_, line, w, A, m, n):
    return neg_gauss(class_, line, w, A) + m * w + n


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
         / (np.sqrt(2 * np.pi) * sigma))
    return -A * np.exp(-((mu - w) ** 2) / (2 * sigma**2))


def plot_balmer_fits(class_, balmer_lines, stamps, comps, fits, emcee,
                     n_show=5, save_path=None, title_prefix=None):

    prefix = '' if title_prefix is None else f'{title_prefix}\n'

    n_lines = min(n_show, len(balmer_lines))

    fig = plt.figure(figsize=(10, 4 * n_lines))

    for i, line in enumerate(balmer_lines[:n_lines]):
        wave = stamps[i][0]
        line_flux = stamps[i][1]
        mask = emcee[i][4]
        popt_mc = emcee[i][5]
        n_mc = popt_mc.shape[0]

        wave_to_plot = np.linspace(np.min(stamps[i][0]),
                                   np.max(stamps[i][0]),
                                   100)

        # Left panel: main fit
        plt.subplot(n_lines, 2, 2 * i + 1)

        plt.step(wave, line_flux, label='Data', color='black')

        fit_curve = comps[i][line + '_'] + comps[i]['polynomial']
        plt.plot(wave, fit_curve, label='Gaussian Fit', color='red')

        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Flux')

        param_key = f"{line}_height"
        A_val = -fits[i].params[param_key].value
        A_err = fits[i].params[param_key].stderr
        A_text = (f"A: {A_val:.2f} ± N/A" if A_err is None
                  else f"A: {A_val:.2f} ± {A_err:.2f}")

        plt.title(f"{prefix}Absorption Line: {line}\n{A_text}")
        plt.legend()
        plt.ylim(np.min(fit_curve) - 2, np.max(fit_curve) + 2)

        # Right panel: MC fits
        plt.subplot(n_lines, 2, 2 * i + 2)

        plt.plot(wave[mask], line_flux[mask], "rx", label='Masked data')

        for j in range(n_mc):
            plt.plot(
                wave_to_plot,
                absorption_func(class_, line, wave_to_plot, *popt_mc[j]),
                "k-",
                alpha=0.05
            )

        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Flux')
        plt.title(
            f"{prefix}A = {emcee[i][0]:.2f} ± {emcee[i][1]:.2f}\n"
            f"EW = {emcee[i][2]:.2f} ± {emcee[i][3]:.2f} Å"
        )
        plt.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format='pdf')
    plt.show()

    return fig


def balmer_absorption_correction(info):
    """
    Applies the Balmer absorption correction to the flux data using the
    MCMC results from the model fitting, plots the fits, and saves the
    corrected spectrum.
    """
    class_ = SPECTRALDATA(info)
    _ = REDSHIFT(class_)
    lines = ['H1_4340A', 'H1_4102A', 'H1_3970A', 'H1_3889A', 'H1_3835A',
            'H1_3798A', 'H1_3771A', 'H1_3750A', 'H1_3734A', 'H1_3722A']
    windows = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]

    fits, comps, stamps, emcee = [], [], [], []
    data = read_data(class_)
    corrected_flux = data[1].copy()
    sigmas = first_sigma_est(class_, data, plot=True)

    for line, window in zip(lines, windows):
        masked_data = mask_nonuse_emission_line(
            class_, sigmas, line, [data[0], corrected_flux, data[2]])
        isolated_data = isolate_emission_line(
            class_, line, window=window, datas=masked_data)
        stamps.append(isolated_data)

        fit = model(class_, line, isolated_data, sigmas)
        fits.append(fit)
        comps.append(fit.eval_components(x=isolated_data[0]))

        emcee.append(model_mcmc(class_, line, isolated_data))

    EW_lines = [emcee[i][2] for i in range(len(lines))]
    EW_med = np.median(EW_lines[:5])

    for i, line in enumerate(lines):
        g_ew = neg_gauss_EW(class_, line, data[0], emcee[i], EW_med, i)
        g_2 = neg_gauss_EW(class_, line, data[0], emcee[i], EW_lines[i], i)
        corrected_flux -= g_2 if np.min(g_2) < np.min(g_ew) else g_ew

    plot_balmer_fits(
        class_, lines, stamps, comps, fits, emcee, n_show=5,
        save_path=f'{proj_DIR}bal_abs/balmer_fits_{class_.gal_id}_.pdf',
        title_prefix='Before correction'
    )

    absorption_alpha = neg_gauss_EW(class_, 'H1_6563A', data[0],
                                    emcee[0], EW_med, 0)
    absorption_beta = neg_gauss_EW(class_, 'H1_4861A', data[0],
                                   emcee[0], EW_med, 0)
    corrected_flux -= (absorption_alpha + absorption_beta)

    fits_corr, comps_corr, stamps_corr, emcee_corr = refit_lines_on_flux(
        class_, lines, windows, corrected_flux, data, sigmas)

    plot_balmer_fits(
        class_, lines, stamps_corr, comps_corr, fits_corr, emcee_corr,
        n_show=5,
        save_path=f'{proj_DIR}bal_abs/balmer_fits_{class_.gal_id}_corr.pdf',
        title_prefix='After correction'
    )

    save_corrected_data(class_, data, corrected_flux)

    return f'{class_.gal_id}: Balmer absorption correction applied and saved.'


# =============================================================================
#
# Program
#
# =============================================================================
# if __name__ == '__main__':
#     galaxies = list_galaxies()
#     for gal in galaxies:
#         info = get_galaxy(gal)
#         balmer_absorption_correction(info)
