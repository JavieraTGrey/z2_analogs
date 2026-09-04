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
from em_lines import model_bright_lines
from GaussianFitting import fitSpectrum

proj_DIR = '/Users/javieratoro/Desktop/thesis/proyecto 2024-2/'

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
def isolate_emission_line(class_, line, datas):
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
    if line in ['H1_4340A', 'H1_4102A', 'H1_3970A', 'H1_3889A',
                'H1_6563A', 'H1_4861A']:
        lower_bound = line_center - 50
        upper_bound = line_center + 50
    elif line == 'H1_3835A':
        lower_bound = line_center - 35
        upper_bound = line_center + 35
    else:
        lower_bound = line_center - 20
        upper_bound = line_center + 20

    mask = (wave >= lower_bound) & (wave <= upper_bound)

    return wave[mask], flux[mask], sigma[mask]


def model(class_, label, datas, sigmas, vel=550):
    """
    Create the model for the absorption with a negative gaussian
    and a 1-degree polynomial continuum, and fit it with lmfit.
    """
    wave, flux_, err = datas

    lam = class_.linelist_dict[label]
    center = lam * (1 + class_.redshift)
    sigma = get_sigma(label, class_, sigmas)

    size = 3
    mask_line = (wave > center - size * sigma) & (wave < center + size * sigma)

    flux = flux_.copy()
    flux[mask_line] = np.nan

    gaussian = GaussianModel(prefix=label + '_')

    polydeg = 1
    polynomial = PolynomialModel(degree=polydeg)

    comp_mult = gaussian + polynomial
    pars_mult = comp_mult.make_params()

    pars_mult.add(name='z', value=class_.redshift, vary=False)
    pars_mult.add(name='sigma_v', value=vel, vary=False)

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


def model_mc(class_, label, isolated_data, sigmas, n_mc=100):
    wave, flux, sigma = isolated_data
    fit_mc, comps = [], []
    As = []
    for i in range(n_mc):
        yoff = flux + np.random.randn(len(flux)) * sigma
        fit = model(class_, label, (wave, yoff, sigma), sigmas)
        fit_mc.append(fit)
        comps.append(fit.eval_components(x=wave))
        param_key = f"{label}_height"
        height = -1*fit.params[param_key].value
        m = fit.params['c1'].value
        n = fit.params['c0'].value
        As.append((height, m, n))
    return fit_mc, comps, As


def get_EW_from_As(class_, label, As, vel=550):

    w_center = class_.linelist_dict[label]
    z = class_.redshift
    mu = w_center * (1 + z)
    sigma = w_center * (1 + z) * vel / 3e5

    As = np.array(As)
    height, m, n = As[:, 0], As[:, 1], As[:, 2]

    continuum_at_mu = m * mu + n
    EW = (height * np.sqrt(2 * np.pi * sigma**2)
          / continuum_at_mu
          / (1 + z))

    return EW


def get_corrected_ews(scale, balmer_lines):
    balmer_lines = {
        'H1_6563A': (0.930),
        'H1_4861A': (1.0),
        'H1_4340A': (0.964),
        'H1_4102A': (0.927),
        'H1_3889A': (0.893),
        'H1_3835A': (0.868),
        'H1_3798A': (0.824),
        'H1_3771A': (0.778),
        'H1_3750A': (0.750)}

    final_EWs = {}

    for _, (line_name, ratio) in balmer_lines.items():
        final_EWs[line_name] = ratio * scale

    return final_EWs


def model_mc_scipy(class_, label, stamp, n_mc=100, rng=None):

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
    Run the isolate -> model -> model_mc_scipy

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

        emcee.append(model_mc_scipy(class_, line, isolated_data))

    return fits, comps, stamps, emcee


def neg_gauss(class_, line, w, A):
    w_center = class_.linelist_dict[line]
    z = class_.redshift
    mu = w_center * (1 + z)
    sigma = w_center * (1 + z) * 550 / 3e5
    return -A * np.exp(-((mu - w) ** 2) / (2 * sigma**2))


def absorption_func(class_, line, w, A, m, n):
    return neg_gauss(class_, line, w, A) + m * w + n


def neg_gauss_EW(class_, line, w, As, EW_, num, vel=550):
    w_center = class_.linelist_dict[line]
    z = class_.redshift
    mu = w_center * (1 + z)
    sigma = w_center * (1 + z) * vel / 3e5

    _, m, n = As[num]

    continuum_at_mu = m * mu + n
    A = (EW_ * (1 + z)
         * continuum_at_mu
         / (np.sqrt(2 * np.pi) * sigma))
    return -A * np.exp(-((mu - w) ** 2) / (2 * sigma**2))


def balmer_absorption_correction(info):
    """
    Applies the Balmer absorption correction to the flux data using the
    MCMC results from the model fitting, plots the fits, and saves the
    corrected spectrum.
    """
    class_ = SPECTRALDATA(info)
    _ = REDSHIFT(class_)
    lines = ['H1_4340A', 'H1_4102A', 'H1_3970A', 'H1_3835A',
             'H1_3798A', 'H1_3771A', 'H1_3750A', 'H1_3734A', 'H1_3722A']

    fits, comps, stamps, emcee = [], [], [], []
    data = read_data(class_)
    corrected_flux = data[1].copy()
    sigmas = first_sigma_est(class_, data, plot=True)

    for line in lines:
        print(lines)
        masked_data = mask_nonuse_emission_line(
            class_, sigmas, line, [data[0], corrected_flux, data[2]])
        isolated_data = isolate_emission_line(
            class_, line, datas=masked_data)
        stamps.append(isolated_data)

        fit = model(class_, line, isolated_data, sigmas)
        fits.append(fit)
        comps.append(fit.eval_components(x=isolated_data[0]))

        emcee.append(model_mc_scipy(class_, line, isolated_data))

    EW_lines = [emcee[i][2] for i in range(len(lines))]
    EW_med = np.median(EW_lines[:5])

    for i, line in enumerate(lines):
        g_ew = neg_gauss_EW(class_, line, data[0], emcee[i], EW_med, i)
        g_2 = neg_gauss_EW(class_, line, data[0], emcee[i], EW_lines[i], i)
        corrected_flux -= g_2 if np.min(g_2) < np.min(g_ew) else g_ew

    fig = plot_balmer_fits(
        class_, lines, stamps, comps, fits, emcee, n_show=5,
        save_path=f'{proj_DIR}bal_abs/balmer_fits_{class_.gal_id}_.pdf',
        title_prefix='Before correction'
    )

    # absorption_alpha = neg_gauss_EW(class_, 'H1_6563A', data[0],
    #                                 emcee[0], EW_med, 0)
    # absorption_beta = neg_gauss_EW(class_, 'H1_4861A', data[0],
    #                                emcee[0], EW_med, 0)
    # corrected_flux -= (absorption_alpha + absorption_beta)

    # fits_corr, comps_corr, stamps_corr, emcee_corr = refit_lines_on_flux(
    #     class_, lines, windows, corrected_flux, data, sigmas)

    # fig = plot_balmer_fits(
    #     class_, lines, stamps_corr, comps_corr, fits_corr, emcee_corr,
    #     n_show=5,
    #     save_path=f'{proj_DIR}bal_abs/balmer_fits_{class_.gal_id}_corr.pdf',
    #     title_prefix='After correction'
    # )

    # save_corrected_data(class_, data, corrected_flux)

    # return f'{class_.gal_id}: Balmer absorption correction applied and saved.'
    return fits, comps, stamps, emcee, fig


def get_stamps(class_, data, lines):
    wave, flux, err = data[0], data[1], data[2]
    lines_df = class_.line_list
    cte = (1 + class_.redshift)
    stamps = []
    for bright_line in lines:
        stamp = flux.copy()
        for line_label in lines:
            if bright_line == line_label:
                continue
            if line_label == 'He1_5016A':
                continue
            info = lines_df[lines_df['name'] == line_label]
            w3, w4 = info[['w3', 'w4']].values[0]

            mask_line = (wave > w3*cte) & (wave < w4*cte)
            stamp[mask_line] = np.nan
        info_bright = lines_df[lines_df['name'] == bright_line]
        w1, w6 = info_bright[['w1', 'w6']].values[0]
        mask_stamp = (wave > w1*cte) & (wave < w6*cte)
        stamps.append((wave[mask_stamp], stamp[mask_stamp], err[mask_stamp]))
    return stamps


def model_emission_lines(class_, red, lines, data, corrected_data, num_mc=100):
    stamps_flux = get_stamps(class_, data, lines)
    results = {}
    for i, line in enumerate(lines):
        fits_or, fits_corr = [], []
        As_or, As_corr = [], []
        for j in range(num_mc):
            yoff_or = stamps_flux[i][1] + np.random.randn(len(stamps_flux[i][1])) * stamps_flux[i][2]
            yoff_corr = corrected_data[i][1] + np.random.randn(len(corrected_data[i][1])) * corrected_data[i][2]
            fit_or = model_bright_lines(red,
                                        (stamps_flux[i][0],
                                         yoff_or, stamps_flux[i][2]),
                                        [line], plot=True)
            fit_corr = model_bright_lines(red,
                                          (corrected_data[i][0],
                                           yoff_corr, corrected_data[i][2]),
                                          [line], plot=True)
            fits_or.append(fit_or)
            fits_corr.append(fit_corr)
            A_or = fit_or.params[f'{line}_broad_amplitude'].value + fit_or.params[f'{line}_narrow_amplitude'].value
            A_corr = fit_corr.params[f'{line}_broad_amplitude'].value + fit_corr.params[f'{line}_narrow_amplitude'].value
            As_or.append(A_or)
            As_corr.append(A_corr)

        As_or = np.array(As_or)
        As_corr = np.array(As_corr)
        pct_diffs = (As_corr - As_or) / As_or * 100

        diff = np.median(As_corr) - np.median(As_or)
        combined_err = np.sqrt(np.std(As_corr)**2 + np.std(As_or)**2)
        significance = diff / combined_err

        results[line] = {
            'fits_or': fits_or,
            'fits_corr': fits_corr,
            'As_or': As_or,
            'As_corr': As_corr,
            'pct_diffs': pct_diffs,
            'significance': significance,
        }
    return results


def plot_pct_diff_histograms(results, save_path=None):
    lines = list(results.keys())
    ncols = 3
    nrows = int(np.ceil(len(lines) / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows),
                            squeeze=False)
    axs_flat = axs.flatten()

    for i, line in enumerate(lines):
        pct = results[line]['pct_diffs']
        sig = results[line]['significance']
        ax = axs_flat[i]
        ax.hist(pct, bins=30, color='steelblue', alpha=0.8)
        ax.axvline(np.median(pct), color='red', linestyle='--')
        ax.set_title(line)
        ax.set_xlabel('% difference (corrected vs original)')
        ax.text(0.05, 0.9, f'σ = {sig:.2f}', transform=ax.transAxes,
                fontsize=10, fontweight='bold')

    for k in range(len(lines), len(axs_flat)):
        axs_flat[k].axis('off')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format='pdf')
    plt.show()
    return fig


def get_EW_array_from_As(class_, label, As, vel=550):
    w_center = class_.linelist_dict[label]
    z = class_.redshift
    mu = w_center * (1 + z)
    sigma = w_center * (1 + z) * vel / 3e5

    As = np.array(As)
    height, m, n = As[:, 0], As[:, 1], As[:, 2]

    continuum_at_mu = m * mu + n
    EW = (height * np.sqrt(2 * np.pi * sigma**2)
          / continuum_at_mu
          / (1 + z))
    return EW


def get_EW_from_As_(class_, label, As, vel=550):
    EW = get_EW_array_from_As(class_, label, As, vel=vel)
    return np.mean(EW), np.std(EW)


def plot_all_fits(class_, balmer_lines, stamps, params,
                  n_show=None, save_path=None, ylim=(5, 30)):

    n_lines = len(balmer_lines) if n_show is None else min(n_show,
                                                           len(balmer_lines))

    fig, axs = plt.subplots(n_lines, 2, figsize=(10, 3.5 * n_lines),
                            squeeze=False)

    for i in range(n_lines):
        line = balmer_lines[i]
        wave, flux, _ = stamps[i]
        As = np.array(params[i])
        n_mc = As.shape[0]
        median_params = np.median(As, axis=0)

        wave_to_plot = np.linspace(np.min(wave), np.max(wave), 200)

        # Left: data + subset of MC curves
        plot_idx = np.random.choice(n_mc, size=min(100, n_mc), replace=False)
        axs[i, 0].step(wave, flux, color='grey', label='Data')
        for j in plot_idx:
            axs[i, 0].plot(wave_to_plot,
                           absorption_func(class_, line, wave_to_plot, *As[j]),
                           'k-', alpha=0.05)
        axs[i, 0].set_title(f'{line}: MC iterations')
        axs[i, 0].set_xlabel('Wavelength (Å)')
        axs[i, 0].set_ylabel('Flux')
        axs[i, 0].set_ylim(*ylim)
        axs[i, 0].legend()

        # Right: data + median fit
        axs[i, 1].step(wave, flux, color='grey', label='Data')
        axs[i, 1].plot(wave_to_plot,
                       absorption_func(class_,
                                       line, wave_to_plot, *median_params),
                       color='red', label='Median fit')
        axs[i, 1].set_title(f'{line}: median fit')
        axs[i, 1].set_xlabel('Wavelength (Å)')
        axs[i, 1].set_ylabel('Flux')
        axs[i, 1].set_ylim(*ylim)
        axs[i, 1].legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format='png')
    plt.show()
    return fig


def plot_ew_histograms(class_, balmer_lines, params, save_path=None):
    EW_all_ratios = {
        'H1_6563A': 0.930,
        'H1_4861A': 1.0,
        'H1_4340A': 0.964,
        'H1_4102A': 0.927,
        'H1_3889A': 0.893,
        'H1_3835A': 0.868,
        'H1_3798A': 0.824,
        'H1_3771A': 0.778,
        'H1_3750A': 0.750}
    n_lines = len(balmer_lines)
    ncols = 3
    nrows = int(np.ceil(n_lines / ncols))

    fig = plt.figure()
    # axs_flat = axs.flatten()

    for i, line in enumerate(balmer_lines):
        EW_arr = get_EW_array_from_As(class_, line, params[i])
        EW_hb = EW_arr / EW_all_ratios[line]
        # ax = axs_flat[i]
        plt.hist(EW_hb, bins=30, alpha=0.8)
        plt.axvline(np.mean(EW_hb), color='red', linestyle='--',
                    label=f'mean={np.mean(EW_hb):.2f}')
        # plt.set_title(line)
        plt.xlabel(r'EW ($H_\beta$) from different lines')
        # plt.set_ylabel('Count')
        plt.legend(fontsize=8)

    # # hide unused subplots
    # for k in range(n_lines, len(axs_flat)):
    #     axs_flat[k].axis('off')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format='png')
    plt.show()
    return fig


def plot_gaussian_comparison(class_, line, w, As_line, EW_original,
                             EW_reconstructed, num=0, window=50,
                             stamp=None, ax=None, ylim=(5, 30)):

    w_center = class_.linelist_dict[line]
    z = class_.redshift
    mu = w_center * (1 + z)

    wave_to_plot = np.linspace(mu - window, mu + window, 300)

    g_original = neg_gauss_EW(class_, line, wave_to_plot, As_line,
                              EW_original, num=num)
    g_recon = neg_gauss_EW(class_, line, wave_to_plot, As_line,
                           EW_reconstructed, num=num)

    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(6, 4))

    if stamp is not None:
        wave, flux, _ = stamp
        # mask = (wave > mu - window) & (wave < mu + window)
        ax.step(wave * (1 + z), flux, color='black', alpha=0.3,
                label='Data')
        # ax.set_xlim(np.min(wave[mask]), np.max(wave[mask]))

    m, n = As_line[num][1], As_line[num][2]
    continuum = m * wave_to_plot + n

    ax.plot(wave_to_plot, continuum + g_original, color='tab:blue',
            label=f'Original fit (EW={EW_original:.2f} Å)')
    ax.plot(wave_to_plot, continuum + g_recon, color='tab:orange',
            linestyle='--',
            label=f'Reconstructed (EW={EW_reconstructed:.2f} Å)')
    ax.axvline(mu, color='gray', linestyle=':', alpha=0.5)
    ax.set_title(line)
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Flux')
    ax.legend(fontsize=8)
    ax.set_ylim(ylim[0], ylim[1])

    if own_ax:
        plt.tight_layout()
        plt.show()
        return fig
    return ax


def plot_all_gaussian_comparisons(class_, lines, idx_map,
                                  stamps, params, Ews, EW_new,
                                  ylim=(5, 30),
                                  save_path=None):

    n_panels = len(lines) + 2  # + Halpha + Hbeta
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                            squeeze=False)
    axs_flat = axs.flatten()

    panel = 0
    for line, idx in zip(lines, idx_map):
        plot_gaussian_comparison(
            class_, line, stamps[idx][0], params[idx],
            EW_original=np.median(Ews[idx]), EW_reconstructed=EW_new[line],
            stamp=stamps[idx], ax=axs_flat[panel], ylim=ylim
        )
        panel += 1

    for i, line in enumerate(('H1_6563A', 'H1_4861A')):
        data = read_data(class_)
        stamp_new = isolate_emission_line(class_, line, data)
        plot_gaussian_comparison(
            class_, line, None, params[i],
            EW_original=np.median(EW_new[line]), EW_reconstructed=EW_new[line],
            stamp=stamp_new, ax=axs_flat[panel], ylim=ylim
        )
        axs_flat[panel].set_title(f'{line} (reconstructed fit)')
        panel += 1

    for k in range(panel, len(axs_flat)):
        axs_flat[k].axis('off')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format='png')
    plt.show()
    return fig
