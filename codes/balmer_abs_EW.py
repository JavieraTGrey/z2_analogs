# Equivalent width calculation and plotting functions

import astropy.constants as const
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameter
from lmfit.models import GaussianModel, PolynomialModel
import os
import pandas as pd
from GaussianFitting import fitSpectrum
from dust_correction import E_BV_, get_flux, f_int

# from scipy.stats import gaussian_kde
# from scipy.signal import find_peaks


proj_DIR = '/Users/javieratoro/Desktop/thesis/proyecto 2024-2/'
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


# --------------------------------------------------------
# Auxiliar functions
# --------------------------------------------------------
def neg_gauss(class_, line, w, A):
    w_center = class_.linelist_dict[line]
    z = class_.redshift
    mu = w_center * (1 + z)
    sigma = w_center * (1 + z) * 550 / 3e5
    return -A * np.exp(-((mu - w) ** 2) / (2 * sigma**2))


def absorption_func(class_, line, w, A, m, n):
    return neg_gauss(class_, line, w, A) + m * w + n


def neg_gauss_EW(class_, line, w, As, EW_, vel=550):
    w_center = class_.linelist_dict[line]
    z = class_.redshift
    mu = w_center * (1 + z)
    sigma = w_center * (1 + z) * vel / 3e5

    _, m, n = np.mean(As, axis=0)

    continuum_at_mu = m * mu + n
    A = (EW_ * (1 + z)
         * continuum_at_mu
         / (np.sqrt(2 * np.pi) * sigma))
    return -A * np.exp(-((mu - w) ** 2) / (2 * sigma**2))


def neg_gauss_EW_single(class_, line, w, A_triplet, EW_, vel=550):
    w_center = class_.linelist_dict[line]
    z = class_.redshift
    mu = w_center * (1 + z)
    sigma = w_center * (1 + z) * vel / 3e5
    _, m, n = A_triplet
    continuum_at_mu = m * mu + n
    A = (EW_ * (1 + z) * continuum_at_mu) / (np.sqrt(2 * np.pi) * sigma)
    return -A * np.exp(-((mu - w) ** 2) / (2 * sigma**2))


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

# --------------------------------------------------------
# Working functions
# --------------------------------------------------------


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


# Sigma estimation for line masking
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


# Get sigma for either narrow or borad component
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


# Model single absorption feature
def model_single_abs(class_, label, datas, sigmas, vel=550):
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


# Monte Carlo modeling of single H line
def model_mc_abs(class_, label, isolated_data, sigmas, vel=550, n_mc=100):
    wave, flux, sigma = isolated_data
    fit_mc, comps = [], []
    As = []
    for i in range(n_mc):
        yoff = flux + np.random.randn(len(flux)) * sigma
        fit = model_single_abs(class_, label, (wave, yoff, sigma), sigmas, vel=vel)
        fit_mc.append(fit)
        comps.append(fit.eval_components(x=wave))
        param_key = f"{label}_height"
        height = -1*fit.params[param_key].value
        m = fit.params['c1'].value
        n = fit.params['c0'].value
        As.append((height, m, n))
    return fit_mc, comps, As


# Save corrected data
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


# Get EW from the MC params for a single H line
def get_EW_from_mc(class_, label, As, vel=550):

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


# Model all lines from a list and correct data
def model_from_list_lines(class_, data, sigmas,
                          lines, vel=550, n_mc=1000, min_amp_sigma=3.0):

    fits, fits_eval_stamp, stamps, params = {}, {}, {}, {}
    corrected_flux = data[1].copy()
    corr_stamps, EW_lines, EW_hb_samples, EW_hb_weights = {}, {}, [], []
    line_significance = {}
    corrected_all_flux = []

    for line in lines:
        masked_data = mask_nonuse_emission_line(
            class_, sigmas, line, [data[0], corrected_flux, data[2]])
        isolated_data = isolate_emission_line(
            class_, line, datas=masked_data)
        stamps[line] = isolated_data

        fit_mc, comps, param = model_mc_abs(class_, line, isolated_data,
                                            sigmas, vel=vel, n_mc=n_mc)
        param = np.asarray(param)
        params[line] = param
        fits[line] = fit_mc
        fits_eval_stamp[line] = comps

        absorption = neg_gauss(class_, line, data[0],
                               np.mean(param, axis=0)[0])
        corrected_flux -= absorption

        corr_stamps[line] = [isolated_data[0],
                             isolated_data[1] - comps[0][f'{line}_'],
                             isolated_data[2]]

        if line == 'H1_3970A':
            continue  # Skip H1_3970A for EW_hb_samples calculation

        EW_line = get_EW_from_mc(class_, line, param, vel=vel)
        EW_lines[line] = EW_line

        EW_hb_line = EW_line / EW_all_ratios[line]
        EW_hb_samples.append(EW_hb_line)

        # --- per-draw weight (amplitude vs local continuum noise) ---
        amplitude = param[:, 0]
        noise_level = np.median(isolated_data[2])
        significance = np.abs(amplitude) / noise_level
        weight_draw = np.clip(significance / min_amp_sigma, 0, 1)
        weight_draw = np.where(amplitude > 0, weight_draw, 0.0)

        # --- NEW: line-level weight (is the fit consistent with zero overall?) ---
        mean_amplitude = np.mean(amplitude)
        amplitude_std = np.std(amplitude)
        amplitude_std_safe = amplitude_std if amplitude_std > 0 else np.inf
        sig_line = np.abs(mean_amplitude) / amplitude_std_safe
        line_significance[line] = sig_line
        weight_line_level = np.clip(sig_line / min_amp_sigma, 0, 1)

        # combine: a line that's not significant overall gets down-weighted
        # across ALL its draws, on top of any per-draw discounting
        weight_line = weight_draw * weight_line_level
        EW_hb_weights.append(weight_line)

    EW_hb_stack = np.array(EW_hb_samples)          # shape (n_lines, n_mc)
    EW_hb_weight_stack = np.array(EW_hb_weights)   # shape (n_lines, n_mc)

    weight_sum = np.sum(EW_hb_weight_stack, axis=0)
    weight_sum_safe = np.where(weight_sum == 0, 1, weight_sum)
    EW_hb_der_mc = np.sum(EW_hb_stack * EW_hb_weight_stack, axis=0) / weight_sum_safe
    EW_hb_der_mc = np.where(weight_sum == 0,
                            np.mean(EW_hb_stack, axis=0),
                            EW_hb_der_mc)

    EW_hb_der = np.mean(EW_hb_der_mc)
    EW_hb_der_err = np.std(EW_hb_der_mc)

    line_der = ['H1_6563A', 'H1_4861A']
    for line in line_der:
        masked_data = mask_nonuse_emission_line(
            class_, sigmas, line, [data[0], corrected_flux, data[2]])
        isolated_data = isolate_emission_line(
            class_, line, datas=masked_data)
        stamps[line] = isolated_data

        fit_mc, comps, param = model_mc_abs(class_, line, isolated_data,
                                            sigmas, vel=vel, n_mc=n_mc)
        params[line] = param
        absorption_mc = np.zeros((n_mc, len(data[0])))
        for i in range(n_mc):
            EW_single = EW_hb_der_mc[i] * EW_all_ratios[line]
            absorption_mc[i] = neg_gauss_EW_single(class_, line, data[0],
                                                   param[i],
                                                   EW_single,
                                                   vel=vel)

        absorption_mean = np.mean(absorption_mc, axis=0)
        absorption_std = np.std(absorption_mc, axis=0)

        corrected_flux -= absorption_mean
        corrected_sigma_full = np.sqrt(data[2]**2 + absorption_std**2)

        absorption_std_small = np.interp(isolated_data[0], data[0],
                                         absorption_std)
        corrected_sigma_small = np.sqrt(isolated_data[2]**2 +
                                        absorption_std_small**2)

        corr_stamps[line] = [isolated_data[0],
                             isolated_data[1] - np.interp(isolated_data[0],
                                                          data[0],
                                                          absorption_mean),
                             corrected_sigma_small]

        corrected_all_flux.append([data[0], corrected_flux,
                                   corrected_sigma_full])

    return (fits, fits_eval_stamp, stamps, params, corr_stamps, EW_lines,
            EW_hb_samples, corrected_all_flux, EW_hb_der, EW_hb_der_err,
            EW_hb_stack, EW_hb_der_mc, EW_hb_weight_stack, line_significance)


# Model flux for lines in list
def model_flux_list(class_, stamp, linelist, sigmas,
                    ratio=None, plot=False):
    z_init = class_.redshift
    linelist_dict = class_.linelist_dict

    narrow_gaussians = [
        GaussianModel(prefix=f'{label}_narrow_') for label in linelist]
    broad_gaussians = [
        GaussianModel(prefix=f'{label}_broad_') for label in linelist]

    polydeg = 0
    polynomial = PolynomialModel(degree=polydeg)

    model = narrow_gaussians[0] + broad_gaussians[0]
    for n, b in zip(narrow_gaussians[1:], broad_gaussians[1:]):
        model += n + b
    model += polynomial

    pars = model.make_params()

    pars.add('z', value=z_init, vary=True,
             min=z_init - 1e-2, max=z_init + 1e-2)

    sigma_v_narrow, sigma_v_broad = sigmas

    if sigma_v_narrow.value is None:
        pars.add('sigma_v_narrow', value=50, min=20, max=65, vary=True)
    else:
        pars.add('sigma_v_narrow', value=sigma_v_narrow.value, vary=False)

    if sigma_v_broad.value is None:
        pars.add('sigma_v_broad', value=80, min=65, max=100, vary=True)
    else:
        pars.add('sigma_v_broad', value=sigma_v_broad.value, vary=False)

    if ratio is None:
        pars.add('broad_to_narrow_ratio', value=0.3, vary=True, min=0.0)
    else:
        pars.add('broad_to_narrow_ratio', value=ratio, vary=False)

    for label in linelist:
        lam0 = linelist_dict[label]

        pars.add(f'{label}_narrow_center', value=lam0, vary=False,
                 expr=f'{lam0}*(1+z)')
        pars.add(f'{label}_narrow_amplitude', value=50, vary=True, min=0.0)
        pars.add(f'{label}_narrow_sigma', vary=False,
                 expr=f'(sigma_v_narrow/3e5)*{label}_narrow_center')

        pars.add(f'{label}_broad_center', value=lam0, vary=False,
                 expr=f'{lam0}*(1+z)')
        pars.add(f'{label}_broad_amplitude', vary=False,
                 expr=f'broad_to_narrow_ratio*{label}_narrow_amplitude')
        pars.add(f'{label}_broad_sigma', vary=False,
                 expr=f'(sigma_v_broad/3e5)*{label}_broad_center')

    for i in range(polydeg + 1):
        pars[f'c{i}'].set(value=1.0, vary=True)

    out = model.fit(stamp[1], pars, x=stamp[0],
                    nan_policy='omit', max_nfev=1000)

    if plot:
        plotting_fit(stamp[0], stamp[1], out, linelist, class_)
    return out


# Model fluxes offr H alpha and H beta
def model_fluxes_ha_hb(class_, lines, data, sigmas, corrected_data,
                       save_data=None, num_mc=100):
    stamps_flux = get_stamps(class_, data, lines)
    results = {}
    for i, line in enumerate(lines):
        fits_or, fits_corr = [], []
        As_or, As_corr = [], []
        for j in range(num_mc):
            yoff_or = stamps_flux[i][1] + np.random.randn(len(stamps_flux[i][1])) * stamps_flux[i][2]
            yoff_corr = corrected_data[line][1] + np.random.randn(len(corrected_data[line][1])) * corrected_data[line][2]
            fit_or = model_flux_list(class_,
                                     (stamps_flux[i][0],
                                      yoff_or, stamps_flux[i][2]),
                                     [line], plot=False, sigmas=sigmas)
            fit_corr = model_flux_list(class_,
                                       (corrected_data[line][0],
                                        yoff_corr, corrected_data[line][2]),
                                       [line], plot=False, sigmas=sigmas)
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
        if save_data is not None:
            pd_dict = pd.DataFrame(results)
            pd_dict.to_csv(f'{save_data}/results_all_{class_.names[0][:-5]}.csv')
    return results


# Whole algorithm modeling all lines + plots + saving data
def all_lines_balmer_estimation(class_, data, sigmas, vel=550,
                                save_path=f'{proj_DIR}images_try'):
    lines = list(EW_all_ratios.keys())[2:-1]
    lines.append('H1_3970A')

    fit_estimation = model_from_list_lines(class_,
                                           data,
                                           sigmas,
                                           lines,
                                           vel=vel)
    (fits, fits_eval_stamp, stamps, params,
     corr_stamps, EW_lines,
     EW_hb_samples, corrected_all_flux,
     EW_hb_der, EW_hb_der_err,
     EW_hb_stack, EW_hb_der_mc) = fit_estimation

    fig1 = plot_all_fits(class_, lines, stamps, params,
                         save_path=save_path)
    fig2, fig3 = plot_ew_histograms(class_, EW_hb_samples,
                                    save_path=save_path)

    results = model_fluxes_ha_hb(class_, lines[:-1],
                                 data, sigmas, corr_stamps,
                                 save_data=save_path,
                                 num_mc=1000)

    fig4 = plot_pct_diff_histograms(class_, results, save_path=save_path)

    return results, fit_estimation, fig1, fig2, fig3, fig4


# ---------------------------------------------
# Plotting functions
# ---------------------------------------------

def plotting_fit(lams, flux, fit, linelist, class_):
    fig2, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(8, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
    )

    comps = fit.eval_components(x=lams)
    best_fit = comps['polynomial'].copy()
    for line in linelist:
        best_fit += (comps[line + '_narrow_'] +
                     comps[line + '_broad_'])

    # --- Main spectrum ---
    narrow_key = [key for key, _ in comps.items() if 'narrow' in key]
    broad_key = [key for key, _ in comps.items() if 'broad' in key]

    ax.plot(lams, flux, lw=1, drawstyle='steps-mid',
            label='Observed spectrum')

    ax.plot(lams, best_fit, lw=1, drawstyle='steps-mid',
            label='Best model')

    ax.plot(lams, comps['polynomial'], 'r-',
            label='Polynomial component',
            lw=1, drawstyle='steps-mid', alpha=0.3)

    for i, broad_line in enumerate(broad_key):
        ax.plot(lams, comps[broad_line] + comps['polynomial'],
                'k--',
                label='Broad component' if i == len(broad_key)-1 else None,
                alpha=0.3)

    for i, narrow_line in enumerate(narrow_key):
        ax.plot(lams, comps[narrow_line] + comps['polynomial'],
                'g--',
                label='Narrow component' if i == len(narrow_key)-1 else None,
                alpha=0.3)

    # --- Residuals ---
    residuals = (flux - best_fit)

    ax_res.plot(lams, residuals,
                color='black',
                lw=1,
                drawstyle='steps-mid')

    ax_res.axhline(0, color='grey', linestyle='--', lw=1)

    ax_res.set_ylabel('Residuals', size=16)
    ax_res.set_xlabel(r'Obs. Wavelength ($\AA$)', size=20)
    # ax_res.set_ylim(-40, 40)

    # --- Emission line markers ---
    for label in linelist:
        obs_lam = (class_.linelist_dict[label] *
                   (1 + fit.params['z'].value))

        ax.axvline(obs_lam, linestyle='--',
                   linewidth=0.5, color='grey')

        ax.text(obs_lam, 0.99, '\n'+label, rotation=90, ha='center', va='top',
                color='k', size=8, transform=ax.get_xaxis_transform())

    # --- Formatting ---
    str = r'Flux ($10^{-17}\ \mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$)'
    ax.set_ylabel(str, size=20)

    ax.legend(fontsize=14, frameon=True)
    ax_res.tick_params(
        axis='both',
        which='major',
        labelsize=16,
        width=2,
        length=6
    )
    ax.tick_params(
        axis='both',
        which='major',
        labelsize=16,
        width=2,
        length=6, labelbottom=False
    )
    plt.show()
    return fig2


def plot_pct_diff_histograms(class_, results, save_path=None):
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
        ax.set_xlabel('flux percentage difference (corrected vs original)')
        ax.text(0.05, 0.9, f"$\\sigma$ = {sig:.2f}", transform=ax.transAxes,
                fontsize=10, fontweight='bold')
        lo, hi = np.percentile(pct, [1, 99])
        ax.set_xlim(lo, hi)

    for k in range(len(lines), len(axs_flat)):
        axs_flat[k].axis('off')

    plt.tight_layout()
    if save_path is not None:
        path = f'{save_path}/percentage_{class_.names[0][:-5]}.png'
        plt.savefig(path, format='png')
    plt.show()
    return fig


def plot_all_fits(class_, balmer_lines, stamps, params,
                  n_show=None, save_path=None, ylim=(5, 30)):

    n_lines = len(balmer_lines) if n_show is None else min(n_show,
                                                           len(balmer_lines))

    fig, axs = plt.subplots(n_lines, 2, figsize=(10, 3.5 * n_lines),
                            squeeze=False)

    for i, line in enumerate(balmer_lines[:n_lines]):
        wave, flux, _ = stamps[line]
        As = np.array(params[line])
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
        path = f'{save_path}/all_fits_{class_.names[0][:-5]}.png'
        plt.savefig(path, format='png')
    plt.show()
    return fig


def plot_ew_histograms(class_, EW_hb_samples, lines=None,  save_path=None):
    if lines is None:
        lines = ['H1_4340A', 'H1_4102A', 'H1_3889A', 'H1_3835A',
                 'H1_3798A', 'H1_3771A']

    n_lines = len(lines)
    ncols = 3
    nrows = int(np.ceil(n_lines / ncols))

    fig, axs = plt.subplots(nrows, ncols, squeeze=False,
                            figsize=(4*ncols, 3*nrows))
    axs_flat = axs.flatten()

    for i, line in enumerate(lines):
        EW_hb = EW_hb_samples[i]
        ax = axs_flat[i]
        ax.hist(EW_hb, bins=30, alpha=0.8)
        ax.axvline(np.mean(EW_hb), color='red', linestyle='--',
                   label=f'mean={np.mean(EW_hb):.2f}')
        ax.set_title(line)
        ax.set_xlabel(r'EW ($H_\beta$)')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

    # hide unused subplots
    for k in range(n_lines, len(axs_flat)):
        axs_flat[k].axis('off')

    plt.tight_layout()
    if save_path is not None:
        path = f'{save_path}/ew_histograms_{class_.names[0][:-5]}.png'
        plt.savefig(path, format='png')
    plt.show()

    fig2, axs2 = plt.subplots(1, 2, figsize=(6, 4))
    all_EW_hb = np.concatenate(EW_hb_samples)
    axs2[0].hist(all_EW_hb, bins=30, alpha=0.8)
    axs2[0].axvline(np.mean(all_EW_hb), color='red', linestyle='--',
                    label=f'mean={np.mean(all_EW_hb):.2f}')
    axs2[0].set_title('All Balmer lines combined')
    axs2[0].set_xlabel(r'EW ($H_\beta$)')
    axs2[0].set_ylabel('Count')
    axs2[0].legend(fontsize=8)

    for i, line in enumerate(lines):
        axs2[1].hist(EW_hb_samples[i], bins=15, alpha=0.5,
                     label=line)
    axs2[1].set_title('Individual Balmer lines')
    axs2[1].set_xlabel(r'EW ($H_\beta$)')
    axs2[1].set_ylabel('Count')
    axs2[1].legend(fontsize=8)
    plt.tight_layout()
    if save_path is not None:
        path = f'{save_path}/ew_all_histograms_{class_.names[0][:-5]}.png'
        plt.savefig(path, format='png')
    plt.show()
    plt.show()

    return fig, fig2


def plot_corr_flux(data, corrected_all_flux, line,
                   xlim=(4000, 4500), save_path=None):
    wave, flux = data[0], data[1]
    print(corrected_all_flux)
    fig = plt.figure(figsize=(8, 5))
    plt.step(wave, flux, color='grey', label='Original flux', lw=1,
             drawstyle='steps-mid', alpha=0.5)
    plt.step(corrected_all_flux[0], corrected_all_flux[1],
             color='blue', label='Corrected flux', lw=1, drawstyle='steps-mid',
             alpha=0.5)
    plt.title(f'Flux comparison for {line}')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.legend()
    plt.xlim(xlim)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format='png')
    plt.show()
    return fig


# ----------------------------------------------
# Checking uncertainties
# ----------------------------------------------


# Model how this affects the Hb line
def model_hb_effect(class_, data, sigmas, EW_hb_stack, params, vel=550):
    line = 'H1_4861A'
    param = params[line]
    n_param, group_size = EW_hb_stack.shape

    lines_to_fit = ['H1_6563A', 'H1_4861A', 'O3_5007A',
                    'N2_6548A', 'N2_6583A']

    corrected_flux = np.zeros((n_param * group_size, len(data[1])))
    fits, amplitudes = [], []
    idx = 0
    for p in range(n_param):
        for j in range(group_size):
            absorption = neg_gauss_EW_single(class_, line, data[0],
                                             param[p], EW_hb_stack[p, j],
                                             vel=vel)
            corrected_flux[idx] = data[1] - absorption

            fit = model_flux_list(class_, (data[0], corrected_flux[idx],
                                           data[2]),
                                  lines_to_fit, sigmas)
            fits.append(fit)

            A_broad = fit.params[f'{line}_broad_amplitude'].value
            A_narrow = fit.params[f'{line}_narrow_amplitude'].value
            amplitude = A_broad + A_narrow
            amplitudes.append(amplitude)
            print(idx)
            idx += 1

    df = pd.DataFrame({'corr_flux': list(corrected_flux),
                       'fits': fits,
                       'amplitudes': amplitudes})
    df.to_csv(f'{proj_DIR}images_try/EW_fluxes_{class_.names[0][:-5]}.csv')
    return df


def dust_correction():
    names = ['J0023', 'J0136', 'J0020', 'J0203', 'J0243', 'J0333',
             'J0404', 'J2204', 'J2258', 'J2336', 'J0328']

    columns_ = ['ID', 'mass', 'z_red', 'z_blue']
    for line in lines['name']:
        columns_.append(line + '_flux')
        columns_.append(line + '_fluxerr')

    df = pd.DataFrame(columns=columns_)

    for name in names:
        data = pd.read_csv(DIR + f'lines/{name}/{name}_model_parts.csv')
        all_rows = {'ID': data['ID'][0], 'mass': data['mass'][0],
                    'z_red': data['z_red'][0], "z_blue": data['z_blue'][0]}
        for line in lines['name']:
            print(f'Correcting line {line}')
            wl = lines[lines['name'] == line]['vacuum_wave'].values
            E_BV, _ = E_BV_(name)
            flux, fluxerr = get_flux(name, line)
            f_corr = f_int(wl, flux, E_BV)
            f_corr_err = f_int(wl, fluxerr, E_BV)
            all_rows[line + '_flux'] = f_corr[0]
            all_rows[line + '_fluxerr'] = f_corr_err[0]

        df.loc[-1] = all_rows
        df.index = df.index + 1
    # df.to_csv(DIR + 'lines/magE2024_master_Dcorr_parts.csv')


# def peaks_estimation(class_, fit_estimation,
#                      save_path='/Users/javieratoro/Desktop/thesis/proyecto 2024-2/images_try'):
#     (fits, fits_eval_stamp, stamps, params,
#      corr_stamps, EW_lines,
#      EW_hb_samples, corrected_all_flux,
#      EW_hb_der, EW_hb_der_err,
#      EW_hb_stack, EW_hb_der_mc) = fit_estimation

#     lines_to_use = list(EW_all_ratios.keys())
#     lines_to_use.append('H1_3970A')
#     peak_locations, line_to_group, _ = find_ew_groups_kde(EW_hb_samples,
#                                                           lines_to_use)
#     group_stats = get_group_ew_stats(EW_hb_samples, lines_to_use,
#                                      line_to_group, peak_locations)

#     if save_path is not None:
#         pd_df = pd.DataFrame(group_stats)
#         pd_df.to_csv(f'{save_path}/groups_{class_.names[0][:-5]}.csv')

#     return group_stats
