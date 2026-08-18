from GALAXIES import PROJ_DIR
from lmfit.models import GaussianModel, PolynomialModel

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


# First read spectra
def read_data(class_):
    """
    Reads MW dust corrected data if it exists, otherwise reads the
    uncorrected data.
    """
    gal_id = class_.names[0][:5]
    print(f'Reading data for {gal_id}')
    file_path = f'{PROJ_DIR}dust/{gal_id}/dcorr_{gal_id}.csv'
    if os.path.exists(file_path):
        print("Using MW dust corrected data")
        dcorr = pd.read_csv(file_path)
        arrays = dcorr.to_numpy().T
        wave, flux, sigma = arrays

    else:
        print("Using uncorrected data")
        wave, flux, sigma, _ = class_.datas[0]
    return wave, flux, sigma


def get_stamps(red):
    wave, flux, err = red.wave, red.flux, red.sigma
    lines = red.spectra.line_list['name'].values
    lines_df = red.spectra.line_list
    bright_lines = ['H1_6563A', 'H1_4861A', 'O3_4959A', 'O3_5007A']
    cte = (1+red.spectra.redshift)
    stamps = []
    for bright_line in bright_lines:
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


def plotting_fit(lams, flux, fit, linelist, red):
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
        obs_lam = (red.spectra.linelist_dict[label] *
                   (1 + fit.params['z'].value))

        ax.axvline(obs_lam, linestyle='--',
                   linewidth=0.5, color='grey')

        ax.text(obs_lam, 0.99, '\n'+label, rotation=90, ha='center', va='top',
                color='k', size=8, transform=ax.get_xaxis_transform())

    # --- Formatting ---
    str = r'Flux ($10^{-17}\ \mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$'
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


def model_bright_lines(red, stamp, linelist,
                       sigma_v_narrow=None, sigma_v_broad=None,
                       ratio=None, plot=False):
    z_init = red.spectra.redshift
    linelist_dict = red.spectra.linelist_dict

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
             min=z_init - 1e-3, max=z_init + 1e-3)

    if sigma_v_narrow is None:
        pars.add('sigma_v_narrow', value=50, min=20, max=65, vary=True)
    else:
        pars.add('sigma_v_narrow', value=sigma_v_narrow, vary=False)

    if sigma_v_broad is None:
        pars.add('sigma_v_broad', value=80, min=65, max=500, vary=True)
    else:
        pars.add('sigma_v_broad', value=sigma_v_broad, vary=False)

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
        pars[f'c{i}'].set(value=0.0, vary=False)

    out = model.fit(stamp[1], pars, x=stamp[0],
                    nan_policy='omit', max_nfev=1000)

    if plot:
        plotting_fit(stamp[0], stamp[1], out, linelist, red)
    return out


def mc_sigma_and_ratio(red, stamp, linelist, n_mc=100, rng=None):

    wave, flux, err = stamp[0], stamp[1], stamp[2]
    rng = np.random.default_rng() if rng is None else rng

    sigma_narrow_samples = []
    sigma_broad_samples = []
    ratio_samples = []

    for i in range(n_mc):
        flux_mc = rng.normal(loc=flux, scale=err)
        stamp_mc = (wave, flux_mc, err)

        try:
            out = model_bright_lines(red, stamp_mc, linelist, plot=False)
        except Exception:
            continue

        sigma_narrow_samples.append(out.params['sigma_v_narrow'].value)
        sigma_broad_samples.append(out.params['sigma_v_broad'].value)
        ratio_samples.append(out.params['broad_to_narrow_ratio'].value)

    sigma_narrow_samples = np.array(sigma_narrow_samples)
    sigma_broad_samples = np.array(sigma_broad_samples)
    ratio_samples = np.array(ratio_samples)

    return {
        'sigma_narrow_samples': sigma_narrow_samples,
        'sigma_broad_samples': sigma_broad_samples,
        'ratio_samples': ratio_samples,
    }


def run_bright(red):
    bright_lines = ['H1_6563A', 'H1_4861A', 'O3_4959A', 'O3_5007A']
    stamps = get_stamps(red)
    samples = []
    for line, stamp in zip(bright_lines, stamps):
        if line == 'O3_5007A':
            sample = mc_sigma_and_ratio(red, stamp, [line, 'He1_5016A'])
        else:
            sample = mc_sigma_and_ratio(red, stamp, [line])
        samples.append(sample)
    return samples


def get_ratios(red):
    samples = run_bright(red)
    narrow_samples = [s['sigma_narrow_sampl'] for s in samples]
    broad_samples = [s['sigma_broad_samples'] for s in samples]
    ratios_samples = [s['ratio_samples'] for s in samples]

    sigma_narrow = np.median(np.concatenate(narrow_samples))
    sigma_broad = np.median(np.concatenate(broad_samples))
    ratio = np.median(np.concatenate(ratios_samples))

    return sigma_narrow, sigma_broad, ratio


def plot_sigma_histograms(samples, bins=40, figsize=(12, 4)):

    narrow_samples = np.concatenate([s['sigma_narrow_sampl'] for s in samples])
    broad_samples = np.concatenate([s['sigma_broad_samples'] for s in samples])
    ratio_samples = np.concatenate([s['ratio_samples'] for s in samples])

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    panels = [
        (narrow_samples, r'$\sigma_{narrow}$', axes[0]),
        (broad_samples,  r'$\sigma_{broad}$',  axes[1]),
        (ratio_samples,  r'$\sigma_{broad}/\sigma_{narrow}$', axes[2]),
    ]

    for data, label, ax in panels:
        median_val = np.median(data)

        ax.hist(data, bins=bins, color='steelblue', alpha=0.7, edgecolor='k')
        ax.axvline(median_val, color='crimson', linestyle='--', linewidth=2)

        ax.text(
            0.97, 0.95,
            f'median = {median_val:.3f}',
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=11,
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8}
        )

        ax.set_xlabel(label)
        ax.set_ylabel('Counts')

    plt.tight_layout()
    plt.show()
    return fig

# Usage:
# fig = plot_sigma_histograms(samples)


def mc_flux_errors(red, stamp, linelist, sigma_v_narrow, sigma_v_broad,
                   ratio, n_mc=100, rng=None):

    wave, flux, err = stamp[0], stamp[1], stamp[2]
    rng = np.random.default_rng() if rng is None else rng

    narrow_flux = {label: [] for label in linelist}
    broad_flux = {label: [] for label in linelist}

    for i in range(n_mc):
        flux_mc = rng.normal(loc=flux, scale=err)
        stamp_mc = (wave, flux_mc, err)

        try:
            out = model_bright_lines(
                red, stamp_mc, linelist,
                sigma_v_narrow=sigma_v_narrow,
                sigma_v_broad=sigma_v_broad,
                ratio=ratio,
                plot=False
            )
        except Exception:
            continue

        for label in linelist:
            narrow = f'{label}_narrow_amplitude'
            broad = f'{label}_broad_amplitude'
            narrow_flux[label].append(out.params[narrow].value)
            broad_flux[label].append(out.params[broad].value)

    results = {}
    for label in linelist:
        narrow_arr = np.array(narrow_flux[label])
        broad_arr = np.array(broad_flux[label])
        results[label] = {
            'narrow_flux': np.median(narrow_arr),
            'narrow_fluxerr': np.std(narrow_arr),
            'broad_flux': np.median(broad_arr),
            'broad_fluxerr': np.std(broad_arr),
            'flux': np.median(narrow_arr + broad_arr),
            'fluxerr': np.std(narrow_arr + broad_arr),
            'narrow_samples': narrow_arr,
            'broad_samples': broad_arr,
        }

    return results


def fit_lines_full(red, n_mc=100,
                   plot=False, rng=None):

    rng = np.random.default_rng() if rng is None else rng

    sigma_narrow, sigma_broad, ratio = get_ratios(red)

    final_fit = model_bright_lines(
        red, [red.wave, red.flux, red.sigma],
        red.spectra.line_name,
        sigma_v_narrow=sigma_narrow,
        sigma_v_broad=sigma_broad,
        ratio=ratio,
        plot=plot
    )

    fluxes = mc_flux_errors(red, (red.wave, red.flux, red.sigma),
                            red.spectra.line_name,
                            sigma_v_narrow=sigma_narrow,
                            sigma_v_broad=sigma_broad,
                            ratio=ratio,
                            n_mc=n_mc, rng=rng)

    return {
        'mc1': (sigma_narrow, sigma_broad, ratio),
        'final_fit': final_fit,
        'fluxes': fluxes,
    }
