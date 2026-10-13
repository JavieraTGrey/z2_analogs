import astropy.constants as const
from GaussianFitting import fitSpectrum
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, PolynomialModel
import pandas as pd
import os

main_DIR = "/Users/javieratoro/Desktop/thesis/"
proj_DIR = f"{main_DIR}proyecto 2024-2/"
code_DIR = f"{proj_DIR}codes"
os.chdir(code_DIR)


# -----------------------------------------
# Data Preparation
# -----------------------------------------


# Read data for galaxies
def read_data(class_):
    """
    Reads Balmer absorption corrected data if it exists, otherwise reads the
    MW dust corrected data, and finally if none exists reads the
    uncorrected data.

    Eventually change to dust corrected data.

    """
    gal_id = class_.names[0][:5]
    print(f'Reading data for {gal_id}')
    file_path_balmer = f'{proj_DIR}new_results/bcorr_{gal_id}_new.csv'
    file_path_MW = f'{proj_DIR}dust/{gal_id}/dcorr_{gal_id}.csv'

    # Balmer + MW corrected data
    if os.path.exists(file_path_balmer):
        print("Using Balmer Absorption + MW dust corrected data")
        bcorr = pd.read_csv(file_path_balmer)
        arrays = bcorr.to_numpy().T
        wave, flux, sigma = arrays
    # MW dust corrected data
    elif os.path.exists(file_path_MW):
        print("Using MW Dust corrected data")
        dcorr = pd.read_csv(file_path_MW)
        arrays = dcorr.to_numpy().T
        wave, flux, sigma = arrays

    # Uncorrected data
    else:
        print("Using uncorrected data")
        wave, flux, sigma, _ = class_.datas[0]
    return wave, flux, sigma


# Read sigma from old measurement for masking
def read_sigma(class_):
    gal_id = class_.names[0][:5]
    path_old = f'{proj_DIR}lines/{gal_id}/{gal_id}_iter_.csv'
    if os.path.exists(path_old):
        old_fit = pd.read_csv(path_old)
        sigma_narrow = np.mean(old_fit['sigma_v_narrow'])
        sigma_broad = np.mean(old_fit['sigma_v_broad'])
        return sigma_narrow, sigma_broad
    else:
        raise ('Path error: File not found')


# Mask emission lines other than specific line
def mask_nonuse_emission_line(class_, lines, data):
    """
    Mask every emission line other than `lines`, 2.5 sigma around its
    center, setting flux to NaN and sigma to 0 in that window.

    """
    wave, flux, err = data

    cte = (1 + class_.redshift)
    masked_flux = flux.copy()
    masked_sigma = err.copy()
    sigmas = read_sigma(class_)

    bright_lines = ['O2_3726A', 'O2_3729A', 'H1_6563A', 'H1_4861A', 'H1_4340A',
                    'O3_5007A', 'O3_4959A', 'N2_6548A', 'N2_6583A', 'S2_6716A',
                    'S2_6731A']

    for label in class_.linelist_dict:
        if label in lines:
            continue
        center = class_.linelist_dict[label] * cte

        if label in bright_lines:
            sigma = (center / const.c.to('km/s').value) * sigmas[1]
        else:
            sigma = (center / const.c.to('km/s').value) * sigmas[0]

        size = 2.5
        mask_line = (wave > center - size * sigma) & \
                    (wave < center + size * sigma)

        masked_flux[mask_line] = np.nan
        masked_sigma[mask_line] = 0.0

    return wave, masked_flux, masked_sigma


# Create stamp for each line we are measuring
def get_stamp(class_, data, lines):
    wave, flux, err = data[0], data[1], data[2]
    lines_df = class_.line_list
    cte = (1 + class_.redshift)
    stamps = {}
    for line in lines:
        stamp = flux.copy()

        info_line = lines_df[lines_df['name'] == line]
        w2, w5 = info_line[['w2', 'w5']].values[0]
        mask_stamp = (wave > w2*cte) & (wave < w5*cte)

        stamps[line] = (wave[mask_stamp], stamp[mask_stamp], err[mask_stamp])
    return stamps


# -----------------------------------------
# Model function
# -----------------------------------------


# Model for a list of lines (could be just one)
def model_flux_list(class_, stamp, linelist, deg=1,
                    ratio=None, plot=False):
    z_init = class_.redshift
    linelist_dict = class_.linelist_dict

    narrow_gaussians = [
        GaussianModel(prefix=f'{label}_narrow_') for label in linelist]
    broad_gaussians = [
        GaussianModel(prefix=f'{label}_broad_') for label in linelist]

    polydeg = deg
    polynomial = PolynomialModel(degree=polydeg)

    model = narrow_gaussians[0] + broad_gaussians[0]
    for n, b in zip(narrow_gaussians[1:], broad_gaussians[1:]):
        model += n + b
    model += polynomial

    pars = model.make_params()

    pars.add('sigma_v_narrow', value=50, min=20, max=65, vary=True)

    pars.add('sigma_v_broad', value=80, min=65, max=120, vary=True)

    if ratio is None:
        pars.add('broad_to_narrow_ratio', value=0.3, vary=True, min=0.0)
    else:
        pars.add('broad_to_narrow_ratio', value=ratio, vary=False)

    for label in linelist:

        pars.add(f'z_{label}', value=z_init, vary=True,
                 min=0, max=0.1)

        lam0 = linelist_dict[label]

        pars.add(f'{label}_narrow_center', value=lam0, vary=False,
                 expr=f'{lam0}*(1+z_{label})')
        pars.add(f'{label}_narrow_amplitude', value=50, vary=True, min=0.0)
        pars.add(f'{label}_narrow_sigma', vary=False,
                 expr=f'(sigma_v_narrow/3e5)*{label}_narrow_center')

        pars.add(f'{label}_broad_center', value=lam0, vary=False,
                 expr=f'{lam0}*(1+z_{label})')
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


# MC model running on a list of lines (can be individual)
def MC_model_from_lines(class_, stamps, linelist, n_mc=100,
                        deg=1, ratio=None, plot=False):

    wave, flux, err = stamps[linelist[0]]

    fits = []
    for i in range(n_mc):
        yoff = flux + np.random.randn(len(flux)) * err
        fit = model_flux_list(class_, (wave, yoff, err), linelist,
                              deg=deg, ratio=ratio, plot=plot)
        fits.append(fit)

    return fits


# Create dict with MC fits of every line in list
def create_dict(class_, stamps, linelist_all, n_mc=100,
                deg=1, ratio=None, plot=False):
    dict_ = {}
    for linelist in linelist_all:
        linelist_ = np.asarray(linelist)
        if linelist_.size == 1:
            linelist = [linelist]

        fits = MC_model_from_lines(class_, stamps, linelist, n_mc=n_mc,
                                   deg=deg, ratio=ratio, plot=plot)
        dict_[linelist[0]] = fits
    return dict_


# -----------------------------------------
# Save information
# -----------------------------------------


def save_data_dict(class_, dict_, linelist):

    linelist_ = np.asarray(linelist)
    if linelist_.size == 1:
        linelist = [linelist]

    # dataframe setup-
    columns = ['i', 'sigma_v_narrow', 'sigma_v_broad',
               'broad_to_narrow_ratio']
    for label in linelist:
        columns += [f'{label}_narrow_amplitude',
                    f'{label}_broad_amplitude']


    info = pd.DataFrame(columns=columns)

    for i in range(num):
        yoff = np.random.randn(len(class_.flux)) * class_.sigma
        fit, df = fit_auroral_new(class_, ratio, flux_=class_.flux + yoff)
        info = pd.concat([info, df], ignore_index=True)
        comps.append(fit)

    res = pd.DataFrame(columns=['ID', 'mass', 'z', 'name', 'flux',
                                'fluxerr',
                                'narrow_flux', 'narrow_fluxerr',
                                'broad_flux', 'broad_fluxerr'])

    for label in info.columns[2:]:
        if 'broad' in label:
            continue
        else:
            label = label[:7]
            narrow_ = np.asarray(info[label+"_narrow_amplitude"])
            broad_ = np.asarray(info[label+"_broad_amplitude"])
            flux = narrow_ + broad_
            row = {'ID': class_.spectra.names[0],
                   'mass': class_.spectra.mass,
                   'z': class_.spectra.redshift,
                   'name': label,
                   'flux': np.median(flux),
                   'fluxerr': np.std(flux),
                   'narrow_flux': np.median(info[label+"_narrow_amplitude"]),
                   'narrow_fluxerr': np.std(info[label+"_narrow_amplitude"]),
                   'broad_flux': np.median(info[label+"_broad_amplitude"]),
                   'broad_fluxerr': np.std(info[label+"_broad_amplitude"])}
            res.loc[len(res)] = row

    res.to_csv(f'{proj_DIR}lines/{class_.spectra.gal_id}/'
               f'{class_.spectra.gal_id}_auroral_model_parts.csv',
               index=False)
    info.to_csv(f'{proj_DIR}lines/{class_.spectra.gal_id}/'
                f'{class_.spectra.gal_id}_auroral_iter_parts.csv',
                index=False)

    return info, res


# -----------------------------------------
# Running algorithm
# -----------------------------------------


# -----------------------------------------
# Plotting functions
# -----------------------------------------

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
                   (1 + fit.params[f'z_{label}'].value))

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
