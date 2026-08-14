from GALAXIES import PROJ_DIR
# from GaussianFitting import fitSpectrum
from lmfit.models import GaussianModel, PolynomialModel
# from lmfit.parameter import Parameter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns


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


def fitting_lines(lams, flux, linelist, z_init=0,
                  params=None, nfev=1000, fit=None,
                  A_ratio=None, plot=False):
    lams = np.asarray(lams)
    flux = np.asarray(flux)

    narrow_gaussians = []
    broad_gaussians = []

    for label in linelist.keys():
        narrow_gaussians.append(GaussianModel(prefix=f'{label}_narrow_'))
        broad_gaussians.append(GaussianModel(prefix=f'{label}_broad_'))

    polydeg = 7
    polynomial = PolynomialModel(degree=polydeg)

    model = narrow_gaussians[0] + broad_gaussians[0]
    for n, b in zip(narrow_gaussians[1:], broad_gaussians[1:]):
        model += n + b

    model += polynomial

    pars = model.make_params()

    if params is None:
        pars.add('z', value=z_init, vary=True,
                 min=z_init-1e-4, max=z_init+1e-4)

        for label in linelist:
            lam0 = linelist[label]

            if fit is not None:
                sigma_v_nar = fit.params['sigma_v_narrow'].value
                sigma_v_bro = fit.params['sigma_v_broad'].value
                pars.add('sigma_v_narrow', value=sigma_v_nar, vary=False)
                pars.add('sigma_v_broad', value=sigma_v_bro, vary=False)

            else:
                pars.add('sigma_v_narrow', value=50, min=20, max=70)
                pars.add('sigma_v_broad', value=90, min=70, max=500)

                pars.add(f'{label}_broad_amplitude', value=0.3,
                         vary=True, min=0.0)

            if A_ratio is not None:
                ratio_median = np.median(A_ratio)
                pars.add('ratio', value=ratio_median, vary=False)
                pars.add(f'{label}_broad_amplitude', vary=False,
                         expr=(f'ratio*{label}_narrow_amplitude*'
                               '(sigma_v_broad/sigma_v_narrow)'))

            pars.add(f'{label}_narrow_center',
                     value=lam0, vary=False,
                     expr=f'{lam0}*(1+z)')

            pars.add(f'{label}_narrow_amplitude',
                     value=50, vary=True, min=0.0)

            pars.add(f'{label}_narrow_sigma',
                     vary=False,
                     expr=f'(sigma_v_narrow/3e5)*{label}_narrow_center',
                     min=0)

            pars.add(f'{label}_broad_center',
                     value=lam0, vary=False,
                     expr=f'{lam0}*(1+z)')

            pars.add(f'{label}_broad_sigma',
                     vary=False,
                     expr=f'(sigma_v_broad/3e5)*{label}_broad_center',
                     min=0)

        for i in range(polydeg+1):
            pars[f'c{i}'].set(value=0.0, vary=False)

    else:
        pars = params

    out = model.fit(flux, pars, x=lams,
                    nan_policy='omit',
                    max_nfev=nfev)
    if plot:
        plotting_fit(lams, flux, out, linelist)
    return out


def plotting_fit(lams, flux, out, linelist):
    fig2, ax = plt.subplots(figsize=(10, 4))
    comps = out.eval_components(x=lams)

    ax.plot(lams, flux, lw=1, drawstyle='steps-mid', label='Observed spectrum')
    ax.plot(lams, out.best_fit, lw=1, drawstyle='steps-mid',
            label='Best model')
    ax.plot(lams, comps['polynomial'], 'r-', label='Polynomial component',
            lw=1, drawstyle='steps-mid', alpha=0.3)

    broad_key = [key for key, _ in comps.items() if 'broad' in key]
    narrow_key = [key for key, _ in comps.items() if 'narrow' in key]

    for i, broad_line in enumerate(broad_key):
        if i == len(broad_key) - 1:
            ax.plot(lams, comps[broad_line] + comps['polynomial'], 'k--',
                    label='Broad component', alpha=0.3)
        else:
            ax.plot(lams, comps[broad_line] + comps['polynomial'], 'k--',
                    alpha=0.3)

    for i, narrow_line in enumerate(narrow_key):
        if i == len(narrow_key) - 1:
            ax.plot(lams, comps[narrow_line] + comps['polynomial'], 'g--',
                    label='narrow component', alpha=0.3)
        else:
            ax.plot(lams, comps[narrow_line] + comps['polynomial'], 'g--',
                    alpha=0.3)

    for label in linelist.keys():
        z = out.params['z'].value
        obs_lam = linelist[label] * (1+z)
        ax.axvline(obs_lam, linestyle='--', linewidth=1, color='grey', lw=0.5)
        ax.text(obs_lam, 0.99, '\n'+label, rotation=90, ha='center', va='top',
                color='k', size=8, transform=ax.get_xaxis_transform())

    ax.set_xlabel(r'Obs. Wavelength ($\AA$)', size=14)
    ax.set_ylabel(r'Flux ($10^{-17} erg/s/cm^{2}/\AA$)', size=14)
    ax.set_xlim([np.min(lams), np.max(lams)])
    ax.legend()
    fig2.tight_layout()
    plt.show()


def a_ratio_dist(out, plot=False):
    ratios = []
    for label in out.params.keys():
        if 'narrow_height' in label:
            broad_label = label.replace('narrow', 'broad')
            ratio = out.params[broad_label].value / out.params[label].value
            ratios.append(ratio)
    ratios = np.array(ratios)
    ratios = ratios[(ratios > 9e-2) & (ratios < 1)]
    if plot:
        plt.figure()
        sns.boxplot(data=ratios, width=0.3)
        sns.stripplot(data=ratios, color='black', size=8, jitter=True,
                      alpha=0.7)
        plt.title("Narrow to broad Amplitude Ratios")
        plt.ylabel("Value")
        plt.show()
    return ratios
