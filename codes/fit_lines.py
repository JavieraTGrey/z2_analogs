# %matplotlib widget
from lines import REDUC_LINES
from corr import REDSHIFT, SPECTRALDATA
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, PolynomialModel
import pandas as pd
import os

import seaborn as sns



main_DIR = "/Users/javieratoro/Desktop/thesis/"
proj_DIR = f"{main_DIR}proyecto 2024-2/"
code_DIR = f"{proj_DIR}codes"
os.chdir(code_DIR)

def fitting_lines(lams, flux, linelist, z_init=0,
                  params=None, nfev=1000, fit=None,
                  A_ratio=None, plot=False):
    lams = np.asarray(lams)
    flux = np.asarray(flux)

    narrow_gaussians = []
    broad_gaussians  = []

    for label in linelist.keys():
        narrow_gaussians.append(GaussianModel(prefix=f'{label}_narrow_'))
        broad_gaussians.append(GaussianModel(prefix=f'{label}_broad_'))

    polydeg = 7
    polynomial = PolynomialModel(degree=polydeg)

    model = narrow_gaussians[0] + broad_gaussians[0]
    for n,b in zip(narrow_gaussians[1:], broad_gaussians[1:]):
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

                pars.add(f'{label}_broad_amplitude', value=0.3, vary=True, min=0.0)

            if A_ratio is not None:
                ratio_median = np.median(A_ratio)
                pars.add('ratio', value=ratio_median, vary=False)
                pars.add(f'{label}_broad_amplitude', vary=False,
                            expr=f'ratio*{label}_narrow_amplitude*(sigma_v_broad/sigma_v_narrow)')

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
    ratios = ratios[(ratios>9e-2) & (ratios<1)]
    if plot:
        plt.figure()
        sns.boxplot(data=ratios, width=0.3)
        sns.stripplot(data=ratios, color='black', size=8, jitter=True, alpha=0.7)
        plt.title("Narrow to broad Amplitude Ratios")
        plt.ylabel("Value")
        plt.show()
    return ratios

def MC_fluxes(class_, wave, flux):
    linelist = class_.spectra.linelist_dict
    bright_lines = ['H_14', 'H_13', 'O2_3725', 'O2_3727', 'H_alpha', 'H_beta',
                    'O3_5008', 'O3_4959', 'N2_6550', 'N2_6585',
                    'S2_6716','S2_6730']

    lines = class_.spectra.linelist_dict
    bright_dict = {k: v for k, v in linelist.items() if k in bright_lines}
    faint_dict  = {k: v for k, v in linelist.items() if k not in bright_lines}

    lines = [bright_dict, faint_dict]

    def fit(wave, flux, linelist):
        print('bright')
        bright = fitting_lines(wave, flux,
                    linelist[0],
                    z_init=class_.spectra.redshift,
                    nfev=1000, fit=None, A_ratio=None)
        print('bright done')
        ratios = a_ratio_dist(bright)
        print('faint')
        faint = fitting_lines(wave, flux,
                    linelist[1],
                    z_init=class_.spectra.redshift,
                    nfev=1000, fit=bright, A_ratio=ratios)
        print('faint done')
        return bright, faint

    bright, faint = fit(wave, flux, lines)

    fit_ = bright, faint
    return fit_, lines


def save_info(class_, num):
    columns = ['iter', 'n_eval_red', 'n_eval_blue' , 'success_red', 'success_blue',
               'message', 'ier', 'z',
               'sigma_v_narrow_red', 'sigma_v_broad_red', 'sigma_v_narrow_bright',
               'sigma_v_broad_bright']

    for label in class_.spectra.linelist_dict.keys():
        columns.append(str(label) + '_narrow')
        columns.append(str(label) + '_broad')

    df = pd.DataFrame(columns=columns)
    print('Starting MC iteration')
    i = 0
    n_success = 0

    while n_success < num:
        i += 1

        # ---- Safe noise generation
        noise = np.random.randn(len(class_.flux)) * class_.sigma
        flux_mc = class_.flux + noise

        result = MC_fluxes(class_, class_.wave, flux_mc)

        if result is None:
            print(f"Skipping MC iteration {i}")
            continue

        fits, labels = result
        bright, faint = fits

        row = {'iter': i + 1,
                'z': float(bright.params['z'].value),
                'sigma_v_narrow_bright': float(bright.params['sigma_v_narrow'].value),
                'sigma_v_broad_bright': float(bright.params['sigma_v_broad'].value),
                'sigma_v_narrow_faint': float(faint.params['sigma_v_narrow'].value),
                'sigma_v_broad_faint': float(faint.params['sigma_v_broad'].value),
                'n_eval_faint': faint.nfev,
                'n_eval_bright': bright.nfev,
                'success_faint': faint.success,
                'success_bright': bright.success,
                'message': f"Blue: {bright.lmdif_message} | Red: {faint.lmdif_message}",
                'ier': [bright.ier, faint.ier]}
        for label, fit in zip(labels, fits):
            for label in label.keys():
                row[f"{label}_narrow"] = fit.params[f'{label}_narrow_amplitude'].value
                row[f"{label}_broad"] = fit.params[f'{label}_broad_amplitude'].value
    
        df.loc[len(df)] = row
        n_success += 1

    save_dir = f'{proj_DIR}lines/{class_.spectra.gal_id}/'
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f'{save_dir}{class_.spectra.gal_id}_iter_parts.csv', index = False)
    return df

def save_data(dict, num):

    spec = SPECTRALDATA(dict)
    _ = REDSHIFT(spec)
    class_ = REDUC_LINES(spec)

    info = save_info(class_, num)
    df = pd.DataFrame(columns=['ID', 'mass', 'z', 'name', 'flux',
                               'fluxerr',
                               'narrow_flux', 'narrow_fluxerr',
                               'broad_flux', 'broad_fluxerr'])

    for label in class_.spectra.linelist_dict.keys():
        narrow_ = np.asarray(info[label+"_narrow"])
        broad_ = np.asarray(info[label+"_broad"])
        flux = narrow_ + broad_

        row = {'ID': class_.spectra.names[0],
            'mass': class_.spectra.mass,
            'z': np.median(info['z']),
            'name': label,
            'flux': np.median(flux),
            'fluxerr': np.std(flux),
            'narrow_flux': np.median(narrow_),
            'narrow_fluxerr': np.std(narrow_),
            'broad_flux': np.median(broad_),
            'broad_fluxerr': np.std(broad_),
        }
        df.loc[len(df)] = row

    save_dir = f'{proj_DIR}lines/{class_.spectra.gal_id}/'
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f'{save_dir}{class_.spectra.gal_id}_model_parts.csv',
                index=False)
    print(f'Saving in {save_dir}')
    return df

# =============================================================================
#
# Program
#
# =============================================================================

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
    # save_data(J0328_SPEC, 100)
    # save_data(J0020_SPECTRA, 100)
    # save_data(J0203_SPECTRA, 100)
    # save_data(J0243_SPECTRA, 100)
    # save_data(J0333_SPECTRA, 100)
    # save_data(J0404_SPECTRA, 100)
    # save_data(J2204_SPECTRA, 100)
    # save_data(J2258_SPECTRA, 100)
    # save_data(J2336_SPECTRA, 100)