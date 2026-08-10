import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, PolynomialModel
from lmfit.parameter import Parameter
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
                pars.add(f'{label}_broad_amplitude', vary=False, value=0.0,
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
    ratios = ratios[(ratios > 9e-2) & (ratios < 1)]
    if len(ratios) < 5:
        ratios = np.array([0, 0])
        plot = False
    if plot:
        plt.figure()
        sns.boxplot(data=ratios, width=0.3)
        sns.stripplot(data=ratios, color='black', size=8,
                      jitter=True, alpha=0.7)
        plt.title("Narrow to broad Amplitude Ratios")
        plt.ylabel("Value")
        plt.show()
    return ratios


def valid_for_fit(wave, flux, min_points=10):
    """Check whether a spectrum chunk is fit-able."""
    if wave is None or flux is None:
        return False
    if len(wave) < min_points:
        return False
    if not np.any(np.isfinite(flux)):
        return False
    if np.nanstd(flux) == 0:
        return False
    return True


def MC_fluxes(class_, wave, flux):
    linelist = class_.spectra.linelist_dict
    bright_lines = ['H_14', 'H_13', 'O2_3725', 'O2_3727', 'H_alpha', 'H_beta',
                    'O3_5008', 'O3_4959', 'N2_6550', 'N2_6585',
                    'S2_6716', 'S2_6730']

    mask = (class_.wave < 4500*(1+class_.spectra.redshift))

    wave1, flux1 = wave[mask], flux[mask]
    wave2, flux2 = wave[~mask], flux[~mask]

    if not valid_for_fit(wave1, flux1):
        return None
    if not valid_for_fit(wave2, flux2):
        return None

    lines = class_.spectra.linelist_dict
    bright_dict = {k: v for k, v in linelist.items() if k in bright_lines}
    faint_dict = {k: v for k, v in linelist.items() if k not in bright_lines}

    blue_lines_b = {k: v for k, v in bright_dict.items() if v < 4500}
    red_lines_b = {k: v for k, v in bright_dict.items() if v >= 4500}

    blue_lines_f = {k: v for k, v in faint_dict.items() if v < 4500}
    red_lines_f = {k: v for k, v in faint_dict.items() if v >= 4500}
    lines = blue_lines_b, blue_lines_f, red_lines_b, red_lines_f

    def fit(wave, flux, linelist):
        bright = fitting_lines(wave, flux,
                               linelist[0],
                               z_init=class_.spectra.redshift,
                               nfev=1000, fit=None, A_ratio=None)
        ratios = a_ratio_dist(bright)
        faint = fitting_lines(wave, flux,
                              linelist[1],
                              z_init=class_.spectra.redshift,
                              nfev=1000, fit=bright, A_ratio=ratios)
        return bright, faint

    try:
        bright, faint = fit(wave1, flux1, (blue_lines_b, blue_lines_f))
        bright2, faint2 = fit(wave2, flux2, (red_lines_b, red_lines_f))
    except Exception:
        return None

    fit_ = bright, faint, bright2, faint2
    return fit_, lines


def save_info(class_, num):
    columns = ['iter', 'n_eval_red', 'n_eval_blue', 'success_red',
               'success_blue',
               'message', 'ier', 'z_blue', 'z_red',
               'sigma_v_narrow_red', 'sigma_v_broad_red',
               'sigma_v_narrow_bright',
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
            continue

        fits, labels = result
        blue_bright, _, red_bright, _ = fits

        row = {'iter': i + 1,
               'z_blue': float(blue_bright.params['z'].value),
               'z_red': float(red_bright.params['z'].value),
               'sigma_v_narrow_bright': float(blue_bright.params['sigma_v_narrow'].value),
               'sigma_v_broad_bright': float(blue_bright.params['sigma_v_broad'].value),
               'sigma_v_narrow_red': float(red_bright.params['sigma_v_narrow'].value),
               'sigma_v_broad_red': float(red_bright.params['sigma_v_broad'].value),
               'n_eval_red': red_bright.nfev,
               'n_eval_blue': blue_bright.nfev,
               'success_red': red_bright.success,
               'success_blue': blue_bright.success,
               'message': f"Blue: {blue_bright.lmdif_message} | Red: {red_bright.lmdif_message}",
               'ier': [blue_bright.ier, red_bright.ier]}
        for label, fit in zip(labels, fits):
            for label in label.keys():
                row[f"{label}_narrow"] = fit.params[f'{label}_narrow_amplitude'].value
                row[f"{label}_broad"] = fit.params[f'{label}_broad_amplitude'].value

        df.loc[len(df)] = row
        n_success += 1

    save_dir = f'{proj_DIR}lines/{class_.spectra.gal_id}/'
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f'{save_dir}{class_.spectra.gal_id}_iter_parts.csv',
              index=False)
    return df


def save_data(class_, num):
    info = save_info(class_, num)
    df = pd.DataFrame(columns=['ID', 'mass', 'z_red', 'z_blue', 'name', 'flux',
                               'fluxerr',
                               'narrow_flux', 'narrow_fluxerr',
                               'broad_flux', 'broad_fluxerr'])

    for label in class_.spectra.linelist_dict.keys():
        narrow_ = np.asarray(info[label+"_narrow"])
        broad_ = np.asarray(info[label+"_broad"])
        flux = narrow_ + broad_

        row = {'ID': class_.spectra.names[0],
               'mass': class_.spectra.mass,
               'z_red': np.median(info['z_red']),
               'z_blue': np.median(info['z_blue']),
               'name': label,
               'flux': np.median(flux),
               'fluxerr': np.std(flux),
               'narrow_flux': np.median(narrow_),
               'narrow_fluxerr': np.std(narrow_),
               'broad_flux': np.median(broad_),
               'broad_fluxerr': np.std(broad_)
               }
        df.loc[len(df)] = row

    save_dir = f'{proj_DIR}lines/{class_.spectra.gal_id}/'
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f'{save_dir}{class_.spectra.gal_id}_model_parts.csv',
              index=False)
    print(f'Saving in {save_dir}')
    return df


def fit_auroral_new(class_, ratio, flux_, showplot=False):

    auroral_lines = ['S2_4068', 'O3_4363',
                     'N2_5756', 'O1_6300',
                     'O1_6363', 'S3_6312',
                     'O2_7320']

    # get info from lines
    stamps, sigmas, fits = [], [], []

    wave, flux = class_.wave.copy(), flux_.copy()

    params = pd.read_csv(f'{proj_DIR}lines/{class_.spectra.gal_id}/'
                         f'{class_.spectra.gal_id}_iter_parts.csv')

    cte = (1 + class_.spectra.redshift)

    # dataframe setup
    columns = ['sigma_v_narrow', 'sigma_v_broad']
    for label in auroral_lines:
        columns += [f'{label}_narrow_amplitude',
                    f'{label}_broad_amplitude']
    columns += ['O2_7331_narrow_amplitude', 'O2_7331_broad_amplitude',
                'S2_4076_narrow_amplitude', 'S2_4076_broad_amplitude']

    df = pd.DataFrame(columns=columns)

    row = {col: np.nan for col in df.columns}

    # get stamps
    for auroral in auroral_lines:
        if auroral == 'O2_7320':
            exclude_labels = ['O2_7320', 'O2_7331']
        elif auroral == 'S2_4068':
            exclude_labels = ['S2_4068', 'S2_4076']
        else:
            exclude_labels = [auroral]

        wave_copy, flux_copy = wave.copy(), flux.copy()

        lam = class_.spectra.linelist_dict[auroral]
        center = lam * cte
        # get sigma for each line according to the line wavelength
        if lam > 4500:
            sigma_v_broad = np.median(params['sigma_v_broad_red'])
            sigma_v_narrow = np.median(params['sigma_v_narrow_red'])
        else:
            sigma_v_broad = np.median(params['sigma_v_broad_bright'])
            sigma_v_narrow = np.median(params['sigma_v_narrow_bright'])
        sigmas.append([sigma_v_narrow, sigma_v_broad])

        center = class_.spectra.linelist_dict[auroral] * (1+class_.spectra.redshift)
        sep = 35
        mask = (wave > center - sep) & (wave < center + sep)
        wave_line, flux_line = wave_copy[mask], flux_copy[mask]

        for l_ in class_.spectra.linelist_dict.keys():
            if exclude_labels and l_ in exclude_labels:
                continue
            c_ = class_.spectra.linelist_dict[l_] * (1+class_.spectra.redshift)
            if center - sep < c_ < center + sep:
                mask2 = (wave_line > c_ - 0.1*sigma_v_narrow) & (wave_line < c_ + 0.1*sigma_v_narrow)
                flux_line[mask2] = np.nan

        flux_line[flux_line < -2] = np.nan

        stamps.append((wave_line, flux_line))

        # Create model for line
        na_gss = [GaussianModel(prefix=f"{lab}_narrow_") for lab in exclude_labels]
        br_gss = [GaussianModel(prefix=f"{lab}_broad_") for lab in exclude_labels]

        if len(exclude_labels) == 1:
            sum_of_gaussians = na_gss[0] + br_gss[0]
        else:
            sum_of_gaussians = br_gss[0] + na_gss[0]
            for narrow, broad_value in zip(na_gss[1:], br_gss[1:]):
                sum_of_gaussians += (broad_value + narrow)

        polynomial = PolynomialModel(degree=1)
        comp_mult = sum_of_gaussians + polynomial

        pars_mult = comp_mult.make_params()

        for i in range(2):
            pars_mult[f'c{i}'].set(value=0, vary=False)

        pars_mult.add(name='z', value=class_.spectra.redshift, vary=True)
        pars_mult.add(name='sigma_v_narrow', value=sigma_v_narrow, vary=False)
        pars_mult.add(name='sigma_v_broad', value=sigma_v_broad, vary=False)

        for lab in exclude_labels:
            lam = class_.spectra.linelist_dict[lab]
            # narrow
            pars_mult[f"{lab}_narrow_center"] = Parameter(
                name=f"{lab}_narrow_center",
                value=lam, vary=True,
                expr=f'{lam:6.2f}*(1+z)'
            )
            pars_mult[f"{lab}_narrow_amplitude"] = Parameter(
                name=f"{lab}_narrow_amplitude",
                value=1, vary=True, expr=None#, min=0.0
            )
            pars_mult[f"{lab}_narrow_sigma"] = Parameter(
                name=f"{lab}_narrow_sigma",
                value=lam, vary=False,
                expr=f'(sigma_v_narrow/3e5)*{lab}_narrow_center'
            )
            # broad
            pars_mult[f"{lab}_broad_center"] = Parameter(
                name=f"{lab}_broad_center",
                value=lam, vary=True,
                expr=f'{lam:6.2f}*(1+z)'
            )

            pars_mult['A_ratio'] = Parameter(
                name='A_ratio',
                value=np.median(ratio),
                vary=False,
                expr=f'{np.median(ratio)}*(sigma_v_broad/sigma_v_narrow)'
            )

            pars_mult[f'{lab}_broad_amplitude'] = Parameter(
                name=f'{lab}_broad_amplitude', vary=False,
                expr=f'A_ratio*{lab}_narrow_amplitude'
            )

            pars_mult[f"{lab}_broad_sigma"] = Parameter(
                name=f"{lab}_broad_sigma",
                value=lam, vary=False,
                expr=f'(sigma_v_broad/3e5)*{lab}_broad_center'
            )

        # Fitting line
        fit = comp_mult.fit(flux_line, pars_mult, x=wave_line,
                            nan_policy='omit', max_nfev=1000)

        fits.append(fit)

        for label in exclude_labels:
            row['sigma_v_narrow'] = sigma_v_narrow
            row['sigma_v_broad'] = sigma_v_broad
            name_narrow = f"{label}_narrow_amplitude"
            name_broad = f"{label}_broad_amplitude"
            row[name_narrow] = float(fit.params[name_narrow].value)
            row[name_broad] = float(fit.params[name_broad].value)

    df.loc[len(df)] = row

    if showplot:
        fig, axes = plt.subplots(3, 3, figsize=(13, 11))
        for stamp, fit, ax, label in zip(stamps, fits,
                                         axes.flatten(), auroral_lines):
            wave_line, flux_line = stamp
            comp = fit.eval_components(x=wave_line)
            best = fit.eval(x=wave_line)
            ax.plot(wave_line, flux_line, drawstyle='steps-mid', color='blue',
                    label='Observed data', alpha=0.5)

            ax.plot(wave_line, best, 'black', lw=1,
                    label='Fit spectra', drawstyle='steps-mid')
            ax.plot(wave_line,
                    comp[f'{label}_narrow_'] + comp['polynomial'],
                    'green', linestyle='--', lw=1, alpha=0.5,
                    drawstyle='steps-mid', label='Narrow component')
            ax.plot(wave_line,
                    comp[f'{label}_broad_'] + comp['polynomial'],
                    'red', linestyle='--', lw=1, alpha=0.5,
                    drawstyle='steps-mid', label='Broad component')
            ax.set_xlabel(r'Obs. Wavelength ($\AA$)', size=14)
            ax.set_ylabel(r'Flux ($10^{-17} erg/s/cm^{2}/\AA$)', size=14)
            ax.set_title(f'{label}')
            if label == 'O2_7320':
                ax.plot(wave_line,
                        comp['O2_7331_narrow_'] + comp['polynomial'],
                        'green', linestyle='--', lw=1, alpha=0.5,
                        drawstyle='steps-mid', label='Narrow component')
                ax.plot(wave_line,
                        comp['O2_7331_broad_'] + comp['polynomial'],
                        'red', linestyle='--', lw=1, alpha=0.5,
                        drawstyle='steps-mid', label='Broad component')
                ax.set_title(f'{label}, O2_7331')
            if label == 'S2_4068':
                ax.plot(wave_line,
                        comp['S2_4076_narrow_'] + comp['polynomial'],
                        'green', linestyle='--', lw=1, alpha=0.5,
                        drawstyle='steps-mid', label='Narrow component')
                ax.plot(wave_line,
                        comp['S2_4076_broad_'] + comp['polynomial'],
                        'red', linestyle='--', lw=1, alpha=0.5,
                        drawstyle='steps-mid', label='Broad component')
                ax.set_title(f'{label}, S2_4076')
        axes[0, 2].legend(bbox_to_anchor=(1, 1.6), loc='upper right')
        axes[2, 2].axis('off')
        axes[2, 1].axis('off')
        plt.tight_layout()
        plt.show()
    return fits, df


def save_data_auroral_MC(class_, ratio, num):
    auroral_lines = ['S2_4068', 'O3_4363',
                     'N2_5756', 'O1_6300',
                     'O1_6363', 'S3_6312',
                     'O2_7320']
    comps = []

    # dataframe setup
    columns = ['sigma_v_narrow', 'sigma_v_broad']
    for label in auroral_lines:
        columns += [f'{label}_narrow_amplitude',
                    f'{label}_broad_amplitude']
    columns += ['O2_7331_narrow_amplitude', 'O2_7331_broad_amplitude',
                'S2_4076_narrow_amplitude', 'S2_4076_broad_amplitude']

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
