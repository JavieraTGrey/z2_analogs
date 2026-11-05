"""
Stellar Balmer absorption correction (series method)
"""
import os

import astropy.constants as const
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GaussianFitting import fitSpectrum

proj_DIR = '/Users/javieratoro/Desktop/thesis/proyecto 2024-2/'

# EW(line) / EW(H_beta) of the stellar absorption
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

EXCLUDE_FROM_EW = {'H1_3750A', 'H1_3734A', 'H1_3722A', 'H1_3704A',
                   'H1_3970A'}


# ================================================================
# Absorption profiles
# ================================================================

GAMMA_PER_SIGMA = np.sqrt(2 * np.log(2))


def balmer_shape(w, mu, sigma, eta=0.0):
    gamma = GAMMA_PER_SIGMA * sigma
    gauss = np.exp(-0.5 * ((w - mu) / sigma)**2)
    lorentz = 1.0 / (1.0 + ((w - mu) / gamma)**2)
    return (1 - eta) * gauss + eta * lorentz


def balmer_area(sigma, eta=0.0):
    gamma = GAMMA_PER_SIGMA * sigma
    return (1 - eta) * np.sqrt(2 * np.pi) * sigma + eta * np.pi * gamma


def neg_gauss(class_, line, w, A, vel=550, eta=0.0):
    w_center = class_.linelist_dict[line]
    z = class_.redshift
    mu = w_center * (1 + z)
    sigma = w_center * (1 + z) * vel / 3e5
    return -A * balmer_shape(w, mu, sigma, eta)


def absorption_func(class_, line, w, A, m, n):
    return neg_gauss(class_, line, w, A) + m * w + n


# ================================================================
# Data preparation
# ================================================================

def read_data(class_):

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


def mask_nonuse_emission_line(class_, sigmas, line, data):

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


# ================================================================
# EW of the Balmer lines -> EW(H_beta)
# ================================================================

def get_EW_from_mc(class_, label, As, vel=550, eta=0.0):

    w_center = class_.linelist_dict[label]
    z = class_.redshift
    mu = w_center * (1 + z)
    sigma = w_center * (1 + z) * vel / 3e5

    As = np.array(As)
    height, m, n = As[:, 0], As[:, 1], As[:, 2]

    continuum_at_mu = m * mu + n
    EW = (height * balmer_area(sigma, eta)
          / continuum_at_mu
          / (1 + z))

    return EW


def line_chi2(class_, line, stamp, param, sigmas, vel=550, core_size=3.0):

    wave, flux, err = stamp
    median_params = np.median(np.asarray(param), axis=0)
    model = absorption_func(class_, line, wave, *median_params)

    center = class_.linelist_dict[line] * (1 + class_.redshift)
    sig_em = get_sigma(line, class_, sigmas)
    core = (wave > center - core_size * sig_em) & \
           (wave < center + core_size * sig_em)

    good = np.isfinite(flux) & np.isfinite(err) & (err > 0) & ~core
    n_free = 3                                  # height, m, n
    dof = np.sum(good) - n_free
    if dof <= 0:
        return np.nan
    chi2 = np.sum(((flux[good] - model[good]) / err[good])**2)
    return chi2 / dof


def ew_hbeta_draws(class_, params, vel=550, min_snr=3.0, lines=None,
                   stamps=None, sigmas=None, snr_mode='none',
                   at_bound_tol=1e-3, ew_draws=None):

    if lines is None:
        lines = []
        for line in EW_all_ratios:
            if line in ('H1_6563A', 'H1_4861A'):
                continue
            if line in EXCLUDE_FROM_EW:
                continue
            if ew_draws is not None and line not in ew_draws:
                continue
            lines.append(line)

    lines_clean = []
    for line in lines:
        if line in EXCLUDE_FROM_EW:
            print(f'  {line}: excluded (EXCLUDE_FROM_EW)')
        else:
            lines_clean.append(line)
    lines = lines_clean

    rows = []
    used, scaled, weights, medians, eff_var = [], [], [], [], []
    for line in lines:
        from_series = ew_draws is not None and line in ew_draws
        if from_series:
            ew = np.asarray(ew_draws[line])
        elif line in params:
            ew = get_EW_from_mc(class_, line, np.asarray(params[line]),
                                vel=vel)
        else:
            continue
        ratio = EW_all_ratios[line]
        med = np.median(ew)
        std = np.std(ew)
        snr = med / std if std > 0 else 0.0

        if (not from_series and stamps is not None and sigmas is not None
                and line in stamps):
            chi2_nu = line_chi2(class_, line, stamps[line], params[line],
                                sigmas, vel=vel)
        else:
            chi2_nu = np.nan
        if np.isfinite(chi2_nu) and chi2_nu > 0:
            w_chi2 = min(1.0, 1.0 / chi2_nu)
        else:
            w_chi2 = 1.0

        if snr_mode == 'none':
            w_snr = 1.0          # every line kept, also EW <= 0
        elif med <= 0 or snr <= 0:
            w_snr = 0.0
        elif snr_mode == 'hard':
            w_snr = 1.0 if snr >= min_snr else 0.0
        else:
            w_snr = min(1.0, snr / min_snr)

        frac_at_bound = np.mean(np.abs(ew) < at_bound_tol)
        at_bound = frac_at_bound > 0.5
        if at_bound:
            w_snr = 0.0

        widen = np.sqrt(1.0 / w_chi2)
        ew_line = med + (ew - med) * widen

        var = (std * widen / ratio)**2
        w = w_snr / var if var > 0 else 0.0

        rows.append({'line': line, 'EW': med, 'EW_std': std, 'SN': snr,
                     'EW_Hb_equiv': med / ratio, 'chi2_nu': chi2_nu,
                     'at_bound_%': 100 * frac_at_bound,
                     'w_snr': w_snr, 'w_chi2': w_chi2, 'weight': w})
        if w > 0:
            used.append(line)
            scaled.append(ew_line / ratio)
            weights.append(w)
            medians.append(med / ratio)
            eff_var.append(var / w_snr)

    table = pd.DataFrame(rows)
    if len(used) == 0:
        print(table.to_string(index=False))
        raise ValueError('No Balmer line has a usable absorption EW; '
                         'do not correct H_alpha/H_beta for this galaxy.')

    scaled = np.array(scaled)              # (n_lines, n_mc)
    weights = np.array(weights)
    medians = np.array(medians)

    shares = []
    for _, row in table.iterrows():
        shares.append(100 * row['weight'] / np.sum(weights))
    table['share_%'] = shares

    ew_hb = np.zeros(scaled.shape[1])
    for k in range(len(used)):
        ew_hb += weights[k] * scaled[k]
    ew_hb /= np.sum(weights)

    chi2_between = np.nan
    if len(used) > 1:
        mean_med = np.sum(weights * medians) / np.sum(weights)
        chi2_between = (np.sum((medians - mean_med)**2 / np.array(eff_var))
                        / (len(used) - 1))
        if chi2_between > 1:
            centre = np.median(ew_hb)
            ew_hb = centre + (ew_hb - centre) * np.sqrt(chi2_between)

    print(table.to_string(index=False, float_format='%.2f'))
    if np.isfinite(chi2_between) and chi2_between > 1:
        print(f'  lines disagree (chi2_nu = {chi2_between:.2f}): '
              f'errors widened by {np.sqrt(chi2_between):.2f}')
    print(f'  EW(Hb) = {np.median(ew_hb):.2f} +- {np.std(ew_hb):.2f} A '
          f'from {used}')
    if np.median(ew_hb) <= 0:
        print('  WARNING: EW(Hb) <= 0, no detectable absorption; the '
              'H_alpha/H_beta correction will be ~0 (or negative).')
    table.attrs['chi2_between'] = chi2_between
    return ew_hb, used, table


# ================================================================
# H_alpha, H_beta, H_gamma correction from EW(H_beta)
# ================================================================

def continuum_for_ew(wave, flux, err, mu, sig_abs, ew_rest, z, eta=0.0):

    g = (ew_rest * (1 + z) / balmer_area(sig_abs, eta)
         * balmer_shape(wave, mu, sig_abs, eta))
    x_m = wave - mu * g
    x_n = 1.0 - g

    good = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    design = np.column_stack([x_m[good], x_n[good]]) / err[good][:, None]
    target = flux[good] / err[good]
    coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return coeffs[0], coeffs[1]


def correct_ha_hb(class_, data, corrected_flux, sigmas, params, vel=550,
                  min_snr=3.0, core_size=4.0, lines_for_ew=None,
                  ratio_frac=0.0, seed=None, stamps=None, snr_mode='none',
                  ew_draws=None, ratio_lines=('H1_4861A', 'H1_6563A'),
                  ew_hb_override=None, ew_line_override=None, eta=0.0):

    print('Correcting H_beta and H_alpha from the EW of the other lines...')
    rng = np.random.default_rng(seed)
    ew_hb, used, ew_table = ew_hbeta_draws(
        class_, params, vel=vel, min_snr=min_snr, lines=lines_for_ew,
        stamps=stamps, sigmas=sigmas, snr_mode=snr_mode, ew_draws=ew_draws)
    n_mc = len(ew_hb)
    if ew_hb_override is not None:
        # EW(Hb) fixed by hand (e.g. from H_beta's own wings), as
        # (mean, std): draws from a normal distribution
        mean_o, std_o = ew_hb_override
        ew_hb = rng.normal(mean_o, std_o, n_mc)
        used = ['override']
        print(f'  EW(Hb) OVERRIDDEN: {mean_o:.2f} +- {std_o:.2f} A '
              f'(the series value above is kept only for reference)')
    z = class_.redshift
    wave = data[0]
    corrected_flux = corrected_flux.copy()
    abs_info = {}

    for line in ratio_lines:
        masked = mask_nonuse_emission_line(
            class_, sigmas, line, [wave, corrected_flux, data[2]])
        wave_s, flux_s, err_s = isolate_emission_line(class_, line,
                                                      datas=masked)
        mu = class_.linelist_dict[line] * (1 + z)
        sig_abs = mu * vel / 3e5
        sig_em = get_sigma(line, class_, sigmas)

        flux_s = flux_s.copy()
        core = (wave_s > mu - core_size * sig_em) & \
               (wave_s < mu + core_size * sig_em)
        flux_s[core] = np.nan

        ew_line = ew_hb * EW_all_ratios[line]
        if ratio_frac > 0:
            ew_line = ew_line * (1 + ratio_frac * rng.standard_normal(n_mc))
        if ew_line_override is not None and line in ew_line_override:
            mean_o, std_o = ew_line_override[line]
            ew_line = rng.normal(mean_o, std_o, n_mc)
            print(f'  {line}: EW set from its own wings: {mean_o:.2f} +- '
                  f'{std_o:.2f} A')
        param = np.zeros((n_mc, 3))
        for i in range(n_mc):
            yoff = flux_s + rng.standard_normal(len(flux_s)) * err_s
            m, n = continuum_for_ew(wave_s, yoff, err_s, mu, sig_abs,
                                    ew_line[i], z, eta)
            cont_mu = m * mu + n
            height = ew_line[i] * (1 + z) * cont_mu / \
                balmer_area(sig_abs, eta)
            param[i, 0] = height
            param[i, 1] = m
            param[i, 2] = n
        params[line] = param

        median_height = np.median(param[:, 0])
        corrected_flux -= neg_gauss(class_, line, wave, median_height,
                                    vel=vel, eta=eta)

        # integrated absorption flux (observed frame, flux units x A)
        f_abs = param[:, 0] * balmer_area(sig_abs, eta)
        abs_info[line] = {
            'EW_median': np.median(ew_line),
            'EW_std': np.std(ew_line),
            'height_median': median_height,
            'vel': vel,
            'eta': eta,
            'F_abs_median': np.median(f_abs),
            'F_abs_std': np.std(f_abs),
            'lines_used': ' '.join(used),
        }
        print(f'  {line}: EW = {np.median(ew_line):.2f} +- '
              f'{np.std(ew_line):.2f} A, F_abs = {np.median(f_abs):.3g} +- '
              f'{np.std(f_abs):.2g}')

    return corrected_flux, params, abs_info, ew_hb, ew_table


# ================================================================
# Saving
# ================================================================

def save_corrected_data(class_, data, corrected_flux, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/bcorr_{class_.gal_id}_new.csv'

    df = pd.DataFrame({'wave': data[0], 'flux': corrected_flux,
                       'sigma': data[2]})
    df.to_csv(save_path, index=False)

    print(f'Saved Balmer absorption corrected data to: {save_path}')
    return save_path


def save_abs_info(class_, abs_info, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/abs_hahb_{class_.gal_id}_new.csv'
    rows = []
    for line in abs_info:
        row = {'line': line}
        for key in abs_info[line]:
            row[key] = abs_info[line][key]
        rows.append(row)
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f'Saved H_alpha/H_beta absorption info to: {save_path}')
    return save_path


def save_ew_table(class_, ew_table, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/ew_weights_{class_.gal_id}_new.csv'
    ew_table.to_csv(save_path, index=False)
    print(f'Saved EW weights table to: {save_path}')
    return save_path


def save_mc_realizations(class_, params, save_dir, vel=550, extra_fabs=None,
                         eta=0.0):

    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/mc_realizations_{class_.gal_id}_new.h5'

    with h5py.File(save_path, 'w') as f:
        for line, param in params.items():
            grp = f.create_group(line)
            grp.create_dataset('height', data=param[:, 0], compression='gzip')
            grp.create_dataset('m', data=param[:, 1], compression='gzip')
            grp.create_dataset('n', data=param[:, 2], compression='gzip')
            EW_samples = get_EW_from_mc(class_, line, param, vel=vel, eta=eta)
            grp.create_dataset('EW', data=EW_samples, compression='gzip')
            mu = class_.linelist_dict[line] * (1 + class_.redshift)
            f_abs = param[:, 0] * balmer_area(mu * vel / 3e5, eta)
            grp.create_dataset('F_abs', data=f_abs, compression='gzip')
        if extra_fabs is not None:
            for line in extra_fabs:
                if line in f:
                    continue
                grp = f.create_group(line)
                grp.create_dataset('F_abs', data=extra_fabs[line],
                                   compression='gzip')

    print(f'Saved {len(params)} lines x '
          f'{next(iter(params.values())).shape[0]} '
          f'MC realizations to: {save_path}')
    return save_path


# ================================================================
# Plots
# ================================================================

def _save(fig, save_path, name, class_):
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        path = f'{save_path}/{name}_{class_.gal_id}.png'
        fig.savefig(path, format='png', dpi=150, bbox_inches='tight')
        print(f'Saved {path}')


def _bins(values, min_bins=8, max_bins=40):

    values = np.asarray(values)
    values = values[np.isfinite(values)]
    edges = np.histogram_bin_edges(values, bins='fd')
    n = len(edges) - 1
    if n < min_bins or n > max_bins:
        n = int(np.clip(n, min_bins, max_bins))
        edges = np.linspace(values.min(), values.max(), n + 1)
    return edges


def plot_ew_histograms(class_, ew_draws, used, ew_hb, ew_table,
                       save_path=None):

    lines = list(ew_table['line'])
    ew_hb_med = np.median(ew_hb)
    ew_hb_std = np.std(ew_hb)

    ncols = 3
    nrows = int(np.ceil(len(lines) / ncols))
    fig, axs = plt.subplots(nrows, ncols, squeeze=False,
                            figsize=(4 * ncols, 3 * nrows))
    axs_flat = axs.flatten()
    scaled = {}
    for i in range(len(lines)):
        line = lines[i]
        row = ew_table[ew_table['line'] == line].iloc[0]
        ew_eq = np.asarray(ew_draws[line]) / EW_all_ratios[line]
        scaled[line] = ew_eq
        ax = axs_flat[i]
        if line in used:
            color = 'tab:blue'
            label = f'used, share {row["share_%"]:.0f}%'
        else:
            color = 'tab:red'
            label = 'not used'
        ax.hist(ew_eq, bins=_bins(ew_eq), color=color, alpha=0.7,
                label=label)
        ax.axvline(np.median(ew_eq), color='k', ls='--', lw=1,
                   label=f'median = {np.median(ew_eq):.2f}')
        ax.axvspan(ew_hb_med - ew_hb_std, ew_hb_med + ew_hb_std,
                   color='green', alpha=0.15, label='adopted EW(Hb)')
        ax.set_title(line, fontsize=10)
        ax.set_xlabel(r'EW / ratio = EW(H$\beta$) equivalent [$\AA$]')
        ax.set_ylabel('Count')
        ax.legend(fontsize=7)
    for k in range(len(lines), len(axs_flat)):
        axs_flat[k].axis('off')
    plt.tight_layout()
    _save(fig, save_path, 'ew_histograms', class_)
    plt.show()

    fig2, axs2 = plt.subplots(1, 2, figsize=(11, 4))
    all_vals = []
    for line in lines:
        all_vals.append(scaled[line])
    common = _bins(np.concatenate(all_vals), max_bins=50)
    for line in lines:
        ls = '-' if line in used else ':'
        axs2[0].hist(scaled[line], bins=common, histtype='step', lw=1.5,
                     ls=ls, label=line)
    axs2[0].axvline(ew_hb_med, color='k', ls='--')
    axs2[0].set_title('Individual Balmer lines (dotted = not used)')
    axs2[0].set_xlabel(r'EW(H$\beta$) equivalent [$\AA$]')
    axs2[0].set_ylabel('Count')
    axs2[0].legend(fontsize=7)

    names = [('H1_4861A', r'H$\beta$', 'tab:blue'),
             ('H1_6563A', r'H$\alpha$', 'tab:red'),
             ('H1_4340A', r'H$\gamma$', 'tab:green')]
    adopted = []
    for line, tex, color in names:
        adopted.append(ew_hb * EW_all_ratios[line])
    common2 = _bins(np.concatenate(adopted))
    for line, tex, color in names:
        ew_line = ew_hb * EW_all_ratios[line]
        axs2[1].hist(ew_line, bins=common2, histtype='step', lw=1.8,
                     color=color,
                     label=rf'EW({tex}) = {np.median(ew_line):.2f} $\pm$ '
                           rf'{np.std(ew_line):.2f}')
    axs2[1].set_title(f'Adopted absorption (from {len(used)} lines)')
    axs2[1].set_xlabel(r'EW [$\AA$]')
    axs2[1].set_ylabel('Count')
    axs2[1].legend(fontsize=8)
    plt.tight_layout()
    _save(fig2, save_path, 'ew_all_histograms', class_)
    plt.show()
    return fig, fig2


def plot_hahb_correction(class_, data, flux_before, flux_after, params,
                         abs_info, save_path=None, half_width=60):
    z = class_.redshift
    wave = data[0]
    line_names = ['H1_4861A', 'H1_6563A']
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for k in range(2):
        line = line_names[k]
        mu = class_.linelist_dict[line] * (1 + z)
        m = (wave > mu - half_width) & (wave < mu + half_width)
        axs[k].step(wave[m], flux_before[m], where='mid', color='grey',
                    label='before')
        axs[k].step(wave[m], flux_after[m], where='mid', color='C0',
                    label='after')
        absorption = neg_gauss(class_, line, wave[m],
                               abs_info[line]['height_median'],
                               vel=abs_info[line].get('vel', 550),
                               eta=abs_info[line].get('eta', 0.0))
        cont = (np.median(params[line][:, 1]) * wave[m]
                + np.median(params[line][:, 2]))
        axs[k].plot(wave[m], cont, 'k:', lw=1, label='continuum')
        axs[k].plot(wave[m], cont + absorption, 'r-',
                    label='continuum + absorption')
        lo = np.nanpercentile(flux_before[m], 2)
        axs[k].set_ylim(min(0, lo), 3 * np.nanmedian(flux_before[m]))
        axs[k].set_title(f"{line}: EW = {abs_info[line]['EW_median']:.2f} "
                         f"$\\pm$ {abs_info[line]['EW_std']:.2f} $\\AA$")
        axs[k].set_xlabel(r'Obs. wavelength ($\AA$)')
        axs[k].set_ylabel('Flux')
        axs[k].legend(fontsize=8)
    plt.tight_layout()
    _save(fig, save_path, 'hahb_correction', class_)
    plt.show()
    return fig


# ================================================================
# How much do the H_beta and H_alpha fluxes change with the correction?
# ================================================================

def _fit_emission(class_, wave, flux, err, labels, window, sigmas, deg=1):

    z = class_.redshift
    sel = ((wave > window[0]) & (wave < window[1]) & np.isfinite(flux)
           & np.isfinite(err) & (err > 0))
    w, f, e = wave[sel], flux[sel], err[sel]
    x = (w - w.mean()) / (w.max() - w.min())
    cols = []
    for j in range(deg + 1):
        cols.append(x**j)
    for label in labels:
        mu = class_.linelist_dict[label] * (1 + z)
        for sv in (sigmas[0].value, sigmas[1].value):
            sig = mu * sv / const.c.to('km/s').value
            cols.append(np.exp(-0.5 * ((w - mu) / sig)**2)
                        / (np.sqrt(2 * np.pi) * sig))
    A = np.column_stack(cols)
    coef, _, _, _ = np.linalg.lstsq(A / e[:, None], f / e, rcond=None)
    fluxes = {}
    for k in range(len(labels)):
        fluxes[labels[k]] = coef[deg + 1 + 2 * k] + coef[deg + 2 + 2 * k]
    return fluxes, (w, f, A @ coef)


def _window(class_, line, default_half=50.0):
    z = class_.redshift
    try:
        info = class_.line_list[class_.line_list['name'] == line]
        w1, w6 = info[['w1', 'w6']].values[0]
        return w1 * (1 + z), w6 * (1 + z)
    except Exception:
        mu = class_.linelist_dict[line] * (1 + z)
        return mu - default_half, mu + default_half


def hahb_flux_change(class_, data, flux_before, params, abs_info, sigmas,
                     deg=1, save_path=None, plot=True):
    """
    Fits H_beta and H_alpha + [N II] (as FINAL_LINES does: Gaussians +
    a local polynomial continuum in the w1-w6 window) on the spectrum
    BEFORE the H_alpha/H_beta correction and on the spectrum with each
    absorption draw added back, and reports how much each flux changes:

      change_%  = 100 (F_after - F_before) / F_before  (median, std over
                  the absorption draws)
      upper_%   = 100 F_abs / F_before  (all the absorbed flux)
      leak      = (F_after - F_before) / F_abs  (fraction of the absorbed
                  flux that the line fit actually gets back)

    The pixel noise is the same before and after, so the spread of
    change_% is the uncertainty from the absorption correction only.
    """
    wave, _, err = data
    groups = {'H1_4861A': ['H1_4861A'],
              'H1_6563A': ['N2_6548A', 'H1_6563A', 'N2_6583A']}
    rows = []
    stamps = {}
    draws = {}          # per draw: F_after and % change of each line
    for line in groups:
        labels = []
        for lab in groups[line]:
            if lab in class_.linelist_dict:
                labels.append(lab)
        window = _window(class_, line)
        f0, st0 = _fit_emission(class_, wave, flux_before, err, labels,
                                window, sigmas, deg)
        F0 = f0[line]
        vel = abs_info[line].get('vel', 550)
        eta = abs_info[line].get('eta', 0.0)
        heights = np.asarray(params[line])[:, 0]
        changes, leaks = [], []
        for i in range(len(heights)):
            after = flux_before - neg_gauss(class_, line, wave, heights[i],
                                            vel=vel, eta=eta)
            fi, _ = _fit_emission(class_, wave, after, err, labels, window,
                                  sigmas, deg)
            changes.append(fi[line] - F0)
            mu = class_.linelist_dict[line] * (1 + class_.redshift)
            f_abs = heights[i] * balmer_area(mu * vel / 3e5, eta)
            leaks.append((fi[line] - F0) / f_abs if f_abs != 0 else np.nan)
        changes = np.array(changes)
        draws[f'F_before_{line}'] = np.full(len(changes), F0)
        draws[f'F_after_{line}'] = F0 + changes
        draws[f'change_%_{line}'] = 100 * changes / F0
        f_abs_med = abs_info[line]['F_abs_median']
        after_med = flux_before - neg_gauss(class_, line, wave,
                                            abs_info[line]['height_median'],
                                            vel=vel, eta=eta)
        _, st1 = _fit_emission(class_, wave, after_med, err, labels, window,
                               sigmas, deg)
        stamps[line] = (st0, st1, window)
        rows.append({'line': line, 'F_before': F0,
                     'F_after': F0 + np.median(changes),
                     'change_%': 100 * np.median(changes) / F0,
                     'change_err_%': 100 * np.std(changes) / F0,
                     'upper_%': 100 * f_abs_med / F0,
                     'leak': np.nanmedian(leaks)})
    table = pd.DataFrame(rows)
    hb = table[table['line'] == 'H1_4861A'].iloc[0]
    ha = table[table['line'] == 'H1_6563A'].iloc[0]
    r0 = ha['F_before'] / hb['F_before']
    r1 = ha['F_after'] / hb['F_after']
    print('\nFlux change from the H_alpha/H_beta absorption correction:')
    print(table.to_string(index=False, float_format='%.3f'))
    print(f'  Ha/Hb: {r0:.3f} -> {r1:.3f} ({100 * (r1 / r0 - 1):+.2f} %)')
    table.attrs['HaHb_before'] = r0
    table.attrs['HaHb_after'] = r1

    # per-draw table (draw i of Hb and Ha come from the same EW(Hb) draw,
    # so their ratio per draw is consistent)
    per_draw = pd.DataFrame(draws)
    per_draw.insert(0, 'draw', np.arange(len(per_draw)))
    per_draw['HaHb_before'] = r0
    per_draw['HaHb_after'] = (per_draw['F_after_H1_6563A']
                              / per_draw['F_after_H1_4861A'])
    per_draw['change_%_HaHb'] = 100 * (per_draw['HaHb_after'] / r0 - 1)
    table.attrs['per_draw'] = per_draw

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        names = ['H1_4861A', 'H1_6563A']
        for k in range(2):
            (w0, f0_, m0), (w1, f1_, m1), window = stamps[names[k]]
            row = table[table['line'] == names[k]].iloc[0]
            axs[k].step(w0, f0_, where='mid', color='grey', lw=0.8,
                        label='before')
            axs[k].step(w1, f1_, where='mid', color='C0', lw=0.8,
                        label='after')
            axs[k].plot(w0, m0, 'k--', lw=1, label='fit before')
            axs[k].plot(w1, m1, 'r-', lw=1, label='fit after')
            top = np.nanmedian(f0_)
            axs[k].set_ylim(0, 4 * top)
            axs[k].set_title(f"{names[k]}: flux {row['change_%']:+.2f} "
                             f"$\\pm$ {row['change_err_%']:.2f} %")
            axs[k].set_xlabel(r'Obs. wavelength ($\AA$)')
            axs[k].legend(fontsize=8)
        plt.tight_layout()
        _save(fig, save_path, 'hahb_flux_change', class_)
        plt.show()

        # histogram of the % change over all draws
        fig2, axs2 = plt.subplots(1, 3, figsize=(14, 3.8))
        cols = [('change_%_H1_4861A', r'H$\beta$'),
                ('change_%_H1_6563A', r'H$\alpha$'),
                ('change_%_HaHb', r'H$\alpha$/H$\beta$')]
        for k in range(3):
            col, tex = cols[k]
            v = per_draw[col].values
            axs2[k].hist(v, bins=_bins(v), color='C0', alpha=0.75)
            med = np.median(v)
            p16, p84 = np.percentile(v, [16, 84])
            axs2[k].axvline(med, color='k', ls='--')
            axs2[k].axvspan(p16, p84, color='grey', alpha=0.2)
            axs2[k].axvline(0, color='r', lw=0.8)
            axs2[k].set_title(f'{tex}: {med:+.2f} (+{p84 - med:.2f} / '
                              f'-{med - p16:.2f}) %')
            axs2[k].set_xlabel('change after correction [%]')
            axs2[k].set_ylabel('draws')
        plt.tight_layout()
        _save(fig2, save_path, 'hahb_flux_change_draws', class_)
        plt.show()
    return table


# ================================================================
# THE pipeline: one call per galaxy
# ================================================================

def balmer_dir(class_, base_dir=None):
    """Folder of one galaxy: {base_dir}/{gal}, default
    {proj_DIR}new_results/balmer_abs/{gal}."""
    if base_dir is None:
        base_dir = f'{proj_DIR}new_results/balmer_abs_final'
    return f'{base_dir}/{class_.gal_id}'


def run_balmer_series_pipeline(class_, n_mc=1000, base_dir=None, plot=True,
                               deg=4, seed=None, save=True, use_hgamma=True,
                               vel_metal=None, ew_hb_override=None,
                               vel_grid=None, rest_window=(3730., 4160.),
                               hgamma_own=True,
                               subtract_kinds=('balmer', 'metal', 'cn'),
                               min_snr=3.0, hgamma_in_ew=True,
                               ew_line_override=None, eta_grid=None):

    import balmer_series as bs

    save_dir = balmer_dir(class_, base_dir)
    plot_dir = None
    if save:
        plot_dir = f'{save_dir}/plots'
        os.makedirs(plot_dir, exist_ok=True)
        print(f'Saving in {save_dir} (plots in {plot_dir})')

    data = read_data(class_)
    wave, flux, err = data
    good = np.isfinite(err) & (err > 0)
    if not np.all(good):
        print(f'{np.sum(~good)} pixels with sigma <= 0 or NaN set to NaN')
        flux = np.where(good, flux, np.nan)
        err = np.where(good, err, np.nan)
        data = (wave, flux, err)
    sigmas = first_sigma_est(class_, (wave[good], flux[good], err[good]),
                             plot=False)

    # 1. series fit, hydrogen absorption added back
    if vel_metal is None:
        vel_metal = np.array([75., 100., 150., 200., 250., 300.])
    if vel_grid is None:
        # physical range for stellar Balmer absorption (Gaussian sigma)
        vel_grid = np.arange(300., 801., 50.)
    res = bs.fit_balmer_series(class_, data, sigmas, deg=deg, n_mc=n_mc,
                               vel_grid=vel_grid, vel_metal=vel_metal,
                               rest_window=rest_window, seed=seed,
                               eta_grid=eta_grid)
    v_best = res['vel_best']
    if v_best <= np.min(vel_grid) or v_best >= np.max(vel_grid):
        print(f'  WARNING: Balmer width at the edge of the grid ({v_best:.0f} '
              f'km/s): the width is not constrained by the data. Check the '
              f'series plot (continuum shape at the window edges); try a '
              f'lower deg or a narrower rest_window.')
    if (res['vel_metal'] <= np.min(vel_metal)
            or res['vel_metal'] >= np.max(vel_metal)):
        print(f"  note: metal width at the edge of its grid "
              f"({res['vel_metal']:.0f} km/s)")
    if plot:
        bs.plot_series(res, class_, save_path=plot_dir)
    corrected_flux = flux - bs.series_absorption(res, wave,
                                                 kinds=subtract_kinds,
                                                 min_snr=min_snr)
    vel = float(np.median(res['vel_draws']))
    eta = float(np.median(res['eta_draws']))
    if eta_grid is not None and len(np.atleast_1d(eta_grid)) > 1:
        if (eta <= np.min(eta_grid) and np.min(eta_grid) > 0) or \
                eta >= np.max(eta_grid):
            print(f'  note: eta at the edge of its grid ({eta:.2f})')

    ew_draws = {}
    for line in res['ew_draws']:
        ew_draws[line] = res['ew_draws'][line]
    res_hg = None
    if use_hgamma:
        print('\nH_gamma local fit (width fixed to the series value):')
        res_hg = bs.fit_hgamma(class_, data, sigmas, vel, n_mc=n_mc,
                               vel_metal=vel_metal, seed=seed, eta=eta)
        if plot:
            bs.plot_series(res_hg, class_, save_path=plot_dir,
                           name='hgamma_fit')
        if hgamma_in_ew:
            ew_draws['H1_4340A'] = res_hg['ew_draws']['H1_4340A']
        if not hgamma_own:
            hg_kinds = []
            for kind in subtract_kinds:
                if kind != 'balmer':
                    hg_kinds.append(kind)
            corrected_flux = corrected_flux - bs.series_absorption(
                res_hg, wave, kinds=tuple(hg_kinds), min_snr=min_snr)

    # 2-3. EW(Hb) from the series -> Hb, Ha (and Hg, unless it is
    # corrected with its own local fit)

    own_hg = use_hgamma and hgamma_own and res_hg is not None
    if own_hg:
        ratio_lines = ('H1_4861A', 'H1_6563A')
    else:
        ratio_lines = ('H1_4861A', 'H1_6563A', 'H1_4340A')
    flux_before = corrected_flux.copy()
    params = {}
    corrected_flux, params, abs_info, ew_hb, ew_table = correct_ha_hb(
        class_, data, corrected_flux, sigmas, params, vel=vel,
        ew_draws=ew_draws, seed=seed, ratio_lines=ratio_lines,
        ew_hb_override=ew_hb_override, ew_line_override=ew_line_override,
        eta=eta)

    if own_hg:
        # H_gamma corrected with the absorption of its OWN local fit (the
        # red dip at H_gamma in the hgamma_fit plot)
        corrected_flux = corrected_flux - bs.series_absorption(
            res_hg, wave, kinds=subtract_kinds, min_snr=min_snr)
        k_hg = res_hg['names'].index('H1_4340A')
        ew_hg = res_hg['ew_draws']['H1_4340A']
        mu = class_.linelist_dict['H1_4340A'] * (1 + class_.redshift)
        sig = mu * vel / 3e5
        f_abs_hg = res_hg['height_draws'][:, k_hg] * bs.balmer_area(sig, eta)
        abs_info['H1_4340A'] = {
            'EW_median': np.median(ew_hg), 'EW_std': np.std(ew_hg),
            'height_median': np.median(res_hg['height_draws'][:, k_hg]),
            'vel': vel, 'eta': eta, 'F_abs_median': np.median(f_abs_hg),
            'F_abs_std': np.std(f_abs_hg),
            'lines_used': 'own local fit'}
        print(f"  H1_4340A: corrected with its own fit, EW = "
              f"{np.median(ew_hg):.2f} +- {np.std(ew_hg):.2f} A")

    if plot:
        used = []
        for _, row in ew_table.iterrows():
            if row['weight'] > 0:
                used.append(row['line'])
        plot_ew_histograms(class_, ew_draws, used, ew_hb, ew_table,
                           save_path=plot_dir)
        plot_hahb_correction(class_, data, flux_before, corrected_flux,
                             params, abs_info, save_path=plot_dir)

    change = hahb_flux_change(class_, data, flux_before, params, abs_info,
                              sigmas, save_path=plot_dir, plot=plot)

    # absorbed flux per draw of the series hydrogen lines
    extra = {}
    for k in range(len(res['absorbers'])):
        ab = res['absorbers'][k]
        if ab['kind'] != 'balmer':
            continue
        f_abs = np.zeros(len(res['vel_draws']))
        for i in range(len(res['vel_draws'])):
            _, integral, _ = bs.absorber_profile(ab, res['wave'], res['z'],
                                                 res['vel_draws'][i],
                                                 res['vel_metal'],
                                                 eta=res['eta_draws'][i])
            f_abs[i] = res['height_draws'][i, k] * integral
        extra[ab['name']] = f_abs
    if own_hg:
        extra['H1_4340A'] = f_abs_hg

    print('\nSummary:')
    for line in abs_info:
        print(f"  {line}: EW_abs = {abs_info[line]['EW_median']:.2f} +- "
              f"{abs_info[line]['EW_std']:.2f} A")

    if save:
        save_corrected_data(class_, (wave, flux, err), corrected_flux,
                            save_dir)
        save_abs_info(class_, abs_info, save_dir)
        save_ew_table(class_, ew_table, save_dir)
        change.to_csv(f'{save_dir}/flux_change_{class_.gal_id}_new.csv',
                      index=False)
        change.attrs['per_draw'].to_csv(
            f'{save_dir}/flux_change_draws_{class_.gal_id}_new.csv',
            index=False)
        print(f'Saved flux_change_{class_.gal_id}_new.csv and '
              f'flux_change_draws_{class_.gal_id}_new.csv in {save_dir}')
        save_mc_realizations(class_, params, save_dir, vel=vel,
                             extra_fabs=extra, eta=eta)
    return corrected_flux, res, abs_info, ew_hb


# ================================================================
# After the line fitting: absorption uncertainty into the line fluxes
# ================================================================

def abs_flux_draws(class_, h5_path, vel=550):
    """Integrated absorption flux of every corrected line, per MC draw,
    from mc_realizations_{gal}_new.h5:  F_abs = height sqrt(2 pi) sigma."""
    z = class_.redshift
    draws = {}
    with h5py.File(h5_path, 'r') as f:
        for line in f.keys():
            if 'F_abs' in f[line]:
                draws[line] = f[line]['F_abs'][:]
                continue
            height = f[line]['height'][:]
            mu = class_.linelist_dict[line] * (1 + z)
            sig_abs = mu * vel / 3e5
            draws[line] = height * np.sqrt(2 * np.pi) * sig_abs
    return draws


def add_abs_uncertainty(info, f_abs_draws, seed=None, leak=1.0):

    rng = np.random.default_rng(seed)
    out = info.copy()

    n_draws = len(next(iter(f_abs_draws.values())))
    realizations = out['i'].unique()
    perm = rng.permutation(n_draws)
    pick = {}
    for k in range(len(realizations)):
        pick[realizations[k]] = perm[k % n_draws]

    medians = {}
    for line in f_abs_draws:
        medians[line] = np.median(f_abs_draws[line])

    new_nar, new_bro, shifts = [], [], []
    for _, row in out.iterrows():
        line = row['label']
        nar = row['nar_amplitude']
        bro = row['bro_amplitude']
        if line in f_abs_draws:
            j = pick[row['i']]
            if isinstance(leak, dict):
                frac = leak.get(line, 1.0)
            else:
                frac = leak
            shift = frac * (f_abs_draws[line][j] - medians[line])
        else:
            shift = 0.0
        total = nar + bro
        if total > 0:
            factor = 1 + shift / total
            nar = nar * factor
            bro = bro * factor
        else:
            nar = nar + shift
        new_nar.append(nar)
        new_bro.append(bro)
        shifts.append(shift)

    out['nar_amplitude'] = new_nar
    out['bro_amplitude'] = new_bro
    out['abs_shift'] = shifts
    return out


def abs_error_report(info_before, info_after, lines):

    rows = []
    for line in lines:
        b = info_before[info_before['label'] == line]
        a = info_after[info_after['label'] == line]
        if len(b) == 0:
            continue
        fb = (b['nar_amplitude'] + b['bro_amplitude']).values
        fa = (a['nar_amplitude'] + a['bro_amplitude']).values
        med = np.median(fb)
        p16b, p84b = np.percentile(fb, [16, 84])
        p16a, p84a = np.percentile(fa, [16, 84])
        rows.append({'line': line, 'flux': med,
                     'err_before_%': 50 * (p84b - p16b) / med,
                     'err_after_%': 50 * (p84a - p16a) / med})
    report = pd.DataFrame(rows)
    print(report.to_string(index=False, float_format='%.2f'))
    return report


def add_abs_and_save(class_, proj_DIR, bal_dir=None, tag_in='_mw',
                     tag_out='_mw_abs', vel=550, seed=None, leak=None,
                     use_flux_change=True):

    gal = class_.gal_id
    if bal_dir is None:
        bal_dir = balmer_dir(class_, f'{proj_DIR}new_results/balmer_abs_final/')
    folder = f'{proj_DIR}new_results/{gal}/'
    info = pd.read_csv(f'{folder}lines/{gal}_lines_info{tag_in}.csv')
    draws = abs_flux_draws(class_, f'{bal_dir}/mc_realizations_{gal}_new.h5',
                           vel=vel)

    leak_dict = {}
    if use_flux_change:
        fc = pd.read_csv(f'{bal_dir}/flux_change_draws_{gal}_new.csv')
        summary = pd.read_csv(f'{bal_dir}/flux_change_{gal}_new.csv')
        leak_hb = float(summary.loc[summary['line'] == 'H1_4861A',
                                    'leak'].values[0])
        if leak is None:
            leak = leak_hb
        for line in ('H1_4861A', 'H1_6563A'):
            draws[line] = (fc[f'F_after_{line}'].values
                           - fc[f'F_before_{line}'].values)
            leak_dict[line] = 1.0      # already the change of the fitted flux
        print(f'H_beta, H_alpha: measured flux change per draw; other '
              f'Balmer lines: absorbed flux x leak = {leak:.3f}')
    elif leak is None:
        leak = 1.0
    for line in draws:
        if line not in leak_dict:
            leak_dict[line] = leak

    out = add_abs_uncertainty(info, draws, seed=seed, leak=leak_dict)
    abs_error_report(info, out, list(draws.keys()))
    os.makedirs(f'{folder}/lines', exist_ok=True)
    out.to_csv(f'{folder}lines/{gal}_lines_info{tag_out}.csv', index=False)
    print(f'Saved {gal}_lines_info{tag_out}.csv')
    return out
