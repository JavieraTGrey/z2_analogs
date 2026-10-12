import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from balmer_abs_EW import mask_nonuse_emission_line, isolate_emission_line
from balmer_abs_EW import model_flux_list

DIR = '/Users/javieratoro/Desktop/thesis/proyecto 2024-2/'

# ================================================================
# Lists for the whole pipeline
# ================================================================

DIAGNOSTIC_LINES = {
    'T_e_OIII': ['O3_4363A', 'O3_4959A', 'O3_5007A'],
    'T_e_NII': ['N2_5755A', 'N2_6548A', 'N2_6583A'],
    'n_e_SII': ['S2_6716A', 'S2_6731A'],
    'n_e_OII': ['O2_3726A', 'O2_3729A', 'O2_7320A', 'O2_7330A'],
    'balmer': ['H1_6563A', 'H1_4861A'],
}

BALMER_DECREMENT = ['H1_6563A', 'H1_4861A', 'H1_4340A', 'H1_4102A']

LINE_KEY_MAP = {
    'H1_6563A': 'H_alpha', 'H1_4861A': 'H_beta',
    'H1_4340A': 'H_gamma', 'H1_4102A': 'H_delta',
}

# Case B ratios
THEORETICAL_RATIOS = {
    'H1_6563A': 2.86,   # H_alpha / H_beta
    'H1_4861A': 1.0,    # H_beta / H_beta
    'H1_4340A': 0.464,  # H_gamma / H_beta
    'H1_4102A': 0.256,  # H_delta / H_beta
}

# ================================================================
# General Functions
# ================================================================


def k_cal(wl, Rv=3.1):
    """
    Calzetti extinction curve with Rv = 3.1, safe for any wavelength
    array (does not assume the input is sorted ascending).

    """
    wl = np.asarray(wl, dtype=float)
    wl_um = wl / 1e4  # convert to um
    k = np.empty_like(wl_um)

    mask_low = wl < 6300
    mask_high = ~mask_low

    k[mask_low] = (2.659 * (-2.156 + (1.509 / wl_um[mask_low])
                            - (0.198 / wl_um[mask_low]**2)
                            + (0.011 / wl_um[mask_low]**3)) + Rv)
    k[mask_high] = 2.659 * (-1.857 + 1.040 / wl_um[mask_high]) + Rv

    return k


def deredden_spectrum(wave, flux, E_BV, flux_err=None):
    """
    Apply the Calzetti dust attenuation correction to a FULL spectrum
    (wavelength + flux arrays), rather than a single line flux.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    k = k_cal(wave)
    atten_factor = 10 ** (0.4 * E_BV * k)

    corrected_flux = flux * atten_factor

    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=float)
        corrected_err = flux_err * atten_factor
        return corrected_flux, corrected_err

    return corrected_flux


def save_corrected_spectrum(name, wave, corrected_flux, corrected_err,
                            save_dir=None):
    """
    Save the dust-corrected spectrum to a CSV.
    """
    if save_dir is None:
        save_dir = f'{DIR}dust_corr'
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/dustcorr_{name}_new.csv'
    df = pd.DataFrame({'wave': wave, 'flux': corrected_flux,
                       'sigma': corrected_err})
    df.to_csv(save_path, index=False)

    print(f'Saved dust corrected spectrum to: {save_path}')
    return save_path


# ================================================================
# Get E(B-V)
# ================================================================


def estimate_E_BV_mc(class_,  balmer_results):
    """
    y_balmer = slope * x_balmer
    log10(R_obs/R_int) = -0.4*E(B-V) * (k(lambda) - k(hbeta))
    """
    lines = BALMER_DECREMENT
    wl = {line: class_.linelist_dict[line] for line in lines}

    hb_wl = wl['H1_4861A']

    wl_array = np.array([wl[line] for line in lines])
    hb_k = k_cal(np.array([hb_wl]))[0]

    K_cal_subs = k_cal(wl_array) - hb_k

    x_balmer = np.array(K_cal_subs)

    total_flux = np.array([
        balmer_results[line]['narrow'] + balmer_results[line]['broad']
        for line in lines
    ])

    hb_idx = lines.index('H1_4861A')
    ratios = total_flux / total_flux[hb_idx]

    theo = np.array([THEORETICAL_RATIOS[line] for line in lines])[:, None]
    y_balmer = np.log10(ratios / theo)

    mask = np.arange(len(lines)) != hb_idx
    x = x_balmer[mask]
    y = y_balmer[mask, :]

    # Linear regression through (0,0)
    slope = -np.sum(x[:, None] * y, axis=0) / np.sum(x**2)
    E_BV_samples = slope / 0.4

    return np.median(E_BV_samples), np.std(E_BV_samples), E_BV_samples


# ================================================================
# Read data + get fluxes
# ================================================================


def load_corrected_spectrum(gal_id, save_dir):
    path = f'{save_dir}/bcorr_{gal_id}_new.csv'
    df = pd.read_csv(path)
    return df['wave'].values, df['flux'].values, df['sigma'].values


def measure_line_flux_samples(class_, wave, flux, sigma, line, sigmas,
                              n_mc=100):

    masked_data = mask_nonuse_emission_line(class_, sigmas, line,
                                            [wave, flux, sigma])
    isolated = isolate_emission_line(class_, line, datas=masked_data)

    narrow_samples, broad_samples = [], []
    for _ in range(n_mc):
        yoff = isolated[1] #+ np.random.randn(len(isolated[1])) * isolated[2]
        fit = model_flux_list(class_, (isolated[0], yoff, isolated[2]),
                              [line], sigmas, plot=False)
        narrow_samples.append(fit.params[f'{line}_narrow_amplitude'].value)
        broad_samples.append(fit.params[f'{line}_broad_amplitude'].value)

    return np.array(narrow_samples), np.array(broad_samples)


def measure_lines(class_, wave, flux, sigma, lines, sigmas, n_mc=100):
    results = {}
    for line in lines:
        print(f'Measuring {line}...')
        narrow, broad = measure_line_flux_samples(
            class_, wave, flux, sigma, line, sigmas, n_mc=n_mc)
        results[line] = {'narrow': narrow, 'broad': broad}
    return results


# ================================================================
# Correct the spectra
# ================================================================


def correct_and_save_spectrum(gal_id, wave, flux, sigma, E_BV, save_dir):

    corrected_flux, corrected_err = deredden_spectrum(
        wave, flux, E_BV, flux_err=sigma)

    save_path = save_corrected_spectrum(gal_id, wave, corrected_flux,
                                        corrected_err, save_dir=save_dir)

    return corrected_flux, corrected_err, save_path


def measure_all_diagnostics(class_, wave, corrected_flux, corrected_err,
                            sigmas, n_mc=100, line_groups=None):

    if line_groups is None:
        line_groups = DIAGNOSTIC_LINES

    all_results = {}
    for group, lines in line_groups.items():
        print(f'--- Measuring group: {group} ---')
        all_results[group] = measure_lines(
            class_, wave, corrected_flux, corrected_err, lines, sigmas,
            n_mc=n_mc)
    return all_results


# ----------------------------------------------------------------
# Diagnostic plots
# ----------------------------------------------------------------

def plot_dust_correction(wave, flux, corrected_flux, gal_id, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(wave, flux, label='Before correction', alpha=0.6,
            color='black')
    ax.step(wave, corrected_flux, label='After correction',
            color='red')
    ax.set_xlabel(r'Wavelength (\AA)')
    ax.set_ylabel(r'Flux (10^{-17} ergs/s/cm^{2}/\AA)')
    ax.set_title(f'{gal_id}: dust correction')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png')
    plt.show()
    return fig


def plot_E_BV_distribution(gal_id, E_BV_samples, save_path=None):
    median = np.median(E_BV_samples)
    std = np.std(E_BV_samples)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(E_BV_samples, bins=40, color='purple', alpha=0.8)
    ax.axvline(median, color='black', linestyle='--',
               label=f'median = {median:.3f} ± {std:.3f}')
    ax.set_xlabel('E(B-V)')
    ax.set_ylabel('Count')
    ax.set_title(f'{gal_id}: E(B-V) Monte Carlo distribution')
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png')
    plt.show()
    return fig


def plot_balmer_decrement_check(gal_id, Ha_total_samples, Hb_total_samples,
                                save_path=None):
    ratio_samples = Ha_total_samples / Hb_total_samples
    median_ratio = np.median(ratio_samples)
    std_ratio = np.std(ratio_samples)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(['Observed\n(corrected)', 'Theoretical'],
           [median_ratio, 2.86], color=['blue', 'gray'],
           yerr=[std_ratio, 0], capsize=5)
    ax.set_ylabel(r'H$\alpha$ / H$\beta$')
    ax.set_title(f'{gal_id}: Balmer decrement check\n'
                 f'{median_ratio:.2f} ± {std_ratio:.2f} vs 2.86')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png')
    plt.show()
    return fig


def plot_flux_comparison(line, flux_before_samples, flux_after_samples,
                         save_path=None):
    med_before = np.median(flux_before_samples)
    med_after = np.median(flux_after_samples)
    std_before = np.std(flux_before_samples)
    std_after = np.std(flux_after_samples)

    pct_diff = (med_after - med_before) / med_before * 100
    combined_err = np.sqrt(std_before**2 + std_after**2)
    significance = (med_after - med_before) / combined_err

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(flux_before_samples, bins=30, alpha=0.6, color='black',
            label=f'Before (med={med_before:.2f})')
    ax.hist(flux_after_samples, bins=30, alpha=0.6, color='tab:red',
            label=f'After (med={med_after:.2f})')
    ax.set_title(f'{line}: {pct_diff:+.1f}% diff, {significance:.2f}σ')
    ax.set_xlabel(r'Flux (10^{-17} ergs/s/cm^{2}/\AA)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png')
    plt.show()

    return {'line': line, 'pct_diff': pct_diff, 'significance': significance,
            'median_before': med_before, 'median_after': med_after}


# ----------------------------------------------------------------
# Median values
# ----------------------------------------------------------------

def run_dust_pipeline(class_, gal_id, sigmas, bal_abs_dir, save_dir,
                      n_mc=100, line_groups=None, make_plots=True):

    os.makedirs(save_dir, exist_ok=True)

    if line_groups is None:
        line_groups = DIAGNOSTIC_LINES

    all_diag_lines = sorted({line for lines in line_groups.values()
                             for line in lines})
    all_lines = sorted(set(BALMER_DECREMENT) | set(all_diag_lines))

    wave, flux, sigma = load_corrected_spectrum(gal_id, bal_abs_dir)

    before_results = measure_lines(class_, wave, flux, sigma,
                                   all_lines, sigmas, n_mc=n_mc)
    balmer_results = {line: before_results[line]
                      for line in BALMER_DECREMENT}

    E_BV, E_BV_err, E_BV_samples = estimate_E_BV_mc(class_, balmer_results)

    corrected_flux, corrected_err, spec_save_path = \
        correct_and_save_spectrum(gal_id, wave, flux, sigma, E_BV, save_dir)

    after_results = measure_lines(class_, wave, corrected_flux, corrected_err,
                                  all_diag_lines, sigmas, n_mc=n_mc)
    diagnostics = {group: {line: after_results[line] for line in lines}
                   for group, lines in line_groups.items()}

    results = {
        'gal_id': gal_id,
        'E_BV': E_BV, 'E_BV_err': E_BV_err, 'E_BV_samples': E_BV_samples,
        'wave': wave, 'flux_before': flux, 'flux_after': corrected_flux,
        'err_after': corrected_err,
        'balmer_results': balmer_results,
        'before_results': before_results,
        'diagnostics': diagnostics,
        'spectrum_save_path': spec_save_path,
    }

    if make_plots:
        plot_dir = f'{save_dir}/plots'
        os.makedirs(plot_dir, exist_ok=True)

        plot_dust_correction(wave, flux, corrected_flux, gal_id,
                             save_path=f'{plot_dir}/dust_corr_{gal_id}.png')

        plot_E_BV_distribution(gal_id, E_BV_samples,
                               save_path=f'{plot_dir}/E_BV_mc_{gal_id}.png')

        Ha_total_before = (balmer_results['H1_6563A']['narrow']
                           + balmer_results['H1_6563A']['broad'])
        Hb_total_before = (balmer_results['H1_4861A']['narrow']
                           + balmer_results['H1_4861A']['broad'])
        plot_balmer_decrement_check(
            gal_id, Ha_total_before, Hb_total_before,
            save_path=f'{plot_dir}/balmer_decrement_{gal_id}.png')

        comparison_results = []
        for line in all_diag_lines:
            before_total = (before_results[line]['narrow']
                            + before_results[line]['broad'])
            after_total = (after_results[line]['narrow']
                           + after_results[line]['broad'])

            comp = plot_flux_comparison(
                line, before_total, after_total,
                save_path=f'{plot_dir}/flux_compare_{line}_{gal_id}.png')
            comparison_results.append(comp)

        results['comparison_results'] = comparison_results

    return results

# ----------------------------------------------------------------
# Results for each MC realization
# ----------------------------------------------------------------


def fit_line_once(class_, wave, flux, err, line, sigmas):
    masked_data = mask_nonuse_emission_line(class_, sigmas, line,
                                            [wave, flux, err])
    isolated = isolate_emission_line(class_, line, datas=masked_data)

    fit = model_flux_list(class_, isolated, [line], sigmas, plot=False)
    narrow = fit.params[f'{line}_narrow_amplitude'].value
    broad = fit.params[f'{line}_broad_amplitude'].value
    return narrow + broad


def E_BV_single_draw(class_, balmer_fluxes):

    lines = BALMER_DECREMENT
    hb_flux = balmer_fluxes['H1_4861A']

    wl = {line: class_.linelist_dict[line] for line in lines}
    hb_wl = wl['H1_4861A']

    x_vals, y_vals, ratios_out = [], [], {}
    for line in lines:
        ratio = balmer_fluxes[line] / hb_flux
        ratios_out[line] = ratio
        if line == 'H1_4861A':
            continue
        x = k_cal(np.array([wl[line]]))[0] - k_cal(np.array([hb_wl]))[0]
        y = np.log10(ratio / THEORETICAL_RATIOS[line])
        x_vals.append(x)
        y_vals.append(y)

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    slope = -np.sum(x_vals * y_vals) / np.sum(x_vals**2)
    E_BV = slope / 0.4

    return E_BV, ratios_out


def run_full_mc_pipeline(class_, gal_id, sigmas, bal_abs_dir, save_dir,
                         n_mc=200, line_groups=None, save_spectrum=True):

    if line_groups is None:
        line_groups = DIAGNOSTIC_LINES

    all_diag_lines = sorted({line for lines in line_groups.values()
                             for line in lines})

    wave, flux, sigma = load_corrected_spectrum(gal_id, bal_abs_dir)

    balmer_flux_samples = {line: np.zeros(n_mc) for line in BALMER_DECREMENT}
    E_BV_samples = np.zeros(n_mc)
    decrement_samples = {line: np.zeros(n_mc) for line in BALMER_DECREMENT}
    diag_before_samples = {line: np.zeros(n_mc) for line in all_diag_lines}
    diag_after_samples = {line: np.zeros(n_mc) for line in all_diag_lines}

    for i in range(n_mc):
        if i % max(1, n_mc // 10) == 0:
            print(f'  draw {i}/{n_mc}')

        flux_i = flux + np.random.randn(len(flux)) * sigma

        balmer_fluxes_i = {}
        for line in BALMER_DECREMENT:
            f = fit_line_once(class_, wave, flux_i, sigma, line, sigmas)
            balmer_fluxes_i[line] = f
            balmer_flux_samples[line][i] = f

        E_BV_i, ratios_i = E_BV_single_draw(class_, balmer_fluxes_i)
        E_BV_samples[i] = E_BV_i
        for line in BALMER_DECREMENT:
            decrement_samples[line][i] = ratios_i[line]

        corrected_flux_i = deredden_spectrum(wave, flux_i, E_BV_i)

        for line in all_diag_lines:
            diag_before_samples[line][i] = fit_line_once(
                class_, wave, flux_i, sigma, line, sigmas)
            diag_after_samples[line][i] = fit_line_once(
                class_, wave, corrected_flux_i, sigma, line, sigmas)

    pct_diff_samples, significance = {}, {}
    for line in all_diag_lines:
        before = diag_before_samples[line]
        after = diag_after_samples[line]
        pct_diff_samples[line] = (after - before) / before * 100
        diff = after - before
        significance[line] = np.median(diff) / np.std(diff)

    results = {
        'gal_id': gal_id,
        'wave': wave, 'flux': flux, 'sigma': sigma,
        'balmer_flux_samples': balmer_flux_samples,
        'E_BV_samples': E_BV_samples,
        'decrement_samples': decrement_samples,
        'diag_before_samples': diag_before_samples,
        'diag_after_samples': diag_after_samples,
        'pct_diff_samples': pct_diff_samples,
        'significance': significance,
        'line_groups': line_groups,
    }
    E_BV_median = np.median(E_BV_samples)
    E_BV_std = np.std(E_BV_samples)
    results['E_BV_median'] = E_BV_median
    results['E_BV_std'] = E_BV_std

    if save_spectrum:
        corrected_flux, corrected_err = deredden_spectrum(
            wave, flux, E_BV_median, flux_err=sigma)
        spec_save_path = save_corrected_spectrum(
            gal_id, wave, corrected_flux, corrected_err, save_dir=save_dir)
        results['corrected_flux'] = corrected_flux
        results['corrected_err'] = corrected_err
        results['spectrum_save_path'] = spec_save_path
    return results


def plot_full_mc_results(results, save_dir):

    gal_id = results['gal_id']
    plot_dir = f'{save_dir}/plots_full_mc'
    os.makedirs(plot_dir, exist_ok=True)
    figs = {}

    fig, axs = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, line in zip(axs, BALMER_DECREMENT):
        samples = results['balmer_flux_samples'][line]
        ax.hist(samples, bins=30, color='tab:blue', alpha=0.8)
        ax.axvline(np.median(samples), color='black', linestyle='--')
        ax.set_title(f'{line}\nmedian={np.median(samples):.2f}')
        ax.set_xlabel('Flux')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/balmer_fluxes_{gal_id}.png', format='png')
    plt.show()
    figs['balmer_fluxes'] = fig

    figs['E_BV'] = plot_E_BV_distribution(
        gal_id, results['E_BV_samples'],
        save_path=f'{plot_dir}/E_BV_{gal_id}.png')

    fig, axs = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, line in zip(axs, BALMER_DECREMENT):
        samples = results['decrement_samples'][line]
        ax.hist(samples, bins=10, color='tab:green', alpha=0.8)
        # ax.axvline(THEORETICAL_RATIOS[line], color='red', linestyle='--',
        #            label=f'theory={THEORETICAL_RATIOS[line]}')
        median = np.median(samples)
        text = f'median={median:.2f}\ntheory={THEORETICAL_RATIOS[line]}'
        ax.axvline(np.median(samples), color='black', linestyle=':',
                   label=text)
        ax.set_title(f'{line} / H_beta')
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/decrements_{gal_id}.png', format='png')
    plt.show()
    figs['decrements'] = fig

    diag_figs = {}
    for line in results['diag_before_samples']:
        before = results['diag_before_samples'][line]
        after = results['diag_after_samples'][line]
        pct = results['pct_diff_samples'][line]
        sig = results['significance'][line]

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].hist(before, bins=10, alpha=0.6, color='black', label='Before')
        axs[0].hist(after, bins=10, alpha=0.6, color='tab:red', label='After')
        axs[0].set_title(f'{line}: flux before/after')
        axs[0].set_xlabel('Flux')
        axs[0].legend(fontsize=8)

        axs[1].hist(pct, bins=10, color='tab:purple', alpha=0.8)
        axs[1].axvline(np.median(pct), color='black', linestyle='--')
        axs[1].set_title(f'{line}: paired % diff\n{sig:.2f}$\\sigma$')
        axs[1].set_xlabel('% difference')

        plt.tight_layout()
        plt.savefig(f'{plot_dir}/diag_{line}_{gal_id}.png', format='png')
        plt.show()
        diag_figs[line] = fig

    figs['diagnostics'] = diag_figs
    return figs


def get_representative_corrected_spectrum(results):

    wave = results['wave']
    flux = results['flux']
    E_BV_median = results.get('E_BV_median',
                              np.median(results['E_BV_samples']))

    if 'corrected_flux' in results:
        corrected_flux = results['corrected_flux']
    else:
        corrected_flux = deredden_spectrum(wave, flux, E_BV_median)

    return wave, flux, corrected_flux, E_BV_median


def plot_corrected_spectrum_full(results, save_path=None):

    wave, flux, corrected_flux, E_BV_median = \
        get_representative_corrected_spectrum(results)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.step(wave, flux, label='Before dereddening', color='black', alpha=0.6)
    ax.step(wave, corrected_flux, label='After dereddening', color='tab:red')
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Flux')
    ax.set_title(f"{results['gal_id']}: full spectrum "
                 f"(E(B-V)={E_BV_median:.3f})")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png')
    plt.show()
    return fig


def plot_corrected_spectrum_line_windows(class_, results, lines=None,
                                         window=30, save_path=None):

    if lines is None:
        lines = BALMER_DECREMENT

    wave, flux, corrected_flux, E_BV_median = \
        get_representative_corrected_spectrum(results)

    z = class_.redshift
    fig, axs = plt.subplots(1, len(lines), figsize=(5 * len(lines), 4),
                            squeeze=False)
    axs = axs[0]

    for ax, line in zip(axs, lines):
        center = class_.linelist_dict[line] * (1 + z)
        mask = (wave > center - window) & (wave < center + window)

        ax.step(wave[mask], flux[mask], label='Before', color='black',
                alpha=0.6)
        ax.step(wave[mask], corrected_flux[mask], label='After',
                color='tab:red')
        ax.axvline(center, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(line)
        ax.set_xlabel('Wavelength (Å)')
        ax.legend(fontsize=8)

    axs[0].set_ylabel('Flux')
    fig.suptitle(f"{results['gal_id']}: line windows "
                 f"(E(B-V)={E_BV_median:.3f})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png')
    plt.show()
    return fig


def plot_calzetti_curve(E_BV=None, wave_range=(3500, 9500), class_=None,
                        lines_to_mark=None, save_path=None):

    wave_grid = np.linspace(wave_range[0], wave_range[1], 1000)
    k_grid = k_cal(wave_grid)

    ncols = 2 if E_BV is not None else 1
    fig, axs = plt.subplots(1, ncols, figsize=(6 * ncols, 4.5), squeeze=False)
    axs = axs[0]

    axs[0].plot(wave_grid, k_grid, color='black')
    axs[0].set_xlabel('Wavelength (Å)')
    axs[0].set_ylabel(r'$k(\lambda)$')
    axs[0].set_title('Calzetti attenuation curve')

    if E_BV is not None:
        atten_grid = 10 ** (0.4 * E_BV * k_grid)
        axs[1].plot(wave_grid, atten_grid, color='tab:red')
        axs[1].axhline(1.0, color='gray', linestyle=':', alpha=0.6)
        axs[1].set_xlabel('Wavelength (Å)')
        axs[1].set_ylabel('Attenuation factor (flux multiplier)')
        axs[1].set_title(f'Applied correction, E(B-V)={E_BV:.3f}')

    if class_ is not None and lines_to_mark is not None:
        z = class_.redshift
        for line in lines_to_mark:
            obs_wl = class_.linelist_dict[line] * (1 + z)
            if wave_range[0] <= obs_wl <= wave_range[1]:
                for ax in axs:
                    ax.axvline(obs_wl, color='tab:blue', linestyle='--',
                               alpha=0.3)
                axs[0].text(obs_wl, axs[0].get_ylim()[1] * 0.98, line,
                            rotation=90, ha='right', va='top', fontsize=7,
                            color='tab:blue')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png')
    plt.show()
    return fig


if __name__ == '__main__':
    # Example usage, once you have a class_/sigmas object for a galaxy:
    # results = run_full_mc_pipeline(
    #     class_, gal_id='J0203', sigmas=sigmas,
    #     bal_abs_dir=f'{DIR}bal_abs', save_dir=f'{DIR}dust_corr', n_mc=200)
    # figs = plot_full_mc_results(results, save_dir=f'{DIR}dust_corr')
    pass
