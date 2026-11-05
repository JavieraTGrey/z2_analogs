"""
Calculate and correct for dust on the galaxies (main file).

The code is split in:
    dust_laws.py   the attenuation laws (k_law)
    dust_data.py   line lists, Case B ratios, loading the MC fluxes
    dust_fit.py    E(B-V), fitted law shapes, law comparison table
    dust_plots.py  plots
    dust_att.py    this file: run_dust, corrections and saving

Main entry points (all available from this file):
    res   = run_dust(class_, ...)                  E(B-V) and corrected fluxes
    table = make_law_comparison_plots(res, dir)    compare the laws
    make_all_plots(res, dir)                       plots for the adopted law
    save_corrected(res, law)                       save fluxes for any law
"""
from GALAXIES import PROJ_DIR
import os
import numpy as np
import pandas as pd
from dust_laws import RV_RANGE, k_law, ebv_scale, register_law, correction_factor
from dust_data import (EXCLUDE_DEFAULT, case_b_ratios, rest_wavelength,
                       load_flux_samples, select_hydrogen_lines)
from dust_fit import (ebv_per_line, ebv_combined, fit_law_param, fit_hybrid,
                      law_summary_table)
from dust_plots import (make_all_plots, make_law_comparison_plots,
                        plot_ebv_histograms, plot_ebv_vs_wavelength,
                        plot_decrement_curve, plot_relative_k, plot_alam_av,
                        plot_change_histograms, plot_attenuation_curves,
                        plot_law_ranking, plot_attenuation_and_ranking)

# ================================================================
# Correct the saved fluxes
# ================================================================

def correct_info(info, E_samples, law, linelist_dict=None, clip_negative=True):
    """
    Dereddened copy of the info table.
    Each row (one line, one MC realization i) is corrected with the
    E(B-V) of that same realization.
    """
    out = info.copy()

    factors = []
    for _, row in out.iterrows():
        label = row['label']
        i = row['i']

        # E(B-V) of this realization
        E = E_samples.loc[i]

        # Delete E(b-v) < 0 and change to 0
        if clip_negative and E < 0:
            E = 0

        # correction factor at this line's wavelength
        wl = rest_wavelength(label, linelist_dict)
        factor = correction_factor(wl, E, law)[0]
        factors.append(factor)

    out['corr_factor'] = factors
    out['nar_amplitude'] = out['nar_amplitude'] * out['corr_factor']
    out['bro_amplitude'] = out['bro_amplitude'] * out['corr_factor']
    return out


def flux_summary(info):
    """
    Median flux and 16-84 percentile error for each line
    """
    rows = []
    for label in info['label'].unique():
        # all realizations of this line
        d = info[info['label'] == label]
        d = d[d['nar_amplitude'] > 0]

        flux = d['nar_amplitude'] + d['bro_amplitude']

        if len(flux) == 0:
            rows.append({'label': label, 'flux': -9999.9,
                         'fluxerr': -9999.9, 'n_fits': 0})
            continue

        p16 = np.percentile(flux, 16)
        p50 = np.percentile(flux, 50)
        p84 = np.percentile(flux, 84)

        rows.append({'label': label,
                     'flux': p50,
                     'fluxerr': (p84 - p16) / 2,
                     'n_fits': len(flux)})

    summary = pd.DataFrame(rows)
    summary = summary.set_index('label')
    return summary


def save_corrected(res, law, results_dir=None):
    """Correct and save the fluxes with any law already computed in res
    (no refitting). Use after run_dust(..., apply_law=None) to save one or
    several laws. 'power' and 'hybrid' mean the fitted ones.
    Saves {gal}_lines_info_dustcorr_{law}.csv (all realizations) and
    {gal}_lines_summ_dustcorr_{law}.csv (median flux and error per line),
    and returns the corrected info table."""
    if law == 'power':
        law = res['power_law']
    if law == 'hybrid':
        law = res['hybrid_law']
    corrected = correct_info(res['info'], res['combined'][law], law,
                             res['linelist_dict'])
    gal_id = res['gal_id']
    results_dir = results_dir or f'{PROJ_DIR}new_results'
    safe = law.replace(':', '_')
    tag = res.get('tag', '')
    folder = f'{results_dir}/{gal_id}/lines/'
    os.makedirs(folder, exist_ok=True)
    corrected.to_csv(f'{folder}{gal_id}_lines_info{tag}_dustcorr_{safe}.csv',
                     index=False)
    flux_summary(corrected).to_csv(
        f'{folder}{gal_id}_lines_summ{tag}_dustcorr_{safe}.csv')
    print(f'Saved {gal_id}_lines_info/summ{tag}_dustcorr_{safe}.csv in {folder}')
    return corrected



# ================================================================
# Run everything for one galaxy
# ================================================================


def run_dust(class_, laws=('ccm', 'calzetti', 'power'), tag='_mw_abs',
             extra_exclude=(),
             rv_grid=None, hybrid=False, hybrid_red_frac=None,
             hybrid_smooth=False, hybrid_shape='power',
             apply_law='ccm',
             exclude=EXCLUDE_DEFAULT, min_snr=3.0, tem=1e4,
             den=100.0, save=True):
    """
    Dust correction for one galaxy, using the saved MC line fluxes.
 
    laws      : laws to compute E(B-V) with. 'power' means: fit beta;
                'ccm:fit' means: fit Cardelli's R_V;
                'calzetti:fit' means: fit the modified Calzetti tilt delta.
    rv_grid   : extra R_V values, e.g. {'ccm': [2.5, 4.0, 5.0],
                                         'calzetti': [3.25, 4.85]}
    hybrid    : also build the H-alpha-anchored hybrid law
    hybrid_red_frac : Cardelli fraction redward of the split for the hybrid
                (None = chosen by the red lines; 1 = Cardelli; 0 = Calzetti)
                (split: set dust_laws.HYBRID_SPLIT before calling)
    hybrid_smooth : only for hybrid_shape='power': True = no corner at the
                split (same slope on both sides)
    hybrid_shape : blue part of the hybrid: 'quad' = smooth quadratic in
                1/lambda (needs split < H-beta), 'power' = power law
    apply_law : law used for the saved corrected fluxes
                ('power' and 'hybrid' mean the fitted ones).
                None: only compute and compare the laws; no fluxes are
                corrected or saved (E(B-V) samples are still saved).
    tem, den  : T_e and n_e for the Case B ratios
    """
    gal_id = class_.names[0][:5]
 
    air_wave = {}
    for _, row in class_.load_line_list()[0].iterrows():
        air_wave[row['name']] = row['wavelength']
 
    vac_wave = {}
    for _, row in class_.load_line_list()[0].iterrows():
        vac_wave[row['name']] = row['vacuum_wave']
 
    print(f'=== {gal_id} ===')
 
    # load the saved line fits ----
    results_dir = f'{PROJ_DIR}new_results'
    info = pd.read_csv(f'{results_dir}/{gal_id}/lines/{gal_id}_lines_info{tag}.csv')
    # drop unnecessary columns
    if 'Unnamed: 0' in info.columns:
        info = info.drop(columns=['Unnamed: 0'])
 
    # Read fluxes as a table
    file_table = load_flux_samples(gal_id, info=info)
 
    # Check the Hydrogen lines to use
    excl = set(exclude)
    for line in extra_exclude:
        excl.add(line)
    lines = select_hydrogen_lines(file_table, air_wave, excl, min_snr)
 
    # Intrinsic ratios
    case_ratios = case_b_ratios(lines, vac_wave, tem=tem, den=den)
    # keep only lines that got a Case B ratio
    good_lines = []
    for line in lines:
        if line in case_ratios:
            good_lines.append(line)
 
    print(f'Using {len(good_lines)} H I lines: {good_lines}')
 
    # Build the full list of laws
    law_list = []
    power_law = None
    ccm_fit_law = None
    calz_fit_law = None
    for law in laws:
        if law == 'power':
            betas = np.arange(0.2, 8.01, 0.05)
            beta = fit_law_param(file_table, good_lines, case_ratios,
                                 air_wave, 'power', betas)
            b50 = beta.median()
            print(f'power law: beta = {b50:.2f} '
                  f'+{beta.quantile(0.84) - b50:.2f} '
                  f'-{b50 - beta.quantile(0.16):.2f}')
            if b50 <= betas[0] + 0.01 or b50 >= betas[-1] - 0.01:
                print('  warning: beta is at the edge of the grid; '
                      'the lines do not follow a power law')
            law = f'power:{b50:.2f}'
            power_law = law
        if law == 'ccm:fit':
            lo, hi = RV_RANGE['ccm']
            rvs = np.arange(lo, hi + 0.001, 0.05)
            rv_fit = fit_law_param(file_table, good_lines, case_ratios,
                                   air_wave, 'ccm', rvs)
            r50 = rv_fit.median()
            print(f'Cardelli: R_V = {r50:.2f} '
                  f'+{rv_fit.quantile(0.84) - r50:.2f} '
                  f'-{r50 - rv_fit.quantile(0.16):.2f}')
            if r50 <= lo + 0.01 or r50 >= hi - 0.01:
                print(f'  warning: R_V at the edge of the range {lo}-{hi}')
            law = f'ccm:{r50:.2f}'
            ccm_fit_law = law
        if law == 'calzetti:fit':
            deltas = np.arange(-1.0, 0.41, 0.02)
            d_fit = fit_law_param(file_table, good_lines, case_ratios,
                                  air_wave, 'calzetti_mod', deltas)
            d50 = d_fit.median()
            law = f'calzetti_mod:{d50:.2f}'
            print(f'modified Calzetti: delta = {d50:.2f} '
                  f'+{d_fit.quantile(0.84) - d50:.2f} '
                  f'-{d50 - d_fit.quantile(0.16):.2f}, '
                  f'R_V = {k_law(5500.0, law)[0]:.2f}')
            if d50 <= deltas[0] + 0.01 or d50 >= deltas[-1] - 0.01:
                print('  warning: delta at the edge of the grid (-1 to 0.4)')
            calz_fit_law = law
        law_list.append(law)
 
    if rv_grid is not None:
        for name in rv_grid:
            for rv in rv_grid[name]:
                law_list.append(f'{name}:{rv}')
 
    for law in law_list:
        register_law(law)
 
    # E(B-V) for each realization
    per_line = {}
    combined = {}
    for law in law_list:
        per_line[law] = ebv_per_line(file_table, good_lines, case_ratios, law,
                                     air_wave)
        combined[law] = ebv_combined(file_table, good_lines, case_ratios, law,
                                     air_wave)
 
        e = combined[law] * ebv_scale(law)     # always E(B-V)
        med = e.median()
        up = e.quantile(0.84) - med
        low = med - e.quantile(0.16)
        print(f'E(B-V) [{law}] = {med:.3f} +{up:.3f} -{low:.3f}')
 
    # Hybrid law: E(B-V) from H-alpha/H-beta only
    hybrid_law = None
    hybrid_npar = 0
    if hybrid and 'H1_6563A' in good_lines:
        hybrid_law, hybrid_npar = fit_hybrid(file_table, good_lines,
                                             case_ratios, air_wave,
                                             red_frac=hybrid_red_frac,
                                             smooth=hybrid_smooth,
                                             shape=hybrid_shape)
        register_law(hybrid_law)
        law_list.append(hybrid_law)
        per_line[hybrid_law] = ebv_per_line(file_table, good_lines,
                                            case_ratios, hybrid_law, air_wave)
        combined[hybrid_law] = per_line[hybrid_law]['H1_6563A']
        e = combined[hybrid_law]
        print(f'E(B-V) [{hybrid_law}] = {e.median():.3f} (from H-alpha)')
 
    # 'power' and 'hybrid' mean the fitted versions
    if apply_law == 'power':
        apply_law = power_law
    if apply_law == 'hybrid':
        apply_law = hybrid_law
 
    # Correct the line fluxes (skipped when no law is applied yet)
    corrected = None
    corrected_summary = None
    if apply_law is not None:
        corrected = correct_info(info, combined[apply_law], apply_law,
                                 air_wave)
        corrected_summary = flux_summary(corrected)
    else:
        print('No law applied: compare the laws, then rerun with apply_law')
 
    # Save new data
    if save:
        gal = class_.gal_id
        out_dir = f'{results_dir}/{gal}/lines/'
        os.makedirs(out_dir, exist_ok=True)
        for law in law_list:
            # E(B-V) from each line ratio (power/linear converted from A_Hb)
            edv_perline = pd.DataFrame(per_line[law]) * ebv_scale(law)
            safe = law.replace(':', '_')     # no ':' in file names
            edv_perline.to_csv(f'{out_dir}{tag}_ebv_perline_{safe}.csv')
 
        # E(B-V) per realization for every law (power/linear converted
        # from A_Hb), one column per law
        ebv_save = {}
        for law in law_list:
            ebv_save[law] = combined[law] * ebv_scale(law)
        pd.DataFrame(ebv_save).to_csv(f'{out_dir}/{gal_id}_{tag}_ebv_samples.csv')
        if corrected is not None:
            # law name in the file names, e.g. ..._dustcorr_ccm.csv, so
            # nothing is overwritten (not even the uncorrected tables)
            safe = apply_law.replace(':', '_')
            corrected.to_csv(
                f'{out_dir}/{gal_id}_lines_info{tag}_dustcorr_{safe}.csv',
                index=False)
            corrected_summary.to_csv(
                f'{out_dir}/{gal_id}_lines_summ{tag}_dustcorr_{safe}.csv')
        print(f'Saved results in {out_dir}')
 
    # Plotting info
    res = {
        'gal_id': gal_id,
        'lines': good_lines,
        'R': case_ratios,
        'F': file_table,
        'linelist_dict': air_wave,
        'per_line': per_line,
        'combined': combined,
        'apply_law': apply_law,
        'laws': law_list,
        'power_law': power_law,
        'ccm_fit_law': ccm_fit_law,
        'calz_fit_law': calz_fit_law,
        'hybrid_law': hybrid_law,
        'hybrid_npar': hybrid_npar,
        'corrected_info': corrected,
        'corrected_summary': corrected_summary,
        'tag': tag,
        'info': info
    }
    return res
