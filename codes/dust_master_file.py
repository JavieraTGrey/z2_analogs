"""
Master tables of the dust-corrected line fluxes, in the format of
magE2024_master_Dcorr.csv (one row per galaxy, {label}_flux and
{label}_fluxerr for every line, with the same line labels as the rest of
the pipeline, e.g. H1_4861A, O3_5007A), for each dust law, plus the same table
with one row per galaxy and MC realization, so that ratios, T_e and n_e
can be computed in every realization.

For each law it writes, in out_dir:
  magE2024_master{tag}_{law}.csv               median and half 16-84 range
  magE2024_master{tag}_{law}_realizations.csv  one row per galaxy and i
law 'none' uses the fluxes before the internal dust correction.
"""
import os
import glob
import re
import numpy as np
import pandas as pd

MISSING = -9999.9


def label_wavelength(label):
    m = re.match(r'^([A-Za-z]+\d*)_(\d+)A$', label)
    if m is None:
        return np.inf
    return float(m.group(2))


def sort_labels(labels):
    pairs = []
    for label in labels:
        pairs.append((label_wavelength(label), label))
    pairs.sort()
    out = []
    for _, label in pairs:
        out.append(label)
    return out


def info_path(gal, law, tag, results_dir):
    folder = f'{results_dir}/{gal}/lines'
    if law == 'none':
        return f'{folder}/{gal}_lines_info{tag}.csv'
    if law == 'power':
        found = sorted(glob.glob(f'{folder}/{gal}_lines_info{tag}_dustcorr_power_*.csv'))
        if len(found) == 0:
            return None
        return found[-1]
    return f'{folder}/{gal}_lines_info{tag}_dustcorr_{law}.csv'


def realization_fluxes(info):
    d = info.copy()
    d['flux'] = d['nar_amplitude'] + d['bro_amplitude']
    table = d.pivot_table(index='i', columns='label', values='flux',
                          aggfunc='first')
    narrow = d.pivot_table(index='i', columns='label',
                           values='nar_amplitude', aggfunc='first')
    return table, narrow


def summarize(values):
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return MISSING, MISSING
    p16 = np.percentile(v, 16)
    p50 = np.percentile(v, 50)
    p84 = np.percentile(v, 84)
    return p50, (p84 - p16) / 2


def build_master(classes, law, tag='_final_mw_abs',
                 results_dir=None, out_dir=None, proj_DIR='',
                 drop_negative=True):
    if results_dir is None:
        results_dir = f'{proj_DIR}new_results'
    if out_dir is None:
        out_dir = f'{results_dir}/master'
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []
    real_rows = []
    for class_ in classes:
        gal = class_.gal_id
        path = info_path(gal, law, tag, results_dir)
        if path is None or not os.path.exists(path):
            print(f'{gal}: no file for law {law} ({path})')
            continue
        info = pd.read_csv(path)
        if 'Unnamed: 0' in info.columns:
            info = info.drop(columns=['Unnamed: 0'])
        fluxes, narrow = realization_fluxes(info)
        if drop_negative:
            fluxes = fluxes.where(narrow > 0)

        names = sort_labels(list(fluxes.columns))

        gal_name = class_.names[0]
        mass = class_.mass

        row = {'ID': gal_name, 'mass': mass, 'z': class_.redshift}
        errs = {}
        for name in names:
            f, e = summarize(fluxes[name].values)
            row[f'{name}_flux'] = f
            row[f'{name}_fluxerr'] = e
            errs[name] = e
        summary_rows.append(row)

        for i in fluxes.index:
            r = {'ID': gal_name, 'mass': mass, 'z': class_.redshift, 'i': i}
            for name in names:
                val = fluxes.loc[i, name]
                r[f'{name}_flux'] = val if np.isfinite(val) else MISSING
                r[f'{name}_fluxerr'] = errs[name]
            real_rows.append(r)

    summ = pd.DataFrame(summary_rows).fillna(MISSING)
    real = pd.DataFrame(real_rows).fillna(MISSING)
    p1 = f'{out_dir}/magE2024_master{tag}_{law}.csv'
    p2 = f'{out_dir}/magE2024_master{tag}_{law}_realizations.csv'
    summ.to_csv(p1)
    real.to_csv(p2)
    print(f'Saved {p1}\nSaved {p2}')
    return summ, real
