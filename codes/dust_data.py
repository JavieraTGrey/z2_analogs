"""
Inputs for the dust correction: line lists, Case B ratios and the saved
Monte Carlo fluxes.
"""
from GALAXIES import PROJ_DIR
import numpy as np
import pandas as pd
import pyneb as pn

# ================================================================
# Line lists
# ================================================================

# -----------------------------------------
# Overall information
# -----------------------------------------
# Exclude lines from H library (blended and low S/N)
EXCLUDE_DEFAULT = {
    'H1_3704A',
    'H1_3722A',
    'H1_3734A',
    'H1_3771A',
    'H1_3889A',
    'H1_3970A',
    'H1_8392A',
    'H1_8438A',
    'H1_8467A',
    'H1_8502A',
    'H1_8545A',
    'H1_8413A'}


# Lines for the extrapolation check
DIAGNOSTIC_GROUPS = {
    # auroral lines (T_e)
    'auroral': ['O3_4363A',                      # T[O III]
                'N2_5755A',                      # T[N II]
                'O2_7319A', 'O2_7320A',          # T[O II]
                'O2_7330A', 'O2_7331A',          # T[O II]
                'S3_6312A',                      # S++
                'S2_4069A', 'S2_4076A'],         # T[S II]
    # nebular lines paired with the auroral
    'nebular': ['O3_4959A', 'O3_5007A',
                'N2_6548A', 'N2_6583A',
                'O2_3726A', 'O2_3729A',
                'S2_6716A', 'S2_6731A'],
    # other ionic abundances
    'abundance': ['Ne3_3869A', 'Ar3_7136A',
                  'He1_5876A', 'He2_4686A'],
}


DIAGNOSTIC_LINES = []


for g in DIAGNOSTIC_GROUPS.values():
    for line in g:
        if line not in DIAGNOSTIC_LINES:
            DIAGNOSTIC_LINES.append(line)


LINE_ROLE = {}


for g, ls in DIAGNOSTIC_GROUPS.items():
    for line in ls:
        LINE_ROLE[line] = g


# Line ratios used for the physical conditions: (numerator, denominator,
# what it measures). Sums are summed fluxes.
DIAGNOSTIC_RATIOS = {
    'T[O III]':  (['O3_4363A'], ['O3_4959A', 'O3_5007A'], 'T_e high-ion'),
    'T[N II]':   (['N2_5755A'], ['N2_6548A', 'N2_6583A'], 'T_e low-ion'),
    'T[O II]':   (['O2_7319A', 'O2_7320A', 'O2_7330A', 'O2_7331A'],
                  ['O2_3726A', 'O2_3729A'], 'T_e low-ion'),
    'T[S II]':   (['S2_4069A', 'S2_4076A'], ['S2_6716A', 'S2_6731A'],
                  'T_e low-ion'),
    'n[S II]':   (['S2_6716A'], ['S2_6731A'], 'n_e low-ion'),
    'n[O II]':   (['O2_3729A'], ['O2_3726A'], 'n_e low-ion'),
    'O+/H+':     (['O2_3726A', 'O2_3729A'], ['H1_4861A'], 'abundance'),
    'O+/H+ (7325)': (['O2_7319A', 'O2_7320A', 'O2_7330A', 'O2_7331A'],
                     ['H1_4861A'], 'abundance'),
    'O++/H+':    (['O3_4959A', 'O3_5007A'], ['H1_4861A'], 'abundance'),
    'N+/H+':     (['N2_6548A', 'N2_6583A'], ['H1_4861A'], 'abundance'),
    'S+/H+':     (['S2_6716A', 'S2_6731A'], ['H1_4861A'], 'abundance'),
    'S++/H+ (6312)': (['S3_6312A'], ['H1_4861A'], 'abundance'),
    'Ne++/H+':   (['Ne3_3869A'], ['H1_4861A'], 'abundance'),
    'Ar++/H+':   (['Ar3_7136A'], ['H1_4861A'], 'abundance'),
    'He+/H+':    (['He1_5876A'], ['H1_4861A'], 'abundance'),
    'He++/H+':   (['He2_4686A'], ['H1_4861A'], 'abundance'),
}


# lines to check (labels not in your table are skipped)
NEBULAR = ['O2_3726A', 'O2_3729A', 'Ne3_3869A', 'O3_4959A', 'O3_5007A',
           'N2_6548A', 'N2_6583A', 'S2_6716A', 'S2_6731A', 'Ar3_7136A']


AURORAL = ['S2_4069A', 'S2_4076A', 'O3_4363A', 'N2_5755A', 'S3_6312A',
           'O2_7319A', 'O2_7320A', 'O2_7330A', 'O2_7331A']


# ================================================================
# Case B intrinsic ratios
# ================================================================

# Get the hydrogen transition from the Rydberg formula
def _hydrogen_levels(wl_vac, tol=3.0):
    R_H = 109677.58   # cm^-1
    best = None
    for n_low in (2, 3):  # Could be Paschen or Balmer
        # Up to 40 levels
        for n_up in range(n_low + 1, 41):
            # Rydberg formula
            lam_vac = 1e8 / (R_H * (1/n_low**2 - 1/n_up**2))

            # Check how "far away" is this from a known level
            d = abs(lam_vac - wl_vac)
            if best is None or d < best[0]:
                # Replace if closer
                best = (d, n_up, n_low)
    # Return the levels we got
    return best[1:] if best[0] < tol else None


def case_b_ratios(labels, vac_dict, linelist_dict=None, tem=1e4, den=100.0):
    """F(line)/F(H-beta) for Case B. Uses PyNeb when installed (all Balmer
    and Paschen lines), otherwise the Balmer fallback table."""

    H1 = pn.RecAtom('H', 1)
    hb = H1.getEmissivity(tem, den, 4, 2)
    source = 'PyNeb'

    ratios = {}
    for lab in labels:
        wl = rest_wavelength(lab, linelist_dict)
        if pn is not None:
            lev = _hydrogen_levels(vac_dict[lab])
            if lev is None:
                print(f'  {lab}: no H I transition near {wl:.1f} A, skipped')
                continue
            ratios[lab] = float(H1.getEmissivity(tem, den, *lev) / hb)

    print(f'Case B ratios from {source}, T={tem:.0f} K, n_e={den:.0f} cm^-3')
    return ratios


# Get restframe wv from the class_info
def rest_wavelength(label, linelist_dict=None):
    if linelist_dict is not None and label in linelist_dict:
        return float(linelist_dict[label])
    # H1_6573A, drop H1_ and the A at the end
    return float(label.split('_')[1].rstrip('A'))


# ================================================================
# Load saved MC fluxes
# ================================================================

# Get Path
def info_path(gal_id, results_dir=None, tag=''):
    results_dir = results_dir or f'{PROJ_DIR}new_results'
    return f'{results_dir}/{gal_id}/{gal_id}_lines_info{tag}.csv'


def load_flux_samples(gal_id, results_dir=None, info=None, tag=''):
    """Wide table of total flux (narrow + broad): rows = MC realization i,
    columns = line label.

    Changes the table to be
    (iter, line1, line 2), (0, num1, num2), .... etc
    """
    if info is None:
        info = pd.read_csv(info_path(gal_id, results_dir, tag))
        # Remove leftover index columns
        drop = []
        for c in info.columns:
            if c.startswith('Unnamed'):
                drop.append(c)
        info = info.drop(columns=drop)
    info = info.copy()
    # Add flux column to the table
    info['flux'] = info['nar_amplitude'] + info['bro_amplitude']
    return info.pivot_table(index='i', columns='label', values='flux')


def flux_table(info):
    """rows = MC realization i, columns = line, values = narrow + broad."""
    d = info.copy()
    d['flux'] = d['nar_amplitude'] + d['bro_amplitude']
    return d.pivot_table(index='i', columns='label', values='flux')


def summary_stats(values):
    """median and half of the 16-84 percentile range, ignoring NaN/inf."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    p16 = np.percentile(v, 16)
    p50 = np.percentile(v, 50)
    p84 = np.percentile(v, 84)
    return p50, (p84 - p16) / 2


# ================================================================
# Hydrogen lines used for E(B-V)
# ================================================================

def select_hydrogen_lines(file_table, linelist_dict=None,
                          exclude=EXCLUDE_DEFAULT,
                          min_snr=3.0):
    """
    H I lines with S/N >= min_snr (from the MC spread),
    minus selected lines
    """
    # Get all labels for Hydrogen lines
    labels = []
    for c in file_table.columns:
        if c.startswith('H1_'):
            labels.append(c)

    keep = []
    for lab in labels:
        # Get 100 iteration info for lab
        v = file_table[lab].dropna()
        q16, q50, q84 = np.percentile(v, [16, 50, 84])
        snr = q50 / max((q84 - q16) / 2, 1e-30)  # Evita dividir by 0

        if lab in exclude:
            print(f'  {lab}: excluded (blend)')
        elif snr < min_snr:
            print(f'  {lab}: excluded (S/N = {snr:.1f})')
        else:
            keep.append(lab)

    # sort by wavelength
    wl = []
    for line in keep:
        wl.append(rest_wavelength(line, linelist_dict))
    order = np.argsort(wl)
    keep_sorted = []
    for n in order:
        keep_sorted.append(keep[n])
    return keep_sorted
