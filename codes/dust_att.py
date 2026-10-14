"""
Calculate and correct for dust on the galaxies
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GALAXIES import PROJ_DIR


# -----------------------------------------
# Overall information
# -----------------------------------------

LAWS = ('ccm', 'calzetti')

RV = {'ccm': 3.1, 'calzetti': 4.05}

LAW_LABEL = {'ccm': 'Cardelli (R$_V$=3.1)',
             'calzetti': 'Calzetti (R$_V$=4.05)'}

LAW_COLOR = {'ccm': 'teal', 'calzetti': 'orange'}

LAW_MARKER = {'ccm': 'o', 'calzetti': 's'}


# Exclude lines from H library (blended and low S/N)
EXCLUDE_DEFAULT = {
    'H1_3889A',
    'H1_3970A',
    'H1_3722A',
    'H1_3734A',
    'H1_8502A',
    'H1_8545A'}

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
    'T[S II]':   (['S2_4069A', 'S2_4076A'], ['S2_6716A', 'S2_6731A'], 'T_e low-ion'),
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

# Subset labelled on the correction-factor plot (all lines would overlap)
PLOT_LINES = ['O2_3729A', 'O2_3726A', 'Ne3_3869A', 'S2_4069A', 'O3_4363A',
              'O3_5007A', 'N2_5755A', 'S3_6312A', 'N2_6583A', 'S2_6716A',
              'Ar3_7136A', 'O2_7320A']


# ================================================================
# Attenuation curves
# ================================================================


def k_ccm(wl, Rv=3.1):
    """Cardelli, Clayton & Mathis (1989) k(lambda) = A_lambda / E(B-V).
    wl in Angstrom (rest frame). Valid 1000 A - 3.3 micron."""

    # Get x in 1/microns
    x = 1e4 / np.atleast_1d(np.asarray(wl, dtype=float))

    # Copy shape of x but filled of nans
    a = np.full_like(x, np.nan)
    b = np.full_like(x, np.nan)

    # IR part
    ir = (x >= 0.3) & (x < 1.1)
    a[ir] = 0.574 * x[ir]**1.61
    b[ir] = -0.527 * x[ir]**1.61

    # Optical part
    opt = (x >= 1.1) & (x < 3.3)
    y = x[opt] - 1.82

    a[opt] = (1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4
              + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7)

    b[opt] = (1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4
              - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7)

    # UV part
    uv = (x >= 3.3) & (x <= 8.0)
    xu = x[uv]

    fa = np.where(xu >= 5.9, -0.04473*(xu-5.9)**2 - 0.009779*(xu-5.9)**3, 0)
    fb = np.where(xu >= 5.9, 0.2130*(xu-5.9)**2 + 0.1207*(xu-5.9)**3, 0)

    a[uv] = 1.752 - 0.316*xu - 0.104 / ((xu - 4.67)**2 + 0.341) + fa
    b[uv] = -3.090 + 1.825*xu + 1.206 / ((xu - 4.62)**2 + 0.263) + fb

    return Rv * a + b


def k_calzetti(wl, Rv=4.05):
    """Calzetti et al. (2000) k(lambda). wl in Angstrom (rest frame).
    Valid 1200 A - 2.2 micron."""

    # Get lam in 1/microns
    lam = np.atleast_1d(np.asarray(wl, dtype=float)) / 1e4   # micron
    k = np.where(
        lam < 0.63,
        2.659 * (-2.156 + 1.509/lam - 0.198/lam**2 + 0.011/lam**3) + Rv,
        2.659 * (-1.857 + 1.040/lam) + Rv)
    return k


def k_law(wl, law):
    if law == 'ccm':
        return k_ccm(wl)
    if law == 'calzetti':
        return k_calzetti(wl)


def correction_factor(wl, E_BV, law):
    '''
    f_corr = f_obs * 10**(0.4 E(B-V)*k(lamda))
    Multiply observed fluxes by this to correct them.
    '''
    return 10 ** (0.4 * np.asarray(E_BV) * k_law(wl, law))


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

    try:
        import pyneb as pn
        H1 = pn.RecAtom('H', 1)
        hb = H1.getEmissivity(tem, den, 4, 2)
        source = 'PyNeb'
    except Exception:
        pn = None
        source = 'fallback table (Storey & Hummer 1995)'
 
    ratios = {}
    for lab in labels:
        wl = rest_wavelength(lab, linelist_dict)
        if pn is not None:
            lev = _hydrogen_levels(vac_dict[lab])
            if lev is None:
                print(f'  {lab}: no H I transition near {wl:.1f} A, skipped')
                continue
            ratios[lab] = float(H1.getEmissivity(tem, den, *lev) / hb)
        else:
            key = min(CASE_B_FALLBACK, key=lambda w: abs(w - wl))
            if abs(key - wl) < 3:
                ratios[lab] = CASE_B_FALLBACK[key]
            else:
                print(f'  {lab}: not in fallback table (install PyNeb), skipped')
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

def info_path(gal_id, results_dir=None):
    results_dir = results_dir or f'{PROJ_DIR}new_results'
    return f'{results_dir}/{gal_id}/{gal_id}_lines_info.csv'


def load_flux_samples(gal_id, results_dir=None, info=None):
    """Wide table of total flux (narrow + broad): rows = MC realization i,
    columns = line label.

    Changes the table to be
    (iter, line1, line 2), (0, num1, num2), .... etc
    """
    if info is None:
        info = pd.read_csv(info_path(gal_id, results_dir))
        # Remove leftover index columns
        info = info.drop(columns=[c for c in info.columns
                                  if c.startswith('Unnamed')])
    info = info.copy()
    # Add flux column to the table
    info['flux'] = info['nar_amplitude'] + info['bro_amplitude']
    return info.pivot_table(index='i', columns='label', values='flux')


# ================================================================
# E(B-V)
# ================================================================
 
def select_hydrogen_lines(F, linelist_dict=None, exclude=EXCLUDE_DEFAULT,
                          min_snr=3.0):
    """H I lines with S/N >= min_snr (from the MC spread), minus blends."""
    labels = [c for c in F.columns if c.startswith('H1_')]
    keep = []
    for lab in labels:
        v = F[lab].dropna()
        q16, q50, q84 = np.percentile(v, [16, 50, 84])
        snr = q50 / max((q84 - q16) / 2, 1e-30)
        if lab in exclude:
            print(f'  {lab}: excluded (blend)')
        elif snr < min_snr:
            print(f'  {lab}: excluded (S/N = {snr:.1f})')
        else:
            keep.append(lab)
    if 'H1_4861A' not in keep:
        raise ValueError('H-beta is not usable; cannot measure E(B-V).')
    return sorted(keep, key=lambda l: rest_wavelength(l, linelist_dict))
 
 
def ebv_per_line(F, lines, R, law, linelist_dict=None):
    """E(B-V) from each ratio F_j/F_Hb, per realization (DataFrame i x line)."""
    k_hb = k_law(rest_wavelength('H1_4861A', linelist_dict), law)[0]
    out = {}
    for lab in lines:
        if lab == 'H1_4861A':
            continue
        dk = k_law(rest_wavelength(lab, linelist_dict), law)[0] - k_hb
        with np.errstate(divide='ignore', invalid='ignore'):
            obs = F[lab] / F['H1_4861A']
            out[lab] = -2.5 / dk * np.log10(obs / R[lab])
    return pd.DataFrame(out)
 
 
def ebv_combined(F, lines, R, law, linelist_dict=None):
    """Weighted fit of log10(F_j/R_j) = a - 0.4 E k_j over all lines, per
    realization. Weights from the MC scatter of each line."""
    k = np.array([k_law(rest_wavelength(l, linelist_dict), law)[0]
                  for l in lines])
    with np.errstate(divide='ignore', invalid='ignore'):
        Y = np.log10(F[lines].to_numpy() / np.array([R[l] for l in lines]))
    sig = np.nanstd(Y, axis=0)
    w = 1 / np.where(sig > 0, sig, np.nan)**2
 
    E = np.full(len(Y), np.nan)
    for n, y in enumerate(Y):
        ok = np.isfinite(y) & np.isfinite(w)
        if ok.sum() < 2:
            continue
        A = np.vstack([np.ones(ok.sum()), -0.4 * k[ok]]).T
        sw = np.sqrt(w[ok])
        coef, *_ = np.linalg.lstsq(A * sw[:, None], y[ok] * sw, rcond=None)
        E[n] = coef[1]
    return pd.Series(E, index=F.index, name=f'E_BV_{law}')
 
 
# ================================================================
# Correct the saved fluxes
# ================================================================
 
def correct_info(info, E_samples, law, linelist_dict=None, clip_negative=True):
    """Dereddened copy of the info table, realization by realization."""
    E = E_samples.clip(lower=0) if clip_negative else E_samples
    out = info.copy()
    wl = out['label'].map(lambda l: rest_wavelength(l, linelist_dict))
    fac = correction_factor(wl.to_numpy(), E.reindex(out['i']).to_numpy(), law)
    out['corr_factor'] = fac
    out['nar_amplitude'] = out['nar_amplitude'] * fac
    out['bro_amplitude'] = out['bro_amplitude'] * fac
    return out
 
 
def flux_summary(info):
    d = info.assign(flux=info['nar_amplitude'] + info['bro_amplitude'])
    d = d[d['nar_amplitude'] > 0]
    q = d.groupby('label')['flux'].quantile([0.16, 0.5, 0.84]).unstack()
    return pd.DataFrame({'flux': q[0.5], 'fluxerr': (q[0.84] - q[0.16]) / 2,
                         'n_fits': d.groupby('label').size()})
 
 
# ================================================================
# Run everything for one galaxy
# ================================================================
 
def run_dust(gal_id, linelist_dict=None, laws=LAWS, apply_law='ccm',
             results_dir=None, info=None, exclude=EXCLUDE_DEFAULT,
             min_snr=3.0, tem=1e4, den=100.0, spec_range=None, save=True):
    """
    linelist_dict : class_.linelist_dict (rest wavelengths). Optional.
    apply_law     : law used for the corrected fluxes that get saved.
    spec_range    : (min, max) REST wavelength of the spectrum, for the
                    extrapolation check, e.g. (wave.min(), wave.max())/(1+z).
    """
    print(f'=== {gal_id} ===')
    ll = pd.read_csv('lines_info.csv')
    linelist_dict = dict(zip(ll['name'], ll['wavelength']))
    vac_dict = dict(zip(ll['name'], ll['wave_vac']))
    if info is None:
        info = pd.read_csv(info_path(gal_id, results_dir))
        info = info.drop(columns=[c for c in info.columns
                                  if c.startswith('Unnamed')])
    F = load_flux_samples(gal_id, info=info)
 
    lines = select_hydrogen_lines(F, linelist_dict, exclude, min_snr)
    R = case_b_ratios(lines, vac_dict, linelist_dict, tem, den)
    lines = [l for l in lines if l in R]
    print(f'Using {len(lines)} H I lines: {lines}')
 
    res = {'gal_id': gal_id, 'lines': lines, 'R': R, 'F': F,
           'linelist_dict': linelist_dict, 'spec_range': spec_range,
           'per_line': {}, 'combined': {}}
    for law in laws:
        res['per_line'][law] = ebv_per_line(F, lines, R, law, linelist_dict)
        res['combined'][law] = ebv_combined(F, lines, R, law, linelist_dict)
        e = res['combined'][law]
        print(f'  E(B-V) [{law}] = {e.median():.3f} '
              f'+{e.quantile(.84) - e.median():.3f} '
              f'-{e.median() - e.quantile(.16):.3f}')
 
    corrected = correct_info(info, res['combined'][apply_law], apply_law,
                             linelist_dict)
    res['corrected_info'] = corrected
    res['corrected_summary'] = flux_summary(corrected)
    res['apply_law'] = apply_law
 
    if save:
        out_dir = f"{results_dir or DIR + 'new_results'}/{gal_id}"
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(res['combined']).to_csv(f'{out_dir}/{gal_id}_ebv_samples.csv')
        corrected.to_csv(f'{out_dir}/{gal_id}_lines_info_dustcorr.csv', index=False)
        res['corrected_summary'].to_csv(f'{out_dir}/{gal_id}_lines_summ_dustcorr.csv')
        print(f'Saved E(B-V) samples and dust-corrected tables in {out_dir}')
    return res
 
 
# ================================================================
# Plots
# ================================================================
 
def _q(v):
    v = pd.Series(v).dropna()
    return v.quantile(0.16), v.median(), v.quantile(0.84)
 
 
def plot_ebv_per_line(res, law='ccm', bins=30, save_path=None):
    """(3) E(B-V) from each line ratio, one panel per ratio, all
    realizations. H-alpha/H-beta first. Dashed line = combined fit."""
    per = res['per_line'][law]
    comb = res['combined'][law]
    cols = sorted(per.columns, key=lambda c: (c != 'H1_6563A',
                  rest_wavelength(c, res['linelist_dict'])))
    ncol = min(4, len(cols))
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.8 * nrow),
                            sharex=True, squeeze=False)
    finite = per.replace([np.inf, -np.inf], np.nan).stack()
    lo, hi = finite.quantile([0.01, 0.99])
    edges = np.linspace(min(lo, comb.min()), max(hi, comb.max()), bins + 1)
    c_med = comb.median()
 
    for ax, lab in zip(axs.flat, cols):
        v = per[lab].replace([np.inf, -np.inf], np.nan).dropna()
        c = '#1b4f9c' if lab == 'H1_6563A' else '#7a9cc6'
        ax.hist(v, bins=edges, color=c, alpha=0.85)
        q16, q50, q84 = _q(v)
        ax.axvline(q50, color='black', lw=1.5)
        ax.axvline(c_med, color='0.4', ls='--', lw=1)
        name = 'H$\\alpha$/H$\\beta$' if lab == 'H1_6563A' else f'{lab}/H$\\beta$'
        ax.set_title(f'{name}: {q50:.2f}$^{{+{q84-q50:.2f}}}_{{-{q50-q16:.2f}}}$',
                     fontsize=9, fontweight='bold' if lab == 'H1_6563A' else None)
    for ax in axs.flat[len(cols):]:
        ax.set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel('E(B-V)')
    fig.suptitle(f"{res['gal_id']} - {LAW_LABEL[law]} - dashed: all-line fit "
                 f'{c_med:.3f}', fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
 
 
def plot_ebv_vs_wavelength(res, save_path=None):
    """(4) E(B-V) implied by each line vs wavelength, both laws. With the
    right law all lines give the same E(B-V) (flat)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for n, law in enumerate(res['per_line']):
        per = res['per_line'][law]
        wl = np.array([rest_wavelength(c, res['linelist_dict']) for c in per])
        q = np.array([_q(per[c].replace([np.inf, -np.inf], np.nan))
                      for c in per])
        off = (n - 0.5) * 15
        ax.errorbar(wl + off, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]],
                    fmt=LAW_MARKER[law], color=LAW_COLOR[law], ms=6, capsize=2,
                    label=LAW_LABEL[law])
        c16, c50, c84 = _q(res['combined'][law])
        ax.axhspan(c16, c84, color=LAW_COLOR[law], alpha=0.12)
        ax.axhline(c50, color=LAW_COLOR[law], lw=1, ls='--')
    ax.axhline(0, color='0.6', lw=0.8)
    ax.set_xlabel(r'Rest wavelength ($\AA$)')
    ax.set_ylabel(r'E(B-V) from line / H$\beta$')
    ax.set_title(f"{res['gal_id']}: band = all-line fit (16-84%)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
 
 
def plot_decrement_curve(res, save_path=None):
    """(4)+(5) Observed vs theoretical decrement: top, observed
    (F/F_Hb)/(R/R_Hb) per line with the model of each law; bottom, obs/model.
    Shaded = wavelength range constrained by the H I lines."""
    ld = res['linelist_dict']
    lines = res['lines']
    F, R = res['F'], res['R']
    wl = np.array([rest_wavelength(l, ld) for l in lines])
    obs = pd.DataFrame({l: (F[l] / F['H1_4861A']) / R[l] for l in lines})
    q = np.array([_q(obs[l]) for l in lines])
 
    lo, hi = res['spec_range'] or (wl.min() - 300, wl.max() + 1500)
    grid = np.linspace(lo, hi, 600)
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1.3],
                                               'hspace': 0.05})
    for a in (ax, axr):
        a.axvspan(wl.min(), wl.max(), color='0.92', zorder=0)
 
    ax.errorbar(wl, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]],
                fmt='o', color='black', ms=6, capsize=2, zorder=3,
                label='Observed')
    k_hb = {law: k_law(rest_wavelength('H1_4861A', ld), law)[0]
            for law in res['combined']}
    for law in res['combined']:
        E = res['combined'][law].dropna().to_numpy()
        curves = 10 ** (-0.4 * E[:, None] * (k_law(grid, law) - k_hb[law]))
        c16, c50, c84 = np.percentile(curves, [16, 50, 84], axis=0)
        ax.plot(grid, c50, color=LAW_COLOR[law], lw=2, label=LAW_LABEL[law])
        ax.fill_between(grid, c16, c84, color=LAW_COLOR[law], alpha=0.2)
        model_at = 10 ** (-0.4 * np.median(E) * (k_law(wl, law) - k_hb[law]))
        axr.errorbar(wl, q[:, 1] / model_at,
                     yerr=[(q[:, 1] - q[:, 0]) / model_at,
                           (q[:, 2] - q[:, 1]) / model_at],
                     fmt=LAW_MARKER[law], color=LAW_COLOR[law], ms=5, capsize=2)
    ax.axhline(1, color='0.6', ls=':', lw=1)
    axr.axhline(1, color='0.6', ls=':', lw=1)
    ax.set_ylabel(r'(F/F$_{\rm H\beta}$) / (F/F$_{\rm H\beta}$)$_{\rm Case\,B}$')
    axr.set_ylabel('obs / model')
    axr.set_xlabel(r'Rest wavelength ($\AA$)')
    ax.set_title(f"{res['gal_id']}: grey = range covered by H I lines")
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.09)
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
 
 
def plot_correction_factor(res, diag_lines=DIAGNOSTIC_LINES, save_path=None):
    """(5)+(6) Correction factor 10^(0.4 E k) across the spectrum for both
    laws (top) and their ratio (bottom). Grey = constrained by H I lines;
    everything outside is extrapolation. Ticks mark diagnostic lines."""
    ld = res['linelist_dict']
    wl_h = np.array([rest_wavelength(l, ld) for l in res['lines']])
    lo, hi = res['spec_range'] or (wl_h.min() - 300, wl_h.max() + 1500)
    grid = np.linspace(lo, hi, 600)
 
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1.3],
                                               'hspace': 0.05})
    for a in (ax, axr):
        a.axvspan(wl_h.min(), wl_h.max(), color='0.92', zorder=0)
 
    med = {}
    for law in res['combined']:
        E = res['combined'][law].clip(lower=0).dropna().to_numpy()
        fac = correction_factor(grid[None, :], E[:, None], law)
        c16, c50, c84 = np.percentile(fac, [16, 50, 84], axis=0)
        med[law] = c50
        ax.plot(grid, c50, color=LAW_COLOR[law], lw=2, label=LAW_LABEL[law])
        ax.fill_between(grid, c16, c84, color=LAW_COLOR[law], alpha=0.2)
 
    if {'ccm', 'calzetti'} <= set(med):
        axr.plot(grid, med['ccm'] / med['calzetti'], color='black', lw=1.5)
        axr.axhline(1, color='0.6', ls=':', lw=1)
        axr.set_ylabel('Cardelli / Calzetti')
 
    for lab in diag_lines:
        w = rest_wavelength(lab, ld) if (ld is None or lab in ld) else None
        if w is None or not (lo <= w <= hi):
            continue
        ax.axvline(w, color='0.5', lw=0.6, ls=':')
        ax.text(w, 0.98, lab, rotation=90, ha='right', va='top', fontsize=7,
                color='0.3', transform=ax.get_xaxis_transform())
    ax.set_ylabel(r'Correction factor $10^{0.4\,E(B-V)\,k(\lambda)}$')
    axr.set_xlabel(r'Rest wavelength ($\AA$)')
    ax.set_title(f"{res['gal_id']}: grey = constrained, white = extrapolated")
    ax.legend(frameon=False, fontsize=9, loc='lower left')
    fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.09)
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
 
 
def extrapolation_table(res, lines=DIAGNOSTIC_LINES):
    """(6) For each diagnostic line: inside the H I range or how far outside,
    and its correction factor under each law."""
    ld = res['linelist_dict']
    wl_h = np.array([rest_wavelength(l, ld) for l in res['lines']])
    rows = []
    for lab in lines:
        if ld is not None and lab not in ld:
            continue
        w = rest_wavelength(lab, ld)
        row = {'line': lab, 'rest_wl': w,
               'outside_by_A': max(wl_h.min() - w, w - wl_h.max(), 0)}
        for law in res['combined']:
            E = res['combined'][law].clip(lower=0)
            f = correction_factor(w, E.to_numpy(), law)
            row[f'factor_{law}'] = np.nanmedian(f)
            row[f'factor_{law}_err'] = (np.nanpercentile(f, 84)
                                        - np.nanpercentile(f, 16)) / 2
        if {'ccm', 'calzetti'} <= set(res['combined']):
            row['ccm_over_calzetti'] = row['factor_ccm'] / row['factor_calzetti']
        rows.append(row)
    return pd.DataFrame(rows).set_index('line')
 
 
def make_all_plots(res, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    g = res['gal_id']
    for law in res['per_line']:
        plot_ebv_per_line(res, law, save_path=f'{plot_dir}/{g}_ebv_per_line_{law}.png')
    plot_ebv_vs_wavelength(res, save_path=f'{plot_dir}/{g}_ebv_vs_wl.png')
    plot_decrement_curve(res, save_path=f'{plot_dir}/{g}_decrement_curve.png')
    plot_correction_factor(res, save_path=f'{plot_dir}/{g}_correction_factor.png')
    plot_relative_k(res, 'ebv', save_path=f'{plot_dir}/{g}_relative_k.png')
    plot_relative_k(res, 'halpha', save_path=f'{plot_dir}/{g}_relative_k_norm.png')
    tab = extrapolation_table(res)
    tab.to_csv(f'{plot_dir}/{g}_extrapolation.csv')
    return tab
 
 
def plot_relative_k(res, normalize='ebv', save_path=None):
    """
    Relative attenuation curve k(lambda) - k(H-beta): theory vs observed.
 
    Observed points come from each H I decrement:
        k_j - k_Hb = -2.5 log10[(F_j/F_Hb)/R_j] / E(B-V)
 
    normalize='ebv'    : uses each law's median all-line E(B-V), so each law
                         has its own observed points (they depend on E).
    normalize='halpha' : divides by the H-alpha value instead,
                         (k - k_Hb)/(k_Ha - k_Hb). Observed points then need
                         NO E(B-V) and NO law: a purely empirical curve shape
                         to compare against the theoretical shapes.
    """
    ld = res['linelist_dict']
    lines, F, R = res['lines'], res['F'], res['R']
    wl = np.array([rest_wavelength(l, ld) for l in lines])
    wl_hb = rest_wavelength('H1_4861A', ld)
    wl_ha = rest_wavelength('H1_6563A', ld)
    with np.errstate(divide='ignore', invalid='ignore'):
        logr = pd.DataFrame({l: np.log10((F[l] / F['H1_4861A']) / R[l])
                             for l in lines})
 
    lo, hi = res['spec_range'] or (wl.min() - 300, wl.max() + 1500)
    grid = np.linspace(lo, hi, 600)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axvspan(wl.min(), wl.max(), color='0.92', zorder=0)
 
    if normalize == 'halpha':
        obs = logr.div(logr['H1_6563A'], axis=0)
        q = np.array([_q(obs[l].replace([np.inf, -np.inf], np.nan)) for l in lines])
        ax.errorbar(wl, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]],
                    fmt='o', color='black', ms=6, capsize=2, zorder=3,
                    label='Observed (no E(B-V) needed)')
        for law in res['combined']:
            kd = (k_law(grid, law) - k_law(wl_hb, law)[0]) / \
                 (k_law(wl_ha, law)[0] - k_law(wl_hb, law)[0])
            ax.plot(grid, kd, color=LAW_COLOR[law], lw=2, label=LAW_LABEL[law])
        ylab = r'$[k(\lambda)-k({\rm H\beta})]\,/\,[k({\rm H\alpha})-k({\rm H\beta})]$'
    else:
        for n, law in enumerate(res['combined']):
            E = res['combined'][law].median()
            kd = k_law(grid, law) - k_law(wl_hb, law)[0]
            ax.plot(grid, kd, color=LAW_COLOR[law], lw=2, label=LAW_LABEL[law])
            obs = -2.5 * logr / E
            q = np.array([_q(obs[l].replace([np.inf, -np.inf], np.nan)) for l in lines])
            off = (n - 0.5) * 15
            ax.errorbar(wl + off, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]],
                        fmt=LAW_MARKER[law], mfc='white', color=LAW_COLOR[law],
                        ms=6, capsize=2, zorder=3,
                        label=f'Observed, E(B-V)={E:.3f}')
        ylab = r'$k(\lambda) - k({\rm H\beta})$'

    ax.axhline(0, color='0.6', ls=':', lw=1)
    ax.set_xlabel(r'Rest wavelength ($\AA$)')
    ax.set_ylabel(ylab)
    ax.set_title(f"{res['gal_id']}: grey = range covered by H I lines")
    ax.legend(frameon=False, fontsize=9,
              loc='lower right' if normalize == 'halpha' else 'upper right')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_alam_av(res, laws=None, extend=(0.3, 3.4), save_path=None):
    """
    Cardelli, Clayton & Mathis (1989, Fig. 4)-style plot: A_lambda/A_V vs
    1/lambda, computed curve (solid) vs observed H I points, with an
    obs - computed inset.
 
    The Balmer decrement only measures k(lambda) - k(H-beta); the absolute
    A_lambda/A_V needs R_V, so each observed point is anchored to the law:
        A_lambda/A_V = [k_obs(lambda) - k(H-beta) + k_law(H-beta)] / R_V
    with k_obs(lambda) - k(H-beta) = -2.5 log10[(F/F_Hb)/R_caseB] / E(B-V).
    """
    laws = laws or list(res['combined'])
    ld = res['linelist_dict']
    lines, F, R = res['lines'], res['F'], res['R']
    wl = np.array([rest_wavelength(l, ld) for l in lines])
    x_obs = 1e4 / wl
    wl_hb = rest_wavelength('H1_4861A', ld)
    with np.errstate(divide='ignore', invalid='ignore'):
        logr = pd.DataFrame({l: np.log10((F[l] / F['H1_4861A']) / R[l])
                             for l in lines})
 
    # curve over the requested 1/lambda range, clipped to each law's validity
    x_grid = np.linspace(extend[0], extend[1], 600)
    wl_h = (wl.min(), wl.max())
    sr = res['spec_range']
 
    fig, ax = plt.subplots(figsize=(7, 6))
    inset = ax.inset_axes([0.42, 0.06, 0.55, 0.22])
    ax.axvspan(1e4 / wl_h[1], 1e4 / wl_h[0], color='0.92', zorder=0)
    if sr:
        for a in (ax,):
            a.axvline(1e4 / sr[0], color='0.6', lw=0.8, ls=':')
            a.axvline(1e4 / sr[1], color='0.6', lw=0.8, ls=':')
 
    for law in laws:
        rv = RV[law]
        E = res['combined'][law].median()
        valid = (x_grid >= (0.45 if law == 'calzetti' else 0.3)) & \
                (x_grid <= (8.0 if law == 'ccm' else 1e4 / 1200))
        ax.plot(x_grid[valid], k_law(1e4 / x_grid[valid], law) / rv,
                color=LAW_COLOR[law], lw=2,
                label=f'Computed: {LAW_LABEL[law]}')
 
        k_hb = k_law(wl_hb, law)[0]
        obs = (-2.5 * logr / E + k_hb) / rv
        q = np.array([_q(obs[l].replace([np.inf, -np.inf], np.nan)) for l in lines])
        ax.errorbar(x_obs, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]],
                    fmt=LAW_MARKER[law], mfc='white', color=LAW_COLOR[law],
                    ms=6, capsize=2, zorder=3,
                    label=f'Observed H I, E(B-V)={E:.3f}')
 
        model = k_law(wl, law) / rv
        inset.errorbar(x_obs, q[:, 1] - model,
                       yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]],
                       fmt=LAW_MARKER[law], mfc='white', color=LAW_COLOR[law],
                       ms=4, capsize=1.5)
 
    # reference band: V (5500 A) and B (4400 A)
    for name, w in (('V', 5500), ('B', 4400)):
        ax.annotate(name, (1e4 / w, k_law(w, laws[0])[0] / RV[laws[0]]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9)
        ax.plot(1e4 / w, k_law(w, laws[0])[0] / RV[laws[0]], 'k+', ms=8)
 
    inset.axhline(0, color='0.5', lw=0.8)
    inset.set_xlim(ax.get_xlim() if False else (1e4 / wl_h[1] - 0.15,
                                                 1e4 / wl_h[0] + 0.15))
    inset.set_ylabel('obs $-$ comp', fontsize=8)
    inset.tick_params(labelsize=7)
    ax.set_xlabel(r'$1/\lambda$ ($\mu$m$^{-1}$)')
    ax.set_ylabel(r'$A_\lambda / A_V$')
    ax.set_title(f"{res['gal_id']}: grey = range covered by H I lines")
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
