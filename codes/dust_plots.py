"""
Plots for the dust correction and the comparison of laws.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dust_laws import RV, LAW_LABEL, LAW_COLOR, LAW_MARKER, k_law, ebv_scale
from dust_data import (rest_wavelength, flux_table, summary_stats,
                       NEBULAR, AURORAL)
from dust_fit import law_summary_table

# ================================================================
# Helpers
# ================================================================

def _q(v):
    v = pd.Series(v).dropna()
    return v.quantile(0.16), v.median(), v.quantile(0.84)


def _log_decrement(res):
    """Rest wavelengths of the H I lines and, per realization,
    log10[(F/F_Hb) / (F/F_Hb)_CaseB] (DataFrame: realizations x lines)."""
    ld = res['linelist_dict']
    F, R = res['F'], res['R']
    wl = []
    logr = {}
    with np.errstate(divide='ignore', invalid='ignore'):
        for l in res['lines']:
            wl.append(rest_wavelength(l, ld))
            v = np.log10((F[l] / F['H1_4861A']) / R[l])
            logr[l] = v.replace([np.inf, -np.inf], np.nan)
    return np.array(wl), pd.DataFrame(logr)


def _quantiles(table, lines):
    """16, 50, 84 percentiles of each column, as an array (lines x 3)."""
    q = []
    for l in lines:
        q.append(_q(table[l]))
    return np.array(q)


def _inv_forward(wl):
    with np.errstate(divide='ignore'):
        return -1.0 / np.asarray(wl, dtype=float)


def _inv_backward(v):
    with np.errstate(divide='ignore'):
        return -1.0 / np.asarray(v, dtype=float)


def set_wavelength_scale(ax, xscale='linear'):
    """'linear', 'log', or 'inverse' (spaced like 1/lambda, still labelled
    in Angstrom). 'inverse' spreads the blue Balmer lines the most."""
    if xscale == 'linear':
        return
    if xscale == 'log':
        ax.set_xscale('log')
    elif xscale == 'inverse':
        ax.set_xscale('function', functions=(_inv_forward, _inv_backward))
    ticks = [3500, 4000, 4500, 5000, 6000, 7000, 8000]
    labels = []
    for t in ticks:
        labels.append(str(t))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.minorticks_off()


def dodge(wl, n, n_laws, width=0.004):
    """Shift the points of law n sideways by a small fraction of the
    wavelength, so the laws do not sit on top of each other."""
    centre = (n_laws - 1) / 2
    return np.asarray(wl) * (1 + (n - centre) * width)


# ================================================================
# E(B-V) and the attenuation curve
# ================================================================

def plot_ebv_histograms(res, law, bins=30, save_path=None):
    """
    Left : E(B-V) from the all-line fit, one value per realization (the measurement).
    Right: E(B-V) from each line ratio separately (the check).
    """
    # line columns, ordered by wavelength
    lines = []
    wavelengths = []
    # always in E(B-V) (power/linear laws are fitted per A_Hb)
    ebv_table = res['per_line'][law] * ebv_scale(law)
    combined = res['combined'][law] * ebv_scale(law)

    for col in ebv_table.columns:
        if col != 'i':
            lines.append(col)
            wavelengths.append(float(col.split('_')[1].rstrip('A')))
    order = np.argsort(wavelengths)
    sorted_lines = []
    for n in order:
        sorted_lines.append(lines[n])
    lines = sorted_lines

    # combined values, without NaN
    comb = combined.dropna().to_numpy()

    # common bins covering both panels
    all_values = list(comb)
    for l in lines:
        for v in ebv_table[l]:
            if np.isfinite(v):
                all_values.append(v)
    edges = np.histogram_bin_edges(all_values, bins=bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ---- 1) the measurement: all-line fit ----
    med = np.median(comb)
    p16 = np.percentile(comb, 16)
    p84 = np.percentile(comb, 84)
    ax1.hist(comb, bins=bins, color='0.6')
    ax1.axvline(med, color='black', lw=2,
                label=f'median = {med:.3f}$^{{+{p84 - med:.3f}}}_{{-{med - p16:.3f}}}$')
    ax1.axvspan(p16, p84, color='0.3', alpha=0.15)
    ax1.set_title(f'All-line fit ({len(comb)} realizations)')
    ax1.set_xlabel('E(B-V)')
    ax1.set_ylabel('Number of realizations')
    # ax1.set_xlim(comb.min(), comb.max())
    ax1.legend(frameon=False, fontsize=9)

    # ---- 2) each line ratio separately ----
    for n in range(len(lines)):
        l = lines[n]
        v = ebv_table[l].to_numpy()
        v = v[np.isfinite(v)]
        ax2.hist(v, lw=2, bins=edges,
                 label=f'{l}: {np.median(v):.3f}', alpha=0.3)
    ax2.axvline(med, color='black', lw=2, ls='--', label=f'all-line fit = {med:.3f}')
    ax2.set_title('Each line ratio separately')
    ax2.set_xlabel('E(B-V)')
    ax2.legend(frameon=False, fontsize=8)

    fig.suptitle(f"{res['gal_id']} - {LAW_LABEL[law]}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_ebv_vs_wavelength(res, laws=None, xscale='linear', save_path=None):
    """(4) E(B-V) implied by each line vs wavelength, one colour per law.
    With the right law all lines give the same E(B-V) (flat).
    Dashed line = the E(B-V) each law adopts (all-line fit; H-alpha only
    for the hybrid)."""
    if laws is None:
        laws = list(res['per_line'])
    ld = res['linelist_dict']

    fig, ax = plt.subplots(figsize=(9, 5))
    for n in range(len(laws)):
        law = laws[n]
        scale = ebv_scale(law)          # 1, or A_Hb -> E(B-V) for power laws
        per = res['per_line'][law] * scale
        wl = []
        med = []
        lo = []
        hi = []
        for c in per.columns:
            q16, q50, q84 = _q(per[c].replace([np.inf, -np.inf], np.nan))
            wl.append(rest_wavelength(c, ld))
            med.append(q50)
            lo.append(q50 - q16)
            hi.append(q84 - q50)
        ax.errorbar(dodge(wl, n, len(laws)), med, yerr=[lo, hi],
                    fmt=LAW_MARKER[law], color=LAW_COLOR[law], ms=6,
                    capsize=2, label=LAW_LABEL[law])
        ax.axhline(res['combined'][law].median() * scale,
                   color=LAW_COLOR[law], lw=1, ls='--')

    set_wavelength_scale(ax, xscale)
    ax.set_xlabel(r'Rest wavelength ($\AA$)', fontsize=18)
    ax.set_ylabel(r'E(B-V) from line / H$\beta$', fontsize=18)
    ax.set_title(f"{res['gal_id']}: dashed = adopted E(B-V)", fontsize=18)
    ax.legend(frameon=True, fontsize=10, ncol=2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_decrement_curve(res, laws=None, xscale='linear', ylog=False,
                         save_path=None):
    """(4)+(5) Observed vs theoretical decrement: top, observed
    (F/F_Hb)/(R/R_Hb) per line with the model of each law; bottom, obs/model.
    Shaded = wavelength range constrained by the H I lines.
    ylog=True: log y on the top panel, so 0.5 and 2 are equally far from 1."""
    if laws is None:
        laws = list(res['combined'])
    ld = res['linelist_dict']
    lines = res['lines']
    F, R = res['F'], res['R']

    wl = []
    q = []
    for l in lines:
        wl.append(rest_wavelength(l, ld))
        q.append(_q((F[l] / F['H1_4861A']) / R[l]))
    wl = np.array(wl)
    q = np.array(q)

    lo, hi = wl.min() - 300, wl.max() + 1500
    grid = np.linspace(lo, hi, 600)
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1.3],
                                               'hspace': 0.05})
    # for a in (ax, axr):
    #     a.axvspan(wl.min(), wl.max(), color='0.92', zorder=0)

    ax.errorbar(wl, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]],
                fmt='o', color='black', ms=6, capsize=2, zorder=3,
                label='Observed')
    wl_hb = rest_wavelength('H1_4861A', ld)
    for n in range(len(laws)):
        law = laws[n]
        E = res['combined'][law].median()
        k_hb = k_law(wl_hb, law)[0]
        curve = 10 ** (-0.4 * E * (k_law(grid, law) - k_hb))
        ax.plot(grid, curve, color=LAW_COLOR[law], lw=2, label=LAW_LABEL[law])
        model_at = 10 ** (-0.4 * E * (k_law(wl, law) - k_hb))
        axr.errorbar(dodge(wl, n, len(laws)), q[:, 1] / model_at,
                     yerr=[(q[:, 1] - q[:, 0]) / model_at,
                           (q[:, 2] - q[:, 1]) / model_at],
                     fmt=LAW_MARKER[law], color=LAW_COLOR[law], ms=5,
                     capsize=2)
    if ylog:
        ax.set_yscale('log')
    set_wavelength_scale(axr, xscale)
    ax.axhline(1, color='0.6', ls=':', lw=1)
    axr.axhline(1, color='0.6', ls=':', lw=1)
    ax.set_ylabel(r'(F/F$_{\rm H\beta}$) / (F/F$_{\rm H\beta}$)$_{\rm Case\,B}$',
                  fontsize=18)
    axr.set_ylabel('obs / model', fontsize=18)
    axr.set_xlabel(r'Rest wavelength ($\AA$)', fontsize=18)
    ax.set_title(f"{res['gal_id']}: grey = range covered by H I lines",
                 fontsize=18)
    ax.legend(frameon=False, fontsize=10, loc='upper left', ncol=2)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.09)
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_relative_k(res, laws=None, xscale='linear', save_path=None):
    """
    Relative attenuation curve k(lambda) - k(H-beta): theory vs observed.
    Observed points come from each H I decrement:
        k_j - k_Hb = -2.5 log10[(F_j/F_Hb)/R_j] / E
    with E each law's adopted value, so each law has its own points.
    (For the power and linear laws k and E are per A_Hb instead of per
    E(B-V); the points and curve are consistent either way.)
    """
    if laws is None:
        laws = list(res['combined'])
    ld = res['linelist_dict']
    lines = res['lines']
    wl, logr = _log_decrement(res)
    wl_hb = rest_wavelength('H1_4861A', ld)

    lo, hi = wl.min() - 300, wl.max() + 1500
    grid = np.linspace(lo, hi, 600)
    fig, ax = plt.subplots(figsize=(9, 5))
    # ax.axvspan(wl.min(), wl.max(), color='0.92', zorder=0)

    for n in range(len(laws)):
        law = laws[n]
        E = res['combined'][law].median()
        kd = k_law(grid, law) - k_law(wl_hb, law)[0]
        ax.plot(grid, kd, color=LAW_COLOR[law], lw=2, label=LAW_LABEL[law])
        q = _quantiles(-2.5 * logr / E, lines)
        ax.errorbar(dodge(wl, n, len(laws)), q[:, 1],
                    yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]],
                    fmt=LAW_MARKER[law], mfc='white', color=LAW_COLOR[law],
                    ms=6, capsize=2, zorder=3,
                    label=f'Obs., E(B-V)={E * ebv_scale(law):.3f}')

    set_wavelength_scale(ax, xscale)
    ax.axhline(0, color='0.6', ls=':', lw=1)
    ax.set_xlabel(r'Rest wavelength ($\AA$)', fontsize=18)
    ax.set_ylabel(r'$k(\lambda) - k({\rm H\beta})$', fontsize=18)
    ax.set_title(f"{res['gal_id']}: grey = range covered by H I lines",
                 fontsize=18)
    ax.legend(frameon=False, fontsize=9, loc='upper right', ncol=2)
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
    if laws is None:
        laws = list(res['combined'])
    ld = res['linelist_dict']
    lines = res['lines']
    wl, logr = _log_decrement(res)
    x_obs = 1e4 / wl
    wl_hb = rest_wavelength('H1_4861A', ld)

    # curve over the requested 1/lambda range, clipped to each law's validity
    x_grid = np.linspace(extend[0], extend[1], 600)
    wl_h = (wl.min(), wl.max())
    sr = wl.min() - 300, wl.max() + 1500

    fig, ax = plt.subplots(figsize=(9, 6.5))
    inset = ax.inset_axes([0.42, 0.06, 0.55, 0.22])
    # ax.axvspan(1e4 / wl_h[1], 1e4 / wl_h[0], color='0.92', zorder=0)
    if sr:
        for a in (ax,):
            a.axvline(1e4 / sr[0], color='0.6', lw=0.8, ls=':')
            a.axvline(1e4 / sr[1], color='0.6', lw=0.8, ls=':')

    for n in range(len(laws)):
        law = laws[n]
        rv = k_law(5500.0, law)[0]     # A_lambda/A_V = k(lambda)/k(V)
        E = res['combined'][law].median()
        x_n = dodge(x_obs, n, len(laws), width=0.006)
        # Calzetti is defined only from 2.2 micron (x = 0.45) to 1200 A
        x_min = 0.3
        if law.startswith('calzetti'):
            x_min = 0.45
        valid = (x_grid >= x_min) & (x_grid <= 1e4 / 1200)
        ax.plot(x_grid[valid], k_law(1e4 / x_grid[valid], law) / rv,
                color=LAW_COLOR[law], lw=2,
                label=f'Computed: {LAW_LABEL[law]}')

        k_hb = k_law(wl_hb, law)[0]
        obs = (-2.5 * logr / E + k_hb) / rv
        q = _quantiles(obs, lines)
        ax.errorbar(x_n, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]],
                    fmt=LAW_MARKER[law], mfc='white', color=LAW_COLOR[law],
                    ms=6, capsize=2, zorder=3,
                    label=f'Obs., E(B-V)={E * ebv_scale(law):.3f}')

        model = k_law(wl, law) / rv
        inset.errorbar(x_n, q[:, 1] - model,
                       yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]],
                       fmt=LAW_MARKER[law], mfc='white', color=LAW_COLOR[law],
                       ms=4, capsize=1.5)

    # reference band: V (5500 A) and B (4400 A)
    for name, w in (('V', 5500), ('B', 4400)):
        ax.annotate(name, (1e4 / w, k_law(w, laws[0])[0] / RV[laws[0]]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=18)
        ax.plot(1e4 / w, k_law(w, laws[0])[0] / RV[laws[0]], 'k+', ms=8)

    inset.axhline(0, color='0.5', lw=0.8)
    inset.set_xlim(ax.get_xlim() if False else (1e4 / wl_h[1] - 0.15,
                                                 1e4 / wl_h[0] + 0.15))
    inset.set_ylabel('obs $-$ comp', fontsize=18)
    inset.tick_params(labelsize=7)
    ax.set_xlabel(r'$1/\lambda$ ($\mu$m$^{-1}$)', fontsize=18)
    ax.set_ylabel(r'$A_\lambda / A_V$', fontsize=18)
    ax.set_title(f"{res['gal_id']}: grey = range covered by H I lines",
                 fontsize=18)
    ax.legend(frameon=False, fontsize=8, loc='upper left', ncol=2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def _draw_attenuation(ax, axr, res, laws, linthresh, xscale, labels=True):
    """Observed A_lambda - A_Hb (one set of points) and each law's curve on
    ax; observed - law (symlog) on axr. See plot_attenuation_curves."""
    ld = res['linelist_dict']
    lines = res['lines']
    wl, logr = _log_decrement(res)
    wl_hb = rest_wavelength('H1_4861A', ld)

    # observed: one value per realization, then median and 16-84
    q = _quantiles(-2.5 * logr, lines)

    lo, hi = wl.min() - 300, wl.max() + 1500
    grid = np.linspace(lo, hi, 600)
    # for a in (ax, axr):
    #     a.axvspan(wl.min(), wl.max(), color='0.92', zorder=0)

    biggest = linthresh                  # largest |residual|, for the y range
    for n in range(len(laws)):
        law = laws[n]
        E = res['combined'][law].median()        # median of the realizations
        k_hb = k_law(wl_hb, law)[0]
        curve = E * (k_law(grid, law) - k_hb)
        model_at = E * (k_law(wl, law) - k_hb)
        for r in (q[:, 2] - model_at, q[:, 0] - model_at):
            biggest = max(biggest, np.nanmax(np.abs(r)))

        label = LAW_LABEL[law]
        if not labels:
            label = None
        ax.plot(grid, curve, color=LAW_COLOR[law], lw=2, label=label)
        axr.errorbar(dodge(wl, n, len(laws)), q[:, 1] - model_at,
                     yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]],
                     fmt=LAW_MARKER[law], color=LAW_COLOR[law], ms=5,
                     capsize=2)

    ax.errorbar(wl, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]],
                fmt='o', color='black', ms=7, capsize=2, zorder=5,
                label='Observed')
    ax.axhline(0, color='0.6', ls=':', lw=1)
    axr.axhline(0, color='0.6', ls=':', lw=1)

    # symmetric log: linear for |residual| < linthresh, log outside
    # axr.set_yscale('line')
    axr.axhspan(-linthresh, linthresh, color='0.85', alpha=0.4, zorder=0)
    axr.set_ylim(-1.5 * biggest, 1.5 * biggest)
    # ticks = []
    # tick_labels = []
    # # for t in (-1, -0.3, -0.1, -0.03, -0.01, 0, 0.01, 0.03, 0.1, 0.3, 1):
    # #     if abs(t) <= 1.5 * biggest and (t == 0 or abs(t) >= linthresh / 2):
    # #         ticks.append(t)
    # #         tick_labels.append(f'{t:g}')
    # axr.set_yticks(ticks)
    # axr.set_yticklabels(tick_labels)
    axr.minorticks_off()
    set_wavelength_scale(axr, xscale)

    ax.set_ylabel(r'$A_\lambda - A_{\rm H\beta}$ (mag)', fontsize=16)
    axr.set_ylabel('obs $-$ law\n(mag)', fontsize=12)
    axr.set_xlabel(r'Rest wavelength ($\AA$)', fontsize=16)


def _draw_ranking(ax1, ax2, ax3, table, show_labels=True):
    """Delta BIC, reduced chi^2 and E(B-V) per law, best at the top.
    See plot_law_ranking."""
    laws = list(table.index)              # already sorted by BIC
    y = np.arange(len(laws))[::-1]        # best at the top
    for n in range(len(laws)):
        law = laws[n]
        color = LAW_COLOR[law]
        dbic = table.loc[law, 'dBIC']
        chi2nu = table.loc[law, 'chi2_nu']
        ax1.barh(y[n], max(dbic, 0.05), color=color)
        ax1.text(max(dbic, 0.05) * 1.05 + 0.1, y[n], f'{dbic:.0f}',
                 va='center', fontsize=9)
        ax2.barh(y[n], chi2nu, color=color)
        ax2.text(chi2nu * 1.08, y[n], f'{chi2nu:.1f}', va='center', fontsize=9)
        ax3.errorbar(table.loc[law, 'E_BV'], y[n],
                     xerr=[[table.loc[law, 'err_lo']],
                           [table.loc[law, 'err_hi']]],
                     fmt=LAW_MARKER[law], color=color, ms=7, capsize=3)

    # Delta BIC: symmetric log so 0-10 and 10-100 both show
    ax1.set_xscale('symlog', linthresh=2)
    xmax = max(10, 3 * table['dBIC'].max())
    ax1.set_xlim(0, xmax)
    ticks = []
    for t in (0, 2, 6, 10, 30, 100, 300, 1000):
        if t <= xmax:
            ticks.append(t)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(ticks)
    ax1.minorticks_off()
    for v in (2, 6):
        ax1.axvline(v, color='0.4', ls='--', lw=1)
    ax1.set_xlabel(r'$\Delta$BIC (0 = best)', fontsize=13)

    # reduced chi^2: log scale, 1 = fits within the errors
    ax2.set_xscale('log')
    ax2.set_xlim(0.5, 3 * max(2, table['chi2_nu'].max()))
    ax2.axvline(1, color='0.4', ls='--', lw=1)
    ax2.set_xlabel(r'$\chi^2_\nu$', fontsize=13)

    # E(B-V) of each law
    ax3.axvline(table['E_BV'].iloc[0], color='0.4', ls=':', lw=1)
    ax3.set_xlabel('E(B-V)', fontsize=13)

    ax1.set_yticks(y)
    if show_labels:
        labels = []
        for n in range(len(laws)):
            labels.append(LAW_LABEL[laws[n]])
        ax1.set_yticklabels(labels, fontsize=10)
    for a in (ax1, ax2, ax3):
        a.set_ylim(-0.6, len(laws) - 0.4)
        a.grid(axis='x', color='0.9', lw=0.8)


def plot_attenuation_curves(res, laws=None, xscale='linear', linthresh=0.02,
                            save_path=None):
    """
    One set of observed points and every law's curve, in magnitudes.
    Observed (no law, no E(B-V) needed), for each H I line:
        A_lambda - A_Hb = -2.5 log10[(F/F_Hb)_obs / (F/F_Hb)_CaseB]
    (median of the realizations, error bars 16-84).
    Curve of each law, with its median E:
        A_lambda - A_Hb = E * [k(lambda) - k(H-beta)]
    (for the power and linear laws E is A_Hb and k is per A_Hb; the
    product is the same attenuation). Bottom: observed - curve (mag), on a
    symmetric log scale: linear within +-linthresh, logarithmic outside.
    """
    if laws is None:
        laws = list(res['combined'])

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1.3],
                                               'hspace': 0.05})
    _draw_attenuation(ax, axr, res, laws, linthresh, xscale)
    ax.set_title(f"{res['gal_id']}: grey = range covered by H I lines",
                 fontsize=18)
    if len(laws) <= 4:
        ax.legend(frameon=False, fontsize=9, loc='upper right')
        fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.09)
    else:
        # many laws: legend outside, to the right of the plot
        fig.set_size_inches(13, 6.5)
        ax.legend(frameon=False, fontsize=9, loc='upper left',
                  bbox_to_anchor=(1.01, 1.0))
        fig.subplots_adjust(left=0.09, right=0.66, top=0.93, bottom=0.09)
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_law_ranking(table, gal_id='', save_path=None):
    """
    How well each law fits, from the law summary table (best at the top):
      left  : Delta BIC (symlog). < 2: as good as the best; 2-6: worse;
              > 6: clearly worse (dashed lines at 2 and 6)
      middle: reduced chi^2 (log). 1 = fits within the errors
      right : the E(B-V) each law gives (median, 16-84 error bars)
    """
    n_laws = len(table)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 0.45 * n_laws + 1.8),
                                        sharey=True,
                                        gridspec_kw={'width_ratios': [1.2, 1, 1]})
    _draw_ranking(ax1, ax2, ax3, table)
    fig.suptitle(f'{gal_id}: laws ranked by BIC (best at the top)', fontsize=15)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_attenuation_and_ranking(res, laws=None, table=None, xscale='linear',
                                 linthresh=0.02, save_path=None):
    """
    The attenuation-curve plot (left) next to the ranking of the same laws
    (right: Delta BIC, reduced chi^2, E(B-V)). The colours match, so the
    law names on the ranking work as the legend of the curves.
    """
    if laws is None:
        laws = list(res['combined'])
    if table is None:
        table = law_summary_table(res)
    # ranking only for the laws in this plot (Delta BIC within them)
    keep = []
    for law in table.index:
        if law in laws:
            keep.append(law)
    sub = table.loc[keep].copy()
    sub['dBIC'] = sub['BIC'] - sub['BIC'].min()

    fig = plt.figure(figsize=(21, max(6.5, 0.45 * len(keep) + 2.5)))
    gs = fig.add_gridspec(2, 4, width_ratios=[2.6, 1.1, 0.9, 0.9],
                          height_ratios=[3, 1.3], hspace=0.05, wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[1, 0], sharex=ax)
    ax1 = fig.add_subplot(gs[:, 1])
    ax2 = fig.add_subplot(gs[:, 2], sharey=ax1)
    ax3 = fig.add_subplot(gs[:, 3], sharey=ax1)

    _draw_attenuation(ax, axr, res, laws, linthresh, xscale, labels=False)
    ax.tick_params(labelbottom=False)
    ax.legend(frameon=False, fontsize=11, loc='upper right')   # 'Observed'
    _draw_ranking(ax1, ax2, ax3, sub)
    # the law names go on the right of the ranking, next to the bars
    ax1.yaxis.tick_right()
    ax1.tick_params(axis='y', labelright=False)
    ax3.yaxis.tick_right()
    ax3.tick_params(axis='y', labelright=True, labelleft=False)
    ax2.tick_params(axis='y', labelleft=False)
    labels = []
    for n in range(len(keep)):
        labels.append(LAW_LABEL[keep[n]])
    ax3.set_yticks(np.arange(len(keep))[::-1])
    ax3.set_yticklabels(labels, fontsize=10)

    fig.suptitle(f"{res['gal_id']}: attenuation curves (left) and laws "
                 'ranked by BIC (right, best at the top)', fontsize=16)
    fig.subplots_adjust(left=0.05, right=0.80, top=0.92, bottom=0.09)
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ================================================================
# Effect of the correction on the lines
# ================================================================

def plot_change_histograms(info_obs, info_corr, lines=None, relative=False,
                           ref='H1_4861A', bins=20, title='', save_path=None):
    """
    One histogram per line of the % change from the dust correction,
    using all MC realizations. The text in each panel gives the median
    % change and the significance.

    relative=False : % change of the flux itself
    relative=True  : % change of the flux relative to H-beta
    """
    if lines is None:
        lines = NEBULAR + AURORAL

    obs = flux_table(info_obs)
    corr = flux_table(info_corr)

    # keep only lines that exist, ordered by wavelength
    present = []
    wl = []
    for l in lines:
        if l in obs.columns:
            present.append(l)
            wl.append(float(l.split('_')[1].rstrip('A')))
    order = np.argsort(wl)
    lines = []
    for n in order:
        lines.append(present[n])

    ncol = 4
    nrow = int(np.ceil(len(lines) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.7 * nrow),
                            squeeze=False)
    axs = axs.flatten()

    for n in range(len(lines)):
        l = lines[n]
        ax = axs[n]

        # observed and corrected values for every realization
        if relative:
            v_obs = obs[l] / obs[ref]
            v_corr = corr[l] / corr[ref]
        else:
            v_obs = obs[l]
            v_corr = corr[l]

        # % change, realization by realization
        pct = 100 * (v_corr / v_obs - 1)
        pct = pct.to_numpy()
        pct = pct[np.isfinite(pct)]

        # median and 16-84 range of the % change
        p16 = np.percentile(pct, 16)
        p50 = np.percentile(pct, 50)
        p84 = np.percentile(pct, 84)

        # significance: corrected vs observed, using each one's spread
        o_med, o_err = summary_stats(v_obs)
        c_med, c_err = summary_stats(v_corr)
        sig = (c_med - o_med) / np.sqrt(o_err**2 + c_err**2)

        if l in AURORAL:
            color = '#e07b00'
        else:
            color = '#2a6fdb'

        ax.hist(pct, bins=bins, color=color, alpha=0.8)
        ax.axvline(p50, color='black', lw=1.5)
        ax.axvspan(p16, p84, color='black', alpha=0.1)

        text = (f'{p50:+.1f}$^{{+{p84 - p50:.1f}}}_{{-{p50 - p16:.1f}}}$ %\n'
                f'{sig:.1f}$\\sigma$')
        ax.text(0.97, 0.95, text, transform=ax.transAxes, ha='right', va='top',
                fontsize=9, bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        ax.set_title(l, fontsize=10)

    # hide unused panels
    for n in range(len(lines), len(axs)):
        axs[n].set_visible(False)

    if relative:
        xlabel = r'% change in flux / H$\beta$'
    else:
        xlabel = '% change in flux'
    for ax in axs[:len(lines)]:
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(labelsize=8)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ================================================================
# All plots for one galaxy
# ================================================================

def make_all_plots(res, plot_dir, laws=None):
    """Plots for one galaxy. laws: laws to show (default Cardelli, Calzetti
    and the law applied to the saved fluxes)."""
    os.makedirs(plot_dir, exist_ok=True)
    g = res['gal_id']
    if laws is None:
        laws = list(res['laws'])
        if res['apply_law'] is not None and res['apply_law'] not in laws:
            laws.append(res['apply_law'])

    for law in laws:
        safe = law.replace(':', '_')
        plot_ebv_histograms(res, law,
                            save_path=f'{plot_dir}/{g}_ebv_hist_{safe}.png')
    plot_ebv_vs_wavelength(res, laws, save_path=f'{plot_dir}/{g}_ebv_vs_wl.png')
    plot_decrement_curve(res, laws,
                         save_path=f'{plot_dir}/{g}_decrement_curve.png')
    plot_relative_k(res, laws, save_path=f'{plot_dir}/{g}_relative_k.png')
    plot_attenuation_curves(res, laws,
                            save_path=f'{plot_dir}/{g}_attenuation.png')
    plot_alam_av(res, laws, save_path=f'{plot_dir}/{g}_alam_av.png')

    # % change histograms only when a law was applied
    if res['corrected_info'] is not None:
        plot_change_histograms(res['info'], res['corrected_info'],
                               title=f'{g} - flux',
                               save_path=f'{plot_dir}/{g}_change_flux.png')
        plot_change_histograms(res['info'], res['corrected_info'],
                               relative=True,
                               title=f'{g} - relative to Hb',
                               save_path=f'{plot_dir}/{g}_change_rel.png')
    plt.close('all')


def make_law_comparison_plots(res, plot_dir, xscale='inverse'):
    """The three comparison plots for three groups of laws:
       shapes : Cardelli (R_V=3.1 and fitted), Calzetti, SMC, linear, power
       rv     : Cardelli and Calzetti with every R_V that was run
       hybrid : Cardelli and Calzetti vs the H-alpha-anchored hybrid
    Also returns and
    saves the table of E(B-V) and chi^2 per law (best law first)."""
    os.makedirs(plot_dir, exist_ok=True)
    g = res['gal_id']

    table = law_summary_table(res)
    table.to_csv(f'{plot_dir}/{g}_law_summary.csv')
    plot_law_ranking(table, g, save_path=f'{plot_dir}/{g}_law_ranking.png')

    shapes = []
    rv = []
    for law in res['laws']:
        name = law.split(':')[0]
        if name in ('ccm', 'calzetti'):
            rv.append(law)
        if law in ('ccm', 'calzetti', 'smc', 'linear') or name == 'power' \
                or law == res['ccm_fit_law'] or law == res['calz_fit_law']:
            shapes.append(law)
        if law == res['calz_fit_law']:
            rv.append(law)
    groups = {'shapes': shapes, 'rv': rv, 'all': list(res['laws'])}
    if res['hybrid_law'] is not None:
        groups['hybrid'] = ['ccm', 'calzetti', res['hybrid_law']]

    for name in groups:
        laws = groups[name]
        if len(laws) == 0:
            continue
        plot_ebv_vs_wavelength(res, laws, xscale,
                               save_path=f'{plot_dir}/{g}_ebv_vs_wl_{name}.png')
        plot_decrement_curve(res, laws, xscale,
                             save_path=f'{plot_dir}/{g}_decrement_{name}.png')
        plot_attenuation_curves(res, laws, xscale,
                                save_path=f'{plot_dir}/{g}_attenuation_{name}.png')
        plot_attenuation_and_ranking(
            res, laws, table, xscale,
            save_path=f'{plot_dir}/{g}_attenuation_ranking_{name}.png')
        plot_alam_av(res, laws,
                     save_path=f'{plot_dir}/{g}_alam_av_{name}.png')
        plt.close('all')

    print(table.round(3))
    return table