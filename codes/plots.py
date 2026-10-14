import glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from FINAL_LINES import add_derived
from GALAXIES import PROJ_DIR

C_H, C_M = '#2a6fdb', '#e07b00'
LABEL = {'ratio_H': 'H I', 'ratio_metal': 'metals'}


def load_all_info():
    frames = []
    for path in sorted(glob.glob(f'{PROJ_DIR}new_results/*/*_lines_info.csv')):
        d = pd.read_csv(path)
        d = d.drop(columns=[c for c in d.columns if c.startswith('Unnamed')])
        d['gal'] = os.path.basename(path)[:5]
        frames.append(d)
    return add_derived(pd.concat(frames, ignore_index=True))


def ratio_samples(all_info, col='b_to_n_ratio'):
    """One value per (galaxy, ratio type, realization)"""
    d = all_info[(all_info['nar_amplitude'] > 0) & ~all_info['ratio_on_bound']]
    d = d[~d['label'].str.startswith(('He1_', 'He2_'))]
    free = d.groupby(['gal', 'label'])['b_to_n_ratio'].transform('std') > 0
    d = d[free]
    return d.drop_duplicates(['gal', 'ratio_type', 'i', 'b_to_n_ratio'])


def plot_ratio_hists(rs, col='b_to_n_ratio', bins=30):
    gals = sorted(rs['gal'].unique())
    ncol = 3
    nrow = int(np.ceil(len(gals) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                            sharex=True, squeeze=False)
    edges = np.histogram_bin_edges(rs[col].dropna(), bins=bins)

    for ax, gal in zip(axs.flat, gals):
        g = rs[rs['gal'] == gal]
        for rtype, c in [('ratio_H', C_H), ('ratio_metal', C_M)]:
            v = g.loc[g['ratio_type'] == rtype, col]
            if v.empty:
                continue
            ax.hist(v, bins=edges, color=c, alpha=0.45,
                    label=f'{LABEL[rtype]}: {v.median():.2f}')
            ax.axvline(v.median(), color=c, lw=2)
        ax.set_title(gal, fontsize=11)
        ax.legend(fontsize=8, frameon=False)
    for ax in axs.flat[len(gals):]:
        ax.set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel({'b_to_n_ratio': 'Broad/narrow flux ratio',
                       'peak_ratio': 'Broad/narrow peak ratio',
                       'broad_frac': 'Broad flux fraction'}.get(col, col))
    fig.tight_layout()
    fig.savefig(f'{PROJ_DIR}new_results/ratios_histograms.png', format='png')
    fig.savefig(f'{PROJ_DIR}new_results/ratios_histograms.pdf', format='pdf')
    return fig


def plot_ratio_summary(rs, col='b_to_n_ratio'):
    stats = (rs.groupby(['gal', 'ratio_type'])[col]
               .quantile([0.16, 0.5, 0.84]).unstack())
    gals = sorted(rs['gal'].unique())
    y = np.arange(len(gals))

    fig, ax = plt.subplots(figsize=(6, 0.5 * len(gals) + 1.5))
    for rtype, c, dy in [('ratio_H', C_H, -0.12), ('ratio_metal', C_M, 0.12)]:
        s = stats.xs(rtype, level='ratio_type').reindex(gals)
        ax.errorbar(s[0.5], y + dy,
                    xerr=[s[0.5] - s[0.16], s[0.84] - s[0.5]],
                    fmt='o', color=c, ms=7, capsize=3, label=LABEL[rtype])
    ax.set_yticks(y, gals)
    ax.invert_yaxis()
    ax.set_xlabel('Broad/narrow ratio')
    ax.legend(frameon=False)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{PROJ_DIR}new_results/ratios_summary.png', format='png')
    fig.savefig(f'{PROJ_DIR}new_results/ratios_summary.pdf', format='pdf')
    return fig, stats


def plot_H_vs_metal(ratio_stats):
    s = ratio_stats.unstack('ratio_type')
    xH, xM = s[(0.5, 'ratio_H')], s[(0.5, 'ratio_metal')]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.errorbar(xH, xM,
                xerr=[xH - s[(0.16, 'ratio_H')], s[(0.84, 'ratio_H')] - xH],
                yerr=[xM - s[(0.16, 'ratio_metal')], s[(0.84, 'ratio_metal')] - xM],
                fmt='o', color='0.25', ms=7, capsize=3)
    for gal in s.index:
        ax.annotate(gal, (xH[gal], xM[gal]), xytext=(5, 5),
                    textcoords='offset points', fontsize=8)
    lim = [0, np.nanmax([xH.max(), xM.max()]) * 1.3]
    ax.plot(lim, lim, ls='--', color='0.6', lw=1, label='1:1')
    ax.set(xlim=lim, ylim=lim, xlabel='Broad/narrow ratio, H I',
           ylabel='Broad/narrow ratio, metals')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f'{PROJ_DIR}new_results/ratios_vs.png', format='png')
    fig.savefig(f'{PROJ_DIR}new_results/ratios_vs.pdf', format='pdf')
    return fig


def plot_broad_frac_heatmap(all_info, max_bound=0.5):
    d = all_info[all_info['nar_amplitude'] > 0]
    med = d.pivot_table(index='label', columns='gal', values='broad_frac',
                        aggfunc='median')
    bound = d.pivot_table(index='label', columns='gal', values='ratio_on_bound',
                          aggfunc='mean')
    med = med.where(bound <= max_bound)

    fig, ax = plt.subplots(figsize=(0.7 * med.shape[1] + 3, 0.35 * med.shape[0] + 1.5))
    im = ax.imshow(med.values, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(med.shape[1]), med.columns, rotation=45)
    ax.set_yticks(range(med.shape[0]), med.index)
    ax.set_facecolor('0.9')
    for (r, c), v in np.ndenumerate(med.values):
        if np.isfinite(v):
            ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if v > 0.6 else 'black')
    fig.colorbar(im, ax=ax, label='Broad flux fraction')
    fig.tight_layout()
    fig.savefig(f'{PROJ_DIR}new_results/ratios_heatmap.png', format='png')
    fig.savefig(f'{PROJ_DIR}new_results/ratios_heatmap.pdf', format='pdf')
    return fig


def plot_sigmas(all_info):
    d = (all_info[all_info['nar_amplitude'] > 0]
           .drop_duplicates(['gal', 'i', 'sigma_v_narrow', 'sigma_v_broad']))
    q = d.groupby('gal')[['sigma_v_narrow', 'sigma_v_broad']].quantile([0.16, 0.5, 0.84]).unstack()
    fig, ax = plt.subplots(figsize=(5, 5))
    xn, xb = q[('sigma_v_narrow', 0.5)], q[('sigma_v_broad', 0.5)]
    ax.errorbar(xn, xb,
                xerr=[xn - q[('sigma_v_narrow', 0.16)], q[('sigma_v_narrow', 0.84)] - xn],
                yerr=[xb - q[('sigma_v_broad', 0.16)], q[('sigma_v_broad', 0.84)] - xb],
                fmt='o', color='0.25', ms=7, capsize=3)
    for gal in q.index:
        ax.annotate(gal, (xn[gal], xb[gal]), xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set(xlabel=r'$\sigma_{\rm narrow}$ (km s$^{-1}$)', ylabel=r'$\sigma_{\rm broad}$ (km s$^{-1}$)')
    fig.tight_layout()
    fig.savefig(f'{PROJ_DIR}new_results/gal_sigmas.png', format='png')
    fig.savefig(f'{PROJ_DIR}new_results/gal_sigmas.pdf', format='pdf')
    return fig


def plots_em_lines_ratios():
    all_info = load_all_info()
    rs = ratio_samples(all_info)
    fig = plot_ratio_hists(rs)
    fig2, ratio_stats = plot_ratio_summary(rs)
    ratio_stats.to_csv(f'{PROJ_DIR}new_results/ratio_summary_all.csv')
    fig3 = plot_H_vs_metal(ratio_stats)
    fig4 = plot_broad_frac_heatmap(all_info)
    fig5 = plot_sigmas(all_info)
    return fig, fig2, fig3, fig4, fig5
