"""
Continuum subtraction before FINAL_LINES, with the continuum uncertainty
carried into the error spectrum.
"""
import os
import numpy as np
import pandas as pd
import spline_lines as sl

C_KMS = sl.C_KMS


# ================================================================
# Pieces
# ================================================================

WIDE_LINES = ('H1_6563A', 'H1_4861A', 'O3_5007A', 'O3_4959A')


def line_mask(fit, kin, linelist_dict, min_sigma=3.0, frac_err=0.3,
              wide_lines=WIDE_LINES, wide_kms=1500.0, bright_snr=30.0,
              bright_kms=1000.0):

    w = fit['wave']
    lines_model = fit['model'] - fit['continuum']
    mask = lines_model > frac_err * fit['err']
    for label in fit['labels']:
        mu = linelist_dict[label] * (1 + fit['z'])
        s = mu * kin['sigma_v_broad'] / C_KMS
        near = np.abs(w - mu) < min_sigma * s
        mask[near] = True
        if label in wide_lines:
            mask[np.abs(w - mu) < mu * wide_kms / C_KMS] = True
        elif near.sum() > 0:
            peak_snr = np.max(lines_model[near] / fit['err'][near])
            if peak_snr > bright_snr:
                mask[np.abs(w - mu) < mu * bright_kms / C_KMS] = True
    return mask


def spline_stat_error(fit):

    Aw = fit['A'] / fit['err'][:, None]
    cov = np.linalg.pinv(Aw.T @ Aw)
    n = fit['n_spl']
    B = fit['A'][:, :n]
    cov_s = cov[:n, :n]
    var = np.einsum('ij,jk,ik->i', B, cov_s, B)
    return np.sqrt(np.clip(var, 0, None))


def _sideband(wave, resid, free, edge, direction, width, min_pix,
              max_width):

    w = width
    while True:
        if direction < 0:
            m = free & (wave < edge) & (wave >= edge - w)
        else:
            m = free & (wave > edge) & (wave <= edge + w)
        if m.sum() >= min_pix or w >= max_width:
            return m
        w = w * 1.5


def local_correction(wave, resid, free, half_window=25.0, step=5.0,
                     min_pix=15, side_width=None, bridge=True):

    nodes = np.arange(wave.min(), wave.max() + step, step)
    node_w = []
    node_med = []
    node_err = []
    for x in nodes:
        m = free & (np.abs(wave - x) <= half_window)
        n = m.sum()
        if n < min_pix:
            continue
        v = resid[m]
        node_w.append(x)
        node_med.append(np.median(v))
        node_err.append(1.2533 * np.std(v) / np.sqrt(n))
    node_w = np.array(node_w)
    corr = np.interp(wave, node_w, np.array(node_med))
    corr_err = np.interp(wave, node_w, np.array(node_err))
    if not bridge:
        return corr, corr_err

    if side_width is None:
        side_width = half_window
    # masked stretches (runs of non-free pixels)
    gaps = []
    i = 0
    n_pix = len(wave)
    while i < n_pix:
        if free[i]:
            i += 1
            continue
        j = i
        while j < n_pix and not free[j]:
            j += 1
        gaps.append((i, j - 1))
        i = j
    for (i0, i1) in gaps:
        if i0 == 0 or i1 == n_pix - 1:
            continue                      # at the edge of the spectrum
        lo = wave[i0]
        hi = wave[i1]
        ml = _sideband(wave, resid, free, lo, -1, side_width, min_pix,
                       4 * side_width)
        mr = _sideband(wave, resid, free, hi, +1, side_width, min_pix,
                       4 * side_width)
        if ml.sum() < 5 or mr.sum() < 5:
            continue
        xl = np.median(wave[ml])
        xr = np.median(wave[mr])
        # the running median at the sideband centres (so the bridge joins
        # the rest of the correction without a step)
        yl = np.interp(xl, wave, corr)
        yr = np.interp(xr, wave, corr)
        el = 1.2533 * np.std(resid[ml]) / np.sqrt(ml.sum())
        er = 1.2533 * np.std(resid[mr]) / np.sqrt(mr.sum())
        span = (wave >= xl) & (wave <= xr)
        t = (wave[span] - xl) / (xr - xl)
        corr[span] = yl + t * (yr - yl)
        # error of a straight line through two independent points
        corr_err[span] = np.sqrt(((1 - t) * el)**2 + (t * er)**2)
    return corr, corr_err


def one_continuum(wave, flux, err, linelist_dict, z0, kin, dlam,
                  half_window):
    """Spline (joint fit) + local correction for one knot spacing.
    Returns the continuum on the fit pixels and its pieces."""
    fit = sl.fit_spectrum(wave, flux, err, linelist_dict, z0, kin,
                          dlam=dlam, rescale=False)
    mask = line_mask(fit, kin, linelist_dict)
    free = ~mask
    resid = fit['flux'] - fit['continuum']
    corr, corr_err = local_correction(fit['wave'], resid, free,
                                      half_window=half_window)
    return {'fit': fit, 'cont': fit['continuum'] + corr,
            'spline': fit['continuum'], 'local': corr,
            'sigma_local': corr_err, 'free': free}


def noise_scale(wave, r, free, half_window=150.0, step=25.0, max_lag=4,
                n_line=8, min_pix=40):
    """How wrong the pixel errors are, as a function of wavelength.
    r = (data - continuum) / err on the pixels; only line-free pixels used.
      scatter : robust std of r (should be 1 if err is right; it also
                absorbs real pixel-scale structure the continuum misses)
      corr    : extra factor because neighbouring pixels are correlated
                (resampling, line spread function). For a sum over n_line
                pixels the variance grows by 1 + 2 sum_k (1 - k/n) rho_k.
    Returns both, interpolated to every pixel."""
    nodes = np.arange(wave.min(), wave.max() + step, step)
    nw = []
    ns = []
    nc = []
    idx_all = np.arange(len(wave))
    for x in nodes:
        m = free & (np.abs(wave - x) <= half_window)
        if m.sum() < min_pix:
            continue
        v = r[m]
        idx = idx_all[m]
        s = 1.4826 * np.median(np.abs(v - np.median(v)))
        var_factor = 1.0
        pos = {}
        for j in range(len(idx)):
            pos[idx[j]] = j
        for k in range(1, max_lag + 1):
            # pairs of line-free pixels k apart
            a = []
            b = []
            for j in range(len(idx)):
                if idx[j] + k in pos:
                    a.append(v[j])
                    b.append(v[pos[idx[j] + k]])
            if len(a) < 20:
                continue
            rho = np.corrcoef(a, b)[0, 1]
            if k < n_line:
                var_factor = var_factor + 2 * (1 - k / n_line) * rho
        nw.append(x)
        ns.append(s)
        nc.append(np.sqrt(max(var_factor, 1.0)))
    nw = np.array(nw)
    return np.interp(wave, nw, np.array(ns)), np.interp(wave, nw, np.array(nc))


def sky_pixels(wave, err, free, half_window=15.0, k=1.5, grow=2):

    n = len(wave)
    base = np.zeros(n)
    for i in range(n):
        lo = np.searchsorted(wave, wave[i] - half_window)
        hi = np.searchsorted(wave, wave[i] + half_window, side='right')
        base[i] = np.median(err[lo:hi])
    spike = (err > k * base) & free
    flag = spike.copy()
    for i in np.where(spike)[0]:
        flag[max(0, i - grow):min(n, i + grow + 1)] = True
    return flag


def residual_envelope(resid, flag, grow=2):

    n = len(resid)
    env = np.zeros(n)
    for i in np.where(flag)[0]:
        env[i] = np.max(np.abs(resid[max(0, i - grow):min(n, i + grow + 1)]))
    return env

# ================================================================
# Main function
# ================================================================


def build_continuum(wave, flux, err, linelist_dict, z0, kin,
                    dlam_best=200.0, dlams=(100.0, 150.0, 200.0, 300.0),
                    half_window=25.0, rescale_err=True, correlated=True,
                    sky_inflate=True, sky_waves=None, sky_half=1.5,
                    sky_k=1.5, out_nsig=5.0):
    """
    Continuum and its uncertainty on the full wavelength grid.
    Returns a DataFrame with wave, flux (continuum-subtracted), err
    (pixel + continuum), err_pix, cont, cont_err, sigma_stat, sigma_local,
    sigma_knots, spline, local, line_free.
    """
    good = np.isfinite(flux) & np.isfinite(err) & (err > 0)

    # the adopted continuum
    best = one_continuum(wave, flux, err, linelist_dict, z0, kin,
                         dlam_best, half_window)
    fw = best['fit']['wave']
    sigma_stat = spline_stat_error(best['fit'])

    # the same with other knot spacings: spread = model uncertainty
    conts = []
    for dlam in dlams:
        if dlam == dlam_best:
            conts.append(best['cont'])
        else:
            other = one_continuum(wave, flux, err, linelist_dict, z0, kin,
                                  dlam, half_window)
            conts.append(other['cont'])
    conts = np.array(conts)
    sigma_knots = np.std(conts, axis=0, ddof=1)

    cont_err = np.sqrt(sigma_stat**2 + best['sigma_local']**2 + sigma_knots**2)

    # are the pixel errors right? (data - cont)/err on the line-free
    # pixels should scatter as N(0,1); if not, rescale (DJA does the same)
    fw_free = best['free']
    r = (best['fit']['flux'] - best['cont']) / best['fit']['err']
    scatter, corr = noise_scale(fw, r, fw_free)
    print(f"Pixel errors: scatter of line-free residuals / err = "
          f"{np.median(scatter):.2f} (median), correlation factor = "
          f"{np.median(corr):.2f}")
    factor = np.ones(len(fw))
    if rescale_err:
        factor = factor * np.maximum(scatter, 1.0)
    if correlated:
        factor = factor * corr
        # the running median is an average over correlated pixels too
        best['sigma_local'] = best['sigma_local'] * corr
        cont_err = np.sqrt(sigma_stat**2 + best['sigma_local']**2
                           + sigma_knots**2)

    flag = np.zeros(len(fw), dtype=bool)
    infl = np.zeros(len(fw))
    if sky_inflate:
        err_pix_fit = best['fit']['err'] * factor
        resid_lines = best['fit']['flux'] - best['fit']['model'] - best['local']
        spike = sky_pixels(fw, best['fit']['err'], fw_free, k=sky_k)
        outlier = fw_free & (np.abs(resid_lines) > out_nsig * err_pix_fit)
        flag = spike | outlier
        if sky_waves is not None:
            for s in sky_waves:
                flag[np.abs(fw - s) <= sky_half] = True
        env = residual_envelope(resid_lines, flag)
        infl[flag] = env[flag]
        print(f'Sky/outlier pixels with inflated error: {flag.sum()}')
    infl_full = np.zeros(len(wave))
    infl_full[np.where(good)[0]] = infl
    flag_full = np.zeros(len(wave), dtype=bool)
    flag_full[np.where(good)[0]] = flag

    # back to the full grid (bad pixels keep their original values)
    cont_full = np.interp(wave, fw, best['cont'])
    out = pd.DataFrame({'wave': wave})
    out['err_pix'] = np.sqrt((err * np.interp(wave, fw, factor))**2
                             + infl_full**2)
    out['sky_flag'] = flag_full
    out['err_factor'] = np.interp(wave, fw, factor)
    out['cont'] = cont_full
    out['cont_err'] = np.interp(wave, fw, cont_err)
    out['sigma_stat'] = np.interp(wave, fw, sigma_stat)
    out['sigma_local'] = np.interp(wave, fw, best['sigma_local'])
    out['sigma_knots'] = np.interp(wave, fw, sigma_knots)
    out['spline'] = np.interp(wave, fw, best['spline'])
    out['local'] = np.interp(wave, fw, best['local'])
    free_full = np.zeros(len(wave), dtype=bool)
    free_full[np.where(good)[0]] = best['free']
    out['line_free'] = free_full
    out['flux'] = flux - cont_full
    out['err'] = np.sqrt(out['err_pix']**2 + out['cont_err']**2)
    out.loc[~good, 'err'] = err[~good]
    return out, best['fit']


def save_for_final_lines(class_, table, proj_DIR, out_name='cont_sub'):
    """Two files in new_results/cont_sub/{gal}/ (created if missing):
      {gal}_contsub.csv        wave, flux, err   (what read_data reads)
      {gal}_contsub_extra.csv  the continuum, its error pieces, err_pix
    """
    gal = class_.gal_id
    folder = f'{proj_DIR}new_results/{out_name}/{gal}'
    os.makedirs(folder, exist_ok=True)
    table[['wave', 'flux', 'err']].rename(columns={'err': 'sigma'}).to_csv(
        f'{folder}/{gal}_contsub.csv', index=False)
    table.to_csv(f'{folder}/{gal}_contsub_extra.csv', index=False)
    print(f'Saved {gal}_contsub.csv and {gal}_contsub_extra.csv in {folder}')


def report_at_lines(table, linelist_dict, z, labels, kin):
    """Continuum and its error under each line, and the implied error on
    the line flux if the continuum shifts coherently:
        sigma_F(cont) = cont_err * W,  W = sqrt(2 pi) * sigma_line (A)
    (the flux width of a Gaussian line)."""
    rows = []
    for label in labels:
        mu = linelist_dict[label] * (1 + z)
        s = mu * kin['sigma_v_narrow'] / C_KMS
        i = np.argmin(np.abs(table['wave'].values - mu))
        row = table.iloc[i]
        rows.append({'label': label, 'cont': row['cont'],
                     'cont_err': row['cont_err'],
                     'stat': row['sigma_stat'], 'local': row['sigma_local'],
                     'knots': row['sigma_knots'],
                     'sigma_F_cont': row['cont_err'] * np.sqrt(2 * np.pi) * s})
    return pd.DataFrame(rows).set_index('label')


def plot_continuum(class_, table, linelist_dict, z, zoom, half_width=60.0,
                   save_path=None):
    """Full spectrum + continuum; zooms with the continuum, its error band
    and the line-free pixels used for the local correction; and under each
    zoom the continuum-subtracted spectrum (what FINAL_LINES fits) with the
    saved error."""
    import matplotlib.pyplot as plt
    n = len(zoom)
    fig = plt.figure(figsize=(4 * n, 11))
    gs = fig.add_gridspec(3, n, height_ratios=[1, 1.2, 1], hspace=0.35)
    ax0 = fig.add_subplot(gs[0, :])
    data = table['flux'] + table['cont']
    ax0.plot(table['wave'], data, color='0.6', lw=0.5)
    ax0.plot(table['wave'], table['cont'], color='C3', lw=1.2,
             label='continuum (spline + local)')
    ax0.plot(table['wave'], table['spline'], color='C0', lw=0.8, ls='--',
             label='spline only')
    ax0.set_ylim(np.nanpercentile(table['cont'], 1) * 0.6,
                 np.nanpercentile(table['cont'], 99) * 1.8)
    ax0.legend(frameon=False)
    ax0.set_title(class_.gal_id)
    for k in range(n):
        mu = linelist_dict[zoom[k]] * (1 + z)
        hw = half_width
        if zoom[k] == 'H1_6563A':
            hw = 100.0     # Halpha + [NII] complex
        m = (np.abs(table['wave'] - mu) < hw).values
        t = table[m]
        d = (t['flux'] + t['cont']).values
        ax = fig.add_subplot(gs[1, k])
        ax.step(t['wave'], d, where='mid', color='k', lw=0.7)
        free = t['line_free'].values
        ax.plot(t['wave'][free], d[free], '.', color='C2', ms=2,
                label='line-free')
        ax.fill_between(t['wave'], t['cont'] - t['cont_err'],
                        t['cont'] + t['cont_err'], color='C3', alpha=0.3)
        ax.plot(t['wave'], t['cont'], color='C3')
        ax.plot(t['wave'], t['spline'], color='C0', ls='--', lw=0.8)
        top = np.nanpercentile(t['cont'], 50) + 8 * np.nanmedian(t['err_pix'])
        ax.set_ylim(np.nanmin(t['cont']) - 6 * np.nanmedian(t['err_pix']), top)
        ax.set_title(zoom[k])
        if k == 0:
            ax.legend(fontsize=7, loc='upper left')

        # continuum-subtracted spectrum (what FINAL_LINES fits)
        ax2 = fig.add_subplot(gs[2, k])
        ax2.fill_between(t['wave'], -t['err'], t['err'], color='0.85',
                         step='mid', label='$\\pm$ err (saved)')
        ax2.step(t['wave'], t['flux'], where='mid', color='C0', lw=0.7)
        if 'sky_flag' in t.columns:
            sk = t['sky_flag'].values
            ax2.plot(t['wave'][sk], np.zeros(sk.sum()), '|', color='C1', ms=6,
                     label='inflated (sky/outlier)')
        ax2.axhline(0, color='k', lw=0.6)
        e_med = np.nanmedian(t['err'])
        ax2.set_ylim(-5 * e_med, 10 * e_med)
        ax2.set_xlabel(r'Obs. wavelength ($\AA$)', fontsize=9)
        if k == 0:
            ax2.set_ylabel('continuum-subtracted')
            ax2.legend(fontsize=7, loc='upper left')
    if save_path is not None:
        fig.savefig(save_path, dpi=130, bbox_inches='tight')
    return fig


# ================================================================
# One call for a galaxy
# ================================================================

def run(class_, info_bright, proj_DIR, dlam_best=200.0, half_window=25.0,
        plot=True, bcorr_path=None, out_name='cont_sub', sky_inflate=True,
        sky_waves=None):
    """
    class_      : SPECTRALDATA object
    info_bright : lines_info table of the current bright-line fits (only
                  used for the widths and broad-to-narrow ratios)
    bcorr_path  : the MW + Balmer corrected spectrum (wave, flux, sigma).
                  Default new_results/balmer_abs/{gal}/bcorr_{gal}_new.csv
                  (where run_balmer_series_pipeline saves it). It is read
                  directly (not through FINAL_LINES.read_data, which will
                  return the continuum-subtracted file once it exists).
    """
    gal = class_.gal_id
    if bcorr_path is None:
        bcorr_path = (f'{proj_DIR}new_results/balmer_abs_final/{gal}/'
                      f'bcorr_{gal}_new.csv')
    d = pd.read_csv(bcorr_path)
    wave = d.iloc[:, 0].to_numpy(dtype=float)
    flux = d.iloc[:, 1].to_numpy(dtype=float)
    err = d.iloc[:, 2].to_numpy(dtype=float)
    print(f'Continuum subtraction of {bcorr_path}')
    kin = sl.kinematics_from_info(info_bright)
    table, fit = build_continuum(wave, flux, err, class_.linelist_dict,
                                 class_.redshift, kin, dlam_best=dlam_best,
                                 half_window=half_window,
                                 sky_inflate=sky_inflate, sky_waves=sky_waves)
    save_for_final_lines(class_, table, proj_DIR, out_name=out_name)
    if plot:
        zoom = []
        for label in ('H1_4861A', 'H1_4340A', 'O3_4363A', 'O2_3729A',
                      'N2_5755A', 'S3_6312A', 'O2_7320A'):
            if label in class_.linelist_dict:
                zoom.append(label)
        plot_continuum(class_, table, class_.linelist_dict, fit['z'], zoom,
                       save_path=f'{proj_DIR}new_results/{out_name}/{gal}/'
                                 f'{gal}_continuum.png')
    return table, fit
