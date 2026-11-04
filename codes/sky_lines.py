"""
Where are the badly subtracted sky lines, and which emission lines do they
touch? Built for the dust correction (Balmer lines) but checks every line.

Idea
----
A night-sky line leaves two traces in a reduced spectrum, at a FIXED
observed wavelength (the same for every galaxy, whatever its redshift):
  1. a peak in the error spectrum (the sky photons add Poisson noise), and
  2. if the subtraction was poor, a spike or a hole in the flux.
On the line-free pixels of each galaxy both are easy to find. Because the
galaxies have different redshifts, a sky line hidden under H-alpha in one
galaxy falls on clean continuum in the others. So:

  step 1  detect sky pixels on the line-free part of every galaxy
          (err / running median of err > err_thresh, or |flux / err| > resid_thresh
           in the continuum-subtracted spectrum),
  step 2  stack the nine galaxies in OBSERVED wavelength: a feature found
          in >= min_gal galaxies is a sky line -> empirical sky catalog,
  step 3  for each galaxy and each emission line, check whether a catalog
          sky line, a known sky line ([O I], Na D, Hg) or a telluric
          absorption band falls in the core or the wings of the line.

No OH wavelengths are typed in by hand: the catalog comes from your own
MagE spectra (same instrument, same sky). The few known lines are only used
to label the catalog and to tell whether the spectra are in air or vacuum.

Input: the files written by continuum_subtract.run, i.e.
    new_results/cont_sub/{gal}/{gal}_contsub_extra.csv
(wave, flux = continuum-subtracted, err, err_pix, err_factor, line_free, ...)

Output (new_results/sky_check/):
    sky_catalog.csv               the empirical sky lines (observed A)
    {gal}_sky_lines.csv           every emission line with its verdict
    balmer_sky_summary.csv        H-alpha, H-beta, H-gamma, H-delta per galaxy
    sky_overview.png              all galaxies: sky lines vs Balmer lines
    {gal}_balmer_sky.png          zooms on the Balmer lines
    {gal}_spectrum_sky.png        the whole spectrum with sky lines, telluric
                                  bands and every emission line marked
"""
import os
import numpy as np
import pandas as pd

C_KMS = 299792.458

# Night-sky emission lines (air wavelengths). Only for labelling.
KNOWN_SKY_AIR = {'[OI] 5577': 5577.338, '[OI] 6300': 6300.304,
                 '[OI] 6364': 6363.776, 'NaD2 5890': 5889.950,
                 'NaD1 5896': 5895.924, 'HgI 4047': 4046.565,
                 'HgI 4358': 4358.335, 'HgI 5461': 5460.735}

# Telluric absorption bands (observed A, approximate edges). These do not
# show up in the error spectrum: they are corrected (or not) by the
# telluric correction, so they are checked by position only.
TELLURIC_BANDS = {'O2 B band': (6865.0, 6920.0),
                  'H2O 7200': (7160.0, 7340.0),
                  'O2 A band': (7590.0, 7700.0),
                  'H2O 8200': (8130.0, 8350.0),
                  'H2O 9300': (8950.0, 9800.0)}

BALMER = ('H1_6563A', 'H1_4861A', 'H1_4340A', 'H1_4102A')
BALMER_NAMES = {'H1_6563A': 'Halpha', 'H1_4861A': 'Hbeta',
                'H1_4340A': 'Hgamma', 'H1_4102A': 'Hdelta'}


def air_to_vac(lam_air):
    """Air -> vacuum (Morton 1991 / IAU standard)."""
    s2 = (1e4 / lam_air)**2
    n = 1 + 8.34254e-5 + 2.406147e-2 / (130 - s2) + 1.5998e-4 / (38.9 - s2)
    return lam_air * n


def known_sky(vacuum=True):
    out = {}
    for name in KNOWN_SKY_AIR:
        lam = KNOWN_SKY_AIR[name]
        if vacuum:
            lam = air_to_vac(lam)
        out[name] = lam
    return out


# ================================================================
# Reading
# ================================================================

def load_galaxy(class_, proj_DIR):
    """The continuum-subtracted spectrum and its pieces for one galaxy."""
    gal = class_.gal_id
    t = pd.read_csv(f'{proj_DIR}new_results/cont_sub/{gal}/'
                    f'{gal}_contsub_extra.csv')
    d = {'gal': gal, 'z': class_.redshift,
         'wave': t['wave'].to_numpy(dtype=float),
         'flux': t['flux'].to_numpy(dtype=float),
         'err': t['err'].to_numpy(dtype=float),
         'cont': t['cont'].to_numpy(dtype=float),
         'line_free': t['line_free'].to_numpy(dtype=bool)}
    # the pixel error before the rescaling of continuum_subtract
    d['err_raw'] = (t['err_pix'] / t['err_factor']).to_numpy(dtype=float)
    return d


def kinematics(class_, proj_DIR, default_narrow=50.0, default_broad=150.0):
    """Line widths from the bright-line fits (same as continuum_subtract);
    falls back to typical values if the file is not there."""
    gal = class_.gal_id
    path = f'{proj_DIR}new_results/{gal}/{gal}_lines_info.csv'
    if os.path.exists(path):
        import spline_lines as sl
        kin = sl.kinematics_from_info(pd.read_csv(path))
        return kin
    print(f'{gal}: no lines_info, using sigma_v = {default_narrow} / '
          f'{default_broad} km/s')
    return {'sigma_v_narrow': default_narrow, 'sigma_v_broad': default_broad}


def read_transmission(path, R=4100.0, vacuum=True, wave_col=0, trans_col=1):
    """Telluric transmission curve (e.g. from ESO SkyCalc, TAPAS, or the
    ratio of a telluric standard star to its model). Any text/csv/fits
    table with wavelength and transmission (0-1).
    Units are guessed from the range: micron (< 3), nm (< 3000) or A.
    It is smoothed to the MagE resolution R (Gaussian, constant R).
    Returns {'wave': A, 'trans': 0-1}."""
    if path.endswith('.fits') or path.endswith('.fit'):
        from astropy.table import Table
        t = Table.read(path)
        cols = t.colnames
        w = np.asarray(t[cols[wave_col]], dtype=float)
        tr = np.asarray(t[cols[trans_col]], dtype=float)
    else:
        arr = np.genfromtxt(path, comments='#', delimiter=None,
                            invalid_raise=False)
        if np.isnan(arr[0]).all():
            arr = arr[1:]
        if arr.ndim == 1 or np.isnan(arr).all():
            t = pd.read_csv(path, sep=None, engine='python')
            arr = t.to_numpy(dtype=float)
        w = arr[:, wave_col]
        tr = arr[:, trans_col]
    good = np.isfinite(w) & np.isfinite(tr)
    w = w[good]
    tr = tr[good]
    if np.nanmax(w) < 3.0:
        w = w * 1e4            # micron -> A
    elif np.nanmax(w) < 3000.0:
        w = w * 10.0           # nm -> A
    order = np.argsort(w)
    w = w[order]
    tr = tr[order]
    if not vacuum:
        w = w / (air_to_vac(w) / w)      # vacuum model -> air spectra
    if R is not None and R > 0:
        # resample to constant d(ln lambda), smooth, back
        lnw = np.log(w)
        dln = np.median(np.diff(lnw))
        grid = np.arange(lnw[0], lnw[-1], dln)
        tg = np.interp(grid, lnw, tr)
        sig = (1.0 / R) / 2.3548 / dln          # in pixels
        half = int(np.ceil(4 * sig))
        if half >= 1:
            k = np.exp(-0.5 * (np.arange(-half, half + 1) / sig)**2)
            k = k / k.sum()
            tg = np.convolve(tg, k, mode='same')
        w = np.exp(grid)
        tr = tg
    print(f'Transmission: {w.min():.0f}-{w.max():.0f} A, smoothed to R = {R}')
    return {'wave': w, 'trans': tr}


def read_pypeit_telluric(path, R=None):
    """Telluric transmission from PypeIt (pypeit_tellfit / pypeit_coadd_1dspec
    outputs): the *_tellcorr.fits spectrum (column 'telluric') or the
    *_tellmodel.fits file (columns 'WAVE', 'TELLURIC'; one row per order for
    echelle data such as MagE). Wavelength in A, already at the instrument
    resolution, so no smoothing unless R is given.
    Returns {'wave', 'trans'} and, when the file has them, 'flux' (telluric
    corrected) and 'obj_model'."""
    from astropy.io import fits
    out = None
    with fits.open(path) as hdul:
        for hdu in hdul:
            if not isinstance(hdu, fits.BinTableHDU):
                continue
            cols = {}
            for name in hdu.columns.names:
                cols[name.lower()] = name
            wkey = None
            for cand in ('wave', 'wave_grid_mid', 'opt_wave', 'box_wave'):
                if cand in cols:
                    wkey = cols[cand]
                    break
            if wkey is None or 'telluric' not in cols:
                continue
            w = np.asarray(hdu.data[wkey], dtype=float).ravel()
            t = np.asarray(hdu.data[cols['telluric']], dtype=float).ravel()
            out = {'wave': w, 'trans': t}
            for extra in ('flux', 'obj_model'):
                if extra in cols:
                    out[extra] = np.asarray(hdu.data[cols[extra]],
                                            dtype=float).ravel()
            break
    if out is None:
        raise ValueError(f'No wavelength + telluric columns in {path}')
    good = np.isfinite(out['wave']) & np.isfinite(out['trans']) & (out['wave'] > 0)
    order = np.argsort(out['wave'][good])
    for key in list(out.keys()):
        out[key] = out[key][good][order]
    if R is not None and R > 0:
        lnw = np.log(out['wave'])
        dln = np.median(np.diff(lnw))
        grid = np.arange(lnw[0], lnw[-1], dln)
        tg = np.interp(grid, lnw, out['trans'])
        sig = (1.0 / R) / 2.3548 / dln
        half = int(np.ceil(4 * sig))
        if half >= 1:
            k = np.exp(-0.5 * (np.arange(-half, half + 1) / sig)**2)
            tg = np.convolve(tg, k / k.sum(), mode='same')
        out = {'wave': np.exp(grid), 'trans': tg}
    print(f'PypeIt telluric: {out["wave"].min():.0f}-{out["wave"].max():.0f} A, '
          f'min T = {np.nanmin(out["trans"]):.2f}')
    return out


def transmission_for(gal, transmission, R=4100.0, vacuum=True, pypeit=True):
    """transmission: None, one path for all galaxies, a dict {gal: path},
    or a pattern with {gal} in it (e.g. '.../{gal}_tellmodel.fits')."""
    if transmission is None:
        return None
    if isinstance(transmission, dict):
        if gal not in transmission:
            print(f'{gal}: no telluric file given')
            return None
        path = transmission[gal]
    else:
        path = transmission.replace('{gal}', gal)
    if not os.path.exists(path):
        print(f'{gal}: {path} not found, telluric check uses band edges')
        return None
    if pypeit and (path.endswith('.fits') or path.endswith('.fit')):
        return read_pypeit_telluric(path)
    return read_transmission(path, R=R, vacuum=vacuum)


def transmission_at(trans, wave):
    """Transmission interpolated on a wavelength grid (1 outside the curve)."""
    if trans is None:
        return np.ones(len(wave))
    return np.interp(wave, trans['wave'], trans['trans'], left=1.0, right=1.0)


# ================================================================
# Step 1: sky pixels in one galaxy
# ================================================================

def running_median(wave, y, good, half_window=15.0, step=2.0, min_pix=10):
    nodes = np.arange(wave.min(), wave.max() + step, step)
    xs = []
    ms = []
    for x in nodes:
        m = good & (np.abs(wave - x) <= half_window)
        if m.sum() < min_pix:
            continue
        xs.append(x)
        ms.append(np.median(y[m]))
    return np.interp(wave, np.array(xs), np.array(ms))


def detect_sky(d, err_thresh=1.4, resid_thresh=5.0, half_window=15.0):
    """Flags on the LINE-FREE pixels of one galaxy:
        err_ratio  = err / running median(err)   (sky noise peak)
        resid      = flux / err                  (badly subtracted sky)
    Returns the boolean flag and the two diagnostics."""
    wave = d['wave']
    good = np.isfinite(d['err_raw']) & (d['err_raw'] > 0) & np.isfinite(d['flux'])
    base = running_median(wave, d['err_raw'], good, half_window)
    err_ratio = np.full(len(wave), np.nan)
    err_ratio[good] = d['err_raw'][good] / base[good]
    resid = np.full(len(wave), np.nan)
    resid[good] = d['flux'][good] / d['err'][good]
    use = good & d['line_free']
    flag = np.zeros(len(wave), dtype=bool)
    for i in range(len(wave)):
        if not use[i]:
            continue
        if err_ratio[i] > err_thresh or abs(resid[i]) > resid_thresh:
            flag[i] = True
    d['err_ratio'] = err_ratio
    d['resid'] = resid
    d['sky_flag'] = flag
    d['covered'] = use
    return flag


# ================================================================
# Step 2: empirical sky catalog from all galaxies
# ================================================================

def build_sky_catalog(gal_data, bin_width=0.5, min_gal=2, pad=1):
    """Stack the flags of all galaxies on a common observed grid. A bin
    flagged in >= min_gal galaxies is sky. Neighbouring sky bins are merged
    into one feature (lo, hi, centre, n_gal, n_covered, max err ratio)."""
    lo = 1e9
    hi = 0.0
    for d in gal_data:
        lo = min(lo, d['wave'].min())
        hi = max(hi, d['wave'].max())
    edges = np.arange(lo, hi + bin_width, bin_width)
    nb = len(edges) - 1
    n_flag = np.zeros(nb, dtype=int)
    n_cov = np.zeros(nb, dtype=int)
    peak = np.zeros(nb)
    for d in gal_data:
        idx = np.floor((d['wave'] - lo) / bin_width).astype(int)
        flagged_bins = set()
        covered_bins = set()
        for i in range(len(idx)):
            b = idx[i]
            if b < 0 or b >= nb:
                continue
            if d['covered'][i]:
                covered_bins.add(b)
            if d['sky_flag'][i]:
                for bb in range(b - pad, b + pad + 1):
                    if 0 <= bb < nb:
                        flagged_bins.add(bb)
                if np.isfinite(d['err_ratio'][i]):
                    peak[b] = max(peak[b], d['err_ratio'][i])
        for b in flagged_bins:
            n_flag[b] += 1
        for b in covered_bins:
            n_cov[b] += 1

    rows = []
    b = 0
    while b < nb:
        if n_flag[b] < min_gal:
            b += 1
            continue
        start = b
        while b < nb and n_flag[b] >= min_gal:
            b += 1
        stop = b - 1
        seg_peak = 0.0
        seg_n = 0
        seg_cov = 0
        for k in range(start, stop + 1):
            seg_peak = max(seg_peak, peak[k])
            seg_n = max(seg_n, n_flag[k])
            seg_cov = max(seg_cov, n_cov[k])
        rows.append({'lo': edges[start], 'hi': edges[stop + 1],
                     'centre': 0.5 * (edges[start] + edges[stop + 1]),
                     'n_gal': seg_n, 'n_covered': seg_cov,
                     'max_err_ratio': round(seg_peak, 2)})
    cat = pd.DataFrame(rows)
    return cat


def label_catalog(cat, vacuum=True, tol=1.5):
    """Name the catalog features that match a known sky line."""
    known = known_sky(vacuum)
    names = []
    for _, row in cat.iterrows():
        name = ''
        for k in known:
            if row['lo'] - tol <= known[k] <= row['hi'] + tol:
                name = k
        names.append(name)
    cat['known'] = names
    return cat


def air_or_vacuum(gal_data, window=4.0):
    """Where is the [O I] 5577 sky line in the error spectra: at the air
    (5577.34) or the vacuum (5578.89) wavelength? Median over galaxies of
    the position of the error peak near 5578."""
    pos = []
    for d in gal_data:
        m = (np.abs(d['wave'] - 5578.1) < window) & np.isfinite(d['err_ratio'])
        if m.sum() < 3:
            continue
        i = np.argmax(d['err_ratio'][m])
        pos.append(d['wave'][m][i])
    if len(pos) == 0:
        print('[O I] 5577 not covered: cannot tell air or vacuum, assuming vacuum')
        return True
    p = np.median(pos)
    vac = abs(p - air_to_vac(5577.338)) < abs(p - 5577.338)
    print(f'[O I] 5577 sky line found at {p:.2f} A -> spectra in '
          f'{"VACUUM" if vac else "AIR"} wavelengths')
    return vac


# ================================================================
# Step 3: which emission lines are touched
# ================================================================

def check_lines(d, linelist_dict, kin, cat, vacuum=True, n_core=2.5,
                n_wing=3.0, labels=None, trans=None, t_thresh=0.95):
    """For each emission line of one galaxy:
        core  = +- n_core * sigma_narrow   (where most of the flux is)
        wings = +- n_wing * sigma_broad    (the fit window around it)
    and the sky features, known sky lines and telluric bands that overlap.
    verdict: 'SKY IN CORE' > 'TELLURIC' > 'sky in wings' > 'clean'.
    Known sky lines ([O I], Na D, Hg) count only if they were detected in
    the spectra; otherwise they are listed in known_sky_nearby."""
    z = d['z']
    known = known_sky(vacuum)
    if labels is None:
        labels = list(linelist_dict.keys())
    rows = []
    for label in labels:
        mu = linelist_dict[label] * (1 + z)
        if mu < d['wave'].min() or mu > d['wave'].max():
            continue
        s_n = mu * kin['sigma_v_narrow'] / C_KMS
        s_b = mu * kin['sigma_v_broad'] / C_KMS
        core = n_core * s_n
        wing = max(core, n_wing * s_b)

        in_core = []
        in_wing = []
        nearest = np.inf
        for _, row in cat.iterrows():
            gap = 0.0
            if row['hi'] < mu:
                gap = mu - row['hi']
            elif row['lo'] > mu:
                gap = row['lo'] - mu
            nearest = min(nearest, gap)
            tag = f"{row['centre']:.1f}"
            if isinstance(row.get('known', ''), str) and row.get('known', ''):
                tag = tag + f" ({row['known']})"
            if gap <= core:
                in_core.append(tag)
            elif gap <= wing:
                in_wing.append(tag)
        # known sky lines that were NOT found in the catalog (e.g. weak in
        # these nights): reported, but they do not change the verdict
        known_near = []
        for k in known:
            gap = abs(known[k] - mu)
            if gap <= wing:
                found = False
                for t in in_core + in_wing:
                    if k in t:
                        found = True
                if not found:
                    known_near.append(f'{known[k]:.1f} ({k}, not detected)')
        tell = []
        t_min_core = np.nan
        t_min_win = np.nan
        if trans is not None:
            # real transmission curve: minimum over the core and the window
            xs = np.linspace(mu - wing, mu + wing, 200)
            tv = transmission_at(trans, xs)
            in_c = np.abs(xs - mu) <= core
            t_min_core = float(np.min(tv[in_c]))
            t_min_win = float(np.min(tv))
            if t_min_core < t_thresh:
                tell.append(f'T_core = {t_min_core:.2f}')
            elif t_min_win < t_thresh:
                tell.append(f'T_wings = {t_min_win:.2f}')
        else:
            for band in TELLURIC_BANDS:
                b_lo, b_hi = TELLURIC_BANDS[band]
                if mu + wing >= b_lo and mu - wing <= b_hi:
                    tell.append(band)

        if len(in_core) > 0:
            verdict = 'SKY IN CORE'
        elif len(tell) > 0:
            verdict = 'TELLURIC'
        elif len(in_wing) > 0:
            verdict = 'sky in wings'
        else:
            verdict = 'clean'
        rows.append({'line': label, 'obs_wave': round(mu, 2),
                     'core_A': round(core, 1), 'window_A': round(wing, 1),
                     'verdict': verdict,
                     'sky_in_core': '; '.join(in_core),
                     'sky_in_wings': '; '.join(in_wing),
                     'telluric': '; '.join(tell),
                     'T_min_core': round(t_min_core, 3) if np.isfinite(t_min_core) else np.nan,
                     'T_min_window': round(t_min_win, 3) if np.isfinite(t_min_win) else np.nan,
                     'known_sky_nearby': '; '.join(known_near),
                     'nearest_sky_A': round(nearest, 1) if np.isfinite(nearest) else np.nan})
    return pd.DataFrame(rows)


def sky_mask(d, cat, pad=0.5):
    """Boolean mask of the catalog sky features on this galaxy's grid
    (True = sky). Use it to drop pixels in FINAL_LINES, e.g.
        flux[mask] = np.nan   (read_data already uses nan_policy='omit')."""
    mask = np.zeros(len(d['wave']), dtype=bool)
    for _, row in cat.iterrows():
        m = (d['wave'] >= row['lo'] - pad) & (d['wave'] <= row['hi'] + pad)
        mask[m] = True
    return mask


# ================================================================
# Plots
# ================================================================

def plot_overview(gal_data, cat, linelist_dict_by_gal, save_path=None):
    """One row per galaxy, observed wavelength on x: sky features (grey),
    telluric bands (blue), and where the Balmer lines fall (coloured)."""
    import matplotlib.pyplot as plt
    n = len(gal_data)
    fig, ax = plt.subplots(figsize=(15, 0.55 * n + 1.8))
    for band in TELLURIC_BANDS:
        b_lo, b_hi = TELLURIC_BANDS[band]
        ax.axvspan(b_lo, b_hi, color='C0', alpha=0.12)
        ax.text(0.5 * (b_lo + b_hi), n - 0.35, band, ha='center', fontsize=7,
                color='C0')
    for _, row in cat.iterrows():
        alpha = min(0.9, 0.15 + 0.1 * row['n_gal'])
        ax.axvspan(row['lo'], row['hi'], color='0.3', alpha=alpha, lw=0)
    colors = {'H1_6563A': 'C3', 'H1_4861A': 'C2', 'H1_4340A': 'C1',
              'H1_4102A': 'C4'}
    for j in range(n):
        d = gal_data[j]
        ll = linelist_dict_by_gal[d['gal']]
        for label in BALMER:
            if label not in ll:
                continue
            mu = ll[label] * (1 + d['z'])
            ax.plot([mu, mu], [j - 0.35, j + 0.35], color=colors[label], lw=2.5,
                    label=BALMER_NAMES[label] if j == 0 else None)
    ax.set_yticks(range(n))
    names = []
    for d in gal_data:
        names.append(d['gal'])
    ax.set_yticklabels(names)
    ax.set_ylim(-0.6, n - 0.1)
    ax.set_xlabel(r'Observed wavelength ($\AA$)')
    ax.set_title('Sky features found in the spectra (grey, darker = found in more '
                 'galaxies), telluric bands (blue) and Balmer lines')
    ax.legend(loc='upper left', ncol=4, fontsize=8)
    if save_path is not None:
        fig.savefig(save_path, dpi=130, bbox_inches='tight')
    return fig


VERDICT_COLORS = {'SKY IN CORE': 'C3', 'TELLURIC': 'C0',
                  'sky in wings': 'C1', 'clean': 'C2'}


def plot_spectrum_sky(d, cat, lines_table, n_rows=5, save_path=None,
                      flux_range=(-1.0, 5.0), trans=None):
    """The whole spectrum of one galaxy, split in n_rows rows, with
      grey bands  : sky features of the catalog (darker = in more galaxies),
                    a thin black line on top of each where it is known
                    ([O I], Na D, Hg),
      blue bands  : telluric absorption bands,
      ticks + names of the emission lines, coloured by verdict
                    (red = sky in core, blue = telluric, orange = sky in
                    wings, green = clean),
    and under each row err / median(err): the sky lines are the peaks.
    trans: telluric transmission (read_transmission); drawn in blue on the
    right axis (0-1). Without it, the approximate bands are shaded.
    flux_range: y range in units of the median continuum of each row."""
    import matplotlib.pyplot as plt
    wave = d['wave']
    data = d['flux'] + d['cont']
    edges = np.linspace(wave.min(), wave.max(), n_rows + 1)
    fig = plt.figure(figsize=(16, 3.6 * n_rows))
    outer = fig.add_gridspec(n_rows, 1, hspace=0.32)
    for r in range(n_rows):
        lo = edges[r]
        hi = edges[r + 1]
        m = (wave >= lo) & (wave <= hi)
        inner = outer[r, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax = fig.add_subplot(inner[0, 0])
        ax2 = fig.add_subplot(inner[1, 0], sharex=ax)
        if trans is None:
            for band in TELLURIC_BANDS:
                b_lo, b_hi = TELLURIC_BANDS[band]
                if b_hi > lo and b_lo < hi:
                    ax.axvspan(b_lo, b_hi, color='C0', alpha=0.13, lw=0)
                    ax2.axvspan(b_lo, b_hi, color='C0', alpha=0.13, lw=0)
                    ax.text(max(b_lo, lo), 1.0, ' ' + band, color='C0',
                            fontsize=7, va='top',
                            transform=ax.get_xaxis_transform())
        else:
            axt = ax.twinx()
            tm = (trans['wave'] >= lo) & (trans['wave'] <= hi)
            axt.fill_between(trans['wave'][tm], trans['trans'][tm], 1.0,
                             color='C0', alpha=0.18, lw=0)
            axt.plot(trans['wave'][tm], trans['trans'][tm], color='C0', lw=0.8)
            axt.set_ylim(0.0, 1.05)
            axt.set_ylabel('telluric T', fontsize=8, color='C0')
            axt.tick_params(axis='y', labelsize=7, colors='C0')
        for _, row in cat.iterrows():
            if row['hi'] < lo or row['lo'] > hi:
                continue
            alpha = min(0.8, 0.15 + 0.08 * row['n_gal'])
            ax.axvspan(row['lo'], row['hi'], color='0.35', alpha=alpha, lw=0)
            ax2.axvspan(row['lo'], row['hi'], color='0.35', alpha=alpha, lw=0)
            known = row.get('known', '')
            if isinstance(known, str) and known:
                ax.text(row['centre'], 0.02, known, rotation=90, fontsize=6,
                        color='0.2', ha='right', va='bottom',
                        transform=ax.get_xaxis_transform())
        ax.step(wave[m], data[m], where='mid', color='k', lw=0.4)
        ax.plot(wave[m], d['cont'][m], color='C3', lw=0.8)
        c_med = np.nanmedian(d['cont'][m])
        ax.set_ylim(flux_range[0] * c_med, flux_range[1] * c_med)
        if lines_table is not None:
            for _, ln in lines_table.iterrows():
                mu = ln['obs_wave']
                if mu < lo or mu > hi:
                    continue
                col = VERDICT_COLORS.get(ln['verdict'], 'k')
                ax.axvline(mu, ymin=0.66, ymax=0.76, color=col, lw=1.5)
                ax.text(mu, 0.77, ln['line'], rotation=90, fontsize=6.5,
                        color=col, ha='center', va='bottom',
                        transform=ax.get_xaxis_transform(), clip_on=True)
        ax2.step(wave[m], d['err_ratio'][m], where='mid', color='C0', lw=0.5)
        ax2.axhline(1, color='k', lw=0.4)
        ax2.set_ylim(0.6, 3.0)
        ax.set_xlim(lo, hi)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_ylabel('flux', fontsize=8)
        ax2.set_ylabel('err/med', fontsize=8)
        if r == n_rows - 1:
            ax2.set_xlabel(r'Observed wavelength ($\AA$)')
    tell_txt = 'blue = telluric bands'
    if trans is not None:
        tell_txt = 'blue = telluric transmission (right axis)'
    fig.suptitle(f"{d['gal']} (z = {d['z']:.4f}): grey = sky lines found in the "
                 f"spectra, {tell_txt}; line names coloured by verdict "
                 "(red sky in core, blue telluric, orange sky in wings, green clean)",
                 fontsize=10, y=0.905)
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig


def plot_telluric(d, trans, lines_table, cat=None, n_rows=4, save_path=None,
                  t_thresh=0.95):
    """For each wavelength chunk, three panels (as in telluric-correction
    figures):
      flux         : black = spectrum used (telluric corrected), red = the
                     same before the telluric correction (flux * T), blue =
                     continuum; the emission lines marked and named, coloured
                     by verdict
      transmission : the telluric transmission T, with the line positions;
                     red shading where T < t_thresh under a line core
      residual     : continuum-subtracted flux / err, with the sky features
                     of the catalog in grey"""
    import matplotlib.pyplot as plt
    wave = d['wave']
    data = d['flux'] + d['cont']
    T = transmission_at(trans, wave)
    before = data * T
    edges = np.linspace(wave.min(), wave.max(), n_rows + 1)
    fig = plt.figure(figsize=(15, 5.2 * n_rows))
    outer = fig.add_gridspec(n_rows, 1, hspace=0.25)
    for r in range(n_rows):
        lo = edges[r]
        hi = edges[r + 1]
        m = (wave >= lo) & (wave <= hi)
        inner = outer[r, 0].subgridspec(3, 1, height_ratios=[2.2, 1, 1],
                                        hspace=0.0)
        ax = fig.add_subplot(inner[0, 0])
        axt = fig.add_subplot(inner[1, 0], sharex=ax)
        axr = fig.add_subplot(inner[2, 0], sharex=ax)

        ax.step(wave[m], before[m], where='mid', color='C3', lw=0.5,
                label='before telluric correction')
        ax.step(wave[m], data[m], where='mid', color='k', lw=0.5,
                label='telluric corrected')
        ax.plot(wave[m], d['cont'][m], color='C0', lw=1.2, label='continuum')
        c_med = np.nanmedian(d['cont'][m])
        ax.set_ylim(-0.3 * c_med, 3.5 * c_med)

        tm = (trans['wave'] >= lo) & (trans['wave'] <= hi)
        axt.plot(trans['wave'][tm], trans['trans'][tm], color='k', lw=0.6)
        axt.axhline(t_thresh, color='0.5', ls=':', lw=0.8)
        axt.set_ylim(-0.05, 1.1)

        axr.step(wave[m], d['resid'][m], where='mid', color='k', lw=0.4)
        axr.axhline(0, color='0.5', lw=0.6)
        axr.set_ylim(-8, 8)
        if cat is not None:
            for _, row in cat.iterrows():
                if row['hi'] < lo or row['lo'] > hi:
                    continue
                axr.axvspan(row['lo'], row['hi'], color='0.6', alpha=0.4, lw=0)

        if lines_table is not None:
            for _, ln in lines_table.iterrows():
                mu = ln['obs_wave']
                if mu < lo or mu > hi:
                    continue
                col = VERDICT_COLORS.get(ln['verdict'], 'k')
                ax.axvline(mu, color=col, lw=0.6, ls='--', alpha=0.7)
                axt.axvline(mu, color=col, lw=0.6, ls='--', alpha=0.7)
                ax.text(mu, 0.98, ln['line'], rotation=90, fontsize=6.5,
                        color=col, ha='center', va='top',
                        transform=ax.get_xaxis_transform(), clip_on=True)
                core = ln['core_A']
                if transmission_at(trans, np.array([mu]))[0] < 1.0:
                    tc = transmission_at(trans, np.linspace(mu - core, mu + core, 50))
                    if np.min(tc) < t_thresh:
                        axt.axvspan(mu - core, mu + core, color='C3', alpha=0.3,
                                    lw=0)

        ax.set_xlim(lo, hi)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(axt.get_xticklabels(), visible=False)
        ax.set_ylabel('Flux', fontsize=9)
        axt.set_ylabel('Transmission', fontsize=9)
        axr.set_ylabel('(data-cont)/err', fontsize=8)
        if r == 0:
            ax.legend(fontsize=8, loc='upper right', ncol=3)
        if r == n_rows - 1:
            axr.set_xlabel(r'Observed wavelength ($\AA$)')
    fig.suptitle(f"{d['gal']} (z = {d['z']:.4f}): telluric transmission and "
                 "emission lines (red = sky in core, blue = telluric, orange = "
                 "sky in wings, green = clean)", fontsize=10, y=0.9)
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig


def plot_balmer(d, cat, linelist_dict, kin, save_path=None, half_width=40.0,
                trans=None):
    """Zooms on the Balmer lines of one galaxy: continuum-subtracted flux
    with the sky features shaded, and err / running median of err."""
    import matplotlib.pyplot as plt
    labels = []
    for label in BALMER:
        if label in linelist_dict:
            mu = linelist_dict[label] * (1 + d['z'])
            if d['wave'].min() < mu < d['wave'].max():
                labels.append(label)
    n = len(labels)
    fig, axs = plt.subplots(2, n, figsize=(4 * n, 5.5), sharex='col',
                            gridspec_kw={'height_ratios': [2, 1]})
    if n == 1:
        axs = axs.reshape(2, 1)
    for k in range(n):
        label = labels[k]
        mu = linelist_dict[label] * (1 + d['z'])
        hw = half_width
        if label == 'H1_6563A':
            hw = 70.0
        m = np.abs(d['wave'] - mu) < hw
        ax = axs[0, k]
        ax2 = axs[1, k]
        for _, row in cat.iterrows():
            if row['hi'] > mu - hw and row['lo'] < mu + hw:
                ax.axvspan(row['lo'], row['hi'], color='0.5', alpha=0.35, lw=0)
                ax2.axvspan(row['lo'], row['hi'], color='0.5', alpha=0.35, lw=0)
        if trans is None:
            for band in TELLURIC_BANDS:
                b_lo, b_hi = TELLURIC_BANDS[band]
                if b_hi > mu - hw and b_lo < mu + hw:
                    ax.axvspan(b_lo, b_hi, color='C0', alpha=0.12)
        else:
            axt = ax.twinx()
            tm = np.abs(trans['wave'] - mu) < hw
            axt.plot(trans['wave'][tm], trans['trans'][tm], color='C0', lw=1)
            axt.set_ylim(0.0, 1.05)
            axt.tick_params(axis='y', labelsize=7, colors='C0')
        s_n = mu * kin['sigma_v_narrow'] / C_KMS
        ax.axvspan(mu - 2.5 * s_n, mu + 2.5 * s_n, color='C3', alpha=0.12)
        ax.step(d['wave'][m], d['flux'][m], where='mid', color='k', lw=0.7)
        ax.axhline(0, color='k', lw=0.5)
        e_med = np.nanmedian(d['err'][m])
        ax.set_ylim(-6 * e_med, 15 * e_med)
        ax.set_title(f"{d['gal']} {BALMER_NAMES[label]}")
        ax2.step(d['wave'][m], d['err_ratio'][m], where='mid', color='C0', lw=0.7)
        ax2.axhline(1, color='k', lw=0.5)
        ax2.set_ylim(0.5, 3)
        ax2.set_xlabel(r'Obs. wavelength ($\AA$)')
        if k == 0:
            ax.set_ylabel('continuum-subtracted flux')
            ax2.set_ylabel('err / median err')
    fig.suptitle('grey = sky features from the catalog, red = line core '
                 '(+-2.5 sigma narrow), blue = telluric band', fontsize=9)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig


# ================================================================
# One call for the whole sample
# ================================================================

def run_all(classes, proj_DIR, err_thresh=1.4, resid_thresh=5.0, min_gal=2,
            plot=True, transmission=None, R=4100.0, t_thresh=0.95,
            pypeit=True):
    """transmission: telluric transmission, one per galaxy (see
    transmission_for): a dict {gal: path}, a pattern with {gal}, or one
    path for all. PypeIt .fits files are read with read_pypeit_telluric.
    Without it the telluric check uses the approximate band edges in
    TELLURIC_BANDS."""
    out_dir = f'{proj_DIR}new_results/sky_check'
    os.makedirs(out_dir, exist_ok=True)

    gal_data = []
    kins = {}
    lls = {}
    for class_ in classes:
        d = load_galaxy(class_, proj_DIR)
        detect_sky(d, err_thresh, resid_thresh)
        gal_data.append(d)
        kins[d['gal']] = kinematics(class_, proj_DIR)
        lls[d['gal']] = class_.linelist_dict

    vacuum = air_or_vacuum(gal_data)
    cat = build_sky_catalog(gal_data, min_gal=min_gal)
    cat = label_catalog(cat, vacuum)
    cat.to_csv(f'{out_dir}/sky_catalog.csv', index=False)
    print(f'{len(cat)} sky features found in >= {min_gal} galaxies '
          f'(saved sky_catalog.csv)')

    summary = []
    for d in gal_data:
        gal = d['gal']
        trans = transmission_for(gal, transmission, R=R, vacuum=vacuum,
                                 pypeit=pypeit)
        t = check_lines(d, lls[gal], kins[gal], cat, vacuum, trans=trans,
                        t_thresh=t_thresh)
        t.to_csv(f'{out_dir}/{gal}_sky_lines.csv', index=False)
        row = {'gal': gal}
        for label in BALMER:
            sel = t[t['line'] == label]
            if len(sel) == 0:
                row[BALMER_NAMES[label]] = 'not covered'
                continue
            r = sel.iloc[0]
            txt = r['verdict']
            detail = r['sky_in_core'] or r['telluric'] or r['sky_in_wings']
            if detail:
                txt = txt + ': ' + detail
            row[BALMER_NAMES[label]] = txt
        summary.append(row)
        bad = t[t['verdict'] != 'clean']
        print(f'\n{gal}: {len(bad)} of {len(t)} lines touched by sky/telluric')
        if len(bad) > 0:
            print(bad[['line', 'obs_wave', 'verdict', 'sky_in_core',
                       'telluric', 'sky_in_wings']].to_string(index=False))
        if plot:
            plot_balmer(d, cat, lls[gal], kins[gal], trans=trans,
                        save_path=f'{out_dir}/{gal}_balmer_sky.png')
            plot_spectrum_sky(d, cat, t, trans=trans,
                              save_path=f'{out_dir}/{gal}_spectrum_sky.png')
            if trans is not None:
                plot_telluric(d, trans, t, cat, t_thresh=t_thresh,
                              save_path=f'{out_dir}/{gal}_telluric.png')

    summ = pd.DataFrame(summary).set_index('gal')
    summ.to_csv(f'{out_dir}/balmer_sky_summary.csv')
    print('\nBalmer lines for the dust correction:')
    print(summ.to_string())
    if plot:
        plot_overview(gal_data, cat, lls, save_path=f'{out_dir}/sky_overview.png')
    return cat, summ, gal_data