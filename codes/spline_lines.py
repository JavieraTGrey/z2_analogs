"""
Continuum and emission lines fitted together, the way the DAWN JWST Archive
(DJA, msaexp) fits NIRSpec spectra:

    model(lambda) = sum_s  c_s * B_s(lambda)          (continuum: B-splines)
                  + sum_l  F_l * G_l(lambda)          (lines: Gaussians)

The spline shapes B_s and the line shapes G_l are fixed (fixed knots, fixed
redshift and widths), so the model is LINEAR in the unknowns c_s and F_l and
one weighted least-squares solve gives all of them at once. The redshift is
chosen on a small grid (lowest chi^2), as msaexp does.

Each line keeps our narrow + broad model, with the widths and the
broad-to-narrow flux ratios fixed (from the previous lmfit fits):
    G_l = N(sigma_n) + r * N(sigma_b),   N = unit-area Gaussian
so F_l is the NARROW flux and the broad flux is r * F_l.

Uncertainties: the errors are rescaled so the residuals are ~N(0,1)
(DJA does the same), then 100 Monte Carlo realizations of the spectrum are
refitted. The output table has the same columns as FINAL_LINES
(label, i, nar_amplitude, bro_amplitude, ...), so dust_att runs unchanged.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

C_KMS = 299792.458


# ================================================================
# Building blocks
# ================================================================

def spline_basis(wave, dlam=200.0, order=3):
    """Cubic B-splines with knots every ~dlam Angstrom across the spectrum.
    Returns the matrix (n_pixels x n_splines). Larger dlam = smoother
    continuum. The knots must be much wider than the lines (a line is
    ~5 A wide), otherwise the continuum would absorb line flux."""
    lo = wave.min()
    hi = wave.max()
    n_inner = max(int((hi - lo) / dlam), 1)
    inner = np.linspace(lo, hi, n_inner + 1)
    knots = np.concatenate([np.full(order, lo), inner, np.full(order, hi)])
    return BSpline.design_matrix(wave, knots, order).toarray()


def unit_gaussian(wave, mu, sigma):
    """Gaussian with area 1 (so its coefficient is a flux)."""
    return np.exp(-0.5 * ((wave - mu) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)


def is_recomb(label):
    return label.startswith(('H1_', 'He1_', 'He2_'))


def lines_in_range(wave, linelist_dict, z, margin=10.0):
    """Labels of the lines whose observed centre falls inside the spectrum."""
    keep = []
    for label in linelist_dict:
        mu = linelist_dict[label] * (1 + z)
        if wave.min() + margin < mu < wave.max() - margin:
            keep.append(label)
    return keep


def line_templates(wave, labels, linelist_dict, z, kin):
    """Matrix (n_pixels x n_lines): narrow + r * broad for each line,
    normalized to unit NARROW flux."""
    cols = []
    for label in labels:
        mu = linelist_dict[label] * (1 + z)
        s_n = mu * kin['sigma_v_narrow'] / C_KMS
        s_b = mu * kin['sigma_v_broad'] / C_KMS
        r = kin['ratio_H'] if is_recomb(label) else kin['ratio_metal']
        cols.append(unit_gaussian(wave, mu, s_n) + r * unit_gaussian(wave, mu, s_b))
    return np.column_stack(cols)


def solve(A, flux, err):
    """Weighted linear least squares. Returns the coefficients, chi^2 and
    the model."""
    w = 1.0 / err
    coef = np.linalg.lstsq(A * w[:, None], flux * w, rcond=None)[0]
    model = A @ coef
    chi2 = np.sum(((flux - model) * w)**2)
    return coef, chi2, model


# ================================================================
# Kinematics from the previous (lmfit) fits
# ================================================================

def kinematics_from_info(info):
    """Widths and broad-to-narrow ratios to fix in the linear fit, taken
    from the bright-line fits of FINAL_LINES (lines_info.csv): medians over
    the fits that are not on a bound, as in get_fixed_ratios."""
    d = info.copy()
    d['peak_ratio'] = d['b_to_n_ratio'] * d['sigma_v_narrow'] / d['sigma_v_broad']
    ok = (d['b_to_n_ratio'] > 1e-3) & (d['peak_ratio'] < 0.99) & (d['nar_amplitude'] > 0)
    H = d['label'].str.startswith('H1_')
    metal = d['ratio_type'] == 'ratio_metal'
    kin = {'sigma_v_narrow': d.loc[ok, 'sigma_v_narrow'].median(),
           'sigma_v_broad': d.loc[ok, 'sigma_v_broad'].median(),
           'ratio_H': d.loc[ok & H, 'b_to_n_ratio'].median(),
           'ratio_metal': d.loc[ok & metal, 'b_to_n_ratio'].median()}
    print('Fixed kinematics:', end='')
    for k in kin:
        print(f'  {k} = {kin[k]:.3f}', end='')
    print()
    return kin


# ================================================================
# The fit
# ================================================================

def fit_spectrum(wave, flux, err, linelist_dict, z0, kin, dlam=200.0,
                 dv_grid=100.0, dv_step=5.0, rescale=True):
    """
    Best redshift (grid of +-dv_grid km/s around z0) and the joint
    spline + Gaussian fit at that redshift.
    Returns a dict with z, labels, line fluxes (narrow), continuum, model,
    the design matrix and the (rescaled) errors.
    """
    good = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    w = wave[good]
    f = flux[good]
    e = err[good]

    B = spline_basis(w, dlam=dlam)
    n_spl = B.shape[1]

    # redshift grid: the line shapes move, the splines do not
    best = None
    dvs = np.arange(-dv_grid, dv_grid + dv_step / 2, dv_step)
    for dv in dvs:
        z = z0 + dv / C_KMS * (1 + z0)
        labels = lines_in_range(w, linelist_dict, z)
        A = np.column_stack([B, line_templates(w, labels, linelist_dict, z, kin)])
        coef, chi2, model = solve(A, f, e)
        if best is None or chi2 < best['chi2']:
            best = {'z': z, 'labels': labels, 'A': A, 'coef': coef,
                    'chi2': chi2, 'model': model}

    # rescale the errors so that the residuals are ~N(0,1)
    # (robust standard deviation of the normalized residuals)
    scale = 1.0
    if rescale:
        r = (f - best['model']) / e
        mad = np.median(np.abs(r - np.median(r)))
        scale = 1.4826 * mad
        print(f'Error scale factor: {scale:.2f}')
    e = e * scale

    coef = best['coef']
    out = {'z': best['z'],
           'labels': best['labels'],
           'line_flux': coef[n_spl:],
           'spline_coef': coef[:n_spl],
           'wave': w, 'flux': f, 'err': e,
           'continuum': B @ coef[:n_spl],
           'model': best['model'],
           'A': best['A'], 'n_spl': n_spl,
           'chi2_nu': best['chi2'] / (len(f) - len(coef)) / scale**2,
           'err_scale': scale}
    print(f"z = {out['z']:.6f}, {len(out['labels'])} lines, {n_spl} splines, "
          f"chi2_nu (after rescaling) = {out['chi2_nu']:.2f}")
    return out


def monte_carlo(fit, kin, n_mc=100, seed=None):
    """Refit n_mc noisy copies of the spectrum (same z, splines and line
    shapes). Returns a table in the FINAL_LINES lines_info format:
    one row per line and realization."""
    rng = np.random.default_rng(seed)
    A = fit['A']
    n_spl = fit['n_spl']
    labels = fit['labels']
    rows = []
    for i in range(n_mc):
        f_i = fit['flux'] + rng.standard_normal(len(fit['flux'])) * fit['err']
        coef, _, _ = solve(A, f_i, fit['err'])
        for j in range(len(labels)):
            label = labels[j]
            ratio_name = 'ratio_H' if is_recomb(label) else 'ratio_metal'
            r = kin[ratio_name]
            F_n = coef[n_spl + j]
            rows.append({'label': label, 'i': i,
                         'sigma_v_narrow': kin['sigma_v_narrow'],
                         'sigma_v_broad': kin['sigma_v_broad'],
                         'ratio_type': ratio_name,
                         'b_to_n_ratio': r,
                         'nar_amplitude': F_n,
                         'bro_amplitude': r * F_n})
    return pd.DataFrame(rows)


def summarize(info):
    """Median total flux and half the 16-84 range per line (negative fluxes
    are kept: in a linear fit a non-detection scatters around zero)."""
    rows = []
    for label in info['label'].unique():
        d = info[info['label'] == label]
        flux = d['nar_amplitude'] + d['bro_amplitude']
        p16, p50, p84 = np.percentile(flux, [16, 50, 84])
        rows.append({'label': label, 'flux': p50,
                     'fluxerr': (p84 - p16) / 2,
                     'snr': p50 / max((p84 - p16) / 2, 1e-30),
                     'n_fits': len(flux)})
    return pd.DataFrame(rows).set_index('label')


def equivalent_widths(fit, linelist_dict, kin):
    """Rest-frame emission EW = total line flux / continuum at the line
    / (1 + z). Useful for the Balmer-absorption estimate of H-beta."""
    rows = []
    for j in range(len(fit['labels'])):
        label = fit['labels'][j]
        r = kin['ratio_H'] if is_recomb(label) else kin['ratio_metal']
        F = fit['line_flux'][j] * (1 + r)
        mu = linelist_dict[label] * (1 + fit['z'])
        cont = np.interp(mu, fit['wave'], fit['continuum'])
        rows.append({'label': label, 'EW_rest': F / cont / (1 + fit['z'])})
    return pd.DataFrame(rows).set_index('label')


# ================================================================
# Plot
# ================================================================

def plot_fit(fit, linelist_dict, zoom=('H1_4861A', 'O3_5007A', 'H1_6563A'),
             half_width=40.0, save_path=None):
    """Full spectrum with the continuum (top) and zooms on a few lines,
    each with data, continuum, total model and residuals / sigma."""
    import matplotlib.pyplot as plt
    n = len(zoom)
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, n, height_ratios=[1.3, 1, 0.5], hspace=0.5)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(fit['wave'], fit['flux'], color='0.6', lw=0.5, label='data')
    ax0.plot(fit['wave'], fit['continuum'], color='C3', lw=1.5,
             label='spline continuum')
    top = np.nanpercentile(fit['continuum'], 99) * 4
    ax0.set_ylim(np.nanpercentile(fit['continuum'], 1) * 0.5, top)
    ax0.set_xlabel(r'Observed wavelength ($\AA$)')
    ax0.legend(frameon=False)

    for k in range(n):
        label = zoom[k]
        mu = linelist_dict[label] * (1 + fit['z'])
        m = np.abs(fit['wave'] - mu) < half_width
        ax = fig.add_subplot(gs[1, k])
        axr = fig.add_subplot(gs[2, k], sharex=ax)
        ax.step(fit['wave'][m], fit['flux'][m], where='mid', color='k', lw=0.8)
        ax.plot(fit['wave'][m], fit['continuum'][m], color='C3')
        ax.plot(fit['wave'][m], fit['model'][m], color='C0')
        ax.set_title(label)
        res = (fit['flux'][m] - fit['model'][m]) / fit['err'][m]
        axr.step(fit['wave'][m], res, where='mid', color='k', lw=0.8)
        axr.axhspan(-1, 1, color='0.85')
        axr.set_ylim(-6, 6)
        axr.set_ylabel(r'res / $\sigma$')
    if save_path is not None:
        fig.savefig(save_path, dpi=130, bbox_inches='tight')
    return fig


# ================================================================
# Run for one galaxy
# ================================================================

def run_galaxy(class_, wave, flux, err, info_bright, dlam=200.0, n_mc=100,
               out_dir=None, seed=None):
    """
    class_      : the SPECTRALDATA object (linelist_dict, redshift, gal_id)
    wave, flux, err : the spectrum (e.g. read_data(class_) from FINAL_LINES:
                  MW + Balmer-absorption corrected)
    info_bright : the lines_info table of the previous bright-line fits,
                  used only to fix the widths and ratios
    Saves {gal}_lines_info_spline.csv and {gal}_lines_summ_spline.csv
    (and the continuum) if out_dir is given.
    """
    kin = kinematics_from_info(info_bright)
    fit = fit_spectrum(wave, flux, err, class_.linelist_dict,
                       class_.redshift, kin, dlam=dlam)
    info = monte_carlo(fit, kin, n_mc=n_mc, seed=seed)
    summ = summarize(info)
    if out_dir is not None:
        g = class_.gal_id
        info.to_csv(f'{out_dir}/{g}_lines_info_spline.csv', index=False)
        summ.to_csv(f'{out_dir}/{g}_lines_summ_spline.csv')
        pd.DataFrame({'wave': fit['wave'], 'continuum': fit['continuum'],
                      'model': fit['model']}).to_csv(
            f'{out_dir}/{g}_continuum_spline.csv', index=False)
    return fit, info, summ
