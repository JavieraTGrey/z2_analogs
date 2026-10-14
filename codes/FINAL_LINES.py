import astropy.constants as const
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, PolynomialModel
import pandas as pd
import os

main_DIR = "/Users/javieratoro/Desktop/thesis/"
proj_DIR = f"{main_DIR}proyecto 2024-2/"
code_DIR = f"{proj_DIR}codes"
os.chdir(code_DIR)


def get_line_groups():
    bright_lines = ['O2_3726A', 'O2_3729A', 'H1_6563A',
                    'H1_4861A', 'H1_4340A', 'O3_5007A',
                    'O3_4959A', 'N2_6548A', 'N2_6583A',
                    'S2_6716A', 'S2_6731A', 'He1_5016A']

    bright_lines_group = [('O2_3726A', 'O2_3729A'),
                          ('H1_4340A'), ('H1_4861A'),
                          ('O3_4959A'), ('O3_5007A', 'He1_5016A'),
                          ('N2_6548A', 'H1_6563A', 'N2_6583A'),
                          ('S2_6716A', 'S2_6731A')]
    return bright_lines, bright_lines_group

# -----------------------------------------
# Data Preparation
# -----------------------------------------


# Read data for galaxies
def read_data(class_):
    """
    Reads Balmer absorption corrected data if it exists, otherwise reads the
    MW dust corrected data, and finally if none exists reads the
    uncorrected data.

    Eventually change to dust corrected data.

    """
    gal_id = class_.names[0][:5]
    print(f'Reading data for {gal_id}')
    file_path_balmer = f'{proj_DIR}new_results/bcorr_{gal_id}_new.csv'
    file_path_MW = f'{proj_DIR}dust/{gal_id}/dcorr_{gal_id}.csv'

    # Balmer + MW corrected data
    if os.path.exists(file_path_balmer):
        print("Using Balmer Absorption + MW dust corrected data")
        bcorr = pd.read_csv(file_path_balmer)
        arrays = bcorr.to_numpy().T
        wave, flux, sigma = arrays
    # MW dust corrected data
    elif os.path.exists(file_path_MW):
        print("Using MW Dust corrected data")
        dcorr = pd.read_csv(file_path_MW)
        arrays = dcorr.to_numpy().T
        wave, flux, sigma = arrays

    # Uncorrected data
    else:
        print("Using uncorrected data")
        wave, flux, sigma, _ = class_.datas[0]
    return wave, flux, sigma


# Read sigma from old measurement for masking
def read_sigma(class_):
    gal_id = class_.names[0][:5]
    path_old = f'{proj_DIR}lines/{gal_id}/{gal_id}_iter_.csv'
    if os.path.exists(path_old):
        old_fit = pd.read_csv(path_old)
        sigma_narrow = np.mean(old_fit['sigma_v_narrow'])
        sigma_broad = np.mean(old_fit['sigma_v_broad'])
        return sigma_narrow, sigma_broad
    else:
        raise FileNotFoundError(f'File not found: {path_old}')


# Mask emission lines other than specific line
def mask_nonuse_emission_line(class_, lines, data):
    """
    Mask every emission line other than `lines`, 2.5 sigma around its
    center, setting flux to NaN and sigma to 0 in that window.

    """
    wave, flux, err = data

    cte = (1 + class_.redshift)
    masked_flux = flux.copy()
    masked_sigma = err.copy()
    sigmas = read_sigma(class_)

    bright_lines = ['O2_3726A', 'O2_3729A', 'H1_6563A', 'H1_4861A', 'H1_4340A',
                    'O3_5007A', 'O3_4959A', 'N2_6548A', 'N2_6583A', 'S2_6716A',
                    'S2_6731A']

    for label in class_.linelist_dict:
        if label in lines:
            continue
        center = class_.linelist_dict[label] * cte

        if label in bright_lines:
            sigma = (center / const.c.to('km/s').value) * sigmas[1]
        else:
            sigma = (center / const.c.to('km/s').value) * sigmas[0]

        size = 2.5
        mask_line = (wave > center - size * sigma) & \
                    (wave < center + size * sigma)

        masked_flux[mask_line] = np.nan
        masked_sigma[mask_line] = 0.0

    return wave, masked_flux, masked_sigma


# Create stamp for each line we are measuring
def get_stamp(class_, data, lines):
    wave, flux, err = data[0], data[1], data[2]
    lines_df = class_.line_list
    cte = (1 + class_.redshift)
    stamps = {}
    for line in lines:
        stamp = flux.copy()

        info_line = lines_df[lines_df['name'] == line]
        w2, w5 = info_line[['w2', 'w5']].values[0]
        mask_stamp = (wave > w2*cte) & (wave < w5*cte)

        stamps[line] = (wave[mask_stamp], stamp[mask_stamp], err[mask_stamp])
    return stamps


# -----------------------------------------
# Model function
# -----------------------------------------

# Check if is a recombination line
def is_recomb(label):
    return label.startswith(('H1_', 'He1_', 'He2_'))


# Model for a list of lines (could be just one)
def model_flux_list(class_, stamp, linelist, deg=1,
                    ratio=None, plot=False):
    z_init = class_.redshift
    linelist_dict = class_.linelist_dict

    narrow_gaussians = [
        GaussianModel(prefix=f'{label}_narrow_') for label in linelist]
    broad_gaussians = [
        GaussianModel(prefix=f'{label}_broad_') for label in linelist]

    polydeg = deg
    polynomial = PolynomialModel(degree=polydeg)

    model = narrow_gaussians[0] + broad_gaussians[0]
    for n, b in zip(narrow_gaussians[1:], broad_gaussians[1:]):
        model += n + b
    model += polynomial

    pars = model.make_params()

    pars.add('sigma_v_narrow', value=50, min=20, max=80, vary=True)

    pars.add('sigma_v_broad', value=80, min=65, max=300, vary=True)

    if ratio is None:
        pars.add('ratio_H', value=0.4, min=0.0, max=5, vary=True)
        pars.add('ratio_metal', value=0.4, min=0.0, max=5, vary=True)
    else:
        pars.add('ratio_H', value=ratio[0], vary=False)
        pars.add('ratio_metal', value=ratio[1], vary=False)

    for label in linelist:

        pars.add(f'z_{label}', value=z_init, vary=True,
                 min=z_init*0.75, max=z_init*1.25)

        lam0 = linelist_dict[label]

        pars.add(f'{label}_narrow_center', value=lam0, vary=False,
                 expr=f'{lam0}*(1+z_{label})')
        pars.add(f'{label}_narrow_amplitude', value=50, vary=True, min=0.0)
        pars.add(f'{label}_narrow_sigma', vary=False,
                 expr=f'(sigma_v_narrow/3e5)*{label}_narrow_center')

        pars.add(f'{label}_broad_center', value=lam0, vary=False,
                 expr=f'{lam0}*(1+z_{label})')

        r = 'ratio_H' if is_recomb(label) else 'ratio_metal'
        pars.add(f'{label}_broad_amplitude', vary=False,
                 expr=f'{r}*{label}_narrow_amplitude')

        pars.add(f'{label}_broad_sigma', vary=False,
                 expr=f'(sigma_v_broad/3e5)*{label}_broad_center')

    for i in range(polydeg + 1):
        pars[f'c{i}'].set(value=1.0, vary=True)

    out = model.fit(stamp[1], pars, x=stamp[0],
                    nan_policy='omit', max_nfev=1000)

    if plot:
        plotting_fit(stamp[0], stamp[1], stamp[2], out, linelist, class_)
    return out


# MC model running on a list of lines (can be individual)
def MC_model_from_lines(class_, stamps, linelist, n_mc=100,
                        deg=1, ratio=None, plot=False):

    wave, flux, err = stamps[linelist[0]]

    fits = []
    for i in range(n_mc):
        yoff = flux + np.random.randn(len(flux)) * err
        fit = model_flux_list(class_, (wave, yoff, err), linelist,
                              deg=deg, ratio=ratio, plot=plot)
        fits.append(fit)

    return fits


# Create dict with MC fits of every line in list
def create_dict(class_, stamps, linelist_all, n_mc=100,
                deg=1, ratio=None, plot=False):
    dict_ = {}
    for linelist in linelist_all:
        linelist_ = np.asarray(linelist)
        if linelist_.size == 1:
            linelist = [linelist]

        fits = MC_model_from_lines(class_, stamps, linelist, n_mc=n_mc,
                                   deg=deg, ratio=ratio, plot=plot)
        dict_[linelist[0]] = fits
    return dict_


# -----------------------------------------
# Save information
# -----------------------------------------


def _as_list(lines):
    """('H1_4340A') is just a string, so wrap single lines in a list."""
    if np.asarray(lines).size == 1:
        return [lines] if isinstance(lines, str) else list(lines)
    return list(lines)


def fits_to_rows(dict_, linelist):
    """One row per (line, MC realization) with the fitted values."""
    rows = []
    for lines in linelist:
        lines = _as_list(lines)
        fits = dict_[lines[0]]
        for line in lines:
            ratio_name = 'ratio_H' if is_recomb(line) else 'ratio_metal'
            for i, fit in enumerate(fits):
                p = fit.params
                rows.append({
                    'label': line,
                    'i': i,
                    'sigma_v_narrow': p['sigma_v_narrow'].value,
                    'sigma_v_broad':  p['sigma_v_broad'].value,
                    'ratio_type':     ratio_name,
                    'b_to_n_ratio':   p[ratio_name].value,
                    'nar_amplitude':  p[f'{line}_narrow_amplitude'].value,
                    'bro_amplitude':  p[f'{line}_broad_amplitude'].value,
                })
    return rows


def add_derived(info):
    """Peak-height ratio, broad fraction and a flag for bad ratios."""
    info = info.copy()
    info['flux'] = info['nar_amplitude'] + info['bro_amplitude']
    info['peak_ratio'] = (info['b_to_n_ratio']
                          * info['sigma_v_narrow'] / info['sigma_v_broad'])
    info['broad_frac'] = info['bro_amplitude'] / info['flux']
    # ratio collapsed to ~0 or broad peak as tall as narrow: not a measurement
    info['ratio_on_bound'] = ((info['b_to_n_ratio'] < 1e-3) |
                              (info['peak_ratio'] >= 0.99))
    return info


def summarize(info):
    """Median of the good MC fits per line
      plus a 16-84 percentile flux error."""
    res = add_derived(info)
    # drop only failed fits; keep ratios on a bound, their flux is still valid
    res = res[np.isfinite(res['nar_amplitude']) & (res['nar_amplitude'] > 0)]

    cols = ['sigma_v_narrow', 'sigma_v_broad', 'b_to_n_ratio',
            'peak_ratio', 'broad_frac',
            'nar_amplitude', 'bro_amplitude', 'flux']

    g = res.groupby('label')
    summary = g[cols].median()
    q = g['flux'].quantile([0.16, 0.84]).unstack()
    summary['fluxerr'] = (q[0.84] - q[0.16]) / 2
    summary['n_fits'] = g.size()
    summary['frac_on_bound'] = g['ratio_on_bound'].mean()
    return summary


def get_fixed_ratios(info):
    """Median ratio_H (H I lines) and ratio_metal from fits not on a bound."""
    d = add_derived(info)
    ok = (~d['ratio_on_bound']) & (d['nar_amplitude'] > 0)

    H = d['label'].str.startswith('H1_')
    metal = d['ratio_type'] == 'ratio_metal'

    med_H = d.loc[ok & H, 'b_to_n_ratio'].median()
    med_metal = d.loc[ok & metal, 'b_to_n_ratio'].median()
    print(f'ratio_H = {med_H:.3f} (from {(ok & H).sum()} fits), '
          f'ratio_metal = {med_metal:.3f} (from {(ok & metal).sum()} fits)')
    return med_H, med_metal


def _paths(class_):
    gal_id = class_.names[0][:5]
    out_dir = f'{proj_DIR}new_results/{gal_id}'
    os.makedirs(out_dir, exist_ok=True)
    return (f'{out_dir}/{gal_id}_lines_info.csv',
            f'{out_dir}/{gal_id}_lines_summ.csv')


def save_tables(class_, info, summary):
    info_path, summ_path = _paths(class_)
    info.to_csv(info_path, index=False)
    summary.to_csv(summ_path)          # keeps 'label' as the index


def load_tables(class_):
    info_path, summ_path = _paths(class_)
    info = pd.read_csv(info_path)
    # older files were saved with the index as an extra column
    info = info.drop(columns=[c for c in info.columns
                              if c.startswith('Unnamed')])
    summary = pd.read_csv(summ_path, index_col='label')
    return info, summary


def save_data_dict(class_, dict_, linelist):
    info = pd.DataFrame(fits_to_rows(dict_, linelist))
    summary = summarize(info)
    save_tables(class_, info, summary)
    return info, summary

# -----------------------------------------
# Running algorithm
# -----------------------------------------


def measure_all_lines(class_, linelist_all, linelist_group, n_mc=100, deg=1,
                      ratio=None, plot=False):
    data = read_data(class_)
    masked_data = mask_nonuse_emission_line(class_, linelist_all, data)
    print('Getting stamps')
    stamps = get_stamp(class_, masked_data, linelist_all)
    print('MC line modeling')
    dict_all = create_dict(class_, stamps, linelist_group, n_mc=n_mc,
                           deg=deg, ratio=ratio, plot=plot)
    print('Saving information')
    info, summary = save_data_dict(class_, dict_all, linelist_group)
    print('Done')
    return info, summary


def measure_other_lines(class_, other_lines=None, n_mc=100, deg=1,
                        plot=False):

    info, _ = load_tables(class_)
    ratio = get_fixed_ratios(info)

    done = set(info['label'])
    if other_lines is None:
        other_lines = list(class_.linelist_dict)
    other_lines = [label for label in other_lines if label not in done]

    data = read_data(class_)
    wave = data[0]
    cte = 1 + class_.redshift
    lines = [label for label in other_lines
             if np.nanmin(wave) < class_.linelist_dict[label] * cte
             < np.nanmax(wave)]
    skipped = sorted(set(other_lines) - set(lines))
    if skipped:
        print(f'Outside the spectrum, skipped: {skipped}')
    if not lines:
        print('No new lines to fit')
        return info, summarize(info)

    masked = mask_nonuse_emission_line(class_, lines, data)
    print(f'MC fitting {len(lines)} lines together, ratio fixed')
    fits = MC_model_from_lines(class_, {lines[0]: masked}, lines,
                               n_mc=n_mc, deg=deg, ratio=ratio, plot=plot)

    new_info = pd.DataFrame(fits_to_rows({lines[0]: fits}, [lines]))
    info = pd.concat([info, new_info], ignore_index=True)

    summary = summarize(info)
    save_tables(class_, info, summary)
    print('Done')
    return info, summary


def measure_everything(class_, n_mc=100, plot=False):

    bright_lines, bright_groups = get_line_groups()
    _ = fit_quality_table(class_, bright_lines, bright_groups, plot=False)

    print('--- Step 1: bright lines (free ratio) ---')
    measure_all_lines(class_, bright_lines, bright_groups,
                      n_mc=n_mc, deg=1, ratio=None, plot=plot)

    print('--- Step 2: other lines (fixed ratio) ---')
    info, summary = measure_other_lines(class_,
                                        other_lines=class_.line_name.values,
                                        n_mc=n_mc, deg=5, plot=plot)
    return info, summary

# -----------------------------------------
# Plotting functions
# -----------------------------------------


def plotting_fit(lams, flux, err, fit, linelist, class_):
    fig2, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(8, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
    )

    comps = fit.eval_components(x=lams)
    best_fit = comps['polynomial'].copy()
    for line in linelist:
        best_fit += (comps[line + '_narrow_'] +
                     comps[line + '_broad_'])

    # --- Main spectrum ---
    narrow_key = [key for key, _ in comps.items() if 'narrow' in key]
    broad_key = [key for key, _ in comps.items() if 'broad' in key]

    ax.plot(lams, flux, lw=1, drawstyle='steps-mid',
            label='Observed spectrum')

    ax.plot(lams, best_fit, lw=1, drawstyle='steps-mid',
            label='Best model')

    ax.plot(lams, comps['polynomial'], 'r-',
            label='Polynomial component',
            lw=1, drawstyle='steps-mid', alpha=0.3)

    for i, broad_line in enumerate(broad_key):
        ax.plot(lams, comps[broad_line] + comps['polynomial'],
                'k--',
                label='Broad component' if i == len(broad_key)-1 else None,
                alpha=0.3)

    for i, narrow_line in enumerate(narrow_key):
        ax.plot(lams, comps[narrow_line] + comps['polynomial'],
                'g--',
                label='Narrow component' if i == len(narrow_key)-1 else None,
                alpha=0.3)

    # --- Residuals ---
    residuals = (flux - best_fit)

    ax_res.plot(lams, residuals,
                color='black',
                lw=1,
                drawstyle='steps-mid')

    ax_res.fill_between(lams, -3*err, 3*err,
                        color='grey',
                        alpha=0.2, step='mid')

    ax_res.axhline(0, color='grey', linestyle='--', lw=1)

    ax_res.set_ylabel(r'Residuals', size=16)
    ax_res.set_xlabel(r'Obs. Wavelength ($\AA$)', size=20)
    # ax_res.set_ylim(-40, 40)

    # --- Emission line markers ---
    for label in linelist:
        obs_lam = (class_.linelist_dict[label] *
                   (1 + fit.params[f'z_{label}'].value))

        ax.axvline(obs_lam, linestyle='--',
                   linewidth=0.5, color='grey')

        ax.text(obs_lam, 0.99, '\n'+label, rotation=90, ha='center', va='top',
                color='k', size=8, transform=ax.get_xaxis_transform())

    # --- Formatting ---
    ylabel = r'Flux ($10^{-17}\ \mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$)'
    ax.set_ylabel(ylabel, size=20)

    ax.legend(fontsize=14, frameon=True)
    ax_res.tick_params(
        axis='both',
        which='major',
        labelsize=16,
        width=2,
        length=6
    )
    ax.tick_params(
        axis='both',
        which='major',
        labelsize=16,
        width=2,
        length=6, labelbottom=False
    )
    plt.show()
    return fig2


# -----------------------------------------
# Checking uncertainties
# -----------------------------------------


def fit_quality(class_, out, stamp, linelist, edge_frac=0.15):
    """Per-line check: data vs model integrated over each line's window."""
    wave, flux, err = stamp
    best_fit = out.eval(x=wave)
    comps = out.eval_components(x=wave)
    dlam = np.gradient(wave)
    p = out.params

    # continuum from the outer edges of the stamp
    n = len(wave)
    k = max(3, int(edge_frac * n))
    side = np.zeros(n, bool)
    side[:k] = side[-k:] = True
    ok = side & np.isfinite(flux)
    cont = np.polyval(np.polyfit(wave[ok], flux[ok], 1), wave)

    # all fitted line emission (no polynomial)
    lines_model = np.sum([comps[f'{l}_narrow_'] + comps[f'{l}_broad_']
                          for l in linelist], axis=0)

    # goodness of fit for the whole group
    good = np.isfinite(flux) & (err > 0)
    resid = (flux - best_fit)[good] / err[good]
    redchi = np.sum(resid**2) / (good.sum() - out.nvarys)
    max_res = np.max(np.abs(resid))

    # windows: split at midpoints between observed line centers
    centers = {l: class_.linelist_dict[l] * (1 + p[f'z_{l}'].value)
               for l in linelist}
    ordered = sorted(linelist, key=centers.get)
    mids = [0.5 * (centers[a] + centers[b])
            for a, b in zip(ordered[:-1], ordered[1:])]
    bounds = [-np.inf] + mids + [np.inf]

    rows = []
    for l, lo, hi in zip(ordered, bounds[:-1], bounds[1:]):
        win = ~side & (wave >= lo) & (wave < hi)
        direct = np.nansum(((flux - cont) * dlam)[win])
        model_win = np.nansum((lines_model * dlam)[win])
        rows.append({
            'line': l,
            'fitted_flux': (p[f'{l}_narrow_amplitude'].value
                            + p[f'{l}_broad_amplitude'].value),
            'direct_over_model': direct / model_win,
            'redchi': redchi,
            'max_abs_resid_sigma': max_res,
        })
    return rows


def fit_quality_table(class_, linelist_all, linelist_group, deg=1, plot=True):
    data = read_data(class_)
    masked = mask_nonuse_emission_line(class_, linelist_all, data)
    stamps = get_stamp(class_, masked, linelist_all)

    rows = []
    for group in linelist_group:
        group = _as_list(group)
        stamp = stamps[group[0]]
        out = model_flux_list(class_, stamp, group, deg=deg)
        stats = fit_quality(class_, out, stamp, group)
        rows.extend(stats)
        if plot:
            plotting_fit(stamp[0], stamp[1], stamp[2], out, group, class_)

    df = pd.DataFrame(rows)
    gal_id = class_.names[0][:5]
    df.to_csv(f'{proj_DIR}new_results/{gal_id}/{gal_id}_fit_quality.csv',
              index=False)
    return df
