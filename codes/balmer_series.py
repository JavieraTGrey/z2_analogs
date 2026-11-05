"""
Joint fit of the high-order Balmer absorption series (H14 ... H_delta)
with ONE polynomial continuum for the whole region.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial import legendre
from scipy.optimize import lsq_linear

import balmer_abs_EW as bal

C_KMS = 299792.458
CA_K_VAC = 3934.777

# vacuum wavelengths (A) of the metal absorbers
METAL_LINES_VAC = {
    'CaK_3934A': [3934.777],
    'FeI_3861A': [3861.00],
    'FeI_4047A': [4046.96],
    'FeI_4065A': [4064.74],
    'FeI_4073A': [4072.86],
    'SrII_4079A': [4078.86],
    'AlI_3945A': [3945.12],
    'AlI_3963A': [3962.64],
    # around H_gamma
    'CaI_4228A': [4227.92],
    'FeI_4273A': [4272.96],
    'FeI_4327A': [4326.98],
    'FeI_4385A': [4384.78],
    'FeI_4406A': [4405.98],
}
MG_TRIPLET_VAC = [3830.44, 3833.39, 3839.38]
CN_BAND_VAC = (3840.0, 3884.1)     # ramp start, bandhead
CH_BAND_VAC = (4281.0, 4315.2)     # G band (CH): ramp start, bandhead


def series_absorbers(class_, rest_window, metals=True, mg_triplet=True,
                     cn_band=True):
    """
    List of absorbers inside the window. Each one is a dict:
      name, kind ('balmer' | 'metal' | 'cn'), rest (list of rest
      wavelengths; for 'cn' = (start, bandhead)).
    """
    lo, hi = rest_window
    absorbers = []
    for label in class_.linelist_dict:
        lam = class_.linelist_dict[label]
        if label.startswith('H1_') and lo <= lam <= hi:
            absorbers.append({'name': label, 'kind': 'balmer',
                              'rest': [lam]})
    if metals:
        for name in METAL_LINES_VAC:
            lams = METAL_LINES_VAC[name]
            if lo <= lams[0] <= hi:
                absorbers.append({'name': name, 'kind': 'metal',
                                  'rest': lams})
    elif lo <= CA_K_VAC <= hi:
        absorbers.append({'name': 'CaK_3934A', 'kind': 'metal',
                          'rest': [CA_K_VAC]})
    if metals and mg_triplet and lo <= MG_TRIPLET_VAC[0] <= hi:
        absorbers.append({'name': 'MgI_3833A', 'kind': 'metal',
                          'rest': MG_TRIPLET_VAC})
    if metals and cn_band and lo <= CN_BAND_VAC[0] and CN_BAND_VAC[1] <= hi:
        absorbers.append({'name': 'CN_3883', 'kind': 'cn',
                          'rest': list(CN_BAND_VAC)})
    if metals and cn_band and lo <= CH_BAND_VAC[0] and CH_BAND_VAC[1] <= hi:
        absorbers.append({'name': 'CH_4300', 'kind': 'cn',
                          'rest': list(CH_BAND_VAC)})
    return absorbers


GAMMA_PER_SIGMA = np.sqrt(2 * np.log(2))


def balmer_shape(wave, mu, sig, eta):
    gamma = GAMMA_PER_SIGMA * sig
    gauss = np.exp(-0.5 * ((wave - mu) / sig)**2)
    lorentz = 1.0 / (1.0 + ((wave - mu) / gamma)**2)
    return (1 - eta) * gauss + eta * lorentz


def balmer_area(sig, eta):
    gamma = GAMMA_PER_SIGMA * sig
    return (1 - eta) * np.sqrt(2 * np.pi) * sig + eta * np.pi * gamma


def absorber_profile(ab, wave, z, vel, vel_metal, shift=0.0, eta=0.0):
    """Unit-depth profile (positive) of one absorber on `wave`, and the
    integral of that profile in A (observed), used for the EW."""
    if ab['kind'] == 'cn':
        a = (ab['rest'][0] + shift) * (1 + z)
        b = (ab['rest'][1] + shift) * (1 + z)
        prof = np.where((wave > a) & (wave < b), (wave - a) / (b - a), 0.0)
        return prof, 0.5 * (b - a), 0.5 * (a + b)
    v = vel if ab['kind'] == 'balmer' else vel_metal
    e = eta if ab['kind'] == 'balmer' else 0.0
    prof = np.zeros(len(wave))
    integral = 0.0
    centre = 0.0
    n = len(ab['rest'])
    for lam in ab['rest']:
        mu = (lam + shift) * (1 + z)
        sig = mu * v / C_KMS
        prof += balmer_shape(wave, mu, sig, e) / n
        integral += balmer_area(sig, e) / n
        centre += mu / n
    return prof, integral, centre


def emission_mask(class_, wave, sigmas, core_size):
    """True where any emission line core is (to be masked)."""
    z = class_.redshift
    mask = np.zeros(len(wave), dtype=bool)
    for label in class_.linelist_dict:
        center = class_.linelist_dict[label] * (1 + z)
        if center < wave[0] - 50 or center > wave[-1] + 50:
            continue
        sig_em = bal.get_sigma(label, class_, sigmas)
        mask |= (wave > center - core_size * sig_em) & \
                (wave < center + core_size * sig_em)
    return mask


def design_matrix(wave, absorbers, z, vel, vel_metal, deg, shift=0.0,
                  eta=0.0):
    """Columns: Legendre P_0..P_deg, then -profile for every absorber."""
    x = 2 * (wave - wave[0]) / (wave[-1] - wave[0]) - 1
    cols = []
    leg = legendre.legvander(x, deg)
    for j in range(deg + 1):
        cols.append(leg[:, j])
    for ab in absorbers:
        prof, _, _ = absorber_profile(ab, wave, z, vel, vel_metal, shift,
                                      eta)
        cols.append(-prof)
    return np.column_stack(cols)


def solve(wave, flux, err, good, absorbers, z, vel, vel_metal, deg,
          shift=0.0, positive_balmer=False, eta=0.0):
    """Weighted linear least squares. Polynomial free; metal/CN depths
    >= 0; Balmer depths free unless positive_balmer."""
    A = design_matrix(wave, absorbers, z, vel, vel_metal, deg, shift, eta)
    Aw = A[good] / err[good][:, None]
    bw = flux[good] / err[good]
    lower = np.full(A.shape[1], -np.inf)
    for k in range(len(absorbers)):
        if absorbers[k]['kind'] != 'balmer' or positive_balmer:
            lower[deg + 1 + k] = 0.0
    if np.all(np.isinf(lower)):
        coef, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)
    else:
        coef = lsq_linear(Aw, bw, bounds=(lower, np.inf),
                          method='bvls').x
    resid = bw - Aw @ coef
    return coef, np.sum(resid**2), A


def best_vel(wave, flux, err, good, absorbers, z, vel_grid, vel_metal_grid,
             deg, positive_balmer=False, eta_grid=(0.0,)):
    """
    Profile over the Balmer velocity grid, the metal velocity grid and the
    Balmer shape grid (eta). Returns (vel, coef, chi2, A, vel_metal, eta)
    of the best combination, and the chi2 profile over the Balmer velocity
    grid (minimised over the other two).
    vel_metal_grid can be a single number (fixed width).
    """
    vm_grid = np.atleast_1d(np.asarray(vel_metal_grid, dtype=float))
    et_grid = np.atleast_1d(np.asarray(eta_grid, dtype=float))
    best = None
    chi2_list = []
    for vel in vel_grid:
        best_here = np.inf
        for vm in vm_grid:
            for et in et_grid:
                coef, chi2, A = solve(wave, flux, err, good, absorbers, z,
                                      vel, vm, deg,
                                      positive_balmer=positive_balmer,
                                      eta=et)
                if chi2 < best_here:
                    best_here = chi2
                if best is None or chi2 < best[2]:
                    best = (vel, coef, chi2, A, vm, et)
        chi2_list.append(best_here)
    return best, np.array(chi2_list)


def eta_profile(wave, flux, err, good, absorbers, z, vel, vel_metal_grid,
                deg, eta_grid, positive_balmer=False):
    vm_grid = np.atleast_1d(np.asarray(vel_metal_grid, dtype=float))
    out = []
    for et in eta_grid:
        best_here = np.inf
        for vm in vm_grid:
            _, chi2, _ = solve(wave, flux, err, good, absorbers, z, vel, vm,
                               deg, positive_balmer=positive_balmer, eta=et)
            if chi2 < best_here:
                best_here = chi2
        out.append(best_here)
    return np.array(out)


def ews_from_coef(coef, wave, absorbers, z, vel, vel_metal, deg, shift=0.0,
                  eta=0.0):
    """Rest-frame EW of every absorber from one solution."""
    ews = np.zeros(len(absorbers))
    heights = coef[deg + 1:]
    for k in range(len(absorbers)):
        _, integral, centre = absorber_profile(absorbers[k], wave, z, vel,
                                               vel_metal, shift, eta)
        xc = 2 * (centre - wave[0]) / (wave[-1] - wave[0]) - 1
        cont = legendre.legval(xc, coef[:deg + 1])
        ews[k] = heights[k] * integral / cont / (1 + z)
    return ews, heights


def _prepare(class_, data, sigmas, rest_window, core_size, metals,
             mg_triplet, cn_band):
    z = class_.redshift
    wave_all, flux_all, err_all = data
    sel = ((wave_all >= rest_window[0] * (1 + z))
           & (wave_all <= rest_window[1] * (1 + z)))
    wave = wave_all[sel]
    flux = flux_all[sel]
    err = err_all[sel].copy()
    absorbers = series_absorbers(class_, rest_window, metals=metals,
                                 mg_triplet=mg_triplet, cn_band=cn_band)
    em_mask = emission_mask(class_, wave, sigmas, core_size)
    good = (~em_mask) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    return wave, flux, err, good, absorbers, z


def fit_balmer_series(class_, data, sigmas, rest_window=(3730., 4160.),
                      deg=4, vel_grid=None, vel_metal=None, core_size=4.0,
                      n_mc=1000, rescale=True, metals=True, mg_triplet=True,
                      cn_band=True, positive_balmer=False, seed=None,
                      eta_grid=None):
    """
    data = (wave, flux, err), MW-corrected and NOT yet corrected for
    Balmer absorption in this window.

    Returns a dict with the EW draws of every absorber, the Balmer
    velocity draws, the best fit, and what is needed for plots and for
    the correction of the spectrum.
    """
    if vel_grid is None:
        vel_grid = np.arange(250., 1001., 50.)
    if vel_metal is None:
        vel_metal = np.arange(100., 201., 25.)
    if eta_grid is None:
        eta_grid = np.array([0.0])
    eta_grid = np.atleast_1d(np.asarray(eta_grid, dtype=float))
    rng = np.random.default_rng(seed)

    wave, flux, err, good, absorbers, z = _prepare(
        class_, data, sigmas, rest_window, core_size, metals, mg_triplet,
        cn_band)
    names = []
    for ab in absorbers:
        names.append(ab['name'])

    # --- best fit on the data ---
    (vel0, coef0, chi2_0, A0, vm0, eta0), chi2_grid = best_vel(
        wave, flux, err, good, absorbers, z, vel_grid, vel_metal, deg,
        positive_balmer, eta_grid)
    dof = np.sum(good) - (deg + 1) - len(absorbers) - 2
    chi2_nu = chi2_0 / dof
    scale = np.sqrt(chi2_nu) if (rescale and chi2_nu > 1) else 1.0
    err_mc = err * scale
    chi2_eta = eta_profile(wave, flux, err, good, absorbers, z, vel0,
                           vel_metal, deg, eta_grid, positive_balmer)
    print(f'Series fit: {len(absorbers)} absorbers, deg {deg}, best v = '
          f'{vel0:.0f} km/s (metals {vm0:.0f} km/s, eta {eta0:.2f}), '
          f'chi2_nu = {chi2_nu:.2f} (errors x {scale:.2f} for the MC)')
    if len(eta_grid) > 1:
        txt = []
        for j in range(len(eta_grid)):
            txt.append(f'{eta_grid[j]:.2f}: '
                       f'{(chi2_eta[j] - chi2_eta.min()) / chi2_nu:.1f}')
        print('  delta chi2 (rescaled) vs eta -> ' + ', '.join(txt))

    # --- Monte Carlo ---
    ew_draws = np.zeros((n_mc, len(absorbers)))
    h_draws = np.zeros((n_mc, len(absorbers)))
    vel_draws = np.zeros(n_mc)
    vm_draws = np.zeros(n_mc)
    eta_draws = np.zeros(n_mc)
    model0 = A0 @ coef0
    for i in range(n_mc):
        fake = model0 + rng.standard_normal(len(wave)) * err_mc
        (vel_i, coef_i, _, _, vm_i, eta_i), _ = best_vel(
            wave, fake, err_mc, good, absorbers, z, vel_grid, vel_metal,
            deg, positive_balmer, eta_grid)
        ews, heights = ews_from_coef(coef_i, wave, absorbers, z, vel_i,
                                     vm_i, deg, eta=eta_i)
        ew_draws[i] = ews
        h_draws[i] = heights
        vel_draws[i] = vel_i
        vm_draws[i] = vm_i
        eta_draws[i] = eta_i

    ew0, _ = ews_from_coef(coef0, wave, absorbers, z, vel0, vm0, deg,
                           eta=eta0)
    rows = []
    for k in range(len(absorbers)):
        med = np.median(ew_draws[:, k])
        std = np.std(ew_draws[:, k])
        ratio = bal.EW_all_ratios.get(names[k], np.nan)
        rows.append({'line': names[k], 'kind': absorbers[k]['kind'],
                     'EW_best': ew0[k], 'EW': med, 'EW_std': std,
                     'SN': med / std if std > 0 else 0,
                     'EW_Hb_equiv': med / ratio})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format='%.2f'))
    print(f'  Balmer v = {np.median(vel_draws):.0f} +- '
          f'{np.std(vel_draws):.0f} km/s, metals v = '
          f'{np.median(vm_draws):.0f} +- {np.std(vm_draws):.0f} km/s, '
          f'eta = {np.median(eta_draws):.2f} +- {np.std(eta_draws):.2f}')

    ew_dict = {}
    for k in range(len(absorbers)):
        ew_dict[names[k]] = ew_draws[:, k]

    return {'names': names, 'absorbers': absorbers, 'ew_draws': ew_dict,
            'height_draws': h_draws, 'vel_draws': vel_draws,
            'vel_best': vel0, 'vel_metal': vm0, 'vel_metal_draws': vm_draws,
            'eta_best': eta0, 'eta_draws': eta_draws, 'eta_grid': eta_grid,
            'chi2_eta': chi2_eta, 'coef_best': coef0,
            'deg': deg, 'wave': wave, 'flux': flux, 'err': err,
            'good': good, 'chi2_nu': chi2_nu, 'chi2_grid': chi2_grid,
            'vel_grid': vel_grid, 'table': table, 'z': z}


def series_absorption(res, wave, kinds=('balmer', 'metal', 'cn'), min_snr=3.0,
                      verbose=True):

    vel = np.median(res['vel_draws'])
    eta = float(np.median(res.get('eta_draws', np.array([0.0]))))
    prof = np.zeros(len(wave))
    for k in range(len(res['absorbers'])):
        ab = res['absorbers'][k]
        if ab['kind'] not in kinds:
            continue
        h_draws = res['height_draws'][:, k]
        h = np.median(h_draws)
        s = np.std(h_draws)
        snr = h / s if s > 0 else 0.0
        if ab['kind'] != 'balmer' and snr < min_snr:
            if verbose:
                print(f"  {ab['name']}: not subtracted (S/N = {snr:.1f})")
            continue
        if verbose:
            print(f"  {ab['name']}: subtracted (S/N = {snr:.1f})")
        p, _, _ = absorber_profile(ab, wave, res['z'], vel, res['vel_metal'],
                                   eta=eta)
        prof -= h * p
    return prof


def plot_series(res, class_, save_path=None, name='series_fit'):
    wave, flux, good = res['wave'], res['flux'], res['good']
    deg, vel, z = res['deg'], res['vel_best'], res['z']
    eta = res.get('eta_best', 0.0)
    A = design_matrix(wave, res['absorbers'], z, vel, res['vel_metal'], deg,
                      eta=eta)
    coef = res['coef_best']
    model = A @ coef
    cont = A[:, :deg + 1] @ coef[:deg + 1]
    balmer_only = cont.copy()
    for k in range(len(res['absorbers'])):
        if res['absorbers'][k]['kind'] == 'balmer':
            balmer_only += A[:, deg + 1 + k] * coef[deg + 1 + k]

    fig, (ax, axr, axv) = plt.subplots(
        3, 1, figsize=(12, 9),
        gridspec_kw={'height_ratios': [3, 1, 1.3], 'hspace': 0.35})
    ax.step(wave, flux, where='mid', color='lightgrey', lw=0.8,
            label='data (masked)')
    flux_good = np.where(good, flux, np.nan)
    ax.step(wave, flux_good, where='mid', color='k', lw=0.8,
            label='data used')
    ax.plot(wave, cont, 'C0--', label='polynomial continuum')
    ax.plot(wave, balmer_only, 'C2-', lw=1, label='continuum + Balmer only')
    ax.plot(wave, model, 'r-', lw=1.2,
            label=f'full model (v = {vel:.0f}, metals {res["vel_metal"]:.0f}, '
                  f'eta {eta:.2f})')
    for ab in res['absorbers']:
        _, _, centre = absorber_profile(ab, wave, z, vel, res['vel_metal'])
        color = 'C2' if ab['kind'] == 'balmer' else 'C1'
        ax.axvline(centre, color=color, lw=0.5, ls=':')
        ax.text(centre, 0.02, ab['name'].replace('_', ' '), rotation=90,
                fontsize=7, color=color, transform=ax.get_xaxis_transform())
    lo, hi = np.nanpercentile(flux_good, [1, 99])
    ax.set_ylim(lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo))
    ax.set_ylabel('Flux')
    ax.set_title(f'{class_.gal_id}: joint Balmer series fit, '
                 f'$\\chi^2_\\nu$ = {res["chi2_nu"]:.2f}')
    ax.legend(fontsize=8, ncol=2)

    resid = np.where(good, (flux - model) / res['err'], np.nan)
    axr.step(wave, resid, where='mid', color='k', lw=0.7)
    axr.axhline(0, color='r', lw=0.8)
    axr.set_ylabel('(data-model)/err')
    axr.set_xlabel(r'Obs. wavelength ($\AA$)')

    dchi2 = res['chi2_grid'] - res['chi2_grid'].min()
    axv.plot(res['vel_grid'], dchi2 / res['chi2_nu'], 'o-')
    axv.axvline(550, color='grey', ls='--', label='550 km/s (old fixed)')
    axv.set_xlabel(r'Balmer absorption $\sigma_v$ [km/s]')
    axv.set_ylabel(r'$\Delta\chi^2$ (rescaled)')
    axv.set_ylim(0, 25)
    axv.legend(fontsize=8)

    if save_path is not None:
        fig.savefig(f'{save_path}/{name}_{class_.gal_id}.png', dpi=150,
                    bbox_inches='tight')
    plt.show()
    return fig


# ================================================================
# Test for the fit
# ================================================================

def _ew_and_err(wave, flux, err, good, absorbers, z, vel, vel_metal, deg,
                shift=0.0, eta=0.0):
    """EWs of the best fit at fixed vel, with linear (covariance) errors
    rescaled by sqrt(chi2_nu) (covariance of the unbounded problem: an
    approximation when a bound is active)."""
    coef, chi2, A = solve(wave, flux, err, good, absorbers, z, vel,
                          vel_metal, deg, shift, eta=eta)
    Aw = A[good] / err[good][:, None]
    cov = np.linalg.pinv(Aw.T @ Aw)
    dof = np.sum(good) - A.shape[1]
    cov = cov * max(1.0, chi2 / dof)
    ews, _ = ews_from_coef(coef, wave, absorbers, z, vel, vel_metal, deg,
                           shift, eta)
    ew_err = np.zeros(len(absorbers))
    for k in range(len(absorbers)):
        h_err = np.sqrt(cov[deg + 1 + k, deg + 1 + k])
        h = coef[deg + 1 + k]
        ew_err[k] = abs(ews[k]) * h_err / abs(h) if h != 0 else np.nan
    return ews, ew_err


def shift_test(class_, data, sigmas, shifts=None, rest_window=(3730., 4160.),
               deg=4, vel=None, vel_metal=None, core_size=4.0, metals=True,
               mg_triplet=True, cn_band=True, plot=True, eta=0.0):
    """
    NULL TEST. Move ALL absorbers by the same rest-frame shift (A) and
    refit. If the dips are real absorption, the mean Balmer EW is large
    only at shift = 0. Lines in bal.EXCLUDE_FROM_EW are not averaged.
    """
    if shifts is None:
        shifts = np.arange(-40., 40.1, 4.)
    wave, flux, err, good, absorbers, z = _prepare(
        class_, data, sigmas, rest_window, core_size, metals, mg_triplet,
        cn_band)
    if vel_metal is None:
        vel_metal = np.arange(100., 201., 25.)
    vel_grid = np.arange(250., 1001., 50.) if vel is None else [vel]
    (vel, _, _, _, vel_metal, _), _ = best_vel(wave, flux, err, good,
                                               absorbers, z, vel_grid,
                                               vel_metal, deg,
                                               eta_grid=[eta])

    mean_ew, mean_err, n_used = [], [], []
    for sh in shifts:
        keep = []
        for ab in absorbers:
            prof, _, centre = absorber_profile(ab, wave, z, vel, vel_metal,
                                               sh, eta)
            half = 2 * centre * vel / C_KMS
            inside = (centre > wave[0] + half) and (centre < wave[-1] - half)
            near = good & (prof > 0.3)
            if inside and np.sum(near) >= 5:
                keep.append(ab)
        ews, ew_err = _ew_and_err(wave, flux, err, good, keep, z, vel,
                                  vel_metal, deg, sh, eta)
        idx = []
        for j in range(len(keep)):
            if (keep[j]['kind'] == 'balmer'
                    and keep[j]['name'] not in bal.EXCLUDE_FROM_EW
                    and np.isfinite(ew_err[j]) and ew_err[j] > 0):
                idx.append(j)
        if len(idx) == 0:
            mean_ew.append(np.nan)
            mean_err.append(np.nan)
            n_used.append(0)
            continue
        w = 1 / ew_err[idx]**2
        mean_ew.append(np.sum(w * ews[idx]) / np.sum(w))
        mean_err.append(1 / np.sqrt(np.sum(w)))
        n_used.append(len(idx))
    mean_ew, mean_err = np.array(mean_ew), np.array(mean_err)
    for i in range(len(shifts)):
        print(f'  shift {shifts[i]:+5.0f}: EW = {mean_ew[i]:6.2f} +- '
              f'{mean_err[i]:.2f}  ({n_used[i]} H lines)')

    if plot:
        plt.figure(figsize=(7, 4))
        plt.errorbar(shifts, mean_ew, mean_err, fmt='o-')
        plt.axhline(0, color='grey')
        plt.axvline(0, color='r', ls=':')
        top = np.nanmax(np.abs(mean_ew) + mean_err)
        plt.ylim(-1.2 * top, 1.2 * top)
        plt.xlabel(r'shift of all absorbers [$\AA$, rest]')
        plt.ylabel(r'mean H EW [$\AA$]')
        plt.title(f'{class_.gal_id}: shift (null) test, v = {vel:.0f} km/s')
        plt.show()
    return shifts, mean_ew, mean_err


def degree_test(class_, data, sigmas, degs=(2, 3, 4, 5, 6, 8),
                rest_window=(3730., 4160.), vel_metal=None, core_size=4.0,
                metals=True, mg_triplet=True, cn_band=True, eta=0.0):
    """EWs vs polynomial degree. Line absorption: stable. Continuum
    leaking into the lines: EWs drift with the degree."""
    wave, flux, err, good, absorbers, z = _prepare(
        class_, data, sigmas, rest_window, core_size, metals, mg_triplet,
        cn_band)
    if vel_metal is None:
        vel_metal = np.arange(100., 201., 25.)
    rows = []
    for deg in degs:
        (vel, _, chi2, _, vm, _), _ = best_vel(wave, flux, err, good,
                                               absorbers, z,
                                               np.arange(250., 1001., 50.),
                                               vel_metal, deg,
                                               eta_grid=[eta])
        ews, ew_err = _ew_and_err(wave, flux, err, good, absorbers, z, vel,
                                  vm, deg, eta=eta)
        row = {'deg': deg, 'vel': vel, 'vel_metal': vm,
               'chi2_nu': chi2 / (np.sum(good) - deg - 1 - len(absorbers) - 1)}
        for k in range(len(absorbers)):
            row[absorbers[k]['name']] = f'{ews[k]:.2f}+-{ew_err[k]:.2f}'
        rows.append(row)
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    return table


def fit_hgamma(class_, data, sigmas, vel, rest_window=(4205., 4445.), deg=2,
               n_mc=1000, vel_metal=None, seed=None, eta=0.0):

    return fit_balmer_series(class_, data, sigmas, rest_window=rest_window,
                             deg=deg, vel_grid=np.array([vel]), n_mc=n_mc,
                             vel_metal=vel_metal, seed=seed,
                             eta_grid=np.array([eta]))
