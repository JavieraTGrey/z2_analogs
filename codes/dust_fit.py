"""
E(B-V) and the shape of the attenuation law from the H I lines, and the
table that compares the laws.
"""
import numpy as np
import pandas as pd
import dust_laws
from dust_laws import RV, k_law, ebv_scale, LAM_HB, red_slope_x
from dust_data import rest_wavelength

# ================================================================
# E(B-V)
# ================================================================

def ebv_per_line(file_table, lines, case_ratios, law, linelist_dict):
    """
    E(B-V) from each ratio F_j/F_Hb, per MC realization (DataFrame i x line)
    F_j/F_Hb = 10**(-0.4*E(B-V)*deltaK)
    E(B-V) = -1/0.4 *log10F_j/F_Hb)/ deltaK
    """
    # Get Hbeta information
    wl_hb = rest_wavelength('H1_4861A', linelist_dict)
    k_hb = k_law(wl_hb, law)[0]
    out = {}
    for lab in lines:
        # Exclude H_beta
        if lab == 'H1_4861A':
            continue
        wl_dk = rest_wavelength(lab, linelist_dict)
        dk = k_law(wl_dk, law)[0] - k_hb

        # Obs ratio
        obs = file_table[lab] / file_table['H1_4861A']
        out[lab] = (-1/0.4) * (1 / dk) * np.log10(obs / case_ratios[lab])
    return pd.DataFrame(out)


def ebv_combined(file_table, lines, case_ratios, law, linelist_dict=None):
    """Weighted fit of log10(F_j/R_j) = a - 0.4 E k_j over all lines, per
    realization. Weights from the MC scatter of each line."""
    k = np.array([])
    for line in lines:
        kw = k_law(rest_wavelength(line, linelist_dict), law)
        k = np.append(k, kw)

    ratios = []
    for line in lines:
        ratios.append(case_ratios[line])

    ratios = np.array(ratios)
    fluxes = file_table[lines].to_numpy()

    Y = np.log10(fluxes / ratios)

    sig = np.nanstd(Y, axis=0)
    sig[sig <= 0] = np.nan
    w = 1 / sig**2

    E = np.full(len(Y), np.nan)

    for n, y in enumerate(Y):
        ok = np.isfinite(y) & np.isfinite(w)
        if ok.sum() < 2:
            continue
        ones = np.ones(ok.sum())
        x = -0.4 * k[ok]
        A = np.column_stack([ones, x])

        sw = np.sqrt(w[ok])
        A_w = A * sw[:, None]
        y_w = y[ok] * sw

        result = np.linalg.lstsq(A_w, y_w, rcond=None)
        coef = result[0]
        E[n] = coef[1]
    return pd.Series(E, index=file_table.index, name=f'E_BV_{law}')


# ================================================================
# Fitting the shape of the law
# ================================================================

def median_decrement(file_table, lines, case_ratios, linelist_dict):
    """Per line: median of y = log10[(F/F_Hb) / (F/F_Hb)_CaseB], its MC
    scatter, and the rest wavelength."""
    y = {}
    s = {}
    wl = {}
    for lab in lines:
        v = np.log10((file_table[lab] / file_table['H1_4861A'])
                     / case_ratios[lab])
        v = v.replace([np.inf, -np.inf], np.nan).dropna()
        y[lab] = v.median()
        s[lab] = max(v.std(), 1e-3)     # avoid dividing by ~0 (H-beta)
        wl[lab] = rest_wavelength(lab, linelist_dict)
    return y, s, wl


def decrement_chi2(y, s, wl, labels, law, E):
    """chi^2 of the lines in labels against the law with this E(B-V).
    Model: y = -0.4 E [k(lambda) - k(H-beta)]."""
    k_hb = k_law(wl['H1_4861A'], law)[0]
    chi2 = 0.0
    for lab in labels:
        model = -0.4 * E * (k_law(wl[lab], law)[0] - k_hb)
        chi2 = chi2 + ((y[lab] - model) / s[lab])**2
    return chi2


def fit_law_param(file_table, lines, case_ratios, linelist_dict,
                  name, values):
    """Best value of a law's shape parameter for each realization.
    name   : 'power' (values = beta) or 'ccm' (values = R_V)
    values : the grid to try
    For every value, do the same weighted fit as ebv_combined and keep the
    value with the lowest chi^2."""
    wl = []
    ratios = []
    for line in lines:
        wl.append(rest_wavelength(line, linelist_dict))
        ratios.append(case_ratios[line])
    wl = np.array(wl)

    # k at each line for every value on the grid (computed once)
    k_grid = []
    for v in values:
        k_grid.append(k_law(wl, f'{name}:{v:.2f}'))

    Y = np.log10(file_table[lines].to_numpy() / np.array(ratios))
    sig = np.nanstd(Y, axis=0)
    sig[sig <= 0] = np.nan
    w = 1 / sig**2

    best = np.full(len(Y), np.nan)
    for n in range(len(Y)):
        y = Y[n]
        ok = np.isfinite(y) & np.isfinite(w)
        if ok.sum() < 3:
            continue
        sw = np.sqrt(w[ok])
        best_chi2 = np.inf
        for j in range(len(values)):
            x = -0.4 * k_grid[j][ok]
            A = np.column_stack([np.ones(ok.sum()), x])
            coef = np.linalg.lstsq(A * sw[:, None], y[ok] * sw, rcond=None)[0]
            chi2 = np.sum(((y[ok] - A @ coef) * sw)**2)
            if chi2 < best_chi2:
                best_chi2 = chi2
                best[n] = values[j]
    return pd.Series(best, index=file_table.index, name=name)


def law_chi2(file_table, lines, case_ratios, linelist_dict, law, E=None):
    """chi^2 of one law against the median H I fluxes, the same way for
    every law: model log10(F/R) = a - 0.4 E k(lambda), weighted by the MC
    scatter of each line.
    E=None : fit a and E (all-line laws)
    E=value: E is fixed (hybrid, from H-alpha) and only a is fitted
    Returns chi^2 and the number of lines."""
    y = []
    s = []
    k = []
    for lab in lines:
        v = np.log10(file_table[lab] / case_ratios[lab])
        v = v.replace([np.inf, -np.inf], np.nan).dropna()
        y.append(v.median())
        s.append(max(v.std(), 1e-3))
        k.append(k_law(rest_wavelength(lab, linelist_dict), law)[0])
    y = np.array(y)
    w = 1 / np.array(s)**2
    k = np.array(k)

    if E is None:
        A = np.column_stack([np.ones(len(y)), -0.4 * k])
        sw = np.sqrt(w)
        coef = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        model = A @ coef
    else:
        # best intercept for a fixed E is the weighted mean
        a = np.sum(w * (y + 0.4 * E * k)) / np.sum(w)
        model = a - 0.4 * E * k
    chi2 = np.sum(w * (y - model)**2)
    return chi2, len(y)


def _slope(law, wl, step=10.0):
    """dk/dlambda of a law at wl (per Angstrom), centred difference."""
    return (k_law(wl + step, law)[0] - k_law(wl - step, law)[0]) / (2 * step)


def _hybrid_name(split, frac, beta, amp):
    return f'hybrid:{split:.0f}:{frac:.2f}:{beta:.2f}:{amp:.4f}'


def fit_hybrid(file_table, lines, case_ratios, linelist_dict,
               red_frac=None, betas=None, fracs=None, split=None,
               smooth=False, shape='quad'):
    """
    The H-alpha-anchored hybrid law, fitted on the median decrement
    D(lambda) = A_lambda - A_Hb = -2.5 log10[(F/F_Hb)_obs / (F/F_Hb)_CaseB]

      1. E(B-V) from H-alpha/H-beta only.
      2. Red part (>= split): frac*Cardelli + (1-frac)*Calzetti.
         The red H I lines choose frac.
      3. Blue part (< split): a power law that starts where the red part
         ends:  D = D_red(split) + c * [(lam/lam_Hb)**-beta - (split/lam_Hb)**-beta]
         The blue H I lines choose beta AND its height c (= E(B-V)*amp),
         so the blue part is free to follow the blue points.

    split < H-beta (e.g. 4100): H-beta and H-alpha are both in the red
        part, so E(B-V) does not depend on the blue part and c is fitted.
    split >= H-beta (e.g. H-alpha): H-beta is in the blue part, so the
        blue power law has to go through H-beta (D = 0) and H-alpha;
        that fixes its height, and only beta is fitted (as before).
    One blue line only: beta cannot be chosen, it is fixed to 1 (linear).

    smooth=True (only for split < H-beta): the blue part must also have the
        same SLOPE as the red part at the split, so there is no corner.
        That fixes the height: amp = -slope_red * split / (beta * (split/lam_Hb)**-beta)
        and only beta is fitted. Smooth, but it may not reach the blue points.

    shape='quad' (only for split < H-beta; the default): the blue part is a
        quadratic in x = 1/lambda that starts with the same value AND slope
        as the red part (no corner) and then curves up:
            D = D_red(split) + E*slope_red*(x - xs) + c*(x - xs)**2
        c (= E(B-V)*q) is fitted to the blue lines (weighted least squares).
        Smooth and it follows the blue points. Name 'hybrid:split:frac:quad:q'.
    shape='power': the power law described above.

    red_frac : fix the Cardelli fraction (1 = Cardelli, 0 = Calzetti)
    betas    : grid of beta to try (default 0.1 to 5.0)
    split    : Angstrom (default dust_laws.HYBRID_SPLIT)
    Returns (law name, number of fitted parameters), the name being
    'hybrid:split:frac:beta:amp'.
    """
    if split is None:
        split = dust_laws.HYBRID_SPLIT
    if betas is None:
        betas = np.arange(0.1, 10.01, 0.1)
    if fracs is None:
        fracs = np.linspace(0, 1, 21)
    if red_frac is not None:
        fracs = [red_frac]

    y, s, wl = median_decrement(file_table, lines, case_ratios, linelist_dict)
    # in magnitudes: D = A_lambda - A_Hb and its error
    D = {}
    sD = {}
    for lab in lines:
        D[lab] = -2.5 * y[lab]
        sD[lab] = 2.5 * s[lab]

    # divide the lines at the split
    red = []
    blue = []
    for lab in lines:
        if lab in ('H1_4861A', 'H1_6563A'):
            continue
        if wl[lab] >= split:
            red.append(lab)
        else:
            blue.append(lab)
    print(f'  hybrid: split at {split:.0f} A; red lines {red}, blue lines {blue}')

    hb_in_red = wl['H1_4861A'] >= split
    if len(red) == 0 and red_frac is None:
        # no red line can choose frac: fit with any frac (it barely matters
        # blueward), then choose it below by matching slopes at the split
        fracs = [0.5]
    if len(blue) == 1:
        betas = [1.0]
        print('  hybrid: only one blue line, beta fixed to 1 (linear)')
    if len(blue) == 0:
        betas = [1.0]
        print('  hybrid: no blue lines, beta fixed to 1 (linear)')

    if shape == 'quad' and not hb_in_red:
        print('  hybrid: quadratic needs split < H-beta, using a power law')
        shape = 'power'
    if shape == 'quad':
        return _fit_hybrid_quad(y, s, wl, D, sD, red, blue, split, fracs,
                                red_frac)

    best_chi2 = np.inf
    best = None
    for frac in fracs:
        red_law = f'hybrid:{split:.0f}:{frac:.2f}:1.00:0'   # red part only
        for beta in betas:
            # power-law shape of the blue part, 0 at the split
            g = {}
            for lab in lines:
                g[lab] = (wl[lab] / LAM_HB)**(-beta) - (split / LAM_HB)**(-beta)

            if hb_in_red:
                # E(B-V) from H-alpha with the red part only
                k_ha = k_law(wl['H1_6563A'], red_law)[0]
                k_hb = k_law(wl['H1_4861A'], red_law)[0]
                E = D['H1_6563A'] / (k_ha - k_hb)
                D_split = E * (k_law(split, red_law)[0] - k_hb)
                # height of the blue part: weighted least squares on the
                # blue lines, c = sum(w r g) / sum(w g^2)
                top = 0.0
                bottom = 0.0
                for lab in blue:
                    wgt = 1 / sD[lab]**2
                    top = top + wgt * (D[lab] - D_split) * g[lab]
                    bottom = bottom + wgt * g[lab]**2
                c = 0.0
                if bottom > 0:
                    c = top / bottom
                amp = c / E
                if smooth:
                    # slope of the blue power law at the split equals the
                    # slope of the red part (d k / d lambda)
                    s_red = _slope(red_law, split + 20.0)
                    amp = -s_red * split / (beta * (split / LAM_HB)**(-beta))
            else:
                # blue power law fixed by H-beta: amp per E(B-V) as for
                # the plain power law (E(B-V) = A(4400) - A(5500))
                amp = 1.0 / ebv_scale(f'power:{beta:.2f}')
                trial = _hybrid_name(split, frac, beta, amp)
                dk = k_law(wl['H1_6563A'], trial)[0] \
                    - k_law(wl['H1_4861A'], trial)[0]
                E = D['H1_6563A'] / dk

            law = _hybrid_name(split, frac, beta, amp)
            chi2 = decrement_chi2(y, s, wl, red + blue, law, E)
            if chi2 < best_chi2:
                best_chi2 = chi2
                best = (frac, beta, amp)

    frac, beta, amp = best

    if len(red) == 0 and red_frac is None and not hb_in_red:
        # nothing in the red part can choose frac: match the slope of the
        # blue power law at the split (no kink)
        blue_law = f'power:{beta:.2f}'
        s_blue = _slope(blue_law, split) / ebv_scale(blue_law)
        s_ccm = _slope('ccm', split)
        s_calz = _slope('calzetti', split)
        frac = (s_blue - s_calz) / (s_ccm - s_calz)
        if frac < 0 or frac > 1:
            print(f'  hybrid: slope match needs frac={frac:.2f}, '
                  'clipped to 0-1 (small kink at the split)')
        frac = min(max(frac, 0.0), 1.0)

    if len(betas) > 1 and (beta <= betas[0] + 1e-9 or beta >= betas[-1] - 1e-9):
        print(f'  warning: hybrid beta = {beta:.1f} is at the edge of the grid')
    if amp < 0:
        print('  warning: the blue part goes DOWN to the blue (amp < 0)')

    # parameters: E (from H-alpha), beta, and frac / the blue height if fitted
    n_par = 1
    if len(betas) > 1:
        n_par = n_par + 1
    if hb_in_red and len(blue) > 0 and not smooth:
        n_par = n_par + 1
    if red_frac is None and len(fracs) > 1 and len(red) > 0:
        n_par = n_par + 1

    law = _hybrid_name(split, frac, beta, amp)
    print(f'  hybrid: red = {frac:.0%} Cardelli, blue = power law '
          f'beta {beta:.1f}, amp {amp:.2f}  ->  {law}')
    return law, n_par


def _fit_hybrid_quad(y, s, wl, D, sD, red, blue, split, fracs, red_frac):
    """Quadratic blue part (see fit_hybrid, shape='quad'). For each Cardelli
    fraction: E(B-V) from H-alpha, then the curvature c from the blue lines;
    keep the fraction with the lowest chi^2 over all the H I lines."""
    xs = 1e4 / split
    best_chi2 = np.inf
    best = None
    for frac in fracs:
        red_law = f'hybrid:{split:.0f}:{frac:.2f}:quad:0'   # straight blue part
        k_ha = k_law(wl['H1_6563A'], red_law)[0]
        k_hb = k_law(wl['H1_4861A'], red_law)[0]
        E = D['H1_6563A'] / (k_ha - k_hb)
        # c = sum(w r g) / sum(w g^2), r = what is left after the straight
        # part, g = (x - xs)^2
        top = 0.0
        bottom = 0.0
        for lab in blue:
            straight = E * (k_law(wl[lab], red_law)[0] - k_hb)
            g = (1e4 / wl[lab] - xs)**2
            wgt = 1 / sD[lab]**2
            top = top + wgt * (D[lab] - straight) * g
            bottom = bottom + wgt * g**2
        c = 0.0
        if bottom > 0:
            c = top / bottom
        q = c / E
        law = f'hybrid:{split:.0f}:{frac:.2f}:quad:{q:.4f}'
        chi2 = decrement_chi2(y, s, wl, red + blue, law, E)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best = (frac, q)

    frac, q = best
    if q < 0:
        print('  warning: the blue part curves DOWN (q < 0)')
    # parameters: E (from H-alpha), q, and frac if the red lines chose it
    n_par = 1
    if len(blue) > 0:
        n_par = n_par + 1
    if red_frac is None and len(fracs) > 1 and len(red) > 0:
        n_par = n_par + 1
    law = f'hybrid:{split:.0f}:{frac:.2f}:quad:{q:.4f}'
    print(f'  hybrid: red = {frac:.0%} Cardelli, blue = smooth quadratic '
          f'q {q:.2f}  ->  {law}')
    return law, n_par


# ================================================================
# Comparing the laws
# ================================================================

# Lines whose dust correction (relative to H-beta) goes in the summary table
COMPARE_LINES = {'[O II] 3727': 3727.4, '[Ne III] 3869': 3868.8,
                 '[O III] 4363': 4363.2, '[O III] 5007': 5006.8,
                 '[N II] 6583': 6583.5, '[S II] 6716': 6716.4,
                 '[O II] 7325': 7325.0, '[S III] 9069': 9068.6}


def law_summary_table(res):
    """One row per law, sorted from best to worst, with only quantities
    that mean the same for every law (whatever its normalisation):
      E_BV, err_lo, err_hi : E(B-V) = A(4400) - A(5500) (median, 16-84)
      A_Hb, A_V            : attenuation at H-beta and in V (mag)
      R_V                  : A_V / E(B-V)
      corr <line>          : correction factor of F(line)/F(H-beta)
      n_par, chi2, dof, chi2_nu, BIC, dBIC : how well the law fits
    """
    fitted = [res['power_law'], res['ccm_fit_law'], res['calz_fit_law']]
    wl_hb = rest_wavelength('H1_4861A', res['linelist_dict'])
    rows = []
    for law in res['laws']:
        E_fit = res['combined'][law]            # E(B-V), or A_Hb for power
        E = E_fit * ebv_scale(law)               # always E(B-V)
        med = E.median()
        e_fit = E_fit.median()
        k_hb = k_law(wl_hb, law)[0]

        if law == res['hybrid_law']:
            chi2, n = law_chi2(res['F'], res['lines'], res['R'],
                               res['linelist_dict'], law, E=e_fit)
            # a + the hybrid's own fitted parameters (from fit_hybrid)
            n_par = 1 + res.get('hybrid_npar', 3)
        else:
            chi2, n = law_chi2(res['F'], res['lines'], res['R'],
                               res['linelist_dict'], law)
            n_par = 2
            if law in fitted:
                n_par = 3          # beta, R_V or delta was fitted
        dof = max(n - n_par, 1)

        row = {'law': law,
               'E_BV': med,
               'err_lo': med - E.quantile(0.16),
               'err_hi': E.quantile(0.84) - med,
               'A_Hb': e_fit * k_hb,
               'A_V': e_fit * k_law(5500.0, law)[0],
               'R_V': RV[law]}
        for name in COMPARE_LINES:
            k_line = k_law(COMPARE_LINES[name], law)[0]
            row[f'corr {name}'] = 10**(0.4 * e_fit * (k_line - k_hb))
        row['n_par'] = n_par
        row['chi2'] = chi2
        row['dof'] = dof
        row['chi2_nu'] = chi2 / dof
        row['BIC'] = chi2 + n_par * np.log(n)
        rows.append(row)
    table = pd.DataFrame(rows).set_index('law')
    table['dBIC'] = table['BIC'] - table['BIC'].min()
    return table.sort_values('BIC')