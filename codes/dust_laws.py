"""
Dust attenuation laws.

Every law gives k(lambda) = A_lambda / E(B-V), wl in Angstrom (rest frame),
through k_law(wl, law), where law is a name with optional parameters:
    'ccm', 'ccm:2.5'            Cardelli, Clayton & Mathis (1989)
    'calzetti', 'calzetti:3.5'  Calzetti et al. (2000)
    'calzetti_mod:-0.3'         modified Calzetti (Noll+09, Salim+18)
    'smc'                       SMC Bar, Gordon et al. (2003)
    'linear', 'power:1.4'       power law normalised at H-beta (per A_Hb)
    'hybrid:4100:0.60:2.30:1.85'  H-alpha-anchored hybrid (split, frac,
                                  beta, amp; see k_hybrid)
Test with test_dust_laws.py.
"""
import numpy as np
from scipy.interpolate import CubicSpline

# -----------------------------------------
# Names, colours, markers and R_V of each law
# (register_law adds new laws to these)
# -----------------------------------------

LAWS = ('ccm', 'calzetti')

RV = {'ccm': 3.1, 'calzetti': 4.05}

LAW_LABEL = {'ccm': 'Cardelli (R$_V$=3.1)',
             'calzetti': 'Calzetti (R$_V$=4.05)'}

LAW_COLOR = {'ccm': 'teal', 'calzetti': 'orange'}

LAW_MARKER = {'ccm': 'o', 'calzetti': 's'}


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


# ---- extra laws ------------------------------------------------

# Reasonable R_V ranges (CCM: Milky Way sightlines; Calzetti: 4.05 +- 0.80)
RV_RANGE = {'ccm': (2.2, 5.5), 'calzetti': (3.25, 4.85)}

# B and V effective wavelengths (micron): E(B-V) is defined by k(B)-k(V)=1
LAM_B = 0.44
LAM_V = 0.55

# Hybrid law: default split used by fit_hybrid (Angstrom). Redward:
# Cardelli/Calzetti mix; blueward: a power law. The split that was used is
# saved in the law name, so changing this later does not change a fitted law.
HYBRID_SPLIT = 4100.0

# SMC: Gordon et al. (2003) SMC Bar average curve, built exactly as in the
# dust_extinction package (G03_SMCBar): natural cubic spline in the
# optical/NIR, Fitzpatrick & Massa (1990) shape in the UV.
G03_RV = 2.74
G03_FM90 = {'C1': -4.959, 'C2': 2.264, 'C3': 0.389, 'C4': 0.461,
            'xo': 4.6, 'gamma': 1.0}
# optical/NIR anchor points: wavelength (micron) and A_lambda/A_V
G03_OPTNIR_LAM = [2.198, 1.65, 1.25, 0.81, 0.65, 0.55, 0.44, 0.37]
G03_OPTNIR_AAV = [0.11, 0.169, 0.25, 0.567, 0.801, 1.00, 1.374, 1.672]


def _g03_uv_aav(x):
    """UV part, A_lambda/A_V at x = 1/lambda (micron^-1)."""
    p = G03_FM90
    e = p['C1'] + p['C2'] * x
    e = e + p['C3'] * x**2 / ((x**2 - p['xo']**2)**2 + x**2 * p['gamma']**2)
    y = np.clip(x - 5.9, 0, None)          # far-UV rise only above x = 5.9
    e = e + p['C4'] * (0.5392 * y**2 + 0.05644 * y**3)
    return e / G03_RV + 1.0


def _g03_spline():
    """Natural cubic spline through: zero at infinite wavelength, the
    optical/NIR anchors, and the UV curve at 2700 and 2600 A."""
    xs = [0.0]
    ys = [0.0]
    for n in range(len(G03_OPTNIR_LAM)):
        xs.append(1.0 / G03_OPTNIR_LAM[n])
        ys.append(G03_OPTNIR_AAV[n])
    for lam_A in (2700.0, 2600.0):
        x_uv = 1e4 / lam_A
        xs.append(x_uv)
        ys.append(_g03_uv_aav(np.array([x_uv]))[0])
    return CubicSpline(xs, ys, bc_type='natural')


G03_SPLINE = _g03_spline()


def k_smc(wl):
    """SMC Bar curve of Gordon et al. (2003). wl in Angstrom.
    Valid 1000 A - 3.3 micron. Returns k = R_V * A_lambda / A_V."""
    x = 1e4 / np.atleast_1d(np.asarray(wl, dtype=float))
    aav = np.full_like(x, np.nan)
    opt = (x >= 0.3) & (x < 1e4 / 2700.0)
    uv = (x >= 1e4 / 2700.0) & (x <= 10.0)
    aav[opt] = G03_SPLINE(x[opt])
    aav[uv] = _g03_uv_aav(x[uv])
    return G03_RV * aav


LAM_HB = 4861.3     # H-beta rest wavelength (Angstrom, air)


def k_power(wl, beta):
    """Power-law attenuation normalised at H-beta, all in Angstrom:
        A_lambda = A_Hb * (lambda / lambda_Hb)^-beta
    Returns A_lambda / A_Hb, so for this law the 'E' that the fits give
    is A_Hb (the attenuation at H-beta, in mag), not E(B-V).
    Use ebv_scale(law) to convert it to E(B-V)."""
    wl = np.atleast_1d(np.asarray(wl, dtype=float))
    return (wl / LAM_HB)**(-beta)


def k_linear(wl):
    """Straight line in 1/lambda through zero at infinite wavelength:
    the power law with beta = 1 (normalised at H-beta, like k_power)."""
    return k_power(wl, 1.0)


def k_calzetti_mod(wl, delta):
    """Modified Calzetti (Noll et al. 2009; Salim et al. 2018, no bump):
    Calzetti tilted by (lambda / 0.55 micron)^delta.
    delta < 0: steeper than Calzetti (lower R_V); delta > 0: flatter.
    The new R_V is set so that k(B) - k(V) = 1 still holds."""
    lam = np.atleast_1d(np.asarray(wl, dtype=float)) / 1e4
    rv = 4.05
    rv_mod = rv / ((rv + 1) * (LAM_B / LAM_V)**delta - rv)
    return k_calzetti(wl, rv) * (rv_mod / rv) * (lam / LAM_V)**delta


def ebv_scale(law):
    """Multiply a law's fitted 'E' by this to get E(B-V).
    Power and linear laws are normalised at H-beta, so their 'E' is A_Hb
    and E(B-V) = A(4400) - A(5500) = A_Hb * [k(4400) - k(5500)].
    All other laws are already per E(B-V), so the factor is 1."""
    name = law.split(':')[0]
    if name in ('power', 'linear'):
        return k_law(4400.0, law)[0] - k_law(5500.0, law)[0]
    return 1.0


def k_hybrid(wl, split, frac, beta, amp):
    """Hybrid law, per E(B-V):
    Redward of split : frac*Cardelli + (1-frac)*Calzetti
    Blueward of split: a power law that starts where the red part ends,
        k = k_red(split) + amp * [(wl/wl_Hb)**-beta - (split/wl_Hb)**-beta]
    amp is the height of the blue part per unit E(B-V): with E(B-V) from
    H-alpha, A_blue - A_split = E(B-V) * amp * [...]."""
    w = np.atleast_1d(np.asarray(wl, dtype=float))
    red = frac * k_ccm(w) + (1 - frac) * k_calzetti(w)
    k_split = frac * k_ccm(split)[0] + (1 - frac) * k_calzetti(split)[0]
    blue = k_split + amp * ((w / LAM_HB)**(-beta) - (split / LAM_HB)**(-beta))
    return np.where(w >= split, red, blue)


def red_slope_x(split, frac):
    """Slope dk/dx of the red part (frac*Cardelli + (1-frac)*Calzetti) at
    the split, with x = 1/lambda in micron^-1 (measured just redward)."""
    lam2 = split + 20.0
    k1 = frac * k_ccm(split)[0] + (1 - frac) * k_calzetti(split)[0]
    k2 = frac * k_ccm(lam2)[0] + (1 - frac) * k_calzetti(lam2)[0]
    return (k1 - k2) / (1e4 / split - 1e4 / lam2)


def k_hybrid_quad(wl, split, frac, q):
    """Hybrid law with a smooth quadratic blue part, per E(B-V):
    Redward of split : frac*Cardelli + (1-frac)*Calzetti
    Blueward of split, with x = 1/lambda (micron^-1) and xs = 1/split:
        k = k_red(xs) + slope_red * (x - xs) + q * (x - xs)**2
    Same value AND same slope as the red part at the split (no corner);
    q is how fast it curves upward to the blue."""
    w = np.atleast_1d(np.asarray(wl, dtype=float))
    red = frac * k_ccm(w) + (1 - frac) * k_calzetti(w)
    k_split = frac * k_ccm(split)[0] + (1 - frac) * k_calzetti(split)[0]
    dx = 1e4 / w - 1e4 / split
    blue = k_split + red_slope_x(split, frac) * dx + q * dx**2
    return np.where(w >= split, red, blue)


def k_law(wl, law):
    """law is a name, optionally with parameters after ':'
        'ccm', 'ccm:2.5'            Cardelli, R_V = 3.1 or 2.5
        'calzetti', 'calzetti:3.5'  Calzetti, R_V = 4.05 or 3.5
        'calzetti_mod:-0.3'         modified Calzetti, delta = -0.3
        'smc'                       Gordon et al. (2003) SMC Bar
        'linear'                    straight line in 1/lambda
        'power:1.4'                 power law, beta = 1.4
        'hybrid:4100:0.60:2.30:1.85'  split at 4100 A, 60% Cardelli +
                                    40% Calzetti in the red, power law
                                    (beta 2.3, amp 1.85) in the blue
        'hybrid:4100:1.00:quad:9.5'  same red part, smooth quadratic in
                                    1/lambda in the blue (q = 9.5)
    """
    parts = law.split(':')
    name = parts[0]

    if name in ('ccm', 'calzetti'):
        rv = RV[name]
        if len(parts) > 1:
            rv = float(parts[1])
            lo, hi = RV_RANGE[name]
            if rv < lo or rv > hi:
                print(f'  warning: R_V={rv} outside {lo}-{hi} for {name}')
        if name == 'ccm':
            return k_ccm(wl, rv)
        return k_calzetti(wl, rv)
    if name == 'calzetti_mod':
        return k_calzetti_mod(wl, float(parts[1]))
    if name == 'smc':
        return k_smc(wl)
    if name == 'linear':
        return k_linear(wl)
    if name == 'power':
        return k_power(wl, float(parts[1]))
    if name == 'hybrid' and parts[3] == 'quad':
        return k_hybrid_quad(wl, float(parts[1]), float(parts[2]),
                             float(parts[4]))
    if name == 'hybrid':
        return k_hybrid(wl, float(parts[1]), float(parts[2]),
                        float(parts[3]), float(parts[4]))
    raise ValueError(f'Unknown law: {law}')


# ---- names, colours and markers for any law --------------------

PALETTE = ['teal', 'orange', '#7b3294', '#d01c8b', '#4dac26', '#2c7bb6',
           '#a6611a', '#666666', '#e66101', '#1b9e77', '#e7298a', '#1f78b4',
           '#b8860b', '#8c564b', '#17becf', '#bcbd22']
MARKERS = ['o', 's', '^', 'v', 'D', 'P', 'X', '<', '>', 'h', '*', 'p',
           'd', 'H', '8', '1']


def register_law(law):
    """Add label, colour, marker and R_V for a law so every plot can use it."""
    if law in LAW_COLOR:
        return
    n = len(LAW_COLOR)
    # first palette colour that no law uses yet
    color = PALETTE[n % len(PALETTE)]
    for c in PALETTE:
        if c not in LAW_COLOR.values():
            color = c
            break
    LAW_COLOR[law] = color
    LAW_MARKER[law] = MARKERS[n % len(MARKERS)]
    # R_V = A_V / E(B-V)  (= k(V) when k is per E(B-V))
    RV[law] = float(k_law(5500.0, law)[0] / ebv_scale(law))

    parts = law.split(':')
    name = parts[0]
    if name == 'ccm':
        LAW_LABEL[law] = f'Cardelli (R$_V$={parts[1]})'
    elif name == 'calzetti':
        LAW_LABEL[law] = f'Calzetti (R$_V$={parts[1]})'
    elif name == 'calzetti_mod':
        LAW_LABEL[law] = (rf'Mod. Calzetti ($\delta$={parts[1]}, '
                          f'R$_V$={RV[law]:.2f})')
    elif name == 'smc':
        LAW_LABEL[law] = 'SMC Bar (Gordon+03)'
    elif name == 'linear':
        LAW_LABEL[law] = r'Linear in 1/$\lambda$'
    elif name == 'power':
        LAW_LABEL[law] = rf'Power law ($\beta$={parts[1]})'
    elif name == 'hybrid':
        pct = int(round(100 * float(parts[2])))
        blue_txt = rf'$\beta$={parts[3]}'
        if parts[3] == 'quad':
            blue_txt = 'quadratic'
        LAW_LABEL[law] = (rf'Hybrid (split {parts[1]} $\AA$, blue: '
                          rf'{blue_txt}, red: {pct}% CCM)')
    else:
        LAW_LABEL[law] = law


def correction_factor(wl, E_BV, law):
    '''
    f_corr = f_obs * 10**(0.4 E(B-V)*k(lamda))
    Multiply observed fluxes by this to correct them.
    '''
    return 10 ** (0.4 * np.asarray(E_BV) * k_law(wl, law))