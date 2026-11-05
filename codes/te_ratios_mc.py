# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy as upy

# Path and useful libs
import os
docPath = '/Users/javieratoro/Desktop/thesis/proyecto 2024-2'
magePath = docPath + '/lines/'
os.sys.path.append(magePath)

# The mode defines which component are we using for the direct method and if
# it is corrected by extinction or not.
mode = 'lines'

# =============================================================================
#
# Data
#
# =============================================================================

MISSING = -9999.9


def line(table, label):
    f = table[f'{label}_flux'].to_numpy(dtype=float).copy()
    e = table[f'{label}_fluxerr'].to_numpy(dtype=float).copy()
    bad = (f <= 0) | (f == MISSING) | ~np.isfinite(f)
    f[bad] = np.nan
    e[(e == MISSING) | ~np.isfinite(e)] = 0.0
    return upy.uarray(f, e)


def pair(table, label_a, label_b):
    fa = table[f'{label_a}_flux'].to_numpy(dtype=float).copy()
    fb = table[f'{label_b}_flux'].to_numpy(dtype=float).copy()
    ea = table[f'{label_a}_fluxerr'].to_numpy(dtype=float).copy()
    eb = table[f'{label_b}_fluxerr'].to_numpy(dtype=float).copy()
    fa[(fa == MISSING) | ~np.isfinite(fa) | (fa < 0)] = 0.0
    fb[(fb == MISSING) | ~np.isfinite(fb) | (fb < 0)] = 0.0
    ea[(ea == MISSING) | ~np.isfinite(ea)] = 0.0
    eb[(eb == MISSING) | ~np.isfinite(eb)] = 0.0
    f = fa + fb
    f[f <= 0] = np.nan
    return upy.uarray(f, np.sqrt(ea**2 + eb**2))


def positive(u):
    v = upy.nominal_values(u).astype(float).copy()
    e = upy.std_devs(u).astype(float).copy()
    bad = ~np.isfinite(v) | (v <= 0)
    v[bad] = np.nan
    e[bad] = 0.0
    return upy.uarray(v, e)


def load_lines(table):
    L = {}
    L['OII3727'] = line(table, 'O2_3726A')
    L['OII3729'] = line(table, 'O2_3729A')
    L['NE3868'] = line(table, 'Ne3_3869A')
    L['OIII4363'] = line(table, 'O3_4363A')
    L['SII4068'] = line(table, 'S2_4069A')
    L['SII4076'] = line(table, 'S2_4076A')
    L['Hb'] = line(table, 'H1_4861A')
    L['OIII4959'] = line(table, 'O3_4959A')
    L['OIII5007'] = line(table, 'O3_5007A')
    L['NII5755'] = line(table, 'N2_5755A')
    L['OI6300'] = line(table, 'O1_6300A')
    L['SIII6312'] = line(table, 'S3_6312A')
    L['NII6548'] = line(table, 'N2_6548A')
    L['Ha'] = line(table, 'H1_6563A')
    L['NII6584'] = line(table, 'N2_6583A')
    L['SII6716'] = line(table, 'S2_6716A')
    L['SII6730'] = line(table, 'S2_6731A')
    L['OII7319'] = pair(table, 'O2_7319A', 'O2_7320A')
    L['OII7330'] = pair(table, 'O2_7330A', 'O2_7331A')
    L['Hg'] = line(table, 'H1_4340A')
    L['ARIII7136'] = line(table, 'Ar3_7136A')
    return L

# =============================================================================
#
# Functions
#
# =============================================================================

# =============================================================================
# For testing
# =============================================================================


# Mock unc float to make tests
unc_test = ufloat(1, 0.1)
unc_test_func = upy.log10([unc_test, unc_test])


def is_unc(obj):
    '''
    Return True if the obj is an uncertainty array and False if not.
    '''
    try:
        boolean = (type(obj[0]) is type(unc_test))
        if not boolean:
            boolean = (type(obj[0]) is type(unc_test_func[0]))
    except:
        boolean = (type(obj) is type(unc_test))
        if not boolean:
            boolean = (type(obj) is type(unc_test_func[0]))
    return boolean


def has_uplim(unc_array):
    '''
    This function checks if the input array contain upper limits (Uncertainties
    equal to 0)
    '''

    assert (is_unc(unc_array))

    bool_uplim = np.sum(upy.std_devs(unc_array) == 0.0)
    if bool_uplim > 0:
        return True
    else:
        return False


def assign_uplims(uarray, mask):
    '''
    This function takes an array that should have upper limits where the mask
    indicates and assigns it.
    '''
    assert (is_unc(uarray))
    uarray_ul = np.where(mask, uarray,
                         upy.uarray(upy.nominal_values(uarray),
                                    0 * upy.nominal_values(uarray)))
    return uarray_ul

# =============================================================================
# More functions
# =============================================================================


def Tlow_ne(temp, path, galaxies, OII7330, OII7319, OII3729, OII3727,
            SII6730, SII6716, SII4076, SII4068, plot=False):
    ne_toplot = upy.uarray(np.linspace(10, 1000, 1000),
                           np.full_like(np.linspace(10, 1000, 1000), 0))

    t_low_it = []
    t_low_it_err = []

    for i in range(len(galaxies)):

        if temp == 'TO2':
            t_n = TO2(OII7330[i], OII7319[i], OII3729[i], OII3727[i],
                      ne_toplot)
        elif temp == 'TS2':
            t_n = TS2(SII6730[i], SII6716[i], SII4076[i], SII4068[i],
                      ne_toplot)

        n_t_n = ne(SII6730[i], SII6716[i], t_n)

        diff = np.abs(upy.nominal_values(ne_toplot)
                      - upy.nominal_values(n_t_n))
        if np.all(np.isnan(diff)):
            t_low_it.append(np.nan)
            t_low_it_err.append(np.nan)
            continue
        index_min = np.nanargmin(diff)

        if plot:
            fig, axs = plt.subplots(1, 2)
            fig.set_size_inches((9, 3))
            axs[0].plot(upy.nominal_values(t_n), upy.nominal_values(ne_toplot))
            axs[0].plot(upy.nominal_values(t_n), upy.nominal_values(n_t_n))
            axs[1].plot(upy.nominal_values(t_n), diff)
            axs[0].set_xlabel('Low ionization temperature')
            axs[0].set_ylabel('Electron density')
            axs[0].set_title(galaxies[i])
            axs[1].set_xlabel(temp)
            axs[1].set_ylabel('|n - n(t(n))|')
            axs[1].axvline(upy.nominal_values(t_n)[index_min], ls='--')
            fig.savefig(path + f'n_vs_{temp}_{galaxies[i]}.pdf',
                        bbox_inches='tight')
            plt.close()

        t_low_it.append(upy.nominal_values(t_n)[index_min])
        t_low_it_err.append(upy.std_devs(t_n)[index_min])

    t_low_new = upy.uarray(t_low_it, t_low_it_err)
    n_e_new = ne(SII6730, SII6716, t_low_new)

    return n_e_new, t_low_new

# =============================================================================
# Line ratios
# =============================================================================


def N2_func(niia, ha):
    assert (is_unc(niia))
    assert (is_unc(ha))

    if has_uplim(niia) or has_uplim(ha):
        mask = (upy.std_devs(niia) == 0.) + (upy.std_devs(ha) == 0.)
        return assign_uplims(upy.log10(niia / ha), mask)
    else:
        return upy.log10(niia / ha)


def R3_func(oiiia, hb):
    assert (is_unc(oiiia))
    assert (is_unc(hb))

    if has_uplim(oiiia) or has_uplim(hb):
        mask = (upy.std_devs(oiiia) == 0.) + (upy.std_devs(hb) == 0.)
        return assign_uplims(upy.log10(oiiia / hb), mask)
    else:
        return upy.log10(oiiia / hb)


def R2_func(oiic, oiid, hb):
    assert (is_unc(oiic))
    assert (is_unc(oiid))
    assert (is_unc(hb))

    oii = oiic + oiid

    if has_uplim(oiic) or has_uplim(oiid) or has_uplim(hb):
        mask = (upy.std_devs(oiic) == 0.) + (upy.std_devs(hb) == 0.) + \
               (upy.std_devs(oiid) == 0.)
        return assign_uplims(upy.log10(oii / hb), mask)
    else:
        return upy.log10(oii / hb)


def R23_func(oiiia, oiiib, oiic, oiid, hb):
    assert (is_unc(oiiia))
    assert (is_unc(oiiib))
    assert (is_unc(oiic))
    assert (is_unc(oiid))
    assert (is_unc(hb))

    oiii = oiiia + oiiib
    oii = oiic + oiid
    num = oiii + oii
    return upy.log10(num / hb)


def O3N2_func(oiiia, hb, niia, ha):
    assert (is_unc(oiiia))
    assert (is_unc(hb))
    assert (is_unc(niia))
    assert (is_unc(ha))

    num = oiiia / hb
    den = niia / ha
    return upy.log10(num / den)


def R3N2_func(oiiia, niia):
    assert (is_unc(oiiia))
    assert (is_unc(niia))
    return upy.log10(oiiia / niia)


def N2O2_func(niia, oiid):
    assert (is_unc(niia))
    assert (is_unc(oiid))
    return upy.log10(niia / oiid)


def O3O2_func(oiiia, oiic, oiid):
    assert (is_unc(oiiia))
    assert (is_unc(oiic))
    assert (is_unc(oiid))
    oii = oiic + oiid
    return upy.log10(oiiia / oii)


def S2_func(siia, siib, ha):
    assert (is_unc(siia))
    assert (is_unc(siib))
    assert (is_unc(ha))
    sii = siia + siib
    return upy.log10(sii / ha)


def O3S2_func(oiiia, hb, siia, siib, ha):
    assert (is_unc(oiiia))
    assert (is_unc(hb))
    assert (is_unc(siia))
    assert (is_unc(siib))
    assert (is_unc(ha))
    sii = siia + siib
    num = oiiia / hb
    den = sii / ha
    return upy.log10(num / den)


def N2S2_func(niia, siia, siib):
    assert (is_unc(niia))
    assert (is_unc(siia))
    assert (is_unc(siib))
    sii = siia + siib
    return upy.log10(niia / sii)


def NE3O2_func(neiii, oiia, oiib):
    assert (is_unc(neiii))
    assert (is_unc(oiia))
    assert (is_unc(oiib))
    oii = oiia + oiib
    return upy.log10(neiii / oii)


def NE3_func(neiii, hg):
    assert (is_unc(neiii))
    assert (is_unc(hg))
    return upy.log10(neiii / hg)


def RO2NE3_func(neiii, oiia, oiib, hg):
    assert (is_unc(neiii))
    assert (is_unc(oiia))
    assert (is_unc(oiib))
    assert (is_unc(hg))
    oii = oiia + oiib
    return upy.log10((neiii + oii) / hg)


def AR3_func(ariii, hg):
    assert (is_unc(ariii))
    assert (is_unc(hg))
    return upy.log10(ariii / hg)


def AR3O3_func(ariii, oiii):
    assert (is_unc(ariii))
    assert (is_unc(oiii))
    return upy.log10(ariii / oiii)

# =============================================================================
# Electron density & temperatures
# =============================================================================


def TO3(oiiia, oiiib, oiiic):
    assert (is_unc(oiiia))
    assert (is_unc(oiiib))
    assert (is_unc(oiiic))
    RO3 = (oiiia + oiiib) / oiiic
    t_OIII = 0.784 - (1.357e-4 * RO3) + (48.44 / RO3)
    return t_OIII


def ne(siia, siib, to3):
    assert (is_unc(siia))
    assert (is_unc(siib))
    assert (is_unc(to3))
    RS2 = siib / siia
    ne = 1e3 * ((RS2 * a0_t(to3)) + a1_t(to3)) / ((RS2 * b0_t(to3)) +
                                                  b1_t(to3))
    if np.sum(upy.nominal_values(ne) < 0) > 0:
        print('WARNING: Negative electron density is unphysical.')
    return ne


def TO2(oiia, oiib, oiic, oiid, n):
    assert (is_unc(oiia))
    assert (is_unc(oiib))
    assert (is_unc(oiic))
    assert (is_unc(oiid))
    assert (is_unc(n))
    RO2 = (oiic + oiid) / (oiia + oiib)
    return a0_n(n) + (a1_n(n) * RO2) + (a2_n(n) / RO2)


def TN2(niia, niib, niic):
    assert (is_unc(niia))
    assert (is_unc(niib))
    assert (is_unc(niic))
    RN2 = (niia + niib) / niic
    t_NII = 0.6153 - (1.529e-4 * RN2) + (35.3641 / RN2)
    return t_NII


def TS2(siia, siib, siic, siid, n):
    assert (is_unc(siia))
    assert (is_unc(siib))
    assert (is_unc(siic))
    assert (is_unc(siid))
    assert (is_unc(n))
    RS2_p = (siia + siib) / (siic + siid)
    term1 = s0_n(n)
    term2 = s1_n(n) * RS2_p
    term3 = s2_n(n) / RS2_p
    term4 = s3_n(n) / RS2_p / RS2_p
    tS2 = term1 + term2 + term3 + term4
    return tS2


def TS3(siiia, siiib, siiic):
    assert (is_unc(siiia))
    assert (is_unc(siiib))
    assert (is_unc(siiic))
    RS3 = (siiia + siiib) / siiic
    tS3 = 0.5147 + (3.187e-4 * RS3) + (23.64041 / RS3)
    return tS3


# =============================================================================
# Auxiliary Functions
# =============================================================================

# For the density n([SII]), polynomials as a function of temperature

def a0_t(t):
    assert (is_unc(t))
    return 16.054 - (7.79 / t) - (11.32 * t)


def a1_t(t):
    assert (is_unc(t))
    return -22.66 + (11.08 / t) + (16.02 * t)


def b0_t(t):
    assert (is_unc(t))
    return -21.61 + (11.89 / t) + (14.59 * t)


def b1_t(t):
    assert (is_unc(t))
    return 9.17 - (5.09 / t) - (6.18 * t)


# For the low excitation zone temperature [OII], polynomials as a function of
# the density


def a0_n(n):
    assert (is_unc(n))
    return 0.2526 - (3.57e-4 * n) - (0.43 / n)


def a1_n(n):
    assert (is_unc(n))
    return 1.36e-3 + (5.42e-6 * n) + (4.81e-3 / n)


def a2_n(n):
    assert (is_unc(n))
    return 35.624 - (0.0172 * n) + (25.12 / n)


# For the low excitation zone temperature [SII], polynomials as a function of
# the density


def s0_n(n):
    assert (is_unc(n))
    term1 = 34.79 / n
    term2 = 321.82 / n / n
    return 0.99 + term1 + term2


def s1_n(n):
    assert (is_unc(n))
    term1 = 0.628 / n
    term2 = 5.744 / n / n
    return -0.0087 + term1 + term2


def s2_n(n):
    assert (is_unc(n))
    term1 = (926.5 / n)
    term2 = 94.78 / n / n
    return -7.123 + term1 - term2


def s3_n(n):
    assert (is_unc(n))
    term1 = 768.852 / n
    term2 = 5113 / n / n
    return 102.82 + term1 - term2


# =============================================================================
# Metallicity & more abundances
# =============================================================================


def single_log_OH(oiic, oiid, hb, tl, n):
    assert (is_unc(oiic))
    assert (is_unc(oiid))
    assert (is_unc(hb))
    assert (is_unc(tl))
    assert (is_unc(n))
    oii = oiic + oiid
    log = upy.log10(oii / hb)
    s1 = 5.887 + (1.641 / tl) - (0.543 * upy.log10(tl)) + (1.14e-4 * n)
    return log + s1


def double_log_OH(oiiia, oiiib, hb, th):
    assert (is_unc(oiiia))
    assert (is_unc(oiiib))
    assert (is_unc(hb))
    assert (is_unc(th))
    oiii = oiiia + oiiib
    log = upy.log10(oiii / hb)
    s1 = 6.1868 + (1.2491 / th) - (0.5816 * upy.log10(th))
    return log + s1


def log_OH_12(oiiia, oiiib, oiic, oiid, hb, tl, th, n):
    single_OH = 10**(single_log_OH(oiic, oiid, hb, tl, n) - 12)
    double_OH = 10**(double_log_OH(oiiia, oiiib, hb, th) - 12)
    total_log_OH = 12 + upy.log10(single_OH + double_OH)
    return total_log_OH


def log_NO(niia, oiic, oiid, tl):
    assert (is_unc(niia))
    assert (is_unc(oiic))
    assert (is_unc(oiid))
    assert (is_unc(tl))
    oii = oiic + oiid
    log = upy.log10(niia / oii)
    s1 = 0.493 - (0.025 * tl) - (0.687 / tl) + (0.1621 * upy.log(tl))
    return log + s1


def log_NH_12(niia, niib, hb, tl):
    assert (is_unc(niia))
    assert (is_unc(niib))
    assert (is_unc(hb))
    assert (is_unc(tl))
    n_ratio = (niia + niib) / hb
    log = upy.log10(n_ratio)
    s1 = 6.291 + (0.90221 / tl) - (0.5511 * upy.log10(tl))
    return log + s1


# =============================================================================
#
# One realization
#
# =============================================================================


def compute_all(L, galaxies, plot_path=None):
    OII3727 = L['OII3727']
    OII3729 = L['OII3729']
    NE3868 = L['NE3868']
    OIII4363 = L['OIII4363']
    SII4068 = L['SII4068']
    SII4076 = L['SII4076']
    Hb = L['Hb']
    OIII4959 = L['OIII4959']
    OIII5007 = L['OIII5007']
    NII5755 = L['NII5755']
    NII6548 = L['NII6548']
    Ha = L['Ha']
    NII6584 = L['NII6584']
    SII6716 = L['SII6716']
    SII6730 = L['SII6730']
    OII7319 = L['OII7319']
    OII7330 = L['OII7330']
    Hg = L['Hg']
    ARIII7136 = L['ARIII7136']

    N2 = N2_func(NII6584, Ha)
    R3 = R3_func(OIII5007, Hb)
    R2 = R2_func(OII3729, OII3727, Hb)
    R23 = R23_func(OIII5007, OIII4959, OII3729, OII3727, Hb)
    O3N2 = O3N2_func(OIII5007, Hb, NII6584, Ha)
    R3N2 = R3N2_func(OIII5007, NII6584)
    N2O2 = N2O2_func(NII6584, OII3727)
    O3O2 = O3O2_func(OIII5007, OII3729, OII3727)
    S2 = S2_func(SII6730, SII6716, Ha)
    O3S2 = O3S2_func(OIII5007, Hb, SII6730, SII6716, Ha)
    N2S2 = N2S2_func(NII6584, SII6730, SII6716)
    NE3O2 = NE3O2_func(NE3868, OII3729, OII3727)
    R = 0.47*(R2) + 0.88*(R3)
    NE3 = NE3_func(NE3868, Hg)
    RO2NE3 = RO2NE3_func(NE3868, OII3729, OII3727, Hg)
    AR3 = AR3_func(ARIII7136, Ha)
    AR3O3 = AR3O3_func(ARIII7136, OIII5007)

    t_high = positive(TO3(OIII5007, OIII4959, OIII4363))
    t_low_NII = positive(TN2(NII6584, NII6548, NII5755))

    n_e = positive(ne(SII6730, SII6716, t_high))
    t_low_OII = positive(TO2(OII7330, OII7319, OII3729, OII3727, n_e))
    t_low_SII = positive(TS2(SII6730, SII6716, SII4076, SII4068, n_e))

    plot = plot_path is not None
    ne_SII, tS2_ne = Tlow_ne('TS2', plot_path, galaxies, OII7330, OII7319,
                             OII3729, OII3727, SII6730, SII6716, SII4076,
                             SII4068, plot=plot)
    ne_OII, tO2_ne = Tlow_ne('TO2', plot_path, galaxies, OII7330, OII7319,
                             OII3729, OII3727, SII6730, SII6716, SII4076,
                             SII4068, plot=plot)
    ne_SII = positive(ne_SII)
    tS2_ne = positive(tS2_ne)
    ne_OII = positive(ne_OII)
    tO2_ne = positive(tO2_ne)

    oxygen = log_OH_12(OIII5007, OIII4959, OII3729, OII3727, Hb,
                       tO2_ne, t_high, ne_OII)
    nitrogen = log_NH_12(NII6584, NII6548, Hb, tO2_ne)
    NO_ratio = log_NO(NII6584, OII3729, OII3727, tO2_ne)

    out = {'tOIII': t_high, 'tNII': t_low_NII, 'ne_OIII': n_e,
           'tOII': t_low_OII, 't(SII)_OIII': t_low_SII,
           'ne_OII': ne_OII, 'ne_SII': ne_SII, 'tOII_ne': tO2_ne,
           't(SII)_ne': tS2_ne, '12_log_OH': oxygen, '12_log_NH': nitrogen,
           'log_NO': NO_ratio, 'N2': N2, 'R3': R3, 'R2': R2, 'R23': R23,
           'O3N2': O3N2, 'R3N2': R3N2, 'N2O2': N2O2, 'O3O2': O3O2,
           'Ne3O2': NE3O2, 'S2': S2, 'O3S2': O3S2, 'N2S2': N2S2, 'R': R,
           'Ne3': NE3, 'RO2NE3': RO2NE3, 'AR3': AR3, 'AR3O3': AR3O3}
    values = {}
    for key in out:
        values[key] = upy.nominal_values(out[key])
    return values


# =============================================================================
#
# All realizations of one file
#
# =============================================================================


def run_file(path, out_dir, name, plot=False):
    os.makedirs(out_dir, exist_ok=True)
    table = pd.read_csv(path, index_col=0)
    ids = list(table['ID'].unique())

    rows = []
    for i in sorted(table['i'].unique()):
        sub = table[table['i'] == i]
        order = []
        for gal in ids:
            order.append(np.where(sub['ID'].to_numpy() == gal)[0][0])
        sub = sub.iloc[order]
        L = load_lines(sub)
        plot_path = None
        if plot and i == 0:
            plot_path = f'{out_dir}/{name}_'
        values = compute_all(L, sub['ID'].to_numpy(), plot_path)
        for k in range(len(sub)):
            row = {'ID': sub['ID'].iloc[k], 'mass': sub['mass'].iloc[k],
                   'z': sub['z'].iloc[k], 'i': i}
            for key in values:
                row[key] = values[key][k]
            rows.append(row)
        print(f'{name}: realization {i} done')

    real = pd.DataFrame(rows)
    real.to_csv(f'{out_dir}/{name}_realizations.csv', index=False)

    keys = []
    for col in real.columns:
        if col not in ('ID', 'mass', 'z', 'i'):
            keys.append(col)
    summary = []
    for gal in ids:
        d = real[real['ID'] == gal]
        row = {'ID': gal, 'mass': d['mass'].iloc[0], 'z': d['z'].iloc[0]}
        for key in keys:
            v = d[key].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                row[key] = np.nan
                row[f'{key}_ERR'] = np.nan
                row[f'{key}_N'] = 0
                continue
            p16 = np.percentile(v, 16)
            p50 = np.percentile(v, 50)
            p84 = np.percentile(v, 84)
            row[key] = p50
            row[f'{key}_ERR'] = (p84 - p16) / 2
            row[f'{key}_N'] = len(v)
        summary.append(row)
    summ = pd.DataFrame(summary)
    summ.to_csv(f'{out_dir}/{name}_summary.csv', index=False)
    print(f'Saved {out_dir}/{name}_realizations.csv and {name}_summary.csv')
    return summ, real

# =============================================================================
#
# Program
#
# =============================================================================

if __name__ == '__main__':

    master_dir = docPath + '/new_results/master'
    out_dir = docPath + '/new_results/master/te'
    tag = '_mw_abs'
    for law in ['none', 'ccm', 'calzetti', 'power']:
        path = f'{master_dir}/magE2024_master{tag}_{law}_realizations.csv'
        run_file(path, out_dir, f'temden{tag}_{law}', plot=True)
