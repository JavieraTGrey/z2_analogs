# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy as upy

# Path and useful libs
import os
docPath = '/Users/javieratoro/Desktop/proyecto 2024-2'
magePath = docPath + '/lines/'
os.sys.path.append(magePath)

# The mode defines which component are we using for the direct method and if
# it is corrected by extinction or not.
mode = 'lines'

# =============================================================================
#
# Data preparation
#
# =============================================================================

# Read the data
el_cat = pd.read_csv(magePath+'magE2024_master_Dcorr.csv')
print(el_cat)
# Assigning arrays & emission lines
galaxies = el_cat['ID'].to_numpy()

OII3727 = upy.uarray(el_cat['O2_3725_flux'].to_numpy(),
                     el_cat['O2_3725_fluxerr'].to_numpy())  # oiid

OII3729 = upy.uarray(el_cat['O2_3727_flux'].to_numpy(),
                     el_cat['O2_3727_fluxerr'].to_numpy())  # oiic

OIII4363 = upy.uarray(el_cat['O3_4363_flux'].to_numpy(),
                      el_cat['O3_4363_fluxerr'].to_numpy())  # oiiic

# SII4068 = upy.uarray(el_cat['[SII]4068'].to_numpy(),
#                     el_cat['[SII]4068_err'].to_numpy()) # siib
#
# SII4076 = upy.uarray(el_cat['[SII]4076'].to_numpy(),
#                     el_cat['[SII]4076_err'].to_numpy()) # siia

Hb = upy.uarray(el_cat['H_beta_flux'].to_numpy(),
                el_cat['H_beta_fluxerr'].to_numpy())

OIII4959 = upy.uarray(el_cat['O3_4959_flux'].to_numpy(),
                      el_cat['O3_4959_fluxerr'].to_numpy())  # oiiib

OIII5007 = upy.uarray(el_cat['O3_5008_flux'].to_numpy(),
                      el_cat['O3_5008_fluxerr'].to_numpy())  # oiiia

NII5755 = upy.uarray(el_cat['N2_5756_flux'].to_numpy(),
                     el_cat['N2_5756_fluxerr'].to_numpy())  # niic

OI6300 = upy.uarray(el_cat['O1_6300_flux'].to_numpy(),
                    el_cat['O1_6300_fluxerr'].to_numpy())  # oi

SIII6312 = upy.uarray(el_cat['S3_6312_flux'].to_numpy(),
                      el_cat['S3_6312_fluxerr'].to_numpy())  # siiic

NII6548 = upy.uarray(el_cat['N2_6550_flux'].to_numpy(),
                     el_cat['N2_6550_fluxerr'].to_numpy())  # niib

Ha = upy.uarray(el_cat['H_alpha_flux'].to_numpy(),
                el_cat['H_alpha_fluxerr'].to_numpy())

NII6584 = upy.uarray(el_cat['N2_6585_flux'].to_numpy(),
                     el_cat['N2_6585_fluxerr'].to_numpy())  # niia

SII6716 = upy.uarray(el_cat['S2_6716_flux'].to_numpy(),
                     el_cat['S2_6716_fluxerr'].to_numpy())  # siib

SII6730 = upy.uarray(el_cat['S2_6730_flux'].to_numpy(),
                     el_cat['S2_6730_fluxerr'].to_numpy())  # siia

OII7319 = upy.uarray(el_cat['O2_7322_flux'].to_numpy(),
                     el_cat['O2_7322_fluxerr'].to_numpy())  # oiib

OII7330 = upy.uarray(el_cat['O2_7333_flux'].to_numpy(),
                     el_cat['O2_7333_fluxerr'].to_numpy())  # oiia

# SIII9069 = upy.uarray(el_cat['[SIII]9069'].to_numpy(),
#                       el_cat['[SIII]9069_err'].to_numpy())  # siiib

# SIII9532 = upy.uarray(el_cat['[SIII]9532'].to_numpy(),
#                       el_cat['[SIII]9532_err'].to_numpy())  # siiia


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

    assert(is_unc(unc_array))

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
    assert(is_unc(uarray))
    uarray_ul = np.where(mask, uarray,
                         upy.uarray(upy.nominal_values(uarray),
                                    0 * upy.nominal_values(uarray)))
    return uarray_ul

# =============================================================================
# More functions
# =============================================================================


def Tlow_ne(temp, path):
    '''
    This function gives a consistent estimation of the electron density and the
    temperature of the low ionization zone (either T02 or TS2). The reasoning
    behind is based on the way the density and the low ionization temperature
    are defined depending on each other. Each galaxy defines a ratio,
    determining the shape of both functions. Thus, by evaluating both expresions
    we can find the point where they cross, giving a density-temperature pair
    (ne, Tl). Plots are generated to assess the estimation.

    Params
    ------
    temp : str
        String indicating the temperature to use for the estimation. (e.g. 'TO2')

    galname : str
        Name of the galaxy. (e.g. 'J0021')

    path : str
        Path to save the plots.


    Output
    ------

    ne_it : uncertainty array
        Array of calculated densities through this method along with their
        errors.

    t_low_it : uncertainty array
        Array of calculated temperatures through this method along with their
        errors.

    '''

    # Explore plots of density and low ionization temperature
    # Reasonable density range for our galaxies.
    ne_toplot = upy.uarray(np.linspace(10, 1000, 1000),
                           np.full_like(np.linspace(10, 1000, 1000), 0))

    # Lists to store the temperature and error values.
    t_low_it = []
    t_low_it_err = []

    # For each galaxy in the sample
    for i in range(len(galaxies)):

        print('Plotting density vs temperature for galaxy:', galaxies[i])

        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches((9, 3))

        # Depending if you are using TO2 or TS2...
        if temp == 'TO2':
            t_n = TO2(OII7330[i], OII7319[i], OII3729[i], OII3727[i],
                      ne_toplot)
        elif temp == 'TS2':
            print('No flux reported for emission line [SII]4076 and [SII]4068')
            # t_n = TS2(SII6730[i], SII6716[i], SII4076[i], SII4068[i],
            # ne_toplot)

        # Density from this temperature
        n_t_n = ne(SII6730[i], SII6716[i], t_n)

        # t(n) vs n
        axs[0].plot(upy.nominal_values(t_n), upy.nominal_values(ne_toplot))

        # t(n) vs n(t(n))
        axs[0].plot(upy.nominal_values(t_n), upy.nominal_values(n_t_n))

        # n - n(t(n)) vs t(n)
        axs[1].plot(upy.nominal_values(t_n),
                    np.abs(upy.nominal_values(ne_toplot) \
                           - upy.nominal_values(n_t_n)))

        # minimum is the zero
        index_min = np.argmin(np.abs(upy.nominal_values(ne_toplot) \
                                     -  upy.nominal_values(n_t_n)))

        axs[0].set_xlabel('Low ionization temperature')
        axs[0].set_ylabel('Electron density')
        axs[0].set_title(galaxies[i])
        axs[1].set_xlabel(temp)
        axs[1].set_ylabel('|n - n(t(n))|')
        axs[1].axvline(upy.nominal_values(t_n)[index_min], ls='--')

        print('Value where functions cross: '+temp+' =', t_n[index_min])
        print(type(upy.nominal_values(t_n)[index_min]))
        t_low_it.append(upy.nominal_values(t_n)[index_min])
        t_low_it_err.append(upy.std_devs(t_n)[index_min])

        fig.savefig(magePath + f'images/n_vs_tO2_{galaxies[i]}.pdf', bbox_inches='tight')
        print()
        plt.close()

    # T low and n_e determined simultaneously
    t_low_new = upy.uarray(t_low_it, t_low_it_err)
    n_e_new = ne(SII6730, SII6716, t_low_new)

    return n_e_new, t_low_new

# =============================================================================
# Line ratios
# =============================================================================


def N2_func(niia, ha):
    assert(is_unc(niia))
    assert(is_unc(ha))

    if has_uplim(niia) or has_uplim(ha):
        mask = (upy.std_devs(niia) == 0.) + (upy.std_devs(ha) == 0.)
        return assign_uplims(upy.log10(niia / ha), mask)
    else:
        return upy.log10(niia / ha)


def R3_func(oiiia, hb):
    assert(is_unc(oiiia))
    assert(is_unc(hb))

    if has_uplim(oiiia) or has_uplim(hb):
        mask = (upy.std_devs(oiiia) == 0.) + (upy.std_devs(hb) == 0.)
        return assign_uplims(upy.log10(oiiia / hb), mask)
    else:
        return upy.log10(oiiia / hb)


def R2_func(oiic, oiid, hb):
    assert(is_unc(oiic))
    assert(is_unc(oiid))
    assert(is_unc(hb))

    oii = oiic + oiid

    if has_uplim(oiic) or has_uplim(oiid) or has_uplim(hb):
        mask = (upy.std_devs(oiic) == 0.) + (upy.std_devs(hb) == 0.) + \
               (upy.std_devs(oiid) == 0.)
        return assign_uplims(upy.log10(oii / hb), mask)
    else:
        return upy.log10(oii / hb)


def R23_func(oiiia, oiiib, oiic, oiid, hb):
    assert(is_unc(oiiia))
    assert(is_unc(oiiib))
    assert(is_unc(oiic))
    assert(is_unc(oiid))
    assert(is_unc(hb))

    oiii = oiiia + oiiib
    oii = oiic + oiid
    num = oiii + oii
    return upy.log10(num / hb)


def O3N2_func(oiiia, hb, niia, ha):
    assert(is_unc(oiiia))
    assert(is_unc(hb))
    assert(is_unc(niia))
    assert(is_unc(ha))

    num = oiiia / hb
    den = niia / ha
    return upy.log10(num / den)


def R3N2_func(oiiia, niia):
    assert(is_unc(oiiia))
    assert(is_unc(niia))
    return upy.log10(oiiia / niia)


def N2O2_func(niia, oiid):
    assert(is_unc(niia))
    assert(is_unc(oiid))
    return upy.log10(niia / oiid)


def O3O2_func(oiiia, oiic, oiid):
    assert(is_unc(oiiia))
    assert(is_unc(oiic))
    assert(is_unc(oiid))
    oii = oiic + oiid
    return upy.log10(oiiia / oii)


def Ne3O2_func(neiiia, oiic, oiid):
    assert(is_unc(neiiia))
    assert(is_unc(oiic))
    assert(is_unc(oiid))
    oii = oiic + oiid
    return upy.log10(neiiia / oii)


def S2_func(siia, siib, ha):
    assert(is_unc(siia))
    assert(is_unc(siib))
    assert(is_unc(ha))
    sii = siia + siib
    return upy.log10(sii / ha)


def O3S2_func(oiiia, hb, siia, siib, ha):
    assert(is_unc(oiiia))
    assert(is_unc(hb))
    assert(is_unc(siia))
    assert(is_unc(siib))
    assert(is_unc(ha))
    sii = siia + siib
    num = oiiia / hb
    den = sii / ha
    return upy.log10(num / den)


def N2S2_func(niia, siia, siib, ha):
    assert(is_unc(niia))
    assert(is_unc(siia))
    assert(is_unc(siib))
    assert(is_unc(ha))
    sii = siia + siib
    s1 = niia / sii
    s2 = niia / ha
    # According to Garg et al. 2023
    return upy.log10(s1 + 0.264 * s2)


# =============================================================================
# Electron density & temperatures
# =============================================================================


def TO3(oiiia, oiiib, oiiic):
    assert(is_unc(oiiia))
    assert(is_unc(oiiib))
    assert(is_unc(oiiic))
    RO3 = (oiiia + oiiib) / oiiic
    t_OIII = 0.784 - (1.357e-4 * RO3) + (48.44 / RO3)
    return t_OIII


def ne(siia, siib, to3):
    assert(is_unc(siia))
    assert(is_unc(siib))
    assert(is_unc(to3))
    RS2 = siib / siia
    ne = 1e3 * ((RS2 * a0_t(to3)) + a1_t(to3)) / ((RS2 * b0_t(to3)) + b1_t(to3))
    if np.sum(upy.nominal_values(ne) < 0) > 0:
        print('WARNING: Negative electron density is unphysical.')
    return ne


def TO2(oiia, oiib, oiic, oiid, n):
    assert(is_unc(oiia))
    assert(is_unc(oiib))
    assert(is_unc(oiic))
    assert(is_unc(oiid))
    assert(is_unc(n))
    RO2 = (oiic + oiid) / (oiia + oiib)
    return a0_n(n) + (a1_n(n) * RO2) + (a2_n(n) / RO2)


def TN2(niia, niib, niic):
    assert(is_unc(niia))
    assert(is_unc(niib))
    assert(is_unc(niic))
    RN2 = (niia + niib) / niic
    t_NII = 0.6153 - (1.529e-4 * RN2) + (35.3641 / RN2)
    return t_NII


def TS2(siia, siib, siic, siid, n):
    assert(is_unc(siia))
    assert(is_unc(siib))
    assert(is_unc(siic))
    assert(is_unc(siid))
    assert(is_unc(n))
    RS2_p = (siia + siib) / (siic + siid)
    term1 = s0_n(n)
    term2 = s1_n(n) * RS2_p
    term3 = s2_n(n) / RS2_p
    term4 = s3_n(n) / RS2_p / RS2_p
    tS2 = term1 + term2 + term3 + term4
    return tS2


def TS3(siiia, siiib, siiic):
    assert(is_unc(siiia))
    assert(is_unc(siiib))
    assert(is_unc(siiic))
    RS3 = (siiia + siiib) / siiic
    tS3 = 0.5147 + (3.187e-4 * RS3) + (23.64041 / RS3)
    return tS3


# =============================================================================
# Auxiliary Functions
# =============================================================================

# For the density n([SII]), polynomials as a function of temperature

def a0_t(t):
    assert(is_unc(t))
    return 16.054 - (7.79 / t) - (11.32 * t)


def a1_t(t):
    assert(is_unc(t))
    return -22.66 + (11.08 / t) + (16.02 * t)


def b0_t(t):
    assert(is_unc(t))
    return -21.61 + (11.89 / t) + (14.59 * t)


def b1_t(t):
    assert(is_unc(t))
    return 9.17 - (5.09 / t) - (6.18 * t)


# For the low excitation zone temperature [OII], polynomials as a function of
# the density


def a0_n(n):
    assert(is_unc(n))
    return 0.2526 - (3.57e-4 * n) - (0.43 / n)


def a1_n(n):
    assert(is_unc(n))
    return 1.36e-3 + (5.42e-6 * n) + (4.81e-3 / n)


def a2_n(n):
    assert(is_unc(n))
    return 35.624 - (0.0172 * n) + (25.12 / n)


# For the low excitation zone temperature [SII], polynomials as a function of
# the density


def s0_n(n):
    assert(is_unc(n))
    term1 = 34.79 / n
    term2 = 321.82 / n / n
    return 0.99 + term1 + term2


def s1_n(n):
    assert(is_unc(n))
    term1 = 0.628 / n
    term2 = 5.744 / n / n
    return -0.0087 + term1 + term2


def s2_n(n):
    assert(is_unc(n))
    term1 = (926.5 / n)
    term2 = 94.78 / n / n
    return -7.123 + term1 - term2


def s3_n(n):
    assert(is_unc(n))
    term1 = 768.852 / n
    term2  = 5113 / n / n
    return 102.82 + term1 - term2


# =============================================================================
# Metallicity & more abundances
# =============================================================================


def single_log_OH(oiic, oiid, hb, tl, n):
    assert(is_unc(oiic))
    assert(is_unc(oiid))
    assert(is_unc(hb))
    assert(is_unc(tl))
    assert(is_unc(n))
    oii = oiic + oiid
    log = upy.log10(oii / hb)
    s1 = 5.887 + (1.641 / tl) - (0.543 * upy.log10(tl)) + (1.14e-4 * n)
    return log + s1


def double_log_OH(oiiia, oiiib, hb, th):
    assert(is_unc(oiiia))
    assert(is_unc(oiiib))
    assert(is_unc(hb))
    assert(is_unc(th))
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
    assert(is_unc(niia))
    assert(is_unc(oiic))
    assert(is_unc(oiid))
    assert(is_unc(tl))
    oii = oiic + oiid
    log = upy.log10(niia / oii)
    s1 = 0.493 - (0.025 * tl) - (0.687 / tl) + (0.1621 * upy.log(tl))
    return log + s1


def log_NH_12(niia, niib, hb, tl):
    assert(is_unc(niia))
    assert(is_unc(niib))
    assert(is_unc(hb))
    assert(is_unc(tl))
    n_ratio = (niia + niib) / hb
    log = upy.log10(n_ratio)
    s1 = 6.291 + (0.90221 / tl) - (0.5511 * upy.log10(tl))
    return log + s1


# =============================================================================
#
# Program
#
# =============================================================================

if __name__ == '__main__':

    # Calculate ratios
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
    N2S2 = N2S2_func(NII6584, SII6730, SII6716, Ha)

    # Temperatures and densities
    t_high = TO3(OIII5007, OIII4959, OIII4363)
    # t_mid = TS3(SIII9532, SIII9069, SIII6312)
    t_low_NII = TN2(NII6584, NII6548, NII5755)

    # First determination of density & T[OII] temperature.
    n_e = ne(SII6730, SII6716, t_high)
    t_low_OII = TO2(OII7330, OII7319, OII3729, OII3727, n_e)
    # t_low_SII = TS2(SII6730, SII6716, SII4076, SII4068, n_e)

    if mode == 'lines':
        # ne_SII, tS2_ne = Tlow_ne('TS2', magePath + 'results/metals/n_vs_tS2_')
        ne_OII, tO2_ne = Tlow_ne('TO2', magePath)

    else:
        ne_SII = np.full_like(n_e, np.nan)
        ne_OII = np.full_like(n_e, np.nan)
        tS2_ne = np.full_like(n_e, np.nan)
        tO2_ne = np.full_like(n_e, np.nan)

    print('T[OIII] '+mode, t_high)
    print('Densities '+mode, n_e)

    # Abundances calculations. Using values for T_low and n determined together
    if mode == 'lines':
        oxygen = log_OH_12(OIII5007, OIII4959, OII3729, OII3727, Hb,
                           tO2_ne, t_high, ne_OII)
        nitrogen = log_NH_12(NII6584, NII6548, Hb, tO2_ne)
        NO_ratio = log_NO(NII6584, OII3729, OII3727, tO2_ne)

    else:
        oxygen = log_OH_12(OIII5007, OIII4959, OII3729, OII3727, Hb,
                           t_low_OII, t_high, n_e)
        nitrogen = log_NH_12(NII6584, NII6548, Hb, t_low_OII)
    # Build and save catalogs

    # Temperatures and densities
    temden = pd.DataFrame({'Source': galaxies,
                           'tOIII': upy.nominal_values(t_high),
                           'tOIII_err': upy.std_devs(t_high),

                           # T(SIII) = T_intermedium
                           # 't(SIII)': upy.nominal_values(t_mid),
                           # 't(SIII)_err': upy.std_devs(t_mid),

                           'tNII': upy.nominal_values(t_low_NII),
                           'tNII_err': upy.std_devs(t_low_NII),

                           # Density estiamted using T_high
                           'ne_OIII': upy.nominal_values(n_e),
                           'ne_OIII_err': upy.std_devs(n_e),

                           # T(O2) = T_low estimated using density from T_high
                           'tOII': upy.nominal_values(t_low_OII),
                           'tOII_err': upy.std_devs(t_low_OII),

                           # T(S2) = T_low estimated using density from T_high
                           # 't(SII)_OIII': upy.nominal_values(t_low_SII),
                           # 't(SII)_OIII_err': upy.std_devs(t_low_SII),

                           # Density determined by consistent ne vs TO2
                           'ne_OII': upy.nominal_values(ne_OII),
                           'ne_OII_err': upy.std_devs(ne_OII),

                           # Density determined by consistent ne vs TS2
                           # 'ne_SII': upy.nominal_values(ne_SII),
                           # 'ne_SII_err': upy.std_devs(ne_SII),

                           # T(O2) = T_low determined  by consistent ne vs TO2
                           'tOII_ne': upy.nominal_values(tO2_ne),
                           'tOII_ne_err': upy.std_devs(tO2_ne),

                           # T(O2) = T_low determined  by consistent ne vs TS2
                           # 't(SII)_ne': upy.nominal_values(tS2_ne),
                           # 't(SII)_ne_err': upy.std_devs(tS2_ne),

                           # Oxygen abundance
                           '12_log_OH': upy.nominal_values(oxygen),
                           '12_log_OH_err': upy.std_devs(oxygen),

                           # Nitrogen Abundance
                           '12_log_NH': upy.nominal_values(nitrogen),
                           '12_log_NH_err': upy.std_devs(nitrogen),

                           # N/O ratio
                           'log_NO': upy.nominal_values(NO_ratio),
                           'log_NO_err': upy.std_devs(NO_ratio),

                           })

    # Saving
    temden.to_csv(magePath + f'ratios/temden_{mode}.csv', index=None)
