import numpy as np
import pandas as pd
from uncertainties import unumpy as upy
from uncertainties import ufloat
# import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os


def save_catalog(galname, catalog, path):
    '''
    Function to save and/or update the emission line, equivalent width and
    velocity catalogs taken from the fluxes notebooks for each galaxy.
    '''

    catalog_exist = os.path.exists(path)
    if catalog_exist:
        print('Existent catalog (.csv) file.')

        catalog_all = pd.read_csv(path, on_bad_lines='skip')
        if np.sum(catalog_all['galname'] == galname) == 1:
            print('Existent data for ' + galname + ', overwriting...')
            index = np.where(catalog_all['galname'] == galname)[0][0]
            catalog_all.loc[index] = np.array(catalog.loc[0])
            catalog_all.to_csv(path, index=None)
            print('Done.')

        else:
            print('Non existent data for ' + galname + ', adding new data...')
            catalog.to_csv(path, mode='a',
                           header=False, index=None)
            print('Done.')
    else:
        print('File does not exist, creating new catalog...')
        catalog.to_csv(path, sep=',', index=None)
        print('Done.')


def k_cal(wl, Rv=3.1):
    """
    Calzetti extinction curve with Rv = 3.1. For an array of
    values of wavelength.

    Params
    ------
        wl : wavelength
        Rv : don't remember the name of this cosntant

    Output
    ------
        k : float value of the Calzetti curve at those wavelengths
    """

    wl_low = wl[wl < 6300]/1e4  # convert to um
    wl_high = wl[wl >= 6300]/1e4  # convert to um

    k_low = 2.659 * (-2.156 + (1.509 / wl_low) -
                     (0.198 / wl_low**2) + (0.011 / wl_low**3)) + Rv
    k_high = 2.659 * (-1.857 + 1.040 / wl_high) + Rv

    k = np.concatenate((k_low, k_high))
    return k


def residuals_balmer(params_balmer, x, data, uncertainty):

    slope = params_balmer['slope']
    # x = x_balmer[mask_balmer]
    model = x * (-slope)
    output = (model - data / uncertainty)

    return output


def linfunc(x, slope):
    return -1*x*slope


def E_BV_(name):

    DIR = '/Users/javieratoro/Desktop/proyecto 2024-2/'
    lines = pd.read_csv(DIR + 'CSV_files/emission_lines.csv')
    fluxes = pd.read_csv(DIR + f'lines/{name}_master_model.csv')

    Ha_w = lines[lines['name'] == 'H_alpha']['vacuum_wave'].values[0]
    Hb_w = lines[lines['name'] == 'H_beta']['vacuum_wave'].values[0]
    Hc_w = lines[lines['name'] == 'H_gamma']['vacuum_wave'].values[0]
    Hd_w = lines[lines['name'] == 'H_delta']['vacuum_wave'].values[0]

    x_balmer = np.array([k_cal(Ha_w) - k_cal(Hb_w),
                        k_cal(Hb_w) - k_cal(Hb_w),
                        k_cal(Hc_w) - k_cal(Hb_w),
                        k_cal(Hd_w) - k_cal(Hb_w)])

    Ha_flux = ufloat(np.mean(fluxes['H_alpha_narrow'] +
                             fluxes['H_alpha_broad']),
                     np.std(fluxes['H_alpha_narrow'] +
                            fluxes['H_alpha_broad']))

    Hb_flux = ufloat(np.mean(fluxes['H_beta_narrow'] +
                             fluxes['H_beta_broad']),
                     np.std(fluxes['H_beta_narrow'] +
                            fluxes['H_beta_broad']))

    Hc_flux = ufloat(np.mean(fluxes['H_gamma_narrow'] +
                             fluxes['H_gamma_broad']),
                     np.std(fluxes['H_gamma_narrow'] +
                            fluxes['H_gamma_broad']))

    Hd_flux = ufloat(np.mean(fluxes['H_delta_narrow']),
                     np.std(fluxes['H_delta_narrow']))

    y_balmer = upy.log10([(Ha_flux / Hb_flux) / 2.86,
                          (Hb_flux / Hb_flux) / 1.0,
                          (Hc_flux / Hb_flux) / 0.464,
                          (Hd_flux / Hb_flux) / 0.256])
    mask_balmer = (y_balmer != 0)
    y = y_balmer[mask_balmer]
    x = np.asarray(x_balmer[mask_balmer]).flatten()

    popt, pcov = curve_fit(linfunc, x,
                           upy.nominal_values(y),
                           0.1, upy.std_devs(y))
    E_BV = popt[0]/0.4
    E_BV_ERR = np.sqrt(np.diag(pcov))[0]/0.4
    decrement_dict = {'galname': name,
                      'Ha/Hb': (Ha_flux / Hb_flux).nominal_value,
                      'Ha/Hb_err': (Hb_flux / Hb_flux).std_dev,
                      'Hg/Hb': (Hc_flux / Hb_flux).nominal_value,
                      'Hg/Hb_err': (Hc_flux / Hb_flux).std_dev,
                      'Hd/Hb': (Hd_flux / Hb_flux).nominal_value,
                      'Hd/Hb_err': (Hd_flux / Hb_flux).std_dev,
                      'E_BV': E_BV,
                      'E_BV_err': E_BV_ERR
                      }

    decrement_dict_pd = pd.DataFrame(data=decrement_dict, index=[0])
    save_catalog(name, decrement_dict_pd, DIR + '/results/bal_decrements.csv')
    return E_BV, E_BV_ERR


def f_int(wl, line, E_BV):
    if type(wl) is not np.ndarray:
        wl = np.asarray(wl)
    return line * 10 ** (0.4 * E_BV * k_cal(wl))


def get_flux(name, line):
    DIR = '/Users/javieratoro/Desktop/proyecto 2024-2/'

    bright_lines = ['O2_3725', 'O2_3727', 'H_alpha', 'H_beta',
                    'H_gamma', 'O3_5008', 'O3_4959', 'N2_6550',
                    'N2_6585', 'S2_6716', 'S2_6730']

    auroral_lines = ['N2_5756', 'O1_6363',
                     'O3_4363', 'S3_6312',
                     'O2_7322', 'O2_7333',
                     'O2_7322_7333']

    fluxes = pd.read_csv(DIR + f'lines/{name}_master_model.csv')

    if line in auroral_lines:
        narrow_ = fluxes[line + '_narrow_amplitude'].values
        broad_ = fluxes[line + '_broad_amplitude'].values
        flux = narrow_ + broad_
    else:
        narrow = fluxes[line + '_narrow'].values
        flux = narrow
        if line in bright_lines:
            broad = fluxes[line + '_broad'].values
            flux += broad

    return flux


def save_fluxes():
    DIR = '/Users/javieratoro/Desktop/proyecto 2024-2/'
    lines = pd.read_csv(DIR + 'CSV_files/emission_lines.csv')

    names = ['J0020', 'J0203', 'J0243', 'J0033',
             'J2204', 'J2258', 'J2336',
             'J0023', 'J0136']

    columns_ = ['ID', 'mass', 'z']
    for line in lines['name']:
        columns_.append(line + '_flux')
        columns_.append(line + '_fluxerr')

    df = pd.DataFrame(columns=columns_)

    for name in names:
        data = pd.read_csv(DIR + f'lines/{name}.csv')
        all_rows = {'ID': data['ID'][0], 'mass': data['mass'][0],
                    'z': data['z'][0]}
        for line in lines['name']:
            print(f'Correcting line {line}')
            wl = lines[lines['name'] == line]['vacuum_wave'].values
            E_BV, _ = E_BV_(name)
            flux = get_flux(name, line)
            f_corr = f_int(wl, flux, E_BV)
            all_rows[line + '_flux'] = np.mean(f_corr)
            all_rows[line + '_fluxerr'] = np.std(f_corr)

        df.loc[-1] = all_rows
        df.index = df.index + 1
    df.to_csv(DIR + 'lines/magE2024_master_Dcorr.csv')


save_fluxes()
