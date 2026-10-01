import numpy as np
import pandas as pd
from uncertainties import unumpy as upy
from uncertainties import ufloat
# import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

DIR = '/Users/javieratoro/Desktop/thesis/proyecto 2024-2/'


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
    Calzetti extinction curve with Rv = 3.1, safe for any wavelength
    array (does not assume the input is sorted ascending).

    Params
    ------
        wl : array of wavelengths (Angstrom)
        Rv : total-to-selective extinction ratio

    Output
    ------
        k : array of Calzetti curve values, same shape/order as wl
    """
    wl = np.asarray(wl, dtype=float)
    wl_um = wl / 1e4  # convert to um
    k = np.empty_like(wl_um)

    mask_low = wl < 6300
    mask_high = ~mask_low

    k[mask_low] = (2.659 * (-2.156 + (1.509 / wl_um[mask_low])
                            - (0.198 / wl_um[mask_low]**2)
                            + (0.011 / wl_um[mask_low]**3)) + Rv)
    k[mask_high] = 2.659 * (-1.857 + 1.040 / wl_um[mask_high]) + Rv

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

    lines = pd.read_csv(DIR + 'CSV_files/emission_line1.csv')
    fluxes = pd.read_csv(DIR + f'lines/{name}_line_fluxes_parts.csv')

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
                      'Ha/Hb_err': (Ha_flux / Hb_flux).std_dev,
                      'Hg/Hb': (Hc_flux / Hb_flux).nominal_value,
                      'Hg/Hb_err': (Hc_flux / Hb_flux).std_dev,
                      'Hd/Hb': (Hd_flux / Hb_flux).nominal_value,
                      'Hd/Hb_err': (Hd_flux / Hb_flux).std_dev,
                      'E_BV': E_BV,
                      'E_BV_err': E_BV_ERR
                      }

    decrement_dict_pd = pd.DataFrame(data=decrement_dict, index=[0])
    path = '/results/bal_decrements_parts.csv'
    save_catalog(name, decrement_dict_pd, DIR + path)
    return E_BV, E_BV_ERR


def f_int(wl, line, E_BV):
    """
    Dust-correct a SINGLE line flux (or array of per-galaxy flux
    values for one line), given a wavelength and E(B-V).
    """
    if type(wl) is not np.ndarray:
        wl = np.asarray(wl)
    print(wl, line)
    return line * 10 ** (0.4 * E_BV * k_cal(wl))


def deredden_spectrum(wave, flux, E_BV, flux_err=None):
    """
    Apply the Calzetti dust attenuation correction to a FULL spectrum
    (wavelength + flux arrays), rather than a single line flux.

    Parameters
    ----------
    wave : array
        Wavelength array (rest-frame, Angstrom).
    flux : array
        Flux array, same length as wave.
    E_BV : float
        Color excess, e.g. from E_BV_() (derived from the Balmer
        decrement of already absorption-corrected line fluxes).
    flux_err : array or None
        Optional flux error array, corrected with the same factor.

    Returns
    -------
    corrected_flux : array
    corrected_err : array (only returned if flux_err was given)
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    k = k_cal(wave)
    atten_factor = 10 ** (0.4 * E_BV * k)

    corrected_flux = flux * atten_factor

    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=float)
        corrected_err = flux_err * atten_factor
        return corrected_flux, corrected_err

    return corrected_flux


def dust_correct_galaxy_spectrum(name, wave, flux, flux_err=None):
    """
    Compute E(B-V) for a galaxy from its (already absorption-corrected)
    line flux catalog, then apply the dust correction to its full
    spectrum.

    Returns
    -------
    corrected_flux (and corrected_err, if flux_err given), E_BV, E_BV_err
    """
    E_BV, E_BV_err = E_BV_(name)
    print(f'{name}: applying dust correction with '
          f'E(B-V) = {E_BV:.3f} +/- {E_BV_err:.3f}')

    result = deredden_spectrum(wave, flux, E_BV, flux_err=flux_err)

    if flux_err is not None:
        corrected_flux, corrected_err = result
        return corrected_flux, corrected_err, E_BV, E_BV_err

    return result, E_BV, E_BV_err


def save_corrected_spectrum(name, wave, corrected_flux, corrected_err,
                            save_dir=None):
    """
    Save the dust-corrected spectrum to a CSV, matching the same
    file-based handoff pattern used by the Balmer absorption
    correction step (save_corrected_data).
    """
    if save_dir is None:
        save_dir = f'{DIR}dust_corr'
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/dustcorr_{name}_new.csv'
    df = pd.DataFrame({'wave': wave, 'flux': corrected_flux,
                       'sigma': corrected_err})
    df.to_csv(save_path, index=False)

    print(f'Saved dust corrected spectrum to: {save_path}')
    return save_path


def get_flux(name, line):

    fluxes = pd.read_csv(DIR + 'lines/magE2024_master_au_parts.csv')

    for id in fluxes['ID'].values:
        if id[:5] == name:
            name_ = id
    flux = fluxes[fluxes['ID'] == name_][f'{line}_flux'].values[0]
    fluxerr = fluxes[fluxes['ID'] == name_][f'{line}_fluxerr'] .values[0]
    return flux, fluxerr


def save_fluxes():
    lines = pd.read_csv(DIR + 'CSV_files/emission_line1.csv')

    names = ['J0023', 'J0136', 'J0020', 'J0203', 'J0243', 'J0333',
             'J0404', 'J2204', 'J2258', 'J2336', 'J0328']

    columns_ = ['ID', 'mass', 'z_red', 'z_blue']
    for line in lines['name']:
        columns_.append(line + '_flux')
        columns_.append(line + '_fluxerr')

    df = pd.DataFrame(columns=columns_)

    for name in names:
        data = pd.read_csv(DIR + f'lines/{name}/{name}_model_parts.csv')
        all_rows = {'ID': data['ID'][0], 'mass': data['mass'][0],
                    'z_red': data['z_red'][0], "z_blue": data['z_blue'][0]}
        for line in lines['name']:
            print(f'Correcting line {line}')
            wl = lines[lines['name'] == line]['vacuum_wave'].values
            E_BV, _ = E_BV_(name)
            flux, fluxerr = get_flux(name, line)
            f_corr = f_int(wl, flux, E_BV)
            f_corr_err = f_int(wl, fluxerr, E_BV)
            all_rows[line + '_flux'] = f_corr[0]
            all_rows[line + '_fluxerr'] = f_corr_err[0]

        df.loc[-1] = all_rows
        df.index = df.index + 1
    df.to_csv(DIR + 'lines/magE2024_master_Dcorr_parts.csv')


# if __name__ == '__main__':
    # save_fluxes()

    # Example of the new full-spectrum dust correction, once you have
    # a reduced/absorption-corrected spectrum (wave, flux, err) loaded:
    #
    # corrected_flux, corrected_err, E_BV, E_BV_err = \
    #     dust_correct_galaxy_spectrum('J0203', wave, flux, flux_err=err)
    # save_corrected_spectrum('J0203', wave, corrected_flux, corrected_err)
