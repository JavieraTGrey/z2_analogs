import re
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d

# new module for spectra resample and 
# natural spline. modified by BN 2021

# from spectres import spectres
from .natural_spline import *
from sklearn.neighbors import KernelDensity

# for dust correction. BN 03/01/2022

import uncertainties.umath as uum
from uncertainties import ufloat
from uncertainties import unumpy

_MATCH_STRING = re.compile(r"(\d+) \d+ \d+ (\d+\.\d+) (\d+\.\d+) "
                          r"\d+ \d+\.\d+ -?\d+\.\d+ -?\d+\.\d+")

class SimpleSpectrum(object):

    def __init__(self, lda=[], flux=[], e_flux=None):
        """
        lda : np.ndarray 1D
            The reference wavelengths in \AA. Assumed to be monotonically
            increasing.
        flux : np.ndarray 1D
            The flux at each wavelength set by lda. Assumes units are
            [erg/s/cm^2/\AA]. Same shape as lda.
        e_flux : np.ndarray 1D
            The statistical error associated to the flux. It must have same
            shape as flux.
        """
        self.lda = np.array(lda).copy()
        self.flux = np.array(flux).copy()
        self.e_flux = None if e_flux is None else np.array(e_flux).copy()

    def plot(self, ax=None, label=None, z=0):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.lda / (1 + z), self.flux, label=label)

    def to_fits(self, path, overwrite=True):
        data = Table({'FLUX': self.flux,
                      'WAVELENGTH': self.lda})
        if self.e_flux is not None:
            data['E_FLUX'] = self.e_flux
        data.write(path, overwrite=overwrite)

    @classmethod
    def from_fitstable(cls, path, flux='FLUX', lda='WAVELENGTH',
                       e_flux='STATERROR', hdu_nbr=1):
        data = fits.getdata(path, hdu_nbr)
        output = cls(flux=data[flux], lda=data[lda])
        if e_flux is not None:
            output.e_flux = data[e_flux]
        return output


class MageMultispec(object):
    """
    Useful to manipulate the multispec files that come out of the carpy
    pipeline to reduce MagE data.
    """

    def __init__(self, data, header):
        self.data = data.copy()
        self.header = header.copy()
        self.set_wave_solution()
        self.first_order = min(self.wave_solution.keys())
        self.n_orders = self.data.shape[1]
        self.calibrated_flux = None

    def set_wave_solution(self):
        wspec_wave_solution_string = re.findall(r"WAT2_\d+=\s('.+')",
                                                repr(self.header))
        wspec_wave_solution_string = ''.join(wspec_wave_solution_string)
        wspec_wave_solution_string = wspec_wave_solution_string.replace("'", "")
        wspec_wave_solution_list = re.findall(r'spec\d+ = "(.+?)"',
                                              wspec_wave_solution_string)
        wave_solution = {}
        for l in wspec_wave_solution_list:
            info = _MATCH_STRING.match(l)
            wave_solution[int(info.group(1))] = {'w0': float(info.group(2)),
                                                 'dw': float(info.group(3))}
        self.wave_solution = wave_solution

    def wavelength(self, order):
        N = self.data.shape[-1]
        wavelength = (self.wave_solution[order]['w0'] +
                      self.wave_solution[order]['dw'] * np.arange(N))
        return wavelength

    def _wavelength_grid(self):
        wavelength_grid = []
        for i in range(self.first_order, self.first_order+self.n_orders):
            wavelength_grid.append(self.wavelength(i))
        return np.array(wavelength_grid)

    def _get_spec(self, axis, order):
        return self.data[axis, order-self.first_order, :]

    def sky(self, order):
        return self._get_spec(0, order)

    def flux(self, order):
        return self._get_spec(1, order)

    def noise(self, order):
        return self._get_spec(2, order)

    def ston(self, order):
        return self._get_spec(3, order)

    def lamp(self, order):
        return self._get_spec(4, order)

    def plot(self, order, ax=None, component='flux', z=0):
        if ax==None:
            ax = plt.gca()
        label = 'order = {:d}'.format(order)
        data_to_plot = self.__getattribute__(component)(order)
        ax.plot(self.wavelength(order) / (1+z), data_to_plot, label=label)

    def to_fits(self, path, overwrite=True):
        fits.writeto(path, self.data, header=self.header, overwrite=overwrite)

    @classmethod
    def from_fname(cls, path):
        data, header = fits.getdata(path, 0, header=True)
        return cls(data, header)

    def simple_spectrum_given_lda(self, new_lda, flux_density=True, 
                                  new_log=True, orig_log=False):
        resampled_fluxes = np.empty((self.n_orders, len(new_lda))) * np.nan
        resampled_errors = np.empty((self.n_orders, len(new_lda))) * np.nan
        for i in range(self.n_orders):
            order = i + self.first_order
            old_lda = self.wavelength(order) 
            
            flux = self.flux(order)
            error = self.noise(order)
            new_fluxes, new_errors = spec_resample(new_lda, old_lda, flux,
                                                   spec_errors=error,
                                                   flux_density=True,
                                                   new_log=new_log,
                                                   orig_log=orig_log)
            resampled_fluxes[i,:] = new_fluxes.copy()
            resampled_errors[i,:] = new_errors.copy()
        # Doing a weighted mean with nan handling.
        weights = 1./resampled_errors**2
        
        weighted_fluxes = (np.nansum(resampled_fluxes * weights, axis=0) /
                           np.nansum(weights, axis=0))
        std_error = 1./np.sqrt(np.nansum(weights, axis=0))

        output = SimpleSpectrum(lda=new_lda, flux=weighted_fluxes,
                                e_flux=std_error)
        return output


def flux_calibrate_multispec(multispec_target, multispec_reference_observed,
                             singlespec_reference_calibrated):
    reference_flux = interp1d(singlespec_reference_calibrated.lda,
                              singlespec_reference_calibrated.flux,
                              kind='linear', bounds_error=False,
                              fill_value='extrapolate', assume_sorted=True)
    for i in range(multispec_target.n_orders):
        order = i + multispec_target.first_order
        print("Flux calibrating order : {}".format(order))
        ref_lda = multispec_target.wavelength(order)

        # New changes using spectres package for spec resample by BN
        obs_flux = spectres(ref_lda, 
                            multispec_reference_observed.wavelength(order), 
                            multispec_reference_observed.flux(order))
        
        calibration_factor = reference_flux(ref_lda) / obs_flux

        calibrated_flux = multispec_target.flux(order) * calibration_factor
        calibrated_error = multispec_target.noise(order) * calibration_factor

        multispec_target.flux(order)[:] = calibrated_flux
        multispec_target.noise(order)[:] = calibrated_error

def optimal_log_lda_from(lda_grid):
    """
    Given a grid of lda arrays, what is the optimal grid in log spacing.

    Here, the optimal will be such that the biggest bin in log spacing is as
    big as the smallest in linear spacing as to not lose resolution. 
    
    In the following steps, the resampled spectrum will be heavy on correlated
    noise because of this choice.
    """
    best_log_step = np.diff(np.log10(lda_grid), axis=1).min()
    log_lda_0 = np.log10(lda_grid.min())
    log_lda_f = np.log10(lda_grid.max())
    log_lda = np.arange(log_lda_0, log_lda_f+best_log_step, best_log_step)
    optimal_lda = 10**log_lda
    return optimal_lda


# This is just to explore the factors (by BN)


def flux_calibrate_multispec_factor(multispec_target, multispec_reference_observed,
                             singlespec_reference_calibrated):
    reference_flux = interp1d(singlespec_reference_calibrated.lda,
                              singlespec_reference_calibrated.flux,
                              kind='linear', bounds_error=False,
                              fill_value=np.nan, assume_sorted=True)
    factor_dict = {
        
    }
    for i in range(multispec_target.n_orders):
        order = i + multispec_target.first_order

        print("Flux calibrating order : {}".format(order))
        ref_lda = multispec_target.wavelength(order)
        
        # New changes from spectres
        obs_flux = spectres(ref_lda, 
                            multispec_reference_observed.wavelength(order), 
                            multispec_reference_observed.flux(order))
        
        calibration_factor = reference_flux(ref_lda) / obs_flux
        factor_dict[str(order)] = calibration_factor
        calibrated_flux = multispec_target.flux(order) * calibration_factor
        calibrated_error = multispec_target.noise(order) * calibration_factor
    return factor_dict

# functions to correct for dust extinction the calibrated flux
# according to Calzetti et al. 1997 + 2000 for starburst galaxies
# Edited by BN 03/01/2022


def k_cal_wl(wl, Rv=3.1):
    """
    Calzetti extinction curve with Rv = 3.1. For individual
    values of wavelength.
    
    Params
    ------
        wl : wavelength
        Rv : don't remember the name of this cosntant
        
    Output
    ------
        k : float value of the Calzetti curve at that wavelength
    """

    if wl < 6300:
        k = 2.659 * (-2.156 + (1.509 / wl) - (0.198 / wl**2) + \
                     (0.011 / wl**3)) + Rv
    elif wl >= 6300:
        k = 2.659 * (-1.857 + 1.040 / wl) + Rv
    return k


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

    wl_low = wl[wl < 6300]
    wl_high = wl[wl >= 6300]
    
    k_low = 2.659 * (-2.156 + (1.509 / wl_low) - (0.198 / wl_low**2) + \
                     (0.011 / wl_low**3)) + Rv 
    k_high = 2.659 * (-1.857 + 1.040 / wl_high) + Rv

    k = np.concatenate((k_low, k_high))
    return k


def dcorr_wl(wl, fl, ha, ha_err, hb, hb_err):
    """
    This function corrects flux values for dust extinction
    using H alpha and H beta Balmer lines.
    
    Params
    ------
        wl : wavelength
        fl : flux
        ha : H alpha intensity
        ha_err : H alpha intensity uncertainty
        hb : H beta intensity
        hb_err : H beta intensity uncertainty
        
    Output
    ------
        f_int : flux values of the corrected spectrum
    """
    
    R_int = 2.86 # for n_e = 100 cm^-3, and T_e = 10000 K
    Ha = ufloat(ha, ha_err)
    Hb = ufloat(hb, hb_err)
    R_obs = Ha/Hb
    E_B_V_obs = -2.5 * uum.log(R_obs)
    E_B_V_int = -2.5 * uum.log(R_int)
    E_B_V = E_B_V_obs - E_B_V_int
    f_int = fl * (10**(0.4 * E_B_V * k_cal(wl)))
    return f_int


def dust_correction(multispec_target, ha, ha_err, hb, hb_err):
    """
    This function corrects a spectrum for dust extinction
    using H alpha and H beta Balmer lines. It changes all
    the flux values of the multispec object.
    
    Params
    ------
        multispec_target : MageMultispec object with the spectrum of the
        galaxy.
        ha : H alpha intensity
        ha_err : H alpha intensity uncertainty
        hb : H beta intensity
        hb_err : H beta intensity uncertainty
        
    Output
    ------
        None
    """

    R_int = 2.86 # for n_e = 100 cm^-3, and T_e = 10000 K
    Ha = ufloat(ha, ha_err)
    Hb = ufloat(hb, hb_err)
    R_obs = Ha/Hb
    E_B_V_obs = -2.5 * uum.log(R_obs)
    E_B_V_int = -2.5 * uum.log(R_int)
    E_B_V = E_B_V_obs - E_B_V_int
    
    for i in range(multispec_target.n_orders):
        order = i + multispec_target.first_order
        print("Dust correction order : {}".format(order))
        
        f_obs = unumpy.uarray(multispec_target.flux(order), np.abs(multispec_target.noise(order)))
        f_int = f_obs * (10**(0.4 * E_B_V * k_cal(multispec_target.wavelength(order))))
        
        multispec_target.flux(order)[:] = unumpy.nominal_values(f_int)
        multispec_target.noise(order)[:] = unumpy.std_devs(f_int)

