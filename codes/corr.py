import os

import astropy.constants as const
import functools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyneb as pn

from astropy.io import ascii, fits
from GaussianFitting import fitSpectrum
from lmfit import Parameter
from lmfit.models import GaussianModel, PolynomialModel
import time
proj_DIR = "/Users/javieratoro/Desktop/proyecto 2024-2/"
code_DIR = f"{proj_DIR}codes"
os.chdir(code_DIR)


def log_method_call(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        print(f"Executed {func.__name__} in {end_time - start_time:.4f}s")
        return result
    return wrapper


class SPECTRALDATA:
    '''
    Receives a dictionary with:
     - DIR: The directory of the astronomical spectra.
     - FILES: The names of the files for each science data,
     completes the directory to read the file.
     - redshift: First hand estimation of the source redshift.
     - names: The source names.
     - mass: The mass of the source.
     EX:
     dicc = {
                'DIR': '/Users/name/desktop/observations',
                'FILES': ['reduction1.fits', 'reduction2.fits'],
                'redshift': 0.086,
                'names': ['J0328+0031'],
                'mass': 9.8}
    '''
    def __init__(self, spectra):
        """Read spectral data loading and management."""
        self.name_class = 'SPECTRALDATA'
        self.DIR = spectra.get('DIR')
        self.FILES = spectra.get('FILES')
        self.redshift = spectra.get('redshift')
        self.names = spectra.get('names')
        self.mass = spectra.get('mass')
        self.line_list, self.linelist_dict = self.load_line_list()
        self.line_wave_or = self.line_list['vacuum_wave']
        self.lines_waves = self.line_wave_or * (1 + self.redshift)
        self.line_name = self.line_list['name']

    em_path = f'{proj_DIR}CSV_files/emission_lines.csv'

    def load_line_list(self, path=em_path):
        """Loads the emission line list from a CSV file."""
        line_list = pd.read_csv(path)
        linelist_table = ascii.read(path)
        linelist_dict = {}
        for line in linelist_table:
            linelist_dict[line['name']] = line['vacuum_wave']
        return line_list, linelist_dict


class REDSHIFT:
    '''
    Receives the SPECTRALDATA class and calculates source's redshift according
    to H_alpha and O[III]5007 emission lines. Updating the new value in the
    SPECTRALDATA class.
    '''
    def __init__(self, spectra):
        self.spectra = spectra
        self.spectra.hdus, self.spectra.datas, self.spectra.zs = [], [], []

        self.zs = []
        for i in self.spectra.FILES:
            path = self.spectra.DIR + i
            hdu = fits.open(path)
            self.spectra.hdus.append(hdu)
            mask = (hdu[1].data['wave'] > 3650)
            data = []
            # Read and save the info from the hdu
            data.append(hdu[1].data['wave'][mask])
            data.append(hdu[1].data['flux'][mask])
            data.append(hdu[1].data['sigma'][mask])
            data.append(i.partition("/")[0])  # name of the folder
            self.spectra.datas.append(data)

            self.reset_redshift(data)

        cte = (1 + self.spectra.redshift)
        self.spectra.lines_waves = self.spectra.line_list['vacuum_wave']*cte

    @log_method_call
    def reset_redshift(self, data):
        """
        Function to calculate and set the redshift on the SPECTRALDATA class
        """
        w, flux, _, name = data

        lines = ['O3_5008', 'H_alpha']
        list_wave_rest = self.spectra.line_wave_or
        list_wave_obs = self.spectra.lines_waves

        zs = [
            np.median(
                (w[mask][np.argmax(flux[mask])] / or_wave) - 1
            ) if mask.any() else np.nan  # Avoid errors if mask is empty
            for line in lines
            for or_wave, w_obs in zip(
                list_wave_rest[self.spectra.line_name == line].values,
                list_wave_obs[self.spectra.line_name == line].values
            )
            if (mask := (w > w_obs - 150) & (w < w_obs + 150)).any()
        ]

        if zs:  # Ensure zs is not empty before calculating median
            self.spectra.redshift = np.median(zs)
            print(f'New calculated for {name}, z = {self.spectra.redshift}')
        self.spectra.zs.append(zs)


class MW_DUST_CORR:
    """
    Receives the SPECTRALDATA class, a boolean according if the Hb
    normalized curve is used, and a boolean for plotting the correction;
    and returns the dust-corrected spectra using PyNeb.
    Uses the MW extinction curve from CCM89.

    The spectra MUST be in rest-frame wavelength.
    """

    def __init__(self, spectra, rel_Hb=False, plot=True):
        self.Hb = rel_Hb
        self.plot = plot
        self.spectra = spectra

        # Unpack spectrum data
        self.wl, self.fl, self.err, _ = self.spectra.datas[0]
        self.gal_id = self.spectra.names[0][:5]
        self.MW_dust_corr()

    @log_method_call
    def MW_dust_corr(self):
        """
        Run the Milky Way dust correction and save the corrected data and
        the figures if asked
        """
        # Read extinction data and apply correction
        self.read_extinction()

        print(f'Performing dust correction for {self.gal_id}')

        # Retrieve E_BV value safely
        where = (self.extinction['objname'] == self.gal_id),
        E_BV_table = float(self.extinction.loc[where, 'E_B_V_SandF'].iloc[0])
        self.IRSA_E_BV = E_BV_table

        # Apply extinction correction
        rc = pn.RedCorr(E_BV=E_BV_table, R_V=3.1, law='CCM89')
        corr = rc.getCorrHb(self.wl) if self.Hb else rc.getCorr(self.wl)

        dcorr_fl = self.fl * corr
        dcorr_err = self.err * corr
        msg = 'Using Hb normalized curve.' if self.Hb else 'Using total curve.'
        print(f"Extinction correction done! {msg}")

        # Plot corrected spectra
        if self.plot:
            self.plot_spectra(dcorr_fl, E_BV_table)

        # Save corrected data
        self.save_corrected_data(dcorr_fl, dcorr_err)

    def read_extinction(self):
        """Reads extinction table and applies dust correction."""
        ext_file = f'{proj_DIR}CSV_files/extinction.tbl'
        self.extinction = pd.read_table(ext_file, comment='#', sep=r'\s+')

        # Clean column names
        self.extinction.rename(columns=lambda x: x[1:], inplace=True)
        self.extinction.drop(index=[0, 1], inplace=True)

        # Get galaxy ID and check if it exists in extinction table

        if self.gal_id not in self.extinction['objname'].values:
            raise ValueError(f"Galaxy ID {self.gal_id} not found in table.")

    def plot_spectra(self, dcorr_fl, E_BV_table):
        """Plots the observed vs dust-corrected spectrum."""
        fig, ax = plt.subplots(figsize=(10, 3))

        ax.plot(self.wl, self.fl, color='red', lw=0.5, label='Observed flux')
        ax.plot(self.wl, dcorr_fl, color='blue', alpha=0.8, lw=0.5,
                label='Dust corrected')

        ax.set_xlabel(r'$\lambda$ (Angstrom)', fontsize=15)
        ax.set_ylabel(r'Flux (erg / s / cm$^{2}$)', fontsize=15)
        ax.set_title(f"Object: {self.gal_id}, "
                     f"z = {np.round(self.spectra.redshift, 2)}, "
                     f"E$_{{B - V}}$ = {E_BV_table}")
        ax.legend()

        save_path = f'{proj_DIR}dust/dcorr_{self.gal_id}.pdf'
        fig.savefig(save_path, bbox_inches='tight')
        print(f'Saved figure at: {save_path}')

    def save_corrected_data(self, dcorr_fl, dcorr_err):
        """Saves the corrected data to a CSV file."""
        save_path = f'{proj_DIR}dust/dcorr_{self.gal_id}.csv'
        df = pd.DataFrame({'wave': self.wl, 'flux': dcorr_fl,
                           'sigma': dcorr_err})
        df.to_csv(save_path, index=False)
        print(f'Saved MW dust corrected data to: {save_path}')


class BALMER_ABS:
    """
    Class to estimate and correct the Balmer absoprtion on a galaxy spectra.


    Receives the SPECTRALDATA class and a boolean for plotting the correction;
    and save the balmer-corrected spectra using a build-in model.

    The spectra MUST be in rest-frame wavelength.
    """

    def __init__(self, spectra, showplot=False, verbose=False):
        self.spectra = spectra
        self.gal_id = self.spectra.names[0][:5]
        # Set Balmer lines names
        self.balmer_lines = ['H_alpha', 'H_beta', 'H_gamma', 'H_delta',
                             'H_epsilon', 'H_6', 'H_7', 'H_8', 'H_9', 'H_10',
                             'H_11']

        # Read MW dust corrected data if it exist
        self.read_data()
        self.bal_abs_corr(plot=showplot)

    @log_method_call
    def bal_abs_corr(self, plot=False):
        print(f'Performing Balmer absorption correction for {self.gal_id}')

        # Get sigma from first model
        sigmas = self.first_sigma_est()

        # Get stamps masked and unmasked
        stamps = self.get_stamp(sigmas)

        # Get model for each emission line
        comps = self.model_absorption(stamps)

        # Apply balmer absorption correction
        new_flux = self.correct_data(stamps, comps, plot=plot)

        # Save data
        self.save_corrected_data(new_flux)

        print("Balmer absorption correction done!")

    def read_data(self):
        """
        Reads MW duct corrected data if it exist, otherwise reads the
        uncorrected data.
        """
        print(f'Reading data for {self.gal_id}')
        file_path = f'{proj_DIR}dust/dcorr_{self.gal_id}.csv'
        if os.path.exists(file_path):
            print("Using MW dust corrected data")
            dcorr = pd.read_csv(file_path)
            arrays = dcorr.to_numpy().T  # Transpose to get column-wise array
            self.wave, self.flux, self.sigma = arrays

        else:
            print("Using uncorrected data")
            self.wave, self.flux, self.sigma, _ = self.spectra.datas[0]

    def first_sigma_est(self, plot=False):
        """
        Get the velocity dispersion (sigma) of the narrow and/or broad
        Gaussian components from the spectral fitting.

        Add plot=True to visualize the spectral fitting model
        """

        fit = fitSpectrum(self.wave, self.flux, self.sigma,
                          linelist=self.spectra.linelist_dict,
                          z_init=self.spectra.redshift,
                          weights=1/self.sigma**2,
                          showPlot=plot,
                          broad=True, nfev=1000)

        sigma_narrow = fit.params['sigma_v_narrow']
        sigma_broad = fit.params['sigma_v_broad']

        return [sigma_narrow, sigma_broad]

    def get_stamp(self, sigmas):
        """
        Create stamps for every emission line 25*sigma from center,
        masking the emission lines for future modelling.
        """
        bright_lines = ['O2_3725', 'O2_3727', 'H_alpha', 'H_beta', 'H_gamma',
                        'O3_5008', 'O3_4959', 'N2_6550', 'N2_6585', 'S2_6716',
                        'S2_6730']
        self.masked_flux = self.flux.copy()
        stamps = []

        [sigma_narr, sigma_broad] = sigmas

        def get_sigma(label):
            if label in bright_lines:
                sigma = (center / const.c.to('km/s').value) * sigma_broad.value
            else:
                sigma = (center / const.c.to('km/s').value) * sigma_narr.value
            return sigma

        # Mask every emission line 3 sigma from center
        cte = (1 + self.spectra.redshift)
        for label in self.spectra.linelist_dict:
            center = self.spectra.linelist_dict[label] * cte
            sigma = get_sigma(label)

            print(center, sigma)

            mask_below = (self.wave > center - 3.5*sigma)
            mask_up = (self.wave < center + 3.5*sigma)
            mask_line = mask_below & mask_up

            # Every line to nan
            self.masked_flux[mask_line] = np.nan

        # Create the stamp 25 sigma from center
        for label in self.balmer_lines:
            center = self.spectra.linelist_dict[label] * cte
            sigma = get_sigma(label)
            mask_below = (self.wave > center - 25*sigma)
            mask_up = (self.wave < center + 25*sigma)
            mask_stamp = mask_below & mask_up

            masked_stamp = self.masked_flux[mask_stamp]
            stamp = self.flux[mask_stamp]
            masked_wave = self.wave[mask_stamp]

            stamps.append([masked_wave, stamp, masked_stamp])
        return stamps

    def model_absorption(self, stamps):
        """
        Create the model for the absoption with a negative gaussian
        and a 1-degree polynomial
        """
        comps = []
        for label, stamp_ in zip(self.balmer_lines, stamps):
            masked_wave, _, masked_stamp = stamp_

            # Gaussian model for balmer abs
            gaussian = GaussianModel(prefix=label+'_')

            # Create a polynomial model for the continuum
            polydeg = 1
            polynomial = PolynomialModel(degree=polydeg)

            comp_mult = gaussian + polynomial
            pars_mult = comp_mult.make_params()

            pars_mult.add(name='z', value=self.spectra.redshift,
                          vary=False)

            min_val = 300 if label == 'H_alpha' else 30
            pars_mult.add(name='sigma_v', value=300, min=min_val,
                          max=800)

            # Loop through emission lines to define parameters
            # for narrow and broad
            min_ampl = -15 if label in self.balmer_lines[-3:] else -100
            if label == 'H_8':
                min_ampl = -40
            lam = self.spectra.linelist_dict[label]
            for param in ['center', 'amplitude', 'sigma']:
                narrow_key = f'{label}_{param}'
                if param == 'center':
                    value = lam
                    vary_ = False
                    min_ = None
                    max_ = None
                    expr = f'{lam:6.2f}*(1+z)'
                elif param == 'amplitude':
                    value = -10
                    vary_ = True
                    min_ = min_ampl
                    max_ = 0
                    expr = None
                elif param == 'sigma':
                    vary_ = True
                    min_ = None
                    max_ = None
                    expr = f'(sigma_v/3e5)*{label}_center'
                pars_mult[narrow_key] = Parameter(name=narrow_key,
                                                  value=value,
                                                  vary=vary_, expr=expr,
                                                  min=min_, max=max_)
            for i in range(polydeg+1):
                pars_mult[f'c{i:1.0f}'].set(value=0)

            out_comp_mult = comp_mult.fit(masked_stamp, pars_mult,
                                          x=masked_wave,
                                          nan_policy='omit',
                                          max_nfev=1000)

            comps.append(out_comp_mult)
        return comps

    def correct_data(self, stamps, comps, plot=False):
        """
        Compute the balmer correction and plot the correction if indicated
        """
        new_flux = self.flux.copy()
        wave = self.wave.copy()
        for stamp_, comp, label in zip(stamps, comps, self.balmer_lines):
            mask_wave, stamp, _ = stamp_
            mask = (wave >= np.min(mask_wave)) & (wave <= np.max(mask_wave))
            model_flux = comp.eval_components(x=mask_wave)
            balmer_abs = model_flux[f'{label}_']
            new_flux[mask] -= balmer_abs

            if plot:
                _, axs = plt.subplots(1, 2, figsize=(10, 4))
                axs[0].plot(mask_wave, stamp, lw=1, drawstyle='steps-mid',
                            alpha=0.5)
                axs[0].plot(mask_wave,
                            model_flux[f'{label}_'] + model_flux['polynomial'],
                            lw=1, drawstyle='steps-mid',
                            label='Balmer abs model')
                axs[0].set_title(f'{label} balmer absorption')
                axs[0].set_xlabel(r'Wavelength $\AA$')
                axs[0].set_ylabel(r'Flux (erg / s / cm$^{2}$)')
                axs[0].legend()
                axs[1].plot(wave[mask], stamp, alpha=0.5,
                            lw=1, drawstyle='steps-mid',
                            label='Uncorrected flux')
                axs[1].plot(wave[mask], new_flux[mask], alpha=0.5,
                            lw=1, drawstyle='steps-mid',
                            label='Corrected flux')
                axs[1].plot(wave[mask], model_flux['polynomial'],
                            alpha=0.3, label='Continuum')
                axs[1].set_xlabel(r'Wavelength $\AA$')
                axs[1].set_ylabel(r'Flux (erg / s / cm$^{2}$)')
                plt.legend()
                path = f'{proj_DIR}bal_abs/{self.gal_id}/{label}_balabs.pdf'
                plt.savefig(path,
                            format='pdf')
                plt.show()
        return new_flux

    # def save_corrected_data(self, new_flux):
    #     """Saves the corrected data to a CSV file."""
    #     save_path = f'{proj_DIR}bal_abs/bcorr_{self.gal_id}.csv'
    #     df = pd.DataFrame({'wave': self.wave, 'flux': new_flux,
    #                        'sigma': self.sigma})
    #     df.to_csv(save_path, index=False)
    #     print(f'Saved Balmer Absorption corrected data to: {save_path}')

    def save_corrected_data(self, new_flux):
        """
        Saves the corrected data to a CSV file,
        creating the directory if needed.
        """
        # Define the directory and file path
        save_dir = f'{proj_DIR}bal_abs'
        save_path = f'{save_dir}/bcorr_{self.gal_id}.csv'

        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Create and save DataFrame
        df = pd.DataFrame({'wave': self.wave, 'flux': new_flux,
                           'sigma': self.sigma})
        df.to_csv(save_path, index=False)

        print(f'Saved Balmer Absorption corrected data to: {save_path}')
