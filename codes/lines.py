import os

import functools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GaussianFitting import fitSpectrum, fitSpectrumMC
from lmfit import Parameter
from lmfit.models import GaussianModel, PolynomialModel
from scipy.ndimage import gaussian_filter1d
import time

main_DIR = "/Users/javieratoro/Desktop/thesis/"
proj_DIR = f"{main_DIR}proyecto 2024-2/"
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


class REDUC_LINES:
    def __init__(self, spectra):
        self.spectra = spectra
        self.model1IT = None
        self.model_MC = None
        self.read_data()

    def read_data(self):
        """
        Reads MW dust + Balmer corrected data if it exist, otherwise reads the
        uncorrected data.
        """
        print(f'Reading data for {self.spectra.gal_id}')
        file_path = f'{proj_DIR}bal_abs/bcorr_{self.spectra.gal_id}.csv'
        if os.path.exists(file_path):
            print("Using MW dust + Balmer corrected data")
            dcorr = pd.read_csv(file_path)
            arrays = dcorr.to_numpy().T  # Transpose to get column-wise array
            self.wave, self.flux, self.sigma = arrays
        else:
            print("Using uncorrected data")
            self.wave, self.flux, self.sigma, _ = self.spectra.datas[0]

    def fit_spectra(self, show_Plot=False, broad=True, nfev=None):

        sm_noise = gaussian_filter1d(self.sigma, sigma=25)

        fit = fitSpectrum(self.wave, self.flux, self.sigma,
                          linelist=self.spectra.linelist_dict,
                          z_init=self.spectra.redshift, weights=1/sm_noise**2,
                          showPlot=show_Plot,
                          broad=broad, nfev=nfev)
        self.model1IT = fit
        return fit

    def fit_MC(self, numMC=2, showPlot=False, broad=True, nfev=None):

        if self.model1IT is None:
            print('Calculating the first initial params')
            fit = fitSpectrum(self.wave, self.flux, self.sigma,
                              linelist=self.spectra.linelist_dict,
                              z_init=self.spectra.redshift,
                              weights=1/self.sigma**2,
                              showPlot=showPlot,
                              broad=broad, nfev=nfev)
            self.model1IT = fit
        params = self.model1IT.params

        info_MC_IT = fitSpectrumMC(self.wave, self.flux, self.sigma,
                                   linelist=self.spectra.linelist_dict,
                                   z_init=self.spectra.redshift,
                                   weights=1/self.sigma**2,
                                   numMC=numMC, showPlot=showPlot,
                                   init_params=params, broad=broad, nfev=nfev)
        self.model_MC = info_MC_IT
        df = pd.DataFrame(columns=['ID', 'mass', 'z', 'name', 'flux',
                                   'fluxerr',
                                   'narrow_flux', 'narrow_fluxerr',
                                   'broad_flux', 'broad_fluxerr'])

        self.bright_lines = ['O2_3725', 'O2_3727', 'H_alpha', 'H_beta',
                             'H_gamma', 'O3_5008', 'O3_4959', 'N2_6550',
                             'N2_6585', 'S2_6716', 'S2_6730']

        for label in self.spectra.linelist_dict.keys():
            narrow_ = np.asarray(info_MC_IT[label+"_narrow"])
            row = {'ID': self.spectra.names[0],
                   'mass': self.spectra.mass,
                   'z': np.median(info_MC_IT['z']),
                   'name': label,
                   'flux': np.mean(narrow_),
                   'fluxerr': np.std(narrow_),
                   'narrow_flux': np.mean(info_MC_IT[label+"_narrow"]),
                   'narrow_fluxerr': np.std(info_MC_IT[label+"_narrow"]),
                   'broad_flux': -9999.9,
                   'broad_fluxerr': -9999.9}

            if label in self.bright_lines:
                broad_ = np.asarray(info_MC_IT[label+"_broad"])
                flux = narrow_ + broad_
                row['flux'] = np.mean(flux)
                row['fluxerr'] = np.std(narrow_)
                row['broad_flux'] = np.mean(info_MC_IT[label+"_broad"])
                row['broad_fluxerr'] = np.std(info_MC_IT[label+"_broad"])

            df.loc[len(df)] = row

        df.to_csv(f'{proj_DIR}lines/{self.spectra.gal_id}_model.csv',
                  index=False)
        info_MC_IT.to_csv(f'{proj_DIR}lines/{self.spectra.gal_id}_iter.csv',
                          index=False)

    def fit_auroral(self, wave, flux, showplot=False, return_comps=False):
        stamps, comps = [], []
        params = pd.read_csv(f'{proj_DIR}lines/{self.spectra.gal_id}_iter.csv')

        sigma_broad = np.median(params['sigma_v_broad'])
        sigma_narrow = np.median(params['sigma_v_narrow'])

        self.cte = 1 + self.spectra.redshift
        center_N2 = self.spectra.linelist_dict['N2_6585'] * self.cte
        center_N1 = self.spectra.linelist_dict['N2_6550'] * self.cte
        self.sep = 2.5 * (center_N2 - center_N1)

        auroral_lines = ['N2_5756', 'O1_6363',
                         'O3_4363', 'S3_6312',
                         'O2_7322']

        # dataframe setup
        columns = ['sigma_v_narrow', 'sigma_v_broad']
        for label in auroral_lines:
            columns += [f'{label}_narrow_amplitude',
                        f'{label}_broad_amplitude']
        columns += ['O2_7333_narrow_amplitude', 'O2_7333_broad_amplitude']

        df = pd.DataFrame(columns=columns)
        row = {'sigma_v_narrow': sigma_narrow,
               'sigma_v_broad': sigma_broad}

        # Apply mask + contamination removal
        def extract_line(label, exclude_labels=None):
            """Return wave_line, flux_line around a line, masking
            nearby contamination."""
            center = self.spectra.linelist_dict[label] * self.cte
            mask = (wave > center - self.sep) & (wave < center + self.sep)
            wave_line, flux_line = wave[mask], flux[mask].copy()

            for l_ in self.spectra.linelist_dict.keys():
                if exclude_labels and l_ in exclude_labels:
                    continue
                c_ = self.spectra.linelist_dict[l_] * self.cte
                if center - self.sep < c_ < center + self.sep:
                    mask2 = (wave_line > c_ - 4) & (wave_line < c_ + 4)
                    flux_line[mask2] = np.nan

            med = np.nanmedian(flux_line)
            flux_line[flux_line < med - 5] = np.nan

            return wave_line, flux_line

        # Fitting routine
        def build_fit(wave_line, flux_line, labels, sigma_n, sigma_b):

            na_gss = [GaussianModel(prefix=f"{lab}_narrow_") for lab in labels]
            br_gss = [GaussianModel(prefix=f"{lab}_broad_") for lab in labels]

            if len(labels) == 1:
                sum_of_gaussians = na_gss[0] + br_gss[0]
            else:
                sum_of_gaussians = br_gss[0] + na_gss[0]
                for narrow, broad_value in zip(na_gss[1:], br_gss[1:]):
                    sum_of_gaussians += (broad_value + narrow)

            polydeg = 1
            polynomial = PolynomialModel(degree=polydeg)
            comp_mult = sum_of_gaussians + polynomial

            pars_mult = comp_mult.make_params()
            pars_mult.add(name='z', value=self.spectra.redshift, vary=False)
            pars_mult.add(name='sigma_v_narrow', value=sigma_n, vary=False)
            pars_mult.add(name='sigma_v_broad', value=sigma_b, vary=False)

            for lab in labels:
                lam = self.spectra.linelist_dict[lab]
                # narrow
                pars_mult[f"{lab}_narrow_center"] = Parameter(
                    name=f"{lab}_narrow_center",
                    value=lam, vary=False,
                    expr=f'{lam:6.2f}*(1+z)'
                )
                pars_mult[f"{lab}_narrow_amplitude"] = Parameter(
                    name=f"{lab}_narrow_amplitude",
                    value=1, vary=True, expr=None, min=0.0
                )
                pars_mult[f"{lab}_narrow_sigma"] = Parameter(
                    name=f"{lab}_narrow_sigma",
                    value=lam, vary=False,
                    expr=f'(sigma_v_narrow/3e5)*{lab}_narrow_center'
                )
                # broad
                pars_mult[f"{lab}_broad_center"] = Parameter(
                    name=f"{lab}_broad_center",
                    value=lam, vary=False,
                    expr=f'{lam:6.2f}*(1+z)'
                )
                pars_mult[f"{lab}_broad_amplitude"] = Parameter(
                    name=f"{lab}_broad_amplitude",
                    value=0.3, vary=True, expr=None, min=0.0
                )
                pars_mult[f"{lab}_broad_sigma"] = Parameter(
                    name=f"{lab}_broad_sigma",
                    value=lam, vary=False,
                    expr=f'(sigma_v_broad/3e5)*{lab}_broad_center'
                )

            for i in range(polydeg + 1):
                pars_mult[f'c{i}'].set(value=0)

            return comp_mult.fit(flux_line, pars_mult, x=wave_line,
                                 nan_policy='omit', max_nfev=1000)

        # -------------------------------
        # Main loop
        # -------------------------------
        for label in auroral_lines:
            print(label)
            o_labels = ['O2_7322', 'O2_7333']
            if label in o_labels:
                # handle O II doublet together
                # build region for both (use first as reference for mask)
                wave_line, flux_line = extract_line('O2_7322',
                                                    exclude_labels=o_labels)
                fit = build_fit(wave_line, flux_line, o_labels,
                                sigma_narrow, sigma_broad)
                stamps.append([wave_line, flux_line])
                comps.append(fit)
                for lab in o_labels:
                    name_narrow = f"{lab}_narrow_amplitude"
                    name_broad = f"{lab}_broad_amplitude"
                    row[name_narrow] = float(fit.params[name_narrow].value)
                    row[name_broad] = float(fit.params[name_broad].value)

            else:
                wave_line, flux_line = extract_line(label, [label])
                fit = build_fit(wave_line, flux_line, [label],
                                sigma_narrow, sigma_broad)
                stamps.append([wave_line, flux_line])
                comps.append(fit)
                name_narrow = f"{label}_narrow_amplitude"
                name_broad = f"{label}_broad_amplitude"
                row[name_narrow] = float(fit.params[name_narrow].value)
                row[name_broad] = float(fit.params[name_broad].value)

        df.loc[len(df)] = row
        if showplot is True:
            fig, axs = plt.subplots(3, 2, figsize=(10, 12))
            axs = axs.ravel()

            for idx, (stamp, label, fit) in enumerate(zip(stamps,
                                                          auroral_lines,
                                                          comps)):
                lamb, flux = stamp
                # print(fit.params)
                comp = fit.eval_components(x=lamb)
                best = fit.eval(x=lamb)
                axs[idx].set_title(label)
                axs[idx].plot(lamb, flux, 'red', lw=1, drawstyle='steps-mid',
                              label='Masked spectrum', alpha=0.5)
                axs[idx].plot(lamb, best, 'black', lw=1, alpha=0.5,
                              label='Best fit')
                axs[idx].plot(lamb,
                              comp[f'{label}_narrow_'] + comp['polynomial'],
                              'teal', linestyle='--', lw=1, alpha=0.5,
                              label='Narrow component')
                axs[idx].plot(lamb,
                              comp[f'{label}_broad_'] + comp['polynomial'],
                              'blue', linestyle='--', lw=1, alpha=0.5,
                              label='Broad component')
                if label in o_labels:
                    axs[idx].plot(lamb,
                                  comp['O2_7333_narrow_'] + comp['polynomial'],
                                  'teal', linestyle='--', lw=1, alpha=0.5,
                                  label='Narrow component')
                    axs[idx].plot(lamb,
                                  comp['O2_7333_broad_'] + comp['polynomial'],
                                  'blue', linestyle='--', lw=1, alpha=0.5,
                                  label='Broad component')

                axs[idx].set_xlabel(r'Obs. Wavelength ($\AA$)', size=14)
                axs[idx].set_ylabel(r'Flux ($10^{-17} erg/s/cm^{2}/\AA$)',
                                    size=14)
                axs[idx].set_xlim([np.min(lamb), np.max(lamb)])
                axs[idx].set_ylim([np.min(fit.best_fit) - 5,
                                   np.max(fit.best_fit) + 5])
                axs[idx].legend()

            for ax in axs[len(stamps):]:
                ax.axis("off")

            fig.tight_layout()
            plt.show()
        if return_comps is True:
            return df, [comps, stamps]
        else:
            return df

    def fit_MC_auroral(self, numMC=400):

        columns = ['sigma_v_narrow', 'sigma_v_broad']

        auroral_lines = ['N2_5756', 'O1_6363',
                         'O3_4363', 'S3_6312',
                         'O2_7322', 'O2_7333']

        for label in auroral_lines:
            name1 = str(label) + "_narrow_amplitude"
            name2 = str(label) + "_broad_amplitude"
            columns.append(name1)
            columns.append(name2)

        df = pd.DataFrame(columns=columns)

        for i in range(numMC):
            # Create a data set with random offsets scaled by uncertainties

            yoff = np.random.randn(len(self.flux)) * self.sigma

            # res = self.fit_auroral(wave, flux + yoff, self)
            res = self.fit_auroral(self.wave, self.flux + yoff)

            df = pd.concat([df, res], ignore_index=True)

        return df

    def fit_MC_auroral2(self, return_comps=False, numMC_=100):
        _, flux, _, _ = self.datas[0]

        info = self.fitSpectrumMC_auroral(numMC=numMC_)

        df = pd.DataFrame(columns=['ID', 'mass', 'z', 'name', 'flux',
                                   'fluxerr',
                                   'narrow_flux', 'narrow_fluxerr',
                                   'broad_flux', 'broad_fluxerr'])

        auroral_lines = ['N2_5756', 'O1_6363',
                         'O3_4363', 'S3_6312',
                         'O2_7322', 'O2_7333']

        for label in auroral_lines:  # Corrected loop variable
            narrow_ = np.asarray(info[label+"_narrow_amplitude"])
            broad_ = np.asarray(info[label+"_broad_amplitude"])
            flux = narrow_ + broad_
            row = {'ID': self.names[0],
                   'mass': self.mass,
                   'z': self.redshift,
                   'name': label,
                   'flux': np.median(flux),
                   'fluxerr': np.std(flux),
                   'narrow_flux': np.median(info[label+"_narrow_amplitude"]),
                   'narrow_fluxerr': np.std(info[label+"_narrow_amplitude"]),
                   'broad_flux': np.median(info[label+"_broad_amplitude"]),
                   'broad_fluxerr': np.std(info[label+"_broad_amplitude"])}
            df.loc[len(df)] = row

        narrow_1 = np.asarray(info['O2_7322_narrow_amplitude'])
        broad_1 = np.asarray(info["O2_7322_broad_amplitude"])
        narrow_2 = np.asarray(info["O2_7333_narrow_amplitude"])
        broad_2 = np.asarray(info["O2_7333_broad_amplitude"])
        flux = narrow_1 + broad_1 + narrow_2 + broad_2
        row = {'ID': self.names[0],
               'mass': self.mass,
               'z': self.redshift,
               'name': 'O2_7322_7333',
               'flux': np.median(flux),
               'fluxerr': np.std(flux),
               'narrow_flux': np.median(narrow_1 + narrow_2),
               'narrow_fluxerr': np.std(narrow_1 + narrow_2),
               'broad_flux': np.median(broad_1 + broad_2),
               'broad_fluxerr': np.std(broad_1 + broad_2)}
        df.loc[len(df)] = row

        path = '/Users/javieratoro/Desktop/proyecto 2024-2/lines/'
        df.to_csv(path + f'{self.names[0][:5]}_auroral.csv', index=False)
        info.to_csv(path + f'{self.names[0][:5]}_auroral_model.csv',
                    index=False)
        return df
