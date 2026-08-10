import os

import astropy.constants as const
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
    """
    Class to model the emission line fluxes,
    Receives an instance of the class SPECTRALDATA
    """
    def __init__(self, spectra):
        self.spectra = spectra
        self.spectra.model1IT = None
        self.model_MC = None
        self.flux = None
        self.bright_lines = ['O2_3725', 'O2_3727', 'H_alpha', 'H_beta',
                             'H_gamma', 'O3_5008', 'O3_4959', 'N2_6550',
                             'N2_6585', 'S2_6716', 'S2_6730']

        self.read_data()

    def read_data(self):
        """
        Reads MW dust + Balmer corrected data if it exist, otherwise reads the
        uncorrected data.
        """
        print(f'Reading data for {self.spectra.gal_id}')
        save_path = (f'{proj_DIR}cont_subs/{self.spectra.gal_id}/'
                     f'cont_corr_{self.spectra.gal_id}_new.csv')
        save_path2 = (f'{proj_DIR}dust/{self.spectra.gal_id}/'
                      f'dcorr_{self.spectra.gal_id}.csv')

        if os.path.exists(save_path):
            print('Using MW dust + Balmer abs corrected and ' +
                  'continuum substracted data')
            dcorr = pd.read_csv(save_path)
            arrays = dcorr.to_numpy().T
            self.wave, self.flux, self.sigma = arrays

        elif os.path.exists(save_path2):
            print("Using MW dust corrected  data")
            dcorr = pd.read_csv(save_path2)
            arrays = dcorr.to_numpy().T
            self.wave, self.flux, self.sigma = arrays
        else:
            print("Using uncorrected data")
            self.wave, self.flux, self.sigma, _ = self.spectra.datas[0]

    def fit_spectra(self, show_Plot=False,
                    broad=True, nfev=1000):
        """
        Fits the spectra with a composite model between a polinomial and
        multiple gaussians for each emission line.
        """
        sm_noise = gaussian_filter1d(self.sigma, sigma=25)

        fit = fitSpectrum(self.wave, self.flux, self.sigma,
                          linelist=self.spectra.linelist_dict,
                          z_init=self.spectra.redshift, weights=1/sm_noise**2,
                          showPlot=show_Plot,
                          broad=broad, nfev=nfev)
        return fit

    @log_method_call
    def fit_MC(self, numMC=2, showPlot=False, broad=True,
               nfev=1000):
        """
        Function to create a MC loop on the fit_spectra function.
        Saves the information of each iteration in a pandas DataFrame that is
        returned at the end of the loop
        """

        if self.spectra.model1IT is None:
            print('Calculating the first initial params')
            fit = fitSpectrum(self.wave, self.flux, self.sigma,
                              linelist=self.spectra.linelist_dict,
                              z_init=self.spectra.redshift,
                              weights=1/self.sigma**2,
                              showPlot=showPlot,
                              broad=broad, nfev=nfev)
            self.spectra.model1IT = fit
        params = self.spectra.model1IT.params

        print('Starting MC iteration')

        MC_IT = fitSpectrumMC(self.wave, self.flux, self.sigma,
                              linelist=self.spectra.linelist_dict,
                              z_init=self.spectra.redshift,
                              weights=1/self.sigma**2,
                              numMC=numMC, showPlot=showPlot,
                              init_params=params, broad=broad, nfev=nfev)

        df = pd.DataFrame(columns=['ID', 'mass', 'z', 'name', 'flux',
                                   'fluxerr',
                                   'narrow_flux', 'narrow_fluxerr',
                                   'broad_flux', 'broad_fluxerr'])

        self.bright_lines = ['O2_3725', 'O2_3727', 'H_alpha', 'H_beta',
                             'H_gamma', 'O3_5008', 'O3_4959', 'N2_6550',
                             'N2_6585', 'S2_6716', 'S2_6730']

        for label in self.spectra.linelist_dict.keys():
            narrow_ = np.asarray(MC_IT[label+"_narrow"])
            row = {'ID': self.spectra.names[0],
                   'mass': self.spectra.mass,
                   'z': np.median(MC_IT['z']),
                   'name': label,
                   'flux': np.mean(narrow_),
                   'fluxerr': np.std(narrow_),
                   'narrow_flux': np.mean(MC_IT[label+"_narrow"]),
                   'narrow_fluxerr': np.std(MC_IT[label+"_narrow"]),
                   'broad_flux': -9999.9,
                   'broad_fluxerr': -9999.9}

            if label in self.bright_lines:
                broad_ = np.asarray(MC_IT[label+"_broad"])
                flux = narrow_ + broad_
                row['flux'] = np.mean(flux)
                row['fluxerr'] = np.std(narrow_)
                row['broad_flux'] = np.mean(MC_IT[label+"_broad"])
                row['broad_fluxerr'] = np.std(MC_IT[label+"_broad"])

            df.loc[len(df)] = row

        save_dir = f'{proj_DIR}lines/{self.spectra.gal_id}/'
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(f'{save_dir}{self.spectra.gal_id}_model.csv',
                  index=False)
        MC_IT.to_csv(f'{save_dir}{self.spectra.gal_id}_iter_.csv',
                     index=False)

    def fit_auroral(self, wave, flux, showplot=False, return_comps=False):
        """
        Makes special fitting on auroral lines
        """
        stamps, comps = [], []
        params = pd.read_csv(f'{proj_DIR}lines/{self.spectra.gal_id}/'
                             f'{self.spectra.gal_id}_iter_.csv')

        sigma_broad = np.median(params['sigma_v_broad'])
        sigma_narrow = np.median(params['sigma_v_narrow'])

        self.cte = 1 + self.spectra.redshift
        center_N2 = self.spectra.linelist_dict['N2_6585'] * self.cte
        center_N1 = self.spectra.linelist_dict['N2_6550'] * self.cte
        self.sep = 2.5 * (center_N2 - center_N1)

        auroral_lines = ['N2_5756', 'O1_6300',
                         'O1_6363', 'O3_4363',
                         'S3_6312', 'O2_7320']

        # dataframe setup
        columns = ['sigma_v_narrow', 'sigma_v_broad']
        for label in auroral_lines:
            columns += [f'{label}_narrow_amplitude',
                        f'{label}_broad_amplitude']
        columns += ['O2_7331_narrow_amplitude', 'O2_7331_broad_amplitude']

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
            pars_mult.add(name='z', value=self.spectra.redshift, vary=True)
            pars_mult.add(name='sigma_v_narrow', value=sigma_n, vary=False)
            pars_mult.add(name='sigma_v_broad', value=sigma_b, vary=False)

            for lab in labels:
                lam = self.spectra.linelist_dict[lab]
                # narrow
                pars_mult[f"{lab}_narrow_center"] = Parameter(
                    name=f"{lab}_narrow_center",
                    value=lam, vary=True,
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
                    value=lam, vary=True,
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
            o_labels = ['O2_7320', 'O2_7331']
            if label in o_labels:
                # handle O II doublet together
                # build region for both (use first as reference for mask)
                wave_line, flux_line = extract_line('O2_7320',
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
                                  comp['O2_7331_narrow_'] + comp['polynomial'],
                                  'teal', linestyle='--', lw=1, alpha=0.5,
                                  label='Narrow component')
                    axs[idx].plot(lamb,
                                  comp['O2_7331_broad_'] + comp['polynomial'],
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

    @log_method_call
    def fit_MC_auroral(self, numMC=400, showPlot=False):
        """
        MC fitting loop on auroral emission lines
        """
        columns = ['sigma_v_narrow', 'sigma_v_broad']

        auroral_lines = ['N2_5756', 'O1_6363',
                         'O3_4363', 'S3_6312',
                         'O2_7320', 'O2_7331']

        for label in auroral_lines:
            name1 = str(label) + "_narrow_amplitude"
            name2 = str(label) + "_broad_amplitude"
            columns.append(name1)
            columns.append(name2)

        info = pd.DataFrame(columns=columns)

        for i in range(numMC):
            # Create a data set with random offsets scaled by uncertainties

            yoff = np.random.randn(len(self.flux)) * self.sigma

            df = self.fit_auroral(self.wave, self.flux + yoff,
                                  showplot=showPlot)

            info = pd.concat([info, df], ignore_index=True)

        res = pd.DataFrame(columns=['ID', 'mass', 'z', 'name', 'flux',
                                    'fluxerr',
                                    'narrow_flux', 'narrow_fluxerr',
                                    'broad_flux', 'broad_fluxerr'])

        auroral_lines = ['N2_5756', 'O1_6363',
                         'O3_4363', 'S3_6312',
                         'O2_7320', 'O2_7331']

        for label in auroral_lines:  # Corrected loop variable
            narrow_ = np.asarray(info[label+"_narrow_amplitude"])
            broad_ = np.asarray(info[label+"_broad_amplitude"])
            flux = narrow_ + broad_
            row = {'ID': self.spectra.names[0],
                   'mass': self.spectra.mass,
                   'z': self.spectra.redshift,
                   'name': label,
                   'flux': np.median(flux),
                   'fluxerr': np.std(flux),
                   'narrow_flux': np.median(info[label+"_narrow_amplitude"]),
                   'narrow_fluxerr': np.std(info[label+"_narrow_amplitude"]),
                   'broad_flux': np.median(info[label+"_broad_amplitude"]),
                   'broad_fluxerr': np.std(info[label+"_broad_amplitude"])}
            res.loc[len(res)] = row

        narrow_1 = np.asarray(info['O2_7320_narrow_amplitude'])
        broad_1 = np.asarray(info["O2_7320_broad_amplitude"])
        narrow_2 = np.asarray(info["O2_7331_narrow_amplitude"])
        broad_2 = np.asarray(info["O2_7331_broad_amplitude"])
        flux = narrow_1 + broad_1 + narrow_2 + broad_2
        row = {'ID': self.spectra.names[0],
               'mass': self.spectra.mass,
               'z': self.spectra.redshift,
               'name': 'O2_7320_7331',
               'flux': np.median(flux),
               'fluxerr': np.std(flux),
               'narrow_flux': np.median(narrow_1 + narrow_2),
               'narrow_fluxerr': np.std(narrow_1 + narrow_2),
               'broad_flux': np.median(broad_1 + broad_2),
               'broad_fluxerr': np.std(broad_1 + broad_2)}
        res.loc[len(res)] = row

        res.to_csv(f'{proj_DIR}lines/{self.spectra.gal_id}/'
                   f'{self.spectra.gal_id}_auroral_model.csv',
                   index=False)
        info.to_csv(f'{proj_DIR}lines/{self.spectra.gal_id}/'
                    f'{self.spectra.gal_id}_auroral_iter.csv',
                    index=False)
        return res

    def first_sigma_est(self, plot=False):
        if self.spectra.model1IT is None:
            fit = fitSpectrum(self.wave, self.flux, self.sigma,
                              linelist=self.spectra.linelist_dict,
                              z_init=self.spectra.redshift,
                              weights=1/self.sigma**2,
                              showPlot=plot,
                              broad=True, nfev=1000)
            self.spectra.model1IT = fit
        else:
            fit = self.spectra.model1IT

        sigma_narrow = fit.params['sigma_v_narrow']
        sigma_broad = fit.params['sigma_v_broad']

        return [sigma_narrow, sigma_broad]

    def model(self, wave, flux, labels, plot=False):
        center = self.spectra.linelist_dict[labels[0]]
        cte = 1 + self.spectra.redshift

        stamp = flux.copy()
        o2_doublet = ['O2_3725', 'O2_3727', 'H_14', 'H_13']
        sep = 25 if labels[0] in o2_doublet else 40
        mask = (wave > center*cte - sep) & (wave < center*cte + sep)
        stamp = stamp[mask]
        wave_stamp = wave[mask]
        for label_ in self.spectra.linelist_dict:
            if label_ in labels:
                continue
            c_ = self.spectra.linelist_dict[label_] * cte
            sigmas = self.first_sigma_est(plot=False)

            def get_sigma(label, sigmas):
                bright_lines = ['O2_3725', 'O2_3727', 'H_alpha', 'H_beta',
                                'H_gamma', 'O3_5008', 'O3_4959', 'N2_6550',
                                'N2_6585',
                                'S2_6716', 'S2_6730']
                if label in bright_lines:
                    sigma = (center / const.c.to('km/s').value) * sigmas[1].value
                else:
                    sigma = (center / const.c.to('km/s').value) * sigmas[0].value
                return sigma

            sigma_ = get_sigma(label_, sigmas)
            mask2 = (wave > c_ - 3*sigma_) & (wave < c_ + 3*sigma_)
            stamp[mask2[mask]] = np.nan

        na_gss = [GaussianModel(prefix=f"{lab}_narrow_") for lab in labels]
        br_gss = [GaussianModel(prefix=f"{lab}_broad_") for lab in labels]

        # Create a polynomial model for the continuum
        polydeg = 1
        polynomial = PolynomialModel(degree=polydeg)

        if len(labels) == 1:
            sum_of_gaussians = na_gss[0] + br_gss[0]
        else:
            sum_of_gaussians = br_gss[0] + na_gss[0]
            for narrow, broad_value in zip(na_gss[1:], br_gss[1:]):
                sum_of_gaussians += (broad_value + narrow)

        comp_mult = sum_of_gaussians + polynomial
        pars_mult = comp_mult.make_params()

        pars_mult.add(name='sigma_v_narrow', value=50, min=20, max=70)
        pars_mult.add(name='sigma_v_broad', value=90, min=70, max=450)

        # narrow parameters
        for label in labels:
            pars_mult.add(name=f'z_{label}', value=self.spectra.redshift,
                          vary=True, min=self.spectra.redshift - 1e-4,
                          max=self.spectra.redshift + 1e-4)

            center = self.spectra.linelist_dict[label]
            pars_mult[f'{label}_narrow_center'] = Parameter(
                name=f'{label}_narrow_center',
                value=center,
                vary=False,
                expr=f'{center:6.2f}*(1+z_{label})'
            )
            pars_mult[f'{label}_narrow_amplitude'] = Parameter(
                name=f'{label}_narrow_amplitude',
                value=1,
                min=0.0,
                vary=True
            )
            pars_mult[f'{label}_narrow_sigma'] = Parameter(
                name=f'{label}_narrow_sigma',
                vary=False,
                expr=f'(sigma_v_narrow/3e5)*{label}_narrow_center'
            )

            pars_mult[f'{label}_broad_center'] = Parameter(
                name=f'{label}_broad_center',
                value=center,
                vary=False,
                expr=f'{center:6.2f}*(1+z_{label})'
            )
            pars_mult[f'{label}_broad_amplitude'] = Parameter(
                name=f'{label}_broad_amplitude',
                value=0.3,
                min=0.0,
                vary=True
            )
            pars_mult[f'{label}_broad_sigma'] = Parameter(
                name=f'{label}_broad_sigma',
                vary=False,
                expr=f'(sigma_v_broad/3e5)*{label}_broad_center'
            )

        for i in range(polydeg+1):
            pars_mult[f'c{i:1.0f}'].set(value=0)

        out_comp_mult = comp_mult.fit(stamp, pars_mult, x=wave_stamp,
                                      nan_policy='omit', max_nfev=1000)

        if plot is True:
            model = out_comp_mult.eval_components(x=wave_stamp)
            best = out_comp_mult.eval(x=wave_stamp)
            plt.figure()
            plt.step(wave_stamp, stamp, color='red', label='Observed spectra')
            plt.step(wave_stamp, best, color='black', label='Line model')
            for label in labels:
                plt.step(wave_stamp,
                         model[f'{label}_narrow_'] + model['polynomial'],
                         color='green',
                         alpha=0.3)
                plt.step(wave_stamp,
                         model[f'{label}_broad_'] + model['polynomial'],
                         color='blue', alpha=0.3)
                if label == labels[0]:
                    plt.step(wave_stamp,
                             model[f'{label}_narrow_'] + model['polynomial'],
                             color='green', label='Narrow component',
                             alpha=0.3)
                    plt.step(wave_stamp,
                             model[f'{label}_broad_'] + model['polynomial'],
                             color='blue', label='Broad component', alpha=0.3)
            plt.xlabel(r"Wavelength [$\AA$]")
            plt.ylabel(r'Flux ($10^{-17} erg/s/cm^{2}/\AA$)')
            plt.title(labels)
            plt.legend()
            plt.show()
        return [wave_stamp, stamp], out_comp_mult

    def fit_lines(self, wave, flux, showplot=False, ind_plot=False):
        doublets = np.array([['H_alpha', 'N2_6585'], ['He1_3898', 'H_8'],
                             ['H_epsilon', 'Ne3_3967']])
        o2_doublet = ['O2_3725', 'O2_3727', 'H_14', 'H_13']
        first = doublets[:, 0]
        second = doublets[:, 1]
        stamps, comps = [], []
        lines = self.spectra.line_name.values

        for doublet in doublets:
            lines = lines[lines != doublet[0]]
            lines = lines[lines != doublet[1]]

        for o2_line in o2_doublet:
            lines = lines[lines != o2_line]

        lines_plot = np.append(lines, first)

        for label in lines:
            stamp, fit = self.model(wave, flux, [label], plot=ind_plot)
            stamps.append(stamp)
            comps.append(fit)

        for doublet in doublets:
            stamp, fit = self.model(wave, flux,  doublet, plot=ind_plot)
            stamps.append(stamp)
            comps.append(fit)

        stamp_o, fit_o = self.model(wave, flux, o2_doublet,
                                    plot=ind_plot)
        comp_o = fit_o.eval_components(x=stamp_o[0])

        if showplot is True:
            fig2, axs = plt.subplots(2, 1, figsize=(10, 8))
            axs[0].step(wave, flux, label='Full Spectrum', color='gray',
                        alpha=0.5)
            axs[0].set_title('Maked Spectral Stamps')
            axs[1].set_title('Fitted Spectral Lines')
            axs[1].step(wave, flux, label='Full Spectrum', color='gray',
                        alpha=0.5)
            axs[1].set_xlabel(r'Obs. Wavelength ($\AA$)', size=14)
            axs[1].set_ylabel(r'Flux ($10^{-17} erg/s/cm^{2}/\AA$)', size=14)

            lamb, flux = stamp_o
            best = fit_o.eval(x=lamb)
            axs[0].step(lamb, flux, lw=1, drawstyle='steps-mid')
            axs[1].step(lamb, best, 'black', lw=1)

            def plot(label, fit, comp):
                axs[1].step(lamb,
                            comp[f'{label}_narrow_'] + comp['polynomial'],
                            'teal', linestyle='--', lw=1, alpha=0.3)
                axs[1].step(lamb, comp[f'{label}_broad_'] + comp['polynomial'],
                            'blue', linestyle='--', lw=1, alpha=0.3)
                cte_obs = (1 + fit.params[f'z_{label}'].value)
                obs_lam = self.spectra.linelist_dict[label] * cte_obs
                axs[1].axvline(obs_lam, linestyle='--', linewidth=1,
                               color='grey', lw=0.5)
                axs[1].text(obs_lam, 0.99, '\n'+label, rotation=90,
                            ha='center', va='top',
                            color='k', size=8,
                            transform=axs[1].get_xaxis_transform())
                axs[0].axvline(obs_lam, linestyle='--', linewidth=1,
                               color='grey', lw=0.5)
                axs[0].text(obs_lam, 0.99, '\n'+label, rotation=90,
                            ha='center', va='top',
                            color='k', size=8,
                            transform=axs[0].get_xaxis_transform())

            for label_o2 in o2_doublet:
                plot(label_o2, fit_o, comp_o)

            fits = []
            for stamp, label, fit in zip(stamps,
                                         lines_plot,
                                         comps):
                lamb, flux = stamp
                comp = fit.eval_components(x=lamb)
                fits.append(comp)
                best = fit.eval(x=lamb)
                axs[0].step(lamb, flux, lw=1, drawstyle='steps-mid')
                axs[1].step(lamb, best, 'black', lw=1)
                plot(label, fit, comp)

                if label in doublets:
                    idx = first.tolist().index(label)
                    label2 = second[idx]
                    plot(label2, fit, comp)

            plt.show()
        return comps, fit_o, lines_plot

    def fit_lines_MC(self, numMC=100, showplot=False, ind_plot=False):

        doublets = np.array([['H_alpha', 'N2_6585'], ['He1_3898', 'H_8'],
                             ['H_epsilon', 'Ne3_3967']])
        o2_doublet = ['O2_3725', 'O2_3727', 'H_14', 'H_13']

        columns = ['iter', 'n_eval', 'success', 'message', 'ier', 'z',
                   'sigma_v_narrow', 'sigma_v_broad']

        for label in self.spectra.line_name:
            columns.append(str(label) + '_narrow')
            columns.append(str(label) + '_broad')

        df = pd.DataFrame(columns=columns)

        def extract_amplitudes(fit, label, row):
            narrow_key = f"{label}_narrow_amplitude"
            broad_key = f"{label}_broad_amplitude"
            row[f"{label}_narrow"] = float(fit.params[narrow_key].value)
            row[f"{label}_broad"] = float(fit.params[broad_key].value)
            return row

        for i in range(numMC):
            yoff = self.flux + np.random.randn(len(self.flux)) * self.sigma
            fits, fit_o, lines = self.fit_lines(self.wave, yoff,
                                                showplot=showplot,
                                                ind_plot=ind_plot)

            row = {'iter': i + 1,
                   'z': float(self.spectra.redshift),
                   'sigma_v_narrow': float(fits[0].params['sigma_v_narrow'].value),
                   'sigma_v_broad': float(fits[0].params['sigma_v_broad'].value),
                   'n_eval': fits[0].nfev,
                   'success': fits[0].success,
                   'message': fits[0].lmdif_message,
                   'ier': fits[0].ier
                   }
            for fit, label in zip(fits, lines):
                if label in doublets:
                    idx = doublets[:, 0].tolist().index(label)
                    for label_ in doublets[idx]:
                        extract_amplitudes(fit, label_, row)
                else:
                    extract_amplitudes(fit, label, row)

            for label_o2 in o2_doublet:
                extract_amplitudes(fit_o, label_o2, row)

        df.loc[len(df)] = row
        save_dir = f'{proj_DIR}lines/{self.spectra.gal_id}/'
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(f'{save_dir}{self.spectra.gal_id}_iter_single_line.csv',
                  index=False)
        return df

    def save_data(self, info):
        df = pd.DataFrame(columns=['ID', 'mass', 'z', 'name', 'flux',
                                   'fluxerr_16', 'fluxerr_84',
                                   'narrow_flux', 'narrow_fluxerr_16',
                                   'narrow_fluxerr_84', 'broad_flux',
                                   'broad_fluxerr_16', 'broad_fluxerr_84'])

        def get_error(flux):
            median_flux = np.median(flux)
            flux_16, flux_84 = np.percentile(flux, [16, 84])
            flux_err_low = median_flux - flux_16
            flux_err_high = flux_84 - median_flux
            return median_flux, flux_err_low, flux_err_high

        for label in self.spectra.linelist_dict.keys():
            narrow_ = np.asarray(info[label+"_narrow"])
            broad_ = np.asarray(info[label+"_broad"])
            flux = narrow_ + broad_
            median_flux, flux_low, flux_high = get_error(flux)
            median_narr, narr_low, narr_high = get_error(narrow_)
            median_brd, brd_low, brd_high = get_error(broad_)
            row = {'ID': self.spectra.names[0],
                   'mass': self.spectra.mass,
                   'z': np.median(info['z']),
                   'name': label,
                   'flux': median_flux,
                   'fluxerr_16': flux_low,
                   'fluxerr_84': flux_high,
                   'narrow_flux': median_narr,
                   'narrow_fluxerr_16': narr_low,
                   'narrow_fluxerr_84': narr_high,
                   'broad_flux': median_brd,
                   'broad_fluxerr_16': brd_low,
                   'broad_fluxerr_84': brd_high
                   }
            df.loc[len(df)] = row

        save_dir = f'{proj_DIR}lines/{self.spectra.gal_id}/'
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(f'{save_dir}{self.spectra.gal_id}_model_single_line.csv',
                  index=False)
        return df

    def run_single_model(self, numMC=100):
        df = self.fit_lines_MC(self, numMC=numMC, showplot=False)
        _ = self.save_data(self, df)
        print('Done')
