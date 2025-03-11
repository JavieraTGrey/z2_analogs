import matplotlib.pyplot as plt
from astropy.io import fits, ascii
import datetime
import numpy as np
import pandas as pd
from magE.plotutils import *
from GaussianFitting import fitSpectrum, fitSpectrumMC
from scipy.ndimage import gaussian_filter1d
from lmfit.models import GaussianModel, PolynomialModel
from lmfit import Parameter
import pyneb as pn


class REDUC_LINES:
    def __init__(self, SPECTRA):
        self.SPECTRA = SPECTRA
        self.DIR = SPECTRA['DIR']
        self.FILES = SPECTRA['FILES']
        self.redshift = SPECTRA['redshift']
        self.names = SPECTRA['names']
        self.mass = SPECTRA['mass']
        self.MC_unc = None
        self.init = None
        self.init_corr = None
        path = '/Users/javieratoro/Desktop/proyecto 2024-2/CSV_files/emission_lines.csv'
        self.line_list = pd.read_csv(path)
        self.line_name = self.line_list['name']
        self.line_wave_or = self.line_list['vacuum_wave']
        self.lines_waves = self.line_list['vacuum_wave']*(1 + self.redshift)
        self.new_spectra = None

        linelist_table = ascii.read(path)
        self.linelist_dict = {}
        for line in linelist_table:
            self.linelist_dict[line['name']] = line['vacuum_wave']

        print(f'Using directory : {self.DIR}')
        self.hdus = []
        self.datas = []

        self.zs = []
        for i in self.FILES:
            path = SPECTRA['DIR'] + i
            print(path)
            hdu = fits.open(path)
            self.hdus.append(hdu)
            mask = (hdu[1].data['wave'] > 3650)
            data = []
            data.append(hdu[1].data['wave'][mask])
            data.append(hdu[1].data['flux'][mask])
            data.append(hdu[1].data['sigma'][mask])
            data.append(i.partition("/")[0])
            self.datas.append(data)

            self.reset_redshift(data)

        self.redshift = np.median(self.zs)
        self.lines_waves = self.line_list['vacuum_wave']*(1 + self.redshift)

    def reset_redshift(self, data):
        wavelength, flux, _, name = data
        lines = ['O3_5008', 'H_alpha']
        zs = []
        for line in lines:
            or_wave = self.line_wave_or[self.line_name == line].values
            wave_O3 = self.lines_waves[self.line_name == line].values
            mask = ((wavelength > wave_O3 - 150) &
                    (wavelength < wave_O3 + 150))
            wave = wavelength[mask]
            fluxes = flux[mask]
            peak_index = np.argmax(fluxes)
            peak_wavelength = wave[peak_index]
            z = (peak_wavelength / or_wave) - 1
            zs.append(z)
        print(f'New calculated redshift for {name},  z = {np.median(zs)}')
        self.redshift = np.median(zs)
        self.zs.append(zs)

    def MW_dust_corr(self, E_BV, rel_Hb=False, plot=True):
        '''
        Receives a spectrum (wl, fl, err) and a reddening constant E_BV and returns
        the dust corrected spectra using PyNeb. If rel_Hb=True, the correction is
        done using the extinction curve normalized by the extinction in Hb. The MW
        extinction curve from CCM89 is used (see PyNeb list of extinctions curve).

        The spectra MUST be in rest-frame wavelength.
        '''
        wl, fl, err, _ = self.datas[0]
        rc = pn.RedCorr(E_BV=E_BV, R_V=3.1, law='CCM89')

        if rel_Hb:
            dcorr_fl = fl * rc.getCorrHb(wl)
            dcorr_err = err * rc.getCorrHb(wl)
            print('Extinction correction done! Used Hb normalized curve.')
        else:
            dcorr_fl = fl * rc.getCorr(wl)
            dcorr_err = err * rc.getCorr(wl)
            print('Extinction correction done! Used total curve.')

        if plot is True:
            rf_wl = wl/(1 + self.redshift)

            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(10, 3)

            ax.plot(rf_wl, fl, color=ccd[3],
                    lw=0.5, label='Observed flux')
            ax.plot(rf_wl, dcorr_fl, color=cz[0],
                    alpha=0.8, lw=0.5, label='Dust corrected')

            ax.set_ylabel(r'Flux (erg / s / cm$^{2}$)', fontsize=15)
            ax.set_xlabel(r'$\lambda$ (Angstrom)', fontsize=15)
            title = f"Object: {self.names[0][:5]}"
            ax.set_title(title +
                         f', z = {str(np.round(self.redshift, 2))}' +
                         ' E$_{B - V}$ = ' + str(E_BV))

            DIR = '/Users/javieratoro/Desktop/proyecto 2024-2/'

            fig.savefig(f'{DIR}dust/dcorr_{self.names[0][:5]}.pdf',
                        bbox_inches='tight')
            print(f'Saved figure at: {DIR}dust/dcorr_{self.names[0][:5]}.pdf')
            print('Saving MW dust corrected data')
            df = pd.DataFrame({'wave': wl, 'flux': dcorr_fl,
                               'sigma': dcorr_err})
            df.to_csv(f'{DIR}dust/dcorr_{self.names[0][:5]}.csv')

    def balmer_absorption(self, showplot=False, default=True):
        balmer_lines = ['H_alpha', 'H_beta', 'H_gamma', 'H_delta',
                        'H_epsilon', 'H_6', 'H_7', 'H_8', 'H_9', 'H_10', 'H_11']

        wave1, flux1, _, _ = self.datas[0]
        self.new_spectra = flux1.copy()
        stamps = []
        comps = []
        new_fluxes = []

        def get_center_and_sep(label):
            """Return the center and separation value based on the label."""
            center = self.linelist_dict[label] * (1 + self.redshift)

            if label == 'H_alpha':
                center_N2 = self.linelist_dict['N2_6585'] * (1 + self.redshift)
                center_N1 = self.linelist_dict['N2_6550'] * (1 + self.redshift)
                sep = 2.5 * (center_N2 - center_N1)
            elif label in ['H_8', 'H_9', 'H_10', 'H_11']:
                center_H2 = self.linelist_dict['H_8'] * (1 + self.redshift)
                center_H1 = self.linelist_dict['H_9'] * (1 + self.redshift)
                sep = 1.25 * (center_H2 - center_H1)
            else:
                center_2 = self.linelist_dict['H_epsilon'] * (1 + self.redshift)
                center_1 = self.linelist_dict['Ne3_3868'] * (1 + self.redshift)
                sep = 1.35 * (center_2 - center_1)
            return center, sep

        for label in balmer_lines:
            center, sep = get_center_and_sep(label)

            # Select a stamp
            lamb = wave1[(wave1 < center+sep) & (wave1 > center-sep)]
            flux2 = flux1[(wave1 < center+sep) & (wave1 > center-sep)]

            # Mask the emission line
            # if 'alpha' in label or 'beta' in label:
            #     mask = (wave1 < center+seps[2][0]) & (wave1 > center-seps[2][1])
            # elif not any(char.isdigit() for char in label) and not ('alpha' in label or 'beta' in label):
            #     mask = (wave1 < center+seps[1][0]) & (wave1 > center-seps[1][1])
            # else:
            #     mask = (wave1 < center+seps[0][0]) & (wave1 > center-seps[0][1])

            mask = (wave1 < center+sep/8) & (wave1 > center-sep/8)

            new_flux = flux1.copy()

            new_flux[flux1 > np.median(new_flux[(wave1 < center+sep) & (wave1 > center-sep)]) + 2] = np.nan

            new_flux[mask] = np.nan

            new_flux = new_flux[(wave1 < center+sep) & (wave1 > center-sep)]

            narrow_gaussians = GaussianModel(prefix=label+'_narrow_')

            polydeg = 1
            polynomial = PolynomialModel(degree=polydeg)

            comp_mult = narrow_gaussians + polynomial
            pars_mult = comp_mult.make_params()

            pars_mult.add(name='z', value=self.redshift, vary=False)
            small_h = ['H_7', 'H_8', 'H_9', 'H_10', 'H_11']
            min_v = 200 if label in small_h else 500
            # max_v = 700 if label in small_h else 2000
            # guess = 400  if label in small_h else 900
            if default is True:
                max_v = 700 if label in small_h else 1000
                guess = 400 if label in small_h else 700
            else:
                max_v = 1000
                guess = 500 if label in small_h else 800

            pars_mult.add(name='sigma_v_narrow', value=guess, min=min_v, max=max_v)

            lam = self.linelist_dict[label]
            for param in ['center', 'amplitude', 'sigma']:
                narrow_key = f'{label}_narrow_{param}'
                if param == 'center':
                    value = lam
                    vary_ = False
                    min_ = None
                    max_ = None
                    expr = f'{lam:6.2f}*(1+z)'
                elif param == 'amplitude':
                    value = -20
                    vary_ = True
                    min_ = -100
                    max_ = 0
                    expr = None
                elif param == 'sigma':
                    vary_ = True
                    min_ = None
                    max_ = None
                    expr = f'(sigma_v_narrow/3e5)*{label}_narrow_center'
                pars_mult[narrow_key] = Parameter(name=narrow_key, value=value,
                                                  vary=vary_, expr=expr,
                                                  min=min_, max=max_)

            for i in range(polydeg+1):
                pars_mult[f'c{i:1.0f}'].set(value=0)

            out_comp_mult = comp_mult.fit(new_flux, pars_mult, x=lamb,
                                          nan_policy='omit', max_nfev=1000)

            comp = out_comp_mult.eval_components(x=lamb)
            balmer_abs = comp[f'{label}_narrow_'] - np.median(comp[f'{label}_narrow_'])

            self.new_spectra[(wave1 < center+sep) & (wave1 > center-sep)] -= balmer_abs
            stamps.append(self.new_spectra[(wave1 < center+sep) & (wave1 > center-sep)])
            comps.append(comp)
            new_fluxes.append(new_flux)
            # fig = plt.figure()
            # plt.title(label)
            # plt.plot(lamb, flux2)
            # plt.plot(lamb, comp[f'{label}_narrow_'] + comp['polynomial'])
            # plt.plot(lamb, balmer_abs)
            # plt.plot(lamb, flux2 - comp[f'{label}_narrow_'])
            # plt.show()

            header = self.hdus[0][1].header
            header['EXTNAME_2'] = 'ST_AB_CORR'
            header['DATE'] = str(datetime.date.today())

            hdu = fits.PrimaryHDU(self.new_spectra, header)

            hdul = fits.HDUList([hdu])
            self.hdu_corrected = hdul
            DIR = '/Users/javieratoro/Desktop/proyecto 2024-2/balmer_absorption/'
            hdul.writeto(DIR + f'{self.names[0][:5]}_STELLAR_abscorr.fits',
                         overwrite=True)

        if showplot is True:
            fig, axs = plt.subplots(4, 3, figsize=(13, 12))
            axs = axs.ravel()

            for idx, (stamp, label, comp, new_flux) in enumerate(zip(stamps,
                                                                     balmer_lines,
                                                                     comps,
                                                                     new_fluxes)):
                center, sep = get_center_and_sep(label)

                lamb = wave1[(wave1 < center+sep) & (wave1 > center-sep)]
                flux2 = flux1[(wave1 < center + sep) & (wave1 > center - sep)]

                axs[idx].set_title(label)
                axs[idx].plot(lamb, flux2, 'red', lw=1, drawstyle='steps-mid',
                              label='Observed spectrum', alpha=0.5)
                axs[idx].plot(lamb, new_flux, 'blue', lw=1, drawstyle='steps-mid',
                              alpha=0.5)
                axs[idx].plot(lamb, comp['polynomial'], 'green', lw=1,
                              alpha=0.5, label='model')
                axs[idx].plot(lamb, comp[f'{label}_narrow_'], 'teal', lw=1,
                              alpha=0.5)
                axs[idx].plot(lamb, flux2 - (comp[f'{label}_narrow_']),
                              'magenta', lw=1,
                              alpha=0.5, label='Corrected')
                axs[idx].vlines(center, 0, np.max(flux2), 'grey', '--',
                                alpha=0.3)
                axs[idx].set_xlabel(r'Obs. Wavelength ($\AA$)', size=14)
                axs[idx].set_ylabel(r'Flux ($10^{-17} erg/s/cm^{2}/\AA$)',
                                    size=14)
                axs[idx].set_xlim([np.min(lamb), np.max(lamb)])
                axs[idx].set_ylim([np.min(comp[f'{label}_narrow_']) - 5, 30])
                axs[idx].legend()

            for ax in axs[len(stamps):]:
                ax.axis("off")

            fig.tight_layout()
            plt.show()

    def plot_new_spectra(self):
        if self.new_spectra is None:
            raise ValueError("There is not balmer absorption calibrated data. Run balmer_absorption again")

        fig = plt.figure(constrained_layout=True, figsize=(13, 6))
        axs = fig.subplot_mosaic([['Left', 'TopRight',
                                   'TopRight2'],
                                  ['Left', 'Bottom', 'Bottom']],
                                 gridspec_kw={'width_ratios': [2, 1, 1]})

        wave, flux, _, _ = self.datas[0]

        axs['Left'].step(wave, flux, alpha=0.5, label='Obs spectra',
                         lw=1, drawstyle='steps-mid')
        axs['Left'].step(wave, self.new_spectra, alpha=0.5,
                         label='Corrected spectra',
                         lw=1, drawstyle='steps-mid')

        axs['Left'].set_xlabel(r'Wavelength ($\AA$)')
        axs['Left'].set_ylabel(r'Flux ($10^{-17} erg/s/cm^{2}/\AA$)')
        axs['Left'].set_title(self.names[0] + f',z = {np.round(self.redshift, 4)}')
        axs['Left'].legend(markerscale=5, frameon=False,
                           bbox_to_anchor=(0.95, 0.75),
                           loc='upper right', borderaxespad=0)
        axs['Left'].set_xlim(3600, 9550)

        axs['Left'].minorticks_on()
        axs['Left'].tick_params(which='major', length=10, width=1.2,
                                direction='in')
        axs['Left'].tick_params(which='minor', length=5, width=1.2,
                                direction='in')
        axs['Left'].xaxis.set_ticks_position('both')
        axs['Left'].yaxis.set_ticks_position('both')
        if self.lines_waves is not None and self.line_name is not None:
            for wave1, label in zip(self.lines_waves, self.line_name):
                axs['Left'].axvline(x=wave1, color='gray',
                                    linestyle='--', alpha=0.2)
                axs['Left'].text(wave1, 0.95, '\n'+label,
                                 rotation=90, ha='center', va='top',
                                 color='k', size=8,
                                 transform=axs['Left'].get_xaxis_transform())

        axs['TopRight'].set_title(r'$H_{\beta}$')
        xlim = self.lines_waves[self.line_name == 'H_beta'].values
        axs['TopRight'].step(wave, flux, alpha=0.5, label='Obs spectra',
                             lw=1, drawstyle='steps-mid')
        axs['TopRight'].step(wave, self.new_spectra, alpha=0.5,
                             label='Corrected spectra',
                             lw=1, drawstyle='steps-mid')

        wave_O3 = self.lines_waves[self.line_name == 'H_beta'].values
        resta_O3 = np.abs(wave - wave_O3)
        flux_O3 = flux[np.argmin(resta_O3)]
        axs['TopRight'].set_xlim(xlim-50, xlim+50)
        axs['TopRight'].set_ylim(0, flux_O3/5)

        axs['TopRight2'].set_title(r'$H_{\gamma}$')
        axs['TopRight2'].step(wave, flux, alpha=0.5, label='Obs spectra',
                              lw=1, drawstyle='steps-mid')
        axs['TopRight2'].step(wave, self.new_spectra, alpha=0.5,
                              label='Corrected spectra',
                              lw=1, drawstyle='steps-mid')

        xlim1 = self.lines_waves[self.line_name == 'H_gamma'].values
        wave_hb = self.lines_waves[self.line_name == 'H_gamma'].values
        resta_hb = np.abs(wave - wave_hb)
        flux_hb = flux[np.argmin(resta_hb)]
        axs['TopRight2'].set_xlim(xlim1-30, xlim1+30)
        axs['TopRight2'].set_ylim(-5, flux_hb + 10)

        axs['Bottom'].step(wave, flux, alpha=0.5, label='Obs spectra',
                           lw=1, drawstyle='steps-mid')
        axs['Bottom'].step(wave, self.new_spectra, alpha=0.5,
                           label='Corrected spectra',
                           lw=1, drawstyle='steps-mid')
        xlim_out = self.lines_waves[self.line_name == 'Ne3_3970'].values
        xlim_in = self.lines_waves[self.line_name == 'H_11'].values
        wave_hb = self.lines_waves[self.line_name == 'Ne3_3970'].values
        resta_hb = np.abs(wave - wave_hb)
        flux_hb = flux[np.argmin(resta_hb)]

        axs['Bottom'].set_xlim(xlim_in - 50, xlim_out + 50)
        axs['Bottom'].set_ylim(0, flux_hb)
        axs['Bottom'].minorticks_on()
        axs['Bottom'].tick_params(which='major', length=10,
                                  width=1.2,
                                  direction='in')
        axs['Bottom'].tick_params(which='minor', length=5,
                                  width=1.2,
                                  direction='in')
        axs['Bottom'].xaxis.set_ticks_position('both')
        axs['Bottom'].yaxis.set_ticks_position('both')

        mask = ((self.lines_waves.values < wave_hb) | (self.lines_waves.values == wave_hb))
        for wavelength2, label1 in zip(self.lines_waves[mask], self.line_name[mask]):
            axs['Bottom'].axvline(x=wavelength2, color='gray',
                                  linestyle='--', alpha=0.2)
            axs['Bottom'].text(wavelength2, 0.95, '\n'+label1, rotation=90,
                               ha='center', va='top', color='k', size=8,
                               transform=axs['Bottom'].get_xaxis_transform())
        path = '/Users/javieratoro/Desktop/proyecto 2024-2/images/calibrated_spectra_'
        plt.savefig(path + self.DIR[-6:-1] + '.pdf',
                    format='pdf')
        plt.show()

    def plot_spectra(self):
        for name1 in self.names:
            print(f'Plotting for {name1}')
            fig = plt.figure(constrained_layout=True, figsize=(13, 6))
            axs = fig.subplot_mosaic([['Left', 'TopRight',
                                       'TopRight2', 'TopRight3'],
                                      ['Left', 'Bottom', 'Bottom', 'Bottom3']],
                                     gridspec_kw={'width_ratios': [2, 1, 1, 1]
                                                  })

            for data in self.datas:
                wave, flux, _, name = data
                axs['Left'].step(wave, flux, alpha=0.5, label=name)

            axs['Left'].set_xlabel(r'Wavelength ($\AA$)')
            axs['Left'].set_ylabel(r'Flux ($10^{-17} erg/s/cm^{2}/\AA$)')
            axs['Left'].set_title(name1 + f',z = {np.round(self.redshift, 4)}')
            axs['Left'].legend(markerscale=5, frameon=False,
                               bbox_to_anchor=(0.95, 0.8),
                               loc='upper right', borderaxespad=0)
            axs['Left'].set_xlim(3600, 9550)

            axs['Left'].minorticks_on()
            axs['Left'].tick_params(which='major', length=10, width=1.2,
                                    direction='in')
            axs['Left'].tick_params(which='minor', length=5, width=1.2,
                                    direction='in')
            axs['Left'].xaxis.set_ticks_position('both')
            axs['Left'].yaxis.set_ticks_position('both')
            if self.lines_waves is not None and self.line_name is not None:
                for wave1, label in zip(self.lines_waves, self.line_name):
                    axs['Left'].axvline(x=wave1, color='gray',
                                        linestyle='--', alpha=0.2)
                    axs['Left'].text(wave1, 0.95, '\n'+label,
                                     rotation=90, ha='center', va='top',
                                     color='k', size=8,
                                     transform=axs['Left'].get_xaxis_transform())

            axs['TopRight'].set_title(r'$H_{\gamma}$ + [OIII]$\lambda$4363')
            xlim = self.lines_waves[self.line_name == 'O3_4363'].values
            for i in range(0, len(self.datas)):
                wave, flux, _, name = self.datas[i]
                axs['TopRight'].step(wave, flux, alpha=0.5, label=name)
                if i == len(self.datas)-1:
                    wave_O3 = self.lines_waves[self.line_name == 'O3_4363'].values
                    resta_O3 = np.abs(wave - wave_O3)
                    flux_O3 = flux[np.argmin(resta_O3)]
                    axs['TopRight'].set_xlim(xlim-50, xlim+50)
                    axs['TopRight'].set_ylim(0.1, flux_O3+10)

            axs['TopRight2'].set_title(r'[O II]$\lambda\lambda$3725,3727')
            for i in range(0, len(self.datas)):
                wave, flux, _, name = self.datas[i]
                axs['TopRight2'].step(wave, flux, alpha=0.5, label=name)
                if i == len(self.datas)-1:
                    xlim1 = 0.5*(self.lines_waves[self.line_name == 'O2_3725'].values +
                                 self.lines_waves[self.line_name == 'O2_3727'].values)
                    wave_hb = self.lines_waves[self.line_name == 'O2_3725'].values
                    resta_hb = np.abs(wave - wave_hb)
                    flux_hb = flux[np.argmin(resta_hb)]
                    axs['TopRight2'].set_xlim(xlim1-30, xlim1+30)
                    axs['TopRight2'].set_ylim(-5, flux_hb + 100)

            for i in range(0, len(self.datas)):
                wave, flux, _,  name = self.datas[i]
                axs['Bottom'].step(wave, flux, alpha=0.5, label=name)
                if i == len(self.datas) - 1:
                    xlim_out = self.lines_waves[self.line_name == 'Ar3_7753'].values
                    xlim_in = self.lines_waves[self.line_name == 'He1_7067'].values
                    wave_hb = self.lines_waves[self.line_name == 'He1_7067'].values
                    resta_hb = np.abs(wave - wave_hb)
                    flux_hb = flux[np.argmin(resta_hb)]

                    axs['Bottom'].set_xlim(xlim_in - 100, xlim_out + 100)
                    axs['Bottom'].set_ylim(flux_hb - 15, flux_hb + 15)
                    axs['Bottom'].minorticks_on()
                    axs['Bottom'].tick_params(which='major', length=10,
                                              width=1.2,
                                              direction='in')
                    axs['Bottom'].tick_params(which='minor', length=5,
                                              width=1.2,
                                              direction='in')
                    axs['Bottom'].xaxis.set_ticks_position('both')
                    axs['Bottom'].yaxis.set_ticks_position('both')

            mask = ((self.lines_waves.values > wave_hb) | (self.lines_waves.values == wave_hb))
            for wavelength2, label1 in zip(self.lines_waves[mask], self.line_name[mask]):
                axs['Bottom'].axvline(x=wavelength2, color='gray',
                                      linestyle='--', alpha=0.2)
                axs['Bottom'].text(wavelength2, 0.95, '\n'+label1, rotation=90,
                                   ha='center', va='top', color='k', size=8,
                                   transform=axs['Bottom'].get_xaxis_transform())

            axs['TopRight3'].set_title(r'$H_{\beta}$ + [OIII]$\lambda$$\lambda$4959,5007')
            xlim = 0.5*(self.lines_waves[self.line_name == 'O3_5008'].values +
                        self.lines_waves[self.line_name == 'H_beta'].values)
            for i in range(0, len(self.datas)):
                wave, flux, _, name = self.datas[i]
                axs['TopRight3'].step(wave, flux, alpha=0.5, label=name)
                if i == len(self.datas)-1:
                    wave_O3 = self.lines_waves[self.line_name == 'H_beta'].values
                    resta_O3 = np.abs(wave - wave_O3)
                    flux_O3 = flux[np.argmin(resta_O3)]
                    axs['TopRight3'].set_xlim(xlim-150, xlim+150)
                    axs['TopRight3'].set_ylim(0.1, flux_O3+100)

            axs['Bottom3'].set_title(r'$H_{\alpha}$ + [NII]$\lambda$$\lambda$6550,6585')
            for i in range(0, len(self.datas)):
                wave, flux, _, name = self.datas[i]
                axs['Bottom3'].step(wave, flux, alpha=0.5, label=name)
                if i == len(self.datas)-1:
                    xlim1 = (self.lines_waves[self.line_name == 'H_alpha'].values)
                    resta_hb = np.abs(wave - wave_hb)
                    flux_hb = flux[np.argmin(resta_hb)]
                    axs['Bottom3'].set_xlim(xlim1-100, xlim1+100)
                    axs['Bottom3'].set_ylim(-5, flux_hb + 100)
            path = '/Users/javieratoro/Desktop/proyecto 2024-2/images/spectra_'
            plt.savefig(path + name1[:5] + '.pdf', format='pdf')
            plt.show()

    def fit_spectra(self, mode, show_Plot=False, broad=True, nfev=None):
        if mode == 'corrected':
            wave, _, noise, _ = self.datas[0]
            hdu = self.hdu_corrected
            flux = hdu[0].data
        else:
            wave, flux, noise, _ = self.datas[0]

        sm_noise = gaussian_filter1d(noise, sigma=25)

        fit = fitSpectrum(wave, flux, noise, linelist=self.linelist_dict,
                          z_init=self.redshift, weights=1/sm_noise**2,
                          showPlot=show_Plot,
                          broad=broad, nfev=nfev)
        self.model = fit
        return fit

    def plot_fit(self, mode, show_fit=False, broad=True, nfev=None):
        if mode == 'corrected':
            wave, _, noise, _ = self.datas[0]
            hdu = self.hdu_corrected
            flux = hdu[0].data
            if self.init_corr is None:
                print('Modeling spectra')
                fit = self.fit_spectra(mode, show_Plot=show_fit,
                                       broad=broad, nfev=None)
                self.init_corr = fit
        else:
            wave, flux, noise, _ = self.datas[0]
            if self.init is None:
                print('Modeling spectra')
                fit = self.fit_spectra(mode, show_Plot=show_fit,
                                       broad=broad, nfev=None)
                self.init = fit

        fit = self.init
        fig = plt.figure(constrained_layout=True, figsize=(13, 6))
        axs = fig.subplot_mosaic([['Left', 'TopRight',
                                   'TopRight2', 'TopRight3'],
                                  ['Left', 'Bottom', 'Bottom', 'Bottom3']],
                                 gridspec_kw={'width_ratios': [2, 1, 1, 1]
                                              })
        axs['Left'].step(wave, flux, alpha=0.5, label='Obs spectrum')

        axs['Left'].fill_between(wave, -noise, noise,
                                 label='Error spectrum',
                                 zorder=-2, color='0.6')
        axs['Left'].axhline(0, color='k', ls='-', zorder=-1)
        axs['Left'].step(wave, fit.best_fit, '-', lw=0.5,
                         label='Best-model fit')

        axs['Left'].set_xlabel(r'Wavelength ($\AA$)')
        axs['Left'].set_ylabel(r'Flux ($10^{-17} erg/s/cm^{2}/\AA$)')
        axs['Left'].set_title(self.names[0] + f', z = {np.round(self.redshift, 4)}')
        axs['Left'].legend(markerscale=5, frameon=False,
                           bbox_to_anchor=(0.3, 0.97),
                           loc='upper right', borderaxespad=0)
        axs['Left'].set_xlim(3600, 9550)
        axs['Left'].minorticks_on()
        axs['Left'].tick_params(which='major', length=10, width=1.2,
                                direction='in')
        axs['Left'].tick_params(which='minor', length=5, width=1.2,
                                direction='in')
        axs['Left'].xaxis.set_ticks_position('both')
        axs['Left'].yaxis.set_ticks_position('both')
        if self.lines_waves is not None and self.line_name is not None:
            for wave_1, label in zip(self.lines_waves, self.line_name):
                axs['Left'].axvline(x=wave_1, color='gray',
                                    linestyle='--', alpha=0.2)
                axs['Left'].text(wave_1, 0.85, '\n'+label,
                                 rotation=90, ha='center', va='top',
                                 color='k', size=8,
                                 transform=axs['Left'].get_xaxis_transform())

        axs['TopRight'].set_title(r'$H_{\gamma}$ + [OIII]$\lambda$4363')
        xlim = self.lines_waves[self.line_name == 'O3_4363'].values
        axs['TopRight'].step(wave, flux, alpha=0.5)
        axs['TopRight'].fill_between(wave, -noise, noise,
                                     label='Error spectrum',
                                     zorder=-2, color='0.6')

        if fit is not None:
            axs['TopRight'].step(wave, fit.best_fit, '-', lw=0.5)

        wave_O3 = self.lines_waves[self.line_name == 'O3_4363'].values
        resta_O3 = np.abs(wave - wave_O3)
        flux_O3 = flux[np.argmin(resta_O3)]
        axs['TopRight'].set_xlim(xlim-50, xlim+50)
        axs['TopRight'].set_ylim(0.1, flux_O3+10)

        axs['TopRight2'].set_title(r'[O II]$\lambda\lambda$3725,3727')
        axs['TopRight2'].step(wave, flux, alpha=0.5)
        axs['TopRight2'].fill_between(wave, -noise, noise,
                                      label='Error spectrum',
                                      zorder=-2, color='0.6')

        if fit is not None:
            axs['TopRight2'].step(wave, fit.best_fit, '-', lw=0.5)

        xlim1 = 0.5*(self.lines_waves[self.line_name == 'O2_3725'].values +
                     self.lines_waves[self.line_name == 'O2_3727'].values)
        wave_hb = self.lines_waves[self.line_name == 'O2_3725'].values
        resta_hb = np.abs(wave - wave_hb)
        flux_hb = flux[np.argmin(resta_hb)]
        axs['TopRight2'].set_xlim(xlim1 - 30, xlim1 + 30)
        axs['TopRight2'].set_ylim(-5, flux_hb + 100)

        axs['Bottom'].step(wave, flux, alpha=0.5)
        axs['Bottom'].fill_between(wave, -noise, noise,
                                   label='Error spectrum',
                                   zorder=-2, color='0.6')
        axs['Bottom'].step(wave, fit.best_fit, '-', lw=0.5)
        xlim_out = self.lines_waves[self.line_name == 'Ar3_7753'].values
        xlim_in = self.lines_waves[self.line_name == 'He1_7067'].values
        wave_hb = self.lines_waves[self.line_name == 'He1_7067'].values
        resta_hb = np.abs(wave - wave_hb)
        flux_hb = flux[np.argmin(resta_hb)]
        axs['Bottom'].set_xlim(xlim_in - 100, xlim_out + 100)
        axs['Bottom'].set_ylim(flux_hb - 15, flux_hb + 15)
        axs['Bottom'].minorticks_on()
        axs['Bottom'].tick_params(which='major', length=10, width=1.2,
                                  direction='in')
        axs['Bottom'].tick_params(which='minor', length=5, width=1.2,
                                  direction='in')
        axs['Bottom'].xaxis.set_ticks_position('both')
        axs['Bottom'].yaxis.set_ticks_position('both')

        val = self.lines_waves.values
        mask = ((val > wave_hb) | (val == wave_hb))
        for wavelength2, label1 in zip(self.lines_waves[mask],
                                       self.line_name[mask]):
            axs['Bottom'].axvline(x=wavelength2, color='gray',
                                  linestyle='--', alpha=0.2)
            axs['Bottom'].text(wavelength2, 0.95, '\n'+label1, rotation=90,
                               ha='center', va='top', color='k', size=8,
                               transform=axs['Bottom'].get_xaxis_transform())

        axs['TopRight3'].set_title(r'$H_{\beta}$ + [OIII]$\lambda$$\lambda$4959,5007')
        xlim = 0.5*(self.lines_waves[self.line_name == 'O3_5008'].values +
                    self.lines_waves[self.line_name == 'H_beta'].values)
        axs['TopRight3'].step(wave, flux, alpha=0.5)
        axs['TopRight3'].fill_between(wave, -noise, noise,
                                      label='Error spectrum',
                                      zorder=-2, color='0.6')
        axs['TopRight3'].step(wave, fit.best_fit, '-', lw=0.5)
        wave_O3 = self.lines_waves[self.line_name == 'H_beta'].values
        resta_O3 = np.abs(wave - wave_O3)
        flux_O3 = flux[np.argmin(resta_O3)]
        axs['TopRight3'].set_xlim(xlim-150, xlim+150)
        axs['TopRight3'].set_ylim(-5, flux_O3+100)

        axs['Bottom3'].set_title(r'$H_{\alpha}$ + [NII]$\lambda$$\lambda$6550,6585')
        axs['Bottom3'].step(wave, flux, alpha=0.5)
        axs['Bottom3'].fill_between(wave, -noise, noise,
                                    label='Error spectrum',
                                    zorder=-2, color='0.6')

        axs['Bottom3'].step(wave, fit.best_fit, '-', lw=0.5)
        xlim1 = (self.lines_waves[self.line_name == 'H_alpha'].values)
        resta_hb = np.abs(wave - wave_hb)
        flux_hb = flux[np.argmin(resta_hb)]
        axs['Bottom3'].set_xlim(xlim1-100, xlim1+100)
        axs['Bottom3'].set_ylim(-5, flux_hb + 100)
        path = '/Users/javieratoro/Desktop/proyecto 2024-2/images/spectra_'
        plt.savefig(path + self.DIR[-6:-1] + '.pdf',
                    format='pdf')
        plt.show()

    def fit_MC(self, mode, numMC=2, showPlot=False, broad=True, nfev=None):
        if mode == 'corrected':
            wave, _, noise, _ = self.datas[0]
            hdu = self.hdu_corrected
            flux = hdu[0].data
        else:
            wave, flux, noise, _ = self.datas[0]

        if self.init is None:
            print('Calculating the first init params')
            fit = fitSpectrum(wave, flux, noise, linelist=self.linelist_dict,
                              z_init=self.redshift, weights=1/noise**2,
                              showPlot=showPlot,
                              broad=broad, nfev=nfev)
            self.init = fit
        params = self.init.params

        info = fitSpectrumMC(wave, flux, noise,
                             linelist=self.linelist_dict,
                             z_init=self.redshift, weights=1/noise**2,
                             numMC=numMC, showPlot=showPlot,
                             init_params=params, broad=broad, nfev=nfev)
        self.MC_unc = info
        df = pd.DataFrame(columns=['ID', 'mass', 'z', 'name', 'flux',
                                   'fluxerr',
                                   'narrow_flux', 'narrow_fluxerr',
                                   'broad_flux', 'broad_fluxerr'])

        self.bright_lines = ['O2_3725', 'O2_3727', 'H_alpha', 'H_beta',
                             'H_gamma', 'O3_5008', 'O3_4959', 'N2_6550',
                             'N2_6585', 'S2_6716', 'S2_6730']

        for label in self.linelist_dict.keys():  # Corrected loop variable
            narrow_ = np.asarray(info[label+"_narrow"])
            row = {'ID': self.names[0],
                   'mass': self.mass,
                   'z': np.median(info['z']),
                   'name': label,
                   'flux': np.mean(narrow_),
                   'fluxerr': np.std(narrow_),
                   'narrow_flux': np.mean(info[label+"_narrow"]),
                   'narrow_fluxerr': np.std(info[label+"_narrow"]),
                   'broad_flux': -9999.9,
                   'broad_fluxerr': -9999.9}

            if label in self.bright_lines:
                broad_ = np.asarray(info[label+"_broad"])
                flux = narrow_ + broad_
                row['flux'] = np.mean(flux)
                row['fluxerr'] = np.std(narrow_)
                row['broad_flux'] = np.mean(info[label+"_broad"])
                row['broad_fluxerr'] = np.std(info[label+"_broad"])

            df.loc[len(df)] = row
        if mode == 'corrected':
            df.to_csv(f'/Users/javieratoro/Desktop/proyecto 2024-2/balmer_absorption/{self.names[0][:5]}.csv', index=False)
            info.to_csv(f'/Users/javieratoro/Desktop/proyecto 2024-2/balmer_absorption/{self.names[0][:5]}_model.csv', index=False)
        else:
            df.to_csv(f'/Users/javieratoro/Desktop/proyecto 2024-2/lines/{self.names[0][:5]}.csv', index=False)
            info.to_csv(f'/Users/javieratoro/Desktop/proyecto 2024-2/lines/{self.names[0][:5]}_model.csv', index=False)

    def fit_auroral(self, wave1, flux1, showplot=False, return_comps=False):
        stamps, comps = [], []
        path = '/Users/javieratoro/Desktop/proyecto 2024-2/lines/'
        params = pd.read_csv(path + f'{self.names[0][:5]}_model.csv')
        sigma_broad = np.median(params['sigma_v_broad'])
        sigma_narrow = np.median(params['sigma_v_narrow'])

        def get_center_and_sep(label):
            """Return the center and separation value based on the label."""
            center = self.linelist_dict[label] * (1 + self.redshift)
            center_N2 = self.linelist_dict['N2_6585'] * (1 + self.redshift)
            center_N1 = self.linelist_dict['N2_6550'] * (1 + self.redshift)
            sep = 2.5 * (center_N2 - center_N1)
            return center, sep

        auroral_lines = ['N2_5756', 'O1_6363',
                         'O3_4363', 'S3_6312',
                         'O2_7322']

        columns = ['sigma_v_narrow', 'sigma_v_broad']

        for label in auroral_lines:
            columns.append(str(label) + '_narrow_amplitude')
            columns.append(str(label) + '_broad_amplitude')

        columns.append('O2_7333_narrow_amplitude')
        columns.append('O2_7333_broad_amplitude')

        df = pd.DataFrame(columns=columns)
        row = {'sigma_v_narrow': sigma_narrow,
               'sigma_v_broad': sigma_broad}

        for label in auroral_lines:
            o2 = ['O2_7322', 'O2_7333']
            if label in o2:
                narrow_gaussians = []
                broad_gaussians = []
                for label in o2:
                    center, sep = get_center_and_sep(label)

                    mask = ((wave1 < center+sep) & (wave1 > center-sep))

                    lamb = wave1[mask]
                    flux2 = flux1[mask]

                    for label_ in self.linelist_dict.keys():
                        center_, _ = get_center_and_sep(label_)
                        if center_ < center+sep or center_ > center-sep:
                            if label_ not in o2:
                                sep_ = 4
                                mask2 = ((lamb < center_+sep_) & (lamb > center_-sep_))
                                flux2[mask2] = np.nan

                                med = np.median(flux2)
                                flux2[flux2 < med-1] = np.nan

                    # Narrow component
                    narrow_gaussian = GaussianModel(prefix=label+'_narrow_')
                    narrow_gaussians.append(narrow_gaussian)

                    broad_gaussian = GaussianModel(prefix=label+'_broad_')
                    broad_gaussians.append(broad_gaussian)

                polydeg = 1
                polynomial = PolynomialModel(degree=polydeg)

                sum_of_gaussians = broad_gaussians[0] + narrow_gaussians[0]
                for narrow, broad_value in zip(narrow_gaussians[1:],
                                               broad_gaussians[1:]):
                    sum_of_gaussians += (broad_value + narrow)

                comp_mult = sum_of_gaussians + polynomial
                pars_mult = comp_mult.make_params()

                pars_mult.add(name='z', value=self.redshift, vary=False)

                pars_mult.add(name='sigma_v_narrow', value=sigma_narrow,
                              vary=False)

                pars_mult.add(name='sigma_v_broad', value=sigma_broad,
                              vary=False)

                for label in o2:
                    lam = self.linelist_dict[label]
                    for param in ['center', 'amplitude', 'sigma']:
                        narrow_key = f'{label}_narrow_{param}'
                        if param == 'center':
                            value = lam
                            vary = False
                            expr = f'{lam:6.2f}*(1+z)'
                        elif param == 'amplitude':
                            value = 1
                            vary = True
                            expr = None
                        elif param == 'sigma':
                            vary = False
                            expr = f'(sigma_v_narrow/3e5)*{label}_narrow_center'
                        pars_mult[narrow_key] = Parameter(name=narrow_key,
                                                          value=value,
                                                          vary=vary, expr=expr,
                                                          min=0.0)

                        broad_key = f'{label}_broad_{param}'
                        if param == 'center':
                            value = lam
                            vary = False
                            expr = f'{lam:6.2f}*(1+z)'
                        elif param == 'amplitude':
                            value = 0.3
                            vary = True
                            expr = None
                        elif param == 'sigma':
                            vary = False
                            expr = f'(sigma_v_broad/3e5)*{label}_broad_center'
                        pars_mult[broad_key] = Parameter(name=broad_key,
                                                         value=value, min=0.0,
                                                         vary=vary, expr=expr)

                for i in range(polydeg+1):
                    pars_mult[f'c{i:1.0f}'].set(value=0)

                fit = comp_mult.fit(flux2, pars_mult, x=lamb,
                                    nan_policy='omit', max_nfev=1000)

                stamps.append([lamb, flux2])
                comps.append(fit)
                for label in o2:
                    name1 = str(label) + "_narrow_amplitude"
                    name2 = str(label) + "_broad_amplitude"
                    row[name1] = float(fit.params[name1].value)
                    row[name2] = float(fit.params[name2].value)

            else:
                center, sep = get_center_and_sep(label)

                mask = ((wave1 < center+sep) & (wave1 > center-sep))

                lamb = wave1[mask]
                flux2 = flux1[mask]

                for label_ in self.linelist_dict.keys():
                    center_, _ = get_center_and_sep(label_)
                    if center_ < center+sep or center_ > center-sep:
                        if label_ != label:
                            sep_ = 4
                            mask2 = ((lamb < center_+sep_) &
                                     (lamb > center_-sep_))
                            flux2[mask2] = np.nan

                            med = np.median(flux2)
                            flux2[flux2 < med-1] = np.nan

                narrow_gaussians = []
                broad_gaussians = []

                # Narrow component
                narrow_gaussian = GaussianModel(prefix=label+'_narrow_')
                narrow_gaussians.append(narrow_gaussian)

                broad_gaussian = GaussianModel(prefix=label+'_broad_')
                broad_gaussians.append(broad_gaussian)

                polydeg = 1
                polynomial = PolynomialModel(degree=polydeg)

                sum_of_gaussians = broad_gaussians[0] + narrow_gaussians[0]
                for narrow, broad_value in zip(narrow_gaussians[1:],
                                               broad_gaussians[1:]):
                    sum_of_gaussians += (broad_value + narrow)

                comp_mult = sum_of_gaussians + polynomial
                pars_mult = comp_mult.make_params()

                pars_mult.add(name='z', value=self.redshift, vary=False)

                pars_mult.add(name='sigma_v_narrow', value=sigma_narrow,
                              vary=False)

                pars_mult.add(name='sigma_v_broad', value=sigma_broad,
                              vary=False)

                lam = self.linelist_dict[label]
                for param in ['center', 'amplitude', 'sigma']:
                    narrow_key = f'{label}_narrow_{param}'
                    if param == 'center':
                        value = lam
                        vary = False
                        expr = f'{lam:6.2f}*(1+z)'
                    elif param == 'amplitude':
                        value = 1
                        vary = True
                        expr = None
                    elif param == 'sigma':
                        vary = False
                        expr = f'(sigma_v_narrow/3e5)*{label}_narrow_center'
                    pars_mult[narrow_key] = Parameter(name=narrow_key,
                                                      value=value,
                                                      vary=vary, expr=expr,
                                                      min=0.0)

                    broad_key = f'{label}_broad_{param}'
                    if param == 'center':
                        value = lam
                        vary = False
                        expr = f'{lam:6.2f}*(1+z)'
                    elif param == 'amplitude':
                        value = 0.3
                        vary = True
                        expr = None
                    elif param == 'sigma':
                        vary = False
                        expr = f'(sigma_v_broad/3e5)*{label}_broad_center'
                    pars_mult[broad_key] = Parameter(name=broad_key,
                                                     value=value, min=0.0,
                                                     vary=vary, expr=expr)

                for i in range(polydeg+1):
                    pars_mult[f'c{i:1.0f}'].set(value=0)

                fit = comp_mult.fit(flux2, pars_mult, x=lamb,
                                    nan_policy='omit', max_nfev=1000)

                stamps.append([lamb, flux2])
                comps.append(fit)

                name1 = str(label) + "_narrow_amplitude"
                name2 = str(label) + "_broad_amplitude"
                row[name1] = float(fit.params[name1].value)
                row[name2] = float(fit.params[name2].value)

        df.loc[len(df)] = row

        if showplot is True:
            fig, axs = plt.subplots(3, 2, figsize=(10, 12))
            axs = axs.ravel()

            for idx, (stamp, label, fit) in enumerate(zip(stamps,
                                                          auroral_lines,
                                                          comps)):
                center, sep = get_center_and_sep(label)

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
                if label in o2:
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

    def fitSpectrumMC_auroral(self, numMC=400):

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

        wave, flux, noise, _ = self.datas[0]

        for i in range(numMC):
            # Create a data set with random offsets scaled by uncertainties

            yoff = np.random.randn(len(flux)) * noise

            # res = self.fit_auroral(wave, flux + yoff, self)
            res = self.fit_auroral(wave, flux + yoff)

            df = pd.concat([df, res], ignore_index=True)

        return df

    def fit_MC_auroral(self, return_comps=False, numMC_=100):
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





# import matplotlib.pyplot as plt
# from astropy.io import fits, ascii
# import datetime
# import numpy as np
# import pandas as pd
# from GaussianFitting import fitSpectrum, fitSpectrumMC
# from scipy.ndimage import gaussian_filter1d
# from lmfit.models import GaussianModel, PolynomialModel
# from lmfit import Parameter


# class REDUC_LINES:
#     def __init__(self, SPECTRA):
#         self.SPECTRA = SPECTRA
#         self.DIR = SPECTRA['DIR']
#         self.FILES = SPECTRA['FILES']
#         self.redshift = SPECTRA['redshift']
#         self.names = SPECTRA['names']
#         self.mass = SPECTRA['mass']
#         self.MC_unc = None
#         self.init = None
#         self.init_corr = None
#         path = '/Users/javieratoro/Desktop/proyecto 2024-2/CSV_files/emission_lines.csv'
#         self.line_list = pd.read_csv(path)
#         self.line_name = self.line_list['name']
#         self.line_wave_or = self.line_list['vacuum_wave']
#         self.lines_waves = self.line_list['vacuum_wave']*(1 + self.redshift)
#         self.new_spectra = None

#         linelist_table = ascii.read(path)
#         self.linelist_dict = {}
#         for line in linelist_table:
#             self.linelist_dict[line['name']] = line['vacuum_wave']

#         print(f'Using directory : {self.DIR}')
#         self.hdus = []
#         self.datas = []

#         self.zs = []
#         for i in self.FILES:
#             path = SPECTRA['DIR'] + i
#             print(path)
#             hdu = fits.open(path)
#             self.hdus.append(hdu)
#             mask = (hdu[1].data['wave'] > 3650)
#             data = []
#             data.append(hdu[1].data['wave'][mask])
#             data.append(hdu[1].data['flux'][mask])
#             data.append(hdu[1].data['sigma'][mask])
#             data.append(i.partition("/")[0])
#             self.datas.append(data)

#             self.reset_redshift(data)

#         self.redshift = np.median(self.zs)
#         self.lines_waves = self.line_list['vacuum_wave']*(1 + self.redshift)

#     def reset_redshift(self, data):
#         wavelength, flux, _, name = data
#         lines = ['O3_5008', 'H_alpha']
#         zs = []
#         for line in lines:
#             or_wave = self.line_wave_or[self.line_name == line].values
#             wave_O3 = self.lines_waves[self.line_name == line].values
#             mask = ((wavelength > wave_O3 - 150) &
#                     (wavelength < wave_O3 + 150))
#             wave = wavelength[mask]
#             fluxes = flux[mask]
#             peak_index = np.argmax(fluxes)
#             peak_wavelength = wave[peak_index]
#             z = (peak_wavelength / or_wave) - 1
#             zs.append(z)
#         print(f'New calculated redshift for {name},  z = {np.median(zs)}')
#         self.redshift = np.median(zs)
#         self.zs.append(zs)

#     def balmer_absorption(self, showplot=False):
#         balmer_lines = ['H_alpha', 'H_beta', 'H_gamma', 'H_delta',
#                         'H_epsilon', 'H_6', 'H_7', 'H_8', 'H_9', 'H_10', 'H_11']

#         seps = [(2, 2), (5, 5), (10, 10)]
#         wave1, flux1, _, _ = self.datas[0]
#         self.new_spectra = flux1.copy()
#         stamps = []
#         comps = []
#         new_fluxes = []

#         def get_center_and_sep(label):
#             """Return the center and separation value based on the label."""
#             center = self.linelist_dict[label] * (1 + self.redshift)

#             if label == 'H_alpha':
#                 center_N2 = self.linelist_dict['N2_6585'] * (1 + self.redshift)
#                 center_N1 = self.linelist_dict['N2_6550'] * (1 + self.redshift)
#                 sep = 2.5 * (center_N2 - center_N1)
#             elif label in ['H_8', 'H_9', 'H_10', 'H_11']:
#                 center_H2 = self.linelist_dict['H_8'] * (1 + self.redshift)
#                 center_H1 = self.linelist_dict['H_9'] * (1 + self.redshift)
#                 sep = 1.2 * (center_H2 - center_H1)
#             else:
#                 center_2 = self.linelist_dict['H_epsilon'] * (1 + self.redshift)
#                 center_1 = self.linelist_dict['Ne3_3868'] * (1 + self.redshift)
#                 sep = 1.35 * (center_2 - center_1)
#             return center, sep

#         for label in balmer_lines:
#             center, sep = get_center_and_sep(label)

#             lamb = wave1[(wave1 < center+sep) & (wave1 > center-sep)]
#             flux2 = flux1[(wave1 < center+sep) & (wave1 > center-sep)]

#             if 'alpha' in label or 'beta' in label:
#                 mask = (wave1 < center+seps[2][0]) & (wave1 > center-seps[2][1])
#             elif not any(char.isdigit() for char in label) and not ('alpha' in label or 'beta' in label):
#                 mask = (wave1 < center+seps[1][0]) & (wave1 > center-seps[1][1])
#             else:
#                 mask = (wave1 < center+seps[0][0]) & (wave1 > center-seps[0][1])

#             new_flux = flux1.copy()

#             new_flux[flux1 > 10] = np.nan

#             new_flux[mask] = np.nan

#             new_flux = new_flux[(wave1 < center+sep) & (wave1 > center-sep)]

#             narrow_gaussians = GaussianModel(prefix=label+'_narrow_')

#             polydeg = 1
#             polynomial = PolynomialModel(degree=polydeg)

#             comp_mult = narrow_gaussians + polynomial
#             pars_mult = comp_mult.make_params()

#             pars_mult.add(name='z', value=self.redshift)
#             min_v = 250 if label in ['H_6', 'H_9', 'H_10', 'H_11'] else 300
#             max_v = 2000 if label == 'H_alpha' else 500
#             guess = 500 if label == 'H_alpha' else 350
#             pars_mult.add(name='sigma_v_narrow', value=guess, min=min_v, max=max_v)

#             lam = self.linelist_dict[label]
#             for param in ['center', 'amplitude', 'sigma']:
#                 narrow_key = f'{label}_narrow_{param}'
#                 if param == 'center':
#                     value = lam
#                     vary = False
#                     min_ = None
#                     max_ = None
#                     expr = f'{lam:6.2f}*(1+z)'
#                 elif param == 'amplitude':
#                     value = 1
#                     vary = True
#                     min_ = -100
#                     max_ = 0
#                     expr = None
#                 elif param == 'sigma':
#                     vary = True
#                     min_ = None
#                     max_ = None
#                     expr = f'(sigma_v_narrow/3e5)*{label}_narrow_center'
#                 pars_mult[narrow_key] = Parameter(name=narrow_key, value=value,
#                                                   vary=vary, expr=expr,
#                                                   min=min_, max=max_)

#             for i in range(polydeg+1):
#                 pars_mult[f'c{i:1.0f}'].set(value=0)

#             out_comp_mult = comp_mult.fit(new_flux, pars_mult, x=lamb,
#                                           nan_policy='omit', max_nfev=1000)

#             comp = out_comp_mult.eval_components(x=lamb)
#             balmer_abs = comp[f'{label}_narrow_'] - np.median(comp[f'{label}_narrow_'])

#             self.new_spectra[(wave1 < center+sep) & (wave1 > center-sep)] -= balmer_abs
#             stamps.append(self.new_spectra[(wave1 < center+sep) & (wave1 > center-sep)])
#             comps.append(comp)
#             new_fluxes.append(new_flux)

#             header = self.hdus[0][1].header
#             header['EXTNAME_2'] = 'ST_AB_CORR'
#             header['DATE'] = str(datetime.date.today())

#             hdu = fits.PrimaryHDU(self.new_spectra, header)

#             hdul = fits.HDUList([hdu])
#             self.hdu_corrected = hdul
#             DIR = '/Users/javieratoro/Desktop/proyecto 2024-2/balmer_absorption/'
#             hdul.writeto(DIR + f'{self.names[0][:5]}_STELLAR_abscorr.fits',
#                          overwrite=True)

#         if showplot is True:
#             fig, axs = plt.subplots(4, 3, figsize=(13, 12))
#             axs = axs.ravel()

#             for idx, (stamp, label, comp, new_flux) in enumerate(zip(stamps,
#                                                                      balmer_lines,
#                                                                      comps,
#                                                                      new_fluxes)):
#                 center, sep = get_center_and_sep(label)

#                 lamb = wave1[(wave1 < center+sep) & (wave1 > center-sep)]
#                 flux2 = flux1[(wave1 < center + sep) & (wave1 > center - sep)]

#                 axs[idx].set_title(label)
#                 axs[idx].plot(lamb, flux2, 'red', lw=1, drawstyle='steps-mid',
#                               label='Observed spectrum', alpha=0.5)
#                 axs[idx].plot(lamb, new_flux, 'blue', lw=1, drawstyle='steps-mid',
#                               label='Line masked spectrum', alpha=0.5)
#                 axs[idx].plot(lamb, comp['polynomial'], 'green', lw=1,
#                               alpha=0.5)
#                 axs[idx].plot(lamb, comp[f'{label}_narrow_'], 'teal', lw=1,
#                               alpha=0.5)
#                 axs[idx].plot(lamb, flux2 - (comp[f'{label}_narrow_']),
#                               'magenta', lw=1,
#                               alpha=0.5, label='Corrected')
#                 axs[idx].vlines(center, 0, np.max(flux2), 'grey', '--',
#                                 alpha=0.3)
#                 axs[idx].set_xlabel(r'Obs. Wavelength ($\AA$)', size=14)
#                 axs[idx].set_ylabel(r'Flux ($10^{-17} erg/s/cm^{2}/\AA$)',
#                                     size=14)
#                 axs[idx].set_xlim([np.min(lamb), np.max(lamb)])
#                 axs[idx].set_ylim([np.min(comp[f'{label}_narrow_']) - 5, 30])
#                 axs[idx].legend()

#             for ax in axs[len(stamps):]:
#                 ax.axis("off")

#             fig.tight_layout()
#             plt.show()

#     def plot_new_spectra(self):
#         if self.new_spectra is None:
#             raise ValueError("There is not balmer absorption calibrated data. Run balmer_absorption again")

#         fig = plt.figure(constrained_layout=True, figsize=(13, 6))
#         axs = fig.subplot_mosaic([['Left', 'TopRight',
#                                    'TopRight2'],
#                                   ['Left', 'Bottom', 'Bottom']],
#                                  gridspec_kw={'width_ratios': [2, 1, 1]})

#         wave, flux, _, _ = self.datas[0]

#         axs['Left'].step(wave, flux, alpha=0.5, label='Obs spectra',
#                          lw=1, drawstyle='steps-mid')
#         axs['Left'].step(wave, self.new_spectra, alpha=0.5,
#                          label='Corrected spectra',
#                          lw=1, drawstyle='steps-mid')

#         axs['Left'].set_xlabel(r'Wavelength ($\AA$)')
#         axs['Left'].set_ylabel(r'Flux ($10^{-17} erg/s/cm^{2}/\AA$)')
#         axs['Left'].set_title(self.names[0] + f',z = {np.round(self.redshift, 4)}')
#         axs['Left'].legend(markerscale=5, frameon=False,
#                            bbox_to_anchor=(0.95, 0.75),
#                            loc='upper right', borderaxespad=0)
#         axs['Left'].set_xlim(3600, 9550)

#         axs['Left'].minorticks_on()
#         axs['Left'].tick_params(which='major', length=10, width=1.2,
#                                 direction='in')
#         axs['Left'].tick_params(which='minor', length=5, width=1.2,
#                                 direction='in')
#         axs['Left'].xaxis.set_ticks_position('both')
#         axs['Left'].yaxis.set_ticks_position('both')
#         if self.lines_waves is not None and self.line_name is not None:
#             for wave1, label in zip(self.lines_waves, self.line_name):
#                 axs['Left'].axvline(x=wave1, color='gray',
#                                     linestyle='--', alpha=0.2)
#                 axs['Left'].text(wave1, 0.95, '\n'+label,
#                                  rotation=90, ha='center', va='top',
#                                  color='k', size=8,
#                                  transform=axs['Left'].get_xaxis_transform())

#         axs['TopRight'].set_title(r'$H_{\beta}$')
#         xlim = self.lines_waves[self.line_name == 'H_beta'].values
#         axs['TopRight'].step(wave, flux, alpha=0.5, label='Obs spectra',
#                              lw=1, drawstyle='steps-mid')
#         axs['TopRight'].step(wave, self.new_spectra, alpha=0.5,
#                              label='Corrected spectra',
#                              lw=1, drawstyle='steps-mid')

#         wave_O3 = self.lines_waves[self.line_name == 'H_beta'].values
#         resta_O3 = np.abs(wave - wave_O3)
#         flux_O3 = flux[np.argmin(resta_O3)]
#         axs['TopRight'].set_xlim(xlim-50, xlim+50)
#         axs['TopRight'].set_ylim(0, flux_O3/5)

#         axs['TopRight2'].set_title(r'$H_{\gamma}$')
#         axs['TopRight2'].step(wave, flux, alpha=0.5, label='Obs spectra',
#                               lw=1, drawstyle='steps-mid')
#         axs['TopRight2'].step(wave, self.new_spectra, alpha=0.5,
#                               label='Corrected spectra',
#                               lw=1, drawstyle='steps-mid')

#         xlim1 = self.lines_waves[self.line_name == 'H_gamma'].values
#         wave_hb = self.lines_waves[self.line_name == 'H_gamma'].values
#         resta_hb = np.abs(wave - wave_hb)
#         flux_hb = flux[np.argmin(resta_hb)]
#         axs['TopRight2'].set_xlim(xlim1-30, xlim1+30)
#         axs['TopRight2'].set_ylim(-5, flux_hb + 10)

#         axs['Bottom'].step(wave, flux, alpha=0.5, label='Obs spectra',
#                            lw=1, drawstyle='steps-mid')
#         axs['Bottom'].step(wave, self.new_spectra, alpha=0.5,
#                            label='Corrected spectra',
#                            lw=1, drawstyle='steps-mid')
#         xlim_out = self.lines_waves[self.line_name == 'Ne3_3970'].values
#         xlim_in = self.lines_waves[self.line_name == 'H_11'].values
#         wave_hb = self.lines_waves[self.line_name == 'Ne3_3970'].values
#         resta_hb = np.abs(wave - wave_hb)
#         flux_hb = flux[np.argmin(resta_hb)]

#         axs['Bottom'].set_xlim(xlim_in - 50, xlim_out + 50)
#         axs['Bottom'].set_ylim(0, flux_hb)
#         axs['Bottom'].minorticks_on()
#         axs['Bottom'].tick_params(which='major', length=10,
#                                   width=1.2,
#                                   direction='in')
#         axs['Bottom'].tick_params(which='minor', length=5,
#                                   width=1.2,
#                                   direction='in')
#         axs['Bottom'].xaxis.set_ticks_position('both')
#         axs['Bottom'].yaxis.set_ticks_position('both')

#         mask = ((self.lines_waves.values < wave_hb) | (self.lines_waves.values == wave_hb))
#         for wavelength2, label1 in zip(self.lines_waves[mask], self.line_name[mask]):
#             axs['Bottom'].axvline(x=wavelength2, color='gray',
#                                   linestyle='--', alpha=0.2)
#             axs['Bottom'].text(wavelength2, 0.95, '\n'+label1, rotation=90,
#                                ha='center', va='top', color='k', size=8,
#                                transform=axs['Bottom'].get_xaxis_transform())
#         path = '/Users/javieratoro/Desktop/proyecto 2024-2/images/calibrated_spectra_'
#         plt.savefig(path + self.DIR[-6:-1] + '.pdf',
#                     format='pdf')
#         plt.show()

#     def plot_spectra(self):
#         for name1 in self.names:
#             print(f'Plotting for {name1}')
#             fig = plt.figure(constrained_layout=True, figsize=(13, 6))
#             axs = fig.subplot_mosaic([['Left', 'TopRight',
#                                        'TopRight2', 'TopRight3'],
#                                       ['Left', 'Bottom', 'Bottom', 'Bottom3']],
#                                      gridspec_kw={'width_ratios': [2, 1, 1, 1]
#                                                   })

#             for data in self.datas:
#                 wave, flux, _, name = data
#                 axs['Left'].step(wave, flux, alpha=0.5, label=name)

#             axs['Left'].set_xlabel(r'Wavelength ($\AA$)')
#             axs['Left'].set_ylabel(r'Flux ($10^{-17} erg/s/cm^{2}/\AA$)')
#             axs['Left'].set_title(name1 + f',z = {np.round(self.redshift, 4)}')
#             axs['Left'].legend(markerscale=5, frameon=False,
#                                bbox_to_anchor=(0.95, 0.8),
#                                loc='upper right', borderaxespad=0)
#             axs['Left'].set_xlim(3600, 9550)

#             axs['Left'].minorticks_on()
#             axs['Left'].tick_params(which='major', length=10, width=1.2,
#                                     direction='in')
#             axs['Left'].tick_params(which='minor', length=5, width=1.2,
#                                     direction='in')
#             axs['Left'].xaxis.set_ticks_position('both')
#             axs['Left'].yaxis.set_ticks_position('both')
#             if self.lines_waves is not None and self.line_name is not None:
#                 for wave1, label in zip(self.lines_waves, self.line_name):
#                     axs['Left'].axvline(x=wave1, color='gray',
#                                         linestyle='--', alpha=0.2)
#                     axs['Left'].text(wave1, 0.95, '\n'+label,
#                                      rotation=90, ha='center', va='top',
#                                      color='k', size=8,
#                                      transform=axs['Left'].get_xaxis_transform())

#             axs['TopRight'].set_title(r'$H_{\gamma}$ + [OIII]$\lambda$4363')
#             xlim = self.lines_waves[self.line_name == 'O3_4363'].values
#             for i in range(0, len(self.datas)):
#                 wave, flux, _, name = self.datas[i]
#                 axs['TopRight'].step(wave, flux, alpha=0.5, label=name)
#                 if i == len(self.datas)-1:
#                     wave_O3 = self.lines_waves[self.line_name == 'O3_4363'].values
#                     resta_O3 = np.abs(wave - wave_O3)
#                     flux_O3 = flux[np.argmin(resta_O3)]
#                     axs['TopRight'].set_xlim(xlim-50, xlim+50)
#                     axs['TopRight'].set_ylim(0.1, flux_O3+10)

#             axs['TopRight2'].set_title(r'[O II]$\lambda\lambda$3725,3727')
#             for i in range(0, len(self.datas)):
#                 wave, flux, _, name = self.datas[i]
#                 axs['TopRight2'].step(wave, flux, alpha=0.5, label=name)
#                 if i == len(self.datas)-1:
#                     xlim1 = 0.5*(self.lines_waves[self.line_name == 'O2_3725'].values +
#                                  self.lines_waves[self.line_name == 'O2_3727'].values)
#                     wave_hb = self.lines_waves[self.line_name == 'O2_3725'].values
#                     resta_hb = np.abs(wave - wave_hb)
#                     flux_hb = flux[np.argmin(resta_hb)]
#                     axs['TopRight2'].set_xlim(xlim1-30, xlim1+30)
#                     axs['TopRight2'].set_ylim(-5, flux_hb + 100)

#             for i in range(0, len(self.datas)):
#                 wave, flux, _,  name = self.datas[i]
#                 axs['Bottom'].step(wave, flux, alpha=0.5, label=name)
#                 if i == len(self.datas) - 1:
#                     xlim_out = self.lines_waves[self.line_name == 'Ar3_7753'].values
#                     xlim_in = self.lines_waves[self.line_name == 'He1_7067'].values
#                     wave_hb = self.lines_waves[self.line_name == 'He1_7067'].values
#                     resta_hb = np.abs(wave - wave_hb)
#                     flux_hb = flux[np.argmin(resta_hb)]

#                     axs['Bottom'].set_xlim(xlim_in - 100, xlim_out + 100)
#                     axs['Bottom'].set_ylim(flux_hb - 15, flux_hb + 15)
#                     axs['Bottom'].minorticks_on()
#                     axs['Bottom'].tick_params(which='major', length=10,
#                                               width=1.2,
#                                               direction='in')
#                     axs['Bottom'].tick_params(which='minor', length=5,
#                                               width=1.2,
#                                               direction='in')
#                     axs['Bottom'].xaxis.set_ticks_position('both')
#                     axs['Bottom'].yaxis.set_ticks_position('both')

#             mask = ((self.lines_waves.values > wave_hb) | (self.lines_waves.values == wave_hb))
#             for wavelength2, label1 in zip(self.lines_waves[mask], self.line_name[mask]):
#                 axs['Bottom'].axvline(x=wavelength2, color='gray',
#                                       linestyle='--', alpha=0.2)
#                 axs['Bottom'].text(wavelength2, 0.95, '\n'+label1, rotation=90,
#                                    ha='center', va='top', color='k', size=8,
#                                    transform=axs['Bottom'].get_xaxis_transform())

#             axs['TopRight3'].set_title(r'$H_{\beta}$ + [OIII]$\lambda$$\lambda$4959,5007')
#             xlim = 0.5*(self.lines_waves[self.line_name == 'O3_5008'].values +
#                         self.lines_waves[self.line_name == 'H_beta'].values)
#             for i in range(0, len(self.datas)):
#                 wave, flux, _, name = self.datas[i]
#                 axs['TopRight3'].step(wave, flux, alpha=0.5, label=name)
#                 if i == len(self.datas)-1:
#                     wave_O3 = self.lines_waves[self.line_name == 'H_beta'].values
#                     resta_O3 = np.abs(wave - wave_O3)
#                     flux_O3 = flux[np.argmin(resta_O3)]
#                     axs['TopRight3'].set_xlim(xlim-150, xlim+150)
#                     axs['TopRight3'].set_ylim(0.1, flux_O3+100)

#             axs['Bottom3'].set_title(r'$H_{\alpha}$ + [NII]$\lambda$$\lambda$6550,6585')
#             for i in range(0, len(self.datas)):
#                 wave, flux, _, name = self.datas[i]
#                 axs['Bottom3'].step(wave, flux, alpha=0.5, label=name)
#                 if i == len(self.datas)-1:
#                     xlim1 = (self.lines_waves[self.line_name == 'H_alpha'].values)
#                     resta_hb = np.abs(wave - wave_hb)
#                     flux_hb = flux[np.argmin(resta_hb)]
#                     axs['Bottom3'].set_xlim(xlim1-100, xlim1+100)
#                     axs['Bottom3'].set_ylim(-5, flux_hb + 100)
#             path = '/Users/javieratoro/Desktop/proyecto 2024-2/images/spectra_'
#             plt.savefig(path + name1[:5] + '.pdf', format='pdf')
#             plt.show()

#     def fit_spectra(self, mode, show_Plot=False, broad=True, nfev=None):
#         if mode == 'corrected':
#             wave, _, noise, _ = self.datas[0]
#             hdu = self.hdu_corrected
#             flux = hdu[0].data
#         else:
#             wave, flux, noise, _ = self.datas[0]

#         sm_noise = gaussian_filter1d(noise, sigma=25)

#         fit = fitSpectrum(wave, flux, noise, linelist=self.linelist_dict,
#                           z_init=self.redshift, weights=1/sm_noise**2,
#                           showPlot=show_Plot,
#                           broad=broad, nfev=nfev)
#         self.model = fit
#         return fit

#     def plot_fit(self, mode, show_fit=False, broad=True, nfev=None):
#         if mode == 'corrected':
#             wave, _, noise, _ = self.datas[0]
#             hdu = self.hdu_corrected
#             flux = hdu[0].data
#             if self.init_corr is None:
#                 print('Modeling spectra')
#                 fit = self.fit_spectra(mode, show_Plot=show_fit,
#                                        broad=broad, nfev=None)
#                 self.init_corr = fit
#         else:
#             wave, flux, noise, _ = self.datas[0]
#             if self.init is None:
#                 print('Modeling spectra')
#                 fit = self.fit_spectra(mode, show_Plot=show_fit,
#                                        broad=broad, nfev=None)
#                 self.init = fit

#         fit = self.init
#         fig = plt.figure(constrained_layout=True, figsize=(13, 6))
#         axs = fig.subplot_mosaic([['Left', 'TopRight',
#                                    'TopRight2', 'TopRight3'],
#                                   ['Left', 'Bottom', 'Bottom', 'Bottom3']],
#                                  gridspec_kw={'width_ratios': [2, 1, 1, 1]
#                                               })
#         axs['Left'].step(wave, flux, alpha=0.5, label='Obs spectrum')

#         axs['Left'].fill_between(wave, -noise, noise,
#                                  label='Error spectrum',
#                                  zorder=-2, color='0.6')
#         axs['Left'].axhline(0, color='k', ls='-', zorder=-1)
#         axs['Left'].step(wave, fit.best_fit, '-', lw=0.5,
#                          label='Best-model fit')

#         axs['Left'].set_xlabel(r'Wavelength ($\AA$)')
#         axs['Left'].set_ylabel(r'Flux ($10^{-17} erg/s/cm^{2}/\AA$)')
#         axs['Left'].set_title(self.names[0] + f', z = {np.round(self.redshift, 4)}')
#         axs['Left'].legend(markerscale=5, frameon=False,
#                            bbox_to_anchor=(0.3, 0.97),
#                            loc='upper right', borderaxespad=0)
#         axs['Left'].set_xlim(3600, 9550)
#         axs['Left'].minorticks_on()
#         axs['Left'].tick_params(which='major', length=10, width=1.2,
#                                 direction='in')
#         axs['Left'].tick_params(which='minor', length=5, width=1.2,
#                                 direction='in')
#         axs['Left'].xaxis.set_ticks_position('both')
#         axs['Left'].yaxis.set_ticks_position('both')
#         if self.lines_waves is not None and self.line_name is not None:
#             for wave_1, label in zip(self.lines_waves, self.line_name):
#                 axs['Left'].axvline(x=wave_1, color='gray',
#                                     linestyle='--', alpha=0.2)
#                 axs['Left'].text(wave_1, 0.85, '\n'+label,
#                                  rotation=90, ha='center', va='top',
#                                  color='k', size=8,
#                                  transform=axs['Left'].get_xaxis_transform())

#         axs['TopRight'].set_title(r'$H_{\gamma}$ + [OIII]$\lambda$4363')
#         xlim = self.lines_waves[self.line_name == 'O3_4363'].values
#         axs['TopRight'].step(wave, flux, alpha=0.5)
#         axs['TopRight'].fill_between(wave, -noise, noise,
#                                      label='Error spectrum',
#                                      zorder=-2, color='0.6')

#         if fit is not None:
#             axs['TopRight'].step(wave, fit.best_fit, '-', lw=0.5)

#         wave_O3 = self.lines_waves[self.line_name == 'O3_4363'].values
#         resta_O3 = np.abs(wave - wave_O3)
#         flux_O3 = flux[np.argmin(resta_O3)]
#         axs['TopRight'].set_xlim(xlim-50, xlim+50)
#         axs['TopRight'].set_ylim(0.1, flux_O3+10)

#         axs['TopRight2'].set_title(r'[O II]$\lambda\lambda$3725,3727')
#         axs['TopRight2'].step(wave, flux, alpha=0.5)
#         axs['TopRight2'].fill_between(wave, -noise, noise,
#                                       label='Error spectrum',
#                                       zorder=-2, color='0.6')

#         if fit is not None:
#             axs['TopRight2'].step(wave, fit.best_fit, '-', lw=0.5)

#         xlim1 = 0.5*(self.lines_waves[self.line_name == 'O2_3725'].values +
#                      self.lines_waves[self.line_name == 'O2_3727'].values)
#         wave_hb = self.lines_waves[self.line_name == 'O2_3725'].values
#         resta_hb = np.abs(wave - wave_hb)
#         flux_hb = flux[np.argmin(resta_hb)]
#         axs['TopRight2'].set_xlim(xlim1 - 30, xlim1 + 30)
#         axs['TopRight2'].set_ylim(-5, flux_hb + 100)

#         axs['Bottom'].step(wave, flux, alpha=0.5)
#         axs['Bottom'].fill_between(wave, -noise, noise,
#                                    label='Error spectrum',
#                                    zorder=-2, color='0.6')
#         axs['Bottom'].step(wave, fit.best_fit, '-', lw=0.5)
#         xlim_out = self.lines_waves[self.line_name == 'Ar3_7753'].values
#         xlim_in = self.lines_waves[self.line_name == 'He1_7067'].values
#         wave_hb = self.lines_waves[self.line_name == 'He1_7067'].values
#         resta_hb = np.abs(wave - wave_hb)
#         flux_hb = flux[np.argmin(resta_hb)]
#         axs['Bottom'].set_xlim(xlim_in - 100, xlim_out + 100)
#         axs['Bottom'].set_ylim(flux_hb - 15, flux_hb + 15)
#         axs['Bottom'].minorticks_on()
#         axs['Bottom'].tick_params(which='major', length=10, width=1.2,
#                                   direction='in')
#         axs['Bottom'].tick_params(which='minor', length=5, width=1.2,
#                                   direction='in')
#         axs['Bottom'].xaxis.set_ticks_position('both')
#         axs['Bottom'].yaxis.set_ticks_position('both')

#         val = self.lines_waves.values
#         mask = ((val > wave_hb) | (val == wave_hb))
#         for wavelength2, label1 in zip(self.lines_waves[mask],
#                                        self.line_name[mask]):
#             axs['Bottom'].axvline(x=wavelength2, color='gray',
#                                   linestyle='--', alpha=0.2)
#             axs['Bottom'].text(wavelength2, 0.95, '\n'+label1, rotation=90,
#                                ha='center', va='top', color='k', size=8,
#                                transform=axs['Bottom'].get_xaxis_transform())

#         axs['TopRight3'].set_title(r'$H_{\beta}$ + [OIII]$\lambda$$\lambda$4959,5007')
#         xlim = 0.5*(self.lines_waves[self.line_name == 'O3_5008'].values +
#                     self.lines_waves[self.line_name == 'H_beta'].values)
#         axs['TopRight3'].step(wave, flux, alpha=0.5)
#         axs['TopRight3'].fill_between(wave, -noise, noise,
#                                       label='Error spectrum',
#                                       zorder=-2, color='0.6')
#         axs['TopRight3'].step(wave, fit.best_fit, '-', lw=0.5)
#         wave_O3 = self.lines_waves[self.line_name == 'H_beta'].values
#         resta_O3 = np.abs(wave - wave_O3)
#         flux_O3 = flux[np.argmin(resta_O3)]
#         axs['TopRight3'].set_xlim(xlim-150, xlim+150)
#         axs['TopRight3'].set_ylim(-5, flux_O3+100)

#         axs['Bottom3'].set_title(r'$H_{\alpha}$ + [NII]$\lambda$$\lambda$6550,6585')
#         axs['Bottom3'].step(wave, flux, alpha=0.5)
#         axs['Bottom3'].fill_between(wave, -noise, noise,
#                                     label='Error spectrum',
#                                     zorder=-2, color='0.6')

#         axs['Bottom3'].step(wave, fit.best_fit, '-', lw=0.5)
#         xlim1 = (self.lines_waves[self.line_name == 'H_alpha'].values)
#         resta_hb = np.abs(wave - wave_hb)
#         flux_hb = flux[np.argmin(resta_hb)]
#         axs['Bottom3'].set_xlim(xlim1-100, xlim1+100)
#         axs['Bottom3'].set_ylim(-5, flux_hb + 100)
#         path = '/Users/javieratoro/Desktop/proyecto 2024-2/images/spectra_'
#         plt.savefig(path + self.DIR[-6:-1] + '.pdf',
#                     format='pdf')
#         plt.show()

#     def fit_MC(self, mode, numMC=2, showPlot=False, broad=True, nfev=None):
#         if mode == 'corrected':
#             wave, _, noise, _ = self.datas[0]
#             hdu = self.hdu_corrected
#             flux = hdu[0].data
#         else:
#             wave, flux, noise, _ = self.datas[0]

#         if self.init is None:
#             print('Calculating the first init params')
#             fit = fitSpectrum(wave, flux, noise, linelist=self.linelist_dict,
#                               z_init=self.redshift, weights=1/noise**2,
#                               showPlot=showPlot,
#                               broad=broad, nfev=nfev)
#             self.init = fit
#         params = self.init.params

#         info = fitSpectrumMC(wave, flux, noise,
#                              linelist=self.linelist_dict,
#                              z_init=self.redshift, weights=1/noise**2,
#                              numMC=numMC, showPlot=showPlot,
#                              init_params=params, broad=broad, nfev=nfev)
#         self.MC_unc = info
#         df = pd.DataFrame(columns=['ID', 'mass', 'z', 'name', 'flux',
#                                    'fluxerr',
#                                    'narrow_flux', 'narrow_fluxerr',
#                                    'broad_flux', 'broad_fluxerr'])

#         self.bright_lines = ['O2_3725', 'O2_3727', 'H_alpha', 'H_beta',
#                              'H_gamma', 'O3_5008', 'O3_4959', 'N2_6550',
#                              'N2_6585', 'S2_6716', 'S2_6730']

#         for label in self.linelist_dict.keys():  # Corrected loop variable
#             narrow_ = np.asarray(info[label+"_narrow"])
#             row = {'ID': self.names[0],
#                    'mass': self.mass,
#                    'z': np.median(info['z']),
#                    'name': label,
#                    'flux': np.mean(narrow_),
#                    'fluxerr': np.std(narrow_),
#                    'narrow_flux': np.mean(info[label+"_narrow"]),
#                    'narrow_fluxerr': np.std(info[label+"_narrow"]),
#                    'broad_flux': -9999.9,
#                    'broad_fluxerr': -9999.9}

#             if label in self.bright_lines:
#                 broad_ = np.asarray(info[label+"_broad"])
#                 flux = narrow_ + broad_
#                 row['flux'] = np.mean(flux)
#                 row['fluxerr'] = np.std(narrow_)
#                 row['broad_flux'] = np.mean(info[label+"_broad"])
#                 row['broad_fluxerr'] = np.std(info[label+"_broad"])

#             df.loc[len(df)] = row
#         if mode == 'corrected':
#             df.to_csv(f'/Users/javieratoro/Desktop/proyecto 2024-2/balmer_absorption/{self.names[0][:5]}.csv', index=False)
#             info.to_csv(f'/Users/javieratoro/Desktop/proyecto 2024-2/balmer_absorption/{self.names[0][:5]}_model.csv', index=False)
#         else:
#             df.to_csv(f'/Users/javieratoro/Desktop/proyecto 2024-2/lines/{self.names[0][:5]}.csv', index=False)
#             info.to_csv(f'/Users/javieratoro/Desktop/proyecto 2024-2/lines/{self.names[0][:5]}_model.csv', index=False)

#     def fit_auroral(self, wave1, flux1, showplot=False, return_comps=False):
#         stamps, comps = [], []
#         path = '/Users/javieratoro/Desktop/proyecto 2024-2/lines/'
#         params = pd.read_csv(path + f'{self.names[0][:5]}_model.csv')
#         sigma_broad = np.median(params['sigma_v_broad'])
#         sigma_narrow = np.median(params['sigma_v_narrow'])

#         def get_center_and_sep(label):
#             """Return the center and separation value based on the label."""
#             center = self.linelist_dict[label] * (1 + self.redshift)
#             center_N2 = self.linelist_dict['N2_6585'] * (1 + self.redshift)
#             center_N1 = self.linelist_dict['N2_6550'] * (1 + self.redshift)
#             sep = 2.5 * (center_N2 - center_N1)
#             return center, sep

#         auroral_lines = ['N2_5756', 'O1_6363',
#                          'O3_4363', 'S3_6312',
#                          'O2_7322']

#         columns = ['sigma_v_narrow', 'sigma_v_broad']

#         for label in auroral_lines:
#             columns.append(str(label) + '_narrow_amplitude')
#             columns.append(str(label) + '_broad_amplitude')

#         columns.append('O2_7333_narrow_amplitude')
#         columns.append('O2_7333_broad_amplitude')

#         df = pd.DataFrame(columns=columns)
#         row = {'sigma_v_narrow': sigma_narrow,
#                'sigma_v_broad': sigma_broad}

#         for label in auroral_lines:
#             o2 = ['O2_7322', 'O2_7333']
#             if label in o2:
#                 narrow_gaussians = []
#                 broad_gaussians = []
#                 for label in o2:
#                     center, sep = get_center_and_sep(label)

#                     mask = ((wave1 < center+sep) & (wave1 > center-sep))

#                     lamb = wave1[mask]
#                     flux2 = flux1[mask]

#                     for label_ in self.linelist_dict.keys():
#                         center_, _ = get_center_and_sep(label_)
#                         if center_ < center+sep or center_ > center-sep:
#                             if label_ not in o2:
#                                 sep_ = 4
#                                 mask2 = ((lamb < center_+sep_) & (lamb > center_-sep_))
#                                 flux2[mask2] = np.nan

#                                 med = np.median(flux2)
#                                 flux2[flux2 < med-1] = np.nan

#                     # Narrow component
#                     narrow_gaussian = GaussianModel(prefix=label+'_narrow_')
#                     narrow_gaussians.append(narrow_gaussian)

#                     broad_gaussian = GaussianModel(prefix=label+'_broad_')
#                     broad_gaussians.append(broad_gaussian)

#                 polydeg = 1
#                 polynomial = PolynomialModel(degree=polydeg)

#                 sum_of_gaussians = broad_gaussians[0] + narrow_gaussians[0]
#                 for narrow, broad_value in zip(narrow_gaussians[1:],
#                                                broad_gaussians[1:]):
#                     sum_of_gaussians += (broad_value + narrow)

#                 comp_mult = sum_of_gaussians + polynomial
#                 pars_mult = comp_mult.make_params()

#                 pars_mult.add(name='z', value=self.redshift, vary=False)

#                 pars_mult.add(name='sigma_v_narrow', value=sigma_narrow,
#                               vary=False)

#                 pars_mult.add(name='sigma_v_broad', value=sigma_broad,
#                               vary=False)

#                 for label in o2:
#                     lam = self.linelist_dict[label]
#                     for param in ['center', 'amplitude', 'sigma']:
#                         narrow_key = f'{label}_narrow_{param}'
#                         if param == 'center':
#                             value = lam
#                             vary = False
#                             expr = f'{lam:6.2f}*(1+z)'
#                         elif param == 'amplitude':
#                             value = 1
#                             vary = True
#                             expr = None
#                         elif param == 'sigma':
#                             vary = False
#                             expr = f'(sigma_v_narrow/3e5)*{label}_narrow_center'
#                         pars_mult[narrow_key] = Parameter(name=narrow_key,
#                                                           value=value,
#                                                           vary=vary, expr=expr,
#                                                           min=0.0)

#                         broad_key = f'{label}_broad_{param}'
#                         if param == 'center':
#                             value = lam
#                             vary = False
#                             expr = f'{lam:6.2f}*(1+z)'
#                         elif param == 'amplitude':
#                             value = 0.3
#                             vary = True
#                             expr = None
#                         elif param == 'sigma':
#                             vary = False
#                             expr = f'(sigma_v_broad/3e5)*{label}_broad_center'
#                         pars_mult[broad_key] = Parameter(name=broad_key,
#                                                          value=value, min=0.0,
#                                                          vary=vary, expr=expr)

#                 for i in range(polydeg+1):
#                     pars_mult[f'c{i:1.0f}'].set(value=0)

#                 fit = comp_mult.fit(flux2, pars_mult, x=lamb,
#                                     nan_policy='omit', max_nfev=1000)

#                 stamps.append([lamb, flux2])
#                 comps.append(fit)
#                 for label in o2:
#                     name1 = str(label) + "_narrow_amplitude"
#                     name2 = str(label) + "_broad_amplitude"
#                     row[name1] = float(fit.params[name1].value)
#                     row[name2] = float(fit.params[name2].value)

#             else:
#                 center, sep = get_center_and_sep(label)

#                 mask = ((wave1 < center+sep) & (wave1 > center-sep))

#                 lamb = wave1[mask]
#                 flux2 = flux1[mask]

#                 for label_ in self.linelist_dict.keys():
#                     center_, _ = get_center_and_sep(label_)
#                     if center_ < center+sep or center_ > center-sep:
#                         if label_ != label:
#                             sep_ = 4
#                             mask2 = ((lamb < center_+sep_) &
#                                      (lamb > center_-sep_))
#                             flux2[mask2] = np.nan

#                             med = np.median(flux2)
#                             flux2[flux2 < med-1] = np.nan

#                 narrow_gaussians = []
#                 broad_gaussians = []

#                 # Narrow component
#                 narrow_gaussian = GaussianModel(prefix=label+'_narrow_')
#                 narrow_gaussians.append(narrow_gaussian)

#                 broad_gaussian = GaussianModel(prefix=label+'_broad_')
#                 broad_gaussians.append(broad_gaussian)

#                 polydeg = 1
#                 polynomial = PolynomialModel(degree=polydeg)

#                 sum_of_gaussians = broad_gaussians[0] + narrow_gaussians[0]
#                 for narrow, broad_value in zip(narrow_gaussians[1:],
#                                                broad_gaussians[1:]):
#                     sum_of_gaussians += (broad_value + narrow)

#                 comp_mult = sum_of_gaussians + polynomial
#                 pars_mult = comp_mult.make_params()

#                 pars_mult.add(name='z', value=self.redshift, vary=False)

#                 pars_mult.add(name='sigma_v_narrow', value=sigma_narrow,
#                               vary=False)

#                 pars_mult.add(name='sigma_v_broad', value=sigma_broad,
#                               vary=False)

#                 lam = self.linelist_dict[label]
#                 for param in ['center', 'amplitude', 'sigma']:
#                     narrow_key = f'{label}_narrow_{param}'
#                     if param == 'center':
#                         value = lam
#                         vary = False
#                         expr = f'{lam:6.2f}*(1+z)'
#                     elif param == 'amplitude':
#                         value = 1
#                         vary = True
#                         expr = None
#                     elif param == 'sigma':
#                         vary = False
#                         expr = f'(sigma_v_narrow/3e5)*{label}_narrow_center'
#                     pars_mult[narrow_key] = Parameter(name=narrow_key,
#                                                       value=value,
#                                                       vary=vary, expr=expr,
#                                                       min=0.0)

#                     broad_key = f'{label}_broad_{param}'
#                     if param == 'center':
#                         value = lam
#                         vary = False
#                         expr = f'{lam:6.2f}*(1+z)'
#                     elif param == 'amplitude':
#                         value = 0.3
#                         vary = True
#                         expr = None
#                     elif param == 'sigma':
#                         vary = False
#                         expr = f'(sigma_v_broad/3e5)*{label}_broad_center'
#                     pars_mult[broad_key] = Parameter(name=broad_key,
#                                                      value=value, min=0.0,
#                                                      vary=vary, expr=expr)

#                 for i in range(polydeg+1):
#                     pars_mult[f'c{i:1.0f}'].set(value=0)

#                 fit = comp_mult.fit(flux2, pars_mult, x=lamb,
#                                     nan_policy='omit', max_nfev=1000)

#                 stamps.append([lamb, flux2])
#                 comps.append(fit)

#                 name1 = str(label) + "_narrow_amplitude"
#                 name2 = str(label) + "_broad_amplitude"
#                 row[name1] = float(fit.params[name1].value)
#                 row[name2] = float(fit.params[name2].value)

#         df.loc[len(df)] = row

#         if showplot is True:
#             fig, axs = plt.subplots(3, 2, figsize=(10, 12))
#             axs = axs.ravel()

#             for idx, (stamp, label, fit) in enumerate(zip(stamps,
#                                                           auroral_lines,
#                                                           comps)):
#                 center, sep = get_center_and_sep(label)

#                 lamb, flux = stamp
#                 # print(fit.params)
#                 comp = fit.eval_components(x=lamb)
#                 best = fit.eval(x=lamb)
#                 axs[idx].set_title(label)
#                 axs[idx].plot(lamb, flux, 'red', lw=1, drawstyle='steps-mid',
#                               label='Masked spectrum', alpha=0.5)
#                 axs[idx].plot(lamb, best, 'black', lw=1, alpha=0.5,
#                               label='Best fit')
#                 axs[idx].plot(lamb,
#                               comp[f'{label}_narrow_'] + comp['polynomial'],
#                               'teal', linestyle='--', lw=1, alpha=0.5,
#                               label='Narrow component')
#                 axs[idx].plot(lamb,
#                               comp[f'{label}_broad_'] + comp['polynomial'],
#                               'blue', linestyle='--', lw=1, alpha=0.5,
#                               label='Broad component')
#                 if label in o2:
#                     axs[idx].plot(lamb,
#                                   comp['O2_7333_narrow_'] + comp['polynomial'],
#                                   'teal', linestyle='--', lw=1, alpha=0.5,
#                                   label='Narrow component')
#                     axs[idx].plot(lamb,
#                                   comp['O2_7333_broad_'] + comp['polynomial'],
#                                   'blue', linestyle='--', lw=1, alpha=0.5,
#                                   label='Broad component')

#                 axs[idx].set_xlabel(r'Obs. Wavelength ($\AA$)', size=14)
#                 axs[idx].set_ylabel(r'Flux ($10^{-17} erg/s/cm^{2}/\AA$)',
#                                     size=14)
#                 axs[idx].set_xlim([np.min(lamb), np.max(lamb)])
#                 axs[idx].set_ylim([np.min(fit.best_fit) - 5,
#                                    np.max(fit.best_fit) + 5])
#                 axs[idx].legend()

#             for ax in axs[len(stamps):]:
#                 ax.axis("off")

#             fig.tight_layout()
#             plt.show()
#         if return_comps is True:
#             return df, [comps, stamps]
#         else:
#             return df

#     def fitSpectrumMC_auroral(self, numMC=400):

#         columns = ['sigma_v_narrow', 'sigma_v_broad']

#         auroral_lines = ['N2_5756', 'O1_6363',
#                          'O3_4363', 'S3_6312',
#                          'O2_7322', 'O2_7333']

#         for label in auroral_lines:
#             name1 = str(label) + "_narrow_amplitude"
#             name2 = str(label) + "_broad_amplitude"
#             columns.append(name1)
#             columns.append(name2)

#         df = pd.DataFrame(columns=columns)

#         wave, flux, noise, _ = self.datas[0]

#         for i in range(numMC):
#             # Create a data set with random offsets scaled by uncertainties

#             yoff = np.random.randn(len(flux)) * noise

#             # res = self.fit_auroral(wave, flux + yoff, self)
#             res = self.fit_auroral(wave, flux + yoff)

#             df = pd.concat([df, res], ignore_index=True)

#         return df

#     def fit_MC_auroral(self, return_comps=False, numMC_=100):
#         _, flux, _, _ = self.datas[0]

#         info = self.fitSpectrumMC_auroral(numMC=numMC_)

#         df = pd.DataFrame(columns=['ID', 'mass', 'z', 'name', 'flux',
#                                    'fluxerr',
#                                    'narrow_flux', 'narrow_fluxerr',
#                                    'broad_flux', 'broad_fluxerr'])

#         auroral_lines = ['N2_5756', 'O1_6363',
#                          'O3_4363', 'S3_6312',
#                          'O2_7322', 'O2_7333']

#         for label in auroral_lines:  # Corrected loop variable
#             narrow_ = np.asarray(info[label+"_narrow_amplitude"])
#             broad_ = np.asarray(info[label+"_broad_amplitude"])
#             flux = narrow_ + broad_
#             row = {'ID': self.names[0],
#                    'mass': self.mass,
#                    'z': self.redshift,
#                    'name': label,
#                    'flux': np.median(flux),
#                    'fluxerr': np.std(flux),
#                    'narrow_flux': np.median(info[label+"_narrow_amplitude"]),
#                    'narrow_fluxerr': np.std(info[label+"_narrow_amplitude"]),
#                    'broad_flux': np.median(info[label+"_broad_amplitude"]),
#                    'broad_fluxerr': np.std(info[label+"_broad_amplitude"])}
#             df.loc[len(df)] = row

#         narrow_1 = np.asarray(info['O2_7322_narrow_amplitude'])
#         broad_1 = np.asarray(info["O2_7322_broad_amplitude"])
#         narrow_2 = np.asarray(info["O2_7333_narrow_amplitude"])
#         broad_2 = np.asarray(info["O2_7333_broad_amplitude"])
#         flux = narrow_1 + broad_1 + narrow_2 + broad_2
#         row = {'ID': self.names[0],
#                'mass': self.mass,
#                'z': self.redshift,
#                'name': 'O2_7322_7333',
#                'flux': np.median(flux),
#                'fluxerr': np.std(flux),
#                'narrow_flux': np.median(narrow_1 + narrow_2),
#                'narrow_fluxerr': np.std(narrow_1 + narrow_2),
#                'broad_flux': np.median(broad_1 + broad_2),
#                'broad_fluxerr': np.std(broad_1 + broad_2)}
#         df.loc[len(df)] = row

#         path = '/Users/javieratoro/Desktop/proyecto 2024-2/lines/'
#         df.to_csv(path + f'{self.names[0][:5]}_auroral.csv', index=False)
#         info.to_csv(path + f'{self.names[0][:5]}_auroral_model.csv',
#                     index=False)
#         return df
