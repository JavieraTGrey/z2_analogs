from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp
from astropy.table import Table
from scipy.interpolate import interp1d
import re
import pandas as pd
from bokeh.plotting import figure
from bokeh.io import output_notebook, show

_MATCH_STRING = re.compile(r"(\d+) \d+ \d+ (\d+\.\d+) (\d+\.\d+) "
                          r"\d+ \d+\.\d+ \d+\.\d+ \d+\.\d+")

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


'''
Modelo parabólico para fitear el continuo del espectro
'''


def parabola(x, a, b, c):
    return  a * x**2 + b * x + c

'''
Funcion gausiana para graficar componentes de los fiteos
'''
def gausiana(x, z, a, c, bal):
    H = bal*(1 + z)
    A = a/(np.sqrt(2*np.pi)*c)
    return A*np.exp(-(x - H)**2 / 2 / c**2)

'''
Función que sustrae el continuo de alrededor de las líneas de emisión
gaussianas. Recibe el arreglo de longitudes de onda, flujo, el error
asociado al flujo e intervalos que indican dónde está el continuo para
trabajar sin la línea de emisión. Los intervalos deben ser del tipo:
[[x1, x2], [x3, x4]]. La RMS que entrega la función es la que corresponde
a la de los datos sin sacarles el continuo.
'''


def sust_continuo(x, y, err, intervals):
    low, high = intervals
    x_low = x[low[0]: low[1]]
    x_high = x[high[0]: high[1]]
    y_low = y[low[0]: low[1]]
    y_high = y[high[0]: high[1]]
    err_low = err[low[0]: low[1]]
    err_high = err[high[0]: high[1]]
    x_to_fit = np.concatenate((x_low, x_high))
    y_to_fit = np.concatenate((y_low, y_high))
    err_to_fit = np.concatenate((err_low, err_high))
    opt, cov = sp.curve_fit(parabola, x_to_fit, y_to_fit, sigma=err_to_fit)
    fit = parabola(x, *opt)
    n = len(y_to_fit)
    RMS = np.sqrt(np.sum((y_to_fit - parabola(x_to_fit, *opt))**2) / n)
    return x[low[0]:high[1]], y[low[0]:high[1]] - fit[low[0]:high[1]], err[low[0]:high[1]], RMS


'''
Clase que entrega fiteos de lineas de emision en un intervalo
para cierta funcion, y redshift dados.
'''
class FittedSpectrum(object):
    def __init__(self, wavelength, flux, noise, intervals, linefunc, n, z=0.0775, sigma1=1.0, sigma2=1.5):
        self.redshift = z
        self.wavelength = wavelength
        self.flux = flux
        self.noise = noise
        self.intervals = intervals
        self.linefunc = linefunc
        self.line = n
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.lines = np.array(['H-alpha report:', 'H-beta report:', 'H-gamma report:',
                          'H-delta report:','H-epsilon report:', 'H-zeta report:',
                          'H-eta report:'])

    def fit(self):
        wl, flx, noise, RMS = sust_continuo(self.wavelength, self.flux,
                                            self.noise, self.intervals)
        init = (self.redshift, max(flx), max(flx)/2, self.sigma1, self.sigma2)
        opt, cov = sp.curve_fit(self.linefunc, wl, flx, p0=init,
                                sigma=noise)
        return wl, flx, noise, RMS, opt, cov

    def plot(self, p=1):
        if p == 1:
            lok = 'top_right'
        if p == 2:
            lok = 'top_left'
        wl, flx, noise, RMS, opt, cov = self.fit()
        F = figure(plot_height=550,
                   plot_width=650,
                   x_axis_label='Wavelength [Å]',
                   y_axis_label='Flux [erg/cm^2/s]',
                   background_fill_color="#fafafa")
        F.circle(wl, flx, legend_label='Data', fill_color='blue')
        F.vbar(wl, 0.0001, top=flx+noise, bottom=flx-noise)
        F.line(wl, self.linefunc(wl, *opt), legend_label='Fit',
               line_color='orange')
        F.line(wl, noise, line_color='green', legend_label='Noise')
        F.line(wl, -noise, line_color='green')
        F.line(wl, np.ones(len(wl))*RMS, legend_label='RMS',
               line_color='red', line_dash='dashed')
        F.line(wl, -np.ones(len(wl))*RMS,
               line_color='red', line_dash='dashed')
        F.legend.location = lok
        F.yaxis.axis_label_text_font_size = "16pt"
        F.xaxis.axis_label_text_font_size = "16pt"
        F.yaxis.major_label_text_font_size = "10pt"
        F.xaxis.major_label_text_font_size = "10pt"
        show(F)
        
        
        '''
        lok = None
        if p == 1:
            lok = 'upper right'
        if p == 2:
            lok = 'upper left'
        wl, flx, noise, RMS, opt, cov = self.fit()
        fig = plt.figure()
        plt.title(self.lines[self.line])
        plt.errorbar(wl, flx, noise, marker='.', ls='None', label='Data')
        plt.plot(wl, self.linefunc(wl, *opt), label='Fit')
        plt.plot(wl, noise, color='green', label='Noise')
        plt.plot(wl, -noise, color='green')
        plt.axhline(RMS, ls='--', color='red', label='RMS')
        plt.legend(loc=lok)
        plt.show()
        '''
    
    def report(self):
        wl, flx, noise, RMS, opt, cov = self.fit()
        print('-------------------------------------------------------')
        print(self.lines[self.line],
            'z = ' + str(opt[0]) + ' ± ' + str(np.sqrt(np.diag(cov)[0])),
            'F1 = '+ str(opt[1]) + ' ± ' + str(np.sqrt(np.diag(cov)[1])),
            'F2 = '+ str(opt[2]) + ' ± ' + str(np.sqrt(np.diag(cov)[2])),
            'sigma1 = ' + str(opt[3]) + ' ± ' + str(np.sqrt(np.diag(cov)[3])),
            'sigma2 = ' + str(opt[4]) + ' ± ' + str(np.sqrt(np.diag(cov)[4])),
              sep='\n')







'''
specline = fits.open('galSpecLine-dr8.fits')[1].data
flujos = np.array([specline['H_ALPHA_FLUX'],
                   specline['H_ALPHA_FLUX_ERR'],
                   specline['H_BETA_FLUX'],
                   specline['H_BETA_FLUX_ERR']])

not_nan = np.isfinite(specline['H_ALPHA_FLUX']) * np.isfinite(specline['H_BETA_FLUX'])

ha_flux = specline['H_ALPHA_FLUX'][not_nan]
hb_flux = specline['H_BETA_FLUX'][not_nan]
ha_flux_err = specline['H_ALPHA_FLUX_ERR'][not_nan]
hb_flux_err = specline['H_BETA_FLUX_ERR'][not_nan]

Ha_dHa = ha_flux/ha_flux_err
Hb_dHb = hb_flux/hb_flux_err

ha_sn_3 = Ha_dHa > 3
hb_sn_3 = Hb_dHb > 3
high_sn = ha_sn_3 * hb_sn_3

hab = specline['H_ALPHA_FLUX'][high_sn] / specline['H_BETA_FLUX'][high_sn]

plt.hist(hab, bins=100)
'''
