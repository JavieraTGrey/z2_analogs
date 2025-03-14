import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, PolynomialModel
from lmfit.parameter import Parameter
import pandas as pd


def fitSpectrum(lams, flux, flux_error, linelist, z_init=0,
                weights=None, showPlot=False, params=None,
                broad=True, nfev=None):

    if hasattr(lams, 'value'):
        lams = lams.value
    if hasattr(flux, 'value'):
        flux = flux.value
    if hasattr(flux_error, 'value'):
        flux_error = flux_error.value

    narrow_gaussians = []
    broad_gaussians = []
    for label in linelist.keys():
        # Narrow component
        narrow_gaussian = GaussianModel(prefix=label+'_narrow_')
        narrow_gaussians.append(narrow_gaussian)

        if broad is True:
            broad_gaussian = GaussianModel(prefix=label+'_broad_')
            broad_gaussians.append(broad_gaussian)

    # Create a polynomial model for the continuum
    polydeg = 5
    polynomial = PolynomialModel(degree=polydeg)

    # Combine models
    if broad is True:
        sum_of_gaussians = broad_gaussians[0] + narrow_gaussians[0]
        for narrow, broad_value in zip(narrow_gaussians[1:],
                                       broad_gaussians[1:]):
            sum_of_gaussians += (broad_value + narrow)
    else:
        sum_of_gaussians = narrow_gaussians[0]
        for gaussian in narrow_gaussians[1:]:
            sum_of_gaussians += gaussian

    comp_mult = sum_of_gaussians + polynomial
    pars_mult = comp_mult.make_params()

    if params is None:
        pars_mult.add(name='z', value=z_init, vary=False)

        # Add the 'sigma_v' parameter to tie narrow sigmas together
        pars_mult.add(name='sigma_v_narrow', value=50, min=20, max=70)

        # Loop through emission lines to define parameters for narrow and broad
        for label in linelist.keys():
            lam = linelist[label]
            for param in ['center', 'amplitude', 'sigma']:
                narrow_key = f'{label}_narrow_{param}'
                if param == 'center':
                    value = lam
                    vary = True
                    expr = f'{lam:6.2f}*(1+z)'
                elif param == 'amplitude':
                    value = 1
                    vary = True
                    expr = None
                elif param == 'sigma':
                    vary = True
                    expr = f'(sigma_v_narrow/3e5)*{label}_narrow_center'
                pars_mult[narrow_key] = Parameter(name=narrow_key, value=value,
                                                  vary=vary, expr=expr,
                                                  min=0.0)

        bright_lines = ['O2_3725', 'O2_3727', 'H_alpha', 'H_beta', 'H_gamma',
                        'O3_5008', 'O3_4959', 'N2_6550', 'N2_6585', 'S2_6716',
                        'S2_6730']
        if broad is True:
            pars_mult.add(name='sigma_v_broad', value=90, min=70, max=500)

            for label in bright_lines:
                lam = linelist[label]
                for param in ['center', 'amplitude', 'sigma']:
                    broad_key = f'{label}_broad_{param}'
                    if param == 'center':
                        value = lam
                        vary = True
                        expr = f'{lam:6.2f}*(1+z)'
                    elif param == 'amplitude':
                        value = 0.3
                        vary = True
                        expr = None
                    elif param == 'sigma':
                        vary = True
                        expr = f'(sigma_v_broad/3e5)*{label}_broad_center'
                    pars_mult[broad_key] = Parameter(name=broad_key,
                                                     value=value, min=0.0,
                                                     vary=vary, expr=expr)

        for i in range(polydeg+1):
            pars_mult[f'c{i:1.0f}'].set(value=0)

    else:
        pars_mult = params

    if weights is None:
        out_comp_mult = comp_mult.fit(flux, pars_mult, x=lams,
                                      nan_policy='omit', max_nfev=nfev)
    else:
        out_comp_mult = comp_mult.fit(flux, pars_mult, x=lams,
                                      weights=weights, nan_policy='omit',
                                      max_nfev=nfev)

    if showPlot:
        plot_spec_fit(lams, flux, flux_error, linelist, z=None,
                      model=out_comp_mult, broad=broad)

    return out_comp_mult


def plot_spec_fit(lams, flux, flux_error, linelist, z=None, model=None,
                  broad=True):
    if z is None:
        if model is not None:
            z = model.params['z'].value
        else:
            z = 0

    fig2, ax = plt.subplots(figsize=(10, 4))
    comps = model.eval_components(x=lams)
    narrow_key = [key for key, _ in comps.items() if 'narrow' in key]
    ax.plot(lams, flux, lw=1, drawstyle='steps-mid', label='Observed spectrum')
    ax.plot(lams, model.best_fit, lw=1, drawstyle='steps-mid',
            label='Best model')
    ax.plot(lams, comps['polynomial'], 'r-', label='Polynomial component',
            lw=1, drawstyle='steps-mid', alpha=0.3)

    if broad is True:
        broad_key = [key for key, _ in comps.items() if 'broad' in key]
        for i, broad_line in enumerate(broad_key):
            if i == len(broad_key) - 1:
                ax.plot(lams, comps[broad_line] + comps['polynomial'], 'k--',
                        label='Broad component', alpha=0.3)
            else:
                ax.plot(lams, comps[broad_line] + comps['polynomial'], 'k--',
                        alpha=0.3)

    for i, narrow_line in enumerate(narrow_key):
        if i == len(narrow_key) - 1:
            ax.plot(lams, comps[narrow_line] + comps['polynomial'], 'g--',
                    label='narrow component', alpha=0.3)
        else:
            ax.plot(lams, comps[narrow_line] + comps['polynomial'], 'g--',
                    alpha=0.3)

    for label in linelist.keys():
        obs_lam = linelist[label] * (1+z)
        ax.axvline(obs_lam, linestyle='--', linewidth=1, color='k', lw=0.5)
        ax.text(obs_lam, 0.99, '\n'+label, rotation=90, ha='center', va='top',
                color='k', size=8, transform=ax.get_xaxis_transform())

    ax.set_xlabel(r'Obs. Wavelength ($\AA$)', size=14)
    ax.set_ylabel(r'Flux ($10^{-17} erg/s/cm^{2}/\AA$)', size=14)
    ax.set_xlim([np.min(lams), np.max(lams)])
    ax.legend()
    fig2.tight_layout()

    return fig2


def calculate_updated_error_spectrum(flux, model_spectrum, flux_error, wht_1D):
    # wht_1D is the 1D weight array!
    # Calculate the residuals between the flux and model spectrum
    residual = flux - model_spectrum

    # Calculate the scaled residuals
    scaled_residual = residual / flux_error

    # Get the valid pixels based on wht_1D > 0
    valid_pixels = np.where(wht_1D > 0)[0]
    scaled_residual_valid = scaled_residual[valid_pixels]

    # Use masked array to ignore NaN values
    scaled_residual_valid = np.ma.masked_invalid(scaled_residual_valid)

    # Calculate the lower and upper bounds for outlier removal
    q1 = np.percentile(scaled_residual_valid.compressed(), 25)
    q3 = np.percentile(scaled_residual_valid.compressed(), 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.8 * iqr
    upper_bound = q3 + 1.8 * iqr

    # Apply the outlier filter
    filtered_residuals = scaled_residual_valid[(
        scaled_residual_valid >= lower_bound) &
        (scaled_residual_valid <= upper_bound)]

    # Calculate the scaling factor as the standard deviation of
    # filtered_residuals
    scaling_factor = np.ma.std(filtered_residuals)

    print("The scaling factor for the error is: ", scaling_factor)

    # Scale the flux error
    scaled_flux_error = flux_error * scaling_factor

    return scaled_flux_error


def fitSpectrumMC(lams, flux, scaled_flux_error, linelist, z_init=0.1,
                  weights=None, numMC=400, nfev=None,
                  showPlot=False, init_params=None, broad=True):

    columns = ['iter', 'n_eval', 'success', 'message', 'ier', 'z',
               'sigma_v_narrow', 'sigma_v_broad']

    bright_lines = ['O2_3725', 'O2_3727', 'H_alpha', 'H_beta',
                    'H_gamma', 'O3_5008', 'O3_4959', 'N2_6550',
                    'N2_6585', 'S2_6716', 'S2_6730']

    for label in linelist:
        columns.append(str(label) + '_narrow')
        if str(label) in bright_lines:
            columns.append(str(label) + '_broad')

    df = pd.DataFrame(columns=columns)

    if init_params is None:
        fit_init = fitSpectrum(lams, flux, scaled_flux_error,
                               linelist=linelist, z_init=z_init,
                               weights=weights, showPlot=showPlot,
                               broad=broad, nfev=nfev)
        init_params = fit_init.params

    for i in range(numMC):
        # Create a data set with random offsets scaled by uncertainties
        yoff = flux + np.random.randn(len(flux)) * scaled_flux_error

        fit = fitSpectrum(lams, yoff, scaled_flux_error,
                          linelist=linelist,
                          z_init=z_init, weights=weights, showPlot=showPlot,
                          params=init_params, broad=broad, nfev=nfev)

        row = {'iter': i + 1,
               'z': float(fit.params['z'].value),
               'sigma_v_narrow': float(fit.params['sigma_v_narrow'].value),
               'sigma_v_broad': -9999.0,
               'n_eval': fit.nfev,
               'success': fit.success,
               'message': fit.lmdif_message,
               'ier': fit.ier}
        if broad is True:
            row['sigma_v_broad'] = float(fit.params['sigma_v_broad'].value)

        for label in linelist:  # Corrected loop variable
            name1 = str(label) + "_narrow_amplitude"
            name2 = str(label) + "_broad_amplitude"
            row[str(label) + "_narrow"] = float(fit.params[name1].value)
            if str(label) in bright_lines:
                row[str(label) + "_broad"] = float(fit.params[name2].value)

        df.loc[len(df)] = row
    return df
