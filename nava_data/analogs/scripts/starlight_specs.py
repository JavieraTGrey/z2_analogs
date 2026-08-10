# Created on Mon Feb 23th, 2024
# @author: bnavarrete

# This script takes the output of the STARLIGHT run, subtracts the stellar
# spectrum, and saves the final spectrum. It also saves the estimation of the
# stellar mass and extinction in magnitudes.

import numpy as np
import pandas as pd
import spectres
import os
import sys
magePath = '/home/benjamin/Documents/analogs/'
os.sys.path.append(magePath)

from magE.functions import save_catalog
from magE.plotutils import * 
from magE.constutils import *

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

# =============================================================================
#
# Global data, variables, etc.
#
# =============================================================================

# Path to models
modelsPath = '/home/benjamin/Documents/analogs/specs/ssp_models/'

# Redshifts and velocity dispersions used (Estimated from emission line fitting)

z = {'J0021': 0.09867719705615696,
     'J0023': 0.05337982520594247,
     'J0136': 0.059743843974085,
     'J0240': 0.08253587076459773,
     'J0252': 0.028290341351223554,
     'J0305': 0.08586106612890987,
     'J0950': 0.09805976839067501,
     'J1146': 0.05679983702007108,
     'J1226': 0.09451828919104312,
     'J1444': 0.03904556311382511,
     'J1448': 0.02766687028469216,
     'J1624': 0.03162901232702408,
     'J2101': 0.19652130752330071,
     'J2119': 0.034036666237891026,
     'J2212': 0.17735090032776832,
     'J2215': 0.07777318597770254,
     'J2225': 0.06697772673981063,
     'J2337': 0.0720277216924494
}

vel_disp = {'J0021': 193.2136322343019,
            'J0023': 117.46443239198206,
            'J0136': 111.22532538686987,
            'J0240': 116.34711304200162,
            'J0252': 106.91182112571903,
            'J0305': 141.5183392299635,
            'J0950': 130.7466102321719,
            'J1146': 72.64085936643463,
            'J1226': 125.44791494410336,
            'J1444': 82.3129293007496,
            'J1448': 116.68796556940187,
            'J1624': 108.17841450649067,
            'J2101': 87.94299872787136,
            'J2119': 92.45356210565836,
            'J2212': 118.90581695051276,
            'J2215': 103.33177830191447,
            'J2225': 113.86331871094488,
            'J2337': 120.54227977987833
}

spec_lims = {'J0021': np.array([3300, 10000]),
             'J0023': np.array([3300, 10000]),
             'J0136': np.array([3300, 10000]),
             'J0240': np.array([3300, 10000]),
             'J0252': np.array([3400, 10000]),
             'J0305': np.array([3300, 10000]),
             'J0950': np.array([3350,  8500]),
             'J1146': np.array([3550,  8950]),
             'J1226': np.array([3400, 10000]),
             'J1444': np.array([3550, 10000]),
             'J1448': np.array([3600, 10000]),
             'J1624': np.array([3400, 10000]),
             'J2101': np.array([3300,  8000]),
             'J2119': np.array([3600, 10000]),
             'J2212': np.array([3000,  9000]),
             'J2215': np.array([3300, 10000]),
             'J2225': np.array([3300, 10000]),
             'J2337': np.array([3300, 10000])
}

norms = {'J0021': 2.828721E+01,
         'J0023': 5.480447E+00,
         'J0136': 6.351274E+00,
         'J0240': 5.070773E+00,
         'J0252': 7.336310E+00,
         'J0305': 6.334548E+00,
         'J0950': 1.765180E+00,
         'J1146': 4.478281E-01,
         'J1226': 5.304925E+00,
         'J1444': 2.526953E+00,
         'J1448': 3.939305E+01,
         'J1624': 1.013901E+00,
         'J2101': 3.582791E+00,
         'J2119': 5.566973E+00,
         'J2212': 1.287464E+00,
         'J2215': 6.604016E-01,
         'J2225': 1.417379E+00,
         'J2337': 5.468073E+00
}

stellar_mass = {'J0021': 9.09644E+04,
                'J0023': 2.31317E+04,
                'J0136': 1.21893E+04,
                'J0240': 3.58146E+04,
                'J0252': 3.45991E+04,
                'J0305': 1.24254E+04,
                'J0950': 3.16398E+04,
                'J1146': 8.76437E+01,
                'J1226': 1.81891E+04,
                'J1444': 2.72406E+03,
                'J1448': 1.84642E+04,
                'J1624': 3.75373E+03,
                'J2101': 4.51621E+04,
                'J2119': 2.06420E+04,
                'J2212': 8.60332E+03,
                'J2215': 3.06633E+03,
                'J2225': 8.99659E+03,
                'J2337': 2.17626E+04
}



if __name__ == '__main__':
    
    # Read STARLIGHT output and galaxy spectrum
    
    for galname in galnames:
        if galname == 'J2212_comp':
            continue
        else:
            
            # Read galaxy spectrum
            spectrum = pd.read_csv('./specs/final_spec/' + galname + '.csv')
            wave = spectrum['wave'].to_numpy()
            flux = spectrum['flux'].to_numpy()
            err = spectrum['noise'].to_numpy()
            igal = np.argmax(np.array(galnames) == galname)
            
            # Stellar mass
            lum_distance = cosmo.luminosity_distance(z[galname]) * \
                        (1 * u.cm / (3.24078e-25 * u.Mpc))
            m_star = np.log10(stellar_mass[galname] * 1e-17 * (1/3.826e33) * 4 * np.pi * ((lum_distance.value)**2))
            
            # Read STARLIGHT output
            with open('./specs/ssp_models/' + galname+'.cxt.sc4.C99.im.CAL.BS', 'r') as bf:
                
                # limits of lines to read
                line_lims = [531, 7231]
                
                # To store lines
                lines = []
                for i, line in enumerate(bf):
                    
                    # read lines inside limits
                    if (i >= line_lims[0]) & (i <= line_lims[1]):
                        lines.append(line.strip())
                    elif i > 7231:
                        # don't read after line 7 to save time
                        break
            
            # Formatting in usable variables
            wl = []
            best_model = []
            for element in lines:
                wl.append(float(element.split()[0]))
                best_model.append(float(element.split()[2]))
            
            # Resample model spec
            rf_wave = wave / (1 + z[galname])
            mask_spec = (rf_wave > spec_lims[galname][0]) & (rf_wave < spec_lims[galname][1])
            res_model = spectres.spectres(rf_wave[mask_spec],
                                          np.array(wl), np.array(best_model))
            
            # Subtracted spectrum
            subspec = ((flux[mask_spec] / norms[galname] / 1e-17) - res_model) * norms[galname] * 1e-17
            
            # Data frame to store
            properties_dict = {'galname': galname,
                            'z': z[galname],
                            'm_star': m_star}
            
            subspec_dict = {'wave': rf_wave[mask_spec],
                            'flux': subspec,
                            'err': err[mask_spec]}
            
            properties_pd = pd.DataFrame(data=properties_dict, index=[0])
            subspec_pd = pd.DataFrame(data=subspec_dict, index=np.arange(0, len(subspec), 1))
            save_catalog(galname, properties_pd, magePath + '/results/ssp_props.csv')
            save_catalog(galname, subspec_pd, magePath + '/specs/final_subspecs/'+galname+'.csv')