"""
Created on Sun Dec 4 11:23 2022

@author: bnavarrete

This script has information about useful constants for the analysis
"""

import numpy as np

specobjID = {'J2225': 422244936562272256,
             'J2215': 1243134008619984896,
             'J0240': 513494502204270592,
             'J0021': 439223316213950464,
             'J0950': 300731025703593984,
             'J1146': 318786936665827328,
             'J1226': 3242617920756410368,
             'J1444': 661039343871223808,
             'J1448': 346799464301750272,
             'J2101': 717307953417316352,
             'J2119': 1111371552277948416,
             'J0023': 735339668748396544,
             'J0136': 1691125637219117056,
             'J0252': 797248221209454592,
             'J0305': 798423599642863616,
             'J2212': 1243114217410684928,
             'J2212_comp': 1163229473462577152,
             'J2337': 767911057829160960,
             'J1624': 409879001813772288
}

# Wavelengths in air for emission lines (From NIST)
# Here I store the lines that usually appears in our spectra

wave_cat = {'[SIII]9532': 9531.100, '[SIII]9069': 9068.600,
            'P11': 8862.89, 'P12': 8750.46, 'P13': 8665.02, 'P14': 8598.39,
            'P15': 8545.38, 'P16': 8502.49, 'P17': 8467.26, '[OI]8446': 8446.36,
            'P18': 8437.95, 'P19': 8413.32, '[ArIII]7751': 7751.06,
            '[OII]7330': 7330.19, '[OII]7319': 7319.92, 'HeI7281': 7281.349,
            '[ArIII]7135': 7135.8, 'HeI7065': 7065.19, '[SII]6730': 6730.815,
            '[SII]6716': 6716.440, 'HeI6678': 6678.151,
            '[NII]6584': 6583.45, 'Ha': 6562.79, '[NII]6548': 6548.05,
            '[OI]6363': 6363.776, '[SIII]6312': 6312.06, '[OI]6300': 6300.304,
            'HeI5875': 5875.621, '[NII]5755': 5754.59, 'HeI5015': 5015.6783,
            '[OIII]5007': 5006.843, '[FeIII]4985': 4985.86,
            '[OIII]4959': 4958.911, '4921': 4921.8, # I COULD NOT FIND THIS LINE
            # MIGHT BE A SKY LINE
            'Hb': 4861.35, '[ArIV]4740': 4740.20,
            '[ArIV]4711': 4711.35, 'HeII4685': 4685.710, '[FeIII]4658': 4658.05,
            'HeI4471': 4471.4802, 'HeI4388': 4387.9296, '[OIII]4363': 4363.209,
            '[FeII]4360': 4359.332,
            'Hg': 4340.472, 'Hd': 4101.710277, '[SII]4076': 4076.349, '[SII]4068': 4068.60,
            'HeI4026': 4026.1914, 'H7': 3970.075, '[NeIII]3967': 3967.47,
            'H8': 3889.064, 'HeI3888': 3888.648, '[NeIII]3868': 3868.76,
            'H9': 3835.397, 'HeI3819': 3819.6074, 'H10': 3797.909,
            'H11': 3770.633, 'H12': 3750.129869, 'H13': 3734.369, '[OII]3729': 3728.815,
            '[OII]3726': 3726.032, 'H14': 3721.945, 'H15': 3711.978, 'H16': 3703.859,
            'H17': 3697.157, 'H18': 3691.551
}

bright = ['Ha', '[OIII]5007', '[OIII]4959', 'Hb']

to_fit = ['Ha', 'Hb', 'Hg', 'Hd', 'H7', 'H8', 'H9', 'H10', '[OII]3726',
          '[OII]3729', '[SII]6716', '[SII]6730', '[OIII]5007', '[OIII]4959',
          '[OI]6300', '[OI]6363', '[NII]6548', '[NII]6584']

# In some galaxies telluric correction was not good. Use these ranges
# to mask telluric effects in the pPXF fit. These have to be corrected by 
# the redshift of each galaxy to create the mask.
telluric = [[3554, 3584],
            [7627, 7721],
]

# line FeIII 4985.87 was taken from Berg et al. 2021 (Zotero)
# At the end it does exist, but NIST was giving me 4987.25 in vacuum, in air
# it is ok.

# dictionary of source properties

galnames = ['J2225', 'J2215', 'J0240', 'J0021',
            'J0950', 'J1146', 'J1226', 'J1444', 'J1448', 'J2101', 'J2119',
            'J0023', 'J0136', 'J0252', 'J0305', 'J2212', 'J2212_comp', 'J2337',
            'J1624']

galnames_ext = ['J2225-0011', 'J2215+0002', 'J0240-0828', 'J0021+0052',
                'J0950+0042', 'J1146+0053', 'J1226+0415', 'J1444+0409',
                'J1448-0110', 'J2101-0555', 'J2119+0052', 'J0023-0948',
                'J0136-0037', 'J0252+0114', 'J0305+0040', 'J2212+0006',
                'J2212+0006_comp','J2337-0010', 'J1624-0022']

ra = ['336.292220', '333.846060', '40.217490', '5.254290',
      '147.597160', '176.705570', '186.549560', '221.172380',
      '222.022420', '315.309980', '319.992950', '5.915040',
      '24.127470', '43.142910', '46.396320', '333.179400',
      '333.178280358', '354.466400', '246.042127615']

dec = ['-0.198030', '0.046330', '-8.474280', '0.880030',
       '0.708130', '0.896140', '4.260020', '4.161590',
       '-1.182690', '-5.919530', '0.875990', '-9.813530',
       '-0.632210', '1.245540', '0.683120', '0.113490',
       '0.113230814', '-0.166810', '-0.367370172']

ra_h = ['22:25:10.13', '22:15:23.05', '02:40:52.19', '00:21:01.02',
        '09:50:23.31', '11:46:49.33', '12:26:11.89', '14:44:41.37',
        '14:48:05.38', '21:01:14.39', '21:19:58.30', '00:23:39.61',
        '01:36:30.59', '02:52:34.29', '03:05:35.11', '22:12:43.05',
        '22:12:42.78', '23:37:51.93', '16:24:10.11']

dec_h = ['-00:11:52.89', '+00:02:46.79', '-08:28:27.41', '+00:52:48.11',
         '+00:42:29.25', '+00:53:46.09', '+04:15:36.07', '+04:09:41.73',
         '-01:10:57.68', '-05:55:10.29', '+00:52:33.55', '-09:48:48.72',
         '-00:37:55.97', '+01:14:43.94', '+00:40:59.24', '+00:06:48.55',
         '+00:06:47.63', '-00:10:00.50', '-00:22:02.60']

z_SDSS = [0.067, 0.077, 0.082, 0.098,
          0.098, 0.057, 0.094, 0.039,
          0.027, 0.196, 0.034, 0.053,
          0.059, 0.028, 0.086, 0.177,
          0.177, 0.072, 0.031]

# A more precise estimation of the redshift is given by manual inspection of
# Ha, Hb and OIII simultaneously in each spectrum. For J0136 and J0252 the
# guessed redshift is the same than in z_SDSS since there is no data for
# these.
z_BN = [0.067, 0.07775, 0.08255, 0.0987,
        0.098035, 0.0568, 0.09452, 0.03905,
        0.0277, 0.1965, 0.03405, 0.05337,
        0.05974, 0.02829, 0.08587, 0.17735,
        0.17754, 0.07203, 0.03162]

# All given in km/s.
# When no vel disp estimation is given in SDSS, use the vdisp error
sigma_SDSS = [19.885, 643.78, 745.16, 748.35,
              808.64, 46.558, 686.11, 48.374,
              60.11, 648.82, 16.19, 14.008,
              20.958, 723.10, 25.015, 704.66,
              850.00, 30.455, 14.124]

# For masking emission lines in pPXF, we use these window parmeter to give a
# width around the line wavelengths to account for line width.
window = [12, 14, 20, 20,
          16, 9, 16, 11,
          18, 14, 9, 13,
          None, None, 17, 14,
          None, 13, None]

source_dict = {'name': np.array(galnames),
               'ext_name': np.array(galnames_ext),
               'RA': np.array(ra, dtype=float),
               'DEC': np.array(dec, dtype=float),
               'z_SDSS': np.array(z_SDSS),
               'z_BN': np.array(z_BN),
               'vel_disp': np.array(sigma_SDSS),
               'window': np.array(window)
}


# arc widths for intrinsic width correction
arc_slope18 = 9.1260e-05
arc_slope19a = 9.1375e-05
arc_slope19b = 1.6064e-04
arc_slope23 = 1.0181e-04

arc_intercept18 = 0.04415798
arc_intercept19a = 0.04053405
arc_intercept19b = -0.04641747
arc_intercept23 = -0.02634920

# Temperature relations

slope_o3o2 = 0.5405749144322302
slope_o3o2_std = 0.2643923712267538
intercept_o3o2 = 0.35313980869589195
intercept_o3o2_std = 0.34693508724574956

slope_o3s3 = 0.8378785778037261
slope_o3s3_std = 0.2511248804431554
intercept_o3s3 = 0.20929001271505945
intercept_o3s3_std = 0.32645567645361817

slope_s3o2 = 0.7395637045795564
slope_s3o2_std = 0.5799004541508995
intercept_s3o2 = 0.16112932802863478
intercept_s3o2_std = 0.7478419462633727

# Temperature relations

slope_o3o2_final = 0.4591676113236236
slope_o3o2_final_std = 0.28097981963755947
intercept_o3o2_final = 0.4867283097651363
intercept_o3o2_final_std = 0.3772562764327305

slope_o3s3_final = 0.8688552307961621
slope_o3s3_final_std = 0.20368923550978135
intercept_o3s3_final = 0.16552662154461664
intercept_o3s3_final_std = 0.2755373496714309

slope_s3o2_final = 0.5548747885173688
slope_s3o2_final_std = 0.3058034409694138
intercept_s3o2_final = 0.42582678177553623
intercept_s3o2_final_std = 0.41143456855246124


# for old fitting
oiia_wav = 7332.01
oiib_wav = 7321.99
siia_wav = 6732.68
siib_wav = 6718.29
ha_wav = 6564.614
niib_wav = 6549.86
niia_wav = 6585.27
oiiia_wav = 5008.239
oiiib_wav = 4960.295
hb_wav = 4862.721
oiii_wav = 4364.436
oiic_wav = 3729.875
oiid_wav = 3727.092
neiiia_wav = 3868.75
