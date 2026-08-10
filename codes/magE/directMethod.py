"""
This python script contain the directT class which calculates
metallicities using the direct Te method detailed in
Pérez-Montero et al. (2017).

Code by BN in 2021
"""

from uncertainties import ufloat
import uncertainties.umath as uum

class directT(object):

    def __init__(self, I_lines, I_err):
        self.I_OII3726, self.I_OII3729, self.I_NeIII3868, self.I_OIII4363, \
        self.I_HB, self.I_OIII4959, self.I_OIII5007, self.I_NII6548, \
        self.I_HA, self.I_NII6583, self.I_SII6716, self.I_SII6731, \
        self.I_OII7319, self.I_OII7330 = I_lines
        
        self.I_OII3726_err, self.I_OII3729_err, self.I_NeIII3868_err, \
        self.I_OIII4363_err, self.I_HB_err, self.I_OIII4959_err, \
        self.I_OIII5007_err, self.I_NII6548_err, self.I_HA_err, \
        self.I_NII6583_err, self.I_SII6716_err, self.I_SII6731_err, \
        self.I_OII7319_err, self.I_OII7330_err = I_err
    
    @classmethod
    def to_unc(self, I, I_err):
        I = ufloat(I, I_err)
        return I

    def to_format(self, x):
        value = x.nominal_value
        err = x.std_dev
        return (value, err)

    def RO3(self):
        I_4959 = self.to_unc(self.I_OIII4959, self.I_OIII4959_err)
        I_5007 = self.to_unc(self.I_OIII5007, self.I_OIII5007_err)
        I_4363 = self.to_unc(self.I_OIII4363, self.I_OIII4363_err)
        RO3 = (I_4959 + I_5007) / I_4363
        self.R_O3 = RO3

    def RO2(self):
        I_3726 = self.to_unc(self.I_OII3726, self.I_OII3726_err)
        I_3729 = self.to_unc(self.I_OII3729, self.I_OII3729_err)
        I_7319 = self.to_unc(self.I_OII7319, self.I_OII7319_err)
        I_7330 = self.to_unc(self.I_OII7330, self.I_OII7330_err)
        RO2 = (I_3726 + I_3729) / (I_7319 + I_7330)
        self.R_O2 = RO2

    def RS2(self):
        I_6716 = self.to_unc(self.I_SII6716, self.I_SII6716_err)
        I_6731 = self.to_unc(self.I_SII6731, self.I_SII6731_err)
        RS2 = I_6716 / I_6731
        self.R_S2 = RS2

    def tOIII(self):
        tOIII = 0.7840 - 0.0001357 * self.R_O3 + (48.44 / self.R_O3)
        self.t_OIII = tOIII

    def ne(self):
        a0 = 16.054 - (7.79 / self.t_OIII) - (11.32 * self.t_OIII)
        a1 = -22.66 + (11.08 / self.t_OIII) + (16.02 * self.t_OIII)
        b0 = -21.61 + (11.89 / self.t_OIII) + (14.59 * self.t_OIII)
        b1 = 9.17 - (5.09 / self.t_OIII) - (6.18 * self.t_OIII)
        num = (self.R_S2 * a0) + a1
        den = (self.R_S2 * b0) + b1
        self.n_e = (num / den) * 1e3

    def tOII(self):
        a0 = 0.2526 - (0.000357 * self.n_e) - (0.43 / self.n_e)
        a1 = 0.00136 + (self.n_e * 5.42 * 1e-6) + (0.00481 / self.n_e)
        a2 = 35.624 - (0.0172 * self.n_e) + (25.12 / self.n_e)
        tOII = a0 + (a1 * self.R_O2) + (a2 / self.R_O2)
        self.t_OII = tOII

    def O_H(self):
        tl = self.t_OII
        I_3726 = self.to_unc(self.I_OII3726, self.I_OII3726_err)
        I_3729 = self.to_unc(self.I_OII3729, self.I_OII3729_err)
        I_HB = self.to_unc(self.I_HB, self.I_HB_err)
        log_O_H_12 = uum.log10((I_3726 + I_3729) / I_HB) + \
        5.887 + (1.64 / tl) - (0.543 * uum.log10(tl)) \
        + (0.000114 * self.n_e)
        O_H = 10**(log_O_H_12 - 12)
        return(O_H)

    def O2_H(self):
        th = self.t_OIII
        I_4959 = self.to_unc(self.I_OIII4959, self.I_OIII4959_err)
        I_5007 = self.to_unc(self.I_OIII5007, self.I_OIII5007_err)
        I_HB = self.to_unc(self.I_HB, self.I_HB_err)
        log_O2_H_12 = uum.log10((I_4959 + I_5007) / I_HB) + 6.1868 + \
        (1.2491 / th) - (0.5816 * uum.log10(th))
        O2_H = 10**(log_O2_H_12 - 12)
        return O2_H

    def OxAbundance(self):
        O_abundance = 12 + uum.log10(self.O_H() + self.O2_H())
        self.log_O_H_12 = O_abundance

    def NAbundance(self):
        tl = self.t_OII
        I_6548 = self.to_unc(self.I_NII6548, self.I_NII6548_err)
        I_6583 = self.to_unc(self.I_NII6583, self.I_NII6583_err)
        I_HB = self.to_unc(self.I_HB, self.I_HB_err)
        N_abundance = uum.log10((I_6548 + I_6583) / I_HB) + 6.291 + \
        (0.90221 / tl) - (0.5511 * uum.log10(tl))
        self.log_N_H_12 = N_abundance

    def ratio_N_O(self):
        tl = self.t_OII
        I_6583 = self.to_unc(self.I_NII6583, self.I_NII6583_err)
        I_3726 = self.to_unc(self.I_OII3726, self.I_OII3726_err)
        I_3729 = self.to_unc(self.I_OII3729, self.I_OII3729_err)
        log_N_O = uum.log10(I_6583 / (I_3726 + I_3729)) + 0.493 - \
        (0.025 * tl) - (0.687 / tl) + (0.1621 * uum.log10(tl))
        self.log_N_O = log_N_O
        
    def R23(self):
        I_3726 = self.to_unc(self.I_OII3726, self.I_OII3726_err)
        I_3729 = self.to_unc(self.I_OII3729, self.I_OII3729_err) 
        I_4959 = self.to_unc(self.I_OIII4959, self.I_OIII4959_err)
        I_5007 = self.to_unc(self.I_OIII5007, self.I_OIII5007_err)
        I_HB = self.to_unc(self.I_HB, self.I_HB_err)
        R23 = (I_4959 + I_5007 + I_3726 + I_3729)/I_HB
        self.R23 = uum.log10(R23)
    
    def O32(self):
        I_3726 = self.to_unc(self.I_OII3726, self.I_OII3726_err)
        I_3729 = self.to_unc(self.I_OII3729, self.I_OII3729_err) 
        I_4959 = self.to_unc(self.I_OIII4959, self.I_OIII4959_err)
        I_5007 = self.to_unc(self.I_OIII5007, self.I_OIII5007_err)
        O32 = (I_4959 + I_5007)/(I_3726 + I_3729)
        self.O32 = uum.log10(O32)
        
    def N2(self):
        I_6583 = self.to_unc(self.I_NII6583, self.I_NII6583_err)
        I_HA = self.to_unc(self.I_HA, self.I_HA_err)
        N2 = I_6583/I_HA
        self.N2 = uum.log10(N2)
        
    def O3N2(self):
        I_5007 = self.to_unc(self.I_OIII5007, self.I_OIII5007_err)
        I_HB = self.to_unc(self.I_HB, self.I_HB_err)
        I_6583 = self.to_unc(self.I_NII6583, self.I_NII6583_err)
        I_HA = self.to_unc(self.I_HA, self.I_HA_err) 
        O3N2 = (I_5007/I_HB)/(I_6583/I_HA)
        self.O3N2 = uum.log10(O3N2)

def run_all(direct_class):
    direct_class.RO3()
    direct_class.RO2()
    direct_class.RS2()
    direct_class.tOIII()
    direct_class.ne()
    direct_class.tOII()
    direct_class.OxAbundance()
    direct_class.NAbundance()
    direct_class.ratio_N_O()
    direct_class.R23()
    direct_class.O32()
    direct_class.N2()
    direct_class.O3N2()