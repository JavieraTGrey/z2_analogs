#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:27:48 2022

@author: gdago
"""
from astropy.io import fits
import os
import sys
magePath = '/home/benjamin/Documents/analogs/'
os.sys.path.append(magePath)
from magE.spec.spec_classes import *

import numpy as np
import matplotlib.pyplot as plt
from os.path import basename

specfldr = '/home/benjamin/Documents/analogs/specs/fits/' # HERE IS WERE I STORE THE NON-MERGED SPECTRA
filename = 'j2119+0052_1.fits'
specfile = specfldr+filename # THIS IS ORIGINAL MAGE FILE

templpath= '/home/benjamin/Documents/stellar_libraries/EMILES_PADOVA00_BASE_UN_FITS/' # THIS IS THE FOLDER WHERE I HAVE THE TEMPLATE FITS FILES

# THE FOLLOWING STRING TELLS THE SCRIPT WHICH KIND OF TEMPLATES I WANT TO USE:
# "Eun" STANDS FOR "EXTENDED MILES" AND "UNIMODAL" ("Mun" WOULD BE "MILES"+"UNIMODAL")
# 1.3 IS THE SLOPE OF THE IMF
# THIS IS A STANDARD APPROACH, YOU CAN FIND MORE INFORMATION IN THE MILES WEBSITE
templglob = 'Eun1.3*.fits'

def save_spectrum(specfile,wave,spec,path,error=None):

    hdr = fits.Header()

    hdr['HIERARCH NAME'] = basename(specfile)[:-5]
    primary_hdu = fits.PrimaryHDU(header=hdr)
    
    col1 = fits.Column(name='lam', format='D', array=wave)
    col2 = fits.Column(name='flux', format='D', array=spec)
    coldefs1 = fits.ColDefs([col1, col2])
    if error is not None:
        col3 = fits.Column(name='noise', format='D', array=error)
        coldefs1 = fits.ColDefs([col1, col2, col3])
    
    hdu1 = fits.BinTableHDU.from_columns(coldefs1)

    hdulis = fits.HDUList([primary_hdu, hdu1])
    hdulis.writeto(path,overwrite=True)
    
    print('spectrum saved to '+path)
    del hdu1,hdulis,primary_hdu


def read_file(f):
    
    hdu = fits.open(f)
    
    h = hdu[0].header
    data = hdu[0].data
    
    hdu.close()
    del hdu
    
    return h,data


def cut_orders(l,f,n):

    # J2225-0011
    '''dict_orders = {6:[9520,10200],
               7:[8150,9480],
               8:[7118,8290],
               9:[6440,7225],
               10:[5850,6505],
               11:[5175,6000],
               12:[4900,5440],
               13:[4500,5040],
               14:[4210,4670],
               15:[3930,4315],
               16:[3750,4070]}'''
    
    # J2215+0002
    '''dict_orders = {6:[9520,10200],
               7:[8150,9480],
               8:[7118,8290],
               9:[6326,7363],
               10:[5850,6505],
               11:[5175,6000],
               12:[4900,5440],
               13:[4500,5040],
               14:[4160,4717],
               15:[3930,4350],
               16:[3750,4070]}'''
    
    # J0240-0828
    '''dict_orders = {6:[9520,10200],
               7:[8150,9480],
               8:[7162,8290],
               9:[6331,7346],
               10:[5850,6505],
               11:[5212,6000],
               12:[4830,5502],
               13:[4500,5040],
               14:[4160,4730],
               15:[3930,4350],
               16:[3750,4070]}'''
    
    # J0021+0052
    '''dict_orders = {6:[9520,10200],
               7:[8150,9480],
               8:[7162,8290],
               9:[6331,7346],
               10:[5850,6505],
               11:[5212,6000],
               12:[4830,5524],
               13:[4480,5040],
               14:[4160,4667],
               15:[3930,4375],
               16:[3750,4120]}'''
    
    # J0950+0042
    '''dict_orders = {6:[9520,10350],
               7:[8150,9480],
               8:[7120,8290],
               9:[6331,7365],
               10:[5820,6590],
               11:[5212,6000],
               12:[4780,5524],
               13:[4645,5090],
               14:[4260,4712],
               15:[3050,4297],
               16:[3785,4000]}'''
    
    # J1146+0053
    '''dict_orders = {6:[9500,10200],
               7:[8150,9480],
               8:[7120,8290],
               9:[6460,7365],
               10:[5820,6590],
               11:[5212,6000],
               12:[4850,5524],
               13:[4480,5040],
               14:[4180,4685],
               15:[3920,4360],
               16:[3760,4100]}'''
    
    # J1444+0409
    '''dict_orders = {6:[9500,10200],
               7:[8150,9470],
               8:[7120,8290],
               9:[6400,7365],
               10:[5820,6590],
               11:[5180,6000],
               12:[4830,5470],
               13:[4480,5075],
               14:[4220,4685],
               15:[3920,4340],
               16:[3760,4060]}'''
    
    # J1448-0110
    '''dict_orders = {6:[9500,9990],
               7:[8150,9470],
               8:[7120,8290],
               9:[6340,7365],
               10:[5820,6590],
               11:[5316,5961],
               12:[4830,5470],
               13:[4445,5100],
               14:[4195,4685],
               15:[3895,4340],
               16:[3700,4095]}'''
    
    # J2101-0555
    '''dict_orders = {6:[9500,9990],
               7:[8150,9470],
               8:[7130,8270],
               9:[6535,7144],
               10:[5750,6590],
               11:[5180,6025],
               12:[4830,5470],
               13:[4445,4980],
               14:[4225,4685],
               15:[3950,4320],
               16:[3725,4030]}'''
    
    # J1226+0415
    '''dict_orders = {6:[9500,10050],
               7:[8150,9470],
               8:[7130,8270],
               9:[6370,7370],
               10:[5835,6590],
               11:[5230,5740],
               12:[4850,5525],
               13:[4475,4980],
               14:[4145,4685],
               15:[3980,4370],
               16:[3800,4090]}'''
    
    # J2119+0052
    dict_orders = {6:[9500,10050],
               7:[8150,9470],
               8:[7130,8270],
               9:[6430,7340],
               10:[5835,6590],
               11:[5400,5950],
               12:[4850,5525],
               13:[4475,5080],
               14:[4220,4685],
               15:[3940,4300],
               16:[3740,4055]}
    
    # J0136-0037
    '''dict_orders = {6:[9500,10375],
               7:[8150,9470],
               8:[7123,8270],
               9:[6535,7205],
               10:[5900,6590],
               11:[5230,5983],
               12:[4830,5470],
               13:[4495,5030],
               14:[4185,4665],
               15:[3925,4360],
               16:[3725,4030]}'''
    
    # J0252+0114
    '''dict_orders = {6:[9500,10550],
               7:[8150,9470],
               8:[7123,8270],
               9:[6423,7350],
               10:[5850,6590],
               11:[5400,5950],
               12:[4830,5470],
               13:[4456,5100],
               14:[4185,4665],
               15:[3945,4312],
               16:[3725,4030]}'''
    
    mask = (l>dict_orders[n][0]) & (l<dict_orders[n][1])
    
    lam1 = l[mask]
    flux1 = f[mask]
    
    return lam1,flux1


def rebin_spec(waves,lam_in,flux_in,dl):
    
    flux_out = np.full_like(waves, np.nan)
    for i in range(len(flux_out)):
        mask = (lam_in>waves[i]-dl/2) & (lam_in<waves[i]+dl/2)
        if len(flux_in[mask])>0:
            flux_out[i] = np.nanmean(flux_in[mask])
    return flux_out

def wNanAverage(fluxes, noises):
    flux = []
    noise = []
    tflux = np.transpose(fluxes)
    tnoise = np.transpose(noises)
    for i in range(len(tflux)):
        nanmask = np.where(~np.isnan(tflux[i]))[0]
        w = tnoise[i][nanmask]**-2
        if len(w)>0:
            flux.append(np.average(tflux[i][nanmask], weights=w))
            noise.append(np.sqrt(np.sum(w)**-1))
    return flux, noise
    


if __name__=='__main__':
    
    h,data = read_file(specfile)
    
    magespec = MageMultispec(data,h)
    
    lam1,flux1 = cut_orders(magespec.wavelength(7),magespec.flux(7),7)
    lam0,flux0 = cut_orders(magespec.wavelength(16),magespec.flux(16),16)
    
    dl = lam1[-1]-lam1[-2]
    waves = np.arange(lam0[0],lam1[-1]+dl,dl) # THIS IS THE WAVELENGTH I USE
    fluxes = []
    noises = []
    print(dl)

    for i in range(7,17):
        lam,flux = cut_orders(magespec.wavelength(i),magespec.flux(i),i)
        lam_n,noise = cut_orders(magespec.wavelength(i), magespec.noise(i), i)
        del lam_n
        print('dlambda : ' + str(lam[1]-lam[0]) + ' for order ' + str(i))
        dlam = lam[1]-lam[0]
        fluxes.append(rebin_spec(waves,lam,flux,dlam))
        noises.append(rebin_spec(waves,lam,noise,dlam))
        
    amerged_flux, amerged_noise = wNanAverage(fluxes, noises)
    #merged_flux = np.nanmean(fluxes,axis=0) # THIS IS THE MERGED FLUX I USE AS INPUT
    #merged_noise = np.nanmean(noises,axis=0)
    #median_norm = np.nanmedian(merged_flux) # THIS IS THE NORMALIZATION FACTOR
    savefldr = specfldr.replace('fits', 'merged')
    save_spectrum(specfile,waves,amerged_flux,savefldr+filename,error=amerged_noise)
