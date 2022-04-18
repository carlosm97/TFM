#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 18:01:21 2021

@author: carlos
"""
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

from numina.array.display.ximshow import ximshow
from numina.array.display.ximplotxy import ximplotxy
import os
from scipy import stats, optimize
import time
os.chdir('/home/carlos/Desktop/MSc/TFM/GRS_1915+105_EMIR')
drc=next(os.walk('/home/carlos/Desktop/MSc/TFM/GRS_1915+105_EMIR'))
GRS=drc[1]
GRS.remove('PROBLEMATIC')

def spectractor(f,l1m=None,l2m=None,remove=False):
    '''
    Function to extract the spectrum from the ABBA combined images. 
    
    PARAMETERS :
    ------------
    f : str, 
        Name of the folder whose OB we want to extract. 
        
    l1m & l2m : int, (optional)
        Left and right limit of the stectra in the middle. Deffault=None. In 
        this case, the program ask for the limit after plotting the mean spatial
        distribution.
        
    l1l & l2l : int, (optional)
        Left and right limit of the stectra in the middle. Deffault=None. In 
        this case, the program ask for the limit after plotting the mean spatial
        distribution.  
        
    remuve : bool, (optional)
        If True, it removes the existing spectra and write a new one. Deffault=False. 
    '''
    if remove==True:
        try: os.remove('spectra/'+f+'.fits')
        except: pass
    dt=os.listdir(f+'/data')
    frame = sum(map(lambda x : 'SPECTRA' in x , dt))/2
    with fits.open(f+'/obsid_abba_results/reduced_mos_abba_combined.fits', mode='readonly') as hdulist:
        image_headerK = hdulist[0].header
        image_dataK = hdulist[0].data
        try: error_dataK = hdulist[1].data
        except: pass
    plt.close('all')
    # image dimensions
    naxis2, naxis1 = image_dataK.shape
    
    # read wavelength calibration from header
    crpix1 = image_headerK['crpix1']
    crval1 = image_headerK['crval1']
    cdelt1 = image_headerK['cdelt1']
    exptime = image_headerK['exptime']
    print('crpix1, crval1, cdelt1:', crpix1, crval1, cdelt1)
    print('exptime:', exptime)
    
    #ximshow(image_dataK, image_bbox=(400,2400,1223,1242), title='GRS 1915+105')
    #ximshow(image_dataK, image_bbox=(400,2400,1170,1190), title='GRS 1915+105')
    
    # To select the integration limits to be considered object spectra, we can show
    # the sum along the spectral direction:
    if l1m==None or l2m==None:
        y_data = np.sum(image_dataK, axis=1)
        plt.figure('Example')
        plt.plot(y_data)
        plt.grid()
        #plt.draw()
        while plt.fignum_exists('Example')==True:
            plt.pause(2)
        if l1m==None or l2m==None:
            print('Middle limit=')
            l1m=int(input())
            l2m=int(input())   
    # extract and coadd individual spectra by coadding rows
    
    spm = np.sum(image_dataK[l1m:l2m,], axis=0)
    spectrumK = spm
    if np.mean(spm)<0:
        spectrumK = -spectrumK 
    spectrumK = spectrumK / (frame * exptime)   # I am adding a hole serie ABBA, wich 
    try:     
        err = np.sum(error_dataK[l1m:l2m,], axis=0)
        errorK = err
        if np.mean(err)<0:
            errorK = -errorK 
        errorK = errorK / (frame * exptime)   # I am adding a hole serie ABBA, wich 
    except: pass
    # has 4 exposures. 
    # plot spectrum
    fig, ax1= plt.subplots()
    waveK = crval1 + (np.arange(1, naxis1 + 1) - crpix1) * cdelt1
    print(waveK[0],waveK[-1])
    ax1.plot(waveK, spectrumK)
    wvminJ = waveK[spectrumK>0][0]
    wvmaxJ = waveK[spectrumK>0][-1]
    ax1.set_xlim([wvminJ, wvmaxJ])
    ax1.set_xlabel('wavelength (Angstrom)')
    ax1.set_ylabel('count rate (ADU/s)')
    ax1.set_title('GRS 1915+105 (grism K), '+f[10:13]+', OB'+f[-4:])
    ax1.grid()
    ax1.set_ylim(0,np.mean(spectrumK)+1.65*np.std(spectrumK))
    plt.show()
    plt.savefig('spectra/'+str(f)+'_spectra.jpeg')
    plt.show()   
    # Creating fits with the spectrum. 
    #col1 = fits.Column(name='Wavelength', format='E', array=waveK)
    #col2 = fits.Column(name='Intensity', format='E', array=spectrumK)
    #table_hdu = fits.BinTableHDU.from_columns([col1, col2])
    hdu_sp = fits.PrimaryHDU(data=spectrumK,header=image_headerK)
    try: 
        hdu_err = fits.ImageHDU(data=errorK,header=image_headerK)
        hdulFinal = fits.HDUList([hdu_sp,hdu_err])
    except: 
        hdulFinal = fits.HDUList([hdu_sp])
    #hdul = fits.HDUList([header, table_hdu])
    hdulFinal.writeto('spectra/'+f+'.fits')
    return None
# In[]
spectractor('GRS1915_K_18A_01_OB0001',1119,1136,remove=True)
spectractor('GRS1915_K_18A_02_OB0003',1176,1190,remove=True)
spectractor('GRS1915_K_18A_03_OB0005',1167,1185,remove=True)
spectractor('GRS1915_K_18A_04_OB0007',1175,1190,remove=True)
spectractor('GRS1915_K_18A_05_OB0001',1173,1189,remove=True)
spectractor('GRS1915_K_18A_06_OB0003',1173,1188,remove=True)
spectractor('GRS1915_K_18A_07_OB0005',1169,1190,remove=True)
#spectractor('GRS1915_K_18A_08_OB0007',)
spectractor('GRS1915_K_18A_09_OB0009',1179,1193,remove=True)
spectractor('GRS1915_K_18A_10_OB0011',1175,1195,remove=True)
spectractor('GRS1915_K_21A_01_OB0073',1175,1187,remove=True)
spectractor('GRS1915_K_21A_02_OB0074',1170,1188,remove=True)
spectractor('GRS1915_K_21A_03_OB0075',1118,1132,remove=True)
spectractor('GRS1915_K_21A_04_OB0076',1222,1241,remove=True)
spectractor('GRS1915_K_21A_05_OB0078',1173,1188,remove=True)
plt.close('all')
# In[read_and_plot]
def read_plot(fit):
    '''
    Function to read the spectra files. 

    Parameters
    ----------
    fit : str,
        Name of the fits file to be read.

    Returns
    -------
    data : astropy.io.fits.fitsrec.FITS_rec,
        Table in dictionary-like. Wavelength=data['Wavelength'], Intensity=data['Intensity']
    header : astropy.io.fits.header.Header,
        Header of the fits file. 

    '''
    with fits.open('spectra/'+fit, mode='readonly') as hdulist:
        header = hdulist[0].header
        data = hdulist[0].data
    #wv,I=data['Wavelength'],data['Intensity']
    I = data
    naxis1 = len(data)
    crpix1 = header['crpix1']
    crval1 = header['crval1']
    cdelt1 = header['cdelt1']   
    wv = crval1 + (np.arange(1, naxis1 + 1) - crpix1) * cdelt1
    plt.plot(wv,I)
    plt.grid()
    wvminJ = wv[I>0][0]
    wvmaxJ = wv[I>0][-1]
    plt.xlim([wvminJ, wvmaxJ])   
    plt.xlabel('wavelength (Angstrom)')
    plt.ylabel('count rate (ADU/s)') 
    plt.title(fit[:-5])
    return data,header
# Example
rtn=read_plot('GRS1915_K_21A_05_OB0078.fits')

