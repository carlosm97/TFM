#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 11:12:50 2021

@author: Martínez-Sebastián

GOAL : Have all the functions used to do photometry
"""
import matplotlib.pyplot as plt
import numpy as np
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
import os
from astropy.io import fits
import pandas as pd
from scipy.interpolate import griddata
from astropy.wcs import WCS

def radius_selection(dat, positions,rad=3,r_sep=3,verbose=False,fix=False,INVERSE=False):
    # Hay que poner la mattriz de background como un parametro. 
    '''
    Function to select the optimous radious to integrate the star flux. It does 
    an iterative process stopped when the increment of the flux of the star is
    greater than the error associated to the measurement. 
     The error is stimated as random noise of the source, random noise of the 
    sky, and an annulus around the star in the background-substracted image. 
    
    Parameters
    ----------
    dat : ndarray_like,
        Background substracted data. 
        
    positions : list,
        Central position of the star to be studied. 
        
    rad : float, optional
        Radius to start the iteration with. The default is 3.
    
    r_sep : float, optional
        Separation between the star radious and the radios of the inner cirle 
        of the annulus. The default is 3.
        
    verbose : bool, optional
        If True, the different iterations print the result of the increment from
        the last iteration to the new and the error of the masurement. The 
        default is False.

    Returns
    -------
    rad : int,
        Radius fitted in the loop to integrate the star flux.
        
    dif : float
        Counts of the star integrated and background substracted.
        
    aperture : photutils.aperture.circle.CircularAperture,
        Circular apperture around the star. The integrated counts are those inside
        the apperture. 
        
    err : float,
        Error associated to the star measurement.
    '''
    
    err = 0
    incr = 100
    source_flux = 0
    dif_err = 0
    if INVERSE==False:
        x,y = int(positions[0][1]),int(positions[0][0])
        x_old,y_old=0,0
        while x_old!=x or y_old!=y:
            posit_new = np.where(dat==np.max(dat[x-1:x+2,y-1:y+2]))
            x_old,y_old = x,y
            y,x = int(posit_new[1][0]),int(posit_new[0][0])
    elif INVERSE==True:
        x,y = int(positions[0][0]),int(positions[0][1])
        x_old,y_old=0,0
        while x_old!=x or y_old!=y:
            posit_new = np.where(dat==np.max(dat[x-1:x+2,y-1:y+2]))
            x_old,y_old = x,y
            y,x = int(posit_new[0][0]),int(posit_new[1][0])
    positions = [(y,x)]
    #while abs(dif_old-dif)>0.05*dif:
    if fix==False:
        while incr>20*dif_err and incr>5*np.sqrt(source_flux): #and incr<incr_old:
            # Current radius selection criteria: the increment of considering a radius 
            # one pixel grater should be greater than the error associated to the measurement. 
            source_old = source_flux
            err_old = err
            rad+=1
            rin = rad+r_sep
            aperture = CircularAperture(positions, r=rad)
            r_an = np.sqrt((r_sep+rad)**2+rad**2)-(r_sep+rad)
            annulus_aperture = CircularAnnulus(positions, r_in=rin, r_out=rin+r_an)
            phot_table = aperture_photometry(dat, aperture)
            phot_table['aperture_sum'].info.format = '%.8g'  # for consistent table output
            annulus_masks = annulus_aperture.to_mask(method='center')
            annulus_data = annulus_masks[0].multiply(dat)
            median = np.median(annulus_data[annulus_data>1])
            sky = median*np.pi*rad**2
            source_flux = phot_table[0][3]-sky
            err = np.sqrt(phot_table[0][3])
            dif_err = np.sqrt(err**2-err_old**2)
            incr = source_flux-source_old
            if verbose==True:
                print(incr,err)

    elif fix==True:
        rin = rad+r_sep
        aperture = CircularAperture(positions, r=rad)
        r_an = np.sqrt((r_sep+rad)**2+rad**2)-(r_sep+rad)
        annulus_aperture = CircularAnnulus(positions, r_in=rin, r_out=rin+r_an)
        phot_table = aperture_photometry(dat, aperture)
        phot_table['aperture_sum'].info.format = '%.8g'  # for consistent table output
        annulus_masks = annulus_aperture.to_mask(method='center')
        annulus_data = annulus_masks[0].multiply(dat)
        median = np.median(annulus_data[annulus_data>1])
        sky = median*np.pi*rad**2        
        err = np.sqrt(phot_table[0][3])
        source_flux = phot_table[0][3]-sky
        #print(phot_table[0][3],phot_table_an[0][3],dif)
    if verbose==True:
        print('Selected radius:',rad, 'Star electrons: ',phot_table[0][3], 'Sky electrons: ',sky)
        return rad, source_flux, aperture, err, annulus_aperture,median
    #plt.figure()
    #plt.imshow(dat)
    #aperture.plot(color='white', lw=2,label='Photometry aperture')
    #print('SNR = ',dif/err)
    else:
        return rad, source_flux, aperture, err, annulus_aperture#,phot_table_an[0][3]

def Photometry(dat, positions, magnitudes, obj_name, wcs,verbose=False,PLOT = False): 
    '''
    Function to do differential photometry of the first object. The differential
    photometry of the reference objects is done too in order to have a reference.

    Parameters
    ----------
    dat : ndarray,
        array of the data already corrected from flat.     
    
    positions : Tuple,
        Tupple containing tupples of positions.
        
    magnitudes : Tuple,
        Bibliographic magnitudes of reference stars. There is one less magnitude 
        than objects. The magnitude of the first object is not given, in order 
        to be computed from the other star magnitudes. 
        A comprobation is done by computing the diferential photometry of all 
        objects with all (excepting the  first). 
    
    obj_name : Tubple, 
        Tuple with names of the objects. 

    Returns
    -------
        None, but a plot with marked objects with the radii selected for the
        integration
        

    '''
    # As we want to work in electrons, not ADUs, we convert them with the value
    # gain=3.3 e-/ADU:
    dat = 3.3*dat
    flux, dflux, ap, sky = [], [], [], []
    r_min = np.inf
    for posit in positions :        
        RS = radius_selection(dat, posit, rad=1,r_sep=5)
        # After selecting the radious for all the stars, we choose the minimum of them all
        if r_min > RS[0]:
            r_min = RS[0]
    for posit in positions :
        rs = radius_selection(dat, posit, rad=r_min,r_sep=8,fix=True,verbose=True)
        flux.append(rs[1])
        dflux.append(rs[3])
        ap.append([rs[2],rs[4]])
        sky.append(rs[-1])
        del rs
    M = []
    dM = []
    for p in range(len(flux)):
        st_p,sdm = [], []
        if p>1:
            for i in range(1,p):
                m = magnitudes[i-1]-2.5*np.log10(flux[p]/flux[i])
                dm = abs(-2.5*(1/flux[p]/np.log(10))*dflux[p])+abs(2.5*(1/flux[i]/np.log(10))*dflux[i])
                st_p.append(m), sdm.append([obj_name[i],dm])
        for i in range(p+1,len(flux)):
            m = magnitudes[i-1]-2.5*np.log10(flux[p]/flux[i])
            dm = abs(-2.5*(1/flux[p]/np.log(10))*dflux[p])+abs(2.5*(1/flux[i]/np.log(10))*dflux[i])
            st_p.append(m), sdm.append([obj_name[i],dm])
        if verbose==True:
            if p==0:
                print(obj_name[p],'=',st_p)
            else:
                print(obj_name[p],'=',st_p,'Bibliography: ',magnitudes[p-1])
        M.append(st_p), dM.append(sdm)
        del st_p,sdm  
    if PLOT==True:
        fig = plt.figure()
        fig.add_subplot(111,projection=wcs)
        plt.grid(color='Grey')
        im1 = plt.imshow(dat,cmap='Greys_r',origin='lower',vmin=np.nanmedian(dat),vmax=2*np.nanmedian(dat))
        #im1 = plt.imshow(data,cmap='Greys_r',origin='lower',vmin=np.nanmedian(data),vmax=2*np.nanmedian(data))
        plt.colorbar(im1)
        plt.xlabel('x'),plt.ylabel('y')
        for i in range(len(ap)):
            ap[i][1].plot(color='blue', lw=2)
            ap[i][0].plot(color='orange', lw=2)
            plt.text(positions[i][0][0],positions[i][0][1],obj_name[i],c='red')
    return M, dM, r_min, sky
   

def Forced_Photometry(dat, rad, positions, magnitudes, obj_name, wcs,verbose=False,PLOT = False,inverse=False): 
    '''
    Function to do differential photometry of the first object. The differential
    photometry of the reference objects is done too in order to have a reference.

    Parameters
    ----------
    dat : ndarray,
        array of the data already corrected from flat.     
    
    rad : float, 
        Radius around the centre which define the apperture. 
    
    positions : Tuple,
        Tupple containing tupples of positions.
        
    magnitudes : Tuple,
        Bibliographic magnitudes of reference stars. There is one less magnitude 
        than objects. The magnitude of the first object is not given, in order 
        to be computed from the other star magnitudes. 
        A comprobation is done by computing the diferential photometry of all 
        objects with all (excepting the  first). 
    
    obj_name : Tubple, 
        Tuple with names of the objects. 

    Returns
    -------
        None, but a plot with marked objects with the radii selected for the
        integration
        

    '''
    # As we want to work in electrons, not ADUs, we convert them with the value
    # GAIN=3.3 e-/ADU:
    dat = 3.3*dat
    flux, dflux, ap, sky = [], [], [], []
    r_min = rad
    for posit in positions :
        rs = radius_selection(dat, posit, rad=r_min,r_sep=4,fix=True,verbose=True,INVERSE=inverse)
        flux.append(rs[1])
        dflux.append(rs[3])
        ap.append([rs[2],rs[4]])
        sky.append(rs[-1])
        del rs
    M = []
    dM = []
    for p in range(len(flux)):
        st_p,sdm = [], []
        if p>1:
            for i in range(1,p):
                m = magnitudes[i-1]-2.5*np.log10(flux[p]/flux[i])
                dm = abs(-2.5*(1/flux[p]/np.log(10))*dflux[p])+abs(2.5*(1/flux[i]/np.log(10))*dflux[i])
                st_p.append(m), sdm.append(dm) 
        for i in range(p+1,len(flux)):
            m = magnitudes[i-1]-2.5*np.log10(flux[p]/flux[i])
            dm = abs(-2.5*(1/flux[p]/np.log(10))*dflux[p])+abs(2.5*(1/flux[i]/np.log(10))*dflux[i])
            st_p.append(m), sdm.append(dm)
        if verbose==True:
            if p==0:
                print(obj_name[p],'=',st_p)
            else:
                print(obj_name[p],'=',st_p,'Bibliography: ',magnitudes[p-1])
        M.append(st_p), dM.append(sdm)
        del st_p  
    if PLOT==True:
        fig = plt.figure()
        fig.add_subplot(111,projection=wcs)
        plt.grid(color='Grey')
        im1 = plt.imshow(dat,cmap='Greys_r',origin='lower',vmin=np.nanmedian(dat[650:1450,650:1450]),vmax=2*np.nanmedian(dat[650:1450,650:1450]))
        #im1 = plt.imshow(data,cmap='Greys_r',origin='lower',vmin=np.nanmedian(data),vmax=2*np.nanmedian(data))
        plt.colorbar(im1)
        plt.xlabel('x'),plt.ylabel('y')
        for i in range(len(ap)):
            ap[i][1].plot(color='blue', lw=2)
            ap[i][0].plot(color='orange', lw=2)
            plt.text(positions[i][0][0],positions[i][0][1],obj_name[i],c='red')
    return M, dM, sky


def plot_photometry(dat, rad, positions, obj_name, wcs, a=None, b=None):
    ap = []
    r_min = rad
    for posit in positions :
        rs = radius_selection(dat, posit, rad=r_min,r_sep=4,fix=True,verbose=True)
        ap.append([rs[2],rs[4]])
    fig = plt.figure()
    fig.add_subplot(111,projection=wcs)
    posit = np.vstack(positions)
    if a==None and b==None:
        plt.xlim(min(posit[:,0])-200,max(posit[:,0])+200),plt.ylim(min(posit[:,1])-200,max(posit[:,1])+200)
    else:
        plt.xlim(a),plt.ylim(b)
    plt.grid(color='Grey')
    im1 = plt.imshow(dat,cmap='Greys_r',origin='lower',vmin=np.nanmedian(dat[650:1450,650:1450]),vmax=2*np.nanmedian(dat[650:1450,650:1450]))
    #im1 = plt.imshow(data,cmap='Greys_r',origin='lower',vmin=np.nanmedian(data),vmax=2*np.nanmedian(data))
    plt.colorbar(im1)
    plt.xlabel('x'),plt.ylabel('y')
    for i in range(len(ap)):
        ap[i][0].plot(color='orange', lw=2)
        plt.text(positions[i][0][0]+5,positions[i][0][1]+5,obj_name,c='red')
    plt.show()
    return None

if __name__=='__main__':
    plot_photometry(dat, radio/2, positions, names, prj)



def sel_im(path):
    '''
    Function to find the images in a given directory. 

    Parameters
    ----------
    path : str,
        Path to the directory to be analized.

    Returns
    -------
    image : list,
        List with the path & name to the images.
    '''
    lst = os.listdir(path)
    image = []
    for im in lst:
        if 'IMAGE' in im:
            image.append(path+'/'+im)
    return image
def clear_image(path, FF, projection=False):
    im = fits.open(path)
    dat = im[0].data/FF 
    wcs = WCS(im[0].header)
    air_mass = np.copy(im[0].header['AIRMASS'])
    hour = np.copy(im[0].header['DATE-OBS'])
    dat[dat==np.inf]=0
    dat[pd.isna(dat)]=0
    # dat[dat>4.4e4]=0
    X,Y = np.meshgrid(np.arange(2048), np.arange(2048))
    interpolated = griddata(np.where(dat!=0),dat[dat!=0],(X,Y),method='nearest')   
    im.close()
    print(air_mass)
    del dat, im
    if projection==False:
        return interpolated,air_mass
    if projection==True:
        return interpolated, wcs, air_mass, hour

def plotting_multiple_stars(DATE,photometry_list,name_list,color_list,error=0,dif=False):    
    fig, axs = plt.subplots(len(photometry_list),sharex=True,sharey=True)
    axs[0].invert_yaxis()
    i = 0
    for ph in photometry_list:
        for j in range(len(ph[0,:])):
            if dif==True:
                if i==0:
                    axs[i].errorbar(DATE,ph[:,j]-np.nanmean(ph[:,j]),fmt='o',c=color_list[j],label=name_list[j+1],yerr=error)
                    axs[i].legend()
                    axs[i].set_ylabel(name_list[0])
                elif i>0 and i-1>j:
                    axs[i].errorbar(DATE,ph[:,j]-np.nanmean(ph[:,j]),fmt='o',c=color_list[j],yerr=error)   
                elif i>0 and i-1<=j:
                    axs[i].errorbar(DATE,ph[:,j]-np.nanmean(ph[:,j]),fmt='o',c=color_list[j+1],yerr=error) 
                com=sorted(zip(DATE,np.nanmean(ph,1)-np.nanmean(ph)))
            elif dif==False:
                if i==0:
                    axs[i].errorbar(DATE,ph[:,j],fmt='o',c=color_list[j],label=name_list[j+1],yerr=error)
                    axs[i].legend()
                    axs[i].set_ylabel(name_list[0])
                elif i>0 and i-1>j:
                    axs[i].errorbar(DATE,ph[:,j],fmt='o',c=color_list[j],yerr=error)   
                elif i>0 and i-1<=j:
                    axs[i].errorbar(DATE,ph[:,j],fmt='o',c=color_list[j+1],yerr=error)       
                com=sorted(zip(DATE,np.nanmean(ph,1)))
        axs[i].grid()
        A,B=[],[]
        for j in range(len(com)):
            A.append(com[j][0]),B.append(com[j][1])
        axs[i].plot(A,B,'k')
        if i>0:
            axs[i].set_ylabel(color_list[i-1])
        i+=1
    fig.text(0.5, 0.04, 'MJD', ha='center')
    fig.text(0.04, 0.5, r'$\Delta$m', va='center', rotation='vertical')



#%% Ensayo y error 
'''
#dat, prj, airmass = clear_image(Im_list[0],FF_clean,projection=True)
# plt.figure()
# plt.imshow(dat,cmap='Greys_r',origin='lower',vmin=np.nanmedian(dat),vmax=2*np.nanmedian(dat))
# plt.colorbar()

posit = [(1214.,1330.)]

posit_19151209_1056272 = [(1299.,1384.)]
m_19151209_1056272 = 13.518

posit_19151020_1057011 = [(1083,1315.)]
m_19151020_1057011 = 14.262

posit_19151267_1057031 = [(1235.,1204.)]
m_19151267_1057031 = 14.408

posit_19151313_1057000 = [(1272.,1199.)]
m_19151313_1057000 = 13.202

pos = posit, posit_19151209_1056272, posit_19151020_1057011, posit_19151267_1057031, posit_19151313_1057000 
mag = m_19151209_1056272, m_19151020_1057011, m_19151267_1057031, m_19151313_1057000

names = 'GRS1915+105','19151209+1056272','19151020+1057011','19151267+1057031','19151313+1057000'
    
#def radius_selection(dat, positions,rad=3,r_sep=3,verbose=False,fix=False):
    # Hay que poner la mattriz de background como un parametro. 


#Photometry(dat, pos, mag, names, prj)


positions = posit
rad = 3
r_sep= 0
verbose, fix = False, False
source_flux = 0
incr = 100
err = 0
dif_err = 0


x,y = int(positions[0][1]),int(positions[0][0])
x_old,y_old=0,0
while x_old!=x or y_old!=y:
    posit_new = np.where(dat==np.max(dat[x-1:x+2,y-1:y+2]))
    x_old,y_old = x,y
    y,x = int(posit_new[1][0]),int(posit_new[0][0])
#while abs(dif_old-dif)>0.05*dif:
positions = [(y,x)]
while incr>5*dif_err: #and incr<incr_old:
    # Current radius selection criteria: the increment of considering a radius 
    # one pixel grater should be greater than the error associated to the measurement. 
    dif_old = source_flux
    err_old = err
    rad+=1
    rin = rad+r_sep
    aperture = CircularAperture(positions, r=rad)
    r_an = np.sqrt((r_sep+rad)**2+rad**2)-(r_sep+rad)
    annulus_aperture = CircularAnnulus(positions, r_in=rin, r_out=rin+r_an)
    phot_table = aperture_photometry(dat, aperture)
    phot_table['aperture_sum'].info.format = '%.8g'  # for consistent table output
    annulus_masks = annulus_aperture.to_mask(method='center')
    annulus_data = annulus_masks[0].multiply(dat)
    median = np.median(annulus_data[annulus_data>1])
    sky = median*np.pi*rad**2
    source_flux = phot_table[0][3]-sky
    err = np.sqrt(phot_table[0][3])
    dif_err = np.sqrt(err**2-err_old**2)
    incr = source_flux-dif_old
    if verbose==True:
        print(incr,err)

plt.imshow(dat,vmin=2.2e4,vmax=4e4,cmap='Greys_r')
aperture[0].plot(color = 'white')


# Typical seeing ~3 pixels. 
print('Selected radius:',rad, 'Star electrons: ',phot_table[0][3], 'Sky electrons: ',sky)
#plt.figure()
#plt.imshow(dat)
#aperture.plot(color='white', lw=2,label='Photometry aperture')
#print('SNR = ',dif/err)
#print(rad, dif, aperture, err, annulus_aperture)#,phot_table_an[0][3]




'''
    
    
    
    
    
    
    
    
    
    
    
    