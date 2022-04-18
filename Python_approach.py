#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 20:14:33 2022

@author: carlos
"""
import matplotlib.pyplot as plt
import matplotlib 
import numpy as np
import os
from astropy.io import fits
import pandas as pd
from astropy.wcs import WCS
from trm import molly
from astropy.time import Time
import pylab
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import re
import progressbar
import time

def vel_redshit(lam):
    c_km_s_1 = 299792.458
    return (lam - Brgamma)*c_km_s_1/Brgamma
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]
def interp_error(X,p_1,p_2):
    return MAXI['e_2_20keV'][p_1]+(X-MAXI['#MJD'][p_1])/(MAXI['#MJD'][p_2]-MAXI['#MJD'][p_1])*(MAXI['e_2_20keV'][p_2]-MAXI['e_2_20keV'][p_1])
def interp(v,vec,p):
    if v<vec:
        x, y = [MAXI['#MJD'][p],MAXI['#MJD'][p+1]],[MAXI['2_20keV'][p],MAXI['2_20keV'][p+1]]
        p_1,p_2 = p, p+1
        
    elif v>vec:
        x, y = [MAXI['#MJD'][p-1],MAXI['#MJD'][p]],[MAXI['2_20keV'][p-1],MAXI['2_20keV'][p]]
        p_1,p_2 = p-1, p
    f = interp1d(x,y)
    fe = interp_error(vec,p_1,p_2)
    int1 =  f(vec)
    return int1,fe
def double_Gauss(x,A1,A2,m1,m2,s1,s2):
    y = 1+A1*np.exp(-1*(x-m1)**2/(2*s1**2))+A2*np.exp(-1*(x-m2)**2/(2*s2**2))
    return y    
def double_peak(x,A1,A2,A_broad,m1,m2,m_broad,s,s_broad):
    y = 1+A1*np.exp(-1*(x-m1)**2/(2*s**2))+A2*np.exp(-1*(x-m2)**2/(2*s**2))+A_broad*np.exp(-1*(x-m_broad)**2/(2*s_broad**2))
    return y   
def Gauss(x,A,m,s):
    y = 1+A*np.exp(-1*(x-m)**2/(2*s**2))
    return y   

Brgamma = 21661
HeII = 21885
HeI = 21126

# Mean spectra index for the first case : 14=average, 15=2017, 16=2018, 17=2021
mean = [14,15,16,17]

path_img = '/home/carlos/Desktop/MSc/TFM/Reduced_spectra/MolecfitCorrected-20220209T161944Z-001/MolecfitCorrected/Figures/'

os.chdir("/home/carlos/Desktop/MSc/TFM/Reduced_spectra/MolecfitCorrected-20220209T161944Z-001/MolecfitCorrected/")
Br_gamma_b2 = molly.rmolly('GRS1915_BrGamma_B2_interpolated.mol')

#%%
# Getting K in-band flux.
os.chdir('/home/carlos/Desktop/MSc/TFM/GRS_1915+105_EMIR')

#flux_17 = pd.read_csv('flux_Jy_GRS1915_17.txt', delimiter = "\\s+",header = 0,index_col=False,names = ['MJD','EW','dEW'])
flux_17 = pd.read_csv('flux_Jy_GRS1915_17.txt',index_col=False)
f17 = flux_17.columns
F_17 = np.zeros([int(len(f17)/4),4])
flux_21 = pd.read_csv('flux_Jy_GRS1915_21.txt',index_col=False)
f21 = flux_21.columns
F_21 = np.zeros([int(len(f21)/4),4])
for f in range(len(f17)):
    F_17[int(f/4),f%4] = float(re.findall("\d+\.\d+",f17[f])[0])
for f in range(len(f21)):
    F_21[int(f/4),f%4] = float(re.findall("\d+\.\d+",f21[f])[0])   
F_17, F_21 = np.sort(F_17,axis=0), np.sort(F_21,axis=0)
# Getting the average K flux for a given day: 
F17_indiv, F17_days = [], []
F17_mean = []
F17_indiv.append(F_17[0,1:]),F17_days.append(F_17[0,0])
for i in range(1,len(F_17)):
    if F_17[i,0]-F_17[i-1,0]<0.5:
        F17_indiv.append(F_17[i,1:]), F17_days.append(F_17[i,0])
    else: 
        F17_mean.append([np.mean(F17_days),np.std(F17_days),np.mean(F17_indiv),np.std(F17_indiv)])
        F17_indiv,F17_days = [],[]
        F17_indiv.append(F_17[i,1:])
        F17_days.append(F_17[i,0])
F17_mean.append([np.mean(F17_days),np.std(F17_days),np.mean(F17_indiv),np.std(F17_indiv)])
F17_mean = np.vstack(F17_mean)

F21_indiv, F21_days = [], []
F21_mean = []
F21_indiv.append(F_21[0,1:]),F21_days.append(F_21[0,0])
for i in range(1,len(F_21)):
    if F_21[i,0]-F_21[i-1,0]<0.5:
        F21_indiv.append(F_21[i,1:]), F21_days.append(F_21[i,0])
    else: 
        F21_mean.append([np.mean(F21_days),np.std(F21_days),np.mean(F21_indiv),np.std(F21_indiv)])
        F21_indiv,F21_days = [],[]
        F21_indiv.append(F_21[i,1:])
        F21_days.append(F_21[i,0])
F21_mean.append([np.mean(F21_days),np.std(F21_days),np.mean(F21_indiv),np.std(F21_indiv)])
F21_mean = np.vstack(F21_mean)
'''
#%%
####### Plotting spetra with different binnings and marked lines

os.chdir("/home/carlos/Desktop/MSc/TFM/Reduced_spectra/MolecfitCorrected-20220209T161944Z-001/MolecfitCorrected/GROUPS")

directory = os.listdir()
lsdir = []
for i in directory:
    if ".mol" in i:
        lsdir.append(i)
#specs_17, specs_18, specs_21 = molly.rmolly('GRS1915_K_17_ALL.mol'),molly.rmolly('GRS1915_K_18_ALL.mol')molly.rmolly('GRS1915_K_21_ALL.mol')

#plt.plot(specs.wave,specs.f)
#plt.plot(specs[0].wave,specs[0].f)


os.chdir("/home/carlos/Desktop/MSc/TFM/Reduced_spectra/MolecfitCorrected-20220209T161944Z-001/MolecfitCorrected/")
        
Br_gamma_b1 = molly.rmolly('GRS1915_BrGamma_ALL.mol')
Br_gamma_b2 = molly.rmolly('GRS1915_BrGamma_B2.mol')
Br_gamma_b3 = molly.rmolly('GRS1915_BrGamma_B3.mol')
Br_gamma_b4 = molly.rmolly('GRS1915_BrGamma_B4.mol')
Br_gamma_b5 = molly.rmolly('GRS1915_BrGamma_B5.mol')

plt.close('all')
fig, axs = plt.subplots(5,sharex=True)
for i in range(0,14):
    axs[0].step(Br_gamma_b1[i].wave,Br_gamma_b1[i].f,alpha=0.3)
    axs[1].step(Br_gamma_b2[i].wave,Br_gamma_b2[i].f,alpha=0.3)
    axs[2].step(Br_gamma_b3[i].wave,Br_gamma_b3[i].f,alpha=0.3)
    axs[3].step(Br_gamma_b4[i].wave,Br_gamma_b4[i].f,alpha=0.3)
    axs[4].step(Br_gamma_b5[i].wave,Br_gamma_b5[i].f,alpha=0.3)
color = ['b','r','m']
lab= ['mean 17','mean 18','mean 21']
for i in range(15,18):
    axs[0].step(Br_gamma_b1[i].wave,Br_gamma_b1[i].f,c=color[i-15],label=lab[i-15],linewidth=1)
    axs[1].step(Br_gamma_b2[i].wave,Br_gamma_b2[i].f,c=color[i-15])
    axs[2].step(Br_gamma_b3[i].wave,Br_gamma_b3[i].f,c=color[i-15])
    axs[3].step(Br_gamma_b4[i].wave,Br_gamma_b4[i].f,c=color[i-15])
    axs[4].step(Br_gamma_b5[i].wave,Br_gamma_b5[i].f,c=color[i-15])
    
axs[0].step(Br_gamma_b1[14].wave,Br_gamma_b1[14].f,'k',linewidth=2,label='Total mean')
axs[0].axvline(Brgamma,c='r',markersize = 5),axs[0].axvline(HeII,c='g',markersize = 5),axs[0].axvline(HeI,c='g',markersize = 5)
axs[1].step(Br_gamma_b2[14].wave,Br_gamma_b2[14].f,'k',linewidth=2,label='Total mean')
axs[1].axvline(Brgamma,c='r',markersize = 5),axs[1].axvline(HeII,c='g',markersize = 5),axs[1].axvline(HeI,c='g',markersize = 5)
axs[2].step(Br_gamma_b3[14].wave,Br_gamma_b3[14].f,'k',linewidth=2,label='Total mean')
axs[2].axvline(Brgamma,c='r',markersize = 5),axs[2].axvline(HeII,c='g',markersize = 5),axs[2].axvline(HeI,c='g',markersize = 5)
axs[3].step(Br_gamma_b4[14].wave,Br_gamma_b4[14].f,'k',linewidth=2,label='Total mean')
axs[3].axvline(Brgamma,c='r',markersize = 5),axs[3].axvline(HeII,c='g',markersize = 5),axs[3].axvline(HeI,c='g',markersize = 5)
axs[4].step(Br_gamma_b5[14].wave,Br_gamma_b5[14].f,'k',linewidth=2,label='Total mean')
axs[4].axvline(Brgamma,c='r',markersize = 5),axs[4].axvline(HeII,c='g',markersize = 5),axs[4].axvline(HeI,c='g',markersize = 5)

axs[0].text(Brgamma-20,1.21,r'Br$\gamma$',fontsize=14,c='r'),axs[0].text(HeII-20,1.21,r'He II (21885)',fontsize=14,c='g'),axs[0].text(HeI-20,1.21,r'He I (21126)',fontsize=14,c='g')
plt.xlabel(r'Wavelength ($\AA$)')
axs[0].set_ylim(0.9,1.2),axs[1].set_ylim(0.9,1.2),axs[2].set_ylim(0.9,1.2),axs[3].set_ylim(0.9,1.2),axs[4].set_ylim(0.9,1.2)
axs[0].grid(),axs[1].grid(),axs[2].grid(),axs[3].grid(),axs[4].grid()
plt.suptitle('Binning (pixels)')
axs[0].set_ylabel(r'Binning = 1'),axs[1].set_ylabel(r'Binning = 2'),axs[2].set_ylabel(r'Binning = 3'),axs[3].set_ylabel(r'Binning = 4'),axs[4].set_ylabel(r'Binning = 5')
axs[0].legend()


# In[Group Study]
#### Study of the groups (independently) with different binnings 
Br_gamma_b1[0].head['HJD']

[Br_gamma_b1[0].head['Year']]

d2017, d2018, d2021, mean = [],[],[],[]
for i in range(0,len(Br_gamma_b1)):
    if i>=len(Br_gamma_b1)-4: # Because I know that the last 4 spectra are the mean.
        mean.append(i)
    elif Br_gamma_b1[i].head['Year']==2017:
        d2017.append(i)
    elif Br_gamma_b1[i].head['Year']==2018:
        d2018.append(i)
    else:
        d2021.append(i)

plt.close('all')
fig1, ax1 = plt.subplots(3,sharex=True)
fig2, ax2 = plt.subplots(3,sharex=True)
fig3, ax3 = plt.subplots(3,sharex=True)
ax1_1,ax1_2,ax1_3 = ax1[0],ax1[1],ax1[2]
ax2_1,ax2_2,ax2_3 = ax2[0],ax2[1],ax2[2]
ax3_1,ax3_2,ax3_3 = ax3[0],ax3[1],ax3[2]

Brgamma = 21661
HeII = 21885
HeI = 21126

#-------------------------------------2017-------------------------------------
for i in d2017:
    ax1_1.step(Br_gamma_b1[i].wave,Br_gamma_b1[i].f,alpha=.5)
    ax2_1.step(Br_gamma_b2[i].wave,Br_gamma_b2[i].f,alpha=.5)
    ax3_1.step(Br_gamma_b3[i].wave,Br_gamma_b3[i].f,alpha=.5)
ax1_1.step(Br_gamma_b1[mean[1]].wave,Br_gamma_b1[mean[1]].f,c='r',label='Group average')
ax2_1.step(Br_gamma_b2[mean[1]].wave,Br_gamma_b2[mean[1]].f,c='r',label='Group average')
ax3_1.step(Br_gamma_b3[mean[1]].wave,Br_gamma_b3[mean[1]].f,c='r',label='Group average')

ax1_1.step(Br_gamma_b1[mean[0]].wave,Br_gamma_b1[mean[0]].f,c='k',label='Total average')
ax2_1.step(Br_gamma_b2[mean[0]].wave,Br_gamma_b2[mean[0]].f,c='k',label='Total average')
ax3_1.step(Br_gamma_b3[mean[0]].wave,Br_gamma_b3[mean[0]].f,c='k',label='Total average')
#-------------------------------------2018-------------------------------------
for i in d2018:
    ax1_2.step(Br_gamma_b1[i].wave,Br_gamma_b1[i].f,alpha=.5)
    ax2_2.step(Br_gamma_b2[i].wave,Br_gamma_b2[i].f,alpha=.5)
    ax3_2.step(Br_gamma_b3[i].wave,Br_gamma_b3[i].f,alpha=.5)
ax1_2.step(Br_gamma_b1[mean[2]].wave,Br_gamma_b1[mean[2]].f,c='r')
ax2_2.step(Br_gamma_b2[mean[2]].wave,Br_gamma_b2[mean[2]].f,c='r')
ax3_2.step(Br_gamma_b3[mean[2]].wave,Br_gamma_b3[mean[2]].f,c='r')

ax1_2.step(Br_gamma_b1[mean[0]].wave,Br_gamma_b1[mean[0]].f,c='k')
ax2_2.step(Br_gamma_b2[mean[0]].wave,Br_gamma_b2[mean[0]].f,c='k')
ax3_2.step(Br_gamma_b3[mean[0]].wave,Br_gamma_b3[mean[0]].f,c='k')
#-------------------------------------2021-------------------------------------
for i in d2021:
    ax1_3.step(Br_gamma_b1[i].wave,Br_gamma_b1[i].f,alpha=.5)
    ax2_3.step(Br_gamma_b2[i].wave,Br_gamma_b2[i].f,alpha=.5)
    ax3_3.step(Br_gamma_b3[i].wave,Br_gamma_b3[i].f,alpha=.5)
ax1_3.step(Br_gamma_b1[mean[3]].wave,Br_gamma_b1[mean[3]].f,c='r')
ax2_3.step(Br_gamma_b2[mean[3]].wave,Br_gamma_b2[mean[3]].f,c='r')
ax3_3.step(Br_gamma_b3[mean[3]].wave,Br_gamma_b3[mean[3]].f,c='r')

ax1_3.step(Br_gamma_b1[mean[0]].wave,Br_gamma_b1[mean[0]].f,c='k')
ax2_3.step(Br_gamma_b2[mean[0]].wave,Br_gamma_b2[mean[0]].f,c='k')
ax3_3.step(Br_gamma_b3[mean[0]].wave,Br_gamma_b3[mean[0]].f,c='k')

fig1.legend(),fig2.legend(),fig3.legend()
# General settings
ax1_1.set_ylim(0.9,1.2),ax2_1.set_ylim(0.9,1.2),ax3_1.set_ylim(0.9,1.2)
ax1_2.set_ylim(0.9,1.2),ax2_2.set_ylim(0.9,1.2),ax3_2.set_ylim(0.9,1.2)
ax1_3.set_ylim(0.9,1.2),ax2_3.set_ylim(0.9,1.2),ax3_3.set_ylim(0.9,1.2)
ax1_1.grid(),ax2_1.grid(),ax3_1.grid()
ax1_2.grid(),ax2_2.grid(),ax3_2.grid()
ax1_3.grid(),ax2_3.grid(),ax3_3.grid()
fig1.suptitle('No binning'),fig2.suptitle('Binning : 2 pixels'),fig3.suptitle('Binning : 3 pixels')
ax1_3.set_xlabel(r'Wavelenght ($\AA$)'),ax2_3.set_xlabel(r'Wavelenght ($\AA$)'),ax3_3.set_xlabel(r'Wavelenght ($\AA$)')
ax1_1.set_ylabel(r'2017'),ax2_1.set_ylabel(r'2017'),ax3_1.set_ylabel(r'2017')
ax1_2.set_ylabel(r'2018'),ax2_2.set_ylabel(r'2018'),ax3_2.set_ylabel(r'2018')
ax1_3.set_ylabel(r'2021'),ax2_3.set_ylabel(r'2021'),ax3_3.set_ylabel(r'2021')

ax1_1.axvline(Brgamma,c='m',markersize = 5),ax1_1.axvline(HeII,c='b',markersize = 5),ax1_1.axvline(HeI,c='g',markersize = 5)
ax1_2.axvline(Brgamma,c='m',markersize = 5),ax1_2.axvline(HeII,c='b',markersize = 5),ax1_2.axvline(HeI,c='g',markersize = 5)
ax1_3.axvline(Brgamma,c='m',markersize = 5),ax1_3.axvline(HeII,c='b',markersize = 5),ax1_3.axvline(HeI,c='g',markersize = 5)
ax2_1.axvline(Brgamma,c='m',markersize = 5),ax2_1.axvline(HeII,c='b',markersize = 5),ax2_1.axvline(HeI,c='g',markersize = 5)
ax2_2.axvline(Brgamma,c='m',markersize = 5),ax2_2.axvline(HeII,c='b',markersize = 5),ax2_2.axvline(HeI,c='g',markersize = 5)
ax2_3.axvline(Brgamma,c='m',markersize = 5),ax2_3.axvline(HeII,c='b',markersize = 5),ax2_3.axvline(HeI,c='g',markersize = 5)
ax3_1.axvline(Brgamma,c='m',markersize = 5),ax3_1.axvline(HeII,c='b',markersize = 5),ax3_1.axvline(HeI,c='g',markersize = 5)
ax3_2.axvline(Brgamma,c='m',markersize = 5),ax3_2.axvline(HeII,c='b',markersize = 5),ax3_2.axvline(HeI,c='g',markersize = 5)
ax3_3.axvline(Brgamma,c='m',markersize = 5),ax3_3.axvline(HeII,c='b',markersize = 5),ax3_3.axvline(HeI,c='g',markersize = 5)

ax1_1.text(Brgamma-20,1.21,r'Br$\gamma$',fontsize=14,c='m'),ax1_1.text(HeII-20,1.21,r'He II (21885)',fontsize=14,c='b'),ax1_1.text(HeI-20,1.21,r'He I (21126)',fontsize=14,c='g')
ax2_1.text(Brgamma-20,1.21,r'Br$\gamma$',fontsize=14,c='m'),ax2_1.text(HeII-20,1.21,r'He II (21885)',fontsize=14,c='b'),ax2_1.text(HeI-20,1.21,r'He I (21126)',fontsize=14,c='g')
ax3_1.text(Brgamma-20,1.21,r'Br$\gamma$',fontsize=14,c='m'),ax3_1.text(HeII-20,1.21,r'He II (21885)',fontsize=14,c='b'),ax3_1.text(HeI-20,1.21,r'He I (21126)',fontsize=14,c='g')

plt.show()

#%%
# Summary figure with MAXI LC, average spectra, and EW.

lin=False
os.chdir('/home/carlos/Desktop/MSc/TFM/GRS_1915+105_EMIR/MAXI_LC')
MAXI = pd.read_csv('MAXI_LC_GRS1915+105.dat',sep = ' ')
plt.close('all')
fig = plt.figure()
fig.set_figheight(6)
fig.set_figwidth(6)
ax1 = plt.subplot2grid(shape=(6, 6), loc=(0, 0), colspan=4, rowspan=5)
ax2 = plt.subplot2grid(shape=(6, 6), loc=(0, 4), colspan=2, rowspan=2)
ax3 = plt.subplot2grid(shape=(6, 6), loc=(2, 4), colspan=2, rowspan=2)
ax4 = plt.subplot2grid(shape=(6, 6), loc=(4, 4), colspan=2, rowspan=2)
ax5 = plt.subplot2grid(shape=(6, 6), loc=(5, 0), colspan=4, rowspan=1) 

ax2.get_shared_x_axes().join(ax2, ax3)
ax4.get_shared_x_axes().join(ax4, ax2)
ax2.get_shared_y_axes().join(ax2, ax3)
ax4.get_shared_y_axes().join(ax4, ax2)
ax1.get_shared_x_axes().join(ax1, ax5)

ax2.set_ylim(0.95,1.25)
ax1.tick_params('x', labelbottom=False)
ax2.tick_params('both', right=True, left=False, labelright=True,labelleft=False,labelbottom=False)
ax3.tick_params('both', right=True, left=False, labelright=True,labelleft=False, labelbottom=False)
ax4.tick_params('y', right=True, left=False, labelright=True,labelleft=False)
os.chdir('/home/carlos/Desktop/MSc/TFM/Reduced_spectra/MolecfitCorrected-20220209T161944Z-001/MolecfitCorrected')
obs_days = os.listdir()
Br_gamma_b2 = molly.rmolly('GRS1915_BrGamma_B2_interpolated.mol')
colors = 'g','b','orange'
years = '2017','2018','2021'
for da in range(-3,0):
    d=Br_gamma_b2[da].head['HJD']-2400000
    if lin==True: 
        ax1.axvline(d,c=colors[da],label=years[da],linestyle='-.')
    else:
        try:
            ax1.plot(d,Y_MAXI[da],'*',c=colors[da],label=years[da],markeredgewidth=3)
        except:
            ax1.axvline(d,c=colors[da],label=years[da],linestyle='-.')


#ax5.errorbar(data_t['MJD'][-3:]-2400000.5,data_t['EW'][-3:],fmt='none',yerr=data_t['dEW'][-3:],\
#             linewidth=2, ecolor = ['g','b','orange'])
#ax5.errorbar(data_c['MJD'][-3:]-2400000.5,data_c['EW'][-3:],fmt='none',yerr=data_c['dEW'][-3:],\
#             linewidth=2, ecolor = ['g','b','orange'])
#ax5.errorbar(data_w['MJD'][-3:]-2400000.5,data_w['EW'][-3:],fmt='none',yerr=data_w['dEW'][-3:],\
#             linewidth=2, ecolor = ['g','b','orange'])
#for i in range(len(data_c[-3:])):
#    ax5.plot(data_c['MJD'][15+i]-2400000.5,data_c['EW'][15+i],'v',c=colors[i],markeredgewidth=0.02)
#    ax5.plot(data_w['MJD'][15+i]-2400000.5,data_w['EW'][15+i],'D',c=colors[i],markeredgewidth=0.02)
#    ax5.plot(data_t['MJD'][15+i]-2400000.5,data_t['EW'][15+i],'s',c=colors[i],markeredgewidth=0.02)
#ax5.set_ylim(ymin=0)
#ax5.plot(data_c['MJD'][15+i]-2400000.5,-100,'sk',label='Total')
#ax5.plot(data_c['MJD'][15+i]-2400000.5,-100,'vk',label='Centre')
#ax5.plot(data_c['MJD'][15+i]-2400000.5,-100,'Dk',label='Wings')
#ax5.legend()

EW_m1 = pd.read_csv('EW_B2_interpolated_corrected_mask1.txt', delimiter = "\\s+",header = 3,index_col=False,names = ['MJD','EW','dEW'])
EW_m2 = pd.read_csv('EW_B2_interpolated_corrected_mask2.txt', delimiter = "\\s+",header = 3,index_col=False,names = ['MJD','EW','dEW'])
for i in range(15,18):
    ax5.plot(EW_m1['MJD'][i]-2400000.5,EW_m1['EW'][i],'v',c=colors[i-15],markeredgewidth=0.02)
    ax5.plot(EW_m2['MJD'][i]-2400000.5,EW_m2['EW'][i],'D',c=colors[i-15],markeredgewidth=0.02)
ax5.set_ylim(ymin=4.5)
ax5.plot(EW_m2['MJD'][i]-2400000.5,-100,'vk',label='Narrow mask')
ax5.plot(EW_m2['MJD'][i]-2400000.5,-100,'Dk',label='Broad mask')
ax5.legend()

ax2.axvline(0,linestyle='--',c='grey'),ax2.axhline(1,linestyle='--',c='grey')
ax3.axvline(0,linestyle='--',c='grey'),ax3.axhline(1,linestyle='--',c='grey')
ax4.axvline(0,linestyle='--',c='grey'),ax4.axhline(1,linestyle='--',c='grey')
ax1.grid(),ax5.grid()

ax2.axvline(vel_redshit(21620),linestyle=':',c='k'),ax2.axvline(vel_redshit(21700),linestyle=':',c='k')
ax3.axvline(vel_redshit(21620),linestyle=':',c='k'),ax3.axvline(vel_redshit(21700),linestyle=':',c='k')
ax4.axvline(vel_redshit(21620),linestyle=':',c='k'),ax4.axvline(vel_redshit(21700),linestyle=':',c='k')
ax2.axvline(vel_redshit(21580),linestyle=':',c='grey'),ax2.axvline(vel_redshit(21700),linestyle=':',c='grey')
ax3.axvline(vel_redshit(21580),linestyle=':',c='grey'),ax3.axvline(vel_redshit(21700),linestyle=':',c='grey')
ax4.axvline(vel_redshit(21580),linestyle=':',c='grey'),ax4.axvline(vel_redshit(21700),linestyle=':',c='grey')

ax2.axvline(vel_redshit(HeII),linestyle='-.',c='k')
ax3.axvline(vel_redshit(HeII),linestyle='-.',c='k')
ax4.axvline(vel_redshit(HeII),linestyle='-.',c='k')
ax2.text(vel_redshit(HeII-100),1.26,'He II')

os.chdir('/home/carlos/Desktop/MSc/TFM/GRS_1915+105_EMIR/PHOTOMETRY_17')
x = np.arange(0, 10, 0.1)
y = np.cos(x)
ax1.errorbar(MAXI['#MJD'],MAXI['2_20keV'],yerr = MAXI['e_2_20keV'],fmt='.',markersize = 0.5,c = 'k')
v17,v18,v21,vmean = vel_redshit(Br_gamma_b2[mean[1]].wave),vel_redshit(Br_gamma_b2[mean[2]].wave),vel_redshit(Br_gamma_b2[mean[3]].wave),vel_redshit(Br_gamma_b2[mean[0]].wave)
ax2.step(v17,Br_gamma_b2[mean[1]].f,'g',label ='Average 2017')
ax3.step(v18,Br_gamma_b2[mean[2]].f,'b',label ='Average 2018')
ax4.step(v21,Br_gamma_b2[mean[3]].f,'orange',label ='Average 2021')
ax2.step(vmean,Br_gamma_b2[mean[0]].f,'k',alpha=0.6)
ax3.step(vmean,Br_gamma_b2[mean[0]].f,'k',alpha=0.6)
ax4.step(vmean,Br_gamma_b2[mean[0]].f,'k',alpha=0.6)
ax1.legend()

ax5.set_xlabel('MJD'),ax1.set_ylabel(r'2-20 keV [Photons cm$^{-2}$s$^{-1}]$'),ax5.set_ylabel(r'EW [\AA]')
ax4.set_xlabel(r'Velocity [km/s]')

# display plot
plt.show()

#%%
##############################PLOT EW vs MAXI##################################
Y_MAXI, Y_dMAXI = [], []
for i in range(3):    
    p1,v1 = find_nearest(MAXI['#MJD'],EW_m1['MJD'][15+i]-2400000.5)
    f1,fe = interp(v1,EW_m1['MJD'][15+i]-2400000.5,p1)
    Y_MAXI.append(f1),Y_dMAXI.append(fe)
Y_EW_1,Y_dEW_1 = np.array(EW_m1['EW'][15:]), np.array(EW_m1['dEW'][15:])
Y_EW_2,Y_dEW_2 = np.array(EW_m2['EW'][15:]), np.array(EW_m2['dEW'][15:])
years = ['2017','2018','2021']
fig1,axs = plt.subplots()

axs.errorbar(Y_EW_1,Y_MAXI,fmt='none',xerr=Y_dEW_1,yerr=Y_dMAXI,c='k',markeredgewidth=1,label='Narrow mask')
axs.errorbar(Y_EW_2,Y_MAXI,fmt='none',xerr=Y_dEW_2,yerr=Y_dMAXI,c='r',markeredgewidth=1,label='Broad mask')
for i in range(len(years)):
    axs.text((Y_EW_1[i]+Y_EW_2[i])/2-0.2,Y_MAXI[i]+0.1,years[i])
axs.legend()
axs.grid()
axs.set_ylabel(r'2-20 keV [Photons cm$^{-2}$s$^{-1}]$'),axs.set_xlabel(r'EW [$\AA$]')

############################PLOT MAXI vs K-flux################################
# Interpolatting MAXI
#maxi,dmaxi = [],[]
maxi21,dmaxi21 = [],[]
maxi17,dmaxi17 = [],[]
for i in range(len(F17_mean[:,0])):    
    p1,v1 = find_nearest(MAXI['#MJD'],F17_mean[:,0][i])
    f1,fe = interp(v1,F17_mean[:,0][i],p1)
    #maxi.append(f1),dmaxi.append(fe)
    maxi17.append(f1),dmaxi17.append(fe)
for i in range(len(F21_mean[:,0])):    
    p1,v1 = find_nearest(MAXI['#MJD'],F21_mean[:,0][i])
    f1,fe = interp(v1,F21_mean[:,0][i],p1)
    #maxi.append(f1),dmaxi.append(fe)
    maxi21.append(f1),dmaxi21.append(fe)
# Plotting:
fig,axs = plt.subplots()
date_mean, flux_mean, fluxerr_mean = np.concatenate((F17_mean[:,0],F21_mean[:,0])), np.concatenate((F17_mean[:,2],F21_mean[:,2])), np.concatenate((F17_mean[:,3],F21_mean[:,3]))
#axs.errorbar(maxi,flux_mean,yerr=fluxerr_mean,xerr=dmaxi,fmt='none')
#axs.errorbar(maxi17,F17_mean[:,2],yerr=F17_mean[:,3],xerr=dmaxi17,fmt='none',c='b')
axs.errorbar(maxi21,F21_mean[:,2],yerr=F21_mean[:,3],xerr=dmaxi21,fmt='none',c='r')

axs.grid()
axs.set_xlabel(r'2-20 keV [Photons cm$^{-2}$s$^{-1}]$'),axs.set_ylabel(r'In-band K flux [Jy]')

# Building a linear fit:
A = np.polyfit(maxi21,F21_mean[:,2],1)    
axs.plot(maxi21,np.polyval(A,maxi21),label='Y='+str(A[0])[:7]+'X+'+str(A[1])[:7])
axs.legend()
#############################PLOT EW vs K-flux#################################
np.array(EW_m1['EW'][15:])
common_phot_EW = []
for d in range(len(np.around(date_mean))):
    if np.around(date_mean)[d] in np.around(np.array(EW_m1['MJD']-2400000.5)):
        common_phot_EW.append([np.around(date_mean)[d],flux_mean[d],fluxerr_mean[d],np.array(EW_m1['EW'])[d],np.array(EW_m1['dEW'])[d]])

common_phot_EW = np.vstack(common_phot_EW)

fig,axs = plt.subplots()
axs.errorbar(common_phot_EW[:,1],common_phot_EW[:,3],xerr=common_phot_EW[:,2],yerr=common_phot_EW[:,4],fmt='none')
axs.grid()
axs.set_ylabel(r'EW [$\AA$]'),axs.set_xlabel(r'In-band K flux [Jy]')

plt.show()
#%%
#############################Double gaussian fits##############################
MC = True
steps_MC = 1e2
plt.close('all')
fig,[ax1,ax2,ax3,ax4] = plt.subplots(1,4,sharey=True,sharex=True)
ax1.axvline(0,c='grey',linestyle='--'),ax1.axhline(1,c='grey',linestyle='--')
ax2.axvline(0,c='grey',linestyle='--'),ax2.axhline(1,c='grey',linestyle='--')
ax3.axvline(0,c='grey',linestyle='--'),ax3.axhline(1,c='grey',linestyle='--')
ax4.axvline(0,c='grey',linestyle='--'),ax4.axhline(1,c='grey',linestyle='--')
ax1.step(v17,Br_gamma_b2[mean[1]].f,'g',label ='Average 2017')
ax2.step(v18,Br_gamma_b2[mean[2]].f,'b',label ='Average 2018')
ax4.step(vmean,Br_gamma_b2[mean[0]].f,'k',label='Average')
ax3.step(v21,Br_gamma_b2[mean[3]].f,'orange',label ='Average 2021')
fig.supxlabel(r'Velocity [km/s]')
ax1.set_ylim(0.95,1.25),ax1.set_xlim(-3000,3000)
ax1.set_title('2017'),ax2.set_title('2018'),ax3.set_title('2021'),ax4.set_title('Average')

init_val17 = [0.1,0.05,0,-600,1000/2.355,500/2.355]
init_val18 = [0.2,0.05,0,-600,1000/2.355,500/2.355]
init_val21 = [0.15,0.05,0,-600,1000/2.355,500/2.355]
init_valAv = [0.1,0.08,0,-600,1000/2.355,500/2.355]
#==================================Curve fit=================================== 
mask17,mask18,mask21,maskAv = (v17>-3000) & (v17<1500),(v18>-3000) & (v18<1500),(v21>-3000) & (v21<1500),(vmean>-2000) & (vmean<1500)
parameters17, covariance17 = curve_fit(double_Gauss, v17[mask17], Br_gamma_b2[mean[1]].f[mask17],p0=init_val17)
fit17_A1,fit17_A2,fit17_m1,fit17_m2,fit17_s1,fit17_s2 = parameters17
parameters18, covariance18 = curve_fit(double_Gauss, v18[mask18], Br_gamma_b2[mean[2]].f[mask18],p0=init_val18)
fit18_A1,fit18_A2,fit18_m1,fit18_m2,fit18_s1,fit18_s2 = parameters18
parameters21, covariance21 = curve_fit(double_Gauss, v21[mask21], Br_gamma_b2[mean[3]].f[mask21],p0=init_val21)
fit21_A1,fit21_A2,fit21_m1,fit21_m2,fit21_s1,fit21_s2 = parameters21
parametersAv, covarianceAv = curve_fit(double_Gauss, vmean[maskAv], Br_gamma_b2[mean[0]].f[maskAv],p0=init_valAv)
fitAv_A1,fitAv_A2,fitAv_m1,fitAv_m2,fitAv_s1,fitAv_s2 = parametersAv
#==================================Curve fit=================================== 
#============================Plotting gaussian fits============================
# 2017
ax1.plot(vmean,Gauss(vmean,fit17_A1,fit17_m1,fit17_s1),'b--',linewidth=0.7)
ax1.plot(vmean,Gauss(vmean,fit17_A2,fit17_m2,fit17_s2),'b--',linewidth=0.7)
ax1.plot(v17,double_Gauss(v17,fit17_A1,fit17_A2,fit17_m1,fit17_m2,fit17_s1,fit17_s2),'r')
# 2018
ax2.plot(vmean,Gauss(vmean,fit18_A1,fit18_m1,fit18_s1),'b--',linewidth=0.7)
ax2.plot(vmean,Gauss(vmean,fit18_A2,fit18_m2,fit18_s2),'b--',linewidth=0.7)
ax2.plot(v18,double_Gauss(v18,fit18_A1,fit18_A2,fit18_m1,fit18_m2,fit18_s1,fit18_s2),'r')
# 2021
ax3.plot(vmean,Gauss(vmean,fit21_A1,fit21_m1,fit21_s1),'b--',linewidth=0.7)
ax3.plot(vmean,Gauss(vmean,fit21_A2,fit21_m2,fit21_s2),'b--',linewidth=0.7)
ax3.plot(v21,double_Gauss(v21,fit21_A1,fit21_A2,fit21_m1,fit21_m2,fit21_s1,fit21_s2),'r')
# Average
ax4.plot(vmean,Gauss(vmean,fitAv_A1,fitAv_m1,fitAv_s1),'b--',linewidth=0.7)
ax4.plot(vmean,Gauss(vmean,fitAv_A2,fitAv_m2,fitAv_s2),'b--',linewidth=0.7)
ax4.plot(vmean,double_Gauss(vmean,fitAv_A1,fitAv_A2,fitAv_m1,fitAv_m2,fitAv_s1,fitAv_s2),'r')
#============================Plotting gaussian fits============================
plt.show()
#============================Trying different fits=============================
if MC:
    par_17,par_18,par_21,par_av = [],[],[],[]
    cov_17,cov_18,cov_21,cov_av = [],[],[],[]
    bar = progressbar.ProgressBar(maxval=int(steps_MC))
    I = 0
    bar.start()
    for i in range(int(steps_MC)):
        var_a1,var_a2 = np.random.normal(loc = 1, scale=0.15), np.random.normal(loc = 1, scale=0.15)
        var_m1,var_m2 = np.random.normal(loc = 1, scale=0.15), np.random.normal(loc = 1, scale=0.15) 
        var_s1,var_s2 = np.random.normal(loc = 1, scale=0.15), np.random.normal(loc = 1, scale=0.15)
        try:
            init_val17 = [0.1*var_a1,0.05*var_a2,10*var_m1,-600*var_m2,1000/2.355*var_s1,500/2.355*var_s2]
            init_val18 = [0.2*var_a1,0.05*var_a2,10*var_m1,-600*var_m2,1000/2.355*var_s1,500/2.355*var_s2]
            init_val21 = [0.15*var_a1,0.05*var_a2,10*var_m1,-600*var_m2,1000/2.355*var_s1,500/2.355*var_s2]
            init_valAv = [0.1*var_a1,0.08*var_a2,10*var_a1,-600*var_m2,1000/2.355*var_s1,500/2.355*var_s2]
            parameters17, covariance17 = curve_fit(double_Gauss, v17[mask17], Br_gamma_b2[mean[1]].f[mask17],p0=init_val17)
            parameters18, covariance18 = curve_fit(double_Gauss, v18[mask18], Br_gamma_b2[mean[2]].f[mask17],p0=init_val18)
            parameters21, covariance21 = curve_fit(double_Gauss, v21[mask21], Br_gamma_b2[mean[3]].f[mask18],p0=init_val21)
            parametersAv, covarianceAv = curve_fit(double_Gauss, vmean[maskAv], Br_gamma_b2[mean[0]].f[maskAv],p0=init_valAv)
            if parameters17[0]>=0 and parameters17[1]>=0:
                par_17.append(parameters17)
                cov_17.append([covariance17])
            if parameters18[0]>=0 and parameters18[1]>=0:
                par_18.append(parameters18)
                cov_18.append([covariance18])
            if parameters21[0]>=0 and parameters21[1]>=0:
                par_21.append(parameters21)
                cov_21.append([covariance21])
            if parametersAv[0]>=0 and parametersAv[1]>=0:
                par_av.append(parametersAv)
                cov_av.append([covarianceAv])
        except: pass
        I += 1
        bar.update(I)
    bar.finish()    
    
    par_17, par_18, par_21, par_av = np.vstack(par_17),np.vstack(par_18),np.vstack(par_21),np.vstack(par_av)
    cov_17, cov_18, cov_21, cov_av = np.vstack(cov_17),np.vstack(cov_18),np.vstack(cov_21),np.vstack(cov_av)
    
    #np.median, np.std
    #np.sqrt(np.diag())
    #fit21_A1,fit21_A2,fit21_m1,fit21_m2,fit21_s1,fit21_s2 = 
    
    parameters17, errors_17 = np.median(par_17,axis=0), np.sqrt(np.diag(np.median(cov_17,axis=0)))
    parameters18, errors_18 = np.median(par_18,axis=0), np.sqrt(np.diag(np.median(cov_18,axis=0)))
    parameters21, errors_21 = np.median(par_21,axis=0), np.sqrt(np.diag(np.median(cov_21,axis=0)))
    parametersAv, errors_av = np.median(par_av,axis=0), np.sqrt(np.diag(np.median(cov_av,axis=0)))
print('\n')
print(parameters17,'\n',np.std((par_17),axis=0)[:-2],'\n',np.std(abs(par_17),axis=0)[-2:])
print(parameters18,'\n',np.std((par_18),axis=0)[:-2],'\n',np.std(abs(par_18),axis=0)[-2:])
print(parameters21,'\n',np.std((par_21),axis=0)[:-2],'\n',np.std(abs(par_21),axis=0)[-2:])
print(parametersAv,'\n',np.std((par_av),axis=0)[:-2],'\n',np.std(abs(par_av),axis=0)[-2:])

print(len(par_17),len(par_18),len(par_21),len(par_av))

'''




#%% Repeating the previous process for the other groups.
#####################Double gaussian fits to other groups######################

matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
matplotlib.rcParams.update({'font.size': 22})

os.chdir("/home/carlos/Desktop/MSc/TFM/Reduced_spectra/MolecfitCorrected-20220209T161944Z-001/MolecfitCorrected/")
EW2_m1 = pd.read_csv('EW2_B2_interpolated_corrected_mask1.txt', delimiter = "\\s+",header = 3,index_col=False,names = ['MJD','EW','dEW'])
EW2_m2 = pd.read_csv('EW2_B2_interpolated_corrected_mask2.txt', delimiter = "\\s+",header = 3,index_col=False,names = ['MJD','EW','dEW'])
EW = pd.read_csv('EW_B2_interpolated_corrected_mask1.txt', delimiter = "\\s+",header = 3,index_col=False,names = ['MJD','EW','dEW'])
lin=True
os.chdir('/home/carlos/Desktop/MSc/TFM/GRS_1915+105_EMIR/MAXI_LC')
MAXI = pd.read_csv('MAXI_LC_GRS1915+105.dat',sep = ' ')
plt.close('all')
fig = plt.figure()#,constrained_layout=True)
'''
fig.set_figheight(6)
fig.set_figwidth(8)
ax1 = plt.subplot2grid(shape=(6, 8), loc=(0, 0), colspan=4, rowspan=5)
ax8 = plt.subplot2grid(shape=(6, 8), loc=(5, 0), colspan=4, rowspan=1) 
#ax1.margins(x=300)
ax2 = plt.subplot2grid(shape=(6, 8), loc=(0, 5), colspan=1, rowspan=3)
ax3 = plt.subplot2grid(shape=(6, 8), loc=(0, 6), colspan=1, rowspan=3)
ax4 = plt.subplot2grid(shape=(6, 8), loc=(0, 7), colspan=1, rowspan=3)
ax5 = plt.subplot2grid(shape=(6, 8), loc=(3, 5), colspan=1, rowspan=3)
ax6 = plt.subplot2grid(shape=(6, 8), loc=(3, 6), colspan=1, rowspan=3)
ax7 = plt.subplot2grid(shape=(6, 8), loc=(3, 7), colspan=1, rowspan=3)

ax9 = plt.subplot2grid(shape=(6, 8), loc=(0, 4), colspan=1, rowspan=5)
ax10 = plt.subplot2grid(shape=(6, 8), loc=(5, 4), colspan=1, rowspan=1)
'''
fig.set_figheight(20)
fig.set_figwidth(28)
ax1 = plt.subplot2grid(shape=(20, 28), loc=(0, 0), colspan=15, rowspan=17)
ax8 = plt.subplot2grid(shape=(20, 28), loc=(17, 0), colspan=15, rowspan=3) 
#ax1.margins(x=300)

ax2 = plt.subplot2grid(shape=(20, 28), loc=(0, 19), colspan=3, rowspan=10)
ax3 = plt.subplot2grid(shape=(20, 28), loc=(0, 22), colspan=3, rowspan=10)
ax4 = plt.subplot2grid(shape=(20, 28), loc=(0, 25), colspan=3, rowspan=10)
ax5 = plt.subplot2grid(shape=(20, 28), loc=(10, 19), colspan=3, rowspan=10)
ax6 = plt.subplot2grid(shape=(20, 28), loc=(10, 22), colspan=3, rowspan=10)
ax7 = plt.subplot2grid(shape=(20, 28), loc=(10, 25), colspan=3, rowspan=10)

ax9 = plt.subplot2grid(shape=(20, 28), loc=(0, 15), colspan=4, rowspan=17)
ax10 = plt.subplot2grid(shape=(20, 28), loc=(17, 15), colspan=4, rowspan=3)

plt.subplots_adjust(wspace=2,hspace=0.5)

ax2.get_shared_x_axes().join(ax2, ax3)
ax3.get_shared_x_axes().join(ax3, ax4)
ax4.get_shared_x_axes().join(ax4, ax5)
ax5.get_shared_x_axes().join(ax5, ax6)
ax6.get_shared_x_axes().join(ax6, ax7)
ax2.get_shared_y_axes().join(ax2, ax3)
ax3.get_shared_y_axes().join(ax3, ax4)
ax4.get_shared_y_axes().join(ax4, ax5)
ax5.get_shared_y_axes().join(ax5, ax6)
ax6.get_shared_y_axes().join(ax6, ax7)
ax1.get_shared_x_axes().join(ax1, ax8)
ax10.get_shared_x_axes().join(ax10, ax9)
ax1.tick_params('x', labelbottom=False)
ax2.tick_params('both', left=False,right=True, labelbottom=False,labelleft=False)
ax3.tick_params('both', left=False,right=True, labelbottom=False,labelleft=False)
ax4.tick_params('both', left=False,right=True, labelbottom=False,labelleft=False,labelright=True)
ax5.tick_params('y', left=False,right=True, labelleft=False)
ax6.tick_params('y', left=False,right=True, labelleft=False)
ax7.tick_params('y', left=False,right=True, labelleft=False,labelright=True)
ax3.tick_params('x', left=False,right=True, labelbottom=False)
ax9.tick_params('x', left=False,right=True, labelbottom=False)
ax1.set_xlim(57500,59700)
ax2.set_ylim(0.95,1.25),ax2.set_xlim(-6000,6000)



os.chdir('/home/carlos/Desktop/MSc/TFM/Reduced_spectra/MolecfitCorrected-20220209T161944Z-001/MolecfitCorrected')
obs_days = os.listdir()
Br_gamma_mean2_b2 = molly.rmolly('average2_spectra_B2_BrGamma_interpolated.mol')

v17_2,v18_2,v21_2 = vel_redshit(Br_gamma_mean2_b2[0].wave),vel_redshit(Br_gamma_mean2_b2[1].wave),vel_redshit(Br_gamma_mean2_b2[-1].wave)

years = '2017','2018A','2018B','2021A','2021B','2021C'
colors = 'g','olive','darkgoldenrod','red','firebrick','indianred'
for da in range(len(Br_gamma_mean2_b2)):
    d=Br_gamma_mean2_b2[da].head['HJD']-2400000
    if lin==True: 
        ax1.axvline(d,c=colors[da],label=years[da],linestyle='-.')
    else:
        try:
            ax1.plot(d,Y_MAXI[da],'*',c=colors[da],label=years[da],markeredgewidth=3)
        except:
            ax1.axvline(d,c=colors[da],label=years[da],linestyle='-.')
ax1.errorbar(MAXI['#MJD'],MAXI['2_20keV'],yerr = MAXI['e_2_20keV'],fmt='.',markersize = 0.5,c = 'k')

for i in range(len(Br_gamma_mean2_b2)):
    ax8.plot(EW2_m1['MJD'][i]-2400000.5,EW2_m1['EW'][i],'v',c=colors[i],markeredgewidth=0.02)
#    ax8.plot(EW2_m2['MJD'][i]-2400000.5,EW2_m2['EW'][i],'D',c=colors[i],markeredgewidth=0.02)
ax8.set_ylim(ymin=3.)
ax8.plot(EW2_m2['MJD'][i]-2400000.5,-100,'vk',label='Narrow mask')
#ax8.plot(EW2_m2['MJD'][i]-2400000.5,-100,'Dk',label='Broad mask')
#ax8.legend(loc='upper left')


def vel_redshit(lam):
    c_km_s_1 = 299792.458
    return (lam - Brgamma)*c_km_s_1/Brgamma

ax2.axvline(0,linestyle='--',c='grey'),ax2.axhline(1,linestyle='--',c='grey')
ax3.axvline(0,linestyle='--',c='grey'),ax3.axhline(1,linestyle='--',c='grey')
ax4.axvline(0,linestyle='--',c='grey'),ax4.axhline(1,linestyle='--',c='grey')
ax5.axvline(0,linestyle='--',c='grey'),ax5.axhline(1,linestyle='--',c='grey')
ax6.axvline(0,linestyle='--',c='grey'),ax6.axhline(1,linestyle='--',c='grey')
ax7.axvline(0,linestyle='--',c='grey'),ax7.axhline(1,linestyle='--',c='grey')
ax1.grid(),ax8.grid()

ax2.axvline(vel_redshit(21620),linestyle=':',c='k'),ax2.axvline(vel_redshit(21700),linestyle=':',c='k')
ax3.axvline(vel_redshit(21620),linestyle=':',c='k'),ax3.axvline(vel_redshit(21700),linestyle=':',c='k')
ax4.axvline(vel_redshit(21620),linestyle=':',c='k'),ax4.axvline(vel_redshit(21700),linestyle=':',c='k')
ax5.axvline(vel_redshit(21620),linestyle=':',c='k'),ax5.axvline(vel_redshit(21700),linestyle=':',c='k')
ax6.axvline(vel_redshit(21620),linestyle=':',c='k'),ax6.axvline(vel_redshit(21700),linestyle=':',c='k')
ax7.axvline(vel_redshit(21620),linestyle=':',c='k'),ax7.axvline(vel_redshit(21700),linestyle=':',c='k')
ax2.axvline(vel_redshit(21580),linestyle=':',c='grey'),ax2.axvline(vel_redshit(21700),linestyle=':',c='grey')
ax3.axvline(vel_redshit(21580),linestyle=':',c='grey'),ax3.axvline(vel_redshit(21700),linestyle=':',c='grey')
ax4.axvline(vel_redshit(21580),linestyle=':',c='grey'),ax4.axvline(vel_redshit(21700),linestyle=':',c='grey')
ax5.axvline(vel_redshit(21580),linestyle=':',c='grey'),ax5.axvline(vel_redshit(21700),linestyle=':',c='grey')
ax6.axvline(vel_redshit(21580),linestyle=':',c='grey'),ax6.axvline(vel_redshit(21700),linestyle=':',c='grey')
ax7.axvline(vel_redshit(21580),linestyle=':',c='grey'),ax7.axvline(vel_redshit(21700),linestyle=':',c='grey')

ax2.axvline(vel_redshit(HeII),linestyle='-.',c='k')
ax3.axvline(vel_redshit(HeII),linestyle='-.',c='k')
ax4.axvline(vel_redshit(HeII),linestyle='-.',c='k')
ax5.axvline(vel_redshit(HeII),linestyle='-.',c='k')
ax6.axvline(vel_redshit(HeII),linestyle='-.',c='k')
ax7.axvline(vel_redshit(HeII),linestyle='-.',c='k')
ax2.text(vel_redshit(HeII-100),1.26,'He II'),ax3.text(vel_redshit(HeII-100),1.26,'He II'),ax4.text(vel_redshit(HeII-100),1.26,'He II')

os.chdir('/home/carlos/Desktop/MSc/TFM/GRS_1915+105_EMIR/PHOTOMETRY_17')
x = np.arange(0, 10, 0.1)
y = np.cos(x)
ax1.errorbar(MAXI['#MJD'],MAXI['2_20keV'],yerr = MAXI['e_2_20keV'],fmt='.',markersize = 0.5,c = 'k')
v17,v18,v21,vmean = vel_redshit(Br_gamma_b2[mean[1]].wave),vel_redshit(Br_gamma_b2[mean[2]].wave),vel_redshit(Br_gamma_b2[mean[3]].wave),vel_redshit(Br_gamma_b2[mean[0]].wave)
ax2.step(v17_2,Br_gamma_mean2_b2[0].f,colors[0],label = years[0])
ax3.step(v18_2,Br_gamma_mean2_b2[1].f,colors[1],label = years[1])
ax4.step(v21_2,Br_gamma_mean2_b2[2].f,colors[2],label = years[2])
ax5.step(v17_2,Br_gamma_mean2_b2[3].f,colors[3],label = years[3])
ax6.step(v18_2,Br_gamma_mean2_b2[4].f,colors[4],label = years[4])
ax7.step(v21_2,Br_gamma_mean2_b2[5].f,colors[5],label = years[5])
ax2.text(-4800,1.23,years[0]),ax3.text(-4800,1.23,years[1]),ax4.text(-4800,1.23,years[2])
ax5.text(-4800,1.23,years[3]),ax6.text(-4800,1.23,years[4]),ax7.text(-4800,1.23,years[5])



ax9.errorbar(MAXI['#MJD'],MAXI['2_20keV'],yerr = MAXI['e_2_20keV'],fmt='.',markersize = 0.5,c = 'k')
for da in range(len(Br_gamma_mean2_b2)):
    d=Br_gamma_mean2_b2[da].head['HJD']-2400000
    if lin==True: 
        ax9.axvline(d,c=colors[da],label=years[da],linestyle='-.')
    else:
        try:
            ax9.plot(d,Y_MAXI[da],'*',c=colors[da],label=years[da],markeredgewidth=3)
        except:
            ax9.axvline(d,c=colors[da],label=years[da],linestyle='-.')
ax9.errorbar(MAXI['#MJD'],MAXI['2_20keV'],yerr = MAXI['e_2_20keV'],fmt='.',markersize = 0.5,c = 'k')
for i in range(len(Br_gamma_mean2_b2)):
    ax10.plot(EW2_m1['MJD'][i]-2400000.5,EW2_m1['EW'][i],'v',c=colors[i],markeredgewidth=0.02)
#    ax10.plot(EW2_m2['MJD'][i]-2400000.5,EW2_m2['EW'][i],'D',c=colors[i],markeredgewidth=0.02)
ax10.plot(EW2_m2['MJD'][i]-2400000.5,-100,'vk',label='Narrow mask')
#ax10.plot(EW2_m2['MJD'][i]-2400000.5,-100,'Dk',label='Broad mask')
ax9.set_xlim(59425,59500),ax9.set_ylim(0,0.5),ax10.set_ylim(3.8,7.2)
ax10.set_xlim(59430,59490),ax10.grid(),ax9.grid()
ax10.set_xticks([59435,59460,59485])

plt.savefig(path_img+'summary.png')

# display plot
plt.show()

matplotlib.rcdefaults()
#%%

date_mean, flux_mean, fluxerr_mean = np.concatenate((F17_mean[:,0],F21_mean[:,0])), np.concatenate((F17_mean[:,2],F21_mean[:,2])), np.concatenate((F17_mean[:,3],F21_mean[:,3]))
##############################PLOT EW vs MAXI##################################
Y_MAXI_av, Y_dMAXI_av = [], []
Y_MAXI_day, Y_dMAXI_day = [], []
for i in range(len(EW2_m1)):    
    p1,v1 = find_nearest(MAXI['#MJD'],EW2_m1['MJD'][i]-2400000.5)
    f1,fe = interp(v1,EW2_m1['MJD'][i]-2400000.5,p1)
    Y_MAXI_av.append(f1),Y_dMAXI_av.append(fe)
for i in range(14):
    p1,v1 = find_nearest(MAXI['#MJD'],EW['MJD'][i]-2400000.5)
    f1,fe = interp(v1,EW['MJD'][i]-2400000.5,p1)
    Y_MAXI_day.append(f1),Y_dMAXI_day.append(fe)  
   
Y_EW2_1,Y_dEW2_1 = np.array(EW2_m1['EW']), np.array(EW2_m1['dEW'])
Y_EW2_2,Y_dEW2_2 = np.array(EW2_m2['EW']), np.array(EW2_m2['dEW'])
Y_EW_1_day,Y_dEW_1_day = np.array(EW2_m1['EW'][:14]), np.array(EW2_m1['dEW'][:14])
Y_EW_2_day,Y_dEW_2_day = np.array(EW2_m2['EW'][:14]), np.array(EW2_m2['dEW'][:14])
fig1,axs = plt.subplots(constrained_layout=True)

for i in range(len(Y_MAXI_av)):
    axs.errorbar(Y_EW2_2[i],Y_MAXI_av[i],fmt='none',xerr=Y_dEW2_2[i],yerr=Y_dMAXI_av[i],c=colors[i],markeredgewidth=1,label=years[i])
axs.legend()
axs.grid()
axs.set_ylabel(r'2-20 keV [Photons cm$^{-2}$s$^{-1}]$'),axs.set_xlabel(r'EW [$\AA$]')

plt.savefig(path_img+'MAXI_EW.png')

#############################PLOT EW vs K-flux#################################
# Getting points with both photometry and spectroscoy
common_phot_EW = []
for d in range(len(np.around(date_mean))):
    if np.around(date_mean)[d] in np.around(np.array(EW['MJD']-2400000.5)):
        common_phot_EW.append([np.around(date_mean)[d],flux_mean[d],fluxerr_mean[d],np.array(EW['EW'])[d],np.array(EW['dEW'])[d]])
common_phot_EW = np.vstack(common_phot_EW)

fig,axs = plt.subplots(constrained_layout=True)
for i in range(len(common_phot_EW[:,0])):
    if common_phot_EW[i,0]<58050: j=0
    elif common_phot_EW[i,0]<58310: j =1
    elif common_phot_EW[i,0]<58370: j =2
    elif common_phot_EW[i,0]<59450: j =3
    elif common_phot_EW[i,0]<59455: j =4
    elif common_phot_EW[i,0]<59480: j =5
    axs.errorbar(common_phot_EW[i,1],common_phot_EW[i,3],xerr=common_phot_EW[i,2],yerr=common_phot_EW[i,4],c=colors[j],fmt='none')
axs.set_ylim(1.3,8.3),axs.set_xlim(0.002,0.007)
ghost = [-100,-100,-100,-100,-100,-100]
for g in range(len(ghost)):
    axs.errorbar(ghost[g],ghost[g],xerr=ghost[g],yerr=ghost[g],c=colors[g],label=years[g])

axs.grid()
axs.set_ylabel(r'EW [$\AA$]'),axs.set_xlabel(r'In-band K flux [Jy]')
axs.legend()
plt.savefig(path_img+'EW_K.png')


############################PLOT MAXI vs K-flux################################
# Interpolatting MAXI
fig,axs = plt.subplots(constrained_layout=True)
'''
maxi17,dmaxi17 = [],[]
for i in range(len(F17_mean[:,0])):    
    p1,v1 = find_nearest(MAXI['#MJD'],F17_mean[:,0][i])
    f1,fe = interp(v1,F17_mean[:,0][i],p1)
    #maxi.append(f1),dmaxi.append(fe)
    maxi17.append(f1),dmaxi17.append(fe)
axs.errorbar(maxi17,F17_mean[:,2],yerr=F17_mean[:,3],xerr=dmaxi17,fmt='none',c='b')
'''
maxi21,dmaxi21 = [],[]
for i in range(len(F21_mean[:,0])):    
    p1,v1 = find_nearest(MAXI['#MJD'],F21_mean[:,0][i])
    f1,fe = interp(v1,F21_mean[:,0][i],p1)
    #maxi.append(f1),dmaxi.append(fe)
    maxi21.append(f1),dmaxi21.append(fe)
# Plotting:
for i in range(len(F21_mean)):
    if F21_mean[i,0]<59450: j =3
    elif F21_mean[i,0]<59455: j =4
    elif F21_mean[i,0]<59480: j =5
    axs.errorbar(maxi21[i],F21_mean[i,2],yerr=F21_mean[i,3],xerr=dmaxi21[i],fmt='none',c=colors[j])
axs.grid()
axs.set_xlabel(r'2-20 keV [Photons cm$^{-2}$s$^{-1}]$'),axs.set_ylabel(r'In-band K flux [Jy]')
axs.set_xlim(0.05,0.35),axs.set_ylim(0.002,0.005)
ghost = [-100,-100,-100]
for g in range(len(ghost)):
    axs.errorbar(ghost[g],ghost[g],xerr=ghost[g],yerr=ghost[g],c=colors[g+3],label=years[g+3])
# Building a linear fit:
A = np.polyfit(maxi21,F21_mean[:,2],1)    
axs.plot(maxi21,np.polyval(A,maxi21),label='Y='+str(A[0])[:7]+'X+'+str(A[1])[:7])
axs.legend()
plt.savefig(path_img+'MAXI_K.png')
plt.show()




#%%
#############################Double gaussian fits##############################
MC = False
steps_MC = 1e3
#plt.close('all')
fig,ax = plt.subplots(3,2,sharey=True,sharex=True, constrained_layout=True)
[ax1,ax4],[ax2,ax5],[ax3,ax6] = ax
ax1.axvline(0,c='grey',linestyle='--'),ax1.axhline(1,c='grey',linestyle='--')
ax2.axvline(0,c='grey',linestyle='--'),ax2.axhline(1,c='grey',linestyle='--')
ax3.axvline(0,c='grey',linestyle='--'),ax3.axhline(1,c='grey',linestyle='--')
ax4.axvline(0,c='grey',linestyle='--'),ax4.axhline(1,c='grey',linestyle='--')
ax5.axvline(0,c='grey',linestyle='--'),ax5.axhline(1,c='grey',linestyle='--')
ax6.axvline(0,c='grey',linestyle='--'),ax6.axhline(1,c='grey',linestyle='--')

ax1.step(v17_2,Br_gamma_mean2_b2[0].f,colors[0],label ='Average 2017'),ax1.set_title(years[0])
ax2.step(v18_2,Br_gamma_mean2_b2[1].f,colors[1],label ='Average 2018'),ax2.set_title(years[1])
ax3.step(v18_2,Br_gamma_mean2_b2[2].f,colors[2],label ='Average 2021'),ax3.set_title(years[2])
ax4.step(v21_2,Br_gamma_mean2_b2[3].f,colors[3],label ='Average 2017'),ax4.set_title(years[3])
ax5.step(v21_2,Br_gamma_mean2_b2[4].f,colors[4],label ='Average 2018'),ax5.set_title(years[4])
ax6.step(v21_2,Br_gamma_mean2_b2[5].f,colors[5],label ='Average 2021'),ax6.set_title(years[5])
fig.supxlabel(r'Velocity [km/s]')
ax1.set_ylim(0.95,1.25),ax1.set_xlim(-3000,3000)
init_val17 = [0.1,0.02,0,-750,1000/2.355,700/2.355]
#init_val18 = [0.2,0.05,0,-600,1000/2.355,500/2.355]
init_val18 = [0.2,0.05,0,-1000,1000/2.355,1200/2.355]
init_val21 = [0.15,0.02,0,-800,1000/2.355,300/2.355]
init_valAv = [0.1,0.08,0,-600,1000/2.355,500/2.355]
#==================================Curve fit=================================== 
mask17_2,mask18_2,mask21_2, maskAv = (v17_2>-3000) & (v17_2<1500),(v18_2>-3000) & (v18_2<1500),(v21_2>-3000) & (v21_2<1500), (vmean>-2000) & (vmean<1500)
parameters17, covariance17 = curve_fit(double_Gauss, v17_2[mask17_2], Br_gamma_mean2_b2[0].f[mask17_2],p0=init_val17)
fit17_A1,fit17_A2,fit17_m1,fit17_m2,fit17_s1,fit17_s2 = parameters17

parameters18A, covariance18A = curve_fit(double_Gauss, v18_2[mask18_2], Br_gamma_mean2_b2[1].f[mask18_2],p0=init_val18)
parameters18B, covariance18B = curve_fit(double_Gauss, v18_2[mask18_2], Br_gamma_mean2_b2[2].f[mask18_2],p0=init_val18)
fit18A_A1,fit18A_A2,fit18A_m1,fit18A_m2,fit18A_s1,fit18A_s2 = parameters18A
fit18B_A1,fit18B_A2,fit18B_m1,fit18B_m2,fit18B_s1,fit18B_s2 = parameters18B

parameters21A, covariance21A = curve_fit(double_Gauss, v21_2[mask21_2], Br_gamma_mean2_b2[3].f[mask21_2],p0=init_val21)
parameters21B, covariance21B = curve_fit(double_Gauss, v21_2[mask21_2], Br_gamma_mean2_b2[4].f[mask21_2],p0=init_val21)
parameters21C, covariance21C = curve_fit(double_Gauss, v21_2[mask21_2], Br_gamma_mean2_b2[5].f[mask21_2],p0=init_val21)
fit21A_A1,fit21A_A2,fit21A_m1,fit21A_m2,fit21A_s1,fit21A_s2 = parameters21A
fit21B_A1,fit21B_A2,fit21B_m1,fit21B_m2,fit21B_s1,fit21B_s2 = parameters21B
fit21C_A1,fit21C_A2,fit21C_m1,fit21C_m2,fit21C_s1,fit21C_s2 = parameters21C

parametersAv, covarianceAv = curve_fit(double_Gauss, vmean[maskAv], Br_gamma_b2[mean[0]].f[maskAv],p0=init_valAv)
fitAv_A1,fitAv_A2,fitAv_m1,fitAv_m2,fitAv_s1,fitAv_s2 = parametersAv

#==================================Curve fit=================================== 

#============================Plotting gaussian fits============================
# 2017
ax1.plot(vmean,Gauss(vmean,fit17_A1,fit17_m1,fit17_s1),'b--',linewidth=0.7)
ax1.plot(vmean,Gauss(vmean,fit17_A2,fit17_m2,fit17_s2),'b--',linewidth=0.7)
ax1.plot(v17,double_Gauss(v17,fit17_A1,fit17_A2,fit17_m1,fit17_m2,fit17_s1,fit17_s2),'navy')
# 2018
ax2.plot(vmean,Gauss(vmean,fit18A_A1,fit18A_m1,fit18A_s1),'b--',linewidth=0.7)
ax2.plot(vmean,Gauss(vmean,fit18A_A2,fit18A_m2,fit18A_s2),'b--',linewidth=0.7)
ax2.plot(v18,double_Gauss(v18,fit18A_A1,fit18A_A2,fit18A_m1,fit18A_m2,fit18A_s1,fit18A_s2),'navy')
ax3.plot(vmean,Gauss(vmean,fit18B_A1,fit18B_m1,fit18B_s1),'b--',linewidth=0.7)
ax3.plot(vmean,Gauss(vmean,fit18B_A2,fit18B_m2,fit18B_s2),'b--',linewidth=0.7)
ax3.plot(v18,double_Gauss(v18,fit18B_A1,fit18B_A2,fit18B_m1,fit18B_m2,fit18B_s1,fit18B_s2),'navy')
# 2021
ax4.plot(v21,Gauss(v21,fit21A_A1,fit21A_m1,fit21A_s1),'b--',linewidth=0.7)
ax4.plot(v21,Gauss(v21,fit21A_A2,fit21A_m2,fit21A_s2),'b--',linewidth=0.7)
ax4.plot(v21,double_Gauss(v21,fit21A_A1,fit21A_A2,fit21A_m1,fit21A_m2,fit21A_s1,fit21A_s2),'navy')
ax5.plot(v21,Gauss(v21,fit21B_A1,fit21B_m1,fit21B_s1),'b--',linewidth=0.7)
ax5.plot(v21,Gauss(v21,fit21B_A2,fit21B_m2,fit21B_s2),'b--',linewidth=0.7)
ax5.plot(v21,double_Gauss(v21,fit21B_A1,fit21B_A2,fit21B_m1,fit21B_m2,fit21B_s1,fit21B_s2),'navy')
ax6.plot(v21,Gauss(v21,fit21C_A1,fit21C_m1,fit21C_s1),'b--',linewidth=0.7)
ax6.plot(v21,Gauss(v21,fit21C_A2,fit21C_m2,fit21C_s2),'b--',linewidth=0.7)
ax6.plot(v21,double_Gauss(v21,fit21C_A1,fit21C_A2,fit21C_m1,fit21C_m2,fit21C_s1,fit21C_s2),'navy')
#============================Plotting gaussian fits============================
plt.show()
#============================Trying different fits=============================
if MC:
    par_17,par_18A,par_18B,par_21A,par_21B,par_21C = [],[],[],[],[],[]
    bar = progressbar.ProgressBar(maxval=int(steps_MC))
    I = 0
    bar.start()
    for i in range(int(steps_MC)):
        var_a1,var_a2 = np.random.normal(loc = 1, scale=0.15), np.random.normal(loc = 1, scale=0.15)
        var_m1,var_m2 = np.random.normal(loc = 1, scale=0.15), np.random.normal(loc = 1, scale=0.15) 
        var_s1,var_s2 = np.random.normal(loc = 1, scale=0.15), np.random.normal(loc = 1, scale=0.15)
        try:
            init_val17 = [0.1*var_a1,0.02*var_a2,10*var_m1,-750*var_m2,1000/2.355*var_s1,700/2.355*var_s2]
            init_val18 = [0.2*var_a1,0.05*var_a2,10*var_m1,-1000*var_m2,1000/2.355*var_s1,1200/2.355*var_s2]
            init_val21 = [0.15*var_a1,0.02*var_a2,10*var_m1,-800*var_m2,1000/2.355*var_s1,300/2.355*var_s2]
            
            parameters17, covariance17 = curve_fit(double_Gauss, v17_2[mask17_2], Br_gamma_mean2_b2[0].f[mask17_2],p0=init_val17)
            
            parameters18A, covariance18A = curve_fit(double_Gauss, v18_2[mask18_2], Br_gamma_mean2_b2[1].f[mask18_2],p0=init_val18)
            parameters18B, covariance18B = curve_fit(double_Gauss, v18_2[mask18_2], Br_gamma_mean2_b2[2].f[mask18_2],p0=init_val18)
            
            parameters21A, covariance21A = curve_fit(double_Gauss, v21_2[mask21_2], Br_gamma_mean2_b2[3].f[mask21_2],p0=init_val21)
            parameters21B, covariance21B = curve_fit(double_Gauss, v21_2[mask21_2], Br_gamma_mean2_b2[4].f[mask21_2],p0=init_val21)
            parameters21C, covariance21C = curve_fit(double_Gauss, v21_2[mask21_2], Br_gamma_mean2_b2[5].f[mask21_2],p0=init_val21)
            
            if parameters17[0]>=0 and parameters17[1]>=0:
                par_17.append(parameters17)
            if parameters18A[0]>=0 and parameters18A[1]>=0:
                par_18A.append(parameters18A)
                par_18B.append(parameters18B)
            if parameters21A[0]>=0 and parameters21A[1]>=0:
                par_21A.append(parameters21A)
                par_21B.append(parameters21B)
                par_21C.append(parameters21C)
        except: pass
        I += 1
        bar.update(I)
    bar.finish()    
    
    par_17, par_18A, par_18B, par_21A, par_21B, par_21C = np.vstack(par_17),np.vstack(par_18A),np.vstack(par_18B),np.vstack(par_21A),np.vstack(par_21B),np.vstack(par_21C)

    parameters17 = np.median(par_17,axis=0)
    parameters18A = np.median(par_18A,axis=0)
    parameters18B = np.median(par_18B,axis=0)
    parameters21A = np.median(par_21A,axis=0)
    parameters21B = np.median(par_21B,axis=0)
    parameters21C = np.median(par_21C,axis=0)
    
    print('17',parameters17,'\n',np.std((par_17),axis=0)[:-2],'\n',np.std(abs(par_17),axis=0)[-2:])
    print('18A',parameters18A,'\n',np.std((par_18A),axis=0)[:-2],'\n',np.std(abs(par_18A),axis=0)[-2:])
    print('18B',parameters18B,'\n',np.std((par_18B),axis=0)[:-2],'\n',np.std(abs(par_18B),axis=0)[-2:])
    print('21A',parameters21A,'\n',np.std((par_21A),axis=0)[:-2],'\n',np.std(abs(par_21A),axis=0)[-2:])
    print('21B',parameters21B,'\n',np.std((par_21B),axis=0)[:-2],'\n',np.std(abs(par_21B),axis=0)[-2:])
    print('21C',parameters21C,'\n',np.std((par_21C),axis=0)[:-2],'\n',np.std(abs(par_21C),axis=0)[-2:])
    
#%%
############################ Fitting double peak ##############################
#############################Double gaussian fits##############################

### New binning for 2018B and 2021C ###
os.chdir('/home/carlos/Desktop/MSc/TFM/Reduced_spectra/MolecfitCorrected-20220209T161944Z-001/MolecfitCorrected')
obs_days = os.listdir()
Br_gamma_badSNR_b5 = molly.rmolly('average_spectra_B5_BrGamma_interpolated_bad_SNR.mol')
v5_18,v5_21 = vel_redshit(Br_gamma_badSNR_b5[0].wave),vel_redshit(Br_gamma_badSNR_b5[1].wave)
plt.close('all')


fig,ax = plt.subplots(3,2,sharey=True,sharex=True,figsize=(10,14), constrained_layout=True)

[ax1,ax4],[ax2,ax5],[ax3,ax6] = ax
'''
ax1.axvline(vel_redshit(HeII),linestyle='-.',c='k')
ax2.axvline(vel_redshit(HeII),linestyle='-.',c='k')
ax3.axvline(vel_redshit(HeII),linestyle='-.',c='k')
ax4.axvline(vel_redshit(HeII),linestyle='-.',c='k')
ax5.axvline(vel_redshit(HeII),linestyle='-.',c='k')
ax6.axvline(vel_redshit(HeII),linestyle='-.',c='k')
ax1.text(vel_redshit(HeII-100),1.26,'He II'),ax4.text(vel_redshit(HeII-100),1.26,'He II')
'''
ax1.axvline(0,c='grey',linestyle='--'),ax1.axhline(1,c='grey',linestyle='--')
ax2.axvline(0,c='grey',linestyle='--'),ax2.axhline(1,c='grey',linestyle='--')
ax3.axvline(0,c='grey',linestyle='--'),ax3.axhline(1,c='grey',linestyle='--')
ax4.axvline(0,c='grey',linestyle='--'),ax4.axhline(1,c='grey',linestyle='--')
ax5.axvline(0,c='grey',linestyle='--'),ax5.axhline(1,c='grey',linestyle='--')
ax6.axvline(0,c='grey',linestyle='--'),ax6.axhline(1,c='grey',linestyle='--')

ax1.step(v17_2,Br_gamma_mean2_b2[0].f,colors[0],label ='Average 2017'),ax1.set_title(years[0])
ax2.step(v18_2,Br_gamma_mean2_b2[1].f,colors[1],label ='Average 2018A'),ax2.set_title(years[1])
ax3.step(v18_2,Br_gamma_mean2_b2[2].f,colors[2],label ='Average 2018B',linewidth=0.5),ax3.set_title(years[2])
ax3.step(v5_18,Br_gamma_badSNR_b5[0].f,colors[2])
ax4.step(v21_2,Br_gamma_mean2_b2[3].f,colors[3],label ='Average 2021A'),ax4.set_title(years[3])
ax5.step(v21_2,Br_gamma_mean2_b2[4].f,colors[4],label ='Average 2021B'),ax5.set_title(years[4])
ax6.step(v21_2,Br_gamma_mean2_b2[5].f,colors[5],label ='Average 2021C',linewidth=0.5),ax6.set_title(years[5])
ax6.step(v5_21,Br_gamma_badSNR_b5[1].f,colors[5])


fig.supxlabel(r'Velocity [km/s]')
ax1.set_ylim(0.95,1.25),ax1.set_xlim(-3000,3000)
init_dp_val17 = [0.1,0.1,-150,110,200/2.355]
#init_val18 = [0.2,0.05,0,-600,1000/2.355,500/2.355]
init_dp_val18A = [0.16,0.22,-137,137,280/2.355]
init_dp_val18B = [0.15,0.15,-160,180,200/2.355]
init_dp_val21A = [0.13,0.13,-180,130,225/2.355]
init_dp_val21B = [0.15,0.19,-100,100,270/2.355]
init_dp_val21C = [0.11,0.14,-240,150,210/2.355]

#==================== Double peak function parameters fixed ===================
fit17_s2_exp,fit18A_s2_exp = fit17_s2,fit18A_s2
#==================== Double peak function parameters fixed ===================
dp17 = lambda x,A1,A2,m1,m2,s: double_peak(x,A1,A2,fit17_A2,m1,m2,fit17_m2,s,fit17_s2_exp)
dp18A = lambda x,A1,A2,m1,m2,s: double_peak(x,A1,A2,fit18A_A2,m1,m2,fit18A_m2,s,fit18A_s2_exp)
dG18B = lambda x,A1,A2,m1,m2,s: double_Gauss(x,A1,A2,m1,m2,s,s)
dp21A = lambda x,A1,A2,m1,m2,s: double_peak(x,A1,A2,fit21A_A2,m1,m2,fit21A_m2,s,fit21A_s2)
dp21B = lambda x,A1,A2,m1,m2,s: double_peak(x,A1,A2,fit21B_A2,m1,m2,fit21B_m2,s,fit21B_s2)
dG21C = lambda x,A1,A2,m1,m2,s: double_Gauss(x,A1,A2,m1,m2,s,s)

dpAv = lambda x,A1,A2,m1,m2,s: double_peak(x,A1,A2,fitAv_A2,m1,m2,fitAv_m2,s,fitAv_s2)

#==================================Curve fit=================================== 
mask17_2,mask18_2,mask21_2 = (v17_2>-3000) & (v17_2<1500),(v18_2>-3000) & (v18_2<1500),(v21_2>-3000) & (v21_2<1500)
parameters17, covariance17 = curve_fit(dp17, v17_2[mask17_2], Br_gamma_mean2_b2[0].f[mask17_2],p0=init_dp_val17)
fit17_dp_A1,fit17_dp_A2,fit17_dp_m1,fit17_dp_m2,fit17_dp_s = parameters17

parameters18A, covariance18A = curve_fit(dp18A,v18_2[mask18_2], Br_gamma_mean2_b2[1].f[mask18_2],p0=init_dp_val18A)
parameters18B, covariance18B = curve_fit(dG18B, v18_2[mask18_2], Br_gamma_mean2_b2[2].f[mask18_2],p0=init_dp_val18B)
fit18A_dp_A1,fit18A_dp_A2,fit18A_dp_m1,fit18A_dp_m2,fit18A_dp_s = parameters18A
fit18B_dp_A1,fit18B_dp_A2,fit18B_dp_m1,fit18B_dp_m2,fit18B_dp_s = parameters18B

parameters21A, covariance21A = curve_fit(dp21A, v21_2[mask21_2], Br_gamma_mean2_b2[3].f[mask21_2],p0=init_dp_val21A)
parameters21B, covariance21B = curve_fit(dp21B, v21_2[mask21_2], Br_gamma_mean2_b2[4].f[mask21_2],p0=init_dp_val21B)
parameters21C, covariance21C = curve_fit(dG21C, v21_2[mask21_2], Br_gamma_mean2_b2[5].f[mask21_2],p0=init_dp_val21C)
fit21A_dp_A1,fit21A_dp_A2,fit21A_dp_m1,fit21A_dp_m2,fit21A_dp_s = parameters21A
fit21B_dp_A1,fit21B_dp_A2,fit21B_dp_m1,fit21B_dp_m2,fit21B_dp_s = parameters21B
fit21C_dp_A1,fit21C_dp_A2,fit21C_dp_m1,fit21C_dp_m2,fit21C_dp_s = parameters21C

parametersAv, covarianceAv = curve_fit(dpAv, vmean[maskAv],Br_gamma_b2[mean[0]].f[maskAv],p0=init_dp_val21A)
fitAv_dp_A1,fitAv_dp_A2,fitAv_dp_m1,fitAv_dp_m2,fitAv_dp_s = parametersAv

#==================================Curve fit=================================== 

#============================Plotting gaussian fits============================
ax1.plot(vmean[maskAv],dpAv(vmean[maskAv],*parametersAv),'k',linewidth=1)
ax2.plot(vmean[maskAv],dpAv(vmean[maskAv],*parametersAv),'k',linewidth=1)
ax3.plot(vmean[maskAv],dpAv(vmean[maskAv],*parametersAv),'k',linewidth=1)
ax4.plot(vmean[maskAv],dpAv(vmean[maskAv],*parametersAv),'k',linewidth=1)
ax5.plot(vmean[maskAv],dpAv(vmean[maskAv],*parametersAv),'k',linewidth=1)
ax6.plot(vmean[maskAv],dpAv(vmean[maskAv],*parametersAv),'k',linewidth=1)


# 2017
ax1.plot(vmean,Gauss(vmean,fit17_dp_A1,fit17_dp_m1,fit17_dp_s),'b--',linewidth=0.7)
ax1.plot(vmean,Gauss(vmean,fit17_dp_A2,fit17_dp_m2,fit17_dp_s),'b--',linewidth=0.7)
ax1.plot(vmean,Gauss(vmean,fit17_A2,fit17_m2,fit17_s2_exp),'b--',linewidth=0.7)
ax1.plot(v17,dp17(v17,*parameters17),'blue')
# 2018
ax2.plot(vmean,Gauss(vmean,fit18A_dp_A1,fit18A_dp_m1,fit18A_dp_s),'b--',linewidth=0.7)
ax2.plot(vmean,Gauss(vmean,fit18A_dp_A2,fit18A_dp_m2,fit18A_dp_s),'b--',linewidth=0.7)
ax2.plot(vmean,Gauss(vmean,fit18A_A2,fit18A_m2,fit18A_s2_exp),'b--',linewidth=0.7)
ax2.plot(v18,dp18A(v18,*parameters18A),'blue')
ax3.plot(vmean,Gauss(vmean,fit18B_dp_A1,fit18B_dp_m1,fit18B_dp_s),'b--',linewidth=0.7)
ax3.plot(vmean,Gauss(vmean,fit18B_dp_A2,fit18B_dp_m2,fit18B_dp_s),'b--',linewidth=0.7)
ax3.plot(v18,dG18B(v18,*parameters18B),'blue')
# 2021
ax4.plot(v21,Gauss(v21,fit21A_dp_A1,fit21A_dp_m1,fit21A_dp_s),'b--',linewidth=0.7)
ax4.plot(v21,Gauss(v21,fit21A_dp_A2,fit21A_dp_m2,fit21A_dp_s),'b--',linewidth=0.7)
ax4.plot(v21,Gauss(v21,fit21A_A2,fit21A_m2,fit21A_s2),'b--',linewidth=0.7)
ax4.plot(v21,dp21A(v21,*parameters21A),'blue')
ax5.plot(v21,Gauss(v21,fit21B_dp_A1,fit21B_dp_m1,fit21B_dp_s),'b--',linewidth=0.7)
ax5.plot(v21,Gauss(v21,fit21B_dp_A2,fit21B_dp_m2,fit21B_dp_s),'b--',linewidth=0.7)
ax5.plot(v21,Gauss(v21,fit21B_A2,fit21B_m2,fit21B_s2),'b--',linewidth=0.7)
ax5.plot(v21,dp21B(v21,*parameters21B),'blue')
ax6.plot(v21,Gauss(v21,fit21C_dp_A1,fit21C_dp_m1,fit21C_dp_s),'b--',linewidth=0.7)
ax6.plot(v21,Gauss(v21,fit21C_dp_A2,fit21C_dp_m2,fit21C_dp_s),'b--',linewidth=0.7)
ax6.plot(v21,dG21C(v21,*parameters21C),'blue')




#============================Plotting gaussian fits============================
plt.show()
#============================Trying different fits=============================


plt.savefig(path_img+'double_peak_fit.png')

#%%
#=============================== continuum std ================================
os.chdir('/home/carlos/Desktop/MSc/TFM/Reduced_spectra/MolecfitCorrected-20220209T161944Z-001/MolecfitCorrected')
obs_days = os.listdir()
Br_gamma_badSNR_b5 = molly.rmolly('average_spectra_B5_BrGamma_interpolated_bad_SNR.mol')
v5_18,v5_21 = vel_redshit(Br_gamma_badSNR_b5[0].wave),vel_redshit(Br_gamma_badSNR_b5[1].wave)
plt.close('all')



fig,ax = plt.subplots(3,2,sharey=True,sharex=True,figsize=(10,14), constrained_layout=True)

[ax1,ax4],[ax2,ax5],[ax3,ax6] = ax
'''
ax1.axvline(vel_redshit(HeII),linestyle='-.',c='k')
ax2.axvline(vel_redshit(HeII),linestyle='-.',c='k')
ax3.axvline(vel_redshit(HeII),linestyle='-.',c='k')
ax4.axvline(vel_redshit(HeII),linestyle='-.',c='k')
ax5.axvline(vel_redshit(HeII),linestyle='-.',c='k')
ax6.axvline(vel_redshit(HeII),linestyle='-.',c='k')
ax1.text(vel_redshit(HeII-100),1.26,'He II'),ax4.text(vel_redshit(HeII-100),1.26,'He II')
'''
ax1.set_ylim(0.95,1.25),ax1.set_xlim(-3000,3000)
ax1.axvline(0,c='grey',linestyle='--'),ax1.axhline(1,c='grey',linestyle='--')
ax2.axvline(0,c='grey',linestyle='--'),ax2.axhline(1,c='grey',linestyle='--')
ax3.axvline(0,c='grey',linestyle='--'),ax3.axhline(1,c='grey',linestyle='--')
ax4.axvline(0,c='grey',linestyle='--'),ax4.axhline(1,c='grey',linestyle='--')
ax5.axvline(0,c='grey',linestyle='--'),ax5.axhline(1,c='grey',linestyle='--')
ax6.axvline(0,c='grey',linestyle='--'),ax6.axhline(1,c='grey',linestyle='--')

ax1.step(v17_2,Br_gamma_mean2_b2[0].f,colors[0],label ='Average 2017'),ax1.set_title(years[0])
ax2.step(v18_2,Br_gamma_mean2_b2[1].f,colors[1],label ='Average 2018A'),ax2.set_title(years[1])
ax3.step(v18_2,Br_gamma_mean2_b2[2].f,colors[2],label ='Average 2018B',linewidth=0.5),ax3.set_title(years[2])
ax3.step(v5_18,Br_gamma_badSNR_b5[0].f,colors[2])
ax4.step(v21_2,Br_gamma_mean2_b2[3].f,colors[3],label ='Average 2021A'),ax4.set_title(years[3])
ax5.step(v21_2,Br_gamma_mean2_b2[4].f,colors[4],label ='Average 2021B'),ax5.set_title(years[4])
ax6.step(v21_2,Br_gamma_mean2_b2[5].f,colors[5],label ='Average 2021C',linewidth=0.5),ax6.set_title(years[5])
ax6.step(v5_21,Br_gamma_badSNR_b5[1].f,colors[5])

#============================Plotting gaussian fits============================
ax1.plot(vmean[maskAv],dpAv(vmean[maskAv],*parametersAv),'k',linewidth=1)
ax2.plot(vmean[maskAv],dpAv(vmean[maskAv],*parametersAv),'k',linewidth=1)
ax3.plot(vmean[maskAv],dpAv(vmean[maskAv],*parametersAv),'k',linewidth=1)
ax4.plot(vmean[maskAv],dpAv(vmean[maskAv],*parametersAv),'k',linewidth=1)
ax5.plot(vmean[maskAv],dpAv(vmean[maskAv],*parametersAv),'k',linewidth=1)
ax6.plot(vmean[maskAv],dpAv(vmean[maskAv],*parametersAv),'k',linewidth=1)

# 2017
ax1.plot(vmean,Gauss(vmean,fit17_dp_A1,fit17_dp_m1,fit17_dp_s),'b--',linewidth=0.7)
ax1.plot(vmean,Gauss(vmean,fit17_dp_A2,fit17_dp_m2,fit17_dp_s),'b--',linewidth=0.7)
ax1.plot(vmean,Gauss(vmean,fit17_A2,fit17_m2,fit17_s2_exp),'b--',linewidth=0.7)
ax1.plot(v17,dp17(v17,*parameters17),'blue')
# 2018
ax2.plot(vmean,Gauss(vmean,fit18A_dp_A1,fit18A_dp_m1,fit18A_dp_s),'b--',linewidth=0.7)
ax2.plot(vmean,Gauss(vmean,fit18A_dp_A2,fit18A_dp_m2,fit18A_dp_s),'b--',linewidth=0.7)
ax2.plot(vmean,Gauss(vmean,fit18A_A2,fit18A_m2,fit18A_s2_exp),'b--',linewidth=0.7)
ax2.plot(v18,dp18A(v18,*parameters18A),'blue')
ax3.plot(vmean,Gauss(vmean,fit18B_dp_A1,fit18B_dp_m1,fit18B_dp_s),'b--',linewidth=0.7)
ax3.plot(vmean,Gauss(vmean,fit18B_dp_A2,fit18B_dp_m2,fit18B_dp_s),'b--',linewidth=0.7)
ax3.plot(v18,dG18B(v18,*parameters18B),'blue')
# 2021
ax4.plot(v21,Gauss(v21,fit21A_dp_A1,fit21A_dp_m1,fit21A_dp_s),'b--',linewidth=0.7)
ax4.plot(v21,Gauss(v21,fit21A_dp_A2,fit21A_dp_m2,fit21A_dp_s),'b--',linewidth=0.7)
ax4.plot(v21,Gauss(v21,fit21A_A2,fit21A_m2,fit21A_s2),'b--',linewidth=0.7)
ax4.plot(v21,dp21A(v21,*parameters21A),'blue')
ax5.plot(v21,Gauss(v21,fit21B_dp_A1,fit21B_dp_m1,fit21B_dp_s),'b--',linewidth=0.7)
ax5.plot(v21,Gauss(v21,fit21B_dp_A2,fit21B_dp_m2,fit21B_dp_s),'b--',linewidth=0.7)
ax5.plot(v21,Gauss(v21,fit21B_A2,fit21B_m2,fit21B_s2),'b--',linewidth=0.7)
ax5.plot(v21,dp21B(v21,*parameters21B),'blue')
ax6.plot(v21,Gauss(v21,fit21C_dp_A1,fit21C_dp_m1,fit21C_dp_s),'b--',linewidth=0.7)
ax6.plot(v21,Gauss(v21,fit21C_dp_A2,fit21C_dp_m2,fit21C_dp_s),'b--',linewidth=0.7)
ax6.plot(v21,dG21C(v21,*parameters21C),'blue')

#============================Plotting gaussian fits============================
plt.show()
#============================Trying different fits=============================

cross,axis = [],[]
ContLim_17,ContLim_18A,ContLim_18B,ContLim_21A,ContLim_21B,ContLim_21C = [ax1],[ax2],[ax3],[ax4],[ax5],[ax6]
cont_selection, cont_lim_line = [],[]
from matplotlib.backend_bases import MouseButton


def on_move(event):
    if event.inaxes:
        try: 
            cross[-1][0].remove(),cross[-1][1].remove()
        except: pass
        AX = event.inaxes  # the axes instance
        CRS = AX.axvline(event.xdata,c='r',alpha=0.4),AX.axhline(event.ydata,c='r',alpha=0.4)
        
        cross.append(CRS), axis.append(AX)
        plt.draw()
def on_click(event):
    if event.button is MouseButton.LEFT and event.dblclick:
        AX = event.inaxes
        print(AX.title.get_text()[2:],event.xdata)
        lim = AX.axvline(event.xdata,c='r')
        plt.draw()
        # Continuum selection
        globals()['ContLim_'+AX.title.get_text()[2:]].append(event.xdata)
        cont_lim_line.append([AX,event.xdata,lim])
        cont_lim = globals()['ContLim_'+AX.title.get_text()[2:]]
        if len(cont_lim)%2==1:
            cont = AX.plot(cont_lim[-2:],[1.05,1.05],'b'),AX.plot([cont_lim[-2],cont_lim[-2]],[1.04,1.06],'b'),AX.plot([cont_lim[-1],cont_lim[-1]],[1.04,1.06],'b')
            cont_selection.append([AX,cont_lim[-2],cont_lim[-1],cont])
    elif event.button is MouseButton.MIDDLE:
        for k in range(len(years)):
            j = years[k][2:]
            msc = globals()['ContLim_'+j]
            globals()['M'+j] = False
            try:
                for i in range(1,len(msc)):
                    if i%2==1:
                        m = 0
                        if msc[i]<msc[i+1]:
                            m = globals()['v'+j[:2]+'_2'] > msc[i]
                            m &= globals()['v'+j[:2]+'_2'] < msc[i+1]
                        else: 
                            m = globals()['v'+j[:2]+'_2'] < msc[i]
                            m &= globals()['v'+j[:2]+'_2'] > msc[i+1]
                        globals()['M'+j] += m
                print(1-np.std(Br_gamma_mean2_b2[k].f[globals()['M'+j]]))
                msc[0].axhline(1-np.std(Br_gamma_mean2_b2[k].f[globals()['M'+j]]),ls='--',c='grey')
                msc[0].axhline(1+np.std(Br_gamma_mean2_b2[k].f[globals()['M'+j]]),ls='--',c='grey')
                plt.draw()
            except: print('tried ',j)
    elif event.button is MouseButton.RIGHT and not event.dblclick:
        AX = event.inaxes
        idx, val = find_nearest(globals()['ContLim_'+AX.title.get_text()[2:]][1:], event.xdata)
        globals()['ContLim_'+AX.title.get_text()[2:]].remove(val)
        cont_lim_line_arr = np.vstack(cont_lim_line)
        cont_lim_line_arr[(cont_lim_line_arr[:,0]==AX) & (cont_lim_line_arr[:,1]==val)][0][2].remove()
        cont_selection_arr = np.vstack(cont_selection)
        try: 
            cont_selection_arr[(cont_selection_arr[:,0]==AX) & (cont_selection_arr[:,1]==val)][0][-1][0][0].remove()
            cont_selection_arr[(cont_selection_arr[:,0]==AX) & (cont_selection_arr[:,1]==val)][0][-1][1][0].remove()
            cont_selection_arr[(cont_selection_arr[:,0]==AX) & (cont_selection_arr[:,1]==val)][0][-1][2][0].remove()
        except:
            try: 
                cont_selection_arr[(cont_selection_arr[:,0]==AX) & (cont_selection_arr[:,2]==val)][0][-1][0][0].remove()
                cont_selection_arr[(cont_selection_arr[:,0]==AX) & (cont_selection_arr[:,2]==val)][0][-1][1][0].remove()
                cont_selection_arr[(cont_selection_arr[:,0]==AX) & (cont_selection_arr[:,2]==val)][0][-1][2][0].remove()
            except: pass
    elif event.button is MouseButton.RIGHT and event.dblclick:
        print('disconnecting callback')
        try: 
            cross[-1][0].remove(),cross[-1][1].remove()
        except: pass
        plt.disconnect(binding_id)
        
binding_id = plt.connect('button_press_event', on_click)
plt.connect('motion_notify_event', on_move)

#=============================== continuum std ================================








'''
#%%
###########Group study to see the representative the mean value is#############
plt.close('all')

init_val17 = [0.1,0.05,0,-600,1000/2.355,500/2.355]
init_val18 = [0.2,0.05,0,-600,1000/2.355,500/2.355]
init_val21 = [0.15,0.05,0,-600,1000/2.355,500/2.355]
init_valAv = [0.1,0.08,0,-600,1000/2.355,500/2.355]
#==================================Curve fit=================================== 
mask17,mask18,mask21,maskAv = (v17>-2000) & (v17<1500),(v18>-3000) & (v18<1500),(v21>-3000) & (v21<1500),(vmean>-2000) & (vmean<1500)
par_17,par_18,par_21 = [],[],[]
for i in range(len(Br_gamma_b2)):
    if i<4:
        p17, c17 = curve_fit(double_Gauss, v17[mask17], Br_gamma_b2[i].f[mask17],p0=parameters17)
        par_17.append(p17)
    elif i<9:
        p18, c18 = curve_fit(double_Gauss, v18[mask18], Br_gamma_b2[i].f[mask18],p0=parameters18)
        par_18.append(p18)
    elif i<14:
        p21, c21 = curve_fit(double_Gauss, v21[mask21], Br_gamma_b2[i].f[mask21],p0=parameters21)
        par_21.append(p21)
#==================================Curve fit=================================== 
par_17, par_18, par_21 = np.vstack(par_17),np.vstack(par_18),np.vstack(par_21)

#print(par_17[:,:2],'\n',par_18[:,:2],'\n',par_21[:,:2])
fig,axs = plt.subplots(1,4,sharex=True,sharey=True)
fig.suptitle('2017'),fig.supxlabel(r'Velocity [km/s]')
for i in range(0,4):
    t = Time(Br_gamma_b2[i].head['HJD'],format='jd')
    axs[i].set_title('Used '+t.iso[5:10])
    axs[i].axvline(0,linestyle='--',c='grey'),axs[i].axhline(1,linestyle='--',c='grey')
    axs[i].step(v17[mask17], Br_gamma_b2[i].f[mask17])
    axs[i].plot(v17[mask17],double_Gauss(v17[mask17],fit17_A1,fit17_A2,fit17_m1,fit17_m2,fit17_s1,fit17_s2),'k',linewidth=0.5)
    axs[i].plot(v17[mask17],Gauss(v17[mask17],par_17[i,0],par_17[i,2],par_17[i,4]),'b--',linewidth=0.7)
    axs[i].plot(v17[mask17],Gauss(v17[mask17],par_17[i,1],par_17[i,3],par_17[i,5]),'b--',linewidth=0.7)
    axs[i].plot(v17[mask17],double_Gauss(v17[mask17],*par_17[i]),'r',linewidth=0.5)
plt.ylim(ymin=0.95)
fig,axs = plt.subplots(1,5,sharex=True,sharey=True)
plt.suptitle('2018'),fig.supxlabel(r'Velocity [km/s]')
for i in range(4,9):
    t = Time(Br_gamma_b2[i].head['HJD'],format='jd')
    if i==6 or i==7 or i==8:
        axs[i-4].set_title(t.iso[5:10])
    else:
        axs[i-4].set_title('Used '+t.iso[5:10])
    axs[i-4].axvline(0,linestyle='--',c='grey'),axs[i-4].axhline(1,linestyle='--',c='grey')
    axs[i-4].step(v18[mask18], Br_gamma_b2[i].f[mask18])
    axs[i-4].plot(v18[mask18],double_Gauss(v18[mask18],fit18_A1,fit18_A2,fit18_m1,fit18_m2,fit18_s1,fit18_s2),'k',linewidth=0.5)
    axs[i-4].plot(v18[mask18],Gauss(v18[mask18],par_18[i-4,0],par_18[i-4,2],par_18[i-4,4]),'b--',linewidth=0.7)
    axs[i-4].plot(v18[mask18],Gauss(v18[mask18],par_18[i-4,1],par_18[i-4,3],par_18[i-4,5]),'b--',linewidth=0.7)
    axs[i-4].plot(v18[mask18],double_Gauss(v18[mask18],*par_18[i-4]),'r',linewidth=0.5)
plt.ylim(ymin=0.95)
fig,axs = plt.subplots(1,5,sharex=True,sharey=True)
plt.suptitle('2021'),fig.supxlabel(r'Velocity [km/s]')
for i in range(9,14):
    t = Time(Br_gamma_b2[i].head['HJD'],format='jd')
    if i==12 or i==13:
       axs[i-9].set_title(t.iso[5:10])
    else:
       axs[i-9].set_title('Used '+t.iso[5:10])
    axs[i-9].axvline(0,linestyle='--',c='grey'),axs[i-9].axhline(1,linestyle='--',c='grey')   
    axs[i-9].step(v21[mask21], Br_gamma_b2[i].f[mask21])
    axs[i-9].plot(v21[mask21],double_Gauss(v21[mask21],fit21_A1,fit21_A2,fit21_m1,fit21_m2,fit21_s1,fit21_s2),'k',linewidth=0.5)
    axs[i-9].plot(v21[mask21],Gauss(v21[mask21],par_21[i-9,0],par_21[i-9,2],par_21[i-9,4]),'b--',linewidth=0.7)
    axs[i-9].plot(v21[mask21],Gauss(v21[mask21],par_21[i-9,1],par_21[i-9,3],par_21[i-9,5]),'b--',linewidth=0.7)
    axs[i-9].plot(v21[mask21],double_Gauss(v17[mask21],*par_21[i-9]),'r',linewidth=0.5)    
plt.ylim(ymin=0.95)
'''
#%%
K_band = pd.read_csv('K_spec.txt',sep = '\t',header=1, names=['wave','trans'])

plt.figure(constrained_layout=True)
plt.plot(K_band['wave'],K_band['trans'])
plt.ylabel('Transmittance [%]',fontsize=18), plt.xlabel(r'Wavelength $[\mu {\rm m}]$',fontsize=18)
plt.grid(),plt.title(r'K$_{\rm spec}$',fontsize=18)
plt.ylim(ymin=0),plt.xlim(min(K_band['wave']),max(K_band['wave']))
plt.tick_params('both',labelsize=15)
plt.savefig(path_img+'Kspec.png')


#%%
fig, axs = plt.subplots(constrained_layout=True)
for da in range(len(Br_gamma_mean2_b2)):
    d=Br_gamma_mean2_b2[da].head['HJD']-2400000
    if lin==True: 
        axs.axvline(d,c=colors[da],label=years[da],linestyle='-.')
    else:
        try:
            axs.plot(d,Y_MAXI[da],'*',c=colors[da],label=years[da],markeredgewidth=3)
        except:
            axs.axvline(d,c=colors[da],label=years[da],linestyle='-.')
axs.errorbar(MAXI['#MJD'],MAXI['2_20keV'],yerr = MAXI['e_2_20keV'],fmt='.',markersize = 0.5,c = 'k')
axs.set_xlabel('MJD [days]',fontsize=18),axs.set_ylabel(r'2-20 keV [Photons cm$^{-2}$s$^{-1}]$',fontsize=18)
axs.set_xlim(57750,59700)
axs.grid(),axs.legend(loc='upper right')
plt.tick_params('both',labelsize=15)
plt.savefig(path_img+'MAXI_EMIR.png')
'''
#%%

plt.figure()
mask = (MAXI['#MJD']>59300) & (MAXI['#MJD']<59500)
plt.plot(MAXI['4_10keV'][mask]/MAXI['2_4keV'][mask],MAXI['2_20keV'][mask])
plt.xscale('log')
plt.yscale('log')
'''