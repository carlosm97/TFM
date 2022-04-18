#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 19:37:54 2021

@author: Martínez-Sebastián

GOAL : To do photometry from both observationals campaigns 
"""

EMIR_scale = 0.2 #"/pixel. According to user manua. 


from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

#from numina.array.display.ximshow import ximshow
#from numina.array.display.ximplotxy import ximplotxy
import os
from scipy import stats, optimize
from astropy.stats import biweight_location, mad_std
import time
from astropy.stats import sigma_clipped_stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib import rc
import progressbar
from scipy.interpolate import griddata
import pandas as pd
os.chdir('/home/carlos/Desktop/MSc/TFM/GRS_1915+105_EMIR')
from Photometry_function import radius_selection, Photometry, sel_im, clear_image,Forced_Photometry,plotting_multiple_stars,plot_photometry
from astropy.time import Time
import pandas as pd

'''
runcell('-----------------------------------2021-------------------------------------', '/home/carlos/Desktop/MSc/TFM/GRS_1915+105_EMIR/Photometry.py')
runcell('Using "good" images:', '/home/carlos/Desktop/MSc/TFM/GRS_1915+105_EMIR/Photometry.py')
'''

#%% FF combined

runcell(0, '/home/carlos/Desktop/MSc/TFM/GRS_1915+105_EMIR/Photometry.py')
FF_list = os.listdir('./GTC55-21A.OB0001a_FlatsFiller/GTC55-21A/OB0001a/flat')
if 'master_flat_21.fits' not in FF_list:
    bar = progressbar.ProgressBar(maxval=len(FF_list)).start()
    i = 0
    FF_all = [] #np.empty([len(FF_list),2048,2048])
    for f in FF_list: 
        flat = fits.open('./GTC55-21A.OB0001a_FlatsFiller/GTC55-21A/OB0001a/flat/'+\
                              f)
        FF_all.append(flat[0].data.copy())
        flat.close()
        del flat
        i+=1
        bar.update(i)
    del FF_list
    bar.finish()
    FF_all = np.asarray(FF_all)
    # Combining 300 FF to get our master_flat
    n=0
    m = []
    while n<len(FF_all)/30:
        med = np.median(FF_all[30*n:(n+1)*30,:,:], axis=0)
        m.append(med)
        del med
        n+=1
        print(n)
    m = np.asarray(m)
    med = np.median(m,axis=0)
    del m, FF_all
    FF_clean = med/np.mean(med)
    del med
    #plt.subplots()
    #plt.imshow(FF_clean)
    hdu_sp = fits.PrimaryHDU(data=FF_clean)
    hdulFinal = fits.HDUList([hdu_sp])
    hdulFinal.writeto('./GTC55-21A.OB0001a_FlatsFiller/GTC55-21A/OB0001a/flat/master_flat_21.fits')
    del hdu_sp, hdulFinal
else: 
    # Reading master flat if it already exists
    print('Masterflat already exists')
    hdu=fits.open('./GTC55-21A.OB0001a_FlatsFiller/GTC55-21A/OB0001a/flat/master_flat_21.fits')
    FF_clean=hdu[0].data.copy()
    FF_clean=FF_clean/np.median(FF_clean)
    hdu.close
    del hdu, FF_list
'''
hdu = fits.PrimaryHDU(data=data,header=head[1])
hdulFinal = fits.HDUList([hdu])
hdulFinal.writeto('../proof_image.fits')
'''
# <codecell> 
'''SELECTION OF IMAGESS'''
plt.close('all')
# Different days images:
OB1 = sel_im('./GRS1915_K_21A_01_OB0073/data')
OB2 = sel_im('./GRS1915_K_21A_02_OB0074/data')
OB3 = sel_im('./GRS1915_K_21A_03_OB0075/data')
OB4 = sel_im('./GRS1915_K_21A_04_OB0076/data')
OB5 = sel_im('./GRS1915_K_21A_05_OB0078/data')
Im_list = OB1 + OB2 + OB3 + OB4 + OB5
del Im_list
Im_list = sel_im('./PHOTOMETRY_21')
print(Im_list)

# First step: to do a cleaning just selecting real images of our target field. 

TEST = False
if TEST==True:
    for image in Im_list:
        dat = clear_image(image, FF_clean)
        plt.figure()
        plt.title(image[-41:-22])
        plt.imshow(dat,cmap='Greys_r',origin='lower',vmin=np.nanmedian(dat),vmax=2*np.nanmedian(dat))
        plt.colorbar()
        del dat
# After looking with Aladin, we conclude that all images are valid for the period
# 21A. 

# Importing seeing (previously observed) as a dictionary:
df = pd.read_csv("FWHM_21.txt", sep=" ")
df = np.asarray(df)
dic = {}
for val in df:
    if val[0]!='IMAGE':
        dic[val[0]] = np.mean(val[1:])
del df

#%%-----------------------------------2021------------------------------------- 
#3 selected stars with APPERTURE:=1.5*FWHM
# Reference: m_19151073_1057181, m_19151313_1057000, m_19151430_1056079
runcell('FF combined', '/home/carlos/Desktop/MSc/TFM/GRS_1915+105_EMIR/Photometry.py')
runcell(2, '/home/carlos/Desktop/MSc/TFM/GRS_1915+105_EMIR/Photometry.py')
# We delete bad images. 

# The desired image order is: 
'''
'./PHOTOMETRY_21/0003137768-20210918-EMIR-STARE_IMAGE.fits'
'./PHOTOMETRY_21/0003137764-20210918-EMIR-STARE_IMAGE.fits'
'./PHOTOMETRY_21/0003076375-20210819-EMIR-STARE_IMAGE.fits'
'./PHOTOMETRY_21/0003100872-20210825-EMIR-STARE_IMAGE.fits'
'./PHOTOMETRY_21/0003076371-20210819-EMIR-STARE_IMAGE.fits' # 1915 1430  is not well centered 
'./PHOTOMETRY_21/0003108037-20210826-EMIR-STARE_IMAGE.fits'
'./PHOTOMETRY_21/0003097002-20210824-EMIR-STARE_IMAGE.fits'
'''

Im_list.remove('./PHOTOMETRY_21/0003137760-20210918-EMIR-STARE_IMAGE.fits')
Im_list.remove('./PHOTOMETRY_21/0003076367-20210819-EMIR-STARE_IMAGE.fits')

# We convert from Vega to AB system to easily get the flux
#m_19151073_1057181 = 14.262+1.85 # 1
m_19151073_1057181 = 13.782+1.85 # 1
m_19151313_1057000 = 13.202+1.85 # 3 
m_19151430_1056079 = 12.675+1.85 # 4
# Position of the selected stars:
POSITIONS = [[(1214.,1330.)],[(1083,1315.)],[(1272.,1199.)],[(1495,1377)]],\
    [[(1210.,1323.)],[(1079.,1308.)],[(1268.,1191.)],[(1491,1369)]],\
    [[(1214.,1330.)],[(1083.,1314.)],[(1271.,1197.)],[(1495,1376)]],\
    [[(1214.,1329.)],[(1083.,1314.)],[(1272.,1198.)],[(1495,1375)]],\
    [[(1214.,1329.)],[(1083.,1314.)],[(1272.,1198.)],[(1494,1375)]],\
    [[(1214.,1329.)],[(1083.,1314.)],[(1271.,1198.)],[(1494,1371)]],\
    [[(1210.,1028.)],[(1079.,1014.)],[(1268.,897.)],[(1489,1073)]]
#[[(1000.,1040.)],[(870.,1025.)],[(1059.,909.)],[(1280,1085)]],\
#[[(1016.,1037.)],[(886.,1023.)],[(1074.,906.)],[(1296,1083)]],\

    


mag = m_19151073_1057181, m_19151313_1057000, m_19151430_1056079
dm_GRS1915 = []
dm_19151073_1057181, dm_19151313_1057000, dm_19151430_1056079 =[],[],[]
name_list = 'mean', '19:15:10.73 +10:57:18.1', '19:15:13.13 +10:57:00.0', '19:15:14.30 +10:56:07.9'
names = name_list

DATE,mag_error,sky = [],[],[]

for i in range(len(POSITIONS)):
    pos = POSITIONS[i]
    dat, prj, airmass, hour = clear_image(Im_list[i],FF_clean,projection=True)
    #M,dM,rmin,s= Photometry(dat, pos, mag, names, prj)
    radio = dic[Im_list[i]]*1.5
    M,dM,s= Forced_Photometry(dat, radio/2, pos, mag, names, prj,PLOT=False)
    plot_photometry(dat, 5, [pos[0]], 'GRS1915+105', prj)
    del dat, pos, prj
    dm_GRS1915.append(M[0])
    dm_19151073_1057181.append(M[1]), dm_19151313_1057000.append(M[2]), dm_19151430_1056079.append(M[3])
    mag_error.append(dM)
    #DATES.append(Im_list[i][-30:-22])
    #rad.append(rmin)
    sky.append(s)
    t = Time(hour, format='isot', scale='utc')
    DATE.append(t.mjd)
    
    
# Having an idea of the flattened of flat field
for i in range(len(sky)):
    #print('Sky error=',np.sqrt(sky[i]))
    print(abs(sky[i]-np.mean(sky[i])))

dm_GRS1915 = np.asarray(dm_GRS1915)

dm_19151073_1057181, dm_19151313_1057000, dm_19151430_1056079 = \
    np.asarray(dm_19151073_1057181), np.asarray(dm_19151313_1057000), np.asarray(dm_19151430_1056079)

mag_error = np.asarray(mag_error)
photometry_list = dm_GRS1915, dm_19151073_1057181, dm_19151313_1057000, dm_19151430_1056079
color_list  = 'red','green','blue'
plotting_multiple_stars(DATE,photometry_list,name_list,color_list,error=0.05,dif=False)
plt.figure()
plt.plot(dm_GRS1915,'.')
#del mag_error,photometry_list,DATE,name_list,color_list,dm_GRS1915, dm_19151073_1057181,\
#    dm_19151313_1057000, dm_19151430_1056079,names

FLUX_GRS1915_Jy_21 = 10**(-dm_GRS1915/2.5)*3631 # Jy
FLUX_GRS1915_cgs_21 = FLUX_GRS1915_Jy_21*1e-23 # erg/s/cm**2/Hz
try: os.remove('flux_Jy_GRS1915_21.txt')
except: pass
A = []
for i in range(len(DATE)):
    A.append([DATE[i],FLUX_GRS1915_Jy_21[i]])
with open('flux_Jy_GRS1915_21.txt', 'a') as f:
    f.write(str(A))
f.close()
#%%
#%%-----------------------------------2017------------------------------------- 

# In[Using else's flats in Ks]

runcell(0, '/home/carlos/Desktop/MSc/TFM/GRS_1915+105_EMIR/Photometry.py')
FF_list = []
for i in os.listdir('./FLATS_17'):
    if 'fits' in i:
        FF_list.append(i)    
if 'master_flat_17.fits' not in FF_list:
    # Creating and saving master flat
    bar = progressbar.ProgressBar(maxval=len(FF_list)).start()
    i = 0
    FF_1,FF_2,FF_3,FF_4,FF_5 = [],[],[],[],[]
    for f in FF_list: 
        flat = fits.open('./FLATS_17/'+f)
        #FF_all.append(flat[0].data.copy())
        if i<100:
            FF_1.append(flat[0].data.copy())
        elif i<200:
            if i==100:
                FF1_med=np.median(np.asarray(FF_1),axis=0)
                del FF_1
            FF_2.append(flat[0].data.copy())
        elif i<300:
            if i==200:
                FF2_med=np.median(np.asarray(FF_2),axis=0)
                del FF_2
            FF_3.append(flat[0].data.copy())
        elif i<400:
            if i==300:
                FF3_med=np.median(np.asarray(FF_3),axis=0)
                del FF_3
            FF_4.append(flat[0].data.copy())
        else:
            if i==400:
                FF4_med=np.median(np.asarray(FF_4),axis=0)
                del FF_4
            FF_5.append(flat[0].data.copy())
        flat.close()
        del flat
        i+=1
        bar.update(i)
    FF5_med = np.median(np.asarray(FF_5),axis=0)
    del FF_5
    FF_clean=np.median([FF1_med*100,FF2_med*100,FF3_med*100,FF4_med*100,FF5_med*79],axis=0)
    FF_clean=FF_clean/np.median(FF_clean)
    del FF1_med,FF2_med,FF3_med,FF4_med,FF5_med
    bar.finish()
    del bar, f, i, FF_list
    hdu_sp = fits.PrimaryHDU(data=FF_clean)
    hdulFinal = fits.HDUList([hdu_sp])
    hdulFinal.writeto('./FLATS_17/master_flat_17.fits')
    del hdu_sp, hdulFinal,FF_list
else: 
    # Reading master flat if it already exists
    print('Masterflat already exists')
    hdu=fits.open('./FLATS_17/master_flat_17.fits')
    FF_clean=hdu[0].data.copy()
    FF_clean=FF_clean/np.median(FF_clean)
    hdu.close
    del hdu, FF_list,i
#plt.figure()
#plt.imshow(FF_clean,vmin=0.8,vmax=1.2)
#plt.colorbar()

df = pd.read_csv("./FWHM_18.txt", sep=" ")
df = np.asarray(df)
dic = {}
for val in df:
    if val[0]!='IMAGE':
        dic[val[0]] = np.mean(val[1:])
del df



# In[Images_extraction]
images = os.listdir('./PHOTOMETRY_17')
images.remove('combine.py')
cd = '/home/carlos/Desktop/MSc/TFM/GRS_1915+105_EMIR/'



#%% Using the flat field to study the images:
''' 
im = os.listdir('./PHOTOMETRY_17')
im.remove('0001354640-20170921-EMIR-STARE_IMAGE.fits')
images = []
for i in im:
    images.append('./PHOTOMETRY_17/'+i)
images.remove('./PHOTOMETRY_17/combine.py')
# The desired image order is: 
    
1'./PHOTOMETRY_17/0001303035-20170813-EMIR-STARE_IMAGE.fits',
2'./PHOTOMETRY_17/0001643036-20180724-EMIR-STARE_IMAGE.fits', # Possible bad FF
3'./PHOTOMETRY_17/0001354637-20170921-EMIR-STARE_IMAGE.fits', # Ellongued 
4'./PHOTOMETRY_17/0001303032-20170813-EMIR-STARE_IMAGE.fits',
5'./PHOTOMETRY_17/0001627891-20180716-EMIR-STARE_IMAGE.fits', # HORRIBLE: SKIP 
6'./PHOTOMETRY_17/0001753266-20180829-EMIR-STARE_IMAGE.fits', # HORRIBLE + BAD FF
7'./PHOTOMETRY_17/0001621703-20180705-EMIR-STARE_IMAGE.fits', 
8'./PHOTOMETRY_17/0001303029-20170813-EMIR-STARE_IMAGE.fits',
9'./PHOTOMETRY_17/0001621712-20180705-EMIR-STARE_IMAGE.fits', # BAD FF
10'./PHOTOMETRY_17/0001627903-20180716-EMIR-STARE_IMAGE.fits', # BAD FF
11'./PHOTOMETRY_17/0001627912-20180716-EMIR-STARE_IMAGE.fits', # HORRIBLE + BAD FF
12'./PHOTOMETRY_17/0001627909-20180716-EMIR-STARE_IMAGE.fits', # HORRIBLE
13'./PHOTOMETRY_17/0001627888-20180716-EMIR-STARE_IMAGE.fits', # HORRIBLE
14'./PHOTOMETRY_17/0001303041-20170813-EMIR-STARE_IMAGE.fits',
15'./PHOTOMETRY_17/0001366511-20171001-EMIR-IMAGE.fits', 
16'./PHOTOMETRY_17/0001627915-20180716-EMIR-STARE_IMAGE.fits', # HORRIBLE + BAD FF
17'./PHOTOMETRY_17/0001377890-20171004-EMIR-IMAGE.fits',
18'./PHOTOMETRY_17/0001627894-20180716-EMIR-STARE_IMAGE.fits', # HORRIBLE 
19'./PHOTOMETRY_17/0001558126-20180524-EMIR-STARE_IMAGE.fits',
20'./PHOTOMETRY_17/0001627900-20180716-EMIR-STARE_IMAGE.fits', # BAS SNR
21'./PHOTOMETRY_17/0001716608-20180823-EMIR-STARE_IMAGE.fits', # BAD FF
22'./PHOTOMETRY_17/0001377893-20171004-EMIR-IMAGE.fits',
23'./PHOTOMETRY_17/0001643042-20180724-EMIR-STARE_IMAGE.fits', # BAD FF
24'./PHOTOMETRY_17/0001366505-20171001-EMIR-IMAGE.fits',
25'./PHOTOMETRY_17/0001366508-20171001-EMIR-IMAGE.fits', # SNR meh... 
26'./PHOTOMETRY_17/0001627897-20180716-EMIR-STARE_IMAGE.fits', # HORRIBLE
27'./PHOTOMETRY_17/0001303038-20170813-EMIR-STARE_IMAGE.fits',
28'./PHOTOMETRY_17/0001303044-20170813-EMIR-STARE_IMAGE.fits', # NEAR DARK
29'./PHOTOMETRY_17/0001627906-20180716-EMIR-STARE_IMAGE.fits', # WEAK + BAD FF
30'./PHOTOMETRY_17/0001643039-20180724-EMIR-STARE_IMAGE.fits', # BAD FF
31'./PHOTOMETRY_17/0001621706-20180705-EMIR-STARE_IMAGE.fits', # SNR + FF meh... 
32'./PHOTOMETRY_17/0001716605-20180823-EMIR-STARE_IMAGE.fits', # WEAK + BAD FF 
33'./PHOTOMETRY_17/0001354634-20170921-EMIR-STARE_IMAGE.fits', # ELLONGED
34'./PHOTOMETRY_17/0001716611-20180823-EMIR-STARE_IMAGE.fits', # BAD FF
35'./PHOTOMETRY_17/0001303047-20170813-EMIR-STARE_IMAGE.fits', # SPOT
36'./PHOTOMETRY_17/0001314004-20170902-EMIR-IMAGE.fits', # BAD
37'./PHOTOMETRY_17/0001303050-20170813-EMIR-STARE_IMAGE.fits',
38'./PHOTOMETRY_17/0001621709-20180705-EMIR-STARE_IMAGE.fits'] # FF MEH

POSITIONS =  [[(893,1034)],[(950,904)],[(762,1020)],[(1171,1081)]],\
    [[(1011,1043)],[(1068,911)],[(881,1027)],[(1289,1088)]],\
    [[(1208,1033)],[(1266,902)],[(1076,1018)],[(1487,1081)]],\
    [[(892,1035)],[(951,904)],[(762,1020)],[(1171,1082)]],\
    [[(1032,1028)],[(1091,896)],[(902,1013)],[(1312,1074)]],\
    [[(1214,1030)],[(1272,899)],[(1083,1014)],[(1493,1077)]],\
    [[(1023,1043)],[(1081,912)],[(893,1027)],[(1302,1089)]],\
    [[(973,1035)],[(1030,904)],[(842,1019)],[(1250,1081)]],\
    [[(1215,1030)],[(1272,899)],[(1083,1015)],[(1494,1076)]],\
    [[(1213,1026)],[(1271,895)],[(1083,1011)],[(1496,1076)]],\
    [[(1215,1031)],[(1272,901)],[(1084,1017)],[(1496,1075)]],\
    [[(1215,1031)],[(1272,901)],[(1084,1017)],[(1496,1075)]],\
    [[(1032,1028)],[(1091,896)],[(902,1013)],[(1312,1074)]],\
    [[(1214,1030)],[(1272,901)],[(1083,1015)],[(1493,1078)]],\
    [[(1216,1031)],[(1275,900)],[(1086,1016)],[(1496,1077)]],\
    [[(1214,1030)],[(1273,900)],[(1085,1016)],[(1493,1076)]],\
    [[(999,1020)],[(1057,890)],[(869,1004)],[(1277,1067)]],\
    [[(1028,1028)],[(1088,895)],[(897,1012)],[(1312,1074)]],\
    [[(1216,1032)],[(1273,901)],[(1085,1017)],[(1495,1078)]],\
    [[(1035,1028)],[(1092,897)],[(905,1013)],[(1314,1073)]],\
    [[(1215,1031)],[(1273,900)],[(1084,1016)],[(1494,1078)]],\
    [[(1216,1032)],[(1273,901)],[(1085,1017)],[(1495,1078)]],\
    [[(1215,1030)],[(1272,899)],[(1084,1015)],[(1494,1076)]],\
    [[(1253,1035)],[(1311,904)],[(1123,1020)],[(1533,1082)]],\
    [[(1215,1031)],[(1273,900)],[(1084,1016)],[(1494,1078)]],\
    [[(1034,1030)],[(1090,897)],[(900,1014)],[(1313,1074)]],\
    [[(973,1035)],[(1030,904)],[(842,1019)],[(1250,1081)]],\
    [[(1135,1031)],[(1193,900)],[(1004,1015)],[(1414,1077)]],\
    [[(1215,1030)],[(1273,900)],[(1084,1016)],[(1494,1077)]],\
    [[(1216,1029)],[(1274,898)],[(1086,1014)],[(1495,1078)]],\
    [[(1211,1030)],[(1269,899)],[(1080,1015)],[(1491,1076)]],\
    [[(1019,1009)],[(1077,878)],[(888,994)],[(1298,1055)]],\
    [[(1207,1033)],[(1266,901)],[(1077,1017)],[(1487,1080)]],\
    [[(1215,1030)],[(1273,899)],[(1083,1015)],[(1494,1076)]],\
    [[(1136,1031)],[(1194,898)],[(1005,1016)],[(1414,1077)]],\
    [[(1216,1030)],[(1274,899)],[(1086,1015)],[(1496,1076)]],\
    [[(1214,1030)],[(1272,901)],[(1083,1015)],[(1493,1078)]],\
    [[(1215,1030)],[(1273,900)],[(1084,1016)],[(1494,1077)]]



m_19151313_1057000 = 13.202+1.85 # 1
#m_19151073_1057181 = 14.262+1.85 # 2
m_19151073_1057181 = 13.782+1.85 # 2
m_19151430_1056079 = 12.675+1.85 # 3   
STARS = 'GRS1915+105','19:15:13.13 +10:57:00.0','19:15:10.73 +10:57:18.1', '19:15:14.30 +10:56:07.9'


mag = m_19151313_1057000, m_19151073_1057181, m_19151430_1056079
dm_GRS1915 = []
dm_19151073_1057181, dm_19151313_1057000, dm_19151430_1056079 =[],[],[]
name_list = 'mean', '19:15:13.13 +10:57:00.0', '19:15:10.73 +10:57:18.1', '19:15:14.30 +10:56:07.9'
names = name_list

DATE,mag_error,sky = [],[],[]
#images.remove('./0001354640-20170921-EMIR-STARE_IMAGE.fits')
#POSITIONS.remove(POSITIONS[12])
for i in range(len(POSITIONS)):#range(13,23):
    #pos = tuple(INVERSA)[i]
    print(i)
    pos = POSITIONS[i]
    dat, prj, airmass, hour = clear_image(images[i],FF_clean,projection=True)
    #M,dM,rmin,s= Photometry(dat, pos, mag, names, prj)
    radio = dic[images[i][16:]]*1.5
    M,dM,s= Forced_Photometry(dat, radio/2, pos, mag, names, prj,PLOT=True)
    del dat, prj
    dm_GRS1915.append(M[0])
    dm_19151073_1057181.append(M[2]), dm_19151313_1057000.append(M[1]), dm_19151430_1056079.append(M[3])
    mag_error.append(dM)
    #DATES.append(Im_list[i][-30:-22])
    #rad.append(rmin)
    sky.append(s)
    t = Time(hour, format='isot', scale='utc')
    DATE.append(t.mjd)
# Having an idea of the flattened of flat field
for i in range(len(sky)):
    #print('Sky error=',np.sqrt(sky[i]))
    print(abs(sky[i]-np.mean(sky[i])))

dm_GRS1915 = np.asarray(dm_GRS1915)

dm_19151073_1057181, dm_19151313_1057000, dm_19151430_1056079= \
    np.asarray(dm_19151073_1057181), np.asarray(dm_19151313_1057000), np.asarray(dm_19151430_1056079)

mag_error = np.asarray(mag_error)
photometry_list = dm_GRS1915, dm_19151073_1057181, dm_19151313_1057000, dm_19151430_1056079
color_list  = 'red','green','blue'
plotting_multiple_stars(DATE,photometry_list,name_list,color_list,error=0.05)
plt.figure()
plt.plot(dm_GRS1915,'.')

'''
#%% Using "good" images:
    
runcell("[Using else's flats in Ks]", '/home/carlos/Desktop/MSc/TFM/GRS_1915+105_EMIR/Photometry.py')
runcell('[Images_extraction]', '/home/carlos/Desktop/MSc/TFM/GRS_1915+105_EMIR/Photometry.py')

im = os.listdir('./PHOTOMETRY_17')
im.remove('0001354640-20170921-EMIR-STARE_IMAGE.fits')
images = []
for i in im:
    images.append('./PHOTOMETRY_17/'+i)
images.remove('./PHOTOMETRY_17/combine.py')
# The desired image order is: 
'''
images1 = ['./PHOTOMETRY_17/0001303035-20170813-EMIR-STARE_IMAGE.fits',
    './PHOTOMETRY_17/0001627891-20180716-EMIR-STARE_IMAGE.fits',
    './PHOTOMETRY_17/0001354637-20170921-EMIR-STARE_IMAGE.fits',
    './PHOTOMETRY_17/0001314004-20170902-EMIR-IMAGE.fits']
# The problem in'./PHOTOMETRY_17/0001377893-20171004-EMIR-IMAGE.fits' with "blue" is that is completely decentered, which results in a 
POSITIONS1 =  [[(893,1034)],[(950,904)],[(762,1020)],[(1171,1081)]],\
    [[(1032,1028)],[(1091,896)],[(902,1013)],[(1312,1074)]],\
    [[(1208,1033)],[(1266,902)],[(1076,1018)],[(1487,1081)]],\
    [[(1216,1030)],[(1274,899)],[(1086,1015)],[(1496,1076)]]

'''
images1 = ['./PHOTOMETRY_17/0001303035-20170813-EMIR-STARE_IMAGE.fits',
    './PHOTOMETRY_17/0001303032-20170813-EMIR-STARE_IMAGE.fits',
    './PHOTOMETRY_17/0001303029-20170813-EMIR-STARE_IMAGE.fits',
    './PHOTOMETRY_17/0001303041-20170813-EMIR-STARE_IMAGE.fits',
    './PHOTOMETRY_17/0001558126-20180524-EMIR-STARE_IMAGE.fits',
    './PHOTOMETRY_17/0001377893-20171004-EMIR-IMAGE.fits',
    './PHOTOMETRY_17/0001366505-20171001-EMIR-IMAGE.fits',
    './PHOTOMETRY_17/0001303038-20170813-EMIR-STARE_IMAGE.fits',
    './PHOTOMETRY_17/0001303050-20170813-EMIR-STARE_IMAGE.fits',
    './PHOTOMETRY_17/0001621709-20180705-EMIR-STARE_IMAGE.fits']

POSITIONS1 =  [[(893,1034)],[(950,904)],[(762,1020)],[(1171,1081)]],\
    [[(892,1035)],[(951,904)],[(762,1020)],[(1171,1082)]],\
    [[(973,1035)],[(1030,904)],[(842,1019)],[(1250,1081)]],\
    [[(1214,1030)],[(1272,901)],[(1083,1015)],[(1493,1078)]],\
    [[(1216,1032)],[(1273,901)],[(1085,1017)],[(1495,1078)]],\
    [[(1216,1032)],[(1273,901)],[(1085,1017)],[(1495,1078)]],\
    [[(1253,1035)],[(1311,904)],[(1123,1020)],[(1533,1082)]],\
    [[(973,1035)],[(1030,904)],[(842,1019)],[(1250,1081)]],\
    [[(1214,1030)],[(1272,901)],[(1083,1015)],[(1493,1078)]],\
    [[(1215,1030)],[(1273,900)],[(1084,1016)],[(1494,1077)]]

#    './PHOTOMETRY_17/0001377890-20171004-EMIR-IMAGE.fits',       [[(999,1020)],[(1057,890)],[(869,1004)],[(1277,1067)]],\ 
#

m_19151313_1057000 = 13.202+1.85 # 1
#m_1915107364_10571818 = 14.262+1.85 # 2
m_19151073_1057181 = 13.782+1.85 # 2
m_19151430_1056079 = 12.675+1.85 # 3   
STARS = 'GRS1915+105','19:15:13.13 +10:57:00.0','19:15:10.73 +10:57:18.1', '19:15:14.30 +10:56:07.9'


mag = m_19151313_1057000, m_19151073_1057181, m_19151430_1056079
dm_GRS1915 = []
dm_19151073_1057181, dm_19151313_1057000, dm_19151430_1056079 =[],[],[]
name_list = 'mean','19:15:10.73 +10:57:18.1','19:15:13.13 +10:57:00.0','19:15:14.30 +10:56:07.9'
names = name_list

DATE,mag_error,sky = [],[],[]
#images.remove('./0001354640-20170921-EMIR-STARE_IMAGE.fits')
#POSITIONS.remove(POSITIONS[12])
for i in range(len(POSITIONS1)):#range(13,23):
    #pos = tuple(INVERSA)[i]
    print(i)
    pos = POSITIONS1[i]
    dat, prj, airmass, hour = clear_image(images1[i],FF_clean,projection=True)
    #M,dM,rmin,s= Photometry(dat, pos, mag, names, prj)
    radio = dic[images1[i][16:]]*1.5
    M,dM,s= Forced_Photometry(dat, radio/2, pos, mag, names, prj,PLOT=False)
    plot_photometry(dat, 5, [pos[0]], 'GRS1915+105', prj)
    del dat, prj
    dm_GRS1915.append(M[0])
    dm_19151073_1057181.append(M[2]), dm_19151313_1057000.append(M[1]), dm_19151430_1056079.append(M[3])
    mag_error.append(dM)
    #DATES.append(Im_list[i][-30:-22])
    #rad.append(rmin)
    sky.append(s)
    t = Time(hour, format='isot', scale='utc')
    DATE.append(t.mjd)
# Having an idea of the flattened of flat field
for i in range(len(sky)):
    #print('Sky error=',np.sqrt(sky[i]))
    print(abs(sky[i]-np.mean(sky[i])))

dm_GRS1915 = np.asarray(dm_GRS1915)

dm_19151073_1057181, dm_19151313_1057000, dm_19151430_1056079= \
    np.asarray(dm_19151073_1057181), np.asarray(dm_19151313_1057000), np.asarray(dm_19151430_1056079)

mag_error = np.asarray(mag_error)
photometry_list = dm_GRS1915, dm_19151073_1057181, dm_19151313_1057000, dm_19151430_1056079
color_list  = 'red','green','blue'
plotting_multiple_stars(DATE,photometry_list,name_list,color_list,error=0.05)
plt.figure()
plt.plot(dm_GRS1915,'.')


FLUX_GRS1915_Jy_17 = 10**(-dm_GRS1915/2.5)*3631 # Jy
FLUX_GRS1915_cgs_17 = FLUX_GRS1915_Jy_17*1e-23 # erg/s/cm**2/Hz
try: os.remove('flux_Jy_GRS1915_17.txt')
except: pass
A = []
for i in range(len(DATE)):
    A.append([DATE[i],FLUX_GRS1915_Jy_17[i]])
with open('flux_Jy_GRS1915_17.txt', 'a') as f:
    f.write(str(A))
f.close()        

#%%  sigma_clipping and combination: https://ccdproc.readthedocs.io/en/latest/image_combination.html
runcell(0, '/home/carlos/Desktop/MSc/TFM/GRS_1915+105_EMIR/Photometry.py')
from astropy import units as u
from astropy.nddata import CCDData
from ccdproc import Combiner
images = os.listdir('./PHOTOMETRY_17')
images.remove('combine.py')
images.remove('0001753266-20180829-EMIR-STARE_IMAGE.fits')
images.remove('0001621703-20180705-EMIR-STARE_IMAGE.fits')
images.remove('0001621712-20180705-EMIR-STARE_IMAGE.fits')
images.remove('0001621706-20180705-EMIR-STARE_IMAGE.fits')
images.remove('0001621709-20180705-EMIR-STARE_IMAGE.fits')
images.remove('0001354634-20170921-EMIR-STARE_IMAGE.fits')
images.remove('0001354640-20170921-EMIR-STARE_IMAGE.fits')
images.remove('0001354637-20170921-EMIR-STARE_IMAGE.fits')
images.remove('0001314004-20170902-EMIR-IMAGE.fits')
IM = []
for i in range(len(images)):
    dat, prj, airmass, hour = clear_image('.PHOTOMETRY_17/'+images[i],1,projection=True)
    IM.append(CCDData(dat,unit=u.adu))
    del dat, prj, airmass, hour
combiner = Combiner(IM)
del IM
#%% Let's sigma_clipping the resulting image 

#combiner.sigma_clipping(low_thresh=5, high_thresh=3, func=np.ma.median)
combiner.clip_extrema(nlow=0, nhigh=15) # Clipping the most extreme images
combined_median = combiner.median_combine() # combining images
plt.imshow(combined_median)
# <codecell> 
''' FF GENERATION FOR 18'''

plt.close('all')
# Different days images:

    
OB1 = sel_im('./GRS1915_K_18A_01_OB0001/data')
OB2 = sel_im('./GRS1915_K_18A_02_OB0003/data')
OB3 = sel_im('./GRS1915_K_18A_03_OB0005/data')
OB4 = sel_im('./GRS1915_K_18A_04_OB0007/data')
OB5 = sel_im('./GRS1915_K_18A_05_OB0001/data')    
OB6 = sel_im('./GRS1915_K_18A_06_OB0003/data')
OB7 = sel_im('./GRS1915_K_18A_07_OB0005/data')
OB8 = sel_im('./GRS1915_K_18A_08_OB0007/data')
OB9 = sel_im('./GRS1915_K_18A_09_OB0009/data')
OB10 = sel_im('./GRS1915_K_18A_10_OB0011/data')

Im_list = OB1 + OB2 + OB3 + OB4 + OB5 + OB6 + OB7 + OB8 + OB9 + OB10
Im_list.remove('./GRS1915_K_18A_10_OB0011/data/0001753266-20180829-EMIR-STARE_IMAGE.fits')
Im_list.remove('./GRS1915_K_18A_06_OB0003/data/0001621703-20180705-EMIR-STARE_IMAGE.fits')
Im_list.remove('./GRS1915_K_18A_06_OB0003/data/0001621712-20180705-EMIR-STARE_IMAGE.fits')
Im_list.remove('./GRS1915_K_18A_06_OB0003/data/0001621706-20180705-EMIR-STARE_IMAGE.fits')
Im_list.remove('./GRS1915_K_18A_06_OB0003/data/0001621709-20180705-EMIR-STARE_IMAGE.fits')
Im_list.remove('./GRS1915_K_18A_03_OB0005/data/0001354634-20170921-EMIR-STARE_IMAGE.fits')
Im_list.remove('./GRS1915_K_18A_03_OB0005/data/0001354640-20170921-EMIR-STARE_IMAGE.fits')
Im_list.remove('./GRS1915_K_18A_03_OB0005/data/0001354637-20170921-EMIR-STARE_IMAGE.fits')
Im_list.remove('./GRS1915_K_18A_02_OB0003/data/0001314004-20170902-EMIR-IMAGE.fits')

long_img = ['./GRS1915_K_18A_10_OB0011/data/0001753266-20180829-EMIR-STARE_IMAGE.fits',
            './GRS1915_K_18A_06_OB0003/data/0001621703-20180705-EMIR-STARE_IMAGE.fits',
            './GRS1915_K_18A_06_OB0003/data/0001621712-20180705-EMIR-STARE_IMAGE.fits',
            './GRS1915_K_18A_06_OB0003/data/0001621706-20180705-EMIR-STARE_IMAGE.fits',
            './GRS1915_K_18A_06_OB0003/data/0001621709-20180705-EMIR-STARE_IMAGE.fits',
            './GRS1915_K_18A_03_OB0005/data/0001354634-20170921-EMIR-STARE_IMAGE.fits',
            './GRS1915_K_18A_03_OB0005/data/0001354640-20170921-EMIR-STARE_IMAGE.fits',
            './GRS1915_K_18A_03_OB0005/data/0001354637-20170921-EMIR-STARE_IMAGE.fits',
            './GRS1915_K_18A_02_OB0003/data/0001314004-20170902-EMIR-IMAGE.fits']

FF = []
head = []
i=0
for image in Im_list:
    data = fits.open(image)
    FF.append(data[0].data)
    head.append(data[0].header)
    data.close()
    del data
median_clean = np.median(FF, axis=0)
FF_18A = median_clean/np.mean(median_clean)


datFF, airmassFF = clear_image(Im_list[0],FF_18A)
dat, airmass = clear_image(Im_list[0],1)
fig,axs=plt.subplots(1,3)
axs[0].imshow(dat,vmin=0.8*np.mean(dat),vmax=1.2*np.mean(dat))
axs[1].imshow(datFF,vmin=0.8*np.mean(datFF),vmax=1.2*np.mean(datFF))



FF_long = []
head = []
i=0
for image in long_img:
    data = fits.open(image)
    FF_long.append(data[0].data)
    head.append(data[0].header)
    data.close()
    del data
median_clean = np.median(FF_long, axis=0)
FF_18A_long = median_clean/np.mean(median_clean[:,750:1250])

datFF, airmassFF = clear_image(long_img[0],FF_18A_long)
dat, airmass = clear_image(long_img[0],1)
fig,axs=plt.subplots(2)
axs[0].imshow(dat,vmin=0.8*np.mean(dat[750:950,:]),vmax=1.2*np.mean(dat[750:950,:]))
axs[1].imshow(datFF,vmin=0.8*np.mean(datFF[750:950,:]),vmax=1.2*np.mean(datFF[750:950,:]))
axs[0].set_ylim(600,1454),axs[1].set_ylim(600,1454)
del median_clean, head, FF, FF_long, airmassFF
# In[] BCK stimation from combined images. 
try: del bkg, sigma_clip, bkg_estimator, airmassFF, datFF
except: pass
# Estimating 2D background

sigma_clip = SigmaClip(sigma=3.)
bkg_estimator = MedianBackground()
bkg = Background2D(FF_18A_long, (5, 5), filter_size=(10, 10),sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
print(bkg.background_median)  
print(bkg.background_rms_median)  
# Background image
fig, ax = plt.subplots(1,2)
ax[1].imshow(bkg.background, origin='lower', cmap='Greys_r', interpolation='nearest',vmin=0.8)
ax[0].imshow(FF_18A_long, origin='lower', cmap='Greys_r',vmin=0.8)
ax[0].set_xlim(600,1454),ax[1].set_xlim(600,1454)
#plt.colorbar()
#plt.title('Background')

datFF, airmassFF = clear_image(long_img[0],bkg.background)
dat, airmass = clear_image(long_img[0],1)
fig,axs=plt.subplots(3)
axs[0].imshow(dat,vmin=0.8*np.mean(dat[750:950,:]),vmax=1.2*np.mean(dat[750:950,:]))
axs[1].imshow(datFF,vmin=0.8*np.mean(datFF[750:950,:]),vmax=1.2*np.mean(datFF[750:950,:]))
axs[2].imshow(FF_18A_long,vmin=0.8)
axs[0].set_ylim(600,1454),axs[1].set_ylim(600,1454),axs[2].set_ylim(600,1454)

# PROBLEMS: We have either star residues or cosmetic defaults. Maybe the second 
# is less damaging.



# <codecell> WORKING WITHOUT FLAT FIELD
'''
# Data visualization
os.chdir('./GRS1915_K_21A_04_OB0076')
#dat = fits.open('data/0003108037-20210826-EMIR-STARE_IMAGE.fits')
dat = fits.open('obsid_0003108037_results/result_image.fits')
#flt = fits.open('data/master_flat.fits')
data = dat[0].data
header = dat[0].header
#flat = flt[0].data
plt.close('all')
norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(data, norm=norm, origin='lower', cmap='Greys_r',interpolation='nearest')
plt.colorbar()

# Estimating 2D background

sigma_clip = SigmaClip(sigma=3.)
bkg_estimator = MedianBackground()
bkg = Background2D(data, (50, 50), filter_size=(3, 3),sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
print(bkg.background_median)  
print(bkg.background_rms_median)  
# Background image
plt.figure()
plt.imshow(bkg.background, origin='lower', cmap='Greys_r', interpolation='nearest')
plt.colorbar()
plt.title('Background')

# Background substracted image: 
plt.figure()
from astropy.wcs import WCS
wcs = WCS(dat[0].header)
plt.subplot(projection=wcs)
plt.imshow(data-bkg.background, norm=norm, origin='lower', cmap='Greys_r', \
           interpolation='nearest',vmin=0,vmax=0.95*np.max(data))
plt.colorbar()
plt.grid()
ax = plt.gca()
ax.invert_yaxis()
plt.title('Background substracted images')


plt.figure()
plt.imshow(data-bkg.background, norm=norm, origin='lower', cmap='Greys_r', \
           interpolation='nearest',vmin=0,vmax=0.95*np.max(data))
plt.colorbar()
plt.show()

data_new=data-bkg.background

noise = np.mean(bkg.background)
# Aperture photometry without correct corrections
# Creating aperture object:
plt.close('all')
posit = [(1329.,1214.)]
posit_2MASSJ19150912 = [(1316.,986.)]
posit_2MASSJ19151202 = [(1045.,1070.)]
posit_2MASSJ19151187= [(1219.,1069.)]
# To choose the aperture:
#fig,axs=plt.subplots(2)
#axs[0].plot(data_new[1214,:])
#axs[1].plot(data_new[:,1329])

GRS1915 = radius_selection(data_new, posit)
TWO_MASSJ19150912 = radius_selection(data_new, posit_2MASSJ19150912 ) #2MASS J19150912+1057111 
TWO_MASSJ19151202 = radius_selection(data_new, posit_2MASSJ19151202) # 2MASS J19151202+1057466 
TWO_MASSJ19151187 = radius_selection(data_new, posit_2MASSJ19151187) # 2MASS J19151187+1055466 



dGRS1915 = GRS1915[3]
dTWO_MASSJ19150912 = TWO_MASSJ19150912[3]
dTWO_MASSJ19151202 = TWO_MASSJ19151202[3]
dTWO_MASSJ19151187 = TWO_MASSJ19151187[3]


m_2MASSJ19150912_K = 10.13
m_2MASSJ19151202_K = 11.514
m_2MASSJ19151187_K = 13.78
m_GRS1915_K = -2.5*np.log10(GRS1915[1]/TWO_MASSJ19150912[1])+m_2MASSJ19150912_K
print(r'm_{GRS1915+105}=',m_GRS1915_K)
m_GRS1915_K = -2.5*np.log10(GRS1915[1]/TWO_MASSJ19151202[1])+m_2MASSJ19151202_K
print(r'm_{GRS1915+105}=',m_GRS1915_K)
m_GRS1915_K = -2.5*np.log10(GRS1915[1]/TWO_MASSJ19151187[1])+m_2MASSJ19151187_K
print(r'm_{GRS1915+105}=',m_GRS1915_K)
plt.figure()
plt.imshow(data_new, norm=norm, origin='lower', cmap='Greys_r', \
           interpolation='nearest',vmin=0,vmax=0.95*np.max(data_new))
GRS1915[2].plot(color='white', lw=2)#,label='Photometry aperture')
TWO_MASSJ19150912[2].plot(color='white', lw=2)#,label='Photometry aperture')
TWO_MASSJ19151202[2].plot(color='white', lw=2)
TWO_MASSJ19151187[2].plot(color='white', lw=2)
plt.text(posit[0][0],posit[0][1],'GRS 1915+105',c='red')
plt.text(posit_2MASSJ19151202[0][0],posit_2MASSJ19151202[0][1],\
         '2MASS J19151202+1057466',c='red')
plt.text(posit_2MASSJ19151202[0][0],posit_2MASSJ19151202[0][1],\
         '2MASS J19151202+1057466',c='red')
# Error stimation
dm_GRS1915 = abs(-2.5*dGRS1915/GRS1915[1]/np.log(10))+abs(2.5*dTWO_MASSJ19150912/TWO_MASSJ19150912[1])/np.log(10)+0.03
print(dm_GRS1915)
dm_GRS1915 = abs(-2.5*dGRS1915/GRS1915[1]/np.log(10))+abs(2.5*dTWO_MASSJ19151202/TWO_MASSJ19151202[1])/np.log(10)+0.03
print(dm_GRS1915)
dm_GRS1915 = abs(-2.5*dGRS1915/GRS1915[1]/np.log(10))+abs(2.5*dTWO_MASSJ19151187/TWO_MASSJ19151187[1])/np.log(10)+0.03
print(dm_GRS1915)
plt.show()
'''
#%% Flat generation from raw images 21A_05
'''
os.chdir('/home/carlos/Desktop/MSc/TFM/GRS_1915+105_EMIR/GRS1915_K_21A_05_OB0078')

plt.close('all')

obs_calib = ['0003137758-20210918-EMIR-STARE_IMAGE_raw.fits','0003137762-20210918-EMIR-STARE_IMAGE_raw.fits',\
       '0003137766-20210918-EMIR-STARE_IMAGE_raw.fits']
obs_star = ['0003137759-20210918-EMIR-STARE_IMAGE_raw.fits','0003137763-20210918-EMIR-STARE_IMAGE_raw.fits',\
       '0003137767-20210918-EMIR-STARE_IMAGE_raw.fits']
    
W_S, WO_S = [], []
head = []
i=0
for rn in range(len(obs_calib)):
    dat_calib = fits.open('../RawImaging_GRS1915_K_21A_05_OB0078-20211114T125518Z-001/RawImaging_GRS1915_K_21A_05_OB0078/data/'+\
                    #image+'/result_image.fits')
                    obs_calib[rn])
    dat_star =  fits.open('../RawImaging_GRS1915_K_21A_05_OB0078-20211114T125518Z-001/RawImaging_GRS1915_K_21A_05_OB0078/data/'+\
                    #image+'/result_image.fits')
                    obs_star[rn])
    W_S.append(dat_star[0].data), WO_S.append(dat_calib[0].data)
    head.append(dat_star[0].header)
    dat_calib.close(),dat_star.close()
median_clean = np.median(WO_S, axis=0)
FF_clean = median_clean/np.mean(median_clean)
median_star = np.median(W_S, axis=0)
FF_star = median_star/np.mean(median_star)

# Images: 

fig,[ax1,ax2,ax3]= plt.subplots(1,3)
im1 = ax1.imshow(W_S[0],vmin=1e4,vmax=3e4,cmap = 'Greys')#0.5*np.max(dat[0].data))
im2 = ax2.imshow(W_S[0]/FF_clean,vmin=1e4,vmax=3e4,cmap = 'Greys')#1e-2*np.ma.masked_invalid(dat[0].data/FF).max())
im3 = ax3.imshow(W_S[0]/FF_star,vmin=1e4,vmax=3e4,cmap = 'Greys')

ax1.set_title('Raw image')
ax2.set_title('Using FF from images without stars')
ax3.set_title('Using FF from images with stars')
ax2.set_yticks([]),ax3.set_yticks([]) 

divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')


divider = make_axes_locatable(ax3)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im3, cax=cax, orientation='vertical')

#hdu = fits.PrimaryHDU(data=dat[0].data/FF_star,header=dat[0].header)
#hdulFinal = fits.HDUList([hdu])
#hdulFinal.writeto('../proof_image.fits')

# Flat field vs. cleaned image. 

fig,[ax1,ax2]= plt.subplots(1,2)
im1 = ax1.imshow(FF_star,cmap = 'Greys')
ax1.set_title('FF')
im2 = ax2.imshow(W_S[0]/FF_star,vmin=1e4,vmax=3e4,cmap = 'Greys')
ax2.set_title('Using FF from images with stars')
ax2.set_yticks([]) 

divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')
'''
#%% FIRST APPROACH TO THE PHOTOMETRY WITH 4 STARS:
'''
# Doing photometry of several objects for the different images:
m_19151209_1056272 = 13.518
m_19151073_1057181 = 14.262
m_19151267_1057031 = 14.408
m_19151313_1057000 = 13.202   

POSITIONS = [[(1214.,1330.)],[(1299.,1384.)],[(1083,1315.)],[(1235.,1204.)],[(1272.,1199.)]],\
    [[(1000.,1040.)],[(1086.,1093.)],[(870.,1025.)],[(1022.,914.)],[(1059.,909.)]],\
    [[(1214.,1329.)],[(1299.,1383.)],[(1083.,1314.)],[(1235.,1203.)],[(1272.,1198.)]],\
    [[(1210.,1028.)],[(1295.,1081.)],[(1079.,1014.)],[(1231.,902.)],[(1268.,897.)]],\
    [[(1214.,1330.)],[(1299.,1383.)],[(1083.,1314.)],[(1235.,1203.)],[(1271.,1197.)]],\
    [[(1214.,1329.)],[(1299.,1382.)],[(1083.,1314.)],[(1235.,1203.)],[(1272.,1198.)]],\
    [[(1210.,1323.)],[(1295.,1377.)],[(1079.,1308.)],[(1231.,1198.)],[(1268.,1191.)]],\
    [[(1016.,1037.)],[(1102.,1091.)],[(886.,1023.)],[(1038.,912.)],[(1074.,906.)]],\
    [[(1214.,1329.)],[(1299.,1383.)],[(1083.,1314.)],[(1235.,1203.)],[(1271.,1198.)]]
mag = m_19151209_1056272, m_19151073_1057181, m_19151267_1057031, m_19151313_1057000
names = 'GRS1915+105','19151209+1056272','19151020+1057011','19151267+1057031','19151313+1057000'

dm_GRS1915, dm_19151209_1056272, dm_19151073_1057181, dm_19151267_1057031, dm_19151313_1057000 = [],[],[],[],[]
DATE,mag_error,rad,sky = [],[],[],[]

for i in range(len(POSITIONS)):
    pos = POSITIONS[i]
    dat, prj, airmass,hour = clear_image(Im_list[i],FF_clean,projection=True)
    M,dM,rmin,s= Photometry(dat, pos, mag, names, prj)
    del dat, pos, prj
    dm_GRS1915.append(M[0])
    dm_19151209_1056272.append(M[1]), dm_19151073_1057181.append(M[2])
    dm_19151267_1057031.append(M[3]), dm_19151313_1057000.append(M[4])
    mag_error.append(dM)
    rad.append(rmin)
    sky.append(s)
    t = Time(hour, format='isot', scale='utc')
    DATE.append(t.mjd)
# Having an idea of the flattened of flat field
for i in range(len(sky)):
    #print('Sky error=',np.sqrt(sky[i]))
    print(abs(sky[i]-np.mean(sky[i])))

dm_GRS1915 = np.asarray(dm_GRS1915)
dm_19151209_1056272 = np.asarray(dm_19151209_1056272)
dm_19151073_1057181 = np.asarray(dm_19151073_1057181)
dm_19151267_1057031 = np.asarray(dm_19151267_1057031)
dm_19151313_1057000 = np.asarray(dm_19151313_1057000)

#mag_error = np.asarray(mag_error)

DATE = []
for day in DATES: 
    DAY = day[:4]+'-'+day[4:6]+'-'+day[6:8]+'T00:00'
    t = Time(DAY, format='isot', scale='utc')
    DATE.append(t.mjd)

Colors according to the reference star used:
19151209_1056272 : red
19151020_1057011 : green
19151267_1057031 : blue
19151313_1057000 : orange

fig, axs = plt.subplots(5,sharex=True,sharey=True)

axs[0].errorbar(DATE,dm_GRS1915[:,0]-np.mean(dm_GRS1915[:,0]),yerr=0.05,fmt='o',c='red')
axs[0].errorbar(DATE,dm_GRS1915[:,1]-np.mean(dm_GRS1915[:,1]),yerr=0.05,fmt='o',c='green')
axs[0].errorbar(DATE,dm_GRS1915[:,2]-np.mean(dm_GRS1915[:,2]),yerr=0.05,fmt='o',c='blue')
axs[0].errorbar(DATE,dm_GRS1915[:,3]-np.mean(dm_GRS1915[:,3]),yerr=0.05,fmt='o',c='orange')
axs[0].plot(DATE,np.mean(dm_GRS1915,1)-np.mean(dm_GRS1915),'k')#,'xk',markersize=12)

axs[1].errorbar(DATE,dm_19151073_1057181[:,0]-np.mean(dm_19151073_1057181[:,0]),yerr=0.05,fmt='o',c='green')
axs[1].errorbar(DATE,dm_19151073_1057181[:,1]-np.mean(dm_19151073_1057181[:,1]),yerr=0.05,fmt='o',c='blue')
axs[1].errorbar(DATE,dm_19151073_1057181[:,2]-np.mean(dm_19151073_1057181[:,2]),yerr=0.05,fmt='o',c='orange')
axs[1].plot(DATE,np.mean(dm_19151073_1057181,1)-np.mean(dm_19151073_1057181),'k')

axs[2].errorbar(DATE,dm_19151209_1056272[:,0]-np.mean(dm_19151209_1056272[:,0]),yerr=0.05,fmt='o',c='red')
axs[2].errorbar(DATE,dm_19151209_1056272[:,1]-np.mean(dm_19151209_1056272[:,1]),yerr=0.05,fmt='o',c='blue')
axs[2].errorbar(DATE,dm_19151209_1056272[:,2]-np.mean(dm_19151209_1056272[:,2]),yerr=0.05,fmt='o',c='orange')
axs[2].plot(DATE,np.mean(dm_19151209_1056272,1)-np.mean(dm_19151209_1056272),'k')

axs[3].errorbar(DATE,dm_19151267_1057031[:,0]-np.mean(dm_19151267_1057031[:,0]),yerr=0.05,fmt='o',c='red')
axs[3].errorbar(DATE,dm_19151267_1057031[:,1]-np.mean(dm_19151267_1057031[:,1]),yerr=0.05,fmt='o',c='green')
axs[3].errorbar(DATE,dm_19151267_1057031[:,2]-np.mean(dm_19151267_1057031[:,2]),yerr=0.05,fmt='o',c='orange')
axs[3].plot(DATE,np.mean(dm_19151267_1057031,1)-np.mean(dm_19151267_1057031),'k')
 
axs[4].errorbar(DATE,dm_19151313_1057000[:,0]-np.mean(dm_19151313_1057000[:,0]),yerr=0.05,fmt='o',c='red')
axs[4].errorbar(DATE,dm_19151313_1057000[:,1]-np.mean(dm_19151313_1057000[:,1]),yerr=0.05,fmt='o',c='green')
axs[4].errorbar(DATE,dm_19151313_1057000[:,2]-np.mean(dm_19151313_1057000[:,2]),yerr=0.05,fmt='o',c='blue')
axs[4].plot(DATE,np.mean(dm_19151313_1057000,1)-np.mean(dm_19151313_1057000),'k')

axs[0].grid(),axs[1].grid(),axs[2].grid(),axs[3].grid(),axs[4].grid()
fig.text(0.5, 0.04, 'MJD', ha='center')
fig.text(0.04, 0.5, r'$\Delta$m', va='center', rotation='vertical')
#axs[0].set_ylim(-0.4,0.4),axs[1].set_ylim(-0.4,0.4),axs[2].set_ylim(-0.4,0.4),axs[3].set_ylim(-0.4,0.4),axs[4].set_ylim(-0.4,0.4)

axs[0].set_ylabel('GRS1915+105')
axs[1].set_ylabel('Red')
axs[2].set_ylabel('Green')
axs[3].set_ylabel('Blue')
axs[4].set_ylabel('Orange')
axs[0].legend(['mean','19:15:12.09 +10:56:27.2','19:15:10.73 +10:57:18.1','19:15:12.67 +10:57:03.1','19:15:13.13 +10:57:00.0'])
axs[0].set_title('Selected radius')
'''


