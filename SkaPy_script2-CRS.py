# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 10:33:04 2018

@author: blepillier
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import matplotlib.gridspec as gridspec
import itertools
from time import clock
import os
import pickle
import mplstereonet
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


#%%Import excel file to be analyzed

# Where is your file is located
#loaddir = r'D:\blepillier\Desktop\00-Tmp\201804-Doc BatL\04 - Python'
loaddir = r'K:\gse\PW\PW-shared\BatL\17-SCL_MPS\201807_SCL_DFN_Gemex'

# Where the results should be saved
#savedir = r'D:\blepillier\Desktop\00-Tmp\201804-Doc BatL\04 - Python'
savedir = r'K:\gse\PW\PW-shared\BatL\17-SCL_MPS\201807_SCL_DFN_Gemex'

#%% BOQUILLAS

#           *** BOQ1 ***

# Import CSV file
# Give filename
filename = '{}/kinkBOQ1.csv'
#filename = '{}/tmp_PracticeTool.csv'

try:
    kinkBoq1 = pd.read_csv(filename.format(loaddir), delimiter=',', index_col=[0], engine='python')
    print('File loaded successfuly!')
except:
    print('!!! Check file - Error when LOADING !!!')

# Calculate (x1,y1)
kinkBoq1['x1'] = kinkBoq1['xo'] + kinkBoq1['fx']
kinkBoq1['y1'] = kinkBoq1['yo'] + kinkBoq1['fy']

# Here are Real coordinates at the outcrop
x_boq1 = 694460
y_boq1 = 2178485

kinkBoq1['xo_ref'] = kinkBoq1['xo'] + x_boq1
kinkBoq1['yo_ref'] = kinkBoq1['yo'] + y_boq1
kinkBoq1['x1_ref'] = kinkBoq1['x1'] + x_boq1
kinkBoq1['y1_ref'] = kinkBoq1['y1'] + y_boq1
kinkBoq1['z'] = 1440

kinkBoq1.to_csv('Exp_Boq1.csv')
#kinkBoq1.to_csv('Exp_Boq1.dat')

#            *** BOQ2 ***

# Import CSV file
# Give filename
filename = '{}/kinkBOQ2.csv'
#filename = '{}/tmp_PracticeTool.csv'

try:
    kinkBoq2 = pd.read_csv(filename.format(loaddir), delimiter=',', index_col=[0], engine='python')
    print('File loaded successfuly!')
except:
    print('!!! Check file - Error when LOADING !!!')
    
# Calculate (x1,y1)
kinkBoq2['x1'] = kinkBoq2['xo'] + kinkBoq2['fx']
kinkBoq2['y1'] = kinkBoq2['yo'] + kinkBoq2['fy']

# Here are Real coordinates at the outcrop
x_boq2 = 694479
y_boq2 = 2178455

kinkBoq2['xo_ref'] = kinkBoq2['xo'] + x_boq2
kinkBoq2['yo_ref'] = kinkBoq2['yo'] + y_boq2
kinkBoq2['x1_ref'] = kinkBoq2['x1'] + x_boq2
kinkBoq2['y1_ref'] = kinkBoq2['y1'] + y_boq2
kinkBoq2['z'] = 1450

kinkBoq2.to_csv('Exp_Boq2.csv')
#kinkBoq2.to_csv('Exp_Boq2.dat')

#%% ELDORADO

#            *** ELD1 ***

# Import CSV file
# Give filename
filename = '{}/kinkELD1.csv'
#filename = '{}/tmp_PracticeTool.csv'

try:
    kinkEld1 = pd.read_csv(filename.format(loaddir), delimiter=',', index_col=[0], engine='python')
    print('File loaded successfuly!')
except:
    print('!!! Check file - Error when LOADING !!!')

# Calculate (x1,y1)
kinkEld1['x1'] = kinkEld1['xo'] + kinkEld1['fx']
kinkEld1['y1'] = kinkEld1['yo'] + kinkEld1['fy']

# Here are Real coordinates at the outcrop
x_eld1 = 694415.51
y_eld1 = 2178086.99

kinkEld1['xo_ref'] = kinkEld1['xo'] + x_eld1
kinkEld1['yo_ref'] = kinkEld1['yo'] + y_eld1
kinkEld1['x1_ref'] = kinkEld1['x1'] + x_eld1
kinkEld1['y1_ref'] = kinkEld1['y1'] + y_eld1
kinkEld1['z'] = 1462

kinkEld1.to_csv('Exp_Eld1.csv')

#            *** ELD2 ***

# Import CSV file
# Give filename
filename = '{}/kinkEld2.csv'
#filename = '{}/tmp_PracticeTool.csv'

try:
    kinkEld2 = pd.read_csv(filename.format(loaddir), delimiter=',', index_col=[0], engine='python')
    print('File loaded successfuly!')
except:
    print('!!! Check file - Error when LOADING !!!')

# Calculate (x1,y1)
kinkEld2['x1'] = kinkEld2['xo'] + kinkEld2['fx']
kinkEld2['y1'] = kinkEld2['yo'] + kinkEld2['fy']

# Here are Real coordinates at the outcrop
x_eld2 = 694419
y_eld2 = 2178114

kinkEld2['xo_ref'] = kinkEld2['xo'] + x_eld2
kinkEld2['yo_ref'] = kinkEld2['yo'] + y_eld2
kinkEld2['x1_ref'] = kinkEld2['x1'] + x_eld2
kinkEld2['y1_ref'] = kinkEld2['y1'] + y_eld2
kinkEld2['z'] = 1457

kinkEld2.to_csv('Exp_Eld2.csv')

#%% TATATILA

# Import CSV file
# Give filename
filename = '{}/kinkTAT.csv'
#filename = '{}/tmp_PracticeTool.csv'

try:
    kinkTat = pd.read_csv(filename.format(loaddir), delimiter=',', index_col=[0], engine='python')
    print('File loaded successfuly!')
except:
    print('!!! Check file - Error when LOADING !!!')

# Calculate (x1,y1)
kinkTat['x1'] = kinkTat['xo'] + kinkTat['fx']
kinkTat['y1'] = kinkTat['yo'] + kinkTat['fy']

# Here are Real coordinates at the outcrop
x_Tat = 698793.76
y_Tat = 2180079.71

kinkTat['xo_ref'] = kinkTat['xo'] + x_Tat
kinkTat['yo_ref'] = kinkTat['yo'] + y_Tat
kinkTat['x1_ref'] = kinkTat['x1'] + x_Tat
kinkTat['y1_ref'] = kinkTat['y1'] + y_Tat
kinkTat['z'] = 1850

kinkTat.to_csv('Exp_Tat.csv')

#%% Pueblo Nuevo PNO

# Import CSV file
# Give filename
filename1 = '{}/kinkPNO.csv'
filename2 = '{}/Scl_surveyPNO.csv'
#filename = '{}/tmp_PracticeTool.csv'

try:
    kinkPNO = pd.read_csv(filename1.format(loaddir), delimiter=',', index_col=[0], engine='python')
    print('File loaded successfuly!')
except:
    print('!!! Check file - Error when LOADING !!!')

try:
    SurveyPNO = pd.read_csv(filename2.format(loaddir), delimiter=',', index_col=[0], engine='python')
    print('File loaded successfuly!')
except:
    print('!!! Check file - Error when LOADING !!!')

# Calculate (x1,y1)
kinkPNO['x1'] = kinkPNO['xo'] + kinkPNO['fx']
kinkPNO['y1'] = kinkPNO['yo'] + kinkPNO['fy']

# Here are Real coordinates at the outcrop
x_PNO = 693029.53
y_PNO = 2180263.17

SurveyPNO['xo_ref'] = SurveyPNO['Xout'] + x_PNO
SurveyPNO['yo_ref'] = SurveyPNO['Yout'] + y_PNO


kinkPNO['xo_ref'] = kinkPNO['xo'] + x_PNO
kinkPNO['yo_ref'] = kinkPNO['yo'] + y_PNO
kinkPNO['x1_ref'] = kinkPNO['x1'] + x_PNO
kinkPNO['y1_ref'] = kinkPNO['y1'] + y_PNO
kinkPNO['z'] = 2043

kinkPNO.to_csv('Exp_PNO.csv')

#%% Rinconada RIN

# Import CSV file
# Give filename
filename = '{}/kinkRIN.csv'
#filename = '{}/tmp_PracticeTool.csv'

try:
    kinkRin = pd.read_csv(filename.format(loaddir), delimiter=',', index_col=[0], engine='python')
    print('File loaded successfuly!')
except:
    print('!!! Check file - Error when LOADING !!!')

# Calculate (x1,y1)
kinkRin['x1'] = kinkRin['xo'] + kinkRin['fx']
kinkRin['y1'] = kinkRin['yo'] + kinkRin['fy']

# Here are Real coordinates at the outcrop
x_Rin = 692787.68
y_Rin = 2175790.93

kinkRin['xo_ref'] = kinkRin['xo'] + x_Rin
kinkRin['yo_ref'] = kinkRin['yo'] + y_Rin
kinkRin['x1_ref'] = kinkRin['x1'] + x_Rin
kinkRin['y1_ref'] = kinkRin['y1'] + y_Rin
kinkRin['z'] = 1883

kinkRin.to_csv('Exp_Rin.csv')

#%% SAT

# Import CSV file
# Give filename
filename = '{}/kinkSAT.csv'

try:
    kinkSat = pd.read_csv(filename.format(loaddir), delimiter=',', index_col=[0], engine='python')
    print('File loaded successfuly!')
except:
    print('!!! Check file - Error when LOADING !!!')

# Calculate (x1,y1)
kinkSat['x1'] = kinkSat['xo'] + kinkSat['fx']
kinkSat['y1'] = kinkSat['yo'] + kinkSat['fy']

# Here are Real coordinates at the outcrop
x_Sat = 679318.10
y_Sat = 2156777.25

kinkSat['xo_ref'] = kinkSat['xo'] + x_Sat
kinkSat['yo_ref'] = kinkSat['yo'] + y_Sat
kinkSat['x1_ref'] = kinkSat['x1'] + x_Sat
kinkSat['y1_ref'] = kinkSat['y1'] + y_Sat
kinkSat['z'] = 2492

kinkSat.to_csv('Exp_Sat.csv')

#%% PLOTS

 # Define Figure/plot specs
fig9 = plt.figure(figsize=(7,4))
row = 1 
column = 1
gs = gridspec.GridSpec(row,column) 

# Define subplots axes:
ax0 = plt.subplot(gs[0,0]) 
ax0 = fig9.add_subplot(gs[0,0])

color='k'

#ax0.plot(kinkBoq1['xo_ref'],kinkBoq1['yo_ref'], color)
#ax0.plot(kinkBoq2['xo_ref'],kinkBoq2['yo_ref'], color)
#ax0.plot(kinkEld1['xo_ref'],kinkEld1['yo_ref'], color)
#ax0.plot(kinkEld2['xo_ref'],kinkEld2['yo_ref'], color)
#ax0.plot(kinkTat['xo_ref'],kinkTat['yo_ref'], color)
ax0.plot(SurveyPNO['xo_ref'],SurveyPNO['yo_ref'], color)
#ax0.plot(kinkRin['xo_ref'],kinkRin['yo_ref'], color)
#ax0.plot(kinkSat['xo_ref'],kinkSat['yo_ref'], color)

# PLOTTING ALL EXTRAPOLATED FRACTURES

#ax0.quiver(kinkBoq1['xo_ref'], kinkBoq1['yo_ref'], kinkBoq1['fx'], kinkBoq1['fy'], color= kinkBoq1['fapcol'], width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkBoq1['xo_ref'], kinkBoq1['yo_ref'], -kinkBoq1['fx'], -kinkBoq1['fy'], color= kinkBoq1['fapcol'], width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkEld1['xo_ref'], kinkEld1['yo_ref'], kinkEld1['fx'], kinkEld1['fy'], color= kinkEld1['fapcol'], width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkEld1['xo_ref'], kinkEld1['yo_ref'], -kinkEld1['fx'], -kinkEld1['fy'], color= kinkEld1['fapcol'], width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkBoq2['xo_ref'], kinkBoq2['yo_ref'], kinkBoq2['fx'], kinkBoq2['fy'], color= kinkBoq2['fapcol'], width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkBoq2['xo_ref'], kinkBoq2['yo_ref'], -kinkBoq2['fx'], -kinkBoq2['fy'], color= kinkBoq2['fapcol'], width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkEld2['xo_ref'], kinkEld2['yo_ref'], kinkEld2['fx'], kinkEld2['fy'], color= kinkEld2['fapcol'], width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkEld2['xo_ref'], kinkEld2['yo_ref'], -kinkEld2['fx'], -kinkEld2['fy'], color= kinkEld2['fapcol'], width=0.001, headlength=0, headaxislength=0, scale=0.1)
#
#ax0.quiver(kinkTat['xo_ref'], kinkTat['yo_ref'], kinkTat['fx'], kinkTat['fy'], color= kinkTat['fapcol'], width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkTat['xo_ref'], kinkTat['yo_ref'], -kinkTat['fx'], -kinkTat['fy'], color= kinkTat['fapcol'], width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkPNO['xo_ref'], kinkPNO['yo_ref'], kinkPNO['fx'], kinkPNO['fy'], color= kinkPNO['fapcol'], width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkPNO['xo_ref'], kinkPNO['yo_ref'], -kinkPNO['fx'], -kinkPNO['fy'], color= kinkPNO['fapcol'], width=0.001, headlength=0, headaxislength=0, scale=0.1)
#
#ax0.quiver(kinkRin['xo_ref'], kinkRin['yo_ref'], kinkRin['fx'], kinkRin['fy'], color= kinkRin['fapcol'], width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkRin['xo_ref'], kinkRin['yo_ref'], -kinkRin['fx'], -kinkRin['fy'], color= kinkRin['fapcol'], width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkSat['xo_ref'], kinkSat['yo_ref'], kinkSat['fx'], kinkSat['fy'], color= kinkSat['fapcol'], width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkSat['xo_ref'], kinkSat['yo_ref'], -kinkSat['fx'], -kinkSat['fy'], color= kinkSat['fapcol'], width=0.001, headlength=0, headaxislength=0, scale=0.1)

# PLOTTING Background fractures Axes using AVG values

#ax0.quiver(kinkBoq1['xo_ref'], kinkBoq1['yo_ref'], kinkBoq1['fxmean'], kinkBoq1['fymean'], color= 'lightgrey', width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkBoq1['xo_ref'], kinkBoq1['yo_ref'], -kinkBoq1['fxmean'], -kinkBoq1['fymean'], color='lightgrey', width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkEld1['xo_ref'], kinkEld1['yo_ref'], kinkEld1['fxmean'], kinkEld1['fymean'], color= 'lightgrey', width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkEld1['xo_ref'], kinkEld1['yo_ref'], -kinkEld1['fxmean'], -kinkEld1['fymean'], color= 'lightgrey', width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkBoq2['xo_ref'], kinkBoq2['yo_ref'], kinkBoq2['fxmean'], kinkBoq2['fymean'], color= 'lightgrey', width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkBoq2['xo_ref'], kinkBoq2['yo_ref'], -kinkBoq2['fxmean'], -kinkBoq2['fymean'], color= 'lightgrey', width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkEld2['xo_ref'], kinkEld2['yo_ref'], kinkEld2['fxmean'], kinkEld2['fymean'], color= 'lightgrey', width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkEld2['xo_ref'], kinkEld2['yo_ref'], -kinkEld2['fxmean'], -kinkEld2['fymean'], color= 'lightgrey', width=0.001, headlength=0, headaxislength=0, scale=0.1)
#
#ax0.quiver(kinkTat['xo_ref'], kinkTat['yo_ref'], kinkTat['fxmean'], kinkTat['fymean'], color= 'lightgrey', width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkTat['xo_ref'], kinkTat['yo_ref'], -kinkTat['fxmean'], -kinkTat['fymean'], color= 'lightgrey', width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkPNO['xo_ref'], kinkPNO['yo_ref'], kinkPNO['fxmean'], kinkPNO['fymean'], color= 'lightgrey', width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkPNO['xo_ref'], kinkPNO['yo_ref'], -kinkPNO['fxmean'], -kinkPNO['fymean'], color= 'lightgrey', width=0.001, headlength=0, headaxislength=0, scale=0.1)
#
#ax0.quiver(kinkRin['xo_ref'], kinkRin['yo_ref'], kinkRin['fxmean'], kinkRin['fymean'], color= 'lightgrey', width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkRin['xo_ref'], kinkRin['yo_ref'], -kinkRin['fxmean'], -kinkRin['fymean'], color= 'lightgrey', width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkSat['xo_ref'], kinkSat['yo_ref'], kinkSat['fxmean'], kinkSat['fymean'], color= 'lightgrey', width=0.001, headlength=0, headaxislength=0, scale=0.1)
#ax0.quiver(kinkSat['xo_ref'], kinkSat['yo_ref'], -kinkSat['fxmean'], -kinkSat['fymean'], color= 'lightgrey', width=0.001, headlength=0, headaxislength=0, scale=0.1)

# PLOTTING SCANLINES

#ax0.quiver(kinkBoq1['xo_ref'], kinkBoq1['yo_ref'], kinkBoq1['fx'], kinkBoq1['fy'], color= kinkBoq1['f_col'], width=0.001, headlength=0, headaxislength=0, scale=None)
#ax0.quiver(kinkBoq1['xo_ref'], kinkBoq1['yo_ref'], -kinkBoq1['fx'], -kinkBoq1['fy'], color= kinkBoq1['f_col'], width=0.001, headlength=0, headaxislength=0, scale=None)
#ax0.quiver(kinkEld1['xo_ref'], kinkEld1['yo_ref'], kinkEld1['fx'], kinkEld1['fy'], color= kinkEld1['f_col'], width=0.001, headlength=0, headaxislength=0, scale=None)
#ax0.quiver(kinkEld1['xo_ref'], kinkEld1['yo_ref'], -kinkEld1['fx'], -kinkEld1['fy'], color= kinkEld1['f_col'], width=0.001, headlength=0, headaxislength=0, scale=None)
#ax0.quiver(kinkBoq2['xo_ref'], kinkBoq2['yo_ref'], kinkBoq2['fx'], kinkBoq2['fy'], color= kinkBoq2['f_col'], width=0.001, headlength=0, headaxislength=0, scale=None)
#ax0.quiver(kinkBoq2['xo_ref'], kinkBoq2['yo_ref'], -kinkBoq2['fx'], -kinkBoq2['fy'], color= kinkBoq2['f_col'], width=0.001, headlength=0, headaxislength=0, scale=None)
#ax0.quiver(kinkEld2['xo_ref'], kinkEld2['yo_ref'], kinkEld2['fx'], kinkEld2['fy'], color= kinkEld2['f_col'], width=0.001, headlength=0, headaxislength=0, scale=None)
#ax0.quiver(kinkEld2['xo_ref'], kinkEld2['yo_ref'], -kinkEld2['fx'], -kinkEld2['fy'], color= kinkEld2['f_col'], width=0.001, headlength=0, headaxislength=0, scale=None)
#
#ax0.quiver(kinkTat['xo_ref'], kinkTat['yo_ref'], kinkTat['fx'], kinkTat['fy'], color= kinkTat['f_col'], width=0.001, headlength=0, headaxislength=0, scale=None)
#ax0.quiver(kinkTat['xo_ref'], kinkTat['yo_ref'], -kinkTat['fx'], -kinkTat['fy'], color= kinkTat['f_col'], width=0.001, headlength=0, headaxislength=0, scale=None)
ax0.quiver(kinkPNO['xo_ref'], kinkPNO['yo_ref'], kinkPNO['fx'], kinkPNO['fy'], color= kinkPNO['f_col'], width=0.001, headlength=0, headaxislength=0, scale=None)
ax0.quiver(kinkPNO['xo_ref'], kinkPNO['yo_ref'], -kinkPNO['fx'], -kinkPNO['fy'], color= kinkPNO['f_col'], width=0.001, headlength=0, headaxislength=0, scale=None)
#
#ax0.quiver(kinkRin['xo_ref'], kinkRin['yo_ref'], kinkRin['fx'], kinkRin['fy'], color= kinkRin['f_col'], width=0.001, headlength=0, headaxislength=0, scale=None)
#ax0.quiver(kinkRin['xo_ref'], kinkRin['yo_ref'], -kinkRin['fx'], -kinkRin['fy'], color= kinkRin['f_col'], width=0.001, headlength=0, headaxislength=0, scale=None)
#ax0.quiver(kinkSat['xo_ref'], kinkSat['yo_ref'], kinkSat['fx'], kinkSat['fy'], color= kinkSat['f_col'], width=0.001, headlength=0, headaxislength=0, scale=None)
#ax0.quiver(kinkSat['xo_ref'], kinkSat['yo_ref'], -kinkSat['fx'], -kinkSat['fy'], color= kinkSat['f_col'], width=0.001, headlength=0, headaxislength=0, scale=None)

plt.axis('equal')
ax0.grid()
fig9.tight_layout()
fig9.savefig('Scanlines_UTMQ14_Boq', dpi=300)

#%%

#Boquillas1 xyz (Main)
# lat 19,69232 / 19.69223
# long -97,14502 / -97.14484
# alt 1440 m
# UTM 694460 m E
# UTM 2178485 m N

#Boquillas2 xyz (Upper)
# lat 19,692235 / 19.69196
# long -97,144769 / -97.14466
# alt 1450 m
# UTM 694479 m E
# UTM 2178455 m N

#Eldorado1 xyz (Main)
# lat 19,688638
# long -97,145308
# alt 1462 m
# UTM 694415.51 m E
# UTM 2178086.99 m N

#Eldorado2 xyz (Lower)
# lat 19.68888
# long -97.14527
# alt 1456.5 m
# UTM 694419 m E
# UTM 2178114 m N

#PNO xyz
# lat 19,70843
# long -97,1583
# alt 2042.6 m
# UTM 693029.53 m E
# UTM 2180263.17 m N

#Tat xyz
# lat 19,706200
# long -97,103345
# alt 1850.4
# UTM 698793.76 m E
# UTM 2180079.71 m N


#Rinco xyz
# lat 19.668059
# long -97.161068
# alt 1882.9 m
# UTM 692787.68 m E
# UTM 2175790.93 m N

#SAT xyz
# lat 19,497576
# long -97,291318
# alt 2492 m
# UTM 679318.10 m E
# UTM 2156777.25 m N