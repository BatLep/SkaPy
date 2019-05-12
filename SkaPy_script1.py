
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 12:02:35 2018

@author: blepillier

NOTE 1:
    For the stereonet projections to work properly you should run from 
    Anaconda command prompt the following commands:
        python -m pip install --upgrade pip setuptools wheel
    and
        pip install mplstereonet

    For stereonet documentation check:    https://github.com/joferkington/mplstereonet

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


#%%Import excel file to be analyzed (excel needs o be saved as .csv)

# Where is your file is located
loaddir = r'D:\blepillier\Desktop\BackUp_2018\201807-SCL_DFN_Gemex\Input'

# Where the results should be saved
savedir = r'D:\blepillier\Desktop\BackUp_2018\201807-SCL_DFN_Gemex'

# Import CSV file
# Give filename
filename = '{}/201807_GeMex-Frac_BatL.csv'

# Save the csv as df means a PANDA DATAFRAME
try:
    df = pd.read_csv(filename.format(loaddir), delimiter=';', index_col=[0], engine='python')
    print('File loaded successfuly!')
except:
    print('!!! Check file - Error when LOADING !!!')

#%% DataFrame Tuning
    
    # for the purpose of statistic calculation VS database tuning
    # df is the main dataframe, with NO DUPLICATES 
    # (refering to the field 'Q' standing for Quantity as the number of fractures per interval)
    # from this initial df, we here-after create a second dataframe with duplicates: df2

df2 = df.loc[np.repeat(df.index.values, df['Q'])]

#%%
#           --- PART 1 - QUICK LOOK AT THE DATA ---
#    
#%% first Quick look at the data

# Print out df columns:
print("dataframe columns are:\n", df.columns)

# Get series out of dataframe and convert to np.arrays
Ap = np.array(df2.F_Ap)
H = np.array(df2.F_H)

# Sort the data and re-index for sorted data:
Ap_sorted = np.sort(Ap)
H_sorted = np.sort(H)
x_sorted = range(0, len(df2.F_H))

# Calculate Exceedance Frequencies EFs:
EF_a = []
EF_h = []
i = 0

for idx, i in enumerate(Ap_sorted):
    i = (len(Ap_sorted) - idx) / (len(Ap_sorted))
    EF_a.append(i)
    
for idx, i in enumerate(H_sorted):    
    i = (len(H_sorted) - idx) / (len(H_sorted))
    EF_h.append(i)

# Figure 1, here give a color reference for Ap & H:
col_Ap = 'b'
col_H = 'c'
col_St = 'r'

# Fig is made of 3 plots: 1/All frac_Heights - 2/All frac_Ap - 3/Both

# Define Figure/plot specs
fig1 = plt.figure(figsize=(12,4),facecolor='w')
row = 2 
column = 3     #
gs = gridspec.GridSpec(row,column)

# define subplot axes
ax0 = plt.subplot(gs[:,0]) 
ax1 = plt.subplot(gs[:,1])
ax2 = plt.subplot(gs[0,2])
ax3 = plt.subplot(gs[1,2])
gs.update(wspace=0.45 ,hspace=0)

ax0.scatter(EF_h, H_sorted, c=col_H, s=2)
ax0.set_xscale('log')
ax0.set_xlim(0.1, 10)
ax0.set_yscale('log')
ax0.set_ylim(0.1, 100)
ax0.set_xlabel('Fracture Heights')
ax0.set_ylabel('Exceedance Frequency')

ax1.scatter(EF_a, Ap_sorted, c=col_Ap, s=2)
ax1.set_xscale('log')
ax1.set_xlim(0.0001, 100)
ax1.set_yscale('log')
ax1.set_ylim(0.01, 10)
ax1.set_xlabel('Fracture Apertures')
ax1.set_ylabel('Exceedance Frequency')

# Plot Ap & H as hist, (sort of bi-histogram inverted = back to back):

ax2.bar(df2.index, df2.F_H, color=col_H)
ax2.set_ylim(0, 1.05*df.F_H.max())
ax2.set_ylabel('Heights (m)', color=col_H, fontsize=8)
ax2.xaxis.set_ticks_position('none')
ax2.get_yaxis().set_label_coords(-0.15, 0.5)

ax3.bar(df2.index, df2.F_Ap, color=col_Ap)
ax3.set_ylim(0, 1.05*df.F_Ap.max())
ax3.invert_yaxis()
ax3.set_ylabel('Apertures (mm)', color=col_Ap, fontsize=8)
ax3.set_xlabel('Sorted fracture indices')
ax3.get_yaxis().set_label_coords(-0.15, 0.5)

plt.tight_layout()
plt.suptitle('Quick look at all dataset', y=1.10, fontsize=10)

plt.show()
fig1.savefig('Figure_01_EF', dpi=300)

#%% STEREOPLOTS & ROSE DIAGRAM ! (For the entire dataset)
    
# Define Figure/plot specs
fig2 = plt.figure(figsize=(7,4))
row = 1 
column = 2
gs = gridspec.GridSpec(row,column) 

# STEREOGRAM: Define subplots axes:
ax0 = plt.subplot(gs[0,0]) 
ax1 = plt.subplot(gs[0,1])

# STEREOGRAM: Density Contouring:
ax0 = fig2.add_subplot(gs[0,0], projection='stereonet')
ax0.plane(df2.F_Az.dropna(), df2.F_Dip.dropna(), c='k', linewidth=0.1)
ax0.density_contourf(df2.F_Az.dropna(), df2.F_Dip.dropna(), measurement='poles', cmap='Reds')
ax0.set_title('Steronet & Density coutour of the Poles', y=1.10, fontsize=8)
ax0.grid()

# ROSE DIAGRAM: Calculate the number of strikes every 10° using np.hist 
bin_edges = np.arange(-5, 366, 10)
number_of_strikes, bin_edges = np.histogram(df2.F_Az.dropna(), bin_edges)

# ROSE DIAGRAM: Sum the last value with the first value.
number_of_strikes[0] += number_of_strikes[-1]

# ROSE DIAGRAM: Sum the first half 0-180° with the second half 180-360° to achieve the "mirrored behavior" of Rose Diagrams.
half = np.sum(np.split(number_of_strikes[:-1], 2), 0)
two_halves = np.concatenate([half, half])

# ROSE DIAGRAM: Create the Rose Diagram:

ax1 = fig2.add_subplot(gs[0,1], projection='polar')
ax1.bar(np.deg2rad(np.arange(0, 360, 10)), two_halves, width=np.deg2rad(10), bottom=0.0, color='.8', edgecolor='k')
ax1.set_theta_zero_location('N')    
ax1.set_theta_direction(-1)
ax1.set_thetagrids(np.arange(0, 360, 45), labels=np.arange(0, 360, 45))
ax1.set_rgrids(np.arange(0, two_halves.max() + 1, 50), labels=None, fontsize=6)
ax1.set_title('Rose Diagram of the Fractures', y=1.10, fontsize=8)

plt.suptitle('All fracture sets', y=1.10, fontsize=10)
#fig2.tight_layout()
fig2.show()
fig2.savefig('Figure_02_Stero_All', dpi=300)

#%%
#           --- PART 2 - NOW Working On CLASSES 
#
#%% Function to create the classes based on [Strike and Dip] = [F_Az and F_Dip]

def set_class(row):
    if row['F_Dip'] <= 40 and 85 <= row['F_Az'] <= 120 or row['F_Dip'] <= 40 and 265 <= row['F_Az'] <= 300:
        return 'F6'
    elif 43 <= row['F_Az'] <= 75 or 225 <= row['F_Az'] <= 255:
        return 'F1'
    elif 120 <= row['F_Az'] <= 145 or 300 <= row['F_Az'] <= 325:
        return 'F2'
    elif 10 <= row['F_Az'] <= 40 or 190 <= row['F_Az'] <= 220:
        return 'F3'
    elif 90 <= row['F_Az'] <= 120 or 270 <= row['F_Az'] <= 300:
        return 'F4'
    elif 0 <= row['F_Az'] <= 10 or 145 <= row['F_Az'] <= 190 or 325 <= row['F_Az'] <= 360:
        return 'F5'
    elif 75 <= row['F_Az'] <= 90 or 255 <= row['F_Az'] <= 290:
        return 'F7'    
    else:
        return 'nan'

df = df.assign(F_Class=df.apply(set_class, axis=1))
df2 = df2.assign(F_Class=df.apply(set_class, axis=1))
print('*** Classification Done! ***')

#%% Assigning AVG values for DFN construction, based on Geological interp.
df['F_Az_mean'] = np.nan

df.F_Az_mean[df['F_Class'] == 'F6'] = 105
df.F_Az_mean[df['F_Class'] == 'F1'] = 60
df.F_Az_mean[df['F_Class'] == 'F2'] = 135
df.F_Az_mean[df['F_Class'] == 'F3'] = 30
df.F_Az_mean[df['F_Class'] == 'F4'] = 110
df.F_Az_mean[df['F_Class'] == 'F5'] = 170
df.F_Az_mean[df['F_Class'] == 'F7'] = 80

df['F_Dip_mean'] = np.nan

for F in df['F_Class'].unique():
    df['F_Dip_mean'][df['F_Class'] == F] = round(df.F_Dip.dropna()[df['F_Class'] == F].mean())


#%% STEREOPLOTS & ROSE DIAGRAM  per F_Class!

# Here is a for loop going in F_Class and pick-out all different values: F1, F2,..F7 and Fnan:
for F in df2['F_Class'].unique():
#for F in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']:
    sF = df2.F_Az[df['F_Class'] == F]
    dF = df2.F_Dip[df['F_Class'] == F]
    
    # Define Figure/plot specs
    fig3 = plt.figure(figsize=(7,4))
    row = 1 
    column = 2
    gs = gridspec.GridSpec(row,column) 
    
    # STEREOGRAM: Define subplots axes:
    ax0 = plt.subplot(gs[0,0]) 
    ax1 = plt.subplot(gs[0,1]) 
    
    # STEREOGRAM: Density Contouring:
    ax0 = fig3.add_subplot(gs[0,0], projection='stereonet')
    ax0.plane(sF, dF, c='k', linewidth=0.1)
    ax0.density_contourf(sF.dropna(), dF.dropna(), measurement='poles', cmap='Reds')
    ax0.set_title('Steronet & Density contour of the Poles', y=1.10, fontsize=8, va='bottom')
    ax0.grid()
    
    # ROSE DIAGRAM: Calculate the number of strikes every 10° using np.hist 
    bin_edges = np.arange(-5, 366, 10)
    number_of_strikes, bin_edges = np.histogram(sF.dropna(), bin_edges)
    
    # ROSE DIAGRAM: Sum the last value with the first value.
    number_of_strikes[0] += number_of_strikes[-1]
    
    # ROSE DIAGRAM: Sum the first half 0-180° with the second half 180-360° to achieve the "mirrored behavior" of Rose Diagrams.
    half = np.sum(np.split(number_of_strikes[:-1], 2), 0)
    two_halves = np.concatenate([half, half])
    
    # ROSE DIAGRAM: Create the Rose Diagram:
    
    ax1 = fig3.add_subplot(gs[0,1], projection='polar')
    ax1.bar(np.deg2rad(np.arange(0, 360, 10)), two_halves, width=np.deg2rad(10), bottom=0.0, color='.8', edgecolor='k')
    ax1.set_theta_zero_location('N')    
    ax1.set_theta_direction(-1)
    ax1.set_thetagrids(np.arange(0, 360, 45), labels=np.arange(0, 360, 45))
    ax1.set_rgrids(np.arange(0, two_halves.max() + 1, 50), labels=None, fontsize=6)
    ax1.set_title('Rose Diagram of the Fractures', y=1.10, fontsize=8, va='bottom')
    
    plt.suptitle('Fracture set %s'%F, y=1.10, fontsize=10)
#    fig3.tight_layout()
    fig3.show()
    fig3.savefig('Figure_03_SteroRoses_%s'%F, dpi=300)
      
#%% BoxPlots All Families Heights

# Define Figure/plot specs
fig4 =plt.figure(figsize=(7,5),facecolor='w')
column = 1     
row = 1 
gs = gridspec.GridSpec(row,column) 

# define subplot axes
ax0 = plt.subplot(gs[0,0]) 

# Create an empty matrix HF((n_row, n_col)) where to store the data:
n_row = df2.shape[0]
n_col = df2['F_Class'].unique().shape[0]
HF = np.empty((n_row, n_col))
HF[:] = np.nan  # fill the matrix with nan

# X and Y are just here to control what is happening in the loop:
X = []
Y = []
column = 0

for F in df2['F_Class'].unique():
    hF = df2.F_H.dropna()[df['F_Class'] == F]#.as_matrix()
    m_row = hF.shape[0]  
    HF[0:m_row, column] = hF
    column = column + 1
    X.append(F)
    Y.append(m_row)

HF = pd.DataFrame(HF, columns=df2['F_Class'].unique())
#print('m_row', m_row)
#print('Y is:', Y)    
#print('X is:', X)
#print('Shape of X:', type(X))  
#print('HF is:', HF)
#print('the shape of HF is:', HF.shape)
#print('hF is:', hF)
#print('the shape of hF is:', hF.shape)
#print('column:', column)

HF.boxplot(ax=ax0, vert=True, patch_artist=None, showmeans=True, rot=90)
ax0.set_title('Boxplot of fracture heights per fracture class', fontsize=8)   
ax0.set_ylabel('Fracture Height', fontsize=8)
ax0.grid(None)
plt.ylim((-10,50))  
fig4.show()
fig4.savefig('Figure_04_BoxPlots_All', dpi=300)

#%%
#           --- PART 3 - BoxPlots = CLASSES and OUTCROPS
#
#%% Box Plots & bi-histograms Per Outcrops:

# Here is a for loop creating a new df per Outcrop:

for name in df2['Outcrop Name'].unique():
    df_outcrop = pd.DataFrame(df2.loc[df['Outcrop Name'] == name])
    
    # Define Figure/plot specs
    fig5 =plt.figure(figsize=(12,4),facecolor='w')
    column = 3     
    row = 2 
    gs = gridspec.GridSpec(row,column) 
    gs.update(wspace=0.45 ,hspace=0)
    
    # Define subplots axes:
    ax0 = plt.subplot(gs[:,0]) 
    ax1 = plt.subplot(gs[:,1])
    ax2 = plt.subplot(gs[0,2])
    ax3 = plt.subplot(gs[1,2])
    
    # Create an empty matrix HF((n_row, n_col)) where to store the data:
    n_row = df_outcrop.shape[0]
    n_col = df_outcrop['F_Class'].unique().shape[0]
    HF_outcrop = np.empty((n_row, n_col))
    HF_outcrop[:] = np.nan  # fill the matrix with nan
    AF_outcrop = np.empty((n_row, n_col))
    AF_outcrop[:] = np.nan  # fill the matrix with nan    
    
    
    # X and Y are just here to control what is happening in the loop:
    X = []
    Y = []
    column1 = 0
    column2 = 0
    
    # Exporting Quantiles as CSV file:
    for OC in df_outcrop['F_Class'].unique():
        Ap_Quantile = (df_outcrop['F_Ap'][df_outcrop['F_Class'] == OC ].quantile([0.1, 0.5, 0.9]))
        H_Quantile = (df_outcrop['F_H'][df_outcrop['F_Class'] == OC ].quantile([0.1, 0.5, 0.9]))
        df_outcrop['Ap_std'] = df_outcrop['F_Ap'][df_outcrop['F_Class'] == OC ].std
        Stats_OC = pd.concat([Ap_Quantile, H_Quantile], axis=1)
        Stats_OC.to_csv('Outcrops_Quantiles' + name + OC + '.csv')

    for F_out in df_outcrop['F_Class'].unique():
        hF = df_outcrop.F_H.dropna()[df_outcrop['F_Class'] == F_out]#.as_matrix()
        m_row = hF.shape[0]  
        HF_outcrop[0:m_row, column1] = hF
        column1 = column1 + 1
        X.append(F_out)
        Y.append(m_row)
    
    HF_outcrop = pd.DataFrame(HF_outcrop, columns=df_outcrop['F_Class'].unique())
    HF_outcrop.boxplot(ax=ax0, vert=True, patch_artist=None, showmeans=True, rot=90)
    
    ax0.set_title('Boxplot of fracture heights per fracture class', fontsize=8)   
    ax0.set_ylim(-10, 50)
    ax0.set_ylabel('Fracture Height', fontsize=8)
    ax0.grid(None)

    for F_out in df_outcrop['F_Class'].unique():
        aF = df_outcrop.F_Ap.dropna()[df_outcrop['F_Class'] == F_out]#.as_matrix()
        m_row = aF.shape[0]  
        AF_outcrop[0:m_row, column2] = aF
        column2 = column2 + 1
        X.append(F_out)
        Y.append(m_row)
        
    AF_outcrop = pd.DataFrame(AF_outcrop, columns=df_outcrop['F_Class'].unique())
    AF_outcrop.boxplot(ax=ax1, vert=True, patch_artist=None, showmeans=True, rot=90)
    
    ax1.set_title('Boxplot of fracture Apertures per fracture class', fontsize=8)   
    ax1.set_ylim(-0.5, 2)
    ax1.set_ylabel('Fracture Aperture', fontsize=8)
    ax1.grid(None)  
  
    # Creating Now the Bi-Histogram:
    ax2.bar(df_outcrop.index, df_outcrop.F_H, color=col_H)
#    ax2.set_ylim(0, 1.05*df_outcrop.F_H.max())
    ax2.set_ylim(0, 50)
    ax2.set_ylabel('Heights (m)', color=col_H, fontsize=8)
    ax2.xaxis.set_ticks_position('none')
    ax2.get_yaxis().set_label_coords(-0.15, 0.5)
    
    ax3.bar(df_outcrop.index, df_outcrop.F_Ap, color=col_Ap)
#    ax3.set_ylim(0, 1.05*df_outcrop.F_Ap.max())
    ax3.set_ylim(0, 1)
    ax3.invert_yaxis()
    ax3.set_ylabel('Apertures (mm)', color=col_Ap, fontsize=8)
    ax3.set_xlabel('Fracture indices', fontsize=8) 
    ax3.get_yaxis().set_label_coords(-0.15, 0.5)
        
    plt.suptitle('Outcrop of %s'%name, y=1.10, fontsize=10)
#    fig5.tight_layout()
    fig5.show()
    fig5.savefig('Figure_05_BoxPlots_Outcrops_%s'%name, dpi=300)

#%%
#           --- PART 4 - BoxPlots = CLASSES and RockType
#
#%% Box Plots & bi-histograms Per RockType:

# Here is a for loop creating a new df per Outcrop:

for RT in df2['Rock_Type'].unique():
    df_rocktype = pd.DataFrame(df2.loc[df['Rock_Type'] == RT])
    
    # Define Figure/plot specs
    fig6 =plt.figure(figsize=(12,4),facecolor='w')
    column = 3     
    row = 2 
    gs = gridspec.GridSpec(row,column) 
    gs.update(wspace=0.45 ,hspace=0)
    
    # STEREOGRAM: Define subplots axes:
    ax0 = plt.subplot(gs[:,0]) 
    ax1 = plt.subplot(gs[:,1])
    ax2 = plt.subplot(gs[0,2])
    ax3 = plt.subplot(gs[1,2])
    
    # Create an empty matrix HF((n_row, n_col)) where to store the data:
    n_row = df_rocktype.shape[0]
    n_col = df_rocktype['F_Class'].unique().shape[0]
    HF_rocktype = np.empty((n_row, n_col))
    HF_rocktype[:] = np.nan  # fill the matrix with nan
    AF_rocktype = np.empty((n_row, n_col))
    AF_rocktype[:] = np.nan  # fill the m
    
    # X and Y are just here to control what is happening in the loop:
    X = []
    Y = []
    column1 = 0
    column2 = 0
  
    # Exporting Quantiles as CSV file:
    for R in df_rocktype['F_Class'].unique():
        Ap_Quantile = (df_rocktype['F_Ap'][df_rocktype['F_Class'] == R ].quantile([0.1, 0.5, 0.9]))
        H_Quantile = (df_rocktype['F_H'][df_rocktype['F_Class'] == R ].quantile([0.1, 0.5, 0.9]))
        Stats_RT = pd.concat([Ap_Quantile, H_Quantile], axis=1)
        Stats_RT.to_csv('Exp_RT_Quantiles' + RT + R + '.csv')
       
    for F_rt in df_rocktype['F_Class'].unique():
        rtF = df_rocktype.F_H.dropna()[df_rocktype['F_Class'] == F_rt]#.as_matrix()
        m_row = rtF.shape[0]  
        HF_rocktype[0:m_row, column1] = rtF
        column1 = column1 + 1
        X.append(F_out)
        Y.append(m_row)
    
    HF_rocktype = pd.DataFrame(HF_rocktype, columns=df_rocktype['F_Class'].unique())
   
    HF_rocktype.boxplot(ax=ax0, vert=True, patch_artist=None, showmeans=True, rot=90)
    ax0.set_title('Boxplot of fracture heights per fracture class', fontsize=8)   
    ax0.set_ylim(-10, 50)
    ax0.set_ylabel('Fracture Height', fontsize=8)
    ax0.grid(None)
    
    for F_rt in df_rocktype['F_Class'].unique():
        rtF = df_rocktype.F_Ap.dropna()[df_rocktype['F_Class'] == F_rt]#.as_matrix()
        m_row = rtF.shape[0]  
        AF_rocktype[0:m_row, column2] = rtF
        column2 = column2 + 1
        X.append(F_out)
        Y.append(m_row)
    
    AF_rocktype = pd.DataFrame(AF_rocktype, columns=df_rocktype['F_Class'].unique())
   
    AF_rocktype.boxplot(ax=ax1, vert=True, patch_artist=None, showmeans=True, rot=90)
    ax1.set_title('Boxplot of fracture Apertures per fracture class', fontsize=8)   
#    ax1.set_ylim(0.5, 1.05*df_outcrop.F_Ap.max())
    ax1.set_ylim(-0.5, 2)
    ax1.set_ylabel('Fracture Aperture', fontsize=8)
    ax1.grid(None)  
       
    # Creating Now the Bi-Histogram:
    ax2.bar(df_rocktype.index, df_rocktype['F_H'].dropna(), color=col_H)
    ax2.set_ylim(0, 50)
    ax2.set_ylabel('Heights (m)', color=col_H, fontsize=8)
    ax2.xaxis.set_ticks_position('none')
    ax2.get_yaxis().set_label_coords(-0.15, 0.5)
    
    ax3.bar(df_rocktype.index, df_rocktype['F_Ap'].dropna(), color=col_Ap)
    #    ax2.set_ylim(0, 1.05*df_outcrop.F_Ap.max())
    ax3.set_ylim(0, 1)
    ax3.invert_yaxis()
    ax3.set_ylabel('Apertures (mm)', color=col_Ap, fontsize=8)
    ax3.set_xlabel('Fracture indices', fontsize=8) 
    ax3.get_yaxis().set_label_coords(-0.15, 0.5)
        
    plt.suptitle('Rock type %s'%RT, y=1.10, fontsize=10)
    #    fig6.tight_layout()
    fig6.show()
    fig6.savefig('Figure_06_BoxPlots_RockTypes_%s'%RT, dpi=300)


#%%
#           --- PART 5 - SCANLINE
#    
#%% SCANLINE
    
# In df calculate (dX, DY) create new columns:
df['dX'] = np.sin(np.radians(df.Surf_Dir))*(df.m_out-df.m_in)
df['dY'] = np.cos(np.radians(df.Surf_Dir))*(df.m_out-df.m_in)

# This 'Conditions' tool is just to change colors with Fracture sets
conditions = [df['F_Class'] == 'F1', df['F_Class'] == 'F2', df['F_Class'] == 'F3', df['F_Class'] == 'F4', df['F_Class'] == 'F5', df['F_Class'] == 'F6', df['F_Class'] == 'F7']
choices = ['cyan', 'green', 'navy', 'red', 'violet', 'darkviolet', 'orange']
df['col'] = np.select(conditions, choices, default= 'black') 
writer = df.col

# Here, creating the new column where to add coordinates:
df['Xin'] = np.nan
df['Yin'] = np.nan
df['Xout'] = np.nan
df['Yout'] = np.nan

# Here, I add a first row = 0 for (XY) at 0 = (00)
edd = df.iloc[0:1,:].copy(deep=True)
edd[:] = 0
edd.index = [0]
df = pd.concat([edd, df])
df = df.reset_index(drop=True)

for i in range (len(df['F_Az'])):
    if df['F_Az'].iloc[i] > 180:
        df['F_Az'].iloc[i] = df['F_Az'].iloc[i]-180

df['fx'] = np.sin(np.deg2rad(df['F_Az'])) * df['F_H']
df['fy'] = np.cos(np.deg2rad(df['F_Az'])) * df['F_H']

for name in df['Out_Code'].dropna().unique():
   
#    print('OC name', name)    
    # Define Figure/plot specs
    fig7 = plt.figure(figsize=(7,4))
    row = 1 
    column = 1
    gs = gridspec.GridSpec(row,column) 
    
    # Define subplots axes:
    ax0 = plt.subplot(gs[0,0]) 
    ax0 = fig7.add_subplot(gs[0,0])
    
    # Tmp outcrop df:
    df_outcrop = pd.DataFrame(df.loc[df['Out_Code'] == name])
    
    edd = df_outcrop.iloc[0:1,:].copy(deep=True)
    edd[:] = 0
    edd.index = [0]
    df_outcrop = pd.concat([edd, df_outcrop])
    df_outcrop = df_outcrop.reset_index(drop=True)
        
    # Find only the unique combination of m_in and m_out in outcrop df:
    test = df_outcrop[['m_in','m_out']].drop_duplicates().copy(deep=True)

    # get the indexes over which we have unique combinations
    indexes = test.index.values.tolist()
    
    # iterate over i calling the value of the indexes (indexes[i])
    for i in range(len(indexes)-1):
        df_outcrop['Xin'].iloc[[indexes[0]]] = 0
        df_outcrop['Yin'].iloc[[indexes[0]]] = 0
        df_outcrop['Xin'].iloc[[indexes[i+1]]] = df_outcrop['Xout'].iloc[[indexes[i]]].values
        df_outcrop['Yin'].iloc[[indexes[i+1]]] = df_outcrop['Yout'].iloc[[indexes[i]]].values
        df_outcrop['Xout'].iloc[[indexes[i+1]]] = df_outcrop['Xin'].iloc[[indexes[i+1]]].values + df_outcrop['dX'].iloc[[indexes[i+1]]].values
        df_outcrop['Yout'].iloc[[indexes[i+1]]] = df_outcrop['Yin'].iloc[[indexes[i+1]]].values + df_outcrop['dY'].iloc[[indexes[i+1]]].values
  
    df_outcrop['Xin'] = df_outcrop['Xin'].fillna(method='ffill')
    df_outcrop['Yin'] = df_outcrop['Yin'].fillna(method='ffill')
    df_outcrop['Xout'] = df_outcrop['Xout'].fillna(method='ffill')
    df_outcrop['Yout'] = df_outcrop['Yout'].fillna(method='ffill')

    for j in range(1, len(df_outcrop)):
        df_outcrop['a'] = (df_outcrop['Xout'] - df_outcrop['Xin']) / (df_outcrop['Q'] + 1)
        df_outcrop['b'] = (df_outcrop['Yout'] - df_outcrop['Yin']) / (df_outcrop['Q'] + 1)
        
        for qi in range(1, df_outcrop['Q'][j]+1): 
            xo = df_outcrop['Xin'].iloc[j] + qi * df_outcrop['a'].loc[j]
            yo = df_outcrop['Yin'].iloc[j] + qi * df_outcrop['b'].iloc[j]             
            ax0.quiver(xo, yo, df_outcrop['fx'].iloc[j], df_outcrop['fy'].iloc[j], color= df_outcrop['col'].iloc[j], width=0.001, headlength=0, headaxislength=0, scale=None)
            ax0.quiver(xo, yo, -df_outcrop['fx'].iloc[j], -df_outcrop['fy'].iloc[j], color= df_outcrop['col'].iloc[j], width=0.001, headlength=0, headaxislength=0, scale=None)
              
#    fig7.tight_layout()
    ax0.plot(df_outcrop['Xout'], df_outcrop['Yout'], 'g')  
#    ax0.fill_between(df_outcrop['Xout'], df_outcrop['Yout'], df_outcrop['Yout'].max(), facecolor='tan', edgecolor='black', label='outcrop', alpha=.3)
    ax0.set_xlabel('x [m]')
    ax0.set_ylabel('y [m]')
    ax0.grid()
    ax0.axis('square')
    ax0.set_title('ax0 Outcrop of %s'%name)
    fig7.savefig('Figure_07_Scanline_%s'%name, dpi=300)
#    plt.suptitle('PLT of %s'%name, y=1.10, fontsize=10)
#    ax0.set_title('Fracture_Scanline')
               
print('*** ScanLine Done ***')
print()
#%%
#           --- PART 5 B - SCANLINE to DFN
#    
#%% SCANLINE
    
coord_xo = []
coord_yo = []
test_F_F = []
kink = pd.DataFrame(columns = ['coord_xo', 'coord_yo', 'F_F', 'F_Ap', 'F_H' ])


# In df calculate (dX, DY) create new columns:
df['dX'] = np.sin(np.radians(df.Surf_Dir))*(df.m_out-df.m_in)
df['dY'] = np.cos(np.radians(df.Surf_Dir))*(df.m_out-df.m_in)

# This 'Conditions' tool is just to change colors with Fracture sets
conditions = [df['F_Class'] == 'F1', df['F_Class'] == 'F2', df['F_Class'] == 'F3', df['F_Class'] == 'F4', df['F_Class'] == 'F5', df['F_Class'] == 'F6', df['F_Class'] == 'F7']
choices = ['cyan', 'green', 'navy', 'red', 'violet', 'darkviolet', 'orange']
df['col'] = np.select(conditions, choices, default= 'black') 
writer = df.col

# Here, creating the new column where to add coordinates:
df['Xin'] = np.nan
df['Yin'] = np.nan
df['Xout'] = np.nan
df['Yout'] = np.nan

# Here, I add a first row = 0 for (XY) at 0 = (00)
edd = df.iloc[0:1,:].copy(deep=True)
edd[:] = 0
edd.index = [0]
df = pd.concat([edd, df])
df = df.reset_index(drop=True)

for i in range (len(df['F_Az'])):
    if df['F_Az'].iloc[i] > 180:
        df['F_Az'].iloc[i] = df['F_Az'].iloc[i]-180

df['fx'] = np.sin(np.deg2rad(df['F_Az'])) * df['F_H']
df['fy'] = np.cos(np.deg2rad(df['F_Az'])) * df['F_H']

for name in df['Out_Code'].dropna().unique():
   
#    print('OC name', name)    
    # Define Figure/plot specs
    fig8 = plt.figure(figsize=(7,4))
    row = 1 
    column = 1
    gs = gridspec.GridSpec(row,column) 
    
    # Define subplots axes:
    ax0 = plt.subplot(gs[0,0]) 
    ax0 = fig8.add_subplot(gs[0,0])
    
    # Tmp outcrop df:
    df_outcrop = pd.DataFrame(df.loc[df['Out_Code'] == name])
    
    edd = df_outcrop.iloc[0:1,:].copy(deep=True)
    edd[:] = 0
    edd.index = [0]
    df_outcrop = pd.concat([edd, df_outcrop])
    df_outcrop = df_outcrop.reset_index(drop=True)
        
    # Find only the unique combination of m_in and m_out in outcrop df:
    test = df_outcrop[['m_in','m_out']].drop_duplicates().copy(deep=True)

    # get the indexes over which we have unique combinations
    indexes = test.index.values.tolist()
    
    # iterate over i calling the value of the indexes (indexes[i])
    for i in range(len(indexes)-1):
#        print('i', i)
#        print('indexes', indexes)
        df_outcrop['Xin'].iloc[[indexes[0]]] = 0
        df_outcrop['Yin'].iloc[[indexes[0]]] = 0
        df_outcrop['Xin'].iloc[[indexes[i+1]]] = df_outcrop['Xout'].iloc[[indexes[i]]].values
        df_outcrop['Yin'].iloc[[indexes[i+1]]] = df_outcrop['Yout'].iloc[[indexes[i]]].values
        df_outcrop['Xout'].iloc[[indexes[i+1]]] = df_outcrop['Xin'].iloc[[indexes[i+1]]].values + df_outcrop['dX'].iloc[[indexes[i+1]]].values
        df_outcrop['Yout'].iloc[[indexes[i+1]]] = df_outcrop['Yin'].iloc[[indexes[i+1]]].values + df_outcrop['dY'].iloc[[indexes[i+1]]].values
  
    df_outcrop['Xin'] = df_outcrop['Xin'].fillna(method='ffill')
    df_outcrop['Yin'] = df_outcrop['Yin'].fillna(method='ffill')
    df_outcrop['Xout'] = df_outcrop['Xout'].fillna(method='ffill')
    df_outcrop['Yout'] = df_outcrop['Yout'].fillna(method='ffill')

    for j in range(1, len(df_outcrop)):
        df_outcrop['a'] = (df_outcrop['Xout'] - df_outcrop['Xin']) / (df_outcrop['Q'] + 1)
        df_outcrop['b'] = (df_outcrop['Yout'] - df_outcrop['Yin']) / (df_outcrop['Q'] + 1)
#        print('j', j)
#        print('a', df_outcrop.a.iloc[j])
#        print('b', df_outcrop.b.iloc[j])  
        
        for qi in range(1, df_outcrop['Q'][j]+1): 
            xo = df_outcrop['Xin'].iloc[j] + qi * df_outcrop['a'].loc[j]
            yo = df_outcrop['Yin'].iloc[j] + qi * df_outcrop['b'].iloc[j]
            coord_xo = np.append(coord_xo, xo)
            coord_yo = np.append(coord_yo, yo)
            test_F_F = np.append(test_F_F, df_outcrop['F_F'].iloc[j] )
   
#            ax0.quiver(xo, yo, df_outcrop['fx'].iloc[j], df_outcrop['fy'].iloc[j], color= df_outcrop['col'].iloc[j], width=0.001, linestyle= '--', headlength=0, headaxislength=0, scale=0.1)
#            ax0.quiver(xo, yo, -df_outcrop['fx'].iloc[j], -df_outcrop['fy'].iloc[j], color= df_outcrop['col'].iloc[j], width=0.001, linestyle= '--', headlength=0, headaxislength=0, scale=0.1)
            
            ax0.quiver(xo, yo, df_outcrop['fx'].iloc[j], df_outcrop['fy'].iloc[j], color= df_outcrop['col'].iloc[j], width=0.001, headlength=0, headaxislength=0, scale=0.1)
            ax0.quiver(xo, yo, -df_outcrop['fx'].iloc[j], -df_outcrop['fy'].iloc[j], color= df_outcrop['col'].iloc[j], width=0.001, headlength=0, headaxislength=0, scale=0.1)   
            
#    fig8.tight_layout()
    ax0.plot(df_outcrop['Xout'], df_outcrop['Yout'], 'g')  
#    ax0.fill_between(df_outcrop['Xout'], df_outcrop['Yout'], df_outcrop['Yout'].max(), facecolor='tan', edgecolor='black', label='outcrop', alpha=.3)
    ax0.set_xlabel('x [m]')
    ax0.set_ylabel('y [m]')
    ax0.grid()
    ax0.axis('square')
    ax0.set_title('ax0 Outcrop of %s'%name)
    fig8.savefig('Figure_08_Scanline_%s'%name, dpi=300)
#    plt.suptitle('PLT of %s'%name, y=1.10, fontsize=10)
#    ax0.set_title('Fracture_Scanline')
               
print('*** ScanLine Done ***')
print()



#%%
#           --- PART 6 - DFN
#    
#%% DFN
    
# In df calculate (dX, DY) create new columns:
df['dX'] = np.sin(np.radians(df.Surf_Dir))*(df.m_out-df.m_in)
df['dY'] = np.cos(np.radians(df.Surf_Dir))*(df.m_out-df.m_in)

# This is a short loop to give a color code based on F_Aperture ranges:
def Ap_class(row):
    if row['F_Ap'] == 0:
        return 'grey'
    elif 0.001 <= row['F_Ap'] < 0.1:
        return 'powderblue'
    elif 0.01 <= row['F_Ap'] < 1:
        return 'deepskyblue' 
    elif 1 <= row['F_Ap'] <= 5:
        return 'mediumblue' 
    else:
        return 'nan'

df = df.assign(coldfn=df.apply(Ap_class, axis=1))
df2 = df2.assign(coldfn=df.apply(Ap_class, axis=1))

#conditions = [df['F_Ap'] == 0, (0 < df['F_Ap'] < 0.1), (0.1 <= df['F_Ap'] < 1), (1 <= df['F_Ap'] <= 5)]
#conditions = [df['F_Ap'] == 0, df['F_Ap'] == (0.001:0.1), df['F_Ap'] == (0.1:1), df['F_Ap'] == (1:5)]

#choices = ['grey', 'deepskyblue']

#df['coldfn'] = np.select(conditions, choices, default= 'black') 
#writer = df.col

# Here, creating the new column where to add coordinates:
df['Xin'] = np.nan
df['Yin'] = np.nan
df['Xout'] = np.nan
df['Yout'] = np.nan

# Here, I add a first row = 0 for (XY) at 0 = (00)
edd = df.iloc[0:1,:].copy(deep=True)
edd[:] = 0
edd.index = [0]
df = pd.concat([edd, df])
df = df.reset_index(drop=True)

for i in range (len(df['F_Az'])):
    if df['F_Az'].iloc[i] > 180:
        df['F_Az'].iloc[i] = df['F_Az'].iloc[i]-180

df['fxmean'] = np.sin(np.deg2rad(df['F_Az_mean'])) * df['F_H']
df['fymean'] = np.cos(np.deg2rad(df['F_Az_mean'])) * df['F_H']

for name in df['Out_Code'].dropna().unique():
   
#    print('OC name', name)    
    # Define Figure/plot specs
    fig9 = plt.figure(figsize=(7,4))
    row = 1 
    column = 1
    gs = gridspec.GridSpec(row,column) 
    
    # Define subplots axes:
    ax0 = plt.subplot(gs[0,0]) 
    ax0 = fig9.add_subplot(gs[0,0])
    
    # Tmp outcrop df:
    df_outcrop = pd.DataFrame(df.loc[df['Out_Code'] == name])
    
    edd = df_outcrop.iloc[0:1,:].copy(deep=True)
    edd[:] = 0
    edd.index = [0]
    df_outcrop = pd.concat([edd, df_outcrop])
    df_outcrop = df_outcrop.reset_index(drop=True)
        
    # Find only the unique combination of m_in and m_out in outcrop df:
    test = df_outcrop[['m_in','m_out']].drop_duplicates().copy(deep=True)

    # get the indexes over which we have unique combinations
    indexes = test.index.values.tolist()
    
    # iterate over i calling the value of the indexes (indexes[i])
    for i in range(len(indexes)-1):
        df_outcrop['Xin'].iloc[[indexes[0]]] = 0
        df_outcrop['Yin'].iloc[[indexes[0]]] = 0
        df_outcrop['Xin'].iloc[[indexes[i+1]]] = df_outcrop['Xout'].iloc[[indexes[i]]].values
        df_outcrop['Yin'].iloc[[indexes[i+1]]] = df_outcrop['Yout'].iloc[[indexes[i]]].values
        df_outcrop['Xout'].iloc[[indexes[i+1]]] = df_outcrop['Xin'].iloc[[indexes[i+1]]].values + df_outcrop['dX'].iloc[[indexes[i+1]]].values
        df_outcrop['Yout'].iloc[[indexes[i+1]]] = df_outcrop['Yin'].iloc[[indexes[i+1]]].values + df_outcrop['dY'].iloc[[indexes[i+1]]].values
  
    df_outcrop['Xin'] = df_outcrop['Xin'].fillna(method='ffill')
    df_outcrop['Yin'] = df_outcrop['Yin'].fillna(method='ffill')
    df_outcrop['Xout'] = df_outcrop['Xout'].fillna(method='ffill')
    df_outcrop['Yout'] = df_outcrop['Yout'].fillna(method='ffill')
    
    ax0.plot(df['Xout'], df['Yout'], 'k')  
    
    kinklist = []
    
    for j in range(1, len(df_outcrop)):
        df_outcrop['a'] = (df_outcrop['Xout'] - df_outcrop['Xin']) / (df_outcrop['Q'] + 1)
        df_outcrop['b'] = (df_outcrop['Yout'] - df_outcrop['Yin']) / (df_outcrop['Q'] + 1)
        
        for qi in range(1, df_outcrop['Q'][j]+1): 
            xo = df_outcrop['Xin'].iloc[j] + qi * df_outcrop['a'].loc[j]
            yo = df_outcrop['Yin'].iloc[j] + qi * df_outcrop['b'].iloc[j]
            kinklist.append({'xo': xo, 'yo': yo, 'MD': df_outcrop['Xout'].iloc[j], 'fx': df_outcrop['fx'].iloc[j], 'fy': df_outcrop['fy'].iloc[j], 'fxmean': df_outcrop['fxmean'].iloc[j], 'fymean': df_outcrop['fymean'].iloc[j], 'fH': df_outcrop['F_H'].iloc[j], 'fap': df_outcrop['F_Ap'].iloc[j], 'fazmean': df_outcrop['F_Az_mean'].iloc[j], 'faz': df_outcrop['F_Az'].iloc[j], 'fdipmean': df_outcrop['F_Dip_mean'].iloc[j], 'fdip': df_outcrop['F_Dip'].iloc[j], 'fapcol': df_outcrop['coldfn'].iloc[j], 'f_col': df_outcrop['col'].iloc[j], 'outCode': df_outcrop['Out_Code'].iloc[j], 'F_Class': df_outcrop['F_Class'].iloc[j]})                            
   
            ax0.quiver(xo, yo, df_outcrop['fxmean'].iloc[j], df_outcrop['fymean'].iloc[j], color= df_outcrop['coldfn'].iloc[j], width=0.001, linestyle= '--', headlength=0, headaxislength=0, scale=0.1)
            ax0.quiver(xo, yo, -df_outcrop['fxmean'].iloc[j], -df_outcrop['fymean'].iloc[j], color= df_outcrop['coldfn'].iloc[j], width=0.001, linestyle= '--', headlength=0, headaxislength=0, scale=0.1)

        df_kink = pd.DataFrame.from_dict(kinklist)
#    df_kink['x1'] = df_kink['xo'] + df_kink['fx']
#    df_kink['y1'] = df_kink['yo'] + df_kink['fy']
   
    df_kink.to_csv('kink' + str(name) + '.csv')                      
            
#    fig9.tight_layout()
    ax0.plot(df_outcrop['Xin'], df_outcrop['Yin'], 'g')
    ax0.plot(df_outcrop['Xout'], df_outcrop['Yout'], 'g')
    ax0.set_xlabel('x [m]')
    ax0.set_ylabel('y [m]')
    ax0.axis('square')
    ax0.set_title('DFN for Outcrop %s'%name)
    fig9.savefig('Figure_09_DFN_%s'%name, dpi=300)
    fig9.savefig('Figure_09_DFN' + str(name) + '.svg')
    
    tmp = pd.concat([df_outcrop['Xout'], df_outcrop['Yout']], axis=1)
    tmp.to_csv('Scl_survey' + str(name) + '.csv')
#    plt.suptitle('PLT of %s'%name, y=1.10, fontsize=10)     
        
print('*** DFN Done ***')
print()

#%%
#           --- PART 7 - Stereonets & Rose per outcrop
#    
#%% STEREOPLOTS & ROSE DIAGRAM  per ['Outcrop Name']!

# Here is a for loop going in F_Class and pick-out all different values: F1, F2,..F7 and Fnan:
for Outname in df2['Outcrop Name'].unique():
#for F in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']:
    sF = df2.F_Az[df['Outcrop Name'] == Outname]
    dF = df2.F_Dip[df['Outcrop Name'] == Outname]
    
    # Define Figure/plot specs
    fig10 = plt.figure(figsize=(7,4))
    row = 1 
    column = 2
    gs = gridspec.GridSpec(row,column) 
    
    # STEREOGRAM: Define subplots axes:
    ax0 = plt.subplot(gs[0,0]) 
    ax1 = plt.subplot(gs[0,1]) 
    
    # STEREOGRAM: Density Contouring:
    ax0 = fig10.add_subplot(gs[0,0], projection='stereonet')
    ax0.plane(sF, dF, c='k', linewidth=0.1)
    ax0.density_contourf(sF.dropna(), dF.dropna(), measurement='poles', cmap='Reds')
    ax0.set_title('Steronet & Density contour of the Poles', y=1.10, fontsize=8, va='bottom')
    ax0.grid()
    
    # ROSE DIAGRAM: Calculate the number of strikes every 10° using np.hist 
    bin_edges = np.arange(-5, 366, 10)
    number_of_strikes, bin_edges = np.histogram(sF.dropna(), bin_edges)
    
    # ROSE DIAGRAM: Sum the last value with the first value.
    number_of_strikes[0] += number_of_strikes[-1]
    
    # ROSE DIAGRAM: Sum the first half 0-180° with the second half 180-360° to achieve the "mirrored behavior" of Rose Diagrams.
    half = np.sum(np.split(number_of_strikes[:-1], 2), 0)
    two_halves = np.concatenate([half, half])
    
    # ROSE DIAGRAM: Create the Rose Diagram:
    
    ax1 = fig10.add_subplot(gs[0,1], projection='polar')
    ax1.bar(np.deg2rad(np.arange(0, 360, 10)), two_halves, width=np.deg2rad(10), bottom=0.0, color='.8', edgecolor='k')
    ax1.set_theta_zero_location('N')    
    ax1.set_theta_direction(-1)
    ax1.set_thetagrids(np.arange(0, 360, 45), labels=np.arange(0, 360, 45))
    ax1.set_rgrids(np.arange(0, two_halves.max() + 1, 50), labels=None, fontsize=6)
    ax1.set_title('Rose Diagram of the Fractures', y=1.10, fontsize=8, va='bottom')
    
    plt.suptitle('Fracture set %s'%Outname, y=1.10, fontsize=10)
    fig10.show()
    fig10.savefig('Figure_10_SteroRoses_Outcrops_%s'%Outname, dpi=300)
