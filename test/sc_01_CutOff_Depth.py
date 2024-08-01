# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 22:56:50 2023

@author: kehok
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import geopandas as gpd
import pickle
from sklearn.cluster import k_means

# KetilH stuff
from earthquake.quake import cut_off_depth

#---------------------------
#  Input and output folders
#---------------------------

block = True

shp = '../data/shp/'
pkl = '../data/pkl/'
csv = '../data/csv/'
excel = '../data/excel/'
png = 'png/'

#-------------------------------------
#   Read input data
#-------------------------------------

# Read earthquake hypocenters: OLD FILE
# fname = 'Earthquake_Data_1981_2019_in_Basin_utm11N.csv'
# df = pd.read_csv(csv+fname)
# df['depth'] = -df['z'] 
# n_clu = 4*5

# Read earthquake hypocenters: NEW FILE
fname = 'SCEC_Hauksson_1981_2022_Relocated_in_Basin.xlsx'
with pd.ExcelFile(excel+fname) as fid:
    df = pd.read_excel(fid)
    df['depth'] = 1e3*df['depth']
    df['z'] = -df['depth']            # depth>0, z<0 blow MSL
    n_clu = 4*8

#-------------------------------------
#   Read some shp files for plotting 
#-------------------------------------

# Salton Sea shoreline
file_ss = 'SaltonSea.shp'
salton_11N = gpd.read_file(shp + file_ss)
    
# Scale meters to km
scl = 1e-3
salton_11N.geometry = salton_11N.geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))
salton = salton_11N

#------------------------------------
#   Clustering
#------------------------------------
        
# Spatial clustering using kMeans
key_x, key_y = 'x', 'y'
centroid, clu_id, inertia = k_means(df[[key_x, key_y]], n_clu, 
                                    n_init='auto', algorithm='elkan')

df['clu_id'] = clu_id.astype(int)

fname_out = 'Earthquake_Data_with_CluID_utm11N.xlsx'
with pd.ExcelWriter(excel+fname_out) as fid:
    df.to_excel(fid, index=False)

#------------------------------------
#  PLot clusters and
#  cut-off depth per cluster
#------------------------------------

figa, ax = plt.subplots(1, figsize=(12,12))

style = 'log'
nbin = 20

scl = 1e-3
for idd in df['clu_id'].unique():
    ind = df['clu_id']==idd
    dfc = df[ind]
    ndd = dfc.shape[0]
    print(f'idd = {idd:2d}: n_eq = {dfc.shape[0]}')
    
    ax.scatter(scl*dfc['x'], scl*dfc['y'], marker='.', c=idd*np.ones((ndd)), 
               cmap=cm.tab20b, label=f'cluster {idd}', vmin=0, vmax=n_clu-1)
  
    # Mean cut-off per cluster
    depth_mean = np.mean(dfc['depth'])
    depth_std  = np.std(dfc['depth'])
    depth_max  = np.max(dfc['depth'])
    df.loc[ind, 'depth_mean'] = depth_mean
    df.loc[ind, 'depth_std']  = depth_std
    df.loc[ind, 'depth_max']  = depth_max

for jj in range(n_clu):
    xc, yc = scl*centroid[jj,0], scl*centroid[jj,1]
    ax.text(xc, yc, f'{jj}')

salton.plot(ax=ax, color='c', linewidth=0.95, label='shoreline')
ax.set_title('kmeans clustering')
ax.legend()
ax.set_xlabel('x [km] (11N)')
ax.set_ylabel('y [km] (11N)')
ax.axis('scaled')

figb, bx = plt.subplots(1, figsize=(12,12))
cut_off = scl*(df['depth_mean'] + df['depth_std'])
sc = bx.scatter(scl*df['x'], scl*df['y'], marker='.', c=cut_off)#, vmin=0.85, vmax=1.05)
cb = bx.figure.colorbar(sc, ax=bx)
salton.plot(ax=bx, color='c', linewidth=0.95, label='shoreline')
bx.set_xlabel('x [km] (11N)')
bx.set_ylabel('y [km] (11N)')
bx.set_title('Cut-off depth per cluster [km]')
bx.axis('scaled')

figa.tight_layout(pad=1)
figb.tight_layout(pad=1)
figa.savefig(png + 'EQ_Clusters.png')
figb.savefig(png + 'EQ_CutOff_Depth_per_Cluster.png')

#---------------------------------------------------
#   Cut-off depth 
#---------------------------------------------------

x1, x2 =  np.floor(np.min(df['x'])), np.floor(np.max(df['x'])) + 1e3
y1, y2 =  np.floor(np.min(df['y'])), np.floor(np.max(df['y'])) + 1e3

# Compute cut-off depth
dx, dy = 500., 500.
nx = int(np.ceil((x2-x1)/dx))
ny = int(np.ceil((y2-y1)/dy))

x = np.linspace(x1,x1+(nx-1)*dx, nx)
y = np.linspace(y1,y1+(ny-1)*dy, ny)

eq = cut_off_depth(df, x, y, key_z='depth', verbose=1)

with open(pkl + 'Cut_off_Depth.pkl', 'wb') as fid:
    pickle.dump(eq, fid)

#-----------------------------------
#  Plot cut-off depth 
#-----------------------------------

fig, axs = plt.subplots(2,2, figsize=(14,10))
for jj in range(3):
    ax = axs.ravel()[jj]
    xp = scl*eq.gx.ravel()
    yp = scl*eq.gy.ravel()
    zp = scl*eq.grd[jj].ravel() 
    sc = ax.scatter(xp, yp, marker='.', c=zp, vmin=6, vmax=12)
    cb = ax.figure.colorbar(sc, ax=ax)
    salton.plot(ax=ax, color='c', linewidth=0.95, label='shoreline')
    ax.set_xlabel('x [km] (11N)')
    ax.set_ylabel('y [km] (11N)')
    ax.set_title(f'eq cutoff ({eq.label[jj]})')
    ax.axis('scaled')

jj = 2
ax = axs.ravel()[3]
xp = scl*eq.gx.ravel()
yp = scl*eq.gy.ravel()
zp = eq.grd[jj].ravel() - eq.grd[0].ravel() 
sc = ax.scatter(xp, yp, marker='.', c=zp)
cb = ax.figure.colorbar(sc, ax=ax)
salton.plot(ax=ax, color='c', linewidth=0.95, label='shoreline')
ax.set_xlabel('x [km] (11N)')
ax.set_ylabel('y [km] (11N)')
ax.set_title(f'eq cutoff diff {eq.label[jj]} - {eq.label[0]}')
ax.axis('scaled')

fig.suptitle('Earthquake cut-off depth (3 alternatives)')
fig.tight_layout(pad=1)
fig.savefig(png + 'EQ_CutOff_Depth_Comparison.png')

plt.show(block=block)


    
    