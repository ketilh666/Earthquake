# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:18:44 2024

@author: KEHOK
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as img
import pandas as pd
import geopandas as gpd
import pickle
import os

from sklearn.cluster import k_means
import geodetic_conversion as gc

# KetilH stuff
from smooties import smooth2d
from gravmag.common import gridder
#from earthquake.quake import two_point_corr
import earthquake.quake as quake

#---------------------------
#  Input and output folders
#---------------------------

block = True

plot_cult = False

shp = '../data/shp/'
shp_scec = '../data/CFm6.1_release_2023/obj/preferred/traces/shp/'
pics = '../data/pics/'
irap = '../data/irap/'

pkl = '../data/pkl/'
csv = '../data/csv/'
excel = '../data/excel/'

png = 'png/'
if not os.path.isdir(png): os.mkdir(png)

#---------------------------
#   Run pars
#---------------------------

# Run two-point correlation on AOI?
run_corr = True 

# Index of AOI in level 1 clustering
kdd = 31
clu_list = [8, 17, kdd]
# clu_list = [8, 26, kdd]
# clu_list = [8, 12, kdd]

# Re-run the level 2 clustering?
run_clu2 = False

n_clu2 = 3
fname_clu2 = 'Level2_Clustered_Data_n3.pkl'
clu_list2 = [0,1] 

# Clusters to be lumed together for enitre SSGF
n_clu0 = 1 
clu_list0 = [4, 8, 12, 13, 17, 18, 20, 26, 31]  

zmin =  500.0 # Min hypocenter depth
zmax = 3000.0 # Max hypocenter depth
# zmax = 9000.0 # Max hypocenter depth

# Level 1 clusters
dr = 40.
rmax = 2000.

rscl = 2.0

# All data
dr0 = dr*rscl
rmax0 = rmax*rscl

# Level 2 clusters
dr2 = dr/rscl
rmax2 = rmax/rscl

verbose = 1

#######################################################
#
#  Importing data and plotting maps of clusters
#
#######################################################

#-----------------------------------s
#   Cut-off depths
#-----------------------------------

with open(pkl + 'Cut_off_Depth.pkl', 'rb') as fid:
    eq = pickle.load(fid)

with open(pkl + 'TopBasement.pkl', 'rb') as fid:
    gm = pickle.load(fid)

#----------------------------------------
#   Read some shape files for plotting
#----------------------------------------

scl = 1e-3 # m to km

rhyolites = gpd.read_file(shp + 'rhyolites.shp')
rhyolites.geometry = rhyolites.geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))

# Discretized faults
file_list_faults = [
  'BrawleyFaults.shp',
  'BrownFaults.shp',
  'CalpatriaFaults.shp',
  'RedFaults.shp',
  'RedHillFaults.shp',
  'SaltonSeaFaults.shp',
  'SuperstitionMoountainFaults.shp',
  'YellowFaults.shp'   
 ]

name_list_faults = [ff.split('.')[0] for ff in file_list_faults]
fault_list = [None for ff in file_list_faults]
for jj, fn in enumerate(file_list_faults):
    fault_list[jj] = gpd.read_file(shp + fn)
    fault_list[jj].geometry = fault_list[jj].geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))

# Faults from SCEC
file_trc = 'CFM6.1_traces.shp' # Faults identified on the surface
file_bld = 'CFM6.1_blind.shp'  # Blind faults
trc_wgs = gpd.read_file(shp_scec + file_trc)
bld_wgs = gpd.read_file(shp_scec + file_bld)

# Transform to UTM11N or vice versa
epsg_wgs =  4326
epsg_11N = 26711
trc_11N = trc_wgs.to_crs({'init': f'epsg:{epsg_11N}'})
bld_11N = bld_wgs.to_crs({'init': f'epsg:{epsg_11N}'})

# Salton Sea shoreline
salton_11N = gpd.read_file(shp + 'SaltonSea.shp')
bsz_11N = gpd.read_file(shp + 'BSZ.shp')

# License boundaries
bhe_11N = gpd.read_file(shp + 'BHE.shp')
ctr_11N = gpd.read_file(shp + 'CTR.shp')
esm_11N = gpd.read_file(shp + 'ESM.shp')
phoenix_11N = gpd.read_file(shp + 'Phoenix.shp')
    
# Scale meters to km
trc_11N.geometry = trc_11N.geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))
bld_11N.geometry = bld_11N.geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))
salton_11N.geometry = salton_11N.geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))
bsz_11N.geometry = bsz_11N.geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))
bhe_11N.geometry = bhe_11N.geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))
ctr_11N.geometry = ctr_11N.geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))
esm_11N.geometry = esm_11N.geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))
phoenix_11N.geometry = phoenix_11N.geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))

#-------------------------------
#  Read well data from Argonne
#-------------------------------

fname_wells = 'Argonne_SaltonSea_Subset.xlsx'

with pd.ExcelFile(excel + fname_wells) as fid:
    df_wells = pd.read_excel(fid)    
    ind_to_plot = [2,3] # selected wells
    df_wells = df_wells.iloc[ind_to_plot]
    
# Compute utm coords
lon = np.array(df_wells['lon'])
lat = np.array(df_wells['lat'])
df_wells['x'], df_wells['y'] = gc.wgs_to_utm(lat, lon, 11, 'N')
    
df_wells.loc[2, ['x', 'y']] = [632287.00, 3675238.00]
df_wells.loc[4, ['name', 'label', 'x', 'y']] = ['HR_13_1', 'HR_13_1', 632613.62, 3675079.62]
df_wells.loc[5, ['name', 'label', 'x', 'y']] = ['HR_13_2', 'HR_13_2', 633213.08, 3675078.92]
df_wells.loc[6, ['name', 'label', 'x', 'y']] = ['HR_13_1', 'HR_13_3', 632885.50, 3675079.49]

#-------------------------------------
# Read EQ input
#-------------------------------------

fname = 'Earthquake_Data_with_CluID_and_b_value_utm11N.xlsx'
with pd.ExcelFile(excel+fname) as fid:
    df_all = pd.read_excel(fid)
    # df_all['depth'] = -df_all['z']

#--------------------------
# lon/lat or UTM
#--------------------------

key_x, key_y = 'x', 'y'
key_id  = 'clu_id'
key_id2 = 'clu_id2'

# Reset:
df = df_all.copy()
n_clu = df[key_id].max() + 1

# Filter on mc
# mc, mc2 = 1.75, 6.21
mc, mc2  = 0.0, 3.0
ind = (df.magnitude>=mc) & (df.magnitude<=mc2)
df = df[ind]

#-----------------------------------------------
# Level 2 clustering (clusters in clusters)
#-----------------------------------------------

# Rerun level 2 clustering
if run_clu2:

    # Level 1 cluster of interest
    df2 = df[df[key_id]==kdd].copy()

    # Level 2 spatial clustering using kMeans
    centroid, clu_id2, inertia = k_means(df2[[key_x, key_y]], n_clu2, 
                                        n_init='auto', algorithm='elkan')

    df2[key_id2] = clu_id2.astype(int)

    clu_list2 = [jj for jj in range(n_clu2)]

    with open(pkl + fname_clu2, 'wb') as fid:
        pickle.dump([df2, n_clu2], fid)

else:

    with open(pkl + fname_clu2, 'rb') as fid:
        df2, n_clu2 = pickle.load(fid)

#------------------------------------
# PLot the clusters
#------------------------------------

# PLot level 1 clustering
# xtnt = scl*np.array([eq.x[0], eq.x[-1], eq.y[0], eq.y[-1]])
xtnt = np.array([610, 650, 3630, 3700])
cent_x = np.zeros(n_clu)
cent_y = np.zeros(n_clu)

figa, ax = plt.subplots(1, figsize=(10,10))

sm = 1.0
# for idd in df[key_id].unique():
ssgf_list = []
for jj, idd in enumerate(clu_list0):
    ind = df[key_id]==idd
    dfc = df[ind]
    ssgf_list.append(dfc) # For later use
    ndd = dfc.shape[0]
    cent_x[idd] = np.nanmean(dfc[key_x])
    cent_y[idd] = np.nanmean(dfc[key_y])
    print(f'idd = {idd:2d}: n_eq = {dfc.shape[0]}')
    
    ax.scatter(scl*dfc[key_x], scl*dfc[key_y], marker='.', c=idd*np.ones((ndd)), 
            s=sm*dfc.magnitude, cmap=cm.tab20b, vmin=0, vmax=n_clu-1)

ax.set_title('kmeans clustering level 1')
for jj in range(n_clu):
    ax.text(scl*cent_x[jj], scl*cent_y[jj], f'{jj}')

# Level 2 clusters
xtnt2 = np.array([630, 645, 3670, 3685])
cent_x2 = np.zeros(n_clu2)
cent_y2 = np.zeros(n_clu2)

figb, bx = plt.subplots(1, figsize=(8,10))

sm = 8.0
for idd in df2[key_id2].unique():
    ind = df2[key_id2]==idd
    dfc = df2[ind]
    ndd = dfc.shape[0]
    cent_x2[idd] = np.nanmean(dfc[key_x])
    cent_y2[idd] = np.nanmean(dfc[key_y])
    print(f'idd = {idd:2d}: n_eq = {dfc.shape[0]}')
    
    bx.scatter(scl*dfc[key_x], scl*dfc[key_y], marker='.', c=idd*np.ones((ndd)), 
            s=sm*dfc.magnitude, cmap=cm.tab20b, vmin=0, vmax=n_clu2-1)

bx.set_title('kmeans clustering level 2')
for jj in range(n_clu2):
    bx.text(scl*cent_x2[jj], scl*cent_y2[jj], f'{kdd}_{jj}')

# Put some cultutal stuff on all plots
for ak in [ax, bx]:
    ak.set_xlabel('x [km]')
    ak.set_ylabel('y [km]')
    trc_11N.plot(ax=ak, color='c', linewidth=1.0, label='SCEC trace')
    bld_11N.plot(ax=ak, color='b', linewidth=1.0, label='SCEC blind')
    salton_11N.plot(ax=ak, color='y', linewidth=1.0, label='Shoreline')
    
    if plot_cult:
        bhe_11N.boundary.plot(ax=ak, color='tab:orange', linewidth=1.5, label='BHE')
        ctr_11N.boundary.plot(ax=ak, color='tab:pink', linewidth=1.5, label='CTR')
        esm_11N.boundary.plot(ax=ak, color='tab:red', linewidth=1.5, label='ESM')
        phoenix_11N.boundary.plot(ax=ak, color='tab:olive', linewidth=1.5, label='Phoenix')
        mlist = ['.','.','o','d','s','v','^']
        for idd in df_wells.index:
            xw, yw = df_wells.loc[idd, key_x], df_wells.loc[idd, key_y]
            lab =  df_wells.loc[idd, 'label']
            ak.scatter(scl*xw, scl*yw, marker=mlist[idd], color='m', edgecolor='k', label=lab) 

    ak.axis('scaled')
    ak.legend()
    ak.set_xlabel('x [km] (11N)')
    ak.set_ylabel('y [km] (11N)')

# Zoom level 1 plot
ax.set_xlim(xtnt[0], xtnt[1])
ax.set_ylim(xtnt[2], xtnt[3])

# Zoom level 2 plot
bx.set_xlim(xtnt2[0], xtnt2[1])
bx.set_ylim(xtnt2[2], xtnt2[3])

figa.tight_layout(pad=1)
figb.tight_layout(pad=1)
figa.savefig(png + f'Level1_Clusters_Subset.png')
figb.savefig(png + f'Level2_Clusters_n{n_clu2}.png')

###############################################
#
#  Two point correlation analysis starts here
#
###############################################

#--------------------------------------------
#   All data
#--------------------------------------------

density = True
if run_corr:

    df0 = pd.concat(ssgf_list)
    key_dum = 'dum_id'
    df0[key_dum] = 0
    clu_dum = [0]
    fig_p0 = quake.power_correl(df0, clu_dum, dr0, key_id=key_dum, 
                                rmax=rmax0, zmin=zmin, zmax=zmax, 
                                density=density, verbose=verbose)

    #---------------------------------------------
    # Level 1 correlations
    #---------------------------------------------

    fig_p1 = quake.power_correl(df, clu_list, dr, key_id=key_id, 
                                rmax=rmax, zmin=zmin, zmax=zmax, 
                                density=density, verbose=verbose)

    #---------------------------------------------
    #  Level 3 correlations
    #---------------------------------------------

    fig_p2 = quake.power_correl(df2, clu_list2, dr2, key_id=key_id2, 
                                rmax=rmax2, zmin=zmin, zmax=zmax, 
                                density=density, verbose=verbose)

    # Save plots
    fig_p0.savefig(png + f'TwoPoint_Corr_AllData_zmax{zmax:.0f}.png')
    fig_p1.savefig(png + f'TwoPoint_Corr_SSGF_zmax{zmax:.0f}.png')
    fig_p2.savefig(png + f'TwoPoint_Corr_HR_zmax{zmax:.0f}.png')

    print(f'rmax ={rmax}')
    print(f'rmax2={rmax2}')

plt.show(block=block)

