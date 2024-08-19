# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 22:56:50 2023

@author: kehok
"""

import numpy as np
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
import khio.grid as khio
from gravmag.common import gridder
from gravmag.common import MapData
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

scl = 1e-3 # m to km for x and y

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
    # ind_to_plot = [0,1,2,3,4,5,6,7,11,12,13,14,20,21] 
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

#----------------------------------------------
#  Read Stimac temperature image and T grids
#----------------------------------------------

S = img.imread(pics + 'Stimac_HeatSource.png')
R = S[::-1,:,:]
x1, y1 = 614200, 3657870
xlen, ylen = 30100, 27200
x2, y2 = x1+xlen, y1+ylen
xtnt = [x1, x1+xlen, y1, y1+ylen]
heat = {'img': R, 'x1' :x1, 'x2':x1+xlen, 'y1': y1, 'y2': y1+ylen, 'label': 'Stimac'}

# Read irap grid
wrk = khio.read_irap_grid(irap + 'Tg_UTM11N.grd')
Tg = MapData(wrk[0], wrk[1], [0], wrk[2])
Tg.label = ['Tg']

with open(pkl + 'LF_Isotherms.pkl', 'rb') as fid:
    isot = pickle.load(fid)

#-------------------------------------
# Read EQ input
#-------------------------------------

fname = 'Earthquake_Data_with_CluID_utm11N.xlsx'
with pd.ExcelFile(excel+fname) as fid:
    df_all = pd.read_excel(fid)
    n_clu = df_all['clu_id'].max() + 1

print(f'### n_clu={n_clu}')

# Interpolate Sediment thickness at EQ locations
xi = gm.gx.ravel()
yi = gm.gy.ravel()
zi = gm.grd[1].ravel() # sediment thickness
df_all['sed_thickness']= gridder(xi, yi, zi, df_all['x'], df_all['y'])

#--------------------------
# QC plot input data
#--------------------------

key_x, key_y = 'x', 'y'
df = df_all.copy()

sm = 1.0 # EQ magnitude scaling for scatter plot
scld = 1e-3 # depth scaing, m to km

# Select data for plotting
z1, z2 = 0, 15
ind = (scld*df.depth > z1) & (scld*df.depth < z2)
df0 = df[ind].sort_values(by='magnitude')

xtnt = scl*np.array([eq.x[0], eq.x[-1], eq.y[0], eq.y[-1]])

fig, ax = plt.subplots(1, figsize=(12,8))
kkk = 1
im = ax.imshow(scl*gm.grd[kkk], origin='lower', extent=xtnt, cmap='gray')
sc = ax.scatter(scl*df0[key_x], scl*df0[key_y], c=df0.magnitude, marker='o', s=sm*df0.magnitude)
trc_11N.plot(ax=ax, color='c', linewidth=1.0, label='SCEC trace')
bld_11N.plot(ax=ax, color='b', linewidth=1.0, label='SCEC blind')
salton_11N.plot(ax=ax, color='y', linewidth=1.0, label='Shoreline')
bsz_11N.plot(ax=ax, color='tab:purple', linewidth=1.0, label='Brawley SZ')

if plot_cult:
    bhe_11N.boundary.plot(ax=ax, color='tab:orange', linewidth=1.5, label='BHE')
    ctr_11N.boundary.plot(ax=ax, color='tab:pink', linewidth=1.5, label='CTR')
    esm_11N.boundary.plot(ax=ax, color='tab:red', linewidth=1.5, label='ESM')
    phoenix_11N.boundary.plot(ax=ax, color='tab:olive', linewidth=1.5, label='Phoenix')

mlist = ['.','.','o','d','s','v','^']
if plot_cult:
    for idd in df_wells.index:
        xw, yw = df_wells.loc[idd, key_x], df_wells.loc[idd, key_y]
        lab =  df_wells.loc[idd, 'label']
        ax.scatter(scl*xw, scl*yw, marker=mlist[idd], color='m', edgecolor='k', label=lab) 

ax.set_xlim(xtnt[0], xtnt[1])
ax.set_ylim(xtnt[2], xtnt[3])
ax.legend()
ce = ax.figure.colorbar(sc, ax=ax, label='Magnitude [-]')
cb = ax.figure.colorbar(im, ax=ax, label=f'{gm.label[kkk]} [km]')

ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_title('Earthquake magnitudes on sedimentary thickness map')

fig.tight_layout(pad=1)
fig.savefig(png + 'Sed_Thickness_and_Quakes.png')

#---------------------------------------
# clustering magnitude vs depth rel tbq
#---------------------------------------

df3 = df_all
df3['depth_rel'] = df3['depth'] - df3['sed_thickness']
key = 'depth'

# Clustering?
n_clu_d = 1 
ind = np.isfinite(df3[key])
df3 = df3[ind]
if n_clu_d>1:
    centroid, clu_id, inertia = k_means(df3[[key, 'magnitude']], n_clu_d, n_init='auto')
else:
    clu_id = np.zeros_like(ind, dtype=int)
    
fig, ax = plt.subplots(1, figsize=(8,6))
for idd in np.unique(clu_id):
    ind = clu_id == idd
    mwrk, dwrk = df3[ind]['magnitude'], df3[ind][key]
    ax.scatter(mwrk, scl*dwrk, label=f'cluster {idd}', marker='o', s=10*mwrk)
    
ax.set_xlabel('Magnitude [-]')
if key == 'depth_rel':
    ax.set_ylabel('Source depth rel TB [km]')
    ax.set_title('Magnitude vs depth rel TopBasement')
    ax.set_ylim(-5, 11)
else:
    ax.set_ylabel('Source depth[km]')
    ax.set_title('Magnitude vs source depth')    
    ax.set_ylim(0, 16)
ax.invert_yaxis()
fig.tight_layout(pad=1.)
fig.savefig(png + f'EQ_Magnitude_vs_{key}_{n_clu_d}.png')

#------------------------------------------------
# b-values for entire catalogue
#------------------------------------------------

mc, mc2 = 1.9, 6.21
magnitude = np.array(df_all['magnitude'])

# b-value by line reg
dm = 0.1
b, a = quake.reg_b_value(magnitude, mc, mc2, dm)

# b-value by Aki (1965) method
delm = 0.0
b_aki, std_aki = quake.aki_b_value(magnitude, mc, mc2, delm)
a_aki = a - 0.10 # by trial and error

print (f'Regression: b={b:.3f},  a={a:.3f}')
print (f'Aki MLE:    b={b_aki:.3f} +/- {std_aki:.3f}')

#----------------------------------
# Plot Gutenberg-Richter trend
#----------------------------------

suptitle = 'Gutenberg-Richter law: Salton Sea SCEC 1980-2019 catalogue'
label, label2 = 'Linear regression', 'Aki (1965) MLE' 

fig = quake.plot_gutenberg_richter(magnitude, mc, mc2, dm, b=b, a=a, label=label,
                                   b2=b_aki, a2=a_aki, label2=label2, suptitle=suptitle)

fig.savefig(png + 'SaltonSea_Gutenberg_Richter_All.png')

#------------------------------------
# b-values clustered
#------------------------------------

# Reset:
df = df_all.copy()
# mc, mc2  = 1.75, 4.25
mc, mc2  = 2.0, 4.0
ind = (df.magnitude>=mc) & (df.magnitude<=mc2)
df = df[ind]
df['b'] = 0.0

# Spatial clustering using kMeans: CLustering in sc_02...py
cent_x = np.zeros(n_clu)
cent_y = np.zeros(n_clu)

figa, ax = plt.subplots(1, figsize=(10,10))
figk, kxs = plt.subplots(4, n_clu//4, figsize=(24,10))
style = 'log'
nbin = 20

df_bval = pd.DataFrame(columns=['cluster', 'nn', 'a', 'b', 'b_mle', 'std_mle'], index=range(0,n_clu))

for idd in df['clu_id'].unique():
    ind = df['clu_id']==idd
    dfc = df[ind]
    ndd = dfc.shape[0]
    cent_x[idd] = np.nanmean(dfc[key_x])
    cent_y[idd] = np.nanmean(dfc[key_y])
    print(f'idd = {idd:2d}: n_eq = {dfc.shape[0]}')
    
    magnitude = np.array(dfc['magnitude'])
    ax.scatter(scl*dfc['x'], scl*dfc['y'], marker='.', c=idd*np.ones((ndd)), 
               s=sm*magnitude, cmap=cm.tab20b, vmin=0, vmax=n_clu-1)
    
    # b-value by line reg
    dm = 0.1
    b, a = quake.reg_b_value(magnitude, mc, mc2, dm)

    # b-value by Aki (1965) method
    delm = 0.0
    b_mle, std_mle = quake.aki_b_value(magnitude, mc, mc2, delm)
    a_mle = a + 0.1

    # For plotting
    [Nh, be] = np.histogram(dfc.magnitude, bins=nbin, range=(mc,mc2))
    m = (be[0:-1] + be[1:])/2
    Ncum = np.cumsum(Nh[::-1])[::-1]
    marr = np.linspace(mc, mc2,101)
    Narr = 10**(a-b*marr)

    # print (f'Regression: idd={idd}, b={b:.3f},  a={a:.3f}')
    # print (f'Aki MLE:               b={b_mle:.3f} +/- {std_mle:.3f}')

    nn = dfc.shape[0]
    df_bval.loc[idd, ['cluster', 'nn', 'a', 'b', 'b_mle', 'std_mle']] = [idd, nn, a, b, b_mle, std_mle]
    df.loc[ind, ['b', 'b_mle', 'std_mle']]  = [b, b_mle, std_mle]
    # Assign b-value to all EQ in the cluster (for output)
    jnd = df_all['clu_id']==idd
    df_all.loc[jnd, ['b', 'b_mle', 'std_mle']]  = [b, b_mle, std_mle]

    # Mean cut-off by cluster
    depth_mean = np.mean(dfc['depth'])
    depth_std  = np.std(dfc['depth'])
    depth_max  = np.max(dfc['depth'])
    df.loc[ind, 'depth_mean'] = depth_mean
    df.loc[ind, 'depth_std']  = depth_std
    df.loc[ind, 'depth_max']  = depth_max

    kx = kxs.ravel()[idd]
    kx.plot(marr, Narr, 'k-')
    kx.plot(m, Ncum,'r-o')    
    kx.set_yscale('log')
    kx.set_title(f'cluster {idd}: n={dfc.shape[0]}, b_mle={b_mle:.2f}')
    kx.set_xlim(mc, mc2)
    kx.set_ylim(1*10**0, 2*10**3)

ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_title('kmeans clustering')
for jj in range(n_clu):
    ax.text(scl*cent_x[jj], scl*cent_y[jj], f'{jj}')

#-----------------------------------------
# Dump Excel file with clu_id and b-value
#-----------------------------------------

fname_out = 'Earthquake_Data_with_CluID_and_b_value_utm11N.xlsx'
with pd.ExcelWriter(excel+fname_out) as fid:
    df_all.to_excel(fid, index=False)

fname_b = 'Cluster_b_value.xlsx'
with pd.ExcelWriter(excel+fname_b) as fid:
    df_bval.to_excel(fid, index=False)

#--------------------
# More plots
#--------------------

figc, cx = plt.subplots(1, figsize=(12,10))
sc = cx.scatter(scl*df['x'], scl*df['y'], marker='.', c=df['b'], 
                s=sm*df.magnitude, vmin=1.0, vmax=1.3)
cb = cx.figure.colorbar(sc, ax=cx)
cx.set_xlabel('x [km]')
cx.set_ylabel('y [km]')
cx.set_title('LinReg b-value for each cluster')

figd, dx = plt.subplots(1, figsize=(12,10))
cut_off = scl*(df['depth_mean'] + df['depth_std'])
sc = dx.scatter(scl*df['x'], scl*df['y'], marker='.', c=cut_off, s=sm*df.magnitude) 
cb = dx.figure.colorbar(sc, ax=dx)
dx.set_xlabel('x [km]')
dx.set_ylabel('y [km]')
dx.set_title('Cut-off depth per cluster [km]')

fige, ex = plt.subplots(1, figsize=(12,10))
sc = ex.scatter(scl*df['x'], scl*df['y'], marker='.', c=df['b_mle'], 
                s=sm*df.magnitude, vmin=0.9, vmax=1.2)
cb = ex.figure.colorbar(sc, ax=ex)
ex.set_xlabel('x [km]')
ex.set_ylabel('y [km]')
ex.set_title('Aki (1965) b-value for each cluster')

figf, fx = plt.subplots(1, figsize=(12,10))
sc = fx.scatter(scl*df['x'], scl*df['y'], marker='.', c=df['std_mle'], 
                s=sm*df.magnitude, vmin=0, vmax=0.10)
cb = fx.figure.colorbar(sc, ax=fx)
fx.set_xlabel('x [km]')
fx.set_ylabel('y [km]')
fx.set_title('Aki (1965) b-value STD for each cluster')

# PLot with Hauksson AOI
smh = 0.5
lon = np.array([-115.9, -115.4, -115.4, -115.9, -115.9])
lat = np.array([32.8, 32.8, 33.42, 33.42, 32.8])
xh, yh = gc.wgs_to_utm(lat, lon, 11, 'N')
figb, bx = plt.subplots(1, figsize=(9,10))
sc = bx.scatter(scl*df['x'], scl*df['y'], marker='.', c=df['b_mle'], 
                s=smh*df['magnitude'], vmin=0.9, vmax=1.2)
cb = bx.figure.colorbar(sc, ax=bx, label='b-value')
bx.set_xlabel('x [km]')
bx.set_ylabel('y [km]')
bx.set_title('Aki (1965) b-value for each cluster')

for ak in [ax, bx, cx, dx, ex, fx]:
# for ak in [bx]:
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
    ak.set_xlim(xtnt[0], xtnt[1])
    ak.set_ylim(xtnt[2], xtnt[3])
    ak.legend()
    ak.set_xlabel('x [km] (11N)')
    ak.set_ylabel('y [km] (11N)')

# Adjust to rectangle of the Hauksson figure
bx.set_xlim(scl*xh[0], scl*xh[1])
bx.set_ylim(scl*yh[0], scl*yh[2])

figh, hx = plt.subplots(1)
sc = hx.scatter(df_bval['nn'], df_bval['std_mle'])
hx.set_xlabel('n  [-]')
hx.set_ylabel('std [-]')
hx.set_ylim(0.0, 0.3)
hx.set_title('Aki (1965) b-value uncertainty vs no of samples in cluster')

figa.tight_layout(pad=1)
figk.tight_layout(pad=1)
figc.tight_layout(pad=1)
figd.tight_layout(pad=1)
fige.tight_layout(pad=1)
figf.tight_layout(pad=1)
figb.tight_layout(pad=1)
figh.tight_layout(pad=1)

figa.savefig(png + 'EQ_Clusters.png')
figk.savefig(png + f'EQ_Gutenberg_Richter_per_Cluster_{style}.png')
figc.savefig(png + 'EQ_b_value_per_Cluster.png')
figd.savefig(png + 'EQ_cut_off_depth_per_Cluster.png')
fige.savefig(png + 'EQ_Aki_b_value_per_Cluster.png')
figb.savefig(png + 'EQ_Aki_b_value_per_Cluster_with_Hauksson_AOI.png')
figf.savefig(png + 'EQ_Aki_b_uncertainty_per_Cluster.png')
figh.savefig(png + 'EQ_Aki_b_uncertainty_vs_no_of_samples.png')

#------------------------------------
#   PLot of Tg and Stimac
#------------------------------------

# b-values
key_b = 'b_mle'
c, s =  df[key_b], 4*df['magnitude']
# vmin, vmax, ver = 1.0, 1.14, 'b_values'
vmin, vmax, ver = 0.98, 1.12, 'b_values'
xp, yp = scl*df['x'], scl*df['y']

eq.grd_sm = [None for gg in eq.grd]
for jj in range(3):
    eq.grd_sm[jj] = smooth2d(eq.grd[jj], 1, method='median')
    ind = np.isnan(eq.grd[jj])
    eq.grd_sm[jj][ind] = np.nan

fig, axs = plt.subplots(1,2, figsize=(21,10))
 
xtnt_St = scl*np.array([heat['x1'], heat['x2'], heat['y1'], heat['y2']])
ax = axs.ravel()[0]
im = ax.imshow(heat['img'], origin='lower', extent=xtnt_St, cmap=cm.Oranges)
sc = ax.scatter(xp, yp, marker='.', c=c, s=s, vmin=vmin, vmax=vmax)
cb = ax.figure.colorbar(sc, ax=ax, label=f'{ver}')
# cc = ax.figure.colorbar(im, ax=ax, label='Tg [oC/km]')
ax.set_title('Stimac et al. (2017)')

xtnt_Tg = scl*np.array([Tg.x[0], Tg.x[-1], Tg.y[0], Tg.y[-1]])
ax = axs.ravel()[1]
grd = 10*Tg.grd[0] # oC/100m to oC/km
im = ax.imshow(grd, origin='lower', extent=xtnt_Tg, cmap=cm.Oranges)
sc = ax.scatter(xp, yp, marker='.', c=c, s=s, vmin=vmin, vmax=vmax)
# cb = ax.figure.colorbar(sc, ax=ax, label='b-value [-]')
cc = ax.figure.colorbar(im, ax=ax, label='Tg [oC/km]')
ax.set_title('Shallow Tg (Hulen et al., 2002) and faults (Kaspereit et al., 2016)')

for ak in axs.ravel():
    
    rhyolites.plot(ax=ak, color='m', linewidth=0.95, label='Rhyolites')    
    for jj, fault in enumerate(fault_list[0:5]):
        if jj==0: lab = 'Kaspereit et al. (2016)'
        else: lab= ''
        fault.plot(ax=ak, color='tab:purple', linewidth=1.5, label=lab)
            
    for jj, fault in enumerate(fault_list[5:]):
        if jj==0: lab = 'Brothers et al. (2009)'
        else: lab= ''
        fault.plot(ax=ak, color='tab:green', linewidth=1.5, label=lab)
                    
    salton_11N.plot(ax=ak, color='y', linewidth=1.0, label='Shoreline')
    trc_11N.plot(ax=ak, color='c', linewidth=1.0, label='SCEC trace')
    bld_11N.plot(ax=ak, color='b', linewidth=1.0, label='SCEC blind')
 
    if plot_cult:
        bhe_11N.boundary.plot(ax=ak, color='tab:orange', linewidth=1.5, label='BHE')
        ctr_11N.boundary.plot(ax=ak, color='tab:pink', linewidth=1.5, label='CTR')
        esm_11N.boundary.plot(ax=ak, color='tab:red', linewidth=1.5, label='ESM')

        mlist = ['.','.','o','D','s','v','^']
        for idd in df_wells.index:
            xw, yw = df_wells.loc[idd, key_x], df_wells.loc[idd, key_y]
            lab =  df_wells.loc[idd, 'label']
            ak.scatter(scl*xw, scl*yw, marker=mlist[idd], color='m', edgecolor='k', label=lab) 

    ak.axis('scaled')
    ak.set_xlim(615, 643)
    ak.set_ylim(3648, 3683)
    ak.legend(loc='lower left')
    ak.set_xlabel('x [km] (11N)')
    ak.set_ylabel('y [km] (11N)')

fig.tight_layout(pad=1.0)
fig.savefig(png + f'Tg_and_{ver}.png')

#------------------------------------
#   PLot of isotherms
#------------------------------------

f2m = 0.3048 # feet to meters

fig, axs = plt.subplots(1,2, figsize=(21,10))
 
xtnt_isot = scl*np.array([isot.x[0], isot.x[-1], isot.y[0], isot.y[-1]])
ax = axs.ravel()[0]
grd, lab, unit = isot.grd[0].copy(), isot.label_F[0], '[oF]'
ind = grd<-30000
grd[ind] = np.nan
im = ax.imshow(f2m*grd, origin='lower', extent=xtnt_isot, cmap=cm.YlOrRd)
sc = ax.scatter(xp, yp, marker='.', c=c, s=s, vmin=vmin, vmax=vmax)
cb = ax.figure.colorbar(im, ax=ax, label=f'500F isotherm [m]')
ax.set_title('500F (260C) isotherm')

ax = axs.ravel()[1]
grd, lab, unit = isot.grd[0]-isot.grd[1], 'diff 600-500', '[oF]'
grd[ind] = np.nan
# grd = isot.grd[0] - grd # JUST TESTING
im = ax.imshow(f2m*grd, origin='lower', extent=xtnt_Tg, cmap=cm.YlOrRd)
sc = ax.scatter(xp, yp, marker='.', c=c, s=s, vmin=vmin, vmax=vmax)
cc = ax.figure.colorbar(im, ax=ax, label='Isochore [m]')
ax.set_title('500F-600F isochore')

for ak in axs.ravel():
    
    rhyolites.plot(ax=ak, color='m', linewidth=0.95, label='Rhyolites')    
    for jj, fault in enumerate(fault_list[0:5]):
        if jj==0: lab = 'Kaspereit et al. (2016)'
        else: lab= ''
        fault.plot(ax=ak, color='tab:purple', linewidth=1.5, label=lab)
            
    for jj, fault in enumerate(fault_list[5:]):
        if jj==0: lab = 'Brothers et al. (2009)'
        else: lab= ''
        fault.plot(ax=ak, color='tab:green', linewidth=1.5, label=lab)
                    
    salton_11N.plot(ax=ak, color='y', linewidth=1.0, label='Shoreline')
    trc_11N.plot(ax=ak, color='c', linewidth=1.0, label='SCEC trace')
    bld_11N.plot(ax=ak, color='b', linewidth=1.0, label='SCEC blind')

    if plot_cult:
        bhe_11N.boundary.plot(ax=ak, color='tab:orange', linewidth=1.5, label='BHE')
        ctr_11N.boundary.plot(ax=ak, color='tab:pink', linewidth=1.5, label='CTR')
        esm_11N.boundary.plot(ax=ak, color='tab:red', linewidth=1.5, label='ESM')

        mlist = ['.','.','o','D','s','v','^']
        for idd in df_wells.index:
            xw, yw = df_wells.loc[idd, key_x], df_wells.loc[idd, key_y]
            lab =  df_wells.loc[idd, 'label']
            ak.scatter(scl*xw, scl*yw, marker=mlist[idd], color='m', edgecolor='k', label=lab) 

    ak.axis('scaled')
    ak.set_xlim(615, 643)
    ak.set_ylim(3648, 3683)
    ak.legend(loc='lower left')
    ak.set_xlabel('x [km] (11N)')
    ak.set_ylabel('y [km] (11N)')

fig.tight_layout(pad=1.0)
fig.savefig(png + f'Isotherm_500F_and_{ver}.png')

plt.show(block=block)


    
    