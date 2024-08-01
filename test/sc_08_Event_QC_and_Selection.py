# -*- coding: utf-8 -*-
"""
Created on Fri May 03 09:15:26 2024

@author: KEHOK
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import geopandas as gpd
import pickle

import earthquake.scec as scec

#----------------------------
#  Folder names
#----------------------------

block = True

pkl = '../data/pkl/'
excel = '../data/excel/'
excel_bb = '../data/excel_bb/' # Lots of files for beachball analysis
shp = '../data/shp/'
png = 'png_focal/'

#---------------------------------
#  Read input data
#---------------------------------

fname = 'SCEC_Hauksson_1981_2022_WGS_Relocated_inBasin.xlsx'
fname_s = 'SCEC_Stations_inBasin.xlsx'

# Events in basin
with pd.ExcelFile(excel+fname) as fid:
    df_in = pd.read_excel(fid)

# Stations in basin
with pd.ExcelFile(excel+fname_s) as fid:
    df_s = pd.read_excel(fid)

# File with Gregoires list of events
with open(pkl + 'phases_downloaded_30jul2024_(Gregoire).pkl', 'rb') as fid:
    df_list, fail_list = pickle.load(fid)

# Entire world
lon1, lon2, lat1, lat2 = -180., 180., -90., 90.

# with open(pkl+'Events_in_AOI.pkl', 'rb') as ff:
#     df2, df2_s = pickle.load(ff)
    
# Adjust epicenters to relocated positions:
for eq in df_list:
    
    eid_object = eq.loc[0,'eid'] # This is a obspy.core.event.resourceid.ResourceIdentifier
    eid = int(eid_object.id)
    #ind = np.where(df2['eid'] == eid)[0][0]
    ind = df_in['eid'] == eid
    
    try:
        eid_found = df_in[ind]['eid'].values[0]
        lon, lat = df_in[ind]['lon'].values[0] ,df_in[ind]['lat'].values[0] 
        print(eid_found, eid, lon, lat)
    
        eq['lon0_reloc'] = lon
        eq['lat0_reloc'] = lat
        
    except:
        print('not found', eid)
        eq['lon0_reloc'] = eq['lon0'] 
        eq['lat0_reloc'] = eq['lat0'] 
        
# PLot position errors
fig = plt.figure()
for jj, eq in enumerate(df_list):

    if jj == 0: lab1, lab2 = 'Relocated', 'Initial'
    else: lab1, lab2 = '', ''
    
    plt.plot(eq.loc[0]['lon0_reloc'], eq.loc[0]['lat0_reloc'] , 'ro', label=lab1)
    plt.plot(eq.loc[0]['lon0'], eq.loc[0]['lat0'] , 'kx', label=lab2)

plt.legend()
plt.xlabel('Lon [deg]')
plt.ylabel('Lat [deg]')
plt.title('Event file vs reloc (Hauksson, 2022)')
fig.savefig(png + 'Event_file_vs_Reloc.png')

#--------------------------------
#  Compute average velocity 
#--------------------------------

for jj, dfw in enumerate(df_list):
    
    if dfw is not None:
             
        dfw['r'], dfw['savg'], dfw['vavg'] = 0., 0., 0.
        
        z = dfw['depth0']
        d = dfw['dist']
        t = dfw['time']
        
        r = np.sqrt(z**2+d**2)
        v = r/t
        s = 1/v
    
        dfw['r'] = r    
        dfw['savg'] = s    
        dfw['vavg'] = v

#----------------
# Read some cultural data
#----------------

scl = 1e-0

epsg_wgs =  4326
epsg_11N = 26711

# Salton Sea shoreline
salton_11N = gpd.read_file(shp + 'SaltonSea.shp')

# Licenses
bhe_11N = gpd.read_file(shp + 'BHE.shp')
ctr_11N = gpd.read_file(shp + 'CTR.shp')
esm_11N = gpd.read_file(shp + 'ESM.shp')

bhe_11N.set_crs(epsg=epsg_11N, inplace=True)
ctr_11N.set_crs(epsg=epsg_11N, inplace=True)
esm_11N.set_crs(epsg=epsg_11N, inplace=True)

# Transform to UTM11N or vice versa
salton_wgs = salton_11N.to_crs({'init': f'epsg:{epsg_wgs}'})
bhe_wgs = bhe_11N.to_crs({'init': f'epsg:{epsg_wgs}'})
ctr_wgs = ctr_11N.to_crs({'init': f'epsg:{epsg_wgs}'})
esm_wgs = esm_11N.to_crs({'init': f'epsg:{epsg_wgs}'})

#----------------------------------
#   Get the data insided AOI
#----------------------------------

# Select data
mag_min = 0.0
npick_min = 3
depth_max = 10.0 # max hypocenter depth [km]
vavg_max = 10.0 # km/s
dist_max = 500.0 # km. Max epicenter distance

P_in_AOI_list, S_in_AOI_list = scec.scec_data_selection(df_list, 
                                    lon1, lon2, lat1, lat2, depth_max, vavg_max,
                                    mag_min, dist_max, npick_min=npick_min, 
                                    split_ps=True, verbose=1)

# Remove picks with unknown polarity
P_for_bb_list = []
for jj, eq in enumerate(P_in_AOI_list):
    
    ind = ~eq['polarity'].isnull()
    eq = eq[ind]
    eq = eq.reset_index()
    
    if eq.shape[0] >= 0:
        P_for_bb_list.append(eq)
        # print(jj, eq.shape[0])

S_for_bb_list = []
for jj, eq in enumerate(S_in_AOI_list):
    
    ind = ~eq['polarity'].isnull()
    eq = eq[ind]
    eq = eq.reset_index()
    
    if eq.shape[0] >= 0:
        S_for_bb_list.append(eq)
        # print(jj, eq.shape[0])

# Dump to Excel files
fname_list = []
select_list = range(len(P_for_bb_list))
for jj in select_list:
    
    eq = P_for_bb_list[jj]
    eid = eq['eid'][0]
    fname = f'BB_AnalysisData_eid_{eid}.xlsx'
    fname_list.append(fname)
    
    with pd.ExcelWriter(excel_bb + fname) as ff:
        eq.to_excel(ff, sheet_name='P-data', index=False)

#-------------------
# PLot rays
#-------------------

# Merge data ito one df, split in P and S waves
P_for_bb = pd.concat(P_for_bb_list, ignore_index=True)

pad = 0.2
fig, axs = plt.subplots(1,2, figsize=(18,8))

area = 'Gregoires_event_list'

ax = axs.ravel()[0]
dfw = P_for_bb
nray_P = dfw.shape[0]

for idd in dfw.index:
    x = np.array([dfw.loc[idd,'lon0'], dfw.loc[idd,'lon']])
    y = np.array([dfw.loc[idd,'lat0'], dfw.loc[idd,'lat']])
    ax.plot(x,y, 'b-', zorder=0)
        
ax = axs.ravel()[1]

for idd in dfw.index:
    d = np.array([0, dfw.loc[idd,'dist']])
    z = -np.array([0, dfw.loc[idd,'depth0']])
    ax.plot(d,z, 'b-', zorder=0)

ax.set_xlabel('Offset [km]')
ax.set_ylabel('Depth [km]')
        
for ax in axs.ravel()[0:1]:

    salton_wgs.plot(ax=ax, color='y', linewidth=1.0, label='Shoreline')
    bhe_wgs.boundary.plot(ax=ax, color='tab:orange', linewidth=1.5, label='BHE')
    ctr_wgs.boundary.plot(ax=ax, color='tab:pink', linewidth=1.5, label='CTR')
    esm_wgs.boundary.plot(ax=ax, color='tab:red', linewidth=1.5, label='ESM')
    
    ax.set_xlabel('Lon [deg]')
    ax.set_ylabel('Lat [deg]')
    ax.axis('scaled')

    # Epicenters and seismic stations
    for ii, eq in enumerate(P_for_bb_list):

        if ii==0:
            lab_eq, lab_ss = 'Epicenters', 'Seismic stations'
        else:
            lab_eq, lab_ss = '', ''
            
        
        ax.plot(eq['lon0'], eq['lat0'], 'g.', label=lab_eq)
        ax.plot(eq['lon'], eq['lat'], 'rv', label=lab_ss)
        
    ax.legend()    
    
ax = axs.ravel()[0]
ax.set_title(f'P-rays (n={nray_P})')
ax = axs.ravel()[1]
ax.set_title(f'P-rays (n={nray_P})')

fig.suptitle(f'CEC 1981 - 2022 relocated (Hauksson), Gregoires list')
fig.tight_layout(pad=1)
fig.savefig(png + f'SCEC_for_Beachballs_Gregoires_List.png')

plt.show(block=block)
