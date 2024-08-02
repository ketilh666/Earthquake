# -*- coding: utf-8 -*-
"""
Created on Wed Mmay 02 12:42:23 2024

@author: KEHOK
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import geopandas as gpd
import pickle
import os

from pystp.client import STPClient

#----------------------------
#  Folders
#----------------------------

block = True

shp = '../data/shp/'
pkl = '../data/pkl/'
excel = '../data/excel/'

if not os.path.isdir(png): os.mkdir(png)
png = 'png/'

#-----------------------------------------
# Read some cultural data for plotting
#-----------------------------------------

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

# Transform from UTM11N to WGS84
salton_wgs = salton_11N.to_crs({'init': f'epsg:{epsg_wgs}'})
bhe_wgs = bhe_11N.to_crs({'init': f'epsg:{epsg_wgs}'})
ctr_wgs = ctr_11N.to_crs({'init': f'epsg:{epsg_wgs}'})
esm_wgs = esm_11N.to_crs({'init': f'epsg:{epsg_wgs}'})

#----------------------------
# Read input data
#----------------------------

fname = 'SCEC_Hauksson_1981_2022_WGS_Relocated_inBasin.xlsx'
fname_s = 'SCEC_Stations_inBasin.xlsx'

# Read relocated hypocenters
with pd.ExcelFile(excel+fname) as fid:
    df_in = pd.read_excel(fid)

# Read locations of seismic stations
with pd.ExcelFile(excel+fname_s) as fid:
    df_s = pd.read_excel(fid)

#------------------------------------------------------------
# Select data by eid and download traveltime picks from SCEC
#------------------------------------------------------------

# Gregoires suggested list of events
evt_list = [
     9059316,
     9039664,
     9039606,
    10320761,
    11065613,
    15105052,
    15114473,
    15116089,
    15116121,
    15116321,
    11069757,
    15344513,
    15344585,
    15344465,
    15354321,
    15354537
        ]

# Output filename
tag = '30jul2024_(Gregoire)'
fname_download = f'phases_downloaded_{tag}.pkl'

nevt = len(evt_list)

# Need this shit because the server sometimes fails
restart = False
if restart:
    with open(pkl + fname_download, 'rb') as fid:
        df_list, fail_list = pickle.load(fid)
else:
    df_list = [None for evt in evt_list]

# Connect to server
stp = STPClient('stp3.gps.caltech.edu', 9999) 
stp.connect()

# Always reset this
fail_list = []

# Download what's missing
for kk in range(nevt):
    
    if df_list[kk] is None:
        
        try:
            
            evids = [evt_list[kk]]
            
            event = stp.get_events(evids=evids)
            phase, df = stp.get_phases(evids=evids, make_df=True)
    
            df_list[kk] = df[0].copy()
            
            print(f'Got {kk}, {evids}')

        except:
            
            fail_list.append(kk)
            print(f'Failed {kk}, {evids} !!!!!!!')
            
    else:
        pass
    
stp.disconnect()

# Dump te data we have got
with open(pkl + fname_download, 'wb') as fid:
    pickle.dump([df_list, fail_list], fid)

#-----------------------------------
# Make a QC plot of the data
#-----------------------------------

# Positions are not relocated
lon = [eq.loc[0, 'lon0']for eq in df_list]
lat = [eq.loc[0, 'lat0']for eq in df_list]
mag =  [eq.loc[0, 'mag'] for eq in df_list]
eid  = [eq.loc[0, 'eid']for eq in df_list]

fig, ax = plt.subplots(1, figsize=(8,6))
sc = ax.scatter(lon, lat, c=mag, marker='o', s=12, label=eid)

salton_wgs.plot(ax=ax, color='y', linewidth=1.0, label='Shoreline')
bhe_wgs.boundary.plot(ax=ax, color='tab:orange', linewidth=1.5, label='BHE')
ctr_wgs.boundary.plot(ax=ax, color='tab:pink', linewidth=1.5, label='CTR')
esm_wgs.boundary.plot(ax=ax, color='tab:red', linewidth=1.5, label='ESM')

ax.axis('scaled')

# Zoom on earthquakes in basin:
lon1, lon2 = -115.70, -115.48
lat1, lat2 =   33.08,   33.28
ax.set_xlim(lon1, lon2)
ax.set_ylim(lat1, lat2)

ax.set_xlabel('Lon [deg]')
ax.set_ylabel('Lat [deg]')
ax.set_title(f'Hypocenters (not reloc) of events downloaded from SCEC')
fig.tight_layout(pad=1)

plt.show(block=block)

    




