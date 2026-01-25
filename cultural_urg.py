# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 09:49:27 2025

@author: KEHOK
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import geopandas as gpd
# import pickle
import pyproj

#--------------------------
# EPGS codes
#--------------------------

epsg = {}
epsg['WGS'] =  4326
epsg['32N'] = 23032
epsg['RGF'] =  2154

#------------------------------------
# PLot cultural
#------------------------------------

def plot_cultural(map_elements, **kwargs):
    """PLot map with cultural data, ready for geophsyical add ons"""

    zoom = kwargs.get('zoom', 1)
    crs = kwargs.get('crs', 'WGS')
    scl = kwargs.get('scl', 1.0)
    
    if crs == 'RGF':
        if zoom == 10:
            pass
        elif zoom==4:
            x1, x2 = 1050, 1066
            y1, y2 = 6870, 6886            
        elif zoom==3:
            x1, x2 = 1045, 1080
            y1, y2 = 6850, 6890
        elif zoom==2:
            x1, x2 = 1020, 1120
            y1, y2 = 6820, 6940
            # zoom_lab = 'zoom2'    
        elif zoom==1:
            x1, x2 =  800, 1200
            y1, y2 = 6650, 7100
            # zoom_lab = 'zoom1'    
        else:
            x1, x2 =  0, 3400
            y1, y2 = 5500, 9600
            # zoom_lab = 'zoom0'    

    countries = map_elements['countries']
    cities = map_elements['cities']
    lics = map_elements['lics']
    holes = map_elements['holes']
    faults = map_elements['faults']
    rivers = map_elements['rivers']

    key_x, key_y = 'x_RGF', 'y_RGF'

    # cmap = 'viridis'
    
    fig, ax = plt.subplots(1, figsize=(16,14))
    
    ax.axis('scaled')
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)

    countries.boundary.plot(ax=ax, color='royalblue', linewidth=1.10, label='borders')
    
    kols = ['orangered', 'springgreen', 'deepskyblue']
    if zoom<3:  
        for kk, key in enumerate(['normal', 'strike', 'thrust']):
            faults[key].plot(ax=ax, color=kols[kk], linewidth=0.90, label=f'IGME {key}')
    
    if zoom>0:
        for jj, lic in enumerate(lics):
            if jj == 0: lab = 'LdF licences'
            else: lab = ''
            lic.plot(ax=ax, color='gold', linewidth=1.5, label=lab)    
    
    if zoom>0:
        cities['color'] = 'red'
        ax.scatter(scl*cities[key_x], scl*cities[key_y], 
                   c=cities['color'], marker='o', label='Cities and plants')
        for idd in cities.index:
            xc, yc = scl*cities.loc[idd, key_x], scl*cities.loc[idd, key_y]
            txt = cities.loc[idd, 'name']
            ax.text(xc, yc, f'{txt}')
            
        rivers.plot(ax=ax, color='silver', linewidth=rivers['Strahler'], 
                    zorder=0, label='Rivers')
    
    if zoom >1:
        ax.scatter(scl*holes[key_x], scl*holes[key_y], 
                   c=holes['color'], marker='o', label='Boreholes')
        for idd in holes.index:
            xc, yc = scl*holes.loc[idd, key_x], scl*holes.loc[idd, key_y]
            txt = holes.loc[idd, 'name']
            ax.text(xc, yc, f'{txt}')

    ax.axis('scaled')
    
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)
    
    if crs == 'RGF':
        ax.set_xlabel('Easting (RGF) [km]')
        ax.set_ylabel('Northing (RGF) [km]')

    return fig, ax

#-------------------------------------
#   Read some shp files for plotting 
#-------------------------------------

def read_cultural(**kwargs):
    """Read a lot of stuff for plotting on a map.
    
    All inputs are WGS84 (lon, lat).
    """
    
    crs = kwargs.get('crs', 'WGS')
    scl = kwargs.get('scl', 1.0)
    
    # Transformers
    # wgs_to_32N = pyproj.Transformer.from_crs(epsg['WGS'], epsg['32N'], always_xy=True)
    # wgs_to_RGF = pyproj.Transformer.from_crs(epsg['WGS'], epsg['RGF'], always_xy=True)

    # Diretories
    shp = '../Data/shp/'
    excel_reg = '../Data/excel/'

    # Faults
    file_faults = 'IGME_5000_faults/Faults.shp'
    faults_wgs = gpd.read_file(shp + file_faults)
    
    faults, ind_flt = {}, {} 
    ind_flt['normal']  = faults_wgs['Portr_Line']=='fault'
    ind_flt['strike'] = (faults_wgs['Portr_Line']=='strike/slip fault') | \
                 (faults_wgs['Portr_Line']=='strike/slip fault dextral') | \
                 (faults_wgs['Portr_Line']=='strike/slip fault sinistral')
    ind_flt['trans'] = faults_wgs['Portr_Line']=='transform fault'
    ind_flt['thrust'] = faults_wgs['Portr_Line']=='thrust'
    
    for key in ind_flt.keys():
        faults[key] = faults_wgs[ind_flt[key]].to_crs(epsg[crs])
        faults[key] = faults[key].geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))
        
    # Licenses
    licdir = 'Licences/'
    perg_names = [
        'PERG_Les_Poteries.shp',
        'PERL_Les_Colombages.shp',
        'PERL_Les_Poteries_Minerals.shp',
        'PERL_PERG_Les_Sources.shp',
        'PERL_Plaine_du_Rhin_(Fonroche).shp'
        ]
    
    pergs_RGF = [None for p in perg_names]
    pergs_wgs = [None for p in perg_names]
    for jj, fname in enumerate(perg_names):
        pergs_RGF[jj] = gpd.read_file(shp + licdir + fname)
        pergs_wgs[jj] = pergs_RGF[jj].to_crs(epsg['WGS'])
        # scale
        pergs_RGF[jj] = pergs_RGF[jj].geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))
        
    
    # Map
    file_country = 'World_countries/World_Countries__Generalized_.shp'
    country_all = gpd.read_file(shp + file_country)
    
    countries_of_interest = [
        'Austria',
        'Belgium',
        'France',
        'Germany',
        'Italy',
        'Liechtenstein',
        'Luxembourg',
        'Netherlands',
        'Spain',    
        'Switzerland',
        'United Kingdom'
        ]
    
    cc_list = []
    for cc in country_all['COUNTRY'].unique():    
        if cc in countries_of_interest:
            wrk = country_all[country_all['COUNTRY'] == cc]
            cc_list.append(wrk)
        
    # countries = gpd.GeoDataFrame(pd.concat(cc_list, ignore_index=True))
    countries = country_all
    if crs != 'WGS':
        countries = countries.to_crs(epsg[crs])
        countries = countries.geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))
    
    with pd.ExcelFile(excel_reg + 'URG_Cities_and_Plants.xlsx') as fid:
        ch = pd.read_excel(fid) 
        
    # Split dataframe
    ib = ch[ch['type']=='Borehole'].index[0]
    cities = ch[ch.index<ib].copy()
    holes  = ch[ch.index>=ib].copy()
    
    # License boundaries
    if crs == 'WGS':
        lics = pergs_wgs
        cities['x'] = cities['lon']
        cities['y'] = cities['lat']
        holes['x'] = holes['lon']
        holes['y'] = holes['lat']
    else:
        lics = pergs_RGF
        cities['x'] = scl*cities['x_RGF']
        cities['y'] = scl*cities['y_RGF']
        holes['x'] = scl*holes['y_RGF']
        holes['y'] = scl*holes['x_RGF']
        
    # Rivers
    file_rivers = 'Rivers_Europe/lyr_f493e110_e667_4d65_9071_9837bad66522.shp'
    rivers_all = gpd.read_file(shp + file_rivers)
    rivers = rivers_all[rivers_all['MAJ_NAME']=='Rhine']
    rivers = rivers.to_crs(epsg[crs])
    # rivers = rivers.geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))
    rivers.geometry = rivers.geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))
    
    # Return a dict with map elements
    map_elements = {
        'countries': countries,
        'cities': cities,
        'lics': lics,
        'holes': holes,
        'faults': faults,
        'rivers': rivers
        }
        
    # return countries, cities, lics, holes, faults
    return map_elements
