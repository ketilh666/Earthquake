# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 08:33:24 2024

Finctions for dealing with data from earthquake data from SCEC

@author: KEHOK
"""

import numpy as np
import pandas as pd

def read_SCEC_catalog(fname, **kwargs):
    """Read earthquake catalog file from SCEC
    
    Parameters
    ----------
    fname: string. full filename with path
    
    Returns
    -------
    df: pd.DataFrame with data
    
    Programmed: KetilH, 15. February 2024    
    """
    
    nhl = kwargs.get('nhl', 10) # No of header lines
    
    dfwrk = pd.read_csv(fname, skiprows=nhl-1, delim_whitespace=True)
    
    # Last line is junk
    dfwrk = dfwrk.drop(index=dfwrk.index[-1])
    
    # Make a new dataframe with numbers and strings 
    cols = ['lon', 'lat', 'depth', 'mag', 'q',
            'm', 'et', 'gt', 'evid', 'nph', 'ngrm',
            'year', 'month', 'day', 'hour', 'minute', 'date', 'time']
    df = pd.DataFrame(columns=cols)
    
    # Floats
    cols_num = ['lon', 'lat', 'depth', 'mag', 'evid', 'nph', 'ngrm']
    for col in cols_num:
        df[col] = dfwrk[col.upper()].astype(float)

    # Strings
    cols_str = ['q', 'm', 'et', 'gt']
    for col in cols_str:
        df[col] = dfwrk[col.upper()]
        
    # Date and time
    nl = df.shape[0]
    day, month, year = np.ones(nl), np.ones(nl), np.ones(nl)
    hour, minute, sec = np.ones(nl), np.ones(nl), np.ones(nl)
    key_d = dfwrk.columns[0]
    key_t = dfwrk.columns[1]
    for ii in df.index:
        wrkd = dfwrk.loc[ii, key_d].split('/')
        wrkt = dfwrk.loc[ii, key_t].split(':')
        year[ii] = int(wrkd[0])
        month[ii] = int(wrkd[1])
        day[ii] = int(wrkd[2])
        hour[ii] = int(wrkt[0])
        minute[ii] = int(wrkt[1])
        sec[ii] = float(wrkt[2])
    
    df['year'] = year
    df['month'] = month
    df['day'] = day
    df['hour'] = hour
    df['minute'] = minute
    df['sec'] = sec
    
    df['date'] = dfwrk[key_d]
    df['time'] = dfwrk[key_t]
    
    return df   
    
#-------------------------------------------------------
#  Functions for selecting events from SCEC event data
#-------------------------------------------------------

def remove_duplicates(df, key='terr', label='label', phase='phase'):
    """Remove trveltime duplicates when same event is picked multiple times on
    different geophone components
    
    Parameters
    ----------
    df: pd.DataFrame. Traveltime picks from one earthquake event
    
    Optionals
    ---------
    key: str. Column to use for selecting among multiple picks (default is key='terr')
    label: str. key for unique identifer for seismic station (default is label='label')
    phase: str. key for seismic phase, P or S (default is phase='phase')
    
    Returns
    -------
    df2: pd.DataFrame. Unique traveltime picks selected form df based on df[key].min()
    
    Programmed: KetilH, 18. March 2024
    """

    # Remove duplicates
    labels_unique = df[label].unique()
    phases_unique = df[phase].unique()
    idd_list = []
    
    for lab in labels_unique:      
        for ph in phases_unique:
            
            jnd = (df[phase]==ph) & (df[label]==lab)
            ww = df[jnd][key]
            
            if ww.shape[0]>0:
                # print(lab, ph)
                ind = np.argmin(ww)
                idd_list.append(ww.index[ind])
        
    df2 = df.loc[idd_list,:]
    return df2

def scec_data_selection(df_list, lon1, lon2, lat1, lat2, dmax, vmax=8.0,
                        mag_min=2, dist_max=50.0, npick_min=3, 
                        key='tarr', label='label', phase='phase', split_ps=False,
                        **kwargs):
    """ Select data by windowing and P&S splitting 
    
    Parameters
    ----------
    df_list: list of pd.DataFrame. Traveltime picks from earthquake events, subset of a catalog
    lon1, lon2: float. Min and max longitude of AOI 
    lat1, lat2: float. Min and max latiitude of AOI 
    dmax: float. Max hypocenter depth
    vmax: float. Max average velocity (get rid of bad data with unphysical velocity)
    mag_min: float. Minimum magnitude (default=2)
    dist_max: float. Max epicente distance
    
    Optionals
    ---------
    key: str. Column to use for selecting among multiple picks (default is key='terr')
    label: str. key for unique identifer for seismic station (default is label='label')
    phase: str. key for seismic phase, P or S (default is phase='phase')
    
    Returns
    -------
    df2: pd.DataFrame. Unique traveltime picks selected form df based on df[key].min()

    Programmed: KetilH, 18. March 2024
    """

    verbose = kwargs.get('verbose', 0)
    
    if verbose >0:
        print(f'lon1, lon2 = {lon1}deg, {lon2}deg')
        print(f'lat1, lat2 = {lat1}deg, {lat2}deg')
        print(f'dmax, dist_max = {dmax}km, {dist_max}km')
        print(f'vmax, mag_min = {vmax}km/s, {mag_min}')
        print(f'npick_min = {npick_min}')


    df_in_AOI_list = [] # Whole damn thing
    df_P_in_AOI_list = [] # List for gathering P-wave picks
    df_S_in_AOI_list = [] # List for gathering S-wave picks
    

    for jj, dfw in enumerate(df_list):
        
        if dfw is not None:
                    
            # Create a label to identify seismic nodes
            dfw['label'] = dfw['network'] + '_' + dfw['station']
            
            # Data in AOI
            ind = (dfw['lon'] >= lon1) & (dfw['lon'] <= lon2) & \
                  (dfw['lat'] >= lat1) & (dfw['lat'] <= lat2) & \
                  (dfw['depth0'] <= dmax)  & (dfw['mag'] >= mag_min) & \
                  (dfw['vavg'] <= vmax)  & (dfw['dist']<=dist_max) & \
                  (dfw['time'] > 0.0) 
                      
            wrk1= dfw[ind]
                  
            # SPlit P and S
            wrk1_P = wrk1[wrk1['phase']=='P']
            wrk1_S = wrk1[wrk1['phase']=='S']
                           
            wrk2   = remove_duplicates(wrk1)
            wrk2_P = remove_duplicates(wrk1_P)
            wrk2_S = remove_duplicates(wrk1_S)
    
            if wrk2_P.shape[0] >= npick_min:
                df_in_AOI_list.append(wrk2)
                df_P_in_AOI_list.append(wrk2_P)
                df_S_in_AOI_list.append(wrk2_S)
                
    if split_ps:
        return df_P_in_AOI_list, df_S_in_AOI_list 
    else:   
        return df_in_AOI_list
    