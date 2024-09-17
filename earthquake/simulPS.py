# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 08:18:01 2024

Functions making input files for simulPS14 earthquake tomo 

@author: KEHOK
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Some parameters hardcoded in the fortran code of simulPS14
MAXEV = 600 # Max number of events (quakes)
MAXOBS = 100 # Max number of traveltime picks per event
MAXSTA = 200 # Max number fo stations
MXPARI = 5000 # Max number of unknowns to invert for
MAXPAR = 30000 # Max number of pars in model spesification (vp and vp/vs)
MAXNX = 30 # Max number of nodes in x
MAXNY = 30 # Max number of nodes in y
MAXNZ = 30 # Max number of nodes in z

#-------------------------------------
# Make job control file
#-------------------------------------

def write_CNTL(cntl, cdir='./', fname='CNTL'):
    """Write the CNTL input file for simulPS14.
    
    See Appendix A, page 11 in 
    Evans, Eberhardt-Phillips and Thurber (1994):
    USER'S MANUAL FOR SIMULPS12 FOR IMAGING VP AND vp lvs :
    A DERIVATIVE OF THE "THURBER" TOMOGRAPHIC INVERSION SIMUL3
    FOR LOCAL EARTHQUAKES AND EXPLOSIONS
    
    Parameters
    ----------
    cntl: dict. Control parameters set by the user
    cdir: str. Output directory (default is cdir='./')
    fname: str. Output filename (default is fname='STNS')
    
    Returns
    -------
    ierr: int. ierr=0 if success

    Programmed: KetilH, 20. March 2024
    """
    
    out_file = cdir+fname
    print(f'write_CNTL: out_file = {out_file}')

    try:
        
        with open(out_file, 'w') as fid:
            # fmt0 = '{:3d} {:4d} {:4d} {:6.2f} {:3d} {:3d} {:3d}\n'
            # fmt1 = '{:3d} {:4.1f} {:6.3f} {:6.3f} {:5.1f} {:5.2f} {:5.2f} {:5.2f}\n'
            fmt1 = '{:d} {:d} {:d} {:f} {:d} {:d} {:d}\n'
            fmt2 = '{:d} {:f} {:f} {:f} {:f} {:f} {:f} {:f}\n'
            fmt3 = '{:d} {:f} {:f} {:d} {:f} {:f} {:f} {:f}\n'
            fmt4 = '{:d} {:d} {:d} {:f} {:d} {:f} {:d}\n'
            fmt5 = '{:f} {:f} {:f} {:f} {:f}\n'
            fmt6 = '{:d} {:d} {:f} {:f}\n'
            fmt7 = '{:f} {:f} {:d} {:d}\n'
            fmt8 = '{:d} {:d} {:d}\n'
            fmt9 = '{:d} {:d} {:d}\n'
            fmt = fmt1 + fmt2 + fmt3 + fmt4 + fmt5 + fmt6 + fmt7 + fmt8 + fmt9
            fid.write(fmt.format(*cntl.values()))  
        
        ierr = 0
    
    except:
        print('write_CNTL: Something went wrong')
        ierr = 1
    
    return ierr

#-------------------------------------
# Make the station data file
#   x-axis (lon) points west
#   y-axis (lat) points north
#   z-axis points down
#-------------------------------------

def crazy_format(lon, lat):
    """ Convert decimal lon and lat to the insane format used in simulPS14 """
    
    lon_degree = int(np.abs(lon))
    lat_degree = int(np.abs(lat))
    
    lon_minute = 60*(np.abs(lon)-lon_degree)
    lat_minute = 60*(np.abs(lat)-lat_degree)
    
    if lat < 0.0:
        hemi = 'S'
    else:
        hemi = 'N'
        
    if lon < 0.0:
        wrld = 'W'
    else:
        wrld = 'E'
        
    lat_str = '{:2d}{:1s}{:5.2f}'.format(lat_degree, hemi, lat_minute)
    lon_str = '{:3d}{:1s}{:5.2f}'.format(lon_degree, wrld, lon_minute)

    return lon_str, lat_str

def write_STNS(df_list, cdir='./', fname='STNS',label='label', 
               lon0=None, lat0=None, rota=0.0, nzco=0):
    """Write the STNS input file for simulPS14.
    
    See Appendix A, page 14 in 
    Evans, Eberhardt-Phillips and Thurber (1994):
    USER'S MANUAL FOR SIMULPS12 FOR IMAGING VP AND vp lvs :
    A DERIVATIVE OF THE "THURBER" TOMOGRAPHIC INVERSION SIMUL3
    FOR LOCAL EARTHQUAKES AND EXPLOSIONS
    
    Note on the (right-handed) coordinate system used in simulPS14:
        x (longitude) is positive towards west
        y (latitude) is positive towards north
        z is positive down
    The lan/lat format is integer degree and decimmal minute, e.g.
        (lon, lat) = (-115.5, 30.25) = (115W30.0, 30N15.0)
    
    Parameters
    ----------
    df_list: pd.DataFrame or list of dataframes with earthquake data from all stations
    cdir: str. Output directory (default is cdir='./')
    fname: str. Output filename (default is fname='STNS')
    label: str. Column in df_P with unique station identifiers (network+station)
    lon0, lat0: float. Origin of the velocity grid (why is it in this file?).
        The user manual reccomends setting (lon0, lat0) in the center of the model.
        Default is lon0=mean(lon), lat0=mean(lat)
    rota: float. Rotation angel of the velocity grid.
    
    Returns
    -------
    ierr: int. ierr=0 if success
    syno: dict. Mapping of network+station to Rnnn (4 character ids)
    
    Programmed: KetilH, 20. March 2024
    """
    
    # List as input?
    if isinstance(df_list, list): 
        df_P = pd.concat(df_list, ignore_index=True)
        cin = 'list'
    else:
        df_P = df_list
        cin = 'pd.DataFrame'

    out_file = cdir + fname
    
    # Center of grid given?
    if lon0 is None: lon0 = df_P['lon'].mean()
    if lat0 is None: lat0 = df_P['lat'].mean()
    
    st_unique = df_P[label].unique()
    nsts = len(st_unique)
    syno = {'R' + str(jj+1).zfill(3): st for jj, st in enumerate(st_unique)}

    print(f'write_STNS: out_file = {out_file}')
    print(f' o lon0, lat0 {lon0}, {lat0}')
    print(f' o Input data is {cin}')
    print(f' o nsts {nsts}')

    try:
    
        if(nsts<=25): 
            
            key_list = ['head1', 'head2'] + list(syno.keys())
            stns = {key: None for key in key_list}
            
            lon0_degree = -int(lon0) # x-axis is positive towards west
            lat0_degree =  int(lat0)
            
            lon0_minute = 60*(np.abs(lon0)-int(np.abs(lon0)))
            lat0_minute = 60*(np.abs(lat0)-int(np.abs(lat0)))
            
            # Header lines are free format
            head1 = [lat0_degree, lat0_minute, lon0_degree, lon0_minute, rota, nzco]
            stns['head1'] = '  {:f} {:f} {:f} {:f} {:f} {:d}\n'.format(*head1)
            stns['head2'] = '  {:d}\n'.format(nsts)
            
            lon_arr, lat_arr = [], [] # Don't need these
            
            for key, st in zip(syno.keys(), syno.values()):
                # print(f'{st:7s} -> {key:4s}')
            
                ind = df_P[label] == st
                idd_list = df_P.index[ind]
            
                lon = df_P.loc[idd_list[0], 'lon']
                lat = df_P.loc[idd_list[0], 'lat']
                elev = -int(df_P.loc[idd_list[0], 'elev'])
                
                vp_corr = 0.0
                vpvs_corr = 0.0
                flag = 1
                
                lon_arr.append(lon)
                lat_arr.append(lat)
                
                lon_str, lat_str = crazy_format(lon, lat)
                
                vals = [key, lat_str, lon_str, elev, vp_corr, vpvs_corr, flag]
            
                stns[key] = '  {:4s}{:8s}{:>10s}{:5d}{:5.2f}{:5.2f}{:3d}\n'.format(*vals)
            
            # Dump stuff to file (insane formatting)
            with open(out_file, 'w') as fid:
                for key in stns.keys():
                    # print(f'write to file: {key}')
                    fid.write(stns[key])
                    
            ierr = 0
                
        else:
            print(f'Too many stations: nsts={nsts}')
            ierr = 2
            
    except:
        print('write_STNS: Something went wrong')
        ierr = 1

    return syno, ierr

#-----------------------------------
#  Write event file
#-----------------------------------


def write_EQKS(df_list, syn_fw,  cdir='./', fname='EQKS', label='label'):
    """Write the EQKS input file for simulPS14.
    
    See Appendix A, page 16 in 
    Evans, Eberhardt-Phillips and Thurber (1994):
    USER'S MANUAL FOR SIMULPS12 FOR IMAGING VP AND vp lvs :
    A DERIVATIVE OF THE "THURBER" TOMOGRAPHIC INVERSION SIMUL3
    FOR LOCAL EARTHQUAKES AND EXPLOSIONS
    
    Note on the (right-handed) coordinate system used in simulPS14:
        x (longitude) is positive towards west
        y (latitude) is positive towards north
        z is positive down
    The lan/lat format is integer degree and decimmal minute, e.g.
        (lon, lat) = (-115.5, 30.25) = (115W30.0, 30N15.0)
    
    Parameters
    ----------
    df_list: pd.DataFrame or list of dataframes with earthquake data from all stations
    syn_fw: dict. Mapping of network+station to Rnnn (4 character ids)
    cdir: str. Output directory (default is cdir='./')
    fname: str. Output filename (default is fname='STNS')
    label: str. Column in df_P with unique station identifiers (network+station)
    
    Returns
    -------
    ierr: int. ierr=0 if success
    
    Programmed: KetilH, 21. March 2024
    """
    
    out_file = cdir + fname
        
    # List as input?
    if isinstance(df_list, list): 
        df_P = pd.concat(df_list)
        cin_P = 'list'
    else:
        df_P = df_list
        cin_P = 'pd.DataFrame'
           
    # Unique event IDs for P and S picks
    ev_P_unique = list(df_P['eid'].unique())
    
    print(f'write_EQKS: out_file = {out_file}')
    print(f' o Input data is {cin_P}')
    print(f' o No of events with picks: {len(ev_P_unique)}')
    
    if len(ev_P_unique) > 600:
        print('Number of events too high (max is 600) !!!')
    
    # Inverse dictionary  for station map
    syn_bw = {val:key for key,val in zip(syn_fw.keys(), syn_fw.values())}
    
    # Wtrite tp and ts picks to file
    try:
    
        ev_head = []
        ev_list = []
        
        # Make formatted strings for output
        for jj, eid in enumerate(ev_P_unique):       
            # print(eid)
            
            # Find all P-wave picks for current event
            indP = (df_P['eid'] == eid) & (df_P['phase']=='P')
            wrkP = df_P[indP]    
            st_P = list(wrkP[label])
        
            # Find all P-wave picks for current event
            indS = (df_P['eid'] == eid) & (df_P['phase']=='S')
            wrkS = df_P[indS]    
            st_S = list(wrkS[label])
        
            # First line with common data
            idd0 = wrkP.index[0]
            time0 = wrkP.loc[idd0, 'time0']
            yy = str(time0.year % 100).zfill(2)
            mm = str(time0.month).zfill(2)
            dd = str(time0.day).zfill(2)
            yymmdd = yy+mm+dd
            
            jjstr = str(jj).zfill(4)
            zero = 0
        
            # Hypocenter and magnitude
            lon0 = wrkP.loc[idd0, 'lon0']
            lat0 = wrkP.loc[idd0, 'lat0']
            lon_str, lat_str = crazy_format(lon0, lat0)    
            depth0 = wrkP.loc[idd0, 'depth0']
            mag = wrkP.loc[idd0, 'mag']
            
            # Header line
            vals = [yymmdd, jjstr, zero, lat_str, lon_str, depth0, mag]
            ss = '{:7s}{:4s}{:6.2f}{:>9s}{:>10s}{:7.2f}{:7.2f}\n'.format(*vals)
            ev_head.append(ss)
            
            # Traveltime data
            kkk = 0
            pk_list = []
            for idd in wrkP.index:
                lab = wrkP.loc[idd,label]
                rec = syn_bw[lab]
                
                # P-pick
                ph  = wrkP.loc[idd, 'phase']
                tp = wrkP.loc[idd, 'time']
                rem = '{:<2s}{:1s}'.format(ph, str(0))
                ss = '{:<5s}{:3s}{:6.2f}'.format(rec, rem, tp)
                kkk += 1
                pk_list.append(ss)
                # print(idd, lab, kkk, ss)
           
                # S-pick?
                if lab in st_S:
                    jdd = wrkS.index[wrkS['label'] == lab][0]         
                    ph  = wrkS.loc[jdd, 'phase']
                    ts = wrkS.loc[jdd, 'time']
                    rem = '{:<2s}{:1s}'.format(ph+'P', str(0))
                    ss = '{:<5s}{:3s}{:6.2f}'.format(rec, rem, ts-tp)
                    kkk += 1
                    pk_list.append(ss)
                    # print(jdd, lab, kkk, ss)
          
            ev_list.append(pk_list)
        
        # Dump data to file:
        with open(out_file, 'w') as fid:
            
            nevent = len(ev_list)
            lbat = 6
            for jj in range(nevent):
                
                picks = ev_list[jj]
                npick = len(picks)
                nbat = npick // lbat
                nrst = npick % lbat
                
                fmt_b = lbat*'{:14s}' + '\n'
                fmt_r = nrst*'{:14s}' + '\n'
                
                fid.write(ev_head[jj])
                for ii in range(nbat):
                    fid.write(fmt_b.format(*picks[ii*lbat:(ii+1)*lbat]))
                
                fid.write(fmt_r.format(*picks[nbat*lbat:]))
                fid.write('{:1d}\n'.format(0))

        ierr = 0

    except:
        print('write_EQKS: Something went wrong')
        ierr = 1

    return ierr     
    
# OK HIT 21/3-2024    

def write_MOD(mod, cdir='./', fname='MOD'):
    """Write the MOD input file for simulPS14.
    
    See Appendix A, page 15 in 
    Evans, Eberhardt-Phillips and Thurber (1994):
    USER'S MANUAL FOR SIMULPS12 FOR IMAGING VP AND vp lvs :
    A DERIVATIVE OF THE "THURBER" TOMOGRAPHIC INVERSION SIMUL3
    FOR LOCAL EARTHQUAKES AND EXPLOSIONS
    
    Parameters
    ----------
    mod: dict. Definition of model grid and initial vp and vp/vs models
    cdir: str. Output directory (default is cdir='./')
    fname: str. Output filename (default is fname='STNS')
    
    Returns
    -------
    ierr: int. ierr=0 if success

    Programmed: KetilH, 20. March 2024
    """


    out_file = cdir + fname
    print(f'write_MOD: out_file = {out_file}')
    
    try:
    
        with open(out_file, 'w') as fid:
            
            vals = [mod['bld'], mod['nx'], mod['ny'], mod['nz']]
            fid.write('{:3.1f} {:d} {:d} {:d}\n'.format(*vals))
            
            fmt_x = mod['nx']*'{:.1f} ' + '\n'
            fmt_y = mod['ny']*'{:.1f} ' + '\n'
            fmt_z = mod['nz']*'{:.1f} ' + '\n'
            fid.write(fmt_x.format(*mod['xn']))
            fid.write(fmt_y.format(*mod['yn']))
            fid.write(fmt_z.format(*mod['zn']))
            
            fmt_f = '{:d} {:d} {:d}\n'
            for jxf, jyf, jzf in zip(mod['ixf'], mod['iyf'], mod['izf']):
                fid.write(fmt_f.format(jxf, jyf, jzf))
            
            
            # End of node mask marker
            fid.write('{:2d}{:2d}{:2d}\n'.format(0,0,0))
            
            fmt_v = mod['nx']*'{:.2f} ' + '\n'
            for jz in range(mod['nz']):
                for jy in range(mod['ny']):
                    fid.write(fmt_v.format(*mod['vp'][jz,jy,:]))
        
            for jz in range(mod['nz']):
                for jy in range(mod['ny']):
                    fid.write(fmt_v.format(*mod['ps_rat'][jz,jy,:]))
        
        ierr = 0

    except:
        print('write_MOD: Something went wrong')
        ierr = 1

    return ierr

def write_RAYTRAC(raytrac, cdir='./', fname='RAYTRAC'):
    """Write the MOD input file for simulPS14.
    
    See Appendix A, page ??? in 
    Evans, Eberhardt-Phillips and Thurber (1994):
    USER'S MANUAL FOR SIMULPS12 FOR IMAGING VP AND vp lvs :
    A DERIVATIVE OF THE "THURBER" TOMOGRAPHIC INVERSION SIMUL3
    FOR LOCAL EARTHQUAKES AND EXPLOSIONS
    
    Parameters
    ----------
    raytrac: dict. Ray tracing pars
    cdir: str. Output directory (default is cdir='./')
    fname: str. Output filename (default is fname='STNS')
    
    Returns
    -------
    ierr: int. ierr=0 if success

    Programmed: KetilH, 23. March 2024
    """
    out_file = cdir + fname
    print(f'write_RAYTRAC: out_file = {out_file}')    
    
    try:

        with open(out_file, 'w') as fid:
            
            vals = [raytrac['iheter'], raytrac['epsob'], raytrac['epsca'], 
                    raytrac['ides'], raytrac['ampr'], raytrac['iterrai']] 
            fid.write('{:d} {:.3f} {:.1f} {:d} {:.1f} {:d}\n'.format(*vals))
            
            vals = [raytrac['dxrt'], raytrac['dyrt'], raytrac['dzrt']]
            fid.write('{:.1f} {:.1f} {:.1f}\n'.format(*vals))
            ierr = 0

    except:
        print('write_RAYTRAC: Something went wrong')
        ierr = 1

    return ierr

#--------------------------------------------------
#   Read output model file from simulPS14
#--------------------------------------------------       

def read_MOD(cdir='./', fname='velomod.out', mod_ini=None, **kwargs):
    """Read output tomography model from simulPS14. 
    Same format as the initial model file.
    
    See page ??? in 
    Evans, Eberhardt-Phillips and Thurber (1994):
    USER'S MANUAL FOR SIMULPS12 FOR IMAGING VP AND vp lvs :
    A DERIVATIVE OF THE "THURBER" TOMOGRAPHIC INVERSION SIMUL3
    FOR LOCAL EARTHQUAKES AND EXPLOSIONS

    Parameters
    ----------    raytrac: dict. Ray tracing pars
    cdir: str. Output directory (default is cdir='./')
    fname: str. Output filename (default is fname='velmodo.out')
    mod_ini: dict. Initial model (default is mod_ini=None). Same format as output.
                   Used to cmput a mask for where model has been updated
    
    kwargs
    ------
    agc: dict. Parameters for geodetic conversion from local cartesian to lon and lat
    iusep: int. Read vp model (default is iusep=1), as given in CNTL file
    iusep: int. Read vp model (default is iuses=0), as given in CNTL file
    verbose: int. rint shit? (default is verbose=0)
    
    Returns
    -------
    mod: dict with fields
        nx, ny, nz: int
        xn, yn, zn: Node coordinates in multiples of bld
        bld: 1.0 or 0.1 (see maual )
        vp: array of floats. P-wave velocity
        vs: array of floats. S-wave velocity
        ps_rat: array of floats. vp/vs-ratio
        
    Programmed: KetilH, 8. April 2024
    """
    
    agc = kwargs.get('agc', None)
    iusep = kwargs.get('iusep', 1) # Read vp model?
    iuses = kwargs.get('iuses', 0) # Read vp/vs-model?
    verbose = kwargs.get('verbose', 0)
    
    with open(cdir+fname, 'r') as fid:
        line_list = fid.readlines()    
        for jj, line in enumerate(line_list):
            line_list[jj] = line.strip()

    mod = {}
    
    kkk = 0
    wlist = line_list[kkk].split()
    mod['bld'] = float(wlist[0])
    mod['nx'] = int(wlist[1])
    mod['ny'] = int(wlist[2]) 
    mod['nz'] = int(wlist[3])
    
    try:
        it = int(wlist[4])
    except:
        it = None
    
    # Long lines are broken in velomod.out
    kkk += 1
    wlist = line_list[kkk].split()
    if len(wlist) < mod['nx']:
        kkk += 1
        wlist = wlist + line_list[kkk].split()
    mod['xn'] = np.array(wlist).astype(float)
    
    kkk += 1
    wlist = line_list[kkk].split()
    if len(wlist) < mod['ny']:
        kkk += 1
        wlist = wlist + line_list[kkk].split()
    mod['yn'] = np.array(wlist).astype(float)
    
    kkk += 1
    wlist = line_list[kkk].split()
    if len(wlist) < mod['nz']:
        kkk += 1
        wlist = wlist + line_list[kkk].split()
    mod['zn'] = np.array(wlist).astype(float)

    # Lon and lat
    if isinstance(agc, dict):
        lon0, lat0 = agc['lon0'], agc['lat0']
        km2lon = 1/agc['km_per_lon'] 
        km2lat = 1/agc['km_per_lat'] 
        mod['lon'] = lon0 - km2lon*mod['xn'] # positive x-dir is west
        mod['lat'] = lat0 + km2lat*mod['yn'] # positive y-dir is north
    
    jj = kkk + 1
    while line_list[jj].split() != ['0', '0', '0']:
        jj += 1

    if verbose>0:
        print('simulPS.read_MOD:')
        print(' o nx, ny, nz = {}, {}, {}'.format(mod['nx'], mod['ny'], mod['nz']))
        print(f' o Start of 3D model at line {jj+1}')

    # Read the vp grid
    if iusep == 1:
    
        if (verbose>0): print(' o Read vp-model')
        mod['vp'] = np.zeros((mod['nz'], mod['ny'], mod['nx']), dtype=float)    
        for jz in range(mod['nz']):
            for jy in range(mod['ny']):
                jj += 1
                mod['vp'][jz, jy, :] = np.array(line_list[jj].split()).astype(float)
        
    else:
        if (verbose>0): print(' o No vp-model')
        
    # Read the vp/vs-grid?
    if iuses == 1:
        
        if (verbose>0): print(' o Read vp/vs-model')
        mod['ps_rat'] = np.zeros((mod['nz'], mod['ny'], mod['nx']), dtype=float)    
        for jz in range(mod['nz']):
            for jy in range(mod['ny']):
                jj += 1
                mod['ps_rat'][jz, jy, :] = np.array(line_list[jj].split()).astype(float)
    
    else:
        if (verbose>0): print(' o No vp/vs-model')

    # Compute a mask for where model has been updated
    mod['mask'] = np.ones((mod['nz'], mod['ny'], mod['nx']), dtype=float)
    if isinstance(mod_ini, dict): 
        if 'vp' in mod.keys():
            tiny = 1e-3
            dif = np.abs(mod['vp'] - mod_ini['vp'])
            mod['mask'][dif<tiny] = np.nan

    return mod

#------------------------------------------------------
# Read the iteration summary file
#------------------------------------------------------

def read_itersum(cdir='./', fname='itersum.txt', **kwargs):
    """Read iteration summary from simulPS14; 
    
    See page ??? in 
    Evans, Eberhardt-Phillips and Thurber (1994):
    USER'S MANUAL FOR SIMULPS12 FOR IMAGING VP AND vp lvs :
    A DERIVATIVE OF THE "THURBER" TOMOGRAPHIC INVERSION SIMUL3
    FOR LOCAL EARTHQUAKES AND EXPLOSIONS

    Parameters
    ----------    raytrac: dict. Ray tracing pars
    cdir: str. Output directory (default is cdir='./')
    fname: str. Output filename (default is fname='velmodo.out')
    
    kwargs
    ------
    verbose: int. rint shit? (default is verbose=0)
    
    Returns
    -------
    tomo_mod: dict with fields
        nx, ny, nz: int
        xn, yn, zn: Node coordinates in multiples of bld
        bld: 1.0 or 0.1 (see maual )
        vp: array of floats. P-wave velocity
        vs: array of floats. S-wave velocity
        ps_rat: array of floats. vp/vs-ratio
        
    Programmed: KetilH, 8. April 2024
    """

    # fid = open(cdir+fname, 'r')
    # line_list = fid.readlines()    
    # fid.close()

    with open(cdir+fname, 'r') as fid:
        line_list = fid.readlines()    
        for jj, line in enumerate(line_list):
            line_list[jj] = line.strip()
    
    # File starts with blank line?
    jj = 0
    if line_list[jj] == '': jj+=1
    
    # GEt some pars
    keys1 = line_list[jj].split()
    jj += 1
    vals1 = line_list[jj].split()
    
    jj+=2
    keys2 = line_list[jj].split()
    jj += 1
    vals2 = line_list[jj].split()
    
    keys = keys1 + keys2
    vals = vals1 + vals2
    
    kh = {key: float(val) for (key, val) in zip(keys, vals)}
    
    keys_int = ['neqs', 'nsht', 'nbls', 'wtsht', 'nitloc'] 
    for key in keys_int:
        kh[key] = int(kh[key])
    
    # Get the rms errors
    kh['rms'] = []
    kh['rms_w'] = []
    for kk in range(jj+1, len(line_list)):
        slist = line_list[kk].split()
        if len(slist)>2:
            # if slist[0] == 'f-test': break
            if (slist[0] == 'unweighted') & (slist[1] == 'rms='):
                kh['rms'].append(float(slist[2][:-1]))
                kh['rms_w'].append(float(slist[5][:]))
            
    return kh  
    
#--------------------------------------
#  PLot model
#--------------------------------------

def plot_MOD(mod, **kwargs):
    """ PLot simulps14 mode.
    
    Parameters
    ----------     
    mod: dict. Model from simulps14
    
    kwargs
    ------
    key_x: str. key for x-coord (default is key_x='lon')
    key_y: str. key for y-coord (default is key_x='lat')
    verbose: int. Print shit? (default is verbose=0)

    Returns
    -------
    fig: figure oobject

    Programmed: KetilH, 8. April 2024    
    """
    
    key_x = kwargs.get('key_x', 'lon')
    key_y = kwargs.get('key_y', 'lat')
    key_mod = kwargs.get('key_mod', 'vp')
    title = kwargs.get('title', 'simulPS14 model')
    verbose = kwargs.get('verbose', 0)
    mask = kwargs.get('mask', False)
    curves = kwargs.get('curves', None)
    shading = kwargs.get('shading', 'auto')
    
    if key_x == 'lon':
        unt = '[deg]'
    else:
        unt = 'km'
    
    nplane = len(mod['zn'])-2
    if nplane < 6:
        nrow = 1
    elif nplane < 9:
        nrow = 2
    else:
        nrow = 3
    ncol = nplane//nrow
    
    fig, axs = plt.subplots(nrow, ncol, figsize=(14,3*nrow))
    for kk in range(nrow*ncol):
        ax = axs.ravel()[kk]
        jj = kk+1
        x, y = mod[key_x], mod[key_y]
        
        if mask:
            rmask = mod['mask'][jj,:,:]
        else:
            rmask = np.ones_like(mod[key_mod][jj,:,:])
        
        grd = rmask*mod[key_mod][jj,:,:]
        im = ax.pcolormesh(x, y, grd, shading=shading)
        cb = ax.figure.colorbar(im, ax=ax)
        ax.axis('scaled')
        ax.set_xlabel(f'{key_x} {unt}')
        ax.set_ylabel(f'{key_y} {unt}')
        z = mod['zn'][jj]
        ax.set_title(f'{key_mod}: z = {z}')

        # Cultural
        kols = ['tab:orange', 'tab:pink', 'tab:red', 'tab:olive'] 
        if curves != None:
            for ik, crv in enumerate(curves):
                crv.plot(ax=ax, color=kols[ik], linewidth=1.0)
        
        ax.set_xlim(x[-2], x[1])
        ax.set_ylim(y[1], y[-2])

    fig.suptitle(f'{title}')
    fig.tight_layout(pad=1.0)

    return fig
