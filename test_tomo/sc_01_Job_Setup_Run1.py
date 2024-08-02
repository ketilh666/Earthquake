# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 08:33:08 2024

@author: KEHOK
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import geopandas as gpd
import pickle

import earthquake.simulPS as sps
import earthquake.quake as quake

#------------------------------------
#  Job setup
#------------------------------------

krun = 1
simulps = f'run_{krun}/'
excel = '../data/excel_tomo/'
pkl = '../data/pkl_tomo/'

write_cntl_only = False
plot_qc = True

block = True

#---------------------------------
# Read 1D velocity model
#---------------------------------

#fname_vel ='Imperial_Valley_1D_Velocity_Model.xlsx'
fname_vel ='Well_Trend_1D_Velocity_Model_FINAL.xlsx'
with pd.ExcelFile(excel + fname_vel) as fid:
    vel1d_in = pd.read_excel(fid, sheet_name='Data') 
    
if 'ps_rat' not in vel1d_in.columns:
    vel1d_in['ps_rat'] = vel1d_in['vp']/vel1d_in['vs']

vp_avg = vel1d_in['vp'].mean()
vs_avg = vel1d_in['vs'].mean()
ps_avg = vp_avg/vs_avg

# Make an initial model for synt testing
vel1d = vel1d_in.copy()

# scl initial vp/vs model
ps_scl = 0.9
ps_scl = 1.0

#----------------------------------------------------------------------
#  Read inpt data
#  For dfferent purposes when setting up a tomo-job, it's advantageous 
#  Have the data both sorted by event and in a merged dataframe
#----------------------------------------------------------------------

# Coordinates of AOI rectangle in lon and lat
with open(pkl + 'Tomo_AOI.pkl', 'rb') as ff:
    lon1, lon2, lat1, lat2 = pickle.load(ff)

# Read earthquake time picks
with open(pkl + 'picks_in_AOI.pkl', 'rb') as fid:
    eq_in_AOI_list_in = pickle.load(fid)

eq_in_AOI_list = eq_in_AOI_list_in

#-----------------------------------------------
# Merge data into one df (not really needed)
#-----------------------------------------------

# Merge data into one df (not really needed)
eq_in_AOI = pd.concat(eq_in_AOI_list, ignore_index=True)

#-------------------------------------
# Make the CNTL file
# Key control parameters for iteration
#  o rmstop (soft temination)
#  o nitmax (hard termination)
# Regularization parameters varied:
#  o vpdamp
#  o vpvsdamp
#-------------------------------------

fname = 'CNTL'

cntl = {
    # Line 1
     'neqs': len(eq_in_AOI_list), # Number of earthquakes in data
     'nsht': 0, # Number of shots in data
     'nbls': 0, # Number of blasts
     'wtsht': 0.00, # weighting of shots rel earthquakes
     'kout': 3, # output-file control parameter
     'kout2': 0, # printout (unit16) control parameter
     'kout3': 0, # yet another output control parameter
     # Line 2
     'nitloc': 3, # Max number of iterations for hypocenter calculation
     'wtsp': 0.5, # For hypocenter solution, the weight of the ts -tp residual relative to tp residual
     'eigtol': 0.020, # Singular value decomposition (SVD) cutoff in hypocentral adjustments.
     'rmscut': 0.002, # value of RMS residual below which hypocentral adjustments are terminated
     'zmin': 0.0, # Minimum hypocenter depth (which is negative for events above sea level)
     'dxmax': 0.50, # Maximum horizontal hypocentral adjustment allowed in each iteration
     'rderr': 0.05, # Estimate of traveltime reading/picking error (often in the range 0.01-0.05 s).
     'ercof': 0.01, # for hypoinverse-like error calculations. Set 0.0<erco/<1.0 to include RMS residual in hypocenter error estimate
     # Line 3
     'nhitct': 7, # minimum DWS for a parameter to be included in the inversion (usually >5).
     'dvpmx': 0.40, # maximum vp adjustment allowed per iteration
     'dvpvsmx': 0.20, # maximum vp/vs adjustment allowed per iteration
     'idmp': 0, # 1 to recalculate the damping value for succeeding iterations, 0 to keep damping constant
     'vpdamp': 10.0, # NB!!! damping parameter used in velocity (vp) inversion
     'vpvsdamp': 30.0, # NB!!! damping parameter used in velocity (vp/vs) inversion
     'stadamp': 2.0, # damping parameter used in velocity (station delays) inversion
     'stepl': 0.50, # NB! step length (km) used for calculation of partial derivatives along the raypath
     # Line4: Here's most of the important pars !!!
     'ires': 2, # NB! controls computation of R and C (0/1/2/3)
     'i3d': 1, # NB! flag for using ray pseudo-bending (0/1/2/3/4)
     'nitmax': 5, # NB!!! maximum number of iterations of the velocity-inversion/hypocenter-relocation loop
     'snrmct': 0.003, # NB!!! cutoff value for solution norm. simulps!2 will stop iterating if the solution norm is less than snrmct
     'ihomo': 1, # NB! force raytracing to be in vertical planes for ihomo iterations (0/1)
     'rmstop': 0.000001, # NB!!! overall RMS residual for termination of program (if not stopped by nitmax)
     'ifixl': 0, # number of velocity inversion steps to keep hypocenters fixed at start
     # Line 5
     'delt1': 15.0, # NB! epicentral-distance weighting factors for all parts of the inversion
     'delt2': 30.0, # NB! epicentral-distance weighting factors for all parts of the inversion
     'res1': 0.50, # Controls weighting as a function of residual.
     'res2': 1.00, # Weight is 1.0 below resl, 0.0 above res3, and 0.02 at res2, with linear tapers
     'res3': 3.00, # From resl to resl, and res2 to res3. Note that res3 probably should be set very high (e.g., 3 s) in early inversion steps
     # Line6
     'ndip': 9, # Number of rotation angles (from -90° to +90°) of the plane of the ray to be used in the search for the fastest traveltime
     'iskip': 4, # NB! Number of (higher) rotation angles to be skipped on either side of vertical in this search.
     'scale1': 1.00, # NB! Step length (km) for the traveltime computation. Set no larger than the node-grid spacing.
     'scale2': 1.00, # scale (km) for the number of paths tried in the raytracing (roughly, the interval between bends in a ray).
     # Line7
     'xfax': 1.5, # convergence enhancement factor for pseudo-bending (1.2 <xfac < 1.5)
     'tlim': 0.002, # traveltime difference below which to terminate iteration (use 0.0005 <tlim < 0.002 s).
     'nitpb1': 5, # NB! maximum number of iterations allowed for pseudo-bending (use 5 to 10).
     'nitpb2': 10, # NB! nitpb (1) constrains raypaths shorter than deltl and nitpb (2) constrains raypaths longer than deltl
     # Line8
     'iuseP': 0, # 1 to invert tp for vp
     'iuseS': 1, # 1 to invert ts-tp for vp/vs
     'invdelay': 0, # 1 to invert for station delays
     # Line9
     'iuseq': 0, # Invert for Q-factor (?)
     'dqmax': 0, # Parameter for Q-inversion (?)
     'qdamp': 0 # ANother parameter for Q-inversion (?)
     }

# Pars for sps.write_STNS
lon0 = (lon1+lon2)/2 # Origin (center) of grid
lat0 = (lat1+lat2)/2 # Origin (center) of grid
rota = 0.0 # rotation angle
nzco = 0 # Always? Don't know what this parameter means. It's zero in tutorial

#-------------------------------------------
# Initial velocity model and grid def
#-------------------------------------------

# Pars for initil model setup
x_nodes = np.array([-22.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 22.0], dtype=float) 
y_nodes = np.array([-20.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0], dtype=float) 

if cntl['iuseS'] == 0:
    z_nodes = np.array([-2.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 9.0, 14.0], dtype=float)
else:
    # z_nodes = np.array([-2.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 14.0], dtype=float)
    z_nodes = np.array([-2.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 14.0], dtype=float)
    

# Nodes to fix (not update)
ix_fix = np.array([1], dtype=int)
iy_fix = np.array([1], dtype=int)
iz_fix = np.array([1], dtype=int)

nx, ny, nz = x_nodes.shape[0], y_nodes.shape[0], z_nodes.shape[0]
vp = 0*np.ones((nz,ny,nx), dtype=float)
ps_rat = 0*np.ones((nz,ny,nx), dtype=float)

# Continue here:
vp1 = np.interp(z_nodes, vel1d['depth'], vel1d['vp'])
if ps_scl:
    ps_rat1 =  ps_scl*np.interp(z_nodes, vel1d['depth'], vel1d['ps_rat'])
else:
    ps_rat1 = 2.0*np.ones_like(vp1)

for jy in range(ny):
    for jx in range(nx):
        vp[:,jy,jx] = vp1
        ps_rat[:,jy,jx] = ps_rat1
        
mod = {
       'bld': 1.0, # must be 0.1 or 1.0 km, see manual
       'nx': x_nodes.shape[0], # Model dim
       'ny': y_nodes.shape[0], # Model dim
       'nz': z_nodes.shape[0], # Model dim
       'xn': x_nodes, # x = jx*bld
       'yn': y_nodes, # y = jy*bld
       'zn': z_nodes, # z = jy*bld
       'ixf': ix_fix, # Nodes to fix
       'iyf': iy_fix, # Nodes to fix
       'izf': iz_fix, # Nodes to fix
       'vp' : vp,
       'ps_rat' : ps_rat
       }

#--------------------------------
#  Ray tracing pars
#--------------------------------

raytrac={
    'iheter': 20, 
    'epsob': 0.001, 
    'epsca': 0.1, 
    'ides': 0, 
    'ampr': 1.0, 
    'iterrai': 1,
    'dxrt': 1.0,   # In units of bld
    'dyrt': 1.0,   # In units of bld
    'dzrt': 1.0    # In units of bld
    }

cdir = simulps
fname = 'RAYTRAC'
out_file = cdir + fname

#-----------------------------------
# Check some inputs
#-----------------------------------

print('Checks and warnings:')

# Events
if len(eq_in_AOI_list) > sps.MAXEV: print(' o Too many events')
else: print(' o Events OK')

# Time picks
npick_max = 0
for dfwrk in eq_in_AOI_list:
    if dfwrk.shape[0] > npick_max: npick_max = dfwrk.shape[0]
    
if npick_max > sps.MAXOBS: print(' o Too many traveltime picks')
else: print(' o TT picks OK')

# Stations
nsta = len(eq_in_AOI['label'].unique())
if nsta > sps.MAXSTA: print(' o Too many stations')
else: print(' o Stations OK')

# Model size
if   mod['nx'] > sps.MAXNX: print(' o nx too big')
elif mod['ny'] > sps.MAXNY: print(' o ny too big')
elif mod['nz'] > sps.MAXNZ: print(' o nz too big')
else: print(' o Model def OK')

# Inversion pars:
niv = 2*((mod['nx']-2)*(mod['ny']-2)*(mod['nz']-2))
if niv > sps.MXPARI: print(f' o Too many unknowns in inversion: {niv}>{sps.MXPARI}')
else: print(' o Inversion pars OK')

#-----------------------------------
# Function calls
#-----------------------------------

# Write job control (CNTL) file
ierr = sps.write_CNTL(cntl, simulps)

# Wrtie the grid definition and initial model (MOD) file
ierr = sps.write_MOD(mod, simulps)

if not write_cntl_only:

    # Write station (STNS) file
    syn_fw, ierr = sps.write_STNS(eq_in_AOI_list, simulps, 
                                  lon0=lon0, lat0=lat0, nzco=nzco)
    
    # Write earth quake tp and ts picks
    ierr = sps.write_EQKS(eq_in_AOI_list, syn_fw, simulps)
    
    ierr = sps.write_RAYTRAC(raytrac, simulps)
    
# QC plot initial model?
if plot_qc:
    fig, axs = plt.subplots(1,2)
    ax = axs.ravel()[0]
    ax.plot(vel1d['vp'], vel1d['depth'], '-')
    ax.plot(vp1, z_nodes, 'o-')
    ax.invert_yaxis()
    
    ax = axs.ravel()[1]
    ax.plot(vel1d['ps_rat'], vel1d['depth'], '-')
    ax.invert_yaxis()
    ax.plot(ps_rat1, z_nodes, 'o-')

    plt.show(block=block)
    
print('THE END')
