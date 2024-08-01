# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:04:18 2024

@author: KEHOK
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import geopandas as gpd
import pickle

import obspy.imaging.beachball as bb

import earthquake.focal as focal

#-----------------------------
# Input folders
#-----------------------------

block = True

excel_bb = '../data/excel_bb/'
shp = '../data/shp/'
png = 'png_focal/'

#-----------------------------
# User pars
#-----------------------------

# Ray tracing or straight-ray approximation?
trace_rays = True

# Parameters for the exponential velocity trend 
phi0 = 0.387 # Porosity
v0 = 2.199 # km/s
v8 = 5.848 # km/s
gamma = 0.392 # 1/km

# Max epicenter to seismic station dstance
dist_max = 50.0 # [km] 

# Center of the grid (from STNS)
agc = {
    'lon0': -115.59, # Center of the grid (from STNS)
    'lat0':   33.18, # Center of the grid (from STNS)
    'km_per_lon': 60*1.5544, # Approx km per degree locally (from output.txt)
    'km_per_lat': 60*1.8485  # Approx km per degree locally (from output.txt)
    }

# Variances for the Bayes inversion
# sig_s0, sig_d0, sig_r0, cpri = 10, 20, 1e6, 'Narrow' 
sig_s0, sig_d0, sig_r0, cpri = 1.0e6, 1.0e6, 1.0e6, 'Infty' 

# Run analysis only or full inversion?
# mode, kplot = 'analysis', True
mode, kplot = 'inversion', True

# plot posterior mean or MAP?
post_est = 'MAP'
# post_est = 'Mean'

#-----------------------------------
# Read some cultural data
#-----------------------------------

scl = 1e-0

epsg_wgs =  4326
epsg_11N = 26711

fname = 'rhyolites.shp'
rhyolites = gpd.read_file(shp + fname)
rhyolites.geometry = rhyolites.geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))

# Salton Sea shoreline
salton_11N = gpd.read_file(shp + 'SaltonSea.shp')
bsz_11N = gpd.read_file(shp + 'BSZ.shp')

# Licenses
bhe_11N = gpd.read_file(shp + 'BHE.shp')
ctr_11N = gpd.read_file(shp + 'CTR.shp')
esm_11N = gpd.read_file(shp + 'ESM.shp')

bhe_11N.set_crs(epsg=epsg_11N, inplace=True)
ctr_11N.set_crs(epsg=epsg_11N, inplace=True)
esm_11N.set_crs(epsg=epsg_11N, inplace=True)

# Transform to UTM11N or vice versa
salton_wgs = salton_11N.to_crs({'init': f'epsg:{epsg_wgs}'})
bsz_wgs = bsz_11N.to_crs({'init': f'epsg:{epsg_wgs}'})    
bhe_wgs = bhe_11N.to_crs({'init': f'epsg:{epsg_wgs}'})
ctr_wgs = ctr_11N.to_crs({'init': f'epsg:{epsg_wgs}'})
esm_wgs = esm_11N.to_crs({'init': f'epsg:{epsg_wgs}'})

#------------------------------------
# Read data for beachball analysis
#------------------------------------

fname_list = [
    'BB_AnalysisData_eid_9059316.xlsx',
    'BB_AnalysisData_eid_9039664.xlsx',
    'BB_AnalysisData_eid_9039606.xlsx',
    'BB_AnalysisData_eid_10320761.xlsx',
    'BB_AnalysisData_eid_11065613.xlsx',
    'BB_AnalysisData_eid_15105052.xlsx',
    'BB_AnalysisData_eid_15114473.xlsx',
    'BB_AnalysisData_eid_15116089.xlsx',
    'BB_AnalysisData_eid_15116121.xlsx',
    'BB_AnalysisData_eid_15116321.xlsx',
    'BB_AnalysisData_eid_11069757.xlsx',
    'BB_AnalysisData_eid_15344513.xlsx',
    'BB_AnalysisData_eid_15344585.xlsx',
    'BB_AnalysisData_eid_15344465.xlsx',
    'BB_AnalysisData_eid_15354321.xlsx',
    'BB_AnalysisData_eid_15354537.xlsx'
        ]

P_for_bb_list = [None for fname in fname_list]
for jj, fname in enumerate(fname_list):
    with pd.ExcelFile(excel_bb + fname) as fid:
        P_for_bb_list[jj] = pd.read_excel(fid)

# Filtering        
for jj, eq in enumerate(P_for_bb_list):
    ind = eq['dist'] <= dist_max
    P_for_bb_list[jj] = eq[ind]
        
g2d_lat = agc['km_per_lat']
g2d_lon = agc['km_per_lon']
r2d = 180.0/np.pi

# Event selection
kk_list = [k for k in range(len(P_for_bb_list))]

#----------------------------------------
# Solution space for strike, dip, rake
#----------------------------------------

sarr = np.linspace(0,  180, 31)  # Strike search grid
darr = np.linspace(15,  90, 16)  # Dip    search grid
rarr = np.linspace(-90, 90, 37)  # Rake   search grid

ns, nd, nr = sarr.shape[0], darr.shape[0], rarr.shape[0]
nn = ns*nd*nr

print(f'nn = {nn}')

#--------------------------------------------------
# Prior mean and variance for Bayesian inversion
#--------------------------------------------------

# Prior mean and variance
neq = len(P_for_bb_list)
mu_s,  mu_d,  mu_r  = 180.*np.ones(neq), 90.*np.ones(neq), 0.*np.ones(neq)

kk =   0; mu_s[kk], mu_d[kk], mu_r[kk] = 180, 90,  10
kk =   1; mu_s[kk], mu_d[kk], mu_r[kk] =  60, 70,  20
kk =   2; mu_s[kk], mu_d[kk], mu_r[kk] =  40, 80,  15 
kk =   3; mu_s[kk], mu_d[kk], mu_r[kk] = 100, 85,   0
kk =   4; mu_s[kk], mu_d[kk], mu_r[kk] =  35, 20,  50
kk =   5; mu_s[kk], mu_d[kk], mu_r[kk] =  50, 55,  50
kk =   6; mu_s[kk], mu_d[kk], mu_r[kk] =  25, 25,  50
kk =   7; mu_s[kk], mu_d[kk], mu_r[kk] =  65, 75,  35  
kk =   8; mu_s[kk], mu_d[kk], mu_r[kk] = 160, 70, -70 
kk =   9; mu_s[kk], mu_d[kk], mu_r[kk] =  35, 80,  10
kk =  10; mu_s[kk], mu_d[kk], mu_r[kk] =  50, 60,  55
kk =  11; mu_s[kk], mu_d[kk], mu_r[kk] = 160, 80,  -5
kk =  12; mu_s[kk], mu_d[kk], mu_r[kk] =  70, 65,  45
kk =  13; mu_s[kk], mu_d[kk], mu_r[kk] =  55, 65,  25 
kk =  14; mu_s[kk], mu_d[kk], mu_r[kk] =  70, 60,   0  
kk =  15; mu_s[kk], mu_d[kk], mu_r[kk] = 170, 90,   0

# Scale variance with number of picks
n_max = max([eq.shape[0] for eq in P_for_bb_list])
sig_s = np.zeros_like(mu_s)
sig_d = np.zeros_like(mu_d)
sig_r = np.zeros_like(mu_r)
for kk in range(len(P_for_bb_list)):
    npick = P_for_bb_list[kk].shape[0]
    sig_s[kk] = (n_max/npick)*sig_s0
    sig_d[kk] = (n_max/npick)*sig_d0
    sig_r[kk] = (n_max/npick)*sig_r0
    
#-------------------------------------
# Precompute ray tracing
#-------------------------------------

if trace_rays: cray = 'RayTraced'
else: cray = 'Geometric'

# Used in rpgen (not important)
ps_rat = 1.75
nu = 0.5*(ps_rat**2-2.0)/(ps_rat**2-1.0)

# Ray tracing?
if trace_rays:
    
    # Depth range
    zmax, nz = 12.0, 800
    dz = zmax/nz
    zarr = np.linspace(0,zmax,nz)
    
    # Velocity model
    vtrend = focal.vel_trend(zarr, v0, v8, gamma)
    phitrend = focal.por_trend(zarr, phi0, gamma)
        
    # Slowness range
    d2r, r2d = np.pi/180.,  180.0/np.pi
    tko_arr = np.linspace(1,89, 89) # Take-off angle from horizontal (grazing angle)
    parr = np.cos(d2r*tko_arr)/v0   # Horizontal slowness
    nray = parr.shape[0]
        
    # Trace some arrays
    ray_list = [None for p in parr]
    for jj,p in enumerate(parr):
        # print(f'jj = {jj:2d}: p = {p:5.3f} s/km')
        ray_list[jj] = focal.raytrace(zarr, v0, v8, gamma, p) 
    
    # Make som plots
    if kplot:
        fig, axs = plt.subplots(1, 3, figsize=(18,4), width_ratios=[1,1,6])
        # PLot the vporosity trend
        ax = axs.ravel()[0]
        ax.plot(phitrend, zarr, label=f'gam = {gamma}/km')
        ax.invert_yaxis()
        ax.legend()
        ax.set_xlabel('phi [-]')
        ax.set_ylabel('depth [km]')
        ax.set_title('Porosity trend')
        
        # PLot the velocity trend
        ax = axs.ravel()[1]
        ax.plot(vtrend, zarr, label=f'gam = {gamma}/km')
        ax.invert_yaxis()
        ax.legend()
        ax.set_xlabel('vp [km/s]')
        ax.set_ylabel('depth [km]')
        ax.set_title('Velocity trend')
        
        # Plot the rays
        ax = axs.ravel()[2]
        xr_list, xa_list = [], []
        n_plot = 25        
        inc = np.maximum(int(nray/n_plot),1)
        for jj in range(0,nray,inc):
            ax.plot(ray_list[jj]['x'], ray_list[jj]['z'], '-')
        ax.invert_yaxis()
        ax.set_xlabel('offset [km]')
        ax.set_ylabel('depth [km]')
        ax.axis('scaled')
        ax.set_title('Ray-tracing (some of the rays)')
 
        # plot hypocenter offsets
        for kk in kk_list[0:5]:
            eq = P_for_bb_list[kk]
            eid = eq.loc[0,'eid']
            ax.scatter(eq['dist'],  eq['depth0'], marker='o', label=f'eid={eid}')
        ax.legend()
            
        fig.tight_layout(pad=1.)
        fig.savefig(png + 'Ray_Tracing.png')
        
#----------------------------
#  Bayes inversion
#----------------------------

# Loop over events
fm_list = [None for kk in kk_list]
ns_list = [None for kk in kk_list]
for jj, kk in enumerate(kk_list):
    
    # Prior mean and variance
    pp = {
          'mu_s': mu_s[kk], 'sig_s': sig_s[kk], # Prior strike
          'mu_d': mu_d[kk], 'sig_d': sig_d[kk], # Prior dip
          'mu_r': mu_r[kk], 'sig_r': sig_r[kk]  # Prior rake
          }
    
    # Noise mean and variance
    err = {
           'mu': 0.,
           'sig': 1.0      
           }

    # Data for current event
    eq = P_for_bb_list[kk]
    
    # Ray azimuth
    delx = g2d_lon*(eq['lon'] - eq['lon0_reloc'])
    dely = g2d_lat*(eq['lat'] - eq['lat0_reloc'])
    delr = eq['dist']
    deld = eq['depth0'] - 1e-3*eq['elev']
    AZI = r2d*np.arctan2(delx,dely) # Ray azimuth angles

    if trace_rays:
        TKO = np.zeros_like(AZI)
        # Work arrays
        r2 = np.zeros(nray)
        s2 = np.zeros(nray)

        # Loop over picks for current event
        npick = eq.shape[0]
        for jp in range(npick):
          
            xp = eq.loc[jp, 'dist'] 
            zp = eq.loc[jp,'depth0'] #- 1e-3*eq.loc[jp,'elev']
            for jr, ray in enumerate(ray_list):
                xr, zr = np.real(ray['x']), np.real(ray['z'])
                wrk = (xr-xp)**2 + (zr-zp)**2
                idd = np.argmin(wrk)
                r2[jr] = wrk[idd]
                s2[jr] = ray['ss'][idd]
            
            kop = np.argmin(r2)
            vel = focal.vel_trend(zp, v0, v8, gamma)
            ss = s2[kop]
            arg = np.minimum(1.0, ray_list[kop]['p']*vel)
            phi = r2d*np.arcsin(arg)
            TKO[jp] = ss*(90-phi)

            # print(f'{jp}, {xp}, {zp}, {kop}, {ss}, {phi:5.1f}, {TKO[jp]:5.1f}')
            
    else:
        
        # Ray take off angle, ignoring ray bending
        TKO = 1*r2d*np.arctan(deld/delr) # ray grazing angle (rel horizontal plane)
        #TKO = 0.0

    # Store in df
    new_keys = ['dely', 'delr', 'deld', 'AZI', 'TKO']
    new_data = np.array([dely, delr, deld, AZI, TKO]).T
    eq[new_keys] = new_data
    
    fm, figs = focal.bayes_beach(eq, sarr, darr, rarr, pp, err, nu, 
                                 mode=mode, kplot=kplot, verbose=1, kk=kk)

    fm_list[jj] = fm
    
    ns_list[jj] = (kk, fm.shape[0])
        
    # Save plots to png
    if kplot:    
        for kf, fig in enumerate(figs):
            fname = f'Gregoires_List_QC_plot_{kf}_{kk}_{cpri}_Prior_{cray}.png'
            fig.savefig(png + fname)

#-------------------------
#  summarize results
#-------------------------

# Cost function minima, usually more than one
with pd.ExcelWriter(excel_bb + f'Cost_Function_Minima_from_{mode}.xlsx') as fid:
    for jj, kk in enumerate(kk_list):
        sname = str(fm_list[jj].loc[0,'eid'])
        fm_list[jj].to_excel(fid, index=False, sheet_name=sname)

if mode.lower()[0] == 'i':

    nn = len(fm_list)
    mu_s_map = np.zeros(nn)
    mu_d_map = np.zeros(nn)
    mu_r_map = np.zeros(nn)
    eids = np.zeros(nn, dtype=int)
    npks = np.zeros(nn, dtype=int)
    for kk, fm in enumerate(fm_list):
        eids[kk] = fm.loc[0,'eid']
        npks[kk] = P_for_bb_list[kk].shape[0]
        mu_s_map[kk] = fm.loc[0,'mu_s_map']
        mu_d_map[kk] = fm.loc[0,'mu_d_map']
        mu_r_map[kk] = fm.loc[0,'mu_r_map']

    cols = ['eid', 'npick', 'strike', 'dip', 'rake']
    data = np.array([eids, npks, mu_s_map, mu_d_map, mu_r_map]).T
    fm_summary = pd.DataFrame(columns=cols, data=data)

    with pd.ExcelWriter(f'Inversion_Results_{cpri}_Prior_{cray}.xlsx') as fid:
        fm_summary.to_excel(fid, index=False)

#-------------------------
#  PLot results 
#-------------------------

if mode.lower()[0] == 'i':
    
    #--- PLot beachballls on map
    # nrow, ncol = 3, len(kk_list)
    fig, ax = plt.subplots(1,1, figsize=(12,8))
    bb_list = [None for kk in kk_list]
    for jj, kk in enumerate(kk_list):
    
        eid = fm_list[jj].loc[0,'eid']
        s = fm_list[jj].loc[0,'mu_s_map']
        d = fm_list[jj].loc[0,'mu_d_map']
        r = fm_list[jj].loc[0,'mu_r_map']
        idd = P_for_bb_list[kk].index[0]
        lon = P_for_bb_list[kk].loc[idd,'lon0_reloc']
        lat = P_for_bb_list[kk].loc[idd,'lat0_reloc']
        bw = bb.beach([s,d,r], xy=[lon, lat], width=0.003)
        ax.add_collection(bw)
        bb_list[jj] = bw
        
    salton_wgs.plot(ax=ax, color='y', linewidth=1.0, label='Shoreline')
    bsz_wgs.plot(ax=ax, color='tab:purple', linewidth=1.0, label='Brawley SZ')
    bhe_wgs.boundary.plot(ax=ax, color='tab:orange', linewidth=1.5, label='BHE')
    ctr_wgs.boundary.plot(ax=ax, color='tab:pink', linewidth=1.5, label='CTR')
    esm_wgs.boundary.plot(ax=ax, color='tab:red', linewidth=1.5, label='ESM')
    
    ax.set_aspect('equal')
    ax.set_xlim(-115.590, -115.539)
    ax.set_ylim(33.196, 33.241)
    
    fig.suptitle(f'Focal mechanisms ({post_est})')
    fig.tight_layout(pad=2.)
    fig.savefig(png + f'BeachBalls_on_Map_{cray}_{cpri}_prior_{post_est}.png')
   
    #--- PLot all the beach balls
    nrow, ncol = 4, len(kk_list)//4
    fig, axs = plt.subplots(nrow, ncol, figsize=(12,10))
    for jj, kk in enumerate(kk_list):
        ax = axs.ravel()[jj]

        if post_est.lower() == 'mean':
            s = fm_list[jj].loc[0,'mu_s_post']
            d = fm_list[jj].loc[0,'mu_d_post']
            r = fm_list[jj].loc[0,'mu_r_post']
        else:
            s = fm_list[jj].loc[0,'mu_s_map']
            d = fm_list[jj].loc[0,'mu_d_map']
            r = fm_list[jj].loc[0,'mu_r_map']
       
        eid = fm_list[jj].loc[0,'eid']
        bw = bb.beach([s,d,r], xy=[0.5, 0.5], width=0.5)
        ax.add_collection(bw)
        ax.axis('off')
        ax.axis('square')
        ax.set_title(f'eid = {eid}')
        
    fig.suptitle(f'Focal mechanisms ({post_est})')
    fig.tight_layout(pad=2.)
    fig.savefig(png + f'BeachBalls_{cray}_{cpri}_prior_{post_est}.png')
        
    #--- PLot data vs modeling
    fig, axs = plt.subplots(nrow, ncol, figsize=(14,10))
    for jj, kk in enumerate(kk_list):
        ax = axs.ravel()[jj]

        if post_est.lower() == 'mean':
            s = fm_list[jj].loc[0,'mu_s_post']
            d = fm_list[jj].loc[0,'mu_d_post']
            r = fm_list[jj].loc[0,'mu_r_post']
        else:
            s = fm_list[jj].loc[0,'mu_s_map']
            d = fm_list[jj].loc[0,'mu_d_map']
            r = fm_list[jj].loc[0,'mu_r_map']

        eid = fm_list[jj].loc[0,'eid']

        AZI = P_for_bb_list[kk]['AZI'].sort_values()
        TKO = P_for_bb_list[kk].loc[AZI.index, 'TKO']
        ipol =  P_for_bb_list[kk].loc[AZI.index, 'ipol']

        ax.plot(np.array([-180., 0, 180]), np.array([0., 0., 0.]), 'k-')
        sc = ax.plot(AZI, 1.1*ipol, 'r-', zorder=0)
        sc = ax.scatter(AZI, 1.1*ipol, c='r', s=36, label='data')
        cb = ax.figure.colorbar(sc, ax=ax, label='Ray take-off [deg]')
        Gp = focal.rpgen(s, d, r, 0, nu, TKO, AZI, P_only=True)
        ax.scatter(AZI, 0.9*np.sign(Gp), c=TKO, s=36, label='model')
        if jj==3: ax.legend()
        ax.set_title(f'eid = {eid}')
        ax.set_xlabel('Ray azimuth [deg]')
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_ylabel('sign(Gp) [-]')

    fig.suptitle(f'Data vs forward modeling ({post_est})')
    fig.tight_layout(pad=2.)
    fig.savefig(png + f'Data_fit_{cray}_{cpri}_prior_{post_est}.png')
    
plt.show(block=block)
