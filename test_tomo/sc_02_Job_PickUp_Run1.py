# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:23:06 2024

@author: KEHOK
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

import earthquake.simulPS as sps

#-------------------------------
#  Input files
#-------------------------------

krun = 1
simulps = f'run_{krun}/'

png = 'png/'
if not os.path.isdir(png): os.mkdir(png)

block = 'True'

itest_list = [2,6,13] 
#itest_list = [6]

fname_mod_ini_list = []
fname_rms_list = []
fname_mod_list = []

for itest in itest_list:
    fname_mod_ini_list.append(f'test_{itest}/MOD')
    fname_rms_list.append(f'test_{itest}/itersum.txt')
    fname_mod_list.append(f'test_{itest}/velomod.out')

# Center of the grid (from STNS)
agc = {
    'lon0': -115.59, # Center of the grid (from STNS)
    'lat0':   33.18, # Center of the grid (from STNS)
    'km_per_lon': 60*1.5544, # Approx km per degree locally (from output.txt)
    'km_per_lat': 60*1.8485 # Approx km per degree locally (from output.txt)
    }

#------------------------------
# Read inputs
#------------------------------

mod_ini_list = []
mod_list = []
rms_list = []

for jj, itest in enumerate(itest_list):
    print(jj)

    if itest <= 10:
        iusep, iuses = 1, 0    
    elif itest <= 20:
        iusep, iuses = 1, 1   
    else: 
        iusep, iuses = 0, 1   

    fname_mod_ini = fname_mod_ini_list[jj]
    fname_mod_out = fname_mod_list[jj]
    fname_rms = fname_rms_list[jj]

    mod_ini = sps.read_MOD(simulps, fname_mod_ini, agc=agc, verbose=1, iuses=1)
    mod = sps.read_MOD(simulps, fname_mod_out, mod_ini, agc=agc, 
                       verbose=1, iusep=iusep, iuses=iuses)
    rms = sps.read_itersum(simulps, fname_rms)

    mod_ini_list.append(mod_ini)
    mod_list.append(mod)
    rms_list.append(rms)

#----------------------------------------
# Make a plot
#----------------------------------------

# plOT THE RMS ERRORS
fig, ax = plt.subplots(1)

key = 'rms_w'
for jj, rms in enumerate(rms_list):

    itarr = np.array([it for it in range(len(rms[key]))])
    ax.plot(itarr, rms[key], 'o-', label=fname_rms_list[jj].split('/')[0])

ax.legend()
ax.set_xlim(0, 5)
ax.set_ylim(0, 0.40)
ax.set_xlabel('iteration [-]')
ax.set_ylabel(f'{key} error[-]')
ax.set_title(f'simulPS14: {simulps} RMS error')
fig.tight_layout(pad=1.0)
fig.savefig(png + 'RMS_error_' + str(itest_list)+ '.png') 

# PLot models
key_x, key_y = 'lon', 'lat'
mask = True
if mask: lab_mask = 'mask_on'
else:    lab_mask = 'mask_off'

for jj, itest in enumerate(itest_list):
    
    if itest <= 10:
        iusep, iuses = 1, 0    
    elif itest <= 20:
        iusep, iuses = 1, 1   
    else: 
        iusep, iuses = 0, 1   

    if iusep == 1:
        mod_out = mod_list[jj]
        title = fname_mod_list[jj] + ': vp model'
        pngfile = f'run_{krun}_test_{itest}_vp_model_{lab_mask}.png'
        print(pngfile)
        key_mod='vp'
        fig = sps.plot_MOD(mod_out, title=title, mask=mask,
                           key_x=key_x, key_y=key_y, key_mod=key_mod)
        fig.savefig(png + pngfile)
    
    if iuses == 1:
        title = fname_mod_list[jj] + ': vp/vs model'
        pngfile = f'run_{krun}_test_{itest}_ps_model_{lab_mask}.png'
        key_mod='ps_rat'
        fig = sps.plot_MOD(mod_out, title=title, mask=mask,
                           key_x=key_x, key_y=key_y, key_mod=key_mod)
        fig.savefig(png + pngfile)
        
    if iusep*iuses == 1:
        
        mod_out['vs'] = mod_out['vp']/mod_out['ps_rat']
        title = fname_mod_list[jj] + ': vs model'
        pngfile = f'run_{krun}_test_{itest}_vs_model_{lab_mask}.png'
        key_mod='vs'
        fig = sps.plot_MOD(mod_out, title=title, mask=mask,
                           key_x=key_x, key_y=key_y, key_mod=key_mod)
        fig.savefig(png + pngfile)
    
plt.show(block=block)
