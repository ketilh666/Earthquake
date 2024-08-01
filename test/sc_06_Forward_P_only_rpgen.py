# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:30:54 2024

@author: KEHOK
"""

import numpy as np
import matplotlib.pyplot as plt
import obspy.imaging.beachball as bb

from earthquake.focal import rpgen

#----------------------
# Inputs
#----------------------

png = 'png/'
block = True

# Fault properties
strike = 0.
dip = 90.0
rake_list = [0., 180., -90., 90.]

# Poisson ration
ps_rat = 1.8
nu = 0.5*(ps_rat**2-2)/(ps_rat**2-1)

# Ray azimuths
azm_max = 360.0
azm = np.linspace(0,azm_max,361)

# Ray take off angle 
tko = 60.0 

gamma = 0.0

fig, ax  = plt.subplots(1, figsize=(8,5))

for jj in range(len(rake_list)):
    
    rake = rake_list[jj] 
    
    Gp = rpgen(strike, dip, rake, gamma, nu, tko, azm, P_only=True)
    
    ax.plot(azm, Gp, label=f'rake={rake}')
        
    ax.plot(azm, np.zeros_like(azm), 'k-')
    ax.legend()
    ax.set_xlabel('Azimuth [deg]')
    ax.set_ylabel('Amplitude [-]')
    ax.set_title(f'P-wave radiation: strike={strike} deg, dip={dip} deg, take-off tko = {tko} deg')
    
fig.tight_layout(pad=1.0)
fig.savefig(png + f'rpgen_Ponly_dip{int(dip)}_tko{int(tko)}.png')

fig, axs = plt.subplots(1,4, figsize=(12,4)) 
for jj in range(len(rake_list)):
    
    ax = axs.ravel()[jj]
    
    rake = rake_list[jj] 
    bw = bb.beach([strike, dip, rake], xy=[0.5,0.5], width=0.5)
    ax.add_collection(bw)
    ax.axis('off')
    ax.axis('square')
    ax.set_title(f'rake = {rake} deg')
    
fig.suptitle(f'strike = {strike}, dip = {dip}')
fig.tight_layout(pad=1.)
fig.savefig(png + f'BeachBall_Illustration_dip{int(dip)}.png')

plt.show(block=block)
