# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:30:54 2024

@author: KEHOK
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from earthquake.focal import rpgen

#----------------------
# Inputs
#----------------------

block = True

png = 'png/'
if not os.path.isdir(png): os.mkdir(png)

strike_0 = 0.0
dip_0 = 90.0
rake_0 = [0. ,45.0, -90.]

# Poisson ration
ps_rat = 1.8
nu = 0.5*(ps_rat**2-2)/(ps_rat**2-1)

# Take off angle 
tko_0 = +30.

azm_max = 180.0
azm = np.linspace(0,azm_max,181)
tko = tko_0 

strike = strike_0 
dip = dip_0 
gamma = 0.0

fig, axs = plt.subplots(2,2, figsize=(16,7))

for jj in range(len(rake_0)):
    
    rake = rake_0[jj]*np.ones_like(azm)
    
    Gp, Gs, Gsh, Gsv = rpgen(strike, dip, rake, gamma, nu, tko, azm, P_only=False)
    
    ax = axs.ravel()[0]
    ax.plot(azm, Gp, label=f'rake={rake_0[jj]}')
    
    ax = axs.ravel()[1]
    ax.plot(azm, Gs, label=f'rake={rake_0[jj]}')

    ax = axs.ravel()[2]
    ax.plot(azm, Gsh, label=f'rake={rake_0[jj]}')

    ax = axs.ravel()[3]
    ax.plot(azm, Gsv, label=f'rake={rake_0[jj]}')
    
tit_list = ['Gp', 'Gs', 'Gsh', 'Gsv']
for kk in range(len(axs.ravel())):
    ax.plot(azm, np.zeros_like(azm), 'k-')
    ax = axs.ravel()[kk]
    ax.legend()
    ax.set_xlabel('Azimuth [deg]')
    ax.set_ylabel('Amplitude [-]')
    ax.set_title(tit_list[kk])
    
fig.suptitle(f'rpgen output: take-off polar angle tko_0={tko_0} deg, strike={strike} deg, dip={dip} deg')
fig.tight_layout(pad=1.0)

fig.savefig(png + f'rpgen_radiation_tko{int(tko_0)}.png')

plt.show(block=block)
