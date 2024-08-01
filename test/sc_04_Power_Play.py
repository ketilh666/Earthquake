#------------------------------------------------
# Some numerical investigations of power laws
#------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

import earthquake.quake as quake

#---------------------------
#  Run pars
#---------------------------

block = True
png = 'png/'

run_mod = False
run_sim = True

a_list = [0.5, 1.0]

#---------------------------
#  PLot power laws
#---------------------------

if run_mod:

    dr, r2 = 50.0, 4000.0
    nr = int(r2/dr) 
    r = np.linspace(dr, r2, nr)

    lw = 3.
    density = True
    finite = False
    kpp = False
    fig = quake.power_play(r, a_list, density=density, 
                        finite=finite, lw=lw, kpp=kpp, verbose=1)
    fig.savefig(png + 'Power_Laws.png')

#----------------------------
#  Run simulation
#----------------------------

if run_sim:

    density = False
    finite = False

    # Por vs log perm trend: log_perm = alfa*por + beta
    alfa, beta = 14.50208301607354, -1.966584795783571
 
    for jj, a in enumerate(a_list):

        mu0_phi = 0.15   # porosity mean
        sig0_phi = 0.02 # porosity variance (diagonal)
        rho1 = 0.9 # Correlation with nearest neighbor
        n = 10000

        x = np.linspace(0, 1000, 11)
        y = np.linspace(0,  800,  9) 

        dd = quake.power_model(x, y, a, mu0_phi, sig0_phi, alfa, beta, n, 
                               rho1=rho1, dist='lognorm', verbose=1, kplot=True)

        # Save the figs to png
        f_roots = ['corrrel', 'pdf', 'por', 'perm']
        for kk, fig in enumerate(dd['figs']):
            fname = f'Power_Law_{f_roots[kk]}_{jj}.png'
            fig.savefig(png + fname)

plt.show(block=block)




