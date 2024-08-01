# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:06:46 2024

@author: KEHOK
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

# KetilH stuff
from gravmag.common import MapData

#-----------------------------------------------------
#   Statistical simulation
#-----------------------------------------------------

def power_model(x, y, a, mu0_phi, sig0_phi, alfa=14.5, beta=-2.0, n=1000, **kwargs):
    """ Simulate n samples from a correlated normal ditribution.
    
    The correlation is computed by the two-point power-law correlation. 
    
    Parameters
    ----------
    x, y: array of floats. (x,y) grid or axes
       shape=(ny,nx) or shape=nx, shape=ny  
    a: float. power correlation power law corr ~ 1/r**a
    mu0: float. Mean porosity (as for a normal distribution)
    sig0: float. Diagonal covariance of porosity. sig = rho*sig0**2
        shape=(ny,nx) if sig0 is array
    alfa: float. por-perm correlation; log_perm=alfa*por + beta
    beta: float. por-perm correlation; log_perm=alfa*por + beta
    n: int. Number of realizations to simulate (default is n=1000)
    
    kwargs
    ------
    rho1: float. Nearest neighbor correlation (default is rho1=0.9)
    dist: str. 'norm' or 'lognorm' multivariate distribution (default is 'lognorm')
    density: bool. Normalize as pdf? (default is density=False)
    finite: bool. Finite at x=0? (default is finite=False)
    verbose: int. Print shit if verbose>0
    kplot: bool. PLot or not? (default is kplot=False)

    Returns
    -------
    samps: array of floats, shape=(n,ny,nx). 
    
    Programmed: KetilH, 26. July 2024.
    """

    density = kwargs.get('density', False)
    finite  = kwargs.get('finite', False)
    dist = kwargs.get('dist', 'lognormal')
    rho1 = kwargs.get('rho1', 0.90) # Nearest neighbor correlation
    verbose  = kwargs.get('verbose', 0)
    kplot = kwargs.get('kplot', False)

    # Prepare the mean and variance (phi is lognorm ~(mu0_phi, sig0_phi))
    if dist.lower()[0] == 'l':
        # lognorm to norm parameters
        ww2 = (sig0_phi/mu0_phi)**2
        mu0  = np.log(mu0_phi/np.sqrt(1+ww2))
        sig0 = np.sqrt(np.log(1.0 + ww2))  

    else:
        # normal distribution
        mu0  = mu0_phi
        sig0 = sig0_phi
        print('power_model: Normal distrubution not inplemented')

    # Spatial grid
    if x.ndim == 1:
        gx, gy = np.meshgrid(x, y)
    else: 
        gx, gy = x, y

    # 2D grids to vectors
    xx, yy = gx.ravel(), gy.ravel()
    ns = xx.shape[0]

    # print?
    if verbose>0:
        print('earthquake.quake.power_model:')
        print(f' o a = {a}')
        print(f' o n = {n}')
        print(f' o ns = {ns}')
        print(f' o mu0_phi  = {mu0_phi}')
        print(f' o sig0_phi = {sig0_phi}')
        print(f' o mu0   = {mu0}')
        print(f' o sig0 = {sig0}')

    # Normalization of the power-law correlation:
    dx, dy = x[1]-x[0], y[1]-y[0]
    r1 = np.min([dx, dy])
    r2 = np.sqrt((np.max(x)-np.min(x))**2 + (np.max(y)-np.min(y))**2)
   
    if density:
        # Normalize as pdf (analytical)
        tiny = 1e-6
        if np.abs(a-1.0) < tiny:
            rn = rho1*1.0/np.log(r2/r1)
        else:
            rn = rho1*(1-a)/(r2**(1-a)-r1**(1-a))

    else:
        # Normalize on r1
        rn = rho1*r1**a

    # Compute the correlation matrix
    rho = np.zeros((ns,ns)) # correlation matrix
    rrr = np.zeros((ns,ns)) # distances
    for jj in range(ns):
        rrr[jj,jj] = 0.0 
        rho[jj,jj] = 1.0
        for ii in range(jj+1,ns):
            rrr[jj,ii] = np.sqrt((xx[jj]-xx[ii])**2 + (yy[jj]-yy[ii])**2)
            rho[jj,ii] = rn/rrr[jj,ii]**a
            rho[ii,jj] = rho[jj,ii]

    ### Run simulations
    rng = np.random.default_rng()
    mu  = mu0*np.ones_like(xx)
    sig = rho*sig0**2
    samps = rng.multivariate_normal(mu, sig, size=n, method='svd')

    # Select some models at random (for plotting)
    por_list  = []
    perm_list = []
    n_rand_mod = np.min([5*4, n])
    ind_rand = np.random.randint(0, n-1, n_rand_mod)
    for ind in ind_rand:
        wrk_por = np.reshape(np.exp(samps[ind,:]), gx.shape)
        wrk_perm = np.exp(np.log(10)*(alfa*wrk_por + beta))
        por_list.append(wrk_por)
        perm_list.append(wrk_perm)

    ### Plot simulation results
    fig_list = []
    if kplot:

        # PLot correlation and covariance matrix
        fig, axs = plt.subplots(1,3, figsize=(15,4))

        ax = axs.ravel()[0]
        im = ax.imshow(rrr, cmap='magma')
        cb = ax.figure.colorbar(im, ax=ax)
        ax.set_title('Radial distance')

        ax = axs.ravel()[1]
        im = ax.imshow(rho, cmap='magma')
        cb = ax.figure.colorbar(im, ax=ax)
        ax.set_title('Correlation')

        ax = axs.ravel()[2]
        im = ax.imshow(sig, cmap='magma')
        cb = ax.figure.colorbar(im, ax=ax)
        ax.set_title('Variance')

        for ax in axs.ravel():
            ax.set_xlabel('index_1')
            ax.set_ylabel('index_2')

        fig.suptitle(f'Power law correlation and variance: a={a}')
        fig.tight_layout(pad=1.0)
        fig_list.append(fig)

        # Cross pLot some paris of locations 
        fig, axs = plt.subplots(2,4, figsize=(12,8))
        for jj in range(4):
            
            vmin = np.max([0, mu0_phi-5*sig0_phi])
            vmax = mu0_phi+7*sig0_phi

            ax = axs.ravel()[jj]
            ax.hist(np.exp(samps[:,jj+1]), bins=n//100, density=True)
            ax.set_xlim(vmin, vmax)
            ax.set_title(f'index {jj+1} pdf')

            ax = axs.ravel()[jj+4]
            ax.scatter(np.exp(samps[:,0]), np.exp(samps[:,jj+1]))
            ax.axis('scaled')
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
            ax.set_title(f'index 0 vs {jj+1}')

            mean = np.mean(np.exp(samps[:,jj+1]))
            std  = np.std(np.exp(samps[:,jj+1]))
            print(f'mean and std = {mean}, {std}')

        fig.suptitle(f'Power law: a={a}')
        fig.tight_layout(pad=1.0)
        fig_list.append(fig)

        # PLot some simulated por and perm models
        scl = 1e-3
        xtnt = scl*np.array([x[0], x[-1], y[0], y[-1]])
        nrow = 5 
        ncol = n_rand_mod//nrow
    
        fig, axs =  plt.subplots(nrow, ncol, figsize=(13,12))
        for jj in range(ncol*nrow):
            ax = axs.ravel()[jj]
            ind = ind_rand[jj]
            vmin = np.max([0, mu0_phi-3*sig0_phi])
            vmax = mu0_phi+5*sig0_phi
            im = ax.imshow(por_list[jj], origin='lower', extent=xtnt, 
                           vmin=vmin, vmax=vmax)
            cb = ax.figure.colorbar(im, ax=ax)
            ax.axis('scaled')
            ax.set_title(f'Model {ind}')
            ax.set_xlabel('x [km]')
            ax.set_ylabel('y [km]')
        
        fig.suptitle(f'Porosity [-] (sampled from the pdf): a={a}')
        fig.tight_layout(pad=2.0)
        fig_list.append(fig)

        fig, axs =  plt.subplots(nrow, ncol, figsize=(13,12))
        for jj in range(ncol*nrow):
            ax = axs.ravel()[jj]
            ind = ind_rand[jj]
            wmin = np.max([0, mu0_phi-3*sig0_phi])
            wmax = mu0_phi+3*sig0_phi
            vmin = np.exp(np.log(10)*(alfa*wmin+beta))
            vmax = np.exp(np.log(10)*(alfa*wmax+beta))
            im = ax.imshow(perm_list[jj], origin='lower', extent=xtnt, 
                           vmin=vmin, vmax=vmax)
            cb = ax.figure.colorbar(im, ax=ax)
            ax.axis('scaled')
            ax.set_title(f'Model {ind}')
            ax.set_xlabel('x [km]')
            ax.set_ylabel('y [km]')
        
        fig.suptitle(f'Permeability [mD] (sampled from the pdf): a={a}')
        fig.tight_layout(pad=2.0)
        fig_list.append(fig)

    dd = {
        'dist': dist,
        'samps': samps,
        'por_models': por_list,
        'perm_model': perm_list,
        'alfa': alfa,
        'beta': beta,
        'figs': fig_list
        }

    return dd

#--------------------------------------------------------
#  Power-law analysis by two-point correlation function
#--------------------------------------------------------

def power_correl(df, clu_list, dr=100., **kwargs):
    """ Analyse spatial two-point correlation laws of earth quake clusters.   
    
    Parameters
    ----------
    df: array of floats. Regular array of spatial coordinates [m]
    a_list: list of powers (defulat is a_list=[0.5, 1.0])

    kwargs
    ------
    a_list: list of powers (defulat is a_list=[0.5, 1.0])
    density: bool. Normalised pdf? (default si density=True)
    finite: bool. Finite at x=0? (default is finite=False)

    Returns
    -------
    fig: pyplot figrue object
    
    Programmed: KetilH, 23. July 2024.
    """

    # Get the kwargs
    key_x = kwargs.get('key_x', 'x')
    key_y = kwargs.get('key_y', 'y')
    key_z = kwargs.get('key_z', 'depth') # depth>0
    key_id = kwargs.get('key_id', 'clu_id')
    rmax = kwargs.get('rmax', 4000.0)
    zmin = kwargs.get('zmin', 0.0)    # min depth
    zmax = kwargs.get('zmax', np.inf) # max depth
    a_list = kwargs.get('a_list', [.5, 1.0])
    density = kwargs.get('density', True)
    finite  = kwargs.get('finite', False)
    verbose  = kwargs.get('verbose', 0)

    # Print some shit?
    if verbose>0:
        print(f'earthquake.quake.power_law:')
        print(f' o dr = {dr}')
        print(f' o rmax = {rmax}')
        print(f' o zmin = {zmin}')
        print(f' o zmax = {zmax}')

    # Compute two-point correlation function
    corrs = [None for idd in clu_list]
    rrs   = [None for idd in clu_list]
    rws   = [None for idd in clu_list]
    for jj, idd in enumerate(clu_list):

        ind = ((df[key_id] == idd)) & \
              (df[key_z]>zmin) & (df[key_z]<zmax) 
        x = np.array(df[ind][key_x])
        y = np.array(df[ind][key_y])
        corrs[jj], rrs[jj], rws[jj] = two_point_correl(x, y, dr=dr)
  
        # Normalize as pdf
        jnd = rrs[jj] <= rmax
        rrs[jj] = rrs[jj][jnd]
        rws[jj] = rws[jj][jnd]
        corrs[jj] = corrs[jj][jnd]
        rf= 1/np.sum(corrs[jj]*dr)
        corrs[jj] = rf*corrs[jj]

    # Power-law regression
    i1, ri2 = 1, 0.75
    pows = [None for idd in clu_list]
    plaws = [None for idd in clu_list]
    for jj, idd in enumerate(clu_list):

        i2 = int(ri2*corrs[jj].shape[0]) # skip edge effect
        ind = corrs[jj][i1:i2] > 0.
        rwrk = np.log(rrs[jj][i1:i2][ind])
        cwrk = np.log(corrs[jj][i1:i2][ind])
        ok = cwrk.shape[0]>1

        if ok:
            bb, aa = np.polyfit(rwrk, cwrk, 1)
        else:
            bb, aa = 0.0, 0.0

        plaws[jj] = aa*rrs[jj]**bb
        pows[jj] = -bb
        print(f'idd, pow = {idd}, {pows[jj]}')

        if ok:
            rf = 1/np.sum(plaws[jj]*dr)
            plaws[jj] = rf*plaws[jj]

    # Compute Fourier spectrum
    karrs   = [None for idd in clu_list]
    corrs_k = [None for idd in clu_list]
    plaws_k = [None for idd in clu_list]
    for jj, idd in enumerate(clu_list):
        nk = rrs[jj].shape[0]
        # print(f'idd, nk = {idd}, {nk}')
        karrs[jj] = 2*np.pi*np.fft.fftfreq(nk, dr)
        corrs_k[jj] = np.fft.fft(corrs[jj])
        plaws_k[jj] = np.fft.fft(plaws[jj])

    # Plot theoretical power-law template
    lw = 0.5
    nr = int(rmax/dr)
    r  = np.linspace(dr, rmax, nr)
    fig = power_play(r, a_list, density=density, finite=finite, lw=lw)
    axs = fig.axes

    # PLot data in r-domain
    kols = [TABLEAU_COLORS[key] for key in TABLEAU_COLORS.keys()]
    for ax in [axs[0], axs[3]]:

        for jj, idd in enumerate(clu_list):
            i2 = int(ri2*corrs[jj].shape[0]) # skip edge effect
            ax.plot(rrs[jj], corrs[jj], 'o', color=kols[jj], 
                    label=f'{key_id}={idd} (a={pows[jj]:.2f})') 
            ax.plot(rrs[jj][i1:i2], plaws[jj][i1:i2], '-', color=kols[jj]) 

    # PLot data in k-domain
    for ax in [axs[1], axs[4]]:
        
        for jj, idd in enumerate(clu_list):
            nk = karrs[jj].shape[0]
            ik1, ik2 = 1, nk//2 - (1 + nk%2)
            ax.plot(karrs[jj][ik1:ik2], np.abs(corrs_k[jj])[ik1:ik2], 
                    'o', color=kols[jj], label=f'{key_id}={idd})') 

    # PLot data vs wavelength
    for ax in [axs[2], axs[5]]:

        for jj, idd in enumerate(clu_list):
            lam = np.zeros_like(karrs[jj])
            lam[1:] = 2*np.pi/karrs[jj][1:]
            nk = karrs[jj].shape[0]
            ik1, ik2 = 1, nk//2 - (1 + nk%2)
            ax.plot(lam[ik1:ik2], np.abs(corrs_k[jj])[ik1:ik2], 
                    'o', color=kols[jj], label=f'{key_id}={idd}') 

    for ax in axs:
        ax.legend()

    fig.tight_layout(pad=1.0)

    return fig

def two_point_correl(x, y, **kwargs):
    """Compute two-point correlation function of earthquakes, 
    which is a function of distance r only.
    
    The code is ported from the Matlab function given by Leary et al. (2019).
    
    Parameters
    ----------
    x, y: array of floats. Coordinates of earthquakes
    
    kwargs
    ------
    dr: float. radial sampling (default is dr=100m)
     
    Returns
    -------
    corr_n: Array of float. Correlation function
    rr_n: Array of float. radial distances with non-zero contributions 
    rw_n: Array of int. Number of contributions in each non-empty bin
    
    Programmed: KetilH, 27. February 2024
    """
    
    # Get the kwargs
    dr = kwargs.get('dr', 100.0)

    # Domain size
    xspan = np.max(x)-np.min(x)
    yspan = np.max(y)-np.min(y)
    rspan = np.sqrt((xspan/2)**2 + (yspan/2)**2)
    
    # Radial gridding
    nr = int(np.round(rspan/dr))
    rr = np.linspace(dr,nr*dr,nr)
    
    # Average EQ density
    neq = x.shape[0]
    dens_avg = neq/(xspan*yspan)       # Average EQ density
    r_area = (2*np.pi*rr*dr)*dens_avg  # Normalized radial area
    
    # Allocate output arrays
    corr = np.zeros_like(rr, dtype=float)
    rw = np.zeros_like(rr, dtype=int)
    
    # Compute correlation function
    for jj in range(neq):
        xj, yj = x[jj], y[jj]
        delr = np.hypot(x-xj, y-yj)
        delr = delr[delr>1e-2*dr] # Remove the zero at xj, yj
        maxr = np.min([np.max(x)-xj, xj-np.min(x), np.max(y)-yj, yj-np.min(y)]) 
        delr = delr[delr<maxr] # clip at max unbiased r
        rw[rr<maxr] += 1
        [hist, bin_edges] = np.histogram(delr, bins=np.concatenate([[-np.inf], rr]))
        corr = corr + hist/r_area
            
    ind = rw != 0
    corr_n = corr[ind]/rw[ind]
    rr_n = rr[ind]
    rw_n = rw[ind]
    
    return corr_n, rr_n, rw_n

def power_play(x, a_list=[0.5, 1.], **kwargs):
    """ Plot power-law template in r-space and k-space. 
    
    Investigate power laws in space and wavenumber co-ordinates. 

    Parameters
    ----------
    x: array of floats. Regular array of spatial coordinates [m]
    a_list: list of powers (defulat is a_list=[0.5, 1.0])

    kwargs
    ------
    density: bool. Normalised pdf? (default si density=True)
    finite: bool. Finite at x=0? (default is finite=False)
    lw: float. Linewidth

    Returns
    -------
    fig: pyplot figrue object
    
    Programmed: KetilH, 18. July 2024.
    """

    # kwargs
    density = kwargs.get('density', True)
    finite  = kwargs.get('finite', False)
    lw = kwargs.get('lw', 1.0)
    kpp = kwargs.get('kpp', False) # Plot perm correlation?
    verbose = kwargs.get('verbose', 0) # Print shit?

    # Compute x-domain functions
    f_list = [None for a in a_list]
    # w_list = f_list.copy() # For testing analytical normalization
    nx = x.shape[0]
    dx = x[1]-x[0]

    if verbose>0:
        print('earthquake.quake.power_law:')
        print(f' o nr = {dx}')
        print(f' o dr = {nx}')
        print(f' o a_list = {a_list}')

    for jj, a in enumerate(a_list):

        if finite:
            f_list[jj] = 1/(1+x**2)**(a/2)

        else:
            f_list[jj] = 1/x**a
            # w_list[jj] = 1/x**a

        # Normalize
        if density:
            rn = 1.0/np.sum(f_list[jj]*dx)
            # if np.abs(a-1.0) < 1e-6:
            #     rw = 1.0/np.log(x[-1]/x[0])
            # else:
            #     rw = (1-a)/(x[-1]**(1-a)-x[0]**(1-a))
            # print(f'rn, rw = {rn}, {rw}')
        else:
            rn = 1.0/f_list[jj][0]

        f_list[jj] = rn*f_list[jj]
        # w_list[jj] = rw*w_list[jj]

    # Fourier domain
    # k_nyq, dk = np.pi/dx, 2*np.pi/(nx*dx)
    ik_nyq = nx//2 
    karr = 2*np.pi*np.fft.fftfreq(nx, dx)
    karr[ik_nyq] = -karr[ik_nyq] # Fix som shit from np.fftfreq

    F_list = [None for a in a_list]
    ik1, ik2 = 1, ik_nyq - (1 + nx%2)
    for jj, a in enumerate(a_list):

        F_list[jj] = np.fft.fft(f_list[jj])
        
    # PLot
    ls_list = ['k-', 'k--', 'k-.', 'k:']
    fig, axs = plt.subplots(2,3, figsize=(18,10))

    ax = axs.ravel()[0]
    for f, a, ls in zip(f_list, a_list, ls_list):
        ax.plot(x, f, ls, label=f'a={a}', lw=lw)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('f(x)')

    ax = axs.ravel()[1]
    for F, a, ls in zip(F_list, a_list, ls_list):
        ax.plot(karr[ik1:ik2+1], np.abs(F[ik1:ik2+1]), ls, label=f'a={a}', lw=lw)
    ax.set_xlabel('k [1/m]')
    ax.set_ylabel('F(k)')

    ax = axs.ravel()[2]
    lam = np.zeros_like(karr)
    lam[1:] = 2*np.pi/karr[1:]
    for F, a, ls in zip(F_list, a_list, ls_list):
        ax.plot(lam[ik1:ik2+1], np.abs(F[ik1:ik2+1]), ls, label=f'a={a}', lw=lw)
    ax.set_xlabel('\u03bb [m]') # unicode for lower case lambda is 03bb
    ax.set_ylabel('F(\u03bb)')  # unicode for lower case lambda is 03bb

    ax = axs.ravel()[3]
    for f, a, ls in zip(f_list, a_list, ls_list):
        ax.plot(x, f, ls, label=f'a={a}', lw=lw)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('f(x)')

    # # Testing analytical normalization
    # for w in w_list:
    #     axs.ravel()[0].plot(x,w,'m-')
    #     axs.ravel()[3].plot(x,w,'m-')

    ax = axs.ravel()[4]
    for F, a, ls in zip(F_list, a_list, ls_list):
        ax.plot(karr[ik1:ik2+1], np.abs(F[ik1:ik2+1]), ls, label=f'a={a}', lw=lw)
    ax.set_xlabel('k [1/m]')
    ax.set_ylabel('F(k)')

    ax = axs.ravel()[5]
    for F, a, ls in zip(F_list, a_list, ls_list):
        ax.plot(lam[ik1:ik2+1], np.abs(F[ik1:ik2+1]), ls, label=f'a={a}', lw=lw)
    ax.set_xlabel('\u03bb [m]') # unicode for lower case lambda is 03bb
    ax.set_ylabel('F(\u03bb)')  # unicode for lower case lambda is 03bb

    for ax in axs.ravel()[3:6]:
        ax.set_xscale('log')
        ax.set_yscale('log')

    for ax in axs.ravel():
        ax.legend()

    if density:
        tit = 'Normalized as pdf'
    else:
        tit = 'Normalized on x[1]'

    fig.suptitle(f'Power laws: {tit}')
    fig.tight_layout(pad=2.)

    # Flow properties
    if kpp:
        c = 1e3
        fig2, bx = plt.subplots(1)
        for f, a, ls in zip(f_list, a_list, ls_list):
            bx.plot(x, np.exp(c*f), ls, label=f'a={a}', lw=lw)
        bx.set_xlabel('x [m]')
        bx.set_ylabel('exp(c*f(x))')
        bx.set_title('exp(f(x))')
        bx.legend()


    return fig

#----------------------------------------------
#   b-value estimation ala Aki
#----------------------------------------------

def aki_b_mle(df, x, y, mc, **kwargs):
    """Compute b-values using the Aki (1965) method. 

    Earthquakes are binned on a regular (x,y)-grid
    
    Parameters
    ----------
    df: pd.DataFrame. Earthquakes.
        Source locations in columns key_x, key_y, key_z
        Magnitudes in column key_m 
    y: y-coordinates of a regular grid, shape=ny
    x: x-coordinates of a regular grid, shape=nx
    mc: float. Completeness magnitude
        
    kwargs
    ------
    key_x: str, x-coordinate in df (default = 'x')
    key_y: str, y-coordinate in df (default = 'y')
    key_z: str, z-coordinate in df (default = 'z')
    key_m: str, magnitude in df (default = 'magnitude')
    delm: float. See Aki (1965)
    verbose: int. Print shit?
    
    Returns
    -------
    eq: MapData object with grids
        b_value

        Programmed: KetilH, 17. January 2024
    """
    
    # Get the kwargs
    key_x = kwargs.get('key_x', 'x')
    key_y = kwargs.get('key_y', 'y')
    key_z = kwargs.get('key_z', 'z')
    key_m = kwargs.get('key_z', 'magnitude')
    delm = kwargs.get('delm', 0.0)
    verbose = kwargs.get('verbose', 0)

    if verbose>0:
        print(f'quakes.aki_b_mlh: mc={mc}')

    # Create a cube for binning:
    eq = MapData(x, y, 0)
    nx, ny = eq.nx, eq.ny
    
    # Map to grid
    ix_arr =  np.round((df[key_x].values - eq.x[0])/eq.dx).astype(int)
    iy_arr =  np.round((df[key_y].values - eq.y[0])/eq.dy).astype(int)
    
    ibin_arr = ix_arr + eq.nx*iy_arr
    
    bval = np.nan*np.zeros_like(eq.z[0]) # b-value
    bstd = np.nan*np.zeros_like(eq.z[0]) # variance of b-value
    magnitude = np.abs(np.array(df[key_m]))
    ibin_unique = np.unique(ibin_arr)
    for jj, ibin in enumerate(ibin_unique):
        ix, iy = ibin%nx, ibin//nx
        #print(jj, ibin, iy, ix)
        ind = ibin_arr==ibin
        mags = magnitude[ind]
        nn = mags.shape[0]
        mavg = np.nanmean(mags)
        mdif = mavg - mc + delm 
        if mdif > 0.1:
            bval[iy,ix] = np.log10(np.exp(1))/mdif
            bstd[iy,ix] = bval[iy,ix]/np.sqrt(nn)
        
    eq.grd = [bval, bstd]
    eq.label = ['b_value', 'b_std']
    
    return eq

def aki_b_value(magnitude, mc, mc2=7.0, delm=0.0, **kwargs):
    """Compute b-values using the Aki (1965) method.  
    
    Parameters
    ----------
    magnitude: Array of floats, shape=[n]. Earthquake magnitudes
    mc: float. Completeness magnitude
    mc2: float. Max magnitude to use (default is mc2=7.0)
    delm: float. Corection for finite mc. See Aki (1965) (default is delm=0.0)
        
    kwargs
    ------
    verbose: int. Print shit?
   
    Returns
    -------
    bval: float. b-value
    bstd: float. std of b-value

    Programmed: KetilH, 17. January 2024
    """

    verbose = kwargs.get('verbose', 0)

    ind = (magnitude>=mc) & (magnitude<=mc2)
    magwrk = magnitude[ind]
    n = magwrk.shape[0]

    bval = np.log10(np.exp(1)) / (np.mean(magwrk) - mc + delm)
    bstd = bval/np.sqrt(n)

    # Alternative formula for bstd
    # bstd = 2.3*bstd**2*np.sqrt( np.sum( (magwrk-bval)**2 )/(n*(n-1)) )

    if verbose>0:
        print('quake.aki_b_value:')
        print(f' o n = {n}')
        print(f' o bval = {bval}')
        print(f' o bstd = {bstd}')

    return bval, bstd

def reg_b_value(magnitude, mc, mc2=7.0, dm=0.1, **kwargs):
    """Compute b-values by simple linear regression.  
    
    Parameters
    ----------
    magnitude: Array of floats, shape=[n]. Earthquake magnitudes
    mc: float. Completeness magnitude
    mc2: float. Max magnitude to use (default is mc2=7.0)
    dm: float. Binning increment for histogram (default is dm=0.1) 
        
    kwargs
    ------
    verbose: int. Print shit?
   
    Returns
    -------
    bval: float. b-value
    bstd: float. std of b-value

    Programmed: KetilH, 17. January 2024
    """

    verbose = kwargs.get('verbose', 0)

    ind = (magnitude>=mc) & (magnitude<=mc2)
    magwrk = magnitude[ind]
    n = magwrk.shape[0]

    # Histogram 
    bins_def = np.arange(mc, mc2+dm, dm)
    hist, bins = np.histogram(magwrk, bins=bins_def)
    bins = (bins[0:-1]+bins[1:])/2

    hist_cum = np.cumsum(hist[::-1])[::-1] 
    jnd = hist_cum>0
    hwrk, bwrk = hist_cum[jnd], bins[jnd]
 
    # Linear regression
    bw, a = np.polyfit(bwrk, np.log10(hwrk), 1)
    b = -bw

    if verbose>0:
        print('quake.reg_b_value:')
        print(f' o n = {n}')
        print(f' o bval = {bval}')
        print(f' o a    = {a}')
        
    return b, a

def plot_gutenberg_richter(magnitude, mc=0.0, mc2=7.0, dm=0.1, **kwargs):
    """ Plot Gutenberg Richter trend. 
    
    Parameters
    ----------
    magnitude: Array of floats, shape=[n]. Earthquake magnitudes
    mc: float. Completeness magnitude (default is mc=0.0)
    mc2: float. Max magnitude to use (default is mc2=7.0)
    dm: float. Binning increment for histogram (default is dm=0.1) 

    kwargs
    ------
    b, a: floats. b-value and intercept
    label: str. Legend label for (b,a)
    b2, a2: floats. 2nd set of b-value and intercept
    label2: str. 2nd Legend label
    suptitle: str. suptitle for the figure

    Returns
    -------
    fig: figure object 
     
    Programmed: KetilH, 17. January 2024    
    """

    # Get the kwargs
    b = kwargs.get('b', 0.0)
    a = kwargs.get('a', 0.0)
    label = kwargs.get('label', '')
    b2 = kwargs.get('b2', 0.0)
    a2 = kwargs.get('a2', 0.0)
    label2 = kwargs.get('label2', '')
    suptitle = kwargs.get('suptitle', 'Gutenberg-Richter law') 

    # Make log_hist for all bins (for reference plotting)
    bins_all= np.arange(0, mc2+dm, dm)
    hist_ref, bins_ref = np.histogram(magnitude,bins=bins_all)
    bins_ref = (bins_ref[0:-1]+bins_ref[1:])/2
    hist_cum_ref = np.cumsum(hist_ref[::-1])[::-1]

    ind = bins_ref >= mc
    bins = bins_ref[ind]
    hist_cum = hist_cum_ref[ind]

    # PLot
    fig, axs = plt.subplots(1,2, figsize=(12,6))

    ax = axs.ravel()[0]
    ax.bar(bins_ref, hist_cum_ref, width=0.1)
    ax.bar(bins, hist_cum, width=0.1)
    ax.plot(bins,hist_cum,'g-o', label=f'EQ data (mc={mc})')
    ax.set_xlabel('Magnitude [-]')
    ax.set_ylabel('Count [-]')
    ax.set_title('Linear magnitude distribution')
    if b>0:  ax.plot(bins,10**(a-b*bins),'k-', label=label)
    if b2>0: ax.plot(bins,10**(a2-b2*bins),'r-', label=label2)
    ax.legend()

    ax = axs.ravel()[1]
    ax.bar(bins_ref, hist_cum_ref, width=0.1, log='True')
    ax.bar(bins, hist_cum, width=0.1, log='True')
    ax.plot(bins,hist_cum,'g-o', label=f'EQ data (mc={mc})')
    ax.set_xlabel('Magnitude [-]')
    ax.set_ylabel('Count [-]')
    ax.set_title('Log10 magnitude distribution')
    if b>0:  ax.plot(bins,10**(a-b*bins),'k-', label=label)
    if b2>0: ax.plot(bins,10**(a2-b2*bins),'r-', label=label2)
    ax.legend()

    fig.suptitle(suptitle)
    fig.tight_layout(pad=1.)

    return fig


#----------------------------------------------
#   Earth quake cut-off depth
#----------------------------------------------

def cut_off_depth(df, x, y, **kwargs):
    """Compute earthquake cut-off depths (on a grid defined by x, y) 
    
    Parameters
    ----------
    df: pd.DataFrame. Earthquakes
        Source locations in columns key_x, key_y, key_z 
    x: x-coordinates of a regular grid, shape=nx
    y: y-coordinates of a regular grid, shape=ny
        
    kwargs
    ------
    key_x: str, x-coordinate in df (default = 'x')
    key_y: str, y-coordinate in df (default = 'y')
    key_z: str, z-coordinate in df (default = 'z')
    verbose: int. Print shit?
    quantile: float. Quaantile measure of cut-off (default = 0.95)
    
    Returns
    -------
    eq: MapData object with grids
        depth_mean
        depth_std
        depth_max
        depth_P90
   
    Programmed: KetilH, 16. January 2024
    """

    # Get the kwargs
    key_x = kwargs.get('key_x', 'x')
    key_y = kwargs.get('key_y', 'y')
    key_z = kwargs.get('key_z', 'z')
    verbose = kwargs.get('verbose', 0)
    quantile = kwargs.get('quantile', 0.95)

    # Positive z up or down?
    sgnz = np.mean(df[key_z])/np.abs(np.mean(df[key_z]))
    if verbose>0:
        print(f'quakes.cut_off_depth: sgnz={sgnz}')

    # Create a cube for binning:
    eq = MapData(x, y, 0)
    nx, ny = eq.nx, eq.ny
    
    # Map to grid
    ix_arr =  np.round((df[key_x].values - eq.x[0])/eq.dx).astype(int)
    iy_arr =  np.round((df[key_y].values - eq.y[0])/eq.dy).astype(int)
    
    ibin_arr = ix_arr + eq.nx*iy_arr
    
    depth_mean = np.nan*np.zeros_like(eq.z[0])
    depth_std = np.nan*np.zeros_like(eq.z[0])
    depth_max = np.nan*np.zeros_like(eq.z[0])
    depth_Pq = np.nan*np.zeros_like(eq.z[0])
    depth = np.abs(np.array(df[key_z]))
    ibin_unique = np.unique(ibin_arr)
    for jj, ibin in enumerate(ibin_unique):
        ix, iy = ibin%nx, ibin//nx
        #print(jj, ibin, iy, ix)
        ind = ibin_arr==ibin
        depth_mean[iy,ix] = np.mean(depth[ind])
        depth_std[iy,ix] = np.std(depth[ind])
        depth_max[iy,ix] = np.max(depth[ind])
        depth_Pq[iy,ix] = np.quantile(depth[ind], quantile)
    
    # Swap sign of z?
    depth_mean_std = depth_mean + 1.0*depth_std
    
    eq.grd = [depth_max, depth_Pq, depth_mean_std, depth_mean, depth_std]
    qq = int(100*quantile)
    eq.label = ['max', f'P{qq}', 'mean+std', 'mean', 'std']
    
    return eq

