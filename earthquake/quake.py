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

def power_model(x, y, a, mu0_phi, sig0_phi, alfa, beta, n=1000, **kwargs):
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
    n: int. Number of realizations to simulate (default is n=1000)
    alfa, beta: float. Coefficients of log_perm = alfa*phi + beta
    
    kwargs
    ------
    pscale: str. 'log10', 'log' or 'linear' (default is 'log10')
    rho1: float. Nearest neighbor correlation (default is rho1=0.9)
    dist: str. 'norm' or 'lognorm' multivariate distribution (default is 'lognorm')
    density: bool. Normalize as pdf? (default is density=False)
    finite: bool. Finite at x=0? (default is finite=False)
    verbose: int. Print shit if verbose>0
    kplot: bool. PLot or not? (default is kplot=False)

    Returns
    -------
    dd: dict with keys 
        'dist': str. type of distribution,
        'ind_rand': index of 20 random models,
        'por_models': list of 20 random porosity models,
        'gx': array of floats. x coordinates of the models 
        'gy': array of floats. y-coordinates of the models
        'figs': fig_list

    Programmed: KetilH, 26. July 2024.
    """

    pscale = kwargs.get('pscale', 'log10')
    density = kwargs.get('density', False)
    finite  = kwargs.get('finite', False)
    dist = kwargs.get('dist', 'lognormal')
    rho1 = kwargs.get('rho1', 0.90) # Nearest neighbor correlation
    verbose  = kwargs.get('verbose', 0)
    kplot = kwargs.get('kplot', False)
    check_cov = kwargs.get('check_cov', False)

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
        print(f' o check_cov = {check_cov}')

    # Normalization of the power-law correlation:
    dx, dy = x[1]-x[0], y[1]-y[0]
    r1 = np.min([dx, dy])
    r2 = np.sqrt((np.max(x)-np.min(x))**2 + (np.max(y)-np.min(y))**2)
   
    tiny = 1e-6
    if density:
        # Normalize as pdf (analytical)
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
            if np.abs(a) > tiny:
                rho[jj,ii] = rn/rrr[jj,ii]**a
                rho[ii,jj] = rho[jj,ii]
            # else:
            #     rho[ii,jj] = 0.0
            #     rho[jj,ii] = 0.0

    # Create the covariance matrix
    mu  = mu0*np.ones_like(xx)
    sig = rho*sig0**2

    # Check if the covariance matrix is positive definite:
    if check_cov: sig = _pos_def(sig)
 
    ### Run simulations
    rng = np.random.default_rng()
    samps = rng.multivariate_normal(mu, sig, size=n, method='svd')

    # Select some models at random (for plotting and return)
    por_list  = []
    n_rand_mod = np.min([5*4, n])
    ind_rand = np.random.randint(0, n-1, n_rand_mod)
    for ind in ind_rand:
        wrk_por = np.reshape(np.exp(samps[ind,:]), gx.shape)
        por_list.append(wrk_por)

    # Compute permeability (pscale='log10' usually)
    kplot_p = kplot
    perm_list, perm_a_list, perm_h_list, perm_g_list = [], [], [], [] 

    for jj, phi in enumerate(por_list):

        pp = average_perm(phi, alfa, beta, pscale=pscale, kplot=kplot_p)
        kplot_p = False # PLot only the first

        keys = list(pp.keys())[0:4]
        ppls = [perm_list, perm_a_list, perm_h_list, perm_g_list]
        for key, ppl in zip(keys, ppls):
            ppl.append(pp[key])


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

        scl = 1e-3
        title = f'Porosity [-] (sampled from the pdf): a={a}'
        fig_por = plot_sample_models(por_list, x, y, scl=scl, idds=ind_rand, title=title)
        fig_list.append(fig_por)

        title = f'log10 perm [mD]: a={a}'
        fig_perm = plot_sample_models(perm_list, x, y, scl=scl, idds=ind_rand, title=title)
        fig_list.append(fig_perm)

        title = f'log10 perm_geom [mD]: a={a}'
        fig_perm_g = plot_sample_models(perm_g_list, x, y, scl=scl, idds=ind_rand, title=title)
        fig_list.append(fig_perm_g)

    dd = {
        'dist_name': dist,
        # 'samps': samps,
        'ind_rand': ind_rand,
        'por_mods': por_list,
        'perm_mods': perm_list,
        'perm_a_mods': perm_a_list,
        'perm_h_mods': perm_h_list,
        'perm_g_mods': perm_g_list,
        'pscale': pscale,
        'x': x, 
        'y': y,
        'sig': sig,
        'figs': fig_list
        }

    return dd

def plot_sample_models(models, x, y, scl=1e-3, **kwargs):
    """Plot models samples drawn from a distribution"""
    
    # PLot some simulated porosity models
    nmod = len(models)
    xtnt = scl*np.array([x[0], x[-1], y[0], y[-1]])
    nrow = 5 
    ncol = nmod//nrow
    if nmod%nrow > 0: ncol += 1 # Some panels will be empty

    idds = kwargs.get('idds', [jj for jj in range(nmod)])
    title = kwargs.get('title', 'Models (sampled from pdf)')

    # Find vmin and vmax for colorbars
    vmin = np.array(models).min()
    vmax = np.array(models).max()

    fig, axs =  plt.subplots(nrow, ncol, figsize=(13,12))
    for jj in range(nmod):
        ax = axs.ravel()[jj]
        idd = idds[jj]
        # vmin = np.max([0, mu0_phi-3*sig0_phi])
        # vmax = mu0_phi+5*sig0_phi
        im = ax.imshow(models[jj], origin='lower', extent=xtnt, vmin=vmin, vmax=vmax)
        cb = ax.figure.colorbar(im, ax=ax)
        ax.axis('scaled')
        ax.set_title(f'Model {idd}')
        ax.set_xlabel('x [km]')
        ax.set_ylabel('y [km]')
    
    fig.suptitle(title)
    fig.tight_layout(pad=2.0)

    return fig


#--------------------------------------------------------
# Compute average log10 permeability
#--------------------------------------------------------

def average_perm(phi, alfa, beta, **kwargs):
    """Compute aritmetic, harmonic and geometric averages of 
    permeability from a grid of porosity.
    
    Parameters
    ----------
        phi: shape (ny, nx)  array of floats
        alfa, beta: float. Coefficients of log_perm = alfa*phi + beta
    
    kwargs
    ------
        pscale: str. 'log10', 'log' or 'linear' (default is 'log10')
        kplot: bool. QC plot?
    
    Returns
    -------
    dd: dict with keys:
        'perm': array of floats. log10 perm, no averaging
        'perm_a': array of floats. log10 perm, aritmetic average
        'perm_h': array of floats. log10 perm, harmonic average
        'perm_g': array of floats. log10 perm, geometric average
        'pscale:' 'log10', 'log' or 'linear'
        'fig': figure object
         
    Example
    -------

        dd = average_perm(phi, alfa, beta, pscale='log10', kplot=True)

    Programmed: KetilH, 7. Ocotber 2024
    """
    
    pscale = kwargs.get('pscale', 'log10')
    kplot = kwargs.get('kplot', False)
    
    perm_h = np.ones_like(phi)
    perm_a = np.ones_like(phi)
    perm_g = np.ones_like(phi)

    # log10 perm from correlation with porosity    
    perm = np.exp(np.log(10.)*alfa*phi + beta)

    # Harmonic averaging
    ny, nx = phi.shape[0], phi.shape[1]
    for jy in range(0,ny):
        jyf = np.max([jy-1, 0])
        jyl = np.min([jy+1, ny-1])
        # print(jy, jyf, jyl)
        for jx in range(0,nx):
            jxf = np.max([jx-1, 0])
            jxl = np.min([jx+1, nx-1])
            rw = perm[jyf:jyl+1, jxf:jxl+1]
            nn = np.prod(rw.shape)
            perm_a[jy, jx] = (1/nn)*np.sum(rw)
            perm_h[jy, jx] = nn/np.sum(1.0/rw)
            perm_g[jy, jx] = np.prod(rw)**(1/nn)
            
    if pscale.lower() == 'log10':
        perm   = np.log10(perm)
        perm_a = np.log10(perm_a)
        perm_h = np.log10(perm_h)
        perm_g = np.log10(perm_g)

    elif pscale.lower() == 'log':
        perm   = np.log(perm)
        perm_a = np.log(perm_a)
        perm_h = np.log(perm_h)
        perm_g = np.log(perm_g)

    keys = ['perm', 'perm_a', 'perm_h', 'perm_g']
    vals = [perm, perm_a, perm_h, perm_g]
    dd= {key:val for key, val in zip(keys, vals)}

    # Make a QC plot?
    fig = None
    if kplot:
        pmin, pmax = np.min(perm), np.max(perm)
        fig, axs = plt.subplots(2,2, figsize=(12,8))
        
        for jj, (p, t) in enumerate(zip(vals,keys)):
            ax = axs.ravel()[jj]
            im = ax.imshow(p, origin='lower', vmin=pmin, vmax=pmax)
            ax.figure.colorbar(im, ax=ax)
            ax.set_title(f'{pscale} {t}')

        fig.suptitle('Average log10 Permeability')
        fig.tight_layout(pad=1.0)

    dd['fig'] = fig

    return dd
    
#--------------------------------------------------------
#  Power-law analysis by two-point correlation function
#--------------------------------------------------------

def power_correl(df, clu_list, dr=100., **kwargs):
    """ Analyse spatial two-point correlation laws of earth quake clusters.   
    
    Parameters
    ----------
    df: pd.DataFrame. Earthquake data. 
            Hypcenter in columns [key_x, key_y, key_x]
            Clster ID in column key_id
    clu_list: list of cluster IDs to analyze
    dr: Sampling interval of the two-point correlation

    kwargs
    ------
    a_list: list of powers for templeate plot (defulat is a_list=[0.5, 1.0])
    density: bool. Normalised pdf? (default si density=True)
    finite: bool. Finite at r=0? (default is finite=False)
    key_x, key_y, key_z: str. Columns in df for hype center
            Default is [key_x, key_y, key_z] = ['x', 'y', 'depth']
    key_id: Column in df with cluster ID (output from e.g. sklearn.k_means)
            Default is key_id = 'clu_id'
    rmin, rmax: Min and max distance to use in linear regression analysis
            Default is [rmin, rmax] = [100, 1000] meters
    zmin, zmax: Min and max hypocenter depth or z-coord to include in analysis
            Default is [zmin, zmax] = [0, np.inf]
    verbose: int. Print some shit if verbose>0 (default is verbose=0)

    Returns
    -------
    fig: pyplot figure object

    Example
    -------

        key_id = 'id'
        clu_list = list(df[key_id].unique()) # Select all clusters
        fig = power_correl(df, clu_list, dr, a_list=[0.5, 1.0, 1.5],
                           rmin=200, rmax=2000, key_id=key_id) 
    
    Programmed: KetilH, 23. July 2024.
    """

    # Get the kwargs
    key_x = kwargs.get('key_x', 'x')
    key_y = kwargs.get('key_y', 'y')
    key_z = kwargs.get('key_z', 'depth') # depth>0 or z<0 below MSL
    key_id = kwargs.get('key_id', 'clu_id')
    rmin = kwargs.get('rmin',  100.0) # for linreg
    rmax = kwargs.get('rmax', 1000.0) # for linreg
    zmin = kwargs.get('zmin', 0.0)    # min depth or z
    zmax = kwargs.get('zmax', np.inf) # max depth or z
    a_list = kwargs.get('a_list', [.5, 1.0])
    density = kwargs.get('density', True)
    finite  = kwargs.get('finite', False)
    verbose  = kwargs.get('verbose', 0)

    # Range for two-point correlation
    rmin_corr = max([dr, rmin-2*dr]) 
    rmax_corr = rmax + 4*dr
    nrr = int((rmax_corr-rmin_corr)/dr) + 1
    rr = np.linspace(rmin_corr, rmax_corr, nrr)

    # Print some shit?
    if verbose>0:
        print(f'earthquake.quake.power_law:')
        print(f' o key_id = {key_id}')
        print(f' o dr = {dr}')
        print(f' o rmin = {rmin}')
        print(f' o rmax = {rmax}')
        print(f' o zmin = {zmin}')
        print(f' o zmax = {zmax}')

    # Compute two-point correlation function
    corrs = [None for idd in clu_list] 
    rrs   = [None for idd in clu_list]
    rws   = [None for idd in clu_list]
    for jj, idd in enumerate(clu_list):

        ind = (df[key_id] == idd) & (df[key_z]>zmin) & (df[key_z]<zmax) 
        x = np.array(df[ind][key_x])
        y = np.array(df[ind][key_y])
        corrs[jj], rrs[jj], rws[jj] = two_point_correl(x, y, dr)
  
        # Normalize as pdf on the inteval [rmin_corr, rmax_corr]
        jnd = (rrs[jj] >= rmin_corr) & (rrs[jj] <= rmax_corr)
        rrs[jj] = rrs[jj][jnd]
        rws[jj] = rws[jj][jnd]
        corrs[jj] = corrs[jj][jnd]
        rf= 1/np.sum(corrs[jj]*dr)
        corrs[jj] = rf*corrs[jj]

    # Power-law regression
    rarrs = [None for idd in clu_list]
    plaws = [None for idd in clu_list]
    pows  = [None for idd in clu_list]
    for jj, idd in enumerate(clu_list):

        ind = (rrs[jj]>=rmin) & (rrs[jj]<=rmax) & (corrs[jj]>0.)
        rwrk = np.log(rrs[jj][ind])
        cwrk = np.log(corrs[jj][ind])
        
        ok = cwrk.shape[0]>1
        if ok:
            aa, bb = np.polyfit(rwrk, cwrk, 1)
        else:
            aa, bb = 0.0, 0.0

        rarrs[jj]  = rrs[jj][ind]
        plaws[jj] = np.exp(bb)*rrs[jj][ind]**aa
        pows[jj]  = -aa
        print(f'idd, a = {idd}, {pows[jj]}')

        # Normalize (for nicer plotting; we only need the slopes)
        if ok:
            # Normalize linreg estimates
            pwrk = np.exp(bb)*rr**aa
            rf = 1.0/np.sum(pwrk*dr)
            plaws[jj] = rf*plaws[jj]
            corrs[jj] = rf*corrs[jj]

        # # Debugging
        # plt.figure()
        # plt.plot(rwrk, cwrk, 'o')
        # plt.plot(rwrk, bb+aa*rwrk, '-')
        # plt.title(f'aa={aa}')
        # plt.figure()
        # plt.plot(rrs[jj][ind], corrs[jj][ind], 'o')
        # plt.plot(rarrs[jj], plaws[jj], '-')
        # plt.title(f'aa={aa}')

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
    fig = power_play(rr, a_list, density=density, finite=finite, lw=lw)
    axs = fig.axes

    # PLot data in r-domain
    kols = [TABLEAU_COLORS[key] for key in TABLEAU_COLORS.keys()]
    for ax in [axs[0], axs[3]]:

        for jj, idd in enumerate(clu_list):
            ax.plot(rrs[jj], corrs[jj], 'o', color=kols[jj], 
                    label=f'{key_id}={idd} (a={pows[jj]:.2f})') 
            ax.plot(rarrs[jj], plaws[jj], '-', color=kols[jj]) 

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

def two_point_correl(x, y, dr=100.0):
    """Compute two-point correlation function of earthquakes, 
    which is a function of distance r only.
    
    The code is ported from the Matlab function given by Leary et al. (2019).
    
    Parameters
    ----------
    x, y: array of floats. Coordinates of earthquakes
    dr: float. radial sampling (default is dr=100m)
    
    Returns
    -------
    corr_n: Array of float. Correlation function
    rr_n: Array of float. radial distances with non-zero contributions 
    rw_n: Array of int. Number of contributions in each non-empty bin
    
    Programmed: KetilH, 27. February 2024
    """
    
    # Domain size
    xspan = np.max(x)-np.min(x)
    yspan = np.max(y)-np.min(y)
    rspan = np.sqrt((xspan/2)**2 + (yspan/2)**2)
    
    print(f'xspan, yspean, rspan = {xspan}, {yspan}', {rspan})

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
    # ind = (rw>0) & (corr>0.0)
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
        else:
            rn = 1.0/f_list[jj][0]

        f_list[jj] = rn*f_list[jj]

    # Fourier domain
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

    # Aki (1965) does not provide th a-value. Use regression method
    dm = kwargs.get('dm', 0.1)
    bins_def = np.arange(mc, mc2+dm, dm)
    hist, bins = np.histogram(magwrk, bins=bins_def)
    bins = (bins[0:-1]+bins[1:])/2
    hist_cum = np.cumsum(hist[::-1])[::-1] 
    jnd = hist_cum>0
    hwrk, bwrk = hist_cum[jnd], bins[jnd]
    aval = np.sum(np.log10(hwrk)+bval*bwrk)/bwrk.shape[0]
 
    # Alternative formula for bstd
    # bstd = 2.3*bstd**2*np.sqrt( np.sum( (magwrk-bval)**2 )/(n*(n-1)) )

    if verbose>0:
        print('quake.aki_b_value:')
        print(f' o n = {n}')
        print(f' o aval = {aval}')
        print(f' o bval = {bval}')
        print(f' o bstd = {bstd}')

    return bval, aval

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
    ret_all: bool (default=False). Return all data?
   
    Returns
    -------
    bval: float. b-value
    bstd: float. std of b-value

    Programmed: KetilH, 17. January 2024
    """

    verbose = kwargs.get('verbose', 0)
    ret_all = kwargs.get('ret_all', False)

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
        print(f' o a = {a}')
        print(f' o b = {b}')
        
    if ret_all:
        return b, a, bwrk, hwrk
    else:
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
    label = kwargs.get('label', 'Lin.reg')
    b2 = kwargs.get('b2', 0.0)
    a2 = kwargs.get('a2', 0.0)
    label2 = kwargs.get('label2', 'Aki (1965)')
    xlabel = kwargs.get('xlabel' , 'Magnitude [-]')
    suptitle = kwargs.get('suptitle', 'Gutenberg-Richter law') 
 
    # Make log_hist for all bins (for reference plotting)
    mmax = dm*np.round(np.max(magnitude)/dm)
    bins_all= np.arange(0, mmax+dm, dm)
    # bins_all= np.arange(0, mc2+dm, dm)
    hist_ref, bins_ref = np.histogram(magnitude,bins=bins_all)
    bins_ref = (bins_ref[0:-1]+bins_ref[1:])/2
    hist_cum_ref = np.cumsum(hist_ref[::-1])[::-1]

    ind = (bins_ref >= mc) & (bins_ref<=mc2)
    # ind = bins_ref >= mc
    bins = bins_ref[ind]
    hist_cum = hist_cum_ref[ind]
    print(bins)

    # PLot
    fig, axs = plt.subplots(1,2, figsize=(12,6))

    label_b = label + f' b={b:.2f}, a={a:.2f}'
    label2_b = label2 + f' b={b2:.2f}, a={a2:.2f}'

    ax = axs.ravel()[0]
    ax.bar(bins_ref, hist_cum_ref, width=0.1)
    ax.bar(bins, hist_cum, width=0.1)
    ax.plot(bins,hist_cum,'g-o', label=f'EQ data (mc={mc})')
    ax.set_xlabel(f'{xlabel}')
    ax.set_ylabel('Event count [-]')
    ax.set_title('Linear magnitude distribution')
    if b>0:  ax.plot(bins,10**(a-b*bins),'k-', label=label_b)
    if b2>0: ax.plot(bins,10**(a2-b2*bins),'b-', label=label2_b)
    ax.legend()

    ax = axs.ravel()[1]
    ax.bar(bins_ref, hist_cum_ref, width=0.1, log='True')
    ax.bar(bins, hist_cum, width=0.1, log='True')
    ax.plot(bins,hist_cum,'g-o', label=f'EQ data (mc={mc})')
    ax.set_xlabel(f'{xlabel}')
    ax.set_ylabel('Event count [-]')
    ax.set_title('Log10 magnitude distribution')
    if b>0:  ax.plot(bins,10**(a-b*bins),'k-', label=label_b)
    if b2>0: ax.plot(bins,10**(a2-b2*bins),'b-', label=label2_b)
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
    depth_std  = np.nan*np.zeros_like(eq.z[0])
    depth_max  = np.nan*np.zeros_like(eq.z[0])
    depth_Pq   = np.nan*np.zeros_like(eq.z[0])
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


def _pos_def(sig):
    """Check if covariance matrix is positive definite"""

    fig, ax = plt.subplots(1)

    cont, stop, kk = True, False, 0
    while cont:

        eigv = np.linalg.eigvals(sig)    
        pos_def = np.all(eigv > 0)

        ax.plot(np.sort(np.real(eigv))[::-1], '-', label=f'iter={kk}')

        if not pos_def:     
            sig += np.abs(np.min(eigv))*np.eye(*sig.shape)

        kk += 1
        cont = not pos_def and (kk<10) # Run max 10 iterations

        print(f'iter, min, max eigv = {kk, np.min(eigv)}, {np.max(eigv)}')

    ax.set_title('Eigenvalues of the covariance matrix')
    ax.legend()
    fig.tight_layout(pad=1.0)
    fig.savefig('Eigenvalues_of_the_covariance_matrix.png')

    return sig
