# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:15:49 2024

@author: KEHOK
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

import obspy.imaging.beachball as bb

#---------------------------------------------------
#  Bayesian inversion for strike, dip, rake
#---------------------------------------------------

def bayes_beach(eq, a1, a2, a3, pp, err, nu=0.25, **kwargs):
    """ Bayesian estimation of beachball parameters (strike, dip, rake).  
    
    The method is based on the work published by Langet et al., NORSAR, (2020).
    
    A simplified version of the source charcteristics is applied, using only the 
    signed of picked P-wave arrivals from a singel earthquake, recorded on a
    number of seismic stations.
    
    Input data must come in the form of a Pandas dataframe, where the following 
    columns have to be present:
        
    Following the varaible naming of Kwiatek:
        'AZI': ray take-off azimuth from north (north=0 deg, east=90 deg etc)
        'TKO': ray take-off angle, from horizontal (horizontal is 0 deg, vertical up is 90.deg?)
    From the SCEC data server:
        'polarity': Picked P-polarity on seismigram ('positive' or 'negative')
    
    Parameters
    ----------
    eq: pd.DatafFrame including at least the following columns:
        azi: ray take-off azimuth from north (north=0 deg, east=90 deg etc)
        pol: ray take off polar angle, from horizontal (horizontal is 0 deg, vertical up is 90.deg)
        polarity: Picked polarity on seismigram ('positive' or 'negative')
    a1: array of float. strike axis for grid search inversion
    a2: array of float. dip    axis for grid search inversion
    a3: array of float. rake   axis for grid search inversion
    pp: dict. Parameters for prior distribution:
        mu_s, mu_d, mu_r: Prior means for strike, dip and rake
        sig_s, sig_d, sig_r: Corresponding prior variances
    err: dict. mean and variance for the data noise
    nu: float. Poisson ratio at hypocenter (default is nu=0.25)
    gam. float. Tensile angle for beyond double-couple sources
    
    kwargs
    ------
    mode: str. analysis only or full inversion (default is mode='inversion')
        Run analysis only if mode='analysis', 
    gam: Tensile angle for beyond-DC sources (default is gam=0.)
    kplot: bool. QC plots? (default is kplot=False)
    verbose: int. Print shit? (default is verbose=0)
    
    Returns
    -------
    dd: dict
    
    Programmed: KetilH 16.April 2024
    """

    mode = kwargs.get('mode', 'inversion')
    gam = kwargs.get('gam', 0.)
    kplot = kwargs.get('kplot', False)
    verbose = kwargs.get('verbose', 0)
    kk = kwargs.get('kk', None) # for labeling plots
    
    if verbose>0:
        print('earthquake.focal.bayes_beach:')
        eid, npick = eq.loc[0,'eid'], eq.shape[0]
        azi_mean, azi_std = eq['AZI'].mean(), eq['AZI'].std() 
        print(f' o kk, eid, npick = {kk}, {eid}, {npick}')
        print(f' o AZI mean, std = {azi_mean:6.2f}, {azi_std:6.2f}')

    # Priors (Gaussian)
    mu1, mu2, mu3 = pp['mu_s'], pp['mu_d'], pp['mu_r']
    sig1, sig2, sig3 = pp['sig_s'], pp['sig_d'], pp['sig_r']
    mu_err = err['mu'] # ALways zero 
    sig_err = err['sig']

    # Grids for grid search      
    da1 = a1[1] - a1[0] 
    da2 = a2[1] - a2[0]
    da3 = a3[1] - a3[0]
    ga1, ga2, ga3 = np.meshgrid(a1, a2, a3, indexing='ij')        
    
    # Make data for sign=+/-1 based on 'polarity' 
    ind_pos = eq['polarity'] == 'positive'
    ind_neg = eq['polarity'] == 'negative'
    eq.loc[ind_pos, 'ipol'] = +1.0
    eq.loc[ind_neg, 'ipol'] = -1.0
    
    # Initialize list for catching forward modeling results
    Gp_list = [None for idd in eq.index]
    Fp_list = [None for idd in eq.index]
        
    # Loop over picks for current event
    arg_mle = np.zeros_like(ga1)
    for jj, idd in enumerate(eq.index):
        
        data = eq.loc[idd, 'ipol']
        # print(jj, data)
        
        AZI, TKO = eq.loc[idd, 'AZI'], eq.loc[idd, 'TKO']
        Gp = rpgen(ga1, ga2, ga3, gam, nu, TKO, AZI, P_only=True)
        Fp = np.sign(Gp)
        # Redundant?
        Gp_list[jj] = Gp
        Fp_list[jj] = Fp
        
        arg_mle = arg_mle + (data-Fp)**2
        
    # Store possible solutions suggested by mle part
    amin = np.min(arg_mle)
    try:
        
        # Best
        ja1, ja2, ja3 = np.where(arg_mle<=amin)
        
        # The index in array a1 (strike)
        fm = pd.DataFrame(
            columns=['ja1', 'ja2', 'ja3', 'strike', 'dip', 'rake'], 
            data=np.array([ja1, ja2, ja3, a1[ja1], a2[ja2], a3[ja3]]).T
            )
        
        # # for plotting only        
        # if len(ja1) > 0:
        #     kkk1 = int(np.mean(a1[ja1])/da1)
        # else:
        #     kkk1 = 0

    except:
        fm = pd.DataFrame()
        
    fm['amin'] = amin
    fm['eid'] = eq.loc[eq.index[0],'eid']
    fm['kk'] = kk
        
    # Run the full Bayesian inversion    
    if mode.lower()[0] == 'i': 
    
        # Compute prior for strike, dip, rake
        # NB: the index order from 3d meshgrid is (n1,n2,n3) -> (s,d,r)
        arg1 = (ga1 - mu1)**2/sig1**2
        arg2 = (ga2 - mu2)**2/sig2**2
        arg3 = (ga3 - mu3)**2/sig3**2
        rn = 1.0/(np.sqrt((2*np.pi)**3)*(sig1*sig2*sig3))
        pdf_pri = rn*np.exp(-0.5*(arg1+arg2+arg3))

        # Compute likelihood and posterior    
        pdf_mle = np.exp(-arg_mle/(2*sig_err))
        pdf_post = pdf_mle*pdf_pri
        rn = 1.0/(np.sum(pdf_post)*da1*da2*da3)  # Normalization
        pdf_post = rn*pdf_post
    
        # Marginals:
        pdf_marg1 = np.sum(pdf_post, axis=(1,2))*da2*da3
        pdf_marg2 = np.sum(pdf_post, axis=(0,2))*da1*da3
        pdf_marg3 = np.sum(pdf_post, axis=(0,1))*da1*da2
    
        mu1_post = np.sum(a1*pdf_marg1)*da1
        mu1_map = a1[np.argmax(pdf_marg1)]
        
        mu2_post = np.sum(a2*pdf_marg2)*da2
        mu2_map = a2[np.argmax(pdf_marg2)]
    
        mu3_post = np.sum(a3*pdf_marg3)*da3
        mu3_map = a3[np.argmax(pdf_marg3)]
        
        new_cols = ['mu_s_map', 'mu_d_map', 'mu_r_map',
                    'mu_s_post', 'mu_d_post', 'mu_r_post']
        new_data = [mu1_map, mu2_map,mu3_map, mu1_post, mu2_post, mu3_post]
        fm[new_cols] = new_data
    
    # Make some QC plots?
    figs = []
    if kplot:
        
        if mode.lower()[0] == 'i':
        
            # QC plot1:
            fig, axs = plt.subplots(1,3 , figsize=(16, 4))
            ax = axs.ravel()[0]
            ax.plot(a1, pdf_marg1)    
            ax.set_xlabel('strike [deg]')
            ax.set_title('marginal pdf strike')
            
            ax = axs.ravel()[1]
            ax.plot(a2, pdf_marg2)    
            ax.set_title('marginal pdf dip')
            ax.set_xlabel('dip [deg]')
            
            ax = axs.ravel()[2]
            ax.plot(a3, pdf_marg3)    
            ax.set_xlabel('rake [deg]')
            ax.set_title('marginal pdf rake')
            
            for ax in axs.ravel():
                ax.set_ylabel('pdf [-]')
    
            eid = fm.loc[0, 'eid']
            npick = eq.shape[0]
            fig.suptitle(f'eid={eid}: Marginal posterior distributions. npick={npick}')
            fig.tight_layout(pad=1.)
            figs.append(fig)
    
        else:

            fig, axs = plt.subplots(1,3, figsize=(8,4))
            ax = axs.ravel()[0]
            ax.hist(fm['strike'], np.linspace(0,180,37))
            ax.set_xlabel('strike [deg]')
            
            ax = axs.ravel()[1]
            ax.hist(fm['dip'], np.linspace(0,90,19))
            ax.set_xlabel('dip [deg]')
            
            ax = axs.ravel()[2]
            ax.hist(fm['rake'], np.linspace(-90,90,37))
            ax.set_xlabel('rake [deg]')
            
            eid, npick, amin = fm.loc[0,'eid'], eq.shape[0], int(fm.loc[0,'amin'])
            fig.suptitle(f'{kk}: eid = {eid} (npick={npick}, amin={amin})')
            fig.tight_layout(pad=1.0)
            figs.append(fig)

            # # QC plot2
            # fig, ax = plt.subplots(1,1, figsize=(16,4))
            # ax.plot(arg_mle.ravel(), '.')
            # ind = np.where(arg_mle.ravel() < 1)
            # ax.plot(ind[0], arg_mle.ravel()[ind[0]], 'ro')
            
            # ax.set_xlabel('flattened array index [-]')
            # ax.set_ylabel('Objective function [-]')
    
            # eid = fm.loc[0, 'eid']
            # fig.suptitle(f'eid={eid}: Where is Waldo?')
            # fig.tight_layout(pad=1.)
            # figs.append(fig)

    # Return variables
    return fm, figs

#---------------------------------------------------
#  FOrward modeling function
#---------------------------------------------------

def rpgen(strike, dip, rake, gamma, sigma, TKO, AZM, **kwargs):
    
    """
    Matlab header
    -------------
    %RPGEN Calculate radiation pattern using shear-tensile source model.
    %   rpgen(strike,dip,rake,gamma,sigma, TKO, AZM) calculates P-wave, S-wave,
    %   SH-wave and SV-wave radiation pattern using shear-tensile source model
    %   presented in [see references 1, 2, 3 for details]. All input angles 
    %   (strike, dip, rake of the fault, tensile angle gamma, takeoff angle 
    %   TKO and azimuth from the source to the observation point AZM) should 
    %   be in degrees. The takeoff angle is measure from bottom. The function 
    %   returns matrices of the same size as input TKO and AZM matrices. 
    %
    %   Input parameters:
    %
    %     strike, dip, rake: fault plane parameters (degrees).
    %     gamma:  tensile angle in degrees (0 degrees for pure shear faulting, 
    %             90 degrees for pure tensile opening).
    %     sigma:  Poisson's ratio.
    %     TKO:    matrix of takeoff angles for which to calculate the correspo-
    %             nding radiation pattern coefficients (degrees, the takeoff 
    %             angles are measured from bottom).
    %     AZM:    matrix of corresponding azimuths (in degrees) for which the 
    %             radiation pattern coefficients should be calculated.
    %
    %   Output parameters:
    %   
    %     Gp, Gs, Gsh, Gsv - P-wave, S-wave, SH-wave, and SV-wave radiation 
    %     pattern coefficients calculated for corresponding takeoff angles 
    %     and azimuths specified in TKO and AZM matrices.
    %
    %   References:
    %
    %     [1] Kwiatek, G. and Y. Ben-Zion (2013). Assessment of P and S wave 
    %         energy radiated from very small shear-tensile seismic events in 
    %         a deep South African mine. J. Geophys. Res. 118, 3630-3641, 
    %         doi: 10.1002/jgrb.50274
    %     [2] Ou, G.-B., 2008, Seismological Studies for Tensile Faults. 
    %         Terrestrial, Atmospheric and Oceanic Sciences 19, 463.
    %     [3] Vavryèuk, V., 2001. Inversion for parameters of tensile 
    %         earthquakes.” J. Geophys. Res. 106 (B8): 16339–16355. 
    %         doi: 10.1029/2001JB000372.
    
    %   Copyright 2012-2013 Grzegorz Kwiatek.
    %   $Revision: 1.3 $  $Date: 2013/09/15 $    
    
    Ported to Python: KetilH, 11.April 2024
    """
    
    P_only = kwargs.get('P_only', True)
        
    sin = np.sin
    cos = np.cos    
    pi = np.pi
    
    strike = strike * pi/180;
    dip = dip * pi / 180;
    rake = rake * pi / 180;
    gamma = gamma * pi / 180;
    TKO = TKO * pi / 180;
    AZM = AZM * pi / 180;

    Gp = cos(TKO)*(cos(TKO)*(sin(gamma)*(2*cos(dip)**2 - (2*sigma)/(2*sigma - 1)) + sin(2*dip)*cos(gamma)*sin(rake)) - cos(AZM)*sin(TKO)*(cos(gamma)*(cos(2*dip)*sin(rake)*sin(strike) + cos(dip)*cos(rake)*cos(strike)) - sin(2*dip)*sin(gamma)*sin(strike)) + sin(AZM)*sin(TKO)*(cos(gamma)*(cos(2*dip)*cos(strike)*sin(rake) - cos(dip)*cos(rake)*sin(strike)) - sin(2*dip)*cos(strike)*sin(gamma))) + sin(AZM)*sin(TKO)*(cos(TKO)*(cos(gamma)*(cos(2*dip)*cos(strike)*sin(rake) - cos(dip)*cos(rake)*sin(strike)) - sin(2*dip)*cos(strike)*sin(gamma)) + cos(AZM)*sin(TKO)*(cos(gamma)*(cos(2*strike)*cos(rake)*sin(dip) + (sin(2*dip)*sin(2*strike)*sin(rake))/2) - sin(2*strike)*sin(dip)**2*sin(gamma)) + sin(AZM)*sin(TKO)*(cos(gamma)*(sin(2*strike)*cos(rake)*sin(dip) - sin(2*dip)*cos(strike)**2*sin(rake)) - sin(gamma)*((2*sigma)/(2*sigma - 1) - 2*cos(strike)**2*sin(dip)**2))) - cos(AZM)*sin(TKO)*(cos(TKO)*(cos(gamma)*(cos(2*dip)*sin(rake)*sin(strike) + cos(dip)*cos(rake)*cos(strike)) - sin(2*dip)*sin(gamma)*sin(strike)) - sin(AZM)*sin(TKO)*(cos(gamma)*(cos(2*strike)*cos(rake)*sin(dip) + (sin(2*dip)*sin(2*strike)*sin(rake))/2) - sin(2*strike)*sin(dip)**2*sin(gamma)) + cos(AZM)*sin(TKO)*(cos(gamma)*(sin(2*dip)*sin(rake)*sin(strike)**2 + sin(2*strike)*cos(rake)*sin(dip)) + sin(gamma)*((2*sigma)/(2*sigma - 1) - 2*sin(dip)**2*sin(strike)**2)));

    if not P_only:
        Gs = ((sin(AZM)*sin(TKO)*(cos(AZM)*cos(TKO)*(cos(gamma)*(cos(2*strike)*cos(rake)*sin(dip) + (sin(2*dip)*sin(2*strike)*sin(rake))/2) - sin(2*strike)*sin(dip)**2*sin(gamma)) - sin(TKO)*(cos(gamma)*(cos(2*dip)*cos(strike)*sin(rake) - cos(dip)*cos(rake)*sin(strike)) - sin(2*dip)*cos(strike)*sin(gamma)) + cos(TKO)*sin(AZM)*(cos(gamma)*(sin(2*strike)*cos(rake)*sin(dip) - sin(2*dip)*cos(strike)**2*sin(rake)) - sin(gamma)*((2*sigma)/(2*sigma - 1) - 2*cos(strike)**2*sin(dip)**2))) - cos(TKO)*(sin(TKO)*(sin(gamma)*(2*cos(dip)**2 - (2*sigma)/(2*sigma - 1)) + sin(2*dip)*cos(gamma)*sin(rake)) + cos(AZM)*cos(TKO)*(cos(gamma)*(cos(2*dip)*sin(rake)*sin(strike) + cos(dip)*cos(rake)*cos(strike)) - sin(2*dip)*sin(gamma)*sin(strike)) - cos(TKO)*sin(AZM)*(cos(gamma)*(cos(2*dip)*cos(strike)*sin(rake) - cos(dip)*cos(rake)*sin(strike)) - sin(2*dip)*cos(strike)*sin(gamma))) + cos(AZM)*sin(TKO)*(sin(TKO)*(cos(gamma)*(cos(2*dip)*sin(rake)*sin(strike) + cos(dip)*cos(rake)*cos(strike)) - sin(2*dip)*sin(gamma)*sin(strike)) + cos(TKO)*sin(AZM)*(cos(gamma)*(cos(2*strike)*cos(rake)*sin(dip) + (sin(2*dip)*sin(2*strike)*sin(rake))/2) - sin(2*strike)*sin(dip)**2*sin(gamma)) - cos(AZM)*cos(TKO)*(cos(gamma)*(sin(2*dip)*sin(rake)*sin(strike)**2 + sin(2*strike)*cos(rake)*sin(dip)) + sin(gamma)*((2*sigma)/(2*sigma - 1) - 2*sin(dip)**2*sin(strike)**2))))**2 + (cos(TKO)*(cos(AZM)*(cos(gamma)*(cos(2*dip)*cos(strike)*sin(rake) - cos(dip)*cos(rake)*sin(strike)) - sin(2*dip)*cos(strike)*sin(gamma)) + sin(AZM)*(cos(gamma)*(cos(2*dip)*sin(rake)*sin(strike) + cos(dip)*cos(rake)*cos(strike)) - sin(2*dip)*sin(gamma)*sin(strike))) - sin(AZM)*sin(TKO)*(sin(AZM)*(cos(gamma)*(cos(2*strike)*cos(rake)*sin(dip) + (sin(2*dip)*sin(2*strike)*sin(rake))/2) - sin(2*strike)*sin(dip)**2*sin(gamma)) - cos(AZM)*(cos(gamma)*(sin(2*strike)*cos(rake)*sin(dip) - sin(2*dip)*cos(strike)**2*sin(rake)) - sin(gamma)*((2*sigma)/(2*sigma - 1) - 2*cos(strike)**2*sin(dip)**2))) + cos(AZM)*sin(TKO)*(sin(AZM)*(cos(gamma)*(sin(2*dip)*sin(rake)*sin(strike)**2 + sin(2*strike)*cos(rake)*sin(dip)) + sin(gamma)*((2*sigma)/(2*sigma - 1) - 2*sin(dip)**2*sin(strike)**2)) + cos(AZM)*(cos(gamma)*(cos(2*strike)*cos(rake)*sin(dip) + (sin(2*dip)*sin(2*strike)*sin(rake))/2) - sin(2*strike)*sin(dip)**2*sin(gamma))))**2)**(1/2);
    
        Gsh = cos(TKO)*(cos(AZM)*(cos(gamma)*(cos(2*dip)*cos(strike)*sin(rake) - cos(dip)*cos(rake)*sin(strike)) - sin(2*dip)*cos(strike)*sin(gamma)) + sin(AZM)*(cos(gamma)*(cos(2*dip)*sin(rake)*sin(strike) + cos(dip)*cos(rake)*cos(strike)) - sin(2*dip)*sin(gamma)*sin(strike))) - sin(AZM)*sin(TKO)*(sin(AZM)*(cos(gamma)*(cos(2*strike)*cos(rake)*sin(dip) + (sin(2*dip)*sin(2*strike)*sin(rake))/2) - sin(2*strike)*sin(dip)**2*sin(gamma)) - cos(AZM)*(cos(gamma)*(sin(2*strike)*cos(rake)*sin(dip) - sin(2*dip)*cos(strike)**2*sin(rake)) - sin(gamma)*((2*sigma)/(2*sigma - 1) - 2*cos(strike)**2*sin(dip)**2))) + cos(AZM)*sin(TKO)*(sin(AZM)*(cos(gamma)*(sin(2*dip)*sin(rake)*sin(strike)**2 + sin(2*strike)*cos(rake)*sin(dip)) + sin(gamma)*((2*sigma)/(2*sigma - 1) - 2*sin(dip)**2*sin(strike)**2)) + cos(AZM)*(cos(gamma)*(cos(2*strike)*cos(rake)*sin(dip) + (sin(2*dip)*sin(2*strike)*sin(rake))/2) - sin(2*strike)*sin(dip)**2*sin(gamma)));
    
        Gsv = sin(AZM)*sin(TKO)*(cos(AZM)*cos(TKO)*(cos(gamma)*(cos(2*strike)*cos(rake)*sin(dip) + (sin(2*dip)*sin(2*strike)*sin(rake))/2) - sin(2*strike)*sin(dip)**2*sin(gamma)) - sin(TKO)*(cos(gamma)*(cos(2*dip)*cos(strike)*sin(rake) - cos(dip)*cos(rake)*sin(strike)) - sin(2*dip)*cos(strike)*sin(gamma)) + cos(TKO)*sin(AZM)*(cos(gamma)*(sin(2*strike)*cos(rake)*sin(dip) - sin(2*dip)*cos(strike)**2*sin(rake)) - sin(gamma)*((2*sigma)/(2*sigma - 1) - 2*cos(strike)**2*sin(dip)**2))) - cos(TKO)*(sin(TKO)*(sin(gamma)*(2*cos(dip)**2 - (2*sigma)/(2*sigma - 1)) + sin(2*dip)*cos(gamma)*sin(rake)) + cos(AZM)*cos(TKO)*(cos(gamma)*(cos(2*dip)*sin(rake)*sin(strike) + cos(dip)*cos(rake)*cos(strike)) - sin(2*dip)*sin(gamma)*sin(strike)) - cos(TKO)*sin(AZM)*(cos(gamma)*(cos(2*dip)*cos(strike)*sin(rake) - cos(dip)*cos(rake)*sin(strike)) - sin(2*dip)*cos(strike)*sin(gamma))) + cos(AZM)*sin(TKO)*(sin(TKO)*(cos(gamma)*(cos(2*dip)*sin(rake)*sin(strike) + cos(dip)*cos(rake)*cos(strike)) - sin(2*dip)*sin(gamma)*sin(strike)) + cos(TKO)*sin(AZM)*(cos(gamma)*(cos(2*strike)*cos(rake)*sin(dip) + (sin(2*dip)*sin(2*strike)*sin(rake))/2) - sin(2*strike)*sin(dip)**2*sin(gamma)) - cos(AZM)*cos(TKO)*(cos(gamma)*(sin(2*dip)*sin(rake)*sin(strike)**2 + sin(2*strike)*cos(rake)*sin(dip)) + sin(gamma)*((2*sigma)/(2*sigma - 1) - 2*sin(dip)**2*sin(strike)**2)));
  
    if P_only:    
        return Gp
       
    else:
        return Gp, Gs, Gsh, Gsv 

#----------------------------------------------------------------------------------
# Simple ray-tracer for expenetial trend velocity model (ala Scalter and Christie)
#----------------------------------------------------------------------------------

# Lambda-fuctions for exponential velocity trend
por_trend = lambda z, phi0, gamma: phi0*np.exp(-gamma*z)
vel_trend = lambda z, v0, v8, gamma: v8 + (v0-v8)*np.exp(-gamma*z)

def raytrace(z, v0, v8, gamma, p):
    """Simple raytracing in 1D exponential-trend model.
    Compute x(z). Nasty integral solved with Mathematica.
    
    Parameters
    ----------
    v0, v8, gamma: floats. Parmeters of the exponential velocuty trend (unit is km/s)
    z: (array of) float. Depths of interest (unit is km)
    p: (array of) float. Slowness (unit is s/km)
    
    Returns
    -------
    ray: dict with fields
        'x': array of float. Ray x-xoordinate (unit is km)
        'z': array of float. Ray z-xoordinate (unit is km)
        'p': array of float. Ray slowness (unit is s/km)
        'ss': array of float. +1 for down-going, -1 for up-going ray element
    
    Programmed: KetilH, 30. April 2024 
    """
    
    a, b, g = v8, v0, gamma
        
    zd, z0 = z, z[0]
    x0 = _ray_helper(z0, a, b, g, p) 
    xa = _ray_helper(zd, a, b, g, p) 
    xd = xa - x0

    # Ray turning?
    eps = 1e-6
    w_list = np.where(np.abs(np.imag(xd))<eps)[0]
    if len(w_list) == xd.shape[0]:
        # Ray does not turn
        x2, z2 = xd, zd
        ss = +1.0*np.ones_like(x2)
    else:
        # Ray does turn
        kz_turn = w_list[-1]
        x_turn = (xd[kz_turn+1] + xd[kz_turn])/2
        z2 = np.concatenate((zd[0:kz_turn+1:1], zd[kz_turn::-1]))
        x2 = np.concatenate((xd[0:kz_turn+1:1], 2*x_turn - xd[kz_turn::-1]))
        s_dn = -1.0*np.ones_like(zd[0:kz_turn+1:1])
        s_up = +1.0*np.ones_like(zd[kz_turn::-1])
        ss = np.concatenate((s_dn, s_up))
    
    return {'x': x2, 'z': z2, 'p': p, 'ss': np.real(ss), 'unit': 'km'}
        
def _ray_helper(z, a, b, g, p):
    """Helper function doing the actual compute for raytrace.
    Computing complex-valued sqrt and arcsin. Clean-up later"""
    
    # Make stuff complex 
    a0 = 1 - (a*p)**2 + 0j
    a1 = 1-(a*p)**2 + p**2*(-(a - b)**2)* \
            np.exp(-2*g*z) + 2*a*p**2*(a - b)* \
            np.exp(-g*z) + 0j
            
    r0 = np.sqrt(a0)
    r1 = np.sqrt(a0*a1)
        
    s1 = (p/g)*a*(np.log(r1 -\
        (a*p)**2 + a*p**2*(a - b)* \
        np.exp(-g*z) + 1) + g*z)/r0 
                
    a2 = p*np.exp(-g*z)*(a*(np.exp(g*z) - 1) + b) + 0j
    s2 = (1/g)*( -np.arcsin(a2))
    
    x = s1 + s2
    
    return x

#-------------------------------------------------
# Testing the 1D exponential-trend raytracer
#-------------------------------------------------

if __name__ == '__main__':

    # Depth range
    zmax, nz = 12.0, 800
    dz = zmax/nz
    zarr = np.linspace(0,zmax,nz)
    
    # Velocity model
    phi0 = 0.387 # 
    v0 = 2.199 # km/s
    v8 = 5.848 # km/s
    gamma = 0.392 # 1/km
    vtrend = vel_trend(zarr, v0, v8, gamma)
    phitrend = por_trend(zarr, phi0, gamma)
        
    # Slowness range
    d2r = np.pi/180.
    nray = 30
    pmax = np.sin(d2r*80.)/vel_trend(0,v0,v8,gamma)
    dp = pmax/nray
    parr = np.linspace(dp,pmax,nray)
    
    # Trace som arrays
    ray_list = [None for p in parr]
    for jj,p in enumerate(parr):
        print(f'jj = {jj:2d}: p = {p:5.3f} s/km')
        ray_list[jj] = raytrace(zarr, v0, v8, gamma, p) 
        
    # Make som plots
    fig, axs = plt.subplots(1, 3, figsize=(14,6), width_ratios=[1,1,3])
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
    for jj in range(nray):
        ax.plot(ray_list[jj]['x'], ray_list[jj]['z'], '-')
    ax.invert_yaxis()
    ax.set_xlabel('offset [km]')
    ax.set_ylabel('depth [km]')
    ax.set_title('Ray-tracing')
    
    klab = int(100*gamma)
    fig.savefig(f'Ray_Trace_Test_{klab}.png')
     
    
    
    






