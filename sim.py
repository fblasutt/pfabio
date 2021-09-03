# run simulation for one indvidual without uncertainty

import numpy as np
import co
from scipy.interpolate import pchip_interpolate

def simNoUncer_interp(reform, policyA1, policyC, policyh, V, par):
    # Arguments for output
    cpath = np.nan+ np.zeros((par.T, 1))           # consumption
    hpath = np.nan+ np.zeros((par.T, 1))           # earnings path
    Epath = np.nan+ np.zeros((par.T, 1))           # earnings path
    Epath_tau = np.nan+ np.zeros((par.T, 1))       # earnings path
    EPpath = np.nan+ np.zeros((par.T, 1))          # earning points path
    EPpath_c = np.nan+ np.zeros((par.T, 1))        # cumulative earning points path
    EPpath_behav = np.nan+ np.zeros((par.T, 1))    # earning points path
    EPpath_behav_c = np.nan+ np.zeros((par.T, 1))  # cumulative earning points path
    vpath = np.nan+ np.zeros((par.T, 1))           # value
    apath = np.nan+ np.zeros((par.T + 1,1))        # assets at start of each period, decided 1 period ahead and so includes period T+1
    
    
    # GET ASSET GRID
    [MinAss, MaxAss] = co.getMinAndMaxAss(reform, par)
    
    Agrid = np.nan+ np.zeros((par.T+1, par.numPtsA))
    for t in range(par.T+1):  # 1:par.T + 1 -> 0:par.T
         Agrid[t, :] = np.transpose(np.linspace(MinAss[t], MaxAss[t], par.numPtsA))
    
    apath[0, 0] = par.startA; 
    
    
    # Obtain paths using the initial condition and the policy and value functions
    
    for t in range(par.T):  # loop through time periods for a particular individual
    
        vpath[t  , 0] = pchip_interpolate(Agrid[t,:], V[t, :], apath[t,0])
        apath[t+1, 0] = pchip_interpolate(Agrid[t,:], policyA1[t,:], apath[t,0])
        
        cpath[t, 0] = pchip_interpolate(Agrid[t,:], policyC[t,:], apath[t,0])
        hpath[t, 0] = pchip_interpolate(Agrid[t,:], policyh[t,:], apath[t,0])
        Epath[t, 0] = hpath[t, 0]*par.w;
        
        Epath_tau[t,0] = hpath[t,0]*par.w*(1+par.tau[t,reform+0])
        
        if reform == 0:
            EPpath[t, 0] = Epath[t,0]/par.E_bar_now
            EPpath_behav[t,0] = Epath[t,0]/par.E_bar_now
        else:
                if t+1 >=3 & t+1 <=10:
                    EPpath[t, 0] = Epath[t,0]/par.E_bar_now*1.5
                    EPpath_behav[t,0] = Epath[t,0]/par.E_bar_now
                else:
                    EPpath[t, 0] = Epath[t,0]/par.E_bar_now
                    EPpath_behav[t,0] = Epath[t,0]/par.E_bar_now
    
        
        # tax_rev = -tau(t,1)*Epath(t,1);
    
    EPpath_c[0,0] = EPpath[0,0]+5
    EPpath_behav_c[0,0] = EPpath_behav[0,0]+5
    for t in range(1,par.T):
        EPpath_c[t,0] = EPpath_c[t-1,0] + EPpath[t,0]
        EPpath_behav_c[t,0] = EPpath_behav_c[t-1,0] + EPpath_behav[t,0]
    
    
    EPpath_m_c = EPpath_c*par.rho
    EPpath_behav_m_c = EPpath_behav_c*par.rho
    
    return cpath, apath, hpath, Epath, Epath_tau, vpath, EPpath, EPpath_c, EPpath_m_c, EPpath_behav, EPpath_behav_c, EPpath_behav_m_c

