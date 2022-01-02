# run simulation for one indvidual without uncertainty

import numpy as np
import co
from scipy.interpolate import pchip_interpolate
from consav import linear_interp
from interpolation.splines import  eval_linear
#https://www.econforge.org/interpolation.py/

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
    ppath = np.nan+ np.zeros((par.T + 1,1))        # points at start of each period, decided 1 period ahead and so includes period T+1
    
    
    # Initial condition
    apath[0, 0] = par.startA; 
    ppath[0, 0] = par.startP; 
    
    
    # Obtain paths using the initial condition and the policy and value functions
    
    for t in range(par.T-1):  # loop through time periods for a particular individual
    
        point=np.array([apath[t,0],ppath[t,0]]) #where to interpolate
        vpath[t  , 0] = linear_interp.interp_2d(par.agrid,par.pgrid,V[t,:,:],apath[t,0],ppath[t,0])#eval_linear(par.mgrid,V[t,:,:],point)
        apath[t+1, 0] = linear_interp.interp_2d(par.agrid,par.pgrid,policyA1[t, :,:],apath[t,0],ppath[t,0])#eval_linear(par.mgrid,policyA1[t, :,:],point)
        ppath[t+1, 0]=  ppath[t, 0]
        cpath[t, 0] = linear_interp.interp_2d(par.agrid,par.pgrid,policyC[t, :,:],apath[t,0],ppath[t,0])#eval_linear(par.mgrid,policyC[t, :,:],point)
        hpath[t, 0] = linear_interp.interp_2d(par.agrid,par.pgrid,policyh[t, :,:],apath[t,0],ppath[t,0])#eval_linear(par.mgrid,policyh[t, :,:],point)
        Epath[t, 0] = hpath[t, 0]*par.w[t];
        
        Epath_tau[t,0] = hpath[t,0]*par.w[t]
        
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
    
    
    
    EPpath_c[0,0] = EPpath[0,0]+5
    EPpath_behav_c[0,0] = EPpath_behav[0,0]+5
    for t in range(1,par.T):
        EPpath_c[t,0] = EPpath_c[t-1,0] + EPpath[t,0]
        EPpath_behav_c[t,0] = EPpath_behav_c[t-1,0] + EPpath_behav[t,0]
    
    
    EPpath_m_c = EPpath_c*par.rho
    EPpath_behav_m_c = EPpath_behav_c*par.rho
    
    return cpath, apath, hpath, Epath, Epath_tau, vpath, EPpath, EPpath_c, EPpath_m_c, EPpath_behav, EPpath_behav_c, EPpath_behav_m_c

