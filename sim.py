# run simulation for one indvidual without uncertainty

import numpy as np
import co
from numba import njit
from scipy.interpolate import pchip_interpolate
from consav import linear_interp
from interpolation.splines import  eval_linear
#https://www.econforge.org/interpolation.py/

def simNoUncer_interp(p, model, Tstart=-1, Astart=0.0, Pstart=0.0):
#def simNoUncer_interp(reform, policyA1, policyC, policyh, V, p, Tstart=None, Astart=None, Pstart=None):
  
    
    # Get policy functions
    policyA1=model['A']
    policyC=model['c']
    policyh=model['h']
    V=model['V']
    reform=model['model']
    

        
        
    ppath,cpath,apath,hpath,Epath=\
        fast_simulate(Tstart,Astart,Pstart,p.T,p.N,p.agrid,p.pgrid,p.w,p.E_bar_now,
                      model['A'],model['c'],model['h'],model['V'],model['model'])
    
    return {'p':ppath,'c':cpath,'A':apath,'h':hpath,'wh':Epath}
    
#@njit
def fast_simulate(Tstart,Astart,Pstart,T,N,agrid,pgrid,w,E_bar_now,
                  policyA1,policyC,policyh,V,reform):

    # Arguments for output
    cpath = np.nan+ np.zeros((T, N))           # consumption
    hpath = np.nan+ np.zeros((T, N))           # earnings path
    Epath = np.nan+ np.zeros((T, N))           # earnings path
    apath = np.nan+ np.zeros((T + 1,N))        # assets at start of each period, decided 1 period ahead and so includes period T+1   
    ppath = np.nan+ np.zeros((T + 1,N))        # points at start of each period, decided 1 period ahead and so includes period T+1
    
    
    # Modified initial conditions

    Ti=Tstart
    apath[Ti, :] = Astart; 
    ppath[Ti, :] = Pstart; 
        

    
    # Obtain paths using the initial condition and the policy and value functions
    
    for t in range(Ti,T-1):  # loop through time periods for a pticular individual
        for n in range(N):
    
            #Get the discrete choices first...
            Vi=np.zeros(4)-np.inf
            for pp in range(4):
                Vi[pp]=linear_interp.interp_2d(agrid,pgrid,V[t,pp,:,:,0],apath[t,n],ppath[t,n])
                if pp==0:Vi[pp]+0.000000001
                if pp==2:Vi[pp]+0.000000001
            Vi[np.isnan(Vi)]=-1e8
            i=np.argmax(Vi)
            A1p=policyA1[t,i, :,:,0]
            Cp=policyC[t,i, :,:,0]
            hp=policyh[t,i, :,:,0]


            apath[t+1, n] = linear_interp.interp_2d(agrid,pgrid,A1p,apath[t,n],ppath[t,n])#eval_linear(p.mgrid,policyA1[t, :,:],point)
            cpath[t, n] = linear_interp.interp_2d(agrid,pgrid,Cp,apath[t,n],ppath[t,n])#eval_linear(p.mgrid,policyC[t, :,:],point)
            hpath[t, n] = linear_interp.interp_2d(agrid,pgrid,hp,apath[t,n],ppath[t,n])#eval_linear(p.mgrid,policyh[t, :,:],point)
            Epath[t, n] = hpath[t, n]*w[t,0]
            
            if reform == 0:
                ppath[t+1, n]=  ppath[t, n]+w[t,0]*hpath[t, n]/E_bar_now
            else:
                            
                if ((t >=3) & (t <=10)):
                    ppath[t+1, n]=  ppath[t, n]+1.5*w[t,0]*hpath[t, n]/E_bar_now
                else:
                    ppath[t+1, n]=  ppath[t, n]+w[t,0]*hpath[t, n]/E_bar_now
        
     
    return ppath,cpath,apath,hpath,Epath