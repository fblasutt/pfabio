# run simulation for one indvidual without uncertainty

import numpy as np
from numba import njit
from consav import linear_interp
import quantecon as qe
#https://www.econforge.org/interpolation.py/

def simNoUncer_interp(p, model, Tstart=-1, Astart=0.0, Pstart=0.0, Vstart= -1.0*np.ones((2,2,2,2,2))):
 
    
    #Set seed
    np.random.seed(2)
    
    #Distribution of types
    # Type below
    tw=qe.MarkovChain(p.Π).simulate(p.N)
    
    #Call the simulator
    ppath,cpath,apath,hpath,Epath=\
        fast_simulate(Tstart,Astart,Pstart,Vstart,p.amax,p.T,p.N,p.agrid,p.pgrid,p.w,p.E_bar_now,tw,
                      model['A'],model['c'],model['h'],model['V'],model['model'])
    
    return {'p':ppath,'c':cpath,'A':apath,'h':hpath,'wh':Epath}
    
@njit
def fast_simulate(Tstart,Astart,Pstart,Vstart,amax,T,N,agrid,pgrid,w,E_bar_now,tw,
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
    
    
    #If Vstart if provided, adjust wealth so that utility matches Vstart
    if np.min(Vstart)!=-1.0:apath=adjust_wealth(Vstart, N, V, apath, ppath, amax, agrid, pgrid,Ti,tw)
       
    
    # Obtain paths using the initial condition and the policy and value functions  
    for t in range(Ti,T-1):  # loop through time periods for a pticular individual
        for n in range(N):
            
            #Get the discrete choices first...
            Vi=np.zeros(4)-np.inf
            for pp in range(4):
                Vi[pp]=linear_interp.interp_2d(agrid,pgrid,V[t,pp,:,:,tw[n]],apath[t,n],ppath[t,n])

            Vi[np.isnan(Vi)]=-1e8
            i=np.argmax(Vi)
            A1p=policyA1[t,i, :,:,tw[n]]
            Cp=policyC[t,i, :,:,tw[n]]
            hp=policyh[t,i, :,:,tw[n]]


            apath[t+1, n] = linear_interp.interp_2d(agrid,pgrid,A1p,apath[t,n],ppath[t,n])#eval_linear(p.mgrid,policyA1[t, :,:],point)
            cpath[t, n] = linear_interp.interp_2d(agrid,pgrid,Cp,apath[t,n],ppath[t,n])#eval_linear(p.mgrid,policyC[t, :,:],point)
            hpath[t, n] = linear_interp.interp_2d(agrid,pgrid,hp,apath[t,n],ppath[t,n])#eval_linear(p.mgrid,policyh[t, :,:],point)
            Epath[t, n] = w[t,tw[n]]
            
            if reform == 0:
                ppath[t+1, n]=  ppath[t, n]+w[t,tw[n]]*hpath[t, n]/E_bar_now
            else:
                            
                if ((t >=3) & (t <=10)):
                    ppath[t+1, n]=  ppath[t, n]+1.5*w[t,tw[n]]*hpath[t, n]/E_bar_now
                else:
                    ppath[t+1, n]=  ppath[t, n]+w[t,tw[n]]*hpath[t, n]/E_bar_now
        
     
    return ppath,cpath,apath,hpath,Epath

@njit
def adjust_wealth(Vstart, N, V, apath, ppath, amax, agrid, pgrid, Ti, tw):
    
    step=amax/1000
    for n in range(N):
        
        #Standart Utility
        Vi=np.zeros(4)-np.inf
        for pp in range(4):
            Vi[pp]=linear_interp.interp_2d(agrid,pgrid,V[Ti,pp,:,:,tw[n]],apath[Ti,n],ppath[Ti,n])
        Vi[np.isnan(Vi)]=-1e8
        Vmaxi=np.max(Vi)
        
        #New utility
        Vm=np.zeros(4)-np.inf
        for pp in range(4):
            Vm[pp]=linear_interp.interp_2d(agrid,pgrid,Vstart[Ti,pp,:,:,tw[n]],apath[Ti,n],ppath[Ti,n])
        Vm[np.isnan(Vm)]=-1e8
        Vmaxm=np.max(Vm)
        
        #Compare the two differences in utility

        if Vmaxi>Vmaxm:
            for i in range(100):
                
                addition=step*i
                Vi2=np.zeros(4)-np.inf
                for pp in range(4):
                    Vi2[pp]=linear_interp.interp_2d(agrid,pgrid,V[Ti,pp,:,:,tw[n]],apath[Ti,n]-addition,ppath[Ti,n])
                Vi2[np.isnan(Vi2)]=-1e8                   
                Vmaxi2=np.max(Vi2)
                
                if Vmaxi2<Vmaxm:
                    apath[Ti,n]=apath[Ti,n]-addition
                    break
        else:
            for i in range(100):
                
                addition=step*i
                Vi2=np.zeros(4)-np.inf
                for pp in range(4):
                    Vi2[pp]=linear_interp.interp_2d(agrid,pgrid,V[Ti,pp,:,:,tw[n]],apath[Ti,n]+addition,ppath[Ti,n])
                Vi2[np.isnan(Vi2)]=-1e8                   
                Vmaxi2=np.max(Vi2)
                
                if Vmaxi2>Vmaxm:
                    apath[Ti,n]=apath[Ti,n]+addition
                    break              
        
    return apath