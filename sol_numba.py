# this is the key program to solve the model

# import packages
# Time
import time
import numpy as np
import co # user defined functions
from scipy.optimize import root
#from scipy.interpolate import interp1d
from scipy.interpolate import pchip_interpolate
from numba import jit, njit, prange, int64, float64


from interpolation.splines import  eval_linear,UCGrid, CGrid, nodes
#https://www.econforge.org/interpolation.py/


def solveEulerEquation(reform, par):
    
    time_start = time.time()
    
    r=par.r;delta=par.delta;gamma_c=par.gamma_c;R=par.R;tau=par.tau;beta=par.beta;
    w=np.array(par.w);agrid=par.agrid;y_N=par.y_N;gamma_h=par.gamma_h;T=par.T;numPtsA=par.numPtsA
    numPtsP=par.numPtsP;pgrid=par.pgrid;maxHours=par.maxHours
    
    agrid_box=np.transpose(np.tile(agrid,(numPtsP,1)))
    
    policyA1,policyh,policyC,V,policyp = np.empty((5,T, numPtsA, numPtsP))
        
    
    solveEulerEquation1(policyA1, policyh, policyC, policyp,agrid_box,reform,r,delta,gamma_c,R,tau,beta,w,agrid,y_N,gamma_h,T,numPtsA,numPtsP,pgrid,maxHours)

    elapsed = time.time() - time_start    
    print('Finished, Reform =', reform, 'Elapsed time is', elapsed, 'seconds')   
    
    return policyA1,policyh,policyC,V,policyp

@njit
def solveEulerEquation1(policyA1, policyh, policyC, policyp,agrid_box,reform,r,delta,gamma_c,R,tau,\
                        beta,w,agrid,y_N,gamma_h,T,numPtsA,numPtsP,pgrid,maxHours):
    
    # The rest is interior solution
    """ Use the method of endogenous gridpoint to solve the model.
        To improve it further: jit it, then use math.power, not *
    """

    for t in range(T-1,-1,-1):

        if t == T-1:  # last period
            
            policyA1[t,:,:] = np.zeros((numPtsA, numPtsP));  # optimal savings
            policyh[t,:,:] = np.zeros((numPtsA, numPtsP));  # optimal earnings
            policyC[t,:,:] = agrid_box*(1+r) + y_N    # optimal consumption
            policyp[t,:,:]=policyp[t,:,:] # pension points do not change
                      
        else:
                      
            #How much consumption today? Use Euler equation
            ce=policyC[t+1,:,:]*((1+r)/(1+delta))**(-1/gamma_c)
            
            #How much work? This follows from the FOC
            he=100
            
            #How much assets? Just use the BC!
            ae=(agrid_box-w[t]*he*(1-tau)-y_N+ce)/(1+r)
            
            #Now, back on the main grid(speed can be improved below...)
            for i in range(numPtsP):
                
                #policyC[t,:,i]=np.interp(agrid, ae[:,i],ce[:,i])
                #policyA1[t,:,i]=np.interp(agrid, ae[:,i],agrid)
                
                policyC[t,:,i],policyA1[t,:,i]=solveEulerEquation2(agrid,ae,ce,i)
                
            policyh[t,:,:]=100
            policyp[t,:,:]=policyp[t+1,:,:]
            
@njit           
def solveEulerEquation2(agrid,ae,ce,i):
    
    pC=np.interp(agrid, ae[:,i],ce[:,i])
    pA=np.interp(agrid, ae[:,i],agrid)
    return pC,pA
    


