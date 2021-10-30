# this is the key program to solve the model

# import packages
# Time
import time
import numpy as np
import co # user defined functions
from scipy.optimize import root
#from scipy.interpolate import interp1d
import math
#from scipy.optimize import bisect
from quantecon.optimize.root_finding import brentq 
from numba import jit, njit, prange, int64, float64
#from interpolation import interp
import quantecon as qe
from interpolation.splines import  eval_linear,UCGrid, CGrid, nodes
import numexpr as ne
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

#@njit(fastmath=True)
def solveEulerEquation1(policyA1, policyh, policyC, policyp,agrid_box,reform,r,delta,gamma_c,R,tau,\
                        beta,w,agrid,y_N,gamma_h,T,numPtsA,numPtsP,pgrid,maxHours):
    
    # The rest is interior solution
    """ Use the method of endogenous gridpoint to solve the model.
        To improve it further: jit it, then use math.power, not *
    """
    
    mult=math.pow(((1+r)/(1+delta)),(-1/gamma_c))
    
    policyA1[T-1,:,:] = np.zeros((numPtsA, numPtsP))  # optimal savings
    policyh[T-1,:,:] = np.zeros((numPtsA, numPtsP));  # optimal earnings
    policyC[T-1,:,:] = agrid_box*(1+r) + y_N    # optimal consumption
    
    for t in range(T-2,-1,-1):
                         
        #How much consumption today? Use Euler equation
        ce=policyC[t+1,:,:]*mult
       
        #How much work? This follows from the FOC
        wt=w[t]
        
        if (t+1<=R):
            he=ne.evaluate('maxHours-(wt*(1-tau)/beta*(ce**(-gamma_c)))**(-1/gamma_h)')
        if (t+1>R):he=np.zeros((numPtsA, numPtsP))
                  
        #How much assets? Just use the BC!
        ae=ne.evaluate('(agrid_box-wt*he*(1-tau)-y_N+ce)/(1+r)')
       
        # Now, back on the main grid(speed can be improved below...)
        Pc,policyA1[t,:,:]=solveEulerEquation2(agrid,ae,ce,numPtsP,numPtsA,\
                                               r,wt,y_N,tau,gamma_c,gamma_h,beta,maxHours,t,R)
           
        if (t+1<=R):policyh[t,:,:]=  \
                 ne.evaluate('maxHours-(wt*(1-tau)/beta*(Pc**(-gamma_c)))**(-1/gamma_h)')
                 
        policyC[t,:,:]=Pc
         
        policyp[t,:,:]=policyp[t+1,:,:]
            
@njit(parallel=True)           
def solveEulerEquation2(agrid,ae,ce,numPtsP,numPtsA,r,wt,y_N,tau,gamma_c,gamma_h,beta,maxHours,t,R):
    
    pC,pA=np.empty((2,numPtsA,numPtsP),dtype=np.float64)
    where=np.empty((numPtsA,numPtsP),dtype=np.bool_)
    
    #Unconstrained maximization
    for i in prange(numPtsP):           
        pC[:,i]=np.interp(agrid, ae[:,i],ce[:,i])
        pA[:,i]=np.interp(agrid, ae[:,i],agrid)
     
    
    #Get where constraints are binding
    for j in prange(numPtsA):
        for i in prange(numPtsP): 
            where[j,i]=(pA[j,i]<=agrid[0])
            
    #Where constraints are binding, obtain consumption
    for j in prange(numPtsA):
        tup=(agrid[j],r,wt,y_N,tau,gamma_c,gamma_h,beta,maxHours)
        for i in prange(numPtsP): 
            if where[j,i]:
                if (t+1>R):
                    pC[j,i]=(1+r)*agrid[j]+y_N
                else:
                    pC[j,i]=brentq(minim,agrid[j]*(1+r),agrid[j]*(1+r)+y_N+maxHours*wt*(1-tau),args=tup)[0]
    return pC,pA


@njit
def minim(x,a,r,w,y_N,tau,gamma_c,gamma_h,beta,maxHours):
    h=-np.power(np.power(x,-gamma_c)*w*(1-tau)/beta,-1/gamma_h)+maxHours
    return (1+r)*a+w*h*(1-tau)+y_N-x
    
 
