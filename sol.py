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
from scipy.interpolate import pchip_interpolate

def solveEulerEquation(reform, par):
    
    # The rest is interior solution
    """ Use the method of endogenous gridpoint to solve the model.
        To improve it further: jit it, then use math.power, not *
    """
    
    time_start = time.time()
    
    V        = np.nan + np.zeros((par.T, par.numPtsA, par.numPtsP))
    policyA1 = np.nan + np.zeros((par.T, par.numPtsA, par.numPtsP))
    policyC  = np.nan + np.zeros((par.T, par.numPtsA, par.numPtsP))
    policyh  = np.nan + np.zeros((par.T, par.numPtsA, par.numPtsP))
    policyp  = np.nan + np.zeros((par.T, par.numPtsA, par.numPtsP))
        
    agrid_box=np.transpose(np.tile(par.agrid,(par.numPtsP,1)))
    pgrid_box=np.tile(par.pgrid,(par.numPtsA, 1))
        
    for t in range(par.T-1,-1,-1): # par.T-1 to 0
    
        if t == par.T-1:  # last period
            
            policyA1[par.T-1,:,:] = np.zeros((1,par.numPtsA, par.numPtsP));  # optimal savings
            policyh[par.T-1,:,:] = np.zeros((1,par.numPtsA, par.numPtsP));  # optimal earnings
            policyC[par.T-1,:,:] = agrid_box*(1+par.r) + par.y_N    # optimal consumption
            V[par.T-1,:,:]  = co.utility(policyC[par.T-1,:,:], policyh[par.T-1,:,:], par)  # value of consumption
            policyp[par.T-1,:,:]=policyp[t,:,:] # pension points do not change
            print('Passed period', t+1 ,'of', par.T)
            
        else:
           
            V1  = V[t+1,:,:]
        

            #How much consumption today? Use Euler equation
            ce=policyC[t+1,:,:]*((1+par.r)/(1+par.delta))**(-1/par.gamma_c)
            
            #How much work? This follows from the FOC
            he=0*(t+1>par.R)+  \
                 (t+1<=par.R)*(par.maxHours-(par.w*(1-par.tau)/par.beta*(ce**(-par.gamma_c)))**(-1/par.gamma_h))
            
            #How much assets? Just use the BC!
            ae=(agrid_box-par.w*he*(1-par.tau)-par.y_N+ce)/(1+par.r)
            
            #Now, back on the main grid(speed can be improved below...)
            policyC[t,:,:]=np.transpose(np.array([np.interp(par.agrid, ae[:,i],ce[:,i]) for i in range(par.numPtsP)]))
            policyh[t,:,:]=0*(t+1>par.R)+ \
                            (t+1<=par.R)*np.transpose(np.array([np.interp(par.agrid, ae[:,i],he[:,i]) for i in range(par.numPtsP)]))
            policyA1[t,:,:]=np.transpose(np.array([np.interp(par.agrid, ae[:,i],par.agrid) for i in range(par.numPtsP)]))
            
            #Pension points simplified
            policyp[t,:,:]=policyp[t+1,:,:]
            

            #Compute the value function, finally
            V[t,:]=co.utility(policyC[t,:,:], policyh[t,:,:], par) + \
                1/(1+par.delta) * np.transpose(np.array([np.interp(policyA1[t,:,i],par.agrid,V1[:,i]) for i in range(par.numPtsP)]))
         
            print('Passed period', t+1 ,'of', par.T)
     
            
    elapsed = time.time() - time_start    
    print('Finished, Reform =', reform, 'Elapsed time is', elapsed, 'seconds')    
        
    return policyA1, policyh, policyC, V, policyp
    


