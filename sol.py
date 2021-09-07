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
    
    V        = np.nan + np.zeros((par.T, par.numPtsA))
    policyA1 = np.nan + np.zeros((par.T, par.numPtsA))
    policyC  = np.nan + np.zeros((par.T, par.numPtsA))
    policyh  = np.nan + np.zeros((par.T, par.numPtsA))
        
    
        
    for t in range(par.T-1,-1,-1): # par.T-1 to 0
    
        if t == par.T-1:  # last period
            policyC[par.T-1,:] = np.transpose(par.agrid)*(1+par.r) + par.y_N  # optimal consumption
            policyA1[par.T-1,:] = np.zeros((1,par.numPtsA));  # optimal savings
            policyh[par.T-1,:] = np.zeros((1,par.numPtsA));  # optimal earnings
            V[par.T-1,:]  = co.utility(policyC[par.T-1,:], policyh[par.T-1,:], par)  # value of consumption
            print('Passed period', t+1 ,'of', par.T)
            
        else:
           
            V1  = V[t+1,:]
        

            #How much consumption today? Use Euler equation
            ce=policyC[t+1,:]*((1+par.r)/(1+par.delta))**(-1/par.gamma_c)
            
            #How much work? This follows from the FOC
            he=0*(t+1>par.R)+(t+1<=par.R)*(par.maxHours-(par.w*(1-par.tau)/par.beta*(ce**(-par.gamma_c)))**(-1/par.gamma_h))
            
            #How much assets? Just use the BC!
            ae=(par.agrid-par.w*he*(1-par.tau)-par.y_N+ce)/(1+par.r)
            
            #Now, back on the main grid
            policyC[t,:]=np.interp(par.agrid, ae,ce)
            policyh[t,:]=0*(t+1>par.R)+(t+1<=par.R)*np.interp(par.agrid, ae,he)
            policyA1[t,:]=np.interp(par.agrid, ae,par.agrid)

            #Compute the value function, finally
            V[t,:]=co.utility(policyC[t,:], policyh[t,:], par) + 1/(1+par.delta) * np.interp(policyA1[t,:],par.agrid,V1)
         
            print('Passed period', t+1 ,'of', par.T)
     
            
    elapsed = time.time() - time_start    
    print('Finished, Reform =', reform, 'Elapsed time is', elapsed, 'seconds')    
        
    return policyA1, policyh, policyC, V
    


