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
    
    time_start = time.time()
    
    V        = np.nan + np.zeros((par.T, par.numPtsA))
    policyA1 = np.nan + np.zeros((par.T, par.numPtsA))
    policyC  = np.nan + np.zeros((par.T, par.numPtsA))
    policyh  = np.nan + np.zeros((par.T, par.numPtsA))
        

    # (Yifan) I combine the 3 loops into 1 here
    # then I vectorised the grid point so it is more efficient
    # start iteration from T to 1:
    
        
    for t in range(par.T-1,-1,-1): # par.T-1 to 0
    
        if t == par.T-1:  # last period
            policyC[par.T-1,:] = np.transpose(par.agrid)*(1+par.r) + par.y_N  # optimal consumption
            policyA1[par.T-1,:] = np.zeros((1,par.numPtsA));  # optimal savings
            policyh[par.T-1,:] = np.zeros((1,par.numPtsA));  # optimal earnings
            V[par.T-1,:]  = co.utility(policyC[par.T-1,:], policyh[par.T-1,:], par)  # value of consumption
            print('Passed period', t+1 ,'of', par.T)
            
        else:
           
            V1  = V[t+1,:]
    
          
    
            # The rest is interior solution
            """ This is a slower version to find root:
            for i in range(par.numPtsA):  # 1:par.numPtsA
                if not index[0, i]:  # only update non-binding constraint, (index == 0)
                    print(i)
                    def seekA1(x):
                        return co.eulerforzero(x, reform, t, Agrid[t, i], policyC[t+1,:], Agrid1,par)
                    
                    # but how to define constraint? I defined initial guess to be mean of lower and upper bound
                    sol = root(seekA1, np.mean([lbA1[0,i],ubA1[0,i]]), tol = par.tol)
                    policyA1[t,i] = sol.x
                   # Matlab: policyA1(t,i) = fzero(@(x) co.eulerforzero(x, reform, t, Agrid(t, i), policyC(t+1,:), Agrid1,par),[lbA1(i) ubA1(i)], optimset('TolX',par.tol));
            """
    

            #How much consumption today? Use Euler equation
            policyC[t,:]=policyC[t+1,:]*((1+par.r)/(1+par.delta))**(-1/par.gamma_c)
            
            #How much work?
            policyh[t,:]=0*(t+1>par.R)+(t+1<=par.R)*(par.maxHours-(par.w*(1-par.tau)/par.beta*(policyC[t,:]**(-par.gamma_c)))**(-1/par.gamma_h))
            
            #How much assets? Just use the bc
            policyA1[t,:]=(par.agrid-par.w*policyh[t,:]*(1-par.tau)-par.y_N+policyC[t,:])/(1+par.r)
            
            #Compute the value function...
            Vf=co.utility(policyC[t,:], policyh[t,:], par) + 1/(1+par.delta) * pchip_interpolate(par.agrid, V1 , par.agrid)
            
            #...And then interpolate it on gridpoints!
            V[t,:]=pchip_interpolate(policyA1[t,:], Vf, par.agrid)
         
            print('Passed period', t+1 ,'of', par.T)
     
            
    elapsed = time.time() - time_start    
    print('Finished, Reform =', reform, 'Elapsed time is', elapsed, 'seconds')    
        
    return policyA1, policyh, policyC, V
    


