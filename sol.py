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


def solveEulerEquation(reform, par):
    
    time_start = time.time()
    
    V        = np.nan + np.zeros((par.T, par.numPtsA))
    policyA1 = np.nan + np.zeros((par.T, par.numPtsA))
    policyC  = np.nan + np.zeros((par.T, par.numPtsA))
    policyh  = np.nan + np.zeros((par.T, par.numPtsA))
        
    if reform == 1:
        par.tau_current = par.tau[:, 1]
    else:
        par.tau_current = par.tau[:, 0]
    
    [MinAss,MaxAss] = co.getMinAndMaxAss(reform, par)
    
    # set up asset grid
    Agrid = np.nan + np.zeros((par.T + 1, par.numPtsA))
    
    for t in range(par.T+1):  # 1:par.T + 1 -> 0:par.T
        Agrid[t, :] = np.transpose(np.linspace(MinAss[t], MaxAss[t], par.numPtsA))
    # (Yifan) I combine the 3 loops into 1 here
    # then I vectorised the grid point so it is more efficient
    # start iteration from T to 1:
    
        
    for t in range(par.T-1,-1,-1): # par.T-1 to 0
    
        if t == par.T-1:  # last period
            policyC[par.T-1,:] = np.transpose(Agrid[par.T-1, :]) + par.y_N  # optimal consumption
            policyA1[par.T-1,:] = np.zeros((1,par.numPtsA));  # optimal savings
            policyh[par.T-1,:] = np.zeros((1,par.numPtsA));  # optimal earnings
            V[par.T-1,:]  = co.utility(policyC[par.T-1,:], policyh[par.T-1,:], par)  # value of consumption
            print('Passed period', t+1 ,'of', par.T)
            
        elif (t+1>=par.R) & (t+1<=par.T-1): #retired, t within R to T-1
            Agrid1 = Agrid[t+1, :]
            V1  = V[t+1,:]
    
            # lower and upper bound of asset today (a vector now)
            lbA1 = np.tile(Agrid[t+1,0], (1,par.numPtsA))  # repmat(Agrid[t, 0], [1, par.numPtsA])
            ubA1 = (Agrid[t, :] + par.y_N - par.minCons)*(1+par.r)
            # transpose ubA1 from long to wide
            ubA1 = np.transpose(ubA1.reshape(-1, 1))
            # replace matlab sign(x) with (x > 0)
            signoflowerbound = (co.eulerforzero(lbA1, reform, t, Agrid[t,:], policyC[t+1,:], Agrid1, par) > 0 )
    
            # index vector is true if constraint binds
            index = (signoflowerbound == 1) | (ubA1 - lbA1 < par.tol)
    
            policyA1[t, np.where(index)] = lbA1[index]  #use np.where to index with logical array
    
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
    
            def seekA1(x):  # a multi dimentional function, only update when index is false (constraint not bind)
                return co.eulerforzero(x, reform, t, Agrid[t, np.where(1-index)[1]], policyC[t+1,:], Agrid1,par)
            
            # initial guess np.concatenate((lbA1[0,np.where(1-index), ubA1[0,np.where(1-index))
            X_mat = np.array(  (lbA1[np.where(1-index)], ubA1[np.where(1-index)] ))
            X0 = np.mean( X_mat , axis = 0)
            sol = root(seekA1, X0, tol = par.tol)  # root from scipy library
            policyA1[t,np.where(1-index)[1]] = sol.x
            
            
            #use 'c_h_opt_retire' here
            policyh[t,:] = 0 # retired
            policyC[t,:] = par.y_N + Agrid[t,:] - policyA1[t,:] / (1+par.r)
            # use 'objectivefunc_retire' here
            V[t,:] = co.utility(policyC[t,:], policyh[t,:], par) + 1/(1+par.delta) * pchip_interpolate(Agrid1, V1 , policyA1[t,:])
            print('Passed period', t+1 ,'of', par.T)
            
        else: # for working age population
            
            Agrid1 = Agrid[t+1,:]
            V1 = V[t+1,:]
            # lower and upper bound of asset today (a vector now)
            lbA1 = np.tile(Agrid[t+1,0], (1,par.numPtsA))  # repmat(Agrid[t, 0], [1, par.numPtsA])
            ubA1 = (Agrid[t, :] + par.maxHours*par.w*(1+par.tau_current[t]) + par.y_N - par.minCons)*(1+par.r)
            # transpose ubA1 from long to wide
            ubA1 = np.transpose(ubA1.reshape(-1, 1))
            signoflowerbound = (co.eulerforzero(lbA1.reshape(-1,1)[:,0], reform, t, Agrid[t,:], policyC[t+1,:], Agrid1, par) > 0 )
    
            # index vector is true if constraint binds
            index = (signoflowerbound == 1) | (ubA1 - lbA1 < par.tol)
            
            policyA1[t, np.where(index)] = lbA1[index]
            # else, the rest is interior solution
            def seekA1_work(x):  # a multi dimentional function, only update when index is false (constraint not bind)
                return co.eulerforzero(x, reform, t, Agrid[t, np.where(1-index)[1]], policyC[t+1,:], Agrid1, par)
            
            # initial guess of solution: mean of lower and opper bound
            X_mat = np.array(  (lbA1[np.where(1-index)], ubA1[np.where(1-index)] ))
            X0 = np.mean( X_mat , axis = 0)
            sol = root(seekA1_work, X0, tol = par.tol)  # root from scipy library
            policyA1[t,np.where(1-index)[1]] = sol.x
            
            
            # use 'c_h_opt' here
            [policyC[t,:], policyh[t,:]] = co.c_h_opt(reform, t, Agrid[t, :], policyA1[t,:], par)
           
            V[t,:] = co.utility(policyC[t,:],policyh[t,:],par) + 1/(1+par.delta)*pchip_interpolate(Agrid1, V1, policyA1[t,:])
            print('Passed period', t+1 ,'of', par.T)
            
    elapsed = time.time() - time_start    
    print('Finished, Reform =', reform, 'Elapsed time is', elapsed, 'seconds')    
        
    return policyA1, policyh, policyC, V
    


