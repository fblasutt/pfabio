# This co.py file stores all miscellaneous functions needed to perform sol.py
# Yifan Lyu, 30th Aug 2021
import numpy as np
#import tools
from scipy.interpolate import pchip_interpolate
from scipy.optimize import root

def setup():
    class par: pass

    # Economic Environment: set parameters
    par.T = 55           # Number of time periods
    par.R = 35           # Retirement period
    par.r = 0.015        # Interest rate
    par.delta = 0.015    # Discount rate
    par.beta = 10        # Utility weight on leisure
    par.gamma_c = 1      # risk parameter on consumption
    par.gamma_h = 1.525  # risk parameter on labour
    par.w = 16           # Hourly wage
    par.y_N = 48000      # Unearned income
    par.E_bar_now = 30000  # Average earnings
    par.q = 0            # Fixed cost of participation
    par.rho = 350        # Dollar value of points
    par.startA = 10000   # Assets people start life with
    par.tax = .2         # marginal tax rate

    # precision parameters
    par.tol = 1e-7       # max allowed error
    par.minCons = 1e-5   # min allowed consumption
    par.minHours = 1e-5  # min allowed hours
    par.maxHours = 1880

    # 2. GENERATE GRID
    par.tau = gentax(par)
    # par.interpMethod = 'pchip'  # interpolation methods (no longer used)
    par.numPtsA = 20
    
    return par

def gentax(par):
    # -------------------------------------------------------------------------
    # DESCRIPTION
    # 
    # this function generate tax rate, tau, given all other parameters, par
    # used together with setup()

    # GET TAU GRID
    tau = np.zeros((par.T, 2))
    mu = 0
    for j in range(0, par.T-par.R+1, 1):
        mu = mu + 1 / (1 + par.delta) ** j

    for t in range(1, par.R, 1):  # from 1 to the period before retirement
        if (t >= 3) & (t <= 10):
            tau[t-1, 0] = 1 / (1 + par.delta)**(par.R - t) * par.rho / par.E_bar_now * mu - par.tax
            tau[t-1, 1] = 1 / (1 + par.delta)**(par.R - t) * par.rho / par.E_bar_now * mu * 1.5 - par.tax
        else:
            tau[t-1, :] = 1 / (1 + par.delta)**(par.R - t) * par.rho / par.E_bar_now * mu - par.tax
            #au[t-1, 1] = 1 / (1 + par.delta)**(par.R - t) * par.rho / par.E_bar_now * mu - par.tax

    return tau

def getMinAndMaxAss(reform, par):
# return [BC, maxA]
# -------------------------------------------------------------------------
# DESCRIPTION
# This fuction returns the minimum and maximum on the asset grid in each
# year. The minimum is the natural borrowing constraint. The maximum is how
# much would one have if saving everything, conditional on initial assets.
# ------------------------------------------------------------------------
# Initialise the output matrices

        BC = np.nan + np.zeros((par.T + 1, 1))
        maxA = np.nan + np.zeros((par.T + 1, 1))
    
        if reform ==1:
            tau_current = par.tau[:,1]
        else:
            tau_current = par.tau[:,0]

# ------------------------------------------------------------------------
# Iteratively, calculate the borrowing constraints and maximum on asset
# grid

# Borrowing constraints
        BC[par.T] = 0  #BC(T+1) = 0
        for ixt in range(par.T,par.R-1,-1):  # par.T:-1: par.R
            BC[ixt-1] = BC[ixt] / (1 + par.r) - par.y_N + par.minCons  # - rho*gamma_h*40/E_bar_now


        for ixt in range(par.R-1,0,-1):  # par.R-1:-1: 1
            BC[ixt-1] = BC[ixt] / (1 + par.r) - par.y_N + par.minCons - par.maxHours * par.w * (1 + tau_current[ixt-1])  # - gamma_h

# Maximum Assets
        maxA[0] = par.startA + 1
        for ixt in range(2,par.R,1):  # 2:par.R - 1
# maxA(ixt) = (maxA(ixt - 1)  ) * (1+r)
            maxA[ixt-1] = (maxA[ixt-2] + par.maxHours * par.w * (1 + tau_current[ixt-1]) + par.y_N) * (1 + par.r)

        for ixt in range(par.R,par.T+2,1): #par.R:1: par.T + 1
            maxA[ixt-1] = (maxA[ixt-2] + par.y_N) * (1 + par.r)

# check for errors: return error if interval in assets is empty
        for ixt in range(1,par.T+2,1): #1:1: par.T + 1
             assert (maxA[ixt-1] >= BC[ixt-1]), 'maxA < BC'  # if maxA[ixt-1] < BC[ixt-1]:

        return BC, maxA

def utility(c,h,par):

    if par.gamma_c == 1:
        utils_c = np.log(c)
    else:
        utils_c = c**(1-par.gamma_c)/(1-par.gamma_c)

    if par.gamma_h == 1:
        utils_h = np.log(h)
    else:
        utils_h = (par.maxHours - h)**(1 - par.gamma_h) / (1 - par.gamma_h)

    utils = utils_c + par.beta*utils_h - (h==0)*par.q

    return utils

def eulerforzero(A1, reform, t, A0, policyC, Agrid1, par):
    # -------------------------------------------------------------------------
    # DESCRIPTION
    # This function returns the following quantity:
    # u'(c_t) - b(1+r)u'(c_t+1)
    # This quantity equals 0 if the Euler equation u'(c_t) = b(1+r)u'(c_t+1) is
    # satified at c_t
    # ------------------------------------------------------------------------
    # vectorised eulerforzero
    # input: A1: next period asset vector (or lower bound for next period asset)
    #        A0: this period asset vector
    #        policyC: next period optimal consumption -> use to interpolate
    #        Agird1: next period asset grid           -> use to interpolate
    #        par: vector of parameters

    #c1_interp = griddedInterpolant(Agrid1, policyC, par.interpMethod);
    #np.interp(x, xp, fp, left=None, right=None, period=None)
    #c1_interp = np.interp(A1, Agrid1, policyC) #linear interpolate
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.pchip_interpolate.html#scipy.interpolate.pchip_interpolate
    c1_interp = pchip_interpolate(Agrid1, policyC, A1)
    
    assert (c1_interp > 0).all(), 'c1 is less than 0'

    if t+1 < par.R:  # not retired
        [c , h] = c_h_opt(reform,t,A0,A1,par)
        euler = (c1_interp**par.gamma_c) - (1 + par.r) / (1 + par.delta) * (c**par.gamma_c)
        euler = euler.reshape(-1,1)[:,0] #reshape to dimension 1
        
    else:  # retired
        # h = 0
        c = par.y_N + A0 - A1/(1 + par.r)
        euler = c**(-par.gamma_c) - (1 + par.r)/(1 + par.delta) * (c1_interp**(-par.gamma_c))

    return euler

def solveforh(h,A0,A1,tau_current,par):
        # check if budget constraint binds under max/min labour
  
    c = A0 - (A1/(1+par.r)) + par.w*h*(1+tau_current) + par.y_N
    # a problem here: funcion initial guess may yield negative consumption!
    assert(c >= 0).all(), 'c is less than 0 here'
    
    out = c**(par.gamma_c) - par.w * (1+tau_current)*(par.maxHours-h)**(par.gamma_h)/par.beta
   
    return out


def c_h_opt(reform, t, A0, A1, par):
    # -------------------------------------------------------------------------
    # DESCRIPTION
    # this function is for working individuals only
    # return optimal h and find corresponding c as a
    # function of grid on A
    
    if reform ==1:
        tau_current = par.tau[t,1]
    else:
        tau_current = par.tau[t,0]
            
    # lower bound for minimum working hours   
    lb_temp = (par.minCons-(par.y_N + A0 - A1/(1+par.r)))/(par.w*(1+tau_current))
    lb = np.minimum(np.maximum(lb_temp,1e-5),par.maxHours-1e-10)   
    
    # Compute interior solution
    signoflowerbound = (solveforh(lb, A0, A1, tau_current,par)>0)
    # vector indicating if constraint binds. positve => min h does bind
    
    # if loop
    index = (signoflowerbound == 1) | (par.maxHours-lb<=par.tol)
    #index = index.reshape(-1,1)[:,0]
    
    h = np.nan + np.zeros((1, np.size(index))) # specify size of h
    h[0,np.where(index)] = lb[index]  # need to work enough to have minimum consumption
    # else
    signofupperbound = (solveforh(par.maxHours,A0, A1, tau_current,par)>0)
 
    #debug, sign of lower and upper bound cannnot be greater than 0 simultaneously
    assert(signoflowerbound[np.where(1-index)]*signofupperbound[np.where(1-index)] !=1 ).all(), 'Sign of lower bound and upperbound are the same - no solution to h. Bug likely'
    

    # initial guess
    X_mat = np.array(  (lb[np.where(1-index)], par.maxHours*np.ones(np.sum(1-index)) ))
    #X_mat = np.array(  (lb[np.where(1-index)], lb[np.where(1-index)] ))
    X0 = np.mean( X_mat , axis = 0)

 
    if np.size(X0) > 0:   # only find root if needed
    #solve for interior h      
        
        def seekh(x):  # a multi dimentional function, only update when index is false (constraint not bind)   
                return solveforh(x, A0[np.where(1-index)], A1[np.where(1-index)], tau_current, par) 

        sol = root(seekh, X0, tol = par.tol)  # root from scipy library
        h[0,np.where(1-index)] = sol.x


    c = par.w*(1+tau_current)*h + par.y_N + A0 - A1/(1+par.r)
    return c, h
                     
                     
