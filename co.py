# This co.py file stores all miscellaneous functions needed to perform sol.py
# Yifan Lyu, 30th Aug 2021
import numpy as np
#import tools
from scipy.interpolate import pchip_interpolate
from scipy.optimize import root
from interpolation.splines import UCGrid

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
    par.tau = .2         # marginal tax rate

    # precision parameters
    par.tol = 1e-7       # max allowed error
    par.minCons = 1e-5   # min allowed consumption
    par.minHours = 1e-5  # min allowed hours
    par.maxHours = 1880

    # 2. GENERATE GRID
    
    # Assets
    par.numPtsA = 20
    par.agrid=np.linspace(0,250000,par.numPtsA)
    par.startA = 10000   # Assets people start life with
    
    # Pension points
    par.numPtsP = 30
    par.pgrid=np.linspace(0,par.R,par.numPtsP) # max one point per year in the law...
    par.startP = 1   # points people start life with
    
    #Multidimensional grid
    par.mgrid=UCGrid((par.agrid[0],par.agrid[-1],par.numPtsA),(par.pgrid[0],par.pgrid[-1],par.numPtsP))
    return par



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

  
                     
