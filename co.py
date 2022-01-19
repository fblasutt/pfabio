# This co.py file stores all miscellaneous functions needed to perform sol.py
# Fabio Blasutto
import numpy as np
from interpolation.splines import CGrid
from consav.grids import nonlinspace # grids

class setup():
    
    def __init__(self):

        # Economic Environment: set parameters
        self.T = 55           # Number of time periods
        self.R = 35           # Retirement period
        self.r = 0.015        # Interest rate
        self.delta = 0.015    # Discount rate
        self.beta = 10        # Utility weight on leisure
        self.gamma_c = 1      # risk parameter on consumption!!!Check in upperenvelop if not 1
        self.gamma_h = 1.525  # risk parameter on labour
        self.y_N = 48000      # Unearned income
        self.E_bar_now = 30000  # Average earnings
        self.q = 0            # Fixed cost of participation
        self.rho =350#0.3#350       # Dollar value of points
        self.tau = 0.2#.2         # marginal tax rate
        
        # Hourly wage
        self.w=np.zeros(self.T)
        for t in range(self.T):self.w[t]=16#6+t*0.2#16
    
        # precision parameters
        self.tol = 1e-7       # max allowed error
        self.minCons = 1e-5   # min allowed consumption
        self.minHours = 1e-5  # min allowed hours
        self.maxHours = 1880
        
    
        # 2. GENERATE GRID
        
        # Assets
        self.numPtsA = 40
        self.agrid=nonlinspace(0.00001,250000,self.numPtsA,1.4)#np.linspace(0.001,250000,self.numPtsA)
        self.startA = 10000   # Assets people start life with
        
        # Pension points
        self.numPtsP =40
        self.pgrid=nonlinspace(0.0,self.R,self.numPtsP,1.4)#np.linspace(0,self.R,self.numPtsP)## # max one point per year in the law...
        self.startP = 0   # points people start life with
        
        #Multidimensional grid
        self.mgrid=CGrid((self.agrid[0],self.agrid[-1],self.numPtsA),(self.pgrid[0],self.pgrid[-1],self.numPtsP))
   

# Define the utility function
def utility(c,h,par):

    utils_c=-np.inf*np.ones(c.shape)
    where=(c>0)
    if par.gamma_c == 1:
        utils_c[where] = np.log(c[where])
    else:
        utils_c[where] = c[where]**(1-par.gamma_c)/(1-par.gamma_c)

    if par.gamma_h == 1:
        utils_h = np.log(h)
    else:
        utils_h = (par.maxHours - h)**(1 - par.gamma_h) / (1 - par.gamma_h)

    utils = utils_c + par.beta*utils_h - (h==0)*par.q

    return utils

def mcutility(c,par):

    utils_c=np.inf*np.ones(c.shape)
    where=(c>0)
    if par.gamma_c == 1:
        utils_c[where] = 1/c[where]*par.rho
    else:
        utils_c[where] = c[where]**(-par.gamma_c)*par.rho

  

    return utils_c




  
                     
