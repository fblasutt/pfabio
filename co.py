# This co.py file stores all miscellaneous functions needed to perform sol.py
# Fabio Blasutto
import numpy as np
from interpolation.splines import CGrid

class setup():
    
    def __init__(self):

        # Economic Environment: set parameters
        self.T = 55           # Number of time periods
        self.R = 35           # Retirement period
        self.r = 0.015        # Interest rate
        self.delta = 0.015    # Discount rate
        self.beta = 10        # Utility weight on leisure
        self.gamma_c = 1      # risk parameter on consumption
        self.gamma_h = 1.525  # risk parameter on labour
        self.y_N = 0#48000      # Unearned income
        self.E_bar_now = 30000  # Average earnings
        self.q = 0            # Fixed cost of participation
        self.rho =0#350        # Dollar value of points
        self.tau = 0.0#.2         # marginal tax rate
        
        # Hourly wage
        self.w=np.zeros(self.T)
        for t in range(self.T):self.w[t]=6+t*0.2#16
    
        # precision parameters
        self.tol = 1e-7       # max allowed error
        self.minCons = 1e-5   # min allowed consumption
        self.minHours = 1e-5  # min allowed hours
        self.maxHours = 1880
        
    
        # 2. GENERATE GRID
        
        # Assets
        self.numPtsA = 200
        self.agrid=np.linspace(0.001,250000,self.numPtsA)
        self.startA = 0#10000   # Assets people start life with
        
        # Pension points
        self.numPtsP = 300
        self.pgrid=np.linspace(0,self.R,self.numPtsP) # max one point per year in the law...
        self.startP = 1   # points people start life with
        
        #Multidimensional grid
        self.mgrid=CGrid((self.agrid[0],self.agrid[-1],self.numPtsA),(self.pgrid[0],self.pgrid[-1],self.numPtsP))
   

# Define the utility function
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




  
                     
