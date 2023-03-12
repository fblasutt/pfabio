# This co.py file stores all miscellaneous functions needed to perform sol.py
# Fabio Blasutto
import numpy as np
from interpolation.splines import CGrid
from consav.grids import nonlinspace # grids
from quantecon.markov.approximation import rouwenhorst

class setup():
    
    def __init__(self):

        # Economic Environment: set pameters
        self.T = 55           # Number of time periods
        self.R = 35           # Retirement period
        self.r = 0.015        # Interest rate
        self.δ = 0.015    # Discount rate
        self.β = 13.6        # Utility weight on leisure
        self.γc = 1      # risk pameter on consumption!!!Check in upperenvelop if not 1
        self.γh = 1.571     # risk pameter on labour
        self.y_N = 48000      # Unearned income
        self.E_bar_now = 30000  # Average earnings
        self.q = 0            # Fixed cost of pticipation
        self.ρ =350       # Dollar value of points
        self.τ = 0.2#.2         # marginal tax rate
        
        # Hourly wage
        self.wM=np.zeros(self.T)
        for t in range(self.T):self.wM[t]=16
        
        # Hourly wage dispersion
        self.nw=3
        self.σ=0.2 #dispersion of wages
        self.wv=rouwenhorst(self.nw, 0.0, self.σ,0.0).state_values
        
        #Create actual wages
        self.w=np.zeros((self.T,self.nw))
        for t in range(self.T):self.w[t,:]=np.exp(np.log(self.wM[t])+0*self.wv)
        
    
        # precision pameters
        self.tol = 1e-7       # max allowed error
        self.minCons = 1e-5   # min allowed consumption
        self.minHours = 1e-5  # min allowed hours
        self.maxHours = 1880
        
        # simulations
        self.N = 10000        # agents to simulate
        
    
        # 2. GENERATE GRID
        
        # Assets
        self.numPtsA = 40
        self.agrid=nonlinspace(0.0,250000,self.numPtsA,1.4)#np.linspace(0.0,250000,self.numPtsA)#
        self.startA = 10000   # Assets people start life with
        
        # Pension points
        self.numPtsP =40
        self.pgrid=nonlinspace(0.0,self.R,self.numPtsP,1.4)#np.linspace(0,self.R,self.numPtsP)## # max one point per year in the law...
        self.startP = 0   # points people start life with
        
        #Multidimensional grid
        self.mgrid=CGrid((self.agrid[0],self.agrid[-1],self.numPtsA),(self.pgrid[0],self.pgrid[-1],self.numPtsP))
   

# Define the utility function
def utility(c,h,p):

    utils_c=-np.inf*np.ones(c.shape)
    where=(c>0.000000001)
    if p.γc == 1:
        utils_c[where] = np.log(c[where])
    else:
        utils_c[where] = c[where]**(1-p.γc)/(1-p.γc)

    if p.γh == 1:
        utils_h = np.log(h)
    else:
        utils_h = (p.maxHours-h)**(1-p.γh) / (1-p.γh)#(h)**(1+1/p.γh) / (1+1/p.γh)

    utils = utils_c + p.β*utils_h - (h==0)*p.q

    return utils

def mcutility(c,p):

    utils_c=np.inf*np.ones(c.shape)
    where=(c>0.000000001)
    if p.γc == 1:
        utils_c[where] = 1/c[where]*p.ρ
    else:
        utils_c[where] = c[where]**(-p.γc)*p.ρ

  

    return utils_c




  
                     
