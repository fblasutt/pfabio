# This co.py file stores all miscellaneous functions needed to perform sol.py
# Fabio Blasutto
import numpy as np
from interpolation.splines import CGrid
from consav.grids import nonlinspace # grids
from quantecon.markov.approximation import rouwenhorst

class setup():
    
    def __init__(self):

        # Economic Environment: set pameters
        self.T = 60           # Number of time periods
        self.R = 45           # Retirement period
        self.r = 0.0        # Interest rate
        self.δ = 0.015    # Discount rate
        self.β = 0.0        # Utility weight on leisure
        self.γc = 1      # risk pameter on consumption!!!Check in upperenvelop if not 1
        self.γh = 1.09    # risk pameter on labour
        self.y_N = 0.2      # Unearned income
        self.E_bar_now = 30000  # Average earnings
        self.q = 1.0            # Fixed cost of pticipation
        self.ρ =0.01       # Dollar value of points
        self.τ = 0.0#.2         # marginal tax rate
        self.ϵ=0.000000001
        
        # Hourly wage
        self.wM=np.zeros(self.T)
        for t in range(self.T):self.wM[t]=0.4+0.0*t
        
        # Hourly wage dispersion
        self.nw=2
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
        self.NA = 50
        self.amin=-10.0
        self.amax=30.0
        self.agrid=np.linspace(self.amin,self.amax,self.NA)#nonlinspace(0.0,450000,self.NA,1.4)#np.linspace(0.0,250000,self.NA)#
        self.startA = 0.0   # Assets people start life with
        
        # Pension points
        self.NP =30
        self.pgrid=nonlinspace(0.0,self.R,self.NP,1.4)#np.linspace(0,self.R,self.NP)## # max one point per year in the law...
        self.startP = 0   # points people start life with
        
        #Multidimensional grid
        self.mgrid=CGrid((self.agrid[0],self.agrid[-1],self.NA),(self.pgrid[0],self.pgrid[-1],self.NP))
   

# Define the utility function
def utility(c,h,p):

    utils_c=np.log(c*0.0+0.000000001)
    where=(c>0.000000001)
    if p.γc == 1:
        utils_c[where] = np.log(c[where])
    else:
        utils_c[where] = c[where]**(1-p.γc)/(1-p.γc)



    if p.γh == 1:
        utils_h = np.log(h)
    else:
        utils_h = (p.maxHours-h)**(1-p.γh) / (1-p.γh)#(h)**(1+1/p.γh) / (1+1/p.γh)

    utils = utils_c + p.β*utils_h 

    return utils

def mcutility(c,p):

    utils_c=np.inf*np.ones(c.shape)
    where=(c>0.000000001)
    if p.γc == 1:
        utils_c[where] = 1/c[where]*p.ρ
    else:
        utils_c[where] = c[where]**(-p.γc)*p.ρ

  

    return utils_c




  
                     
