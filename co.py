# This co.py file stores all miscellaneous functions needed to perform sol.py
# Fabio Blasutto
import numpy as np
from interpolation.splines import CGrid
from consav.grids import nonlinspace # grids
from quantecon.markov.approximation import rouwenhorst
from scipy.stats import norm 

class setup():
    
    def __init__(self):

        # Economic Environment: set pameters
        self.T = 55           # Number of time periods
        self.R = 35           # Retirement period
        self.r = 0.015        # Interest rate
        self.δ = 0.015    # Discount rate
        self.β = 0.00005      # Utility weight on leisure
        self.γc = 1      # risk pameter on consumption!!!Check in upperenvelop if not 1
        self.γh = .7    # risk pameter on labour
        self.E_bar_now = 30000/1200  # Average earnings
        self.q = 0.21           # Fixed cost of pticipation
        self.ρ =350/1200       # Dollar value of points
        self.ϵ=0.000000001
        self.σ=0.012          #Size of taste shock
        
        
        # Levels of WLS
        self.wls=np.array([0.0,0.5,1.0])
        self.nwls=len(self.wls)
        
        # Hourly wage 
        self.wM=np.zeros(self.T) 
        for t in range(self.T):self.wM[t]=13+0.05*t 
        
        # Taxes
        self.τ=np.zeros(self.T) 
        for t in range(self.T):self.τ[t]=0.2
         
        # Hourly wage dispersion 
        self.nw=2
        self.σw=0.5 #dispersion of wages 
        self.wv=np.linspace(-self.σw,self.σw,self.nw) 
        self.Π=rouwenhorst(self.nw, 0.0, self.σ,0.0).P 
        self.Π=np.ones(self.Π.shape)/self.nw
        
        # Earnings of men
        self.y_N=np.zeros((self.T,self.nw)) 
        for t in range(self.R):self.y_N[t,:]=48000/1200
        for t in range(self.R,self.T):self.y_N[t,:]=48000/1200*0.4
        
        #Create actual wages 
        self.w=np.zeros((self.T,self.nw)) 
        for t in range(self.T):self.w[t,:]=np.exp(np.log(self.wM[t])+self.wv) 
         
        
    
        # precision pameters
        self.tol = 1e-7       # max allowed error
        self.minCons = 1e-5   # min allowed consumption
        self.minHours = 1e-5  # min allowed hours
        self.maxHours = 1880
        
        # simulations
        self.N = 10000        # agents to simulate
        
        # 2. GENERATE GRID
        
        # Assets
        self.NA = 100
        self.amin=0.0
        self.amax=650000/1200
        self.agrid=np.linspace(self.amin,self.amax,self.NA)#nonlinspace(0.0,450000,self.NA,1.4)#np.linspace(0.0,250000,self.NA)#
        
        
                 
         
        #Initial assets 
        self.Aμ = 0.0        # Assets people start life with (ave) 
        self.Aσ = 10000.0/1200   # Assets people start life with 
        self.startAd   = norm.cdf(self.agrid[1:],self.Aμ,self.Aσ)-\
                        norm.cdf(self.agrid[:-1],self.Aμ,self.Aσ) 
        self.startAd = np.append(self.startAd,0.0) 
        self.startAd[self.startAd<0.00001]=0.0 
        self.startAd=np.cumsum(self.startAd/self.startAd.sum()) 
        self.startApr=np.cumsum(np.ones(self.N)/self.N) 
        self.startAt=np.zeros(self.N,dtype=np.int64) 
        for i in range(self.N):self.startAt[i]=np.argmin(self.startApr[i]>self.startAd) 
        self.startA=self.agrid[self.startAt] 
        
        # Pension points
        self.NP =20
        self.pgrid=nonlinspace(0.0,self.R,self.NP,1.4)#np.linspace(0,self.R,self.NP)## # max one point per year in the law...
        self.startP = 0   # points people start life with
        
        #Multidimensional grid
        self.mgrid=CGrid((self.agrid[0],self.agrid[-1],self.NA),(self.pgrid[0],self.pgrid[-1],self.NP))
        
        ##############################################################
        #Compute the present value of life-time wealth for each group
        ##############################################################

            
            
#Compute the present value of life-time wealth for each group
def Ωt(p,ts):
    Ω=np.zeros(p.w.shape)
    
    #Working life part...
    for t in range(ts,p.R):
        for ti in range(t,p.R):
            Ω[t,:]=Ω[t,:]+((p.y_N+p.maxHours*p.w[ti,:])/(1+p.r)**(ti-t))
            
            
    #Max points and welath afterwards
    p.maxpoints=np.zeros(p.nw)
    for t in range(ts,p.R):p.maxpoints=p.maxpoints+p.maxHours*p.w[t,:]/p.E_bar_now

    for t in range(ts,p.T):
        for ti in range(max(t,p.R),p.T):
            Ω[t,:]=Ω[t,:]+((p.y_N+p.maxpoints*p.ρ)/(1+p.r)**(ti-t))    

    return Ω
 
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
        utils_h = (h)**(1+1/p.γh) / (1+1/p.γh)#(p.maxHours-h)**(1-p.γh) / (1-p.γh)#

    utils = utils_c - p.β*utils_h 

    return utils

def mcutility(c,p):

    utils_c=np.inf*np.ones(c.shape)
    where=(c>0.000000001)
    if p.γc == 1:
        utils_c[where] = 1/c[where]*p.ρ
    else:
        utils_c[where] = c[where]**(-p.γc)*p.ρ

  

    return utils_c




  
                     
