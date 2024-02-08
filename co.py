# This co.py file stores all miscellaneous functions needed to perform sol.py 
# Fabio Blasutto 
import numpy as np 
from interpolation.splines import CGrid 
from consav.grids import nonlinspace # grids 
from quantecon.markov.approximation import rouwenhorst 
from scipy.stats import norm  
from numba import njit,prange 
import quantecon as qe

class setup(): 
     
    def __init__(self): 
     
        # Economic Environment: set pameters 
        self.T = 56           # Number of time periods 
        self.R = 36           # Retirement period 
        self.r = 0.015        # Interest rate 
        self.δ =0.04538683#0.00983949    # Discount rate 
        self.β =0.63269594  # Utility weight on leisure 
        self.γc = 1.0      # risk pameter on consumption!!!Check in upperenvelop if not 1 
        self.γh = 1.0    # risk pameter on labour 
        self.scale=1000 
        self.scale_e=0.68
        self.E_bar_now = 38800/self.scale*self.scale_e  # Average earnings 
        self.q =0.26729438 #Fixed cost of pticipation 
        self.q_mini =0.26729438*0.37069459#0.18283181*0.30219591 
        self.ρ =350/self.scale      # Dollar value of points 
        self.ϵ=0.000000001 
        self.σ=0.03#0.001#0.00428793          #Size of taste shock 
                     
              
        # Levels of WLS. From GSOEP hrs/week = (10/ 20 / 38.5 ) 
        self.wls=np.array([0.0,10.0,19.25,28.875,38.5])/38.5 
         
        self.nwls=len(self.wls) 
            
        # Hourly wage  
        self.wM=np.zeros((self.T,self.nwls))  
        for t in range(self.T): 
            for i in range(self.nwls): 
                if i ==1:self.wM[t,i]=np.exp(2.440857+.0099643*t -.0002273*t**2)/self.scale*38.5*52*self.scale_e 
                if i ==2:self.wM[t,i]=np.exp(2.440857+.0099643*t -.0002273*t**2)/self.scale*38.5*52*self.scale_e 
                if i ==3:self.wM[t,i]=np.exp(          2.440857+.0099643*t -.0002273*t**2)/self.scale*38.5*52*self.scale_e 
                 
         
        # Taxes 
        self.τ=np.zeros(self.T)  
        for t in range(self.T):self.τ[t]=0.2 
          
        # Hourly wage dispersion  
        self.nw=11 
        self.σw=0.31 #dispersion of wages  
        self.wv,self.Π=addaco_dist(self.σw,self.nw)
         
 
        #Create actual wages  
        self.w=np.zeros((self.T,self.nwls,self.nw))  
        for t in range(self.T): 
            for i in range(self.nwls): 
                if i>=1: 
                    self.w[t,i,:]=np.exp(np.log(self.wM[t,i])+self.wv)  
                elif i<1: 
                    self.w[t,i,:]=self.wM[t,i]  
                     
          
  
        # Earnings of men 
        self.y_N=np.zeros((self.T,self.nw))  
        for t in range(self.R): 
            for i in range(self.nw): 
                self.y_N[t,i]=np.exp(10.14251+.0232318*t-.0005649*t**2+self.wv[i]*0.5)/self.scale*self.scale_e 
                 
        for t in range(self.R,self.T): 
            for i in range(self.nw):             
                 self.y_N[t,i]=self.y_N[self.R-1,i]*0.4/self.scale 
         
     
        # precision pameters 
        self.tol = 1e-7       # max allowed error 
        self.minCons = 1e-5   # min allowed consumption 
        self.minHours = 1e-5  # min allowed hours 
        self.maxHours = 1880 
         
        # simulations 
        self.N = 20000        # agents to simulate 
         
        # 2. GENERATE GRID 
         
        # Assets 
        self.NA = 20 
        self.amin=0.0 
        self.amax=1000000/self.scale 
        self.agrid=nonlinspace(self.amin,self.amax,self.NA,1.4)#np.linspace(self.amin,self.amax,self.NA)#np.linspace(0.0,250000,self.NA)# 
         
         
                  
          
        #Initial assets  
        self.Aμ = 30000.0/self.scale        # Assets people start life with (ave)  
        self.Aσ = 20000.0/self.scale   # Assets people start life with  
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
        self.NP =7 
        self.startP = 5.99 
        self.pgrid=nonlinspace(self.startP,self.R,self.NP,1.4)#np.linspace(0,self.R,self.NP)## # max one point per year in the law... 
          # points people start life with 
         
        #Multidimensional grid 
        self.mgrid=CGrid((self.agrid[0],self.agrid[-1],self.NA),(self.pgrid[0],self.pgrid[-1],self.NP)) 
        
                #Set seed 
        np.random.seed(2) 
         
        #Distribution of types and taste shocks 
        self.tw=np.sort(qe.MarkovChain(self.Π).simulate(self.N))# Type here 
        self.ts=np.random.rand(self.T,self.N) 
         
        ############################################################## 
        #Compute the present value of life-time wealth for each group 
        ############################################################## 
 
from scipy.stats import norm 
 
def addaco_dist(sd_z,npts): 
   
 
    #Probabilities per period 
    ϵ=sd_z*norm.ppf((np.cumsum(np.ones(npts+1))-1)/npts) 
     
 
    X=np.zeros(npts) 
    for i in range(npts): 
        X[i]= sd_z*npts*(norm.pdf(ϵ[i]/sd_z,0.0,1.0)-\
                         norm.pdf(ϵ[i+1]/sd_z,0.0,1.0)) 
    Pi = np.ones((npts,npts))/npts 
 
    return X, Pi 
             
# #Compute the present value of life-time wealth for each group 
# def Ωt(p,ts): 
#     Ω=np.zeros(p.w.shape) 
     
#     #Working life part... 
#     for t in range(ts,p.R): 
#         for ti in range(t,p.R): 
#             Ω[t,:]=Ω[t,:]+((p.y_N+p.maxHours*p.w[ti,:])/(1+p.r)**(ti-t)) 
             
             
#     #Max points and welath afterwards 
#     p.maxpoints=np.zeros(p.nw) 
#     for t in range(ts,p.R):p.maxpoints=p.maxpoints+p.maxHours*p.w[t,:]/p.E_bar_now 
 
#     for t in range(ts,p.T): 
#         for ti in range(max(t,p.R),p.T): 
#             Ω[t,:]=Ω[t,:]+((p.y_N+p.maxpoints*p.ρ)/(1+p.r)**(ti-t))     
 
#     return Ω 
  
# Define the utility function 
def utility(c,h,p): 
 
    utils_c=np.full_like(c,-1e-8,dtype=np.float64) 
    where=(c>0.000000001) 
 
    utils_c[where] = np.log(c[where]) #(c[where])**(1-p.γc)/(1-p.γc) 
     
 
    return utils_c - p.β*(h)**(1+1/p.γh) / (1+1/p.γh) 
 
def mcutility(c,p): 
 
    utils_c=np.inf*np.ones(c.shape) 
    where=(c>0.000000001) 
    if p.γc == 1: 
        utils_c[where] = 1/c[where]*p.ρ 
    else: 
        utils_c[where] = c[where]**(-p.γc)*p.ρ 
 
   
 
    return utils_c 
 
 
 
 
   
                      
