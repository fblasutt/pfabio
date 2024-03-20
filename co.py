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
        self.T = 56          # Number of time periods 
        self.R = 36           # Retirement period 
        self.r = 0.015        # Interest rate 
        self.δ =0.00454988#0.00983949    # Discount rate 
        self.β =  0.0 # Utility weight on leisure 
        self.ζ = 0.0 #time cost of children under age 11
        self.γc = 1.0      # risk pameter on consumption!!!Check in upperenvelop if not 1 
        self.γh = 1.0    # risk pameter on labour 
        self.scale=1000 
        self.scale_e=0.92365653
        
        #https://www.gesetze-im-internet.de/sgb_6/ appendix 1 54256
        #exchange rate 1.9569471624266144
        self.E_bar_now = 27740.65230618203/self.scale*1.4  # Average earnings 
        
        
        # Levels of WLS. From GSOEP hrs/week = (10/ 20 / 38.5 ) 
        self.wls=np.array([0.0,10.0,20.0,38.5])/38.5 
        self.wls_point = np.array([0.0,0.2,1.0,1.0]) #smallest position on 
         
        self.nwls=len(self.wls) 
        
        
        self.q =np.array([0.0,0.01095495, 0.19125356, 0.63027815])  #Fixed cost of pticipation - mean
        self.σq = 0.41788212 #Fixed cost of pticipation -sd 
        self.nq = 2
        
        self.q_gridt,self.Πq=addaco_dist(self.σq,self.nq) 
        
        self.q_grid=np.ones((self.nq,self.nwls))
        for iq in range(self.nq):
            for il in range(self.nwls):
                self.q_grid[iq,il] = self.q[il]
                if il==0: self.q_grid[iq,il] = self.q_gridt[iq]
        
        
        self.q_mini =0.0#0.21689193*0.42137996 #0.18283181*0.30219591 
        self.ρ =350/self.scale      # Dollar value of points 
        self.ϵ=0.000000001 
        self.σ=0.0001#0.00428793          #Size of taste shock 
        self.Pmax = 1 #threshold for pension points reform
        self.add_points=1.5 #point multiplicator during reform
        #self.apoints=np.array([0.0,0.0,1.0,1.0])
        
                       
              

            
        # Hourly wage  
        self.wM=np.zeros((self.T,self.nwls))  
        for t in range(self.T): 
            for i in range(self.nwls): 
                self.wM[t,i]=np.exp(-0.1806434+.0297458*t -.0005628 *t**2)/self.scale
                
         
        # Taxes 
        self.τ=np.zeros(self.T)  
        for t in range(self.T):self.τ[t]=0.2
        

  
        
        
        
        # Hourly wage dispersion   
        self.nw=10 
        self.σw=0.31 #dispersion of wages   
        self.wv2,self.Π=addaco_dist(self.σw,self.nw) 
         
        self.wv=np.array([ 
   7.631935, 
   8.090471, 
   8.286575 , 
   8.44488, 
   8.580534 , 
   8.709507, 
   8.827328, 
   8.96867, 
   9.145868, 
   9.563336]) 
          
  
        #Create actual wages   
        self.w=np.zeros((self.T,self.nwls,self.nw))   
        for t in range(self.T):  
            for i in range(self.nwls):  
                if i>=1:  
                    self.w[t,i,:]=np.exp(self.wM[t,i]+self.wv)/self.scale/13.96*38.5
                # elif i<1:  
                     
          
  
        self.wv_men=np.array([16735.71,16920.84,19447.71,18881.82,22105.11,18759.2,19781.31,21205.22,24253.39,26751.28]) 
        self.y_N=np.zeros((self.T,self.nw))   
        for t in range(self.R):  
            for i in range(self.nw):  
                self.y_N[t,i]=(-2930.40118+501.4029*(t)-11.82488*(t)**2+self.wv_men[i])/self.scale
                  
        for t in range(self.R,self.T):  
            for i in range(self.nw):              
                 self.y_N[t,i]=self.y_N[self.R-1,i]*0.45
         
        

        # simulations 
        self.N = 20000        # agents to simulate 
         
        # 2. GENERATE GRID 
         
        # Assets 
        self.NA = 15
        self.amin=0.0
        self.amax=1000000/self.scale 
        self.agrid=nonlinspace(self.amin,self.amax,self.NA,1.4)#np.linspace(self.amin,self.amax,self.NA)#np.linspace(0.0,250000,self.NA)# 
         
         
                  
          
        #Initial assets  
        self.startA=np.zeros(self.N)  
        assets=np.array([5877.601,21127.74,10671.82,7991.483,21614.71,15574.05,15981.4,24145.36,34662.95,32189.31]) 
        for i in range(self.N): 
            index=int(i/self.N*10) 
            self.startA[i]=assets[index]/self.scale 
         
        # Pension points 
        self.NP =5
        self.startPd = np.array([2.509712,2.732802,3.061959,2.951443,3.380279,3.498868,3.528733,4.050392,4.264604,4.733748])
        
        self.startP=np.zeros(self.N) 
        for i in range(self.N): 
            index=int(i/self.N*10) 
            self.startP[i]=self.startPd[index]
        
        self.pgrid=nonlinspace(self.startP.min(),self.R*2,self.NP,1.4)#np.linspace(0,self.R,self.NP)## # max one point per year in the law... 
          # points people start life with 
         
        #Multidimensional grid 
        self.mgrid=CGrid((self.agrid[0],self.agrid[-1],self.NA),(self.pgrid[0],self.pgrid[-1],self.NP)) 
        
                #Set seed 
        np.random.seed(2) 
         
        #Distribution of types and taste shocks 
        self.tw=np.sort(qe.MarkovChain(self.Π).simulate(self.N))# Type here 
        self.ts=np.random.rand(self.T,self.N) 
        
        self.q_sim = np.zeros((self.T,self.N),dtype=np.int32)  
        
        j=0
        for i in range(self.N):
            self.q_sim[:,i] = j
            j = j+1 if j<self.nq-1 else 0
         
        ############################################################## 
        #Compute the present value of life-time wealth for each group 
        ############################################################## 
 
from scipy.stats import norm 
 

def hours(params,data,beg,end):
    
    D=data['h'][beg:end,:]
    return np.mean(D==1)*10.0+np.mean(D==2)*20.0+np.mean(D==3)*38.5
    
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
 
 
 
 
   
                      
