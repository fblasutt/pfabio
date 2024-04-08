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
        self.r = 0.03        # Interest rate 
        self.δ =  0.01041525#0.00983949    # Discount rate 
        self.β =  0.0 # Utility weight on leisure 
        self.ζ = 0.0 #time cost of children under age 11
        self.γc = 1.0      # risk pameter on consumption!!!Check in upperenvelop if not 1 
        self.γh = 1.0    # risk pameter on labour 
        self.scale=1000 

        
        #https://www.gesetze-im-internet.de/sgb_6/ appendix 1 54256
        #exchange rate 1.9569471624266144
        self.E_bar_now = 27740.65230618203/self.scale  # Average earnings 
        
            
        # Levels of WLS. From GSOEP hrs/week = (10/ 20 / 38.5 ) 
        self.wls=np.array([0.0,9.36, 21.17, 36.31])/36.31#np.array([0.0,10.0,20.0,38.5])/38.5 
        self.wls_point = np.array([0.0,0.0,1.0,1.0]) #smallest position on 
         
        self.nwls=len(self.wls) 
        
         
        self.q =np.array([0.0,0.35730499*(-0.33149012),0.35730499*( 0.17445595),0.35730499])  #Fixed cost of pticipation - mean

        self.σq = 1.43579454  #Fixed cost of pticipation -sd 
        self.ρq =0.0690892#0.00195224
        self.nq = 4
        

        
        self.q_mini =0.0#0.21689193*0.42137996 #0.18283181*0.30219591 
        self.ρ =350/self.scale      # Dollar value of points 
        self.ϵ=0.000000001 
        self.σ=0.025#0.00428793          #Size of taste shock 
        self.Pmax = 1 #threshold for pension points reform
        self.add_points=1.5 #point multiplicator during reform
        self.points_base=1.0
        #self.apoints=np.array([0.0,0.0,1.0,1.0])
        
                       
              

            
        # Hourly wage  
        self.wM=np.zeros((self.T,self.nwls))  
        for t in range(self.T): 
            for i in range(self.nwls): 
                self.wM[t,i]=-0.1806434+.0297458*t -.0005628 *t**2
                
         
        # Taxes 
        self.τ=np.zeros(self.T)  
        for t in range(self.T):self.τ[t]=0.2
        
      
        
        # Hourly wage dispersion   
        self.nw=10 
        self.σw=0.31 #dispersion of wages   
        self.wv2,self.Π=addaco_dist(self.σw,0.0,self.nw) 
         
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
          
  
        self.q_grid=np.zeros((self.nq,self.nwls,10))
        self.q_grid_π=np.zeros((self.nq,10))
        self.q_gridt,_=addaco_dist(self.σq,0.0,self.nq)

        for il in range(self.nwls):
            for iw in range(10):
                for iq in range(self.nq):self.q_grid[iq,il,iw] += self.q[il]
                if il<1: 
                    
                    mean = self.σq/.5376561*self.ρq*(np.log(self.wv[iw])-2.15279653495785)
                    sd = (1-self.ρq**2)**0.5*self.σq
                    
                    q_gridt,π = addaco_dist(sd,mean,self.nq) 
                    self.q_grid_π[:,iw] =  π[0]
                    for iq in range(self.nq):
                    
                        self.q_grid[iq,il,iw] += self.q_gridt[iq]+(4-iw)*self.ρq#q_gridt[iq]
                        
        # self.q_gridt,_=addaco_dist(self.σq,0.0,self.nq) 
        # self.q_grid=np.ones((self.nq,self.nwls,10)) 
        # for iq in range(self.nq): 
        #     for il in range(self.nwls): 
        #         self.q_grid[iq,il,:] = self.q[il] 
        #         if il<1: self.q_grid[iq,il,:] = self.q[il]+self.q_gridt[iq] 
                        
        #Create actual wages    
        self.w=np.zeros((self.T,self.nwls,self.nw))   
        for t in range(self.T):  
            for i in range(self.nwls):  
        
                self.w[t,i,:]=np.exp(self.wM[t,i]+self.wv)/self.scale/13.96*36.31
                if i==1: #miniwages are floored at 325*12 euros a year
                    self.w[t,i,:]=np.minimum(324*12/self.scale/self.wls[i],self.w[t,i,:])
                
                     
          
  
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
        self.amax=1300000/self.scale 
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
            self.startP[i]=self.startPd[index]+3.0
        
        self.pgrid=nonlinspace(self.startP.min(),self.R*2,self.NP,1.0)#np.linspace(0,self.R,self.NP)## # max one point per year in the law... 
          # points people start life with 
         
        #Multidimensional grid 
        self.mgrid=CGrid((self.agrid[0],self.agrid[-1],self.NA),(self.pgrid[0],self.pgrid[-1],self.NP)) 
        
                #Set seed 
        np.random.seed(2) 
         
        #Distribution of types and taste shocks 
        self.tw=np.sort(qe.MarkovChain(self.Π).simulate(self.N))# Type here 
        self.ts=np.random.rand(self.T,self.N) 
        
        self.q_sim = np.zeros(self.N,dtype=np.int32) 
        j=0
        for i in range(self.N):
            self.q_sim[i] = j
            j = j+1 if j<self.nq-1 else 0
        
        # ya=100
        # y=np.linspace(0.0,200.0,1000)
        # tax = np.zeros(1000)
        # atax = np.zeros(1000)
        
        # for i in range(len(y)): tax[i] = after_tax_income(y[i],y[i],ya)
        
      
        
        # import matplotlib.pyplot as plt
        # plt.plot(y,tax)
        

        
        # np.random.seed(3) 
        # self.Πq = np.ones((self.nq,10))/3
        
        # for iw in range(10): 
        #     if iw<5:#if just to get symmetry
        #         self.Πq[0,iw] = 1/3 - (5-iw)*self.ρq
        #         self.Πq[2,iw] = 1/3 + (5-iw)*self.ρq
        #     else:
        #         self.Πq[0,iw] = 1/3 - (5-1-iw)*self.ρq
        #         self.Πq[2,iw] = 1/3 + (5-1-iw)*self.ρq
            
        # self.q_sim = np.zeros((self.N),dtype=np.int32)+2
        # self.q_sim[np.random.rand(self.N)<np.cumsum(self.Πq,axis=0)[1][self.tw]]=1
        # self.q_sim[np.random.rand(self.N)<np.cumsum(self.Πq,axis=0)[0][self.tw]]=0
        

        #self.q_sim = np.repeat(self.q_sim[:,None],self.T,axis=1).T
        
        ############################################################## 
        #Compute the present value of life-time wealth for each group 
        ############################################################## 
 
from scipy.stats import norm 
 


#taxes: based on page 72 in https://www.fabian-kindermann.de/docs/progressive_pensions.pdf
#                           https://www.fabian-kindermann.de/docs/women_redistribution pg 20
#needs to be updated with 2000 rules: 
#    file:///C:/Users/32489/Downloads/Incentives_to_Work_The_Case_of_Germany.pdf
#    https://taxation-customs.ec.europa.eu/system/files/2016-09/structures2003.pdf pg 117
#    https://www.wiwiss.fu-berlin.de/fachbereich/vwl/corneo/dp/BachCorneoSteiner_DP080208.pdf
#    https://docs.iza.org/dp2245.pdf income splitting

@njit
def after_tax_income(y1g,y2g,y_mean,fraction,τ,no_retired = True):
    
    y1c = min(y1g*fraction,2*y_mean)
    y2c = min(y2g,         2*y_mean)
    
    payroll_tax_1 = τ*y1c
    payroll_tax_2 = τ*y2c
    
    y1 = y1g -  payroll_tax_1 if no_retired else y1g
    y2 = y2g -  payroll_tax_2 if no_retired else y2g
    
    share_married=1.65
    j_income = (y1+y2)/(share_married)
    rel_income = j_income/y_mean
    
    tax=0.0
    
    if rel_income<0.24: 
        tax = 0.0
    
    elif (rel_income<0.37) & (rel_income>=0.24): 
        tax = y_mean*share_married*(\
                                (rel_income-0.24)*0.14+\
                                (rel_income-0.24)*((0.2397-0.14)*(rel_income-0.24)/(0.37-0.24))/2                               
                                )
            
    elif (rel_income<1.46) & (rel_income>=0.37): 
        tax = y_mean*share_married*(\
                                 (0.37-0.24)*0.14+\
                                 (0.37-0.24)*((0.2397-0.14)*(0.37-0.24)/(0.37-0.24))/2+\
                                
                                 (rel_income-0.37)*0.2397+\
                                 (rel_income-0.37)*((0.42-0.2397)*(rel_income-0.37)/(1.46-0.37))/2     
                                
                                 )
    else:
        
        tax = y_mean*share_married*(\
                                 (0.37-0.24)*0.14+\
                                 (0.37-0.24)*((0.2397-0.14)*(0.37-0.24)/(0.37-0.24))/2+\
                                
                                 (1.46-0.37)*0.2397+\
                                 (1.46-0.37)*((0.42-0.2397)*(1.46-0.37)/(1.46-0.37))/2 +\
                                     
                                 (rel_income-1.46)*0.42
                                          
                                 )
    nety = y1g*(1-τ) + y2g#nety = y1g*(1-τ) + y2g if no_retired else  y1g + y2g 

#nety = y1g*(1-τ) + y2g*(1-τ) - tax - payroll_tax_1 - payroll_tax_2 if no_retired else  y1g + y2g - tax
    
    return nety

def hours(params,data,beg,end):
    
    D=data['h'][beg:end,:]
    return np.mean(D==1)*9.36+np.mean(D==2)*21.17+np.mean(D==3)*36.31

def hours_pr(params,data,beg,end):
    
    D=data['wls_pr'][beg:end,:]
    return D[:,:,1]*9.36+D[:,:,2]*21.17+D[:,:,3]*36.31

def addaco_dist(sd_z,mu,npts): 
   
 
    #Probabilities per period 
    ϵ=sd_z*norm.ppf((np.cumsum(np.ones(npts+1))-1)/npts) 
     
 
    X=np.zeros(npts) 
    for i in range(npts): 
        X[i]= mu+sd_z*npts*(norm.pdf((ϵ[i]-mu)/sd_z,0.0,1.0)-\
                         norm.pdf((ϵ[i+1]-mu)/sd_z,0.0,1.0)) 
    Pi = np.ones((npts,npts))/npts 
 
    return X, Pi 
             
             
 
@njit
def log(c,q):
    return np.log(np.maximum(c,0.00000000001))-q
 

