# This co.py file stores all miscellaneous functions needed to perform sol.py 
# Fabio Blasutto 
import numpy as np 
from consav.grids import nonlinspace # grids 
from scipy.stats import norm  
from numba import njit,prange 
import quantecon as qe

class setup(): 
     
    def __init__(self): 
     
        # Size of gridpoints:
        self.nw=10     #groups by economic status
        self.nq =5    #fixed points, preference for working
        self.NA = 150  #assets gridpoints
        self.NP =33    #pension points gridpoints
        self.nwls = 4  #hours choice
        
        # First estimated parameters
        self.δ =  0.01616545#0.00983949    # Discount rate
        self.q =np.array([0.0,0.39215417*(0.4549387),0.39215417*(0.35289862),0.39215417])  #Fixed cost of pticipation - mean
        self.σq = 0.13307952  #Fixed cost of pticipation -sd 
        self.ρq =0.0#0.00195224
        
        # Economic Environment: set pameters 
        self.T = 56         # Number of time periods 
        self.R = 36         # Retirement period 
        self.r = 0.03      # Interest rate 
        self.σ=0.001        #Size of taste shock 
                
        #Income
        self.scale=1000 #Show everything in 1000 euros
            
        # Hours choice
        self.wls=np.array([0.0,9.36, 21.17, 36.31])/36.31 #From GSOEP hrs/week = (10/ 20 / 38.5 ) 
          
        #Pension
        self.E_bar_now = 27740.65230618203/self.scale  # Average earnings: ttps://www.gesetze-im-internet.de/sgb_6/ appendix 1 54256, exchange rate 1.9569471624266144
        self.ρ =350/self.scale      #Dollar value of points 
        self.Pmax = 1               #Threshold for pension points reform
        self.add_points=1.5         #point multiplicator during reform
        self.points_base=1.0        #standard points if not reform
        self.wls_point = np.array([0.0,0.0,1.0,1.0])      #share of income relevant for pension contributions 
       
            
        # Income 

        self.wM=np.zeros((self.T,self.nwls))#trend in women's earnings
        for t in range(self.T): self.wM[t,:]=-0.1806434+.0297458*t -.0005628 *t**2
                
        
        self.wv=np.array([ 7.631935, 8.090471, 8.286575 ,
                          8.44488, 8.580534 , 8.709507,
                          8.827328, 8.96867, 9.145868, 9.563336]) #w's income groups initial income
        
        self.w=np.zeros((self.T,self.nwls,self.nw))#final grid for w's income
        for t in range(self.T):  
            for i in range(self.nwls):  
        
                self.w[t,i,:]=np.exp(self.wM[t,i]+self.wv)/self.scale/13.96*36.31
                if i==1: #miniwages are floored at 325*12 euros a year
                    self.w[t,i,:]=np.minimum(324*12/self.scale/self.wls[i],self.w[t,i,:])
                    
                    
        self.wv_men=np.array([16735.71,16920.84,19447.71,
                              18881.82,22105.11,18759.2,
                              19781.31,21205.22,24253.39,26751.28]) #m's income groups initial income
        
        self.wv_men=np.array([16735.71,16920.84,19447.71,18881.82,22105.11,18759.2,19781.31,21205.22,24253.39,26751.28]) 
        self.y_N=np.zeros((self.T,self.nw))   
        for t in range(self.R):  
            for i in range(self.nw):  
                self.y_N[t,i]=(-2930.40118+501.4029*(t)-11.82488*(t)**2+self.wv_men[i])/self.scale
                   
        for t in range(self.R,self.T):  #income at retirement for men
            self.y_N[t,:]=self.y_N[self.R-1,:]*0.45
                  
         
        # Payroll taxes: https://www.nber.org/papers/w10525 
        self.τ = np.array([0.195 for t in range(self.T)])
        self.tax = 0.0 #tax useful only for computing elasticities
      
        
        #Disutility from working
        self.q_grid=np.zeros((self.nq,self.nwls,10))
        self.q_grid_π=np.zeros((self.nq,10))
        self.q_gridt,_=addaco_dist(self.σq,0.0,self.nq)

        for il in range(1,self.nwls):
            for iw in range(10):
                for iq in range(self.nq):
                    
                    self.q_grid[iq,il,iw]= self.q[il]-self.q_gridt[iq]+(4-iw)*self.ρq
                # if il<1: 
                    
                #     mean = self.σq/.5376561*self.ρq*(np.log(self.wv[iw])-2.15279653495785)
                #     sd = (1-self.ρq**2)**0.5*self.σq
                    
                #     q_gridt,π = addaco_dist(sd,mean,self.nq) 
                #     self.q_grid_π[:,iw] =  π[0]
                #     for iq in range(self.nq):
                    
                #         self.q_grid[iq,il,iw] += self.q_gridt[iq]+(4-iw)*self.ρq#q_gridt[iq]
                        
     
        # Assets  grid   
        self.amin=-160000/self.scale
        self.amax=1000000/self.scale 
        self.agrid=nonlinspace(self.amin,self.amax,self.NA,1.0)
        
        #Pension points grid
        self.pgrid=nonlinspace(5.509712,7.733748+self.R*2,self.NP,1.0)         
    
        #######################################################################
        #Simulations
        #######################################################################
        np.random.seed(2) 
         
        self.N = 20000        # agents to simulate 
        
        #Initial assets  
        self.startA=np.zeros(self.N)  
        assets=np.array([5877.601,21127.74,10671.82,7991.483,21614.71,15574.05,15981.4,24145.36,34662.95,32189.31]) 
        for i in range(self.N): 
            index=int(i/self.N*10) 
            self.startA[i]=assets[index]/self.scale 
            
        #Initial pension points
        self.startPd = np.array([2.509712,2.732802,3.061959,2.951443,3.380279,3.498868,3.528733,4.050392,4.264604,4.733748])
        
        self.startP=np.zeros(self.N) 
        for i in range(self.N): 
            index=int(i/self.N*10) 
            self.startP[i]=self.startPd[index]+3.0
                    
        #Distribution of types
        self.Π = np.ones((10,10))/10.0
        self.tw=np.sort(qe.MarkovChain(self.Π).simulate(self.N))# Type here 
        
        #Distribution of taste shocks
        self.ts=np.random.rand(self.T,self.N) 
        
        #Distribution of labor preferences fixed effects
        self.q_sim = np.zeros(self.N,dtype=np.int32) 
        j=0
        for i in range(self.N):
            self.q_sim[i] = j
            j = j+1 if j<self.nq-1 else 0
        
     
        
#taxes: based on page 72 in https://www.fabian-kindermann.de/docs/progressive_pensions.pdf
#                           https://www.fabian-kindermann.de/docs/women_redistribution pg 20
# http://www.parmentier.de/steuer/index.php?site=einkommensteuersatz
# euro/de: 1.95583

# 13500(->13500/1.95583/27740.65=0.2488) 0.229

# 17496(-> 17496/1.95583/27740.65= 0.3224) 0.25

# 114696(-> 114696/1.95583/27740.65=2.1139) 0.51
@njit
def after_tax_income(etax,y1g,y2g,y_mean,fraction,τ,no_retired = True):
    
    y1c = min(y1g*fraction,2*y_mean)
    y2c = min(y2g,         2*y_mean)
    
    payroll_tax_1 = τ*y1c
    payroll_tax_2 =  0.195*y2c if τ>0 else 0.0
    
    #Compute taxable income
    y1 = y1g -  payroll_tax_1 if no_retired else y1g
    y2 = y2g -  payroll_tax_2 if no_retired else y2g
    
    #If mini-job, then taxable income for woman is 0
    if fraction<1: y1 = 0.0
    
    share_married=1.65
    j_income = (y1+y2)/(share_married)
    rel_income = j_income/y_mean
    
    tax=0.0
    
    if rel_income<0.2488: 
        tax = 0.0
    
    elif (rel_income<0.25) & (rel_income>=0.2488): 
        tax = y_mean*share_married*(\
                                (rel_income-0.2488)*0.229+\
                                (rel_income-0.2488)*((0.25-0.229)*(rel_income-0.2488)/(0.25-0.2488))/2                               
                                )
            
    elif (rel_income<2.1139) & (rel_income>=0.25): 
        tax = y_mean*share_married*(\
                                 (0.25-0.2488)*0.229+\
                                 (0.25-0.2488)*((0.25-0.229)*(0.25-0.2488)/(0.25-0.2488))/2+\
                                
                                 (rel_income-0.25)*0.25+\
                                 (rel_income-0.25)*((0.51-0.25)*(rel_income-0.25)/(2.1139-0.25))/2     
                                
                                 )
    else:
        
        tax = y_mean*share_married*(\
                                 (0.25-0.2488)*0.229+\
                                 (0.25-0.2488)*((0.25-0.229)*(0.25-0.2488)/(0.25-0.2488))/2+\
                                
                                 (2.1139-0.25)*0.25+\
                                 (2.1139-0.25)*((0.51-0.25)*(2.1139-0.25)/(2.1139-0.25))/2 +\
                                     
                                 (rel_income-2.1139)*0.51
                                          
                                 )
   

    nety = y1g*(1-etax) + y2g - tax - payroll_tax_1 - payroll_tax_2 if no_retired else  y1g + y2g - tax
    
    return nety, etax*y1g

@njit(parallel=True)
def compute_atax_income_points(etax,T,R,nwls,nw,NP,τ,add_points,points_base,wls,w,E_bar_now,Pmax,wls_point,y_N,pgrid,ρ):

    income = np.zeros((T,nwls,nw,NP))
    income2 = np.zeros((T,nwls,nw,NP))
    point =  np.zeros((T,nwls,nw,2))#last dimension if for policy-no policy
    
    for t in prange(T):
        for i in range(nwls):
            for iw in range(nw):
                for ip in range(NP):
                                          
                    tax=τ[t]      if (i>1) else 0.0
                    
                    policy_timing=((t >=8) & (t <=11))
                    
                   
                    #Multiplier of points based on points
                    mp_policy=add_points if policy_timing else points_base
                    mp_base  =points_base
                    
                    
                    point[t,i,iw,0] = points(mp_base  ,wls[i]*w[t,i,iw],E_bar_now,Pmax,wls_point[i])
                    point[t,i,iw,1] = points(mp_policy,wls[i]*w[t,i,iw],E_bar_now,Pmax,wls_point[i])
                    
                    if (t+1<=R): 
                        
                        income[t,i,iw,ip], income2[t,i,iw,ip]  = after_tax_income(etax,w[t,i,iw]*wls[i],y_N[t,iw],E_bar_now,wls_point[i],tax) 
                    else:            
                        
                        income[t,i,iw,ip], income2[t,i,iw,ip]  = after_tax_income(etax,ρ*pgrid[ip]     ,y_N[t,iw],E_bar_now,wls_point[i],tax,False)

    return income,point, income2  
               
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


@njit
def points(mp,earnings,E_bar_now,Pmax,wls_point):
    return np.minimum(np.maximum(np.minimum(mp*earnings/E_bar_now,Pmax),earnings/E_bar_now),2)*wls_point
 

