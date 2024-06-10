# This co.py file stores all miscellaneous functions needed to perform sol.py 
# Fabio Blasutto 
import numpy as np 
from consav.grids import nonlinspace # grids 
from scipy.stats import norm  
from numba import njit,prange 
import quantecon as qe
from scipy.integrate import quad
from scipy.stats import gamma

class setup(): 
     
    def __init__(self): 
     
        # Size of gridpoints:
        self.nq = 3    #fixed points, preference for working
        self.NA = 20  #assets gridpoints
        self.NP =7    #pension points gridpoints
        self.nwls = 4  #hours choice
        
        # First estimated parameters
        self.δ =  0.01 #0.00983949    # Discount rate
            
        self.q =np.array([0.0,0.16527134,0.09969482,1.0])  #Fixed cost of pticipation - mean
        self.σq =0.25623355   #Fixed cost of pticipation -sd 
        self.ρq =0.0#-0.4#0.00195224
   
        self.qshape = 0.80355901
        self.qscale = 1.73845578
        
        # Economic Environment: set pameters 
        self.T = 55         # Number of time periods 
        self.R = 35         # Retirement period 
        self.r = 0.02      # Interest rate 
        self.σ=0.001        #Size of taste shock 
                
        #Income
        self.scale=1000 #Show everything in 1000 euros
            
        # Hours choice
        self.wls=np.array([0.0,10.0, 20.0, 38.5])/38.5 #From GSOEP hrs/week = (10/ 20 / 38.5 ) 
        
        # income of men and women: sd of income shocks in t=0 and after that
        self.σzw=0.084;self.σ0zw= 0.4583;self.σzm=0.114;self.σ0zm=0.43
        self.nzw=3;self.nzm=3;self.nw = self.nzw*self.nzm
          
        #Pension
        self.E_bar_now = 27740.65230618203/self.scale  # Average earnings: ttps://www.gesetze-im-internet.de/sgb_6/ appendix 1 54256, exchange rate 1.9569471624266144
        self.ρ =348/self.scale      #Dollar value of points 
        self.Pmax = 1               #Threshold for pension points reform
        self.add_points=1.5         #point multiplicator during reform
        self.points_base=1.0        #standard points if not reform
        self.wls_point = np.array([0.0,0.0,1.0,1.0])      #share of income relevant for pension contributions 
       
        ##############
        # Income 
        ###############
        
        # uncertainty
        self.grid_zw,self.Π_zw, self.Π_zw0 =addaco_nonst(self.T,self.σzw,self.σ0zw,self.nzw)
        self.grid_zm,self.Π_zm, self.Π_zm0 =addaco_nonst(self.T,self.σzm,self.σ0zm,self.nzm)
        

        self.Π=[np.kron(self.Π_zw[t],self.Π_zm[t]) for t in range(self.T-1)] # couples trans matrix    
        self.Π0=np.kron(self.Π_zw0[0],self.Π_zm0[0])
        

        # for iw in range(5): 
        #     for jw in range(5): 
        #         for im in range(5): 
        #             for jm in range(5): 
                 
        #                   j=jm*5+jw 
        #                   i=im*5+iw 
        #                   #par.Πl[t][j,i]=par.Πlw[t][jw,iw] if ((jw==jm)) else 0.0 
        #                   self.Π0[j,i]=self.Π_zw0[0][jw,iw] if ((jw==jm)) else 0.0 
        

        
        self.w=np.zeros((self.T,self.nwls,self.nw))#final grid for w's income        
        for t in range(self.T)  :
            for iz in range(self.nw):    
                for i in range(self.nwls):
                    self.w[t,i,iz]=np.exp(8.25+.0297458*t -.0005628 *t**2 + self.grid_zw[t][iz//self.nzm])/self.scale/13.96*38.5
                    if i==1:#miniwages are floored at 325*12 euros a year 
                        self.w[t,i,iz]=np.minimum(324*12/self.scale/self.wls[i],self.w[t,i,iz])
                        
        self.y_N=np.zeros((self.T,self.nw))#final grid for w's income        
        for t in range(self.T)  :
            for iz in range(self.nw):    
                if t<self.R: self.y_N[t,iz]=np.exp(8.75+.0297458*t -.0005628 *t**2 + self.grid_zm[t][iz%self.nzm])/self.scale/13.96*38.5
                else:        self.y_N[t,iz]=self.y_N[self.R-1,iz]*0.45
  
     
         
        # Payroll taxes: https://www.nber.org/papers/w10525 
        self.τ = np.array([0.195 for t in range(self.T)])
        self.tax = np.array([0.0 for t in range(self.T)])
        self.tbase = np.array([1.0 for t in range(self.T)])#tax base 
       
      
        
        #Disutility from working
        self.q_grid=np.zeros((self.nq,self.nwls,self.nw))
        # self.q_grid_π=np.zeros((self.nq,self.nw))
        # self.q_gridt,_=addaco_dist(self.σq,0.0,self.nq)
        
        self.q_gridt = dist_gamma(self.qshape,self.qscale,self.nq)

        for il in range(1,self.nwls):
            for iw in range(self.nw):
                for iq in range(self.nq):
                    
                    self.q_grid[iq,il,iw]= self.q[il]*self.q_gridt[iq]
                    
            
        # Assets  grid   
        self.amin=0.0/self.scale
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
                    
        #Distribution of types in first period and shocks to be used
        self.tw=np.sort(qe.MarkovChain(self.Π0.T).simulate(self.N,init=self.nw//2))# Type here 
        self.shock_z=np.random.random_sample((self.N,self.T))
        
        #Distribution of taste shocks
        self.ts=np.random.rand(self.T,self.N) 
        
        #Fixed effect
        

#taxes: based on page 72 in https://www.fabian-kindermann.de/docs/progressive_pensions.pdf
#                           https://www.fabian-kindermann.de/docs/women_redistribution pg 20
# http://www.parmentier.de/steuer/index.php?site=einkommensteuersatz
# euro/de: 1.95583? Year 2000

# 13500(->13500/1.95583/27740.65=0.2488) 0.229

# 17496(-> 17496/1.95583/27740.65= 0.3224) 0.25

# 114696(-> 114696/1.95583/27740.65=2.1139) 0.51
@njit
def after_tax_income(tbase,y1g,y2g,y_mean,fraction,τ,no_retired = True):
    
    y1c = min(y1g*fraction*tbase,2*y_mean)
    y2c = min(y2g,         2*y_mean)
    
    payroll_tax_1 = τ*y1c
    payroll_tax_2 = τ*y2c
    
    #Compute taxable income
    y1 = y1g*tbase -  payroll_tax_1 if no_retired else y1g
    y2 = y2g       -  payroll_tax_2 if no_retired else y2g
    
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
   

    total_taxes = tax + payroll_tax_1 + payroll_tax_2 if no_retired else tax
    
    return y1g + y2g - total_taxes, total_taxes

#@njit(parallel=True)
def compute_atax_income_points(etax,tbase,T,R,nwls,nw,NP,τ,add_points,points_base,wls,w,E_bar_now,Pmax,wls_point,y_N,pgrid,ρ):

    income = np.zeros((T,nwls,nw,NP))
    total_taxes = np.zeros((T,nwls,nw,NP))
    total_taxes_mod = np.zeros((T,nwls,nw,NP))
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
                        
                        income[t,i,iw,ip], total_taxes[t,i,iw,ip]  = after_tax_income(tbase[t],w[t,i,iw]*wls[i],y_N[t,iw],E_bar_now,wls_point[i],tax)
                        
                        if not np.allclose(etax[t],0.0):#case where we are computing elasticities
                            _, total_taxes_mod[t,i,iw,ip]  = after_tax_income(tbase[t],w[t,i,iw]*wls[i]*(1-etax[t]),y_N[t,iw],E_bar_now,wls_point[i],tax)
                                                
                    else:            
                        
                        income[t,i,iw,ip], total_taxes[t,i,iw,ip]  = after_tax_income(1.0,ρ*pgrid[ip]     ,y_N[t,iw],E_bar_now,wls_point[i],tax,False)

    return income,point, total_taxes, total_taxes_mod
               
def hours(params,data,beg,end):
    
    D=data['h'][beg:end,:]
    return np.mean(D==1)*10.0+np.mean(D==2)*20.0+np.mean(D==3)*38.5

def hours_pr(params,data,beg,end):
    
    D=data['wls_pr'][beg:end,:]
    return D[:,:,1]*10.0+D[:,:,2]*20.0+D[:,:,3]*38.5

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
    return np.log(np.maximum(c,0.00000000001))-q#(np.maximum(c,0.00000000001))**(1-2)/(1-2)-q#


@njit
def points(mp,earnings,E_bar_now,Pmax,wls_point):
    return np.minimum(np.maximum(np.minimum(mp*earnings/E_bar_now,Pmax),earnings/E_bar_now),2)*wls_point


###########################
# Uncertainty below       #
###########################
 
def sd_rw(T,sigma_persistent,sigma_init):
    
    if isinstance(sigma_persistent,np.ndarray):
        return np.sqrt([sigma_init**2 + t*sigma_persistent[t]**2 for t in range(T)])
    else:
        return np.sqrt(sigma_init**2 + np.arange(0,T)*(sigma_persistent**2))
    
def sd_rw_trans(T,sigma_persistent,sigma_init,sigma_transitory):
    return sd_rw(T, sigma_persistent, sigma_init)

    
    
def normcdf_tr(z,nsd=5):
        
        z = np.minimum(z, nsd*np.ones_like(z))
        z = np.maximum(z,-nsd*np.ones_like(z))
            
        pup = norm.cdf(nsd,0.0,1.0)
        pdown = norm.cdf(-nsd,0.0,1.0)
        const = pup - pdown
        
        return (norm.cdf(z,0.0,1.0)-pdown)/const
    
    
def normcdf_ppf(z): return norm.ppf(z,0.0,1.0)       
        

def addaco_nonst(T=40,sigma_persistent=0.05,sigma_init=0.2,npts=50):
  
    # start with creating list of points
    sd_z = sd_rw(T,sigma_persistent,sigma_init)
    sd_z0 = np.array([np.sqrt(sd_z[t]**2-sigma_init**2) for t in range(T)])
        
    Pi = list();Pi0 = list();X = list();Int=list()


    #Probabilities per period
    Pr=normcdf_ppf((np.cumsum(np.ones(npts+1))-1)/npts)
    
    #Create interval limits
    for t in range(0,T):Int = Int + [Pr*sd_z[t]]
        
    
    #Create gridpoints
    for t in range(T):
        line=np.zeros(npts)
        for i in range(npts):
            line[i]= sd_z[t]*npts*(norm.pdf(Int[t][i]  /sd_z[t],0.0,1.0)-\
                                   norm.pdf(Int[t][i+1]/sd_z[t],0.0,1.0))
            
        X = X + [line]

    def integrand(x,e,e1,sd,sds):
        return np.exp(-(x**2)/(2*sd**2))*(norm.cdf((e1-x)/sds,0.0,1.0)-\
                                          norm.cdf((e- x)/sds,0.0,1.0))
            
            
    #Fill probabilities
    for t in range(1,T):
        Pi_here = np.zeros([npts,npts]);Pi_here0 = np.zeros([npts,npts])
        for i in range(npts):
            for jj in range(npts):
                
                Pi_here[i,jj]=npts/np.sqrt(2*np.pi*sd_z[t-1]**2)\
                    *quad(integrand,Int[t-1][i],Int[t-1][i+1],
                     args=(Int[t][jj],Int[t][jj+1],sd_z[t],sigma_persistent))[0]
                
                Pi_here0[i,jj]= norm.cdf(Int[t][jj+1],0.0,sigma_init)-\
                                   norm.cdf(Int[t][jj],0.0,sigma_init)
                                   
            #Adjust probabilities to get exactly 1: the integral is an approximation
            Pi_here[i,:]=Pi_here[i,:]/np.sum(Pi_here[i,:])
            Pi_here0[i,:]=Pi_here0[i,:]/np.sum(Pi_here0[i,:])
                
        Pi = Pi + [Pi_here.T]
        Pi0 = Pi0 + [Pi_here0.T]
        
    return X, Pi, Pi0   

def rouw_nonst_one(sd0,sd1,npts):
   
    # this generates one-period Rouwenhorst transition matrix
    assert(npts>=2)
    pi0 = 0.5*(1+(sd0/sd1))
    Pi = np.array([[pi0,1-pi0],[1-pi0,pi0]])
    assert(pi0<1)
    assert(pi0>0)
    for n in range(3,npts+1):
        A = np.zeros([n,n])
        A[0:(n-1),0:(n-1)] = Pi
        B = np.zeros([n,n])
        B[0:(n-1),1:n] = Pi
        C = np.zeros([n,n])
        C[1:n,1:n] = Pi
        D = np.zeros([n,n])
        D[1:n,0:(n-1)] = Pi
        Pi = pi0*A + (1-pi0)*B + pi0*C + (1-pi0)*D
        Pi[1:n-1] = 0.5*Pi[1:n-1]
        
        assert(np.all(np.abs(np.sum(Pi,axis=1)-1)<1e-5 ))
    
    return Pi


def rouw_nonst(T=40,sigma_persistent=0.05,sigma_init=0.2,npts=10):
   
    sd_z = sd_rw(T,sigma_persistent,sigma_init)
    sd_z0 = np.array([np.sqrt(sd_z[t]**2-sigma_init**2) for t in range(T)])
       
    Pi = list();Pi0 = list();X = list()

    for t in range(0,T):
        nsd = np.sqrt(npts-1)
        X = X + [np.linspace(-nsd*sd_z[t],nsd*sd_z[t],num=npts)]
        
        if t >= 1: Pi = Pi +   [rouw_nonst_one(sd_z[t-1],sd_z[t] ,npts).T]
        if t >= 1: Pi0 = Pi0 + [rouw_nonst_one(sd_z0[t-1],sd_z[t-1],npts).T]
       
    return X, Pi, Pi0      
        
 


@njit(fastmath=True)
def mc_simulate(statein,Piin,shocks):
    """This simulates transition one period ahead for a Markov chain
    
    Args: 
        Statein: scalar giving the initial state
        Piin: transition matrix n*n [post,initial]
        shocks: scalar in [0,1]
    
    """ 
    return  np.sum(np.cumsum(Piin[:,statein])<shocks)





def dist_gamma(k,theta,npts):
    """
    Discretize the gamma function into npts points. Idea: first divide
    X space into npts+1 points. Between each point there should be the same 
    probability
    

    Parameters
    ----------
    k : TYPE: real
        DESCRIPTION: shape parameter of gamma distribution
    theta : TYPE: real
        DESCRIPTION:  scale parameter of gamma distribution
    npts : TYPE; int
        DESCRIPTION: number of points gamma function should be discretized

    Returns
    -------
    discr : TYPE np.array, one dimension of length npts
        DESCRIPTION.

    """
    
    #percent point function, equal spacing for percentiles
    zvals = gamma.ppf(np.linspace(0.0, 1.0, npts+1), k,scale=theta)
    
    #expected value between two consecutive zvals (note the rescaling)
    discr = np.array([gamma.expect(lambda x:x,args=(k,),scale=theta,lb=zvals[i],ub=zvals[i+1])/
                      gamma.expect(lambda x:1,args=(k,),scale=theta,lb=zvals[i],ub=zvals[i+1])for i in range(npts)])
    
    return discr


