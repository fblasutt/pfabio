# this is the key program to solve the model

# import packages
# Time
import time
import numpy as np
#from scipy.interpolate import interp1d
#from scipy.optimize import bisect as brentq
from quantecon.optimize.root_finding import brentq 
from numba import njit, prange, jit
import numexpr as ne
from interpolation.splines import CGrid, eval_linear, nodes
import matplotlib.pyplot as plt
#https://www.econforge.org/interpolation.py/

from consav import linear_interp

def solveEulerEquation(reform, par):
    
    time_start = time.time()
    
    r=par.r;delta=par.delta;gamma_c=par.gamma_c;R=par.R;tau=par.tau;beta=par.beta;
    w=np.array(par.w);agrid=par.agrid;y_N=par.y_N;gamma_h=par.gamma_h;T=par.T;numPtsA=par.numPtsA
    numPtsP=par.numPtsP;pgrid=par.pgrid;maxHours=par.maxHours;rho=par.rho;E_bar_now=par.E_bar_now;
    mgrid=par.mgrid
    
    policyA1,policyh,policyC,V,policyp,pmutil = np.empty((6,T, numPtsA, numPtsP))
        
    
    solveEulerEquation1(policyA1, policyh, policyC, policyp,pmutil,\
                        reform,r,delta,gamma_c,R,tau,beta,w,agrid,y_N,\
                        gamma_h,T,numPtsA,numPtsP,pgrid,maxHours,rho,E_bar_now,mgrid)

    elapsed = time.time() - time_start    
    print('Finished, Reform =', reform, 'Elapsed time is', elapsed, 'seconds')   
    
    return policyA1,policyh,policyC,V,policyp

#@njit(fastmath=True)
def solveEulerEquation1(policyA1, policyh, policyC, policyp,pmutil,reform,r,delta,gamma_c,R,tau,\
                        beta,w,agrid,y_N,gamma_h,T,numPtsA,numPtsP,pgrid,maxHours,rho,E_bar_now,mgrid):
    
    # The rest is interior solution
    """ Use the method of endogenous gridpoint to solve the model.
        To improve it further: jit it, then use math.power, not *
    """
    
    agrid_box=np.transpose(np.tile(agrid,(numPtsP,1)))
    pgrid_box=np.tile(pgrid,(numPtsA,1))
    
    policyA1[T-1,:,:] = np.zeros((numPtsA, numPtsP))  # optimal savings
    policyh[T-1,:,:] = np.zeros((numPtsA, numPtsP))   # optimal earnings
    policyC[T-1,:,:] = agrid_box*(1+r) + y_N +\
                                rho*pgrid_box         # optimal consumption
    pmutil[T-1,:,:]=\
        np.power(policyC[T-1,:,:],-gamma_c)*rho       # mu of more pension points
    
    for t in range(T-2,-1,-1):
                         
        #Define initial variable for fast coputation later
        pmu=pmutil[t+1,:,:]
        wt=w[t]
        policy=((t+1 >=3) & (t+1 <=10) & (reform==1))
     
        if policy: 
            mult_pens=ne.evaluate('1.5*wt/E_bar_now*pmu/(1+delta)')
        else:
            mult_pens=ne.evaluate('1.0*wt/E_bar_now*pmu/(1+delta)')
        
        ################################################
        #Endogenous gridpoints here
        ##############################################
        
        #Get consumption from Eurler Equation
        ce=policyC[t+1,:,:]*np.power(((1+r)/(1+delta)),(-1/gamma_c))
             
        #How much work? This follows from the FOC      
        if (t+1<=R):            
            #Not retired
            he=ne.evaluate('maxHours-((mult_pens+wt*(1-tau)*(ce**(-gamma_c)))/beta)**(-1/gamma_h)')
            
            #Check conditions for additional points
            #if (policy):
             #   he_m=ne.evaluate('maxHours-((1.0*wt/E_bar_now*pmu/(1+delta)+wt*(1-tau)*(ce**(-gamma_c)))/beta)**(-1/gamma_h)')
             #   he[(1.5*he*wt/E_bar_now>1)]=he_m[(1.5*he*wt/E_bar_now>1)]
                
        #Retired case        
        if (t+1>R):he=np.zeros((numPtsA, numPtsP))
                  
        
        #How much points should you have given the decisions?
        
        #Retired
        if (t+1>R): pe=pgrid_box
        
        # #Not retired
        # if (t+1<=R):        
        #     if policy:
                
        #         #For counting points under the reform, check the condition
        #         hem=np.maximum(np.minimum(1.5*he*wt/E_bar_now,np.ones(np.shape(he))),he*wt/E_bar_now)
        #         pe=ne.evaluate('pgrid_box-hem')
                
        #     else:
        #         #Normal condition
        #         pe=ne.evaluate('pgrid_box-    he*wt/E_bar_now')
        pe=ne.evaluate('pgrid_box-    he*wt/E_bar_now')        
                
        #How much assets? Just use the Budget constraint!
        if (t+1<=R):  
            ae=ne.evaluate('(agrid_box-wt*he*(1-tau)-y_N+ce)/(1+r)')
        else:
            ae=ne.evaluate('(agrid_box-rho*pe       -y_N+ce)/(1+r)')
            
       
        ################################################
        # Now interpolate to be back on grid...
        ###############################################
        
        # This gets consumption and assets back on grid
        policyC[t,:,:],policyA1[t,:,:],pp=solveEulerEquation2(agrid,agrid_box,pgrid_box,ae,ce,pe,numPtsP,numPtsA,\
                                               r,wt,y_N,tau,gamma_c,gamma_h,\
                                               beta,maxHours,t,R,pgrid,rho,E_bar_now,\
                                               mult_pens,mgrid)
            
        #Given consumption and assets, obtain optimal hours on grid
        Pc=policyC[t,:,:] #Needed for computation below
        Pa=policyA1[t,:,:] #Needed for computation below
        if (t+1<=R):
                 #aaa=ne.evaluate('(Pc+Pa-(1+r)*agrid_box-y_N)/(wt*(1-tau))')  
                 #aaa1=ne.evaluate('maxHours-((mult_pens+wt*(1-tau)*(Pc**(-gamma_c)))/beta)**(-1/gamma_h)')
                 policyh[t,:,:]=  ne.evaluate('(Pc+Pa-(1+r)*agrid_box-y_N)/(wt*(1-tau))')  
                      
                  
  
        #Update marginal utility of having more pension points
        if (t+1>R): 
            pmutil[t,:,:]=ne.evaluate('(pmu+rho*Pc**(-gamma_c))/(1+delta)')
        else:
            pmutil[t,:,:]=ne.evaluate('pmu/(1+delta)')
            
        #Check points consistency
        #print((np.mean((pgrid_box+wt*policyh[t,:,:]/E_bar_now-pp)[:,0:2900])))
               
      
@njit(parallel=True)           
def solveEulerEquation2(agrid,agrid_box,pgrid_box,ae,ce,pe,numPtsP,numPtsA,r,wt,y_N,tau,gamma_c,
                        gamma_h,beta,maxHours,t,R,pgrid,rho,E_bar_now,mult_pens,mgrid):
    
    pCv,pAv,pPv,pC,pA,pP,pPg=np.empty((7,numPtsA,numPtsP),dtype=np.float64)
    where=np.empty((numPtsA,numPtsP),dtype=np.bool_)
    
    #Method of interpolation:
    #https://www.sciencedirect.com/science/article/pii/S0165188916301920?via%3Dihub
    #Note that this method could easily be used to handle non-convexities
        #Consav interpolations
    for j in prange(numPtsA):           
        linear_interp.interp_1d_vec( pe[j,:],ae[j,:],pgrid,pAv[j,:])
        linear_interp.interp_1d_vec( pe[j,:],pgrid,pgrid,pPv[j,:])
        
    for i in prange(numPtsP): 
        linear_interp.interp_1d_vec( ae[:,i],agrid,pAv[:,i],pPg[:,i])
        
    #Interpolate to be on the grid of assets
    for i in prange(numPtsP):           
      #  pC[:,i]=np.interp(agrid, pAv[:,i],pCv[:,i])
        linear_interp.interp_1d_vec( pAv[:,i],pPg[:,i],agrid,pA[:,i])
        linear_interp.interp_1d_vec( pAv[:,i],pPv[:,i],agrid,pP[:,i])
        
    if np.min(pA)<agrid[0]:
        for i in prange(numPtsA): 
            for j in prange(numPtsP): 
                pA[i,j]=max(pA[i,j],agrid[0])
                
    if np.min(pP)<pgrid[0]:
        for i in prange(numPtsA): 
            for j in prange(numPtsP): 
                pP[i,j]=max(pP[i,j],pgrid[0])
        
        
    #Post decision gridpoints
    for i in prange(numPtsA-1):        
         for j in prange(numPtsP-1): 
             
            #Limits of the bounding box on the lower triangle
            mina=min(ae[i,j],ae[i+1,j],ae[i,j+1])
            maxa=max(ae[i,j],ae[i+1,j],ae[i,j+1])
            minp=min(pe[i,j],pe[i+1,j],pe[i,j+1])
            maxp=max(pe[i,j],pe[i+1,j],pe[i,j+1])
             
            denom=(pe[i+1,j]-pe[i,j+1])*(ae[i,j]  -ae[i,j+1])+\
                  (ae[i,j+1]-ae[i+1,j])*(pe[i,j]  -pe[i,j+1])
            #Grid where you want to interpoalte
            for ii in prange(numPtsA):        
                for jj in prange(numPtsP): 
                    
                    if (ae[ii,jj]>0) & (pe[ii,jj]>0):
                        #Get if the point are considering falls within the
                        #bounding box delimited by the triangle defined above
                        if ((agrid[ii]>=mina) & (agrid[ii]<maxa)\
                          & (pgrid[jj]>=minp) & (pgrid[jj]<maxp)): 
                            
                            #Get the weights to apply the barycentric interpolation
                            wa=((pe[i+1,j]-pe[i,j+1])*(agrid[ii]-ae[i,j+1])+\
                                (ae[i,j+1]-ae[i+1,j])*(pgrid[jj]-pe[i,j+1]))/denom
                                   
                            wb=((pe[i,j+1]-pe[i,j  ])*(agrid[ii]-ae[i,j+1])+\
                                (ae[i,j]-  ae[i,j+1])*(pgrid[jj]-pe[i,j+1]))/denom
                             
                            wc=1.0-wa-wb       
                            pA[ii,jj]=wa*ae[i,j]+wb*ae[i+1,j]+wc*ae[i,j+1]
                            pP[ii,jj]=wa*pe[i,j]+wb*pe[i+1,j]+wc*pe[i,j+1]
                            #pC[ii,jj]=wa*ce[i,j]+wb*ce[i+1,j]+wc*ce[i,j+1]
                            
    
   
    
    ########################################################
    #Below handle the case of binding borrowing constraints
    #######################################################
    
    #Where does constraints bind?
    for j in prange(numPtsA):
        for i in prange(numPtsP): 
            where[j,i]=(pA[j,i]<=agrid[0])
            
    #Update consumption where constraints are binding
    for j in prange(numPtsA):
        for i in prange(numPtsP): 
            
            tup=(agrid[j],r,y_N,gamma_c,gamma_h,maxHours,mult_pens[j,i],wt,tau,beta,pA[j,i])
            
            #if where[j,i]:#Here constraints bind...
            if (t+1>R):
                
                #If retired...
                pC[j,i]=(1+r)*agrid[j]+y_N+pgrid[i]*rho-max(pA[j,i],agrid[0])
            else:
                #If not retired
                hmin=0.000001#agrid[j]*(1+r)+y_N
                hmax=agrid[j]*(1+r)+y_N+maxHours*wt*(1-tau)-max(pA[j,i],agrid[0])
                
                #Rootfinding on FOCs to get optimal consumption!
                #HERE I SHOULD ALSO HANDLE MAX PENSION POINT ISSUE!!!
                pC[j,i]=brentq(minim,hmin,hmax,args=tup)[0]
                        
    return pC,pA,pP


@njit
def minim(x,a,r,y_N,gamma_c,gamma_h,maxHours,mult_pens,w,tau,beta,a1):
    h=-np.power((np.power(x,-gamma_c)*w*(1-tau)+mult_pens)/beta,-1/gamma_h)+maxHours
    return (1+r)*a+w*h*(1-tau)+y_N-a1-x


@njit#(nopython=False)
def interp_npp(grid,xnew,return_wnext=True,trim=False,trim_half=False):    
    # this finds grid positions and weights for performing linear interpolation
    # this implementation uses numpy
    j=np.empty(grid.shape,dtype=np.int32)
    wnext=np.empty(grid.shape,dtype=np.float64)
    #if trim: xnew = np.minimum(grid[-1], np.maximum(grid[0],xnew) )
    #if trim_half: xnew = np.maximum(grid[0],xnew) 
    
    j = np.minimum( np.searchsorted(grid,xnew,side='left')-1, grid.size-2 )
    wnext = (xnew - grid[j])/(grid[j+1] - grid[j])
    
    return j, (1-wnext)#(wnext if return_wnext else 1-wnext) 
 
