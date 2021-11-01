# this is the key program to solve the model

# import packages
# Time
import time
import numpy as np
#from scipy.interpolate import interp1d
#from scipy.optimize import bisect as brentq
from quantecon.optimize.root_finding import brentq 
from numba import njit, prange
import numexpr as ne
from interpolation.splines import CGrid, eval_linear, nodes
#https://www.econforge.org/interpolation.py/


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
    
    mult=np.power(((1+r)/(1+delta)),(-1/gamma_c))

    
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
        policy=((t+1 >=3 & t+1 <=10) & (reform))
        
        if policy: 
            mult_pens=ne.evaluate('1.5*wt/E_bar_now*pmu/(1+delta)')
        else:
            mult_pens=ne.evaluate('1.0*wt/E_bar_now*pmu/(1+delta)')
        
        ################################################
        #Endogenous gridpoints here
        ##############################################
        
        #Get consumption from Eurler Equation
        ce=policyC[t+1,:,:]*mult
        
       
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
        
        #Not retired
        if (t+1<=R):        
            if policy:
                
                #For counting points under the reform, check the condition
                hem=he.copy()#np.maximum(np.minimum(1.5*he*wt/E_bar_now,np.ones(np.shape(he))),he*wt/E_bar_now)
                pe=ne.evaluate('pgrid_box-hem')
                
            else:
                #Normal condition
                pe=ne.evaluate('pgrid_box-    he*wt/E_bar_now')
                
        #How much assets? Just use the Budget constraint!
        if (t+1<=R):  
            ae=ne.evaluate('(agrid_box-wt*he*(1-tau)-y_N+ce)/(1+r)')
        else:
            ae=ne.evaluate('(agrid_box-rho*pe-y_N+ce)/(1+r)')
            
       
        ################################################
        # Now interpolate to be back on grid...
        ###############################################
        
        # This gets consumption and assets back on grid
        policyC[t,:,:],policyA1[t,:,:]=solveEulerEquation2(agrid,agrid_box,pgrid_box,ae,ce,pe,numPtsP,numPtsA,\
                                               r,wt,y_N,tau,gamma_c,gamma_h,\
                                               beta,maxHours,t,R,pgrid,rho,E_bar_now,\
                                               mult_pens,mgrid)
            
        #Given consumption and assets, obtain optimal hours on grid
        Pc=policyC[t,:,:] #Needed for computation below
        if (t+1<=R):
                 policyh[t,:,:]=  \
                 ne.evaluate('maxHours-((mult_pens+wt*(1-tau)*(Pc**(-gamma_c)))/beta)**(-1/gamma_h)')
                 
        #Update marginal utility of having more pension points
        if (t+1>R): 
            pmutil[t,:,:]=ne.evaluate('(pmu+rho*Pc**(-gamma_c))/(1+delta)')
        else:
            pmutil[t,:,:]=ne.evaluate('pmu/(1+delta)')
               
      
@njit(parallel=True)           
def solveEulerEquation2(agrid,agrid_box,pgrid_box,ae,ce,pe,numPtsP,numPtsA,r,wt,y_N,tau,gamma_c,
                        gamma_h,beta,maxHours,t,R,pgrid,rho,E_bar_now,mult_pens,mgrid):
    
    pCv,pAv,pPv,pC,pA,pP=np.empty((6,numPtsA,numPtsP),dtype=np.float64)
    where=np.empty((numPtsA,numPtsP),dtype=np.bool_)
    
    #Interpolate to be on the grid of assets
    for i in prange(numPtsP):           
        pCv[:,i]=np.interp(agrid, ae[:,i],ce[:,i])
        pAv[:,i]=np.interp(agrid, ae[:,i],agrid)
        pPv[:,i]=np.interp(agrid, ae[:,i],pe[:,i])
        
    #Interpolate to be on the grid of pension points    
    for j in prange(numPtsA):           
        pC[j,:]=np.interp(pgrid, pe[j,:],pCv[j,:])
        pA[j,:]=np.interp(pgrid, pe[j,:],pAv[j,:])
        pP[j,:]=np.interp(pgrid, pe[j,:],pPv[j,:])
     
    
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
            
            tup=(agrid[j],r,y_N,gamma_c,gamma_h,maxHours,mult_pens[j,i],wt,tau,beta)
            
            if where[j,i]:#Here constraints bind...
                if (t+1>R):
                    
                    #If retired...
                    pC[j,i]=(1+r)*agrid[j]+y_N+pgrid[j]*rho-agrid[0]
                else:
                    #If not retired
                    hmin=agrid[j]*(1+r)
                    hmax=agrid[j]*(1+r)+y_N+maxHours*wt*(1-tau)-agrid[0]
                    
                    #Rootfinding on FOCs to get optimal consumption!
                    #HERE I SHOULD ALSO HANDLE MAX PENSION POINT ISSUE!!!
                    pC[j,i]=brentq(minim,hmin,hmax,args=tup)[0]
                    
    return pC,pA


@njit
def minim(x,a,r,y_N,gamma_c,gamma_h,maxHours,mult_pens,w,tau,beta):
    h=-np.power((np.power(x,-gamma_c)*w*(1-tau)+mult_pens)/beta,-1/gamma_h)+maxHours
    return (1+r)*a+w*h*(1-tau)+y_N-x
    
 
