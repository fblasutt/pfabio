# this is the key program to solve the model

# import packages
# Time
import time
import numpy as np
#from scipy.interpolate import interp1d
from scipy.optimize import bisect as brentq
#from quantecon.optimize.root_finding import brentq 
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
            mult_pens=ne.evaluate('(1.5*wt/E_bar_now*pmu/(1+delta)+wt*(1-tau))/beta')
        else:
            mult_pens=ne.evaluate('(1.0*wt/E_bar_now*pmu/(1+delta)+wt*(1-tau))/beta')
        
        
        #How much consumption today? Use Euler equation
        ce=policyC[t+1,:,:]*mult
        
       
        #How much work? This follows from the FOC
        
        if (t+1<=R):
            he=ne.evaluate('maxHours-(mult_pens*(ce**(-gamma_c)))**(-1/gamma_h)')
            if (policy):
                he_m=ne.evaluate('maxHours-((1.0*wt/E_bar_now*pmu/(1+delta)+wt*(1-tau))/beta*(ce**(-gamma_c)))**(-1/gamma_h)')
                he[(1.5*he*wt/E_bar_now>1)]=he_m[(1.5*he*wt/E_bar_now>1)]
                
        if (t+1>R):he=np.zeros((numPtsA, numPtsP))
                  
        #How much assets? Just use the BC!
        ae=ne.evaluate('(agrid_box-wt*he*(1-tau)-y_N+ce)/(1+r)')
        
        #How much points should you have?
        if (t+1>R): pe=pgrid_box
        if (t+1<=R):        
            if policy:
                hem=np.maximum(np.minimum(1.5*he*wt/E_bar_now,np.ones(np.shape(he))),he*wt/E_bar_now)
                pe=ne.evaluate('pgrid_box-hem')
            else:
                pe=ne.evaluate('pgrid_box-    he*wt/E_bar_now')
            
       
        # Now, back on the main grid(speed can be improved below...)
        policyC[t,:,:],policyA1[t,:,:]=solveEulerEquation2(agrid,agrid_box,pgrid_box,ae,ce,pe,numPtsP,numPtsA,\
                                               r,wt,y_N,tau,gamma_c,gamma_h,\
                                               beta,maxHours,t,R,pgrid,rho,E_bar_now,\
                                               mult_pens,mgrid)
            
        #Obtain optimal hours (if retired, 0 hours)
        Pc=policyC[t,:,:] #Needed for computation below
        if (t+1<=R):
                 policyh[t,:,:]=  \
                 ne.evaluate('maxHours-(mult_pens*(Pc**(-gamma_c)))**(-1/gamma_h)')
                 
        #Update marginal utility of having more pension points
        if (t+1>R): 
            pmutil[t,:,:]=ne.evaluate('pmu/(1+delta)+rho*Pc**(-gamma_c)')
        else:
            pmutil[t,:,:]=ne.evaluate('pmu/(1+delta)')
               
      
#@njit(parallel=True)           
def solveEulerEquation2(agrid,agrid_box,pgrid_box,ae,ce,pe,numPtsP,numPtsA,r,wt,y_N,tau,gamma_c,
                        gamma_h,beta,maxHours,t,R,pgrid,rho,E_bar_now,mult_pens,mgrid):
    
    pCv,pAv,pPv,pC,pA,pP=np.empty((6,numPtsA,numPtsP),dtype=np.float64)
    where=np.empty((numPtsA,numPtsP),dtype=np.bool_)
    
    
    for i in prange(numPtsP):           
        pCv[:,i]=np.interp(agrid, ae[:,i],ce[:,i])
        pAv[:,i]=np.interp(agrid, ae[:,i],agrid)
        pPv[:,i]=np.interp(agrid, ae[:,i],pe[:,i])
        
    for j in prange(numPtsA):           
        pC[j,:]=np.interp(pgrid, pe[j,:],pCv[j,:])
        pA[j,:]=np.interp(pgrid, pe[j,:],pAv[j,:])
        pP[j,:]=np.interp(pgrid, pe[j,:],pPv[j,:])
     
    
    #Get where constraints are binding
    for j in prange(numPtsA):
        for i in prange(numPtsP): 
            where[j,i]=(pA[j,i]<=agrid[0])
            
    #Where constraints are binding, obtain consumption
    for j in prange(numPtsA):
        for i in prange(numPtsP): 
            tup=(agrid[j],r,y_N,gamma_c,gamma_h,maxHours,mult_pens[j,i],wt,tau)
            if where[j,i]:
                if (t+1>R):
                    pC[j,i]=(1+r)*agrid[j]+y_N+pgrid[j]*rho
                else:
                    hmin=agrid[j]*(1+r)
                    hmax=agrid[j]*(1+r)+y_N+maxHours*wt*(1-tau)
                    pC[j,i]=brentq(minim,hmin,hmax,args=tup)#[0]
                    
    return pC,pA


#@njit
def minim(x,a,r,y_N,gamma_c,gamma_h,maxHours,mult_pens,w,tau):
    h=-np.power(np.power(x,-gamma_c)*mult_pens,-1/gamma_h)+maxHours
    return (1+r)*a+w*h*(1-tau)+y_N-x
    
 
