# this is the key program to solve the model

# import packages
# Time
import time
import numpy as np
from consav.grids import nonlinspace # grids
from consav import linear_interp,upperenvelope
import numexpr as ne
import upperenvelop
from numba import njit,prange
import co
import time


uppere=upperenvelope.create(co.log)

def solveEulerEquation(p,model='baseline'):
    
    
    #Translate keyword for model into number
    if model=='baseline':      reform=0
    if model=='pension reform':reform=1
    
    #Start counting time
    time_start = time.time()
        
    #Initiate some variables
    policyA1,policyC,pr,V,policyp,pmutil= np.zeros((6,p.T,p.nwls,p.NA, p.NP,p.nw,p.nq))-1e8
    V1=np.zeros((p.T,p.NA, p.NP,p.nw,p.nq))
    holes=np.ones((p.T,p.nwls,p.NA, p.NP,p.nw,p.nq))
    #Call the routing to solve the model
    solveEulerEquation1(policyA1, policyC, policyp,V,V1,pmutil,pr,holes,reform,
                        p.r,p.δ,p.γc,p.R,p.τ,p.β,p.q,p.amin,p.wls,p.nwls,
                        np.array(p.w),p.agrid,p.y_N,p.γh,p.T,p.NA,p.nw,p.σ,
                        p.NP,p.pgrid,p.ρ,p.E_bar_now,p.q_mini,
                        p.Pmax,p.add_points,p.nq,p.ζ,p.wls_point,p.q_grid,p.points_base)
                        

    #End timer and print elapsed time
    elapsed = time.time() - time_start    
    #print('Finished, Reform =', reform, 'Elapsed time is', elapsed, 'seconds')   
    
    return {'A':policyA1,'c':policyC,'V':V,'V1':V1,'p':policyp,'pr':pr,'model':reform,'holes':holes}

#@profile
#@njit(parallel=True)
def solveEulerEquation1(policyA1, policyC, policyp,V,V1,pmutil,pr,holes,reform,
                        r,δ,γc,R,τt,β,q,amin,wls,nwls,
                        w,agrid,y_N,γh,T,NA,nw,σ,
                        NP,pgrid,ρ,E_bar_now,q_mini,
                        Pmax,add_points,nq,ζ_temp,wls_point,q_grid,points_base):
    
    """ Use the method of endogenous gridpoint in 2 dimensions to solve the model.
        Source: JDruedahlThomas and Jørgensen (2017). This method is robust to
        non-convexities
        
        Note that the grids for assets constrained and not constrained
        are different to avoid extrapolation
    """

    
    ce,pe,ae,ce_bc,pe_bc,ae_bc=np.zeros((6,nwls,NA, NP, nw, nq))
    c1=np.zeros((NA, NP,nw,nq))
    

    #Grid for consumption
    cgrid=np.linspace(1e-10,agrid[-1]*(1+r)+np.max(y_N)+np.max(w)*(1-np.min(τt)),NA)
    
    #Last period decisions below
    for ia in prange(NA): 
        for ip in range(NP): 
            for iw in range(nw): 
                for iq in range(nq): 
                    idx=(T-1,0,ia,ip,iw,iq)
                    
                    policyA1[idx] = 0.0   # optimal savings
                    policyC[idx] = agrid[ia]*(1+r) + co.after_tax_income(ρ*pgrid[ip],y_N[T-1,iw],E_bar_now,wls_point[0],τt[T-1],False)   
                    policyp[idx] = pgrid[ip] # optimal consumption                                
                    V[idx]=co.log(policyC[idx],q_grid[iq,0,iw]) 
   
    #Decisions below
    for t in range(T-2,-2,-1):
         

        #Get variables useful for next iteration t-1
        expectation(t,NA,NP,nw,nq,V,V1[t+1,...],σ,γc,pr,c1,policyC)    
       
        
        if t==-1:break

        
        τ=τt[t]
        #y_Nt=y_N_box[t,:,:,:]
        policy=((t >=8) & (t <=11) & (reform==1))
        
        ζ = ζ_temp if t<=11 else 0.0
        
        #Multiplier of points based on points
        mp=add_points if policy else points_base
        
            ###################################################################
            #Not retired case
            ###################################################################

        for ia in prange(NA): 
            for ip in range(NP): 
                for iw in range(nw): 
                    for iq in range(nq): 
                        for i in range(nwls):                           
                            if ((t+1<=R) | ((t+1>R) & (i==0))):
                                      
                                idx=(i,ia,ip,iw,iq)
                                tax=τ      if (i>1) else 0.0
                                
                                if (t+1<=R): income  = co.after_tax_income(w[t,i,iw]*wls[i],y_N[t,iw],E_bar_now,wls_point[i],tax,False) 
                                else:        income  = co.after_tax_income(ρ*pgrid[ip]     ,y_N[t,iw],E_bar_now,wls_point[i],tax,False)
                                

                                ce[idx]=c1[ia,ip,iw,iq]*(1+δ)/(1+r)
                                pe[idx]=pgrid[ip]-np.maximum(np.minimum(mp*wls[i]*w[t,i,iw]/E_bar_now,Pmax),wls[i]*w[t,i,iw]/E_bar_now)*wls_point[i] if (t+1<=R) else pgrid[ip]
                                ae[idx]=(agrid[ia]-income+ce[idx])/(1+r)#Savings
                                
                          

        ################################################
        # Now interpolate to be back on grid...
        ###############################################

        
        #Not retired

        
        # a. find policy for constrained and unconstrained choices
        
        # #tic=time.time();
        # #Unconstrained
        aem=np.ones(ae.shape);cem=np.ones(ae.shape);vem=np.ones(ae.shape)
        for ia in prange(NA):
            for iw in range(nw):
                for iq in range(nq):
                    for i in range(nwls):
                        if ((t+1<=R) | ((t+1>R) & (i==0))):
                            
                            
                            idx=(i,ia,slice(None),iw,iq);tidx=(t,i,ia,slice(None),iw,iq);t1idx=(t+1,ia,slice(None),iw,iq)
                            
                            #Get policy functions conistent with pension points
                            linear_interp.interp_1d_vec(pe[idx],ce[idx],pgrid,cem[idx])
                            linear_interp.interp_1d_vec(pe[idx],ae[idx],pgrid,aem[idx])
                            
                            
                            policyp[tidx]=pgrid+np.maximum(np.minimum(mp*wls[i]*w[t,i,iw]/E_bar_now,Pmax),wls[i]*w[t,i,iw]/E_bar_now)*wls_point[i] if (t+1<=R) else pgrid
                            linear_interp.interp_1d_vec(pgrid,V1[t1idx],policyp[tidx],vem[idx])
                           
        
        
                            
                            
        for iw in prange(nw):
            for ip in range(NP):
                for iq in range(nq):
                    for i in range(nwls):
                        if ((t+1<=R) | ((t+1>R) & (i==0))):
                            
                            
                            tax=τ      if (i>1) else 0.0
                            
                            tidx=(t,i,slice(None),ip,iw,iq);idx=(i,slice(None),ip,iw,iq);idxp=(i,0,ip,iw,iq)
                            t1idx=(t+1,slice(None),slice(None),iw,iq)
                            #Preliminaries
                           
                           
                             
                            
                            if (t+1<=R): income  = co.after_tax_income(w[t,i,iw]*wls[i],y_N[t,iw],E_bar_now,wls_point[i],tax,False) 
                            else:        income  = co.after_tax_income(ρ*pgrid[ip]     ,y_N[t,iw],E_bar_now,wls_point[i],tax,False)
                            cash=agrid*(1+r)+income;cashe=aem[idx]*(1+r)+income
                            
                            
                            # linear_interp.interp_1d_vec(aem[idx],cem[idx],agrid,policyC[tidx])
                            # constrained = cash<=cashe[0]
                            # policyC[tidx][constrained]=cash[constrained]
                            # policyA1[tidx] =cash-policyC[tidx]
                            # EV=np.ones(NA)
                            # linear_interp.interp_2d_vec(agrid,pgrid,V1[t1idx],policyA1[tidx],policyp[tidx][0]*np.ones(NA),EV)
                            # V[tidx]=co.log(policyC[tidx],q_grid[iq,i,iw])+EV/(1+δ)
                            
                            uppere(agrid,cashe,cem[idx],vem[idx]/(1+δ),cash,policyC[tidx],V[tidx],*(q_grid[iq,i,iw],))
                            policyA1[tidx] =cash-policyC[tidx]
                            
                            #if (policy) & (i==2):tax[1,1]=3
                            
    
                     
     
@njit(parallel=True)
def expectation(t,NA,NP,nw,nq,V,V1,σ,γc,pr,c1,policyC):                
        #Get variables useful for next iteration t-1
        for ia in prange(NA):
            for ip in range(NP):
                for iw in range(nw):
                    for iq in range(nq):
                    
                        idx=(t+1,slice(None),ia,ip,iw,iq)
                        lc=np.max(V[idx])/σ#local normalizing variable
                        V1[ia,ip,iw,iq] = σ*np.euler_gamma+σ*(lc+np.log(np.sum(np.exp(V[idx]/σ-lc)))  )
                        pr[idx]=np.exp(V[idx]/σ-(V1[ia,ip,iw,iq]-σ*np.euler_gamma)/σ)
                        c1[ia,ip,iw,iq] = np.sum(pr[idx]*policyC[idx])
 
