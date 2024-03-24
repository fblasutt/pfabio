# this is the key program to solve the model

# import packages
# Time
import time
import numpy as np
from consav.grids import nonlinspace # grids
from consav import linear_interp
import numexpr as ne
import upperenvelop
from numba import njit,prange
import co
import time


def solveEulerEquation(p,model='baseline'):
    
    
    #Translate keyword for model into number
    if model=='baseline':      reform=0
    if model=='pension reform':reform=1
    
    #Start counting time
    time_start = time.time()
        
    #Initiate some variables
    policyA1,policyC,pr,V,policyp,pmutil= np.zeros((6,p.T,p.nwls,p.NA, p.NP,p.nw,p.nq))-1e8
    V1=np.zeros((p.T,p.NA, p.NP,p.nw,p.nq))
    holes=np.ones((p.T,p.nwls,p.NA, p.NP,p.nw,p.nq,2))
    #Call the routing to solve the model
    solveEulerEquation1(policyA1, policyC, policyp,V,V1,pmutil,pr,holes,reform,
                        p.r,p.δ,p.γc,p.R,p.τ,p.β,p.q,p.amin,p.wls,p.nwls,
                        np.array(p.w),p.agrid,p.y_N,p.γh,p.T,p.NA,p.nw,p.σ,
                        p.NP,p.pgrid,p.ρ,p.E_bar_now,p.q_mini,
                        p.Pmax,p.add_points,p.nq,p.ζ,p.wls_point,p.q_grid)
                        

    #End timer and print elapsed time
    elapsed = time.time() - time_start    
    #print('Finished, Reform =', reform, 'Elapsed time is', elapsed, 'seconds')   
    
    return {'A':policyA1,'c':policyC,'V':V,'V1':V1,'p':policyp,'pr':pr,'model':reform}

#@profile
@njit(parallel=True)
def solveEulerEquation1(policyA1, policyC, policyp,V,V1,pmutil,pr,holes,reform,
                        r,δ,γc,R,τt,β,q,amin,wls,nwls,
                        w,agrid,y_N,γh,T,NA,nw,σ,
                        NP,pgrid,ρ,E_bar_now,q_mini,
                        Pmax,add_points,nq,ζ_temp,wls_point,q_grid):
    
    """ Use the method of endogenous gridpoint in 2 dimensions to solve the model.
        Source: JDruedahlThomas and Jørgensen (2017). This method is robust to
        non-convexities
        
        Note that the grids for assets constrained and not constrained
        are different to avoid extrapolation
    """

    
    ce,pe,ae,ce_bc,pe_bc,ae_bc=np.zeros((6,nwls,NA, NP, nw, nq))
    c1=np.zeros((NA, NP,nw,nq))
    

    #Grid for consumption
    cgrid=np.linspace(agrid[0]*0.001,agrid[-1]*(1+r)+np.max(y_N)+np.max(w)*(1-np.min(τt)),NA)
    
    #Last period decisions below
    for ia in prange(NA): 
        for ip in range(NP): 
            for iw in range(nw): 
                for iq in range(nq): 
                    idx=(T-1,0,ia,ip,iw,iq)
                    
                    policyA1[idx] = 0.0   # optimal savings
                    policyC[idx] = agrid[ia]*(1+r) + co.after_tax_income(ρ*pgrid[ip],y_N[T-1,iw],E_bar_now,wls_point[0],τt[T-1],False)   
                    policyp[idx] = pgrid[ip] # optimal consumption                              
                    pmutil[idx]=1.0/policyC[idx]   # mu of more pension points        
                    V[idx]=np.log(policyC[idx]) - q_grid[iq,0,iw]
   
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
        mp=add_points if policy else 1.0
        
            ###################################################################
            #Not retired case
            ###################################################################

        for ia in prange(NA): 
            for ip in range(NP): 
                for iw in range(nw): 
                    for iq in range(nq): 
                        for i in range(nwls):
                                      
                            idx=(i,ia,ip,iw,iq)
                           
                            if (t+1<=R):
                                                          
                                
                                #Unconstrained
                                ce[idx]=c1[ia,ip,iw,iq]*((1+r)/(1+δ))**-1
                                pe[idx]=pgrid[ip]-np.maximum(np.minimum(mp*wls[i]*w[t,i,iw]/E_bar_now,Pmax),wls[i]*w[t,i,iw]/E_bar_now)*wls_point[i]   #Pens. points
                                ae[idx]=(agrid[ia]-co.after_tax_income(w[t,i,iw]*wls[i],y_N[t,iw],E_bar_now,wls_point[i],τ)+ce[idx])/(1+r)#Savings
                                
                                #Constrained (assets)
                                pe_bc[idx]=pgrid[ip]-  np.maximum(np.minimum(mp*wls[i]*w[t,i,iw]/E_bar_now,Pmax),wls[i]*w[t,i,iw]/E_bar_now)*wls_point[i]      #Pens. points
                                             
                                ce_bc[idx]=cgrid[ia]
                                ae_bc[idx]=(ce_bc[idx] - co.after_tax_income(w[t,i,iw]*wls[i],y_N[t,iw],E_bar_now,wls_point[i],τ)+amin)/(1+r)#Savings
                                        
                            else:
                               
                              if i==0:
                                
                                ###################################################################
                                #Retired case
                                ###################################################################
                                ce[idx]=c1[ia,ip,iw,iq]*((1+r)/(1+δ))**-1#Euler equation
                                pe[idx]=pgrid[ip]                                  #Pens. points 
                                ae[idx]=(agrid[ia]-co.after_tax_income(ρ*pe[idx],y_N[t,iw],E_bar_now,wls_point[0],τ,False)+ce[idx])/(1+r)#Savings
                  

        ################################################
        # Now interpolate to be back on grid...
        ###############################################

        
        #Not retired
        if (t+1<=R):     
            
            # a. find policy for constrained and unconstrained choices
            
            #tic=time.time();
            #Unconstrained
            for i in prange(nwls):
  
                #Penalty for working?
                q_pen=q_grid# if i>0 else np.zeros(q_grid.shape)
                
                #Mini jobs below
                #modify for the mini-jobs case
                tax=τ      if (i>1) else 0.0
                q_min=0.0  if i!=1 else q_mini
                
                #Computation below
                upperenvelop.compute(policyC[t,i,...],policyA1[t,i,:,:,:],policyp[t,i,:,:,:],V[t,i,...],holes[t,i,...], 
                        pe[i,...],ae[i,...],ce[i,...],pe_bc[i,...],ae_bc[i,...],ce_bc[i,...],#computed above... 
                        i, # which foc to take in upperenvelop 
                        V1[t+1,...], 
                        γc,γh,ρ,agrid,pgrid,β,r,w[t,i,:],tax,y_N[t,:],E_bar_now,Pmax,δ,q_pen,amin,wls[i],mp,q_min,wls_point[i],ζ)  
     

        # #Retired
        else:
            
            for i in prange(NP):
                for j in range(nw):
                    for iq in range(nq):
                        
                        idx=(0,slice(None),i,j,iq)
     
                        policyA1[t,*idx]=np.interp(agrid, ae[idx],agrid)
                        EV=np.interp(policyA1[t,*idx],agrid,V1[t+1,*idx[1:]])
                        
                        for ia in range(NA): 
                            idx_ia=(0,ia,i,j,iq)
                                                 
                            policyC[t,*idx_ia] =agrid[ia]*(1+r)+co.after_tax_income(ρ*pe[idx_ia],y_N[t,j],E_bar_now,wls_point[0],τ,False)-policyA1[t,*idx_ia]
                            policyp[t,*idx_ia]=pgrid[i]
                            V[t,*idx_ia]=co.log(policyC[t,*idx_ia])-q_grid[iq,0,j]+EV[ia]/(1+δ)
                     
     
@njit
def expectation(t,NA,NP,nw,nq,V,V1,σ,γc,pr,c1,policyC):                
        #Get variables useful for next iteration t-1
        for i_n in prange(NA):
            for i_m in range(NP):
                for i_w in range(nw):
                    for iq in range(nq):
                    
                        idx=(t+1,slice(None),i_n,i_m,i_w,iq)
                        lc=np.max(V[idx])/σ#local normalizing variable
                        V1[i_n,i_m,i_w,iq] = σ*np.euler_gamma+σ*(lc+np.log(np.sum(np.exp(V[idx]/σ-lc)))  )
                        pr[idx]=np.exp(V[idx]/σ-(V1[i_n,i_m,i_w,iq]-σ*np.euler_gamma)/σ)
                        c1[i_n,i_m,i_w,iq] = np.sum(pr[idx]*policyC[t+1,:,i_n,i_m,i_w,iq])
 