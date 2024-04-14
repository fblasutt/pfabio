# this is the key program to solve the model

# import packages
import numpy as np
from consav import linear_interp,upperenvelope
from numba import njit,prange
import co
uppere=upperenvelope.create(co.log)

def solveEulerEquation(p,model='baseline'):
    
    
    #Translate keyword for model into number
    if model=='baseline':      reform=0
    if model=='pension reform':reform=1
    

    #Initiate some variables
    policyA1,policyC,pr,V,policyp,pmutil= np.zeros((6,p.T,p.nwls,p.NA, p.NP,p.nw,p.nq))-1e8
    EV=np.zeros((p.T,p.NA, p.NP,p.nw,p.nq))
    
    #Precompute after-tax income and points for a given decision
    p.income, p.points, p.taxes, p.taxes_mod = co.compute_atax_income_points(p.tax,p.tbase,p.T,p.R,p.nwls,p.nw,p.NP,p.τ,\
                                                          p.add_points,p.points_base,p.wls,p.w,\
                                                          p.E_bar_now,p.Pmax,p.wls_point,p.y_N,p.pgrid,p.ρ)
    
    #Call the routing to solve the model
    solveEulerEquation1(policyA1, policyC, policyp,V,EV,pmutil,pr,reform,
                        p.r,p.δ,p.R,p.τ,p.q,p.amin,p.wls,p.nwls,
                        np.array(p.w),p.income,p.points,p.agrid,p.y_N,p.T,p.NA,p.nw,p.σ,
                        p.NP,p.pgrid,p.ρ,p.E_bar_now,
                        p.Pmax,p.add_points,p.nq,p.wls_point,p.q_grid,p.points_base)
                        

    
    return {'A':policyA1,'c':policyC,'V':V,'V1':EV,'p':policyp,'pr':pr,'model':reform}

#@profile
@njit(parallel=True)
def solveEulerEquation1(policyA1, policyC, policyp,V,EV,pmutil,pr,reform,
                        r,δ,R,τt,q,amin,wls,nwls,
                        w,income,points,agrid,y_N,T,NA,nw,σ,
                        NP,pgrid,ρ,E_bar_now,
                        Pmax,add_points,nq,wls_point,q_grid,points_base):
    
    
    ce,pe,ae,ce_bc,pe_bc,ae_bc,aem,cem,vem=np.zeros((9,nwls,NA, NP, nw, nq))
    Ec=np.zeros((NA, NP,nw,nq))

    
    #Last period decisions below
    for ia in prange(NA): 
        for ip in range(NP): 
            for iw in range(nw): 
                for iq in range(nq): 
                    idx=(T-1,0,ia,ip,iw,iq);idw = (T-1,0,iw,ip)
                    
                    policyA1[idx] = 0.0   # optimal savings
                    policyC[idx] = agrid[ia]*(1+r) + income[idw]
                    policyp[idx] = pgrid[ip] # optimal consumption                                
                    V[idx]=co.log(policyC[idx],q_grid[iq,0,iw]) 
   
    #Loop backward and solve the model by backward induction
    for t in range(T-2,-2,-1):
         

        #Integration (beginning of period expectation, before taste shocks are realized)
        expectation(t,NA,NP,nw,nq,V,EV[t+1,...],σ,pr,Ec,policyC)    
        if t==-1:break
        
        
        #Find policy and value function on the endogenous grid
        for ia in prange(NA): 
            for iw in range(nw): 
                for iq in range(nq): 
                    for i in range(nwls):                           
                        if ((t<R) | ((t>=R) & (i==0))):
                            
                            #Use the euler equation + BC to find optimal consumtpion
                            #ce for (endogenous) grid of assets ae
                            for ip in range(NP): 
                                  
                                idx=(i,ia,ip,iw,iq);idw = (t,i,iw,ip);idp = (t,i,iw,reform) 
    
                                ce[idx]=Ec[ia,ip,iw,iq]*(1+δ)/(1+r)#Euler equation                           
                                pe[idx]=pgrid[ip]-points[idp] if (t+1<=R) else pgrid[ip]
                                ae[idx]=(agrid[ia]-income[idw]+ce[idx])/(1+r)#BC
                            
                          
                            #Interpolate to get policy related to right points in t
                            idx=(i,ia,slice(None),iw,iq);tidx=(t,i,ia,slice(None),iw,iq);
                            t1idx=(t+1,ia,slice(None),iw,iq);idp = (t,i,iw,reform) 
                            
                            #Get policy functions conistent with pension points
                            linear_interp.interp_1d_vec(pe[idx],ce[idx],pgrid,cem[idx])
                            linear_interp.interp_1d_vec(pe[idx],ae[idx],pgrid,aem[idx])                        
                            
                            policyp[tidx]=pgrid+points[idp] if (t+1<=R) else pgrid
                            linear_interp.interp_1d_vec(pgrid,EV[t1idx],policyp[tidx],vem[idx])
                       
                                            
        #Interpolate to get decisions on grid + apply upper-envelop algorithm                
        for ip in prange(NP):
            for iw in range(nw):
                for iq in range(nq):
                    for i in range(nwls):
                        if ((t<R) | ((t>=R) & (i==0))):
                            
    
                            tidx=(t,i,slice(None),ip,iw,iq);idx=(i,slice(None),ip,iw,iq);idw = (t,i,iw,ip)

                            uppere(agrid,agrid+cem[idx],cem[idx],vem[idx]/(1+δ),agrid*(1+r)+income[idw],policyC[tidx],V[tidx],*(q_grid[iq,i,iw],))
                            policyA1[tidx] =agrid*(1+r)+income[idw]-policyC[tidx]
                            
     
@njit(parallel=True)
def expectation(t,NA,NP,nw,nq,V,EV,σ,pr,Ec,policyC):                
        #Get variables useful for next iteration t-1
        for ia in prange(NA):
            for ip in range(NP):
                for iw in range(nw):
                    for iq in range(nq):
                    
                        idx=(t+1,slice(None),ia,ip,iw,iq)
                        lc=np.max(V[idx])/σ#local normalizing variable
                        EV[ia,ip,iw,iq] = σ*np.euler_gamma+σ*(lc+np.log(np.sum(np.exp(V[idx]/σ-lc)))  )
                        pr[idx]=np.exp(V[idx]/σ-(EV[ia,ip,iw,iq]-σ*np.euler_gamma)/σ)
                        Ec[ia,ip,iw,iq] = np.sum(pr[idx]*policyC[idx])
 
