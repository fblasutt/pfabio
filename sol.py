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
    policyA1,pr,V,policyp,policyp_exp,pmutil= np.zeros((6,p.T,p.nwls,p.NA, p.NP,p.nw,p.nq,2))-1e8  
    policyC =  np.zeros((p.T,p.nwls,p.NA, p.NP,p.nw,p.nq,2))+1e-10 
    EV,EVb=np.zeros((2,p.T,p.NA, p.NP,p.nw,p.nq,2)) 
     
    #Precompute after-tax income and points for a given decision 
    p.income, p.points,p.points_exp, p.taxes, p.income_mod = co.compute_atax_income_points(p.tax,p.tbase,p.T,p.R,p.nwls,p.nw,p.NP,p.τ,\
                                                          p.add_points,p.add_points_exp,p.points_base,p.wls,p.w,\
                                                          p.E_bar_now,p.Pmax,p.wls_point,p.y_N,p.pgrid,p.ρ) 
     
    #Call the routing to solve the model 
    solveEulerEquation1(policyA1, policyC, policyp,policyp_exp,V,EV,EVb,pmutil,pr,reform, 
                        p.r,p.δ,p.R,p.α,p.q,p.nwls, 
                        np.array(p.w),p.income,p.points,p.points_exp,p.agrid,p.T,p.NA,p.nw,p.σ, 
                        p.NP,p.pgrid, 
                        p.nq,p.wls_point,p.q_grid,p.Π, 
                        p.age_ret,p.points_mult) 
                         
 
     
    return {'A':policyA1,'c':policyC,'V':V,'V1':EV,'p':policyp,'p_exp':policyp_exp,'pr':pr,'model':reform} 
 
#@profile 
@njit(parallel=True) 
def solveEulerEquation1(policyA1, policyC, policyp,policyp_exp,V,EV,EVb,pmutil,pr,reform, 
                        r,δ,R,α,q,nwls, 
                        w,income,points,points_exp,agrid,T,NA,nw,σ, 
                        NP,pgrid, 
                        nq,wls_point,q_grid,Π, 
                        age_ret,points_mult): 
     
     
    ce,pe,ae,ce_bc,pe_exp,aem,cem,vem,aem=np.zeros((9,nwls,NA, NP, nw, nq, 2))-1e8 
     
 
     
    #Last period decisions below 
    for ia in prange(NA):  
        for ip in range(NP):  
            for iw in range(nw):  
                for iq in range(nq):  
 
                    idx=(T-1,0,ia,ip,iw,iq,1);idw = (T-1,0,iw,ip,1) 
                     
                    policyA1[idx] = agrid[ia]*(1+r)   # optimal savings 
                    policyC[idx] = agrid[ia]*(1+r)+1e-9 
                    policyp[idx] = pgrid[ip] # optimal consumption 
                    policyp_exp[idx] = pgrid[ip] # optimal consumption                                 
                    V[idx]=co.log(policyC[idx],q_grid[iq,0,iw],α) 
    
    #Loop backward and solve the model by backward induction 
    for t in range(T-2,-2,-1): 
          
 
        #Integration (beginning of period expectation, before taste shocks are realized) 
        E_mutil_c=np.zeros((NA, NP,nw,nq, 2)) 
        expectation(t,NA,NP,nw,nq,V,EV[t+1,...],EVb[t+1,...],σ,α,pr,E_mutil_c,policyC,Π[t])     
        if t==-1:break 
         
         
        #Find policy and value function on the endogenous grid 
        for ia in prange(NA):  
            for iw in range(nw):  
                for iq in range(nq):  
                    for i in range(nwls):       
                        for ir in range(2):  
                             
                            if (((ir==0) & (t<=age_ret[-1]))    | #cannot retire yet case                              
                                ((ir==1) & (t>=age_ret[0]) &  (i==0))):    #have already retired 
                                 
                                irp = 1 if ((i==0) & (t>=age_ret[0]))  | (t>age_ret[-1]) else ir 
                                 
                                #Use the euler equation + BC to find optimal consumtpion 
                                #ce for (endogenous) grid of assets ae 
                                for ip in range(NP):  
                                       
                                    idx=(i,ia,ip,iw,iq,ir) 
                                    idp = (t,i,iw,reform,ir)   
         
                                    ce[idx]=(E_mutil_c[ia,ip,iw,iq,irp]*(1+r)/(1+δ))**(-1/α)#Euler equation                            
                                     
                                    ae[idx]=(agrid[ia]-income[t,i,iw,ip,ir]+ce[idx])/(1+r)#BC 
                                     
                                    if   ((ir==0)  & (t<age_ret[0])):       pe[idx] =  pgrid[ip]-points[idp] 
                                    elif   ((ir==0)  & (t>=age_ret[0]) & (i>0)):       pe[idx] =  pgrid[ip]-points[idp]                                     
                                    elif ((ir==0)  & (t>=age_ret[0]) & (i==0)):       pe[idx] =  pgrid[ip]/points_mult[t-R+2] 
                                    elif  (ir==1)  & (t>=age_ret[0]):  pe[idx] =  pgrid[ip] 
                                     
                                    if   ((ir==0)  & (t<age_ret[0])):       pe_exp[idx] =  pgrid[ip]-points_exp[idp] 
                                    elif   ((ir==0)  & (t>=age_ret[0]) & (i>0)):       pe_exp[idx] =  pgrid[ip]-points_exp[idp]                                     
                                    elif ((ir==0)  & (t>=age_ret[0]) & (i==0)):       pe_exp[idx] =  pgrid[ip]/points_mult[t-R+2] 
                                    elif  (ir==1)  & (t>=age_ret[0]):  pe_exp[idx] =  pgrid[ip] 
                                  
                                                               
                                #Interpolate to get policy related to right points in t 
                                idx=(i,ia,slice(None),iw,iq,ir);tidx=(t,i,ia,slice(None),iw,iq,ir); 
                                t1idx=(t+1,ia,slice(None),iw,iq,irp);idp = (t,i,iw,reform,ir) 
                                 
                                #Get policy functions conistent with pension points 
                                linear_interp.interp_1d_vec(pe[idx],ce[idx],pgrid,cem[idx]) 
                                linear_interp.interp_1d_vec(pe[idx],ae[idx],pgrid,aem[idx])    
                                 
                                if   ((ir==0)  & (t<age_ret[0])):       policyp[tidx] =  pgrid+points[idp] 
                                elif ((ir==0)  & (t>=age_ret[0]) & (i>0)):       policyp[tidx] =  pgrid+points[idp] 
                                elif ((ir==0)  & (t>=age_ret[0]) & (i==0)):       policyp[tidx] =  pgrid*points_mult[t-R+2] 
                                elif  (ir==1)  & (t>=age_ret[0]):  policyp[tidx] =  pgrid 
                                 
                                if   ((ir==0)  & (t<age_ret[0])):       policyp_exp[tidx] =  pgrid+points_exp[idp] 
                                elif ((ir==0)  & (t>=age_ret[0]) & (i>0)):       policyp_exp[tidx] =  pgrid+points_exp[idp] 
                                elif ((ir==0)  & (t>=age_ret[0]) & (i==0)):       policyp_exp[tidx] =  pgrid*points_mult[t-R+2] 
                                elif  (ir==1)  & (t>=age_ret[0]):  policyp_exp[tidx] =  pgrid 
                                 
                                                             
                                linear_interp.interp_1d_vec(pgrid,EV[t1idx],policyp[tidx],vem[i,ia,slice(None),iw,iq,ir]) 
                                 
                                
                                                                           
        #Interpolate to get decisions on grid + apply upper-envelop algorithm    
              
        for ip in prange(NP): 
            for iw in range(nw): 
                for iq in range(nq): 
                    for ir in range(2):  
                        for i in range(nwls): 
                            
                            if (((ir==0) & (t<=age_ret[-1]))    | #cannot retire yet case                              
                                ((ir==1) & (t>=age_ret[0]) &  (i==0))):    #have already retired 
                             
     
                                tidx=(t,i,slice(None),ip,iw,iq,ir);idx=(i,slice(None),ip,iw,iq,ir); 
                                idw = (t,i,iw,ip,ir)  
                             
                                 
                                uppere(agrid,agrid+cem[idx],cem[idx],vem[idx]/(1+δ),agrid*(1+r)+income[idw],policyC[tidx],V[tidx],*(q_grid[iq,i,iw],α,)) 
                                policyA1[tidx] =agrid*(1+r)+income[idw]-policyC[tidx] 
                                 
                                 
                                 
                             
        
@njit(parallel=True) 
def expectation(t,NA,NP,nw,nq,V,EV,EVb,σ,α,pr,E_mutil_c,policyC,Πt):                 
    #Get variables useful for next iteration t-1 
    for ia in prange(NA): 
        for ip in range(NP): 
            for iw in range(nw): 
                for iq in range(nq): 
                    for ir in range(2): 
                     
                        # Expected value in t+1 - AFTER shock iw' is realized - before taste shock takes place 
                        idx=(t+1,slice(None),ia,ip,iw,iq, ir) 
                        lc=np.max(V[idx])/σ#local normalizing variable 
                        EVb[ia,ip,iw,iq,ir] = σ*np.euler_gamma+σ*(lc+np.log(np.sum(np.exp(V[idx]/σ-lc)))  ) 
                        pr[idx]=np.exp(V[idx]/σ-(EVb[ia,ip,iw,iq,ir]-σ*np.euler_gamma)/σ) 
  
 
    for ia in prange(NA): 
        for ip in range(NP): 
            for iw in range(nw): 
                for iq in range(nq):              
                    for ir in range(2): 
                        for iwp in range(nw): 
                                               
                            idxp=(t+1,slice(None),ia,ip,iwp,iq, ir) 
                            
                            # Expected value and marg util from C in t+1 BEFORE shock iw' is realized, given iw is current shock 
                            E_mutil_c[ia,ip,iw,iq,ir] += Πt[iwp,iw]*np.sum(pr[idxp]*policyC[idxp]**(-α)) 
                            EV[ia,ip,iw,iq,ir]        += Πt[iwp,iw]*EVb[ia,ip,iwp,iq,ir] 
  
