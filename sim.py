# run simulation for one indvidual without uncertainty 
 
import numpy as np 
from numba import njit,prange  
from consav import linear_interp 
import co 
 
#https://www.econforge.org/interpolation.py/ 
 
def simNoUncer_interp(p, model, Tstart=0, Astart=0.0, Pstart=0.0,izstart=0, Vstart= -1.0*np.ones((2,2,2,2,2)),cadjust=1.0): 
  
    np.random.seed(2)  
    p.q_sim = np.zeros(p.N,dtype=np.int32)         
    means = np.linspace(-p.ρq ,p.ρq ,p.nw//p.nzm) 
    for iw in range(p.nw): 
        
        iswage=(p.tw==iw) 
        iswagelen=np.sum(iswage) 
         
        p.q_sim[iswage]=np.minimum(np.maximum(np.array(np.random.uniform(0.0+means[iw//p.nzm],p.nq+means[iw//p.nzm],size=iswagelen),dtype=np.int32),0),p.nq-1) 
        #p.q_sim[iswage]=np.random.randint(0.0,p.nq-1,size=iswagelen) 
     
    p.q_sim=np.array(np.random.uniform(0.0,p.nq,size=p.N),dtype=np.int32)
 
     
    #Call the simulator 
    epath,ppath,cpath,apath,hpath,pepath,pepath2,pepath3,vpath,evpath,wpath,w_pr_path,v_pr_path,eataxpath,eataxpath_mod,iz,ir=\
        fast_simulate(Tstart,Astart,Pstart,izstart,Vstart,p.amax,p.T,p.N,p.agrid,p.pgrid,p.w,p.E_bar_now,p.Pmax,p.add_points,p.tw,p.ts,p.wls,p.nwls, 
                      p.δ,p.q_grid,p.σ,p.taxes,p.income_mod,p.income,p.Π,p.shock_z, 
                      model['A'],model['c'],model['p'],model['p_exp'],model['pr'],model['V'],model['V1'],model['model'],cadjust,p.wls_point,p.q_sim,p.points_base,p.R,p.r,p.y_N,p.τ,p.age_ret) 
     
    return {'wh':epath,'p':ppath,'c':cpath,'A':apath,'h':hpath,'pb':pepath, 'pb2':pepath2,'pb3':pepath3, 'v':vpath,'ev':evpath,'w':wpath,'wls_pr':w_pr_path,'v_pr':v_pr_path,'taxes':eataxpath,'income_mod':eataxpath_mod,'iz':iz,'ir':ir} 
     
@njit(parallel=True) 
def fast_simulate(Tstart,Astart,Pstart,izstart,Vstart,amax,T,N,agrid,pgrid,w,E_bar_now,Pmax,add_points,tw,ts,wls,nwls,δ,q,σ,taxes,income_mod,income,Π,shock_z, 
                  policyA1,policyC,policyP,policyP_exp,pr,V,V1,reform,cadjust,wls_point,q_sim,points_base,R,r,y_N,τ,age_ret): 
 
    # Arguments for output 
    cpath = np.nan+ np.zeros((T, N))           # consumption 
    hpath = np.zeros((T, N),dtype=np.int32)           # earnings path 
    pepath = np.nan+ np.zeros((T, N))           #corrected potints path 
     
    pepath2 = np.nan+ np.zeros((T, N))           #additional caregiver credits 
    pepath3 = np.zeros((T, N))           #additional caregiver credits 
    epath = np.zeros((T, N))            # earnings path 
    eataxpath = np.zeros((T, N))        #after tax earnings 
    epath_mod = np.zeros((T, N))        #after tax earnings 
    wpath = np.nan+ np.zeros((T, N))           # wage path 
    vpath = np.nan+ np.zeros((T, N))           # utility 
    evpath = np.nan+ np.zeros((T, N))           # expected utility 
    apath = np.nan+ np.zeros((T,N))        # assets at start of each period, decided 1 period ahead and so includes period T+1    
    ppath = np.nan+ np.zeros((T,N))        # points at start of each period, decided 1 period ahead and so includes period T+1 
    ppath_exp = np.nan+ np.zeros((T, N))           #corrected potints path 
    w_pr_path = np.zeros((T,N, nwls)) 
    v_pr_path = np.zeros((T,N, nwls)) 
    iz =np.zeros((T, N),dtype=np.int32)  
    ir =np.zeros((T, N),dtype=np.int32)  
     
    # Modified initial conditions 
    Ti=Tstart 
    apath[Ti, :] = Astart;  
    ppath[Ti, :] = pepath3[Ti, :] =  Pstart;  
    ppath_exp[Ti, :] = pepath3[Ti, :] =  Pstart;  
    iz[Ti, :] = izstart 
     
     
    #If Vstart if provided, adjust wealth so that utility matches Vstart 
    #if np.min(Vstart)!=-1.0:apath=adjust_wealth(Vstart, N, V, apath, ppath, amax, agrid, pgrid,Ti,tw,nwls) 
        
     
    # Obtain paths using the initial condition and the policy and value functions   
    for n in prange(N): 
        for t in range(Ti,T):  # loop through time periods for a pticular individual 
 
            if t>Ti: iz[t,n] = co.mc_simulate(iz[t-1,n],Π[t-1],shock_z[n,t])  
                      
             
            iq = q_sim[n]#[iz[t,n]][n] 
            #policy=((t >=4) & (t <=11))#& (reform==1)) 
            policy2=((t >=4) & (t <=11) & (reform==1)) 
            mp=add_points #if policy else 1.0 
            mp2=add_points if policy2 else points_base 
            mp3=add_points if t>=4 else points_base 
            
            ref = 1 if policy2 else 0
            
            #Get the discrete choices first... 
             
            #i=0 
             
            for i in range(nwls):               
                v_pr_path[t,n,i]=linear_interp.interp_2d(agrid,pgrid,V[t,i,:,:,iz[t,n],iq,ir[t,n]],apath[t,n],ppath[t,n]) 
                 
            lc=np.max(v_pr_path[t,n])/σ#local normalizing variable 
            Vmax = σ*np.euler_gamma+σ*(lc+np.log(np.sum(np.exp(v_pr_path[t,n]/σ-lc)))  ) 
            
            for i in range(nwls):    
                w_pr_path[t,n,i]=np.exp(v_pr_path[t,n,i]/σ-(Vmax-σ*np.euler_gamma)/σ) 
              
            #for i in range(nwls):               
             #   w_pr_path[t,n,i]=linear_interp.interp_2d(agrid,pgrid,pr[t,i,:,:,iz[t,n],iq],apath[t,n],ppath[t,n]) 
                
 
            for pp in range(nwls): 
                 i=pp 
                 if ts[t,n]<np.sum(w_pr_path[t,n,:i+1]):break 
                  
             
            A1p=policyA1[t,i, :,:,iz[t,n],iq,ir[t,n]] 
            Pp    =policyP[t,i, :,:,iz[t,n],iq,ir[t,n]] 
            Pp_exp=policyP_exp[t,i, :,:,iz[t,n],iq,ir[t,n]] 
            Cp=policyC[t,i, :,:,iz[t,n],iq,ir[t,n]] 
            if ((i==0) & (ir[t,n] ==0) & (age_ret[-1]>=t>=age_ret[0])): ir[t,n] = 1 
             
             
              
            cpath[t, n] = linear_interp.interp_2d(agrid,pgrid,Cp,apath[t,n],ppath[t,n]) 
            hpath[t, n] = i #linear_interp.interp_2d(agrid,pgrid,hp,apath[t,n],ppath[t,n]) 
            pepath[t, n] = np.maximum(np.minimum(mp2*wls[i]*w[t,i,iz[t,n]]/E_bar_now,Pmax),wls[i]*w[t,i,iz[t,n]]/E_bar_now)*(wls_point[i])-wls[i]*w[t,i,iz[t,n]]/E_bar_now*(wls_point[i]) 
            pepath2[t, n]= np.maximum(np.minimum(mp *wls[i]*w[t,i,iz[t,n]]/E_bar_now,Pmax),wls[i]*w[t,i,iz[t,n]]/E_bar_now)*(wls_point[i])-wls[i]*w[t,i,iz[t,n]]/E_bar_now*(wls_point[i]) 
             
            wpath[t, n] = w[t,i,iz[t,n]]#!!! not sure if useful 
            epath[t, n] = income[t,i,iz[t,n],0,ir[t,n]] if ir[t,n]==0 else np.interp(ppath[t, n],pgrid,income[t,i,iz[t,n],:,ir[t,n]]) 
            eataxpath[t, n] = taxes[t,i,iz[t,n],0,ir[t,n]] if ir[t,n]==0 else np.interp(ppath[t, n],pgrid,taxes[t,i,iz[t,n],:,ir[t,n]]) 
            epath_mod[t, n] = income_mod[t,i,iz[t,n],0,ir[t,n]] if ir[t,n]==0 else np.interp(ppath_exp[t, n],pgrid,income[t,i,iz[t,n],:,ir[t,n]]) 
            
             
            evpath[t, n] = linear_interp.interp_2d(agrid,pgrid,V1[t,:,:,iz[t,n],iq,ir[t,n]],apath[t,n],ppath[t,n])#+σ*np.euler_gamma-σ*np.log(prs[i])             
            vpath[t, n] = np.log(cpath[t, n]*cadjust)-q[iq,hpath[t, n],iz[t,n]]+σ*np.euler_gamma-σ*np.log(w_pr_path[t,n,i]) 
             
             
             
            if t<T-1:apath[t+1, n] = linear_interp.interp_2d(agrid,pgrid,A1p,apath[t,n],ppath[t,n]) 
            if t<T-1:ppath[t+1, n] =     linear_interp.interp_2d(agrid,pgrid,Pp    ,apath[t,n],ppath[t,n]) 
            if t<T-1:ppath_exp[t+1, n] = linear_interp.interp_2d(agrid,pgrid,Pp_exp,apath[t,n],ppath_exp[t,n]) 
            if t<T-1:pepath3[t+1, n]= pepath3[t, n] + np.maximum(np.minimum(mp3 *wls[i]*w[t,i,iz[t,n]]/E_bar_now,Pmax),wls[i]*w[t,i,iz[t,n]]/E_bar_now)*(wls_point[i]) 
            if t<T-1: ir[t+1,n] = 1 if ir[t,n] == 1 else 0 
             
    return epath,ppath,cpath,apath,hpath,pepath,pepath2,pepath3,vpath,evpath,wpath, w_pr_path, v_pr_path, eataxpath, epath_mod,iz,ir 
