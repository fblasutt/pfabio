# run simulation for one indvidual without uncertainty

import numpy as np
from numba import njit,prange 
from consav import linear_interp
#https://www.econforge.org/interpolation.py/

def simNoUncer_interp(p, model, Tstart=0, Astart=0.0, Pstart=0.0, Vstart= -1.0*np.ones((2,2,2,2,2)),cadjust=1.0):
 
    
    np.random.seed(3) 
    p.Πq = np.ones((p.nq,10))/2
    
    for iw in range(10): 
        if iw<5:#if just to get symmetry
            p.Πq[0,iw] = 1/2 - (5-iw)*p.ρq
            p.Πq[1,iw] = 1/2 + (5-iw)*p.ρq
        else:
            p.Πq[0,iw] = 1/2 - (5-1-iw)*p.ρq
            p.Πq[1,iw] = 1/2 + (5-1-iw)*p.ρq
        
    p.q_sim = np.zeros((p.N),dtype=np.int32)+1
    #p.q_sim[np.random.rand(p.N)<np.cumsum(p.Πq,axis=0)[1][p.tw]]=1
    p.q_sim[np.random.rand(p.N)<np.cumsum(p.Πq,axis=0)[0][p.tw]]=0
    
    

        
    
    #Call the simulator
    epath,ppath,cpath,apath,hpath,pepath,pepath2,vpath,evpath,wpath=\
        fast_simulate(Tstart,Astart,Pstart,Vstart,p.amax,p.T,p.N,p.agrid,p.pgrid,p.w,p.E_bar_now,p.Pmax,p.add_points,p.tw,p.ts,p.wls,p.nwls,
                      p.δ,p.q_mini,p.q_grid,p.β,p.γh,p.σ,
                      model['A'],model['c'],model['p'],model['pr'],model['V'],model['V1'],model['model'],cadjust,p.wls_point,p.q_sim)
    
    return {'wh':epath,'p':ppath,'c':cpath,'A':apath,'h':hpath,'pb':pepath, 'pb2':pepath2, 'v':vpath,'ev':evpath,'w':wpath}
    
@njit(parallel=True)
def fast_simulate(Tstart,Astart,Pstart,Vstart,amax,T,N,agrid,pgrid,w,E_bar_now,Pmax,add_points,tw,ts,wls,nwls,δ,q_mini,q,β,γh,σ,
                  policyA1,policyC,policyP,pr,V,V1,reform,cadjust,wls_point,q_sim):

    # Arguments for output
    cpath = np.nan+ np.zeros((T, N))           # consumption
    hpath = np.zeros((T, N),dtype=np.int32)           # earnings path
    pepath = np.nan+ np.zeros((T, N))           #corrected potints path
    pepath2 = np.nan+ np.zeros((T, N))           #additional caregiver credits
    epath = np.zeros((T, N))            # earnings path
    wpath = np.nan+ np.zeros((T, N))           # wage path
    vpath = np.nan+ np.zeros((T, N))           # utility
    evpath = np.nan+ np.zeros((T, N))           # expected utility
    apath = np.nan+ np.zeros((T,N))        # assets at start of each period, decided 1 period ahead and so includes period T+1   
    ppath = np.nan+ np.zeros((T,N))        # points at start of each period, decided 1 period ahead and so includes period T+1
    
    
    # Modified initial conditions
    Ti=Tstart
    apath[Ti, :] = Astart; 
    ppath[Ti, :] = Pstart; 
    
    
    #If Vstart if provided, adjust wealth so that utility matches Vstart
    #if np.min(Vstart)!=-1.0:apath=adjust_wealth(Vstart, N, V, apath, ppath, amax, agrid, pgrid,Ti,tw,nwls)
       
    
    # Obtain paths using the initial condition and the policy and value functions  
    for n in prange(N):
        for t in range(Ti,T):  # loop through time periods for a pticular individual

            
            iq = q_sim[n]
            #policy=((t >=4) & (t <=11))#& (reform==1))
            policy2=((t >=4) & (t <=11) & (reform==1))
            mp=add_points #if policy else 1.0
            mp2=add_points if policy2 else 1.0
            #Get the discrete choices first...
            prs=np.zeros(nwls)
            #i=0
            
            for pp in range(nwls):
                i=pp
                prs[pp]=linear_interp.interp_2d(agrid,pgrid,pr[t,pp,:,:,tw[n],iq],apath[t,n],ppath[t,n])
                if ts[t,n]<np.sum(prs):break
                    

            A1p=policyA1[t,i, :,:,tw[n],iq]
            Pp=policyP[t,i, :,:,tw[n],iq]
            Cp=policyC[t,i, :,:,tw[n],iq]

            

            cpath[t, n] = linear_interp.interp_2d(agrid,pgrid,Cp,apath[t,n],ppath[t,n])
            hpath[t, n] = i#linear_interp.interp_2d(agrid,pgrid,hp,apath[t,n],ppath[t,n])
            pepath[t, n] = np.maximum(np.minimum(mp2*wls[i]*w[t,i,tw[n]]/E_bar_now,Pmax),wls[i]*w[t,i,tw[n]]/E_bar_now)*(wls_point[i])-wls[i]*w[t,i,tw[n]]/E_bar_now*(wls_point[i])
            pepath2[t, n]= np.maximum(np.minimum(mp *wls[i]*w[t,i,tw[n]]/E_bar_now,Pmax),wls[i]*w[t,i,tw[n]]/E_bar_now)*(wls_point[i])-wls[i]*w[t,i,tw[n]]/E_bar_now*(wls_point[i])
            wpath[t, n] = w[t,3,tw[n]]
            epath[t, n] = wpath[t, n]*wls[hpath[t, n]]*wls_point[i]
            evpath[t, n] = linear_interp.interp_2d(agrid,pgrid,V1[t,:,:,tw[n],iq],apath[t,n],ppath[t,n])#+σ*np.euler_gamma-σ*np.log(prs[i])            
            vpath[t, n] = np.log(cpath[t, n]*cadjust)-q[iq,hpath[t, n]]+σ*np.euler_gamma-σ*np.log(prs[i])
            
            if t<T-1:apath[t+1, n] = linear_interp.interp_2d(agrid,pgrid,A1p,apath[t,n],ppath[t,n])
            if t<T-1:ppath[t+1, n] = linear_interp.interp_2d(agrid,pgrid,Pp,apath[t,n],ppath[t,n])
        
     
    return epath,ppath,cpath,apath,hpath,pepath,pepath2,vpath,evpath,wpath
