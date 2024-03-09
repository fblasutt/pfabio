# run simulation for one indvidual without uncertainty

import numpy as np
from numba import njit
from consav import linear_interp
#https://www.econforge.org/interpolation.py/

def simNoUncer_interp(p, model, Tstart=0, Astart=0.0, Pstart=0.0, Vstart= -1.0*np.ones((2,2,2,2,2)),cadjust=1.0):
 
    

    
    #Call the simulator
    epath,ppath,cpath,apath,hpath,pepath,pepath2,vpath,evpath,wpath=\
        fast_simulate(Tstart,Astart,Pstart,Vstart,p.amax,p.T,p.N,p.agrid,p.pgrid,p.w,p.E_bar_now,p.Pmax,p.add_points,p.tw,p.ts,p.wls,p.nwls,
                      p.δ,p.q_mini,p.q,p.β,p.γh,p.σ,
                      model['A'],model['c'],model['p'],model['pr'],model['V'],model['V1'],model['model'],cadjust,p.wls_point)
    
    return {'wh':epath,'p':ppath,'c':cpath,'A':apath,'h':hpath,'pb':pepath, 'pb2':pepath2, 'v':vpath,'ev':evpath,'w':wpath}
    
@njit
def fast_simulate(Tstart,Astart,Pstart,Vstart,amax,T,N,agrid,pgrid,w,E_bar_now,Pmax,add_points,tw,ts,wls,nwls,δ,q_mini,q,β,γh,σ,
                  policyA1,policyC,policyP,pr,V,V1,reform,cadjust,wls_point):

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
    for t in range(Ti,T):  # loop through time periods for a pticular individual
        for n in range(N):
            
            policy=((t >=4) & (t <=11))#& (reform==1))
            policy2=((t >=4) & (t <=11) & (reform==1))
            mp=add_points #if policy else 1.0
            mp2=add_points if policy2 else 1.0
            #Get the discrete choices first...
            prs=np.zeros(nwls)
            for pp in range(nwls):
                prs[pp]=linear_interp.interp_2d(agrid,pgrid,pr[t,pp,:,:,tw[n]],apath[t,n],ppath[t,n])

           
            i=np.argmin((ts[t,n]>np.cumsum(prs)))
            A1p=policyA1[t,i, :,:,tw[n]]
            Pp=policyP[t,i, :,:,tw[n]]
            Cp=policyC[t,i, :,:,tw[n]]
            Vp=V[t,i, :,:,tw[n]]

            

            cpath[t, n] = linear_interp.interp_2d(agrid,pgrid,Cp,apath[t,n],ppath[t,n])
            hpath[t, n] = i#linear_interp.interp_2d(agrid,pgrid,hp,apath[t,n],ppath[t,n])
            pepath[t, n] = np.maximum(np.minimum(mp2*wls[i]*w[t,i,tw[n]]/E_bar_now,Pmax),wls[i]*w[t,i,tw[n]]/E_bar_now)*(i>wls_point)-wls[i]*w[t,i,tw[n]]/E_bar_now*(i>wls_point)
            pepath2[t, n]= np.maximum(np.minimum(mp *wls[i]*w[t,i,tw[n]]/E_bar_now,Pmax),wls[i]*w[t,i,tw[n]]/E_bar_now)*(i>wls_point)-wls[i]*w[t,i,tw[n]]/E_bar_now*(i>wls_point)
            wpath[t, n] = w[t,3,tw[n]]
            epath[t, n] = wpath[t, n]*wls[hpath[t, n]]*(i>wls_point)
            evpath[t, n] = linear_interp.interp_2d(agrid,pgrid,V1[t,:,:,tw[n]],apath[t,n],ppath[t,n])#+σ*np.euler_gamma-σ*np.log(prs[i])
            
            vpath[t, n] = np.log(cpath[t, n]*cadjust)-β*wls[hpath[t, n]]**(1+1/γh)/(1+1/γh)-q*(hpath[t, n]>0)+q_mini*(hpath[t, n]==1)+\
                σ*np.euler_gamma-σ*np.log(prs[i])
            
            if t<T-1:apath[t+1, n] = linear_interp.interp_2d(agrid,pgrid,A1p,apath[t,n],ppath[t,n])
            if t<T-1:ppath[t+1, n] = linear_interp.interp_2d(agrid,pgrid,Pp,apath[t,n],ppath[t,n])
        
     
    return epath,ppath,cpath,apath,hpath,pepath,pepath2,vpath,evpath,wpath

@njit
def adjust_wealth(Vstart, N, V, apath, ppath, amax, agrid, pgrid, Ti, tw,nwls):
    
    step=amax/1000
    for n in range(N):
        
        #Standart Utility
        Vi=np.zeros(nwls)-np.inf
        for pp in range(nwls):
            Vi[pp]=linear_interp.interp_2d(agrid,pgrid,V[Ti,pp,:,:,tw[n]],apath[Ti,n],ppath[Ti,n])
        Vi[np.isnan(Vi)]=-1e8
        Vmaxi=np.max(Vi)
        
        #New utility
        Vm=np.zeros(nwls)-np.inf
        for pp in range(nwls):
            Vm[pp]=linear_interp.interp_2d(agrid,pgrid,Vstart[Ti,pp,:,:,tw[n]],apath[Ti,n],ppath[Ti,n])
        Vm[np.isnan(Vm)]=-1e8
        Vmaxm=np.max(Vm)
        
        #Compare the two differences in utility

        if Vmaxi>Vmaxm:
            for i in range(100):
                
                addition=step*i
                Vi2=np.zeros(nwls)-np.inf
                for pp in range(nwls):
                    Vi2[pp]=linear_interp.interp_2d(agrid,pgrid,V[Ti,pp,:,:,tw[n]],apath[Ti,n]-addition,ppath[Ti,n])
                Vi2[np.isnan(Vi2)]=-1e8                   
                Vmaxi2=np.max(Vi2)
                
                if Vmaxi2<Vmaxm:
                    apath[Ti,n]=apath[Ti,n]-addition
                    break
        else:
            for i in range(100):
                
                addition=step*i
                Vi2=np.zeros(nwls)-np.inf
                for pp in range(nwls):
                    Vi2[pp]=linear_interp.interp_2d(agrid,pgrid,V[Ti,pp,:,:,tw[n]],apath[Ti,n]+addition,ppath[Ti,n])
                Vi2[np.isnan(Vi2)]=-1e8                   
                Vmaxi2=np.max(Vi2)
                
                if Vmaxi2>Vmaxm:
                    apath[Ti,n]=apath[Ti,n]+addition
                    break              
        
    return apath