# this is the key program to solve the model

# import packages
# Time
import time
import numpy as np
from consav.grids import nonlinspace # grids
from consav import linear_interp
import numexpr as ne
import upperenvelop
import co


def solveEulerEquation(p,model='baseline'):
    
    
    #Translate keyword for model into number
    if model=='baseline':      reform=0
    if model=='pension reform':reform=1
    
    #Start counting time
    time_start = time.time()
        
    #Initiate some variables
    policyA1,policyh,policyC,V,policyp,pmutil,whic= np.zeros((7,p.T,4,p.NA, p.NP,p.nw))-1e8
    holes=np.ones((p.T,4,p.NA, p.NP,p.nw))
    #Call the routing to solve the model
    solveEulerEquation1(policyA1, policyh, policyC, policyp,V,pmutil,whic,holes,reform,p)

    #End timer and print elapsed time
    elapsed = time.time() - time_start    
    print('Finished, Reform =', reform, 'Elapsed time is', elapsed, 'seconds')   
    
    return {'A':policyA1,'h':policyh,'c':policyC,'V':V,'p':policyp,'which':whic,'model':reform}

def solveEulerEquation1(policyA1, policyh, policyC, policyp,V,pmutil,whic,holes,reform,p):
    
    """ Use the method of endogenous gridpoint in 2 dimensions to solve the model.
        Source: JDruedahlThomas and Jørgensen (2017). This method is robust to
        non-convexities
        
        Note that the grids for assets constrained and not constrained
        are different to avoid extrapolation
    """
    #Initialize some variables
    r=p.r;δ=p.δ;γc=p.γc;R=p.R;τ=p.τ;β=p.β;q=p.q;amin=p.amin;
    w=np.array(p.w);agrid=p.agrid;y_N=p.y_N;γh=p.γh;T=p.T;NA=p.NA;nw=p.nw;
    NP=p.NP;pgrid=p.pgrid;maxHours=p.maxHours;ρ=p.ρ;E_bar_now=p.E_bar_now;
    
    ce,he,pe,ae=np.zeros((4,4,NA, NP,nw))
    V1,c1,pmu=np.zeros((3,NA, NP,nw))
    
    #Grid for assets and points
    agrid_box=np.repeat(np.transpose(np.tile(agrid,(NP,1)))[:,:,np.newaxis],nw,axis=2)
    pgrid_box=np.repeat(np.tile(pgrid,(NA,1))[:,:,np.newaxis],nw,axis=2)
    
    #Grid for consumption
    cgrid=nonlinspace(agrid[0],agrid[-1]*(1+r)+y_N+np.max(w)*maxHours*(1-τ),NA,1.4)
    cgrid_box=np.repeat(np.transpose(np.tile(cgrid,(NP,1)))[:,:,np.newaxis],nw,axis=2)
    
    #Last period decisions below
    policyA1[T-1,3,:,:,:] = np.zeros((NA, NP,nw))   # optimal savings
    policyh[T-1,3,:,:,:] = np.zeros((NA, NP,nw))    # optimal earnings
    policyC[T-1,3,:,:,:] = agrid_box*(1+r) + y_N +ρ*pgrid_box # optimal consumption                              
    pmutil[T-1,3,:,:,:]=co.mcutility(policyC[T-1,3,:,:,:], p)   # mu of more pension points        
    V[T-1,3,:,:,:]=co.utility(policyC[T-1,3,:,:,:],policyh[T-1,3,:,:,:],p)
                    
    #Decisions below
    for t in range(T-2,-1,-1):
                         
        #Get variables useful for next iteration t-1
        ia = np.argmax(V[t+1,:,:,:,:],axis=0)
        for i_n in range(NA):
            for i_m in range(NP):
                for i_w in range(nw):
                    
                    pmu[i_n,i_m,i_w]= pmutil[t+1,ia[i_n,i_m,i_w],i_n,i_m,i_w].copy()
                    V1[i_n,i_m,i_w] = V[t+1,ia[i_n,i_m,i_w],i_n,i_m,i_w].copy()
                    c1[i_n,i_m,i_w] = policyC[t+1,ia[i_n,i_m,i_w],i_n,i_m,i_w].copy()
                    
        #Grid for wages
        wbox=np.repeat(np.repeat(w[t,np.newaxis,:],NP,axis=0)[np.newaxis,:,:],NA,axis=0)    
        
        #Define initial variable for fast coputation later
        pmuc=np.repeat(pmu[0,np.newaxis,:,:],NA,axis=0)
        
        wt=w[t,:]
        policy=((t >=3) & (t <=10) & (reform==1))
        
        #Multiplier of points based on points
        mp=1.5 if policy else 1.0
        
        ################################################
        #Endogenous gridpoints here
        ##############################################
        
             
        #How much work? This follows from the FOC      
        if (t+1<=R):    
            
            ###################################################################
            #Not retired case
            ###################################################################
            
            #Unconstrained
            ce[0,...]=c1*np.power(((1+r)/(1+δ)),(-1/γc)) #Euler eq.
            he[0,...]=((mp*wt/E_bar_now*pmu/(1+δ)\
                             +wt*(1-τ)*(ce[0,...]**(-γc)))/β)**(γh)
            pe[0,...]=pgrid_box-    mp*he[0,...]*wt/E_bar_now   #Pens. points
            ae[0,...]=(agrid_box-wt*he[0,...]*(1-τ)-y_N+ce[0,...])/(1+r)#Savings
            
            #Constrained (assets)
            he[1,...]=((mp*wt/E_bar_now*pmuc/(1+δ)+wt*(1-τ)*(cgrid_box**(-γc)))/β)**(γh)        #Labor supply
            pe[1,...]=pgrid_box-   mp*he[1,...]*wt/E_bar_now      #Pens. points
            ce[1,...]=cgrid_box.copy()
            ae[1,...]=(ce[1,...] - wt*(1-τ)*he[1,...] - y_N+amin)/(1+r)#Savings
            
            
            #Constrained hours and assets...
            he[2,...]=np.zeros((NA, NP,nw))
            pe[2,...]=pgrid_box.copy()      #Pens. points
            ce[2,...]=cgrid_box.copy()
            ae[2,...]=(ce[2,...] - y_N +amin)/(1+r)#Savings
            
            
            #Constrained (hours) -> Euler equation already writte above
            ce[3,...]=ce[0,...].copy()
            he[3,...]=np.zeros((NA, NP,nw))
            pe[3,...]=pgrid_box.copy()    #Pens. points
            ae[3,...]=(agrid_box-y_N+ce[3,...])/(1+r)#Savings
            
                

             
        else:
            
            ###################################################################
            #Retired case
            ###################################################################
            ce[3,...]=c1*np.power(((1+r)/(1+δ)),(-1/γc)) #Euler equation
            he[3,...]=np.zeros((NA, NP,nw))                    #Hours
            pe[3,...]=pgrid_box.copy()                                   #Pens. points 
            ae[3,...]=(agrid_box-ρ*pe[3,...]-y_N+ce[3,...])/(1+r)#Savings
                  

        ################################################
        # Now interpolate to be back on grid...
        ###############################################

        
        #Not retired
        if (t+1<=R):     
            
            # a. find policy for constrained and unconstrained choices
            
            #Unconstrained
            for i in range(4):
                
                #Penalty for not working?
                pen=q if i<2 else 0.0
                
                #Computation below
                upperenvelop.compute(policyC[t,i,...],policyh[t,i,...],V[t,i,...],holes[t,i,...],
                        pe[i,...],ae[i,...],ce[i,...],he[i,...],#computed above...
                        i, # which foc to take in upperenvelop
                        V1,
                        γc,maxHours,γh,ρ,agrid,pgrid,β,r,wt,τ,y_N,E_bar_now,δ,pen,amin) 
            
                #Assets here
                policyA1[t,i,:,:,:]=agrid_box*(1+r)+y_N+wbox*(1-τ)*policyh[t,i,:,:,:]-policyC[t,i,:,:,:]
                        
            
        #Retired
        else:
            
            for i in range(NP):
                for j in range(nw):
                    #linear_interp.interp_1d_vec(ae[:,i],agrid,agrid,policyA1[t,:,i])#np.interp(agrid, ae[:,i],agrid)
                    policyA1[t,3,:,i,j]=np.interp(agrid, ae[3,:,i,j],agrid)
                    policyC[t,3,:,i,j] =agrid*(1+r)+ρ*pe[3,:,i,j]+y_N-policyA1[t,3,:,i,j]
                    policyh[t,3,:,i,j] =he[3,:,i,j]
                    V[t,3,:,i,j]=co.utility(policyC[t,3,:,i,j],policyh[t,3,:,i,j],p)+\
                         (1/(1+δ))*np.interp(policyA1[t,3,:,i,j],agrid,V1[:,i,j])
                         
                         
        
            
        #Update marginal utility of having more pension points
        Pmua=np.zeros((4,NA, NP,nw))
        for pp in range(4):
            
            points=pgrid_box+wt*policyh[t,pp,:,:,:]/E_bar_now         
            
            for i in range(NP):
                for j in range(nw):
                
                    linear_interp.interp_2d_vec(agrid,pgrid,pmu[:,:,j],policyA1[t,pp,:,i,j],points[:,i,j],Pmua[pp,:,i,j])# #pmu on the grid!!!
        if (t+1>R): 
            pmutil[t,...]=(Pmua+ρ*policyC[t,:,:,:,:]**(-γc))/(1+δ)
        else:
            pmutil[t,...]=Pmua/(1+δ)     
            
            

 