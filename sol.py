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
    solveEulerEquation1(policyA1, policyC, policyp,V,V1,pmutil,pr,holes,reform,p)

    #End timer and print elapsed time
    elapsed = time.time() - time_start    
    #print('Finished, Reform =', reform, 'Elapsed time is', elapsed, 'seconds')   
    
    return {'A':policyA1,'c':policyC,'V':V,'V1':V1,'p':policyp,'pr':pr,'model':reform}

#@profile
#@njit
def solveEulerEquation1(policyA1, policyC, policyp,V,V1,pmutil,pr,holes,reform,p):
    
    """ Use the method of endogenous gridpoint in 2 dimensions to solve the model.
        Source: JDruedahlThomas and Jørgensen (2017). This method is robust to
        non-convexities
        
        Note that the grids for assets constrained and not constrained
        are different to avoid extrapolation
    """
    #Initialize some variables
    r=p.r;δ=p.δ;γc=p.γc;R=p.R;τt=p.τ;β=p.β;q=p.q;amin=p.amin;wls=p.wls;nwls=p.nwls;
    w=np.array(p.w);agrid=p.agrid;y_N=p.y_N;γh=p.γh;T=p.T;NA=p.NA;nw=p.nw;σ=p.σ;
    NP=p.NP;pgrid=p.pgrid;ρ=p.ρ;E_bar_now=p.E_bar_now;q_mini=p.q_mini
    Pmax=p.Pmax;add_points=p.add_points;nq=p.nq
    
    ce,pe,ae,ce_bc,pe_bc,ae_bc=np.zeros((6,p.nwls,NA, NP, nw, nq))
    c1=np.zeros((NA, NP,nw,nq))
    
    #Grid for assets and points
    agrid_box=np.repeat(np.repeat(np.transpose(np.tile(agrid,(NP,1)))[:,:,np.newaxis,np.newaxis],nw,axis=2),nq,axis=3)
    pgrid_box=np.repeat(np.repeat(np.tile(pgrid,(NA,1))[:,:,np.newaxis,np.newaxis],nw,axis=2),nq,axis=3)
    y_N_box=np.repeat(np.repeat(np.repeat(y_N[:,np.newaxis,np.newaxis,:,np.newaxis],NA,axis=1),NP,axis=2),nq,axis=4)
    
    #Grid for consumption
    cgrid=nonlinspace(agrid[0]*0.001,agrid[-1]*(1+r)+np.max(y_N)+np.max(w)*(1-np.min(τt)),NA,1.4)
    cgrid_box=np.repeat(np.repeat(np.transpose(np.tile(cgrid,(NP,1)))[:,:,np.newaxis,np.newaxis],nw,axis=2),nq,axis=3)
    
    #Last period decisions below
    policyA1[T-1,0,...] = np.zeros((NA, NP,nw,nq))   # optimal savings
    policyC[T-1,0,...] = agrid_box*(1+r) + y_N_box[T-1,] +ρ*pgrid_box # optimal consumption   
    policyp[T-1,0,...] = pgrid_box # optimal consumption                              
    pmutil[T-1,0,...]=co.mcutility(policyC[T-1,0,...], p)   # mu of more pension points        
    V[T-1,0,...]=co.utility(policyC[T-1,0,...],wls[0],p) #np.log(policyC[T-1,0,:,:,:])
   
    #Decisions below
    for t in range(T-2,-2,-1):
         

        #Get variables useful for next iteration t-1
        expectation(t,NA,NP,nw,nq,V,V1[t+1,...],σ,γc,pr,c1,policyC)    
       
        
        if t==-1:break

        
        τ=τt[t]
        y_Nt=y_N_box[t,:,:,:]
        policy=((t >=8) & (t <=11) & (reform==1))
        
        ζ = p.ζ if t<=11 else 0.0
        
        #Multiplier of points based on points
        mp=add_points if policy else 1.0
        
        ################################################
        #Endogenous gridpoints here
        ##############################################
        
             
        #How much work? This follows from the FOC      
        if (t+1<=R):    
            
            ###################################################################
            #Not retired case
            ###################################################################
            
            #cgrid_box=np.maximum(cgrid_box,0.00000000001)

            for i in range(nwls):
                
                wt=w[t,i,:]
                
                #modify for the mini-jobs case
                tax=τ  if (i>1) else 0.0
                
                #Unconstrained
                ce[i,...]=c1*np.power(((1+r)/(1+δ)),(-1/γc)) #Euler eq.
                pe[i,...]=pgrid_box-np.maximum(np.minimum(mp*wls[i]*wt[:,None]/E_bar_now,Pmax),wls[i]*wt[:,None]/E_bar_now)*p.wls_point[i]   #Pens. points
                ae[i,...]=(agrid_box-wt[:,None]*wls[i]*(1-tax)-y_Nt+ce[i,...])/(1+r)#Savings
                
                #Constrained (assets)
                pe_bc[i,...]=pgrid_box-  np.maximum(np.minimum(mp*wls[i]*wt[:,None]/E_bar_now,Pmax),wls[i]*wt[:,None]/E_bar_now)*p.wls_point[i]     #Pens. points
                             
                ce_bc[i,...]=cgrid_box.copy()
                ae_bc[i,...]=(ce_bc[i,...] - wt[:,None]*(1-tax)*wls[i] - y_Nt+amin)/(1+r)#Savings
                
                
            
        

             
        else:
            
            ###################################################################
            #Retired case
            ###################################################################
            ce[0,...]=c1*np.power(((1+r)/(1+δ)),(-1/γc)) #Euler equation
            pe[0,...]=pgrid_box.copy()                                   #Pens. points 
            ae[0,...]=(agrid_box-ρ*pe[0,...]-y_Nt+ce[0,...])/(1+r)#Savings
                  

        ################################################
        # Now interpolate to be back on grid...
        ###############################################

        
        #Not retired
        if (t+1<=R):     
            
            # a. find policy for constrained and unconstrained choices
            
            #tic=time.time();
            #Unconstrained
            for i in range(nwls):
               
                    
                wt=w[t,i,:]
                
                #Penalty for working?
                q_pen=p.q_grid #if i>0 else np.zeros(p.q_grid.shape)
                
                #Mini jobs below
                #modify for the mini-jobs case
                tax=τ      if (i>1) else 0.0
                #mpp=mp     if i!=1 else 0.0
                q_min=0.0  if i!=1 else q_mini
                
                #Computation below
                upperenvelop.compute(policyC[t,i,...],policyA1[t,i,:,:,:],policyp[t,i,:,:,:],V[t,i,...],holes[t,i,...], 
                        pe[i,...],ae[i,...],ce[i,...],pe_bc[i,...],ae_bc[i,...],ce_bc[i,...],#computed above... 
                        i, # which foc to take in upperenvelop 
                        V1[t+1,...], 
                        γc,γh,ρ,agrid,pgrid,β,r,wt,tax,y_Nt,E_bar_now,Pmax,δ,q_pen,amin,wls[i],mp,q_min,p.wls_point[i],ζ)  
     
     
            #toc=time.time()         
            #print('2 {}'.format(toc-tic))
        #Retired
        else:
            
            for i in range(NP):
                for j in range(nw):
                    for iq in range(nq):

                        policyA1[t,0,:,i,j,iq]=np.interp(agrid, ae[0,:,i,j,iq],agrid)
                        policyC[t,0,:,i,j,iq] =agrid*(1+r)+ρ*pe[0,:,i,j,iq]+y_Nt[:,i,j,iq]-policyA1[t,0,:,i,j,iq]
                        policyp[t,0,:,:,:,:]=pgrid_box
                        V[t,0,:,i,j,iq]=co.utility(policyC[t,0,:,i,j,iq],wls[0],p)+\
                             (1/(1+δ))*np.interp(policyA1[t,0,:,i,j,iq],agrid,V1[t+1,:,i,j,iq])
                         
     
@njit
def expectation(t,NA,NP,nw,nq,V,V1,σ,γc,pr,c1,policyC):                
        #Get variables useful for next iteration t-1
        for i_n in prange(NA):
            for i_m in range(NP):
                for i_w in range(nw):
                    for iq in range(nq):
                    
                        lc=np.max(V[t+1,:,i_n,i_m,i_w,iq])/σ#local normalizing variable
                        V1[i_n,i_m,i_w,iq] = σ*np.euler_gamma+σ*(lc+np.log(np.sum(np.exp(V[t+1,:,i_n,i_m,i_w,iq]/σ-lc)))  )
                        pr[t+1,:,i_n,i_m,i_w,iq]=np.exp(V[t+1,:,i_n,i_m,i_w,iq]/σ-(V1[i_n,i_m,i_w,iq]-σ*np.euler_gamma)/σ)
                        c1[i_n,i_m,i_w,iq] = np.sum(pr[t+1,:,i_n,i_m,i_w,iq]*policyC[t+1,:,i_n,i_m,i_w,iq])
 