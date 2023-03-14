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
    policyA1,policyh,policyC,V,policyp,pmutil,whic = np.empty((7,p.T, p.numPtsA, p.numPtsP,p.nw))
     
    #Call the routing to solve the model
    solveEulerEquation1(policyA1, policyh, policyC, policyp,V,whic,pmutil,reform,p)

    #End timer and print elapsed time
    elapsed = time.time() - time_start    
    print('Finished, Reform =', reform, 'Elapsed time is', elapsed, 'seconds')   
    
    return {'A':policyA1,'h':policyh,'c':policyC,'V':V,'p':policyp,'which':whic,'model':reform}

def solveEulerEquation1(policyA1, policyh, policyC, policyp,V,whic,pmutil,reform,p):
    
    """ Use the method of endogenous gridpoint in 2 dimensions to solve the model.
        Source: JDruedahlThomas and Jørgensen (2017). This method is robust to
        non-convexities
        
        Note that the grids for assets constrained and not constrained
        are different to avoid extrapolation
    """
    #Initialize some variables
    r=p.r;δ=p.δ;γc=p.γc;R=p.R;τ=p.τ;β=p.β;q=p.q;
    w=np.array(p.w);agrid=p.agrid;y_N=p.y_N;γh=p.γh;T=p.T;numPtsA=p.numPtsA;nw=p.nw;
    numPtsP=p.numPtsP;pgrid=p.pgrid;maxHours=p.maxHours;ρ=p.ρ;E_bar_now=p.E_bar_now;
    
    #Grid for assets and points
    agrid_box=np.repeat(np.transpose(np.tile(agrid,(numPtsP,1)))[:,:,np.newaxis],nw,axis=2)
    pgrid_box=np.repeat(np.tile(pgrid,(numPtsA,1))[:,:,np.newaxis],nw,axis=2)
    
    #Grid for consumption
    cgrid=nonlinspace(agrid[0],agrid[-1]*(1+r)+y_N+np.max(w)*maxHours*(1-τ),numPtsA,1.4)
    cgrid_box=np.repeat(np.transpose(np.tile(cgrid,(numPtsP,1)))[:,:,np.newaxis],nw,axis=2)
    
    #Last period decisions below
    policyA1[T-1,:,:,:] = np.zeros((numPtsA, numPtsP,nw))   # optimal savings
    policyh[T-1,:,:,:] = np.zeros((numPtsA, numPtsP,nw))    # optimal earnings
    policyC[T-1,:,:,:] = agrid_box*(1+r) + y_N +ρ*pgrid_box # optimal consumption                              
    pmutil[T-1,:,:,:]=co.mcutility(policyC[T-1,:,:,:], p)   # mu of more pension points        
    V[T-1,:,:,:]=co.utility(policyC[T-1,:,:,:],policyh[T-1,:,:,:],p)
    
    #Decisions below
    for t in range(T-2,-1,-1):
                         
        #Grid for wages
        wbox=np.repeat(np.repeat(w[t,np.newaxis,:],numPtsP,axis=0)[np.newaxis,:,:],numPtsA,axis=0)    
        
        #Define initial variable for fast coputation later
        pmu=pmutil[t+1,:,:,:]
        pmuc=np.repeat(pmu[0,np.newaxis,:,:],numPtsA,axis=0)
        
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
            ce=policyC[t+1,:,:,:]*np.power(((1+r)/(1+δ)),(-1/γc)) #Euler eq.
            he=ne.evaluate('maxHours-((mp*wt/E_bar_now*pmu/(1+δ)\
                             +wt*(1-τ)*(ce**(-γc)))/β)**(-1/γh)') #Labor supply
            pe=ne.evaluate('pgrid_box-    mp*he*wt/E_bar_now')    #Pens. points
            ae=ne.evaluate('(agrid_box-wt*he*(1-τ)-y_N+ce)/(1+r)')#Savings
            
            #Constrained (assets)
            heca=ne.evaluate('maxHours-((mp*wt/E_bar_now*pmuc/(1+δ)+wt*(1-τ)*\
                            (cgrid_box**(-γc)))/β)**(-1/γh)')        #Labor supply
            peca=ne.evaluate('pgrid_box-   mp*heca*wt/E_bar_now')      #Pens. points
            aeca=ne.evaluate('(cgrid_box - wt*(1-τ)*heca - y_N)/(1+r)')#Savings
            
            #Constrained (hours) -> Euler equation already writte above
            hech=np.zeros((numPtsA, numPtsP,nw))
            pech=ne.evaluate('pgrid_box')    #Pens. points
            aech=ne.evaluate('(agrid_box-y_N+ce)/(1+r)')#Savings
                
            #Constrained hours and assets...
            hec=np.zeros((numPtsA, numPtsP,nw))
            pec=ne.evaluate('pgrid_box')      #Pens. points
            aec=ne.evaluate('(cgrid_box - y_N)/(1+r)')#Savings
             
        else:
            
            ###################################################################
            #Retired case
            ###################################################################
            ce=policyC[t+1,:,:,:]*np.power(((1+r)/(1+δ)),(-1/γc)) #Euler equation
            he=np.zeros((numPtsA, numPtsP,nw))                    #Hours
            pe=pgrid_box.copy()                                   #Pens. points 
            ae=ne.evaluate('(agrid_box-ρ*pe       -y_N+ce)/(1+r)')#Savings
                  

        ################################################
        # Now interpolate to be back on grid...
        ###############################################
          
        which,policyCu,policyhu,Vu,policyCcl,policyhcl,Vcl,policyCca,policyhca,\
            Vca,policyCc,policyhc,Vc,policyCch,policyhch,Vch,holesch=np.zeros((17,numPtsA,numPtsP,nw))
        holesu,holesca,holesch,holesc = np.ones((4,numPtsA,numPtsP,nw))
        
        #Not retired
        if (t+1<=R):     
            
            # a. find policy for constrained and unconstrained choices
            
            #Unconstrained
            upperenvelop.compute(policyCu,policyhu,Vu,holesu,
                    pe,ae,ce,he,#computed above...
                    1, # which foc to take in upperenvelop
                    V[t+1,:,:,:],
                    γc,maxHours,γh,ρ,agrid,pgrid,β,r,wt,τ,y_N,E_bar_now,δ,q) 
            
            #A constrained
            upperenvelop.compute(policyCca,policyhca,Vca,holesca,
                     peca,aeca,cgrid_box,heca,#computed above...
                     3, # which foc to take in upperenvelop
                     V[t+1,:,:,:],
                     γc,maxHours,γh,ρ,agrid,pgrid,β,r,wt,τ,y_N,E_bar_now,δ,q) 
            
            #A AND l constrained (not relevant now...)
            upperenvelop.compute(policyCc,policyhc,Vc,holesc,
                     pec,aec,cgrid_box,hec,#computed above...
                     4, # which foc to take in upperenvelop
                     V[t+1,:,:,:],
                     γc,maxHours,γh,ρ,agrid,pgrid,β,r,wt,τ,y_N,E_bar_now,δ,0.0) 
             
            #h constrained
            upperenvelop.compute(policyCch,policyhch,Vch,holesch,
                     pech,aech,ce,hech,#computed above...
                     2, # which foc to take in upperenvelop
                     V[t+1,:,:,:],
                     γc,maxHours,γh,ρ,agrid,pgrid,β,r,wt,τ,y_N,E_bar_now,δ,0.0)   
                        
            # b. upper envelope    
            seg_max = np.zeros(4)
            for i_n in range(numPtsA):
                for i_m in range(numPtsP):
                    for i_w in range(nw):
        
                        # i. find max
                        seg_max[0] = Vu[i_n,i_m,i_w]
                        seg_max[1] = Vca[i_n,i_m,i_w]
                        seg_max[2] = Vc[i_n,i_m,i_w]-100000
                        seg_max[3] = Vch[i_n,i_m,i_w]
       
                        i = np.argmax(seg_max)
                        which[i_n,i_m,i_w]=i
                        V[t,i_n,i_m,i_w]=seg_max[i]
                        
                        if i == 0:
                            policyC[t,i_n,i_m,i_w] = policyCu[i_n,i_m,i_w]
                            policyh[t,i_n,i_m,i_w] = policyhu[i_n,i_m,i_w]
                        elif i == 1:
                            policyC[t,i_n,i_m,i_w] = policyCca[i_n,i_m,i_w]
                            policyh[t,i_n,i_m,i_w] = policyhca[i_n,i_m,i_w]
                        elif i == 2:
                            policyC[t,i_n,i_m,i_w] = policyCc[i_n,i_m,i_w]
                            policyh[t,i_n,i_m,i_w] = policyhc[i_n,i_m,i_w]
                        elif i == 3:
                            policyC[t,i_n,i_m,i_w] = policyCch[i_n,i_m,i_w]
                            policyh[t,i_n,i_m,i_w] = policyhch[i_n,i_m,i_w]    
                            
                        whic[t,i_n,i_m,i_w]=i
                        
            #Complete
            policyA1[t,:,:,:]=agrid_box*(1+r)+y_N+wbox*(1-τ)*policyh[t,:,:,:]-policyC[t,:,:,:]
            
        #Retired
        else:
            
            for i in range(numPtsP):
                for j in range(nw):
                    #linear_interp.interp_1d_vec(ae[:,i],agrid,agrid,policyA1[t,:,i])#np.interp(agrid, ae[:,i],agrid)
                    policyA1[t,:,i,j]=np.interp(agrid, ae[:,i,j],agrid)
                    policyC[t,:,i,j] =agrid*(1+r)+ρ*pe[:,i,j]+y_N-policyA1[t,:,i,j]
                    policyh[t,:,i,j] =he[:,i,j]
                    V[t,:,i,j]=co.utility(policyC[t,:,i,j],policyh[t,:,i,j],p)+\
                         (1/(1+δ))*np.interp(policyA1[t,:,i,j],agrid,V[t+1,:,i,j])
            
        #Update marginal utility of having more pension points
        Pc=policyC[t,:,:,:]
        Ph=policyh[t,:,:,:]
        points=ne.evaluate('pgrid_box+wt*Ph/E_bar_now')
        
        Pmua=np.zeros((numPtsA, numPtsP,nw))
        for i in range(numPtsP):
            for j in range(nw):
                linear_interp.interp_2d_vec(agrid,pgrid,pmu[:,:,j],policyA1[t,:,i,j],points[:,i,j],Pmua[:,i,j])# #pmu on the grid!!!
        if (t+1>R): 
            pmutil[t,:,:]=ne.evaluate('(Pmua+ρ*Pc**(-γc))/(1+δ)')
        else:
            pmutil[t,:,:]=ne.evaluate('Pmua/(1+δ)')     
 