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


def solveEulerEquation(reform, par):
    
    #Start counting time
    time_start = time.time()
        
    #Initiate some variables
    policyA1,policyh,policyC,V,policyp,pmutil,whic = np.empty((7,par.T, par.numPtsA, par.numPtsP))
     
    #Call the routing to solve the model
    solveEulerEquation1(policyA1, policyh, policyC, policyp,V,whic,pmutil,reform,par)

    #End timer and print elapsed time
    elapsed = time.time() - time_start    
    print('Finished, Reform =', reform, 'Elapsed time is', elapsed, 'seconds')   
    
    return policyA1,policyh,policyC,V,policyp,whic

#@njit(fastmath=True)
def solveEulerEquation1(policyA1, policyh, policyC, policyp,V,whic,pmutil,reform,par):
    
    # The rest is interior solution
    """ Use the method of endogenous gridpoint to solve the model.
        To improve it further: jit it, then use math.power, not *
    """
    #Initialize some variables
    r=par.r;delta=par.delta;gamma_c=par.gamma_c;R=par.R;tau=par.tau;beta=par.beta;
    w=np.array(par.w);agrid=par.agrid;y_N=par.y_N;gamma_h=par.gamma_h;T=par.T;numPtsA=par.numPtsA
    numPtsP=par.numPtsP;pgrid=par.pgrid;maxHours=par.maxHours;rho=par.rho;E_bar_now=par.E_bar_now;
    
    #Grid for assets and points
    agrid_box=np.transpose(np.tile(agrid,(numPtsP,1)))
    pgrid_box=np.tile(pgrid,(numPtsA,1))
    
    #Grid for consumption
    cgrid=nonlinspace(agrid[0],agrid[-1]*(1+r)+y_N+np.max(w)*maxHours*(1-tau),numPtsA,1.4)
    cgrid_box=np.transpose(np.tile(cgrid,(numPtsP,1)))
    
    #Last period decisions below
    policyA1[T-1,:,:] = np.zeros((numPtsA, numPtsP))  # optimal savings
    policyh[T-1,:,:] = np.zeros((numPtsA, numPtsP))   # optimal earnings
    policyC[T-1,:,:] = agrid_box*(1+r) + y_N +\
                                rho*pgrid_box        # optimal consumption
    pmutil[T-1,:,:]=co.mcutility(policyC[T-1,:,:], par)       # mu of more pension points        
    V[T-1,:,:]=co.utility(policyC[T-1,:,:],policyh[T-1,:,:],par)
    
    #Decisions below
    for t in range(T-2,-1,-1):
                         
        
        #Define initial variable for fast coputation later
        pmu=pmutil[t+1,:,:]
        pmuc=np.reshape(np.repeat(pmu[0,:],numPtsA),(numPtsA,numPtsP),order="F")
        
        wt=w[t]
        policy=((t+1 >=3) & (t+1 <=10) & (reform==1))
     
        if policy: 
            mult_pens=ne.evaluate('1.5*wt/E_bar_now*pmu/(1+delta)')
            mult_pensc=ne.evaluate('1.5*wt/E_bar_now*pmuc/(1+delta)')
        else:
            mult_pens=ne.evaluate('1.0*wt/E_bar_now*pmu/(1+delta)')
            mult_pensc=ne.evaluate('1.0*wt/E_bar_now*pmuc/(1+delta)')
        
        ################################################
        #Endogenous gridpoints here
        ##############################################
        
        #Get consumption from Eurler Equation
        ce=policyC[t+1,:,:]*np.power(((1+r)/(1+delta)),(-1/gamma_c))
             
        #How much work? This follows from the FOC      
        if (t+1<=R):            
            #Not retired,unconstrained
            he=ne.evaluate('maxHours-((mult_pens+wt*(1-tau)*(ce**(-gamma_c)))/beta)**(-1/gamma_h)')
  
            #Not retired, constrained
            hec=ne.evaluate('maxHours-((mult_pensc+wt*(1-tau)*(cgrid_box**(-gamma_c)))/beta)**(-1/gamma_h)')
        else:        
        #Retired case        
            he=np.zeros((numPtsA, numPtsP))
                  
        
        #Retired
        if (t+1<=R):
            
            if policy:
                #How much points should you have given the decisions? (Un)constrained
                pe=ne.evaluate('pgrid_box-    1.5*he*wt/E_bar_now')  
                pec=ne.evaluate('pgrid_box-   1.5*hec*wt/E_bar_now')
            else:
                                #How much points should you have given the decisions? (Un)constrained
                pe=ne.evaluate('pgrid_box-    he*wt/E_bar_now')  
                pec=ne.evaluate('pgrid_box-   hec*wt/E_bar_now')
            
        elif (t+1>R): 
            
            
            pe=pgrid_box.copy()
                        
        #How much assets? Just use the Budget constraint!
        if (t+1<=R):  
            ae=ne.evaluate('(agrid_box-wt*he*(1-tau)-y_N+ce)/(1+r)')
            
            aec=ne.evaluate('(cgrid_box - wt*(1-tau)*hec - y_N)/(1+r)')
        else:
            ae=ne.evaluate('(agrid_box-rho*pe       -y_N+ce)/(1+r)')

        ################################################
        # Now interpolate to be back on grid...
        ###############################################
          
        which,policyCu,policyhu,Vu,policyCcl,policyhcl,Vcl,policyCca,policyhca,Vca,policyCc,policyhc,Vc=np.zeros((13,numPtsA,numPtsP))
        holesu,holescl,holesca = np.ones((3,numPtsA,numPtsP))
        
        #Not retired
        if (t+1<=R):     
            
            #Unconstrained
            upperenvelop.compute(policyCu,policyhu,Vu,holesu,
                    pe,ae,ce,he,#computed above...
                    1, #should be 1
                    V[t+1,:,:],
                    gamma_c,maxHours,gamma_h,rho,agrid,pgrid,beta,r,wt,tau,y_N,E_bar_now,delta) #should be dropeed
            
            #A constrained
            upperenvelop.compute(policyCca,policyhca,Vca,holesca,
                     pec,aec,cgrid_box,hec,#pe,ae,ce,he,#computed above...
                     3, #should be 1
                     V[t+1,:,:],
                     gamma_c,maxHours,gamma_h,rho,agrid,pgrid,beta,r,wt,tau,y_N,E_bar_now,delta) #should be dropeed
            
            #A AND l constrained
            policyCc=agrid_box*(1+r) + y_N
            Vc=co.utility(policyCc,policyhc,par)+\
             1/(1+delta)*np.reshape(np.repeat(V[t+1,0,:],numPtsA),(numPtsA,numPtsP),order="F")#np.repeat(V[t+1,0,:],numPtsA).reshape(numPtsA,numPtsP)
                        
            # b. upper envelope    
            seg_max = np.zeros(3)
            for i_n in range(numPtsA):
                for i_m in range(numPtsP):
        
                    # i. find max
                    seg_max[0] = Vu[i_n,i_m]
                    seg_max[1] = Vca[i_n,i_m]
                    seg_max[2] = Vc[i_n,i_m]

        
                    i = np.argmax(seg_max)
                    which[i_n,i_m]=i
                    V[t,i_n,i_m]=seg_max[i]
                    
                    if i == 0:
                        policyC[t,i_n,i_m] = policyCu[i_n,i_m]
                        policyh[t,i_n,i_m] = policyhu[i_n,i_m]
                    elif i == 1:
                        policyC[t,i_n,i_m] = policyCca[i_n,i_m]
                        policyh[t,i_n,i_m] = policyhca[i_n,i_m]
                    elif i == 2:
                        policyC[t,i_n,i_m] = policyCc[i_n,i_m]
                        policyh[t,i_n,i_m] = policyhc[i_n,i_m]
                        
            #Complete
            policyA1[t,:,:]=agrid_box*(1+r)+y_N+wt*(1-tau)*policyh[t,:,:]-policyC[t,:,:]
            whic[t,:,:]=which
        #Retired
        else:
            
            for i in range(numPtsP):               
                linear_interp.interp_1d_vec(ae[:,i],agrid,agrid,policyA1[t,:,i])#np.interp(agrid, ae[:,i],agrid)
                policyC[t,:,i] =agrid*(1+r)+rho*pe[:,i]+y_N-policyA1[t,:,i]
                policyh[t,:,i] =he[:,i]
                V[t,:,i]=co.utility(policyC[t,:,i],policyh[t,:,i],par)+\
                     (1/(1+delta))*np.interp(policyA1[t,:,i],agrid,V[t+1,:,i])
            
        #Update marginal utility of having more pension points
        Pc=policyC[t,:,:]
        Ph=policyh[t,:,:]
        points=ne.evaluate('pgrid_box+wt*Ph/E_bar_now')
        
        Pmua=np.zeros((numPtsA, numPtsP))
        for i in range(numPtsP):
            linear_interp.interp_2d_vec(agrid,pgrid,pmu,policyA1[t,:,i],points[:,i],Pmua[:,i])# #pmu on the grid!!!
        if (t+1>R): 
            pmutil[t,:,:]=ne.evaluate('(Pmua+rho*Pc**(-gamma_c))/(1+delta)')
        else:
            pmutil[t,:,:]=ne.evaluate('Pmua/(1+delta)')     
 