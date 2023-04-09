# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 14:56:15 2022

@author: Fabio
"""


import dfols
import co # user defined functions
import sol
import sim
import numpy as np
from scipy.optimize import minimize

#Actual Program

    

#Function to minimize
def q(pt):

    #Define main parameters
    p = co.setup()
    
    #..and update them
    p.q=pt[0]
    p.β =pt[1]
    p.σ =pt[2]
    
    #Pension reform
    ModP= sol.solveEulerEquation(p,model='pension reform')

    #Baseline
    ModB = sol.solveEulerEquation(p,model='baseline')
  

    #Baseline
    SB= sim.simNoUncer_interp(p,ModB,Tstart=0,Astart=p.startA,Pstart=np.zeros(p.N))

    #Pension reform
    SP= sim.simNoUncer_interp(p,ModP,Tstart=3,Astart=SB['A'][3,:],Pstart=SB['p'][3,:])
   
   
    
    #
    shpo=np.mean(SB['h'][3:11,:]>0)
    sh05=np.mean(SB['h'][3:11,:]==1)
    eff=np.mean(SP['h'][3:11,:]>0)-np.mean(SB['h'][3:11,:]>0)
    
    #Print the point
    print("The point is {}, the moments are {}, {}, {}".format(pt,shpo,sh05,eff))   
    
    ans=[((shpo-0.64)/0.64)**2,\
         ((sh05-0.33)/0.33)**2,\
         ((eff-0.099)/0.099)**2]
        
    return ans#((shpo-0.64)/0.64)**2+((sh05-0.33)/0.33)**2+((eff-0.099)/0.099)**2#ans
            
            
            
#Initialize seed
np.random.seed(10)


#Define initial point (xc) and boundaries (xl,xu)
xc=np.array([0.09631098,0.52197238,0.00311286])
xl=np.array([0.06,0.4,0.0005])
xu=np.array([0.15,0.8,0.01])

#Optimization below
res=dfols.solve(q, xc, rhobeg = 0.3, rhoend=1e-8, maxfun=600, bounds=(xl,xu),
                npt=len(xc)+5,scaling_within_bounds=True, 
                user_params={'tr_radius.gamma_dec':0.98,'tr_radius.gamma_inc':1.0,
                              'tr_radius.alpha1':0.9,'tr_radius.alpha2':0.95,
                              'regression.momentum_extra_steps':True},
                objfun_has_noise=False)
 


#The point is [0.09800833, 0.50142732, 0.00124261 ], the moments are 0.6376375, 0.3382125, 0.09378124999999998
  

# resu=res.x.copy()
# print(resu)
  


    
    
    