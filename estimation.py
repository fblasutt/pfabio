# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 14:56:15 2022

@author: Fabio
"""


import co # user defined functions
import sol
import sim
import numpy as np
import pybobyqa
from scipy.optimize import dual_annealing,differential_evolution
import scipy

#Actual Program

    

#Function to minimize
def q(pt):

    #Define main parameters
    p = co.setup()
    
    #..and update them
    p.q=pt[0]
    p.β =pt[1]
    p.δ =pt[2]
    p.q_mini=pt[3]*pt[0]
    
    #Pension reform
    ModP= sol.solveEulerEquation(p,model='pension reform')

    #Baseline
    ModB = sol.solveEulerEquation(p,model='baseline')
  

    #Baseline
    SB= sim.simNoUncer_interp(p,ModB,Tstart=0,Astart=p.startA,Pstart=np.ones(p.N)*p.startP)

    #Pension reform
    SP= sim.simNoUncer_interp(p,ModP,Tstart=3,Astart=SB['A'][3,:],Pstart=SB['p'][3,:])
   
   
    
    #
    shpo=np.mean(SB['h'][3:11,:]==2)
    sh1=np.mean(SB['h'][3:11,:]==3)
    sh_min=np.mean(SB['h'][3:11,:]==1)
    eff_e=np.mean(SP['h'][3:11,:]>0)-np.mean(SB['h'][3:11,:]>0)
    eff_full=np.mean(SP['h'][3:11,:][SP['h'][3:11,:]>0]==3)-np.mean(SB['h'][3:11,:][SB['h'][3:11,:]>0]==3)
    eff_points=np.mean(np.diff(SP['p'][3:11,:],axis=0))-np.mean(np.diff(SB['p'][3:11,:],axis=0))
    eff_h=(np.mean(SP['h'][3:11,:]==0)*0.0+np.mean(SP['h'][3:11,:]==1)*10.0+np.mean(SP['h'][3:11,:]==2)*20+np.mean(SP['h'][3:11,:]==3)*38.5)-\
          (np.mean(SB['h'][3:11,:]==0)*0.0+np.mean(SB['h'][3:11,:]==1)*10.0+np.mean(SB['h'][3:11,:]==2)*20+np.mean(SB['h'][3:11,:]==3)*38.5)
    
    #Print the point
    print("The point is {}, the moments are {}, {}, {}, {} , {}, {}, {}".format(pt,shpo,sh1,sh_min,eff_h,eff_e,eff_full,eff_points))   

        
    #return ((shpo-0.65)/0.65)**2+((sh1-0.1984)/0.1984)**2+((eff-0.1)/0.1)**2+((0.256-sh_min)/0.256)**2
    #return ((shpo-0.1956)/0.1956)**2+((sh1-0.1984)/0.1984)**2+((eff-0.1)/0.1)**2+((0.256-sh_min)/0.256)**2
    return ((shpo-0.1956)/0.1956)**2+((sh1-0.1984)/0.1984)**2+((eff_h-3)/3)**2+((0.256-sh_min)/0.256)**2
            
            
            
#Initialize seed
np.random.seed(10)


#Define initial point (xc) and boundaries (xl,xu)
xc=np.array([0.1372664,  0.5643915,  0.02987829, 0.2816896])#xc=np.array([0.16,0.6,0.0159,0.3]) #
xl=np.array([0.08,0.05,0.00,0.1])
xu=np.array([0.6,1.0,0.04,1.0])

#Optimization below
res=pybobyqa.solve(q, xc, rhobeg = 0.3, rhoend=1e-8, maxfun=200, bounds=(xl,xu),
                npt=len(xc)+5,scaling_within_bounds=True, seek_global_minimum=False,
                user_params={'tr_radius.gamma_dec':0.98,'tr_radius.gamma_inc':1.0,
                              'tr_radius.alpha1':0.9,'tr_radius.alpha2':0.95},
                objfun_has_noise=False)
 
#res = scipy.optimize.minimize(q,xc,bounds=list(zip(list(xl), list(xu))),method='Nelder-Mead',tol=1e-5)
#res = differential_evolution(q,bounds=list(zip(list(xl), list(xu))),disp=True,mutation=(0.1, 0.5),recombination=0.8) 
 

#The point is [0.09800833, 0.50142732, 0.00124261 ], the moments are 0.6376375, 0.3382125, 0.09378124999999998
  

# resu=res.x.copy()
# print(resu)
  


    
    
    