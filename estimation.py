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
    SP= sim.simNoUncer_interp(p,ModP,Tstart=8,Astart=SB['A'][8,:],Pstart=SB['p'][8,:])
   
   
    
    #
    shpo=np.mean(SB['h'][7,:]==2)
    sh1=np.mean(SB['h'][7,:]>=3)
    sh_min=np.mean(SB['h'][7,:]==1)
    sh_noem=np.mean(SB['h'][7,:]==0)
    sh_h=np.mean(SB['h'][8:12,:]==1)*10.0+np.mean(SB['h'][8:12,:]==2)*19.25+np.mean(SB['h'][8:12,:]==3)*28.875+np.mean(SB['h'][8:12,:]==4)*38.5
    eff_e=np.mean(SP['h'][8:12,:]>0)-np.mean(SB['h'][8:12,:]>0)
    eff_full=np.mean(SP['h'][8:12,:][SP['h'][8:12,:]>0]==3)-np.mean(SB['h'][8:12,:][SB['h'][8:12,:]>0]==3)
    eff_points=np.mean(np.diff(SP['p'][8:12,:],axis=0))-np.mean(np.diff(SB['p'][8:12,:],axis=0))
    eff_h=(np.mean(SP['h'][8:12,:]==1)*10.0+np.mean(SP['h'][8:12,:]==2)*19.25+np.mean(SP['h'][8:12,:]==3)*28.875+np.mean(SP['h'][8:12,:]==4)*38.5)-\
          (np.mean(SB['h'][8:12,:]==1)*10.0+np.mean(SB['h'][8:12,:]==2)*19.25+np.mean(SB['h'][8:12,:]==3)*28.875+np.mean(SB['h'][8:12,:]==4)*38.5)
    eff_earn=np.nanmean(np.diff(SP['p'][8:13,:],axis=0))-np.nanmean(np.diff(SB['p'][8:13,:],axis=0))-np.mean(SP['pb'][8:12,:])
    #Print the point
    print("The point is {}, the moments are {}, {}, {}, {} , {}, {}".format(pt,sh_h,sh_noem,sh_min,eff_h,eff_e,eff_earn))   

        
    #return ((shpo-0.65)/0.65)**2+((sh1-0.1984)/0.1984)**2+((eff-0.1)/0.1)**2+((0.256-sh_min)/0.256)**2
    #return ((shpo-0.1956)/0.1956)**2+((sh1-0.1984)/0.1984)**2+((eff-0.1)/0.1)**2+((0.256-sh_min)/0.256)**2
    return ((sh_h-13.96)/13.96)**2+((sh_noem-0.36)/0.36)**2+((0.256-sh_min)/0.256)**2+((eff_h-3.565)/3.565)**2#+((0.099-eff_e)/0.099)**2
            
            
            
#Initialize seed
np.random.seed(10)


#Define initial point (xc) and boundaries (xl,xu)
xc=np.array([0.250, 1.07, 0.0143, 0.324])
xl=np.array([0.08,0.05,0.00,0.1])
xu=np.array([0.6,1.2,0.07,1.3])


#Optimization below
# res=pybobyqa.solve(q, xc, rhobeg = 0.3, rhoend=1e-5, maxfun=200, bounds=(xl,xu),
#                 npt=len(xc)+5,scaling_within_bounds=True, seek_global_minimum=True,
#                 user_params={'tr_radius.gamma_dec':0.98,'tr_radius.gamma_inc':1.0,
#                               'tr_radius.alpha1':0.9,'tr_radius.alpha2':0.95},
#                 objfun_has_noise=False)
 
res = scipy.optimize.minimize(q,xc,bounds=list(zip(list(xl), list(xu))),method='Nelder-Mead',tol=1e-5)
#res = differential_evolution(q,bounds=list(zip(list(xl), list(xu))),disp=True,mutation=(0.1, 0.5),recombination=0.8) 
 

#The point is [0.09800833, 0.50142732, 0.00124261 ], the moments are 0.6376375, 0.3382125, 0.09378124999999998
  

# resu=res.x.copy()
# print(resu)
  


    
    
    