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
import dfols

#Actual Program

    

#Function to minimize
def q(pt):

    #Define main parameters
    p = co.setup()
    
    #..and update them
    p.q =np.array([0.0,pt[2]*pt[0],pt[1]*pt[0],pt[0]])  #Fixed cost of pticipation - mean
    
    p.δ=pt[3]
    #p.ρq=pt[5]
    p.σq =pt[4] #Fixed cost of pticipation -sd 
    
    p.q_gridt,p.Πq=co.addaco_dist(p.σq,p.nq) 
    p.q_grid=np.ones((p.nq,p.nwls))
    for iq in range(p.nq):
        for il in range(p.nwls):
            p.q_grid[iq,il] = p.q[il]
            if il<1: p.q_grid[iq,il] = p.q[il]+p.q_gridt[iq]
   

    
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
    sh_h=co.hours(p,SB,7,8)
    
    s_hl=np.mean(SB['wh'][7,:])*p.scale
    
    eff_e=np.mean(SP['h'][8:12,:]>0)-np.mean(SB['h'][8:12,:]>0)
    eff_full=np.mean(SP['h'][8:12,:][SP['h'][8:12,:]>0]==3)-np.mean(SB['h'][8:12,:][SB['h'][8:12,:]>0]==3)
    eff_nomarg=np.mean(SP['h'][8:12,:][SP['h'][8:12,:]>0]==1)-np.mean(SB['h'][8:12,:][SB['h'][8:12,:]>0]==1)
    eff_points=np.mean(np.diff(SP['p'][8:12,:],axis=0))-np.mean(np.diff(SB['p'][8:12,:],axis=0))
    eff_h=(np.mean(SP['h'][8:12,:]==1)*10.0+np.mean(SP['h'][8:12,:]==2)*20.0+np.mean(SP['h'][8:12,:]==3)*38.5)-\
          (np.mean(SB['h'][8:12,:]==1)*10.0+np.mean(SB['h'][8:12,:]==2)*20.0+np.mean(SB['h'][8:12,:]==3)*38.5)
    eff_earn=np.nanmean(np.diff(SP['p'][8:12,:],axis=0))-np.nanmean(np.diff(SB['p'][8:12,:],axis=0))-np.mean(SP['pb'][8:12,:])
    
    pension_points=np.nanmean(np.diff(SB['p'][7:9,:],axis=0))
    #Print the point
    print("The point is {}, the moments are {}, {}, {}, {} , {},  {}, {}, {}".format(pt,sh1,shpo,sh_min,eff_h,eff_e,eff_full,eff_nomarg,eff_earn))   

        
    #return ((shpo-0.65)/0.65)**2+((sh1-0.1984)/0.1984)**2+((eff-0.1)/0.1)**2+((0.256-sh_min)/0.256)**2
    #return ((shpo-0.1956)/0.1956)**2+((sh1-0.1984)/0.1984)**2+((eff-0.1)/0.1)**2+((0.256-sh_min)/0.256)**2
    #return ((sh_h-13.96)/13.96)**2+((sh_noem-0.36)/0.36)**2+((0.256-sh_min)/0.256)**2+((eff_h-3.565)/3.565)**2+((0.099-eff_e)/0.099)**2
    return [((sh1-.1984)/.1984),((shpo-.1986)/.1986),((sh_min-.256)/.256),((eff_nomarg+.115)/.115),((eff_e-.099)/.099)]#,((eff_full-0.045)/0.045),((eff_nomarg+0.115)/0.115)]#,((sh_hl-6108)/6108)]#,((eff_marg+0.115)/0.115)]#+((0.099-eff_e)/0.099)**2
    #return ((sh_min-.256)/.256)**2+((shpo-.1986)/.1986)**2+((sh1-.1984)/.1984)**2+((eff_e-0.099)/0.099)**2+((eff_full-0.045)/0.045)**2+((eff_nomarg+0.115)/0.115)**2#        
            
            
#Initialize seed
np.random.seed(10)


#Define initial point (xc) and boundaries (xl,xu)

#[0.76349505, 0.47244936, 0.24951372, 0.01017474, 0.11123974]
xc=np.array([0.38291281, 0.29281174, 0.33105918, 0.02279778, 0.09087302])
xl=np.array([0.05,-0.4,-0.6,-0.05,0.0001])
xu=np.array([2.5 ,0.9,0.5,0.04,0.8])



#Optimization below
res=dfols.solve(q, xc, rhobeg = 0.3, rhoend=1e-5, maxfun=200, bounds=(xl,xu),
                npt=len(xc)+5,scaling_within_bounds=True, 
                user_params={'tr_radius.gamma_dec':0.98,'tr_radius.gamma_inc':1.0,
                              'tr_radius.alpha1':0.9,'tr_radius.alpha2':0.95},
                objfun_has_noise=False)
#q([0.20127328, 1.41087164, 0.00589961, 0.25583906])
#res = scipy.optimize.minimize(q,xc,bounds=list(zip(list(xl), list(xu))),method='Nelder-Mead',tol=1e-5)
#res = differential_evolution(q,bounds=list(zip(list(xl), list(xu))),disp=True,mutation=(0.1, 0.5),recombination=0.8) 
 

#The point is [0.09800833, 0.50142732, 0.00124261 ], the moments are 0.6376375, 0.3382125, 0.09378124999999998
  

# resu=res.x.copy()
# print(resu)
  


    
    
    