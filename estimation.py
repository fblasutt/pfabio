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
import TikTak
#Actual Program

    
#Initialize seed
np.random.seed(10)


#Define initial point (xc) and boundaries (xl,xu)

# [1.01608098 0.12180904 0.20933763 0.76992266 1.74844589]



xc=np.array([0.09900606, 0.15953799, 0.78209658, 1.86331691])#marginal is target

#xc=np.array([0.15125272, 0.41818386, 0.97546089, 0.91376809])#full is target

xl=np.array([0.01,0.01,0.1,0.1])
xu=np.array([0.5,0.99   ,2.6 ,7.5])


#Function to minimize
def q(pt):

    
    #Define main parameters
    p = co.setup()
    
    #..and update them
    p.q =np.array([0.0,pt[1],pt[0],1.0])   #Fixed cost of pticipation - mean
    
    p.qshape=pt[2]
    
    p.qscale =pt[3] #Fixed cost of pticipation -sd 
    #p.δ=pt[0]
    
    #Disutility from working
    p.q_grid=np.zeros((p.nq,p.nwls,p.nw))
    # p.q_grid_π=np.zeros((p.nq,p.nw))
    # p.q_gridt,_=addaco_dist(p.σq,0.0,p.nq)
    
    p.q_gridt = co.dist_gamma(p.qshape,p.qscale,p.nq)

    for il in range(1,p.nwls):
        for iw in range(p.nw):
            for iq in range(p.nq):
                
                p.q_grid[iq,il,iw]= p.q_gridt[iq]*p.q[il]
   

    
    #Pension reform
    ModP= sol.solveEulerEquation(p,model='pension reform')

    #Baseline
    ModB = sol.solveEulerEquation(p,model='baseline')
  

    #Baseline
    SB= sim.simNoUncer_interp(p,ModB,Tstart=0,Astart=p.startA,Pstart=np.ones(p.N)*p.startP,izstart=p.tw)

    #Pension reform
    SP= sim.simNoUncer_interp(p,ModP,Tstart=8,Astart=SB['A'][8,:],Pstart=SB['pb3'][8,:],izstart=SB['iz'][8,:])
   
   
    
    #
    sh_part=np.mean(SB['h'][7,:]==2)
    sh_full=np.mean(SB['h'][7,:]>=3)
    sh_min=np.mean(SB['h'][7,:]==1)
    sh_noem=np.mean(SB['h'][7,:]==0)
    sh_h=co.hours(p,SB,7,8)
    
    s_hl=np.mean(SB['wh'][7,:])*p.scale
    
    eff_e=np.mean(SP['h'][8:12,:]>0)-np.mean(SB['h'][8:12,:]>0)
    eff_full=np.mean(SP['h'][8:12,:][SP['h'][8:12,:]>0]==3)-np.mean(SB['h'][8:12,:][SB['h'][8:12,:]>0]==3)
    eff_nomarg=np.mean(SP['h'][8:12,:][SP['h'][8:12,:]>0]==1)-np.mean(SB['h'][8:12,:][SB['h'][8:12,:]>0]==1)
    eff_points=np.mean(np.diff(SP['p'][8:12,:],axis=0))-np.mean(np.diff(SB['p'][8:12,:],axis=0))

    eff_h = co.hours(p,SP,8,12)-co.hours(p,SB,8,12)
    eff_earn=np.nanmean(np.diff(SP['p'][8:12,:],axis=0))-np.nanmean(np.diff(SB['p'][8:12,:],axis=0))-np.mean(SP['pb'][8:12,:])
    
    pension_points=np.nanmean(np.diff(SB['p'][7:9,:],axis=0))
    #Print the point
    print("The point is {}, the moments are shfull {}, sh_part {}, sh_min {}, eff_h {} , eff_e {}, eff_full  {}, eff_marg {}, eff_nomarg {}, eff_points {}, s_hl {}".format(pt,sh_full,sh_part,sh_min,eff_h,eff_e,eff_full,eff_nomarg,eff_earn,eff_points,s_hl))   

    #0.015*10+0.045*38.5+(0.099-0.045)*20 = 2.9625
    #(0.015+0.02)/2*10+(0.045+0.016)/2*38.5+((0.099+0.029)/2-(0.045+0.016)/2)*20 = 2.01925
    
    
   
    # print(np.array([((sh_full-.1984)/.1984)**2,((sh_part-.1986)/.1986)**2,((sh_min-.256)/.256)**2,((eff_h- 2.317)/ 2.317)**2,((eff_e-.064)/.064)**2,((eff_full+.0705)/.0705)**2]).sum())
    # return [((sh_full-.1984)/.1984),((sh_part-.1986)/.1986),((sh_min-.256)/.256),((eff_h- 2.317)/ 2.317),((eff_e-.064)/.064),((eff_full+.0705)/.0705)]
        
    print(np.array([((sh_full-.1984)/.1984)**2,((sh_part-.1986)/.1986)**2,((sh_min-.256)/.256)**2,((eff_h- 2.317)/ 2.317)**2,((eff_e-.064)/.064)**2]).sum())
    return [((sh_full-.1984)/.1984),((sh_part-.1986)/.1986),((sh_min-.256)/.256),((eff_h- 2.317)/ 2.317),((eff_e-.064)/.064)]          
            


# [ 0.40706012  0.03525281 -0.51941101  0.00186123  1.60048109  0.03695673] first tentative σ=0.0005
# 0.37349381, -0.01739811, -0.6       ,  0.00287586,  1.59080139, 0.03220926] current
#Optimization below


import numpy as np


if __name__ == '__main__':
    

    computation_options = { "num_workers" : 30,        # use four processes in parallel
                            "working_dir" : "working" # where to save results in progress (in case interrupted)
                            }
    
    global_search_options = { "num_points" : 13}  # number of points in global pre-test
    
    local_search_options = {  "algorithm"    : "dfols", # local search algorithm
                                                          # can be either BOBYQA from NLOPT or NelderMead from scipy
                              "num_restarts" : 90,      # how many local searches to do
                              "shrink_after" : 60,       # after the first [shrink_after] restarts we begin searching
                                                          # near the best point we have found so far
                              "xtol_rel"     : 1e-6,     # relative tolerance on x
                              "ftol_rel"     : 1e-6     # relative tolerance on f
                            }
    
    opt = TikTak.TTOptimizer(computation_options, global_search_options, local_search_options, skip_global=False)
    x,fx = opt.minimize(q,xl,xu)
    print(f'The minimizer is {x}')
    print(f'The objective value at the min is {fx}')
    
    
    # res=dfols.solve(q, xc, rhobeg = 0.3, rhoend=1e-6, maxfun=300, bounds=(xl,xu),
    #                 npt=len(xc)+5,scaling_within_bounds=True, 
    #                 user_params={'tr_radius.gamma_dec':0.98,'tr_radius.gamma_inc':1.0,
    #                               'tr_radius.alpha1':0.9,'tr_radius.alpha2':0.95},
    #                 objfun_has_noise=False)
