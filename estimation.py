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


# perfectly identified
xc=np.array([0.19982245, 0.38350866, 0.2363179 , 0.56099277, 1.74864462])#r 0.04, delta 0.02
xc=np.array([0.14529946, 0.42081934, 0.99172148, 0.43656438, 0.80467666])#r 0.04, delta 0.01
xc=np.array([0.16950232, 0.28640178, 0.46205568, 0.73043218, 1.65330702])#r 0.03, delta 0.02
xc=np.array([0.18656332, 0.33594766, 0.33601567, 0.63696179, 1.72893227])#r 0.03, delta 0.01

# full employment is a target
xc=np.array([0.05256064, 0.08915266, 0.79875736, 0.96143436, 1.632832  ])#r 0.03, delta 0.01
xc=np.array([0.18802455, 0.3653759,  1.06389732, 0.50306275])#r 0.03, delta 0.02 -server 7010102
xc=np.array([0.17357468, 0.46547367, 0.69454225, 0.46659082])#r 0.04, delta 0.02 -server 7010101
#xc=np.array([0.15472905, 0.420859,   0.29382598, 0.57752101, 1.20152824])#r 0.04, delta 0.01 -server 7010099

# full employment is a target + more gridpoints for assets
#r 0.03, delta 0.01 - server 7011108
#r 0.03, delta 0.02 -server 7010896
#r 0.04, delta 0.02 -server 7011120
#r 0.04, delta 0.01 -server 7011127

# full employment is a target + more gridpoints for assets + CRRA=1 (r=0.03 does not work well)
#xc=np.array([0.11594253, 0.244793,   1.02060783, 0.8120724 ])#r 0.03, delta 0.01 - server 7011565
#xc=np.array([0.16514195, 0.38928563, 0.47987529, 0.6016247 ])#r 0.03, delta 0.02 -server 7011563
#xc=np.array([0.16848745, 0.4529646,  0.46272694, 0.49871028])#r 0.04, delta 0.02 -server 7011536
#xc=np.array([0.14458546, 0.41952063, 0.53731818, 0.4611148 ])#r 0.04, delta 0.01 -server 7011506

# full employment is a target + more gridpoints for assets + CRRA=1.5 - did not work well...

# full employment is NOT a target + more gridpoints for assets + CRRA=1.0 - did not work well...
xc=np.array([0.17206982, 0.44021362, 0.41734822, 0.44760134])# server r=0.03 delta 0.02 7012537
#xc=np.array([0.16585031, 0.39283926, 0.5159664,  0.39508767])# server r=0.03 delta 0.01 7012528
#xc=np.array([0.15425383, 0.43292663, 0.45066023, 0.42982237])# server r=0.04 delta 0.02 7012275
#xc=np. array([0.14659822, 0.41706892, 0.54015245, 0.44289126])# server r=0.04 delta 0.01 7012444

# full employment is NOT a target + more gridpoints for assets + CRRA=1.5 - did not work well...

xc=np.array([0.15856754, 0.33206964, 0.10493158, 0.64323186])# server r=0.03 delta 0.02 7071015
# [0.14899152 0.31280701 0.12715104 0.66284272] server r=0.03 delta 0.01 7071019
# [0.12659703 0.30590884 0.13007448 0.7038301 ] server r=0.04 delta 0.01 7071021
#xc=np.array([0.13297845, 0.32533234, 0.10953125, 0.68818306])# server r=0.04 delta 0.02 7071023

xl=np.array([0.01, 0.01, 0.01, 0.0])
xu=np.array([0.5 , 0.99, 1.99, 1.0])


#Function to minimize
def q(pt):

    
    #Define main parameters
    p = co.setup()
    
    #..and update them
    p.q =np.array([0.0,pt[1],pt[0],1.0])   #Fixed cost of pticipation - mean
    
    p.qmean=pt[2]
    
    p.qvar =pt[3] #Fixed cost of pticipation -sd 
   
    
    #Disutility from working
    p.q_grid=np.zeros((p.nq,p.nwls,p.nw))
    # p.q_grid_π=np.zeros((p.nq,p.nw))
    # p.q_gridt,_=addaco_dist(p.σq,0.0,p.nq)
    
    p.q_gridt = np.linspace(p.qmean-p.qmean*p.qvar,p.qmean+p.qmean*p.qvar,p.nq)#co.dist_gamma(p.qshape,p.qscale,p.nq)

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
        
    print(np.array([((sh_full-.1984)/.1984)**2,((sh_part-.1986)/.1986)**2,((sh_min-.256)/.256)**2,((eff_h- 2.317)/ 2.317)**2,((eff_e-.064)/.064)**2,0*((eff_nomarg+.0705)/.0705)**2]).sum())
    return [((sh_full-.1984)/.1984),((sh_part-.1986)/.1986),((sh_min-.256)/.256),((eff_h- 2.317)/ 2.317),((eff_e-.064)/.064),0*((eff_nomarg+.0705)/.0705)]          
            


# [ 0.40706012  0.03525281 -0.51941101  0.00186123  1.60048109  0.03695673] first tentative σ=0.0005
# 0.37349381, -0.01739811, -0.6       ,  0.00287586,  1.59080139, 0.03220926] current
#Optimization below


import numpy as np


if __name__ == '__main__':
    

    # computation_options = { "num_workers" : 8,        # use four processes in parallel
    #                         "working_dir" : "working" # where to save results in progress (in case interrupted)
    #                         }
    
    # global_search_options = { "num_points" : 11}  # number of points in global pre-test
    
    # local_search_options = {  "algorithm"    : "dfols", # local search algorithm
    #                                                       # can be either BOBYQA from NLOPT or NelderMead from scipy
    #                           "num_restarts" : 16,      # how many local searches to do
    #                           "shrink_after" : 16,       # after the first [shrink_after] restarts we begin searching
    #                                                       # near the best point we have found so far
    #                           "xtol_rel"     : 1e-6,     # relative tolerance on x
    #                           "ftol_rel"     : 1e-6     # relative tolerance on f
    #                         }
    
    # opt = TikTak.TTOptimizer(computation_options, global_search_options, local_search_options, skip_global=False
    #                           )
    # x,fx = opt.minimize(q,xl,xu)
    # print(f'The minimizer is {x}')
    # print(f'The objective value at the min is {fx}')
    
    
    res=dfols.solve(q, xc, rhobeg = 0.3, rhoend=1e-6, maxfun=300, bounds=(xl,xu),
                    npt=len(xc)+5,scaling_within_bounds=True, 
                    user_params={'tr_radius.gamma_dec':0.98,'tr_radius.gamma_inc':1.0,
                                  'tr_radius.alpha1':0.9,'tr_radius.alpha2':0.95},
                    objfun_has_noise=False)
