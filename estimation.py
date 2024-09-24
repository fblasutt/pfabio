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
import pandas as pd
import statsmodels.formula.api as smf

#Actual Program

    
#Initialize seed
np.random.seed(10)
p = co.setup()


baseline_sample=np.array(pd.read_excel('frequencies.xlsx')) 
indexes=np.array(np.random.choice(baseline_sample[:,0], size=p.N, replace=True, p=baseline_sample[:,5]),dtype=np.int32) 
final_sample= baseline_sample[:,1:-1][indexes] 
 
treated=np.repeat(final_sample[:,0][:,None],p.T,axis=1).T
agei=np.repeat(final_sample[:,1][:,None],p.T,axis=1).T
agef=np.repeat((final_sample[:,1]+final_sample[:,3]-final_sample[:,2])[:,None],p.T,axis=1).T
year=(np.cumsum(np.ones(p.T))-1)[:,None]-agei+np.repeat(final_sample[:,2][:,None],p.T,axis=1).T
age=(np.cumsum(np.ones((p.N,p.T)),axis=1)-1).T
treatment=((age>=3)  & (year>=2001))
after_treatment=((age>=3)  & (year>=2001))

#Define initial point (xc) and boundaries (xl,xu)
xc=np.array([0.17897197, 0.28365274, 0.41639799, 0.61443697])#good for old targets
xc=np.array([0.17767284, 0.2962443 , 0.41233575, 0.58322198])#good for new targets
xc=np.array([0.18812077, 0.30812166, 0.41186075, 0.5675635 ])#good for new moments with variance weighting 
xc=np.array([0.1902654 , 0.30777262, 0.4160138 , 0.57285105])#good for nex moments excluding employment

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
    SB= sim.simNoUncer_interp(p,ModB,Tstart=np.zeros(p.N,dtype=np.int16),Astart=p.startA,Pstart=np.ones((p.T,p.N))*p.startP,izstart=p.tw)

    #Pension reform
    ini_treated=p.T-np.cumsum(treatment==True,axis=0)[-1,:]#first child age at which women are treated
    SP= sim.simNoUncer_interp(p,ModP,Tstart=ini_treated,Astart=SB['A'],Pstart=SB['pb3'],izstart=SB['iz'])
   
    #create unique dictionary for relevant simulated data
    S=dict()
    for i in ['h','wh','pb2','pb','c']:
        S[i]=np.zeros((p.T,p.N))
        S[i][after_treatment] =SP[i][after_treatment]
        S[i][~after_treatment]=SB[i][~after_treatment]
        
    ###################################################
    #Average labor supply in 2000 for treatment group #
    ###################################################
    sh_part=np.mean(S['h'][((year==2000) & (age>=3) & (age<=10))]==2)
    sh_full=np.mean(S['h'][((year==2000) & (age>=3) & (age<=10))]>=3)
    sh_min=np.mean(S['h'][((year==2000) & (age>=3) & (age<=10))]==1)
    mean_earnings=np.mean(S['wh'][((year==2000) & (age>=3) & (age<=10))])*p.scale
    mean_points=np.mean(S['pb'][((year==2000) & (age>=3) & (age<=10))])
    
    #####################################
    #Difference in differences analysis
    #####################################
    
    #Relevant variables not yet creted
    age_3_10=np.zeros((p.T,p.N))
    age_3_10[(age>=3) & (age<=10)]=1
    after_2000=np.array(year>=2001)
    hours = co.hours_value(p,S,0,p.T)
    employed=np.array(S['h']>0,dtype=np.float64)
    not_marginal=np.array(S['h']>1,dtype=np.float64)
    marginal=np.array(S['h']==1,dtype=np.float64)
    full=np.array(S['h']>=3,dtype=np.float64)
    earnings=S['wh']*p.scale
    points=S['pb']
    points_behavioral=S['pb2']
    sample=(((age>=3)  & (age<=10)) |\
           ((age>=15) & (age<=20))) & (age>=agei) & (age<=agef) 
    
    df=np.array(np.stack((year.flatten(),  
                               age.flatten(),   
                               age_3_10.flatten(),   
                               after_2000.flatten(),
                               hours.flatten(),
                               employed.flatten(),
                               not_marginal.flatten(),
                               marginal.flatten(),
                               full.flatten(),
                               earnings.flatten(),
                               sample.flatten(),
                               points_behavioral.flatten(),
                               points.flatten()),
                               axis=0).T,dtype=np.float64)        
  
    dfa=pd.DataFrame(data=df,columns=['year','age','age_3_10','after_2000','hours','employed','not_marginal','marginal','full','earnings','sample','points_behavioral','points'])        
                            
    
    formula='age_3_10*after_2000+age_3_10+after_2000+C(age)'
    eff_h   =smf.ols(formula='hours ~'+formula,data = dfa[dfa['sample']==1]).fit().params['age_3_10:after_2000']
    eff_e   =smf.ols(formula='employed ~'+formula,data = dfa[dfa['sample']==1]).fit().params['age_3_10:after_2000']
    eff_full=smf.ols(formula='full ~'+formula,data = dfa[(dfa['sample']==1) & (dfa['employed']==1)]).fit().params['age_3_10:after_2000']
    eff_marg=smf.ols(formula='marginal ~'+formula,data = dfa[(dfa['sample']==1) & (dfa['employed']==1)]).fit().params['age_3_10:after_2000']
    eff_earn=smf.ols(formula='earnings ~'+formula,data = dfa[(dfa['sample']==1) & (dfa['employed']==1)]).fit().params['age_3_10:after_2000']
    eff_points=smf.ols(formula='points ~'+formula,data = dfa[(dfa['sample']==1)]).fit().params['age_3_10:after_2000']
    eff_points_behavioral=smf.ols(formula='points_behavioral ~'+formula,data = dfa[(dfa['sample']==1)]).fit().params['age_3_10:after_2000']
        
    # years=np.array(range(1995,2007))
    # import matplotlib.pyplot as plt
    # pars=smf.ols(formula='hours ~ age_3_10*C(year,Treatment(reference=2000))+age_3_10+C(age)+C(year,Treatment(reference=2000))',data = dfa[dfa['sample']==1]).fit().params
    
    # pars_array=np.array([pars['age_3_10:C(year, Treatment(reference=2000))[T.'+str(i)+'.0]'] for i in np.delete(years,5)])
    # plt.plot(np.delete(years,5),pars_array)
    
    
    print("The point is {}, the moments are shfull {}, sh_part {}, sh_min {}, eff_h {} , eff_e {}, eff_full  {}, eff_marg {}, eff_earn {}, eff_points {}".format(pt,sh_full,sh_part,sh_min,eff_h,eff_e,eff_full,eff_marg,eff_earn,eff_points))   


    # print(np.array([((sh_full-.1984)/.1984)**2,((sh_part-.1986)/.1986)**2,((sh_min-.256)/.256)**2,((eff_h- 2.84)/ 2.84)**2,((eff_e-.0772)/.0772)**2]).sum()) 
    # return [((sh_full-.1984)/.1984),((sh_part-.1986)/.1986),((sh_min-.256)/.256),((eff_h- 2.84)/ 2.84),((eff_e-.0772)/.0772)]        

    print(np.array([((sh_full-.1984)/.1984)**2,((sh_part-.1986)/.1986)**2,((sh_min-.256)/.256)**2,((eff_h- 2.84)/ 2.84)**2,0*((eff_e-.0772)/.0772)**2]).sum()) 
    return [((sh_full-.1984)/.1984),((sh_part-.1986)/.1986),((sh_min-.256)/.256),((eff_h- 2.84)/ 2.84),0*((eff_e-.0772)/.0772)]        


    # print(np.array([((sh_full-.1984)/0.0058)**2,((sh_part-.1986)/0.00589)**2,((sh_min-.256)/0.00644)**2,((eff_h- 2.84)/0.822)**2,((eff_e-.0772)/.0257)**2]).sum()) 
    # return [((sh_full-.1984)/0.0058),((sh_part-.1986)/0.00589),((sh_min-.256)/0.00644),((eff_h- 2.84)/0.822),((eff_e-.0772)/.0257)]        
   
            
    # print(np.array([((sh_full-.1984)/.1984)**2,((sh_part-.1986)/.1986)**2,((sh_min-.256)/.256)**2,((eff_h- 2.317)/ 2.317)**2,((eff_e-.064)/.064)**2]).sum()) 
    # return [((sh_full-.1984)/.1984),((sh_part-.1986)/.1986),((sh_min-.256)/.256),((eff_h- 2.317)/ 2.317),((eff_e-.064)/.064)]           


# [ 0.40706012  0.03525281 -0.51941101  0.00186123  1.60048109  0.03695673] first tentative σ=0.0005
# 0.37349381, -0.01739811, -0.6       ,  0.00287586,  1.59080139, 0.03220926] current
#Optimization below


import numpy as np


if __name__ == '__main__':
    

    # computation_options = { "num_workers" : 16,        # use four processes in parallel
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
