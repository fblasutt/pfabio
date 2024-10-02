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
xc=np.array([0.18984307, 0.39156348, 0.42656235, 0.38537558])#hours target, 7
xc=np.array([0.17315422, 0.34080107, 0.47102776, 0.6111401 ])#points target, 7

xl=np.array([0.01, 0.01, 0.01, 0.0]) 
xu=np.array([0.5 , 0.99, 1.99, 2.0]) 
 
 
#Function to minimize 
def q(pt,additional_tests=False): 
 
     
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
    SP= sim.simNoUncer_interp(p,ModP,Years=year,Tstart=ini_treated,Astart=SB['A'],Pstart=SB['pb3'],izstart=SB['iz']) 
    
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
    
    b2001_2003=np.array((year>=2001) & (year<=2003))
    b2004_2006=np.array((year>=2004) & (year<=2006))
    hours = co.hours_value(p,S,0,p.T) 
    employed=np.array(S['h']>0,dtype=np.float64) 
    not_marginal=np.array(S['h']>1,dtype=np.float64) 
    marginal=np.array(S['h']==1,dtype=np.float64) 
    full=np.array(S['h']>=3,dtype=np.float64) 
    earnings=S['wh']*p.scale 
    points=S['pb'] 
    points_behavioral=S['pb2'] 
     
     
    #isolate rich mothers 
    years_risk=(year<=2000) & (age>=3)  & (age<=10)# & (age>=agei) & (age<=agef) 
    rich= years_risk & (points>=1) 
     
    select_rich= (np.sum(rich,axis=0)==np.sum(years_risk,axis=0)) & (np.sum(rich,axis=0)>0) 
    keep_rich=np.repeat(select_rich[None,:],p.T,axis=0) 
     
     
    start=age==ini_treated 
    no_retroactive_points=np.repeat((SB['p'][start]==SB['pb3'][start])[None,:],p.T,axis=0) 
     
    sample=(((age>=3)  & (age<=10)) |\
           ((age>=15) & (age<=20) & no_retroactive_points  )) & (age>=agei) & (age<=agef)  
     
    df=np.array(np.stack((year.flatten(),   
                               age.flatten(),    
                               age_3_10.flatten(),    
                               after_2000.flatten(), 
                               b2001_2003.flatten(),
                               b2004_2006.flatten(),
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
   
    dfa=pd.DataFrame(data=df,columns=['year','age','age_3_10','after_2000','b2001_2003','b2004_2006','hours','employed','not_marginal','marginal','full','earnings','sample','points_behavioral','points'])         
                             
    formula='age_3_10*after_2000+age_3_10+after_2000+C(age)' 
    param='age_3_10:after_2000'
    
    formula='age_3_10*b2001_2003+age_3_10*b2004_2006+age_3_10+C(year)+C(age)' 
    param='age_3_10:b2004_2006'
    
    eff_h   =smf.ols(formula='hours ~'+formula,data = dfa[dfa['sample']==1]).fit().params[param] 
    eff_e   =smf.ols(formula='employed ~'+formula,data = dfa[dfa['sample']==1]).fit().params[param] 
    eff_full=smf.ols(formula='full ~'+formula,data = dfa[(dfa['sample']==1) & (dfa['employed']==1)]).fit().params[param] 
    eff_marg=smf.ols(formula='marginal ~'+formula,data = dfa[(dfa['sample']==1) & (dfa['employed']==1)]).fit().params[param] 
    eff_earn=smf.ols(formula='earnings ~'+formula,data = dfa[(dfa['sample']==1)]).fit().params[param] 
    eff_points=smf.ols(formula='points ~'+formula,data = dfa[(dfa['sample']==1)]).fit().params[param] 
    eff_points_behavioral=smf.ols(formula='points_behavioral ~'+formula,data = dfa[(dfa['sample']==1)]).fit().params[param] 
         
    # years=np.array(range(1995,2007)) 
    # import matplotlib.pyplot as plt 
    # pars=smf.ols(formula='hours ~ age_3_10*C(year,Treatment(reference=2000))+age_3_10+C(age)+C(year,Treatment(reference=2000))',data = dfa[dfa['sample']==1]).fit().params 
     
    # pars_array=np.array([pars['age_3_10:C(year, Treatment(reference=2000))[T.'+str(i)+'.0]'] for i in np.delete(years,5)]) 
    # plt.plot(np.delete(years,5),pars_array) 
    
    if additional_tests:
        
        #True effects below
        # group=(age>=3)  & (age<=10) & (year>=2001)
        # eff_ht=(co.hours_value(p,SP,0,p.T)[group]-co.hours_value(p,SB,0,p.T)[group]).mean()
        # eff_et=(SP['h'][group]>0).mean()-(SB['h'][group]>0).mean()
        # eff_earnt=(SP['wh'][group].mean()-SB['wh'][group].mean())*p.scale 
        # eff_fullt=(SP['h'][group]>=3).mean()-(SB['h'][group]>=3).mean()
        # eff_pointst=(SP['pb']-SB['pb'])[group].mean()
    
    
        #Table with parameters + targeted moments  
        def p42(x): return str('%4.2f' % x)  
        def p43(x): return str('%4.3f' % x)     
        def p40(x): return str('%4.0f' % x)  
        
        table=r'\begin{table}[htbp]'+\
                r'\caption{Model parameters and fit}\label{table:model_param}'+\
                r'\centering\footnotesize'+\
                r'\begin{tabular}{lcccc}'+\
                r' \toprule '+\
                r" Parameter & Value & \multicolumn{3}{c}{Target statistics}  \\\cline{3-5} "+\
                r" &  &  Name & Data & Model  \\"+\
                r'\midrule   '+\
                r' Cost of working - mini ($q_{10}$)   &'+p43(p.q[1]*p.qmean)+'& Share mini-jobs           & 0.26 &'+p42(sh_min)+'\\\\'+\
                r' Cost of working - part ($q_{20}$)   &'+p43(p.q[2]*p.qmean)+'& Share part-time           & 0.20 &'+p42(sh_part)+'\\\\'+\
                r' Cost of working - full ($q_{38.5}$)      &'+p43(p.qmean)+'& Share full time      & 0.20 &'+p42(sh_full)+'\\\\'+\
                r' Fixed effects dispersion ($\sigma_q$)   &'+p43(p.qvar)+'& Effect of the reform on hours  & 3.56 & '+p42(eff_h)+'\\\\'+\
                r'  \bottomrule'+\
              """\end{tabular}"""+\
              r'\end{table}' 
               
        #r' Fixed effects dispersion ($\sigma_q$)   &'+p43(p.qvar)+'& \\begin{tabular}{@{}c@{}}Effect of the reform on employment \\\\ Effect of the reform on hours\\end{tabular}   & \\begin{tabular}{@{}c@{}}0.06 \\\\ 2.31\\end{tabular}& \\begin{tabular}{@{}c@{}}'+p42(eff_e)+' \\\\'+p42(eff_h)+'\\end{tabular}\\\\'+\
        #Write table to tex file 
        with open('C:/Users/32489/Dropbox/occupation/model/pfabio/output/table_params.tex', 'w') as f: 
            f.write(table) 
            f.close() 
            
             
        ############################### 
        #Compute nontargeted moments 
        ############################## 
         
        def share(r,δ,T,k,per): 
            #Model-based annuitization to compute MPE: https://michael-graber.github.io/pdf/Golosov-Graber-Mogstad-Novgorodsky-2023.pdf pg 38 
            share = np.array([ ((1+r)/(1+δ))**(t+1)*(δ/(1+δ))*(1-(1/(1+δ))**(T-k))**-1  for t in range(per)]) 
             
            return share 
         
        adjust=np.ones(SP['c'].shape)/((1+p.r)**(np.cumsum(np.ones(p.T))-1.0))[:,None] 
        adjustr = adjust*(1+p.r)**3
         
        #MPE out of pension wealth, using tretroactive credits 
        SB_retro= sim.simNoUncer_interp(p,ModB,Tstart=np.zeros(p.N,dtype=np.int16)+3,Astart=SB['A']+1,Pstart=SB['p'],izstart=SB['iz']) 
         
        change_earn  =((np.nanmean((SB_retro['w'][3:11,:]*p.wls[SB_retro['h'][3:11,:]]*adjustr[3:11,:]).sum(axis=0)))-\
                        np.nanmean((SB['w'][3:11,:]      *p.wls[SB['h'][3:11,:]]      *adjustr[3:11,:]).sum(axis=0))) 
             
        
         
        #Below annuitization like in Golosov (2024), assuming that agents sommth consumption 
        change_pweal_s = (11-3)*((p.r/(1+p.r))*(1-(1/(1+p.r))**(p.T-3))**-1)#*np.mean((p.ρ*(SB['pb3']-SB['p'])*adjustr*(SB['ir']==1)).sum(axis=0)) 
         
         
        # #Below annuitization like in Golosov (2024), NOT assuming that agents sommth consumption if PIH where r could be different than δ 
        # change_pweal_s2 = share(p.r,p.δ,p.T,8,12-8).sum()#*np.mean((p.ρ*(SB['pb3']-SB['p'])*adjustr*(SB['ir']==1)).sum(axis=0)) 
         
        # #Below model-consistent annuitization, where wealth is allocated according to consumption path. How to get it, 
        # #use the intertemporal budget constraint and take out of summation c0 (future consumtion is replaced by ct/c0). 
        # #Then manage the intertemporal BC to have c0=stuff: use it to get annuity value of future pension wealth. then 
        # #sum the implied consumtion for periods 8 to 12. This can be checked against change_pweal_s2 
        # ct_over_c0_discounted=np.mean(SB['c'][8:,:],axis=1)/np.mean(SB['c'][8,:])*adjustr[8:,0] 
        # c0=np.mean((p.ρ*(SB['pb3']-SB['p'])*adjustr*(SB['ir']==1)).sum(axis=0))/ct_over_c0_discounted.sum() 
         
        # change_pweal_d = c0*((np.mean(SB['c'][8:,:],axis=1)/np.mean(SB['c'][8,:]))[:4]).sum() 
         
        #Finally, the marginal propensity to earn 
        MPE = change_earn/(change_pweal_s) 
    
        ############################################ 
        #Table with parameters 
        ###########################################      
        table=r'\begin{table}[htbp]'+\
                r'\caption{Non-targeted moments}\label{table:nontargeted_moments}'+\
                r'\centering\footnotesize'+\
                r'\begin{tabular}{lcc}'+\
                r' \toprule '+\
                r" Effect of the reform on &   Data & Model  \\"+\
                r'\midrule   '+\
                r' Pension points   & 0.15 &'+p42(eff_points)+'\\\\'+\
                r' Behavioral pension points   & 0.10 &'+p42(eff_points_behavioral)+'\\\\'+\
                r' Work full time    & 0.05 &'+p42(eff_e)+'\\\\'+\
                r' Marginal employment    & -0.12 &'+p42(eff_marg)+'\\\\'+\
                r' Non-marginal employment earnings (\euro)    & 2809 &'+p40(eff_earn)+'\\\\'+\
                r'Employed    & 0.10 &'+p42(eff_e)+'\\\\'+\
                r'\toprule   '+\
                r" Other moments &   Data & Model  \\"+\
                r'\midrule   '+\
                r' Marginal propensity to earn (MPE)      & -0.51\text{ to }-0.12 &'+p42(MPE)+'\\\\'+\
                r'  \bottomrule'+\
                r'\multicolumn{3}{l}{}'+\
              """\end{tabular}"""+\
              r'\end{table}' 
               
        #Write table to tex file 
        with open('C:/Users/32489/Dropbox/occupation/model/pfabio/output/table_nontargetd.tex', 'w') as f: 
            f.write(table) 
            f.close() 
             
         
    print("The point is {}, the moments are shfull {}, sh_part {}, sh_min {}, eff_h {} , eff_e {}, eff_full  {}, eff_marg {}, eff_earn {}, eff_points {}".format(pt,sh_full,sh_part,sh_min,eff_h,eff_e,eff_full,eff_marg,eff_earn,eff_points))    
     
     
    # print(np.array([((sh_full-.1984)/.1984)**2,((sh_part-.1986)/.1986)**2,((sh_min-.256)/.256)**2,((eff_h- 2.84)/ 2.84)**2,((eff_e-.0772)/.0772)**2]).sum())  
    # return [((sh_full-.1984)/.1984),((sh_part-.1986)/.1986),((sh_min-.256)/.256),((eff_h- 2.84)/ 2.84),((eff_e-.0772)/.0772)]         
 
    # print(np.array([((sh_full-.1984)/.1984)**2,((sh_part-.1986)/.1986)**2,((sh_min-.256)/.256)**2,((eff_h- 2.84)/ 2.84)**2,0*((eff_e-.0772)/.0772)**2]).sum())  
    # return [((sh_full-.1984)/.1984),((sh_part-.1986)/.1986),((sh_min-.256)/.256),((eff_h- 2.84)/ 2.84),0*((eff_e-.0772)/.0772)]         
 
 
    # print(np.array([((sh_full-.1984)/0.0058)**2,((sh_part-.1986)/0.00589)**2,((sh_min-.256)/0.00644)**2,((eff_h- 2.84)/0.822)**2,((eff_e-.0772)/.0257)**2]).sum())  
    # return [((sh_full-.1984)/0.0058),((sh_part-.1986)/0.00589),((sh_min-.256)/0.00644),((eff_h- 2.84)/0.822),((eff_e-.0772)/.0257)]         
    
             
    print(np.array([((sh_full-.1984)/.1984)**2,((sh_part-.1986)/.1986)**2,((sh_min-.256)/.256)**2,((eff_points- 0.153)/ 0.153)**2]).sum())  
    return [((sh_full-.1984)/.1984),((sh_part-.1986)/.1986),((sh_min-.256)/.256),((eff_points- 0.153)/ 0.153)]            
 
 
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
     
     
    res=dfols.solve(q, xc, rhobeg = 0.1, rhoend=1e-6, maxfun=300, bounds=(xl,xu), 
                    npt=len(xc)+5,scaling_within_bounds=True,  
                    user_params={'tr_radius.gamma_dec':0.98,'tr_radius.gamma_inc':1.0, 
                                  'tr_radius.alpha1':0.9,'tr_radius.alpha2':0.95}, 
                    objfun_has_noise=False) 
