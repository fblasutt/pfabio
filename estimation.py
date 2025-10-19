# -*- coding: utf-8 -*- 
""" 
Created on Mon Dec 26 14:56:15 2022 
 
@author: Fabio 
""" 
 
 
import co # user defined functions 
import sol 
import sim 
import numpy as np 
#import pybobyqa 
#from scipy.optimize import dual_annealing,differential_evolution 
#import scipy 
import dfols 
import TikTak 
import pandas as pd 
import statsmodels.formula.api as smf 

from consav.grids import nonlinspace
 
import statsmodels.api as sm 
from pyhdfe import create

#Actual Program 
 
     
#Initialize seed 
np.random.seed(10) 
p = co.setup() 
 
 
# baseline_sample=np.array(pd.read_excel('frequencies.xlsx'))  
# indexes=np.array(np.random.choice(baseline_sample[:,0], size=p.N, replace=True, p=baseline_sample[:,5]),dtype=np.int32)  
# final_sample= baseline_sample[:,1:-1][indexes]  
  
# treated=np.repeat(final_sample[:,0][:,None],p.T,axis=1).T 
# agei=np.repeat(final_sample[:,1][:,None],p.T,axis=1).T 
# agef=np.repeat((final_sample[:,1]+final_sample[:,3]-final_sample[:,2])[:,None],p.T,axis=1).T 
# year=(np.cumsum(np.ones(p.T))-1)[:,None]-agei+np.repeat(final_sample[:,2][:,None],p.T,axis=1).T 
# age=(np.cumsum(np.ones((p.N,p.T)),axis=1)-1).T 
# treatment=((age>=3)  & (year>=2001)) 
# after_treatment=((age>=3)  & (year>=2001)) 
 
# #Define initial point (xc) and boundaries (xl,xu) 


 
baseline_sample = np.array([ 
    [0, 1981, 7051.6918], 
    [1, 1982, 7388.0949], 
    [2, 1983, 5569.5051], 
    [3, 1984, 6784.1683], 
    [4, 1985, 7075.0722], 
    [5, 1986, 7494.654], 
    [6, 1987, 7474.4081], 
    [7, 1988, 16049.141], 
    [8, 1989, 8315.3082], 
    [9, 1990, 8069.842], 
    [10, 1998, 6921.9224] 
]) 
 
#transform frequency in probabilites that sums up to 1  
baseline_sample[:,-1]=baseline_sample[:,-1]/baseline_sample[:,-1].sum() 
 
indexes=np.array(np.random.choice(baseline_sample[:,0], size=p.N, replace=True, p=baseline_sample[:,2]),dtype=np.int32)   
final_sample= baseline_sample[:,1:-1][indexes][:,0] 
age=(np.cumsum(np.ones((p.N,p.T)),axis=1)-1).T  
year=final_sample+age


#GREAT BELOW!!!
xc=np.array([0.1310856 , 0.3202975 , 0.4121633 , 0.98693138])

#Updated with real effect, not did
xc=np.array([0.17592452, 0.3555737,  0.43904321, 0.78818227])

#xc=np.array([-0.06775778,  0.16516238,  0.32015753,  1.92032006])

xc=np.array([-0.0529766 ,  0.18276247,  0.3285415 ,  1.8103877 ])

xc=np.array([0.1529766 ,  0.18276247,  0.3285415 ,  .303877 ])

xc=np.array([0.34560199, 0.52, 0.43,2.5])

xc=np.array([0.29504107, 0.4849266,  0.44567841, 1.37806935])

xc=np.array([0.34144022, 0.50589427, 0.44879404, 2.19344653])

xc=np.array([0.3535853,  0.51203419, 0.47489412, 1.64679895])
xc=np.array([0.3535853,  0.51203419, 0.47489412, 1.64679895])

#xc=np.array([0.31870392, 0.48824139, 0.43375664, 3.09494168])

xc=np.array([0.27023343, 0.4760718,  0.55284867, 4.62954094])

xc=np.array([0.22501409, 0.48762826, 0.5971833,  7.4069089])

xc=np.array([0.27992511, 0.48387357, 0.57024019, 3.31312607])

xc=np.array([0.23921454, 0.46522483, 0.52899638, 3.5481326])

xc=np.array([0.23774461, 0.4633808,  0.52856484, 3.27975242])

xc=np.array([0.27416538, 0.46535092, 0.54906985, 2.86587419])

xc=np.array([0.27685536, 0.46364342, 0.541645,   3.18341448])

xc=np.array([0.28340884, 0.46445241, 0.51329834, 4.39689782])

xc=np.array([0.26938113, 0.44570762, 0.4990007,  3.91073213])

xc=np.array([0.26753108, 0.44459888, 0.49707642, 3.98322824])

xc=np.array([0.27074305, 0.44609739, 0.49950347, 4.44674655])

xc=np.array([0.23682466, 0.43099417, 0.47350977, 4.2106033 ])

#Good for 0.99-0.01
xc=np.array([0.27369009, 0.45786094, 0.50632289, 3.61066082])


xl=np.array([-0.15,0.001,0.05,.0001]) 
xu=np.array([0.5, 0.8, .99,15.0]) 

 
#Function to minimize 
def q(pt,additional_tests=False): 

     
    #Define main parameters 
    p = co.setup() 
     
    #..and update them 
    p.q =np.array([0.0,pt[0]*pt[2],pt[1]*pt[2],pt[2]])   #Fixed cost of pticipation - mean 
     
     
    p.qvar =pt[3]*pt[2] #Fixed cost of pticipation -sd  
    
    def grid_fat_tails(gmin,gmax,gridpoints):
        """Create a grid with fat tail, centered and symmetric around gmin+gmax
        
        Args: 
            gmin (float): min of grid
            gmax(float): max of grids
            gridpoints(int): number of gridpoints (odd number)
        
        """ 
        odd_num = np.mod(gridpoints,2)
        mid=(gmax+gmin)/2.0
        summ=gmin+gmax
        first_part = mid-np.flip(nonlinspace(gmin,mid,(gridpoints+odd_num)//2,1.3))+gmin#nonlinspace(gmin,mid,(gridpoints+odd_num)//2,1.3)
        last_part = nonlinspace(mid,gmax,(gridpoints+odd_num)//2 ,1.3)[1:]#np.flip(summ - nonlinspace(gmin,mid,(gridpoints-odd_num)//2 + 1,1.3))[1:]
        return np.append(first_part,last_part)

    #Disutility from working 
    p.q_grid=np.zeros((p.nq,p.nwls,p.nw)) 
    p.q_gridt = np.linspace(-p.qvar,p.qvar,p.nzw)#grid_fat_tails(-p.qvar,p.qvar,p.nzw)# np.linspace(p.qmean-p.qmean*p.qvar,p.qmean+p.qmean*p.qvar,p.nq)#co.dist_gamma(p.qshape,p.qscale,p.nq) 
 
    for il in range(1,p.nwls): 
        for iw in range(p.nw): 
            for iq in range(p.nq): 
                 
                p.q_grid[iq,il,iw]=p.q[il]+ p.q_gridt[iq]
    
    # p.q_gridt = np.linspace(1.0-p.qvar,1.0+p.qvar,p.nq)#np.linspace(p.qmean-p.qmean*p.qvar,p.qmean+p.qmean*p.qvar,p.nq)#co.dist_gamma(p.qshape,p.qscale,p.nq)  
  
    # for il in range(1,p.nwls):  
    #     for iw in range(p.nw):  
    #         for iq in range(p.nq):  
                  
    #             p.q_grid[iq,il,iw]= p.q_gridt[iq]*p.q[il]   
     
    #Pension reform 
    ModP= sol.solveEulerEquation(p,model='pension reform') 
 
    #Baseline 
    ModB = sol.solveEulerEquation(p,model='baseline') 
   
 
    #Baseline 
    SB= sim.simNoUncer_interp(p,ModB,Years=year,Tstart=np.zeros(p.N,dtype=np.int16),Astart=p.startA,Pstart=np.ones((p.T,p.N))*p.startP,izstart=p.tw) 
 
    #Pension reform  
    ini_treated=np.ones(p.N,dtype=np.int32)*3 
    ini_treated[final_sample!=1998]=np.zeros(p.N,dtype=np.int32)[final_sample!=1998]+11 
    #p.T-np.cumsum(treatment==True,axis=0)[-1,:]#first child age at which women are treated  
    SP= sim.simNoUncer_interp(p,ModP,Years=year,Tstart=ini_treated,Astart=SB['A'],Pstart=SB['p'],izstart=SB['iz'])  
     
    #create unique dictionary for relevant simulated data  
    S=dict()  
    for i in ['h','wh','pb2','pb','c']:  
        S[i]=np.zeros((p.T,p.N))  
        S[i][age>=ini_treated] =SP[i][age>=ini_treated]  
        S[i][age<ini_treated]=SB[i][age<ini_treated]  
         
    ################################################### 
    #Average labor supply in 2000 for treatment group # 
    ################################################### 

    subset=((age>=3) & (age<=10) & (age>=ini_treated))
    sh_part=np.mean(S['h'][subset]==2) 
    sh_full=np.mean(S['h'][subset]>=3) 
    sh_min=np.mean(S['h'][subset]==1) 
    
    print(sh_part,sh_full,sh_min)
    
    mean_earnings=np.mean(S['wh'][subset])*p.scale 
    mean_points=np.mean(S['pb'][subset]) 
     
    #####################################  
    #Event study analysis  
    #####################################  
     
     
    #Covariates 
    hours = co.hours_value(p,S,0,p.T)  
    employed=np.array(S['h']>0,dtype=np.float64)  
    not_marginal=np.array(S['h']>1,dtype=np.float64)  
    marginal=np.array(S['h']==1,dtype=np.float64)  
    full=np.array(S['h']>=3,dtype=np.float64)  
    earnings=np.log(1+S['wh']*p.scale)  
    points=S['pb']  
    points_behavioral=S['pb2']  
    event_time=age.copy() 
    event_time[age>=8]=8 
    treat_group=(np.repeat((final_sample==1998)[:,None],p.T,axis=1).T)  
    idd=np.repeat(np.cumsum(np.ones(p.N))[:,None],p.T,axis=1).T 
    event_time_PER_treat=event_time*treat_group 
     
    #Sample 
    subset=(age<=8) 
 
    # Combine into a DataFrame 
    df = pd.DataFrame({ 
        "hours":hours[subset], 
        "employed":employed[subset], 
        "not_marginal":not_marginal[subset], 
        "marginal":marginal[subset], 
        "full":full[subset], 
        "earnings":earnings[subset], 
        "points":points[subset], 
        "points_behavioral":points_behavioral[subset], 
        "event_time":event_time[subset], 
        "idd":idd[subset], 
        "age":age[subset], 
        "event_time_PER_treat":event_time_PER_treat[subset] 
    }) 
     
   
 
    reference_value=2 
    event_cats = sorted(df['event_time'].unique()) 
    if reference_value in event_cats: 
        event_cats.remove(reference_value) 
        event_cats = [reference_value] + event_cats 
         
    event_cats = sorted(df['event_time_PER_treat'].unique()) 
    if reference_value in event_cats: 
        event_cats.remove(reference_value) 
        event_cats = [reference_value] + event_cats 
         
     
     
    # Example: your data frame 
    # df must contain columns: y, x1, x2, firm, year, region 
     
    # Step 1: Create the fixed effects structure 
    fe_df = df[['idd', 'event_time']].astype('category') 
     
    # Step 2: Create the HDFE projector 
    hdfe = create(fe_df) 
     
    # Create categorical with this ordering 
    df['event_cat'] = pd.Categorical(df['event_time_PER_treat'], categories=event_cats) 
 
    # Create dummies, drop_first will now drop your reference group 
    event_dummies = pd.get_dummies(df['event_cat'], prefix='event', drop_first=True) 
     
    # Residualize both y and X 
    y_resid = hdfe.residualize(df[['hours']].values) 
    X_resid = hdfe.residualize(event_dummies.values) 
     
    # OLS on residuals 
    model = sm.OLS(y_resid, X_resid) 
    results = model.fit() 
     
     
    eff_h=sm.OLS(hdfe.residualize(df[['hours']].values), X_resid).fit().params[2:].mean() 
    eff_e=sm.OLS(hdfe.residualize(df[['employed']].values), X_resid).fit().params[2:].mean() 
    eff_nme=sm.OLS(hdfe.residualize(df[['not_marginal']].values), X_resid).fit().params[2:].mean() 
    eff_full=sm.OLS(hdfe.residualize(df[['full']].values), X_resid).fit().params[2:].mean() 
    eff_marg=sm.OLS(hdfe.residualize(df[['marginal']].values), X_resid).fit().params[2:].mean() 
    eff_earn=sm.OLS(hdfe.residualize(df[['earnings']].values), X_resid).fit().params[2:].mean() 
    eff_points=sm.OLS(hdfe.residualize(df[['points']].values), X_resid).fit().params[2:].mean() 
    eff_points_behavioral=sm.OLS(hdfe.residualize(df[['points_behavioral']].values), X_resid).fit().params[2:].mean() 
 
     
     
     
    
     
    #True effects below 
    group=(age>=3)  & (age<=8) & (treat_group) 
    
    eff_ht=(co.hours_value(p,SP,0,p.T)[group]-co.hours_value(p,SB,0,p.T)[group]).mean()
    eff_nmet=(SP['h'][group]>1).mean()-(SB['h'][group]>1).mean()
    eff_et=(SP['h'][group]>0).mean()-(SB['h'][group]>0).mean()
    eff_earnt=(SP['wh'][group].mean()-SB['wh'][group].mean())*p.scale 
    eff_fullt=(SP['h'][group]>=3).mean()-(SB['h'][group]>=3).mean()
    eff_pointst=(SP['pb']-SB['pb'])[group].mean()
    eff_margt=np.mean(SP['h'][group]==1)-np.mean(SB['h'][group]==1) 
    
    #below for paper
    #(SP['h'][group][(SB['h'][group]==1)]>1).mean()
    
    if additional_tests:
        
       
    
    
        #Table with parameters + targeted moments  
        def p42(x): return str('%4.2f' % x)  
        def p43(x): return str('%4.3f' % x)     
        def p40(x): return str('%4.0f' % x)  
        
        table=r'\begin{table}[htbp]\centering'+\
                r'\caption{Model parameters and fit}\label{table:model_param}'+\
                r'\footnotesize'+\
                r'\begin{tabular}{lcccc}'+\
                r' \toprule '+\
                r" Parameter & Value & \multicolumn{3}{c}{Target statistics}  \\\cline{3-5} "+\
                r" &  &  Name & Data & Model  \\"+\
                r'\midrule   '+\
                r' Cost of working - mini ($q_{10}$)   &'+p43(p.q[2])+'& Share mini-jobs           & 0.26 &'+p42(sh_min)+'\\\\'+\
                r' Cost of working - part ($q_{20}$)   &'+p43(p.q[1])+'& Share part-time           & 0.19 &'+p42(sh_part)+'\\\\'+\
                r' Cost of working - full ($q_{38.5}$)      &'+p43(p.q[3])+'& Share full time      & 0.20 &'+p42(sh_full)+'\\\\'+\
                r' Fixed effects distribution ($q_{LIM}$)    &'+p43(p.qvar)+'& Effect of the reform on non-marginal employment  & 0.11 & '+p42(eff_nme)+'\\\\'+\
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
            #Model-based annuitization to compute MPE: https://michael-graber.github.io/pdf/Golosov-Graber-Mogstad-Novgorodsky-2023.pdf pg 38 of the appendix
            share = np.array([ ((1+r)/(1+δ))**(t+1)*(δ/(1+δ))*(1-(1/(1+δ))**(T-k))**-1  for t in range(per)]) 
             
            return share 
         
        adjust=np.ones(SP['c'].shape)/((1+p.r)**(np.cumsum(np.ones(p.T))-1.0))[:,None] 
        adjustr = adjust*(1+p.r)**3
         
        #MPE out of pension wealth, using tretroactive credits 
        SB_retro= sim.simNoUncer_interp(p,ModB,Years=year,Tstart=np.zeros(p.N,dtype=np.int16)+3,Astart=SB['A']+1,Pstart=SB['p'],izstart=SB['iz']) 
         
        change_earn  =(np.nanmean((SB_retro['w'][3:11,:]*p.wls[SB_retro['h'][3:11,:]]*adjustr[3:11,:]).sum(axis=0)))-\
                      (np.nanmean((SB['w'][3:11,:]*p.wls[SB['h'][3:11,:]]*adjustr[3:11,:]).sum(axis=0)))
                 
            
                
        #Below annuitization like in Golosov (2024), assuming that agents sommth consumption 
        change_pweal_s = (11-3)*((p.r/(1+p.r))*(1-(1/(1+p.r))**(p.T-3))**-1)#*np.mean((p.ρ*(SB['pb3']-SB['p'])*adjustr*(SB['ir']==1)).sum(axis=0)) 
         
         
        # #Below annuitization like in Golosov (2024), NOT assuming that agents sommth consumption if PIH where r could be different than δ 
        change_pweal_s2 = share(p.r,p.δ,p.T,8,12-8).sum()#*np.mean((p.ρ*(SB['pb3']-SB['p'])*adjustr*(SB['ir']==1)).sum(axis=0)) 
         
        # #Below model-consistent annuitization, where wealth is allocated according to consumption path. How to get it, 
        # #use the intertemporal budget constraint and take out of summation c0 (future consumtion is replaced by ct/c0). 
        # #Then manage the intertemporal BC to have c0=stuff: use it to get annuity value of future pension wealth. then 
        # #sum the implied consumtion for periods 8 to 12. This can be checked against change_pweal_s2 
        # ct_over_c0_discounted=np.mean(SB['c'][8:,:],axis=1)/np.mean(SB['c'][8,:])*adjustr[8:,0] 
        # c0=np.mean((p.ρ*(SB['pb3']-SB['p'])*adjustr*(SB['ir']==1)).sum(axis=0))/ct_over_c0_discounted.sum() 
         
        # change_pweal_d = c0*((np.mean(SB['c'][8:,:],axis=1)/np.mean(SB['c'][8,:]))[:4]).sum() 
        
        change_pweal_furbo=(SB_retro['c'][3:11]-SB['c'][3:11]).sum()/(SB_retro['c'][3:]-SB['c'][3:]).sum()
         
        
        #Finally, the marginal propensity to earn 
        MPE = change_earn/(change_pweal_furbo*1) 
    
        ############################################ 
        #Table with parameters 
        ###########################################      
        table=r'\begin{table}[htbp]\centering'+\
            r'\begin{threeparttable}'+\
                r'\caption{Non-targeted moments}\label{table:nontargeted_moments}'+\
                r'\footnotesize'+\
                r'\begin{tabular}{lcc}'+\
                r' \toprule '+\
                r" Effect of the reform on &   Data & Model  \\"+\
                r'\midrule   '+\
                r' Pension points   & 0.15 &'+p42(eff_points)+'\\\\'+\
                r' Behavioral pension points   & 0.10 &'+p42(eff_points_behavioral)+'\\\\'+\
                r' Work full time    & 0.05 &'+p42(eff_full)+'\\\\'+\
                r' Marginal employment    & -0.12 &'+p42(eff_marg)+'\\\\'+\
                r' Non-marginal employment earnings (\euro)    & 2809 &'+p40(eff_earn)+'\\\\'+\
                r'Employed    & 0.10 &'+p42(eff_e)+'\\\\'+\
                r'\toprule   '+\
                r" Other moments &   Data & Model  \\"+\
                r'\midrule   '+\
                r' Marginal propensity to earn (MPE)      & -0.51\text{ to }-0.12 &'+p42(MPE)+'\\\\'+\
                r'  \bottomrule'+\
              """\end{tabular}"""+\
                  r'\begin{tablenotes}[flushleft]\small\item \textsc{Notes:} The numbers related to the effect of the reform in the data are the DiD coefficients reported in Tables \ref{pension_table:main_outcomes} and \ref{pension_table:pension_contrib}. The model counterparts are obtained by running the same DiD models as in the empirical section, but using simulated data.''\\\\'+\
                  r'\end{tablenotes}'+\
                 r'\end{threeparttable}'+\
              r'\end{table}' 
               
        #Write table to tex file 
        with open('C:/Users/32489/Dropbox/occupation/model/pfabio/output/table_nontargetd.tex', 'w') as f: 
            f.write(table) 
            f.close() 
             
         
    print("The point is {}, the moments are shfull {}, sh_part {}, sh_min {}, eff_h {} , eff_e {}, eff_full  {}, eff_marg {}, eff_earn {}, eff_points {}, eff_points_behavioral {},  eff nonmarignal employment {} ".format(pt,sh_full,sh_part,sh_min,eff_h,eff_e,eff_full,eff_marg,eff_earn,eff_points,eff_points_behavioral,eff_nme))    
     
     

             
    # print(np.array([((sh_full-.175)/.175)**2,((sh_part-.142)/.142)**2,((sh_min-.30)/.30)**2,((eff_h-2.6))**2]).sum())   
    # return [((sh_full-.175)/.175),((sh_part-.142)/.142),((sh_min-.30)/.30),((eff_h-2.6))]             
 
    print("The point is {}".format(np.array([((sh_full-.175)/.175)**2,((sh_part-.142)/.142)**2,((sh_min-.3018)/.3018)**2,((eff_earn-1.1))**2]).sum()))
    return [((sh_full-.175)/.175),((sh_part-.142)/.142),((sh_min-.3018)/.3018),((eff_earn-1.1))]     

    
# [ 0.40706012  0.03525281 -0.51941101  0.00186123  1.60048109  0.03695673] first tentative σ=0.0005 
# 0.37349381, -0.01739811, -0.6       ,  0.00287586,  1.59080139, 0.03220926] current 
#Optimization below 
 
 
import numpy as np 
 
 
if __name__ == '__main__': 
     
 
    # computation_options = { "num_workers" :7,        # use four processes in parallel 
    #                         "working_dir" : "working" # where to save results in progress (in case interrupted) 
    #                         } 
     
    # global_search_options = { "num_points" : 10}  # number of points in global pre-test 
     
    # local_search_options = {  "algorithm"    : "dfols", # local search algorithm 
    #                                                       # can be either BOBYQA from NLOPT or NelderMead from scipy 
    #                           "num_restarts" : 7,      # how many local searches to do 
    #                           "shrink_after" : 7,       # after the first [shrink_after] restarts we begin searching 
    #                                                       # near the best point we have found so far 
    #                           "xtol_rel"     : 1e-6,     # relative tolerance on x 
    #                           "ftol_rel"     : 1e-6     # relative tolerance on f 
    #                         } 
     
    # opt = TikTak.TTOptimizer(computation_options, global_search_options, local_search_options, skip_global=False
    #                           ) 
    # x,fx = opt.minimize(q,xl,xu) 
    # print(f'The minimizer is {x}') 
    # print(f'The objective value at the min is {fx}') 
     
     
    res=dfols.solve(q, xc, rhobeg = 0.3, rhoend=1e-6, maxfun=250, bounds=(xl,xu), 
                    npt=len(xc)+5,scaling_within_bounds=True,  
                    user_params={'tr_radius.gamma_dec':0.98,'tr_radius.gamma_inc':1.0, 
                                  'tr_radius.alpha1':0.9,'tr_radius.alpha2':0.95}, 
                    objfun_has_noise=False) 
